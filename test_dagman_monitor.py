#!/usr/bin/env python3
"""
Fixed unit tests for dagman_monitor.py

Test suite covering core functionality that works with the current API.
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Import the module under test
from dagman_monitor import (
    JobStatus, JobInfo, DAGStatusMonitor, DAGManLogParser,
    HTCONDOR_EVENT_CODES, TIMESTAMP_FORMATS, DAG_FILE_PATTERNS
)


class TestJobInfo(unittest.TestCase):
    """Test the JobInfo dataclass."""

    def test_job_info_initialization(self):
        """Test JobInfo object creation with various parameters."""
        job = JobInfo(
            name="run0-train_epoch0",
            status=JobStatus.RUNNING,
            cluster_id=12345,
            run_uuid="test-uuid-123",
            epoch=1
        )

        self.assertEqual(job.name, "run0-train_epoch0")
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertEqual(job.cluster_id, 12345)
        self.assertEqual(job.run_uuid, "test-uuid-123")
        self.assertEqual(job.epoch, 1)
        self.assertIsNone(job.submit_time)
        self.assertEqual(job.retries, 0)

    def test_job_info_defaults(self):
        """Test JobInfo with minimal parameters uses correct defaults."""
        job = JobInfo(name="test-job")

        self.assertEqual(job.name, "test-job")
        self.assertEqual(job.status, JobStatus.UNKNOWN)
        self.assertIsNone(job.cluster_id)
        self.assertIsNone(job.run_uuid)
        self.assertIsNone(job.epoch)
        self.assertEqual(job.retries, 0)


class TestDAGManLogParser(unittest.TestCase):
    """Test the DAGManLogParser class."""

    def setUp(self):
        self.parser = DAGManLogParser()

    def test_parse_timestamp_modern_format(self):
        """Test parsing modern timestamp format (YYYY-MM-DD HH:MM:SS)."""
        timestamp_str = "2025-04-25 13:34:44"
        result = self.parser.parse_timestamp(timestamp_str)
        expected = datetime(2025, 4, 25, 13, 34, 44)
        self.assertEqual(result, expected)

    def test_parse_timestamp_legacy_format(self):
        """Test parsing legacy timestamp format (MM/dd/yy HH:MM:SS)."""
        timestamp_str = "04/25/25 13:34:44"
        result = self.parser.parse_timestamp(timestamp_str)
        expected = datetime(2025, 4, 25, 13, 34, 44)
        self.assertEqual(result, expected)

    def test_parse_timestamp_empty_string(self):
        """Test parsing empty timestamp raises ValueError."""
        with self.assertRaises(ValueError):
            self.parser.parse_timestamp("")

    def test_parse_timestamp_invalid_format(self):
        """Test parsing invalid timestamp falls back to current time."""
        timestamp_str = "invalid-timestamp"
        result = self.parser.parse_timestamp(timestamp_str)
        # Should return current time, so just check it's a datetime
        self.assertIsInstance(result, datetime)

    def test_parse_log_line_job_submitted(self):
        """Test parsing job submission event."""
        log_line = "000 (12634824.000.000) 2025-04-25 13:34:44 Job submitted from host: <host-info>"
        result = self.parser.parse_log_line(log_line)

        self.assertIsNotNone(result)
        self.assertEqual(result['event_type'], "job_submitted")
        self.assertEqual(result['cluster_id'], 12634824)
        self.assertEqual(result['timestamp'], datetime(2025, 4, 25, 13, 34, 44))

    def test_parse_log_line_job_executing(self):
        """Test parsing job execution event."""
        log_line = "001 (12634831.000.000) 2025-04-25 14:23:26 Job executing on host: <host-info>"
        result = self.parser.parse_log_line(log_line)

        self.assertIsNotNone(result)
        self.assertEqual(result['event_type'], "job_executing")
        self.assertEqual(result['cluster_id'], 12634831)
        self.assertEqual(result['timestamp'], datetime(2025, 4, 25, 14, 23, 26))

    def test_parse_log_line_transfer_started(self):
        """Test parsing transfer start event."""
        log_line = "040 (12634831.000.000) 2025-04-25 14:23:00 Started transferring input files"
        result = self.parser.parse_log_line(log_line)

        self.assertIsNotNone(result)
        self.assertEqual(result['event_type'], "transfer_input_started")
        self.assertEqual(result['cluster_id'], 12634831)
        self.assertEqual(result['timestamp'], datetime(2025, 4, 25, 14, 23, 0))

    def test_parse_log_line_invalid(self):
        """Test parsing invalid log line returns None."""
        invalid_lines = [
            "This is not a valid log line",
            "123 invalid format",
            "",  # Empty line
        ]

        for line in invalid_lines:
            with self.subTest(line=line):
                result = self.parser.parse_log_line(line)
                self.assertIsNone(result)


class TestDAGStatusMonitorBasic(unittest.TestCase):
    """Test basic DAGStatusMonitor functionality."""

    def setUp(self):
        # Create a temporary DAG file for testing
        self.temp_dag = tempfile.NamedTemporaryFile(mode='w', suffix='.dag', delete=False)
        self.temp_dag.write("""
JOB run0-train_epoch0 pretraining.submit
VARS run0-train_epoch0 epoch="0" run_uuid="test-uuid-123"
""")
        self.temp_dag.close()
        self.dag_manager = DAGStatusMonitor(self.temp_dag.name)

    def tearDown(self):
        import os
        os.unlink(self.temp_dag.name)

    def test_extract_epoch_from_job_name_valid(self):
        """Test extracting epoch number from valid job names."""
        test_cases = [
            ("run0-train_epoch5", 5),
            ("run10-train_epoch15", 15),
            ("run123-train_epoch0", 0),
        ]

        for job_name, expected_epoch in test_cases:
            with self.subTest(job_name=job_name):
                result = self.dag_manager.extract_epoch_from_job_name(job_name)
                self.assertEqual(result, expected_epoch)

    def test_extract_epoch_from_job_name_invalid(self):
        """Test extracting epoch from invalid job names returns None."""
        invalid_names = [
            "invalid-job-name",
            "run0-something-else",
            "not-a-job",
            "",
        ]

        for job_name in invalid_names:
            with self.subTest(job_name=job_name):
                result = self.dag_manager.extract_epoch_from_job_name(job_name)
                self.assertIsNone(result)

    def test_natural_sort_key(self):
        """Test natural sorting key generation for job names."""
        job_name = "run5-train_epoch10"
        result = self.dag_manager.natural_sort_key(job_name)
        self.assertEqual(result, (5, 10))


if __name__ == "__main__":
    unittest.main()
