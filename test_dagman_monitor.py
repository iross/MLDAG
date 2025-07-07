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
import csv
import io

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


class TestCSVExport(unittest.TestCase):
    """Test CSV export functionality including hidden compute detection."""

    def setUp(self):
        # Create a temporary DAG file for testing
        self.temp_dag = tempfile.NamedTemporaryFile(mode='w', suffix='.dag', delete=False)
        self.temp_dag.write("""
JOB run0-train_epoch0 pretraining.submit
VARS run0-train_epoch0 epoch="0" run_uuid="test-uuid-123"
JOB run0-train_epoch1 pretraining.submit  
VARS run0-train_epoch1 epoch="1" run_uuid="test-uuid-123"
""")
        self.temp_dag.close()
        self.dag_manager = DAGStatusMonitor(self.temp_dag.name)
        
        # Create test jobs with different scenarios
        self.setup_test_jobs()

    def tearDown(self):
        import os
        os.unlink(self.temp_dag.name)

    def setup_test_jobs(self):
        """Set up test jobs with various states for CSV export testing."""
        # Long completed job (normal case)
        long_job = JobInfo(
            name="run0-train_epoch0",
            status=JobStatus.COMPLETED,
            cluster_id=12345,
            run_uuid="test-uuid-123",
            epoch=0,
            submit_time=datetime(2025, 1, 1, 10, 0, 0),
            start_time=datetime(2025, 1, 1, 10, 5, 0),
            end_time=datetime(2025, 1, 1, 14, 5, 0),  # 4 hours
            total_bytes_sent=1000000,
            total_bytes_received=2000000,
            resource_name="ospool"
        )
        
        # Short completed job (potential hidden compute case)
        short_job = JobInfo(
            name="run0-train_epoch1",
            status=JobStatus.COMPLETED,
            cluster_id=12346,
            run_uuid="test-uuid-123", 
            epoch=1,
            submit_time=datetime(2025, 1, 1, 15, 0, 0),
            start_time=datetime(2025, 1, 1, 15, 5, 0),
            end_time=datetime(2025, 1, 1, 15, 25, 0),  # 20 minutes
            total_bytes_sent=500000,
            total_bytes_received=1000000,
            resource_name="expanse"
        )
        
        # Running job
        running_job = JobInfo(
            name="run0-train_epoch2",
            status=JobStatus.RUNNING,
            cluster_id=12347,
            run_uuid="test-uuid-123",
            epoch=2,
            submit_time=datetime(2025, 1, 1, 16, 0, 0),
            start_time=datetime(2025, 1, 1, 16, 5, 0),
            total_bytes_sent=100000,
            total_bytes_received=200000,
            resource_name="bridges2"
        )
        
        # Idle job
        idle_job = JobInfo(
            name="run0-train_epoch3", 
            status=JobStatus.IDLE,
            cluster_id=12348,
            run_uuid="test-uuid-123",
            epoch=3,
            submit_time=datetime(2025, 1, 1, 17, 0, 0),
            resource_name="delta"
        )

        self.dag_manager.jobs = {
            "run0-train_epoch0": long_job,
            "run0-train_epoch1": short_job,
            "run0-train_epoch2": running_job,
            "run0-train_epoch3": idle_job
        }

    def test_csv_export_basic_functionality(self):
        """Test basic CSV export creates correct format and content."""
        output_buffer = io.StringIO()
        
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value = output_buffer
            
            # Mock the path operations
            with patch('pathlib.Path.mkdir'), \
                 patch.object(self.dag_manager, 'get_queued_cluster_ids', return_value=set()), \
                 patch.object(self.dag_manager, 'should_show_job', return_value=True), \
                 patch.object(self.dag_manager, '_matches_run_filter', return_value=True):
                
                # Capture the CSV content by intercepting the write calls
                written_content = []
                original_write = output_buffer.write
                
                def capture_write(content):
                    written_content.append(content)
                    return original_write(content)
                
                output_buffer.write = capture_write
                
                self.dag_manager.export_to_csv("test_output.csv")
                
                # Reconstruct the full CSV content
                full_content = ''.join(written_content)
                
                # Parse the CSV content
                csv_reader = csv.DictReader(io.StringIO(full_content))
                rows = list(csv_reader)
                
                # Verify we have the expected number of rows
                self.assertEqual(len(rows), 4)
                
                # Check header fields are present
                expected_fields = [
                    'Job Name', 'Run Number', 'Epoch', 'Run UUID', 'HTCondor Cluster ID',
                    'Targeted Resource', 'Status', 'Submit Time', 'Start Time', 'End Time',
                    'Duration (seconds)', 'Duration (human)', 'Total Bytes Sent', 
                    'Total Bytes Received'
                ]
                
                for field in expected_fields:
                    self.assertIn(field, csv_reader.fieldnames)

    def test_csv_export_job_data_accuracy(self):
        """Test that CSV export contains accurate job data."""
        output_buffer = io.StringIO()
        
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value = output_buffer
            
            with patch('pathlib.Path.mkdir'), \
                 patch.object(self.dag_manager, 'get_queued_cluster_ids', return_value=set()), \
                 patch.object(self.dag_manager, 'should_show_job', return_value=True), \
                 patch.object(self.dag_manager, '_matches_run_filter', return_value=True), \
                 patch.object(self.dag_manager, '_detect_hidden_compute', return_value=(False, 0, "")):
                
                written_content = []
                original_write = output_buffer.write
                
                def capture_write(content):
                    written_content.append(content)
                    return original_write(content)
                
                output_buffer.write = capture_write
                
                self.dag_manager.export_to_csv("test_output.csv")
                
                full_content = ''.join(written_content)
                csv_reader = csv.DictReader(io.StringIO(full_content))
                rows = list(csv_reader)
                
                # Find the completed long job
                long_job_row = next(row for row in rows if row['Job Name'] == 'run0-train_epoch0')
                
                self.assertEqual(long_job_row['Run Number'], '0')
                self.assertEqual(long_job_row['Epoch'], '0')
                self.assertEqual(long_job_row['HTCondor Cluster ID'], '12345')
                self.assertEqual(long_job_row['Status'], 'COMPLETED')
                self.assertEqual(long_job_row['Duration (seconds)'], '14400')  # 4 hours
                self.assertEqual(long_job_row['Targeted Resource'], 'ospool')
                self.assertEqual(long_job_row['Total Bytes Sent'], '1000000')
                self.assertEqual(long_job_row['Total Bytes Received'], '2000000')

    def test_csv_export_actual_compute_job_substitution(self):
        """Test that short epochs get replaced with actual compute job data."""
        output_buffer = io.StringIO()
        
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value = output_buffer
            
            with patch('pathlib.Path.mkdir'), \
                 patch.object(self.dag_manager, 'get_queued_cluster_ids', return_value=set()), \
                 patch.object(self.dag_manager, 'should_show_job', return_value=True), \
                 patch.object(self.dag_manager, '_matches_run_filter', return_value=True), \
                 patch.object(self.dag_manager, '_detect_hidden_compute', return_value=(True, 14400, "12999")), \
                 patch.object(self.dag_manager, '_get_cluster_timing_details', return_value={
                     'start_time': '2025-01-01T10:00:00',
                     'end_time': '2025-01-01T14:00:00'
                 }):
                
                written_content = []
                original_write = output_buffer.write
                
                def capture_write(content):
                    written_content.append(content)
                    return original_write(content)
                
                output_buffer.write = capture_write
                
                self.dag_manager.export_to_csv("test_output.csv")
                
                full_content = ''.join(written_content)
                csv_reader = csv.DictReader(io.StringIO(full_content))
                rows = list(csv_reader)
                
                # Find the short job that should be replaced
                short_job_row = next(row for row in rows if row['Job Name'] == 'run0-train_epoch1')
                
                # Should show actual compute job data instead of short job
                self.assertEqual(short_job_row['HTCondor Cluster ID'], '12999')  # Actual compute cluster
                self.assertEqual(short_job_row['Duration (seconds)'], '14400')  # 4 hours, not 20 minutes
                self.assertEqual(short_job_row['Duration (human)'], '4:00:00')
                self.assertEqual(short_job_row['Start Time'], '2025-01-01T10:00:00')
                self.assertEqual(short_job_row['End Time'], '2025-01-01T14:00:00')

    def test_csv_export_duration_calculations(self):
        """Test duration calculations for different job states."""
        output_buffer = io.StringIO()
        
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value = output_buffer
            
            with patch('pathlib.Path.mkdir'), \
                 patch.object(self.dag_manager, 'get_queued_cluster_ids', return_value=set()), \
                 patch.object(self.dag_manager, 'should_show_job', return_value=True), \
                 patch.object(self.dag_manager, '_matches_run_filter', return_value=True), \
                 patch.object(self.dag_manager, '_detect_hidden_compute', return_value=(False, 0, "")), \
                 patch('dagman_monitor.datetime') as mock_datetime:
                
                # Mock current time for running job duration calculation
                mock_datetime.now.return_value = datetime(2025, 1, 1, 18, 5, 0)
                
                written_content = []
                original_write = output_buffer.write
                
                def capture_write(content):
                    written_content.append(content)
                    return original_write(content)
                
                output_buffer.write = capture_write
                
                self.dag_manager.export_to_csv("test_output.csv")
                
                full_content = ''.join(written_content)
                csv_reader = csv.DictReader(io.StringIO(full_content))
                rows = list(csv_reader)
                
                # Check completed job duration
                completed_row = next(row for row in rows if row['Job Name'] == 'run0-train_epoch0')
                self.assertEqual(completed_row['Duration (seconds)'], '14400')  # 4 hours
                
                # Check short completed job duration  
                short_row = next(row for row in rows if row['Job Name'] == 'run0-train_epoch1')
                self.assertEqual(short_row['Duration (seconds)'], '1200')  # 20 minutes
                
                # Check running job duration (mocked current time - start time)
                running_row = next(row for row in rows if row['Job Name'] == 'run0-train_epoch2')
                self.assertEqual(running_row['Duration (seconds)'], '7200')  # 2 hours
                
                # Check idle job has no duration
                idle_row = next(row for row in rows if row['Job Name'] == 'run0-train_epoch3')
                self.assertEqual(idle_row['Duration (seconds)'], '')

    def test_csv_export_resource_mapping(self):
        """Test resource name mapping in CSV export."""
        output_buffer = io.StringIO()
        
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value = output_buffer
            
            with patch('pathlib.Path.mkdir'), \
                 patch.object(self.dag_manager, 'get_queued_cluster_ids', return_value=set()), \
                 patch.object(self.dag_manager, 'should_show_job', return_value=True), \
                 patch.object(self.dag_manager, '_matches_run_filter', return_value=True), \
                 patch.object(self.dag_manager, '_detect_hidden_compute', return_value=(False, 0, "")):
                
                written_content = []
                original_write = output_buffer.write
                
                def capture_write(content):
                    written_content.append(content)
                    return original_write(content)
                
                output_buffer.write = capture_write
                
                self.dag_manager.export_to_csv("test_output.csv")
                
                full_content = ''.join(written_content)
                csv_reader = csv.DictReader(io.StringIO(full_content))
                rows = list(csv_reader)
                
                # Check major resources are preserved
                resource_mapping = {row['Job Name']: row['Targeted Resource'] for row in rows}
                
                self.assertEqual(resource_mapping['run0-train_epoch0'], 'ospool')
                self.assertEqual(resource_mapping['run0-train_epoch1'], 'expanse')
                self.assertEqual(resource_mapping['run0-train_epoch2'], 'bridges2')
                self.assertEqual(resource_mapping['run0-train_epoch3'], 'delta')

    @patch('pathlib.Path.exists')
    def test_hidden_compute_detection_no_dag_log(self, mock_exists):
        """Test hidden compute detection when DAG log doesn't exist."""
        mock_exists.return_value = False
        
        result = self.dag_manager._detect_hidden_compute("run0-train_epoch1", 1200)
        
        self.assertEqual(result, (False, 0, ""))

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_hidden_compute_detection_no_status_85(self, mock_exists, mock_file):
        """Test hidden compute detection when no status 85 failures found."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "No status 85 failures here"
        
        result = self.dag_manager._detect_hidden_compute("run0-train_epoch1", 1200)
        
        self.assertEqual(result, (False, 0, ""))

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_hidden_compute_detection_success(self, mock_exists, mock_file):
        """Test successful hidden compute detection."""
        mock_exists.return_value = True
        
        # Mock DAGMan log content with status 85 failure
        dagman_content = """
Node run0-train_epoch1 job proc (12999.0.0) failed with status 85
"""
        
        # Mock metl.log content with timing data
        metl_content = """
001 (12999.0.0) 2025-01-01 10:00:00 Job executing on host: test.host
005 (12999.0.0) 2025-01-01 14:00:00 Job terminated.
"""
        
        def mock_open_side_effect(filename, mode='r'):
            if 'dagman.out' in str(filename):
                return mock_open(read_data=dagman_content).return_value
            elif 'metl.log' in str(filename):
                return mock_open(read_data=metl_content).return_value
            return mock_open().return_value
        
        with patch('builtins.open', side_effect=mock_open_side_effect):
            # Test with short reported duration (20 minutes) vs actual work (4 hours)
            result = self.dag_manager._detect_hidden_compute("run0-train_epoch1", 1200)
            
            # Should detect hidden work: 4 hours (14400s) > 2x reported (1200s)
            self.assertEqual(result[0], True)  # has_hidden_work
            self.assertEqual(result[1], 14400)  # hidden_duration (4 hours)
            self.assertEqual(result[2], "12999")  # failed_cluster_id

    def test_hidden_compute_detection_long_duration_ignored(self):
        """Test that long duration jobs are not checked for hidden compute."""
        # Test with 45 minutes (2700s) - should be ignored as it's >= 30 min threshold
        result = self.dag_manager._detect_hidden_compute("run0-train_epoch1", 2700)
        
        self.assertEqual(result, (False, 0, ""))

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_get_cluster_timing_from_metl_success(self, mock_exists, mock_file):
        """Test successful cluster timing extraction from metl.log."""
        mock_exists.return_value = True
        
        metl_content = """
001 (12345.0.0) 2025-01-01 10:30:00 Job executing on host: test.host
005 (12345.0.0) 2025-01-01 12:45:00 Job terminated.
"""
        mock_file.return_value.read.return_value = metl_content
        
        # Mock the file reading properly
        mock_file.return_value.__iter__ = lambda self: iter(metl_content.split('\n'))
        
        result = self.dag_manager._get_cluster_timing_from_metl("12345")
        
        # 12:45:00 - 10:30:00 = 2:15:00 = 8100 seconds
        self.assertEqual(result, 8100)

    @patch('pathlib.Path.exists')
    def test_get_cluster_timing_from_metl_no_file(self, mock_exists):
        """Test cluster timing extraction when metl.log doesn't exist."""
        mock_exists.return_value = False
        
        result = self.dag_manager._get_cluster_timing_from_metl("12345")
        
        self.assertIsNone(result)

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')  
    def test_get_cluster_timing_from_metl_no_timing_data(self, mock_exists, mock_file):
        """Test cluster timing extraction when no timing data found."""
        mock_exists.return_value = True
        
        metl_content = """
Some other log entries that don't match cluster 12345
"""
        mock_file.return_value.__iter__ = lambda self: iter(metl_content.split('\n'))
        
        result = self.dag_manager._get_cluster_timing_from_metl("12345")
        
        self.assertIsNone(result)

    def test_csv_export_error_handling(self):
        """Test CSV export error handling for file I/O issues."""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            # Should not raise exception, but print error message
            with patch.object(self.dag_manager.console, 'print') as mock_print:
                self.dag_manager.export_to_csv("test_output.csv")
                mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()
