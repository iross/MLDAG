#!/usr/bin/env python3
"""
DAGMan log monitoring for MLDAG training runs.
Monitors .dag.dagman.log files and tracks job status changes.

This module provides real-time monitoring of HTCondor DAGMan workflows,
parsing log files to track training job progress across distributed compute resources.
"""

import csv
import json
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, cast
from dataclasses import dataclass, field
import subprocess

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.columns import Columns


# Constants
HTCONDOR_EVENT_CODES = {
    'JOB_SUBMITTED': '000',
    'JOB_EXECUTING': '001', 
    'JOB_TERMINATED': '005',
    'JOB_HELD': '012',
    'JOB_RELEASED': '013',
    'TRANSFER_INPUT': '040'
}

TIMESTAMP_FORMATS = {
    'MODERN': '%Y-%m-%d %H:%M:%S',
    'LEGACY': '%m/%d/%y %H:%M:%S'
}

DAG_FILE_PATTERNS = {
    'JOB_LINE': r'JOB\s+(run\d+-train_epoch\d+)\s+',
    'VARS_LINE': r'VARS\s+(run\d+-train_epoch\d+)\s+.*?epoch="(\d+)".*?run_uuid="([^"]+)"',
    'JOB_NAME': r'run(\d+)-train_epoch(\d+)',
    'VARS_EXTRACTION': r'VARS\s+{job_name}\s+(.+)'
}

DISPLAY_CONFIG = {
    'MAX_UUID_WIDTH': 8,
    'MAX_RESOURCE_WIDTH': 15,
    'MAX_CLUSTER_WIDTH': 12,
    'PROGRESS_BAR_WIDTH': 20,
    'RESCUE_RETRY_PREFIX': 'RETRY ',
    'RESCUE_DONE_PREFIX': 'DONE ',
    'DEFAULT_REFRESH_INTERVAL': 60.0,
    'DEFAULT_TIMEOUT': 10
}


class JobStatus(Enum):
    """Status enumeration for DAG jobs."""
    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    TRANSFERRING = "TRANSFERRING"
    RUNNING = "RUNNING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HELD = "HELD"
    REMOVED = "REMOVED"


@dataclass
class JobInfo:
    """Information about a DAG job.
    
    Attributes:
        name: Job name from DAG file
        status: Current job status
        submit_time: When job was submitted to queue
        start_time: When job started executing
        end_time: When job finished (completed/failed)
        cluster_id: HTCondor cluster ID
        run_uuid: Training run identifier
        epoch: Training epoch number
        resource_name: Target compute resource
        retries: Number of retry attempts
        last_event_time: Most recent log event timestamp
    """
    name: str
    status: JobStatus = JobStatus.UNKNOWN
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cluster_id: Optional[int] = None
    run_uuid: Optional[str] = None
    epoch: Optional[int] = None
    resource_name: Optional[str] = None
    retries: int = 0
    last_event_time: Optional[datetime] = None
    total_bytes_sent: int = 0
    total_bytes_received: int = 0


@dataclass 
class TrainingRunStatus:
    """Status of a complete training run (collection of jobs).
    
    Attributes:
        run_uuid: Unique identifier for this training run
        jobs: Dictionary mapping job names to JobInfo objects
        total_epochs: Total number of epochs in this run
        completed_epochs: Number of completed epochs
        failed_epochs: Number of failed epochs
    """
    run_uuid: str
    jobs: Dict[str, JobInfo] = field(default_factory=dict)
    total_epochs: int = 0
    completed_epochs: int = 0
    failed_epochs: int = 0
    
    @property
    def progress_percent(self) -> float:
        """Calculate completion percentage for this training run."""
        if self.total_epochs == 0:
            return 0.0
        return (self.completed_epochs / self.total_epochs) * 100


class DAGManLogParser:
    """Parser for DAGMan log files.
    
    Handles parsing of HTCondor ClassAd log events and DAG node associations.
    Updated to handle both old format (MM/dd/yy) and new format (YYYY-MM-DD) timestamps.
    """
    
    def __init__(self) -> None:
        """Initialize the log parser with regex patterns for different log formats."""
        # Flexible timestamp pattern that matches both formats
        modern_ts = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        legacy_ts = r'\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}'
        timestamp_pattern = f'({modern_ts}|{legacy_ts})'
        
        # Event patterns using flexible timestamp
        self.patterns = {
            'job_submitted': re.compile(
                rf'^000 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Job submitted from host:'
            ),
            'job_executing': re.compile(
                rf'^001 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Job executing on host:'
            ),
            'job_terminated': re.compile(
                rf'^005 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Job terminated\.'
            ),
            'job_held': re.compile(
                rf'^012 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Job was held\.'
            ),
            'job_released': re.compile(
                rf'^013 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Job was released\.'
            ),
            'transfer_input_started': re.compile(
                rf'^040 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Started transferring input files'
            ),
            'transfer_input_finished': re.compile(
                rf'^040 \((\d+)\.\d+\.\d+\) {timestamp_pattern} Finished transferring input files'
            ),
            'dag_node': re.compile(r'^\s*DAG Node: (.+)$'),
        }
    
    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from log file.
        
        Supports both modern (YYYY-MM-DD HH:MM:SS) and legacy (MM/dd/yy HH:MM:SS) formats.
        
        Args:
            timestamp_str: Timestamp string from log file
            
        Returns:
            Parsed datetime object, or current time if parsing fails
            
        Raises:
            ValueError: If timestamp_str is empty or None
        """
        if not timestamp_str or not timestamp_str.strip():
            raise ValueError("Timestamp string cannot be empty")
            
        timestamp_str = timestamp_str.strip()
        
        # Try modern format first
        try:
            return datetime.strptime(timestamp_str, TIMESTAMP_FORMATS['MODERN'])
        except ValueError:
            pass
            
        # Fall back to legacy format
        try:
            return datetime.strptime(timestamp_str, TIMESTAMP_FORMATS['LEGACY'])
        except ValueError:
            # Log the parsing failure for debugging
            self.console.print(f"[dim yellow]Warning: Could not parse timestamp: {timestamp_str}[/dim yellow]")
            return datetime.now()
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line and extract job event info.
        
        Args:
            line: Raw log line from DAGMan log file
            
        Returns:
            Dictionary with event information or None if no event found
        """
        line = line.strip()
        if not line or line == '...':
            return None
        
        # Check for DAG node information first
        dag_match = self.patterns['dag_node'].search(line)
        if dag_match:
            return {
                'event_type': 'dag_node',
                'dag_node_name': dag_match.group(1),
                'raw_line': line
            }
            
        # Try event patterns
        for event_type, pattern in self.patterns.items():
            if event_type == 'dag_node':
                continue
                
            match = pattern.search(line)
            if match:
                cluster_id = int(match.group(1))
                timestamp_str = match.group(2)
                timestamp = self.parse_timestamp(timestamp_str)
                
                return {
                    'event_type': event_type,
                    'timestamp': timestamp,
                    'cluster_id': cluster_id,
                    'raw_line': line
                }
        
        return None


class DAGStatusMonitor:
    """Monitor DAG status by parsing DAGMan logs and node status files.
    
    Provides both one-time status reports and live monitoring capabilities
    for HTCondor DAG workflows.
    """
    
    def __init__(self, dag_file: str = "pipeline.dag") -> None:
        """Initialize the DAG status monitor.
        
        Args:
            dag_file: Path to the DAG file to monitor
            
        Raises:
            ValueError: If dag_file is empty or invalid
            FileNotFoundError: If dag_file does not exist
        """
        if not dag_file or not dag_file.strip():
            raise ValueError("DAG file path cannot be empty")
            
        self.dag_file = Path(dag_file)
        if not self.dag_file.exists():
            raise FileNotFoundError(f"DAG file not found: {self.dag_file}")
            
        self.dagman_log = self.dag_file.with_suffix('.dag.dagman.log')
        self.nodes_log = self.dag_file.with_suffix('.dag.nodes.log')
        self.node_status_file = self.dag_file.with_suffix('.dag.status')
        
        # Look for metl.log in the same directory
        self.metl_log = self.dag_file.parent / "metl.log"
        
        self.parser = DAGManLogParser()
        self.jobs: Dict[str, JobInfo] = {}
        self.training_runs: Dict[str, TrainingRunStatus] = {}
        self.console = Console()
        
        # Track last processed position in log file for incremental parsing
        self.last_log_position = 0
        self.last_metl_log_position = 0
        
        # Cluster ID to DAG node name mapping
        self.cluster_to_dagnode: Dict[int, str] = {}
        
        # Track rescue file parsing to avoid re-parsing
        self.rescue_files_processed: Set[Path] = set()
        
        # Optional run filtering
        self.run_filter: Optional[Set[str]] = None
        
        # Cache job timing information from metl.log
        self.metl_job_timing: Dict[int, Dict[str, Union[datetime, int, str]]] = {}
        
        # Cache of planned training runs from DAG file
        self.planned_training_runs: Dict[str, Dict[str, Union[int, List[str]]]] = {}
        
        # Performance optimizations: cache frequently accessed data
        self._dag_file_content_cache: Optional[str] = None
        self._dag_file_mtime: Optional[float] = None
        self._htcondor_status_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self._htcondor_cache_time: float = 0
        self._htcondor_cache_ttl: float = 5.0  # Cache HTCondor status for 5 seconds
        
        self._parse_planned_training_runs()
        
    def set_run_filter(self, filter_runs: Optional[str]) -> None:
        """Set filter for specific training runs.
        
        Args:
            filter_runs: Comma-separated list of run numbers or UUIDs to filter by
            
        Raises:
            ValueError: If filter_runs contains invalid criteria
        """
        if not filter_runs or not filter_runs.strip():
            self.run_filter = None
            return
            
        # Parse the filter criteria
        criteria = [item.strip() for item in filter_runs.split(',') if item.strip()]
        if not criteria:
            self.run_filter = None
            return
            
        self.run_filter = set()
        
        for criterion in criteria:
            if not criterion:
                continue
                
            # Check if it's a run number (just digits)
            if criterion.isdigit():
                run_num = int(criterion)
                if run_num < 0:
                    raise ValueError(f"Run number must be non-negative: {criterion}")
                # Add run number as "run{number}"
                self.run_filter.add(f"run{criterion}")
            else:
                # Assume it's a UUID or partial UUID
                self.run_filter.add(criterion)
    
    def _matches_run_filter(self, job: JobInfo) -> bool:
        """Check if a job matches the current run filter.
        
        Args:
            job: JobInfo object to check
            
        Returns:
            True if job matches filter (or no filter set), False otherwise
        """
        if not self.run_filter:
            return True
            
        # Check run UUID (exact or partial match)
        if job.run_uuid is not None and self.run_filter:
            job_run_uuid = cast(str, job.run_uuid)  # Type narrowing after None check
            for filter_item in self.run_filter:
                if filter_item in job_run_uuid:
                    return True
        
        # Check run number from job name
        if job.name:
            match = re.match(DAG_FILE_PATTERNS['JOB_NAME'], job.name)
            if match:
                run_name = f"run{match.group(1)}"
                if run_name in self.run_filter:
                    return True
        
        return False
        
    def _get_dag_file_content(self) -> str:
        """Get DAG file content with caching to avoid repeated file reads.
        
        Returns:
            Content of DAG file as string, empty if file not found
        """
        try:
            current_mtime = self.dag_file.stat().st_mtime
            
            # Check if we need to reload the file
            if (self._dag_file_content_cache is None or 
                self._dag_file_mtime is None or 
                current_mtime > self._dag_file_mtime):
                
                with open(self.dag_file, 'r') as f:
                    self._dag_file_content_cache = f.read()
                self._dag_file_mtime = current_mtime
                
            return self._dag_file_content_cache
            
        except (FileNotFoundError, OSError):
            return ""
    
    def parse_job_vars(self, job_name: str) -> Dict[str, str]:
        """Extract job variables from DAG file.
        
        Parses VARS lines in the DAG file to extract metadata like run_uuid,
        epoch, and resource assignments.
        
        Args:
            job_name: Name of the job to extract variables for
            
        Returns:
            Dictionary of variable name/value pairs
        """
        vars_dict: Dict[str, str] = {}
        content = self._get_dag_file_content()
        if not content:
            return vars_dict
                
        # Look for VARS lines for this job
        pattern = rf'VARS\s+{re.escape(job_name)}\s+(.+)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for match in matches:
            # Parse key="value" pairs (with quotes)
            var_pattern = r'(\w+)="([^"]*)"'
            var_matches = re.findall(var_pattern, match)
            for key, value in var_matches:
                vars_dict[key] = value
            
            # Also parse unquoted key=value pairs
            unquoted_pattern = r'(\w+)=([^\s"]+)'
            unquoted_matches = re.findall(unquoted_pattern, match)
            for key, value in unquoted_matches:
                if key not in vars_dict:  # Don't overwrite quoted values
                    vars_dict[key] = value
                    
        return vars_dict
    
    def parse_metl_log_timing(self, incremental: bool = True) -> None:
        """Parse metl.log file to extract job timing information.
        
        Extracts submission (000), execution start (001), and termination (005) times
        for all jobs in the metl.log file. This provides accurate runtime duration
        data for completed jobs that may be missing from the DAGMan logs.
        
        The metl.log file contains detailed HTCondor ClassAd events including:
        - Job submission events (000) with DAG Node names
        - Job execution events (001) with start timestamps  
        - Job termination events (005) with completion timestamps
        
        This timing data is then matched with completed jobs from rescue files
        to populate missing duration information in the monitoring display.
        
        Args:
            incremental: If True, only process new entries since last read.
                        If False, process entire log file.
        """
        if not self.metl_log.exists():
            return
            
        try:
            with open(self.metl_log, 'r') as f:
                if incremental:
                    f.seek(self.last_metl_log_position)
                    lines = f.readlines()
                    self.last_metl_log_position = f.tell()
                else:
                    lines = f.readlines()
                    self.metl_job_timing.clear()
                    
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line == '...':
                        i += 1
                        continue
                    
                    # Look for job events (000 = submission, 001 = execution, 005 = termination)
                    event_match = re.match(r'^(000|001|005) \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if event_match:
                        event_code = event_match.group(1)
                        cluster_id = int(event_match.group(2))
                        timestamp_str = event_match.group(3)
                        
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            i += 1
                            continue
                        
                        # Initialize timing dict for this cluster if not exists
                        if cluster_id not in self.metl_job_timing:
                            self.metl_job_timing[cluster_id] = {}
                        
                        # Store timing information based on event type
                        if event_code == '000':  # Job submission
                            self.metl_job_timing[cluster_id]['submit_time'] = timestamp
                            # Look for DAG Node on next line
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                dag_match = re.match(r'^\s*DAG Node: (.+)$', next_line)
                                if dag_match:
                                    dag_node_name = dag_match.group(1)
                                    self.cluster_to_dagnode[cluster_id] = dag_node_name
                                    self.metl_job_timing[cluster_id]['dag_node'] = dag_node_name
                        elif event_code == '001':  # Job executing
                            self.metl_job_timing[cluster_id]['start_time'] = timestamp
                        elif event_code == '005':  # Job terminated
                            self.metl_job_timing[cluster_id]['end_time'] = timestamp
                            
                            # Look for data transfer information in the following lines
                            j = i + 1
                            while j < len(lines) and not lines[j].strip().startswith('...'):
                                line_content = lines[j].strip()
                                bytes_sent_match = re.search(r'(\d+)\s+-\s+Total Bytes Sent By Job', line_content)
                                bytes_received_match = re.search(r'(\d+)\s+-\s+Total Bytes Received By Job', line_content)
                                
                                if bytes_sent_match:
                                    self.metl_job_timing[cluster_id]['total_bytes_sent'] = int(bytes_sent_match.group(1))
                                if bytes_received_match:
                                    self.metl_job_timing[cluster_id]['total_bytes_received'] = int(bytes_received_match.group(1))
                                j += 1
                            
                    i += 1
                    
        except FileNotFoundError:
            pass
        except Exception as e:
            # Log error but don't fail the entire monitoring
            self.console.print(f"[yellow]Warning: Error parsing metl.log: {e}[/yellow]")
    
    def find_rescue_files(self) -> List[Path]:
        """Find all rescue files for this DAG.
        
        Returns:
            List of rescue file paths, sorted by sequence number
        """
        rescue_files: List[Path] = []
        dag_dir = self.dag_file.parent
        dag_basename = self.dag_file.stem
        
        # Look for rescue files (dag.rescue001, dag.rescue002, etc.)
        for rescue_file in dag_dir.glob(f"{dag_basename}.dag.rescue*"):
            rescue_files.append(rescue_file)
        
        # Sort by rescue number
        def rescue_sort_key(path: Path) -> int:
            suffix = path.suffix
            if suffix.startswith('.rescue'):
                try:
                    return int(suffix[7:])  # Extract number after '.rescue'
                except ValueError:
                    return 0
            return 0
        
        return sorted(rescue_files, key=rescue_sort_key)
    
    def parse_rescue_file(self, rescue_file: Path) -> Dict[str, JobStatus]:
        """Parse a rescue file to extract completed job information.
        
        Args:
            rescue_file: Path to the rescue file to parse
            
        Returns:
            Dictionary mapping job names to their status from the rescue file
        """
        job_statuses: Dict[str, JobStatus] = {}
        
        if not rescue_file.exists():
            return job_statuses
            
        try:
            with open(rescue_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('DONE '):
                        job_name = line[5:].strip()  # Remove 'DONE ' prefix
                        job_statuses[job_name] = JobStatus.COMPLETED
                    elif line.startswith('RETRY '):
                        # Extract job name from "RETRY job_name retry_count"
                        parts = line.split()
                        if len(parts) >= 2:
                            job_name = parts[1]
                            # Don't overwrite DONE status with RETRY
                            if job_name not in job_statuses:
                                job_statuses[job_name] = JobStatus.IDLE
        except FileNotFoundError:
            pass
            
        return job_statuses
    
    def process_rescue_files(self) -> None:
        """Process all rescue files to get completed job information."""
        rescue_files = self.find_rescue_files()
        
        for rescue_file in rescue_files:
            if rescue_file in self.rescue_files_processed:
                continue
                
            rescue_statuses = self.parse_rescue_file(rescue_file)
            
            # Update job information based on rescue file
            for job_name, status in rescue_statuses.items():
                # Parse job variables to get metadata if we don't have this job yet
                if job_name not in self.jobs:
                    job_vars = self.parse_job_vars(job_name)
                    self.jobs[job_name] = JobInfo(
                        name=job_name,
                        run_uuid=job_vars.get('run_uuid'),
                        epoch=int(job_vars.get('epoch', 0)) if job_vars.get('epoch') else None,
                        resource_name=job_vars.get('ResourceName')
                    )
                
                job = self.jobs[job_name]
                
                # Only update status if it's not already set from metl.log
                # metl.log is the source of truth for real-time status
                # Rescue files are just DAGMan bookkeeping and can be stale
                if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RUNNING, JobStatus.TRANSFERRING, JobStatus.HELD]:
                    job.status = status
                    
                    # Try to get timing information from metl.log for completed jobs
                    if status == JobStatus.COMPLETED and not job.end_time:
                        timing_applied = False
                        best_timing = None
                        best_cluster_id = None
                        
                        # Search for timing info by looking through all cluster IDs
                        # Find the most recent successful completion for this job
                        # For completed jobs, allow timing from metl.log even if cluster not in current DAGMan log
                        # (DAGMan logs can be rotated, but metl.log might have older data)
                        for cluster_id, timing_info in self.metl_job_timing.items():
                            if (timing_info.get('dag_node') == job_name and 
                                timing_info.get('end_time') and 
                                timing_info.get('start_time')):
                                
                                # For completed jobs from rescue file, accept any matching DAG node
                                # For other jobs, still require cluster to be in current DAG
                                if (status == JobStatus.COMPLETED or 
                                    cluster_id in self.cluster_to_dagnode):
                                    
                                    # Prefer the most recent completion (highest cluster ID as proxy)
                                    if best_timing is None or cluster_id > best_cluster_id:
                                        best_timing = timing_info
                                        best_cluster_id = cluster_id
                        
                        # Apply the best timing information found
                        if best_timing:
                            if 'submit_time' in best_timing:
                                job.submit_time = best_timing['submit_time']
                            if 'start_time' in best_timing:
                                job.start_time = best_timing['start_time']
                            if 'end_time' in best_timing:
                                job.end_time = best_timing['end_time']
                            if 'total_bytes_sent' in best_timing:
                                job.total_bytes_sent = best_timing['total_bytes_sent']
                            if 'total_bytes_received' in best_timing:
                                job.total_bytes_received = best_timing['total_bytes_received']
                            if best_cluster_id:
                                job.cluster_id = best_cluster_id
                            timing_applied = True
                        
                        # If no timing found in metl.log, use rescue file modification time as fallback
                        if not timing_applied and not job.end_time:
                            try:
                                job.end_time = datetime.fromtimestamp(rescue_file.stat().st_mtime)
                            except OSError:
                                job.end_time = datetime.now()
                
                # Update training run status
                if job.run_uuid:
                    self.update_training_run_status(job)
        
        # Apply metl.log timing and transfer data to all jobs that have it
        # This is separate from rescue file status since jobs may have completed
        # but been restarted, so they don't show as COMPLETED in rescue files
        self.apply_metl_data_to_all_jobs()
            
        self.rescue_files_processed.add(rescue_file)
    
    def apply_metl_data_to_all_jobs(self) -> None:
        """Apply metl.log timing and transfer data to all jobs that have it.
        
        This method applies timing and transfer data from metl.log to jobs
        regardless of their rescue file status, since jobs may have completed
        but been restarted and thus don't show as COMPLETED in rescue files.
        
        IMPORTANT: Only applies data for jobs that are actually defined in the current DAG,
        to avoid mixing data from other DAGs that share the same metl.log file.
        """
        # Get the set of job names actually defined in this DAG
        defined_jobs = set(self.jobs.keys())
        
        # Pre-filter timing info to only include DAG-relevant clusters
        # This avoids O(n²) behavior in the inner loop
        # For completed jobs, include timing even if cluster not in current DAGMan log
        dag_relevant_timing = {}
        for cluster_id, timing_info in self.metl_job_timing.items():
            dag_node = timing_info.get('dag_node')
            if (dag_node in defined_jobs and 
                timing_info.get('end_time')):
                
                # Allow timing data for jobs defined in this DAG
                # Even if cluster not in current DAGMan log (for older completed jobs)
                
                if dag_node not in dag_relevant_timing:
                    dag_relevant_timing[dag_node] = []
                dag_relevant_timing[dag_node].append((cluster_id, timing_info))
        
        # Sort timing entries by cluster ID for each job to prefer most recent
        for job_name in dag_relevant_timing:
            dag_relevant_timing[job_name].sort(key=lambda x: x[0], reverse=True)
        
        for job in self.jobs.values():
            if hasattr(job, '_metl_data_applied'):
                continue
                
            # Get the best (most recent) timing info for this job
            timing_entries = dag_relevant_timing.get(job.name, [])
            if timing_entries:
                best_cluster_id, best_timing = timing_entries[0]
                
                # Apply the best timing and transfer information found
                if 'total_bytes_sent' in best_timing:
                    job.total_bytes_sent = best_timing['total_bytes_sent']
                if 'total_bytes_received' in best_timing:
                    job.total_bytes_received = best_timing['total_bytes_received']
                # Also apply timing if not already set
                if 'submit_time' in best_timing and not job.submit_time:
                    job.submit_time = best_timing['submit_time']
                if 'start_time' in best_timing and not job.start_time:
                    job.start_time = best_timing['start_time']
                if 'end_time' in best_timing and not job.end_time:
                    job.end_time = best_timing['end_time']
                if best_cluster_id and not job.cluster_id:
                    job.cluster_id = best_cluster_id
                
                job._metl_data_applied = True
    
    def _filter_cluster_mapping_to_current_dag(self) -> None:
        """Filter cluster_to_dagnode mapping to only include clusters from this DAG.
        
        Since metl.log is shared between DAGs, we need to filter out cluster mappings
        from other DAGs to prevent cross-contamination of job data.
        """
        if not self.dagman_log.exists():
            return
            
        # Get the set of cluster IDs that actually appear in this DAG's logs
        dag_cluster_ids = set()
        
        try:
            with open(self.dagman_log, 'r') as f:
                content = f.read()
                
            # Look for cluster ID assignments in this DAG's log
            import re
            cluster_matches = re.findall(r'assigned HTCondor ID \((\d+)\.\d+\.\d+\)', content)
            dag_cluster_ids.update(int(cid) for cid in cluster_matches)
            
        except (OSError, IOError):
            pass
        
        # Filter cluster_to_dagnode to only include clusters from this DAG
        filtered_mapping = {}
        for cluster_id, dag_node in self.cluster_to_dagnode.items():
            if cluster_id in dag_cluster_ids:
                filtered_mapping[cluster_id] = dag_node
        
        self.cluster_to_dagnode = filtered_mapping
    
    def _parse_planned_training_runs(self) -> None:
        """Parse the DAG file to extract all planned training runs and their epochs.
        
        This builds a complete picture of what's planned, not just what's been submitted.
        """
        self.planned_training_runs.clear()
        
        content = self._get_dag_file_content()
        if not content:
            return
        
        # Find all JOB lines that match training pattern
        job_pattern = r'JOB\s+(run\d+-train_epoch\d+)\s+'
        job_matches = re.findall(job_pattern, content, re.MULTILINE)
        
        # Find all VARS lines with run_uuid and epoch
        vars_pattern = r'VARS\s+(run\d+-train_epoch\d+)\s+.*?epoch="(\d+)".*?run_uuid="([^"]+)"'
        vars_matches = re.findall(vars_pattern, content, re.MULTILINE)
        
        # Build mapping of job_name -> (run_uuid, epoch)
        job_to_run_epoch = {}
        for job_name, epoch, run_uuid in vars_matches:
            job_to_run_epoch[job_name] = (run_uuid, int(epoch))
        
        # Group by run_uuid and count epochs
        for job_name in job_matches:
            if job_name in job_to_run_epoch:
                run_uuid, epoch = job_to_run_epoch[job_name]
                
                if run_uuid not in self.planned_training_runs:
                    self.planned_training_runs[run_uuid] = {
                        'total_epochs': 0,
                        'max_epoch': 0,
                        'job_names': []
                    }
                
                self.planned_training_runs[run_uuid]['total_epochs'] += 1
                self.planned_training_runs[run_uuid]['max_epoch'] = max(
                    self.planned_training_runs[run_uuid]['max_epoch'], epoch
                )
                self.planned_training_runs[run_uuid]['job_names'].append(job_name)
    
    def build_cluster_mapping(self, lines: List[str]) -> None:
        """Build mapping from HTCondor cluster IDs to DAG node names.
        
        Args:
            lines: List of lines from the log file
        """
        self.cluster_to_dagnode.clear()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line == '...':
                i += 1
                continue
            
            # Look for job events followed by DAG node info
            event = self.parser.parse_log_line(line)
            if event and event.get('event_type') != 'dag_node' and 'cluster_id' in event:
                cluster_id = event['cluster_id']
                # Look ahead for DAG Node line (should be next line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    next_event = self.parser.parse_log_line(next_line)
                    if (next_event and 
                        next_event.get('event_type') == 'dag_node' and 
                        cluster_id not in self.cluster_to_dagnode):
                        dag_node_name = next_event['dag_node_name']
                        self.cluster_to_dagnode[cluster_id] = dag_node_name
            
            i += 1
    
    def natural_sort_key(self, job_name: str) -> Tuple[int, int]:
        """Generate sort key for natural ordering of job names.
        
        Args:
            job_name: Job name like "run10-train_epoch5"
            
        Returns:
            Tuple of (run_number, epoch_number) for sorting
        """
        try:
            match = re.match(DAG_FILE_PATTERNS['JOB_NAME'], job_name)
            if match:
                run_num = int(match.group(1))
                epoch_num = int(match.group(2))
                return (run_num, epoch_num)
        except (ValueError, AttributeError):
            pass
        
        # Fallback for any other format
        return (999, 999)
    
    def update_job_info(self, event: Dict[str, Any], job_name: str) -> None:
        """Update job information based on log event.
        
        Args:
            event: Parsed log event dictionary
            job_name: Name of the job to update
        """
        if job_name not in self.jobs:
            # Parse job variables to get metadata if we have a DAG node name
            if not job_name.startswith('cluster_'):
                job_vars = self.parse_job_vars(job_name)
                self.jobs[job_name] = JobInfo(
                    name=job_name,
                    run_uuid=job_vars.get('run_uuid'),
                    epoch=int(job_vars.get('epoch', 0)) if job_vars.get('epoch') else None,
                    resource_name=job_vars.get('ResourceName')
                )
            else:
                self.jobs[job_name] = JobInfo(name=job_name)
        
        job = self.jobs[job_name]
        job.last_event_time = event['timestamp']
        
        if 'cluster_id' in event:
            job.cluster_id = event['cluster_id']
        
        # Update status based on event type
        if event['event_type'] == 'job_submitted':
            job.status = JobStatus.IDLE
            # Reset execution times for new submission (retry/resubmission)
            if job.submit_time is None or event['timestamp'] > job.submit_time:
                job.submit_time = event['timestamp']
                job.start_time = None  # Clear previous execution time
                job.end_time = None    # Clear previous end time
        elif event['event_type'] == 'job_executing':
            job.status = JobStatus.RUNNING
            # Only update if this is a newer execution
            if job.start_time is None or event['timestamp'] > job.start_time:
                job.start_time = event['timestamp']
                job.end_time = None  # Clear any previous end time
        elif event['event_type'] == 'job_terminated':
            job.status = JobStatus.COMPLETED
            # Only update end time if we have a valid start time and this is after it
            if job.start_time and event['timestamp'] > job.start_time:
                job.end_time = event['timestamp']
        elif event['event_type'] == 'job_held':
            job.status = JobStatus.HELD
        elif event['event_type'] == 'job_released':
            job.status = JobStatus.IDLE
        elif event['event_type'] == 'transfer_input_started':
            job.status = JobStatus.TRANSFERRING
        elif event['event_type'] == 'transfer_input_finished':
            job.status = JobStatus.IDLE
            
        # Update training run status
        if job.run_uuid is not None:
            self.update_training_run_status(job)
    
    
    def update_training_run_status(self, job: JobInfo) -> None:
        """Update training run aggregated status.
        
        Args:
            job: JobInfo object to update training run status for
        """
        run_uuid = job.run_uuid
        if not run_uuid:
            return
            
        if run_uuid not in self.training_runs:
            # Use planned total epochs if available, otherwise fall back to submitted count
            planned_total = (self.planned_training_runs.get(run_uuid, {})
                           .get('total_epochs', 0))
            self.training_runs[run_uuid] = TrainingRunStatus(
                run_uuid=run_uuid,
                total_epochs=planned_total
            )
        
        tr_status = self.training_runs[run_uuid]
        tr_status.jobs[job.name] = job
        
        # Count epochs from submitted jobs
        completed = sum(1 for j in tr_status.jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in tr_status.jobs.values() if j.status == JobStatus.FAILED)
        submitted_total = len(tr_status.jobs)
        
        tr_status.completed_epochs = completed
        tr_status.failed_epochs = failed
        
        # Use planned total if available and larger than submitted count
        planned_total = (self.planned_training_runs.get(run_uuid, {})
                        .get('total_epochs', 0))
        tr_status.total_epochs = max(planned_total, submitted_total)
    
    def _initialize_planned_training_runs(self) -> None:
        """Initialize training run status for all planned runs from DAG file."""
        for run_uuid, planned_info in self.planned_training_runs.items():
            if run_uuid not in self.training_runs:
                self.training_runs[run_uuid] = TrainingRunStatus(
                    run_uuid=run_uuid,
                    total_epochs=planned_info['total_epochs']
                )
    
    def process_log_entries(self, incremental: bool = True) -> None:
        """Process log entries from DAG log files.
        
        Args:
            incremental: If True, only process new entries since last read.
                        If False, process entire log file.
        """
        # Parse metl.log first to get timing information
        self.parse_metl_log_timing(incremental)
        
        # Filter cluster_to_dagnode to only include clusters that appear in this DAG's logs
        self._filter_cluster_mapping_to_current_dag()
        
        # Process rescue files to get authoritative completion status (after metl.log parsing)
        self.process_rescue_files()
        
        # Find the best log file to use
        log_file = self._select_log_file()
        if not log_file:
            return
            
        try:
            lines = self._read_log_file(log_file, incremental)
            if not lines:
                return
                
            # Build cluster to DAG node mapping if processing full file
            if not incremental:
                self.build_cluster_mapping(lines)
                self._initialize_planned_training_runs()
            
            self._process_log_lines(lines)
                        
        except (FileNotFoundError, IOError):
            pass
    
    def _select_log_file(self) -> Optional[Path]:
        """Select the best available log file to process."""
        candidates = [
            self.nodes_log,
            self.dagman_log, 
            Path("metl.log")
        ]
        
        for log_file in candidates:
            if log_file.exists():
                return log_file
        return None
    
    def _read_log_file(self, log_file: Path, incremental: bool) -> List[str]:
        """Read lines from log file with proper positioning and error handling."""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                if incremental:
                    # Ensure seek position is valid
                    file_size = f.seek(0, 2)  # Seek to end to get file size
                    if self.last_log_position > file_size:
                        self.last_log_position = 0  # Reset if file was truncated
                    f.seek(self.last_log_position)
                    lines = f.readlines()
                    self.last_log_position = f.tell()
                else:
                    lines = f.readlines()
                    self.last_log_position = 0
            return lines
        except (OSError, IOError) as e:
            self.console.print(f"[yellow]Warning: Error reading log file {log_file}: {e}[/yellow]")
            return []
    
    def _process_log_lines(self, lines: List[str]) -> None:
        """Process a list of log lines for events."""
        for line in lines:
            line = line.strip()
            if not line or line == '...':
                continue
            
            event = self.parser.parse_log_line(line)
            if not event or event.get('event_type') == 'dag_node':
                continue
                
            if 'cluster_id' in event:
                cluster_id = event['cluster_id']
                job_name = self.cluster_to_dagnode.get(cluster_id, f'cluster_{cluster_id}')
                self.update_job_info(event, job_name)
    
    
    def get_htcondor_status(self) -> Dict[int, Dict[str, Any]]:
        """Get job status from HTCondor queue with caching.
        
        Returns:
            Dictionary mapping cluster IDs to HTCondor job information
        """
        current_time = time.time()
        
        # Return cached result if still valid
        if (self._htcondor_status_cache is not None and 
            current_time - self._htcondor_cache_time < self._htcondor_cache_ttl):
            return self._htcondor_status_cache
        
        job_status: Dict[int, Dict[str, Any]] = {}
        try:
            # Use condor_q to get current job status
            result = subprocess.run(
                ['condor_q', '-json'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                jobs_data = json.loads(result.stdout)
                for job in jobs_data:
                    cluster_id = job.get('ClusterId')
                    if cluster_id:
                        job_status[cluster_id] = {
                            'status': job.get('JobStatus', 0),
                            'hold_reason': job.get('HoldReason', ''),
                            'last_job_status': job.get('LastJobStatus', 0)
                        }
            
            # Cache the result
            self._htcondor_status_cache = job_status
            self._htcondor_cache_time = current_time
            
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            # HTCondor not available - return empty status
            # Only log if it's an unexpected error (not just htcondor not found)
            if not isinstance(e, FileNotFoundError):
                self.console.print(f"[dim]HTCondor query failed: {type(e).__name__}[/dim]")
            
            # Cache empty result to avoid repeated failed calls
            self._htcondor_status_cache = job_status
            self._htcondor_cache_time = current_time
            
        return job_status
    
    def get_queued_cluster_ids(self) -> Set[int]:
        """Get set of cluster IDs that are currently queued in HTCondor.
        
        Returns:
            Set of cluster IDs that are actively in HTCondor queue
        """
        htc_status = self.get_htcondor_status()
        return set(htc_status.keys())
    
    def should_show_job(self, job: JobInfo, queued_cluster_ids: Set[int]) -> bool:
        """Determine if a job should be shown in the display.
        
        Shows jobs that are:
        - COMPLETED (from rescue file or logs)
        - RUNNING (currently executing)
        - IDLE jobs that are actually queued in HTCondor (when HTCondor is available)
        - HELD jobs
        - FAILED jobs
        
        When HTCondor is not available, shows jobs that have been submitted (have cluster IDs).
        Excludes IDLE jobs that are planned in DAG but not yet submitted.
        
        Args:
            job: JobInfo object to check
            queued_cluster_ids: Set of cluster IDs currently in HTCondor queue
            
        Returns:
            True if job should be displayed, False otherwise
        """
        # Always show these definitive statuses
        definitive_statuses = {
            JobStatus.COMPLETED, JobStatus.RUNNING, JobStatus.TRANSFERRING, 
            JobStatus.HELD, JobStatus.FAILED
        }
        if job.status in definitive_statuses:
            return True
        
        # For IDLE and UNKNOWN jobs, show based on queue status or cluster ID
        if job.status in [JobStatus.IDLE, JobStatus.UNKNOWN]:
            return self._should_show_queued_job(job, queued_cluster_ids)
        
        return False
    
    def _should_show_queued_job(self, job: JobInfo, queued_cluster_ids: Set[int]) -> bool:
        """Check if a queued job should be shown based on HTCondor availability."""
        if job.cluster_id is None:
            return False
        
        # If HTCondor is available, check if job is in queue
        if queued_cluster_ids:
            return job.cluster_id in queued_cluster_ids
        
        # HTCondor not available, show if job has cluster ID (was submitted)
        return True
    
    def create_status_table(self, verbose: bool = False, exclude_helper: bool = True, show_all: bool = False) -> Table:
        """Create a rich table showing current job status.
        
        By default, shows only jobs that are actively in the system:
        - COMPLETED jobs (from rescue file or logs)
        - RUNNING jobs (currently executing)
        - IDLE jobs that are actually queued in HTCondor
        - HELD jobs
        - FAILED jobs
        
        Args:
            verbose: Currently unused, kept for API compatibility
            exclude_helper: Exclude annex_helper jobs if True
            show_all: Show all jobs including planned but unsubmitted ones if True
            
        Returns:
            Rich Table object with job status information
        """
        # Determine table title based on filtering and display options
        filter_suffix = f" [Filtered: {', '.join(sorted(self.run_filter))}]" if self.run_filter else ""
        
        if show_all:
            title = f"DAG Job Status (All Jobs){filter_suffix}"
        elif verbose:
            title = f"DAG Job Status (Active Jobs Only){filter_suffix}"
        else:
            title = f"DAG Job Status (Latest Epoch per Run){filter_suffix}"
            
        table = Table(title=title)
            
        table.add_column("Run", justify="right", style="cyan", no_wrap=True)
        table.add_column("Epoch", justify="right", style="green")
        table.add_column("Run UUID", style="blue", max_width=8)
        table.add_column("HTCondor Job ID", justify="right", style="dim white", max_width=12)
        table.add_column("Targeted Resource", style="yellow", max_width=15)
        table.add_column("Duration", style="white")
        table.add_column("Status", style="magenta")
        
        # Get currently queued cluster IDs from HTCondor
        queued_cluster_ids = self.get_queued_cluster_ids()
        
        # Filter jobs: exclude helper jobs and optionally filter to active jobs only
        filtered_jobs = []
        for job in self.jobs.values():
            if exclude_helper and job.name == "annex_helper":
                continue
            if not self._matches_run_filter(job):
                continue
            if show_all or self.should_show_job(job, queued_cluster_ids):
                filtered_jobs.append(job)
        
        # If not verbose, show only the latest epoch per training run
        if not verbose:
            # Group jobs by training run (run_uuid)
            jobs_by_run = {}
            for job in filtered_jobs:
                if job.run_uuid:
                    if job.run_uuid not in jobs_by_run:
                        jobs_by_run[job.run_uuid] = []
                    jobs_by_run[job.run_uuid].append(job)
                else:
                    # Jobs without run_uuid (shouldn't happen, but handle gracefully)
                    if 'no_uuid' not in jobs_by_run:
                        jobs_by_run['no_uuid'] = []
                    jobs_by_run['no_uuid'].append(job)
            
            # For each run, keep only the job with the highest epoch number
            latest_jobs = []
            for _, jobs in jobs_by_run.items():
                if jobs:
                    # Sort by epoch number and take the highest
                    jobs_with_epochs = [j for j in jobs if j.epoch is not None]
                    jobs_without_epochs = [j for j in jobs if j.epoch is None]
                    
                    if jobs_with_epochs:
                        latest_job = max(jobs_with_epochs, key=lambda x: x.epoch)
                        latest_jobs.append(latest_job)
                    elif jobs_without_epochs:
                        # If no epochs, just take the first job
                        latest_jobs.append(jobs_without_epochs[0])
            
            filtered_jobs = latest_jobs
        
        sorted_jobs = sorted(filtered_jobs, key=lambda x: self.natural_sort_key(x.name))
        
        seen_runs: Set[str] = set()
        
        for job in sorted_jobs:
            duration = ""
            if job.status == JobStatus.IDLE:
                duration = ""
            elif job.start_time and job.end_time:
                duration_delta = job.end_time - job.start_time
                if duration_delta.total_seconds() >= 0:
                    duration = str(duration_delta).split('.')[0]
                else:
                    duration = "[red]negative[/red]"
            elif job.start_time:
                duration_delta = datetime.now() - job.start_time
                if duration_delta.total_seconds() >= 0:
                    duration = str(duration_delta).split('.')[0]
                else:
                    duration = "[red]clock skew[/red]"
            
            # Extract run number from job name
            run_match = re.match(r'run(\d+)-train_epoch(\d+)', job.name)
            run_number = run_match.group(1) if run_match else job.name
            
            # Only show run number if we haven't seen it before
            display_run = run_number if run_number not in seen_runs else ""
            seen_runs.add(run_number)
            
            # Color-code status
            status_colors = {
                "COMPLETED": "[green]COMPLETED[/green]",
                "RUNNING": "[blue]RUNNING[/blue]",
                "TRANSFERRING": "[cyan]TRANSFERRING[/cyan]",
                "IDLE": "[yellow]IDLE[/yellow]",
                "HELD": "[red]HELD[/red]",
                "FAILED": "[red]FAILED[/red]",
            }
            status_style = status_colors.get(job.status.value, job.status.value)
            
            # Show cluster ID if available, otherwise show "N/A" for completed jobs
            if job.cluster_id:
                cluster_display = str(job.cluster_id)
            elif job.status == JobStatus.COMPLETED:
                cluster_display = "N/A"
            else:
                cluster_display = ""
            
            # Prepare row data
            row_data = [
                display_run,
                str(job.epoch) if job.epoch else "",
                job.run_uuid[:8] if job.run_uuid else "",
                cluster_display,
                job.resource_name or "",
                duration,
                status_style
            ]
            
            table.add_row(*row_data)
        
        return table
    
    def create_training_summary_table(self, exclude_helper: bool = True, show_all: bool = False) -> Table:
        """Create a summary table of training progress.
        
        By default shows only active jobs (same filtering as status table).
        
        Args:
            exclude_helper: Exclude annex_helper jobs if True
            show_all: Show all jobs including planned but unsubmitted ones if True
            verbose: Show all epochs if True, latest epoch per run if False
            
        Returns:
            Rich Table object with training summary
        """
        # Determine table title based on filtering
        filter_suffix = f" [Filtered: {', '.join(sorted(self.run_filter))}]" if self.run_filter else ""
        
        if show_all:
            title = f"Training Summary (All Jobs){filter_suffix}"
        else:
            title = f"Training Summary (Active Jobs Only){filter_suffix}"
            
        table = Table(title=title)
            
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="green")
        table.add_column("Total Time", style="yellow")
        
        # Get currently queued cluster IDs from HTCondor
        queued_cluster_ids = self.get_queued_cluster_ids()
        
        # Filter jobs: exclude helper jobs and optionally filter to active jobs only
        # For training summary, we always want ALL active jobs to get accurate counts
        filtered_jobs = []
        for job in self.jobs.values():
            if exclude_helper and job.name == "annex_helper":
                continue
            if not self._matches_run_filter(job):
                continue
            if show_all or self.should_show_job(job, queued_cluster_ids):
                filtered_jobs.append(job)
        
        if not filtered_jobs:
            table.add_row("No training jobs found", "", "")
            return table
        
        # Categorize jobs
        completed_jobs = [job for job in filtered_jobs if job.status == JobStatus.COMPLETED]
        running_jobs = [job for job in filtered_jobs if job.status == JobStatus.RUNNING]
        transferring_jobs = [job for job in filtered_jobs if job.status == JobStatus.TRANSFERRING]
        failed_jobs = [job for job in filtered_jobs if job.status == JobStatus.FAILED]
        idle_jobs = [job for job in filtered_jobs if job.status == JobStatus.IDLE]
        held_jobs = [job for job in filtered_jobs if job.status == JobStatus.HELD]
        
        # Calculate total time for completed epochs
        completed_time = timedelta(0)
        for job in completed_jobs:
            if job.start_time and job.end_time:
                completed_time += job.end_time - job.start_time
        
        # Calculate total time for running epochs (so far)
        running_time = timedelta(0)
        for job in running_jobs:
            if job.start_time:
                running_time += datetime.now() - job.start_time
        
        # Add summary rows
        table.add_row("Epochs Completed", str(len(completed_jobs)), 
                     str(completed_time).split('.')[0] if completed_time else "")
        table.add_row("Epochs Running", str(len(running_jobs)), 
                     str(running_time).split('.')[0] if running_time else "")
        table.add_row("Epochs Transferring", str(len(transferring_jobs)), "")
        table.add_row("Epochs Idle", str(len(idle_jobs)), "")
        table.add_row("Epochs Failed", str(len(failed_jobs)), "")
        table.add_row("Epochs Held", str(len(held_jobs)), "")
        
        # Total compute time (completed + ongoing)
        total_compute_time = completed_time + running_time
        table.add_row("Total Compute Time", "", 
                     str(total_compute_time).split('.')[0] if total_compute_time else "")
        
        return table
    
    def create_training_run_table(self) -> Table:
        """Create a table showing training run progress.
        
        Shows only training runs that have active jobs.
        
        Returns:
            Rich Table object with training run progress
        """
        # Determine table title based on filtering
        filter_suffix = f" [Filtered: {', '.join(sorted(self.run_filter))}]" if self.run_filter else ""
        title = f"Training Run Progress (Active Runs Only){filter_suffix}"
        table = Table(title=title)
        table.add_column("Run", justify="right", style="cyan", no_wrap=True)
        table.add_column("Run UUID", style="blue", max_width=8)
        table.add_column("Progress", style="magenta")
        table.add_column("Completed", justify="right", style="green") 
        table.add_column("Running", justify="right", style="blue")
        table.add_column("Idle (Queued)", justify="right", style="yellow")
        table.add_column("Held", justify="right", style="red")
        table.add_column("Failed", justify="right", style="red")
        
        # Get currently queued cluster IDs from HTCondor
        queued_cluster_ids = self.get_queued_cluster_ids()
        
        # Sort by run number extracted from the first job name in each training run
        def get_run_number(tr):
            if tr.jobs:
                # Get the first job name to extract run number
                first_job_name = next(iter(tr.jobs.keys()))
                match = re.match(r'run(\d+)-', first_job_name)
                if match:
                    return int(match.group(1))
            return 999  # fallback for runs without jobs
        
        for tr in sorted(self.training_runs.values(), key=get_run_number):
            # Count job states, but only for active jobs that match the filter
            active_jobs = [j for j in tr.jobs.values() 
                          if self.should_show_job(j, queued_cluster_ids) and self._matches_run_filter(j)]
            
            if not active_jobs:
                # Skip training runs with no active jobs
                continue
            
            running = sum(1 for j in active_jobs if j.status == JobStatus.RUNNING)
            idle_queued = sum(1 for j in active_jobs if j.status == JobStatus.IDLE)
            completed = sum(1 for j in active_jobs if j.status == JobStatus.COMPLETED)
            held = sum(1 for j in active_jobs if j.status == JobStatus.HELD)
            failed = sum(1 for j in active_jobs if j.status == JobStatus.FAILED)
            
            # Calculate progress based on planned total
            progress_percent = (completed / tr.total_epochs * 100) if tr.total_epochs > 0 else 0
            progress_bar = f"[{'█' * int(progress_percent / 5)}{'░' * (20 - int(progress_percent / 5))}] {progress_percent:.1f}%"
            
            # Extract run number for display
            run_number = get_run_number(tr)
            run_display = str(run_number) if run_number != 999 else "?"
            
            table.add_row(
                run_display,
                tr.run_uuid[:8],
                progress_bar,
                str(completed),
                str(running),
                str(idle_queued),
                str(held),
                str(failed)
            )
        
        return table
    
    def export_to_csv(self, output_file: str, show_all: bool = False) -> None:
        """Export job status data to CSV file.
        
        Args:
            output_file: Path to output CSV file
            show_all: Show all jobs including planned but unsubmitted ones
            
        Raises:
            ValueError: If output_file is empty
            OSError: If unable to write to output file
        """
        if not output_file or not output_file.strip():
            raise ValueError("Output file path cannot be empty")
            
        output_path = Path(output_file)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get currently queued cluster IDs from HTCondor
        queued_cluster_ids = self.get_queued_cluster_ids()
        
        # Filter jobs: exclude helper jobs and optionally filter to active jobs only
        filtered_jobs = []
        for job in self.jobs.values():
            if job.name == "annex_helper":
                continue
            if not self._matches_run_filter(job):
                continue
            if show_all or self.should_show_job(job, queued_cluster_ids):
                filtered_jobs.append(job)
        
        # Sort jobs
        sorted_jobs = sorted(filtered_jobs, key=lambda x: self.natural_sort_key(x.name))
        
        # Write to CSV
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Job Name', 'Run Number', 'Epoch', 'Run UUID', 'HTCondor Cluster ID', 
                    'Targeted Resource', 'Status', 'Submit Time', 'Start Time', 'End Time', 
                    'Duration (seconds)', 'Duration (human)', 'Total Bytes Sent', 'Total Bytes Received'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for job in sorted_jobs:
                    # Extract run number from job name
                    run_match = re.match(r'run(\d+)-train_epoch(\d+)', job.name)
                    run_number = run_match.group(1) if run_match else ""
                    
                    # Calculate duration
                    duration_seconds = ""
                    duration_human = ""
                    if job.status == JobStatus.IDLE:
                        duration_seconds = ""
                        duration_human = ""
                    elif job.start_time and job.end_time:
                        duration_delta = job.end_time - job.start_time
                        if duration_delta.total_seconds() >= 0:
                            duration_seconds = str(int(duration_delta.total_seconds()))
                            duration_human = str(duration_delta).split('.')[0]
                        else:
                            duration_seconds = "0"
                            duration_human = "negative"
                    elif job.start_time:
                        duration_delta = datetime.now() - job.start_time
                        if duration_delta.total_seconds() >= 0:
                            duration_seconds = str(int(duration_delta.total_seconds()))
                            duration_human = str(duration_delta).split('.')[0]
                        else:
                            duration_seconds = "0"
                            duration_human = "clock skew"
                    
                    # Format timestamps
                    submit_time_str = job.submit_time.isoformat() if job.submit_time else ""
                    start_time_str = job.start_time.isoformat() if job.start_time else ""
                    end_time_str = job.end_time.isoformat() if job.end_time else ""
                    
                    # Map resource names: keep major resources, others become "ospool"
                    major_resources = {"expanse", "bridges2", "delta", "anvil"}
                    resource_name = job.resource_name or ""
                    if resource_name and resource_name.lower() not in major_resources:
                        resource_name = "ospool"
                    
                    writer.writerow({
                        'Job Name': job.name,
                        'Run Number': run_number,
                        'Epoch': job.epoch if job.epoch is not None else "",
                        'Run UUID': job.run_uuid or "",
                        'HTCondor Cluster ID': job.cluster_id if job.cluster_id is not None else "",
                        'Targeted Resource': resource_name,
                        'Status': job.status.value,
                        'Submit Time': submit_time_str,
                        'Start Time': start_time_str,
                        'End Time': end_time_str,
                        'Duration (seconds)': duration_seconds,
                        'Duration (human)': duration_human,
                        'Total Bytes Sent': job.total_bytes_sent,
                        'Total Bytes Received': job.total_bytes_received
                    })
                    
            self.console.print(f"[green]CSV data exported to {output_file}[/green]")
            
        except IOError as e:
            self.console.print(f"[red]Error writing CSV file {output_file}: {e}[/red]")
    
    def monitor_once(self, verbose: bool = False, show_all: bool = False, show_progress: bool = False) -> None:
        """Single monitoring cycle - process logs and update status.
        
        Args:
            verbose: Include HTCondor job IDs in output
            show_all: Show all jobs including planned but unsubmitted ones
            show_progress: Show training run progress table
        """
        self.process_log_entries(incremental=True)
        
        # Display tables
        job_table = self.create_status_table(verbose=verbose, show_all=show_all)
        summary_table = self.create_training_summary_table(show_all=show_all)
        
        self.console.print(job_table)
        self.console.print()
        self.console.print(summary_table)
        
        if show_progress and self.training_runs:
            self.console.print()
            training_table = self.create_training_run_table()
            self.console.print(training_table)
        
    def monitor_live(self, refresh_interval: float = 2.0, verbose: bool = False, show_all: bool = False, show_progress: bool = False) -> None:
        """Live monitoring with rich display updates.
        
        Args:
            refresh_interval: Time in seconds between updates
            verbose: Include HTCondor job IDs in output
            show_all: Show all jobs including planned but unsubmitted ones
            show_progress: Show training run progress table
        """
        with Live(self.create_status_table(verbose=verbose, show_all=show_all), 
                 refresh_per_second=1/refresh_interval) as live:
            try:
                while True:
                    self.process_log_entries(incremental=True)
                    
                    # Update display with tables
                    job_table = self.create_status_table(verbose=verbose, show_all=show_all)
                    summary_table = self.create_training_summary_table(show_all=show_all)
                    
                    if show_progress and self.training_runs:
                        training_table = self.create_training_run_table()
                        tables = Columns([job_table, summary_table, training_table])
                    else:
                        tables = Columns([job_table, summary_table])
                    
                    live.update(tables)
                    time.sleep(refresh_interval)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    
    def debug_timing_info(self) -> None:
        """Debug method to show timing information extraction."""
        self.console.print(f"[cyan]Metl.log timing data extracted for {len(self.metl_job_timing)} jobs:[/cyan]")
        
        # Get completed jobs from rescue file for comparison
        rescue_files = self.find_rescue_files()
        completed_in_rescue = set()
        for rescue_file in rescue_files:
            rescue_statuses = self.parse_rescue_file(rescue_file)
            for job_name, status in rescue_statuses.items():
                if status == JobStatus.COMPLETED:
                    completed_in_rescue.add(job_name)
        
        # Show timing data
        timing_jobs = set()
        for cluster_id, timing_info in self.metl_job_timing.items():
            if 'dag_node' in timing_info and timing_info.get('end_time'):
                dag_node = timing_info['dag_node']
                timing_jobs.add(dag_node)
                
                duration = ""
                if timing_info.get('start_time') and timing_info.get('end_time'):
                    duration = str(timing_info['end_time'] - timing_info['start_time']).split('.')[0]
                
                self.console.print(f"  {dag_node} (cluster {cluster_id}): duration={duration}")
        
        self.console.print(f"\n[green]Jobs with timing data: {len(timing_jobs)}[/green]")
        self.console.print(f"[yellow]Jobs completed in rescue file: {len(completed_in_rescue)}[/yellow]")
        
        # Show jobs that are completed but missing timing data
        missing_timing = completed_in_rescue - timing_jobs
        if missing_timing:
            self.console.print(f"\n[red]Completed jobs missing timing data ({len(missing_timing)}):[/red]")
            for job in sorted(missing_timing):
                self.console.print(f"  {job}")


def main() -> None:
    """Main entry point for the DAG monitor."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Monitor DAG execution status")
    parser.add_argument("dag_file", nargs="?", default="pipeline.dag", 
                       help="Path to DAG file (default: pipeline.dag)")
    parser.add_argument("--live", action="store_true", default=True,
                       help="Enable live monitoring (default)")
    parser.add_argument("--once", action="store_true", 
                       help="Run once and exit (disables live monitoring)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Include HTCondor job IDs in output")
    parser.add_argument("--show-all", action="store_true",
                       help="Show all jobs including planned but unsubmitted ones (default: active jobs only)")
    parser.add_argument("--refresh-interval", type=float, default=60,
                       help="Refresh interval in seconds for live monitoring (default: 60)")
    parser.add_argument("--debug-timing", action="store_true",
                       help="Show debug information about timing data extraction")
    parser.add_argument("--show-progress", action="store_true",
                       help="Show training run progress table (default: hidden)")
    parser.add_argument("--filter-runs", type=str, metavar="RUNS",
                       help="Filter to specific training runs (comma-separated list of run numbers or UUIDs). "
                            "Examples: --filter-runs 1,3,5 or --filter-runs abc123,def456 or --filter-runs 1,abc123")
    parser.add_argument("--csv-output", action="store_true",
                       help="Export job status data to CSV file with timestamp and exit")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.refresh_interval <= 0:
        parser.error("Refresh interval must be positive")
        
    # Initialize monitor with error handling
    try:
        monitor = DAGStatusMonitor(args.dag_file)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error initializing monitor: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set run filter if specified
    if args.filter_runs:
        try:
            monitor.set_run_filter(args.filter_runs)
        except ValueError as e:
            print(f"Error in run filter: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Process entire log file initially
    monitor.process_log_entries(incremental=False)
    
    if args.debug_timing:
        monitor.debug_timing_info()
        return
    
    if args.csv_output:
        try:
            # Process rescue files to get complete timing and transfer data
            monitor.process_rescue_files()
            
            # Create progress directory if it doesn't exist
            progress_dir = Path("./progress")
            progress_dir.mkdir(exist_ok=True)
            
            # Generate timestamp-based filename in progress directory
            dag_basename = Path(args.dag_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = progress_dir / f"{dag_basename}_{timestamp}.csv"
            monitor.export_to_csv(str(csv_filename), show_all=args.show_all)
            return
        except (ValueError, OSError) as e:
            print(f"Error exporting CSV: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during CSV export: {e}", file=sys.stderr)
            sys.exit(1)
    
    try:
        if args.once or not args.live:
            monitor.monitor_once(verbose=args.verbose, show_all=args.show_all, show_progress=args.show_progress)
        else:
            monitor.monitor_live(args.refresh_interval, verbose=args.verbose, show_all=args.show_all, show_progress=args.show_progress)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()