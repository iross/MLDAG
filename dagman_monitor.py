#!/usr/bin/env python3
"""
DAGMan log monitoring for MLDAG training runs.
Monitors .dag.dagman.log files and tracks job status changes.

This module provides real-time monitoring of HTCondor DAGMan workflows,
parsing log files to track training job progress across distributed compute resources.
"""

import json
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import subprocess

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.columns import Columns


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
        # Updated patterns for modern log format (YYYY-MM-DD HH:MM:SS)
        self.patterns = {
            'job_submitted': re.compile(
                r'^000 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Job submitted from host:'
            ),
            'job_executing': re.compile(
                r'^001 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Job executing on host:'
            ),
            'job_terminated': re.compile(
                r'^005 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Job terminated\.'
            ),
            'job_held': re.compile(
                r'^012 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Job was held\.'
            ),
            'job_released': re.compile(
                r'^013 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Job was released\.'
            ),
            'transfer_input_started': re.compile(
                r'^040 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Started transferring input files'
            ),
            'transfer_input_finished': re.compile(
                r'^040 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Finished transferring input files'
            ),
            'dag_node': re.compile(
                r'^\s*DAG Node: (.+)$'
            ),
        }
        
        # Legacy patterns for older log format (MM/dd/yy HH:MM:SS)
        self.legacy_patterns = {
            'job_submitted': re.compile(
                r'^000 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Job submitted from host:'
            ),
            'job_executing': re.compile(
                r'^001 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Job executing on host:'
            ),
            'job_terminated': re.compile(
                r'^005 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Job terminated\.'
            ),
            'job_held': re.compile(
                r'^012 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Job was held\.'
            ),
            'job_released': re.compile(
                r'^013 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Job was released\.'
            ),
            'transfer_input_started': re.compile(
                r'^040 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Started transferring input files'
            ),
            'transfer_input_finished': re.compile(
                r'^040 \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Finished transferring input files'
            ),
        }
    
    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from log file.
        
        Supports both modern (YYYY-MM-DD HH:MM:SS) and legacy (MM/dd/yy HH:MM:SS) formats.
        
        Args:
            timestamp_str: Timestamp string from log file
            
        Returns:
            Parsed datetime object, or current time if parsing fails
        """
        # Try modern format first
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
            
        # Fall back to legacy format
        try:
            return datetime.strptime(timestamp_str, '%m/%d/%y %H:%M:%S')
        except ValueError:
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
            
        # Try modern patterns first, then legacy
        for pattern_set in [self.patterns, self.legacy_patterns]:
            for event_type, pattern in pattern_set.items():
                if event_type == 'dag_node':
                    continue
                    
                match = pattern.search(line)
                if match:
                    groups = match.groups()
                    
                    # Regular events with cluster ID
                    cluster_id = int(groups[0])
                    timestamp = self.parse_timestamp(groups[1])
                    
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
        """
        self.dag_file = Path(dag_file)
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
        
        
        # Cache job timing information from metl.log
        self.metl_job_timing: Dict[int, Dict[str, datetime]] = {}
        
        # Cache of planned training runs from DAG file
        self.planned_training_runs: Dict[str, Dict[str, Any]] = {}
        self._parse_planned_training_runs()
        
        # Track rescue file parsing to avoid re-parsing
        self.rescue_files_processed: Set[Path] = set()
        
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
        try:
            with open(self.dag_file, 'r') as f:
                content = f.read()
                
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
                    
        except FileNotFoundError:
            pass
            
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
                
                # Only update status if it's not already marked as completed from logs
                # This gives precedence to log-based status over rescue file status
                if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    job.status = status
                    
                    # Try to get timing information from metl.log for completed jobs
                    if status == JobStatus.COMPLETED and not job.end_time:
                        timing_applied = False
                        best_timing = None
                        best_cluster_id = None
                        
                        # Search for timing info by looking through all cluster IDs
                        # Find the most recent successful completion for this job
                        for cluster_id, timing_info in self.metl_job_timing.items():
                            if (timing_info.get('dag_node') == job_name and 
                                timing_info.get('end_time') and 
                                timing_info.get('start_time')):
                                
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
            
            self.rescue_files_processed.add(rescue_file)
    
    def _parse_planned_training_runs(self) -> None:
        """Parse the DAG file to extract all planned training runs and their epochs.
        
        This builds a complete picture of what's planned, not just what's been submitted.
        """
        self.planned_training_runs.clear()
        
        try:
            with open(self.dag_file, 'r') as f:
                content = f.read()
            
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
        
        except FileNotFoundError:
            pass
    
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
        match = re.match(r'run(\d+)-train_epoch(\d+)', job_name)
        if match:
            run_num = int(match.group(1))
            epoch_num = int(match.group(2))
            return (run_num, epoch_num)
        else:
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
            job.submit_time = event['timestamp']
        elif event['event_type'] == 'job_executing':
            job.status = JobStatus.RUNNING
            job.start_time = event['timestamp']
        elif event['event_type'] == 'job_terminated':
            job.status = JobStatus.COMPLETED
            job.end_time = event['timestamp']
        elif event['event_type'] == 'job_held':
            job.status = JobStatus.HELD
        elif event['event_type'] == 'job_released':
            job.status = JobStatus.IDLE
        elif event['event_type'] == 'transfer_input_started':
            job.status = JobStatus.TRANSFERRING
        elif event['event_type'] == 'transfer_input_finished':
            # Job finished transferring, but not yet executing - keep as IDLE until execution starts
            job.status = JobStatus.IDLE
            
        # Update training run status
        if job.run_uuid:
            self.update_training_run_status(job)
    
    
    def update_training_run_status(self, job: JobInfo) -> None:
        """Update training run aggregated status.
        
        Args:
            job: JobInfo object to update training run status for
        """
        if not job.run_uuid:
            return
            
        if job.run_uuid not in self.training_runs:
            # Use planned total epochs if available, otherwise fall back to submitted count
            planned_total = (self.planned_training_runs.get(job.run_uuid, {})
                           .get('total_epochs', 0))
            self.training_runs[job.run_uuid] = TrainingRunStatus(
                run_uuid=job.run_uuid,
                total_epochs=planned_total
            )
        
        tr_status = self.training_runs[job.run_uuid]
        tr_status.jobs[job.name] = job
        
        # Count epochs from submitted jobs
        completed = sum(1 for j in tr_status.jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in tr_status.jobs.values() if j.status == JobStatus.FAILED)
        submitted_total = len(tr_status.jobs)
        
        tr_status.completed_epochs = completed
        tr_status.failed_epochs = failed
        
        # Use planned total if available and larger than submitted count
        planned_total = (self.planned_training_runs.get(job.run_uuid, {})
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
        
        # Process rescue files to get authoritative completion status (after metl.log parsing)
        self.process_rescue_files()
        
        # Try nodes.log first (more detailed), then dagman.log, then metl.log
        metl_log = Path("metl.log")
        if self.nodes_log.exists():
            log_file = self.nodes_log
        elif self.dagman_log.exists():
            log_file = self.dagman_log
        elif metl_log.exists():
            log_file = metl_log
        else:
            return
        
        if not log_file.exists():
            return
            
        try:
            with open(log_file, 'r') as f:
                if incremental:
                    f.seek(self.last_log_position)
                    lines = f.readlines()
                    self.last_log_position = f.tell()
                else:
                    lines = f.readlines()
                    
                # Build cluster to DAG node mapping if processing full file
                if not incremental:
                    self.build_cluster_mapping(lines)
                    # Initialize all planned training runs
                    self._initialize_planned_training_runs()
                
                # Process all events
                for line in lines:
                    line = line.strip()
                    if not line or line == '...':
                        continue
                    
                    event = self.parser.parse_log_line(line)
                    if event:
                        if event.get('event_type') == 'dag_node':
                            continue
                        elif 'cluster_id' in event:
                            cluster_id = event['cluster_id']
                            
                            # Get job name from cluster mapping or use cluster ID
                            job_name = self.cluster_to_dagnode.get(cluster_id, f'cluster_{cluster_id}')
                            
                            self.update_job_info(event, job_name)
                        
        except FileNotFoundError:
            pass
    
    
    def get_htcondor_status(self) -> Dict[int, Dict[str, Any]]:
        """Get job status from HTCondor queue.
        
        Returns:
            Dictionary mapping cluster IDs to HTCondor job information
        """
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
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            # HTCondor not available - return empty status
            pass
            
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
        # Always show completed, running, transferring, held, and failed jobs
        if job.status in [JobStatus.COMPLETED, JobStatus.RUNNING, JobStatus.TRANSFERRING, JobStatus.HELD, JobStatus.FAILED]:
            return True
        
        # For IDLE jobs, show if they're queued in HTCondor OR if HTCondor is unavailable 
        # but they have a cluster ID (indicating they were submitted)
        if job.status == JobStatus.IDLE:
            if queued_cluster_ids:  # HTCondor is available
                return job.cluster_id and job.cluster_id in queued_cluster_ids
            else:  # HTCondor not available, fallback to cluster ID check
                return job.cluster_id is not None
        
        # Don't show unknown status jobs unless they have a cluster ID
        if job.status == JobStatus.UNKNOWN:
            if queued_cluster_ids:  # HTCondor is available
                return job.cluster_id and job.cluster_id in queued_cluster_ids
            else:  # HTCondor not available, show if has cluster ID
                return job.cluster_id is not None
        
        return False
    
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
        if show_all:
            table = Table(title="DAG Job Status (All Jobs)")
        elif verbose:
            table = Table(title="DAG Job Status (Active Jobs Only)")
        else:
            table = Table(title="DAG Job Status (Latest Epoch per Run)")
            
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
            for run_uuid, jobs in jobs_by_run.items():
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
            if job.start_time and job.end_time:
                duration = str(job.end_time - job.start_time).split('.')[0]
            elif job.start_time:
                duration = str(datetime.now() - job.start_time).split('.')[0]
            
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
    
    def create_training_summary_table(self, exclude_helper: bool = True, show_all: bool = False, verbose: bool = False) -> Table:
        """Create a summary table of training progress.
        
        By default shows only active jobs (same filtering as status table).
        
        Args:
            exclude_helper: Exclude annex_helper jobs if True
            show_all: Show all jobs including planned but unsubmitted ones if True
            verbose: Show all epochs if True, latest epoch per run if False
            
        Returns:
            Rich Table object with training summary
        """
        if show_all:
            table = Table(title="Training Summary (All Jobs)")
        else:
            table = Table(title="Training Summary (Active Jobs Only)")
            
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
        table = Table(title="Training Run Progress (Active Runs Only)")
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
            # Count job states, but only for active jobs
            active_jobs = [j for j in tr.jobs.values() if self.should_show_job(j, queued_cluster_ids)]
            
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
        summary_table = self.create_training_summary_table(show_all=show_all, verbose=verbose)
        
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
                    summary_table = self.create_training_summary_table(show_all=show_all, verbose=verbose)
                    
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
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = DAGStatusMonitor(args.dag_file)
    
    # Process entire log file initially
    monitor.process_log_entries(incremental=False)
    
    if args.debug_timing:
        monitor.debug_timing_info()
        return
    
    if args.once or not args.live:
        monitor.monitor_once(verbose=args.verbose, show_all=args.show_all, show_progress=args.show_progress)
    else:
        monitor.monitor_live(args.refresh_interval, verbose=args.verbose, show_all=args.show_all, show_progress=args.show_progress)


if __name__ == "__main__":
    main()