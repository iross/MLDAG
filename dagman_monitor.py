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
        
        self.parser = DAGManLogParser()
        self.jobs: Dict[str, JobInfo] = {}
        self.training_runs: Dict[str, TrainingRunStatus] = {}
        self.console = Console()
        
        # Track last processed position in log file for incremental parsing
        self.last_log_position = 0
        
        # Cluster ID to DAG node name mapping
        self.cluster_to_dagnode: Dict[int, str] = {}
        
        # Cache of planned training runs from DAG file
        self.planned_training_runs: Dict[str, Dict[str, Any]] = {}
        self._parse_planned_training_runs()
        
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
        # Try nodes.log first (more detailed), fall back to dagman.log
        log_file = self.nodes_log if self.nodes_log.exists() else self.dagman_log
        
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
                    if event and event.get('event_type') != 'dag_node' and 'cluster_id' in event:
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
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return job_status
    
    def create_status_table(self, verbose: bool = False, exclude_helper: bool = True) -> Table:
        """Create a rich table showing current job status.
        
        Args:
            verbose: Include HTCondor job IDs if True
            exclude_helper: Exclude annex_helper jobs if True
            
        Returns:
            Rich Table object with job status information
        """
        table = Table(title="DAG Job Status")
        table.add_column("Run", justify="right", style="cyan", no_wrap=True)
        table.add_column("Epoch", justify="right", style="green")
        table.add_column("Run UUID", style="blue", max_width=8)
        if verbose:
            table.add_column("HTCondor Job ID", justify="right", style="dim white", max_width=12)
        table.add_column("Targeted Resource", style="yellow", max_width=15)
        table.add_column("Duration", style="white")
        table.add_column("Status", style="magenta")
        
        # Filter and sort jobs
        filtered_jobs = [job for job in self.jobs.values() 
                        if not exclude_helper or job.name != "annex_helper"]
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
                "IDLE": "[yellow]IDLE[/yellow]",
                "HELD": "[red]HELD[/red]",
                "FAILED": "[red]FAILED[/red]"
            }
            status_style = status_colors.get(job.status.value, job.status.value)
            
            # Prepare row data
            row_data = [
                display_run,
                str(job.epoch) if job.epoch else "",
                job.run_uuid[:8] if job.run_uuid else "",
            ]
            
            if verbose:
                row_data.append(str(job.cluster_id) if job.cluster_id else "")
            
            row_data.extend([
                job.resource_name or "",
                duration,
                status_style
            ])
            
            table.add_row(*row_data)
        
        return table
    
    def create_training_summary_table(self, exclude_helper: bool = True) -> Table:
        """Create a summary table of training progress.
        
        Args:
            exclude_helper: Exclude annex_helper jobs if True
            
        Returns:
            Rich Table object with training summary
        """
        table = Table(title="Training Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="green")
        table.add_column("Total Time", style="yellow")
        
        # Filter jobs
        filtered_jobs = [job for job in self.jobs.values() 
                        if not exclude_helper or job.name != "annex_helper"]
        
        if not filtered_jobs:
            table.add_row("No training jobs found", "", "")
            return table
        
        # Categorize jobs
        completed_jobs = [job for job in filtered_jobs if job.status == JobStatus.COMPLETED]
        running_jobs = [job for job in filtered_jobs if job.status == JobStatus.RUNNING]
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
        
        Returns:
            Rich Table object with training run progress
        """
        table = Table(title="Training Run Progress")
        table.add_column("Run UUID", style="cyan", max_width=12)
        table.add_column("Progress", style="magenta")
        table.add_column("Completed", justify="right", style="green") 
        table.add_column("Running", justify="right", style="blue")
        table.add_column("Idle", justify="right", style="yellow")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Total Planned", justify="right", style="dim blue")
        
        for tr in sorted(self.training_runs.values(), key=lambda x: x.run_uuid):
            # Count job states
            running = sum(1 for j in tr.jobs.values() if j.status == JobStatus.RUNNING)
            idle = sum(1 for j in tr.jobs.values() if j.status == JobStatus.IDLE)
            submitted_total = len(tr.jobs)
            
            # Calculate progress based on planned total
            progress_percent = (tr.completed_epochs / tr.total_epochs * 100) if tr.total_epochs > 0 else 0
            progress_bar = f"[{'█' * int(progress_percent / 5)}{'░' * (20 - int(progress_percent / 5))}] {progress_percent:.1f}%"
            
            # Show submitted vs planned in total column
            total_display = f"{tr.total_epochs}"
            if submitted_total < tr.total_epochs:
                # Some epochs not yet submitted
                total_display = f"{submitted_total}/{tr.total_epochs}"
            
            table.add_row(
                tr.run_uuid[:12],
                progress_bar,
                str(tr.completed_epochs),
                str(running),
                str(idle),
                str(tr.failed_epochs),
                total_display
            )
        
        return table
    
    def monitor_once(self, verbose: bool = False) -> None:
        """Single monitoring cycle - process logs and update status.
        
        Args:
            verbose: Include HTCondor job IDs in output
        """
        self.process_log_entries(incremental=True)
        
        # Display tables
        job_table = self.create_status_table(verbose=verbose)
        summary_table = self.create_training_summary_table()
        
        self.console.print(job_table)
        self.console.print()
        self.console.print(summary_table)
        
        if self.training_runs:
            self.console.print()
            training_table = self.create_training_run_table()
            self.console.print(training_table)
        
    def monitor_live(self, refresh_interval: float = 2.0, verbose: bool = False) -> None:
        """Live monitoring with rich display updates.
        
        Args:
            refresh_interval: Time in seconds between updates
            verbose: Include HTCondor job IDs in output
        """
        with Live(self.create_status_table(verbose=verbose), 
                 refresh_per_second=1/refresh_interval) as live:
            try:
                while True:
                    self.process_log_entries(incremental=True)
                    
                    # Update display with tables
                    job_table = self.create_status_table(verbose=verbose)
                    summary_table = self.create_training_summary_table()
                    
                    if self.training_runs:
                        training_table = self.create_training_run_table()
                        tables = Columns([job_table, summary_table, training_table])
                    else:
                        tables = Columns([job_table, summary_table])
                    
                    live.update(tables)
                    time.sleep(refresh_interval)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Monitoring stopped by user[/yellow]")


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
    parser.add_argument("--refresh-interval", type=float, default=60,
                       help="Refresh interval in seconds for live monitoring (default: 60)")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = DAGStatusMonitor(args.dag_file)
    
    # Process entire log file initially
    monitor.process_log_entries(incremental=False)
    
    if args.once or not args.live:
        monitor.monitor_once(verbose=args.verbose)
    else:
        monitor.monitor_live(args.refresh_interval, verbose=args.verbose)


if __name__ == "__main__":
    main()