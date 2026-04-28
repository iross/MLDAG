#!/usr/bin/env python3
"""
HTCondor job status monitoring using condor_q and condor_history commands.
Provides detailed job information and resource usage statistics.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
import typer


class HTCondorJobStatus(Enum):
    """HTCondor job status codes"""
    IDLE = 1
    RUNNING = 2
    REMOVED = 3
    COMPLETED = 4
    HELD = 5
    TRANSFERRING_OUTPUT = 6
    SUSPENDED = 7


@dataclass
class HTCondorJob:
    """HTCondor job information"""
    cluster_id: int
    proc_id: int = 0
    status: HTCondorJobStatus = HTCondorJobStatus.IDLE
    owner: str = ""
    job_description: str = ""
    queue_date: Optional[datetime] = None
    job_start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    wall_clock_time: int = 0  # seconds
    cpu_time_used: float = 0.0  # seconds
    memory_usage: int = 0  # MB
    disk_usage: int = 0  # KB
    machine_name: str = ""
    hold_reason: str = ""
    exit_code: Optional[int] = None
    
    # Custom attributes from job ClassAd
    run_uuid: Optional[str] = None
    epoch: Optional[int] = None
    resource_name: Optional[str] = None
    
    @property
    def job_id(self) -> str:
        """Full job ID as cluster.proc"""
        return f"{self.cluster_id}.{self.proc_id}"
    
    @property
    def runtime(self) -> Optional[timedelta]:
        """Calculate job runtime"""
        if self.job_start_date and self.completion_date:
            return self.completion_date - self.job_start_date
        elif self.job_start_date:
            return datetime.now() - self.job_start_date
        return None
    
    @property
    def status_string(self) -> str:
        """Human readable status string"""
        status_map = {
            HTCondorJobStatus.IDLE: "Idle",
            HTCondorJobStatus.RUNNING: "Running", 
            HTCondorJobStatus.REMOVED: "Removed",
            HTCondorJobStatus.COMPLETED: "Completed",
            HTCondorJobStatus.HELD: "Held",
            HTCondorJobStatus.TRANSFERRING_OUTPUT: "Transferring",
            HTCondorJobStatus.SUSPENDED: "Suspended"
        }
        return status_map.get(self.status, "Unknown")


class HTCondorMonitor:
    """Monitor HTCondor jobs using command line tools"""
    
    def __init__(self):
        self.console = Console()
        self.jobs: Dict[str, HTCondorJob] = {}
        
    def run_condor_command(self, cmd: List[str], timeout: int = 30) -> Optional[str]:
        """Run a condor command and return output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.console.print(f"[red]Command failed: {' '.join(cmd)}[/red]")
                self.console.print(f"[red]Error: {result.stderr}[/red]")
                return None
                
        except subprocess.TimeoutExpired:
            self.console.print(f"[red]Command timed out: {' '.join(cmd)}[/red]")
            return None
        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Command error: {e}[/red]") 
            return None
    
    def parse_job_from_classad(self, job_data: Dict) -> HTCondorJob:
        """Parse HTCondor job from ClassAd JSON"""
        cluster_id = job_data.get('ClusterId', 0)
        proc_id = job_data.get('ProcId', 0)
        
        job = HTCondorJob(
            cluster_id=cluster_id,
            proc_id=proc_id,
            status=HTCondorJobStatus(job_data.get('JobStatus', 1)),
            owner=job_data.get('Owner', ''),
            job_description=job_data.get('Cmd', ''),
            wall_clock_time=job_data.get('JobCurrentStartDate', 0),
            cpu_time_used=job_data.get('RemoteUserCpu', 0.0),
            memory_usage=job_data.get('MemoryUsage', 0),
            disk_usage=job_data.get('DiskUsage', 0),
            machine_name=job_data.get('RemoteHost', ''),
            hold_reason=job_data.get('HoldReason', ''),
            exit_code=job_data.get('ExitCode')
        )
        
        # Parse timestamps
        if 'QDate' in job_data:
            job.queue_date = datetime.fromtimestamp(job_data['QDate'])
        if 'JobCurrentStartDate' in job_data:
            job.job_start_date = datetime.fromtimestamp(job_data['JobCurrentStartDate'])
        if 'CompletionDate' in job_data:
            job.completion_date = datetime.fromtimestamp(job_data['CompletionDate'])
            
        # Extract custom job attributes (from VARS in DAG)
        job.run_uuid = job_data.get('run_uuid')
        if 'epoch' in job_data:
            try:
                job.epoch = int(job_data['epoch'])
            except (ValueError, TypeError):
                pass
        job.resource_name = job_data.get('ResourceName')
        
        return job
    
    def get_queue_jobs(self) -> List[HTCondorJob]:
        """Get jobs currently in the queue"""
        output = self.run_condor_command(['condor_q', '-json'])
        if not output:
            return []
            
        try:
            jobs_data = json.loads(output)
            jobs = []
            
            for job_data in jobs_data:
                job = self.parse_job_from_classad(job_data)
                jobs.append(job)
                self.jobs[job.job_id] = job
                
            return jobs
            
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Failed to parse condor_q JSON: {e}[/red]")
            return []
    
    def get_history_jobs(self, constraint: str = "", limit: int = 100) -> List[HTCondorJob]:
        """Get completed jobs from history"""
        cmd = ['condor_history', '-json']
        if constraint:
            cmd.extend(['-constraint', constraint])
        cmd.extend(['-limit', str(limit)])
        
        output = self.run_condor_command(cmd)
        if not output:
            return []
            
        try:
            jobs_data = json.loads(output)
            jobs = []
            
            for job_data in jobs_data:
                job = self.parse_job_from_classad(job_data)
                jobs.append(job)
                self.jobs[job.job_id] = job
                
            return jobs
            
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Failed to parse condor_history JSON: {e}[/red]")
            return []
    
    def get_jobs_by_run_uuid(self, run_uuid: str) -> List[HTCondorJob]:
        """Get all jobs for a specific training run UUID"""
        constraint = f'run_uuid == "{run_uuid}"'
        
        # Get both queued and historical jobs
        queue_jobs = [job for job in self.get_queue_jobs() if job.run_uuid == run_uuid]
        history_jobs = self.get_history_jobs(constraint=constraint)
        
        return queue_jobs + history_jobs
    
    def create_job_status_table(self, jobs: List[HTCondorJob]) -> Table:
        """Create a rich table showing job status"""
        table = Table(title="HTCondor Job Status")
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Run UUID", style="blue", max_width=8)
        table.add_column("Epoch", justify="right", style="green")
        table.add_column("Resource", style="yellow", max_width=15)
        table.add_column("Runtime", style="white")
        table.add_column("CPU Time", style="bright_blue")
        table.add_column("Memory", justify="right", style="bright_green")
        table.add_column("Machine", style="dim", max_width=20)
        
        for job in sorted(jobs, key=lambda x: x.cluster_id):
            runtime_str = ""
            if job.runtime:
                runtime_str = str(job.runtime).split('.')[0]
                
            cpu_time_str = ""
            if job.cpu_time_used > 0:
                cpu_time_str = f"{job.cpu_time_used:.1f}s"
                
            memory_str = ""
            if job.memory_usage > 0:
                memory_str = f"{job.memory_usage}MB"
                
            table.add_row(
                job.job_id,
                job.status_string,
                job.run_uuid[:8] if job.run_uuid else "",
                str(job.epoch) if job.epoch else "",
                job.resource_name or "",
                runtime_str,
                cpu_time_str,
                memory_str,
                job.machine_name.split('@')[-1] if '@' in job.machine_name else job.machine_name
            )
        
        return table
    
    def create_summary_table(self, jobs: List[HTCondorJob]) -> Table:
        """Create a summary table of job statistics"""
        table = Table(title="Job Summary")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Total CPU Time", justify="right", style="green")
        table.add_column("Avg Memory", justify="right", style="blue")
        
        # Group by status
        status_stats = {}
        for job in jobs:
            status = job.status_string
            if status not in status_stats:
                status_stats[status] = {
                    'count': 0,
                    'total_cpu': 0.0,
                    'total_memory': 0,
                    'memory_count': 0
                }
            
            stats = status_stats[status]
            stats['count'] += 1
            stats['total_cpu'] += job.cpu_time_used
            if job.memory_usage > 0:
                stats['total_memory'] += job.memory_usage
                stats['memory_count'] += 1
        
        for status, stats in status_stats.items():
            avg_memory = stats['total_memory'] / stats['memory_count'] if stats['memory_count'] > 0 else 0
            
            table.add_row(
                status,
                str(stats['count']),
                f"{stats['total_cpu']:.1f}s",
                f"{avg_memory:.0f}MB" if avg_memory > 0 else "N/A"
            )
        
        return table
    
    def monitor_run_uuid(self, run_uuid: str, live: bool = False, refresh_interval: float = 5.0):
        """Monitor jobs for a specific run UUID"""
        if live:
            from rich.live import Live
            from rich.columns import Columns
            
            with Live(refresh_per_second=1/refresh_interval) as live_display:
                try:
                    while True:
                        jobs = self.get_jobs_by_run_uuid(run_uuid)
                        
                        if not jobs:
                            live_display.update(f"[yellow]No jobs found for run UUID: {run_uuid}[/yellow]")
                        else:
                            tables = Columns([
                                self.create_job_status_table(jobs),
                                self.create_summary_table(jobs)
                            ])
                            live_display.update(tables)
                        
                        time.sleep(refresh_interval)
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Monitoring stopped by user[/yellow]")
        else:
            jobs = self.get_jobs_by_run_uuid(run_uuid)
            if jobs:
                self.console.print(self.create_job_status_table(jobs))
                self.console.print(self.create_summary_table(jobs))
            else:
                self.console.print(f"[yellow]No jobs found for run UUID: {run_uuid}[/yellow]")
    
    def monitor_all_jobs(self, live: bool = False, refresh_interval: float = 5.0):
        """Monitor all current jobs"""
        if live:
            from rich.live import Live
            from rich.columns import Columns
            
            with Live(refresh_per_second=1/refresh_interval) as live_display:
                try:
                    while True:
                        jobs = self.get_queue_jobs()
                        
                        if not jobs:
                            live_display.update("[yellow]No jobs in queue[/yellow]")
                        else:
                            tables = Columns([
                                self.create_job_status_table(jobs),
                                self.create_summary_table(jobs)
                            ])
                            live_display.update(tables)
                        
                        time.sleep(refresh_interval)
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Monitoring stopped by user[/yellow]")
        else:
            jobs = self.get_queue_jobs()
            if jobs:
                self.console.print(self.create_job_status_table(jobs))
                self.console.print(self.create_summary_table(jobs))
            else:
                self.console.print("[yellow]No jobs in queue[/yellow]")


def main():
    app = typer.Typer()
    
    @app.command()
    def monitor(
        run_uuid: Optional[str] = typer.Option(None, help="Monitor jobs for specific run UUID"),
        live: bool = typer.Option(True, help="Enable live monitoring"),
        refresh_interval: float = typer.Option(5.0, help="Refresh interval in seconds"),
        history: bool = typer.Option(False, help="Include completed jobs from history")
    ):
        """Monitor HTCondor job status"""
        monitor = HTCondorMonitor()
        
        if run_uuid:
            monitor.monitor_run_uuid(run_uuid, live, refresh_interval)
        else:
            monitor.monitor_all_jobs(live, refresh_interval)
    
    @app.command()
    def summary(
        run_uuid: Optional[str] = typer.Option(None, help="Show summary for specific run UUID"),
        limit: int = typer.Option(50, help="Limit number of historical jobs to check")
    ):
        """Show job summary statistics"""
        monitor = HTCondorMonitor()
        
        if run_uuid:
            jobs = monitor.get_jobs_by_run_uuid(run_uuid)
        else:
            jobs = monitor.get_queue_jobs()
            if limit > 0:
                jobs.extend(monitor.get_history_jobs(limit=limit))
        
        if jobs:
            monitor.console.print(monitor.create_summary_table(jobs))
        else:
            monitor.console.print("[yellow]No jobs found[/yellow]")
    
    app()


if __name__ == "__main__":
    main()