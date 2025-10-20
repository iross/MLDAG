#!/usr/bin/env python3
"""
Simple one-time DAG status report script.
Uses the existing DAGStatusMonitor to generate a status report.
"""

import sys
from pathlib import Path
from dagman_monitor import DAGStatusMonitor

def generate_status_report(dag_file: str, verbose: bool = False):
    """Generate a one-time status report for a DAG"""
    dag_path = Path(dag_file)
    
    if not dag_path.exists():
        print(f"Error: DAG file '{dag_file}' not found")
        return 1
    
    print(f"DAG Status Report for: {dag_file}")
    print("=" * 50)
    
    # Initialize monitor and process logs
    monitor = DAGStatusMonitor(dag_file)
    
    # Process both log files if they exist
    dagman_log = Path(dag_file).with_suffix('.dag.dagman.log')
    nodes_log = Path(dag_file).with_suffix('.dag.nodes.log')
    
    print(f"Checking log files:")
    print(f"  DAGMan log: {dagman_log} ({'exists' if dagman_log.exists() else 'missing'})")
    print(f"  Nodes log: {nodes_log} ({'exists' if nodes_log.exists() else 'missing'})")
    print()
    
    # Process the logs
    if nodes_log.exists():
        # Use nodes.log as the primary source
        monitor.dagman_log = nodes_log
    
    # Reset position to read entire log file
    monitor.last_log_position = 0
    
    # Fix the regex patterns for the actual log format
    import re
    from datetime import datetime
    from dagman_monitor import JobStatus
    
    # Updated patterns for this log format
    patterns = {
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
    
    def parse_timestamp(timestamp_str: str) -> datetime:
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return datetime.now()
    
    # Process logs manually with correct patterns
    print("Processing log entries...")
    if monitor.dagman_log.exists():
        with open(monitor.dagman_log, 'r') as f:
            lines = f.readlines()
            print(f"Found {len(lines)} lines in log file")
            
            jobs = {}
            cluster_to_dagnode = {}  # Map cluster IDs to DAG node names
            
            # First pass: build cluster ID to DAG node mapping
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line == '...':
                    i += 1
                    continue
                
                # Look for job events followed by DAG node info
                for event_type, pattern in patterns.items():
                    if event_type == 'dag_node':
                        continue
                        
                    match = pattern.search(line)
                    if match:
                        cluster_id = int(match.groups()[0])
                        # Look ahead for DAG Node line (should be next line)
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            dag_match = patterns['dag_node'].search(next_line)
                            if dag_match and cluster_id not in cluster_to_dagnode:
                                dag_node_name = dag_match.group(1)
                                cluster_to_dagnode[cluster_id] = dag_node_name
                        break
                
                i += 1
            
            print(f"Found {len(cluster_to_dagnode)} cluster mappings")
            
            # Second pass: process all events with proper names
            for line in lines:
                line = line.strip()
                if not line or line == '...' or patterns['dag_node'].search(line):
                    continue
                
                # Parse HTCondor log events
                for event_type, pattern in patterns.items():
                    if event_type == 'dag_node':
                        continue
                        
                    match = pattern.search(line)
                    if match:
                        groups = match.groups()
                        cluster_id = int(groups[0])
                        timestamp = parse_timestamp(groups[1])
                        
                        # Get job name from cluster mapping or use cluster ID
                        job_name = cluster_to_dagnode.get(cluster_id, f'cluster_{cluster_id}')
                        
                        if job_name not in jobs:
                            from dagman_monitor import JobInfo
                            
                            # Parse job variables to get metadata if we have a DAG node name
                            if job_name != f'cluster_{cluster_id}':
                                job_vars = monitor.parse_job_vars(job_name)
                                jobs[job_name] = JobInfo(
                                    name=job_name,
                                    run_uuid=job_vars.get('run_uuid'),
                                    epoch=int(job_vars.get('epoch', 0)) if job_vars.get('epoch') else None,
                                    resource_name=job_vars.get('ResourceName')
                                )
                            else:
                                jobs[job_name] = JobInfo(name=job_name)
                        
                        job = jobs[job_name]
                        job.cluster_id = cluster_id
                        
                        if event_type == 'job_submitted':
                            job.status = JobStatus.IDLE
                            job.submit_time = timestamp
                        elif event_type == 'job_executing':
                            job.status = JobStatus.RUNNING
                            job.start_time = timestamp
                        elif event_type == 'job_terminated':
                            job.status = JobStatus.COMPLETED
                            job.end_time = timestamp
                        elif event_type == 'job_held':
                            job.status = JobStatus.HELD
                        
                        job.last_event_time = timestamp
                        break
            
            # Update monitor with parsed jobs
            monitor.jobs = jobs
    
    print(f"Found {len(monitor.jobs)} jobs after processing")
    print()
    
    # Display status tables
    from rich.console import Console
    console = Console()
    
    # Create custom job status table
    from rich.table import Table
    table = Table(title="DAG Job Status")
    table.add_column("Run", justify="right", style="cyan", no_wrap=True)
    table.add_column("Epoch", justify="right", style="green")
    table.add_column("Run UUID", style="blue", max_width=8)
    if verbose:
        table.add_column("HTCondor Job ID", justify="right", style="dim white", max_width=12)
    table.add_column("Targeted Resource", style="yellow", max_width=15)
    table.add_column("Duration", style="white")
    table.add_column("Status", style="magenta")
    
    # Filter out annex_helper and sort by natural number
    import re
    
    def natural_sort_key(job_name):
        """Extract numbers from job name for natural sorting"""
        # Extract run number and epoch number from job name like "run10-train_epoch5"
        match = re.match(r'run(\d+)-train_epoch(\d+)', job_name)
        if match:
            run_num = int(match.group(1))
            epoch_num = int(match.group(2))
            return (run_num, epoch_num)
        else:
            # Fallback for any other format
            return (999, 999)
    
    filtered_jobs = [job for job in monitor.jobs.values() if job.name != "annex_helper"]
    sorted_jobs = sorted(filtered_jobs, key=lambda x: natural_sort_key(x.name))
    
    seen_runs = set()  # Track which run numbers we've already displayed
    
    for job in sorted_jobs:
        duration = ""
        if job.start_time and job.end_time:
            duration = str(job.end_time - job.start_time).split('.')[0]
        elif job.start_time:
            duration = str(datetime.now() - job.start_time).split('.')[0]
        
        # Extract run number from job name
        run_match = re.match(r'run(\d+)-train_epoch(\d+)', job.name)
        run_number = run_match.group(1) if run_match else "?"
        
        # Only show run number if we haven't seen it before
        display_run = run_number if run_number not in seen_runs else ""
        seen_runs.add(run_number)
        
        # Color-code status
        if job.status.value == "COMPLETED":
            status_style = "[green]COMPLETED[/green]"
        elif job.status.value == "RUNNING":
            status_style = "[blue]RUNNING[/blue]"
        elif job.status.value == "IDLE":
            status_style = "[yellow]IDLE[/yellow]"
        elif job.status.value == "HELD":
            status_style = "[red]HELD[/red]"
        else:
            status_style = job.status.value
            
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
    
    console.print(table)
    
    # Training run progress table (if applicable)
    if monitor.training_runs:
        print()
        training_table = monitor.create_training_run_table()
        console.print(training_table)
    
    # Create summary table
    print()
    from rich.table import Table
    summary_table = Table(title="Training Summary")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Count", justify="right", style="green")
    summary_table.add_column("Total Time", style="yellow")
    
    if filtered_jobs:
        from dagman_monitor import JobStatus
        from datetime import timedelta
        
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
        summary_table.add_row("Epochs Completed", str(len(completed_jobs)), str(completed_time).split('.')[0])
        summary_table.add_row("Epochs Running", str(len(running_jobs)), str(running_time).split('.')[0])
        summary_table.add_row("Epochs Idle", str(len(idle_jobs)), "")
        summary_table.add_row("Epochs Failed", str(len(failed_jobs)), "")
        summary_table.add_row("Epochs Held", str(len(held_jobs)), "")
        
        # Total compute time (completed + ongoing)
        total_compute_time = completed_time + running_time
        summary_table.add_row("Total Compute Time", "", str(total_compute_time).split('.')[0])
        
        console.print(summary_table)
    else:
        print("No training jobs found in DAG logs")
    
    return 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a one-time DAG status report")
    parser.add_argument("dag_file", help="Path to the DAG file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Include HTCondor job IDs in the output")
    
    args = parser.parse_args()
    
    return generate_status_report(args.dag_file, args.verbose)

if __name__ == "__main__":
    sys.exit(main())