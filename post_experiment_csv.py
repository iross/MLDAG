#!/usr/bin/env python3
"""
Post-experiment CSV creation tool.

This script uses DAG files, DAGMan log files, and metl.log to create comprehensive 
CSV reports of all job execution attempts with proper job-to-DAG mapping.
"""

import csv
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
import glob


@dataclass
class JobAttempt:
    """Information about a single attempt to run a job."""
    cluster_id: int
    job_name: str
    attempt_sequence: int
    dag_source: str
    
    # Timing information
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Status information
    final_status: Optional[str] = None
    
    # Transfer information
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    # GPU information
    gpu_count: int = 0
    gpu_device_name: str = ""
    gpu_memory_mb: int = 0
    
    # Additional event times
    held_time: Optional[datetime] = None
    released_time: Optional[datetime] = None
    evicted_time: Optional[datetime] = None
    aborted_time: Optional[datetime] = None


class SimpleCSVGenerator:
    """Generate CSV reports using DAG files, DAGMan output files, and metl.log."""
    
    def __init__(self, dag_files: Optional[List[str]] = None):
        """Initialize the generator.
        
        Args:
            dag_files: List of DAG files. If None, auto-detect *.dag files.
        """
        if dag_files is None:
            dag_files = glob.glob("*.dag")
            
        self.dag_files = [f for f in dag_files if Path(f).exists()]
        if not self.dag_files:
            raise FileNotFoundError("No DAG files found")
        
        # Find corresponding .dag.dagman.out files (contain cluster-to-job mappings)
        self.dagman_out_files = []
        for dag_file in self.dag_files:
            dagman_out = f"{dag_file}.dagman.out"
            if Path(dagman_out).exists():
                self.dagman_out_files.append(dagman_out)
        
        print(f"Found DAG files: {self.dag_files}")
        print(f"Found DAGMan output files: {self.dagman_out_files}")
    
    def parse_dag_files(self) -> Dict[str, str]:
        """Parse DAG files to extract job name to DAG source mapping.
        
        Returns:
            Dictionary mapping job names to DAG source names
        """
        job_to_dag = {}
        
        for dag_file in self.dag_files:
            dag_source = Path(dag_file).stem
            
            with open(dag_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Look for JOB lines: JOB job_name submit_file
                    if line.startswith('JOB '):
                        parts = line.split()
                        if len(parts) >= 2:
                            job_name = parts[1]
                            job_to_dag[job_name] = dag_source
        
        print(f"Found {len(job_to_dag)} jobs across {len(self.dag_files)} DAG files")
        return job_to_dag
    
    def parse_dagman_out_files(self) -> Dict[str, Dict[int, str]]:
        """Parse DAGMan output files to create node-to-jobid mapping for each DAG.
        
        Returns:
            Dictionary mapping dag_source to {cluster_id: job_name} mappings
        """
        dag_mappings = {}
        
        for dagman_out_file in self.dagman_out_files:
            # Extract DAG source from filename (remove .dag.dagman.out)
            dag_source = Path(dagman_out_file).name.replace('.dag.dagman.out', '')
            
            print(f"Parsing {dagman_out_file} for cluster mappings...")
            
            dag_mappings[dag_source] = {}
            
            with open(dagman_out_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for cluster assignment lines in DAGMan output
                # Format: "assigned HTCondor ID (cluster_id.proc.subproc)"
                cluster_match = re.search(r'assigned HTCondor ID \((\d+)\.\d+\.\d+\)', line)
                if cluster_match:
                    cluster_id = int(cluster_match.group(1))
                    
                    # Look ahead for the corresponding DAG node submission
                    # Find the IMMEDIATE next "Submitting HTCondor Node" line within a reasonable range
                    for j in range(i + 1, min(i + 15, len(lines))):
                        next_line = lines[j].strip()
                        
                        # Look for "Submitting HTCondor Node <job_name>"
                        node_match = re.search(r'Submitting HTCondor Node (.+)', next_line)
                        if node_match:
                            job_name = node_match.group(1)
                            dag_mappings[dag_source][cluster_id] = job_name
                            break
                
                # Also look for ULOG_SUBMIT events which have a different format
                # Format: "Event: ULOG_SUBMIT for HTCondor Node <job_name> (cluster_id.proc.subproc)"
                elif 'Event: ULOG_SUBMIT for HTCondor Node' in line:
                    submit_match = re.search(r'Event: ULOG_SUBMIT for HTCondor Node (.+?) \((\d+)\.\d+\.\d+\)', line)
                    if submit_match:
                        job_name = submit_match.group(1)
                        cluster_id = int(submit_match.group(2))
                        # ULOG_SUBMIT events are more authoritative, so overwrite any existing mapping
                        dag_mappings[dag_source][cluster_id] = job_name
                
                i += 1
            
            print(f"  Found {len(dag_mappings[dag_source])} cluster mappings for {dag_source}")
        
        total_mappings = sum(len(mapping) for mapping in dag_mappings.values())
        print(f"Total cluster mappings from DAGMan output files: {total_mappings}")
        return dag_mappings
    
    def parse_rescue_events(self) -> Dict[str, Set[int]]:
        """Parse DAGMan output files to find clusters that were rescued.
        
        Returns:
            Dictionary mapping dag_source to set of rescued cluster_ids
        """
        rescued_clusters = {}
        
        for dagman_out_file in self.dagman_out_files:
            # Extract DAG source from filename (remove .dag.dagman.out)
            dag_source = Path(dagman_out_file).name.replace('.dag.dagman.out', '')
            
            rescued_clusters[dag_source] = set()
            
            with open(dagman_out_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Look for "ERROR: DAGMan lost track of node <job_name> (ClusterId=<cluster_id>)"
                    rescue_match = re.search(r'ERROR: DAGMan lost track of node .+ \(ClusterId=(\d+)\)', line)
                    if rescue_match:
                        cluster_id = int(rescue_match.group(1))
                        rescued_clusters[dag_source].add(cluster_id)
                    
                    # Also look for "Event: ULOG_JOB_TERMINATED for unknown Node (cluster_id.proc.subproc)"
                    unknown_match = re.search(r'Event: ULOG_JOB_TERMINATED for unknown Node \((\d+)\.\d+\.\d+\)', line)
                    if unknown_match:
                        cluster_id = int(unknown_match.group(1))
                        rescued_clusters[dag_source].add(cluster_id)
            
            if rescued_clusters[dag_source]:
                print(f"  Found {len(rescued_clusters[dag_source])} rescued clusters for {dag_source}")
        
        return rescued_clusters
    
    def find_job_and_dag_for_cluster(self, cluster_id: int, dag_mappings: Dict[str, Dict[int, str]]) -> Tuple[str, str]:
        """Find the job name and DAG source for a given cluster ID.
        
        Args:
            cluster_id: HTCondor cluster ID to look up
            dag_mappings: DAG mappings from parse_dagman_out_files()
            
        Returns:
            Tuple of (job_name, dag_source)
        """
        # Search through all DAG mappings to find this cluster
        for dag_source, cluster_map in dag_mappings.items():
            if cluster_id in cluster_map:
                return cluster_map[cluster_id], dag_source
        
        # Not found in any DAG mapping
        return None, None
    
    def parse_metl_log(self) -> Dict[int, List[Dict[str, Any]]]:
        """Parse metl.log to extract all execution attempts by cluster ID.
        
        Returns:
            Dictionary mapping cluster_id to list of attempt data
        """
        metl_log_path = Path("metl.log")
        if not metl_log_path.exists():
            raise FileNotFoundError("metl.log not found")
        
        cluster_attempts = {}
        
        with open(metl_log_path, 'r') as f:
            lines = f.readlines()
        
        # First pass: find all execution attempts (001 events)
        execution_events = []
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for execution events
            exec_match = re.match(r'^001 \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Job executing', line)
            if exec_match:
                cluster_id = int(exec_match.group(1))
                timestamp_str = exec_match.group(2)
                
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
                
                execution_events.append({
                    'cluster_id': cluster_id,
                    'start_time': timestamp,
                    'line_index': i
                })
        
        print(f"Found {len(execution_events)} execution events")
        
        # Second pass: collect all events by cluster
        all_events = {}
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for any relevant event
            event_match = re.match(r'^(000|001|004|005|009|012|013) \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if event_match:
                event_code = event_match.group(1)
                cluster_id = int(event_match.group(2))
                timestamp_str = event_match.group(3)
                
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
                
                if cluster_id not in all_events:
                    all_events[cluster_id] = []
                
                all_events[cluster_id].append({
                    'event_code': event_code,
                    'timestamp': timestamp,
                    'line': line,
                    'line_index': i
                })
        
        # Group execution events by cluster and create attempt records
        for exec_event in execution_events:
            cluster_id = exec_event['cluster_id']
            start_time = exec_event['start_time']
            
            if cluster_id not in cluster_attempts:
                cluster_attempts[cluster_id] = []
            
            # Find the attempt sequence number (how many executions for this cluster so far)
            attempt_sequence = len([a for a in cluster_attempts[cluster_id]]) + 1
            
            attempt = {
                'cluster_id': cluster_id,
                'attempt_sequence': attempt_sequence,
                'start_time': start_time,
                'line_index': exec_event['line_index']
            }
            
            # Find events for this specific attempt
            cluster_events = all_events.get(cluster_id, [])
            
            # Find most recent submission before this execution
            submission_events = [e for e in cluster_events 
                               if e['event_code'] == '000' and e['timestamp'] <= start_time]
            if submission_events:
                latest_submission = max(submission_events, key=lambda x: x['timestamp'])
                attempt['submit_time'] = latest_submission['timestamp']
            
            # Find the next execution time for this cluster (to bound this attempt)
            next_exec_times = [e['start_time'] for e in execution_events 
                             if e['cluster_id'] == cluster_id and e['start_time'] > start_time]
            next_exec_time = min(next_exec_times) if next_exec_times else None
            
            # Process events for this attempt
            attempt_events = [e for e in cluster_events 
                            if e['timestamp'] >= start_time and 
                            (next_exec_time is None or e['timestamp'] < next_exec_time)]
            
            for event in attempt_events:
                if event['event_code'] == '005':  # Termination
                    attempt['end_time'] = event['timestamp']
                    
                    # Parse exit code to determine final status
                    exit_code = self._parse_exit_code(lines, event['line_index'])
                    if exit_code == 85:
                        attempt['final_status'] = 'checkpointed'
                    elif exit_code == 1:
                        attempt['final_status'] = 'failed'
                    else:
                        attempt['final_status'] = 'completed'
                    
                    # Look for transfer data
                    self._parse_transfer_data(lines, event['line_index'], attempt)
                    break
                elif event['event_code'] == '004':  # Evicted
                    attempt['evicted_time'] = event['timestamp']
                    attempt['end_time'] = event['timestamp']  # Use eviction time as end time for duration calculation
                    attempt['final_status'] = 'evicted'
                elif event['event_code'] == '009':  # Aborted
                    attempt['aborted_time'] = event['timestamp']
                    attempt['end_time'] = event['timestamp']  # Use abort time as end time for duration calculation
                    attempt['final_status'] = 'aborted'
                elif event['event_code'] == '012':  # Held
                    attempt['held_time'] = event['timestamp']
                    if not attempt.get('final_status'):
                        attempt['final_status'] = 'held'
                elif event['event_code'] == '013':  # Released
                    attempt['released_time'] = event['timestamp']
                    if attempt.get('final_status') == 'held':
                        attempt['final_status'] = 'released'
            
            # Parse GPU info from execution line
            self._parse_gpu_info(lines, exec_event['line_index'], attempt)
            
            cluster_attempts[cluster_id].append(attempt)
        
        return cluster_attempts
    
    def _parse_exit_code(self, lines: List[str], line_index: int) -> Optional[int]:
        """Parse exit code from lines following a termination event."""
        for i in range(line_index + 1, min(line_index + 10, len(lines))):
            line = lines[i].strip()
            
            # Look for "(1) Normal termination (return value X)"
            exit_match = re.search(r'Normal termination \(return value (\d+)\)', line)
            if exit_match:
                return int(exit_match.group(1))
        
        return None
    
    def _parse_transfer_data(self, lines: List[str], line_index: int, attempt: Dict[str, Any]):
        """Parse transfer data from lines following a termination event."""
        for i in range(line_index + 1, min(line_index + 20, len(lines))):
            line = lines[i].strip()
            
            bytes_sent_match = re.search(r'TotalBytesSent = (\d+)', line)
            if bytes_sent_match:
                attempt['total_bytes_sent'] = int(bytes_sent_match.group(1))
                
            bytes_received_match = re.search(r'TotalBytesReceived = (\d+)', line)
            if bytes_received_match:
                attempt['total_bytes_received'] = int(bytes_received_match.group(1))
    
    def _parse_gpu_info(self, lines: List[str], line_index: int, attempt: Dict[str, Any]):
        """Parse GPU information from lines following an execution event."""
        for i in range(line_index + 1, min(line_index + 10, len(lines))):
            line = lines[i].strip()
            
            if 'DetectedGpus = ' in line:
                gpu_match = re.search(r'DetectedGpus = "(\d+)"', line)
                if gpu_match:
                    attempt['gpu_count'] = int(gpu_match.group(1))
            elif 'CUDADeviceName = ' in line:
                device_match = re.search(r'CUDADeviceName = "([^"]*)"', line)
                if device_match:
                    attempt['gpu_device_name'] = device_match.group(1)
            elif 'CUDAGlobalMemoryMb = ' in line:
                memory_match = re.search(r'CUDAGlobalMemoryMb = (\d+)', line)
                if memory_match:
                    attempt['gpu_memory_mb'] = int(memory_match.group(1))
    
    def extract_job_name_from_metl(self, cluster_id: int) -> Optional[str]:
        """Try to extract job name from DAG Node info in metl.log submission events."""
        metl_log_path = Path("metl.log")
        if not metl_log_path.exists():
            return None
        
        with open(metl_log_path, 'r') as f:
            for line in f:
                if f'000 ({cluster_id}.' in line and 'DAG Node:' in line:
                    dag_match = re.search(r'DAG Node: (\S+)', line)
                    if dag_match:
                        return dag_match.group(1)
        
        return None
    
    def generate_attempts(self) -> List[JobAttempt]:
        """Generate all JobAttempt objects."""
        job_to_dag = self.parse_dag_files()
        dag_mappings = self.parse_dagman_out_files()
        rescued_clusters = self.parse_rescue_events()
        cluster_attempts = self.parse_metl_log()
        
        attempts = []
        
        for cluster_id, attempt_list in cluster_attempts.items():
            for attempt_data in attempt_list:
                # Try to find job name and DAG source for this cluster
                job_name = None
                dag_source = "unknown"
                
                # First priority: DAGMan output file mappings (most authoritative)
                job_name, dag_source = self.find_job_and_dag_for_cluster(cluster_id, dag_mappings)
                
                if not job_name:
                    # Fallback: try to extract from metl.log DAG Node info
                    job_name = self.extract_job_name_from_metl(cluster_id)
                    
                    if job_name and job_name in job_to_dag:
                        dag_source = job_to_dag[job_name]
                    elif job_name:
                        # Job name found in metl.log but not in DAG files
                        dag_source = "not_in_dag"
                    else:
                        # No job name found, create generic name
                        job_name = f"unmapped_cluster_{cluster_id}"
                        dag_source = "unmapped"
                
                attempt = JobAttempt(
                    cluster_id=cluster_id,
                    job_name=job_name,
                    attempt_sequence=attempt_data['attempt_sequence'],
                    dag_source=dag_source
                )
                
                # Fill in timing data
                attempt.submit_time = attempt_data.get('submit_time')
                attempt.start_time = attempt_data.get('start_time')
                attempt.end_time = attempt_data.get('end_time')
                attempt.final_status = attempt_data.get('final_status')
                
                # Check if this cluster was rescued by DAGMan
                for dag_src, rescued_set in rescued_clusters.items():
                    if cluster_id in rescued_set:
                        attempt.final_status = 'RESCUED'
                        break
                
                # Fill in transfer data
                attempt.total_bytes_sent = attempt_data.get('total_bytes_sent', 0)
                attempt.total_bytes_received = attempt_data.get('total_bytes_received', 0)
                
                # Fill in GPU data
                attempt.gpu_count = attempt_data.get('gpu_count', 0)
                attempt.gpu_device_name = attempt_data.get('gpu_device_name', '')
                attempt.gpu_memory_mb = attempt_data.get('gpu_memory_mb', 0)
                
                # Fill in event times
                attempt.held_time = attempt_data.get('held_time')
                attempt.released_time = attempt_data.get('released_time')
                attempt.evicted_time = attempt_data.get('evicted_time')
                attempt.aborted_time = attempt_data.get('aborted_time')
                
                attempts.append(attempt)
        
        # Sort by job name, cluster ID, and attempt sequence
        attempts.sort(key=lambda x: (x.job_name, x.cluster_id, x.attempt_sequence))
        
        return attempts
    
    def format_datetime(self, dt: Optional[datetime]) -> str:
        """Format datetime for CSV output."""
        return dt.isoformat() if dt else ""
    
    def calculate_duration(self, start: Optional[datetime], end: Optional[datetime]) -> Tuple[str, str]:
        """Calculate duration between two timestamps."""
        if not start or not end:
            return "", ""
        
        duration_delta = end - start
        if duration_delta.total_seconds() < 0:
            return "0", "negative_time"
        
        duration_seconds = str(int(duration_delta.total_seconds()))
        duration_human = str(duration_delta).split('.')[0]
        
        return duration_seconds, duration_human
    
    def export_csv(self, output_file: str):
        """Export all attempts to CSV."""
        attempts = self.generate_attempts()
        
        print(f"Generated {len(attempts)} attempts for CSV export")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'Job Name', 'DAG Source', 'HTCondor Cluster ID', 'Attempt Sequence',
            'Final Status', 'Submit Time', 'Start Time', 'End Time',
            'Execution Duration (seconds)', 'Execution Duration (human)',
            'Total Duration (seconds)', 'Total Duration (human)',
            'Total Bytes Sent', 'Total Bytes Received',
            'Number of GPUs', 'GPU Device Name', 'GPU Memory MB',
            'Held Time', 'Released Time', 'Evicted Time', 'Aborted Time'
        ]
        
        # Group attempts by job name for global attempt numbering
        job_attempts = {}
        for attempt in attempts:
            if attempt.job_name not in job_attempts:
                job_attempts[attempt.job_name] = []
            job_attempts[attempt.job_name].append(attempt)
        
        # Sort within each job
        for job_name in job_attempts:
            job_attempts[job_name].sort(key=lambda x: (x.cluster_id, x.attempt_sequence))
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for job_name, job_attempt_list in sorted(job_attempts.items()):
                for global_attempt_num, attempt in enumerate(job_attempt_list, 1):
                    exec_duration_sec, exec_duration_human = self.calculate_duration(
                        attempt.start_time, attempt.end_time)
                    total_duration_sec, total_duration_human = self.calculate_duration(
                        attempt.submit_time, attempt.end_time)
                    
                    writer.writerow({
                        'Job Name': attempt.job_name,
                        'DAG Source': attempt.dag_source,
                        'HTCondor Cluster ID': attempt.cluster_id,
                        'Attempt Sequence': attempt.attempt_sequence,
                        'Final Status': attempt.final_status or "",
                        'Submit Time': self.format_datetime(attempt.submit_time),
                        'Start Time': self.format_datetime(attempt.start_time),
                        'End Time': self.format_datetime(attempt.end_time),
                        'Execution Duration (seconds)': exec_duration_sec,
                        'Execution Duration (human)': exec_duration_human,
                        'Total Duration (seconds)': total_duration_sec,
                        'Total Duration (human)': total_duration_human,
                        'Total Bytes Sent': attempt.total_bytes_sent,
                        'Total Bytes Received': attempt.total_bytes_received,
                        'Number of GPUs': attempt.gpu_count,
                        'GPU Device Name': attempt.gpu_device_name,
                        'GPU Memory MB': attempt.gpu_memory_mb,
                        'Held Time': self.format_datetime(attempt.held_time),
                        'Released Time': self.format_datetime(attempt.released_time),
                        'Evicted Time': self.format_datetime(attempt.evicted_time),
                        'Aborted Time': self.format_datetime(attempt.aborted_time)
                    })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive CSV reports using DAG files, DAGMan output files, and metl.log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CSV with all attempts (auto-detect *.dag files)
  python simple_post_experiment_csv.py all_attempts.csv
  
  # Use specific DAG files
  python simple_post_experiment_csv.py output.csv --dag-files file1.dag file2.dag
        """
    )
    
    parser.add_argument("output_file", help="Output CSV file path")
    parser.add_argument("--dag-files", nargs="+", help="Specific DAG files to use")
    
    args = parser.parse_args()
    
    generator = SimpleCSVGenerator(args.dag_files)
    generator.export_csv(args.output_file)
    print(f"CSV exported to: {args.output_file}")


if __name__ == "__main__":
    main()