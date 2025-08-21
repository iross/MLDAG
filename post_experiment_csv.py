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
    gpu_capability: str = ""
    gpu_driver_version: str = ""
    
    # Additional event times
    held_time: Optional[datetime] = None
    released_time: Optional[datetime] = None
    evicted_time: Optional[datetime] = None
    aborted_time: Optional[datetime] = None
    
    # Resource information
    targeted_resource: str = ""


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
        
        # Find corresponding .dag.nodes.log files (contain detailed GPU information)
        self.nodes_log_files = []
        for dag_file in self.dag_files:
            nodes_log = f"{dag_file}.nodes.log"
            if Path(nodes_log).exists():
                self.nodes_log_files.append(nodes_log)
        
        print(f"Found DAG files: {self.dag_files}")
        print(f"Found DAGMan output files: {self.dagman_out_files}")
        print(f"Found DAG nodes log files: {self.nodes_log_files}")
    
    def parse_dag_files(self) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
        """Parse DAG files to extract job name to DAG source mapping and resource names.
        
        Returns:
            Tuple of (job_to_dag_mapping, (job_name, dag_source)_to_resource_mapping)
        """
        job_to_dag = {}
        job_dag_to_resource = {}
        
        for dag_file in self.dag_files:
            dag_source = Path(dag_file).stem
            
            with open(dag_file, 'r') as f:
                content = f.read()
                
                # Parse JOB lines: JOB job_name submit_file
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('JOB '):
                        parts = line.split()
                        if len(parts) >= 2:
                            job_name = parts[1]
                            job_to_dag[job_name] = dag_source
                
                # Parse VARS lines to extract ResourceName for this specific DAG
                vars_pattern = r'VARS\s+(\S+)\s+(.+)'
                for match in re.finditer(vars_pattern, content, re.MULTILINE):
                    job_name = match.group(1)
                    vars_content = match.group(2)
                    
                    # Look for ResourceName="value" in the VARS line
                    resource_match = re.search(r'ResourceName="([^"]*)"', vars_content)
                    if resource_match:
                        # Store with both job_name and dag_source as key
                        job_dag_to_resource[(job_name, dag_source)] = resource_match.group(1)
        
        print(f"Found {len(job_to_dag)} jobs across {len(self.dag_files)} DAG files")
        print(f"Found {len(job_dag_to_resource)} job-DAG resource mappings")
        return job_to_dag, job_dag_to_resource
    
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
    
    def parse_nodes_log_files(self) -> Dict[int, Dict[str, Any]]:
        """Parse DAG nodes log files to extract detailed GPU information.
        
        Returns:
            Dictionary mapping cluster_id to GPU details dict
        """
        gpu_details = {}
        
        for nodes_log_file in self.nodes_log_files:
            print(f"Parsing {nodes_log_file} for GPU details...")
            
            with open(nodes_log_file, 'r') as f:
                lines = f.readlines()
            
            current_cluster = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for execution events to get cluster ID
                exec_match = re.match(r'^001 \((\d+)\.\d+\.\d+\)', line)
                if exec_match:
                    current_cluster = int(exec_match.group(1))
                    continue
                
                # Look for GPU detail lines (only if we have a current cluster)
                if current_cluster and line.startswith('GPUs_GPU_') and ' = [' in line:
                    # Parse GPU details from the line
                    # Format: GPUs_GPU_xxxxx = [ Id = "GPU-xxxxx"; ... DeviceName = "NVIDIA A40"; ... ]
                    
                    details = self._parse_gpu_details_line(line)
                    if details:
                        if current_cluster not in gpu_details:
                            gpu_details[current_cluster] = {
                                'devices': [],
                                'device_name': '',
                                'capability': '',
                                'memory_mb': 0,
                                'driver_version': ''
                            }
                        
                        gpu_details[current_cluster]['devices'].append(details)
                        
                        # Update aggregate info (use first GPU's info, or combine)
                        if not gpu_details[current_cluster]['device_name']:
                            gpu_details[current_cluster]['device_name'] = details.get('DeviceName', '')
                            gpu_details[current_cluster]['capability'] = details.get('Capability', '')
                            gpu_details[current_cluster]['memory_mb'] = details.get('GlobalMemoryMb', 0)
                            gpu_details[current_cluster]['driver_version'] = details.get('DriverVersion', '')
                        elif len(gpu_details[current_cluster]['devices']) > 1:
                            # Multiple GPUs - just use device name without count prefix
                            first_device = gpu_details[current_cluster]['devices'][0].get('DeviceName', 'Unknown')
                            gpu_details[current_cluster]['device_name'] = first_device
                            # For memory, sum all GPUs
                            total_memory = sum(d.get('GlobalMemoryMb', 0) for d in gpu_details[current_cluster]['devices'])
                            gpu_details[current_cluster]['memory_mb'] = total_memory
            
            print(f"  Found GPU details for {len([c for c in gpu_details])} clusters")
        
        return gpu_details
    
    def _parse_gpu_details_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a GPU details line from nodes log file."""
        try:
            # Extract the content between [ and ]
            bracket_match = re.search(r'\[(.*)\]', line)
            if not bracket_match:
                return None
            
            content = bracket_match.group(1)
            
            # Parse key-value pairs separated by semicolons
            details = {}
            pairs = [pair.strip() for pair in content.split(';')]
            
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    
                    # Convert numeric values
                    if key in ['GlobalMemoryMb', 'ClockMhz']:
                        try:
                            details[key] = int(float(value))
                        except ValueError:
                            details[key] = value
                    elif key in ['Capability', 'DriverVersion']:
                        try:
                            details[key] = str(float(value))
                        except ValueError:
                            details[key] = value
                    else:
                        details[key] = value
            
            return details
        except Exception:
            return None
    
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
                'line_index': exec_event['line_index'],
                'gpu_capability': '',
                'gpu_driver_version': '',
                'actual_resource': ''
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
            
            # Look for "Total Bytes Sent By Job" format
            bytes_sent_match = re.search(r'(\d+)\s+-\s+Total Bytes Sent By Job', line)
            if bytes_sent_match:
                attempt['total_bytes_sent'] = int(bytes_sent_match.group(1))
                
            # Look for "Total Bytes Received By Job" format  
            bytes_received_match = re.search(r'(\d+)\s+-\s+Total Bytes Received By Job', line)
            if bytes_received_match:
                attempt['total_bytes_received'] = int(bytes_received_match.group(1))
                
            # Also check for alternative formats (legacy support)
            legacy_sent_match = re.search(r'TotalBytesSent = (\d+)', line)
            if legacy_sent_match:
                attempt['total_bytes_sent'] = int(legacy_sent_match.group(1))
                
            legacy_received_match = re.search(r'TotalBytesReceived = (\d+)', line)
            if legacy_received_match:
                attempt['total_bytes_received'] = int(legacy_received_match.group(1))
    
    def _parse_gpu_info(self, lines: List[str], line_index: int, attempt: Dict[str, Any]):
        """Parse GPU information and resource info from lines following an execution event."""
        gpu_details = []
        
        for i in range(line_index + 1, min(line_index + 20, len(lines))):
            line = lines[i].strip()
            
            # Look for AvailableGPUs = { GPUs_GPU_xxxxx, GPUs_GPU_yyyyy }
            if 'AvailableGPUs = ' in line:
                gpu_match = re.search(r'AvailableGPUs = \{ ([^}]+) \}', line)
                if gpu_match:
                    gpu_list_str = gpu_match.group(1)
                    # Split by comma to count GPUs and get device names
                    gpu_devices = [gpu.strip() for gpu in gpu_list_str.split(',')]
                    attempt['gpu_count'] = len(gpu_devices)
                    # Use the first GPU ID as device name (will be overridden by detailed parsing if available)
                    attempt['gpu_device_name'] = gpu_devices[0]
            
            # Look for detailed GPU specifications: GPUs_GPU_xxxxx = [ ... ]
            elif line.startswith('GPUs_GPU_') and ' = [' in line:
                details = self._parse_gpu_details_line(line)
                if details:
                    gpu_details.append(details)
            
            # Look for GLIDEIN_ResourceName (indicates OSPool execution)
            elif 'GLIDEIN_ResourceName = ' in line:
                resource_match = re.search(r'GLIDEIN_ResourceName = "([^"]*)"', line)
                if resource_match:
                    # If there's a GLIDEIN_ResourceName, use it (indicates OSPool)
                    attempt['actual_resource'] = resource_match.group(1)
            
            
            # Keep legacy patterns in case they exist elsewhere
            elif 'DetectedGpus = ' in line:
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
        
        # Process detailed GPU information if found
        if gpu_details:
            attempt['gpu_count'] = len(gpu_details)
            
            if len(gpu_details) == 1:
                # Single GPU
                details = gpu_details[0]
                attempt['gpu_device_name'] = details.get('DeviceName', '')
                attempt['gpu_memory_mb'] = details.get('GlobalMemoryMb', 0)
                attempt['gpu_capability'] = details.get('Capability', '')
                attempt['gpu_driver_version'] = details.get('DriverVersion', '')
            else:
                # Multiple GPUs - just use the device name without count prefix
                first_gpu = gpu_details[0]
                attempt['gpu_device_name'] = first_gpu.get('DeviceName', 'Unknown')
                attempt['gpu_memory_mb'] = sum(d.get('GlobalMemoryMb', 0) for d in gpu_details)
                attempt['gpu_capability'] = first_gpu.get('Capability', '')
                attempt['gpu_driver_version'] = first_gpu.get('DriverVersion', '')
    
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
        job_to_dag, job_dag_to_resource = self.parse_dag_files()
        dag_mappings = self.parse_dagman_out_files()
        rescued_clusters = self.parse_rescue_events()
        gpu_details = self.parse_nodes_log_files()
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
                
                # Fill in GPU data (from metl.log parsing)
                attempt.gpu_count = attempt_data.get('gpu_count', 0)
                attempt.gpu_device_name = attempt_data.get('gpu_device_name', '')
                attempt.gpu_memory_mb = attempt_data.get('gpu_memory_mb', 0)
                attempt.gpu_capability = attempt_data.get('gpu_capability', '')
                attempt.gpu_driver_version = attempt_data.get('gpu_driver_version', '')
                
                # Override with detailed GPU info from nodes log if available
                if cluster_id in gpu_details:
                    details = gpu_details[cluster_id]
                    attempt.gpu_device_name = details.get('device_name', attempt.gpu_device_name)
                    attempt.gpu_memory_mb = details.get('memory_mb', attempt.gpu_memory_mb)
                    attempt.gpu_capability = details.get('capability', '')
                    attempt.gpu_driver_version = details.get('driver_version', '')
                    if not attempt.gpu_count:  # If not set from metl.log
                        attempt.gpu_count = len(details.get('devices', []))
                
                # Fill in event times
                attempt.held_time = attempt_data.get('held_time')
                attempt.released_time = attempt_data.get('released_time')
                attempt.evicted_time = attempt_data.get('evicted_time')
                attempt.aborted_time = attempt_data.get('aborted_time')
                
                # Fill in resource information (use actual if available, otherwise targeted)
                actual_resource = attempt_data.get('actual_resource', '')
                
                if actual_resource:
                    # If there's a GLIDEIN_ResourceName, it indicates OSPool execution
                    attempt.targeted_resource = self._map_resource_name(actual_resource)
                else:
                    # No GLIDEIN_ResourceName means it ran on the targeted resource
                    # Look up the correct resource for this specific job-DAG combination
                    raw_targeted_resource = job_dag_to_resource.get((job_name, dag_source), '')
                    attempt.targeted_resource = self._map_resource_name(raw_targeted_resource)
                
                attempts.append(attempt)
        
        # Sort by job name, cluster ID, and attempt sequence
        attempts.sort(key=lambda x: (x.job_name, x.cluster_id, x.attempt_sequence))
        
        return attempts
    
    def _map_resource_name(self, resource_name: str) -> str:
        """Map resource names using dagman_monitor's logic: keep major resources, others become 'ospool'."""
        major_resources = {"expanse", "bridges2", "delta", "anvil"}
        
        if resource_name and resource_name.lower() in major_resources:
            return resource_name
        elif resource_name:
            return "ospool"
        else:
            return ""
    
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
            'Final Status', 'Targeted Resource', 'Submit Time', 'Start Time', 'End Time',
            'Execution Duration (seconds)', 'Execution Duration (human)',
            'Total Duration (seconds)', 'Total Duration (human)',
            'Total Bytes Sent', 'Total Bytes Received',
            'Number of GPUs', 'GPU Device Name', 'GPU Memory MB', 'GPU Capability', 'GPU Driver Version',
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
                        'Targeted Resource': attempt.targeted_resource,
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
                        'GPU Capability': attempt.gpu_capability,
                        'GPU Driver Version': attempt.gpu_driver_version,
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