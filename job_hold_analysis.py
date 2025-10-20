#!/usr/bin/env python3
"""
Analyze job holds from HTCondor metl.log file.
Extracts job ID, execution location, and hold reason for each held job.
"""

import re
import csv
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def extract_job_id(line: str) -> Optional[str]:
    """Extract job ID from HTCondor log line."""
    match = re.search(r'\((\d+\.\d+\.\d+)\)', line)
    return match.group(1) if match else None

def extract_host_info(line: str) -> Optional[str]:
    """Extract host information from job execution line."""
    # Look for alias= pattern first (cleaner host names)
    alias_match = re.search(r'alias=([^&]+)', line)
    if alias_match:
        return alias_match.group(1)
    
    # Fallback to PrivNet= if available
    privnet_match = re.search(r'PrivNet=([^&]+)', line)
    if privnet_match:
        return privnet_match.group(1)
    
    # Last resort: extract IP from the beginning of host string
    ip_match = re.search(r'<(\d+\.\d+\.\d+\.\d+)', line)
    return ip_match.group(1) if ip_match else "Unknown"

def parse_dag_files() -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    """Parse DAG files to extract job name to DAG source mapping and resource names.
    
    Returns:
        Tuple of (job_to_dag_mapping, (job_name, dag_source)_to_resource_mapping)
    """
    job_to_dag = {}
    job_dag_to_resource = {}

    dag_files = glob.glob("*.dag")
    
    for dag_file in dag_files:
        if not Path(dag_file).exists():
            continue
            
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

    return job_to_dag, job_dag_to_resource

def parse_dagman_out_files() -> Dict[str, Dict[int, str]]:
    """Parse DAGMan output files to create node-to-jobid mapping for each DAG.
    
    Returns:
        Dictionary mapping dag_source to {cluster_id: job_name} mappings
    """
    dag_mappings = {}
    
    dagman_out_files = glob.glob("*.dag.dagman.out")
    
    for dagman_out_file in dagman_out_files:
        # Extract DAG source from filename (remove .dag.dagman.out)
        dag_source = Path(dagman_out_file).name.replace('.dag.dagman.out', '')

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

    return dag_mappings

def find_job_and_dag_for_cluster(cluster_id: int, dag_mappings: Dict[str, Dict[int, str]]) -> Tuple[Optional[str], Optional[str]]:
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

def map_resource_name(resource_name: str) -> str:
    """Map resource names using same logic as post_experiment_csv: keep major resources, others become 'ospool'."""
    major_resources = {"expanse", "bridges2", "delta", "anvil"}
    
    if resource_name and resource_name.lower() in major_resources:
        return resource_name
    elif resource_name:
        return "ospool"
    else:
        return ""

def parse_job_holds(log_file: str) -> List[Dict[str, str]]:
    """Parse job holds from HTCondor log file."""
    # Parse DAG files and mappings to get targeted resource info
    job_to_dag, job_dag_to_resource = parse_dag_files()
    dag_mappings = parse_dagman_out_files()
    
    job_holds = []
    job_locations = {}  # Track where jobs were last executing
    job_glidein_resources = {}  # Track GLIDEIN_ResourceName for jobs
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Track job execution locations
        if "001" in line and "Job executing on host:" in line:
            job_id = extract_job_id(line)
            if job_id:
                host_info = extract_host_info(line)
                job_locations[job_id] = host_info
                
                # Look for GLIDEIN_ResourceName in subsequent lines
                for j in range(i + 1, min(i + 20, len(lines))):
                    resource_line = lines[j].strip()
                    if re.match(r'^\d{3} \(', resource_line):
                        break  # Hit next log entry
                    
                    # Look for GLIDEIN_ResourceName (indicates OSPool execution)
                    if 'GLIDEIN_ResourceName = ' in resource_line:
                        resource_match = re.search(r'GLIDEIN_ResourceName = "([^"]*)"', resource_line)
                        if resource_match:
                            job_glidein_resources[job_id] = resource_match.group(1)
                            break
        
        # Process job holds
        elif "012" in line and "Job was held." in line:
            job_id = extract_job_id(line)
            if job_id:
                # Get location from our tracking dict
                location = job_locations.get(job_id, "Unknown")
                
                # Collect hold reason from subsequent lines
                hold_reasons = []
                j = i + 1
                while j < len(lines):
                    reason_line = lines[j]  # Don't strip yet so we can check for tabs
                    reason_line_stripped = reason_line.strip()
                    
                    # Stop if we hit a new log entry marker or empty/ellipsis line
                    if reason_line_stripped in ['...', ''] or re.match(r'^\d{3} \(', reason_line_stripped):
                        break
                    
                    # Process tabbed reason lines (check original line for tab)
                    if reason_line.startswith('\t'):
                        # Clean up the reason line
                        clean_reason = reason_line.replace('\t', '').strip()
                        # Skip code lines but keep everything else
                        if clean_reason and not re.match(r'^Code \d+', clean_reason):
                            hold_reasons.append(clean_reason)
                    
                    j += 1
                
                # Join all reason lines
                full_reason = ' '.join(hold_reasons) if hold_reasons else "No reason provided"
                
                # Determine targeted resource
                cluster_id_int = int(job_id.split('.')[0])  # Extract cluster ID from job_id
                
                # First, check execution location for major resources (most reliable)
                targeted_resource = ""
                if ("delta.ncsa.illinois.edu" in location or 
                    "delta.ncsa.illinois.edu" in full_reason or
                    "gpua" in location and "delta" in full_reason):
                    targeted_resource = "delta"
                elif ("expanse.sdsc.edu" in location or 
                      "expanse.sdsc.edu" in full_reason or
                      "exp-" in location):
                    targeted_resource = "expanse"
                elif ("anvil.rcac.purdue.edu" in location or
                      "anvil.rcac.purdue.edu" in full_reason or
                      location.startswith("g") and "anvil" in location):
                    targeted_resource = "anvil"
                elif ("bridges2.psc.edu" in location or
                      "bridges2.psc.edu" in full_reason or
                      location.startswith("v") and "bridges2" in location):
                    targeted_resource = "bridges2"
                
                # If we didn't detect a major resource from location, check other sources
                if not targeted_resource:
                    # Check if we have GLIDEIN_ResourceName (indicates OSPool execution)
                    if job_id in job_glidein_resources:
                        targeted_resource = map_resource_name(job_glidein_resources[job_id])
                    else:
                        # Find the targeted resource from DAG files
                        job_name, dag_source = find_job_and_dag_for_cluster(cluster_id_int, dag_mappings)
                        
                        if job_name and dag_source:
                            # Look up the targeted resource for this job-DAG combination
                            raw_targeted_resource = job_dag_to_resource.get((job_name, dag_source), '')
                            targeted_resource = map_resource_name(raw_targeted_resource)
                        else:
                            # Default to ospool if we can't determine
                            targeted_resource = "ospool"
                
                job_holds.append({
                    'job_id': job_id,
                    'location': location,
                    'hold_reason': full_reason,
                    'targeted_resource': targeted_resource
                })
        
        i += 1
    
    return job_holds

def write_csv(job_holds: List[Dict[str, str]], output_file: str):
    """Write job holds to CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['job_id', 'location', 'targeted_resource', 'hold_reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for hold in job_holds:
            writer.writerow(hold)

def main():
    log_file = "metl.log"
    output_file = "job_holds_analysis.csv"
    
    print(f"Parsing job holds from {log_file}...")
    job_holds = parse_job_holds(log_file)
    
    print(f"Found {len(job_holds)} job holds")
    
    print(f"Writing results to {output_file}...")
    write_csv(job_holds, output_file)
    
    print("Analysis complete!")
    print(f"\nSample results:")
    for i, hold in enumerate(job_holds[:5]):
        reason_preview = hold['hold_reason'][:100] + "..." if len(hold['hold_reason']) > 100 else hold['hold_reason']
        print(f"  {hold['job_id']} @ {hold['location']} (targeted: {hold['targeted_resource']}): {reason_preview}")
    
    # Show a few more detailed examples
    print(f"\nDetailed examples:")
    for i, hold in enumerate(job_holds[:3]):
        print(f"\n{i+1}. Job ID: {hold['job_id']}")
        print(f"   Location: {hold['location']}")
        print(f"   Targeted Resource: {hold['targeted_resource']}")
        print(f"   Reason: {hold['hold_reason']}")

if __name__ == "__main__":
    main()