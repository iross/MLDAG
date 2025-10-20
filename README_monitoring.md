# MLDAG Monitoring System

This monitoring system provides comprehensive tracking of HTCondor DAG-based machine learning training pipelines.

## Components

### 1. DAGMan Log Monitor (`dagman_monitor.py`)
- Monitors DAGMan log files for job status changes
- Tracks job submissions, executions, completions, and failures
- Parses job variables (run_uuid, epoch, resource_name) from DAG files
- Provides training run progress tracking

### 2. HTCondor Job Monitor (`htcondor_monitor.py`) 
- Uses `condor_q` and `condor_history` to get detailed job information
- Tracks resource usage (CPU time, memory, disk)
- Monitors job holds and failure reasons
- Provides machine assignment information

### 3. Integrated Monitor (`mldag_monitor.py`)
- Combines DAGMan and HTCondor monitoring
- Correlates DAG jobs with HTCondor cluster IDs
- Provides unified job status and resource utilization views
- Training run progress tracking with detailed statistics

## Setup

1. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install pydantic typer pyyaml rich watchdog
```

2. Make sure HTCondor command-line tools are available:
```bash
which condor_q condor_history
```

## Usage

### Monitor All Jobs (Live)
```bash
python mldag_monitor.py monitor --live --refresh-interval 5.0
```

### Monitor Specific Training Run
```bash
python mldag_monitor.py monitor --run-uuid abc123 --live
```

### One-time Status Check
```bash
python mldag_monitor.py status
```

### DAGMan-only Monitoring
```bash
python dagman_monitor.py monitor --dag-file pipeline.dag --live
```

### HTCondor-only Monitoring
```bash
python htcondor_monitor.py monitor --live
python htcondor_monitor.py summary --limit 100
```

## Features

### Job Status Tracking
- Real-time job status updates from both DAG and HTCondor perspectives
- Job retry counting and failure tracking
- Hold reason detection and display

### Training Run Progress
- Progress bars showing completion percentage per training run
- Epoch-level tracking for multi-epoch training jobs
- Resource allocation and utilization per training run

### Resource Utilization
- CPU time, memory, and disk usage tracking
- Resource efficiency metrics by compute resource
- Average runtime calculations

### Live Monitoring
- Auto-refreshing terminal displays using Rich library
- Configurable refresh intervals
- Keyboard interrupt handling for clean exits

## File Structure

The monitoring system expects:
- `pipeline.dag` - Main DAG file (configurable)
- `pipeline.dag.dagman.log` - DAGMan log file (auto-detected)
- `pipeline.dag.status` - Node status file (auto-detected)

## Integration with Existing Code

The monitors extract metadata from your existing DAG structure:
- Job names from `JOB` declarations
- Variables from `VARS` lines (run_uuid, epoch, ResourceName)
- Resource assignments from job submit descriptions

## Troubleshooting

### HTCondor Command Issues
If `condor_q` or `condor_history` fail:
- Ensure HTCondor is properly installed and configured
- Check that you can manually run these commands
- Verify network connectivity to HTCondor collector

### Missing Job Information
If jobs don't show expected metadata:
- Verify VARS declarations in DAG file include run_uuid, epoch, ResourceName
- Check that job names match between DAG and HTCondor submissions

### Performance
For large DAGs:
- Increase refresh intervals to reduce system load
- Use run_uuid filtering to focus on specific training runs
- Consider monitoring subsets of jobs rather than entire DAG