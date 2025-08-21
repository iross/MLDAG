---
id: task-7
title: Experiment report generation
status: To Do
assignee: []
created_date: '2025-08-20 18:18'
labels: []
dependencies: []
---

## Description
Create some scripts that help generate "final" statistics for experiment runs. Things of interest:

- Breakdown of how many epochs were created in each resource (Expanse, Delta, Anvil, AWS, Bridges-2, OSPool)
- Breakdown of how much time was spent on each resource (Expanse, Delta, Anvil, AWS, Bridges-2, OSPool)
  - Including time spent on successful vs unsuccessful runs (Out of x hours spent computing, how many succeeded and resulted in a successful epoch)
- Resource requests and usage:
  - What GPUs were used and in what mixture?
  - How much memory was requested? Used?
  - How much CPU was requested? Used?
  - How much disk was requested? Used?
  - How much network bandwidth was requested? Used?

Inputs should be csvs that are created via the new `post_experiment_csv.py` script.

Use pandas and seaborn (and matplotlib, if necessary) as the primary tools for data analysis and visualization.

The input csv should be a command line argument with a default of job_summary.csv.

The style of the figures should be consistent across all the different plots.

## Acceptance Criteria

- [ ] Create a script that takes CSV input from `post_experiment_csv.py` with default filename `job_summary.csv`
- [ ] Generate resource utilization breakdown showing:
  - [ ] Number of completed epochs per resource (expanse, delta, anvil, bridges2, ospool)
  - [ ] Total computation time per resource 
  - [ ] Success rate per resource (successful vs unsuccessful time)
- [ ] Generate GPU utilization analysis showing:
  - [ ] Distribution of GPU types used across resources
  - [ ] GPU memory utilization patterns
  - [ ] GPU usage efficiency metrics
- [ ] Generate data transfer analysis:
  - [ ] Network usage per resource (bytes sent/received)
  - [ ] Transfer efficiency metrics per job type
  - [ ] Data transfer patterns over time
- [ ] Create visualizations using pandas/seaborn/matplotlib with:
  - [ ] Consistent styling and color schemes
  - [ ] Clear labels and titles
  - [ ] Appropriate chart types for each metric
  - [ ] Export figures as individual PNG files
- [ ] Generate summary statistics report including:
  - [ ] Overall experiment efficiency metrics
  - [ ] Resource-specific performance insights
  - [ ] Recommendations for future experiments
- [ ] Command-line interface supporting:
  - [ ] Input CSV file specification
  - [ ] Output directory for generated reports
  - [ ] Optional filtering by date range or resource
  - [ ] Verbose logging option

## Implementation Plan

1. **Data Analysis and Enhancement**
   - Analyze current CSV output from `post_experiment_csv.py` 
   - Identify missing resource usage data (CPU, memory, disk, network)
   - Enhance CSV generator to extract resource usage from HTCondor logs
   - Design data structures for efficient analysis

2. **Core Analysis Script Development**
   - Create main `experiment_report.py` script with CLI interface
   - Implement data loading and validation functions
   - Create helper functions for epoch extraction from job names
   - Implement resource efficiency calculations

3. **Resource Utilization Analysis**
   - Implement epoch counting per resource with success/failure breakdown
   - Calculate total computation time per resource
   - Compute success rates and efficiency metrics
   - Generate resource comparison tables

4. **GPU Analysis Module**
   - Implement GPU type distribution analysis
   - Calculate GPU memory utilization patterns
   - Analyze GPU usage efficiency across resources
   - Create GPU performance comparison metrics

5. **Data Transfer Analysis**
   - Implement network usage analysis using existing bytes sent/received data
   - Calculate transfer efficiency metrics per resource and job type
   - Create transfer pattern analysis over time
   - Generate data movement visualizations

6. **Visualization Framework**
   - Set up consistent seaborn/matplotlib styling
   - Create reusable plotting functions for each chart type
   - Implement resource utilization plots (bar charts, pie charts)
   - Create time series plots for resource usage over time
   - Generate GPU utilization heatmaps and distributions
   - Export all figures as individual PNG files

7. **Report Generation**
   - Implement summary statistics calculations
   - Create formatted text reports with key insights
   - Generate executive summary with recommendations
   - Add export functionality for multiple formats

8. **Testing and Documentation**
   - Test with current job_output.csv data
   - Validate calculations against known results
   - Create usage documentation and examples
   - Add error handling for malformed data

## Implementation Notes

### Research Findings:
- **Available Data**: Current CSV contains resource types (expanse, delta, anvil, bridges2, ospool), job statuses, GPU info, timing data, and transfer data
- **Epoch Extraction**: Job names contain epoch information (epoch0-epoch29) that can be parsed for analysis
- **Transfer Data**: Total bytes sent/received already available in CSV output
- **Resources**: 5 distinct resources with good coverage across all major HPC centers

### Implementation Decisions:
- **Input**: Default to `job_summary.csv` (updated from `job_output.csv`)
- **Network Analysis**: Focus on job data transfer (bytes sent/received) rather than HTCondor internal metrics  
- **Visualization**: Individual PNG files for each chart type with consistent styling
- **Resource Usage**: Defer CPU/memory/disk usage enhancement to future iteration
- **Scope**: Build comprehensive analysis with current data, enhance data collection later

### Key Analysis Areas:
1. **Resource Efficiency**: Epoch completion rates and computation time per resource
2. **GPU Utilization**: Distribution and efficiency across different GPU types
3. **Transfer Analysis**: Network usage patterns and efficiency metrics
4. **Success Metrics**: Resource-specific performance insights and recommendations

*Implementation details to be added during development*
