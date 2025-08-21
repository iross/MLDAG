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
