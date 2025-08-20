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
- Breakdown of successful vs unsuccessful runs (Out of x hours spent computing, how many succeeded and resulted in a successful epoch)
- Resource requests and usage:
  - What GPUs were used and in what mixture?
  - How much memory was requested? Used?
  - How much CPU was requested? Used?
  - How much disk was requested? Used?

Inputs should be csvs that are created via the `dagman_monitor.py --csv` script.
