---
id: task-8
title: Add transfer input timing to experiment reports
status: In Progress
assignee: []
created_date: '2025-10-20 13:59'
updated_date: '2025-10-20 14:00'
labels: []
dependencies: []
---

## Description

Enhance experiment_report.py and post_experiment_csv.py to extract and display how long each job spent in the transferring input state from metl.log

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 experiment_report.py extracts transfer input timing from metl.log,post_experiment_csv.py includes transfer input duration in CSV output,Transfer timing data is accurately parsed and displayed,Documentation updated to reflect new metrics
<!-- AC:END -->

## Implementation Plan

1. Analyze metl.log format to understand transfer input state logging
2. Review current experiment_report.py implementation to understand data extraction patterns
3. Review current post_experiment_csv.py implementation to understand CSV generation
4. Add parser logic to extract transfer input state timestamps from metl.log
5. Calculate duration for each job's transfer input phase
6. Integrate transfer timing into experiment_report.py output
7. Add transfer input duration column to post_experiment_csv.py CSV output
8. Test with sample metl.log files to verify accuracy
9. Update documentation/comments to explain new metrics
