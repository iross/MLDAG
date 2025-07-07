---
id: task-4
title: Better handling of epoch completion
status: Done
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
---

## Description
It seems like "live" monitoring doesn't work well when an epoch finished and the next epoch is submitted. Check the logic of how the monitoring table handles DAG nodes that are submitted after the previous begins.

## Implementation Plan

1. Understand the current live monitoring logic and identify the issue\n2. Test live monitoring to reproduce the epoch transition problem\n3. Analyze how DAG node submission timing affects the monitoring table\n4. Fix the logic to properly handle epoch completion → next epoch submission\n5. Test the fix with live monitoring during epoch transitions


## Implementation Notes

Successfully improved live monitoring for epoch transitions. Problem: Default mode showed 'latest epoch per run' which caused gaps during transitions. Solution: Implemented smart epoch selection with priority: 1) RUNNING/TRANSFERRING jobs, 2) IDLE queued jobs, 3) HELD jobs, 4) Latest COMPLETED jobs. Result: Smooth transitions with no gaps - always shows most relevant epoch status.
## Acceptance Criteria

- [x] Live monitoring correctly shows epoch transitions
- [x] Completed epochs remain visible until next epoch starts
- [x] Next epoch appears promptly when submitted
- [x] No missing or duplicate epochs in live display
- [x] Monitoring table updates smoothly during transitions
