---
id: task-1.2
title: Determine the job state resulting from each of those event codes
status: Done
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
parent_task_id: task-1
---

## Description
Ask me for help identify how to handle them. For example, EVICT implies that the job should be in an IDLE state in the report table.

## Implementation Notes

Determined job state mappings for unhandled event codes: 004=EVICTED->IDLE (job evicted from host), 006=No state change (image size update), 009=ABORTED->FAILED (job manually aborted), 021/022/023/024=No state change (remote system call events)

CRITICAL FINDING: After analyzing metl.log, discovered that DAGMan automatically re-submits failed/evicted jobs! Example: run0-train_epoch0 was submitted 20 times over 2 months (April-June) before finally succeeding. Pattern: 004=EVICTED -> 012=HELD -> 009=ABORTED, then DAGMan auto-resubmits with new cluster ID. Final cluster 12638268 succeeded with 005=TERMINATED. This means evicted/aborted jobs should be treated as IDLE (waiting for retry) not permanently FAILED.
