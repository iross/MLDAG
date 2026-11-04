---
id: task-1.3
title: Update the monitor code to handle those state transitions
status: Done
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
parent_task_id: task-1
---

## Description

## Implementation Plan

1. Add new event code constants for 004, 006, 009, 021-024\n2. Add regex patterns for parsing these event codes\n3. Add event handlers in parse_metl_log_timing function\n4. Add event handlers in update_job_info function\n5. Test the implementation with current metl.log data

## Implementation Notes

Successfully implemented handling for all missing HTCondor event codes: 004=EVICTED->IDLE, 006=IMAGE_SIZE (no status change), 009=ABORTED->IDLE, 021-024=REMOTE_EVENTS (no status change). Added regex patterns, metl.log parsing, and status mapping. Testing shows jobs are now correctly handled and evicted/aborted jobs show as IDLE (waiting for DAGMan retry) instead of missing from table.
