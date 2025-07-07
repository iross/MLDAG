---
id: task-1
title: Ensure that all HTCondor event codes are caught and handled
status: Done
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
---

## Description
Currently, not all job event codes are read from metl.log. This task contains two subtasks. See them for more details.

## Implementation Plan

1. Start with subtask 1.1 - Scan metl.log for unhandled HTCondor event codes\n2. Work on subtask 1.2 - Determine proper job states for each event code\n3. Complete subtask 1.3 - Update dagman_monitor.py to handle all event codes\n4. Test and validate all event codes are properly handled

## Implementation Notes

Successfully ensured all HTCondor event codes are caught and handled. Key discovery: DAGMan automatically retries failed/evicted jobs, so these should show as IDLE not FAILED. Implemented: Constants for all event codes (000,001,004,005,006,009,012,013,021-024,040), regex patterns for parsing, event handlers in metl.log parsing, status mapping in apply_metl_data_to_all_jobs, and event handling in update_job_info. All subtasks completed.
