---
id: task-1.1
title: Scan the job event log for unhandled event codes.
status: Done
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
parent_task_id: task-1
---

## Description
For example, 004 marks an EVICT event but is not (I believe) handled in the code.

## Implementation Notes

Found unhandled HTCondor event codes in metl.log: 004 (448 occurrences), 006 (22042 occurrences), 009 (253 occurrences), 021 (904 occurrences), 022 (82 occurrences), 023 (69 occurrences), 024 (13 occurrences). Currently only handling: 000, 001, 005, 012, 013, 040.

Completed scan of metl.log. Found these unhandled HTCondor event codes: 004=Job evicted (448 times), 006=Image size updated (22042 times), 009=Job aborted (253 times), 021=Messages from starter (904 times), 022=(82 times), 023=(69 times), 024=(13 times). Currently only handling: 000, 001, 005, 012, 013, 040.
