---
id: TASK-19
title: Test provenance tracking using full simulated log files based on historical
status: Done
assignee:
  - claude
created_date: '2026-06-16 00:55'
updated_date: '2026-06-16'
labels: []
dependencies: []
---

## Description

Unit tests for `monitor_once` cover isolated event types. This task creates end-to-end tests that feed a complete, realistic HTCondor event log (modelled after historical production logs) through multiple `monitor_once` calls and verify the full sequence of provenance events emitted is correct. This validates the monitor's behaviour across realistic multi-job, multi-poll scenarios.

## Acceptance Criteria

- [x] At least one test uses a multi-event log covering submission (000), executing (001), transfer events (040), and completion (005) for multiple concurrent jobs
- [x] Tests verify the correct sequence of provenance event types is emitted for the full lifecycle of each job
- [x] Tests cover the scenario where DAGNodeName is present in the 000 block (new format) and the NDJSON job.submitted record is available
- [x] Tests cover multi-poll processing (log grows across calls, offsets carry over correctly)
- [x] All tests pass with the existing test runner

## Implementation Plan

1. Create `tests/provenance/test_log_monitor_integration.py`
2. Write helper to generate a realistic multi-job HTCondor log file
3. Write a full-lifecycle test: submit → queued (via DAGNodeName) → executing → transfer input → transfer output → complete (005, no provenance event but offset advances)
4. Write a multi-job test verifying two concurrent jobs' events are correctly attributed to their respective run_ids
5. Write a multi-poll test: log delivered in chunks, verify events emitted correctly across polls
6. Run full test suite to confirm all pass

## Implementation Notes

Created `tests/provenance/test_log_monitor_integration.py` with 8 tests using two concurrent simulated jobs (cluster IDs 5100001/5100002, run-aaa/run-bbb):
- Full single-poll lifecycle for job A: queued → transfer input → executing → transfer output
- Full single-poll lifecycle for job B: queued → executing → evicted → held → released
- Cross-contamination test: events attributed only to correct run_id
- cluster_id correctness test
- Multi-poll offset advancement test
- Events-emitted-only-once test (idempotency via offset)
- 000 header/body split across poll boundary test
- NDJSON written after 000 seen (pending → resolved on next poll) test

All 188 tests pass.

