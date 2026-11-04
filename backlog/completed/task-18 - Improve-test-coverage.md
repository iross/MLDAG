---
id: TASK-18
title: Improve test coverage
status: Done
assignee:
  - claude
created_date: '2026-06-16 00:54'
updated_date: '2026-06-16'
labels: []
dependencies: []
---

## Description

The log_monitor module has several functions and behaviors that lack direct unit tests. This task adds coverage for untested code paths introduced or present in the fix/post-run-id-hint-v2 branch, including pending lookup persistence, .run_id file writing on resolution, and direct tests for helper functions.

## Acceptance Criteria

- [x] `_save_pending` / `_load_pending` are covered by unit tests (roundtrip, missing file, corrupt file)
- [x] Tests verify that a `.run_id` file is written to `log_dir` when run_id is resolved from a DAGNodeName event 000 block
- [x] Tests verify that a `.run_id` file is written when a pending lookup is resolved on a subsequent poll
- [x] Tests verify that a `.run_id` file is written when a pending lookup is resolved inline during event processing
- [x] `_resolve_run_id` is tested directly for all four resolution paths (cache, .run_id marker, .ad ClassAd, unknown fallback)
- [x] All new tests pass with the existing test runner

## Implementation Plan

1. Add `_save_pending`, `_load_pending` to the imports in `test_log_monitor.py`
2. Write roundtrip, missing-file, and corrupt-file tests for pending persistence
3. Write tests verifying `.run_id` file is written for each resolution path in `monitor_once`
4. Write direct tests for `_resolve_run_id`
5. Run full test suite to confirm no regressions

## Implementation Notes

Added 18 new tests to `tests/provenance/test_log_monitor.py`:
- Imported `_load_pending`, `_resolve_run_id`, `_save_pending` (previously untested)
- 3 tests for `_save_pending`/`_load_pending` (roundtrip, missing file, corrupt file)
- 4 tests for `_resolve_run_id` covering all resolution paths: cache hit, `.run_id` marker, `.ad` ClassAd, unknown fallback
- 3 tests verifying `.run_id` file is written in each path that resolves run_id from a DAGNodeName 000 block (immediate resolution, pending resolution on next poll, inline resolution during event processing)

All 188 tests pass.

