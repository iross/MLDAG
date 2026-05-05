---
id: task-15
title: Implement lineage reconstruction and provenance query CLI
status: Done
assignee: []
created_date: '2026-04-27 20:32'
updated_date: '2026-04-29 00:45'
labels: []
dependencies:
  - task-11
  - task-12
parent_task_id: task-16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Command-line tooling that consumes the sidecar chain (task-11) and NDJSON event log (task-12) to answer lineage questions: reconstruct the full training history of any checkpoint, query which sites handled a run, find runs that migrated, and show loss trajectories across multi-site runs. This is the user-facing payoff of the provenance infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 walk_lineage(checkpoint_path) returns the ordered list of sidecar records from epoch 0 to the given checkpoint
- [x] #2 query_run(run_id, log_dir) returns all events for a run in chronological order from the NDJSON log
- [x] #3 CLI subcommand: lineage <checkpoint_path> — prints the ancestry chain
- [x] #4 CLI subcommand: events <run_id> — prints all events for a run
- [x] #5 Output is human-readable by default; --json flag emits compact JSON
- [x] #6 Works without a database — reads only sidecar files and NDJSON logs
- [x] #7 Missing or corrupt sidecar files produce a clear error message and non-zero exit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created mldag/provenance/query.py with: walk_lineage() that follows parent_hash links back to epoch 0 via _find_checkpoint_by_hash() (rglob sidecar search), then reverses to return oldest-first; query_run() that reads the per-run NDJSON file, skips malformed lines, and sorts by ts. Typer CLI app with two subcommands: 'lineage <checkpoint>' and 'events <run_id>', both supporting --json for machine-readable output and --log-dir for non-default NDJSON dirs. Both raise meaningful errors on missing/corrupt files with non-zero exit. Added mldag-query console script entry point in pyproject.toml. 15 tests pass; full suite 72/72.
<!-- SECTION:NOTES:END -->
