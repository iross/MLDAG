---
id: task-15
title: Implement lineage reconstruction and provenance query CLI
status: To Do
assignee: []
created_date: '2026-04-27 20:32'
labels: []
dependencies:
  - task-11
  - task-12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Command-line tooling that consumes the sidecar chain (task-11) and NDJSON event log (task-12) to answer lineage questions: reconstruct the full training history of any checkpoint, query which sites handled a run, find runs that migrated, and show loss trajectories across multi-site runs. This is the user-facing payoff of the provenance infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 walk_lineage(checkpoint_path) returns the ordered list of sidecar records from epoch 0 to the given checkpoint,query_run(run_id) returns all events for a run in chronological order from the NDJSON log,CLI supports at minimum: lineage <checkpoint_path> and events <run_id>,Output is human-readable by default and machine-readable (JSON) with a flag,Works without a database — reads only sidecar files and NDJSON event logs,Handles missing or corrupt sidecar files gracefully with a clear error message
<!-- AC:END -->
