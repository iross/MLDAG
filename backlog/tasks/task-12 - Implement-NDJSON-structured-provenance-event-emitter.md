---
id: task-12
title: Implement NDJSON structured provenance event emitter
status: To Do
assignee: []
created_date: '2026-04-27 20:28'
labels: []
parent_task_id: task-16
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Python module that emits structured lifecycle events to a newline-delimited JSON file, one event per line. Events follow the common envelope schema from provenance_design.md (schema_version, type, run_id, ts, plus type-specific fields). Starting with NDJSON on the shared filesystem keeps the implementation zero-infrastructure while remaining trivially loadable into pandas or DuckDB for analysis. The storage location (shared FS path convention) must be decided before implementing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 emit_event(event_type, run_id, **fields) appends one JSON line to the run's event log file,Common envelope fields (schema_version, type, run_id, ts) are always present and correct,All seven event types from the design are supported: job.submitted, job.assigned, epoch.started, epoch.completed, job.migrated, job.failed, job.completed,Event log file path follows a documented convention (e.g. output/provenance/<run_id>.ndjson),Concurrent writes from different epochs are safe (atomic line appends),Module has no external dependencies,Unit tests cover envelope fields, all event types, and file append behavior
<!-- AC:END -->
