---
id: TASK-30
title: Add cluster_id filtering and provenance.db dump to event-log scan
status: Done
assignee:
  - claude
created_date: '2026-08-24 16:26'
updated_date: '2026-08-24 16:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
mldag-query scan always dumped every job in a log to stdout/JSON with no way to narrow scope, and had no way to persist results into the same provenance.db workflow already in use for DAGMan runs. condor_history (from task-28) already stores per-job summary data queried from HTCondor's own history; scan's event-log-derived summaries are the same shape of data from a different source, so they belong in the same table, distinguished by a source column.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mldag-query scan accepts a repeatable --cluster-id to filter to a specific set of clusters (every proc of a matched cluster is kept)
- [x] #2 mldag-query scan --db <path> writes results into provenance.db's condor_history table with source='event_log'
- [x] #3 condor_history rows from db enrich-history are tagged source='condor_history'; rows from scan --db are tagged source='event_log', and both coexist without collision
- [x] #4 condor_history's primary key is (cluster_id, proc_id), fixing the same job-array bug just fixed in event_log_scan.py (a cluster_id alone is not unique for queue N job arrays)
- [x] #5 condor_history gains a status column populated from both sources (JobStatus for condor_history rows, event_log_scan's own status for event_log rows) so it can be queried uniformly
- [x] #6 Unit tests cover: cluster_id filtering (matching set, keeps every proc of a matched cluster, None scans everything), job-array condor_history enrichment (one row per proc, already-enriched tracked per cluster not per proc), and write_scan_records (event_log rows written correctly, coexisting with condor_history rows for other procs of the same cluster)
- [ ] #7 jobad.py's in-job $_CONDOR_JOB_AD capture (task-26) can also enrich condor_history via a third source ('jobad'), reusing the same merge-based upsert; it never writes remote_wall_clock_s/cpus_usage/memory_usage_mb/gpus_usage since those are meaningless before the job has run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rework condor_history's schema: composite PK (cluster_id, proc_id), add proc_id/job_name/site/gpu_ids/status/source columns
2. history_enrich.py: map ProcId, tag source='condor_history', derive status from JobStatus, fix already-enriched query to DISTINCT cluster_id, add write_scan_records()/_row_from_scan_record() for the event_log path
3. event_log_scan.py: add a cluster_ids allowlist parameter to scan_event_log(), filtering at block-flush time
4. query.py: add repeatable --cluster-id and --db options to the scan command
5. Update README
6. Tests for filtering, job-array condor_history enrichment, and write_scan_records; verify against rocksrocksrocks.log end to end
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reworked condor_history's schema (db.py): composite PRIMARY KEY (cluster_id, proc_id) -- fixing the identical job-array bug just fixed in event_log_scan.py, since condor_history returns one ClassAd per proc, not per cluster -- plus new proc_id/job_name/site/gpu_ids/status/arguments/source columns. history_enrich.py now maps ProcId/Arguments, tags every row source='condor_history', derives a status string from the numeric JobStatus ClassAd attribute via a small lookup table, and fixes the already-enriched check to scope on source='condor_history' specifically (not "any row exists"). Added write_scan_records()/_row_from_scan_record() mapping event_log_scan.py's record shape onto the same table with source='event_log'.

Follow-up fix after initial review: the first version of _upsert_history_row used INSERT OR REPLACE, which blindly replaces the whole row -- writing from one source after the other had already written the same (cluster_id, proc_id) would silently NULL out whichever columns only the first write populated. Reworked to a real merge: ON CONFLICT DO UPDATE with COALESCE(excluded.col, condor_history.col) per column, raw payload split into separate condor_history_json/event_log_json/jobad_json columns (each nullable, only ever written by its own source), and `source` unioned into a sorted comma-set (idempotent -- writing the same source twice doesn't duplicate it) rather than overwritten.

Second follow-up, addressing "can jobad enrich the same way": added enrich_from_jobad_events(), a third source mirroring jobad.py's in-job $_CONDOR_JOB_AD capture (task-26, emitted as a job.assigned event from pretrain_local.sh) into condor_history (source='jobad'), reading from provenance.db's already-built `events` table rather than re-touching NDJSON. capture_job_ad_fields() now always includes cluster_id/proc_id when present in the ad (previously omitted -- needed as the join key), which flows into job.assigned's existing payload with zero changes to pretrain_local.sh itself, preserving "instant postscript" behavior exactly as-is. Deliberately excludes remote_wall_clock_s/cpus_usage/memory_usage_mb/gpus_usage from this path even though capture_job_ad_fields() can return them: $_CONDOR_JOB_AD is captured at job start, before the job has run, so those would be near-zero placeholders, not real usage -- verified this is never written even when the source event happens to carry them. Wired as `mldag-query db enrich-jobad`.

No migration path was written for existing condor_history rows under the old schema (CREATE TABLE IF NOT EXISTS is a no-op against an existing table) -- the db.py module docstring already documents provenance.db as a rebuildable cache over source-of-truth files, so the fix is to delete and rebuild it, not migrate it.

Verified end-to-end against rocksrocksrocks.log (--cluster-id filtering, --db dump, idempotent re-scan) and a synthetic job.assigned event (db build -> db enrich-jobad -> correct fields, usage fields correctly absent). 20 new/updated tests across test_event_log_scan.py, test_history_enrich.py, and test_jobad.py. Full suite: 276 passed, 2 skipped; ruff clean; ty clean except the two pre-existing build_database diagnostics and the expected htcondor2 unresolved-import.
<!-- SECTION:NOTES:END -->
