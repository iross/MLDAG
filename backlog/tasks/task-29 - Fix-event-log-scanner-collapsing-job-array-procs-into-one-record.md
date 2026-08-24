---
id: TASK-29
title: Fix event-log scanner collapsing job-array procs into one record
status: Done
assignee:
  - claude
created_date: '2026-08-24 16:08'
updated_date: '2026-08-24 16:08'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
mldag-query scan (task-27) keyed jobs by cluster_id alone, reusing log_monitor.py's _ANY_HEADER_RE which discards proc_id -- correct for DAGMan-submitted jobs (always cluster.0) but wrong for queue N job arrays, where many procs share one cluster_id. Confirmed against a real production log (rocksrocksrocks.log): 11 clusters but 275 distinct cluster.proc pairs, and scan only reported 11 records, silently overwriting every proc but the last one seen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Jobs are keyed by (cluster_id, proc_id), so every proc in a job-array cluster gets its own record
- [x] #2 run_id/.ad enrichment (cluster-id-keyed by design) is still applied correctly to every proc sharing a cluster
- [x] #3 CLI output shows cluster.proc as a stable unique identifier even when run_id/job_name is shared across procs
- [x] #4 Verified against rocksrocksrocks.log: 275 jobs reported, not 11
- [x] #5 Unit tests cover: multiple procs in one cluster produce separate records, sorting by (cluster_id, proc_id), and run_id enrichment shared across procs while proc_id keeps rows distinct
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a local _JOB_HEADER_RE in event_log_scan.py capturing proc_id (log_monitor.py's _ANY_HEADER_RE deliberately discards it)
2. Key records by (cluster_id, proc_id) tuple instead of cluster_id; thread proc_id through _new_record/_flush_block/scan_event_log's block-tracking state
3. Update query.py's _format_scan to show cluster.proc as the stable identifier alongside any resolved run_id/job_name
4. Add regression tests: multi-proc cluster produces separate records, sorted by (cluster_id, proc_id), run_id enrichment shared across procs while proc_id keeps rows distinct
5. Verify against rocksrocksrocks.log (275 jobs, not 11) and run full suite/lint/type checks
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: log_monitor.py's _ANY_HEADER_RE only captures cluster_id (correct for DAGMan, which always submits cluster.0) so event_log_scan.py inherited that blind spot for queue N job arrays. Added a local _JOB_HEADER_RE in event_log_scan.py that also captures proc_id, and rekeyed records/scan state to (cluster_id, proc_id) tuples throughout (_new_record, _flush_block, scan_event_log's block-tracking, final sort). run_id/.ad enrichment stays cluster_id-keyed (matching how post.py/log_monitor.py name their marker/classad files) so it correctly applies the same run_id to every proc in a cluster. query.py's _format_scan now always shows cluster.proc as the row identifier, with run_id/job_name shown alongside rather than replacing it, so rows stay distinguishable even when every proc in an array shares one run_id. Verified against the real production log that surfaced this (rocksrocksrocks.log): now reports 275 individual jobs instead of 11 collapsed clusters. Added 3 regression tests to tests/provenance/test_event_log_scan.py (17 total in that file). Full suite: 271 passed, 2 skipped; ruff and ty clean.
<!-- SECTION:NOTES:END -->
