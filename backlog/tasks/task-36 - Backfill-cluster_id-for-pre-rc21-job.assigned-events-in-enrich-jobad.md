---
id: TASK-36
title: Backfill cluster_id for pre-rc21 job.assigned events in enrich-jobad
status: Done
assignee:
  - claude
created_date: '2026-08-26 14:59'
updated_date: '2026-08-26 14:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
capture_job_ad_fields() only started including cluster_id/proc_id as of v0.1.0rc21. job.assigned events written by jobs pinned to an older MLDAG_VERSION (pretrain_local.sh installs a specific pinned version inside each job) have neither field, so enrich_from_jobad_events() silently skipped every one of them -- confirmed against a real db: 86 job.assigned events, 0 with cluster_id.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 job.assigned events lacking cluster_id are backfilled by cross-referencing any other event sharing the same run_id (job.executing/job.queued/etc from log_monitor.py always carry both run_id and cluster_id together)
- [x] #2 A job.assigned event with no cluster_id of its own and no cluster_id-bearing event sharing its run_id is skipped, not counted, not a crash
- [x] #3 proc_id defaults to 0 when backfilled this way
- [x] #4 Regression tests cover: successful backfill via a shared run_id, no backfill source available, and a cluster_id-bearing event for an unrelated run_id not leaking in
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
enrich_from_jobad_events() now builds a run_id -> cluster_id map from every cluster_id-bearing event in the table, and uses it to backfill cluster_id for job.assigned events that lack it (proc_id defaults to 0 when backfilled). Events with neither their own cluster_id nor a same-run_id fallback are skipped without error. 3 new tests. Full suite: 278 passed, 2 skipped (26 in test_history_enrich.py verified separately against a htcondor2 stub); ruff/ty clean.
<!-- SECTION:NOTES:END -->
