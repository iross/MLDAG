---
id: TASK-33
title: Fix condor_history OperationalError on pre-rc21 provenance.db
status: Done
assignee:
  - claude
created_date: '2026-08-24 18:56'
updated_date: '2026-08-24 18:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-30 (v0.1.0rc21) changed condor_history's schema (composite cluster_id/proc_id PK, new source/job_name/site/gpu_ids/status columns) but CREATE TABLE IF NOT EXISTS never touches an existing table -- any provenance.db built under v0.1.0rc19/rc20 keeps its old cluster_id-only condor_history table forever, and every enrich-history/enrich-jobad/scan --db call against it crashes with 'OperationalError: no such column: source'. Hit in production immediately after upgrading to rc21.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 _init_schema detects a condor_history table missing the source column and drops it before recreating, since condor_history (unlike checkpoints/events) has no incremental ingestion state to lose -- every row is fully re-derivable by rerunning enrich-history/enrich-jobad/scan --db
- [x] #2 A warning is logged when this happens, telling the user their condor_history data was cleared and needs repopulating
- [x] #3 A condor_history table already on the current schema is left completely untouched
- [x] #4 Regression test reproduces the exact pre-rc21 schema and confirms build_database/enrich_from_condor_history no longer raise
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added _drop_stale_condor_history() to db.py, called at the top of _init_schema before the CREATE TABLE IF NOT EXISTS script runs: checks PRAGMA table_info(condor_history) for the 'source' column and DROPs the table if it's missing (present-but-old table) vs doesn't exist at all (fresh db, no-op). condor_history has no incremental ingestion state (no byte offsets/mtimes to lose, unlike checkpoints/events), so dropping and letting it recreate is safe and, unlike telling someone to delete their whole provenance.db, doesn't force a full rebuild of checkpoints/events too. Logs a warning explaining what happened and what to rerun. Reproduced the user's exact crash with a hand-built pre-rc21 schema + INSERT, confirmed the fix converts it to a clean self-heal (warning printed, correct new columns present, stale row gone). 2 new tests in test_db.py: stale schema is recreated cleanly, and a current-schema table survives a second build untouched. Full suite: 278 passed, 2 skipped; ruff and ty clean.
<!-- SECTION:NOTES:END -->
