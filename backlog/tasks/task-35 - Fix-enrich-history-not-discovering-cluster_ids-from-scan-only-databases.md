---
id: TASK-35
title: Fix enrich-history not discovering cluster_ids from scan-only databases
status: Done
assignee:
  - claude
created_date: '2026-08-24 20:13'
updated_date: '2026-08-24 20:13'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
enrich_from_condor_history() only discovered candidate cluster_ids from the events table (built by db build from NDJSON). A provenance.db populated purely via scan --db (an ad hoc batch with no DAGMan/NDJSON instrumentation) has an empty events table, so enrich-history reported 'No new cluster_ids to enrich' even though those exact jobs were already sitting in condor_history waiting to be enriched further.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Candidate cluster_ids are the union of what's in the events table and what's already in condor_history (any source), not just events
- [x] #2 A db built purely via scan --db (or enrich-jobad), with no events table content at all, is now a valid input to enrich-history
- [x] #3 Regression test covers a scan-only db with zero events rows
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed _all_cluster_ids_from_events to _all_known_cluster_ids and unioned in SELECT DISTINCT cluster_id FROM condor_history alongside the existing events-table query. Reproduced the exact user scenario (scan --db into a fresh provenance.db, then enrich-history) and confirmed it now attempts the query instead of reporting nothing to do. 1 new regression test (a scan-only db with zero events rows). Full suite: 278 passed, 2 skipped (24 in test_history_enrich.py verified separately against a htcondor2 stub); ruff/ty clean.
<!-- SECTION:NOTES:END -->
