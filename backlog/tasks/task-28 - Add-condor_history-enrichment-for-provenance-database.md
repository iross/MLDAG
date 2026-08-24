---
id: TASK-28
title: Add condor_history enrichment for provenance database
status: Done
assignee:
  - claude
created_date: '2026-08-24 14:53'
updated_date: '2026-08-24 15:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
provenance.db is built only from NDJSON events and checkpoint sidecars. condor_history holds ClassAd fields (final remote host, exit status, hold-reason history, requested vs used resources) that the event-log/NDJSON pipeline never captures. Backfilling an existing database from condor_history via the HTCondor Python bindings lets this data be added after the fact, for jobs already in the db.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new condor_history table stores the queried ClassAd subset keyed by cluster_id
- [x] #2 Enrichment queries condor_history via the HTCondor Python bindings (htcondor2), not the condor_history CLI
- [x] #3 cluster_ids are discovered from the existing events table so only already-known jobs are queried
- [x] #4 cluster_ids with no matching historical record are skipped silently, not treated as errors
- [x] #5 Already-enriched cluster_ids are skipped on repeat runs unless a full-rescan is requested
- [x] #6 A CLI command (mldag-query db enrich-history) exposes this
- [x] #7 Unit tests cover the enrichment logic against a mocked Schedd, since htcondor2 is a Linux-only dependency (pytest.importorskip like test_daggen.py)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a condor_history table to db.py's schema (cluster_id PK, a fixed set of ClassAd-derived columns, full classad_json, queried_at)
2. Add mldag/provenance/history_enrich.py: discover cluster_ids from the events table (json_extract on payload_json) that aren't already in condor_history, batch-query them via htcondor2.Schedd().history(), insert results, count not-found cluster_ids without erroring
3. Lazy-import htcondor2 only inside the query function (module does the top-level import like daggen.py, but query.py's CLI wiring imports history_enrich lazily inside the command body so unrelated query.py commands keep working without htcondor2 installed)
4. Wire 'db enrich-history' into query.py's db_app
5. Unit tests against a mocked Schedd (pytest.importorskip("htcondor2") like test_daggen.py) covering: enrichment inserts rows, already-enriched cluster_ids skipped without full_rescan, full_rescan re-queries, cluster_ids with no historical match are skipped silently
6. Run full test suite
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a condor_history table to db.py's schema (cluster_id PK, a fixed set of mapped ClassAd columns, full sanitized classad_json blob, queried_at). Added mldag/provenance/history_enrich.py: enrich_from_condor_history() discovers cluster_ids from the events table via json_extract on payload_json, skips ones already in condor_history unless full_rescan is set, batches the rest into ClusterId==... constraint queries against htcondor2.Schedd().history(), and counts (not errors) any cluster_id condor_history has no record for. A query failure for one batch (schedd unreachable, etc.) is logged and counted, not fatal to the run. Environment is projected only to extract run_id (via post.py's run_id_from_classad) and is never persisted, even in the classad_json blob -- reusing post.py's _SENSITIVE_AD_KEYS blocklist so this can't leak secrets like WANDB_API_KEY into provenance.db. htcondor2 is imported lazily (inside _get_schedd and inside query.py's CLI command body), matching the pattern in test_daggen.py, so the rest of mldag-query keeps working without it installed. Wired as 'mldag-query db enrich-history'. 7 new tests in tests/provenance/test_history_enrich.py against a mocked Schedd (pytest.importorskip("htcondor2"), so they run for real in CI on Linux and skip locally on this Mac dev machine -- verified locally anyway with a throwaway htcondor2 stub module on PYTHONPATH). Full suite: 268 passed, 2 skipped; ruff clean; ty clean except the pre-existing/expected unresolved-import for htcondor2 (same as daggen.py) and two pre-existing build_database diagnostics in query.py unrelated to this change.
<!-- SECTION:NOTES:END -->
