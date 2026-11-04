---
id: TASK-32
title: Add progress output to db enrich-history
status: Done
assignee:
  - claude
created_date: '2026-08-24 18:48'
updated_date: '2026-08-24 18:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
enrich_from_condor_history() ran completely silently until it returned, printing only a final summary line. Against an events table with hundreds of cluster_ids batched into many condor_history queries, this gave no sign of life -- and separately, when the target list was empty (e.g. events table not built yet, or pointed at the wrong --db path) the output was an unexplained wall of zeros with no indication why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 enrich_from_condor_history() accepts an optional on_progress callback, invoked before the first query with the total cluster_id/batch count, and after each batch with running enriched/not-found counts
- [x] #2 When there is nothing to enrich, on_progress is told why ("No new cluster_ids to enrich.") instead of the CLI just printing an all-zero summary with no explanation
- [x] #3 mldag-query db enrich-history wires on_progress to typer.echo so progress prints live to the console
- [x] #4 on_progress is optional and defaults to a no-op, so existing callers/tests are unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add on_progress callback param to enrich_from_condor_history(), defaulting to a no-op
2. Call it before the first query (total cluster_id/batch count) and after each batch (running enriched/not-found)
3. Call it with a clear message when target_ids is empty, instead of silently returning all-zero stats
4. Wire on_progress=typer.echo in query.py's db enrich-history command
5. Tests: batches reported, nothing-to-do reported, omitting on_progress doesn't error
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an optional on_progress callback to enrich_from_condor_history(), defaulting to a no-op. Called once up front with the total cluster_id/batch count, once per batch with running enriched/not-found counts, and once with a clear explanation ("No new cluster_ids to enrich.") when there's nothing to do -- this directly addresses the confusing all-zero-stats output from a --db pointing at a db whose events table has nothing enrichable. Wired to typer.echo in mldag-query db enrich-history so it prints live. 3 new tests in test_history_enrich.py (23 total in that file). Full suite: 276 passed, 2 skipped; ruff clean; ty clean except the two pre-existing/unrelated build_database diagnostics and the expected htcondor2 unresolved-import.
<!-- SECTION:NOTES:END -->
