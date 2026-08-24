---
id: TASK-34
title: Add per-batch timing and tunable batch size to enrich-history
status: Done
assignee:
  - claude
created_date: '2026-08-24 19:56'
updated_date: '2026-08-24 19:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
condor_history's Schedd.history() scans its backing store sequentially per query regardless of constraint complexity, so batching cluster_ids into many small queries multiplies total scan work instead of reducing it. A user hit this directly: enrich-history was very slow with no visibility into why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each batch's progress message reports how long that schedd.history() call took
- [x] #2 batch_size is exposed as --batch-size on mldag-query db enrich-history so it can be tuned without a code change
- [x] #3 Default batch size raised from 50 to 500, since fewer/larger batches means less total scanning, not more
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added per-batch timing (time.monotonic() around each schedd.history() call, reported in the on_progress message) and a tunable --batch-size CLI option (also exposed on the library function, already had a batch_size param). Raised the default from 50 to 500. Verified locally against a stub Schedd that this doesn't change behavior, only visibility/tunability. Full suite: 278 passed, 2 skipped; ruff/ty clean.
<!-- SECTION:NOTES:END -->
