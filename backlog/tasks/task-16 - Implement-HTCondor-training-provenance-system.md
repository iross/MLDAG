---
id: task-16
title: Implement HTCondor training provenance system
status: To Do
assignee: []
created_date: '2026-04-28 20:49'
labels: []
dependencies:
  - task-9
  - task-10
  - task-11
  - task-12
  - task-13
  - task-14
  - task-15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
End-to-end lineage tracking for multi-site training jobs, implemented as structured events and checkpoint sidecars rather than post-hoc log scraping.

### The problem with log scraping

The current approach reconstructs training history by parsing HTCondor event logs and ClassAds after the fact. This works for operational questions ("is the DAG still running?") but fails for lineage questions, for three reasons.

**It attributes credit to the wrong job.** HTCondor marks a job "successful" when it exits zero — not when it does real training work. The run9/epoch3 investigation (`analysis/analyze_run9_epoch3.py`) required a bespoke 80-line script to discover that the job DAGMan credited with success (cluster 12711655) loaded an existing checkpoint and ran validation for 3 hours at 8% GPU utilization, while the job that actually trained the epoch (cluster 12704651) was evicted and marked failed. This ambiguity is unresolvable from logs alone because logs record job lifecycle events, not artifact production events. Only a content-addressed `checkpoint_out_hash` in the epoch record proves which compute event produced which artifact.

**It cannot follow a checkpoint across sites.** When a job migrates — evicted from OSPool, restarted on Expanse, checkpointed again — its history is split across event logs from different systems with no shared key. Correlation by job name and timing is fragile and breaks under retries. The `analysis/` directory contains ~15 one-off scripts that exist because these cross-site questions could not be answered directly. A single `run_id` threaded through every event and every sidecar makes the full multi-site history a single query.

**It does not travel with the artifact.** A checkpoint file produced today is opaque in six months: which run, which epoch, which GPU, which data version, which code commit. Answering those questions requires the original logs to still exist and be accessible. A sidecar written alongside the checkpoint is self-describing forever — the lineage is embedded in the artifact itself, not in a system that may be gone.

### What structured provenance adds

Structured events (NDJSON, one per lifecycle boundary) and checkpoint sidecars (`.provenance.json` co-located with each `.ckpt`) together answer questions that log scraping cannot:

- Which job produced checkpoint X? (sidecar `checkpoint_hash` is content-addressed)
- What was the full training path of this checkpoint? (walk `parent_hash` chain backward)
- Which sites handled epochs 40–60 of run Y? (query event log by `run_id` and `epoch_num`)
- Did any job this week claim success without doing real training? (compare `epoch.started` → `epoch.completed` pairs against checkpoint hashes)
- What were the actual GPU hours consumed by a run, accounting for evictions? (sum `wall_time_s` from `job.completed` events, not job durations)

The implementation is deliberately zero-infrastructure: NDJSON files live next to DAG output, sidecars live next to checkpoints. No collector service, no database, no credentials. The event log is greppable and DuckDB-queryable immediately. If query volume later warrants a database, the migration is trivial — the schema is stable and versioned from day one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A complete provenance record exists for any training run: run ID, submission metadata, per-epoch site and environment, checkpoint hashes, and resource usage
- [ ] The full training history of any checkpoint is reconstructable by walking the sidecar chain without consulting any external system
- [ ] All lifecycle events (submitted, assigned, epoch started/completed, completed/failed) are queryable by run ID from the NDJSON event log
- [ ] Resource usage (wall time, CPU, GPU, memory) is captured for every completed job via HTCondor ClassAds
- [ ] The system works across job migrations and retries — the run ID threads through all attempts
<!-- AC:END -->
