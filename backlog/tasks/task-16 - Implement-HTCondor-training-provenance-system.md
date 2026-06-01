---
id: task-16
title: Implement HTCondor training provenance system
status: Done
assignee: []
created_date: '2026-04-28 20:49'
updated_date: '2026-04-29 00:45'
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

---

### Where metadata is created and where it goes

Two concerns: what is written, and where. The answer to "are we slogging JSON files all over the place" is no — there are exactly two output locations per run.

#### When each piece is emitted

```
Access Point                    Execute Node                    Access Point
(DAG generation)                (inside pretrain.sh)            (SCRIPT POST)
      |                                |                               |
      |                                |                               |
  job.submitted                   [job starts]                  [job exits]
  ─────────────               site_info.json written          reads job ClassAd
  written to:                  job.assigned emitted            job.completed or
  run_id.ndjson                ─────────────────────           job.failed emitted
                               written to:                     ─────────────────
                               run_id.ndjson                   written to:
                                                               run_id.ndjson
                                    │
                               [each epoch]
                               epoch.started emitted
                               ... training ...
                               checkpoint saved
                               epoch.completed emitted
                               .provenance.json written
                               ──────────────────────────
                               events  → run_id.ndjson
                               sidecar → alongside .ckpt
```

#### Where the files land

```
shared_filesystem/
│
├── provenance/
│   └── a3f9cc12.ndjson        ← ONE file per run, all events append here
│                                 ~2 KB per epoch; a 30-epoch run ≈ 60 KB total
│
└── training_logs/
    └── a3f9cc12/
        └── checkpoints/
            ├── epoch=0-step=500-val_loss=0.52.ckpt
            ├── epoch=0-step=500-val_loss=0.52.ckpt.provenance.json  ← sidecar
            ├── epoch=1-step=1000-val_loss=0.48.ckpt
            ├── epoch=1-step=1000-val_loss=0.48.ckpt.provenance.json ← sidecar
            └── ...
```

**New files per run: 1 NDJSON + 1 sidecar per checkpoint.** The sidecars live exactly where the checkpoints already live. The NDJSON lives in one central `provenance/` directory. Nothing appears in unexpected places; nothing is written to the execute node's local disk after the job exits.

#### What each file contains

The NDJSON is a stream of lifecycle events — one line each, all for the same run:

```jsonc
{"schema_version":"1.0","type":"job.submitted","run_id":"a3f9cc12","ts":"...","hyperparams_hash":"cc39f1","code_commit":"f4a88bc"}
{"schema_version":"1.0","type":"job.assigned","run_id":"a3f9cc12","ts":"...","site":"chtc-gpu04","hostname":"gpu04.chtc.wisc.edu","gpu":"NVIDIA H200","cuda":"12.2"}
{"schema_version":"1.0","type":"epoch.started","run_id":"a3f9cc12","ts":"...","epoch":0,"checkpoint_in_hash":null}
{"schema_version":"1.0","type":"epoch.completed","run_id":"a3f9cc12","ts":"...","epoch":0,"checkpoint_out_hash":"sha256:8bc1f3","loss":0.52,"duration_s":31240}
{"schema_version":"1.0","type":"job.completed","run_id":"a3f9cc12","ts":"...","wall_time_s":31580,"cpu_usage":3.8,"gpu_usage":0.94,"peak_memory_mb":38400}
```

The sidecar is a self-contained record that makes the checkpoint file auditable in isolation:

```jsonc
{
  "schema_version": "1.0",
  "checkpoint_hash": "sha256:8bc1f3...",
  "parent_hash": null,
  "run_id": "a3f9cc12",
  "epoch": 0,
  "produced_at": {"site": "chtc-gpu04", "hostname": "gpu04.chtc.wisc.edu", "ts": "..."},
  "environment": {"python": "3.11.4", "cuda": "12.2", "framework_version": "2.3.0", "code_commit": "f4a88bc"},
  "training": {"loss": 0.52, "epoch_duration_s": 31240}
}
```

The `parent_hash` field chains sidecars together across epochs and sites. Walking it backward from any checkpoint reconstructs the full training lineage without consulting the NDJSON log or any other system.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A complete provenance record exists for any training run: run ID, submission metadata, per-epoch site and environment, checkpoint hashes, and resource usage
- [ ] #2 The full training history of any checkpoint is reconstructable by walking the sidecar chain without consulting any external system
- [ ] #3 All lifecycle events (submitted, assigned, epoch started/completed, completed/failed) are queryable by run ID from the NDJSON event log
- [ ] #4 Resource usage (wall time, CPU, GPU, memory) is captured for every completed job via HTCondor ClassAds
- [ ] #5 The system works across job migrations and retries — the run ID threads through all attempts
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All subtasks complete: task-9 (PROVENANCE_RUN_ID in daggen), task-10 (site fingerprint in pretrain.sh), task-11 (sidecar writer), task-12 (NDJSON event emitter), task-13 (POST script + job.submitted), task-14 (checkpoint watcher), task-15 (lineage/events CLI). New modules: mldag/provenance/{__init__,events,sidecar,post,watcher,query}.py. New scripts: provenance_post.sh. 72 tests, all passing.
<!-- SECTION:NOTES:END -->
