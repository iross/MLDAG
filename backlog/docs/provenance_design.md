# HTCondor Training Provenance — Design Plan

## Overview

Two complementary systems that together give full lineage tracking for multi-site training jobs:

1. **Structured Provenance Events** — a streaming operational record of what happened, when, and where
2. **Checkpoint Sidecar Manifests** — a self-contained archival record attached to each checkpoint file

These are linked by `checkpoint_hash`, making them cross-referenceable without a shared database.

---

## System 1: Structured Provenance Events

### Goal
Replace free-form log lines with discrete, schema'd events emitted at lifecycle boundaries and stored in a queryable format.

### Event Taxonomy

| Event Type | Key Fields |
|---|---|
| `job.submitted` | `run_id`, `submit_time`, `hyperparams`, `data_version`, `code_commit` |
| `job.assigned` | `run_id`, `site`, `slot`, `site_fingerprint` |
| `epoch.started` | `run_id`, `epoch_num`, `checkpoint_in_hash`, `wall_time` |
| `epoch.completed` | `run_id`, `epoch_num`, `checkpoint_out_hash`, `loss`, `duration_s` |
| `job.migrated` | `run_id`, `from_site`, `to_site`, `reason`, `checkpoint_hash` |
| `job.failed` | `run_id`, `site`, `epoch_num`, `error_class`, `message` |
| `job.completed` | `run_id`, `final_checkpoint_hash`, `total_epochs`, `total_wall_time_s` |

All events share a common envelope:
```json
{
  "schema_version": "1.0",
  "type": "<event_type>",
  "run_id": "<uuid>",
  "ts": "<iso8601>",
  ...event-specific fields
}
```

### Emission Points

- **HTCondor wrapper/submit scripts** → `job.submitted`, `job.assigned`, `job.migrated`, `job.completed`, `job.failed`
- **Epoch callback or hook in training loop** → `epoch.started`, `epoch.completed`
- **Fallback**: a file-watcher on the checkpoint output directory can approximate epoch events if the training code cannot be modified

### Run ID Threading

A UUID is generated at submit time and injected as an environment variable (`PROVENANCE_RUN_ID`). Every event, log line, checkpoint, and sidecar carries this ID. When a job migrates, the ID follows it — the full multi-site history of any run is reconstructable by querying on this single field.

### Storage

**Start with NDJSON** (newline-delimited JSON, one event per line, one file per run). This is:
- Immediately human-readable and grepable
- Trivially loadable into pandas, DuckDB, or SQLite
- Zero-infrastructure to adopt

Example record:
```json
{"schema_version": "1.0", "type": "epoch.completed", "run_id": "a3f9cc12-...", "epoch": 12, "site": "chtc-gpu04", "loss": 0.342, "checkpoint_hash": "sha256:8bc1f3...", "duration_s": 847, "ts": "2026-04-27T14:32:01Z"}
```

Migrate to SQLite or Postgres later if query volume warrants it — the schema is stable either way.

### Key Queries Unlocked

- Which sites handled epochs 40–60 of run X?
- Did any job migrate mid-run this week? Why?
- What was the loss trajectory across a multi-site run?
- Which runs touched a specific site on a given day?
- Which runs are currently in-flight?

---

## System 2: Checkpoint Sidecar Manifests

### Goal
Every checkpoint file gets a companion `.provenance.json` that makes it fully self-describing — complete history without consulting any external system.

### Sidecar Schema

```json
{
  "schema_version": "1.0",
  "checkpoint_hash": "sha256:8bc1f3...",
  "parent_hash": "sha256:a92d44...",
  "run_id": "a3f9cc12-...",
  "epoch": 12,
  "produced_at": {
    "site": "chtc-gpu04",
    "hostname": "gpu04.chtc.wisc.edu",
    "slot": "slot1_1",
    "ts": "2026-04-27T14:32:01Z"
  },
  "environment": {
    "python": "3.11.4",
    "cuda": "12.2",
    "framework": "pytorch",
    "framework_version": "2.3.0",
    "code_commit": "f4a88bc"
  },
  "inputs": {
    "data_version": "v2.1",
    "hyperparams_hash": "cc39f1..."
  },
  "training": {
    "loss": 0.342,
    "epoch_duration_s": 847
  }
}
```

### The Lineage Chain

`parent_hash` points to the sidecar of the checkpoint this one was trained from. This creates a **linked list across all checkpoints**, spanning sites and interrupted runs. Walking the chain backward from any checkpoint reconstructs its full provenance without a database.

### Sidecar Generation

A wrapper script runs after the training framework saves each checkpoint:

```python
def write_sidecar(checkpoint_path, run_id, epoch, parent_hash, site_info, env_info, training_metrics):
    checkpoint_hash = sha256_file(checkpoint_path)
    sidecar = {
        "schema_version": "1.0",
        "checkpoint_hash": f"sha256:{checkpoint_hash}",
        "parent_hash": parent_hash,
        "run_id": run_id,
        "epoch": epoch,
        "produced_at": site_info,
        "environment": env_info,
        "training": training_metrics,
    }
    sidecar_path = Path(str(checkpoint_path) + ".provenance.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return checkpoint_hash  # becomes parent_hash for the next epoch
```

`site_info` and `env_info` are captured once at job startup (hostname, CUDA version, Python version, code commit, etc.) and reused for all epochs in that job.

### Lineage Reconstruction

```python
def load_lineage(checkpoint_path, checkpoint_store):
    """Walk the parent_hash chain to reconstruct full training history."""
    chain = []
    current = checkpoint_path
    while current:
        sidecar = json.loads(Path(str(current) + ".provenance.json").read_text())
        chain.append(sidecar)
        parent_hash = sidecar.get("parent_hash")
        current = checkpoint_store.find_by_hash(parent_hash) if parent_hash else None
    return list(reversed(chain))
```

Output: ordered list of sidecar records from epoch 0 to present, showing every site the model passed through.

---

## Integration Points

### HTCondor Pre/Post Scripts

```
# In the .sub file:
+PreCmd = "provenance_pre.sh"
+PostCmd = "provenance_post.sh"
```

- `provenance_pre.sh` — captures site fingerprint, emits `job.assigned`, sets `PROVENANCE_RUN_ID` in the job environment
- `provenance_post.sh` — emits `job.completed` or `job.failed`, records final checkpoint hash

### HTCondor Job Wrapper

The submit-side wrapper generates the `run_id` UUID, injects it and other metadata into the ClassAd or environment, and emits `job.submitted`.

### Cross-System Link

Both systems share `checkpoint_hash` and `run_id` as join keys:
- An `epoch.completed` event references `checkpoint_hash`
- The sidecar *is* the artifact with that hash

You can navigate from event → sidecar or sidecar → event history at any time.

---

## Schema Versioning

- Both systems carry `schema_version` as a top-level field
- Version the schema alongside training code in the same repo
- Old records remain parseable by checking `schema_version` before deserializing
- Breaking changes bump the major version; additive fields bump the minor version

---

## Rollout Order

1. **Run ID threading** — lowest friction, immediate cross-system traceability
2. **Site fingerprint capture** — one script, runs at job start, no training code changes
3. **Sidecar writer** — post-checkpoint hook, self-contained
4. **NDJSON event log** — wire up submit/pre/post scripts first, epoch events second
5. **Lineage query tooling** — build once events and sidecars are flowing
6. **Dashboard / live view** — optional, build on top of the event log

---

## Open Questions for Implementation

- Where does the NDJSON event log live? (shared filesystem, object store, central collector service?)
- How are checkpoints addressed for lineage walk — by path on a shared FS, or by hash lookup in an index?
- Should `job.migrated` be emitted by HTCondor's eviction hook, or reconstructed from consecutive `job.assigned` events for the same `run_id`?
- Is there an existing metadata store in the framework this should write into, or is this greenfield?
