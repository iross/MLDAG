---
id: TASK-21
title: Add SQLite database for querying checkpoint and provenance data
status: To Do
assignee: []
created_date: '2026-08-05 16:09'
updated_date: '2026-08-05 16:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Checkpoint lineage (checkpoint_prov/*.ckpt.provenance.json) and event history (provenance/*.ndjson) are only queryable by reading individual files by hand or one run at a time via mldag-query. There's no way to ask cross-run questions (best val_loss per run, epoch counts, duration trends) without a throwaway script. A local SQLite database built from the existing artifacts makes this data directly queryable with SQL while the NDJSON/sidecar files stay the source of truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A build command loads checkpoint sidecar files into a SQLite database without modifying source files
- [ ] #2 The database has one row per checkpoint sidecar with run_id/epoch/hashes/environment fields, and full training metrics are accessible
- [ ] #3 A build command loads NDJSON event log files into the same database with run_id/type/timestamp and event-specific fields accessible
- [ ] #4 Re-running the build command against unchanged source data does not create duplicate rows
- [ ] #5 Malformed or unparseable source files, including known-bad unknown:<cluster_id>.ndjson files from task-20, are skipped with a warning rather than aborting the whole build
- [ ] #6 Documented example queries answer at minimum: best val_loss per run, epoch counts per run, and checkpoint lineage/duration lookups
- [ ] #7 Unit tests cover the loader against fixture sidecar/ndjson data, including malformed and duplicate-event cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Data reality check (informs the design below)

- `checkpoint_prov/`: 1651 sidecar files, 72 distinct `run_id`s, all `schema_version: "1.0"`, all with a non-empty `training` block. The `training` dict has 61 distinct keys across the corpus (Lightning `test/pearson_*` metrics etc.) and only 4 keys (`train_loss_step`, `val_loss`, `train_loss_epoch`, `duration_s`) appear in >=90% of files — the rest are long-tail/sparse. This means `training` cannot be flattened into fixed columns; it must stay as a JSON blob column with the well-known fields also hoisted into their own columns for indexing/filtering.
- `provenance/*.ndjson`: confirms task-20's bug empirically. Run `afce8935` has 4 rows for `epoch: 12` (2x `epoch.started`, 2x `epoch.completed` with different `checkpoint_out_hash` values) from a resumed job re-scanning the checkpoint directory. The loader must not assume `(run_id, epoch, type)` is unique.
- There are two overlapping-but-not-identical provenance directories at the repo root: `provenance/` (1844 files) and `output/provenance/` (1847 files), differing by a handful of files. This is leftover from a path convention change, not something this task should silently paper over — the build command takes explicit `--event-dir` (repeatable) so the user picks the source(s), and a follow-up cleanup task should reconcile/retire one of the two directories.
- ~540 of the ndjson files are named `unknown:<cluster_id>.ndjson` (the task-20 bug) — these have a real `run_id` field inside each JSON line (it's the envelope's `run_id`, independent of filename), so the loader keys rows by the in-record `run_id`, not the filename. This means most `unknown:*` events actually land under their correct run once loaded, even before task-20's underlying bug is fixed. Files where the JSON itself fails to parse are skipped with a warning.

### Schema (SQLite)

```sql
CREATE TABLE checkpoints (
    checkpoint_path   TEXT PRIMARY KEY,   -- absolute or repo-relative path to the .ckpt (sidecar minus suffix)
    run_id            TEXT NOT NULL,
    epoch             INTEGER,
    checkpoint_hash   TEXT,
    parent_hash       TEXT,
    schema_version    TEXT,
    hostname          TEXT,
    slot              TEXT,
    gpu_model         TEXT,
    gpu_count         INTEGER,
    gpu_id            TEXT,
    produced_at_ts    TEXT,
    python            TEXT,
    cuda              TEXT,
    code_commit       TEXT,
    mldag_version     TEXT,
    val_loss          REAL,
    train_loss_step   REAL,
    train_loss_epoch  REAL,
    duration_s        REAL,
    training_json     TEXT,   -- full `training` dict, json-encoded, for the long-tail metrics
    source_file       TEXT NOT NULL,
    source_mtime      REAL NOT NULL,
    ingested_at       TEXT NOT NULL
);
CREATE INDEX idx_checkpoints_run_epoch ON checkpoints(run_id, epoch);
CREATE INDEX idx_checkpoints_hash ON checkpoints(checkpoint_hash);

CREATE TABLE events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    type           TEXT NOT NULL,
    ts             TEXT NOT NULL,
    epoch          INTEGER,          -- NULL for non-epoch events
    payload_json   TEXT NOT NULL,    -- full event dict as-parsed, including envelope
    source_file    TEXT NOT NULL,
    source_line    INTEGER NOT NULL,
    ingested_at    TEXT NOT NULL,
    UNIQUE(source_file, source_line)  -- re-running the build is idempotent per physical line
);
CREATE INDEX idx_events_run_type ON events(run_id, type);
CREATE INDEX idx_events_run_epoch ON events(run_id, epoch);
```

No unique constraint on `(run_id, epoch, type)` in `events` — duplicates from the task-20 bug are real signal, not loader noise. A documented query (`SELECT run_id, epoch, type, COUNT(*) ... HAVING COUNT(*) > 1`) surfaces them instead of hiding them.

### Module layout

New module `mldag/provenance/db.py`:
- `build_database(db_path, checkpoint_dirs, event_dirs, *, incremental=True) -> BuildStats` — creates the schema if absent, walks `checkpoint_dirs` for `*.ckpt.provenance.json` and `event_dirs` for `*.ndjson`, upserts rows.
- Checkpoints: keyed by `checkpoint_path` (`INSERT OR REPLACE`), skipped if `source_mtime` unchanged since last ingest (fast re-run).
- Events: keyed by `(source_file, source_line)` (`INSERT OR IGNORE`), so appended lines in a growing `.ndjson` are picked up on the next build without re-reading already-ingested lines (track per-file byte offset the same way `log_monitor.py` already does, reusing that pattern rather than inventing a new one).
- Parse errors (bad JSON, missing required fields) are caught per-file/per-line, logged via `logging.warning`, and counted in `BuildStats`; they never abort the whole build.
- CLI: extend the existing `mldag-query` Typer app (`mldag/provenance/query.py`) with a `db` sub-app: `mldag-query db build [--db provenance.db] [--checkpoint-dir DIR ...] [--event-dir DIR ...] [--full-rescan]`. Defaults: `--checkpoint-dir checkpoint_prov`, `--event-dir output/provenance`. No new console-script entry point needed.

### Non-goals (explicitly out of scope for this task)

- Fixing the task-20 root cause (duplicate/overwritten epoch events, `unknown:*` filenames). This task only needs to tolerate that data, not repair it.
- Reconciling the `provenance/` vs `output/provenance/` directory split — flagged above as a separate cleanup task.
- Any live/streaming ingestion, web UI, or dashboard. This is a batch/on-demand build command producing a local `.db` file, analogous to how `mldag-query` already reads files on demand.
- Backfilling historical `.run_id`/cache files or anything else from `log_monitor.py`'s state — those are inputs to *producing* the ndjson files, not the ndjson files themselves.

### Testing approach

- Fixture sidecar and ndjson files under `tests/provenance/fixtures/` (small, hand-written, covering: normal checkpoint, checkpoint missing optional fields, duplicate epoch events with different hashes, a truncated/corrupt JSON line, an `unknown:<cluster_id>.ndjson` file).
- Unit tests in `tests/provenance/test_db.py`: schema creation, checkpoint upsert + idempotent re-run, event append-only ingest + idempotent re-run, malformed-line handling (build succeeds, stats report the skip count), duplicate-epoch data preserved (not deduped).
<!-- SECTION:PLAN:END -->
