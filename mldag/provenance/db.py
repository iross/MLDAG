"""SQLite database builder for checkpoint sidecars and NDJSON provenance events.

Builds/refreshes a local SQLite database from the existing provenance
artifacts (checkpoint sidecars and NDJSON event logs) so they're queryable
with SQL across runs, without modifying the source files -- they stay the
source of truth. Both malformed data and known-recurring quirks (duplicate
epoch events from a resumed job re-scanning its checkpoint directory,
unknown:<cluster_id> run_id fallbacks) are tolerated rather than treated
as fatal; a build never aborts partway through.

CLI: see `mldag-query db build --help`.

Example queries once built:

    -- Best (lowest) val_loss per run
    SELECT run_id, MIN(val_loss) AS best_val_loss
    FROM checkpoints GROUP BY run_id ORDER BY best_val_loss;

    -- Epoch count per run
    SELECT run_id, COUNT(DISTINCT epoch) AS epochs
    FROM checkpoints GROUP BY run_id ORDER BY epochs DESC;

    -- Checkpoint lineage/duration for one run, oldest first
    SELECT epoch, checkpoint_hash, parent_hash, duration_s
    FROM checkpoints WHERE run_id = ? ORDER BY epoch;

    -- Epochs with duplicate started/completed events (a real data quirk,
    -- not hidden by the loader -- see task-20/task-22)
    SELECT run_id, epoch, type, COUNT(*) AS n
    FROM events WHERE epoch IS NOT NULL
    GROUP BY run_id, epoch, type HAVING COUNT(*) > 1;
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "provenance.db"
DEFAULT_CHECKPOINT_DIR = "checkpoint_prov"
DEFAULT_EVENT_DIR = "output/provenance"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_path   TEXT PRIMARY KEY,
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
    training_json     TEXT,
    source_file       TEXT NOT NULL,
    source_mtime      REAL NOT NULL,
    ingested_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run_epoch ON checkpoints(run_id, epoch);
CREATE INDEX IF NOT EXISTS idx_checkpoints_hash ON checkpoints(checkpoint_hash);

CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    type           TEXT NOT NULL,
    ts             TEXT NOT NULL,
    epoch          INTEGER,
    payload_json   TEXT NOT NULL,
    source_file    TEXT NOT NULL,
    source_line    INTEGER NOT NULL,
    ingested_at    TEXT NOT NULL,
    UNIQUE(source_file, source_line)
);
CREATE INDEX IF NOT EXISTS idx_events_run_type ON events(run_id, type);
CREATE INDEX IF NOT EXISTS idx_events_run_epoch ON events(run_id, epoch);

CREATE TABLE IF NOT EXISTS event_file_state (
    source_file TEXT PRIMARY KEY,
    byte_offset INTEGER NOT NULL,
    line_count  INTEGER NOT NULL
);

-- Per-job summary data, from condor_history (history_enrich.py), a raw
-- event-log scan (event_log_scan.py), or an in-job $_CONDOR_JOB_AD capture
-- mirrored from the `events` table (jobad.py, via history_enrich.py's
-- enrich_from_jobad_events) -- `source` names which have contributed to a
-- row (comma-joined when more than one has). Keyed by (cluster_id,
-- proc_id): a single cluster_id can hold many procs (`queue N` job arrays),
-- so cluster_id alone is not unique (see task-29 -- event_log_scan.py hit
-- the identical bug). A write from one source merges into an existing row
-- column-by-column (COALESCE, new value wins only where the new write
-- actually has one) rather than replacing it outright, so writing from
-- multiple sources for the same job accumulates data instead of one
-- clobbering another's fields with NULL. The jobad source deliberately
-- never contributes remote_wall_clock_s/cpus_usage/memory_usage_mb/
-- gpus_usage: it captures the job ad at submission, before the job has run,
-- so those fields would be near-zero placeholders, not real usage.
CREATE TABLE IF NOT EXISTS condor_history (
    cluster_id        INTEGER NOT NULL,
    proc_id           INTEGER NOT NULL DEFAULT 0,
    run_id            TEXT,
    job_name          TEXT,
    owner             TEXT,
    cmd               TEXT,
    arguments         TEXT,
    job_status        INTEGER,
    exit_code         INTEGER,
    remote_host       TEXT,
    last_remote_host  TEXT,
    request_cpus      INTEGER,
    request_memory    INTEGER,
    request_gpus      INTEGER,
    remote_wall_clock_s REAL,
    cpus_usage        REAL,
    memory_usage_mb   REAL,
    gpus_usage        REAL,
    gpu_ids           TEXT,
    resource_name     TEXT,
    site              TEXT,
    status            TEXT,
    hold_reason       TEXT,
    qdate             TEXT,
    job_start_date    TEXT,
    completion_date   TEXT,
    source            TEXT NOT NULL,
    condor_history_json TEXT,
    event_log_json    TEXT,
    jobad_json        TEXT,
    queried_at        TEXT NOT NULL,
    PRIMARY KEY (cluster_id, proc_id)
);
"""

_CHECKPOINT_COLUMNS = [
    "checkpoint_path", "run_id", "epoch", "checkpoint_hash", "parent_hash",
    "schema_version", "hostname", "slot", "gpu_model", "gpu_count", "gpu_id",
    "produced_at_ts", "python", "cuda", "code_commit", "mldag_version",
    "val_loss", "train_loss_step", "train_loss_epoch", "duration_s",
    "training_json", "source_file", "source_mtime", "ingested_at",
]


@dataclass
class BuildStats:
    """Outcome counts from a build_database() run; str()s into a summary line."""

    checkpoints_ingested: int = 0
    checkpoints_skipped_unchanged: int = 0
    checkpoints_skipped_malformed: int = 0
    events_ingested: int = 0
    events_skipped_malformed: int = 0

    def __str__(self) -> str:
        return (
            f"checkpoints: {self.checkpoints_ingested} ingested, "
            f"{self.checkpoints_skipped_unchanged} unchanged, "
            f"{self.checkpoints_skipped_malformed} malformed | "
            f"events: {self.events_ingested} ingested, "
            f"{self.events_skipped_malformed} malformed"
        )


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _ingest_checkpoint_sidecar(
    conn: sqlite3.Connection,
    sidecar_path: Path,
    *,
    stats: BuildStats,
    now: str,
    full_rescan: bool,
) -> None:
    try:
        mtime = sidecar_path.stat().st_mtime
    except OSError as exc:
        stats.checkpoints_skipped_malformed += 1
        logger.warning("Cannot stat %s: %s", sidecar_path, exc)
        return

    checkpoint_path = str(sidecar_path)[: -len(".provenance.json")]

    if not full_rescan:
        row = conn.execute(
            "SELECT source_mtime FROM checkpoints WHERE checkpoint_path = ?",
            (checkpoint_path,),
        ).fetchone()
        if row is not None and row[0] == mtime:
            stats.checkpoints_skipped_unchanged += 1
            return

    try:
        data = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        stats.checkpoints_skipped_malformed += 1
        logger.warning("Cannot parse %s: %s", sidecar_path, exc)
        return

    if "run_id" not in data:
        stats.checkpoints_skipped_malformed += 1
        logger.warning("%s missing required 'run_id' field, skipping", sidecar_path)
        return

    produced_at = data.get("produced_at") or {}
    environment = data.get("environment") or {}
    training = data.get("training") or {}

    row = {
        "checkpoint_path": checkpoint_path,
        "run_id": data["run_id"],
        "epoch": data.get("epoch"),
        "checkpoint_hash": data.get("checkpoint_hash"),
        "parent_hash": data.get("parent_hash"),
        "schema_version": data.get("schema_version"),
        "hostname": produced_at.get("hostname"),
        "slot": produced_at.get("slot"),
        "gpu_model": produced_at.get("gpu_model"),
        "gpu_count": produced_at.get("gpu_count"),
        "gpu_id": produced_at.get("gpu_id"),
        "produced_at_ts": produced_at.get("ts"),
        "python": environment.get("python"),
        "cuda": environment.get("cuda"),
        "code_commit": environment.get("code_commit"),
        "mldag_version": environment.get("mldag_version"),
        "val_loss": training.get("val_loss"),
        "train_loss_step": training.get("train_loss_step"),
        "train_loss_epoch": training.get("train_loss_epoch"),
        "duration_s": training.get("duration_s"),
        "training_json": json.dumps(training),
        "source_file": str(sidecar_path),
        "source_mtime": mtime,
        "ingested_at": now,
    }
    columns = ", ".join(_CHECKPOINT_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _CHECKPOINT_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO checkpoints ({columns}) VALUES ({placeholders})", row
    )
    stats.checkpoints_ingested += 1


def _ingest_event_file(
    conn: sqlite3.Connection,
    ndjson_path: Path,
    *,
    stats: BuildStats,
    now: str,
    full_rescan: bool,
) -> None:
    key = str(ndjson_path)
    try:
        size = ndjson_path.stat().st_size
    except OSError as exc:
        stats.events_skipped_malformed += 1
        logger.warning("Cannot stat %s: %s", ndjson_path, exc)
        return

    byte_offset, line_count = 0, 0
    if not full_rescan:
        row = conn.execute(
            "SELECT byte_offset, line_count FROM event_file_state WHERE source_file = ?",
            (key,),
        ).fetchone()
        if row is not None:
            byte_offset, line_count = row
            if size < byte_offset:
                byte_offset, line_count = 0, 0  # file was truncated or recreated

    if size == byte_offset:
        return  # nothing new to ingest; skip opening the file entirely

    try:
        with open(ndjson_path, "rb") as f:
            f.seek(byte_offset)
            new_bytes = f.read()
    except OSError as exc:
        stats.events_skipped_malformed += 1
        logger.warning("Cannot read %s: %s", ndjson_path, exc)
        return

    line_no = line_count
    for raw in new_bytes.decode("utf-8", errors="replace").splitlines():
        line_no += 1
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            stats.events_skipped_malformed += 1
            logger.warning("Cannot parse %s line %d: %s", ndjson_path, line_no, exc)
            continue

        run_id = event.get("run_id")
        event_type = event.get("type")
        ts = event.get("ts")
        if not run_id or not event_type or not ts:
            stats.events_skipped_malformed += 1
            logger.warning(
                "%s line %d missing a required field (run_id/type/ts), skipping",
                ndjson_path, line_no,
            )
            continue

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO events
                (run_id, type, ts, epoch, payload_json, source_file, source_line, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, event_type, ts, event.get("epoch"),
                json.dumps(event), key, line_no, now,
            ),
        )
        if cursor.rowcount:
            stats.events_ingested += 1

    conn.execute(
        """
        INSERT INTO event_file_state (source_file, byte_offset, line_count)
        VALUES (?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            byte_offset = excluded.byte_offset, line_count = excluded.line_count
        """,
        (key, byte_offset + len(new_bytes), line_no),
    )


def build_database(
    db_path: str | Path,
    checkpoint_dirs: list[str | Path],
    event_dirs: list[str | Path],
    *,
    full_rescan: bool = False,
) -> BuildStats:
    """Build or refresh db_path from checkpoint sidecars and NDJSON event logs.

    Safe to call repeatedly: unchanged checkpoint sidecars (by mtime) are
    skipped, and event files are read incrementally from the byte offset
    recorded on the previous call, so re-running does not re-parse the
    full provenance directory or create duplicate rows.

    Args:
        db_path: Path to the SQLite database file; created if absent.
        checkpoint_dirs: Directories searched recursively for
            *.ckpt.provenance.json sidecar files.
        event_dirs: Directories searched (non-recursively, matching how
            NDJSON event logs are always written flat) for *.ndjson files.
        full_rescan: Re-ingest every checkpoint and re-read every event
            file from the beginning, ignoring recorded mtimes/offsets.

    Returns:
        BuildStats with counts of what was ingested/skipped/malformed.
    """
    now = datetime.now(timezone.utc).isoformat()
    stats = BuildStats()
    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn)
        for d in checkpoint_dirs:
            for sidecar_path in sorted(Path(d).rglob("*.ckpt.provenance.json")):
                _ingest_checkpoint_sidecar(
                    conn, sidecar_path, stats=stats, now=now, full_rescan=full_rescan
                )
        for d in event_dirs:
            for ndjson_path in sorted(Path(d).glob("*.ndjson")):
                _ingest_event_file(
                    conn, ndjson_path, stats=stats, now=now, full_rescan=full_rescan
                )
        conn.commit()
    finally:
        conn.close()
    return stats
