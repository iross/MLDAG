import json
import sqlite3
from pathlib import Path

from mldag.provenance.db import BuildStats, build_database

# The condor_history schema as shipped through v0.1.0rc20: bare cluster_id
# PRIMARY KEY, no source/proc_id/job_name/site/gpu_ids/status columns.
_OLD_CONDOR_HISTORY_SCHEMA = """
CREATE TABLE condor_history (
    cluster_id        INTEGER PRIMARY KEY,
    run_id            TEXT,
    owner             TEXT,
    cmd               TEXT,
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
    resource_name     TEXT,
    hold_reason       TEXT,
    qdate             TEXT,
    job_start_date    TEXT,
    completion_date   TEXT,
    classad_json      TEXT NOT NULL,
    queried_at        TEXT NOT NULL
);
"""


def _write_sidecar(ckpt_dir: Path, name: str, **overrides) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "1.0",
        "checkpoint_hash": f"sha256:{name}",
        "parent_hash": None,
        "run_id": "run-abc",
        "epoch": 0,
        "produced_at": {"hostname": "gpu01", "slot": "slot1_1", "ts": "2026-08-05T00:00:00+00:00"},
        "environment": {"python": "3.9.17", "cuda": "11.6", "mldag_version": "0.1.0rc11"},
        "training": {"val_loss": 12.5, "train_loss_step": 14.0, "duration_s": 100.0},
    }
    data.update(overrides)
    path = ckpt_dir / f"{name}.ckpt.provenance.json"
    path.write_text(json.dumps(data))
    return path


def _write_event(prov_dir: Path, filename: str, events: list[dict]) -> Path:
    prov_dir.mkdir(parents=True, exist_ok=True)
    path = prov_dir / filename
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def _query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# --- checkpoint ingestion ---


def test_build_database_ingests_checkpoint_sidecar(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(ckpt_dir, "epoch=0-step=1-val_loss=12.50")
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [ckpt_dir], [])

    assert stats.checkpoints_ingested == 1
    rows = _query(db_path, "SELECT run_id, epoch, val_loss, hostname FROM checkpoints")
    assert rows == [("run-abc", 0, 12.5, "gpu01")]


def test_build_database_hoists_sparse_training_metrics_into_json_blob(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(
        ckpt_dir, "epoch=0-step=1-val_loss=12.50",
        training={"val_loss": 12.5, "test/pearson_total_score": 0.97},
    )
    db_path = tmp_path / "provenance.db"

    build_database(db_path, [ckpt_dir], [])

    row = _query(db_path, "SELECT training_json FROM checkpoints")[0]
    training = json.loads(row[0])
    assert training["test/pearson_total_score"] == 0.97


def test_build_database_handles_missing_optional_fields(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(
        ckpt_dir, "epoch=0-step=1-val_loss=12.50",
        produced_at={}, environment={}, training={},
    )
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [ckpt_dir], [])

    assert stats.checkpoints_ingested == 1
    row = _query(db_path, "SELECT hostname, val_loss FROM checkpoints")[0]
    assert row == (None, None)


def test_build_database_skips_malformed_checkpoint_json(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "broken.ckpt.provenance.json").write_text("{not valid json")
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [ckpt_dir], [])

    assert stats.checkpoints_ingested == 0
    assert stats.checkpoints_skipped_malformed == 1
    assert _query(db_path, "SELECT COUNT(*) FROM checkpoints") == [(0,)]


def test_build_database_skips_checkpoint_missing_run_id(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "no_run_id.ckpt.provenance.json").write_text(json.dumps({"epoch": 0}))
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [ckpt_dir], [])

    assert stats.checkpoints_skipped_malformed == 1


def test_build_database_finds_nested_checkpoint_sidecars(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    nested = ckpt_dir / "run-abc" / "checkpoints"
    _write_sidecar(nested, "epoch=0-step=1-val_loss=12.50")
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [ckpt_dir], [])

    assert stats.checkpoints_ingested == 1


def test_build_database_checkpoint_rerun_skips_unchanged(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(ckpt_dir, "epoch=0-step=1-val_loss=12.50")
    db_path = tmp_path / "provenance.db"

    build_database(db_path, [ckpt_dir], [])
    stats2 = build_database(db_path, [ckpt_dir], [])

    assert stats2.checkpoints_ingested == 0
    assert stats2.checkpoints_skipped_unchanged == 1
    assert _query(db_path, "SELECT COUNT(*) FROM checkpoints") == [(1,)]


def test_build_database_checkpoint_rerun_reingests_when_content_changes(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    path = _write_sidecar(ckpt_dir, "epoch=0-step=1-val_loss=12.50")
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [ckpt_dir], [])

    import os
    import time
    time.sleep(0.01)
    data = json.loads(path.read_text())
    data["training"]["val_loss"] = 9.0
    path.write_text(json.dumps(data))
    os.utime(path, None)  # ensure a newer mtime even on coarse filesystem clocks

    stats2 = build_database(db_path, [ckpt_dir], [])

    assert stats2.checkpoints_ingested == 1
    row = _query(db_path, "SELECT val_loss FROM checkpoints")[0]
    assert row == (9.0,)


def test_build_database_full_rescan_reingests_unchanged_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(ckpt_dir, "epoch=0-step=1-val_loss=12.50")
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [ckpt_dir], [])

    stats2 = build_database(db_path, [ckpt_dir], [], full_rescan=True)

    assert stats2.checkpoints_ingested == 1
    assert stats2.checkpoints_skipped_unchanged == 0


# --- event ingestion ---


def test_build_database_ingests_ndjson_events(tmp_path):
    prov_dir = tmp_path / "provenance"
    _write_event(prov_dir, "run-abc.ndjson", [
        {"type": "job.submitted", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00", "job_name": "j0"},
        {"type": "epoch.started", "run_id": "run-abc", "ts": "2026-08-05T00:01:00+00:00", "epoch": 0},
    ])
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [], [prov_dir])

    assert stats.events_ingested == 2
    rows = _query(db_path, "SELECT type, epoch FROM events ORDER BY id")
    assert rows == [("job.submitted", None), ("epoch.started", 0)]


def test_build_database_loads_unknown_cluster_id_files(tmp_path):
    """unknown:<cluster_id>.ndjson files load fine -- keyed by the embedded run_id, not filename."""
    prov_dir = tmp_path / "provenance"
    _write_event(prov_dir, "unknown:5772633.ndjson", [
        {"type": "job.executing", "run_id": "unknown:5772633", "ts": "2026-07-23T06:32:31+00:00"},
    ])
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [], [prov_dir])

    assert stats.events_ingested == 1
    rows = _query(db_path, "SELECT run_id, type FROM events")
    assert rows == [("unknown:5772633", "job.executing")]


def test_build_database_skips_malformed_event_line(tmp_path):
    prov_dir = tmp_path / "provenance"
    prov_dir.mkdir()
    (prov_dir / "run-abc.ndjson").write_text(
        json.dumps({"type": "job.submitted", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00"}) + "\n"
        "{not valid json\n"
    )
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [], [prov_dir])

    assert stats.events_ingested == 1
    assert stats.events_skipped_malformed == 1


def test_build_database_skips_event_missing_required_field(tmp_path):
    prov_dir = tmp_path / "provenance"
    _write_event(prov_dir, "run-abc.ndjson", [{"type": "job.submitted", "run_id": "run-abc"}])  # no ts
    db_path = tmp_path / "provenance.db"

    stats = build_database(db_path, [], [prov_dir])

    assert stats.events_ingested == 0
    assert stats.events_skipped_malformed == 1


def test_build_database_preserves_duplicate_epoch_events(tmp_path):
    """Duplicate epoch.started/completed pairs (task-20/22's resumed-job quirk) are kept, not deduped."""
    prov_dir = tmp_path / "provenance"
    _write_event(prov_dir, "run-abc.ndjson", [
        {"type": "epoch.started", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00", "epoch": 12},
        {"type": "epoch.completed", "run_id": "run-abc", "ts": "2026-08-05T00:01:00+00:00", "epoch": 12,
         "checkpoint_out_hash": "sha256:aaa"},
        {"type": "epoch.started", "run_id": "run-abc", "ts": "2026-08-05T00:01:00+00:00", "epoch": 12},
        {"type": "epoch.completed", "run_id": "run-abc", "ts": "2026-08-05T00:02:00+00:00", "epoch": 12,
         "checkpoint_out_hash": "sha256:bbb"},
    ])
    db_path = tmp_path / "provenance.db"

    build_database(db_path, [], [prov_dir])

    rows = _query(
        db_path,
        "SELECT type, COUNT(*) FROM events WHERE epoch = 12 GROUP BY type",
    )
    assert dict(rows) == {"epoch.started": 2, "epoch.completed": 2}


def test_build_database_event_rerun_does_not_duplicate(tmp_path):
    prov_dir = tmp_path / "provenance"
    _write_event(prov_dir, "run-abc.ndjson", [
        {"type": "job.submitted", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00"},
    ])
    db_path = tmp_path / "provenance.db"

    build_database(db_path, [], [prov_dir])
    stats2 = build_database(db_path, [], [prov_dir])

    assert stats2.events_ingested == 0
    assert _query(db_path, "SELECT COUNT(*) FROM events") == [(1,)]


def test_build_database_event_rerun_picks_up_appended_lines(tmp_path):
    prov_dir = tmp_path / "provenance"
    path = _write_event(prov_dir, "run-abc.ndjson", [
        {"type": "job.submitted", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00"},
    ])
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [prov_dir])

    with path.open("a") as f:
        f.write(json.dumps({"type": "job.completed", "run_id": "run-abc", "ts": "2026-08-05T01:00:00+00:00"}) + "\n")

    stats2 = build_database(db_path, [], [prov_dir])

    assert stats2.events_ingested == 1
    assert _query(db_path, "SELECT COUNT(*) FROM events") == [(2,)]


def test_build_database_event_rerun_does_not_reread_unchanged_bytes(tmp_path):
    prov_dir = tmp_path / "provenance"
    path = _write_event(prov_dir, "run-abc.ndjson", [
        {"type": "job.submitted", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00"},
    ])
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [prov_dir])

    read_calls = []
    real_open = open

    def _tracking_open(p, *args, **kwargs):
        read_calls.append(str(p))
        return real_open(p, *args, **kwargs)

    import builtins
    from unittest.mock import patch

    with patch.object(builtins, "open", _tracking_open):
        build_database(db_path, [], [prov_dir])
    assert str(path) not in read_calls


def test_build_database_full_rescan_does_not_duplicate_events(tmp_path):
    prov_dir = tmp_path / "provenance"
    _write_event(prov_dir, "run-abc.ndjson", [
        {"type": "job.submitted", "run_id": "run-abc", "ts": "2026-08-05T00:00:00+00:00"},
    ])
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [prov_dir])

    stats2 = build_database(db_path, [], [prov_dir], full_rescan=True)

    assert stats2.events_ingested == 0  # INSERT OR IGNORE no-ops on the already-seen row
    assert _query(db_path, "SELECT COUNT(*) FROM events") == [(1,)]


# --- example queries (documents + smoke-tests AC #6) ---


def test_example_query_best_val_loss_per_run(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(ckpt_dir, "a", run_id="run-1", epoch=0, training={"val_loss": 12.0})
    _write_sidecar(ckpt_dir, "b", run_id="run-1", epoch=1, training={"val_loss": 8.0})
    _write_sidecar(ckpt_dir, "c", run_id="run-2", epoch=0, training={"val_loss": 20.0})
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [ckpt_dir], [])

    rows = _query(
        db_path,
        "SELECT run_id, MIN(val_loss) FROM checkpoints GROUP BY run_id ORDER BY run_id",
    )
    assert rows == [("run-1", 8.0), ("run-2", 20.0)]


def test_example_query_epoch_count_per_run(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(ckpt_dir, "a", run_id="run-1", epoch=0)
    _write_sidecar(ckpt_dir, "b", run_id="run-1", epoch=1)
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [ckpt_dir], [])

    rows = _query(
        db_path,
        "SELECT run_id, COUNT(DISTINCT epoch) FROM checkpoints GROUP BY run_id",
    )
    assert rows == [("run-1", 2)]


def test_example_query_lineage_by_parent_hash(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    _write_sidecar(ckpt_dir, "a", run_id="run-1", epoch=0, checkpoint_hash="sha256:aaa", parent_hash=None)
    _write_sidecar(ckpt_dir, "b", run_id="run-1", epoch=1, checkpoint_hash="sha256:bbb", parent_hash="sha256:aaa")
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [ckpt_dir], [])

    rows = _query(
        db_path,
        "SELECT epoch, checkpoint_hash, parent_hash FROM checkpoints WHERE run_id = ? ORDER BY epoch",
        ("run-1",),
    )
    assert rows == [(0, "sha256:aaa", None), (1, "sha256:bbb", "sha256:aaa")]


# --- self-healing a stale (pre-v0.1.0rc21) condor_history schema ---


def test_build_database_recreates_stale_condor_history_schema(tmp_path):
    """Regression test: upgrading from <=v0.1.0rc20 must not crash with
    'OperationalError: no such column: source' against an already-built
    provenance.db whose condor_history table predates the current schema."""
    db_path = tmp_path / "provenance.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_OLD_CONDOR_HISTORY_SCHEMA)
    conn.execute(
        "INSERT INTO condor_history (cluster_id, owner, classad_json, queried_at) "
        "VALUES (123, 'iross', '{}', '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    stats = build_database(db_path, [], [])  # must not raise

    assert stats.events_ingested == 0
    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(condor_history)")}
    conn.close()
    assert "source" in columns
    assert "proc_id" in columns
    # The stale row is gone -- it's fully re-derivable by rerunning
    # enrich-history/enrich-jobad/scan --db, which the old row's shape
    # couldn't safely be migrated into anyway (e.g. no proc_id).
    assert _query(db_path, "SELECT COUNT(*) FROM condor_history") == [(0,)]


def test_build_database_leaves_current_condor_history_schema_alone(tmp_path):
    """A condor_history table already on the current schema is left untouched."""
    from mldag.provenance.history_enrich import write_scan_records

    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [])
    write_scan_records(db_path, [{"cluster_id": 500, "proc_id": 0, "status": "held"}])

    build_database(db_path, [], [])  # a second build must not wipe it out

    assert _query(db_path, "SELECT cluster_id, proc_id FROM condor_history") == [(500, 0)]


def test_build_stats_str_reports_all_counts():
    stats = BuildStats(
        checkpoints_ingested=1, checkpoints_skipped_unchanged=2, checkpoints_skipped_malformed=3,
        events_ingested=4, events_skipped_malformed=5,
    )
    s = str(stats)
    assert "1 ingested" in s
    assert "2 unchanged" in s
    assert "3 malformed" in s
    assert "4 ingested" in s
    assert "5 malformed" in s
