import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip(
    "htcondor2", reason="htcondor2 is a Linux-only dependency; history_enrich is untestable elsewhere"
)

from mldag.provenance.db import build_database
from mldag.provenance.history_enrich import EnrichStats, enrich_from_condor_history


def _write_event(prov_dir: Path, filename: str, events: list[dict]) -> Path:
    prov_dir.mkdir(parents=True, exist_ok=True)
    path = prov_dir / filename
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def _seed_db(tmp_path: Path, cluster_ids: list[int]) -> Path:
    """Build a provenance.db whose events table references the given cluster_ids."""
    prov_dir = tmp_path / "provenance"
    events = [
        {"type": "job.executing", "run_id": "run-abc", "ts": "2026-06-01T00:00:00Z", "cluster_id": cid}
        for cid in cluster_ids
    ]
    _write_event(prov_dir, "run-abc.ndjson", events)
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [prov_dir])
    return db_path


def _query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


class _FakeSchedd:
    """Stand-in for htcondor2.Schedd; .history() returns pre-canned ClassAd-like dicts."""

    def __init__(self, ads_by_cluster: dict[int, dict], raise_for: set[int] | None = None):
        self.ads_by_cluster = ads_by_cluster
        self.raise_for = raise_for or set()
        self.history_calls: list[str] = []

    def history(self, constraint: str, projection: list[str], match: int):
        self.history_calls.append(constraint)
        for cid in self.raise_for:
            if str(cid) in constraint:
                raise RuntimeError(f"schedd unreachable (simulated) for {cid}")
        return [
            ad
            for cid, ad in self.ads_by_cluster.items()
            if f"ClusterId == {cid}" in constraint
        ]


def _ad(cluster_id: int, run_id: str = "run-abc", **overrides) -> dict:
    ad = {
        "ClusterId": cluster_id,
        "Owner": "iross",
        "JobStatus": 4,
        "ExitCode": 0,
        "RemoteHost": "slot1_1@node.example.edu",
        "RequestCpus": 4,
        "RemoteWallClockTime": 3600.0,
        "Environment": f"PROVENANCE_RUN_ID={run_id} WANDB_API_KEY=super-secret-value",
    }
    ad.update(overrides)
    return ad


# --- basic enrichment ---


def test_enrich_inserts_rows_for_new_cluster_ids(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100, 200])
    schedd = _FakeSchedd({100: _ad(100), 200: _ad(200)})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    stats = enrich_from_condor_history(db_path)

    assert stats.enriched == 2
    assert stats.not_found == 0
    rows = _query(db_path, "SELECT cluster_id, run_id, owner, exit_code FROM condor_history ORDER BY cluster_id")
    assert rows == [(100, "run-abc", "iross", 0), (200, "run-abc", "iross", 0)]


def test_enrich_no_cluster_ids_in_events_is_a_noop(tmp_path, monkeypatch):
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [])
    schedd = _FakeSchedd({})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    stats = enrich_from_condor_history(db_path)

    assert stats == EnrichStats()
    assert schedd.history_calls == []


# --- secrets handling ---


def test_environment_never_persisted_run_id_extracted(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100])
    schedd = _FakeSchedd({100: _ad(100, run_id="run-xyz")})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    enrich_from_condor_history(db_path)

    (classad_json,) = _query(db_path, "SELECT classad_json FROM condor_history WHERE cluster_id = 100")[0]
    assert "WANDB_API_KEY" not in classad_json
    assert "Environment" not in json.loads(classad_json)
    (run_id,) = _query(db_path, "SELECT run_id FROM condor_history WHERE cluster_id = 100")[0]
    assert run_id == "run-xyz"


# --- not-found vs error handling ---


def test_cluster_id_not_found_in_history_counted_not_error(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100, 200])
    schedd = _FakeSchedd({100: _ad(100)})  # 200 has no historical record
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    stats = enrich_from_condor_history(db_path)

    assert stats.enriched == 1
    assert stats.not_found == 1
    rows = _query(db_path, "SELECT cluster_id FROM condor_history")
    assert rows == [(100,)]


def test_query_error_increments_stat_and_other_batches_still_run(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100, 200])
    schedd = _FakeSchedd({100: _ad(100), 200: _ad(200)}, raise_for={100})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    stats = enrich_from_condor_history(db_path, batch_size=1)

    assert stats.query_errors == 1
    assert stats.enriched == 1
    rows = _query(db_path, "SELECT cluster_id FROM condor_history")
    assert rows == [(200,)]


# --- repeat-run / full_rescan behaviour ---


def test_already_enriched_cluster_ids_skipped_without_full_rescan(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100])
    schedd = _FakeSchedd({100: _ad(100)})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)
    assert len(schedd.history_calls) == 1

    stats = enrich_from_condor_history(db_path)

    assert stats.enriched == 0
    assert stats.already_enriched == 1
    assert len(schedd.history_calls) == 1  # no new query issued


def test_full_rescan_requeries_already_enriched_cluster_ids(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100])
    schedd = _FakeSchedd({100: _ad(100)})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)

    stats = enrich_from_condor_history(db_path, full_rescan=True)

    assert stats.enriched == 1
    assert len(schedd.history_calls) == 2
