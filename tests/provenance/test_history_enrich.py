import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip(
    "htcondor2", reason="htcondor2 is a Linux-only dependency; history_enrich is untestable elsewhere"
)

from mldag.provenance.db import build_database
from mldag.provenance.history_enrich import (
    EnrichStats,
    enrich_from_condor_history,
    write_scan_records,
)


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
    """Stand-in for htcondor2.Schedd; .history() returns pre-canned ClassAd-like dicts.

    A cluster_id maps to a single ad or a list of ads (multiple procs under
    one cluster, e.g. a `queue N` job array).
    """

    def __init__(
        self, ads_by_cluster: dict[int, dict | list[dict]], raise_for: set[int] | None = None
    ):
        self.ads_by_cluster = {
            cid: (ads if isinstance(ads, list) else [ads]) for cid, ads in ads_by_cluster.items()
        }
        self.raise_for = raise_for or set()
        self.history_calls: list[str] = []

    def history(self, constraint: str, projection: list[str], match: int):
        self.history_calls.append(constraint)
        for cid in self.raise_for:
            if f"ClusterId == {cid}" in constraint:
                raise RuntimeError(f"schedd unreachable (simulated) for {cid}")
        result = []
        for cid, ads in self.ads_by_cluster.items():
            if f"ClusterId == {cid}" in constraint:
                result.extend(ads)
        return result


def _ad(cluster_id: int, run_id: str = "run-abc", proc_id: int = 0, **overrides) -> dict:
    ad = {
        "ClusterId": cluster_id,
        "ProcId": proc_id,
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

    (classad_json,) = _query(
        db_path, "SELECT condor_history_json FROM condor_history WHERE cluster_id = 100"
    )[0]
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


# --- job arrays: many procs under one cluster_id ---


def test_job_array_stores_one_row_per_proc(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100])
    schedd = _FakeSchedd({100: [_ad(100, proc_id=0), _ad(100, proc_id=1), _ad(100, proc_id=2)]})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    stats = enrich_from_condor_history(db_path)

    assert stats.enriched == 3
    rows = _query(
        db_path, "SELECT cluster_id, proc_id FROM condor_history ORDER BY proc_id"
    )
    assert rows == [(100, 0), (100, 1), (100, 2)]


def test_second_run_skips_whole_cluster_once_any_proc_is_enriched(tmp_path, monkeypatch):
    """already_enriched is tracked per cluster_id, so a job array isn't re-queried
    proc-by-proc -- one condor_history query for a cluster returns every proc."""
    db_path = _seed_db(tmp_path, [100])
    schedd = _FakeSchedd({100: [_ad(100, proc_id=0), _ad(100, proc_id=1)]})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)

    stats = enrich_from_condor_history(db_path)

    assert stats.already_enriched == 1
    assert len(schedd.history_calls) == 1


# --- source/status tagging ---


def test_condor_history_rows_tagged_source_and_status(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [100])
    schedd = _FakeSchedd({100: _ad(100, JobStatus=5)})  # 5 = Held
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )

    enrich_from_condor_history(db_path)

    rows = _query(db_path, "SELECT source, status FROM condor_history WHERE cluster_id = 100")
    assert rows == [("condor_history", "held")]


# --- write_scan_records: event_log_scan.py output into the same table ---


def test_write_scan_records_inserts_with_event_log_source(tmp_path):
    db_path = tmp_path / "provenance.db"
    build_database(db_path, [], [])
    scan_records = [
        {
            "cluster_id": 500,
            "proc_id": 0,
            "run_id": "run-scan",
            "site": "gpu01.example.edu",
            "execute_host": "slot1_1@gpu01.example.edu",
            "wall_time_s": 1200.0,
            "cpu_usage": 2.0,
            "status": "completed",
        }
    ]

    written = write_scan_records(db_path, scan_records)

    assert written == 1
    rows = _query(
        db_path,
        "SELECT cluster_id, proc_id, run_id, site, remote_host, remote_wall_clock_s, "
        "cpus_usage, status, source FROM condor_history WHERE cluster_id = 500",
    )
    assert rows == [(500, 0, "run-scan", "gpu01.example.edu", "slot1_1@gpu01.example.edu", 1200.0, 2.0, "completed", "event_log")]


def test_write_scan_records_coexists_with_condor_history_rows_for_other_procs(tmp_path, monkeypatch):
    """event_log and condor_history rows for different procs of the same cluster don't collide."""
    db_path = _seed_db(tmp_path, [600])
    schedd = _FakeSchedd({600: _ad(600, proc_id=0)})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)

    write_scan_records(db_path, [{"cluster_id": 600, "proc_id": 1, "status": "held"}])

    rows = _query(
        db_path, "SELECT proc_id, source FROM condor_history WHERE cluster_id = 600 ORDER BY proc_id"
    )
    assert rows == [(0, "condor_history"), (1, "event_log")]


# --- merge semantics: writing from both sources for the SAME (cluster_id, proc_id)
# accumulates fields rather than one write clobbering the other's columns with NULL.


def test_event_log_write_does_not_erase_condor_history_only_fields(tmp_path, monkeypatch):
    db_path = _seed_db(tmp_path, [700])
    schedd = _FakeSchedd({700: _ad(700, Owner="iross", ExitCode=0, RequestCpus=4)})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)

    write_scan_records(
        db_path, [{"cluster_id": 700, "proc_id": 0, "site": "gpu01.example.edu", "status": "completed"}]
    )

    row = _query(
        db_path,
        "SELECT owner, exit_code, request_cpus, site, status, source "
        "FROM condor_history WHERE cluster_id = 700 AND proc_id = 0",
    )[0]
    assert row == ("iross", 0, 4, "gpu01.example.edu", "completed", "condor_history,event_log")


def test_condor_history_write_does_not_erase_event_log_only_fields(tmp_path, monkeypatch):
    """Same as above, opposite write order."""
    db_path = _seed_db(tmp_path, [701])
    write_scan_records(
        db_path, [{"cluster_id": 701, "proc_id": 0, "site": "gpu02.example.edu", "job_name": "run0-epoch0"}]
    )

    schedd = _FakeSchedd({701: _ad(701, Owner="iross")})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)

    row = _query(
        db_path,
        "SELECT owner, site, job_name, source FROM condor_history WHERE cluster_id = 701 AND proc_id = 0",
    )[0]
    assert row == ("iross", "gpu02.example.edu", "run0-epoch0", "condor_history,event_log")


def test_second_condor_history_write_updates_changed_fields(tmp_path, monkeypatch):
    """A later condor_history write (e.g. job finished) updates fields it has new data for."""
    db_path = _seed_db(tmp_path, [702])
    schedd = _FakeSchedd({702: _ad(702, JobStatus=2)})  # running
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    enrich_from_condor_history(db_path)

    enrich_from_condor_history(db_path, full_rescan=True)  # same schedd; still JobStatus=2
    schedd.ads_by_cluster[702] = [_ad(702, JobStatus=4)]  # now completed
    enrich_from_condor_history(db_path, full_rescan=True)

    (status,) = _query(db_path, "SELECT status FROM condor_history WHERE cluster_id = 702")[0]
    assert status == "completed"


def test_event_log_only_cluster_is_not_already_enriched_for_condor_history(tmp_path, monkeypatch):
    """A cluster touched only by scan --db was never condor_history-queried."""
    db_path = _seed_db(tmp_path, [703])
    write_scan_records(db_path, [{"cluster_id": 703, "proc_id": 0, "status": "held"}])

    schedd = _FakeSchedd({703: _ad(703)})
    monkeypatch.setattr(
        "mldag.provenance.history_enrich._get_schedd", lambda *a, **k: schedd
    )
    stats = enrich_from_condor_history(db_path)

    assert stats.already_enriched == 0
    assert stats.enriched == 1
    assert len(schedd.history_calls) == 1
