"""Tests for the standalone HTCondor event log scanner (no DAGMan instrumentation required)."""

import json
from pathlib import Path

from mldag.provenance.event_log_scan import scan_event_log
from mldag.provenance.log_monitor import site_from_slotname


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def _write_ndjson(prov_dir: Path, run_id: str, job_name: str) -> None:
    prov_dir.mkdir(parents=True, exist_ok=True)
    event = {"type": "job.submitted", "run_id": run_id, "job_name": job_name}
    (prov_dir / f"{run_id}.ndjson").write_text(json.dumps(event) + "\n")


# --- site_from_slotname ---


def test_site_from_slotname_plain_host():
    assert site_from_slotname("slot1_1@ip-172-31-31-209.ec2.internal") == "ip-172-31-31-209.ec2.internal"


def test_site_from_slotname_glidein_double_at():
    slotname = "slot1_9@glidein_2498702_695587462@spark-agpu225.chtc.wisc.edu"
    assert site_from_slotname(slotname) == "spark-agpu225.chtc.wisc.edu"


# --- scan_event_log: no enrichment available (pure event-log-only fallback) ---


def test_scan_pure_fallback_no_enrichment(tmp_path):
    """No log_dir/provenance_log_dir given real data: still get duration + site, keyed by cluster_id."""
    log = tmp_path / "batch.log"
    _write(
        log,
        "001 (999.000.000) 2026-06-01 08:00:00 Job executing on host: <10.0.0.1:1234>\n"
        "\tSlotName: slot1_1@ip-172-31-1-1.ec2.internal\n"
        "...\n"
        "005 (999.000.000) 2026-06-01 09:00:00 Job terminated.\n"
        "\tPartitionable Resources :       Usage  Request Allocated\n"
        "\t   Cpus                 :        2.00        4         4 \n"
        "\t   TimeExecute (s)      :    3600                       \n"
        "...\n",
    )

    records = scan_event_log(log)

    assert len(records) == 1
    r = records[0]
    assert r["cluster_id"] == 999
    assert "run_id" not in r
    assert r["site"] == "ip-172-31-1-1.ec2.internal"
    assert r["execute_host"] == "slot1_1@ip-172-31-1-1.ec2.internal"
    assert r["wall_time_s"] == 3600.0
    assert r["cpu_usage"] == 2.0
    assert r["status"] == "completed"


def test_scan_missing_dirs_do_not_error(tmp_path):
    """Passing nonexistent log_dir/provenance_log_dir is a silent no-op, not an error."""
    log = tmp_path / "batch.log"
    _write(log, "012 (5.000.000) 2026-06-01 08:00:00 Job was held.\n...\n")

    records = scan_event_log(
        log,
        log_dir=tmp_path / "no_such_ads",
        provenance_log_dir=tmp_path / "no_such_provenance",
    )

    assert len(records) == 1
    assert records[0]["cluster_id"] == 5
    assert "run_id" not in records[0]
    assert records[0]["status"] == "held"


# --- scan_event_log: opportunistic enrichment via provenance_log_dir (job_name -> run_id) ---


def test_scan_enriches_run_id_from_provenance_log_dir(tmp_path):
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"
    _write(
        log,
        "000 (5055662.000.000) 2026-06-01 08:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n"
        "001 (5055662.000.000) 2026-06-01 08:05:00 Job executing on host: <10.0.0.1:1234>\n"
        "\tSlotName: slot1_1@node.example.edu\n"
        "...\n",
    )

    records = scan_event_log(log, provenance_log_dir=prov_dir)

    assert len(records) == 1
    r = records[0]
    assert r["job_name"] == "run0-train_epoch0"
    assert r["run_id"] == "run-abc"
    assert r["site"] == "node.example.edu"


def test_scan_no_run_id_when_job_name_not_in_index(tmp_path):
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "some-other-job")
    log = tmp_path / "metl.log"
    _write(
        log,
        "000 (5055662.000.000) 2026-06-01 08:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )

    records = scan_event_log(log, provenance_log_dir=prov_dir)

    assert records[0]["job_name"] == "run0-train_epoch0"
    assert "run_id" not in records[0]


# --- scan_event_log: opportunistic enrichment via log_dir (.run_id marker / .ad classad) ---


def test_scan_enriches_run_id_from_run_id_marker(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    (ad_dir / "42.run_id").write_text("run-marked")
    log = tmp_path / "metl.log"
    _write(log, "012 (42.000.000) 2026-06-01 08:00:00 Job was held.\n...\n")

    records = scan_event_log(log, log_dir=ad_dir)

    assert records[0]["run_id"] == "run-marked"


def test_scan_enriches_resource_name_from_classad(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    (ad_dir / "42.ad").write_text(
        'Environment = "PROVENANCE_RUN_ID=run-from-ad OTHER=val"\n'
        'GLIDEIN_ResourceName = "Expanse"\n'
    )
    log = tmp_path / "metl.log"
    _write(log, "012 (42.000.000) 2026-06-01 08:00:00 Job was held.\n...\n")

    records = scan_event_log(log, log_dir=ad_dir)

    assert records[0]["run_id"] == "run-from-ad"
    assert records[0]["resource_name"] == "Expanse"


def test_scan_unresolved_classad_does_not_set_unknown_run_id(tmp_path):
    """No .run_id/.ad match for the cluster_id: no run_id field at all (not "unknown:...")."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    log = tmp_path / "metl.log"
    _write(log, "012 (777.000.000) 2026-06-01 08:00:00 Job was held.\n...\n")

    records = scan_event_log(log, log_dir=ad_dir)

    assert "run_id" not in records[0]


# --- scan_event_log: full lifecycle / status derivation ---


def test_scan_status_aborted(tmp_path):
    log = tmp_path / "metl.log"
    _write(log, "009 (1.000.000) 2026-06-01 08:00:00 Job was aborted.\n...\n")
    records = scan_event_log(log)
    assert records[0]["status"] == "aborted"


def test_scan_status_executing(tmp_path):
    log = tmp_path / "metl.log"
    _write(log, "001 (1.000.000) 2026-06-01 08:00:00 Job executing on host: <1.2.3.4:1>\n...\n")
    records = scan_event_log(log)
    assert records[0]["status"] == "executing"


def test_scan_status_held_then_released_is_not_held(tmp_path):
    log = tmp_path / "metl.log"
    _write(
        log,
        "012 (1.000.000) 2026-06-01 08:00:00 Job was held.\n...\n"
        "013 (1.000.000) 2026-06-01 08:05:00 Job was released.\n...\n",
    )
    records = scan_event_log(log)
    assert records[0]["status"] != "held"


def test_scan_multiple_clusters_sorted_by_cluster_id(tmp_path):
    log = tmp_path / "metl.log"
    _write(
        log,
        "012 (200.000.000) 2026-06-01 08:00:00 Job was held.\n...\n"
        "012 (100.000.000) 2026-06-01 08:00:00 Job was held.\n...\n",
    )
    records = scan_event_log(log)
    assert [r["cluster_id"] for r in records] == [100, 200]


def test_scan_missing_terminator_still_flushes_last_block(tmp_path):
    """A block with no trailing '...' (e.g. log truncated mid-write) is still captured."""
    log = tmp_path / "metl.log"
    _write(log, "012 (1.000.000) 2026-06-01 08:00:00 Job was held.\n")
    records = scan_event_log(log)
    assert len(records) == 1
    assert records[0]["status"] == "held"


# --- scan_event_log: job arrays (many procs under one cluster) ---
#
# Regression coverage for a real production log (11 clusters, 275
# cluster.proc pairs via `queue N`) where every proc in a cluster was
# silently collapsed into a single record.


def test_scan_multiple_procs_in_one_cluster_are_separate_records(tmp_path):
    log = tmp_path / "batch.log"
    _write(
        log,
        "000 (7000.000.000) 2026-06-01 08:00:00 Job submitted from host: <1.2.3.4:9618>\n...\n"
        "000 (7000.001.000) 2026-06-01 08:00:01 Job submitted from host: <1.2.3.4:9618>\n...\n"
        "000 (7000.002.000) 2026-06-01 08:00:02 Job submitted from host: <1.2.3.4:9618>\n...\n"
        "001 (7000.000.000) 2026-06-01 08:05:00 Job executing on host: <10.0.0.1:1>\n"
        "\tSlotName: slot1_1@nodeA.example.edu\n...\n"
        "001 (7000.001.000) 2026-06-01 08:05:01 Job executing on host: <10.0.0.2:1>\n"
        "\tSlotName: slot1_1@nodeB.example.edu\n...\n"
        "005 (7000.000.000) 2026-06-01 09:00:00 Job terminated.\n"
        "\tPartitionable Resources :       Usage  Request Allocated\n"
        "\t   TimeExecute (s)      :    3300                       \n...\n"
        "012 (7000.002.000) 2026-06-01 08:10:00 Job was held.\n...\n",
    )

    records = scan_event_log(log)

    assert len(records) == 3
    by_proc = {r["proc_id"]: r for r in records}
    assert set(by_proc) == {0, 1, 2}
    assert all(r["cluster_id"] == 7000 for r in records)

    assert by_proc[0]["site"] == "nodeA.example.edu"
    assert by_proc[0]["wall_time_s"] == 3300.0
    assert by_proc[0]["status"] == "completed"

    assert by_proc[1]["site"] == "nodeB.example.edu"
    assert by_proc[1]["status"] == "executing"

    assert by_proc[2]["status"] == "held"
    assert "site" not in by_proc[2]


def test_scan_sorted_by_cluster_then_proc(tmp_path):
    log = tmp_path / "batch.log"
    _write(
        log,
        "012 (100.002.000) 2026-06-01 08:00:00 Job was held.\n...\n"
        "012 (100.000.000) 2026-06-01 08:00:00 Job was held.\n...\n"
        "012 (100.001.000) 2026-06-01 08:00:00 Job was held.\n...\n",
    )
    records = scan_event_log(log)
    assert [r["proc_id"] for r in records] == [0, 1, 2]
    assert all(r["cluster_id"] == 100 for r in records)


def test_scan_run_id_enrichment_shared_across_procs_but_job_id_stays_unique(tmp_path):
    """A cluster-level .run_id marker applies to every proc, but proc_id keeps rows distinct."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    (ad_dir / "8000.run_id").write_text("run-array")
    log = tmp_path / "batch.log"
    _write(
        log,
        "012 (8000.000.000) 2026-06-01 08:00:00 Job was held.\n...\n"
        "012 (8000.001.000) 2026-06-01 08:00:00 Job was held.\n...\n",
    )

    records = scan_event_log(log, log_dir=ad_dir)

    assert len(records) == 2
    assert all(r["run_id"] == "run-array" for r in records)
    assert {r["proc_id"] for r in records} == {0, 1}
