import json
from pathlib import Path

from mldag.provenance.repair import (
    build_cluster_to_job,
    build_job_to_run_id,
    recover,
)


def _write_metl_log(path: Path, cluster_id: int, job_name: str) -> None:
    path.write_text(
        f"000 ({cluster_id}.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        f'    [ DAGNodeName = "{job_name}"; JobBatchName = "{job_name}" ]\n'
        "...\n"
    )


def _write_unknown(prov_dir: Path, cluster_id: int, event_type: str = "job.executing") -> Path:
    prov_dir.mkdir(parents=True, exist_ok=True)
    path = prov_dir / f"unknown:{cluster_id}.ndjson"
    event = {
        "schema_version": "1.0", "type": event_type, "run_id": f"unknown:{cluster_id}",
        "cluster_id": cluster_id, "source": "htcondor_event_log",
    }
    path.write_text(json.dumps(event) + "\n")
    return path


def _write_job_submitted(prov_dir: Path, run_id: str, job_name: str) -> None:
    prov_dir.mkdir(parents=True, exist_ok=True)
    event = {"schema_version": "1.0", "type": "job.submitted", "run_id": run_id, "job_name": job_name}
    (prov_dir / f"{run_id}.ndjson").write_text(json.dumps(event) + "\n")


# --- build_cluster_to_job ---


def test_build_cluster_to_job_parses_dagnode_blocks(tmp_path):
    log = tmp_path / "metl.log"
    _write_metl_log(log, 555, "run0-train_epoch0")
    assert build_cluster_to_job(log) == {555: "run0-train_epoch0"}


def test_build_cluster_to_job_ignores_non_000_events(tmp_path):
    log = tmp_path / "metl.log"
    log.write_text(
        "001 (555.000.000) 2026-04-29 10:00:00 Job executing.\n"
        '    [ DAGNodeName = "run0-train_epoch0" ]\n'
    )
    assert build_cluster_to_job(log) == {}


# --- build_job_to_run_id ---


def test_build_job_to_run_id_from_dag_vars(tmp_path):
    dag = tmp_path / "x.dag"
    dag.write_text('VARS run0-train_epoch0 epoch="1" run_uuid="run-abc"\n')
    assert build_job_to_run_id(dag, []) == {"run0-train_epoch0": "run-abc"}


def test_build_job_to_run_id_from_ndjson_job_submitted(tmp_path):
    prov_dir = tmp_path / "prov"
    _write_job_submitted(prov_dir, "run-xyz", "run0-train_epoch2")
    assert build_job_to_run_id(None, [prov_dir]) == {"run0-train_epoch2": "run-xyz"}


def test_build_job_to_run_id_from_job_completed_event(tmp_path):
    """post.py writes job.completed (not job.submitted) -- must count as a source too."""
    prov_dir = tmp_path / "prov"
    prov_dir.mkdir()
    (prov_dir / "run-abc.ndjson").write_text(
        json.dumps({
            "type": "job.completed", "run_id": "run-abc", "job_name": "run0-train_epoch12",
            "source": "dagman_post_script_classad",
        }) + "\n"
    )
    assert build_job_to_run_id(None, [prov_dir]) == {"run0-train_epoch12": "run-abc"}


def test_build_job_to_run_id_ignores_unknown_files(tmp_path):
    prov_dir = tmp_path / "prov"
    _write_unknown(prov_dir, 999)
    assert build_job_to_run_id(None, [prov_dir]) == {}


def test_build_job_to_run_id_dag_vars_take_precedence_over_ndjson(tmp_path):
    dag = tmp_path / "x.dag"
    dag.write_text('VARS run0-train_epoch0 epoch="1" run_uuid="run-from-dag"\n')
    prov_dir = tmp_path / "prov"
    _write_job_submitted(prov_dir, "run-from-ndjson", "run0-train_epoch0")
    assert build_job_to_run_id(dag, [prov_dir]) == {"run0-train_epoch0": "run-from-dag"}


# --- recover ---


def test_recover_resolves_and_merges_unknown_file(tmp_path):
    prov_dir = tmp_path / "prov"
    log = tmp_path / "metl.log"
    _write_metl_log(log, 555, "run0-train_epoch0")
    _write_job_submitted(prov_dir, "run-abc", "run0-train_epoch0")
    unknown_path = _write_unknown(prov_dir, 555)

    stats = recover([prov_dir], log, None)

    assert stats.merged == 1
    assert not unknown_path.exists()
    dest_lines = (prov_dir / "run-abc.ndjson").read_text().splitlines()
    events = [json.loads(line) for line in dest_lines]
    assert any(e["type"] == "job.executing" and e["run_id"] == "run-abc" for e in events)


def test_recover_dry_run_makes_no_changes(tmp_path):
    prov_dir = tmp_path / "prov"
    log = tmp_path / "metl.log"
    _write_metl_log(log, 555, "run0-train_epoch0")
    _write_job_submitted(prov_dir, "run-abc", "run0-train_epoch0")
    unknown_path = _write_unknown(prov_dir, 555)

    dest = prov_dir / "run-abc.ndjson"
    events_before = dest.read_text()

    stats = recover([prov_dir], log, None, dry_run=True)

    assert stats.merged == 1
    assert unknown_path.exists()
    assert dest.read_text() == events_before  # untouched: the job.submitted event only


def test_recover_reports_unresolved_when_dagnode_missing(tmp_path):
    """metl.log not covering the cluster's time window (task-22's actual current state)."""
    prov_dir = tmp_path / "prov"
    log = tmp_path / "metl.log"
    log.write_text("")  # stale/empty log -- no DAGNodeName data at all
    unknown_path = _write_unknown(prov_dir, 555)

    stats = recover([prov_dir], log, None)

    assert stats.merged == 0
    assert stats.skipped_no_dagnode == 1
    assert unknown_path.exists()


def test_recover_reports_unresolved_when_run_id_unknown(tmp_path):
    prov_dir = tmp_path / "prov"
    log = tmp_path / "metl.log"
    _write_metl_log(log, 555, "run0-train_epoch0")
    unknown_path = _write_unknown(prov_dir, 555)
    # No job.submitted record and no dag_file -- job_name resolves but run_id doesn't.

    stats = recover([prov_dir], log, None)

    assert stats.merged == 0
    assert stats.skipped_no_run_id == 1
    assert unknown_path.exists()


def test_recover_merges_appending_to_existing_run_file(tmp_path):
    prov_dir = tmp_path / "prov"
    log = tmp_path / "metl.log"
    _write_metl_log(log, 555, "run0-train_epoch0")
    _write_job_submitted(prov_dir, "run-abc", "run0-train_epoch0")
    (prov_dir / "run-abc.ndjson").write_text(
        json.dumps({"type": "job.submitted", "run_id": "run-abc", "job_name": "run0-train_epoch0"}) + "\n"
    )
    _write_unknown(prov_dir, 555)

    recover([prov_dir], log, None)

    events = [
        json.loads(line)
        for line in (prov_dir / "run-abc.ndjson").read_text().splitlines()
    ]
    assert len(events) == 2
    assert {e["type"] for e in events} == {"job.submitted", "job.executing"}


def test_recover_deduplicates_byte_identical_files_across_dirs(tmp_path):
    """Regression test for task-22's finding: `provenance/` was a byte-identical
    stale copy of `output/provenance/`. Identical unknown files across dirs must
    be merged once, not double-counted or double-appended."""
    dir_a = tmp_path / "output_provenance"
    dir_b = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    _write_metl_log(log, 555, "run0-train_epoch0")
    _write_job_submitted(dir_a, "run-abc", "run0-train_epoch0")

    path_a = _write_unknown(dir_a, 555)
    dir_b.mkdir(parents=True, exist_ok=True)
    path_b = dir_b / "unknown:555.ndjson"
    path_b.write_text(path_a.read_text())  # byte-identical copy

    stats = recover([dir_a, dir_b], log, None)

    assert stats.merged == 1
    assert stats.duplicates_removed == 1
    events = [
        json.loads(line)
        for line in (dir_a / "run-abc.ndjson").read_text().splitlines()
    ]
    # job.submitted (from setup) + exactly one job.executing (not two -- the
    # duplicate from dir_b must not also get merged in).
    assert len(events) == 2
    executing = [e for e in events if e["type"] == "job.executing"]
    assert len(executing) == 1


def test_recover_skips_unparseable_filename(tmp_path):
    prov_dir = tmp_path / "prov"
    prov_dir.mkdir()
    log = tmp_path / "metl.log"
    log.write_text("")
    (prov_dir / "unknown.ndjson").write_text(
        json.dumps({"type": "job.executing", "run_id": "unknown"}) + "\n"
    )

    stats = recover([prov_dir], log, None)

    assert stats.skipped_unparseable_name == 1
    assert stats.merged == 0
