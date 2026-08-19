import json
from pathlib import Path

from mldag.provenance.log_monitor import (
    _accumulate_usage_field,
    _load_cache,
    _load_job_index,
    _load_offset,
    _load_pending,
    _parse_event_line,
    _refresh_job_submitted_index,
    _resolve_run_id,
    _save_cache,
    _save_job_index,
    _save_offset,
    _save_pending,
    monitor_once,
)

# --- helpers ---


def _write_log(path: Path, content: str) -> None:
    path.write_text(content)


def _write_ad(ad_dir: Path, cluster_id: int, run_id: str) -> None:
    ad_dir.mkdir(parents=True, exist_ok=True)
    (ad_dir / f"{cluster_id}.ad").write_text(
        f'Environment = "PROVENANCE_RUN_ID={run_id} OTHER=val"\n'
        f'GLIDEIN_ResourceName = "Expanse"\n'
        f"RemoteWallClockTime = 3600\n"
    )


def _write_ndjson(prov_dir: Path, run_id: str, job_name: str) -> None:
    prov_dir.mkdir(parents=True, exist_ok=True)
    event = {"type": "job.submitted", "run_id": run_id, "job_name": job_name}
    (prov_dir / f"{run_id}.ndjson").write_text(json.dumps(event) + "\n")


def _read_events(log_dir: Path, run_id: str) -> list[dict]:
    path = log_dir / f"{run_id}.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- _parse_event_line ---


def test_parse_event_line_executing_new_ts():
    line = "001 (12345.000.000) 2026-04-01 10:00:00 Job executing on host."
    result = _parse_event_line(line)
    assert result is not None
    assert result[0] == "001"
    assert result[1] == 12345


def test_parse_event_line_eviction_new_ts():
    line = "004 (12345.000.000) 2026-04-01 10:00:00 Job was evicted."
    result = _parse_event_line(line)
    assert result is not None
    code, cluster_id, ts = result
    assert code == "004"
    assert cluster_id == 12345
    assert ts.year == 2026


def test_parse_event_line_hold_new_ts():
    line = "012 (99.000.000) 2026-04-01 11:00:00 Job was held."
    result = _parse_event_line(line)
    assert result is not None
    assert result[0] == "012"
    assert result[1] == 99


def test_parse_event_line_release_new_ts():
    line = "013 (99.000.000) 2026-04-01 11:05:00 Job was released."
    result = _parse_event_line(line)
    assert result is not None
    assert result[0] == "013"


def test_parse_event_line_hold_legacy_ts():
    line = "012 (42.000.000) 04/01 11:00:00 Job was held."
    result = _parse_event_line(line)
    assert result is not None
    assert result[0] == "012"
    assert result[1] == 42


def test_parse_event_line_aborted_new_ts():
    line = "009 (12345.000.000) 2026-04-01 10:00:00 Job was aborted."
    result = _parse_event_line(line)
    assert result is not None
    assert result[0] == "009"


def test_parse_event_line_reconnected_new_ts():
    line = "023 (12345.000.000) 2026-04-01 10:00:00 Job reconnected to slot1_1@node."
    result = _parse_event_line(line)
    assert result is not None
    assert result[0] == "023"


def test_parse_event_line_transfer_returns_040():
    for desc in [
        "Started transferring input files",
        "Finished transferring input files",
        "Started transferring output files",
        "Finished transferring output files",
    ]:
        line = f"040 (12345.000.000) 2026-04-01 10:00:00 {desc}"
        result = _parse_event_line(line)
        assert result is not None, f"expected match for: {desc}"
        assert result[0] == "040"


def test_parse_event_line_submit_returns_none():
    line = "000 (12345.000.000) 2026-04-01 09:00:00 Job submitted from host."
    assert _parse_event_line(line) is None


def test_parse_event_line_irrelevant_code_returns_none():
    line = "006 (12345.000.000) 2026-04-01 10:00:00 Image size updated."
    assert _parse_event_line(line) is None


def test_parse_event_line_non_event_returns_none():
    assert _parse_event_line("    some indented log line") is None


# --- _refresh_job_submitted_index ---


def test_refresh_job_submitted_index_finds_match(tmp_path):
    _write_ndjson(tmp_path, "run-abc", "run0-train_epoch0")
    index: dict = {}
    offsets: dict = {}
    _refresh_job_submitted_index(tmp_path, index, offsets)
    assert index.get("run0-train_epoch0") == "run-abc"


def test_refresh_job_submitted_index_no_match_for_unknown_job(tmp_path):
    _write_ndjson(tmp_path, "run-abc", "run0-train_epoch0")
    index: dict = {}
    offsets: dict = {}
    _refresh_job_submitted_index(tmp_path, index, offsets)
    assert "run0-train_epoch99" not in index


def test_refresh_job_submitted_index_empty_dir(tmp_path):
    index: dict = {}
    offsets: dict = {}
    _refresh_job_submitted_index(tmp_path, index, offsets)
    assert index == {}


def test_refresh_job_submitted_index_does_not_reread_unchanged_files(tmp_path):
    """Repeated calls must not re-parse bytes already scanned (the O(n) rescan this replaces)."""
    _write_ndjson(tmp_path, "run-abc", "run0-train_epoch0")
    index: dict = {}
    offsets: dict = {}
    _refresh_job_submitted_index(tmp_path, index, offsets)
    ndjson_path = tmp_path / "run-abc.ndjson"
    recorded_offset = offsets[str(ndjson_path)]
    assert recorded_offset == ndjson_path.stat().st_size

    read_calls = []
    real_open = open

    def _tracking_open(path, *args, **kwargs):
        read_calls.append(str(path))
        return real_open(path, *args, **kwargs)

    import builtins
    from unittest.mock import patch

    with patch.object(builtins, "open", _tracking_open):
        _refresh_job_submitted_index(tmp_path, index, offsets)
    assert str(ndjson_path) not in read_calls


def test_refresh_job_submitted_index_picks_up_appended_events(tmp_path):
    """A second job.submitted appended to the same file is indexed without rereading old bytes."""
    prov_dir = tmp_path
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    index: dict = {}
    offsets: dict = {}
    _refresh_job_submitted_index(prov_dir, index, offsets)

    with open(prov_dir / "run-abc.ndjson", "a") as f:
        f.write(json.dumps({
            "type": "job.submitted", "run_id": "run-abc", "job_name": "run0-train_epoch1",
        }) + "\n")

    _refresh_job_submitted_index(prov_dir, index, offsets)
    assert index.get("run0-train_epoch0") == "run-abc"
    assert index.get("run0-train_epoch1") == "run-abc"


def test_save_and_load_job_index_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    _save_job_index(path, {"job-a": "run-1"}, {"/tmp/a.ndjson": 42})
    index, offsets = _load_job_index(path)
    assert index == {"job-a": "run-1"}
    assert offsets == {"/tmp/a.ndjson": 42}


def test_load_job_index_returns_empty_when_missing(tmp_path):
    index, offsets = _load_job_index(tmp_path / "missing.json")
    assert index == {}
    assert offsets == {}


def test_load_job_index_returns_empty_on_corrupt(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("not json")
    index, offsets = _load_job_index(path)
    assert index == {}
    assert offsets == {}


def test_monitor_once_job_submitted_in_wrong_dir_stays_pending_not_unknown(tmp_path):
    """Regression test for task-22: if job.submitted lands in a directory log_monitor
    isn't configured to search (e.g. pre.py and log_monitor disagreeing on
    PROVENANCE_LOG_DIR), the cluster_id must stay in pending_lookups -- retryable
    once pointed at the right directory -- rather than resolving from a place it
    was never told to look.
    """
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    wrong_prov_dir = tmp_path / "wrong_provenance"
    right_prov_dir = tmp_path / "right_provenance"
    _write_ndjson(right_prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    pending: dict = {}
    monitor_once(
        log, 0, log_dir=ad_dir, provenance_log_dir=wrong_prov_dir, pending_lookups=pending
    )
    assert pending.get(5055662) == "run0-train_epoch0"

    # Repointed at the directory job.submitted actually lives in: resolves on retry.
    pending_after_fix: dict = dict(pending)
    monitor_once(
        log, 0, log_dir=ad_dir, provenance_log_dir=right_prov_dir,
        pending_lookups=pending_after_fix,
    )
    assert 5055662 not in pending_after_fix


# --- monitor_once: cache population from event 000 + DAG Node ---


def test_monitor_once_event_000_dag_node_populates_cache(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    cache: dict = {}
    monitor_once(
        log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir, run_id_cache=cache
    )
    assert cache.get(5055662) == "run-abc"
    events = _read_events(prov_dir, "run-abc")
    queued = [e for e in events if e["type"] == "job.queued"]
    assert len(queued) == 1
    assert queued[0]["cluster_id"] == 5055662
    assert queued[0]["job_name"] == "run0-train_epoch0"


def test_monitor_once_event_000_then_executing_resolves_run_id(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n"
        "001 (5055662.000.000) 2026-04-29 10:05:00 Job executing on host: <10.0.0.1:1234>\n"
        "...\n",
    )
    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)
    events = _read_events(prov_dir, "run-abc")
    assert (
        len(events) == 3
    )  # job.submitted (written earlier) + job.queued + job.executing
    executing = next(e for e in events if e["type"] == "job.executing")
    assert executing["run_id"] == "run-abc"
    assert executing["cluster_id"] == 5055662


def test_monitor_once_pending_lookup_resolved_on_next_poll(tmp_path):
    """DAG Node seen but NDJSON not yet written → stored, resolved on next poll."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"

    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    cache: dict = {}
    state: dict = {"cluster_id": None}
    pending: dict = {}
    offset = monitor_once(
        log,
        0,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        multiline_state=state,
        pending_lookups=pending,
    )

    assert cache.get(5055662) is None
    assert pending.get(5055662) == "run0-train_epoch0"

    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")

    monitor_once(
        log,
        offset,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        multiline_state=state,
        pending_lookups=pending,
    )

    assert cache.get(5055662) == "run-abc"
    assert 5055662 not in pending
    events = _read_events(prov_dir, "run-abc")
    queued = [e for e in events if e["type"] == "job.queued"]
    assert len(queued) == 1
    assert queued[0]["cluster_id"] == 5055662
    assert queued[0]["job_name"] == "run0-train_epoch0"


def test_monitor_once_pending_resolved_before_executing_event(tmp_path):
    """Executing event in second poll uses run_id resolved from pending_lookups."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"

    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    cache: dict = {}
    state: dict = {"cluster_id": None}
    pending: dict = {}
    offset = monitor_once(
        log,
        0,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        multiline_state=state,
        pending_lookups=pending,
    )

    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    with open(log, "a") as f:
        f.write(
            "001 (5055662.000.000) 2026-04-29 10:05:00 Job executing on host: <10.0.0.1:1234>\n...\n"
        )

    monitor_once(
        log,
        offset,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        multiline_state=state,
        pending_lookups=pending,
    )

    events = _read_events(prov_dir, "run-abc")
    queued = [e for e in events if e["type"] == "job.queued"]
    assert len(queued) == 1
    assert queued[0]["cluster_id"] == 5055662
    executing = [e for e in events if e["type"] == "job.executing"]
    assert len(executing) == 1
    assert executing[0]["run_id"] == "run-abc"
    assert executing[0]["cluster_id"] == 5055662


def test_monitor_once_dag_node_across_poll_boundary(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"

    first = "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
    log.write_text(first)
    cache: dict = {}
    state: dict = {"cluster_id": None}
    offset = monitor_once(
        log,
        0,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        multiline_state=state,
    )
    assert cache.get(5055662) is None  # DAG Node line not seen yet
    assert state["cluster_id"] == 5055662  # pending

    with open(log, "a") as f:
        f.write(
            '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n...\n'
        )
    monitor_once(
        log,
        offset,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        multiline_state=state,
    )
    assert cache.get(5055662) == "run-abc"
    events = _read_events(prov_dir, "run-abc")
    queued = [e for e in events if e["type"] == "job.queued"]
    assert len(queued) == 1
    assert queued[0]["cluster_id"] == 5055662


# --- monitor_once: event emission ---


def test_monitor_once_transfer_events_emitted(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 12345, "run-abc")
    log = tmp_path / "metl.log"
    # HTCondor uses code 040 for all file transfer events; direction is in the description
    _write_log(
        log,
        "040 (12345.000.000) 2026-04-29 10:00:00 Started transferring input files\n"
        "040 (12345.000.000) 2026-04-29 10:01:00 Finished transferring input files\n"
        "040 (12345.000.000) 2026-04-29 11:00:00 Started transferring output files\n"
        "040 (12345.000.000) 2026-04-29 11:05:00 Finished transferring output files\n",
    )

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-abc")
    types = [e["type"] for e in events]
    assert types == [
        "transfer.input.started",
        "transfer.input.completed",
        "transfer.output.started",
        "transfer.output.completed",
    ]
    assert all(e["run_id"] == "run-abc" for e in events)
    assert all(e["source"] == "htcondor_event_log" for e in events)


def test_monitor_once_executing_emits_job_executing(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 12345, "run-abc")
    log = tmp_path / "metl.log"
    _write_log(log, "001 (12345.000.000) 2026-04-01 10:00:00 Job executing on host.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-abc")
    assert len(events) == 1
    assert events[0]["type"] == "job.executing"
    assert events[0]["run_id"] == "run-abc"
    assert events[0]["source"] == "htcondor_event_log"


def test_monitor_once_eviction_emits_job_migrated(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 12345, "run-abc")
    log = tmp_path / "metl.log"
    _write_log(log, "004 (12345.000.000) 2026-04-01 10:00:00 Job was evicted.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-abc")
    assert len(events) == 1
    assert events[0]["type"] == "job.migrated"
    assert events[0]["run_id"] == "run-abc"
    assert events[0]["cluster_id"] == 12345
    assert events[0]["condor_event_ts"].startswith("2026")
    assert events[0]["source"] == "htcondor_event_log"


def test_monitor_once_hold_emits_job_held(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 99, "run-xyz")
    log = tmp_path / "metl.log"
    _write_log(log, "012 (99.000.000) 2026-04-01 11:00:00 Job was held.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-xyz")
    assert events[0]["type"] == "job.held"


def test_monitor_once_release_emits_job_released(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 99, "run-xyz")
    log = tmp_path / "metl.log"
    _write_log(log, "013 (99.000.000) 2026-04-01 11:05:00 Job was released.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-xyz")
    assert events[0]["type"] == "job.released"


def test_monitor_once_no_classad_uses_unknown_run_id(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    _write_log(log, "012 (55555.000.000) 2026-04-01 11:00:00 Job was held.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "unknown:55555")
    assert events[0]["type"] == "job.held"


def test_monitor_once_cache_resolves_hold_without_classad(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    _write_log(log, "012 (88.000.000) 2026-04-01 11:00:00 Job was held.\n")

    cache = {88: "run-cached"}
    monitor_once(
        log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir, run_id_cache=cache
    )

    events = _read_events(prov_dir, "run-cached")
    assert len(events) == 1
    assert events[0]["type"] == "job.held"


def test_monitor_once_run_id_marker_resolves_hold_without_classad(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    (ad_dir / "77.run_id").write_text("run-held")
    log = tmp_path / "metl.log"
    _write_log(log, "012 (77.000.000) 2026-04-01 11:00:00 Job was held.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-held")
    assert len(events) == 1
    assert events[0]["type"] == "job.held"
    assert events[0]["run_id"] == "run-held"


def test_monitor_once_run_id_marker_resolves_release_without_classad(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    (ad_dir / "77.run_id").write_text("run-held")
    log = tmp_path / "metl.log"
    _write_log(log, "013 (77.000.000) 2026-04-01 11:05:00 Job was released.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-held")
    assert len(events) == 1
    assert events[0]["type"] == "job.released"


def test_monitor_once_returns_new_byte_offset(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    content = "004 (12345.000.000) 2026-04-01 10:00:00 Job was evicted.\n"
    _write_log(log, content)

    new_offset = monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)
    assert new_offset == len(content.encode())


def test_monitor_once_skips_already_read_bytes(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 12345, "run-abc")
    log = tmp_path / "metl.log"
    first_line = "006 (12345.000.000) 2026-04-01 09:00:00 Image size updated.\n"
    second_line = "004 (12345.000.000) 2026-04-01 10:00:00 Job was evicted.\n"
    log.write_text(first_line + second_line)

    offset1 = monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)
    offset2 = monitor_once(log, offset1, log_dir=ad_dir, provenance_log_dir=prov_dir)
    assert offset2 == offset1

    events = _read_events(prov_dir, "run-abc")
    assert len(events) == 1
    assert events[0]["type"] == "job.migrated"


def test_monitor_once_missing_log_file_returns_zero(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    new_offset = monitor_once(
        tmp_path / "nonexistent.log", 0, log_dir=ad_dir, provenance_log_dir=prov_dir
    )
    assert new_offset == 0


def test_monitor_once_includes_resource_name_from_classad(tmp_path):
    ad_dir = tmp_path / "ads"
    prov_dir = tmp_path / "provenance"
    _write_ad(ad_dir, 12345, "run-abc")
    log = tmp_path / "metl.log"
    _write_log(log, "004 (12345.000.000) 2026-04-01 10:00:00 Job was evicted.\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-abc")
    assert events[0].get("resource_name") == "Expanse"


# --- _load_offset / _save_offset ---


def test_save_and_load_offset_roundtrip(tmp_path):
    log = tmp_path / "metl.log"
    log.write_text("x" * 1000)
    offset_path = tmp_path / ".log_monitor.offset"
    _save_offset(offset_path, 1000)
    assert _load_offset(offset_path, log) == 1000


def test_load_offset_returns_zero_when_file_missing(tmp_path):
    log = tmp_path / "metl.log"
    log.write_text("x" * 100)
    assert _load_offset(tmp_path / ".log_monitor.offset", log) == 0


def test_load_offset_resets_when_log_recreated(tmp_path):
    log = tmp_path / "metl.log"
    log.write_text("x" * 100)
    offset_path = tmp_path / ".log_monitor.offset"
    # Save an offset larger than the current log (simulates log recreation)
    _save_offset(offset_path, 50_000)
    assert _load_offset(offset_path, log) == 0


def test_load_offset_returns_zero_on_corrupt_file(tmp_path):
    log = tmp_path / "metl.log"
    log.write_text("x" * 100)
    offset_path = tmp_path / ".log_monitor.offset"
    offset_path.write_text("not-a-number")
    assert _load_offset(offset_path, log) == 0


# --- _save_cache / _load_cache (Fix 1: cache persistence across SERVICE restarts) ---


def test_save_and_load_cache_roundtrip(tmp_path):
    cache = {12345: "run-abc", 99: "run-xyz"}
    cache_path = tmp_path / ".log_monitor.cache.json"
    _save_cache(cache_path, cache)
    assert _load_cache(cache_path) == cache


def test_load_cache_returns_empty_dict_when_missing(tmp_path):
    assert _load_cache(tmp_path / ".log_monitor.cache.json") == {}


def test_load_cache_returns_empty_dict_on_corrupt(tmp_path):
    cache_path = tmp_path / ".log_monitor.cache.json"
    cache_path.write_text("not-json")
    assert _load_cache(cache_path) == {}


def test_monitor_once_prepopulated_cache_resolves_transfer_event(tmp_path):
    """Cache loaded from disk (e.g. after SERVICE restart) resolves a 040 event."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "040 (5055662.000.000) 2026-04-29 10:00:00 Started transferring input files\n",
    )

    cache = {5055662: "run-abc"}
    monitor_once(
        log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir, run_id_cache=cache
    )

    events = _read_events(prov_dir, "run-abc")
    assert len(events) == 1
    assert events[0]["type"] == "transfer.input.started"
    assert events[0]["run_id"] == "run-abc"


# --- Fix 2: inline pending_lookups retry when _resolve_run_id returns unknown ---


def test_monitor_once_same_poll_000_and_transfer_with_ndjson_available(tmp_path):
    """000 and 040 in same poll with NDJSON present: transfer resolves via cache set by 000."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n"
        "040 (5055662.000.000) 2026-04-29 10:00:01 Started transferring input files\n",
    )
    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-abc")
    types = [e["type"] for e in events]
    assert "job.queued" in types
    assert "transfer.input.started" in types
    transfer = next(e for e in events if e["type"] == "transfer.input.started")
    assert transfer["run_id"] == "run-abc"


# --- _save_pending / _load_pending ---


def test_save_and_load_pending_roundtrip(tmp_path):
    pending = {12345: "run0-train_epoch0", 99: "run1-train_epoch3"}
    pending_path = tmp_path / ".log_monitor.pending.json"
    _save_pending(pending_path, pending)
    assert _load_pending(pending_path) == pending


def test_load_pending_returns_empty_dict_when_missing(tmp_path):
    assert _load_pending(tmp_path / ".log_monitor.pending.json") == {}


def test_load_pending_returns_empty_dict_on_corrupt(tmp_path):
    pending_path = tmp_path / ".log_monitor.pending.json"
    pending_path.write_text("not-json")
    assert _load_pending(pending_path) == {}


# --- _resolve_run_id ---


def test_resolve_run_id_from_cache(tmp_path):
    cache = {12345: "run-cached"}
    run_id, resource = _resolve_run_id(12345, tmp_path, cache)
    assert run_id == "run-cached"
    assert resource == {}


def test_resolve_run_id_from_run_id_marker(tmp_path):
    (tmp_path / "99.run_id").write_text("run-from-marker")
    cache: dict = {}
    run_id, resource = _resolve_run_id(99, tmp_path, cache)
    assert run_id == "run-from-marker"
    assert cache[99] == "run-from-marker"


def test_resolve_run_id_from_classad(tmp_path):
    _write_ad(tmp_path, 77, "run-from-ad")
    cache: dict = {}
    run_id, resource = _resolve_run_id(77, tmp_path, cache)
    assert run_id == "run-from-ad"
    assert cache[77] == "run-from-ad"
    assert resource.get("resource_name") == "Expanse"


def test_resolve_run_id_unknown_fallback(tmp_path):
    cache: dict = {}
    run_id, resource = _resolve_run_id(55555, tmp_path, cache)
    assert run_id == "unknown:55555"
    assert 55555 not in cache


# --- .run_id file written on resolution ---


def test_monitor_once_run_id_file_written_on_000_resolution(tmp_path):
    """When run_id resolved from DAGNodeName in 000 block, .run_id file is written."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    run_id_file = ad_dir / "5055662.run_id"
    assert run_id_file.exists(), ".run_id file should be written after 000 resolution"
    assert run_id_file.read_text().strip() == "run-abc"


def test_monitor_once_run_id_file_written_on_pending_resolution(tmp_path):
    """When pending lookup resolves on next poll, .run_id file is written."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    cache: dict = {}
    pending: dict = {}
    offset = monitor_once(
        log,
        0,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        pending_lookups=pending,
    )
    assert pending.get(5055662) == "run0-train_epoch0"
    assert not (ad_dir / "5055662.run_id").exists()

    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    monitor_once(
        log,
        offset,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        pending_lookups=pending,
    )

    run_id_file = ad_dir / "5055662.run_id"
    assert run_id_file.exists(), ".run_id file should be written when pending resolves"
    assert run_id_file.read_text().strip() == "run-abc"


def test_monitor_once_run_id_file_written_on_inline_pending_resolution(tmp_path):
    """When event triggers inline pending resolution, .run_id file is written."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "000 (5055662.000.000) 2026-04-29 10:00:00 Job submitted from host: <1.2.3.4:9618>\n"
        '    [ DAGNodeName = "run0-train_epoch0"; JobBatchName = "run0-train_epoch0" ]\n'
        "...\n",
    )
    cache: dict = {}
    pending: dict = {}
    offset = monitor_once(
        log,
        0,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        pending_lookups=pending,
    )

    _write_ndjson(prov_dir, "run-abc", "run0-train_epoch0")
    with open(log, "a") as f:
        f.write(
            "001 (5055662.000.000) 2026-04-29 10:05:00 Job executing on host: <10.0.0.1:1234>\n...\n"
        )

    monitor_once(
        log,
        offset,
        log_dir=ad_dir,
        provenance_log_dir=prov_dir,
        run_id_cache=cache,
        pending_lookups=pending,
    )

    run_id_file = ad_dir / "5055662.run_id"
    assert (
        run_id_file.exists()
    ), ".run_id file should be written on inline pending resolution"
    assert run_id_file.read_text().strip() == "run-abc"


# --- 005 (Job terminated) resource-usage banner parsing ---
#
# job_ad_file is not a real HTCondor submit command -- condor_submit silently
# ignores it, confirmed against a live pool. Resource usage instead comes from
# parsing the 005 event's Partitionable Resources banner, verified against real
# entries pulled from a live metl.log.


def test_accumulate_usage_field_parses_full_banner():
    fields: dict = {}
    for line in [
        "   Cpus                 :        4.05        4         4 ",
        '   GPUs                 :        0.95        1         1 "GPU-e3b28da1"',
        "   Memory (MB)          :    47022       65536     65536 ",
        "   TimeExecute (s)      :    39645                       ",
    ]:
        _accumulate_usage_field(line.strip(), fields)

    assert fields == {
        "cpu_usage": 4.05,
        "gpu_usage": 0.95,
        "gpu_ids": "GPU-e3b28da1",
        "peak_memory_mb": 47022.0,
        "wall_time_s": 39645.0,
    }


def test_accumulate_usage_field_blank_gpu_usage_still_captures_ids():
    """Observed on a real 4-GPU job: the Usage column can be blank while
    Request/Allocated are still populated."""
    fields: dict = {}
    _accumulate_usage_field(
        'GPUs                 :                     4         4 "GPU-a,GPU-b"', fields
    )
    assert "gpu_usage" not in fields
    assert fields["gpu_ids"] == "GPU-a,GPU-b"


def test_accumulate_usage_field_ignores_unrelated_lines():
    fields: dict = {}
    _accumulate_usage_field("Disk (KB)            : 56826224    80485760  80489102", fields)
    _accumulate_usage_field("Partitionable Resources :       Usage  Request Allocated", fields)
    assert fields == {}


def test_monitor_once_005_emits_job_resource_usage(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    (ad_dir / "12651505.run_id").write_text("run-usage")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "005 (12651505.000.000) 2025-08-21 13:57:06 Job terminated.\n"
        "\t(1) Normal termination (return value 0)\n"
        "\t\tUsr 0 23:23:00, Sys 0 00:58:48  -  Run Remote Usage\n"
        "\tPartitionable Resources :       Usage  Request Allocated Assigned\n"
        "\t   Cpus                 :        4.05        4         4 \n"
        '\t   GPUs                 :        0.95        1         1 "GPU-e3b28da1"\n'
        "\t   Memory (MB)          :    47022       65536     65536 \n"
        "\t   TimeExecute (s)      :    39645                       \n"
        "...\n",
    )

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-usage")
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "job.resource_usage"
    assert e["wall_time_s"] == 39645.0
    assert e["cpu_usage"] == 4.05
    assert e["peak_memory_mb"] == 47022.0
    assert e["gpu_usage"] == 0.95
    assert e["gpu_ids"] == "GPU-e3b28da1"
    assert e["source"] == "htcondor_event_log"


def test_monitor_once_005_blank_gpu_usage_omits_field(tmp_path):
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    (ad_dir / "1.run_id").write_text("run-multi-gpu")
    log = tmp_path / "metl.log"
    _write_log(
        log,
        "005 (1.000.000) 2025-08-26 16:21:25 Job terminated.\n"
        "\tPartitionable Resources :       Usage   Request Allocated Assigned\n"
        '\t   GPUs                 :                     4         4 "GPU-a,GPU-b,GPU-c,GPU-d"\n'
        "\t   TimeExecute (s)      :    28437                        \n"
        "...\n",
    )

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    events = _read_events(prov_dir, "run-multi-gpu")
    assert len(events) == 1
    assert "gpu_usage" not in events[0]
    assert events[0]["gpu_ids"] == "GPU-a,GPU-b,GPU-c,GPU-d"
    assert events[0]["wall_time_s"] == 28437.0


def test_monitor_once_005_no_matching_rows_emits_nothing(tmp_path):
    """A banner that never matches any known row (malformed/unexpected format)
    must not produce an empty job.resource_usage event."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    (ad_dir / "99.run_id").write_text("run-empty")
    log = tmp_path / "metl.log"
    _write_log(log, "005 (99.000.000) 2026-04-01 10:00:00 Job terminated.\n...\n")

    monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)

    assert _read_events(prov_dir, "run-empty") == []


def test_monitor_once_005_banner_split_across_polls(tmp_path):
    """HTCondor writes the whole banner atomically in practice, but state must
    still persist correctly if a poll boundary lands in the middle of it."""
    ad_dir = tmp_path / "ads"
    ad_dir.mkdir()
    prov_dir = tmp_path / "provenance"
    (ad_dir / "55.run_id").write_text("run-split")
    log = tmp_path / "metl.log"
    state: dict = {"cluster_id": None}

    first_half = (
        "005 (55.000.000) 2026-04-01 10:00:00 Job terminated.\n"
        "\tPartitionable Resources :       Usage  Request Allocated Assigned\n"
        "\t   Cpus                 :        2.00        4         4 \n"
    )
    _write_log(log, first_half)
    offset = monitor_once(
        log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir, multiline_state=state
    )
    assert _read_events(prov_dir, "run-split") == []  # banner not terminated yet
    assert state["usage_cluster_id"] == 55

    with open(log, "a") as f:
        f.write("\t   TimeExecute (s)      :    100                       \n...\n")
    monitor_once(
        log, offset, log_dir=ad_dir, provenance_log_dir=prov_dir, multiline_state=state
    )

    events = _read_events(prov_dir, "run-split")
    assert len(events) == 1
    assert events[0]["cpu_usage"] == 2.0
    assert events[0]["wall_time_s"] == 100.0
    assert state["usage_cluster_id"] is None
