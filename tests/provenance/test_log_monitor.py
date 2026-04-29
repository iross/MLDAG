import json
from pathlib import Path

import pytest

from mldag.provenance.log_monitor import _parse_event_line, monitor_once


# --- helpers ---


def _write_log(path: Path, content: str) -> None:
    path.write_text(content)


def _write_ad(ad_dir: Path, cluster_id: int, run_id: str) -> None:
    ad_dir.mkdir(parents=True, exist_ok=True)
    (ad_dir / f"{cluster_id}.ad").write_text(
        f'Environment = "PROVENANCE_RUN_ID={run_id} OTHER=val"\n'
        f'GLIDEIN_ResourceName = "Expanse"\n'
        f'RemoteWallClockTime = 3600\n'
    )


def _read_events(log_dir: Path, run_id: str) -> list[dict]:
    path = log_dir / f"{run_id}.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- _parse_event_line ---


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


def test_parse_event_line_irrelevant_code_returns_none():
    line = "001 (12345.000.000) 2026-04-01 10:00:00 Job executing on host."
    assert _parse_event_line(line) is None


def test_parse_event_line_non_event_returns_none():
    assert _parse_event_line("    some indented log line") is None


# --- monitor_once ---


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
    first_line = "001 (12345.000.000) 2026-04-01 09:00:00 Job executing.\n"
    second_line = "004 (12345.000.000) 2026-04-01 10:00:00 Job was evicted.\n"
    log.write_text(first_line + second_line)

    # First call reads both lines but only first_line is in initial content
    offset1 = monitor_once(log, 0, log_dir=ad_dir, provenance_log_dir=prov_dir)
    # Second call from saved offset should see nothing new (file unchanged)
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
