import json
from pathlib import Path

import pytest

from mldag.provenance.post import (
    emit_post_event,
    parse_classad,
    resource_fields_from_classad,
    run_id_from_classad,
)


def _write_ad(tmp_path: Path, attrs: dict, filename: str = "12345.ad") -> Path:
    lines = []
    for k, v in attrs.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    p = tmp_path / filename
    p.write_text("\n".join(lines) + "\n")
    return p


SAMPLE_AD = {
    "RemoteWallClockTime": 3600,
    "CPUsUsage": 0.98,
    "MemoryUsage": 4096,
    "GPUsUsage": 0.87,
    "GLIDEIN_ResourceName": "CHTC-Spark-CE1",
    "Environment": "PROVENANCE_RUN_ID=run-abc123 OTHER=val",
}


# --- parse_classad ---


def test_parse_classad_integer_field(tmp_path):
    _write_ad(tmp_path, {"MemoryUsage": 4096})
    ad = parse_classad(tmp_path / "12345.ad")
    assert ad["MemoryUsage"] == 4096
    assert isinstance(ad["MemoryUsage"], int)


def test_parse_classad_float_field(tmp_path):
    _write_ad(tmp_path, {"CPUsUsage": 0.98})
    ad = parse_classad(tmp_path / "12345.ad")
    assert abs(ad["CPUsUsage"] - 0.98) < 1e-9


def test_parse_classad_quoted_string_field(tmp_path):
    _write_ad(tmp_path, {"Environment": "PROVENANCE_RUN_ID=abc FOO=bar"})
    ad = parse_classad(tmp_path / "12345.ad")
    assert ad["Environment"] == "PROVENANCE_RUN_ID=abc FOO=bar"


def test_parse_classad_missing_file_returns_empty(tmp_path):
    ad = parse_classad(tmp_path / "nonexistent.ad")
    assert ad == {}


def test_parse_classad_skips_blank_lines(tmp_path):
    (tmp_path / "12345.ad").write_text("\n\nMemoryUsage = 512\n\n")
    ad = parse_classad(tmp_path / "12345.ad")
    assert ad["MemoryUsage"] == 512


# --- run_id_from_classad ---


def test_run_id_from_classad(tmp_path):
    _write_ad(tmp_path, SAMPLE_AD)
    ad = parse_classad(tmp_path / "12345.ad")
    assert run_id_from_classad(ad) == "run-abc123"


def test_run_id_from_classad_missing_returns_unknown():
    assert run_id_from_classad({}) == "unknown"


def test_run_id_from_classad_no_env_returns_unknown():
    assert run_id_from_classad({"Environment": "UNRELATED=val"}) == "unknown"


# --- resource_fields_from_classad ---


def test_resource_fields_all_present(tmp_path):
    _write_ad(tmp_path, SAMPLE_AD)
    ad = parse_classad(tmp_path / "12345.ad")
    fields = resource_fields_from_classad(ad)
    assert fields["wall_time_s"] == 3600
    assert abs(fields["cpu_usage"] - 0.98) < 1e-9
    assert fields["peak_memory_mb"] == 4096
    assert abs(fields["gpu_usage"] - 0.87) < 1e-9
    assert fields["resource_name"] == "CHTC-Spark-CE1"


def test_resource_fields_no_gpu(tmp_path):
    ad = {k: v for k, v in SAMPLE_AD.items() if k != "GPUsUsage"}
    _write_ad(tmp_path, ad)
    fields = resource_fields_from_classad(parse_classad(tmp_path / "12345.ad"))
    assert "gpu_usage" not in fields
    assert "wall_time_s" in fields


def test_resource_fields_no_glidein(tmp_path):
    ad = {k: v for k, v in SAMPLE_AD.items() if k != "GLIDEIN_ResourceName"}
    _write_ad(tmp_path, ad)
    fields = resource_fields_from_classad(parse_classad(tmp_path / "12345.ad"))
    assert "resource_name" not in fields


def test_resource_fields_empty_ad():
    assert resource_fields_from_classad({}) == {}


# --- emit_post_event ---


def _read_events(log_dir: Path, run_id: str) -> list[dict]:
    path = log_dir / f"{run_id}.ndjson"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_emit_post_event_completed(tmp_path):
    _write_ad(tmp_path, SAMPLE_AD)
    emit_post_event("run0-train_epoch0", 0, "12345", log_dir=tmp_path)
    events = _read_events(tmp_path, "run-abc123")
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "job.completed"
    assert e["run_id"] == "run-abc123"
    assert e["job_name"] == "run0-train_epoch0"
    assert e["wall_time_s"] == 3600
    assert e["peak_memory_mb"] == 4096
    assert "exit_code" not in e


def test_emit_post_event_failed(tmp_path):
    _write_ad(tmp_path, SAMPLE_AD)
    emit_post_event("run0-train_epoch0", 1, "12345", log_dir=tmp_path)
    events = _read_events(tmp_path, "run-abc123")
    e = events[0]
    assert e["type"] == "job.failed"
    assert e["exit_code"] == 1
    assert e["wall_time_s"] == 3600


def test_emit_post_event_failed_with_hold_reason(tmp_path):
    ad = {**SAMPLE_AD, "HoldReason": "Disk quota exceeded"}
    _write_ad(tmp_path, ad)
    emit_post_event("run0-train_epoch0", 2, "12345", log_dir=tmp_path)
    e = _read_events(tmp_path, "run-abc123")[0]
    assert e["type"] == "job.failed"
    assert e["hold_reason"] == "Disk quota exceeded"


def test_emit_post_event_missing_ad_file(tmp_path):
    # No ad file written — run_id is unknown, no resource fields
    emit_post_event("run0-train_epoch0", 0, "99999", log_dir=tmp_path)
    events = _read_events(tmp_path, "unknown")
    e = events[0]
    assert e["type"] == "job.completed"
    assert e["run_id"] == "unknown"
    assert "wall_time_s" not in e


def test_emit_post_event_completed_no_hold_reason_field(tmp_path):
    _write_ad(tmp_path, SAMPLE_AD)
    emit_post_event("run0-train_epoch0", 0, "12345", log_dir=tmp_path)
    e = _read_events(tmp_path, "run-abc123")[0]
    assert "hold_reason" not in e
