import json
import time
from pathlib import Path

import pytest

from mldag.provenance.watcher import (
    _load_site_info,
    _parse_epoch,
    _parse_val_loss,
    _read_metrics_csv,
    _sorted_by_mtime,
    scan_once,
    watch_and_emit,
)


def _make_ckpt(path: Path, content: bytes = b"weights") -> Path:
    path.write_bytes(content)
    return path


def _read_events(log_dir: Path, run_id: str) -> list[dict]:
    p = log_dir / f"{run_id}.ndjson"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --- _load_site_info ---


def test_load_site_info_reads_file(tmp_path):
    (tmp_path / "site_info.json").write_text(
        json.dumps({
            "hostname": "gpu04.chtc.wisc.edu",
            "slot": "slot1_1",
            "gpu_model": "A100",
            "gpu_count": 4,
            "gpu_id": "GPU-abc123",
            "python": "3.11.4",
            "cuda": "12.2",
            "code_commit": "abc123",
        })
    )
    site, env = _load_site_info(tmp_path)
    assert site["hostname"] == "gpu04.chtc.wisc.edu"
    assert site["gpu_model"] == "A100"
    assert site["gpu_id"] == "GPU-abc123"
    assert env["python"] == "3.11.4"
    assert env["cuda"] == "12.2"
    assert "gpu_id" not in env


def test_load_site_info_missing_file(tmp_path):
    site, env = _load_site_info(tmp_path)
    assert site == {}
    assert env == {}


def test_load_site_info_malformed_json(tmp_path):
    (tmp_path / "site_info.json").write_text("not json")
    site, env = _load_site_info(tmp_path)
    assert site == {}
    assert env == {}


# --- _sorted_by_mtime ---


def test_sorted_by_mtime_ordering(tmp_path):
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    c = tmp_path / "c.ckpt"
    a.write_bytes(b"a")
    time.sleep(0.01)
    b.write_bytes(b"b")
    time.sleep(0.01)
    c.write_bytes(b"c")
    result = _sorted_by_mtime([c, a, b])
    assert result == [a, b, c]


# --- watch_and_emit (synchronous: pre-place checkpoint files, idle_timeout=0) ---


def test_watch_and_emit_single_checkpoint(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    types = [e["type"] for e in events]
    assert "epoch.started" in types
    assert "epoch.completed" in types


def test_watch_and_emit_emits_started_before_completed(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    types = [e["type"] for e in events]
    assert types.index("epoch.started") < types.index("epoch.completed")


def test_watch_and_emit_epoch0_started_has_null_in_hash(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    started = next(e for e in events if e["type"] == "epoch.started")
    assert started["checkpoint_in_hash"] is None


def test_watch_and_emit_completed_has_out_hash(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert completed["checkpoint_out_hash"].startswith("sha256:")
    assert completed["source"] == "checkpoint_file_watcher"


def test_watch_and_emit_completed_has_duration_s(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert "duration_s" in completed


def test_watch_and_emit_sidecar_written(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    assert Path(str(ckpt) + ".provenance.json").exists()


def test_watch_and_emit_parent_hash_chain(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt0 = _make_ckpt(tmp_path / "epoch0.ckpt", b"weights0")
    time.sleep(0.01)
    ckpt1 = _make_ckpt(tmp_path / "epoch1.ckpt", b"weights1")

    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)

    sidecar0 = json.loads(Path(str(ckpt0) + ".provenance.json").read_text())
    sidecar1 = json.loads(Path(str(ckpt1) + ".provenance.json").read_text())
    assert sidecar0["parent_hash"] is None
    assert sidecar1["parent_hash"] == sidecar0["checkpoint_hash"]


def test_watch_and_emit_second_epoch_started_carries_prior_hash(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt", b"weights0")
    time.sleep(0.01)
    _make_ckpt(tmp_path / "epoch1.ckpt", b"weights1")

    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)

    events = _read_events(log_dir, "run-1")
    started_events = [e for e in events if e["type"] == "epoch.started"]
    assert started_events[0]["checkpoint_in_hash"] is None
    assert started_events[1]["checkpoint_in_hash"].startswith("sha256:")


def test_watch_and_emit_reads_site_info_for_sidecar(tmp_path):
    log_dir = tmp_path / "provenance"
    (tmp_path / "site_info.json").write_text(
        json.dumps({"hostname": "gpu04", "slot": "slot1", "python": "3.11", "cuda": "12.2",
                    "code_commit": "abc", "gpu_model": "A100", "gpu_count": 1})
    )
    ckpt = _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    sidecar = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert sidecar["produced_at"]["hostname"] == "gpu04"


def test_watch_and_emit_no_checkpoints_exits_immediately(tmp_path):
    log_dir = tmp_path / "provenance"
    # Should exit immediately with idle_timeout=0 and no files
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    assert not (log_dir / "run-1.ndjson").exists()


# --- _parse_val_loss ---


def test_parse_val_loss_lightning_filename(tmp_path):
    p = tmp_path / "epoch=4-step=102968-val_loss=0.3421.ckpt"
    p.write_bytes(b"")
    assert abs(_parse_val_loss(p) - 0.3421) < 1e-9


def test_parse_val_loss_scientific_notation(tmp_path):
    p = tmp_path / "epoch=0-step=1000-val_loss=1.2e-3.ckpt"
    p.write_bytes(b"")
    assert abs(_parse_val_loss(p) - 0.0012) < 1e-9


def test_parse_val_loss_no_match_returns_none(tmp_path):
    p = tmp_path / "checkpoint_epoch_5.ckpt"
    p.write_bytes(b"")
    assert _parse_val_loss(p) is None


def test_watch_and_emit_includes_val_loss_when_present(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch=0-step=1000-val_loss=0.42.ckpt")
    watch_and_emit(tmp_path, "run-1", pattern="*.ckpt", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert abs(completed["val_loss"] - 0.42) < 1e-9
    assert completed["val_loss_source"] == "checkpoint_filename"


def test_watch_and_emit_no_val_loss_when_filename_plain(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert "val_loss" not in completed


def test_watch_and_emit_two_checkpoints_two_completed_events(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt", b"w0")
    time.sleep(0.01)
    _make_ckpt(tmp_path / "epoch1.ckpt", b"w1")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = [e for e in events if e["type"] == "epoch.completed"]
    assert len(completed) == 2


# --- _parse_epoch ---


def test_parse_epoch_lightning_filename(tmp_path):
    assert _parse_epoch(tmp_path / "epoch=4-step=102968-val_loss=0.34.ckpt") == 4


def test_parse_epoch_no_match_returns_none(tmp_path):
    assert _parse_epoch(tmp_path / "checkpoint_epoch_5.ckpt") is None


def test_parse_epoch_zero(tmp_path):
    assert _parse_epoch(tmp_path / "epoch=0-step=100.ckpt") == 0


# --- _read_metrics_csv ---


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_read_metrics_csv_basic(tmp_path):
    _write_metrics_csv(tmp_path / "metrics.csv", [
        {"epoch": "0", "step": "100", "val_loss": "0.543", "train_loss": "0.8"},
        {"epoch": "1", "step": "200", "val_loss": "0.432", "train_loss": "0.7"},
    ])
    result = _read_metrics_csv(tmp_path)
    assert abs(result[0]["val_loss"] - 0.543) < 1e-9
    assert abs(result[1]["val_loss"] - 0.432) < 1e-9


def test_read_metrics_csv_merges_rows_per_epoch(tmp_path):
    _write_metrics_csv(tmp_path / "metrics.csv", [
        {"epoch": "0", "step": "50", "train_loss": "0.8", "val_loss": ""},
        {"epoch": "0", "step": "100", "train_loss": "", "val_loss": "0.543"},
    ])
    result = _read_metrics_csv(tmp_path)
    assert abs(result[0]["val_loss"] - 0.543) < 1e-9
    assert abs(result[0]["train_loss"] - 0.8) < 1e-9


def test_read_metrics_csv_missing_returns_empty(tmp_path):
    assert _read_metrics_csv(tmp_path) == {}


def test_read_metrics_csv_skips_non_numeric(tmp_path):
    _write_metrics_csv(tmp_path / "metrics.csv", [
        {"epoch": "0", "step": "100", "val_loss": "0.5", "mode": "train"},
    ])
    result = _read_metrics_csv(tmp_path)
    assert "mode" not in result[0]
    assert "val_loss" in result[0]


# --- scan_once ---


def test_scan_once_no_checkpoints_is_noop(tmp_path):
    log_dir = tmp_path / "provenance"
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    assert not (log_dir / "run-1.ndjson").exists()


def test_scan_once_emits_epoch_events(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    assert [e["type"] for e in events] == ["epoch.started", "epoch.completed"]


def test_scan_once_uses_parsed_epoch_number(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch=3-step=300.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    assert events[0]["epoch"] == 3


def test_scan_once_writes_sidecar(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    assert Path(str(ckpt) + ".provenance.json").exists()


def test_scan_once_duration_s_with_start_time(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    start = ckpt.stat().st_mtime - 60.0
    scan_once(tmp_path, "run-1", log_dir=log_dir, start_time=start)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert abs(completed["duration_s"] - 60.0) < 1.0


def test_scan_once_no_duration_s_without_start_time(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir, start_time=None)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert "duration_s" not in completed


def test_scan_once_val_loss_from_csv_preferred(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch=0-step=100-val_loss=0.99.ckpt")
    _write_metrics_csv(tmp_path / "metrics.csv", [
        {"epoch": "0", "step": "100", "val_loss": "0.42"},
    ])
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert abs(completed["val_loss"] - 0.42) < 1e-9
    assert completed["val_loss_source"] == "metrics_csv"


def test_scan_once_val_loss_fallback_to_filename(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch=0-step=100-val_loss=0.55.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = next(e for e in events if e["type"] == "epoch.completed")
    assert abs(completed["val_loss"] - 0.55) < 1e-9
    assert completed["val_loss_source"] == "checkpoint_filename"


def test_scan_once_sidecar_training_populated_from_csv(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    _write_metrics_csv(tmp_path / "metrics.csv", [
        {"epoch": "0", "step": "100", "val_loss": "0.42", "train_loss": "0.7"},
    ])
    start = ckpt.stat().st_mtime - 30.0
    scan_once(tmp_path, "run-1", log_dir=log_dir, start_time=start)
    sidecar = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert abs(sidecar["training"]["val_loss"] - 0.42) < 1e-9
    assert abs(sidecar["training"]["train_loss"] - 0.7) < 1e-9
    assert sidecar["training"]["duration_s"] > 0


def test_scan_once_sidecar_training_empty_without_csv(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir, start_time=None)
    sidecar = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert sidecar["training"] == {}


def test_scan_once_extra_sidecar_written(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch=0-step=100.ckpt")
    scan_once(tmp_path, "run-1", log_dir=log_dir, extra_sidecar={"disk_read_mbs": 1200.5})
    sidecar = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert sidecar["extra"] == {"disk_read_mbs": 1200.5}


def test_watch_and_emit_sidecar_training_populated(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt = _make_ckpt(tmp_path / "epoch=0-step=100-val_loss=0.33.ckpt")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    sidecar = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert "duration_s" in sidecar["training"]
    assert abs(sidecar["training"]["val_loss"] - 0.33) < 1e-9


def test_scan_once_parent_hash_chain(tmp_path):
    log_dir = tmp_path / "provenance"
    ckpt0 = _make_ckpt(tmp_path / "epoch=0-step=100.ckpt", b"w0")
    time.sleep(0.01)
    ckpt1 = _make_ckpt(tmp_path / "epoch=1-step=200.ckpt", b"w1")
    scan_once(tmp_path, "run-1", log_dir=log_dir)
    s0 = json.loads(Path(str(ckpt0) + ".provenance.json").read_text())
    s1 = json.loads(Path(str(ckpt1) + ".provenance.json").read_text())
    assert s0["parent_hash"] is None
    assert s1["parent_hash"] == s0["checkpoint_hash"]
