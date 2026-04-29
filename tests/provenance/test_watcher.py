import json
import time
from pathlib import Path

import pytest

from mldag.provenance.watcher import _load_site_info, _sorted_by_mtime, watch_and_emit


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
            "python": "3.11.4",
            "cuda": "12.2",
            "code_commit": "abc123",
        })
    )
    site, env = _load_site_info(tmp_path)
    assert site["hostname"] == "gpu04.chtc.wisc.edu"
    assert site["gpu_model"] == "A100"
    assert env["python"] == "3.11.4"
    assert env["cuda"] == "12.2"


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


def test_watch_and_emit_two_checkpoints_two_completed_events(tmp_path):
    log_dir = tmp_path / "provenance"
    _make_ckpt(tmp_path / "epoch0.ckpt", b"w0")
    time.sleep(0.01)
    _make_ckpt(tmp_path / "epoch1.ckpt", b"w1")
    watch_and_emit(tmp_path, "run-1", poll_interval=0, idle_timeout=0, log_dir=log_dir)
    events = _read_events(log_dir, "run-1")
    completed = [e for e in events if e["type"] == "epoch.completed"]
    assert len(completed) == 2
