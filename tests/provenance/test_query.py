import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mldag.provenance.query import app, query_run, walk_lineage
from mldag.provenance.sidecar import write_sidecar

runner = CliRunner()

SITE = {"hostname": "gpu04", "slot": "slot1", "gpu_model": "A100", "gpu_count": 1}
ENV = {"python": "3.11", "cuda": "12.2", "code_commit": "abc"}


def _make_ckpt(path: Path, content: bytes = b"weights") -> Path:
    path.write_bytes(content)
    return path


def _build_chain(tmp_path: Path, n: int) -> list[Path]:
    """Create n checkpoints with a valid sidecar chain."""
    ckpts = []
    parent_hash = None
    for i in range(n):
        ckpt = _make_ckpt(tmp_path / f"epoch{i}.ckpt", f"weights{i}".encode())
        parent_hash = write_sidecar(ckpt, "run-test", i, parent_hash, SITE, ENV, {})
        ckpts.append(ckpt)
    return ckpts


def _write_events(log_dir: Path, run_id: str, events: list[dict]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.ndjson"
    with open(log_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# --- walk_lineage ---


def test_walk_lineage_single_epoch(tmp_path):
    ckpts = _build_chain(tmp_path, 1)
    chain = walk_lineage(ckpts[0])
    assert len(chain) == 1
    assert chain[0]["epoch"] == 0
    assert chain[0]["parent_hash"] is None


def test_walk_lineage_three_epochs_oldest_first(tmp_path):
    ckpts = _build_chain(tmp_path, 3)
    chain = walk_lineage(ckpts[2])
    assert len(chain) == 3
    assert chain[0]["epoch"] == 0
    assert chain[1]["epoch"] == 1
    assert chain[2]["epoch"] == 2


def test_walk_lineage_parent_hash_chain_is_intact(tmp_path):
    ckpts = _build_chain(tmp_path, 3)
    chain = walk_lineage(ckpts[2])
    assert chain[0]["parent_hash"] is None
    assert chain[1]["parent_hash"] == chain[0]["checkpoint_hash"]
    assert chain[2]["parent_hash"] == chain[1]["checkpoint_hash"]


def test_walk_lineage_missing_sidecar_raises(tmp_path):
    ckpt = _make_ckpt(tmp_path / "epoch0.ckpt")
    with pytest.raises(FileNotFoundError, match="No sidecar"):
        walk_lineage(ckpt)


def test_walk_lineage_corrupt_sidecar_raises(tmp_path):
    ckpt = _make_ckpt(tmp_path / "epoch0.ckpt")
    (tmp_path / "epoch0.ckpt.provenance.json").write_text("not json")
    with pytest.raises(ValueError, match="Corrupt sidecar"):
        walk_lineage(ckpt)


def test_walk_lineage_missing_parent_raises(tmp_path):
    ckpts = _build_chain(tmp_path, 2)
    # Remove epoch0 sidecar to break the chain
    (tmp_path / "epoch0.ckpt.provenance.json").unlink()
    with pytest.raises(FileNotFoundError):
        walk_lineage(ckpts[1])


# --- query_run ---


def test_query_run_returns_events_sorted_by_ts(tmp_path):
    log_dir = tmp_path / "provenance"
    events = [
        {"schema_version": "1.0", "type": "epoch.completed", "run_id": "r1", "ts": "2026-01-01T01:00:00+00:00"},
        {"schema_version": "1.0", "type": "job.submitted",   "run_id": "r1", "ts": "2026-01-01T00:00:00+00:00"},
        {"schema_version": "1.0", "type": "epoch.started",   "run_id": "r1", "ts": "2026-01-01T00:30:00+00:00"},
    ]
    _write_events(log_dir, "r1", events)
    result = query_run("r1", log_dir)
    assert [e["type"] for e in result] == ["job.submitted", "epoch.started", "epoch.completed"]


def test_query_run_missing_log_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No event log"):
        query_run("nonexistent-run", tmp_path)


def test_query_run_skips_malformed_lines(tmp_path):
    log_dir = tmp_path / "provenance"
    log_dir.mkdir()
    (log_dir / "r2.ndjson").write_text(
        '{"schema_version":"1.0","type":"job.submitted","run_id":"r2","ts":"2026-01-01T00:00:00+00:00"}\n'
        "not json\n"
        '{"schema_version":"1.0","type":"job.completed","run_id":"r2","ts":"2026-01-01T01:00:00+00:00"}\n'
    )
    result = query_run("r2", log_dir)
    assert len(result) == 2


# --- CLI: lineage ---


def test_cli_lineage_human_output(tmp_path):
    ckpts = _build_chain(tmp_path, 2)
    result = runner.invoke(app, ["lineage", str(ckpts[1])])
    assert result.exit_code == 0
    assert "epoch" in result.output
    assert "Lineage" in result.output


def test_cli_lineage_json_output(tmp_path):
    ckpts = _build_chain(tmp_path, 2)
    result = runner.invoke(app, ["lineage", str(ckpts[1]), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 2


def test_cli_lineage_missing_checkpoint_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["lineage", str(tmp_path / "missing.ckpt")])
    assert result.exit_code != 0


# --- CLI: events ---


def test_cli_events_human_output(tmp_path):
    log_dir = tmp_path / "provenance"
    _write_events(log_dir, "run-xyz", [
        {"schema_version": "1.0", "type": "job.submitted", "run_id": "run-xyz",
         "ts": "2026-01-01T00:00:00+00:00"},
    ])
    result = runner.invoke(app, ["events", "run-xyz", "--log-dir", str(log_dir)])
    assert result.exit_code == 0
    assert "job.submitted" in result.output


def test_cli_events_json_output(tmp_path):
    log_dir = tmp_path / "provenance"
    _write_events(log_dir, "run-xyz", [
        {"schema_version": "1.0", "type": "job.submitted", "run_id": "run-xyz",
         "ts": "2026-01-01T00:00:00+00:00"},
    ])
    result = runner.invoke(app, ["events", "run-xyz", "--log-dir", str(log_dir), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["type"] == "job.submitted"


def test_cli_events_missing_run_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["events", "no-such-run", "--log-dir", str(tmp_path)])
    assert result.exit_code != 0
