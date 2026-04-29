import json
import hashlib
from pathlib import Path

import pytest

from mldag.provenance.sidecar import sha256_file, write_sidecar, SCHEMA_VERSION


SITE_INFO = {"site": "chtc-gpu04", "hostname": "gpu04.chtc.wisc.edu", "slot": "slot1_1"}
ENV_INFO = {"python": "3.11.4", "cuda": "12.2", "framework": "pytorch", "framework_version": "2.3.0", "code_commit": "f4a88bc"}
METRICS = {"loss": 0.342, "epoch_duration_s": 847}


def _make_checkpoint(tmp_path: Path, name: str = "epoch.ckpt", content: bytes = b"fake checkpoint data") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# --- sha256_file ---

def test_sha256_file_matches_hashlib(tmp_path):
    ckpt = _make_checkpoint(tmp_path, content=b"some bytes")
    expected = hashlib.sha256(b"some bytes").hexdigest()
    assert sha256_file(ckpt) == expected


def test_sha256_file_accepts_str_path(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    assert sha256_file(str(ckpt)) == sha256_file(ckpt)


def test_sha256_file_different_contents_differ(tmp_path):
    a = _make_checkpoint(tmp_path, "a.ckpt", b"aaa")
    b = _make_checkpoint(tmp_path, "b.ckpt", b"bbb")
    assert sha256_file(a) != sha256_file(b)


# --- write_sidecar ---

def test_write_sidecar_creates_file(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    write_sidecar(ckpt, "run-abc", 0, None, SITE_INFO, ENV_INFO, METRICS)
    assert Path(str(ckpt) + ".provenance.json").exists()


def test_write_sidecar_schema_version(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    write_sidecar(ckpt, "run-abc", 0, None, SITE_INFO, ENV_INFO, METRICS)
    data = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert data["schema_version"] == SCHEMA_VERSION


def test_write_sidecar_required_fields_present(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    write_sidecar(ckpt, "run-abc", 3, "sha256:deadbeef", SITE_INFO, ENV_INFO, METRICS)
    data = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    for field in ("schema_version", "checkpoint_hash", "parent_hash", "run_id", "epoch",
                  "produced_at", "environment", "training"):
        assert field in data, f"missing field: {field}"


def test_write_sidecar_epoch0_parent_hash_none(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    write_sidecar(ckpt, "run-abc", 0, None, SITE_INFO, ENV_INFO, METRICS)
    data = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert data["parent_hash"] is None


def test_write_sidecar_nonzero_epoch_carries_parent_hash(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    write_sidecar(ckpt, "run-abc", 5, "sha256:abc123", SITE_INFO, ENV_INFO, METRICS)
    data = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert data["parent_hash"] == "sha256:abc123"


def test_write_sidecar_checkpoint_hash_is_prefixed_sha256(tmp_path):
    ckpt = _make_checkpoint(tmp_path, content=b"model weights")
    write_sidecar(ckpt, "run-abc", 0, None, SITE_INFO, ENV_INFO, METRICS)
    data = json.loads(Path(str(ckpt) + ".provenance.json").read_text())
    assert data["checkpoint_hash"].startswith("sha256:")
    expected = "sha256:" + hashlib.sha256(b"model weights").hexdigest()
    assert data["checkpoint_hash"] == expected


def test_write_sidecar_returns_hash_for_chaining(tmp_path):
    ckpt = _make_checkpoint(tmp_path, content=b"weights")
    returned = write_sidecar(ckpt, "run-abc", 0, None, SITE_INFO, ENV_INFO, METRICS)
    assert returned == "sha256:" + hashlib.sha256(b"weights").hexdigest()


def test_write_sidecar_parent_chain_across_epochs(tmp_path):
    ckpt0 = _make_checkpoint(tmp_path, "epoch0.ckpt", b"epoch0 weights")
    ckpt1 = _make_checkpoint(tmp_path, "epoch1.ckpt", b"epoch1 weights")

    hash0 = write_sidecar(ckpt0, "run-abc", 0, None, SITE_INFO, ENV_INFO, METRICS)
    write_sidecar(ckpt1, "run-abc", 1, hash0, SITE_INFO, ENV_INFO, METRICS)

    data0 = json.loads(Path(str(ckpt0) + ".provenance.json").read_text())
    data1 = json.loads(Path(str(ckpt1) + ".provenance.json").read_text())

    assert data0["parent_hash"] is None
    assert data1["parent_hash"] == data0["checkpoint_hash"]
