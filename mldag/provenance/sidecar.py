"""Checkpoint sidecar writer.

Writes a .provenance.json file alongside each checkpoint recording its
content hash, parent hash, run context, and training metrics.  The
parent_hash chain makes the full training lineage reconstructable from
the files alone — no database required.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "1.0"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(
    checkpoint_path: str | Path,
    run_id: str,
    epoch: int,
    parent_hash: Optional[str],
    site_info: dict,
    env_info: dict,
    training_metrics: dict,
    extra: dict | None = None,
) -> str:
    """Write a .provenance.json sidecar alongside the checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        run_id: Stable identifier for the training run (PROVENANCE_RUN_ID).
        epoch: Epoch number this checkpoint represents.
        parent_hash: sha256:<hex> of the checkpoint this was trained from,
            or None for epoch 0.
        site_info: Dict matching the produced_at schema (hostname, site, ts, …).
        env_info: Dict matching the environment schema (python, cuda, …).
        training_metrics: Dict matching the training schema (loss, duration_s, …).

    Returns:
        The prefixed hash string "sha256:<hex>" of the checkpoint file,
        suitable for use as parent_hash in the next epoch's sidecar.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = f"sha256:{sha256_file(checkpoint_path)}"

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_hash": checkpoint_hash,
        "parent_hash": parent_hash,
        "run_id": run_id,
        "epoch": epoch,
        "produced_at": {
            **site_info,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
        "environment": env_info,
        "training": training_metrics,
    }
    if extra:
        sidecar["extra"] = extra

    sidecar_path = Path(str(checkpoint_path) + ".provenance.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return checkpoint_hash
