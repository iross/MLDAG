"""Checkpoint-based provenance events: one-shot and polling modes.

One-shot mode (recommended): call after training completes.
    python3 -m mldag.provenance.watcher <watch_dir> <run_id> --one-shot \\
        [--start-time <unix_float>] [--pattern "*.ckpt"]

  Scans watch_dir once for all checkpoint files, reads metrics.csv for
  per-epoch metrics (val_loss etc.), computes duration_s from mtime
  differences, writes .provenance.json sidecars, and emits epoch events.

Polling mode (legacy): run as a background process during training.
    python3 -m mldag.provenance.watcher <watch_dir> <run_id> \\
        [--pattern "*.ckpt"] [--poll-interval 10] [--idle-timeout 3600]

val_loss is sourced from metrics.csv when available; otherwise parsed from
the checkpoint filename (PyTorch Lightning convention: epoch=N-step=M-val_loss=V.ckpt).
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event
from mldag.provenance.sidecar import write_sidecar

_SHUTDOWN = False


def _handle_sigterm(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True


def _load_site_info(watch_dir: Path) -> tuple[dict, dict]:
    """Read site_info.json written by the execute-node capture function.

    Returns (site_info, env_info) split as expected by write_sidecar.
    """
    site_json = watch_dir / "site_info.json"
    if not site_json.exists():
        return {}, {}
    try:
        data = json.loads(site_json.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    site_keys = {"hostname", "slot", "gpu_model", "gpu_count", "gpu_id"}
    site_info = {k: v for k, v in data.items() if k in site_keys}
    env_info = {k: v for k, v in data.items() if k not in site_keys}
    return site_info, env_info


def _parse_val_loss(checkpoint_path: Path) -> float | None:
    """Extract val_loss from a Lightning-style checkpoint filename.

    Matches the pattern: epoch=N-step=M-val_loss=V[.anything].ckpt
    Returns None if the filename does not contain a val_loss field.
    """
    m = re.search(r"val_loss=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", checkpoint_path.name)
    return float(m.group(1)) if m else None


def _sorted_by_mtime(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def _parse_epoch(checkpoint_path: Path) -> int | None:
    """Extract epoch number from a Lightning-style checkpoint filename."""
    m = re.search(r"\bepoch=(\d+)", checkpoint_path.name)
    return int(m.group(1)) if m else None


def _read_metrics_csv(watch_dir: Path) -> dict[int, dict]:
    """Search for metrics.csv under watch_dir; return {epoch: {metric: value}}.

    Multiple rows per epoch are merged; later rows win for duplicate keys.
    Empty and non-numeric values are skipped.
    """
    import csv as _csv

    candidates = sorted(
        watch_dir.rglob("metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}
    result: dict[int, dict] = {}
    try:
        with open(candidates[0], newline="") as f:
            for row in _csv.DictReader(f):
                try:
                    epoch = int(float(row["epoch"]))
                except (KeyError, ValueError, TypeError):
                    continue
                metrics = result.setdefault(epoch, {})
                for k, v in row.items():
                    if k in ("epoch", "step") or not v:
                        continue
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        pass
    except OSError:
        return {}
    return result


def scan_once(
    watch_dir: str | Path,
    run_id: str,
    *,
    pattern: str = "*.ckpt",
    log_dir: str | Path = _DEFAULT_LOG_DIR,
    start_time: float | None = None,
    extra_sidecar: dict | None = None,
) -> None:
    """Single-pass scan: emit epoch events and write sidecars for all checkpoints.

    Args:
        watch_dir: Directory to search recursively for checkpoint files.
        run_id: Stable run identifier (PROVENANCE_RUN_ID).
        pattern: Glob pattern for checkpoint files.
        log_dir: Directory for NDJSON event logs.
        start_time: Unix timestamp of training start, used to compute
            duration_s for the first epoch. If None, duration_s is omitted
            for the first checkpoint.
    """
    watch_dir = Path(watch_dir)
    log_dir = Path(log_dir)

    checkpoints = _sorted_by_mtime(list(watch_dir.rglob(pattern)))
    if not checkpoints:
        return

    csv_metrics = _read_metrics_csv(watch_dir)
    site_info, env_info = _load_site_info(watch_dir)

    parent_hash: str | None = None
    prev_time: float | None = start_time

    for seq_index, ckpt_path in enumerate(checkpoints):
        parsed = _parse_epoch(ckpt_path)
        epoch = parsed if parsed is not None else seq_index
        ckpt_mtime = ckpt_path.stat().st_mtime
        duration_s = round(ckpt_mtime - prev_time, 3) if prev_time is not None else None

        epoch_csv = csv_metrics.get(epoch, {})
        val_loss = epoch_csv.get("val_loss") or _parse_val_loss(ckpt_path)
        val_loss_source = "metrics_csv" if "val_loss" in epoch_csv else "checkpoint_filename"

        training_data: dict = {**epoch_csv}
        if duration_s is not None:
            training_data["duration_s"] = duration_s
        if val_loss is not None and "val_loss" not in epoch_csv:
            training_data["val_loss"] = val_loss

        emit_event(
            "epoch.started", run_id, log_dir=log_dir,
            epoch=epoch, checkpoint_in_hash=parent_hash,
            source="checkpoint_file_watcher",
        )

        new_hash = write_sidecar(
            ckpt_path, run_id, epoch, parent_hash, site_info, env_info, training_data,
            extra_sidecar or {},
        )

        completed_fields: dict = {"checkpoint_out_hash": new_hash, "source": "checkpoint_file_watcher"}
        if duration_s is not None:
            completed_fields["duration_s"] = duration_s
        if val_loss is not None:
            completed_fields["val_loss"] = val_loss
            completed_fields["val_loss_source"] = val_loss_source

        emit_event("epoch.completed", run_id, log_dir=log_dir, epoch=epoch, **completed_fields)

        parent_hash = new_hash
        prev_time = ckpt_mtime


def watch_and_emit(
    watch_dir: str | Path,
    run_id: str,
    *,
    pattern: str = "*.ckpt",
    poll_interval: float = 10.0,
    idle_timeout: float = 3600.0,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
    extra_sidecar: dict | None = None,
) -> None:
    """Poll watch_dir until SIGTERM or idle_timeout with no new checkpoints.

    Args:
        watch_dir: Directory to watch (recursively) for new checkpoint files.
        run_id: Stable run identifier (PROVENANCE_RUN_ID).
        pattern: Glob pattern for checkpoint files.
        poll_interval: Seconds between directory scans.
        idle_timeout: Exit after this many seconds with no new checkpoints.
        log_dir: Directory for NDJSON event logs.
    """
    signal.signal(signal.SIGTERM, _handle_sigterm)

    watch_dir = Path(watch_dir)
    log_dir = Path(log_dir)

    seen: set[Path] = set()
    parent_hash: str | None = None
    epoch_index: int = 0
    last_new_file_time: float = time.monotonic()

    while not _SHUTDOWN:
        current = set(watch_dir.rglob(pattern))
        new_files = _sorted_by_mtime(list(current - seen))

        for ckpt_path in new_files:
            site_info, env_info = _load_site_info(watch_dir)
            epoch_start = time.time()

            emit_event(
                "epoch.started",
                run_id,
                log_dir=log_dir,
                epoch=epoch_index,
                checkpoint_in_hash=parent_hash,
                source="checkpoint_file_watcher",
            )

            duration_s = round(time.time() - epoch_start, 3)
            val_loss = _parse_val_loss(ckpt_path)
            training_data: dict = {"duration_s": duration_s}
            if val_loss is not None:
                training_data["val_loss"] = val_loss

            new_hash = write_sidecar(
                ckpt_path,
                run_id,
                epoch_index,
                parent_hash,
                site_info,
                env_info,
                training_data,
                extra_sidecar or {},
            )

            completed_fields: dict = {
                "checkpoint_out_hash": new_hash,
                "duration_s": duration_s,
                "source": "checkpoint_file_watcher",
            }
            if val_loss is not None:
                completed_fields["val_loss"] = val_loss
                completed_fields["val_loss_source"] = "checkpoint_filename"
            emit_event("epoch.completed", run_id, log_dir=log_dir, epoch=epoch_index, **completed_fields)

            parent_hash = new_hash
            epoch_index += 1
            last_new_file_time = time.monotonic()

        seen.update(new_files)

        if time.monotonic() - last_new_file_time > idle_timeout:
            break

        time.sleep(poll_interval)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint provenance watcher")
    parser.add_argument("watch_dir", help="Directory to watch/scan for checkpoints")
    parser.add_argument("run_id", help="PROVENANCE_RUN_ID for this run")
    parser.add_argument("--pattern", default="*.ckpt", help="Glob pattern for checkpoints")
    parser.add_argument("--one-shot", action="store_true",
                        help="Scan once and exit (run after training completes)")
    parser.add_argument("--start-time", type=float, default=None, metavar="UNIX_TS",
                        help="Unix timestamp of training start for epoch 0 duration_s")
    parser.add_argument("--poll-interval", type=float, default=10.0, metavar="S")
    parser.add_argument("--idle-timeout", type=float, default=3600.0, metavar="S")
    parser.add_argument(
        "--extra-sidecar-json", default=None, metavar="FILE",
        help="JSON file whose contents are merged into each sidecar's 'extra' field",
    )
    args = parser.parse_args()

    log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    extra_sidecar: dict = {}
    if args.extra_sidecar_json:
        try:
            extra_sidecar = json.loads(Path(args.extra_sidecar_json).read_text())
        except (OSError, json.JSONDecodeError):
            pass

    if args.one_shot:
        scan_once(
            args.watch_dir,
            args.run_id,
            pattern=args.pattern,
            log_dir=log_dir,
            start_time=args.start_time,
            extra_sidecar=extra_sidecar,
        )
    else:
        watch_and_emit(
            args.watch_dir,
            args.run_id,
            pattern=args.pattern,
            poll_interval=args.poll_interval,
            idle_timeout=args.idle_timeout,
            log_dir=log_dir,
            extra_sidecar=extra_sidecar,
        )


if __name__ == "__main__":
    main()
