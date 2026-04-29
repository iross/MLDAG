"""Background file-watcher for checkpoint-based provenance events.

Polls a directory for new checkpoint files.  For each new checkpoint (in
mtime order) it:
  1. Emits epoch.started with checkpoint_in_hash (previous checkpoint's hash)
  2. Writes a .provenance.json sidecar
  3. Emits epoch.completed with checkpoint_out_hash and duration_s

Designed to run as a background process in pretrain.sh alongside the training
script.  Exits on SIGTERM (from the EXIT trap in pretrain.sh) or after
idle_timeout seconds with no new checkpoints.

Usage:
    python3 -m mldag.provenance.watcher <watch_dir> <run_id> \\
        [--pattern "*.ckpt"] [--poll-interval 10] [--idle-timeout 3600]

Loss is not captured here because it requires access to training internals.
To include loss, call emit_event("epoch.completed", ..., loss=...) directly
from the training script instead.
"""

import json
import os
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
    site_keys = {"hostname", "slot", "gpu_model", "gpu_count"}
    site_info = {k: v for k, v in data.items() if k in site_keys}
    env_info = {k: v for k, v in data.items() if k not in site_keys}
    return site_info, env_info


def _sorted_by_mtime(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def watch_and_emit(
    watch_dir: str | Path,
    run_id: str,
    *,
    pattern: str = "*.ckpt",
    poll_interval: float = 10.0,
    idle_timeout: float = 3600.0,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
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
            )

            new_hash = write_sidecar(
                ckpt_path,
                run_id,
                epoch_index,
                parent_hash,
                site_info,
                env_info,
                {},
            )

            duration_s = round(time.time() - epoch_start, 3)
            emit_event(
                "epoch.completed",
                run_id,
                log_dir=log_dir,
                epoch=epoch_index,
                checkpoint_out_hash=new_hash,
                duration_s=duration_s,
            )

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
    parser.add_argument("watch_dir", help="Directory to watch for checkpoints")
    parser.add_argument("run_id", help="PROVENANCE_RUN_ID for this run")
    parser.add_argument("--pattern", default="*.ckpt", help="Glob pattern for checkpoints")
    parser.add_argument("--poll-interval", type=float, default=10.0, metavar="S")
    parser.add_argument("--idle-timeout", type=float, default=3600.0, metavar="S")
    args = parser.parse_args()

    log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    watch_and_emit(
        args.watch_dir,
        args.run_id,
        pattern=args.pattern,
        poll_interval=args.poll_interval,
        idle_timeout=args.idle_timeout,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
