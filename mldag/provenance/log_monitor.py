"""HTCondor event log monitor for eviction, hold, and release provenance events.

Tails the HTCondor event log (default: metl.log) and emits provenance events
for the three job lifecycle transitions the SCRIPT PRE/POST pair cannot see:

  004  Evicted  → job.migrated   (job left the execute node; will retry)
  012  Held     → job.held       (job put on hold by HTCondor)
  013  Released → job.released   (job released from hold)

Run this as a DAGMan SERVICE so it stays alive for the life of the DAG:

  SERVICE provenance_monitor provenance_monitor.sub

The ClusterId → run_id mapping is resolved by reading the per-cluster
ClassAd files written by job_ad_file (output/provenance/<ClusterId>.ad).
For hold events the ClassAd is not yet written; run_id falls back to
"unknown:<ClusterId>" so the event can still be correlated manually.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event
from mldag.provenance.post import parse_classad, run_id_from_classad, resource_fields_from_classad

# HTCondor event log timestamp formats (new-style YYYY-MM-DD, legacy MM/DD)
_TS_NEW = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_TS_OLD = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})")

_CODES = {"004", "012", "013"}

_CODE_TO_EVENT = {
    "004": "job.migrated",
    "012": "job.held",
    "013": "job.released",
}


def _parse_event_line(line: str) -> tuple[str, int, datetime] | None:
    """Return (event_code, cluster_id, timestamp) or None if not a relevant event."""
    m = _TS_NEW.match(line)
    if m:
        code, cluster_id, ts_str = m.group(1), int(m.group(2)), m.group(3)
        if code not in _CODES:
            return None
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return code, cluster_id, ts

    m = _TS_OLD.match(line)
    if m:
        code, cluster_id = m.group(1), int(m.group(2))
        if code not in _CODES:
            return None
        year = datetime.now().year
        ts_str = f"{year}/{m.group(3)} {m.group(4)}"
        ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return code, cluster_id, ts

    return None


def _resolve_run_id(cluster_id: int, log_dir: Path) -> tuple[str, dict]:
    """Return (run_id, resource_fields) from the cluster's ClassAd, if available."""
    ad_path = log_dir / f"{cluster_id}.ad"
    ad = parse_classad(ad_path)
    run_id = run_id_from_classad(ad) if ad else f"unknown:{cluster_id}"
    return run_id, resource_fields_from_classad(ad)


def _scan_new_lines(log_path: Path, byte_offset: int) -> tuple[list[str], int]:
    """Read any new content from log_path beyond byte_offset."""
    try:
        with open(log_path, "rb") as f:
            f.seek(byte_offset)
            new_bytes = f.read()
        new_offset = byte_offset + len(new_bytes)
        lines = new_bytes.decode("utf-8", errors="replace").splitlines()
        return lines, new_offset
    except FileNotFoundError:
        return [], byte_offset


def monitor_once(
    log_path: Path,
    byte_offset: int,
    *,
    log_dir: Path,
    provenance_log_dir: Path,
) -> int:
    """Process any new lines in log_path, emit events, return new byte offset."""
    lines, new_offset = _scan_new_lines(log_path, byte_offset)
    for line in lines:
        parsed = _parse_event_line(line.strip())
        if parsed is None:
            continue
        code, cluster_id, ts = parsed
        event_type = _CODE_TO_EVENT[code]
        run_id, resource = _resolve_run_id(cluster_id, log_dir)
        emit_event(
            event_type,
            run_id,
            log_dir=provenance_log_dir,
            cluster_id=cluster_id,
            condor_event_ts=ts.isoformat(),
            source="htcondor_event_log",
            **resource,
        )
    return new_offset


def watch_log(
    log_path: str | Path,
    *,
    log_dir: str | Path,
    provenance_log_dir: str | Path = _DEFAULT_LOG_DIR,
    poll_interval: float = 5.0,
) -> None:
    """Tail log_path indefinitely, emitting provenance events for 004/012/013."""
    log_path = Path(log_path)
    log_dir = Path(log_dir)
    provenance_log_dir = Path(provenance_log_dir)
    byte_offset = 0

    while True:
        byte_offset = monitor_once(
            log_path,
            byte_offset,
            log_dir=log_dir,
            provenance_log_dir=provenance_log_dir,
        )
        time.sleep(poll_interval)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HTCondor event log provenance monitor")
    parser.add_argument("--log-file", default="metl.log", help="HTCondor event log to watch")
    parser.add_argument(
        "--classad-dir",
        default="output/provenance",
        help="Directory containing per-cluster .ad files (job_ad_file output)",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0, metavar="S")
    args = parser.parse_args()

    provenance_log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    watch_log(
        args.log_file,
        log_dir=args.classad_dir,
        provenance_log_dir=provenance_log_dir,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
