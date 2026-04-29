"""HTCondor event log monitor for eviction, hold, and release provenance events.

Tails the HTCondor event log (default: metl.log) and emits provenance events
for the three job lifecycle transitions the SCRIPT PRE/POST pair cannot see:

  004  Evicted  → job.migrated   (job left the execute node; will retry)
  012  Held     → job.held       (job put on hold by HTCondor)
  013  Released → job.released   (job released from hold)

Run this as a DAGMan SERVICE so it stays alive for the life of the DAG:

  SERVICE provenance_monitor provenance_monitor.sub

The ClusterId → run_id mapping is resolved in order:
  1. In-memory cache — populated by tailing the .dagman.out file, which logs
     "Submitting HTCondor Node <name> job(s)..." and
     "N job(s) submitted to cluster <id>." for every submission.
     The job name is cross-referenced against NDJSON files where the PRE
     script wrote a job.submitted event containing both job_name and run_id.
  2. <cluster_id>.run_id marker written by the job at start (requires shared FS)
  3. <cluster_id>.ad ClassAd written by HTCondor on job exit
  4. "unknown:<cluster_id>" fallback
"""

from __future__ import annotations

import json
import os
import re
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

# DAGMan output log patterns
_DAGMAN_SUBMIT_RE = re.compile(r"Submitting HTCondor Node (\S+) job\(s\)\.\.\.")
_DAGMAN_CLUSTER_RE = re.compile(r"\d+ job\(s\) submitted to cluster (\d+)\.")


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


def _scan_dagman_log(
    dagman_path: Path,
    byte_offset: int,
    pending_node: list[str | None],
) -> tuple[list[tuple[str, int]], int]:
    """Scan dagman.out for (job_name, cluster_id) pairs since byte_offset.

    pending_node is a single-element list used to carry the last-seen node
    name across calls so a pair split across two polls is still matched.
    """
    try:
        with open(dagman_path, "rb") as f:
            f.seek(byte_offset)
            new_bytes = f.read()
        new_offset = byte_offset + len(new_bytes)
    except FileNotFoundError:
        return [], byte_offset

    pairs: list[tuple[str, int]] = []
    for line in new_bytes.decode("utf-8", errors="replace").splitlines():
        m = _DAGMAN_SUBMIT_RE.search(line)
        if m:
            pending_node[0] = m.group(1)
            continue
        m = _DAGMAN_CLUSTER_RE.search(line)
        if m and pending_node[0] is not None:
            pairs.append((pending_node[0], int(m.group(1))))
            pending_node[0] = None
    return pairs, new_offset


def _job_name_to_run_id(job_name: str, provenance_log_dir: Path) -> str | None:
    """Search NDJSON files for a job.submitted event matching job_name."""
    for ndjson_path in provenance_log_dir.glob("*.ndjson"):
        try:
            for line in ndjson_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("type") == "job.submitted" and event.get("job_name") == job_name:
                    return event.get("run_id")
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _resolve_run_id(
    cluster_id: int,
    log_dir: Path,
    run_id_cache: dict[int, str],
) -> tuple[str, dict]:
    """Return (run_id, resource_fields) for cluster_id.

    Resolution order: in-memory cache → .run_id marker → .ad ClassAd →
    "unknown:<cluster_id>".
    """
    if cluster_id in run_id_cache:
        return run_id_cache[cluster_id], {}

    run_id_path = log_dir / f"{cluster_id}.run_id"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        run_id_cache[cluster_id] = run_id
        return run_id, {}

    ad_path = log_dir / f"{cluster_id}.ad"
    ad = parse_classad(ad_path)
    if ad:
        run_id = run_id_from_classad(ad) or f"unknown:{cluster_id}"
        if not run_id.startswith("unknown:"):
            run_id_cache[cluster_id] = run_id
        return run_id, resource_fields_from_classad(ad)

    return f"unknown:{cluster_id}", {}


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
    run_id_cache: dict[int, str] | None = None,
) -> int:
    """Process any new lines in log_path, emit events, return new byte offset."""
    if run_id_cache is None:
        run_id_cache = {}
    lines, new_offset = _scan_new_lines(log_path, byte_offset)
    for line in lines:
        parsed = _parse_event_line(line.strip())
        if parsed is None:
            continue
        code, cluster_id, ts = parsed
        event_type = _CODE_TO_EVENT[code]
        run_id, resource = _resolve_run_id(cluster_id, log_dir, run_id_cache)
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
    dagman_log: str | Path | None = None,
    poll_interval: float = 5.0,
) -> None:
    """Tail log_path indefinitely, emitting provenance events for 004/012/013.

    If dagman_log is provided, it is tailed alongside log_path to build a
    cluster_id → run_id cache from DAGMan's submission records and the
    existing NDJSON provenance files.
    """
    log_path = Path(log_path)
    log_dir = Path(log_dir)
    provenance_log_dir = Path(provenance_log_dir)
    dagman_log_path = Path(dagman_log) if dagman_log else None

    byte_offset = 0
    dagman_offset = 0
    pending_node: list[str | None] = [None]
    run_id_cache: dict[int, str] = {}

    while True:
        if dagman_log_path:
            pairs, dagman_offset = _scan_dagman_log(
                dagman_log_path, dagman_offset, pending_node
            )
            for job_name, cluster_id in pairs:
                run_id = _job_name_to_run_id(job_name, provenance_log_dir)
                if run_id:
                    run_id_cache[cluster_id] = run_id

        byte_offset = monitor_once(
            log_path,
            byte_offset,
            log_dir=log_dir,
            provenance_log_dir=provenance_log_dir,
            run_id_cache=run_id_cache,
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
    parser.add_argument(
        "--dagman-log",
        default=None,
        metavar="FILE",
        help="DAGMan output log (*.dagman.out) for cluster_id → run_id mapping",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0, metavar="S")
    args = parser.parse_args()

    provenance_log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    watch_log(
        args.log_file,
        log_dir=args.classad_dir,
        provenance_log_dir=provenance_log_dir,
        dagman_log=args.dagman_log,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
