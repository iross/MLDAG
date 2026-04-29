"""HTCondor event log monitor for job lifecycle provenance events.

Tails the HTCondor event log (default: metl.log) and emits provenance events
for job lifecycle transitions the SCRIPT PRE/POST pair cannot see:

  001  Executing                  → job.executing
  004  Evicted                    → job.migrated
  009  Aborted                    → job.aborted
  012  Held                       → job.held
  013  Released                   → job.released
  023  Reconnected                → job.reconnected
  040  Started/Finished transferring input/output files
       → transfer.input.started / transfer.input.completed
       → transfer.output.started / transfer.output.completed
       (HTCondor uses code 040 for all four; direction is parsed from the description)

Run this as a DAGMan SERVICE so it stays alive for the life of the DAG:

  SERVICE provenance_monitor provenance_monitor.sub

The ClusterId → run_id mapping is resolved in order:
  1. In-memory cache — populated by reading event 000 (job submitted) blocks
     from metl.log itself, which contains "DAG Node: <name>" body lines when
     submitted via DAGMan.  The node name is cross-referenced against NDJSON
     files where the PRE script wrote a job.submitted event with both
     job_name and run_id.
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

# Matches any HTCondor event log header line; groups: (code, cluster_id, ...)
_TS_NEW = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_TS_OLD = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})")
_ANY_HEADER_RE = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\)")
_DAGNODE_RE = re.compile(r"DAG Node:\s+(\S+)")

# Codes that emit provenance events
_CODES = {"001", "004", "009", "012", "013", "023", "040"}

_CODE_TO_EVENT = {
    "001": "job.executing",
    "004": "job.migrated",
    "009": "job.aborted",
    "012": "job.held",
    "013": "job.released",
    "023": "job.reconnected",
    # 040 covers all file transfer events; event type is determined from description text
}

# HTCondor uses code 040 for all file transfer events; direction and phase are in the description
_TRANSFER_RE = re.compile(r"\b(Started|Finished) transferring (input|output) files", re.IGNORECASE)
_TRANSFER_EVENT_MAP = {
    ("started",  "input"):  "transfer.input.started",
    ("finished", "input"):  "transfer.input.completed",
    ("started",  "output"): "transfer.output.started",
    ("finished", "output"): "transfer.output.completed",
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
    multiline_state: dict | None = None,
    pending_lookups: dict[int, str] | None = None,
) -> int:
    """Process any new lines in log_path, emit events, return new byte offset.

    multiline_state is a dict that persists across calls to track partial
    event 000 blocks (the "DAG Node:" body line may arrive in a later poll).
    pending_lookups maps cluster_id → job_name for DAG Node entries whose
    NDJSON job.submitted record wasn't visible yet; retried on every call.
    Pass the same dicts on every call from watch_log.
    """
    if run_id_cache is None:
        run_id_cache = {}
    if multiline_state is None:
        multiline_state = {"cluster_id": None}
    if pending_lookups is None:
        pending_lookups = {}

    # Retry any cluster_id → job_name mappings that weren't resolved last poll.
    for cluster_id, job_name in list(pending_lookups.items()):
        run_id = _job_name_to_run_id(job_name, provenance_log_dir)
        if run_id:
            run_id_cache[cluster_id] = run_id
            del pending_lookups[cluster_id]

    lines, new_offset = _scan_new_lines(log_path, byte_offset)
    for line in lines:
        stripped = line.strip()

        # Track event 000 blocks to build cluster_id → run_id cache.
        # Event 000 body contains "DAG Node: <name>" when submitted by DAGMan.
        hm = _ANY_HEADER_RE.match(stripped)
        if hm:
            if hm.group(1) == "000":
                multiline_state["cluster_id"] = int(hm.group(2))
            else:
                multiline_state["cluster_id"] = None

        dn_m = _DAGNODE_RE.search(stripped)
        if dn_m and multiline_state.get("cluster_id") is not None:
            cluster_id = multiline_state["cluster_id"]
            if cluster_id not in run_id_cache:
                job_name = dn_m.group(1)
                run_id = _job_name_to_run_id(job_name, provenance_log_dir)
                if run_id:
                    run_id_cache[cluster_id] = run_id
                else:
                    pending_lookups[cluster_id] = job_name
            multiline_state["cluster_id"] = None

        # Emit provenance events for tracked codes
        parsed = _parse_event_line(stripped)
        if parsed is None:
            continue
        code, cluster_id, ts = parsed
        if code == "040":
            tm = _TRANSFER_RE.search(stripped)
            if tm is None:
                continue
            event_type = _TRANSFER_EVENT_MAP[(tm.group(1).lower(), tm.group(2).lower())]
        else:
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


def _load_offset(offset_path: Path, log_path: Path) -> int:
    """Return the saved byte offset, or 0 if none or the log was recreated."""
    try:
        saved = int(offset_path.read_text().strip())
        # If the log file is smaller than the saved offset it was recreated; reset.
        if log_path.exists() and saved > log_path.stat().st_size:
            return 0
        return saved
    except (OSError, ValueError):
        return 0


def _save_offset(offset_path: Path, byte_offset: int) -> None:
    try:
        offset_path.parent.mkdir(parents=True, exist_ok=True)
        offset_path.write_text(str(byte_offset))
    except OSError:
        pass


def watch_log(
    log_path: str | Path,
    *,
    log_dir: str | Path,
    provenance_log_dir: str | Path = _DEFAULT_LOG_DIR,
    poll_interval: float = 5.0,
) -> None:
    """Tail log_path indefinitely, emitting provenance events."""
    log_path = Path(log_path)
    log_dir = Path(log_dir)
    provenance_log_dir = Path(provenance_log_dir)

    offset_path = provenance_log_dir / ".log_monitor.offset"
    byte_offset = _load_offset(offset_path, log_path)
    run_id_cache: dict[int, str] = {}
    multiline_state: dict = {"cluster_id": None}
    pending_lookups: dict[int, str] = {}

    while True:
        byte_offset = monitor_once(
            log_path,
            byte_offset,
            log_dir=log_dir,
            provenance_log_dir=provenance_log_dir,
            run_id_cache=run_id_cache,
            multiline_state=multiline_state,
            pending_lookups=pending_lookups,
        )
        _save_offset(offset_path, byte_offset)
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
