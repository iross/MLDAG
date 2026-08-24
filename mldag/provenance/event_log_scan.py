"""Standalone HTCondor event log scanner -- no DAGMan provenance instrumentation required.

log_monitor.py's parsing (timestamps, the 005 resource-usage banner, DAGNodeName
lookup) is wired directly into DAGMan-specific run_id resolution and NDJSON
emission. A batch of jobs run without the PRE/POST provenance scripts (no
job.submitted NDJSON, no run_id) still produces a normal HTCondor event log,
and there's no way to get duration/site/resource-usage out of that log without
the full pipeline.

scan_event_log() reuses log_monitor.py's regexes/helpers directly (so the two
never drift on event-log syntax) but needs neither NDJSON events nor a run_id:
it summarizes each job's lifecycle straight from the event log. Jobs are keyed
by (cluster_id, proc_id), not cluster_id alone -- a single cluster can hold
many procs (e.g. `queue N` job arrays; confirmed against a real production
log with 11 clusters spanning 275 cluster.proc pairs), and log_monitor.py's
own _ANY_HEADER_RE discards proc_id entirely since DAGMan-submitted jobs are
(at least today) always cluster.0, so it never needed it.

Enrichment with run_id/job_name is opportunistic -- pass log_dir (a
classad/.run_id marker directory) and/or provenance_log_dir (an NDJSON
directory) if they exist for the run that produced this log, and matching
jobs are enriched; anything that doesn't resolve (no directories passed, no
matching files, job never went through the provenance pipeline) is left keyed
by bare cluster_id/proc_id, silently. Note run_id/.ad resolution is still
cluster_id-keyed (matching how post.py/log_monitor.py write `<cluster_id>.ad`
and `.run_id` files), so every proc in a cluster resolves to the same run_id.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from mldag.provenance.log_monitor import (
    _DAGNODE_RE,
    _SLOTNAME_RE,
    _TS_NEW,
    _TS_OLD,
    _accumulate_usage_field,
    _refresh_job_submitted_index,
    _resolve_run_id,
    site_from_slotname,
)

# Unlike log_monitor._ANY_HEADER_RE, captures proc_id (group 3) too -- see
# module docstring for why that distinction matters here.
_JOB_HEADER_RE = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\)")

# Codes whose body we care about; anything else's body lines are still
# collected (in case a block never terminates with "...") but discarded.
_HELD_CODE = "012"
_RELEASED_CODE = "013"
_ABORTED_CODE = "009"


def _parse_any_ts(header_line: str) -> str | None:
    """Return the ISO timestamp on any HTCondor event header line, or None."""
    m = _TS_NEW.match(header_line)
    if m:
        ts = datetime.strptime(m.group(3), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return ts.isoformat()
    m = _TS_OLD.match(header_line)
    if m:
        year = datetime.now().year
        ts = datetime.strptime(f"{year}/{m.group(3)} {m.group(4)}", "%Y/%m/%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        return ts.isoformat()
    return None


def _new_record(cluster_id: int, proc_id: int) -> dict:
    return {
        "cluster_id": cluster_id,
        "proc_id": proc_id,
        "held_count": 0,
        "released_count": 0,
        "aborted": False,
    }


def _flush_block(
    records: dict[tuple[int, int], dict],
    code: str,
    cluster_id: int,
    proc_id: int,
    ts: str | None,
    lines: list[str],
) -> None:
    rec = records.setdefault((cluster_id, proc_id), _new_record(cluster_id, proc_id))
    if code == "000":
        for line in lines:
            m = _DAGNODE_RE.search(line)
            if m:
                rec["job_name"] = m.group(1)
                break
    elif code == "001":
        rec["executing_ts"] = ts
        for line in lines:
            m = _SLOTNAME_RE.match(line)
            if m:
                rec["execute_host"] = m.group(1)
                rec["site"] = site_from_slotname(m.group(1))
                break
    elif code == "005":
        usage: dict = {}
        for line in lines:
            _accumulate_usage_field(line, usage)
        rec["terminated_ts"] = ts
        rec.update(usage)
    elif code == _HELD_CODE:
        rec["held_count"] += 1
    elif code == _RELEASED_CODE:
        rec["released_count"] += 1
    elif code == _ABORTED_CODE:
        rec["aborted"] = True


def _status(rec: dict) -> str:
    if rec.get("aborted"):
        return "aborted"
    if "terminated_ts" in rec:
        return "completed"
    if rec["held_count"] > rec["released_count"]:
        return "held"
    if "executing_ts" in rec:
        return "executing"
    return "submitted"


def scan_event_log(
    log_path: str | Path,
    *,
    log_dir: str | Path | None = None,
    provenance_log_dir: str | Path | None = None,
) -> list[dict]:
    """Summarize every job in log_path: duration, execute host/site, resource usage.

    Works on any HTCondor event log, instrumented by the DAGMan provenance
    pipeline or not -- unlike log_monitor.monitor_once, this never touches
    NDJSON or emits provenance events; it only reads log_path.

    Args:
        log_path: HTCondor event log to scan (read once, in full).
        log_dir: Optional directory of per-cluster `<id>.run_id` markers
            and/or `<id>.ad` ClassAds (as written by the DAGMan provenance
            pipeline). If given and a match exists for a cluster_id, its
            run_id and any resource fields (e.g. resource_name) are merged
            in. Silently skipped for any cluster_id with no match.
        provenance_log_dir: Optional NDJSON event directory. If given, a
            job_name recovered from this log's own DAGNodeName lines is
            looked up against `job.submitted` records there to resolve a
            run_id. Silently skipped when unavailable or unmatched.

    Returns:
        One dict per (cluster_id, proc_id) seen, sorted by cluster_id then
        proc_id, containing at least "cluster_id", "proc_id", and "status";
        other keys ("run_id", "job_name", "site", "execute_host",
        "executing_ts", "terminated_ts", "wall_time_s", "cpu_usage",
        "peak_memory_mb", "gpu_usage", "gpu_ids", "resource_name") are
        present only when resolved.
    """
    log_path = Path(log_path)
    records: dict[tuple[int, int], dict] = {}

    block_code: str | None = None
    block_cluster_id: int | None = None
    block_proc_id: int | None = None
    block_ts: str | None = None
    block_lines: list[str] = []

    def flush() -> None:
        nonlocal block_code, block_cluster_id, block_proc_id, block_ts, block_lines
        if block_code is not None and block_cluster_id is not None and block_proc_id is not None:
            _flush_block(records, block_code, block_cluster_id, block_proc_id, block_ts, block_lines)
        block_code, block_cluster_id, block_proc_id, block_ts, block_lines = None, None, None, None, []

    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "...":
            flush()
            continue
        header = _JOB_HEADER_RE.match(line)
        if header:
            flush()  # tolerate a missing "..." terminator on the previous block
            block_code = header.group(1)
            block_cluster_id = int(header.group(2))
            block_proc_id = int(header.group(3))
            block_ts = _parse_any_ts(line)
            block_lines = [line]
            continue
        if block_code is not None:
            block_lines.append(line)
    flush()

    if provenance_log_dir is not None:
        job_index: dict[str, str] = {}
        _refresh_job_submitted_index(Path(provenance_log_dir), job_index, {})
        for rec in records.values():
            job_name = rec.get("job_name")
            if job_name and job_name in job_index:
                rec["run_id"] = job_index[job_name]

    if log_dir is not None:
        log_dir = Path(log_dir)
        cache: dict[int, str] = {}
        for rec in records.values():
            run_id, resource = _resolve_run_id(rec["cluster_id"], log_dir, cache)
            if not run_id.startswith("unknown:"):
                rec.setdefault("run_id", run_id)
            for key, value in resource.items():
                rec.setdefault(key, value)

    for rec in records.values():
        rec["status"] = _status(rec)

    return [records[key] for key in sorted(records)]
