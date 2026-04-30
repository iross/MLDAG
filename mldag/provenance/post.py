"""POST-script logic for DAGMan job lifecycle provenance.

Called by DAGMan after each training job exits.  Reads the HTCondor ClassAd
written by job_ad_file, extracts resource usage and the run ID, then appends
a job.completed or job.failed event to the per-run NDJSON log.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event


def parse_classad(path: Path | str) -> dict:
    """Parse an HTCondor ClassAd file into a plain dict.

    Unknown value types are kept as strings.  Returns an empty dict if the
    file does not exist (job may have been evicted before writing the ad).
    """
    attrs: dict = {}
    try:
        for line in Path(path).read_text().splitlines():
            m = re.match(r"^(\w+)\s*=\s*(.+)$", line.strip())
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip()
            if raw.startswith('"') and raw.endswith('"'):
                attrs[key] = raw[1:-1]
            else:
                try:
                    attrs[key] = int(raw)
                except ValueError:
                    try:
                        attrs[key] = float(raw)
                    except ValueError:
                        attrs[key] = raw
    except FileNotFoundError:
        pass
    return attrs


def run_id_from_classad(ad: dict) -> str:
    """Extract PROVENANCE_RUN_ID from the ClassAd Environment string."""
    m = re.search(r"PROVENANCE_RUN_ID=([^\s\"]+)", ad.get("Environment", ""))
    return m.group(1) if m else "unknown"


def resource_fields_from_classad(ad: dict) -> dict:
    """Return the subset of ClassAd fields that map to the provenance schema."""
    mapping = {
        "RemoteWallClockTime": "wall_time_s",
        "CPUsUsage": "cpu_usage",
        "MemoryUsage": "peak_memory_mb",
        "GPUsUsage": "gpu_usage",
        "GLIDEIN_ResourceName": "resource_name",
    }
    return {schema_key: ad[ad_key] for ad_key, schema_key in mapping.items() if ad_key in ad}


def emit_post_event(
    job_name: str,
    exit_code: int,
    cluster_id: str,
    *,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
) -> None:
    """Emit job.completed or job.failed using data from the HTCondor ClassAd."""
    log_dir = Path(log_dir)
    ad = parse_classad(log_dir / f"{cluster_id}.ad")
    run_id = run_id_from_classad(ad)
    resource = resource_fields_from_classad(ad)

    if exit_code == 0:
        emit_event(
            "job.completed", run_id, log_dir=log_dir, job_name=job_name,
            source="dagman_post_script_classad", **resource,
        )
    else:
        extra: dict = {"exit_code": exit_code, **resource}
        hold_reason = ad.get("HoldReason", "")
        if hold_reason:
            extra["hold_reason"] = hold_reason
        emit_event(
            "job.failed", run_id, log_dir=log_dir, job_name=job_name,
            source="dagman_post_script_classad", **extra,
        )


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <job_name> <exit_code> <cluster_id>",
            file=sys.stderr,
        )
        sys.exit(1)
    job_name, exit_code_str, cluster_id = sys.argv[1], sys.argv[2], sys.argv[3]
    # $JOBID expands to ClusterId.ProcId (e.g. "5555662.0"); job_ad_file uses
    # only ClusterId, so strip the proc part to match the filename.
    cluster_id = cluster_id.split(".")[0]
    log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    emit_post_event(job_name, int(exit_code_str), cluster_id, log_dir=log_dir)


if __name__ == "__main__":
    main()
