"""POST-script logic for DAGMan job lifecycle provenance.

Called by DAGMan after each training job exits. Extracts the run ID from a
ClassAd (see parse_classad/run_id_from_classad) and appends a job.completed
or job.failed event to the per-run NDJSON log.

The ClassAd-reading helpers here (parse_classad, resource_fields_from_classad,
load_classad_field_mapping) are also reused by mldag.provenance.jobad, which
reads $_CONDOR_JOB_AD from *within* a running job -- job_ad_file, a submit
command that would have let this module read a ClassAd file written on job
exit, is not real (condor_submit silently ignores it); see task-25.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

from mldag.constants import DEFAULT_CLASSAD_FIELDS_FILE
from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event

# ClassAd attributes that must never be captured into provenance, even if an
# experiment's provenance_fields.yaml requests them — e.g. Environment carries
# secrets such as WANDB_API_KEY (see daggen.py's env_vars construction).
_SENSITIVE_AD_KEYS = {"Environment"}

# Used when no field-mapping file is configured or the configured file
# doesn't exist, so existing deployments keep working with zero config.
_DEFAULT_FIELD_MAPPING = {
    "RemoteWallClockTime": "wall_time_s",
    "CPUsUsage": "cpu_usage",
    "MemoryUsage": "peak_memory_mb",
    "GPUsUsage": "gpu_usage",
    "GLIDEIN_ResourceName": "resource_name",
    "Arguments": "arguments",
    "RequestCpus": "request_cpus",
    "RequestMemory": "request_memory",
    "RequestGpus": "request_gpus",
}


def _snake_case(name: str) -> str:
    """Derive a schema key from a bare ClassAd attribute name, e.g. "RequestCpus" -> "request_cpus"."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return s.lower()


def load_classad_field_mapping(path: Path | str | None) -> dict[str, str]:
    """Load the ad_key -> schema_key mapping used to extract fields from a job ClassAd.

    Falls back to _DEFAULT_FIELD_MAPPING when path is None or the file doesn't
    exist, so an experiment repo only needs provenance_fields.yaml once it
    wants to deviate from the defaults. A bare list entry (as opposed to a
    "AdKey: schema_key" mapping entry) has its schema key derived via
    _snake_case.

    Raises:
        ValueError: if a requested field is in _SENSITIVE_AD_KEYS.
    """
    if path is None:
        return dict(_DEFAULT_FIELD_MAPPING)
    path = Path(path)
    if not path.exists():
        return dict(_DEFAULT_FIELD_MAPPING)

    data = yaml.safe_load(path.read_text()) or {}
    raw_fields = data.get("fields", [])
    mapping: dict[str, str] = {}
    if isinstance(raw_fields, dict):
        entries = raw_fields.items()
    else:
        entries = ((ad_key, None) for ad_key in raw_fields)
    for ad_key, schema_key in entries:
        if ad_key in _SENSITIVE_AD_KEYS:
            raise ValueError(
                f"Refusing to capture ClassAd attribute {ad_key!r} into provenance: "
                f"it is on the sensitive-key blocklist ({sorted(_SENSITIVE_AD_KEYS)}) "
                f"and may carry secrets. Remove it from {path}."
            )
        mapping[ad_key] = schema_key or _snake_case(ad_key)
    return mapping


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


def resource_fields_from_classad(ad: dict, mapping: dict[str, str] | None = None) -> dict:
    """Return the subset of ClassAd fields that map to the provenance schema.

    Args:
        mapping: ad_key -> schema_key, as returned by load_classad_field_mapping().
            Defaults to _DEFAULT_FIELD_MAPPING when not supplied.
    """
    if mapping is None:
        mapping = _DEFAULT_FIELD_MAPPING
    return {schema_key: ad[ad_key] for ad_key, schema_key in mapping.items() if ad_key in ad}


def emit_post_event(
    job_name: str,
    exit_code: int,
    cluster_id: str,
    *,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
    run_id_hint: str | None = None,
    fields_file: str | Path | None = None,
) -> None:
    """Emit job.completed or job.failed using data from the HTCondor ClassAd.

    Args:
        run_id_hint: run_id to use when ClassAd and .run_id marker both fail.
            Pass the run_uuid from DAG generation so the event is always
            attributed even when job_ad_file is not supported by the schedd.
        fields_file: path to a provenance_fields.yaml mapping; see
            load_classad_field_mapping(). Defaults to the built-in mapping
            when None or the file doesn't exist.
    """
    log_dir = Path(log_dir)
    ad = parse_classad(log_dir / f"{cluster_id}.ad")
    run_id = run_id_from_classad(ad)
    if run_id == "unknown":
        marker = log_dir / f"{cluster_id}.run_id"
        if marker.exists():
            run_id = marker.read_text().strip()
    if run_id == "unknown" and run_id_hint:
        run_id = run_id_hint
    mapping = load_classad_field_mapping(fields_file)
    resource = resource_fields_from_classad(ad, mapping)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name")
    parser.add_argument("exit_code", type=int)
    parser.add_argument("cluster_id")
    parser.add_argument(
        "--run-id", default=None, dest="run_id",
        help="run_uuid embedded at DAG generation time; used as fallback when "
             "the ClassAd file is unavailable (e.g. job_ad_file not supported).",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="NDJSON event log directory; overrides PROVENANCE_LOG_DIR. "
             "DAG generation should always pass this explicitly so it can "
             "never drift from what log_monitor is configured to search. "
             "Must appear before --post-hook (which consumes the remainder).",
    )
    parser.add_argument(
        "--fields-file", default=None, dest="fields_file",
        help=f"YAML file mapping ClassAd attributes to provenance schema keys "
             f"(see load_classad_field_mapping). Falls back to the built-in "
             f"default mapping when unset or the file doesn't exist. DAG "
             f"generation bakes this in explicitly (default filename: "
             f"{DEFAULT_CLASSAD_FIELDS_FILE}) so it can't drift.",
    )
    parser.add_argument(
        "--post-hook", nargs=argparse.REMAINDER, default=[],
        metavar="CMD [ARGS...]",
        help="Optional command + args to run after provenance is recorded. "
             "Everything after --post-hook is forwarded as-is; DAGMan expands "
             "$MACRO tokens before this script is called. POST-script macros: "
             "$NODE $RETURN $JOBID $CLUSTERID $RETRY $MAX_RETRIES $SUCCESS "
             "$PRE_SCRIPT_RETURN $EXIT_CODES $DAGID $DAG_STATUS.",
    )
    args = parser.parse_args()
    # $JOBID expands to ClusterId.ProcId (e.g. "5555662.0"); job_ad_file uses
    # only ClusterId, so strip the proc part to match the filename.
    cluster_id = args.cluster_id.split(".")[0]
    log_dir = args.log_dir or os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    emit_post_event(
        args.job_name, args.exit_code, cluster_id,
        log_dir=log_dir, run_id_hint=args.run_id, fields_file=args.fields_file,
    )
    if args.post_hook:
        import subprocess
        result = subprocess.run(args.post_hook)
        if result.returncode != 0:
            print(
                f"WARNING: post-hook {args.post_hook[0]!r} exited {result.returncode}",
                file=sys.stderr,
            )
    sys.exit(args.exit_code)


if __name__ == "__main__":
    main()
