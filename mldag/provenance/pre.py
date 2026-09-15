"""PRE-script logic: emit job.submitted just before DAGMan submits the job.

Invoked directly by DAGMan as:
    /path/to/python -m mldag.provenance.pre <run_uuid> <job_name> <epoch>

The Python path is embedded at DAG generation time by daggen.py (sys.executable),
so the script works regardless of PATH in the DAGMan environment.
"""

from __future__ import annotations

import os
import sys

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_uuid")
    parser.add_argument("job_name")
    parser.add_argument("epoch", type=int)
    parser.add_argument(
        "--log-dir", default=None,
        help="NDJSON event log directory; overrides PROVENANCE_LOG_DIR. "
             "DAG generation should always pass this explicitly so it can "
             "never drift from what log_monitor is configured to search.",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    emit_event(
        "job.submitted",
        args.run_uuid,
        log_dir=log_dir,
        job_name=args.job_name,
        epoch=args.epoch,
        source="dagman_pre_script",
    )


if __name__ == "__main__":
    main()

