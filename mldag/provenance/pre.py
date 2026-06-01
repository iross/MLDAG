"""PRE-script logic: emit job.submitted just before DAGMan submits the job.

Invoked directly by DAGMan as:
    /path/to/python -m mldag.provenance.pre <run_uuid> <job_name> <epoch> [--annex <name>]

The Python path is embedded at DAG generation time by daggen.py (sys.executable),
so the script works regardless of PATH in the DAGMan environment.

For ANNEX resources, --annex <name> causes pre_request_annex.sh to be called
after the provenance event is emitted (DAGMan allows only one SCRIPT PRE per node).
"""

from __future__ import annotations

import os
import subprocess
import sys

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_uuid")
    parser.add_argument("job_name")
    parser.add_argument("epoch", type=int)
    parser.add_argument("--annex", default="", metavar="NAME",
                        help="Annex resource name; triggers pre_request_annex.sh")
    args = parser.parse_args()

    log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    emit_event(
        "job.submitted",
        args.run_uuid,
        log_dir=log_dir,
        job_name=args.job_name,
        epoch=args.epoch,
        source="dagman_pre_script",
    )

    if args.annex:
        subprocess.run(
            ["pre_request_annex.sh", args.annex, f"{args.annex}_annex"],
            check=True,
        )


if __name__ == "__main__":
    main()

