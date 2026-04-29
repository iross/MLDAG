"""PRE-script logic: emit job.submitted just before DAGMan submits the job.

Called from provenance_pre.sh, which DAGMan runs on the AP immediately
before submitting the HTCondor job for each DAG node.
"""

import os
import sys

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <run_uuid> <job_name> <epoch>",
            file=sys.stderr,
        )
        sys.exit(1)
    run_uuid, job_name, epoch_str = sys.argv[1], sys.argv[2], sys.argv[3]
    log_dir = os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR)
    emit_event(
        "job.submitted",
        run_uuid,
        log_dir=log_dir,
        job_name=job_name,
        epoch=int(epoch_str),
    )


if __name__ == "__main__":
    main()
