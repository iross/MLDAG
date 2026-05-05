#!/bin/bash
set -euo pipefail

# Called by DAGMan: SCRIPT POST <node> provenance_post.sh $JOB $RETURN $JOBID
JOB_NAME="${1:?job name required}"
EXIT_CODE="${2:?exit code required}"
JOBID="${3:?HTCondor job ID (cluster.proc) required}"

# Extract just the cluster ID — proc is always 0 for single-job submissions
CLUSTER_ID="${JOBID%%.*}"

python3 -m mldag.provenance.post "$JOB_NAME" "$EXIT_CODE" "$CLUSTER_ID"
