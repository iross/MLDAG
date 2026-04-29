#!/bin/bash
set -euo pipefail

# Args baked in by daggen.py at DAG generation time (VARS macros are not
# available in SCRIPT PRE args, so values are embedded directly).
RUN_UUID="${1:?run_uuid required}"
JOB_NAME="${2:?job name required}"
EPOCH="${3:?epoch required}"
ANNEX_NAME="${4:-}"  # non-empty only for ANNEX resources

python3 -m mldag.provenance.pre "$RUN_UUID" "$JOB_NAME" "$EPOCH"

# For ANNEX resources, also run the annex request script (DAGMan allows only
# one SCRIPT PRE per node, so we chain it here).
if [ -n "$ANNEX_NAME" ]; then
    pre_request_annex.sh "$ANNEX_NAME" "${ANNEX_NAME}_annex"
fi
