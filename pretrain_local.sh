#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ]
then
    echo "No arguments supplied. Exiting."
    exit 1
else
    epochs=$1
    run_uuid=$2
    random_seed=$3
    dataset_name="${4:-gb1}"
fi

export PROVENANCE_RUN_ID="$run_uuid"

_provenance_capture_and_emit() {
    python3 - <<'PYEOF'
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""

gpu_count = torch.cuda.device_count()
gpu_model = torch.cuda.get_device_name(0) if gpu_count > 0 else "none"
cuda = torch.version.cuda or "unknown"
hostname = run(["hostname", "-f"]) or run(["hostname"]) or "unknown"
python_ver = run([sys.executable, "--version"]).replace("Python ", "")
commit = run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
slot = os.environ.get("_CONDOR_SLOT", "unknown")
run_id = os.environ.get("PROVENANCE_RUN_ID", "unknown")
log_dir = Path(os.environ.get("PROVENANCE_LOG_DIR", "output/provenance"))

site_info = {
    "hostname": hostname,
    "slot": slot,
    "gpu_model": gpu_model,
    "gpu_count": gpu_count,
}
env_info = {
    "python": python_ver,
    "cuda": cuda,
    "code_commit": commit,
}

Path("site_info.json").write_text(json.dumps({**site_info, **env_info}, indent=2))

log_dir.mkdir(parents=True, exist_ok=True)
event = {
    "schema_version": "1.0",
    "type": "job.assigned",
    "run_id": run_id,
    "ts": datetime.now(timezone.utc).isoformat(),
    "source": "execute_node",
    "site_info_source": "torch_cuda",
    **site_info,
    **env_info,
}
log_path = log_dir / f"{run_id}.ndjson"
with open(log_path, "a") as f:
    f.write(json.dumps(event, separators=(",", ":")) + "\n")

cluster_id = os.environ.get("CONDOR_CLUSTERID", "")
if cluster_id:
    (log_dir / f"{cluster_id}.run_id").write_text(run_id)
PYEOF
}

_provenance_capture_and_emit || exit 1

# Launch checkpoint watcher in background; trap ensures clean shutdown on exit
python3 -m mldag.provenance.watcher "$PWD" "$run_uuid" &
WATCHER_PID=$!
trap 'kill "$WATCHER_PID" 2>/dev/null; wait "$WATCHER_PID" 2>/dev/null || true' EXIT

#echo "Copying ${dataset_name} dataset"
# cp "/staging/iaross/processed-${dataset_name}.tar.gz" .
#echo "Untarring ${dataset_name} dataset"
#mkdir -p ${dataset_name}
#tar -xvzf processed-${dataset_name}.tar.gz -C "${dataset_name}" --strip-components=1

#unzip cleaned_data_test.zip -d precleaned
# rm "processed-${dataset_name}.tar.gz"

ln -s /workspace/metl/data/

parent_dir=$(realpath "${PWD}/${dataset_name}"/splits/*/)
splits_dir=$(basename "${parent_dir}")

pwd
env

mkdir wandb
mkdir wandb_data
export WANDB_DIR=$PWD/wandb
export WANDB_DATA_DIR=$PWD/wandb_data
export WANDB_CACHE_DIR=$PWD/wandb/.cache
export WANDB_CONFIG_DIR=$PWD/wandb/.config
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_local.txt \
    --ds_fn "$PWD/${dataset_name}/${dataset_name}.db"   \
    --split_dir "$PWD/${dataset_name}/splits/${splits_dir}" \
    --max_epochs $epochs --uuid=$run_uuid  \
    --random_seed $random_seed
