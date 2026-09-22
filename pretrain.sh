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
fi

#echo "Copying global dataset"
# cp /staging/iaross/processed-global.tar.gz .
_provenance_capture_and_emit() {
    # Best-effort: mldag.provenance.autocapture.capture_and_emit() never
    # raises internally, but this wraps the whole thing anyway (torch
    # itself might be missing) so a broken capture step can never abort the
    # training job -- unlike the old inline heredoc this replaced, which ran
    # under `|| exit 1` and would fail the whole job over e.g. a torch
    # import error. GPU info is detected here via torch.cuda (more precise
    # than autocapture's own nvidia-smi/proc fallback, which is used when no
    # override is passed) and handed to capture_and_emit as an override.
    python3 - <<'PYEOF'
try:
    from mldag.provenance.autocapture import capture_and_emit

    gpu_info = None
    try:
        import torch

        gpu_count = torch.cuda.device_count()
        gpu_info = {
            "gpu_count": gpu_count,
            "gpu_model": torch.cuda.get_device_name(0) if gpu_count > 0 else "none",
        }
        try:
            if gpu_count > 0:
                gpu_info["gpu_id"] = str(torch.cuda.get_device_properties(0).uuid)
        except AttributeError:
            from mldag.provenance.autocapture import default_gpu_info

            gpu_info["gpu_id"] = default_gpu_info()[0]["gpu_id"]
    except ImportError:
        pass

    capture_and_emit(gpu_info=gpu_info)
except Exception:
    pass
PYEOF
}

_provenance_capture_and_emit

echo "Untarring global dataset"
tar xzvf processed-global.tar.gz

#unzip cleaned_data_test.zip -d precleaned
rm processed-global.tar.gz

ln -s /workspace/metl/data/

pwd
env

#mkdir wandb
#mkdir wandb_data
#export WANDB_DIR=$PWD/wandb
#xport WANDB_DATA_DIR=$PWD/wandb_data
#export WANDB_CACHE_DIR=$PWD/wandb/.cache
#export WANDB_CONFIG_DIR=$PWD/wandb/.config
#export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

_TRAIN_START=$(python3 -c "import time; print(time.time())")

python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_global.txt \
    --ds_fn $PWD/global/global.db   \
    --split_dir $PWD/global/splits/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098 \
    --max_epochs $epochs --uuid=$run_uuid  \
    --random_seed $random_seed

python3 -m mldag.provenance.watcher "$PWD" "${PROVENANCE_RUN_ID:-$run_uuid}" \
    --one-shot --start-time "$_TRAIN_START"

