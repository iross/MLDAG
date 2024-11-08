#!/bin/bash

if [ $# -eq 0 ]
then
    echo "No arguments supplied. Exiting."
    exit 1
else
    epochs=$1
    run_uuid=$2
fi

hostname
echo "Running with max_epochs: $epochs"

ln -s /workspace/metl/data/
ln -s /workspace/metl/pretrained_models/


python /workspace/metl/code/train_target_model.py @/workspace/metl/args/finetune_avgfp_local.txt --enable_progress_bar false --enable_simple_progress_messages --max_epochs $epochs --uuid=$run_uuid --unfreeze_backbone_at_epoch 25

