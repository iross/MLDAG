#!/bin/bash
if [ $# -eq 0 ]
then
    echo "No arguments supplied. Exiting."
    exit 1
else
    epochs=$1
    run_uuid=$2
    random_seed=$3
fi
nvidia-smi

#echo "Copying global dataset"
# cp /staging/iaross/processed-global.tar.gz .
echo "Untarring global dataset"
tar xzvf processed-global.tar.gz

#unzip cleaned_data_test.zip -d precleaned
rm processed-global.tar.gz

ln -s /workspace/metl/data/

pwd
env

mkdir wandb
mkdir wandb_data
export WANDB_DIR=$PWD/wandb
export WANDB_DATA_DIR=$PWD/wandb_data
export WANDB_CACHE_DIR=$PWD/wandb/.cache
export WANDB_CONFIG_DIR=$PWD/wandb/.config
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_avgfp_local.txt \
    --ds_fn $PWD/global/global.db   \
    --split_dir $PWD/global/splits/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098 \
    --max_epochs $epochs --uuid=$run_uuid  \
    --random_seed $random_seed



    #--use_wandb \
    #--wandb_online --wandb_project metl_global_$run_uuid\_$epochs \

#python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_avgfp_local.txt \
    #--ds_fn $PWD/global/global.db   \
    #--split_dir $PWD/global/splits/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098 \
    #--max_epochs $epochs --uuid=$run_uuid  \
    #--condor_checkpoint_every_n_epochs 1 \
    ##--use_wandb \
    ##--wandb_online --wandb_project metl_global_$run_uuid \
    #--random_seed $random_seed
