#!/bin/bash
if [ $# -eq 0 ]
then
    echo "No arguments supplied. Exiting."
    exit 1
else
    epochs=$1
    run_uuid=$2
fi

#cp /staging/iaross/cleaned_data_test.zip .

unzip cleaned_data_test.zip -d precleaned
rm cleaned_data_test.zip

ln -s /workspace/metl/data/

python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_avgfp_local.txt \
    --ds_fn $PWD/precleaned/data/rosetta_data/avgfp/avgfp.db   \
    --split_dir $PWD/precleaned/data/rosetta_data/avgfp/splits/standard_tr0.8_tu0.1_te0.1_w5fb1ed670b87_r4991 \
    --max_epochs $epochs --uuid=$run_uuid 