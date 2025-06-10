#!/bin/bash
if [ $# -eq 0 ]
then
    echo "No arguments supplied. Exiting."
    exit 1
else
    epochs=$1
    run_uuid=$2
fi

#echo "Copying global dataset"
# cp /staging/iaross/processed-global.tar.gz .
echo "Untarring global dataset"
tar xzvf processed-global.tar.gz

#unzip cleaned_data_test.zip -d precleaned
rm processed-global.tar.gz

ln -s /workspace/metl/data/

python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_avgfp_local.txt \
    --ds_fn $PWD/global/global.db   \
    --split_dir $PWD/global/splits/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098 \
    --max_epochs $epochs --uuid=$run_uuid 