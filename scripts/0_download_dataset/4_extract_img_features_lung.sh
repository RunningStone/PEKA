#!/bin/bash

script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
echo "PROJECT_ROOT: $PROJECT_ROOT"

DATASET_ROOT=$PROJECT_ROOT/PEKA/DATA/
echo "DATASET_ROOT: $DATASET_ROOT"

subdataset_folder="${PROJECT_ROOT}/PEKA/DATA/other_cancer/lung_in_hest/" # need to prepare before
echo "subdataset_folder: $subdataset_folder"

# H-optimus-0
python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/5_patch_feature_embeder.py \
    --model_name "hf-hub:bioptimus/H-optimus-0" \
    --subdataset_folder $subdataset_folder \
    --num_features 1536 \
    --patch_size 224

# UNI
python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/5_patch_feature_embeder.py \
    --model_name "hf-hub:MahmoodLab/UNI" \
    --subdataset_folder $subdataset_folder \
    --num_features 1024 \
    --patch_size 224