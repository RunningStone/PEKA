#!/bin/bash

# dynamic get project root, because script location is fixed, but project root is not fixed
export PROJECT_ROOT=$(dirname $(dirname $(dirname $(pwd))))
echo "PROJECT_ROOT: $PROJECT_ROOT"

TISSUE_NAME="breast"
#"breast_visium_26k" , "breast_xenium_100k", 
DATASET_NAME="breast_visium_26k"
#"scMulan" "scFoundation"
SCLLM_EMBEDDER_NAME="scFoundation"
#"ckpt_scMulan" "default_model"
CKPT="default_model"
N_CLUSTERS=100

# 为breast_visium_26k数据集生成聚类标签
echo "Processing $DATASET_NAME dataset with $SCLLM_EMBEDDER_NAME..."
python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/7_generate_cluster_labels_for_KD.py \
    --project_root "$PROJECT_ROOT" \
    --tissue_name "$TISSUE_NAME" \
    --dataset_name "$DATASET_NAME" \
    --scLLM_embedder_name "$SCLLM_EMBEDDER_NAME" \
    --ckpt "$CKPT" \
    --n_clusters "$N_CLUSTERS" \
    --use_gpu

