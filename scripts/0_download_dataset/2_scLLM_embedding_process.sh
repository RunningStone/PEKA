#!/bin/bash

# dynamic get project root, because script location is fixed, but project root is not fixed
export PROJECT_ROOT=$(dirname $(dirname $(dirname $(pwd))))

# set dataset name
export DATASET_NAME="breast_visium_26k"  # set dataset name
export TISSUE_TYPE="breast"
# set scLLM embedder name
export SCLLM_EMBEDDER_NAME="scFoundation"

# optional parameters - if not set, will use default values in config file
export CKPT_NAME=""
export MODEL_MODE=""
export TOP_K=""

# run python script
python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/3_scLLM_embedding_process.py
