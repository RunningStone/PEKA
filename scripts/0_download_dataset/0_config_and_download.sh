#!/bin/bash
# dynamic get project root, because script location is fixed, but project root is not fixed
script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
echo "PROJECT_ROOT: $PROJECT_ROOT"

CODE_ROOT="${PROJECT_ROOT}/PEKA/"
SRC_ROOT="${PROJECT_ROOT}/PEKA/peka/"
EXTERNAL_MODELS_ROOT="${PROJECT_ROOT}/PEKA/peka/External_models/" 

FORCE_DOWNLOAD=True

# config project paths
python ${SRC_ROOT}/Exp_helper/0_config_runable.py

# download HEST1k database, if exists then not download or force download to update dataset
python ${SRC_ROOT}/Exp_helper/1_dataset_downloader.py #--force_download ${FORCE_DOWNLOAD}
