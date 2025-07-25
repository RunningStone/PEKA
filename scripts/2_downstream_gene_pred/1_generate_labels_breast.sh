#!/bin/bash
# dynamic get project root, because script location is fixed, but project root is not fixed
script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
DATABASE_ROOT="${PROJECT_ROOT}/PEKA/DATA/breast/"

DATABASE="peka_breast_datasets.csv"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DATABASE_ROOT: $DATABASE_ROOT"

# config file path
CONFIG_FILE="${script_dir}/config_gene_bin_breast.txt"

# default value
BIN_NB=51
N_TOP_HVG=50
HVG_FLAVOR="seurat"
BATCH_KEY=""
HVG_JSON=""

# if config file exists, read parameters from config file
if [ -f "$CONFIG_FILE" ]; then
    echo "Reading configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
fi

# if HVG JSON file is provided, use specified HVG JSON file
if [ -n "$HVG_JSON" ]; then
    # use specified HVG JSON file
    python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/6_gene_bin_generator.py \
    --project_root $PROJECT_ROOT \
    --database_root $DATABASE_ROOT \
    --bin_nb $BIN_NB \
    --hvg_json "$HVG_JSON"
else
    # use HVG calculation mode
    python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/6_gene_bin_generator.py \
    --project_root $PROJECT_ROOT \
    --database_root $DATABASE_ROOT \
    --bin_nb $BIN_NB \
    --n_top_hvg $N_TOP_HVG \
    --hvg_flavor $HVG_FLAVOR \
    ${BATCH_KEY:+"--batch_key"} $BATCH_KEY
fi
