#!/bin/bash
# dynamic get project root, because script location is fixed, but project root is not fixed
script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
DATABASE_ROOT="${PROJECT_ROOT}/PEKA/DATA/breast/"

DATABASE="peka_breast_datasets.csv"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DATABASE_ROOT: $DATABASE_ROOT"

python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/2_peka_dataset_generator.py \
--project_root $PROJECT_ROOT \
--database_root $DATABASE_ROOT \
--datasets_predefine $DATABASE