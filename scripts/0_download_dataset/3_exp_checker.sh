#!/bin/bash

script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
echo "PROJECT_ROOT: $PROJECT_ROOT"

conda init bash
conda activate hest 

python ${PROJECT_ROOT}/HistoMIL2/src/Exp_helper/3_model_required_rec_check.py \
    --model_name "hf-hub:bioptimus/H-optimus-0" \
    --input_size 3 224 224 \
    --lora_r 32 \
    --lora_alpha 16 \
    --target_modules qkv proj \
    --cuda_ram_test \
    --run_lora_test