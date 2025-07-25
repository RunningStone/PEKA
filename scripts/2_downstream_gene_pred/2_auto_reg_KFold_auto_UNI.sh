#!/bin/bash
# 动态获得项目路径：因为脚本位置固定，但项目路径不固定
script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")

# Default values
TISSUE_TYPE="breast"
DATASET_NAME="breast_visium_26k"
EMBEDDER_NAME="scFoundation"
GENE_LIST_JSON="/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/HistoMIL2/scripts/2_downstream_gene_pred/top_50_genes_Visium_Homo_sapien_Breast_Cancer.json"
BASE_OUTPUT_ROOT="/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/OUTPUT"
WITH_INDEPENDENT_TEST_SET=false
EPOCHS=300
IMAGE_ENCODER_NAME="UNI"

# Arrays for iteration
BINNED_OPTIONS=(false true)
FEATURE_TYPES=("image_encoder" "histomil2" "scLLM" "image_encoder+histomil2")

# Iterate over all combinations
for USE_BINNED in "${BINNED_OPTIONS[@]}"; do
    for FEATURE_TYPE in "${FEATURE_TYPES[@]}"; do
        echo "=== Running with USE_BINNED=${USE_BINNED} and FEATURE_TYPE=${FEATURE_TYPE} ==="
        
        # Create specific output directory for this combination
        binned_str=$([ "$USE_BINNED" = true ] && echo "binned" || echo "raw")
        OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/${DATASET_NAME}/${binned_str}/"
        
        echo "PROJECT_ROOT: $PROJECT_ROOT"
        echo "TISSUE_TYPE: $TISSUE_TYPE"
        echo "DATASET_NAME: $DATASET_NAME"
        echo "EMBEDDER_NAME: $EMBEDDER_NAME"
        echo "GENE_LIST_JSON: $GENE_LIST_JSON"
        echo "USE_BINNED: $USE_BINNED"
        echo "OUTPUT_ROOT: $OUTPUT_ROOT"
        echo "FEATURE_TYPE: $FEATURE_TYPE"
        echo "WITH_INDEPENDENT_TEST_SET: $WITH_INDEPENDENT_TEST_SET"
        echo "EPOCHS: $EPOCHS"

        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_ROOT"

        # Run the Python script with parameters
        cmd="python ${PROJECT_ROOT}/HistoMIL2/scripts/2_downstream_gene_pred/task_gene_expr_reg_KFold.py \
        --project_root $PROJECT_ROOT \
        --tissue_type $TISSUE_TYPE \
        --dataset_name $DATASET_NAME \
        --embedder_name $EMBEDDER_NAME \
        --gene_list_json \"$GENE_LIST_JSON\" \
        --output_root $OUTPUT_ROOT \
        --feature_type $FEATURE_TYPE \
        --epochs $EPOCHS \
        --image_encoder_name $IMAGE_ENCODER_NAME \
        --mask_zero_values"

        # Add optional flags based on conditions
        if [ "$USE_BINNED" = true ]; then
            cmd="$cmd --use_binned"
        fi

        if [ "$WITH_INDEPENDENT_TEST_SET" = true ]; then
            cmd="$cmd --with_independent_test_set"
        fi

        # Execute the command
        echo "Executing: $cmd"
        eval $cmd
        
        echo "=== Finished combination USE_BINNED=${USE_BINNED} and FEATURE_TYPE=${FEATURE_TYPE} ==="
        echo
    done
done