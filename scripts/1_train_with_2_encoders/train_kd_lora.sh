#!/bin/bash

# Dataset configuration components
TISSUE_TYPE="breast"
DATASET_NAME="${TISSUE_TYPE}_visium_26k"
SCLLM="scFoundation"
GEN_LABEL="clustered100"

# Construct dataset config path
DATASET_CONFIG="Datasets/${DATASET_NAME}_${SCLLM}_with_${GEN_LABEL}_label.yaml"

# Other default values
PROJECT_ROOT="PATH_TO_PROJECT_ROOT"

MODEL_CONFIG="Models/H-optimus-0_LoRA_MLP.yaml"
OPTIMIZER_CONFIG="Optimizers/default.yaml"
TRAINER_CONFIG="Trainers/default.yaml"
PHASE1_CKPT="PATH_TO_PHASE1_CKPT"
#PHASE1_CKPT=""
PHASE1_EPOCHS=20
PHASE1_LR=1e-4
PHASE1_HIDDEN_DIM=512

# Experiment name
EXP_NAME="KD_LoRA_${TISSUE_TYPE}_${DATASET_NAME}_${SCLLM}_${GEN_LABEL}"

# Run the training script with the provided arguments
python kd_lora_train.py \
  --dataset-config "$DATASET_CONFIG" \
  --model-config "$MODEL_CONFIG" \
  --optimizer-config "$OPTIMIZER_CONFIG" \
  --trainer-config "$TRAINER_CONFIG" \
  --phase1-ckpt "$PHASE1_CKPT" \
  --phase1-epochs "$PHASE1_EPOCHS" \
  --phase1-lr "$PHASE1_LR" \
  --phase1-hidden-dim "$PHASE1_HIDDEN_DIM" \
  --exp-name "$EXP_NAME"
