# PEKA: Parameter Efficient Knowledge Transfer for Accurate Gene Expression Prediction

PEKA is a novel framework that teaches pathology foundation models to accurately predict gene expression using parameter-efficient knowledge transfer techniques. The project combines histopathology image analysis with single-cell genomics through knowledge distillation and Parameter Efficient Fine-tuning (PEFT) to achieve efficient and accurate gene expression prediction from tissue images.

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Step-by-Step Experimental Guide](#step-by-step-experimental-guide)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

PEKA addresses the challenge of predicting gene expression patterns from histopathology images by:

1. **Knowledge Distillation**: Transferring knowledge from pre-trained single-cell foundation models to histopathology models
2. **Parameter Efficiency**: Using LoRA (Low-Rank Adaptation),Adaptive LoRA(AdaLoRA),Bone to minimize computational overhead while maintaining performance
3. **Multi-modal Integration**: Combining histopathology images with spatial transcriptomics data
4. **Scalable Pipeline**: Providing an end-to-end workflow from data preprocessing to model evaluation

### Key Features
- Integration with HEST1K dataset for large-scale histopathology analysis
- Support for multiple cancer types (breast, kidney, liver, lung)
- Knowledge distillation framework with dual encoders
- Parameter-efficient fine-tuning using LoRA,Adaptive LoRA(AdaLoRA),Bone
- Comprehensive evaluation pipeline with K-fold cross-validation
- Integration with Weights & Biases for experiment tracking

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git with LFS support
- Sufficient storage space (>100GB for full datasets)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone --recursive <repository-url>
   cd PEKA
   ```

2. **Set Up Python Environment**
   ```bash
    # follow last step 
    cd /REPO_LOCATION/PEKA/peka/External_models/HEST

    conda create -n "hest" python=3.9
    conda activate hest
    pip install -e .

    pip install     --extra-index-url=https://pypi.nvidia.com     cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.*     raft-dask-cu12==24.6.*
    pip install -U 'wandb>=0.12.10'
    pip install hydra_zen,peft

    # Optional: Install FAISS-GPU for GPU-accelerated clustering
    pip install faiss-gpu
   ```

3. **Configure Environment Variables**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your credentials
   nano .env
   ```
   
   Required environment variables:
   ```
   WANDB_API_KEY=your_wandb_api_key
   HF_TOKEN=your_huggingface_token
   WANDB_ENTITY=your_wandb_entity
   HEST1K_STORAGE_PATH=/path/to/hest1k/storage
   ```

4. **Initialize Project Configuration**
   ```bash
   cd /REPO_LOCATION/PEKA/scripts/0_download_dataset
   bash 0_config_and_download.sh
   ```

### Directory Structure Setup

The initialization script will create the following directory structure:
```
PEKA/
├── DATA/                 # Dataset storage
├── OUTPUT/              # Experiment outputs
├── Pretrained/          # Pre-trained model checkpoints
└── PEKA/               # Source code
```

## Project Structure

```
PEKA/
├── .env.example                    # Environment variables template
├── .gitignore                     # Git ignore rules
├── .gitmodules                    # Git submodules configuration
├── README.md                      # This file
│
├── DATA/                          # Dataset directory (created during setup)
│   └── HEST1K/                   # HEST1K dataset storage
│
├── hydra_zen/                     # Hydra configuration files
│   └── Configs/                  # Model and experiment configurations
│
├── peka/                         # Main source code directory
│   ├── __init__.py              # Package initialization
│   ├── Data/                    # Data processing modules
│   │   ├── dataset_factory.py   # Dataset creation utilities
│   │   ├── download_helper.py   # Dataset download utilities
│   │   └── preprocessing.py     # Data preprocessing functions
│   │
│   ├── Model/                   # Model architectures
│   │   ├── LLM/                # Language model components
│   │   ├── base.py             # Base model classes
│   │   └── utils.py            # Model utilities
│   │
│   ├── Trainer/                 # Training frameworks
│   │   ├── KD_LoRA.py          # Knowledge distillation with LoRA,Adaptive LoRA(AdaLoRA),Bone,etc.
│   │   └── base_trainer.py     # Base training classes
│   │
│   ├── External_models/         # External model integrations
│   │   ├── HEST/               # HEST model integration
│   │   └── scFoundation/       # scFoundation model integration
│   │
│   ├── Exp_helper/             # Experiment utilities
│   │   ├── 0_config_runable.py # Environment configuration
│   │   ├── 1_dataset_downloader.py # Dataset download manager
│   │   └── experiment_helpers.py # Experiment management utilities
│   │
│   ├── Hydra_helper/           # Hydra configuration helpers
│   │   ├── experiment_helpers.py # Experiment configuration
│   │   └── pl_model_helpers.py  # PyTorch Lightning model helpers
│   │
│   └── DownstreamTasks_helper/ # Downstream task utilities
│       ├── gene_prediction.py  # Gene expression prediction
│       └── evaluation.py       # Model evaluation metrics
│
├── scripts/                      # Experimental pipeline scripts
│   ├── 0_download_dataset/      # Phase 0: Data preparation
│   │   ├── 0_config_and_download.sh        # Initial setup
│   │   ├── 1_generate_peka_datasets_*.sh # Dataset generation
│   │   ├── 2_scLLM_embedding_process.sh     # Embedding processing
│   │   ├── 4_extract_img_features_*.sh      # Image feature extraction
│   │   └── 5_generate_cluster_label_for_KD.sh # Clustering for KD
│   │
│   ├── 1_train_with_2_encoders/ # Phase 1: Model training
│   │   ├── kd_lora_train.py     # Main training script
│   │   ├── kd_lora_inference.py # Inference script
│   │   ├── simple_train.py      # Simplified training
│   │   └── train_kd_lora.sh     # Training shell script
│   │
│   └── 2_downstream_gene_pred/  # Phase 2: Gene prediction
│       ├── step1_process_hvg.py # Highly variable genes processing
│       ├── step2_inference_feature_vectors.py # Feature extraction
│       ├── step3_task_gene_expr_reg_KFold.py  # K-fold regression
│       ├── 1_generate_labels_*.sh # Label generation
│       ├── 2_auto_reg_KFold_*.sh  # Automated regression
│       └── top_50_genes_*.json    # Gene lists for different cancer types
│
└── support_files/               # Additional support files
    ├── configs/                 # Additional configuration files
    └── documentation/           # Additional documentation
```

### Key Components Description

#### Core Modules (`peka/`)
- **Data**: Handles dataset downloading, preprocessing, and loading
- **Model**: Contains model architectures and utilities
- **Trainer**: Implements training frameworks including PEKA
- **External_models**: Integration with external models (HEST, scFoundation)
- **Exp_helper**: Experiment management and configuration utilities
- **Hydra_helper**: Configuration management using Hydra-zen
- **DownstreamTasks_helper**: Utilities for downstream gene prediction tasks

#### Experimental Scripts (`scripts/`)
- **Phase 0** (`0_download_dataset/`): Data preparation and preprocessing
- **Phase 1** (`1_train_with_2_encoders/`): Model training with knowledge distillation
- **Phase 2** (`2_downstream_gene_pred/`): Gene expression prediction and evaluation

## Step-by-Step Experimental Guide

### Phase 0: Data Preparation

#### Step 0.1: Initial Configuration and Dataset Download
```bash
cd scripts/0_download_dataset
bash 0_config_and_download.sh
```

This script will:
- Configure project paths and Python imports
- Download the HEST1K dataset (if not already present)
- Create necessary directory structures
- Verify environment setup

#### Step 0.2: Generate PEKA Datasets

**For Breast Cancer:**
```bash
bash 1_generate_peka_datasets_breast.sh
```

**For Other Cancer Types:**
```bash
bash 1_generate_peka_datasets_other.sh
```

These scripts process the raw HEST1K data into PEKA format suitable for training.

#### Step 0.3: Process scLLM Embeddings
```bash
bash 2_scLLM_embedding_process.sh
```

Generates single-cell embeddings using foundation models for knowledge distillation.

#### Step 0.4: Extract Image Features

Run feature extraction for each cancer type:
```bash
bash 4_extract_img_features_breast.sh
bash 4_extract_img_features_kidney.sh
bash 4_extract_img_features_liver.sh
bash 4_extract_img_features_lung.sh
```

#### Step 0.5: Generate Cluster Labels for Knowledge Distillation
```bash
bash 5_generate_cluster_label_for_KD.sh
```

Creates cluster labels needed for the knowledge distillation process.

#### Step 0.6: Verify Data Preparation
```bash
bash 3_exp_checker.sh
```

Validates that all preprocessing steps completed successfully.

### Phase 1: Model Training with Dual Encoders

#### Step 1.1: Knowledge Distillation with LoRA Training

**Basic Training:**
```bash
cd scripts/1_train_with_2_encoders
python kd_lora_train.py \
    --dataset_config dataset_configs/breast_cancer.yaml \
    --model_config model_configs/kd_lora_base.yaml \
    --output_dir ../../OUTPUT/kd_lora_breast \
    --experiment_name "breast_kd_lora_exp1"
```

**Training with Clustering:**
```bash
python kd_lora_train_with_cluster.py \
    --dataset_config dataset_configs/breast_cancer.yaml \
    --model_config model_configs/kd_lora_cluster.yaml \
    --output_dir ../../OUTPUT/kd_lora_breast_cluster \
    --experiment_name "breast_kd_lora_cluster_exp1"
```

**Automated Training Script:**
```bash
bash train_kd_lora.sh
```

#### Step 1.2: Model Inference
```bash
python kd_lora_inference.py \
    --model_checkpoint ../../OUTPUT/kd_lora_breast/best_model.ckpt \
    --dataset_config dataset_configs/breast_cancer.yaml \
    --output_dir ../../OUTPUT/inference_results
```

#### Step 1.3: Reproduce Experiments
```bash
python reproduce_kd_lora_experiment.py \
    --config_file experiment_configs/reproduction_config.yaml
```

### Phase 2: Downstream Gene Expression Prediction

#### Step 2.1: Process Highly Variable Genes
```bash
cd scripts/2_downstream_gene_pred
python step1_process_hvg.py \
    --cancer_type breast \
    --input_dir ../../DATA/processed \
    --output_dir ../../OUTPUT/hvg_analysis
```

#### Step 2.2: Extract Feature Vectors
```bash
python step2_inference_feature_vectors.py \
    --model_checkpoint ../../OUTPUT/kd_lora_breast/best_model.ckpt \
    --dataset_config ../1_train_with_2_encoders/dataset_configs/breast_cancer.yaml \
    --output_dir ../../OUTPUT/feature_vectors
```

#### Step 2.3: Gene Expression Regression with K-Fold Validation

**Manual Execution:**
```bash
python step3_task_gene_expr_reg_KFold.py \
    --gene_list_json top_50_genes_Visium_Homo_sapien_Breast_Cancer.json \
    --feature_dir ../../OUTPUT/feature_vectors \
    --output_dir ../../OUTPUT/gene_regression \
    --cancer_type breast \
    --k_folds 5
```

**Automated Regression (H0 Model):**
```bash
bash 2_auto_reg_KFold_auto_H0.sh
```

**Automated Regression (UNI Model):**
```bash
bash 2_auto_reg_KFold_auto_UNI.sh
```

#### Step 2.4: Generate Labels for Different Cancer Types
```bash
bash 1_generate_labels_breast.sh
```

#### Step 2.5: Analysis and Visualization

**Compare Mutual Information:**
```bash
python 3_compare_mutual_information.py \
    --results_dir ../../OUTPUT/gene_regression \
    --output_dir ../../OUTPUT/analysis
```

**Plot Gene Correlations:**
```bash
python 4_plot_gene_correlation.py \
    --results_dir ../../OUTPUT/gene_regression \
    --gene_list top_50_genes_Visium_Homo_sapien_Breast_Cancer.json \
    --output_dir ../../OUTPUT/visualizations
```

### Complete Pipeline Execution

For a full end-to-end experiment:

```bash
# Phase 0: Data Preparation
cd scripts/0_download_dataset
bash 0_config_and_download.sh
bash 1_generate_peka_datasets_breast.sh
bash 2_scLLM_embedding_process.sh
bash 4_extract_img_features_breast.sh
bash 5_generate_cluster_label_for_KD.sh
bash 3_exp_checker.sh

# Phase 1: Model Training
cd ../1_train_with_2_encoders
bash train_kd_lora.sh

# Phase 2: Gene Prediction
cd ../2_downstream_gene_pred
bash 1_generate_labels_breast.sh
bash 2_auto_reg_KFold_auto_UNI.sh
python 3_compare_mutual_information.py
python 4_plot_gene_correlation.py
```

## Configuration

### Environment Variables

Ensure your `.env` file contains:
```
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_ENTITY=your_wandb_entity_name
HEST1K_STORAGE_PATH=/path/to/hest1k/dataset
REPO_PATH=/path/to/peka/repository
DATA_PATH=/path/to/data/directory
PROJECT_PATH=/path/to/project/root
```

### Model Configuration

Model configurations are stored in `hydra_zen/Configs/`. Key configuration files include:
- Dataset configurations for different cancer types
- Model architecture configurations
- Training hyperparameter configurations
- Evaluation metric configurations

### Experiment Tracking

PEKA uses Weights & Biases for experiment tracking. Ensure you have:
1. A WANDB account
2. Proper API key in your `.env` file
3. Correct entity name configured

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration files
   - Use gradient accumulation
   - Consider using mixed precision training

2. **Dataset Download Issues**
   - Check internet connection
   - Verify HEST1K_STORAGE_PATH permissions
   - Ensure sufficient disk space

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path configuration
   - Ensure external models are properly initialized

4. **HuggingFace Authentication**
   - Verify HF_TOKEN is valid
   - Check token permissions
   - Ensure you're logged in: `huggingface-cli login`

5. **WANDB Issues**
   - Verify WANDB_API_KEY is correct
   - Check entity name
   - Try: `wandb login`

### Performance Optimization

1. **Memory Optimization**
   - Use gradient checkpointing
   - Implement data loading optimizations
   - Consider model parallelism for large models

2. **Speed Optimization**
   - Use mixed precision training
   - Optimize data loading with multiple workers
   - Consider distributed training for multiple GPUs

### Getting Help

- Check the issue tracker for known problems
- Review configuration files for parameter settings
- Verify environment setup using the configuration script
- Check WANDB logs for detailed training information

---

**Note**: This project requires significant computational resources and storage space. Ensure your system meets the hardware requirements before beginning the experimental pipeline.
