"""
1. 生成指定模型在指定数据集上的的推理结果
2. 推理结果保存在指定位置
"""

import os
import json
import torch
import random
import numpy as np
import scipy
import scipy.sparse
import pandas as pd
import scanpy as sc
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import glob
import argparse
import sys
import h5py
from datetime import datetime

def setup_paths(project_root: str) -> str:
    """Setup project paths and environment"""
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "HistoMIL2"))
    
    # Add external models path
    external_module_path = os.path.join(project_root, "HistoMIL2/histomil2/External_models/")
    sys.path.append(external_module_path)
    print(f" ⭐️ external_module_path: {external_module_path}")
    
    # Load environment variables
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, "HistoMIL2", ".env")
    load_dotenv(dotenv_path=env_path)
    
    return project_root

def set_args():
    parser = argparse.ArgumentParser(description='Generate feature vectors using trained model')
    parser.add_argument('--project_root', type=str, required=True, help='Path to HistoMIL2 project')
    parser.add_argument('--tissue_type', type=str, required=True, help='Type of tissue (e.g., breast, other_cancer)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., breast_visium_26k)')
    parser.add_argument('--scllm', type=str, required=True, help='Name of the scLLM model')
    parser.add_argument('--scllm_ckpt', type=str, default="default", help='Name of the scLLM checkpoint')
    parser.add_argument('--target_scllm_dim', type=int, default=512, help='Dimension of the scLLM model')
    parser.add_argument('--image_encoder_name', type=str, default="H0", help='Name of the image encoder model:(H0,UNI)')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint file')
    parser.add_argument('--lora_r', type=int, default=64, help='Rank of the LoRA matrix')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save feature vectors')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    
    return args

def main():
    # Parse arguments
    args = set_args()
    
    # Setup paths and environment
    project_root = setup_paths(args.project_root)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" ⭐️ Using device: {device}")
    
    # Setup dataset paths
    dataset_save_folder = f"{args.project_root}/HistoMIL2/DATA/{args.tissue_type}/{args.dataset_name}/"
    print(f" ⭐️ dataset_save_folder: {dataset_save_folder}")
    
    # Import required modules after path setup
    from hydra_zen import instantiate, load_from_yaml
    from histomil2.DownstreamTasks_helper.inference import inference_from_folder
    all_configs_path = f"{args.project_root}/HistoMIL2/hydra_zen/Configs/"
    try:
        # Load model configuration
        print(" Loading model config...")
        model_config = load_from_yaml(os.path.join(all_configs_path, args.model_config))
        
        # Initialize model
        print(" Initializing model...")
        model = instantiate(model_config, target_dim=args.target_scllm_dim, lora_r=args.lora_r)
        
        # Load checkpoint
        print(f" Loading checkpoint from {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location=device) 
        # 移除"model."前缀
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('model.'):
                new_k = k[6:]  # 截掉'model.'
                new_state_dict[new_k] = v
            else:
                if k.startswith('classifier.'):
                    # 截掉 classifier. 原来的checkpoint中包含classifier, 在inference时不需要
                    continue
                else:
                    new_state_dict[k] = v
            


        model.load_state_dict(new_state_dict) # not need to keep classifier for inference
        model = model.to(device)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f" ⭐️ Output directory: {args.output_dir}")
        
        # Run inference
        print(" Starting inference...")
        inference_from_folder(
            model=model,
            dataset_save_folder=dataset_save_folder,
            scLLM_emb_name=args.scllm,
            scLLM_emb_ckpt=args.scllm_ckpt,
            output_dir=args.output_dir,
            adata_prefix="HEST_breast_adata_",
            img_prefix="patch_224_0.5_"
        )
        
        print(" ✅ Inference completed successfully!")
        
    except Exception as e:
        print(f" ❌ Error during inference: {str(e)}")
        raise e

if __name__ == '__main__':
    main()