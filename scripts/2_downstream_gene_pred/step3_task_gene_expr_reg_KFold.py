"""
Train gene expression regressors using scLLM embeddings.
This script implements a pipeline to:
1. Load genes from JSON file
2. Train regressors for each gene using scLLM embeddings
3. Save model checkpoints and evaluation metrics
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

def setup_paths(project_root: str) -> str:
    """Setup project paths and environment"""
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "PEKA"))
    
    # Add external models path
    external_module_path = os.path.join(project_root, "PEKA/peka/External_models/")
    sys.path.append(external_module_path)
    print(f" ⭐️ external_module_path: {external_module_path}")
    
    # Load environment variables
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, "PEKA", ".env")
    load_dotenv(dotenv_path=env_path)
    
    return project_root


def load_genes_from_json(json_path: str) -> List[str]:
    """Load gene list from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or 'genes' not in data:
                raise ValueError("JSON file must contain a 'genes' key with list of genes")
            return data['genes']
    except Exception as e:
        print(f"Error loading genes from JSON file: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', type=str, required=True, help='Path to PEKA project')
    parser.add_argument('--tissue_type', type=str, required=True, help='Type of tissue (e.g., breast, other_cancer)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., breast_visium_26k)')
    parser.add_argument('--embedder_name', type=str, required=True, help='Name of the scLLM model')
    parser.add_argument('--image_encoder_name', type=str, default="H0", help='Name of the image encoder model:(H0,UNI)')
    parser.add_argument('--image_backbone', type=str, default="", help='Name of the image backbone model:(H-optimus-0 , UNI)')
    
    parser.add_argument('--gene_list_json', type=str, required=True, help='Path to JSON file containing list of genes')
    parser.add_argument('--output_root', type=str, required=True, help='Root directory for output')
    parser.add_argument('--use_binned', action='store_true', help='If set, use binned data; otherwise use continuous values')
    parser.add_argument('--feature_type', type=str, required=True, help='Type of feature ( peka, image, scLLM)')
    parser.add_argument('--with_independent_test_set', action='store_true', help='If set, use binned data; otherwise use continuous values')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--mask_zero_values', action='store_true', help='If set, mask zero values')
    parser.add_argument('--not_use_scLLM_name_as_subfolder', action='store_true', help='If set, do not use scLLM name as subfolder')
    
    args = parser.parse_args()
    
    # Create config from command line arguments
    config = {
        'project_root': args.project_root,
        'tissue_type': args.tissue_type,
        'dataset_name': args.dataset_name,
        'embedder_name': args.embedder_name,
        'gene_list_json': args.gene_list_json,
        'output_root': args.output_root,
        'use_binned': args.use_binned,
        'feature_type': args.feature_type,
        'with_independent_test_set': args.with_independent_test_set,
        'image_encoder_name': args.image_encoder_name,
        'image_backbone': args.image_backbone,
        'mask_zero_values': args.mask_zero_values
    }
    # Setup paths
    project_root = setup_paths(config['project_root'])
    print(f"Project root: {project_root}")
    print(config)
    from peka.DownstreamTasks_helper.train_and_val_exp import train_and_val_step,train_and_val_step_KFold
    from peka.DownstreamTasks_helper.gene_expression_prediction import load_data,get_dataset_paths
    from peka.DownstreamTasks_helper.train_and_val_exp import plot_gene_correlations
    


    paths = get_dataset_paths(project_root, 
                            config['tissue_type'], 
                            config['dataset_name'], 
                            config['embedder_name'],
                            feature_type=config['feature_type'],
                            image_encoder_name=config['image_encoder_name'],
                            image_backbone=config['image_backbone'],
                            use_scLLM_name_as_subfolder=False if args.not_use_scLLM_name_as_subfolder else True)
    
    # Load genes from JSON file
    genes = load_genes_from_json(config['gene_list_json'])
    print(f"Loaded {len(genes)} genes from JSON file")
    
    # Setup output directory
    data_type = "binned" if config['use_binned'] else "raw"
    output_root = config['output_root'] + f"/{config['image_encoder_name']}/"
    output_dir = os.path.join(output_root, f'{config["feature_type"]}_gene_level_{data_type}_regression_{config["embedder_name"]}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Load data
    embeddings_dict, labels_dict = load_data(paths, genes, 
                                           feature_type=config['feature_type'],  
                                           img_prefix="patch_224_0.5_",
                                           embed_prefix="HEST_breast_adata_",
                                           use_binned=config['use_binned'],
                                           mask_zero_values=config['mask_zero_values'])
    
    # Filter out genes not found in all datasets
    missing_genes = set(genes) - set(labels_dict.keys())
    if missing_genes:
        print("\nWarning: The following genes were not found in all datasets:")
        for gene in sorted(missing_genes):
            print(f"- {gene}")
        
        # Update genes list
        genes = [gene for gene in genes if gene in labels_dict]
        print(f"\nProceeding with {len(genes)} genes that are present in all datasets")
    
    # Train regressors for each gene
    results = []
    for gene in genes:
        print(f"\nTraining regressor for gene: {gene}")
        
        # Get labels and embeddings for this gene
        labels = labels_dict[gene]
        embeddings = embeddings_dict[gene]
        if embeddings.shape[0] >321:# train should larger than 4/5 of 320 = 256
            result = train_and_val_step_KFold(embeddings, labels,
                                    output_dir,gene, 
                                    config,
                                    Ksplit=5,
                                    epochs=args.epochs)

            results.append(result)
        else:
            print(f"Skipping {gene} due to insufficient data points")
            print(f"Labels shape: {labels.shape}, Embeddings shape: {embeddings.shape}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'gene_regression_results.csv'), index=False)
    
    # Plot gene correlations
    plot_gene_correlations(results_df, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    
if __name__ == '__main__':
    main()
