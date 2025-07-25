"""
Gene expression binning utility for PEKA.
This script reads aligned_adata files and creates binned versions of gene expression data.
Supports both specific gene binning and highly variable genes selection.
"""

import argparse
import os
import sys
import numpy as np
import scanpy as sc
from pathlib import Path
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse
import logging
import json

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Gene expression binning utility')
    parser.add_argument("--project_root", type=str, default='./',
                        help='Project folder')
    parser.add_argument("--database_root", type=str, default='./PEKA/DATA/',
                        help='Processed database storage folder')
    parser.add_argument("--bin_nb", type=int, default=10,
                        help='Number of bins for gene expression discretization')
    parser.add_argument("--hvg_json", type=str, default=None,
                        help='Path to JSON file containing list of genes to process')
    parser.add_argument("--n_top_hvg", type=int, default=200,
                        help='Number of highly variable genes to select (used only if hvg_json is not provided)')
    parser.add_argument("--hvg_flavor", type=str, default='seurat',
                        help='Method for HVG selection (seurat/cell_ranger/seurat_v3)')
    parser.add_argument("--batch_key", type=str, default=None,
                        help='Batch key for HVG calculation')
    return parser.parse_args()

def setup_paths(args):
    """Setup project paths and environment"""
    proj_path = os.path.abspath(args.project_root)
    sys.path.append(proj_path)
    sys.path.append(os.path.join(proj_path, "PEKA"))
    
    # Add external models path
    external_module_path = os.path.join(proj_path, "PEKA/peka/External_models/")
    sys.path.append(external_module_path)
    print(f" ⭐️ external_module_path: {external_module_path}")
    sys.path.append(os.path.join(external_module_path, "HEST/src/"))
    
    # Load environment variables
    from dotenv import load_dotenv
    env_path = os.path.join(proj_path, "PEKA", ".env")
    load_dotenv(dotenv_path=env_path)
    
    return proj_path

def get_database_configs(database_root):
    """Read database configurations"""
    from peka.Data.database_helper import check_database_status
    datasets_info = check_database_status(database_root, None)
    return datasets_info



def load_genes_from_json(json_path):
    """Load gene list from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or 'genes' not in data:
                raise ValueError("JSON file must contain a 'genes' key with list of genes")
            return data['genes']
    except Exception as e:
        logger.error(f"Error loading genes from JSON file: {e}")
        raise


def process_adata_file(adata_path, bin_nb, 
                      hvg_json=None, n_top_hvg=200, hvg_flavor='seurat', batch_key=None):
    """Process a single adata file and create binned version"""

    from peka.Data.utils import calcuate_hvg, aggregate_duplicate_genes, generate_binned_adata
    print(f"Processing {adata_path}")
    
    # Read original adata
    adata = sc.read_h5ad(adata_path)
    
    # Handle duplicate gene names first
    if not adata.var_names.is_unique:
        print("Found duplicate gene names. Aggregating by taking mean values...")
        adata = aggregate_duplicate_genes(adata)
    
    # If JSON file is provided, load genes from it
    if hvg_json is not None:
        genes = load_genes_from_json(hvg_json)
        valid_genes = [gene for gene in genes if gene in adata.var_names]
        if len(valid_genes) < len(genes):
            missing_genes = set(genes) - set(valid_genes)
            logger.warning(f"Some genes were not found in the dataset: {missing_genes}")
        adata_with_hvg = adata[:, valid_genes]
    else:
        # Calculate highly variable genes
        raise NotImplementedError('need to generate hvg first')
    
    return generate_binned_adata(adata_with_hvg, bin_nb)

def process_database(database_path, bin_nb, 
                     hvg_json=None, n_top_hvg=200, hvg_flavor='seurat', batch_key=None):
    """Process all adata files in a database"""
    aligned_dir = os.path.join(database_path, 'aligned_adata')
    binned_dir = os.path.join(database_path, 'binned_adata')
    
    if not os.path.exists(aligned_dir):
        print(f"Warning: aligned_adata directory not found at {aligned_dir}")
        return
        
    os.makedirs(binned_dir, exist_ok=True)
    print(f"Processing aligned_adata in: {aligned_dir}")
    print(f"Saving binned data to: {binned_dir}")
    
    # Process each adata file
    for adata_file in os.listdir(aligned_dir):
        if adata_file.endswith('.h5ad'):
            input_path = os.path.join(aligned_dir, adata_file)
            output_path = os.path.join(binned_dir, adata_file)
            
            print(f"\nProcessing file: {adata_file}")
            # Process and save binned adata
            adata_bin = process_adata_file(
                input_path, 
                bin_nb,
                hvg_json=hvg_json,
                n_top_hvg=n_top_hvg,
                hvg_flavor=hvg_flavor,
                batch_key=batch_key
            )
            adata_bin.write_h5ad(output_path)
            print(f"Saved binned data to {output_path}")

def main():
    print("🤖 Running 6_gene_bin_generator.py")
    args = parse_args()
    proj_path = setup_paths(args)
    
    # Get database configurations
    datasets_info = get_database_configs(args.database_root)
    
    # Process each dataset directory
    for dataset_info in datasets_info:
        dataset_name = list(dataset_info.keys())[0]
        dataset_obj = dataset_info[dataset_name]
        
        dataset_path = os.path.join(args.database_root, dataset_name)
        if os.path.exists(dataset_path):
            print(f"\n📊 Processing dataset: {dataset_name}")
            print(f"Dataset path: {dataset_path}")
            process_database(
                dataset_path, 
                args.bin_nb,
                hvg_json=args.hvg_json,
                n_top_hvg=args.n_top_hvg,
                hvg_flavor=args.hvg_flavor,
                batch_key=args.batch_key
            )
        else:
            print(f"Warning: Dataset directory not found: {dataset_path}")

if __name__ == "__main__":
    main()
