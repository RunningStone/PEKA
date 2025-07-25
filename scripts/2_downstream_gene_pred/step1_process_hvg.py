#!/usr/bin/env python
import os
import scanpy as sc
import anndata as ad
from pathlib import Path
from scipy.sparse import issparse
from peka.Data.utils import calculate_hvg
import numpy as np
import json
def process_anndata_folder(input_dir: str, output_dir: str, output_filename: str, batch_key: str, n_top_hvg: int = 50):
    """
    Process all h5ad files in a folder, merge them, and calculate HVG.
    """
    h5ad_files = list(Path(input_dir).glob("*.h5ad"))
    if not h5ad_files:
        raise ValueError(f"No .h5ad files found in {input_dir}")
    
    print(f"Found {len(h5ad_files)} h5ad files")
    
    # Read and merge all files
    adatas = []
    for file in h5ad_files:
        print(f"Reading {file}")
        adata = sc.read_h5ad(file)
        print(f"Original adata shape: {adata.shape}")
        # 确保变量名唯一
        adata.var_names_make_unique()
        # 如果batch_key不存在，添加文件名作为batch
        if batch_key not in adata.obs:
            adata.obs[batch_key] = file.stem
        # 确保batch_key是category类型
        adata.obs[batch_key] = adata.obs[batch_key].astype('category')
        print(f"Adding {file} to adatas, adata shape: {adata.shape}")
        adatas.append(adata)
    
    print("Concatenating datasets...")
    # 改用outer join以保留所有基因
    combined_adata = ad.concat(
        adatas,
        join='outer',  # 保留所有基因
        merge='first',  # 使用重复注释的第一次出现
        fill_value=0  # 用0填充缺失值
    )
    
    # 再次确保基因名唯一
    combined_adata.var_names_make_unique()
    print(f"Combined dataset shape: {combined_adata.shape}")
    
    # 如果数据是稀疏的，转换为密集矩阵
    if issparse(combined_adata.X):
        combined_adata.X = combined_adata.X.toarray()
    
    # 基本预处理
    sc.pp.normalize_total(combined_adata)
    sc.pp.log1p(combined_adata)
    
    # 计算HVG
    print("Calculating highly variable genes...")
    try:
        sc.pp.highly_variable_genes(
            combined_adata,
            n_top_genes=n_top_hvg,
            batch_key=batch_key,
            flavor='seurat',
            subset=False
        )
        hvg_genes = combined_adata.var_names[combined_adata.var.highly_variable].tolist()
    except Exception as e:
        print(f"Warning: Error in calculating HVG: {str(e)}")
        print("Falling back to basic variance calculation...")
        # 使用基本方差计算作为后备方案
        gene_vars = np.var(combined_adata.X, axis=0)
        top_var_idx = np.argsort(gene_vars)[-n_top_hvg:]
        hvg_genes = combined_adata.var_names[top_var_idx].tolist()
    
    # 保存HVG列表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        hvg_json_path = os.path.join(output_dir, output_filename)
        with open(hvg_json_path, 'w') as f:
            json.dump({"genes": hvg_genes}, f, indent=2)
        print(f"Saved HVG list to: {hvg_json_path}")
    
    print(f"Found {len(hvg_genes)} highly variable genes")
    return hvg_genes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process AnnData files and calculate HVG")
    parser.add_argument("--input-dir", required=True, help="Directory containing h5ad files")
    parser.add_argument("--output-dir", required=True, help="Directory to save HVG results")
    parser.add_argument("--output-filename", required=True, help="Name of the output JSON file")
    parser.add_argument("--batch-key", required=True, help="Key for batch information in adata.obs")
    parser.add_argument("--n-top-hvg", type=int, default=50, help="Number of HVG to select")
    
    args = parser.parse_args()
    process_anndata_folder(args.input_dir, args.output_dir, args.output_filename, args.batch_key, args.n_top_hvg)
