import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import anndata
from typing import Optional, Dict, Any
import argparse
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Will use sklearn KMeans for CPU clustering.")

def setup_paths(project_root: str) -> None:
    """Setup project paths and environment"""
    sys.path.append(f'{project_root}/')
    sys.path.append(f'{project_root}/PEKA/')
    sys.path.append(f'{project_root}/PEKA/peka/External_models/')

def gpu_kmeans_cluster(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Using FAISS in GPU for KMeans clustering"""
    # Ensure data is float32
    data = data.astype(np.float32)
    d = data.shape[1]  # Data dimension
    
    # Initialize kmeans object
    kmeans = faiss.Kmeans(d, n_clusters, niter=300, verbose=True, gpu=True)
    
    # Run clustering
    print("Using FAISS-GPU for KMeans clustering...")
    kmeans.train(data)
    
    # Get nearest centroids
    _, labels = kmeans.index.search(data, 1)
    return labels.reshape(-1)

def cpu_kmeans_cluster(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Using sklearn in CPU for KMeans clustering"""
    print("Using sklearn-CPU for KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

def process_dataset(dataset_folder: str,
                     scLLM_emb_name: str, n_clusters: int, 
                     scLLM_emb_ckpt: str = "default", use_gpu: bool = True) -> None:
    """Process dataset and generate cluster labels"""
    # Set folder paths
    paired_seq_folder = f'{dataset_folder}/scLLM_embed/{scLLM_emb_name}/{scLLM_emb_ckpt}/paired_seq/'
    embedding_folder = f'{dataset_folder}/scLLM_embed/{scLLM_emb_name}/{scLLM_emb_ckpt}/embeddings/'

    # Collect all embeddings
    all_embeddings = []
    file_barcode_map = []  # Record each embedding's corresponding file and barcode

    # Collect anndata files
    anndata_files = [f for f in os.listdir(paired_seq_folder) if f.endswith('.h5ad')]
    
    print("Collecting all dataset embeddings...")
    for anndata_file in anndata_files:
        # Read anndata file
        adata_path = os.path.join(paired_seq_folder, anndata_file)
        adata = anndata.read_h5ad(adata_path)
        barcodes = adata.obs.index.values
        filter_flags = adata.obs['filter_flag'].values
        
        # Get QC-passed barcodes
        filter_barcodes = barcodes[~filter_flags]
        
        # Find corresponding embedding file
        base_name = os.path.splitext(anndata_file)[0]
        emb_file = os.path.join(embedding_folder, f"{base_name}.npy")
        
        if not os.path.exists(emb_file):
            print(f"Warning: embedding file not found for {anndata_file}")
            continue
            
        # Load embeddings
        embeddings = np.load(emb_file)
        
        # Ensure counts match
        if len(filter_barcodes) != embeddings.shape[0]:
            print(f"Warning: mismatch in {anndata_file}, skipping")
            continue
            
        # Add to total collection
        all_embeddings.append(embeddings)
        file_barcode_map.extend([(anndata_file, barcode) for barcode in filter_barcodes])

    # Merge all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"Begin to cluster {len(all_embeddings)} samples...")
    
    # Select clustering method
    if use_gpu and FAISS_AVAILABLE:
        cluster_labels = gpu_kmeans_cluster(all_embeddings, n_clusters)
    else:
        if use_gpu and not FAISS_AVAILABLE:
            print("Warning: GPU clustering requested but FAISS not available. Falling back to CPU clustering.")
        cluster_labels = cpu_kmeans_cluster(all_embeddings, n_clusters)
    
    print("Begin to write cluster labels back to anndata files...")
    # Write cluster labels back to corresponding anndata files
    current_idx = 0
    for anndata_file in anndata_files:
        adata_path = os.path.join(paired_seq_folder, anndata_file)
        if not os.path.exists(adata_path):
            continue
            
        adata = anndata.read_h5ad(adata_path)
        barcodes = adata.obs.index.values
        filter_flags = adata.obs['filter_flag'].values
        filter_barcodes = barcodes[~filter_flags]
        
        # Get the labels for this file
        n_samples = len(filter_barcodes)
        file_labels = cluster_labels[current_idx:current_idx + n_samples]
        
        # Create a label array with the same size as the original obs, initialized to -1
        full_labels = np.full(len(barcodes), -1)
        # Fill in the cluster labels for non-filtered positions
        full_labels[~filter_flags] = file_labels
        
        # Add to obs
        adata.obs[f'gen_clustered_label_{n_clusters}'] = full_labels
        
        # Save updated anndata file
        adata.write_h5ad(adata_path)
        
        current_idx += n_samples

    print("Cluster labels generated successfully!")

def main():
    parser = argparse.ArgumentParser(description='Generate cluster labels for knowledge distillation')
    parser.add_argument('--project_root', type=str, required=True, help='Project root directory')
    parser.add_argument('--tissue_name', type=str, required=True, help='Tissue name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--scLLM_embedder_name', type=str, required=True, help='scLLM embedder name')
    parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters')
    parser.add_argument('--ckpt', type=str, default='default', help='Checkpoint name')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for clustering if available')
    
    args = parser.parse_args()
    
    # Setup paths
    setup_paths(args.project_root)
    
    # Build dataset path
    dataset_folder = f"{args.project_root}/PEKA/DATA/{args.tissue_name}/{args.dataset_name}"
    
    # Process dataset
    process_dataset(
        dataset_folder=dataset_folder,
        scLLM_emb_name=args.scLLM_embedder_name,
        n_clusters=args.n_clusters,
        scLLM_emb_ckpt=args.ckpt,
        use_gpu=args.use_gpu
    )

if __name__ == "__main__":
    main()
