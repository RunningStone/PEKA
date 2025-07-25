import os
from typing import Dict, Any, List, Tuple
import torch
import scanpy as sc
import pandas as pd
import numpy as np
import yaml
import glob
import h5py
import scipy
import scipy.sparse

# Sort files by index number
def get_index(filename):
    return int(os.path.basename(filename).split('_')[-1].split('.')[0])

def get_dataset_paths(project_root: str, tissue_type: str, 
                      dataset_name: str, 
                      embedder_name: str,
                      checkpoint_name: str="default_model",
                      feature_type: str="scLLM",
                      image_encoder_name: str="H0", # H-optimus-0
                      image_backbone:str = "", # H-optimus-0
                      use_scLLM_name_as_subfolder:bool=True) -> Dict[str, str]:
    """Get dataset related paths
    
    Args:
        project_root: Path to PEKA project
        tissue_type: Type of tissue (e.g., breast, other_cancer)
        dataset_name: Name of the dataset (e.g., breast_visium_26k)
        embedder_name: Name of the scLLM model
        feature_type: Type of feature (e.g., peka, image_encoder, scLLM)
        
    Returns:
        Dictionary containing all relevant paths
    """
    data_root = os.path.join(project_root, "PEKA", "DATA")
    paths = {
        'dataset_root': os.path.join(data_root, tissue_type, dataset_name),
        'ref_dir': os.path.join(data_root, tissue_type, dataset_name, 'scLLM_embed'),
        'img_dir': os.path.join(data_root, tissue_type, dataset_name, 'patches'),
        'bin_dir': os.path.join(data_root, tissue_type, dataset_name, 'binned_adata'),
        
    }
    # Add specific subdirectories
    paths['seq_path'] = os.path.join(paths['ref_dir'], embedder_name, checkpoint_name, 'paired_seq')
    paths['img_path'] = os.path.join(paths['img_dir'])

    # for different feature types
    print(f"Feature type: {feature_type} to set embed_dir and embed_path")
    if feature_type == "scLLM":
        paths['embed_dir'] = os.path.join(data_root, tissue_type, dataset_name, 'scLLM_embed')
        paths['embed_path'] = os.path.join(paths['embed_dir'], embedder_name, checkpoint_name, 'embeddings')
    elif feature_type == "peka":
        paths['embed_dir'] = os.path.join(data_root, tissue_type, dataset_name, 'peka_embed', image_encoder_name)
        if use_scLLM_name_as_subfolder:
            paths['embed_path'] = os.path.join(paths['embed_dir'], embedder_name, checkpoint_name)
        else:
            paths['embed_path'] = os.path.join(paths['embed_dir'])
    elif feature_type == "image_encoder":
        paths['embed_dir'] = os.path.join(data_root, tissue_type, dataset_name, 'patches_embed')
        paths['embed_path'] = os.path.join(paths['embed_dir'], image_backbone)
    elif feature_type == "image_encoder+peka":
        # embedding from peka
        paths['embed_dir'] = os.path.join(data_root, tissue_type, dataset_name, 'peka_embed', image_encoder_name)
        if use_scLLM_name_as_subfolder:
            paths['embed_path'] = os.path.join(paths['embed_dir'], embedder_name, checkpoint_name)
        else:
            paths['embed_path'] = os.path.join(paths['embed_dir'])
        # paired image embedding from image backbone
        paths['embed_dir_add'] = os.path.join(data_root, tissue_type, dataset_name, 'patches_embed')
        paths['embed_path_add'] = os.path.join(paths['embed_dir_add'], image_backbone)
    else:
        raise ValueError(f"Unrecognized feature type: {feature_type}")
    
    print(paths)
    # Verify paths exist
    for key, path in paths.items():
        if not os.path.exists(path):
            #raise ValueError(f"Path does not exist: {path} ({key})")
            print(f"Path does not exist: {path} ({key})")
            
    return paths

def feature_type_settings(feature_type: str,
                            img_prefix: str, embed_prefix:str,):
    """Get settings for different feature types
    
    Args:
        feature_type: Type of feature (e.g., scLLM, peka, image_encoder)
        img_prefix: Prefix for image files
        embed_prefix: Prefix for embedding files
        
    Returns:
        Tuple containing:
        - img_prefix: Updated image prefix
        - embed_prefix: Updated embedding prefix
        - embedding_need_mask: Type of mask needed for embeddings
    """
    if feature_type == "scLLM":
        embedding_need_mask = "seq_mask" # in seq_qc, but no intersection with image filter
        return img_prefix, embed_prefix, embedding_need_mask
    elif feature_type == "peka":
        embedding_need_mask = None # theoretically it is seq_qc after image filter
        return img_prefix, embed_prefix, embedding_need_mask
    elif feature_type == "image_encoder":
        embedding_need_mask = "image_mask"
        embed_prefix = img_prefix # img_prefix is patch_224_0.5_ 形式的
        return img_prefix, embed_prefix, embedding_need_mask
    elif feature_type == "image_encoder+peka":
        embedding_need_mask = None
        #embed_prefix = img_prefix # img_prefix is patch_224_0.5_ 形式的
        return img_prefix, embed_prefix, embedding_need_mask
    else:
        raise ValueError(f"Unrecognized feature type: {feature_type}")

def _mask_embedding(embedding: np.ndarray,seq_mask,image_mask,valid_barcodes):
    
    # 没有指定mask就匹配一个mask
    current_emb_len = embedding.shape[0]
    valid_emb_len = len(valid_barcodes)
    if current_emb_len != valid_emb_len:
        print("Warning: number of embeddings does not match number of valid barcodes try to filter")
        if current_emb_len == len(image_mask):
            embedding = embedding[image_mask]
        elif current_emb_len == len(seq_mask):
            embedding = embedding[seq_mask]
        else:
            raise ValueError(f"Number of embeddings {current_emb_len} does not match number of valid barcodes {valid_emb_len} or seq_mask {len(seq_mask)} or image_mask {len(image_mask)}")
    assert embedding.shape[0] == valid_emb_len, f"Number of embeddings {embedding.shape[0]} does not match number of valid barcodes {valid_emb_len}"
    print(f"Filtered embedding shape: {embedding.shape}, now it has {valid_emb_len} cells")
    return embedding
def _filter_from_image_and_seq_qc(seq_file: str,feature_type: str,
                                  img_prefix: str, embed_prefix:str, 
                                  paths: Dict[str, str],):
    """
       Load embeddings and filter valid samples
       Args:
           seq_file: Path to paired seq file
           feature_type: Type of feature (e.g., scLLM, peka, image_encoder)
           img_prefix: Prefix for image files e.g. patch_224_0.5_
           embed_prefix: Prefix for embedding files e.g. HEST_breast_adata_
           paths: Dictionary containing paths to data files created by `prepare_data_paths`
       Returns:
           mask: Mask for filtering embeddings
    """
    
    # Get index from filename
    idx = get_index(seq_file)
    print(f"\nProcessing index {idx}")
    img_prefix, embed_prefix, embedding_need_mask = \
        feature_type_settings(feature_type, img_prefix, embed_prefix)
    # 1. Load paired seq data and get filter_flag
    #print(f"Reading paired seq file: {os.path.basename(seq_file)}")
    seq_adata = sc.read_h5ad(seq_file)
    filter_flags = seq_adata.obs['filter_flag'].values
    seq_barcodes = seq_adata.obs_names.values
    filter_barcodes = seq_barcodes[~filter_flags]
    print(f"Found {len(filter_barcodes)} cells after applying filter_flag")
    
    # 2. Load image barcodes
    img_file = os.path.join(paths['img_path'], 
                            img_prefix + str(idx) + ".h5")
    if not os.path.exists(img_file):
        raise ValueError(f"Image file not found: {img_file}")
        
    with h5py.File(img_file, 'r') as f:
        img_barcodes = f['barcode'][:, 0]
        img_barcodes = [bc.decode('utf-8') if isinstance(bc, bytes) else str(bc) 
                        for bc in img_barcodes]
    
    # 3. Find valid barcodes (intersection of filtered sequence barcodes and image barcodes)
    valid_barcodes = list(set(filter_barcodes).intersection(set(img_barcodes)))
    print(f"Found {len(valid_barcodes)} valid cells after intersecting sequence and image barcodes")
    
    # 4. Create mask for filtering embeddings
    # The mask should match the order of barcodes in the image file
    image_mask = [bc in valid_barcodes for bc in img_barcodes]
    seq_mask = [bc in valid_barcodes for bc in seq_barcodes]
    # 5. Load and filter embedding
    if feature_type in ["image_encoder+peka"]:
        # mixture of image and peka embeddings
        embed_file1 = os.path.join(paths['embed_path'], f"{embed_prefix}{str(idx)}.npy") #peka emb
        embed_file2 = os.path.join(paths['embed_path_add'], f"{img_prefix}{str(idx)}.npy") # image emb
        if not os.path.exists(embed_file1) or not os.path.exists(embed_file2):
            raise ValueError(f"Embedding file not found: {embed_file1} or {embed_file2}")
        
        print(f"Reading embedding file: {embed_file1}")
        print(f"Reading embedding file: {embed_file2}")

        embedding1 = np.load(embed_file1)
        embedding2 = np.load(embed_file2)
        print(f"Original embedding shape from file: {embedding1.shape}")
        print(f"Original embedding shape from file: {embedding2.shape}")
        valid_embedding1 = _mask_embedding(embedding1,seq_mask,image_mask,valid_barcodes)
        valid_embedding2 = _mask_embedding(embedding2,seq_mask,image_mask,valid_barcodes)
        embedding = np.concatenate((valid_embedding1, valid_embedding2), axis=1)
    
    else:
        embed_file = os.path.join(paths['embed_path'], f"{embed_prefix}{str(idx)}.npy")
        if not os.path.exists(embed_file):
            raise ValueError(f"Embedding file not found: {embed_file}")
            
        print(f"Reading embedding file: {embed_file}")
        
        embedding = np.load(embed_file)
        print(f"Original embedding shape from file: {embedding.shape}")
    
    embedding = _mask_embedding(embedding,seq_mask,image_mask,valid_barcodes)

    if len(valid_barcodes) != len(embedding):
            raise ValueError(f"Mismatch between number of embeddings ({len(embedding)}) and valid barcodes ({len(valid_barcodes)}) for index {idx}")
        
    return seq_adata, embedding, image_mask, seq_mask, valid_barcodes

def _create_binned_labels(embedding, paths, idx, genes, valid_barcodes, labels_dict, binned_prefix:str="HEST_breast_adata_"):
    """Create labels from binned data
    
    Args:
        paths: Dictionary containing paths to data files
        idx: Index of the current file
        genes: List of genes to process
        valid_barcodes: List of valid barcodes to filter data
        labels_dict: Dictionary to store labels
        binned_prefix: Prefix for binned data files
        
    Returns:
        Dictionary mapping gene names to their labels array
    """
    # Load and filter binned data
    bin_file = os.path.join(paths['bin_dir'], f"{binned_prefix}{idx}.h5ad")
    if not os.path.exists(bin_file):
        raise ValueError(f"Binned data file not found: {bin_file}")
    
    print(f"Reading binned data file: {os.path.basename(bin_file)}")
    bin_adata = sc.read_h5ad(bin_file)
    
    # Create a mapping of valid barcodes for this file
    valid_barcode_set = set(valid_barcodes)
    valid_bin_mask = [barcode in valid_barcode_set for barcode in bin_adata.obs_names]
    
    # Filter binned data
    filtered_adata = bin_adata[valid_bin_mask].copy()
    print(f"Filtered binned data from {len(bin_adata)} to {len(filtered_adata)} cells")
    
    # Verify counts after filtering
    if len(filtered_adata) != len(embedding):
        raise ValueError(f"After filtering, data length ({len(filtered_adata)}) does not match embedding length ({len(embedding)}) for index {idx}")
    
    # Store labels for requested genes
    for gene in genes:
        if gene not in filtered_adata.var_names:
            print(f"Warning: Gene {gene} not found in binned data")
            continue
            
        # Get gene expression values
        gene_idx = filtered_adata.var_names.get_loc(gene)
        if isinstance(filtered_adata.X, scipy.sparse.spmatrix):
            labels_dict[gene].append(filtered_adata.X[:, gene_idx].toarray().flatten())
        else:
            labels_dict[gene].append(filtered_adata.X[:, gene_idx])
            
    return labels_dict

def _create_raw_labels(embedding,seq_adata,seq_mask, genes,labels_dict,embeddings_dict, mask_zero_values:bool=True):
    # add log transform and normalize as normal bioinformatics pipeline
    print("Adding log transform and normalizing")
    sc.pp.normalize_total(seq_adata, target_sum=1e4)
    sc.pp.log1p(seq_adata, base=2)
    
    # Use continuous values from paired seq data
    filtered_seq_adata = seq_adata[seq_mask].copy()
    
    # Store labels for requested genes
    for gene in genes:
        if gene in filtered_seq_adata.var_names:
            gene_idx = filtered_seq_adata.var_names.get_loc(gene)
            gene_expr = filtered_seq_adata.X[:, gene_idx]
            
            # Convert sparse matrix to dense if needed
            if scipy.sparse.issparse(gene_expr):
                gene_expr = gene_expr.toarray().flatten()
            
            # Filter out cells where this gene has zero expression
            non_zero_mask = gene_expr != 0
            if np.sum(non_zero_mask) > 0:  
                if mask_zero_values:
                    labels_dict[gene].append(gene_expr[non_zero_mask])
                    embeddings_dict[gene].append(embedding[non_zero_mask])
                else:
                    labels_dict[gene].append(gene_expr)
                    embeddings_dict[gene].append(embedding)
            else:
                print(f"Warning: Gene {gene} has all zero expression in this batch")
    return labels_dict, embeddings_dict


def _label_stats(embeddings,final_labels,genes):
    # Final verification
    print(f"\n=== Final Statistics ===")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Number of genes processed: {len(final_labels)}")
    print(f"Labels shape for each gene: {next(iter(final_labels.values())).shape}")
    
    # Print label statistics for each gene
    print("\n=== Label Statistics ===")
    for gene, labels in final_labels.items():
        if np.random.random() < 0.2:
            min_val = np.min(labels)
            max_val = np.max(labels)
            mean_val = np.mean(labels)
            std_val = np.std(labels)
            print(f"\nGene: {gene}")
            print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
            print(f"  Mean ± Std: {mean_val:.3f} ± {std_val:.3f}")
    
    # Report missing genes
    missing_genes = set(genes) - set(final_labels.keys())
    if missing_genes:
        print("\nWarning: The following genes were not found in the dataset:")
        for gene in sorted(missing_genes):
            print(f"- {gene}")

def _load_binned_data(paths: Dict[str, str], 
                genes: List[str], 
                feature_type:str,
                img_prefix: str = "patch_224_0.5_",
                embed_prefix: str = "HEST_breast_adata_",
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load embeddings, expression data and labels
    
    Args:
        paths: Dictionary containing paths to data files
        genes: List of genes to process
        feature_type: Type of feature (e.g., scLLM, peka, image_encoder)
        img_prefix: Prefix for image files
        embed_prefix: Prefix for embedding files
        
    Returns:
        Tuple containing:
        - embeddings: numpy array of embeddings
        - labels_dict: dictionary mapping gene names to their labels array
    """
    print("Loading data...")
    # Get all paired seq files and sort by index
    paired_seq_files = glob.glob(os.path.join(paths['seq_path'], "*.h5ad"))
    if not paired_seq_files:
        raise ValueError(f"No paired seq files found in {paths['seq_path']}")
    
    paired_seq_files = sorted(paired_seq_files, key=get_index)
    
    # Initialize lists to store data
    embeddings_list = []
    labels_dict = {gene: [] for gene in genes}  # Initialize with requested genes only
    
    print("\n=== Processing files by index ===")
    
    for seq_file in paired_seq_files:
        # Get index from filename
        idx = get_index(seq_file)
        print(f"\nProcessing index {idx}")
        
        # get embedding
        seq_adata, embedding, embedding_mask, seq_mask,valid_barcodes = _filter_from_image_and_seq_qc(
                                    seq_file,
                                    feature_type,
                                    img_prefix,
                                    embed_prefix,
                                    paths)
        
        # Store embedding
        embeddings_list.append(embedding)
        
        # read binned labels
        labels_dict = _create_binned_labels(embedding, paths, idx, genes, 
                                valid_barcodes, labels_dict, binned_prefix=embed_prefix)

    # Concatenate embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Create embeddings dictionary with same embeddings for each gene
    embeddings_dict = {gene: embeddings for gene in genes if gene in labels_dict}
    
    # Filter labels_dict to only include genes that have data
    labels_dict = {gene: labels for gene, labels in labels_dict.items() if len(labels) > 0}
    
    # Concatenate labels for each gene
    for gene in labels_dict:
        if labels_dict[gene]:  # Check if we have any data for this gene
            try:
                labels_dict[gene] = np.concatenate(labels_dict[gene], axis=0)
            except Exception as e:
                print(f"Warning: Could not process gene {gene}: {str(e)}")
                del labels_dict[gene]
                del embeddings_dict[gene]
    
    _label_stats(embeddings_dict,labels_dict,genes)
    
    return embeddings_dict, labels_dict


def _load_raw_data(paths: Dict[str, str], 
                genes: List[str], 
                feature_type:str,
                img_prefix: str = "patch_224_0.5_",
                embed_prefix: str = "HEST_breast_adata_",
                mask_zero_values: bool = True,
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load embeddings, expression data and labels
    
    Args:
        paths: Dictionary containing paths to data files
        genes: List of genes to process
        feature_type: Type of feature (e.g., scLLM, peka, image_encoder)
        img_prefix: Prefix for image files
        embed_prefix: Prefix for embedding files
        
    Returns:
        Tuple containing:
        - embeddings_dict: dictionary mapping gene names to their embeddings array
        - labels_dict: dictionary mapping gene names to their labels array
    """
    print("Loading data...")
    # Get all paired seq files and sort by index
    paired_seq_files = glob.glob(os.path.join(paths['seq_path'], "*.h5ad"))
    if not paired_seq_files:
        raise ValueError(f"No paired seq files found in {paths['seq_path']}")
    
    paired_seq_files = sorted(paired_seq_files, key=get_index)
    
    # Initialize lists to store data
    embeddings_dict = {gene: [] for gene in genes}  # Initialize with requested genes only
    labels_dict = {gene: [] for gene in genes}
    
    print("\n=== Processing files by index ===")
    
    for seq_file in paired_seq_files:
        # get embedding
        seq_adata, embedding, embedding_mask, seq_mask,valid_barcodes = _filter_from_image_and_seq_qc(
                                    seq_file,
                                    feature_type,
                                    img_prefix,
                                    embed_prefix,
                                    paths)
        
        # get labels
        labels_dict, embeddings_dict = _create_raw_labels(embedding,seq_adata,seq_mask, genes,labels_dict,embeddings_dict, mask_zero_values)
    
    # Concatenate data for each gene
    final_labels = {}
    final_embeddings = {}
    
    for gene in genes:
        if gene in labels_dict and labels_dict[gene]:  # Check if we have any data for this gene
            try:
                final_labels[gene] = np.concatenate(labels_dict[gene], axis=0)
                final_embeddings[gene] = np.concatenate(embeddings_dict[gene], axis=0)
            except Exception as e:
                print(f"Warning: Could not process gene {gene}: {str(e)}")
    
    _label_stats(final_embeddings,final_labels,genes)
    
    return final_embeddings, final_labels


def load_data(paths: Dict[str, str], 
                genes: List[str], 
                feature_type:str,
                img_prefix: str = "patch_224_0.5_",
                embed_prefix: str = "HEST_breast_adata_",
                use_binned: bool = False,
                mask_zero_values: bool = True,
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load embeddings and labels data
    
    Args:
        paths: Dictionary containing paths to data files
        genes: List of genes to process
        feature_type: Type of feature (scLLM, peka, image_encoder)
        img_prefix: Prefix for image files
        embed_prefix: Prefix for embedding files
        use_binned: If True, use binned data; otherwise use raw data
        
    Returns:
        Tuple containing:
        - embeddings_dict: dictionary mapping gene names to their embeddings array
        - labels_dict: dictionary mapping gene names to their labels array
    """
    if use_binned:
        return _load_binned_data(paths, genes, feature_type, img_prefix, embed_prefix)
    else:
        return _load_raw_data(paths, genes, feature_type, img_prefix, embed_prefix, mask_zero_values)
