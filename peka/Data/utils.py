import scanpy as sc
import pandas as pd
from scipy.sparse import issparse
import os
import json
import numpy as np
from peka import logger

def aggregate_duplicate_genes(adata):
    """
    Aggregate duplicate genes by taking their mean values.
    Returns a new AnnData object with unique genes.
    """
    # Get expression matrix
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    
    # Create DataFrame with gene names as columns
    df = pd.DataFrame(X, columns=adata.var_names)
    
    # Use transposed groupby to avoid deprecation warning
    df_agg = df.T.groupby(level=0).mean().T
    
    # Create new AnnData object with aggregated values
    adata_unique = sc.AnnData(
        X=df_agg.values,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=df_agg.columns)
    )
    
    return adata_unique

def generate_binned_adata(adata_with_hvg, bin_nb):
    def _get_obs_rep(adata, layer=None):
        """Get observation representation from adata."""
        return adata.layers[layer] if layer is not None else adata.X

    def _digitize(x, bins):
        """Digitize values into bins."""
        return np.digitize(x, bins)

    # Initialize binned data storage
    n_bins = bin_nb
    binned_rows = []
    bin_edges = []
    
    # Get expression data
    layer_data = _get_obs_rep(adata_with_hvg, layer=None)
    layer_data = layer_data.A if issparse(layer_data) else layer_data
    
    if layer_data.min() < 0:
        raise ValueError(f"Assuming non-negative data, but got min value {layer_data.min()}.")
    
    # Process each cell's expression data
    for row_idx, row in enumerate(layer_data):
        if row.max() == 0:
            logger.warning(
                f"Row {row_idx} contains all zeros. Please make sure this is expected."
            )
            binned_rows.append(np.zeros(row.shape[0], dtype=np.int64))
            bin_edges.append(np.array([0] * n_bins))
            continue
        
        # Create binned row for this cell
        binned_row = np.zeros(row.shape[0], dtype=np.int64)
        
        # Find non-zero values
        non_zero_mask = row > 0
        if np.any(non_zero_mask):
            non_zero_values = row[non_zero_mask]
            # Calculate bins using non-zero values
            bins = np.quantile(non_zero_values, np.linspace(0, 1, n_bins - 1))
            # Digitize non-zero values
            non_zero_digits = _digitize(non_zero_values, bins)
            # Verify binning results
            assert non_zero_digits.min() >= 1
            assert non_zero_digits.max() <= n_bins - 1
            # Assign binned values back to original positions
            binned_row[non_zero_mask] = non_zero_digits
        
        binned_rows.append(binned_row)
        bin_edges.append(np.concatenate([[0], bins]) if np.any(non_zero_mask) else np.array([0] * n_bins))
    
    # Stack all binned rows and create new AnnData
    binned_matrix = np.stack(binned_rows)
    adata_bin = sc.AnnData(
        X=binned_matrix,
        obs=adata_with_hvg.obs.copy(),
        var=adata_with_hvg.var.copy()
    )
    
    # Store bin edges
    adata_bin.obsm["bin_edges"] = np.stack(bin_edges)
    
    return adata_bin    

def calculate_hvg(adata, batch_key, output_dir,
                n_top_hvg: int=50, hvg_flavor:'str'='seurat', filename=None):
    """Calculate highly variable genes with robust preprocessing.
    
    Args:
        adata: AnnData object
        batch_key: Key for batch information in adata.obs
        output_dir: Directory to save the HVG list
        n_top_hvg: Number of highly variable genes to select
        hvg_flavor: Method for identifying highly variable genes
        filename: Optional filename for the output JSON
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Convert to dense if sparse and handle special values
    if issparse(adata.X):
        adata.X = adata.X.toarray()
    
    # Replace inf/nan with 0 and ensure float32
    adata.X = np.nan_to_num(adata.X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Basic preprocessing
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # Calculate highly variable genes
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_hvg,
            batch_key=batch_key,
            flavor=hvg_flavor,
            subset=False
        )
    except Exception as e:
        print(f"Warning: Error in calculating HVG: {str(e)}")
        print("Falling back to basic variance calculation...")
        # Fallback to basic variance calculation if scanpy method fails
        gene_vars = np.var(adata.X, axis=0)
        top_var_idx = np.argsort(gene_vars)[-n_top_hvg:]
        adata.var['highly_variable'] = False
        adata.var.loc[adata.var_names[top_var_idx], 'highly_variable'] = True
    
    # Get the names and indices of highly variable genes
    hvg_genes = adata.var_names[adata.var.highly_variable].tolist()
    
    # Save HVG list to JSON if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = "hvg_genes.json"
        hvg_json_path = os.path.join(output_dir, filename)
        with open(hvg_json_path, 'w') as f:
            json.dump({"genes": hvg_genes}, f, indent=2)
        print(f"Saved HVG list to: {hvg_json_path}")
    
    return len(hvg_genes), hvg_genes, adata.var.highly_variable