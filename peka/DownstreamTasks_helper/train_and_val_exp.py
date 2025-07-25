"""

Train and evaluate a single gene regressor. by following the steps from HEST1k paper

PCA(256) and Ridge Regression
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import glob
import sys
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

class GeneRegressor:
    def __init__(self, input_dim, n_components=256, alpha=1.0):
        """Initialize GeneRegressor with PCA and Ridge Regression.
        
        Args:
            input_dim: Input feature dimension
            n_components: Number of PCA components (default: 256)
            alpha: Ridge regression regularization strength (default: 1.0)
        """
        self.n_components = n_components
        self.alpha = alpha
        self.pca = PCA(n_components=n_components)
        self.ridge = Ridge(alpha=alpha)
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit PCA and Ridge regression on training data.
        
        Args:
            X: Input features [N, input_dim]
            y: Target values [N]
        """
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
            
        # Fit PCA followed by Ridge regression
        X_pca = self.pca.fit_transform(X)
        self.ridge.fit(X_pca, y)
        self.is_fitted = True
        return self
        
    def inference(self, features, batch_size=None):
        """Perform inference using fitted PCA and Ridge regression.
        
        Args:
            features: Input features array or tensor
            batch_size: Not used, kept for API compatibility
            
        Returns:
            Predicted values as numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Convert to numpy if tensor
        if torch.is_tensor(features):
            features = features.cpu().numpy()
            
        # Transform and predict
        features_pca = self.pca.transform(features)
        predictions = self.ridge.predict(features_pca)
        
        return predictions

def train_regressor(X_train, y_train, X_val, y_val, input_dim, device, epochs=200):
    """Train a single gene regressor"""
    model = GeneRegressor(input_dim)
    model.fit(X_train, y_train)
    
    best_correlation = float('-inf')
    best_mse = float('inf')
    best_model = None
    patience = epochs // 10
    patience_counter = 0
    
    best_model = model
    val_pred = model.inference(X_val)
    #current_correlation, _ = pearsonr(y_val, val_pred)
    current_mse = mean_squared_error(y_val, val_pred)
    # Load best model
    return best_model

def evaluate_regressor(model, X_test, y_test, device):
    """Evaluate regressor performance"""
    y_pred = model.inference(X_test)
        
    metrics = calculate_metrics(y_test, y_pred)
    return metrics

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various regression metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Pearson Correlation
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # Cosine Similarity
    cos_sim = 1 - cosine(y_true, y_pred)
    
    # KL Divergence
    # Normalize predictions and true values to make them proper distributions
    y_true_norm = y_true - np.min(y_true)
    y_true_norm = y_true_norm / np.sum(y_true_norm)
    y_pred_norm = y_pred - np.min(y_pred)
    y_pred_norm = y_pred_norm / np.sum(y_pred_norm)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    y_true_norm = y_true_norm + epsilon
    y_pred_norm = y_pred_norm + epsilon
    
    kl_div = scipy.stats.entropy(y_true_norm, y_pred_norm)
    
    return {
        'mse': mse,
        'pearson_correlation': pearson_corr,
        'cosine_similarity': cos_sim,
        'kl_divergence': kl_div
    }

def plot_scatter(y_true, y_pred, save_path):
    """Plot scatter plot of true vs predicted values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig(save_path)
    plt.close()

def plot_gene_correlations(results_df: pd.DataFrame, output_dir: str):
    """Plot gene correlations as a scatter plot
    
    Args:
        results_df: DataFrame containing gene results
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by correlation for better visualization
    results_df = results_df.sort_values('pearson_correlation', ascending=True)
    
    # Create scatter plot
    plt.scatter(range(len(results_df)), results_df['pearson_correlation'], alpha=0.6)
    
    # Add gene names as annotations
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.annotate(row['gene'], 
                    (i, row['pearson_correlation']),
                    xytext=(0, 5), 
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    rotation=45,
                    fontsize=8)
    
    # Customize plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Genes')
    plt.ylabel('Pearson Correlation')
    plt.title('Gene Expression Prediction Performance')
    
    # Add mean correlation line
    mean_corr = results_df['pearson_correlation'].mean()
    plt.axhline(y=mean_corr, color='r', linestyle='--', alpha=0.5)
    plt.text(len(results_df)/2, mean_corr, f'Mean Correlation: {mean_corr:.3f}', 
             ha='center', va='bottom', color='r')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'gene_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gene correlation plot saved to {output_dir}/gene_correlations.png")


def train_and_val_step(embeddings,labels,
                    output_dir,gene, 
                    config,
                    with_independent_test_set:bool=False,epochs=200,):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, 
                                                        test_size=0.2, random_state=2025)
    if with_independent_test_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.2, random_state=2025)
    else:
        X_val = X_test
        y_val = y_test
    
    print(f"\nTraining with {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train regressor
    model = train_regressor(X_train, y_train, X_val, y_val,
                            input_dim=embeddings.shape[1],
                            device=device,
                            epochs=epochs)
    
    # Evaluate
    metrics = evaluate_regressor(model, X_test, y_test, device)
    print(f"Performance - MSE: {metrics['mse']:.3f}\n Pearson Correlation: {metrics['pearson_correlation']:.3f}\n Cosine Similarity: {metrics['cosine_similarity']:.3f}\n KL Divergence: {metrics['kl_divergence']:.3f}\n")
    
    # Save results
    ckpt_path = os.path.join(output_dir, 'ckpt', f'{gene}_{config["embedder_name"]}_mse_{metrics["mse"]:.3f}.pt')
    #torch.save(model.state_dict(), ckpt_path)
    
    plot_path = os.path.join(output_dir, 'plots', f'{gene}_scatter_plot.png')
    plot_scatter(y_test, model.inference(X_test), plot_path)
    
    results = {
            'gene': gene,
            'mse': metrics['mse'],
            'pearson_correlation': metrics['pearson_correlation'],
            'cosine_similarity': metrics['cosine_similarity'],
            'kl_divergence': metrics['kl_divergence']
        }
    return results


def train_and_val_step_KFold(embeddings,labels,
                    output_dir,gene, 
                    config,with_independent_test_set=True,
                    Ksplit=5,epochs=200,):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize 5-fold cross validation
    k_fold = KFold(n_splits=Ksplit, shuffle=True, random_state=2025)
    
    # Initialize lists to store metrics for each fold
    fold_metrics = {
        'mse': [],
        'pearson_correlation': [],
        'cosine_similarity': [],
        'kl_divergence': []
    }
    
    print(f"\nPerforming 5-fold cross validation with {len(embeddings)} total samples")
    
    # Perform k-fold cross validation
    for fold, (train_idx, test_idx) in enumerate(k_fold.split(embeddings)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        if with_independent_test_set:
            # Further split training data to get validation set
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.2, random_state=2025)
        else:
            X_val = X_test
            y_val = y_test
        
        print(f"\nFold {fold + 1}/5:")
        print(f"Training with {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train regressor
        model = train_regressor(X_train, y_train, X_val, y_val,
                              input_dim=embeddings.shape[1],
                              device=device,
                              epochs=epochs)
        
        # Evaluate
        metrics = evaluate_regressor(model, X_test, y_test, device)
        print(f"Fold {fold + 1} Performance - MSE: {metrics['mse']:.3f}, Pearson: {metrics['pearson_correlation']:.3f}, Cosine: {metrics['cosine_similarity']:.3f}, KL: {metrics['kl_divergence']:.3f}")
        
        # Store metrics for this fold
        for metric_name in fold_metrics.keys():
            fold_metrics[metric_name].append(metrics[metric_name])
        
        # Save model checkpoint for each fold
        ckpt_path = os.path.join(output_dir, 'ckpt', f'{gene}_{config["embedder_name"]}_fold{fold+1}_mse_{metrics["mse"]:.3f}.pt')
        #torch.save(model.state_dict(), ckpt_path)
        
        # Save scatter plot for each fold
        plot_path = os.path.join(output_dir, 'plots', f'{gene}_fold{fold+1}_scatter_plot.png')
        plot_scatter(y_test, model.inference(X_test), plot_path)
    
    # Calculate mean and std of metrics across folds
    results = {
        'gene': gene,
    }
    
    print("\nOverall Performance across 5 folds:")
    for metric_name in fold_metrics.keys():
        mean_value = np.mean(fold_metrics[metric_name])
        std_value = np.std(fold_metrics[metric_name])
        results[f'{metric_name}'] = mean_value
        results[f'{metric_name}_mean'] = mean_value
        results[f'{metric_name}_std'] = std_value
        print(f"{metric_name}: {mean_value:.3f} ± {std_value:.3f}")
    
    return results