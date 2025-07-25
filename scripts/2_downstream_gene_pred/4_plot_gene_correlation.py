import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_gene_correlations(base_dir, output_dir, tissue):
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    # Read the three CSV files
    backbone_path = os.path.join(base_dir, "image_encoder/H-optimus-0_Bone_MLP/image_encoder_gene_level_raw_regression_scFoundation/gene_regression_results.csv")
    peka_path = os.path.join(base_dir, "histomil2/H-optimus-0_Bone_MLP/histomil2_gene_level_raw_regression_scFoundation/gene_regression_results.csv")
    peka_image_path = os.path.join(base_dir, "image_encoder+histomil2/H-optimus-0_Bone_MLP/image_encoder+histomil2_gene_level_raw_regression_scFoundation/gene_regression_results.csv")

    # Read CSV files
    backbone_df = pd.read_csv(backbone_path)
    peka_df = pd.read_csv(peka_path)
    peka_image_df = pd.read_csv(peka_image_path)

    # Sort genes based on PEKA+image results
    sorted_genes = peka_image_df.sort_values('pearson_correlation_mean')['gene'].values

    # Calculate mean PCC for each method
    backbone_mean = backbone_df['pearson_correlation_mean'].mean()
    peka_mean = peka_df['pearson_correlation_mean'].mean()
    peka_image_mean = peka_image_df['pearson_correlation_mean'].mean()

    # Create the plot with larger size for A4 paper
    plt.figure(figsize=(16, 10))

    # Plot points for each method with larger markers
    plt.scatter(range(len(sorted_genes)), 
               backbone_df.set_index('gene').loc[sorted_genes, 'pearson_correlation_mean'],
               color='gray', alpha=0.6, s=100, label=f'Backbone (mean PCC: {backbone_mean:.3f})')
    
    plt.scatter(range(len(sorted_genes)), 
               peka_df.set_index('gene').loc[sorted_genes, 'pearson_correlation_mean'],
               color='blue', alpha=0.6, s=100, label=f'PEKA (mean PCC: {peka_mean:.3f})')
    
    plt.scatter(range(len(sorted_genes)), 
               peka_image_df.set_index('gene').loc[sorted_genes, 'pearson_correlation_mean'],
               color='red', alpha=0.6, s=100, label=f'PEKA+Image (mean PCC: {peka_image_mean:.3f})')

    # Customize the plot
    plt.xticks(range(len(sorted_genes)), sorted_genes, rotation=90)
    plt.xlabel('Genes')
    plt.ylabel('Pearson Correlation Mean')
    plt.title(f'Gene-level Prediction Performance Comparison ({tissue})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot as PDF
    output_path = os.path.join(output_dir, f'gene_correlation_comparison_{tissue}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    base_dir = "/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/OUTPUT/1_downstream_hvg_pred/breast/breast_visium_26k/raw/H-optimus-0_Bone_MLP"
    #base_dir = "/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/OUTPUT/1_downstream_hvg_pred/other_cancer/kidney_in_hest/raw/H-optimus-0_Bone_MLP"
    #base_dir = "/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/OUTPUT/1_downstream_hvg_pred/other_cancer/liver_in_hest/raw/H-optimus-0_Bone_MLP"
    #base_dir = "/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/OUTPUT/1_downstream_hvg_pred/other_cancer/lung_in_hest/raw/H-optimus-0_Bone_MLP"
    output_dir = "/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/OUTPUT"
    #tissue = "Kidney"  # 设置组织类型
    tissue = "Breast"  # 设置组织类型
    plot_gene_correlations(base_dir, output_dir, tissue)
