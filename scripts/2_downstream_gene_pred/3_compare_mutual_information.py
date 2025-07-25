
import os
import numpy as np
import scanpy as sc
import h5py
from typing import Dict, List, Tuple
from sklearn.feature_selection import mutual_info_regression

import sys
sys.path.append("/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/HistoMIL2")

from histomil2.DownstreamTasks_helper.gene_expression_prediction import get_dataset_paths
from histomil2.DownstreamTasks_helper.gene_expression_prediction import (
    get_dataset_paths,
    _filter_from_image_and_seq_qc,
    feature_type_settings
)

import numpy as np
from scipy.special import digamma
import faiss
from bisect import bisect

def nn_faiss(x, k):
    """使用FAISS计算每个点的最近邻
    
    参数:
    x : ndarray, shape (n_samples, n_features)
        输入数据点集
    k : int
        为每个点搜索的最近邻数量
        
    返回:
    d : ndarray, shape (n_samples, k)
        每个点到其k个最近邻的距离
    """
    # 确保数据类型是float32(FAISS要求)
    x = x.astype(np.float32)
    
    # 创建FAISS索引
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    
    # 搜索k+1个最近邻(包括点本身)
    distances, _ = index.search(x, k + 1)
    return distances

def estimate_mi_continuous(X, Y, n_neighbors=5, max_search_neighbors=100):
    """估计两个连续变量集合之间的互信息
    
    参数:
    X : ndarray, shape (n_samples, n_features_x)
        第一个连续变量集合
    Y : ndarray, shape (n_samples, n_features_y)
        第二个连续变量集合
    n_neighbors : int
        用于MI估计的最近邻数量
    max_search_neighbors : int
        FAISS搜索时的最大近邻数量，用于平衡精度和速度
        
    返回:
    float : 估计的互信息值
    """
    n_samples = X.shape[0]
    if n_samples != Y.shape[0]:
        raise ValueError(f"样本数量必须相同 {Y.shape[0]} {n_samples}")
    
    # 在联合空间中找到k近邻
    XY = np.hstack([X, Y]).astype(np.float32)
    dist_xy = nn_faiss(XY, n_neighbors)
    radius_xy = np.nextafter(dist_xy[:, -1], 0)
    
    # 在X空间中找到半径内的点数
    dist_x = nn_faiss(X.astype(np.float32), max_search_neighbors)
    nx = np.array([bisect(row, radius) for radius, row in zip(radius_xy, dist_x)])
    
    # 在Y空间中找到半径内的点数
    dist_y = nn_faiss(Y.astype(np.float32), max_search_neighbors)
    ny = np.array([bisect(row, radius) for radius, row in zip(radius_xy, dist_y)])
    
    # 计算互信息估计
    mi = (digamma(n_neighbors) + digamma(n_samples) - 
          np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    
    # 处理异常值
    if mi > 100:  # 如果互信息估计值过高，可能是近似计算出现问题
        mi = 0
    
    return max(0, mi)

def normalize_mi_continuous(mi, X, Y, n_neighbors=5):
    """计算标准化的互信息
    
    参数:
    mi : float
        已计算的互信息值
    X : ndarray
        第一个变量集合
    Y : ndarray
        第二个变量集合
    n_neighbors : int
        用于熵估计的近邻数量
        
    返回:
    float : 标准化的互信息值 (0到1之间)
    """
    n_samples = len(X)
    
    # 估计X的熵
    dist_x = nn_faiss(X.astype(np.float32), n_neighbors)
    h_x = (np.log(n_samples) - digamma(n_neighbors) + 
           digamma(n_samples) + np.mean(np.log(dist_x[:, -1])) * X.shape[1])
    
    # 估计Y的熵
    dist_y = nn_faiss(Y.astype(np.float32), n_neighbors)
    h_y = (np.log(n_samples) - digamma(n_neighbors) + 
           digamma(n_samples) + np.mean(np.log(dist_y[:, -1])) * Y.shape[1])
    
    # 计算标准化互信息
    nmi = mi / min(h_x, h_y)
    return max(0, min(1, nmi))  # 确保在[0,1]范围内 

def get_model_paths(dataset_root: str) -> Dict[str, List[str]]:
    """获取数据集中所有可用模型的路径
    
    Args:
        dataset_root: 数据集根目录路径
        
    Returns:
        包含histomil2和patches模型路径的字典
    """
    model_paths = {
        'histomil2': [],
        'patches': []
    }
    
    # 获取histomil2模型路径
    histomil2_dir = os.path.join(dataset_root, 'histomil2_embed')
    if os.path.exists(histomil2_dir):
        for model_name in os.listdir(histomil2_dir):
            model_path = os.path.join(histomil2_dir, model_name)
            if os.path.isdir(model_path) and len(os.listdir(model_path)) > 0:
                model_paths['histomil2'].append(model_name)
    
    # 获取patches模型路径
    patches_dir = os.path.join(dataset_root, 'patches_embed')
    if os.path.exists(patches_dir):
        for model_name in os.listdir(patches_dir):
            model_path = os.path.join(patches_dir, model_name)
            if os.path.isdir(model_path) and len(os.listdir(model_path)) > 0:
                model_paths['patches'].append(model_name)
    
    return model_paths

def load_all_embeddings(seq_files: List[str], 
                       feature_type: str,
                       img_prefix: str,
                       embed_prefix: str,
                       paths: Dict[str, str]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """加载所有的嵌入文件并合并
    
    Args:
        seq_files: 序列文件列表
        feature_type: 特征类型
        img_prefix: 图像文件前缀
        embed_prefix: 嵌入文件前缀
        paths: 路径字典
        
    Returns:
        embeddings_dict: 每个文件的嵌入字典
        barcodes_dict: 每个文件的条形码字典
    """
    embeddings_dict = {}
    barcodes_dict = {}
    
    for i, seq_file in enumerate(seq_files):
        try:
            print(f"\nProcessing index {i}")
            _, embeddings, _, _, valid_barcodes = _filter_from_image_and_seq_qc(
                seq_file=seq_file,
                feature_type=feature_type,
                img_prefix=img_prefix,
                embed_prefix=embed_prefix,
                paths=paths
            )
            file_key = os.path.basename(seq_file)
            embeddings_dict[file_key] = embeddings
            barcodes_dict[file_key] = valid_barcodes
            
        except Exception as e:
            print(f"Error processing file {seq_file}: {str(e)}")
            continue
    
    if not embeddings_dict:
        raise ValueError("No valid embeddings found")
        
    return embeddings_dict, barcodes_dict

def calculate_and_save_mi(raw_seq_embed: np.ndarray, 
                         model_embed: np.ndarray, 
                         model_type: str,
                         model_name: str,
                         results: List[Dict],
                         n_neighbors: int = 5,
                         max_search_neighbors: int = 100):
    """计算互信息并保存结果
    
    Args:
        raw_seq_embed: 原始序列嵌入
        model_embed: 模型嵌入
        model_type: 模型类型 (histomil2 或 patches)
        model_name: 模型名称
        results: 结果列表
        n_neighbors: 用于MI估计的近邻数量
        max_search_neighbors: FAISS搜索的最大近邻数量
    """
    mi = estimate_mi_continuous(raw_seq_embed, model_embed, 
                              n_neighbors=n_neighbors,
                              max_search_neighbors=max_search_neighbors)
    nmi = normalize_mi_continuous(mi, raw_seq_embed, model_embed, 
                                n_neighbors=n_neighbors)
    
    results.append({
        'model_type': model_type,
        'model_name': model_name,
        'mi': mi,
        'nmi': nmi
    })

def main(dataset_root: str,
         embedder_name: str = "scFoundation",
         checkpoint_name: str = "default_model",
         output_file: str = "mi_results.csv"):
    """计算数据集中所有模型与原始序列之间的互信息
    
    Args:
        dataset_root: 数据集根目录路径
        embedder_name: scLLM模型名称
        checkpoint_name: 检查点名称
        output_file: 输出文件名
    """
    # 获取原始序列路径
    seq_path = os.path.join(dataset_root, 'scLLM_embed', embedder_name, 
                           checkpoint_name, 'paired_seq')
    
    # 获取所有序列文件
    seq_files = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith('.h5ad')])
    if not seq_files:
        raise ValueError(f"No .h5ad files found in {seq_path}")

    print(f"Found {len(seq_files)} sequence files")
    
    # 获取所有可用模型路径
    model_paths = get_model_paths(dataset_root)
    
    # 准备paths字典用于_filter_from_image_and_seq_qc函数
    base_paths = {
        'dataset_root': dataset_root,
        'ref_dir': os.path.join(dataset_root, 'scLLM_embed'),
        'img_dir': os.path.join(dataset_root, 'patches'),
        'bin_dir': os.path.join(dataset_root, 'binned_adata'),
        'seq_path': seq_path,
        'img_path': os.path.join(dataset_root, 'patches'),
        'embed_dir': os.path.join(dataset_root, 'scLLM_embed', embedder_name, checkpoint_name),
        'embed_path': os.path.join(dataset_root, 'scLLM_embed', embedder_name, checkpoint_name, 'embeddings')
    }
    
    # 加载所有原始序列数据
    print(f"\nLoading all sequence embeddings...")
    raw_seq_embeds, raw_barcodes = load_all_embeddings(
        seq_files=seq_files,
        feature_type="scLLM",
        img_prefix="patch_224_0.5_",
        embed_prefix="HEST_breast_adata_",
        paths=base_paths
    )
    print(f"Loaded embeddings for {len(raw_seq_embeds)} files")
    
    # 存储结果
    results = []
    
    # 计算histomil2模型的互信息
    for model_name in model_paths['histomil2']:
        print(f"\nProcessing histomil2 model: {model_name}")
        paths = base_paths.copy()
        paths['embed_dir'] = os.path.join(dataset_root, 'histomil2_embed', model_name)
        paths['embed_path'] = paths['embed_dir']
        
        try:
            model_embeds, model_barcodes = load_all_embeddings(
                seq_files=seq_files,
                feature_type="histomil2",
                img_prefix="patch_224_0.5_",
                embed_prefix="HEST_breast_adata_",
                paths=paths
            )
            
            # 对每个文件分别处理
            all_raw_embeds = []
            all_model_embeds = []
            
            for file_key in raw_seq_embeds.keys():
                if file_key not in model_embeds:
                    print(f"Warning: {file_key} not found in model embeddings")
                    continue
                
                # 获取当前文件的条形码和嵌入
                raw_bc = raw_barcodes[file_key]
                model_bc = model_barcodes[file_key]
                
                # 找到共同的条形码
                common_bc = set(raw_bc) & set(model_bc)
                if not common_bc:
                    print(f"Warning: No common barcodes found for {file_key}")
                    continue
                
                # 创建条形码到索引的映射
                raw_bc_dict = {bc: i for i, bc in enumerate(raw_bc)}
                model_bc_dict = {bc: i for i, bc in enumerate(model_bc)}
                
                # 获取共同条形码的索引
                raw_indices = [raw_bc_dict[bc] for bc in common_bc]
                model_indices = [model_bc_dict[bc] for bc in common_bc]
                
                # 添加到总的嵌入列表中
                all_raw_embeds.append(raw_seq_embeds[file_key][raw_indices])
                all_model_embeds.append(model_embeds[file_key][model_indices])
            
            if not all_raw_embeds:
                raise ValueError("No valid embeddings found after filtering")
            
            # 合并所有文件的嵌入
            final_raw_embed = np.vstack(all_raw_embeds)
            final_model_embed = np.vstack(all_model_embeds)
            
            print(f"Using total {final_raw_embed.shape[0]} samples for MI calculation")
            calculate_and_save_mi(final_raw_embed, final_model_embed, 
                                'histomil2', model_name, results)
        except Exception as e:
            print(f"Error processing histomil2 model {model_name}: {str(e)}")
    
    # 计算patches模型的互信息
    for model_name in model_paths['patches']:
        print(f"\nProcessing patches model: {model_name}")
        paths = base_paths.copy()
        paths['embed_dir'] = os.path.join(dataset_root, 'patches_embed')
        paths['embed_path'] = os.path.join(paths['embed_dir'], model_name)
        
        try:
            model_embeds, model_barcodes = load_all_embeddings(
                seq_files=seq_files,
                feature_type="image_encoder",
                img_prefix="patch_224_0.5_",
                embed_prefix="patch_224_0.5_",
                paths=paths
            )
            
            # 对每个文件分别处理
            all_raw_embeds = []
            all_model_embeds = []
            
            for file_key in raw_seq_embeds.keys():
                if file_key not in model_embeds:
                    print(f"Warning: {file_key} not found in model embeddings")
                    continue
                
                # 获取当前文件的条形码和嵌入
                raw_bc = raw_barcodes[file_key]
                model_bc = model_barcodes[file_key]
                
                # 找到共同的条形码
                common_bc = set(raw_bc) & set(model_bc)
                if not common_bc:
                    print(f"Warning: No common barcodes found for {file_key}")
                    continue
                
                # 创建条形码到索引的映射
                raw_bc_dict = {bc: i for i, bc in enumerate(raw_bc)}
                model_bc_dict = {bc: i for i, bc in enumerate(model_bc)}
                
                # 获取共同条形码的索引
                raw_indices = [raw_bc_dict[bc] for bc in common_bc]
                model_indices = [model_bc_dict[bc] for bc in common_bc]
                
                # 添加到总的嵌入列表中
                all_raw_embeds.append(raw_seq_embeds[file_key][raw_indices])
                all_model_embeds.append(model_embeds[file_key][model_indices])
            
            if not all_raw_embeds:
                raise ValueError("No valid embeddings found after filtering")
            
            # 合并所有文件的嵌入
            final_raw_embed = np.vstack(all_raw_embeds)
            final_model_embed = np.vstack(all_model_embeds)
            
            print(f"Using total {final_raw_embed.shape[0]} samples for MI calculation")
            calculate_and_save_mi(final_raw_embed, final_model_embed, 
                                'patches', model_name, results)
        except Exception as e:
            print(f"Error processing patches model {model_name}: {str(e)}")
    
    # 将结果保存为CSV文件
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print("\nMutual Information Results:")
    print(df.to_string())

if __name__ == "__main__":
    # Example usage
    main(
        #dataset_root="/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/HistoMIL2/DATA/breast/breast_visium_26k",
        #dataset_root="/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/HistoMIL2/DATA/other_cancer/kidney_in_hest",
        #dataset_root="/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/HistoMIL2/DATA/other_cancer/liver_in_hest",
        dataset_root="/home/pan/Experiments/EXPs/2025_HistoMIL2_workspace/HistoMIL2/DATA/other_cancer/lung_in_hest",
        embedder_name="scFoundation",
        checkpoint_name="default_model",
        output_file="mi_results.csv"
    )
