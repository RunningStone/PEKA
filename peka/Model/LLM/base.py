import os
import pandas as pd
import scanpy as sc
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

from peka import logger


def filter_zero_row(adata):
    if not sp.issparse(adata.X):
        sparse_matrix = sp.csr_matrix(adata.X)
    else:
        sparse_matrix = adata.X
    nonzero_per_row = np.diff(sparse_matrix.indptr)
    filter_flag = nonzero_per_row == 0
    is_zero_flag = np.any(filter_flag)

    zero_row_count = np.sum(filter_flag)
    retained_row_count = adata.shape[0] - zero_row_count
    print(f"total rows: {adata.shape}, zero rows: {zero_row_count}, retained rows: {retained_row_count}")
    return  filter_flag, is_zero_flag

class scLLM_QC_preprocess:
    """
    from align_adata_folder to get h5ad file to do qc
    """
    def __init__(self,data_root,dataset_name,scLLM_embedder_name:str,ckpt_name:str="default"):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.scLLM_embedder_name = scLLM_embedder_name

        # 数据库文件夹设置
        self.database_folder = self.data_root + "/" + self.dataset_name
        self.align_adata_folder = self.database_folder + "/aligned_adata/" 
        assert os.path.exists(self.align_adata_folder), f" 🤖 align_adata_folder: {self.align_adata_folder} not exists, please rerun 1_generate_peka_datasets.sh"
        self.ckpt_name = ckpt_name
        self.cleaned_ckpt_name = ckpt_name.replace("/","_") 
        self.scLLM_embedder_folder = self.database_folder + "/scLLM_embed/" + self.scLLM_embedder_name 
        self.paired_seq_folder = self.scLLM_embedder_folder +  f"/{self.cleaned_ckpt_name}/paired_seq/"
        self.embeddings_folder = self.scLLM_embedder_folder + f"/{self.cleaned_ckpt_name}/embeddings/"
        os.makedirs(self.scLLM_embedder_folder, exist_ok=True)
        os.makedirs(self.embeddings_folder, exist_ok=True)
        os.makedirs(self.paired_seq_folder, exist_ok=True)

        # 总的 index 文件，读取所有已经存在的database
        self.dataset_index_loc = self.database_folder + f"/{self.dataset_name}.csv"
        self.dataset_index_df = pd.read_csv(self.dataset_index_loc)

        # ckpt 
        self.pretrained_ckpt_dir = data_root + f"/Pretrained/{self.scLLM_embedder_name}"
        os.makedirs(self.pretrained_ckpt_dir, exist_ok=True)
        

    def _get_raw_h5ad_files(self):
        # h5ad 文件路径
        self.align_gene_name_adata_locs = [f for f in os.listdir(self.align_adata_folder) if f.endswith('.h5ad')]
        assert len(self.align_gene_name_adata_locs) == len(self.dataset_index_df), " 🤖 number of h5ad files and dataset index does not match."
        logger.info(f" 🤖 {len(self.align_gene_name_adata_locs)} h5ad files found in {self.align_adata_folder}\n    example: {self.align_gene_name_adata_locs[0]}")


    def model_related_qc_step(self,adata):
        # toy func, replace with your own QC step
        filter_flag = None
        logger.info(f" 🤖 base class model related QC step for {adata.filename} empty step..")
        return filter_flag

    def model_related_embed_infer_step(self,adata,**kwargs):
        # toy func, replace with your own embed infer step
        embed_array = None
        logger.info(f" 🤖 base class model related embed infer step for {adata.filename} empty step..")
        return embed_array

    def run(self,**kwargs):
        self.current_idx = 0
        self.not_processed_idx = []
        for idx in tqdm(range(len(self.dataset_index_df)),desc=" 🤖 processing subdataset"):
            self.current_idx = idx
            adata_loc = f"{self.align_adata_folder}/HEST_breast_adata_{idx}.h5ad"
            try:
                adata = sc.read_h5ad(adata_loc)
                adata, filter_flag = self.model_related_qc_step(adata) 
                # 保存处理后的adata
                adata.write(f"{self.paired_seq_folder}/HEST_breast_adata_{idx}.h5ad")
                adata_filtered = adata[~filter_flag].copy()
                embed_array = self.model_related_embed_infer_step(adata_filtered,**kwargs)
                np.save(f"{self.embeddings_folder}/HEST_breast_adata_{idx}.npy",embed_array)
            except Exception as e:
                logger.error(f"[Error]!!! processing {idx}th sample: {e}")
                self.not_processed_idx.append(idx)
                continue
            else:
                logger.info(f"Successfully processed {idx}th sample")
        if len(self.not_processed_idx) > 0:
            logger.warning(f" 🤖 {len(self.not_processed_idx)} samples not processed..")
            csv_not_processed = f"{self.embeddings_folder}/not_processed_idx.csv"
            data_dict = {"fname":[f"HEST_breast_adata_{idx}.h5ad" for idx in self.not_processed_idx],
                         "idx":self.not_processed_idx,
                         }
            pd.DataFrame(data_dict).to_csv(csv_not_processed,index=False)
            logger.info(f" 🤖 not processed samples info saved to {csv_not_processed}")

    def valid_check(self, ):
        """
        检查文件数量和一致性，并更新元数据文件
        """
        logger.info(f" 🤖 checking file count and consistency for {self.scLLM_embedder_name} {self.ckpt_name}..")
        try:
            # 获取各文件夹的文件数量
            index_file_count = len(self.dataset_index_df)
            aligned_file_count = len([f for f in os.listdir(self.align_adata_folder) if f.endswith('.h5ad')])
            paired_seq_file_count = len([f for f in os.listdir(self.paired_seq_folder) if f.endswith('.h5ad')])
            embeddings_file_count = len([f for f in os.listdir(self.embeddings_folder) if f.endswith('.npy')])

            # 检查文件数量是否一致
            if not (index_file_count == aligned_file_count == paired_seq_file_count == embeddings_file_count):
                logger.error(
                    f"File count mismatch: index({index_file_count}), aligned({aligned_file_count}), "
                    f"paired_seq({paired_seq_file_count}), embeddings({embeddings_file_count})"
                )
                return

            # 检查每个paired_seq文件是否有对应的embedding文件，并验证数据一致性
            for idx in range(index_file_count):
                paired_seq_file = f"{self.paired_seq_folder}/HEST_breast_adata_{idx}.h5ad"
                embedding_file = f"{self.embeddings_folder}/HEST_breast_adata_{idx}.npy"

                if not os.path.exists(paired_seq_file) or not os.path.exists(embedding_file):
                    logger.error(f"Missing files for idx {idx}: {paired_seq_file} or {embedding_file}")
                    return

                # 加载数据检查行数一致性
                adata = sc.read_h5ad(paired_seq_file)
                embedding = np.load(embedding_file)

                if np.sum(~adata.obs['filter_flag']) != embedding.shape[0]:
                    logger.error(
                        f"Mismatch in row count for idx {idx}: filter_flag({np.sum(adata.obs['filter_flag'])}) "
                        f"vs embedding rows({embedding.shape[0]})"
                    )
                    logger.error(f" 🤖 {idx}th sample not valid, please check the data with {idx}th sample..")
                    return

            # 记录每个embedding文件的列数（特征维度）和总行数（样本数）
            feature_dim = None
            sample_nb = 0
            for embedding_file in os.listdir(self.embeddings_folder):
                if embedding_file.endswith('.npy'):
                    embedding = np.load(f"{self.embeddings_folder}/{embedding_file}")
                    if feature_dim is None:
                        feature_dim = embedding.shape[1]
                    elif feature_dim != embedding.shape[1]:
                        logger.error(f"Feature dimension mismatch in file: {embedding_file}")
                        return
                    sample_nb += embedding.shape[0]

            # 更新或创建meta.csv文件
            meta_csv_path = f"{self.scLLM_embedder_folder}/meta.csv"
            meta_data = {
                "ckpt_name": [self.cleaned_ckpt_name],
                "ckpt_root": [self.pretrained_ckpt_dir],
                "sample_nb": [sample_nb],
                "feature_dim": [feature_dim]
            }

            if os.path.exists(meta_csv_path):
                meta_df = pd.read_csv(meta_csv_path)
                meta_df = meta_df[meta_df['ckpt_name'] != self.cleaned_ckpt_name]  # 删除旧条目
                meta_df = pd.concat([meta_df, pd.DataFrame(meta_data)], ignore_index=True)
            else:
                meta_df = pd.DataFrame(meta_data)
            logger.info(f" 🤖 update meta data info: {meta_df}")
            meta_df.to_csv(meta_csv_path, index=False)
            logger.info(f"Meta information updated in {meta_csv_path}")

        except Exception as e:
            logger.error(f"Error in check_and_update_metadata: {e}")
