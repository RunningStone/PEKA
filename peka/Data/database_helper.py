import pandas as pd
import shutil
import os

from peka import logger
from peka.Data.hest1k_helper import HEST1K_sub_database


def create_hest1k_sub_database_instance(csv_file_path, dataset_name, 
                                        data_root=None, 
                                        hest_storage_path=None, 
                                        copy_flag=False):
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    # Find the specified dataset row
    row = df[df['dataset_name'] == dataset_name]
    if row.empty:
        raise ValueError(f"Dataset name '{dataset_name}' not found in the CSV file.")
    # Convert DataFrame to Series
    row = row.iloc[0]
    # Log output
    logger.info(f" 🤖 read dataset info: {dataset_name}")
    # If copy_flag is True, modify corresponding fields
    if copy_flag:
        assert data_root is not None and hest_storage_path is not None, "data_root and hest_storage_path must be provided if copy_flag is True"
        row['dataset_storage_folder'] = data_root
        row['hest_loc'] = hest_storage_path
    else:
        assert row['dataset_storage_folder'] is not None and row['hest_loc'] is not None, "dataset_storage_folder and hest_loc must be provided if copy_flag is False"
    # Convert platform string to list
    platform_list = row['platform'].split() if pd.notna(row['platform']) else []
    # Replace _ with space
    platform_list = [p.replace("_", " ") for p in platform_list]
    # Create HEST1K_sub_database instance
    dataset = HEST1K_sub_database(
        dataset_name=row['dataset_name'],
        dataset_storage_folder=row['dataset_storage_folder'],
        hest_loc=row['hest_loc'],
        source_index_loc=row['source_index_loc'] if pd.notna(row['source_index_loc']) else None,
        organ=row['organ'],
        species=row['species'],
        oncotree_code=row['oncotree_code'] if pd.notna(row['oncotree_code']) else None,
        platform=platform_list,
        # image info
        patch_size=row['patch_size'],
        pixel_size=row['pixel_size'],
        # Preprocessing information
        align_gene_name_files=row['align_gene_name_files'] if pd.notna(row['align_gene_name_files']) else None,
        extract_patches_files=row['extract_patches_files'] if pd.notna(row['extract_patches_files']) else None,
        with_scLLM_embed_files=row['with_scLLM_embed_files'] if pd.notna(row['with_scLLM_embed_files']) else 0,
    )
    return dataset

def check_database_status(data_root, hest_storage_path, 
                          dataset_predefine: str = "histomil2_breast_datasets.csv"):
    target_csv_path = f"{data_root}/dataset_config.csv"
    copy_flag = False
    logger.info(f" 🤖 check database status in {data_root}")
    # 1. Check if dataset_config.csv exists
    if not os.path.exists(target_csv_path):
        logger.info(f" 🤖 dataset_config.csv not found in {data_root},copy it...")
        copy_flag = True
        proj_path = os.getenv("PROJECT_PATH")
        logger.info(f" 🤖 read source data root: {proj_path}\n    and use sub-datamodules data_root: {data_root}")
        source_csv_path = f"{proj_path}/HistoMIL2/hydra_zen/Configs/Datasets/{dataset_predefine}"
        # Copy the file from source to target
        shutil.copyfile(source_csv_path, target_csv_path)
        logger.info(f"🤖 Copied dataset configuration from {source_csv_path} \n    to {target_csv_path}")

    # 2. Read dataset_config.csv and generate dataset objects
    logger.info(f" 🤖 read dataset configuration and generate dataset objects")
    # Use for loop to iterate through dataset name list and create instances
    datasets_info = []
    target_datasets_infos = pd.read_csv(target_csv_path)
    dataset_name_list = target_datasets_infos['dataset_name'].to_list()
    for dataset_name in dataset_name_list:
        dataset = create_hest1k_sub_database_instance(
            csv_file_path=target_csv_path,
            dataset_name=dataset_name,
            data_root=data_root,
            hest_storage_path=hest_storage_path,
            copy_flag=copy_flag
        )
        datasets_info.append({dataset_name: dataset})

    # 3. Update dataset_config.csv
    if copy_flag:
        logger.info(f" 🤖 update dataset_config.csv")
        target_datasets_infos["dataset_storage_folder"] = [dataset.dataset_storage_folder for i in range(len(datasets_info))]
        target_datasets_infos["hest_loc"] = [dataset.hest_loc for i in range(len(datasets_info))]
        target_datasets_infos["align_gene_name_files"] = [dataset.align_gene_name_files for i in range(len(datasets_info))]
        target_datasets_infos["extract_patches_files"] = [dataset.extract_patches_files for i in range(len(datasets_info))]
        target_datasets_infos["with_scLLM_embed_files"] = [dataset.with_scLLM_embed_files for i in range(len(datasets_info))]
        target_datasets_infos.to_csv(target_csv_path, index=False)
        logger.info(f" 🤖 update dataset_config.csv for storage folder and preprocess info")
    logger.info(f" 🤖 check database status done")
    return datasets_info


def get_preprocess_status(data_root,dataset_name:str,scLLM_emb_name:str=None):
    """
    Check the preprocess status of dataset_name: align_gene_name, extract_patches, with_scLLM_embed
    Then update dataset_config.csv
    """
    logger.info(f" 🤖 checking dataset {dataset_name} preprocess status..")
    # 1. Check if dataset_config.csv exists
    target_csv_path = f"{data_root}/dataset_config.csv"
    if not os.path.exists(target_csv_path):
        raise ValueError(f"Dataset {dataset_name} not found in dataset_config.csv.")
    database = create_hest1k_sub_database_instance(
        csv_file_path=target_csv_path,
        dataset_name=dataset_name,)
    database_folder = database.dataset_storage_folder + "/" + dataset_name
    # 2. Check align_gene_name folder and the number of internal files, then update the count in dataset_index_loc file
    logger.info(f" 🤖 checking align_gene_name files..")
    align_gene_name_folder = database_folder + "/aligned_gene_name"
    align_adata_folder = database_folder + "/aligned_adata"
    if not os.path.exists(align_gene_name_folder):
        align_gene_name = None
    else:
        # Number of csv files    
        align_gene_name_csv = len([f for f in os.listdir(align_gene_name_folder) if f.endswith('.csv')])
        # Number of h5ad files
        align_gene_name_adata = len([f for f in os.listdir(align_adata_folder) if f.endswith('.h5ad')])
        if align_gene_name_csv != align_gene_name_adata:
            logger.error(f"Gene name alignment not consistent, please delete current file and re-run 1_generate_histomil2_datasets.sh .")
            align_gene_name = None
        else:
            align_gene_name = align_gene_name_csv
    logger.info(f" 🤖 align_gene_name files checked, {align_gene_name} files found.")
    # 3. Check extract_patches folder and the number of internal files, then update the count in dataset_index_loc file
    logger.info(f" 🤖 checking extract_patches files..")
    extract_patches_folder = database_folder + "/patches/"
    if not os.path.exists(extract_patches_folder):
        extract_patches = None
    else:
        extract_patches_h5 = len([f for f in os.listdir(extract_patches_folder) if f.endswith('.h5')])
        extract_patches_png = len([f for f in os.listdir(extract_patches_folder) if f.endswith('.png')])
        if extract_patches_h5 != extract_patches_png:
            logger.error(f" 🤖 extract_patches not consistent, please delete current file and re-run 1_generate_histomil2_datasets.sh .")
            extract_patches = None
        else:
            extract_patches = extract_patches_h5
    logger.info(f" 🤖 extract_patches files checked, {extract_patches} files found.")
    # 4. Check how many scLLM subfolders are in the with_scLLM_embed folder, and whether each folder contains all files.
    logger.info(f" 🤖 checking with_scLLM_embed methods..")
    scLLM_embed_folder = database_folder + "/scLLM_embed/"
    if not os.path.exists(scLLM_embed_folder):
        scLLM_embed = None
    else:
        # If scLLM_emb_name is None, select a subfolder
        scLLM_emb_subfolders = [f for f in os.listdir(scLLM_embed_folder) if os.path.isdir(os.path.join(scLLM_embed_folder, f))]
        if len(scLLM_emb_subfolders) == 0:
            scLLM_embed = None
        else:   
            scLLM_embed = len(scLLM_emb_subfolders)
    logger.info(f" 🤖 with_scLLM_embed checked, {scLLM_embed} methods folders found.")
    # 5. Update dataset_config.csv
    df = pd.read_csv(target_csv_path)
    df.loc[df['dataset_name'] == dataset_name, 'align_gene_name_files'] = align_gene_name
    df.loc[df['dataset_name'] == dataset_name, 'extract_patches_files'] = extract_patches
    df.loc[df['dataset_name'] == dataset_name, 'with_scLLM_embed_files'] = scLLM_embed
    df.to_csv(target_csv_path, index=False)
    logger.info(f" 🤖 dataset {dataset_name} preprocess status checked and updated.")