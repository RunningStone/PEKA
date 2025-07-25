import pandas as pd
import os
import shutil
import sys
from hest import load_hest
import re
from tqdm import tqdm
from dataclasses import dataclass
from biomart import BiomartServer
from pathlib import Path
import scanpy as sc
import h5py
import torch
from torchvision import transforms
import numpy as np
import time

from peka import logger
HEST_DB_INDEX_PREFIX = "HEST_v1_1_0.csv"


###########################################################################################
# HEST database index operations
###########################################################################################
def get_latest_version_hest_index(hest_loc, prefix="HEST_", extension=".csv"):
    logger.info(f"🤖 searching for latest version of HEST index in {hest_loc}")
    version_files = [f for f in os.listdir(hest_loc) if f.startswith(prefix) and f.endswith(extension)]
    
    if not version_files:
        print(f"No CSV files with the hest database index found in the folder.")
        return None, None
    
    # Extract version numbers and find the latest
    version_files.sort(key=lambda f: [int(num) for num in re.findall(r'\d+', f)], reverse=True)
    latest_file = version_files[0]
    logger.info(f"Latest version file found: {latest_file}")
    return os.path.join(hest_loc, latest_file), latest_file

def get_hest_db_index(hest_loc, hest_db_index_prefix=HEST_DB_INDEX_PREFIX):
    file_loc, file_name = get_latest_version_hest_index(hest_loc, prefix=hest_db_index_prefix)
    if file_loc is None:
        # Copy from repo using prefix
        logger.info(f"Copying {hest_db_index_prefix} from repo to {hest_loc}")
        code_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
        hest_db_index_file_copy_loc = f"{code_path}/support_files/{hest_db_index_prefix}"
        shutil.copy(hest_db_index_file_copy_loc, hest_loc)
        file_loc = os.path.join(hest_loc, hest_db_index_prefix)
        file_name = hest_db_index_prefix
    return file_loc, file_name


###########################################################################################
# Construct sub database from HEST database
###########################################################################################

@dataclass
class HEST1K_sub_database:
    """
    HEST1K_sub_database is a class to define a sub database from HEST1k database.
    """
    # dataset info
    dataset_name:str
    dataset_storage_folder:str # where to save all custom dataset folders

    # hest info
    hest_loc:str
    hest_db_index_prefix:str = HEST_DB_INDEX_PREFIX

    # filter info
    source_index_loc:str = None # if none, use hest_db_index file to get index file

    organ:str = None
    species:str = "Homo sapiens"
    oncotree_code:str = None
    platform:list = None # ["Xenium", "Visium", "ST"]

    # patch info
    patch_size:int = 224
    pixel_size:int = 0.5

    # preprocess info
    align_gene_name_files:int = None
    extract_patches_files:int = None
    with_scLLM_embed_files:int = 0 # how many scllm embed saved



def explore_local_hest_db(hest_loc):
    logger.info(f"📃 exploring local HEST1k database in {hest_loc}")
    # 1- Read the whole hest dataset
    hest_data = load_hest(hest_loc)

    # 3- Access objects and check their format
    for st in hest_data:
        # ST (adata):
        adata = st.adata
        print('\n* 📃 Scanpy adata:')
        print(adata)

        # WSI:
        wsi = st.wsi
        logger.info('\n* 📃 WSI data:')
        logger.info(wsi)

        # check the format of adata
        logger.info(adata.var.columns)
        logger.info(adata.obs.columns)
        logger.info(adata.X.shape)
    
    # 3- show example of adata
    logger.info(f"🤖 show example of adata:")
    logger.info(hest_data[70].adata.var)

def explore_constructed_db(index_csv):
    logger.info(f"🤖 exploring constructed HEST1k database in {index_csv}")
    filtered_df = pd.read_csv(index_csv)
    # check min gene number
    gene_nb = filtered_df["nb_genes"].to_list()
    gene_nb = [int(s) for s in gene_nb]


    spots_nb =filtered_df["spots_under_tissue"].to_list()
    spots_nb = [int(s) for s in spots_nb]

    logger.info(filtered_df["st_technology"].value_counts())
    logger.info(f" total spots: {sum(spots_nb)}")
    logger.info(f"min gene number: {min(gene_nb)}")
    logger.info(f"max gene number: {max(gene_nb)}")
    logger.info(filtered_df["dataset_title"].value_counts())
    logger.info(filtered_df["tissue"].value_counts())
    # check unique and count
    logger.info(filtered_df["oncotree_code"].value_counts())
    # check unique and count
    logger.info(filtered_df["disease_state"].value_counts())
    logger.info(filtered_df["disease_comment"].value_counts())

def construct_hest1k_tissue_index(hest_loc, data_root, source_index_csv=None, organ="Breast", species="Homo sapiens" , target_oncotree_code=None):
    logger.info(f"🤖 constructing HEST1k tissue index for {organ} {species} in {data_root}")
    if source_index_csv is None:
        hest_db_index_loc, hest_db_index_name = get_hest_db_index(hest_loc)
        source_name = "hest"
    else:
        hest_db_index_loc = source_index_csv
        source_name = str(Path(source_index_csv).stem)

    meta_df = pd.read_csv(hest_db_index_loc)
    mask = (meta_df['species'] == species) & (meta_df['organ'] == organ)
    fname = organ
    if target_oncotree_code is not None:
        # Ex: IDC  invasive ductal carcinoma 
        mask = mask & (meta_df['oncotree_code'] == target_oncotree_code)
        fname = fname + f"_on_{target_oncotree_code}"
        logger.info(f"🤖 filtering by oncotree code: {target_oncotree_code}")
    filtered_df = meta_df[mask]
    logger.info(f"🤖 filtered_df as:\nshape: {filtered_df.shape}")
    logger.info(filtered_df.head())
    # save to csv
    try:
        filtered_df_save_loc = os.path.join(data_root, f"temp_{fname}_in_{source_name}.csv")
        filtered_df.to_csv(filtered_df_save_loc, index=False)
        logger.info(f"🤖 saved to {filtered_df_save_loc}")
    except Exception as e:
        logger.error(f"🤖 error in saving filtered_df to {filtered_df_save_loc}: {e}")
    return filtered_df_save_loc

def construct_platform_index(hest_loc, data_root, source_index_csv=None, platform_list=["Xenium", "Visium", "ST"]):
    logger.info(f"🤖 constructing HEST1k platform index filtered by {platform_list} in {data_root}") 
    platform_name = "_".join(platform_list)
    if source_index_csv is None:
        hest_db_index_loc, hest_db_index_name = get_hest_db_index(hest_loc)
        source_name = "hest"
    else:
        hest_db_index_loc = source_index_csv
        source_name = str(Path(source_index_csv).stem)
    meta_df = pd.read_csv(hest_db_index_loc)
    filtered_df = meta_df[meta_df["st_technology"].isin(platform_list)] 
    logger.info(f"🤖 filtered_df as:\nshape: {filtered_df.shape}")
    logger.info(filtered_df.head())
    try:
        filtered_df_save_loc = os.path.join(data_root, f"temp_{platform_name}_on_{source_name}.csv")
        filtered_df.to_csv(filtered_df_save_loc, index=False)
        logger.info(f"🤖 saved to {filtered_df_save_loc}")
    except Exception as e:
        logger.error(f"🤖 error in saving filtered_df to {filtered_df_save_loc}: {e}")
    return filtered_df_save_loc

def get_final_index_file(temp_files,filtered_loc, final_index_loc):
    logger.info(f"🤖 getting final index file from {filtered_loc} to {final_index_loc}")
    try:
        filtered_df = pd.read_csv(filtered_loc)
        filtered_df.to_csv(final_index_loc, index=False)
    except Exception as e:
        logger.error(f"🤖 error in saving filtered_df to {final_index_loc}: {e}")
    # remove temp files
    for temp_file in temp_files:
        os.remove(temp_file)
    return final_index_loc

def construct_sub_dataset_index(data_para:HEST1K_sub_database, with_explore:bool=False):
    """
    Construct a sub dataset from HEST1k database.
    """
    logger.info(f"Constructing sub dataset: {data_para.dataset_name}")
    # 0- check dataset folder
    data_root = f"{data_para.dataset_storage_folder}/{data_para.dataset_name}/"
    index_floc = data_root + f"/{data_para.dataset_name}.csv"
    os.makedirs(data_root, exist_ok=True)

    temp_files = []
    # 1- construct tissue index
    filtered_loc =construct_hest1k_tissue_index(data_para.hest_loc, data_root, 
                                  source_index_csv=data_para.source_index_loc, 
                                  organ=data_para.organ, 
                                  species=data_para.species, 
                                  target_oncotree_code=data_para.oncotree_code)
    temp_files.append(filtered_loc)
    # 2- filter by platform
    filtered_loc = construct_platform_index(data_para.hest_loc, data_root, 
                                  source_index_csv=filtered_loc, 
                                  platform_list=data_para.platform)
    temp_files.append(filtered_loc)
    # 3- confirm index loc 
    # save final index file to index_floc
    index_floc = get_final_index_file(temp_files, filtered_loc, index_floc)

    logger.info(f"🤖 dataset construction finished, saved to {index_floc}")
    if with_explore:
        explore_constructed_db(index_floc)
    return index_floc

def load_subdataset(hest_loc,
                    datasets_folder, 
                    subdataset_name,):
    # Check if hest index exists, if it exists then read it, otherwise copy from repo
    subdataset_folder = f"{datasets_folder}/{subdataset_name}/"
    filtered_index_floc = f"{subdataset_folder}/{subdataset_name}.csv"

    # Check if file exists
    if not os.path.exists(filtered_index_floc):
        logger.error(f"index of current subdataset does not exist. Please check the index file.")
        sys.exit(1)
    else:
        filtered_df = pd.read_csv(filtered_index_floc)

    print(filtered_df.head())
    id_list = filtered_df['id'].values

    # Read the entire hest dataset
    hest_data = load_hest(hest_loc, id_list=id_list)
    len_hest_data = len(hest_data)
    return hest_data, len_hest_data, subdataset_folder
###########################################################################################
# Extract patches from HEST database :: image side
###########################################################################################
def extract_patches_from_hest(
        hest_data, len_hest_data, subdataset_folder,
        # patch info
        patch_size, pixel_size,
        force_process:bool=False,
        ):
    patch_save_root = f"{subdataset_folder}/patches/"
    os.makedirs(patch_save_root, exist_ok=True)
    error_files = []
    # Process data
    for idx, st in enumerate(hest_data):
        patch_save_dir = f"{patch_save_root}/"
        f_name = f"patch_{patch_size}_{pixel_size}_{idx}"
        if not force_process and os.path.exists(f"{patch_save_dir}/{f_name}.h5"):
            logger.info(f"🤖 patches for {idx} already exist, not force process, skip")
            continue
        else:
            try:
                st.dump_patches(
                patch_save_dir,
                name=f_name,
                target_patch_size=patch_size,
                target_pixel_size=pixel_size
                )
                logger.info(f"Saved {idx}/{len_hest_data} WSI all patches")
            except Exception as e:
                logger.error(f"🤖 error in saving patches for {idx}: {e}")
                error_files.append(idx)
    if len(error_files) > 0:
        pd.DataFrame(error_files, columns=["id"]).to_csv(f"{patch_save_root}/error_files.csv", index=False)
        logger.warning(f"🤖 {len(error_files)}/{len_hest_data} files are not saved due to errors")
    else:
        logger.info(f"🤖 all files are saved successfully")

def extract_img_vectors(subdataset_folder,
                        model_name:str,
                        model_instance:torch.nn.Module, 
                        num_features:int,# Feature dimension defined by the model
                        patch_size:int, pixel_size:float,
                        transform_instance:transforms.Compose=None,
                        force_process:bool=False):
    
    # step 0: Check patches folder
    patch_loc = f"{subdataset_folder}/patches/"
    folders = [f for f in os.listdir(patch_loc) if f.endswith('.h5') and os.path.isfile(os.path.join(patch_loc, f))]
    logger.info(f"🤖 found {len(folders)} patches files")
    model_folder_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    patch_embed_save_root = f"{subdataset_folder}/patches_embed/{model_folder_name}/"
    print(patch_embed_save_root)
    os.makedirs(patch_embed_save_root, exist_ok=True)
    # step 1: Define transform 
    if transform_instance is None:
        mean = (0.707223, 0.578729, 0.703617)
        std = (0.211883, 0.230117, 0.177517)
        logger.info(f"🤖 using default transform with mean: {mean} and std: {std}")
        transform_instance = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=mean, 
                        std=std
                    ),
        ])

    # step 2: Process each patch
    for idx in range(len(folders)):
        # Step a: Load the data
        f_name = f"patch_{patch_size}_{pixel_size}"
        img_loc = f"{patch_loc}/{f_name}_{idx}.h5"
        logger.info(f"🤖 loading data from {img_loc}")
        features_root = f"{patch_embed_save_root}/"
        emb_fname = f"{f_name}_{idx}.npy"

        # Calculation logic
        try:
            data = h5py.File(img_loc, 'r')
            print(f"data keys: {data.keys()}")
            barcodes = data['barcode'][:]
            coords = data['coords'][:]
            assert coords.shape[0] == len(barcodes)
            num_samples = coords.shape[0]
            print(f"🤖 loading data with {num_samples} samples,from {img_loc} ")
            # Step b: Extract features
            features_list = []
            for i in tqdm(range(num_samples)):
                input_img = data['img'][i, ... ]
                input_tensor = transforms.ToPILImage()(input_img)
                # Using mixed precision for faster inference.
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.inference_mode():
                        features = model_instance(transform_instance(input_tensor).unsqueeze(0).to("cuda"))

                assert features.shape == (1, num_features)
                features_list.append(features.detach().cpu().numpy())
                # Generate new 'feature' data with shape (20549, 1536)
            features_np = np.concatenate(features_list, axis=0)
        except Exception as e:
            logger.error(f"Error in feature extraction: {idx}")
            logger.error(f"Error message: {e}")
        else:
            # step c: Create new npy file
            try:
                floc = features_root + emb_fname
                np.save(floc, features_np)
            except:
                print(f" Error in creating new npy file: {idx}")
            else:
                data_time_stamp = time.time()
                data_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data_time_stamp))
                print(f"🤖 saved {idx}/{len(folders)} WSI patch features to {floc} at {data_time_stamp}")
                print("New npy file creation completed")

###########################################################################################
# Extract sequencing data from HEST database :: sequencing
###########################################################################################
def preprocess_gene_name(gene):
    gene = re.sub(r'^__\w+\.', '', gene)
    parts = re.split(r'[.,;:\s]\s*', gene)
    for part in parts:
        if re.match(r'^[A-Za-z0-9]+$', part):
            return part
    return gene

def gene_name_alignment(hest_data,subdataset_folder):
    logger.info(f"🤖 aligning gene names for {subdataset_folder}")
    gene_name_map_folder = f"{subdataset_folder}/aligned_gene_name/"
    os.makedirs(gene_name_map_folder, exist_ok=True)
    new_adata_folder = f"{subdataset_folder}/aligned_adata/"
    os.makedirs(new_adata_folder, exist_ok=True)
    # Connect to Ensembl's BioMart service
    server = BiomartServer("http://www.ensembl.org/biomart")

    # Get human gene dataset
    dataset = server.datasets['hsapiens_gene_ensembl']

    # Get all gene symbols and corresponding Ensembl gene IDs
    response = dataset.search({
        'attributes': ['ensembl_gene_id', 'external_gene_name']
    })

    # Parse results and establish bidirectional mapping
    lines = response.content.decode('utf-8').split('\n')
    gene_name_to_id = {}
    ensembl_id_to_gene_name = {}
    for line in lines:
        if line:
            data_in_line = line.strip().split('\t')
            if len(data_in_line) == 2:
                ensembl_id, gene_name = data_in_line
                gene_name_to_id[gene_name] = ensembl_id
                ensembl_id_to_gene_name[ensembl_id] = gene_name


    for i, hest_d in enumerate(hest_data):
        print(f"Processing {i}th dataset..")
        gene_index = hest_d.adata.var.index
        mapping_table = pd.DataFrame(index=gene_index, columns=['Gene_Symbol', 'Ensembl_ID'])

        for gene in gene_index:
            gene_clean = preprocess_gene_name(gene)
            # Determine gene identifier type
            if re.match(r'^ENSG\d{11}$', gene_clean):
                # If it's an Ensembl gene ID
                ensembl_id = gene_clean
                gene_symbol = ensembl_id_to_gene_name.get(ensembl_id, None)
            else:
                # Assume it's a gene symbol
                gene_symbol = gene_clean
                ensembl_id = gene_name_to_id.get(gene_clean, None)
            mapping_table.loc[gene, 'Gene_Symbol'] = gene_symbol
            mapping_table.loc[gene, 'Ensembl_ID'] = ensembl_id
        # Save mapping table
        
        mapping_table.to_csv(f"{gene_name_map_folder}/{i}.csv")

        new_adata = sc.AnnData(X=hest_data[i].adata.X, obs=hest_data[i].adata.obs, var=hest_data[i].adata.var)
        new_adata.var_names = mapping_table["Gene_Symbol"].astype(str).values
        new_adata.obsm["spatial"] = hest_data[i].adata.obsm["spatial"]
        new_adata.write(f"{new_adata_folder}/HEST_breast_adata_{i}.h5ad")