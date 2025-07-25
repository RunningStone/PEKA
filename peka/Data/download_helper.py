import os
import json
import requests
from typing import List
import cellxgene_census
from dotenv import load_dotenv, set_key

from peka import logger


class CustomDatasetDownloader:
    def __init__(self,base_folder, definition_file,verbose=False):
        self.base_folder = base_folder
        self.definition_file = definition_file
        self.verbose = verbose
        self.datasets = self.load_datasets()

    def load_datasets(self):
        logger.info(f"Loading datasets from {self.definition_file}")
        with open(self.definition_file, 'r') as file:
            return json.load(file)['datasets']

    def download_file(self, url, dest_folder):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        response = requests.get(url, stream=True)
        filename = os.path.join(dest_folder, url.split('/')[-1])
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        if self.verbose: logger.debug(f"Downloaded {filename}")

    def process_dataset(self, dataset):
        print(f"Downloading dataset {dataset['name']}")
        dataset_name = dataset['name']
        base_folder = os.path.join(self.base_folder, dataset_name)
        
        # Create base folder for dataset
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # Download meta info files
        meta_info_folder = os.path.join(base_folder, 'meta_info')
        for url in dataset['meta_info']:
            self.download_file(url, meta_info_folder)
        
        # Process each sample
        for sample in dataset['samples']:
            sample_name = sample['name']
            sample_folder = os.path.join(base_folder, sample_name)
            if not os.path.exists(sample_folder):
                os.makedirs(sample_folder)
            
            # Download sample files
            for url in sample['files']:
                self.download_file(url, sample_folder)
        
        print(f"Processed dataset {dataset_name}")

    def create_index(self):
        index_file = os.path.join('datasets', 'index.json')
        with open(index_file, 'w') as file:
            json.dump(self.datasets, file, indent=4)
        print(f"Index file created at {index_file}")

    def run(self):
        for dataset in self.datasets:
            self.process_dataset(dataset)
        self.create_index()

        
"""
if __name__ == '__main__':
    downloader = DatasetDownloader('datasets_definition.json')
    downloader.run()
"""


def download_hest1k(hest_storage_path, hf_token=None, chose_ver:str="dataset"):
    
    if not hf_token:
        logger.warning("Warning: HF_TOKEN is not set in .env file. Can only use snapshot download.")
        chose_ver = "snapshot"

    if chose_ver == "snapshot":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id="MahmoodLab/hest", repo_type='dataset', local_dir=hest_storage_path)

    elif chose_ver == "dataset":
        from huggingface_hub import login
        # Use environment variable HF_TOKEN or pass token as parameter
        if hf_token is None:
            hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
        else:
            print("Warning: No Hugging Face token provided. Set HF_TOKEN environment variable or pass hf_token parameter.")

        import datasets
        # Note that the full dataset is around 1TB of data

        dataset = datasets.load_dataset(
            'MahmoodLab/hest', 
            cache_dir=hest_storage_path,
            patterns='*'
        )

#################################################################################################################
#
#################################################################################################################


def get_latest_census_version() -> str:
    """
    Get the latest available census version from cellxgene_census
    """
    available_versions = cellxgene_census.get_census_version_directory()
    if not available_versions:
        raise ValueError("No census versions available")
    
    # Get the latest version
    latest_version = sorted(available_versions)[-1]
    logger.info(f"Using census version: {latest_version}")
    return latest_version

def get_census_version() -> str:
    """
    Get census version from .env file or fetch latest version
    """
    # Load .env file
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(env_file)
    
    # Try to get version from .env
    version = os.getenv('CELLXGENE_VERSION')
    if not version:
        # Get latest version and save to .env
        version = get_latest_census_version()
        set_key(env_file, 'CELLXGENE_VERSION', version)
        logger.info(f"Saved census version {version} to .env file")
    
    return version

def retrieve_soma_idx(query_name, value_filter) -> List[str]:
    """
    This function is used to retrieve cell soma ids from cellxgene census based on the query name
    """
    version = get_census_version()
    with cellxgene_census.open_soma(census_version=version) as census:
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = value_filter[query_name],
        column_names = ["soma_joinid"]
    )
    cell_metadata = cell_metadata.concat()
    cell_metadata = cell_metadata.to_pandas()
    return cell_metadata["soma_joinid"].to_list()

def download_cellxgene_partition(partition_idx, query_name, output_dir, index_dir, partition_size):
    """
    This function is used to download the partition_idx partition of the query_name
    """
    # define id partition
    id_list = load2list(query_name, index_dir)
    id_partition =  define_partition(partition_idx, id_list, partition_size)
    version = get_census_version()
    with cellxgene_census.open_soma(census_version=version) as census:
        adata = cellxgene_census.get_anndata(census,    
                                            organism="Homo sapiens",
                                            obs_coords=id_partition,
                                            )
    # prepare the query dir if not exist
    query_dir = os.path.join(output_dir, query_name)
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    adata.write_h5ad(query_adata_path)
    return query_adata_path

def del_partition(partition_idx, query_name, output_dir, index_dir, partition_size):
    query_dir = os.path.join(output_dir, query_name)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    os.remove(query_adata_path)


# build soma index for following download

def convert2file(idx_list: List[str], query_name: str, output_dir: str) -> None:
    """
    This function is used to convert the retrieved idx_list to file by query_name
    """

    # set up the dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{query_name}.idx")

    # write to the file
    with open(file_path, 'w') as fp:
        for item in idx_list:
            fp.write("%s\n" % item)

def build_soma_idx(query_name, value_filter, output_dir) -> None:
    """
    This function is used to build the soma idx for cells under query_name
    """
    idx_list = retrieve_soma_idx(query_name, value_filter)
    convert2file(idx_list, query_name, output_dir)

def define_partition(partition_idx, id_list, partition_size) -> List[str]:
    """
    This function is used to define the partition for each job

    partition_idx is the partition index, which is an integer, and 0 <= partition_idx <= len(id_list) // MAX_PARTITION_SIZE
    """
    i = partition_idx * partition_size
    return id_list[i:i + partition_size]


def load2list(query_name, soma_id_dir) -> List[int]:
    """
    This function is used to load the idx list from file
    """
    file_path = os.path.join(soma_id_dir, f"{query_name}.idx")
    with open(file_path, 'r') as fp:
        idx_list = fp.readlines()
    idx_list = [int(x.strip()) for x in idx_list]
    return idx_list