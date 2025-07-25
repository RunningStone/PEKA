"""
This file is used to download HEST1k dataset

for following steps:
    (1) config project path
    (2) config python import path add peka and external models
    (3) check folders Data, OUTPUT, Pretrained to create them if not exist 
    (4) check .env file which save key environment variables
    (5) check HF_TOKEN
    (6) check HEST1K_STORAGE_PATH
"""

print("🤖 running 1_dataset_downloader.py")
# config runable path
import sys
import os
proj_path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(proj_path)
print(f" ⭐️ proj_path: {proj_path}")

# add python import path
code_path = str(proj_path) + "/PEKA/"
sys.path.append(code_path)
print(f" ⭐️ code_path: {code_path}")
# add external models path
external_module_path = str(proj_path) + "/PEKA/External_models/"
sys.path.append(external_module_path)
print(f" ⭐️ external_module_path: {external_module_path}")
sys.path.append(external_module_path + "/HEST/src/")

# import peka
from peka import logger
from peka.Data.download_helper import download_hest1k
from peka.Data.hest1k_helper import get_latest_version_hest_index
import dotenv

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HEST1k dataset.")
    parser.add_argument('--force_download', type=bool, default=False, help="Force download the HEST1k dataset.")
    args = parser.parse_args()

    env_file_path = str(proj_path) + "/PEKA/.env"
    print(f"⭐️ loading env file from {env_file_path}")
    dotenv.load_dotenv(dotenv_path=env_file_path)

    hest_storage_path = os.getenv("HEST1K_STORAGE_PATH")
    hf_token = os.getenv("HF_TOKEN")

    index_full_path, index_file_name = get_latest_version_hest_index(hest_storage_path)
    print(f"index_full_path: {index_full_path}")
    print(f"index_file_name: {index_file_name}")
    if index_full_path is None or args.force_download:
        print(f"🚨 HEST1k Database not found, or force download is set as {args.force_download}, so download HEST1k Database.")
        download_hest1k(hest_storage_path, hf_token)
        
    else:
        print(f"📃 HEST1k Database in : {hest_storage_path}")
        print(f"📃 HEST1k Database index file: {index_file_name} in {index_full_path}.")


