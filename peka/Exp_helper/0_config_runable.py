"""
This file is used to config environment for running experiments

for following steps:
    (1) config project path
    (2) config python import path add peka and external models
    (3) check folders Data, OUTPUT, Pretrained to create them if not exist 
    (4) check .env file which save key environment variables
    (5) check HF_TOKEN
    (6) check HEST1K_STORAGE_PATH
"""

print("🤖 running 0_config_runable.py")
# config runable
import sys
import os
proj_path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(proj_path)
print(f" ⭐️ proj_path: {proj_path}")

# add python import path
code_path = str(proj_path) + "/PEKA/"
sys.path.append(code_path)
print(f" ⭐️ code_path: {code_path}")


try:
    import peka
    from peka import logger
    from peka.Data.download_helper import CustomDatasetDownloader
except Exception as e:
    logger.error(f" 🤖 try to import src blocks named peka, check error log: {e}")


# add external models path
external_module_path = str(code_path) + "/peka/External_models/"
sys.path.append(external_module_path)
logger.info(f" ⭐️ external_module_path: {external_module_path}")
sys.path.append(external_module_path + "/HEST/src/")

# external models
try:
    print(" 🤖 try to import external models")
    import hest
    print("    ⭐️ hest imported")
    import scFoundation
    print("    ⭐️ scFoundation imported")
    print(" ⭐️ All external models imported")
except Exception as e:
    logger.error(f" 🤖 external models not imported, check error log: {e}") 


# check folders, if not exist, create them
# Data, OUTPUT, Pretrained, 
data_folder_loc = str(proj_path) + "/DATA/"
output_folder_loc = str(proj_path) + "/OUTPUT/"
pretrained_folder_loc = str(proj_path) + "/Pretrained/"
logger.info(f" ⭐️ check folders: \n{data_folder_loc},\n {output_folder_loc},\n {pretrained_folder_loc} \n")
if not os.path.exists(data_folder_loc): os.makedirs(data_folder_loc)
if not os.path.exists(output_folder_loc): os.makedirs(output_folder_loc)
if not os.path.exists(pretrained_folder_loc): os.makedirs(pretrained_folder_loc)
logger.info(f" ⭐️ All folders checked and created")

# check .env
from dotenv import load_dotenv
env_path = str(code_path) + "/.env"
load_dotenv(dotenv_path=env_path)

# get environment variables
hest_storage_path = os.getenv("HEST1K_STORAGE_PATH")
hf_token = os.getenv("HF_TOKEN")

if not hest_storage_path:
    logger.error("Error: HEST1K_STORAGE_PATH is not set in .env file. Please configure it to continue.")
    sys.exit()

# check HF_TOKEN
if not hf_token:
    logger.warning("Warning: HF_TOKEN is not set in .env file. Can only use snapshot download.")

logger.info(f" ⭐️ hest_storage_path: {hest_storage_path}. you can change it in .env file.")