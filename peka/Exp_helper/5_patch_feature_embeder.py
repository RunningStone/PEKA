"""
用于快速尝试，预先提取patch的特征，用于后续的实验
"""
print("🤖 running 4_patch_feature_embeder.py")
# config project path
import argparse
import sys
import os
import torch
import timm
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
proj_path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(proj_path)
sys.path.append(proj_path + "/PEKA/")
load_dotenv(dotenv_path=f"{proj_path}/PEKA/.env")
hf_token = os.getenv("HF_TOKEN")
print(f"⭐️ proj_path: {proj_path}")
print(f"⭐️ hf_token: {hf_token}")

# set paths
notebook_path = os.getcwd()
root = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
code_root = root + "/PEKA/"
env_file_path = f"{code_root}/.env"
data_root = code_root + "DATA/"  
ckpt_folder = f"{root}/Pretrained/"
OUTPUT_ROOT = f"{root}/OUTPUT/"
all_configs_path = f"{code_root}/hydra_zen/Configs/"
print(f" code root folder for project: {code_root}")
print(f" data root folder for project: {data_root}")
print(f" ckpt folder for project: {ckpt_folder}")
print(f" output folder for project: {OUTPUT_ROOT}")
print(f" configs root folder for project: {all_configs_path}")

# add paths
sys.path.append(root)
sys.path.append(f'/{code_root}/')
sys.path.append(f'/{code_root}/peka/External_models/')
sys.path.append(f'/{code_root}/peka/External_models/HEST/src/')

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract image vectors using a specified model.")
    parser.add_argument("--model_name", type=str, default="hf-hub:bioptimus/H-optimus-0", help="Name of the model.")
    parser.add_argument("--subdataset_folder", type=str, required=True, help="Folder for the subdataset.")
    parser.add_argument("--num_features", type=int, default=1536, help="Number of features for the model output.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for image processing.")

    args = parser.parse_args()


    login(token=hf_token)

    # Import project-specific helper
    from peka.Data.hest1k_helper import extract_img_vectors

    # Load model
    model = timm.create_model(args.model_name, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.to("cuda")
    model.eval()

    # Run the extraction function
    extract_img_vectors(
        subdataset_folder=args.subdataset_folder,
        model_name=args.model_name.split(":")[-1],  # Simplified model name for extraction
        model_instance=model,
        num_features=args.num_features,
        patch_size=args.patch_size,
        pixel_size=0.5  # This is kept fixed but can be added as an argument if needed
    )
