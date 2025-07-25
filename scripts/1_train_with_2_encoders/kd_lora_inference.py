import sys
import os
import argparse
import torch
import pytorch_lightning as pl
from hydra_zen import instantiate, load_from_yaml

# set paths
notebook_path = os.getcwd()
root = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
code_root = root + "/PEKA/"
env_file_path = f"{code_root}/.env"
data_root = code_root + "DATA/"  
ckpt_folder = f"{root}/Pretrained/"
OUTPUT_ROOT = f"{root}/OUTPUT/"
all_configs_path = f"{code_root}/hydra_zen/Configs/"

#   add paths
sys.path.append(root)
sys.path.append(f'/{code_root}/')
sys.path.append(f'/{code_root}/peka/External_models/')
sys.path.append(f'/{code_root}/peka/External_models/HEST/src/')

import peka
# environment settings
import dotenv
dotenv.load_dotenv(env_file_path)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

from huggingface_hub import login
login(token=HF_TOKEN)

from peka.Trainer.KD_LoRA import pl_KD_LoRA
from peka.Hydra_helper.experiment_helpers import save_experiment_configs
from peka.Hydra_helper.pl_model_helpers import create_pl_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--trainer_config', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save inference results')
    return parser.parse_args()

def inference_from_folder(model, dataset_save_folder, scLLM_emb_name, scLLM_emb_ckpt, output_dir, adata_prefix="adata", img_prefix="patches"):
    """
    Perform inference using the trained model directly from data folders
    
    Args:
        model: trained model
        dataset_save_folder: root folder containing all data
        scLLM_emb_name: name of the embedding
        scLLM_emb_ckpt: checkpoint of the embedding
        output_dir: directory to save features
        adata_prefix: prefix for adata files
        img_prefix: prefix for image files
    """
    model.eval()
    import numpy as np
    import os
    import os.path as osp
    import h5py
    import glob
    import torch
    
    # setup folders
    img_folder = f'{dataset_save_folder}/patches'
    anndata_folder = f'{dataset_save_folder}/scLLM_embed/{scLLM_emb_name}/{scLLM_emb_ckpt}/paired_seq'
    
    # get all adata files
    adata_files = glob.glob(osp.join(anndata_folder, f"{adata_prefix}*.h5ad"))
    
    def find_img_file(anndata_file):
        """Find corresponding image file for an adata file"""
        base_name = os.path.splitext(os.path.basename(anndata_file))[0]
        file_idx = int(base_name.split(adata_prefix)[-1])
        img_file = osp.join(img_folder, f"{img_prefix}_{file_idx}.h5")
        return img_file
    
    # process each adata file
    with torch.no_grad():
        for adata_file in adata_files:
            print(f"Processing {adata_file}")
            
            # load adata file
            adata = sc.read_h5ad(adata_file)
            barcodes = adata.obs.index.values
            filter_flags = adata.obs['filter_flag'].values
            filter_barcodes = barcodes[~filter_flags]
            
            # Find and load corresponding image file
            img_file = find_img_file(adata_file)
            if not osp.exists(img_file):
                print(f"Warning: Image file {img_file} not found, skipping...")
                continue
                
            # Create barcode to image index mapping
            with h5py.File(img_file, 'r') as f:
                img_barcodes = f['barcode'][:, 0]
                img_barcodes = [bc.decode('utf-8') if isinstance(bc, bytes) else str(bc) 
                              for bc in img_barcodes]
                img_data = f['patches'][:]  # Load all patches
                
            # Get indices of filtered barcodes in image data
            valid_indices = [idx for idx, bc in enumerate(img_barcodes) 
                           if bc in filter_barcodes]
            
            if not valid_indices:
                print(f"No valid patches found for {adata_file}, skipping...")
                continue
                
            # Process patches in batches
            batch_size = 32
            all_features = []
            
            for i in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[i:i + batch_size]
                batch_imgs = torch.from_numpy(img_data[batch_indices]).float()
                
                # Forward pass through model
                features = model(batch_imgs)
                all_features.append(features)
            
            # Concatenate all features
            file_features = torch.cat(all_features, dim=0)
            
            # Save features
            base_name = osp.splitext(osp.basename(adata_file))[0]
            save_path = osp.join(output_dir, f'{base_name}.npy')
            np.save(save_path, file_features.cpu().numpy())
            print(f"Features saved to {save_path}")
            

def main():
    args = get_args()
    
    # load configs
    print(" Loading Zen Configs...")
    model_config = load_from_yaml(os.path.join(all_configs_path, args.model_config))
    pl_model_config = load_from_yaml(os.path.join(all_configs_path, "PL_Model/kd_lora.yaml"))
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # instantiate model
    model = instantiate(model_config, target_dim=embedding_dim)
    
    # create KD_LoRA model
    kd_model = create_pl_model(
        model_instance=model,
        optimizer_instance_list=[],  # No optimizers needed for inference
        scheduler_instance_list=[],  # No schedulers needed for inference
        metrics_factory=None,  # No metrics needed for inference
        loss_instance=None,  # No loss needed for inference
        pl_model_config=pl_model_config,
        model_type="kd_lora"
    )
    
    # load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    # load model 
    model_state_dict = torch.load(args.phase2_ckpt)
    # 移除"model."前缀
    new_state_dict = {}
    for k, v in model_state_dict['state_dict'].items():
        if k.startswith('model.'):
            new_k = k[6:]  # 截掉'model.'
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict,strict=False)
    
    # inference
    predictions = inference(kd_model, test_loader, args.output_dir)
    
if __name__ == "__main__":
    main()
