import tqdm
import scanpy as sc
import numpy as np
import os
import os.path as osp
import h5py
import glob
import torch

def inference_from_folder(model,
                          dataset_save_folder,
                          scLLM_emb_name,
                          scLLM_emb_ckpt,
                          output_dir,
                          adata_prefix="HEST_breast_adata_",
                          img_prefix="patch_224_0.5_"):
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
    print("change model to inference")
    model.eval()

    # 检查模型精度
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")

    # Setup folders
    img_folder = f'{dataset_save_folder}/patches'
    anndata_folder = f'{dataset_save_folder}/scLLM_embed/{scLLM_emb_name}/{scLLM_emb_ckpt}/paired_seq'

    # Get all adata files
    adata_files = glob.glob(osp.join(anndata_folder, f"{adata_prefix}*.h5ad"))
    nb_adata_files = len(adata_files)
    print(f"Looking for adata files in: {anndata_folder}")
    print(f"Number of adata files found: {len(adata_files)}")
    print(f"Looking for image files in: {img_folder}")

    def find_img_file(anndata_file):
        """Find corresponding image file for an adata file"""
        base_name = os.path.splitext(os.path.basename(anndata_file))[0]
        file_idx = int(base_name.split(adata_prefix)[-1])
        img_file = osp.join(img_folder, f"{img_prefix}{file_idx}.h5")
        img_file = osp.join(img_folder, f"{img_prefix}{file_idx}.h5")
        return img_file
    print("start inference for files.")
    # Process each adata file
    with torch.no_grad():
        for adata_file in adata_files:
            print(f"Processing {adata_file}")

            # Load adata file
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
                img_data = f['img'][:]  # Load all patches

            # Get indices of filtered barcodes in image data
            valid_indices = [idx for idx, bc in enumerate(img_barcodes)
                           if bc in filter_barcodes]

            if not valid_indices:
                print(f"No valid patches found for {adata_file}, skipping...")
                continue

            # Process patches in batches
            batch_size = 32
            all_features = []

            for i in tqdm.tqdm(range(0, len(valid_indices), batch_size)):
                batch_indices = valid_indices[i:i + batch_size]
                image = img_data[batch_indices]
                image = image.transpose(0, 3, 1, 2)
                batch_imgs = torch.from_numpy(image).to(dtype=model_dtype, device='cuda')  # 转换为与模型相同的精度
                #print(f"batch_imgs shape: {batch_imgs.shape}")
                # Forward pass through model
                features = model(batch_imgs)
                all_features.append(features)

            # Concatenate all features
            file_features = torch.cat(all_features, dim=0)

            try:
                # Save features
                base_name = osp.splitext(osp.basename(adata_file))[0]
                save_path = osp.join(output_dir, f'{base_name}.npy')
                print(f"Attempting to save features to: {save_path}")
                np.save(save_path, file_features.cpu().numpy())
                print(f"Features successfully successfully saved to:: {save_path}")
            except Exception as e:
                print(f"Error while saving features: {str(e)}")