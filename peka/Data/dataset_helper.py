import h5py
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset
import anndata
from tqdm import tqdm
from PIL import Image 
from typing import Optional, List, Tuple

class BatchLocalityDataset(Dataset):
    def __init__(self, dataset_save_folder, scLLM_emb_name,
                 img_prefix,
                 random_sample_barcode=True, # Whether to randomly select barcode 
                 batch_switch_interval=5, embedding_prefix="", 
                 read_data_and_keep_in_mem=False, scLLM_emb_ckpt:str="default",
                 label_name:str=None,
                 train_phase: Optional[str] = "train",
                 train_val_split_dict: Optional[dict] = None,
                 val_ratio: Optional[float] = None,
                 split_seed: int = 42  # Random seed
                 ): 
        
        # step 0: init parameters
        img_folder = f'{dataset_save_folder}/patches/'
        anndata_folder = f'{dataset_save_folder}/scLLM_embed/{scLLM_emb_name}/{scLLM_emb_ckpt}/paired_seq/'
        embedding_folder = f'{dataset_save_folder}/scLLM_embed/{scLLM_emb_name}/{scLLM_emb_ckpt}/embeddings/'

        self.adata_prefix = "HEST_breast_adata_"
        self.img_prefix = img_prefix
        self.img_folder = img_folder

        self.embedding_prefix = embedding_prefix
        self.read_data_and_keep_in_mem = read_data_and_keep_in_mem

        self.random_sample_barcode = random_sample_barcode
        self.label_name = label_name
        self.label_data_in_files = {}  # Used to store label data for each file

        self.train_val_split_dict = train_val_split_dict
        self.train_phase = train_phase
        self.val_ratio = val_ratio
        self.split_seed = split_seed

        # Collect Anndata files (.h5ad)
        self.anndata_files = [os.path.join(anndata_folder, f) for f in os.listdir(anndata_folder) if f.endswith('.h5ad')]
        # Collect embedding files (.npy)
        self.embedding_files = [os.path.join(embedding_folder, f) for f in os.listdir(embedding_folder) if f.endswith('.npy')]

        # Files that exist in both anndata_files and embedding_files, remove those that don't exist. Note that the two lists have different file extensions, only compare filenames
        common_fname = self._get_common_fname(self.anndata_files, self.embedding_files)
        self.anndata_files = [f for f in self.anndata_files if os.path.basename(f).split(".")[0] in common_fname]
        self.embedding_files = [f for f in self.embedding_files if os.path.basename(f).split(".")[0] in common_fname]

        # step 1: load embedding and image from anndata files
        self._load_anndata_and_embeddings()
        self._create_img_index()

        # If label_name is specified, load label data
        if self.label_name is not None:
            self._load_labels()
        
        # step 2: filter valid samples and set train phase
        self._filter_all_valid_samples()
        self.set_train_phase()
        
        # Control batch switching logic
        self.batch_switch_interval = batch_switch_interval
        self.current_file_index = 0
        self.batch_counter = 0


    ###########################################################################################
    # Load data
    ###########################################################################################
    def _get_common_fname(self, anndata_files, embedding_files):
        fname_in_anndata = [os.path.basename(f).split(".")[0] for f in anndata_files]
        fname_in_embedding = [os.path.basename(f).split(".")[0] for f in embedding_files]
        common_fname = [f for f in fname_in_anndata if f in fname_in_embedding]
        print(f"Total {len(common_fname)} files")
        return common_fname


    def _load_anndata_and_embeddings(self):
        """Load Anndata files and embedding vectors, and associate embedding vectors with barcodes"""
        self.embedding_data_in_files = {}
        self.barcodes_in_files = {}
        for anndata_file in self.anndata_files:
            embedding_data_in_file = {}
            barcodes_in_file = []
            # 读取 Anndata 文件
            adata = anndata.read_h5ad(anndata_file)
            barcodes = adata.obs.index.values
            filter_flags = adata.obs['filter_flag'].values
            #print(f" Number of rows where filter_flags is False: {len(filter_flags)-np.sum(filter_flags)}")
            # 获取通过 QC 的条形码
            filter_barcodes = barcodes[~filter_flags]
            print(f" Number of barcodes that passed QC in {anndata_file}: {len(filter_barcodes)}")
            # Find corresponding embedding file
            embedding_file = self._find_embedding_file(anndata_file)
            if embedding_file is None:
                continue
            embeddings = np.load(embedding_file)
            # Check if embedding vector and barcode counts match
            if len(filter_barcodes) != embeddings.shape[0]:
                raise ValueError(f"In {anndata_file}, embedding vector count does not match filtered barcode count")
            # Associate embedding vectors with barcodes
            for i, (barcode, embedding) in enumerate(zip(filter_barcodes, embeddings)):
                if self.read_data_and_keep_in_mem:
                    embedding_data_in_file[barcode] = embedding
                else:
                    embedding_data_in_file[barcode] = i
            self.embedding_dim = embeddings.shape[1]
            self.embedding_data_in_files.update({anndata_file: embedding_data_in_file})
            self.barcodes_in_files.update({anndata_file: filter_barcodes})

    def _find_embedding_file(self, anndata_file):
        """Find the embedding file corresponding to the Anndata file"""
        # Assume embedding file has same name as Anndata file, but with .npy extension
        base_name = os.path.splitext(os.path.basename(anndata_file))[0]
        
        if self.embedding_prefix != "":
            base_name = base_name + self.embedding_prefix 
            #print(f" with prefix, full name of npy file: {base_name}")
        for emb_file in self.embedding_files:
            emb_base_name = os.path.splitext(os.path.basename(emb_file))[0]
            #print(f" {emb_base_name} || {base_name}")
            if emb_base_name == base_name:
                return emb_file
        print(f"In {anndata_file} && {base_name}, corresponding embedding file not found")
        return None

    def _create_img_index(self,):
        """Create mapping from barcode to (file index, intra-file index)"""
        self.img_index_in_files = {}
        for file_idx, file in enumerate(self.anndata_files):
            img_index_in_file = {}
            img_data_file = self._find_img_file(file)
            with h5py.File(img_data_file, 'r') as f:
                # Read barcodes and convert to strings
                barcodes = f['barcode'][:, 0]
                barcodes = [barcode.decode('utf-8') if isinstance(barcode, bytes) else str(barcode) for barcode in barcodes]
                for idx, barcode in enumerate(barcodes):
                    img_index_in_file[barcode] = idx
            self.img_index_in_files.update({file: img_index_in_file})

    def _find_img_file(self, anndata_file):
        """Find the image file corresponding to the Anndata file"""
        base_name = os.path.splitext(os.path.basename(anndata_file))[0]
        file_idx = int(base_name.split(self.adata_prefix)[-1])
        #print(f" {self.adata_prefix} base_name: {base_name} || file_idx: {file_idx} || {self.img_prefix}")
        img_file = self.img_folder + f"/{self.img_prefix}_{file_idx}.h5"
        return img_file
    
    ###########################################################################################
    # Final filtering step, maintain an index of available samples
    ###########################################################################################
    def _filter_all_valid_samples(self):
        """Filter out samples that have both embedding and image data"""
        valid_samples = []
        self.valid_barcodes_in_files = {}  # Store valid barcodes for each file
        
        for file_idx, anndata_file in enumerate(self.anndata_files):
            embedding_file = self._find_embedding_file(anndata_file)
            img_file = self._find_img_file(anndata_file)
            
            if embedding_file and img_file:
                # Find barcodes that exist in all data structures
                barcodes = set(self.barcodes_in_files[anndata_file])
                barcodes = barcodes.intersection(set(self.embedding_data_in_files[anndata_file].keys()))
                barcodes = barcodes.intersection(set(self.img_index_in_files[anndata_file].keys()))
                
                if self.label_name is not None:
                    barcodes = barcodes.intersection(set(self.label_data_in_files[anndata_file].keys()))
                
                valid_barcodes = sorted(list(barcodes))  # Sort to ensure consistent order
                if valid_barcodes:  # Only add when there are valid barcodes
                    self.valid_barcodes_in_files[anndata_file] = valid_barcodes
                    valid_samples.extend([(file_idx, i) for i in range(len(valid_barcodes))])
        
        if not valid_samples:
            raise ValueError("No valid samples found! Please check your data files.")
        
        self.valid_samples_idx = valid_samples
        self.valid_samples_counts = len(valid_samples)
        
        # Initialize file handler
        self.current_file_index = 0
        self._switch_file(self.anndata_files[0])
        
        print(f"Total valid samples: {self.valid_samples_counts}")

    def _split_dataset(self):
        """Split dataset into train and validation indices"""
        if not hasattr(self, 'valid_samples_idx'):
            raise ValueError("Dataset must have valid_samples_idx to split")
            
        if self.val_ratio is None:
            raise ValueError("val_ratio must be specified for dataset splitting")
            
        # Set random seed for reproducibility
        random.seed(self.split_seed)
        
        # Get all indices
        all_indices = list(range(len(self.valid_samples_idx)))
        
        # Generate train and validation indices
        val_size = int(len(all_indices) * self.val_ratio)
        val_indices = random.sample(all_indices, val_size)
        train_indices = list(set(all_indices) - set(val_indices))
        
        # Sort indices for better reproducibility
        train_indices.sort()
        val_indices.sort()
        
        # Return split dictionary
        return {
            "train": train_indices,
            "val": val_indices
        }

    def set_train_phase(self):
        """Set the dataset to training or validation phase"""
        # If train_val_split_dict and val_ratio are not specified, use all samples
        if self.train_val_split_dict is None and self.val_ratio is None:
            self.train_val_split_dict = {
                "train": list(range(len(self.valid_samples_idx))),
                "val": []
            }
            self.train_phase = "train"  # Force set to train
        elif self.train_val_split_dict is None:
            if self.val_ratio is None:
                raise ValueError("val_ratio must be specified when train_val_split_dict is not provided")
            self.train_val_split_dict = self._split_dataset()
        
        if self.train_phase not in ["train", "val"]:
            raise ValueError(f"Invalid train_phase value: {self.train_phase}. Must be 'train' or 'val'")
        
        # Select indices based on phase
        selected_indices = self.train_val_split_dict[self.train_phase]
        
        # If random sampling is enabled, shuffle the selected indices
        if self.random_sample_barcode:
            random.seed(self.split_seed)  # Maintain consistent randomness
            selected_indices = selected_indices.copy()  # Avoid modifying original indices
            random.shuffle(selected_indices)
        
        # Update valid samples
        self.valid_samples_idx = [self.valid_samples_idx[i] for i in selected_indices]
        self.valid_samples_counts = len(self.valid_samples_idx)
        
        print(f"Dataset phase: {self.train_phase}")
        print(f"Train/Val split dict provided: {self.train_val_split_dict is not None}")
        print(f"Dataset size: {self.valid_samples_counts}")

    def _load_labels(self):
        """Load label data"""
        for anndata_file in self.anndata_files:
            label_data_in_file = {}
            # 读取 Anndata 文件
            adata = anndata.read_h5ad(anndata_file)
            if self.label_name not in adata.obs.columns:
                raise ValueError(f"Label column '{self.label_name}' not found in {anndata_file}")
                
            barcodes = adata.obs.index.values
            filter_flags = adata.obs['filter_flag'].values
            labels = adata.obs[self.label_name].values
            
            # Get barcodes that passed QC and corresponding labels
            filter_barcodes = barcodes[~filter_flags]
            filter_labels = labels[~filter_flags]
            
            # Associate labels with barcodes
            for barcode, label in zip(filter_barcodes, filter_labels):
                label_data_in_file[barcode] = label
            self.label_data_in_files[anndata_file] = label_data_in_file

    def __len__(self):
        return self.valid_samples_counts

    ###########################################################################################
    # Data loading logic
    ###########################################################################################
    def _switch_file(self, adata_file=None):
        """Switch to the next image file"""
        # If a file is specified, switch to the specified file
        if adata_file is not None:
            self.current_file = adata_file
        else:
            # Otherwise, cyclically switch to the next file
            self.current_file_index = (self.current_file_index + 1) % len(self.anndata_files)
            self.current_file = self.anndata_files[self.current_file_index]
            
        # Close previous file handle (if exists)
        if hasattr(self, 'img_h5_file_handler'):
            self.img_h5_file_handler.close()
        
        # Open new file handle
        self.img_h5_file_handler = h5py.File(self._find_img_file(self.current_file), 'r')
        self.emb_file_handler = np.load(self._find_embedding_file(self.current_file))
        
        # Reset batch counter
        self.batch_counter = 0

    def _read_data_pair(self, idx=None):
        """Read data pair, if idx is not None, only read the embedding corresponding to idx"""
        if idx is not None:
            # If idx is provided, use it to get file and barcode
            file_idx, barcode_idx = self.valid_samples_idx[idx]
            anndata_file = self.anndata_files[file_idx]
            barcode = self.valid_barcodes_in_files[anndata_file][barcode_idx]
        else:
            # Otherwise, randomly select a barcode from the current file
            anndata_file = self.current_file
            barcode = random.choice(self.valid_barcodes_in_files[anndata_file])

        # Load image and embedding vector
        image = self._load_image_by_barcode(anndata_file, barcode)
        embedding = self._load_embedding_by_barcode(anndata_file, barcode)

        if self.label_name is not None:
            label = self.label_data_in_files[anndata_file][barcode]
            return embedding, image, label
        return embedding, image

    def _load_image_by_barcode(self,anndata_file, barcode):
        """Load image from corresponding h5 file based on barcode"""
        img_idx = self.img_index_in_files[anndata_file][barcode]
        image = self.img_h5_file_handler['img'][img_idx, :, :, :]
        return image

    def _load_embedding_by_barcode(self, anndata_file, barcode):
        """Load embedding vector from corresponding npy file based on barcode"""
        embedding_idx = self.embedding_data_in_files[anndata_file][barcode]
        if not self.read_data_and_keep_in_mem:
            embedding = self.emb_file_handler[embedding_idx, ...] # 1dim or 2dim feature
        else:
            embedding = embedding_idx
        return embedding

    def __getitem__(self, idx):
        if self.random_sample_barcode:
            # Check if file switching is needed
            if self.batch_counter >= self.batch_switch_interval:
                self._switch_file()  # This will reset batch_counter

            # Randomly select one from valid barcodes in current file
            if self.label_name is not None:
                embedding, image, label = self._read_data_pair()  # random sample with label
            else:
                embedding, image = self._read_data_pair()  # random sample without label

            self.batch_counter += 1  # Increment counter
        else:
            # If not random sampling, select file based on index
            file_idx = self.valid_samples_idx[idx][0]
            self._switch_file(adata_file=self.anndata_files[file_idx])
            if self.label_name is not None:
                embedding, image, label = self._read_data_pair(idx)  # not random sample with label
            else:
                embedding, image = self._read_data_pair(idx)  # not random sample without label

        # Convert to torch tensor
        #Image [224, 224, 3] to [3, 224, 224]
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        embedding = torch.from_numpy(embedding).float()
        
        if self.label_name is not None:
            label = torch.tensor(label).long()
            return image, embedding,  label
        return image, embedding
