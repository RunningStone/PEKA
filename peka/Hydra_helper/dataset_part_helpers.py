"""
some naive code to simplify the usage of hydra-zen

"""
from peka import logger
from peka.Data.dataset_helper import BatchLocalityDataset
from peka.Data.database_helper import get_preprocess_status

import pandas as pd
from torch.utils.data import DataLoader

def dataset_generator(data_root:str, 
                           tissue_name:str,
                           task:str,
                           # 
                           random_sample_barcode:bool,
                           batch_switch_interval:int,
                           scLLM_emb_name:str,
                           scLLM_emb_prefix:str,
                           patch_size:int,
                           pixel_size:float,
                           img_prefix:str,
                           scLLM_emb_ckpt:str,
                           #
                           batch_size:int,
                           num_workers:int,
                           additional_dataloader_para:dict,
                           label_name:str=None,
                           split_dataset:bool=False,  # whether split dataset
                           shuffle:bool=False,  # depends on split_dataset
                           val_ratio:float=0.2,  # validation ratio
                           split_seed:int=42,  # random seed
                           ):
    logger.info(f"🤖 start dataset config {task}")
    current_tissue_folder = f"{data_root}/{tissue_name}/"
    dataset_config_df = pd.read_csv(f"{current_tissue_folder}/dataset_config.csv")
    dataset_names_list = dataset_config_df["dataset_name"].tolist()
    assert task in dataset_names_list, f"task {task} not in predefined datasets. Available tasks: {dataset_names_list}"
    dataset_save_folder = f"{current_tissue_folder}/{task}/"
    print(f"dataset_save_folder: {dataset_save_folder}")
    dataset_index_loc = f"{dataset_save_folder}/{task}.csv"
    # check if completed preprocess and has corresponding scLLM embedding
    get_preprocess_status(current_tissue_folder,task)
    img_fname = img_prefix.format(patch_size=patch_size, pixel_size=pixel_size)

    if split_dataset:
        # create train dataset
        train_dataset = BatchLocalityDataset(
            dataset_save_folder=dataset_save_folder, 
            scLLM_emb_name=scLLM_emb_name,
            random_sample_barcode=random_sample_barcode,
            img_prefix=img_fname, 
            batch_switch_interval=batch_switch_interval,
            embedding_prefix=scLLM_emb_prefix,
            scLLM_emb_ckpt=scLLM_emb_ckpt,
            label_name=label_name,
            train_phase="train",
            val_ratio=val_ratio,
            split_seed=split_seed
        )
        
        # create val dataset, use train dataset's split info
        val_dataset = BatchLocalityDataset(
            dataset_save_folder=dataset_save_folder,
            scLLM_emb_name=scLLM_emb_name,
            random_sample_barcode=False,  # val dataset doesn't need random sampling
            img_prefix=img_fname,
            batch_switch_interval=batch_switch_interval,
            embedding_prefix=scLLM_emb_prefix,
            scLLM_emb_ckpt=scLLM_emb_ckpt,
            label_name=label_name,
            train_phase="val",
            train_val_split_dict=train_dataset.train_val_split_dict,  # use train dataset's split info
            split_seed=split_seed
        )
        
        # create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,  # train dataset needs shuffle
            **additional_dataloader_para
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # val dataset doesn't need shuffle
            **additional_dataloader_para
        )
        
        logger.info(f"Train set size: {len(train_dataset)}")
        logger.info(f"Val set size: {len(val_dataset)}")
        return train_loader, val_loader, train_dataset.embedding_dim
    
    else:
        # when not split dataset
        dataset = BatchLocalityDataset(
            dataset_save_folder=dataset_save_folder,
            scLLM_emb_name=scLLM_emb_name,
            random_sample_barcode=random_sample_barcode,
            img_prefix=img_fname,
            batch_switch_interval=batch_switch_interval,
            embedding_prefix=scLLM_emb_prefix,
            scLLM_emb_ckpt=scLLM_emb_ckpt,
            label_name=label_name,
            train_phase="train"  # default train phase
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **additional_dataloader_para
        )
        
        logger.info(f"Dataset size: {len(dataset)}")
        return loader, None, dataset.embedding_dim