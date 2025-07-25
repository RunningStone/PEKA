"""
This file is used to process scLLM embedding
"""


import sys
import os
import pandas as pd
import shutil
from typing import Optional, Dict, Any

def setup_paths(project_root: str) -> None:
    """set project paths"""
    sys.path.append(f'{project_root}/')
    sys.path.append(f'{project_root}/PEKA/')
    sys.path.append(f'{project_root}/PEKA/peka/External_models/')

def get_pretrained_config(project_root: str) -> pd.DataFrame:
    """get or create pretrained config file"""
    pretrained_dir = f"{project_root}/DATA/Pretrained/"
    config_file = os.path.join(pretrained_dir, "pretrained_ckpt_idx.csv")
    
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    
    if not os.path.exists(config_file):
        source_config = f"{project_root}/PEKA/support_files/scLLM_configs.csv"
        df = pd.read_csv(source_config)
        df['local_ready'] = False
        df.to_csv(config_file, index=False)
    
    return pd.read_csv(config_file)

def parse_additional_info(info_str: str) -> Dict[str, Any]:
    """parse additional_info field"""
    if pd.isna(info_str):
        return {}
    
    params = {}
    pairs = info_str.split()
    for pair in pairs:
        key, value = pair.split("=")
        # try to convert value to appropriate type
        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
        except:
            pass
        params[key] = value
    return params

def get_model_config(config_df: pd.DataFrame, scLLM_embedder_name: str) -> Dict[str, Any]:
    """get model config"""
    model_configs = config_df[config_df['Algorithm'] == scLLM_embedder_name]
    if len(model_configs) == 0:
        raise ValueError(f"No config found for model {scLLM_embedder_name}")
    
    # use the first matching config
    config = model_configs.iloc[0]
    params = {
        'ckpt_name': config['ckpt_name']
    }
    
    # add additional parameters
    if not pd.isna(config['additional_info']):
        params.update(parse_additional_info(config['additional_info']))
    
    return params

def main():
    # get environment variables
    project_root = os.getenv('PROJECT_ROOT')
    tissue_type = os.getenv('TISSUE_TYPE')
    dataset_name = os.getenv('DATASET_NAME')
    scLLM_embedder_name = os.getenv('SCLLM_EMBEDDER_NAME')
    
    if not all([project_root, dataset_name, scLLM_embedder_name]):
        raise ValueError("Required environment variables are not set")
    
    # set paths
    setup_paths(project_root)
    
    # get config
    config_df = get_pretrained_config(project_root)
    model_params = get_model_config(config_df, scLLM_embedder_name)
    
    # get optional parameters
    optional_params = {}
    for key in model_params.keys():
        env_value = os.getenv(key.upper())
        if env_value is not None:
            optional_params[key] = env_value
        else:
            optional_params[key] = model_params[key]
    print("optional_params: ", optional_params)
    
    # import and run embedder
    from peka.Model.LLM.utils import get_scLLM_embedder
    embedder = get_scLLM_embedder(
        data_root=f"{project_root}/PEKA/DATA/{tissue_type}",
        dataset_name=dataset_name,
        scLLM_embedder_name=scLLM_embedder_name,
        #ckpt_name = optional_params['ckpt_name'],
        **optional_params
    )
    
    embedder.run()
    embedder.valid_check()

if __name__ == "__main__":
    main()