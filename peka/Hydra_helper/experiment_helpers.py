"""
Helper functions for experiment configuration management.
"""
import os
import shutil
from datetime import datetime
import yaml
from typing import Dict, Optional
from hydra_zen import load_from_yaml

def save_experiment_configs(output_dir: str,
                          config_files: dict,
                          experiment_name: str = None):
    """
    Save experiment configurations to the output directory with timestamp.
    
    Args:
        output_dir: Base output directory
        config_files: Dictionary of config names and their file paths
        experiment_name: Optional name for the experiment
    
    Returns:
        str: Path to the experiment directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name if not provided
    if experiment_name is None:
        experiment_name = "experiment"
    
    # Create experiment directory with timestamp
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    config_dir = os.path.join(experiment_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Save each config file
    for config_name, config_path in config_files.items():
        # Create subdirectory for config type
        config_type_dir = os.path.join(config_dir, config_name)
        os.makedirs(config_type_dir, exist_ok=True)
        
        # Copy config file
        config_filename = os.path.basename(config_path)
        dest_path = os.path.join(config_type_dir, config_filename)
        shutil.copy2(config_path, dest_path)
        
        # Also save as YAML if not already YAML
        if not config_filename.endswith('.yaml'):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            yaml_path = os.path.join(config_type_dir, f"{os.path.splitext(config_filename)[0]}.yaml")
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
    
    print(f"✨ Experiment configs saved to: {experiment_dir}")
    return experiment_dir

def load_experiment_configs(experiment_dir: str) -> Dict[str, any]:
    """
    Load experiment configurations from a saved experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        dict: Dictionary containing loaded configurations
    """
    print(f"🔄 Loading experiment configs from: {experiment_dir}")
    
    config_dir = os.path.join(experiment_dir, "configs")
    if not os.path.exists(config_dir):
        raise ValueError(f"Config directory not found: {config_dir}")
    
    configs = {}
    
    # Load each config type
    for config_type in os.listdir(config_dir):
        type_dir = os.path.join(config_dir, config_type)
        if os.path.isdir(type_dir):
            # Find yaml file in the directory
            yaml_files = [f for f in os.listdir(type_dir) if f.endswith('.yaml')]
            if yaml_files:
                yaml_path = os.path.join(type_dir, yaml_files[0])
                configs[config_type] = load_from_yaml(yaml_path)
    
    print("📚 Loaded configurations:")
    for config_type in configs.keys():
        print(f"  - {config_type}")
    
    return configs

def find_latest_experiment(base_dir: str, experiment_prefix: Optional[str] = None) -> str:
    """
    Find the latest experiment directory in the base directory.
    
    Args:
        base_dir: Base directory containing experiment directories
        experiment_prefix: Optional prefix to filter experiments
        
    Returns:
        str: Path to the latest experiment directory
    """
    experiments = []
    
    # List all directories in base_dir
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if directory matches prefix
            if experiment_prefix is None or item.startswith(experiment_prefix):
                # Check if it has configs directory
                if os.path.exists(os.path.join(item_path, "configs")):
                    experiments.append(item_path)
    
    if not experiments:
        raise ValueError(f"No experiment directories found in: {base_dir}")
    
    # Sort by modification time and return the latest
    latest = max(experiments, key=os.path.getmtime)
    print(f"📂 Found latest experiment: {os.path.basename(latest)}")
    return latest
