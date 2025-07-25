"""
Script to reproduce an experiment using saved configurations.
"""
import sys
import os
import argparse
from datetime import datetime

# Setup paths
notebook_path = os.getcwd()
root = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
code_root = root + "/PEKA/"
env_file_path = f"{code_root}/.env"
data_root = code_root + "DATA/"
output_dir = f"{root}/OUTPUT/"

sys.path.append(root)
sys.path.append(f'/{code_root}/')
sys.path.append(f'/{code_root}/peka/External_models/')
sys.path.append(f'/{code_root}/peka/External_models/HEST/src/')

import peka
import dotenv
dotenv.load_dotenv(env_file_path)

import pytorch_lightning as pl
import torch
from hydra_zen import instantiate
from peka.Trainer.KD_LoRA import KD_LoRA_Model
from peka.Hydra_helper.experiment_helpers import load_experiment_configs, find_latest_experiment

def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce a saved experiment')
    parser.add_argument('--experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--latest', action='store_true', help='Use latest experiment')
    parser.add_argument('--prefix', type=str, help='Prefix to filter experiments when using --latest')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Find experiment directory
    if args.latest:
        experiment_dir = find_latest_experiment(output_dir, args.prefix)
    else:
        if args.experiment_dir is None:
            raise ValueError("Must provide either --experiment_dir or --latest")
        experiment_dir = args.experiment_dir
    
    # Load configurations
    configs = load_experiment_configs(experiment_dir)
    
    # Create new experiment directory for reproduction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reproduction_dir = os.path.join(output_dir, f"reproduction_{os.path.basename(experiment_dir)}_{timestamp}")
    os.makedirs(reproduction_dir)
    
    # Get number of classes from dataset
    num_classes = 100
    USE_SPLIT_DATASET = True
    
    # Load dataset
    train_loader, val_loader, target_dim = instantiate(
        configs['dataset'],
        data_root=data_root,
        split_dataset=USE_SPLIT_DATASET,
        val_ratio=0.2,
        split_seed=42,
        random_sample_barcode=True if not USE_SPLIT_DATASET else False
    )
    
    # Phase 1: Train MLP classifier
    teacher_classifier = KD_LoRA_Model.train_phase1(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=target_dim,
        classifier_hidden_dim=512,
        num_classes=num_classes,
        save_path=os.path.join(reproduction_dir, "phase1", "classifier.pt"),
        device='cuda',
        num_epochs=20,
        learning_rate=1e-4
    )
    
    # Initialize student model with LoRA
    model = instantiate(configs['model'], target_dim=target_dim)
    
    # Phase 2: Train LoRA model with knowledge distillation
    kd_model = KD_LoRA_Model(
        model_instance=model,
        loss_instance=None,
        metrics_factory=None,
        optimizer_instance_list=[torch.optim.Adam(model.parameters(), lr=1e-4)],
        num_classes=num_classes,
        classifier_hidden_dim=512,
        input_dim=target_dim,
        temperature=2.0,
        alpha=0.5,
        lora_save_path=os.path.join(reproduction_dir, "phase2", "lora")
    )
    
    # Setup teacher model
    kd_model.setup_teacher_model(teacher_classifier)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=1,
        default_root_dir=os.path.join(reproduction_dir, "phase2"),
    )
    
    # Train model
    trainer.fit(kd_model, train_loader, val_loader)
    
    print(f"✨ Experiment reproduced in: {reproduction_dir}")

if __name__ == "__main__":
    main()
