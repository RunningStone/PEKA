import sys
import os
import argparse
import datetime
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
    parser = argparse.ArgumentParser(description='Knowledge Distillation with LoRA Training')
    parser.add_argument("--dataset_config", type=str, required=True,
                      help='Path to dataset config file relative to hydra_zen/Configs/')
    parser.add_argument("--model_config", type=str, required=True,
                      help='Path to model config file relative to hydra_zen/Configs/')
    parser.add_argument("--optimizer_config", type=str, required=True,
                      help='Path to optimizer config file relative to hydra_zen/Configs/')
    parser.add_argument("--trainer_config", type=str, required=True,
                      help='Path to trainer config file relative to hydra_zen/Configs/')
    parser.add_argument("--phase1_ckpt", type=str, default='',
                      help='Path to phase 1 checkpoint file. If provided, phase 1 training will be skipped.')
    parser.add_argument("--phase1_epochs", type=int, default=20,
                      help='Number of epochs for phase 1 training')
    parser.add_argument("--phase1_lr", type=float, default=1e-4,
                      help='Learning rate for phase 1 training')
    parser.add_argument("--phase1_hidden_dim", type=int, default=512,
                      help='Hidden dimension for phase 1 classifier')
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for wandb logging")
    return parser.parse_args()

def train_phase_2(
    train_loader,
    val_loader,
    teacher_classifier,
    output_dir,
    model,
    optimizer_instance_list,
    scheduler_instance_list,
    metrics_factory,
    loss_instance,
    trainer_config,
    pl_model_config
):
    """Phase 2: Train LoRA model with knowledge distillation"""
    print(f" Phase 2: Training LoRA model with knowledge distillation...")
    
    # update pl_model_config
    pl_model_config.lora_save_path = os.path.join(output_dir, "phase2", "lora")
    
    # create KD_LoRA model
    kd_model = create_pl_model(
        model_instance=model,
        optimizer_instance_list=optimizer_instance_list,
        scheduler_instance_list=scheduler_instance_list,
        metrics_factory=metrics_factory,
        loss_instance=loss_instance,
        pl_model_config=pl_model_config,
        model_type="kd_lora"
    )
    
    # set teacher model
    kd_model.setup_teacher_model(teacher_classifier)
    
    # create trainer
    trainer = instantiate(trainer_config,
        # Basic configurations
        entity=WANDB_ENTITY,
        exp_name=get_args().exp_name,
        task_type="classification",
        class_nb=pl_model_config.num_classes,
        
        # model
        model_name="H-optimus-0_LoRA_MLP",
        ckpt_folder=ckpt_folder,
        
        # model and training components
        max_epochs=20,
        trainer_output_dir=output_dir,
        additional_pl_paras={},
        
        # logging
        wandb_api_key=WANDB_API_KEY,
    )
    
    # train model
    trainer.fit(kd_model, train_loader, val_loader)
    
    return kd_model

def main():
    args = get_args()
    
    # load configs
    print(" Loading Zen Configs...")
    dataset_config = load_from_yaml(os.path.join(all_configs_path, args.dataset_config))
    model_config = load_from_yaml(os.path.join(all_configs_path, args.model_config))
    optimizers_config_zen = load_from_yaml(os.path.join(all_configs_path, "Optimizers/kd_lora.yaml"))
    trainer_config = load_from_yaml(os.path.join(all_configs_path, args.trainer_config))
    pl_model_config = load_from_yaml(os.path.join(all_configs_path, "PL_Model/kd_lora.yaml"))
    
    # set output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    database_name = os.path.splitext(os.path.basename(args.dataset_config))[0].split('_')[0:2]
    database_name = '_'.join(database_name)
    output_dir = OUTPUT_ROOT 
    
    # save experiment configs
    config_files = {
        'dataset': os.path.join(all_configs_path, args.dataset_config),
        'model': os.path.join(all_configs_path, args.model_config),
        'optimizer': os.path.join(all_configs_path, "Optimizers/kd_lora.yaml"),
        'trainer': os.path.join(all_configs_path, args.trainer_config),
        'pl_model': os.path.join(all_configs_path, "PL_Model/kd_lora.yaml")
    }
    experiment_dir = save_experiment_configs(
        output_dir=output_dir,
        config_files=config_files,
        experiment_name=f"kd_lora_{database_name}"
    )
    output_dir = experiment_dir
    
    # load dataset
    train_loader, val_loader, target_dim = instantiate(
        dataset_config,
        data_root=data_root,
        split_dataset=True,
        val_ratio=0.2,
        split_seed=42,
        random_sample_barcode=False
    )
    
    # instantiate model and optimizer
    model = instantiate(model_config, target_dim=target_dim)
    
    # print model trainable parameters with MB
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6} MB")
    
    # get optimizer and other instances
    optimizer_instance_list, scheduler_instance_list, metrics_factory, loss_instance = \
        instantiate(optimizers_config_zen)

    
    # Phase 1: load or train MLP classifier
    if args.phase1_ckpt:
        print(f"Loading phase 1 checkpoint from {args.phase1_ckpt}")
        teacher_classifier = pl_KD_LoRA.load_phase1_model(
            checkpoint_path=args.phase1_ckpt,
            input_dim=target_dim,
            classifier_hidden_dim=args.phase1_hidden_dim,
            num_classes=pl_model_config.num_classes,
            device='cuda'
        )
    else:
        print("No phase 1 checkpoint provided. Training phase 1...")
        teacher_classifier = pl_KD_LoRA.train_phase1(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=target_dim,
            classifier_hidden_dim=args.phase1_hidden_dim,
            num_classes=pl_model_config.num_classes,
            save_path=os.path.join(output_dir, "phase1", "classifier.pt"),
            device='cuda',
            num_epochs=args.phase1_epochs,
            learning_rate=args.phase1_lr
        )
    
    # Phase 2: train LoRA model with knowledge distillation
    final_model = train_phase_2(
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_classifier=teacher_classifier,
        output_dir=output_dir,
        model=model,
        optimizer_instance_list=optimizer_instance_list,
        scheduler_instance_list=scheduler_instance_list,
        metrics_factory=metrics_factory,
        loss_instance=loss_instance,
        trainer_config=trainer_config,
        pl_model_config=pl_model_config
    )

if __name__ == "__main__":
    main()
