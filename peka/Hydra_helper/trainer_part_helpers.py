from peka import logger
import os
import pytorch_lightning as pl
from typing import Optional

def trainer_config(
        # Basic configurations
        project: str,
        entity: str,
        exp_name: str,
        task_type: str,

        class_nb: int,  # declare class_nb as int

        # Model
        model_name: str,
        ckpt_folder: str,

        # Model and training components
        # Gradient clipping
        clip_grad: float,
        # Maximum number of epochs
        max_epochs: int,
        # Training output directory
        trainer_output_dir: str,
        additional_pl_paras: dict, 
        # Logger
        with_logger: str,
        # wandb api key
        wandb_api_key: str,
        # Save model
        save_ckpt: bool,
        # Model save format
        ckpt_format: str = "{epoch:02d}-{accuracy_val:.2f}",
        # Model save parameters
        ckpt_para: dict = {
            "save_top_k": 1,
            "mode": "max",
            "monitor": "accuracy_val",
        }
    ):

    logger.info(f"build trainer for model.")
    trainer_additional_dict = additional_pl_paras
    callbacks_list = []

    # 3. clip gradient
    if clip_grad is not None:
        logger.debug(f"clip gradient with value {clip_grad}")
        trainer_additional_dict.update({"gradient_clip_val":clip_grad,
                                        "gradient_clip_algorithm":"value"})

    # 4. create learning rate logger
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_list.append(lr_monitor)
    if with_logger=="wandb":
        logger.debug(f"with logger wandb")
        # 4. Create wandb logger
        from pytorch_lightning.loggers import WandbLogger
        os.environ["WANDB_API_KEY"]=wandb_api_key

        wandb_logger = WandbLogger(project=project,
                                    entity=entity,
                                    name=exp_name)
        trainer_additional_dict.update({"logger":wandb_logger})

    if save_ckpt:
        # 4. check point
        from pytorch_lightning.callbacks import ModelCheckpoint
        # init ckpt related paras
        ckpt_dir = ckpt_folder+"/"+exp_name+"/"
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger.debug(f"for exp {exp_name} \
                                    Checkpoint with paras {ckpt_para}")
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                            filename=ckpt_format,
                                                **ckpt_para,
                                            )
        
        logger.info(f"Best model will be saved at {ckpt_dir} as {ckpt_format}")
        callbacks_list.append(checkpoint_callback)
        
    if len(callbacks_list)>=1: trainer_additional_dict.update(
                                        {"callbacks":callbacks_list})
    # 4. Trainer and fit
    trainer = pl.Trainer(default_root_dir=trainer_output_dir,
                            max_epochs=max_epochs,
                            **trainer_additional_dict
                            )
    logger.info(f"trainer is built.")
    return trainer