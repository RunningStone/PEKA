import os
import pytorch_lightning as pl

from peka import logger

def build_trainer(trainer_paras):
    logger.info(f"build trainer for model.")
    trainer_additional_dict = trainer_paras.additional_pl_paras
    callbacks_list = []

    # 3. clip gradient
    if trainer_paras.clip_grad is not None:
        logger.debug(f"clip gradient with value {trainer_paras.clip_grad}")
        trainer_additional_dict.update({"gradient_clip_val":trainer_paras.clip_grad,
                                        "gradient_clip_algorithm":"value"})

    # 4. create learning rate logger
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_list.append(lr_monitor)
    if trainer_paras.with_logger=="wandb":
        logger.debug(f"with logger wandb")
        # 4. Create wandb logger
        from pytorch_lightning.loggers import WandbLogger
        os.environ["WANDB_API_KEY"]=trainer_paras.wandb_api_key

        wandb_logger = WandbLogger(project=trainer_paras.project,
                                    entity=trainer_paras.entity,
                                    name=trainer_paras.exp_name)
        trainer_additional_dict.update({"logger":wandb_logger})

    if trainer_paras.save_ckpt:
        # 4. check point
        from pytorch_lightning.callbacks import ModelCheckpoint
        # init ckpt related paras
        ckpt_paras = trainer_paras.ckpt_para
        ckpt_name = trainer_paras.ckpt_format
        ckpt_dir = trainer_paras.ckpt_folder+"/"+trainer_paras.exp_name+"/"
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger.debug(f"for exp {trainer_paras.exp_name} \
                                    Checkpoint with paras {ckpt_paras}")
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                            filename=ckpt_name,
                                                **ckpt_paras,
                                            )
        
        logger.info(f"Best model will be saved at {ckpt_dir} as {ckpt_name}")
        callbacks_list.append(checkpoint_callback)
        
    if len(callbacks_list)>=1: trainer_additional_dict.update(
                                        {"callbacks":callbacks_list})
    # 4. Trainer and fit
    trainer = pl.Trainer(default_root_dir=trainer_paras.trainer_output_dir,
                            max_epochs=trainer_paras.max_epochs,
                            **trainer_additional_dict
                            )
    logger.info(f"trainer is built.")
    return trainer
