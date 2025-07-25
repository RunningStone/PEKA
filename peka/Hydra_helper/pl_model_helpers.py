from typing import List, Optional, Union
import os
from hydra_zen import instantiate
import torch.nn as nn

from peka.Trainer.pl_basic import pl_basic
from peka.Trainer.KD_LoRA import pl_KD_LoRA
from peka import logger

def create_pl_model(
    model_instance: nn.Module,
    optimizer_instance_list: list,
    scheduler_instance_list: list,
    metrics_factory,
    loss_instance,
    pl_model_config: dict,
    model_type: str = "basic",
    **kwargs
) -> Union[pl_basic, pl_KD_LoRA]:
    """unified pl_model creation function
    
    Args:
        model_instance: already instantiated model
        optimizer_instance_list: already instantiated optimizer list
        scheduler_instance_list: already instantiated scheduler list
        metrics_factory: already instantiated metrics factory
        loss_instance: already instantiated loss function
        pl_model_config: PL model config
        model_type: model type, optional ["basic", "kd_lora"]
        **kwargs: additional parameters
        
    Returns:
        created PL model instance
    """
    # create different PL models based on model type
    if model_type == "basic":
        # create basic PL model
        pl_model = pl_basic(
            model_instance=model_instance,
            loss_instance=loss_instance(),
            metrics_factory=metrics_factory,
            optimizer_instance_list=optimizer_instance_list,
            scheduler_instance_list=scheduler_instance_list,
        )
        
    elif model_type == "kd_lora":
        # check if loss_instance is in allowed loss list [CrossEntropyLoss]
        import functools
        import torch.nn as nn
        
        if not isinstance(loss_instance, functools.partial) \
            or not issubclass(loss_instance.func, nn.CrossEntropyLoss):
            raise ValueError("KD_LoRA model requires CrossEntropyLoss.")
        
        # create KD_LoRA model
        pl_model = pl_KD_LoRA(
            model_instance=model_instance,
            loss_instance=loss_instance(),  # Loss in KD_LoRA internal processing
            metrics_factory=metrics_factory,
            optimizer_instance_list=optimizer_instance_list,
            scheduler_instance_list=scheduler_instance_list,
            num_classes=pl_model_config.num_classes,
            classifier_hidden_dim=pl_model_config.classifier_hidden_dim,
            input_dim=pl_model_config.input_dim,
            temperature=pl_model_config.temperature,
            alpha=pl_model_config.alpha,
            lora_save_path=pl_model_config.lora_save_path
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} pl_model")
    return pl_model
