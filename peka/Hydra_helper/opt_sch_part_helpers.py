import torch
from typing import Optional
from functools import partial

from peka import logger
def dynamic_call(function_path, kwargs):
    # try to separate module path and function name
    try:
        module_path, function_name = function_path.rsplit('.', 1)
    except ValueError:
        raise ValueError(f"Invalid function path '{function_path}'.")
    
    # dynamically import module and get function/class
    module = __import__(module_path, fromlist=[function_name])
    func = getattr(module, function_name)
    
    # call function/class and pass parameters
    return func(**kwargs)

def opt_sch_config(
        metrics_names: list,
        metrics_paras: dict,
        loss_name: str,
        loss_paras: dict,
        optimizer_name_list: list,
        optimizer_paras_list: list[dict],
        scheduler_name_list: list,
        scheduler_paras_list: list[dict],
        n_classes: Optional[int] = None,
    ):
    from peka.Trainer.metrics import MetricsFactory
    metrics_factory = MetricsFactory(metrics_names, n_classes, metrics_paras)

    logger.info(f"loss_name: {loss_name} with loss_paras: {loss_paras}")
    
    # use independent function to get loss function
    loss_instance = get_loss_fn(loss_name, loss_paras)

    optimizer_instance_list = []
    scheduler_instance_list = []
    
    # use independent function to get optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(optimizer_name_list[0], optimizer_paras_list[0], scheduler_name_list[0], scheduler_paras_list[0])
    optimizer_instance_list.append(optimizer)
    scheduler_instance_list.append(scheduler)

    return optimizer_instance_list, scheduler_instance_list, metrics_factory, loss_instance




def get_optimizer_scheduler(optimizer_name, optimizer_params, scheduler_name=None, scheduler_params=None):
    # from name to class
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer = partial(optimizer_cls, **optimizer_params)
    if scheduler_name is not None:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = partial(scheduler_cls, **scheduler_params)
    else:
        scheduler = None
    return optimizer, scheduler


def get_loss_fn(loss_name, loss_params):
    loss_cls = getattr(torch.nn, loss_name)
    loss_fn = partial(loss_cls, **loss_params)
    return loss_fn
