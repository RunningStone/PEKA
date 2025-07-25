"""
A basic class for different PL protocols
"""
import pytorch_lightning as pl
import os,sys
import torch

from peka import logger

class pl_basic(pl.LightningModule):
    def __init__(self,
                model_instance,  # model instance
                loss_instance,
                metrics_factory,

                optimizer_instance_list:list,
                scheduler_instance_list:list = None,
                ):
        super(pl_basic, self).__init__()
        """
        A basic class for different PL protocols:
        """
        logger.info("init pytorch-lightning basic part...")
        # ----> create model
        self.model = model_instance

        # ----> create loss
        self.loss_fn = loss_instance
        # ----> create optimizer and scheduler
        self.optimizer_instance_list = optimizer_instance_list
        self.scheduler_instance_list = scheduler_instance_list
        
        # ----> create metrics
        self.metrics_factory = metrics_factory
        self.create_metrics()

        self.train_epoch_datas = []
        self.val_epoch_datas = []

    def configure_optimizers(self):
        # opt_instances is a list of partial functions, need to pass in model parameters
        opt_list = [opt_fn(self.model.parameters()) for opt_fn in self.optimizer_instance_list]
        if self.scheduler_instance_list:
            sch_list = [sch_fn(optimizer=opt) for sch_fn, opt in zip(self.scheduler_instance_list, opt_list)]
            return opt_list, sch_list
        else:
            return opt_list

    def create_metrics(self):
        self.bar_metrics = self.metrics_factory.metrics["metrics_on_bar"]
        self.valid_metrics = self.metrics_factory.metrics["metrics_template"].clone(prefix='val_')
        self.train_metrics = self.metrics_factory.metrics["metrics_template"].clone(prefix='train_')

    # modify data preprocess and post process methods
    def train_data_preprocess(self, batch):
        """
        Data preprocess for training
        """
        img_input_batch, target = batch
        return img_input_batch, target

    def train_post_process(self, preds, target, loss):
        """
        Post process for training
        """
        out = {"preds": preds, "target": target, "loss": loss}
        return out

    def val_data_preprocess(self, batch):
        """
        Data preprocess for validation
        """
        img_input_batch, target = batch
        return img_input_batch, target

    def val_post_process(self, preds, target, loss):
        """
        Post process for validation
        """
        out = {"preds": preds, "target": target, "loss": loss}
        return out

    def log_train_metrics(self, outlist, bar_name: str):
        preds = self.collect_step_output(key="preds", out=outlist, dim=0)
        target = self.collect_step_output(key="target", out=outlist, dim=0)
        # Log metrics
        self.log(bar_name, self.bar_metrics(preds, target), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics(preds, target), on_epoch=True, logger=True)

    def log_val_metrics(self, outlist, bar_name: str):
        preds = self.collect_step_output(key="preds", out=outlist, dim=0)
        target = self.collect_step_output(key="target", out=outlist, dim=0)
        # Log metrics
        self.log(bar_name, self.bar_metrics(preds, target), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(preds, target), on_epoch=True, logger=True)

    def training_step(self, batch, batch_idx):
        # Data preprocess
        img_input_batch, target = self.train_data_preprocess(batch)

        # Forward step
        preds = self.model(img_input_batch)
        # Loss computation
        loss = self.loss_fn(preds, target)

        # Post process
        out = self.train_post_process(preds, target, loss)
        self.train_epoch_datas.append(out)
        return out

    def validation_step(self, batch, batch_idx):
        # Data preprocess
        img_input_batch, target = self.val_data_preprocess(batch)

        # Forward step
        preds = self.model(img_input_batch)

        # Loss computation
        loss = self.loss_fn(preds, target)

        # Post process
        out = self.val_post_process(preds, target, loss)
        self.val_epoch_datas.append(out)
        return out
    
    def collect_step_output(self, key, out, dim):
        return torch.cat([item[key] for item in out], dim=dim)

    def on_train_epoch_end(self,):
        train_epoch_datas = self.train_epoch_datas
        self.log_train_metrics(train_epoch_datas,
            bar_name = self.metrics_factory.metrics_names[0]+"_train")
        self.train_epoch_datas = []
        # reset metrics
        self.bar_metrics.reset()
        self.train_metrics.reset()

    def on_validation_epoch_end(self,):
        val_epoch_datas = self.val_epoch_datas
        self.log_val_metrics(val_epoch_datas,
            bar_name = self.metrics_factory.metrics_names[0]+"_val")
        self.val_epoch_datas = []
        # reset metrics
        self.bar_metrics.reset()
        self.valid_metrics.reset()