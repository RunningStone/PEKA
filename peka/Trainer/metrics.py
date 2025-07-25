"""
Naive code to wrap torchmetrics
"""

import torchmetrics
from peka import logger


metrics_dict = {
    # classification
    "accuracy": torchmetrics.Accuracy,
    "cohen_kappa": torchmetrics.CohenKappa,
    "f1_score": torchmetrics.F1Score,
    "recall": torchmetrics.Recall,
    "precision": torchmetrics.Precision,
    "specificity": torchmetrics.Specificity, # class nb >2

    "auroc": torchmetrics.AUROC,
    "roc": torchmetrics.ROC,
    "confusion_matrix": torchmetrics.ConfusionMatrix,

    # regression
    "MSE":torchmetrics.MeanSquaredError,
    "CosSim":torchmetrics.CosineSimilarity,
}
# older version of torchmetrics 
metrics_para_dict_old = {
"classification":{
    "accuracy": {"average": "micro"},
    "cohen_kappa": {},
    "f1_score": {"average": "macro"},
    "recall": {"average": "macro"},
    "precision": {"average": "macro"},
    "specificity": {"average": "macro"},

    "auroc": {"average": "macro"},
    "roc": {"average": "macro"},
    "confusion_matrix": {"normalize": "true"},
},
"regression":{
    "MSE": {},
    "CosSim": {"reduction" : 'mean'},
},
}
metrics_para_dict_new= {
"classification":{
    "accuracy": {"task":'multiclass',"average": "micro"},
    "cohen_kappa": {"task":'multiclass',},
    "f1_score": {"task":'multiclass',"average": "macro"},
    "recall": {"task":'multiclass',"average": "macro"},
    "precision": {"task":'multiclass',"average": "macro"},
    "specificity": {"task":'multiclass',"average": "macro"},

    "auroc": {"task":'multiclass',"average": "macro"},
    "roc": {"task":'multiclass',"average": "macro"},
    "confusion_matrix": {"task":'multiclass',"normalize": "true"},
},

"regression":{
    "MSE": {},
    "CosSim": {"reduction" : 'mean'},
},
}
# make sure it work for older version and new version
if int(torchmetrics.__version__.split(".")[1])>=11 or int(torchmetrics.__version__.split(".")[0])>=1:
    metrics_para_dict = metrics_para_dict_new
else:
    metrics_para_dict = metrics_para_dict_old


class MetricsFactory:
    def __init__(self,  metrics_names=["auroc","accuracy"],n_classes=None,metrics_paras:dict=metrics_para_dict):
        """
        MetricsFactory to create metrics for training, validation and testing
        Args:
            n_classes (int): number of classes
            metrics_names (list, optional): list of metrics names. Defaults to ["auroc","accuracy"].
        
        """
        self.n_classes = n_classes
        task_type = "classification" if self.n_classes else "regression"
        logger.info(f"current run task type is: {task_type} with n_classes: {self.n_classes}")

        self.metrics_names = metrics_names
        self.paras = metrics_paras
        self.metrics = {}

        
        
        self.get_metrics(task_type)

    def get_metrics_classification(self):
        """
        get metrics for classification task
        """
        logger.info(f"getting metrics for classification task")
        metrics_fn_list = []
        bar_metrics = self.metrics_names[0]
        assert bar_metrics in metrics_dict.keys(), f"{bar_metrics} not in metrics_dict"
        related_paras = self.paras["classification"][bar_metrics]
        self.metrics["metrics_on_bar"] = metrics_dict[bar_metrics](num_classes =self.n_classes,
                                                                    **related_paras)
        for i in range(1,len(self.metrics_names)):
            name = self.metrics_names[i]
            related_paras = self.paras["classification"][name]
            metrics_fn_list.append(
                                    metrics_dict[name](num_classes =self.n_classes,
                                                        **related_paras)
                                    )
        metrics_template = torchmetrics.MetricCollection(metrics_fn_list)
        self.metrics["metrics_template"] = metrics_template


    def get_metrics_regression(self):
        """
        get metrics for classification task
        """
        logger.info(f"getting metrics for regression task")
        metrics_fn_list = []
        bar_metrics = self.metrics_names[0]
        assert bar_metrics in metrics_dict.keys(), f"{bar_metrics} not in metrics_dict"
        related_paras = self.paras["regression"][bar_metrics]
        self.metrics["metrics_on_bar"] = metrics_dict[bar_metrics](**related_paras)
        for i in range(1,len(self.metrics_names)):
            name = self.metrics_names[i]
            related_paras = self.paras["regression"][name]
            metrics_fn_list.append(
                                    metrics_dict[name](**related_paras)
                                    )
        metrics_template = torchmetrics.MetricCollection(metrics_fn_list)
        self.metrics["metrics_template"] = metrics_template

    def get_metrics(self,task_type:str):
        if task_type == "classification":
            self.get_metrics_classification()
        elif task_type == "regression":
            self.get_metrics_regression()
        else:
            raise NotImplementedError
