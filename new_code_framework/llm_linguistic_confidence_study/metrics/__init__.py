from acc import Accuracy
from ece import ECE
from auroc import AUROC


class MetricEvaluator:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.metric_evaluator = self.get_metric_evaluator(metric_name)

    def evaluate(self, responses_df):
        # return the float value of the metric
        return self.metric_evaluator.evaluate(responses_df)

    def get_metric_evaluator(self, metric_name):
        if metric_name == "acc":
            return Accuracy()
        elif metric_name == "ece-15":
            return ECE(bin=15)
        elif metric_name == "ece-10":
            return ECE(bin=10)
        elif metric_name == "auroc":
            return AUROC()
        else:
            raise ValueError(f"Metric {metric_name} not found")