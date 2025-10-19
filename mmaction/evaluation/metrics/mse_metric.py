import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


@METRICS.register_module()
class MSEMetric(BaseMetric):
    """Mean Squared Error Metric for regression tasks."""

    def __init__(self, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples):
        """
        Args:
            data_batch: The batch of input data.
            data_samples: The output predictions and ground truths from the model.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_scores']
            target = data_sample['gt_label']
            self.results.append({
                'pred': float(pred),
                'target': float(target)
            })

    def compute_metrics(self, results):
        preds = np.array([res['pred'] for res in results])
        targets = np.array([res['target'] for res in results])

        mse = np.mean((preds - targets) ** 2)
        return {'MSE': mse}
