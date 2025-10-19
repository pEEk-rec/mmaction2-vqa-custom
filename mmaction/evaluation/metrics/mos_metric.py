# mmaction2/mmaction/evaluation/metrics/mos_metric.py

from typing import List, Optional, Sequence
import numpy as np
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS

def mos_to_quality_label(mos):
    """Convert MOS to quality class."""
    if mos < 2.0:
        return 0  # Bad
    elif mos < 3.0:
        return 1  # Poor
    elif mos < 4.0:
        return 2  # Fair
    elif mos < 4.5:
        return 3  # Good
    else:
        return 4  # Excellent

@METRICS.register_module()
class MOSMetric(BaseMetric):
    """MOS evaluation metric for video quality assessment.
    
    Computes SRCC, PLCC, RMSE, MAE, and classification accuracy.
    
    Args:
        collect_device (str): Device for collecting results.
        prefix (str): Prefix for metric names.
    """
    
    default_prefix: Optional[str] = 'mos'
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples."""
        for data_sample in data_samples:
            result = dict()
            
            # Get predicted MOS
            pred_scores = data_sample.get('pred_scores')
            if pred_scores is not None:
                if isinstance(pred_scores, dict) and 'mos' in pred_scores:
                    mos_pred = pred_scores['mos']
                else:
                    mos_pred = pred_scores
                
                if hasattr(mos_pred, 'cpu'):
                    mos_pred = mos_pred.cpu().item()
                result['mos_pred'] = float(mos_pred)
            
            # Get ground truth MOS
            gt_label = data_sample.get('gt_label')
            if gt_label is not None:
                if hasattr(gt_label, 'mos'):
                    mos_gt = gt_label.mos
                else:
                    mos_gt = gt_label
                
                if hasattr(mos_gt, 'cpu'):
                    mos_gt = mos_gt.cpu().item()
                elif hasattr(mos_gt, 'item'):
                    mos_gt = mos_gt.item()
                
                result['mos_gt'] = float(mos_gt)
            
            if 'mos_pred' in result and 'mos_gt' in result:
                self.results.append(result)
    
    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute the metrics from processed results.
        
        Args:
            results: The processed results of each batch.
            
        Returns:
            dict: The computed metrics.
        """
        try:
            from scipy.stats import spearmanr, pearsonr
        except ImportError:
            raise ImportError('scipy is required for MOS evaluation. Install with: pip install scipy')
        
        # Extract predictions and ground truth
        mos_preds = np.array([res['mos_pred'] for res in results])
        mos_gts = np.array([res['mos_gt'] for res in results])
        
        metrics_results = {}
        
        # SRCC (Spearman Rank Correlation Coefficient)
        srcc, _ = spearmanr(mos_preds, mos_gts)
        metrics_results['SRCC'] = float(srcc)
        
        # PLCC (Pearson Linear Correlation Coefficient)
        plcc, _ = pearsonr(mos_preds, mos_gts)
        metrics_results['PLCC'] = float(plcc)
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((mos_preds - mos_gts) ** 2))
        metrics_results['RMSE'] = float(rmse)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(mos_preds - mos_gts))
        metrics_results['MAE'] = float(mae)
        
        # Classification accuracy (derived from MOS)
        class_preds = np.array([mos_to_quality_label(m) for m in mos_preds])
        class_gts = np.array([mos_to_quality_label(m) for m in mos_gts])
        accuracy = np.mean(class_preds == class_gts)
        metrics_results['accuracy'] = float(accuracy)
        
        return metrics_results

# Alias for backward compatibility
MOSEvaluator = MOSMetric