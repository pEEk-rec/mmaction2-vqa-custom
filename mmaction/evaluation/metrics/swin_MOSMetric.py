"""
VQA Evaluation Metrics for Video Quality Assessment (KoNViD-150k)
Computes SROCC and PLCC for MOS prediction evaluation
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import List, Sequence

from mmaction.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class swin_MOSMetric(BaseMetric):
    """
    Metric for Video Quality Assessment evaluation.
    
    Computes:
    - SROCC (Spearman Rank Correlation Coefficient) - Primary metric
    - PLCC (Pearson Linear Correlation Coefficient) - Primary metric
    
    Args:
        collect_device (str): Device for collecting results. Default: 'cpu'
        prefix (str): Prefix for metric names. Default: None
    """
    
    default_prefix = 'vqa'
    
    def __init__(self, 
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples.
        
        Args:
            data_batch (dict): Batch data from dataloader
            data_samples (Sequence[dict]): Predictions from model
        """
        for sample in data_samples:
            # Extract predicted MOS from model output
            mos_pred = sample.get('pred_scores', {}).get('mos', None)
            
            # Extract ground truth MOS from data sample
            mos_true = sample.get('gt_label', {}).get('mos', None)
            
            # Validate both values exist
            if mos_pred is not None and mos_true is not None:
                result = dict(
                    mos_pred=float(mos_pred),
                    mos_true=float(mos_true)
                )
                self.results.append(result)
    
    def compute_metrics(self, results: List[dict]) -> dict:
        """
        Compute SROCC and PLCC metrics from collected results.
        
        Args:
            results (List[dict]): Collected results from all batches
        
        Returns:
            dict: Computed metrics with keys 'mos_srocc' and 'mos_plcc'
        """
        if len(results) == 0:
            return {
                'mos_srocc': 0.0,
                'mos_plcc': 0.0
            }
        
        # Extract predictions and ground truth
        mos_preds = np.array([r['mos_pred'] for r in results])
        mos_trues = np.array([r['mos_true'] for r in results])
        
        # Validate arrays
        if len(mos_preds) != len(mos_trues):
            raise ValueError(
                f"Mismatch: {len(mos_preds)} predictions vs {len(mos_trues)} ground truths"
            )
        
        # Compute metrics
        metrics = {}
        
        # SROCC (Spearman Rank Correlation Coefficient)
        try:
            srcc, _ = spearmanr(mos_preds, mos_trues)
            metrics['mos_srocc'] = float(srcc) if not np.isnan(srcc) else 0.0
        except Exception as e:
            print(f"Warning: SROCC computation failed: {e}")
            metrics['mos_srocc'] = 0.0
        
        # PLCC (Pearson Linear Correlation Coefficient)
        try:
            plcc, _ = pearsonr(mos_preds, mos_trues)
            metrics['mos_plcc'] = float(plcc) if not np.isnan(plcc) else 0.0
        except Exception as e:
            print(f"Warning: PLCC computation failed: {e}")
            metrics['mos_plcc'] = 0.0
        
        return metrics
