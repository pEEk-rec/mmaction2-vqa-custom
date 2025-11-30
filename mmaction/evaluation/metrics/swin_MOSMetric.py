"""
VQA Evaluation Metrics for Video Quality Assessment (KoNViD-150k)
Computes SROCC and PLCC for MOS prediction evaluation
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import List, Sequence
import torch

from mmaction.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class swin_MOSMetric(BaseMetric):
    """
    Metric for Video Quality Assessment evaluation.
    
    Computes:
    - SROCC (Spearman Rank Correlation Coefficient)
    - PLCC (Pearson Linear Correlation Coefficient)
    - RMSE (Root Mean Square Error)
    """
    
    default_prefix = 'swin_MOS'
    
    def __init__(self, 
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
    
    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        """Process one batch of data samples."""
        for sample in data_samples:
            # Handle both dict and ActionDataSample object
            if isinstance(sample, dict):
                # Dict format (validation output)
                mos_pred = sample.get('pred_mos')
                mos_true = sample.get('gt_mos')
                
                if mos_true is None:
                    mos_true = sample.get('mos')
                
                # Convert tensors if needed
                if isinstance(mos_true, torch.Tensor):
                    mos_true = mos_true.cpu().item()
                if isinstance(mos_pred, torch.Tensor):
                    mos_pred = mos_pred.cpu().item()
                    
            else:
                # ActionDataSample object
                if hasattr(sample, 'pred_mos'):
                    mos_pred = sample.pred_mos
                else:
                    raise ValueError(f"Sample missing pred_mos: {sample}")
                
                if hasattr(sample, 'gt_mos'):
                    mos_true = sample.gt_mos
                elif hasattr(sample, 'mos'):
                    mos_true = sample.mos
                else:
                    raise ValueError(f"Sample missing gt_mos: {sample}")
                
                # Convert tensors
                if isinstance(mos_true, torch.Tensor):
                    mos_true = mos_true.cpu().item()
                if isinstance(mos_pred, torch.Tensor):
                    mos_pred = mos_pred.cpu().item()
            
            # Validate
            if mos_pred is None or mos_true is None:
                continue
            
            # Store results
            result = dict(
                mos_pred=float(mos_pred),
                mos_true=float(mos_true)
            )
            self.results.append(result)
    
    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute SROCC, PLCC, and RMSE metrics."""
        if len(results) == 0:
            return {'SROCC': 0.0, 'PLCC': 0.0, 'RMSE': 0.0}
        
        # Extract predictions and ground truth
        mos_preds = np.array([r['mos_pred'] for r in results])
        mos_trues = np.array([r['mos_true'] for r in results])
        
        metrics = {}
        
        # SROCC
        try:
            srcc, _ = spearmanr(mos_preds, mos_trues)
            metrics['SROCC'] = float(srcc) if not np.isnan(srcc) else 0.0
        except:
            metrics['SROCC'] = 0.0
        
        # PLCC
        try:
            plcc, _ = pearsonr(mos_preds, mos_trues)
            metrics['PLCC'] = float(plcc) if not np.isnan(plcc) else 0.0
        except:
            metrics['PLCC'] = 0.0
        
        # RMSE
        try:
            rmse = np.sqrt(np.mean((mos_preds - mos_trues) ** 2))
            metrics['RMSE'] = float(rmse)
        except:
            metrics['RMSE'] = 0.0
        
        return metrics
