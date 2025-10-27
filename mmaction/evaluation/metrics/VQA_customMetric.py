# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence
import numpy as np
from scipy.stats import spearmanr, pearsonr
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS


@METRICS.register_module()
class VQAMetric(BaseMetric):
    """Video Quality Assessment Metric.
    
    Computes:
    - MOS: MAE, RMSE, SRCC, PLCC
    - Quality Class: Accuracy
    - Distortion Type: Accuracy
    """
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
    
    def get_meta(self, ds):
        """Extract metainfo or dict from data sample."""
        # If it's a dict, return as-is
        if isinstance(ds, dict):
            return ds
        
        # If it's an object with top-level attributes, create dict
        result = {}
        
        # Try top-level attributes first (set by predict)
        for key in ['pred_mos', 'pred_qclass', 'pred_dtype', 'mos', 'quality_class', 'distortion_type']:
            if hasattr(ds, key):
                result[key] = getattr(ds, key)
        
        # Also try metainfo
        if hasattr(ds, 'metainfo'):
            result.update(ds.metainfo)
        
        return result
    
    def process(self, data_batch, data_samples):
        print("="*60)
        print("[METRIC] process() called")
        print(f"[METRIC] Num samples: {len(data_samples)}")
        
        for idx, ds in enumerate(data_samples):
            print(f"\n[METRIC] Sample {idx}:")
            print(f"  Type: {type(ds)}")
            
            meta = self.get_meta(ds)
            print(f"  Metainfo keys: {list(meta.keys())}")
            
            pm = meta.get('pred_mos', None)
            print(f"  pred_mos: {pm}")
            
            if pm is None:
                print(f"  [WARNING] Skipping - no pred_mos found")
                continue
            
            try:
                item = {
                    'pred_mos': float(pm),
                    'gt_mos': float(meta['mos']),
                    'pred_qclass': int(meta['pred_qclass']),
                    'gt_qclass': int(meta['quality_class']),
                    'pred_dtype': int(meta['pred_dtype']),
                    'gt_dtype': int(meta['distortion_type']),
                }
                self.results.append(item)
                print(f"  [SUCCESS] Added to results")
            except (KeyError, ValueError, TypeError) as e:
                print(f"  [ERROR] {e}")
                continue
        
        print(f"\n[METRIC] Total results collected: {len(self.results)}")
        print("="*60)

    
    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute metrics from collected results.
        
        Args:
            results: List of result dicts from process()
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Collect MOS predictions and ground truth
        pred_mos_list = []
        gt_mos_list = []
        for r in results:
            if r['pred_mos'] is not None and r['gt_mos'] is not None:
                pred_mos_list.append(float(r['pred_mos']))
                gt_mos_list.append(float(r['gt_mos']))
        
        # Compute MOS metrics
        if len(pred_mos_list) > 0:
            pred_mos = np.array(pred_mos_list)
            gt_mos = np.array(gt_mos_list)
            
            # MAE
            mae = np.mean(np.abs(pred_mos - gt_mos))
            metrics['mos_mae'] = float(mae)
            
            # RMSE
            rmse = np.sqrt(np.mean((pred_mos - gt_mos) ** 2))
            metrics['mos_rmse'] = float(rmse)
            
            # SRCC (Spearman correlation)
            if len(pred_mos) > 1:
                srcc, _ = spearmanr(pred_mos, gt_mos)
                metrics['mos_srcc'] = float(srcc) if not np.isnan(srcc) else 0.0
            else:
                metrics['mos_srcc'] = 0.0
            
            # PLCC (Pearson correlation)
            if len(pred_mos) > 1:
                plcc, _ = pearsonr(pred_mos, gt_mos)
                metrics['mos_plcc'] = float(plcc) if not np.isnan(plcc) else 0.0
            else:
                metrics['mos_plcc'] = 0.0
        else:
            metrics['mos_mae'] = 0.0
            metrics['mos_rmse'] = 0.0
            metrics['mos_srcc'] = 0.0
            metrics['mos_plcc'] = 0.0
        
        # Collect quality class predictions
        pred_qclass_list = []
        gt_qclass_list = []
        for r in results:
            if r['pred_qclass'] is not None and r['gt_qclass'] is not None:
                pred_qclass_list.append(int(r['pred_qclass']))
                gt_qclass_list.append(int(r['gt_qclass']))
        
        # Compute quality class accuracy
        if len(pred_qclass_list) > 0:
            pred_qclass = np.array(pred_qclass_list)
            gt_qclass = np.array(gt_qclass_list)
            qclass_acc = np.mean(pred_qclass == gt_qclass)
            metrics['qclass_acc'] = float(qclass_acc)
        else:
            metrics['qclass_acc'] = 0.0
        
        # Collect distortion type predictions
        pred_dtype_list = []
        gt_dtype_list = []
        for r in results:
            if r['pred_dtype'] is not None and r['gt_dtype'] is not None:
                pred_dtype_list.append(int(r['pred_dtype']))
                gt_dtype_list.append(int(r['gt_dtype']))
        
        # Compute distortion type accuracy
        if len(pred_dtype_list) > 0:
            pred_dtype = np.array(pred_dtype_list)
            gt_dtype = np.array(gt_dtype_list)
            dtype_acc = np.mean(pred_dtype == gt_dtype)
            metrics['dtype_acc'] = float(dtype_acc)
        else:
            metrics['dtype_acc'] = 0.0
        
        return metrics