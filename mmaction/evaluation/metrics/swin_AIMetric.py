# Copyright (c) OpenMMLab. All rights reserved.
# AI-Generated Video Quality Metric for Stage 0 Frozen Encoder Probing

from typing import List, Optional
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import precision_score, recall_score, f1_score
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS


@METRICS.register_module()
class swin_AIMetric(BaseMetric):
    """
    AI-Generated Video Quality Assessment Metric for Stage 0.
    
    Evaluates frozen Swin-T encoder on:
    1. MOS Regression: MAE, RMSE, SRCC, PLCC
    2. Quality Classification (5-class): Accuracy
    3. Binary Artifact Detection: Accuracy, Precision, Recall, F1
    """
    
    default_prefix: Optional[str] = None  # FIXED: Remove prefix for checkpoint compatibility
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
    
    def _extract_value(self, ds, key, default=0):
        """Safely extract value from data sample."""
        # Check metainfo first
        if hasattr(ds, 'metainfo') and key in ds.metainfo:
            return ds.metainfo[key]
        
        # Check top-level attributes
        if hasattr(ds, key):
            return getattr(ds, key)
        
        # Check if ds is a dict
        if isinstance(ds, dict) and key in ds:
            return ds[key]
        
        return default
    
    def process(self, data_batch, data_samples):
        """Process batch and collect predictions + ground truth."""
        for idx, ds in enumerate(data_samples):
            try:
                # Extract predictions
                pred_mos = self._extract_value(ds, 'pred_mos', None)
                pred_qclass = self._extract_value(ds, 'pred_qclass', None)
                if pred_qclass is None:
                    pred_qclass = self._extract_value(ds, 'pred_quality_class', None)
                
                # Extract ground truth
                gt_mos = self._extract_value(ds, 'mos', None)
                gt_qclass = self._extract_value(ds, 'quality_class', None)
                
                # Skip if core values missing
                if pred_mos is None or gt_mos is None:
                    continue
                
                # Collect results
                item = {
                    'pred_mos': float(pred_mos),
                    'gt_mos': float(gt_mos),
                    'pred_qclass': int(pred_qclass) if pred_qclass is not None else 0,
                    'gt_qclass': int(gt_qclass) if gt_qclass is not None else 0,
                    
                    # Artifact flags
                    'pred_hallucination': int(self._extract_value(ds, 'pred_hallucination', 0)),
                    'gt_hallucination': int(self._extract_value(ds, 'hallucination_flag', 0)),
                    
                    'pred_lighting': int(self._extract_value(ds, 'pred_lighting', 0)),
                    'gt_lighting': int(self._extract_value(ds, 'lighting_flag', 0)),
                    
                    'pred_spatial': int(self._extract_value(ds, 'pred_spatial', 0)),
                    'gt_spatial': int(self._extract_value(ds, 'spatial_flag', 0)),
                    
                    'pred_rendering': int(self._extract_value(ds, 'pred_rendering', 0)),
                    'gt_rendering': int(self._extract_value(ds, 'rendering_flag', 0)),
                }
                
                self.results.append(item)
                
            except Exception as e:
                continue
    
    def _compute_binary_metrics(self, pred, gt, task_name):
        """Compute binary classification metrics."""
        metrics = {}
        
        if len(pred) == 0:
            return {
                f'{task_name}_acc': 0.0,
                f'{task_name}_precision': 0.0,
                f'{task_name}_recall': 0.0,
                f'{task_name}_f1': 0.0
            }
        
        pred = np.array(pred)
        gt = np.array(gt)
        
        # Accuracy
        acc = np.mean(pred == gt)
        metrics[f'{task_name}_acc'] = float(acc)
        
        # Precision, Recall, F1
        if len(np.unique(gt)) == 1:
            if np.unique(gt)[0] == 0:
                metrics[f'{task_name}_precision'] = 1.0 if np.sum(pred) == 0 else 0.0
                metrics[f'{task_name}_recall'] = 1.0
                metrics[f'{task_name}_f1'] = 0.0
            else:
                metrics[f'{task_name}_precision'] = 1.0 if np.sum(pred) == len(pred) else 0.0
                metrics[f'{task_name}_recall'] = metrics[f'{task_name}_precision']
                metrics[f'{task_name}_f1'] = metrics[f'{task_name}_precision']
        else:
            precision = precision_score(gt, pred, zero_division=0)
            recall = recall_score(gt, pred, zero_division=0)
            f1 = f1_score(gt, pred, zero_division=0)
            
            metrics[f'{task_name}_precision'] = float(precision)
            metrics[f'{task_name}_recall'] = float(recall)
            metrics[f'{task_name}_f1'] = float(f1)
        
        return metrics
    
    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute all metrics from collected results."""
        if not results:
            return dict(
                mos_mae=0.0, mos_rmse=0.0, mos_srcc=0.0, mos_plcc=0.0,
                qclass_acc=0.0,
                hallucination_acc=0.0, lighting_acc=0.0, spatial_acc=0.0,
                rendering_acc=0.0,
                num_samples=0
            )
        
        metrics = {}
        
        # ===== 1. MOS Regression =====
        pred_mos = np.array([float(r['pred_mos']) for r in results])
        gt_mos = np.array([float(r['gt_mos']) for r in results])
        
        # MAE
        mae = np.mean(np.abs(pred_mos - gt_mos))
        metrics['mos_mae'] = float(mae)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_mos - gt_mos) ** 2))
        metrics['mos_rmse'] = float(rmse)
        
        # SRCC (Spearman correlation)
        if len(pred_mos) > 1:
            try:
                srcc, _ = spearmanr(pred_mos, gt_mos)
                metrics['mos_srcc'] = float(srcc) if not np.isnan(srcc) else 0.0
            except:
                metrics['mos_srcc'] = 0.0
        else:
            metrics['mos_srcc'] = 0.0
        
        # PLCC (Pearson correlation)
        if len(pred_mos) > 1:
            try:
                plcc, _ = pearsonr(pred_mos, gt_mos)
                metrics['mos_plcc'] = float(plcc) if not np.isnan(plcc) else 0.0
            except:
                metrics['mos_plcc'] = 0.0
        else:
            metrics['mos_plcc'] = 0.0
        
        # ===== 2. Quality Classification (5-class) =====
        pred_qclass = np.array([int(r['pred_qclass']) for r in results])
        gt_qclass = np.array([int(r['gt_qclass']) for r in results])
        qclass_acc = np.mean(pred_qclass == gt_qclass)
        metrics['qclass_acc'] = float(qclass_acc)
        
        # ===== 3. Binary Artifact Detection =====
        artifact_tasks = [
            ('hallucination', 'pred_hallucination', 'gt_hallucination'),
            ('lighting', 'pred_lighting', 'gt_lighting'),
            ('spatial', 'pred_spatial', 'gt_spatial'),
            ('rendering', 'pred_rendering', 'gt_rendering')
        ]
        
        for task_name, pred_key, gt_key in artifact_tasks:
            pred_list = [int(r[pred_key]) for r in results]
            gt_list = [int(r[gt_key]) for r in results]
            
            task_metrics = self._compute_binary_metrics(pred_list, gt_list, task_name)
            metrics.update(task_metrics)
        
        # ===== Summary Statistics =====
        metrics['num_samples'] = len(results)
        
        # ===== Print Summary (Only Once) =====
        print("\n" + "="*70)
        print("📊 [AI-VQA Evaluation Summary]")
        print("="*70)
        print(f"MOS Regression:")
        print(f"  MAE:  {metrics['mos_mae']:.4f}")
        print(f"  RMSE: {metrics['mos_rmse']:.4f}")
        print(f"  SRCC: {metrics['mos_srcc']:.4f} ⭐")
        print(f"  PLCC: {metrics['mos_plcc']:.4f}")
        print(f"\nQuality Classification (5-class):")
        print(f"  Accuracy: {metrics['qclass_acc']:.4f}")
        print(f"\nArtifact Detection:")
        print(f"  Hallucination → Acc: {metrics['hallucination_acc']:.4f}, F1: {metrics['hallucination_f1']:.4f}")
        print(f"  Lighting      → Acc: {metrics['lighting_acc']:.4f}, F1: {metrics['lighting_f1']:.4f}")
        print(f"  Spatial       → Acc: {metrics['spatial_acc']:.4f}, F1: {metrics['spatial_f1']:.4f}")
        print(f"  Rendering     → Acc: {metrics['rendering_acc']:.4f} (untrained)")
        print(f"\nTotal Samples: {metrics['num_samples']}")
        print("="*70 + "\n")
        
        return metrics
