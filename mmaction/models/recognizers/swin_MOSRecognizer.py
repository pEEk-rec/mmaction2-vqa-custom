"""
Custom 3D Recognizer for Video Quality Assessment
Handles MOS regression + Quality classification with frozen backbone
"""

import torch
from typing import List, Union

from mmaction.registry import MODELS
from mmaction.models import Recognizer3D
from mmengine.structures import LabelData


@MODELS.register_module()
class swin_MOSRecognizer(Recognizer3D):
    """
    3D recognizer model framework for Video Quality Assessment.
    
    Supports:
    - Frozen backbone feature extraction
    - Dual-task head (MOS regression + quality classification)
    - Compatible with VideoQualityHead
    
    Args:
        backbone (dict): Config for backbone network
        cls_head (dict): Config for VQA classification head
        data_preprocessor (dict): Config for data preprocessor
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Freeze backbone if specified
        if hasattr(self.backbone, 'frozen_stages'):
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def train(self, mode=True):
        """Override train mode to keep backbone in eval mode."""
        super().train(mode)
        if hasattr(self.backbone, 'frozen_stages') and self.backbone.frozen_stages >= 0:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        return self
    
    def loss(self, inputs: torch.Tensor, data_samples: List, **kwargs) -> dict:
        """
        Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs (torch.Tensor): Raw videos with shape [N, C, T, H, W]
            data_samples (List): List of data samples containing annotations
        
        Returns:
            dict: Dictionary of losses (loss_mos, loss_quality)
        """
        # Preprocess data samples to add gt_label structure
        data_samples = self._prepare_data_samples(data_samples)
        
        # Extract features from frozen backbone (no gradient)
        with torch.no_grad() if hasattr(self.backbone, 'frozen_stages') else torch.enable_grad():
            feats = self.extract_feat(inputs)
        
        # Calculate losses using VQA head
        losses = self.cls_head.loss(feats, data_samples)
        
        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: List, **kwargs) -> List:
        """
        Predict MOS scores and quality classes.
        
        Args:
            inputs (torch.Tensor): Raw videos with shape [N, C, T, H, W]
            data_samples (List): List of data samples
        
        Returns:
            List[dict]: Predictions with structure compatible with VQAMetric
        """
        # Extract features from backbone
        feats = self.extract_feat(inputs)
        
        # Get predictions from VQA head
        predictions = self.cls_head.predict(feats, data_samples)
        
        # Reformat for VQAMetric compatibility
        formatted_predictions = []
        for i, (pred, sample) in enumerate(zip(predictions, data_samples)):
            # Extract ground truth
            if hasattr(sample, 'gt_label'):
                mos_true = sample.gt_label.mos
                quality_true = sample.gt_label.quality_class
            else:
                mos_true = sample.get('mos', 0.0)
                quality_true = sample.get('quality_class', 0)
            
            # Format prediction for metric
            formatted_pred = dict(
                pred_scores=dict(
                    mos=pred['mos'],
                    quality_class=pred['quality_class']
                ),
                gt_label=dict(
                    mos=mos_true,
                    quality_class=quality_true
                )
            )
            formatted_predictions.append(formatted_pred)
        
        return formatted_predictions
    
    def _prepare_data_samples(self, data_samples: List) -> List:
        """
        Prepare data samples by structuring gt_label properly.
        
        Args:
            data_samples (List): Raw data samples from dataloader
        
        Returns:
            List: Processed data samples with gt_label structure
        """
        processed_samples = []
        
        for sample in data_samples:
            # Create LabelData structure if needed
            if not hasattr(sample, 'gt_label'):
                gt_label = LabelData()
                gt_label.mos = torch.tensor(sample.get('mos', 0.0), dtype=torch.float32)
                gt_label.quality_class = torch.tensor(sample.get('quality_class', 0), dtype=torch.long)
                sample.gt_label = gt_label
            else:
                # Ensure gt_label has required fields
                if not hasattr(sample.gt_label, 'mos'):
                    sample.gt_label.mos = torch.tensor(sample.get('mos', 0.0), dtype=torch.float32)
                if not hasattr(sample.gt_label, 'quality_class'):
                    sample.gt_label.quality_class = torch.tensor(sample.get('quality_class', 0), dtype=torch.long)
            
            processed_samples.append(sample)
        
        return processed_samples
    
    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone.
        
        Args:
            inputs (torch.Tensor): Input videos [N, C, T, H, W]
        
        Returns:
            torch.Tensor: Extracted features
        """
        # Forward through backbone
        feats = self.backbone(inputs)
        
        # Handle tuple outputs (e.g., SlowFast)
        if isinstance(feats, tuple):
            feats = torch.cat(feats, dim=1)
        
        return feats
