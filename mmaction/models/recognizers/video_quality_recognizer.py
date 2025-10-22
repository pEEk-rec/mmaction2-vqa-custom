# Copyright (c) OpenMMLab. All rights reserved.
"""Video Quality Assessment Recognizer using Video Swin Transformer."""

import torch
from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class VideoQualityRecognizer(BaseRecognizer):
    """Video Quality Assessment Recognizer.
    
    Predicts Mean Opinion Score (MOS) and quality attributes for videos.
    Uses Video Swin Transformer as backbone.
    
    Args:
        backbone (dict): Backbone config (Video Swin Transformer)
        cls_head (dict): Classification head config (VideoQualityHead)
        data_preprocessor (dict): Data preprocessing config
        train_cfg (dict): Training config
        test_cfg (dict): Testing config
    """
    
    def __init__(self, 
                 backbone,
                 cls_head,
                 data_preprocessor=None,
                 train_cfg=None,
                 test_cfg=None):
        
        # Set default data preprocessor for Video Swin
        if data_preprocessor is None:
            data_preprocessor = dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape='NCTHW'
            )
        
        # CRITICAL FIX: Build data_preprocessor using mmaction's MODELS registry
        # before passing to parent, so parent receives the built object not a dict
        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        
        super().__init__(
            backbone=backbone,
            cls_head=cls_head,
            data_preprocessor=data_preprocessor,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
    
    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from input video tensor.
        
        Args:
            inputs (torch.Tensor): Video tensor [N, C, T, H, W] or [N, M, C, T, H, W]
        
        Returns:
            torch.Tensor: Extracted features [N*M, C]
        """
        # Flatten multi-clip: [N, M, C, T, H, W] -> [N*M, C, T, H, W]
        if inputs.ndim == 6:
            inputs = inputs.reshape(-1, *inputs.shape[2:])

        x = self.backbone(inputs)
        if isinstance(x, tuple):
            x = x[-1]

        # Global average pool over T/H/W -> [N*M, C]
        if x.ndim == 5:
            x = x.mean(dim=[-3, -2, -1])
        elif x.ndim != 2:
            raise ValueError(f"Unsupported feature shape {x.shape}")

        return x

    
    def loss(self, inputs, data_samples, **kwargs):
        """Forward and compute losses.
        
        Args:
            inputs (torch.Tensor): Video tensor [N, C, T, H, W]
            data_samples (list): List of QualityDataSample with ground truth
            **kwargs: Additional arguments
        
        Returns:
            dict: Dictionary of losses
        """
        # Extract features
        feats = self.extract_feat(inputs)
        
        # Compute losses using classification head
        losses = self.cls_head.loss(feats, data_samples, **kwargs)
        
        return losses
    
    def predict(self, inputs, data_samples, **kwargs):
        """Inference - predict quality scores.
        
        Args:
            inputs (torch.Tensor): Video tensor
            data_samples (list): List of QualityDataSample
            **kwargs: Additional arguments
        
        Returns:
            list: Predictions with predicted MOS and attributes
        """
        # Extract features
        feats = self.extract_feat(inputs)
        
        # Get predictions from head
        predictions = self.cls_head.predict(feats, data_samples, **kwargs)
        
        return predictions