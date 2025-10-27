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
        
        # If 4D [C, T, H, W], add batch dimension
        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(0)  # [1, C, T, H, W]
        
        # Ensure correct shape [B, C, T, H, W]
        if inputs.ndim != 5:
            raise ValueError(f"Expected 5D input [B,C,T,H,W], got shape {inputs.shape}")

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
        """Forward and compute losses."""
        # Extract features
        feats = self.extract_feat(inputs)
        
        # Compute losses using classification head
        losses = self.cls_head.loss(feats, data_samples, **kwargs)
        
        return losses
    
    def predict(self, inputs, datasamples, **kwargs):
        """Inference - predict quality scores."""
        print("="*60)
        print("[RECOGNIZER] predict() called")
        
        # Handle list input from val_step
        
        print(f"[RECOGNIZER] Input shape: {inputs.shape}")
        print(f"[RECOGNIZER] Input dtype: {inputs.dtype}")
        print(f"[RECOGNIZER] Input range: [{inputs.min():.2f}, {inputs.max():.2f}]")
        print(f"[RECOGNIZER] Num datasamples: {len(datasamples)}")
        
        # Extract features
        feats = self.extract_feat(inputs)
        
        print(f"[RECOGNIZER] Features shape: {feats.shape}")
        
        # Get predictions from head - IMPORTANT: capture returned datasamples
        datasamples = self.cls_head.predict(feats, datasamples, **kwargs)
        
        print(f"[RECOGNIZER] Returned {len(datasamples)} predictions")
        if len(datasamples) > 0:
            sample = datasamples[0]
            if hasattr(sample, 'metainfo'):
                print(f"[RECOGNIZER] Sample 0 metainfo keys: {list(sample.metainfo.keys())}")
                print(f"[RECOGNIZER] Has pred_mos: {'pred_mos' in sample.metainfo}")
                if 'pred_mos' in sample.metainfo:
                    print(f"[RECOGNIZER] pred_mos value: {sample.metainfo['pred_mos']}")
        print("="*60)
        
        return datasamples

    
    def val_step(self, data):
        """Validation step - must return predictions with metainfo."""
        print("\n" + "="*60)
        print("[RECOGNIZER] val_step() called")
        print(f"[RECOGNIZER] data keys: {list(data.keys())}")
        
        # CRITICAL: Let data_preprocessor handle the data first!
        # This converts uint8 to float32 and normalizes
        data = self.data_preprocessor(data, training=False)
        
        inputs = data['inputs']
        datasamples = data['data_samples']
        
        print(f"[RECOGNIZER] After preprocessor - inputs type: {type(inputs)}")
        print(f"[RECOGNIZER] After preprocessor - inputs shape: {inputs.shape}")
        print(f"[RECOGNIZER] After preprocessor - inputs dtype: {inputs.dtype}")
        
        # Now call predict with preprocessed data
        outputs = self.predict(inputs, datasamples)
        
        print(f"[RECOGNIZER] val_step returning {len(outputs)} outputs")
        print("="*60 + "\n")
        return outputs