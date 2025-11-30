# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from mmaction.registry import MODELS
from .recognizer3d import Recognizer3D


@MODELS.register_module()
class swin_MOSRecognizer(Recognizer3D):
    """Video Quality Assessment Recognizer for MOS prediction.
    
    This recognizer extends Recognizer3D to handle dual-task learning:
    - Quality classification (4 classes)
    - MOS regression (continuous score)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extract_feat(self, inputs: Tensor) -> Tensor:
        """Extract features from input tensor.
        
        Args:
            inputs (Tensor): Input tensor
        
        Returns:
            Tensor: Extracted features
        """
        # Debug: Print input shape
        # print(f"DEBUG: Input shape: {inputs.shape}, ndim: {inputs.ndim}")
        
        # Handle different input dimensions
        if inputs.ndim == 6:
            # Shape: (N, num_clips, C, T, H, W)
            # This happens when num_clips > 1 or during multi-clip testing
            n, num_clips, c, t, h, w = inputs.shape
            inputs = inputs.view(n * num_clips, c, t, h, w)
            feats = self.backbone(inputs)
            
            # Reshape and average features
            if feats.ndim == 5:
                c_out, t_out, h_out, w_out = feats.shape[1:]
                feats = feats.view(n, num_clips, c_out, t_out, h_out, w_out)
                feats = feats.mean(dim=1)
            
        elif inputs.ndim == 5:
            # Shape: (N, C, T, H, W) - Expected format
            feats = self.backbone(inputs)
            
        else:
            raise ValueError(
                f"Expected 5D or 6D input tensor, got {inputs.ndim}D with shape {inputs.shape}"
            )
        
        return feats
    
    def loss(self, inputs: Tensor, data_samples: list, **kwargs) -> Dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs (Tensor): Raw Inputs of the recognizer.
            data_samples (list): The batch data samples.
            
        Returns:
            dict: A dictionary of loss components.
        """
        feats = self.extract_feat(inputs)
        loss_dict = self.cls_head.loss(feats, data_samples, **kwargs)
        return loss_dict
    
    def predict(self, inputs: Tensor, data_samples: list, **kwargs) -> list:
        """Predict results from a batch of inputs and data samples.
        
        Args:
            inputs (Tensor): Raw Inputs of the recognizer.
            data_samples (list): The batch data samples.
            
        Returns:
            list: Prediction results (MOS scores and quality classes).
        """
        feats = self.extract_feat(inputs)
        predictions = self.cls_head.predict(feats, data_samples, **kwargs)
        return predictions