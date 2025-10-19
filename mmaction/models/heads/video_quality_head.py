# Copyright (c) OpenMMLab. All rights reserved.
"""Video Quality Prediction Head for MOS regression."""

import torch
import torch.nn as nn
from mmaction.registry import MODELS


@MODELS.register_module()
class VideoQualityHead(nn.Module):
    """Quality prediction head for MOS regression.
    
    Architecture: Features → FC1 → ReLU → Dropout → FC2 → MOS
    
    Args:
        in_channels (int): Input feature dimension (768 for Video Swin-T)
        hidden_dim (int): Hidden layer dimension
        dropout_ratio (float): Dropout probability
        loss_weight (float): Loss weight
    """
    
    def __init__(self,
                 in_channels=768,
                 hidden_dim=256,
                 dropout_ratio=0.5,
                 loss_weight=1.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.loss_weight = loss_weight
        
        # MOS regression head
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # MSE loss for regression
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Features [N, in_channels]
        
        Returns:
            torch.Tensor: MOS predictions [N, 1]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def loss(self, feats, data_samples, **kwargs):
        """Calculate MSE loss.
        
        Args:
            feats (torch.Tensor): Extracted features [N, in_channels]
            data_samples (list): List with ground truth MOS
        
        Returns:
            dict: Loss dictionary
        """
        preds = self.forward(feats).squeeze(-1)  # [N]
        
        gt_mos = torch.tensor([ds.mos for ds in data_samples],
                              dtype=torch.float32,
                              device=preds.device)
        
        loss = self.loss_fn(preds, gt_mos)
        
        return {'loss': loss * self.loss_weight}
    
    def predict(self, feats, data_samples, **kwargs):
        """Predict MOS scores.
        
        Args:
            feats (torch.Tensor): Features
            data_samples (list): Data samples
        
        Returns:
            list: Predictions with pred_mos attribute
        """
        preds = self.forward(feats).squeeze(-1)
        
        for i, ds in enumerate(data_samples):
            ds.pred_mos = preds[i].item()
        
        return data_samples
