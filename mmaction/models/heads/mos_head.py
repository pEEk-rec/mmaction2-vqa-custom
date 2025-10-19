# mmaction2/mmaction/models/heads/mos_head.py

import torch
import torch.nn as nn
from mmaction.registry import MODELS
from .base import BaseHead

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

@MODELS.register_module()
class MOSRegressionHead(BaseHead):
    """Head for MOS regression.
    
    Quality classification is derived from predicted MOS scores.
    
    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of classes (kept for compatibility)
        dropout_ratio (float): Dropout ratio
        init_std (float): Std for weight initialization
        mos_range (tuple): Range of MOS scores (min, max)
        spatial_type (str): Pooling type (ignored, kept for compatibility)
        loss_cls (dict): Loss config (ignored, kept for compatibility)
    """
    
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 mos_range: tuple = (1.0, 5.0),
                 spatial_type: str = 'avg',  # Accept but ignore
                 loss_cls: dict = None,      # Accept but ignore
                 **kwargs):
        
        # Remove unsupported kwargs before passing to parent
        supported_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['spatial_type', 'loss_cls']:
                supported_kwargs[key] = value
        
        super().__init__(num_classes, in_channels, **supported_kwargs)
        
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.mos_min, self.mos_max = mos_range
        self.mos_scale = self.mos_max - self.mos_min
        
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        # Single regression head for MOS
        self.fc_mos = nn.Linear(self.in_channels, 1)
        
        # Sigmoid to constrain output to [0, 1], then scale to MOS range
        self.sigmoid = nn.Sigmoid()
    
    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.fc_mos.weight, 0, self.init_std)
        nn.init.constant_(self.fc_mos.bias, 0)
    
    def forward(self, x, **kwargs):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Features from backbone [N, C, T, H, W]
            
        Returns:
            torch.Tensor: MOS predictions [N]
        """
        # Global average pooling over spatial and temporal dimensions
        if len(x.shape) == 5:  # [N, C, T, H, W]
            x = x.mean([2, 3, 4])  # [N, C]
        elif len(x.shape) == 4:  # [N, C, H, W]
            x = x.mean([2, 3])  # [N, C]
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        # MOS prediction
        mos_pred = self.fc_mos(x)  # [N, 1]
        mos_pred = self.sigmoid(mos_pred) * self.mos_scale + self.mos_min  # Scale to [mos_min, mos_max]
        mos_pred = mos_pred.squeeze(-1)  # [N]
        
        return mos_pred
    
    def loss(self, feats, data_samples, **kwargs):
        """Calculate loss."""
        mos_pred = self(feats, **kwargs)
        
        # Extract MOS - try different possible locations
        mos_gt_list = []
        for sample in data_samples:
            # Check where MOS is stored
            if hasattr(sample, 'mos'):
                mos_gt_list.append(sample.mos)
            elif hasattr(sample.gt_label, 'mos'):
                mos_gt_list.append(sample.gt_label.mos)
            else:
                # gt_label itself is the MOS value
                mos_gt_list.append(sample.gt_label)
        
        # Convert to tensor
        mos_gt = torch.stack([
            torch.as_tensor(m, dtype=torch.float32, device=mos_pred.device) 
            for m in mos_gt_list
        ]).squeeze()
        
        losses = dict()
        loss_mos = nn.functional.smooth_l1_loss(mos_pred, mos_gt)
        losses['loss_cls'] = loss_mos
        
        with torch.no_grad():
            class_pred = torch.tensor([mos_to_quality_label(m.item()) for m in mos_pred], device=mos_pred.device)
            class_gt = torch.tensor([mos_to_quality_label(m.item()) for m in mos_gt], device=mos_gt.device)
            losses['acc'] = (class_pred == class_gt).float().mean()
            losses['mae'] = torch.abs(mos_pred - mos_gt).mean()
        
        return losses
    
    def predict(self, feats, data_samples, **kwargs):
        """Predict MOS scores.
        
        Args:
            feats (torch.Tensor): Features from backbone
            data_samples (list): List of data samples
            
        Returns:
            list: List of data samples with predictions
        """
        mos_pred = self(feats, **kwargs)
        
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_scores = dict(mos=mos_pred[i])
        
        return data_samples