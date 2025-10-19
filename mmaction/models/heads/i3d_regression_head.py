# mmaction/models/heads/i3d_regression_head.py
import torch
import torch.nn as nn
from torch import nn, Tensor
from typing import Dict, List, Union
from mmengine.model.weight_init import normal_init
from mmaction.registry import MODELS
from mmaction.models.heads.base import BaseHead
from mmaction.utils import SampleList

@MODELS.register_module()
class I3DRegressionHead(BaseHead):
    """Regression head that outputs a single MOS value for video quality assessment.

    Supports both 2D ([N, C, H, W]) and 3D ([N, C, D, H, W]) backbone features.
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 dropout_ratio: float = 0.5,
                 spatial_type: str = 'avg',
                 init_std: float = 0.01,
                 loss_cls: Dict = dict(type='MSELoss'),
                 **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        
        # Build loss function from config
        if isinstance(loss_cls, dict):
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_cls

        # Dropout
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = nn.Identity()

        # Pooling layers
        if self.spatial_type == 'avg':
            self.avg_pool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise NotImplementedError(f"spatial_type '{self.spatial_type}' not supported")

        # Regression layer - properly initialized upfront
        self.fc_reg = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initialize weights of the regression head."""
        normal_init(self.fc_reg, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass of the regression head.
        
        Args:
            x (Tensor): Input features from backbone
            
        Returns:
            Tensor: Regression outputs [N, 1]
        """
        # Apply appropriate pooling based on input dimensions
        if x.ndim == 5:  # 3D features: [N, C, T, H, W]
            x = self.avg_pool_3d(x)  # [N, C, 1, 1, 1]
        elif x.ndim == 4:  # 2D features: [N, C, H, W]
            x = self.avg_pool_2d(x)  # [N, C, 1, 1]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")

        # Flatten to [N, C]
        x = x.flatten(1)
        
        # Apply dropout
        x = self.dropout(x)

        # Regression output
        x = self.fc_reg(x)  # [N, num_classes]
        return x

    def loss(self, feats: Tensor, data_samples: SampleList, **kwargs) -> Dict[str, Tensor]:
        """Compute regression loss.
        
        Args:
            feats (Tensor): Features from backbone
            data_samples (SampleList): List of data samples with ground truth labels
            
        Returns:
            Dict[str, Tensor]: Loss dictionary
        """
        # Forward pass to get predictions
        preds = self(feats)  # [N, 1]
        
        # Extract ground truth labels
        targets = self._extract_targets(data_samples, preds.device, preds.dtype)
        
        # Compute loss
        loss_value = self.loss_fn(preds, targets)
        
        return {'loss_cls': loss_value}  # Use 'loss_cls' to match MMAction2 conventions

    def predict(self, feats: Tensor, data_samples: SampleList, **kwargs) -> SampleList:
        """Predict regression values for given features.
        
        Args:
            feats (Tensor): Features from backbone
            data_samples (SampleList): List of data samples
            
        Returns:
            SampleList: Updated data samples with predictions
        """
        # Forward pass
        preds = self(feats)  # [N, 1]
        
        # Update data samples with predictions
        for i, data_sample in enumerate(data_samples):
            pred_score = preds[i].cpu().numpy()
            data_sample.pred_score = pred_score
            
        return data_samples

    def _extract_targets(self, data_samples: SampleList, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Extract ground truth targets from data samples.
        
        Args:
            data_samples (SampleList): List of data samples
            device (torch.device): Target device
            dtype (torch.dtype): Target dtype
            
        Returns:
            Tensor: Ground truth targets [N, 1]
        """
        targets = []
        for data_sample in data_samples:
            # Handle different possible label formats
            if hasattr(data_sample, 'gt_label'):
                if isinstance(data_sample.gt_label, torch.Tensor):
                    target = data_sample.gt_label.item()
                else:
                    target = float(data_sample.gt_label)
            elif hasattr(data_sample, 'gt_labels'):
                # For compatibility with different MMAction2 versions
                if isinstance(data_sample.gt_labels, torch.Tensor):
                    target = data_sample.gt_labels.item()
                else:
                    target = float(data_sample.gt_labels[0] if isinstance(data_sample.gt_labels, (list, tuple)) else data_sample.gt_labels)
            else:
                raise ValueError(f"No ground truth label found in data_sample: {data_sample}")
            
            targets.append(target)
        
        # Convert to tensor
        targets = torch.tensor(targets, device=device, dtype=dtype).view(-1, 1)
        return targets