# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model.weight_init import normal_init

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class swin_MOSHead(BaseHead):
    """Multi-task head for Video Quality Assessment.
    
    Predicts both MOS (regression) and quality class (classification).
    
    Args:
        num_classes (int): Number of quality classes (e.g., 5 for poor to excellent).
        in_channels (int): Number of channels in input feature.
        loss_mos (dict or ConfigDict): Config for MOS regression loss.
            Default: dict(type='MSELoss', loss_weight=1.5)
        loss_cls (dict or ConfigDict): Config for quality classification loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=0.5)
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for initialization. Default: 0.01.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_mos: ConfigType = dict(type='MSELoss', loss_weight=1.5),
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss', loss_weight=0.5),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        # Dropout layer
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # Spatial pooling
        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        # MOS regression head (predicts single continuous value)
        self.fc_mos = nn.Linear(self.in_channels, 1)

        # Quality classification head (predicts discrete quality class)
        self.fc_quality = nn.Linear(self.in_channels, self.num_classes)

        # Build loss functions
        self.loss_mos_fn = MODELS.build(loss_mos)
        self.loss_cls_fn = MODELS.build(loss_cls)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_mos, std=self.init_std)
        normal_init(self.fc_quality, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward pass to predict MOS and quality class.

        Args:
            x (Tensor): Input features with shape [N, in_channels, T, H, W].

        Returns:
            Tuple[Tensor, Tensor]: 
                - mos_pred: MOS predictions with shape [N, 1]
                - quality_pred: Quality class scores with shape [N, num_classes]
        """
        # Spatial pooling: [N, in_channels, T, H, W] -> [N, in_channels, 1, 1, 1]
        if self.avg_pool is not None:
            x = self.avg_pool(x)

        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # Flatten: [N, in_channels, 1, 1, 1] -> [N, in_channels]
        x = x.view(x.shape[0], -1)

        # MOS regression head
        mos_pred = self.fc_mos(x)  # [N, 1]

        # Quality classification head
        quality_pred = self.fc_quality(x)  # [N, num_classes]

        return mos_pred, quality_pred

    def loss(self, feats: Tensor, data_samples: list, **kwargs) -> dict:
        """Calculate losses from the extracted features.

        Args:
            feats (Tensor): Features from backbone with shape [N, C, T, H, W].
            data_samples (list): List of data samples containing ground truth.

        Returns:
            dict: A dictionary of loss components.
        """
        # Forward pass
        mos_pred, quality_pred = self.forward(feats)

        # Extract ground truth from data_samples
        mos_gt = torch.stack([sample.gt_label.mos for sample in data_samples]).to(feats.device)
        quality_gt = torch.stack([sample.gt_label.quality_class for sample in data_samples]).to(feats.device)

        # Reshape MOS predictions and targets
        mos_pred = mos_pred.squeeze(-1)  # [N, 1] -> [N]
        mos_gt = mos_gt.float()  # Ensure float type

        # Calculate losses
        loss_mos = self.loss_mos_fn(mos_pred, mos_gt)
        loss_quality = self.loss_cls_fn(quality_pred, quality_gt)

        # Return loss dict
        losses = dict()
        losses['loss_mos'] = loss_mos
        losses['loss_quality'] = loss_quality

        return losses

    def predict(self, feats: Tensor, data_samples: list, **kwargs) -> list:
        """Predict MOS and quality class for inference.

        Args:
            feats (Tensor): Features from backbone.
            data_samples (list): List of data samples.

        Returns:
            list: List of prediction results.
        """
        # Forward pass
        mos_pred, quality_pred = self.forward(feats)

        # Get quality class predictions
        quality_pred_labels = quality_pred.argmax(dim=1)

        # Clamp MOS predictions to valid range [1.0, 5.0]
        mos_pred = torch.clamp(mos_pred.squeeze(-1), min=1.0, max=5.0)

        # Prepare predictions
        predictions = []
        for i in range(len(data_samples)):
            pred_dict = dict(
                mos=mos_pred[i].item(),
                quality_class=quality_pred_labels[i].item(),
                quality_scores=quality_pred[i].cpu().numpy()
            )
            predictions.append(pred_dict)

        return predictions
