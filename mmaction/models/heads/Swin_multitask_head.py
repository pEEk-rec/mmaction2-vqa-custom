import torch
import torch.nn as nn
from mmaction.registry import MODELS
from .base import BaseHead

@MODELS.register_module()
class MultitaskHead(BaseHead):
    def __init__(self,
                 num_classes_action,
                 num_classes_quality, 
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes_action = num_classes_action
        self.num_classes_quality = num_classes_quality
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        
        # Shared feature processing
        self.dropout = nn.Dropout(p=dropout_ratio)
        
        # Task-specific heads (MLPs as suggested)
        # Action recognition head
        self.action_head = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(1024, num_classes_action)
        )
        
        # MOS regression head  
        self.mos_head = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.GELU(), 
            nn.Dropout(dropout_ratio),
            nn.Linear(1024, 1)  # Single scalar output
        )
        
        # Quality classification head
        self.quality_head = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.GELU(),
            nn.Dropout(dropout_ratio), 
            nn.Linear(1024, num_classes_quality)
        )
        
        # Uncertainty weighting parameters (trainable)
        self.log_var_action = nn.Parameter(torch.zeros(()))
        self.log_var_mos = nn.Parameter(torch.zeros(()))
        self.log_var_quality = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        # x: [batch_size, in_channels]
        x = self.dropout(x)
        
        # Get predictions from each head
        action_logits = self.action_head(x)
        mos_pred = self.mos_head(x)
        quality_logits = self.quality_head(x)
        
        return {
            'action_logits': action_logits,
            'mos_pred': mos_pred, 
            'quality_logits': quality_logits
        }
    
    def loss(self, feats, data_samples, **kwargs):
        """Compute multitask loss with uncertainty weighting"""
        predictions = self.forward(feats)
        
        # Extract labels from data_samples
        action_labels = torch.stack([sample.gt_label for sample in data_samples])
        mos_labels = torch.stack([sample.get('mos_label', torch.tensor(0.0)) for sample in data_samples])
        quality_labels = torch.stack([sample.get('quality_label', torch.tensor([0])) for sample in data_samples])
        
        # Compute individual losses
        loss_action = nn.CrossEntropyLoss()(predictions['action_logits'], action_labels)
        loss_mos = nn.L1Loss()(predictions['mos_pred'].squeeze(), mos_labels.float())  # MAE
        loss_quality = nn.CrossEntropyLoss()(predictions['quality_logits'], quality_labels)
        
        # Uncertainty-weighted total loss
        total_loss = (loss_action / (2 * torch.exp(self.log_var_action)) + self.log_var_action / 2 +
                     loss_mos / (2 * torch.exp(self.log_var_mos)) + self.log_var_mos / 2 +
                     loss_quality / (2 * torch.exp(self.log_var_quality)) + self.log_var_quality / 2)
        
        losses = {
            'loss_cls': total_loss,
            'loss_action': loss_action,
            'loss_mos': loss_mos, 
            'loss_quality': loss_quality
        }
        
        return losses
