from mmengine.model import BaseModule
from mmaction.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class MSELoss(nn.MSELoss):
    """MSELoss wrapper for MMAction2 registry."""
    
    def __init__(self, loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        return super().forward(pred, target) * self.loss_weight