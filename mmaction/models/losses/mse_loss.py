import torch
from mmengine.model import BaseModule, MODELS

@MODELS.register_module()
class MyMSELoss(BaseModule):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * self.loss_fn(pred, target)
