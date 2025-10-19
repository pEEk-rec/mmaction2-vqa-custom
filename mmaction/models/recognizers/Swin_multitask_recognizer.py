import torch
from mmaction.registry import MODELS
from .base import BaseRecognizer

@MODELS.register_module()
class MultitaskRecognizer(BaseRecognizer):
    def __init__(self, backbone, cls_head, **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, **kwargs)
        
    def extract_feat(self, inputs):
        """Extract features using backbone"""
        return self.backbone(inputs)
    
    def loss(self, inputs, data_samples, **kwargs):
        """Forward and compute losses"""
        feats = self.extract_feat(inputs)
        losses = self.cls_head.loss(feats, data_samples, **kwargs)
        return losses
    
    def predict(self, inputs, data_samples, **kwargs):
        """Inference"""
        feats = self.extract_feat(inputs)
        predictions = self.cls_head(feats)
        return predictions
