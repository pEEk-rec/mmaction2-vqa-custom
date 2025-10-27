# mmaction/models/heads/multitask_head.py
# Copyright (c) 2025
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmaction.registry import MODELS
from mmaction.utils import SampleList


def _stack_from_metainfo(
    data_samples: SampleList,
    key: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    vals: List[torch.Tensor] = []
    for ds in data_samples:
        v = ds.metainfo[key]
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        vals.append(v)
    out = torch.stack(vals, dim=0)
    if dtype is not None:
        out = out.to(dtype)
    if device is not None:
        out = out.to(device)
    return out


@MODELS.register_module()
class MultiTaskHead(BaseModule):
    """Multi-task head for MOS regression + quality/distortion classification.

    Inputs:
        - feats: Tensor of shape [N, C, T, H, W] or [N, C]
    Outputs (forward):
        - dict with:
            'mos': Tensor [N, 1]
            'qclass_logits': Tensor [N, num_quality_classes]
            'dtype_logits': Tensor [N, num_distortion_types]

    Aggregation:
        - During loss/predict, if multiple clips per sample were used
          (N = B * num_segs), predictions are aggregated to B by:
            * classification: 'score' or 'prob'
            * mos: mean over clips
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_quality_classes: int = 2,
        num_distortion_types: int = 6,
        dropout_ratio: float = 0.5,
        average_clips: Optional[str] = "score",  # 'score' | 'prob' | None
        loss_mos: Dict = dict(type="SmoothL1Loss", loss_weight=1.0),
        loss_qclass: Dict = dict(type="CrossEntropyLoss", loss_weight=1.0),
        loss_dtype: Dict = dict(type="CrossEntropyLoss", loss_weight=1.0),
        init_cfg: Optional[Dict] = None,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_quality_classes = num_quality_classes
        self.num_distortion_types = num_distortion_types
        self.average_clips = average_clips
        self.use_layernorm = use_layernorm

        # Pooling to [N, C]
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Shared projection (optional)
        self.norm = nn.LayerNorm(in_channels) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

        # Branches
        self.mos_head = nn.Linear(in_channels, 1)
        self.qclass_head = nn.Linear(in_channels, num_quality_classes)
        self.dtype_head = nn.Linear(in_channels, num_distortion_types)

        # Losses
        self.loss_mos = nn.SmoothL1Loss() # or nn.MSELoss()
        self.loss_qclass = MODELS.build(loss_qclass)
        self.loss_dtype = MODELS.build(loss_dtype)

    def _pool_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [N, C, T, H, W] or [N, C]
        if x.ndim == 5:
            x = self.avg_pool3d(x)  # [N, C, 1, 1, 1]
            x = x.flatten(1)        # [N, C]
        elif x.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported feature shape {x.shape}")
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        feats = self._pool_if_needed(x)
        mos = self.mos_head(feats)  # [N, 1]
        qclass_logits = self.qclass_head(feats)  # [N, Nq]
        dtype_logits = self.dtype_head(feats)    # [N, Nd]
        return dict(mos=mos, qclass_logits=qclass_logits, dtype_logits=dtype_logits)

    @staticmethod
    def _average_clips_scores(
        scores: torch.Tensor, num_segs: int, mode: Optional[str]
    ) -> torch.Tensor:
        """Aggregate classification scores across clips."""
        if mode not in ["score", "prob", None]:
            raise ValueError(f"average_clips {mode} not in ['score','prob',None]")
        if num_segs == 1 or mode is None:
            return scores
        B = scores.shape[0] // num_segs
        scores = scores.view(B, num_segs, -1)
        if mode == "prob":
            scores = F.softmax(scores, dim=-1).mean(dim=1)
        else:  # 'score'
            scores = scores.mean(dim=1)
        return scores

    @staticmethod
    def _average_clips_regression(values: torch.Tensor, num_segs: int) -> torch.Tensor:
        """Average regression outputs across clips."""
        if num_segs == 1:
            return values
        B = values.shape[0] // num_segs
        values = values.view(B, num_segs, -1).mean(dim=1)
        return values

    def loss(
        self, feats: Union[torch.Tensor, Tuple[torch.Tensor]], data_samples: SampleList, **kwargs
    ) -> Dict[str, torch.Tensor]:
        outs = self.forward(feats, **kwargs)

        # Determine num_segs: assumes len(data_samples) = B, feats batch = B*num_segs
        total_batch = outs["qclass_logits"].shape[0]
        B = len(data_samples)
        assert total_batch % B == 0, "Batch size must be divisible by num samples"
        num_segs = total_batch // B

        # Aggregate across clips
        qclass_logits = self._average_clips_scores(
            outs["qclass_logits"], num_segs, self.average_clips
        )
        dtype_logits = self._average_clips_scores(
            outs["dtype_logits"], num_segs, self.average_clips
        )
        mos_pred = self._average_clips_regression(outs["mos"], num_segs).squeeze(-1)

        # Stack targets from metainfo
        device = qclass_logits.device
        tgt_qclass = _stack_from_metainfo(data_samples, "quality_class", torch.long, device)
        tgt_dtype = _stack_from_metainfo(data_samples, "distortion_type", torch.long, device)
        tgt_mos = _stack_from_metainfo(data_samples, "mos", torch.float32, device)

        # Compute losses
        loss_qclass = self.loss_qclass(qclass_logits, tgt_qclass)
        loss_dtype = self.loss_dtype(dtype_logits, tgt_dtype)
        loss_mos = self.loss_mos(mos_pred, tgt_mos)

        # Weighted sum handled inside each loss via loss_weight; sum here
        total = loss_qclass + loss_dtype + loss_mos
        return dict(loss=total, loss_qclass=loss_qclass, loss_dtype=loss_dtype, loss_mos=loss_mos)

    def predict(self, feats, data_samples, **kwargs):
        print("="*60)
        print("[HEAD] predict() called")
        print(f"[HEAD] Feats shape: {feats.shape}")
        print(f"[HEAD] Num samples: {len(data_samples)}")
        
        # Your existing prediction code
        outs = self.forward(feats, **kwargs)
        
        total_batch = outs['qclass_logits'].shape[0]
        B = len(data_samples)
        num_segs = total_batch // B
        
        # Average clips
        qclass_scores = self._average_clips_scores(outs['qclass_logits'], num_segs, self.average_clips)
        dtype_scores = self._average_clips_scores(outs['dtype_logits'], num_segs, self.average_clips)
        mos_pred = self._average_clips_regression(outs['mos'], num_segs).squeeze(-1)
        
        qclass_pred = qclass_scores.argmax(dim=-1)
        dtype_pred = dtype_scores.argmax(dim=-1)
        
        print(f"[HEAD] MOS predictions (first 3): {mos_pred[:3].tolist()}")
        print(f"[HEAD] QClass predictions (first 3): {qclass_pred[:3].tolist()}")
        
        # Store predictions in metainfo
        for i, ds in enumerate(data_samples):
            pred_mos_val = float(mos_pred[i].item())
            pred_qclass_val = int(qclass_pred[i].item())
            pred_dtype_val = int(dtype_pred[i].item())
            
            # Store in metainfo
            if not hasattr(ds, 'metainfo'):
                ds.metainfo = {}
            ds.metainfo['pred_mos'] = pred_mos_val
            ds.metainfo['pred_qclass'] = pred_qclass_val
            ds.metainfo['pred_dtype'] = pred_dtype_val
            
            # CRITICAL: Also store at top level (for dict conversion)
            ds.pred_mos = pred_mos_val
            ds.pred_qclass = pred_qclass_val
            ds.pred_dtype = pred_dtype_val
        
        return data_samples
