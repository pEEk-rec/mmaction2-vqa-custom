# mmaction/models/heads/swin_ai_head.py
# Copyright (c) 2025
from typing import Dict, List, Optional
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
    """Stack tensor values from data_samples metainfo."""
    vals: List[torch.Tensor] = []
    for ds in data_samples:
        v = {}
        # support DataSample.metainfo being dict-like or None
        meta = getattr(ds, "metainfo", {})
        v_raw = meta.get(key, 0)
        # convert booleans -> numeric, keep torch.Tensor as-is
        if torch.is_tensor(v_raw):
            v = v_raw
        else:
            # ensure numeric scalar
            if isinstance(v_raw, bool):
                v = torch.tensor(float(v_raw) if dtype == torch.float32 else int(v_raw))
            else:
                try:
                    v = torch.tensor(v_raw)
                except Exception:
                    v = torch.tensor(0.0 if dtype == torch.float32 else 0)
        vals.append(v)
    out = torch.stack(vals, dim=0)
    if dtype is not None:
        out = out.to(dtype)
    if device is not None:
        out = out.to(device)
    return out


@MODELS.register_module()
class swin_AIHead(BaseModule):
    """Multi-task head for AI-generated VQA.
    
    Tasks:
    - MOS regression (0-100)
    - 5-class quality classification (Bad/Poor/Fair/Good/Excellent)
    - 3 binary artifact detections (hallucination, lighting, spatial)
    - 1 dummy rendering flag (for evaluation only)
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 5,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
        average_clips: Optional[str] = "prob",
        loss_cls: Dict = dict(type="CrossEntropyLoss", loss_weight=0.5),
        loss_mos: Dict = dict(type="MSELoss", loss_weight=1.5),
        # FIXED: Using CrossEntropyLoss for 2-class artifact detection
        loss_hallucination: Dict = dict(type="CrossEntropyLoss", loss_weight=0.8),
        loss_lighting: Dict = dict(type="CrossEntropyLoss", loss_weight=0.8),
        loss_spatial: Dict = dict(type="CrossEntropyLoss", loss_weight=0.8),
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.average_clips = average_clips

        if spatial_type == "avg":
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise NotImplementedError(f"spatial_type '{spatial_type}' not supported")

        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()

        # Task heads
        self.mos_head = nn.Linear(in_channels, 1)  # MOS regression
        self.cls_head = nn.Linear(in_channels, num_classes)  # 5-class quality

        # Binary artifact detection heads (2 classes each: [FALSE, TRUE])
        self.hallucination_head = nn.Linear(in_channels, 2)
        self.lighting_head = nn.Linear(in_channels, 2)
        self.spatial_head = nn.Linear(in_channels, 2)

        # Dummy rendering (kept for analysis only, not trained)
        self.register_buffer("dummy_rendering_logits", torch.zeros(1, 2))

        # Build loss functions
        self.loss_cls_fn = MODELS.build(loss_cls)
        self.loss_mos_fn = MODELS.build(loss_mos)
        self.loss_hallucination_fn = MODELS.build(loss_hallucination)
        self.loss_lighting_fn = MODELS.build(loss_lighting)
        self.loss_spatial_fn = MODELS.build(loss_spatial)

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool spatial-temporal features to [N, C]."""
        if x.ndim == 5:  # [N, C, T, H, W]
            x = self.avg_pool(x).flatten(1)
        elif x.ndim != 2:
            raise ValueError(f"Unsupported feature shape: {x.shape}")
        return self.dropout(x)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - returns all task outputs."""
        feats = self._pool_features(x)
        
        mos = self.mos_head(feats)  # [N, 1]
        cls_logits = self.cls_head(feats)  # [N, 5]
        
        hallucination_logits = self.hallucination_head(feats)  # [N, 2]
        lighting_logits = self.lighting_head(feats)  # [N, 2]
        spatial_logits = self.spatial_head(feats)  # [N, 2]
        
        # Dummy rendering logits (not trained)
        rendering_logits = self.dummy_rendering_logits.expand(feats.size(0), -1)
        
        return dict(
            mos=mos,
            cls_logits=cls_logits,
            hallucination_logits=hallucination_logits,
            lighting_logits=lighting_logits,
            spatial_logits=spatial_logits,
            rendering_logits=rendering_logits,
        )

    @staticmethod
    def _average_clips_scores(scores, num_segs, mode):
        """Average classification scores across clips."""
        if num_segs == 1 or mode is None:
            return scores
        B = scores.shape[0] // num_segs
        scores = scores.view(B, num_segs, -1)
        if mode == "prob":
            scores = F.softmax(scores, dim=-1).mean(dim=1)
        else:
            scores = scores.mean(dim=1)
        return scores

    @staticmethod
    def _average_clips_regression(values, num_segs):
        """Average regression values across clips."""
        if num_segs == 1:
            return values
        B = values.shape[0] // num_segs
        return values.view(B, num_segs, -1).mean(dim=1)

    def loss(self, feats, data_samples: SampleList, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses."""
        outs = self.forward(feats)
        total_batch = outs["cls_logits"].shape[0]
        B = len(data_samples)
        num_segs = max(1, total_batch // max(1, B))

        # Average predictions across clips
        cls_logits = self._average_clips_scores(outs["cls_logits"], num_segs, self.average_clips)
        mos_pred = self._average_clips_regression(outs["mos"], num_segs).squeeze(-1)
        device = cls_logits.device

        # Get ground truth for main tasks
        tgt_cls = _stack_from_metainfo(data_samples, "quality_class", torch.long, device)
        tgt_mos = _stack_from_metainfo(data_samples, "mos", torch.float32, device)

        # Compute main losses
        loss_cls = self.loss_cls_fn(cls_logits, tgt_cls)
        loss_mos = self.loss_mos_fn(mos_pred, tgt_mos)

        # Average artifact predictions across clips
        hallucination_logits = self._average_clips_scores(
            outs["hallucination_logits"], num_segs, self.average_clips
        )
        lighting_logits = self._average_clips_scores(
            outs["lighting_logits"], num_segs, self.average_clips
        )
        spatial_logits = self._average_clips_scores(
            outs["spatial_logits"], num_segs, self.average_clips
        )

        # FIXED: Get targets as LONG (not float32) for CrossEntropyLoss
        tgt_hallucination = _stack_from_metainfo(
            data_samples, "hallucination_flag", torch.long, device
        )
        tgt_lighting = _stack_from_metainfo(
            data_samples, "lighting_flag", torch.long, device
        )
        tgt_spatial = _stack_from_metainfo(
            data_samples, "spatial_flag", torch.long, device
        )

        # FIXED: Pass full logits [N, 2] to CrossEntropyLoss (not just [:, 1])
        loss_hallucination = self.loss_hallucination_fn(hallucination_logits, tgt_hallucination)
        loss_lighting = self.loss_lighting_fn(lighting_logits, tgt_lighting)
        loss_spatial = self.loss_spatial_fn(spatial_logits, tgt_spatial)

        # Total loss
        total_loss = loss_cls + loss_mos + loss_hallucination + loss_lighting + loss_spatial

        return dict(
            loss=total_loss,
            loss_cls=loss_cls,
            loss_mos=loss_mos,
            loss_hallucination=loss_hallucination,
            loss_lighting=loss_lighting,
            loss_spatial=loss_spatial,
        )

    def predict(self, feats: torch.Tensor, data_samples: SampleList, **kwargs) -> SampleList:
        """Predict all tasks and store in data_samples."""
        outs = self.forward(feats)
        total_batch = outs["cls_logits"].shape[0]
        B = len(data_samples)
        num_segs = max(1, total_batch // max(1, B))

        # Average predictions across clips
        cls_scores = self._average_clips_scores(outs["cls_logits"], num_segs, self.average_clips)
        mos_pred = self._average_clips_regression(outs["mos"], num_segs).squeeze(-1)
        cls_pred = cls_scores.argmax(dim=-1)

        hallucination_scores = self._average_clips_scores(
            outs["hallucination_logits"], num_segs, self.average_clips
        )
        lighting_scores = self._average_clips_scores(
            outs["lighting_logits"], num_segs, self.average_clips
        )
        spatial_scores = self._average_clips_scores(
            outs["spatial_logits"], num_segs, self.average_clips
        )
        rendering_scores = self._average_clips_scores(
            outs["rendering_logits"], num_segs, self.average_clips
        )

        # Store predictions in data_samples
        for i, ds in enumerate(data_samples):
            pred_mos_val = float(mos_pred[i].item())
            pred_qclass_val = int(cls_pred[i].item())
            pred_hallucination_val = int(hallucination_scores[i].argmax().item())
            pred_lighting_val = int(lighting_scores[i].argmax().item())
            pred_spatial_val = int(spatial_scores[i].argmax().item())
            pred_rendering_val = int(rendering_scores[i].argmax().item())
            
            # Update metainfo ONLY (don't set as top-level fields)
            meta_update = dict(
                pred_mos=pred_mos_val,
                pred_quality_class=pred_qclass_val,
                pred_qclass=pred_qclass_val,  # Alias for metric compatibility
                pred_hallucination=pred_hallucination_val,
                pred_lighting=pred_lighting_val,
                pred_spatial=pred_spatial_val,
                pred_rendering=pred_rendering_val,
            )
            
            # Store in metainfo (compatible with both old and new MMEngine)
            if hasattr(ds, "set_metainfo"):
                ds.set_metainfo(meta_update)
            elif hasattr(ds, "metainfo"):
                if ds.metainfo is None:
                    ds.metainfo = {}
                ds.metainfo.update(meta_update)
            else:
                ds.__dict__["metainfo"] = meta_update

        return data_samples