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
    vals: List[torch.Tensor] = []
    for ds in data_samples:
        meta = getattr(ds, "metainfo", {}) or {}
        v_raw = meta.get(key, 0)

        if torch.is_tensor(v_raw):
            v = v_raw
        else:
            if isinstance(v_raw, bool):
                if dtype == torch.float32:
                    v = torch.tensor(float(v_raw), dtype=torch.float32)
                elif dtype == torch.long:
                    v = torch.tensor(int(v_raw), dtype=torch.long)
                else:
                    v = torch.tensor(float(v_raw))
            else:
                try:
                    v = torch.tensor(v_raw)
                except Exception:
                    v = torch.tensor(0.0 if dtype == torch.float32 else 0)

        vals.append(v)

    out = torch.stack(vals, dim=0)
    if dtype is not None: out = out.to(dtype)
    if device is not None: out = out.to(device)
    return out


@MODELS.register_module()
class swin_AIHead(BaseModule):

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 5,
        dropout_ratio: float = 0.25,
        average_clips: Optional[str] = "prob",
        use_uncertainty_weighting: bool = True,
        loss_cls: Dict = dict(type="CrossEntropyLoss"),
        loss_mos: Dict = dict(type="MSELoss"),
        loss_hallucination: Dict = dict(type="BCEWithLogitsLoss", pos_weight=2.07),
        loss_lighting: Dict = dict(type="BCEWithLogitsLoss", pos_weight=3.26),
        loss_spatial: Dict = dict(type="BCEWithLogitsLoss", pos_weight=9.36),
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.average_clips = average_clips
        self.use_uncertainty_weighting = use_uncertainty_weighting

        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
        )

        self.mos_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self.hallucination_head = nn.Linear(384, 1)
        self.lighting_head = nn.Linear(384, 1)
        self.spatial_head = nn.Linear(384, 1)

        self.register_buffer("dummy_rendering_logits", torch.zeros(1, 2))

        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(5))

            self.loss_cls_fn = nn.CrossEntropyLoss()
            self.loss_mos_fn = nn.MSELoss()

            self.register_buffer("pos_weight_hallucination", torch.tensor(2.07))
            self.register_buffer("pos_weight_lighting", torch.tensor(3.26))
            self.register_buffer("pos_weight_spatial", torch.tensor(9.36))

        else:
            self.log_vars = None
            self.loss_cls_fn = nn.CrossEntropyLoss()
            self.loss_mos_fn = nn.MSELoss()

            self.register_buffer("pos_weight_hallucination", torch.tensor(2.07))
            self.register_buffer("pos_weight_lighting", torch.tensor(3.26))
            self.register_buffer("pos_weight_spatial", torch.tensor(9.36))

    def _task_specific_pooling(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_global = x.mean(dim=[2, 3, 4])
        x_temporal = x.mean(dim=[3, 4]).mean(dim=-1)
        x_spatial = x.mean(dim=2).mean(dim=[2, 3])
        return dict(global_feats=x_global, temporal_feats=x_temporal, spatial_feats=x_spatial)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pooled = self._task_specific_pooling(x)

        global_feats = self.shared_fc(pooled["global_feats"])
        mos = self.mos_head(global_feats)
        cls_logits = self.cls_head(global_feats)

        temporal_feats = self.shared_fc(pooled["temporal_feats"])
        hallucination_logit = self.hallucination_head(temporal_feats)

        spatial_feats = self.shared_fc(pooled["spatial_feats"])
        lighting_logit = self.lighting_head(spatial_feats)
        spatial_logit = self.spatial_head(spatial_feats)

        rendering_logits = self.dummy_rendering_logits.expand(mos.size(0), -1)

        hallucination_logits = torch.cat([-hallucination_logit, hallucination_logit], dim=1)
        lighting_logits = torch.cat([-lighting_logit, lighting_logit], dim=1)
        spatial_logits = torch.cat([-spatial_logit, spatial_logit], dim=1)

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
        if num_segs == 1 or mode is None: return scores
        B = scores.shape[0] // num_segs
        scores = scores.view(B, num_segs, -1)
        if mode == "prob":
            return F.softmax(scores, dim=-1).mean(dim=1)
        return scores.mean(dim=1)

    @staticmethod
    def _average_clips_regression(values, num_segs):
        if num_segs == 1: return values
        B = values.shape[0] // num_segs
        return values.view(B, num_segs, -1).mean(dim=1)

    def loss(self, feats, data_samples: SampleList, **kwargs) -> Dict[str, torch.Tensor]:
        outs = self.forward(feats)
        total_batch = outs["cls_logits"].shape[0]
        B = len(data_samples)
        num_segs = max(1, total_batch // max(1, B))

        cls_logits = self._average_clips_scores(outs["cls_logits"], num_segs, self.average_clips)
        mos_pred = self._average_clips_regression(outs["mos"], num_segs).squeeze(-1)
        device = cls_logits.device

        tgt_cls = _stack_from_metainfo(data_samples, "quality_class", torch.long, device)
        tgt_mos = _stack_from_metainfo(data_samples, "mos", torch.float32, device)

        hallucination_logits = (
            self._average_clips_scores(outs["hallucination_logits"], num_segs, self.average_clips)[:, 1:2]
        )
        lighting_logits = (
            self._average_clips_scores(outs["lighting_logits"], num_segs, self.average_clips)[:, 1:2]
        )
        spatial_logits = (
            self._average_clips_scores(outs["spatial_logits"], num_segs, self.average_clips)[:, 1:2]
        )

        tgt_hallucination = (
            _stack_from_metainfo(data_samples, "hallucination_flag", torch.float32, device).unsqueeze(1)
        )
        tgt_lighting = (
            _stack_from_metainfo(data_samples, "lighting_flag", torch.float32, device).unsqueeze(1)
        )
        tgt_spatial = (
            _stack_from_metainfo(data_samples, "spatial_flag", torch.float32, device).unsqueeze(1)
        )

        loss_cls = self.loss_cls_fn(cls_logits, tgt_cls)
        loss_mos = self.loss_mos_fn(mos_pred, tgt_mos)

        bce_hallu = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_hallucination.to(device))
        bce_light = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_lighting.to(device))
        bce_spatial = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_spatial.to(device))

        loss_hallucination = bce_hallu(hallucination_logits, tgt_hallucination)
        loss_lighting = bce_light(lighting_logits, tgt_lighting)
        loss_spatial = bce_spatial(spatial_logits, tgt_spatial)

        if self.use_uncertainty_weighting:
            with torch.no_grad():
                self.log_vars.data.clamp_(-8.0, 8.0)

            losses = torch.stack([loss_mos, loss_cls, loss_hallucination, loss_lighting, loss_spatial])
            weighted_losses = 0.5 * torch.exp(-self.log_vars) * losses + 0.5 * self.log_vars
            total_loss = weighted_losses.sum()

            loss_mos = weighted_losses[0]
            loss_cls = weighted_losses[1]
            loss_hallucination = weighted_losses[2]
            loss_lighting = weighted_losses[3]
            loss_spatial = weighted_losses[4]

        else:
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
        outs = self.forward(feats)
        total_batch = outs["cls_logits"].shape[0]
        B = len(data_samples)
        num_segs = max(1, total_batch // max(1, B))

        cls_scores = self._average_clips_scores(outs["cls_logits"], num_segs, self.average_clips)
        mos_pred = self._average_clips_regression(outs["mos"], num_segs).squeeze(-1)

        mos_pred = mos_pred.clamp(12.0, 90.0)
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

        for i, ds in enumerate(data_samples):
            meta_update = dict(
                pred_mos=float(mos_pred[i].item()),
                pred_quality_class=int(cls_pred[i].item()),
                pred_qclass=int(cls_pred[i].item()),
                pred_hallucination=int(hallucination_scores[i].argmax().item()),
                pred_lighting=int(lighting_scores[i].argmax().item()),
                pred_spatial=int(spatial_scores[i].argmax().item()),
                pred_rendering=int(rendering_scores[i].argmax().item()),
            )
            if hasattr(ds, "set_metainfo"):
                ds.set_metainfo(meta_update)
            elif hasattr(ds, "metainfo"):
                if ds.metainfo is None: ds.metainfo = {}
                ds.metainfo.update(meta_update)
            else:
                ds.__dict__["metainfo"] = meta_update

        return data_samples
