
import torch
from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class swin_AIRecognizer(BaseRecognizer):
    """
    AI-Generated Video Quality Recognizer for Stage 0.
    - Uses frozen Swin-T backbone.
    - Supports MOS regression, 5-class quality classification,
      and binary artifact predictions.
    """

    def __init__(self, backbone, cls_head, data_preprocessor=None, train_cfg=None, test_cfg=None):
        if data_preprocessor is None:
            data_preprocessor = dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            )

        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            backbone=backbone,
            cls_head=cls_head,
            data_preprocessor=data_preprocessor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

    def extract_feat(self, inputs: torch.Tensor, stage=None) -> torch.Tensor:
        """Extract frozen backbone features (robust to both tensors and video paths)."""
        import numpy as np
        import torch.nn.functional as F

        # ---------- Handle string path ----------
        if isinstance(inputs, str):
            frames = None
            try:
                # Prefer Decord (fast + reliable)
                try:
                    from decord import VideoReader, cpu
                    vr = VideoReader(inputs, ctx=cpu(0))
                    total_frames = len(vr)
                    if total_frames == 0:
                        raise ValueError("No frames decoded.")
                    # Sample 16 evenly spaced frames
                    indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
                    frames = vr.get_batch(indices).asnumpy()  # [T,H,W,C]
                except Exception:
                    # Fallback to mmcv
                    import mmcv
                    vr = mmcv.VideoReader(inputs)
                    frames = [vr[i] for i in np.linspace(0, len(vr) - 1, 16).astype(int) if len(vr) > 0]
                    if len(frames) == 0:
                        raise ValueError("mmcv failed to decode any frames.")
                    frames = np.stack(frames)

                # Convert to tensor [1, C, T, H, W]
                frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().cuda() / 255.0
                frames = F.interpolate(frames, size=(16, 224, 224), mode='trilinear', align_corners=False)
                inputs = frames

            except Exception as e:
                raise RuntimeError(f"[DecodeError] Video '{inputs}' failed: {e}")

        # ---------- Handle multi-clip tensors ----------
        if inputs.ndim == 7:
            inputs = inputs.reshape(-1, *inputs.shape[3:])
        elif inputs.ndim == 6:
            inputs = inputs.reshape(-1, *inputs.shape[2:])
        elif inputs.ndim == 4:
            inputs = inputs.unsqueeze(0)

        assert inputs.ndim == 5, f"Expected 5D input [B,C,T,H,W], got {inputs.shape}"

        # ---------- Backbone Forward ----------
        x = self.backbone(inputs)
        if isinstance(x, tuple):
            x = x[-1]

        # ---------- Global Avg Pool ----------
        if x.ndim == 5:
            x = x.mean(dim=[-3, -2, -1])
        elif x.ndim != 2:
            raise ValueError(f"Unexpected feature shape {x.shape}")

        if stage is not None:
            pass

        return x




    def predict(self, inputs, datasamples, **kwargs):
        """Inference: predict quality scores and artifact flags."""
        # Extract features
        with torch.no_grad():
            feats = self.extract_feat(inputs)
        
        # Predict via head
        with torch.no_grad():
            outputs = self.cls_head.predict(feats, datasamples, **kwargs)
        
        return outputs

    def val_step(self, data):
        """Validation step - CLEANED UP (no print statements)."""
        # Preprocess once
        data = self.data_preprocessor(data, training=False)
        inputs = data["inputs"]
        datasamples = data["data_samples"]
        
        # Predict
        with torch.no_grad():
            return self.predict(inputs, datasamples)

    def loss(self, inputs, data_samples, **kwargs):
        """Compute multi-task loss."""
        feats = self.extract_feat(inputs)
        return self.cls_head.loss(feats, data_samples, **kwargs)