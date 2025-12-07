# mmaction/models/recognizers/swin_AIRecognizer.py
import numpy as np
import torch
import torch.nn.functional as F
from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class swin_AIRecognizer(BaseRecognizer):
    """
    AI-Generated Video Quality Recognizer for Stage 0.

    Notes:
    - By default this extract_feat preserves the backbone's spatial+temporal
      dimensions. If the backbone returns a 5D tensor [N,C,T,H,W], we return it
      as-is (no pooling). This makes the recognizer compatible with heads that
      expect pre-pool NCTHW features (task-specific pooling in head).
    - If the backbone returns 2D features [N, C], these are returned unchanged,
      so heads expecting pooled features will also work.
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

    def _device(self):
        """Return device where backbone parameters live (or CPU if none)."""
        try:
            return next(self.backbone.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _safe_to_tensor(self, frames_np: np.ndarray, device: torch.device):
        """
        Convert numpy frames [T,H,W,C] to tensor [1,C,T,H,W] on `device`.
        Ensures float32 and scaling.
        """
        frames_t = torch.from_numpy(frames_np)  # [T,H,W,C]
        # If frames are uint8, convert properly
        if frames_t.dtype == torch.uint8:
            frames_t = frames_t.float().div(255.0)
        else:
            frames_t = frames_t.float()
        # Permute to [C,T,H,W] then add batch dim
        frames_t = frames_t.permute(3, 0, 1, 2).unsqueeze(0).to(device)
        return frames_t

    def extract_feat(self, inputs: torch.Tensor | str, stage=None) -> torch.Tensor:
        """
        Extract features from the (frozen) backbone.

        Inputs may be:
        - a string path to a video file (we decode T frames and return [1,C,T,H,W]),
        - or a Tensor already shaped as [B,C,T,H,W] (or with additional clip dims).
        """
        device = self._device()

        # ---------- Handle string path ----------
        if isinstance(inputs, str):
            frames = None
            try:
                # Prefer Decord
                try:
                    from decord import VideoReader, cpu  # may raise
                    vr = VideoReader(inputs, ctx=cpu(0))
                    total_frames = len(vr)
                    if total_frames <= 0:
                        raise ValueError("no frames decoded by decord")
                    # sample 16 frames evenly (if less than 16, decord indexing will clamp)
                    indices = np.linspace(0, max(total_frames - 1, 0), num=16, dtype=int)
                    frames = vr.get_batch(indices).asnumpy()  # [T,H,W,C]
                except Exception:
                    # fallback to mmcv
                    import mmcv
                    vr = mmcv.VideoReader(inputs)
                    if len(vr) == 0:
                        raise ValueError("mmcv failed to decode frames")
                    # collect evenly spaced frames
                    indices = np.linspace(0, len(vr) - 1, num=16).astype(int)
                    frames_list = [vr[i] for i in indices]
                    frames = np.stack(frames_list)  # [T,H,W,C]

                # convert to tensor on backbone device
                frames_t = self._safe_to_tensor(frames, device)  # [1,C,T,H,W]
                # spatial/temporal resize to (T=16,H=224,W=224) using trilinear
                # note: trilinear expects (N, C, D, H, W)
                frames_t = F.interpolate(frames_t, size=(16, 224, 224), mode="trilinear", align_corners=False)
                inputs = frames_t

            except Exception as e:
                raise RuntimeError(f"[DecodeError] Video '{inputs}' failed: {e}")

        # ---------- Handle tensor inputs ----------
        # Accept input dims: 4D (C,H,W) or 5D (B,C,T,H,W) or 6/7D for clip stacking
        if isinstance(inputs, torch.Tensor):
            # ensure on correct device and dtype
            if inputs.device != device:
                inputs = inputs.to(device)
            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            # If tensor is 4D (C,H,W) assume single clip single sample -> add batch dim & temporal
            if inputs.ndim == 3:
                # [C,H,W] -> [1,C,1,H,W] (awkward; prefer user passes correct shape)
                inputs = inputs.unsqueeze(0).unsqueeze(2)
            elif inputs.ndim == 4:
                # [B,C,H,W] -> [B,C,1,H,W]
                inputs = inputs.unsqueeze(2)
            elif inputs.ndim == 6:
                # possible shape [B, num_clips, C, T, H, W] -> collapse clips batchwise
                B, num_clips, C, T, H, W = inputs.shape
                inputs = inputs.view(B * num_clips, C, T, H, W)
            elif inputs.ndim == 7:
                # [B, num_clips, num_sub, C, T, H, W] -> collapse leading clip dims
                # general handling: flatten first dims to batch
                new_shape = (int(np.prod(inputs.shape[:-4])),) + inputs.shape[-4:]
                inputs = inputs.reshape(new_shape)
            # else if inputs.ndim ==5, expected [B,C,T,H,W] -> ok
        else:
            raise TypeError("inputs must be torch.Tensor or path string")

        assert inputs.ndim == 5, f"Expected 5D input [B,C,T,H,W], got {inputs.shape}"

        # ---------- Backbone forward ----------
        # It's the caller's responsibility to put backbone in eval() / freeze grads if desired.
        x = self.backbone(inputs)
        # Many backbones return tuple(stages...); pick last if tuple
        if isinstance(x, (tuple, list)):
            x = x[-1]

        # ---------- Preserve shape ----------
        # If backbone returns 5D features [B,C,T,H,W], return as-is (head will do pooling).
        # If it returns 2D features [B,C], return those too (head expects pooled features).
        if x.ndim not in (2, 5):
            # Defensive check: some backbones return [B,C,1,1,1] -> squeeze minor dims
            if x.ndim == 4:
                # uncommon; try to keep it as-is
                return x
            # if unexpected, raise explicit error
            raise ValueError(f"Unexpected backbone output shape {x.shape}; expected 2D [B,C] or 5D [B,C,T,H,W].")

        return x

    def predict(self, inputs, datasamples, **kwargs):
        """Inference: predict quality scores and artifact flags."""
        # Extract features
        with torch.no_grad():
            feats = self.extract_feat(inputs)
        # Predict via head (head should handle pooling / averaging across clips)
        with torch.no_grad():
            outputs = self.cls_head.predict(feats, datasamples, **kwargs)
        return outputs

    def val_step(self, data):
        """Validation step (no prints)."""
        data = self.data_preprocessor(data, training=False)
        inputs = data["inputs"]
        datasamples = data["data_samples"]
        with torch.no_grad():
            return self.predict(inputs, datasamples)

    def loss(self, inputs, data_samples, **kwargs):
        """Compute multi-task loss using head.loss()."""
        feats = self.extract_feat(inputs)
        return self.cls_head.loss(feats, data_samples, **kwargs)
