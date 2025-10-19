# tools/check_multitask_head.py
import os
import sys
import torch
from types import SimpleNamespace

def main():
    # Make sure mmaction is importable when running from repo root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(repo_root)  # go up from tools/
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Adjust import path to your actual file name/module
    try:
        from mmaction.models.heads.VQA_multihead import MultiTaskHead
    except Exception:
        from mmaction.models.heads.VQA_multihead import MultiTaskHead  # fallback if you named it differently

    # Minimal ActionDataSample-like stub
    class DS(SimpleNamespace):
        def __init__(self, **kw):
            super().__init__(**kw)
            if not hasattr(self, 'metainfo'):
                self.metainfo = {}
        def set_pred_score(self, x): self.metainfo['pred_score'] = x
        def set_pred_label(self, x): self.metainfo['pred_label'] = x

    B, num_segs = 2, 4
    N = B * num_segs

    # 5D input (the head will pool)
    x = torch.randn(N, 768, 1, 1, 1)

    samples = []
    for i in range(B):
        samples.append(DS(metainfo=dict(
            mos=float(0.5 * i),
            quality_class=int(i % 2),
            distortion_type=int(i % 7)
        )))

    head = MultiTaskHead(
        in_channels=768,
        num_quality_classes=2,
        num_distortion_types=7,
        average_clips='score'
    )

    # Forward
    outs = head(x)
    assert outs['mos'].shape == (N, 1), f"Unexpected mos shape {outs['mos'].shape}"
    assert outs['qclass_logits'].shape[-1] == 2, "Bad quality_class dim"
    assert outs['dtype_logits'].shape[-1] == 7, "Bad distortion_type dim"

    # Loss
    losses = head.loss(x, samples)
    for k in ['loss', 'loss_mos', 'loss_qclass', 'loss_dtype']:
        assert k in losses, f"Missing loss key: {k}"
        assert torch.is_tensor(losses[k]), f"{k} not a tensor"

    # Predict
    pred = head.predict(x, samples)
    assert len(pred) == B, "Predict did not reduce clips to batch size"
    keys = pred[0].metainfo.keys()
    required = {'pred_score', 'pred_label', 'pred_dtype_score', 'pred_dtype_label', 'pred_mos'}
    missing = required - set(keys)
    assert not missing, f"Missing prediction keys: {missing}"

    print("MultiTaskHead check passed: forward/loss/predict OK.")

if __name__ == "__main__":
    main()
