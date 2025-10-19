import os.path as osp
from pathlib import Path
import torch
import numpy as np
import mmengine
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner import load_checkpoint

# Allow numpy types
torch.serialization.add_safe_globals([np.ndarray])
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

_original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = patched_load

from mmaction.registry import MODELS

def mos_to_quality_label(mos):
    if mos < 2.0:
        return 0  # Bad
    elif mos < 3.0:
        return 1  # Poor
    elif mos < 4.0:
        return 2  # Fair
    elif mos < 4.5:
        return 3  # Good
    else:
        return 4  # Excellent

def init_recognizer(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, (str, Path)):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('Config must be a filename or Config object')
    init_default_scope(config.get('default_scope', 'mmaction'))
    if hasattr(config.model, 'backbone') and config.model.backbone.get('pretrained', None):
        config.model.backbone.pretrained = None
    model = MODELS.build(config.model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def inference_recognizer(model, video, test_pipeline=None):
    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)
    if isinstance(video, dict):
        data = video
    elif isinstance(video, str) and osp.exists(video):
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    else:
        raise RuntimeError(f'Unsupported video input type: {type(video)}')
    data = test_pipeline(data)
    data = pseudo_collate([data])
    with torch.no_grad():
        result = model.test_step(data)[0]
    return result

if __name__ == "__main__":
    config = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\configs\recognition\swin\swin_tiny_finevdMOS.py"
    checkpoint = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\swin_tiny_finevd_mos\best_mos_SRCC_epoch_40.pth"
    video_path = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid15k-B\k150kb\orig_6607203845_540_5s.mp4"
    model = init_recognizer(config, checkpoint)
    result = inference_recognizer(model, video_path)
    mos_score_tensor = result.pred_scores.get('mos', None)
    if mos_score_tensor is not None:
        predicted_mos = mos_score_tensor.detach().cpu().item()
        predicted_class = mos_to_quality_label(predicted_mos)
        print("Predicted MOS:", predicted_mos)
        print("Predicted Quality Class:", predicted_class)
    else:
        print("MOS score not found in model output.")
