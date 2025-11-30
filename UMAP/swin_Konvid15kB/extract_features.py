#!/usr/bin/env python3
"""
Extract pooled per-video features from Video Swin-T stages (mmaction2)
Saves per-video .npy files and global stage matrices.

Usage example (PowerShell):
python .\extract_swin_stages.py --checkpoint "path\epoch_48.pth" --config "path\config.py" \
  --video_root "path\videos" --csv "path\konvid_test.csv" --out_dir "path\features_out" --stages 3 4 --device cuda

Requirements:
- mmaction2 installed & in PYTHONPATH (or installed via pip)
- mmcv / mmengine compatible with your mmaction2 & CUDA
- decord, tqdm, numpy, pandas
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# mmaction2 / mmcv imports
from mmengine import Config

# --- CRITICAL: Import mmaction2 to trigger registry of all transforms ---
# This must happen BEFORE building the pipeline
import mmaction
import mmaction.datasets
import mmaction.datasets.transforms

# Import the registry - this is where transforms are registered
from mmaction.registry import TRANSFORMS

# Use mmengine's Compose but with mmaction2's registry already loaded
# The key is that mmaction2 transforms are now registered in the shared registry
from mmengine.dataset import Compose as EngineCompose

# Create a Compose that uses mmaction2's TRANSFORMS registry
class Compose(EngineCompose):
    def __init__(self, transforms):
        # Override to use mmaction2's TRANSFORMS registry
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
            self.transforms.append(transform)

# Also explicitly import the init_recognizer after mmaction is loaded
from mmaction.apis import init_recognizer
# ---------------------------------------------------------------

# ---------- Helpers ----------
def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def pool_feature(arr: np.ndarray):
    """Pool a feature array into 1D vector.
    Handles shapes: (C,), (C,D,H,W), (B,C,D,H,W), (D,H,W,C), (b,d,h,w,c) etc.
    Returns 1D numpy array.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.copy()
    # If shape begins with batch dim 1, remove
    if a.ndim >= 2 and a.shape[0] == 1 and a.ndim > 1:
        a = a[0]
    # Common mm outputs may be (C, D, H, W) or (D,H,W,C) or (B,C,D,H,W)
    # Try to bring channel (feature) axis to first position if possible
    # Heuristics: channel sizes likely 96,192,384,768
    possible_channels = (96, 128, 192, 384, 512, 768, 1024)
    # If last dim looks like channel
    if a.ndim >= 3 and a.shape[-1] in possible_channels:
        # average over all dims except last
        vec = a.mean(axis=tuple(range(a.ndim-1)))
        return vec.reshape(-1)
    # If first dim looks like channel
    if a.shape[0] in possible_channels:
        vec = a.mean(axis=tuple(range(1, a.ndim)))
        return vec.reshape(-1)
    # If shape is (b, c, d, h, w)
    if a.ndim == 5:
        # assume (B,C,D,H,W)
        a2 = a.mean(axis=(0,2,3,4))  # (C,)
        return a2.reshape(-1)
    # Fallback: global mean scalar
    return np.array([a.mean()])

# ---------- Main ----------
def main(args):
    ckpt = Path(args.checkpoint)
    cfg_path = Path(args.config)
    video_root = Path(args.video_root)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    # Load metadata CSV
    meta = pd.read_csv(csv_path)
    if 'filename' not in meta.columns:
        # try common names
        if 'video_name' in meta.columns:
            meta = meta.rename(columns={'video_name': 'filename'})
        else:
            raise ValueError("CSV must contain 'filename' column (video file names).")
    meta['stem'] = meta['filename'].apply(lambda x: Path(x).stem)

    # parse config
    cfg = Config.fromfile(str(cfg_path))
    # build test pipeline from config's test_dataloader.dataset.pipeline (or test_pipeline)
    pipeline_cfg = None
    if 'test_pipeline' in cfg:
        pipeline_cfg = cfg.test_pipeline
    else:
        # try reading from dataloader
        try:
            pipeline_cfg = cfg.test_dataloader['dataset']['pipeline']
        except Exception:
            pipeline_cfg = None

    if pipeline_cfg is None:
        raise ValueError("Could not find test pipeline in config. Provide a config with test_pipeline or test_dataloader.dataset.pipeline.")

    # Build mmaction2-style pipeline (Compose)
    test_pipeline = Compose(pipeline_cfg)

    # init recognizer (load model + weights)
    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    print(f"[INFO] Loading model on device: {device}")
    model = init_recognizer(cfg, str(ckpt), device=device)
    model.eval()
    # If wrapped in DataParallel
    wrapped = False
    if hasattr(model, 'module'):
        backbone = model.module.backbone
        wrapped = True
    else:
        backbone = model.backbone

    # Hook capturing
    features_buffer = {}  # video_stem -> {stage_idx: tensor}
    def make_hook(stage_idx):
        def hook(module, inp, out):
            # convert PyTorch tensor to numpy (cpu)
            try:
                t = out.detach().cpu().numpy()
            except Exception:
                try:
                    t = out[0].detach().cpu().numpy()
                except Exception:
                    t = None
            features_buffer['_last_hook_output_stage_{}'.format(stage_idx)] = t
        return hook

    # Determine layer modules to hook: layers list in backbone
    # layer indices are 0-based; user requested stages e.g., 3 and 4 -> translate to indices 2 and 3
    requested_stage_idxs = []
    for s in args.stages:
        try:
            i = int(s)
            # user stage numbering 1..4 -> index is i-1
            idx = i - 1
            requested_stage_idxs.append(idx)
        except:
            raise ValueError("Stages must be given as integers 1..4")

    # register hooks on backbone.layers[idx]
    hooks = []
    for idx in requested_stage_idxs:
        if idx < 0 or idx >= len(backbone.layers):
            raise ValueError(f"Requested stage index {idx} out of range for backbone with {len(backbone.layers)} layers.")
        module = backbone.layers[idx]
        h = module.register_forward_hook(make_hook(idx+1))
        hooks.append(h)
        print(f"[INFO] Registered hook on backbone.layers[{idx}] (stage {idx+1})")

    # Build iteration: for each video, apply pipeline to get tensors in NCTHW
    saved_per_video = []
    global_stage_feats = {idx: [] for idx in requested_stage_idxs}

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        fname = row['filename']
        stem = Path(fname).stem
        # Build data dict used by pipeline (mimic test pipeline PackActionInputs meta_keys)
        data = dict(
            filename=str(video_root / fname),
            modality='RGB',
            start_index=0
        )
        # Apply pipeline to get processed input (a dict with 'imgs' or 'inputs')
        try:
            processed = test_pipeline(data)
        except Exception as e:
            print(f"[WARN] Pipeline failed for {fname}: {e}")
            continue

        # mmaction2 pipeline returns dict with 'inputs' or 'imgs' depending on pipeline.
        # Find the tensor and convert to torch tensor with shape (B,C,D,H,W)
        # Common key is 'inputs' or 'imgs'
        if 'inputs' in processed:
            inp = processed['inputs']  # maybe numpy array
        elif 'imgs' in processed:
            inp = processed['imgs']
        elif 'imgs_whatever' in processed:
            inp = processed['imgs_whatever']
        else:
            # try meta 'imgs' or pack
            keys = list(processed.keys())
            # pick first tensor-like key
            candidate = None
            for k in keys:
                v = processed[k]
                if isinstance(v, np.ndarray):
                    candidate = v
                    break
            if candidate is None:
                print(f"[WARN] Could not locate input frames for {fname}, processed keys: {keys}")
                continue
            inp = candidate

        # Ensure numpy -> torch and correct dtype
        if isinstance(inp, np.ndarray):
            tensor = torch.from_numpy(inp).unsqueeze(0) if inp.ndim==4 else torch.from_numpy(inp)
        elif isinstance(inp, torch.Tensor):
            tensor = inp
        else:
            try:
                tensor = torch.tensor(inp)
            except Exception as e:
                print(f"[WARN] Unexpected processed input type for {fname}: {type(inp)}")
                continue

        # Ensure shape -> (B, C, D, H, W). Many pipelines produce (C, D, H, W)
        if tensor.ndim == 4:
            # (C, D, H, W) -> add batch
            tensor = tensor.unsqueeze(0)
        # If shape is (B, D, H, W, C) convert to (B, C, D, H, W)
        if tensor.ndim == 5 and tensor.shape[-1] not in (1,3) and tensor.shape[1] not in (1,3):
            # try to detect channel axis
            if tensor.shape[-1] in (1,3):
                tensor = tensor.permute(0,4,1,2,3).contiguous()
        
        # Convert to float32 if needed (uint8 -> float32)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            # Normalize to [0, 1] if values are in [0, 255] range
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        
        tensor = tensor.to(device)

        # Zero buffer
        features_buffer.clear()

        # Forward (with no grads)
        # We only need to run the backbone to trigger hooks, not the full model
        with torch.no_grad():
            try:
                # Run backbone directly to trigger hooks
                # The hooks will capture intermediate layer outputs
                _ = backbone(tensor)
            except Exception as e:
                print(f"[WARN] Backbone forward failed for {fname}: {e}")
                continue

        # For each requested stage, get buffer entry
        per_video_dict = {}
        for idx in requested_stage_idxs:
            key = f"_last_hook_output_stage_{idx+1}"
            if key not in features_buffer or features_buffer[key] is None:
                # If hook didn't capture output (maybe model returned outs), try extracting from model output
                # If model returns tuple or tensor corresponding to out_indices, try to derive
                # Attempt to extract from model.backbone forward output if available
                print(f"[WARN] Hook did not capture stage {idx+1} for {stem}. Trying alternative extraction.")
                # Try to call backbone directly to get outs with desired out_indices
                try:
                    # Try to call backbone.forward and capture returned outs (some backbones return outs according to out_indices)
                    backbone_module = backbone
                    # ensure input shape expected by backbone: N,C,D,H,W?
                    backbone_in = tensor
                    outs = backbone_module(backbone_in)
                    # outs may be a tuple; pick appropriate index
                    if isinstance(outs, (list, tuple)):
                        # mapping: out_indices in config are 0-based indices of layers
                        # find which returned entry corresponds to layer idx
                        # fallback: choose last two
                        if len(outs) >= (idx+1):
                            candidate = outs[idx]
                        else:
                            candidate = outs[-1]
                    else:
                        candidate = outs
                    arr = candidate.detach().cpu().numpy()
                    vec = pool_feature(arr)
                    per_video_dict[f"stage{idx+1}"] = vec
                except Exception as e:
                    print(f"[ERR] Alternative extraction failed for {stem} stage{idx+1}: {e}")
                    per_video_dict[f"stage{idx+1}"] = None
            else:
                arr = features_buffer[key]
                if arr is None:
                    per_video_dict[f"stage{idx+1}"] = None
                else:
                    # arr might be shape (B, D, H, W, C) or similar; pool
                    vec = pool_feature(arr)
                    per_video_dict[f"stage{idx+1}"] = vec

        # Save per-video stage vectors
        for idx in requested_stage_idxs:
            stg = f"stage{idx+1}"
            vec = per_video_dict.get(stg, None)
            if vec is None:
                continue
            # Save per-video file
            out_file = out_dir / f"{stem}__swin__{stg}.npy"
            np.save(out_file, vec)
            # accumulate global
            global_stage_feats[idx].append(vec)

        saved_per_video.append(stem)

    # remove hooks
    for h in hooks:
        h.remove()

    # Save global stage matrices
    for idx in requested_stage_idxs:
        arrs = global_stage_feats.get(idx, [])
        if len(arrs) == 0:
            continue
        M = np.vstack(arrs)
        fname = out_dir / f"features_stage{idx+1}.npy"
        np.save(fname, M)
        print(f"[INFO] Saved global features for stage{idx+1} shape={M.shape} at {fname}")

    # Save index list
    pd.DataFrame({'stem': saved_per_video}).to_csv(out_dir / "extracted_video_list.csv", index=False)
    print("[DONE] Extraction finished. Per-video .npy files and stage matrices saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Swin-T stage features using mmaction2 config & checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    parser.add_argument("--config", required=True, help="path to mmaction2 config.py")
    parser.add_argument("--video_root", required=True, help="root folder with video files")
    parser.add_argument("--csv", required=True, help="CSV with filenames (relative to video_root) and optional annotations (must contain 'filename')")
    parser.add_argument("--out_dir", required=True, help="where to save features")
    parser.add_argument("--stages", nargs="+", default=[3,4], help="stages to extract (1..4). Example: --stages 3 4")
    parser.add_argument("--device", default="cuda", help="device to run on, e.g., 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)