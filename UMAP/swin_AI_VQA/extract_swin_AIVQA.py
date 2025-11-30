#!/usr/bin/env python3
"""
Extract pooled per-video features from Video Swin-T stages (mmaction2) - AI-VQA adapted
Saves per-video .npy files and global stage matrices.

Usage example (PowerShell):
python .\extract_features_aivqa.py --checkpoint "path\epoch_150.pth" --config "path\config.py" \
  --video_root "path\videos" --csv "path\train.csv" --out_dir "path\features_out" --stages 3 4 --device cuda

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
    print(f"[INFO] Loading CSV: {csv_path}")
    meta = pd.read_csv(csv_path)
    
    print(f"[INFO] CSV columns: {meta.columns.tolist()}")
    print(f"[INFO] CSV shape: {meta.shape}")
    print(f"[INFO] First few rows:")
    print(meta.head())
    
    # Flexible column detection for AI-VQA
    filename_col = None
    for col in ['filename', 'video_name', 'name', 'video_filename', 'file']:
        if col in meta.columns:
            filename_col = col
            break
    
    if filename_col is None:
        # If no standard column found, assume first column is filename
        print(f"[WARN] No standard filename column found. Using first column: '{meta.columns[0]}'")
        filename_col = meta.columns[0]
    
    print(f"[INFO] Using column '{filename_col}' as filename column")
    
    # Rename to 'filename' for consistency
    if filename_col != 'filename':
        meta = meta.rename(columns={filename_col: 'filename'})
    
    meta['stem'] = meta['filename'].apply(lambda x: Path(x).stem)
    
    print(f"[INFO] Sample stems:")
    print(meta['stem'].head(10).tolist())

    # parse config
    print(f"[INFO] Loading config: {cfg_path}")
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

    print(f"[INFO] Building pipeline with {len(pipeline_cfg)} transforms")
    # Build mmaction2-style pipeline (Compose)
    test_pipeline = Compose(pipeline_cfg)

    # init recognizer (load model + weights)
    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    print(f"[INFO] Loading model on device: {device}")
    model = init_recognizer(cfg, str(ckpt), device=device)
    model.eval()
    
    # If wrapped in DataParallel
    if hasattr(model, 'module'):
        backbone = model.module.backbone
    else:
        backbone = model.backbone

    print(f"[INFO] Backbone type: {type(backbone).__name__}")
    print(f"[INFO] Backbone has {len(backbone.layers)} layers")

    # Hook capturing
    features_buffer = {}
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

    # Determine layer modules to hook
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
    
    failed_videos = []

    print(f"\n[INFO] Starting feature extraction for {len(meta)} videos...")
    for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting features"):
        fname = row['filename']
        stem = Path(fname).stem
        
        # Build full video path
        video_path = video_root / fname
        
        # Check if file exists
        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            failed_videos.append(stem)
            continue
        
        # Build data dict used by pipeline
        data = dict(
            filename=str(video_path),
            modality='RGB',
            start_index=0
        )
        
        # Apply pipeline to get processed input
        try:
            processed = test_pipeline(data)
        except Exception as e:
            print(f"[WARN] Pipeline failed for {fname}: {e}")
            failed_videos.append(stem)
            continue

        # Find the tensor
        if 'inputs' in processed:
            inp = processed['inputs']
        elif 'imgs' in processed:
            inp = processed['imgs']
        else:
            # try to find first tensor-like key
            candidate = None
            for k in processed.keys():
                v = processed[k]
                if isinstance(v, np.ndarray):
                    candidate = v
                    break
            if candidate is None:
                print(f"[WARN] Could not locate input frames for {fname}, processed keys: {list(processed.keys())}")
                failed_videos.append(stem)
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
                failed_videos.append(stem)
                continue

        # Ensure shape -> (B, C, D, H, W)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 5 and tensor.shape[-1] not in (1,3) and tensor.shape[1] not in (1,3):
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

        # Forward (with no grads) - run backbone directly to trigger hooks
        with torch.no_grad():
            try:
                _ = backbone(tensor)
            except Exception as e:
                print(f"[WARN] Backbone forward failed for {fname}: {e}")
                failed_videos.append(stem)
                continue

        # For each requested stage, get buffer entry
        per_video_dict = {}
        for idx in requested_stage_idxs:
            key = f"_last_hook_output_stage_{idx+1}"
            if key not in features_buffer or features_buffer[key] is None:
                print(f"[WARN] Hook did not capture stage {idx+1} for {stem}. Trying alternative extraction.")
                try:
                    backbone_module = backbone
                    backbone_in = tensor
                    outs = backbone_module(backbone_in)
                    if isinstance(outs, (list, tuple)):
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
                    vec = pool_feature(arr)
                    per_video_dict[f"stage{idx+1}"] = vec

        # Save per-video stage vectors
        any_saved = False
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
            any_saved = True
        
        if any_saved:
            saved_per_video.append(stem)
        else:
            failed_videos.append(stem)

    # remove hooks
    for h in hooks:
        h.remove()

    # Save global stage matrices
    print(f"\n[INFO] Saving global stage matrices...")
    for idx in requested_stage_idxs:
        arrs = global_stage_feats.get(idx, [])
        if len(arrs) == 0:
            print(f"[WARN] No features collected for stage{idx+1}")
            continue
        M = np.vstack(arrs)
        fname = out_dir / f"features_stage{idx+1}.npy"
        np.save(fname, M)
        print(f"[INFO] Saved global features for stage{idx+1} shape={M.shape} at {fname}")

    # Save index list
    pd.DataFrame({'stem': saved_per_video}).to_csv(out_dir / "extracted_video_list.csv", index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("[DONE] Extraction finished!")
    print(f"{'='*80}")
    print(f"Total videos in CSV: {len(meta)}")
    print(f"Successfully extracted: {len(saved_per_video)}")
    print(f"Failed: {len(failed_videos)}")
    if failed_videos:
        print(f"\nFailed videos (first 10): {failed_videos[:10]}")
    print(f"\nOutput directory: {out_dir}")
    print(f"Files saved:")
    print(f"  - extracted_video_list.csv")
    for idx in requested_stage_idxs:
        print(f"  - features_stage{idx+1}.npy")
    print(f"  - {len(saved_per_video)} individual .npy files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Swin-T stage features using mmaction2 config & checkpoint (AI-VQA adapted)")
    parser.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    parser.add_argument("--config", required=True, help="path to mmaction2 config.py")
    parser.add_argument("--video_root", required=True, help="root folder with video files")
    parser.add_argument("--csv", required=True, help="CSV with video filenames and annotations")
    parser.add_argument("--out_dir", required=True, help="where to save features")
    parser.add_argument("--stages", nargs="+", default=[3,4], help="stages to extract (1..4). Example: --stages 3 4")
    parser.add_argument("--device", default="cuda", help="device to run on, e.g., 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)