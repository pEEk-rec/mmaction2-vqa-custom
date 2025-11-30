# ============================================================
# Swin-T Feature Extractor for AI-VQA (Fixed Version)
# ============================================================
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS
from sklearn.preprocessing import StandardScaler
import joblib

# -------- User Paths --------
config_file = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\configs\recognition\swin\swin_AI-VQA_Config.py'
checkpoint = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\swin_Ai-VQA\epoch_40.pth'
device = 'cuda:0'

video_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\standardized_videos'
splits = {
    'train': r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\train.csv',
    'val': r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\val.csv',
    'test': r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\test.csv'
}

out_dir = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\swin_AI-VQA'
os.makedirs(out_dir, exist_ok=True)

NORMALIZE_FEATURES = True

# -------- Load model --------
print("🔧 Loading model...")
cfg = Config.fromfile(config_file)
try:
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location=device)
except Exception as e:
    print(f"⚠️ Standard load failed, trying fallback init_recognizer: {e}")
    from mmaction.apis import init_recognizer
    model = init_recognizer(config_file, checkpoint, device=device)

model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# -------- Hook Setup --------
penultimate_buf = {}

def hook_penultimate(module, inp, out):
    feat = out[0] if isinstance(out, (tuple, list)) else out
    penultimate_buf['feat'] = feat.detach()

handle = None
hook_location = "unknown"

if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm'):
    handle = model.backbone.norm.register_forward_hook(hook_penultimate)
    hook_location = "backbone.norm (output)"
    print(f"✓ Hook attached to: {hook_location}")
elif hasattr(model, 'cls_head'):
    handle = model.cls_head.register_forward_hook(hook_penultimate)
    hook_location = "cls_head (input)"
    print(f"✓ Hook attached to: {hook_location}")
else:
    print("⚠ No hook attached — relying on model output directly.")

# -------- Feature Extraction Helper --------
def extract_feature_vector(feat: torch.Tensor):
    """Convert captured tensor into 1D numpy vector."""
    feat = feat.float().cpu()
    if feat.ndim == 1:
        return feat.numpy()
    elif feat.ndim == 2:
        return feat.mean(dim=0).numpy()
    elif feat.ndim == 3:
        return feat.mean(dim=(0, 1)).numpy()
    elif feat.ndim == 4:
        return feat.mean(dim=(0, 2, 3)).numpy()
    else:
        return feat.flatten().numpy()

# -------- Extraction Loop --------
for split_name, csv_file in splits.items():
    print(f"\n{'='*60}")
    print(f"Extracting {split_name.upper()} features...")
    print(f"{'='*60}")

    df = pd.read_csv(csv_file)
    features_list, mos_list, names = [], [], []
    failed_videos = []

    video_col = "filename"
    mos_col = "mos"

    for idx, row in tqdm(enumerate(df.itertuples()), total=len(df), desc=f"Processing {split_name}"):
        video_path = os.path.join(video_root, getattr(row, video_col))

        if not os.path.exists(video_path):
            print(f"⚠ Video not found: {video_path}")
            failed_videos.append(getattr(row, video_col))
            continue

        try:
            penultimate_buf.clear()

            with torch.no_grad():
                # FIX: Call model's extract_feat directly instead of __call__
                # This avoids unpacking issues from model forward
                feat = model.extract_feat(video_path)
                
                # Debug print for first video
                if idx == 0:
                    print(f"\n✓ Direct extract_feat output:")
                    print(f"   Type: {type(feat)}")
                    print(f"   Shape: {feat.shape if torch.is_tensor(feat) else 'N/A'}")
                
                vec = None
                extraction_method = "unknown"

                # Priority 1: Use extracted features directly
                if torch.is_tensor(feat):
                    vec = extract_feature_vector(feat)
                    extraction_method = "extract_feat method"

                # Priority 2: Hook features (fallback)
                elif 'feat' in penultimate_buf:
                    feat = penultimate_buf['feat']
                    vec = extract_feature_vector(feat)
                    extraction_method = f"Hook ({hook_location})"

                else:
                    print(f"⚠ No embeddings extracted for {getattr(row, video_col)}")
                    failed_videos.append(getattr(row, video_col))
                    continue

                # First video debug print
                if idx == 0:
                    print(f"\n✓ First video feature extraction:")
                    print(f"   Method: {extraction_method}")
                    print(f"   Feature shape: {vec.shape}")
                    print(f"   Feature stats: mean={vec.mean():.4f}, std={vec.std():.4f}")
                    print(f"   Feature range: [{vec.min():.4f}, {vec.max():.4f}]")
                    print(f"   Sample values: {vec[:5]}")

                if np.isnan(vec).any() or np.isinf(vec).any():
                    print(f"⚠ Invalid features (NaN/Inf) for {getattr(row, video_col)}")
                    failed_videos.append(getattr(row, video_col))
                    continue

                features_list.append(vec)
                mos_value = getattr(row, mos_col)
                mos_list.append(mos_value)
                names.append(getattr(row, video_col))

        except Exception as e:
            print(f"✗ Error with {getattr(row, video_col)}: {e}")
            import traceback
            traceback.print_exc()
            failed_videos.append(getattr(row, video_col))
            continue

    # -------- Save Features --------
    if len(features_list) == 0:
        print(f"\n✗ No features extracted for {split_name}.")
        continue

    X = np.vstack(features_list)
    y = np.asarray(mos_list, dtype=np.float32)
    names_array = np.asarray(names)

    print(f"\n📊 {split_name.upper()} Stats:")
    print(f"   Samples: {X.shape[0]}/{len(df)}")
    print(f"   Feature shape: {X.shape}")
    print(f"   Feature mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"   MOS range: [{y.min():.2f}, {y.max():.2f}]")

    corr = np.corrcoef(X.mean(axis=1), y)[0, 1]
    print(f"   Correlation (mean feat vs MOS): {corr:.4f}")

    if NORMALIZE_FEATURES:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        np.save(os.path.join(out_dir, f'features_{split_name}_embeddings.npy'), X_norm)
        np.save(os.path.join(out_dir, f'features_{split_name}_embeddings_raw.npy'), X)
        joblib.dump(scaler, os.path.join(out_dir, f'scaler_{split_name}.pkl'))
        print("   ✓ Features normalized and saved.")
    else:
        np.save(os.path.join(out_dir, f'features_{split_name}_embeddings.npy'), X)
        print("   ✓ Features saved without normalization.")

    np.save(os.path.join(out_dir, f'features_{split_name}_mos.npy'), y)
    np.save(os.path.join(out_dir, f'features_{split_name}_names.npy'), names_array)

    if failed_videos:
        print(f"\n⚠ Failed videos ({len(failed_videos)}): {failed_videos[:5]}{'...' if len(failed_videos)>5 else ''}")

# -------- Cleanup --------
if handle:
    handle.remove()
    print(f"\n✓ Hook removed from {hook_location}")

print("\n" + "="*70)
print("✅ All feature embeddings extracted successfully!")
print("="*70)
print(f"\nOutput directory: {out_dir}")
print("\nSaved files per split:")
for s in splits.keys():
    print(f"  - features_{s}_embeddings.npy")
    print(f"  - features_{s}_embeddings_raw.npy")
    print(f"  - features_{s}_mos.npy")
    print(f"  - features_{s}_names.npy")
    print(f"  - scaler_{s}.pkl\n")
print("✓ Ready for downstream probing / visualization.")
