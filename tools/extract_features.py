"""Extract video features and save as CSV for fusion module.

Extracts feature embeddings from videos using Video Swin Transformer.
Saves as single CSV with columns: filename, mos, quality_class, distortion_type, feat_0, feat_1, ..., feat_N
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from mmaction.datasets import VideoQualityDataset
from mmaction.registry import DATASETS, MODELS


def build_video_swin_backbone():
    """Build Video Swin Transformer backbone with correct parameters."""
    
    # Correct parameter names for MMAction2's SwinTransformer3D
    backbone_cfg = dict(
        type='SwinTransformer3D',
        arch='tiny',  # tiny, small, base, or large
        pretrained=None,
        pretrained2d=False,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True
    )
    
    backbone = MODELS.build(backbone_cfg)
    return backbone


def extract_features_to_csv(dataset, backbone, output_csv='outputs/video_embeddings.csv'):
    """Extract features from all videos and save as CSV."""
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Set model to eval mode
    backbone.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = backbone.to(device)
    
    if torch.cuda.is_available():
        print("Using GPU for feature extraction")
    else:
        print("Using CPU for feature extraction")
    
    # Store results
    results = []
    
    print(f"\nExtracting features from {len(dataset)} videos...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing videos"):
            # Get sample
            sample = dataset[idx]
            inputs = sample['inputs']  # [1, 3, 8, 224, 224]
            data_sample = sample['data_samples']
            
            # Remove the clip dimension: [1, 3, 8, 224, 224] -> [3, 8, 224, 224]
            if inputs.dim() == 5 and inputs.shape[0] == 1:
                inputs = inputs.squeeze(0)  # [3, 8, 224, 224]
            
            # Add batch dimension: [3, 8, 224, 224] -> [1, 3, 8, 224, 224]
            inputs = inputs.unsqueeze(0)
            
            # Convert to float32 and normalize to [0, 1]
            inputs = inputs.float() / 255.0
            
            # Move to device
            inputs = inputs.to(device)
            
            # Extract features using backbone
            features = backbone(inputs)
            
            # Handle Video Swin output (returns tuple)
            if isinstance(features, tuple):
                features = features[-1]  # Take last feature map
            
            # Global average pooling: [B, C, T, H, W] -> [B, C]
            if features.ndim > 2:
                features = features.mean(dim=[-3, -2, -1])
            
            # Move to CPU and convert to numpy
            features = features.cpu().numpy().squeeze(0)  # [feature_dim]
            
            # Get metadata
            filename = data_sample.metainfo.get('filename', f'video_{idx:04d}.mp4')
            filename = os.path.basename(filename)
            
            # Create row dict
            row = {
                'filename': filename,
                'mos': data_sample.mos,
                'quality_class': data_sample.quality_class,
                'distortion_type': data_sample.distortion_type,
            }
            
            # Add feature columns: feat_0, feat_1, ..., feat_N
            for feat_idx, feat_val in enumerate(features):
                row[f'feat_{feat_idx}'] = feat_val
            
            results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("Feature Extraction Complete")
    print("="*60)
    print(f"Total videos processed: {len(df)}")
    print(f"Feature dimension: {features.shape[0]}")
    print(f"Output CSV: {output_csv}")
    print(f"CSV shape: {df.shape}")
    print("\nCSV columns:")
    print(f"  - filename, mos, quality_class, distortion_type")
    print(f"  - feat_0 to feat_{features.shape[0]-1} ({features.shape[0]} features)")
    print("="*60)
    
    # Print sample
    print("\nSample (first 2 rows, first 5 features):")
    sample_cols = ['filename', 'mos', 'quality_class', 'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4']
    print(df[sample_cols].head(2))
    
    return df


def main():
    """Main extraction script."""
    
    print("="*60)
    print("Video Feature Extraction to CSV")
    print("="*60)
    
    # Build dataset
    print("\n1. Loading dataset...")
    pipeline_cfg = [
        dict(type='DecordInit'),
        dict(type='UniformSampleFrames', clip_len=8, num_clips=1, test_mode=False),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 256), keep_ratio=True),
        dict(type='CenterCrop', crop_size=224),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='VideoQualityPack', meta_keys=('img_shape', 'num_clips', 'clip_len', 'filename'))
    ]
    
    dataset_cfg = dict(
        type='VideoQualityDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryMeta\demo_metadata.csv',
        data_root='.',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos'),
        pipeline=pipeline_cfg
    )
    
    dataset = DATASETS.build(dataset_cfg)
    print(f"   Loaded {len(dataset)} videos")
    
    # Build Video Swin backbone
    print("\n2. Building Video Swin Transformer backbone...")
    backbone = build_video_swin_backbone()
    print("   Backbone initialized (Video Swin-Tiny)")
    print("   Note: Using random initialization (no pretrained weights)")
    print("   For better features, use pretrained weights or trained model")
    
    # Extract features
    print("\n3. Extracting features...")
    df = extract_features_to_csv(
        dataset, 
        backbone,
        output_csv=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\video_embeddings.csv'
    )
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
