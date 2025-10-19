"""Test script for video quality assessment pipeline.

This script tests the pipeline in isolation (without dataset).
Only tests video loading and processing - no quality labels required.
"""

import os.path as osp

# Import mmaction and all transforms
import mmaction
from mmaction.datasets.transforms import *
from mmaction.registry import TRANSFORMS

print(f"MMAction2 version: {mmaction.__version__}")

# === Verify Transform Registration ===
required_transforms = [
    'DecordInit', 'UniformSampleFrames', 'DecordDecode', 
    'Resize', 'CenterCrop', 'FormatShape', 'VideoQualityPack'
]

print("\n=== Checking Transform Registration ===")
all_found = True
for t in required_transforms:
    if t in TRANSFORMS:
        print(f"✅ {t}")
    else:
        print(f"❌ {t} NOT FOUND")
        all_found = False

if not all_found:
    print("\n❌ Some transforms not registered!")
    print("Make sure VideoQualityPack is in mmaction/datasets/transforms/__init__.py")
    exit(1)

# === Define Pipeline Configuration ===
pipeline_cfg = [
    dict(type='DecordInit'),  # Initialize video reader
    dict(
        type='UniformSampleFrames',
        clip_len=8,           # Sample 8 frames
        num_clips=1,          # 1 clip for training
        test_mode=False       # Random sampling
    ),
    dict(type='DecordDecode'),  # Decode frames
    dict(
        type='Resize',
        scale=(-1, 256),      # Resize shorter side to 256
        keep_ratio=True
    ),
    dict(
        type='CenterCrop',
        crop_size=224         # Center crop to 224x224
    ),
    dict(
        type='FormatShape',
        input_format='NCTHW'  # Format to [N, C, T, H, W]
    ),
    dict(
        type='VideoQualityPack',
        meta_keys=('img_shape', 'num_clips', 'clip_len', 'filename')
    )
]

# === Build Pipeline ===
print("\n=== Building Pipeline ===")

class ManualPipeline:
    """Manual pipeline using MMAction2's TRANSFORMS registry."""
    
    def __init__(self, transforms_cfg, registry):
        self.transforms = []
        for cfg in transforms_cfg:
            transform = registry.build(cfg)
            self.transforms.append(transform)
    
    def __call__(self, results):
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            results = transform(results)
        return results

pipeline = ManualPipeline(pipeline_cfg, TRANSFORMS)
print("✅ Pipeline built successfully")

# === Test Video ===
video_path = osp.join(
    r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos',
    '00000_07_blur_demo.mp4'
)

if not osp.exists(video_path):
    print(f"\n❌ ERROR: Video not found at {video_path}")
    print(f"Current directory: {osp.dirname(osp.abspath(__file__))}")
    exit(1)

print(f"✅ Video found: {video_path}")

# === Prepare Test Data ===
# For pipeline testing, only filename is required
# Quality labels will be added by Dataset class later
results = dict(filename=video_path)

# === Run Pipeline ===
print("\n=== Running Pipeline ===")
try:
    packed_results = pipeline(results)
    
    inputs = packed_results['inputs']
    data_sample = packed_results['data_samples']
    
    print("\n=== Pipeline Output ===")
    print(f"✅ Input tensor shape: {inputs.shape}")
    print(f"   Expected shape: torch.Size([1, 3, 8, 224, 224])")
    
    # Verify shape
    expected_shape = (1, 3, 8, 224, 224)
    assert inputs.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {inputs.shape}"
    
    print(f"\n✅ Data sample created: {type(data_sample).__name__}")
    print(f"   Metadata keys: {list(data_sample.metainfo.keys())}")
    
    # Check metadata
    if hasattr(data_sample, 'metainfo'):
        print(f"   img_shape: {data_sample.metainfo.get('img_shape', 'N/A')}")
        print(f"   num_clips: {data_sample.metainfo.get('num_clips', 'N/A')}")
        print(f"   clip_len: {data_sample.metainfo.get('clip_len', 'N/A')}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE TEST SUCCESSFUL!")
    print("="*60)
    print("\nPipeline can successfully:")
    print("  ✅ Load video files")
    print("  ✅ Sample frames uniformly")
    print("  ✅ Decode and process frames")
    print("  ✅ Resize and crop to target size")
    print("  ✅ Format to correct tensor shape")
    print("  ✅ Pack into data sample")
    print("\nNext step: Test with Dataset class (Step 3)")

except Exception as e:
    print("\n" + "="*60)
    print("❌ PIPELINE TEST FAILED!")
    print("="*60)
    print(f"Error: {str(e)}\n")
    import traceback
    traceback.print_exc()
    exit(1)
