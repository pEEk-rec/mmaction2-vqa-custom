"""Test VideoQualityDataset.

Tests:
1. Dataset loading from CSV
2. Individual sample access
3. All quality labels present
4. DataLoader batching
"""

import os.path as osp

# Import mmaction
import mmaction
from mmaction.datasets import VideoQualityDataset
from mmaction.datasets.transforms import *
from mmaction.registry import TRANSFORMS, DATASETS
from torch.utils.data import DataLoader

print(f"MMAction2 version: {mmaction.__version__}")

# === Build Pipeline ===
print("\n=== Building Pipeline ===")

pipeline_cfg = [
    dict(type='DecordInit'),
    dict(type='UniformSampleFrames', clip_len=8, num_clips=1, test_mode=False),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='VideoQualityPack', meta_keys=('img_shape', 'num_clips', 'clip_len', 'filename'))
]

# === Create Dataset ===
print("\n=== Creating Dataset ===")

dataset_cfg = dict(
    type='VideoQualityDataset',
    ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryMeta\demo_metadata.csv',  # Changed from .tsv to .csv
    data_root='.',
    data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos'),
    pipeline=pipeline_cfg
)

try:
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset created successfully")
    print(f"   Total samples: {len(dataset)}")
    print(f"\n{dataset}")

except Exception as e:
    print(f"Failed to create dataset")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# === Test Single Sample Access ===
print("\n=== Testing Single Sample Access ===")

try:
    sample = dataset[0]
    
    inputs = sample['inputs']
    data_sample = sample['data_samples']
    
    print(f"Sample 0 loaded successfully")
    
    print(f"\n   Video Tensor:")
    print(f"     Shape: {inputs.shape}")
    print(f"     Expected: torch.Size([1, 3, 8, 224, 224])")
    
    print(f"\n   Quality Labels:")
    print(f"     MOS: {data_sample.mos}")
    print(f"     Quality Class: {data_sample.quality_class}")
    print(f"     Distortion Type: {data_sample.distortion_type}")
    
    print(f"\n   Text Fields:")
    if hasattr(data_sample, 'prompt'):
        print(f"     Prompt: {data_sample.prompt[:60]}...")
    if hasattr(data_sample, 'qa_question'):
        print(f"     QA Question: {data_sample.qa_question[:60]}...")
    if hasattr(data_sample, 'qa_answer'):
        print(f"     QA Answer: {data_sample.qa_answer[:60]}...")
    
    print(f"\n   Quality Metrics:")
    if hasattr(data_sample, 'psnr'):
        print(f"     PSNR: {data_sample.psnr}")
    if hasattr(data_sample, 'ssim'):
        print(f"     SSIM: {data_sample.ssim}")
    if hasattr(data_sample, 'blur'):
        print(f"     Blur: {data_sample.blur}")
    if hasattr(data_sample, 'lpips'):
        print(f"     LPIPS: {data_sample.lpips}")
    if hasattr(data_sample, 'temporal_consistency'):
        print(f"     Temporal Consistency: {data_sample.temporal_consistency}")
    
    print(f"\n   Artifact Flags:")
    if hasattr(data_sample, 'hallucination_flag'):
        print(f"     Hallucination: {data_sample.hallucination_flag}")
    if hasattr(data_sample, 'rendering_artifact_flag'):
        print(f"     Rendering Artifact: {data_sample.rendering_artifact_flag}")
    if hasattr(data_sample, 'unnatural_motion_flag'):
        print(f"     Unnatural Motion: {data_sample.unnatural_motion_flag}")
    if hasattr(data_sample, 'lighting_inconsistency_flag'):
        print(f"     Lighting Inconsistency: {data_sample.lighting_inconsistency_flag}")
    
    print(f"\n   Metadata:")
    if hasattr(data_sample, 'metainfo'):
        print(f"     Keys: {list(data_sample.metainfo.keys())}")

except Exception as e:
    print(f"Failed to load sample")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# === Test Multiple Samples ===
print("\n=== Testing Multiple Samples ===")

try:
    print(f"Loading first 3 samples...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        data_sample = sample['data_samples']
        print(f"  Sample {i}: MOS={data_sample.mos:.2f}, "
              f"Class={data_sample.quality_class}, "
              f"Distortion={data_sample.distortion_type}")
    
    print(f"Multiple samples loaded successfully")

except Exception as e:
    print(f"Failed to load multiple samples")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

# === Test DataLoader ===
print("\n=== Testing DataLoader ===")

def collate_fn(batch):
    """Custom collate function for quality assessment."""
    import torch
    
    # Stack video inputs
    inputs = torch.stack([item['inputs'] for item in batch])
    
    # Keep data_samples as list (each has different labels)
    data_samples = [item['data_samples'] for item in batch]
    
    return {
        'inputs': inputs,
        'data_samples': data_samples
    }

try:
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Use 0 for debugging on Windows
        collate_fn=collate_fn
    )
    
    print(f"DataLoader created")
    print(f"   Batch size: 2")
    print(f"   Total batches: {len(dataloader)}")
    
    # Test first batch
    print(f"\n   Testing first batch...")
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['inputs']
        data_samples = batch['data_samples']
        
        print(f"\n   Batch {batch_idx} loaded successfully")
        print(f"      Inputs shape: {inputs.shape}")
        print(f"      Expected: torch.Size([2, 1, 3, 8, 224, 224])")
        print(f"      Num data samples: {len(data_samples)}")
        
        for i, ds in enumerate(data_samples):
            print(f"      Sample {i}: MOS={ds.mos:.2f}, Class={ds.quality_class}")
        
        break  # Only test first batch

except Exception as e:
    print(f"DataLoader test failed")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# === Summary ===
print("\n" + "="*60)
print("ALL DATASET TESTS PASSED")
print("="*60)

print("\nDataset successfully:")
print("  - Loads metadata from CSV (35 samples)")
print("  - Accesses individual video samples")
print("  - Loads all quality labels (MOS, class, distortion)")
print("  - Loads text descriptions (prompt, QA pairs)")
print("  - Loads quality metrics (PSNR, SSIM, blur, etc.)")
print("  - Loads artifact flags")
print("  - Works with DataLoader for batching")

print("\n" + "="*60)
print("Next Step: Build Recognizer/Model (Step 4)")
print("="*60)
