import sys
sys.path.insert(0, '.')  # Add current directory to path

# Import mmaction2 to register all modules
import mmaction  # This registers all transforms, datasets, etc.

import torch
from mmengine.config import Config

# Load config
cfg = Config.fromfile('configs/recognition/swin/swin_MOSConfig.py')

# Build dataloader
from mmaction.registry import DATASETS

dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(f"Dataset size: {len(dataset)}")

# Get one sample
sample = dataset[0]
print(f"\n=== First Sample ===")
print(f"Sample keys: {sample.keys()}")
print(f"Inputs type: {type(sample['inputs'])}")
print(f"Inputs shape: {sample['inputs'].shape}")
print(f"Inputs dtype: {sample['inputs'].dtype}")

# Check data_samples
data_sample = sample['data_samples']
print(f"\n=== Data Sample ===")
print(f"Type: {type(data_sample)}")
print(f"Attributes: {[attr for attr in dir(data_sample) if not attr.startswith('_')]}")

# Check for labels
if hasattr(data_sample, 'gt_label'):
    print(f"GT Label: {data_sample.gt_label}")
if hasattr(data_sample, 'gt_labels'):
    print(f"GT Labels: {data_sample.gt_labels}")
if hasattr(data_sample, 'mos'):
    print(f"MOS: {data_sample.mos}")
if hasattr(data_sample, 'gt_mos'):
    print(f"GT MOS: {data_sample.gt_mos}")
if hasattr(data_sample, 'quality_class'):
    print(f"Quality class: {data_sample.quality_class}")

print("\n=== Testing Batch ===")
# Test batch loading
from torch.utils.data import DataLoader
from mmengine.dataset import default_collate

dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=0,
    collate_fn=default_collate
)

batch = next(iter(dataloader))
print(f"Batch inputs shape: {batch['inputs'].shape}")
print(f"Batch data_samples length: {len(batch['data_samples'])}")