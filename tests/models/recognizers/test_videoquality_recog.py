"""Test Video Quality Recognizer with Video Swin Transformer."""

import torch
import sys
sys.path.insert(0, '.')

from mmaction.registry import MODELS
from mmengine.structures import BaseDataElement
from mmaction.models.heads import VideoQualityHead


def test_recognizer():
    """Test Video Quality Recognizer."""
    
    print("="*60)
    print("Testing Video Quality Recognizer with Video Swin-T")
    print("="*60)
    
    # Build Video Swin-Tiny backbone config
    print("\n1. Building Video Swin-Tiny backbone...")
    backbone_cfg = dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained=None,
        pretrained2d=False,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True
    )
    
    # Build quality prediction head
    print("2. Building VideoQualityHead...")
    # Video Swin-Tiny output: 768 features
    cls_head_cfg = dict(
        type='VideoQualityHead',
        in_channels=768,
        hidden_dim=256,
        dropout_ratio=0.5
    )
    
    # Build complete recognizer
    print("3. Building VideoQualityRecognizer...")
    recognizer_cfg = dict(
        type='VideoQualityRecognizer',
        backbone=backbone_cfg,
        cls_head=cls_head_cfg
    )
    
    recognizer = MODELS.build(recognizer_cfg)
    print("   ✓ Recognizer built successfully")
    
    # Create dummy video input
    print("\n4. Creating dummy video input...")
    batch_size = 2
    inputs = torch.randn(batch_size, 3, 8, 224, 224)
    print(f"   Input shape: {inputs.shape}")
    print(f"   Format: [Batch, Channels, Frames, Height, Width]")
    
    # Create dummy data samples with MOS labels
    print("\n5. Creating data samples with MOS labels...")
    data_samples = []
    for i in range(batch_size):
        ds = BaseDataElement()
        ds.mos = 50.0 + i * 20.0  # MOS: 50.0, 70.0
        data_samples.append(ds)
    print(f"   Sample 0 MOS: {data_samples[0].mos}")
    print(f"   Sample 1 MOS: {data_samples[1].mos}")
    
    # Test feature extraction
    print("\n6. Testing feature extraction...")
    recognizer.eval()
    with torch.no_grad():
        feats = recognizer.extract_feat(inputs)
    print(f"   ✓ Feature shape: {feats.shape}")
    print(f"   Expected: [2, 768] (batch_size, feature_dim)")
    
    # Test loss calculation (training mode)
    print("\n7. Testing loss calculation...")
    recognizer.train()
    loss_dict = recognizer.loss(inputs, data_samples)
    print(f"   ✓ Loss: {loss_dict['loss'].item():.4f}")
    
    # Test prediction (inference mode)
    print("\n8. Testing prediction...")
    recognizer.eval()
    with torch.no_grad():
        predictions = recognizer.predict(inputs, data_samples)
    
    for i, pred in enumerate(predictions):
        print(f"   Sample {i}: GT MOS={data_samples[i].mos:.1f}, "
              f"Pred MOS={pred.pred_mos:.2f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nArchitecture Summary:")
    print("  Backbone: Video Swin Transformer (Tiny)")
    print("  Feature dim: 768")
    print("  Head: VideoQualityHead (768 → 256 → 1)")
    print("  Output: MOS prediction (single value)")
    print("="*60)


if __name__ == '__main__':
    # Import VideoQualityHead first
    
    test_recognizer()
