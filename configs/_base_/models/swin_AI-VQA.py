# Copyright (c) OpenMMLab. All rights reserved.
# Base model configuration for AI-Generated Video Quality Assessment

model = dict(
    type='swin_MOSRecognizer',  # String reference to custom recognizer class
    backbone=dict(
        type='SwinTransformer3D',  # Built-in MMAction2 Video Swin Transformer
        arch='tiny',  # Swin-Tiny architecture
        pretrained=None,  # Will be set in main config
        pretrained2d=True,  # Load 2D ImageNet weights and inflate to 3D
        patch_size=(2, 4, 4),  # Temporal, Height, Width patch dimensions
        window_size=(8, 7, 7),  # Temporal, Height, Width window for attention
        mlp_ratio=4.,  # MLP hidden dimension ratio
        qkv_bias=True,  # Use bias in QKV projection
        qk_scale=None,  # Auto-calculate Q-K scaling
        drop_rate=0.,  # Dropout rate
        attn_drop_rate=0.,  # Attention dropout rate
        drop_path_rate=0.1,  # Stochastic depth rate
        patch_norm=True),  # Normalize patches
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet mean (RGB)
        std=[58.395, 57.12, 57.375],  # ImageNet std (RGB)
        format_shape='NCTHW'),  # [Batch, Channel, Time, Height, Width]
    cls_head=dict(
        type='swin_AIHead',  # CHANGED: AI-Gen specific head (handles 5 classes)
        in_channels=768,  # Swin-Tiny output dimension
        num_classes=5,  # CHANGED: 4 → 5 (Bad/Poor/Fair/Good/Excellent)
        spatial_type='avg',  # Global average pooling
        dropout_ratio=0.5,  # Dropout before final layers
        average_clips='prob',  # Average predictions across clips
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.5),  # Classification loss
        loss_mos=dict(type='MSELoss', loss_weight=1.5)))  # MOS regression loss

# ============================================================================
# CHANGES FROM KONVID (UGC) VERSION:
# ============================================================================
# 1. type='swin_MOSHead_AIGen' - New head class that supports:
#    - 5 quality classes instead of 4
#    - AI-specific artifact outputs (hallucination, rendering)
#    - Optional artifact detection heads
#
# 2. num_classes=5 - Includes 'Excellent' class for AI-generated videos
#
# 3. All other parameters IDENTICAL to ensure fair comparison:
#    - Same backbone architecture (Swin-Tiny)
#    - Same preprocessing (ImageNet mean/std)
#    - Same loss weights (0.5 for cls, 1.5 for MOS)
#    - Same dropout, pooling, etc.
#
# BACKBONE OUTPUT:
# - Feature dimension: 768 (from Swin-Tiny)
# - This will be used for:
#   * Stage 0: MOS regression + 5-class classification (frozen backbone)
#   * Stage 1: Input to fusion module for multimodal VQA alignment
#   * UMAP visualization and artifact analysis
#
# TRAINING STRATEGY:
# - Backbone will be FROZEN (lr_mult=0.0 in optimizer config)
# - Only train: cls_head (5-class + MOS) and optional artifact detection heads
# - This provides fair comparison with KoNViD results
# ============================================================================