# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.models import (ActionDataPreprocessor, VideoQualityRecognizer,
                             SwinTransformer3D)
from mmaction.models.heads import MultitaskHead
model = dict(
    type=VideoQualityRecognizer,
    backbone=dict(
        type=SwinTransformer3D,
        arch='tiny',
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    data_preprocessor=dict(
        type=ActionDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type=MultitaskHead,
        in_channels=768,
        num_quality_classes=5,     # Good/Bad, can be changed if you extend
        num_distortion_types=7,    # number of distortion categories including 'original'
        dropout_ratio=0.5,
        average_clips='score',
        loss_mos=dict(type='MSELoss', loss_weight=1.0),
        loss_qclass=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_dtype=dict(type='CrossEntropyLoss', loss_weight=1.0)
    )
)
