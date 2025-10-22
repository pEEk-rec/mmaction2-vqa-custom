# Copyright (c) OpenMMLab.
from mmengine.config import read_base

with read_base():
    from ..._base_.models.VQAswin_tiny import *
    from ..._base_.default_runtime import *

from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper, LinearLR
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim import AdamW
import mmaction.models
from mmaction.datasets import (
    CenterCrop, DecordDecode, DecordInit, FormatShape, VideoQualityPack,
    Resize, SampleFrames, VideoQualityDataset
)
from mmaction.engine import SwinOptimWrapperConstructor

# ============================================================
# CRITICAL FIX: Register custom modules BEFORE building model
# ============================================================
custom_imports = dict(
    imports=[
        'mmaction.models.data_preprocessors',
        'mmaction.models.recognizers.video_quality_recognizer',
        'mmaction.models.heads.VQA_multihead',
        'mmaction.datasets.VQA_dataset',
        'mmaction.datasets.transforms.video_quality_pack',
        'mmaction.evaluation.metrics.VQA_customMetric',
    ],
    allow_failed_imports=False
)

# -----------------------------
# Model: MultiTaskHead + robust losses
# -----------------------------
model.update(
    dict(
        type='VideoQualityRecognizer',
        data_preprocessor=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        backbone=dict(
            pretrained=
            'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth'
        ),
        cls_head=dict(
            type='MultiTaskHead',
            in_channels=768,
            num_quality_classes=5,      # adjust
            num_distortion_types=7,     # adjust
            dropout_ratio=0.5,
            average_clips='score',
            # Losses tuned for small VQA dry-run
            loss_mos=dict(
                type='SmoothL1Loss',
                beta=0.5,               # increase if MOS not normalized
                reduction='mean',
                loss_weight=1.0
            ),
            loss_qclass=dict(
                type='CrossEntropyLoss',
                loss_weight=1.0
            ),
            loss_dtype=dict(
                type='CrossEntropyLoss',
                loss_weight=1.0
            ),
        )
    )
)

# -----------------------------
# Dataset: dry-run on same list
# -----------------------------
dataset_type = VideoQualityDataset

data_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos'
ann_file_all = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryMeta\demo_metadata.csv'

file_client_args = dict(io_backend='disk')

# -----------------------------
# Pipelines: quality-preserving, deterministic
# -----------------------------
train_pipeline = [
    dict(type=DecordInit, **file_client_args),
    dict(type=SampleFrames, clip_len=32, frame_interval=2, num_clips=1),
    dict(type=DecordDecode),
    dict(type=Resize, scale=(-1, 256)),
    dict(type=CenterCrop, crop_size=224),
    dict(type=FormatShape, input_format='NCTHW'),
    dict(type=VideoQualityPack, meta_keys=('filename','img_shape','ori_shape','start_index','modality','num_clips','clip_len'))
]
val_pipeline = [
    dict(type=DecordInit, **file_client_args),
    dict(type=SampleFrames, clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type=DecordDecode),
    dict(type=Resize, scale=(-1, 256)),
    dict(type=CenterCrop, crop_size=224),
    dict(type=FormatShape, input_format='NCTHW'),
    dict(type=VideoQualityPack, meta_keys=('filename','img_shape','ori_shape','start_index','modality','num_clips','clip_len'))
]
test_pipeline = [
    dict(type=DecordInit, **file_client_args),
    dict(type=SampleFrames, clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type=DecordDecode),
    dict(type=Resize, scale=(-1, 256)),
    dict(type=CenterCrop, crop_size=224),
    dict(type=FormatShape, input_format='NCTHW'),
    dict(type=VideoQualityPack, meta_keys=('filename','img_shape','ori_shape','start_index','modality','num_clips','clip_len'))
]

# -----------------------------
# Dataloaders: same list for all splits (dry-run)
# -----------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_all,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_all,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_all,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True
    )
)

# -----------------------------
# Evaluators: placeholder (add MAE/RMSE/SRCC/PLCC next)
# -----------------------------
val_evaluator = dict(type='VQAMetric')
test_evaluator = val_evaluator

# -----------------------------
# Loops: 1-epoch dry run
# -----------------------------
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=3, val_begin=1, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# -----------------------------
# Optim: freeze backbone, train heads only
# -----------------------------
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor=SwinOptimWrapperConstructor,
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.0)   # freeze backbone
    )
)

# -----------------------------
# Scheduler: short/no-op for 1 epoch
# -----------------------------
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
]

# -----------------------------
# Hooks
# -----------------------------
default_hooks.update(
    dict(
        checkpoint=dict(interval=1, save_best='mos_mae', rule='less'),
        logger=dict(interval=10)
    )
)

auto_scale_lr = dict(enable=False, base_batch_size=64)