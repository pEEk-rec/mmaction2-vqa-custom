from mmengine.config import read_base
from mmaction.models.heads.i3d_regression_head import I3DRegressionHead

with read_base():
    from ..._base_.models.swin_tiny import *
    from ..._base_.default_runtime import *

# Configure model for regression task
model['backbone']['pretrained'] = None
model['backbone']['pretrained2d'] = False

# Replace classification head with regression head for video quality assessment
model['cls_head'] = dict(
    type='I3DRegressionHead',
    in_channels=768,  # Swin-T feature dimension
    num_classes=1,    # Single quality score output
    dropout_ratio=0.5,
    init_std=0.01,
)

# Optimizer configuration
from mmengine.optim import OptimWrapper
from mmaction.engine import SwinOptimWrapperConstructor
from mmengine.runner import EpochBasedTrainLoop, ValLoop, TestLoop
from mmaction.evaluation import MSEMetric  # Use regression metric

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type='torch.optim.AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.02),
    constructor=SwinOptimWrapperConstructor,
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

# Training configuration
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# Use regression metrics instead of accuracy
val_evaluator = dict(type=MSEMetric)  # Consider implementing PLCC/SRCC for VQA
test_evaluator = val_evaluator

# Data pipeline
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Data loaders - consider increasing batch_size if GPU memory allows
train_dataloader = dict(
    batch_size=2,  # Increased from 1
    num_workers=2,  # Increased for faster data loading
    dataset=dict(
        type='VideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\train.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_train'),
        pipeline=train_pipeline,
        # to_float32=True  # Uncomment if needed
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='VideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\val.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_val'),
        pipeline=val_pipeline,
        test_mode=True,
        # to_float32=True
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='VideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\test.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_test'),  # Fixed path
        pipeline=test_pipeline,
        test_mode=True,
        # to_float32=True
    )
)