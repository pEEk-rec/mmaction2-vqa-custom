from mmengine.config import read_base
from mmaction.models.heads.i3d_regression_head import I3DRegressionHead

with read_base():
    # follow the same base imports as the original config
    from ..._base_.models.swin_tiny import *
    from ..._base_.default_runtime import *

# Train from scratch: disable pretrained backbone weights
model = dict(model)
backbone = dict(model['backbone'])
backbone['pretrained'] = None
# When training from scratch, don't attempt to inflate 2D pretrained weights
backbone['pretrained2d'] = False
model['backbone'] = backbone
model = dict(model)

# You can override other settings here if desired, for example:
# model['cls_head']['num_classes'] = 400
# train_cfg['max_epochs'] = 30

# Define an optim_wrapper compatible with CPU-only environments (so Runner is happy
# when train_dataloader/train_cfg are overridden via CLI). Use OptimWrapper instead
# of AmpOptimWrapper which requires GPU/NPU/MLU/MUSA.
from mmengine.optim import OptimWrapper
from mmaction.engine import SwinOptimWrapperConstructor
from mmengine.runner import EpochBasedTrainLoop, ValLoop, TestLoop
from mmaction.evaluation import MSEMetric

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type='torch.optim.AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor=SwinOptimWrapperConstructor,
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

# Training / evaluation loop configs (provide defaults so CLI overrides work)
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

val_evaluator = dict(type=MSEMetric)
test_evaluator = val_evaluator

# Default dataloaders (will be overridden by CLI options for the smoke run)
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

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='VideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\train.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_train'),
        pipeline=train_pipeline,
       # to_float32=True 
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='VideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\val.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_val'),
        pipeline=val_pipeline,
        test_mode=True,
        #to_float32=True 
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='VideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\test.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_val'),  # using val for testing
        pipeline=test_pipeline,
        test_mode=True,
       # to_float32=True 
    )
)
