# configs/recognition/swin/swin_tiny_finevd_mos.py

_base_ = [
    '../../_base_/models/swin_tiny.py', 
    '../../_base_/default_runtime.py'
]

# Model settings
model = dict(
    backbone=dict(
        arch='tiny',  # tiny, small, base
        drop_path_rate=0.2,
        pretrained=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\checkpoints\swin_tiny_patch244_window877_kinetics400_1k.pth'
    ),
    cls_head=dict(
        type='MOSRegressionHead',
        in_channels=768,  # 768 for tiny, 1024 for base
        num_classes=5,
        dropout_ratio=0.3,
        
     # Add if supported
        mos_range=(1.0, 5.0)  # Adjust based on your MOS range
    )
)

# Dataset settings
dataset_type = 'FineVDDataset'
data_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\videos_all\videos_all_videos'
ann_file_train = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\Annotations\train.json'
ann_file_val = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\Annotations\val.json'
ann_file_test = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\Annotations\test.json'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
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
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Dataloader settings
train_dataloader = dict(
    batch_size=2,  # Adjust based on GPU memory
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True
    )
)

# Evaluation settings
val_evaluator = dict(type='MOSMetric')

test_evaluator = val_evaluator

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    val_begin=1,
    val_interval=5
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=1,
    optimizer=dict(
        type='AdamW',
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.02
    ),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.8)  # Lower LR for pretrained backbone
    ),
    clip_grad=dict(max_norm=40, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.2,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=50
    )
]

# Hooks
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=5,
        save_best='mos/SRCC',
        rule='greater'
    ),
    logger=dict(interval=50)
)

#Custom hooks for performance analysis
custom_hooks = [
    dict(type='EMAHook', momentum=0.0001, priority='ABOVE_NORMAL'),
    dict(
        type='PerformanceAnalysisHook',
        input_shape=(3, 32, 224, 224),
        interval=50,
        log_flops_once=True
    )
]

# Runtime settings
work_dir = './work_dirs/swin_tiny_finevd_mos'

# Auto scale learning rate
auto_scale_lr = dict(enable=False, base_batch_size=64)

# Load pretrained weights
load_from = None
resume_from = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\swin_tiny_finevd_mos\epoch_5.pth'