_base_ = [
    '../../_base_/models/swin_MOS.py', '../../_base_/default_runtime.py'
]

# Model settings - Custom VQA setup
model = dict(
    type='swin_MOSRecognizer',
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
    ),
    cls_head=dict(
        type='swin_MOSHead',
        num_classes=4,
        in_channels=768,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.5),
        loss_mos=dict(type='MSELoss', loss_weight=1.5)
    )
)

# Dataset settings
dataset_type = 'swin_MOSData'
data_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\k150kb'
ann_file_train = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_train.csv'
ann_file_val = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_val.csv'
ann_file_test = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_test.csv'

file_client_args = dict(io_backend='disk')

# Training pipeline - SIMPLIFIED, remove collect_keys
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),  # REMOVE collect_keys - use default behavior
]

# Validation pipeline
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),  # REMOVE collect_keys
]

# Test pipeline
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),  # REMOVE collect_keys
]

# Dataloader settings
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

# Evaluators
val_evaluator = dict(type='swin_MOSMetric')
test_evaluator = val_evaluator

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=50,
    val_begin=1, 
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        lr=2e-4,
        betas=(0.9, 0.999), 
        weight_decay=0.01),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.0)),
    clip_grad=dict(max_norm=1.0, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=30)
]

# Hooks
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3, 
        save_best='swin_MOS/SROCC',
        rule='greater'),
    logger=dict(interval=20))

# Work directory
work_dir = './work_dirs/swin_mos_konvid150k'

# Disable automatic LR scaling
auto_scale_lr = dict(enable=False)
find_unused_parameters = False
