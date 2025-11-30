_base_ = [
    '../../_base_/models/swin_MOS.py', '../../_base_/default_runtime.py'
]

# Model settings - Custom VQA setup
model = dict(
    type='swin_MOSRecognizer',  # Your custom recognizer
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
        lr_mult=0.0  # Freeze backbone to save memory
    ),
    cls_head=dict(
        type='swin_MOSHead',  # Your custom dual-task head
        num_classes=4,  # 4 quality classes
        in_channels=768,  # Swin Tiny output channels
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.5),
        loss_mos=dict(type='MSELoss', loss_weight=1.5)
    )
)

# Dataset settings - Konvid-150k-b
dataset_type = 'swin_MOSData'  # Your custom dataset
data_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\k150kb'  # Update with your video path
ann_file_train = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_train.csv'  # ~1260 videos (80%)
ann_file_val = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_val.csv'      # ~158 videos (10%)
ann_file_test = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_test.csv'    # ~158 videos (10%)

file_client_args = dict(io_backend='disk')

# Training pipeline - Reduced frames for memory
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),  # Reduced to 16 frames
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Validation pipeline
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,  # Reduced to 16 frames
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,  # Reduced to 16 frames
        frame_interval=2,
        num_clips=2,  # Reduced from 4 to 2 clips
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),  # Changed from ThreeCrop to CenterCrop
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Dataloader settings - Optimized for 6GB VRAM
train_dataloader = dict(
    batch_size=2,  # Small batch for 6GB VRAM
    num_workers=4,  # Reduced workers for laptop
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

# Custom VQA evaluator
val_evaluator = dict(type='swin_MOSMetric')  # Your custom metric (SROCC, PLCC, RMSE)
test_evaluator = val_evaluator

# Training configuration - Adjusted for small dataset
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=30,  # Reasonable for 1576 videos
    val_begin=1, 
    val_interval=1)  # Validate every epoch (dataset is small)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - Only training the head (backbone frozen)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        lr=5e-4,  # Moderate LR for small dataset
        betas=(0.9, 0.999), 
        weight_decay=0.01),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.0)),  # Frozen backbone
    clip_grad=dict(max_norm=1.0, norm_type=2))  # Gradient clipping for stability

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=3,  # Short warmup
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=30)
]

# Hooks configuration
default_hooks = dict(
    checkpoint=dict(
        interval=1,  # Save every epoch (small dataset, fast epochs)
        max_keep_ckpts=3, 
        save_best='swin_MOS/SROCC',  # Save best SROCC
        rule='greater'),
    logger=dict(interval=20))  # Log frequently for monitoring

# Disable automatic LR scaling (single GPU)
auto_scale_lr = dict(enable=False)

# Work directory
work_dir = './work_dirs/swin_mos_konvid150k'

# Memory optimization settings
find_unused_parameters = False