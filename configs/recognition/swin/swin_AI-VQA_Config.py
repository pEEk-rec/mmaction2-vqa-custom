_base_ = [
    '../../_base_/models/swin_AI-VQA.py',
    '../../_base_/default_runtime.py'
]

# ============================================================================
# MODEL SETTINGS
# ============================================================================
model = dict(
    backbone=dict(
        arch='tiny',  # tiny, small, base
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
    )
)

# ============================================================================
# DATASET SETTINGS
# ============================================================================
dataset_type = 'swin_AIDataset'
data_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\standardized_videos'

ann_file_train = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\train_new.csv'
ann_file_val   = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\val_new.csv'
ann_file_test  = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\test_new.csv'

file_client_args = dict(io_backend='disk')

# ============================================================================
# PIPELINES
# ============================================================================
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter',
         brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=(
        'filename', 'mos', 'quality_class',
        'hallucination_flag', 'lighting_flag', 'spatial_flag', 'rendering_flag',
        'blurriness', 'gradient', 'lpips', 'temporal_consistency'
    )),
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=(
        'filename', 'mos', 'quality_class',
        'hallucination_flag', 'lighting_flag', 'spatial_flag', 'rendering_flag',
        'blurriness', 'gradient', 'lpips', 'temporal_consistency'
    )),
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=(
        'filename', 'mos', 'quality_class',
        'hallucination_flag', 'lighting_flag', 'spatial_flag', 'rendering_flag'
    )),
]

# ============================================================================
# DATALOADERS
# ============================================================================
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True)
)

# ============================================================================
# METRICS
# ============================================================================
val_evaluator = dict(type='swin_AIMetric')
test_evaluator = val_evaluator

# ================================================================
# OPTIMIZER — Stable Frozen-Backbone AdamW (Best for Stage 0)
# ================================================================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=4e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        # fused AdamW -> faster + more stable on RTX 40-series (if supported)
        fused=True
    ),
    # Official Swin parameter rules (keeps backbone frozen cleanly)
    constructor='SwinOptimWrapperConstructor',
    
    paramwise_cfg=dict(
        # No decay for embeddings + bias terms
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        bias=dict(decay_mult=0.0),

        # Fully freeze backbone
        backbone=dict(lr_mult=0.0),
    ),

    # Smooth gradients for UW + BCE + MSE
    clip_grad=dict(max_norm=3.0)
)
# ================================================================
# LEARNING RATE SCHEDULE — Smooth & Stable
# ================================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,    
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),

    # Long cosine decay — best for SRCC stability and regression tasks
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
        end=150,
        T_max=145,
        eta_min=5e-5
    )
]


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# HOOKS
# ============================================================================
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='mos_srcc',  
        rule='greater',
        save_last=True
    ),
    logger=dict(interval=200, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mos_srcc',
        rule='greater',
        patience=25,
        min_delta=0.005,
        strict=False
    )
]

# ============================================================================
# ENV & MISC
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

work_dir = './work_dirs/swin_Ai-VQA_NEWDesign'
auto_scale_lr = dict(enable=False)
find_unused_parameters = False
device = 'cuda'
