_base_ = [
    '../../_base_/models/swin_AI-VQA.py',  # FIXED: Match your base model file
    '../../_base_/default_runtime.py'
]

# ============================================================================
# MODEL SETTINGS
# ============================================================================
model = dict(
    type='swin_AIRecognizer',
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
    ),
    cls_head=dict(
        type='swin_AIHead',  # CORRECT: Your actual class name
        num_classes=5,       # 5-class quality classification
        in_channels=768,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob',

        # --- Core losses ---
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.5),
        loss_mos=dict(type='MSELoss', loss_weight=1.5),

        # --- Artifact-specific losses (FIXED: CrossEntropyLoss for 2-class) ---
        loss_hallucination=dict(
            type='CrossEntropyLoss', 
            loss_weight=0.8,
            class_weight=[1.0, 2.07]  # Weight for class imbalance
        ),
        loss_lighting=dict(
            type='CrossEntropyLoss', 
            loss_weight=0.8,
            class_weight=[1.0, 3.26]
        ),
        loss_spatial=dict(
            type='CrossEntropyLoss', 
            loss_weight=0.8,
            class_weight=[1.0, 9.36]
        ),
        # NOTE: rendering_flag not trained (no loss defined)
    )
)

# ============================================================================
# DATASET SETTINGS
# ============================================================================
dataset_type = 'swin_AIDataset'
data_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\standardized_videos'

ann_file_train = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\train.csv'
ann_file_val   = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\val.csv'
ann_file_test  = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\test.csv'

# AI-Gen specific configuration
quality_classes = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']  # 5 classes
mos_scale = (0, 100)  # MOS range: 0-100

# Artifact columns tracked
ai_artifact_columns = [
    'hallucination_flag',  
    'lighting_flag',       
    'spatial_flag',        
    'rendering_flag',      
    'blurriness', 'gradient', 'lpips', 'temporal_consistency'  # Metadata
]

file_client_args = dict(io_backend='disk')

# ============================================================================
# PIPELINES (IDENTICAL TO KoNViD FOR FAIR COMPARISON)
# ============================================================================
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=(  # FIXED: Added space
        'filename',
        'mos',
        'quality_class',
        'hallucination_flag',
        'lighting_flag',
        'spatial_flag',
        'rendering_flag',
        'blurriness',
        'gradient',
        'lpips',
        'temporal_consistency'
    )),
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=(  # FIXED: Added space
        'filename',
        'mos',
        'quality_class',
        'hallucination_flag',
        'lighting_flag',
        'spatial_flag',
        'rendering_flag',
        'blurriness',
        'gradient',
        'lpips',
        'temporal_consistency'
    )),
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=(  # FIXED: Added space
        'filename',
        'mos',
        'quality_class',
        'hallucination_flag',
        'lighting_flag',
        'spatial_flag',
        'rendering_flag'
    )),
]

# ============================================================================
# DATALOADER SETTINGS (Optimized for RTX 4050 6GB + 16GB RAM)
# ============================================================================
train_dataloader = dict(
    batch_size=2,       # Safe for 6GB VRAM
    num_workers=4,      # 4 workers for good CPU utilization
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
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
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
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True)
)

# ============================================================================
# METRICS & EVALUATION
# ============================================================================
val_evaluator = dict(type='swin_AIMetric')
test_evaluator = val_evaluator

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=200, 
    val_begin=1, 
    val_interval=3
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# OPTIMIZATION SETTINGS (Mixed Precision AMP)
# ============================================================================
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Enables mixed precision (float16)
    optimizer=dict(
        type='AdamW',
        lr=3e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    ),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.0)  # FROZEN BACKBONE for Stage 0
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.1, 
        by_epoch=True, 
        begin=0, 
        end=5, 
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR', 
        T_max=195, 
        eta_min=1e-7, 
        by_epoch=True, 
        begin=100, 
        end=200
    )
]

# ============================================================================
# HOOKS & LOGGING
# ============================================================================
default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=2,
        save_best='mos_srcc',  # Your metric key
        rule='greater',
        type='CheckpointHook'
    ),
    logger=dict(interval=200, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        vis_backends=[
            dict(type='LocalVisBackend'),
            dict(type='TensorboardVisBackend', log_dir='./work_dirs/swin_Ai-VQA/tb')
        ]
    )
)

# Add early stopping here (separate from default_hooks)
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mos_srcc',  # Match your metric key
        rule='greater',
        patience=5,  # Stop after 5 validations without improvement
        min_delta=0.005,  # Require 0.005 SRCC improvement
        strict=False
    )
]


# REMOVED: Visualization hook (not needed for Stage 0 baseline assessment)

# ============================================================================
# CUDA & ENVIRONMENT SETTINGS
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=True,  # Optimize conv kernels for performance
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# ============================================================================
# OTHER SETTINGS
# ============================================================================
work_dir = './work_dirs/swin_Ai-VQA'
auto_scale_lr = dict(enable=False)
find_unused_parameters = False  # FIXED: Should be False for frozen backbone
device = 'cuda'

# ============================================================================
# NOTES FOR AI-GENERATED DATASET TRAINING
# ============================================================================
# HARDWARE:
# - Optimized for NVIDIA RTX 4050 (6GB VRAM) + Ryzen 7 CPU
# - AMP enabled for 2× faster training and 40% less VRAM usage
# - Batch size 2 is safe; can try 4 if stable
#
# MODEL:
# - Backbone: Frozen Swin-Tiny (768-dim features)
# - Tasks: MOS regression (0-100) + 5-class quality + 3 artifact flags
# - Total trainable parameters: ~2M (only heads, backbone frozen)
#
# TRAINING:
# - 30 epochs recommended for convergence (≈2.5–3 hours on RTX 4050)
# - Best model saved based on SRCC (Spearman correlation)
# - Validation every epoch starting from epoch 1
#
# ARTIFACTS:
# - hallucination_flag: Trained (weight 0.8, class imbalance 2.07)
# - lighting_flag: Trained (weight 0.8, class imbalance 3.26)
# - spatial_flag: Trained (weight 0.8, class imbalance 9.36)
# - rendering_flag: Loaded but NOT trained (analysis only)
#
# STAGE 0 PURPOSE:
# - Assess frozen encoder capability on AI-generated videos
# - Compare with UGC dataset results (same frozen encoder)
# - Identify AI-specific artifact detection capability
# ============================================================================
 