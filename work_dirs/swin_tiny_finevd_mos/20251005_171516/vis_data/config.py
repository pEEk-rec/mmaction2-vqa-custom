ann_file_test = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\Annotations\\test.json'
ann_file_train = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\Annotations\\train.json'
ann_file_val = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\Annotations\\val.json'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\videos_all\\videos_all_videos'
dataset_type = 'FineVDDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=5,
        rule='greater',
        save_best='SRCC',
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        arch='tiny',
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        mlp_ratio=4.0,
        patch_norm=True,
        patch_size=(
            2,
            4,
            4,
        ),
        pretrained=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\checkpoints\\swin_tiny_patch244_window877_kinetics400_1k.pth',
        pretrained2d=False,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer3D',
        window_size=(
            8,
            7,
            7,
        )),
    cls_head=dict(
        dropout_ratio=0.5,
        in_channels=768,
        loss_cls=dict(type='MSELoss'),
        mos_range=(
            1.0,
            5.0,
        ),
        num_classes=5,
        spatial_type='avg',
        type='MOSRegressionHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    constructor='SwinOptimWrapperConstructor',
    optimizer=dict(
        accumulative_counts=4,
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2.5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=50,
        begin=0,
        by_epoch=True,
        end=50,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
resume_from = None
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\Annotations\\test.json',
        data_prefix=dict(
            video=
            'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\videos_all\\videos_all_videos'
        ),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=2,
                num_clips=4,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='FineVDDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='MOSMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=2,
        num_clips=4,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=30, type='EpochBasedTrainLoop', val_begin=1, val_interval=5)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\Annotations\\train.json',
        data_prefix=dict(
            video=
            'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\videos_all\\videos_all_videos'
        ),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=2,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='FineVDDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=16, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\Annotations\\val.json',
        data_prefix=dict(
            video=
            'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_dataset\\videos_all\\videos_all_videos'
        ),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='FineVDDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='MOSMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/swin_tiny_finevd_mos'
