ann_file_all = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryMeta\\demo_metadata.csv'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryVideos'
dataset_type = 'mmaction.datasets.VideoQualityDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=0, save_best='auto', type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        ignore_last=False, interval=10, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    runtime_info=dict(type='mmengine.hooks.RuntimeInfoHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    sync_buffers=dict(type='mmengine.hooks.SyncBuffersHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, type='mmengine.runner.LogProcessor', window_size=20)
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
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
        pretrained2d=True,
        qk_scale=None,
        qkv_bias=True,
        type='mmaction.models.SwinTransformer3D',
        window_size=(
            8,
            7,
            7,
        )),
    cls_head=dict(
        average_clips='score',
        dropout_ratio=0.5,
        in_channels=768,
        loss_dtype=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        loss_mos=dict(
            beta=0.5, loss_weight=1.0, reduction='mean', type='SmoothL1Loss'),
        loss_qclass=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_distortion_types=7,
        num_quality_classes=5,
        type='MultiTaskHead'),
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
    type='mmaction.models.VideoQualityRecognizer')
optim_wrapper = dict(
    constructor='mmaction.engine.SwinOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.001,
        type='torch.optim.AdamW',
        weight_decay=0.02),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.0),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='mmengine.optim.AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.1,
        type='mmengine.optim.LinearLR'),
]
randomness = dict(deterministic=True, diff_rank_seed=False, seed=0)
resume = False
test_cfg = dict(type='mmengine.runner.TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryMeta\\demo_metadata.csv',
        data_prefix=dict(
            video=
            'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryVideos'
        ),
        pipeline=[
            dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='mmaction.datasets.SampleFrames'),
            dict(type='mmaction.datasets.DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='mmaction.datasets.Resize'),
            dict(crop_size=224, type='mmaction.datasets.CenterCrop'),
            dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
            dict(
                meta_keys=(
                    'filename',
                    'img_shape',
                    'ori_shape',
                    'start_index',
                    'modality',
                    'num_clips',
                    'clip_len',
                ),
                type='mmaction.datasets.VideoQualityPack'),
        ],
        test_mode=True,
        type='mmaction.datasets.VideoQualityDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
test_evaluator = dict(type='VQAMetric')
test_pipeline = [
    dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='mmaction.datasets.SampleFrames'),
    dict(type='mmaction.datasets.DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='mmaction.datasets.Resize'),
    dict(crop_size=224, type='mmaction.datasets.CenterCrop'),
    dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
    dict(
        meta_keys=(
            'filename',
            'img_shape',
            'ori_shape',
            'start_index',
            'modality',
            'num_clips',
            'clip_len',
        ),
        type='mmaction.datasets.VideoQualityPack'),
]
train_cfg = dict(
    max_epochs=3,
    type='mmengine.runner.EpochBasedTrainLoop',
    val_begin=1,
    val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryMeta\\demo_metadata.csv',
        data_prefix=dict(
            video=
            'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryVideos'
        ),
        pipeline=[
            dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                type='mmaction.datasets.SampleFrames'),
            dict(type='mmaction.datasets.DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='mmaction.datasets.Resize'),
            dict(crop_size=224, type='mmaction.datasets.CenterCrop'),
            dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
            dict(
                meta_keys=(
                    'filename',
                    'img_shape',
                    'ori_shape',
                    'start_index',
                    'modality',
                    'num_clips',
                    'clip_len',
                ),
                type='mmaction.datasets.VideoQualityPack'),
        ],
        type='mmaction.datasets.VideoQualityDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        type='mmaction.datasets.SampleFrames'),
    dict(type='mmaction.datasets.DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='mmaction.datasets.Resize'),
    dict(crop_size=224, type='mmaction.datasets.CenterCrop'),
    dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
    dict(
        meta_keys=(
            'filename',
            'img_shape',
            'ori_shape',
            'start_index',
            'modality',
            'num_clips',
            'clip_len',
        ),
        type='mmaction.datasets.VideoQualityPack'),
]
val_cfg = dict(type='mmengine.runner.ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryMeta\\demo_metadata.csv',
        data_prefix=dict(
            video=
            'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\VQA_Demo\\DryVideos'
        ),
        pipeline=[
            dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='mmaction.datasets.SampleFrames'),
            dict(type='mmaction.datasets.DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='mmaction.datasets.Resize'),
            dict(crop_size=224, type='mmaction.datasets.CenterCrop'),
            dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
            dict(
                meta_keys=(
                    'filename',
                    'img_shape',
                    'ori_shape',
                    'start_index',
                    'modality',
                    'num_clips',
                    'clip_len',
                ),
                type='mmaction.datasets.VideoQualityPack'),
        ],
        test_mode=True,
        type='mmaction.datasets.VideoQualityDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
val_evaluator = dict(type='VQAMetric')
val_pipeline = [
    dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='mmaction.datasets.SampleFrames'),
    dict(type='mmaction.datasets.DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='mmaction.datasets.Resize'),
    dict(crop_size=224, type='mmaction.datasets.CenterCrop'),
    dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
    dict(
        meta_keys=(
            'filename',
            'img_shape',
            'ori_shape',
            'start_index',
            'modality',
            'num_clips',
            'clip_len',
        ),
        type='mmaction.datasets.VideoQualityPack'),
]
vis_backends = [
    dict(type='mmaction.visualization.LocalVisBackend'),
]
visualizer = dict(
    type='mmaction.visualization.ActionVisualizer',
    vis_backends=[
        dict(type='mmaction.visualization.LocalVisBackend'),
    ])
work_dir = 'D:\\Academics\\SEM-5\\NVIDIA_miniproj\\mmaction2-1\\work_dirs\\vqa_dryrun'
