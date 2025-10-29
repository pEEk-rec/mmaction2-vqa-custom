ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data = dict(samples_per_gpu=1)
data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
dataset_type = 'mmaction.datasets.VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3,
        max_keep_ckpts=5,
        save_best='auto',
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        ignore_last=False, interval=100, type='mmengine.hooks.LoggerHook'),
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
        arch='base',
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        drop_rate=0.0,
        mlp_ratio=4.0,
        patch_norm=True,
        patch_size=(
            2,
            4,
            4,
        ),
        pretrained=
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_224.pth',
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
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=1024,
        num_classes=400,
        spatial_type='avg',
        type='mmaction.models.I3DHead'),
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
        type='mmaction.models.ActionDataPreprocessor'),
    type='mmaction.models.Recognizer3D')
optim_wrapper = dict(
    constructor='mmaction.engine.SwinOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.001,
        type='torch.optim.AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='mmengine.optim.AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2.5,
        start_factor=0.1,
        type='mmengine.optim.LinearLR'),
    dict(
        T_max=30,
        begin=0,
        by_epoch=True,
        end=30,
        eta_min=0,
        type='mmengine.optim.CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='mmengine.runner.TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/kinetics400/kinetics400_val_list_videos.txt',
        data_prefix=dict(video='data/kinetics400/videos_val'),
        pipeline=[
            dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=4,
                test_mode=True,
                type='mmaction.datasets.SampleFrames'),
            dict(type='mmaction.datasets.DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='mmaction.datasets.Resize'),
            dict(crop_size=224, type='mmaction.datasets.ThreeCrop'),
            dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
            dict(type='mmaction.datasets.PackActionInputs'),
        ],
        test_mode=True,
        type='mmaction.datasets.VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
test_evaluator = dict(type='mmaction.evaluation.AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='mmaction.datasets.DecordInit'),
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True,
        type='mmaction.datasets.SampleFrames'),
    dict(type='mmaction.datasets.DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='mmaction.datasets.Resize'),
    dict(crop_size=224, type='mmaction.datasets.ThreeCrop'),
    dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
    dict(type='mmaction.datasets.PackActionInputs'),
]
total_epochs = 1
train_cfg = dict(
    max_epochs=30,
    type='mmengine.runner.EpochBasedTrainLoop',
    val_begin=1,
    val_interval=3)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/kinetics400/kinetics400_train_list_videos.txt',
        data_prefix=dict(video='data/kinetics400/videos_train'),
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
            dict(type='mmaction.datasets.RandomResizedCrop'),
            dict(
                keep_ratio=False,
                scale=(
                    224,
                    224,
                ),
                type='mmaction.datasets.Resize'),
            dict(flip_ratio=0.5, type='mmaction.datasets.Flip'),
            dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
            dict(type='mmaction.datasets.PackActionInputs'),
        ],
        type='mmaction.datasets.VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
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
    dict(type='mmaction.datasets.RandomResizedCrop'),
    dict(
        keep_ratio=False, scale=(
            224,
            224,
        ), type='mmaction.datasets.Resize'),
    dict(flip_ratio=0.5, type='mmaction.datasets.Flip'),
    dict(input_format='NCTHW', type='mmaction.datasets.FormatShape'),
    dict(type='mmaction.datasets.PackActionInputs'),
]
val_cfg = dict(type='mmengine.runner.ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/kinetics400/kinetics400_val_list_videos.txt',
        data_prefix=dict(video='data/kinetics400/videos_val'),
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
            dict(type='mmaction.datasets.PackActionInputs'),
        ],
        test_mode=True,
        type='mmaction.datasets.VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
val_evaluator = dict(type='mmaction.evaluation.AccMetric')
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
    dict(type='mmaction.datasets.PackActionInputs'),
]
vis_backends = [
    dict(type='mmaction.visualization.LocalVisBackend'),
]
visualizer = dict(
    type='mmaction.visualization.ActionVisualizer',
    vis_backends=[
        dict(type='mmaction.visualization.LocalVisBackend'),
    ])
work_dir = 'work_dirs/swin_test'
