# ================================================================
# AI-Generated Video Quality Assessment - Stage 0 (Frozen Encoder)
# ================================================================

model = dict(
    type='swin_AIRecognizer',

    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained='https://download.openmmlab.com/mmaction/recognition/swin/swin_tiny_patch244_window877_k400_1x1x8_100e_kinetics400_rgb/swin_tiny_patch244_window877_k400_1x1x8_100e_kinetics400_rgb_20220406-0952519e.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True
    ),

    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW',
    ),

    cls_head=dict(
        type='swin_AIHead',
        in_channels=768,
        num_classes=5,
        dropout_ratio=0.25,
        average_clips='prob',

        # Enable uncertainty weighting
        use_uncertainty_weighting=True,

        # Provide pos_weight for artifact heads
        loss_cls=dict(type='CrossEntropyLoss'),
        loss_mos=dict(type='MSELoss'),
        loss_hallucination=dict(type='BCEWithLogitsLoss', pos_weight=2.07),
        loss_lighting=dict(type='BCEWithLogitsLoss', pos_weight=3.26),
        loss_spatial=dict(type='BCEWithLogitsLoss', pos_weight=9.36),
    ),
)
