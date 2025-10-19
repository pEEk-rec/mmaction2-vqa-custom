model = dict(
    type='MultitaskRecognizer',  # Change from 'Recognizer3D'
    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained=None,        # No pretrained weights
        pretrained2d=False,     # Don't try to inflate 2D weights when pretrained=None
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'
    ),
    cls_head=dict(
        type='MultitaskHead',    # Change from 'I3DRegressionHead'
        num_classes_action=400,  # Number of action classes (Kinetics-400)
        num_classes_quality=5,   # Number of quality issue classes (adjust as needed)
        in_channels=768,         # Swin-T tiny output channels
        dropout_ratio=0.5
        # Remove the old loss_cls - now handled internally in MultitaskHead
    )
)
