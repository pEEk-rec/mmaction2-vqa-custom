model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        pretrained='checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'  # Download this
    ),
    cls_head=dict(
        type='MOSRegressionHead',
        in_channels=768,
        num_classes=5,
        dropout_ratio=0.5,
        mos_range=(1.0, 5.0)
 # Adjust based on your MOS range
    ),
    train_cfg=None,
    test_cfg=dict(average_clips='prob')
)