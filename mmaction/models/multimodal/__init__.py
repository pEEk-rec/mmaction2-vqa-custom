# Disable multimodal imports entirely (Windows fix)
from mmpretrain.utils.dependency import register_multimodal_placeholder
register_multimodal_placeholder()
