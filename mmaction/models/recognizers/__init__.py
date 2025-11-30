# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizer3d_mm import MMRecognizer3D
from .recognizer_audio import RecognizerAudio
from .recognizer_gcn import RecognizerGCN
from .recognizer_omni import RecognizerOmni
from .Swin_multitask_recognizer import MultitaskRecognizer
from .video_quality_recognizer import VideoQualityRecognizer
from .swin_MOSRecognizer import swin_MOSRecognizer
from .swin_AIRecognizer import swin_AIRecognizer
__all__ = [
    'BaseRecognizer', 'RecognizerGCN', 'Recognizer2D', 'Recognizer3D',
    'RecognizerAudio', 'RecognizerOmni', 'MMRecognizer3D', 'MultitaskRecognizer', 'VideoQualityRecognizer',
    'swin_MOSRecognizer', 'swin_AIRecognizer'
]
