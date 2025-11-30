# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Callable, List, Union

import pandas as pd
import torch
from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class swin_MOSData(BaseActionDataset):
    """Video Quality Assessment dataset for video quality prediction.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and quality annotations.

    The ann_file is a CSV file with the following columns:
    - video_name: filename of the video
    - video_score: MOS (Mean Opinion Score) ranging from 1.0 to 5.0
    - quality_class: Discretized quality class (0-4)

    Example of annotation CSV:

    .. code-block:: txt

        video_name,video_score,quality_class
        orig_10000251326_540_5s.mp4,3.4,2
        orig_10000958013_540_5s.mp4,2.8,2
        orig_10001646563_540_5s.mp4,3.6,3

    Args:
        ann_file (str): Path to the CSV annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 **kwargs):
        self.multi_class = False
        self.num_classes = None
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=False,
            num_classes=None,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load CSV annotation file to get video quality information."""
        exists(self.ann_file)
        
        # Read CSV file
        df = pd.read_csv(self.ann_file)
        
        # Verify required columns exist
        required_cols = ['video_name', 'video_score', 'quality_class']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        data_list = []
        
        for idx, row in df.iterrows():
            video_name = row['video_name']
            mos = float(row['video_score'])
            quality_class = int(row['quality_class'])
            
            # Construct full video path
            if self.data_prefix.get('video') is not None:
                filename = osp.join(self.data_prefix['video'], video_name)
            else:
                filename = video_name
            
            # Create data dict with all necessary fields
            data_info = dict(
                filename=filename,
                label=quality_class,
                mos=mos,
                gt_mos=mos,
                video_score=mos,
                quality_class=quality_class,
                video_name=video_name
            )
            
            data_list.append(data_info)
        
        return data_list
    
    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.
        
        This override ensures MOS field is preserved through the pipeline.
        """
        # Get data info
        data_info = self.get_data_info(idx)
        
        # Store MOS value before pipeline processing
        mos_value = data_info.get('mos') or data_info.get('gt_mos') or data_info.get('video_score')
        
        # Run pipeline
        result = self.pipeline(data_info)
        
        # CRITICAL: Add MOS to data_sample after pipeline processing
        if result is not None and 'data_samples' in result:
            # Ensure gt_mos is added to the ActionDataSample
            result['data_samples'].gt_mos = torch.tensor(float(mos_value), dtype=torch.float32)
        
        return result
