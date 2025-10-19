# Copyright (c) OpenMMLab. All rights reserved.
"""Video Quality Assessment Dataset."""

import os.path as osp
import pandas as pd
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class VQA_trial(BaseDataset):
    """Video Quality Assessment Dataset."""
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root='',
                 data_prefix=dict(video=''),
                 modality='RGB',
                 **kwargs):
        
        self.modality = modality
        self._pipeline_cfg = pipeline
        
        # Call parent __init__ without pipeline to avoid registry issues
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=kwargs.get('test_mode', False),
            serialize_data=False,
            lazy_init=kwargs.get('lazy_init', False)
        )
        
        # Build pipeline manually using TRANSFORMS registry
        self.pipeline = self._build_pipeline(self._pipeline_cfg)
    
    def _build_pipeline(self, pipeline_cfg):
        """Build pipeline from config using TRANSFORMS registry."""
        transforms = []
        for transform_cfg in pipeline_cfg:
            transform = TRANSFORMS.build(transform_cfg)
            transforms.append(transform)
        return transforms
    
    def load_data_list(self):
        """Load data list from CSV annotation file."""
        df = pd.read_csv(self.ann_file, sep=',')
        
        data_list = []
        
        for idx, row in df.iterrows():
            video_filename = row['filename']
            video_path = osp.join(self.data_prefix['video'], video_filename)
            
            data_info = dict(
                filename=video_path,
                mos=float(row['mos']),
                quality_class=str(row['quality_class']),
                distortion_type=str(row['distortion_type']),
                prompt=str(row['prompt']),
                qa_question=str(row['qa_question']),
                qa_answer=str(row['qa_answer']),
                psnr=str(row['psnr']),
                ssim=float(row['ssim']),
                blur=float(row['blur']),
                lpips=float(row['lpips']),
                temporal_consistency=float(row['temporal_consistency']),
                hallucination_flag=(row['hallucination_flag'] == 'TRUE'),
                rendering_artifact_flag=(row['rendering_artifact_flag'] == 'TRUE'),
                unnatural_motion_flag=(row['unnatural_motion_flag'] == 'TRUE'),
                lighting_inconsistency_flag=(row['lighting_inconsistency_flag'] == 'TRUE'),
                split=str(row['split'])
            )
            
            data_list.append(data_info)
        
        return data_list
    
    def get_data_info(self, idx):
        """Get data info by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        return data_info
    
    def prepare_data(self, idx):
        """Prepare data by running through pipeline."""
        data_info = self.get_data_info(idx)
        
        # Run through pipeline transforms
        for transform in self.pipeline:
            data_info = transform(data_info)
            if data_info is None:
                return None
        
        return data_info
    
    def __getitem__(self, idx):
        """Get item by index."""
        return self.prepare_data(idx)
    
    def __repr__(self):
        """Print dataset info."""
        repr_str = (
            f'{self.__class__.__name__}(\n'
            f'    ann_file={self.ann_file},\n'
            f'    data_root={self.data_root},\n'
            f'    data_prefix={self.data_prefix},\n'
            f'    num_samples={len(self)},\n'
            f'    modality={self.modality})'
        )
        return repr_str
