# mmaction2/mmaction/datasets/finevd_dataset.py

from typing import Callable, List, Optional, Union
import os.path as osp
import json
import copy

from mmaction.registry import DATASETS
from .base import BaseActionDataset

def mos_to_quality_label(mos):
    """Convert MOS to quality class."""
    if mos < 2.0:
        return 0  # Bad
    elif mos < 3.0:
        return 1  # Poor
    elif mos < 4.0:
        return 2  # Fair
    elif mos < 4.5:
        return 3  # Good
    else:
        return 4  # Excellent

# mmaction/datasets/finevdMOS_dataset.py

@DATASETS.register_module()
class FineVDDataset(BaseActionDataset):
    
    def __init__(self, ann_file: str, pipeline: List[Union[dict, Callable]],
                 data_prefix: dict = dict(video=''), test_mode: bool = False,
                 modality: str = 'RGB', **kwargs):
        
        self.quality_labels = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']
        super().__init__(ann_file=ann_file, pipeline=pipeline, 
                        data_prefix=data_prefix, test_mode=test_mode,
                        modality=modality, **kwargs)
        
        # Force load if empty
        if len(self.data_list) == 0:
            print(f"WARNING: data_list empty after init, forcing load...")
            self.data_list = self.load_data_list()
            print(f"Loaded {len(self.data_list)} samples")
    
    def load_data_list(self) -> List[dict]:
        if not osp.exists(self.ann_file):
            raise FileNotFoundError(f'{self.ann_file} does not exist')
        
        with open(self.ann_file, 'r') as f:
            annotations = json.load(f)
        
        data_list = []
        for ann in annotations:
            data_info = {
                'filename': ann['video_path'],
                'mos': float(ann['mos']),
                'label': int(ann.get('quality_class', mos_to_quality_label(ann['mos'])))
            }
            data_list.append(data_info)
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def get_data_info(self, idx: int) -> dict:
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} >= {len(self.data_list)}")
        
        data_info = super().get_data_info(idx)
        data_info['mos'] = self.data_list[idx]['mos']
        data_info['gt_label'] = self.data_list[idx]['label']
        
        # FIX: Manually join data_prefix with filename
        if 'filename' in data_info and self.data_prefix:
            video_prefix = self.data_prefix.get('video', '')
            if video_prefix and not osp.isabs(data_info['filename']):
                data_info['filename'] = osp.join(video_prefix, data_info['filename'])
        
        return data_info

def exists(file_path: str, msg: str = None):
    """Check if file exists."""
    if not osp.exists(file_path):
        raise FileNotFoundError(msg or f'{file_path} does not exist')