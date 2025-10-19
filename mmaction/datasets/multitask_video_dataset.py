# mmaction/datasets/multitask_video_dataset.py

import os
import torch
from mmengine.fileio import exists
from mmaction.registry import DATASETS
from mmaction.datasets import VideoDataset
from mmengine.fileio import list_from_file

@DATASETS.register_module()
class MultitaskVideoDataset(VideoDataset):
    """Video dataset for multitask: action, MOS regression, quality classification.
    
    Annotation file format (one line per sample):
        <video_path> <action_label> <mos_score> <quality_label>
    where:
        - video_path: relative path under data_prefix
        - action_label: integer action class
        - mos_score: float quality score (e.g., [0,1] normalized)
        - quality_label: integer quality class (or comma-separated for multilabel)
    """
    
    def load_annotations(self) -> list:
        """Load annotation file and parse action, mos, and quality labels."""
        lines = list_from_file(self.ann_file)
        data_infos = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(
                    f'Annotation line must have 4 fields, got {len(parts)}: {line}'
                )
            rel_path, action_str, mos_str, quality_str = parts
            video_path = os.path.join(self.data_prefix['video'], rel_path)
            if not exists(video_path):
                raise FileNotFoundError(f'Video file not found: {video_path}')
            
            action_label = int(action_str)
            mos_label = float(mos_str)
            # For single-label classification
            quality_label = int(quality_str)
            # For multi-label, you could parse comma separated:
            # quality_label = [int(x) for x in quality_str.split(',')]
            
            info = dict(
                video_path=video_path,
                total_frames=None,     # can be inferred later
                start_index=0,
                modalities=self.modalities,
                label=action_label,    # used by base VideoDataset as gt_label
                mos_label=mos_label,
                quality_label=quality_label
            )
            data_infos.append(info)
        
        return data_infos

    def prepare_train_data(self, idx: int) -> dict:
        """Generate training data for a single sample (with augmentations)."""
        results = super().prepare_train_data(idx)
        # Append MOS and quality labels into results
        info = self.data_infos[idx]
        results['mos_label'] = torch.tensor(info['mos_label'], dtype=torch.float32)
        results['quality_label'] = torch.tensor(info['quality_label'], dtype=torch.long)
        return results

    def prepare_test_data(self, idx: int) -> dict:
        """Generate test/val data for a single sample."""
        results = super().prepare_test_data(idx)
        info = self.data_infos[idx]
        results['mos_label'] = torch.tensor(info['mos_label'], dtype=torch.float32)
        results['quality_label'] = torch.tensor(info['quality_label'], dtype=torch.long)
        return results
