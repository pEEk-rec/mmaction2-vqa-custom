# Copyright (c) OpenMMLab.
import os, os.path as osp
import pandas as pd
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS, TRANSFORMS

# Edit to match your CSV exactly
QUALITY_MAP = {
    'bad': 0, 'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4
}

# Updated to include your CSV values
DISTORTION_MAP = {
    'original': 0,
    'blur': 1,
    'noise': 2,
    'compression': 3,
    'frame_drop': 4,
    'stalling': 5,
    'temporal_subsample': 6,
}

def _to_bool(x):
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ('1', 'true', 'yes', 'y', 't')

@DATASETS.register_module()
class VideoQualityDataset(BaseDataset):
    """Video Quality Assessment Dataset that reads a CSV and emits video path plus quality labels."""

    METAINFO = dict(classes=None)  # not used, multi-task via metainfo

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root='',
                 data_prefix=dict(video=''),
                 modality='RGB',
                 split_filter=None,   # 'train'/'val'/'test' or None
                 mos_normalize=True,  # normalize MOS from [0,100] -> [0,1]
                 check_files=False,   # set True to assert files exist
                 **kwargs):

        self.modality = modality
        self._pipeline_cfg = pipeline
        self.split_filter = split_filter
        self.mos_normalize = mos_normalize
        self.check_files = check_files

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=kwargs.get('test_mode', False),
            serialize_data=False,
            lazy_init=kwargs.get('lazy_init', False)
        )

        self.pipeline = [TRANSFORMS.build(tcfg) for tcfg in self._pipeline_cfg]

    def _parse_quality_class(self, v):
        # Accept int or string labels like 'Bad'
        if pd.isna(v):
            raise ValueError('quality_class is NaN')
        if isinstance(v, (int, float)) and str(v).isdigit():
            return int(v)
        s = str(v).strip().lower()
        if s in QUALITY_MAP:
            return QUALITY_MAP[s]
        if s.isdigit():
            return int(s)
        raise ValueError(f'Unknown quality_class label: {v}')

    def _parse_distortion_type(self, v):
        if pd.isna(v):
            raise ValueError('distortion_type is NaN')
        if isinstance(v, (int, float)) and str(v).isdigit():
            return int(v)
        s = str(v).strip().lower()
        if s in DISTORTION_MAP:
            return DISTORTION_MAP[s]
        if s.isdigit():
            return int(s)
        raise ValueError(f'Unknown distortion_type label: {v}')

    def load_data_list(self):
        df = pd.read_csv(self.ann_file)

        # Optional split filtering
        if self.split_filter is not None and 'split' in df.columns:
            df = df[df['split'].astype(str).str.lower()
                    == str(self.split_filter).lower()].reset_index(drop=True)

        # Validate required columns
        required = ['filename', 'mos', 'quality_class', 'distortion_type']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Missing columns in CSV: {missing}')

        data_list = []
        bad_paths = []

        for _, row in df.iterrows():
            rel = str(row['filename'])
            video_path = osp.join(self.data_prefix.get('video', ''), rel)

            if self.check_files and not osp.isfile(video_path):
                bad_paths.append(video_path)
                continue

            # MOS
            mos_raw = float(row['mos'])
            mos = mos_raw / 100.0 if self.mos_normalize else mos_raw

            # Labels
            qclass = self._parse_quality_class(row['quality_class'])
            dtype = self._parse_distortion_type(row['distortion_type'])

            # Optional fields (safe parsing)
            ssim = float(row['ssim']) if 'ssim' in row and pd.notna(row['ssim']) else None
            blur = float(row['blur']) if 'blur' in row and pd.notna(row['blur']) else None
            lpips = float(row['lpips']) if 'lpips' in row and pd.notna(row['lpips']) else None
            psnr = float(row['psnr']) if 'psnr' in row and pd.notna(row['psnr']) else None

            # Flags
            hallu = _to_bool(row['hallucination_flag']) if 'hallucination_flag' in row else False
            rend = _to_bool(row['rendering_artifact_flag']) if 'rendering_artifact_flag' in row else False
            unmot = _to_bool(row['unnatural_motion_flag']) if 'unnatural_motion_flag' in row else False
            light = _to_bool(row['lighting_inconsistency_flag']) if 'lighting_inconsistency_flag' in row else False

            # Optional text fields
            prompt = str(row['prompt']) if 'prompt' in row and pd.notna(row['prompt']) else ''
            qa_q = str(row['qa_question']) if 'qa_question' in row and pd.notna(row['qa_question']) else ''
            qa_a = str(row['qa_answer']) if 'qa_answer' in row and pd.notna(row['qa_answer']) else ''

            data_info = dict(
                filename=video_path,
                modality=self.modality,
                # core labels for head
                mos=mos,
                quality_class=qclass,
                distortion_type=dtype,
                # extras
                prompt=prompt,
                qa_question=qa_q,
                qa_answer=qa_a,
                psnr=psnr,
                ssim=ssim,
                blur=blur,
                lpips=lpips,
                temporal_consistency=float(row['temporal_consistency']) if 'temporal_consistency' in row and pd.notna(row['temporal_consistency']) else None,
                hallucination_flag=hallu,
                rendering_artifact_flag=rend,
                unnatural_motion_flag=unmot,
                lighting_inconsistency_flag=light,
                split=str(row['split']) if 'split' in row and pd.notna(row['split']) else None
            )
            data_list.append(data_info)

        if self.check_files and bad_paths:
            raise FileNotFoundError(f'{len(bad_paths)} missing video files. First few: {bad_paths[:5]}')

        return data_list

    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        if 'start_index' not in data_info:
            data_info['start_index'] = 0
        metainfo = dict(data_info.get('metainfo', {}))
        for k in ('mos', 'quality_class', 'distortion_type'):
            if k in data_info:
                metainfo[k] = data_info[k]
        metainfo['filename'] = data_info.get('filename', None)
        data_info['metainfo'] = metainfo
        return data_info




    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        for transform in self.pipeline:
            data_info = transform(data_info)
            if data_info is None:
                return None
        return data_info

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'ann_file={self.ann_file}, '
                f'data_root={self.data_root}, '
                f'data_prefix={self.data_prefix}, '
                f'num_samples={len(self)}, '
                f'modality={self.modality}, '
                f'split_filter={self.split_filter}, '
                f'mos_normalize={self.mos_normalize}, '
                f'check_files={self.check_files})')
