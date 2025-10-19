import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample  # preferred

class QualityDataSample(ActionDataSample):
    pass

@TRANSFORMS.register_module()
class VideoQualityPack(BaseTransform):
    """Pack video frames and quality annotations into data sample.

    Expects:
      - imgs: list[np.ndarray HxWxC], img_shape, num_clips, clip_len
      - labels from dataset: mos (float), quality_class (int), distortion_type (int)
    Produces:
      - inputs: torch.Tensor [C, T, H, W]
      - data_samples: QualityDataSample with metainfo holding GT
    """

    def __init__(self, meta_keys=('img_shape', 'ori_shape', 'num_clips', 'clip_len', 'filename', 'start_index', 'modality')):
        self.meta_keys = meta_keys

    def transform(self, results):
        # Frames to tensor [C, T, H, W]
        imgs = results['imgs']

        # Case A: list of frames -> [T,H,W,C] -> [C,T,H,W]
        if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], np.ndarray):
            arr = np.stack(imgs, axis=0)  # [T,H,W,C]
            vid = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().float()

        else:
            vid = to_tensor(imgs)
            # Handle shapes:
            # [T,H,W,C] -> [C,T,H,W]
            if vid.dim() == 4 and vid.shape[-1] in (1, 3):
                vid = vid.permute(3, 0, 1, 2).contiguous()
            # [1,C,T,H,W] -> [C,T,H,W]
            elif vid.dim() == 5 and vid.shape[0] == 1:
                vid = vid.squeeze(0).contiguous()
            # [C,T,H,W] -> ok
            elif vid.dim() == 4 and vid.shape[0] in (1, 3):
                pass
            else:
                raise ValueError(f'Unexpected imgs format: type={type(imgs)} shape={tuple(vid.shape)}')


        ds = QualityDataSample()

        # Build metainfo and ensure numeric dtypes
        meta = {k: results[k] for k in self.meta_keys if k in results}

        mos = results.get('mos', None)
        if mos is not None:
            meta['mos'] = float(mos)

        qc = results.get('quality_class', None)
        if qc is not None:
            meta['quality_class'] = int(qc)

        dt = results.get('distortion_type', None)
        if dt is not None:
            meta['distortion_type'] = int(dt)

        # Optional text/aux fields (kept as metainfo as well)
        for k in ('prompt', 'qa_question', 'qa_answer'):
            if k in results:
                meta[k] = results[k]
        for k in ('psnr', 'ssim', 'lpips', 'temporal_consistency'):
            if k in results:
                v = results[k]
                if isinstance(v, str) and v.lower() == 'inf':
                    v = float('inf')
                meta[k] = float(v)

        for k in ('hallucination_flag', 'rendering_artifact_flag', 'unnatural_motion_flag', 'lighting_inconsistency_flag'):
            if k in results:
                meta[k] = bool(results[k])

        ds.set_metainfo(meta)

        # Return packed dict
        return dict(inputs=vid, data_samples=ds)

    def __repr__(self):
        return f'{self.__class__.__name__}(meta_keys={self.meta_keys})'
