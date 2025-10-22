import os, sys, numpy as np, torch
repo_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1'
sys.path.insert(0, repo_root)

import mmaction.models.recognizers.video_quality_recognizer
import mmaction.models.data_preprocessors
import mmaction.models.heads.VQA_multihead
import mmaction.datasets.VQA_dataset
import mmaction.datasets.transforms.video_quality_pack
import mmaction.evaluation.metrics.VQA_customMetric

from mmengine import Config
from mmaction.registry import MODELS, DATASETS
from mmengine.dataset import DefaultSampler
from mmengine.runner import Runner


def global_avg_pool_3d(x):
    return x.mean(dim=[2,3,4])  # [B,C,T,H,W] -> [B,C]

def main():
    cfg = Config.fromfile(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\mmaction\configs\recognition\swin\VQA_swinConfigPipeline.py')
    ckpt = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\VQA_dryrun\epoch_3.pth'  # final training checkpoint

    model = MODELS.build(cfg.model)
    Runner.load_checkpoint(model, ckpt, map_location='cpu')
    model.eval().cuda()

    # Use val or test loader defined in cfg for deterministic extraction
    ds_cfg = cfg.val_dataloader['dataset']  # or test_dataloader['dataset']
    dataset = DATASETS.build(ds_cfg)
    sampler = DefaultSampler(dataset, shuffle=False)
    loader = Runner.build_dataloader(dataset=dataset, sampler=sampler,
                              batch_size=1, num_workers=2,
                              persistent_workers=False)

    all_feats, rows = [], []
    with torch.no_grad():
        for batch in loader:
            # Let data_preprocessor handle normalization/types
            batch = model.data_preprocessor(batch, training=False)
            inputs = batch['inputs'].cuda()           # [B,C,T,H,W]
            data_samples = batch['data_samples']

            # Forward backbone only; skip head
            feats = model.backbone(inputs)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            if feats.dim() == 5:
                pooled = global_avg_pool_3d(feats)     # [B,C]
            else:
                pooled = feats.mean(dim=1)             # adapt if flattened

            all_feats.append(pooled.cpu().numpy())
            ds = data_samples[0]
            rows.append(dict(
                filename=os.path.basename(ds.metainfo.get('filename', 'unk')),
                mos=ds.metainfo.get('mos', None),
                quality_class=ds.metainfo.get('quality_class', None),
                distortion_type=ds.metainfo.get('distortion_type', None),
            ))

    feats = np.concatenate(all_feats, axis=0)  # [N,C]
    # Save as .npy + .csv for speed and fidelity
    os.makedirs(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\vqa_dryrun', exist_ok=True)
    np.save(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\vqa_dryrun\embeds_final.npy', feats)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(r'work_dirs/vqa_dryrun/embeds_index.csv', index=False)
    print('Saved', feats.shape, 'to embeds_final.npy and index to embeds_index.csv')

if __name__ == '__main__':
    main()
