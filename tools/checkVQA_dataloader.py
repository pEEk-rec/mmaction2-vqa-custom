from mmengine import Config
from mmaction.registry import DATASETS
from mmengine.dataset import DefaultSampler
from mmengine.runner import build_dataloader  # <- use runner helper

# Ensure your dataset class is registered (import its module
from mmaction.datasets.VQA_dataset import VideoQualityDataset  # adjust to your actual path

cfg = Config.fromfile(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\mmaction\configs\recognition\swin\VQA_swinConfigPipeline.py')

train_ds = DATASETS.build(cfg.VideoQualityDataset['dataset'])
print('len(train_ds)=', len(train_ds))

sampler = DefaultSampler(dataset=train_ds, shuffle=False)

train_dl = build_dataloader(
    dataset=train_ds,
    sampler=sampler,
    batch_size=cfg.VideoQualityDataset['batch_size'],
    num_workers=0,
    persistent_workers=False
)

batch = next(iter(train_dl))
print('inputs:', batch['inputs'].shape)
print('meta keys:', batch['data_samples'][0].metainfo.keys())
