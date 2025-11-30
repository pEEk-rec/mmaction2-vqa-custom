from mmengine import Config
from mmaction.registry import MODELS, DATASETS
import torch

cfg = Config.fromfile("configs/recognition/swin/swin_AI-VQA_Config.py")

# build dataset and model (cpu or cuda)
ds_cfg = cfg.test_dataloader['dataset']
dataset = DATASETS.build(ds_cfg)

model = MODELS.build(cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# get one sample and prepare a batch
sample = dataset[0]
# many dataset items store inputs as torch.Tensor already shaped [1,M,C,T,H,W] or [M,C,T,H,W]
inputs = sample['inputs'].unsqueeze(0).to(device)
data_samples = [sample['data_samples']]

with torch.no_grad():
    feats = model.extract_feat(inputs)
    out_samples = model.cls_head.predict(feats, data_samples)

print("Metainfo keys:", list(out_samples[0].metainfo.keys()))
print("Metainfo content:", out_samples[0].metainfo)
