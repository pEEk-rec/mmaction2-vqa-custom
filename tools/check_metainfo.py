from mmengine import Config
from mmaction.registry import MODELS, DATASETS
import torch

cfg = Config.fromfile("configs/recognition/swin/swin_AI-VQA_Config.py")

# --- Build dataset and model ---
dataset = DATASETS.build(cfg.test_dataloader['dataset'])
model = MODELS.build(cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# --- Get one dataset sample ---
sample = dataset[0]

# Wrap properly for ActionDataPreprocessor
data = dict(
    inputs=[sample['inputs']],       # ✅ list, not tensor
    data_samples=[sample['data_samples']]
)

# Normalize via model’s data_preprocessor (uint8 → float32)
data = model.data_preprocessor(data, training=False)

inputs = data['inputs']
data_samples = data['data_samples']

# --- Run through model head ---
with torch.no_grad():
    feats = model.extract_feat(inputs)
    out_samples = model.cls_head.predict(feats, data_samples)

print("✅ Metainfo keys:", list(out_samples[0].metainfo.keys()))
print("✅ Metainfo content:", out_samples[0].metainfo)
