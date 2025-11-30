import os
import torch
from mmengine import Config
from mmengine.dataset.utils import default_collate
from mmaction.registry import MODELS, DATASETS, METRICS

# ============================================================
# CONFIGURATION
# ============================================================
CFG_PATH = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\configs\recognition\swin\swin_AI-VQA_Config.py"

print("\n🚀 Loading config...")
cfg = Config.fromfile(CFG_PATH)
cfg.work_dir = "./work_dirs/swin_AI-VQA"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Using device: {device.upper()}")

# ============================================================
# BUILD DATASET, MODEL, AND METRIC
# ============================================================
print("⚙️ Building dataset, model, and metric...")

dataset = DATASETS.build(cfg.train_dataloader["dataset"])
model = MODELS.build(cfg.model)
metric = METRICS.build(cfg.val_evaluator)

model = model.to(device)
model.eval()

print(f"✅ Dataset loaded: {len(dataset)} samples")
print(f"✅ Model: {model.__class__.__name__}")
print(f"✅ Metric: {metric.__class__.__name__}")

# ============================================================
# FETCH ONE SAMPLE
# ============================================================
print("\n📦 Fetching one sample from dataset...")

try:
    sample = dataset[0]
except Exception as e:
    raise RuntimeError(f"❌ Failed to fetch sample from dataset: {e}")

if not isinstance(sample, dict):
    raise TypeError(f"Expected dict sample, got {type(sample)}")

print(f"✅ Sample keys: {list(sample.keys())}")
if "inputs" in sample:
    print(f"   - Inputs shape: {tuple(sample['inputs'].shape)}")

# ============================================================
# PREPARE BATCH (SIMULATE DATALOADER)
# ============================================================
print("\n🔄 Preparing mini-batch (using default_collate)...")

try:
    batch = default_collate([sample])
    inputs = batch["inputs"].to(device)
    data_samples = batch["data_samples"]
    print(f"✅ Batch prepared: inputs {tuple(inputs.shape)}, {len(data_samples)} samples")
except Exception as e:
    raise RuntimeError(f"❌ Failed during batch collation: {e}")

# ============================================================
# FORWARD + LOSS TEST
# ============================================================
print("\n🧩 Running forward + loss check...")
with torch.no_grad():
    feats = model.extract_feat(inputs)
    losses = model.cls_head.loss(feats, data_samples)

print("✅ Loss components:")
for k, v in losses.items():
    if torch.is_tensor(v):
        v = v.detach().cpu().item()
    print(f"   {k}: {v:.6f}" if isinstance(v, (float, int)) else f"   {k}: {v}")

# ============================================================
# PREDICTION TEST
# ============================================================
print("\n🔍 Running predict() on the same batch...")
with torch.no_grad():
    preds = model.cls_head.predict(feats, data_samples)

first_meta = getattr(preds[0], "metainfo", {})
print("✅ Prediction metainfo keys:", list(first_meta.keys())[:10])

def show_field(name):
    val = first_meta.get(name, "N/A")
    print(f"   {name}: {val}")

print("   --- Predicted fields ---")
for field in [
    "pred_mos", "pred_quality_class", "pred_hallucination",
    "pred_lighting", "pred_spatial", "pred_rendering"
]:
    show_field(field)

# ============================================================
# METRIC COMPUTATION TEST
# ============================================================
print("\n📊 Running metric computation (mock evaluation)...")
metric.process(None, preds)
results = metric.compute_metrics(metric.results)

print("\n✅ Metric Output Summary:")
for k, v in results.items():
    if isinstance(v, float):
        print(f"   {k}: {v:.4f}")
    else:
        print(f"   {k}: {v}")

# ============================================================
# SUMMARY
# ============================================================
print("\n🎯 One-batch sanity test completed successfully!")
