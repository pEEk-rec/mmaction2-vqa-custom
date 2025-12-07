import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

scalars_files = [
    r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\from workstation\swin_AIVQA_ddp_fixed\20251205_173544\vis_data\scalars.json"
]

OUTPUT_PATH = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output_ppt\training_metrics_auto.png"
SKIP_EPOCHS = 3  # skip first 3 epochs (configs warm up)


# ============================================================
# SMART AUTO-DETECTOR FOR METRIC KEYS
# ============================================================

def detect_metric_keys(data):
    """Automatically detect metric keys from a scalars.json line."""
    keys = data.keys()

    srocc = next((k for k in keys if "srcc" in k.lower() or "srocc" in k.lower()), None)
    plcc = next((k for k in keys if "plcc" in k.lower()), None)
    rmse = next((k for k in keys if "rmse" in k.lower()), None)

    return srocc, plcc, rmse


# ============================================================
# LOAD AND PARSE LOG FILES
# ============================================================

print("\n🔍 Loading scalars.json...")

train_losses = []
train_epochs = []

val_srocc = []
val_plcc = []
val_rmse = []
val_epochs = []

detected_srocc_key = None
detected_plcc_key = None
detected_rmse_key = None

for fpath in scalars_files:
    if not os.path.exists(fpath):
        print(f"❌ File not found: {fpath}")
        continue

    print(f"Reading: {fpath}")

    with open(fpath, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                # ==========================
                # 1. Detect metric names dynamically
                # ==========================
                if detected_srocc_key is None:
                    srocc_k, plcc_k, rmse_k = detect_metric_keys(data)
                    if srocc_k:
                        detected_srocc_key = srocc_k
                        detected_plcc_key = plcc_k
                        detected_rmse_key = rmse_k
                        print(f"\n✅ Detected metric keys:")
                        print(f"  SRCC → {detected_srocc_key}")
                        print(f"  PLCC → {detected_plcc_key}")
                        print(f"  RMSE → {detected_rmse_key}")

                # ==========================
                # 2. Collect training loss
                # ==========================
                if "loss" in data:
                    train_losses.append(data["loss"])
                    train_epochs.append(int(data.get("epoch", 0)))


                # ==========================
                # 3. Collect validation metrics
                # ==========================
                if detected_srocc_key and detected_srocc_key in data:

                    val_srocc.append(data.get(detected_srocc_key, 0))
                    val_plcc.append(data.get(detected_plcc_key, 0) if detected_plcc_key else 0)
                    val_rmse.append(data.get(detected_rmse_key, 0) if detected_rmse_key else 0)
                    val_epochs.append(int(data.get("epoch", 0)))

            except json.JSONDecodeError:
                pass


print("\n📌 SUMMARY OF LOADED DATA")
print(f"Train loss points: {len(train_losses)}")
print(f"Validation epochs: {len(val_epochs)}")


# Safe guard against missing validation logs
if len(val_epochs) == 0:
    print("\n❌ ERROR: No validation metrics found in scalars.json!")
    print("Possible reasons:")
    print(" - Wrong metric key naming")
    print(" - Validation hook disabled")
    print(" - Using the wrong scalars.json file")
    exit(1)


# ============================================================
# REMOVE FIRST 3 EPOCHS
# ============================================================

filtered_srocc = val_srocc[SKIP_EPOCHS:]
filtered_plcc = val_plcc[SKIP_EPOCHS:]
filtered_rmse = val_rmse[SKIP_EPOCHS:]
filtered_epochs = list(range(SKIP_EPOCHS + 1, SKIP_EPOCHS + 1 + len(filtered_srocc)))

print(f"\nUsing epochs: {filtered_epochs[0]} → {filtered_epochs[-1]}")


# ============================================================
# TRAINING LOSS SAMPLED EVERY 5 EPOCHS
# ============================================================

loss_epochs = []
loss_values = []

for i, (loss, e) in enumerate(zip(train_losses, train_epochs)):
    if e >= SKIP_EPOCHS and e % 3 == 0:
        if i == len(train_losses) - 1 or train_epochs[i + 1] != e:
            loss_epochs.append(e)
            loss_values.append(loss)

print(f"Training loss sample points: {loss_epochs}")


# ============================================================
# PLOTTING SECTION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("white")


# 1) Training Loss
axes[0, 0].plot(loss_epochs, loss_values, marker="o", color="#1f77b4", linewidth=2)
axes[0, 0].set_title("Training Loss (Every 5 Epochs)")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

# 2) SROCC
axes[0, 1].plot(filtered_epochs, filtered_srocc, marker="o", color="#ff7f0e", linewidth=2)
axes[0, 1].set_title("Validation SRCC")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(True, alpha=0.3)

# 3) PLCC
axes[1, 0].plot(filtered_epochs, filtered_plcc, marker="s", color="#2ca02c", linewidth=2)
axes[1, 0].set_title("Validation PLCC")
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(True, alpha=0.3)

# 4) RMSE
axes[1, 1].plot(filtered_epochs, filtered_rmse, marker="^", color="#d62728", linewidth=2)
axes[1, 1].set_title("Validation RMSE")
axes[1, 1].grid(True, alpha=0.3)


plt.suptitle("Training Metrics (Epochs 4–150)", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, facecolor="white")
plt.close()

print(f"\n✅ Saved figure → {OUTPUT_PATH}")


# ============================================================
# PRINT SUMMARY STATS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SUMMARY (AUTO DETECTED)")
print("=" * 60)

print(f"Best SRCC: {max(filtered_srocc):.4f} at Epoch {filtered_epochs[np.argmax(filtered_srocc)]}")
print(f"Best PLCC: {max(filtered_plcc):.4f} at Epoch {filtered_epochs[np.argmax(filtered_plcc)]}")
print(f"Best RMSE: {min(filtered_rmse):.4f} at Epoch {filtered_epochs[np.argmin(filtered_rmse)]}")

print("\nTraining Loss Samples:")
for e, l in zip(loss_epochs, loss_values):
    print(f"  Epoch {e}: {l:.4f}")

print("\nFinal Epoch Metrics:")
print(f"  SRCC: {filtered_srocc[-1]:.4f}")
print(f"  PLCC: {filtered_plcc[-1]:.4f}")
print(f"  RMSE: {filtered_rmse[-1]:.4f}")

print("\n🎉 Visualization Complete!")
