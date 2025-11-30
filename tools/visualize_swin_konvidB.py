# visualize_training_clean_3to50_FIXED.py
import json
import matplotlib.pyplot as plt
import numpy as np


# ============================================
# STEP 1: Load ALL scalars.json from 3 runs
# ============================================


scalars_files = [
    r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\swin_Ai-VQA_workstation\20251113_171757\vis_data\scalars.json',
]


# Storage for combined data
all_train_losses = []
all_train_epochs = []  # Track which epoch each loss belongs to
all_val_srocc = []
all_val_plcc = []
all_val_rmse = []
all_val_epochs = []


print("Loading scalars.json files...")


for file_idx, log_file in enumerate(scalars_files):
    print(f"  Reading file {file_idx+1}...")
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Training loss (ALL iterations)
                    if 'loss' in data and 'swin_MOS/SROCC' not in data:
                        all_train_losses.append(data['loss'])
                        epoch_val = data.get('epoch', 0)
                        all_train_epochs.append(int(epoch_val))
                    
                    # Validation metrics (ONLY logged at epoch END)
                    if 'swin_MOS/SROCC' in data:
                        all_val_srocc.append(data['swin_MOS/SROCC'])
                        all_val_plcc.append(data['swin_MOS/PLCC'])
                        all_val_rmse.append(data['swin_MOS/RMSE'])
                        epoch_val = data.get('epoch', 0)
                        all_val_epochs.append(int(epoch_val))
                
                except (json.JSONDecodeError, KeyError):
                    pass
    
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {log_file}")


print(f"\n✅ Successfully loaded!")
print(f"   Total validation epochs: {len(all_val_epochs)}")
print(f"   Epoch range: {min(all_val_epochs)} to {max(all_val_epochs)}")


# ============================================
# STEP 2: Skip first 3 epochs and REASSIGN epoch numbers
# ============================================


SKIP_EPOCHS = 3


# Skip first 3 epochs
filtered_val_srocc = all_val_srocc[SKIP_EPOCHS:]
filtered_val_plcc = all_val_plcc[SKIP_EPOCHS:]
filtered_val_rmse = all_val_rmse[SKIP_EPOCHS:]


# IMPORTANT: Reassign epoch numbers sequentially from 4 to 50
filtered_epochs = list(range(3, 3 + len(filtered_val_srocc)))


print(f"\n   After filtering:")
print(f"   Epochs shown: {filtered_epochs[0]} to {filtered_epochs[-1]}")


# ============================================
# STEP 3: Extract training loss for EVERY 10th EPOCH
# ============================================

# Find indices where epoch changes and is divisible by 10
train_loss_epochs_5 = []
train_loss_values_5 = []

for i, (loss, epoch) in enumerate(zip(all_train_losses, all_train_epochs)):
    # Get last loss value at the end of every 10th epoch (e.g., epochs 10, 20, 30, etc.)
    if epoch >= 3 and epoch % 5 == 0:  # Every 10th epoch, starting from epoch 10
        # Store the LAST loss value of this epoch (i.e., just before next epoch starts)
        if i == len(all_train_losses) - 1 or all_train_epochs[i+1] != epoch:
            train_loss_epochs_5.append(epoch)
            train_loss_values_5.append(loss)

print(f"\n   Training loss sampled at epochs: {train_loss_epochs_5}")


# ============================================
# STEP 4: Create Visualizations (2x2 grid)
# ============================================


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')


# ============================================
# Plot 1 (Top Left): Training Loss (Every 10 Epochs)
# ============================================
axes[0, 0].plot(train_loss_epochs_5, train_loss_values_5, marker='o', linewidth=2.5, 
                markersize=8, color='#1f77b4', markerfacecolor='#1f77b4', markeredgewidth=0)
axes[0, 0].set_title('Training Loss (Every 10 Epochs)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
axes[0, 0].set_facecolor('#f8f9fa')
axes[0, 0].set_xticks(train_loss_epochs_5)  # Show all sampled epochs


# ============================================
# Plot 2 (Top Right): Validation SROCC
# ============================================
axes[0, 1].plot(filtered_epochs, filtered_val_srocc, marker='o', linewidth=2.5, 
                markersize=7, color='#ff7f0e', markerfacecolor='#ff7f0e', markeredgewidth=0)
axes[0, 1].set_title('Validation SROCC', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('SROCC', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
axes[0, 1].set_facecolor('#f8f9fa')
axes[0, 1].set_ylim([0, 1.0])
axes[0, 1].set_xticks(range(3, 51, 5))


# ============================================
# Plot 3 (Bottom Left): Validation PLCC
# ============================================
axes[1, 0].plot(filtered_epochs, filtered_val_plcc, marker='s', linewidth=2.5, 
                markersize=7, color='#2ca02c', markerfacecolor='#2ca02c', markeredgewidth=0)
axes[1, 0].set_title('Validation PLCC', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('PLCC', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
axes[1, 0].set_facecolor('#f8f9fa')
axes[1, 0].set_ylim([0, 1.0])
axes[1, 0].set_xticks(range(3, 51, 5))


# ============================================
# Plot 4 (Bottom Right): Validation RMSE
# ============================================
axes[1, 1].plot(filtered_epochs, filtered_val_rmse, marker='^', linewidth=2.5, 
                markersize=7, color='#d62728', markerfacecolor='#d62728', markeredgewidth=0)
axes[1, 1].set_title('Validation RMSE', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('RMSE', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
axes[1, 1].set_facecolor('#f8f9fa')
axes[1, 1].set_xticks(range(3, 51, 5))


# ============================================
# Overall layout
# ============================================
plt.suptitle('Training Metrics (Epochs 3-50)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output_ppt\training_metrics_3to50.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


print("\n✓ Saved: training_metrics_3to50.png")


# ============================================
# SUMMARY STATISTICS
# ============================================
print("\n" + "="*60)
print("TRAINING SUMMARY (EPOCHS 3-50)")
print("="*60)


if filtered_val_srocc:
    best_idx = np.argmax(filtered_val_srocc)
    print(f"\n🏆 Best SROCC: {max(filtered_val_srocc):.4f} at Epoch {filtered_epochs[best_idx]}")


if filtered_val_plcc:
    best_idx = np.argmax(filtered_val_plcc)
    print(f"🏆 Best PLCC: {max(filtered_val_plcc):.4f} at Epoch {filtered_epochs[best_idx]}")


if filtered_val_rmse:
    best_idx = np.argmin(filtered_val_rmse)
    print(f"🏆 Best RMSE: {min(filtered_val_rmse):.4f} at Epoch {filtered_epochs[best_idx]}")


print(f"\n📊 Training Loss at 10-Epoch Intervals:")
for epoch, loss in zip(train_loss_epochs_5, train_loss_values_5):
    print(f"   Epoch {epoch}: {loss:.4f}")

print(f"\n📊 Final Epoch Metrics (Epoch {filtered_epochs[-1]}):")
print(f"   SROCC: {filtered_val_srocc[-1]:.4f}")
print(f"   PLCC: {filtered_val_plcc[-1]:.4f}")
print(f"   RMSE: {filtered_val_rmse[-1]:.4f}")


print(f"\n✅ Visualization generated successfully!")
print("="*60)
