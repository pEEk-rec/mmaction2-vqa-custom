# analyze_test_results.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Load test results
with open(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\results_epo50.pkl', 'rb') as f:
    results = pickle.load(f)

# Load test annotations
test_csv = pd.read_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\Konvid\k150kb\Annotations\konvid150k_test.csv')

# Extract predictions and ground truth
predictions = []
ground_truths = []

for result, row in zip(results, test_csv.iterrows()):
    if isinstance(result, dict):
        pred_mos = result.get('pred_mos', result.get('mos_pred'))
    else:
        pred_mos = result.pred_mos
    
    predictions.append(pred_mos)
    ground_truths.append(row[1]['video_score'])

predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

# Calculate metrics
srocc, _ = spearmanr(predictions, ground_truths)
plcc, _ = pearsonr(predictions, ground_truths)
rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))

print("="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"SROCC: {srocc:.4f}")
print(f"PLCC:  {plcc:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"Total test samples: {len(predictions)}")
print("="*50)

# Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(ground_truths, predictions, alpha=0.6, s=50)
plt.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Ground Truth MOS', fontsize=14)
plt.ylabel('Predicted MOS', fontsize=14)
plt.title(f'Test Set: Predicted vs Ground Truth\nSROCC={srocc:.3f}, PLCC={plcc:.3f}', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\test_predictions_scatter2.png', dpi=300)
plt.close()

# Error distribution
errors = predictions - ground_truths
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (Predicted - True)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title(f'Test Set Error Distribution\nMean Error: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\test_error_distribution2.png', dpi=300)
plt.close()

# Find worst predictions
worst_indices = np.argsort(np.abs(errors))[-10:][::-1]
print("\nTop 10 Worst Predictions:")
print("-"*70)
for idx in worst_indices:
    video_name = test_csv.iloc[idx]['video_name']
    true_mos = ground_truths[idx]
    pred_mos = predictions[idx]
    error = errors[idx]
    print(f"{video_name:40s} True: {true_mos:.2f}  Pred: {pred_mos:.2f}  Error: {error:+.2f}")

print("\n✅ Analysis complete! Check test_predictions_scatter.png and test_error_distribution.png")
