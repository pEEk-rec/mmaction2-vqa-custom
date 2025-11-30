import pandas as pd

# Load your annotations
train_df = pd.read_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\ai_train.csv')
val_df = pd.read_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\ai_val.csv')
test_df = pd.read_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\Annotations\ai_test.csv')

print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")
print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")

# CHECK 1: No overlap between splits
train_files = set(train_df['filename'])
val_files = set(val_df['filename'])
test_files = set(test_df['filename'])

overlap_train_val = train_files & val_files
overlap_train_test = train_files & test_files
overlap_val_test = val_files & test_files

assert len(overlap_train_val) == 0, f"Train-Val overlap: {overlap_train_val}"
assert len(overlap_train_test) == 0, f"Train-Test overlap: {overlap_train_test}"
assert len(overlap_val_test) == 0, f"Val-Test overlap: {overlap_val_test}"
print("✅ No data leakage between splits")

# CHECK 2: Class distribution
print("\nMOS Distribution:")
print(f"Train - Mean: {train_df['mos'].mean():.2f}, Std: {train_df['mos'].std():.2f}")
print(f"Val   - Mean: {val_df['mos'].mean():.2f}, Std: {val_df['mos'].std():.2f}")
print(f"Test  - Mean: {test_df['mos'].mean():.2f}, Std: {test_df['mos'].std():.2f}")

print("\nArtifact Distribution:")
for artifact in ['hallucination_flag', 'lighting_flag', 'spatial_flag', 'rendering_flag']:
    print(f"\n{artifact}:")
    print(f"  Train: {train_df[artifact].sum()}/{len(train_df)} ({100*train_df[artifact].mean():.1f}%)")
    print(f"  Val:   {val_df[artifact].sum()}/{len(val_df)} ({100*val_df[artifact].mean():.1f}%)")
    print(f"  Test:  {test_df[artifact].sum()}/{len(test_df)} ({100*test_df[artifact].mean():.1f}%)")

