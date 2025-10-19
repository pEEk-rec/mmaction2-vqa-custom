import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

def mos_to_quality_label(mos):
    """Convert MOS to quality class."""
    if mos < 2.0: return 0
    elif mos < 3.0: return 1
    elif mos < 4.0: return 2
    elif mos < 4.5: return 3
    else: return 4

# ============================================================================
# PATHS
# ============================================================================
videos_dir = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\videos_all\videos_all_videos'
ann_dir = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\Annotations'
mos_train_csv = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\FineVD\FineVD\MOS_train.csv'
mos_val_csv = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\FineVD\FineVD\MOS_val.csv'

quality_type = 'overall'
os.makedirs(ann_dir, exist_ok=True)

# ============================================================================
# GET AVAILABLE VIDEOS
# ============================================================================
print("Scanning video directory...")
available_videos = set(os.listdir(videos_dir))
print(f"Found {len(available_videos)} videos in folder")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading MOS files...")
train_df = pd.read_csv(mos_train_csv)
val_full_df = pd.read_csv(mos_val_csv)

print(f"CSV entries - Train: {len(train_df)}, Val: {len(val_full_df)}")

video_id_col = 'video_name'
train_df['mos'] = train_df[quality_type]
val_full_df['mos'] = val_full_df[quality_type]
train_df['quality_class'] = train_df['mos'].apply(mos_to_quality_label)
val_full_df['quality_class'] = val_full_df['mos'].apply(mos_to_quality_label)

# ============================================================================
# FILTER TO ONLY EXISTING VIDEOS
# ============================================================================
print("\nFiltering to only videos that exist...")
train_df = train_df[train_df[video_id_col].isin(available_videos)]
val_full_df = val_full_df[val_full_df[video_id_col].isin(available_videos)]

print(f"After filtering - Train: {len(train_df)}, Val: {len(val_full_df)}")
print(f"Total usable: {len(train_df) + len(val_full_df)}")

# ============================================================================
# SPLIT VAL INTO VAL + TEST
# ============================================================================
if len(val_full_df) > 0:
    val_df, test_df = train_test_split(
        val_full_df, test_size=0.5, random_state=42,
        stratify=val_full_df['quality_class']
    )
else:
    print("ERROR: No validation videos found!")
    exit(1)

print(f"\nFinal split:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

# ============================================================================
# CREATE ANNOTATIONS
# ============================================================================
def create_annotations(df, video_id_col):
    annotations = []
    for _, row in df.iterrows():
        video_name = row[video_id_col]
        annotations.append({
            'video_id': str(video_name),
            'video_path': video_name,
            'mos': float(row['mos']),
            'quality_class': int(row['quality_class']),
            'color': float(row['color']),
            'noise': float(row['noise']),
            'artifact': float(row['artifact']),
            'blur': float(row['blur']),
            'temporal': float(row['temporal']),
            'overall': float(row['overall'])
        })
    return annotations

print("\nCreating annotation files...")

train_json = os.path.join(ann_dir, 'train.json')
with open(train_json, 'w') as f:
    json.dump(create_annotations(train_df, video_id_col), f, indent=2)
print(f"✓ {train_json}")

val_json = os.path.join(ann_dir, 'val.json')
with open(val_json, 'w') as f:
    json.dump(create_annotations(val_df, video_id_col), f, indent=2)
print(f"✓ {val_json}")

test_json = os.path.join(ann_dir, 'test.json')
with open(test_json, 'w') as f:
    json.dump(create_annotations(test_df, video_id_col), f, indent=2)
print(f"✓ {test_json}")

print("\n✓ Done!")