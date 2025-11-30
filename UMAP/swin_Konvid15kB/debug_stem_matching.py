"""
Debug script to identify stem matching issues between extracted features and training CSV
"""
import pandas as pd
from pathlib import Path
import argparse

def debug_stems(extracted_list_path, train_csv_path):
    # Load extracted list
    df_extracted = pd.read_csv(extracted_list_path)
    if 'stem' not in df_extracted.columns:
        df_extracted.columns = ['stem']
    
    extracted_stems = set(df_extracted['stem'].astype(str))
    
    # Load training CSV
    df_train = pd.read_csv(train_csv_path)
    
    print("=" * 80)
    print("TRAINING CSV COLUMNS:")
    print("=" * 80)
    print(df_train.columns.tolist())
    print()
    
    print("=" * 80)
    print("TRAINING CSV HEAD (first 5 rows):")
    print("=" * 80)
    print(df_train.head())
    print()
    
    # Try to identify filename column
    filename_col = None
    for col in ['filename', 'video_name', 'name', 'file']:
        if col in df_train.columns:
            filename_col = col
            break
    
    if filename_col is None:
        print("ERROR: Could not find filename column in training CSV!")
        print("Available columns:", df_train.columns.tolist())
        return
    
    # Extract stems from training CSV
    df_train['stem'] = df_train[filename_col].apply(lambda x: Path(str(x)).stem)
    train_stems = set(df_train['stem'].astype(str))
    
    print("=" * 80)
    print("STEM MATCHING ANALYSIS:")
    print("=" * 80)
    print(f"Extracted features stems: {len(extracted_stems)}")
    print(f"Training CSV stems: {len(train_stems)}")
    print()
    
    # Find matching stems
    matching = extracted_stems & train_stems
    print(f"✓ MATCHING stems: {len(matching)}")
    
    # Find non-matching
    in_extracted_not_train = extracted_stems - train_stems
    in_train_not_extracted = train_stems - extracted_stems
    
    print(f"✗ In extracted but NOT in training: {len(in_extracted_not_train)}")
    print(f"✗ In training but NOT in extracted: {len(in_train_not_extracted)}")
    print()
    
    if len(matching) > 0:
        print("=" * 80)
        print("SAMPLE MATCHING STEMS (first 10):")
        print("=" * 80)
        for stem in list(matching)[:10]:
            print(f"  {stem}")
        print()
    
    if len(in_extracted_not_train) > 0:
        print("=" * 80)
        print("SAMPLE EXTRACTED STEMS NOT IN TRAINING (first 10):")
        print("=" * 80)
        for stem in list(in_extracted_not_train)[:10]:
            print(f"  {stem}")
        print()
        
        print("Let's check the actual filenames in extracted list:")
        print(df_extracted.head(10))
        print()
    
    if len(in_train_not_extracted) > 0:
        print("=" * 80)
        print("SAMPLE TRAINING STEMS NOT IN EXTRACTED (first 10):")
        print("=" * 80)
        train_sample = df_train[df_train['stem'].isin(in_train_not_extracted)].head(10)
        for _, row in train_sample.iterrows():
            print(f"  Original: {row[filename_col]} -> Stem: {row['stem']}")
        print()
    
    # Check if it's a test vs train issue
    print("=" * 80)
    print("DIAGNOSTIC HINTS:")
    print("=" * 80)
    
    if len(matching) == 0:
        print("⚠ NO MATCHES FOUND!")
        print()
        print("Possible issues:")
        print("1. You extracted features from TEST set but trying to plot TRAIN set")
        print("2. File naming format differs (e.g., with/without 'orig_' prefix)")
        print("3. File extensions included in one but not the other")
        print()
        
        # Show examples side by side
        print("Sample stems comparison:")
        print(f"  Extracted: {list(extracted_stems)[:3]}")
        print(f"  Training:  {list(train_stems)[:3]}")
    elif len(matching) < len(extracted_stems):
        print(f"⚠ Only partial match: {len(matching)}/{len(extracted_stems)}")
        print("Some extracted features don't have corresponding training labels.")
    else:
        print("✓ All extracted stems found in training CSV!")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug stem matching between extracted features and training CSV")
    parser.add_argument("--extracted_list", required=True, help="Path to extracted_video_list.csv")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    args = parser.parse_args()
    
    debug_stems(args.extracted_list, args.train_csv)