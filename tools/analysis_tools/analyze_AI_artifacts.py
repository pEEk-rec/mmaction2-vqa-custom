import pandas as pd
import torch

# Load your training CSV
df = pd.read_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\AI-VQA\metadata\demo_metadata.csv')

print("="*60)
print("ARTIFACT FLAG DISTRIBUTION ANALYSIS")
print("="*60)

artifact_flags = ['hallucination_flag', 'rendering_flag', 'lighting_flag', 'spatial_flag']
pos_weights = {}

for flag in artifact_flags:
    pos = df[flag].sum()
    neg = len(df) - pos
    ratio = pos / len(df) if len(df) > 0 else 0
    imbalance = neg / pos if pos > 0 else float('inf')
    
    print(f"\n{flag.upper()}:")
    print(f"  Positive: {pos}/{len(df)} ({ratio:.2%})")
    print(f"  Negative: {neg}/{len(df)} ({(1-ratio):.2%})")
    print(f"  Imbalance ratio: {imbalance:.2f}:1 (neg:pos)")
    
    # Calculate pos_weight (ratio of negatives to positives)
    pos_weight = imbalance if pos > 0 else 1.0
    pos_weights[flag] = pos_weight
    
    # Recommendation
    if imbalance > 5.0:
        print(f"  ⚠️  SEVERE IMBALANCE - Recommend pos_weight={pos_weight:.2f}")
    elif imbalance > 2.0:
        print(f"  ⚠️  MODERATE IMBALANCE - Consider pos_weight={pos_weight:.2f}")
    else:
        print(f"  ✅ BALANCED - pos_weight not needed")

print("\n" + "="*60)
print("RECOMMENDED CONFIG:")
print("="*60)
print("\ncls_head=dict(")
print("    type='swin_AIHead',")
print("    num_classes=5,")
print("    in_channels=768,")
print("    loss_mos=dict(type='MSELoss', loss_weight=1.5),")
print("    loss_cls=dict(type='CrossEntropyLoss', loss_weight=0.5),")

for flag in artifact_flags:
    flag_name = flag.replace('_flag', '')
    pw = pos_weights[flag]
    if pw > 2.0:
        print(f"    loss_{flag_name}=dict(type='BCEWithLogitsLoss', loss_weight=0.8, pos_weight=torch.tensor([{pw:.2f}])),")
    else:
        print(f"    loss_{flag_name}=dict(type='BCEWithLogitsLoss', loss_weight=0.8),")
print(")")