import pandas as pd
import numpy as np

train_df = pd.read_csv('AI-VQA/Annotations/train.csv')

# Calculate class weights for imbalanced artifacts
def calculate_class_weight(labels):
    """Calculate inverse frequency weights"""
    pos = labels.sum()
    neg = len(labels) - pos
    neg_weight = 1.0
    pos_weight = neg / pos if pos > 0 else 1.0
    return [neg_weight, pos_weight]

# Compute weights
hallucination_weight = calculate_class_weight(train_df['hallucination_flag'])
lighting_weight = calculate_class_weight(train_df['lighting_flag'])
spatial_weight = calculate_class_weight(train_df['spatial_flag'])
rendering_weight=calculate_class_weight(train_df['rendering_flag'])

print(f"Hallucination: {hallucination_weight}")  # Should be ~[1.0, 2.07]
print(f"Lighting:      {lighting_weight}")       # Should be ~[1.0, 3.26]
print(f"Spatial:       {spatial_weight}")        # Should be ~[1.0, 9.36]
print(f"Spatial:       {rendering_weight}")
# Update config if values differ significantly