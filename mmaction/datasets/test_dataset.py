# test_dataset.py
import json

ann_file = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\Annotations\train.json'

with open(ann_file, 'r') as f:
    data = json.load(f)

print(f"Annotations file has {len(data)} entries")
print(f"First entry: {data[0]}")