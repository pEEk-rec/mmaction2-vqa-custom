import json

ann_dir = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_dataset\Annotations'

for filename in ['train.json', 'val.json', 'test.json']:
    filepath = f'{ann_dir}\\{filename}'
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for item in data:
        if item['video_path'].endswith('.mp4.mp4'):
            item['video_path'] = item['video_path'][:-4]
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed {len(data)} entries in {filename}")