import os
import numpy as np
from decord import VideoReader, cpu

# Path to your video folder
video_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\kinetics400_tiny\kinetics400_tiny\train'

frame_counts = []

# List all video files (assuming common video extensions)
video_extensions = ('.mp4', '.avi', '.mkv', '.mov')

# Iterate all videos in folder (recursive if needed)
for root, _, files in os.walk(video_root):
    for file in files:
        if file.endswith(video_extensions):
            video_path = os.path.join(root, file)
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                frame_counts.append(len(vr))
            except Exception as e:
                print(f"Warning: Could not read {video_path}: {e}")

if frame_counts:
    frame_counts_np = np.array(frame_counts)
    print(f"Videos processed: {len(frame_counts_np)}")
    print(f"Min frames per video: {frame_counts_np.min()}")
    print(f"Median frames per video: {np.median(frame_counts_np)}")
    print(f"Max frames per video: {frame_counts_np.max()}")
else:
    print("No videos found or failed to read any video.")
