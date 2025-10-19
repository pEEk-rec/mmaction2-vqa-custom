"""Create clean GIFs for presentation with visual zoom effect."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from mmaction.datasets.transforms import *
from mmaction.registry import TRANSFORMS


def create_clean_frame(frame, frame_idx, total_frames=8, zoom_factor=1.0):
    """Create clean frame with optional zoom effect."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    
    # Apply zoom by adjusting display limits
    h, w = frame.shape[:2]
    if zoom_factor > 1.0:
        # Calculate center crop area to simulate zoom
        crop_h = int(h / zoom_factor)
        crop_w = int(w / zoom_factor)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        ax.imshow(frame)
        ax.set_xlim(start_w, start_w + crop_w)
        ax.set_ylim(start_h + crop_h, start_h)
    else:
        ax.imshow(frame)
    
    ax.set_title(f'Frame {frame_idx+1}/{total_frames}', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    
    return img[:, :, :3]


def create_stage_gifs(video_path, output_dir=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\Finalstage_gifs2'):
    """Create 4 clean GIFs with visual zoom progression."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Creating Clean Stage GIFs")
    print("="*60)
    
    results = {'filename': video_path, 'modality': 'RGB'}
    
    # Stage 1: Original frames
    print("\nStage 1: Original frames...")
    decord_init = TRANSFORMS.build(dict(type='DecordInit'))
    sample_frames = TRANSFORMS.build(dict(
        type='UniformSampleFrames', clip_len=8, num_clips=1, test_mode=False
    ))
    decode = TRANSFORMS.build(dict(type='DecordDecode'))
    
    results = decord_init(results)
    results = sample_frames(results)
    results = decode(results)
    
    original_frames = results['imgs'].copy()
    print(f"  Size: {original_frames[0].shape[1]}×{original_frames[0].shape[0]}")
    
    gif1_frames = []
    for idx, frame in enumerate(original_frames):
        img = create_clean_frame(frame, idx, zoom_factor=1.0)
        gif1_frames.append(img)
    
    gif1_path = os.path.join(output_dir, '1_original_frames.gif')
    imageio.mimsave(gif1_path, gif1_frames, duration=8000, loop=0)  # 1000ms = 1 second per frame
    print(f"  Saved: {gif1_path}")
    
    # Stage 2: Resize
    print("\nStage 2: After resize...")
    resize = TRANSFORMS.build(dict(type='Resize', scale=(-1, 256), keep_ratio=True))
    results = resize(results)
    
    resized_frames = results['imgs'].copy()
    print(f"  Size: {resized_frames[0].shape[1]}×{resized_frames[0].shape[0]}")
    
    gif2_frames = []
    for idx, frame in enumerate(resized_frames):
        img = create_clean_frame(frame, idx, zoom_factor=1.0)
        gif2_frames.append(img)
    
    gif2_path = os.path.join(output_dir, '2_after_resize.gif')
    imageio.mimsave(gif2_path, gif2_frames, duration=8000, loop=0)  # 1000ms = 1 second per frame
    print(f"  Saved: {gif2_path}")
    
    # Stage 3: Center Crop (224x224)
    print("\nStage 3: After center crop...")
    center_crop = TRANSFORMS.build(dict(type='CenterCrop', crop_size=224))
    results = center_crop(results)
    
    cropped_frames = results['imgs'].copy()
    print(f"  Size: {cropped_frames[0].shape[1]}×{cropped_frames[0].shape[0]}")
    
    gif3_frames = []
    for idx, frame in enumerate(cropped_frames):
        img = create_clean_frame(frame, idx, zoom_factor=1.0)
        gif3_frames.append(img)
    
    gif3_path = os.path.join(output_dir, '3_after_crop.gif')
    imageio.mimsave(gif3_path, gif3_frames, duration=8000, loop=0)  # 1000ms = 1 second per frame
    print(f"  Saved: {gif3_path}")
    
    # Stage 4: Final format
    print("\nStage 4: Final processed...")
    
    gif4_frames = []
    for idx, frame in enumerate(cropped_frames):
        img = create_clean_frame(frame, idx, zoom_factor=1.0)
        gif4_frames.append(img)
    
    gif4_path = os.path.join(output_dir, '4_final_processed.gif')
    imageio.mimsave(gif4_path, gif4_frames, duration=8000, loop=0)  # 1000ms = 1 second per frame
    print(f"  Saved: {gif4_path}")
    
    print("\n" + "="*60)
    print("All GIFs created successfully!")
    print(f"Location: {output_dir}")
    print("\nFiles:")
    print("  1_original_frames.gif (1366×768)")
    print("  2_after_resize.gif (455×256)")
    print("  3_after_crop.gif (224×224)")
    print("  4_final_processed.gif (224×224)")
    print("\nEach frame shown for 1 second (total 8s per GIF)")
    print("="*60)


if __name__ == '__main__':
    video_path = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos\00002_08_blur_demo.mp4'
    
    create_stage_gifs(video_path)
