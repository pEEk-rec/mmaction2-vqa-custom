"""Create individual GIFs showing actual size differences at each stage."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from mmaction.datasets.transforms import *
from mmaction.registry import TRANSFORMS


def create_frame_with_size_overlay(frame, title, stage_num, total_stages=4):
    """Create frame visualization with size information visible."""
    h, w = frame.shape[:2]
    
    # Create figure with fixed size to show scale differences
    # Use different figure sizes to show actual size changes
    if stage_num == 1:
        fig_size = (12, 9)  # Largest for original
    elif stage_num == 2:
        fig_size = (10, 7.5)  # Smaller after resize
    elif stage_num >= 3:
        fig_size = (8, 6)  # Smallest for cropped
    
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=100)
    
    # Display image
    ax.imshow(frame)
    
    # Add border to show frame boundary
    rect = patches.Rectangle((0, 0), w-1, h-1, linewidth=3, 
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # Add size label overlay on image
    ax.text(w//2, 30, f'{w} × {h} pixels', 
            fontsize=20, fontweight='bold', color='yellow',
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    
    return img[:, :, :3]  # RGB only


def create_stage_gifs_with_size_diff(video_path, output_dir=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\outputs\pptstage_gifs'):
    """Create 4 GIFs showing visible size differences."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Creating Stage-wise GIF Visualizations")
    print("Size differences will be visible!")
    print("="*60)
    
    results = {'filename': video_path, 'modality': 'RGB'}
    
    # Stage 1: Original frames
    print("\nStage 1: Loading original frames...")
    decord_init = TRANSFORMS.build(dict(type='DecordInit'))
    sample_frames = TRANSFORMS.build(dict(
        type='UniformSampleFrames', clip_len=8, num_clips=1, test_mode=False
    ))
    decode = TRANSFORMS.build(dict(type='DecordDecode'))
    
    results = decord_init(results)
    results = sample_frames(results)
    results = decode(results)
    
    original_frames = results['imgs'].copy()
    h, w = original_frames[0].shape[:2]
    print(f"  Original size: {w}×{h}")
    
    gif1_frames = []
    for idx, frame in enumerate(original_frames):
        img = create_frame_with_size_overlay(
            frame, 
            f'Stage 1: Original Frame {idx+1}/8',
            stage_num=1
        )
        gif1_frames.append(img)
    
    gif1_path = os.path.join(output_dir, '1_original_frames.gif')
    imageio.mimsave(gif1_path, gif1_frames, duration=0.5, loop=0)
    print(f"  Saved: {gif1_path}")
    
    # Stage 2: Resize
    print("\nStage 2: Resizing frames...")
    resize = TRANSFORMS.build(dict(type='Resize', scale=(-1, 256), keep_ratio=True))
    results = resize(results)
    
    resized_frames = results['imgs'].copy()
    h, w = resized_frames[0].shape[:2]
    print(f"  Resized to: {w}×{h}")
    
    gif2_frames = []
    for idx, frame in enumerate(resized_frames):
        img = create_frame_with_size_overlay(
            frame,
            f'Stage 2: After Resize {idx+1}/8',
            stage_num=2
        )
        gif2_frames.append(img)
    
    gif2_path = os.path.join(output_dir, '2_after_resize.gif')
    imageio.mimsave(gif2_path, gif2_frames, duration=0.5, loop=0)
    print(f"  Saved: {gif2_path}")
    
    # Stage 3: Center Crop
    print("\nStage 3: Center cropping frames...")
    center_crop = TRANSFORMS.build(dict(type='CenterCrop', crop_size=224))
    results = center_crop(results)
    
    cropped_frames = results['imgs'].copy()
    h, w = cropped_frames[0].shape[:2]
    print(f"  Cropped to: {w}×{h}")
    
    gif3_frames = []
    for idx, frame in enumerate(cropped_frames):
        img = create_frame_with_size_overlay(
            frame,
            f'Stage 3: After Center Crop {idx+1}/8',
            stage_num=3
        )
        gif3_frames.append(img)
    
    gif3_path = os.path.join(output_dir, '3_after_crop.gif')
    imageio.mimsave(gif3_path, gif3_frames, duration=0.5, loop=0)
    print(f"  Saved: {gif3_path}")
    
    # Stage 4: Final format
    print("\nStage 4: Final tensor format...")
    format_shape = TRANSFORMS.build(dict(type='FormatShape', input_format='NCTHW'))
    results_formatted = format_shape(results)
    
    final_frames = cropped_frames
    print(f"  Final tensor: [1, 3, 8, 224, 224]")
    
    gif4_frames = []
    for idx, frame in enumerate(final_frames):
        img = create_frame_with_size_overlay(
            frame,
            f'Stage 4: Final Tensor Format {idx+1}/8',
            stage_num=4
        )
        gif4_frames.append(img)
    
    gif4_path = os.path.join(output_dir, '4_final_processed.gif')
    imageio.mimsave(gif4_path, gif4_frames, duration=0.5, loop=0)
    print(f"  Saved: {gif4_path}")
    
    # Create comparison grid
    print("\nCreating size comparison grid...")
    create_size_comparison_image(
        original_frames[0], 
        resized_frames[0], 
        cropped_frames[0],
        output_dir
    )
    
    print("\n" + "="*60)
    print("All visualizations created!")
    print(f"Location: {output_dir}")
    print("\nFiles:")
    print("  1_original_frames.gif")
    print("  2_after_resize.gif")
    print("  3_after_crop.gif")
    print("  4_final_processed.gif")
    print("  size_comparison.png (static comparison)")
    print("="*60)


def create_size_comparison_image(frame1, frame2, frame3, output_dir):
    """Create static image showing size progression."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    frames = [frame1, frame2, frame3]
    titles = ['Original', 'After Resize', 'After Crop']
    
    for ax, frame, title in zip(axes, frames, titles):
        h, w = frame.shape[:2]
        ax.imshow(frame)
        ax.set_title(f'{title}\n{w}×{h} pixels', fontsize=14, fontweight='bold')
        
        # Add border
        rect = patches.Rectangle((0, 0), w-1, h-1, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')
    
    plt.suptitle('Preprocessing Size Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comp_path = os.path.join(output_dir, 'size_comparison.png')
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comp_path}")


if __name__ == '__main__':
    video_path = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos\00002_08_blur_demo.mp4'
    
    create_stage_gifs_with_size_diff(video_path)
