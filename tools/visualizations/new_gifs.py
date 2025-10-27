"""
Create 8 detailed GIFs showing each transformation stage of Video Swin Transformer pipeline.
Each stage shows clear visual differences and transformations.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from mmaction.datasets.transforms import *
from mmaction.registry import TRANSFORMS


def create_detailed_frame(frame, frame_idx, total_frames, stage_num, stage_name, stage_info):
    """Create detailed frame with comprehensive information and visual guides."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 9), dpi=120)
    
    # Main image area (larger)
    ax_img = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    ax_img.imshow(frame)
    
    # Add dimension annotations on the image
    h, w = frame.shape[:2]
    
    # Width annotation (top)
    ax_img.annotate('', xy=(0, -20), xytext=(w, -20),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                    annotation_clip=False)
    ax_img.text(w/2, -40, f'{w}px', ha='center', va='top', 
                fontsize=11, fontweight='bold', color='red')
    
    # Height annotation (left)
    ax_img.annotate('', xy=(-20, 0), xytext=(-20, h),
                    arrowprops=dict(arrowstyle='<->', color='blue', lw=2),
                    annotation_clip=False)
    ax_img.text(-40, h/2, f'{h}px', ha='right', va='center', 
                fontsize=11, fontweight='bold', color='blue', rotation=90)
    
    # Stage title
    title = f'Stage {stage_num}: {stage_name}\nFrame {frame_idx+1}/{total_frames} | {w}×{h}×{frame.shape[2]}'
    ax_img.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax_img.axis('off')
    
    # Info panel at bottom
    ax_info = plt.subplot2grid((5, 1), (4, 0))
    ax_info.axis('off')
    
    # Progress bar
    progress = (frame_idx + 1) / total_frames
    bar_width = 0.85
    bar_start = 0.075
    
    # Background bar
    ax_info.add_patch(
        patches.Rectangle((bar_start, 0.6), bar_width, 0.15, 
                          facecolor='lightgray', edgecolor='black', linewidth=1.5)
    )
    # Progress bar
    ax_info.add_patch(
        patches.Rectangle((bar_start, 0.6), bar_width * progress, 0.15, 
                          facecolor='green', alpha=0.7)
    )
    ax_info.text(bar_start + bar_width/2, 0.675, 
                 f'{int(progress*100)}%', 
                 ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Stage information text
    info_text = f'{stage_info}'
    ax_info.text(0.5, 0.25, info_text, 
                 ha='center', va='center', fontsize=10,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()
    img_out = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    
    return img_out[:, :, :3]


def create_8_stage_gifs(
    video_path, 
    output_dir=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\VideoSwin_8stages',
    fps=15
):
    """Create 8 separate GIFs showing each transformation stage."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Creating 8-Stage Video Swin Transformer Pipeline Visualization")
    print("="*80)
    
    file_client_args = dict(io_backend='disk')
    
    results = {
        'filename': video_path, 
        'modality': 'RGB',
        'start_index': 0
    }
    
    # ============================================================
    # STAGE 1: Original Video (before any processing)
    # ============================================================
    print("\n[Stage 1/8] Original Video (Raw)...")
    
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total_video_frames = len(vr)
    video_fps = vr.get_avg_fps()
    
    # Show first 32 frames of original video
    original_video_frames = []
    for i in range(min(32, total_video_frames)):
        original_video_frames.append(vr[i].asnumpy())
    
    orig_video_h, orig_video_w = original_video_frames[0].shape[:2]
    print(f"  Original video: {orig_video_w}×{orig_video_h}, {total_video_frames} frames, {video_fps:.2f} FPS")
    
    gif1_frames = []
    stage_info = f'Raw video | Total: {total_video_frames} frames | FPS: {video_fps:.2f}'
    for idx, frame in enumerate(original_video_frames):
        img = create_detailed_frame(
            frame, idx, len(original_video_frames), 1,
            "Original Video (Raw)", stage_info
        )
        gif1_frames.append(img)
    
    gif1_path = os.path.join(output_dir, '1_original_video_raw.gif')
    imageio.mimsave(gif1_path, gif1_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 1_original_video_raw.gif")
    
    # ============================================================
    # STAGE 2: DecordInit (Video loaded into memory)
    # ============================================================
    print("\n[Stage 2/8] DecordInit...")
    
    decord_init = TRANSFORMS.build(dict(type='DecordInit', **file_client_args))
    results = decord_init(results)
    print(f"  Video container initialized: {results.get('total_frames')} frames")
    
    # Show same frames as Stage 1 for comparison
    gif2_frames = []
    stage_info = f'Video loaded | Container ready | {results.get("total_frames")} frames available'
    for idx, frame in enumerate(original_video_frames):
        img = create_detailed_frame(
            frame, idx, len(original_video_frames), 2,
            "DecordInit (Video Container)", stage_info
        )
        gif2_frames.append(img)
    
    gif2_path = os.path.join(output_dir, '2_decord_init.gif')
    imageio.mimsave(gif2_path, gif2_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 2_decord_init.gif")
    
    # ============================================================
    # STAGE 3: SampleFrames (Select 32 frames with interval=2)
    # ============================================================
    print("\n[Stage 3/8] SampleFrames...")
    
    sample_frames = TRANSFORMS.build(dict(
        type='SampleFrames', 
        clip_len=32, 
        frame_interval=2, 
        num_clips=1
    ))
    results = sample_frames(results)
    sampled_indices = results['frame_inds']
    print(f"  Sampled {len(sampled_indices)} frames: {sampled_indices[:5].tolist()}...{sampled_indices[-5:].tolist()}")
    
    # Show the SELECTED frames (indices visualized)
    sampled_visual_frames = []
    for idx in sampled_indices:
        sampled_visual_frames.append(vr[int(idx)].asnumpy())
    
    gif3_frames = []
    stage_info = f'Sampled every 2nd frame | Span: 64 positions | clip_len=32, interval=2'
    for idx, frame in enumerate(sampled_visual_frames):
        img = create_detailed_frame(
            frame, idx, len(sampled_visual_frames), 3,
            "SampleFrames (Selected Frames)", stage_info
        )
        gif3_frames.append(img)
    
    gif3_path = os.path.join(output_dir, '3_sample_frames.gif')
    imageio.mimsave(gif3_path, gif3_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 3_sample_frames.gif")
    
    # ============================================================
    # STAGE 4: DecordDecode (Decode into numpy arrays)
    # ============================================================
    print("\n[Stage 4/8] DecordDecode...")
    
    decode = TRANSFORMS.build(dict(type='DecordDecode'))
    results = decode(results)
    
    decoded_frames = results['imgs'].copy()
    total_frames = len(decoded_frames)
    decoded_h, decoded_w = decoded_frames[0].shape[:2]
    print(f"  Decoded {total_frames} frames to numpy arrays: {decoded_w}×{decoded_h}")
    
    gif4_frames = []
    stage_info = f'Decoded to RGB numpy arrays | Shape: (H={decoded_h}, W={decoded_w}, C=3)'
    for idx, frame in enumerate(decoded_frames):
        img = create_detailed_frame(
            frame, idx, total_frames, 4,
            "DecordDecode (Numpy Arrays)", stage_info
        )
        gif4_frames.append(img)
    
    gif4_path = os.path.join(output_dir, '4_decord_decode.gif')
    imageio.mimsave(gif4_path, gif4_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 4_decord_decode.gif")
    
    # ============================================================
    # STAGE 5: Resize (Height to 256, keep aspect ratio)
    # ============================================================
    print("\n[Stage 5/8] Resize...")
    
    resize = TRANSFORMS.build(dict(type='Resize', scale=(-1, 256)))
    results = resize(results)
    
    resized_frames = results['imgs'].copy()
    resized_h, resized_w = resized_frames[0].shape[:2]
    print(f"  Resized from {decoded_w}×{decoded_h} to {resized_w}×{resized_h}")
    
    gif5_frames = []
    stage_info = f'Height=256px, aspect ratio preserved | {decoded_w}×{decoded_h} → {resized_w}×{resized_h}'
    for idx, frame in enumerate(resized_frames):
        img = create_detailed_frame(
            frame, idx, total_frames, 5,
            "Resize (Height=256)", stage_info
        )
        gif5_frames.append(img)
    
    gif5_path = os.path.join(output_dir, '5_resize.gif')
    imageio.mimsave(gif5_path, gif5_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 5_resize.gif")
    
    # ============================================================
    # STAGE 6: CenterCrop (224×224)
    # ============================================================
    print("\n[Stage 6/8] CenterCrop...")
    
    center_crop = TRANSFORMS.build(dict(type='CenterCrop', crop_size=224))
    results = center_crop(results)
    
    cropped_frames = results['imgs'].copy()
    crop_h, crop_w = cropped_frames[0].shape[:2]
    print(f"  Center cropped from {resized_w}×{resized_h} to {crop_w}×{crop_h}")
    
    gif6_frames = []
    stage_info = f'Center 224×224 extracted | {resized_w}×{resized_h} → {crop_w}×{crop_h}'
    for idx, frame in enumerate(cropped_frames):
        img = create_detailed_frame(
            frame, idx, total_frames, 6,
            "CenterCrop (224×224)", stage_info
        )
        gif6_frames.append(img)
    
    gif6_path = os.path.join(output_dir, '6_centercrop.gif')
    imageio.mimsave(gif6_path, gif6_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 6_centercrop.gif")
    
    # ============================================================
    # STAGE 7: FormatShape (Convert to NCTHW tensor format)
    # ============================================================
    print("\n[Stage 7/8] FormatShape...")
    
    format_shape = TRANSFORMS.build(dict(type='FormatShape', input_format='NCTHW'))
    results = format_shape(results)
    
    formatted_imgs = results['imgs']
    print(f"  Formatted to NCTHW: {formatted_imgs.shape}")
    
    # Convert back for visualization
    imgs_visual = formatted_imgs.squeeze(0).transpose(1, 2, 3, 0)
    
    gif7_frames = []
    stage_info = f'NCTHW format | Shape: (N=1, C=3, T=32, H=224, W=224) = {formatted_imgs.shape}'
    for idx in range(total_frames):
        frame = imgs_visual[idx]
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        img = create_detailed_frame(
            frame, idx, total_frames, 7,
            "FormatShape (NCTHW Tensor)", stage_info
        )
        gif7_frames.append(img)
    
    gif7_path = os.path.join(output_dir, '7_format_shape.gif')
    imageio.mimsave(gif7_path, gif7_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 7_format_shape.gif")
    
    # ============================================================
    # STAGE 8: Final Input (Ready for Video Swin Transformer)
    # ============================================================
    print("\n[Stage 8/8] Final Model Input...")
    
    gif8_frames = []
    stage_info = f'Ready for Video Swin Transformer | Batch input: {formatted_imgs.shape}'
    for idx in range(total_frames):
        frame = imgs_visual[idx]
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        img = create_detailed_frame(
            frame, idx, total_frames, 8,
            "Final Model Input", stage_info
        )
        gif8_frames.append(img)
    
    gif8_path = os.path.join(output_dir, '8_final_model_input.gif')
    imageio.mimsave(gif8_path, gif8_frames, fps=fps, loop=0)
    print(f"  ✓ Saved: 8_final_model_input.gif")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("✓ All 8 GIFs created successfully!")
    print("="*80)
    print(f"\n📁 Output Directory: {output_dir}\n")
    
    print("📊 TRANSFORMATION SUMMARY:")
    print(f"  Stage 1: Original Video          → {orig_video_w}×{orig_video_h} ({total_video_frames} frames)")
    print(f"  Stage 2: DecordInit              → Video container loaded")
    print(f"  Stage 3: SampleFrames            → 32 frames selected (interval=2, span=64)")
    print(f"  Stage 4: DecordDecode            → {decoded_w}×{decoded_h} numpy arrays")
    print(f"  Stage 5: Resize                  → {resized_w}×{resized_h}")
    print(f"  Stage 6: CenterCrop              → {crop_w}×{crop_h}")
    print(f"  Stage 7: FormatShape             → {formatted_imgs.shape}")
    print(f"  Stage 8: Final Model Input       → Ready for Video Swin Transformer")
    
    print(f"\n🎬 GIF Details:")
    print(f"  • Frame rate: {fps} FPS")
    print(f"  • Frames per GIF: 32 (except Stage 1-2: showing first 32)")
    print(f"  • Duration: {32/fps:.1f} seconds per GIF")
    
    print(f"\n💾 Memory Progression:")
    original_mem = decoded_h * decoded_w * 3 * 32 / (1024**2)
    resized_mem = resized_h * resized_w * 3 * 32 / (1024**2)
    final_mem = 224 * 224 * 3 * 32 / (1024**2)
    print(f"  • After decode: {original_mem:.2f} MB")
    print(f"  • After resize: {resized_mem:.2f} MB")
    print(f"  • After crop: {final_mem:.2f} MB")
    print(f"  • Batch (size=2): {final_mem * 2:.2f} MB")
    
    print("\n📦 Generated Files:")
    print("  1_original_video_raw.gif      - Raw video before processing")
    print("  2_decord_init.gif             - Video container initialized")
    print("  3_sample_frames.gif           - Frame sampling (32 frames, interval=2)")
    print("  4_decord_decode.gif           - Decoded to numpy arrays")
    print("  5_resize.gif                  - Resized to height=256")
    print("  6_centercrop.gif              - Center cropped to 224×224")
    print("  7_format_shape.gif            - NCTHW tensor format")
    print("  8_final_model_input.gif       - Final input for model")
    print("="*80)


if __name__ == '__main__':
    video_path = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos\00002_08_blur_demo.mp4'
    
    # Create 8-stage GIFs at 15 FPS
    create_8_stage_gifs(video_path, fps=15)
