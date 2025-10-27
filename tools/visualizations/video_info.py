"""
Video Information Analyzer
Analyzes video properties using Decord (same library as MMAction2)
"""

import os
from pathlib import Path

try:
    from decord import VideoReader, cpu
    import numpy as np
    
    def analyze_video(video_path):
        """Comprehensive video analysis."""
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            return
        
        print("="*80)
        print("VIDEO INFORMATION REPORT")
        print("="*80)
        
        # Basic file info
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n📁 FILE INFORMATION:")
        print(f"  • File name: {Path(video_path).name}")
        print(f"  • File path: {video_path}")
        print(f"  • File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
        print(f"  • File extension: {Path(video_path).suffix}")
        
        # Open video with Decord (same as MMAction2)
        vr = VideoReader(video_path, ctx=cpu(0))
        
        # Video properties
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Get frame dimensions from first frame
        first_frame = vr[0].asnumpy()
        height, width = first_frame.shape[:2]
        
        # Calculate duration
        duration_sec = total_frames / fps if fps > 0 else 0
        duration_min = int(duration_sec // 60)
        duration_sec_remain = duration_sec % 60
        
        print(f"\n🎬 VIDEO PROPERTIES:")
        print(f"  • Resolution: {width}×{height} pixels")
        print(f"  • Aspect ratio: {width/height:.2f}:1")
        print(f"  • Total frames: {total_frames:,}")
        print(f"  • Frame rate (FPS): {fps:.2f}")
        print(f"  • Duration: {duration_min}m {duration_sec_remain:.2f}s ({duration_sec:.2f} seconds)")
        print(f"  • Estimated bitrate: {(file_size * 8 / duration_sec / 1000):.2f} kbps")
        
        print(f"\n🖼️ FRAME INFORMATION:")
        print(f"  • Frame shape (H×W×C): {first_frame.shape}")
        print(f"  • Color space: RGB (Decord default)")
        print(f"  • Data type: {first_frame.dtype}")
        print(f"  • Value range: [{first_frame.min()}, {first_frame.max()}]")
        print(f"  • Channels: {first_frame.shape[2]}")
        print(f"  • Memory per frame: {first_frame.nbytes / 1024:.2f} KB")
        print(f"  • Total raw memory: {(first_frame.nbytes * total_frames) / (1024**2):.2f} MB")
        
        # Sampling calculations for Video Swin config
        print(f"\n📊 SAMPLING CALCULATIONS (Video Swin Transformer Config):")
        print(f"  Config: clip_len=32, frame_interval=2, num_clips=1")
        print(f"  • Frames to sample: 32")
        print(f"  • Required temporal span: 32 × 2 = 64 frame positions")
        print(f"  • Temporal duration covered: 64/{fps:.2f} = {64/fps:.2f} seconds")
        print(f"  • Percentage of video sampled: {(64/total_frames)*100:.1f}%")
        print(f"  • Non-overlapping clips possible: {total_frames // 64}")
        print(f"  • Max starting position: frame {max(0, total_frames - 64)}")
        
        # Show example frame indices
        if total_frames >= 64:
            # Center sampling (test_mode=True)
            center_start = (total_frames - 64) // 2
            example_indices = list(range(center_start, center_start + 64, 2))
            print(f"\n📍 EXAMPLE: Center sampling (test_mode=True):")
            print(f"  • Start frame: {center_start}")
            print(f"  • End frame: {center_start + 62}")
            print(f"  • First 10 sampled frames: {example_indices[:10]}")
            print(f"  • Last 10 sampled frames: {example_indices[-10:]}")
            print(f"  • Time span: {center_start/fps:.2f}s to {(center_start+62)/fps:.2f}s")
        else:
            print(f"\n⚠️ WARNING: Video has only {total_frames} frames, needs at least 64!")
        
        # Alternative sampling options
        print(f"\n💡 ALTERNATIVE SAMPLING STRATEGIES:")
        
        print(f"  1. Dense (interval=1, clip_len=32):")
        print(f"     • Frames: 32 | Span: 32 | Duration: {32/fps:.2f}s")
        
        print(f"  2. Current config (interval=2, clip_len=32):")
        print(f"     • Frames: 32 | Span: 64 | Duration: {64/fps:.2f}s")
        
        print(f"  3. Sparse (interval=4, clip_len=32):")
        print(f"     • Frames: 32 | Span: 128 | Duration: {128/fps:.2f}s")
        
        stride_for_full = max(1, total_frames // 32)
        print(f"  4. Full video coverage (interval={stride_for_full}, clip_len=32):")
        print(f"     • Frames: 32 | Span: {min(32 * stride_for_full, total_frames)} | Duration: {duration_sec:.2f}s")
        
        # Memory calculations through pipeline
        print(f"\n💾 MEMORY REQUIREMENTS (Pipeline Stages):")
        
        # Original
        original_mem = first_frame.nbytes * 32 / (1024**2)
        print(f"  1. Original (32×{height}×{width}×3): {original_mem:.2f} MB")
        
        # After resize (height=256, keep aspect ratio)
        resize_w = int(width * (256 / height))
        resize_mem = (256 * resize_w * 3 * 32) / (1024**2)
        print(f"  2. After Resize (32×256×{resize_w}×3): {resize_mem:.2f} MB")
        
        # After crop (224×224)
        crop_mem = (224 * 224 * 3 * 32) / (1024**2)
        print(f"  3. After CenterCrop (32×224×224×3): {crop_mem:.2f} MB")
        
        # Batch sizes
        print(f"\n  Batch memory (train batch_size=2): {crop_mem * 2:.2f} MB")
        print(f"  Batch memory (test batch_size=1): {crop_mem:.2f} MB")
        
        # Frame quality analysis
        print(f"\n🎨 FRAME QUALITY INDICATORS:")
        
        # Sample 5 random frames for analysis
        sample_indices = np.linspace(0, total_frames-1, 5, dtype=int)
        brightness_values = []
        contrast_values = []
        
        for idx in sample_indices:
            frame = vr[idx].asnumpy()
            gray = np.mean(frame, axis=2)  # Simple grayscale
            brightness_values.append(gray.mean())
            contrast_values.append(gray.std())
        
        print(f"  • Avg brightness (5 samples): {np.mean(brightness_values):.2f} (0-255)")
        print(f"  • Avg contrast (5 samples): {np.mean(contrast_values):.2f}")
        print(f"  • Brightness range: [{min(brightness_values):.2f}, {max(brightness_values):.2f}]")
        
        print("\n" + "="*80)
        print("✓ Analysis complete!")
        print("="*80)
    
    
    if __name__ == '__main__':
        # Your video path
        video_path = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\VQA_Demo\DryVideos\00002_08_blur_demo.mp4'
        
        analyze_video(video_path)
        
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nPlease install decord:")
    print("  pip install decord")
