#!/usr/bin/env python
# tools/analysis/analyze_model.py

import argparse
import time
import torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS

def build_model(cfg):
    return MODELS.build(cfg)


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops(model, input_shape=(1, 3, 32, 224, 224)):
    """Calculate FLOPs using ptflops."""
    try:
        from ptflops import get_model_complexity_info
        
        def input_constructor(input_shape):
            return {'imgs': torch.randn(*input_shape)}
        
        flops, params = get_model_complexity_info(
            model,
            input_shape[1:],  # Remove batch dimension
            as_strings=False,
            print_per_layer_stat=False,
            input_constructor=input_constructor
        )
        
        return flops, params
    
    except ImportError:
        print("Warning: ptflops not installed. Install with: pip install ptflops")
        return None, None
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        return None, None

def measure_inference_time(model, input_shape=(1, 3, 32, 224, 224), 
                          num_warmup=10, num_runs=100, device='cuda'):
    """Measure inference time."""
    model.eval()
    
    # Create dummy input
    dummy_input = {'imgs': torch.randn(*input_shape).to(device)}
    
    print(f"Warming up for {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(**dummy_input, mode='tensor')
    
    print(f"Measuring inference time for {num_runs} iterations...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(**dummy_input, mode='tensor')
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times)
    }

def measure_gpu_memory(model, input_shape=(1, 3, 32, 224, 224), device='cuda'):
    """Measure GPU memory usage."""
    if not torch.cuda.is_available():
        return None
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model.eval()
    dummy_input = {'imgs': torch.randn(*input_shape).to(device)}
    
    # Measure memory during inference
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(**dummy_input, mode='tensor')
    
    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'peak': max_allocated
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze model performance')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file path')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-frames', type=int, default=32, help='Number of frames')
    parser.add_argument('--height', type=int, default=224, help='Frame height')
    parser.add_argument('--width', type=int, default=224, help='Frame width')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of inference runs')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Build model
    print("Building model...")
    model = build_model(cfg.model)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    model.to(args.device)
    model.eval()
    
    input_shape = (args.batch_size, 3, args.num_frames, args.height, args.width)
    print(f"Input shape: {input_shape}")
    print(f"Device: {args.device}\n")
    
    # ========================================================================
    # 1. COUNT PARAMETERS
    # ========================================================================
    print(f"{'='*70}")
    print("1. PARAMETERS")
    print(f"{'='*70}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Breakdown by component
    if hasattr(model, 'backbone'):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"Backbone Parameters:   {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    
    if hasattr(model, 'cls_head'):
        head_params = sum(p.numel() for p in model.cls_head.parameters())
        print(f"Head Parameters:       {head_params:,} ({head_params/1e6:.2f}M)")
    
    # ========================================================================
    # 2. CALCULATE FLOPs
    # ========================================================================
    print(f"\n{'='*70}")
    print("2. FLOPs")
    print(f"{'='*70}")
    
    flops, params_flops = calculate_flops(model, input_shape)
    if flops is not None:
        print(f"Total FLOPs:           {flops:,} ({flops/1e9:.2f} GFLOPs)")
        print(f"FLOPs per frame:       {flops/(args.num_frames):,} ({flops/(args.num_frames*1e9):.2f} GFLOPs)")
    else:
        print("FLOPs calculation not available")
    
    # ========================================================================
    # 3. INFERENCE TIME
    # ========================================================================
    print(f"\n{'='*70}")
    print("3. INFERENCE TIME")
    print(f"{'='*70}")
    
    time_stats = measure_inference_time(
        model, input_shape, 
        num_runs=args.num_runs, 
        device=args.device
    )
    
    print(f"Mean:                  {time_stats['mean']*1000:.2f} ms")
    print(f"Std:                   {time_stats['std']*1000:.2f} ms")
    print(f"Min:                   {time_stats['min']*1000:.2f} ms")
    print(f"Max:                   {time_stats['max']*1000:.2f} ms")
    print(f"Median:                {time_stats['median']*1000:.2f} ms")
    print(f"Throughput (FPS):      {1.0/time_stats['mean']:.2f}")
    print(f"Latency per frame:     {time_stats['mean']*1000/args.num_frames:.2f} ms")
    
    # ========================================================================
    # 4. GPU MEMORY
    # ========================================================================
    if args.device == 'cuda':
        print(f"\n{'='*70}")
        print("4. GPU MEMORY USAGE")
        print(f"{'='*70}")
        
        memory_stats = measure_gpu_memory(model, input_shape, device=args.device)
        if memory_stats:
            print(f"Allocated:             {memory_stats['allocated']:.2f} GB")
            print(f"Reserved:              {memory_stats['reserved']:.2f} GB")
            print(f"Peak:                  {memory_stats['peak']:.2f} GB")
            print(f"Memory per sample:     {memory_stats['peak']/args.batch_size:.2f} GB")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model:                 {cfg.model.type}")
    print(f"Backbone:              {cfg.model.backbone.type}")
    print(f"Parameters:            {total_params/1e6:.2f}M")
    if flops:
        print(f"FLOPs:                 {flops/1e9:.2f} GFLOPs")
    print(f"Inference Time:        {time_stats['mean']*1000:.2f} ± {time_stats['std']*1000:.2f} ms")
    print(f"Throughput:            {1.0/time_stats['mean']:.2f} FPS")
    if args.device == 'cuda' and memory_stats:
        print(f"GPU Memory (Peak):     {memory_stats['peak']:.2f} GB")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

