# mmaction2/mmaction/core/hooks/performance_hook.py

import time
import torch
import numpy as np
from mmengine.hooks import Hook
from mmaction.registry import HOOKS

@HOOKS.register_module()
class PerformanceAnalysisHook(Hook):
    """Hook for analyzing model performance metrics.
    
    Automatically measures:
    - FLOPs and Parameters (once at start)
    - Inference time (during validation)
    - GPU memory usage (during training/validation)
    
    Args:
        input_shape (tuple): Input shape for FLOPs calculation (C, T, H, W)
        interval (int): Interval for logging GPU memory (default: 50)
        log_flops_once (bool): Whether to log FLOPs only once (default: True)
    """
    
    priority = 'VERY_LOW'
    
    def __init__(self, 
                 input_shape=(3, 32, 224, 224),
                 interval=50,
                 log_flops_once=True):
        self.input_shape = input_shape
        self.interval = interval
        self.log_flops_once = log_flops_once
        self.flops_logged = False
        
        # Storage for inference times
        self.inference_times = []
    
    def before_run(self, runner):
        """Calculate FLOPs and parameters before training starts."""
        if not self.flops_logged:
            self._calculate_flops_params(runner)
            if self.log_flops_once:
                self.flops_logged = True
    
    def _calculate_flops_params(self, runner):
        """Calculate FLOPs and parameters."""
        try:
            from ptflops import get_model_complexity_info
            
            model = runner.model
            if hasattr(model, 'module'):  # DDP/DP wrapper
                model = model.module
            
            runner.logger.info(f"\n{'='*60}")
            runner.logger.info("Model Performance Analysis")
            runner.logger.info(f"{'='*60}")
            
            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            runner.logger.info(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            runner.logger.info(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            
            # Calculate FLOPs for backbone only (common practice)
            if hasattr(model, 'backbone'):
                runner.logger.info("\nCalculating FLOPs for backbone...")
                
                # Create input constructor - use 'x' instead of 'imgs'
                def input_constructor(input_shape):
                    batch = torch.randn(1, *input_shape)
                    return {'x': batch}  # Changed from 'imgs' to 'x'
                
                try:
                    flops, params = get_model_complexity_info(
                        model.backbone,
                        self.input_shape,
                        as_strings=False,
                        print_per_layer_stat=False,
                        input_constructor=input_constructor
                    )
                    
                    runner.logger.info(f"Backbone FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
                    runner.logger.info(f"Backbone Parameters: {params:,} ({params/1e6:.2f}M)")
                    
                except Exception as e:
                    runner.logger.warning(f"Could not calculate backbone FLOPs: {e}")
            
            runner.logger.info(f"{'='*60}\n")
            
        except ImportError:
            runner.logger.warning(
                "ptflops not installed. Install with: pip install ptflops\n"
                "Skipping FLOPs calculation."
            )
        except Exception as e:
            runner.logger.warning(f"Could not calculate FLOPs/Params: {e}")
    def every_n_iters(self, runner, n):
    # Check if the current iteration count is divisible by n
        current_iter = runner.iter  # or runner._inner_iter?
        return current_iter % n == 0

    def before_train_iter(self, runner, batch_idx, data_batch, **kwargs):
        """Log GPU memory before training iteration."""
        if self.every_n_iters(runner, self.interval):
            self._log_gpu_memory(runner, prefix='Train')
    
    def before_val_iter(self, runner, batch_idx, data_batch, **kwargs):
        """Start timing before validation iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.iter_start_time = time.time()
    
    def after_val_iter(self, runner, batch_idx, data_batch, **kwargs):
        """Calculate inference time after validation iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iter_time = time.time() - self.iter_start_time
        self.inference_times.append(iter_time)
    
    def after_val_epoch(self, runner,metrics=None):
        """Log average inference time after validation epoch."""
        if len(self.inference_times) > 0:
            avg_time = np.mean(self.inference_times)
            std_time = np.std(self.inference_times)
            
            runner.logger.info(f"\n{'='*60}")
            runner.logger.info("Inference Performance:")
            runner.logger.info(f"  Average time per sample: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            runner.logger.info(f"  Throughput (FPS): {1.0/avg_time:.2f}")
            runner.logger.info(f"{'='*60}\n")
            
            # Clear for next epoch
            self.inference_times = []
            
        # Log GPU memory after validation
        self._log_gpu_memory(runner, prefix='Validation')
    
    def _log_gpu_memory(self, runner, prefix=''):
        """Log GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        runner.logger.info(
            f"[{prefix}] GPU Memory - "
            f"Allocated: {allocated:.2f} GB, "
            f"Reserved: {reserved:.2f} GB, "
            f"Peak: {max_allocated:.2f} GB"
        )