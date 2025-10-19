import torch
from functools import reduce
from operator import mul
from typing import Sequence

def window_partition(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The input features of shape (B, D, H, W, C).
        window_size (Sequence[int]): The window size, (w_d, w_h, w_w).

    Returns:
        torch.Tensor: The partitioned windows of shape
            (B*num_windows, w_d*w_h*w_w, C).
    """
    B, D, H, W, C = x.shape
    print(f"Input shape: {x.shape}")
    print(f"Window size: {window_size}")
    
    # Step 1: Reshape to separate window dimensions
    x_reshaped = x.view(B, D // window_size[0], window_size[0], 
                       H // window_size[1], window_size[1], 
                       W // window_size[2], window_size[2], C)
    print(f"After view reshape: {x_reshaped.shape}")
    
    # Step 2: Permute to group windows together
    x_permuted = x_reshaped.permute(0, 1, 3, 5, 2, 4, 6, 7)
    print(f"After permute: {x_permuted.shape}")
    
    # Step 3: Final reshape to flatten windows
    windows = x_permuted.contiguous().view(-1, reduce(mul, window_size), C)
    print(f"Final windows shape: {windows.shape}")
    
    return windows

def test_window_partition():
    """Test the window_partition function with different scenarios"""
    
    print("=" * 60)
    print("TEST 1: Simple 2x2x2 windows")
    print("=" * 60)
    
    # Create a simple test tensor
    B, D, H, W, C = 1, 4, 4, 4, 64
    x = torch.randn(B, D, H, W, C)
    window_size = (2, 2, 2)
    
    # Add some identifiable values to track where they go
    x[0, 0, 0, 0, 0] = 100  # Top-left corner of first frame
    x[0, 0, 0, 1, 0] = 200  # Next to it
    x[0, 1, 0, 0, 0] = 300  # Same position in second frame
    
    print(f"Original values at specific positions:")
    print(f"x[0,0,0,0,0] = {x[0,0,0,0,0].item()}")
    print(f"x[0,0,0,1,0] = {x[0,0,0,1,0].item()}")
    print(f"x[0,1,0,0,0] = {x[0,1,0,0,0].item()}")
    print()
    
    windows = window_partition(x, window_size)
    
    # Calculate expected number of windows
    num_windows_d = D // window_size[0]
    num_windows_h = H // window_size[1] 
    num_windows_w = W // window_size[2]
    total_windows = num_windows_d * num_windows_h * num_windows_w
    
    print(f"\nExpected: {B} * {total_windows} = {B * total_windows} windows")
    print(f"Each window: {reduce(mul, window_size)} tokens")
    print(f"Channels: {C}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Larger example")
    print("=" * 60)
    
    # Test with larger dimensions
    B, D, H, W, C = 2, 8, 14, 14, 96
    x_large = torch.randn(B, D, H, W, C)
    window_size_large = (8, 7, 7)
    
    windows_large = window_partition(x_large, window_size_large)
    
    print("\n" + "=" * 60)
    print("TEST 3: Edge case - exact window fit")
    print("=" * 60)
    
    # Test when dimensions are exact multiples of window size
    B, D, H, W, C = 1, 8, 14, 14, 128
    x_exact = torch.randn(B, D, H, W, C)
    window_size_exact = (4, 7, 7)
    
    windows_exact = window_partition(x_exact, window_size_exact)

def test_tracking_values():
    """Test to see how specific values move through the partition"""
    print("\n" + "=" * 60)
    print("TRACKING TEST: Following specific values")
    print("=" * 60)
    
    B, D, H, W, C = 1, 4, 4, 4, 2  # Small for easy tracking
    x = torch.zeros(B, D, H, W, C)
    
    # Set unique values we can track
    counter = 1
    for d in range(D):
        for h in range(H):
            for w in range(W):
                x[0, d, h, w, 0] = counter
                counter += 1
    
    print("Original tensor (showing first channel only):")
    for d in range(D):
        print(f"Frame {d}:")
        print(x[0, d, :, :, 0])
        print()
    
    window_size = (2, 2, 2)
    windows = window_partition(x, window_size)
    
    print(f"Windows shape: {windows.shape}")
    print("First few windows (first channel only):")
    
    tokens_per_window = reduce(mul, window_size)
    for i in range(min(4, windows.shape[0])):  # Show first 4 windows
        print(f"Window {i}:")
        window_vals = windows[i, :, 0].reshape(window_size)
        print(window_vals)
        print()

if __name__ == "__main__":
    test_window_partition()
    test_tracking_values()