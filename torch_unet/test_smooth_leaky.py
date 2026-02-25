"""
Test script to validate smooth_leaky optimization.

Usage:
    python test_smooth_leaky.py
"""

import torch
import time
from nets import smooth_leaky


def old_smooth_leaky_function(x):
    """Original implementation for comparison."""
    return torch.where(
        x < -1, x, 
        torch.where(
            (x < 1), 
            ((-(torch.abs(x)**3) / 3) + x*(x+2) + (1/3)), 
            3*x
        )
    ) / 3.5


def test_correctness():
    """Test that new implementation matches old implementation."""
    print("Testing correctness...")
    
    # Test various input ranges
    test_cases = [
        torch.linspace(-3, 3, 1000),
        torch.randn(1000),
        torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
    ]
    
    model = smooth_leaky(inplace=False)
    
    for i, x in enumerate(test_cases):
        old_out = old_smooth_leaky_function(x.clone())
        new_out = model(x.clone())
        
        diff = torch.abs(new_out - old_out)
        max_diff = diff.max().item()
        
        print(f"  Test case {i+1}: max difference = {max_diff:.2e}")
        
        if max_diff > 1e-6:
            print(f"    WARNING: Large difference detected!")
        else:
            print(f"    ✓ PASS")
    
    print()


def test_inplace():
    """Test that inplace operation actually modifies input."""
    print("Testing in-place operation...")
    
    x = torch.randn(100)
    x_id = id(x)
    
    # Non-inplace
    model_regular = smooth_leaky(inplace=False)
    out_regular = model_regular(x.clone())
    
    # In-place
    model_inplace = smooth_leaky(inplace=True)
    x_copy = x.clone()
    x_copy_id = id(x_copy)
    out_inplace = model_inplace(x_copy)
    out_inplace_id = id(out_inplace)
    
    print(f"  Non-inplace: input id={id(x)}, output id={id(out_regular)}, same={id(out_regular)==id(x)}")
    print(f"  In-place: input id={x_copy_id}, output id={out_inplace_id}, same={out_inplace_id==x_copy_id}")
    
    if out_inplace_id == x_copy_id:
        print("  ✓ PASS: In-place operation confirmed")
    else:
        print("  ✗ FAIL: In-place operation not working")
    
    print()


def test_memory():
    """Compare memory usage."""
    print("Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("  Skipping (CUDA not available)")
        print()
        return
    
    device = torch.device('cuda')
    
    # Large tensor to see memory difference
    x = torch.randn(4, 16, 48, 128, 128, device=device)
    
    # Old implementation (simulated)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = old_smooth_leaky_function(x)
    old_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # New implementation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = smooth_leaky(inplace=False).to(device)
    _ = model(x)
    new_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # In-place
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_inplace = smooth_leaky(inplace=True).to(device)
    x_copy = x.clone()
    _ = model_inplace(x_copy)
    inplace_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"  Old implementation: {old_mem:.3f} GB")
    print(f"  New implementation: {new_mem:.3f} GB")
    print(f"  New (inplace):      {inplace_mem:.3f} GB")
    print(f"  Memory reduction:   {(1 - new_mem/old_mem)*100:.1f}%")
    print(f"  Inplace reduction:  {(1 - inplace_mem/old_mem)*100:.1f}%")
    
    print()


def test_speed():
    """Compare execution speed."""
    print("Testing speed...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(4, 16, 48, 128, 128, device=device)
    
    # Warmup
    _ = old_smooth_leaky_function(x.clone())
    model = smooth_leaky(inplace=False).to(device)
    _ = model(x.clone())
    
    # Benchmark old
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = old_smooth_leaky_function(x.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    old_time = time.time() - start
    
    # Benchmark new
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(x.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    new_time = time.time() - start
    
    # Benchmark inplace
    model_inplace = smooth_leaky(inplace=True).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x_copy = x.clone()
        _ = model_inplace(x_copy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inplace_time = time.time() - start
    
    print(f"  Old implementation: {old_time*1000:.2f} ms")
    print(f"  New implementation: {new_time*1000:.2f} ms")
    print(f"  New (inplace):      {inplace_time*1000:.2f} ms")
    print(f"  Speedup:            {old_time/new_time:.2f}x")
    print(f"  Speedup (inplace):  {old_time/inplace_time:.2f}x")
    
    print()


def main():
    print("="*60)
    print("smooth_leaky Activation Function Optimization Tests")
    print("="*60)
    print()
    
    test_correctness()
    test_inplace()
    test_memory()
    test_speed()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
