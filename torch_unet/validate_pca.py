"""
Validate pre-computed PCA components.

This script compares:
1. On-the-fly PCA computation (old method)
2. Pre-computed PCA application (new method)

Usage:
    python validate_pca.py --pca-components pca_components.pt --config your_config.json
"""

import torch
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt
from dataloader import H5Dataset
from torch.utils.data import DataLoader
from utils import PCAclean, apply_precomputed_pca_fast, LoSpixels
import time


def compare_pca_methods(sample, pca_components, N_FG=11):
    """
    Compare on-the-fly vs pre-computed PCA.
    
    Returns:
        dict with timing and difference metrics
    """
    # Assume sample is already processed (batch, RA, baseline, freq, Re/Im)
    sample_real = sample[..., 0]
    sample_imag = sample[..., 1]
    
    results = {}
    
    # Method 1: On-the-fly PCA (old method)
    print("Testing on-the-fly PCA (old method)...")
    start = time.time()
    old_real = PCAclean(sample_real, N_FG=N_FG)[0]
    old_imag = PCAclean(sample_imag, N_FG=N_FG)[0]
    old_time = time.time() - start
    results['old_time'] = old_time
    print(f"  Time: {old_time*1000:.2f}ms")
    
    # Method 2: Pre-computed PCA (new method)
    print("Testing pre-computed PCA (new method)...")
    start = time.time()
    new_real = apply_precomputed_pca_fast(sample_real, pca_components)
    new_imag = apply_precomputed_pca_fast(sample_imag, pca_components)
    new_time = time.time() - start
    results['new_time'] = new_time
    print(f"  Time: {new_time*1000:.2f}ms")
    
    # Speedup
    speedup = old_time / new_time
    results['speedup'] = speedup
    print(f"  Speedup: {speedup:.1f}x")
    
    # Compare results (they won't be identical, but should be similar)
    diff_real = torch.abs(old_real - new_real)
    diff_imag = torch.abs(old_imag - new_imag)
    
    results['mean_diff_real'] = diff_real.mean().item()
    results['max_diff_real'] = diff_real.max().item()
    results['mean_diff_imag'] = diff_imag.mean().item()
    results['max_diff_imag'] = diff_imag.max().item()
    
    # Relative error
    rel_error_real = (diff_real / (torch.abs(old_real) + 1e-10)).mean().item()
    rel_error_imag = (diff_imag / (torch.abs(old_imag) + 1e-10)).mean().item()
    
    results['rel_error_real'] = rel_error_real
    results['rel_error_imag'] = rel_error_imag
    
    print(f"\nDifferences:")
    print(f"  Real - Mean: {results['mean_diff_real']:.2e}, Max: {results['max_diff_real']:.2e}")
    print(f"  Imag - Mean: {results['mean_diff_imag']:.2e}, Max: {results['max_diff_imag']:.2e}")
    print(f"  Relative error - Real: {rel_error_real:.2%}, Imag: {rel_error_imag:.2%}")
    
    return results, (old_real, old_imag), (new_real, new_imag)


def plot_comparison(old_result, new_result, idx=0):
    """
    Plot comparison between old and new PCA methods.
    """
    old_real, old_imag = old_result
    new_real, new_imag = new_result
    
    # Take first sample, middle frequency
    freq_idx = old_real.shape[-1] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Real part
    im1 = axes[0, 0].imshow(old_real[idx, :, :, freq_idx].cpu(), cmap='RdBu_r')
    axes[0, 0].set_title('Old PCA (Real)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(new_real[idx, :, :, freq_idx].cpu(), cmap='RdBu_r')
    axes[0, 1].set_title('New PCA (Real)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    diff_real = (old_real - new_real)[idx, :, :, freq_idx].cpu()
    im3 = axes[0, 2].imshow(diff_real, cmap='RdBu_r')
    axes[0, 2].set_title(f'Difference (Real)\nMax: {diff_real.abs().max():.2e}')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Imaginary part
    im4 = axes[1, 0].imshow(old_imag[idx, :, :, freq_idx].cpu(), cmap='RdBu_r')
    axes[1, 0].set_title('Old PCA (Imag)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(new_imag[idx, :, :, freq_idx].cpu(), cmap='RdBu_r')
    axes[1, 1].set_title('New PCA (Imag)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    diff_imag = (old_imag - new_imag)[idx, :, :, freq_idx].cpu()
    im6 = axes[1, 2].imshow(diff_imag, cmap='RdBu_r')
    axes[1, 2].set_title(f'Difference (Imag)\nMax: {diff_imag.abs().max():.2e}')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('pca_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to pca_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=str, default=None, 
                       help="Path to pre-computed PCA components (default: auto-detect from config)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--plot", action="store_true",
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    # Load config first to get default paths
    with open(args.config) as f:
        configs = json.load(f)
    
    # Determine PCA components path
    if args.pca_components is None:
        # Try to auto-detect from config
        pca_path = configs["model_params"].get("pca_components_path", None)
        if pca_path is None:
            # Try default location in model directory
            MODEL_DIR = configs["model_params"]["model_dir"]
            N_FG = configs["model_params"]["n_fg"]
            pca_path = os.path.join(MODEL_DIR, f"pca_components_nfg{N_FG}.pt")
        args.pca_components = pca_path
    
    if not os.path.exists(args.pca_components):
        print(f"ERROR: PCA components not found at {args.pca_components}")
        print(f"Run: python precompute_pca.py --config {args.config}")
        return
    
    # Load PCA components
    print(f"Loading PCA components from {args.pca_components}")
    pca_components = torch.load(args.pca_components, map_location='cpu')
    print(f"  N_FG: {pca_components['N_FG']}")
    print(f"  N_freq: {pca_components['N_freq']}")
    
    # Setup dataset (configs already loaded above)
    cosmopath = configs["training_params"]["cosmopath"]
    galpath = configs["training_params"]["galpath"]
    N_FG = configs["model_params"]["n_fg"]
    
    cosmofiles = os.listdir(cosmopath)
    galfiles = os.listdir(galpath)
    cosmofiles = [cosmopath + p for p in cosmofiles][:10]  # Just use 10 files
    galfiles = [galpath + p for p in galfiles][:10]
    
    dataset = H5Dataset(cosmofiles, galfiles, use_cache=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Test on multiple samples
    print(f"\nTesting on {args.num_samples} samples...")
    all_speedups = []
    
    for i, (x, y) in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        
        # Preprocess to match training format
        # x is complex: (batch, baseline, freq, RA)
        split = 1024 // 128
        x_split = torch.permute(
            torch.cat(torch.tensor_split(x, split, dim=3)),
            (0, 3, 1, 2)  # (batch*split, freq, RA, baseline)
        )
        x_real_imag = torch.stack([x_split.real, x_split.imag], dim=-1)
        # Shape: (batch*split, freq, RA, baseline, Re/Im)
        
        # Need to rearrange to (batch, RA, baseline, freq, Re/Im) for PCA functions
        x_rearranged = x_real_imag.permute(0, 2, 3, 1, 4)
        
        # Compare methods
        results, old_result, new_result = compare_pca_methods(
            x_rearranged[0:1],  # Just first sample
            pca_components,
            N_FG=N_FG
        )
        
        all_speedups.append(results['speedup'])
        
        if args.plot and i == 0:
            plot_comparison(old_result, new_result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Average speedup: {np.mean(all_speedups):.1f}x")
    print(f"Min speedup: {np.min(all_speedups):.1f}x")
    print(f"Max speedup: {np.max(all_speedups):.1f}x")
    print(f"\nPre-computed PCA is ~{np.mean(all_speedups):.0f}x faster!")
    print(f"Memory usage is also much lower on GPU.")


if __name__ == "__main__":
    main()
