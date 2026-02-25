"""
Pre-compute PCA components for foreground removal.

This script computes PCA components from a subset of training data
and saves them for efficient inference during training.

Usage:
    python precompute_pca.py --config path/to/config.json --output pca_components.pt
"""

import torch
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
from dataloader import H5Dataset
from torch.utils.data import DataLoader
from utils import LoSpixels, apply_precomputed_pca_fast


def compute_global_pca_components(dataloader, N_FG=11, num_samples=100, device='cpu'):
    """
    Compute PCA components from a representative sample of data.
    
    Args:
        dataloader: DataLoader for the dataset
        N_FG: Number of foreground components to remove
        num_samples: Number of samples to use for computing PCA
        device: Device to compute on ('cpu' recommended for large covariance matrices)
    
    Returns:
        dict: Contains 'projection_matrix', 'mean', 'N_FG', and metadata
    """
    print(f"Computing PCA components from {num_samples} samples...")
    
    # Collect data for PCA computation
    all_data = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            # Use the contaminated signal (x) to learn foreground structure
            # x is complex, shape: (batch, baseline, freq, RA)
            # We need to process each sample separately and collect
            
            for sample_idx in range(x.shape[0]):
                sample = x[sample_idx]  # (baseline, freq, RA)
                
                # Split and reshape like in preprocessing
                # Assuming gal_mask already applied in dataloader
                # Convert to (freq, RA, baseline) format for PCA
                sample_real = sample.real
                sample_imag = sample.imag
                
                # Process real and imaginary separately
                for data in [sample_real, sample_imag]:
                    # Reshape to (RA*baseline, freq)
                    # Original: (baseline, freq, RA)
                    data_permuted = data.permute(2, 0, 1)  # (RA, baseline, freq)
                    data_flat = data_permuted.reshape(-1, data.shape[1])  # (RA*baseline, freq)
                    all_data.append(data_flat.to(device))
    
    # Stack all samples: shape (N_samples * N_pixels, N_freq)
    all_data = torch.cat(all_data, dim=0)
    print(f"Collected data shape: {all_data.shape}")
    
    # Compute covariance matrix
    print("Computing covariance matrix...")
    # Mean center each frequency
    mean_per_freq = all_data.mean(dim=0, keepdim=True)
    data_centered = all_data - mean_per_freq
    
    # Compute covariance: (N_freq, N_freq)
    cov_matrix = torch.cov(data_centered.T)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Eigendecomposition
    print("Computing eigendecomposition...")
    eigenval, eigenvec = torch.linalg.eigh(cov_matrix)
    
    # Sort by largest eigenvalues (foreground modes)
    idx = torch.argsort(eigenval, descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]
    
    # Keep all EXCEPT the top N_FG foreground modes
    # We'll project out the foreground, keeping the signal
    fg_components = eigenvec[:, :N_FG]  # Shape: (N_freq, N_FG)
    signal_components = eigenvec[:, N_FG:]  # Shape: (N_freq, N_freq - N_FG)
    
    print(f"Foreground components shape: {fg_components.shape}")
    print(f"Signal components shape: {signal_components.shape}")
    print(f"Top 10 eigenvalues: {eigenval[:10].cpu().numpy()}")
    
    # Return components for projection
    pca_components = {
        'fg_components': fg_components.cpu(),  # For projecting out foregrounds
        'signal_components': signal_components.cpu(),  # For projecting to signal space
        'eigenvalues': eigenval.cpu(),
        'mean_per_freq': mean_per_freq.cpu().squeeze(),
        'N_FG': N_FG,
        'N_freq': cov_matrix.shape[0],
    }
    
    return pca_components


def apply_precomputed_pca(data, pca_components, use_signal_projection=False):
    """
    Apply pre-computed PCA to remove foregrounds.
    
    Args:
        data: Input tensor, shape (..., N_freq) where last dim is frequency
        pca_components: Dict from compute_global_pca_components
        use_signal_projection: If True, project to signal subspace instead of removing FG
    
    Returns:
        Cleaned data with same shape as input
    """
    original_shape = data.shape
    N_freq = pca_components['N_freq']
    
    # Reshape to (..., N_freq)
    data_flat = data.reshape(-1, N_freq)
    
    # Mean center
    data_centered = data_flat - pca_components['mean_per_freq'].to(data.device)
    
    if use_signal_projection:
        # Project to signal subspace
        signal_proj = pca_components['signal_components'].to(data.device)
        # Project to signal space and back
        coeffs = torch.matmul(data_centered, signal_proj)  # (..., N_freq - N_FG)
        cleaned = torch.matmul(coeffs, signal_proj.T)  # (..., N_freq)
    else:
        # Remove foreground components (default)
        fg_proj = pca_components['fg_components'].to(data.device)
        # Project to FG space
        fg_coeffs = torch.matmul(data_centered, fg_proj)  # (..., N_FG)
        # Reconstruct and subtract
        fg_recon = torch.matmul(fg_coeffs, fg_proj.T)  # (..., N_freq)
        cleaned = data_centered - fg_recon
    
    # Restore original shape
    cleaned = cleaned.reshape(original_shape)
    
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output file path (default: <model_dir>/pca_components_nfg<N_FG>.pt)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for PCA")
    parser.add_argument("--device", type=str, default="cpu", help="Device for computation")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        configs = json.load(f)
    
    # Setup paths
    cosmopath = configs["training_params"]["cosmopath"]
    galpath = configs["training_params"]["galpath"]
    N_FG = configs["model_params"]["n_fg"]
    MODEL_DIR = configs["model_params"]["model_dir"]
    MODEL_NAME = configs["model_params"]["model_name"]
    
    # Default output path: save to model_dir/model_name/ directory
    if args.output is None:
        # Construct full model path (same as train2.py does)
        full_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        
        # Create model directory if it doesn't exist
        if not os.path.exists(full_model_path):
            os.makedirs(full_model_path)
            print(f"Created model directory: {full_model_path}")
        
        # Generate filename based on N_FG parameter
        output_path = os.path.join(full_model_path, f"pca_components_nfg{N_FG}.pt")
        print(f"Using default output path: {output_path}")
    else:
        output_path = args.output
        print(f"Using specified output path: {output_path}")
    
    cosmofiles = os.listdir(cosmopath)
    galfiles = os.listdir(galpath)
    cosmofiles = [cosmopath + p for p in cosmofiles]
    galfiles = [galpath + p for p in galfiles]
    
    # Use training files
    mask = np.random.rand(len(cosmofiles)) < 0.9
    train_cosmo_files = list(np.array(cosmofiles)[mask])[:configs["training_params"]["num_train"]]
    galmask = np.random.rand(len(galfiles)) < 0.9
    train_gal_files = list(np.array(galfiles)[galmask])
    
    # Create dataset and loader
    print("Creating dataset...")
    dataset = H5Dataset(train_cosmo_files, train_gal_files, use_cache=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Compute PCA components
    pca_components = compute_global_pca_components(
        dataloader, 
        N_FG=N_FG, 
        num_samples=args.num_samples,
        device=args.device
    )
    
    # Save
    torch.save(pca_components, output_path)
    print(f"\nSaved PCA components to {output_path}")
    
    # Print statistics
    print("\nPCA Statistics:")
    print(f"  Number of FG components: {pca_components['N_FG']}")
    print(f"  Frequency bins: {pca_components['N_freq']}")
    print(f"  Variance explained by FG: {pca_components['eigenvalues'][:N_FG].sum() / pca_components['eigenvalues'].sum():.2%}")


if __name__ == "__main__":
    main()
