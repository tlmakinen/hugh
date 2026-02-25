import torch
import torch.nn as nn
import numpy as np


class FastPCALayer(nn.Module):
    """
    Memory-efficient PCA layer using pre-computed components.
    
    Much faster and more memory-efficient than computing PCA on-the-fly.
    Operates entirely on GPU with minimal memory footprint.
    """
    def __init__(self, pca_components_path, device='cuda'):
        super().__init__()
        
        # Load pre-computed components
        pca_components = torch.load(pca_components_path, map_location=device)
        
        # Register as buffers (not parameters, but moved with model)
        self.register_buffer('fg_components', pca_components['fg_components'])
        self.register_buffer('mean_per_freq', pca_components['mean_per_freq'])
        
        self.N_FG = pca_components['N_FG']
        self.N_freq = pca_components['N_freq']
        
    def forward(self, x):
        """
        Apply PCA foreground removal.
        
        Input: x with shape (..., freq) where last dimension is frequency
        Output: cleaned x with same shape
        """
        original_shape = x.shape
        
        # Flatten all dims except frequency
        x_flat = x.reshape(-1, self.N_freq)
        
        # Mean center (subtract per-frequency mean)
        x_centered = x_flat - self.mean_per_freq
        
        # Project to foreground space
        fg_coeffs = torch.matmul(x_centered, self.fg_components)  # (..., N_FG)
        
        # Reconstruct foreground
        fg_recon = torch.matmul(fg_coeffs, self.fg_components.T)  # (..., N_freq)
        
        # Remove foreground
        x_cleaned = x_centered - fg_recon
        
        # Restore original shape
        return x_cleaned.reshape(original_shape)
    
    def extra_repr(self):
        return f'N_FG={self.N_FG}, N_freq={self.N_freq}'


def apply_precomputed_pca_fast(data, pca_components):
    """
    Fast GPU-friendly PCA application using pre-computed components.
    
    Args:
        data: Input tensor with last dimension as frequency (..., N_freq)
        pca_components: Dict with 'fg_components' and 'mean_per_freq'
    
    Returns:
        Cleaned data with same shape as input
    """
    original_shape = data.shape
    N_freq = pca_components['N_freq']
    
    # Move components to same device as data
    fg_components = pca_components['fg_components'].to(data.device)
    mean_per_freq = pca_components['mean_per_freq'].to(data.device)
    
    # Flatten all dims except frequency
    data_flat = data.reshape(-1, N_freq)
    
    # Mean center
    data_centered = data_flat - mean_per_freq
    
    # Project to foreground space and remove
    fg_coeffs = torch.matmul(data_centered, fg_components)  # (..., N_FG)
    fg_recon = torch.matmul(fg_coeffs, fg_components.T)  # (..., N_freq)
    cleaned = data_centered - fg_recon
    
    # Restore shape
    return cleaned.reshape(original_shape)


def LoSpixels(image_cube, mean_center=True):
    '''
    Convert image data cube from [None, Nx, Ny, Nz=freq] to [Nz=freq, Npix] format.

    INPUTS:
    image_cube: input data cube in image form, with shape [Nx, Ny, Nz] where Nz
    mean_center: if True, do mean centering for each frequency slice after 
    '''
        
    # # input (batch, RA, baseline, freq) = (None, 128, 48, 128) freq LAST
    axes_original = image_cube.shape
    image_cube = torch.reshape(image_cube, (axes_original[0]*axes_original[1], axes_original[2], axes_original[3]))

    # swap axes to [Nz, Nx, Ny] for converting to visibilities:
    image_cube = torch.swapaxes(image_cube,0,1)
    image_cube = torch.swapaxes(image_cube,0,2)

    # convert to LoS pixels format [Nz, Npix]
    axes = image_cube.shape
    image_LoSpixels = torch.reshape(image_cube,(axes[0], axes[1]*axes[2]))
    
    # mean center the data:
    if mean_center==True:
        nz = image_LoSpixels.shape[0]
        for i in range(nz):
            image_LoSpixels[i] = image_LoSpixels[i] - torch.mean(image_LoSpixels[i])
    
    return image_LoSpixels

def PCAclean(Input, N_FG=4):
    '''
    Takes input in [Nx,Ny,Nz] data cube form where Nz is number of redshift 
    (frequency) bins. N_FG is number of eigenmodes for PCA to remove
    '''
    
    # Collapse data cube to [Npix,Nz] structure:
    axes = Input.shape
    Input = LoSpixels(Input, mean_center=True)
    Nz,Npix = Input.shape
    
    # Obtain frequency covariance matrix for input data
    C = torch.cov(Input)
    eigenval, eigenvec = torch.linalg.eigh(C)
    eignumb = torch.linspace(1,len(eigenval),len(eigenval))
    eigenval = torch.flip(eigenval, dims=[0]) #Put largest eigenvals first
    A = eigenvec[:,Nz-N_FG:] # Mixing matrix
    S = torch.matmul(A.T,Input) # might need to change
    
    # PCA Component Maps
    Recon_FG = torch.matmul(A,S)
    Residual = Input - Recon_FG
    Residual = torch.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]
    Residual = torch.reshape(Residual,axes)
    
    return Residual,eignumb,eigenval