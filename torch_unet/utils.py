import torch
import torch.nn as nn
import numpy as np

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