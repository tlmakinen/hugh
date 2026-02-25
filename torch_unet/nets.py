import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


from utils import *


def transform_inputs(inputs, scaling=1e5):
    inputs *= scaling
    #inputs += 10.0
    #inputs += 1e-3
    return torch.arcsinh(inputs)

def inv_transform_inputs(inputs, scaling=1e5):
    inputs = torch.sinh(inputs)
    #inputs -= 7.0
    #inputs -= 1e-3
    inputs /= scaling
    return inputs


class smooth_leaky(nn.Module):
    r"""Smooth Leaky rectified linear unit activation function.

    Computes the element-wise function:

    .. math::
    \mathrm{smooth\_leaky}(x) = \begin{cases}
        x, & x \leq -1\\
        - |x|^3/3, & -1 \leq x < 1\\
        3x & x > 1
    \end{cases}

    Args:
    x : input Tensor

    Examples::

        >>> m = smooth_leaky()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Memory-efficient implementation using masked tensor operations
        instead of nested torch.where calls. Reduces intermediate tensor creation.
        """
        x = input
        
        # Pre-allocate output tensor (reuse input if inplace)
        if self.inplace:
            out = x
        else:
            out = torch.empty_like(x)
        
        # Create masks for each region
        mask_low = x < -1     # Region 1: x < -1
        mask_high = x >= 1    # Region 3: x >= 1
        # Region 2 is everything else: -1 <= x < 1
        
        # Apply piecewise function efficiently
        # Region 1: out = x (copy input values)
        out.copy_(x)
        
        # Region 2: out = (-(|x|^3) / 3) + x*(x+2) + (1/3)
        # Only compute for middle region to save operations
        mask_mid = ~(mask_low | mask_high)
        if mask_mid.any():
            x_mid = x[mask_mid]
            x_sq = x_mid * x_mid
            # Compute: -(|x|^3)/3 + x*(x+2) + 1/3 = -(|x|^3)/3 + x^2 + 2x + 1/3
            out[mask_mid] = -(torch.abs(x_mid) * x_sq) / 3.0 + x_sq + 2.0 * x_mid + (1.0/3.0)
        
        # Region 3: out = 3*x
        if mask_high.any():
            out[mask_high] = 3.0 * x[mask_high]
        
        # Scale by 3.5 (in-place)
        out.mul_(1.0 / 3.5)
        
        return out

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


def PCALayer(x, N_FG=7, pca_components=None):
    '''
    Takes input in [Nx,Ny,Nz,(Re/Im)] data cube form where Nz is number of redshift 
    (frequency) bins. N_FG is number of eigenmodes for PCA to remove.
    Cleans foregrounds for Real and Imaginary components separately.
    
    Args:
        x: Input tensor (..., freq, Re/Im)
        N_FG: Number of foreground components (only used if pca_components is None)
        pca_components: Pre-computed PCA components dict (RECOMMENDED for efficiency)
    '''
    #print("input", x.shape) # input (batch, RA, baseline, freq) = (None, 128, 48, 128, re/im)freq LAST

    if pca_components is not None:
        # Use pre-computed PCA (MUCH faster and GPU-friendly)
        from utils import apply_precomputed_pca_fast
        xreal = apply_precomputed_pca_fast(x[..., 0], pca_components)
        ximag = apply_precomputed_pca_fast(x[..., 1], pca_components)
    else:
        # Fallback to on-the-fly PCA (slow, for backward compatibility)
        xreal = PCAclean(x[..., 0], N_FG=N_FG)[0] # output is (None, RA, baseline, freq)
        ximag = PCAclean(x[..., 1], N_FG=N_FG)[0]

    # then transpose to output to UNet: (None, re/im, baseline, RA, freq)
    return torch.permute(torch.stack((xreal, ximag), dim=-1), (0, 4, 2, 1, 3)) 





def conv3x3(inplane,outplane, stride=1,padding="same"):
    return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True,
                    padding_mode="circular")

class BasicBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1, padding=0,
                 act=nn.LeakyReLU):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplane,outplane,padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.act = act(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        return out


# input shape: (128,128,48)
# (0,0,0)
padding = (1,1,1)

class UNet3d(nn.Module):
    """3D UNet model. Heavily inpired by He et al https://github.com/siyucosmo/ML-Recon
    """
    

    def __init__(self, block, filters=16,
                 scaling=1e5, act=nn.LeakyReLU, use_checkpoint=False):
        """initialise model. inherits from BasicBlock class above.

        Args:
            block (nn.Module): Basic Convolutional block
            filters (int, optional): Number of output filters. Defaults to 16.
            scaling (_type_, optional): _description_. Defaults to 1e5.
            use_checkpoint (bool): Enable gradient checkpointing for memory savings
        """
        
        super(UNet3d,self).__init__()
        fs = filters
        self.act = act
        self.scaling = scaling
        self.use_checkpoint = use_checkpoint
        self.layer1 = self._make_layer(block, 2, fs, blocks=2,stride=1,padding=(1,1,1))
        self.layer2 = self._make_layer(block,fs,fs*2, blocks=1,stride=2,padding=(1,1,1)) # (64,64,24)
        self.layer3 = self._make_layer(block,fs*2,fs*2,blocks=2,stride=1,padding=(1,1,1))
        self.layer4 = self._make_layer(block,fs*2,fs*4,blocks=1,stride=2,padding=(1,1,1)) # (32,32,12)
        self.layer5 = self._make_layer(block,fs*4,fs*4,blocks=2,stride=1,padding=(1,1,1))
        self.deconv1 = nn.ConvTranspose3d(fs*4,fs*2,3,stride=2,padding=(1,1,1), output_padding=(1,1,1))
        self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = fs*2,momentum=0.1) 
        self.layer6 = self._make_layer(block,fs*4,fs*2,blocks=2,stride=1,padding=(1,1,1)) # (64,64,24)
        self.deconv2 = nn.ConvTranspose3d(fs*2,fs,3,stride=2,padding=(1,1,1), output_padding=(1,1,1))
        self.deconv_batchnorm2 = nn.BatchNorm3d(num_features=fs,momentum=0.1)
        self.layer7 = self._make_layer(block,fs*2,fs,blocks=2,stride=1,padding=(1,1,1))
        self.deconv4 = nn.Conv3d(fs,2,kernel_size=3,stride=1,padding=(1,1,1),bias=True,
                    padding_mode="circular") #nn.ConvTranspose3d(64,2,1,stride=1,padding=(1,1,1), output_padding=(1,1,1))


    def _make_layer(self,block,inplanes,outplanes,blocks,stride=1,padding=0):
        layers = []
        for i in range(0,blocks):
            layers.append(block(inplanes,outplanes,stride=stride,padding=padding,act=self.act))
            inplanes = outplanes
        return nn.Sequential(*layers)
    
    def _pca_layer(self, N_FG):
        fn = lambda x: PCALayer(x=x, N_FG=N_FG)
        return fn

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        
        # Encoder path (with optional checkpointing for memory savings)
        if self.use_checkpoint and self.training:
            x1 = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x1, use_reentrant=False)
            x2 = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x2, use_reentrant=False)
            x = checkpoint(self.layer5, x, use_reentrant=False)
        else:
            x1 = self.layer1(x)
            x = self.layer2(x1)
            x2 = self.layer3(x)
            x = self.layer4(x2)
            x = self.layer5(x)
        
        # Decoder path (don't checkpoint decoder - less memory intensive)
        x = self.act(inplace=True)(self.deconv_batchnorm1(self.deconv1(x)))
        x = torch.cat((x, x2), dim=1)
        
        if self.use_checkpoint and self.training:
            x = checkpoint(self.layer6, x, use_reentrant=False)
        else:
            x = self.layer6(x)
        
        x = self.act(inplace=True)(self.deconv_batchnorm2(self.deconv2(x)))
        x = torch.cat((x[:, :, :, :], x1[:, :, :, :]), dim=1)
        
        x = self.layer7(x)
        x = self.deconv4(x)
        
        return x