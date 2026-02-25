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
        x = input
        return torch.where(x < -1, x, torch.where((x < 1), ((-(torch.abs(x)**3) / 3) + x*(x+2) + (1/3)), 3*x)) / 3.5

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
                 scaling=1e5, act=nn.LeakyReLU):
        """initialise model. inherits from BasicBlock class above.

        Args:
            block (nn.Module): Basic Convolutional block
            filters (int, optional): Number of output filters. Defaults to 16.
            scaling (_type_, optional): _description_. Defaults to 1e5.
        """
        
        super(UNet3d,self).__init__()
        fs = filters
        self.act = act
        self.scaling = scaling
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

    def forward(self,x):
        
        x1 = self.layer1(x)
        #print("x1", x1.shape)
        x  = self.layer2(x1)
        #print("x", x.shape)
        x2 = self.layer3(x)
        #print("x2", x2.shape)
        x  = self.layer4(x2)
        #print("x layer 4", x.shape)
        x  = self.layer5(x)
        #print("x layer 5", x.shape)
        x  = self.act(inplace=True)(self.deconv_batchnorm1((self.deconv1(x))))
       # print("x up 1", x.shape)
        x  = torch.cat((x,x2),dim=1)
        #print("x cat 1", x.shape)
        x  = self.layer6(x)
        #print("x layer 6", x.shape)
        x  = self.act(inplace=True)(self.deconv_batchnorm2((self.deconv2(x))))
        #print("x up 2", x.shape)
        
        x  = torch.cat((x[:, :, :, :], x1[:, :, :, :]),dim=1)
        #print("x cat 2", x.shape)

        x  = self.layer7(x)
        x  = self.deconv4(x)
        
        #x = torch.exp(x)
        #x = x + minx - 1e3

        return x