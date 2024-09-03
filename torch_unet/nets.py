import torch
import torch.nn as nn
import numpy as np


from utils import *

def PCALayer(x, N_FG=7):
    '''
    Takes input in [Nx,Ny,Nz,(Re/Im)] data cube form where Nz is number of redshift 
    (frequency) bins. N_FG is number of eigenmodes for PCA to remove.
    Cleans foregrounds for Real and Imaginary components separately.
    '''
    #print("input", x.shape) # input (batch, RA, baseline, freq) = (None, 128, 48, 128, re/im)freq LAST

    xreal = PCAclean(x[..., 0], N_FG=N_FG)[0] # output is (None, RA, baseline, freq)
    ximag = PCAclean(x[..., 1], N_FG=N_FG)[0]

    # then transpose to output to UNet: (None, re/im, baseline, RA, freq)
    return torch.permute(torch.stack((xreal, ximag), dim=-1), (0, 4, 2, 1, 3)) 



def preprocess_data(x,y, 
                    N_FG,
                    device,
                    noiseamp=None, 
                    split=1024 // 128
                     ):
        """Helper function for passing data to CPU for PCA preprocessing and then back to the GPU
        for UNet training.

        Args:
            x (torch.Tensor): batch of input foreground contaminated tiles of shape (Re/Im, baseline, freq, RA) (2, 48, 128, 1024)
            y (torch.Tensor): target clean cosmological signal of shape (Re/Im, baseline, freq, RA) (2, 48, 128, 1024)
            N_FG (int): number of PCA foreground components to remove
            device (torch.device): GPU device addres
            noiseamp (float, optional): random white noise amplitude to add to training
            split (_type_, optional): number of tiles to split RA direction into. Defaults to 1024//128.

        Returns:
            tuple(torch.Tensor): (x,y) torch.Tensor pair, each of shape (batch*split, 48, 128, 128)
                                 set by default onto the specified device.
        """
    
        # split ordering (batch, baseline, freq, ra) = (batch*split, 48, 128, 128)
        # then transpose to (batch*split, freq, ra, baseline)
        x = torch.permute(
            torch.cat(torch.tensor_split(x, split, dim=3)),
            (0, 3, 1, 2)
        )
        y = torch.permute(
                torch.cat(torch.tensor_split(y, split, dim=3)),
                (0, 3, 1, 2)
        )
        # then finally get the real and im parts as channels
        # shape: (batch*split, freq, ra, baseline, Re/Im)
        x = torch.stack([x.real, x.imag], dim=-1)
        y = torch.stack([y.real, y.imag], dim=-1)
        
        
        # add white noise to the signal
        if noiseamp is not None:
            x += torch.normal(mean=0.0, std=torch.ones(x.shape)*noiseamp) #.to(device)
        
        # pass x to the pca
        x = PCALayer(x, N_FG=N_FG)
        
        # get y into same shape as model outputs
        y = torch.permute(y, (0, 4, 2, 1, 3))
        
        return x.to(device),y.to(device)


def conv3x3(inplane,outplane, stride=1,padding="same"):
    return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True,
                    padding_mode="circular")

class BasicBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1, padding=0,
                 act=nn.SiLU):
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
                 scaling=1e5, act=nn.SiLU):
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

        # self.layer1 = self._make_layer(block, 2, 64, blocks=2,stride=1,padding=(1,1,1))
        # self.layer2 = self._make_layer(block,64,128, blocks=1,stride=2,padding=(1,1,1)) # (64,64,24)
        # self.layer3 = self._make_layer(block,128,128,blocks=2,stride=1,padding=(1,1,1))
        # self.layer4 = self._make_layer(block,128,256,blocks=1,stride=2,padding=(1,1,1)) # (32,32,12)
        # self.layer5 = self._make_layer(block,256,256,blocks=2,stride=1,padding=(1,1,1))
        # self.deconv1 = nn.ConvTranspose3d(256,128,3,stride=2,padding=(1,1,1), output_padding=(1,1,1))
        # self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = 128,momentum=0.1) 
        # self.layer6 = self._make_layer(block,256,128,blocks=2,stride=1,padding=(1,1,1)) # (64,64,24)
        # self.deconv2 = nn.ConvTranspose3d(128,64,3,stride=2,padding=(1,1,1), output_padding=(1,1,1))
        # self.deconv_batchnorm2 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
        # self.layer7 = self._make_layer(block,128,64,blocks=2,stride=1,padding=(1,1,1))
        # self.deconv4 = nn.Conv3d(64,2,kernel_size=3,stride=1,padding=(1,1,1),bias=True,
        #             padding_mode="circular") #nn.ConvTranspose3d(64,2,1,stride=1,padding=(1,1,1), output_padding=(1,1,1))



    def _make_layer(self,block,inplanes,outplanes,blocks,stride=1,padding=0):
        layers = []
        for i in range(0,blocks):
            layers.append(block(inplanes,outplanes,stride=stride,padding=padding))
            inplanes = outplanes
        return nn.Sequential(*layers)
    
    def _pca_layer(self, N_FG):
        fn = lambda x: PCALayer(x=x, N_FG=N_FG)
        return fn

    def forward(self,x):
        #x_pca = self.pca(x)
        #print("x_pca", x_pca.shape)
        
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

        return x