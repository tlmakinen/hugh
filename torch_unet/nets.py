import torch
import torch.nn as nn
import numpy as np


from utils import *

def PCALayer(x, N_FG=7):
    '''
    Takes input in [Nx,Ny,Nz,(Re/Im)] data cube form where Nz is number of redshift 
    (frequency) bins. N_FG is number of eigenmodes for PCA to remove
    Cleans foregrounds for Real and Imaginary components separately
    '''
    print("input", x.shape)
    x = torch.permute(x, (0, 2, 3, 4, 1))
    print("input perm", x.shape)
    xreal = PCAclean(x[..., 0], N_FG=N_FG)[0]
    ximag = PCAclean(x[..., 1], N_FG=N_FG)[0]

    return torch.permute(torch.stack((xreal, ximag), dim=-1), (0, 4, 1, 2, 3))


def conv3x3(inplane,outplane, stride=1,padding="same"):
    return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True,
                    padding_mode="circular")

class BasicBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1, padding=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplane,outplane,padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


# input shape: (128,128,48)
# (0,0,0)
padding = (1,1,1)

class UNet3d(nn.Module):
    def __init__(self, block, N_FG=7):
        super(UNet3d,self).__init__()
        self.N_FG = N_FG # foreground components
        self._pca = self._pca_layer(N_FG)
        self.layer1 = self._make_layer(block, 2, 64, blocks=2,stride=1,padding=(1,1,1))
        self.layer2 = self._make_layer(block,64,128, blocks=1,stride=2,padding=(1,1,1)) # (64,64,24)
        self.layer3 = self._make_layer(block,128,128,blocks=2,stride=1,padding=(1,1,1))
        self.layer4 = self._make_layer(block,128,256,blocks=1,stride=2,padding=(1,1,1)) # (32,32,12)
        self.layer5 = self._make_layer(block,256,256,blocks=2,stride=1,padding=(1,1,1))
        self.deconv1 = nn.ConvTranspose3d(256,128,3,stride=2,padding=(1,1,1), output_padding=(1,1,1))
        self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = 128,momentum=0.1) 
        self.layer6 = self._make_layer(block,256,128,blocks=2,stride=1,padding=(1,1,1)) # (64,64,24)
        self.deconv2 = nn.ConvTranspose3d(128,64,3,stride=2,padding=(1,1,1), output_padding=(1,1,1))
        self.deconv_batchnorm2 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
        self.layer7 = self._make_layer(block,128,64,blocks=2,stride=1,padding=(1,1,1))
        self.deconv4 = nn.Conv3d(64,2,kernel_size=3,stride=1,padding=(1,1,1),bias=True,
                    padding_mode="circular") #nn.ConvTranspose3d(64,2,1,stride=1,padding=(1,1,1), output_padding=(1,1,1))



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
        x_pca = self._pca(x)
        
        x1 = self.layer1(x_pca)
        print("x1", x1.shape)
        x  = self.layer2(x1)
        print("x", x.shape)
        x2 = self.layer3(x)
        print("x2", x2.shape)
        x  = self.layer4(x2)
        print("x layer 4", x.shape)
        x  = self.layer5(x)
        print("x layer 5", x.shape)
        x  = nn.functional.relu(self.deconv_batchnorm1((self.deconv1(x))),inplace=True)
        print("x up 1", x.shape)
        x  = torch.cat((x,x2),dim=1)
        print("x cat 1", x.shape)
        x  = self.layer6(x)
        print("x layer 6", x.shape)
        x  = nn.functional.relu(self.deconv_batchnorm2((self.deconv2(x))),inplace=True)
        print("x up 2", x.shape)
        
        x  = torch.cat((x[:, :, :, :], x1[:, :, :, :]),dim=1)
        print("x cat 2", x.shape)

        x  = self.layer7(x)
        x  = self.deconv4(x)

        return x, x_pca