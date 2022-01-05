from typing import Any, Callable, Sequence, Optional
from flax.core import freeze, unfreeze
from functools import partial

# flax import
import flax.linen as nn
import optax


import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax



class AsinhLayer(nn.Module):
    bias_init: Callable = nn.initializers.zeros
    a_init: Callable = nn.initializers.ones
    b_init: Callable = nn.initializers.ones
    c_init: Callable = nn.initializers.zeros
    d_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, inputs):

        a = self.param('a', self.a_init, (1,))
        b = self.param('b', self.b_init, (1,))
        c = self.param('c', self.c_init, (1,))
        d = self.param('d', self.d_init, (1,)) 

        y = a*jnp.arcsinh(b*inputs + c) + d
        return y
    


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvBlock(nn.Module):
    filters: int
    strides: int
    dim: int = 3
    kernel: int = 3
    act: Callable = nn.relu
    padding: str = 'SAME'

    @nn.compact
    def __call__(self, x):
        k = self.kernel
        fs = self.filters
        x = nn.Conv(features=fs, kernel_size=(k,)*self.dim, 
                    strides=self.strides, padding=self.padding)(x)
        x = self.act(x)
        return x

class DoubleConvBlock(nn.Module):
    filters: int
    strides: int
    dim: int = 3
    kernel: int = 3
    act: Callable = nn.relu
    padding: str = 'SAME'

    @nn.compact
    def __call__(self, x):
        k = self.kernel
        fs = self.filters
        x = nn.Conv(features=fs, kernel_size=(k,)*self.dim, 
                    strides=self.strides, padding=self.padding)(x)
        x = self.act(x)
        x = nn.Conv(features=fs, kernel_size=(k,)*self.dim, 
                    strides=self.strides, padding=self.padding)(x)
        x = self.act(x)
        return x


class unet(nn.Module):
    """a unet-encoder module"""
    filters : int
    act: Callable = nn.relu
    dim: int = 3
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):

        fs = self.filters
        padding = self.padding
        _shape = x.shape
        
        x = AsinhLayer()(x)
        
        # downsample
        x1 = DoubleConvBlock(fs, strides=None, act=self.act, padding=padding)(x)
        x = ConvBlock(fs, strides=2, act=self.act, padding=padding)(x1) # down to 8
        x2 = DoubleConvBlock(fs*2, strides=None, act=self.act, padding=padding)(x)
        x = ConvBlock(fs*2, strides=2,act=self.act, padding=padding)(x2) # down to 4
        x = DoubleConvBlock(fs*4, strides=None, act=self.act, padding=padding)(x) 

        # upsample path
        x = nn.ConvTranspose(features=fs*4, kernel_size=(1,)*3, strides=(2,)*3, padding=padding)(x) # upsample to 8
        x = self.act(x)
        x = jnp.concatenate([x,x2], axis=-1)
        x = DoubleConvBlock(fs*2, strides=None, act=self.act, padding=padding)(x) 
        x = nn.ConvTranspose(features=fs*2, kernel_size=(1,)*3, strides=(2,)*3, padding=padding)(x) # upsample to 16
        x = self.act(x)
        x = jnp.concatenate([x,x1], axis=-1) # filters add to fs*2
        x = DoubleConvBlock(fs, strides=None,act=self.act, padding=padding)(x) 
        x = nn.ConvTranspose(features=1, kernel_size=(3,)*3, strides=None, padding=padding)(x)

        return x

