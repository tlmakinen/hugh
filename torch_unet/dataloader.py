import h5py
import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import os.path as osp
import os,re

from tqdm import tqdm
import matplotlib.pyplot as plt

from nets import *

class H5Dataset(Dataset):
    def __init__(self, cosmo_h5_file_paths, 
                 gal_h5_file_paths, 
                 use_cache=False,
                 loadcache=False,
                 root="/data101/makinen/hirax_sims/dataloader/",
                 num_cosmo=-1,
                 gal_mask=1024,
                 add_noise=True,
                 split=8 ,# 1024 // 128
                 N_FG=11,
                 noise_amp=1.5e-7,
                 scaling=1e5
                 ):
        """_summary_

        Args:
            cosmo_h5_file_paths (list): paths to cosmo signal files
            gal_h5_file_paths (list): paths to galaxy signal files
            use_cache (bool, optional): whether to use stored in-memory sims. Defaults to False.
            loadcache (bool, optional): whether to load cache from a disk. Defaults to False.
            root (str, optional): cache root directory. Defaults to "/data101/makinen/hirax_sims/dataloader/".
            num_cosmo (int, optional): number of cosmology simulations to load (from cache). Defaults to -1.
            gal_mask (int, optional): index of galactic plane foreground to remove from. Defaults to 1024.

        Methods:
            __getitem__(self, idx): 
                Returns tuple(torch.tensor): (x,y) = (gal+cosmo, cosmo) pairs, each of shape 
                                            (Re/Im, baseline, freq, RA) (2, 48, 128, 1024)
        """
        
        self.cosmo_h5_file_paths = cosmo_h5_file_paths
        self.gal_h5_file_paths = gal_h5_file_paths
        self.use_cache = use_cache
        self.num_cosmo = num_cosmo
        self.gal_mask = gal_mask
        self.add_noise = add_noise
        self.split = split
        self.N_FG = N_FG
        self.noise_amp = noise_amp
        self.scaling=1e5

        self.cosmo_samples = []
        self.gal_samples = []
        
        
        # gather lists of cosmo and galaxy samples
        for i,path in enumerate(cosmo_h5_file_paths):
            # each index becomes the key
            h5_file = h5py.File(path, 'r')
            self.cosmo_samples.append({'file': h5_file, 'key': i})
                        
        for path in gal_h5_file_paths:
            h5_file = h5py.File(path, 'r')
            self.gal_samples.append({'file': h5_file})
        
        
        # initialise a random index for the galaxy samples
        self.gal_indices = torch.randperm(len(gal_h5_file_paths))
        self.len_gal = len(gal_h5_file_paths)
                
        self.root = root
        
        # check to see if the root dir and x and y cache are there
        # and load 
        if loadcache:
            if osp.exists(root + "gal_cache.pt"):
                self.load_cache()

        else:
            self.gal_cache = []
            self.cosmo_cache = []
        

    # the length of the dataset is determined by the length of the cosmo map dataset
    def __len__(self):
        return len(self.cosmo_samples[:self.num_cosmo])

    # this method returns (x,y) = (cosmo + galaxy, cosmo) training pairs
    # here idx indexes the cosmology simulations (foregrounds will be random)
    def __getitem__(self, idx):
        
        if not self.use_cache:
            # choose the file to read in
            cosmo_sample_info = self.cosmo_samples[idx]
            cosmo_file = cosmo_sample_info['file']
            
            cosmo = torch.tensor(np.array(cosmo_file['/vis/']), dtype=torch.complex64)

            # add random foregrounds
            rand_idx = torch.randint(low=0, high=self.len_gal, size=()) #self.gal_indices[idx]
            gal_sample_info = self.gal_samples[rand_idx]

            gal_file = gal_sample_info['file']
            gal = torch.tensor(np.array(gal_file['/vis/']), dtype=torch.complex64)

            # Only append to cache if we're building it
            # IMPORTANT: This prevents memory leak when use_cache=False
            if len(self.gal_cache) < len(self.cosmo_samples):
                self.gal_cache.append(gal)
                self.cosmo_cache.append(cosmo)
            
            # now get x and y
            x = gal + cosmo
            y = cosmo

            x = x[..., :self.gal_mask]
            y = y[..., :self.gal_mask]
            
            
        else:
            # randomise foreground and cosmology combinations from cache
            rand_idx = torch.randint(low=0, high=len(self.gal_cache), size=())
            gal = self.gal_cache[rand_idx][..., :self.gal_mask] # random index from cache
            y = self.cosmo_cache[idx][..., :self.gal_mask] # index of cosmological simulation
            x = gal + y

            # pass through preprocessing step
            #x,y = self.preprocess_data(x,y)

        return x, y
    
    def set_use_cache(self, use_cache):
        if use_cache:
            self.gal_cache = torch.stack(self.gal_cache)
            self.cosmo_cache = torch.stack(self.cosmo_cache)
        else:
            self.gal_cache = []
            self.cosmo_cache = []
        self.use_cache = use_cache
        
    
    def save_cache(self, extension=""):
        if not osp.exists(self.root):
            os.makedirs(self.root)
        torch.save(torch.stack(self.gal_cache), self.root + "gal_cache" + extension + ".pt")
        torch.save(torch.stack(self.cosmo_cache), self.root + "cosmo_cache" + extension + ".pt")
        
    def load_cache(self, extension="", num_cosmo=-1, num_gal=-1):
        self.gal_cache = torch.load(self.root + "gal_cache" + extension + ".pt")[:num_gal, ..., :self.gal_mask]
        self.cosmo_cache = torch.load(self.root + "cosmo_cache" + extension + ".pt")[:num_cosmo, ..., :self.gal_mask]
        self.use_cache = True

    # create function to load specific simulation and cosmology
    def load_sim_and_cosmology(self, cosmo_path, gal_path):
        """Return specified cosmology and foreground realisation
        with associated cosmological parameter

        Args:
            cosmo_path (str): path to .h5 cosmo file
            gal_path (str): full path and filename for galaxy file

        Returns:
            (x,y), cosmology (tuple): returns given (x,y) pair and cosmology label
        """
        cosmo_file = h5py.File(cosmo_path + cosmo_path, 'r')
        gal_file = h5py.File(gal_path, 'r')

        cosmo = torch.tensor(np.array(cosmo_file['/vis/']), dtype=torch.complex64)
        gal = torch.tensor(np.array(gal_file['/vis/']), dtype=torch.complex64)

        x = gal + cosmo
        y = cosmo

        cosmology = self.cosmo_from_fname(cosmo_path)

        return (x[..., self.gal_mask], y[..., self.gal_mask]), cosmology
        
    def cosmo_from_fname(self, filepath):
        filename = filepath.split("/")[-1] # grab .h5 filename
        nums = re.findall(r'\d+', filename)
        if len(nums) < 5:
            return 67.
        else:
            return float(nums[1])
        

    # def preprocess_data(self, x, y):
    
    #     # split ordering (batch, baseline, freq, ra) = (batch*split, 48, 128, 128)
    #     # then transpose to (batch*split, freq, ra, baseline)
    #     x = torch.permute(
    #         torch.cat(torch.tensor_split(x, self.split, dim=3)),
    #         (0, 3, 1, 2)
    #     )
    #     y = torch.permute(
    #             torch.cat(torch.tensor_split(y, self.split, dim=3)),
    #             (0, 3, 1, 2)
    #     )
    #     # then finally get the real and im parts as channels
    #     # shape: (batch*split, freq, ra, baseline, Re/Im)
    #     x = torch.stack([x.real, x.imag], dim=-1)
    #     y = torch.stack([y.real, y.imag], dim=-1)
        
        
    #     # add white noise to the signal
    #     if self.add_noise:
    #         x += torch.normal(mean=0.0, std=torch.ones(x.shape)*self.noise_amp)
        
    #     # pass x to the pca
    #     x = PCALayer(x, N_FG=self.N_FG)
        
    #     # get y into same shape as model outputs
    #     y = torch.permute(y, (0, 4, 2, 1, 3))
        
    #     # scale data here ?
    #     return x, y

    def __del__(self):
        for sample_info in self.cosmo_samples:
            sample_info['file'].close()
        
        for sample_info in self.gal_samples:
            sample_info['file'].close()

class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        while True:
            order = torch.randperm(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return 2**31