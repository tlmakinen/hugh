import h5py
import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import os.path as osp
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

class H5Dataset(Dataset):
    def __init__(self, cosmo_h5_file_paths, 
                 gal_h5_file_paths, 
                 use_cache=False,
                 loadcache=False,
                 root="/data101/makinen/hirax_sims/dataloader/",
                 num_cosmo=-1,
                 gal_mask=1024
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
        """
        
        self.cosmo_h5_file_paths = cosmo_h5_file_paths
        self.gal_h5_file_paths = gal_h5_file_paths
        self.use_cache = use_cache
        self.num_cosmo = num_cosmo
        self.gal_mask = gal_mask

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

            self.gal_cache.append(gal)
            self.cosmo_cache.append(cosmo)
            
            # now get x and y
            x = gal + cosmo
            y = cosmo
            
            
        else:
            # randomise foreground and cosmology combinations from cache
            rand_idx = torch.randint(low=0, high=len(self.gal_cache), size=())
            gal = self.gal_cache[rand_idx] # random index from cache
            y = self.cosmo_cache[idx] # index of cosmological simulation
            x = gal + y
            
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