import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h5py, os


from tqdm import tqdm

def get_cosmo_data(datadir = "/data80/makinen/hirax_sims/230520_ml_newcosmo_vis/",
                   num_train=800,
                  savedir="/data80/makinen/hirax_sims/diffusion/"):
    files = os.listdir(datadir)
    cosmo = []
    params = []
    for i,f in tqdm(enumerate(files)):
        cosmo.append(np.array(h5py.File(datadir + f, "r")['/vis/']).T[:, :, :12])
        
        p = f[9:11]
        if p == "pl":
            p = 67
        
        params.append(float(p)) # get H0 value
        
    if savedir is not None:
        np.save(savedir + "cosmo_vary_H0_vis", cosmo)
        np.save(savedir + "cosmo_vary_H0_params", np.array(params))
        
    return np.array(cosmo), np.array(params)


### main bit

cosmo, params = get_cosmo_data()