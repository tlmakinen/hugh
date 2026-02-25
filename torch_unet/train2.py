# --------------------------------------------------------------------------------------
# CRITICAL: Set PyTorch CUDA memory allocation BEFORE importing torch!
# This helps prevent "CUDA out of memory" errors during long training runs
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import h5py
import helpers
import numpy as np
from pathlib import Path
import torch
import math
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import gc
from accelerate import Accelerator

import os.path as osp
import argparse

import cloudpickle as pickle
import sys,json

from dataloader import *
from nets import *

from nets2_attn import *

# --------------------------------------------------------------------------------------

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    

# --------------------------------------------------------------------------------------
# custom log-mse loss



class logMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        # log-transform the targets
        #actual = transform_inputs(actual, scaling=1e5)

        return torch.log(self.mse(pred, actual))



def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

# --------------------------------------------------------------------------------------




parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
# parser.add_argument("--run-name", type=str, required=False, default="")
# parser.add_argument("--nde-file", type=str, required=False, default="./laptop_run_8nets/")
# parser.add_argument("--summary-file", type=str, required=False, default="final_dry_ABC_four_runs_all.pkl")

args = parser.parse_args()



### READ IN CONFIGS
config_file_path = args.config #'./comparison/configs.json'


with open(config_file_path) as f:
        configs = json.load(f)


# model stuff
# HIDDEN_CHANNELS = configs["model_params"][model_type][model_size]["hidden_channels"]
# NUM_LAYERS = configs["model_params"][model_type][model_size]["num_layers"]
# MODEL_NAME = configs["model_params"][model_type][model_size]["name"]
# TEST_BATCHING = configs["model_params"][model_type][model_size]["test_batching"]
        
FILTERS = configs["model_params"]["filters"]
NOISEAMP = configs["model_params"]["noiseamp"]
N_FG = configs["model_params"]["n_fg"]
MODEL_PATH = configs["model_params"]["model_path"]
MODEL_NAME = configs["model_params"]["model_name"]
ACTIVATION = configs["model_params"]["act"]

# PCA components path (for pre-computed PCA)
PCA_COMPONENTS_PATH = configs["model_params"].get("pca_components_path", None)



# # optimizer schedule
LEARNING_RATE = configs["training_params"]["learning_rate"]
BATCH_SIZE = configs["training_params"]["batch_size"]
EPOCHS = int(configs["training_params"]["epochs"])
GRADIENT_CLIP = float(configs["training_params"]["gradient_clip"])
SCALING = 1e5
#DO_SCHEDULER = bool(int(configs["training_params"]["do_lr_scheduler"]))
SEED = int(configs["training_params"]["seed"])

# # data + out directories
cosmopath = configs["training_params"]["cosmopath"]
galpath = configs["training_params"]["galpath"]


MODEL_DIR = configs["model_params"]["model_dir"]
LOAD_DIR = configs["model_params"]["load_dir"]
LOAD_MODEL = bool(configs["training_params"]["load_model"])

TRAIN_WITH_CACHE = False
ADD_NOISE = configs["training_params"]["add_noise"]


if not os.path.exists(MODEL_DIR):
   # Create a new directory if it does not exist
   os.makedirs(MODEL_DIR)
   print("created new directory", MODEL_DIR)

# ### CONSTRUCT MODEL NAME AND OUTPUT PATH
# MODEL_NAME += "nc_%d_nlyr_%d"%(HIDDEN_CHANNELS, NUM_LAYERS)
# MODEL_PATH = MODEL_DIR + MODEL_NAME
# LOAD_PATH = LOAD_DIR + MODEL_NAME



# --------------------------------------------------------------------------------------
    
print("LOADING DATA AND INITIALISING DATALOADERS")

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-computed PCA components if available
pca_components = None

# If no path specified, try default location in model_name directory
if PCA_COMPONENTS_PATH is None:
    # Construct full model path (same as used for model checkpoints)
    full_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    default_pca_path = os.path.join(full_model_path, f"pca_components_nfg{N_FG}.pt")
    
    if os.path.exists(default_pca_path):
        PCA_COMPONENTS_PATH = default_pca_path
        print(f"Found PCA components at default location: {PCA_COMPONENTS_PATH}")

if PCA_COMPONENTS_PATH and os.path.exists(PCA_COMPONENTS_PATH):
    print(f"Loading pre-computed PCA components from {PCA_COMPONENTS_PATH}")
    pca_components = torch.load(PCA_COMPONENTS_PATH, map_location=device)
    print(f"  - Loaded {pca_components['N_FG']} foreground components")
    print(f"  - Frequency bins: {pca_components['N_freq']}")
else:
    print("WARNING: No pre-computed PCA components found. PCA will be computed on-the-fly (slower).")
    print(f"  To speed up training, run: python precompute_pca.py --config {config_file_path}")
    print(f"  Expected location: {os.path.join(MODEL_DIR, MODEL_NAME, f'pca_components_nfg{N_FG}.pt')}")

cosmofiles = os.listdir(cosmopath)
galfiles = os.listdir(galpath)

# save the filenames 
#save_obj(cosmofiles, "/data101/makinen/hirax_sims/dataloader/cosmofiles")
#save_obj(galfiles, "/data101/makinen/hirax_sims/dataloader/galfile")


cosmofiles = [cosmopath + p for p in cosmofiles]
galfiles =  [galpath + p for p in galfiles]


# random mask for train/val split
mask = np.random.rand(len(cosmofiles)) < 0.9
train_cosmo_files = list(np.array(cosmofiles)[mask])[:configs["training_params"]["num_train"]]
val_cosmo_files = list(np.array(cosmofiles)[~mask])

galmask = np.random.rand(len(galfiles)) < 0.9
train_gal_files = list(np.array(galfiles)[galmask])
val_gal_files = list(np.array(galfiles)[~galmask])

# save the train/val masks
#np.save("/data101/makinen/hirax_sims/dataloader/cosmo_mask", mask)
#np.save("/data101/makinen/hirax_sims/dataloader/gal_mask", galmask)


# --------------------------------------------------------------------------------------




def preprocess_data(x, y, target_device=None, pca_comps=None):
    """
    Preprocess data for training. Memory-optimized version.
    
    Args:
        x: Input contaminated signal (complex)
        y: Target cosmology signal (complex)
        target_device: Device to move data to (if None, keep on current device)
        pca_comps: Pre-computed PCA components (for efficient GPU PCA)
    """
    # Move to target device first if specified
    if target_device is not None:
        x = x.to(target_device, non_blocking=True)
        y = y.to(target_device, non_blocking=True)
    
    # Split and reshape more efficiently to reduce intermediate tensors
    # Original shape: (batch, baseline, freq, ra) = (batch, 48, 128, 1024)
    # Target shape: (batch*split, freq, ra/split, baseline)
    
    batch_size = x.shape[0]
    baseline_dim = x.shape[1]
    freq_dim = x.shape[2]
    ra_dim = x.shape[3]
    ra_split_dim = ra_dim // split
    
    # Reshape to split RA dimension: (batch, baseline, freq, split, ra/split)
    x = x.reshape(batch_size, baseline_dim, freq_dim, split, ra_split_dim)
    y = y.reshape(batch_size, baseline_dim, freq_dim, split, ra_split_dim)
    
    # Permute to: (batch, split, freq, ra/split, baseline)
    x = x.permute(0, 3, 2, 4, 1)
    y = y.permute(0, 3, 2, 4, 1)
    
    # Merge batch and split: (batch*split, freq, ra/split, baseline)
    x = x.reshape(batch_size * split, freq_dim, ra_split_dim, baseline_dim)
    y = y.reshape(batch_size * split, freq_dim, ra_split_dim, baseline_dim)
    
    # Convert to real/imag: (batch*split, freq, ra/split, baseline, Re/Im)
    x = torch.stack([x.real, x.imag], dim=-1)
    y = torch.stack([y.real, y.imag], dim=-1)

    # Add white noise
    if ADD_NOISE:
        x.add_(torch.randn_like(x) * NOISEAMP)  # In-place addition
    
    # PCA cleaning (now GPU-friendly with pre-computed components!)
    x = PCALayer(x, N_FG=N_FG, pca_components=pca_comps)

    # Scale data
    x.mul_(1e5)  # In-place multiplication
    y.mul_(1e5)
    
    # Get y into same shape as model outputs: (batch*split, Re/Im, baseline, ra/split, freq)
    y = y.permute(0, 4, 3, 2, 1)
    
    return x, y
    

def my_collate_fn(batch):
    print("batch", len(batch))
    x,y = batch
    x,y = preprocess_data(x,y)
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------------------


# create train and val datasets and loaders with collate fn

print("INITIALISING dataloaders")
train_dataset = H5Dataset(train_cosmo_files, train_gal_files, use_cache=False)
val_dataset = H5Dataset(val_cosmo_files, val_gal_files, use_cache=False)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,  # Increased from 2 (adjust based on GPU memory)
    num_workers=4,  # Increased from 1 for parallel data loading
    shuffle=True,  # Enable shuffling for better training
    pin_memory=True,  # Keep this - helps with CPU->GPU transfer
    persistent_workers=True,  # Reuse workers to avoid recreation overhead
    prefetch_factor=2,  # Prefetch batches for pipeline efficiency
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,  # Match training batch size
    num_workers=2,  # Use workers for validation too
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)


# --------------------------------------------------------------------------------------

# initialise model and accelerator


print("INITIALISING MODEL")
    

# MEMORY OPTIMIZATION: Reduce split factor to lower memory usage
# Original: split = 1024 // 128 = 8 (effective batch = 2 * 8 = 16 samples)
# Reduced: split = 1024 // 256 = 4 (effective batch = 2 * 4 = 8 samples)
split = 1024 // 256  # Reduce from 8 to 4 chunks per sky simulation

# NOTE: If still OOM, try split = 1024 // 512 = 2

#STEPS_PER_EPOCH = 100 # reshuffle data each time 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#block = BasicBlock(16, 32)

act = smooth_leaky if ACTIVATION == "smooth_leaky" else nn.SiLU
# Enable gradient checkpointing for memory savings (~40% less memory, ~20% slower)
model = UNet3d(BasicBlock, filters=FILTERS, act=act, use_checkpoint=True).to(device)


# SET TO BFLOAT16
model.to(torch.bfloat16)


# start up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


# criterion = LogCoshLoss()

criterion = logMSELoss()

model_path = MODEL_PATH
model_path += MODEL_NAME
accelerator = Accelerator(project_dir=model_path)

model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader)



if LOAD_MODEL:
    accelerator.load_state(model_path)
    history = load_obj(MODEL_DIR + MODEL_NAME + "history.pkl")


# --------------------------------------------------------------------------------------



def train(epoch):
    model.train()  # Enable training mode

    pbar = tqdm(total=len(train_dataloader), position=0)
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    
    pbar2 = tqdm(train_dataloader, leave=True, position=0)
    for i, data in enumerate(pbar2):
        optimizer.zero_grad(set_to_none=True)  # More efficient memory cleanup
        
        # Get data - already on CPU from dataloader
        x, y = data
        
        # Preprocess on GPU (much faster than CPU preprocessing!)
        x, y = preprocess_data(x, y, target_device=device, pca_comps=pca_components)
        
        # Forward pass with bfloat16
        preds = model(x.to(torch.bfloat16)).to(torch.float)
        loss = criterion(preds, y)
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient clipping
        accelerator.clip_grad_value_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()

        total_loss += float(loss)
        total_examples += 1
        
        # Explicitly delete large tensors to help memory management
        del x, y, preds
        
        # Periodically clear cache to reduce fragmentation
        if i % 10 == 0:
            torch.cuda.empty_cache()
            
        pbar2.set_description("current loss: %.4f" % (total_loss / total_examples))
        pbar.update(1)

    pbar.close()
    
    # Clear GPU cache
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def test(plot=False):
    model.eval()

    total_loss = total_examples = 0

    for i, data in tqdm(enumerate(val_dataloader), desc="Validation"):
        # Get data
        x, y = data
        
        # Preprocess on GPU
        x, y = preprocess_data(x, y, target_device=device, pca_comps=pca_components)
                
        # Forward pass
        preds = model(x.to(torch.bfloat16)).to(torch.float)
        loss = criterion(preds, y)

        total_loss += float(loss)
        total_examples += 1
        
        if plot and i == 0:  # Only plot first batch
            preds_cpu = preds.cpu()
            y_cpu = y.cpu()
            
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.title("truth")
            plt.imshow(y_cpu[0, 0, 0, :, :])
            plt.colorbar()

            plt.subplot(132)
            plt.title("network prediction")
            plt.imshow(preds_cpu[0, 0, 0, :, :])
            plt.colorbar()
            
            plt.subplot(133)
            plt.title("residual")
            plt.imshow((preds_cpu[0, 0, 0, :, :] - y_cpu[0, 0, 0, :, :]))
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()

    return total_loss / total_examples

# --------------------------------------------------------------------------------------
# run the training loop
    
print("STARTING TRAINING LOOP")


gc.collect()


# training history

history = {
    "train_loss": [],
    "val_loss": [],
    "test_loss": [],
    "losses": []
}

best_loss = np.inf

# training loop
for epoch in range(1, EPOCHS + 1):

    loss = train(epoch)
    loss = float(loss)

    val_loss = float(test())


    if val_loss < best_loss:

        best_loss = val_loss
        accelerator.save_model(model, model_path)
        accelerator.save_state(model_path)

    # save history
    history["train_loss"].append(loss)
    history["val_loss"].append(val_loss)

    save_obj(history, MODEL_NAME + "_train_history_inprog") # save locally
    save_obj(history, model_path + "train_history_inprog")

    # dump to save memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Loss: {loss:.4f}')

# save the history object
save_obj(history, "train_history")
save_obj(history, MODEL_NAME + "train_history")
save_obj(configs, MODEL_NAME + "configs.json")


# should we do the validation checks here ?
