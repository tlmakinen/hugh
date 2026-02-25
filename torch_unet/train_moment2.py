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
import os
import argparse

import cloudpickle as pickle
import sys,os,json

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

args = parser.parse_args()



### READ IN CONFIGS
config_file_path = args.config #'./comparison/configs.json'


with open(config_file_path) as f:
        configs = json.load(f)


# model stuff
        
FILTERS = configs["model_params"]["filters"]
NOISEAMP = configs["model_params"]["noiseamp"]
N_FG = configs["model_params"]["n_fg"]
MODEL_PATH = configs["model_params"]["model_path"]
MODEL_NAME = configs["model_params"]["model_name"]
ACTIVATION = configs["model_params"]["act"]



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




# --------------------------------------------------------------------------------------
    
print("LOADING DATA AND INITIALISING DATALOADERS")

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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




def preprocess_data(x,y):
    
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
        if ADD_NOISE:
            x += torch.normal(mean=0.0, std=torch.ones(x.shape)*NOISEAMP) #.to(device)
        
        # pass x to the pca
        x = PCALayer(x, N_FG=N_FG)

        x *= 1e5
        y *= 1e5
        
        # log-transformation of input data for network
        #x = transform_inputs(x, scaling=1e5)
        
        # transformation of outputs handled in the loss function
        #y = transform_inputs(y, scaling=1e5)

        
        # get y into same shape as model outputs
        y = torch.permute(y, (0, 4, 2, 1, 3))
        
        return x.to(device),y.to(device)
    

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
    batch_size=2,  # bigger batch ?
    num_workers=1, # how high can we go ?
    shuffle=False,
    pin_memory=True, # do we need this ?
    #collate_fn=my_collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    num_workers=0,
    shuffle=False,
    pin_memory=True
)


# --------------------------------------------------------------------------------------

# initialise model and accelerator




print("LOADING FIRST MOMENT MODEL")
    

split = 1024 // 128 # 8 chunks per sky simulation


#STEPS_PER_EPOCH = 100 # reshuffle data each time 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

act = smooth_leaky if ACTIVATION == "smooth_leaky" else nn.SiLU
model_1 = UNet3d(BasicBlock, filters=FILTERS, act=act).to(device)
model_path = MODEL_PATH
model_path += MODEL_NAME
model_1.load_state_dict(torch.load(model_path + "/pytorch_model.bin"))
model_1.eval()

for name, para in model_1.named_parameters():
    para.requires_grad = False

model_1.to(torch.bfloat16)


print("INITIALISING SECOND MOMENT")

# INITIALISE MOMENT 2 --> "MODEL"
print("INITIALISING SECOND MOMENT MODEL")

act = smooth_leaky if ACTIVATION == "smooth_leaky" else nn.SiLU
model = UNet3d(BasicBlock, filters=FILTERS, act=act).to(device)
model.to(torch.bfloat16)


model_path = MODEL_PATH
model_path += MODEL_NAME
model_path += "_moment_2"





split = 1024 // 128 # 8 chunks per sky simulation


#STEPS_PER_EPOCH = 100 # reshuffle data each time 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#block = BasicBlock(16, 32)

act = smooth_leaky if ACTIVATION == "smooth_leaky" else nn.SiLU
model = UNet3d(BasicBlock, filters=FILTERS, act=act).to(device)


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
    #model.train()

    pbar = tqdm(total=len(train_dataloader), position=0)
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    
    pbar2 = tqdm(train_dataloader, leave=True, position=0)
    for i,data in enumerate(pbar2):
        optimizer.zero_grad()
        
        
        x,y = data
        x,y = preprocess_data(x.cpu(),y.cpu())
        
        # MODIFY THIS FOR SECOND MOMENT TRAINING
        preds = model(x.to(torch.bfloat16)).to(torch.float)
        
        # freeze moment_1
        with torch.no_grad():
            y = (model_1(x.to(torch.bfloat16)).to(torch.float) - y)**2

        loss = criterion(preds, y)
        
        accelerator.clip_grad_value_(model.parameters(), GRADIENT_CLIP) # GRADIENT CLIPPING
        
        optimizer.step() 
        #if DO_SCHEDULER:
        #    lr_scheduler.step()

        total_loss += float(loss) #* int(data.train_mask.sum())
        total_examples += 1 #data.shape #int(data.train_mask.sum())
        
        if not TRAIN_WITH_CACHE:
            train_dataloader.dataset.gal_cache = []
            train_dataloader.dataset.cosmo_cache = []
            
        pbar2.set_description("current loss: %.4f"%(total_loss / total_examples))
            
        pbar.update(1)

    pbar.close()
    # dump to save memory
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def test(plot=False):
    model.eval()

    total_loss = total_examples = 0

    #pbar = tqdm(total=len(val_dataloader), position=0)

    for i,data in tqdm(enumerate(val_dataloader)):
        #if i % 2 == 0:
        x,y = data
        x,y = preprocess_data(x.cpu(),y.cpu())
                
        # MODIFY THIS FOR SECOND MOMENT TRAINING
        preds = model(x.to(torch.bfloat16)).to(torch.float)
        
        # freeze moment_1
        with torch.no_grad():
            y = (model_1(x.to(torch.bfloat16)).to(torch.float) - y)**2

        loss = criterion(preds, y)

        preds = preds.cpu()
        y = y.cpu()

        total_loss += float(loss) #* int(data.train_mask.sum())
        total_examples += 1 #data.shape #int(data.train_mask.sum())
        
        if plot:
            plt.subplot(131)
            plt.title("truth")
            plt.imshow(y[0, 0, 0, :, :])
            plt.colorbar()

            plt.subplot(132)
            plt.title("network prediction")
            plt.imshow(preds[0, 0, 0, :, :])
            plt.colorbar()
            
            
            plt.subplot(133)
            plt.title("residual")
            plt.imshow(((preds[0, 0, 0, :, :] - y[0, 0, 0, :, :])))
            plt.colorbar()
            
            plt.show()
    
    val_dataloader.dataset.cosmo_cache = []
    val_dataloader.dataset.gal_cache = []

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
