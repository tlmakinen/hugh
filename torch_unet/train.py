import h5py
import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import gc
from accelerate import Accelerator

import os.path as osp
import os

import cloudpickle as pickle
import sys,os,json

from dataloader import *
from nets import *

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
        return torch.log(self.mse(pred, actual))

# --------------------------------------------------------------------------------------


### READ IN CONFIGS
config_file_path = sys.argv[1] #'./comparison/configs.json'


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



# # optimizer schedule
LEARNING_RATE = configs["training_params"]["learning_rate"]
BATCH_SIZE = configs["training_params"]["batch_size"]
EPOCHS = int(configs["training_params"]["epochs"])
#DO_SCHEDULER = bool(int(configs["training_params"]["do_lr_scheduler"]))
SEED = int(configs["training_params"]["seed"])

# # data + out directories
cosmopath = configs["training_params"]["cosmopath"]
galpath = configs["training_params"]["galpath"]


MODEL_DIR = configs["model_params"]["model_dir"]
LOAD_DIR = configs["model_params"]["load_dir"]


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

cosmofiles = os.listdir(cosmopath)
galfiles = os.listdir(galpath)

# save the filenames 
#save_obj(cosmofiles, "/data101/makinen/hirax_sims/dataloader/cosmofiles")
#save_obj(galfiles, "/data101/makinen/hirax_sims/dataloader/galfile")



cosmofiles = [cosmopath + p for p in cosmofiles]
galfiles =  [galpath + p for p in galfiles]


# random mask for train/val split
mask = np.random.rand(len(cosmofiles)) < 0.9
train_cosmo_files = list(np.array(cosmofiles)[mask])
val_cosmo_files = list(np.array(cosmofiles)[~mask])

galmask = np.random.rand(len(galfiles)) < 0.9
train_gal_files = list(np.array(galfiles)[galmask])[:configs["training_params"]["num_train"]]
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

# the new collate function is quite generic
#loader = DataLoader(demo, batch_size=50, shuffle=True, 
#                    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

val_dataloader = DataLoader(
    val_dataset,
    num_workers=0,
    shuffle=False,
    pin_memory=True
)


# --------------------------------------------------------------------------------------

# initialise model and accelerator


print("INITIALISING MODEL")
    
# reinitialise the dataloader
# train_dataset = H5Dataset(train_cosmo_files, train_gal_files, use_cache=False)

# #train_dataset.use_cache = False
# #train_dataset.num_cosmo = len(train_dataset.cosmo_cache)

# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=1,
#     num_workers=1,
#     shuffle=True,
# )


split = 1024 // 128 # 8 chunks per sky simulation

TRAIN_WITH_CACHE = False
ADD_NOISE = True
#STEPS_PER_EPOCH = 100 # reshuffle data each time 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#block = BasicBlock(16, 32)
model = UNet3d(BasicBlock, filters=16).to(device)

# start up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = logMSELoss()

model_path = MODEL_PATH
model_path += MODEL_NAME
accelerator = Accelerator(project_dir=model_path)

model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader)


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
        
        x *= model.scaling
        y *= model.scaling # same playing field as network
        
        preds = model(x)
        loss = criterion(preds, y)
        
        accelerator.backward(loss)
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
        
        y *= model.scaling # 
        x *= model.scaling
        
        # model learns residual model(x) = x + y => y_pred = model(x) - x
        preds = model(x) #- x.cpu()
        
        preds = model(x)
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

#MODEL_PATH = "/data101/makinen/hirax_sims/accelerator/"

gc.collect()


# training history

history = {
    "train_loss": [],
    "valid_loss": [],
    "test_loss": [],
    "losses": []
}

best_loss = np.inf

# training loop
for epoch in range(1, EPOCHS + 1):

    loss = train(epoch)
    loss = float(loss)

    #history["losses"].append(loss)
    #history["train_aucs"].append(train_rocauc)
    #history["valid_aucs"].append(valid_rocauc)
    #history["test_aucs"].append(test_rocauc)

    if loss < best_loss:

        best_loss = loss
        accelerator.save_model(model, model_path)
        accelerator.save_state(model_path)

    # save history
    history["train_loss"].append(loss)

    # dump to save memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Loss: {loss:.4f}')

# save the history object
save_obj(history, "train_history")




# should we do the validation checks here ?
