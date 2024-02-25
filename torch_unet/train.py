import h5py
import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import gc
from accelerate import Accelerator

import os.path as osp
import os

import cloudpickle as pickle

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





# model stuff
# HIDDEN_CHANNELS = configs["model_params"][model_type][model_size]["hidden_channels"]
# NUM_LAYERS = configs["model_params"][model_type][model_size]["num_layers"]
# MODEL_NAME = configs["model_params"][model_type][model_size]["name"]
# TEST_BATCHING = configs["model_params"][model_type][model_size]["test_batching"]


# # optimizer schedule
# LEARNING_RATE = configs["training_params"]["learning_rate"]
# EPOCHS = int(configs["training_params"]["epochs"])
# DO_SCHEDULER = bool(int(configs["training_params"]["do_lr_scheduler"]))

# # data + out directories
# DATA_DIR = configs["training_params"]["data_dir"]
# MODEL_DIR = configs["training_params"]["model_dir"]
# LOAD_DIR = configs["training_params"]["load_dir"]


# if not os.path.exists(MODEL_DIR):
#    # Create a new directory if it does not exist
#    os.makedirs(MODEL_DIR)
#    print("created new directory", MODEL_DIR)

# ### CONSTRUCT MODEL NAME AND OUTPUT PATH
# MODEL_NAME += "nc_%d_nlyr_%d"%(HIDDEN_CHANNELS, NUM_LAYERS)
# MODEL_PATH = MODEL_DIR + MODEL_NAME
# LOAD_PATH = LOAD_DIR + MODEL_NAME



# --------------------------------------------------------------------------------------
    
print("LOADING DATA AND INITIALISING DATALOADERS")

# fix random seed
np.random.seed(4)
torch.manual_seed(4)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cosmopath = "/data101/makinen/hirax_sims/cosmo_gaussian_pb/"
galpath = '/data101/makinen/hirax_sims/more_baselines/galaxy_gaussian_pb/'

cosmofiles = os.listdir(cosmopath)
galfiles = os.listdir(galpath)

# save the filenames 
save_obj(cosmofiles, "/data101/makinen/hirax_sims/dataloader/cosmofiles")
save_obj(galfiles, "/data101/makinen/hirax_sims/dataloader/galfile")



cosmofiles = [cosmopath + p for p in cosmofiles]
galfiles =  [galpath + p for p in galfiles]


# random mask for train/val split
mask = np.random.rand(len(cosmofiles)) < 0.9
train_cosmo_files = list(np.array(cosmofiles)[mask])
val_cosmo_files = list(np.array(cosmofiles)[~mask])

galmask = np.random.rand(len(galfiles)) < 0.9
train_gal_files = list(np.array(galfiles)[galmask])
val_gal_files = list(np.array(galfiles)[~galmask])

# save the train/val masks
np.save("/data101/makinen/hirax_sims/dataloader/cosmo_mask", mask)
np.save("/data101/makinen/hirax_sims/dataloader/gal_mask", galmask)



# create train and val datasets and loaders


train_dataset = H5Dataset(train_cosmo_files, train_gal_files, use_cache=False)
val_dataset = H5Dataset(val_cosmo_files, val_gal_files, use_cache=False)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,  # bigger batch ?
    num_workers=5, # how high can we go ?
    shuffle=False,
    pin_memory=True # do we need this ?
)

val_dataloader = DataLoader(
    val_dataset,
    num_workers=0,
    shuffle=False,
    pin_memory=True
)


print("INITIALISING MODEL")

split = 1024 // 128 # 8 chunks per sky simulation

TRAIN_WITH_CACHE = False
N_FG = 7

LEARNING_RATE = 2e-5

max_patience = 25

# create model
model = UNet3d(BasicBlock, filters=8).to(device)

# start up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = logMSELoss()

accel_path = "/data101/makinen/hirax_sims/accelerator/residual_net/"
accelerator = Accelerator(project_dir=accel_path)

model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader)


# --------------------------------------------------------------------------------------



def preprocess_data(x,y):
    """
    PCA preprocessing for foreground sky simulation. 

    Args:
        x (torch.Tensor): foregrounds + cosmological signal
        y (torch.Tensor): clean cosmological signal

    Returns:
        (x,y): Tuple of preprocessed inputs and targets on specified device
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
    
    # pass x to the pca
    x = PCALayer(x, N_FG=N_FG)
    
    # get y into same shape as model outputs
    y = torch.permute(y, (0, 4, 2, 1, 3))
    
    return x.to(device),y.to(device)
    

def train(epoch):

    #model.train()

    pbar = tqdm(total=len(train_dataloader), position=0)
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    
    pbar2 = tqdm(train_dataloader, leave=True, position=0)

    patience = 0
    best_loss = np.inf

    for data in pbar2:
        optimizer.zero_grad()
        
        x,y = data
        x,y = preprocess_data(x,y)
        
        y *= model.scaling # 0,2 playing field
        x *= model.scaling # 0,2 playing field
        
         # model learns residual model(x) = x + y => y_pred = model(x) - x
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

        # count patience
        if total_loss < best_loss:
            best_loss = total_loss
            patience = 0

        else:
            patience += 1
            

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()


    for i,data in tqdm(enumerate(val_dataloader)):
        #if i % 2 == 0:
        x,y = data
        x,y = preprocess_data(x,y)
        
        y *= model.scaling # 
        x *= model.scaling
        
        # model learns residual model(x) = x + y => y_pred = model(x) - x
        preds = model(x).cpu() #- x.cpu()
        
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


    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(val_dataloader), position=0)
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in val_dataloader:
        data = data.to(device)

        
        out = model

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

# --------------------------------------------------------------------------------------
# run the training loop
    
print("STARTING TRAINING LOOP")

EPOCHS = 30

MODEL_PATH = "/data101/makinen/hirax_sims/accelerator/"

gc.collect()

best_loss = np.inf

# training loop
for epoch in range(1, EPOCHS + 1):

    loss = train(epoch)
    loss = float(loss)

    if loss < best_loss:
        
        best_loss = loss
        accelerator.save_model(model, MODEL_PATH)
        accelerator.save_state(MODEL_PATH)
    
    print(f'Loss: {loss:.4f}')

