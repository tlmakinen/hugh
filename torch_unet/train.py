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

# --------------------------------------------------------------------------------------

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    

from dataloader import *
from nets import *

class logMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.log(self.mse(pred, actual))

# --------------------------------------------------------------------------------------


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
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

val_dataloader = DataLoader(
    val_dataset,
    num_workers=0,
    shuffle=False,
)


split = 1024 // 128 # 8 chunks per sky simulation

TRAIN_WITH_CACHE = False
N_FG = 11

LEARNING_RATE = 1e-3

max_patience = 25

# create model

model = UNet3d(BasicBlock, filters=8, N_FG=11).to(device)


# start up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = logMSELoss()

accel_path = "/data101/makinen/hirax_sims/accelerator/"
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
        
        y *= model.scaling # same playing field as network
        
         # model learns residual model(x) = x + y => y_pred = model(x) - x
        preds = model(x)
        loss = criterion(preds, y + x)
        
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
        
        y *= model.scaling # same playing field as network
        
        # model learns residual model(x) = x + y => y_pred = model(x) - x
        preds = model(x).cpu() - x.cpu()
        
        plt.subplot(121)
        plt.imshow(y[0, 0, 0, :, :].cpu().detach().numpy())
        plt.colorbar()

        plt.subplot(122)
        plt.imshow(preds[0, 0, 0, :, :].detach().numpy())
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
# run the loop

EPOCHS = 20

MODEL_PATH = "/data101/makinen/hirax_sims/accelerator/"

gc.collect()

# training loop
for epoch in range(1, EPOCHS + 1):

    loss = train(epoch)
    
    accelerator.save_model(model, MODEL_PATH)
    accelerator.save_state(MODEL_PATH)
    
    print(f'Loss: {loss:.4f}')

