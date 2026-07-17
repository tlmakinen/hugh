# --------------------------------------------------------------------------------------
# CRITICAL: Set PyTorch CUDA memory allocation BEFORE importing torch!
# This helps prevent "CUDA out of memory" errors during long training runs
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import h5py
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
from utils import apply_precomputed_pca_fast

from nets2_attn import *

# v2 model (configurable: GroupNorm, SiLU, resize-conv, axis-aware padding, ...)
from nets_v2 import UNet3dV2

import copy

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

# v2-model knobs (all default to the v1 behaviour if not present).
MODEL_VERSION = configs["model_params"].get("model_version", "v1").lower()
V2_KWARGS = dict(
    filters=FILTERS,
    levels=configs["model_params"].get("levels", 3),
    block=configs["model_params"].get("block", "preact_res"),
    n_blocks_per_stage=configs["model_params"].get("n_blocks_per_stage", 2),
    norm=configs["model_params"].get("norm", "groupnorm"),
    activation=configs["model_params"].get("activation", "silu"),
    pad_modes=tuple(configs["model_params"].get("pad_modes", ["circular", "reflect", "replicate"])),
    downsample_kind=configs["model_params"].get("downsample_kind", "blurpool"),
    upsample_kind=configs["model_params"].get("upsample_kind", "resize_conv"),
    strides=configs["model_params"].get("strides", None),
    attention_gates=bool(configs["model_params"].get("attention_gates", False)),
    use_se=bool(configs["model_params"].get("use_se", False)),
    factored_convs=bool(configs["model_params"].get("factored_convs", False)),
    film_conditioning=bool(configs["model_params"].get("film_conditioning", False)),
    film_cond_dim=int(configs["model_params"].get("film_cond_dim", 8)),
    freq_pos_encoding=bool(configs["model_params"].get("freq_pos_encoding", False)),
    freq_pos_dim=int(configs["model_params"].get("freq_pos_dim", 2)),
    two_conv_head=bool(configs["model_params"].get("two_conv_head", True)),
    norm_groups=int(configs["model_params"].get("norm_groups", 8)),
    conv_kernel_size=configs["model_params"].get("conv_kernel_size", 3),
)
# use_grad_checkpoint config flag is shared with v1; let v2 also honour it.



# # optimizer schedule
LEARNING_RATE = configs["training_params"]["learning_rate"]
BATCH_SIZE = configs["training_params"]["batch_size"]
EPOCHS = int(configs["training_params"]["epochs"])
GRADIENT_CLIP = float(configs["training_params"]["gradient_clip"])
EARLY_STOPPING_PATIENCE = int(configs["training_params"].get("early_stopping_patience", 0))
EARLY_STOPPING_MIN_DELTA = float(configs["training_params"].get("early_stopping_min_delta", 0.0))
TRAIN_NUM_WORKERS = int(configs["training_params"].get("train_num_workers", configs["training_params"].get("num_workers", 4)))
VAL_NUM_WORKERS = int(configs["training_params"].get("val_num_workers", max(1, TRAIN_NUM_WORKERS // 2)))
PREFETCH_FACTOR = int(configs["training_params"].get("prefetch_factor", 2))
MIXED_PRECISION = configs["training_params"].get("mixed_precision", "bf16")
USE_COMPILE = bool(configs["training_params"].get("use_compile", False))
COMPILE_MODE = configs["training_params"].get("compile_mode", "default")
GRAD_ACCUM_STEPS = max(1, int(configs["training_params"].get("grad_accum_steps", 1)))
SCALING = 1e5

# v3 training-side knobs (defaults preserve legacy behaviour).
TARGET_TRANSFORM = str(configs["training_params"].get("target_transform", "linear")).lower()
PREDICT_RESIDUAL = bool(configs["training_params"].get("predict_residual", False))
RA_SHIFT_AUG = bool(configs["training_params"].get("ra_shift_aug", False))
OPTIMIZER_NAME = str(configs["training_params"].get("optimizer", "adam")).lower()
WEIGHT_DECAY = float(configs["training_params"].get("weight_decay", 0.0))
LR_SCHEDULE = str(configs["training_params"].get("lr_schedule", "none")).lower()
WARMUP_FRAC = float(configs["training_params"].get("warmup_frac", 0.05))
EMA_DECAY      = float(configs["training_params"].get("ema_decay", 0.0))
USE_DELAY_LOSS = bool(configs["training_params"].get("use_delay_loss", False))
LAMBDA_DELAY   = float(configs["training_params"].get("lambda_delay", 0.1))

assert TARGET_TRANSFORM in ("linear", "arcsinh"), TARGET_TRANSFORM
assert OPTIMIZER_NAME in ("adam", "adamw"), OPTIMIZER_NAME
assert LR_SCHEDULE in ("none", "onecycle", "cosine"), LR_SCHEDULE

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




def _forward_transform(t):
    """Map physical-units tensor -> training-space (arcsinh or linear scaling)."""
    if TARGET_TRANSFORM == "arcsinh":
        # arcsinh(SCALING * x): near-linear for small x, log-like for large x.
        return torch.arcsinh(t * SCALING)
    # default = legacy multiplicative scaling
    return t.mul(SCALING)


def inverse_transform(t):
    """Map training-space tensor -> physical units."""
    if TARGET_TRANSFORM == "arcsinh":
        return torch.sinh(t) / SCALING
    return t / SCALING


def preprocess_data(x, y, target_device=None, pca_comps=None, training=False):
    """
    Preprocess data for training. Memory-optimized version.
    
    Args:
        x: Input contaminated signal (complex) - shape: (batch, baseline, freq, ra)
        y: Target cosmology signal (complex) - shape: (batch, baseline, freq, ra)
        target_device: Device to move data to (if None, keep on current device)
        pca_comps: Pre-computed PCA components (for efficient GPU PCA)
        training: If True, apply RA-shift augmentation (if enabled).

    Returns:
        x: Model input  - shape: (batch*split, Re/Im, ra/split, freq, baseline)
        y: Loss target  - shape: (batch*split, Re/Im, ra/split, freq, baseline)
    """
    # Move to target device first if specified
    if target_device is not None:
        x = x.to(target_device, non_blocking=True)
        y = y.to(target_device, non_blocking=True)

    # Original shape: (batch, baseline, freq, ra) = (batch, 48, 128, 1024)
    batch_size = x.shape[0]
    baseline_dim = x.shape[1]
    freq_dim = x.shape[2]
    ra_dim = x.shape[3]
    ra_split_dim = ra_dim // split

    # Free, RA-equivariant data augmentation: cyclic shift along the RA axis.
    # PCA cleaning is RA-invariant (acts on freq), so we apply the shift here
    # before the RA-chunking reshape. Both x and y are shifted by the same
    # amount so the supervision stays aligned.
    if training and RA_SHIFT_AUG:
        shift = int(torch.randint(0, ra_dim, (1,)).item())
        if shift:
            x = torch.roll(x, shifts=shift, dims=-1)
            y = torch.roll(y, shifts=shift, dims=-1)

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

    # PCA foreground cleaning along the frequency axis.
    # The current layout has freq at axis 1, so we temporarily permute to put
    # freq as the last axis (required by apply_precomputed_pca_fast which does
    # data.reshape(-1, N_freq)). Then permute back to the model layout below.
    #
    # NOTE: The original train2.py called PCALayer directly on the
    # (B, freq, ra/split, baseline, Re/Im) tensor, which silently reshaped it
    # treating `baseline` as the frequency axis. As a result foregrounds were
    # never actually removed and the network never learned a useful mapping.
    x_pca_in = x.permute(0, 2, 3, 1, 4).contiguous()  # (B, ra/split, baseline, freq, Re/Im)
    if pca_comps is not None:
        xreal = apply_precomputed_pca_fast(x_pca_in[..., 0], pca_comps)
        ximag = apply_precomputed_pca_fast(x_pca_in[..., 1], pca_comps)
    else:
        # Fallback: on-the-fly PCA (slower, only used without precomputed comps)
        from nets import PCAclean
        xreal = PCAclean(x_pca_in[..., 0], N_FG=N_FG)[0]
        ximag = PCAclean(x_pca_in[..., 1], N_FG=N_FG)[0]
    x_cleaned = torch.stack([xreal, ximag], dim=-1)  # (B, ra/split, baseline, freq, Re/Im)

    # Reorder to model layout (B, Re/Im, ra/split, freq, baseline):
    #   axes (0, 1, 2, 3, 4) = (B, ra/split, baseline, freq, Re/Im)
    #   target            = (B, Re/Im,   ra/split, freq, baseline)
    #   permutation       = (0, 4,       1,        3,    2)
    x = x_cleaned.permute(0, 4, 1, 3, 2).contiguous()

    # y target shape (B, Re/Im, ra/split, freq, baseline):
    #   y current axes (0, 1, 2, 3, 4) = (B, freq, ra/split, baseline, Re/Im)
    #   permutation                    = (0, 4,    2,        1,    3)
    y = y.permute(0, 4, 2, 1, 3).contiguous()

    # Map to training space (legacy `* SCALING` or `arcsinh(SCALING * .)`).
    x = _forward_transform(x)
    y = _forward_transform(y)

    if PREDICT_RESIDUAL:
        # Train the network to output the *correction* the PCA-cleaned input
        # needs to recover the truth: residual = transform(truth) - transform(input).
        # The reconstructed cosmology is then `transform(input) + model_out` in
        # transformed space (inverse-transformed back to physical units).
        y = y - x

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
    batch_size=BATCH_SIZE,
    num_workers=TRAIN_NUM_WORKERS,
    shuffle=True,  # Enable shuffling for better training
    pin_memory=True,  # Keep this - helps with CPU->GPU transfer
    persistent_workers=TRAIN_NUM_WORKERS > 0,  # Reuse workers to avoid recreation overhead
    prefetch_factor=PREFETCH_FACTOR if TRAIN_NUM_WORKERS > 0 else None,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=VAL_NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    persistent_workers=VAL_NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR if VAL_NUM_WORKERS > 0 else None,
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
USE_GRAD_CHECKPOINT = bool(configs["training_params"].get("use_grad_checkpoint", False))

if MODEL_VERSION == "v2":
    V2_KWARGS["use_checkpoint"] = USE_GRAD_CHECKPOINT
    print(f"Building UNet3dV2 with: {V2_KWARGS}")
    model = UNet3dV2(**V2_KWARGS).to(device)
elif MODEL_VERSION in ("v1", "legacy"):
    model = UNet3d(BasicBlock, filters=FILTERS, act=act, use_checkpoint=USE_GRAD_CHECKPOINT).to(device)
else:
    raise ValueError(f"Unknown model_version: {MODEL_VERSION!r}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameter count: {n_params/1e6:.2f}M")


# start up the optimizer
if OPTIMIZER_NAME == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


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


class DelaySpectrumLoss(nn.Module):
    """Delay-spectrum loss with inverse-power weighting.

    FFTs predictions and truth along the frequency axis, weights each
    (delay, baseline) mode by 1/mean_truth_power so high-power foreground
    modes don't swamp the EoR-window signal, and returns a log-ratio MSE.

    Crucially, NO wedge mask is applied: the network is free to learn
    signal recovery inside the wedge as well as above it.

    Expects layout (B, 2[Re/Im], ra/split, freq, baseline) in whatever
    compressed space the training loop uses (e.g. arcsinh).
    """

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps
        self._window = None

    def _blackman_harris(self, n: int, device) -> torch.Tensor:
        if self._window is not None and self._window.shape[0] == n:
            return self._window.to(device)
        k = torch.arange(n, dtype=torch.float32)
        w = (0.35875
             - 0.48829 * torch.cos(2 * math.pi * k / (n - 1))
             + 0.14128 * torch.cos(4 * math.pi * k / (n - 1))
             - 0.01168 * torch.cos(6 * math.pi * k / (n - 1)))
        self._window = w
        return w.to(device)

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        # Cast to float32: FFT over fp16 tensors is numerically fragile.
        pred  = pred.float()
        truth = truth.float()

        # Reconstruct complex visibilities: (B, ra/split, freq, baseline)
        z_pred  = torch.complex(pred[:,  0], pred[:,  1])
        z_truth = torch.complex(truth[:, 0], truth[:, 1])

        # Blackman-Harris window along freq axis (dim=2); reduces spectral leakage.
        n_freq = z_pred.shape[2]
        win = self._blackman_harris(n_freq, pred.device)[None, None, :, None]

        D_pred  = torch.fft.fft(z_pred  * win, dim=2)  # (B, ra/split, n_delay, baseline)
        D_truth = torch.fft.fft(z_truth * win, dim=2)

        P_pred  = D_pred.abs().pow(2)
        P_truth = D_truth.abs().pow(2)

        # Inverse-power weights from truth: mean over batch and RA → (n_delay, baseline).
        # Upweights EoR-window modes, downweights foreground-dominated low-delay modes,
        # without zeroing out either — the network can still learn inside the wedge.
        weights = 1.0 / (P_truth.mean(dim=(0, 1)).detach() + self.eps)

        log_diff_sq = (torch.log(P_pred  + self.eps)
                     - torch.log(P_truth + self.eps)).pow(2)
        return (weights[None, None] * log_diff_sq).mean()


delay_criterion = DelaySpectrumLoss() if USE_DELAY_LOSS else None
if USE_DELAY_LOSS:
    print(f"Delay-spectrum loss enabled: lambda_delay={LAMBDA_DELAY}")

# Speed knobs.
#  - cudnn.benchmark = True lets cuDNN pick the fastest conv3d algorithm for
#    our fixed (256, 128, 48) spatial shape; we always feed the same shape so
#    autotune overhead is paid once.
#  - TF32 on matmul / cudnn gives ~3x speedup over plain fp32 on H100 with
#    accuracy that is more than sufficient for this task.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

model_path = MODEL_PATH
model_path += MODEL_NAME
# gradient_accumulation_steps: when > 1, Accelerator divides the loss internally
# inside the `accelerator.accumulate(model):` context and only syncs gradients +
# steps the optimizer every N batches. Lets us keep an effective batch of N x
# `batch_size` on a small GPU (e.g. a 32 GB V100) without changing the model.
accelerator = Accelerator(
    project_dir=model_path,
    mixed_precision=MIXED_PRECISION,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
)
if GRAD_ACCUM_STEPS > 1:
    print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} micro-batches per optimizer step "
          f"(effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")

model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader)


# Optional learning-rate scheduler.
#
# NOTE: we deliberately do NOT call ``accelerator.prepare(scheduler)``.
# Accelerate 0.23 dynamically re-binds ``AcceleratedOptimizer.step`` when
# preparing a scheduler, which then makes ``accelerator.save_state(...)`` fail
# with "Can't pickle AcceleratedOptimizer.step: it's not the same object".
# We step the scheduler ourselves only on sync-gradient iterations below,
# which gives the same end behaviour (one scheduler step per optimiser step)
# without the pickle hazard.
scheduler = None
if LR_SCHEDULE == "onecycle":
    # OneCycleLR is stepped per optimiser update, so its ``total_steps`` is
    # epochs * ceil(batches / grad_accum_steps).
    steps_per_epoch = max(1, math.ceil(len(train_dataloader) / GRAD_ACCUM_STEPS))
    total_steps = EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=WARMUP_FRAC,
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=25.0,
        final_div_factor=1e4,
    )
    print(f"OneCycleLR scheduler: total_steps={total_steps}, max_lr={LEARNING_RATE}, "
          f"pct_start={WARMUP_FRAC}  (stepped manually on sync iterations)")
elif LR_SCHEDULE == "cosine":
    # Cosine is stepped per *epoch* below, not per iteration.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 1e-2
    )
    print(f"CosineAnnealingLR scheduler: T_max={EPOCHS} epochs")


class ModelEma:
    """Exponential moving average of model weights (Polyak averaging).

    Used at validation time for a small free quality bump. Cheap to maintain
    (one extra param-sized buffer).
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        # accelerator.prepare wraps the model -- get the underlying nn.Module
        # so the deepcopy + state_dict update is a no-op compatible w/ DDP.
        base = accelerator.unwrap_model(model)
        self.module = copy.deepcopy(base).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        base = accelerator.unwrap_model(model)
        for ema_p, p in zip(self.module.parameters(), base.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.module.buffers(), base.buffers()):
            ema_b.copy_(b)


ema = ModelEma(model, decay=EMA_DECAY) if EMA_DECAY > 0 else None
if ema is not None:
    print(f"Maintaining EMA of weights with decay={EMA_DECAY}")

# Compile after accelerator.prepare so DDP/autocast wrappers are already in
# place. Shapes are fixed (batch*split, 2, 256, 128, 48), so dynamic=False
# avoids recompilation. First few iterations carry compile/autotune overhead.
# Dynamo currently bails out on some torch.utils.checkpoint internals via
# `torch.random.get_rng_state`; suppress_errors=True lets those regions fall
# back to eager while still compiling the rest of the network.
if USE_COMPILE and hasattr(torch, "compile"):
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 64
        print(f"Compiling model with torch.compile(mode={COMPILE_MODE!r}, dynamic=False)")
        model = torch.compile(model, mode=COMPILE_MODE, dynamic=False)
    except Exception as exc:
        print(f"torch.compile failed, falling back to eager: {exc}")
elif USE_COMPILE:
    print("torch.compile requested but not available in this torch version; skipping.")



def make_history():
    return {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "losses": []
    }


def load_history_if_available():
    history_paths = [
        os.path.join(model_path, "train_history_inprog.pkl"),
        model_path + "train_history_inprog.pkl",
        MODEL_NAME + "_train_history_inprog.pkl",
        MODEL_NAME + "train_history.pkl",
        "train_history.pkl",
    ]
    for history_path in history_paths:
        if os.path.exists(history_path):
            print(f"Loading training history from {history_path}")
            return load_obj(history_path)
    print("No previous training history found; starting a new history.")
    return make_history()


history = make_history()
start_epoch = 1
best_loss = np.inf
epochs_without_improvement = 0

if LOAD_MODEL:
    print(f"Loading accelerator state from {model_path}")
    accelerator.load_state(model_path)
    if scheduler is not None:
        sched_path = os.path.join(model_path, "scheduler_state.pt")
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path))
            print(f"Resumed scheduler from {sched_path}")
    history = load_history_if_available()
    if history["val_loss"]:
        start_epoch = len(history["val_loss"]) + 1
        for val_loss in history["val_loss"]:
            if val_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
                best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        print(f"Resuming at epoch {start_epoch}; best validation loss so far: {best_loss:.4f}")


# --------------------------------------------------------------------------------------



def train(epoch):
    model.train()  # Enable training mode

    pbar = tqdm(total=len(train_dataloader), position=0)
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_delay_loss = total_examples = 0

    pbar2 = tqdm(train_dataloader, leave=True, position=0)
    for i, data in enumerate(pbar2):
        # Get data - already on CPU from dataloader
        x, y = data

        # Preprocess on GPU (much faster than CPU preprocessing!)
        x, y = preprocess_data(x, y, target_device=device, pca_comps=pca_components, training=True)

        # `accelerator.accumulate(model)` handles gradient accumulation: backward
        # only syncs every GRAD_ACCUM_STEPS micro-batches and Accelerate divides
        # the loss internally so `loss` reads as the per-batch value. With
        # GRAD_ACCUM_STEPS=1 this collapses to a no-op.
        with accelerator.accumulate(model):
            with accelerator.autocast():
                preds = model(x).to(torch.float)
                pixel_loss = criterion(preds, y)
                if USE_DELAY_LOSS:
                    # Reconstruct the full arcsinh-compressed signal for the delay
                    # loss. When predicting residuals, preds is the correction and
                    # x is the PCA-cleaned input, so preds+x = full prediction.
                    preds_full = (preds + x) if PREDICT_RESIDUAL else preds
                    truth_full = (y + x)     if PREDICT_RESIDUAL else y
                    d_loss = delay_criterion(preds_full, truth_full)
                    loss = pixel_loss + LAMBDA_DELAY * d_loss
                else:
                    loss = pixel_loss
            accelerator.backward(loss)

            # Only clip / step the scheduler / step the EMA on real sync
            # iterations. accelerator.accumulate already turns optimizer.step()
            # and optimizer.zero_grad() into no-ops on accumulation steps.
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if scheduler is not None and LR_SCHEDULE == "onecycle" and accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None and accelerator.sync_gradients:
                ema.update(model)

        # Track pixel loss for history/val comparison; delay loss displayed separately.
        total_loss += float(pixel_loss)
        if USE_DELAY_LOSS:
            total_delay_loss += float(d_loss)
        total_examples += 1

        # Explicitly delete large tensors to help memory management
        if USE_DELAY_LOSS:
            del preds_full, truth_full
        del x, y, preds

        desc = "loss: %.4f" % (total_loss / total_examples)
        if USE_DELAY_LOSS:
            desc += " | delay: %.4f" % (total_delay_loss / total_examples)
        pbar2.set_description(desc)
        pbar.update(1)

    pbar.close()
    
    # Clear GPU cache
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def test(plot=False):
    model.eval()
    # Use EMA weights for validation (if maintained) -- usually a free quality
    # bump but doesn't hurt to fall back to live weights when ema is disabled.
    eval_model = ema.module if ema is not None else model

    total_loss = total_examples = 0

    for i, data in tqdm(enumerate(val_dataloader), desc="Validation"):
        # Get data
        x, y = data
        
        # Preprocess on GPU (no augmentation during validation)
        x, y = preprocess_data(x, y, target_device=device, pca_comps=pca_components, training=False)
                
        # Forward pass
        with accelerator.autocast():
            preds = eval_model(x).to(torch.float)
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


# training loop
for epoch in range(start_epoch, EPOCHS + 1):

    loss = train(epoch)
    loss = float(loss)

    val_loss = float(test())


    # Step the cosine scheduler once per epoch (OneCycle steps per iteration
    # already inside ``train``).
    if scheduler is not None and LR_SCHEDULE == "cosine":
        scheduler.step()

    if val_loss < best_loss - EARLY_STOPPING_MIN_DELTA:

        best_loss = val_loss
        epochs_without_improvement = 0
        accelerator.save_model(model, model_path)
        try:
            accelerator.save_state(model_path)
        except Exception as exc:
            print(f"  warning: accelerator.save_state failed: {exc}")
        if scheduler is not None:
            # OneCycleLR holds a reference to AcceleratedOptimizer's dynamically
            # re-bound .step in its anneal-func closure; that reference is not
            # picklable. Save only the picklable scalar attributes so we can
            # resume the schedule from `last_epoch`. If even this fails the
            # training continues -- the model weights have already been saved.
            try:
                picklable = {
                    k: v for k, v in scheduler.state_dict().items()
                    if not callable(v)
                }
                torch.save(
                    picklable,
                    os.path.join(model_path, "scheduler_state.pt"),
                )
            except Exception as exc:
                print(f"  warning: scheduler save failed (resume will start LR schedule "
                      f"from epoch 0): {exc}")
        if ema is not None:
            ema_path = os.path.join(model_path, "ema_state_dict.pt")
            torch.save(ema.module.state_dict(), ema_path)
        print(f'New best validation loss: {best_loss:.4f}')
    else:
        epochs_without_improvement += 1

    # save history
    history["train_loss"].append(loss)
    history["val_loss"].append(val_loss)

    save_obj(history, MODEL_NAME + "_train_history_inprog") # save locally
    save_obj(history, os.path.join(model_path, "train_history_inprog"))
    save_obj(history, model_path + "train_history_inprog") # legacy path

    # dump to save memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Loss: {loss:.4f} | Val loss: {val_loss:.4f}')

    if EARLY_STOPPING_PATIENCE > 0 and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f'Early stopping after {epoch} epochs: validation loss did not improve '
            f'for {EARLY_STOPPING_PATIENCE} epochs.'
        )
        break

# save the history object
save_obj(history, "train_history")
save_obj(history, MODEL_NAME + "train_history")
save_obj(history, os.path.join(model_path, "train_history"))
save_obj(configs, MODEL_NAME + "configs.json")


# should we do the validation checks here ?
