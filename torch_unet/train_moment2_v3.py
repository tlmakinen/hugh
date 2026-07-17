import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import gc
from accelerate import Accelerator
import os.path as osp
import argparse
import cloudpickle as pickle
import json
import copy
from tqdm import tqdm

from dataloader import H5Dataset
from nets import smooth_leaky
from utils import apply_precomputed_pca_fast
from nets_v2 import UNet3dV2
from visualize_reconstructions import build_model, load_model_state

# --------------------------------------------------------------------------------------

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class logMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.log(self.mse(pred, actual))


# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    configs = json.load(f)

model_params    = configs["model_params"]
moment2_params  = configs["moment2_params"]
training_params = configs["training_params"]

# ---- model architecture (shared between mean model and variance model) ----
FILTERS       = model_params["filters"]
NOISEAMP      = model_params["noiseamp"]
N_FG          = model_params["n_fg"]
MODEL_VERSION = model_params.get("model_version", "v2").lower()
PCA_COMPONENTS_PATH = model_params.get("pca_components_path", None)

# ---- variance model output location ----
MOMENT2_MODEL_NAME   = moment2_params["model_name"]
MOMENT2_MODEL_PATH   = moment2_params["model_path"]
MEAN_MODEL_CHECKPOINT = moment2_params["mean_model_checkpoint"]

# ---- training knobs ----
LEARNING_RATE         = training_params["learning_rate"]
BATCH_SIZE            = training_params["batch_size"]
EPOCHS                = int(training_params["epochs"])
GRADIENT_CLIP         = float(training_params["gradient_clip"])
EARLY_STOPPING_PATIENCE  = int(training_params.get("early_stopping_patience", 0))
EARLY_STOPPING_MIN_DELTA = float(training_params.get("early_stopping_min_delta", 0.0))
TRAIN_NUM_WORKERS     = int(training_params.get("train_num_workers", 4))
VAL_NUM_WORKERS       = int(training_params.get("val_num_workers", 2))
PREFETCH_FACTOR       = int(training_params.get("prefetch_factor", 2))
MIXED_PRECISION       = training_params.get("mixed_precision", "fp16")
GRAD_ACCUM_STEPS      = max(1, int(training_params.get("grad_accum_steps", 1)))
USE_GRAD_CHECKPOINT   = bool(training_params.get("use_grad_checkpoint", False))
OPTIMIZER_NAME        = str(training_params.get("optimizer", "adamw")).lower()
WEIGHT_DECAY          = float(training_params.get("weight_decay", 1e-4))
LR_SCHEDULE           = str(training_params.get("lr_schedule", "onecycle")).lower()
WARMUP_FRAC           = float(training_params.get("warmup_frac", 0.05))
EMA_DECAY             = float(training_params.get("ema_decay", 0.999))
SEED                  = int(training_params["seed"])
ADD_NOISE             = bool(training_params.get("add_noise", True))
RA_SHIFT_AUG          = bool(training_params.get("ra_shift_aug", True))
cosmopath             = training_params["cosmopath"]
galpath               = training_params["galpath"]

# v3 preprocessing constants -- these must match the mean model exactly
SCALING           = 1e5
TARGET_TRANSFORM  = "arcsinh"
PREDICT_RESIDUAL  = True   # always True for v3; needed so y = residual model_1 predicts

# RA split: 1024 -> 4 × 256 (matches v3 train2.py)
split = 4

# --------------------------------------------------------------------------------------

print("LOADING DATA")
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-computed PCA components
pca_components = None
if PCA_COMPONENTS_PATH and os.path.exists(PCA_COMPONENTS_PATH):
    print(f"Loading pre-computed PCA components from {PCA_COMPONENTS_PATH}")
    pca_components = torch.load(PCA_COMPONENTS_PATH, map_location=device)
    print(f"  - {pca_components['N_FG']} FG components, {pca_components['N_freq']} freq bins")
else:
    print("WARNING: No pre-computed PCA components found — PCA will be computed on-the-fly.")

cosmofiles = sorted([cosmopath + p for p in os.listdir(cosmopath)])
galfiles   = sorted([galpath   + p for p in os.listdir(galpath)])

mask    = np.random.rand(len(cosmofiles)) < 0.9
galmask = np.random.rand(len(galfiles))   < 0.9

num_train = int(training_params.get("num_train", 128))
train_cosmo_files = list(np.array(cosmofiles)[mask])[:num_train]
val_cosmo_files   = list(np.array(cosmofiles)[~mask])
train_gal_files   = list(np.array(galfiles)[galmask])
val_gal_files     = list(np.array(galfiles)[~galmask])

train_dataset = H5Dataset(train_cosmo_files, train_gal_files, use_cache=False)
val_dataset   = H5Dataset(val_cosmo_files,   val_gal_files,   use_cache=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=TRAIN_NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
    persistent_workers=TRAIN_NUM_WORKERS > 0,
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
# Preprocessing (must match train2.py v3 exactly)

def _forward_transform(t):
    return torch.arcsinh(t * SCALING)


def preprocess_data(x, y, target_device=None, pca_comps=None, training=False):
    """Identical to train2.py v3 preprocess_data."""
    if target_device is not None:
        x = x.to(target_device, non_blocking=True)
        y = y.to(target_device, non_blocking=True)

    batch_size    = x.shape[0]
    baseline_dim  = x.shape[1]
    freq_dim      = x.shape[2]
    ra_dim        = x.shape[3]
    ra_split_dim  = ra_dim // split

    if training and RA_SHIFT_AUG:
        shift = int(torch.randint(0, ra_dim, (1,)).item())
        if shift:
            x = torch.roll(x, shifts=shift, dims=-1)
            y = torch.roll(y, shifts=shift, dims=-1)

    x = x.reshape(batch_size, baseline_dim, freq_dim, split, ra_split_dim)
    y = y.reshape(batch_size, baseline_dim, freq_dim, split, ra_split_dim)
    x = x.permute(0, 3, 2, 4, 1)
    y = y.permute(0, 3, 2, 4, 1)
    x = x.reshape(batch_size * split, freq_dim, ra_split_dim, baseline_dim)
    y = y.reshape(batch_size * split, freq_dim, ra_split_dim, baseline_dim)

    x = torch.stack([x.real, x.imag], dim=-1)
    y = torch.stack([y.real, y.imag], dim=-1)

    if ADD_NOISE:
        x.add_(torch.randn_like(x) * NOISEAMP)

    x_pca_in = x.permute(0, 2, 3, 1, 4).contiguous()
    if pca_comps is not None:
        xreal = apply_precomputed_pca_fast(x_pca_in[..., 0], pca_comps)
        ximag = apply_precomputed_pca_fast(x_pca_in[..., 1], pca_comps)
    else:
        from nets import PCAclean
        xreal = PCAclean(x_pca_in[..., 0], N_FG=N_FG)[0]
        ximag = PCAclean(x_pca_in[..., 1], N_FG=N_FG)[0]
    x_cleaned = torch.stack([xreal, ximag], dim=-1)
    x = x_cleaned.permute(0, 4, 1, 3, 2).contiguous()
    y = y.permute(0, 4, 2, 1, 3).contiguous()

    x = _forward_transform(x)
    y = _forward_transform(y)

    # PREDICT_RESIDUAL is always True for v3: y becomes the residual model_1 was trained on
    y = y - x

    return x, y


# --------------------------------------------------------------------------------------
# Models

print("INITIALISING MODELS")
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

# --- Frozen first-moment (mean) model ---
print(f"Building frozen mean model from {MEAN_MODEL_CHECKPOINT}")
model_1 = build_model(model_params, device)
load_model_state(model_1, MEAN_MODEL_CHECKPOINT, device)
model_1.eval()
for p in model_1.parameters():
    p.requires_grad_(False)

# Cast to the training half-precision so the forward pass stays cheap on V100.
model_1_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(MIXED_PRECISION, torch.float32)
model_1 = model_1.to(model_1_dtype)
print(f"Mean model dtype: {model_1_dtype}, param count: {sum(p.numel() for p in model_1.parameters())/1e6:.2f}M")

# --- Fresh second-moment (variance) model ---
# build_model hardcodes use_checkpoint=False (fine for inference). For the
# trainable variance model we need gradient checkpointing enabled, so we
# instantiate UNet3dV2 directly with use_checkpoint=USE_GRAD_CHECKPOINT.
print("Building fresh variance model (same architecture, random init)")
v2_kwargs = dict(
    filters=model_params["filters"],
    levels=model_params.get("levels", 3),
    block=model_params.get("block", "preact_res"),
    n_blocks_per_stage=model_params.get("n_blocks_per_stage", 2),
    norm=model_params.get("norm", "groupnorm"),
    activation=model_params.get("activation", "silu"),
    pad_modes=tuple(model_params.get("pad_modes", ["circular", "reflect", "replicate"])),
    downsample_kind=model_params.get("downsample_kind", "blurpool"),
    upsample_kind=model_params.get("upsample_kind", "resize_conv"),
    strides=model_params.get("strides", None),
    attention_gates=bool(model_params.get("attention_gates", False)),
    use_se=bool(model_params.get("use_se", False)),
    factored_convs=bool(model_params.get("factored_convs", False)),
    film_conditioning=bool(model_params.get("film_conditioning", False)),
    film_cond_dim=int(model_params.get("film_cond_dim", 8)),
    freq_pos_encoding=bool(model_params.get("freq_pos_encoding", False)),
    freq_pos_dim=int(model_params.get("freq_pos_dim", 2)),
    two_conv_head=bool(model_params.get("two_conv_head", True)),
    norm_groups=int(model_params.get("norm_groups", 8)),
    use_checkpoint=USE_GRAD_CHECKPOINT,
)
print(f"Variance model kwargs: {v2_kwargs}")
model = UNet3dV2(**v2_kwargs).to(device)
print(f"Variance model param count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# --------------------------------------------------------------------------------------

criterion = logMSELoss()

if OPTIMIZER_NAME == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model_out_path = os.path.join(MOMENT2_MODEL_PATH, MOMENT2_MODEL_NAME)
os.makedirs(model_out_path, exist_ok=True)

accelerator = Accelerator(
    project_dir=model_out_path,
    mixed_precision=MIXED_PRECISION,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
)
if GRAD_ACCUM_STEPS > 1:
    print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} micro-batches "
          f"(effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")

model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader)

scheduler = None
if LR_SCHEDULE == "onecycle":
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
    print(f"OneCycleLR: total_steps={total_steps}, max_lr={LEARNING_RATE}")
elif LR_SCHEDULE == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 1e-2
    )


class ModelEma:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
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
    print(f"EMA on variance model: decay={EMA_DECAY}")

# --------------------------------------------------------------------------------------

def make_history():
    return {"train_loss": [], "val_loss": []}

history   = make_history()
best_loss = np.inf
epochs_without_improvement = 0


# --------------------------------------------------------------------------------------

def train(epoch):
    model.train()
    total_loss = total_examples = 0
    pbar = tqdm(train_dataloader, leave=True, position=0, desc=f"Epoch {epoch:04d} train")

    for data in pbar:
        x, y = data
        x, y = preprocess_data(x, y, target_device=device, pca_comps=pca_components, training=True)

        with accelerator.accumulate(model):
            # Compute variance target: squared residual of the frozen mean model.
            # All within no_grad — model_1 is frozen and its output is a constant
            # with respect to the variance model's parameters.
            with torch.no_grad():
                mean_pred  = model_1(x.to(model_1_dtype)).float()
                var_target = (mean_pred - y).pow(2).detach()

            with accelerator.autocast():
                var_pred = model(x).float()
                loss = criterion(var_pred, var_target)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if scheduler is not None and LR_SCHEDULE == "onecycle" and accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None and accelerator.sync_gradients:
                ema.update(model)

        total_loss     += float(loss)
        total_examples += 1
        del x, y, mean_pred, var_target, var_pred

        pbar.set_description(f"Epoch {epoch:04d} | loss: {total_loss/total_examples:.4f}")

    gc.collect()
    torch.cuda.empty_cache()
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    eval_model = ema.module if ema is not None else model
    total_loss = total_examples = 0

    for data in tqdm(val_dataloader, desc="Validation", leave=False):
        x, y = data
        x, y = preprocess_data(x, y, target_device=device, pca_comps=pca_components, training=False)

        mean_pred  = model_1(x.to(model_1_dtype)).float()
        var_target = (mean_pred - y).pow(2)

        with accelerator.autocast():
            var_pred = eval_model(x).float()
            loss = criterion(var_pred, var_target)

        total_loss     += float(loss)
        total_examples += 1

    return total_loss / total_examples


# --------------------------------------------------------------------------------------
print("STARTING TRAINING LOOP")
gc.collect()

for epoch in range(1, EPOCHS + 1):
    loss     = float(train(epoch))
    val_loss = float(test())

    if scheduler is not None and LR_SCHEDULE == "cosine":
        scheduler.step()

    if val_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
        best_loss = val_loss
        epochs_without_improvement = 0

        accelerator.save_model(model, model_out_path)
        try:
            accelerator.save_state(model_out_path)
        except Exception as exc:
            print(f"  warning: accelerator.save_state failed: {exc}")

        if scheduler is not None:
            try:
                picklable = {k: v for k, v in scheduler.state_dict().items() if not callable(v)}
                torch.save(picklable, os.path.join(model_out_path, "scheduler_state.pt"))
            except Exception as exc:
                print(f"  warning: scheduler save failed: {exc}")

        if ema is not None:
            ema_path = os.path.join(model_out_path, "ema_state_dict.pt")
            torch.save(ema.module.state_dict(), ema_path)

        print(f"  New best val loss: {best_loss:.4f}")
    else:
        epochs_without_improvement += 1

    history["train_loss"].append(loss)
    history["val_loss"].append(val_loss)
    save_obj(history, os.path.join(model_out_path, "train_history_inprog"))
    save_obj(history, MOMENT2_MODEL_NAME + "_train_history_inprog")

    gc.collect()
    torch.cuda.empty_cache()
    print(f"Epoch {epoch}/{EPOCHS} | train: {loss:.4f} | val: {val_loss:.4f}")

    if EARLY_STOPPING_PATIENCE > 0 and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
        break

save_obj(history, os.path.join(model_out_path, "train_history"))
save_obj(history, MOMENT2_MODEL_NAME + "_train_history")
save_obj(configs,  MOMENT2_MODEL_NAME + "_configs.json")
