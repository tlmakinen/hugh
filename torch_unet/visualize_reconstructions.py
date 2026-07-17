"""Plot 2D reconstruction panels for a trained train2.py checkpoint.

Supports both the legacy ``UNet3d`` (``model_version: v1``) and the configurable
``UNet3dV2`` (``model_version: v2``). Also handles the optional v3 training-side
options: arcsinh target compression and residual-over-PCA prediction.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nets import BasicBlock, PCALayer, UNet3d, smooth_leaky
from nets_v2 import UNet3dV2
from utils import apply_precomputed_pca_fast


def load_vis(path):
    import h5py

    with h5py.File(path, "r") as handle:
        return torch.tensor(np.array(handle["/vis/"]), dtype=torch.complex64)


def load_model_state(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    # Handle checkpoints saved from wrapped/distributed modules.
    state = {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")


def build_model(model_params, device):
    """Build either v1 (UNet3d) or v2 (UNet3dV2) from a config dict."""
    version = model_params.get("model_version", "v1").lower()
    if version == "v2":
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
            use_checkpoint=False,
            conv_kernel_size=model_params.get("conv_kernel_size", 3),
        )
        print(f"Building UNet3dV2: {v2_kwargs}")
        return UNet3dV2(**v2_kwargs).to(device)
    act = smooth_leaky if model_params["act"] == "smooth_leaky" else torch.nn.SiLU
    return UNet3d(BasicBlock, filters=model_params["filters"], act=act, use_checkpoint=False).to(device)


def forward_transform(t, scaling, kind):
    if kind == "arcsinh":
        return torch.arcsinh(t * scaling)
    return t * scaling


def inverse_transform(t, scaling, kind):
    if kind == "arcsinh":
        return torch.sinh(t) / scaling
    return t / scaling


def preprocess_data(
    x,
    y,
    split,
    n_fg,
    noise_amp,
    add_noise,
    pca_components,
    device,
    scaling=1e5,
    target_transform="linear",
):
    """Return ``(x_input_transformed, truth_in_physical_units, pca_cleaned_transformed)``.

    Layout is ``(batch*split, Re/Im, ra/split, freq, baseline)``, matching
    train2.py. PCA is applied along the frequency axis. The truth tensor is
    returned in **physical units** (no scaling, no arcsinh) so the caller can
    inverse-transform the prediction and compute residuals in the right units.
    The PCA-cleaned tensor is returned in transformed space (same as the model
    input) so we can compute ``model_out + pca_cleaned`` for the residual head.
    """
    x = x.to(device)
    y = y.to(device)

    batch_size, baseline_dim, freq_dim, ra_dim = x.shape
    ra_split_dim = ra_dim // split

    x = x.reshape(batch_size, baseline_dim, freq_dim, split, ra_split_dim)
    y = y.reshape(batch_size, baseline_dim, freq_dim, split, ra_split_dim)

    x = x.permute(0, 3, 2, 4, 1)
    y = y.permute(0, 3, 2, 4, 1)

    x = x.reshape(batch_size * split, freq_dim, ra_split_dim, baseline_dim)
    y = y.reshape(batch_size * split, freq_dim, ra_split_dim, baseline_dim)

    x = torch.stack([x.real, x.imag], dim=-1)
    y = torch.stack([y.real, y.imag], dim=-1)

    if add_noise:
        x = x + torch.randn_like(x) * noise_amp

    x_pca_in = x.permute(0, 2, 3, 1, 4).contiguous()
    if pca_components is not None:
        xreal = apply_precomputed_pca_fast(x_pca_in[..., 0], pca_components)
        ximag = apply_precomputed_pca_fast(x_pca_in[..., 1], pca_components)
    else:
        xreal = x_pca_in[..., 0]
        ximag = x_pca_in[..., 1]
    x_cleaned = torch.stack([xreal, ximag], dim=-1)

    x = x_cleaned.permute(0, 4, 1, 3, 2).contiguous()
    y = y.permute(0, 4, 2, 1, 3).contiguous()

    x_input = forward_transform(x, scaling, target_transform)
    return x_input, y, x_input


def choose_files(configs, sample_idx, foreground_idx):
    cosmopath = configs["training_params"]["cosmopath"]
    galpath = configs["training_params"]["galpath"]

    cosmofiles = sorted(os.path.join(cosmopath, name) for name in os.listdir(cosmopath))
    galfiles = sorted(os.path.join(galpath, name) for name in os.listdir(galpath))

    return cosmofiles[sample_idx % len(cosmofiles)], galfiles[foreground_idx % len(galfiles)]


def select_plane(tensor, plane, chunk, component, baseline, ra, freq):
    if plane == "ra_freq":
        image = tensor[chunk, component, :, :, baseline]
        xlabel, ylabel = "Frequency bin", "RA bin"
    elif plane == "baseline_freq":
        image = tensor[chunk, component, ra, :, :].T
        xlabel, ylabel = "Frequency bin", "Baseline"
    elif plane == "ra_baseline":
        image = tensor[chunk, component, :, freq, :]
        xlabel, ylabel = "Baseline", "RA bin"
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return image.detach().cpu().float().numpy(), xlabel, ylabel


def plot_panels(preds, truth, args, output_path):
    """preds, truth are both in physical (un-scaled) units."""
    residual = preds - truth

    images = []
    labels = [("truth", truth), ("network prediction", preds), ("residual", residual)]
    for title, tensor in labels:
        image, xlabel, ylabel = select_plane(
            tensor, args.plane, args.chunk, args.component, args.baseline, args.ra, args.freq,
        )
        images.append((title, image, xlabel, ylabel))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (title, image, xlabel, ylabel) in zip(axes, images):
        im = axis.imshow(image, aspect="auto", origin="lower")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 2D reconstruction panels from a trained train2.py checkpoint.")
    parser.add_argument("--config", default="config_train2_overnight.json")
    parser.add_argument("--checkpoint", default=None, help="Defaults to <model_path>/<model_name>/pytorch_model.bin")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA state dict if present")
    parser.add_argument("--pca-components", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--foreground-idx", type=int, default=0)
    parser.add_argument("--cosmo-file", default=None)
    parser.add_argument("--gal-file", default=None)
    parser.add_argument("--plane", choices=["ra_freq", "baseline_freq", "ra_baseline"], default="ra_freq")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--component", type=int, default=0, help="0=real, 1=imag")
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--ra", type=int, default=0)
    parser.add_argument("--freq", type=int, default=0)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--scaling", type=float, default=1e5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as handle:
        configs = json.load(handle)

    model_params = configs["model_params"]
    training_params = configs["training_params"]
    target_transform = str(training_params.get("target_transform", "linear")).lower()
    predict_residual = bool(training_params.get("predict_residual", False))

    model_dir = os.path.join(model_params["model_dir"], model_params["model_name"])
    if args.use_ema:
        checkpoint_path = args.checkpoint or os.path.join(model_dir, "ema_state_dict.pt")
    else:
        checkpoint_path = args.checkpoint or os.path.join(model_dir, "pytorch_model.bin")
    pca_path = (
        args.pca_components
        or model_params.get("pca_components_path")
        or os.path.join(model_dir, f"pca_components_nfg{model_params['n_fg']}.pt")
    )
    output_path = args.output or os.path.join(model_dir, f"reconstruction_{args.plane}.png")

    device = torch.device(args.device)
    pca_components = torch.load(pca_path, map_location=device)

    model = build_model(model_params, device)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(model_dtype)
    print(f"Loading checkpoint: {checkpoint_path}")
    load_model_state(model, checkpoint_path, device)
    model.eval()

    cosmo_file = args.cosmo_file
    gal_file = args.gal_file
    if cosmo_file is None or gal_file is None:
        cosmo_file, gal_file = choose_files(configs, args.sample_idx, args.foreground_idx)

    cosmo = load_vis(cosmo_file)[..., :1024].unsqueeze(0)
    gal = load_vis(gal_file)[..., :1024].unsqueeze(0)
    x = gal + cosmo
    y = cosmo

    with torch.no_grad():
        x_in_t, truth_phys, pca_cleaned_t = preprocess_data(
            x, y,
            split=args.split,
            n_fg=model_params["n_fg"],
            noise_amp=model_params["noiseamp"],
            add_noise=training_params["add_noise"],
            pca_components=pca_components,
            device=device,
            scaling=args.scaling,
            target_transform=target_transform,
        )
        preds_t = model(x_in_t.to(model_dtype)).to(torch.float)
        if predict_residual:
            preds_t = preds_t + pca_cleaned_t.to(preds_t.dtype)
        preds = inverse_transform(preds_t, args.scaling, target_transform)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plot_panels(preds, truth_phys, args, output_path)
    print(f"Saved reconstruction plot to {output_path}")
    print(f"Cosmology file: {cosmo_file}")
    print(f"Foreground file: {gal_file}")


if __name__ == "__main__":
    main()
