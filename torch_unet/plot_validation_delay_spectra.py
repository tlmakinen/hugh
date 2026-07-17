import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import constants as consts
from astropy import units
from astropy.cosmology import FlatLambdaCDM
from scipy.signal.windows import blackmanharris

from nets import BasicBlock, UNet3d, smooth_leaky
from visualize_reconstructions import (
    build_model,
    forward_transform,
    inverse_transform,
    load_model_state,
    preprocess_data,
)


COSMOLOGY = FlatLambdaCDM(H0=100, Om0=0.3)
NU_21 = 1420.405751768 * units.MHz


def load_vis_and_baselines(path):
    with h5py.File(path, "r") as handle:
        vis = torch.tensor(np.array(handle["/vis/"]), dtype=torch.complex64)
        baselines = np.array(handle["/index_map/baselines/"])
    return vis, baselines


def load_vis(path):
    with h5py.File(path, "r") as handle:
        return torch.tensor(np.array(handle["/vis/"]), dtype=torch.complex64)


def validation_files(configs, sample_idx, foreground_idx):
    rng = np.random.RandomState(int(configs["training_params"]["seed"]))
    cosmopath = configs["training_params"]["cosmopath"]
    galpath = configs["training_params"]["galpath"]

    cosmofiles = [os.path.join(cosmopath, name) for name in os.listdir(cosmopath)]
    galfiles = [os.path.join(galpath, name) for name in os.listdir(galpath)]

    cosmo_mask = rng.rand(len(cosmofiles)) < 0.9
    val_cosmo = list(np.array(cosmofiles)[~cosmo_mask])

    gal_mask = rng.rand(len(galfiles)) < 0.9
    val_gal = list(np.array(galfiles)[~gal_mask])
    if not val_gal:
        val_gal = galfiles

    return val_cosmo[sample_idx % len(val_cosmo)], val_gal[foreground_idx % len(val_gal)]


def z21(freq):
    freq = units.Quantity(freq, unit=units.MHz)
    return (NU_21 / freq - 1).value


def k_perpendicular(bl_length, freq):
    freq = units.Quantity(freq, unit=units.MHz)
    bl_length = units.Quantity(bl_length, unit=units.m)
    wl = freq.to("m", equivalencies=units.spectral())
    k_pp = 2 * np.pi * np.linalg.norm(bl_length) / wl / COSMOLOGY.comoving_transverse_distance(z21(freq))
    return k_pp.to("Mpc^-1")


def delays_from_freqs(freqs):
    freqs = units.Quantity(freqs, unit=units.MHz)
    d_freq = (freqs[1] - freqs[0]).to("ns^-1")
    return np.fft.fftshift(np.fft.fftfreq(len(freqs), d=d_freq))


def k_parallel(delays, freq):
    freq = units.Quantity(freq, unit=units.MHz)
    z = z21(freq)
    eta_to_kh = 2 * np.pi * NU_21 * COSMOLOGY.H(z) / consts.c / (1 + z) ** 2
    return (eta_to_kh * delays).to("Mpc^-1")


def wedge(theta_0, bl_ind, baselines, band_center):
    z = z21(band_center)
    bl = baselines[bl_ind, ...] * units.m
    k_pp = k_perpendicular(np.linalg.norm(bl), band_center)
    k_par = theta_0.to("rad").value * k_pp * (
        COSMOLOGY.H(z) * COSMOLOGY.comoving_transverse_distance(z) / (consts.c * (1 + z))
    ).to("").value
    return k_par * 0.7


def baseline_delay_spectrum(data, bl_ind, freqs, band_center):
    data = data[bl_ind]
    window = blackmanharris(data.shape[0])
    delays = delays_from_freqs(freqs)
    k_par = k_parallel(delays, band_center)
    delay_spec = np.fft.fftshift(np.abs(np.fft.fft(data * window[:, None], axis=0)) ** 2) * units.K ** 2
    return delays, k_par, delay_spec


def tensor_to_complex_cube(tensor, split, scaling):
    """Convert a (batch*split, Re/Im, ra/split, freq, baseline) tensor into a
    complex (baseline, freq, RA) cube in original units (divided by ``scaling``).
    Leading axis is assumed to be ``batch*split`` with batch=1 in this plotting
    context, so split chunks collapse along RA with split-major ordering
    (matching the reshape used in ``preprocess_data``).
    """
    tensor = tensor.detach().cpu().float().numpy() / scaling
    # Source axes: (split, Re/Im, ra/split, freq, baseline)
    #              (0,     1,     2,        3,    4)
    # Target axes: (baseline, freq, split, ra/split, Re/Im)
    # Permutation: (4, 3, 0, 2, 1)
    permuted = tensor.transpose(4, 3, 0, 2, 1)
    baseline_dim = permuted.shape[0]
    freq_dim = permuted.shape[1]
    ra_dim = permuted.shape[2] * permuted.shape[3]
    cube = permuted.reshape(baseline_dim, freq_dim, ra_dim, 2)
    return cube[..., 0] + 1j * cube[..., 1]


def plot_image_panels(cubes, bl_ind, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    panels = [
        ("truth", cubes["truth"]),
        ("PCA cleaned", cubes["pca"]),
        ("UNet", cubes["unet"]),
        ("UNet residual", cubes["unet"] - cubes["truth"]),
    ]
    for axis, (title, cube) in zip(axes, panels):
        image = cube[bl_ind].real
        im = axis.imshow(image, aspect="auto", origin="lower")
        axis.set_title(title)
        axis.set_xlabel("RA bin")
        axis.set_ylabel("Frequency bin")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_delay_spectrum(data, bl_ind, baselines, freqs, output_path, title, band_center, wedge_angle, rel_range):
    _, k_par, delay_spec = baseline_delay_spectrum(data, bl_ind, freqs, band_center)
    ras = np.linspace(180, -180, delay_spec.shape[1] + 1)
    extent = (ras[0], ras[-1], k_par[0].value, k_par[-1].value)
    to_plot = np.log10(np.maximum(delay_spec.value[:, ::-1], 1e-30))
    vmin = to_plot.max() - rel_range

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(to_plot, vmin=vmin, aspect=100, extent=extent, origin="lower", cmap="magma")
    wedge_k = wedge(wedge_angle, bl_ind, baselines, band_center)
    ax.axhline(wedge_k.value, color="white", ls="--")
    ax.axhline(-wedge_k.value, color="white", ls="--")
    ax.set_title(title)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel(r"$k_{\parallel}$ [h Mpc$^{-1}$]")
    fig.colorbar(im, ax=ax, label=r"log$_{10}$ [K$^2$]")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def delay_spectrum_ratio(data, truth, bl_ind, freqs, band_center, residual_power):
    _, k_par, pred_spec = baseline_delay_spectrum(data, bl_ind, freqs, band_center)
    _, _, truth_spec = baseline_delay_spectrum(truth, bl_ind, freqs, band_center)
    if residual_power:
        _, _, resid_spec = baseline_delay_spectrum(data - truth, bl_ind, freqs, band_center)
        ratio = resid_spec / truth_spec
    else:
        ratio = (pred_spec - truth_spec) / truth_spec
    return ratio, k_par


def plot_ratio_2d(data, truth, bl_ind, baselines, freqs, output_path, title, band_center, wedge_angle, residual_power):
    ratio, k_par = delay_spectrum_ratio(data, truth, bl_ind, freqs, band_center, residual_power)
    ras = np.linspace(180, -180, ratio.shape[1] + 1)
    extent = (ras[0], ras[-1], k_par[0].value, k_par[-1].value)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(ratio.value[:, ::-1], aspect=100, vmin=-2, vmax=2, extent=extent, origin="lower", cmap="coolwarm")
    wedge_k = wedge(wedge_angle, bl_ind, baselines, band_center)
    ax.axhline(wedge_k.value, color="white", ls="--")
    ax.axhline(-wedge_k.value, color="white", ls="--")
    ax.set_title(title)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel(r"$k_{\parallel}$ [h Mpc$^{-1}$]")
    fig.colorbar(im, ax=ax, label="relative error")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_ratio_1d(cubes, bl_ind, freqs, output_path, band_center, residual_power):
    pca_ratio, k_par = delay_spectrum_ratio(cubes["pca"], cubes["truth"], bl_ind, freqs, band_center, residual_power)
    unet_ratio, _ = delay_spectrum_ratio(cubes["unet"], cubes["truth"], bl_ind, freqs, band_center, residual_power)

    nyquist = len(k_par) // 2
    pca_y  = np.mean(pca_ratio.value,  axis=-1)[nyquist:]
    unet_y = np.mean(unet_ratio.value, axis=-1)[nyquist:]
    if not residual_power:
        pca_y  = np.abs(pca_y)
        unet_y = np.abs(unet_y)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(k_par[nyquist:].value, pca_y,  label="PCA")
    ax.plot(k_par[nyquist:].value, unet_y, label="UNet")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_{\parallel}$ [h Mpc$^{-1}$]")
    ylabel = (r"$P(\mathrm{pred}-\mathrm{truth}) / P_\mathrm{truth}$"
              if residual_power else
              r"$|P_\mathrm{pred}-P_\mathrm{truth}| / P_\mathrm{truth}$")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.0)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_predictions(args, configs):
    model_params = configs["model_params"]
    training_params = configs["training_params"]
    target_transform = str(training_params.get("target_transform", "linear")).lower()
    predict_residual = bool(training_params.get("predict_residual", False))

    model_dir = os.path.join(model_params["model_dir"], model_params["model_name"])
    if args.use_ema:
        checkpoint_path = args.checkpoint or os.path.join(model_dir, "ema_state_dict.pt")
    else:
        checkpoint_path = args.checkpoint or os.path.join(model_dir, "pytorch_model.bin")
    pca_path = args.pca_components or model_params.get("pca_components_path") or os.path.join(
        model_dir, f"pca_components_nfg{model_params['n_fg']}.pt"
    )

    cosmo_file = args.cosmo_file
    gal_file = args.gal_file
    if cosmo_file is None or gal_file is None:
        cosmo_file, gal_file = validation_files(configs, args.sample_idx, args.foreground_idx)

    cosmo, baselines = load_vis_and_baselines(cosmo_file)
    gal = load_vis(gal_file)
    cosmo = cosmo[..., : args.ra_bins].unsqueeze(0)
    gal = gal[..., : args.ra_bins].unsqueeze(0)

    device = torch.device(args.device)
    pca_components = torch.load(pca_path, map_location=device)
    model = build_model(model_params, device)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(model_dtype)
    print(f"Loading checkpoint: {checkpoint_path}")
    load_model_state(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        # preprocess_data now returns (x_input_transformed, truth_physical_units,
        # pca_cleaned_transformed). truth is already in physical units.
        x_in_t, truth_phys, pca_cleaned_t = preprocess_data(
            gal + cosmo,
            cosmo,
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
        preds_phys = inverse_transform(preds_t, args.scaling, target_transform)
        pca_cleaned_phys = inverse_transform(pca_cleaned_t, args.scaling, target_transform)

    # tensor_to_complex_cube divides by ``scaling``; we already inverse-
    # transformed to physical units, so pass scaling=1 to avoid double-undo.
    cubes = {
        "pca": tensor_to_complex_cube(pca_cleaned_phys, args.split, 1.0),
        "unet": tensor_to_complex_cube(preds_phys, args.split, 1.0),
        "truth": tensor_to_complex_cube(truth_phys, args.split, 1.0),
    }
    return cubes, baselines, cosmo_file, gal_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate validation data and plot reconstruction delay spectra.")
    parser.add_argument("--config", default="config_train2_overnight.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use-ema", action="store_true", help="Load EMA state dict if present")
    parser.add_argument("--pca-components", default=None)
    parser.add_argument("--output-dir", default="plots/validation_delay_spectra")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--foreground-idx", type=int, default=0)
    parser.add_argument("--cosmo-file", default=None)
    parser.add_argument("--gal-file", default=None)
    parser.add_argument("--baseline", type=int, default=9)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--ra-bins", type=int, default=1024)
    parser.add_argument("--scaling", type=float, default=1e5)
    parser.add_argument("--freq-min", type=float, default=500.0)
    parser.add_argument("--freq-max", type=float, default=600.0)
    parser.add_argument("--band-center", type=float, default=550.0)
    parser.add_argument("--wedge-angle", type=float, default=90.0)
    parser.add_argument("--rel-range", type=float, default=12.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as handle:
        configs = json.load(handle)

    os.makedirs(args.output_dir, exist_ok=True)
    cubes, baselines, cosmo_file, gal_file = make_predictions(args, configs)
    freqs_edges = np.linspace(args.freq_min, args.freq_max, cubes["truth"].shape[1] + 1)
    freqs = freqs_edges[:-1] + np.diff(freqs_edges[:2])[0] / 2
    band_center = args.band_center * units.MHz
    wedge_angle = args.wedge_angle * units.deg

    plot_image_panels(cubes, args.baseline, os.path.join(args.output_dir, "validation_reconstruction_panels.png"))
    for name, title in [("truth", "Truth delay spectrum"), ("pca", "PCA-cleaned delay spectrum"), ("unet", "UNet delay spectrum")]:
        plot_delay_spectrum(
            cubes[name],
            args.baseline,
            baselines,
            freqs,
            os.path.join(args.output_dir, f"delay_spectrum_{name}.png"),
            title,
            band_center,
            wedge_angle,
            args.rel_range,
        )

    plot_ratio_2d(
        cubes["unet"],
        cubes["truth"],
        args.baseline,
        baselines,
        freqs,
        os.path.join(args.output_dir, "delay_spectrum_unet_residual_ratio_2d.png"),
        "UNet residual delay-spectrum ratio",
        band_center,
        wedge_angle,
        residual_power=True,
    )
    plot_ratio_2d(
        cubes["pca"],
        cubes["truth"],
        args.baseline,
        baselines,
        freqs,
        os.path.join(args.output_dir, "delay_spectrum_pca_residual_ratio_2d.png"),
        "PCA residual delay-spectrum ratio",
        band_center,
        wedge_angle,
        residual_power=True,
    )
    plot_ratio_1d(
        cubes,
        args.baseline,
        freqs,
        os.path.join(args.output_dir, "delay_spectrum_residual_ratio_1d.png"),
        band_center,
        residual_power=True,
    )
    plot_ratio_1d(
        cubes,
        args.baseline,
        freqs,
        os.path.join(args.output_dir, "delay_spectrum_power_error_1d.png"),
        band_center,
        residual_power=False,
    )

    with open(os.path.join(args.output_dir, "inputs.txt"), "w") as handle:
        handle.write(f"config: {args.config}\n")
        handle.write(f"checkpoint: {args.checkpoint or 'default'}\n")
        handle.write(f"cosmo_file: {cosmo_file}\n")
        handle.write(f"gal_file: {gal_file}\n")
        handle.write(f"baseline: {args.baseline}\n")
    print(f"Saved validation delay-spectrum plots to {args.output_dir}")
    print(f"Cosmology file: {cosmo_file}")
    print(f"Foreground file: {gal_file}")


if __name__ == "__main__":
    main()
