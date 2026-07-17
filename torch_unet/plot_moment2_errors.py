"""Plot moment1 1D delay-spectrum residual with moment2 ±1σ shaded uncertainty bands.

Uncertainty propagation:
  1. moment2 gives σ_arcsinh per pixel (in arcsinh residual space).
  2. Chain rule through sinh: σ_vis = cosh(mean_arcsinh) / scaling * σ_arcsinh
     gives 1σ in physical visibility units.
  3. Triangle inequality in FFT amplitude space:
       P_upper = (|FFT(vis_mean)| + |FFT(σ_vis)|)²
       P_lower = max(|FFT(vis_mean)| - |FFT(σ_vis)|, 0)²
     guarantees P_lower ≤ P_mean ≤ P_upper at every (delay, RA) bin.

Two-panel figure:
  Top  (log y)    : |P_pred − P_truth| / P_truth  + ±1σ band
  Bottom (linear y): signed (P − P_truth) / P_truth — shows where P_lower < P_truth

Usage:
    python plot_moment2_errors.py --config config_moment2_v3_v100.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import units
from scipy.signal.windows import blackmanharris

from visualize_reconstructions import (
    build_model,
    inverse_transform,
    load_model_state,
    preprocess_data,
)
from plot_validation_delay_spectra import (
    delays_from_freqs,
    k_parallel,
    load_vis,
    load_vis_and_baselines,
    tensor_to_complex_cube,
    validation_files,
)


# ---------------------------------------------------------------------------

def make_predictions(args, configs):
    model_params   = configs["model_params"]
    moment2_params = configs["moment2_params"]

    device = torch.device(args.device)
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    cosmo_file, gal_file = validation_files(configs, args.sample_idx, args.foreground_idx)
    cosmo, baselines = load_vis_and_baselines(cosmo_file)
    gal = load_vis(gal_file)
    cosmo = cosmo[..., : args.ra_bins].unsqueeze(0)
    gal   = gal[...,   : args.ra_bins].unsqueeze(0)

    pca_components = torch.load(model_params["pca_components_path"], map_location=device)

    # Mean model (frozen v3 EMA)
    mean_ckpt = moment2_params["mean_model_checkpoint"]
    model_1 = build_model(model_params, device)
    model_1.to(dtype)
    load_model_state(model_1, mean_ckpt, device)
    model_1.eval()
    print(f"Loaded mean model: {mean_ckpt}")

    # Variance model (moment2 EMA)
    var_ckpt = os.path.join(
        moment2_params["model_path"],
        moment2_params["model_name"],
        "ema_state_dict.pt",
    )
    model_2 = build_model(model_params, device)
    model_2.to(dtype)
    load_model_state(model_2, var_ckpt, device)
    model_2.eval()
    print(f"Loaded variance model: {var_ckpt}")

    with torch.no_grad():
        # preprocess_data returns (x_arcsinh, truth_physical, x_arcsinh)
        x_in_t, truth_phys, pca_cleaned_t = preprocess_data(
            gal + cosmo,
            cosmo,
            split=args.split,
            n_fg=model_params["n_fg"],
            noise_amp=model_params["noiseamp"],
            add_noise=False,
            pca_components=pca_components,
            device=device,
            scaling=args.scaling,
            target_transform="arcsinh",
        )

        # Mean model: residual in arcsinh space → full arcsinh prediction
        mu_resid     = model_1(x_in_t.to(dtype)).to(torch.float)
        mean_arcsinh = mu_resid + pca_cleaned_t   # (B*split, 2, ra/split, freq, baseline)

        # Variance model: predicted variance → 1σ in arcsinh residual space
        var_pred      = model_2(x_in_t.to(dtype)).to(torch.float)
        sigma_arcsinh = var_pred.abs().sqrt()

        # Propagate σ to physical visibility space via chain rule through sinh:
        #   d/dx [sinh(x) / s] = cosh(x) / s
        # σ_vis[pixel] = cosh(mean_arcsinh[pixel]) / scaling * σ_arcsinh[pixel]
        sigma_vis = torch.cosh(mean_arcsinh) / args.scaling * sigma_arcsinh

        # Mean prediction in physical visibility units
        mean_phys = inverse_transform(mean_arcsinh, args.scaling, "arcsinh")

    # All outputs in physical units; scaling=1 avoids double-undo in tensor_to_complex_cube
    cubes = {
        "truth":     tensor_to_complex_cube(truth_phys, args.split, 1.0),
        "mean":      tensor_to_complex_cube(mean_phys,  args.split, 1.0),
        "sigma_vis": tensor_to_complex_cube(sigma_vis,  args.split, 1.0),
    }
    return cubes, baselines, cosmo_file, gal_file


# ---------------------------------------------------------------------------

def delay_amplitude(data_bl, freqs, band_center):
    """FFT amplitude |FFT(vis * window)| along the frequency axis.

    Args:
        data_bl: complex array (freq, RA)
    Returns:
        k_par: astropy Quantity (n_delay,)
        amp:   real array (n_delay, n_RA)
    """
    window = blackmanharris(data_bl.shape[0])
    delays = delays_from_freqs(freqs)
    k_par  = k_parallel(delays, band_center)
    amp    = np.fft.fftshift(np.abs(np.fft.fft(data_bl * window[:, None], axis=0)))
    return k_par, amp


def plot_moment2_1d(cubes, bl_ind, freqs, band_center, output_path):
    """Two-panel 1D delay spectrum residual with linearised ±1σ uncertainty band.

    The band is derived via the triangle inequality in FFT amplitude space:
        P_upper = (A_mean + A_sigma)^2
        P_lower = max(A_mean - A_sigma, 0)^2
    This guarantees P_lower ≤ P_mean ≤ P_upper at every bin.
    """
    truth_bl     = cubes["truth"][bl_ind]       # (freq, RA) complex
    mean_bl      = cubes["mean"][bl_ind]
    sigma_vis_bl = cubes["sigma_vis"][bl_ind]

    k_par, amp_truth = delay_amplitude(truth_bl,     freqs, band_center)
    _,     amp_mean  = delay_amplitude(mean_bl,      freqs, band_center)
    _,     amp_sigma = delay_amplitude(sigma_vis_bl, freqs, band_center)

    # Power from amplitudes — triangle inequality guarantees containment
    P_truth = amp_truth ** 2
    P_mean  = amp_mean  ** 2
    P_upper = (amp_mean + amp_sigma) ** 2
    P_lower = np.maximum(amp_mean - amp_sigma, 0.0) ** 2

    # Signed fractional residuals; eps avoids /0 in noise-free truth modes
    eps = 1e-30
    r_mean  = (P_mean  - P_truth) / (P_truth + eps)
    r_upper = (P_upper - P_truth) / (P_truth + eps)
    r_lower = (P_lower - P_truth) / (P_truth + eps)
    # By construction: r_lower <= r_mean <= r_upper at every (delay, RA) bin.

    # Average over RA, restrict to positive delays
    nyq = len(k_par) // 2
    k   = k_par[nyq:].value
    r_mean_1d  = np.mean(r_mean,  axis=-1)[nyq:]
    r_upper_1d = np.mean(r_upper, axis=-1)[nyq:]
    r_lower_1d = np.mean(r_lower, axis=-1)[nyq:]
    # After RA-averaging r_lower_1d ≤ r_mean_1d ≤ r_upper_1d still holds.

    # Absolute values for log panel — r_mean is between r_lower and r_upper
    # so |r_mean| lies within [min(|r_lower|,|r_mean|,|r_upper|),
    #                           max(|r_lower|,|r_mean|,|r_upper|)].
    # (This would NOT be valid without the amplitude-propagation step above.)
    abs_mean  = np.abs(r_mean_1d)
    abs_upper = np.abs(r_upper_1d)
    abs_lower = np.abs(r_lower_1d)
    log_floor = 1e-8
    shade_lo_log = np.maximum(np.minimum.reduce([abs_mean, abs_lower, abs_upper]), log_floor)
    shade_hi_log = np.maximum.reduce([abs_mean, abs_lower, abs_upper])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6), constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ---- Top: log |ΔP/P| ----
    ax_top.fill_between(k, shade_lo_log, shade_hi_log,
                        alpha=0.35, label=r"$\pm 1\sigma$ (moment2)")
    ax_top.plot(k, abs_mean, lw=1.8, label="UNet mean (moment1)")
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$|P_\mathrm{pred} - P_\mathrm{truth}| \,/\, P_\mathrm{truth}$")
    ax_top.legend(framealpha=0.0)

    # ---- Bottom: linear signed ΔP/P ----
    ax_bot.fill_between(k, r_lower_1d, r_upper_1d, alpha=0.35)
    ax_bot.plot(k, r_mean_1d, lw=1.5)
    ax_bot.axhline(0, color="k", lw=0.8, ls="--")
    ax_bot.set_ylabel(r"$\Delta P \,/\, P_\mathrm{truth}$")
    ax_bot.set_xlabel(r"$k_\parallel\ [\mathrm{h\,Mpc}^{-1}]$")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Moment2 ±1σ delay-spectrum uncertainty plot."
    )
    parser.add_argument("--config", default="config_moment2_v3_v100.json")
    parser.add_argument("--output-dir", default="plots/moment2_errors")
    parser.add_argument("--baseline", type=int, default=9)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--foreground-idx", type=int, default=0)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--ra-bins", type=int, default=1024)
    parser.add_argument("--scaling", type=float, default=1e5)
    parser.add_argument("--freq-min", type=float, default=500.0)
    parser.add_argument("--freq-max", type=float, default=600.0)
    parser.add_argument("--band-center", type=float, default=550.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        configs = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    cubes, baselines, cosmo_file, gal_file = make_predictions(args, configs)

    freqs_edges = np.linspace(args.freq_min, args.freq_max, cubes["truth"].shape[1] + 1)
    freqs       = freqs_edges[:-1] + np.diff(freqs_edges[:2])[0] / 2
    band_center = args.band_center * units.MHz

    plot_moment2_1d(
        cubes,
        args.baseline,
        freqs,
        band_center,
        os.path.join(args.output_dir, "moment2_delay_spectrum_errors_1d.png"),
    )

    with open(os.path.join(args.output_dir, "inputs.txt"), "w") as f:
        f.write(f"config: {args.config}\n")
        f.write(f"cosmo_file: {cosmo_file}\n")
        f.write(f"gal_file: {gal_file}\n")
        f.write(f"baseline: {args.baseline}\n")


if __name__ == "__main__":
    main()
