# UNet Architecture Improvement Suggestions

Generated 2026-06-15. Companion to `nets_v2.py` and `config_train2_v3.json`.

## Current default model (`UNet3d` in `nets.py`, `filters=16`)

Encoder (~150 K parameters total):

| layer    | op                                  | channels | stride | spatial | role        |
|----------|-------------------------------------|----------|--------|---------|-------------|
| `layer1` | 2× (Conv3d3³ + BN + act)            | 2 → 16   | 1      | keep    | stem        |
| `layer2` | 1× block                            | 16 → 32  | **2**  | ÷2      | downsample  |
| `layer3` | 2× block                            | 32 → 32  | 1      | keep    | skip `x2`   |
| `layer4` | 1× block                            | 32 → 64  | **2**  | ÷2      | downsample  |
| `layer5` | 2× block                            | 64 → 64  | 1      | keep    | bottleneck  |

Decoder:

| layer     | op                                                   | channels | role    |
|-----------|------------------------------------------------------|----------|---------|
| `deconv1` | `ConvTranspose3d(k=3, s=2, op=1)` + BN + act         | 64 → 32  | ×2      |
| `layer6`  | 2× block (after concat with `x2`)                    | 64 → 32  | refine  |
| `deconv2` | `ConvTranspose3d(k=3, s=2, op=1)` + BN + act         | 32 → 16  | ×2      |
| `layer7`  | 2× block (after concat with `x1`)                    | 32 → 16  | refine  |
| `deconv4` | `Conv3d(k=3, s=1, circular)`                         | 16 → 2   | head    |

`BasicBlock`: `Conv3d(k=3, padding_mode="circular", bias=True)` → `BatchNorm3d` → `act(inplace=True)`. No residual.

Activations / norms / loss / optim: `smooth_leaky` (C¹, scaled cubic in [-1,1]), `BatchNorm3d`, `log(MSE)`, Adam, gradient clip 1.0, `lr = 1e-4`, no LR scheduler, fixed Gaussian noise (`1.5e-7`).

Spatial layout fed to the network: `(B*split=16, 2, 256 RA-chunk, 128 freq, 48 baseline)`. Total downsampling = 4× → bottleneck spatial `(64, 32, 12)`. Max receptive field ≈ 16 freq bins out of 128 — small for delay-space structure.

---

## Improvement suggestions

### A. Activations (smoothness)

1. Swap `smooth_leaky` → **`SiLU`** (Swish) or **`GELU`**. Both C^∞, monotone, standard in modern UNets / diffusion. `SiLU` is already half-wired in `visualize_reconstructions.py`.
2. **Mish** — smooth with a small positive bump near 0, often slightly better than SiLU for regression.
3. Drop `inplace=True` on post-deconv activations — blocks residual addition and confuses `torch.compile`.
4. **Snake / sin activations** (Liu+ 2020) for periodic/spectral data, since the underlying signal has oscillatory delay-domain structure.

### B. Downsampling / upsampling (avoiding gridding & checkerboard)

5. `ConvTranspose3d(k=3, s=2, op=1)` is the textbook source of **checkerboard artifacts** (Odena et al. 2016). Replace with **resize-conv**: `Upsample(scale=2, mode='trilinear', align_corners=False)` → `Conv3d(k=3, padding=1)`.
6. Alternative: **PixelShuffle3d** (sub-pixel convolution). Same speed as deconv, no aliasing.
7. Alternative: **kernel size divisible by stride** (k=4, s=2). Reduces checkerboard without restructuring.
8. **Strided conv for downsampling also aliases.** Replace with **BlurPool** (Zhang 2019): 3³ anti-aliased blur, then stride-2. Or `AvgPool3d(2)` after a stride-1 conv.
9. **Axis-anisotropic strides.** The three spatial axes are physically very different (RA / freq / baseline). Striding 2× on baseline (48 → 24 → 12) discards information not redundant the way RA bins are. Consider `(2,2,1)` strides or keep baseline at full resolution.

### C. Padding modes (currently all-circular: wrong on freq & baseline)

10. Circular padding is correct on RA (true periodicity) but **wrong on frequency** (FFTs assume non-periodic continuation; wrap-around contaminates the wedge) and **wrong on baseline** (UV-plane is not periodic). Use **axis-dependent padding**: `circular` on RA, `reflect` on freq, `replicate` or `zero` on baseline.
11. Same issue affects `deconv4` (final head).

### D. Normalization

12. Replace `BatchNorm3d` with **`GroupNorm(8, C)`** or **`InstanceNorm3d`**. BatchNorm is fragile under mixed precision, mixes statistics across the 16-sample super-batch, and breaks at val/inference time on a single sample. GroupNorm is batch-invariant and the de-facto standard in diffusion / cosmology nets.
13. **Pre-activation residual block** (`norm → act → conv → norm → act → conv + skip`) replaces the bare `Conv→BN→act` block. Better gradient flow and lets you go deeper.

### E. Receptive field & capacity

14. **Add one more down/up level.** Current model has 2 downsamplings (receptive field ~16 freq bins). A third stage gives 32-bin receptive field — a quarter of the band, much better suited to large-scale spectral structure where the wedge lives.
15. **Bump `filters` from 16 → 32 or 64.** The current model has ~150 K params; H100 memory has plenty of room.
16. **Dilated convolutions in the bottleneck** (ASPP) for cheap receptive-field expansion without more downsampling.

### F. Skip connections

17. **Attention gates** on skips (Attention U-Net, Oktay+ 2018). The code already has `nets2_attn.py`.
18. **Channel attention (SE block)** inside each residual block. Cheap; often +0.5 dB.

### G. Output head & target parameterization

19. **Predict the residual `(truth − PCA_cleaned)`** instead of the full truth. Then `output = PCA_cleaned + residual_pred`. Smaller dynamic range, faster convergence, clearer inductive bias as a "learned PCA refinement".
20. **arcsinh dynamic-range compression.** `nets.py` already defines `transform_inputs/inv_transform_inputs` (arcsinh + scaling) but they are unused. Using arcsinh on inputs/targets makes the loss see a near-Gaussian, well-conditioned distribution instead of values spanning ~7 orders of magnitude across foregrounds → cosmology.
21. **Two-conv head** (3×3 + 1×1) instead of single 3×3 — better blending.

### H. Loss

22. **Hybrid loss.** Add a term in the **delay-spectrum domain** — compute `|FFT_freq(pred)|²` and `|FFT_freq(truth)|²` and penalize their difference. Directly disciplines the cosmological observable.
23. **LogCosh** (already coded in `train2.py`, just commented out) is more robust to outliers than logMSE.
24. **Charbonnier / smooth-L1** for the pixel term.

### I. Optimizer / schedule

25. **AdamW** with small weight decay (1e-4 – 1e-2) instead of vanilla Adam.
26. **Cosine annealing or OneCycleLR** with linear warmup (~5% of steps). The current flat LR slows visibly mid-training.
27. **EMA of weights** (decay 0.999) for evaluation — usually 0.1–0.3 dB free improvement.

### J. Data / augmentation

28. **Cyclic RA shifts** as augmentation — free 1024× data multiplication thanks to RA periodicity.
29. **Random subset of baselines** per batch — baseline-equivariant features.
30. **Random PCA-component count** (e.g. `n_fg` ∈ {8, 11, 14}) — robust to PCA hyperparameter.
31. **Per-batch noise amplitude** drawn from a range, not fixed.

### K. Physics-inspired architecture tweaks

32. **Cylindrical / factored convs.** Convolve `(RA, freq)` together (2D) and combine across baselines with a 1D conv. Faster than 3D and matches the natural symmetry of an interferometer.
33. **Baseline-conditioning via FiLM.** Inject baseline length (and frequency) as per-channel scale/shift; wedge structure is determined by `|b|·f/c`.
34. **Sinusoidal positional encoding on freq channel** — helps the net know which frequency a given voxel is.

### L. Numerical / engineering

35. **GroupNorm or LayerNorm** (point 12) — also makes the bf16 path more stable.
36. **Disable `padding_mode="circular"`** while testing reflect/replicate — circular conv kernels go through slower `im2col` paths on H100.

---

## Recommended order of attack

Highest expected ROI per amount of work, given the current state:

1. **Resize-conv upsampling + axis-aware padding** (B5, C10)
2. **SiLU + GroupNorm** (A1, D12)
3. **Predict residual over PCA + arcsinh target compression** (G19, G20)
4. **Add a third down/up level + bump `filters` to 32** (E14, E15)
5. **OneCycleLR + AdamW + EMA** (I25, I26, I27)
6. **Delay-space loss term** (H22) — once basics are stable
7. **Attention gates + physics conditioning** (F17, K33) — once 1–5 are dialled in

Items 1–3 alone would probably take `corr(UNet, truth) = 0.77` and `residual power = 0.41` into much better territory.

---

## What is implemented in `nets_v2.py` + this branch of `train2.py`

| suggestion | flag in `config_train2_v3.json` | default in v3 |
|---|---|---|
| SiLU / GELU / Mish / Snake / smooth_leaky (A1-A4) | `model_params.activation` | `silu` |
| Drop inplace activations (A3) | implicit (always off in v2 blocks) | always |
| Resize-conv upsampling (B5) | `model_params.upsample_kind` | `resize_conv` |
| PixelShuffle3d upsampling (B6) | `model_params.upsample_kind = "pixel_shuffle"` | option |
| ConvTranspose3d (B legacy) | `model_params.upsample_kind = "transpose_k4"` (k=4, s=2) | option |
| BlurPool downsampling (B8) | `model_params.downsample_kind = "blurpool"` | `blurpool` |
| Strided-conv / avgpool downsampling | `model_params.downsample_kind` | option |
| Anisotropic strides (B9) | `model_params.strides` (list of 3-tuples per level) | `[2,2,2]` levels |
| Axis-aware padding (C10, C11) | `model_params.pad_modes` (RA/freq/baseline) | `["circular","reflect","replicate"]` |
| GroupNorm / InstanceNorm / BatchNorm (D12) | `model_params.norm` | `groupnorm` |
| Pre-activation residual block (D13) | `model_params.block` | `preact_res` |
| 3-level encoder/decoder (E14) | `model_params.levels` | `3` |
| Bigger `filters` (E15) | `model_params.filters` | `32` |
| Attention gates on skips (F17) | `model_params.attention_gates` | `false` |
| SE block in each res block (F18) | `model_params.use_se` | `false` |
| FiLM baseline conditioning (K33) | `model_params.film_conditioning` | `false` |
| Frequency positional encoding (K34) | `model_params.freq_pos_encoding` | `false` |
| Factored 2D+1D convs (K32) | `model_params.factored_convs` | `false` |
| Two-conv output head (G21) | `model_params.two_conv_head` | `true` |
| arcsinh target compression (G20) | `training_params.target_transform = "arcsinh"` | `arcsinh` |
| Residual prediction (G19) | `training_params.predict_residual` | `true` |
| RA cyclic-shift augmentation (J28) | `training_params.ra_shift_aug` | `true` |
| AdamW + weight decay (I25) | `training_params.optimizer = "adamw"`, `weight_decay` | `adamw`, 1e-4 |
| OneCycleLR + warmup (I26) | `training_params.lr_schedule = "onecycle"`, `warmup_frac` | `onecycle`, 0.05 |
| EMA of weights (I27) | `training_params.ema_decay` | `0.999` |
