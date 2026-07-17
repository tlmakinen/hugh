"""Configurable 3D U-Net (v2).

A drop-in replacement for ``nets.UNet3d`` with all of the architectural knobs
described in ``SUGGESTIONS.md`` exposed via a single ``UNet3dV2`` class.

Spatial axis convention for the input tensor:
    ``(B*split, channels=2, RA-chunk, freq, baseline)``  -- shape ``(_, 2, 256, 128, 48)``.

Per-axis padding mode therefore maps to ``(D=RA, H=freq, W=baseline)``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class Snake(nn.Module):
    """Snake activation (Liu+ 2020).

    ``snake(x) = x + (1/alpha) * sin(alpha * x)**2``.  Good for signals with a
    natural oscillatory / spectral structure (e.g. delay-domain features).
    """

    def __init__(self, alpha: float = 1.0, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add eps to avoid 0/alpha if learnable alpha drifts to 0.
        alpha = self.alpha + 1e-9
        return x + torch.sin(alpha * x).pow(2) / alpha


def _legacy_smooth_leaky() -> nn.Module:
    # Re-export the existing smooth_leaky for completeness.
    from nets import smooth_leaky

    return smooth_leaky(inplace=False)


def get_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name (case-insensitive)."""
    name = name.lower()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish(inplace=False)
    if name in ("leaky", "leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(0.1, inplace=False)
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "snake":
        return Snake(alpha=1.0, learnable=True)
    if name == "smooth_leaky":
        return _legacy_smooth_leaky()
    raise ValueError(f"Unknown activation: {name!r}")


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def get_norm(name: str, num_channels: int, *, groups: int = 8) -> nn.Module:
    name = name.lower()
    if name in ("groupnorm", "gn", "group"):
        g = min(groups, num_channels)
        # Ensure groups divide channels; fall back to LayerNorm-equivalent.
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if name in ("batchnorm", "bn", "batch"):
        return nn.BatchNorm3d(num_channels)
    if name in ("instancenorm", "in", "instance"):
        return nn.InstanceNorm3d(num_channels, affine=True)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name!r}")


# ---------------------------------------------------------------------------
# Axis-aware padding + Conv
# ---------------------------------------------------------------------------


_MODE_TRANSLATE = {
    "circular": "circular",
    "reflect": "reflect",
    "replicate": "replicate",
    "zero": "constant",
    "zeros": "constant",
    "constant": "constant",
}


def _axis_pad(
    x: torch.Tensor,
    pad: Union[int, Tuple[int, int, int]],
    modes: Sequence[str],
) -> torch.Tensor:
    """Pad each spatial axis with a possibly-different mode and amount.

    ``modes`` is a 3-tuple of strings, one per spatial axis in order
    (D=RA, H=freq, W=baseline). ``pad`` may be a single int (applied to all
    axes) or a 3-tuple ``(pd, ph, pw)`` for per-axis control. Uses ``F.pad``
    which pads from the last axis forwards.
    """
    if isinstance(pad, int):
        pad = (pad, pad, pad)
    pd, ph, pw = pad
    modes = [_MODE_TRANSLATE[m] for m in modes]
    if pw > 0:
        x = F.pad(x, (pw, pw, 0, 0, 0, 0), mode=modes[2])
    if ph > 0:
        x = F.pad(x, (0, 0, ph, ph, 0, 0), mode=modes[1])
    if pd > 0:
        x = F.pad(x, (0, 0, 0, 0, pd, pd), mode=modes[0])
    return x


class AxisAwareConv3d(nn.Module):
    """Conv3d with potentially different padding modes per spatial axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_modes: Sequence[str] = ("circular", "reflect", "replicate"),
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        assert all(k % 2 == 1 for k in kernel_size), "axis-aware conv expects odd kernels"
        self.pads = tuple(k // 2 for k in kernel_size)  # per-axis (RA, freq, baseline)
        self.pad_modes = tuple(pad_modes)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _axis_pad(x, self.pads, self.pad_modes)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Down- and upsampling
# ---------------------------------------------------------------------------


class BlurPool3d(nn.Module):
    """Anti-aliased downsampling (Zhang 2019).

    Only blurs along axes that are actually being downsampled (stride > 1).
    Axes with stride=1 pass through an identity kernel so their resolution
    is fully preserved — critical for anisotropic strides like (2, 2, 1).
    """

    def __init__(self, channels: int, stride: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.channels = channels
        self.stride = stride
        # Per-axis: binomial 1-2-1 blur where striding, identity elsewhere.
        filt_1d = torch.tensor([1.0, 2.0, 1.0])
        filt_id = torch.tensor([1.0])
        filts = [filt_1d if s > 1 else filt_id for s in stride]
        filt_3d = filts[0][:, None, None] * filts[1][None, :, None] * filts[2][None, None, :]
        filt_3d = filt_3d / filt_3d.sum()
        kernel = filt_3d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        # Pad only the axes that are blurred (stride > 1).
        self.pad_sizes = tuple(1 if s > 1 else 0 for s in stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pd, ph, pw = self.pad_sizes  # (RA, freq, baseline)
        x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode="replicate")
        return F.conv3d(x, self.kernel, stride=self.stride, groups=self.channels)


class Downsample3d(nn.Module):
    """Configurable spatial downsampling.

    ``kind`` ∈ {"stride", "avgpool", "blurpool"}.  ``stride`` is per-axis;
    set an axis to 1 to keep its resolution (e.g. ``(2, 2, 1)`` for an
    anisotropic 4× downsample that leaves baseline untouched).
    """

    def __init__(
        self,
        channels: int,
        kind: str = "blurpool",
        stride: Tuple[int, int, int] = (2, 2, 2),
        pad_modes: Sequence[str] = ("circular", "reflect", "replicate"),
    ) -> None:
        super().__init__()
        self.kind = kind.lower()
        self.stride = stride
        if self.kind == "stride":
            self.op = AxisAwareConv3d(channels, channels, 3, stride=stride, pad_modes=pad_modes)
        elif self.kind == "avgpool":
            self.op = nn.AvgPool3d(kernel_size=stride, stride=stride)
        elif self.kind == "blurpool":
            self.op = BlurPool3d(channels, stride=stride)
        else:
            raise ValueError(f"Unknown downsample kind: {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class PixelShuffle3d(nn.Module):
    """Sub-pixel upsampling for 3D tensors (no PyTorch built-in)."""

    def __init__(self, scale: Tuple[int, int, int]) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        sd, sh, sw = self.scale
        assert c % (sd * sh * sw) == 0, "channels not divisible by scale^3"
        out_c = c // (sd * sh * sw)
        x = x.view(b, out_c, sd, sh, sw, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, out_c, d * sd, h * sh, w * sw)


class Upsample3d(nn.Module):
    """Configurable spatial upsampling.

    ``kind``:
        * ``resize_conv``: ``Upsample(trilinear)`` then ``AxisAwareConv3d(k=3)`` (no checkerboard).
        * ``pixel_shuffle``: ``AxisAwareConv3d(in -> out * prod(scale))`` then ``PixelShuffle3d``.
        * ``transpose_k4``: ``ConvTranspose3d(k=4, s=stride)`` -- kernel divisible by stride, much less checkerboard than k=3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kind: str = "resize_conv",
        scale: Tuple[int, int, int] = (2, 2, 2),
        pad_modes: Sequence[str] = ("circular", "reflect", "replicate"),
    ) -> None:
        super().__init__()
        self.kind = kind.lower()
        self.scale = scale
        if self.kind == "resize_conv":
            self.upsample = nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=False)
            self.conv = AxisAwareConv3d(in_channels, out_channels, 3, pad_modes=pad_modes)
        elif self.kind == "pixel_shuffle":
            factor = scale[0] * scale[1] * scale[2]
            self.conv = AxisAwareConv3d(in_channels, out_channels * factor, 3, pad_modes=pad_modes)
            self.shuffle = PixelShuffle3d(scale)
        elif self.kind == "transpose_k4":
            # k=4 with stride=2 gives kernel divisible by stride -> minimal checkerboard.
            # For non-uniform strides we fall back to per-axis ConvTranspose by padding.
            assert all(s in (1, 2) for s in scale), "transpose_k4 supports strides 1 or 2 per axis"
            kernel = tuple(4 if s == 2 else 3 for s in scale)
            padding = tuple(1 if s == 2 else 1 for s in scale)
            self.op = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=scale,
                padding=padding,
            )
        else:
            raise ValueError(f"Unknown upsample kind: {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "resize_conv":
            return self.conv(self.upsample(x))
        if self.kind == "pixel_shuffle":
            return self.shuffle(self.conv(x))
        return self.op(x)


# ---------------------------------------------------------------------------
# Attention / SE / FiLM
# ---------------------------------------------------------------------------


class AttentionGate3d(nn.Module):
    """Additive attention gate (Oktay+ 2018) for a UNet skip connection."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: Optional[int] = None) -> None:
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 1)
        self.W_skip = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.W_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.gate = nn.Sigmoid()

    def forward(self, x_skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Resample gate to skip spatial size if needed.
        if g.shape[-3:] != x_skip.shape[-3:]:
            g = F.interpolate(g, size=x_skip.shape[-3:], mode="trilinear", align_corners=False)
        att = self.gate(self.psi(self.act(self.W_skip(x_skip) + self.W_gate(g))))
        return x_skip * att


class SEBlock3d(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.shape
        y = self.avg(x).view(b, c)
        y = self.fc2(self.act(self.fc1(y)))
        y = self.gate(y).view(b, c, 1, 1, 1)
        return x * y


class FiLM3d(nn.Module):
    """Feature-wise Linear Modulation conditioned on an external feature vector.

    ``cond`` of shape ``(B, cond_dim)`` (broadcast across all spatial axes).
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.view(*scale.shape, 1, 1, 1)
        shift = shift.view(*shift.shape, 1, 1, 1)
        return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PreActResBlock3d(nn.Module):
    """Pre-activation residual block.

    Each block performs ``Norm -> Act -> Conv -> Norm -> Act -> Conv`` plus a
    skip connection.  Optional SE attention and optional FiLM conditioning.
    Can be factored into a 2D conv on (D, H) and a 1D conv on W for cheap
    cylindrical / interferometer-symmetry-aware processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = "groupnorm",
        activation: str = "silu",
        pad_modes: Sequence[str] = ("circular", "reflect", "replicate"),
        use_se: bool = False,
        film_cond_dim: int = 0,
        factored: bool = False,
        norm_groups: int = 8,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
    ) -> None:
        super().__init__()
        self.norm1 = get_norm(norm, in_channels, groups=norm_groups)
        self.act1 = get_activation(activation)
        self.norm2 = get_norm(norm, out_channels, groups=norm_groups)
        self.act2 = get_activation(activation)
        if factored:
            # 3x3x1 then 1x1x3: separates the (RA, freq) plane from baseline.
            self.conv1 = AxisAwareConv3d(in_channels, out_channels, (3, 3, 1), pad_modes=pad_modes)
            self.conv1b = AxisAwareConv3d(out_channels, out_channels, (1, 1, 3), pad_modes=pad_modes)
            self.conv2 = AxisAwareConv3d(out_channels, out_channels, (3, 3, 1), pad_modes=pad_modes)
            self.conv2b = AxisAwareConv3d(out_channels, out_channels, (1, 1, 3), pad_modes=pad_modes)
        else:
            self.conv1 = AxisAwareConv3d(in_channels, out_channels, kernel_size, pad_modes=pad_modes)
            self.conv2 = AxisAwareConv3d(out_channels, out_channels, kernel_size, pad_modes=pad_modes)
            self.conv1b = None
            self.conv2b = None
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()
        self.se = SEBlock3d(out_channels) if use_se else nn.Identity()
        self.film = FiLM3d(out_channels, film_cond_dim) if film_cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        identity = self.skip(x)
        h = self.conv1(self.act1(self.norm1(x)))
        if self.conv1b is not None:
            h = self.conv1b(h)
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        h = self.conv2(self.act2(self.norm2(h)))
        if self.conv2b is not None:
            h = self.conv2b(h)
        h = self.se(h)
        return h + identity


class BasicBlock3d(nn.Module):
    """Light-weight ``Conv -> Norm -> Act`` block (no residual)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = "groupnorm",
        activation: str = "silu",
        pad_modes: Sequence[str] = ("circular", "reflect", "replicate"),
        norm_groups: int = 8,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        **_,
    ) -> None:
        super().__init__()
        self.conv = AxisAwareConv3d(in_channels, out_channels, kernel_size, pad_modes=pad_modes)
        self.norm = get_norm(norm, out_channels, groups=norm_groups)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


def _make_stage(
    block_cls,
    in_channels: int,
    out_channels: int,
    *,
    n_blocks: int,
    block_kwargs: dict,
) -> nn.ModuleList:
    """Stack ``n_blocks`` of ``block_cls``.  First handles channel change."""
    blocks: List[nn.Module] = []
    for i in range(n_blocks):
        blocks.append(block_cls(in_channels if i == 0 else out_channels, out_channels, **block_kwargs))
    return nn.ModuleList(blocks)


# ---------------------------------------------------------------------------
# UNet v2
# ---------------------------------------------------------------------------


class UNet3dV2(nn.Module):
    """Configurable 3D U-Net for HIRAX visibility-cube reconstruction.

    Parameters
    ----------
    in_channels : int
        Input channel count (typically 2 for Re/Im).
    out_channels : int
        Output channel count.
    filters : int
        Base channel count; doubled at each downsampling level.
    levels : int
        Number of down/up stages (2 reproduces the original; 3 is the new default).
    block : {"basic", "preact_res"}
        Building block per stage.
    n_blocks_per_stage : int
        Number of blocks per encoder/decoder stage.
    norm, activation : str
        See ``get_norm`` / ``get_activation``.
    pad_modes : tuple of str
        Padding mode per axis (D=RA, H=freq, W=baseline).
    downsample_kind : {"stride", "avgpool", "blurpool"}
    upsample_kind : {"resize_conv", "pixel_shuffle", "transpose_k4"}
    strides : list of 3-tuples
        Per-level spatial stride. Length must be ``levels``.
    attention_gates : bool
        Wrap each skip connection with an Attention U-Net gate.
    use_se : bool
        Squeeze-and-excite attention inside each residual block.
    factored_convs : bool
        Split 3x3x3 convs into 3x3x1 + 1x1x3 for cylindrical symmetry.
    film_conditioning : bool
        Inject baseline-length feature via FiLM in every residual block.
    freq_pos_encoding : bool
        Concatenate a sinusoidal positional encoding along the frequency axis
        (as 2 extra input channels).
    two_conv_head : bool
        Use a 3x3 + 1x1 output head instead of a single 3x3.
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        filters: int = 32,
        levels: int = 3,
        block: str = "preact_res",
        n_blocks_per_stage: int = 2,
        norm: str = "groupnorm",
        activation: str = "silu",
        pad_modes: Sequence[str] = ("circular", "reflect", "replicate"),
        downsample_kind: str = "blurpool",
        upsample_kind: str = "resize_conv",
        strides: Optional[Iterable[Tuple[int, int, int]]] = None,
        attention_gates: bool = False,
        use_se: bool = False,
        factored_convs: bool = False,
        film_conditioning: bool = False,
        film_cond_dim: int = 8,
        freq_pos_encoding: bool = False,
        freq_pos_dim: int = 2,
        two_conv_head: bool = True,
        norm_groups: int = 8,
        use_checkpoint: bool = False,
        scaling: float = 1e5,
        conv_kernel_size: Union[int, Tuple[int, int, int]] = 3,
    ) -> None:
        super().__init__()
        self.scaling = scaling  # kept for backward compatibility with code that reads it
        self.levels = levels
        self.attention_gates = attention_gates
        self.factored_convs = factored_convs
        self.film_conditioning = film_conditioning
        self.freq_pos_encoding = freq_pos_encoding
        self.freq_pos_dim = freq_pos_dim if freq_pos_encoding else 0
        self.use_checkpoint = use_checkpoint

        if strides is None:
            strides = [(2, 2, 2)] * levels
        strides = [tuple(s) for s in strides]
        assert len(strides) == levels, f"strides must have {levels} entries"
        self.strides = strides
        self.pad_modes = tuple(pad_modes)

        block_cls = {"basic": BasicBlock3d, "preact_res": PreActResBlock3d}[block]
        block_kwargs = dict(
            norm=norm,
            activation=activation,
            pad_modes=self.pad_modes,
            use_se=use_se,
            film_cond_dim=film_cond_dim if film_conditioning else 0,
            factored=factored_convs,
            norm_groups=norm_groups,
            kernel_size=conv_kernel_size,
        )

        # Stem: project input + optional freq pos encoding to base filters.
        stem_in = in_channels + self.freq_pos_dim
        self.stem = AxisAwareConv3d(stem_in, filters, conv_kernel_size, pad_modes=self.pad_modes)

        # Encoder
        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = [filters]
        ch = filters
        for level in range(levels):
            stage = _make_stage(block_cls, ch, ch * 2, n_blocks=n_blocks_per_stage, block_kwargs=block_kwargs)
            self.enc_stages.append(stage)
            ch = ch * 2
            channels.append(ch)
            self.downs.append(Downsample3d(ch, kind=downsample_kind, stride=strides[level], pad_modes=self.pad_modes))

        # Bottleneck
        self.bottleneck = _make_stage(block_cls, ch, ch, n_blocks=n_blocks_per_stage, block_kwargs=block_kwargs)

        # Decoder
        self.ups = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if attention_gates else None
        self.dec_stages = nn.ModuleList()
        for level in reversed(range(levels)):
            skip_ch = channels[level + 1]  # channels after that level's encoder stage
            up = Upsample3d(ch, skip_ch, kind=upsample_kind, scale=strides[level], pad_modes=self.pad_modes)
            self.ups.append(up)
            if attention_gates:
                self.attn_gates.append(AttentionGate3d(skip_ch, skip_ch))
            stage = _make_stage(block_cls, skip_ch * 2, channels[level], n_blocks=n_blocks_per_stage, block_kwargs=block_kwargs)
            self.dec_stages.append(stage)
            ch = channels[level]

        # Output head
        if two_conv_head:
            self.head = nn.Sequential(
                AxisAwareConv3d(ch, ch, conv_kernel_size, pad_modes=self.pad_modes),
                get_activation(activation),
                nn.Conv3d(ch, out_channels, kernel_size=1, bias=True),
            )
        else:
            self.head = AxisAwareConv3d(ch, out_channels, conv_kernel_size, pad_modes=self.pad_modes)

    # ----- helpers ----------------------------------------------------------

    def _maybe_freq_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.freq_pos_encoding:
            return x
        # x: (B, C, RA, freq, baseline)
        b, _, d, h, w = x.shape
        # Sinusoidal encoding along the freq axis with self.freq_pos_dim bands.
        positions = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype)
        bands = torch.arange(1, self.freq_pos_dim // 2 + 1, device=x.device, dtype=x.dtype)
        # Pairs of (sin, cos) per band -> freq_pos_dim total channels.
        feats = []
        for band in bands:
            feats.append(torch.sin(math.pi * band * positions))
            feats.append(torch.cos(math.pi * band * positions))
        if self.freq_pos_dim % 2 == 1:
            # If odd, add one extra sin band
            feats.append(torch.sin(math.pi * (self.freq_pos_dim // 2 + 1) * positions))
        enc = torch.stack(feats, dim=0)  # (freq_pos_dim, freq)
        enc = enc.view(1, self.freq_pos_dim, 1, h, 1).expand(b, -1, d, -1, w)
        return torch.cat([x, enc], dim=1)

    def _stage_forward(self, stage: nn.ModuleList, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        for blk in stage:
            takes_cond = isinstance(blk, PreActResBlock3d)
            if self.use_checkpoint and self.training and x.requires_grad:
                # use_reentrant=False is preferred (works with torch.compile, no
                # warnings) but only available in torch >= 1.11.
                if takes_cond:
                    x = ckpt_fn(blk, x, cond, use_reentrant=False)
                else:
                    x = ckpt_fn(blk, x, use_reentrant=False)
            else:
                x = blk(x, cond) if takes_cond else blk(x)
        return x

    # ----- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._maybe_freq_pos_encoding(x)
        x = self.stem(x)

        skips: List[torch.Tensor] = []
        for stage, down in zip(self.enc_stages, self.downs):
            x = self._stage_forward(stage, x, cond)
            skips.append(x)
            x = down(x)

        x = self._stage_forward(self.bottleneck, x, cond)

        for level, up_op, stage in zip(reversed(range(self.levels)), self.ups, self.dec_stages):
            skip = skips[level]
            x = up_op(x)
            # Make sure spatial size matches the skip exactly (handles odd shapes).
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            if self.attn_gates is not None:
                # ups/dec_stages/attn_gates are indexed in decoder order
                attn_idx = self.levels - 1 - level
                skip = self.attn_gates[attn_idx](skip, x)
            x = torch.cat([x, skip], dim=1)
            x = self._stage_forward(stage, x, cond)

        return self.head(x)


__all__ = [
    "UNet3dV2",
    "PreActResBlock3d",
    "BasicBlock3d",
    "AttentionGate3d",
    "SEBlock3d",
    "FiLM3d",
    "AxisAwareConv3d",
    "Downsample3d",
    "Upsample3d",
    "BlurPool3d",
    "PixelShuffle3d",
    "Snake",
    "get_activation",
    "get_norm",
]
