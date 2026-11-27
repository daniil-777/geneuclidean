"""Composable Google-SWE Neural Network, Geometric Attention, and SE(3) Transformer Layers.

All modules in this library adhere strictly to GenEuclidean's cross-accelerator numerical
contract:
  1. **Zero ``torch.float64`` state:** Every parameter, persistent/non-persistent buffer, and
     analytical embedding table is allocated explicitly in ``torch.float32`` so models run
     natively on Apple Silicon Metal (MPS), NVIDIA CUDA, and CPU SIMD without silent precision
     crashes.
  2. **Exact padding-mask isolation:** Masked dummy tokens/atoms are zeroed prior to linear
     projections and normalized via
     :func:`~geneuclidean.models.encoders.dense_knn.safe_masked_softmax` so padded slots
     contribute exact ``0.0`` weight and cannot produce ``NaN`` gradients on empty rows.
  3. **Exact ``SE(3)`` / ``SO(3)`` symmetry:** Geometric attention, scalar-vector gates, pair
     triangle updates, and :class:`SE3GeometricTransformerBlock` operate in valid-atom centroid
     frames with optional signed Levi-Civita cross/triple-product chiral coupling
     (``use_se3_chirality=True`` for :math:`(R)/(S)` enantiomer discrimination vs. ``False``
     for full :math:`E(3)` mirror invariance).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.models.encoders.dense_knn import (
    build_dense_knn_graph,
    center_masked_coords_with_offset,
    gather_neighbors,
    safe_masked_softmax,
)
from geneuclidean.models.encoders.electrostatics import ScreenedCoulombBias
from geneuclidean.models.encoders.rbf import BesselRBF, CosineCutoff, EquivariantRMSNorm
from geneuclidean.models.encoders.tensor_ops import VectorNeuronLinear, chiral_cross_product


def _default_ffn_dim(d_model: int, multiple_of: int = 32) -> int:
    """Compute SwiGLU/GEGLU hidden width ``~8/3 * d_model`` rounded up to ``multiple_of``."""
    raw = max(multiple_of, int(round(8.0 * d_model / 3.0)))
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# 1. Normalization, Conditioning & Gated Feed-Forward Layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) in pure ``float32``.

    Normalizes trailing feature vectors by their root-mean-square magnitude without mean
    subtraction, preserving exact ``0.0`` on zeroed dummy tokens.

    Args:
        dim: Trailing feature dimension to normalize.
        eps: Positive variance floor for numerical stability.
        elementwise_affine: Whether to apply a learnable per-channel ``float32`` scale vector.
        init_scale: Initial value for the learnable scale parameter (default ``1.0``).
    """

    dim: int
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if init_scale <= 0.0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")

        self.dim = int(dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight: nn.Parameter | None = nn.Parameter(
                torch.full((self.dim,), float(init_scale), dtype=torch.float32)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize trailing axis of ``x`` (`[..., dim] -> [..., dim]`)."""
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected trailing dim {self.dim}, got shape {tuple(x.shape)}")
        x_f = x.to(dtype=torch.float32)
        normed = x_f * torch.rsqrt((x_f * x_f).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normed = normed * self.weight
        return normed.to(dtype=x.dtype)


class QKNorm(nn.Module):
    """Per-head RMS Query/Key Normalization for deep Transformer logit stability.

    Bounds per-head dot-product logits to :math:`[-D_h, D_h]` (Dehghani et al., ViT-22B 2023;
    Abramson et al., AlphaFold3 2024), preventing attention entropy collapse in deep sequence
    and 3D geometric transformers. When applied prior to Rotary Position Embeddings (RoPE),
    strictly preserves exact relative-position inner-product shift invariance.

    Args:
        head_dim: Per-head channel dimension ``D_h``.
        num_q_heads: Optional number of query heads ``H_q`` for distinct per-head scales.
            When ``None``, shares a ``[head_dim]`` scale vector across query heads.
        num_kv_heads: Optional number of key heads ``H_kv`` for distinct per-head scales.
            When ``None``, shares a ``[head_dim]`` scale vector across key heads.
        eps: Positive RMS denominator floor.
    """

    head_dim: int
    num_q_heads: int | None
    num_kv_heads: int | None
    eps: float

    def __init__(
        self,
        head_dim: int,
        num_q_heads: int | None = None,
        num_kv_heads: int | None = None,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if num_q_heads is not None and num_q_heads < 1:
            raise ValueError(f"num_q_heads must be >= 1 when provided, got {num_q_heads}")
        if num_kv_heads is not None and num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be >= 1 when provided, got {num_kv_heads}")

        self.head_dim = int(head_dim)
        self.num_q_heads = int(num_q_heads) if num_q_heads is not None else None
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else None
        self.eps = float(eps)

        q_shape = (
            (1, self.num_q_heads, 1, self.head_dim)
            if self.num_q_heads is not None
            else (self.head_dim,)
        )
        k_shape = (
            (1, self.num_kv_heads, 1, self.head_dim)
            if self.num_kv_heads is not None
            else (self.head_dim,)
        )
        self.q_weight = nn.Parameter(torch.ones(q_shape, dtype=torch.float32))
        self.k_weight = nn.Parameter(torch.ones(k_shape, dtype=torch.float32))

    def norm_q(self, q: Tensor) -> Tensor:
        """Apply per-head RMS normalization to queries ``[..., H_q, L, head_dim]``."""
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"Expected q trailing dim {self.head_dim}, got {tuple(q.shape)}")
        q_f = q.to(dtype=torch.float32)
        inv_rms = torch.rsqrt((q_f * q_f).mean(dim=-1, keepdim=True) + self.eps)
        return ((q_f * inv_rms) * self.q_weight).to(dtype=q.dtype)

    def norm_k(self, k: Tensor) -> Tensor:
        """Apply per-head RMS normalization to keys ``[..., H_kv, L, head_dim]``."""
        if k.shape[-1] != self.head_dim:
            raise ValueError(f"Expected k trailing dim {self.head_dim}, got {tuple(k.shape)}")
        k_f = k.to(dtype=torch.float32)
        inv_rms = torch.rsqrt((k_f * k_f).mean(dim=-1, keepdim=True) + self.eps)
        return ((k_f * inv_rms) * self.k_weight).to(dtype=k.dtype)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Return RMS-normalized ``(q_norm, k_norm)`` matching input shapes and dtypes."""
        return self.norm_q(q), self.norm_k(k)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero conditioning block (Peebles & Xie, DiT ICCV 2023).

    Projects a continuous conditioning embedding ``cond`` (flow/diffusion timestep, 3D pocket
    context, or target property vector) into affine modulation parameters ``(shift, scale)``
    and residual gate ``gate`` via a zero-initialized output linear layer:

    .. math::
        \\tilde{\\mathbf{x}} = (1 + \\boldsymbol{\\gamma}(\\mathbf{c})) \\odot
        \\mathrm{LayerNorm}(\\mathbf{x}) + \\boldsymbol{\\beta}(\\mathbf{c}), \\qquad
        \\mathbf{y} = \\mathbf{x} + \\boldsymbol{\\alpha}(\\mathbf{c}) \\odot
        f(\\tilde{\\mathbf{x}}).

    When ``zero_init=True`` (default), :math:`\\boldsymbol{\\gamma} = \\boldsymbol{\\beta} =
    \\boldsymbol{\\alpha} = \\mathbf{0}` at initialization, guaranteeing that
    :meth:`forward_residual` acts as an exact identity map (:math:`\\mathbf{y} == \\mathbf{x}`).

    Args:
        d_model: Feature width of ``x``.
        cond_dim: Feature width of ``cond`` (defaults to ``d_model``).
        eps: Base LayerNorm epsilon.
        zero_init: Zero-initialize the modulation projection so ``gate == 0`` at init.
        gate_bias_init: Initial bias applied to the ``gate`` chunk when ``zero_init=False``
            (default ``1.0``). Ignored when ``zero_init=True``.
    """

    d_model: int
    cond_dim: int
    eps: float
    zero_init: bool

    def __init__(
        self,
        d_model: int,
        cond_dim: int | None = None,
        *,
        eps: float = 1e-6,
        zero_init: bool = True,
        gate_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        resolved_cond = int(cond_dim) if cond_dim is not None else int(d_model)
        if resolved_cond < 1:
            raise ValueError(f"cond_dim must be >= 1, got {resolved_cond}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.d_model = int(d_model)
        self.cond_dim = resolved_cond
        self.eps = float(eps)
        self.zero_init = bool(zero_init)

        self.norm = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=self.eps)
        self.act = nn.SiLU()
        self.modulation = nn.Linear(self.cond_dim, 3 * self.d_model, bias=True)

        if self.zero_init:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            nn.init.normal_(self.modulation.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.modulation.bias)
            with torch.no_grad():
                self.modulation.bias[2 * self.d_model :].fill_(float(gate_bias_init))

    def modulation_params(
        self,
        cond: Tensor,
        *,
        target_ndim: int = 3,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute broadcast-ready ``(shift, scale, gate)`` tensors from ``cond``."""
        if cond.ndim < 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"Expected cond trailing dim {self.cond_dim}, got shape {tuple(cond.shape)}"
            )
        raw: Tensor = self.modulation(self.act(cond.to(dtype=torch.float32)))
        while raw.ndim < target_ndim:
            raw = raw.unsqueeze(-2)
        shift, scale, gate = raw.chunk(3, dim=-1)
        return shift, scale, gate

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(modulated_x, gate)`` with ``modulated_x = LN(x) * (1 + scale) + shift``."""
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x trailing dim {self.d_model}, got shape {tuple(x.shape)}")
        shift, scale, gate = self.modulation_params(cond, target_ndim=x.ndim)
        x_norm: Tensor = self.norm(x.to(dtype=torch.float32))
        modulated = x_norm * (1.0 + scale) + shift
        return modulated.to(dtype=x.dtype), gate.to(dtype=x.dtype)

    def apply_gate(self, x: Tensor, sublayer_out: Tensor, gate: Tensor) -> Tensor:
        """Apply residual gating ``x + gate * sublayer_out``."""
        return x + gate * sublayer_out

    def forward_residual(
        self,
        x: Tensor,
        cond: Tensor,
        sublayer_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Execute ``x + gate * sublayer_fn(modulated_x)`` (exact identity at zero-init)."""
        modulated, gate = self.forward(x, cond)
        return self.apply_gate(x, sublayer_fn(modulated), gate)


class SwiGLU(nn.Module):
    """SwiGLU gated SiLU feed-forward network (Shazeer, 2020) in ``float32``."""

    d_model: int
    d_ff: int

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        resolved_ff = int(d_ff) if d_ff is not None else _default_ffn_dim(d_model)
        if resolved_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {resolved_ff}")

        self.d_model = int(d_model)
        self.d_ff = resolved_ff
        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.w3 = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate ``W_2(Dropout(SiLU(W_1 x) * (W_3 x)))``."""
        hidden = F.silu(self.w1(x)) * self.w3(x)
        out: Tensor = self.w2(self.dropout(hidden))
        return out


class GEGLU(nn.Module):
    """GEGLU gated GELU feed-forward network (Shazeer, 2020) in ``float32``."""

    d_model: int
    d_ff: int
    approximate: Literal["none", "tanh"]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        approximate: Literal["none", "tanh"] = "tanh",
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        resolved_ff = int(d_ff) if d_ff is not None else _default_ffn_dim(d_model)
        if resolved_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {resolved_ff}")
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")

        self.d_model = int(d_model)
        self.d_ff = resolved_ff
        self.approximate = approximate
        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.w3 = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate ``W_2(Dropout(GELU(W_1 x) * (W_3 x)))``."""
        hidden = F.gelu(self.w1(x), approximate=self.approximate) * self.w3(x)
        out: Tensor = self.w2(self.dropout(hidden))
        return out


class DropPath(nn.Module):
    """Stochastic Depth per-sample residual drop (Huang et al., ECCV 2016).

    Broadcasts a per-batch-item scalar gate ``[B, 1, ..., 1]`` across all token, channel, and
    3D spatial axes, strictly preserving ``SO(3)``/``O(3)`` equivariance when applied to
    3D vector or Cartesian tensor residual branches.
    """

    drop_prob: float
    scale_by_keep: bool

    def __init__(self, drop_prob: float = 0.0, *, scale_by_keep: bool = True) -> None:
        super().__init__()
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError(f"drop_prob must be in [0.0, 1.0), got {drop_prob}")
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-sample stochastic depth during training; exact identity during eval."""
        if self.drop_prob == 0.0 or not self.training or x.ndim == 0 or x.shape[0] == 0:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_mask = torch.empty(mask_shape, dtype=torch.float32, device=x.device).bernoulli_(
            keep_prob
        )
        if self.scale_by_keep:
            keep_mask = keep_mask / keep_prob
        return x * keep_mask.to(dtype=x.dtype)


class SinusoidalFourierTimeEmbedding(nn.Module):
    """Continuous :math:`t \\in [0, 1]` sinusoidal Fourier timestep embedding in ``float32``.

    Maps continuous flow-matching or diffusion times :math:`t \\in [0, 1]` into smooth
    log-spaced sinusoidal features with an optional 2-layer ``SiLU`` projection MLP.

    Args:
        dim: Base sinusoidal embedding dimension (``>= 2``).
        out_dim: Output dimension of the optional MLP (defaults to ``dim``).
        max_period: Maximum sinusoidal wavelength governing minimum frequency.
        time_scale: Multiplier applied to continuous ``t in [0, 1]`` (default ``1000.0``).
        use_mlp: If ``True`` (default), refine sinusoidal features through a 2-layer SiLU MLP.
    """

    dim: int
    out_dim: int
    max_period: float
    time_scale: float
    use_mlp: bool
    freqs: Tensor

    def __init__(
        self,
        dim: int,
        *,
        out_dim: int | None = None,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
        use_mlp: bool = True,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if max_period <= 1.0:
            raise ValueError(f"max_period must be > 1.0, got {max_period}")
        resolved_out = int(out_dim) if out_dim is not None else int(dim)
        if resolved_out < 1:
            raise ValueError(f"out_dim must be >= 1, got {resolved_out}")

        self.dim = int(dim)
        self.out_dim = resolved_out
        self.max_period = float(max_period)
        self.time_scale = float(time_scale)
        self.use_mlp = bool(use_mlp)

        half = self.dim // 2
        exponent = -math.log(self.max_period) * (
            torch.arange(0, half, dtype=torch.float32) / float(max(half, 1))
        )
        self.register_buffer("freqs", torch.exp(exponent), persistent=False)

        self.mlp: nn.Sequential | None
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.dim, self.out_dim),
                nn.SiLU(),
                nn.Linear(self.out_dim, self.out_dim),
            )
        elif self.dim != self.out_dim:
            self.mlp = nn.Sequential(nn.Linear(self.dim, self.out_dim))
        else:
            self.mlp = None

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed continuous timesteps (``[]``, ``[B]``, ``[B, 1]``, ``[B, N]``) to ``out_dim``."""
        t_f = timesteps.to(dtype=torch.float32)
        if t_f.ndim == 0:
            t_f = t_f.unsqueeze(0)
        elif t_f.ndim >= 2 and t_f.shape[-1] == 1:
            t_f = t_f.squeeze(-1)

        angles = (t_f.unsqueeze(-1) * self.time_scale) * self.freqs
        emb = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        if self.mlp is not None:
            out: Tensor = self.mlp(emb)
            return out
        return emb


# ---------------------------------------------------------------------------
# 2. Modern Sequence & Multi-Modal 3D Attention Layers
# ---------------------------------------------------------------------------


def _rotate_half_1d(x: Tensor) -> Tensor:
    """Rotate paired feature halves ``(x1, x2) -> (-x2, x1)`` for 1D RoPE."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_1d(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply 1D Rotary Position Embeddings to query and key tensors ``[B, H, L, Dh]``."""
    q_f = q.to(dtype=torch.float32)
    k_f = k.to(dtype=torch.float32)
    q_rot = (q_f * cos) + (_rotate_half_1d(q_f) * sin)
    k_rot = (k_f * cos) + (_rotate_half_1d(k_f) * sin)
    return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)


class RotaryPositionEmbedding(nn.Module):
    """1D Rotary Position Embedding (Su et al., 2021) supporting explicit position offsets."""

    head_dim: int
    base: float
    inv_freq: Tensor
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        *,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim >= 2, got {head_dim}")
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")

        self.head_dim = int(head_dim)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._build_cache(max_seq_len, inv_freq)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def _build_cache(seq_len: int, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]

    def from_positions(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Compute ``(cos, sin)`` tables for explicit 1D ``[L]`` or 2D ``[B, L]`` positions."""
        pos_f = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        if pos_f.ndim == 1:
            freqs = torch.outer(pos_f, self.inv_freq)
            angles = torch.cat((freqs, freqs), dim=-1)
            return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
        if pos_f.ndim == 2:
            freqs = pos_f.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
            angles = torch.cat((freqs, freqs), dim=-1)
            return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        raise ValueError(f"Expected 1D [L] or 2D [B, L] position_ids, got {tuple(pos_f.shape)}")

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        *,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables of shape ``[1, 1, seq_len, head_dim]``."""
        end = offset + seq_len
        if offset >= 0 and end <= self.cos_cached.shape[2]:
            return (
                self.cos_cached[:, :, offset:end, :].to(device=device),
                self.sin_cached[:, :, offset:end, :].to(device=device),
            )
        pos = torch.arange(offset, end, device=device, dtype=torch.float32)
        cos, sin = self.from_positions(pos)
        return cos.to(device=device), sin.to(device=device)


@dataclass(frozen=True)
class AttentionKVCache:
    """Immutable per-layer Key/Value cache ``[B, H_kv, L_cache, D_h]`` for GQA decoding."""

    k: Tensor
    v: Tensor

    @property
    def seq_len(self) -> int:
        """Number of cached key/value sequence steps."""
        return int(self.k.shape[2])


class GroupedQueryAttention(nn.Module):
    """Grouped-Query / Multi-Head Self-Attention with 1D RoPE, QKNorm, KV cache, and SDPA.

    Applies :class:`QKNorm` *prior* to :class:`RotaryPositionEmbedding` so query-key dot
    products retain exact relative-position shift invariance
    :math:`\\langle \\mathbf{q}_{m+\\Delta}, \\mathbf{k}_{n+\\Delta}\\rangle =
    \\langle \\mathbf{q}_m, \\mathbf{k}_n\\rangle` even with anisotropic learned per-head
    normalization weights. Dispatches to PyTorch's fused
    :func:`torch.nn.functional.scaled_dot_product_attention` fast path when unmasked or causal.
    """

    d_model: int
    num_heads: int
    num_kv_heads: int
    num_kv_groups: int
    head_dim: int
    use_qk_norm: bool
    use_rope: bool
    is_causal: bool
    dropout_p: float

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        *,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
        is_causal: bool = True,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1:
            raise ValueError(f"d_model ({d_model}) and num_heads ({num_heads}) must be >= 1")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        resolved_kv = int(num_kv_heads) if num_kv_heads is not None else int(num_heads)
        if resolved_kv < 1 or num_heads % resolved_kv != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({resolved_kv})"
            )

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = resolved_kv
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.d_model // self.num_heads
        self.use_qk_norm = bool(use_qk_norm)
        self.use_rope = bool(use_rope)
        self.is_causal = bool(is_causal)
        self.dropout_p = float(dropout)

        if self.use_rope and self.head_dim % 2 != 0:
            raise ValueError(f"use_rope=True requires even head_dim, got {self.head_dim}")

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.resid_dropout = nn.Dropout(self.dropout_p)

        self.qk_norm: QKNorm | None = (
            QKNorm(self.head_dim, num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads)
            if self.use_qk_norm
            else None
        )
        self.rope: RotaryPositionEmbedding | None = (
            RotaryPositionEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
            if self.use_rope
            else None
        )

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
        position_ids: Tensor | None = None,
        kv_cache: AttentionKVCache | None = None,
        use_cache: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
        position_ids: Tensor | None = None,
        kv_cache: AttentionKVCache | None = None,
        use_cache: Literal[True],
    ) -> tuple[Tensor, AttentionKVCache]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
        position_ids: Tensor | None = None,
        kv_cache: AttentionKVCache | None = None,
        use_cache: bool,
    ) -> Tensor | tuple[Tensor, AttentionKVCache]: ...

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
        position_ids: Tensor | None = None,
        kv_cache: AttentionKVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, AttentionKVCache]:
        """Run Grouped-Query Self-Attention over ``x [B, L, d_model]``.

        Args:
            x: Input sequence embeddings of shape ``[B, L, d_model]``.
            attn_mask: Optional boolean (``True`` = attend) or additive float bias mask
                broadcastable to ``[B, num_heads, L, L_kv]``.
            key_padding_mask: Optional boolean mask of shape ``[B, L_kv]`` (``True`` = valid).
            is_causal: Optional per-call override of ``self.is_causal``.
            position_ids: Optional explicit integer/float positions ``[L]`` or ``[B, L]``.
            kv_cache: Optional prior :class:`AttentionKVCache` for incremental decoding.
            use_cache: If ``True``, return ``(output, updated_kv_cache)``.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x shape [B, L, {self.d_model}], got {tuple(x.shape)}")

        batch_size, q_len, _ = x.shape
        past_len = kv_cache.seq_len if kv_cache is not None else 0
        causal_flag = self.is_causal if is_causal is None else bool(is_causal)

        q = self.q_proj(x).reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(x)
            .reshape(batch_size, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        if self.rope is not None:
            if position_ids is not None:
                cos, sin = self.rope.from_positions(position_ids)
                cos = cos.to(device=x.device)
                sin = sin.to(device=x.device)
            else:
                cos, sin = self.rope(q_len, device=x.device, offset=past_len)
            q, k = apply_rope_1d(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat((kv_cache.k, k), dim=2)
            v = torch.cat((kv_cache.v, v), dim=2)

        next_cache = AttentionKVCache(k=k, v=v)
        kv_len = k.shape[2]

        if self.num_kv_groups > 1:
            k_attn = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_attn = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_attn = k
            v_attn = v

        dropout_p = self.dropout_p if self.training else 0.0
        can_use_sdpa_fastpath = (
            attn_mask is None
            and key_padding_mask is None
            and not (
                q.device.type == "mps"
                and dropout_p > 0.0
                and not (q.requires_grad or k_attn.requires_grad or v_attn.requires_grad)
            )
        )

        if can_use_sdpa_fastpath:
            if not causal_flag or (causal_flag and q_len == 1 and past_len > 0):
                attn_out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn, attn_mask=None, dropout_p=dropout_p, is_causal=False
                )
            elif q_len == kv_len:
                attn_out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn, attn_mask=None, dropout_p=dropout_p, is_causal=True
                )
            else:
                row_idx = torch.arange(past_len, past_len + q_len, device=x.device).unsqueeze(1)
                col_idx = torch.arange(kv_len, device=x.device).unsqueeze(0)
                sdpa_mask = col_idx <= row_idx
                attn_out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn, attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=False
                )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k_attn.transpose(-2, -1)) * scale
            valid_mask = torch.ones(
                (batch_size, 1, q_len, kv_len), dtype=torch.bool, device=x.device
            )
            if causal_flag:
                row_idx = torch.arange(past_len, past_len + q_len, device=x.device).unsqueeze(1)
                col_idx = torch.arange(kv_len, device=x.device).unsqueeze(0)
                valid_mask = valid_mask & (col_idx <= row_idx).view(1, 1, q_len, kv_len)
            if key_padding_mask is not None:
                if key_padding_mask.shape != (batch_size, kv_len):
                    raise ValueError(
                        f"key_padding_mask shape {tuple(key_padding_mask.shape)} != "
                        f"{(batch_size, kv_len)}"
                    )
                valid_mask = valid_mask & key_padding_mask.to(torch.bool).view(
                    batch_size, 1, 1, kv_len
                )
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    valid_mask = valid_mask & attn_mask
                else:
                    scores = scores + attn_mask.to(dtype=scores.dtype)
            probs = safe_masked_softmax(scores, valid_mask, dim=-1)
            if dropout_p > 0.0:
                probs = F.dropout(probs, p=dropout_p, training=self.training)
            attn_out = torch.matmul(probs, v_attn)

        merged = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out: Tensor = self.resid_dropout(self.o_proj(merged))
        if use_cache:
            return out, next_cache
        return out


class SpatialBiasedCrossAttention(nn.Module):
    """3D Spatial/RBF/Coulomb-biased Multi-Head Cross-Attention with Flamingo ``tanh`` gating.

    Allows query tokens ``x [B, L, d_model]`` (either 3D ligand/complex tokens with
    ``query_coords [B, L, 3]`` or 1D sequence tokens with ``query_coords=None``) to attend
    to 3D pocket/complex memory ``memory [B, M, d_memory]`` with:
      1. Exact ``SE(3)``-invariant Bessel RBF + smooth cosine-cutoff spatial bias.
      2. Optional long-range screened :math:`1/r` Debye-Huckel electrostatic bias
         (:class:`~geneuclidean.models.encoders.electrostatics.ScreenedCoulombBias`).
      3. Exact zero-weight padding isolation over ``memory_mask`` (`False` = dummy atom).
      4. Flamingo-style learnable ``tanh(gate)`` residual modulation preserving pretrained
         query fluency.
    """

    d_model: int
    d_memory: int
    num_heads: int
    head_dim: int
    num_rbf: int
    cutoff: float
    use_coulomb_bias: bool
    gated: bool

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        *,
        d_memory: int | None = None,
        num_rbf: int = 16,
        cutoff: float = 15.0,
        use_coulomb_bias: bool = True,
        use_qk_norm: bool = True,
        gated: bool = True,
        gate_init: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1:
            raise ValueError(f"d_model ({d_model}) and num_heads ({num_heads}) must be >= 1")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        resolved_mem = int(d_memory) if d_memory is not None else int(d_model)
        if resolved_mem < 1:
            raise ValueError(f"d_memory must be >= 1, got {resolved_mem}")

        self.d_model = int(d_model)
        self.d_memory = resolved_mem
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.use_coulomb_bias = bool(use_coulomb_bias)
        self.gated = bool(gated)

        self.norm_q = RMSNorm(self.d_model)
        self.norm_kv = RMSNorm(self.d_memory)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_memory, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_memory, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.qk_norm: QKNorm | None = (
            QKNorm(self.head_dim, num_q_heads=self.num_heads, num_kv_heads=self.num_heads)
            if use_qk_norm
            else None
        )

        self.rbf = BesselRBF(num_rbf=self.num_rbf, cutoff=self.cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=self.cutoff)
        self.rbf_bias_proj = nn.Linear(self.num_rbf, self.num_heads)
        self.pocket_saliency_proj = nn.Linear(self.num_rbf, self.num_heads)

        self.coulomb_q: ScreenedCoulombBias | None
        if self.use_coulomb_bias:
            self.coulomb_q = ScreenedCoulombBias(
                scalar_dim=self.d_model, num_heads=self.num_heads, debye_len=8.0
            )
            self.mem_charge_proj = nn.Sequential(
                nn.Linear(self.d_memory, max(16, self.d_memory // 4)),
                nn.SiLU(),
                nn.Linear(max(16, self.d_memory // 4), 1),
            )
        else:
            self.coulomb_q = None
            self.register_module("mem_charge_proj", None)

        self.attn_dropout = nn.Dropout(float(dropout))
        self.resid_dropout = nn.Dropout(float(dropout))
        if self.gated:
            self.gate: nn.Parameter | None = nn.Parameter(
                torch.tensor(float(gate_init), dtype=torch.float32)
            )
        else:
            self.register_parameter("gate", None)

    def _compute_spatial_bias(
        self,
        q_norm: Tensor,
        kv_norm: Tensor,
        mem_mask: Tensor,
        query_coords: Tensor | None,
        memory_coords: Tensor | None,
        query_mask: Tensor | None,
    ) -> Tensor | None:
        """Compute ``SE(3)``-invariant spatial bias ``[B, H, L, M]`` or ``[B, H, 1, M]``."""
        if memory_coords is None:
            return None
        batch_size, mem_len = mem_mask.shape
        if memory_coords.shape != (batch_size, mem_len, 3):
            raise ValueError(
                f"Expected memory_coords shape {(batch_size, mem_len, 3)}, "
                f"got {tuple(memory_coords.shape)}"
            )
        centered_mem, mem_offset = center_masked_coords_with_offset(memory_coords, mem_mask)

        if query_coords is not None:
            q_len = q_norm.shape[1]
            if query_coords.shape != (batch_size, q_len, 3):
                raise ValueError(
                    f"Expected query_coords shape {(batch_size, q_len, 3)}, "
                    f"got {tuple(query_coords.shape)}"
                )
            q_mask_b = (
                query_mask.to(torch.bool)
                if query_mask is not None
                else torch.ones((batch_size, q_len), device=q_norm.device, dtype=torch.bool)
            )
            q_coords_f = query_coords.to(torch.float32)
            centered_q = torch.where(
                q_mask_b.unsqueeze(-1),
                q_coords_f - mem_offset.unsqueeze(1),
                torch.zeros_like(q_coords_f),
            )
            diff = centered_q.unsqueeze(2) - centered_mem.unsqueeze(1)  # [B, L, M, 3]
            dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
            rbf_env = self.rbf(dist) * self.cutoff_fn(dist).unsqueeze(-1)
            pair_mask = q_mask_b.unsqueeze(2) & mem_mask.unsqueeze(1)
            bias_rbf: Tensor = self.rbf_bias_proj(rbf_env)
            bias_rbf = torch.where(pair_mask.unsqueeze(-1), bias_rbf, torch.zeros_like(bias_rbf))
            bias = bias_rbf.permute(0, 3, 1, 2).contiguous()

            if self.coulomb_q is not None and self.mem_charge_proj is not None:
                charges_q = self.coulomb_q.predict_charges(q_norm, atom_mask=q_mask_b)
                charges_m = torch.tanh(self.mem_charge_proj(kv_norm).squeeze(-1))
                charges_m = torch.where(mem_mask, charges_m, torch.zeros_like(charges_m))
                coul_bias = self.coulomb_q(
                    centered_q,
                    mask_i=q_mask_b,
                    coords_j=centered_mem,
                    mask_j=mem_mask,
                    charges_i=charges_q,
                    charges_j=charges_m,
                )
                bias = bias + coul_bias
            return bias

        # 1D sequence queries attending to 3D pocket coordinates: compute SE(3)-invariant
        # radial-shell pocket saliency per memory atom from centered_mem.
        radii = torch.sqrt((centered_mem * centered_mem).sum(dim=-1) + 1e-8)
        radial_rbf = self.rbf(radii) * self.cutoff_fn(radii).unsqueeze(-1)
        radial_rbf = torch.where(mem_mask.unsqueeze(-1), radial_rbf, torch.zeros_like(radial_rbf))
        mem_bias: Tensor = self.pocket_saliency_proj(radial_rbf)  # [B, M, H]
        mem_bias = torch.where(mem_mask.unsqueeze(-1), mem_bias, torch.zeros_like(mem_bias))
        return mem_bias.permute(0, 2, 1).unsqueeze(2).contiguous()  # [B, H, 1, M]

    @overload
    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None = None,
        query_coords: Tensor | None = None,
        memory_coords: Tensor | None = None,
        query_mask: Tensor | None = None,
        spatial_bias: Tensor | None = None,
        return_attn: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None = None,
        query_coords: Tensor | None = None,
        memory_coords: Tensor | None = None,
        query_mask: Tensor | None = None,
        spatial_bias: Tensor | None = None,
        return_attn: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None = None,
        query_coords: Tensor | None = None,
        memory_coords: Tensor | None = None,
        query_mask: Tensor | None = None,
        spatial_bias: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Attend from ``x [B, L, d_model]`` into 3D ``memory [B, M, d_memory]`` with residual."""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x shape [B, L, {self.d_model}], got {tuple(x.shape)}")
        if memory.ndim != 3 or memory.shape[0] != x.shape[0] or memory.shape[-1] != self.d_memory:
            raise ValueError(
                f"Expected memory shape [{x.shape[0]}, M, {self.d_memory}], "
                f"got {tuple(memory.shape)}"
            )

        batch_size, q_len, _ = x.shape
        mem_len = memory.shape[1]
        if memory_mask is None:
            mem_mask_b = torch.ones((batch_size, mem_len), device=memory.device, dtype=torch.bool)
        else:
            if memory_mask.shape != (batch_size, mem_len):
                raise ValueError(
                    f"memory_mask shape {tuple(memory_mask.shape)} != {(batch_size, mem_len)}"
                )
            mem_mask_b = memory_mask.to(torch.bool)

        safe_memory = torch.where(mem_mask_b.unsqueeze(-1), memory, torch.zeros_like(memory))
        q_norm = self.norm_q(x)
        kv_norm = torch.where(
            mem_mask_b.unsqueeze(-1),
            self.norm_kv(safe_memory),
            torch.zeros_like(safe_memory),
        )

        q = (
            self.q_proj(q_norm)
            .reshape(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(kv_norm)
            .reshape(batch_size, mem_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(kv_norm)
            .reshape(batch_size, mem_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        geom_bias = self._compute_spatial_bias(
            q_norm, kv_norm, mem_mask_b, query_coords, memory_coords, query_mask
        )
        if geom_bias is not None:
            scores = scores + geom_bias
        if spatial_bias is not None:
            scores = scores + spatial_bias

        key_mask = mem_mask_b.view(batch_size, 1, 1, mem_len)
        if query_mask is not None:
            key_mask = key_mask & query_mask.to(torch.bool).view(batch_size, 1, q_len, 1)

        attn_weights = safe_masked_softmax(scores, key_mask, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_probs, v)
        merged = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        cross_out: Tensor = self.resid_dropout(self.o_proj(merged))

        out = x + torch.tanh(self.gate) * cross_out if self.gate is not None else x + cross_out
        if query_mask is not None:
            out = torch.where(query_mask.to(torch.bool).unsqueeze(-1), out, torch.zeros_like(out))

        if return_attn:
            return out, attn_weights
        return out


class _PerceiverResamplerBlock(nn.Module):
    """Single cross-attention + bidirectional latent self-attention + SwiGLU Perceiver block."""

    def __init__(
        self,
        d_out: int,
        num_heads: int,
        num_rbf: int,
        cutoff: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross = SpatialBiasedCrossAttention(
            d_model=d_out,
            num_heads=num_heads,
            d_memory=d_out,
            num_rbf=num_rbf,
            cutoff=cutoff,
            use_coulomb_bias=False,
            use_qk_norm=True,
            gated=False,
            dropout=dropout,
        )
        self.norm_self = RMSNorm(d_out)
        self.self_attn = GroupedQueryAttention(
            d_model=d_out,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            dropout=dropout,
            use_qk_norm=True,
            use_rope=False,
            is_causal=False,
        )
        self.norm_ffn = RMSNorm(d_out)
        self.ffn = SwiGLU(d_model=d_out, dropout=dropout)

    def forward(
        self,
        latents: Tensor,
        memory: Tensor,
        mask: Tensor | None,
        coords: Tensor | None,
    ) -> Tensor:
        latents = self.cross(latents, memory, memory_mask=mask, memory_coords=coords)
        latents = latents + self.self_attn(self.norm_self(latents), is_causal=False)
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


class PerceiverLatentResampler(nn.Module):
    """Compress variable-length masked 3D atom clouds ``[B, N, D_in]`` into ``[B, K, D_out]``.

    Strictly permutation-invariant over input atoms ``N``, ``SE(3)``-invariant when optional 3D
    ``coords [B, N, 3]`` are supplied, and exact-zero invariant to appended padded dummy atoms
    (``mask == False``).
    """

    d_in: int
    d_out: int
    num_latents: int
    num_layers: int
    num_heads: int

    def __init__(
        self,
        d_in: int,
        d_out: int | None = None,
        *,
        num_latents: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        num_rbf: int = 16,
        cutoff: float = 15.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_in < 1:
            raise ValueError(f"d_in must be >= 1, got {d_in}")
        resolved_out = int(d_out) if d_out is not None else int(d_in)
        if resolved_out < 1 or resolved_out % num_heads != 0:
            raise ValueError(
                f"d_out ({resolved_out}) must be >= 1 and divisible by num_heads ({num_heads})"
            )
        if num_latents < 1 or num_layers < 1:
            raise ValueError(
                f"num_latents ({num_latents}) and num_layers ({num_layers}) must be >= 1"
            )

        self.d_in = int(d_in)
        self.d_out = resolved_out
        self.num_latents = int(num_latents)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        self.latents = nn.Parameter(
            torch.randn((1, self.num_latents, self.d_out), dtype=torch.float32) * 0.02
        )
        self.in_proj: nn.Module = (
            nn.Linear(self.d_in, self.d_out, bias=False)
            if self.d_in != self.d_out
            else nn.Identity()
        )
        self.coord_rbf = BesselRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.coord_cutoff = CosineCutoff(cutoff=cutoff)
        self.coord_proj = nn.Linear(num_rbf, self.d_out, bias=False)

        self.blocks = nn.ModuleList(
            [
                _PerceiverResamplerBlock(
                    d_out=self.d_out,
                    num_heads=self.num_heads,
                    num_rbf=num_rbf,
                    cutoff=cutoff,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = RMSNorm(self.d_out)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        *,
        coords: Tensor | None = None,
    ) -> Tensor:
        """Resample ``tokens [B, N, d_in]`` into fixed-size latents ``[B, num_latents, d_out]``."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_in:
            raise ValueError(
                f"Expected tokens shape [B, N, {self.d_in}], got {tuple(tokens.shape)}"
            )
        batch_size, num_tokens, _ = tokens.shape
        if mask is None:
            mask_b = torch.ones((batch_size, num_tokens), device=tokens.device, dtype=torch.bool)
        else:
            if mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != tokens[:2] {(batch_size, num_tokens)}"
                )
            mask_b = mask.to(torch.bool)

        safe_tokens = torch.where(mask_b.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        memory: Tensor = self.in_proj(safe_tokens)

        if coords is not None:
            centered, _ = center_masked_coords_with_offset(coords, mask_b)
            radii = torch.sqrt((centered * centered).sum(dim=-1) + 1e-8)
            geom_rbf = self.coord_rbf(radii) * self.coord_cutoff(radii).unsqueeze(-1)
            geom_emb = self.coord_proj(geom_rbf)
            memory = memory + torch.where(
                mask_b.unsqueeze(-1), geom_emb, torch.zeros_like(geom_emb)
            )

        memory = torch.where(mask_b.unsqueeze(-1), memory, torch.zeros_like(memory))
        latents = self.latents.expand(batch_size, -1, -1).to(dtype=tokens.dtype)

        for block in self.blocks:
            latents = block(latents, memory, mask=mask_b, coords=coords)

        out = self.final_norm(latents)
        has_any = mask_b.any(dim=1, keepdim=True).unsqueeze(-1)
        return torch.where(has_any, out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# 3. SE(3)-Equivariant Neural & Geometric Transformer Layers
# ---------------------------------------------------------------------------


class EquivariantScalarVectorGate(nn.Module):
    """Norm-preserving ``O(3)``-equivariant gated nonlinearity coupling scalars and 3D vectors.

    Couples invariant scalar channels ``scalars [B, N, D_s]`` and equivariant ``l=1`` vector
    channels ``vectors [B, N, D_v, 3]`` via rotation-invariant vector magnitudes and inner
    products (PaiNN / GVP / Equiformer lineage) while smoothly bounding vector magnitudes via
    :class:`~geneuclidean.models.encoders.rbf.EquivariantRMSNorm` and ``2 * sigmoid`` gating.
    """

    scalar_dim: int
    vector_dim: int
    hidden_dim: int
    eps: float

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        hidden_dim: int | None = None,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1:
            raise ValueError(
                f"scalar_dim ({scalar_dim}) and vector_dim ({vector_dim}) must be >= 1"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else 2 * self.scalar_dim
        self.eps = float(eps)

        self.vec_u = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.vec_v = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.vec_norm = EquivariantRMSNorm(self.vector_dim, eps=self.eps)

        in_fused = self.scalar_dim + 2 * self.vector_dim
        self.scalar_gate_mlp = nn.Sequential(
            nn.Linear(in_fused, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2 * self.scalar_dim + self.vector_dim),
        )
        self.scalar_out = nn.Linear(self.scalar_dim, self.scalar_dim)

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return updated ``(scalars_out [B, N, D_s], vectors_out [B, N, D_v, 3])``."""
        if scalars.ndim != 3 or scalars.shape[-1] != self.scalar_dim:
            raise ValueError(
                f"Expected scalars shape [B, N, {self.scalar_dim}], got {tuple(scalars.shape)}"
            )
        if (
            vectors.ndim != 4
            or vectors.shape[:2] != scalars.shape[:2]
            or vectors.shape[2:] != (self.vector_dim, 3)
        ):
            raise ValueError(
                f"Expected vectors shape {(*scalars.shape[:2], self.vector_dim, 3)}, "
                f"got {tuple(vectors.shape)}"
            )

        if mask is not None:
            mask_b = mask.to(torch.bool)
            scalars = torch.where(mask_b.unsqueeze(-1), scalars, torch.zeros_like(scalars))
            vectors = torch.where(
                mask_b.unsqueeze(-1).unsqueeze(-1), vectors, torch.zeros_like(vectors)
            )
        else:
            mask_b = None

        u = self.vec_u(vectors)
        v = self.vec_v(vectors)

        # Smooth O(3)-invariant norms vanishing at v == 0, plus channel inner products <u, v>.
        v_norm = torch.sqrt((v * v).sum(dim=-1) + self.eps) - math.sqrt(self.eps)
        uv_dot = (u * v).sum(dim=-1)
        uv_bounded = uv_dot * torch.rsqrt(uv_dot * uv_dot + 1.0)

        fused = torch.cat((scalars, v_norm, uv_bounded), dim=-1)
        raw_out: Tensor = self.scalar_gate_mlp(fused)
        s_val, s_gate, v_gate_logits = torch.split(
            raw_out,
            [self.scalar_dim, self.scalar_dim, self.vector_dim],
            dim=-1,
        )

        s_out: Tensor = self.scalar_out(s_val * torch.sigmoid(s_gate))
        u_stabilized = self.vec_norm(u)
        v_gate = 2.0 * torch.sigmoid(v_gate_logits).unsqueeze(-1)
        v_out = u_stabilized * v_gate

        if mask_b is not None:
            s_out = torch.where(mask_b.unsqueeze(-1), s_out, torch.zeros_like(s_out))
            v_out = torch.where(mask_b.unsqueeze(-1).unsqueeze(-1), v_out, torch.zeros_like(v_out))
        return s_out, v_out


class TriangleMultiplicativePairUpdate(nn.Module):
    """Masked Edge-Pair Triangle Multiplicative Update (AlphaFold2/3 Pairformer lineage).

    Supports both exact :math:`O(N^3)` ``mode="dense"`` triangle contractions
    (``direction="outgoing"``, ``"incoming"``, or exact symmetry-preserving ``"symmetric"``)
    and :math:`O(N^2 K)` ``mode="knn"`` local geometric triangle updates over 3D :math:`k`-NN
    neighborhoods. Strictly isolates masked dummy atoms (``mask == False``) so padding leaves
    all valid ``(i, j)`` pair updates unchanged to floating-point precision.
    """

    d_pair: int
    d_hidden: int
    direction: Literal["outgoing", "incoming", "symmetric"]
    mode: Literal["dense", "knn"]
    max_neighbors: int
    cutoff: float

    def __init__(
        self,
        d_pair: int,
        *,
        d_hidden: int | None = None,
        direction: Literal["outgoing", "incoming", "symmetric"] = "outgoing",
        mode: Literal["dense", "knn"] = "dense",
        max_neighbors: int = 16,
        cutoff: float = 12.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_pair < 1:
            raise ValueError(f"d_pair must be >= 1, got {d_pair}")
        if direction not in ("outgoing", "incoming", "symmetric"):
            raise ValueError(
                f"direction must be 'outgoing', 'incoming', or 'symmetric', got {direction!r}"
            )
        if mode not in ("dense", "knn"):
            raise ValueError(f"mode must be 'dense' or 'knn', got {mode!r}")
        if max_neighbors < 1 or cutoff <= 0.0:
            raise ValueError("max_neighbors must be >= 1 and cutoff must be > 0")

        self.d_pair = int(d_pair)
        self.d_hidden = int(d_hidden) if d_hidden is not None else int(d_pair)
        self.direction = direction
        self.mode = mode
        self.max_neighbors = int(max_neighbors)
        self.cutoff = float(cutoff)

        self.norm_in = RMSNorm(self.d_pair)
        self.proj_a = nn.Linear(self.d_pair, self.d_hidden, bias=False)
        self.gate_a = nn.Linear(self.d_pair, self.d_hidden, bias=True)

        if self.direction == "symmetric":
            self.proj_b: nn.Linear | None = None
            self.gate_b: nn.Linear | None = None
        else:
            self.proj_b = nn.Linear(self.d_pair, self.d_hidden, bias=False)
            self.gate_b = nn.Linear(self.d_pair, self.d_hidden, bias=True)

        self.norm_mid = RMSNorm(self.d_hidden)
        self.proj_out = nn.Linear(self.d_hidden, self.d_pair, bias=False)
        self.gate_out = nn.Linear(self.d_pair, self.d_pair, bias=True)
        self.dropout = nn.Dropout(float(dropout))

    def _knn_indices(
        self,
        z_norm: Tensor,
        mask_b: Tensor,
        coords: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Resolve local intermediate triangle vertices ``nbr_idx [B, N, K]`` and ``nbr_mask``."""
        batch_size, num_nodes, _, _ = z_norm.shape
        device = z_norm.device
        if num_nodes <= 1:
            return (
                torch.zeros((batch_size, num_nodes, 1), dtype=torch.long, device=device),
                torch.zeros((batch_size, num_nodes, 1), dtype=torch.bool, device=device),
            )
        if coords is not None:
            graph = build_dense_knn_graph(
                coords,
                mask_b,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
            )
            return graph.nbr_idx, graph.nbr_mask

        k_eff = min(self.max_neighbors, num_nodes - 1)
        with torch.no_grad():
            pair_strength = (z_norm * z_norm).sum(dim=-1)
            eye = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0)
            valid_pair = mask_b.unsqueeze(2) & mask_b.unsqueeze(1) & (~eye)
            masked_scores = pair_strength.masked_fill(~valid_pair, float("-inf"))
            top_vals, top_idx = torch.topk(masked_scores, k=k_eff, dim=-1, largest=True)
            nbr_mask = torch.isfinite(top_vals) & valid_pair.gather(-1, top_idx)
            safe_idx = torch.where(nbr_mask, top_idx, torch.zeros_like(top_idx))
        return safe_idx, nbr_mask

    def forward(
        self,
        pair_feats: Tensor,
        mask: Tensor | None = None,
        *,
        coords: Tensor | None = None,
    ) -> Tensor:
        """Apply masked triangle multiplicative update to ``pair_feats [B, N, N, d_pair]``."""
        if (
            pair_feats.ndim != 4
            or pair_feats.shape[1] != pair_feats.shape[2]
            or pair_feats.shape[-1] != self.d_pair
        ):
            raise ValueError(
                f"Expected pair_feats shape [B, N, N, {self.d_pair}], got {tuple(pair_feats.shape)}"
            )

        batch_size, num_nodes, _, _ = pair_feats.shape
        if mask is None:
            mask_b = torch.ones((batch_size, num_nodes), device=pair_feats.device, dtype=torch.bool)
        else:
            if mask.shape != (batch_size, num_nodes):
                raise ValueError(f"mask shape {tuple(mask.shape)} != {(batch_size, num_nodes)}")
            mask_b = mask.to(torch.bool)

        pair_mask = mask_b.unsqueeze(2) & mask_b.unsqueeze(1)  # [B, N, N]
        pm_4d = pair_mask.unsqueeze(-1)
        zeros_pair = torch.zeros_like(pair_feats)
        safe_z = torch.where(pm_4d, pair_feats, zeros_pair)
        if batch_size == 0 or num_nodes == 0:
            return safe_z

        z_norm = torch.where(pm_4d, self.norm_in(safe_z), zeros_pair)
        pm_f = pm_4d.to(dtype=z_norm.dtype)

        a = (self.proj_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))) * pm_f
        if self.direction == "symmetric":
            b = a
        else:
            assert self.proj_b is not None and self.gate_b is not None
            b = (self.proj_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))) * pm_f

        if self.mode == "dense" or self.direction == "symmetric":
            if self.direction == "incoming":
                tri = torch.einsum("bkic,bkjc->bijc", a, b)
            else:
                tri = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            nbr_idx, nbr_mask = self._knn_indices(z_norm, mask_b, coords)
            k_eff = nbr_idx.shape[-1]
            # Gather k along the shared intermediate vertex for each node i in O(N^2 * K).
            idx_exp = nbr_idx.unsqueeze(-1).expand(batch_size, num_nodes, k_eff, self.d_hidden)
            nbr_f = nbr_mask.unsqueeze(-1).to(dtype=a.dtype)
            if self.direction == "outgoing":
                # sum_{k in N(i)} a[b, i, k, c] * b[b, j, k, c]
                a_ik = torch.gather(a, dim=2, index=idx_exp) * nbr_f  # [B, N, K, C]
                b_perm = b.permute(0, 2, 1, 3)  # [B, K_full=N, N_j, C]
                b_gathered = gather_neighbors(b_perm, nbr_idx)  # [B, N_i, K, N_j, C]
                tri = (a_ik.unsqueeze(3) * b_gathered).sum(dim=2)  # [B, N_i, N_j, C]
            else:
                # incoming: sum_{k in N(i)} a[b, k, i, c] * b[b, k, j, c]
                a_ki_all = a.permute(0, 2, 1, 3)  # [B, N_i, N_k, C]
                a_ki = torch.gather(a_ki_all, dim=2, index=idx_exp) * nbr_f  # [B, N_i, K, C]
                b_kj = gather_neighbors(b, nbr_idx)  # [B, N_i, K, N_j, C]
                tri = (a_ki.unsqueeze(3) * b_kj).sum(dim=2)

        # Scale by inverse sqrt of valid nodes so representations stay O(1) across pocket sizes.
        valid_counts = mask_b.sum(dim=-1).clamp(min=1).to(dtype=tri.dtype).view(batch_size, 1, 1, 1)
        tri = tri * torch.rsqrt(valid_counts)
        tri = torch.where(pm_4d.expand_as(tri), tri, torch.zeros_like(tri))

        delta = self.proj_out(self.norm_mid(tri)) * torch.sigmoid(self.gate_out(z_norm))
        delta = self.dropout(delta)
        return torch.where(pm_4d, safe_z + delta, zeros_pair)


class SE3TransformerOutput(NamedTuple):
    """Output tuple emitted by :class:`SE3GeometricTransformerBlock`.

    Attributes:
        scalars: Updated ``SE(3)``-invariant scalar features ``[B, N, scalar_dim]``.
        vectors: Updated ``SO(3)``-equivariant 3D vector features ``[B, N, vector_dim, 3]``.
        pair_feats: Updated ``SE(3)``-invariant pair representation ``[B, N, N, pair_dim]``.
    """

    scalars: Tensor
    vectors: Tensor
    pair_feats: Tensor


class SE3GeometricTransformerBlock(nn.Module):
    """Composable Pre-Norm ``SE(3)``-Equivariant Geometric Transformer Block in ``float32``.

    Jointly updates invariant node scalars ``[B, N, D_s]``, equivariant 3D vectors
    ``[B, N, D_v, 3]``, and pairwise edge representations ``[B, N, N, D_z]`` using:
      1. Centroid-subtracted pairwise Bessel RBF + smooth cosine envelope geometry (eliminating
         floating-point cancellation under large rigid translations :math:`\\mathbf{t}`).
      2. Optional :class:`TriangleMultiplicativePairUpdate` over ``pair_feats``.
      3. Pre-norm conditioning via :class:`AdaLNZero` (when ``cond`` is provided) or
         :class:`RMSNorm` (when ``cond is None``) alongside
         :class:`~geneuclidean.models.encoders.rbf.EquivariantRMSNorm` on 3D vector channels.
      4. Multi-head geometric scalar + Frobenius-vector attention with :class:`QKNorm` and
         optional signed ``SE(3)`` chiral Levi-Civita triple/cross-product coupling
         (``use_se3_chirality=True`` distinguishes :math:`(R)/(S)` enantiomers while preserving
         exact ``SO(3)`` proper-rotation equivariance; ``use_se3_chirality=False`` enforces full
         :math:`E(3)` mirror symmetry).
      5. :class:`EquivariantScalarVectorGate` feed-forward transition + :class:`DropPath`.
    """

    scalar_dim: int
    vector_dim: int
    pair_dim: int
    num_heads: int
    head_scalar_dim: int
    head_vector_dim: int
    use_se3_chirality: bool
    use_triangle_update: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        pair_dim: int = 32,
        num_heads: int = 4,
        cond_dim: int | None = None,
        num_rbf: int = 16,
        cutoff: float = 10.0,
        use_se3_chirality: bool = True,
        use_triangle_update: bool = True,
        triangle_mode: Literal["dense", "knn"] = "dense",
        max_neighbors: int = 16,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        zero_init_adaln: bool = False,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1 or pair_dim < 1 or num_heads < 1:
            raise ValueError("scalar_dim, vector_dim, pair_dim, and num_heads must all be >= 1")
        if scalar_dim % num_heads != 0:
            raise ValueError(
                f"scalar_dim ({scalar_dim}) must be divisible by num_heads ({num_heads})"
            )
        if vector_dim % num_heads != 0:
            raise ValueError(
                f"vector_dim ({vector_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.pair_dim = int(pair_dim)
        self.num_heads = int(num_heads)
        self.head_scalar_dim = self.scalar_dim // self.num_heads
        self.head_vector_dim = self.vector_dim // self.num_heads
        self.use_se3_chirality = bool(use_se3_chirality)
        self.use_triangle_update = bool(use_triangle_update)

        resolved_cond = int(cond_dim) if cond_dim is not None else self.scalar_dim
        self.cond_dim = resolved_cond

        # Pair geometry & optional triangle multiplicative update
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)
        self.rbf_to_pair = nn.Linear(num_rbf, self.pair_dim, bias=False)
        self.scalar_to_pair = nn.Linear(2 * self.scalar_dim, self.pair_dim, bias=False)
        self.triangle_update: TriangleMultiplicativePairUpdate | None = (
            TriangleMultiplicativePairUpdate(
                d_pair=self.pair_dim,
                direction="outgoing",
                mode=triangle_mode,
                max_neighbors=max_neighbors,
                cutoff=cutoff,
                dropout=dropout,
            )
            if self.use_triangle_update
            else None
        )
        self.norm_pair = RMSNorm(self.pair_dim)
        self.pair_to_heads = nn.Linear(self.pair_dim, self.num_heads, bias=False)
        self.attn_to_pair = nn.Linear(self.num_heads, self.pair_dim, bias=False)

        # Pre-norms & AdaLNZero conditioning for Attention and FFN sublayers
        self.norm_s1 = RMSNorm(self.scalar_dim)
        self.norm_v1 = EquivariantRMSNorm(self.vector_dim)
        self.adaln_s1 = AdaLNZero(
            self.scalar_dim,
            resolved_cond,
            zero_init=zero_init_adaln,
            gate_bias_init=1.0,
        )
        self.cond_vec_gate1 = nn.Linear(resolved_cond, self.vector_dim, bias=True)
        if zero_init_adaln:
            nn.init.zeros_(self.cond_vec_gate1.weight)
            nn.init.zeros_(self.cond_vec_gate1.bias)
        else:
            nn.init.normal_(self.cond_vec_gate1.weight, mean=0.0, std=0.02)
            nn.init.ones_(self.cond_vec_gate1.bias)

        # Multi-head scalar + equivariant vector projections
        self.q_s = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.k_s = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.v_s = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.qk_norm = QKNorm(
            self.head_scalar_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
        )

        self.q_v = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.k_v = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.v_v = VectorNeuronLinear(self.vector_dim, self.vector_dim)

        self.edge_dir_gate = nn.Linear(self.pair_dim, self.vector_dim, bias=False)
        self.out_s = nn.Linear(self.scalar_dim + self.vector_dim, self.scalar_dim, bias=False)
        self.out_v = VectorNeuronLinear(2 * self.vector_dim, self.vector_dim)

        # Optional intrinsic SE(3) chiral triple-product & axial-cross branch
        if self.use_se3_chirality:
            self.chiral_edge_weights = nn.Linear(self.pair_dim, 3 * self.num_heads, bias=False)
            self.chiral_scalar_proj = nn.Linear(self.num_heads, self.scalar_dim, bias=False)
            self.chiral_vec_a = VectorNeuronLinear(self.vector_dim, self.vector_dim)
            self.chiral_vec_b = VectorNeuronLinear(self.vector_dim, self.vector_dim)
            self.chiral_vec_scale = nn.Parameter(
                torch.full((1, 1, self.vector_dim, 1), 0.25, dtype=torch.float32)
            )
        else:
            self.register_module("chiral_edge_weights", None)
            self.register_module("chiral_scalar_proj", None)
            self.register_module("chiral_vec_a", None)
            self.register_module("chiral_vec_b", None)
            self.register_parameter("chiral_vec_scale", None)

        # Pre-norm EquivariantScalarVectorGate FFN sublayer
        self.norm_s2 = RMSNorm(self.scalar_dim)
        self.norm_v2 = EquivariantRMSNorm(self.vector_dim)
        self.adaln_s2 = AdaLNZero(
            self.scalar_dim,
            resolved_cond,
            zero_init=zero_init_adaln,
            gate_bias_init=1.0,
        )
        self.cond_vec_gate2 = nn.Linear(resolved_cond, self.vector_dim, bias=True)
        if zero_init_adaln:
            nn.init.zeros_(self.cond_vec_gate2.weight)
            nn.init.zeros_(self.cond_vec_gate2.bias)
        else:
            nn.init.normal_(self.cond_vec_gate2.weight, mean=0.0, std=0.02)
            nn.init.ones_(self.cond_vec_gate2.bias)

        self.sv_ffn = EquivariantScalarVectorGate(
            scalar_dim=self.scalar_dim,
            vector_dim=self.vector_dim,
            dropout=dropout,
        )
        self.drop_path = DropPath(drop_path)
        self.attn_dropout = nn.Dropout(float(dropout))

    def _cond_vec_gate(self, cond: Tensor, proj: nn.Linear, num_nodes: int) -> Tensor:
        """Project conditioning ``cond`` to broadcastable vector gate ``[B, 1|N, D_v, 1]``."""
        c_act = F.silu(cond.to(dtype=torch.float32))
        g: Tensor = proj(c_act)
        if g.ndim == 2:
            return g.view(g.shape[0], 1, self.vector_dim, 1)
        if g.ndim == 3 and g.shape[1] in (1, num_nodes):
            return g.unsqueeze(-1)
        raise ValueError(f"Unsupported cond shape {tuple(cond.shape)} for {num_nodes} nodes")

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        coords: Tensor,
        *,
        pair_feats: Tensor | None = None,
        mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> SE3TransformerOutput:
        """Run one SE(3)-equivariant geometric transformer layer."""
        if scalars.ndim != 3 or scalars.shape[-1] != self.scalar_dim:
            raise ValueError(
                f"Expected scalars shape [B, N, {self.scalar_dim}], got {tuple(scalars.shape)}"
            )
        batch_size, num_nodes, _ = scalars.shape
        if vectors.shape != (batch_size, num_nodes, self.vector_dim, 3):
            raise ValueError(
                f"Expected vectors shape {(batch_size, num_nodes, self.vector_dim, 3)}, "
                f"got {tuple(vectors.shape)}"
            )
        if coords.shape != (batch_size, num_nodes, 3):
            raise ValueError(
                f"Expected coords shape {(batch_size, num_nodes, 3)}, got {tuple(coords.shape)}"
            )

        if mask is None:
            mask_b = torch.ones((batch_size, num_nodes), device=scalars.device, dtype=torch.bool)
        else:
            if mask.shape != (batch_size, num_nodes):
                raise ValueError(f"mask shape {tuple(mask.shape)} != {(batch_size, num_nodes)}")
            mask_b = mask.to(torch.bool)

        m_s = mask_b.unsqueeze(-1)
        m_v = mask_b.unsqueeze(-1).unsqueeze(-1)
        pair_mask = mask_b.unsqueeze(2) & mask_b.unsqueeze(1)
        m_p = pair_mask.unsqueeze(-1)

        s = torch.where(m_s, scalars, torch.zeros_like(scalars))
        v = torch.where(m_v, vectors, torch.zeros_like(vectors))
        centered_coords, _ = center_masked_coords_with_offset(coords, mask_b)

        # 1. Pairwise SE(3)-invariant geometry in centroid frame
        rel_vec = centered_coords.unsqueeze(2) - centered_coords.unsqueeze(1)  # [B, N, N, 3]
        sq_dist = (rel_vec * rel_vec).sum(dim=-1)
        dist = torch.sqrt(sq_dist + 1e-8)
        eye = torch.eye(num_nodes, device=scalars.device, dtype=torch.bool).unsqueeze(0)
        off_diag_valid = pair_mask & (~eye)
        env = self.cutoff_fn(dist) * off_diag_valid.to(dtype=dist.dtype)
        rbf_env = self.rbf(dist) * env.unsqueeze(-1)
        rel_dir = torch.where(
            off_diag_valid.unsqueeze(-1),
            rel_vec * torch.rsqrt(sq_dist + 1e-6).unsqueeze(-1),
            torch.zeros_like(rel_vec),
        )

        s_for_pair = self.norm_s1(s)
        s_pair_in = torch.cat(
            (
                s_for_pair.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, self.scalar_dim),
                s_for_pair.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, self.scalar_dim),
            ),
            dim=-1,
        )
        z_geom = self.rbf_to_pair(rbf_env) + self.scalar_to_pair(s_pair_in)
        if pair_feats is not None:
            expected_pair = (batch_size, num_nodes, num_nodes, self.pair_dim)
            if pair_feats.shape != expected_pair:
                raise ValueError(
                    f"Expected pair_feats shape {expected_pair}, got {tuple(pair_feats.shape)}"
                )
            z = torch.where(m_p, pair_feats + z_geom, torch.zeros_like(z_geom))
        else:
            z = torch.where(m_p, z_geom, torch.zeros_like(z_geom))

        if self.triangle_update is not None:
            z = self.triangle_update(z, mask=mask_b, coords=centered_coords)

        # 2. Pre-norm + optional AdaLNZero conditioning
        if cond is not None:
            h_s1, gate_s1 = self.adaln_s1(s, cond)
            gate_v1 = self._cond_vec_gate(cond, self.cond_vec_gate1, num_nodes)
        else:
            h_s1 = self.norm_s1(s)
            gate_s1 = torch.ones((1, 1, 1), device=s.device, dtype=s.dtype)
            gate_v1 = torch.ones((1, 1, 1, 1), device=v.device, dtype=v.dtype)

        h_s1 = torch.where(m_s, h_s1, torch.zeros_like(h_s1))
        h_v1 = torch.where(m_v, self.norm_v1(v), torch.zeros_like(v))

        # 3. Multi-head scalar + Frobenius-vector geometric attention
        q_s = (
            self.q_s(h_s1)
            .reshape(batch_size, num_nodes, self.num_heads, self.head_scalar_dim)
            .transpose(1, 2)
        )
        k_s = (
            self.k_s(h_s1)
            .reshape(batch_size, num_nodes, self.num_heads, self.head_scalar_dim)
            .transpose(1, 2)
        )
        v_s = (
            self.v_s(h_s1)
            .reshape(batch_size, num_nodes, self.num_heads, self.head_scalar_dim)
            .transpose(1, 2)
        )
        q_s, k_s = self.qk_norm(q_s, k_s)

        q_v = (
            self.q_v(h_v1)
            .reshape(batch_size, num_nodes, self.num_heads, self.head_vector_dim * 3)
            .permute(0, 2, 1, 3)
        )
        k_v = (
            self.k_v(h_v1)
            .reshape(batch_size, num_nodes, self.num_heads, self.head_vector_dim * 3)
            .permute(0, 2, 1, 3)
        )
        v_v = (
            self.v_v(h_v1)
            .reshape(batch_size, num_nodes, self.num_heads, self.head_vector_dim, 3)
            .permute(0, 2, 1, 3, 4)
        )

        logits_s = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.head_scalar_dim)
        logits_v = torch.matmul(q_v, k_v.transpose(-2, -1)) / math.sqrt(3.0 * self.head_vector_dim)
        z_norm = torch.where(m_p, self.norm_pair(z), torch.zeros_like(z))
        pair_bias = self.pair_to_heads(z_norm).permute(0, 3, 1, 2)

        logits = logits_s + logits_v + pair_bias
        key_mask = mask_b.view(batch_size, 1, 1, num_nodes) & mask_b.view(
            batch_size, 1, num_nodes, 1
        )
        attn = safe_masked_softmax(logits, key_mask, dim=-1)
        attn_drop = self.attn_dropout(attn)

        agg_s = (
            torch.matmul(attn_drop, v_s)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, num_nodes, self.scalar_dim)
        )
        agg_v = (
            torch.einsum("bhij,bhjda->bhida", attn_drop, v_v)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, num_nodes, self.vector_dim, 3)
        )

        # Inject equivariant radial displacement vectors weighted by attention & pair features
        attn_mean = attn_drop.mean(dim=1)  # [B, N, N]
        edge_weights = torch.tanh(self.edge_dir_gate(z_norm)) * (attn_mean * env).unsqueeze(-1)
        agg_dir = torch.einsum("bijc,bijd->bicd", edge_weights, rel_dir)

        v_combined = torch.cat((agg_v, agg_dir), dim=2)
        delta_v_attn = self.out_v(v_combined)
        vec_invariants = torch.sqrt((delta_v_attn * delta_v_attn).sum(dim=-1) + 1e-6) - 1e-3
        delta_s_attn = self.out_s(torch.cat((agg_s, vec_invariants), dim=-1))

        # Intrinsic SE(3) chiral triple-product scalar & cross-product vector coupling
        if (
            self.use_se3_chirality
            and self.chiral_edge_weights is not None
            and self.chiral_scalar_proj is not None
            and self.chiral_vec_a is not None
            and self.chiral_vec_b is not None
            and self.chiral_vec_scale is not None
        ):
            # Build 3 distinct learned polar neighbor-displacement combinations per node:
            # Under x -> R x + t (R in O(3)), u_k -> u_k R^T, so <u_1 x u_2, u_3> transforms as
            # det(R) * <u_1 x u_2, u_3>: exact SO(3) invariant, flips sign on chiral enantiomers,
            # and vanishes identically on planar achiral point sets!
            w_chiral = torch.tanh(self.chiral_edge_weights(z_norm)) * env.unsqueeze(-1)
            w1, w2, w3 = w_chiral.chunk(3, dim=-1)  # each [B, N, N, H]
            u1 = torch.einsum("bijh,bijd->bihd", w1, rel_dir)
            u2 = torch.einsum("bijh,bijd->bihd", w2, rel_dir)
            u3 = torch.einsum("bijh,bijd->bihd", w3, rel_dir)
            axial_12 = chiral_cross_product(u1, u2, normalize=True, eps=1.0)
            triple_prod = (axial_12 * u3).sum(dim=-1)
            triple_bounded = triple_prod * torch.rsqrt(triple_prod * triple_prod + 1.0)
            delta_s_attn = delta_s_attn + self.chiral_scalar_proj(triple_bounded)

            # Equivariant SO(3) vector cross-product coupling (polar x polar -> SO(3) vector)
            v_cand = delta_v_attn + h_v1
            v_cross = chiral_cross_product(
                self.chiral_vec_a(v_cand),
                self.chiral_vec_b(agg_dir + h_v1),
                normalize=True,
                eps=1.0,
            )
            delta_v_attn = delta_v_attn + self.chiral_vec_scale * v_cross

        s = torch.where(m_s, s + self.drop_path(gate_s1 * delta_s_attn), torch.zeros_like(s))
        v = torch.where(m_v, v + self.drop_path(gate_v1 * delta_v_attn), torch.zeros_like(v))

        # 4. Pre-norm EquivariantScalarVectorGate FFN sublayer
        if cond is not None:
            h_s2, gate_s2 = self.adaln_s2(s, cond)
            gate_v2 = self._cond_vec_gate(cond, self.cond_vec_gate2, num_nodes)
        else:
            h_s2 = self.norm_s2(s)
            gate_s2 = torch.ones((1, 1, 1), device=s.device, dtype=s.dtype)
            gate_v2 = torch.ones((1, 1, 1, 1), device=v.device, dtype=v.dtype)

        h_s2 = torch.where(m_s, h_s2, torch.zeros_like(h_s2))
        h_v2 = torch.where(m_v, self.norm_v2(v), torch.zeros_like(v))
        delta_s_ffn, delta_v_ffn = self.sv_ffn(h_s2, h_v2, mask=mask_b)

        s_out = torch.where(m_s, s + self.drop_path(gate_s2 * delta_s_ffn), torch.zeros_like(s))
        v_out = torch.where(m_v, v + self.drop_path(gate_v2 * delta_v_ffn), torch.zeros_like(v))

        # 5. Pair representation update from attention weights
        z_delta = self.attn_to_pair(attn.permute(0, 2, 3, 1))
        z_out = torch.where(m_p, z + self.drop_path(z_delta), torch.zeros_like(z))

        return SE3TransformerOutput(scalars=s_out, vectors=v_out, pair_feats=z_out)


__all__ = [
    "AdaLNZero",
    "AttentionKVCache",
    "DropPath",
    "EquivariantRMSNorm",
    "EquivariantScalarVectorGate",
    "GEGLU",
    "GroupedQueryAttention",
    "PerceiverLatentResampler",
    "QKNorm",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SE3GeometricTransformerBlock",
    "SE3TransformerOutput",
    "SinusoidalFourierTimeEmbedding",
    "SpatialBiasedCrossAttention",
    "SwiGLU",
    "TriangleMultiplicativePairUpdate",
    "apply_rope_1d",
]
