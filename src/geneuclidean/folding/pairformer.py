"""Reusable AlphaFold-3 Pairformer & Apple SimpleFold O(L^2) Transformer Primitives.

Implements pure-PyTorch ``torch.float32`` structural trunk primitives tuned for Apple Silicon
MPS, CUDA, and CPU (zero CUDA-only extensions, zero ``float64`` parameters/buffers):

- :class:`SwiGLUTransition`: AF3 Alg. 11 gated feed-forward transition rounded to 32-wide SIMD.
- :class:`TriangleMultiplicativeUpdate`: AF3 Alg. 12/13 outgoing & incoming updates with
  empirically verified MPS kernel dispatch (``einsum`` for ``c_hidden <= 64``,
  ``permute + matmul`` for ``c_hidden > 64``).
- :class:`TriangleAttention`: AF3 Alg. 14/15 chunked starting & ending attention enforcing
  ``head_dim >= 32`` so Apple MPS stays on Metal's fused Flash-SDPA SIMD fast path.
- :class:`PairBiasedAttention`: AF3 Alg. 24 pair-biased multi-head attention with optional
  AdaLN-Zero conditioning, 1D/3D RoPE, and external electrostatic bias injection.
- :class:`PairformerBlock` & :class:`PairformerStack`: AF3 Alg. 17 single + pair stack with
  optional low-rank outer-product mean and gradient checkpointing.
- :class:`SimpleFoldTransformerBlock` & :class:`SimpleFoldTrunk`: Apple MLR ``O(L^2)``
  triangle-free AdaLN-Zero + 1D/3D RoPE DiT trunk with a ``"hybrid"`` Pairformer toggle.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from geneuclidean.models.encoders.electrostatics import ScreenedCoulombBias


def _round_to_multiple(value: int, multiple: int = 32) -> int:
    """Round a positive channel width to the nearest positive multiple of ``multiple``."""
    return max(multiple, int(round(value / multiple)) * multiple)


def _normalize_pair_mask(pair_mask: Tensor | None, batch: int, length: int) -> Tensor | None:
    """Coerce a ``[B, L]`` token mask or ``[B, L, L]`` pair mask into ``[B, L, L]`` bool."""
    if pair_mask is None:
        return None
    if pair_mask.ndim == 2:
        if pair_mask.shape != (batch, length):
            raise ValueError(
                f"Expected token mask shape {(batch, length)}, got {tuple(pair_mask.shape)}"
            )
        m = pair_mask.to(torch.bool)
        return m.unsqueeze(2) & m.unsqueeze(1)
    if pair_mask.ndim == 3:
        if pair_mask.shape != (batch, length, length):
            raise ValueError(
                f"Expected pair mask shape {(batch, length, length)}, got {tuple(pair_mask.shape)}"
            )
        return pair_mask.to(torch.bool)
    raise ValueError(f"Expected 2D or 3D mask, got shape {tuple(pair_mask.shape)}")


def _init_residual_proj(linear: nn.Linear, *, zero_init: bool = False) -> None:
    """Initialize a residual output projection with either zeros or tiny residual scale."""
    if zero_init:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _rotate_half_pairs(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply pairwise 2D rotation ``(x_even, x_odd) -> (x_e*c - x_o*s, x_o*c + x_e*s)``."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def apply_rope_1d_3d(
    q: Tensor,
    k: Tensor,
    *,
    positions: Tensor | None = None,
    coords: Tensor | None = None,
    base: float = 10000.0,
    coord_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Apply pure-``float32`` 1D sequential or 3D axial Rotary Position Embeddings (RoPE).

    When ``coords`` (shape ``[B, L, 3]``) is supplied, partitions ``head_dim`` into three
    even axial subspaces ``(x, y, z)`` rotated by scaled 3D coordinates (plus any remaining
    even dimensions rotated by 1D token positions). Otherwise applies standard 1D RoPE over
    ``positions`` (defaulting to ``0..L-1``).

    Args:
        q: Query tensor of shape ``[B, H, L, dh]`` (even ``dh``).
        k: Key tensor of shape ``[B, H, L, dh]`` (even ``dh``).
        positions: Optional 1D token positions of shape ``[B, L]`` or ``[L]``.
        coords: Optional 3D token coordinates of shape ``[B, L, 3]``.
        base: Geometric frequency base (default ``10000.0``).
        coord_scale: Spatial frequency scale in ``rad / Angstrom`` for 3D RoPE.

    Returns:
        Tuple ``(q_rot, k_rot)`` matching input shapes and ``torch.float32`` dtype.
    """
    batch_size, _, seq_len, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    device = q.device

    if positions is None:
        pos_1d = (
            torch.arange(seq_len, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
    else:
        pos_1d = positions.to(device=device, dtype=torch.float32)
        if pos_1d.ndim == 1:
            pos_1d = pos_1d.unsqueeze(0).expand(batch_size, seq_len)

    if coords is None:
        half = head_dim // 2
        idx = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = torch.pow(float(base), -(2.0 * idx) / float(head_dim))  # [dh/2]
        angles = pos_1d.unsqueeze(-1) * inv_freq.view(1, 1, half)  # [B, L, dh/2]
        cos = torch.cos(angles).unsqueeze(1)  # [B, 1, L, dh/2]
        sin = torch.sin(angles).unsqueeze(1)
        return _rotate_half_pairs(q, cos, sin), _rotate_half_pairs(k, cos, sin)

    if coords.ndim != 3 or coords.shape != (batch_size, seq_len, 3):
        raise ValueError(
            f"Expected coords shape {(batch_size, seq_len, 3)}, got {tuple(coords.shape)}"
        )

    # 3D axial RoPE: split head_dim into 3 equal even blocks (x, y, z) + optional 1D remainder.
    d_axis = (head_dim // 6) * 2
    d_rem = head_dim - 3 * d_axis
    coords_f = coords.to(device=device, dtype=torch.float32) * float(coord_scale)

    cos_parts: list[Tensor] = []
    sin_parts: list[Tensor] = []
    if d_axis > 0:
        half_ax = d_axis // 2
        idx_ax = torch.arange(half_ax, device=device, dtype=torch.float32)
        inv_freq_ax = torch.pow(float(base), -(2.0 * idx_ax) / float(d_axis))
        for axis in range(3):
            ang = coords_f[..., axis : axis + 1] * inv_freq_ax.view(1, 1, half_ax)
            cos_parts.append(torch.cos(ang))
            sin_parts.append(torch.sin(ang))

    if d_rem > 0:
        half_rem = d_rem // 2
        idx_rem = torch.arange(half_rem, device=device, dtype=torch.float32)
        inv_freq_rem = torch.pow(float(base), -(2.0 * idx_rem) / float(d_rem))
        ang_rem = pos_1d.unsqueeze(-1) * inv_freq_rem.view(1, 1, half_rem)
        cos_parts.append(torch.cos(ang_rem))
        sin_parts.append(torch.sin(ang_rem))

    cos_all = torch.cat(cos_parts, dim=-1).unsqueeze(1)  # [B, 1, L, dh/2]
    sin_all = torch.cat(sin_parts, dim=-1).unsqueeze(1)
    return _rotate_half_pairs(q, cos_all, sin_all), _rotate_half_pairs(k, cos_all, sin_all)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with gated residual modulation (DiT / SimpleFold / AF3 Alg. 26).

    Produces ``(x_modulated, gate)`` from conditioning vector ``cond`` (``[B, c_cond]`` or
    ``[B, L, c_cond]``). When ``zero_init=True``, initializes the gate to exact ``0.0``; when
    ``zero_init=False`` (default), uses a small residual gain ``gate_init=0.1`` so residual
    blocks start near identity while propagating finite non-zero gradients on step 0.
    """

    dim: int
    c_cond: int

    def __init__(
        self,
        dim: int,
        c_cond: int,
        *,
        zero_init: bool = False,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if dim < 1 or c_cond < 1:
            raise ValueError(f"dim ({dim}) and c_cond ({c_cond}) must be >= 1")
        self.dim = int(dim)
        self.c_cond = int(c_cond)
        self.norm_x = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.norm_c = nn.LayerNorm(self.c_cond)
        self.proj = nn.Linear(self.c_cond, 3 * self.dim, bias=True)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.proj.bias)
            with torch.no_grad():
                self.proj.bias[2 * self.dim :].fill_(float(gate_init))

    def forward(self, x: Tensor, cond: Tensor | None) -> tuple[Tensor, Tensor]:
        """Return ``(modulated_x [B, L, dim], gate [B, L, dim])``."""
        x_norm = self.norm_x(x)
        if cond is None:
            return x_norm, torch.ones_like(x_norm)
        c = cond.to(dtype=x.dtype)
        if c.ndim == 2:
            c = c.unsqueeze(1)
        params = self.proj(F.silu(self.norm_c(c)))
        shift, scale, gate = params.chunk(3, dim=-1)
        return x_norm * (1.0 + scale) + shift, gate


class SwiGLUTransition(nn.Module):
    """AlphaFold-3 Algorithm 11 SwiGLU gated feed-forward transition.

    Hidden width is set to ``int(dim * expansion * 2 / 3)`` rounded to a multiple of ``32``
    for optimal Apple Silicon Metal SIMD occupancy.
    """

    dim: int
    hidden_dim: int

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        *,
        use_norm: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        self.dim = int(dim)
        self.hidden_dim = _round_to_multiple(int(round(dim * expansion * 2.0 / 3.0)), 32)
        self.norm: nn.LayerNorm | None = nn.LayerNorm(self.dim) if use_norm else None
        self.w_ab = nn.Linear(self.dim, 2 * self.hidden_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        _init_residual_proj(self.w_o, zero_init=zero_init)

    def forward(self, x: Tensor, *, use_norm: bool = True) -> Tensor:
        """Apply SwiGLU transition to tensor ``[..., dim]``."""
        h = self.norm(x) if (use_norm and self.norm is not None) else x
        a, b = self.w_ab(h).chunk(2, dim=-1)
        gated = self.dropout(F.silu(a) * b)
        out: Tensor = self.w_o(gated)
        return out


class TriangleMultiplicativeUpdate(nn.Module):
    """AlphaFold-3 Algorithms 12 (``outgoing``) & 13 (``incoming``) triangular update.

    Automatically dispatches the core O(L^3) contraction based on live Apple M3 Pro MPS
    profiling (see ``docs/alphafold3_nanofold_design.md`` §1.3):
    - ``c_hidden <= 64``: fused ``torch.einsum`` (fastest at compact channel widths).
    - ``c_hidden > 64`` (including ``c_z >= 128``): ``permute + torch.matmul`` (1.67x faster
      on M3 Pro Metal GEMM).
    """

    c_z: int
    c_hidden: int
    mode: Literal["outgoing", "incoming"]

    def __init__(
        self,
        c_z: int,
        c_hidden: int | None = None,
        mode: Literal["outgoing", "incoming"] = "outgoing",
        dropout: float = 0.0,
        *,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if c_z < 1:
            raise ValueError(f"c_z must be >= 1, got {c_z}")
        if mode not in ("outgoing", "incoming"):
            raise ValueError(f"mode must be 'outgoing' or 'incoming', got {mode!r}")
        hidden = int(c_z if c_hidden is None else c_hidden)
        if hidden < 1:
            raise ValueError(f"c_hidden must be >= 1, got {hidden}")

        self.c_z = int(c_z)
        self.c_hidden = hidden
        self.mode = mode

        self.norm_in = nn.LayerNorm(self.c_z)
        self.proj_a = nn.Linear(self.c_z, self.c_hidden, bias=False)
        self.gate_a = nn.Linear(self.c_z, self.c_hidden, bias=False)
        self.proj_b = nn.Linear(self.c_z, self.c_hidden, bias=False)
        self.gate_b = nn.Linear(self.c_z, self.c_hidden, bias=False)
        self.norm_out = nn.LayerNorm(self.c_hidden)
        self.proj_out = nn.Linear(self.c_hidden, self.c_z, bias=False)
        self.gate_out = nn.Linear(self.c_z, self.c_z, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        _init_residual_proj(self.proj_out, zero_init=zero_init)

    @property
    def use_matmul_dispatch(self) -> bool:
        """Return ``True`` when the contraction dispatches to ``permute + matmul``."""
        return self.c_hidden > 64

    def _contract(
        self,
        a: Tensor,
        b: Tensor,
        *,
        force_method: Literal["auto", "einsum", "matmul"] = "auto",
    ) -> Tensor:
        """Compute triangular contraction over ``k`` with MPS-optimal kernel dispatch."""
        if force_method == "auto":
            use_matmul = self.use_matmul_dispatch
        elif force_method == "matmul":
            use_matmul = True
        elif force_method == "einsum":
            use_matmul = False
        else:
            raise ValueError(f"Unknown force_method {force_method!r}")

        if not use_matmul:
            if self.mode == "outgoing":
                return torch.einsum("bikc,bjkc->bijc", a, b)
            return torch.einsum("bkic,bkjc->bijc", a, b)

        a_c = a.permute(0, 3, 1, 2)  # [B, C, L, L]
        b_c = b.permute(0, 3, 1, 2)  # [B, C, L, L]
        if self.mode == "outgoing":
            o_c = torch.matmul(a_c, b_c.transpose(-1, -2))
        else:
            o_c = torch.matmul(a_c.transpose(-1, -2), b_c)
        return o_c.permute(0, 2, 3, 1).contiguous()

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor | None = None,
        *,
        force_method: Literal["auto", "einsum", "matmul"] = "auto",
    ) -> Tensor:
        """Compute residual update ``Delta z [B, L, L, c_z]``."""
        if z.ndim != 4 or z.shape[-1] != self.c_z:
            raise ValueError(f"Expected z shape [B, L, L, {self.c_z}], got {tuple(z.shape)}")
        batch_size, seq_len, seq_len_2, _ = z.shape
        if seq_len != seq_len_2:
            raise ValueError(f"Expected square pair tensor [B, L, L, c_z], got {tuple(z.shape)}")

        z_f = z.to(torch.float32)
        pm = _normalize_pair_mask(pair_mask, batch_size, seq_len)
        if pm is not None:
            pm_4d = pm.unsqueeze(-1)
            z_f = torch.where(pm_4d, z_f, torch.zeros_like(z_f))
        else:
            pm_4d = None

        z_norm = self.norm_in(z_f)
        a = torch.sigmoid(self.gate_a(z_norm)) * self.proj_a(z_norm)
        b = torch.sigmoid(self.gate_b(z_norm)) * self.proj_b(z_norm)
        if pm_4d is not None:
            a = torch.where(pm_4d, a, torch.zeros_like(a))
            b = torch.where(pm_4d, b, torch.zeros_like(b))

        o = self._contract(a, b, force_method=force_method)
        gate = torch.sigmoid(self.gate_out(z_norm))
        out: Tensor = gate * self.proj_out(self.dropout(self.norm_out(o)))
        if pm_4d is not None:
            out = torch.where(pm_4d, out, torch.zeros_like(out))
        return out


class TriangleAttention(nn.Module):
    """AlphaFold-3 Algorithms 14 (``starting``) & 15 (``ending``) chunked triangle attention.

    Enforces ``head_dim >= 32`` so PyTorch's ``F.scaled_dot_product_attention`` stays on
    Apple Metal's fused Flash-SDPA SIMD path (12.2x faster than ``head_dim=16`` on M3 Pro).
    Ending-node attention transposes ``(1, 2)`` on entry and exit so both orientations share
    one unified, chunk-safe row attention kernel with exact ``b_ki`` bias semantics.
    """

    c_z: int
    head_dim: int
    num_heads: int
    mode: Literal["starting", "ending"]
    chunk_size: int | None
    dropout_p: float

    def __init__(
        self,
        c_z: int,
        head_dim: int = 32,
        num_heads: int | None = None,
        mode: Literal["starting", "ending"] = "starting",
        chunk_size: int | None = None,
        dropout: float = 0.0,
        *,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if c_z < 1:
            raise ValueError(f"c_z must be >= 1, got {c_z}")
        if head_dim < 32:
            raise ValueError(
                f"TriangleAttention enforces head_dim >= 32 for Apple MPS Metal Flash-SDPA "
                f"fast path (got head_dim={head_dim})."
            )
        if mode not in ("starting", "ending"):
            raise ValueError(f"mode must be 'starting' or 'ending', got {mode!r}")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 when provided, got {chunk_size}")

        heads = max(1, c_z // head_dim) if num_heads is None else int(num_heads)
        if heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {heads}")

        self.c_z = int(c_z)
        self.head_dim = int(head_dim)
        self.num_heads = heads
        self.mode = mode
        self.chunk_size = chunk_size
        self.dropout_p = float(dropout)

        inner_dim = self.num_heads * self.head_dim
        self.norm = nn.LayerNorm(self.c_z)
        self.qkv = nn.Linear(self.c_z, 3 * inner_dim, bias=False)
        self.bias_proj = nn.Linear(self.c_z, self.num_heads, bias=False)
        self.gate_proj = nn.Linear(self.c_z, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, self.c_z, bias=False)
        _init_residual_proj(self.out_proj, zero_init=zero_init)

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor | None = None,
        *,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Compute triangular attention update ``Delta z [B, L, L, c_z]``."""
        if z.ndim != 4 or z.shape[-1] != self.c_z:
            raise ValueError(f"Expected z shape [B, L, L, {self.c_z}], got {tuple(z.shape)}")
        batch_size, seq_len, seq_len_2, _ = z.shape
        if seq_len != seq_len_2:
            raise ValueError(f"Expected square pair tensor [B, L, L, c_z], got {tuple(z.shape)}")
        if batch_size == 0 or seq_len == 0:
            return torch.zeros_like(z, dtype=torch.float32)

        z_work = z.to(torch.float32)
        pm = _normalize_pair_mask(pair_mask, batch_size, seq_len)

        if self.mode == "ending":
            z_work = z_work.transpose(1, 2).contiguous()
            if pm is not None:
                pm = pm.transpose(1, 2).contiguous()

        if pm is not None:
            pm_4d = pm.unsqueeze(-1)
            z_work = torch.where(pm_4d, z_work, torch.zeros_like(z_work))
        else:
            pm_4d = None

        h = self.norm(z_work)  # [B, L, L, c_z]
        # Shared triangle bias b[b, h, j, k] from (j, k) pair features (or (k, i) for ending).
        tri_bias = self.bias_proj(h).permute(0, 3, 1, 2).contiguous()  # [B, H, L, L]
        if pm is not None:
            tri_bias = torch.where(pm.unsqueeze(1), tri_bias, torch.zeros_like(tri_bias))

        eff_chunk = chunk_size if chunk_size is not None else self.chunk_size
        if eff_chunk is None:
            eff_chunk = 32 if seq_len > 160 else seq_len

        drop_p = self.dropout_p if self.training else 0.0
        inner_dim = self.num_heads * self.head_dim
        out_chunks: list[Tensor] = []

        for i0 in range(0, seq_len, eff_chunk):
            i1 = min(i0 + eff_chunk, seq_len)
            cs = i1 - i0
            h_chunk = h[:, i0:i1, :, :]  # [B, cs, L, c_z]

            qkv_c = self.qkv(h_chunk).reshape(
                batch_size * cs, seq_len, 3, self.num_heads, self.head_dim
            )
            q_c = qkv_c[:, :, 0].transpose(1, 2)  # [B*cs, H, L, dh]
            k_c = qkv_c[:, :, 1].transpose(1, 2)
            v_c = qkv_c[:, :, 2].transpose(1, 2)

            # Expand shared triangle bias across the cs rows in this chunk: [B*cs, H, L, L]
            attn_bias = (
                tri_bias.unsqueeze(1)
                .expand(batch_size, cs, self.num_heads, seq_len, seq_len)
                .reshape(batch_size * cs, self.num_heads, seq_len, seq_len)
            )

            if pm is not None:
                # Key mask for row i: valid entries k in pair_mask[b, i, k].
                key_valid = pm[:, i0:i1, :].reshape(batch_size * cs, 1, 1, seq_len)
                has_any_key = key_valid.any(dim=-1, keepdim=True)  # [B*cs, 1, 1, 1]
                # Guard all-empty rows (e.g. dummy tokens) with finite 0.0 mask before SDPA so
                # softmax never emits NaN in forward or backward; masked outputs are zeroed below.
                neg_inf = torch.full_like(attn_bias, float("-inf"))
                masked_bias = torch.where(key_valid, attn_bias, neg_inf)
                safe_bias = torch.where(has_any_key, masked_bias, torch.zeros_like(attn_bias))
            else:
                safe_bias = attn_bias

            attn_out = F.scaled_dot_product_attention(
                q_c,
                k_c,
                v_c,
                attn_mask=safe_bias,
                dropout_p=drop_p,
            )  # [B*cs, H, L, dh]
            attn_flat = (
                attn_out.transpose(1, 2).contiguous().reshape(batch_size, cs, seq_len, inner_dim)
            )
            gate = torch.sigmoid(self.gate_proj(h_chunk))
            chunk_out = self.out_proj(gate * attn_flat)  # [B, cs, L, c_z]
            if pm_4d is not None:
                chunk_out = torch.where(pm_4d[:, i0:i1], chunk_out, torch.zeros_like(chunk_out))
            out_chunks.append(chunk_out)

        out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]
        if self.mode == "ending":
            out = out.transpose(1, 2).contiguous()
        return out


class PairBiasedAttention(nn.Module):
    """AlphaFold-3 Algorithm 24 pair-biased multi-head self-attention.

    Supports:
    - Pairformer single-track updates (``use_adaln=False``, ``z`` provided),
    - DiffusionTransformer / SimpleFold updates (``use_adaln=True``, ``cond`` provided),
    - Optional 1D/3D RoPE (``use_rope=True``) and external additive biases (e.g.
      :class:`~geneuclidean.models.encoders.electrostatics.ScreenedCoulombBias`).
    - Enforces ``head_dim = c_s // num_heads >= 32`` for Apple MPS Flash-SDPA speed.
    """

    c_s: int
    c_z: int | None
    num_heads: int
    head_dim: int
    use_adaln: bool
    use_rope: bool
    dropout_p: float

    def __init__(
        self,
        c_s: int,
        c_z: int | None = None,
        num_heads: int = 4,
        use_adaln: bool = False,
        c_cond: int | None = None,
        use_rope: bool = False,
        dropout: float = 0.0,
        *,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if c_s < 1 or num_heads < 1:
            raise ValueError(f"c_s ({c_s}) and num_heads ({num_heads}) must be >= 1")
        if c_s % num_heads != 0:
            raise ValueError(f"c_s ({c_s}) must be divisible by num_heads ({num_heads})")
        head_dim = c_s // num_heads
        if head_dim < 32:
            raise ValueError(
                f"PairBiasedAttention enforces head_dim = c_s // num_heads >= 32 for Apple MPS "
                f"Flash-SDPA (got c_s={c_s}, num_heads={num_heads} -> head_dim={head_dim})."
            )

        self.c_s = int(c_s)
        self.c_z = int(c_z) if c_z is not None else None
        self.num_heads = int(num_heads)
        self.head_dim = head_dim
        self.use_adaln = bool(use_adaln)
        self.use_rope = bool(use_rope)
        self.dropout_p = float(dropout)

        self.adaln: AdaLNZero | None
        self.norm_s: nn.LayerNorm | None
        if self.use_adaln:
            self.adaln = AdaLNZero(
                self.c_s,
                int(c_cond if c_cond is not None else self.c_s),
                zero_init=zero_init,
            )
            self.norm_s = None
        else:
            self.adaln = None
            self.norm_s = nn.LayerNorm(self.c_s)

        self.qkv = nn.Linear(self.c_s, 3 * self.c_s, bias=False)
        self.gate_proj = nn.Linear(self.c_s, self.c_s, bias=False)
        self.out_proj = nn.Linear(self.c_s, self.c_s, bias=False)
        _init_residual_proj(self.out_proj, zero_init=zero_init)

        self.norm_z: nn.LayerNorm | None
        self.bias_proj: nn.Linear | None
        if self.c_z is not None:
            self.norm_z = nn.LayerNorm(self.c_z)
            self.bias_proj = nn.Linear(self.c_z, self.num_heads, bias=False)
        else:
            self.norm_z = None
            self.bias_proj = None

    def forward(
        self,
        s: Tensor,
        z: Tensor | None = None,
        mask: Tensor | None = None,
        cond: Tensor | None = None,
        extra_bias: Tensor | None = None,
        *,
        positions: Tensor | None = None,
        coords: Tensor | None = None,
    ) -> Tensor:
        """Compute pair-biased self-attention residual ``Delta s [B, L, c_s]``."""
        if s.ndim != 3 or s.shape[-1] != self.c_s:
            raise ValueError(f"Expected s shape [B, L, {self.c_s}], got {tuple(s.shape)}")
        batch_size, seq_len, _ = s.shape
        if batch_size == 0 or seq_len == 0:
            return torch.zeros_like(s, dtype=torch.float32)

        s_f = s.to(torch.float32)
        mask_b: Tensor | None
        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Expected mask shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
                )
            mask_b = mask.to(torch.bool)
            s_f = torch.where(mask_b.unsqueeze(-1), s_f, torch.zeros_like(s_f))
        else:
            mask_b = None

        if self.adaln is not None:
            h, adaln_gate = self.adaln(s_f, cond)
        else:
            assert self.norm_s is not None
            h = self.norm_s(s_f)
            adaln_gate = None

        qkv = self.qkv(h).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, L, dh]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope_1d_3d(q, k, positions=positions, coords=coords)

        attn_bias: Tensor | None = None
        if z is not None and self.norm_z is not None and self.bias_proj is not None:
            z_f = z.to(torch.float32)
            if mask_b is not None:
                pm = (mask_b.unsqueeze(2) & mask_b.unsqueeze(1)).unsqueeze(-1)
                z_f = torch.where(pm, z_f, torch.zeros_like(z_f))
            attn_bias = self.bias_proj(self.norm_z(z_f)).permute(0, 3, 1, 2).contiguous()

        if extra_bias is not None:
            eb = extra_bias.to(torch.float32)
            attn_bias = eb if attn_bias is None else (attn_bias + eb)

        if mask_b is not None:
            key_valid = mask_b.view(batch_size, 1, 1, seq_len)
            has_any_key = key_valid.any(dim=-1, keepdim=True)
            if attn_bias is None:
                attn_bias = torch.zeros(
                    (batch_size, 1, 1, seq_len), device=s.device, dtype=torch.float32
                )
            else:
                key_valid = key_valid.expand_as(attn_bias)
                has_any_key = has_any_key.expand(
                    batch_size, attn_bias.shape[1], attn_bias.shape[2], 1
                )
            neg_inf = torch.full_like(attn_bias, float("-inf"))
            masked_bias = torch.where(key_valid, attn_bias, neg_inf)
            attn_bias = torch.where(has_any_key, masked_bias, torch.zeros_like(attn_bias))

        drop_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)
        attn_flat = attn_out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.c_s)

        gated = torch.sigmoid(self.gate_proj(h)) * attn_flat
        out: Tensor = self.out_proj(gated)
        if adaln_gate is not None:
            out = adaln_gate * out
        if mask_b is not None:
            out = torch.where(mask_b.unsqueeze(-1), out, torch.zeros_like(out))
        return out


class OuterProductMean(nn.Module):
    """Low-rank single-to-pair outer-product update for MSA-free pocket co-folding."""

    def __init__(self, c_s: int, c_z: int, c_hidden: int = 16, *, zero_init: bool = False) -> None:
        super().__init__()
        self.c_s = int(c_s)
        self.c_z = int(c_z)
        self.c_hidden = int(c_hidden)
        self.norm = nn.LayerNorm(self.c_s)
        self.proj_a = nn.Linear(self.c_s, self.c_hidden, bias=False)
        self.proj_b = nn.Linear(self.c_s, self.c_hidden, bias=False)
        self.out_proj = nn.Linear(self.c_hidden * self.c_hidden, self.c_z, bias=True)
        _init_residual_proj(self.out_proj, zero_init=zero_init)

    def forward(self, s: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute low-rank outer-product pair update ``Delta z [B, L, L, c_z]``."""
        batch_size, seq_len, _ = s.shape
        s_f = s.to(torch.float32)
        if mask is not None:
            m_b = mask.to(torch.bool)
            s_f = torch.where(m_b.unsqueeze(-1), s_f, torch.zeros_like(s_f))
        else:
            m_b = None

        h = self.norm(s_f)
        a = self.proj_a(h)  # [B, L, c_h]
        b = self.proj_b(h)  # [B, L, c_h]
        outer = torch.einsum("bic,bjd->bijcd", a, b).reshape(
            batch_size, seq_len, seq_len, self.c_hidden * self.c_hidden
        )
        out: Tensor = self.out_proj(outer)
        if m_b is not None:
            pm = (m_b.unsqueeze(2) & m_b.unsqueeze(1)).unsqueeze(-1)
            out = torch.where(pm, out, torch.zeros_like(out))
        return out


class PairformerBlock(nn.Module):
    """AlphaFold-3 Algorithm 17 Pairformer block (pair triangle ops + single pair-biased attn)."""

    c_s: int
    c_z: int
    single_attention: bool

    def __init__(
        self,
        c_s: int,
        c_z: int,
        *,
        head_dim: int = 32,
        num_heads_s: int | None = None,
        num_heads_z: int | None = None,
        c_hidden_tri: int | None = None,
        transition_expansion: int = 4,
        chunk_size: int | None = None,
        outer_product_mean: bool = False,
        c_opm: int = 16,
        dropout: float = 0.0,
        single_attention: bool = True,
    ) -> None:
        super().__init__()
        if head_dim < 32:
            raise ValueError(
                f"PairformerBlock enforces head_dim >= 32 for Apple MPS Flash-SDPA "
                f"(got head_dim={head_dim})."
            )
        heads_s = max(1, c_s // head_dim) if num_heads_s is None else int(num_heads_s)
        heads_z = max(1, c_z // head_dim) if num_heads_z is None else int(num_heads_z)

        self.c_s = int(c_s)
        self.c_z = int(c_z)
        self.single_attention = bool(single_attention)

        self.opm: OuterProductMean | None = (
            OuterProductMean(self.c_s, self.c_z, c_hidden=c_opm) if outer_product_mean else None
        )
        self.tri_mult_out = TriangleMultiplicativeUpdate(
            self.c_z, c_hidden=c_hidden_tri, mode="outgoing", dropout=dropout
        )
        self.tri_mult_in = TriangleMultiplicativeUpdate(
            self.c_z, c_hidden=c_hidden_tri, mode="incoming", dropout=dropout
        )
        self.tri_attn_start = TriangleAttention(
            self.c_z,
            head_dim=head_dim,
            num_heads=heads_z,
            mode="starting",
            chunk_size=chunk_size,
            dropout=dropout,
        )
        self.tri_attn_end = TriangleAttention(
            self.c_z,
            head_dim=head_dim,
            num_heads=heads_z,
            mode="ending",
            chunk_size=chunk_size,
            dropout=dropout,
        )
        self.pair_transition = SwiGLUTransition(
            self.c_z, expansion=transition_expansion, dropout=dropout
        )

        self.single_attn: PairBiasedAttention | None
        self.single_transition: SwiGLUTransition | None
        if self.single_attention:
            self.single_attn = PairBiasedAttention(
                self.c_s,
                c_z=self.c_z,
                num_heads=heads_s,
                use_adaln=False,
                use_rope=False,
                dropout=dropout,
            )
            self.single_transition = SwiGLUTransition(
                self.c_s, expansion=transition_expansion, dropout=dropout
            )
        else:
            self.single_attn = None
            self.single_transition = None

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor | None = None,
        *,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one Pairformer block and return updated ``(s [B, L, c_s], z [B, L, L, c_z])``."""
        batch_size, seq_len, _ = s.shape
        pm = (
            _normalize_pair_mask(pair_mask, batch_size, seq_len)
            if pair_mask is not None
            else _normalize_pair_mask(mask, batch_size, seq_len)
        )
        m_b = mask.to(torch.bool) if mask is not None else None

        s_out = s.to(torch.float32)
        z_out = z.to(torch.float32)
        if m_b is not None:
            s_out = torch.where(m_b.unsqueeze(-1), s_out, torch.zeros_like(s_out))
        if pm is not None:
            z_out = torch.where(pm.unsqueeze(-1), z_out, torch.zeros_like(z_out))

        if self.opm is not None:
            z_out = z_out + self.opm(s_out, m_b)
        z_out = z_out + self.tri_mult_out(z_out, pm)
        z_out = z_out + self.tri_mult_in(z_out, pm)
        z_out = z_out + self.tri_attn_start(z_out, pm)
        z_out = z_out + self.tri_attn_end(z_out, pm)
        z_trans = self.pair_transition(z_out)
        if pm is not None:
            z_trans = torch.where(pm.unsqueeze(-1), z_trans, torch.zeros_like(z_trans))
        z_out = z_out + z_trans

        if self.single_attn is not None and self.single_transition is not None:
            s_out = s_out + self.single_attn(s_out, z_out, mask=m_b)
            s_trans = self.single_transition(s_out)
            if m_b is not None:
                s_trans = torch.where(m_b.unsqueeze(-1), s_trans, torch.zeros_like(s_trans))
            s_out = s_out + s_trans

        return s_out, z_out


class PairformerStack(nn.Module):
    """Stack of AlphaFold-3 :class:`PairformerBlock` layers with optional checkpointing."""

    num_blocks: int
    c_s: int
    c_z: int
    grad_checkpoint: bool

    def __init__(
        self,
        num_blocks: int,
        c_s: int,
        c_z: int,
        *,
        head_dim: int = 32,
        num_heads_s: int | None = None,
        num_heads_z: int | None = None,
        c_hidden_tri: int | None = None,
        transition_expansion: int = 4,
        chunk_size: int | None = None,
        outer_product_mean: bool = False,
        c_opm: int = 16,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.num_blocks = int(num_blocks)
        self.c_s = int(c_s)
        self.c_z = int(c_z)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    self.c_s,
                    self.c_z,
                    head_dim=head_dim,
                    num_heads_s=num_heads_s,
                    num_heads_z=num_heads_z,
                    c_hidden_tri=c_hidden_tri,
                    transition_expansion=transition_expansion,
                    chunk_size=chunk_size,
                    outer_product_mean=outer_product_mean,
                    c_opm=c_opm,
                    dropout=dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor | None = None,
        *,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run all Pairformer blocks sequentially."""
        s_cur, z_cur = s, z
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                s_cur, z_cur = checkpoint(
                    block,
                    s_cur,
                    z_cur,
                    mask,
                    pair_mask=pair_mask,
                    use_reentrant=False,
                )
            else:
                s_cur, z_cur = block(s_cur, z_cur, mask, pair_mask=pair_mask)
        return s_cur, z_cur


class SimpleFoldTransformerBlock(nn.Module):
    """Apple MLR SimpleFold ``O(L^2)`` AdaLN-Zero + 1D/3D RoPE DiT Transformer block.

    Uses factorized DiT-style 6-parameter AdaLN-Zero modulation (``shift/scale/gate`` for
    attention and SwiGLU MLP) so parameter count stays strictly leaner than ``PairformerBlock``.
    """

    c_s: int
    num_heads: int

    def __init__(
        self,
        c_s: int,
        num_heads: int,
        *,
        c_cond: int | None = None,
        use_rope: bool = True,
        transition_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_cond_eff = int(c_cond if c_cond is not None else c_s)
        mod_hidden = min(64, c_cond_eff)
        self.c_s = int(c_s)
        self.num_heads = int(num_heads)
        self.norm_attn = nn.LayerNorm(self.c_s, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(self.c_s, elementwise_affine=False)
        self.cond_norm = nn.LayerNorm(c_cond_eff)
        cond_out = nn.Linear(mod_hidden, 6 * self.c_s)
        nn.init.normal_(cond_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(cond_out.bias)
        with torch.no_grad():
            cond_out.bias[2 * self.c_s : 3 * self.c_s].fill_(0.1)
            cond_out.bias[5 * self.c_s :].fill_(0.1)
        self.cond_mod = nn.Sequential(
            nn.Linear(c_cond_eff, mod_hidden),
            nn.SiLU(),
            cond_out,
        )

        self.attn = PairBiasedAttention(
            self.c_s,
            c_z=None,
            num_heads=self.num_heads,
            use_adaln=False,
            use_rope=use_rope,
            dropout=dropout,
        )
        self.mlp = SwiGLUTransition(
            self.c_s, expansion=transition_expansion, dropout=dropout, use_norm=False
        )

    def forward(
        self,
        s: Tensor,
        mask: Tensor | None = None,
        *,
        cond: Tensor | None = None,
        extra_bias: Tensor | None = None,
        positions: Tensor | None = None,
        coords: Tensor | None = None,
    ) -> Tensor:
        """Apply AdaLN-Zero attention + AdaLN-Zero SwiGLU MLP."""
        s_f = s.to(torch.float32)
        m_b = mask.to(torch.bool) if mask is not None else None
        if m_b is not None:
            s_f = torch.where(m_b.unsqueeze(-1), s_f, torch.zeros_like(s_f))

        shift2: Tensor | None = None
        scale2: Tensor | None = None
        gate2: Tensor | None = None
        if cond is not None:
            c = cond.to(dtype=s_f.dtype)
            if c.ndim == 2:
                c = c.unsqueeze(1)
            mod: Tensor = self.cond_mod(F.silu(self.cond_norm(c)))
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
            h_attn = self.norm_attn(s_f) * (1.0 + scale1) + shift1
            delta_attn = gate1 * self.attn(
                h_attn,
                z=None,
                mask=m_b,
                extra_bias=extra_bias,
                positions=positions,
                coords=coords,
            )
        else:
            delta_attn = self.attn(
                s_f,
                z=None,
                mask=m_b,
                extra_bias=extra_bias,
                positions=positions,
                coords=coords,
            )

        if m_b is not None:
            delta_attn = torch.where(m_b.unsqueeze(-1), delta_attn, torch.zeros_like(delta_attn))
        s_mid = s_f + delta_attn

        if shift2 is not None and scale2 is not None and gate2 is not None:
            h_mlp = self.norm_mlp(s_mid) * (1.0 + scale2) + shift2
            delta_mlp = gate2 * self.mlp(h_mlp, use_norm=False)
        else:
            delta_mlp = self.mlp(self.norm_mlp(s_mid), use_norm=False)

        if m_b is not None:
            delta_mlp = torch.where(m_b.unsqueeze(-1), delta_mlp, torch.zeros_like(delta_mlp))
        out: Tensor = s_mid + delta_mlp
        return out


class SimpleFoldTrunk(nn.Module):
    """Apple MLR ``O(L^2)`` AdaLN-Zero + 1D/3D RoPE trunk with optional ``hybrid`` mode.

    Drop-in compatible with :class:`PairformerStack` (returns ``(s_out, z_out)``):
    - In ``trunk_mode="simplefold"``, condenses ``z_init`` into additive attention bias,
      executes ``num_layers`` fast ``O(L^2)`` AdaLN-Zero + RoPE SDPA blocks (with optional
      :class:`ScreenedCoulombBias` when 3D coordinates are provided), and synthesizes
      ``z_out`` via a single factorized low-rank bilinear readout.
    - In ``trunk_mode="hybrid"``, runs ``num_hybrid_pairformer_blocks`` (default ``2``)
      :class:`PairformerBlock` layers first to establish triangle-inequality geometric
      closure, followed by ``max(0, num_layers - num_hybrid_pairformer_blocks)`` fast
      ``O(L^2)`` :class:`SimpleFoldTransformerBlock` layers!
    """

    c_s: int
    c_z: int
    num_layers: int
    num_heads: int
    trunk_mode: Literal["simplefold", "hybrid"]
    use_coulomb_bias: bool
    grad_checkpoint: bool

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_layers: int = 8,
        num_heads: int | None = None,
        *,
        head_dim: int = 32,
        c_cond: int | None = None,
        trunk_mode: Literal["simplefold", "hybrid"] = "simplefold",
        num_hybrid_pairformer_blocks: int = 2,
        use_rope: bool = True,
        use_coulomb_bias: bool = True,
        transition_expansion: int = 4,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if head_dim < 32:
            raise ValueError(f"SimpleFoldTrunk enforces head_dim >= 32, got {head_dim}")
        if trunk_mode not in ("simplefold", "hybrid"):
            raise ValueError(f"trunk_mode must be 'simplefold' or 'hybrid', got {trunk_mode!r}")

        heads = max(1, c_s // head_dim) if num_heads is None else int(num_heads)
        if c_s % heads != 0 or (c_s // heads) < 32:
            raise ValueError(
                f"SimpleFoldTrunk requires c_s ({c_s}) divisible by num_heads ({heads}) "
                f"with head_dim >= 32."
            )

        self.c_s = int(c_s)
        self.c_z = int(c_z)
        self.num_layers = int(num_layers)
        self.num_heads = heads
        self.trunk_mode = trunk_mode
        self.use_coulomb_bias = bool(use_coulomb_bias)
        self.grad_checkpoint = bool(grad_checkpoint)

        n_pf = (
            min(int(num_hybrid_pairformer_blocks), self.num_layers) if trunk_mode == "hybrid" else 0
        )
        n_sf = self.num_layers - n_pf

        self.pairformer_prefix = nn.ModuleList(
            [
                PairformerBlock(
                    self.c_s,
                    self.c_z,
                    head_dim=head_dim,
                    num_heads_s=self.num_heads,
                    transition_expansion=transition_expansion,
                    dropout=dropout,
                )
                for _ in range(n_pf)
            ]
        )

        # Condense initial pair features z_init -> head-wise additive attention bias.
        self.z_init_norm = nn.LayerNorm(self.c_z)
        self.z_init_bias = nn.Linear(self.c_z, self.num_heads, bias=False)

        self.coulomb_bias: ScreenedCoulombBias | None = (
            ScreenedCoulombBias(scalar_dim=self.c_s, num_heads=self.num_heads)
            if self.use_coulomb_bias
            else None
        )
        c_cond_eff = int(c_cond if c_cond is not None else self.c_s)
        self.default_cond = nn.Parameter(torch.zeros((1, c_cond_eff), dtype=torch.float32))

        self.layers = nn.ModuleList(
            [
                SimpleFoldTransformerBlock(
                    self.c_s,
                    self.num_heads,
                    c_cond=c_cond_eff,
                    use_rope=use_rope,
                    transition_expansion=transition_expansion,
                    dropout=dropout,
                )
                for _ in range(n_sf)
            ]
        )

        # Factorized low-rank bilinear readout:
        # z_out = z + W_p(LN(W_l s_i) * LN(W_r s_j)) + W_1 s_i + W_2 s_j
        pair_rank = max(32, min(self.c_z, self.c_s // 2))
        self.s_out_norm = nn.LayerNorm(self.c_s)
        self.pair_left = nn.Linear(self.c_s, pair_rank, bias=False)
        self.pair_right = nn.Linear(self.c_s, pair_rank, bias=False)
        self.pair_left_norm = nn.LayerNorm(pair_rank)
        self.pair_right_norm = nn.LayerNorm(pair_rank)
        self.pair_bilinear_out = nn.Linear(pair_rank, self.c_z, bias=False)
        self.pair_row_proj = nn.Linear(self.c_s, self.c_z, bias=False)
        self.pair_col_proj = nn.Linear(self.c_s, self.c_z, bias=False)
        _init_residual_proj(self.pair_bilinear_out, zero_init=False)
        _init_residual_proj(self.pair_row_proj, zero_init=False)
        _init_residual_proj(self.pair_col_proj, zero_init=False)

    def forward(
        self,
        s: Tensor,
        z_init: Tensor,
        mask: Tensor | None = None,
        cond: Tensor | None = None,
        *,
        positions: Tensor | None = None,
        coords: Tensor | None = None,
        raw_node_feats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run O(L^2) SimpleFold (or hybrid) trunk returning ``(s_out, z_out)``."""
        batch_size, seq_len, _ = s.shape
        m_b = mask.to(torch.bool) if mask is not None else None
        pm = _normalize_pair_mask(m_b, batch_size, seq_len)

        s_cur = s.to(torch.float32)
        z_cur = z_init.to(torch.float32)
        if m_b is not None:
            s_cur = torch.where(m_b.unsqueeze(-1), s_cur, torch.zeros_like(s_cur))
        if pm is not None:
            z_cur = torch.where(pm.unsqueeze(-1), z_cur, torch.zeros_like(z_cur))

        for pf_block in self.pairformer_prefix:
            if self.grad_checkpoint and self.training:
                s_cur, z_cur = checkpoint(pf_block, s_cur, z_cur, m_b, use_reentrant=False)
            else:
                s_cur, z_cur = pf_block(s_cur, z_cur, m_b)

        # Layer-0 pair bias + optional long-range screened Coulomb electrostatic bias.
        pair_bias = self.z_init_bias(self.z_init_norm(z_cur)).permute(0, 3, 1, 2).contiguous()
        if pm is not None:
            pair_bias = torch.where(pm.unsqueeze(1), pair_bias, torch.zeros_like(pair_bias))

        if self.coulomb_bias is not None and coords is not None:
            coul_bias = self.coulomb_bias(
                coords,
                s_cur,
                raw_feats_i=raw_node_feats,
                mask_i=m_b,
            )
            pair_bias = pair_bias + coul_bias

        if cond is None:
            cond_eff = self.default_cond.expand(batch_size, -1)
        elif cond.ndim == 3:
            cond_eff = cond.to(torch.float32) + self.default_cond.unsqueeze(1)
        else:
            cond_eff = cond.to(torch.float32) + self.default_cond
        for idx, layer in enumerate(self.layers):
            layer_bias = pair_bias if idx == 0 else None
            if self.grad_checkpoint and self.training:
                s_cur = checkpoint(
                    layer,
                    s_cur,
                    m_b,
                    cond=cond_eff,
                    extra_bias=layer_bias,
                    positions=positions,
                    coords=coords,
                    use_reentrant=False,
                )
            else:
                s_cur = layer(
                    s_cur,
                    m_b,
                    cond=cond_eff,
                    extra_bias=layer_bias,
                    positions=positions,
                    coords=coords,
                )

        # Synthesize pair representation z_out via factorized low-rank bilinear readout.
        s_norm = self.s_out_norm(s_cur)
        left = self.pair_left_norm(self.pair_left(s_norm))  # [B, L, R]
        right = self.pair_right_norm(self.pair_right(s_norm))  # [B, L, R]
        prod = left.unsqueeze(2) * right.unsqueeze(1)  # [B, L, L, R]
        z_delta = (
            self.pair_bilinear_out(prod)
            + self.pair_row_proj(s_norm).unsqueeze(2)
            + self.pair_col_proj(s_norm).unsqueeze(1)
        )
        z_out = z_cur + z_delta
        if pm is not None:
            z_out = torch.where(pm.unsqueeze(-1), z_out, torch.zeros_like(z_out))
        if m_b is not None:
            s_cur = torch.where(m_b.unsqueeze(-1), s_cur, torch.zeros_like(s_cur))
        return s_cur, z_out


__all__ = [
    "AdaLNZero",
    "OuterProductMean",
    "PairBiasedAttention",
    "PairformerBlock",
    "PairformerStack",
    "SimpleFoldTransformerBlock",
    "SimpleFoldTrunk",
    "SwiGLUTransition",
    "TriangleAttention",
    "TriangleMultiplicativeUpdate",
    "apply_rope_1d_3d",
]
