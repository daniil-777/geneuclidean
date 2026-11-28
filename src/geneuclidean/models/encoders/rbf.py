"""Pure-PyTorch MPS-native 3D geometric primitives, RBF expansions, and pooling layers.

All operations strictly use ``torch.float32`` and native PyTorch tensor ops (no C++
``torch-cluster`` / ``torch-scatter`` extensions and no ``float64`` buffers), ensuring
identical behavior and exact numerical equivariance on CPU, CUDA, and Apple Silicon MPS.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


class PocketEncoderOutput(NamedTuple):
    """Unified output contract returned by every 3D protein pocket encoder.

    Attributes:
        node_emb: Invariant per-atom embeddings of shape ``[B, N, out_dim]`` with
            padded dummy positions strictly zeroed.
        pooled_emb: Permutation-invariant global pocket embedding of shape
            ``[B, out_dim]``.
        vec_emb: Optional equivariant ``l=1`` 3D vector features of shape
            ``[B, N, vector_dim, 3]`` (``None`` for purely distance-invariant encoders
            such as SchNet or ``lmax=0`` e3nn).
    """

    node_emb: Tensor
    pooled_emb: Tensor
    vec_emb: Tensor | None


class RadiusGraph(NamedTuple):
    """Sparse batched 3D radius/k-NN graph over valid (non-padded) pocket atoms.

    Node indices ``src`` and ``dst`` are flattened into ``[0, B * N)`` via
    ``batch_idx * N + local_idx``, enabling direct ``index_add_`` aggregation over
    flattened node tensors ``[B * N, ...]`` with zero cross-batch or dummy-atom edges.

    Attributes:
        src: Source (neighbor) flat node indices of shape ``[E]`` (``torch.long``).
        dst: Destination (target) flat node indices of shape ``[E]`` (``torch.long``).
        edge_vec: Relative displacement vectors ``coords[dst] - coords[src]`` of shape
            ``[E, 3]`` (``torch.float32``).
        edge_dist: Euclidean edge lengths ``||edge_vec||_2`` of shape ``[E]``.
        batch_idx: Batch item index in ``[0, B)`` for each edge of shape ``[E]``.
    """

    src: Tensor
    dst: Tensor
    edge_vec: Tensor
    edge_dist: Tensor
    batch_idx: Tensor


class CosineCutoff(nn.Module):
    """Smooth Behler-Parrinello cosine cutoff envelope vanishing at ``d >= cutoff``."""

    cutoff: float

    def __init__(self, cutoff: float = 5.5) -> None:
        super().__init__()
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate smooth cutoff weights in ``[0, 1]`` for pairwise distances."""
        clamped = distances.clamp(min=0.0, max=self.cutoff)
        envelope = 0.5 * (torch.cos(math.pi * clamped / self.cutoff) + 1.0)
        return envelope * (distances < self.cutoff).to(distances.dtype)


class BesselRBF(nn.Module):
    """Radial Bessel basis functions (Klicpera et al., DimeNet / Schütt et al., PaiNN).

    Evaluated via ``torch.sinc`` so ``d -> 0`` is analytically exact without division
    by zero:
        ``phi_n(d) = sqrt(2 / r_c) * (n * pi / r_c) * sinc(n * d / r_c)``
    where PyTorch's ``sinc(x) = sin(pi * x) / (pi * x)``.
    """

    num_rbf: int
    cutoff: float
    frequencies: Tensor

    def __init__(
        self,
        num_rbf: int = 24,
        cutoff: float = 5.5,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        freqs = torch.arange(1, num_rbf + 1, dtype=torch.float32)
        if trainable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)

    def forward(self, distances: Tensor) -> Tensor:
        """Expand pairwise distances ``[...]`` into ``[..., num_rbf]`` Bessel features."""
        d_scaled = distances.unsqueeze(-1) / self.cutoff
        freq = self.frequencies
        norm_factor = math.sqrt(2.0 / self.cutoff) * (math.pi / self.cutoff)
        return norm_factor * freq * torch.sinc(d_scaled * freq)


class GaussianRBF(nn.Module):
    """Uniformly spaced Gaussian radial basis expansion over ``[start, cutoff]``."""

    num_rbf: int
    cutoff: float
    start: float
    centers: Tensor
    coeff: Tensor

    def __init__(
        self,
        num_rbf: int = 24,
        cutoff: float = 5.5,
        start: float = 0.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if cutoff <= start:
            raise ValueError(f"cutoff ({cutoff}) must be greater than start ({start})")
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.start = float(start)

        centers = torch.linspace(start, cutoff, num_rbf, dtype=torch.float32)
        step = (cutoff - start) / max(num_rbf - 1, 1)
        coeff = torch.tensor(-0.5 / (step * step), dtype=torch.float32)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.coeff = nn.Parameter(coeff)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("coeff", coeff)

    def forward(self, distances: Tensor) -> Tensor:
        """Expand pairwise distances ``[...]`` into ``[..., num_rbf]`` Gaussian features."""
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(self.coeff * (diff * diff))


def center_masked_coords(coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Zero dummy coordinates and subtract each pocket's real-atom centroid.

    Centering prior to distance/edge computation eliminates catastrophic floating-point
    cancellation under large rigid translations (e.g. ``t in [-100, 100]^3``) and
    guarantees bit-exact invariance to dummy atom coordinates.
    """
    mask_bool = atom_mask.to(torch.bool)
    mask_3d = mask_bool.unsqueeze(-1)
    coords_f = coords.to(torch.float32)
    safe_coords = torch.where(mask_3d, coords_f, torch.zeros_like(coords_f))
    counts = mask_bool.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
    centroid = safe_coords.sum(dim=1, keepdim=True) / counts.unsqueeze(-1)
    return torch.where(mask_3d, safe_coords - centroid, torch.zeros_like(safe_coords))


def build_radius_knn_graph(
    coords: Tensor,
    atom_mask: Tensor,
    cutoff: float = 5.5,
    max_neighbors: int = 24,
) -> RadiusGraph:
    """Build a sparse batched 3D radius + top-k nearest-neighbor graph in pure PyTorch.

    Only edges between valid atoms (``atom_mask == True``) within the same batch item,
    excluding self-loops, with ``edge_dist < cutoff`` and at most ``max_neighbors``
    closest neighbors per target atom are emitted.

    Args:
        coords: Batched 3D atomic coordinates of shape ``[B, N, 3]``.
        atom_mask: Boolean validity mask of shape ``[B, N]`` (``True`` for real atoms).
        cutoff: Radial interaction cutoff in Angstroms.
        max_neighbors: Maximum incoming neighbors per atom.

    Returns:
        ``RadiusGraph(src, dst, edge_vec, edge_dist, batch_idx)``.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
    if atom_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"atom_mask shape {tuple(atom_mask.shape)} != coords[:2] {tuple(coords.shape[:2])}"
        )

    batch_size, num_atoms, _ = coords.shape
    device = coords.device
    if num_atoms <= 1 or batch_size == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_vec = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_dist = torch.empty(0, dtype=torch.float32, device=device)
        return RadiusGraph(
            src=empty_long,
            dst=empty_long,
            edge_vec=empty_vec,
            edge_dist=empty_dist,
            batch_idx=empty_long,
        )

    mask_bool = atom_mask.to(torch.bool)
    centered = center_masked_coords(coords, mask_bool)

    # Exact Euclidean distance matrix without GEMM x^2 + y^2 - 2xy cancellation.
    dist_matrix = torch.cdist(centered, centered, compute_mode="donot_use_mm_for_euclid_dist")
    eye = torch.eye(num_atoms, device=device, dtype=torch.bool).unsqueeze(0)
    valid_pair = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1) & (~eye) & (dist_matrix < cutoff)

    k = min(max_neighbors, num_atoms)
    dist_masked = dist_matrix.masked_fill(~valid_pair, float("inf"))
    knn_dist, knn_src_local = torch.topk(dist_masked, k=k, dim=-1, largest=False, sorted=True)
    edge_valid = torch.isfinite(knn_dist)

    batch_idx, dst_local, k_idx = torch.nonzero(edge_valid, as_tuple=True)
    src_local = knn_src_local[batch_idx, dst_local, k_idx]

    offset = batch_idx * num_atoms
    src = offset + src_local
    dst = offset + dst_local

    centered_flat = centered.reshape(batch_size * num_atoms, 3)
    edge_vec = centered_flat[dst] - centered_flat[src]
    edge_dist = torch.sqrt((edge_vec * edge_vec).sum(dim=-1).clamp(min=1e-12))

    return RadiusGraph(
        src=src,
        dst=dst,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        batch_idx=batch_idx,
    )


class EquivariantRMSNorm(nn.Module):
    """O(3)-equivariant RMS normalization for ``l=1`` vector channels ``[..., C_v, 3]``.

    Normalizes vector magnitudes by their rotation-invariant RMS norm across channels
    (or per channel when ``per_channel=True``) while strictly preserving 3D directions
    and keeping zeroed dummy atoms at exactly ``0.0``.
    """

    num_channels: int
    eps: float
    per_channel: bool

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.per_channel = bool(per_channel)
        self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32))

    def forward(self, vec: Tensor) -> Tensor:
        """Apply equivariant RMS normalization to vector tensor ``[..., C_v, 3]``."""
        sq_norm = (vec * vec).sum(dim=-1, keepdim=True)  # [..., C_v, 1]
        if self.per_channel:
            denom = torch.sqrt(sq_norm + self.eps)
        else:
            mean_sq = sq_norm.mean(dim=-2, keepdim=True)  # [..., 1, 1]
            denom = torch.sqrt(mean_sq + self.eps)
        scale = self.weight.view(*([1] * (vec.ndim - 2)), self.num_channels, 1)
        return (vec / denom) * scale


class MaskedAttentionPool(nn.Module):
    """Permutation-invariant masked attention + masked mean pooling over pocket atoms."""

    def __init__(self, in_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim if out_dim is not None else in_dim)
        self.attn_score = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim // 2),
            nn.SiLU(),
            nn.Linear(self.in_dim // 2, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(2 * self.in_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, node_emb: Tensor, atom_mask: Tensor) -> Tensor:
        """Pool ``node_emb[B, N, D]`` over real atoms ``atom_mask[B, N]`` -> ``[B, out_dim]``."""
        mask_bool = atom_mask.to(torch.bool)
        safe_nodes = torch.where(mask_bool.unsqueeze(-1), node_emb, 0.0)

        logits = self.attn_score(safe_nodes).squeeze(-1)  # [B, N]
        masked_logits = logits.masked_fill(~mask_bool, float("-inf"))
        all_masked = ~mask_bool.any(dim=-1, keepdim=True)
        safe_logits = masked_logits.masked_fill(all_masked, 0.0)
        weights = torch.softmax(safe_logits, dim=-1).masked_fill(~mask_bool, 0.0)

        attn_pool = torch.bmm(weights.unsqueeze(1), safe_nodes).squeeze(1)
        counts = mask_bool.sum(dim=1, keepdim=True).clamp(min=1).to(node_emb.dtype)
        mean_pool = safe_nodes.sum(dim=1) / counts

        pooled: Tensor = self.out_proj(torch.cat([attn_pool, mean_pool], dim=-1))
        has_any = mask_bool.any(dim=1, keepdim=True).to(pooled.dtype)
        return pooled * has_any


__all__ = [
    "BesselRBF",
    "CosineCutoff",
    "EquivariantRMSNorm",
    "GaussianRBF",
    "MaskedAttentionPool",
    "PocketEncoderOutput",
    "RadiusGraph",
    "build_radius_knn_graph",
    "center_masked_coords",
]
