"""MPS/CUDA/CPU-native tie-safe dense batched 3D k-NN graph construction and masked softmax.

Eliminates sparse ``index_add_`` atomic scatter contention on Apple Silicon MPS by operating
strictly on contiguous ``[B, N, K, ...]`` dense neighbor tensors in ``torch.float32`` (zero
``float64`` buffers, zero external C++ extensions).

Key mathematical guarantees (see ``docs/pointcloud_cv_3d_atom_encoding_plan.md`` §4.0):
  1. **Equidistant K-th neighbor tie-swap invariance:** When ``adaptive_cutoff=True`` and an
     atom has more than ``K`` valid neighbors inside ``cutoff``, its cosine envelope uses the
     per-atom adaptive cutoff :math:`r_{\\mathrm{eff},i} = \\min(r_c, d_{i,K+1})`. Any neighbor
     sitting on the K-th distance tie shell therefore receives smooth envelope weight
     :math:`f_c(d_{i,K+1}; r_{\\mathrm{eff},i}) = 0` with zero first derivative, yielding exact
     ``0.0`` discontinuity when floating-point rounding permutes tied neighbors under rigid
     :math:`SE(3)` transforms.
  2. **Zero-distance gradient safety:** Pairwise neighbor selection runs outside autograd, while
     differentiable distances and unit directions use smooth guards
     :math:`d_{ij} = \\sqrt{\\|\\mathbf{x}_i - \\mathbf{x}_j\\|_2^2 + 10^{-8}}` and
     :math:`\\hat{\\mathbf{r}}_{ij} = (\\mathbf{x}_i - \\mathbf{x}_j) /`
     :math:`\\sqrt{\\|\\mathbf{x}_i - \\mathbf{x}_j\\|_2^2 + 10^{-6}}`,
     guaranteeing finite gradients even under exact coordinate clashes (:math:`d_{ij} = 0`).
  3. **Edge-case completeness:** Cleanly handles single-atom ions (``N = 1``), small ligands
     (``N <= K``), isolated atoms with zero neighbors inside cutoff (``all_empty``), empty
     batches, and corrupted dummy-atom coordinates/features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DenseKNNGraph:
    """Dense batched 3D neighbor graph over valid (non-padded) atoms.

    Edge vector convention matches :func:`~geneuclidean.models.encoders.rbf.build_radius_knn_graph`:
    ``edge_vec[b, i, k] = centered_coords[b, i] - centered_coords[b, nbr_idx[b, i, k]]``
    (target minus source / ``dst - src``).

    Attributes:
        nbr_idx: Neighbor atom indices in ``[0, N)`` of shape ``[B, N, K]`` (``torch.long``).
            Invalid slots are set to ``0`` so ``gather_neighbors`` is always in-bounds.
        nbr_mask: Boolean edge validity mask of shape ``[B, N, K]`` (``True`` iff both endpoint
            atoms are valid, distinct, and strictly within ``cutoff``).
        edge_vec: Relative displacement vectors ``x_i - x_j`` of shape ``[B, N, K, 3]``
            (``float32``, strictly zeroed on ``~nbr_mask``).
        edge_dist: Smooth Euclidean distances ``sqrt(||x_i - x_j||^2 + 1e-8)`` of shape
            ``[B, N, K]`` (``float32``, strictly zeroed on ``~nbr_mask``).
        edge_dir: Smooth unit edge directions ``(x_i - x_j) / sqrt(||x_i - x_j||^2 + 1e-6)``
            of shape ``[B, N, K, 3]`` (``float32``, strictly zeroed on ``~nbr_mask``).
        edge_env: Smooth Behler-Parrinello cosine cutoff weights evaluated at per-atom
            ``r_eff`` of shape ``[B, N, K]`` in ``[0, 1]`` (zeroed on ``~nbr_mask``).
        r_eff: Per-atom effective cutoff radius :math:`\\min(r_c, d_{i,K+1})` of shape
            ``[B, N, 1]`` (``float32``, clamped to ``>= 1e-4``).
        centered_coords: Centroid-subtracted valid coordinates of shape ``[B, N, 3]``
            (``float32``, strictly zeroed on dummy atoms).
        centroid_offset: Original world-space valid-atom centroid of shape ``[B, 3]``
            (``float32``), preserved for shared-frame pocket-ligand scoring.
    """

    nbr_idx: Tensor
    nbr_mask: Tensor
    edge_vec: Tensor
    edge_dist: Tensor
    edge_dir: Tensor
    edge_env: Tensor
    r_eff: Tensor
    centered_coords: Tensor
    centroid_offset: Tensor


def center_masked_coords_with_offset(
    coords: Tensor,
    atom_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Zero dummy coordinates, subtract valid-atom centroids, and return both tensors.

    Subtracting the valid-atom centroid prior to pairwise geometry eliminates catastrophic
    ``float32`` cancellation under large rigid translations (e.g. ``t in [-100, 100]^3``),
    while returning ``centroid_offset`` preserves exact inter-molecular relative frames for
    joint pocket-ligand scoring.

    Args:
        coords: World-space atomic coordinates of shape ``[B, N, 3]``.
        atom_mask: Boolean validity mask of shape ``[B, N]`` (``True`` for real atoms).

    Returns:
        Tuple ``(centered_coords [B, N, 3], centroid_offset [B, 3])`` in ``torch.float32``.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
    if atom_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"atom_mask shape {tuple(atom_mask.shape)} != coords[:2] {tuple(coords.shape[:2])}"
        )

    mask_bool = atom_mask.to(torch.bool)
    mask_3d = mask_bool.unsqueeze(-1)
    coords_f = coords.to(torch.float32)
    zeros = torch.zeros_like(coords_f)
    safe_coords = torch.where(mask_3d, coords_f, zeros)
    counts = mask_bool.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
    centroid_offset = safe_coords.sum(dim=1) / counts  # [B, 3]
    centered = torch.where(mask_3d, safe_coords - centroid_offset.unsqueeze(1), zeros)
    return centered, centroid_offset


def gather_neighbors(x: Tensor, nbr_idx: Tensor) -> Tensor:
    """Gather neighbor features ``x[b, nbr_idx[b, i, k]]`` via row-vectorized ``index_select``.

    Flattens ``x`` to ``[B * N, F]`` and issues a single contiguous row-vectorized
    :func:`torch.index_select` rather than elementwise ``torch.gather``, avoiding Apple Silicon
    MPS scalar-strided gather bottlenecks (~20x speedup on MPS) while preserving exact
    mathematical equivalence and autograd support on CPU, CUDA, and MPS.

    Args:
        x: Per-atom tensor of shape ``[B, N, *feat_dims]`` (including ``[B, N]``).
        nbr_idx: Neighbor index tensor of shape ``[B, N, K]`` (``torch.long``).

    Returns:
        Gathered neighbor tensor of shape ``[B, N, K, *feat_dims]``.
    """
    if x.ndim < 2:
        raise ValueError(f"Expected x with at least 2 dims [B, N, ...], got {tuple(x.shape)}")
    if nbr_idx.ndim != 3 or nbr_idx.shape[:2] != x.shape[:2]:
        raise ValueError(
            f"nbr_idx shape {tuple(nbr_idx.shape)} incompatible with x shape {tuple(x.shape)}"
        )

    batch_size, num_atoms, k = nbr_idx.shape
    feat_shape = x.shape[2:]
    if batch_size == 0 or num_atoms == 0 or k == 0:
        return x.new_empty((batch_size, num_atoms, k, *feat_shape))

    flat_feat = x.reshape(batch_size * num_atoms, -1)
    if flat_feat.shape[-1] == 0:
        return x.new_empty((batch_size, num_atoms, k, *feat_shape))

    idx_long = nbr_idx if nbr_idx.dtype == torch.long else nbr_idx.to(dtype=torch.long)
    if batch_size == 1:
        flat_idx = idx_long.reshape(-1)
    else:
        offsets = torch.arange(batch_size, device=idx_long.device, dtype=torch.long) * num_atoms
        flat_idx = (idx_long + offsets.view(batch_size, 1, 1)).reshape(-1)
    gathered = torch.index_select(flat_feat, dim=0, index=flat_idx)
    return gathered.reshape(batch_size, num_atoms, k, *feat_shape)


def safe_masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Numerically safe masked softmax with explicit ``all_empty`` forward & backward guard.

    Standard ``softmax(logits.masked_fill(~mask, -inf))`` produces ``NaN`` whenever an entire
    reduction slice is masked out (e.g. an isolated atom with zero neighbors inside cutoff,
    or a dummy atom row). Filling all-empty slices with ``0.0`` before ``softmax`` and zeroing
    masked entries via ``torch.where`` after ``softmax`` guarantees finite outputs and finite
    gradients on CPU, CUDA, and Apple Silicon MPS.

    Args:
        logits: Unnormalized attention logits of arbitrary shape ``[...]``.
        mask: Boolean mask broadcastable to ``logits.shape`` (``True`` for valid entries).
        dim: Dimension along which to normalize.

    Returns:
        Normalized attention weights of shape ``logits.shape`` summing to ``1.0`` on valid
        non-empty slices and strictly ``0.0`` on masked entries and all-empty slices.
    """
    mask_b = mask.to(torch.bool).expand_as(logits)
    all_empty = ~mask_b.any(dim=dim, keepdim=True)
    neg_inf = torch.full_like(logits, float("-inf"))
    masked_logits = torch.where(mask_b, logits, neg_inf)
    safe_logits = torch.where(all_empty, torch.zeros_like(logits), masked_logits)
    probs = torch.softmax(safe_logits, dim=dim)
    return torch.where(mask_b, probs, torch.zeros_like(probs))


def build_dense_knn_graph(
    coords: Tensor,
    atom_mask: Tensor,
    cutoff: float = 5.5,
    max_neighbors: int = 24,
    *,
    adaptive_cutoff: bool = True,
) -> DenseKNNGraph:
    """Build a tie-safe, smooth, dense batched 3D radius + k-NN graph in pure PyTorch.

    Computes up to ``K = min(max_neighbors, max(1, N - 1))`` nearest distinct valid neighbors
    strictly within ``cutoff`` for each valid atom. When ``adaptive_cutoff=True`` and an atom
    has a valid ``(K + 1)``-th neighbor within ``cutoff``, its smooth cosine envelope uses
    :math:`r_{\\mathrm{eff},i} = \\min(r_c, d_{i,K+1})` so equidistant K-th neighbor tie-swaps
    induce exact ``0.0`` discontinuity under rigid :math:`SE(3)` transformations.

    Args:
        coords: Batched 3D atomic coordinates of shape ``[B, N, 3]``.
        atom_mask: Boolean validity mask of shape ``[B, N]`` (``True`` for real atoms).
        cutoff: Hard radial interaction cutoff in Angstroms (``> 0``).
        max_neighbors: Maximum incoming neighbors ``K`` per atom (``>= 1``).
        adaptive_cutoff: If ``True`` (default), adapt per-atom cosine envelope radius to
            ``min(cutoff, d_{i, K+1})`` for exact equidistant tie-swap invariance.

    Returns:
        :class:`DenseKNNGraph` with contiguous ``[B, N, K, ...]`` ``float32`` tensors.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
    if atom_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"atom_mask shape {tuple(atom_mask.shape)} != coords[:2] {tuple(coords.shape[:2])}"
        )
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if max_neighbors < 1:
        raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")

    batch_size, num_atoms, _ = coords.shape
    device = coords.device
    cutoff_f = float(cutoff)
    mask_bool = atom_mask.to(torch.bool)
    centered, centroid_offset = center_masked_coords_with_offset(coords, mask_bool)

    if batch_size == 0 or num_atoms == 0:
        return DenseKNNGraph(
            nbr_idx=torch.empty((batch_size, num_atoms, 0), dtype=torch.long, device=device),
            nbr_mask=torch.empty((batch_size, num_atoms, 0), dtype=torch.bool, device=device),
            edge_vec=torch.empty((batch_size, num_atoms, 0, 3), dtype=torch.float32, device=device),
            edge_dist=torch.empty((batch_size, num_atoms, 0), dtype=torch.float32, device=device),
            edge_dir=torch.empty((batch_size, num_atoms, 0, 3), dtype=torch.float32, device=device),
            edge_env=torch.empty((batch_size, num_atoms, 0), dtype=torch.float32, device=device),
            r_eff=torch.empty((batch_size, num_atoms, 1), dtype=torch.float32, device=device),
            centered_coords=centered,
            centroid_offset=centroid_offset,
        )

    if num_atoms == 1:
        # Single-atom ion guard (e.g., isolated Zn2+): K=1 with all-False mask.
        return DenseKNNGraph(
            nbr_idx=torch.zeros((batch_size, 1, 1), dtype=torch.long, device=device),
            nbr_mask=torch.zeros((batch_size, 1, 1), dtype=torch.bool, device=device),
            edge_vec=torch.zeros((batch_size, 1, 1, 3), dtype=torch.float32, device=device),
            edge_dist=torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device),
            edge_dir=torch.zeros((batch_size, 1, 1, 3), dtype=torch.float32, device=device),
            edge_env=torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device),
            r_eff=torch.full((batch_size, 1, 1), cutoff_f, dtype=torch.float32, device=device),
            centered_coords=centered,
            centroid_offset=centroid_offset,
        )

    k_eff = min(int(max_neighbors), num_atoms - 1)
    k_query = min(k_eff + 1, num_atoms)

    # Neighbor selection runs strictly outside autograd so torch.cdist never injects a
    # 0/0 backward singularity on exact coordinate clashes (d = 0).
    with torch.no_grad():
        dist_matrix = torch.cdist(
            centered.detach(),
            centered.detach(),
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        eye = torch.eye(num_atoms, device=device, dtype=torch.bool).unsqueeze(0)
        valid_pair = (
            mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1) & (~eye) & (dist_matrix < cutoff_f)
        )
        dist_masked = dist_matrix.masked_fill(~valid_pair, float("inf"))
        knn_dist_all, knn_idx_all = torch.topk(
            dist_masked, k=k_query, dim=-1, largest=False, sorted=True
        )
        edge_valid_all = torch.isfinite(knn_dist_all)
        safe_idx_all = torch.where(edge_valid_all, knn_idx_all, torch.zeros_like(knn_idx_all))

    # Recompute exact differentiable displacements and smooth distances from gathered coords.
    nbr_coords_all = gather_neighbors(centered, safe_idx_all)  # [B, N, k_query, 3]
    edge_vec_all = centered.unsqueeze(2) - nbr_coords_all
    sq_d_all = (edge_vec_all * edge_vec_all).sum(dim=-1)
    smooth_dist_all = torch.sqrt(sq_d_all + 1e-8)

    nbr_idx = safe_idx_all[..., :k_eff]
    nbr_mask = edge_valid_all[..., :k_eff]
    edge_vec_k = edge_vec_all[..., :k_eff, :]
    sq_d_k = sq_d_all[..., :k_eff]
    smooth_dist_k = smooth_dist_all[..., :k_eff]

    if adaptive_cutoff and k_query > k_eff:
        has_kplus1 = edge_valid_all[..., k_eff]
        cutoff_tensor = torch.full_like(smooth_dist_all[..., k_eff], cutoff_f)
        d_kplus1 = torch.where(has_kplus1, smooth_dist_all[..., k_eff], cutoff_tensor)
        r_eff = torch.minimum(d_kplus1, cutoff_tensor).clamp(min=1e-4).unsqueeze(-1)
    else:
        r_eff = torch.full((batch_size, num_atoms, 1), cutoff_f, dtype=torch.float32, device=device)

    mask_3d = nbr_mask.unsqueeze(-1)
    zero_vec = torch.zeros_like(edge_vec_k)
    zero_scalar = torch.zeros_like(smooth_dist_k)

    edge_vec = torch.where(mask_3d, edge_vec_k, zero_vec)
    edge_dist = torch.where(nbr_mask, smooth_dist_k, zero_scalar)
    inv_smooth_d = torch.rsqrt(sq_d_k + 1e-6).unsqueeze(-1)
    edge_dir = torch.where(mask_3d, edge_vec_k * inv_smooth_d, zero_vec)

    # Smooth Behler-Parrinello cosine envelope evaluated against per-atom r_eff.
    ratio = (smooth_dist_k / r_eff).clamp(min=0.0, max=1.0)
    raw_env = 0.5 * (torch.cos(math.pi * ratio) + 1.0)
    edge_env = torch.where(nbr_mask, raw_env, zero_scalar)

    return DenseKNNGraph(
        nbr_idx=nbr_idx,
        nbr_mask=nbr_mask,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        edge_dir=edge_dir,
        edge_env=edge_env,
        r_eff=r_eff,
        centered_coords=centered,
        centroid_offset=centroid_offset,
    )


__all__ = [
    "DenseKNNGraph",
    "build_dense_knn_graph",
    "center_masked_coords_with_offset",
    "gather_neighbors",
    "safe_masked_softmax",
]
