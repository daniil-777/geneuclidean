"""Anti-Shortcut 3D Sub-Pocket JEPA Self-Supervised Pretraining (Sonata + Point-JEPA + Concerto).

Implements Method 3 Part B from ``docs/pointcloud_cv_3d_atom_encoding_plan.md`` (§4.3 & Prompt 3):
  1. **Contiguous 3D Spherical Sub-Pocket Block Masking** (:func:`sample_spherical_block_mask`):
     masks ``4.5-6.0 A`` contiguous 3D cavities while guaranteeing disjoint visible/target sets
     with ``>= 4`` visible and ``>= 1`` target atoms per valid pocket.
  2. **Stochastic Sonata Spatial Obscuration** (:func:`apply_sonata_spatial_obscuration`):
     injects coordinate jitter (``0.20 A``) and stochastically zeroes radial shortcut features
     ``node_feats[..., 47:52]`` with probability ``p = 0.80`` on visible student views, killing
     the covalent/radial geometric shortcut while keeping Layer-0 radial weights trained (``20%``).
  3. **Strict Student-Frame Latent Prediction** (:class:`SE3LatentPredictor`):
     expresses masked target coordinates ``tgt_coords`` strictly in the **student's** centered
     visible frame (zero coordinate/frame leak from the independently rotated EMA teacher, and
     zero target chemical features leaked!).
  4. **Anti-Collapse Regularization & Diagnostics**:
     :class:`VICRegLoss` and :func:`compute_effective_rank` (Roy & Vetterli / RankMe) computed
     strictly on unpadded, visible representations ``H_vis = node_emb[visible_mask]``.
  5. **Optional Concerto Cross-Modal InfoNCE** (:class:`ConcertoCrossModalLoss`):
     aligns pooled 3D pocket embeddings with paired 2D/3D ligand embeddings via GenEuclidean's
     symmetric :func:`~geneuclidean.training.losses.compute_drugclip_loss`.

All operations strictly use ``torch.float32`` and avoid unsupported Apple MPS linear-algebra
fallbacks (``PYTORCH_ENABLE_MPS_FALLBACK=0`` clean).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.models.decoders.transformer import RMSNorm, SwiGLU
from geneuclidean.models.encoders.dense_knn import (
    center_masked_coords_with_offset,
    safe_masked_softmax,
)
from geneuclidean.models.encoders.rbf import BesselRBF, CosineCutoff, PocketEncoderOutput
from geneuclidean.models.encoders.subpocket_vn import (
    SubPocketEncodingOutput,
    SubPocketVNJEPAEncoder,
)
from geneuclidean.models.encoders.tensor_ops import (
    VectorNeuronLinear,
    chiral_cross_product,
)
from geneuclidean.training.losses import compute_drugclip_loss

_RADIAL_SHORTCUT_SLICE: slice = slice(47, 52)


class VICRegOutput(NamedTuple):
    """Breakdown of VICReg variance-covariance anti-collapse regularization."""

    total: Tensor
    var_loss: Tensor
    cov_loss: Tensor


class JEPASSLOutput(NamedTuple):
    """Structured forward output from :meth:`SubPocketJEPAPretrainer.forward_ssl`."""

    total_loss: Tensor
    jepa_loss: Tensor
    smooth_l1_loss: Tensor
    cosine_loss: Tensor
    vicreg_loss: Tensor
    var_loss: Tensor
    cov_loss: Tensor
    cross_modal_loss: Tensor
    effective_rank: Tensor
    visible_mask: Tensor
    target_mask: Tensor
    radial_obscured_frac: Tensor


def sample_random_so3_rotations(
    batch_size: int,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``batch_size`` independent Haar-uniform proper rotations ``R in SO(3)`` (``det=+1``).

    Evaluated on CPU for tiny ``[B, 3, 3]`` QR factorizations and transferred to ``device`` so
    Apple Silicon MPS never triggers ``PYTORCH_ENABLE_MPS_FALLBACK=0`` linalg errors.
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")
    dev = torch.device(device)
    if batch_size == 0:
        return torch.empty((0, 3, 3), device=dev, dtype=torch.float32)

    mats = torch.randn((batch_size, 3, 3), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(mats)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    diag_sign = torch.where(diag < 0.0, -torch.ones_like(diag), torch.ones_like(diag))
    q = q * diag_sign.unsqueeze(-2)
    dets = torch.linalg.det(q)
    flip = (dets < 0.0).to(torch.float32).view(batch_size, 1)
    q_col0 = q[:, :, 0] * (1.0 - 2.0 * flip)
    q = torch.cat((q_col0.unsqueeze(-1), q[:, :, 1:]), dim=-1)
    return q.to(device=dev, dtype=torch.float32)


def sample_spherical_block_mask(
    coords: Tensor,
    atom_mask: Tensor,
    *,
    num_blocks: int = 3,
    min_radius: float = 4.5,
    max_radius: float = 6.0,
    max_mask_ratio: float = 0.60,
    min_visible: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample disjoint 3D contiguous spherical block masks ``(visible_mask, target_mask)``.

    Masks ``num_blocks`` spherical sub-cavities of radius ``U(min_radius, max_radius)`` per pocket
    while guaranteeing:
      - ``visible_mask & target_mask`` is strictly ``False`` everywhere,
      - ``visible_mask | target_mask == atom_mask`` for every valid atom,
      - zero dummy atoms appear in either mask, and
      - every pocket with ``>= 2`` valid atoms retains ``>= min(min_visible, n_valid - 1)`` visible
        context atoms and ``>= 1`` masked target atom.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
    if atom_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"atom_mask shape {tuple(atom_mask.shape)} != coords[:2] {tuple(coords.shape[:2])}"
        )
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if min_radius <= 0.0 or max_radius < min_radius:
        raise ValueError(f"Invalid radius range [{min_radius}, {max_radius}]")

    batch_size, num_atoms, _ = coords.shape
    device = coords.device
    mask_cpu = atom_mask.detach().to(device="cpu", dtype=torch.bool)
    coords_cpu = coords.detach().to(device="cpu", dtype=torch.float32)

    visible_cpu = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
    target_cpu = torch.zeros((batch_size, num_atoms), dtype=torch.bool)

    for b in range(batch_size):
        valid_idx = torch.nonzero(mask_cpu[b], as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if n_valid == 0:
            continue
        if n_valid == 1:
            visible_cpu[b, valid_idx[0]] = True
            continue

        eff_min_vis = min(int(min_visible), n_valid - 1)
        max_targets = max(1, min(n_valid - eff_min_vis, int(math.floor(n_valid * max_mask_ratio))))
        valid_coords = coords_cpu[b, valid_idx]  # [n_valid, 3]

        n_anchors = min(int(num_blocks), n_valid)
        perm = torch.randperm(n_valid, generator=generator)
        anchor_local = perm[:n_anchors]
        anchors = valid_coords[anchor_local]  # [n_anchors, 3]

        # Distances from each valid atom to each sampled anchor.
        diff = valid_coords.unsqueeze(1) - anchors.unsqueeze(0)  # [n_valid, n_anchors, 3]
        dists = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        radii = min_radius + (max_radius - min_radius) * torch.rand(
            (1, n_anchors), generator=generator, dtype=torch.float32
        )
        in_sphere = (dists <= radii).any(dim=1)  # [n_valid]
        min_anchor_dist = dists.min(dim=1).values

        n_selected = int(in_sphere.sum().item())
        if n_selected == 0:
            # Ensure at least 1 contiguous target atom (closest to the first anchor).
            nearest = int(torch.argmin(min_anchor_dist).item())
            in_sphere[nearest] = True
        elif n_selected > max_targets:
            # Retain the closest atoms to the block centers up to max_targets.
            masked_positions = torch.nonzero(in_sphere, as_tuple=False).squeeze(-1)
            order = torch.argsort(min_anchor_dist[masked_positions])
            keep_targets = masked_positions[order[:max_targets]]
            in_sphere = torch.zeros(n_valid, dtype=torch.bool)
            in_sphere[keep_targets] = True

        tgt_global = valid_idx[in_sphere]
        vis_global = valid_idx[~in_sphere]
        target_cpu[b, tgt_global] = True
        visible_cpu[b, vis_global] = True

    return visible_cpu.to(device=device), target_cpu.to(device=device)


def apply_sonata_spatial_obscuration(
    coords: Tensor,
    node_feats: Tensor,
    visible_mask: Tensor,
    *,
    coord_jitter_std: float = 0.20,
    radial_drop_prob: float = 0.80,
    radial_slice: slice = _RADIAL_SHORTCUT_SLICE,
    feat_drop_prob: float = 0.20,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply Sonata anti-shortcut coordinate jitter and stochastic radial obscuration.

    Operates strictly on visible student inputs (non-mutating):
      1. Adds ``N(0, coord_jitter_std^2)`` Gaussian coordinate jitter on visible atoms.
      2. With probability ``radial_drop_prob`` (default ``0.80`` per pocket), zeroes radial
         shortcut descriptors ``node_feats[..., 47:52]`` so the student cannot cheat via
         centroid-distance shortcuts while keeping Layer-0 radial weights trained on the
         remaining ``20%`` of views.
      3. Randomly zeroes individual chemical feature channels with probability ``feat_drop_prob``.
      4. Strictly zeroes all non-visible / dummy atom features to prevent target feature leakage.

    Returns:
        Tuple ``(obscured_coords [B, N, 3], obscured_feats [B, N, F], radial_obscured_frac [])``.
    """
    if not (0.0 <= radial_drop_prob <= 1.0):
        raise ValueError(f"radial_drop_prob must lie in [0, 1], got {radial_drop_prob}")
    if not (0.0 <= feat_drop_prob < 1.0):
        raise ValueError(f"feat_drop_prob must lie in [0, 1), got {feat_drop_prob}")
    if coord_jitter_std < 0.0:
        raise ValueError(f"coord_jitter_std must be >= 0, got {coord_jitter_std}")

    batch_size, num_atoms, feat_dim = node_feats.shape
    device = coords.device
    vis_bool = visible_mask.to(device=device, dtype=torch.bool)
    vis_3d = vis_bool.unsqueeze(-1)

    obs_coords = coords.to(torch.float32).clone()
    obs_feats = node_feats.to(torch.float32).clone()

    if coord_jitter_std > 0.0 and batch_size > 0 and num_atoms > 0:
        noise = torch.randn(obs_coords.shape, generator=generator, dtype=torch.float32).to(
            device=device
        ) * float(coord_jitter_std)
        obs_coords = torch.where(vis_3d, obs_coords + noise, obs_coords)

    # Zero all non-visible / dummy features upfront so masked target features can never leak.
    obs_feats = torch.where(vis_3d, obs_feats, torch.zeros_like(obs_feats))

    if feat_drop_prob > 0.0 and batch_size > 0 and num_atoms > 0:
        keep_prob = 1.0 - float(feat_drop_prob)
        keep_mask = (
            torch.rand(obs_feats.shape, generator=generator, dtype=torch.float32).to(device=device)
            < keep_prob
        )
        obs_feats = torch.where(keep_mask, obs_feats, torch.zeros_like(obs_feats))

    # Stochastic per-pocket radial shortcut obscuration (cols 47:52).
    radial_start = 0 if radial_slice.start is None else int(radial_slice.start)
    radial_stop = feat_dim if radial_slice.stop is None else min(feat_dim, int(radial_slice.stop))
    if batch_size > 0 and radial_start < radial_stop:
        if radial_drop_prob >= 1.0:
            drop_pocket = torch.ones((batch_size, 1, 1), device=device, dtype=torch.bool)
        elif radial_drop_prob <= 0.0:
            drop_pocket = torch.zeros((batch_size, 1, 1), device=device, dtype=torch.bool)
        else:
            u = torch.rand((batch_size, 1, 1), generator=generator, dtype=torch.float32).to(
                device=device
            )
            drop_pocket = u < float(radial_drop_prob)

        radial_region = obs_feats[..., radial_start:radial_stop]
        obs_feats[..., radial_start:radial_stop] = torch.where(
            drop_pocket,
            torch.zeros_like(radial_region),
            radial_region,
        )
        vis_counts = vis_bool.sum(dim=1).to(torch.float32)
        total_vis = vis_counts.sum().clamp(min=1.0)
        obscured_vis = (vis_counts * drop_pocket.view(batch_size).to(torch.float32)).sum()
        radial_obscured_frac = (obscured_vis / total_vis).detach()
    else:
        radial_obscured_frac = torch.tensor(0.0, device=device, dtype=torch.float32)

    return obs_coords, obs_feats, radial_obscured_frac


@torch.no_grad()
def compute_effective_rank(embeddings: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Compute Roy & Vetterli (2007) / RankMe effective rank of ``[N_valid, D]`` embeddings.

    Evaluates the symmetric Gram/covariance matrix on ``embeddings.device`` and solves its
    small eigenspectrum on CPU in ``float32`` to avoid unsupported MPS ``svd`` fallbacks.
    Returns a detached scalar ``float32`` tensor in ``[1.0, min(N_valid, D)]``.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D tensor [N_valid, D], got {tuple(embeddings.shape)}")
    n_samples, dim = embeddings.shape
    device = embeddings.device
    if n_samples < 2 or dim < 2:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    x = embeddings.detach().to(torch.float32)
    x_c = x - x.mean(dim=0, keepdim=True)
    if dim <= n_samples:
        gram = (x_c.transpose(0, 1) @ x_c) / float(n_samples)
    else:
        gram = (x_c @ x_c.transpose(0, 1)) / float(n_samples)

    gram_cpu = 0.5 * (gram + gram.transpose(0, 1)).to(device="cpu", dtype=torch.float32)
    eigvals = torch.linalg.eigvalsh(gram_cpu).clamp(min=0.0)
    singular_vals = torch.sqrt(eigvals)
    total = singular_vals.sum()
    if float(total.item()) <= eps:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    probs = singular_vals / (total + eps)
    nonzero = probs > eps
    entropy = -(probs[nonzero] * torch.log(probs[nonzero])).sum()
    return torch.exp(entropy).to(device=device, dtype=torch.float32)


class VICRegLoss(nn.Module):
    """VICReg variance + off-diagonal covariance regularizer on unpadded visible embeddings.

    Expects pre-sliced visible representations ``H_vis [N_vis, D]`` (strictly excluding padded
    dummy atoms and masked zero slots so representation collapse cannot hide behind zeros).
    """

    gamma: float
    var_weight: float
    cov_weight: float
    eps: float

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.gamma = float(gamma)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.eps = float(eps)

    def forward(self, h_vis: Tensor) -> VICRegOutput:
        """Compute ``(total, var_loss, cov_loss)`` on ``h_vis [N_vis, D]``."""
        if h_vis.ndim != 2:
            raise ValueError(f"Expected 2D visible tensor [N_vis, D], got {tuple(h_vis.shape)}")
        n_vis, dim = h_vis.shape
        if n_vis < 2 or dim < 2:
            zero = h_vis.sum() * 0.0
            return VICRegOutput(total=zero, var_loss=zero, cov_loss=zero)

        h_f = h_vis.to(torch.float32)
        h_centered = h_f - h_f.mean(dim=0, keepdim=True)
        var = (h_centered * h_centered).mean(dim=0)
        std = torch.sqrt(var + self.eps)
        var_loss = F.relu(self.gamma - std).mean()

        cov = (h_centered.transpose(0, 1) @ h_centered) / float(n_vis - 1)
        eye = torch.eye(dim, device=h_f.device, dtype=torch.bool)
        cov_off = cov.masked_fill(eye, 0.0)
        cov_loss = (cov_off * cov_off).sum() / float(dim)

        total = self.var_weight * var_loss + self.cov_weight * cov_loss
        return VICRegOutput(total=total, var_loss=var_loss, cov_loss=cov_loss)


class ConcertoCrossModalLoss(nn.Module):
    """Concerto-style symmetric InfoNCE aligning pooled 3D pockets with 2D/3D ligand embeddings.

    Projects pocket embeddings ``[B, pocket_dim]`` and paired ligand embeddings ``[B, ligand_dim]``
    into a shared normalized contrastive space and evaluates GenEuclidean's canonical
    :func:`~geneuclidean.training.losses.compute_drugclip_loss`.
    """

    pocket_dim: int
    ligand_dim: int
    proj_dim: int

    def __init__(
        self,
        pocket_dim: int,
        ligand_dim: int,
        *,
        proj_dim: int = 128,
        init_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if pocket_dim < 1 or ligand_dim < 1 or proj_dim < 4:
            raise ValueError(
                f"Invalid dims pocket={pocket_dim}, ligand={ligand_dim}, proj={proj_dim}"
            )
        if init_temperature <= 0.0:
            raise ValueError(f"init_temperature must be positive, got {init_temperature}")

        self.pocket_dim = int(pocket_dim)
        self.ligand_dim = int(ligand_dim)
        self.proj_dim = int(proj_dim)

        self.pocket_proj = nn.Sequential(
            RMSNorm(self.pocket_dim),
            nn.Linear(self.pocket_dim, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, self.proj_dim, bias=False),
        )
        self.ligand_proj = nn.Sequential(
            RMSNorm(self.ligand_dim),
            nn.Linear(self.ligand_dim, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, self.proj_dim, bias=False),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(init_temperature)), dtype=torch.float32)
        )

    def forward(
        self,
        pocket_emb: Tensor,
        ligand_emb: Tensor,
        *,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute symmetric InfoNCE loss between matched pocket and ligand embeddings."""
        if pocket_emb.ndim != 2 or ligand_emb.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got pocket={tuple(pocket_emb.shape)}, "
                f"ligand={tuple(ligand_emb.shape)}"
            )
        if pocket_emb.shape[0] != ligand_emb.shape[0]:
            raise ValueError(
                f"Batch mismatch: pocket={pocket_emb.shape[0]} vs ligand={ligand_emb.shape[0]}"
            )

        p_in = pocket_emb.to(torch.float32)
        l_in = ligand_emb.to(torch.float32)
        if pair_mask is not None:
            valid = pair_mask.to(device=p_in.device, dtype=torch.bool)
            p_in = p_in[valid]
            l_in = l_in[valid]

        z_p: Tensor = self.pocket_proj(p_in)
        z_l: Tensor = self.ligand_proj(l_in)
        return compute_drugclip_loss(z_p, z_l, self.logit_scale)


class _SE3CrossAttnPredictorLayer(nn.Module):
    """Single SE(3)-invariant target-to-visible cross-attention + SwiGLU predictor layer."""

    emb_dim: int
    num_heads: int
    head_dim: int

    def __init__(
        self,
        emb_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_heads = max(1, min(int(num_heads), self.emb_dim))
        while self.emb_dim % self.num_heads != 0:
            self.num_heads -= 1
        self.head_dim = self.emb_dim // self.num_heads

        self.norm_q = RMSNorm(self.emb_dim, eps=eps)
        self.norm_kv = RMSNorm(self.emb_dim, eps=eps)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.o_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.norm_ff = RMSNorm(self.emb_dim, eps=eps)
        self.ffn = SwiGLU(d_model=self.emb_dim, d_ff=2 * self.emb_dim, dropout=dropout)

    def forward(
        self,
        q_tokens: Tensor,
        ctx_tokens: Tensor,
        ctx_mask: Tensor,
        target_mask: Tensor,
        geo_bias: Tensor,
    ) -> Tensor:
        """Attend from target queries ``[B, N, D]`` to visible context ``[B, C, D]``."""
        batch_size, num_tgt, _ = q_tokens.shape
        num_ctx = ctx_tokens.shape[1]
        tgt_bool = target_mask.to(torch.bool)
        ctx_bool = ctx_mask.to(torch.bool)
        tgt_3d = tgt_bool.unsqueeze(-1)
        zeros_q = torch.zeros_like(q_tokens)

        q_in = torch.where(tgt_3d, self.norm_q(q_tokens), zeros_q)
        kv_in = torch.where(
            ctx_bool.unsqueeze(-1), self.norm_kv(ctx_tokens), torch.zeros_like(ctx_tokens)
        )

        q = (
            self.q_proj(q_in)
            .view(batch_size, num_tgt, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(kv_in)
            .view(batch_size, num_ctx, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(kv_in)
            .view(batch_size, num_ctx, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale + geo_bias  # [B, H, N, C]
        key_mask = ctx_bool[:, None, None, :]
        probs = safe_masked_softmax(scores, key_mask, dim=-1)
        probs = self.attn_drop(probs)

        ctx = (
            torch.matmul(probs, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tgt, self.emb_dim)
        )
        out = torch.where(tgt_3d, q_tokens + self.o_proj(ctx), zeros_q)
        out = torch.where(tgt_3d, out + self.ffn(self.norm_ff(out)), zeros_q)
        return out


class SE3LatentPredictor(nn.Module):
    """Lightweight ``SE(3)``-invariant latent predictor for masked 3D sub-pocket JEPA.

    Predicts teacher target latents ``[B, N, emb_dim]`` strictly from:
      - Student visible context ``(vis_emb, vis_vec, vis_coords)`` (optionally augmented with
        student sub-pocket super-tokens), and
      - Target 3D coordinates ``tgt_coords`` expressed in the **exact same student coordinate
        frame** (with ZERO target chemical features leaked!).

    Cross-attention geometric biases combine Bessel RBFs of relative distances ``||x_t - x_v||``,
    smooth cosine cutoff, polar directional inner products ``<W_p v_v, d_tv>``, and optional
    chiral axial inner products ``<v_v^x, d_tv>`` (exact ``SO(3)``-invariant and reflection-odd).
    """

    emb_dim: int
    vector_dim: int
    num_layers: int
    num_heads: int
    cutoff: float
    use_se3_chirality: bool

    def __init__(
        self,
        emb_dim: int,
        vector_dim: int = 64,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        num_rbf: int = 16,
        cutoff: float = 12.0,
        smooth_delta: float = 0.5,
        use_se3_chirality: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if emb_dim < 4:
            raise ValueError(f"emb_dim must be >= 4, got {emb_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.emb_dim = int(emb_dim)
        self.vector_dim = int(vector_dim)
        self.num_layers = int(num_layers)
        self.num_heads = max(1, min(int(num_heads), self.emb_dim))
        while self.emb_dim % self.num_heads != 0:
            self.num_heads -= 1
        self.cutoff = float(cutoff)
        self.smooth_delta = float(smooth_delta)
        self.use_se3_chirality = bool(use_se3_chirality)

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.emb_dim, dtype=torch.float32) * 0.02)
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=self.cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=self.cutoff)
        self.rbf_bias_proj = nn.Linear(num_rbf, self.num_heads)
        self.rbf_init_proj = nn.Linear(num_rbf, self.emb_dim)

        self.polar_proj: VectorNeuronLinear | None
        self.chiral_a: VectorNeuronLinear | None
        self.chiral_b: VectorNeuronLinear | None
        self.dir_bias_scale: nn.Parameter | None
        if self.vector_dim > 0:
            self.polar_proj = VectorNeuronLinear(self.vector_dim, self.num_heads)
            self.dir_bias_scale = nn.Parameter(
                torch.full((2, self.num_heads), 0.25, dtype=torch.float32)
            )
            if self.use_se3_chirality:
                self.chiral_a = VectorNeuronLinear(self.vector_dim, self.num_heads)
                self.chiral_b = VectorNeuronLinear(self.vector_dim, self.num_heads)
            else:
                self.chiral_a = None
                self.chiral_b = None
        else:
            self.polar_proj = None
            self.dir_bias_scale = None
            self.chiral_a = None
            self.chiral_b = None

        self.layers = nn.ModuleList(
            [
                _SE3CrossAttnPredictorLayer(
                    emb_dim=self.emb_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_head = nn.Sequential(
            RMSNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward(
        self,
        vis_emb: Tensor,
        vis_vec: Tensor | None,
        vis_coords: Tensor,
        visible_mask: Tensor,
        tgt_coords: Tensor,
        target_mask: Tensor,
        *,
        subpocket_emb: Tensor | None = None,
        subpocket_vec: Tensor | None = None,
        subpocket_coords: Tensor | None = None,
        subpocket_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict target latent embeddings ``[B, N, emb_dim]`` in the student's frame."""
        batch_size, num_tgt, _ = tgt_coords.shape
        vis_bool = visible_mask.to(torch.bool)
        tgt_bool = target_mask.to(torch.bool)

        # Concatenate optional student sub-pocket super-tokens onto the visible context bank.
        if (
            subpocket_emb is not None
            and subpocket_coords is not None
            and subpocket_mask is not None
        ):
            ctx_emb = torch.cat((vis_emb, subpocket_emb), dim=1)
            ctx_coords = torch.cat((vis_coords, subpocket_coords), dim=1)
            ctx_mask = torch.cat((vis_bool, subpocket_mask.to(torch.bool)), dim=1)
            if vis_vec is not None and subpocket_vec is not None:
                ctx_vec: Tensor | None = torch.cat((vis_vec, subpocket_vec), dim=1)
            else:
                ctx_vec = None
        else:
            ctx_emb = vis_emb
            ctx_coords = vis_coords
            ctx_mask = vis_bool
            ctx_vec = vis_vec

        diff = tgt_coords.unsqueeze(2) - ctx_coords.unsqueeze(1)  # [B, N_tgt, C, 3]
        sq_d = (diff * diff).sum(dim=-1)
        dist = torch.sqrt(sq_d + 1e-8)
        rbf_env = self.rbf(dist) * self.cutoff_fn(dist).unsqueeze(-1)  # [B, N_tgt, C, R]

        geo_bias: Tensor = self.rbf_bias_proj(rbf_env).permute(0, 3, 1, 2).contiguous()

        if ctx_vec is not None and self.polar_proj is not None and self.dir_bias_scale is not None:
            delta_sq = self.smooth_delta * self.smooth_delta
            dir_tc = diff * torch.rsqrt(sq_d.unsqueeze(-1) + delta_sq)  # [B, N_tgt, C, 3]
            safe_ctx_vec = torch.where(
                ctx_mask.unsqueeze(-1).unsqueeze(-1), ctx_vec, torch.zeros_like(ctx_vec)
            )
            p_vec = self.polar_proj(safe_ctx_vec)  # [B, C, H, 3]
            p_dot = torch.einsum("btcd,bchd->bhtc", dir_tc, p_vec)
            geo_bias = geo_bias + p_dot * self.dir_bias_scale[0].view(1, self.num_heads, 1, 1)

            if self.use_se3_chirality and self.chiral_a is not None and self.chiral_b is not None:
                ax_vec = chiral_cross_product(
                    self.chiral_a(safe_ctx_vec),
                    self.chiral_b(safe_ctx_vec),
                    normalize=True,
                    eps=1.0,
                )  # [B, C, H, 3]
                ax_dot = torch.einsum("btcd,bchd->bhtc", dir_tc, ax_vec)
                geo_bias = geo_bias + ax_dot * self.dir_bias_scale[1].view(1, self.num_heads, 1, 1)

        # Initialize target queries from shared mask token + invariant radial context summary
        # (zero target chemical features!).
        ctx_valid_f = ctx_mask.to(torch.float32)  # [B, C]
        rbf_masked = rbf_env * ctx_valid_f.unsqueeze(1).unsqueeze(-1)
        ctx_denom = ctx_valid_f.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        rbf_summary = rbf_masked.sum(dim=2) / ctx_denom  # [B, N_tgt, R]

        q_tokens = self.mask_token.expand(batch_size, num_tgt, -1) + self.rbf_init_proj(rbf_summary)
        tgt_3d = tgt_bool.unsqueeze(-1)
        q_tokens = torch.where(tgt_3d, q_tokens, torch.zeros_like(q_tokens))

        for layer in self.layers:
            q_tokens = layer(q_tokens, ctx_emb, ctx_mask, tgt_bool, geo_bias)

        pred: Tensor = self.out_head(q_tokens)
        return torch.where(tgt_3d, pred, torch.zeros_like(pred))


class SubPocketJEPAPretrainer(nn.Module):
    """Anti-shortcut 3D Sub-Pocket JEPA + Sonata + optional Concerto SSL Pretrainer.

    Maintains:
      - Online trainable ``student`` encoder,
      - Momentum-updated EMA ``teacher`` encoder (strictly ``requires_grad=False`` & ``eval()``),
      - Student-frame :class:`SE3LatentPredictor` (zero coordinate or chemical leak),
      - Running DINOv2-style target centering buffer ``ema_center``,
      - :class:`VICRegLoss` evaluated strictly on ``H_vis[visible_mask]``, and
      - Optional :class:`ConcertoCrossModalLoss` when paired ligand embeddings are supplied.
    """

    emb_dim: int
    vector_dim: int
    ema_decay_start: float
    ema_decay_end: float
    total_steps: int
    center_momentum: float
    num_mask_blocks: int
    mask_min_radius: float
    mask_max_radius: float
    max_mask_ratio: float
    coord_jitter_std: float
    radial_drop_prob: float
    feat_drop_prob: float
    vicreg_weight: float
    cosine_loss_weight: float
    cross_modal_weight: float
    ema_center: Tensor
    ema_step: Tensor

    def __init__(
        self,
        encoder: nn.Module,
        *,
        emb_dim: int | None = None,
        vector_dim: int | None = None,
        predictor_layers: int = 2,
        predictor_heads: int = 4,
        predictor_cutoff: float = 12.0,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        total_steps: int = 10_000,
        center_momentum: float = 0.9,
        num_mask_blocks: int = 3,
        mask_min_radius: float = 4.5,
        mask_max_radius: float = 6.0,
        max_mask_ratio: float = 0.60,
        coord_jitter_std: float = 0.20,
        radial_drop_prob: float = 0.80,
        feat_drop_prob: float = 0.20,
        vicreg_weight: float = 0.1,
        vicreg_var_weight: float = 1.0,
        vicreg_cov_weight: float = 0.04,
        cosine_loss_weight: float = 0.5,
        cross_modal_dim: int | None = None,
        cross_modal_proj_dim: int = 128,
        cross_modal_weight: float = 0.25,
        use_se3_chirality: bool | None = None,
    ) -> None:
        super().__init__()
        resolved_emb = int(emb_dim if emb_dim is not None else getattr(encoder, "out_dim", 256))
        resolved_vec = int(
            vector_dim if vector_dim is not None else getattr(encoder, "vector_dim", 64)
        )
        resolved_chiral = bool(
            use_se3_chirality
            if use_se3_chirality is not None
            else getattr(encoder, "use_se3_chirality", True)
        )

        self.emb_dim = resolved_emb
        self.vector_dim = resolved_vec
        self.ema_decay_start = float(ema_decay_start)
        self.ema_decay_end = float(ema_decay_end)
        self.total_steps = max(1, int(total_steps))
        self.center_momentum = float(center_momentum)
        self.num_mask_blocks = int(num_mask_blocks)
        self.mask_min_radius = float(mask_min_radius)
        self.mask_max_radius = float(mask_max_radius)
        self.max_mask_ratio = float(max_mask_ratio)
        self.coord_jitter_std = float(coord_jitter_std)
        self.radial_drop_prob = float(radial_drop_prob)
        self.feat_drop_prob = float(feat_drop_prob)
        self.vicreg_weight = float(vicreg_weight)
        self.cosine_loss_weight = float(cosine_loss_weight)
        self.cross_modal_weight = float(cross_modal_weight)

        self.student: nn.Module = encoder
        self.teacher: nn.Module = copy.deepcopy(encoder)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.predictor = SE3LatentPredictor(
            emb_dim=self.emb_dim,
            vector_dim=self.vector_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            cutoff=predictor_cutoff,
            use_se3_chirality=resolved_chiral,
        )
        self.vicreg = VICRegLoss(
            gamma=1.0,
            var_weight=vicreg_var_weight,
            cov_weight=vicreg_cov_weight,
        )

        self.concerto_loss: ConcertoCrossModalLoss | None
        if cross_modal_dim is not None:
            self.concerto_loss = ConcertoCrossModalLoss(
                pocket_dim=self.emb_dim,
                ligand_dim=int(cross_modal_dim),
                proj_dim=int(cross_modal_proj_dim),
            )
        else:
            self.concerto_loss = None

        self.register_buffer(
            "ema_center",
            torch.zeros((1, 1, self.emb_dim), dtype=torch.float32),
        )
        self.register_buffer("ema_step", torch.tensor(0, dtype=torch.long))

    def train(self, mode: bool = True) -> SubPocketJEPAPretrainer:
        """Keep the EMA teacher strictly frozen in ``eval()`` mode regardless of outer mode."""
        super().train(mode)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        return self

    def current_ema_decay(self, step: int | None = None) -> float:
        """Cosine EMA decay schedule rising from ``ema_decay_start`` to ``ema_decay_end``."""
        cur = int(self.ema_step.item()) if step is None else int(step)
        progress = min(1.0, max(0.0, float(cur) / float(self.total_steps)))
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * cosine

    @torch.no_grad()
    def update_ema_teacher(self, decay: float | None = None) -> float:
        """Update EMA teacher parameters & buffers from online student and increment step."""
        tau = float(self.current_ema_decay() if decay is None else decay)
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"EMA decay must lie in [0, 1], got {tau}")

        for p_tea, p_stu in zip(self.teacher.parameters(), self.student.parameters(), strict=True):
            p_tea.data.mul_(tau).add_(p_stu.detach().data, alpha=1.0 - tau)
        for b_tea, b_stu in zip(self.teacher.buffers(), self.student.buffers(), strict=True):
            if b_tea.is_floating_point():
                b_tea.data.mul_(tau).add_(b_stu.detach().data, alpha=1.0 - tau)
            else:
                b_tea.data.copy_(b_stu.detach().data)

        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.ema_step.add_(1)
        return tau

    def _encode_student(
        self,
        coords: Tensor,
        node_feats: Tensor,
        visible_mask: Tensor,
    ) -> tuple[PocketEncoderOutput, SubPocketEncodingOutput | None]:
        """Encode visible student atoms (using sub-pocket super-tokens when supported)."""
        if isinstance(self.student, SubPocketVNJEPAEncoder):
            sub_out = self.student.encode_with_subpockets(coords, node_feats, visible_mask)
            return sub_out.encoder_output, sub_out
        enc_out: PocketEncoderOutput = self.student(coords, node_feats, visible_mask)
        return enc_out, None

    def forward_ssl(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
        *,
        ligand_emb: Tensor | None = None,
        ligand_pair_mask: Tensor | None = None,
        visible_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        teacher_rot: Tensor | None = None,
        apply_obscuration: bool = True,
        update_center: bool = True,
        generator: torch.Generator | None = None,
    ) -> JEPASSLOutput:
        """Compute anti-shortcut 3D Sub-Pocket JEPA + VICReg + optional Concerto SSL loss."""
        coords_f = coords.to(torch.float32)
        feats_f = node_feats.to(torch.float32)
        mask_bool = atom_mask.to(torch.bool)
        batch_size = coords_f.shape[0]
        device = coords_f.device

        if visible_mask is None or target_mask is None:
            vis_mask, tgt_mask = sample_spherical_block_mask(
                coords_f,
                mask_bool,
                num_blocks=self.num_mask_blocks,
                min_radius=self.mask_min_radius,
                max_radius=self.mask_max_radius,
                max_mask_ratio=self.max_mask_ratio,
                generator=generator,
            )
        else:
            vis_mask = visible_mask.to(device=device, dtype=torch.bool) & mask_bool
            tgt_mask = target_mask.to(device=device, dtype=torch.bool) & mask_bool & (~vis_mask)

        if apply_obscuration:
            stu_coords, stu_feats, radial_obscured_frac = apply_sonata_spatial_obscuration(
                coords_f,
                feats_f,
                vis_mask,
                coord_jitter_std=self.coord_jitter_std,
                radial_drop_prob=self.radial_drop_prob,
                feat_drop_prob=self.feat_drop_prob,
                generator=generator,
            )
        else:
            stu_coords = coords_f
            stu_feats = torch.where(vis_mask.unsqueeze(-1), feats_f, torch.zeros_like(feats_f))
            radial_obscured_frac = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Student-frame coordinate alignment: BOTH visible and target coords live in the student's
        # centered visible frame (zero coordinate leak from the teacher's rotated view!).
        vis_coords_c, vis_offset = center_masked_coords_with_offset(stu_coords, vis_mask)
        tgt_coords_c = torch.where(
            tgt_mask.unsqueeze(-1),
            coords_f - vis_offset.unsqueeze(1),
            torch.zeros_like(coords_f),
        )

        # 1. Online Student forward on visible context only.
        stu_out, stu_sub_out = self._encode_student(stu_coords, stu_feats, vis_mask)

        # 2. Frozen EMA Teacher forward on full clean pocket under independent SO(3) rotation.
        self.teacher.eval()
        with torch.no_grad():
            rot = (
                sample_random_so3_rotations(batch_size, device=device, generator=generator)
                if teacher_rot is None
                else teacher_rot.to(device=device, dtype=torch.float32)
            )
            teacher_coords = torch.bmm(coords_f, rot.transpose(-1, -2))
            teacher_out: PocketEncoderOutput = self.teacher(teacher_coords, feats_f, mask_bool)
            raw_teacher_emb = teacher_out.node_emb.detach()

            centered_teacher = raw_teacher_emb - self.ema_center
            normed_teacher = F.layer_norm(centered_teacher, (self.emb_dim,))
            normed_teacher = torch.where(
                mask_bool.unsqueeze(-1), normed_teacher, torch.zeros_like(normed_teacher)
            )

            if self.training and update_center and bool(mask_bool.any().item()):
                batch_center = raw_teacher_emb[mask_bool].mean(dim=0).view(1, 1, self.emb_dim)
                self.ema_center.mul_(self.center_momentum).add_(
                    batch_center, alpha=1.0 - self.center_momentum
                )

        # 3. Student-frame SE(3)-invariant latent predictor over masked targets.
        pred_tgt_emb = self.predictor(
            vis_emb=stu_out.node_emb,
            vis_vec=stu_out.vec_emb,
            vis_coords=vis_coords_c,
            visible_mask=vis_mask,
            tgt_coords=tgt_coords_c,
            target_mask=tgt_mask,
            subpocket_emb=stu_sub_out.subpocket_tokens if stu_sub_out is not None else None,
            subpocket_vec=stu_sub_out.subpocket_vec if stu_sub_out is not None else None,
            subpocket_coords=stu_sub_out.subpocket_coords if stu_sub_out is not None else None,
            subpocket_mask=stu_sub_out.subpocket_mask if stu_sub_out is not None else None,
        )

        # 4. Target-only latent matching loss (Smooth-L1 + cosine similarity).
        if bool(tgt_mask.any().item()):
            pred_flat = pred_tgt_emb[tgt_mask]
            tgt_flat = normed_teacher[tgt_mask].detach()
            smooth_l1 = F.smooth_l1_loss(pred_flat, tgt_flat)
            cos_sim = F.cosine_similarity(pred_flat, tgt_flat, dim=-1, eps=1e-6)
            cosine_loss = (1.0 - cos_sim).mean()
            jepa_loss = smooth_l1 + self.cosine_loss_weight * cosine_loss
        else:
            zero = pred_tgt_emb.sum() * 0.0
            smooth_l1 = zero
            cosine_loss = zero
            jepa_loss = zero

        # 5. Anti-collapse VICReg + RankMe effective rank strictly on visible unpadded atoms.
        if bool(vis_mask.any().item()):
            h_vis = stu_out.node_emb[vis_mask]
            vicreg_out = self.vicreg(h_vis)
            eff_rank = compute_effective_rank(h_vis)
        else:
            zero_v = stu_out.node_emb.sum() * 0.0
            vicreg_out = VICRegOutput(total=zero_v, var_loss=zero_v, cov_loss=zero_v)
            eff_rank = torch.tensor(1.0, device=device, dtype=torch.float32)

        total_loss = jepa_loss + self.vicreg_weight * vicreg_out.total

        # 6. Optional Concerto cross-modal pocket-ligand InfoNCE alignment.
        if ligand_emb is not None and self.concerto_loss is not None:
            cross_modal_loss = self.concerto_loss(
                stu_out.pooled_emb,
                ligand_emb,
                pair_mask=ligand_pair_mask,
            )
            total_loss = total_loss + self.cross_modal_weight * cross_modal_loss
        else:
            cross_modal_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        return JEPASSLOutput(
            total_loss=total_loss,
            jepa_loss=jepa_loss,
            smooth_l1_loss=smooth_l1,
            cosine_loss=cosine_loss,
            vicreg_loss=vicreg_out.total,
            var_loss=vicreg_out.var_loss,
            cov_loss=vicreg_out.cov_loss,
            cross_modal_loss=cross_modal_loss,
            effective_rank=eff_rank,
            visible_mask=vis_mask,
            target_mask=tgt_mask,
            radial_obscured_frac=radial_obscured_frac,
        )

    def pretrain_step(
        self,
        batch: Mapping[str, Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip: float | None = 1.0,
        ema_decay: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        """Run one complete SSL optimization + EMA teacher update step on a batch dict.

        Expected batch keys: ``"coords"``, ``"node_feats"``, ``"atom_mask"``, plus optional
        ``"ligand_emb"`` and ``"ligand_pair_mask"`` for Concerto cross-modal supervision.
        """
        self.train()
        optimizer.zero_grad(set_to_none=True)

        ssl_out = self.forward_ssl(
            coords=batch["coords"],
            node_feats=batch["node_feats"],
            atom_mask=batch["atom_mask"],
            ligand_emb=batch.get("ligand_emb"),
            ligand_pair_mask=batch.get("ligand_pair_mask"),
            generator=generator,
        )
        torch.autograd.backward(ssl_out.total_loss)

        trainable_params = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
        if grad_clip is not None and grad_clip > 0.0 and trainable_params:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip)).item()
            )
        elif trainable_params:
            grad_sq = sum(
                float((p.grad.detach() ** 2).sum().item())
                for p in trainable_params
                if p.grad is not None
            )
            grad_norm = math.sqrt(grad_sq)
        else:
            grad_norm = 0.0

        optimizer.step()
        decay_used = self.update_ema_teacher(decay=ema_decay)

        return {
            "loss": float(ssl_out.total_loss.detach().item()),
            "jepa_loss": float(ssl_out.jepa_loss.detach().item()),
            "smooth_l1_loss": float(ssl_out.smooth_l1_loss.detach().item()),
            "cosine_loss": float(ssl_out.cosine_loss.detach().item()),
            "vicreg_loss": float(ssl_out.vicreg_loss.detach().item()),
            "var_loss": float(ssl_out.var_loss.detach().item()),
            "cov_loss": float(ssl_out.cov_loss.detach().item()),
            "cross_modal_loss": float(ssl_out.cross_modal_loss.detach().item()),
            "effective_rank": float(ssl_out.effective_rank.detach().item()),
            "grad_norm": grad_norm,
            "ema_decay": decay_used,
            "radial_obscured_frac": float(ssl_out.radial_obscured_frac.detach().item()),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        use_teacher: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save SSL pretrainer state and ``encoder_state_dict`` for ``jepa_ssl_ckpt``."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        student_sd = {k: v.detach().cpu() for k, v in self.student.state_dict().items()}
        teacher_sd = {k: v.detach().cpu() for k, v in self.teacher.state_dict().items()}
        payload: dict[str, Any] = {
            "encoder_state_dict": teacher_sd if use_teacher else student_sd,
            "student_state_dict": student_sd,
            "teacher_state_dict": teacher_sd,
            "predictor_state_dict": {
                k: v.detach().cpu() for k, v in self.predictor.state_dict().items()
            },
            "ema_center": self.ema_center.detach().cpu(),
            "ema_step": int(self.ema_step.item()),
            "metadata": dict(metadata) if metadata is not None else {},
        }
        if self.concerto_loss is not None:
            payload["concerto_state_dict"] = {
                k: v.detach().cpu() for k, v in self.concerto_loss.state_dict().items()
            }
        torch.save(payload, out_path)
        return out_path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Restore student, EMA teacher, predictor, and centering buffers from checkpoint."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"JEPA SSL checkpoint not found: {ckpt_path}")
        try:
            payload: Any = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except Exception:
            payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid JEPA checkpoint format in {ckpt_path}")

        if "student_state_dict" in payload:
            self.student.load_state_dict(payload["student_state_dict"], strict=True)
        elif "encoder_state_dict" in payload:
            self.student.load_state_dict(payload["encoder_state_dict"], strict=True)

        if "teacher_state_dict" in payload:
            self.teacher.load_state_dict(payload["teacher_state_dict"], strict=True)
        elif "encoder_state_dict" in payload:
            self.teacher.load_state_dict(payload["encoder_state_dict"], strict=True)

        if "predictor_state_dict" in payload:
            self.predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
        if "concerto_state_dict" in payload and self.concerto_loss is not None:
            self.concerto_loss.load_state_dict(payload["concerto_state_dict"], strict=True)
        if "ema_center" in payload:
            self.ema_center.copy_(payload["ema_center"].to(device=self.ema_center.device))
        if "ema_step" in payload:
            self.ema_step.fill_(int(payload["ema_step"]))

        self.teacher.requires_grad_(False)
        self.teacher.eval()
        meta = payload.get("metadata", {})
        return dict(meta) if isinstance(meta, Mapping) else {}


def load_jepa_encoder_checkpoint(
    encoder: nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    prefer_teacher: bool = True,
) -> None:
    """Load pretrained encoder weights from a ``jepa_ssl_ckpt`` file into any encoder module."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"JEPA SSL checkpoint not found: {path}")
    try:
        payload: Any = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        payload = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid JEPA checkpoint payload type: {type(payload).__name__}")

    key_order = (
        ("teacher_state_dict", "encoder_state_dict", "student_state_dict")
        if prefer_teacher
        else ("student_state_dict", "encoder_state_dict", "teacher_state_dict")
    )
    selected: Mapping[str, Any] = payload
    for key in key_order:
        if key in payload and isinstance(payload[key], Mapping):
            selected = payload[key]
            break

    state = {
        str(k): v.to(dtype=torch.float32) for k, v in selected.items() if isinstance(v, Tensor)
    }
    encoder.load_state_dict(state, strict=True)


PocketJEPAPretrainer = SubPocketJEPAPretrainer


__all__ = [
    "ConcertoCrossModalLoss",
    "JEPASSLOutput",
    "PocketJEPAPretrainer",
    "SE3LatentPredictor",
    "SubPocketJEPAPretrainer",
    "VICRegLoss",
    "VICRegOutput",
    "apply_sonata_spatial_obscuration",
    "compute_effective_rank",
    "load_jepa_encoder_checkpoint",
    "sample_random_so3_rotations",
    "sample_spherical_block_mask",
]
