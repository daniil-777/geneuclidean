"""SubPocketVN-JEPA: Competitive Multi-Scale Chiral Vector-Neuron Sub-Pocket Transformer.

Implements Method 3 from ``docs/pointcloud_cv_3d_atom_encoding_plan.md`` (§4.3):
  1. **Local Dense Chiral Vector-Neuron Attention** (:class:`ChiralLocalVNAttention`) over
     tie-safe :func:`~geneuclidean.models.encoders.dense_knn.build_dense_knn_graph` (zero
     ``index_add_`` scatter contention, exact ``O(N * K)`` scaling up to ``N = 1500+``).
  2. **Chiral VN-SwiGLU Atom Mixing** (:class:`ChiralVNSwiGLU`) combining
     :class:`~geneuclidean.models.decoders.transformer.RMSNorm` / ``SwiGLU`` with
     :class:`~geneuclidean.models.encoders.tensor_ops.VectorNeuronLinear`,
     :class:`~geneuclidean.models.encoders.tensor_ops.VectorNeuronLeakyReLU`,
     :class:`~geneuclidean.models.encoders.tensor_ops.VectorNeuronLayerNorm`, and bounded
     :func:`~geneuclidean.models.encoders.tensor_ops.chiral_cross_product` scalar triple products.
  3. **Competitive Equivariant Sub-Pocket Slot Pooling** (:class:`CompetitiveSubPocketSlotPool`):
     :math:`C^1`-smooth competitive Slot-Attention + 1 spatial localization refinement step
     forming ``M = 24`` sub-pocket super-tokens (**16 general geometric + 8 Dora-style
     pharmacophore/metal-salient slots** softly seeded from ``node_feats[..., 37:42]`` and
     ``node_feats[..., 8]``), protected by a :math:`\\delta = 0.5\\,\\text{\\AA}` smooth epoch-0
     coincident-centroid guard.
  4. **Global M x M Sub-Pocket Transformer** (:class:`SubPocketGlobalTransformer`): native SDPA +
     :class:`~geneuclidean.models.encoders.tensor_ops.VectorNeuronAttentionBlock` biased by
     :class:`~geneuclidean.models.encoders.electrostatics.GeoTransformerPairTripletBias`
     (pairwise RBF + triplet angles + signed PPF) and
     :class:`~geneuclidean.models.encoders.electrostatics.ScreenedCoulombBias` (``LES`` ``1/r``),
     plus a ``Q = 16`` Perceiver bridge aligned directly to
     :class:`~geneuclidean.models.llm_bridge.PocketPerceiverResampler`.

All parameters, buffers, and operations strictly use ``torch.float32`` with zero external C++
extensions, guaranteeing exact :math:`SE(3)` equivariance/chirality on CPU, CUDA, and Apple MPS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.config import EncoderConfig
from geneuclidean.models.decoders.transformer import RMSNorm, SwiGLU
from geneuclidean.models.encoders.dense_knn import (
    DenseKNNGraph,
    build_dense_knn_graph,
    safe_masked_softmax,
)
from geneuclidean.models.encoders.electrostatics import (
    GeoTransformerPairTripletBias,
    ScreenedCoulombBias,
)
from geneuclidean.models.encoders.rbf import (
    BesselRBF,
    EquivariantRMSNorm,
    MaskedAttentionPool,
    PocketEncoderOutput,
)
from geneuclidean.models.encoders.tensor_ops import (
    VectorNeuronAttentionBlock,
    VectorNeuronLayerNorm,
    VectorNeuronLeakyReLU,
    VectorNeuronLinear,
    chiral_cross_product,
)

# Canonical 52-D pocket featurizer indices (see geneuclidean.data.featurizer):
# [8] = catalytic metal flag; [37:42] = aromatic, HBA, HBD, pos_ionizable, neg_ionizable;
# [46] = Amber/Gasteiger partial charge proxy.
_METAL_FEAT_IDX: int = 8
_PHARMACOPHORE_SLICE: slice = slice(37, 42)
_CHARGE_PRIOR_IDX: int = 46


class SubPocketEncodingOutput(NamedTuple):
    """Structured output returned by :meth:`SubPocketVNJEPAEncoder.encode_with_subpockets`.

    Attributes:
        encoder_output: Standard :class:`PocketEncoderOutput` (``node_emb [B, N, out_dim]``,
            ``pooled_emb [B, out_dim]``, ``vec_emb [B, N, vector_dim, 3]``).
        subpocket_tokens: Refined ``SE(3)``-invariant sub-pocket super-tokens ``[B, M, out_dim]``.
        subpocket_coords: ``SO(3)``-equivariant (translation-invariant) sub-pocket centroids in the
            centered pocket frame of shape ``[B, M, 3]``. World-space coordinates equal
            ``subpocket_coords + centroid_offset.unsqueeze(1)``.
        subpocket_vec: ``SE(3)``-equivariant sub-pocket 3D vector embeddings
            ``[B, M, vector_dim, 3]``.
        slot_weights: Atom-to-slot pooling weights ``[B, M, N]`` (strictly ``0.0`` on dummy atoms,
            normalized over valid atoms per active slot).
        subpocket_mask: Boolean sub-pocket validity mask ``[B, M]`` (``True`` when the pocket has
            at least one valid atom).
        centroid_offset: Subtracted world-space valid-atom centroid ``[B, 3]``.
    """

    encoder_output: PocketEncoderOutput
    subpocket_tokens: Tensor
    subpocket_coords: Tensor
    subpocket_vec: Tensor
    slot_weights: Tensor
    subpocket_mask: Tensor
    centroid_offset: Tensor


def _extract_salience_indicator(raw_node_feats: Tensor | None, atom_mask: Tensor) -> Tensor:
    """Compute bounded ``[B, N]`` Dora pharmacophore + catalytic-metal salience flags in ``[0, 1]``.

    Extracts ``max(node_feats[..., 37:42], node_feats[..., 8])`` when the canonical 52-D
    descriptor columns are present, safely falling back to zeros for narrower custom inputs.
    """
    mask_bool = atom_mask.to(torch.bool)
    batch_size, num_atoms = mask_bool.shape
    device = mask_bool.device
    if raw_node_feats is None:
        return torch.zeros((batch_size, num_atoms), device=device, dtype=torch.float32)

    feats_f = raw_node_feats.to(torch.float32)
    feat_dim = feats_f.shape[-1]
    salience = torch.zeros((batch_size, num_atoms), device=device, dtype=torch.float32)
    if feat_dim > _METAL_FEAT_IDX:
        metal = feats_f[..., _METAL_FEAT_IDX].clamp(min=0.0, max=1.0)
        salience = torch.maximum(salience, metal)
    if feat_dim >= 42:
        pharm = feats_f[..., _PHARMACOPHORE_SLICE].clamp(min=0.0, max=1.0).amax(dim=-1)
        salience = torch.maximum(salience, pharm)
    return torch.where(mask_bool, salience, torch.zeros_like(salience))


def _extract_charge_prior(raw_node_feats: Tensor | None, atom_mask: Tensor) -> Tensor:
    """Extract clamped ``[-1, 1]`` partial charge prior from ``node_feats[..., 46]``."""
    mask_bool = atom_mask.to(torch.bool)
    batch_size, num_atoms = mask_bool.shape
    device = mask_bool.device
    if raw_node_feats is None or raw_node_feats.shape[-1] <= _CHARGE_PRIOR_IDX:
        return torch.zeros((batch_size, num_atoms), device=device, dtype=torch.float32)
    q_prior = raw_node_feats[..., _CHARGE_PRIOR_IDX].to(torch.float32).clamp(min=-1.0, max=1.0)
    return torch.where(mask_bool, q_prior, torch.zeros_like(q_prior))


def _build_flat_neighbor_index(nbr_idx: Tensor) -> Tensor:
    """Flatten ``[B, N, K]`` local neighbor indices into ``[B * N * K]`` for fast MPS gather."""
    batch_size, num_atoms, _ = nbr_idx.shape
    offsets = (
        torch.arange(batch_size, device=nbr_idx.device, dtype=torch.long).view(batch_size, 1, 1)
        * num_atoms
    )
    return (offsets + nbr_idx).reshape(-1)


def _gather_neighbors_fast(
    x: Tensor,
    flat_nbr_idx: Tensor,
    batch_size: int,
    num_atoms: int,
    k_nbr: int,
) -> Tensor:
    """Gather ``[B, N, *feat] -> [B, N, K, *feat]`` via flat ``index_select`` for fast MPS."""
    feat_shape = x.shape[2:]
    if batch_size == 0 or num_atoms == 0 or k_nbr == 0:
        return x.new_empty((batch_size, num_atoms, k_nbr, *feat_shape))
    flat_x = x.reshape(batch_size * num_atoms, -1)
    gathered = flat_x.index_select(0, flat_nbr_idx)
    return gathered.reshape(batch_size, num_atoms, k_nbr, *feat_shape)


def _apply_vn_linear_fast(vn_linear: VectorNeuronLinear, vec: Tensor) -> Tensor:
    """Apply ``VectorNeuronLinear`` via contiguous ``matmul(W, vec)`` (zero 4D MPS transpose)."""
    return torch.matmul(vn_linear.linear.weight, vec)


def _apply_vn_lrelu_fast(vn_act: VectorNeuronLeakyReLU, vec: Tensor) -> Tensor:
    """Apply ``VectorNeuronLeakyReLU`` using contiguous ``matmul(W, vec)`` on MPS."""
    k = _apply_vn_linear_fast(vn_act.dir_proj, vec)
    dot = (vec * k).sum(dim=-1, keepdim=True)
    k_sq = (k * k).sum(dim=-1, keepdim=True)
    neg_part = torch.clamp(dot, max=0.0)
    return vec - (1.0 - vn_act.negative_slope) * (neg_part / (k_sq + vn_act.eps)) * k


def _resolve_divisible_heads(dim: int, preferred_heads: int) -> int:
    """Return the largest head count ``<= preferred_heads`` that evenly divides ``dim``."""
    target = max(1, min(int(preferred_heads), int(dim)))
    for h in range(target, 0, -1):
        if dim % h == 0:
            return h
    return 1


def smooth_pair_directions(
    coords_i: Tensor,
    coords_j: Tensor | None = None,
    *,
    smooth_delta: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Compute epoch-0-safe pairwise displacements and smooth unit directions.

    Uses :math:`\\tilde{\\mathbf{d}}_{ij} = (\\boldsymbol{\\mu}_i - \\boldsymbol{\\mu}_j) /`
    :math:`\\sqrt{\\|\\boldsymbol{\\mu}_i - \\boldsymbol{\\mu}_j\\|^2 + \\delta^2}`
    with :math:`\\delta = 0.5\\,\\text{\\AA}` so coincident epoch-0 sub-pocket centroids never
    produce ``0 / 0`` singularities or exploding backward gradients.

    Returns:
        Tuple ``(dist [B, M_i, M_j], smooth_dir [B, M_i, M_j, 3])``.
    """
    if smooth_delta <= 0.0:
        raise ValueError(f"smooth_delta must be positive, got {smooth_delta}")
    cj = coords_i if coords_j is None else coords_j
    diff = coords_i.unsqueeze(2) - cj.unsqueeze(1)  # [B, M_i, M_j, 3]
    sq_d = (diff * diff).sum(dim=-1)
    dist = torch.sqrt(sq_d + 1e-8)
    smooth_dir = diff * torch.rsqrt(sq_d.unsqueeze(-1) + float(smooth_delta * smooth_delta))
    return dist, smooth_dir


class ChiralLocalVNAttention(nn.Module):
    """Dense batched ``O(N * K)`` chiral Vector-Neuron local attention block (zero ``index_add_``).

    Computes multi-head attention logits over each atom's dense ``K``-NN neighborhood from:
      1. Scalar query-key inner products ``<q_i, k_j> / sqrt(d_s)``,
      2. Vector-Neuron Frobenius inner products ``<Q_i^v, K_j^v>_F / sqrt(3 * d_v)``,
      3. Smooth Bessel RBF edge biases,
      4. Directional polar projections ``<q_bar_i, r_hat_ij>`` and ``<k_bar_j, r_hat_ij>``, and
      5. Optional bounded chiral axial bias ``<v_i^x - v_j^x, r_hat_ij>`` breaking mirror
         symmetry (``det(P) = -1``) while vanishing identically on planar/collinear geometries.

    Attention weights use :func:`safe_masked_softmax` multiplied post-softmax by the tie-safe
    per-atom smooth cosine envelope ``graph.edge_env`` (never ``+ log f_c`` inside softmax),
    guaranteeing exact ``0.0`` jump on equidistant ``K``-th neighbor tie-swaps and finite
    gradients on ``N = 1`` ions, isolated atoms, and coordinate clashes.
    """

    scalar_dim: int
    vector_dim: int
    num_heads: int
    attn_scalar_dim: int
    head_dim: int
    vn_attn_dim: int
    vn_head_dim: int
    num_rbf: int
    use_se3_chirality: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        num_heads: int = 8,
        num_rbf: int = 24,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1:
            raise ValueError(
                f"scalar_dim ({scalar_dim}) and vector_dim ({vector_dim}) must be >= 1"
            )
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_heads = _resolve_divisible_heads(self.scalar_dim, num_heads)
        self.num_rbf = int(num_rbf)
        self.use_se3_chirality = bool(use_se3_chirality)

        # Compact per-head scalar & vector widths fused into ONE contiguous neighbor gather.
        full_head_dim = self.scalar_dim // self.num_heads
        self.head_dim = max(8, full_head_dim // 2) if self.scalar_dim >= 128 else full_head_dim
        self.attn_scalar_dim = self.num_heads * self.head_dim

        target_vn_head = max(1, min(2, max(1, self.vector_dim // self.num_heads)))
        if self.vector_dim < self.num_heads:
            target_vn_head = 1
        self.vn_head_dim = int(target_vn_head)
        self.vn_attn_dim = self.num_heads * self.vn_head_dim

        self.scalar_norm = RMSNorm(self.scalar_dim, eps=eps)
        self.vec_norm = VectorNeuronLayerNorm(self.vector_dim, eps=max(eps, 1e-3))

        self.scalar_qkv_proj = nn.Linear(self.scalar_dim, 2 * self.attn_scalar_dim, bias=False)
        self.scalar_out = nn.Linear(self.attn_scalar_dim, self.scalar_dim, bias=False)

        self.vn_q = VectorNeuronLinear(self.vector_dim, self.vn_attn_dim)
        self.vn_kv = VectorNeuronLinear(self.vector_dim, self.vn_attn_dim)
        self.vn_out = VectorNeuronLinear(self.vn_attn_dim + self.num_heads, self.vector_dim)

        # Single fused radial projection for attention bias and directional edge gate.
        self.rbf_edge_proj = nn.Linear(self.num_rbf, 2 * self.num_heads)
        # Exact linear-factorized radial value projection (avoids [B, N, K, D] allocs).
        self.rbf_val_out = nn.Linear(self.num_heads * self.num_rbf, self.scalar_dim, bias=False)

        self.dir_bias_scale = nn.Parameter(torch.full((self.num_heads,), 0.25, dtype=torch.float32))

        self.chiral_a: VectorNeuronLinear | None
        self.chiral_b: VectorNeuronLinear | None
        self.chiral_bias_scale: nn.Parameter | None
        if self.use_se3_chirality:
            self.chiral_a = VectorNeuronLinear(self.vector_dim, self.num_heads)
            self.chiral_b = VectorNeuronLinear(self.vector_dim, self.num_heads)
            self.chiral_bias_scale = nn.Parameter(
                torch.full((self.num_heads,), 0.25, dtype=torch.float32)
            )
        else:
            self.chiral_a = None
            self.chiral_b = None
            self.chiral_bias_scale = None

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        graph: DenseKNNGraph,
        rbf_env: Tensor,
        atom_mask: Tensor,
        *,
        flat_nbr_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one dense local chiral VN-attention residual step on ``(s, v)``."""
        batch_size, num_atoms, _ = s.shape
        k_nbr = graph.nbr_idx.shape[2]
        mask_bool = atom_mask.to(torch.bool)
        mask_s = mask_bool.unsqueeze(-1)
        mask_v = mask_s.unsqueeze(-1)
        zeros_s = torch.zeros_like(s)
        zeros_v = torch.zeros_like(v)

        if batch_size == 0 or num_atoms == 0 or k_nbr == 0:
            return torch.where(mask_s, s, zeros_s), torch.where(mask_v, v, zeros_v)

        nbr_flat = (
            _build_flat_neighbor_index(graph.nbr_idx) if flat_nbr_idx is None else flat_nbr_idx
        )
        bn = batch_size * num_atoms
        vn_c3 = self.vn_head_dim * 3
        mask_f_s = mask_bool.to(torch.float32).unsqueeze(-1)
        mask_f_v = mask_f_s.unsqueeze(-1)

        s_n = self.scalar_norm(s) * mask_f_s
        v_n = self.vec_norm(v) * mask_f_v

        # 1. Fuse scalar + Vector-Neuron Q and KV into ONE contiguous neighbor gather + bmm.
        scale_s = 1.0 / math.sqrt(float(self.head_dim))
        scale_v = 1.0 / math.sqrt(3.0 * float(self.vn_head_dim))
        d_joint = self.head_dim + vn_c3

        q_s_flat, kv_s_flat = self.scalar_qkv_proj(s_n).chunk(2, dim=-1)
        q_s = q_s_flat.reshape(batch_size, num_atoms, self.num_heads, self.head_dim)
        kv_s = kv_s_flat.reshape(batch_size, num_atoms, self.num_heads, self.head_dim)
        q_v = _apply_vn_linear_fast(self.vn_q, v_n).view(
            batch_size, num_atoms, self.num_heads, self.vn_head_dim, 3
        )
        kv_v = _apply_vn_linear_fast(self.vn_kv, v_n).view(
            batch_size, num_atoms, self.num_heads, vn_c3
        )

        q_joint = torch.cat(
            (q_s * scale_s, q_v.reshape(batch_size, num_atoms, self.num_heads, vn_c3) * scale_v),
            dim=-1,
        )  # [B, N, H, d_joint]
        kv_joint = torch.cat((kv_s, kv_v), dim=-1)  # [B, N, H, d_joint]
        kv_nbr = _gather_neighbors_fast(kv_joint, nbr_flat, batch_size, num_atoms, k_nbr)

        # Native [B*N*H, 1, d_joint] @ [B*N*H, d_joint, K] bmm -> [B, N, H, K] (contiguous K!).
        kv_bnh = kv_nbr.permute(0, 1, 3, 2, 4).reshape(bn * self.num_heads, k_nbr, d_joint)
        q_bnh = q_joint.reshape(bn * self.num_heads, 1, d_joint)
        joint_logits = torch.bmm(q_bnh, kv_bnh.transpose(1, 2)).reshape(
            batch_size, num_atoms, self.num_heads, k_nbr
        )

        # 2. Radial Bessel + combined polar/chiral directional bias directly in [B, N, H, K].
        rbf_edge = (
            self.rbf_edge_proj(rbf_env)
            .view(batch_size, num_atoms, k_nbr, 2, self.num_heads)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # [B, N, 2, H, K]
        rbf_bias = rbf_edge[:, :, 0]
        rbf_dir_gate = rbf_edge[:, :, 1]
        edge_dir_bn = graph.edge_dir.reshape(bn, k_nbr, 3)

        dir_vec = q_v.mean(dim=-2) * self.dir_bias_scale.view(1, 1, self.num_heads, 1)
        if self.use_se3_chirality:
            assert (
                self.chiral_a is not None
                and self.chiral_b is not None
                and self.chiral_bias_scale is not None
            )
            v_ax = chiral_cross_product(
                _apply_vn_linear_fast(self.chiral_a, v_n),
                _apply_vn_linear_fast(self.chiral_b, v_n),
                normalize=True,
                eps=1.0,
            )  # [B, N, H, 3]
            dir_vec = dir_vec + v_ax * self.chiral_bias_scale.view(1, 1, self.num_heads, 1)

        geo_dir_bias = torch.bmm(
            dir_vec.reshape(bn, self.num_heads, 3), edge_dir_bn.transpose(1, 2)
        ).reshape(batch_size, num_atoms, self.num_heads, k_nbr)

        logits = joint_logits + rbf_bias + geo_dir_bias

        # 3. C1-smooth virtual-null envelope-weighted softmax (zero log(0), exact tie-swap!).
        # Weighting exp(logits) by edge_env in BOTH numerator and denominator ensures tied
        # boundary neighbors (edge_env == 0) contribute strictly 0 to the partition function.
        edge_env_h = graph.edge_env.unsqueeze(2)  # [B, N, 1, K]
        env_valid = graph.nbr_mask.unsqueeze(2) & (edge_env_h > 0.0)
        masked_logits = torch.where(env_valid, logits, torch.full_like(logits, -1e9))
        max_logit = torch.clamp(masked_logits.max(dim=-1, keepdim=True).values, min=0.0)
        exp_w = torch.where(
            env_valid,
            torch.exp(masked_logits - max_logit) * edge_env_h,
            torch.zeros_like(logits),
        )
        denom = exp_w.sum(dim=-1, keepdim=True) + torch.exp(-max_logit) + 1e-12
        attn = self.attn_dropout(exp_w / denom)  # [B, N, H, K]

        # 4. Single fused bmm aggregation for scalar + VN neighbor values, plus zero-transpose bmm.
        agg_joint = torch.bmm(attn.reshape(bn * self.num_heads, 1, k_nbr), kv_bnh).reshape(
            batch_size, num_atoms, self.num_heads, d_joint
        )
        agg_s_h, agg_v_h = torch.split(agg_joint, [self.head_dim, vn_c3], dim=-1)
        agg_s = agg_s_h.reshape(batch_size, num_atoms, self.attn_scalar_dim)
        agg_v_nbr = agg_v_h.reshape(batch_size, num_atoms, self.vn_attn_dim, 3)

        attn_hk = attn.reshape(bn, self.num_heads, k_nbr)
        agg_rbf = torch.bmm(attn_hk, rbf_env.reshape(bn, k_nbr, self.num_rbf)).reshape(
            batch_size, num_atoms, self.num_heads * self.num_rbf
        )
        ds: Tensor = self.scalar_out(agg_s) + self.rbf_val_out(agg_rbf)

        dir_gate_hk = (attn * (1.0 + rbf_dir_gate)).reshape(bn, self.num_heads, k_nbr)
        agg_dir = torch.bmm(dir_gate_hk, edge_dir_bn).reshape(
            batch_size, num_atoms, self.num_heads, 3
        )
        agg_v_all = torch.cat((agg_v_nbr, agg_dir), dim=-2)
        dv: Tensor = _apply_vn_linear_fast(self.vn_out, agg_v_all)

        s_out = (s + self.resid_dropout(ds)) * mask_f_s
        v_out = (v + dv) * mask_f_v
        return s_out, v_out


class ChiralVNSwiGLU(nn.Module):
    """Atom-wise scalar-vector mixing combining SwiGLU with chiral Vector Neurons.

    Uses :class:`VectorNeuronLinear`, :class:`VectorNeuronLeakyReLU`,
    :class:`VectorNeuronLayerNorm`, :func:`chiral_cross_product`, :class:`RMSNorm`, and
    :class:`SwiGLU`. Strictly preserves ``SE(3)`` equivariance (or ``E(3)`` when
    ``use_se3_chirality=False``) and keeps masked dummy atoms strictly ``0.0``.
    """

    scalar_dim: int
    vector_dim: int
    chiral_dim: int
    use_se3_chirality: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        use_se3_chirality: bool = True,
        ffn_mult: float = 1.5,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.chiral_dim = min(self.vector_dim, 16)
        self.use_se3_chirality = bool(use_se3_chirality)

        self.s_norm = RMSNorm(self.scalar_dim, eps=eps)
        self.v_norm = EquivariantRMSNorm(self.vector_dim, eps=eps)

        # Keep each VectorNeuronLinear output contiguous (avoids slow strided dim=-2 MPS slices).
        self.vn_u = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.vn_w = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.vn_act = VectorNeuronLeakyReLU(self.vector_dim, share_nonlinearity=True, eps=eps)

        self.chiral_a: VectorNeuronLinear | None
        self.chiral_b: VectorNeuronLinear | None
        self.chiral_z: VectorNeuronLinear | None
        if self.use_se3_chirality:
            self.chiral_a = VectorNeuronLinear(self.vector_dim, self.chiral_dim)
            self.chiral_b = VectorNeuronLinear(self.vector_dim, self.chiral_dim)
            self.chiral_z = VectorNeuronLinear(self.vector_dim, self.chiral_dim)
            self.vn_out = VectorNeuronLinear(self.vector_dim + self.chiral_dim, self.vector_dim)
            geo_in_dim = 2 * self.vector_dim + self.chiral_dim
            vec_gate_dim = self.vector_dim + self.chiral_dim
        else:
            self.chiral_a = None
            self.chiral_b = None
            self.chiral_z = None
            self.vn_out = VectorNeuronLinear(self.vector_dim, self.vector_dim)
            geo_in_dim = 2 * self.vector_dim
            vec_gate_dim = self.vector_dim

        self.geo_to_scalar = nn.Linear(geo_in_dim, self.scalar_dim, bias=False)
        d_ff = max(self.scalar_dim, int(round(self.scalar_dim * float(ffn_mult))))
        self.swiglu = SwiGLU(d_model=self.scalar_dim, d_ff=d_ff, dropout=dropout)
        self.scalar_to_vec_gates = nn.Linear(self.scalar_dim, vec_gate_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, s: Tensor, v: Tensor, atom_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Mix per-atom scalar ``[B, N, D]`` and vector ``[B, N, C_v, 3]`` features."""
        mask_f_s = atom_mask.to(torch.float32).unsqueeze(-1)
        mask_f_v = mask_f_s.unsqueeze(-1)

        s_n = self.s_norm(s) * mask_f_s
        v_n = self.v_norm(v) * mask_f_v

        u_v = _apply_vn_linear_fast(self.vn_u, v_n)
        w_v = _apply_vn_linear_fast(self.vn_w, v_n)
        v_act = _apply_vn_lrelu_fast(self.vn_act, u_v)
        v_norm = torch.sqrt((w_v * w_v).sum(dim=-1) + 1e-8)
        uw_dot = (u_v * w_v).sum(dim=-1)

        if self.use_se3_chirality:
            assert (
                self.chiral_a is not None
                and self.chiral_b is not None
                and self.chiral_z is not None
            )
            # Compact chiral_dim triple product <z_v, a_v x b_v>: SO(3)-invariant & reflection-odd.
            v_cross = chiral_cross_product(
                _apply_vn_linear_fast(self.chiral_a, v_n),
                _apply_vn_linear_fast(self.chiral_b, v_n),
                normalize=True,
                eps=1.0,
            )
            z_v = _apply_vn_linear_fast(self.chiral_z, v_n)
            triple_prod = (z_v * v_cross).sum(dim=-1)
            geo_in = torch.cat((v_norm, uw_dot, triple_prod), dim=-1)
        else:
            v_cross = None
            geo_in = torch.cat((v_norm, uw_dot), dim=-1)

        s_fused = s_n + self.geo_to_scalar(geo_in)
        ds: Tensor = self.swiglu(s_fused)

        vec_gates: Tensor = self.scalar_to_vec_gates(s_fused)
        if self.use_se3_chirality:
            assert v_cross is not None
            g_act = vec_gates[..., : self.vector_dim]
            g_chi = vec_gates[..., self.vector_dim :]
            dv_raw = torch.cat((g_act.unsqueeze(-1) * v_act, g_chi.unsqueeze(-1) * v_cross), dim=-2)
        else:
            dv_raw = vec_gates.unsqueeze(-1) * v_act
        dv: Tensor = _apply_vn_linear_fast(self.vn_out, dv_raw)

        s_out = (s + self.resid_dropout(ds)) * mask_f_s
        v_out = (v + dv) * mask_f_v
        return s_out, v_out


class CompetitiveSubPocketSlotPool(nn.Module):
    """C1-smooth Competitive Slot-Attention + 1 Spatial Localization Pass with Dora salient slots.

    Partitions ``M = num_subpockets`` learned slots (default ``24``) into:
      - ``M - M_sal`` (``16``) general geometric sub-pocket slots, and
      - ``M_sal`` (``8``) **Dora-style pharmacophore/metal-salient slots** softly biased toward
        functional anchor atoms (aromatic, HBA, HBD, pos/neg ionizable at cols ``37:42``, and
        catalytic metals at col ``8``).

    Pass 0 applies competitive ``softmax`` across slots ``M`` and normalizes over valid atoms to
    obtain initial equivariant centroids ``mu_0``. Pass 1 subtracts a learnable smooth spatial
    penalty ``softplus(gamma_m) * ||x_i - mu_0_m||^2`` so each slot localizes to a contiguous 3D
    sub-cavity rather than merging distant pocket walls.
    """

    scalar_dim: int
    vector_dim: int
    num_subpockets: int
    num_salient_slots: int
    num_geometric_slots: int
    smooth_delta: float

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        num_subpockets: int = 24,
        num_salient_slots: int = 8,
        smooth_delta: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_subpockets < 2:
            raise ValueError(f"num_subpockets must be >= 2, got {num_subpockets}")
        if smooth_delta <= 0.0:
            raise ValueError(f"smooth_delta must be positive, got {smooth_delta}")

        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_subpockets = int(num_subpockets)
        self.num_salient_slots = max(
            0, min(int(num_salient_slots), max(1, self.num_subpockets // 2))
        )
        self.num_geometric_slots = self.num_subpockets - self.num_salient_slots
        self.smooth_delta = float(smooth_delta)

        self.atom_norm = RMSNorm(self.scalar_dim, eps=eps)
        self.key_proj = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.val_proj = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.vn_val_proj = VectorNeuronLinear(self.vector_dim, self.vector_dim)

        # Orthogonal initialization encourages immediate slot diversity from step 0.
        slot_q = torch.empty((self.num_subpockets, self.scalar_dim), dtype=torch.float32)
        nn.init.orthogonal_(slot_q)
        self.slot_queries = nn.Parameter(slot_q)

        # Learnable spatial localization bandwidth gamma_m = softplus(spatial_gamma_raw).
        self.spatial_gamma_raw = nn.Parameter(
            torch.full((self.num_subpockets,), -1.5, dtype=torch.float32)
        )
        # Learnable positive log-prior weighting Dora pharmacophore/metal anchor atoms.
        self.salience_prior_raw = nn.Parameter(
            torch.full((max(1, self.num_salient_slots),), 1.25, dtype=torch.float32)
        )

        # Smooth unpooling projections back to atom streams.
        self.unpool_s = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.unpool_v = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.unpool_gate_s = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.unpool_gate_v = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def pool(
        self,
        s: Tensor,
        v: Tensor,
        centered_coords: Tensor,
        atom_mask: Tensor,
        *,
        salience_indicator: Tensor | None = None,
        charge_prior: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pool valid atoms into ``M`` competitive localized sub-pocket super-tokens.

        Returns:
            Tuple ``(sub_s [B, M, D], sub_v [B, M, C_v, 3], sub_coords [B, M, 3],
            slot_weights [B, M, N], unpool_weights [B, M, N], sub_mask [B, M],
            sub_charge_prior [B, M])``.
        """
        s_f = s.to(torch.float32)
        v_f = v.to(torch.float32)
        coords_f = centered_coords.to(torch.float32)
        batch_size, num_atoms, _ = s_f.shape
        mask_bool = atom_mask.to(torch.bool)
        mask_slot_atom = mask_bool.unsqueeze(1)  # [B, 1, N]
        has_valid = mask_bool.any(dim=1)  # [B]
        sub_mask = has_valid.unsqueeze(1).expand(batch_size, self.num_subpockets)

        if batch_size == 0 or num_atoms == 0:
            dev = s_f.device
            return (
                torch.zeros((batch_size, self.num_subpockets, self.scalar_dim), device=dev),
                torch.zeros((batch_size, self.num_subpockets, self.vector_dim, 3), device=dev),
                torch.zeros((batch_size, self.num_subpockets, 3), device=dev),
                torch.zeros((batch_size, self.num_subpockets, num_atoms), device=dev),
                torch.zeros((batch_size, self.num_subpockets, num_atoms), device=dev),
                sub_mask,
                torch.zeros((batch_size, self.num_subpockets), device=dev),
            )

        s_n = torch.where(mask_bool.unsqueeze(-1), self.atom_norm(s_f), torch.zeros_like(s_f))
        k_atoms: Tensor = self.key_proj(s_n)  # [B, N, D]
        scale = 1.0 / math.sqrt(float(self.scalar_dim))
        logits_0 = torch.einsum("md,bnd->bmn", self.slot_queries, k_atoms) * scale

        if self.num_salient_slots > 0 and salience_indicator is not None:
            sal_f = salience_indicator.to(torch.float32)
            sal = torch.where(mask_bool, sal_f, torch.zeros_like(sal_f))
            prior_weights = F.softplus(self.salience_prior_raw[: self.num_salient_slots])
            sal_bias = prior_weights.view(1, self.num_salient_slots, 1) * sal.unsqueeze(1)
            if self.num_geometric_slots > 0:
                geo_zeros = torch.zeros(
                    (batch_size, self.num_geometric_slots, num_atoms),
                    device=s_f.device,
                    dtype=torch.float32,
                )
                full_bias = torch.cat((geo_zeros, sal_bias), dim=1)
            else:
                full_bias = sal_bias
            logits_0 = logits_0 + full_bias

        # Pass 0: competitive softmax across slots M, then normalize over valid atoms.
        w_0 = torch.softmax(logits_0, dim=1)
        w_0 = torch.where(mask_slot_atom, w_0, torch.zeros_like(w_0))
        attn_0 = w_0 / (w_0.sum(dim=-1, keepdim=True) + 1e-6)
        mu_0 = torch.bmm(attn_0, coords_f)  # [B, M, 3]

        # Pass 1: smooth spatial localization around each slot's initial centroid.
        diff = coords_f.unsqueeze(1) - mu_0.unsqueeze(2)  # [B, M, N, 3]
        sq_dist = (diff * diff).sum(dim=-1)  # [B, M, N]
        gamma = F.softplus(self.spatial_gamma_raw).view(1, self.num_subpockets, 1)
        logits_1 = logits_0 - gamma * sq_dist

        unpool_weights = torch.softmax(logits_1, dim=1)
        unpool_weights = torch.where(
            mask_slot_atom, unpool_weights, torch.zeros_like(unpool_weights)
        )
        slot_weights = unpool_weights / (unpool_weights.sum(dim=-1, keepdim=True) + 1e-6)

        sub_coords = torch.bmm(slot_weights, coords_f)
        sub_s: Tensor = self.val_proj(torch.bmm(slot_weights, s_f))
        sub_v_pool = torch.bmm(
            slot_weights, v_f.reshape(batch_size, num_atoms, self.vector_dim * 3)
        ).reshape(batch_size, self.num_subpockets, self.vector_dim, 3)
        sub_v: Tensor = _apply_vn_linear_fast(self.vn_val_proj, sub_v_pool)

        if charge_prior is not None:
            q_f = charge_prior.to(torch.float32)
            q_atom = torch.where(mask_bool, q_f, torch.zeros_like(q_f))
            sub_q_prior = torch.bmm(slot_weights, q_atom.unsqueeze(-1)).squeeze(-1)
        else:
            sub_q_prior = torch.zeros(
                (batch_size, self.num_subpockets), device=s_f.device, dtype=torch.float32
            )

        # Strictly zero empty batch rows (pockets with 0 valid atoms).
        sub_mask_1 = sub_mask.unsqueeze(-1)
        sub_s = torch.where(sub_mask_1, sub_s, torch.zeros_like(sub_s))
        sub_coords = torch.where(sub_mask_1, sub_coords, torch.zeros_like(sub_coords))
        sub_v = torch.where(sub_mask_1.unsqueeze(-1), sub_v, torch.zeros_like(sub_v))
        sub_q_prior = torch.where(sub_mask, sub_q_prior, torch.zeros_like(sub_q_prior))

        return (
            sub_s,
            sub_v,
            sub_coords,
            slot_weights,
            unpool_weights,
            sub_mask,
            sub_q_prior,
        )

    def unpool(
        self,
        s: Tensor,
        v: Tensor,
        sub_s: Tensor,
        sub_v: Tensor,
        unpool_weights: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Broadcast refined sub-pocket super-tokens back onto valid atom streams."""
        mask_bool = atom_mask.to(torch.bool)
        mask_s = mask_bool.unsqueeze(-1)
        mask_v = mask_s.unsqueeze(-1)

        batch_size, num_atoms, _ = s.shape
        proj_s: Tensor = self.unpool_s(sub_s)
        proj_v: Tensor = _apply_vn_linear_fast(self.unpool_v, sub_v)
        unpool_t = unpool_weights.transpose(1, 2)  # [B, N, M]
        atom_ds = torch.bmm(unpool_t, proj_s)
        atom_dv = torch.bmm(
            unpool_t, proj_v.reshape(batch_size, self.num_subpockets, self.vector_dim * 3)
        ).reshape(batch_size, num_atoms, self.vector_dim, 3)

        s_out = torch.where(
            mask_s, s + torch.tanh(self.unpool_gate_s) * atom_ds, torch.zeros_like(s)
        )
        v_out = torch.where(
            mask_v, v + torch.tanh(self.unpool_gate_v) * atom_dv, torch.zeros_like(v)
        )
        return s_out, v_out


class _SubPocketGlobalLayer(nn.Module):
    """Single M x M global sub-pocket SDPA + VN-attention + SwiGLU layer."""

    scalar_dim: int
    vector_dim: int
    num_heads: int
    head_dim: int
    vn_heads: int

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        num_heads: int = 4,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_heads = _resolve_divisible_heads(self.scalar_dim, num_heads)
        self.head_dim = self.scalar_dim // self.num_heads
        self.vn_heads = _resolve_divisible_heads(self.vector_dim, self.num_heads)

        self.norm_s = RMSNorm(self.scalar_dim, eps=eps)
        self.qkv_s = nn.Linear(self.scalar_dim, 3 * self.scalar_dim, bias=False)
        self.out_s = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)

        # Cross-couple VN Frobenius similarities into scalar sub-pocket SDPA logits.
        self.vn_q = VectorNeuronLinear(self.vector_dim, self.num_heads)
        self.vn_k = VectorNeuronLinear(self.vector_dim, self.num_heads)
        self.vn_norm = EquivariantRMSNorm(self.vector_dim, eps=eps)
        self.vn_logit_scale = nn.Parameter(torch.full((self.num_heads,), 0.25, dtype=torch.float32))

        self.bias_to_vn: nn.Linear | None
        if self.vn_heads != self.num_heads:
            self.bias_to_vn = nn.Linear(self.num_heads, self.vn_heads, bias=False)
        else:
            self.bias_to_vn = None

        self.vn_block = VectorNeuronAttentionBlock(
            vector_dim=self.vector_dim,
            num_heads=self.vn_heads,
            use_se3_chirality=use_se3_chirality,
            dropout=dropout,
            eps=max(eps, 1e-3),
        )

        self.norm_ff = RMSNorm(self.scalar_dim, eps=eps)
        self.ffn = SwiGLU(d_model=self.scalar_dim, d_ff=2 * self.scalar_dim, dropout=dropout)

    def forward(
        self,
        sub_s: Tensor,
        sub_v: Tensor,
        sub_mask: Tensor,
        geo_bias: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Update sub-pocket scalar ``[B, M, D]`` and vector ``[B, M, C_v, 3]`` tokens."""
        batch_size, num_slots, _ = sub_s.shape
        mask_bool = sub_mask.to(torch.bool)
        mask_s = mask_bool.unsqueeze(-1)
        mask_v = mask_s.unsqueeze(-1)
        zeros_s = torch.zeros_like(sub_s)
        zeros_v = torch.zeros_like(sub_v)

        s_n = torch.where(mask_s, self.norm_s(sub_s), zeros_s)
        v_n = torch.where(mask_v, self.vn_norm(sub_v), zeros_v)

        q, k, v = self.qkv_s(s_n).chunk(3, dim=-1)
        q_h = q.view(batch_size, num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k.view(batch_size, num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v.view(batch_size, num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # Equivariant VN Frobenius similarity [B, H, M, M] added to geometric SDPA bias.
        qv = self.vn_q(v_n).permute(0, 2, 1, 3)  # [B, H, M, 3]
        kv = self.vn_k(v_n).permute(0, 2, 1, 3)  # [B, H, M, 3]
        vn_sim = torch.matmul(qv, kv.transpose(-1, -2)) * (1.0 / math.sqrt(3.0))
        total_bias = geo_bias + vn_sim * self.vn_logit_scale.view(1, self.num_heads, 1, 1)

        # Branch-free scaled dot-product attention (zero GPU->CPU .item() sync stall on MPS).
        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale + total_bias
        key_mask = mask_bool.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
        probs = safe_masked_softmax(scores, key_mask, dim=-1)
        attn_out = torch.matmul(probs, v_h)

        merged = attn_out.transpose(1, 2).contiguous().view(batch_size, num_slots, self.scalar_dim)
        sub_s = torch.where(mask_s, sub_s + self.resid_drop(self.out_s(merged)), zeros_s)
        sub_s = torch.where(mask_s, sub_s + self.ffn(self.norm_ff(sub_s)), zeros_s)

        if self.bias_to_vn is not None:
            vn_bias = self.bias_to_vn(geo_bias.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        else:
            vn_bias = geo_bias
        sub_v = self.vn_block(sub_v, mask=mask_bool, attn_bias=vn_bias)
        sub_v = torch.where(mask_v, sub_v, zeros_v)
        return sub_s, sub_v


class SubPocketGlobalTransformer(nn.Module):
    """Global ``M x M`` sub-pocket Transformer with GeoTransformer triplet/PPF + LES Coulomb bias.

    Combines :class:`GeoTransformerPairTripletBias` (with ``smooth_delta = 0.5 A`` epoch-0
    coincident-centroid protection), optional :class:`ScreenedCoulombBias` (``LES`` ``1/r``),
    native SDPA over scalar super-tokens, and :class:`VectorNeuronAttentionBlock` over equivariant
    3D vector super-tokens. Also exposes a ``Q = 16`` Perceiver bridge projection aligned directly
    to :class:`~geneuclidean.models.llm_bridge.PocketPerceiverResampler`.
    """

    scalar_dim: int
    vector_dim: int
    num_heads: int
    num_layers: int
    use_se3_chirality: bool
    use_coulomb_bias: bool
    num_perceiver_tokens: int

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        num_rbf: int = 16,
        cutoff: float = 20.0,
        smooth_delta: float = 0.5,
        use_se3_chirality: bool = True,
        use_coulomb_bias: bool = True,
        num_perceiver_tokens: int = 16,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_perceiver_tokens < 1:
            raise ValueError(f"num_perceiver_tokens must be >= 1, got {num_perceiver_tokens}")

        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_heads = _resolve_divisible_heads(self.scalar_dim, num_heads)
        self.num_layers = int(num_layers)
        self.use_se3_chirality = bool(use_se3_chirality)
        self.use_coulomb_bias = bool(use_coulomb_bias)
        self.num_perceiver_tokens = int(num_perceiver_tokens)

        self.geo_bias = GeoTransformerPairTripletBias(
            num_heads=self.num_heads,
            num_rbf=num_rbf,
            num_angle_bins=8,
            cutoff=cutoff,
            smooth_delta=smooth_delta,
            use_triplet_angles=True,
            use_ppf=True,
            use_se3_chirality=self.use_se3_chirality,
        )

        self.coulomb_bias: ScreenedCoulombBias | None
        if self.use_coulomb_bias:
            self.coulomb_bias = ScreenedCoulombBias(
                scalar_dim=self.scalar_dim,
                num_heads=self.num_heads,
                r0=1.0,
                debye_len=8.0,
            )
        else:
            self.coulomb_bias = None

        self.axial_a: VectorNeuronLinear | None
        self.axial_b: VectorNeuronLinear | None
        if self.use_se3_chirality:
            ax_dim = min(self.vector_dim, 8)
            self.axial_a = VectorNeuronLinear(self.vector_dim, ax_dim)
            self.axial_b = VectorNeuronLinear(self.vector_dim, ax_dim)
        else:
            self.axial_a = None
            self.axial_b = None

        self.layers = nn.ModuleList(
            [
                _SubPocketGlobalLayer(
                    scalar_dim=self.scalar_dim,
                    vector_dim=self.vector_dim,
                    num_heads=self.num_heads,
                    use_se3_chirality=self.use_se3_chirality,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Optional Q=16 Perceiver bridge cross-attention aligned to PocketPerceiverResampler.
        self.bridge_queries = nn.Parameter(
            torch.randn(1, self.num_perceiver_tokens, self.scalar_dim, dtype=torch.float32) * 0.02
        )
        self.bridge_norm_q = RMSNorm(self.scalar_dim, eps=eps)
        self.bridge_norm_kv = RMSNorm(self.scalar_dim, eps=eps)
        self.bridge_q = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.bridge_k = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.bridge_v = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.bridge_out = nn.Linear(self.scalar_dim, self.scalar_dim, bias=False)
        self.bridge_final_norm = RMSNorm(self.scalar_dim, eps=eps)

    def forward(
        self,
        sub_s: Tensor,
        sub_v: Tensor,
        sub_coords: Tensor,
        sub_mask: Tensor,
        *,
        sub_charge_prior: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run ``M x M`` geometric global self-attention over sub-pocket super-tokens."""
        mask_bool = sub_mask.to(torch.bool)
        if self.use_se3_chirality:
            assert self.axial_a is not None and self.axial_b is not None
            axial_vecs: Tensor | None = chiral_cross_product(
                self.axial_a(sub_v),
                self.axial_b(sub_v),
                normalize=True,
                eps=1.0,
            )
        else:
            axial_vecs = None

        bias: Tensor = self.geo_bias(
            sub_coords,
            mask_i=mask_bool,
            vecs_i=sub_v,
            axial_i=axial_vecs,
        )

        if self.coulomb_bias is not None:
            delta_q = torch.tanh(self.coulomb_bias.charge_mlp(sub_s).squeeze(-1))
            if sub_charge_prior is not None:
                charges = sub_charge_prior.to(torch.float32) + delta_q
            else:
                charges = delta_q
            coulomb: Tensor = self.coulomb_bias(
                sub_coords,
                charges_i=charges,
                mask_i=mask_bool,
            )
            bias = bias + coulomb

        cur_s = sub_s
        cur_v = sub_v
        for layer in self.layers:
            cur_s, cur_v = layer(cur_s, cur_v, mask_bool, bias)
        return cur_s, cur_v

    def perceiver_bridge_tokens(
        self,
        sub_tokens: Tensor,
        sub_mask: Tensor | None = None,
    ) -> Tensor:
        """Project ``M`` sub-pocket tokens ``[B, M, D]`` to ``Q=16`` Perceiver tokens."""
        sub_tokens_f = sub_tokens.to(torch.float32)
        batch_size, num_slots, dim = sub_tokens_f.shape
        if dim != self.scalar_dim:
            raise ValueError(f"Expected sub_tokens dim {self.scalar_dim}, got {dim}")
        if sub_mask is None:
            mask_bool = torch.ones(
                (batch_size, num_slots), device=sub_tokens_f.device, dtype=torch.bool
            )
        else:
            mask_bool = sub_mask.to(torch.bool)

        head_dim = self.scalar_dim // self.num_heads
        q_in = self.bridge_norm_q(self.bridge_queries.expand(batch_size, -1, -1))
        kv_in = torch.where(
            mask_bool.unsqueeze(-1),
            self.bridge_norm_kv(sub_tokens_f),
            torch.zeros_like(sub_tokens_f),
        )

        q = (
            self.bridge_q(q_in)
            .view(batch_size, self.num_perceiver_tokens, self.num_heads, head_dim)
            .transpose(1, 2)
        )
        k = (
            self.bridge_k(kv_in)
            .view(batch_size, num_slots, self.num_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            self.bridge_v(kv_in)
            .view(batch_size, num_slots, self.num_heads, head_dim)
            .transpose(1, 2)
        )

        scale = 1.0 / math.sqrt(float(head_dim))
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H, Q, M]
        probs = safe_masked_softmax(scores, mask_bool[:, None, None, :], dim=-1)
        ctx = (
            torch.matmul(probs, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_perceiver_tokens, self.scalar_dim)
        )
        out: Tensor = self.bridge_final_norm(
            self.bridge_queries.expand(batch_size, -1, -1) + self.bridge_out(ctx)
        )
        has_valid = mask_bool.any(dim=1).view(batch_size, 1, 1)
        return torch.where(has_valid, out, torch.zeros_like(out))


class _SubPocketVNStage(nn.Module):
    """One hierarchical SubPocketVN-JEPA stage (local chiral VN + optional global sub-pockets)."""

    use_global_subpockets: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        num_heads: int = 8,
        num_rbf: int = 24,
        num_subpockets: int = 24,
        num_salient_slots: int = 8,
        global_heads: int = 4,
        global_cutoff: float = 20.0,
        smooth_delta: float = 0.5,
        use_se3_chirality: bool = True,
        use_coulomb_bias: bool = True,
        num_perceiver_tokens: int = 16,
        use_global_subpockets: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_global_subpockets = bool(use_global_subpockets)
        self.local_attn = ChiralLocalVNAttention(
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            num_heads=num_heads,
            num_rbf=num_rbf,
            use_se3_chirality=use_se3_chirality,
            dropout=dropout,
        )
        self.vn_ffn = ChiralVNSwiGLU(
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            use_se3_chirality=use_se3_chirality,
            dropout=dropout,
        )
        self.slot_pool: CompetitiveSubPocketSlotPool | None
        self.global_transformer: SubPocketGlobalTransformer | None
        if self.use_global_subpockets:
            self.slot_pool = CompetitiveSubPocketSlotPool(
                scalar_dim=scalar_dim,
                vector_dim=vector_dim,
                num_subpockets=num_subpockets,
                num_salient_slots=num_salient_slots,
                smooth_delta=smooth_delta,
            )
            self.global_transformer = SubPocketGlobalTransformer(
                scalar_dim=scalar_dim,
                vector_dim=vector_dim,
                num_heads=global_heads,
                num_layers=1,
                num_rbf=min(16, num_rbf),
                cutoff=global_cutoff,
                smooth_delta=smooth_delta,
                use_se3_chirality=use_se3_chirality,
                use_coulomb_bias=use_coulomb_bias,
                num_perceiver_tokens=num_perceiver_tokens,
                dropout=dropout,
            )
        else:
            self.slot_pool = None
            self.global_transformer = None

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        graph: DenseKNNGraph,
        rbf_env: Tensor,
        atom_mask: Tensor,
        *,
        flat_nbr_idx: Tensor,
        salience_indicator: Tensor,
        charge_prior: Tensor,
    ) -> tuple[
        Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None
    ]:
        """Execute local chiral VN attention, atom FFN, and optional sub-pocket global pass."""
        s, v = self.local_attn(s, v, graph, rbf_env, atom_mask, flat_nbr_idx=flat_nbr_idx)
        s, v = self.vn_ffn(s, v, atom_mask)
        if not self.use_global_subpockets:
            return s, v, None, None, None, None, None

        assert self.slot_pool is not None and self.global_transformer is not None
        (
            sub_s,
            sub_v,
            sub_coords,
            slot_weights,
            unpool_weights,
            sub_mask,
            sub_q_prior,
        ) = self.slot_pool.pool(
            s,
            v,
            graph.centered_coords,
            atom_mask,
            salience_indicator=salience_indicator,
            charge_prior=charge_prior,
        )
        sub_s, sub_v = self.global_transformer(
            sub_s,
            sub_v,
            sub_coords,
            sub_mask,
            sub_charge_prior=sub_q_prior,
        )
        s, v = self.slot_pool.unpool(s, v, sub_s, sub_v, unpool_weights, atom_mask)
        return s, v, sub_s, sub_v, sub_coords, slot_weights, sub_mask


class SubPocketVNJEPAEncoder(nn.Module):
    """Competitive Multi-Scale Chiral Vector-Neuron Sub-Pocket Transformer (Method 3).

    Satisfies :class:`~geneuclidean.protocols.PocketEncoderProtocol` via
    ``forward(coords, node_feats, atom_mask) -> PocketEncoderOutput``, and additionally provides:
      - :meth:`encode_with_subpockets`: returns atom embeddings plus ``M = 24`` equivariant
        sub-pocket super-tokens, sub-pocket centroids, and Dora slot weights.
      - :meth:`perceiver_bridge_tokens`: emits ``[B, 16, out_dim]`` soft-prefix tokens aligned
        directly to :class:`~geneuclidean.models.llm_bridge.PocketPerceiverResampler`.
    """

    node_in_dim: int
    hidden_dim: int
    vector_dim: int
    out_dim: int
    num_layers: int
    num_rbf: int
    cutoff: float
    max_neighbors: int
    num_heads: int
    num_subpockets: int
    num_salient_slots: int
    num_perceiver_tokens: int
    use_se3_chirality: bool
    use_coulomb_bias: bool

    def __init__(
        self,
        node_in_dim: int = 52,
        hidden_dim: int = 192,
        vector_dim: int = 64,
        out_dim: int = 256,
        num_layers: int = 4,
        num_rbf: int = 24,
        cutoff: float = 5.5,
        max_neighbors: int = 24,
        num_heads: int = 8,
        num_subpockets: int = 24,
        num_salient_slots: int = 8,
        global_heads: int = 4,
        global_cutoff: float = 20.0,
        smooth_delta: float = 0.5,
        use_se3_chirality: bool = True,
        use_coulomb_bias: bool = True,
        num_perceiver_tokens: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_subpockets < 2:
            raise ValueError(f"num_subpockets must be >= 2, got {num_subpockets}")

        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.num_heads = _resolve_divisible_heads(self.hidden_dim, num_heads)
        self.num_subpockets = int(num_subpockets)
        self.num_salient_slots = max(
            0, min(int(num_salient_slots), max(1, self.num_subpockets // 2))
        )
        self.num_perceiver_tokens = int(num_perceiver_tokens)
        self.use_se3_chirality = bool(use_se3_chirality)
        self.use_coulomb_bias = bool(use_coulomb_bias)

        self.node_embed = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            RMSNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.rbf = BesselRBF(num_rbf=self.num_rbf, cutoff=self.cutoff)
        # Equivariant Layer-0 vector initialization from local tie-safe edge geometry.
        self.init_vec_rbf = nn.Linear(self.num_rbf, self.vector_dim, bias=False)
        self.init_vec_norm = EquivariantRMSNorm(self.vector_dim)

        # Place the M=24 competitive global sub-pocket transformer at the hierarchical midpoint
        # bottleneck (propagates 20 A triplet/PPF + LES Coulomb context into subsequent local VN
        # stages without duplicating M^3 triplet overhead across multiple layers).
        self.global_stage_idx: int = (self.num_layers - 1) // 2
        self.stages = nn.ModuleList(
            [
                _SubPocketVNStage(
                    scalar_dim=self.hidden_dim,
                    vector_dim=self.vector_dim,
                    num_heads=self.num_heads,
                    num_rbf=self.num_rbf,
                    num_subpockets=self.num_subpockets,
                    num_salient_slots=self.num_salient_slots,
                    global_heads=global_heads,
                    global_cutoff=global_cutoff,
                    smooth_delta=smooth_delta,
                    use_se3_chirality=self.use_se3_chirality,
                    use_coulomb_bias=self.use_coulomb_bias,
                    num_perceiver_tokens=self.num_perceiver_tokens,
                    use_global_subpockets=(layer_idx == self.global_stage_idx),
                    dropout=dropout,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.out_proj = nn.Sequential(
            RMSNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.subpocket_out_proj = nn.Sequential(
            RMSNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.perceiver_out_proj = nn.Sequential(
            RMSNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
            RMSNorm(self.out_dim),
        )
        self.pool = MaskedAttentionPool(in_dim=self.out_dim, out_dim=self.out_dim)
        self.vec_out_norm = EquivariantRMSNorm(self.vector_dim)
        self._init_residual_projections()

    def _init_residual_projections(self) -> None:
        """Apply Pre-LN residual scaling so multi-layer activations stay well-conditioned."""
        scale = 1.0 / math.sqrt(float(max(2, 2 * self.num_layers)))
        with torch.no_grad():
            for stage in self.stages:
                assert isinstance(stage, _SubPocketVNStage)
                stage.local_attn.scalar_out.weight.mul_(scale)
                stage.local_attn.rbf_val_out.weight.mul_(scale)
                stage.local_attn.vn_out.linear.weight.mul_(scale)
                stage.vn_ffn.swiglu.w2.weight.mul_(scale)
                stage.vn_ffn.vn_out.linear.weight.mul_(scale)
                if stage.global_transformer is not None:
                    for g_layer in stage.global_transformer.layers:
                        assert isinstance(g_layer, _SubPocketGlobalLayer)
                        g_layer.out_s.weight.mul_(scale)
                        g_layer.ffn.w2.weight.mul_(scale)
                        g_layer.vn_block.out_proj.linear.weight.mul_(scale)
                        g_layer.vn_block.ffn_out.linear.weight.mul_(scale)
                        if g_layer.vn_block.chiral_scale is not None:
                            g_layer.vn_block.chiral_scale.mul_(scale)

    def _run_backbone(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> tuple[SubPocketEncodingOutput, Tensor]:
        """Run the full multi-scale encoder and also return hidden-dim sub-pocket tokens."""
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
        if node_feats.ndim != 3 or node_feats.shape[-1] != self.node_in_dim:
            raise ValueError(
                f"Expected node_feats shape [B, N, {self.node_in_dim}], "
                f"got {tuple(node_feats.shape)}"
            )
        if atom_mask.shape != coords.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} != coords[:2] {tuple(coords.shape[:2])}"
            )

        batch_size, num_atoms, _ = coords.shape
        device = coords.device
        mask_bool = atom_mask.to(torch.bool)
        mask_s = mask_bool.unsqueeze(-1)
        mask_v = mask_s.unsqueeze(-1)

        graph = build_dense_knn_graph(
            coords=coords,
            atom_mask=mask_bool,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            adaptive_cutoff=True,
        )

        feats_f = node_feats.to(torch.float32)
        safe_feats = torch.where(mask_s, feats_f, torch.zeros_like(feats_f))
        s: Tensor = self.node_embed(safe_feats)
        s = torch.where(mask_s, s, torch.zeros_like(s))

        k_nbr = graph.nbr_idx.shape[2]
        flat_nbr_idx = _build_flat_neighbor_index(graph.nbr_idx)
        if batch_size == 0 or num_atoms == 0 or k_nbr == 0:
            rbf_env = torch.empty(
                (batch_size, num_atoms, k_nbr, self.num_rbf), device=device, dtype=torch.float32
            )
            v = torch.zeros(
                (batch_size, num_atoms, self.vector_dim, 3), device=device, dtype=torch.float32
            )
        else:
            # Divide by Bessel frequencies so every radial mode has O(1) amplitude rather than
            # amplifying high-frequency fp32 coordinate ULP noise under large rigid translations.
            inv_freq = (1.0 / self.rbf.frequencies.clamp(min=1.0)).view(1, 1, 1, self.num_rbf)
            rbf_env = self.rbf(graph.edge_dist) * inv_freq * graph.edge_env.unsqueeze(-1)
            rbf_env = torch.where(graph.nbr_mask.unsqueeze(-1), rbf_env, torch.zeros_like(rbf_env))
            init_weights: Tensor = self.init_vec_rbf(rbf_env)  # [B, N, K, C_v]
            norm_k = 1.0 / math.sqrt(float(max(1, k_nbr)))
            bn = batch_size * num_atoms
            v = (
                torch.bmm(
                    init_weights.reshape(bn, k_nbr, self.vector_dim).transpose(1, 2),
                    graph.edge_dir.reshape(bn, k_nbr, 3),
                ).reshape(batch_size, num_atoms, self.vector_dim, 3)
                * norm_k
            )
            v = torch.where(mask_v, self.init_vec_norm(v), torch.zeros_like(v))

        salience = _extract_salience_indicator(safe_feats, mask_bool)
        charge_prior = _extract_charge_prior(safe_feats, mask_bool)

        sub_s = torch.zeros(
            (batch_size, self.num_subpockets, self.hidden_dim),
            device=device,
            dtype=torch.float32,
        )
        sub_v = torch.zeros(
            (batch_size, self.num_subpockets, self.vector_dim, 3),
            device=device,
            dtype=torch.float32,
        )
        sub_coords = torch.zeros(
            (batch_size, self.num_subpockets, 3),
            device=device,
            dtype=torch.float32,
        )
        slot_weights = torch.zeros(
            (batch_size, self.num_subpockets, num_atoms),
            device=device,
            dtype=torch.float32,
        )
        sub_mask = mask_bool.any(dim=1).unsqueeze(1).expand(batch_size, self.num_subpockets)

        for stage in self.stages:
            s, v, st_sub_s, st_sub_v, st_sub_c, st_slot_w, st_sub_m = stage(
                s,
                v,
                graph,
                rbf_env,
                mask_bool,
                flat_nbr_idx=flat_nbr_idx,
                salience_indicator=salience,
                charge_prior=charge_prior,
            )
            if (
                st_sub_s is not None
                and st_sub_v is not None
                and st_sub_c is not None
                and st_slot_w is not None
                and st_sub_m is not None
            ):
                sub_s = st_sub_s
                sub_v = st_sub_v
                sub_coords = st_sub_c
                slot_weights = st_slot_w
                sub_mask = st_sub_m

        node_emb: Tensor = self.out_proj(s)
        node_emb = torch.where(mask_s, node_emb, torch.zeros_like(node_emb))
        pooled_emb: Tensor = self.pool(node_emb, mask_bool)
        vec_emb: Tensor = torch.where(mask_v, self.vec_out_norm(v), torch.zeros_like(v))

        sub_mask_1 = sub_mask.unsqueeze(-1)
        sub_tokens: Tensor = self.subpocket_out_proj(sub_s)
        sub_tokens = torch.where(sub_mask_1, sub_tokens, torch.zeros_like(sub_tokens))
        sub_vec_out = torch.where(
            sub_mask_1.unsqueeze(-1), self.vec_out_norm(sub_v), torch.zeros_like(sub_v)
        )

        enc_out = PocketEncoderOutput(
            node_emb=node_emb,
            pooled_emb=pooled_emb,
            vec_emb=vec_emb,
        )
        full_out = SubPocketEncodingOutput(
            encoder_output=enc_out,
            subpocket_tokens=sub_tokens,
            subpocket_coords=sub_coords,
            subpocket_vec=sub_vec_out,
            slot_weights=slot_weights,
            subpocket_mask=sub_mask,
            centroid_offset=graph.centroid_offset,
        )
        return full_out, sub_s

    def encode_with_subpockets(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> SubPocketEncodingOutput:
        """Encode pockets and return atom outputs together with ``M`` sub-pocket super-tokens."""
        full_out, _ = self._run_backbone(coords, node_feats, atom_mask)
        return full_out

    def perceiver_bridge_tokens(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """Emit ``[B, 16, out_dim]`` soft-prefix tokens aligned to ``PocketPerceiverResampler``."""
        full_out, sub_s_hidden = self._run_backbone(coords, node_feats, atom_mask)
        global_stage = self.stages[self.global_stage_idx]
        assert isinstance(global_stage, _SubPocketVNStage)
        assert global_stage.global_transformer is not None
        bridge_hidden = global_stage.global_transformer.perceiver_bridge_tokens(
            sub_s_hidden,
            sub_mask=full_out.subpocket_mask,
        )
        projected: Tensor = self.perceiver_out_proj(bridge_hidden)
        has_valid = full_out.subpocket_mask.any(dim=1).view(-1, 1, 1)
        return torch.where(has_valid, projected, torch.zeros_like(projected))

    def forward(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode batched 3D protein pockets (implements ``PocketEncoderProtocol``)."""
        full_out, _ = self._run_backbone(coords, node_feats, atom_mask)
        return full_out.encoder_output


def _load_jepa_ssl_checkpoint_into_encoder(
    encoder: nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load encoder weights from a ``SubPocketJEPAPretrainer`` checkpoint."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"JEPA SSL checkpoint not found: {path}")
    try:
        payload: Any = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        payload = torch.load(path, map_location=map_location, weights_only=False)

    raw_sd: Mapping[str, Any]
    if isinstance(payload, Mapping):
        if "encoder_state_dict" in payload and isinstance(payload["encoder_state_dict"], Mapping):
            raw_sd = payload["encoder_state_dict"]
        elif "teacher_state_dict" in payload and isinstance(payload["teacher_state_dict"], Mapping):
            raw_sd = payload["teacher_state_dict"]
        elif "student_state_dict" in payload and isinstance(payload["student_state_dict"], Mapping):
            raw_sd = payload["student_state_dict"]
        elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
            raw_sd = payload["model_state_dict"]
        else:
            raw_sd = payload
    else:
        raise ValueError(f"Unsupported JEPA checkpoint payload type: {type(payload).__name__}")

    cleaned: dict[str, Tensor] = {}
    for key, val in raw_sd.items():
        if not isinstance(val, Tensor):
            continue
        clean_k = str(key)
        for prefix in ("module.", "teacher.", "student.", "encoder."):
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix) :]
        cleaned[clean_k] = val.to(dtype=torch.float32)

    encoder.load_state_dict(cleaned, strict=True)


def build_subpocket_vn_jepa(
    config: EncoderConfig,
    max_neighbors: int = 24,
) -> SubPocketVNJEPAEncoder:
    """Instantiate a :class:`SubPocketVNJEPAEncoder` from a validated ``EncoderConfig``.

    Automatically loads pretrained SSL encoder weights when ``config.jepa_ssl_ckpt`` is set.
    """
    num_salient = min(8, max(1, int(config.num_subpockets) // 3))
    if int(config.num_subpockets) == 24:
        num_salient = 8
    encoder = SubPocketVNJEPAEncoder(
        node_in_dim=config.node_in_dim,
        hidden_dim=config.hidden_dim,
        vector_dim=config.vector_dim,
        out_dim=config.out_dim,
        num_layers=config.num_layers,
        num_rbf=config.num_rbf,
        cutoff=config.cutoff,
        max_neighbors=int(max_neighbors),
        num_heads=config.num_heads,
        num_subpockets=config.num_subpockets,
        num_salient_slots=num_salient,
        use_se3_chirality=config.use_se3_chirality,
        use_coulomb_bias=config.use_coulomb_bias,
        dropout=config.dropout,
    )
    if config.jepa_ssl_ckpt:
        _load_jepa_ssl_checkpoint_into_encoder(encoder, config.jepa_ssl_ckpt)
    return encoder


__all__ = [
    "ChiralLocalVNAttention",
    "ChiralVNSwiGLU",
    "CompetitiveSubPocketSlotPool",
    "SubPocketEncodingOutput",
    "SubPocketGlobalTransformer",
    "SubPocketVNJEPAEncoder",
    "build_subpocket_vn_jepa",
    "smooth_pair_directions",
]
