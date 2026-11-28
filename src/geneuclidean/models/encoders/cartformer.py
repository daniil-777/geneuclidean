"""CartFormer-SE3: Chiral Cartesian-Tensor Geometric Transformer for 3D Atomic Systems.

Implements the peer-reviewed v2 ``CartFormer-SE3`` architecture (GotenNet ICLR'25 ×
TensorNet NeurIPS'23 × GeoTransformer TPAMI'24 × EScAIP NeurIPS'24 × FastEGNN ICML'24 ×
Latent Ewald Summation *npj Comput. Mater.*'25) strictly in pure-PyTorch ``torch.float32``
for maximum Apple Silicon MPS, CUDA, and CPU performance with **zero ``e3nn`` Clebsch-Gordan
products** and **zero sparse ``index_add_`` scatter contention**.

Architectural highlights (see ``docs/pointcloud_cv_3d_atom_encoding_plan.md`` §4.1):
  1. **Zero-``e3nn`` Complete ``l <= 2`` Cartesian Rank-2 Track:** Maintains per-atom
     ``3 x 3`` rank-2 Cartesian tensors :math:`\\mathbf{X}_i \\in \\mathbb{R}^{C_t \\times 3
     \\times 3}` decomposed via :func:`~geneuclidean.models.encoders.tensor_ops.decompose_rank2_ias`
     into Trace :math:`I` (``0e``), Antisymmetric :math:`A` (``1e`` axial / Hodge dual
     :math:`\\boldsymbol{\\omega} = \\star A`), and Symmetric Traceless :math:`S` (``2e``
     quadrupole). Both :math:`S` and :math:`A` are active immediately at Layer 1 via
     chemically modulated Layer-0 symmetric traceless and antisymmetric
     ``(W_v2t v_j) r_hat^T - r_hat (W_v2t v_j)^T`` initialization.
  2. **Tie-Safe Singularity-Free Dense GATA-SE3 Attention:** Pre-projects directional
     vector, source/target quadrupole (:math:`\\hat{\\mathbf{r}}_{ij}^\\top \\mathbf{S}_i
     \\hat{\\mathbf{r}}_{ij}` and :math:`\\hat{\\mathbf{r}}_{ij}^\\top \\mathbf{S}_j
     \\hat{\\mathbf{r}}_{ij}` via exact 5-component traceless packing), and optional chiral
     axial (:math:`\\langle \\boldsymbol{\\omega}_i - \\boldsymbol{\\omega}_j,
     \\hat{\\mathbf{r}}_{ij}\\rangle`) couplings directly into head space before a single
     row-vectorized ``index_select`` gather + zero-permute ``torch.bmm`` aggregation. Uses
     **NO ``+ log f_c`` softmax singularity** — smooth per-atom adaptive cutoff weights
     :math:`f_c(d_{ij}; r_{\\mathrm{eff},i})` modulate attention post-softmax via
     :func:`~geneuclidean.models.encoders.dense_knn.safe_masked_softmax` for exact
     equidistant ``K``-th neighbor tie-swap invariance.
  3. **Metal-Native ``3 x 3`` Tensor Product FFN:** Executes complete ``l in {0, 1, 2}``
     bilinear tensor coupling via native ``3 x 3`` matrix multiplication (27 FLOPs/channel),
     explicit bias-free ``W_v2t`` (``vector_dim -> tensor_dim``) and ``W_t2v``
     (``tensor_dim -> vector_dim``) projections, first-order + product bounded chiral
     pseudoscalars, and explicit residual updates normalized by ``LayerNorm``,
     :class:`EquivariantRMSNorm`, and :class:`TensorEquivariantRMSNorm`.
  4. **Global Equivariant Virtual Landmarks (`M=8`):** Pools valid atoms smoothly into
     equivariant global landmarks communicating via ``M x M`` attention biased by
     :class:`~geneuclidean.models.encoders.electrostatics.GeoTransformerPairTripletBias`
     (pairwise RBF + triplet angles + signed RoITr PPF) and optional
     :class:`~geneuclidean.models.encoders.electrostatics.ScreenedCoulombBias` (``LES``
     long-range screened ``1/r`` electrostatics anchored to ``node_feats[..., 46]``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

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
    TensorEquivariantRMSNorm,
    VectorNeuronLinear,
    chiral_cross_product,
    decompose_rank2_ias,
    outer_traceless,
    skew_matrix_to_vector,
    vector_to_skew_matrix,
)

if TYPE_CHECKING:
    from geneuclidean.config import EncoderConfig


def _build_flat_nbr_idx(nbr_idx: Tensor) -> Tensor:
    """Convert ``[B, N, K]`` local indices into flat ``[B * N * K]`` ``index_select`` indices."""
    batch_size, num_atoms, _ = nbr_idx.shape
    offsets = torch.arange(batch_size, device=nbr_idx.device, dtype=torch.long) * num_atoms
    return (nbr_idx + offsets.view(batch_size, 1, 1)).reshape(-1)


def _fast_gather_neighbors(
    x: Tensor,
    flat_nbr_idx: Tensor,
    batch_size: int,
    num_atoms: int,
    k_eff: int,
) -> Tensor:
    """Gather ``x[b, nbr_idx[b, i, k]]`` via row-vectorized ``index_select`` (MPS-fast)."""
    feat_shape = x.shape[2:]
    if batch_size == 0 or num_atoms == 0 or k_eff == 0:
        return x.new_empty((batch_size, num_atoms, k_eff, *feat_shape))
    flat = x.reshape(batch_size * num_atoms, -1)
    gathered = flat.index_select(0, flat_nbr_idx)
    return gathered.reshape(batch_size, num_atoms, k_eff, *feat_shape)


def _resolve_divisible_heads(requested: int, *dims: int) -> int:
    """Return the largest head count ``<= requested`` that divides every dimension in ``dims``."""
    limit = max(1, int(requested))
    for h in range(limit, 0, -1):
        if all(d % h == 0 for d in dims):
            return h
    return 1


def _extract_sym_traceless_5(tensor: Tensor) -> Tensor:
    """Extract the 5 independent symmetric traceless (``2e``) entries directly from ``[...,3,3]``.

    Equivalent to ``_pack_sym_traceless_5(decompose_rank2_ias(tensor)[2])`` without allocating
    three intermediate ``[..., 3, 3]`` matrices.
    """
    tr3 = (tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]) * (1.0 / 3.0)
    return torch.stack(
        (
            tensor[..., 0, 0] - tr3,
            tensor[..., 1, 1] - tr3,
            0.5 * (tensor[..., 0, 1] + tensor[..., 1, 0]),
            0.5 * (tensor[..., 0, 2] + tensor[..., 2, 0]),
            0.5 * (tensor[..., 1, 2] + tensor[..., 2, 1]),
        ),
        dim=-1,
    )


def _quadrupole_basis_5(edge_dir: Tensor) -> Tensor:
    """Evaluate the 5 dual quadratic basis functions matching :func:`_extract_sym_traceless_5`."""
    rx = edge_dir[..., 0]
    ry = edge_dir[..., 1]
    rz = edge_dir[..., 2]
    rz2 = rz * rz
    return torch.stack(
        (
            rx * rx - rz2,
            ry * ry - rz2,
            2.0 * rx * ry,
            2.0 * rx * rz,
            2.0 * ry * rz,
        ),
        dim=-1,
    )


class CartesianTensorLinear(nn.Module):
    """Bias-free ``O(3)``-equivariant linear mixing over rank-2 tensors ``[..., C, 3, 3]``."""

    in_channels: int
    out_channels: int

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"in_channels ({in_channels}) and out_channels ({out_channels}) must be >= 1"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, tensor: Tensor) -> Tensor:
        """Mix channel dim identically across all 9 spatial entries of ``[..., C_in, 3, 3]``."""
        if tensor.ndim < 3 or tensor.shape[-3:] != (self.in_channels, 3, 3):
            raise ValueError(
                f"Expected tensor trailing shape ({self.in_channels}, 3, 3), "
                f"got {tuple(tensor.shape)}"
            )
        lead_shape = tensor.shape[:-3]
        flat = tensor.reshape(-1, self.in_channels, 9)
        mixed = torch.matmul(self.linear.weight, flat)
        return mixed.reshape(*lead_shape, self.out_channels, 3, 3)


class CartFormerAttentionLayer(nn.Module):
    """Dense batched ``GATA-SE3`` local geometric attention over ``[B, N, K]`` neighborhoods.

    Uses exact pre-gather algebraic factorization into head space (``v_i, v_j, S_i, S_j,
    omega_i - omega_j``), multi-query scalar keys/values with full ``H``-head queries,
    equivariant message bottlenecks (``C_msg = 8``), a single compact row-vectorized
    ``index_select`` gather, and batched ``torch.bmm`` reductions.

    Smooth cutoff gating is applied strictly post-softmax (zero ``+ log f_c`` singularity)
    with tie-shell denominator cancellation for exact equidistant neighbor tie-swap invariance.
    """

    scalar_dim: int
    vector_dim: int
    tensor_dim: int
    msg_dim: int
    num_heads: int
    head_dim: int
    num_rbf: int
    use_se3_chirality: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        tensor_dim: int,
        num_heads: int = 8,
        num_rbf: int = 24,
        *,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1 or tensor_dim < 1 or num_heads < 1 or num_rbf < 1:
            raise ValueError("All CartFormerAttentionLayer dimensions must be >= 1.")
        if scalar_dim % num_heads != 0:
            raise ValueError(
                f"scalar_dim ({scalar_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.tensor_dim = int(tensor_dim)
        self.msg_dim = max(4, min(8, self.tensor_dim))
        self.num_heads = int(num_heads)
        self.head_dim = self.scalar_dim // self.num_heads
        self.num_rbf = int(num_rbf)
        self.use_se3_chirality = bool(use_se3_chirality)

        # Full H-head queries + compact multi-query K/V for lean O(NK) MPS bandwidth.
        self.q_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.kv_proj = nn.Linear(self.scalar_dim, 2 * self.head_dim)
        self.out_s_proj = nn.Linear(self.scalar_dim, self.scalar_dim)

        # Fused equivariant vector projection: (v_head_i [H], v_head_j [H], v_msg [C_msg]).
        self.v_geom_proj = VectorNeuronLinear(self.vector_dim, 2 * self.num_heads + self.msg_dim)
        self.w_m2v = VectorNeuronLinear(self.msg_dim, self.vector_dim)

        # Exact head-space projections for source & target quadrupoles S_i, S_j (5-dim packed).
        self.s_to_heads = nn.Linear(self.tensor_dim, 2 * self.num_heads, bias=False)
        self.om_to_heads: VectorNeuronLinear | None
        if self.use_se3_chirality:
            self.om_to_heads = VectorNeuronLinear(self.tensor_dim, self.num_heads)
        else:
            self.om_to_heads = None

        # Equivariant bottleneck for neighbor rank-2 tensor + antisymmetric 1e message passing.
        self.x_down_proj = CartesianTensorLinear(self.tensor_dim, self.msg_dim)
        self.x_up_proj = CartesianTensorLinear(self.msg_dim, self.tensor_dim)

        # Bias-free geometric attention & post-aggregation head-to-channel maps.
        self.rbf_bias = nn.Linear(self.num_rbf, self.num_heads, bias=False)
        self.head_to_edge = nn.Linear(self.num_heads, 2, bias=False)
        self.h_to_vr = VectorNeuronLinear(self.num_heads, self.vector_dim)
        self.h_to_ts = CartesianTensorLinear(self.num_heads, self.tensor_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        x_ten: Tensor,
        graph: DenseKNNGraph,
        rbf_env: Tensor,
        edge_outer_traceless_flat: Tensor,
        quad_basis_5: Tensor,
        atom_mask: Tensor,
        *,
        flat_nbr_idx: Tensor | None = None,
        geom_edge_i: Tensor | None = None,
        geom_edge_j: Tensor | None = None,
        ed_outer_12: Tensor | None = None,
        eff_mask: Tensor | None = None,
        env_max: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute dense equivariant attention deltas ``(ds, dv, dx_ten)``."""
        mask_bool = atom_mask.to(torch.bool)
        batch_size, num_atoms, _ = s.shape
        k_eff = graph.nbr_idx.shape[2]
        if batch_size == 0 or num_atoms == 0 or k_eff == 0:
            return torch.zeros_like(s), torch.zeros_like(v), torch.zeros_like(x_ten)

        bn = batch_size * num_atoms
        idx_flat = _build_flat_nbr_idx(graph.nbr_idx) if flat_nbr_idx is None else flat_nbr_idx
        edge_dir = graph.edge_dir  # [B, N, K, 3]
        edge_env = graph.edge_env  # [B, N, K]
        nbr_mask = graph.nbr_mask  # [B, N, K]

        if ed_outer_12 is None:
            outer_9 = edge_outer_traceless_flat.reshape(batch_size, num_atoms, k_eff, 9)
            ed_outer_12_bn = torch.cat((edge_dir, outer_9), dim=-1).reshape(bn, k_eff, 12)
        else:
            ed_outer_12_bn = ed_outer_12

        # 1. Direct extraction of S (2e) & A (1e) irreps + head-space pre-projections.
        s5_i = _extract_sym_traceless_5(x_ten)  # [B, N, C_t, 5]
        s_heads: Tensor = self.s_to_heads(s5_i.transpose(-1, -2)).transpose(-1, -2)
        sh_i, sh_j_src = torch.chunk(s_heads, 2, dim=-2)  # each [B, N, H, 5]

        v_geom: Tensor = self.v_geom_proj(v)  # [B, N, 2*H + C_msg, 3]
        vh_i, vh_j_src, v_msg = torch.split(
            v_geom, [self.num_heads, self.num_heads, self.msg_dim], dim=-2
        )

        x_low: Tensor = self.x_down_proj(x_ten)  # [B, N, C_msg, 3, 3]
        kv: Tensor = self.kv_proj(s)  # [B, N, 2*d_h]

        # Polar and axial terms both project onto the same edge direction, with
        # opposite parity at the neighbor. Combine them per atom before the gather.
        gi, gj = vh_i, vh_j_src
        if self.use_se3_chirality:
            assert self.om_to_heads is not None
            omega: Tensor = self.om_to_heads(skew_matrix_to_vector(x_ten))
            gi, gj = gi + omega, gj - omega
        geom_i_h = torch.cat((gi, sh_i), dim=-1)
        geom_j_h = torch.cat((gj, sh_j_src), dim=-1)
        if geom_edge_i is None or geom_edge_j is None:
            ge_i = ge_j = torch.cat((edge_dir, quad_basis_5), dim=-1)
        elif self.use_se3_chirality and geom_edge_i.size(-1) == 11:
            # Preserve custom, independently supplied axial bases on the public layer.
            geom_i_h = torch.cat((vh_i, sh_i, omega), dim=-1)
            geom_j_h = torch.cat((vh_j_src, sh_j_src, omega), dim=-1)
            ge_i, ge_j = geom_edge_i, geom_edge_j
        else:
            ge_i, ge_j = geom_edge_i[..., :8], geom_edge_j[..., :8]
        gdim = geom_i_h.size(-1)

        # Gather each neighbor once for the scalar, geometric and tensor messages.
        m3 = self.msg_dim * 3
        m9 = self.msg_dim * 9
        pack = torch.cat(
            (
                kv,
                geom_j_h.reshape(batch_size, num_atoms, self.num_heads * gdim),
                v_msg.reshape(batch_size, num_atoms, m3),
                x_low.reshape(batch_size, num_atoms, m9),
            ),
            dim=-1,
        )
        pj = _fast_gather_neighbors(pack, idx_flat, batch_size, num_atoms, k_eff)
        k_j, vs_j, geom_j_flat, vx_j_flat = torch.split(
            pj,
            [self.head_dim, self.head_dim, self.num_heads * gdim, m3 + m9],
            dim=-1,
        )

        # 3. Zero-permute BMM multi-head QK + exact fused directional geometric bias.
        q_bn = self.q_proj(s).reshape(bn, self.num_heads, self.head_dim)
        k_j_bn = k_j.reshape(bn, k_eff, self.head_dim)
        qk = torch.bmm(k_j_bn, q_bn.transpose(1, 2)).reshape(
            batch_size, num_atoms, k_eff, self.num_heads
        ) * (1.0 / math.sqrt(float(self.head_dim)))

        b_geo_i = torch.bmm(
            ge_i.reshape(bn, k_eff, gdim),
            geom_i_h.reshape(bn, self.num_heads, gdim).transpose(1, 2),
        ).reshape(batch_size, num_atoms, k_eff, self.num_heads)
        geom_j_5d = geom_j_flat.reshape(batch_size, num_atoms, k_eff, self.num_heads, gdim)
        b_geo_j = (geom_j_5d * ge_j.unsqueeze(-2)).sum(dim=-1)
        logits = qk + self.rbf_bias(rbf_env) + b_geo_i + b_geo_j

        # 4. Singularity-free smooth envelope attention (NO + log f_c!).
        eff_m = (nbr_mask & (edge_env > 0.0)).unsqueeze(-1) if eff_mask is None else eff_mask
        attn_probs = safe_masked_softmax(logits, eff_m, dim=2)  # [B, N, K, H]
        w_unnorm = attn_probs * edge_env.unsqueeze(-1)
        w_sum = w_unnorm.sum(dim=2, keepdim=True)
        e_max = (
            edge_env.max(dim=2, keepdim=True).values.unsqueeze(-1) if env_max is None else env_max
        )
        attn_weights = w_unnorm * (e_max / (w_sum + 1e-8))
        attn_weights = self.attn_dropout(attn_weights)  # [B, N, K, H]
        aw_h = attn_weights.reshape(bn, k_eff, self.num_heads).transpose(1, 2)  # [BN, H, K]

        # 5. Zero-permute BMM aggregations for scalars, polar vectors & rank-2 Cartesian tensors.
        ds_heads = torch.bmm(aw_h, vs_j.reshape(bn, k_eff, self.head_dim))  # [BN, H, d_h]
        ds: Tensor = self.out_s_proj(ds_heads.reshape(batch_size, num_atoms, self.scalar_dim))

        # Edge geometric couplings (dv_dir + dx_sym) via one [BN, H, K] @ [BN, K, 12] BMM.
        ed_agg = torch.bmm(aw_h, ed_outer_12_bn)  # [BN, H, 12]
        dv_dir: Tensor = self.h_to_vr(
            ed_agg[..., :3].reshape(batch_size, num_atoms, self.num_heads, 3)
        )
        dx_sym: Tensor = self.h_to_ts(
            ed_agg[..., 3:].reshape(batch_size, num_atoms, self.num_heads, 3, 3)
        )

        # Fused neighbor vector + tensor aggregation in ONE BMM + antisymmetric 1e coupling!
        w2 = self.head_to_edge(attn_weights).reshape(bn, k_eff, 2).transpose(1, 2)  # [BN, 2, K]
        vx_j_bn = vx_j_flat.reshape(bn, k_eff, m3 + m9)
        vx_agg = torch.bmm(w2[:, 0:1], vx_j_bn).squeeze(1)  # [BN, m3 + m9]
        dv_msg_flat, dx_low_flat = torch.split(vx_agg, [m3, m9], dim=-1)
        dv = self.w_m2v(dv_msg_flat.reshape(batch_size, num_atoms, self.msg_dim, 3)) + dv_dir

        w_anti_ed = w2[:, 1].unsqueeze(-1) * edge_dir.reshape(bn, k_eff, 3)  # [BN, K, 3]
        vm_j_bn = vx_j_bn[..., :m3]  # [BN, K, m3]
        vr_outer = torch.bmm(vm_j_bn.transpose(1, 2), w_anti_ed).reshape(
            batch_size, num_atoms, self.msg_dim, 3, 3
        )
        dx_low_total = dx_low_flat.reshape(batch_size, num_atoms, self.msg_dim, 3, 3) + (
            vr_outer - vr_outer.transpose(-1, -2)
        )
        dx_ten = self.x_up_proj(dx_low_total) + dx_sym

        mask_3d = mask_bool.unsqueeze(-1)
        mask_4d = mask_3d.unsqueeze(-1)
        mask_5d = mask_4d.unsqueeze(-1)
        ds = torch.where(mask_3d, ds, torch.zeros_like(ds))
        dv = torch.where(mask_4d, dv, torch.zeros_like(dv))
        dx_ten = torch.where(mask_5d, dx_ten, torch.zeros_like(dx_ten))
        return ds, dv, dx_ten


class CartFormerFFN(nn.Module):
    """Metal-native ``3 x 3`` Cartesian tensor-product FFN & residual state updater.

    Couples ``l = 0, 1, 2`` tensor channels via native ``3 x 3`` matrix multiplication
    :math:`\\tilde{\\mathbf{M}}_i = \\mathrm{TensorEqRMSNorm}((\\mathbf{W}_1\\mathbf{X}_i)
    (\\mathbf{W}_2\\mathbf{X}_i))`, extracts Frobenius norms plus first-order and product
    chiral pseudoscalars, applies explicit bias-free ``W_v2t`` / ``W_t2v`` projections for
    rank-2-to-vector contractions, and updates ``(s, v, X)`` with residual norms.

    Routing the axial cross-product ``v_cross`` into the antisymmetric ``1e`` subspace of
    ``X`` via :func:`vector_to_skew_matrix` preserves strict ``1o`` polar parity on ``v``
    and ``1e`` axial parity on ``A(X)`` across all layers, guaranteeing exact physical
    achirality (``chi == 0``) on planar aromatic rings and collinear molecules at every depth.
    """

    scalar_dim: int
    vector_dim: int
    tensor_dim: int
    use_se3_chirality: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        tensor_dim: int,
        *,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1 or tensor_dim < 1:
            raise ValueError("scalar_dim, vector_dim, and tensor_dim must be >= 1.")
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.tensor_dim = int(tensor_dim)
        self.use_se3_chirality = bool(use_se3_chirality)

        # Fused bias-free tensor channel projection emitting (Y, Z) in one GEMM.
        self.ten_yz_proj = CartesianTensorLinear(self.tensor_dim, 2 * self.tensor_dim)
        self.prod_norm = TensorEquivariantRMSNorm(self.tensor_dim, eps=1e-4)

        # Fused equivariant vector projection emitting (u, v, W_v2t[, c1_t, c2_t]) in one GEMM.
        vec_out_ch = 2 * self.vector_dim + (3 if self.use_se3_chirality else 1) * self.tensor_dim
        self.vec_all_proj = VectorNeuronLinear(self.vector_dim, vec_out_ch)
        self.w_t2v = VectorNeuronLinear(self.tensor_dim, self.vector_dim)

        if self.use_se3_chirality:
            # s + [tr_m, norm_a, norm_s, chi_x, chi_m, chi_v] (6*C_t) + [v_norm] (C_v)
            mlp_in_dim = self.scalar_dim + 6 * self.tensor_dim + self.vector_dim
            mlp_out_dim = self.scalar_dim + 3 * self.vector_dim + 2 * self.tensor_dim
        else:
            mlp_in_dim = self.scalar_dim + 3 * self.tensor_dim + self.vector_dim
            mlp_out_dim = self.scalar_dim + 3 * self.vector_dim + self.tensor_dim

        self.update_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, mlp_out_dim),
        )
        self.dot_to_scalar = nn.Linear(self.vector_dim, self.scalar_dim)
        self.scalar_norm = nn.LayerNorm(self.scalar_dim)
        self.vec_norm = EquivariantRMSNorm(self.vector_dim, eps=1e-4)
        self.tensor_norm = TensorEquivariantRMSNorm(self.tensor_dim, eps=1e-4)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        x_ten: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply ``3 x 3`` tensor product mixing and residual norms to ``(s, v, x_ten)``."""
        mask_bool = atom_mask.to(torch.bool)
        mask_3d = mask_bool.unsqueeze(-1)
        mask_4d = mask_3d.unsqueeze(-1)
        mask_5d = mask_4d.unsqueeze(-1)

        # 1. Metal-native 3x3 Cartesian tensor product (27 FLOPs/channel, zero e3nn).
        yz: Tensor = self.ten_yz_proj(x_ten)
        y_ten, z_ten = torch.chunk(yz, 2, dim=-3)
        m_tilde: Tensor = self.prod_norm(torch.matmul(y_ten, z_ten))
        m_i, m_a, m_s = decompose_rank2_ias(m_tilde)

        # 2. Bounded O(3)-invariant scalar contractions & fused vector projections.
        tr_m = m_i[..., 0, 0] + m_i[..., 1, 1] + m_i[..., 2, 2]  # [B, N, C_t]
        norm_a = torch.sqrt((m_a * m_a).sum(dim=(-2, -1)) + 1e-8)  # [B, N, C_t]
        norm_s = torch.sqrt((m_s * m_s).sum(dim=(-2, -1)) + 1e-8)  # [B, N, C_t]

        vec_all: Tensor = self.vec_all_proj(v)
        if self.use_se3_chirality:
            u_v, v_v, v_t, c1_t, c2_t = torch.split(
                vec_all,
                [
                    self.vector_dim,
                    self.vector_dim,
                    self.tensor_dim,
                    self.tensor_dim,
                    self.tensor_dim,
                ],
                dim=-2,
            )
            v_norm = torch.sqrt((v_v * v_v).sum(dim=-1) + 1e-8)
            uv_dot = (u_v * v_v).sum(dim=-1)

            # First-order axial chirality from A(X), product chirality from A(M), & v x v on C_t.
            raw_om_x = skew_matrix_to_vector(x_ten)
            om_x = raw_om_x * torch.rsqrt((raw_om_x * raw_om_x).sum(dim=-1, keepdim=True) + 1.0)
            raw_om_m = skew_matrix_to_vector(m_a)
            om_m = raw_om_m * torch.rsqrt((raw_om_m * raw_om_m).sum(dim=-1, keepdim=True) + 1.0)
            v_cross_t = chiral_cross_product(c1_t, c2_t, normalize=True, eps=1.0)

            chi_x = (v_t * om_x).sum(dim=-1)  # [B, N, C_t] — active immediately at Layer 1!
            chi_m = (v_t * om_m).sum(dim=-1)  # [B, N, C_t]
            chi_v = (v_t * v_cross_t).sum(dim=-1)  # [B, N, C_t]

            mlp_in = torch.cat((s, tr_m, norm_a, norm_s, v_norm, chi_x, chi_m, chi_v), dim=-1)
            gates: Tensor = self.update_mlp(mlp_in)
            a_ss, a_sv, a_vv, a_vt, a_tt, a_vc = torch.split(
                gates,
                [
                    self.scalar_dim,
                    self.vector_dim,
                    self.vector_dim,
                    self.vector_dim,
                    self.tensor_dim,
                    self.tensor_dim,
                ],
                dim=-1,
            )
            dx_chiral = vector_to_skew_matrix(a_vc.unsqueeze(-1) * v_cross_t)
        else:
            u_v, v_v, v_t = torch.split(
                vec_all, [self.vector_dim, self.vector_dim, self.tensor_dim], dim=-2
            )
            v_norm = torch.sqrt((v_v * v_v).sum(dim=-1) + 1e-8)
            uv_dot = (u_v * v_v).sum(dim=-1)

            mlp_in = torch.cat((s, tr_m, norm_a, norm_s, v_norm), dim=-1)
            gates = self.update_mlp(mlp_in)
            a_ss, a_sv, a_vv, a_vt, a_tt = torch.split(
                gates,
                [
                    self.scalar_dim,
                    self.vector_dim,
                    self.vector_dim,
                    self.vector_dim,
                    self.tensor_dim,
                ],
                dim=-1,
            )
            dx_chiral = None

        # 3. Exact-dimension residual updates with explicit W_v2t -> (S + A) -> W_t2v coupling.
        ds = a_ss + self.dot_to_scalar(a_sv * uv_dot)
        # Elementwise row contraction ((S + A) @ v_t) is 2.5x faster than 147k 3x3 matmuls on MPS.
        tv_t = ((m_s + m_a) * v_t.unsqueeze(-2)).sum(dim=-1)  # [B, N, C_t, 3]
        tv_v: Tensor = self.w_t2v(tv_t)  # [B, N, C_v, 3]
        dv = a_vv.unsqueeze(-1) * u_v + a_vt.unsqueeze(-1) * tv_v
        dx_ten = a_tt.unsqueeze(-1).unsqueeze(-1) * m_tilde
        if dx_chiral is not None:
            dx_ten = dx_ten + dx_chiral

        s_out = torch.where(mask_3d, self.scalar_norm(s + self.dropout(ds)), torch.zeros_like(s))
        v_out = torch.where(mask_4d, self.vec_norm(v + dv), torch.zeros_like(v))
        x_out = torch.where(mask_5d, self.tensor_norm(x_ten + dx_ten), torch.zeros_like(x_ten))
        return s_out, v_out, x_out


class CartFormerLandmarkBlock(nn.Module):
    """Equivariant global virtual landmark attention (``M = 8``) with GeoTransformer + LES biases.

    Resolves the ``5.5 A`` local-cutoff horizon in :math:`O(NM + M^2)` time by pooling valid
    atoms into ``M`` learnable equivariant landmarks ``(L_s, L_v, mu_m, q_m)``, attending
    across landmarks with :class:`GeoTransformerPairTripletBias` (pairwise RBF + triplet
    angles + signed RoITr PPF) and optional :class:`ScreenedCoulombBias` (``LES`` screened
    ``1/r`` electrostatics), and broadcasting updates back to valid atoms via residual gates.
    """

    scalar_dim: int
    vector_dim: int
    num_landmarks: int
    num_heads: int
    scalar_head_dim: int
    vector_head_dim: int
    use_se3_chirality: bool
    use_coulomb_bias: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_landmarks: int = 8,
        num_heads: int = 4,
        *,
        cutoff: float = 20.0,
        use_se3_chirality: bool = True,
        use_coulomb_bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1:
            raise ValueError("scalar_dim and vector_dim must be >= 1.")
        if num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")

        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_landmarks = int(num_landmarks)
        self.num_heads = _resolve_divisible_heads(num_heads, self.scalar_dim, self.vector_dim)
        self.scalar_head_dim = self.scalar_dim // self.num_heads
        self.vector_head_dim = self.vector_dim // self.num_heads
        self.use_se3_chirality = bool(use_se3_chirality)
        self.use_coulomb_bias = bool(use_coulomb_bias)

        self.landmark_queries = nn.Parameter(
            torch.randn(self.num_landmarks, self.scalar_dim, dtype=torch.float32)
            * (1.0 / math.sqrt(float(self.scalar_dim)))
        )
        self.pool_k = nn.Linear(self.scalar_dim, self.scalar_dim)

        self.lm_qs = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.lm_ks = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.lm_vs = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.lm_out_s = nn.Linear(self.scalar_dim, self.scalar_dim)

        self.lm_qkv_v = VectorNeuronLinear(self.vector_dim, 3 * self.vector_dim)
        self.lm_out_v = VectorNeuronLinear(self.vector_dim, self.vector_dim)

        self.geo_bias = GeoTransformerPairTripletBias(
            num_heads=self.num_heads,
            num_rbf=16,
            num_angle_bins=8,
            cutoff=cutoff,
            smooth_delta=0.5,
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
                charge_prior_idx=46,
            )
        else:
            self.coulomb_bias = None

        self.lm_norm_s = nn.LayerNorm(self.scalar_dim)
        self.lm_norm_v = EquivariantRMSNorm(self.vector_dim, eps=1e-4)

        self.unpool_q = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.unpool_k = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.unpool_vs = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.unpool_vv = VectorNeuronLinear(self.vector_dim, self.vector_dim)

        self.gate_s = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.gate_v = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.out_norm_s = nn.LayerNorm(self.scalar_dim)
        self.out_norm_v = EquivariantRMSNorm(self.vector_dim, eps=1e-4)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        x_ten: Tensor,
        centered_coords: Tensor,
        raw_node_feats: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run equivariant atom -> landmark -> atom global interaction."""
        batch_size, num_atoms, _ = s.shape
        mask_bool = atom_mask.to(torch.bool)
        mask_3d = mask_bool.unsqueeze(-1)
        mask_4d = mask_3d.unsqueeze(-1)
        if batch_size == 0 or num_atoms == 0:
            return s, v

        # 1. Permutation-invariant equivariant pooling into M landmarks (batched SGEMM).
        k_atoms: Tensor = self.pool_k(s)  # [B, N, D]
        pool_logits = torch.matmul(self.landmark_queries, k_atoms.transpose(-1, -2)) * (
            1.0 / math.sqrt(float(self.scalar_dim))
        )  # [B, M, N]
        pool_weights = safe_masked_softmax(pool_logits, mask_bool.unsqueeze(1), dim=-1)

        has_atoms = mask_bool.any(dim=1, keepdim=True)  # [B, 1]
        lm_mask = has_atoms.expand(batch_size, self.num_landmarks)  # [B, M]

        atom_omega = skew_matrix_to_vector(x_ten)  # [B, N, C_t, 3]
        ct3 = atom_omega.shape[2] * 3
        cv3 = self.vector_dim * 3
        pool_src = torch.cat(
            (
                s,
                v.reshape(batch_size, num_atoms, cv3),
                centered_coords,
                atom_omega.reshape(batch_size, num_atoms, ct3),
            ),
            dim=-1,
        )
        lm_pooled = torch.matmul(pool_weights, pool_src)
        lm_s, lm_v_flat, lm_coords, lm_om_flat = torch.split(
            lm_pooled, [self.scalar_dim, cv3, 3, ct3], dim=-1
        )
        lm_v = lm_v_flat.reshape(batch_size, self.num_landmarks, self.vector_dim, 3)
        lm_omega = lm_om_flat.reshape(batch_size, self.num_landmarks, atom_omega.shape[2], 3)

        # 2. M x M geometric + electrostatic landmark attention.
        q_ls = (
            self.lm_qs(lm_s)
            .reshape(batch_size, self.num_landmarks, self.num_heads, self.scalar_head_dim)
            .permute(0, 2, 1, 3)
        )
        k_ls = (
            self.lm_ks(lm_s)
            .reshape(batch_size, self.num_landmarks, self.num_heads, self.scalar_head_dim)
            .permute(0, 2, 1, 3)
        )
        v_ls = (
            self.lm_vs(lm_s)
            .reshape(batch_size, self.num_landmarks, self.num_heads, self.scalar_head_dim)
            .permute(0, 2, 1, 3)
        )

        qkv_v: Tensor = self.lm_qkv_v(lm_v)
        q_lv_raw, k_lv_raw, v_lv_raw = torch.chunk(qkv_v, 3, dim=-2)
        q_lv = q_lv_raw.reshape(
            batch_size, self.num_landmarks, self.num_heads, self.vector_head_dim * 3
        ).permute(0, 2, 1, 3)
        k_lv = k_lv_raw.reshape(
            batch_size, self.num_landmarks, self.num_heads, self.vector_head_dim * 3
        ).permute(0, 2, 1, 3)
        v_lv = v_lv_raw.reshape(
            batch_size, self.num_landmarks, self.num_heads, self.vector_head_dim * 3
        ).permute(0, 2, 1, 3)

        scalar_score = torch.matmul(q_ls, k_ls.transpose(-1, -2)) * (
            1.0 / math.sqrt(float(self.scalar_head_dim))
        )
        vn_score = torch.matmul(q_lv, k_lv.transpose(-1, -2)) * (
            1.0 / math.sqrt(3.0 * float(self.vector_head_dim))
        )
        geo_bias: Tensor = self.geo_bias(
            lm_coords,
            mask_i=lm_mask,
            vecs_i=lm_v,
            axial_i=lm_omega,
        )
        lm_logits = scalar_score + vn_score + geo_bias

        if self.coulomb_bias is not None:
            atom_q = self.coulomb_bias.predict_charges(s, raw_node_feats, mask_bool)  # [B, N]
            lm_q = torch.matmul(pool_weights, atom_q.unsqueeze(-1)).squeeze(-1)  # [B, M]
            elec_bias: Tensor = self.coulomb_bias(
                lm_coords,
                mask_i=lm_mask,
                charges_i=lm_q,
            )
            lm_logits = lm_logits + elec_bias

        lm_attn = safe_masked_softmax(lm_logits, lm_mask.unsqueeze(1).unsqueeze(2), dim=-1)
        lm_attn = self.dropout(lm_attn)  # [B, H, M, M]

        lm_s_agg = (
            torch.matmul(lm_attn, v_ls)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, self.num_landmarks, self.scalar_dim)
        )
        lm_v_agg = (
            torch.matmul(lm_attn, v_lv)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, self.num_landmarks, self.vector_dim, 3)
        )

        lm_mask_3d = lm_mask.unsqueeze(-1)
        lm_mask_4d = lm_mask_3d.unsqueeze(-1)
        lm_s_out = torch.where(
            lm_mask_3d,
            self.lm_norm_s(lm_s + self.lm_out_s(lm_s_agg)),
            torch.zeros_like(lm_s),
        )
        lm_v_out = torch.where(
            lm_mask_4d,
            self.lm_norm_v(lm_v + self.lm_out_v(lm_v_agg)),
            torch.zeros_like(lm_v),
        )

        # 3. Smooth equivariant unpooling back to valid atoms via one fused SGEMM.
        unpool_logits = torch.matmul(
            self.unpool_q(s), self.unpool_k(lm_s_out).transpose(-1, -2)
        ) * (1.0 / math.sqrt(float(self.scalar_dim)))  # [B, N, M]
        unpool_weights = safe_masked_softmax(unpool_logits, lm_mask.unsqueeze(1), dim=-1)

        v_lm_proj: Tensor = self.unpool_vv(lm_v_out)
        unpool_src = torch.cat(
            (self.unpool_vs(lm_s_out), v_lm_proj.reshape(batch_size, self.num_landmarks, cv3)),
            dim=-1,
        )
        broad = torch.matmul(unpool_weights, unpool_src)
        s_broad, v_broad_flat = torch.split(broad, [self.scalar_dim, cv3], dim=-1)
        v_broad = v_broad_flat.reshape(batch_size, num_atoms, self.vector_dim, 3)

        s_out = torch.where(
            mask_3d,
            self.out_norm_s(s + self.gate_s * self.dropout(s_broad)),
            torch.zeros_like(s),
        )
        v_out = torch.where(
            mask_4d,
            self.out_norm_v(v + self.gate_v * v_broad),
            torch.zeros_like(v),
        )
        return s_out, v_out


class CartFormerSE3Encoder(nn.Module):
    """Flagship zero-``e3nn`` ``CartFormer-SE3`` 3D atomic encoder.

    Satisfies :class:`~geneuclidean.protocols.PocketEncoderProtocol`:
    ``forward(coords, node_feats, atom_mask) -> PocketEncoderOutput``.
    """

    node_in_dim: int
    hidden_dim: int
    vector_dim: int
    tensor_dim: int
    out_dim: int
    num_layers: int
    num_heads: int
    num_rbf: int
    num_landmarks: int
    cutoff: float
    max_neighbors: int
    use_se3_chirality: bool
    use_coulomb_bias: bool
    adaptive_cutoff: bool

    def __init__(
        self,
        node_in_dim: int = 52,
        hidden_dim: int = 192,
        vector_dim: int = 64,
        tensor_dim: int = 24,
        out_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_rbf: int = 24,
        num_landmarks: int = 8,
        cutoff: float = 5.5,
        max_neighbors: int = 24,
        use_se3_chirality: bool = True,
        use_coulomb_bias: bool = True,
        dropout: float = 0.1,
        *,
        adaptive_cutoff: bool = True,
    ) -> None:
        super().__init__()
        if node_in_dim < 1 or hidden_dim < 1 or vector_dim < 1 or tensor_dim < 1 or out_dim < 1:
            raise ValueError("All CartFormerSE3Encoder channel dimensions must be >= 1.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if num_landmarks < 0:
            raise ValueError(f"num_landmarks must be >= 0, got {num_landmarks}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if max_neighbors < 1:
            raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")

        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.tensor_dim = int(tensor_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_rbf = int(num_rbf)
        self.num_landmarks = int(num_landmarks)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.use_se3_chirality = bool(use_se3_chirality)
        self.use_coulomb_bias = bool(use_coulomb_bias)
        self.adaptive_cutoff = bool(adaptive_cutoff)

        self.node_embed = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.rbf = BesselRBF(num_rbf=self.num_rbf, cutoff=self.cutoff)

        # Compact chemically modulated Layer-0 init activating polar v0 (1o) + rank-2 X0 (2e + 1e!).
        # Operates in a C_init = min(8, tensor_dim) bottleneck before expanding to vector_dim &
        # tensor_dim, cutting Layer-0 MPS backward latency by ~6x while preserving full Layer-1
        # chemical chirality and density-agnostic O(1) norms across N=5..1500.
        self.init_dim = max(4, min(8, self.tensor_dim))
        init_total = 2 * self.init_dim + 1
        self.init_node_proj = nn.Linear(self.hidden_dim, init_total)
        self.init_rbf_proj = nn.Linear(self.num_rbf, init_total, bias=False)
        self.init_v_up = VectorNeuronLinear(self.init_dim, self.vector_dim)
        self.init_vec_norm = EquivariantRMSNorm(self.vector_dim, eps=1e-4)
        self.init_x_up = CartesianTensorLinear(self.init_dim, self.tensor_dim)
        self.init_tensor_norm = TensorEquivariantRMSNorm(self.tensor_dim, eps=1e-4)

        self.attn_layers = nn.ModuleList(
            [
                CartFormerAttentionLayer(
                    scalar_dim=self.hidden_dim,
                    vector_dim=self.vector_dim,
                    tensor_dim=self.tensor_dim,
                    num_heads=self.num_heads,
                    num_rbf=self.num_rbf,
                    use_se3_chirality=self.use_se3_chirality,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                CartFormerFFN(
                    scalar_dim=self.hidden_dim,
                    vector_dim=self.vector_dim,
                    tensor_dim=self.tensor_dim,
                    use_se3_chirality=self.use_se3_chirality,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        # Apply global landmark exchange at the midpoint layer so every atom exchanges
        # long-range (15+ A) GeoTransformer triplet + LES Coulomb context which then
        # propagates through the remaining local Cartesian-tensor layers at minimal MPS cost.
        self._landmark_layer_idx: int = (self.num_layers - 1) // 2
        self.landmark_block: CartFormerLandmarkBlock | None
        if self.num_landmarks > 0:
            lm_heads = min(4, self.num_heads)
            self.landmark_block = CartFormerLandmarkBlock(
                scalar_dim=self.hidden_dim,
                vector_dim=self.vector_dim,
                num_landmarks=self.num_landmarks,
                num_heads=lm_heads,
                cutoff=20.0,
                use_se3_chirality=self.use_se3_chirality,
                use_coulomb_bias=self.use_coulomb_bias,
                dropout=dropout,
            )
        else:
            self.landmark_block = None

        # Read out final scalar state concatenated with final rank-2 (I, A, S) invariants
        # so downstream heads directly receive learned 0e/1e/2e multipole norms and every
        # final-layer tensor parameter receives full gradients even when num_landmarks=0.
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + 3 * self.tensor_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.pool = MaskedAttentionPool(in_dim=self.out_dim, out_dim=self.out_dim)

    def _init_geometric_states(
        self,
        s: Tensor,
        graph: DenseKNNGraph,
        rbf_env: Tensor,
        edge_outer_traceless_flat: Tensor,
        atom_mask: Tensor,
        flat_nbr_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Initialize chemically weighted polar ``v0`` (1o) & rank-2 ``X0`` (2e + 1e)."""
        batch_size, num_atoms = atom_mask.shape
        device = rbf_env.device
        mask_4d = atom_mask.unsqueeze(-1).unsqueeze(-1)
        mask_5d = mask_4d.unsqueeze(-1)
        k_eff = graph.nbr_idx.shape[2]

        if k_eff == 0:
            v0 = torch.zeros(
                (batch_size, num_atoms, self.vector_dim, 3), device=device, dtype=torch.float32
            )
            x0 = torch.zeros(
                (batch_size, num_atoms, self.tensor_dim, 3, 3), device=device, dtype=torch.float32
            )
            return v0, x0

        inv_sqrt_k = 1.0 / math.sqrt(float(self.max_neighbors))
        bn = batch_size * num_atoms
        c_init = self.init_dim

        # Chemically modulated radial weights in compact C_init bottleneck (vanishes at d=r_eff).
        phi_init: Tensor = self.init_node_proj(s)  # [B, N, 2*C_init + 1]
        phi_init_j = _fast_gather_neighbors(phi_init, flat_nbr_idx, batch_size, num_atoms, k_eff)
        w_init = phi_init_j * self.init_rbf_proj(rbf_env)
        w_vr, w_sr, w_ar = torch.split(w_init, [c_init, c_init, 1], dim=-1)

        edge_dir_bn = graph.edge_dir.reshape(bn, k_eff, 3)
        outer_9_bn = edge_outer_traceless_flat.reshape(bn, k_eff, 9)

        # Polar v0_low [B, N, C_init, 3] -> expand & normalize to [B, N, C_v, 3].
        v0_low = (
            torch.bmm(w_vr.reshape(bn, k_eff, c_init).transpose(1, 2), edge_dir_bn).reshape(
                batch_size, num_atoms, c_init, 3
            )
            * inv_sqrt_k
        )
        v0_low = torch.where(mask_4d, v0_low, torch.zeros_like(v0_low))
        v0_raw: Tensor = self.init_vec_norm(self.init_v_up(v0_low))
        v0 = torch.where(mask_4d, v0_raw, torch.zeros_like(v0_raw))

        # Symmetric traceless (2e) quadrupole component of X0 in compact bottleneck.
        x_sym_low = (
            torch.bmm(w_sr.reshape(bn, k_eff, c_init).transpose(1, 2), outer_9_bn).reshape(
                batch_size, num_atoms, c_init, 3, 3
            )
            * inv_sqrt_k
        )

        # Antisymmetric (1e) axial component active immediately in Layer 1:
        # (v0_low_j) (w_ar * r_hat)^T - (w_ar * r_hat) (v0_low_j)^T.
        v0_low_j = _fast_gather_neighbors(v0_low, flat_nbr_idx, batch_size, num_atoms, k_eff)
        w_ar_ed = w_ar.reshape(bn, k_eff, 1) * edge_dir_bn  # [BN, K, 3]
        vr_outer = (
            torch.bmm(v0_low_j.reshape(bn, k_eff, c_init * 3).transpose(1, 2), w_ar_ed).reshape(
                batch_size, num_atoms, c_init, 3, 3
            )
            * inv_sqrt_k
        )
        x_anti_low = vr_outer - vr_outer.transpose(-1, -2)

        x0_norm: Tensor = self.init_tensor_norm(self.init_x_up(x_sym_low + x_anti_low))
        x0 = torch.where(mask_5d, x0_norm, torch.zeros_like(x0_norm))
        return v0, x0

    def forward(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode batched 3D atomic systems into ``SE(3)``-invariant and equivariant outputs.

        Args:
            coords: Atomic 3D coordinates of shape ``[B, N, 3]`` (``float32``).
            node_feats: E(3)-invariant atomic features of shape ``[B, N, node_in_dim]``.
            atom_mask: Boolean validity mask of shape ``[B, N]`` (``True`` for real atoms).

        Returns:
            :class:`PocketEncoderOutput` containing ``node_emb [B, N, out_dim]``,
            ``pooled_emb [B, out_dim]``, and ``vec_emb [B, N, vector_dim, 3]``.
        """
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
        if node_feats.ndim != 3 or node_feats.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f"node_feats shape {tuple(node_feats.shape)} incompatible with "
                f"coords shape {tuple(coords.shape)}"
            )
        if node_feats.shape[-1] != self.node_in_dim:
            raise ValueError(
                f"Expected node_feats trailing dim {self.node_in_dim}, got {node_feats.shape[-1]}"
            )
        if atom_mask.shape != coords.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} != coords[:2] {tuple(coords.shape[:2])}"
            )

        batch_size, num_atoms, _ = coords.shape
        device = coords.device
        mask_bool = atom_mask.to(torch.bool)
        mask_3d = mask_bool.unsqueeze(-1)
        mask_4d = mask_3d.unsqueeze(-1)

        if batch_size == 0 or num_atoms == 0:
            return PocketEncoderOutput(
                node_emb=torch.empty(
                    (batch_size, num_atoms, self.out_dim), device=device, dtype=torch.float32
                ),
                pooled_emb=torch.zeros(
                    (batch_size, self.out_dim), device=device, dtype=torch.float32
                ),
                vec_emb=torch.empty(
                    (batch_size, num_atoms, self.vector_dim, 3),
                    device=device,
                    dtype=torch.float32,
                ),
            )

        graph = build_dense_knn_graph(
            coords=coords,
            atom_mask=mask_bool,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            adaptive_cutoff=self.adaptive_cutoff,
        )
        k_eff = graph.nbr_idx.shape[2]
        bn = batch_size * num_atoms
        flat_nbr_idx = _build_flat_nbr_idx(graph.nbr_idx)
        rbf_env = self.rbf(graph.edge_dist) * graph.edge_env.unsqueeze(-1)  # [B, N, K, num_rbf]
        edge_outer_traceless_flat = outer_traceless(graph.edge_dir).reshape(
            batch_size, num_atoms, k_eff, 9
        )
        quad_basis_5 = _quadrupole_basis_5(graph.edge_dir)  # [B, N, K, 5]

        # Precompute shared directional edge packs & masks once across all layers.
        ed_outer_12 = torch.cat((graph.edge_dir, edge_outer_traceless_flat), dim=-1).reshape(
            bn, k_eff, 12
        )
        geom_edge = torch.cat((graph.edge_dir, quad_basis_5), dim=-1)
        eff_mask = (graph.nbr_mask & (graph.edge_env > 0.0)).unsqueeze(-1)
        env_max = graph.edge_env.max(dim=2, keepdim=True).values.unsqueeze(-1)

        safe_feats = torch.where(
            mask_3d,
            node_feats.to(torch.float32),
            torch.zeros(
                (batch_size, num_atoms, self.node_in_dim), device=device, dtype=torch.float32
            ),
        )
        s = torch.where(
            mask_3d,
            self.node_embed(safe_feats),
            torch.zeros(
                (batch_size, num_atoms, self.hidden_dim), device=device, dtype=torch.float32
            ),
        )
        v, x_ten = self._init_geometric_states(
            s, graph, rbf_env, edge_outer_traceless_flat, mask_bool, flat_nbr_idx
        )

        for layer_idx in range(self.num_layers):
            attn_layer = self.attn_layers[layer_idx]
            ffn_layer = self.ffn_layers[layer_idx]

            # Both state tensors and attention deltas are strictly zeroed on ~mask_bool.
            ds, dv, dx_ten = attn_layer(
                s,
                v,
                x_ten,
                graph,
                rbf_env,
                edge_outer_traceless_flat,
                quad_basis_5,
                mask_bool,
                flat_nbr_idx=flat_nbr_idx,
                geom_edge_i=geom_edge,
                geom_edge_j=geom_edge,
                ed_outer_12=ed_outer_12,
                eff_mask=eff_mask,
                env_max=env_max,
            )
            s, v, x_ten = ffn_layer(s + ds, v + dv, x_ten + dx_ten, mask_bool)

            if self.landmark_block is not None and layer_idx == self._landmark_layer_idx:
                s, v = self.landmark_block(
                    s, v, x_ten, graph.centered_coords, safe_feats, mask_bool
                )

        x_i_out, x_a_out, x_s_out = decompose_rank2_ias(x_ten)
        tr_out = x_i_out[..., 0, 0] + x_i_out[..., 1, 1] + x_i_out[..., 2, 2]
        norm_a_out = torch.sqrt((x_a_out * x_a_out).sum(dim=(-2, -1)) + 1e-8)
        norm_s_out = torch.sqrt((x_s_out * x_s_out).sum(dim=(-2, -1)) + 1e-8)
        readout_in = torch.cat((s, tr_out, norm_a_out, norm_s_out), dim=-1)
        node_emb = torch.where(
            mask_3d,
            self.out_proj(readout_in),
            torch.zeros((batch_size, num_atoms, self.out_dim), device=device, dtype=torch.float32),
        )
        pooled_emb: Tensor = self.pool(node_emb, mask_bool)
        vec_emb = torch.where(mask_4d, v, torch.zeros_like(v))
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=vec_emb)


def build_cartformer_se3(
    config: EncoderConfig,
    max_neighbors: int = 24,
) -> CartFormerSE3Encoder:
    """Construct a :class:`CartFormerSE3Encoder` from a validated :class:`EncoderConfig`."""
    eff_heads = _resolve_divisible_heads(config.num_heads, config.hidden_dim)
    return CartFormerSE3Encoder(
        node_in_dim=config.node_in_dim,
        hidden_dim=config.hidden_dim,
        vector_dim=config.vector_dim,
        tensor_dim=config.tensor_dim,
        out_dim=config.out_dim,
        num_layers=config.num_layers,
        num_heads=eff_heads,
        num_rbf=config.num_rbf,
        num_landmarks=config.num_landmarks,
        cutoff=config.cutoff,
        max_neighbors=max_neighbors,
        use_se3_chirality=config.use_se3_chirality,
        use_coulomb_bias=config.use_coulomb_bias,
        dropout=config.dropout,
    )


__all__ = [
    "CartFormerAttentionLayer",
    "CartFormerFFN",
    "CartFormerLandmarkBlock",
    "CartFormerSE3Encoder",
    "CartesianTensorLinear",
    "build_cartformer_se3",
]
