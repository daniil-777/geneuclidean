"""DensePaiNN-SE3: Pure dense k-NN MPS-native reimplementation of PaiNN-SE3.

Eliminates sparse ``index_add_`` atomic scatter contention on Apple Silicon MPS by
executing the exact PaiNN-SE3 scalar + directional vector message-passing and chiral
update equations (Schütt et al., ICML 2021 + GenEuclidean ``SE(3)`` chirality) directly
over contiguous ``[B, N, K, ...]`` dense neighbor tensors produced by
:func:`~geneuclidean.models.encoders.dense_knn.build_dense_knn_graph`.

Key engineering & mathematical properties:
  1. **Zero ``index_add_`` / zero ``torch.linalg.cross`` / zero ``float64``:** Every op
     executes natively on Apple Silicon MPS (``PYTORCH_ENABLE_MPS_FALLBACK=0``), CUDA,
     and CPU in ``torch.float32`` with deterministic reduction order. Neighbor gathers
     use a single packed row-vectorized ``index_select`` per layer (avoiding Apple MPS
     ``torch.gather`` scalar scatter bottlenecks).
  2. **Exact tie-safe smoothness:** Precomputed enveloped Bessel radial features
     :math:`\\mathbf{e}_{ij} = f_c(d_{ij}; r_{\\mathrm{eff},i})\\,\\boldsymbol{\\phi}(d_{ij})`
     pass through a bias-free linear filter so neighbors on the equidistant :math:`K`-th
     tie shell (:math:`d_{ij} = r_{\\mathrm{eff},i}`) and masked slots contribute exact
     ``0.0`` with zero first derivative.
  3. **Exact ``SE(3)`` chirality vs. ``E(3)`` toggle:** When ``use_se3_chirality=True``,
     uses the pure-elementwise bounded Levi-Civita cross product
     :func:`~geneuclidean.models.encoders.tensor_ops.chiral_cross_product` and scalar
     triple product :math:`\\langle \\mathbf{V}\\mathbf{v}_i, (\\mathbf{C}_1\\mathbf{v}_i)
     \\times (\\mathbf{C}_2\\mathbf{v}_i)\\rangle` to distinguish mirror-image
     :math:`(R)/(S)` and :math:`L/D` stereochemistry while maintaining exact ``SO(3)``
     equivariance. Setting ``use_se3_chirality=False`` restores strict ``E(3)`` symmetry.
  4. **Systems-ablation control baseline:** Serves as the controlled systems baseline
     disentangling dense ``k``-NN MPS speedup from higher-order architectural advances
     in ``CartFormer-SE3``, ``AtomGS-SE3``, and ``SubPocketVN-JEPA``.
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
)
from geneuclidean.models.encoders.rbf import (
    BesselRBF,
    EquivariantRMSNorm,
    MaskedAttentionPool,
    PocketEncoderOutput,
)
from geneuclidean.models.encoders.tensor_ops import (
    VectorNeuronLinear,
    chiral_cross_product,
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


class DensePaiNNMessageLayer(nn.Module):
    """Dense batched ``[B, N, K]`` PaiNN scalar + directional vector message layer.

    Applies pre-gather scalar MLP factorization (``PointMetaBase`` style) so the MLP
    runs once per atom ``[B, N]`` rather than per edge ``[B, N, K]``, packs ``(phi, v)``
    into a single row-vectorized ``index_select`` per layer, multiplies by bias-free
    enveloped Bessel radial filters, and reduces via contiguous dense ops and Metal SGEMM.
    """

    scalar_dim: int
    vector_dim: int
    num_rbf: int
    norm_scale: float

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_rbf: int,
        norm_scale: float = 1.0 / math.sqrt(12.0),
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1 or num_rbf < 1:
            raise ValueError(
                f"scalar_dim ({scalar_dim}), vector_dim ({vector_dim}), and "
                f"num_rbf ({num_rbf}) must all be >= 1"
            )
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_rbf = int(num_rbf)
        self.norm_scale = float(norm_scale)

        split_dim = self.scalar_dim + 2 * self.vector_dim
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, split_dim),
        )
        # Bias-free projection of enveloped RBF preserves exact 0.0 on tie-shell & masked edges.
        self.rbf_proj = nn.Linear(self.num_rbf, split_dim, bias=False)

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
        """Compute dense residual message updates ``(ds, dv)`` for all atoms.

        Args:
            s: Scalar atom states of shape ``[B, N, scalar_dim]``.
            v: Polar ``l=1`` vector atom states of shape ``[B, N, vector_dim, 3]``.
            graph: Precomputed tie-safe :class:`DenseKNNGraph`.
            rbf_env: Enveloped Bessel RBF features ``rbf(edge_dist) * edge_env[..., None]``
                of shape ``[B, N, K, num_rbf]``.
            atom_mask: Boolean validity mask of shape ``[B, N]``.
            flat_nbr_idx: Optional precomputed flat neighbor index tensor ``[B * N * K]``.

        Returns:
            Tuple ``(ds [B, N, scalar_dim], dv [B, N, vector_dim, 3])`` strictly zeroed
            on masked dummy atoms.
        """
        mask_bool = atom_mask.to(torch.bool)
        batch_size, num_atoms, _ = s.shape
        k_eff = graph.nbr_idx.shape[2]
        if k_eff == 0 or batch_size == 0 or num_atoms == 0:
            return torch.zeros_like(s), torch.zeros_like(v)

        idx_flat = _build_flat_nbr_idx(graph.nbr_idx) if flat_nbr_idx is None else flat_nbr_idx

        # Pre-gather scalar MLP on [B, N, C_s] avoids redundant K-fold MLP compute.
        # Two separate narrower index_select calls avoid allocating a 512-wide packed buffer
        # on Apple MPS (empirically ~90 ms faster in backward than one 512-wide gather).
        phi: Tensor = self.scalar_mlp(s)  # [B, N, C_s + 2*C_v]
        phi_j = _fast_gather_neighbors(phi, idx_flat, batch_size, num_atoms, k_eff)

        # Bias-free filter on enveloped RBF vanishes identically at d = r_eff and on ~nbr_mask.
        filt: Tensor = self.rbf_proj(rbf_env)
        msg = phi_j * filt  # [B, N, K, C_s + 2*C_v]

        m_s, m_vv, m_vr = torch.split(
            msg,
            [self.scalar_dim, self.vector_dim, self.vector_dim],
            dim=-1,
        )

        # Dense contiguous scalar reduction over K (zero sparse index_add_!).
        ds = m_s.sum(dim=2) * self.norm_scale

        # Dense vector reduction + native batched SGEMM for edge_dir coupling.
        v_j = _fast_gather_neighbors(v, idx_flat, batch_size, num_atoms, k_eff)
        dv_vec = (v_j * m_vv.unsqueeze(-1)).sum(dim=2)
        dv_dir = torch.matmul(m_vr.transpose(-1, -2), graph.edge_dir)  # [B, N, C_v, 3]
        dv = (dv_vec + dv_dir) * self.norm_scale

        ds = torch.where(mask_bool.unsqueeze(-1), ds, torch.zeros_like(ds))
        dv = torch.where(mask_bool.unsqueeze(-1).unsqueeze(-1), dv, torch.zeros_like(dv))
        return ds, dv


class DensePaiNNUpdateLayer(nn.Module):
    """Dense batched atom-wise PaiNN scalar-vector mixing with optional ``SE(3)`` chirality.

    Uses a fused :class:`~geneuclidean.models.encoders.tensor_ops.VectorNeuronLinear`
    projection for bias-free ``O(3)``-equivariant vector mixing and
    :func:`~geneuclidean.models.encoders.tensor_ops.chiral_cross_product` (pure elementwise
    Levi-Civita with smooth norm bounding) so the entire layer runs natively on Apple MPS.
    """

    scalar_dim: int
    vector_dim: int
    use_se3_chirality: bool

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1:
            raise ValueError(
                f"scalar_dim ({scalar_dim}) and vector_dim ({vector_dim}) must be >= 1"
            )
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.use_se3_chirality = bool(use_se3_chirality)

        # Fuse (u_proj, v_proj[, chiral_a, chiral_b]) into one bias-free equivariant GEMM.
        proj_mult = 4 if self.use_se3_chirality else 2
        self.vec_proj = VectorNeuronLinear(self.vector_dim, proj_mult * self.vector_dim)

        if self.use_se3_chirality:
            update_in_dim = self.scalar_dim + 2 * self.vector_dim
            update_out_dim = self.scalar_dim + 3 * self.vector_dim
        else:
            update_in_dim = self.scalar_dim + self.vector_dim
            update_out_dim = self.scalar_dim + 2 * self.vector_dim

        self.update_mlp = nn.Sequential(
            nn.Linear(update_in_dim, self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, update_out_dim),
        )
        self.dot_to_scalar = nn.Linear(self.vector_dim, self.scalar_dim)
        self.scalar_norm = nn.LayerNorm(self.scalar_dim)
        self.vec_norm = EquivariantRMSNorm(self.vector_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Mix scalar ``[B, N, C_s]`` and vector ``[B, N, C_v, 3]`` channels per atom."""
        mask_bool = atom_mask.to(torch.bool)
        mask_3d = mask_bool.unsqueeze(-1)
        mask_4d = mask_3d.unsqueeze(-1)

        proj: Tensor = self.vec_proj(v)
        if self.use_se3_chirality:
            u_v, v_v, c_a, c_b = torch.chunk(proj, 4, dim=-2)
            v_norm = torch.sqrt((v_v * v_v).sum(dim=-1) + 1e-8)
            uv_dot = (u_v * v_v).sum(dim=-1)
            v_cross = chiral_cross_product(c_a, c_b, normalize=True, eps=1.0)
            # Scalar triple product <V v, (C1 v) x (C2 v)>: SO(3)-invariant, reflection-odd.
            triple_prod = (v_v * v_cross).sum(dim=-1)

            mlp_in = torch.cat((s, v_norm, triple_prod), dim=-1)
            gates: Tensor = self.update_mlp(mlp_in)
            a_ss, a_sv, a_vv, a_vc = torch.split(
                gates,
                [self.scalar_dim, self.vector_dim, self.vector_dim, self.vector_dim],
                dim=-1,
            )
            ds = a_ss + self.dot_to_scalar(a_sv * uv_dot)
            dv = a_vv.unsqueeze(-1) * u_v + a_vc.unsqueeze(-1) * v_cross
        else:
            u_v, v_v = torch.chunk(proj, 2, dim=-2)
            v_norm = torch.sqrt((v_v * v_v).sum(dim=-1) + 1e-8)
            uv_dot = (u_v * v_v).sum(dim=-1)

            mlp_in = torch.cat((s, v_norm), dim=-1)
            gates = self.update_mlp(mlp_in)
            a_ss, a_sv, a_vv = torch.split(
                gates,
                [self.scalar_dim, self.vector_dim, self.vector_dim],
                dim=-1,
            )
            ds = a_ss + self.dot_to_scalar(a_sv * uv_dot)
            dv = a_vv.unsqueeze(-1) * u_v

        s_out = torch.where(mask_3d, self.scalar_norm(s + self.dropout(ds)), torch.zeros_like(s))
        v_out = torch.where(mask_4d, self.vec_norm(v + dv), torch.zeros_like(v))
        return s_out, v_out


class DensePaiNNSE3Encoder(nn.Module):
    """MPS-native dense ``k``-NN PaiNN-SE3 3D atomic encoder (systems-ablation control).

    Satisfies :class:`~geneuclidean.protocols.PocketEncoderProtocol` with signature
    ``forward(coords, node_feats, atom_mask) -> PocketEncoderOutput(node_emb, pooled_emb, vec_emb)``
    while eliminating sparse ``index_add_`` atomic scatter contention and preserving exact
    equidistant ``K``-th neighbor tie-swap invariance via adaptive smooth cutoffs.
    """

    node_in_dim: int
    hidden_dim: int
    vector_dim: int
    out_dim: int
    num_layers: int
    num_rbf: int
    cutoff: float
    max_neighbors: int
    use_se3_chirality: bool
    adaptive_cutoff: bool

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
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
        *,
        adaptive_cutoff: bool = True,
    ) -> None:
        super().__init__()
        if node_in_dim < 1 or hidden_dim < 1 or vector_dim < 1 or out_dim < 1:
            raise ValueError("All channel dimensions must be >= 1.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if max_neighbors < 1:
            raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")

        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.use_se3_chirality = bool(use_se3_chirality)
        self.adaptive_cutoff = bool(adaptive_cutoff)

        self.node_embed = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.rbf = BesselRBF(num_rbf=self.num_rbf, cutoff=self.cutoff)

        self.message_layers = nn.ModuleList(
            [
                DensePaiNNMessageLayer(
                    scalar_dim=self.hidden_dim,
                    vector_dim=self.vector_dim,
                    num_rbf=self.num_rbf,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                DensePaiNNUpdateLayer(
                    scalar_dim=self.hidden_dim,
                    vector_dim=self.vector_dim,
                    use_se3_chirality=self.use_se3_chirality,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.pool = MaskedAttentionPool(in_dim=self.out_dim, out_dim=self.out_dim)

    def forward(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode batched 3D atomic structures into invariant and equivariant features.

        Args:
            coords: Atomic 3D coordinates of shape ``[B, N, 3]`` (``float32``).
            node_feats: E(3)-invariant per-atom descriptors of shape ``[B, N, node_in_dim]``.
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
        rbf_env = self.rbf(graph.edge_dist) * graph.edge_env.unsqueeze(-1)  # [B, N, K, num_rbf]
        flat_nbr_idx = _build_flat_nbr_idx(graph.nbr_idx)

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
        v = torch.zeros(
            (batch_size, num_atoms, self.vector_dim, 3),
            device=device,
            dtype=torch.float32,
        )
        zero_s = torch.zeros_like(s)
        zero_v = torch.zeros_like(v)

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers, strict=True):
            ds, dv = msg_layer(s, v, graph, rbf_env, mask_bool, flat_nbr_idx=flat_nbr_idx)
            s = torch.where(mask_3d, s + ds, zero_s)
            v = torch.where(mask_4d, v + dv, zero_v)
            s, v = upd_layer(s, v, mask_bool)

        node_emb = torch.where(
            mask_3d,
            self.out_proj(s),
            torch.zeros((batch_size, num_atoms, self.out_dim), device=device, dtype=torch.float32),
        )
        pooled_emb: Tensor = self.pool(node_emb, mask_bool)
        vec_emb = torch.where(mask_4d, v, zero_v)
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=vec_emb)


def build_dense_painn_se3(
    config: EncoderConfig,
    max_neighbors: int = 24,
) -> DensePaiNNSE3Encoder:
    """Construct a :class:`DensePaiNNSE3Encoder` from a validated :class:`EncoderConfig`."""
    return DensePaiNNSE3Encoder(
        node_in_dim=config.node_in_dim,
        hidden_dim=config.hidden_dim,
        vector_dim=config.vector_dim,
        out_dim=config.out_dim,
        num_layers=config.num_layers,
        num_rbf=config.num_rbf,
        cutoff=config.cutoff,
        max_neighbors=max_neighbors,
        use_se3_chirality=config.use_se3_chirality,
        dropout=config.dropout,
    )


__all__ = [
    "DensePaiNNMessageLayer",
    "DensePaiNNSE3Encoder",
    "DensePaiNNUpdateLayer",
    "build_dense_painn_se3",
]
