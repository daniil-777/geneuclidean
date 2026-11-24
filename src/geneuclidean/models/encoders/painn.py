"""PaiNN-SE3: Pure-PyTorch SE(3)/E(3)-equivariant polarizable atom interaction network.

Implements Schütt et al. (ICML 2021) PaiNN with separate scalar (``C_s``) and 3D vector
(``C_v``) channels, Bessel RBFs with smooth cosine cutoff, equivariant RMSNorm, and an
optional ``SE(3)`` cross-product + scalar-triple-product chirality block that breaks
mirror symmetry (``det(R) = -1``) for chiral L-amino-acid protein binding pockets while
preserving exact ``SO(3)`` rotational equivariance (``det(R) = +1``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from geneuclidean.models.encoders.rbf import (
    BesselRBF,
    CosineCutoff,
    EquivariantRMSNorm,
    MaskedAttentionPool,
    PocketEncoderOutput,
    build_radius_knn_graph,
)


class _VectorLinear(nn.Module):
    """Bias-free channel mixing ``[..., C_in, 3] -> [..., C_out, 3]`` preserving O(3)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, vec: Tensor) -> Tensor:
        # Transpose so nn.Linear mixes the channel dimension identically across x, y, z.
        out: Tensor = self.linear(vec.transpose(-1, -2)).transpose(-1, -2)
        return out


class PaiNNMessageBlock(nn.Module):
    """PaiNN scalar + directional vector message passing block."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_rbf: int,
        norm_scale: float = 1.0 / math.sqrt(12.0),
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.norm_scale = float(norm_scale)

        split_dim = scalar_dim + 2 * vector_dim
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, split_dim),
        )
        self.rbf_proj = nn.Linear(num_rbf, split_dim)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        src: Tensor,
        dst: Tensor,
        edge_dir: Tensor,
        rbf_env: Tensor,
        mask_flat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute residual message updates ``(ds, dv)`` over sparse edges."""
        if src.numel() == 0:
            return torch.zeros_like(s), torch.zeros_like(v)

        phi = self.scalar_mlp(s)[src]  # [E, C_s + 2*C_v]
        filt = self.rbf_proj(rbf_env)  # [E, C_s + 2*C_v]
        msg = phi * filt

        m_s, m_vv, m_vr = torch.split(
            msg, [self.scalar_dim, self.vector_dim, self.vector_dim], dim=-1
        )
        m_v = v[src] * m_vv.unsqueeze(-1) + m_vr.unsqueeze(-1) * edge_dir.unsqueeze(-2)

        ds = torch.zeros_like(s)
        ds.index_add_(0, dst, m_s)
        ds = ds * self.norm_scale

        dv = torch.zeros_like(v)
        dv.index_add_(0, dst, m_v)
        dv = dv * self.norm_scale

        mask_f = mask_flat.to(s.dtype)
        ds = ds * mask_f.unsqueeze(-1)
        dv = dv * mask_f.unsqueeze(-1).unsqueeze(-1)
        return ds, dv


class PaiNNUpdateBlock(nn.Module):
    """PaiNN atom-wise scalar-vector mixing with optional SE(3) chirality cross-product."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.use_se3_chirality = bool(use_se3_chirality)

        self.u_proj = _VectorLinear(vector_dim, vector_dim)
        self.v_proj = _VectorLinear(vector_dim, vector_dim)

        self.chiral_proj_a: _VectorLinear | None
        self.chiral_proj_b: _VectorLinear | None
        if self.use_se3_chirality:
            self.chiral_proj_a = _VectorLinear(vector_dim, vector_dim)
            self.chiral_proj_b = _VectorLinear(vector_dim, vector_dim)
            update_in_dim = scalar_dim + 2 * vector_dim
            update_out_dim = scalar_dim + 3 * vector_dim
        else:
            self.chiral_proj_a = None
            self.chiral_proj_b = None
            update_in_dim = scalar_dim + vector_dim
            update_out_dim = scalar_dim + 2 * vector_dim

        self.update_mlp = nn.Sequential(
            nn.Linear(update_in_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, update_out_dim),
        )
        self.dot_to_scalar = nn.Linear(vector_dim, scalar_dim)
        self.scalar_norm = nn.LayerNorm(scalar_dim)
        self.vec_norm = EquivariantRMSNorm(vector_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, s: Tensor, v: Tensor, mask_flat: Tensor) -> tuple[Tensor, Tensor]:
        """Mix scalar and vector channels on each atom while preserving symmetry."""
        u_v = self.u_proj(v)  # [N_tot, C_v, 3]
        v_v = self.v_proj(v)  # [N_tot, C_v, 3]

        v_norm = torch.sqrt((v_v * v_v).sum(dim=-1) + 1e-8)  # [N_tot, C_v]
        uv_dot = (u_v * v_v).sum(dim=-1)  # [N_tot, C_v]

        if self.use_se3_chirality:
            assert self.chiral_proj_a is not None and self.chiral_proj_b is not None
            c_a = self.chiral_proj_a(v)
            c_b = self.chiral_proj_b(v)
            raw_cross = torch.linalg.cross(c_a, c_b, dim=-1)  # [N_tot, C_v, 3]
            # Smoothly bound quadratic cross-product norm while preserving exact SO(3) & chirality.
            cross_denom = torch.sqrt((raw_cross * raw_cross).sum(dim=-1, keepdim=True) + 1.0)
            v_cross = raw_cross / cross_denom
            # Scalar triple product <V v, (C1 v) x (C2 v)>: SO(3)-invariant, reflection-odd!
            triple_prod = (v_v * v_cross).sum(dim=-1)  # [N_tot, C_v]

            mlp_in = torch.cat([s, v_norm, triple_prod], dim=-1)
            gates = self.update_mlp(mlp_in)
            a_ss, a_sv, a_vv, a_vc = torch.split(
                gates,
                [self.scalar_dim, self.vector_dim, self.vector_dim, self.vector_dim],
                dim=-1,
            )
            ds = a_ss + self.dot_to_scalar(a_sv * uv_dot)
            dv = a_vv.unsqueeze(-1) * u_v + a_vc.unsqueeze(-1) * v_cross
        else:
            mlp_in = torch.cat([s, v_norm], dim=-1)
            gates = self.update_mlp(mlp_in)
            a_ss, a_sv, a_vv = torch.split(
                gates, [self.scalar_dim, self.vector_dim, self.vector_dim], dim=-1
            )
            ds = a_ss + self.dot_to_scalar(a_sv * uv_dot)
            dv = a_vv.unsqueeze(-1) * u_v

        mask_f = mask_flat.to(s.dtype)
        s_out = self.scalar_norm(s + self.dropout(ds)) * mask_f.unsqueeze(-1)
        v_out = self.vec_norm(v + dv) * mask_f.unsqueeze(-1).unsqueeze(-1)
        return s_out, v_out


class PaiNNEncoder(nn.Module):
    """Flagship pure-PyTorch PaiNN-SE3 3D protein pocket encoder.

    When ``use_se3_chirality=True``, the encoder is strictly ``SE(3)``-equivariant
    (exact invariance/equivariance under rigid rotations ``det(R) = +1`` and translations,
    and strictly sensitive to mirror reflections ``det(R) = -1``). When ``False``, it is
    strictly ``E(3)``-equivariant (also invariant under reflections).
    """

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
    ) -> None:
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.use_se3_chirality = bool(use_se3_chirality)

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)

        self.message_blocks = nn.ModuleList(
            [
                PaiNNMessageBlock(
                    scalar_dim=hidden_dim,
                    vector_dim=vector_dim,
                    num_rbf=num_rbf,
                )
                for _ in range(num_layers)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                PaiNNUpdateBlock(
                    scalar_dim=hidden_dim,
                    vector_dim=vector_dim,
                    use_se3_chirality=use_se3_chirality,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.pool = MaskedAttentionPool(in_dim=out_dim, out_dim=out_dim)

    def forward(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode a batch of 3D protein pockets into invariant and equivariant features."""
        batch_size, num_atoms, _ = coords.shape
        mask_bool = atom_mask.to(torch.bool)
        mask_flat = mask_bool.reshape(batch_size * num_atoms)
        mask_f_flat = mask_flat.to(torch.float32)

        graph = build_radius_knn_graph(
            coords=coords,
            atom_mask=mask_bool,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

        feats_flat = node_feats.to(torch.float32).reshape(batch_size * num_atoms, self.node_in_dim)
        feats_flat = torch.where(mask_flat.unsqueeze(-1), feats_flat, torch.zeros_like(feats_flat))
        s = self.node_embed(feats_flat) * mask_f_flat.unsqueeze(-1)
        v = torch.zeros(
            (batch_size * num_atoms, self.vector_dim, 3),
            device=coords.device,
            dtype=torch.float32,
        )

        if graph.src.numel() > 0:
            edge_dir = graph.edge_vec / graph.edge_dist.unsqueeze(-1).clamp(min=1e-6)
            env = self.cutoff_fn(graph.edge_dist).unsqueeze(-1)
            rbf_env = self.rbf(graph.edge_dist) * env
        else:
            edge_dir = torch.empty((0, 3), device=coords.device, dtype=torch.float32)
            rbf_env = torch.empty((0, self.rbf.num_rbf), device=coords.device, dtype=torch.float32)

        for msg_block, upd_block in zip(self.message_blocks, self.update_blocks, strict=True):
            ds, dv = msg_block(s, v, graph.src, graph.dst, edge_dir, rbf_env, mask_flat)
            s = (s + ds) * mask_f_flat.unsqueeze(-1)
            v = (v + dv) * mask_f_flat.unsqueeze(-1).unsqueeze(-1)
            s, v = upd_block(s, v, mask_flat)

        node_emb = self.out_proj(s) * mask_f_flat.unsqueeze(-1)
        node_emb = node_emb.reshape(batch_size, num_atoms, self.out_dim)
        pooled_emb = self.pool(node_emb, mask_bool)
        vec_emb = (v * mask_f_flat.unsqueeze(-1).unsqueeze(-1)).reshape(
            batch_size, num_atoms, self.vector_dim, 3
        )
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=vec_emb)


__all__ = ["PaiNNEncoder", "PaiNNMessageBlock", "PaiNNUpdateBlock"]
