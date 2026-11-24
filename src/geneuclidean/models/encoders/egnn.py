"""Pure-PyTorch E(n)-Equivariant Graph Neural Network (EGNN) 3D pocket encoder.

Follows Satorras, Hoogeboom & Welling (ICML 2021) with attention-gated invariant
distance messages, bounded coordinate updates, multi-channel equivariant vector readout,
LayerNorm + residual connections, and strict dummy-atom masking for MPS ``float32``.
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
    center_masked_coords,
)


class EGNNLayer(nn.Module):
    """Single E(3)-equivariant EGNN layer updating scalars, coordinates, and vectors."""

    def __init__(
        self,
        hidden_dim: int,
        vector_dim: int,
        num_rbf: int,
        dropout: float = 0.1,
        coord_step_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.coord_step_scale = float(coord_step_scale)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(hidden_dim, vector_dim),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.vec_norm = EquivariantRMSNorm(vector_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        v: Tensor,
        src: Tensor,
        dst: Tensor,
        rbf_fn: BesselRBF,
        base_env: Tensor,
        deg_inv: Tensor,
        mask_flat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one equivariant message-passing step over sparse pocket edges."""
        mask_f = mask_flat.to(h.dtype)
        if src.numel() == 0:
            return h * mask_f.unsqueeze(-1), x * mask_f.unsqueeze(-1), v * mask_f[:, None, None]

        rel = x[dst] - x[src]  # [E, 3]
        sq_dist = (rel * rel).sum(dim=-1)  # [E]
        dist = torch.sqrt(sq_dist.clamp(min=1e-12))  # [E]
        rbf_feat = rbf_fn(dist) * base_env.unsqueeze(-1)  # [E, num_rbf]

        edge_in = torch.cat([h[dst], h[src], rbf_feat, dist.unsqueeze(-1)], dim=-1)
        m_ij = self.edge_mlp(edge_in)  # [E, hidden_dim]
        attn = self.attn_mlp(m_ij) * base_env.unsqueeze(-1)  # [E, 1]

        # Equivariant normalized directional vector r_ij / (||r_ij|| + 1)
        unit_rel = rel / (dist.unsqueeze(-1) + 1.0)

        # Bounded coordinate update
        coord_weights = self.coord_mlp(m_ij) * base_env.unsqueeze(-1) * self.coord_step_scale
        dx = torch.zeros_like(x)
        dx.index_add_(0, dst, coord_weights * unit_rel)
        dx = dx * deg_inv.unsqueeze(-1)
        x_new = (x + dx) * mask_f.unsqueeze(-1)

        # Multi-channel equivariant vector update
        vec_weights = self.vec_mlp(m_ij) * base_env.unsqueeze(-1)  # [E, C_v]
        dv = torch.zeros_like(v)
        dv.index_add_(0, dst, vec_weights.unsqueeze(-1) * unit_rel.unsqueeze(-2))
        dv = dv * (1.0 / math.sqrt(12.0))
        v_new = self.vec_norm(v + dv) * mask_f.unsqueeze(-1).unsqueeze(-1)

        # Invariant scalar aggregation + step displacement norm
        m_agg = torch.zeros_like(h)
        m_agg.index_add_(0, dst, attn * m_ij)
        m_agg = m_agg * (1.0 / math.sqrt(12.0))
        dx_norm = torch.sqrt((dx * dx).sum(dim=-1, keepdim=True) + 1e-8)

        dh = self.node_mlp(torch.cat([h, m_agg, dx_norm], dim=-1))
        h_new = self.scalar_norm(h + self.dropout(dh)) * mask_f.unsqueeze(-1)
        return h_new, x_new, v_new


class EGNNEncoder(nn.Module):
    """Pure-PyTorch E(3)-equivariant GNN 3D protein pocket encoder."""

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

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    vector_dim=vector_dim,
                    num_rbf=num_rbf,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Bias-free mixing combining multi-channel vectors and total displacement x_L - x_0.
        self.vec_readout = nn.Linear(vector_dim + 1, vector_dim, bias=False)
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
        """Encode 3D pockets into E(3)-invariant node/pooled scalars and equivariant vectors."""
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

        x0 = center_masked_coords(coords, mask_bool).reshape(batch_size * num_atoms, 3)
        x = x0.clone()

        feats_flat = node_feats.to(torch.float32).reshape(batch_size * num_atoms, self.node_in_dim)
        feats_flat = torch.where(mask_flat.unsqueeze(-1), feats_flat, torch.zeros_like(feats_flat))
        h = self.node_embed(feats_flat) * mask_f_flat.unsqueeze(-1)
        v = torch.zeros(
            (batch_size * num_atoms, self.vector_dim, 3),
            device=coords.device,
            dtype=torch.float32,
        )

        if graph.src.numel() > 0:
            base_env = self.cutoff_fn(graph.edge_dist)
            deg = torch.zeros(batch_size * num_atoms, device=coords.device, dtype=torch.float32)
            deg.index_add_(0, graph.dst, torch.ones_like(base_env))
            deg_inv = 1.0 / deg.clamp(min=1.0)
        else:
            base_env = torch.empty(0, device=coords.device, dtype=torch.float32)
            deg_inv = torch.ones(batch_size * num_atoms, device=coords.device, dtype=torch.float32)

        for layer in self.layers:
            h, x, v = layer(
                h=h,
                x=x,
                v=v,
                src=graph.src,
                dst=graph.dst,
                rbf_fn=self.rbf,
                base_env=base_env,
                deg_inv=deg_inv,
                mask_flat=mask_flat,
            )

        disp = (x - x0).unsqueeze(-2)  # [B*N, 1, 3]
        v_cat = torch.cat([v, disp], dim=-2)  # [B*N, C_v + 1, 3]
        vec_mixed = self.vec_readout(v_cat.transpose(-1, -2)).transpose(-1, -2)
        vec_emb = (vec_mixed * mask_f_flat.unsqueeze(-1).unsqueeze(-1)).reshape(
            batch_size, num_atoms, self.vector_dim, 3
        )

        node_emb = (self.out_proj(h) * mask_f_flat.unsqueeze(-1)).reshape(
            batch_size, num_atoms, self.out_dim
        )
        pooled_emb = self.pool(node_emb, mask_bool)
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=vec_emb)


__all__ = ["EGNNEncoder", "EGNNLayer"]
