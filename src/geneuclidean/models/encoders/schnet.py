"""Distance-only E(3)-invariant continuous-filter SchNet 3D pocket encoder.

Serves as the clean scalar-only geometric ablation baseline (Schütt et al., NeurIPS 2017)
against directional/chiral vector encoders (PaiNN-SE3, EGNN, e3nn).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.models.encoders.rbf import (
    CosineCutoff,
    GaussianRBF,
    MaskedAttentionPool,
    PocketEncoderOutput,
    build_radius_knn_graph,
)


class ShiftedSoftplus(nn.Module):
    """Smooth SchNet activation ``ln(0.5 * exp(x) + 0.5) = softplus(x) - ln(2)``."""

    _LN2: float = math.log(2.0)

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self._LN2


class SchNetInteractionBlock(nn.Module):
    """Continuous-filter convolution interaction block over pairwise atomic distances."""

    def __init__(
        self,
        hidden_dim: int,
        num_rbf: int,
        dropout: float = 0.1,
        norm_scale: float = 1.0 / math.sqrt(12.0),
    ) -> None:
        super().__init__()
        self.norm_scale = float(norm_scale)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        src: Tensor,
        dst: Tensor,
        rbf: Tensor,
        edge_env: Tensor,
        mask_flat: Tensor,
    ) -> Tensor:
        """Apply continuous-filter convolution and residual atom-wise update."""
        mask_f = mask_flat.to(h.dtype).unsqueeze(-1)
        if src.numel() == 0:
            out_empty: Tensor = self.norm(h) * mask_f
            return out_empty

        w = self.filter_net(rbf) * edge_env.unsqueeze(-1)  # [E, hidden_dim]
        x_in = self.in_proj(h)
        messages = x_in[src] * w

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        agg = agg * self.norm_scale

        dh = self.out_mlp(agg)
        out: Tensor = self.norm(h + self.dropout(dh)) * mask_f
        return out


class SchNetEncoder(nn.Module):
    """E(3)-invariant continuous-filter SchNet pocket encoder (``vec_emb = None``)."""

    def __init__(
        self,
        node_in_dim: int = 52,
        hidden_dim: int = 192,
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
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)

        self.interactions = nn.ModuleList(
            [
                SchNetInteractionBlock(
                    hidden_dim=hidden_dim,
                    num_rbf=num_rbf,
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
        """Encode 3D pockets using distance-only E(3)-invariant continuous filters."""
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
        h = self.node_embed(feats_flat) * mask_f_flat.unsqueeze(-1)

        if graph.src.numel() > 0:
            rbf = self.rbf(graph.edge_dist)
            edge_env = self.cutoff_fn(graph.edge_dist)
        else:
            rbf = torch.empty((0, self.rbf.num_rbf), device=coords.device, dtype=torch.float32)
            edge_env = torch.empty(0, device=coords.device, dtype=torch.float32)

        for block in self.interactions:
            h = block(h, graph.src, graph.dst, rbf, edge_env, mask_flat)

        node_emb = (self.out_proj(h) * mask_f_flat.unsqueeze(-1)).reshape(
            batch_size, num_atoms, self.out_dim
        )
        pooled_emb = self.pool(node_emb, mask_bool)
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=None)


__all__ = ["SchNetEncoder", "SchNetInteractionBlock", "ShiftedSoftplus"]
