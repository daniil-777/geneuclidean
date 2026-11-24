"""Registry and exports for pluggable 3D SE(3)/E(3) protein pocket encoders."""

from __future__ import annotations

import torch.nn as nn

from geneuclidean.config import EncoderConfig
from geneuclidean.models.encoders.e3nn_encoder import E3NNEncoder
from geneuclidean.models.encoders.egnn import EGNNEncoder
from geneuclidean.models.encoders.painn import PaiNNEncoder
from geneuclidean.models.encoders.rbf import (
    BesselRBF,
    CosineCutoff,
    EquivariantRMSNorm,
    GaussianRBF,
    MaskedAttentionPool,
    PocketEncoderOutput,
    RadiusGraph,
    build_radius_knn_graph,
    center_masked_coords,
)
from geneuclidean.models.encoders.schnet import SchNetEncoder


def build_encoder(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
    """Instantiate a configured 3D pocket encoder from ``EncoderConfig``.

    Args:
        cfg: Validated encoder architecture and hyperparameter config.
        max_neighbors: Maximum radius-graph neighbors per atom (default ``24``).

    Returns:
        Instantiated 3D pocket encoder module implementing
        ``forward(coords, node_feats, atom_mask) -> PocketEncoderOutput``.
    """
    if cfg.arch == "painn":
        return PaiNNEncoder(
            node_in_dim=cfg.node_in_dim,
            hidden_dim=cfg.hidden_dim,
            vector_dim=cfg.vector_dim,
            out_dim=cfg.out_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            cutoff=cfg.cutoff,
            max_neighbors=max_neighbors,
            use_se3_chirality=cfg.use_se3_chirality,
            dropout=cfg.dropout,
        )
    if cfg.arch == "egnn":
        return EGNNEncoder(
            node_in_dim=cfg.node_in_dim,
            hidden_dim=cfg.hidden_dim,
            vector_dim=cfg.vector_dim,
            out_dim=cfg.out_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            cutoff=cfg.cutoff,
            max_neighbors=max_neighbors,
            dropout=cfg.dropout,
        )
    if cfg.arch == "e3nn":
        return E3NNEncoder(
            node_in_dim=cfg.node_in_dim,
            hidden_dim=cfg.hidden_dim,
            vector_dim=cfg.vector_dim,
            out_dim=cfg.out_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            cutoff=cfg.cutoff,
            max_neighbors=max_neighbors,
            e3nn_lmax=cfg.e3nn_lmax,
            use_se3_chirality=cfg.use_se3_chirality,
            dropout=cfg.dropout,
        )
    if cfg.arch == "schnet":
        return SchNetEncoder(
            node_in_dim=cfg.node_in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            cutoff=cfg.cutoff,
            max_neighbors=max_neighbors,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown encoder architecture: {cfg.arch!r}")


__all__ = [
    "BesselRBF",
    "CosineCutoff",
    "E3NNEncoder",
    "EGNNEncoder",
    "EquivariantRMSNorm",
    "GaussianRBF",
    "MaskedAttentionPool",
    "PaiNNEncoder",
    "PocketEncoderOutput",
    "RadiusGraph",
    "SchNetEncoder",
    "build_encoder",
    "build_radius_knn_graph",
    "center_masked_coords",
]
