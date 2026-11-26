"""Registry-backed factory and exports for pluggable 3D SE(3)/E(3) pocket encoders.

Built-in encoders (``painn``, ``egnn``, ``e3nn``, ``schnet``) register automatically
into :data:`~geneuclidean.registry.ENCODER_REGISTRY` on import. External researchers
can plug in new equivariant architectures without modifying GenEuclidean source::

    import geneuclidean as ge

    @ge.register_encoder("my_se3_net")
    def build_my_se3(cfg, max_neighbors=24):
        return MySE3Encoder(node_in_dim=cfg.node_in_dim, out_dim=cfg.out_dim)
"""

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
from geneuclidean.registry import ENCODER_REGISTRY, list_encoders, register_encoder


def _build_painn(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
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


def _build_egnn(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
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


def _build_e3nn(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
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


def _build_schnet(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
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


# Register built-in 3D equivariant encoders idempotently into ENCODER_REGISTRY.
ENCODER_REGISTRY.register(
    "painn",
    _build_painn,
    override=True,
    metadata={"symmetry": "SE(3)/E(3)", "default": True},
)
ENCODER_REGISTRY.register(
    "egnn",
    _build_egnn,
    override=True,
    metadata={"symmetry": "E(3)"},
)
ENCODER_REGISTRY.register(
    "e3nn",
    _build_e3nn,
    override=True,
    metadata={"symmetry": "SE(3)/E(3) spherical tensor"},
)
ENCODER_REGISTRY.register(
    "schnet",
    _build_schnet,
    override=True,
    metadata={"symmetry": "E(3) scalar distance baseline"},
)


def build_encoder(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
    """Instantiate a built-in or custom registered 3D pocket encoder from ``EncoderConfig``.

    Args:
        cfg: Validated encoder architecture and hyperparameter config.
        max_neighbors: Maximum radius-graph neighbors per atom (default ``24``).

    Returns:
        Instantiated 3D pocket encoder module implementing
        ``forward(coords, node_feats, atom_mask) -> PocketEncoderOutput``.
    """
    if cfg.arch not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder architecture: {cfg.arch!r}. Registered: {list_encoders()}"
        )
    factory = ENCODER_REGISTRY.get(cfg.arch)
    module: nn.Module = factory(cfg, max_neighbors=max_neighbors)
    return module


__all__ = [
    "ENCODER_REGISTRY",
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
    "list_encoders",
    "register_encoder",
]
