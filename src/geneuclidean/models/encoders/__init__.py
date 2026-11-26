"""Registry-backed factory and exports for pluggable 3D SE(3)/E(3) pocket encoders.

All 8 built-in 3D pocket encoders (``painn``, ``egnn``, ``e3nn``, ``schnet``,
``dense_painn_se3``, ``cartformer_se3``, ``atomgs_se3``, ``subpocket_vn_jepa``) and their
canonical short aliases (``dense_painn``, ``cartformer``, ``atomgs``, ``subpocket_vn``,
``vn_jepa``) register automatically into :data:`~geneuclidean.registry.ENCODER_REGISTRY` on
import. External researchers can plug in new equivariant architectures without modifying
GenEuclidean source::

    import geneuclidean as ge

    @ge.register_encoder("my_se3_net")
    def build_my_se3(cfg, max_neighbors=24):
        return MySE3Encoder(node_in_dim=cfg.node_in_dim, out_dim=cfg.out_dim)
"""

from __future__ import annotations

import torch.nn as nn

from geneuclidean.config import EncoderConfig
from geneuclidean.models.encoders.atomgs import (
    AtomGSComplexOutput,
    AtomGSLocalOverlapLayer,
    AtomGSOverlapScorer,
    AtomGSSE3Encoder,
    AtomGSSuperEllipsoidLandmarkBlock,
    AtomicGaussianField,
    EquivariantGaussianFieldBuilder,
    build_atomgs_se3,
    fuse_ensemble_4d,
)
from geneuclidean.models.encoders.cartformer import (
    CartesianTensorLinear,
    CartFormerAttentionLayer,
    CartFormerFFN,
    CartFormerLandmarkBlock,
    CartFormerSE3Encoder,
    build_cartformer_se3,
)
from geneuclidean.models.encoders.complex_cross import SE3ComplexCrossAttention
from geneuclidean.models.encoders.dense_knn import (
    DenseKNNGraph,
    build_dense_knn_graph,
    center_masked_coords_with_offset,
    gather_neighbors,
    safe_masked_softmax,
)
from geneuclidean.models.encoders.dense_painn import (
    DensePaiNNMessageLayer,
    DensePaiNNSE3Encoder,
    DensePaiNNUpdateLayer,
    build_dense_painn_se3,
)
from geneuclidean.models.encoders.e3nn_encoder import E3NNEncoder
from geneuclidean.models.encoders.egnn import EGNNEncoder
from geneuclidean.models.encoders.electrostatics import (
    GeoTransformerPairTripletBias,
    ScreenedCoulombBias,
)
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
from geneuclidean.models.encoders.subpocket_vn import (
    ChiralLocalVNAttention,
    ChiralVNSwiGLU,
    CompetitiveSubPocketSlotPool,
    SubPocketEncodingOutput,
    SubPocketGlobalTransformer,
    SubPocketVNJEPAEncoder,
    build_subpocket_vn_jepa,
)
from geneuclidean.models.encoders.tensor_ops import (
    TensorEquivariantRMSNorm,
    VectorNeuronAttentionBlock,
    VectorNeuronLayerNorm,
    VectorNeuronLeakyReLU,
    VectorNeuronLinear,
    chiral_cross_product,
    compose_rank2_ias,
    decompose_rank2_ias,
    outer_traceless,
    skew_matrix_to_vector,
    vector_to_skew_matrix,
)
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


# Register all 8 built-in 3D equivariant encoders idempotently into ENCODER_REGISTRY.
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
ENCODER_REGISTRY.register(
    "dense_painn_se3",
    build_dense_painn_se3,
    aliases=("dense_painn",),
    override=True,
    metadata={"symmetry": "SE(3)/E(3) dense k-NN MPS-native PaiNN"},
)
ENCODER_REGISTRY.register(
    "cartformer_se3",
    build_cartformer_se3,
    aliases=("cartformer",),
    override=True,
    metadata={"symmetry": "SE(3)/E(3) rank-2 Cartesian tensor transformer"},
)
ENCODER_REGISTRY.register(
    "atomgs_se3",
    build_atomgs_se3,
    aliases=("atomgs",),
    override=True,
    metadata={"symmetry": "SE(3)/E(3) anisotropic 3D Gaussian field & ROCS overlap"},
)
ENCODER_REGISTRY.register(
    "subpocket_vn_jepa",
    build_subpocket_vn_jepa,
    aliases=("subpocket_vn", "vn_jepa"),
    override=True,
    metadata={"symmetry": "SE(3)/E(3) chiral Vector-Neuron sub-pocket + 3D JEPA"},
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
    "AtomGSComplexOutput",
    "AtomGSLocalOverlapLayer",
    "AtomGSOverlapScorer",
    "AtomGSSE3Encoder",
    "AtomGSSuperEllipsoidLandmarkBlock",
    "AtomicGaussianField",
    "BesselRBF",
    "CartFormerAttentionLayer",
    "CartFormerFFN",
    "CartFormerLandmarkBlock",
    "CartFormerSE3Encoder",
    "CartesianTensorLinear",
    "ChiralLocalVNAttention",
    "ChiralVNSwiGLU",
    "CompetitiveSubPocketSlotPool",
    "CosineCutoff",
    "DenseKNNGraph",
    "DensePaiNNMessageLayer",
    "DensePaiNNSE3Encoder",
    "DensePaiNNUpdateLayer",
    "E3NNEncoder",
    "EGNNEncoder",
    "EquivariantGaussianFieldBuilder",
    "EquivariantRMSNorm",
    "GaussianRBF",
    "GeoTransformerPairTripletBias",
    "MaskedAttentionPool",
    "PaiNNEncoder",
    "PocketEncoderOutput",
    "RadiusGraph",
    "SE3ComplexCrossAttention",
    "SchNetEncoder",
    "ScreenedCoulombBias",
    "SubPocketEncodingOutput",
    "SubPocketGlobalTransformer",
    "SubPocketVNJEPAEncoder",
    "TensorEquivariantRMSNorm",
    "VectorNeuronAttentionBlock",
    "VectorNeuronLayerNorm",
    "VectorNeuronLeakyReLU",
    "VectorNeuronLinear",
    "build_atomgs_se3",
    "build_cartformer_se3",
    "build_dense_knn_graph",
    "build_dense_painn_se3",
    "build_encoder",
    "build_radius_knn_graph",
    "build_subpocket_vn_jepa",
    "center_masked_coords",
    "center_masked_coords_with_offset",
    "chiral_cross_product",
    "compose_rank2_ias",
    "decompose_rank2_ias",
    "fuse_ensemble_4d",
    "gather_neighbors",
    "list_encoders",
    "outer_traceless",
    "register_encoder",
    "safe_masked_softmax",
    "skew_matrix_to_vector",
    "vector_to_skew_matrix",
]
