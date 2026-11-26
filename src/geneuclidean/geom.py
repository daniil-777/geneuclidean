"""Foundational SE(3)/O(3) 3D Geometric Operations Library for GenEuclidean.

Implements PEP 562 lazy loading (``__getattr__`` / ``__dir__``) with full PEP 561 static type
checking (``if TYPE_CHECKING:``) so ``import geneuclidean.geom as geom`` completes in ``< 15 ms``
with **zero** eager ``torch`` or ``rdkit`` imports.

All underlying tensor primitives execute strictly in ``torch.float32`` without external C++/CUDA
extensions (``torch-cluster``, ``torch-scatter``, ``e3nn`` CG kernels), delivering deterministic
``SE(3)`` equivariance and high SIMD throughput on Apple Silicon MPS, CUDA, and CPU.

Cookbook — Composing a Custom Chiral SE(3) Molecular Block
----------------------------------------------------------

.. code-block:: python

    import torch
    import geneuclidean.geom as geom

    # 1. Build a tie-safe dense k-NN graph (0.0 equidistant tie-swap error, zero index_add_)
    graph = geom.build_dense_knn_graph(coords, atom_mask, cutoff=5.5, max_neighbors=24)

    # 2. Construct O(3)-equivariant 2e quadrupole & chiral 1e axial tensors from edge geometry
    quad = geom.outer_traceless(graph.edge_dir)                     # [B, N, K, 3, 3]
    I, A, S = geom.decompose_rank2_ias(quad)                        # 0e + 1e + 2e irreps
    omega = geom.skew_matrix_to_vector(A)                           # Hodge-dual axial vector

    # 3. Build condition-bounded SPD atomic Gaussians (cond <= 29.8) & ROCS Shape Tanimoto
    sigma = geom.build_anisotropic_covariance_se3(iso_logits, polar_vecs=v, axial_vecs=ax)
    field = geom.AtomicGaussianField(graph.centered_coords, graph.centroid_offset, sigma)
    tanimoto = geom.rocs_shape_tanimoto(pocket_field, ligand_field, pocket_mask, ligand_mask)

    # 4. Add long-range screened 1/r electrostatics (LES) + GeoTransformer triplet/PPF bias
    coulomb = geom.ScreenedCoulombBias(scalar_dim=192, num_heads=8)
    geo_bias = geom.GeoTransformerPairTripletBias(num_heads=8)
    vn_block = geom.VectorNeuronAttentionBlock(vector_dim=64, num_heads=4)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geneuclidean.models.encoders.complex_cross import (
        SE3ComplexCrossAttention as SE3ComplexCrossAttention,
    )
    from geneuclidean.models.encoders.dense_knn import (
        DenseKNNGraph as DenseKNNGraph,
    )
    from geneuclidean.models.encoders.dense_knn import (
        build_dense_knn_graph as build_dense_knn_graph,
    )
    from geneuclidean.models.encoders.dense_knn import (
        center_masked_coords_with_offset as center_masked_coords_with_offset,
    )
    from geneuclidean.models.encoders.dense_knn import (
        gather_neighbors as gather_neighbors,
    )
    from geneuclidean.models.encoders.dense_knn import (
        safe_masked_softmax as safe_masked_softmax,
    )
    from geneuclidean.models.encoders.electrostatics import (
        GeoTransformerPairTripletBias as GeoTransformerPairTripletBias,
    )
    from geneuclidean.models.encoders.electrostatics import (
        ScreenedCoulombBias as ScreenedCoulombBias,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        AtomicGaussianField as AtomicGaussianField,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        GaussianOverlapResult as GaussianOverlapResult,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        build_anisotropic_covariance_se3 as build_anisotropic_covariance_se3,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        cofactor_det_3x3 as cofactor_det_3x3,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        cofactor_inverse_and_logdet_3x3 as cofactor_inverse_and_logdet_3x3,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        cofactor_solve_3x3 as cofactor_solve_3x3,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        fuse_4d_md_covariances as fuse_4d_md_covariances,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        fuse_multiconformer_fields as fuse_multiconformer_fields,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        gaussian_overlap_details_3d as gaussian_overlap_details_3d,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        gaussian_overlap_integral_3d as gaussian_overlap_integral_3d,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        gaussianformer2_occupancy as gaussianformer2_occupancy,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        rocs_self_overlap as rocs_self_overlap,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        rocs_shape_tanimoto as rocs_shape_tanimoto,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        super_ellipsoid_pool as super_ellipsoid_pool,
    )
    from geneuclidean.models.encoders.gaussian_ops import (
        vdw_radius_to_iso_logit as vdw_radius_to_iso_logit,
    )
    from geneuclidean.models.encoders.ligand3d import (
        LIGAND_3D_NODE_DIM as LIGAND_3D_NODE_DIM,
    )
    from geneuclidean.models.encoders.ligand3d import (
        collate_ligand_3d_batch as collate_ligand_3d_batch,
    )
    from geneuclidean.models.encoders.ligand3d import (
        featurize_ligand_3d as featurize_ligand_3d,
    )
    from geneuclidean.models.encoders.rbf import (
        BesselRBF as BesselRBF,
    )
    from geneuclidean.models.encoders.rbf import (
        CosineCutoff as CosineCutoff,
    )
    from geneuclidean.models.encoders.rbf import (
        EquivariantRMSNorm as EquivariantRMSNorm,
    )
    from geneuclidean.models.encoders.rbf import (
        GaussianRBF as GaussianRBF,
    )
    from geneuclidean.models.encoders.rbf import (
        MaskedAttentionPool as MaskedAttentionPool,
    )
    from geneuclidean.models.encoders.rbf import (
        PocketEncoderOutput as PocketEncoderOutput,
    )
    from geneuclidean.models.encoders.rbf import (
        RadiusGraph as RadiusGraph,
    )
    from geneuclidean.models.encoders.rbf import (
        build_radius_knn_graph as build_radius_knn_graph,
    )
    from geneuclidean.models.encoders.rbf import (
        center_masked_coords as center_masked_coords,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        TensorEquivariantRMSNorm as TensorEquivariantRMSNorm,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        VectorNeuronAttentionBlock as VectorNeuronAttentionBlock,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        VectorNeuronLayerNorm as VectorNeuronLayerNorm,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        VectorNeuronLeakyReLU as VectorNeuronLeakyReLU,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        VectorNeuronLinear as VectorNeuronLinear,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        chiral_cross_product as chiral_cross_product,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        compose_rank2_ias as compose_rank2_ias,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        decompose_rank2_ias as decompose_rank2_ias,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        outer_traceless as outer_traceless,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        skew_matrix_to_vector as skew_matrix_to_vector,
    )
    from geneuclidean.models.encoders.tensor_ops import (
        vector_to_skew_matrix as vector_to_skew_matrix,
    )

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Dense tie-safe k-NN & safe softmax (`dense_knn`)
    "DenseKNNGraph": ("geneuclidean.models.encoders.dense_knn", "DenseKNNGraph"),
    "build_dense_knn_graph": ("geneuclidean.models.encoders.dense_knn", "build_dense_knn_graph"),
    "center_masked_coords_with_offset": (
        "geneuclidean.models.encoders.dense_knn",
        "center_masked_coords_with_offset",
    ),
    "gather_neighbors": ("geneuclidean.models.encoders.dense_knn", "gather_neighbors"),
    "safe_masked_softmax": ("geneuclidean.models.encoders.dense_knn", "safe_masked_softmax"),
    # Radial bases, cutoffs & sparse radius graph (`rbf`)
    "BesselRBF": ("geneuclidean.models.encoders.rbf", "BesselRBF"),
    "CosineCutoff": ("geneuclidean.models.encoders.rbf", "CosineCutoff"),
    "EquivariantRMSNorm": ("geneuclidean.models.encoders.rbf", "EquivariantRMSNorm"),
    "GaussianRBF": ("geneuclidean.models.encoders.rbf", "GaussianRBF"),
    "MaskedAttentionPool": ("geneuclidean.models.encoders.rbf", "MaskedAttentionPool"),
    "PocketEncoderOutput": ("geneuclidean.models.encoders.rbf", "PocketEncoderOutput"),
    "RadiusGraph": ("geneuclidean.models.encoders.rbf", "RadiusGraph"),
    "build_radius_knn_graph": ("geneuclidean.models.encoders.rbf", "build_radius_knn_graph"),
    "center_masked_coords": ("geneuclidean.models.encoders.rbf", "center_masked_coords"),
    # Cartesian 3x3 irrep & Vector-Neuron algebra (`tensor_ops`)
    "TensorEquivariantRMSNorm": (
        "geneuclidean.models.encoders.tensor_ops",
        "TensorEquivariantRMSNorm",
    ),
    "VectorNeuronAttentionBlock": (
        "geneuclidean.models.encoders.tensor_ops",
        "VectorNeuronAttentionBlock",
    ),
    "VectorNeuronLayerNorm": ("geneuclidean.models.encoders.tensor_ops", "VectorNeuronLayerNorm"),
    "VectorNeuronLeakyReLU": ("geneuclidean.models.encoders.tensor_ops", "VectorNeuronLeakyReLU"),
    "VectorNeuronLinear": ("geneuclidean.models.encoders.tensor_ops", "VectorNeuronLinear"),
    "chiral_cross_product": ("geneuclidean.models.encoders.tensor_ops", "chiral_cross_product"),
    "compose_rank2_ias": ("geneuclidean.models.encoders.tensor_ops", "compose_rank2_ias"),
    "decompose_rank2_ias": ("geneuclidean.models.encoders.tensor_ops", "decompose_rank2_ias"),
    "outer_traceless": ("geneuclidean.models.encoders.tensor_ops", "outer_traceless"),
    "skew_matrix_to_vector": ("geneuclidean.models.encoders.tensor_ops", "skew_matrix_to_vector"),
    "vector_to_skew_matrix": ("geneuclidean.models.encoders.tensor_ops", "vector_to_skew_matrix"),
    # Long-range electrostatics & GeoTransformer/RoITr geometric biases (`electrostatics`)
    "GeoTransformerPairTripletBias": (
        "geneuclidean.models.encoders.electrostatics",
        "GeoTransformerPairTripletBias",
    ),
    "ScreenedCoulombBias": ("geneuclidean.models.encoders.electrostatics", "ScreenedCoulombBias"),
    # Differentiable anisotropic 3D/4D Gaussian algebra (`gaussian_ops`)
    "AtomicGaussianField": ("geneuclidean.models.encoders.gaussian_ops", "AtomicGaussianField"),
    "GaussianOverlapResult": (
        "geneuclidean.models.encoders.gaussian_ops",
        "GaussianOverlapResult",
    ),
    "build_anisotropic_covariance_se3": (
        "geneuclidean.models.encoders.gaussian_ops",
        "build_anisotropic_covariance_se3",
    ),
    "cofactor_det_3x3": ("geneuclidean.models.encoders.gaussian_ops", "cofactor_det_3x3"),
    "cofactor_inverse_and_logdet_3x3": (
        "geneuclidean.models.encoders.gaussian_ops",
        "cofactor_inverse_and_logdet_3x3",
    ),
    "cofactor_solve_3x3": ("geneuclidean.models.encoders.gaussian_ops", "cofactor_solve_3x3"),
    "fuse_4d_md_covariances": (
        "geneuclidean.models.encoders.gaussian_ops",
        "fuse_4d_md_covariances",
    ),
    "fuse_multiconformer_fields": (
        "geneuclidean.models.encoders.gaussian_ops",
        "fuse_multiconformer_fields",
    ),
    "gaussian_overlap_details_3d": (
        "geneuclidean.models.encoders.gaussian_ops",
        "gaussian_overlap_details_3d",
    ),
    "gaussian_overlap_integral_3d": (
        "geneuclidean.models.encoders.gaussian_ops",
        "gaussian_overlap_integral_3d",
    ),
    "gaussianformer2_occupancy": (
        "geneuclidean.models.encoders.gaussian_ops",
        "gaussianformer2_occupancy",
    ),
    "rocs_self_overlap": ("geneuclidean.models.encoders.gaussian_ops", "rocs_self_overlap"),
    "rocs_shape_tanimoto": ("geneuclidean.models.encoders.gaussian_ops", "rocs_shape_tanimoto"),
    "super_ellipsoid_pool": ("geneuclidean.models.encoders.gaussian_ops", "super_ellipsoid_pool"),
    "vdw_radius_to_iso_logit": (
        "geneuclidean.models.encoders.gaussian_ops",
        "vdw_radius_to_iso_logit",
    ),
    # 52-D compatible 3D ligand featurization (`ligand3d`)
    "LIGAND_3D_NODE_DIM": ("geneuclidean.models.encoders.ligand3d", "LIGAND_3D_NODE_DIM"),
    "collate_ligand_3d_batch": ("geneuclidean.models.encoders.ligand3d", "collate_ligand_3d_batch"),
    "featurize_ligand_3d": ("geneuclidean.models.encoders.ligand3d", "featurize_ligand_3d"),
    # Joint pocket-ligand 3D cross-attention (`complex_cross`)
    "SE3ComplexCrossAttention": (
        "geneuclidean.models.encoders.complex_cross",
        "SE3ComplexCrossAttention",
    ),
}

__all__ = [
    *sorted(_LAZY_ATTRS.keys()),
    "list_primitives",
]


def list_primitives() -> list[str]:
    """Return sorted names of all public 3D geometric operations without importing PyTorch."""
    return sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    """Lazily resolve 3D geometric operations on first attribute access (PEP 562)."""
    if name in _LAZY_ATTRS:
        mod_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose complete lazy 3D geometric library surface to ``dir(geneuclidean.geom)``."""
    return sorted(set(globals().keys()) | set(__all__))
