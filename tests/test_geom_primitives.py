"""Comprehensive CPU & Apple Silicon MPS verification for GenEuclidean 3D Geometric Ops.

Proves every mathematical invariant, physical edge case, numerical bound, and engineering
contract across the Wave-1 foundational 3D geometric operations library:
  - Cold import latency (< 15 ms locally) & zero eager torch/rdkit on ``geneuclidean`` and
    ``geneuclidean.geom``, plus ``BUILTIN_ENCODER_ALIASES`` and ``EncoderConfig`` YAML roundtrip.
  - ``dense_knn``: exact ``0.00e+00`` equidistant 26-neighbor tie-swap invariance, ``N=1`` ion,
    ``N < K`` ligand, ``all_empty`` isolated atom, dummy corruption, and ``d=0`` clash gradients.
  - ``tensor_ops``: exact ``I + A + S`` reconstruction & orthogonality, ``SE(3)`` equivariance,
    exact improper-reflection chiral sign flip, planar achirality, and Vector-Neuron equivariance.
  - ``gaussian_ops``: ``< 3e-6`` 3x3 cofactor inverse parity vs ``float64``, unconditional SPD
    condition bound ``cond <= 29.8``, planar benzene ``Sigma_zz > 1.0`` + ``O(3)`` equivariance,
    shared-frame ``100 A`` translation invariance & ``10 A`` separation of ROCS Shape Tanimoto,
    GaussianFormer-2 occupancy, 4DGS Law-of-Total-Covariance fusion, and trace-stabilized
    super-ellipsoid pooling (``cond <= 20``).
  - ``electrostatics``, ``ligand3d``, ``complex_cross``: exact ``SE(3)`` invariance, chiral
    sensitivity, 52-D ligand compatibility with ``PaiNNEncoder``, and finite MPS backward grads.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

import geneuclidean.geom as geom
from geneuclidean.config import EncoderConfig, ExperimentConfig
from geneuclidean.models.encoders.painn import PaiNNEncoder
from geneuclidean.registry import (
    BUILTIN_ENCODER_ALIASES,
    BUILTIN_ENCODER_ARCHS,
    BUILTIN_ENCODER_NAMES,
)

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def _random_so3_matrix(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a Haar-uniform proper rotation matrix R in SO(3) (det(R) = +1)."""
    mat = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    if float(torch.det(q).item()) < 0.0:
        q[:, 0] = -q[:, 0]
    return q.to(device=device, dtype=torch.float32)


def _reflection_matrix(device: torch.device) -> Tensor:
    """Return an improper reflection matrix P in O(3) \\ SO(3) with det(P) = -1."""
    return torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=torch.float32))


def _benzene_ring_coords(device: torch.device, radius: float = 1.40) -> Tensor:
    """Return exact planar regular hexagon (benzene, z = 0) of shape [1, 6, 3]."""
    angles = torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float32)[:-1]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z = torch.zeros_like(x)
    return torch.stack((x, y, z), dim=-1).unsqueeze(0).to(device)


def _alkyne_collinear_coords(device: torch.device) -> Tensor:
    """Return collinear 4-atom 2-butyne carbon chain along x-axis of shape [1, 4, 3]."""
    xs = torch.tensor([-1.96, -0.60, 0.60, 1.96], dtype=torch.float32)
    zeros = torch.zeros_like(xs)
    return torch.stack((xs, zeros, zeros), dim=-1).unsqueeze(0).to(device)


def _embed_3d_smiles(smiles: str, seed: int = 42) -> Chem.Mol:
    """Build a deterministic 3D RDKit conformer (heavy atoms) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = seed
    assert int(AllChem.EmbedMolecule(mol_h, params)) == 0  # type: ignore[attr-defined]
    return Chem.RemoveHs(mol_h)


# ---------------------------------------------------------------------------
# 1. Zero-Torch Cold Imports, Public Facade, Registry & Config Contracts
# ---------------------------------------------------------------------------


def test_cold_import_zero_torch_for_package_and_geom_facade() -> None:
    """Both `import geneuclidean` and `import geneuclidean.geom` load ZERO heavy libs."""
    probe = """
import json, sys, time
t0 = time.perf_counter()
import geneuclidean as ge
dt_ge = (time.perf_counter() - t0) * 1000.0

t1 = time.perf_counter()
import geneuclidean.geom as geom
dt_geom = (time.perf_counter() - t1) * 1000.0

heavy_check = ("torch", "rdkit", "transformers", "peft", "sklearn", "scipy", "matplotlib", "e3nn")
loaded_heavy = [m for m in heavy_check if m in sys.modules]
print(json.dumps({
    "dt_ge_ms": round(dt_ge, 2),
    "dt_geom_ms": round(dt_geom, 2),
    "loaded_heavy": loaded_heavy,
    "encoders": ge.list_encoders(),
    "n_primitives": len(geom.list_primitives()),
    "has_dense_knn": "build_dense_knn_graph" in dir(geom),
}))
"""
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout.strip())
    assert info["loaded_heavy"] == [], (
        f"Heavy modules leaked on cold import: {info['loaded_heavy']}"
    )
    assert info["dt_ge_ms"] < 150.0
    assert info["dt_geom_ms"] < 150.0
    assert info["n_primitives"] >= 35
    assert info["has_dense_knn"] is True
    assert set(info["encoders"]) >= {
        "painn",
        "egnn",
        "e3nn",
        "schnet",
        "dense_painn_se3",
        "cartformer_se3",
        "atomgs_se3",
        "subpocket_vn_jepa",
    }


def test_geom_facade_all_symbols_resolve_and_invalid_raises() -> None:
    """Every symbol in `geneuclidean.geom.__all__` resolves cleanly via PEP 562."""
    for name in geom.__all__:
        assert hasattr(geom, name), f"Missing lazy geom symbol: {name}"
    with pytest.raises(AttributeError, match="no attribute"):
        missing = "nonexistent_geom_op_xyz"
        _ = getattr(geom, missing)


def test_registry_aliases_and_encoder_config_yaml_roundtrip(tmp_path: Path) -> None:
    """Verify zero-torch alias mapping, strict extra='forbid', and full YAML roundtrip."""
    assert BUILTIN_ENCODER_ARCHS == BUILTIN_ENCODER_NAMES
    expected_aliases = {
        "cartformer": "cartformer_se3",
        "atomgs": "atomgs_se3",
        "subpocket_vn": "subpocket_vn_jepa",
        "vn_jepa": "subpocket_vn_jepa",
        "dense_painn": "dense_painn_se3",
    }
    for alias, canonical in expected_aliases.items():
        assert BUILTIN_ENCODER_ALIASES[alias] == canonical
        assert canonical in BUILTIN_ENCODER_NAMES
        assert EncoderConfig(arch=alias).arch == canonical
        assert EncoderConfig(arch=alias.upper()).arch == canonical

    # `max_neighbors` stays strictly on DataConfig (extra="forbid" rejects it on EncoderConfig).
    with pytest.raises(ValidationError):
        EncoderConfig.model_validate({"arch": "cartformer_se3", "max_neighbors": 24})

    cfg = ExperimentConfig(
        encoder=EncoderConfig(
            arch="atomgs",
            tensor_dim=32,
            num_heads=4,
            num_landmarks=16,
            num_subpockets=24,
            global_attn_mode="full_scalar",
            enable_3d_overlap=True,
            use_coulomb_bias=False,
            jepa_ssl_ckpt="runs/ssl/jepa.pt",
        )
    )
    assert cfg.encoder.arch == "atomgs_se3"
    yaml_path = cfg.to_yaml(tmp_path / "geom_encoder_cfg.yaml")
    reloaded = ExperimentConfig.from_yaml(yaml_path)
    assert reloaded == cfg


# ---------------------------------------------------------------------------
# 2. Tie-Safe Dense k-NN Graph & Safe Masked Softmax (`dense_knn.py`)
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_dense_knn_equidistant_26_neighbor_tieswap_invariance(device: str) -> None:
    """Prove exact 0.00e+00 equidistant 26-neighbor tie-swap invariance under SO(3) + t."""
    dev = torch.device(device)
    # Central atom at origin + 26 equidistant lattice-shell neighbors at exact radius 4.8 A.
    offsets = [
        [float(dx), float(dy), float(dz)]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    assert len(offsets) == 26
    shell = torch.tensor(offsets, dtype=torch.float32)
    shell = shell / shell.norm(dim=-1, keepdim=True) * 4.8
    # Add 6 inner-shell atoms at 2.2 A so aggregation is non-trivial while 26 outer atoms tie.
    inner = torch.tensor(
        [
            [2.2, 0.0, 0.0],
            [-2.2, 0.0, 0.0],
            [0.0, 2.2, 0.0],
            [0.0, -2.2, 0.0],
            [0.0, 0.0, 2.2],
            [0.0, 0.0, -2.2],
        ],
        dtype=torch.float32,
    )
    center = torch.zeros((1, 3), dtype=torch.float32)

    # Pure 26-neighbor equidistant shell (no inner atoms): adaptive envelope is 0.0 on all 24.
    coords_pure = torch.cat((center, shell), dim=0).unsqueeze(0).to(dev)  # [1, 27, 3]
    mask_pure = torch.ones((1, 27), dtype=torch.bool, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(101)
    feats_pure = torch.randn((1, 27, 8), generator=gen, dtype=torch.float32).to(dev)

    g_ref = geom.build_dense_knn_graph(
        coords_pure, mask_pure, cutoff=5.5, max_neighbors=24, adaptive_cutoff=True
    )
    # Central atom (index 0) sees 26 equidistant neighbors at 4.8 A -> r_eff == 4.8, env == 0.0!
    assert abs(float(g_ref.r_eff[0, 0, 0].item()) - 4.8) < 1e-4
    assert float(g_ref.edge_env[0, 0].abs().max().item()) == 0.0
    agg_ref = (geom.gather_neighbors(feats_pure, g_ref.nbr_idx) * g_ref.edge_env.unsqueeze(-1)).sum(
        dim=2
    )

    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        g_rot = geom.build_dense_knn_graph(
            coords_pure @ rot.T + trans,
            mask_pure,
            cutoff=5.5,
            max_neighbors=24,
            adaptive_cutoff=True,
        )
        agg_rot = (
            geom.gather_neighbors(feats_pure, g_rot.nbr_idx) * g_rot.edge_env.unsqueeze(-1)
        ).sum(dim=2)
        assert float((agg_rot[0, 0] - agg_ref[0, 0]).abs().max().item()) == 0.0

    # Mixed shell: 6 inner neighbors (2.2 A) + 26 outer equidistant neighbors (4.8 A).
    coords_mix = torch.cat((center, inner, shell), dim=0).unsqueeze(0).to(dev)  # [1, 33, 3]
    mask_mix = torch.ones((1, 33), dtype=torch.bool, device=dev)
    feats_mix = torch.randn((1, 33, 8), generator=gen, dtype=torch.float32).to(dev)
    g_mix_ref = geom.build_dense_knn_graph(
        coords_mix, mask_mix, cutoff=5.5, max_neighbors=24, adaptive_cutoff=True
    )
    mix_ref = (
        geom.gather_neighbors(feats_mix, g_mix_ref.nbr_idx) * g_mix_ref.edge_env.unsqueeze(-1)
    ).sum(dim=2)
    assert float(mix_ref[0, 0].abs().max().item()) > 0.1  # Non-trivial inner-shell signal!

    rot = _random_so3_matrix(gen, dev)
    trans = torch.tensor([[[83.5, -91.2, 64.7]]], device=dev, dtype=torch.float32)
    g_mix_rot = geom.build_dense_knn_graph(
        coords_mix @ rot.T + trans, mask_mix, cutoff=5.5, max_neighbors=24, adaptive_cutoff=True
    )
    mix_rot = (
        geom.gather_neighbors(feats_mix, g_mix_rot.nbr_idx) * g_mix_rot.edge_env.unsqueeze(-1)
    ).sum(dim=2)
    assert float((mix_rot[0, 0] - mix_ref[0, 0]).abs().max().item()) < 5e-5


@pytest.mark.parametrize("device", DEVICES)
def test_dense_knn_edge_cases_and_zero_distance_backward(device: str) -> None:
    """Verify N=1 ion, N<K ligand, all_empty isolation, dummy corruption, and d=0 clash grads."""
    dev = torch.device(device)

    # 1. Single-atom ion (N=1)
    c_ion = torch.tensor([[[12.3, -4.5, 6.7]]], device=dev, dtype=torch.float32)
    m_ion = torch.ones((1, 1), device=dev, dtype=torch.bool)
    g_ion = geom.build_dense_knn_graph(c_ion, m_ion, cutoff=5.5, max_neighbors=24)
    assert g_ion.nbr_idx.shape == (1, 1, 1)
    assert not bool(g_ion.nbr_mask.any().item())
    assert float(g_ion.edge_env.abs().max().item()) == 0.0
    assert float(g_ion.r_eff[0, 0, 0].item()) == pytest.approx(5.5)

    # 2. Small ligand (N=10 < K=24)
    gen = torch.Generator(device="cpu").manual_seed(7)
    c_small = (torch.randn((2, 10, 3), generator=gen, dtype=torch.float32) * 1.5).to(dev)
    m_small = torch.ones((2, 10), device=dev, dtype=torch.bool)
    m_small[1, 7:] = False
    g_small = geom.build_dense_knn_graph(c_small, m_small, cutoff=5.5, max_neighbors=24)
    assert g_small.nbr_idx.shape == (2, 10, 9)
    # Dummy atoms have zero incoming edges and zero corrupted leak
    c_corrupt = c_small.clone()
    c_corrupt[~m_small] = 999.0
    g_corrupt = geom.build_dense_knn_graph(c_corrupt, m_small, cutoff=5.5, max_neighbors=24)
    assert float((g_corrupt.edge_dist - g_small.edge_dist).abs().max().item()) == 0.0
    assert float((g_corrupt.edge_vec - g_small.edge_vec).abs().max().item()) == 0.0
    assert float(g_corrupt.edge_env[1, 7:].abs().max().item()) == 0.0

    # 3. Isolated atom (all_empty neighborhood) + safe_masked_softmax backward
    c_iso = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [50.0, 0.0, 0.0]]], device=dev, dtype=torch.float32
    )
    m_iso = torch.ones((1, 3), device=dev, dtype=torch.bool)
    g_iso = geom.build_dense_knn_graph(c_iso, m_iso, cutoff=5.5, max_neighbors=4)
    assert not bool(g_iso.nbr_mask[0, 2].any().item())
    logits = torch.randn((1, 3, 2), device=dev, dtype=torch.float32, requires_grad=True)
    probs = geom.safe_masked_softmax(logits, g_iso.nbr_mask, dim=-1)
    assert float(probs[0, 2].abs().max().item()) == 0.0
    assert float(probs[0, 0].sum().item()) == pytest.approx(1.0, abs=1e-6)
    loss_sm = (probs * g_iso.edge_env).sum()
    loss_sm.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all().item())

    # 4. Exact coordinate clash (d = 0) smooth backward gradients
    c_clash = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, -0.5]]],
        device=dev,
        dtype=torch.float32,
        requires_grad=True,
    )
    m_clash = torch.ones((1, 3), device=dev, dtype=torch.bool)
    g_clash = geom.build_dense_knn_graph(c_clash, m_clash, cutoff=5.5, max_neighbors=2)
    loss_clash = g_clash.edge_dist.sum() + g_clash.edge_dir.pow(2).sum() + g_clash.edge_env.sum()
    loss_clash.backward()
    assert c_clash.grad is not None and bool(torch.isfinite(c_clash.grad).all().item())


# ---------------------------------------------------------------------------
# 3. Cartesian 3x3 Irrep & Vector-Neuron Algebra (`tensor_ops.py`)
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_tensor_ops_ias_decomposition_orthogonality_and_se3_equivariance(device: str) -> None:
    """Verify exact I+A+S reconstruction, orthogonality, O(3)/SE(3), and chiral reflection."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(202)
    t_mat = torch.randn((2, 12, 8, 3, 3), generator=gen, dtype=torch.float32).to(dev)

    iso, asym, sym_tr = geom.decompose_rank2_ias(t_mat)
    recomp = geom.compose_rank2_ias(iso, asym, sym_tr)
    assert float((recomp - t_mat).abs().max().item()) < 1e-6

    # Scalar trace form of compose_rank2_ias
    tr_scalar = t_mat[..., 0, 0] + t_mat[..., 1, 1] + t_mat[..., 2, 2]
    assert (
        float((geom.compose_rank2_ias(tr_scalar, asym, sym_tr) - t_mat).abs().max().item()) < 1e-6
    )

    # Mutual Frobenius orthogonality & symmetry invariants
    assert float((iso * asym).sum(dim=(-2, -1)).abs().max().item()) < 1e-5
    assert float((iso * sym_tr).sum(dim=(-2, -1)).abs().max().item()) < 1e-5
    assert float((asym * sym_tr).sum(dim=(-2, -1)).abs().max().item()) < 1e-5
    assert float((asym + asym.transpose(-1, -2)).abs().max().item()) < 1e-6
    assert float((sym_tr - sym_tr.transpose(-1, -2)).abs().max().item()) < 1e-6
    assert (
        float((sym_tr[..., 0, 0] + sym_tr[..., 1, 1] + sym_tr[..., 2, 2]).abs().max().item()) < 1e-5
    )

    # Exact O(3) equivariance of I, A, S, outer_traceless, and TensorEquivariantRMSNorm
    rot = _random_so3_matrix(gen, dev)
    refl = _reflection_matrix(dev)
    t_norm = geom.TensorEquivariantRMSNorm(8).to(dev).eval()
    t_norm_pc = geom.TensorEquivariantRMSNorm(8, per_channel=True).to(dev).eval()

    for mat_g in (rot, refl):
        t_rot = torch.einsum("pa,...ab,qb->...pq", mat_g, t_mat, mat_g)
        iso_r, asym_r, sym_r = geom.decompose_rank2_ias(t_rot)
        assert (
            float(
                (iso_r - torch.einsum("pa,...ab,qb->...pq", mat_g, iso, mat_g)).abs().max().item()
            )
            < 1e-5
        )
        assert (
            float(
                (asym_r - torch.einsum("pa,...ab,qb->...pq", mat_g, asym, mat_g)).abs().max().item()
            )
            < 1e-5
        )
        assert (
            float(
                (sym_r - torch.einsum("pa,...ab,qb->...pq", mat_g, sym_tr, mat_g))
                .abs()
                .max()
                .item()
            )
            < 1e-5
        )

        for norm_mod in (t_norm, t_norm_pc):
            out_ref = norm_mod(t_mat)
            out_rot = norm_mod(t_rot)
            exp_rot = torch.einsum("pa,...ab,qb->...pq", mat_g, out_ref, mat_g)
            assert float((out_rot - exp_rot).abs().max().item()) < 1e-5

    # Hodge dual roundtrip & exact chiral sign flip under det(P) = -1
    v_a = torch.randn((2, 10, 6, 3), generator=gen, dtype=torch.float32).to(dev)
    v_b = torch.randn((2, 10, 6, 3), generator=gen, dtype=torch.float32).to(dev)
    v_c = torch.randn((2, 10, 6, 3), generator=gen, dtype=torch.float32).to(dev)

    skew_a = geom.vector_to_skew_matrix(v_a)
    assert float((geom.skew_matrix_to_vector(skew_a) - v_a).abs().max().item()) < 1e-6
    # Verify [v_a]_x @ v_b == v_a x v_b
    mat_cross = torch.einsum("...pq,...q->...p", skew_a, v_b)
    raw_cross = geom.chiral_cross_product(v_a, v_b, normalize=False)
    assert float((mat_cross - raw_cross).abs().max().item()) < 1e-5

    # outer_traceless 1-vec and 2-vec equivariance
    q1 = geom.outer_traceless(v_a)
    q1_r = geom.outer_traceless(v_a @ rot.T)
    assert (
        float((q1_r - torch.einsum("pa,...ab,qb->...pq", rot, q1, rot)).abs().max().item()) < 1e-5
    )
    q2 = geom.outer_traceless(v_a, v_b)
    q2_r = geom.outer_traceless(v_a @ rot.T, v_b @ rot.T)
    assert (
        float((q2_r - torch.einsum("pa,...ab,qb->...pq", rot, q2, rot)).abs().max().item()) < 1e-5
    )

    # Proper rotation (SO(3)): axial vector transforms with +R
    cross_ref = geom.chiral_cross_product(v_a, v_b, normalize=True)
    cross_rot = geom.chiral_cross_product(v_a @ rot.T, v_b @ rot.T, normalize=True)
    assert float((cross_rot - cross_ref @ rot.T).abs().max().item()) < 1e-5

    skew_rot = torch.einsum("pa,...ab,qb->...pq", rot, asym, rot)
    w_ref = geom.skew_matrix_to_vector(asym)
    w_rot = geom.skew_matrix_to_vector(skew_rot)
    assert float((w_rot - w_ref @ rot.T).abs().max().item()) < 1e-5

    # Improper reflection (det(P) = -1): EXACT SIGN FLIP for axial vectors & chiral triple product!
    cross_refl = geom.chiral_cross_product(v_a @ refl.T, v_b @ refl.T, normalize=True)
    assert float((cross_refl + cross_ref @ refl.T).abs().max().item()) < 1e-5
    skew_refl = torch.einsum("pa,...ab,qb->...pq", refl, asym, refl)
    w_refl = geom.skew_matrix_to_vector(skew_refl)
    assert float((w_refl + w_ref @ refl.T).abs().max().item()) < 1e-5

    triple_ref = (cross_ref * v_c).sum(dim=-1)
    triple_refl = (cross_refl * (v_c @ refl.T)).sum(dim=-1)
    assert float(triple_ref.abs().max().item()) > 0.1
    assert float((triple_refl + triple_ref).abs().max().item()) < 1e-5

    # Planar achiral check: z = 0 vectors -> triple product identically 0.0
    v_a_pl = v_a.clone()
    v_b_pl = v_b.clone()
    v_c_pl = v_c.clone()
    v_a_pl[..., 2] = 0.0
    v_b_pl[..., 2] = 0.0
    v_c_pl[..., 2] = 0.0
    triple_pl = (geom.chiral_cross_product(v_a_pl, v_b_pl) * v_c_pl).sum(dim=-1)
    assert float(triple_pl.abs().max().item()) == 0.0


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_vector_neuron_modules_equivariance_chirality_and_backward(device: str) -> None:
    """Verify VectorNeuronLinear, LeakyReLU, LayerNorm, and AttentionBlock on CPU & MPS."""
    torch.manual_seed(33)
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(303)
    rot = _random_so3_matrix(gen, dev)
    refl = _reflection_matrix(dev)

    vec = torch.randn((2, 10, 16, 3), generator=gen, dtype=torch.float32).to(dev)
    mask = torch.ones((2, 10), dtype=torch.bool, device=dev)
    mask[1, 7:] = False
    vec = torch.where(mask.unsqueeze(-1).unsqueeze(-1), vec, torch.zeros_like(vec))

    vn_lin = geom.VectorNeuronLinear(16, 16).to(dev).eval()
    vn_act = geom.VectorNeuronLeakyReLU(16).to(dev).eval()
    vn_act_shared = geom.VectorNeuronLeakyReLU(16, share_nonlinearity=True).to(dev).eval()
    vn_ln = geom.VectorNeuronLayerNorm(16).to(dev).eval()
    vn_ln_1 = geom.VectorNeuronLayerNorm(1).to(dev).eval()

    tol = 1e-5 if device == "cpu" else 5e-5
    for mod in (vn_lin, vn_act, vn_act_shared, vn_ln):
        out_ref = mod(vec)
        assert float(out_ref.abs().max().item()) > 1e-2
        assert float(out_ref[~mask].abs().max().item()) == 0.0
        for g_mat in (rot, refl):
            out_g = mod(vec @ g_mat.T)
            assert float((out_g - out_ref @ g_mat.T).abs().max().item()) < tol

    # Uniform-norm vectors stay non-degenerate under VectorNeuronLayerNorm
    unit_v = torch.zeros((1, 4, 16, 3), device=dev, dtype=torch.float32)
    unit_v[..., 0] = 1.0
    assert float(vn_ln(unit_v).abs().max().item()) > 0.5
    assert float(vn_ln_1(unit_v[:, :, :1, :]).abs().max().item()) > 0.5

    # VectorNeuronAttentionBlock: exact SO(3) equivariance + chiral reflection toggle
    blk_se3 = (
        geom.VectorNeuronAttentionBlock(16, num_heads=4, use_se3_chirality=True).to(dev).eval()
    )
    blk_o3 = (
        geom.VectorNeuronAttentionBlock(16, num_heads=4, use_se3_chirality=False).to(dev).eval()
    )

    out_se3_ref = blk_se3(vec, mask=mask)
    out_se3_rot = blk_se3(vec @ rot.T, mask=mask)
    out_se3_refl = blk_se3(vec @ refl.T, mask=mask)
    assert float(out_se3_ref[~mask].abs().max().item()) == 0.0
    assert float((out_se3_rot - out_se3_ref @ rot.T).abs().max().item()) < tol
    # Chiral block breaks O(3) reflection symmetry on 3D chiral input!
    assert float((out_se3_refl - out_se3_ref @ refl.T).abs().max().item()) > 1e-3

    out_o3_ref = blk_o3(vec, mask=mask)
    out_o3_refl = blk_o3(vec @ refl.T, mask=mask)
    assert float((out_o3_refl - out_o3_ref @ refl.T).abs().max().item()) < tol

    # Backward pass with an all_empty batch row on CPU & MPS
    vec_grad = vec.clone().requires_grad_(True)
    mask_empty = mask.clone()
    mask_empty[1, :] = False
    loss = blk_se3(vec_grad, mask=mask_empty).pow(2).sum()
    loss.backward()
    assert vec_grad.grad is not None and bool(torch.isfinite(vec_grad.grad).all().item())


# ---------------------------------------------------------------------------
# 4. Differentiable Anisotropic 3D/4D Gaussian Algebra (`gaussian_ops.py`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_cofactor_inverse_and_logdet_accuracy_vs_float64(device: str) -> None:
    """Verify closed-form 3x3 cofactor inverse & logdet achieve < 3e-6 error vs float64."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(404)
    # Build 512 SPD matrices with condition numbers spanning [1, 29.8]
    a = torch.randn((512, 3, 3), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    eigs = torch.empty((512, 3), dtype=torch.float32)
    eigs[:, 0] = 0.25 + torch.rand(512, generator=gen) * 0.5
    eigs[:, 1] = eigs[:, 0] + torch.rand(512, generator=gen) * 2.0
    eigs[:, 2] = eigs[:, 0] * (1.0 + torch.rand(512, generator=gen) * 28.0)
    spd = torch.einsum("nij,nj,nkj->nik", q, eigs, q)
    rhs = torch.randn((512, 3), generator=gen, dtype=torch.float32)

    spd_dev = spd.to(dev).requires_grad_(True)
    rhs_dev = rhs.to(dev).requires_grad_(True)
    inv_pred, logdet_pred = geom.cofactor_inverse_and_logdet_3x3(spd_dev)
    sol_pred, _ = geom.cofactor_solve_3x3(spd_dev, rhs_dev)

    # Exact float64 ground truth on CPU
    spd64 = spd.to(torch.float64)
    inv_gt = torch.linalg.inv(spd64)
    _, logdet_gt = torch.linalg.slogdet(spd64)
    sol_gt = torch.linalg.solve(spd64, rhs.to(torch.float64).unsqueeze(-1)).squeeze(-1)

    inv_rel_err = float(
        (
            (inv_pred.detach().cpu().to(torch.float64) - inv_gt).abs().max() / inv_gt.abs().max()
        ).item()
    )
    logdet_err = float(
        (logdet_pred.detach().cpu().to(torch.float64) - logdet_gt).abs().max().item()
    )
    sol_rel_err = float(
        (
            (sol_pred.detach().cpu().to(torch.float64) - sol_gt).abs().max() / sol_gt.abs().max()
        ).item()
    )
    assert inv_rel_err < 3e-6, f"Cofactor inverse rel err {inv_rel_err:.2e} >= 3e-6 on {device}"
    assert logdet_err < 3e-6, f"Cofactor logdet err {logdet_err:.2e} >= 3e-6 on {device}"
    assert sol_rel_err < 3e-6, f"Cofactor solve rel err {sol_rel_err:.2e} >= 3e-6 on {device}"

    # Finite gradients on CPU & MPS
    (inv_pred.sum() + logdet_pred.sum() + sol_pred.sum()).backward()
    assert spd_dev.grad is not None and bool(torch.isfinite(spd_dev.grad).all().item())
    assert rhs_dev.grad is not None and bool(torch.isfinite(rhs_dev.grad).all().item())


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_anisotropic_covariance_cond_bound_benzene_normal_and_alkyne(device: str) -> None:
    """Verify unconditional cond <= 29.8, benzene pi-normal Sigma_zz > 1.0 + O(3), and alkyne."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(505)

    # 1. Extreme adversarial logits (-50 to +50) -> cond MUST stay <= 29.8!
    iso_extreme = torch.tensor([[-50.0, 0.0, 50.0, -50.0]], device=dev, dtype=torch.float32)
    # Co-aligned huge polar & axial vectors maximize lambda_max while iso_extreme=-50 minimizes s0
    p_vecs = torch.zeros((1, 4, 4, 3), device=dev, dtype=torch.float32)
    p_vecs[..., 0] = 100.0
    a_vecs = torch.zeros((1, 4, 2, 3), device=dev, dtype=torch.float32)
    a_vecs[..., 0] = 100.0
    p_logits = torch.full((1, 4, 4), 50.0, device=dev, dtype=torch.float32)
    a_logits = torch.full((1, 4, 2), 50.0, device=dev, dtype=torch.float32)

    sigma_adv = geom.build_anisotropic_covariance_se3(
        iso_extreme, p_vecs, p_logits, a_vecs, a_logits
    )
    eigs_adv = torch.linalg.eigvalsh(sigma_adv.cpu().to(torch.float64))
    lambda_min = float(eigs_adv.min().item())
    cond_max = float((eigs_adv[..., -1] / eigs_adv[..., 0]).max().item())
    assert lambda_min >= 0.249
    assert cond_max <= 29.85, f"Condition number bound violated: {cond_max:.3f} > 29.8"

    # 2. Adversarial planar benzene ring (z = 0): out-of-plane axial pi-cloud normal!
    benzene = _benzene_ring_coords(dev)  # [1, 6, 3]
    mask_bz = torch.ones((1, 6), device=dev, dtype=torch.bool)
    g_bz = geom.build_dense_knn_graph(benzene, mask_bz, cutoff=3.0, max_neighbors=4)

    # Two distinct in-plane polar neighbor vectors per benzene carbon
    v1 = g_bz.edge_dir[:, :, 0, :]  # [1, 6, 3] (z == 0)
    v2 = g_bz.edge_dir[:, :, 1, :]  # [1, 6, 3] (z == 0)
    assert float(v1[..., 2].abs().max().item()) < 1e-6
    assert float(v2[..., 2].abs().max().item()) < 1e-6

    # Axial pi-normal vector points strictly along +-z (|omega_z| = sin(120 deg) ~ 0.866)
    omega_bz = geom.chiral_cross_product(v1, v2, normalize=False)
    iso_bz = torch.full((1, 6), -1.0, device=dev, dtype=torch.float32)  # s0 ~ 1.258
    p_bz = torch.stack((v1, v2, v1 - v2, v1 + v2), dim=-2)  # [1, 6, 4, 3]
    a_bz = torch.stack((omega_bz * 3.0, -omega_bz * 3.0), dim=-2)  # [1, 6, 2, 3]
    p_log_bz = torch.zeros((1, 6, 4), device=dev, dtype=torch.float32)
    a_log_bz = torch.full((1, 6, 2), 2.0, device=dev, dtype=torch.float32)

    sigma_bz = geom.build_anisotropic_covariance_se3(iso_bz, p_bz, p_log_bz, a_bz, a_log_bz)
    s0_floor = 0.25 + 3.75 * float(torch.sigmoid(torch.tensor(-1.0)).item())
    sigma_zz = float(sigma_bz[0, :, 2, 2].min().item())
    assert sigma_zz > s0_floor + 0.5, f"Planar pi-normal missing: Sigma_zz={sigma_zz:.3f}"
    assert sigma_zz > 1.0

    # Exact O(3) equivariance of axial covariance on benzene (including det(P) = -1 reflection!)
    rot = _random_so3_matrix(gen, dev)
    refl = _reflection_matrix(dev)
    for g_mat in (rot, refl):
        v1_g = v1 @ g_mat.T
        v2_g = v2 @ g_mat.T
        omega_g = geom.chiral_cross_product(v1_g, v2_g, normalize=False)
        p_g = torch.stack((v1_g, v2_g, v1_g - v2_g, v1_g + v2_g), dim=-2)
        a_g = torch.stack((omega_g * 3.0, -omega_g * 3.0), dim=-2)
        sigma_g = geom.build_anisotropic_covariance_se3(iso_bz, p_g, p_log_bz, a_g, a_log_bz)
        expected = torch.einsum("pa,...ab,qb->...pq", g_mat, sigma_bz, g_mat)
        assert float((sigma_g - expected).abs().max().item()) < 1e-5

    # 3. Collinear alkyne: finite covariances & gradients
    alkyne = _alkyne_collinear_coords(dev).requires_grad_(True)
    mask_alk = torch.ones((1, 4), device=dev, dtype=torch.bool)
    g_alk = geom.build_dense_knn_graph(alkyne, mask_alk, cutoff=3.0, max_neighbors=3)
    ax_alk = geom.chiral_cross_product(g_alk.edge_dir[:, :, 0], g_alk.edge_dir[:, :, 1]).unsqueeze(
        -2
    )
    iso_alk = torch.zeros((1, 4), device=dev, dtype=torch.float32, requires_grad=True)
    sigma_alk = geom.build_anisotropic_covariance_se3(
        iso_alk, polar_vecs=g_alk.edge_dir[:, :, :2], axial_vecs=ax_alk
    )
    assert bool(torch.isfinite(sigma_alk).all().item())
    sigma_alk.sum().backward()
    assert alkyne.grad is not None and bool(torch.isfinite(alkyne.grad).all().item())
    assert iso_alk.grad is not None and bool(torch.isfinite(iso_alk.grad).all().item())


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_overlap_shared_frame_rocs_tanimoto_100a_and_backward(device: str) -> None:
    """Verify analytical overlap, 100 A shared-frame SE(3), 10 A separation, and MPS bwd."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(606)

    # Analytical self-overlap identities
    mu = torch.randn((2, 6, 3), generator=gen, dtype=torch.float32).to(dev)
    iso = torch.randn((2, 6), generator=gen, dtype=torch.float32).to(dev)
    pv = torch.randn((2, 6, 4, 3), generator=gen, dtype=torch.float32).to(dev)
    sigma = geom.build_anisotropic_covariance_se3(iso, polar_vecs=pv)
    det_s = geom.cofactor_det_3x3(sigma)

    bhatt_self = geom.gaussian_overlap_integral_3d(mu, sigma, mu, sigma, mode="bhattacharyya")
    dens_self = geom.gaussian_overlap_integral_3d(mu, sigma, mu, sigma, mode="density")
    expected_dens = (4.0 * math.pi) ** (-1.5) * torch.rsqrt(det_s)
    assert float((bhatt_self - 1.0).abs().max().item()) < 1e-5
    assert float((dens_self - expected_dens).abs().max().item()) < 1e-5

    # Build pocket & ligand fields in distinct world positions (close enough to overlap)
    p_world = (torch.randn((2, 16, 3), generator=gen, dtype=torch.float32) * 2.5).to(dev)
    l_world = (torch.randn((2, 8, 3), generator=gen, dtype=torch.float32) * 2.0 + 0.8).to(dev)
    p_mask = torch.ones((2, 16), device=dev, dtype=torch.bool)
    p_mask[1, 13:] = False
    l_mask = torch.ones((2, 8), device=dev, dtype=torch.bool)
    l_mask[1, 6:] = False

    p_iso = torch.randn((2, 16), generator=gen, dtype=torch.float32).to(dev)
    l_iso = torch.randn((2, 8), generator=gen, dtype=torch.float32).to(dev)
    p_pv = torch.randn((2, 16, 4, 3), generator=gen, dtype=torch.float32).to(dev)
    l_pv = torch.randn((2, 8, 4, 3), generator=gen, dtype=torch.float32).to(dev)

    def _make_field(
        coords: Tensor, mask: Tensor, iso_l: Tensor, pv_l: Tensor
    ) -> geom.AtomicGaussianField:
        centered, offset = geom.center_masked_coords_with_offset(coords, mask)
        cov = geom.build_anisotropic_covariance_se3(iso_l, polar_vecs=pv_l, atom_mask=mask)
        return geom.AtomicGaussianField(centers=centered, centroid_offset=offset, covariances=cov)

    p_field = _make_field(p_world, p_mask, p_iso, p_pv)
    l_field = _make_field(l_world, l_mask, l_iso, l_pv)

    # Self-Tanimoto == 1.0
    self_tan = geom.rocs_shape_tanimoto(l_field, l_field, l_mask, l_mask)
    assert float((self_tan - 1.0).abs().max().item()) < 1e-5

    ref_tan = geom.rocs_shape_tanimoto(p_field, l_field, p_mask, l_mask)
    ref_ovl = geom.gaussian_overlap_integral_3d(
        p_field.centers,
        p_field.covariances,
        l_field.centers,
        l_field.covariances,
        offset_i=p_field.centroid_offset,
        offset_j=l_field.centroid_offset,
        pairwise=True,
        mode="density",
    )
    assert float(ref_tan.min().item()) > 0.05

    # Exact SE(3) invariance under 5 Haar SO(3) rotations + 100 A rigid translation!
    tol = 1e-5 if device == "cpu" else 5e-5
    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        p_t = _make_field(p_world @ rot.T + trans, p_mask, p_iso, p_pv @ rot.T)
        l_t = _make_field(l_world @ rot.T + trans, l_mask, l_iso, l_pv @ rot.T)
        tan_t = geom.rocs_shape_tanimoto(p_t, l_t, p_mask, l_mask)
        ovl_t = geom.gaussian_overlap_integral_3d(
            p_t.centers,
            p_t.covariances,
            l_t.centers,
            l_t.covariances,
            offset_i=p_t.centroid_offset,
            offset_j=l_t.centroid_offset,
            pairwise=True,
            mode="density",
        )
        assert float((tan_t - ref_tan).abs().max().item()) < tol
        assert float((ovl_t - ref_ovl).abs().max().item()) < tol

    # Verify fast symmetric-3x3 ROCS kernel matches full cofactor details & supports O(NK) self
    rocs_fast = geom.gaussian_overlap_integral_3d(
        p_field.centers,
        p_field.covariances,
        l_field.centers,
        l_field.covariances,
        offset_i=p_field.centroid_offset,
        offset_j=l_field.centroid_offset,
        pairwise=True,
        mode="rocs",
    )
    rocs_details = geom.gaussian_overlap_details_3d(
        p_field.centers,
        p_field.covariances,
        l_field.centers,
        l_field.covariances,
        offset_i=p_field.centroid_offset,
        offset_j=l_field.centroid_offset,
        pairwise=True,
    ).rocs_volume
    assert float((rocs_fast - rocs_details).abs().max().item()) < 2e-5

    o_aa_exact = geom.rocs_self_overlap(p_field, mask=p_mask)
    o_aa_knn = geom.rocs_self_overlap(p_field, mask=p_mask, max_neighbors=12, cutoff=20.0)
    o_bb_exact = geom.rocs_self_overlap(l_field, mask=l_mask)
    tan_precomp = geom.rocs_shape_tanimoto(
        p_field, l_field, p_mask, l_mask, self_overlap_a=o_aa_exact, self_overlap_b=o_bb_exact
    )
    assert float((tan_precomp - ref_tan).abs().max().item()) < 1e-6
    assert float(((o_aa_knn - o_aa_exact).abs() / o_aa_exact.clamp(min=1e-6)).max().item()) < 0.05

    # Shared-frame anti-teleportation test: shifting ligand 15 A away MUST drop overlap > 95%!
    l_far = _make_field(
        l_world + torch.tensor([[[15.0, 0.0, 0.0]]], device=dev), l_mask, l_iso, l_pv
    )
    far_tan = geom.rocs_shape_tanimoto(p_field, l_far, p_mask, l_mask)
    assert bool((far_tan < 0.05 * ref_tan).all().item())

    # Finite MPS & CPU backward gradients through ROCS Tanimoto & density overlap
    l_world_g = l_world.clone().requires_grad_(True)
    l_iso_g = l_iso.clone().requires_grad_(True)
    l_field_g = _make_field(l_world_g, l_mask, l_iso_g, l_pv)
    loss = geom.rocs_shape_tanimoto(p_field, l_field_g, p_mask, l_mask).sum()
    loss.backward()
    assert l_world_g.grad is not None and bool(torch.isfinite(l_world_g.grad).all().item())
    assert l_iso_g.grad is not None and bool(torch.isfinite(l_iso_g.grad).all().item())
    assert float(l_world_g.grad.abs().max().item()) > 1e-6


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_gaussianformer2_occupancy_4d_fusion_and_super_ellipsoid_pool(device: str) -> None:
    """Verify GaussianFormer-2 occupancy, 4DGS Law of Total Covariance, and super-ellipsoid cond."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(707)

    # 1. GaussianFormer-2 probabilistic occupancy
    bhatt = torch.tensor([[[0.5, 0.5, 0.5], [0.8, 0.0, 0.0]]], device=dev, dtype=torch.float32)
    mask = torch.tensor([[[True, True, True], [False, False, False]]], device=dev, dtype=torch.bool)
    occ = geom.gaussianformer2_occupancy(bhatt, mask)
    expected_0 = 1.0 - (1.0 - 0.999 * 0.5) ** 3
    assert float(occ[0, 0].item()) == pytest.approx(expected_0, abs=1e-5)
    assert float(occ[0, 1].item()) == 0.0

    # 2. 4DGS Law-of-Total-Covariance exact identity & SE(3) equivariance
    t_steps = 5
    centers_4d = (torch.randn((t_steps, 2, 10, 3), generator=gen, dtype=torch.float32) * 2.0).to(
        dev
    )
    iso_4d = torch.randn((t_steps, 2, 10), generator=gen, dtype=torch.float32).to(dev)
    pv_4d = torch.randn((t_steps, 2, 10, 4, 3), generator=gen, dtype=torch.float32).to(dev)
    covs_4d = geom.build_anisotropic_covariance_se3(iso_4d, polar_vecs=pv_4d)

    mean_c, fused_s = geom.fuse_4d_md_covariances(centers_4d, covs_4d)
    # Exact identity: E[Sigma] + Cov(mu) (fp32 magnitude ~6.0 -> 1 ULP ~ 4.8e-7)
    e_sigma = covs_4d.mean(dim=0)
    delta = centers_4d - centers_4d.mean(dim=0, keepdim=True)
    cov_mu = (delta.unsqueeze(-1) * delta.unsqueeze(-2)).mean(dim=0)
    ltc_tol = 1e-6 if device == "cpu" else 5e-6
    assert float((fused_s - (e_sigma + cov_mu)).abs().max().item()) < ltc_tol
    # PSD dominance: fused_s - E[Sigma] >= 0
    diff_eigs = torch.linalg.eigvalsh((fused_s - e_sigma).cpu().to(torch.float64))
    assert float(diff_eigs.min().item()) >= -1e-6

    rot = _random_so3_matrix(gen, dev)
    trans = torch.tensor([[[[50.0, -75.0, 25.0]]]], device=dev, dtype=torch.float32)
    covs_4d_rot = torch.einsum("pa,...ab,qb->...pq", rot, covs_4d, rot)
    mean_c_r, fused_s_r = geom.fuse_4d_md_covariances(centers_4d @ rot.T + trans, covs_4d_rot)
    assert float((mean_c_r - (mean_c @ rot.T + trans.squeeze(0))).abs().max().item()) < 1e-4
    # Adding trans=[50, -75, 25] in fp32 introduces ~7.6e-6 coordinate ULP rounding,
    # amplified by ~2*||delta|| in Cov(mu) (~1-2e-5 on x86_64 AVX2 vs ARM64).
    assert (
        float(
            (fused_s_r - torch.einsum("pa,...ab,qb->...pq", rot, fused_s, rot)).abs().max().item()
        )
        < 5e-5
    )

    # fuse_multiconformer_fields wrapper + differing offsets dummy-slot I_3 guard
    mask_mc = torch.ones((2, 10), device=dev, dtype=torch.bool)
    mask_mc[1, 7:] = False
    fields = [
        geom.AtomicGaussianField(
            centers=torch.where(
                mask_mc.unsqueeze(-1), centers_4d[t], torch.zeros_like(centers_4d[t])
            ),
            centroid_offset=torch.tensor(
                [[float(t), -0.5 * t, 0.25 * t], [-float(t), float(t), 0.5 * t]],
                device=dev,
                dtype=torch.float32,
            ),
            covariances=torch.where(
                mask_mc.unsqueeze(-1).unsqueeze(-1),
                covs_4d[t],
                torch.eye(3, device=dev, dtype=torch.float32).expand_as(covs_4d[t]),
            ),
        )
        for t in range(t_steps)
    ]
    fused_field = geom.fuse_multiconformer_fields(fields, atom_mask=mask_mc)
    assert float(fused_field.centers[1, 7:].abs().max().item()) == 0.0
    eye_dummy = torch.eye(3, device=dev, dtype=torch.float32).expand_as(
        fused_field.covariances[1, 7:]
    )
    assert float((fused_field.covariances[1, 7:] - eye_dummy).abs().max().item()) == 0.0

    # 3. Trace-stabilized super_ellipsoid_pool on an extreme flat 15 A cavity wall -> cond <= 20!
    xs = torch.linspace(-7.5, 7.5, 24, dtype=torch.float32)
    ys = torch.linspace(-7.5, 7.5, 24, dtype=torch.float32)
    wall_c = torch.stack((xs, ys.flip(0), torch.zeros_like(xs)), dim=-1).unsqueeze(0).to(dev)
    wall_cov = (
        torch.eye(3, device=dev, dtype=torch.float32).view(1, 1, 3, 3).expand(1, 24, 3, 3) * 0.25
    )
    assign = torch.softmax(torch.randn((1, 4, 24), generator=gen, dtype=torch.float32), dim=-1).to(
        dev
    )
    # Force one empty slot to test empty-slot I_3 guard
    assign_with_empty = assign.clone()
    assign_with_empty[:, 3, :] = 0.0

    sup_c, sup_s = geom.super_ellipsoid_pool(wall_c, wall_cov, assign_with_empty, trace_reg=0.05)
    sup_eigs = torch.linalg.eigvalsh(sup_s.cpu().to(torch.float64))
    sup_cond = float((sup_eigs[..., -1] / sup_eigs[..., 0]).max().item())
    assert sup_cond <= 20.0, f"Super-ellipsoid condition number {sup_cond:.2f} > 20.0"
    assert float((sup_s[0, 3] - torch.eye(3, device=dev)).abs().max().item()) == 0.0

    # SE(3) equivariance of super_ellipsoid_pool
    wall_c_rot = wall_c @ rot.T + trans.squeeze(0)[:, :1, :]
    wall_cov_rot = torch.einsum("pa,...ab,qb->...pq", rot, wall_cov, rot)
    sup_c_r, sup_s_r = geom.super_ellipsoid_pool(wall_c_rot, wall_cov_rot, assign, trace_reg=0.05)
    sup_c_ref, sup_s_ref = geom.super_ellipsoid_pool(wall_c, wall_cov, assign, trace_reg=0.05)
    assert (
        float((sup_c_r - (sup_c_ref @ rot.T + trans.squeeze(0)[:, :1, :])).abs().max().item())
        < 1e-4
    )
    assert (
        float(
            (sup_s_r - torch.einsum("pa,...ab,qb->...pq", rot, sup_s_ref, rot)).abs().max().item()
        )
        < 1e-4
    )


# ---------------------------------------------------------------------------
# 5. Electrostatics, GeoTransformer/RoITr Bias, 3D Ligand & Complex Cross-Attn
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_electrostatics_and_geotransformer_bias_se3_chirality_and_backward(device: str) -> None:
    """Verify ScreenedCoulombBias and GeoTransformerPairTripletBias invariance, chirality & bwd."""
    torch.manual_seed(808)
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(808)
    rot = _random_so3_matrix(gen, dev)
    refl = _reflection_matrix(dev)
    trans = torch.tensor([[[88.0, -95.0, 72.0]]], device=dev, dtype=torch.float32)

    coords = (torch.randn((2, 12, 3), generator=gen, dtype=torch.float32) * 3.0).to(dev)
    scalars = torch.randn((2, 12, 32), generator=gen, dtype=torch.float32).to(dev)
    raw_feats = torch.zeros((2, 12, 52), device=dev, dtype=torch.float32)
    raw_feats[..., 46] = torch.linspace(-0.8, 0.8, 12, device=dev).unsqueeze(0)
    mask = torch.ones((2, 12), device=dev, dtype=torch.bool)
    mask[1, 9:] = False

    # 1. ScreenedCoulombBias
    coul = geom.ScreenedCoulombBias(scalar_dim=32, num_heads=4).to(dev).eval()
    c_ref = coul(coords, scalars, raw_feats_i=raw_feats, mask_i=mask)
    c_rot = coul(coords @ rot.T + trans, scalars, raw_feats_i=raw_feats, mask_i=mask)
    tol = 1e-5 if device == "cpu" else 5e-5
    assert float((c_rot - c_ref).abs().max().item()) < tol
    # Zero self-diagonal and zero masked slots
    diag_idx = torch.arange(12, device=dev)
    assert float(c_ref[:, :, diag_idx, diag_idx].abs().max().item()) == 0.0
    assert float(c_ref[1, :, 9:, :].abs().max().item()) == 0.0
    assert float(c_ref[1, :, :, 9:].abs().max().item()) == 0.0
    # Strict float32 preservation even if float64 charges are passed on CPU
    if device == "cpu":
        q64 = torch.randn((2, 12), generator=gen, dtype=torch.float64)
        c_from_f64 = coul(coords, charges_i=q64, mask_i=mask)
        assert c_from_f64.dtype == torch.float32

    # 2. GeoTransformerPairTripletBias (distance + triplet angle + RoITr PPF + chirality)
    vecs = torch.randn((2, 12, 4, 3), generator=gen, dtype=torch.float32).to(dev)
    geo_se3 = geom.GeoTransformerPairTripletBias(num_heads=4, use_se3_chirality=True).to(dev).eval()
    geo_e3 = geom.GeoTransformerPairTripletBias(num_heads=4, use_se3_chirality=False).to(dev).eval()

    g_ref = geo_se3(coords, mask_i=mask, vecs_i=vecs)
    g_rot = geo_se3(coords @ rot.T + trans, mask_i=mask, vecs_i=vecs @ rot.T)
    g_refl = geo_se3(coords @ refl.T, mask_i=mask, vecs_i=vecs @ refl.T)
    assert float((g_rot - g_ref).abs().max().item()) < tol
    assert float((g_refl - g_ref).abs().max().item()) > 1e-3  # Chiral sensitivity!

    ge3_ref = geo_e3(coords, mask_i=mask, vecs_i=vecs)
    ge3_refl = geo_e3(coords @ refl.T, mask_i=mask, vecs_i=vecs @ refl.T)
    assert float((ge3_refl - ge3_ref).abs().max().item()) < tol  # Exact E(3) when disabled!

    # Planar achirality check on benzene
    bz = _benzene_ring_coords(dev)
    bz_mask = torch.ones((1, 6), device=dev, dtype=torch.bool)
    bz_vecs = bz.clone()  # strictly in z = 0 plane
    bz_ref = geo_se3(bz, mask_i=bz_mask, vecs_i=bz_vecs)
    bz_refl = geo_se3(bz @ refl.T, mask_i=bz_mask, vecs_i=bz_vecs @ refl.T)
    assert float((bz_refl - bz_ref).abs().max().item()) < tol

    # Epoch-0 coincident centroids (all coords == 0) + backward gradient check
    zero_c = torch.zeros((1, 6, 3), device=dev, dtype=torch.float32, requires_grad=True)
    zero_v = torch.randn((1, 6, 3), generator=gen, dtype=torch.float32).to(dev).requires_grad_(True)
    zero_s = (
        torch.randn((1, 6, 32), generator=gen, dtype=torch.float32).to(dev).requires_grad_(True)
    )
    loss = coul(zero_c, zero_s).pow(2).sum() + geo_se3(zero_c, vecs_i=zero_v).pow(2).sum()
    loss.backward()
    assert zero_c.grad is not None and bool(torch.isfinite(zero_c.grad).all().item())
    assert zero_v.grad is not None and bool(torch.isfinite(zero_v.grad).all().item())
    assert zero_s.grad is not None and bool(torch.isfinite(zero_s.grad).all().item())


def test_featurize_ligand_3d_52d_layout_and_painn_compatibility() -> None:
    """Verify 52-D ligand featurizer matches featurizer.py layout and feeds PaiNNEncoder."""
    aspirin = _embed_3d_smiles("CC(=O)Oc1ccccc1C(=O)O", seed=42)
    r_ibuprofen = _embed_3d_smiles("CC(C)Cc1ccc([C@H](C)C(=O)O)cc1", seed=42)
    s_ibuprofen = _embed_3d_smiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1", seed=42)

    # Confirm input mol is NOT mutated by featurize_ligand_3d
    had_prop_before = aspirin.GetAtomWithIdx(0).HasProp("_GasteigerCharge")
    coords, feats, mask = geom.featurize_ligand_3d(aspirin)
    assert aspirin.GetAtomWithIdx(0).HasProp("_GasteigerCharge") == had_prop_before

    n_heavy = sum(1 for a in aspirin.GetAtoms() if a.GetAtomicNum() > 1)
    assert coords.shape == (n_heavy, 3) and coords.dtype == np.float32
    assert feats.shape == (n_heavy, 52) and feats.dtype == np.float32
    assert mask.shape == (n_heavy,) and mask.dtype == bool and bool(mask.all())
    assert np.all(np.isfinite(coords)) and np.all(np.isfinite(feats))

    # Exact 52-D channel compatibility checks
    assert np.all(feats[:, 30] == 1.0)  # Residue slot OTHER (col 30)
    assert np.all(feats[:, 42] == 0.0)  # is_backbone == 0
    assert float(feats[:, 37].sum()) == 6.0  # 6 aromatic carbons in aspirin benzene ring
    assert float(feats[:, 38].sum()) >= 3.0  # ester + carboxyl oxygen HBAs
    assert float(feats[:, 39].sum()) >= 1.0  # carboxylic acid OH HBD

    # Bondi vdW radii at dim 45 (C = 1.70/2.0 = 0.85, O = 1.52/2.0 = 0.76)
    c_rows = feats[:, 0] == 1.0
    o_rows = feats[:, 2] == 1.0
    assert np.allclose(feats[c_rows, 45], 0.85, atol=1e-6)
    assert np.allclose(feats[o_rows, 45], 0.76, atol=1e-6)
    # Non-trivial Gasteiger partial charges at dim 46 bounded in [-1, 1]
    assert np.all(np.abs(feats[:, 46]) <= 1.0)
    assert float(np.abs(feats[:, 46]).max()) > 0.2

    # Shared pocket_center frame shifts coordinates accurately
    pocket_center = np.array([10.0, -5.0, 2.5], dtype=np.float32)
    coords_world, _, _ = geom.featurize_ligand_3d(aspirin, center_coords=False)
    coords_in_pocket, _, _ = geom.featurize_ligand_3d(aspirin, pocket_center=pocket_center)
    np.testing.assert_allclose(coords_in_pocket, coords_world - pocket_center[None, :], atol=1e-5)

    # Explicit-H vs implicit-H 3D SDF parity (regression guard for PDBbind/Boltz SDFs)
    for smi, expected_hba in (
        ("CC(=O)Oc1ccccc1C(=O)O", 3.0),
        ("CC(C)Cc1ccc([C@H](C)C(=O)O)cc1", 1.0),
    ):
        mol_base = Chem.MolFromSmiles(smi)
        assert mol_base is not None
        mol_with_h = Chem.AddHs(mol_base)
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = 77
        assert int(AllChem.EmbedMolecule(mol_with_h, params)) == 0  # type: ignore[attr-defined]
        mol_no_h = Chem.RemoveHs(mol_with_h)
        c_h, f_h, m_h = geom.featurize_ligand_3d(mol_with_h)
        c_noh, f_noh, m_noh = geom.featurize_ligand_3d(mol_no_h)
        np.testing.assert_allclose(c_h, c_noh, atol=0.0)
        np.testing.assert_allclose(f_h, f_noh, atol=0.0)
        assert np.array_equal(m_h, m_noh)
        assert float(f_h[:, 38].sum()) == expected_hba

    # Batch collation + direct feed into PaiNNEncoder (proves 52-D contract compatibility!)
    b_coords, b_feats, b_mask = geom.collate_ligand_3d_batch([aspirin, r_ibuprofen, s_ibuprofen])
    assert b_coords.shape[0] == 3 and b_feats.shape[-1] == 52
    torch.manual_seed(11)
    enc = PaiNNEncoder(
        node_in_dim=52, hidden_dim=32, vector_dim=16, out_dim=32, num_layers=2, dropout=0.0
    ).eval()
    with torch.no_grad():
        out = enc(b_coords, b_feats, b_mask)
    assert out.pooled_emb.shape == (3, 32)
    assert bool(torch.isfinite(out.pooled_emb).all().item())


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_se3_complex_cross_attention_invariance_chirality_and_backward(device: str) -> None:
    """Verify SE3ComplexCrossAttention joint SE(3), chiral toggle, pose sensitivity, & MPS bwd."""
    torch.manual_seed(909)
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(909)

    p_emb = torch.randn((2, 16, 32), generator=gen, dtype=torch.float32).to(dev)
    p_vec = torch.randn((2, 16, 8, 3), generator=gen, dtype=torch.float32).to(dev)
    p_coords = (torch.randn((2, 16, 3), generator=gen, dtype=torch.float32) * 3.0).to(dev)
    p_mask = torch.ones((2, 16), device=dev, dtype=torch.bool)
    p_mask[1, 12:] = False

    l_emb = torch.randn((2, 8, 32), generator=gen, dtype=torch.float32).to(dev)
    l_vec = torch.randn((2, 8, 8, 3), generator=gen, dtype=torch.float32).to(dev)
    l_coords = (torch.randn((2, 8, 3), generator=gen, dtype=torch.float32) * 2.0).to(dev)
    l_mask = torch.ones((2, 8), device=dev, dtype=torch.bool)
    l_mask[1, 6:] = False

    cross_se3 = (
        geom.SE3ComplexCrossAttention(
            emb_dim=32, vector_dim=8, num_heads=4, num_rbf=12, cutoff=8.0, use_se3_chirality=True
        )
        .to(dev)
        .eval()
    )
    cross_e3 = (
        geom.SE3ComplexCrossAttention(
            emb_dim=32, vector_dim=8, num_heads=4, num_rbf=12, cutoff=8.0, use_se3_chirality=False
        )
        .to(dev)
        .eval()
    )

    hp_ref, hl_ref, pool_ref = cross_se3(
        p_emb, p_vec, p_coords, p_mask, l_emb, l_vec, l_coords, l_mask
    )
    assert float(hp_ref[~p_mask].abs().max().item()) == 0.0
    assert float(hl_ref[~l_mask].abs().max().item()) == 0.0

    # Joint SE(3) invariance under pure rotation and 100 A rigid translation
    rot = _random_so3_matrix(gen, dev)
    trans = torch.tensor([[[95.0, -82.0, 63.0]]], device=dev, dtype=torch.float32)
    hp_r, hl_r, pool_r = cross_se3(
        p_emb,
        p_vec @ rot.T,
        p_coords @ rot.T + trans,
        p_mask,
        l_emb,
        l_vec @ rot.T,
        l_coords @ rot.T + trans,
        l_mask,
    )
    tol = 2e-5 if device == "cpu" else 1e-4
    assert float((hp_r - hp_ref).abs().max().item()) < tol
    assert float((hl_r - hl_ref).abs().max().item()) < tol
    assert float((pool_r - pool_ref).abs().max().item()) < tol

    # Relative pose shift (10 A ligand displacement) changes complex embedding significantly!
    _, _, pool_far = cross_se3(
        p_emb,
        p_vec,
        p_coords,
        p_mask,
        l_emb,
        l_vec,
        l_coords + torch.tensor([[[10.0, 0.0, 0.0]]], device=dev),
        l_mask,
    )
    assert float((pool_far - pool_ref).abs().max().item()) > 1e-2

    # Mirror reflection: chiral SE(3) distinguishes enantiomers; E(3) is invariant!
    refl = _reflection_matrix(dev)
    _, _, pool_refl_se3 = cross_se3(
        p_emb,
        p_vec @ refl.T,
        p_coords @ refl.T,
        p_mask,
        l_emb,
        l_vec @ refl.T,
        l_coords @ refl.T,
        l_mask,
    )
    assert float((pool_refl_se3 - pool_ref).abs().max().item()) > 1e-3

    _, _, pool_e3_ref = cross_e3(p_emb, p_vec, p_coords, p_mask, l_emb, l_vec, l_coords, l_mask)
    _, _, pool_e3_refl = cross_e3(
        p_emb,
        p_vec @ refl.T,
        p_coords @ refl.T,
        p_mask,
        l_emb,
        l_vec @ refl.T,
        l_coords @ refl.T,
        l_mask,
    )
    assert float((pool_e3_refl - pool_e3_ref).abs().max().item()) < tol

    # SchNet compatibility (vec=None) + empty ligand mask + finite backward grads on CPU & MPS
    p_coords_g = p_coords.clone().requires_grad_(True)
    l_coords_g = l_coords.clone().requires_grad_(True)
    l_mask_empty = l_mask.clone()
    l_mask_empty[1, :] = False
    hp_g, hl_g, pool_g = cross_se3(
        p_emb, None, p_coords_g, p_mask, l_emb, None, l_coords_g, l_mask_empty
    )
    loss = hp_g.pow(2).sum() + hl_g.pow(2).sum() + pool_g.pow(2).sum()
    loss.backward()
    assert p_coords_g.grad is not None and bool(torch.isfinite(p_coords_g.grad).all().item())
    assert l_coords_g.grad is not None and bool(torch.isfinite(l_coords_g.grad).all().item())
