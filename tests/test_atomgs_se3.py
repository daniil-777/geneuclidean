"""Comprehensive CPU & Apple Silicon MPS verification suite for ``AtomGS-SE3``.

Verifies every mathematical invariant, physical prior, edge case, dual-mode pocket-ligand
3D complementarity contract, 4D multi-conformer fusion property, backward gradient check,
and Apple M3 Pro MPS performance target for :mod:`geneuclidean.models.encoders.atomgs`
under ``PYTORCH_ENABLE_MPS_FALLBACK=0``:

- Exact ``SO(3)`` rotation + ``100 A`` rigid translation invariance/equivariance
  (``pooled_emb`` / ``node_emb`` / ``vec_emb`` / ``covariances`` err ``< 5e-5``) across
  both ``global_attn_mode="landmarks"`` and ``"full_scalar"``.
- Atom permutation invariance and exact zero dummy-atom corruption sensitivity.
- Unconditional SPD condition bound ``cond(covariances) <= 29.8 < 30``, Bondi vdW
  (``dim 45``) initialization responsiveness, and exact parity between the fused symmetric
  ``3x3`` local cofactor kernel and ``gaussian_overlap_details_3d``.
- Chiral reflection (``det(P) = -1``) discrimination on chiral pockets and equal-bond-length
  ``(R)/(S)`` tetrahedra (``diff > 1e-3`` for ``SE(3)`` vs ``< 5e-5`` for ``E(3)`` toggle).
- Planar benzene ring & collinear alkyne achiral consistency, finite gradients, exact
  ``O(3)`` covariance equivariance, and strong out-of-plane ``Sigma_zz > 1.0`` axial
  :math:`\\pi`-cloud normal on planar benzene.
- Shared-frame ``score_pocket_ligand_overlap`` & ``encode_complex`` (with optional
  ``SE3ComplexCrossAttention``): joint ``SO(3) + 100 A`` invariance (``< 5e-5``), sharp
  ``+5 A`` / ``+10 A`` separation drop (proving zero centroid teleportation), steric clash
  spike at ``0.3 A`` penetration, and parity with ``gaussian_ops.rocs_shape_tanimoto``.
- ``4DGS`` multi-conformer fusion (``fuse_ensemble_4d`` / ``encode_ensemble_4d``):
  ``SE(3)`` equivariance, Law-of-Total-Covariance PSD dominance, and dummy-slot safety.
- Edge cases: ``N=1`` single-atom ion, ``N < K`` small ligand, ``all_empty`` isolated atom,
  fully masked batch row, ``d=0`` clash gradients, and equidistant 26-neighbor tie-swap
  invariance with distinct features.
- Empirical Apple M3 Pro MPS timings (``B=16, N=384, K=24`` fwd/fwd+bwd and
  ``score_pocket_ligand_overlap`` ``B=16, N_P=384, N_L=40`` latency).
"""

from __future__ import annotations

import math
import time
from typing import Literal

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from geneuclidean.config import EncoderConfig
from geneuclidean.models.encoders.atomgs import (
    AtomGSSE3Encoder,
    _build_flat_nbr_idx,
    _local_sym3x3_gaussian_overlaps,
    build_atomgs_se3,
    fuse_ensemble_4d,
)
from geneuclidean.models.encoders.dense_knn import build_dense_knn_graph, gather_neighbors
from geneuclidean.models.encoders.gaussian_ops import (
    build_anisotropic_covariance_se3,
    gaussian_overlap_details_3d,
    rocs_self_overlap,
    rocs_shape_tanimoto,
)
from geneuclidean.models.encoders.ligand3d import featurize_ligand_3d
from geneuclidean.protocols import PocketEncoderProtocol

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _random_so3_matrix(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a Haar-uniform proper rotation matrix R in SO(3) (det(R) = +1)."""
    mat = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    if float(torch.det(q).item()) < 0.0:
        q[:, 0] = -q[:, 0]
    out: Tensor = q.to(device=device, dtype=torch.float32)
    return out


def _reflection_matrix(device: torch.device) -> Tensor:
    """Return improper mirror reflection P = diag(-1, 1, 1) with det(P) = -1."""
    return torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=torch.float32))


def _rotate_cov(cov: Tensor, mat: Tensor) -> Tensor:
    """Compute ``mat @ cov @ mat.T`` for arbitrary leading dimensions ``[..., 3, 3]``."""
    return torch.einsum("pa,...ab,qb->...pq", mat, cov, mat)


def _make_realistic_pocket_batch(
    batch_size: int = 2,
    num_atoms: int = 32,
    node_in_dim: int = 52,
    device: torch.device | str = "cpu",
    seed: int = 123,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a realistic chiral 3D pocket batch with valid vdW (dim 45) & charge (dim 46)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32) * 2.2
    node_feats = torch.randn(
        (batch_size, num_atoms, node_in_dim), generator=gen, dtype=torch.float32
    )
    if node_in_dim > 46:
        node_feats[..., 45] = 0.72 + 0.25 * torch.rand(
            (batch_size, num_atoms), generator=gen, dtype=torch.float32
        )
        node_feats[..., 46] = (
            torch.rand((batch_size, num_atoms), generator=gen, dtype=torch.float32) * 1.2 - 0.6
        )

    atom_mask = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
    for b in range(batch_size):
        valid_count = max(8, num_atoms - 5 - 3 * b)
        atom_mask[b, :valid_count] = True

    dev = torch.device(device)
    return coords.to(dev), node_feats.to(dev), atom_mask.to(dev)


def _make_small_atomgs(
    *,
    global_attn_mode: Literal["landmarks", "full_scalar"] = "landmarks",
    use_se3_chirality: bool = True,
    enable_complex_cross_attn: bool = False,
    num_landmarks: int = 8,
    num_layers: int = 2,
) -> AtomGSSE3Encoder:
    return AtomGSSE3Encoder(
        node_in_dim=52,
        hidden_dim=64,
        vector_dim=16,
        out_dim=64,
        num_layers=num_layers,
        num_heads=4,
        num_rbf=16,
        cutoff=5.5,
        max_neighbors=24,
        num_landmarks=num_landmarks,
        global_attn_mode=global_attn_mode,
        use_se3_chirality=use_se3_chirality,
        enable_3d_overlap=True,
        use_coulomb_bias=True,
        enable_complex_cross_attn=enable_complex_cross_attn,
        dropout=0.0,
    )


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
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = seed
    assert int(AllChem.EmbedMolecule(mol_h, params)) == 0  # type: ignore[attr-defined]
    return Chem.RemoveHs(mol_h)


# ---------------------------------------------------------------------------
# 1. Protocol Compliance, Factory, Zero-Float64, Parity & Condition Bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("global_attn_mode", ["landmarks", "full_scalar"])
def test_protocol_factory_and_zero_float64(
    global_attn_mode: Literal["landmarks", "full_scalar"],
) -> None:
    """Verify PocketEncoderProtocol, build_atomgs_se3 factory, and zero float64 params/buffers."""
    cfg = EncoderConfig(
        arch="atomgs",
        hidden_dim=64,
        vector_dim=16,
        out_dim=64,
        num_layers=2,
        num_heads=4,
        num_rbf=16,
        num_landmarks=8,
        global_attn_mode=global_attn_mode,
        enable_3d_overlap=True,
        use_coulomb_bias=True,
        dropout=0.0,
    )
    assert cfg.arch == "atomgs_se3"
    enc = build_atomgs_se3(cfg, max_neighbors=24)
    assert isinstance(enc, AtomGSSE3Encoder)
    assert isinstance(enc, PocketEncoderProtocol)

    for name, param in enc.named_parameters():
        assert param.dtype == torch.float32, f"Non-float32 param {name}: {param.dtype}"
    for name, buf in enc.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer found: {name}"

    enc_cross = _make_small_atomgs(enable_complex_cross_attn=True)
    for name, param in enc_cross.named_parameters():
        assert param.dtype == torch.float32, f"Non-float32 param {name}: {param.dtype}"
    for name, buf in enc_cross.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer found: {name}"


@pytest.mark.parametrize("device", DEVICES)
def test_local_sym6_overlap_kernel_parity_with_gaussian_overlap_details(device: str) -> None:
    """Verify fused symmetric-3x3 local overlap kernel matches gaussian_overlap_details_3d."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(33)
    coords = (torch.randn((2, 24, 3), generator=gen, dtype=torch.float32) * 2.0).to(dev)
    mask = torch.ones((2, 24), dtype=torch.bool, device=dev)
    mask[1, 19:] = False
    iso = torch.randn((2, 24), generator=gen, dtype=torch.float32).to(dev)
    p_vecs = torch.randn((2, 24, 4, 3), generator=gen, dtype=torch.float32).to(dev)
    a_vecs = torch.randn((2, 24, 2, 3), generator=gen, dtype=torch.float32).to(dev)
    covs = build_anisotropic_covariance_se3(iso, polar_vecs=p_vecs, axial_vecs=a_vecs)

    g = build_dense_knn_graph(coords, mask, cutoff=5.5, max_neighbors=12)
    flat_idx = _build_flat_nbr_idx(g.nbr_idx)
    bhatt, rocs, mahal, ldiv, fvec = _local_sym3x3_gaussian_overlaps(
        covs, g.edge_vec, flat_idx, 24, g.nbr_idx.shape[2]
    )
    nbr_c = gather_neighbors(g.centered_coords, g.nbr_idx)
    nbr_s = gather_neighbors(covs, g.nbr_idx)
    ref = gaussian_overlap_details_3d(
        g.centered_coords.unsqueeze(2), covs.unsqueeze(2), nbr_c, nbr_s
    )

    m = g.nbr_mask
    assert float((bhatt[m] - ref.bhattacharyya[m]).abs().max().item()) < 2e-6
    assert float((rocs[m] - ref.rocs_volume[m]).abs().max().item()) < 2e-5
    assert float((mahal[m] - ref.mahalanobis_dist[m]).abs().max().item()) < 2e-6
    assert float((ldiv[m] - ref.logdet_divergence[m]).abs().max().item()) < 2e-6
    assert float((fvec[m] - ref.force_vec[m]).abs().max().item()) < 2e-6


@pytest.mark.parametrize("device", DEVICES)
def test_covariance_condition_bound_and_vdw_prior(device: str) -> None:
    """Verify unconditional lambda_min >= 0.25, cond <= 29.8, and Bondi vdW dim-45 prior."""
    torch.manual_seed(101)
    dev = torch.device(device)
    enc = _make_small_atomgs().to(dev).eval()
    coords, feats, mask = _make_realistic_pocket_batch(batch_size=2, num_atoms=28, device=dev)

    with torch.no_grad():
        field = enc.encode_gaussian_field(coords, feats, mask)

    valid_covs = field.covariances[mask].detach().cpu().to(torch.float64)
    sym_err = float((valid_covs - valid_covs.transpose(-1, -2)).abs().max().item())
    assert sym_err < 1e-6

    eigs = torch.linalg.eigvalsh(valid_covs)
    lambda_min = float(eigs.min().item())
    cond_max = float((eigs[..., -1] / eigs[..., 0]).max().item())
    assert lambda_min >= 0.249, f"lambda_min={lambda_min:.4f} < 0.25"
    assert cond_max <= 29.85, f"cond_max={cond_max:.3f} > 29.8"

    dummy_covs = field.covariances[~mask]
    eye = torch.eye(3, device=dev, dtype=torch.float32).expand_as(dummy_covs)
    assert float((dummy_covs - eye).abs().max().item()) == 0.0
    assert float(field.centers[~mask].abs().max().item()) == 0.0

    feats_small_vdw = feats.clone()
    feats_large_vdw = feats.clone()
    feats_small_vdw[..., 45] = 0.65
    feats_large_vdw[..., 45] = 1.25
    with torch.no_grad():
        cov_small = enc.encode_gaussian_field(coords, feats_small_vdw, mask).covariances[mask]
        cov_large = enc.encode_gaussian_field(coords, feats_large_vdw, mask).covariances[mask]
    tr_small = float(cov_small.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item())
    tr_large = float(cov_large.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item())
    assert tr_large > tr_small + 0.5, (
        f"vdW dim-45 prior did not scale covariance trace: {tr_small:.3f} vs {tr_large:.3f}"
    )


# ---------------------------------------------------------------------------
# 2. Exact SO(3) + 100 A Translation, Permutation & Dummy Corruption
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("global_attn_mode", ["landmarks", "full_scalar"])
@pytest.mark.parametrize("device", DEVICES)
def test_so3_rotation_and_100a_translation_invariance_equivariance(
    global_attn_mode: Literal["landmarks", "full_scalar"],
    device: str,
) -> None:
    """Verify exact SO(3) + 100 A translation invariance/equivariance (< 5e-5) on CPU & MPS."""
    torch.manual_seed(42)
    dev = torch.device(device)
    enc = _make_small_atomgs(global_attn_mode=global_attn_mode).to(dev).eval()
    coords, node_feats, atom_mask = _make_realistic_pocket_batch(device=dev, seed=7)

    with torch.no_grad():
        ref_out, ref_field = enc.encode_with_field(coords, node_feats, atom_mask)

        pert_out = enc(coords + torch.randn_like(coords) * 1.2, node_feats, atom_mask)
        assert float((pert_out.pooled_emb - ref_out.pooled_emb).abs().max().item()) > 1e-2

    tol = 5e-5
    gen = torch.Generator(device="cpu").manual_seed(2026)

    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        coords_t = coords @ rot.T + trans

        with torch.no_grad():
            out_t, field_t = enc.encode_with_field(coords_t, node_feats, atom_mask)

        node_err = float((out_t.node_emb - ref_out.node_emb).abs().max().item())
        pool_err = float((out_t.pooled_emb - ref_out.pooled_emb).abs().max().item())
        assert ref_out.vec_emb is not None and out_t.vec_emb is not None
        assert float(ref_out.vec_emb.abs().max().item()) > 1e-2
        vec_err = float((out_t.vec_emb - ref_out.vec_emb @ rot.T).abs().max().item())

        expected_cov = _rotate_cov(ref_field.covariances, rot)
        cov_err = float(
            (field_t.covariances[atom_mask] - expected_cov[atom_mask]).abs().max().item()
        )

        assert ref_field.dipole_vecs is not None and field_t.dipole_vecs is not None
        dip_err = float((field_t.dipole_vecs - ref_field.dipole_vecs @ rot.T).abs().max().item())

        assert ref_field.axial_vecs is not None and field_t.axial_vecs is not None
        ax_err = float((field_t.axial_vecs - ref_field.axial_vecs @ rot.T).abs().max().item())

        assert node_err < tol, f"{global_attn_mode} ({device}) node_emb err={node_err:.2e}"
        assert pool_err < tol, f"{global_attn_mode} ({device}) pooled_emb err={pool_err:.2e}"
        assert vec_err < tol, f"{global_attn_mode} ({device}) vec_emb err={vec_err:.2e}"
        assert cov_err < tol, f"{global_attn_mode} ({device}) cov err={cov_err:.2e}"
        assert dip_err < tol, f"{global_attn_mode} ({device}) dipole err={dip_err:.2e}"
        assert ax_err < tol, f"{global_attn_mode} ({device}) axial err={ax_err:.2e}"


@pytest.mark.equivariance
@pytest.mark.parametrize("global_attn_mode", ["landmarks", "full_scalar"])
@pytest.mark.parametrize("device", DEVICES)
def test_permutation_and_dummy_corruption_invariance(
    global_attn_mode: Literal["landmarks", "full_scalar"],
    device: str,
) -> None:
    """Verify exact atom permutation invariance and zero dummy-atom corruption sensitivity."""
    torch.manual_seed(17)
    dev = torch.device(device)
    enc = _make_small_atomgs(global_attn_mode=global_attn_mode).to(dev).eval()
    coords, node_feats, atom_mask = _make_realistic_pocket_batch(device=dev, seed=19)

    gen = torch.Generator(device="cpu").manual_seed(303)
    perm = torch.randperm(coords.shape[1], generator=gen).to(dev)

    with torch.no_grad():
        ref_out, ref_field = enc.encode_with_field(coords, node_feats, atom_mask)
        perm_out, perm_field = enc.encode_with_field(
            coords[:, perm], node_feats[:, perm], atom_mask[:, perm]
        )

    assert float((perm_out.pooled_emb - ref_out.pooled_emb).abs().max().item()) < 1e-5
    assert float((perm_out.node_emb - ref_out.node_emb[:, perm]).abs().max().item()) < 1e-5
    assert ref_out.vec_emb is not None and perm_out.vec_emb is not None
    assert float((perm_out.vec_emb - ref_out.vec_emb[:, perm]).abs().max().item()) < 1e-5
    assert (
        float((perm_field.covariances - ref_field.covariances[:, perm]).abs().max().item()) < 1e-5
    )

    coords_bad = coords.clone()
    feats_bad = node_feats.clone()
    coords_bad[~atom_mask] = 999.0
    feats_bad[~atom_mask] = torch.randn_like(feats_bad[~atom_mask]) * 1e4

    with torch.no_grad():
        bad_out, bad_field = enc.encode_with_field(coords_bad, feats_bad, atom_mask)

    assert float(bad_out.node_emb[~atom_mask].abs().max().item()) == 0.0
    assert bad_out.vec_emb is not None
    assert float(bad_out.vec_emb[~atom_mask].abs().max().item()) == 0.0
    assert float(bad_field.centers[~atom_mask].abs().max().item()) == 0.0
    assert float((bad_out.pooled_emb - ref_out.pooled_emb).abs().max().item()) < 5e-6
    assert float((bad_out.node_emb - ref_out.node_emb).abs().max().item()) < 5e-6
    assert float((bad_field.covariances - ref_field.covariances).abs().max().item()) < 5e-6


# ---------------------------------------------------------------------------
# 3. Chiral Reflection (R)/(S) Discrimination & Planar Benzene / Collinear Alkyne
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_chiral_tetrahedron_and_pocket_reflection_discrimination(device: str) -> None:
    """Verify SE(3) chirality detects mirror reflections while E(3) toggle preserves invariance."""
    torch.manual_seed(31)
    dev = torch.device(device)
    reflect = _reflection_matrix(dev)

    # Regular equal-bond-length chiral tetrahedron C(F)(Cl)(Br)(I) with 4 distinct substituents
    tet = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
        device=dev,
    )
    tet = (tet / math.sqrt(3.0) * 1.54).unsqueeze(0)  # [1, 5, 3]
    tet_mirror = tet @ reflect.T

    gen = torch.Generator(device="cpu").manual_seed(77)
    tet_feats = torch.randn((1, 5, 52), generator=gen, dtype=torch.float32).to(dev)
    tet_feats[0, :, 45] = torch.tensor([0.85, 0.74, 0.88, 0.93, 0.99], device=dev)
    tet_mask = torch.ones((1, 5), dtype=torch.bool, device=dev)

    enc_se3 = _make_small_atomgs(use_se3_chirality=True).to(dev).eval()
    enc_e3 = _make_small_atomgs(use_se3_chirality=False).to(dev).eval()

    with torch.no_grad():
        out_r = enc_se3(tet, tet_feats, tet_mask)
        out_s = enc_se3(tet_mirror, tet_feats, tet_mask)
        out_r_e3 = enc_e3(tet, tet_feats, tet_mask)
        out_s_e3 = enc_e3(tet_mirror, tet_feats, tet_mask)

    tet_se3_diff = float((out_r.pooled_emb - out_s.pooled_emb).abs().max().item())
    tet_e3_diff = float((out_r_e3.pooled_emb - out_s_e3.pooled_emb).abs().max().item())
    assert tet_se3_diff > 1e-3, f"Chiral tetrahedron SE(3) diff too small: {tet_se3_diff:.2e}"
    assert tet_e3_diff < 5e-5, f"E(3) toggle leaked mirror asymmetry: {tet_e3_diff:.2e}"

    p_coords, p_feats, p_mask = _make_realistic_pocket_batch(device=dev, seed=37)
    with torch.no_grad():
        p_l = enc_se3(p_coords, p_feats, p_mask)
        p_d = enc_se3(p_coords @ reflect.T, p_feats, p_mask)
        p_l_e3 = enc_e3(p_coords, p_feats, p_mask)
        p_d_e3 = enc_e3(p_coords @ reflect.T, p_feats, p_mask)
    assert float((p_l.pooled_emb - p_d.pooled_emb).abs().max().item()) > 1e-3
    assert float((p_l_e3.pooled_emb - p_d_e3.pooled_emb).abs().max().item()) < 5e-5


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_planar_benzene_pi_normal_and_collinear_alkyne_achirality(device: str) -> None:
    """Verify benzene Sigma_zz > 1.0 axial pi-normal, O(3) equivariance & achiral consistency."""
    torch.manual_seed(55)
    dev = torch.device(device)
    enc = _make_small_atomgs(use_se3_chirality=True).to(dev).eval()
    reflect = _reflection_matrix(dev)

    # 1. Exact D6h symmetric planar benzene using real 52-D ligand features (all 6 C identical)
    bz_coords = _benzene_ring_coords(dev)
    bz_mol = _embed_3d_smiles("c1ccccc1", seed=42)
    _, bz_feats_np, bz_mask_np = featurize_ligand_3d(bz_mol)
    bz_feats = torch.from_numpy(bz_feats_np).unsqueeze(0).to(dev)
    bz_mask = torch.from_numpy(bz_mask_np).unsqueeze(0).to(dev)

    bz_grad_coords = bz_coords.clone().requires_grad_(True)
    out_bz, field_bz = enc.encode_with_field(bz_grad_coords, bz_feats, bz_mask)

    sigma_zz = float(field_bz.covariances[0, :, 2, 2].min().item())
    assert sigma_zz > 1.0, f"Planar benzene axial pi-normal missing: Sigma_zz={sigma_zz:.3f}"
    sigma_xz_yz = float(field_bz.covariances[0, :, :2, 2].abs().max().item())
    assert sigma_xz_yz < 1e-5, f"Unexpected xz/yz coupling on planar benzene: {sigma_xz_yz:.2e}"

    torch.autograd.backward(out_bz.pooled_emb.pow(2).sum() + field_bz.covariances.sum())
    assert bz_grad_coords.grad is not None
    assert bool(torch.isfinite(bz_grad_coords.grad).all().item())

    with torch.no_grad():
        out_bz_ref, field_bz_ref = enc.encode_with_field(bz_coords, bz_feats, bz_mask)
        out_bz_refl, field_bz_refl = enc.encode_with_field(bz_coords @ reflect.T, bz_feats, bz_mask)

    assert float((out_bz_refl.pooled_emb - out_bz_ref.pooled_emb).abs().max().item()) < 5e-5
    assert float((out_bz_refl.node_emb - out_bz_ref.node_emb).abs().max().item()) < 5e-5
    expected_cov_refl = _rotate_cov(field_bz_ref.covariances, reflect)
    assert float((field_bz_refl.covariances - expected_cov_refl).abs().max().item()) < 5e-5

    # 2. Collinear 2-butyne alkyne: achiral consistency + finite forward & backward grads
    alk_coords = _alkyne_collinear_coords(dev).requires_grad_(True)
    gen = torch.Generator(device="cpu").manual_seed(88)
    alk_feats = torch.randn((1, 4, 52), generator=gen, dtype=torch.float32).to(dev)
    alk_feats[0, :, 45] = 0.85
    alk_mask = torch.ones((1, 4), dtype=torch.bool, device=dev)

    out_alk, field_alk = enc.encode_with_field(alk_coords, alk_feats, alk_mask)
    assert bool(torch.isfinite(out_alk.pooled_emb).all().item())
    assert bool(torch.isfinite(field_alk.covariances).all().item())
    torch.autograd.backward(out_alk.pooled_emb.pow(2).sum() + field_alk.covariances.sum())
    assert alk_coords.grad is not None and bool(torch.isfinite(alk_coords.grad).all().item())

    with torch.no_grad():
        alk_ref = enc(alk_coords.detach(), alk_feats, alk_mask)
        alk_refl = enc(alk_coords.detach() @ reflect.T, alk_feats, alk_mask)
    assert float((alk_refl.pooled_emb - alk_ref.pooled_emb).abs().max().item()) < 5e-5


# ---------------------------------------------------------------------------
# 4. Shared-Frame Pocket-Ligand Complementarity & Complex Cross-Attention
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_shared_frame_pocket_ligand_overlap_and_no_teleportation(device: str) -> None:
    """Verify shared-frame SE(3) invariance, +5 A / +10 A separation, clash spike & ROCS parity."""
    torch.manual_seed(64)
    dev = torch.device(device)
    enc = _make_small_atomgs(enable_complex_cross_attn=True).to(dev).eval()

    # Compact 3D pocket centered near the origin + real RDKit 3D ibuprofen docked inside
    p_coords, p_feats, p_mask = _make_realistic_pocket_batch(
        batch_size=2, num_atoms=36, device=dev, seed=91
    )
    p_coords = p_coords * 0.9

    mol_r = _embed_3d_smiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1", seed=11)
    mol_s = _embed_3d_smiles("CC(C)Cc1ccc([C@H](C)C(=O)O)cc1", seed=12)
    c_r, f_r, m_r = featurize_ligand_3d(mol_r, center_coords=True)
    c_s, f_s, m_s = featurize_ligand_3d(mol_s, center_coords=True)
    assert c_r.shape == c_s.shape
    l_coords = torch.stack((torch.from_numpy(c_r), torch.from_numpy(c_s)), dim=0).to(dev)
    l_feats = torch.stack((torch.from_numpy(f_r), torch.from_numpy(f_s)), dim=0).to(dev)
    l_mask = torch.stack((torch.from_numpy(m_r), torch.from_numpy(m_s)), dim=0).to(dev)

    with torch.no_grad():
        ref_cx = enc.encode_complex(p_coords, p_feats, p_mask, l_coords, l_feats, l_mask)

    rocs_ref = rocs_shape_tanimoto(
        ref_cx.pocket_field,
        ref_cx.ligand_field,
        p_mask,
        l_mask,
        max_self_neighbors=32,
        self_cutoff=12.0,
    )
    rocs_parity = float(
        (ref_cx.overlap_scores["rocs_shape_tanimoto"] - rocs_ref).abs().max().item()
    )
    assert rocs_parity < 1e-5
    assert float(ref_cx.overlap_scores["rocs_shape_tanimoto"].min().item()) > 0.05

    # 1. Exact joint SO(3) + 100 A rigid-motion invariance of pocket + ligand together
    gen = torch.Generator(device="cpu").manual_seed(909)
    tol = 5e-5
    for _ in range(3):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        with torch.no_grad():
            cx_t = enc.encode_complex(
                p_coords @ rot.T + trans,
                p_feats,
                p_mask,
                l_coords @ rot.T + trans,
                l_feats,
                l_mask,
            )
        for key, val in ref_cx.overlap_scores.items():
            diff = float((cx_t.overlap_scores[key] - val).abs().max().item())
            assert diff < tol, f"Joint SE(3) error on {key} ({device}): {diff:.2e}"
        cx_emb_diff = float((cx_t.complex_emb - ref_cx.complex_emb).abs().max().item())
        assert cx_emb_diff < tol, f"Joint SE(3) error on complex_emb ({device}): {cx_emb_diff:.2e}"

    # 2. Strict sensitivity when shifting ONLY the ligand +5 A and +10 A away (NO teleportation!)
    shift_5a = torch.tensor([[[5.0, 0.0, 0.0]]], device=dev, dtype=torch.float32)
    shift_10a = torch.tensor([[[10.0, 0.0, 0.0]]], device=dev, dtype=torch.float32)
    with torch.no_grad():
        l_field_5a = enc.encode_gaussian_field(l_coords + shift_5a, l_feats, l_mask)
        l_field_10a = enc.encode_gaussian_field(l_coords + shift_10a, l_feats, l_mask)
        scores_5a = enc.score_pocket_ligand_overlap(ref_cx.pocket_field, p_mask, l_field_5a, l_mask)
        scores_10a = enc.score_pocket_ligand_overlap(
            ref_cx.pocket_field, p_mask, l_field_10a, l_mask
        )

    rocs_0 = ref_cx.overlap_scores["rocs_shape_tanimoto"]
    rocs_5 = scores_5a["rocs_shape_tanimoto"]
    rocs_10 = scores_10a["rocs_shape_tanimoto"]
    assert bool((rocs_5 < rocs_0 * 0.5).all().item()), (
        f"+5 A shift failed to drop ROCS overlap: {rocs_0.tolist()} vs {rocs_5.tolist()}"
    )
    assert bool((rocs_10 < rocs_0 * 0.05).all().item()), (
        f"+10 A shift failed >95% ROCS drop: {rocs_0.tolist()} vs {rocs_10.tolist()}"
    )
    total_shift_diff = float(
        (scores_5a["total_overlap_score"] - ref_cx.overlap_scores["total_overlap_score"])
        .abs()
        .min()
        .item()
    )
    assert total_shift_diff > 1e-2

    # 3. Steric clash spike when ligand atoms directly penetrate pocket atoms (~0.25 A)
    n_l = l_coords.shape[1]
    l_clash = p_coords[:, :n_l, :] + torch.tensor([[[0.25, 0.0, 0.0]]], device=dev)
    with torch.no_grad():
        l_field_clash = enc.encode_gaussian_field(l_clash, l_feats, l_mask)
        scores_clash = enc.score_pocket_ligand_overlap(
            ref_cx.pocket_field, p_mask, l_field_clash, l_mask
        )
    assert bool(
        (scores_clash["steric_clash_score"] > scores_5a["steric_clash_score"] * 3.0).all().item()
    )


# ---------------------------------------------------------------------------
# 5. 4DGS Multi-Conformer MD Ensemble Fusion
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_4dgs_ensemble_fusion_equivariance_psd_and_dummy_safety(device: str) -> None:
    """Verify fuse_ensemble_4d / encode_ensemble_4d SE(3) equivariance, PSD dominance & dummies."""
    torch.manual_seed(73)
    dev = torch.device(device)
    enc = _make_small_atomgs().to(dev).eval()
    base_coords, base_feats, mask = _make_realistic_pocket_batch(
        batch_size=2, num_atoms=24, device=dev, seed=44
    )

    gen = torch.Generator(device="cpu").manual_seed(515)
    conformers: list[Tensor] = []
    for t in range(3):
        jitter = torch.randn(base_coords.shape, generator=gen, dtype=torch.float32).to(dev) * 0.45
        drift = torch.tensor([[[0.5 * t, -0.3 * t, 0.2 * t]]], device=dev, dtype=torch.float32)
        c_t = base_coords + jitter + drift
        c_t[~mask] = 777.0 + 100.0 * t
        conformers.append(c_t)

    weights = torch.tensor([0.5, 0.3, 0.2], device=dev, dtype=torch.float32)

    with torch.no_grad():
        indiv_fields = [enc.encode_gaussian_field(c, base_feats, mask) for c in conformers]
        fused_ref = enc.fuse_ensemble_4d(indiv_fields, weights=weights, atom_mask=mask)
        fused_via_encode = enc.encode_ensemble_4d(conformers, base_feats, mask, weights=weights)
        fused_standalone = fuse_ensemble_4d(indiv_fields, weights=weights, atom_mask=mask)

    assert float((fused_ref.covariances - fused_via_encode.covariances).abs().max().item()) < 1e-6
    assert float((fused_ref.covariances - fused_standalone.covariances).abs().max().item()) < 1e-6

    eye = torch.eye(3, device=dev, dtype=torch.float32).expand_as(fused_ref.covariances[~mask])
    assert float(fused_ref.centers[~mask].abs().max().item()) == 0.0
    assert float((fused_ref.covariances[~mask] - eye).abs().max().item()) == 0.0
    assert fused_ref.chem_weights is not None
    assert float(fused_ref.chem_weights[~mask].abs().max().item()) == 0.0

    mean_cov = sum(
        w_t * f.covariances for w_t, f in zip(weights.tolist(), indiv_fields, strict=True)
    )
    assert isinstance(mean_cov, Tensor)
    diff_cov = (fused_ref.covariances[mask] - mean_cov[mask]).cpu().to(torch.float64)
    min_eig = float(torch.linalg.eigvalsh(diff_cov).min().item())
    max_eig = float(torch.linalg.eigvalsh(diff_cov).max().item())
    assert min_eig >= -1e-5, f"4D covariance violated PSD dominance: min_eig={min_eig:.2e}"
    assert max_eig > 0.05, f"4D conformational dispersion unexpectedly tiny: max_eig={max_eig:.2e}"

    rot = _random_so3_matrix(gen, dev)
    trans = torch.tensor([[[82.5, -61.0, 49.3]]], device=dev, dtype=torch.float32)
    rot_conformers = [c @ rot.T + trans for c in conformers]
    with torch.no_grad():
        fused_rot = enc.encode_ensemble_4d(rot_conformers, base_feats, mask, weights=weights)

    expected_cov = _rotate_cov(fused_ref.covariances, rot)
    assert float((fused_rot.covariances[mask] - expected_cov[mask]).abs().max().item()) < 5e-5
    center_err = float(
        (fused_rot.centers[mask] - fused_ref.centers[mask] @ rot.T).abs().max().item()
    )
    assert center_err < 5e-5
    assert fused_ref.dipole_vecs is not None and fused_rot.dipole_vecs is not None
    dip_err = float(
        (fused_rot.dipole_vecs[mask] - fused_ref.dipole_vecs[mask] @ rot.T).abs().max().item()
    )
    assert dip_err < 5e-5


# ---------------------------------------------------------------------------
# 6. Edge Cases: N=1 Ion, N<K, all_empty, d=0 Clash, Tie-Swap & MPS Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("global_attn_mode", ["landmarks", "full_scalar"])
@pytest.mark.parametrize("device", DEVICES)
def test_edge_cases_n1_nlt_k_isolated_empty_and_clash_backward(
    global_attn_mode: Literal["landmarks", "full_scalar"],
    device: str,
) -> None:
    """Verify N=1 ion, N<K ligand, all_empty isolated atom, empty row, and d=0 clash grads."""
    torch.manual_seed(99)
    dev = torch.device(device)
    enc = _make_small_atomgs(global_attn_mode=global_attn_mode).to(dev)
    enc.train()

    c1 = torch.tensor([[[3.0, -1.5, 2.0]]], device=dev, dtype=torch.float32, requires_grad=True)
    f1 = torch.randn((1, 1, 52), device=dev, dtype=torch.float32)
    f1[0, 0, 8] = 1.0
    f1[0, 0, 45] = 0.70
    m1 = torch.ones((1, 1), device=dev, dtype=torch.bool)
    out1, field1 = enc.encode_with_field(c1, f1, m1)
    assert bool(torch.isfinite(out1.pooled_emb).all().item())
    assert bool(torch.isfinite(field1.covariances).all().item())
    torch.autograd.backward(out1.pooled_emb.pow(2).sum() + field1.covariances.sum())
    assert c1.grad is not None and bool(torch.isfinite(c1.grad).all().item())

    enc.zero_grad(set_to_none=True)
    c_edge = torch.randn((2, 10, 3), device=dev, dtype=torch.float32) * 1.5
    c_edge[0, 1] = c_edge[0, 0]
    c_edge[0, 9] = torch.tensor([30.0, 0.0, 0.0], device=dev)
    c_edge = c_edge.detach().requires_grad_(True)
    f_edge = torch.randn((2, 10, 52), device=dev, dtype=torch.float32)
    m_edge = torch.ones((2, 10), device=dev, dtype=torch.bool)
    m_edge[1, :] = False

    out_e, field_e = enc.encode_with_field(c_edge, f_edge, m_edge)
    assert bool(torch.isfinite(out_e.node_emb).all().item())
    assert bool(torch.isfinite(out_e.pooled_emb).all().item())
    assert float(out_e.pooled_emb[1].abs().max().item()) == 0.0
    assert float(out_e.node_emb[1].abs().max().item()) == 0.0

    loss_e = out_e.pooled_emb.pow(2).sum() + out_e.node_emb.pow(2).sum() + field_e.covariances.sum()
    if out_e.vec_emb is not None:
        loss_e = loss_e + out_e.vec_emb.pow(2).sum()
    torch.autograd.backward(loss_e)

    assert c_edge.grad is not None and bool(torch.isfinite(c_edge.grad).all().item())
    for name, p in enc.named_parameters():
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all().item()), f"Non-finite grad in {name}"


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_equidistant_26_neighbor_tieswap_invariance(device: str) -> None:
    """Verify tie-swap invariance on 6 inner + 26 equidistant outer atoms with distinct features."""
    torch.manual_seed(111)
    dev = torch.device(device)
    enc = _make_small_atomgs(global_attn_mode="landmarks").to(dev).eval()

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
    coords = torch.cat((center, inner, shell), dim=0).unsqueeze(0).to(dev)
    mask = torch.ones((1, 33), dtype=torch.bool, device=dev)

    gen = torch.Generator(device="cpu").manual_seed(222)
    feats = torch.randn((1, 33, 52), generator=gen, dtype=torch.float32).to(dev)
    feats[..., 45] = 0.85

    with torch.no_grad():
        ref = enc(coords, feats, mask)

    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        with torch.no_grad():
            out_rot = enc(coords @ rot.T + trans, feats, mask)
        center_diff = float((out_rot.node_emb[0, 0] - ref.node_emb[0, 0]).abs().max().item())
        pool_diff = float((out_rot.pooled_emb - ref.pooled_emb).abs().max().item())
        assert center_diff < 5e-5, f"Tie-swap jump on central atom ({device}): {center_diff:.2e}"
        assert pool_diff < 5e-5, f"Tie-swap jump on pooled_emb ({device}): {pool_diff:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_mps_and_cpu_full_complex_backward_finite_gradients(device: str) -> None:
    """Verify finite backward gradients across encode_complex + overlap_scorer + cross_attn."""
    torch.manual_seed(131)
    dev = torch.device(device)
    enc = _make_small_atomgs(enable_complex_cross_attn=True).to(dev)
    enc.train()

    p_coords, p_feats, p_mask = _make_realistic_pocket_batch(
        batch_size=2, num_atoms=24, device=dev, seed=141
    )
    l_coords, l_feats, l_mask = _make_realistic_pocket_batch(
        batch_size=2, num_atoms=12, device=dev, seed=151
    )
    p_coords = p_coords.detach().requires_grad_(True)
    l_coords = l_coords.detach().requires_grad_(True)

    cx = enc.encode_complex(p_coords, p_feats, p_mask, l_coords, l_feats, l_mask)
    loss = (
        cx.complex_emb.pow(2).mean()
        + cx.pocket_out.pooled_emb.pow(2).mean()
        + cx.ligand_out.pooled_emb.pow(2).mean()
        + cx.overlap_scores["total_overlap_score"].mean()
        + cx.overlap_scores["rocs_shape_tanimoto"].mean()
        + cx.overlap_scores["steric_clash_score"].mean()
    )
    assert cx.pocket_emb_cross is not None and cx.ligand_emb_cross is not None
    loss = loss + cx.pocket_emb_cross.pow(2).mean() + cx.ligand_emb_cross.pow(2).mean()
    torch.autograd.backward(loss)

    assert p_coords.grad is not None and bool(torch.isfinite(p_coords.grad).all().item())
    assert l_coords.grad is not None and bool(torch.isfinite(l_coords.grad).all().item())
    grads_seen = 0
    for name, param in enc.named_parameters():
        if param.grad is not None:
            grads_seen += 1
            assert bool(torch.isfinite(param.grad).all().item()), f"Non-finite grad in {name}"
    assert grads_seen == sum(1 for _ in enc.parameters()), (
        f"Expected all parameters to receive grads in encode_complex, got {grads_seen}"
    )


# ---------------------------------------------------------------------------
# 7. Empirical Apple M3 Pro MPS Performance Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS unavailable")
def test_apple_mps_default_scale_timings_and_overlap_latency() -> None:
    """Measure real Apple MPS timings at default scale (B=16, N=384, K=24) & 384x40 overlap."""
    torch.manual_seed(2026)
    dev = torch.device("mps")
    enc = AtomGSSE3Encoder(
        node_in_dim=52,
        hidden_dim=192,
        vector_dim=64,
        out_dim=256,
        num_layers=4,
        num_heads=8,
        num_rbf=24,
        cutoff=5.5,
        max_neighbors=24,
        num_landmarks=16,
        global_attn_mode="landmarks",
        use_se3_chirality=True,
        enable_3d_overlap=True,
        use_coulomb_bias=True,
        dropout=0.0,
    ).to(dev)

    p_coords, p_feats, p_mask = _make_realistic_pocket_batch(
        batch_size=16, num_atoms=384, device=dev, seed=301
    )
    l_coords, l_feats, l_mask = _make_realistic_pocket_batch(
        batch_size=16, num_atoms=40, device=dev, seed=302
    )

    enc.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = enc(p_coords, p_feats, p_mask)
        _sync_device(dev)

        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            _ = enc(p_coords, p_feats, p_mask)
        _sync_device(dev)
        fwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    enc.train()
    for _ in range(2):
        enc.zero_grad(set_to_none=True)
        out = enc(p_coords, p_feats, p_mask)
        loss = out.pooled_emb.pow(2).mean() + out.node_emb.pow(2).mean()
        torch.autograd.backward(loss)
    _sync_device(dev)

    t0 = time.perf_counter()
    for _ in range(iters):
        enc.zero_grad(set_to_none=True)
        out = enc(p_coords, p_feats, p_mask)
        loss = out.pooled_emb.pow(2).mean() + out.node_emb.pow(2).mean()
        torch.autograd.backward(loss)
    _sync_device(dev)
    fwdbwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Measure score_pocket_ligand_overlap latency (B=16, N_P=384, N_L=40) with cached O(P,P)/O(L,L)
    enc.eval()
    with torch.no_grad():
        p_field = enc.encode_gaussian_field(p_coords, p_feats, p_mask)
        l_field = enc.encode_gaussian_field(l_coords, l_feats, l_mask)
        o_pp = rocs_self_overlap(p_field, mask=p_mask, max_neighbors=32, cutoff=12.0)
        o_ll = rocs_self_overlap(l_field, mask=l_mask)
        for _ in range(3):
            _ = enc.score_pocket_ligand_overlap(
                p_field,
                p_mask,
                l_field,
                l_mask,
                pocket_self_overlap=o_pp,
                ligand_self_overlap=o_ll,
            )
        _sync_device(dev)

        t0 = time.perf_counter()
        ovl_iters = 20
        for _ in range(ovl_iters):
            scores = enc.score_pocket_ligand_overlap(
                p_field,
                p_mask,
                l_field,
                l_mask,
                pocket_self_overlap=o_pp,
                ligand_self_overlap=o_ll,
            )
        _sync_device(dev)
        ovl_ms = (time.perf_counter() - t0) * 1000.0 / ovl_iters

    assert all(bool(torch.isfinite(v).all().item()) for v in scores.values())
    n_params = sum(p.numel() for p in enc.parameters())
    print(
        f"\n[AtomGS-SE3 MPS B=16,N=384,K=24] params={n_params:,} | "
        f"fwd={fwd_ms:.2f}ms | fwd+bwd={fwdbwd_ms:.2f}ms | "
        f"overlap(384x40)={ovl_ms:.2f}ms"
    )
    # Also measure PaiNN-SE3 on the exact same device/load to verify relative speedup cleanly
    from geneuclidean.models.encoders.painn import PaiNNEncoder

    painn = PaiNNEncoder(
        node_in_dim=52,
        hidden_dim=192,
        vector_dim=64,
        out_dim=256,
        num_layers=4,
        num_rbf=24,
        cutoff=5.5,
        max_neighbors=24,
        dropout=0.0,
    ).to(dev)
    painn.train()
    for _ in range(2):
        painn.zero_grad(set_to_none=True)
        po = painn(p_coords, p_feats, p_mask)
        torch.autograd.backward(po.pooled_emb.pow(2).mean() + po.node_emb.pow(2).mean())
    _sync_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        painn.zero_grad(set_to_none=True)
        po = painn(p_coords, p_feats, p_mask)
        torch.autograd.backward(po.pooled_emb.pow(2).mean() + po.node_emb.pow(2).mean())
    _sync_device(dev)
    painn_fwdbwd_ms = (time.perf_counter() - t0) * 1000.0 / iters
    speedup = painn_fwdbwd_ms / max(fwdbwd_ms, 1e-6)
    print(
        f"[MPS Speedup vs PaiNN-SE3] PaiNN fwd+bwd={painn_fwdbwd_ms:.2f}ms | "
        f"AtomGS-SE3 speedup={speedup:.2f}x"
    )
    assert fwdbwd_ms < painn_fwdbwd_ms, (
        f"AtomGS-SE3 ({fwdbwd_ms:.2f}ms) should beat PaiNN-SE3 ({painn_fwdbwd_ms:.2f}ms)"
    )
    assert ovl_ms < 10.0, f"MPS overlap scoring too slow: {ovl_ms:.2f}ms"
