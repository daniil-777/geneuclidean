"""Comprehensive CPU & Apple Silicon MPS test suite for CartFormer-SE3 & DensePaiNN-SE3.

Verifies every mathematical invariant, physical edge case, gradient stability requirement,
and Apple M3 Pro MPS throughput target for both:
  - :class:`~geneuclidean.models.encoders.cartformer.CartFormerSE3Encoder`
  - :class:`~geneuclidean.models.encoders.dense_painn.DensePaiNNSE3Encoder`

Coverage (under ``PYTORCH_ENABLE_MPS_FALLBACK=0`` on MPS as well as CPU):
  1. ``PocketEncoderProtocol`` compliance, ``build_cartformer_se3`` / ``build_dense_painn_se3``
     factory contracts, reusable sub-module contracts, and zero ``float64`` params/buffers.
  2. Exact ``SO(3)`` rotation + ``100 A`` rigid translation invariance (``< 5e-5``) and
     ``vec_emb`` equivariance (``< 5e-5``), plus non-vacuous coordinate sensitivity.
  3. Exact atom permutation invariance/equivariance and ``0.0`` dummy-atom sensitivity.
  4. Exact chiral ``(R)/(S)`` mirror-reflection discrimination (``det(P) = -1``, ``> 1e-3``;
     including Layer-1 active chirality in ``CartFormer-SE3``), ``use_se3_chirality=False``
     exact ``E(3)`` toggle, and exact achirality (zero reflection diff) on planar benzene and
     collinear alkyne with finite coordinate gradients.
  5. Equidistant 26-neighbor ``K``-th tie-swap invariance (both pure shell and mixed
     close + tie shell, ``< 1e-6`` under pure rotation).
  6. Edge-case completeness: ``N=1`` single ion, ``N < K`` small ligand, isolated 0-neighbor
     (``all_empty``) atoms, padded variable-length batches, empty mask rows, ``d -> 0``
     exact coordinate clashes, and ``d -> cutoff`` boundary smoothness (zero ``+ log f_c``
     singularity).
  7. Finite ``float32`` forward + backward parameter & coordinate gradients on CPU & MPS.
  8. Empirical Apple Silicon MPS speed check (``B=16, N=384, K=24``) comparing
     ``CartFormer-SE3`` & ``DensePaiNN-SE3`` against sparse ``PaiNN-SE3``.
"""

from __future__ import annotations

import math
import time
from typing import Literal

import pytest
import torch
from torch import Tensor

from geneuclidean.config import EncoderConfig
from geneuclidean.models.encoders.cartformer import (
    CartesianTensorLinear,
    CartFormerAttentionLayer,
    CartFormerFFN,
    CartFormerLandmarkBlock,
    CartFormerSE3Encoder,
    build_cartformer_se3,
)
from geneuclidean.models.encoders.dense_knn import build_dense_knn_graph
from geneuclidean.models.encoders.dense_painn import (
    DensePaiNNMessageLayer,
    DensePaiNNSE3Encoder,
    DensePaiNNUpdateLayer,
    build_dense_painn_se3,
)
from geneuclidean.models.encoders.painn import PaiNNEncoder
from geneuclidean.models.encoders.rbf import PocketEncoderOutput
from geneuclidean.protocols import PocketEncoderProtocol

EncoderKind = Literal["cartformer_se3", "dense_painn_se3"]
ENCODER_KINDS: list[EncoderKind] = ["cartformer_se3", "dense_painn_se3"]
DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _random_so3_matrix(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a Haar-uniform proper rotation matrix R in SO(3) with det(R) = +1."""
    mat = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    if float(torch.det(q).item()) < 0.0:
        q[:, 0] = -q[:, 0]
    q_rot: Tensor = q.to(device=device, dtype=torch.float32)
    return q_rot


def _reflection_matrix(device: torch.device) -> Tensor:
    """Return an improper orthogonal reflection matrix P with det(P) = -1."""
    return torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=torch.float32))


def _make_encoder(
    kind: EncoderKind,
    *,
    num_layers: int = 3,
    use_se3_chirality: bool = True,
    use_coulomb_bias: bool = True,
    num_landmarks: int = 8,
    max_neighbors: int = 24,
    dropout: float = 0.0,
) -> CartFormerSE3Encoder | DensePaiNNSE3Encoder:
    cfg = EncoderConfig(
        arch=kind,
        node_in_dim=52,
        hidden_dim=64,
        vector_dim=16,
        tensor_dim=12,
        out_dim=64,
        num_layers=num_layers,
        num_heads=4,
        num_rbf=16,
        num_landmarks=num_landmarks,
        cutoff=5.5,
        use_se3_chirality=use_se3_chirality,
        use_coulomb_bias=use_coulomb_bias,
        dropout=dropout,
    )
    if kind == "cartformer_se3":
        return build_cartformer_se3(cfg, max_neighbors=max_neighbors)
    return build_dense_painn_se3(cfg, max_neighbors=max_neighbors)


def _make_pocket_batch(
    batch_size: int = 2,
    num_atoms: int = 32,
    node_in_dim: int = 52,
    device: torch.device | str = "cpu",
    seed: int = 123,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a realistic chiral 3D pocket batch with variable valid/padded atom masks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32) * 2.2
    node_feats = torch.randn(
        (batch_size, num_atoms, node_in_dim), generator=gen, dtype=torch.float32
    )
    # Place realistic partial-charge proxies in col 46 in [-1, 1].
    node_feats[..., 46] = torch.tanh(node_feats[..., 46])
    atom_mask = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
    for b in range(batch_size):
        valid_count = max(8, num_atoms - 4 - 3 * b)
        atom_mask[b, :valid_count] = True
    dev = torch.device(device)
    return coords.to(dev), node_feats.to(dev), atom_mask.to(dev)


def _chiral_tetrahedron_batch(
    device: torch.device,
    node_in_dim: int = 52,
    seed: int = 77,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a synthetic 5-atom chiral (R) stereocenter with 4 chemically distinct ligands."""
    # Asymmetric tetrahedral bond lengths & distinct features prevent any mirror plane.
    raw_coords = torch.tensor(
        [
            [
                [0.00, 0.00, 0.00],  # central chiral C*
                [1.54, 0.00, 0.00],  # substituent 1 (C-C)
                [-0.48, 1.38, 0.00],  # substituent 2 (C-N)
                [-0.52, -0.74, 1.25],  # substituent 3 (C-O)
                [-0.60, -0.82, -1.52],  # substituent 4 (C-S / Br)
            ]
        ],
        dtype=torch.float32,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.randn((1, 5, node_in_dim), generator=gen, dtype=torch.float32)
    feats[..., 46] = torch.tensor([[0.15, -0.35, 0.45, -0.55, 0.30]], dtype=torch.float32)
    mask = torch.ones((1, 5), dtype=torch.bool)
    return raw_coords.to(device), feats.to(device), mask.to(device)


def _planar_benzene_batch(
    device: torch.device,
    node_in_dim: int = 52,
    seed: int = 91,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a strictly planar (z = 0) 6-atom aromatic ring with asymmetric atom features."""
    angles = torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float32)[:-1]
    coords = torch.stack(
        (1.40 * torch.cos(angles), 1.40 * torch.sin(angles), torch.zeros_like(angles)),
        dim=-1,
    ).unsqueeze(0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.randn((1, 6, node_in_dim), generator=gen, dtype=torch.float32)
    feats[..., 46] = torch.tanh(feats[..., 46])
    mask = torch.ones((1, 6), dtype=torch.bool)
    return coords.to(device), feats.to(device), mask.to(device)


def _collinear_alkyne_batch(
    device: torch.device,
    node_in_dim: int = 52,
    seed: int = 97,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a strictly collinear (y = z = 0) 4-atom 2-butyne chain."""
    xs = torch.tensor([-1.96, -0.60, 0.60, 1.96], dtype=torch.float32)
    zeros = torch.zeros_like(xs)
    coords = torch.stack((xs, zeros, zeros), dim=-1).unsqueeze(0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.randn((1, 4, node_in_dim), generator=gen, dtype=torch.float32)
    feats[..., 46] = torch.tanh(feats[..., 46])
    mask = torch.ones((1, 4), dtype=torch.bool)
    return coords.to(device), feats.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# 1. Protocol, Builders, Submodules & Zero-float64 Contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ENCODER_KINDS)
def test_protocol_compliance_builders_and_zero_float64(kind: EncoderKind) -> None:
    """Both encoders satisfy PocketEncoderProtocol, resolve config aliases, and have 0 float64."""
    alias = "cartformer" if kind == "cartformer_se3" else "dense_painn"
    cfg = EncoderConfig(arch=alias, hidden_dim=64, vector_dim=16, tensor_dim=12, out_dim=64)
    assert cfg.arch == kind

    enc = (
        build_cartformer_se3(cfg, max_neighbors=24)
        if kind == "cartformer_se3"
        else build_dense_painn_se3(cfg, max_neighbors=24)
    )
    assert isinstance(enc, PocketEncoderProtocol)

    for name, param in enc.named_parameters():
        assert param.dtype == torch.float32, f"Non-float32 param {name}: {param.dtype}"
    for name, buf in enc.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer found: {name}"

    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=20, device="cpu")
    out = enc(coords, node_feats, atom_mask)
    assert isinstance(out, PocketEncoderOutput)
    assert out.node_emb.shape == (2, 20, 64)
    assert out.pooled_emb.shape == (2, 64)
    assert out.vec_emb is not None and out.vec_emb.shape == (2, 20, 16, 3)


def test_reusable_submodules_and_ablation_toggles() -> None:
    """Verify reusable submodule APIs and CartFormer landmark/Coulomb ablation flags."""
    torch.manual_seed(7)
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=16, device="cpu")
    graph = build_dense_knn_graph(coords, atom_mask, cutoff=5.5, max_neighbors=12)

    # DensePaiNN submodules
    msg = DensePaiNNMessageLayer(scalar_dim=32, vector_dim=8, num_rbf=12)
    upd = DensePaiNNUpdateLayer(scalar_dim=32, vector_dim=8, use_se3_chirality=True, dropout=0.0)
    s = torch.randn((2, 16, 32), dtype=torch.float32) * atom_mask.unsqueeze(-1).float()
    v = torch.randn((2, 16, 8, 3), dtype=torch.float32) * atom_mask[:, :, None, None].float()
    rbf = torch.randn((2, 16, graph.nbr_idx.shape[2], 12), dtype=torch.float32)
    ds, dv = msg(s, v, graph, rbf, atom_mask)
    s2, v2 = upd(s + ds, v + dv, atom_mask)
    assert s2.shape == (2, 16, 32) and v2.shape == (2, 16, 8, 3)

    # CartFormer submodules
    tlin = CartesianTensorLinear(8, 8)
    x_ten = torch.randn((2, 16, 8, 3, 3), dtype=torch.float32)
    assert tlin(x_ten).shape == (2, 16, 8, 3, 3)

    attn = CartFormerAttentionLayer(
        scalar_dim=32, vector_dim=8, tensor_dim=8, num_heads=4, num_rbf=12, dropout=0.0
    )
    ffn = CartFormerFFN(scalar_dim=32, vector_dim=8, tensor_dim=8, dropout=0.0)
    lm = CartFormerLandmarkBlock(
        scalar_dim=32, vector_dim=8, num_landmarks=4, num_heads=4, use_coulomb_bias=False
    )
    outer_r = torch.zeros((2, 16, graph.nbr_idx.shape[2], 3, 3), dtype=torch.float32)
    q5 = torch.zeros((2, 16, graph.nbr_idx.shape[2], 5), dtype=torch.float32)
    x_ten.requires_grad_()
    cds, cdv, cdx = attn(s, v, x_ten, graph, rbf, outer_r, q5, atom_mask)
    # The explicit polar/axial edge bases define the unfused chiral projection.
    reference = attn(
        s,
        v,
        x_ten,
        graph,
        rbf,
        outer_r,
        q5,
        atom_mask,
        geom_edge_i=torch.cat((graph.edge_dir, q5, graph.edge_dir), dim=-1),
        geom_edge_j=torch.cat((graph.edge_dir, q5, -graph.edge_dir), dim=-1),
    )
    for fused, separate in zip((cds, cdv, cdx), reference, strict=True):
        torch.testing.assert_close(fused, separate, atol=1e-6, rtol=1e-5)
    grad = torch.autograd.grad(cds.square().sum(), x_ten)[0]
    ref_grad = torch.autograd.grad(reference[0].square().sum(), x_ten)[0]
    torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=1e-5)
    cs, cv, cx = ffn(s + cds, v + cdv, x_ten + cdx, atom_mask)
    cs_lm, cv_lm = lm(cs, cv, cx, graph.centered_coords, node_feats, atom_mask)
    assert cs_lm.shape == (2, 16, 32) and cv_lm.shape == (2, 16, 8, 3)

    # CartFormer with num_landmarks=0 and use_coulomb_bias=False
    enc_no_lm = _make_encoder("cartformer_se3", num_landmarks=0, use_coulomb_bias=False)
    out_no_lm = enc_no_lm(coords, node_feats, atom_mask)
    assert torch.isfinite(out_no_lm.pooled_emb).all()


# ---------------------------------------------------------------------------
# 2. Exact SO(3) Rotation + 100 A Translation Equivariance (CPU & MPS)
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("kind", ENCODER_KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_so3_rotation_and_100a_translation_equivariance(
    kind: EncoderKind,
    device: str,
) -> None:
    """Verify exact SO(3) invariance (scalars) & equivariance (vectors) with 100 A shifts."""
    torch.manual_seed(42)
    dev = torch.device(device)
    enc = _make_encoder(kind, num_layers=3, use_se3_chirality=True).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=11)

    with torch.no_grad():
        ref = enc(coords, node_feats, atom_mask)
        # Non-vacuous check: non-rigid geometry perturbation must move outputs significantly.
        pert_out = enc(coords + torch.randn_like(coords) * 1.2, node_feats, atom_mask)
        assert (pert_out.pooled_emb - ref.pooled_emb).abs().max().item() > 1e-2

    assert ref.vec_emb is not None
    assert ref.vec_emb.abs().max().item() > 1e-2, f"{kind} vec_emb is trivially zero"

    tol_pure_rot = 5e-5
    tol_rot_trans = 1e-4
    gen = torch.Generator(device="cpu").manual_seed(2026)

    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        # Explicit 100 A rigid translation in [-100, 100]^3.
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        coords_rot = coords @ rot.T
        coords_rt = coords_rot + trans

        with torch.no_grad():
            out_rot = enc(coords_rot, node_feats, atom_mask)
            out_rt = enc(coords_rt, node_feats, atom_mask)

        assert out_rot.vec_emb is not None and out_rt.vec_emb is not None
        expected_vec = ref.vec_emb @ rot.T

        rot_node_err = (out_rot.node_emb - ref.node_emb).abs().max().item()
        rot_pool_err = (out_rot.pooled_emb - ref.pooled_emb).abs().max().item()
        rot_vec_err = (out_rot.vec_emb - expected_vec).abs().max().item()

        rt_node_err = (out_rt.node_emb - ref.node_emb).abs().max().item()
        rt_pool_err = (out_rt.pooled_emb - ref.pooled_emb).abs().max().item()
        rt_vec_err = (out_rt.vec_emb - expected_vec).abs().max().item()

        assert rot_node_err < tol_pure_rot, (
            f"{kind} ({device}) pure-rot node err={rot_node_err:.2e}"
        )
        assert rot_pool_err < tol_pure_rot, (
            f"{kind} ({device}) pure-rot pool err={rot_pool_err:.2e}"
        )
        assert rot_vec_err < tol_pure_rot, f"{kind} ({device}) pure-rot vec err={rot_vec_err:.2e}"
        assert rt_node_err < tol_rot_trans, f"{kind} ({device}) rot+100A node err={rt_node_err:.2e}"
        assert rt_pool_err < tol_pure_rot, f"{kind} ({device}) rot+100A pool err={rt_pool_err:.2e}"
        assert rt_vec_err < tol_rot_trans, f"{kind} ({device}) rot+100A vec err={rt_vec_err:.2e}"


# ---------------------------------------------------------------------------
# 3. Permutation Invariance & Dummy-Atom Zero Sensitivity (CPU & MPS)
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("kind", ENCODER_KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_permutation_and_dummy_corruption_invariance(
    kind: EncoderKind,
    device: str,
) -> None:
    """Verify exact atom permutation equivariance and zero sensitivity to corrupted dummies."""
    torch.manual_seed(17)
    dev = torch.device(device)
    enc = _make_encoder(kind, num_layers=3).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=19)

    gen = torch.Generator(device="cpu").manual_seed(303)
    perm = torch.randperm(coords.shape[1], generator=gen).to(dev)

    coords_corrupt = coords.clone()
    feats_corrupt = node_feats.clone()
    coords_corrupt[~atom_mask] = 999.0
    feats_corrupt[~atom_mask] = torch.randn_like(feats_corrupt[~atom_mask]) * 1e4

    with torch.no_grad():
        ref = enc(coords, node_feats, atom_mask)
        out_perm = enc(coords[:, perm], node_feats[:, perm], atom_mask[:, perm])
        out_corrupt = enc(coords_corrupt, feats_corrupt, atom_mask)

    assert ref.vec_emb is not None
    assert out_perm.vec_emb is not None
    assert out_corrupt.vec_emb is not None

    # Permutation equivariance/invariance
    assert (out_perm.pooled_emb - ref.pooled_emb).abs().max().item() < 1e-5
    assert (out_perm.node_emb - ref.node_emb[:, perm]).abs().max().item() < 1e-5
    assert (out_perm.vec_emb - ref.vec_emb[:, perm]).abs().max().item() < 1e-5

    # Dummy slots strictly zeroed and real atoms unaffected (dense ops have zero scatter jitter!)
    assert out_corrupt.node_emb[~atom_mask].abs().max().item() == 0.0
    assert out_corrupt.vec_emb[~atom_mask].abs().max().item() == 0.0
    assert (out_corrupt.node_emb - ref.node_emb).abs().max().item() == 0.0
    assert (out_corrupt.pooled_emb - ref.pooled_emb).abs().max().item() == 0.0
    assert (out_corrupt.vec_emb - ref.vec_emb).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 4. Chiral (R)/(S) Reflection Discrimination & Achiral Planar/Collinear Consistency
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("kind", ENCODER_KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_chiral_rs_tetrahedron_and_achiral_planar_collinear(
    kind: EncoderKind,
    device: str,
) -> None:
    """Verify chiral (R)/(S) sensitivity (diff > 1e-3), E(3) toggle, and 0 planar/collinear diff."""
    torch.manual_seed(31)
    dev = torch.device(device)
    reflect = _reflection_matrix(dev)

    # 4a. Chiral tetrahedron: use_se3_chirality=True distinguishes (R) vs (S) by > 1e-3!
    # Also verify CartFormer-SE3 is chirally active immediately at num_layers=1!
    num_layers_chiral = 1 if kind == "cartformer_se3" else 3
    enc_se3 = _make_encoder(kind, num_layers=num_layers_chiral, use_se3_chirality=True).to(dev)
    enc_se3.eval()

    tet_coords, tet_feats, tet_mask = _chiral_tetrahedron_batch(dev)
    with torch.no_grad():
        out_r = enc_se3(tet_coords, tet_feats, tet_mask)
        out_s = enc_se3(tet_coords @ reflect.T, tet_feats, tet_mask)

    rs_pool_diff = (out_r.pooled_emb - out_s.pooled_emb).abs().max().item()
    rs_node_diff = (out_r.node_emb - out_s.node_emb).abs().max().item()
    assert rs_pool_diff > 1e-3, f"{kind} ({device}) failed R/S pooled diff={rs_pool_diff:.2e}"
    assert rs_node_diff > 1e-3, f"{kind} ({device}) failed R/S node diff={rs_node_diff:.2e}"

    # 4b. Multi-layer SE(3) encoder also discriminates full 3D pockets and stays achiral on planes.
    enc_se3_3l = _make_encoder(kind, num_layers=3, use_se3_chirality=True).to(dev).eval()
    p_coords, p_feats, p_mask = _make_pocket_batch(device=dev, seed=37)
    with torch.no_grad():
        p_l = enc_se3_3l(p_coords, p_feats, p_mask)
        p_d = enc_se3_3l(p_coords @ reflect.T, p_feats, p_mask)
    assert (p_l.pooled_emb - p_d.pooled_emb).abs().max().item() > 1e-3

    # 4c. use_se3_chirality=False restores exact E(3) reflection invariance (< 5e-5).
    enc_e3 = _make_encoder(kind, num_layers=3, use_se3_chirality=False).to(dev).eval()
    with torch.no_grad():
        e3_r = enc_e3(tet_coords, tet_feats, tet_mask)
        e3_s = enc_e3(tet_coords @ reflect.T, tet_feats, tet_mask)
        e3_pl = enc_e3(p_coords, p_feats, p_mask)
        e3_pd = enc_e3(p_coords @ reflect.T, p_feats, p_mask)
    assert (e3_r.pooled_emb - e3_s.pooled_emb).abs().max().item() < 5e-5
    assert (e3_pl.pooled_emb - e3_pd.pooled_emb).abs().max().item() < 5e-5

    # 4d. Strictly planar benzene ring (z = 0): exact zero reflection diff + finite grads!
    bz_coords, bz_feats, bz_mask = _planar_benzene_batch(dev)
    bz_coords_req = bz_coords.clone().requires_grad_(True)
    bz_out = enc_se3_3l(bz_coords_req, bz_feats, bz_mask)
    bz_loss = bz_out.pooled_emb.pow(2).sum() + (
        bz_out.vec_emb.pow(2).sum() if bz_out.vec_emb is not None else 0.0
    )
    (bz_grad,) = torch.autograd.grad(bz_loss, bz_coords_req)
    assert torch.isfinite(bz_grad).all(), f"{kind} ({device}) non-finite grad on planar benzene"

    with torch.no_grad():
        bz_ref = enc_se3_3l(bz_coords, bz_feats, bz_mask)
        bz_mir = enc_se3_3l(bz_coords @ reflect.T, bz_feats, bz_mask)
    bz_diff = (bz_ref.pooled_emb - bz_mir.pooled_emb).abs().max().item()
    assert bz_diff < 2e-5, f"{kind} ({device}) planar benzene reflection diff={bz_diff:.2e}"
    if kind == "cartformer_se3":
        assert bz_ref.vec_emb is not None
        # CartFormer preserves strict 1o polar parity: vec_emb stays purely in-plane (z == 0)!
        assert bz_ref.vec_emb[..., 2].abs().max().item() < 1e-6

    # 4e. Strictly collinear alkyne (y = z = 0): exact zero reflection diff + finite grads!
    alk_coords, alk_feats, alk_mask = _collinear_alkyne_batch(dev)
    alk_coords_req = alk_coords.clone().requires_grad_(True)
    alk_out = enc_se3_3l(alk_coords_req, alk_feats, alk_mask)
    alk_loss = alk_out.pooled_emb.pow(2).sum() + (
        alk_out.vec_emb.pow(2).sum() if alk_out.vec_emb is not None else 0.0
    )
    (alk_grad,) = torch.autograd.grad(alk_loss, alk_coords_req)
    assert torch.isfinite(alk_grad).all(), f"{kind} ({device}) non-finite grad on collinear alkyne"

    with torch.no_grad():
        alk_ref = enc_se3_3l(alk_coords, alk_feats, alk_mask)
        alk_mir = enc_se3_3l(alk_coords @ reflect.T, alk_feats, alk_mask)
    alk_diff = (alk_ref.pooled_emb - alk_mir.pooled_emb).abs().max().item()
    assert alk_diff < 2e-5, f"{kind} ({device}) collinear alkyne reflection diff={alk_diff:.2e}"


# ---------------------------------------------------------------------------
# 5. Equidistant 26-Neighbor Tie-Swap Invariance (CPU & MPS)
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
@pytest.mark.parametrize("kind", ENCODER_KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_equidistant_26_neighbor_tieswap_invariance(
    kind: EncoderKind,
    device: str,
) -> None:
    """Verify tie-swap invariance on both pure 26-neighbor shells and mixed close+tie shells."""
    torch.manual_seed(53)
    dev = torch.device(device)
    enc = _make_encoder(kind, num_layers=2, max_neighbors=24, num_landmarks=4).to(dev).eval()

    # 26 equidistant lattice-shell directions on the sphere of radius 4.8 A around origin.
    shell_dirs = [
        [float(dx), float(dy), float(dz)]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    assert len(shell_dirs) == 26
    shell_t = torch.tensor(shell_dirs, dtype=torch.float32)
    shell_t = shell_t / shell_t.norm(dim=-1, keepdim=True) * 4.8

    # Case A: Pure 26 equidistant neighbors at 4.8 A around central atom (27 atoms total, K=24).
    coords_pure = torch.cat((torch.zeros((1, 3), dtype=torch.float32), shell_t), dim=0).unsqueeze(0)
    gen = torch.Generator(device="cpu").manual_seed(61)
    feats_pure = torch.randn((1, 27, 52), generator=gen, dtype=torch.float32)
    mask_pure = torch.ones((1, 27), dtype=torch.bool)

    # Case B: Mixed shell: 6 close atoms at 2.0 A + 26 equidistant tie-shell atoms at 4.8 A
    # (33 atoms total, K=24 -> 18 of 26 tie neighbors land inside top-24 and swap on rotation!).
    close_t = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=torch.float32,
    )
    coords_mixed = torch.cat(
        (torch.zeros((1, 3), dtype=torch.float32), close_t, shell_t), dim=0
    ).unsqueeze(0)
    feats_mixed = torch.randn((1, 33, 52), generator=gen, dtype=torch.float32)
    mask_mixed = torch.ones((1, 33), dtype=torch.bool)

    for label, coords_cpu, feats_cpu, mask_cpu in (
        ("pure_26_shell", coords_pure, feats_pure, mask_pure),
        ("mixed_close_and_26_tie_shell", coords_mixed, feats_mixed, mask_mixed),
    ):
        coords_d = coords_cpu.to(dev)
        feats_d = feats_cpu.to(dev)
        mask_d = mask_cpu.to(dev)
        with torch.no_grad():
            ref = enc(coords_d, feats_d, mask_d)

        # Test permutation of the tie-shell indices + rigid SO(3) rotation!
        perm_shell = torch.arange(coords_d.shape[1], device=dev)
        tie_start = coords_d.shape[1] - 26
        tie_perm = tie_start + torch.randperm(26, generator=gen).to(dev)
        perm_shell[tie_start:] = tie_perm

        rot = _random_so3_matrix(gen, dev)
        with torch.no_grad():
            out_perm = enc(coords_d[:, perm_shell], feats_d[:, perm_shell], mask_d[:, perm_shell])
            out_rot = enc(coords_d @ rot.T, feats_d, mask_d)

        # Central atom index 0 must remain strictly invariant despite swapping tied neighbors!
        perm_center_err = (out_perm.node_emb[:, 0] - ref.node_emb[:, 0]).abs().max().item()
        perm_pool_err = (out_perm.pooled_emb - ref.pooled_emb).abs().max().item()
        rot_center_err = (out_rot.node_emb[:, 0] - ref.node_emb[:, 0]).abs().max().item()
        rot_pool_err = (out_rot.pooled_emb - ref.pooled_emb).abs().max().item()

        assert perm_center_err < 1e-6, (
            f"{kind} ({device}) {label} perm center={perm_center_err:.2e}"
        )
        assert perm_pool_err < 1e-6, f"{kind} ({device}) {label} perm pool={perm_pool_err:.2e}"
        assert rot_center_err < 5e-5, f"{kind} ({device}) {label} rot center={rot_center_err:.2e}"
        assert rot_pool_err < 5e-5, f"{kind} ({device}) {label} rot pool={rot_pool_err:.2e}"


# ---------------------------------------------------------------------------
# 6. Edge Cases: N=1, N<K, all_empty, Variable Padding, d->0 Clash, d->cutoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ENCODER_KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_edge_cases_n1_small_ligand_isolated_clash_and_boundary(
    kind: EncoderKind,
    device: str,
) -> None:
    """Verify finite forward + backward across all degenerate geometric edge cases."""
    torch.manual_seed(71)
    dev = torch.device(device)
    enc = _make_encoder(kind, num_layers=2, max_neighbors=24, dropout=0.0).to(dev).train()
    gen = torch.Generator(device="cpu").manual_seed(73)

    # 6a. Single-atom ion (N=1, e.g. isolated Zn2+): translation invariance => zero coord grad.
    c_n1 = torch.tensor([[[1.5, -2.0, 0.5]]], device=dev, dtype=torch.float32, requires_grad=True)
    f_n1 = torch.randn((1, 1, 52), generator=gen, dtype=torch.float32).to(dev)
    m_n1 = torch.ones((1, 1), device=dev, dtype=torch.bool)
    out_n1 = enc(c_n1, f_n1, m_n1)
    assert out_n1.node_emb.shape == (1, 1, 64)
    assert torch.isfinite(out_n1.pooled_emb).all()
    (g_n1,) = torch.autograd.grad(out_n1.pooled_emb.sum(), c_n1, allow_unused=True)
    if g_n1 is not None:
        assert torch.isfinite(g_n1).all()
        assert g_n1.abs().max().item() < 1e-6

    # 6b. Small ligand N=10 < K=24 with variable padding & an all-False empty batch row
    c_sm = (torch.randn((3, 10, 3), generator=gen, dtype=torch.float32) * 1.8).to(dev)
    c_sm.requires_grad_(True)
    f_sm = torch.randn((3, 10, 52), generator=gen, dtype=torch.float32).to(dev)
    m_sm = torch.zeros((3, 10), device=dev, dtype=torch.bool)
    m_sm[0, :10] = True
    m_sm[1, :4] = True
    # row 2 is completely empty (all False)
    out_sm = enc(c_sm, f_sm, m_sm)
    assert torch.isfinite(out_sm.node_emb).all() and torch.isfinite(out_sm.pooled_emb).all()
    assert out_sm.pooled_emb[2].abs().max().item() == 0.0
    assert out_sm.node_emb[~m_sm].abs().max().item() == 0.0
    (g_sm,) = torch.autograd.grad(out_sm.pooled_emb.pow(2).sum(), c_sm)
    assert torch.isfinite(g_sm).all()

    # 6c. Isolated atom with 0 neighbors inside cutoff (all_empty neighborhood)
    c_iso = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [50.0, 50.0, 50.0]]],
        device=dev,
        dtype=torch.float32,
        requires_grad=True,
    )
    f_iso = torch.randn((1, 4, 52), generator=gen, dtype=torch.float32).to(dev)
    m_iso = torch.ones((1, 4), device=dev, dtype=torch.bool)
    out_iso = enc(c_iso, f_iso, m_iso)
    assert torch.isfinite(out_iso.node_emb).all() and torch.isfinite(out_iso.pooled_emb).all()
    (g_iso,) = torch.autograd.grad(out_iso.pooled_emb.pow(2).sum(), c_iso)
    assert torch.isfinite(g_iso).all()

    # 6d. Exact d -> 0 coordinate clash AND d -> cutoff boundary (5.4999 A) smoothness!
    c_clash = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # exact 0.0 A clash with atom 0!
                [1e-6, 0.0, 0.0],  # near-zero 1e-6 A clash!
                [5.4999, 0.0, 0.0],  # right at cutoff boundary where f_c -> 0!
                [1.4, 1.2, -0.8],
            ]
        ],
        device=dev,
        dtype=torch.float32,
        requires_grad=True,
    )
    f_clash = torch.randn((1, 5, 52), generator=gen, dtype=torch.float32).to(dev)
    m_clash = torch.ones((1, 5), device=dev, dtype=torch.bool)
    enc.zero_grad(set_to_none=True)
    out_clash = enc(c_clash, f_clash, m_clash)
    assert torch.isfinite(out_clash.node_emb).all()
    assert torch.isfinite(out_clash.pooled_emb).all()
    assert out_clash.vec_emb is not None and torch.isfinite(out_clash.vec_emb).all()

    loss_clash = out_clash.pooled_emb.pow(2).sum() + out_clash.vec_emb.pow(2).sum()
    loss_clash.backward()
    assert c_clash.grad is not None and torch.isfinite(c_clash.grad).all()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"{kind} ({device}) missing grad on {name}"
        assert torch.isfinite(param.grad).all(), f"{kind} ({device}) non-finite grad on {name}"


# ---------------------------------------------------------------------------
# 7. Finite float32 Forward + Backward Parity on Padded Variable-Length Batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ENCODER_KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_float32_forward_backward_all_parameter_grads_finite(
    kind: EncoderKind,
    device: str,
) -> None:
    """Verify full float32 forward + backward produces finite, non-trivial grads on CPU & MPS."""
    torch.manual_seed(83)
    dev = torch.device(device)
    enc = _make_encoder(kind, num_layers=3, dropout=0.0).to(dev).train()
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=4, num_atoms=40, device=dev)
    coords.requires_grad_(True)

    enc.zero_grad(set_to_none=True)
    out = enc(coords, node_feats, atom_mask)
    assert out.node_emb.dtype == torch.float32
    assert out.pooled_emb.dtype == torch.float32
    assert out.vec_emb is not None and out.vec_emb.dtype == torch.float32
    assert torch.isfinite(out.node_emb).all()
    assert torch.isfinite(out.pooled_emb).all()
    assert torch.isfinite(out.vec_emb).all()

    n_valid = atom_mask.sum().clamp(min=1).to(torch.float32)
    loss = (
        out.node_emb.pow(2).sum() / n_valid
        + out.pooled_emb.pow(2).sum() / float(coords.shape[0])
        + out.vec_emb.pow(2).sum() / n_valid
    )
    loss.backward()
    _sync_device(dev)

    assert coords.grad is not None and torch.isfinite(coords.grad).all()
    assert coords.grad[atom_mask].abs().max().item() > 1e-6
    assert coords.grad[~atom_mask].abs().max().item() == 0.0

    total_grad_norm = 0.0
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"{kind} ({device}) parameter {name} received None grad"
        assert torch.isfinite(param.grad).all(), f"{kind} ({device}) non-finite grad on {name}"
        total_grad_norm += float(param.grad.norm().item())
    assert total_grad_norm > 1e-3


# ---------------------------------------------------------------------------
# 8. Empirical Apple Silicon MPS Speed Check (B=16, N=384, K=24)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS unavailable")
def test_mps_throughput_cartformer_and_dense_painn_vs_sparse_painn() -> None:
    """Benchmark CartFormer-SE3 & DensePaiNN-SE3 vs sparse PaiNN-SE3 at B=16, N=384, K=24."""
    dev = torch.device("mps")
    torch.manual_seed(101)
    gen = torch.Generator(device="cpu").manual_seed(101)

    batch_size, num_atoms, max_neighbors = 16, 384, 24
    # Spread atoms over ~12 A binding cavity so avg degree matches realistic PDBbind (~21/atom).
    coords = (torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32) * 4.5).to(
        dev
    )
    node_feats = torch.randn((batch_size, num_atoms, 52), generator=gen, dtype=torch.float32).to(
        dev
    )
    node_feats[..., 46] = torch.tanh(node_feats[..., 46])
    atom_mask = torch.ones((batch_size, num_atoms), device=dev, dtype=torch.bool)

    models: dict[str, torch.nn.Module] = {
        "painn_sparse": PaiNNEncoder(
            node_in_dim=52,
            hidden_dim=192,
            vector_dim=64,
            out_dim=256,
            num_layers=4,
            num_rbf=24,
            cutoff=5.5,
            max_neighbors=max_neighbors,
            use_se3_chirality=True,
            dropout=0.0,
        ).to(dev),
        "dense_painn_se3": DensePaiNNSE3Encoder(
            node_in_dim=52,
            hidden_dim=192,
            vector_dim=64,
            out_dim=256,
            num_layers=4,
            num_rbf=24,
            cutoff=5.5,
            max_neighbors=max_neighbors,
            use_se3_chirality=True,
            dropout=0.0,
        ).to(dev),
        "cartformer_se3": CartFormerSE3Encoder(
            node_in_dim=52,
            hidden_dim=192,
            vector_dim=64,
            tensor_dim=24,
            out_dim=256,
            num_layers=4,
            num_heads=8,
            num_rbf=24,
            num_landmarks=8,
            cutoff=5.5,
            max_neighbors=max_neighbors,
            use_se3_chirality=True,
            use_coulomb_bias=True,
            dropout=0.0,
        ).to(dev),
    }

    timings: dict[str, tuple[float, float]] = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for _ in range(2):
                _ = model(coords, node_feats, atom_mask)
            _sync_device(dev)
            t0 = time.perf_counter()
            iters = 5
            for _ in range(iters):
                _ = model(coords, node_feats, atom_mask)
            _sync_device(dev)
            fwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

        model.train()
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            out = model(coords, node_feats, atom_mask)
            loss = out.pooled_emb.pow(2).mean()
            loss.backward()
        _sync_device(dev)

        t1 = time.perf_counter()
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            out = model(coords, node_feats, atom_mask)
            loss = out.pooled_emb.pow(2).mean()
            loss.backward()
        _sync_device(dev)
        fwdbwd_ms = (time.perf_counter() - t1) * 1000.0 / iters
        timings[name] = (fwd_ms, fwdbwd_ms)

    painn_fwd, painn_fb = timings["painn_sparse"]
    dp_fwd, dp_fb = timings["dense_painn_se3"]
    cf_fwd, cf_fb = timings["cartformer_se3"]

    # Both dense MPS-native encoders must be faster than sparse index_add_ PaiNN-SE3!
    assert dp_fb < painn_fb, (
        f"DensePaiNN-SE3 ({dp_fb:.1f} ms) should beat sparse PaiNN-SE3 ({painn_fb:.1f} ms)"
    )
    assert cf_fb < painn_fb, (
        f"CartFormer-SE3 ({cf_fb:.1f} ms) should beat sparse PaiNN-SE3 ({painn_fb:.1f} ms)"
    )
    assert dp_fwd < painn_fwd and cf_fwd < painn_fwd
