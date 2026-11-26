"""Comprehensive CPU & Apple MPS test suite for SubPocketVN-JEPA and 3D JEPA SSL pretraining.

Verifies:
  - Strict SE(3) rotation + 100 A rigid translation invariance/equivariance (pooled, node, vec,
    subpocket super-tokens, subpocket coords, slot weights, and Q=16 Perceiver bridge tokens).
  - Exact atom permutation invariance/equivariance and 0.0 dummy-atom corruption sensitivity.
  - Chiral mirror-reflection discrimination (det(P) = -1) vs E(3) toggle + synthetic (R)/(S)
    tetrahedral discrimination + achiral planar benzene & collinear alkyne consistency.
  - Dora pharmacophore/metal salient slot weighting (cols 37:42 & col 8), epoch-0 coincident
    centroid smoothness (delta = 0.5 A), boundary-free continuity, equidistant K-th neighbor
    tie-swap invariance, N=1, N<K, all-empty isolated atom, d=0 clash, empty batch row, and
    large N=1024 scaling.
  - SubPocketJEPAPretrainer: spherical block masking, stochastic Sonata radial obscuration
    (cols 47:52), exact student-frame vs rotated-teacher SSL invariance, 5-step training smoke
    test (decreasing loss, EMA teacher update, effective rank > 1.5, zero NaN/Inf grads),
    Concerto cross-modal InfoNCE loss, and jepa_ssl_ckpt save/load roundtrip.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import Tensor

from geneuclidean.config import EncoderConfig
from geneuclidean.models.encoders.subpocket_vn import (
    CompetitiveSubPocketSlotPool,
    SubPocketVNJEPAEncoder,
    build_subpocket_vn_jepa,
    smooth_pair_directions,
)
from geneuclidean.protocols import PocketEncoderProtocol
from geneuclidean.training.jepa_ssl import (
    SubPocketJEPAPretrainer,
    apply_sonata_spatial_obscuration,
    compute_effective_rank,
    sample_random_so3_rotations,
    sample_spherical_block_mask,
)

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def _random_so3_matrix(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a single Haar-uniform proper rotation matrix R in SO(3) (det(R) = +1)."""
    return sample_random_so3_rotations(1, device=device, generator=gen)[0]


def _make_pocket_batch(
    batch_size: int = 2,
    num_atoms: int = 36,
    node_in_dim: int = 52,
    device: torch.device | str = "cpu",
    seed: int = 123,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a realistic chiral 3D pocket batch with pharmacophore flags and variable masks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32) * 2.5
    node_feats = torch.randn(
        (batch_size, num_atoms, node_in_dim), generator=gen, dtype=torch.float32
    )
    if node_in_dim >= 52:
        # Realistic bounded binary flags on metal (8) and pharmacophore (37:42), charges (46),
        # and positive radial descriptors (47:52).
        node_feats[..., 8] = (torch.rand((batch_size, num_atoms), generator=gen) > 0.92).float()
        node_feats[..., 37:42] = (
            torch.rand((batch_size, num_atoms, 5), generator=gen) > 0.75
        ).float()
        node_feats[..., 46] = torch.rand((batch_size, num_atoms), generator=gen) * 2.0 - 1.0
        node_feats[..., 47:52] = torch.rand((batch_size, num_atoms, 5), generator=gen) + 0.1

    atom_mask = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
    for b in range(batch_size):
        valid_count = max(10, num_atoms - 5 - 3 * b)
        atom_mask[b, : min(valid_count, num_atoms)] = True

    dev = torch.device(device)
    return coords.to(dev), node_feats.to(dev), atom_mask.to(dev)


def _small_encoder(
    *,
    use_se3_chirality: bool = True,
    use_coulomb_bias: bool = True,
    num_layers: int = 2,
    num_subpockets: int = 24,
    num_salient_slots: int = 8,
    dropout: float = 0.0,
) -> SubPocketVNJEPAEncoder:
    return SubPocketVNJEPAEncoder(
        node_in_dim=52,
        hidden_dim=64,
        vector_dim=16,
        out_dim=64,
        num_layers=num_layers,
        num_rbf=16,
        cutoff=5.5,
        max_neighbors=24,
        num_heads=4,
        num_subpockets=num_subpockets,
        num_salient_slots=num_salient_slots,
        global_heads=4,
        use_se3_chirality=use_se3_chirality,
        use_coulomb_bias=use_coulomb_bias,
        num_perceiver_tokens=16,
        dropout=dropout,
    )


def _assert_no_float64(module: torch.nn.Module) -> None:
    for name, param in module.named_parameters():
        assert param.dtype != torch.float64, f"float64 parameter found: {name}"
    for name, buf in module.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer found: {name}"


def test_protocol_factory_and_zero_float64() -> None:
    """Verify PocketEncoderProtocol compliance, alias config normalization, and zero float64."""
    cfg = EncoderConfig(
        arch="subpocket_vn",
        node_in_dim=52,
        hidden_dim=64,
        vector_dim=16,
        out_dim=64,
        num_layers=2,
        num_rbf=16,
        num_heads=4,
        num_subpockets=24,
        dropout=0.0,
    )
    assert cfg.arch == "subpocket_vn_jepa"
    enc = build_subpocket_vn_jepa(cfg, max_neighbors=24)
    assert isinstance(enc, PocketEncoderProtocol)
    assert isinstance(enc, SubPocketVNJEPAEncoder)
    _assert_no_float64(enc)

    # Also verify float64 inputs cast cleanly down to float32 without dtype promotion.
    coords, feats, mask = _make_pocket_batch(batch_size=2, num_atoms=20, device="cpu", seed=3)
    out = enc(coords.double(), feats.double(), mask)
    assert out.node_emb.dtype == torch.float32
    assert out.pooled_emb.dtype == torch.float32
    assert out.vec_emb is not None and out.vec_emb.dtype == torch.float32

    # Regression: CompetitiveSubPocketSlotPool.pool casts float64 priors cleanly.
    pool = CompetitiveSubPocketSlotPool(32, 8)
    s_d = torch.randn((1, 6, 32), dtype=torch.float64)
    v_d = torch.randn((1, 6, 8, 3), dtype=torch.float64)
    c_d = torch.randn((1, 6, 3), dtype=torch.float64)
    m_b = torch.ones((1, 6), dtype=torch.bool)
    p_s, p_v, p_c, p_w, p_u, _, p_q = pool.pool(
        s_d,
        v_d,
        c_d,
        m_b,
        salience_indicator=torch.ones((1, 6), dtype=torch.float64),
        charge_prior=torch.ones((1, 6), dtype=torch.float64),
    )
    for t in (p_s, p_v, p_c, p_w, p_u, p_q):
        assert t.dtype == torch.float32


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_so3_rotation_and_100a_translation_equivariance(device: str) -> None:
    """Verify exact SO(3) + 100 A translation invariance/equivariance on CPU and MPS."""
    torch.manual_seed(42)
    dev = torch.device(device)
    enc = _small_encoder(use_se3_chirality=True, dropout=0.0).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=36, device=dev)

    with torch.no_grad():
        ref_sub = enc.encode_with_subpockets(coords, node_feats, atom_mask)
        ref = ref_sub.encoder_output
        ref_bridge = enc.perceiver_bridge_tokens(coords, node_feats, atom_mask)

    assert ref.vec_emb is not None
    assert ref.vec_emb.abs().max().item() > 1e-3, "vec_emb is trivially zero"
    assert ref_bridge.shape == (2, 16, 64)
    assert ref_sub.subpocket_tokens.shape == (2, 24, 64)
    assert ref_sub.subpocket_coords.shape == (2, 24, 3)
    assert ref_sub.slot_weights.shape == (2, 24, 36)

    gen = torch.Generator(device="cpu").manual_seed(2026)
    tol = 5e-5
    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        coords_t = coords @ rot.T + trans

        with torch.no_grad():
            out_sub = enc.encode_with_subpockets(coords_t, node_feats, atom_mask)
            out = out_sub.encoder_output
            out_bridge = enc.perceiver_bridge_tokens(coords_t, node_feats, atom_mask)

        assert out.vec_emb is not None
        pool_err = float((out.pooled_emb - ref.pooled_emb).abs().max().item())
        node_err = float((out.node_emb - ref.node_emb).abs().max().item())
        vec_err = float((out.vec_emb - ref.vec_emb @ rot.T).abs().max().item())
        sub_tok_err = float(
            (out_sub.subpocket_tokens - ref_sub.subpocket_tokens).abs().max().item()
        )
        sub_coord_err = float(
            (out_sub.subpocket_coords - ref_sub.subpocket_coords @ rot.T).abs().max().item()
        )
        sub_vec_err = float(
            (out_sub.subpocket_vec - ref_sub.subpocket_vec @ rot.T).abs().max().item()
        )
        slot_w_err = float((out_sub.slot_weights - ref_sub.slot_weights).abs().max().item())
        bridge_err = float((out_bridge - ref_bridge).abs().max().item())

        assert pool_err < tol, f"({device}) pooled_emb SO(3)+100A err={pool_err:.2e}"
        assert node_err < tol, f"({device}) node_emb SO(3)+100A err={node_err:.2e}"
        assert vec_err < tol, f"({device}) vec_emb SO(3)+100A err={vec_err:.2e}"
        assert sub_tok_err < tol, f"({device}) subpocket_tokens err={sub_tok_err:.2e}"
        assert sub_coord_err < tol, f"({device}) subpocket_coords err={sub_coord_err:.2e}"
        assert sub_vec_err < tol, f"({device}) subpocket_vec err={sub_vec_err:.2e}"
        assert slot_w_err < tol, f"({device}) slot_weights err={slot_w_err:.2e}"
        assert bridge_err < tol, f"({device}) perceiver_bridge_tokens err={bridge_err:.2e}"


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_permutation_invariance_and_dummy_zero_sensitivity(device: str) -> None:
    """Verify exact atom permutation equivariance and 0.0 sensitivity to dummy-atom corruption."""
    torch.manual_seed(17)
    dev = torch.device(device)
    enc = _small_encoder(use_se3_chirality=True, dropout=0.0).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=32, device=dev)

    gen = torch.Generator(device="cpu").manual_seed(303)
    perm = torch.randperm(coords.shape[1], generator=gen).to(dev)

    with torch.no_grad():
        ref = enc.encode_with_subpockets(coords, node_feats, atom_mask)
        perm_out = enc.encode_with_subpockets(
            coords[:, perm], node_feats[:, perm], atom_mask[:, perm]
        )

    assert ref.encoder_output.vec_emb is not None and perm_out.encoder_output.vec_emb is not None
    assert (
        perm_out.encoder_output.pooled_emb - ref.encoder_output.pooled_emb
    ).abs().max().item() < 1e-5
    assert (
        perm_out.encoder_output.node_emb - ref.encoder_output.node_emb[:, perm]
    ).abs().max().item() < 1e-5
    assert (
        perm_out.encoder_output.vec_emb - ref.encoder_output.vec_emb[:, perm]
    ).abs().max().item() < 1e-5
    assert (perm_out.subpocket_tokens - ref.subpocket_tokens).abs().max().item() < 1e-5

    # Corrupt dummy atoms with huge coordinates and features.
    coords_corrupt = coords.clone()
    feats_corrupt = node_feats.clone()
    coords_corrupt[~atom_mask] = 999.0
    feats_corrupt[~atom_mask] = torch.randn_like(feats_corrupt[~atom_mask]) * 1e4

    with torch.no_grad():
        corrupt = enc.encode_with_subpockets(coords_corrupt, feats_corrupt, atom_mask)

    assert corrupt.encoder_output.vec_emb is not None
    assert corrupt.encoder_output.node_emb[~atom_mask].abs().max().item() == 0.0
    assert corrupt.encoder_output.vec_emb[~atom_mask].abs().max().item() == 0.0
    assert corrupt.slot_weights.masked_select(~atom_mask.unsqueeze(1)).abs().max().item() == 0.0
    tol_dummy = 0.0 if device == "cpu" else 5e-6
    assert (
        corrupt.encoder_output.pooled_emb - ref.encoder_output.pooled_emb
    ).abs().max().item() <= tol_dummy
    assert (
        corrupt.encoder_output.node_emb - ref.encoder_output.node_emb
    ).abs().max().item() <= tol_dummy


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_chiral_reflection_discrimination_and_achiral_planar_collinear(device: str) -> None:
    """Verify det(P)=-1 chiral sensitivity, E(3) toggle, (R)/(S), benzene, and alkyne."""
    torch.manual_seed(31)
    dev = torch.device(device)
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=32, device=dev)
    reflect = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=dev, dtype=torch.float32))
    coords_mirror = coords @ reflect.T

    # 1. SE(3) chiral mode distinguishes mirror-image 3D pockets (> 1e-3).
    enc_se3 = _small_encoder(use_se3_chirality=True, dropout=0.0).to(dev).eval()
    with torch.no_grad():
        out_l = enc_se3(coords, node_feats, atom_mask)
        out_d = enc_se3(coords_mirror, node_feats, atom_mask)
    pool_chiral_diff = float((out_l.pooled_emb - out_d.pooled_emb).abs().max().item())
    node_chiral_diff = float((out_l.node_emb - out_d.node_emb).abs().max().item())
    assert pool_chiral_diff > 1e-3, f"Expected chiral diff > 1e-3, got {pool_chiral_diff:.2e}"
    assert node_chiral_diff > 1e-3, f"Expected node chiral diff > 1e-3, got {node_chiral_diff:.2e}"

    # 2. E(3) non-chiral mode is strictly reflection-invariant.
    enc_e3 = _small_encoder(use_se3_chirality=False, dropout=0.0).to(dev).eval()
    with torch.no_grad():
        e3_l = enc_e3(coords, node_feats, atom_mask)
        e3_d = enc_e3(coords_mirror, node_feats, atom_mask)
    tol_e3 = 2e-5 if device == "cpu" else 1e-4
    assert (e3_l.pooled_emb - e3_d.pooled_emb).abs().max().item() < tol_e3
    assert (e3_l.node_emb - e3_d.node_emb).abs().max().item() < tol_e3

    # 3. Synthetic (R) vs (S) tetrahedral chiral center with 4 distinct substituents.
    tetra_r = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [-0.5, 1.35, 0.0],
                [-0.45, -0.75, 1.25],
                [-0.45, -0.60, -1.25],
            ]
        ],
        device=dev,
        dtype=torch.float32,
    )
    tetra_s = tetra_r @ reflect.T
    tetra_feats = torch.zeros((1, 5, 52), device=dev, dtype=torch.float32)
    for idx, elem_col in enumerate((0, 1, 2, 5, 6)):
        tetra_feats[0, idx, elem_col] = 1.0
        tetra_feats[0, idx, 10 + idx] = 1.0
    tetra_mask = torch.ones((1, 5), device=dev, dtype=torch.bool)
    with torch.no_grad():
        r_out = enc_se3(tetra_r, tetra_feats, tetra_mask)
        s_out = enc_se3(tetra_s, tetra_feats, tetra_mask)
    assert (r_out.pooled_emb - s_out.pooled_emb).abs().max().item() > 1e-3

    # 4. Achiral planar benzene ring (z == 0): reflection-invariant + finite grads in SE(3).
    angles = torch.linspace(0.0, 2.0 * math.pi * (5.0 / 6.0), 6, dtype=torch.float32)
    benzene = (
        torch.stack(
            (1.4 * torch.cos(angles), 1.4 * torch.sin(angles), torch.zeros_like(angles)), -1
        )
        .unsqueeze(0)
        .to(dev)
    )
    benz_feats = torch.zeros((1, 6, 52), device=dev, dtype=torch.float32)
    benz_feats[:, :, 0] = 1.0  # Carbon
    benz_feats[:, :, 37] = 1.0  # Aromatic
    benz_mask = torch.ones((1, 6), device=dev, dtype=torch.bool)

    benzene_req = benzene.clone().requires_grad_(True)
    b_out = enc_se3(benzene_req, benz_feats, benz_mask)
    b_loss = b_out.pooled_emb.sum() + b_out.node_emb.sum()
    b_loss.backward()
    assert benzene_req.grad is not None and bool(torch.isfinite(benzene_req.grad).all().item())
    with torch.no_grad():
        b_ref = enc_se3(benzene, benz_feats, benz_mask)
        b_mir = enc_se3(benzene @ reflect.T, benz_feats, benz_mask)
    assert (b_ref.pooled_emb - b_mir.pooled_emb).abs().max().item() < 5e-5
    assert (b_ref.node_emb - b_mir.node_emb).abs().max().item() < 5e-5

    # 5. Achiral collinear alkyne (y == z == 0): reflection-invariant + finite gradients.
    alkyne = torch.tensor(
        [[[-2.6, 0.0, 0.0], [-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]],
        device=dev,
        dtype=torch.float32,
    )
    alk_feats = torch.zeros((1, 4, 52), device=dev, dtype=torch.float32)
    alk_feats[:, :, 0] = 1.0
    alk_mask = torch.ones((1, 4), device=dev, dtype=torch.bool)

    alk_req = alkyne.clone().requires_grad_(True)
    a_out = enc_se3(alk_req, alk_feats, alk_mask)
    (a_out.pooled_emb.sum() + a_out.node_emb.sum()).backward()
    assert alk_req.grad is not None and bool(torch.isfinite(alk_req.grad).all().item())
    with torch.no_grad():
        a_ref = enc_se3(alkyne, alk_feats, alk_mask)
        a_mir = enc_se3(alkyne @ reflect.T, alk_feats, alk_mask)
    assert (a_ref.pooled_emb - a_mir.pooled_emb).abs().max().item() < 5e-5


@pytest.mark.parametrize("device", DEVICES)
def test_dora_salient_slots_bridge_and_edge_cases(device: str) -> None:
    """Verify Dora pharmacophore/metal slots, Q=16 bridge, tie-swap, N=1, N<K, and N=1024."""
    torch.manual_seed(77)
    dev = torch.device(device)
    enc = _small_encoder(use_se3_chirality=True, dropout=0.0).to(dev).eval()

    # 1. Dora salient slot weighting: pharmacophore/metal atoms attract > mass in last 8 slots.
    n_atoms = 32
    gen = torch.Generator(device="cpu").manual_seed(91)
    coords = (torch.randn((2, n_atoms, 3), generator=gen, dtype=torch.float32) * 3.0).to(dev)
    feats = torch.zeros((2, n_atoms, 52), device=dev, dtype=torch.float32)
    feats[:, :, 0] = 1.0  # Identical baseline element across all atoms
    salient_mask = torch.zeros((2, n_atoms), device=dev, dtype=torch.bool)
    salient_mask[:, :8] = True
    feats[:, :6, 37:42] = 1.0  # Aromatic / HBA / HBD / ionizable pharmacophores
    feats[:, 6:8, 8] = 1.0  # Catalytic metal ions
    atom_mask = torch.ones((2, n_atoms), device=dev, dtype=torch.bool)

    with torch.no_grad():
        sub_out = enc.encode_with_subpockets(coords, feats, atom_mask)
        bridge = enc.perceiver_bridge_tokens(coords, feats, atom_mask)

    assert bridge.shape == (2, 16, 64)
    assert bool(torch.isfinite(bridge).all().item())

    # Compare slot mass allocated to salient atoms between geometric (0:16) and Dora (16:24).
    geo_salient_mass = float(sub_out.slot_weights[:, :16, :8].sum(dim=-1).mean().item())
    dora_salient_mass = float(sub_out.slot_weights[:, 16:, :8].sum(dim=-1).mean().item())
    assert dora_salient_mass > geo_salient_mass + 0.05, (
        f"Expected Dora slots ({dora_salient_mass:.3f}) > geometric slots ({geo_salient_mass:.3f})"
    )

    # 2. Epoch-0 coincident centroid direction guard (delta = 0.5 A) -> zero NaN/Inf gradients.
    coincident = torch.zeros((1, 8, 3), device=dev, dtype=torch.float32, requires_grad=True)
    dist, sdir = smooth_pair_directions(coincident, smooth_delta=0.5)
    torch.autograd.backward(dist.sum() + sdir.sum())
    assert coincident.grad is not None and bool(torch.isfinite(coincident.grad).all().item())

    # 3. Single-atom ion (N=1), small ligand (N=12 < K=24), isolated atom, d=0 clash, empty row.
    for case_coords, case_feats, case_mask in [
        # Single-atom Zn2+ ion (N=1)
        (
            torch.tensor([[[1.5, -2.0, 0.5]]], device=dev),
            torch.zeros((1, 1, 52), device=dev),
            torch.ones((1, 1), device=dev, dtype=torch.bool),
        ),
        # Small ligand N=12 < K=24
        _make_pocket_batch(batch_size=1, num_atoms=12, device=dev, seed=5),
        # Duplicate coordinate clash (d=0) + isolated atom 60 A away
        (
            torch.tensor(
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [60.0, 0.0, 0.0]]],
                device=dev,
            ),
            torch.randn((1, 4, 52), device=dev),
            torch.ones((1, 4), device=dev, dtype=torch.bool),
        ),
    ]:
        enc.zero_grad(set_to_none=True)
        c_req = case_coords.clone().requires_grad_(True)
        out = enc(c_req, case_feats, case_mask)
        assert bool(torch.isfinite(out.pooled_emb).all().item())
        assert bool(torch.isfinite(out.node_emb).all().item())
        assert out.vec_emb is not None and bool(torch.isfinite(out.vec_emb).all().item())
        (out.pooled_emb.sum() + out.node_emb.sum() + out.vec_emb.sum()).backward()
        assert c_req.grad is not None and bool(torch.isfinite(c_req.grad).all().item())

    # Empty pocket row in batch -> strictly 0.0 outputs and finite gradients.
    e_coords, e_feats, e_mask = _make_pocket_batch(batch_size=2, num_atoms=16, device=dev, seed=9)
    e_mask[1] = False
    e_out = enc.encode_with_subpockets(e_coords, e_feats, e_mask)
    assert e_out.encoder_output.pooled_emb[1].abs().max().item() == 0.0
    assert e_out.subpocket_tokens[1].abs().max().item() == 0.0
    e_bridge = enc.perceiver_bridge_tokens(e_coords, e_feats, e_mask)
    assert e_bridge[1].abs().max().item() == 0.0

    # 4. Equidistant K-th neighbor tie-swap invariance (26-neighbor cube-lattice + mixed shells).
    # True equidistant cube-lattice sphere ({-1, 0, 1}^3 \ {0}) normalized to exact radius 2.0 A,
    # plus 4 interior atoms at 1.2 A so max_neighbors=12 forces exact K-th boundary ties while
    # interior neighbors carry edge_env > 0. Permuting tied atoms must leave outputs invariant!
    cube_pts = [
        [float(dx), float(dy), float(dz)]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    cube_unit = torch.tensor(cube_pts, dtype=torch.float32)
    cube_unit = cube_unit / cube_unit.norm(dim=-1, keepdim=True)
    # 4a. Pure 26-neighbor cube-lattice sphere (r = 2.0 A, max_neighbors=12)
    pure_tie_coords = (
        torch.cat((torch.zeros((1, 3), dtype=torch.float32), cube_unit * 2.0), dim=0)
        .unsqueeze(0)
        .to(dev)
    )
    pure_tie_feats = torch.randn((1, 27, 52), generator=gen, dtype=torch.float32).to(dev)
    pure_tie_mask = torch.ones((1, 27), device=dev, dtype=torch.bool)
    # 4b. Mixed interior (6 atoms at 1.2 A) + 26-neighbor outer tied shell (r = 2.5 A, K = 12)
    inner_pts = torch.tensor(
        [
            [1.2, 0.0, 0.0],
            [-1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, -1.2, 0.0],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, -1.2],
        ],
        dtype=torch.float32,
    )
    mixed_tie_coords = (
        torch.cat((torch.zeros((1, 3), dtype=torch.float32), inner_pts, cube_unit * 2.5), dim=0)
        .unsqueeze(0)
        .to(dev)
    )
    mixed_tie_feats = torch.randn((1, 33, 52), generator=gen, dtype=torch.float32).to(dev)
    mixed_tie_mask = torch.ones((1, 33), device=dev, dtype=torch.bool)

    enc_k12 = (
        SubPocketVNJEPAEncoder(
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=2,
            num_rbf=16,
            cutoff=5.5,
            max_neighbors=12,
            num_heads=4,
            num_subpockets=24,
            num_salient_slots=8,
            global_heads=4,
            use_se3_chirality=True,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )

    for label, t_coords, t_feats, t_mask, shell_start in (
        ("pure_cube", pure_tie_coords, pure_tie_feats, pure_tie_mask, 1),
        ("mixed_cube", mixed_tie_coords, mixed_tie_feats, mixed_tie_mask, 7),
    ):
        n_tot = t_coords.shape[1]
        shell_perm = torch.randperm(n_tot - shell_start, generator=gen) + shell_start
        full_perm = torch.cat((torch.arange(shell_start), shell_perm)).to(dev)
        rot = _random_so3_matrix(gen, dev)
        trans = torch.tensor([[[73.1, -54.8, 91.4]]], device=dev, dtype=torch.float32)

        with torch.no_grad():
            t_ref = enc_k12(t_coords, t_feats, t_mask)
            t_perm = enc_k12(t_coords[:, full_perm], t_feats[:, full_perm], t_mask[:, full_perm])
            t_rot = enc_k12(t_coords @ rot.T + trans, t_feats, t_mask)

        perm_err = float((t_perm.pooled_emb - t_ref.pooled_emb).abs().max().item())
        rot_err = float((t_rot.pooled_emb - t_ref.pooled_emb).abs().max().item())
        assert perm_err < 1e-5, f"({device}) {label} tie-swap perm err={perm_err:.2e}"
        assert rot_err < 5e-5, f"({device}) {label} tie-swap rot+100A err={rot_err:.2e}"

        # Finite coordinate backward gradients on tied equidistant lattice.
        t_req = t_coords.clone().requires_grad_(True)
        t_bwd = enc_k12(t_req, t_feats, t_mask)
        torch.autograd.backward(t_bwd.pooled_emb.sum() + t_bwd.node_emb.sum())
        assert t_req.grad is not None and bool(torch.isfinite(t_req.grad).all().item())

    # 5. Large N=1024 scaling test (forward + backward finite).
    l_coords, l_feats, l_mask = _make_pocket_batch(
        batch_size=2, num_atoms=1024, device=dev, seed=404
    )
    enc.zero_grad(set_to_none=True)
    l_out = enc(l_coords, l_feats, l_mask)
    assert l_out.node_emb.shape == (2, 1024, 64)
    assert l_out.pooled_emb.shape == (2, 64)
    assert bool(torch.isfinite(l_out.pooled_emb).all().item())
    (l_out.pooled_emb.sum() + l_out.node_emb.mean()).backward()
    for name, p in enc.named_parameters():
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all().item()), f"Non-finite grad at N=1024: {name}"


@pytest.mark.parametrize("device", DEVICES)
def test_jepa_ssl_obscuration_invariance_and_5step_training(
    device: str,
    tmp_path: Path,
) -> None:
    """Verify Sonata obscuration, student-frame invariance, 5-step SSL, Concerto, and ckpt."""
    torch.manual_seed(101)
    dev = torch.device(device)
    coords, node_feats, atom_mask = _make_pocket_batch(
        batch_size=4, num_atoms=48, device=dev, seed=202
    )
    # Add heavy dummy padding to verify VICReg & effective rank ignore padded zeros cleanly.
    atom_mask[:, 30:] = False

    # 1. Spherical block masking invariants.
    gen = torch.Generator(device="cpu").manual_seed(7)
    vis_mask, tgt_mask = sample_spherical_block_mask(coords, atom_mask, generator=gen)
    assert not bool((vis_mask & tgt_mask).any().item()), "visible & target masks must be disjoint"
    assert bool(((vis_mask | tgt_mask) == atom_mask).all().item())
    for b in range(4):
        assert int(vis_mask[b].sum().item()) >= 4
        assert int(tgt_mask[b].sum().item()) >= 1

    # 2. Sonata radial obscuration p=1.0 check (cols 47:52 strictly zeroed on visible atoms).
    _, obs_feats_all, frac_all = apply_sonata_spatial_obscuration(
        coords,
        node_feats,
        vis_mask,
        coord_jitter_std=0.0,
        radial_drop_prob=1.0,
        feat_drop_prob=0.0,
        generator=gen,
    )
    assert float(frac_all.item()) == pytest.approx(1.0)
    assert obs_feats_all[..., 47:52].abs().max().item() == 0.0
    assert obs_feats_all[~vis_mask].abs().max().item() == 0.0

    _, obs_feats_none, frac_none = apply_sonata_spatial_obscuration(
        coords,
        node_feats,
        vis_mask,
        coord_jitter_std=0.0,
        radial_drop_prob=0.0,
        feat_drop_prob=0.0,
        generator=gen,
    )
    assert float(frac_none.item()) == pytest.approx(0.0)
    assert obs_feats_none[vis_mask][:, 47:52].abs().max().item() > 0.05

    # 3. Effective rank collapse diagnostic sanity check.
    collapsed = torch.ones((32, 16), device=dev, dtype=torch.float32)
    assert float(compute_effective_rank(collapsed).item()) == pytest.approx(1.0, abs=1e-4)
    isotropic = torch.eye(16, device=dev, dtype=torch.float32).repeat(2, 1)
    assert float(compute_effective_rank(isotropic).item()) > 10.0

    # 4. Exact student-frame vs rotated-teacher JEPA SSL loss invariance (Theorem 3).
    encoder = _small_encoder(use_se3_chirality=True, dropout=0.0).to(dev)
    pretrainer = SubPocketJEPAPretrainer(
        encoder,
        predictor_layers=2,
        predictor_heads=4,
        ema_decay_start=0.90,
        ema_decay_end=0.99,
        total_steps=20,
        coord_jitter_std=0.10,
        radial_drop_prob=0.80,
        feat_drop_prob=0.10,
        vicreg_weight=0.1,
        cross_modal_dim=64,
        cross_modal_proj_dim=32,
        cross_modal_weight=0.25,
    ).to(dev)
    _assert_no_float64(pretrainer)
    pretrainer.eval()

    r_stu = sample_random_so3_rotations(4, device=dev, generator=gen)
    r_tea_1 = sample_random_so3_rotations(4, device=dev, generator=gen)
    r_tea_2 = sample_random_so3_rotations(4, device=dev, generator=gen)
    t_stu = (torch.rand((4, 1, 3), generator=gen, dtype=torch.float32) * 100.0 - 50.0).to(dev)
    coords_stu_rot = torch.bmm(coords, r_stu.transpose(-1, -2)) + t_stu

    with torch.no_grad():
        ssl_ref = pretrainer.forward_ssl(
            coords,
            node_feats,
            atom_mask,
            visible_mask=vis_mask,
            target_mask=tgt_mask,
            teacher_rot=r_tea_1,
            apply_obscuration=False,
            update_center=False,
        )
        ssl_rot = pretrainer.forward_ssl(
            coords_stu_rot,
            node_feats,
            atom_mask,
            visible_mask=vis_mask,
            target_mask=tgt_mask,
            teacher_rot=r_tea_2,
            apply_obscuration=False,
            update_center=False,
        )
    ssl_inv_err = abs(float(ssl_rot.total_loss.item()) - float(ssl_ref.total_loss.item()))
    assert ssl_inv_err < 1e-4, f"({device}) JEPA SSL student/teacher rotation err={ssl_inv_err:.2e}"

    # 5. 5-step training smoke test with Concerto cross-modal InfoNCE loss.
    init_teacher_params = [p.detach().clone() for p in pretrainer.teacher.parameters()]
    ligand_emb = torch.randn((4, 64), generator=gen, dtype=torch.float32).to(dev)
    batch = {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "ligand_emb": ligand_emb,
    }
    optimizer = torch.optim.AdamW(
        [p for p in pretrainer.parameters() if p.requires_grad],
        lr=3e-3,
        weight_decay=1e-4,
    )

    # Use fixed deterministic mask/obscuration seed per step or mild jitter for clean convergence.
    losses: list[float] = []
    ranks: list[float] = []
    cross_losses: list[float] = []
    pretrainer.coord_jitter_std = 0.02
    pretrainer.feat_drop_prob = 0.0

    for step in range(5):
        step_gen = torch.Generator(device="cpu").manual_seed(500)
        metrics = pretrainer.pretrain_step(
            batch,
            optimizer,
            grad_clip=1.0,
            ema_decay=0.85,
            generator=step_gen,
        )
        losses.append(metrics["loss"])
        ranks.append(metrics["effective_rank"])
        cross_losses.append(metrics["cross_modal_loss"])
        assert math.isfinite(metrics["loss"]), f"Non-finite loss at step {step}"
        assert math.isfinite(metrics["grad_norm"]), f"Non-finite grad_norm at step {step}"
        assert metrics["cross_modal_loss"] > 0.0

        # Confirm zero NaN/Inf gradients and strictly frozen teacher.
        for name, param in pretrainer.student.named_parameters():
            if param.grad is not None:
                assert bool(torch.isfinite(param.grad).all().item()), f"Non-finite grad: {name}"
        for t_param in pretrainer.teacher.parameters():
            assert not t_param.requires_grad
            assert t_param.grad is None

    assert losses[-1] < losses[0], f"({device}) Expected loss decrease: {losses}"
    assert min(ranks) > 1.5, f"({device}) Effective rank collapsed: {ranks}"

    teacher_delta = max(
        float((p_now - p_init).abs().max().item())
        for p_now, p_init in zip(pretrainer.teacher.parameters(), init_teacher_params, strict=True)
    )
    assert teacher_delta > 1e-5, f"({device}) EMA teacher failed to diverge from init"

    # 6. Checkpoint save/load roundtrip via build_subpocket_vn_jepa(config.jepa_ssl_ckpt).
    ckpt_path = tmp_path / f"subpocket_jepa_{device}.pt"
    pretrainer.save_checkpoint(ckpt_path, use_teacher=True, metadata={"steps": 5})

    cfg = EncoderConfig(
        arch="subpocket_vn_jepa",
        node_in_dim=52,
        hidden_dim=64,
        vector_dim=16,
        out_dim=64,
        num_layers=2,
        num_rbf=16,
        cutoff=5.5,
        num_heads=4,
        num_subpockets=24,
        dropout=0.0,
        jepa_ssl_ckpt=str(ckpt_path),
    )
    reloaded = build_subpocket_vn_jepa(cfg, max_neighbors=24).to(dev).eval()
    pretrainer.teacher.eval()
    with torch.no_grad():
        tea_ref = pretrainer.teacher(coords, node_feats, atom_mask)
        rel_out = reloaded(coords, node_feats, atom_mask)
    assert (rel_out.pooled_emb - tea_ref.pooled_emb).abs().max().item() < 1e-6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS unavailable")
def test_mps_full_scale_performance_smoke() -> None:
    """Verify full-scale SubPocketVNJEPAEncoder runs cleanly & beats PaiNN-SE3 on Apple MPS."""
    import time

    from geneuclidean.models.encoders.painn import PaiNNEncoder

    dev = torch.device("mps")
    cfg = EncoderConfig(
        arch="subpocket_vn_jepa",
        node_in_dim=52,
        hidden_dim=192,
        vector_dim=64,
        out_dim=256,
        num_layers=4,
        num_rbf=24,
        cutoff=5.5,
        num_heads=8,
        num_subpockets=24,
        dropout=0.0,
    )
    enc = build_subpocket_vn_jepa(cfg, max_neighbors=24).to(dev)
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

    def _time_fwd_bwd(
        module: SubPocketVNJEPAEncoder | PaiNNEncoder,
        coords: Tensor,
        feats: Tensor,
        mask: Tensor,
    ) -> float:
        for _ in range(2):
            module.zero_grad(set_to_none=True)
            out = module(coords, feats, mask)
            torch.autograd.backward(out.pooled_emb.sum() + out.node_emb.mean())
        torch.mps.synchronize()
        iters = 3
        t0 = time.perf_counter()
        for _ in range(iters):
            module.zero_grad(set_to_none=True)
            out = module(coords, feats, mask)
            torch.autograd.backward(out.pooled_emb.sum() + out.node_emb.mean())
        torch.mps.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / float(iters)

    c_384, f_384, m_384 = _make_pocket_batch(batch_size=16, num_atoms=384, device=dev, seed=11)
    sub_384_ms = _time_fwd_bwd(enc, c_384, f_384, m_384)
    painn_384_ms = _time_fwd_bwd(painn, c_384, f_384, m_384)
    assert sub_384_ms < painn_384_ms * 1.15, (
        f"SubPocketVN ({sub_384_ms:.1f} ms) slower than PaiNN-SE3 ({painn_384_ms:.1f} ms)"
    )

    c_1024, f_1024, m_1024 = _make_pocket_batch(batch_size=4, num_atoms=1024, device=dev, seed=12)
    sub_1024_ms = _time_fwd_bwd(enc, c_1024, f_1024, m_1024)
    assert sub_1024_ms < 800.0, f"MPS fwd+bwd too slow for B=4, N=1024: {sub_1024_ms:.1f} ms"
