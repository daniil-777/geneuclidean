"""Comprehensive CPU + Apple MPS tests for NanoFold3 core folding modules.

Verifies mathematical invariants, Apple Silicon MPS ``float32`` native execution
(under ``PYTORCH_ENABLE_MPS_FALLBACK=0``), reflection-safe Kabsch alignment, Pairformer
& SimpleFold trunks, EDM & Flow-Matching all-atom diffusion, physics co-folding losses,
AF3 Alg. 31 confidence & affinity heads, end-to-end ``.cofold()`` / ``.fold_sequence()``,
and live MPS latency benchmarks at ``L=64`` and ``L=128`` (``N_atoms=256``).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from geneuclidean.folding.confidence import (
    ConfidenceHead,
    ConfidenceLoss,
    compute_tm_scores,
)
from geneuclidean.folding.diffusion import (
    AllAtomDiffusionModule,
    CoFoldingLoss,
    KabschWeightedRigidAlign,
    compute_signed_tetrahedral_volume,
)
from geneuclidean.folding.nanofold3 import (
    ComplexBatch,
    NanoFold3,
    build_nanofold3,
)
from geneuclidean.folding.pairformer import (
    PairBiasedAttention,
    PairformerStack,
    SimpleFoldTrunk,
    TriangleAttention,
    TriangleMultiplicativeUpdate,
)
from geneuclidean.models.encoders.gaussian_ops import cofactor_det_3x3


def _available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


DEVICES = _available_devices()

_MINI_POCKET_PDB = """\
ATOM      1  N   ASP A 103       0.000   0.000   0.000  1.00 85.00           N
ATOM      2  CA  ASP A 103       1.458   0.000   0.000  1.00 86.00           C
ATOM      3  C   ASP A 103       2.009   1.420   0.000  1.00 85.00           C
ATOM      4  O   ASP A 103       1.251   2.390   0.000  1.00 84.00           O
ATOM      5  CB  ASP A 103       1.988  -0.760  -1.220  1.00 82.00           C
ATOM      6  CG  ASP A 103       3.502  -0.790  -1.280  1.00 81.00           C
ATOM      7  OD1 ASP A 103       4.120   0.290  -1.350  1.00 80.00           O
ATOM      8  OD2 ASP A 103       4.060  -1.910  -1.260  1.00 80.00           O
ATOM      9  N   PHE A 104       3.320   1.520   0.000  1.00 87.00           N
ATOM     10  CA  PHE A 104       4.000   2.810   0.000  1.00 88.00           C
ATOM     11  C   PHE A 104       5.510   2.650   0.000  1.00 87.00           C
ATOM     12  O   PHE A 104       6.050   1.550   0.000  1.00 86.00           O
ATOM     13  CB  PHE A 104       3.550   3.620   1.220  1.00 84.00           C
ATOM     14  CG  PHE A 104       2.080   3.950   1.280  1.00 83.00           C
ATOM     15  CD1 PHE A 104       1.580   5.120   0.710  1.00 82.00           C
ATOM     16  CD2 PHE A 104       1.200   3.080   1.910  1.00 82.00           C
ATOM     17  CE1 PHE A 104       0.220   5.410   0.770  1.00 81.00           C
ATOM     18  CE2 PHE A 104      -0.160   3.370   1.970  1.00 81.00           C
ATOM     19  CZ  PHE A 104      -0.650   4.540   1.400  1.00 81.00           C
ATOM     20  N   SER A 105       6.180   3.790   0.000  1.00 88.00           N
ATOM     21  CA  SER A 105       7.630   3.820   0.000  1.00 89.00           C
ATOM     22  C   SER A 105       8.180   5.240   0.000  1.00 88.00           C
ATOM     23  O   SER A 105       7.420   6.210   0.000  1.00 87.00           O
ATOM     24  CB  SER A 105       8.160   3.060  -1.220  1.00 85.00           C
ATOM     25  OG  SER A 105       7.650   1.730  -1.250  1.00 84.00           O
END
"""


def _random_so3_batch(batch_size: int, device: torch.device, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mats = torch.randn(batch_size, 3, 3, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mats)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * torch.where(diag < 0.0, -torch.ones_like(diag), torch.ones_like(diag)).unsqueeze(-2)
    det = cofactor_det_3x3(q)
    flip = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
    q_proper = torch.cat((q[..., :, :2], q[..., :, 2:3] * flip.view(batch_size, 1, 1)), dim=-1)
    return q_proper.to(device)


def _assert_all_float32(module: torch.nn.Module) -> None:
    for name, p in module.named_parameters():
        assert p.dtype == torch.float32, f"Parameter {name} has non-float32 dtype {p.dtype}"
    for name, b in module.named_buffers():
        if b.is_floating_point():
            assert b.dtype == torch.float32, f"Buffer {name} has non-float32 dtype {b.dtype}"


def _make_synthetic_complex_batch(
    batch_size: int = 2,
    n_res: int = 12,
    atoms_per_res: int = 4,
    n_lig: int = 8,
    device: torch.device | None = None,
    seed: int = 101,
) -> ComplexBatch:
    dev = torch.device("cpu") if device is None else device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_poc = n_res * atoms_per_res
    n_tot = n_poc + n_lig
    l_tot = n_res + n_lig

    coords = torch.randn(batch_size, n_tot, 3, generator=gen, dtype=torch.float32) * 3.0
    atom_feats = torch.zeros(batch_size, n_tot, 52, dtype=torch.float32)
    atom_feats[:, :n_poc, 0] = 1.0  # Carbon element
    atom_feats[:, :n_poc, 10] = 1.0  # ALA residue
    atom_feats[:, :n_poc, 42] = 0.0
    atom_feats[:, :n_poc:atoms_per_res, 42] = 1.0  # Backbone flag on first atom per res
    atom_feats[:, n_poc:, 0] = 1.0
    atom_feats[:, n_poc:, 30] = 1.0  # OTHER residue slot for ligand
    atom_feats[:, :, 45] = 0.85  # Bondi vdW / 2.0 (1.70 A)
    atom_feats[:, :, 46] = torch.randn(batch_size, n_tot, generator=gen) * 0.2

    a2t = (
        torch.cat(
            [
                torch.arange(n_res, dtype=torch.long).repeat_interleave(atoms_per_res),
                torch.arange(n_res, l_tot, dtype=torch.long),
            ]
        )
        .unsqueeze(0)
        .expand(batch_size, n_tot)
        .clone()
    )

    atom_mask = torch.ones(batch_size, n_tot, dtype=torch.bool)
    token_mask = torch.ones(batch_size, l_tot, dtype=torch.bool)
    chain_id = (
        torch.cat([torch.zeros(n_res, dtype=torch.long), torch.ones(n_lig, dtype=torch.long)])
        .unsqueeze(0)
        .expand(batch_size, l_tot)
        .clone()
    )
    token_type = (
        torch.cat(
            [
                torch.arange(n_res, dtype=torch.long) % 20,
                torch.full((n_lig,), 21, dtype=torch.long),
            ]
        )
        .unsqueeze(0)
        .expand(batch_size, l_tot)
        .clone()
    )
    res_index = (
        torch.cat([torch.arange(n_res, dtype=torch.long), torch.arange(n_lig, dtype=torch.long)])
        .unsqueeze(0)
        .expand(batch_size, l_tot)
        .clone()
    )

    is_lig = torch.zeros(batch_size, n_tot, dtype=torch.bool)
    is_lig[:, n_poc:] = True
    is_bb = atom_feats[:, :, 42] > 0.5
    flex_mask = ~is_bb

    rep_idx = (
        torch.cat(
            [
                torch.arange(n_res, dtype=torch.long) * atoms_per_res,
                torch.arange(n_poc, n_tot, dtype=torch.long),
            ]
        )
        .unsqueeze(0)
        .expand(batch_size, l_tot)
        .clone()
    )

    # Add simple 1-2 bonds and one tetrahedral chiral quad inside ligand
    b12 = (
        torch.stack([torch.arange(n_poc, n_tot - 1), torch.arange(n_poc + 1, n_tot)], dim=-1)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .clone()
    )
    b12_ref = torch.full((batch_size, n_lig - 1), 1.5, dtype=torch.float32)
    b12_mask = torch.ones((batch_size, n_lig - 1), dtype=torch.bool)
    chiral_q = (
        torch.tensor([[[n_poc, n_poc + 1, n_poc + 2, n_poc + 3]]], dtype=torch.long)
        .expand(batch_size, 1, 4)
        .clone()
    )
    chiral_m = torch.ones((batch_size, 1), dtype=torch.bool)

    return ComplexBatch(
        coords=coords,
        atom_feats=atom_feats,
        atom_to_token=a2t,
        atom_mask=atom_mask,
        token_mask=token_mask,
        chain_id=chain_id,
        token_type=token_type,
        res_index=res_index,
        flex_mask=flex_mask,
        is_ligand_atom=is_lig,
        is_backbone_atom=is_bb,
        token_rep_atom_idx=rep_idx,
        bond_12_idx=b12,
        bond_12_ref=b12_ref,
        bond_12_mask=b12_mask,
        chiral_quads=chiral_q,
        chiral_mask=chiral_m,
        pkd_true=torch.full((batch_size,), 7.2, dtype=torch.float32),
        is_binder_true=torch.ones((batch_size,), dtype=torch.float32),
        centroid_offset=torch.zeros((batch_size, 3), dtype=torch.float32),
    ).to(dev)


# ---------------------------------------------------------------------------
# 1. KabschWeightedRigidAlign Tests (CPU + MPS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_kabsch_exact_so3_recovery_and_translation(device: torch.device) -> None:
    aligner = KabschWeightedRigidAlign().to(device)
    _assert_all_float32(aligner)

    gen = torch.Generator(device="cpu").manual_seed(11)
    x_src = torch.randn(4, 32, 3, generator=gen, dtype=torch.float32).to(device) * 4.0
    rot_true = _random_so3_batch(4, device, seed=19)
    trans_true = torch.tensor(
        [[100.0, -85.0, 42.0], [-95.0, 77.0, -100.0], [50.0, 50.0, 50.0], [-12.0, 88.0, 3.0]],
        dtype=torch.float32,
        device=device,
    )
    x_ref = torch.matmul(x_src, rot_true.transpose(-1, -2)) + trans_true.unsqueeze(1)
    weights = torch.ones(4, 32, device=device, dtype=torch.float32)
    weights[:, -6:] = 0.0  # Padded slots

    # Corrupt padded slots heavily to confirm zero leakage in Kabsch fit.
    x_src_corrupt = x_src.clone()
    x_src_corrupt[:, -6:] = 1e4
    x_aligned, rot_est, t_est = aligner(x_src_corrupt, x_ref, weights, detach_transform=True)

    # Check exact recovery on valid atoms
    diff = x_aligned[:, :-6] - x_ref[:, :-6]
    rmsd = torch.sqrt((diff * diff).sum(dim=-1).mean(dim=-1))
    assert float(rmsd.max().item()) < 1e-4, f"Expected RMSD < 1e-4, got {rmsd}"

    # Verify proper SO(3): det(R) == +1 and R^T R == I
    det_r = cofactor_det_3x3(rot_est)
    assert torch.allclose(det_r, torch.ones_like(det_r), atol=1e-5)
    eye = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand_as(rot_est)
    assert torch.allclose(torch.matmul(rot_est.transpose(-1, -2), rot_est), eye, atol=1e-5)
    assert torch.allclose(rot_est, rot_true, atol=1e-4)
    assert torch.allclose(t_est, trans_true, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_kabsch_strictly_preserves_det_plus_one_on_enantiomers_and_degeneracies(
    device: torch.device,
) -> None:
    aligner = KabschWeightedRigidAlign().to(device)
    gen = torch.Generator(device="cpu").manual_seed(23)

    # Asymmetric chiral 3D cloud vs its mirror enantiomer (improper reflection det = -1).
    x_chiral = torch.randn(3, 20, 3, generator=gen, dtype=torch.float32).to(device) * 3.0
    x_mirror = x_chiral.clone()
    x_mirror[..., 0] = -x_mirror[..., 0]
    w = torch.ones(3, 20, device=device, dtype=torch.float32)

    x_aligned, rot, _ = aligner(x_chiral, x_mirror, w, detach_transform=True)
    det_r = cofactor_det_3x3(rot)
    assert torch.allclose(det_r, torch.ones_like(det_r), atol=1e-5), (
        f"Kabsch reflected an enantiomer! det(R) = {det_r}"
    )
    # Mirror enantiomer CANNOT achieve zero RMSD under any proper SO(3) rotation.
    rmsd_mirror = torch.sqrt(((x_aligned - x_mirror) ** 2).sum(dim=-1).mean(dim=-1))
    assert float(rmsd_mirror.min().item()) > 0.5

    # Collinear, coincident, single-atom, and all-zero weight edge cases with gradients.
    collinear = torch.zeros(4, 6, 3, device=device, dtype=torch.float32)
    collinear[0, :, 0] = torch.arange(6, device=device, dtype=torch.float32)  # Collinear line
    collinear[1] = 2.5  # All coincident
    collinear[2, 0] = torch.tensor([1.0, 2.0, 3.0], device=device)  # Single atom
    collinear[3] = torch.randn(6, 3, generator=gen).to(device)  # Zero-weight batch item
    collinear.requires_grad_(True)

    ref_deg = collinear.detach().clone() + 1.0
    w_deg = torch.ones(4, 6, device=device, dtype=torch.float32)
    w_deg[2, 1:] = 0.0
    w_deg[3, :] = 0.0

    for detach_flag in (True, False):
        if collinear.grad is not None:
            collinear.grad.zero_()
        out_deg, r_deg, t_deg = aligner(collinear, ref_deg, w_deg, detach_transform=detach_flag)
        assert torch.isfinite(out_deg).all()
        assert torch.isfinite(r_deg).all()
        assert torch.isfinite(t_deg).all()
        assert torch.allclose(cofactor_det_3x3(r_deg), torch.ones(4, device=device), atol=1e-4)
        out_deg.sum().backward()
        assert collinear.grad is not None and torch.isfinite(collinear.grad).all()


# ---------------------------------------------------------------------------
# 2. Pairformer & SimpleFold Trunk Tests (CPU + MPS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["outgoing", "incoming"])
def test_triangle_mult_dispatch_and_naive_reference_parity(
    device: torch.device,
    mode: str,
) -> None:
    torch.manual_seed(42)
    for c_z, expect_matmul in ((64, False), (128, True)):
        mod = TriangleMultiplicativeUpdate(c_z=c_z, mode=mode).to(device)  # type: ignore[arg-type]
        _assert_all_float32(mod)
        assert mod.use_matmul_dispatch is expect_matmul

        z = torch.randn(2, 12, 12, c_z, device=device, dtype=torch.float32)
        mask = torch.ones(2, 12, dtype=torch.bool, device=device)
        mask[1, 8:] = False

        out_auto = mod(z, mask, force_method="auto")
        out_ein = mod(z, mask, force_method="einsum")
        out_mm = mod(z, mask, force_method="matmul")
        assert float((out_ein - out_mm).abs().max().item()) < 1e-5
        assert float((out_auto - out_ein).abs().max().item()) < 1e-5
        assert float(out_auto.abs().sum().item()) > 0.0

        # Exact naive textbook AF3 Alg. 12 / 13 reference
        pm = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1)
        z_clean = torch.where(pm, z, torch.zeros_like(z))
        zn = mod.norm_in(z_clean)
        a = torch.where(pm, torch.sigmoid(mod.gate_a(zn)) * mod.proj_a(zn), torch.zeros_like(zn))
        b = torch.where(pm, torch.sigmoid(mod.gate_b(zn)) * mod.proj_b(zn), torch.zeros_like(zn))
        if mode == "outgoing":
            o_ref = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            o_ref = torch.einsum("bkic,bkjc->bijc", a, b)
        ref = torch.sigmoid(mod.gate_out(zn)) * mod.proj_out(mod.norm_out(o_ref))
        ref = torch.where(pm, ref, torch.zeros_like(ref))
        assert float((out_auto - ref).abs().max().item()) < 1e-5

        # Zero dummy leakage under 1e4 corruption
        z_corrupt = z.clone()
        z_corrupt[1, 8:, :, :] = 1e4
        z_corrupt[1, :, 8:, :] = -1e4
        out_corrupt = mod(z_corrupt, mask)
        assert float((out_corrupt[1, :8, :8] - out_auto[1, :8, :8]).abs().max().item()) < 1e-5
        assert float(out_corrupt[1, 8:, :, :].abs().max().item()) == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_triangle_attention_head_dim_enforcement_and_chunk_parity(device: torch.device) -> None:
    with pytest.raises(ValueError, match="head_dim >= 32"):
        TriangleAttention(c_z=64, head_dim=16)
    with pytest.raises(ValueError, match="head_dim"):
        PairBiasedAttention(c_s=64, c_z=64, num_heads=4)

    torch.manual_seed(7)
    for mode in ("starting", "ending"):
        tri = TriangleAttention(c_z=64, head_dim=32, mode=mode).to(device)
        _assert_all_float32(tri)
        z = torch.randn(2, 16, 16, 64, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(2, 16, dtype=torch.bool, device=device)
        mask[1, 10:] = False

        out_full = tri(z, mask, chunk_size=16)
        out_chunk = tri(z, mask, chunk_size=5)
        assert float(out_full.abs().sum().item()) > 0.0
        assert float((out_full - out_chunk).abs().max().item()) < 1e-5

        # All-empty row backward stability
        mask_empty = mask.clone()
        mask_empty[0, :] = False
        out_empty = tri(z, mask_empty, chunk_size=4)
        assert torch.isfinite(out_empty).all()
        out_empty.sum().backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()
        z.grad.zero_()


@pytest.mark.parametrize("device", DEVICES)
def test_pairformer_stack_and_simplefold_trunk_forward_backward(device: torch.device) -> None:
    torch.manual_seed(13)
    s = torch.randn(2, 16, 96, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(2, 16, 16, 64, device=device, dtype=torch.float32, requires_grad=True)
    coords = torch.randn(2, 16, 3, device=device, dtype=torch.float32) * 4.0
    mask = torch.ones(2, 16, dtype=torch.bool, device=device)
    mask[1, 12:] = False

    pf = PairformerStack(num_blocks=2, c_s=96, c_z=64, head_dim=32, outer_product_mean=True).to(
        device
    )
    _assert_all_float32(pf)
    s_pf, z_pf = pf(s, z, mask)
    assert s_pf.shape == (2, 16, 96) and z_pf.shape == (2, 16, 16, 64)
    assert torch.isfinite(s_pf).all() and torch.isfinite(z_pf).all()
    (s_pf.pow(2).mean() + z_pf.pow(2).mean()).backward()
    for name, p in pf.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"

    for mode in ("simplefold", "hybrid"):
        sf = SimpleFoldTrunk(
            c_s=96,
            c_z=64,
            num_layers=3,
            head_dim=32,
            trunk_mode=mode,
            use_rope=True,
            use_coulomb_bias=True,
        ).to(device)
        _assert_all_float32(sf)
        cond = torch.randn(2, 96, device=device, dtype=torch.float32)
        s2 = s.detach().clone().requires_grad_(True)
        z2 = z.detach().clone().requires_grad_(True)
        s_sf, z_sf = sf(s2, z2, mask, cond=cond, coords=coords)
        assert s_sf.shape == (2, 16, 96) and z_sf.shape == (2, 16, 16, 64)
        assert torch.isfinite(s_sf).all() and torch.isfinite(z_sf).all()
        assert float(s_sf[1, 12:].abs().max().item()) == 0.0
        (s_sf.pow(2).mean() + z_sf.pow(2).mean()).backward()
        for name, p in sf.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (
                f"Bad grad in {mode} on {name}"
            )


# ---------------------------------------------------------------------------
# 3. AllAtomDiffusionModule (EDM & Flow-Matching) & CoFoldingLoss Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("diffusion_mode", ["edm", "flow_matching"])
def test_diffusion_seeded_sampling_flex_modes_and_gradients(
    device: torch.device,
    diffusion_mode: str,
) -> None:
    torch.manual_seed(31)
    batch = _make_synthetic_complex_batch(batch_size=2, n_res=8, n_lig=6, device=device)
    s_trunk = torch.randn(2, 14, 96, device=device, dtype=torch.float32)
    z_trunk = torch.randn(2, 14, 14, 64, device=device, dtype=torch.float32)

    diff_mod = AllAtomDiffusionModule(
        c_s=96,
        c_z=64,
        c_atom=64,
        c_token=96,
        head_dim=32,
        num_diff_layers=2,
        num_atom_enc_layers=1,
        num_atom_dec_layers=1,
        diffusion_mode=diffusion_mode,  # type: ignore[arg-type]
    ).to(device)
    _assert_all_float32(diff_mod)

    # Deterministic seeded sampling check
    diff_mod.eval()
    with torch.no_grad():
        c1 = diff_mod.sample(
            batch.coords,
            s_trunk,
            z_trunk,
            batch.atom_feats,
            batch.atom_to_token,
            batch.atom_mask,
            token_mask=batch.token_mask,
            flex_mask=batch.flex_mask,
            flex_mode="flexible_pocket_and_ligand",
            num_steps=4,
            seed=42,
        )
        c2 = diff_mod.sample(
            batch.coords,
            s_trunk,
            z_trunk,
            batch.atom_feats,
            batch.atom_to_token,
            batch.atom_mask,
            token_mask=batch.token_mask,
            flex_mask=batch.flex_mask,
            flex_mode="flexible_pocket_and_ligand",
            num_steps=4,
            seed=42,
        )
        c3 = diff_mod.sample(
            batch.coords,
            s_trunk,
            z_trunk,
            batch.atom_feats,
            batch.atom_to_token,
            batch.atom_mask,
            token_mask=batch.token_mask,
            flex_mask=batch.flex_mask,
            flex_mode="flexible_pocket_and_ligand",
            num_steps=4,
            seed=99,
        )
    assert torch.isfinite(c1).all()
    assert float((c1 - c2).abs().max().item()) < 1e-5
    assert float((c1 - c3).abs().max().item()) > 1e-3

    # Anchored backbone atoms remain strictly fixed in flexible_pocket_and_ligand mode
    assert batch.flex_mask is not None
    anchored = ~batch.flex_mask
    assert float((c1[anchored] - batch.coords[anchored]).abs().max().item()) == 0.0

    # Training forward + CoFoldingLoss backward across all params
    diff_mod.train()
    loss_fn = CoFoldingLoss().to(device)
    x0_pred, x0_tgt, _nl, lw = diff_mod.forward_train(
        batch.coords,
        s_trunk,
        z_trunk,
        batch.atom_feats,
        batch.atom_to_token,
        batch.atom_mask,
        token_mask=batch.token_mask,
        flex_mask=batch.flex_mask,
        flex_mode="de_novo_all_atom",
        seed=42,
    )
    loss_out = loss_fn(
        x_pred=x0_pred,
        x_true=x0_tgt,
        atom_mask=batch.atom_mask,
        loss_weight=lw,
        align=True,
        is_ligand=batch.is_ligand_atom,
        atom_feats=batch.atom_feats,
        bond_12_idx=batch.bond_12_idx,
        bond_12_ref=batch.bond_12_ref,
        bond_12_mask=batch.bond_12_mask,
        chiral_quads=batch.chiral_quads,
        chiral_mask=batch.chiral_mask,
        backbone_mask=batch.is_backbone_atom,
        x_input=batch.coords,
    )
    assert torch.isfinite(loss_out.total)
    loss_out.total.backward()
    for name, p in diff_mod.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            f"Non-finite or missing grad in {diffusion_mode} on {name}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_cofolding_loss_physics_smooth_lddt_chiral_and_vdw_clash(device: torch.device) -> None:
    loss_fn = CoFoldingLoss().to(device)

    # Regular tetrahedron around origin -> known positive signed volume
    tet = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, 1.5],
                [6.0, 6.0, 6.0],  # Well-separated non-bonded atom
                [10.0, 10.0, 10.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones(1, 6, dtype=torch.bool, device=device)
    quads = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.long, device=device)
    vdw = torch.full((1, 6), 1.70, dtype=torch.float32, device=device)
    # Only pairs involving atoms 4/5 are non-bonded here
    nb_mask = torch.zeros(1, 6, 6, dtype=torch.bool, device=device)
    nb_mask[0, 4, 5] = True
    nb_mask[0, 5, 4] = True

    # Exact ground truth: smooth LDDT == 0, chiral == 0, clash == 0
    out_gt = loss_fn(
        tet,
        tet,
        mask,
        align=False,
        vdw_radii=vdw,
        chiral_quads=quads,
        nonbonded_mask=nb_mask,
    )
    assert float(out_gt.smooth_lddt.item()) < 1e-5
    assert float(out_gt.chiral.item()) == 0.0
    assert float(out_gt.clash.item()) == 0.0

    # Chiral mirror inversion (R) -> (S): flips signed volume sign and spikes loss_chiral!
    vol_r = compute_signed_tetrahedral_volume(tet, quads)
    tet_mirror = tet.clone()
    tet_mirror[0, 3, 2] = -tet_mirror[0, 3, 2]  # Invert z-axis of atom 3
    vol_s = compute_signed_tetrahedral_volume(tet_mirror, quads)
    assert float((vol_r * vol_s).item()) < 0.0

    out_inv = loss_fn(
        tet_mirror,
        tet,
        mask,
        align=False,
        vdw_radii=vdw,
        chiral_quads=quads,
        nonbonded_mask=nb_mask,
    )
    assert float(out_inv.chiral.item()) > 5.0
    assert 0.0 < float(out_inv.smooth_lddt.item()) <= 1.0

    # Steric clash test: move atom 5 to 1.0 A from atom 4 (< 0.82 * 3.40 = 2.788 A)
    tet_clash = tet.clone()
    tet_clash[0, 5] = tet_clash[0, 4] + torch.tensor([1.0, 0.0, 0.0], device=device)
    out_clash = loss_fn(
        tet_clash,
        tet,
        mask,
        align=False,
        vdw_radii=vdw,
        chiral_quads=quads,
        nonbonded_mask=nb_mask,
    )
    assert float(out_clash.clash.item()) > 2.0

    # Small-volume chiral regression (|V_true| = 0.216 < 0.5): exact GT must yield chiral == 0.0
    small_tet = tet.clone()
    small_tet[0, :4] = small_tet[0, :4] * 0.4  # edge = 0.6 -> V_true = 0.216
    out_small_gt = loss_fn(
        small_tet,
        small_tet,
        mask,
        align=False,
        vdw_radii=vdw,
        chiral_quads=quads,
        nonbonded_mask=nb_mask,
    )
    assert float(out_small_gt.chiral.item()) == 0.0
    small_mirror = small_tet.clone()
    small_mirror[0, 3, 2] = -small_mirror[0, 3, 2]
    out_small_inv = loss_fn(
        small_mirror,
        small_tet,
        mask,
        align=False,
        vdw_radii=vdw,
        chiral_quads=quads,
        nonbonded_mask=nb_mask,
    )
    assert float(out_small_inv.chiral.item()) > 0.05


def test_real_rdkit_enantiomers_chiral_volume_discrimination() -> None:
    # Real chiral drug molecule: (R)- vs (S)-ibuprofen
    vols: list[float] = []
    chem_mod = Chem
    for smi in ("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1", "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = 42
        assert AllChem.EmbedMolecule(mol, params) == 0  # type: ignore[attr-defined]
        mol_noh = Chem.RemoveHs(mol)
        chiral_centers = chem_mod.FindMolChiralCenters(mol_noh, includeUnassigned=False)  # type: ignore[no-untyped-call]
        assert len(chiral_centers) == 1
        c_idx = int(chiral_centers[0][0])
        nbrs = sorted(int(n.GetIdx()) for n in mol_noh.GetAtomWithIdx(c_idx).GetNeighbors())
        conf = mol_noh.GetConformer()
        pts = torch.tensor(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol_noh.GetNumAtoms())
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        quad = torch.tensor([[[c_idx, nbrs[0], nbrs[1], nbrs[2]]]], dtype=torch.long)
        vols.append(float(compute_signed_tetrahedral_volume(pts, quad).item()))
    assert vols[0] * vols[1] < 0.0, f"Expected opposite chiral signs for (R)/(S), got {vols}"


# ---------------------------------------------------------------------------
# 4. ConfidenceHead & AffinityHead Tests (CPU + MPS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_confidence_head_ranges_symmetry_and_strict_cross_chain_iptm(device: torch.device) -> None:
    torch.manual_seed(55)
    batch = _make_synthetic_complex_batch(batch_size=2, n_res=10, n_lig=6, device=device)
    s_trunk = torch.randn(2, 16, 96, device=device, dtype=torch.float32)
    z_trunk = torch.randn(2, 16, 16, 64, device=device, dtype=torch.float32)
    x_rep = torch.randn(2, 16, 3, device=device, dtype=torch.float32) * 5.0

    head = ConfidenceHead(
        c_s=96,
        c_z=64,
        num_blocks=1,
        head_dim=32,
        enable_affinity_head=True,
    ).to(device)
    _assert_all_float32(head)

    out = head(
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        x_rep=x_rep,
        token_mask=batch.token_mask,
        chain_id=batch.chain_id,
        atom_to_token=batch.atom_to_token,
        atom_mask=batch.atom_mask,
        is_ligand_atom=batch.is_ligand_atom,
        detach_trunk=True,
    )
    assert out.plddt_logits.shape == (2, batch.coords.shape[1], 50)
    assert out.pae_logits.shape == (2, 16, 16, 64)
    assert out.pde_logits.shape == (2, 16, 16, 64)

    # Valid mathematical ranges
    assert float(out.plddt.min().item()) >= 0.0 and float(out.plddt.max().item()) <= 100.0
    assert (
        float(out.ligand_plddt.min().item()) >= 0.0
        and float(out.ligand_plddt.max().item()) <= 100.0
    )
    assert float(out.ptm.min().item()) >= 0.0 and float(out.ptm.max().item()) <= 1.0
    assert float(out.iptm.min().item()) >= 0.0 and float(out.iptm.max().item()) <= 1.0
    assert (
        float(out.ranking_score.min().item()) >= 0.0
        and float(out.ranking_score.max().item()) <= 1.0
    )
    assert float(out.pae.min().item()) >= 0.0
    assert float(out.pde.min().item()) >= 0.0
    assert torch.allclose(out.ranking_score, 0.8 * out.iptm + 0.2 * out.ptm, atol=1e-6)

    # PDE strictly symmetric
    assert float((out.pde - out.pde.transpose(1, 2)).abs().max().item()) < 1e-6

    # AffinityHead outputs valid
    assert out.predicted_pkd is not None and torch.isfinite(out.predicted_pkd).all()
    assert out.binding_prob is not None
    assert (
        float(out.binding_prob.min().item()) >= 0.0 and float(out.binding_prob.max().item()) <= 1.0
    )
    assert out.binder_logit is not None and torch.isfinite(out.binder_logit).all()

    # Exact cross-chain ipTM isolation: mutating ONLY intra-chain PAE logits changes pTM
    # while leaving ipTM strictly invariant (0.0 diff)!
    pae_mod = out.pae_logits.detach().clone()
    same_chain = batch.chain_id.unsqueeze(2) == batch.chain_id.unsqueeze(1)
    pae_mod[same_chain] = pae_mod[same_chain] + 15.0 * torch.linspace(
        -1.0, 1.0, 64, device=device, dtype=torch.float32
    )
    _, ptm_orig, iptm_orig, _ = compute_tm_scores(out.pae_logits, batch.token_mask, batch.chain_id)
    _, ptm_mut, iptm_mut, _ = compute_tm_scores(pae_mod, batch.token_mask, batch.chain_id)
    assert float((iptm_mut - iptm_orig).abs().max().item()) == 0.0
    assert float((ptm_mut - ptm_orig).abs().max().item()) > 1e-2

    # ConfidenceLoss backward finite gradients on all ConfidenceHead + AffinityHead params
    conf_loss_fn = ConfidenceLoss().to(device)
    loss = conf_loss_fn(
        conf=out,
        x_pred_atoms=batch.coords,
        x_true_atoms=batch.coords + 0.5,
        atom_mask=batch.atom_mask,
        x_pred_rep=x_rep,
        x_true_rep=x_rep + 0.5,
        token_mask=batch.token_mask,
        pkd_true=batch.pkd_true,
        is_binder_true=batch.is_binder_true,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"


# ---------------------------------------------------------------------------
# 5. NanoFold3 End-to-End Presets, .cofold(), .fold_sequence() & MPS Benchmark
# ---------------------------------------------------------------------------


def test_nanofold3_presets_parameter_budgets_and_float32() -> None:
    nano = build_nanofold3("nano")
    small = build_nanofold3("small")
    base = build_nanofold3("base")
    for m in (nano, small, base):
        _assert_all_float32(m)

    assert 1_000_000 <= nano.num_parameters <= 2_200_000, f"nano={nano.num_parameters}"
    assert 4_500_000 <= small.num_parameters <= 8_500_000, f"small={small.num_parameters}"
    assert 18_000_000 <= base.num_parameters <= 32_000_000, f"base={base.num_parameters}"
    assert nano.num_parameters < small.num_parameters < base.num_parameters

    # SimpleFold trunk should be leaner than Pairformer trunk at base scale
    base_sf = build_nanofold3("base", trunk_mode="simplefold")
    assert base_sf.num_parameters < base.num_parameters


@pytest.mark.parametrize("device", DEVICES)
def test_nanofold3_nano_end_to_end_cofold_fold_sequence_and_audit(
    device: torch.device,
    tmp_path: Path,
) -> None:
    model = build_nanofold3("nano", trunk_mode="pairformer", diffusion_mode="edm", device=device)
    model.eval()

    # Use real DRD1 pocket if available locally, otherwise self-contained mini pocket PDB.
    real_pocket = Path("data/cns_neuro.nosync/structures/clean_pockets/9LLG_DRD1_A1EKL_pocket.pdb")
    pocket_input: str | Path = real_pocket if real_pocket.is_file() else _MINI_POCKET_PDB
    # Dopamine SMILES (clean drug-like ligand)
    ligand_smi = "NCCc1ccc(O)c(O)c1"

    res1 = model.cofold(
        pocket_input,
        ligand_smi,
        flex_mode="flexible_pocket_and_ligand",
        num_samples=1,
        num_steps=4,
        seed=42,
        run_audit=True,
    )
    res2 = model.cofold(
        pocket_input,
        ligand_smi,
        flex_mode="flexible_pocket_and_ligand",
        num_samples=1,
        num_steps=4,
        seed=42,
        run_audit=False,
    )
    assert res1.backend == "nanofold3"
    assert "ATOM  " in res1.pdb_text and "HETATM" in res1.pdb_text
    assert res1.ligand_mol_3d is not None and res1.ligand_mol_3d.GetNumConformers() == 1
    assert 0.0 <= res1.plddt_mean <= 100.0
    assert 0.0 <= res1.ligand_plddt <= 100.0
    assert 0.0 <= res1.ptm <= 1.0
    assert 0.0 <= res1.iptm <= 1.0
    assert 0.0 <= res1.ranking_score <= 1.0
    assert math.isfinite(res1.predicted_pkd)
    assert 0.0 <= res1.binding_prob <= 1.0
    assert res1.pae_matrix is not None and float(np.min(res1.pae_matrix)) >= 0.0
    assert res1.pose_audit is not None
    assert 0.0 <= float(res1.pose_audit.reality_score) <= 1.0
    # Determinism: exact same coordinates & PDB hash for same seed!
    assert res1.coords is not None and res2.coords is not None
    assert np.allclose(res1.coords, res2.coords, atol=1e-4)
    assert res1.provenance_sha256 == res2.provenance_sha256

    # Single-sequence mini-folding
    seq_res = model.fold_sequence("ACDEFGHIKLMNPQ", num_samples=1, num_steps=4, seed=42)
    assert "ATOM  " in seq_res.pdb_text
    assert seq_res.coords is not None and seq_res.coords.shape == (14 * 4, 3)
    assert np.all(np.isfinite(seq_res.coords))
    assert 0.0 <= seq_res.plddt_mean <= 100.0
    assert 0.0 <= seq_res.ptm <= 1.0

    # End-to-end training step + backward on device
    model.train()
    train_batch = _make_synthetic_complex_batch(batch_size=1, n_res=8, n_lig=6, device=device)
    t_out = model.forward_train(train_batch, seed=42)
    assert torch.isfinite(t_out.loss)
    torch.autograd.backward(t_out.loss)
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad on {name}"

    # Checkpoint save/load roundtrip
    ckpt = model.save_pretrained(tmp_path / "nanofold3_nano.pt")
    reloaded = NanoFold3.from_pretrained(ckpt, device=device)
    assert reloaded.num_parameters == model.num_parameters
    assert reloaded.config == model.config


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS not available")
def test_nanofold3_mps_latency_benchmark_l64_and_l128() -> None:
    """Measure synchronized Apple M3 Pro MPS inference latency at L=64 & L=128 (N_atoms=256)."""
    device = torch.device("mps")
    results: dict[str, float] = {}

    for trunk_mode in ("pairformer", "simplefold"):
        model = build_nanofold3(
            "nano",
            trunk_mode=trunk_mode,
            diffusion_mode="edm",
            device=device,
        )
        model.eval()
        for n_res, atoms_per_res, n_lig in ((48, 5, 16), (64, 3, 64)):  # L=64 & L=128, N=256
            batch = _make_synthetic_complex_batch(
                batch_size=1,
                n_res=n_res,
                atoms_per_res=atoms_per_res,
                n_lig=n_lig,
                device=device,
            )
            assert batch.coords.shape == (1, 256, 3)
            l_tok = int(batch.token_mask.shape[1])
            assert l_tok in (64, 128)

            # Warmup
            _ = model.sample(batch, num_steps=4, num_samples=1, seed=42)
            torch.mps.synchronize()

            t0 = time.perf_counter()
            out = model.sample(batch, num_steps=8, num_samples=1, seed=42)
            torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert torch.isfinite(out.coords).all()
            results[f"{trunk_mode}_L{l_tok}"] = elapsed_ms

    # Verify MPS execution stays comfortably responsive (< 5 seconds even under full test load)
    for label, ms in results.items():
        assert ms > 0.0 and ms < 5000.0, f"Unexpected MPS latency {label}: {ms:.2f} ms"


@pytest.mark.parametrize("device", DEVICES)
def test_zero_flexible_coordinate_leakage_in_trunk_and_denoiser(device: torch.device) -> None:
    """Prove shifting flexible ground-truth coords has exact 0.0 effect on trunk and r_ref."""
    batch_a = _make_synthetic_complex_batch(batch_size=2, n_res=8, n_lig=6, device=device, seed=77)
    assert batch_a.flex_mask is not None
    shifted_coords = batch_a.coords.clone()
    shifted_coords[batch_a.flex_mask] = shifted_coords[batch_a.flex_mask] + 25.0
    batch_b = ComplexBatch(
        coords=shifted_coords,
        atom_feats=batch_a.atom_feats,
        atom_to_token=batch_a.atom_to_token,
        atom_mask=batch_a.atom_mask,
        token_mask=batch_a.token_mask,
        chain_id=batch_a.chain_id,
        token_type=batch_a.token_type,
        res_index=batch_a.res_index,
        flex_mask=batch_a.flex_mask,
        is_ligand_atom=batch_a.is_ligand_atom,
        is_backbone_atom=batch_a.is_backbone_atom,
        token_rep_atom_idx=batch_a.token_rep_atom_idx,
        covalent_adj=batch_a.covalent_adj,
    )

    for trunk_mode in ("pairformer", "simplefold", "hybrid"):
        model = build_nanofold3("nano", trunk_mode=trunk_mode, device=device).eval()
        with torch.no_grad():
            s_a, z_a = model.forward_trunk(batch_a)
            s_b, z_b = model.forward_trunk(batch_b)
            assert float((s_a - s_b).abs().max().item()) == 0.0, f"Leak in {trunk_mode} s_trunk"
            assert float((z_a - z_b).abs().max().item()) == 0.0, f"Leak in {trunk_mode} z_trunk"

            # Fixed r_noisy + shifted r_ref on flexible atoms must produce exact 0.0 diff!
            r_noisy = torch.zeros_like(batch_a.coords)
            sigma = torch.full((2,), 4.0, device=device, dtype=torch.float32)
            out_a = model.diffusion.denoise(
                r_noisy=r_noisy,
                noise_level=sigma,
                s_trunk=s_a,
                z_trunk=z_a,
                atom_feats=batch_a.atom_feats,
                atom_to_token=batch_a.atom_to_token,
                atom_mask=batch_a.atom_mask,
                token_mask=batch_a.token_mask,
                flex_mask=batch_a.flex_mask,
                r_ref=batch_a.coords,
            )
            out_b = model.diffusion.denoise(
                r_noisy=r_noisy,
                noise_level=sigma,
                s_trunk=s_a,
                z_trunk=z_a,
                atom_feats=batch_a.atom_feats,
                atom_to_token=batch_a.atom_to_token,
                atom_mask=batch_a.atom_mask,
                token_mask=batch_a.token_mask,
                flex_mask=batch_a.flex_mask,
                r_ref=batch_b.coords,
            )
            assert float((out_a - out_b).abs().max().item()) == 0.0, (
                f"Leak via r_ref in {trunk_mode} denoiser"
            )

            # De novo mode: shifting ALL coordinates must yield exact 0.0 change in trunk
            all_flex = batch_a.atom_mask
            s_dn_a, z_dn_a = model.forward_trunk(batch_a, flex_mask=all_flex)
            s_dn_b, z_dn_b = model.forward_trunk(batch_b, flex_mask=all_flex)
            assert float((s_dn_a - s_dn_b).abs().max().item()) == 0.0
            assert float((z_dn_a - z_dn_b).abs().max().item()) == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_nanofold3_sample_batched_best_idx_confidence_gather(device: torch.device) -> None:
    """Verify B=2, num_samples=3 gathers per-batch best-ranked confidence tensors cleanly."""
    model = build_nanofold3("nano", trunk_mode="pairformer", device=device).eval()
    batch = _make_synthetic_complex_batch(batch_size=2, n_res=6, n_lig=4, device=device, seed=88)

    out = model.sample(batch, num_steps=3, num_samples=3, seed=42)
    assert out.coords.shape == (3, 2, batch.coords.shape[1], 3)
    assert out.ranking_scores.shape == (3, 2)
    assert out.best_sample_idx.shape == (2,)

    expected_best_scores = out.ranking_scores.max(dim=0).values
    assert torch.allclose(out.confidence.ranking_score, expected_best_scores, atol=1e-6)
    assert out.confidence.plddt.shape == (2, batch.coords.shape[1])
    assert out.confidence.pae.shape == (2, 10, 10)
    assert out.confidence.predicted_pkd is not None and out.confidence.predicted_pkd.shape == (2,)

    # Re-evaluate the chosen sample independently for each batch item and confirm exact match!
    s_trunk, z_trunk = model.forward_trunk(batch, flex_mask=batch.flex_mask)
    for b_idx in range(2):
        best_s = int(out.best_sample_idx[b_idx].item())
        rep_b = model._extract_token_rep_coords(out.coords[best_s], batch)
        conf_ref = model.confidence_head(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            x_rep=rep_b,
            token_mask=batch.token_mask,
            chain_id=batch.chain_id,
            atom_to_token=batch.atom_to_token,
            atom_mask=batch.atom_mask,
            is_ligand_atom=batch.is_ligand_atom,
            detach_trunk=True,
        )
        assert torch.allclose(out.confidence.plddt[b_idx], conf_ref.plddt[b_idx], atol=1e-6)
        assert torch.allclose(out.confidence.pae[b_idx], conf_ref.pae[b_idx], atol=1e-6)
        assert torch.allclose(out.confidence.ptm[b_idx], conf_ref.ptm[b_idx], atol=1e-6)
        assert torch.allclose(out.confidence.iptm[b_idx], conf_ref.iptm[b_idx], atol=1e-6)
