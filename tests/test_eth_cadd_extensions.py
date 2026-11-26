"""Offline CI-safe tests for GenEuclidean's ETH-CADD-aligned frontier modules.

Covers:
1. ``ring_strain``: Cremer-Pople puckering coordinates (chair vs flattened/boat distortion),
   MMFF94s saturated-ring strain auditing, pluggable backend registry, and ``RingStrainFilter``
   / ``RingStrainScorer`` protocol compliance.
2. ``pocket_steering``: Multi-conformation ensemble & contrastive off-target CFG logit
   arithmetic, KV-cached ensemble sampling via ``MoleculeDesigner.design_ensemble``, and exact
   ``PMI_3D`` / ``DeltaPMI_3D`` scoring via ``MoleculeDesigner.score_pmi``.
3. ``scaffold_hop``: 5-axis natural-product scaffold-hopping audit (Ibogaine -> SERT hops,
   SA simplification, DCU quarantine veto).
4. ``mtl_intervention``: Pairwise cosine gradient conflict diagnostics, ``scalarized`` /
   ``pcgrad`` / ``aim_policy`` conflict mitigation on opposing synthetic gradients and on
   a real shared ``PaiNN-SE3`` multi-task backward pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

import geneuclidean as ge
from geneuclidean.cns.scaffold_hop import IBOGAINE_SMILES, PAROXETINE_SMILES
from geneuclidean.config import (
    CVAEConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
    seed_everything,
)
from geneuclidean.evaluation.ring_strain import (
    RING_STRAIN_BACKEND_REGISTRY,
    RingStrainFilter,
    RingStrainScorer,
    analyze_ring_strain,
    compute_cremer_pople,
)
from geneuclidean.models.geneuclidean import GenEuclideanModel
from geneuclidean.protocols import MoleculeFilterProtocol, MoleculeScorerProtocol
from geneuclidean.sampling.pocket_steering import combine_contrastive_ensemble_logits
from geneuclidean.training.mtl_intervention import (
    GradientConflictMediator,
    compute_gradient_conflict_matrix,
    mediate_pcgrad,
)

DRD1_1 = "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"
DCU = "O=C(NC1CCCCC1)NC1CCCCC1"


def _embed_3d_minimized(smiles: str, seed: int = 42, n_confs: int = 1) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = seed
    if n_confs <= 1:
        assert AllChem.EmbedMolecule(mol, params) == 0  # type: ignore[attr-defined]
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=300)  # type: ignore[attr-defined]
        return Chem.RemoveHs(mol)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))  # type: ignore[attr-defined]
    assert len(cids) >= 1
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
    best_cid, best_e = int(cids[0]), float("inf")
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))  # type: ignore[attr-defined]
        ff.Minimize(maxIts=300)
        e = float(ff.CalcEnergy())
        if e < best_e:
            best_e, best_cid = e, int(cid)
    out = Chem.Mol(mol)
    best_conf = Chem.Conformer(mol.GetConformer(best_cid))
    out.RemoveAllConformers()
    out.AddConformer(best_conf, assignId=True)
    return Chem.RemoveHs(out)


def _write_two_pockets(tmp_path: Path) -> tuple[Path, Path]:
    """Create two distinct minimal 3D pockets (on-target vs off-target) for steering tests."""
    p1 = tmp_path / "9LLG_DRD1_a_pocket.pdb"
    p1.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ASP R 103       2.800   0.000   0.000  1.00 20.00           C",
                "ATOM      2  OD1 ASP R 103       1.200   0.800   1.000  1.00 20.00           O",
                "ATOM      3  CA  PHE R 288      -3.200   1.200   0.500  1.00 20.00           C",
                "ATOM      4  OG  SER R 198       0.100  -1.400   0.800  1.00 20.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p2 = tmp_path / "6CM4_DRD2_b_pocket.pdb"
    p2.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ASP A 114       3.600   0.500  -0.400  1.00 20.00           C",
                "ATOM      2  OD1 ASP A 114       2.100   1.100   0.600  1.00 20.00           O",
                "ATOM      3  CA  TRP A 386      -2.600  -1.500   1.200  1.00 20.00           C",
                "ATOM      4  CE2 TRP A 386      -1.100  -2.200   1.900  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p1, p2


# ---------------------------------------------------------------------------
# 1. Saturated-Ring Conformational Strain & Cremer-Pople Puckering
# ---------------------------------------------------------------------------


def test_cremer_pople_distinguishes_chair_from_twist_boat_and_flattened_ring() -> None:
    """Chair cyclohexane gives theta ~ 0 deg, ETKDGv3 twist-boat gives ~90 deg & >5 kcal/mol."""
    # seed=1 lands in global-minimum chair (-3.56 kcal/mol); seed=42 lands in a real twist-boat!
    mol_chair = _embed_3d_minimized("C1CCCCC1", seed=1)
    ring_atoms = mol_chair.GetRingInfo().AtomRings()[0]
    conf = mol_chair.GetConformer()
    xyz = np.asarray(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in ring_atoms
        ],
        dtype=np.float64,
    )
    q_val, theta_val, phi_val, cls_label = compute_cremer_pople(xyz)
    assert q_val > 0.45
    assert theta_val is not None and min(theta_val, 180.0 - theta_val) < 5.0
    assert phi_val is not None
    assert cls_label == "chair"

    # Real ETKDGv3(seed=42) twist-boat local minimum (+5.9 kcal/mol above chair!)
    mol_tb = _embed_3d_minimized("C1CCCCC1", seed=42)
    prof_tb = analyze_ring_strain(mol_tb, strain_threshold_kcal=3.0, n_ref_confs=16, seed=1)
    assert prof_tb.rings[0].pucker_class == "twist_boat"
    assert prof_tb.max_ring_strain_kcal > 4.5
    assert prof_tb.passed is False

    # Artificially flatten all z-displacements -> planar high-strain distortion
    xyz_flat = xyz.copy()
    xyz_flat[:, 2] = 0.0
    q_flat, _, _, cls_flat = compute_cremer_pople(xyz_flat)
    assert q_flat < 0.08
    assert cls_flat == "planar"


def test_ring_strain_detects_distorted_conformer_and_integrates_with_sdk() -> None:
    """Chair piperazide passes (<1.5 kcal/mol), whereas flattened ring triggers high strain."""
    mol_good = _embed_3d_minimized(DRD1_1, seed=1, n_confs=8)
    prof_good = analyze_ring_strain(mol_good, strain_threshold_kcal=3.0, n_ref_confs=16, seed=1)
    assert prof_good.n_saturated_ring_systems == 1
    assert prof_good.max_ring_strain_kcal < 1.5
    assert prof_good.passed is True
    assert prof_good.rings[0].pucker_class == "chair"
    assert "rings" in prof_good.to_dict()

    # Distort the saturated ring atoms into a planar strained geometry
    mol_bad = Chem.Mol(mol_good)
    conf = mol_bad.GetConformer()
    sat_atoms = prof_good.rings[0].atom_indices
    pts = np.asarray(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in sat_atoms
        ],
        dtype=np.float64,
    )
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered)
    # Project onto best-fit plane (eliminating chair pucker) and compress slightly
    projected = centered - np.outer(centered @ vt[2], vt[2])
    for idx, atom_i in enumerate(sat_atoms):
        p_new = center + 0.90 * projected[idx]
        conf.SetAtomPosition(atom_i, (float(p_new[0]), float(p_new[1]), float(p_new[2])))

    prof_bad = analyze_ring_strain(mol_bad, strain_threshold_kcal=3.0, n_ref_confs=12, seed=42)
    assert prof_bad.max_ring_strain_kcal > 4.0
    assert prof_bad.passed is False

    # Protocol & registry integration
    flt = RingStrainFilter(max_strain_kcal=3.0, n_ref_confs=8)
    scr = RingStrainScorer(scale_kcal=3.0, n_ref_confs=8)
    assert isinstance(flt, MoleculeFilterProtocol)
    assert isinstance(scr, MoleculeScorerProtocol)
    assert "mmff94s" in RING_STRAIN_BACKEND_REGISTRY
    reports = ge.score_molecules([DRD1_1], include_admet=False, extra_scorers=[(scr, 0.2)])
    assert flt.keep(reports[0]) is True
    assert 0.5 < reports[0].custom_scores["ring_strain_plausibility"] <= 1.0


# ---------------------------------------------------------------------------
# 2. Multi-Conformation Ensemble Steering, Contrastive Off-Target CFG & PMI_3D
# ---------------------------------------------------------------------------


def test_contrastive_ensemble_logit_math_and_designer_methods(tmp_path: Path) -> None:
    """Verify exact contrastive ensemble CFG formula and end-to-end MoleculeDesigner API."""
    on1 = torch.tensor([[2.0, 0.0, 0.0]])
    on2 = torch.tensor([[0.0, 2.0, 0.0]])
    uncond = torch.tensor([[0.0, 0.0, 0.0]])
    off = torch.tensor([[0.0, 0.0, 3.0]])
    guided = combine_contrastive_ensemble_logits(
        [on1, on2], uncond, on_weights=[0.5, 0.5], w_on=2.0, off_logits=off, w_off=0.5
    )
    # Expected: 0 + 2.0 * ([1, 1, 0] - 0) - 0.5 * ([0, 0, 3] - 0) = [2.0, 2.0, -1.5]
    assert torch.allclose(guided, torch.tensor([[2.0, 2.0, -1.5]]), atol=1e-6)

    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    p1_pdb, p2_pdb = _write_two_pockets(tmp_path)
    p_on1 = ge.Pocket.from_pdb(p1_pdb, target_id="DRD1")
    p_on2 = ge.Pocket.from_pdb(p1_pdb, target_id="DRD1")
    p_off = ge.Pocket.from_pdb(p2_pdb, target_id="DRD2")

    designer = ge.MoleculeDesigner.from_mock(device="cpu", seed=42)
    ens_res = designer.design_ensemble(
        [p_on1, p_on2],
        off_pocket=p_off,
        w_on=1.5,
        w_off=0.4,
        n=3,
        max_new_tokens=12,
        include_admet=False,
        seed=42,
    )
    assert ens_res.metadata["mode"] == "contrastive_ensemble"
    assert ens_res.metadata["off_pocket"] == "DRD2"
    assert len(ens_res.raw_smiles) == 3

    pmis = designer.score_pmi([DRD1_1, PAROXETINE_SMILES], [p_on1, p_on2], off_pocket=p_off)
    assert len(pmis) == 2
    for score in pmis:
        assert score.n_tokens >= 2
        assert np.isfinite(score.pmi_3d)
        assert score.delta_pmi_selectivity is not None
        assert np.isfinite(score.delta_pmi_selectivity)


# ---------------------------------------------------------------------------
# 3. Natural-Product Scaffold-Hopping Audit (Ibogaine -> SERT)
# ---------------------------------------------------------------------------


def test_ibogaine_scaffold_hop_audit_quantifies_novelty_sa_and_hygiene() -> None:
    """Verify scaffold-hop audit simplifies ibogaine cage, scores hops, and vetoes DCU."""
    report = ge.audit_scaffold_hop_campaign(
        [PAROXETINE_SMILES, DRD1_1, IBOGAINE_SMILES, DCU],
        reference_smiles=IBOGAINE_SMILES,
        admet_suite=None,
    )
    assert report["reference"]["sa_score"] > 4.0  # Ibogaine has complex bridged polycyclic core
    assert report["summary"]["n_unique_valid"] == 4
    by_smi = {c["canonical_smiles"]: c for c in report["candidates"]}

    # DRD1_1 & Paroxetine are genuine scaffold hops with positive SA simplification vs Ibogaine
    assert by_smi[DRD1_1]["is_exact_scaffold_hop"] is True
    assert by_smi[DRD1_1]["delta_sa_simplification"] > 2.0
    assert by_smi[DRD1_1]["hop_composite_score"] > by_smi[IBOGAINE_SMILES]["hop_composite_score"]

    # DCU coupling reagent must still be hard-vetoed to 0.0
    assert by_smi[DCU]["hygiene_gate"] == pytest.approx(0.0)
    assert by_smi[DCU]["hop_composite_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Multi-Task Gradient Conflict Diagnostics & AIM-Inspired Intervention
# ---------------------------------------------------------------------------


def test_gradient_conflict_mediation_resolves_opposing_gradients_and_runs_on_painn() -> None:
    """Opposing gradients show negative cosine; PCGrad & AIM eliminate destructive cancellation."""
    g1 = torch.tensor([1.0, 1.0, 0.0])
    g2 = torch.tensor([-1.0, 1.0, 0.0])
    cos_mat, norms, conflicts, total = compute_gradient_conflict_matrix([g1, g2])
    assert total == 1 and conflicts == 0  # orthogonal (cos == 0)

    g_opp = torch.tensor([-2.0, -0.5, 0.0])
    cos2, _, conflicts2, _ = compute_gradient_conflict_matrix([g1, g_opp])
    assert conflicts2 == 1 and float(cos2[0, 1].item()) < -0.5
    merged_pc, _, _ = mediate_pcgrad([g1, g_opp])
    # After PCGrad projection, merged vector must have non-negative inner product with both tasks
    assert float(torch.dot(merged_pc, g1).item()) >= -1e-5
    assert float(torch.dot(merged_pc, g_opp).item()) >= -1e-5

    # End-to-end multi-task backward on a real GenEuclideanModel shared PaiNN-SE3 encoder
    seed_everything(42)
    cfg = ExperimentConfig(
        encoder=EncoderConfig(arch="painn", hidden_dim=32, vector_dim=16, out_dim=32, num_layers=1),
        cvae=CVAEConfig(enabled=False),
        decoder=DecoderConfig(
            arch="transformer", vocab_size=20, d_model=32, n_heads=4, n_layers=1, d_ff=64
        ),
        multitask=MultiTaskConfig(use_pkd_head=True, use_clip_head=True, use_prop_head=True),
    )
    model = GenEuclideanModel(cfg)
    batch = {
        "coords": torch.randn(3, 8, 3),
        "node_feats": torch.randn(3, 8, 52),
        "atom_mask": torch.ones(3, 8, dtype=torch.bool),
        "input_ids": torch.randint(0, 20, (3, 6)),
        "ligand_fp": torch.randint(0, 2, (3, cfg.multitask.ligand_fp_dim)).float(),
    }
    out = model(batch)
    assert out.pred_pkd is not None and out.pred_props is not None
    losses = {
        "lm_nll": nn.functional.cross_entropy(
            out.logits[:, :-1].reshape(-1, 20), batch["input_ids"][:, 1:].reshape(-1)
        ),
        "pkd": nn.functional.mse_loss(out.pred_pkd, torch.full_like(out.pred_pkd, 7.5)),
        "props": nn.functional.mse_loss(out.pred_props, torch.zeros_like(out.pred_props)),
    }
    mediator = GradientConflictMediator(
        task_names=("lm_nll", "pkd", "props"),
        strategy="aim_policy",
    )
    shared = list(model.encoder.parameters())
    diag = mediator.backward(losses, shared_params=shared)
    assert diag.strategy == "aim_policy"
    assert diag.policy_matrix is not None and len(diag.policy_matrix) == 3
    assert len(diag.effective_task_weights) == 3
    assert any(p.grad is not None and float(p.grad.abs().sum().item()) > 0.0 for p in shared)
    assert "cosine_matrix" in diag.to_dict()
