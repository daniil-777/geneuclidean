"""Offline CI-safe tests for GenEuclidean v4.0 Pharma CNS upgrades.

Covers:
1. Industrial MedChem structural triage, coupling-reagent quarantine (DCU, DIU, HOBt,
   DMAP, Ph3P=O, carbodiimides, Michael acceptors, acyl halides, poly-nitroaromatics),
   Lilly-inspired demerit scoring (0..300), ICH S10 phototoxicity, stereocenter burden,
   and Bemis-Murcko scaffold-hop FTO pre-screen with non-legal disclaimer.
2. BACE1 cyclic amidine / amino-oxazine pKa SMARTS accuracy in ``mpo.py``.
3. Split-Conformal binary prediction sets (finite-sample 90% coverage guarantee) and
   FDA Jan 2025 (``FDA-2024-D-4689``) / Joint FDA-EMA Jan 2026 7-Step Credibility Dossier.
4. Surgical ``featurizer.py`` cofactor retention (``FAD``/``ADP``) alongside metals while
   stripping drugs, detergents, lipids, and buffers with strict E(3) invariance.
5. Multiplicative Safety-Gated CNS Reward eliminating v3.1's additive ADMET-sink bleed
   (VMAT2 dimethoxy-isoquinoline scores <= 0.15 on PDE10A; DCU scores 0.0).
6. 24-target *Homo sapiens* Tri-Disorder CNS Atlas + 7 matched counter-targets with
   100% v3.1 backward compatibility, isoform offset aliases, and cofactor propagation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from geneuclidean.cns import (
    ALL_CNS_TARGET_REGISTRY,
    CNS_TARGET_REGISTRY,
    CREDIBILITY_SCHEMA_VERSION,
    DEFAULT_V4_ADMET_MODEL_PATH,
    DEFAULT_V4_STRUCTURES_DIR,
    FTO_MARKUSH_DISCLAIMER,
    REFERENCE_CNS_DRUGS,
    REFERENCE_NON_CNS_DRUGS,
    TRI_DISORDER_TARGET_REGISTRY,
    V4_COUNTER_TARGET_REGISTRY,
    V4_EXPANSION_TARGET_REGISTRY,
    CNSAdmetSuite,
    SplitConformalBinaryCalibrator,
    audit_pocket_hotspots,
    bemis_murcko_scaffold,
    build_credibility_dossier,
    compute_cns_mpo,
    compute_file_sha256,
    compute_multiplicative_cns_reward,
    compute_scaffold_hop_metrics,
    estimate_most_basic_pka,
    export_credibility_dossier,
    find_local_pocket_files,
    get_cns_target,
    hygiene_gate_score,
    is_scaffold_hop,
    list_cns_targets,
    list_tri_disorder_targets,
    load_cns_target_pocket,
    render_credibility_markdown,
    stereocenter_burden,
    triage_batch,
    triage_molecule,
)
from geneuclidean.data.featurizer import (
    DEFAULT_STRUCTURAL_COFACTORS,
    POCKET_NODE_DIM,
    featurize_pdb_pocket_inference,
)

# ---------------------------------------------------------------------------
# 1. MedChem Structural Triage, Reagent Quarantine & Scaffold-Hop Helpers
# ---------------------------------------------------------------------------


def test_reagent_quarantine_blocks_dcu_and_coupling_byproducts() -> None:
    """DCU (v3.1 DRD3_1), DIU, HOBt, DMAP, Ph3P=O, and reactive warheads score 0.0."""
    quarantined_cases: dict[str, str] = {
        "DCU": "O=C(NC1CCCCC1)NC1CCCCC1",
        "DIU": "CC(C)NC(=O)NC(C)C",
        "symmetric_dicyclopentylurea": "O=C(NC1CCCC1)NC1CCCC1",
        "HOBt": "On1nnc2ccccc21",
        "DMAP": "CN(C)c1ccncc1",
        "Ph3PO": "O=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "DCC": "C1CCC(N=C=NC2CCCCC2)CC1",
        "acyl_chloride": "CC(=O)Cl",
        "acrylamide_michael": "C=CC(=O)Nc1ccccc1",
        "maleimide_michael": "O=C1C=CC(=O)N1c1ccccc1",
        "dinitrobenzene": "O=[N+]([O-])c1cccc([N+](=O)[O-])c1",
    }
    for name, smi in quarantined_cases.items():
        res = triage_molecule(smi)
        assert res.valid, f"{name} should parse"
        assert not res.reagent_quarantine_free, f"{name} must trigger reagent quarantine"
        assert len(res.reagent_alerts) >= 1, f"{name} must record alert reason"
        assert res.hygiene_gate == 0.0, f"{name} hygiene_gate must be 0.0, got {res.hygiene_gate}"
        assert not res.triage_pass
        assert hygiene_gate_score(smi) == 0.0

    # PAINS rhodanine must also be hard-vetoed to 0.0
    rhodanine = "O=C1NC(=S)S/C1=C/c1ccc(O)cc1"
    pains_res = triage_molecule(rhodanine)
    assert not pains_res.pains_free
    assert pains_res.hygiene_gate == 0.0

    # Invalid SMILES returns valid=False and hygiene_gate=0.0
    inv = triage_molecule("NOT_A_VALID_SMILES")
    assert not inv.valid and inv.hygiene_gate == 0.0 and inv.lilly_demerit_score == 300


def test_clean_cns_and_ad_drugs_pass_reagent_quarantine_including_drug_ureas() -> None:
    """Clean CNS/AD clinical drugs (including aromatic ureas cariprazine/pimavanserin) pass."""
    clean_drugs: dict[str, str] = {
        "ecopipam": REFERENCE_CNS_DRUGS["ecopipam"],
        "xanomeline": REFERENCE_CNS_DRUGS["xanomeline"],
        "balipodect": REFERENCE_CNS_DRUGS["balipodect"],
        "haloperidol": REFERENCE_CNS_DRUGS["haloperidol"],
        "risperidone": REFERENCE_CNS_DRUGS["risperidone"],
        "cariprazine": "CN(C)C(=O)N[C@H]1CC[C@H](CCN2CCN(c3cccc(Cl)c3Cl)CC2)CC1",
        "pimavanserin": "CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1",
        "safinamide": "C[C@H](NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O",
        "ceperognastat": "CC(=O)Nc1nc(F)c(CN2CC[C@H](OCc3noc(C)n3)C[C@@H]2C)s1",
        "galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
    }
    results = triage_batch(list(clean_drugs.values()))
    for (name, _smi), res in zip(clean_drugs.items(), results, strict=True):
        assert res.valid
        assert res.reagent_quarantine_free, f"{name} false-positived reagent: {res.reagent_alerts}"
        assert res.pains_free, f"{name} false-positived PAINS: {res.pains_alerts}"
        assert res.lilly_pass, (
            f"{name} failed Lilly demerits ({res.lilly_demerit_score}): {res.lilly_demerit_reasons}"
        )
        assert res.hygiene_gate >= 0.70, f"{name} hygiene_gate too low: {res.hygiene_gate}"
        assert isinstance(res.to_dict(), dict)


def test_lilly_demerits_phototox_stereocenters_and_scaffold_hop() -> None:
    """Verify Lilly demerits, ICH S10 phototox flags, stereocenter burden, and FTO hop helper."""
    # Permanent quaternary ammonium (trospium) fails CNS Lilly demerit threshold (>100)
    trospium = triage_molecule(REFERENCE_NON_CNS_DRUGS["trospium_cation"])
    assert trospium.lilly_demerit_score > 100
    assert not trospium.lilly_pass
    assert trospium.hygiene_gate < 0.70

    # 4-fused-ring polyaromatic pyrene triggers fused aromatic + ICH S10 phototox hard reject
    pyrene = triage_molecule("c1cc2ccc3cccc4ccc(c1)c2c34")
    assert pyrene.fused_aromatic_rings >= 4
    assert pyrene.phototox_alert
    assert pyrene.lilly_demerit_score >= 120
    assert not pyrene.lilly_pass
    assert pyrene.hygiene_gate == 0.0

    # Phenothiazine triggers ICH S10 phototoxicity alert
    chlorpromazine = triage_molecule("CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21")
    assert chlorpromazine.phototox_alert
    assert len(chlorpromazine.phototox_alerts) >= 1

    # Macrocycle + excessive rotatable bonds elevate Lilly demerit score
    macro = triage_molecule("O=C1CCCCCCCCCCCNC(=O)CCCCCCCCCC1")
    assert macro.max_ring_size >= 12
    assert macro.lilly_demerit_score >= 65

    # Stereocenter burden distinguishes defined vs undefined stereocenters
    defined_stereo = stereocenter_burden(REFERENCE_CNS_DRUGS["ecopipam"])
    assert defined_stereo["total_chiral_centers"] == 2
    assert defined_stereo["defined_chiral_centers"] == 2
    assert defined_stereo["undefined_chiral_centers"] == 0

    undef_stereo = stereocenter_burden("CC(O)C(N)C(F)C(Cl)c1ccccc1")
    assert undef_stereo["undefined_chiral_centers"] == 4

    # Bemis-Murcko exact vs generic scaffolds & FTO scaffold-hop pre-screen
    ecopipam = REFERENCE_CNS_DRUGS["ecopipam"]
    exact_scaf = bemis_murcko_scaffold(ecopipam, generic=False)
    gen_scaf = bemis_murcko_scaffold(ecopipam, generic=True)
    assert exact_scaf and gen_scaf and exact_scaf != gen_scaf

    ref_pool = [ecopipam, REFERENCE_CNS_DRUGS["haloperidol"]]
    # Same molecule -> RED FTO flag, max_tanimoto == 1.0, not novel
    same_rep = compute_scaffold_hop_metrics(ecopipam, reference_smiles=ref_pool)
    assert same_rep.fto_pre_screen_flag == "RED"
    assert same_rep.max_tanimoto == pytest.approx(1.0, abs=1e-5)
    assert not same_rep.is_novel_scaffold
    assert "Markush" in same_rep.fto_disclaimer
    assert same_rep.fto_disclaimer == FTO_MARKUSH_DISCLAIMER

    # Structurally orthogonal balipodect vs ecopipam/haloperidol -> GREEN novel scaffold hop!
    hop_rep = compute_scaffold_hop_metrics(
        REFERENCE_CNS_DRUGS["balipodect"], reference_smiles=ref_pool
    )
    assert hop_rep.is_novel_scaffold
    assert hop_rep.max_tanimoto < 0.40
    assert hop_rep.fto_pre_screen_flag == "GREEN"
    assert is_scaffold_hop(
        REFERENCE_CNS_DRUGS["balipodect"],
        reference_scaffolds={bemis_murcko_scaffold(s, generic=True) for s in ref_pool},
    )


# ---------------------------------------------------------------------------
# 2. MPO v4.0 BACE1 Cyclic Amidine / Amino-Oxazine pKa SMARTS Fix
# ---------------------------------------------------------------------------


def test_bace1_cyclic_amidine_pka_smarts_refinement() -> None:
    """Verubecestat/Umibecestat/Elenbecestat get ~6.2-7.1 pKa while Metformin stays >=11.5."""
    verubecestat = Chem.MolFromSmiles("CN1C(N)=N[C@](C)(c2cc(NC(=O)c3ccc(F)cn3)ccc2F)CS1(=O)=O")
    umibecestat = Chem.MolFromSmiles(
        "C[C@@]1(c2nc(NC(=O)c3ncc(C(F)(F)F)cc3Cl)ccc2F)CO[C@@](C)(C(F)(F)F)C(N)=N1"
    )
    elenbecestat = Chem.MolFromSmiles(
        "C[C@H]1OC[C@]2(c3cc(NC(=O)c4cnc(C(F)F)cn4)ccc3F)N=C(N)SC[C@H]12"
    )
    metformin = Chem.MolFromSmiles(REFERENCE_NON_CNS_DRUGS["metformin"])
    assert all(m is not None for m in (verubecestat, umibecestat, elenbecestat, metformin))
    assert verubecestat is not None and umibecestat is not None
    assert elenbecestat is not None and metformin is not None

    assert 5.5 <= estimate_most_basic_pka(verubecestat) <= 7.5
    assert 6.0 <= estimate_most_basic_pka(umibecestat) <= 7.5
    assert 6.0 <= estimate_most_basic_pka(elenbecestat) <= 7.5
    assert estimate_most_basic_pka(metformin) >= 11.5

    veru_mpo = compute_cns_mpo(verubecestat)
    assert veru_mpo.t0_pka == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Split-Conformal Binary UQ & FDA Jan 2025 / Joint FDA-EMA Credibility Dossier
# ---------------------------------------------------------------------------


def test_split_conformal_finite_sample_coverage_and_credibility_dossier(tmp_path: Path) -> None:
    """Verify Split-Conformal 90% finite-sample coverage and 7-step FDA/EMA dossier export."""
    rng = np.random.default_rng(12345)
    n_cal, n_test = 600, 1200
    # Well-calibrated synthetic latent probabilities with realistic uncertainty
    p_true_all = rng.beta(0.8, 0.8, size=n_cal + n_test)
    y_all = (rng.random(n_cal + n_test) < p_true_all).astype(np.int64)

    cal = SplitConformalBinaryCalibrator(alpha=0.10).fit(p_true_all[:n_cal], y_all[:n_cal])
    eval_metrics = cal.evaluate_coverage(p_true_all[n_cal:], y_all[n_cal:])
    assert eval_metrics["target_coverage"] == pytest.approx(0.90)
    # Finite-sample guarantee: empirical coverage on exchangeable test set >= 0.88 (target 0.90)
    assert eval_metrics["empirical_coverage"] >= 0.88

    # High-confidence extremes produce singletons; borderline 0.50 flags uncertain_flag=True
    hi_pred = cal.predict_one(0.98)
    lo_pred = cal.predict_one(0.02)
    mid_pred = cal.predict_one(0.50)
    assert hi_pred.prediction_set == (1,) and not hi_pred.uncertain_flag
    assert lo_pred.prediction_set == (0,) and not lo_pred.uncertain_flag
    assert mid_pred.prediction_set == (0, 1) and mid_pred.uncertain_flag
    assert hi_pred.coverage_bound == pytest.approx(0.90)
    assert "set" in mid_pred.to_dict()

    # Mondrian class-conditional mode works cleanly on imbalanced classes
    mondrian = SplitConformalBinaryCalibrator(alpha=0.10, mondrian=True).fit(
        p_true_all[:n_cal], y_all[:n_cal]
    )
    m_eval = mondrian.evaluate_coverage(p_true_all[n_cal:], y_all[n_cal:])
    assert m_eval["empirical_coverage"] >= 0.87

    # Provenance SHA-256 & 7-step FDA Jan 2025 / Joint FDA-EMA Jan 2026 Credibility Dossier
    dummy_artifact = tmp_path / "dummy_weights.bin"
    dummy_artifact.write_bytes(b"geneuclidean-v4-test-provenance")
    sha = compute_file_sha256(dummy_artifact)
    assert sha is not None and len(sha) == 64
    assert compute_file_sha256(tmp_path / "missing.bin") is None

    mock_cv = {
        "endpoints": {
            "bbb": {
                "n_samples": 7802,
                "n_unique_scaffolds": 3200,
                "scaffold_leakage_groups": 0,
                "oof_roc_auc": 0.915,
                "oof_pr_auc": 0.942,
                "oof_brier_score": 0.109,
                "oof_ece": 0.028,
                "calibration_method": "sigmoid",
                "conformal_uq": eval_metrics,
            }
        }
    }
    dossier = build_credibility_dossier(
        admet_cv_metrics=mock_cv,
        physics_gate1_metrics={"median_rmsd_A": 1.02, "posebusters_pass_rate": 1.0},
        provenance_files={"dummy_model": dummy_artifact},
        generated_at_utc="2026-11-25T20:00:00+00:00",
    )
    assert dossier["schema_version"] == CREDIBILITY_SCHEMA_VERSION
    for step_key in (
        "step_1_question_of_interest",
        "step_2_context_of_use",
        "step_3_model_risk_assessment",
        "step_4_credibility_assessment_plan",
        "step_5_provenance_and_execution",
        "step_6_validation_results_and_deviations",
        "step_7_lifecycle_maintenance",
    ):
        assert step_key in dossier
    assert dossier["step_2_context_of_use"]["human_in_the_loop_required"] is True
    assert (
        dossier["step_5_provenance_and_execution"]["artifact_sha256_ledger"]["dummy_model"][
            "sha256"
        ]
        == sha
    )
    assert len(dossier["fda_ema_10_principles_checklist"]) == 10
    assert set(dossier["fda_ema_10_principles_checklist"].values()) == {None}
    key = "4_clear_context_of_use"
    reviewed = build_credibility_dossier(principle_assessments={key: False})
    assert reviewed["fda_ema_10_principles_checklist"][key] is False
    with pytest.raises(ValueError, match="known checklist principle"):
        build_credibility_dossier(principle_assessments={"unknown": True})

    md = render_credibility_markdown(dossier)
    assert "AI Credibility Dossier" in md and "bbb" in md
    j_path, m_path = export_credibility_dossier(
        dossier, tmp_path / "dossier.json", md_path=tmp_path / "dossier.md"
    )
    assert j_path.exists() and m_path is not None and m_path.exists()
    reloaded = json.loads(j_path.read_text(encoding="utf-8"))
    assert reloaded["schema_version"] == CREDIBILITY_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 4. Surgical featurizer.py Cofactor Retention (FAD / ADP) & E(3) Invariance
# ---------------------------------------------------------------------------


def test_featurizer_cofactor_retention_keeps_fad_adp_strips_drugs_and_detergents(
    tmp_path: Path,
) -> None:
    """featurizer.py keeps FAD/ADP when whitelisted while stripping drugs/lipids/detergents."""
    lines = [
        "ATOM      1  N   GLN A 206       1.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  CA  GLN A 206       2.400   0.000   0.000  1.00 20.00           C",
        "ATOM      3  OE1 GLN A 206       3.100   1.200   0.000  1.00 20.00           O",
        "ATOM      4  NE2 GLN A 206       3.100  -1.200   0.000  1.00 20.00           N",
        "ATOM      5  OH  TYR A 435      -2.000   1.500   0.500  1.00 20.00           O",
        "HETATM    6 ZN   ZN  A 501       0.000   2.500   0.000  1.00 20.00          ZN",
        # Integral FAD cofactor (MAOB/MAOA) -> 2 heavy atoms (P + N5)
        "HETATM    7  PA  FAD A 600      -1.000  -2.000   0.000  1.00 20.00           P",
        "HETATM    8  N5  FAD A 600      -1.500  -1.000   0.500  1.00 20.00           N",
        # Integral ADP cofactor (NLRP3) -> 2 heavy atoms (PB + O1B)
        "HETATM    9  PB  ADP A 702       0.500   0.500   2.500  1.00 20.00           P",
        "HETATM   10  O1B ADP A 702       1.200   0.500   3.200  1.00 20.00           O",
        # Organic drug (SAG with Fluorine), detergent (C15), lipid (CLR), buffer (GOL), water (HOH)
        "HETATM   11  F1  SAG A 801       0.000   0.000   0.000  1.00 20.00           F",
        "HETATM   12  C1  C15 A 802       2.000   2.000   2.000  1.00 20.00           C",
        "HETATM   13  C1  CLR A 803      -2.000  -2.000   2.000  1.00 20.00           C",
        "HETATM   14  O1  GOL A 804       1.500  -1.500  -1.500  1.00 20.00           O",
        "HETATM   15  O   HOH A 901       0.200   0.200   0.200  1.00 20.00           O",
        "END",
    ]
    pdb_path = tmp_path / "maob_nlrp3_cofactor_test.pdb"
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert "FAD" in DEFAULT_STRUCTURAL_COFACTORS and "ADP" in DEFAULT_STRUCTURAL_COFACTORS

    # Default retains 5 polymer + 1 ZN + 2 FAD + 2 ADP = 10 atoms (strips SAG/C15/CLR/GOL/HOH)
    f_default = featurize_pdb_pocket_inference(pdb_path, center=[0.0, 0.0, 0.0], pocket_cutoff=15.0)
    assert f_default.coords.shape == (10, 3)
    assert f_default.node_feats.shape == (10, POCKET_NODE_DIM)
    assert np.all(np.isfinite(f_default.node_feats))
    # Element idx 5 (Fluorine from SAG drug) must be 0; idx 4 (Phosphorus from FAD+ADP) must be 2
    assert float(f_default.node_feats[:, 5].sum()) == 0.0
    assert float(f_default.node_feats[:, 4].sum()) == 2.0
    assert float(f_default.node_feats[:, 8].sum()) == 1.0  # ZN metal

    # Target-specific keep_cofactors=("FAD",) keeps FAD (1 P) and strips ADP
    f_fad = featurize_pdb_pocket_inference(
        pdb_path, center=[0.0, 0.0, 0.0], pocket_cutoff=15.0, keep_cofactors=("FAD",)
    )
    assert f_fad.coords.shape == (8, 3)
    assert float(f_fad.node_feats[:, 4].sum()) == 1.0

    # Opt-out keep_cofactors=() strips both FAD and ADP (keeps 5 polymer + 1 ZN = 6)
    f_none = featurize_pdb_pocket_inference(
        pdb_path, center=[0.0, 0.0, 0.0], pocket_cutoff=15.0, keep_cofactors=()
    )
    assert f_none.coords.shape == (6, 3)
    assert float(f_none.node_feats[:, 4].sum()) == 0.0


# ---------------------------------------------------------------------------
# 5. Multiplicative Safety-Gated Reward (Anti-ADMET-Sink & DCU Veto Proof)
# ---------------------------------------------------------------------------


def test_multiplicative_reward_suppresses_vmat2_admet_sink_and_vetoes_dcu() -> None:
    """VMAT2 dimethoxy-isoquinoline scores <=0.15 on PDE10A; DCU scores 0.0; on-target wins."""
    # Exact v3.1 PDE10A_1 additive ADMET sink (MPO v2 ~ 5.94, VMAT2-like scaffold)
    vmat2_sink = "COc1cc2c(cc1OC)C(=O)N(C1CC1)CC2"
    # True on-target PDE10A drug (Balipodect)
    pde10a_true = REFERENCE_CNS_DRUGS["balipodect"]
    # v3.1 DRD3_1 coupling-reagent byproduct (DCU)
    dcu = "O=C(NC1CCCCC1)NC1CCCCC1"

    mpo_sink = compute_cns_mpo(vmat2_sink)
    assert mpo_sink.cns_mpo_v2 >= 5.5  # Confirm ultra-high MPO that fooled v3.1 additive reward

    scores = compute_multiplicative_cns_reward(
        [vmat2_sink, pde10a_true, dcu, "INVALID_SMILES"],
        on_target_prob=[0.01, 0.85, 0.95, 0.90],
        off_target_max_prob=[0.90, 0.05, 0.02, 0.01],
        pred_pkd=[7.8, 8.2, 8.5, 8.0],
        vina_scores=[-8.5, -10.2, -9.0, -9.0],
    )

    sink_sc, on_sc, dcu_sc, inv_sc = scores
    # 1. VMAT2 ADMET sink on PDE10A is strictly crushed (<= 0.15!) despite MPO=5.94
    assert sink_sc.valid
    assert sink_sc.total_reward <= 0.15, (
        f"ADMET sink must score <= 0.15 on PDE10A, got {sink_sc.total_reward:.4f}"
    )
    # 2. True on-target PDE10A lead scores strongly and dominates the off-target sink
    assert on_sc.valid and on_sc.total_reward >= 0.55
    assert on_sc.total_reward > sink_sc.total_reward * 5.0
    # 3. DCU coupling reagent is hard-vetoed to exact 0.0 even with hypothetical high QSAR/Vina!
    assert dcu_sc.valid and dcu_sc.hygiene_gate == 0.0 and dcu_sc.total_reward == 0.0
    # 4. Invalid SMILES -> 0.0
    assert not inv_sc.valid and inv_sc.total_reward == 0.0
    assert isinstance(on_sc.to_dict(), dict)


def test_v4_multi_endpoint_suite_conformal_prediction_and_roundtrip(tmp_path: Path) -> None:
    """Synthetic v4 multi-endpoint suite fits with 0 scaffold leakage, Conformal UQ & AD."""
    cns_smiles = [
        "Cc1ccccc1CN(C)C",
        "COc1ccccc1CCN(C)C",
        "Fc1ccccc1C1CCN(C)CC1",
        "Clc1ccc(CC2CCCCN2)cc1",
        "Cc1ccc2ccccc2c1CN1CCCC1",
        "COc1ccc2[nH]ccc2c1CCN(C)C",
        "c1ccc(OC2CCNCC2)cc1",
        "Cc1cc(F)ccc1NC(=O)C1CCCCN1",
        "Fc1ccc(C(=O)CCCN2CCCCC2)cc1",
        "Clc1ccc(C2=NCCCN2)cc1",
        "Cc1noc(C2CCN(C)CC2)n1",
        "COc1cc2c(cc1OC)CCN(C)C2",
        "c1ccc2c(c1)CCc1ccccc1N2C",
        "Cc1nc2ccccc2n1CCN(C)C",
        "Fc1ccc(C2COc3ccccc3O2)cc1",
    ]
    polar_smiles = [
        "O=C(O)c1ccc(C(=O)O)c(O)c1O",
        "O=C(O)CC(O)(CC(=O)O)C(=O)O",
        "NC(CCC(=O)O)C(=O)NC(CC(=O)O)C(=O)O",
        "O=C(O)c1cc(O)c(O)c(O)c1",
        "OCC(O)C(O)C(O)C(O)C(=O)O",
        "O=C(O)c1ccc(S(=O)(=O)N)cc1O",
        "NC(=O)c1cc(C(=O)O)cc(C(=O)O)c1",
        "O=C(O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
        "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2=O",
        "O=C(O)c1cnc(C(=O)O)nc1O",
        "NC(Cc1ccc(O)cc1)C(=O)NC(CO)C(=O)O",
        "O=C(O)c1cc(S(=O)(=O)O)ccc1O",
        "O=P(O)(O)CC(O)(P(=O)(O)O)c1cccnc1",
        "OCC1OC(O)C(O)C(O)C1O",
        "O=C(O)c1ccc(NC(=O)CC(=O)O)cc1O",
    ]
    smiles = cns_smiles + polar_smiles
    y_bbb = [1] * len(cns_smiles) + [0] * len(polar_smiles)
    y_mdr1 = [0] * len(cns_smiles) + [1] * len(polar_smiles)
    y_5ht2b = [1 if i % 3 == 0 else 0 for i in range(len(cns_smiles))] + [0] * len(polar_smiles)

    suite = CNSAdmetSuite(
        endpoints=("bbb", "mdr1_efflux", "htr2b_avoid"),
        hgb_max_iter=15,
        rf_n_estimators=15,
        conformal_alpha=0.10,
        seed=42,
    )
    m_bbb = suite.fit_endpoint("bbb", smiles, y_bbb, n_splits=3)
    m_mdr = suite.fit_endpoint("mdr1_efflux", smiles, y_mdr1, n_splits=3)
    m_5ht = suite.fit_endpoint("htr2b_avoid", smiles, y_5ht2b, n_splits=3)
    for m in (m_bbb, m_mdr, m_5ht):
        assert m["scaffold_leakage_groups"] == 0
        assert "conformal_uq" in m
        assert 0.0 <= m["conformal_uq"]["empirical_coverage"] <= 1.0

    saved = suite.save(tmp_path / "v4_mini.joblib")
    reloaded = CNSAdmetSuite.load(saved)
    preds = reloaded.predict_v4([cns_smiles[0], polar_smiles[0], "BAD_SMI"])
    assert preds[0].valid and set(preds[0].probabilities.keys()) == {
        "bbb",
        "mdr1_efflux",
        "htr2b_avoid",
    }
    assert preds[0].conformal["bbb"] is not None
    assert preds[0].conformal["bbb"].coverage_bound == pytest.approx(0.90)
    assert not preds[2].valid

    # Multiplicative reward integrates seamlessly with v4 multi-endpoint suite
    mult = compute_multiplicative_cns_reward(
        [cns_smiles[0], "O=C(NC1CCCCC1)NC1CCCCC1"],
        on_target_prob=[0.85, 0.90],
        off_target_max_prob=[0.05, 0.05],
        admet_suite=reloaded,
        ad_damp=True,
    )
    assert mult[0].total_reward > 0.35
    assert mult[1].total_reward == 0.0


# ---------------------------------------------------------------------------
# 6. 24-Target Homo sapiens Tri-Disorder CNS Atlas + 7 Matched Counter-Targets
# ---------------------------------------------------------------------------


def test_24_target_tri_disorder_registry_backward_compat_and_aliases(tmp_path: Path) -> None:
    """Verify all 24 Tri-Disorder targets + 7 counter-targets while keeping v3.1 compat."""
    # Exact v3.1 backward compatibility
    assert len(CNS_TARGET_REGISTRY) == 12
    assert len(list_cns_targets()) == 12

    # Exact v4.0 Tri-Disorder counts: 12 v3.1 + 12 v4 = 24 primary; + 7 counters = 31 total
    assert len(V4_EXPANSION_TARGET_REGISTRY) == 12
    assert len(TRI_DISORDER_TARGET_REGISTRY) == 24
    assert len(V4_COUNTER_TARGET_REGISTRY) == 7
    assert len(ALL_CNS_TARGET_REGISTRY) == 31

    assert len(list_cns_targets(include_v4=True)) == 24
    assert len(list_tri_disorder_targets()) == 24
    assert len(list_tri_disorder_targets(include_counters=True)) == 31

    ad_targets = list_cns_targets(disease_area="alzheimer")
    assert {t.target_id for t in ad_targets} == {"ACHE", "OGA", "NLRP3", "MAOB", "GSK3B", "BACE1"}
    ts_targets = list_cns_targets(disease_area="tourette")
    assert {"DRD1", "VMAT2", "MGLL", "PDE1B"}.issubset({t.target_id for t in ts_targets})
    scz_targets = list_cns_targets(disease_area="schizophrenia")
    assert {"CHRM4", "CHRM1", "HTR1A", "GLYT1", "GPR52"}.issubset(
        {t.target_id for t in scz_targets}
    )

    # Verified human ChEMBL IDs, gene aliases, cofactors, and isoform offset aliases
    assert get_cns_target("OGA").chembl_id == "CHEMBL5921"
    assert get_cns_target("MGEA5").target_id == "OGA"
    assert get_cns_target("SLC6A9").target_id == "GLYT1"
    assert get_cns_target("NLRP3").chembl_id == "CHEMBL1741208"
    assert get_cns_target("NLRP3").keep_cofactors == ("ADP",)
    assert get_cns_target("MAOB").keep_cofactors == ("FAD",)
    assert get_cns_target("MAOA").keep_cofactors == ("FAD",)
    assert "Asp93" in get_cns_target("BACE1").hotspot_isoform_aliases
    assert "Trp338" in get_cns_target("CHRM1").hotspot_isoform_aliases
    assert get_cns_target("CHRM1").primary_pdb_id == "9UCP"
    assert get_cns_target("BACE1").primary_pdb_id == "6EQM"

    # Every target across all 31 entries has valid reference SMILES & primary anchor in hotspots
    for spec in ALL_CNS_TARGET_REGISTRY.values():
        assert spec.primary_anchor_residue in spec.hotspot_residues
        assert len(spec.hotspot_residues) >= 6
        assert len(spec.reference_drugs) >= 1
        for drug in spec.reference_drugs:
            assert Chem.MolFromSmiles(drug.canonical_smiles) is not None, (
                f"Invalid SMILES for {spec.target_id}/{drug.name}"
            )

    # audit_pocket_hotspots respects BACE1 +61 alias (Asp93) and MAOB FAD cofactor
    syn_bace1 = tmp_path / "syn_bace1_5hu1.pdb"
    syn_bace1.write_text(
        "ATOM      1  OD1 ASP A  93       0.000   0.000   0.000  1.00 20.00           O\n"
        "ATOM      2  OD1 ASP A 289       2.000   0.000   0.000  1.00 20.00           O\n"
        "END\n",
        encoding="utf-8",
    )
    bace_audit = audit_pocket_hotspots(syn_bace1, get_cns_target("BACE1"))
    assert "Asp93" in bace_audit["present_hotspots"]
    assert "Asp289" in bace_audit["present_hotspots"]

    syn_maob = tmp_path / "syn_maob.pdb"
    syn_maob.write_text(
        "ATOM      1  OE1 GLN A 206       0.000   0.000   0.000  1.00 20.00           O\n"
        "HETATM    2  N5  FAD A 600       3.000   0.000   0.000  1.00 20.00           N\n"
        "END\n",
        encoding="utf-8",
    )
    maob_audit = audit_pocket_hotspots(syn_maob, get_cns_target("MAOB"))
    assert "Gln206" in maob_audit["present_hotspots"]
    assert "FAD" in maob_audit["present_cofactors"]


def test_local_v4_pockets_and_oracles_if_present() -> None:
    """Smoke-test real local v4_expansion structures & trained 12-endpoint suite when present."""
    if not (DEFAULT_V4_STRUCTURES_DIR / "clean_pockets").exists():
        pytest.skip("data/cns_neuro.nosync/v4_expansion/structures/clean_pockets not present in CI")

    for tid in (
        "CHRM1",
        "HTR1A",
        "GLYT1",
        "MGLL",
        "PDE1B",
        "GPR52",
        "ACHE",
        "OGA",
        "NLRP3",
        "MAOB",
        "GSK3B",
        "BACE1",
    ):
        spec = get_cns_target(tid)
        feat = load_cns_target_pocket(tid)
        assert feat.coords.shape[1] == 3
        assert feat.node_feats.shape == (len(feat.coords), POCKET_NODE_DIM)
        assert np.all(np.isfinite(feat.node_feats))
        found = find_local_pocket_files(tid)
        assert found is not None
        audit = audit_pocket_hotspots(found[0], spec, ligand_sdf=found[1])
        assert audit["anchor_in_contact"] is True
        for cof in spec.keep_cofactors:
            assert cof in audit["present_cofactors"]

    suite = CNSAdmetSuite.try_load(DEFAULT_V4_ADMET_MODEL_PATH)
    if suite is not None:
        assert len(suite.fitted_endpoints) >= 10
        preds = suite.predict_v4(
            [REFERENCE_CNS_DRUGS["xanomeline"], REFERENCE_NON_CNS_DRUGS["ceftriaxone"]]
        )
        assert preds[0].valid and preds[0].probabilities["bbb"] > 0.70
        assert preds[1].valid and preds[1].probabilities["bbb"] < 0.50
