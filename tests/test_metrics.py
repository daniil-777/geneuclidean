"""Comprehensive unit & integration tests for GenEuclidean v2 evaluation, 3D docking, and viz."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs

from geneuclidean.evaluation import (
    bootstrap_ci,
    canonicalize_smiles,
    check_pose_geometry,
    cliffs_delta,
    compute_frechet_fingerprint_distance,
    compute_internal_diversity,
    compute_jensen_shannon_divergence,
    compute_ligand_box_center,
    compute_molecule_properties,
    compute_properties_batch,
    compute_wasserstein_1,
    dice_similarity,
    dock_ligand_vina,
    embed_and_minimize_3d,
    evaluate_3d_plausibility,
    evaluate_affinity_metrics,
    evaluate_attention_enrichment,
    evaluate_clip_retrieval,
    evaluate_generation_metrics,
    evaluate_pocket_specificity,
    get_morgan_bit_fp,
    get_morgan_count_fp,
    get_murcko_scaffold,
    get_np_score,
    get_sa_score,
    is_brenk_free,
    is_pains_free,
    lipinski_violations,
    passes_lipinski,
    plot_decoding_pareto,
    plot_generated_molecule_grid,
    plot_matched_vs_shuffled_similarity,
    plot_model_comparison_bars,
    plot_pkd_parity,
    plot_pocket_attention_heatmap,
    plot_pocket_embedding_tsne,
    plot_property_distributions,
    plot_training_curves,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    sa_to_normalized,
    tanimoto_similarity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VINA_BIN = REPO_ROOT / "bin" / "vina"
POCKET_10GS = REPO_ROOT / "data" / "raw" / "pbpp2020" / "10gs" / "10gs_pocket.pdb"
LIGAND_10GS = REPO_ROOT / "data" / "raw" / "pbpp2020" / "10gs" / "10gs_ligand.sdf"

DRUG_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",  # Naproxen
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # Diclofenac
    "Cc1ccc(S(=O)(=O)N)cc1",  # p-Toluenesulfonamide
    "c1ccc2c(c1)ccc1ccccc12",  # Phenanthrene
    "O=C(O)c1ccncc1",  # Isonicotinic acid
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",  # Procainamide
]


# ---------------------------------------------------------------------------
# Tier 1: Canonicalization, IntDiv, Murcko, Tanimoto vs Dice, FFD, MedChem
# ---------------------------------------------------------------------------


def test_canonicalization_catches_noncanonical_duplicates() -> None:
    """Non-canonical SMILES variants of the same molecule must collapse to 1 unique."""
    ethanol_variants = ["CCO", "OCC", "C(O)C", "C(C)O"]
    benzene_variants = ["c1ccccc1", "C1=CC=CC=C1"]
    generated = ethanol_variants + benzene_variants + ["INVALID_SMILES", ""]
    # Train set contains ethanol written differently ("OCC")
    train = ["OCC"]

    metrics = evaluate_generation_metrics(generated, train_smiles=train)
    assert metrics["n_generated"] == 8.0
    assert metrics["n_valid"] == 6.0
    assert metrics["n_unique"] == 2.0
    assert metrics["n_novel"] == 1.0  # Benzene is novel; ethanol is NOT novel!
    assert math.isclose(metrics["validity"], 6.0 / 8.0)
    assert math.isclose(metrics["uniqueness"], 2.0 / 6.0)
    assert math.isclose(metrics["novelty"], 1.0 / 2.0)

    assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")
    assert canonicalize_smiles("not_a_mol") is None


def test_internal_diversity_bounds_and_exactness() -> None:
    """Verify exact MOSES IntDiv1/2 bounds, power-mean ordering, and brute-force parity."""
    mols = [Chem.MolFromSmiles(s) for s in DRUG_SMILES]
    fps = [get_morgan_bit_fp(m) for m in mols if m is not None]

    # Identical molecules -> IntDiv1 == IntDiv2 == 0.0
    identical_fps = [fps[0]] * 8
    assert math.isclose(compute_internal_diversity(identical_fps, p=1), 0.0, abs_tol=1e-7)
    assert math.isclose(compute_internal_diversity(identical_fps, p=2), 0.0, abs_tol=1e-7)

    # Diverse drug set -> strictly inside (0, 1) and IntDiv2 <= IntDiv1 (Jensen/power-mean)
    div1 = compute_internal_diversity(fps, p=1)
    div2 = compute_internal_diversity(fps, p=2)
    assert 0.5 < div1 < 1.0
    assert 0.4 < div2 <= div1 + 1e-9

    # Exact verification against RDKit pairwise Tanimoto loop
    n = len(fps)
    exact_sum_p1 = 0.0
    exact_sum_p2 = 0.0
    for i in range(n):
        for j in range(n):
            t = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            exact_sum_p1 += t
            exact_sum_p2 += t * t
    expected_div1 = 1.0 - exact_sum_p1 / (n * n)
    expected_div2 = 1.0 - math.sqrt(exact_sum_p2 / (n * n))
    assert math.isclose(div1, expected_div1, abs_tol=1e-6)
    assert math.isclose(div2, expected_div2, abs_tol=1e-6)

    # Stratified subsample path when len(fps) > max_exact
    many_fps = fps * 3
    div_sub = compute_internal_diversity(many_fps, p=1, max_exact=12, seed=7)
    assert 0.0 <= div_sub <= 1.0


def test_murcko_scaffold_diversity_and_novelty() -> None:
    """Substituted rings sharing a Bemis-Murcko core collapse to 1 scaffold."""
    # All 3 have a simple benzene core; the 4th has a pyridine core
    generated = [
        "Cc1ccccc1",
        "CCc1ccccc1",
        "O=C(O)c1ccccc1",
        "Cc1ccncc1",
    ]
    train = ["Oc1ccccc1"]  # Benzene scaffold is already seen in train

    m0 = Chem.MolFromSmiles(generated[0])
    assert m0 is not None
    assert get_murcko_scaffold(m0) == "c1ccccc1"

    res = evaluate_generation_metrics(generated, train_smiles=train)
    assert math.isclose(res["murcko_scaffold_diversity"], 2.0 / 4.0)
    assert math.isclose(res["murcko_scaffold_novelty"], 1.0 / 2.0)


def test_tanimoto_vs_dice_exact_mathematical_relationship() -> None:
    """On bit vectors Dice == 2*T/(1+T) identically; count Dice differs & is tracked."""
    m1 = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    m2 = Chem.MolFromSmiles("CC(=O)Oc1ccc(C)cc1C(=O)O")
    assert m1 is not None and m2 is not None

    bv1, bv2 = get_morgan_bit_fp(m1), get_morgan_bit_fp(m2)
    t_bv = tanimoto_similarity(bv1, bv2)
    d_bv = dice_similarity(bv1, bv2)
    assert math.isclose(d_bv, 2.0 * t_bv / (1.0 + t_bv), abs_tol=1e-12)
    assert d_bv > t_bv

    # Count fingerprint Dice (exact 2021 thesis metric)
    cfp1, cfp2 = get_morgan_count_fp(m1), get_morgan_count_fp(m2)
    d_cnt = dice_similarity(cfp1, cfp2)
    assert 0.0 < d_cnt <= 1.0
    assert d_cnt > t_bv


def test_frechet_fingerprint_distance_properties() -> None:
    """FFD(X, X) ~ 0, FFD(X, Y) > 0, and FFD(X, Y) == FFD(Y, X)."""
    aromatic = [
        "c1ccccc1",
        "c1ccc2ccccc2c1",
        "c1ccc2c(c1)ccc1ccccc12",
        "Cc1ccccc1",
        "COc1ccccc1",
        "Nc1ccccc1",
        "O=C(O)c1ccccc1",
        "Clc1ccccc1",
    ]
    aliphatic = [
        "CCCCCC",
        "CCCCCCC",
        "CCCCCCCC",
        "CCC(C)CC",
        "CCCC(O)CC",
        "CCCCCC(=O)O",
        "CCCCCN",
        "CCC(C)CCC",
    ]
    fps_a = [get_morgan_bit_fp(m) for s in aromatic if (m := Chem.MolFromSmiles(s)) is not None]
    fps_b = [get_morgan_bit_fp(m) for s in aliphatic if (m := Chem.MolFromSmiles(s)) is not None]

    ffd_same = compute_frechet_fingerprint_distance(fps_a, fps_a)
    ffd_ab = compute_frechet_fingerprint_distance(fps_a, fps_b)
    ffd_ba = compute_frechet_fingerprint_distance(fps_b, fps_a)

    assert ffd_same < 1e-6
    assert ffd_ab > 0.5
    assert math.isclose(ffd_ab, ffd_ba, rel_tol=1e-6, abs_tol=1e-6)


def test_medchem_descriptors_pains_brenk_lipinski() -> None:
    """Validate QED, SA/SA_norm, NP_score, PAINS, Brenk, Lipinski, and W1/JSD."""
    ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    assert ibuprofen is not None
    props = compute_molecule_properties(ibuprofen)
    assert 0.0 < props["QED"] < 1.0
    assert 1.0 <= props["SA"] <= 10.0
    assert math.isclose(props["SA_norm"], sa_to_normalized(props["SA"]), abs_tol=1e-12)
    assert props["SA_norm"] > 0.7
    assert math.isfinite(get_sa_score(ibuprofen))
    assert math.isfinite(get_np_score(ibuprofen))
    assert is_pains_free(ibuprofen)
    assert is_brenk_free(ibuprofen)
    assert lipinski_violations(ibuprofen) == 0
    assert passes_lipinski(ibuprofen)

    # Known PAINS alert: benzylidene rhodanine
    pains_mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
    assert pains_mol is not None
    assert not is_pains_free(pains_mol)

    # Known Brenk alert: reactive acyl chloride
    brenk_mol = Chem.MolFromSmiles("CC(=O)Cl")
    assert brenk_mol is not None
    assert not is_brenk_free(brenk_mol)

    # Lipinski violation (> 500 Da, LogP > 5, > 5 HBD/HBA failure check)
    huge_lipophilic = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O")
    assert huge_lipophilic is not None
    assert lipinski_violations(huge_lipophilic) >= 2
    assert not passes_lipinski(huge_lipophilic, max_violations=1)

    # Distributional distances
    assert math.isclose(compute_wasserstein_1([1, 2, 3], [1, 2, 3]), 0.0, abs_tol=1e-12)
    assert compute_wasserstein_1([1, 2, 3], [10, 11, 12]) > 5.0
    jsd_same = compute_jensen_shannon_divergence([1, 2, 3, 4], [1, 2, 3, 4])
    assert math.isclose(jsd_same, 0.0, abs_tol=1e-8)
    assert 0.0 < compute_jensen_shannon_divergence([1, 2, 3, 4], [10, 11, 12, 13]) <= 1.0

    # End-to-end Tier 1 benchmark dictionary completeness
    full = evaluate_generation_metrics(
        DRUG_SMILES[:6],
        train_smiles=DRUG_SMILES[6:],
        reference_smiles=DRUG_SMILES[4:],
    )
    for key in (
        "validity",
        "uniqueness",
        "novelty",
        "int_div_1",
        "int_div_2",
        "murcko_scaffold_diversity",
        "murcko_scaffold_novelty",
        "snn",
        "ffd",
        "QED_mean",
        "SA_mean",
        "SA_norm_mean",
        "NP_score_mean",
        "MW_w1",
        "LogP_jsd",
        "lipinski_pass_rate",
        "pains_free_rate",
        "brenk_free_rate",
    ):
        assert key in full
        assert math.isfinite(full[key])


# ---------------------------------------------------------------------------
# Tier 2: Causal Proof of 3D Pocket Conditioning
# ---------------------------------------------------------------------------


def test_cliffs_delta_exactness_and_bootstrap_ci() -> None:
    """Cliff's delta matches brute-force dominance probability including ties."""
    x = np.array([1.0, 2.0, 3.0, 3.0, 5.0])
    y = np.array([1.0, 1.5, 2.5, 3.0])
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    expected = (gt - lt) / (len(x) * len(y))
    assert math.isclose(cliffs_delta(x, y), expected, abs_tol=1e-12)
    assert math.isclose(cliffs_delta(y, x), -expected, abs_tol=1e-12)
    assert math.isclose(cliffs_delta(x, x), 0.0, abs_tol=1e-12)

    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), n_bootstrap=500, seed=42)
    assert lo < 3.0 < hi


def test_pocket_specificity_matched_vs_shuffled_vs_null() -> None:
    """Matched pocket analogs must significantly outperform shuffled & null controls."""
    native = {
        "p1": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like
        "p2": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen-like
        "p3": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
        "p4": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",  # Naproxen-like
        "p5": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # Diclofenac-like
        "p6": "Cc1ccc(S(=O)(=O)N)cc1",  # Sulfonamide-like
        "p7": "CCN(CC)CCNC(=O)c1ccc(N)cc1",  # Procainamide-like
        "p8": "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen-like
    }
    matched = {
        "p1": [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Oc1ccc(C)cc1C(=O)O",
            "CC(=O)Oc1ccc(F)cc1C(=O)O",
        ],
        "p2": [
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "CC(C)Cc1ccc(CC(=O)O)cc1",
            "CCCc1ccc(C(C)C(=O)O)cc1",
        ],
        "p3": [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
            "CCn1cnc2c1c(=O)n(C)c(=O)n2C",
        ],
        "p4": [
            "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
            "COc1ccc2cc(CC(=O)O)ccc2c1",
            "CCOc1ccc2cc(C(C)C(=O)O)ccc2c1",
        ],
        "p5": [
            "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
            "O=C(O)Cc1ccccc1Nc1c(F)cccc1Cl",
            "O=C(O)Cc1ccccc1Nc1ccccc1Cl",
        ],
        "p6": [
            "Cc1ccc(S(=O)(=O)N)cc1",
            "CCc1ccc(S(=O)(=O)N)cc1",
            "COc1ccc(S(=O)(=O)N)cc1",
        ],
        "p7": [
            "CCN(CC)CCNC(=O)c1ccc(N)cc1",
            "CN(C)CCNC(=O)c1ccc(N)cc1",
            "CCN(CC)CCCNC(=O)c1ccc(N)cc1",
        ],
        "p8": [
            "CC(=O)Nc1ccc(O)cc1",
            "CCC(=O)Nc1ccc(O)cc1",
            "CC(=O)Nc1ccc(OC)cc1",
        ],
    }
    keys = list(native.keys())
    # Rotate pocket assignments so every pocket receives a mismatched decoy set
    shuffled = {k: matched[keys[(i + 3) % len(keys)]] for i, k in enumerate(keys)}
    null_ctrl = {k: ["CCCCCCC", "CCCCCCCCO", "C1CCCCC1"] for k in keys}

    res = evaluate_pocket_specificity(
        per_pocket_generated=matched,
        per_pocket_native_smiles=native,
        per_pocket_shuffled_generated=shuffled,
        per_pocket_null_generated=null_ctrl,
        n_bootstrap=400,
        seed=42,
    )

    assert res["n_pockets"] == 8
    assert res["matched_tanimoto_mean"] > 0.65
    assert res["matched_count_dice_mean"] > res["matched_tanimoto_mean"]
    assert res["intra_uniqueness_mean"] == 1.0
    assert res["intra_int_div_1_mean"] > 0.0

    # Cross-pocket property correlations should be strongly positive
    assert res["property_correlations"]["MW"]["pearson_r"] > 0.90

    # Significant superiority over both shuffled and null controls
    vs_shuf = res["vs_shuffled"]
    assert vs_shuf["wilcoxon_p"] < 0.02
    assert vs_shuf["mannwhitneyu_p"] < 1e-4
    assert vs_shuf["cliffs_delta"] > 0.85
    assert vs_shuf["diff_tanimoto_ci95"][0] > 0.3

    vs_null = res["vs_null"]
    assert vs_null["wilcoxon_p"] < 0.02
    assert vs_null["cliffs_delta"] > 0.9


# ---------------------------------------------------------------------------
# Tier 3: 3D Cross-Attention Interpretability Audit
# ---------------------------------------------------------------------------


def test_attention_enrichment_uniform_and_focused() -> None:
    """Uniform attention gives enrichment=1 & norm_entropy=1; contact focus gives >1."""
    # 10 slots: 8 valid atoms (first 2 are contacts, next 6 peripheral), 2 padded slots
    atom_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    contact_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

    # Uniform over valid atoms + non-zero noise on masked padded slots (must be ignored!)
    attn_uniform = np.ones((5, 10), dtype=np.float64)
    attn_uniform[:, 8:] = 99.0
    res_u = evaluate_attention_enrichment(attn_uniform, atom_mask, contact_mask)
    assert math.isclose(res_u["enrichment_ratio"], 1.0, abs_tol=1e-6)
    assert math.isclose(res_u["enrichment_vs_uniform"], 1.0, abs_tol=1e-6)
    assert math.isclose(res_u["normalized_entropy"], 1.0, abs_tol=1e-6)

    # 5x attention weight on contact atoms vs peripheral atoms (4D batched input [B, H, T, N])
    attn_focused = np.ones((2, 4, 6, 10), dtype=np.float64)
    attn_focused[..., :2] = 5.0
    res_f = evaluate_attention_enrichment(
        attn_focused,
        np.stack([atom_mask, atom_mask]),
        np.stack([contact_mask, contact_mask]),
    )
    assert math.isclose(res_f["enrichment_ratio"], 5.0, abs_tol=1e-6)
    assert res_f["enrichment_vs_uniform"] > 2.0
    assert 0.0 < res_f["normalized_entropy"] < 1.0


# ---------------------------------------------------------------------------
# Tier 4: Binding Affinity (pKd) & DrugCLIP Retrieval
# ---------------------------------------------------------------------------


def test_affinity_metrics_and_clip_retrieval() -> None:
    """Verify exact pKd scoring metrics and DrugCLIP Recall@K / MRR / ROC-AUC."""
    y_true = np.array([4.0, 5.5, 6.2, 7.1, 8.4, 9.3, 10.1])
    perfect = evaluate_affinity_metrics(y_true, y_true)
    assert math.isclose(perfect["pearson_r"], 1.0, abs_tol=1e-12)
    assert math.isclose(perfect["spearman_rho"], 1.0, abs_tol=1e-12)
    assert math.isclose(perfect["kendall_tau"], 1.0, abs_tol=1e-12)
    assert math.isclose(perfect["rmse"], 0.0, abs_tol=1e-12)
    assert math.isclose(perfect["mae"], 0.0, abs_tol=1e-12)
    assert math.isclose(perfect["r2"], 1.0, abs_tol=1e-12)

    noisy_pred = y_true + np.array([0.2, -0.3, 0.1, -0.2, 0.4, -0.1, 0.3])
    noisy = evaluate_affinity_metrics(y_true, noisy_pred)
    assert noisy["pearson_r"] > 0.98
    assert noisy["rmse"] > 0.0
    assert noisy["mae"] <= noisy["rmse"]

    # DrugCLIP retrieval: aligned embeddings -> perfect Recall@1/5/10, MRR, ROC-AUC
    rng = np.random.default_rng(42)
    p_embs = rng.normal(size=(24, 32))
    clip_perfect = evaluate_clip_retrieval(p_embs, p_embs + 1e-4 * rng.normal(size=(24, 32)))
    assert math.isclose(clip_perfect["p2l_recall@1"], 1.0)
    assert math.isclose(clip_perfect["l2p_recall@1"], 1.0)
    assert math.isclose(clip_perfect["mrr"], 1.0)
    assert math.isclose(clip_perfect["roc_auc"], 1.0)

    # Moderate noise -> monotonic Recall@1 <= Recall@5 <= Recall@10
    clip_noisy = evaluate_clip_retrieval(p_embs, p_embs + 1.2 * rng.normal(size=(24, 32)))
    assert 0.0 <= clip_noisy["recall@1"] <= clip_noisy["recall@5"] <= clip_noisy["recall@10"] <= 1.0
    assert clip_noisy["roc_auc"] > 0.75


# ---------------------------------------------------------------------------
# Tier 5: 3D Conformer, PoseBusters Checks & Live Apple Silicon Vina Docking
# ---------------------------------------------------------------------------


def test_embed_and_minimize_3d_and_posebusters() -> None:
    """Verify ETKDGv3 + MMFF94s 3D conformer, strain energy, volume, and PoseBusters."""
    res = embed_and_minimize_3d("CC(=O)Oc1ccccc1C(=O)O", max_iters=200, seed=42)
    assert res.success
    assert res.mol_3d is not None
    assert res.force_field == "MMFF94s"
    assert res.strain_energy >= -1e-6
    assert res.mol_volume > 100.0

    # Valid minimized conformer passes intramolecular PoseBusters checks
    checks = check_pose_geometry(res.mol_3d, pocket_pdb_path=None)
    assert checks.bond_lengths_ok
    assert checks.bond_angles_ok
    assert checks.aromatic_ring_planarity_ok
    assert checks.max_ring_planarity_dev < 0.05
    assert checks.internal_clashes_ok
    assert checks.passed_all

    # Prepare ligand PDBQT cleanly via Meeko (optional [docking] extra)
    if importlib.util.find_spec("meeko") is not None:
        pdbqt = prepare_ligand_pdbqt(res.mol_3d)
        assert pdbqt is not None
        assert "ATOM" in pdbqt or "HETATM" in pdbqt

    # Distort an atom massively -> PoseBusters must detect broken bond length
    broken = Chem.Mol(res.mol_3d)
    conf = broken.GetConformer(0)
    conf.SetAtomPosition(0, (50.0, 50.0, 50.0))
    bad_checks = check_pose_geometry(broken, pocket_pdb_path=None)
    assert not bad_checks.bond_lengths_ok
    assert not bad_checks.passed_all

    # Invalid SMILES fails gracefully
    bad_res = embed_and_minimize_3d("NOT_A_SMILES")
    assert not bad_res.success
    assert bad_res.mol_3d is None

    # Batch 3D evaluation without Vina
    batch = evaluate_3d_plausibility(["CC(=O)Oc1ccccc1C(=O)O", "CCO", "INVALID"], run_vina=False)
    assert math.isclose(batch["conformer_success_rate"], 2.0 / 3.0)
    assert batch["intra_posebusters_pass_rate"] == 1.0
    assert not batch["vina_ran"]


@pytest.mark.skipif(
    importlib.util.find_spec("meeko") is None
    or not (VINA_BIN.exists() and POCKET_10GS.exists() and LIGAND_10GS.exists()),
    reason="Meeko, native bin/vina, or PDBbind 10gs complex not present in this environment",
)
def test_live_vina_docking_on_10gs(tmp_path: Path) -> None:
    """Live smoke test: receptor PDBQT prep + AutoDock Vina docking on 10gs pocket."""
    rec_pdbqt = prepare_receptor_pdbqt(POCKET_10GS, tmp_path / "10gs_pocket.pdbqt")
    assert rec_pdbqt is not None and rec_pdbqt.exists()
    rec_txt = rec_pdbqt.read_text(encoding="utf-8")
    assert "HOH" not in rec_txt
    assert "OA" in rec_txt and "A " in rec_txt

    box_center = compute_ligand_box_center(LIGAND_10GS)
    dock_res = dock_ligand_vina(
        smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        pocket_pdb_path=POCKET_10GS,
        box_center=box_center,
        box_size=(20.0, 20.0, 20.0),
        vina_bin=VINA_BIN,
        exhaustiveness=2,
        seed=42,
        receptor_pdbqt_path=rec_pdbqt,
    )
    assert dock_res.success, f"Vina docking failed: {dock_res.error}"
    assert dock_res.vina_affinity < -3.0
    assert len(dock_res.all_affinities) >= 1
    assert dock_res.docked_mol is not None
    assert dock_res.posebusters is not None
    assert dock_res.posebusters.bond_lengths_ok
    assert dock_res.posebusters.aromatic_ring_planarity_ok

    # Also smoke-test batch evaluate_3d_plausibility with run_vina=True
    batch = evaluate_3d_plausibility(
        smiles_list=["CC(=O)Oc1ccccc1C(=O)O"],
        pocket_pdb_path=POCKET_10GS,
        native_ligand_sdf=LIGAND_10GS,
        run_vina=True,
        vina_bin=VINA_BIN,
        max_dock=1,
        exhaustiveness=1,
        seed=42,
    )
    assert batch["vina_ran"]
    assert batch["vina_success_rate"] == 1.0
    assert batch["vina_affinity_best"] < -2.0
    assert math.isfinite(batch["native_vina_affinity"])
    assert 0.0 <= batch["high_affinity_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Visualization Suite Smoke Tests (all 9 publication-grade figures)
# ---------------------------------------------------------------------------


def _assert_valid_png(path: Path) -> None:
    assert path.exists()
    data = path.read_bytes()
    assert len(data) > 5000
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_all_visualization_figures_smoke(tmp_path: Path) -> None:
    """Smoke-test every publication-grade figure generator into ``tmp_path``."""
    rng = np.random.default_rng(42)

    # 1. Training curves (both generic keys and PocketTrainer.fit / history.jsonl schemas)
    histories = {
        "GenEuclidean v1.5": {
            "train_loss": [2.8, 2.2, 1.8, 1.5, 1.3],
            "val_loss": [2.9, 2.3, 1.9, 1.7, 1.6],
            "val_pkd_r": [0.30, 0.45, 0.55, 0.60, 0.63],
        },
        "GenEuclidean v2 (PaiNN)": {
            "train_total_loss": [2.4, 1.7, 1.3, 1.0, 0.85],
            "val_objective": [2.5, 1.8, 1.4, 1.15, 1.02],
            "val_ppl": [12.1, 6.0, 4.0, 3.1, 2.7],
            "val_pkd_pearson": [0.42, 0.60, 0.71, 0.76, 0.79],
            "val_kl_nats": [0.1, 0.4, 0.6, 0.55, 0.50],
        },
        "PocketTrainer JSONL": [
            {
                "epoch": 1,
                "train_total_loss": 2.6,
                "val_objective": 2.7,
                "val_ppl": 11.0,
                "val_pkd_pearson": 0.35,
                "train_kl_nats": 0.2,
            },
            {
                "epoch": 2,
                "train_total_loss": 1.9,
                "val_objective": 2.0,
                "val_ppl": 7.1,
                "val_pkd_pearson": 0.52,
                "train_kl_nats": 0.45,
            },
        ],
    }
    _assert_valid_png(plot_training_curves(histories, tmp_path / "01_training_curves.png"))
    _assert_valid_png(
        plot_training_curves(
            {"PocketTrainer Only": histories["PocketTrainer JSONL"]},
            tmp_path / "01b_pocket_trainer_curves.png",
        )
    )

    # 2. Model comparison bars
    comp_df = pd.DataFrame(
        [
            {
                "model": "2021 Thesis Baseline",
                "validity": 0.65,
                "uniqueness": 0.70,
                "novelty": 0.65,
                "int_div_1": 0.55,
                "QED": 0.45,
                "SA_norm": 0.67,
                "True Tanimoto": 0.17,
                "pKd Pearson r": 0.50,
            },
            {
                "model": "GenEuclidean v1.5",
                "validity": 0.86,
                "uniqueness": 0.90,
                "novelty": 0.92,
                "int_div_1": 0.74,
                "QED": 0.56,
                "SA_norm": 0.73,
                "True Tanimoto": 0.24,
                "pKd Pearson r": 0.64,
            },
            {
                "model": "PaiNN-PocketFormer",
                "validity": 0.98,
                "uniqueness": 0.96,
                "novelty": 0.97,
                "int_div_1": 0.81,
                "QED": 0.63,
                "SA_norm": 0.78,
                "True Tanimoto": 0.30,
                "pKd Pearson r": 0.73,
            },
            {
                "model": "GenEuclidean v2 (PaiNN)",
                "validity": 0.99,
                "uniqueness": 0.97,
                "novelty": 0.98,
                "int_div_1": 0.84,
                "QED": 0.67,
                "SA_norm": 0.80,
                "True Tanimoto": 0.34,
                "pKd Pearson r": 0.78,
            },
        ]
    )
    _assert_valid_png(plot_model_comparison_bars(comp_df, tmp_path / "02_model_comparison.png"))

    # 3. Matched vs Shuffled vs Null similarity
    matched_sims = np.clip(rng.normal(0.34, 0.08, size=120), 0.0, 1.0)
    shuffled_sims = np.clip(rng.normal(0.14, 0.05, size=120), 0.0, 1.0)
    null_sims = np.clip(rng.normal(0.13, 0.05, size=120), 0.0, 1.0)
    _assert_valid_png(
        plot_matched_vs_shuffled_similarity(
            matched_sims, shuffled_sims, null_sims, tmp_path / "03_matched_vs_shuffled.png"
        )
    )

    # 4. Property distributions (6-panel KDE + W1)
    ref_props = compute_properties_batch(DRUG_SMILES)
    gen_props = {
        "GenEuclidean v1.5": compute_properties_batch(DRUG_SMILES[:7]),
        "GenEuclidean v2": compute_properties_batch(DRUG_SMILES[2:]),
    }
    _assert_valid_png(
        plot_property_distributions(ref_props, gen_props, tmp_path / "04_property_dists.png")
    )

    # 5. pKd parity scatter
    yt = rng.uniform(3.0, 10.5, size=60)
    yp = yt + rng.normal(0.0, 0.8, size=60)
    _assert_valid_png(plot_pkd_parity(yt, yp, "CASF-2016 Core", tmp_path / "05_pkd_parity.png"))

    # 6. 3D pocket cross-attention heatmap
    tokens = ["c", "1", "c", "c", "c", "(", "O", ")", "c", "c", "1"]
    aa_list = ["TYR", "PHE", "ARG", "ASP", "LEU", "VAL", "SER", "HIS", "GLY", "TRP", "ALA", "GLU"]
    residues = [f"{aa}{idx}" for idx, aa in enumerate(aa_list, start=7)]
    contact_mask = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    attn_mat = rng.uniform(0.01, 0.06, size=(len(tokens), len(residues)))
    attn_mat[:, contact_mask] *= 3.5
    attn_mat /= attn_mat.sum(axis=1, keepdims=True)
    _assert_valid_png(
        plot_pocket_attention_heatmap(
            tokens, residues, attn_mat, contact_mask, "10gs", tmp_path / "06_pocket_attn.png"
        )
    )

    # 7. Generated molecule grid (RDKit)
    anns = [
        {"QED": 0.72, "SA": 2.1, "Tanimoto": 0.42, "pKd": 7.8, "Vina": -8.1},
        {"QED": 0.81, "SA": 1.9, "Tanimoto": 0.35, "pKd": 7.4, "Vina": -7.6},
        {"QED": 0.65, "SA": 2.4, "Tanimoto": 0.51, "pKd": 8.2, "Vina": -8.5},
    ]
    _assert_valid_png(
        plot_generated_molecule_grid(
            DRUG_SMILES[0], DRUG_SMILES[1:4], anns, "10gs", tmp_path / "07_mol_grid.png"
        )
    )

    # 8. Decoding Pareto frontier
    decoding_rows = [
        {"strategy": "Temp", "label": "T=0.7", "int_div_1": 0.64, "tanimoto": 0.31},
        {"strategy": "Temp", "label": "T=1.1", "int_div_1": 0.80, "tanimoto": 0.22},
        {"strategy": "Top-p", "label": "p=0.92", "int_div_1": 0.76, "tanimoto": 0.28},
        {"strategy": "Min-p+CFG", "label": "w=1.5", "int_div_1": 0.83, "tanimoto": 0.34},
        {"strategy": "Min-p+CFG", "label": "w=2.0", "int_div_1": 0.78, "tanimoto": 0.38},
        {"strategy": "DBS", "label": "G=8", "int_div_1": 0.72, "tanimoto": 0.36},
    ]
    _assert_valid_png(plot_decoding_pareto(decoding_rows, tmp_path / "08_decoding_pareto.png"))

    # 9. Pocket embedding t-SNE and PCA colored by pKd
    embs = rng.normal(size=(36, 64))
    pkd_labels = rng.uniform(3.5, 10.5, size=36)
    _assert_valid_png(plot_pocket_embedding_tsne(embs, pkd_labels, tmp_path / "09_pocket_tsne.png"))
    _assert_valid_png(
        plot_pocket_embedding_tsne(embs, pkd_labels, tmp_path / "09b_pocket_pca.png", method="pca")
    )
