"""Unit tests for GenEuclidean v3.1 CNS drug-design modules (MPO, ADMET, and 12 CNS targets).

Designed to run 100% offline and deterministically in GitHub Actions CI (even when
``data/cns_neuro`` is absent), while also smoke-testing cached local CNS structures and
the trained ADMET bundle when present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from geneuclidean.cns import (
    CNS_TARGET_REGISTRY,
    DEFAULT_ADMET_MODEL_PATH,
    DEFAULT_CNS_STRUCTURES_DIR,
    REFERENCE_CNS_DRUGS,
    REFERENCE_NON_CNS_DRUGS,
    CNSAdmetSuite,
    audit_pocket_hotspots,
    compute_cns_composite_reward,
    compute_cns_mpo,
    compute_cns_mpo_batch,
    compute_ece,
    desirability_clogd,
    desirability_clogp,
    desirability_hbd,
    desirability_mw,
    desirability_pka,
    desirability_tpsa,
    estimate_clogd_74,
    estimate_most_acidic_pka,
    estimate_most_basic_pka,
    find_local_pocket_files,
    get_cns_target,
    list_cns_targets,
    load_cns_target_pocket,
    make_scaffold_kfold,
)
from geneuclidean.data.splits import compute_murcko_scaffold

# ---------------------------------------------------------------------------
# 1. Exact Pfizer CNS MPO v1 & v2 Piecewise-Linear Boundaries & Clinical Drugs
# ---------------------------------------------------------------------------


def test_pfizer_cns_mpo_exact_piecewise_boundaries() -> None:
    """Verify exact Wager et al. (2010 & 2016) T_0 desirability piecewise boundaries."""
    # ClogP: <=3.0 -> 1.0, >=5.0 -> 0.0
    assert desirability_clogp(-1.0) == pytest.approx(1.0)
    assert desirability_clogp(3.0) == pytest.approx(1.0)
    assert desirability_clogp(4.0) == pytest.approx(0.5)
    assert desirability_clogp(5.0) == pytest.approx(0.0)
    assert desirability_clogp(6.5) == pytest.approx(0.0)

    # ClogD_7.4: <=2.0 -> 1.0, >=4.0 -> 0.0
    assert desirability_clogd(0.0) == pytest.approx(1.0)
    assert desirability_clogd(2.0) == pytest.approx(1.0)
    assert desirability_clogd(3.0) == pytest.approx(0.5)
    assert desirability_clogd(4.0) == pytest.approx(0.0)
    assert desirability_clogd(5.2) == pytest.approx(0.0)

    # MW: <=360 -> 1.0, >=500 -> 0.0
    assert desirability_mw(200.0) == pytest.approx(1.0)
    assert desirability_mw(360.0) == pytest.approx(1.0)
    assert desirability_mw(430.0) == pytest.approx(0.5)
    assert desirability_mw(500.0) == pytest.approx(0.0)
    assert desirability_mw(650.0) == pytest.approx(0.0)

    # TPSA hump: <=20 -> 0.0, 40..90 -> 1.0, >=120 -> 0.0
    assert desirability_tpsa(0.0) == pytest.approx(0.0)
    assert desirability_tpsa(20.0) == pytest.approx(0.0)
    assert desirability_tpsa(30.0) == pytest.approx(0.5)
    assert desirability_tpsa(40.0) == pytest.approx(1.0)
    assert desirability_tpsa(65.0) == pytest.approx(1.0)
    assert desirability_tpsa(90.0) == pytest.approx(1.0)
    assert desirability_tpsa(105.0) == pytest.approx(0.5)
    assert desirability_tpsa(120.0) == pytest.approx(0.0)
    assert desirability_tpsa(160.0) == pytest.approx(0.0)

    # HBD: <=0.5 -> 1.0, >=3.5 -> 0.0 (linear between 0.5 and 3.5)
    assert desirability_hbd(0.0) == pytest.approx(1.0)
    assert desirability_hbd(0.5) == pytest.approx(1.0)
    assert desirability_hbd(1.0) == pytest.approx(5.0 / 6.0)
    assert desirability_hbd(2.0) == pytest.approx(0.5)
    assert desirability_hbd(3.0) == pytest.approx(1.0 / 6.0)
    assert desirability_hbd(3.5) == pytest.approx(0.0)
    assert desirability_hbd(5.0) == pytest.approx(0.0)

    # most_basic_pKa: <=8.0 -> 1.0, >=10.0 -> 0.0
    assert desirability_pka(2.0) == pytest.approx(1.0)
    assert desirability_pka(8.0) == pytest.approx(1.0)
    assert desirability_pka(9.0) == pytest.approx(0.5)
    assert desirability_pka(10.0) == pytest.approx(0.0)
    assert desirability_pka(12.0) == pytest.approx(0.0)


def test_pka_and_henderson_hasselbalch_clogd_chemistry() -> None:
    """Verify SMARTS pKa tiers and physiological pH 7.4 Henderson-Hasselbalch ClogD shifts."""
    benzene = Chem.MolFromSmiles("c1ccccc1")
    aniline = Chem.MolFromSmiles("Nc1ccccc1")
    pyridine = Chem.MolFromSmiles("c1ccncc1")
    guanfacine = Chem.MolFromSmiles(REFERENCE_CNS_DRUGS["guanfacine"])
    metformin = Chem.MolFromSmiles(REFERENCE_NON_CNS_DRUGS["metformin"])
    ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    trospium = Chem.MolFromSmiles(REFERENCE_NON_CNS_DRUGS["trospium_cation"])

    assert all(
        m is not None
        for m in (benzene, aniline, pyridine, guanfacine, metformin, ibuprofen, trospium)
    )
    assert estimate_most_basic_pka(benzene) <= 4.0
    assert 4.0 <= estimate_most_basic_pka(aniline) <= 5.0
    assert 4.8 <= estimate_most_basic_pka(pyridine) <= 5.8
    # Acylguanidine (guanfacine) is resonance-dampened (~8.1) vs free biguanide metformin (~12.0)
    assert 7.5 <= estimate_most_basic_pka(guanfacine) <= 8.6
    assert estimate_most_basic_pka(metformin) >= 11.0
    assert estimate_most_basic_pka(trospium) >= 13.0

    # Acidic carboxylic acid (ibuprofen) must drop ClogD_7.4 strongly relative to ClogP
    acid_pka = estimate_most_acidic_pka(ibuprofen)
    assert acid_pka is not None and 3.5 <= acid_pka <= 5.0
    ibu_res = compute_cns_mpo(ibuprofen)
    assert ibu_res.clogd_74 < ibu_res.clogp - 2.0

    # Neutral benzene should have negligible ionization shift
    benz_logp = compute_cns_mpo(benzene).clogp
    assert estimate_clogd_74(benzene) == pytest.approx(benz_logp, abs=1e-4)


def test_cns_mpo_reference_fda_drugs_vs_non_cns_controls() -> None:
    """Known FDA CNS drugs score high (>=4.0) while polar non-CNS drugs score low."""
    key_cns = [
        "guanfacine",
        "haloperidol",
        "risperidone",
        "ecopipam",
        "valbenazine",
        "tetrabenazine",
        "xanomeline",
        "escitalopram",
        "paroxetine",
        "lumateperone",
        "dexmethylphenidate",
        "bupropion",
        "ulotaront",
        "balipodect",
    ]
    cns_v2_scores: list[float] = []
    for name in key_cns:
        res = compute_cns_mpo(REFERENCE_CNS_DRUGS[name])
        assert res.valid, f"{name} failed RDKit parse"
        # Exact Wager v1 & v2 algebraic identities
        expected_v1 = (
            res.t0_clogp + res.t0_clogd + res.t0_mw + res.t0_tpsa + res.t0_hbd + res.t0_pka
        )
        expected_v2 = res.t0_clogp + res.t0_mw + res.t0_tpsa + res.t0_pka + 2.0 * res.t0_hbd
        assert res.cns_mpo_v1 == pytest.approx(expected_v1, abs=1e-6)
        assert res.cns_mpo_v2 == pytest.approx(expected_v2, abs=1e-6)
        assert res.cns_mpo_v2 >= 4.0, (
            f"Expected CNS drug {name} MPO v2 >= 4.0, got {res.cns_mpo_v2}"
        )
        assert res.pains_free
        cns_v2_scores.append(res.cns_mpo_v2)

    polar_non_cns = ["metformin", "methotrexate", "ceftriaxone", "doxorubicin"]
    non_cns_v2_scores: list[float] = []
    for name in polar_non_cns:
        res = compute_cns_mpo(REFERENCE_NON_CNS_DRUGS[name])
        assert res.valid, f"{name} failed RDKit parse"
        assert res.cns_mpo_v2 <= 3.1, f"Expected non-CNS {name} MPO v2 <= 3.1, got {res.cns_mpo_v2}"
        non_cns_v2_scores.append(res.cns_mpo_v2)

    assert float(np.mean(cns_v2_scores)) - float(np.mean(non_cns_v2_scores)) >= 2.0

    # Batch API, PAINS alert detection, and invalid SMILES handling
    rhodanine_pains = "O=C1NC(=S)S/C1=C/c1ccc(O)cc1"
    batch = compute_cns_mpo_batch(
        [REFERENCE_CNS_DRUGS["xanomeline"], rhodanine_pains, "NOT_SMILES"]
    )
    assert len(batch) == 3
    assert batch[0].valid and batch[0].pains_free and batch[0].bbb_kelder_tpsa_pass
    assert batch[1].valid and not batch[1].pains_free
    assert not batch[2].valid and batch[2].cns_mpo_v1 == 0.0 and batch[2].cns_mpo_v2 == 0.0
    assert isinstance(batch[0].to_dict(), dict)


# ---------------------------------------------------------------------------
# 2. Scaffold-CV ADMET Oracles, Applicability Domain & Composite CNS Reward
# ---------------------------------------------------------------------------


def _make_synthetic_admet_dataset() -> tuple[list[str], list[int], list[int], list[int]]:
    """Construct a fast multi-scaffold synthetic dataset separating lipophilic vs polar chem."""
    # 30 diverse lipophilic CNS-like ring scaffolds (BBB=1, Pgp=0/1 mixed, hERG=1 for basic amines)
    cns_like = [
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
        "Clc1ccc2c(c1)OCC(CN1CCCC1)O2",
        "Cc1csc(C2CCN(C)CC2)n1",
        "COc1ccc(C2CC(=O)N(C)C2)cc1",
        "c1ccc(C2(c3ccccc3)CCNCC2)cc1",
        "Cc1cc2c(cc1Cl)OCC(N(C)C)C2",
        "Fc1cc(Cl)ccc1OCCN1CCCCC1",
        "Cc1ccc(S(=O)(=O)N2CCCCC2)cc1",
        "COc1ccc(NC(=O)c2ccncc2)cc1",
        "c1cnc2c(c1)CCC(N(C)C)C2",
        "Cc1cc(C)c2c(c1)CCN(C)C2=O",
        "Fc1ccc(-c2ccc(CN3CCCC3)cc2)cc1",
        "Clc1cccc(C(=O)N2CCN(C)CC2)c1",
        "COc1cc(F)cc(CN2CCOCC2)c1",
        "Cc1ccc(-c2nc3ccccc3[nH]2)cc1",
        "c1ccc2c(c1)C(CN1CCCC1)c1ccccc1-2",
    ]
    # 30 diverse polar poly-OH / diacid / peptide-like non-CNS scaffolds (BBB=0, hERG=0)
    polar_non_cns = [
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
        "NC(CCCN=C(N)N)C(=O)NC(CC(=O)O)C(=O)O",
        "O=C(O)c1cc2cc(O)c(O)cc2oc1=O",
        "O=C(O)C1CCCN1C(=O)C(N)CC(=O)O",
        "Oc1ccc(C(=O)NCC(=O)NC(CO)C(=O)O)cc1O",
        "O=C(O)c1ccc2c(c1)nc(O)c(C(=O)O)n2",
        "NC(=O)C(O)C(O)C(O)C(=O)NC(=O)N",
        "O=C(O)c1cc(O)nc(C(=O)O)c1O",
        "OCC1OC(Oc2ccc(O)cc2)C(O)C(O)C1O",
        "O=C(O)c1cccc(S(=O)(=O)NCC(=O)O)c1",
        "NC(CO)C(=O)NC(CO)C(=O)NC(CC(=O)O)C(=O)O",
        "O=C(O)C1=CC(O)C(O)C(O)C1",
        "Oc1cc(O)cc(C(=O)NCC(=O)O)c1",
        "O=C(O)c1ncc(O)c(C(=O)O)c1O",
        "NC(=N)NC(=O)c1cc(O)c(O)c(C(=O)O)c1",
        "O=C(O)Cc1ccc(O)c(S(=O)(=O)O)c1O",
    ]
    smiles = cns_like + polar_non_cns
    y_bbb = [1] * len(cns_like) + [0] * len(polar_non_cns)
    y_herg = [1 if i % 5 != 0 else 0 for i in range(len(cns_like))] + [0] * len(polar_non_cns)
    y_pgp = [1 if i % 2 == 0 else 0 for i in range(len(cns_like))] + [
        1 if i % 3 == 0 else 0 for i in range(len(polar_non_cns))
    ]
    return smiles, y_bbb, y_pgp, y_herg


def test_scaffold_kfold_zero_leakage_and_ece() -> None:
    """make_scaffold_kfold guarantees zero scaffold overlap across all K folds."""
    smiles, _, _, _ = _make_synthetic_admet_dataset()
    folds = make_scaffold_kfold(smiles, n_splits=5, seed=42)
    assert len(folds) == 5
    seen_val: list[int] = []
    for tr_idx, va_idx in folds:
        tr_scafs = {compute_murcko_scaffold(smiles[i]) for i in tr_idx.tolist()}
        va_scafs = {compute_murcko_scaffold(smiles[i]) for i in va_idx.tolist()}
        assert len(tr_scafs & va_scafs) == 0
        seen_val.extend(va_idx.tolist())
    assert sorted(seen_val) == list(range(len(smiles)))

    # Perfect calibration yields ECE == 0.0
    assert compute_ece([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], n_bins=10) == pytest.approx(0.0)


def test_cns_admet_suite_fit_predict_ad_serialization_and_composite_reward(tmp_path: Path) -> None:
    """Fast synthetic ADMET fit, AD domain, joblib round-trip, and composite CNS reward."""
    smiles, y_bbb, y_pgp, y_herg = _make_synthetic_admet_dataset()
    suite = CNSAdmetSuite(
        calibration_method="sigmoid",
        hgb_max_iter=20,
        rf_n_estimators=20,
        ad_threshold=0.30,
        seed=42,
    )
    m_bbb = suite.fit_endpoint("bbb", smiles, y_bbb, n_splits=3)
    m_pgp = suite.fit_endpoint("pgp", smiles, y_pgp, n_splits=3)
    m_herg = suite.fit_endpoint("herg", smiles, y_herg, n_splits=3)

    for m in (m_bbb, m_pgp, m_herg):
        assert m["scaffold_leakage_groups"] == 0
        assert 0.0 <= m["oof_brier_score"] <= 1.0
        assert 0.0 <= m["oof_ece"] <= 1.0
    assert m_bbb["oof_roc_auc"] >= 0.85
    assert m_herg["oof_roc_auc"] >= 0.70

    # Applicability Domain checks: exact training molecule -> AD = 1.0; alien perfluoro -> low AD
    in_dom_mol = smiles[0]
    alien_mol = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
    preds = suite.predict([in_dom_mol, alien_mol, "INVALID_SMILES"])
    assert preds[0].valid and preds[0].bbb_ad_tanimoto == pytest.approx(1.0, abs=1e-5)
    assert preds[0].in_domain_all
    assert preds[0].bbb_prob > 0.65
    assert preds[1].valid and preds[1].bbb_ad_tanimoto < 0.30 and not preds[1].bbb_in_domain
    assert not preds[2].valid and not preds[2].admet_pass

    # Compact joblib save/load round-trip + offline CI try_load(missing) == None
    out_file = suite.save(tmp_path / "test_admet.joblib")
    assert out_file.exists()
    reloaded = CNSAdmetSuite.load(out_file)
    preds_re = reloaded.predict([in_dom_mol, alien_mol])
    assert preds_re[0].bbb_prob == pytest.approx(preds[0].bbb_prob, abs=1e-6)
    assert preds_re[0].herg_prob == pytest.approx(preds[0].herg_prob, abs=1e-6)
    assert preds_re[1].bbb_ad_tanimoto == pytest.approx(preds[1].bbb_ad_tanimoto, abs=1e-6)
    assert CNSAdmetSuite.try_load(tmp_path / "does_not_exist.joblib") is None

    # Composite CNS reward: works both offline (admet_suite=None) and with full ADMET + 3D scores
    query = [
        REFERENCE_CNS_DRUGS["xanomeline"],
        REFERENCE_NON_CNS_DRUGS["methotrexate"],
        "O=C1NC(=S)S/C1=C/c1ccc(O)cc1",  # PAINS rhodanine
        "INVALID",
    ]
    offline_scores = compute_cns_composite_reward(query, pred_pkd=[8.5, 5.0, 7.0, 0.0])
    assert offline_scores[0].composite_reward > offline_scores[1].composite_reward
    assert offline_scores[0].composite_reward > offline_scores[2].composite_reward
    assert offline_scores[3].composite_reward == 0.0

    full_scores = compute_cns_composite_reward(
        query,
        pred_pkd=[8.5, 5.0, 7.0, 0.0],
        clip_sim=[0.7, 0.1, 0.2, 0.0],
        vina_scores=[-9.5, -5.0, -6.5, 0.0],
        admet_suite=reloaded,
        ad_damp=True,
    )
    assert all(0.0 <= s.composite_reward <= 1.0 for s in full_scores)
    assert full_scores[0].composite_reward > full_scores[1].composite_reward
    assert full_scores[0].admet is not None
    assert isinstance(full_scores[0].to_dict(), dict)

    with pytest.raises(ValueError, match="vina_scores length"):
        compute_cns_composite_reward(query, vina_scores=[-8.0])


# ---------------------------------------------------------------------------
# 3. Curated 12-Target Human Tourette & Psychiatric 3D Registry
# ---------------------------------------------------------------------------


def test_cns_12_target_registry_chembl_species_and_offline_safety(tmp_path: Path) -> None:
    """Verify all 12 human CNS targets, corrected ChEMBL IDs, SMILES, and offline CI safety."""
    expected_ids = {
        "DRD1",
        "VMAT2",
        "ADRA2A",
        "DRD2",
        "HRH3",
        "PDE10A",
        "CHRM4",
        "HTR2A",
        "SERT",
        "DRD3",
        "TAAR1",
        "DAT",
    }
    assert set(CNS_TARGET_REGISTRY.keys()) == expected_ids

    # Critical live-verified human ChEMBL IDs (protect against CHEMBL1985 / CHEMBL4908 traps!)
    assert get_cns_target("VMAT2").chembl_id == "CHEMBL1893"
    assert get_cns_target("VMAT2").chembl_id != "CHEMBL1985"
    assert get_cns_target("TAAR1").chembl_id == "CHEMBL5857"
    assert get_cns_target("TAAR1").chembl_id != "CHEMBL4908"
    assert get_cns_target("DRD1").chembl_id == "CHEMBL2056"
    assert get_cns_target("DRD2").chembl_id == "CHEMBL217"

    # Human PDE10A must use human 5SJX (NOT rat 3HR1!)
    pde10a = get_cns_target("PDE10A")
    assert pde10a.primary_pdb_id == "5SJX"
    assert "3HR1" not in pde10a.all_pdb_ids

    # Alias lookup by gene symbol, ChEMBL ID, and UniProt ID
    assert get_cns_target("SLC18A2").target_id == "VMAT2"
    assert get_cns_target("SLC6A4").target_id == "SERT"
    assert get_cns_target("SLC6A3").target_id == "DAT"
    assert get_cns_target("CHEMBL1893").target_id == "VMAT2"
    assert get_cns_target("P21728").target_id == "DRD1"

    tourette_targets = list_cns_targets("Tourette")
    assert len(tourette_targets) >= 6
    assert len(list_cns_targets()) == 12

    # Every clinical reference drug SMILES must be RDKit-valid and every target has hotspots
    for spec in CNS_TARGET_REGISTRY.values():
        assert spec.primary_anchor_residue in spec.hotspot_residues
        assert len(spec.hotspot_residues) >= 6
        assert len(spec.reference_drugs) >= 1
        for drug in spec.reference_drugs:
            mol = Chem.MolFromSmiles(drug.canonical_smiles)
            assert mol is not None, f"Invalid SMILES for {spec.target_id}/{drug.name}"

    with pytest.raises(KeyError, match="Unknown CNS target"):
        get_cns_target("NONEXISTENT_TARGET")

    # Clean offline CI behavior when data/cns_neuro is absent
    assert find_local_pocket_files("DRD1", structures_dir=tmp_path) is None
    with pytest.raises(FileNotFoundError, match="No cached 3D pocket"):
        load_cns_target_pocket("DRD1", structures_dir=tmp_path, allow_fetch=False)

    # Hotspot audit works offline on a synthetic PDB file
    syn_pdb = tmp_path / "syn_drd1.pdb"
    syn_pdb.write_text(
        "ATOM      1  CA  ASP R 103       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  OD1 ASP R 103       1.200   0.000   0.000  1.00 20.00           O\n"
        "ATOM      3  OG  SER R 198       3.000   0.000   0.000  1.00 20.00           O\n"
        "END\n",
        encoding="utf-8",
    )
    audit = audit_pocket_hotspots(syn_pdb, get_cns_target("DRD1"))
    assert "Asp103" in audit["present_hotspots"]
    assert "Ser198" in audit["present_hotspots"]
    assert audit["hotspot_coverage"] > 0.0


def test_cached_cns_structures_and_oracles_if_present() -> None:
    """Smoke-test real local CNS pockets & trained ADMET bundle when cached locally."""
    if not (DEFAULT_CNS_STRUCTURES_DIR / "clean_pockets").exists():
        pytest.skip("data/cns_neuro/structures/clean_pockets not present in CI")

    for tid in ("DRD1", "VMAT2", "ADRA2A", "PDE10A", "CHRM4"):
        spec = get_cns_target(tid)
        feat = load_cns_target_pocket(tid)
        assert feat.coords.shape[1] == 3
        assert feat.node_feats.shape == (len(feat.coords), 52)
        assert np.all(np.isfinite(feat.node_feats))
        found = find_local_pocket_files(tid)
        assert found is not None
        audit = audit_pocket_hotspots(found[0], spec, ligand_sdf=found[1])
        assert audit["anchor_in_contact"] is True

    suite = CNSAdmetSuite.try_load(DEFAULT_ADMET_MODEL_PATH)
    if suite is not None:
        preds = suite.predict(
            [REFERENCE_CNS_DRUGS["xanomeline"], REFERENCE_NON_CNS_DRUGS["ceftriaxone"]]
        )
        assert preds[0].bbb_prob > 0.70
        assert preds[1].bbb_prob < 0.50
