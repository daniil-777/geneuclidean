"""Fast offline CI-safe tests for the Tourette & Schizophrenia realistic ligand pipeline.

Verifies the complete end-to-end realistic CNS discovery workflow without requiring network
access or AutoDock Vina execution in CI:
1. 15-target Tourette Syndrome & Schizophrenia target registry coverage.
2. 1-step makeable reaction analog enumeration (:func:`enumerate_makeable_analogs`).
3. 1-cut bioisostere Avoid-ome MMP rescue (:class:`AvoidomeMMPEngine`).
4. Deterministic pH 7.4 chair-seeded conformer generation eliminating ``twist_boat`` rings.
5. 7-Axis Beyond-PoseBusters :class:`PoseRealityAuditor` on embedded 3D CNS complexes.
6. Schema, physical-audit, and 3D `.sdf` verification of
   ``results/tables/ts_scz_realistic_ligands_v53.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from rdkit import Chem

from geneuclidean.audit.pose_auditor import PoseRealityAuditor
from geneuclidean.cns.medchem_triage import triage_molecule
from geneuclidean.cns.mpo import compute_cns_mpo
from geneuclidean.evaluation.ring_strain import analyze_ring_strain
from geneuclidean.leadopt.mmp import AvoidomeMMPEngine
from geneuclidean.synthesis.reactions import (
    enumerate_makeable_analogs,
    match_one_step_retrosynthesis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCRIPT = REPO_ROOT / "scripts" / "run_ts_scz_realistic_campaign.py"
CAMPAIGN_JSON = REPO_ROOT / "results" / "tables" / "ts_scz_realistic_ligands_v53.json"


def _load_campaign_module() -> ModuleType:
    """Import scripts/run_ts_scz_realistic_campaign.py cleanly for dataclass resolution."""
    mod_name = "run_ts_scz_realistic_campaign"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, CAMPAIGN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_minimal_aminergic_pocket_pdb(cation_xyz: np.ndarray) -> str:
    """Build a minimal synthetic Asp103 pocket PDB cleanly salt-bridging ``cation_xyz``."""
    # Place anionic ASP103 OD1 at +3.35 A along +X from the extreme +X protonated [NH+] atom
    # (well within 4.2 A salt-bridge & 3.5 A H-bond cutoffs, with zero heavy-atom clashes).
    cx, cy, cz = float(cation_xyz[0]), float(cation_xyz[1]), float(cation_xyz[2])
    atoms = [
        ("N  ", cx + 6.4, cy + 0.0, cz + 0.0, "N"),
        ("CA ", cx + 5.6, cy + 1.2, cz + 0.0, "C"),
        ("CB ", cx + 4.8, cy + 1.2, cz + 1.2, "C"),
        ("CG ", cx + 4.0, cy + 0.2, cz + 0.8, "C"),
        ("OD1", cx + 3.35, cy + 0.0, cz + 0.0, "O"),
        ("OD2", cx + 3.95, cy - 0.7, cz + 1.5, "O"),
        ("C  ", cx + 6.4, cy + 2.5, cz + 0.0, "C"),
        ("O  ", cx + 7.6, cy + 2.5, cz + 0.0, "O"),
    ]
    lines = [
        f"ATOM  {i:5d}  {name} ASP A 103    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {el}  "
        for i, (name, x, y, z, el) in enumerate(atoms, start=1)
    ]
    return "\n".join(lines) + "\n"


def test_campaign_target_coverage_tourette_and_schizophrenia() -> None:
    """Campaign covers >= 12 distinct human targets spanning both Tourette and Schizophrenia."""
    mod = _load_campaign_module()
    targets = mod.CAMPAIGN_TARGETS
    ids = {t.target_id for t in targets}
    assert len(ids) == len(targets) >= 12

    ts_ids = {t.target_id for t in targets if "tourette" in t.disorders}
    scz_ids = {t.target_id for t in targets if "schizophrenia" in t.disorders}

    expected_ts = {"DRD1", "DRD2", "VMAT2", "HRH3", "PDE10A", "GABRA5", "GAT1", "ADORA2A", "GRM5"}
    expected_scz = {
        "DRD2",
        "DRD3",
        "HTR2A",
        "CHRM1",
        "CHRM4",
        "TAAR1",
        "GPR52",
        "PDE10A",
        "ADORA2A",
    }
    assert expected_ts.issubset(ts_ids)
    assert expected_scz.issubset(scz_ids)

    for cfg in targets:
        assert cfg.pharmacophore_class in mod.MAKEABLE_LIBRARY_BY_CLASS
        assert len(cfg.canonical_anchors) >= 1
        assert len(cfg.pdb_id) == 4


def test_makeable_enumeration_produces_realistic_cns_ligands() -> None:
    """1-step reaction enumeration yields triage-clean, makeable, high-MPO TS & SCZ analogs."""
    # Tourette DRD1 / aminergic indazole core + CNS cyclic amines
    ts_analogs = enumerate_makeable_analogs(
        "O=C(O)c1ccc2[nH]ncc2c1",
        ("CN1CCNCC1", "C1COCCN1", "FC1CCNCC1", "CN1CCC(N)CC1"),
        reactions=("amide_coupling_acid_amine",),
        max_products=20,
        triage_filter=True,
    )
    assert len(ts_analogs) >= 4

    # Schizophrenia PDE10A / ADORA2A non-basic heteroaromatic core
    scz_analogs = enumerate_makeable_analogs(
        "Nc1nc(Cl)cc(C#N)n1",
        ("C1COCCN1", "FC1CCNCC1", "NCC1CC1"),
        reactions=("snar_heteroaryl_fluoride",),
        max_products=20,
        triage_filter=True,
    )
    assert len(scz_analogs) >= 2

    for cand in [*ts_analogs, *scz_analogs]:
        triage = triage_molecule(cand.product_smiles)
        mpo = compute_cns_mpo(cand.product_smiles)
        assert triage.pains_free and triage.brenk_free and triage.triage_pass
        assert mpo.valid and mpo.sa_score <= 4.0 and mpo.cns_mpo_v2 >= 4.2
        assert cand.reaction_name in ("amide_coupling_acid_amine", "snar_heteroaryl_fluoride")

    # Confirm 1-step retrosynthesis matches the amide products
    retro = match_one_step_retrosynthesis(ts_analogs[0].product_smiles)
    assert retro["valid"] and retro["is_solved"] and "amide_bond" in retro["matched_disconnections"]


def test_avoidome_mmp_rescue_improves_cns_profile() -> None:
    """AvoidomeMMPEngine proposes clean 1-cut bioisostere rescues improving CNS MPO v2."""
    engine = AvoidomeMMPEngine(admet_suite=None, property_profile="cns_mpo_v2")
    # Haloperidol (FDA-approved for Tourette & Schizophrenia; high cLogP / hERG parent)
    haloperidol = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    proposals = engine.propose_rescues(haloperidol, max_proposals=15, triage_filter=True)
    assert len(proposals) >= 3
    assert any(p.delta_mpo_v2 > 0.0 for p in proposals)
    for prop in proposals[:5]:
        triage = triage_molecule(prop.analog_smiles)
        assert triage.triage_pass and triage.pains_free and triage.brenk_free
        assert prop.cns_mpo_v2 >= 4.0


def test_chair_conformer_seeding_eliminates_twist_boat_rings() -> None:
    """Chair-seeded pH 7.4 conformer generator rejects twist-boat/boat 6-membered rings."""
    mod = _load_campaign_module()
    test_ligands = (
        ("CN1CCN(C(=O)c2ccc3[nH]ncc3c2)CC1", 1),
        ("Fc1ccc(CN2CCC(F)CC2)cc1", 1),
        ("O=C(c1cnc2ccccc2c1)N1CCOCC1", 0),
    )
    for smi, expected_q in test_ligands:
        seeded = mod.seed_chair_conformers_ph74(
            smi,
            seed=42,
            num_confs=8,
            max_chair_confs=2,
            ring_strain_cutoff_kcal=2.0,
        )
        assert seeded.net_charge_ph74 == expected_q
        assert len(seeded.conformers) >= 1
        for conf_mol, _mmff_e, _rs in seeded.conformers:
            prof = analyze_ring_strain(
                conf_mol,
                strain_threshold_kcal=3.0,
                n_ref_confs=8,
                include_full_ligand_strain=False,
                seed=42,
            )
            assert prof.passed, f"Ring strain failed for {smi}: {prof.max_ring_strain_kcal}"
            assert prof.max_ring_strain_kcal <= 3.0
            for ring in prof.rings:
                if ring.ring_size == 6:
                    assert ring.pucker_class not in ("boat", "twist_boat")


def test_seven_axis_pose_reality_auditor_end_to_end_offline() -> None:
    """Offline pipeline: makeable -> MMP rescue -> chair seed -> 7-axis PoseRealityAuditor."""
    mod = _load_campaign_module()
    # 1. Generate a 1-step makeable aminergic lead
    analogs = enumerate_makeable_analogs(
        "O=C(O)c1ccc2oc(C)nc2c1",
        ("CN1CCNCC1",),
        reactions=("amide_coupling_acid_amine",),
        max_products=5,
        triage_filter=True,
    )
    assert analogs
    parent_smi = analogs[0].product_smiles

    # 2. Apply 1-cut bioisostere MMP rescue
    mmp = AvoidomeMMPEngine(admet_suite=None, property_profile="cns_mpo_v2")
    rescues = mmp.propose_rescues(parent_smi, max_proposals=5, triage_filter=True)
    chosen_smi = rescues[0].analog_smiles if rescues else parent_smi

    # 3. Chair-seeded pH 7.4 conformer generation
    seeded = mod.seed_chair_conformers_ph74(chosen_smi, seed=42, num_confs=8, max_chair_confs=1)
    assert seeded.conformers
    mol_3d = seeded.conformers[0][0]

    # Orient so the [NH+] -> H vector (or outer ray from centroid) points strictly along +X
    # and ensure ASP103 sits cleanly outside the ligand's heavy-atom envelope.
    conf = mol_3d.GetConformer()
    pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    cation_idxs = [
        a.GetIdx() for a in mol_3d.GetAtoms() if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0
    ]
    assert cation_idxs, "Expected protonated basic amine at pH 7.4"
    c_idx = cation_idxs[0]
    heavy_idxs = [a.GetIdx() for a in mol_3d.GetAtoms() if a.GetAtomicNum() > 1]
    nbr_h = [
        n.GetIdx() for n in mol_3d.GetAtomWithIdx(c_idx).GetNeighbors() if n.GetAtomicNum() == 1
    ]
    centroid = pos[heavy_idxs].mean(axis=0)
    axis = (pos[nbr_h[0]] - pos[c_idx]) if nbr_h else (pos[c_idx] - centroid)
    norm = float(np.linalg.norm(axis))
    if norm > 1e-6:
        u = axis / norm
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v = np.cross(u, target)
        s = float(np.linalg.norm(v))
        c = float(np.dot(u, target))
        if s > 1e-6:
            vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            rot = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        elif c < 0.0:
            rot = np.diag([-1.0, -1.0, 1.0])
        else:
            rot = np.eye(3)
        pos = (pos - centroid) @ rot.T
        for i in range(mol_3d.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i].tolist())

    pocket_pdb_text = _make_minimal_aminergic_pocket_pdb(pos[c_idx])

    # 4. Run full 7-Axis PoseRealityAuditor (offline without joblib ADMET)
    auditor = PoseRealityAuditor(
        ring_strain_threshold_kcal=3.0,
        ring_n_ref_confs=8,
        torsion_barrier_threshold_kcal=2.5,
        torsion_max_red=0,
        strict_input_charge=True,
        property_profile="cns_mpo_v2",
        admet_suite=None,
        seed=42,
    )
    report = auditor.audit(
        mol_3d,
        pocket_pdb=pocket_pdb_text,
        name="TS_SCZ_offline_unit_lead",
        canonical_anchors=("ASP103",),
    )
    assert report.passed_posebusters
    assert report.passed_ring_strain
    assert report.passed_torsion_strain
    assert report.passed_microstate
    assert report.passed_interactions
    assert report.passed_hygiene
    assert report.passed_physical_beyond_pb
    assert report.passed_all_7_axes
    assert report.interactions_summary["canonical_anchor_engaged"] is True
    assert report.reality_score >= 0.75

    # 5. Negative control: coupling byproduct DCU must fail MedChem hygiene axis
    dcu_seed = mod.seed_chair_conformers_ph74("O=C(NC1CCCCC1)NC1CCCCC1", seed=42, num_confs=8)
    dcu_rep = auditor.audit(dcu_seed.conformers[0][0], name="DCU_negative_control")
    assert not dcu_rep.passed_hygiene
    assert not dcu_rep.passed_all_7_axes


def test_ts_scz_campaign_json_and_sdf_artifacts_schema() -> None:
    """Validate schema, physical pass rates, metadata, and 3D SDF poses of v5.3 campaign output."""
    if not CAMPAIGN_JSON.is_file():
        pytest.skip(f"Campaign artifact not present at {CAMPAIGN_JSON}")

    data = json.loads(CAMPAIGN_JSON.read_text(encoding="utf-8"))
    assert data["schema_version"] == "geneuclidean-v5.3-ts-scz-realistic-campaign"
    assert data["status"] == "IN_SILICO_COMPUTATIONAL_HYPOTHESES"
    assert "in silico" in data["disclaimer"].lower()

    cov = data["disorder_coverage"]
    assert cov["n_distinct_human_targets"] >= 12
    assert cov["n_tourette_targets"] >= 9
    assert cov["n_schizophrenia_targets"] >= 9
    assert set(cov["tourette_targets"]).issuperset(
        {"DRD1", "DRD2", "VMAT2", "HRH3", "PDE10A", "GABRA5", "GAT1", "ADORA2A", "GRM5"}
    )
    assert set(cov["schizophrenia_targets"]).issuperset(
        {"DRD2", "DRD3", "HTR2A", "CHRM1", "CHRM4", "TAAR1", "GPR52", "PDE10A", "ADORA2A"}
    )

    # Verify Before vs. After Beyond-PoseBusters pass rate beats v5.0 baseline (60.7%)
    bva = data["before_vs_after_beyond_posebusters"]
    raw_top1_rate = float(bva["after_v53_raw_top1_all_docked"]["beyond_pb_physical_pass_rate"])
    mode_aud_rate = float(bva["after_v53_mode_audited_all_docked"]["beyond_pb_physical_pass_rate"])
    flagship_all7_rate = float(bva["after_v53_flagship_leads"]["all_7_axes_pass_rate"])
    assert raw_top1_rate > 0.6071, f"Raw Top-1 Beyond-PB {raw_top1_rate:.3f} must beat 60.7%"
    assert mode_aud_rate > raw_top1_rate or mode_aud_rate >= 0.85
    assert flagship_all7_rate == 1.0, (
        f"Expected 100% 7-axis flagship leads, got {flagship_all7_rate}"
    )

    leads = data["flagship_leads"]
    assert len(leads) >= cov["n_distinct_human_targets"]
    targets_covered = {lead["target_id"] for lead in leads}
    assert len(targets_covered) == cov["n_distinct_human_targets"]

    required_keys = {
        "lead_id",
        "target_id",
        "disorders",
        "pdb_id",
        "smiles",
        "smiles_ph74",
        "vina_dg_kcal_mol",
        "cns_mpo_v2",
        "p_herg",
        "p_5ht2b",
        "p_mdr1",
        "sa_score",
        "retrosynthesis_1step",
        "audit_7axis",
        "ring_strain_kcal",
        "torsion_verdict",
        "plif_3d",
        "chembl_novelty",
        "sdf_path",
        "status",
        "disclaimer",
    }

    for lead in leads:
        assert required_keys.issubset(lead.keys())
        assert lead["status"] == "IN_SILICO_COMPUTATIONAL_HYPOTHESIS"
        assert "in silico" in lead["disclaimer"].lower()
        assert lead["pains_count"] == 0
        assert lead["brenk_count"] == 0
        assert lead["sa_score"] <= 4.0
        assert lead["cns_mpo_v2"] >= 4.2
        assert lead["vina_dg_kcal_mol"] < -5.0
        assert lead["ring_strain_kcal"] <= 3.0
        assert lead["torsion_verdict"]["passed"] is True
        assert lead["torsion_verdict"]["n_red"] == 0
        assert lead["audit_7axis"]["passed_physical_beyond_pb"] is True
        assert lead["audit_7axis"]["passed_all_7_axes"] is True
        assert lead["plif_3d"]["total_interactions"] > 0

        # Verify saved 3D SDF pose exists and parses cleanly
        sdf_file = REPO_ROOT / lead["sdf_path"]
        assert sdf_file.is_file(), f"Missing flagship SDF: {sdf_file}"
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=True, removeHs=False)
        assert len(suppl) == 1 and suppl[0] is not None
        assert suppl[0].GetNumConformers() >= 1
        assert suppl[0].GetProp("STATUS") == "IN_SILICO_COMPUTATIONAL_HYPOTHESIS"
