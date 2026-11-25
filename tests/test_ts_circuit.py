"""100% Offline CI-safe unit tests for Tourette CSTC Basal Ganglia & GABAergic Circuit module.

Verifies:
1. 100% backward compatibility of ``targets.py`` registries (12 core v3.1, 12 v4 expansion,
   24 Tri-Disorder, 7 counter-targets, 31 ALL_CNS_TARGET_REGISTRY) while seamlessly resolving
   the 4 newly curated Tourette CSTC / GABAergic targets (``GABRA5``, ``GAT1``, ``ADORA2A``,
   ``GRM5``) via aliases, gene symbols, ChEMBL IDs, UniProt IDs, and ``include_ts_circuit=True``.
2. Complete 13-target Tourette CSTC circuit audit integrity (9 existing + 4 new expansion),
   including honest clinical grading and verified conformational polarity.
3. Multi-chain interface pocket extractor (``extract_multichain_interface_pocket``) on a
   synthetic multi-copy heteropentameric GABA-A-like PDB: verifies disambiguation between
   two copies of the same ligand (canonical interface vs secondary copy), retention of both
   ``alpha+`` and ``gamma-`` interface chains, ligand/water stripping, and cofactor handling.
4. Mechanism-aware Tourette circuit scoring (``score_tourette_circuit_candidate``): guarantees
   hard veto (0.0) on DCU coupling reagent & PAINS rhodanine, mechanism-specific flags for
   GABA-A alpha1/alpha5, GAT1 zwitterions, ADORA2A ribose vs non-ribose, and GRM5 alkynes.
5. Local smoke test (automatically skipped in offline CI) verifying all 4 curated human
   pockets featurize to ``(N, 52)`` with 0 NaNs and canonical anchors within 4.5 A.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from geneuclidean.cns.targets import (
    ALL_CNS_TARGET_REGISTRY,
    CNS_TARGET_REGISTRY,
    TRI_DISORDER_TARGET_REGISTRY,
    V4_COUNTER_TARGET_REGISTRY,
    V4_EXPANSION_TARGET_REGISTRY,
    audit_pocket_hotspots,
    get_cns_target,
    list_cns_targets,
    list_tri_disorder_targets,
)
from geneuclidean.cns.tourette_circuit import (
    DEFAULT_TS_CIRCUIT_STRUCTURES_DIR,
    TOURETTE_CIRCUIT_EXPANSION_REGISTRY,
    TOURETTE_CSTC_CIRCUIT_AUDIT,
    extract_multichain_interface_pocket,
    find_local_ts_circuit_pocket_files,
    get_cstc_circuit_audit,
    get_tourette_circuit_target,
    list_tourette_circuit_targets,
    load_tourette_circuit_pocket,
    score_tourette_circuit_candidate,
)
from geneuclidean.data.featurizer import POCKET_NODE_DIM

# ---------------------------------------------------------------------------
# 1. Registry Counts, Backward Compatibility & Tourette CSTC Integration
# ---------------------------------------------------------------------------


def test_targets_backward_compatibility_and_ts_circuit_hooks() -> None:
    """Existing v3.1/v4.0 registry sizes remain exact while new CSTC targets resolve cleanly."""
    # Exact backward-compatible counts preserved 100%
    assert len(CNS_TARGET_REGISTRY) == 12
    assert len(list_cns_targets()) == 12
    assert len(V4_EXPANSION_TARGET_REGISTRY) == 12
    assert len(TRI_DISORDER_TARGET_REGISTRY) == 24
    assert len(V4_COUNTER_TARGET_REGISTRY) == 7
    assert len(ALL_CNS_TARGET_REGISTRY) == 31
    assert len(list_cns_targets(include_v4=True)) == 24
    assert len(list_tri_disorder_targets()) == 24
    assert len(list_tri_disorder_targets(include_counters=True)) == 31

    # Opt-in Tourette CSTC expansion hook adds the 4 new verified targets cleanly
    assert len(TOURETTE_CIRCUIT_EXPANSION_REGISTRY) == 4
    assert set(TOURETTE_CIRCUIT_EXPANSION_REGISTRY.keys()) == {
        "GABRA5",
        "GAT1",
        "ADORA2A",
        "GRM5",
    }
    assert len(list_cns_targets(include_ts_circuit=True)) == 16
    assert len(list_cns_targets(include_v4=True, include_ts_circuit=True)) == 28
    assert len(list_tourette_circuit_targets()) == 4
    assert len(list_tourette_circuit_targets(cstc_node="Striatal GABAergic FSI / Tonic Brake")) == 2

    # Seamless alias, gene symbol, UniProt, and ChEMBL ID resolution through both APIs
    assert get_cns_target("GABRA5").primary_pdb_id == "8BHG"
    assert get_cns_target("GABA-A5").target_id == "GABRA5"
    assert get_cns_target("CHEMBL2094122").target_id == "GABRA5"
    assert get_cns_target("SLC6A1").target_id == "GAT1"
    assert get_cns_target("CHEMBL1903").target_id == "GAT1"
    assert get_cns_target("P30531").target_id == "GAT1"
    assert get_cns_target("A2AR").target_id == "ADORA2A"
    assert get_cns_target("CHEMBL251").target_id == "ADORA2A"
    assert get_cns_target("mGluR5").target_id == "GRM5"
    assert get_cns_target("CHEMBL3227").target_id == "GRM5"

    # Active agonist MoA precision on ADORA2A (OPPOSITE of Parkinson's antagonist!)
    a2a = get_tourette_circuit_target("ADORA2A")
    assert a2a.moa_class == "AGONIST"
    assert a2a.primary_pdb_id == "9EE8"
    assert "5NM4" not in a2a.all_pdb_ids and "8GNG" not in a2a.all_pdb_ids

    # GRM5 canonical UniProt numbering primary (4OO9) + T4L offset aliases for 6FFI/6FFH
    grm5 = get_tourette_circuit_target("GRM5")
    assert grm5.moa_class == "NAM"
    assert grm5.primary_pdb_id == "4OO9"
    assert "Ser969" in grm5.hotspot_isoform_aliases
    assert "Ser1809" in grm5.hotspot_isoform_aliases

    # Every new target has valid clinical reference SMILES and primary anchor in hotspots
    for spec in TOURETTE_CIRCUIT_EXPANSION_REGISTRY.values():
        assert spec.primary_anchor_residue in spec.hotspot_residues
        assert len(spec.hotspot_residues) >= 8
        assert len(spec.reference_drugs) >= 3
        for drug in spec.reference_drugs:
            mol = Chem.MolFromSmiles(drug.canonical_smiles)
            assert mol is not None, f"Invalid SMILES for {spec.target_id}/{drug.name}"

    with pytest.raises(KeyError, match="Unknown Tourette CSTC expansion target"):
        get_tourette_circuit_target("NONEXISTENT_CSTC")


def test_13_target_cstc_circuit_audit_completeness_and_honesty() -> None:
    """All 9 existing + 4 new Tourette CSTC targets are audited with verified metadata."""
    audit = get_cstc_circuit_audit()
    assert len(audit) == 13
    existing_9 = {"DRD1", "DRD2", "VMAT2", "ADRA2A", "HRH3", "PDE10A", "MGLL", "PDE1B", "GPR52"}
    new_4 = {"GABRA5", "GAT1", "ADORA2A", "GRM5"}
    assert set(audit.keys()) == existing_9 | new_4

    for tid, entry in audit.items():
        d = entry.to_dict()
        assert d["target_id"] == tid
        assert entry.conformational_state_verified is True
        assert entry.primary_pdb_resolution_angstrom > 0.0
        assert entry.is_new_expansion is (tid in new_4)

    # Verify key honest audit caveats are explicitly captured
    assert "9URL" in TOURETTE_CSTC_CIRCUIT_AUDIT["DRD1"].audit_notes
    assert TOURETTE_CSTC_CIRCUIT_AUDIT["VMAT2"].clinical_evidence_grade == "Phase 2/3 Failed in TS"
    assert "AZD5213" in TOURETTE_CSTC_CIRCUIT_AUDIT["HRH3"].audit_notes
    assert TOURETTE_CSTC_CIRCUIT_AUDIT["GABRA5"].multichain_interface == ("C", "B")

    single = get_cstc_circuit_audit("mGluR5")
    assert "GRM5" in single
    with pytest.raises(KeyError, match="Unknown Tourette CSTC audit target"):
        get_cstc_circuit_audit("UNKNOWN_AUDIT_TARGET")


# ---------------------------------------------------------------------------
# 2. Multi-Chain Interface Pocket Extractor (Synthetic Heteromeric GABA-A)
# ---------------------------------------------------------------------------


def test_extract_multichain_interface_pocket_disambiguates_copies_offline(tmp_path: Path) -> None:
    """Multi-chain extractor isolates exact copy B:QMU502 across chains C+B and strips extras."""
    # Construct synthetic oligomeric PDB with TWO copies of QMU:
    # - Copy 1 (A:QMU501 at x=50): decoy transmembrane site touching only chain A
    # - Copy 2 (B:QMU502 at x=0): canonical alpha5+(C)/gamma2-(B) interface touching C & B
    lines = [
        # Chain A (decoy far away at x=50)
        "ATOM      1  CA  MET A 261      50.000   0.000   0.000  1.00 20.00           C",
        "HETATM    2  C1  QMU A 501      52.000   0.000   0.000  1.00 20.00           C",
        # Chain C (alpha5+ interface at x=-2.5): His105 + Ser209
        "ATOM      3  CA  HIS C 105      -2.500   0.000   0.000  1.00 20.00           C",
        "ATOM      4  NE2 HIS C 105      -1.800   0.500   0.000  1.00 20.00           N",
        "ATOM      5  OG  SER C 209      -2.000  -1.500   0.000  1.00 20.00           O",
        # Far residue on Chain C (> 14 A away, must be excluded)
        "ATOM      6  CA  ALA C 300     -30.000   0.000   0.000  1.00 20.00           C",
        # Chain B (gamma2- interface at x=+2.5): Phe77 + Thr142
        "ATOM      7  CA  PHE B  77       2.500   0.000   0.000  1.00 20.00           C",
        "ATOM      8  CZ  PHE B  77       1.800   0.200   0.000  1.00 20.00           C",
        "ATOM      9  OG1 THR B 142       2.200  -1.200   0.000  1.00 20.00           O",
        # Canonical ligand copy B:QMU502 at origin + nearby water & detergent (must be stripped)
        "HETATM   10  C1  QMU B 502       0.000   0.000   0.000  1.00 20.00           C",
        "HETATM   11  N1  QMU B 502       0.600   0.000   0.000  1.00 20.00           N",
        "HETATM   12  O   HOH B 901       0.000   2.000   0.000  1.00 20.00           O",
        "HETATM   13  C1  DMU B 902       0.000  -2.500   0.000  1.00 20.00           C",
        "END",
    ]
    src_pdb = tmp_path / "syn_gabra5_8bhg.pdb"
    src_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_pdb = tmp_path / "out_gabra5_pocket.pdb"

    res = extract_multichain_interface_pocket(
        src_pdb,
        out_pdb,
        ligand_resname="QMU",
        ligand_chain="B",
        ligand_resseq=502,
        keep_chains=None,  # auto-discover both interacting chains!
        pocket_cutoff=14.0,
    )
    assert out_pdb.exists()
    assert res.is_multichain_interface is True
    assert res.retained_chains == ("B", "C")
    assert "A" not in res.residues_per_chain
    assert res.residues_per_chain == {"B": 2, "C": 2}
    assert "His105" in res.interface_contacts_4p5a["C"]
    assert "Phe77" in res.interface_contacts_4p5a["B"]

    # Written pocket PDB must be strictly ligand-, water-, and detergent-free
    out_text = out_pdb.read_text(encoding="utf-8")
    assert "QMU" not in out_text
    assert "HOH" not in out_text
    assert "DMU" not in out_text
    assert "ALA C 300" not in out_text

    # Audit hotspots directly on extracted multi-chain pocket
    h_audit = audit_pocket_hotspots(out_pdb, get_tourette_circuit_target("GABRA5"))
    assert {"His105", "Ser209", "Phe77", "Thr142"}.issubset(set(h_audit["present_hotspots"]))

    # Missing ligand raises ValueError cleanly
    with pytest.raises(ValueError, match="not found"):
        extract_multichain_interface_pocket(src_pdb, tmp_path / "x.pdb", ligand_resname="ZZZ")


# ---------------------------------------------------------------------------
# 3. Mechanism-Aware Tourette Circuit Scoring & Hard Hygiene Veto
# ---------------------------------------------------------------------------


def test_mechanism_aware_tourette_circuit_scoring_and_dcu_pains_veto() -> None:
    """DCU and PAINS are strictly vetoed to 0.0 while clean mechanism leads score strongly."""
    dcu = "O=C(NC1CCCCC1)NC1CCCCC1"
    rhodanine = "O=C1NC(=S)S/C1=C/c1ccc(O)cc1"

    # Hard veto to 0.0 even with hypothetical strong Vina affinity!
    s_dcu = score_tourette_circuit_candidate(
        dcu, "GABRA5", vina_dg_kcal=-10.0, anchor_engaged=True, posebusters_pass=True
    )
    assert s_dcu.valid and s_dcu.hygiene_gate == 0.0 and s_dcu.circuit_readiness_score == 0.0

    s_pains = score_tourette_circuit_candidate(
        rhodanine, "GRM5", vina_dg_kcal=-10.0, anchor_engaged=True, posebusters_pass=True
    )
    assert s_pains.valid and s_pains.hygiene_gate == 0.0 and s_pains.circuit_readiness_score == 0.0

    # Invalid SMILES returns valid=False and score 0.0
    s_inv = score_tourette_circuit_candidate("INVALID_SMILES", "GAT1")
    assert not s_inv.valid and s_inv.circuit_readiness_score == 0.0

    # Clean clinical drugs score strongly with accurate mechanism-specific flags
    bretazenil = get_tourette_circuit_target("GABRA5").reference_drugs[0].canonical_smiles
    s_bret = score_tourette_circuit_candidate(
        bretazenil, "GABRA5", vina_dg_kcal=-9.5, anchor_engaged=True, posebusters_pass=True
    )
    assert s_bret.valid and s_bret.circuit_readiness_score >= 0.70
    assert "requires_alpha1_vs_alpha5_electrophysiology_selectivity_check" in s_bret.mechanism_flags

    # GAT1 zwitterionic calibration rewards Tiagabine without over-penalizing its COOH headgroup
    tiagabine = get_tourette_circuit_target("GAT1").reference_drugs[0].canonical_smiles
    s_tgi = score_tourette_circuit_candidate(
        tiagabine, "GAT1", vina_dg_kcal=-8.8, anchor_engaged=True, posebusters_pass=True
    )
    assert s_tgi.valid and s_tgi.mechanism_polarity_fit >= 0.85
    assert "zwitterionic_or_acidic_gaba_mimetic_headgroup" in s_tgi.mechanism_flags

    # ADORA2A flags polar ribose vs CNS-favorable non-ribose partial agonist LUF5833
    adenosine = get_tourette_circuit_target("ADORA2A").reference_drugs[0].canonical_smiles
    luf5833 = get_tourette_circuit_target("ADORA2A").reference_drugs[3].canonical_smiles
    s_adn = score_tourette_circuit_candidate(adenosine, "ADORA2A", vina_dg_kcal=-7.5)
    s_luf = score_tourette_circuit_candidate(luf5833, "ADORA2A", vina_dg_kcal=-8.5)
    assert "polar_ribose_agonist_low_bbb_and_peripheral_hypotension_risk" in s_adn.mechanism_flags
    assert "non_ribose_cns_scaffold_requires_functional_gs_agonism_assay" in s_luf.mechanism_flags
    assert s_luf.circuit_readiness_score > s_adn.circuit_readiness_score

    # GRM5 flags diaryl alkyne (Mavoglurant) vs non-alkyne (Fenobam)
    mavoglurant = get_tourette_circuit_target("GRM5").reference_drugs[0].canonical_smiles
    fenobam = get_tourette_circuit_target("GRM5").reference_drugs[3].canonical_smiles
    s_mav = score_tourette_circuit_candidate(mavoglurant, "GRM5", vina_dg_kcal=-9.0)
    s_fen = score_tourette_circuit_candidate(fenobam, "GRM5", vina_dg_kcal=-8.8)
    assert (
        "diaryl_alkyne_mglur5_nam_requires_gsh_reactive_metabolite_screen" in s_mav.mechanism_flags
    )
    assert "diaryl_alkyne_mglur5_nam_requires_gsh_reactive_metabolite_screen" not in (
        s_fen.mechanism_flags
    )
    assert isinstance(s_fen.to_dict(), dict)


# ---------------------------------------------------------------------------
# 4. Offline Missing-File Safety & Optional Local Curated Pocket Smoke Test
# ---------------------------------------------------------------------------


def test_offline_missing_pocket_behavior(tmp_path: Path) -> None:
    """Missing local structures return None / raise FileNotFoundError cleanly in offline CI."""
    assert find_local_ts_circuit_pocket_files("GABRA5", structures_dir=tmp_path) is None
    with pytest.raises(FileNotFoundError, match="No curated 3D pocket"):
        load_tourette_circuit_pocket("GABRA5", structures_dir=tmp_path)


def test_curated_ts_circuit_4pockets_if_present() -> None:
    """Smoke-test all 4 newly curated human GABA/CSTC pockets when cached locally."""
    if not (DEFAULT_TS_CIRCUIT_STRUCTURES_DIR / "clean_pockets").exists():
        pytest.skip("data/cns_neuro.nosync/ts_circuit_expansion/structures not present in CI")

    for tid in ("GABRA5", "GAT1", "ADORA2A", "GRM5"):
        spec = get_tourette_circuit_target(tid)
        feat = load_tourette_circuit_pocket(tid)
        assert feat.coords.shape[1] == 3
        assert feat.node_feats.shape == (len(feat.coords), POCKET_NODE_DIM)
        assert np.all(np.isfinite(feat.node_feats))
        assert np.all(np.isfinite(feat.coords))
        found = find_local_ts_circuit_pocket_files(tid)
        assert found is not None and found[1] is not None
        h_audit = audit_pocket_hotspots(found[0], spec, ligand_sdf=found[1], contact_cutoff=4.5)
        assert h_audit["anchor_in_contact"] is True, (
            f"Primary anchor {spec.primary_anchor_residue} not within 4.5 A for {tid}: {h_audit}"
        )
