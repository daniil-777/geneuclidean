"""Offline CI-safe tests for Phase 1 Beyond-PoseBusters 7-axis Pose & Lead Reality Auditor."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

import geneuclidean as ge
from geneuclidean.audit import PoseRealityAuditor, audit_complex, summarize_audit_batch
from geneuclidean.evaluation.tautomer_microstate import (
    audit_tautomer_microstate_ph74,
    enumerate_tautomer_microstates_ph74,
)
from geneuclidean.evaluation.torsion_strain import (
    TORSION_BACKEND_REGISTRY,
    TorsionStrainFilter,
    TorsionStrainScorer,
    analyze_torsion_strain,
    load_torsion_library_xml,
)
from geneuclidean.protocols import MoleculeFilterProtocol, MoleculeScorerProtocol
from geneuclidean.sdk.profiles import (
    evaluate_property_profile,
    list_property_profiles,
)


def _embed_min(smiles: str, seed: int = 1, n_confs: int = 8) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = seed
    params.timeout = 3
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
    out.RemoveAllConformers()
    out.AddConformer(Chem.Conformer(mol.GetConformer(best_cid)), assignId=True)
    return Chem.RemoveHs(out)


def test_torsion_strain_detects_twisted_amide_and_loads_xml(tmp_path: Path) -> None:
    """Planar N-methylbenzamide passes cleanly; 90-degree twisted amide triggers red flag."""
    mol_ok = _embed_min("CNC(=O)c1ccccc1", seed=1)
    prof_ok = analyze_torsion_strain(mol_ok)
    assert prof_ok.n_rotatable_audited >= 1
    assert prof_ok.passed is True
    assert prof_ok.n_red == 0

    # Twist the amide O=C-N-C dihedral to 90 deg (breaking amide resonance)
    mol_bad = Chem.Mol(mol_ok)
    amide_pat = Chem.MolFromSmarts("[O]=[CX3]-[NX3]-[#6]")
    match = mol_bad.GetSubstructMatch(amide_pat)
    assert len(match) == 4
    rdMolTransforms.SetDihedralDeg(
        mol_bad.GetConformer(), int(match[0]), int(match[1]), int(match[2]), int(match[3]), 90.0
    )
    prof_bad = analyze_torsion_strain(mol_bad)
    assert prof_bad.n_red >= 1
    assert prof_bad.max_local_barrier_kcal > 2.0
    assert prof_bad.passed is False

    # External Roche/ZBH-style Torsion Library XML loader works cleanly
    xml_file = tmp_path / "custom_torsions.xml"
    xml_file.write_text(
        '<library><torsionRule name="custom_amide" smarts="[O]=[CX3]-[NX3]-[#6]">'
        '<angle value="0" tolerance1="20"/><angle value="180" tolerance1="20"/>'
        "</torsionRule></library>\n",
        encoding="utf-8",
    )
    loaded = load_torsion_library_xml(xml_file)
    assert len(loaded) == 1
    assert loaded[0][0].name == "custom_amide"

    # Protocol & registry checks
    flt = TorsionStrainFilter()
    scr = TorsionStrainScorer()
    assert isinstance(flt, MoleculeFilterProtocol)
    assert isinstance(scr, MoleculeScorerProtocol)
    assert "smarts_mmff94s" in TORSION_BACKEND_REGISTRY
    rep = ge.score_molecules(["CNC(=O)c1ccccc1"], include_admet=False, extra_scorers=[(scr, 0.2)])[
        0
    ]
    assert flt.keep(rep) is True
    assert 0.5 < rep.custom_scores["torsion_strain_plausibility"] <= 1.0


def test_tautomer_and_microstate_auditor_catches_enol_and_unprotonated_salt_bridge(
    tmp_path: Path,
) -> None:
    """Detects unstable aliphatic enol tautomer and neutral amine facing anionic ASP103."""
    # Enol of acetone vs keto acetone
    micros = enumerate_tautomer_microstates_ph74("CC(=O)C")
    assert len(micros) >= 1
    keto_audit = audit_tautomer_microstate_ph74("CC(=O)C")
    enol_audit = audit_tautomer_microstate_ph74("CC(O)=C")
    assert keto_audit.tautomer_plausible is True
    assert enol_audit.tautomer_plausible is False
    assert enol_audit.tautomer_stability_gap > 2.0

    # Neutral basic amine placed 3.0 A from anionic ASP103 carboxylate -> flags missed protonation
    mol_amine = _embed_min("CN1CCN(C(=O)c2ccccc2)CC1", seed=1)
    # Locate basic tertiary amine N (connected to methyl)
    conf = mol_amine.GetConformer()
    n_idx = [
        a.GetIdx()
        for a in mol_amine.GetAtoms()
        if a.GetAtomicNum() == 7
        and not any(
            nb.GetAtomicNum() == 6
            and any(b.GetBondType() == Chem.BondType.DOUBLE for b in nb.GetBonds())
            for nb in a.GetNeighbors()
        )
    ][0]
    pos = conf.GetAtomPosition(n_idx)
    pdb_text = (
        f"ATOM      1  OD1 ASP R 103     "
        f"{pos.x + 2.9:7.3f} {pos.y:7.3f} {pos.z:7.3f}  1.00 20.00           O\n"
        f"ATOM      2  OD2 ASP R 103     "
        f"{pos.x + 3.4:7.3f} {pos.y + 1.1:7.3f} {pos.z:7.3f}  1.00 20.00           O\n"
        "END\n"
    )
    neutral_audit = audit_tautomer_microstate_ph74(
        mol_amine, pocket_pdb=pdb_text, strict_input_charge=True
    )
    assert neutral_audit.charge_matches_ph74 is False
    assert neutral_audit.n_unprotonated_salt_bridge_misses >= 1
    assert neutral_audit.passed is False

    # Properly protonated pH 7.4 form passes cleanly
    mol_prot = ge.protonate_ph74(mol_amine, preserve_coords=True)
    prot_audit = audit_tautomer_microstate_ph74(
        mol_prot, pocket_pdb=pdb_text, strict_input_charge=True
    )
    assert prot_audit.charge_matches_ph74 is True
    assert prot_audit.n_unprotonated_salt_bridge_misses == 0
    assert prot_audit.passed is True


def test_indication_agnostic_property_profiles() -> None:
    """All 4 built-in profiles score cleanly and custom profiles register dynamically."""
    names = list_property_profiles()
    for p in ("beyond_ro5", "cns_mpo_v2", "kinase_selectivity", "oral_ro5"):
        assert p in names

    # Erlotinib-like oral kinase drug
    kinase_smi = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
    res_k = evaluate_property_profile(kinase_smi, profile="kinase_selectivity")
    res_o = evaluate_property_profile(kinase_smi, profile="oral_ro5")
    assert res_k.passed is True and 0.4 < res_k.score <= 1.0
    assert res_o.passed is True and 0.5 < res_o.score <= 1.0

    # Large cyclic peptide / bRo5-like scaffold passes beyond_ro5 even when failing oral_ro5
    bro5_smi = "CC(C)CC1NC(=O)C(C)NC(=O)C(Cc2ccccc2)NC(=O)C(CC(C)C)NC(=O)C(C(C)C)NC1=O"
    res_bro5 = evaluate_property_profile(bro5_smi, profile="beyond_ro5")
    assert res_bro5.metrics["mw"] > 500.0
    assert res_bro5.passed is True

    # Custom registration via temporary context
    with ge.PROPERTY_PROFILE_REGISTRY.temporary(
        "tiny_frag",
        lambda m: evaluate_property_profile(m, profile="oral_ro5"),
    ):
        assert "tiny_frag" in list_property_profiles()


def test_beyond_posebusters_7axis_auditor_end_to_end() -> None:
    """Audit clean chair/planar pose vs DCU coupling artifact and verify Markdown/summary export."""
    mol_clean = _embed_min("CN1CCN(C(=O)c2ccc3ccccc3n2)CC1", seed=1)
    mol_clean_prot = ge.protonate_ph74(mol_clean, preserve_coords=True)
    # Place ASP103 OD1 at 3.2 A from N+ along the axis maximizing clearance to other ligand atoms
    conf = mol_clean_prot.GetConformer()
    n_atom = [a for a in mol_clean_prot.GetAtoms() if a.GetFormalCharge() == 1][0]
    n_pos = conf.GetAtomPosition(n_atom.GetIdx())
    # Average of N+'s 3 heavy-atom bond vectors points inward; negate it for the lone-pair/H+ axis!
    bx = sum(conf.GetAtomPosition(nb.GetIdx()).x - n_pos.x for nb in n_atom.GetNeighbors())
    by = sum(conf.GetAtomPosition(nb.GetIdx()).y - n_pos.y for nb in n_atom.GetNeighbors())
    bz = sum(conf.GetAtomPosition(nb.GetIdx()).z - n_pos.z for nb in n_atom.GetNeighbors())
    norm = max(1e-6, (bx * bx + by * by + bz * bz) ** 0.5)
    ux, uy, uz = -bx / norm, -by / norm, -bz / norm
    pdb_ok = (
        f"ATOM      1  OD1 ASP R 103     "
        f"{n_pos.x + 3.2 * ux:7.3f} {n_pos.y + 3.2 * uy:7.3f} {n_pos.z + 3.2 * uz:7.3f}"
        f"  1.00 20.00           O\n"
        f"ATOM      2  CG  ASP R 103     "
        f"{n_pos.x + 4.5 * ux:7.3f} {n_pos.y + 4.5 * uy:7.3f} {n_pos.z + 4.5 * uz:7.3f}"
        f"  1.00 20.00           C\n"
        "END\n"
    )

    rep_ok = audit_complex(
        mol_clean_prot,
        pdb_ok,
        name="DRD1_1_clean",
        property_profile="cns_mpo_v2",
        canonical_anchors=("ASP103",),
        strict_input_charge=True,
    )
    assert rep_ok.passed_posebusters is True
    assert rep_ok.passed_ring_strain is True
    assert rep_ok.passed_torsion_strain is True
    assert rep_ok.passed_microstate is True
    assert rep_ok.passed_interactions is True
    assert rep_ok.passed_hygiene is True
    assert rep_ok.passed_all_7_axes is True
    assert rep_ok.reality_score > 0.75
    assert "| 2. Saturated-Ring Strain" in rep_ok.to_markdown()

    # Coupling byproduct DCU fails Axis 6 (reagent quarantine) even if geometry passes PoseBusters
    mol_dcu = _embed_min("O=C(NC1CCCCC1)NC1CCCCC1", seed=1)
    auditor = PoseRealityAuditor(property_profile="oral_ro5")
    rep_dcu = auditor.audit(mol_dcu, name="DCU_artifact")
    assert rep_dcu.passed_hygiene is False
    assert rep_dcu.hygiene_gate == 0.0
    assert rep_dcu.passed_all_7_axes is False

    summary = summarize_audit_batch([rep_ok, rep_dcu])
    assert summary["n_audited"] == 2
    assert summary["hygiene_pass_rate"] == 0.5


def test_explicit_protons_acids_and_tetrazole_tautomers() -> None:
    from geneuclidean.evaluation.protonation import protonate_mol_ph74

    cases = (
        ("CC(=O)[OH]", -1),
        ("CS(=O)(=O)[OH]", -1),
        ("C[NH2]", 1),
        ("CP(=O)(O)O", -1),
        ("CP(=O)([O-])O", -1),
        ("CP(=O)([O-])[O-]", -2),
        ("c1nnn[nH]1", -1),
        ("c1nn[nH]n1", -1),
    )
    for smiles, charge in cases:
        result = protonate_mol_ph74(Chem.MolFromSmiles(smiles))
        assert Chem.GetFormalCharge(result) == charge
        assert all(a.GetNumRadicalElectrons() == 0 for a in result.GetAtoms())
        assert Chem.MolToSmiles(protonate_mol_ph74(result)) == Chem.MolToSmiles(result)
