"""Offline CI-safe tests for Phase 3 synthesis, MMP lead-opt, OpenFE export & Boltz-2 adapter."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from geneuclidean.interop import (
    audit_cofolded_complex,
    export_openfe_rbfe_network,
    load_cofolded_complex,
)
from geneuclidean.leadopt import AvoidomeMMPEngine, analyze_rgroup_sar
from geneuclidean.synthesis import (
    RETROSYNTHESIS_REGISTRY,
    ROBUST_MEDCHEM_REACTIONS,
    enumerate_makeable_analogs,
    match_one_step_retrosynthesis,
)


def test_reaction_grounded_makeable_synthesis_and_retrosynthesis() -> None:
    """Amide coupling produces sanitized makeable products with explicit synthetic provenance."""
    assert len(ROBUST_MEDCHEM_REACTIONS) >= 15
    assert "template_1step" in RETROSYNTHESIS_REGISTRY

    # Carboxylic acid core + diverse secondary/primary amine building blocks -> clean amides
    core_acid = "O=C(O)c1ccc2ccccc2n1"
    bbs = ["CN1CCNCC1", "C1COCCN1", "NCC2CC2", "Cc1ccc(B(O)O)cc1"]
    cands = enumerate_makeable_analogs(
        core_acid, bbs, reactions=("amide_coupling_acid_amine",), triage_filter=True
    )
    assert len(cands) >= 3
    prods = {c.product_smiles for c in cands}
    # Flagship DRD1_1 piperazide is synthesized in 1 step from quinoline-2-acid + N-Me-piperazine!
    assert "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1" in prods
    for c in cands:
        assert c.reaction_name == "amide_coupling_acid_amine"
        assert c.triage_passed is True
        retro = match_one_step_retrosynthesis(c.product_smiles)
        assert retro["is_solved"] is True
        assert "amide_bond" in retro["matched_disconnections"]


def test_rgroup_sar_cliffs_and_avoidome_mmp_rescues() -> None:
    """R-group decomposition detects single-substituent cliffs & MMP engine proposes rescues."""
    series = [
        ("CN1CCN(C(=O)c2ccc(F)cc2)CC1", 8.6),
        ("CN1CCN(C(=O)c2ccc(Cl)cc2)CC1", 8.4),
        ("CN1CCN(C(=O)c2ccc(OC)cc2)CC1", 6.2),
        ("CN1CCN(C(=O)c2ccccc2)CC1", 6.5),
    ]
    sar = analyze_rgroup_sar(
        [s for s, _ in series],
        [v for _, v in series],
        core_smarts="CN1CCN(C(=O)c2ccccc2)CC1",
        cliff_threshold=1.5,
    )
    assert sar.n_matched == 4
    assert len(sar.cliffs) >= 2
    assert sar.cliffs[0].delta_value >= 2.0

    # Avoid-ome MMP bioisostere rescue proposals on a piperidine-containing lead
    mmp_eng = AvoidomeMMPEngine(admet_suite=None, property_profile="cns_mpo_v2")
    rescues = mmp_eng.propose_rescues("Cc1ccc(C(=O)N2CCCCC2)cc1", max_proposals=15)
    assert len(rescues) >= 3
    transforms = {r.transform_name for r in rescues}
    assert "piperidine_to_morpholine" in transforms or "aryl_H_to_F" in transforms


def test_openfe_rbfe_network_export_and_boltz_cofolded_audit(tmp_path: Path) -> None:
    """Export charge-matched OpenFE RBFE graph & audit a co-folded PDB complex."""
    pocket_pdb = tmp_path / "pocket.pdb"
    pocket_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ASP A 103       5.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  OD1 ASP A 103       3.500   0.000   0.000  1.00 20.00           O",
                "ATOM      3  CA  PHE A 288      -5.000   2.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    congeneric = [
        ("analog_F", "CN1CCN(C(=O)c2ccc(F)cc2)CC1"),
        ("analog_Cl", "CN1CCN(C(=O)c2ccc(Cl)cc2)CC1"),
        ("analog_Me", "CN1CCN(C(=O)c2ccc(C)cc2)CC1"),
        ("analog_acid_diff_q", "O=C(O)c1ccc(C(=O)N2CCN(C)CC2)cc1"),  # zwitterion q=0 vs +1!
    ]
    bundle = export_openfe_rbfe_network(
        congeneric,
        pocket_pdb,
        tmp_path / "openfe_out",
        require_same_charge=True,
        filter_ring_strain=False,
        seed=42,
    )
    assert bundle.n_ligands == 4
    assert bundle.n_edges >= 2
    assert bundle.network_json.is_file()
    assert bundle.driver_script.is_file()
    assert bundle.ligands_sdf.is_file()
    # Every exported RBFE edge strictly enforces Delta formal charge == 0
    for edge in bundle.edges:
        assert edge.delta_formal_charge == 0
        assert 0.0 < edge.lomap_like_score <= 1.0
    net_data = json.loads(bundle.network_json.read_text(encoding="utf-8"))
    assert net_data["schema"] == "geneuclidean_openfe_rbfe_v1"

    # Build a synthetic Boltz-2/AF3-style combined receptor+HETATM PDB file and audit it
    lig = Chem.AddHs(Chem.MolFromSmiles("CNC(=O)c1ccccc1"))
    p = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    p.randomSeed = 1
    assert AllChem.EmbedMolecule(lig, p) == 0  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(lig, mmffVariant="MMFF94s", maxIters=200)  # type: ignore[attr-defined]
    lig_noh = Chem.RemoveHs(lig)
    # Write combined co-folded PDB with HETATM LIG far enough from receptor to be clash-free
    het_lines = [
        line.replace("UNL", "LIG")
        for line in Chem.MolToPDBBlock(lig_noh).splitlines()
        if line.startswith("HETATM")
    ]
    cofold_pdb = tmp_path / "boltz2_pred.pdb"
    cofold_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  PHE A 200       8.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CG  PHE A 200       8.000   1.500   0.000  1.00 20.00           C",
                *het_lines,
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ext_lig, ext_poc = load_cofolded_complex(
        cofold_pdb, ligand_resname="LIG", reference_smiles="CNC(=O)c1ccccc1", pocket_cutoff=15.0
    )
    assert ext_lig.GetNumAtoms() == lig_noh.GetNumAtoms()
    assert "PHE" in ext_poc
    rep = audit_cofolded_complex(
        cofold_pdb,
        ligand_resname="LIG",
        reference_smiles="CNC(=O)c1ccccc1",
        property_profile="oral_ro5",
        pocket_cutoff=15.0,
    )
    assert rep.passed_posebusters is True
    assert rep.passed_torsion_strain is True
