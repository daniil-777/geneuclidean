"""Comprehensive offline CI-safe unit & integration tests for the GenEuclidean Python SDK.

Covers:
1. PEP 562 lazy import isolation & startup latency (< 200 ms, zero torch/transformers/matplotlib).
2. PEP 561 `py.typed` marker packaging hygiene.
3. Industrial MedChem triage via SDK (DCU/reagent/PAINS rejection vs clinical/DRD1_1 pass).
4. Audited deterministic physiological pH 7.4 protonation on clinical CNS & peripheral drugs.
5. Reusable docking utilities (orthosteric box, ring-diverse ETKDGv3/MMFF94s conformers,
   Murcko core RMSD, cross-target ΔΔG selectivity matrix & permutation test, graceful missing Vina).
6. FDA Jan 2025 / EMA Jan 2026 Credibility Dossier JSON & Markdown export + SHA-256 provenance.
7. High-level SDK (`Pocket`, `score_molecules`, `MoleculeDesigner.from_mock`,
   `visualize_complex`, `visualize_gallery`, `visualize_molecules_2d`, registries).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import geneuclidean as ge
from geneuclidean.cns import CNSAdmetSuite

# Verified clinical drugs, flagship GenEuclidean lead, and known artifacts
ECOPIPAM_SMILES = "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21"
DRD1_1_SMILES = "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"
XANOMELINE_SMILES = "CCCCCCOc1nsnc1C1=CCCN(C)C1"
RISPERIDONE_SMILES = "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"
HALOPERIDOL_SMILES = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
DONEPEZIL_SMILES = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
GUANFACINE_SMILES = "NC(N)=NC(=O)Cc1c(Cl)cccc1Cl"
METFORMIN_SMILES = "CN(C)C(=N)NC(=N)N"
TROSPIUM_SMILES = "O=C(O[C@@H]1C[C@@H]2CC[C@H](C1)[N+]21CCCC1)C(O)(c1ccccc1)c1ccccc1"
FLUPENTIXOL_SMILES = "OCCN1CCN(CC/C=C2/c3ccccc3Sc3ccc(C(F)(F)F)cc32)CC1"
MP10_SMILES = "Cn1cc(-c2ccncc2)c(-c2ccc(OCc3ccc4ccccc4n3)cc2)n1"
DCU_SMILES = "O=C(NC1CCCCC1)NC1CCCCC1"
DIU_SMILES = "CC(C)NC(=O)NC(C)C"
HOBT_SMILES = "On1nnc2ccccc21"
DMAP_SMILES = "CN(C)c1ccncc1"
PH3PO_SMILES = "O=P(c1ccccc1)(c1ccccc1)c1ccccc1"
PAINS_RHODANINE_SMILES = "O=C1NC(=S)S/C1=C\\c1ccccc1"


# ---------------------------------------------------------------------------
# Fixtures: Minimal 3D Pocket PDB & 3D Ligand SDF (Zero External Data Needed)
# ---------------------------------------------------------------------------


def _write_mini_3d_complex(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal realistic 3D DRD1 pocket PDB (Asp103 + Phe288 + Ser198) + 3D SDF."""
    pocket_pdb = tmp_path / "9LLG_DRD1_mini_pocket.pdb"
    pocket_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ASP R 103       2.800   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ASP R 103       3.500   1.200   0.000  1.00 20.00           C",
                "ATOM      3  C   ASP R 103       4.900   1.000   0.000  1.00 20.00           C",
                "ATOM      4  O   ASP R 103       5.400   0.000   0.000  1.00 20.00           O",
                "ATOM      5  CB  ASP R 103       3.100   2.000   1.200  1.00 20.00           C",
                "ATOM      6  CG  ASP R 103       1.800   1.500   1.800  1.00 20.00           C",
                "ATOM      7  OD1 ASP R 103       0.900   0.900   1.200  1.00 20.00           O",
                "ATOM      8  OD2 ASP R 103       1.700   1.800   3.000  1.00 20.00           O",
                "ATOM      9  N   PHE R 288      -3.800   0.000   0.000  1.00 20.00           N",
                "ATOM     10  CA  PHE R 288      -3.600   1.400   0.000  1.00 20.00           C",
                "ATOM     11  C   PHE R 288      -4.800   2.200   0.000  1.00 20.00           C",
                "ATOM     12  O   PHE R 288      -5.900   1.700   0.000  1.00 20.00           O",
                "ATOM     13  CG  PHE R 288      -2.200   1.800   0.500  1.00 20.00           C",
                "ATOM     14  CD1 PHE R 288      -1.800   2.900   1.200  1.00 20.00           C",
                "ATOM     15  CD2 PHE R 288      -1.300   0.800   0.200  1.00 20.00           C",
                "ATOM     16  CE1 PHE R 288      -0.500   3.100   1.600  1.00 20.00           C",
                "ATOM     17  CE2 PHE R 288       0.000   1.000   0.600  1.00 20.00           C",
                "ATOM     18  CZ  PHE R 288       0.400   2.100   1.300  1.00 20.00           C",
                "ATOM     19  N   SER R 198       0.000  -4.200   0.000  1.00 20.00           N",
                "ATOM     20  CA  SER R 198       0.000  -3.200   1.000  1.00 20.00           C",
                "ATOM     21  CB  SER R 198       0.200  -2.100   0.100  1.00 20.00           C",
                "ATOM     22  OG  SER R 198       0.100  -1.000   0.900  1.00 20.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mol = Chem.AddHs(Chem.MolFromSmiles("CNCCc1ccccc1"))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(mol, params) == 0  # type: ignore[attr-defined]
    mol = Chem.RemoveHs(mol)
    ligand_sdf = tmp_path / "9LLG_DRD1_mini_lig.sdf"
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(mol)
    writer.close()
    return pocket_pdb, ligand_sdf


def _fit_tiny_admet_bundle(path: Path) -> Path:
    """Train a tiny deterministic CNSAdmetSuite in <1s for offline SDK tests."""
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
        "O=C(O)c1cnc(C(=O)O)nc1O",
        "NC(Cc1ccc(O)cc1)C(=O)NC(CO)C(=O)O",
        "O=C(O)c1cc(S(=O)(=O)O)ccc1O",
        "OCC1OC(O)C(O)C(O)C1O",
    ]
    smiles = cns_smiles + polar_smiles
    y_bbb = [1] * len(cns_smiles) + [0] * len(polar_smiles)
    y_pgp = [1 if i % 2 == 0 else 0 for i in range(len(smiles))]
    y_herg = [1] * len(cns_smiles) + [0] * len(polar_smiles)

    suite = CNSAdmetSuite(hgb_max_iter=10, rf_n_estimators=10, seed=42)
    suite.fit_endpoint("bbb", smiles, y_bbb, n_splits=2)
    suite.fit_endpoint("pgp", smiles, y_pgp, n_splits=2)
    suite.fit_endpoint("herg", smiles, y_herg, n_splits=2)
    return suite.save(path)


# ---------------------------------------------------------------------------
# 1. PEP 562 Lazy-Import Isolation, Startup Latency & PEP 561 py.typed
# ---------------------------------------------------------------------------


def test_pep562_lazy_import_isolation_and_latency() -> None:
    """Bare `import geneuclidean as ge` is <200ms and never loads torch/transformers/matplotlib."""
    probe = """
import json, sys, time
t0 = time.perf_counter()
import geneuclidean as ge
elapsed_ms = (time.perf_counter() - t0) * 1000.0
backbones = ge.list_backbones()
loaded_heavy = [
    m for m in ("torch", "transformers", "peft", "matplotlib", "sklearn", "rdkit", "meeko")
    if m in sys.modules
]
print(json.dumps({
    "version": ge.__version__,
    "elapsed_ms": round(elapsed_ms, 2),
    "loaded_heavy": loaded_heavy,
    "n_backbones": len(backbones),
    "has_pocket": "Pocket" in dir(ge),
    "has_score": "score_molecules" in dir(ge),
    "has_designer": "MoleculeDesigner" in dir(ge),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout.strip())
    assert info["version"] == "3.1.0"
    assert info["elapsed_ms"] < 200.0, f"Lazy import took {info['elapsed_ms']} ms (>= 200 ms)"
    assert info["loaded_heavy"] == [], f"Heavy modules eagerly imported: {info['loaded_heavy']}"
    assert info["n_backbones"] >= 6
    assert info["has_pocket"] and info["has_score"] and info["has_designer"]


def test_pep561_py_typed_and_invalid_attr() -> None:
    """Verify PEP 561 py.typed marker is present and missing attributes raise AttributeError."""
    pkg_dir = Path(ge.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file()
    missing_name = "nonexistent_symbol_xyz"
    with pytest.raises(AttributeError, match="no attribute"):
        _ = getattr(ge, missing_name)


# ---------------------------------------------------------------------------
# 2. Industrial MedChem Triage & Reagent Quarantine via Top-Level SDK
# ---------------------------------------------------------------------------


def test_sdk_medchem_triage_blocks_reagents_and_passes_clinical_leads() -> None:
    """DCU/DIU/HOBt/DMAP/Ph3PO/PAINS are blocked; ecopipam, DRD1_1, and xanomeline pass."""
    for artifact_smi in (
        DCU_SMILES,
        DIU_SMILES,
        HOBT_SMILES,
        DMAP_SMILES,
        PH3PO_SMILES,
        "CC(=O)C=C",
        PAINS_RHODANINE_SMILES,
    ):
        res = ge.triage_molecule(artifact_smi)
        assert res.valid is True
        assert res.hygiene_gate == pytest.approx(0.0), f"Artifact not vetoed: {artifact_smi}"
        assert res.triage_pass is False

    batch = ge.triage_molecules([ECOPIPAM_SMILES, DRD1_1_SMILES, XANOMELINE_SMILES])
    assert len(batch) == 3
    for res in batch:
        assert res.valid is True
        assert res.reagent_quarantine_free is True
        assert res.pains_free is True
        assert res.hygiene_gate >= 0.70
        assert res.exact_murcko_scaffold != ""

    # Ecopipam & DRD1_1 pass full strict MedChem triage cleanly
    assert batch[0].triage_pass is True
    assert batch[1].triage_pass is True

    # Scaffold-hop check: DRD1_1 hops away from Ecopipam's benzazepine scaffold
    ecopipam_scafs = {batch[0].exact_murcko_scaffold, batch[0].generic_murcko_scaffold}
    assert ge.is_scaffold_hop(DRD1_1_SMILES, ecopipam_scafs, max_tanimoto=0.55) is True
    assert ge.is_scaffold_hop(ECOPIPAM_SMILES, ecopipam_scafs, max_tanimoto=0.55) is False


# ---------------------------------------------------------------------------
# 3. Audited Deterministic Physiological pH 7.4 Protonation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "smiles", "expected_charge", "expected_basic"),
    [
        ("ecopipam", ECOPIPAM_SMILES, 1, True),
        ("drd1_1", DRD1_1_SMILES, 1, True),
        ("risperidone", RISPERIDONE_SMILES, 1, True),
        ("haloperidol", HALOPERIDOL_SMILES, 1, True),
        ("donepezil", DONEPEZIL_SMILES, 1, True),
        ("xanomeline", XANOMELINE_SMILES, 1, True),
        ("guanfacine", GUANFACINE_SMILES, 1, True),
        ("metformin_monocation", METFORMIN_SMILES, 1, True),
        ("trospium_quat", TROSPIUM_SMILES, 1, True),
        ("flupentixol_piperazine_mono", FLUPENTIXOL_SMILES, 1, True),
        ("mp10_neutral", MP10_SMILES, 0, False),
        ("dcu_urea_neutral", DCU_SMILES, 0, False),
        ("aniline_neutral", "Nc1ccccc1", 0, False),
        ("sulfonamide_neutral", "CS(=O)(=O)NC", 0, False),
        ("pyridine_neutral", "c1ccncc1", 0, False),
        ("acetic_acid", "CC(=O)O", -1, False),
        ("glycine_zwitterion", "NCC(=O)O", 0, True),
    ],
)
def test_protonation_ph74_clinical_and_idempotent(
    name: str,
    smiles: str,
    expected_charge: int,
    expected_basic: bool,
) -> None:
    """Verify physiological pH 7.4 charge, basic-amine detection, and strict idempotency."""
    assert ge.estimate_physiological_charge(smiles) == expected_charge, name
    assert ge.has_basic_amine(smiles) is expected_basic, name

    prot_once = ge.protonate_ph74(smiles)
    prot_twice = ge.protonate_ph74(prot_once)
    assert isinstance(prot_once, str)
    assert prot_once == prot_twice, f"Non-idempotent pH 7.4 protonation for {name}"

    analysis = ge.analyze_protonation_ph74(smiles)
    assert analysis.net_charge == expected_charge
    assert analysis.protonated_smiles == prot_once
    if name == "glycine_zwitterion":
        assert analysis.is_zwitterion is True
        assert analysis.n_cationic_sites == 1
        assert analysis.n_anionic_sites == 1


def test_protonation_preserves_3d_coords_and_rejects_invalid() -> None:
    """Verify preserve_coords=True retains exact 3D coordinates and invalid SMILES raise."""
    mol = Chem.AddHs(Chem.MolFromSmiles(ECOPIPAM_SMILES))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(mol, params) == 0  # type: ignore[attr-defined]
    mol_noh = Chem.RemoveHs(mol)
    orig_pos = np.asarray(mol_noh.GetConformer().GetPositions(), dtype=np.float64)

    prot_mol = ge.protonate_ph74(mol_noh, preserve_coords=True)
    assert isinstance(prot_mol, Chem.Mol)
    assert prot_mol.GetNumConformers() == 1
    assert Chem.GetFormalCharge(prot_mol) == 1
    prot_pos = np.asarray(prot_mol.GetConformer().GetPositions(), dtype=np.float64)
    np.testing.assert_allclose(prot_pos, orig_pos, atol=1e-6)

    with pytest.raises(ValueError, match="Invalid SMILES"):
        ge.protonate_ph74("NOT_A_VALID_SMILES!!!")


# ---------------------------------------------------------------------------
# 4. Docking Pipeline Utilities & Cross-Target ΔΔG Selectivity Matrix
# ---------------------------------------------------------------------------


def test_orthosteric_box_and_ring_diverse_conformers(tmp_path: Path) -> None:
    """Verify tight per-axis / cubic orthosteric boxes, ETKDGv3 ring conformers, and core RMSD."""
    _, ligand_sdf = _write_mini_3d_complex(tmp_path)

    tight = ge.compute_orthosteric_box(ligand_sdf, mode="tight_per_axis")
    assert tight.mode == "tight_per_axis"
    assert all(14.0 <= s <= 22.0 for s in tight.size)
    assert len(tight.center) == 3

    cubic = ge.compute_orthosteric_box(ligand_sdf, mode="cubic_standard")
    assert cubic.mode == "cubic_standard"
    assert cubic.size[0] == cubic.size[1] == cubic.size[2]
    assert 20.0 <= cubic.size[0] <= 26.0

    # Ring-diverse 2D-to-3D conformer generation
    confs_a = ge.generate_ring_diverse_conformers(
        RISPERIDONE_SMILES, seed=42, num_confs=12, max_ring_confs=3
    )
    confs_b = ge.generate_ring_diverse_conformers(
        RISPERIDONE_SMILES, seed=42, num_confs=12, max_ring_confs=3
    )
    assert 1 <= len(confs_a) <= 3
    assert len(confs_a) == len(confs_b)
    assert confs_a[0].force_field in {"MMFF94s", "UFF"}
    assert confs_a[0].mol_3d.GetNumConformers() == 1
    assert Chem.GetFormalCharge(confs_a[0].mol_3d) == 1

    # Murcko core RMSD: 0.0 for identical pose, 2.0 A under rigid translation
    ref_mol = confs_a[0].mol_3d
    assert ge.compute_murcko_core_rmsd(ref_mol, ref_mol) == pytest.approx(0.0, abs=1e-5)

    shifted = Chem.Mol(ref_mol)
    conf = shifted.GetConformer()
    for i in range(shifted.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 2.0, p.y, p.z))
    assert ge.compute_murcko_core_rmsd(ref_mol, shifted) == pytest.approx(2.0, abs=1e-4)


def test_docking_graceful_missing_vina_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing AutoDock Vina raises a clear RuntimeError (or returns error when optional)."""
    pocket_pdb, ligand_sdf = _write_mini_3d_complex(tmp_path)
    monkeypatch.setattr("geneuclidean.evaluation.docking_pipeline._require_meeko", lambda: None)
    monkeypatch.setattr(
        "geneuclidean.evaluation.docking_pipeline._resolve_vina_binary", lambda _: None
    )

    with pytest.raises(RuntimeError, match="AutoDock Vina executable not found"):
        ge.dock_ligand_to_pocket(
            ECOPIPAM_SMILES,
            pocket_pdb,
            native_ligand_sdf=ligand_sdf,
            vina_bin=tmp_path / "missing_vina",
            require_vina=True,
        )

    res = ge.dock_ligand_to_pocket(
        ECOPIPAM_SMILES,
        pocket_pdb,
        native_ligand_sdf=ligand_sdf,
        vina_bin=tmp_path / "missing_vina",
        require_vina=False,
    )
    assert res.success is False
    assert res.error is not None and "AutoDock Vina" in res.error


def test_cross_target_selectivity_matrix_and_ddg() -> None:
    """Verify thermodynamic ΔΔG and double-centered diagonal selectivity permutation test."""
    ddg = ge.compute_selectivity_ddg(-9.5, [-8.0, -7.8])
    assert ddg["ddg_vs_mean"] == pytest.approx(-1.6, abs=1e-3)
    assert ddg["estimated_fold_selectivity_mean"] > 10.0

    # Synthetic 5x5 matrix with clear diagonal selectivity plus pocket & ligand main effects
    targets = ["DRD1", "DRD2", "VMAT2", "ADRA2A", "PDE10A"]
    pocket_bias = np.array([-0.5, 0.3, -0.8, 0.2, 0.1])
    ligand_bias = np.array([-0.2, 0.1, -0.4, 0.3, 0.2])
    base = -7.5 + ligand_bias[:, None] + pocket_bias[None, :]
    selective = base - 1.8 * np.eye(5)

    res = ge.compute_cross_target_matrix(selective, targets, n_permutations=2000, seed=42)
    assert res.target_names == tuple(targets)
    assert res.diagonal_mean_dg < res.off_diagonal_mean_dg
    assert res.raw_permutation_p < 0.05
    assert res.col_z_permutation_p < 0.05
    assert res.double_centered_permutation_p < 0.05
    assert all(row["col_rank"] == 1 for row in res.per_target)

    # Mapping input form produces identical results
    mapping = {
        (targets[i], targets[j]): [float(selective[i, j])] for i in range(5) for j in range(5)
    }
    res_map = ge.compute_cross_target_matrix(mapping, targets, n_permutations=2000, seed=42)
    assert res_map.double_centered_permutation_p == pytest.approx(res.double_centered_permutation_p)


# ---------------------------------------------------------------------------
# 5. FDA Jan 2025 / EMA Jan 2026 Credibility Dossier Export
# ---------------------------------------------------------------------------


def test_credibility_dossier_json_and_markdown_export(tmp_path: Path) -> None:
    """Verify 7-step FDA/EMA dossier generation, SHA-256 provenance, JSON & Markdown export."""
    dummy_dataset = tmp_path / "dummy_admet.csv"
    dummy_dataset.write_text("SMILES,Y\nCCO,1\n", encoding="utf-8")

    dossier = ge.build_credibility_dossier(
        model_name="GenEuclidean Test Suite",
        model_version="3.1.0",
        admet_cv_metrics={
            "endpoints": {
                "bbb": {
                    "n_samples": 7805,
                    "n_unique_scaffolds": 3400,
                    "scaffold_leakage_groups": 0,
                    "oof_roc_auc": 0.918,
                    "oof_pr_auc": 0.961,
                    "oof_brier_score": 0.088,
                    "oof_ece": 0.021,
                    "calibration_method": "sigmoid",
                }
            }
        },
        provenance_files={"admet_table": dummy_dataset},
    )
    assert "in silico" in dossier.context_of_use.lower()
    assert "in silico" in dossier.disclaimer.lower()

    payload = dossier.to_dict()
    assert payload["step_5_provenance_and_execution"]["n_tracked_artifacts"] == 1
    ledger = payload["step_5_provenance_and_execution"]["artifact_sha256_ledger"]["admet_table"]
    assert ledger["exists"] is True
    assert isinstance(ledger["sha256"], str) and len(ledger["sha256"]) == 64

    json_out = ge.export_credibility_json(dossier, tmp_path / "dossier.json")
    md_out = ge.export_credibility_markdown(dossier, tmp_path / "dossier.md")
    assert json_out.is_file() and md_out.is_file()

    reloaded = json.loads(json_out.read_text(encoding="utf-8"))
    assert reloaded["model_identity"]["version"] == "3.1.0"
    md_text = md_out.read_text(encoding="utf-8")
    assert "AI Credibility Dossier" in md_text
    assert "`bbb`" in md_text


# ---------------------------------------------------------------------------
# 6. High-Level SDK: Registries, Pocket, score_molecules, Designer & Visualizers
# ---------------------------------------------------------------------------


def test_registries_and_pocket_api(tmp_path: Path) -> None:
    """Verify list_targets, get_target, list_backbones, Pocket.from_target, and Pocket.from_pdb."""
    all_targets = ge.list_targets()
    assert len(all_targets) >= 12
    ts_targets = ge.list_targets("tourette")
    assert any(t.target_id == "DRD1" for t in ts_targets)

    drd1 = ge.get_target("DRD1")
    assert drd1.primary_pdb_id == "9LLG"
    assert ge.get_target("CHEMBL1893").target_id == "VMAT2"

    backbones = ge.list_backbones()
    assert "ibm-research/GP-MoLFormer-Uniq" in backbones
    assert "datamol-io/safe-gpt" in backbones
    assert "mock-tiny-gpt2" in backbones

    # Pocket.from_target works cleanly offline even with empty structures_dir
    pocket_meta = ge.Pocket.from_target("DRD1", structures_dir=tmp_path)
    assert pocket_meta.pdb_id == "9LLG"
    assert pocket_meta.summary()["primary_anchor"] == "Asp103"

    # Pocket.from_pdb featurizes, audits hotspots, and computes orthosteric box
    pocket_pdb, ligand_sdf = _write_mini_3d_complex(tmp_path)
    pocket = ge.Pocket.from_pdb(pocket_pdb, ligand_sdf=ligand_sdf, target_id="DRD1")
    feats = pocket.featurize()
    assert feats.node_feats.shape == (len(feats.coords), 52)
    assert np.all(np.isfinite(feats.node_feats))

    audit = pocket.audit_hotspots()
    assert "Asp103" in audit["present_hotspots"]
    assert "Phe288" in audit["present_hotspots"]

    box = pocket.orthosteric_box(mode="tight_per_axis")
    assert all(14.0 <= edge <= 22.0 for edge in box.size)


def test_score_molecules_with_and_without_admet(tmp_path: Path) -> None:
    """Verify score_molecules gates DCU to 0.0, scores DRD1_1/ecopipam, and supports ADMET."""
    reports = ge.score_molecules(
        [ECOPIPAM_SMILES, DRD1_1_SMILES, DCU_SMILES, "INVALID_SMILES"],
        target="DRD1",
        include_triage=True,
        include_admet=False,
    )
    assert len(reports) == 4
    eco, lead, dcu, bad = reports

    assert eco.valid is True and eco.triage_passed is True
    assert eco.net_charge_ph74 == 1 and eco.has_basic_amine is True
    assert eco.cns_mpo_v2 >= 4.0 and eco.composite_reward > 0.50

    assert lead.valid is True and lead.triage_passed is True
    assert lead.net_charge_ph74 == 1 and lead.cns_mpo_v2 >= 4.5

    assert dcu.valid is True and dcu.triage_passed is False
    assert dcu.reagent_quarantine_free is False
    assert dcu.hygiene_gate == pytest.approx(0.0)
    assert dcu.composite_reward == pytest.approx(0.0)

    assert bad.valid is False and bad.composite_reward == pytest.approx(0.0)
    assert eco.to_dict()["target_id"] == "DRD1"

    # With a real tiny fitted ADMET bundle
    oracle_path = _fit_tiny_admet_bundle(tmp_path / "tiny_admet.joblib")
    admet_reports = ge.score_molecules(
        [ECOPIPAM_SMILES],
        target="DRD1",
        include_admet=True,
        oracle_path=oracle_path,
    )
    assert admet_reports[0].admet is not None
    assert 0.0 <= float(admet_reports[0].admet["bbb_prob"]) <= 1.0


def test_molecule_designer_v2_checkpoint_end_to_end(tmp_path: Path) -> None:
    """Verify MoleculeDesigner.from_pretrained loads a v2 checkpoint and designs offline."""
    import torch

    from geneuclidean.config import (
        CVAEConfig,
        DecoderConfig,
        EncoderConfig,
        ExperimentConfig,
        MultiTaskConfig,
    )
    from geneuclidean.data.tokenizer import MolecularTokenizer
    from geneuclidean.models.geneuclidean import GenEuclideanModel

    tok = MolecularTokenizer(representation="smiles").fit(
        [ECOPIPAM_SMILES, DRD1_1_SMILES, "CCO", "CCN", "c1ccccc1"]
    )
    vocab_path = tmp_path / "vocab_smiles.json"
    tok.save(vocab_path)

    cfg = ExperimentConfig(
        encoder=EncoderConfig(arch="painn", hidden_dim=32, vector_dim=16, out_dim=32, num_layers=1),
        cvae=CVAEConfig(enabled=True, latent_dim=16, num_latent_tokens=2),
        decoder=DecoderConfig(
            arch="transformer",
            vocab_size=tok.vocab_size,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
        ),
        multitask=MultiTaskConfig(use_pkd_head=True, use_clip_head=True, use_prop_head=True),
    )
    model = GenEuclideanModel(cfg)
    ckpt_path = tmp_path / "tiny_v2.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg.model_dump(mode="python")}, ckpt_path
    )

    pocket_pdb, ligand_sdf = _write_mini_3d_complex(tmp_path)
    pocket = ge.Pocket.from_pdb(pocket_pdb, ligand_sdf=ligand_sdf, target_id="DRD1")

    designer = ge.MoleculeDesigner.from_pretrained(
        ckpt_path, mode="auto", vocab_path=vocab_path, device="cpu"
    )
    assert designer.backend == "v2"
    design_res = designer.design(
        pocket,
        n=3,
        max_new_tokens=12,
        cfg_scale=1.2,
        include_admet=False,
        seed=42,
    )
    assert design_res.pocket_id == "DRD1"
    assert len(design_res.raw_smiles) == 3
    assert 0.0 <= design_res.valid_rate <= 1.0


def test_molecule_designer_mock_end_to_end(tmp_path: Path) -> None:
    """Verify offline zero-download MoleculeDesigner.from_mock designs against a 3D Pocket."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    pocket_pdb, ligand_sdf = _write_mini_3d_complex(tmp_path)
    pocket = ge.Pocket.from_pdb(pocket_pdb, ligand_sdf=ligand_sdf, target_id="DRD1")

    designer = ge.MoleculeDesigner.from_mock(device="cpu", seed=42)
    design_res = designer.design(
        pocket,
        n=4,
        max_new_tokens=16,
        cfg_scale=1.2,
        include_admet=False,
        seed=42,
    )
    assert design_res.pocket_id == "DRD1"
    assert len(design_res.raw_smiles) == 4
    assert 0.0 <= design_res.valid_rate <= 1.0
    assert "reports" in design_res.to_dict()


def test_visualizers_complex_gallery_and_2d(tmp_path: Path) -> None:
    """Verify visualize_complex (HTML + PNG), visualize_gallery, and visualize_molecules_2d."""
    pocket_pdb, ligand_sdf = _write_mini_3d_complex(tmp_path)

    out_html = tmp_path / "complex.html"
    out_png = tmp_path / "complex.png"
    res = ge.visualize_complex(
        pocket_pdb,
        ligand_sdf,
        out_html=out_html,
        out_png=out_png,
        title="DRD1 Mini Complex",
        pdb_id="9LLG",
        disease="Tourette Syndrome",
        theme="dark",
        metrics={"vina_kcal": -9.26, "cns_mpo": 5.4},
    )
    assert res["html_path"] == out_html and out_html.is_file()
    assert res["png_path"] == out_png and out_png.is_file()
    html_text = out_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "3Dmol" in html_text
    assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    gallery_html = tmp_path / "gallery.html"
    written_gal = ge.visualize_gallery(
        [
            {
                "target_id": "DRD1",
                "title": "DRD1 Showcase",
                "pdb_id": "9LLG",
                "pocket_pdb": pocket_pdb,
                "ligand_sdf": ligand_sdf,
                "metrics": {"cns_mpo": 5.4},
            }
        ],
        out_html=gallery_html,
    )
    assert written_gal == gallery_html and gallery_html.is_file()
    assert "DRD1 Showcase" in gallery_html.read_text(encoding="utf-8")

    grid_png = tmp_path / "leads_2d.png"
    written_2d = ge.visualize_molecules_2d(
        [ECOPIPAM_SMILES, DRD1_1_SMILES, XANOMELINE_SMILES],
        out_png=grid_png,
        labels=["Ecopipam", {"QED": 0.85, "SA": 1.8}, "Xanomeline"],
        reference_smiles=ECOPIPAM_SMILES,
        title="CNS Leads",
    )
    assert written_2d == grid_png and grid_png.is_file()
    assert grid_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
