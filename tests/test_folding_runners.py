"""Offline CI-safe tests for GenEuclidean's open-weights co-folding hub & boltz_adapter upgrades."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

import geneuclidean.folding as folding
from geneuclidean.folding.runners import (
    FOLDING_BACKEND_REGISTRY,
    FoldedComplexResult,
    FoldingJobSpec,
    MissingBackendError,
    NetworkDisabledError,
    export_alphafold3_json,
    export_boltz2_yaml,
    export_protenix_json,
    fold_complex,
    folding_backend_catalog,
    list_folding_backends,
)
from geneuclidean.interop.boltz_adapter import (
    ConfidenceSummary,
    extract_bfactor_plddt,
    extract_bfactor_plddt_from_text,
    load_cofold_confidence_json,
    load_cofolded_complex,
    normalize_plddt_values,
)


def _make_clean_ligand_3d(smiles: str = "CNC(=O)c1ccccc1") -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)  # type: ignore[attr-defined]
    return Chem.RemoveHs(mol)


def _write_mock_complex_mmcif(
    dest: Path,
    lig_mol: Chem.Mol,
    *,
    protein_bfactors: tuple[float, ...] = (91.50, 88.20),
    ligand_bfactor: float = 84.60,
    include_zn_cofactor: bool = False,
) -> Path:
    """Write a minimal valid AF3/Boltz-2-style mmCIF containing protein + ligand (+ optional ZN)."""
    lines = [
        "data_mock_cofold",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    serial = 1
    prot_atoms = [
        ("C", "CA", "PHE", 200, 8.0, 0.0, 0.0, protein_bfactors[0]),
        ("C", "CG", "PHE", 200, 8.0, 1.5, 0.0, protein_bfactors[1]),
    ]
    for elem, aname, rname, rseq, x, y, z, bfac in prot_atoms:
        lines.append(
            f"ATOM {serial} {elem} {aname} . {rname} A 1 {rseq} ? "
            f"{x:.3f} {y:.3f} {z:.3f} 1.00 {bfac:.2f} {rseq} {rname} A {aname} 1"
        )
        serial += 1

    if include_zn_cofactor:
        lines.append(
            f"HETATM {serial} ZN ZN . ZN C 3 . ? 6.000 1.000 0.000 1.00 95.00 301 ZN C ZN 1"
        )
        serial += 1

    conf = lig_mol.GetConformer()
    for idx, atom in enumerate(lig_mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        elem = atom.GetSymbol()
        aname = f"{elem}{idx}"
        lines.append(
            f"HETATM {serial} {elem} {aname} . LIG B 2 . ? "
            f"{pos.x:.3f} {pos.y:.3f} {pos.z:.3f} 1.00 {ligand_bfactor:.2f} 1 LIG B {aname} 1"
        )
        serial += 1

    lines.append("#")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# 1. Cold-Import Speed & Zero Heavy Dependencies
# ---------------------------------------------------------------------------


def test_cold_import_folding_subpackage_fast_and_zero_heavy() -> None:
    """`import geneuclidean.folding` must finish in < 5 ms with zero eager heavy imports."""
    code = (
        "import json, sys, time; "
        "import geneuclidean; "
        "t0 = time.perf_counter(); "
        "import geneuclidean.folding as f; "
        "dt_ms = (time.perf_counter() - t0) * 1000.0; "
        "backends = f.list_folding_backends(); "
        "af3 = f.export_alphafold3_json(protein_sequence='MKTAYIAK', ligand_smiles='CCO'); "
        "b2 = f.export_boltz2_yaml(protein_sequence='MKTAYIAK', ligand_smiles='CCO'); "
        "px = f.export_protenix_json(protein_sequence='MKTAYIAK', ligand_smiles='CCO'); "
        "heavy = [m for m in ('torch', 'rdkit', 'Bio', 'numpy', 'scipy', 'pandas', "
        "'sklearn', 'transformers', 'matplotlib', 'e3nn', 'requests') if m in sys.modules]; "
        "print(json.dumps({'dt_ms': dt_ms, 'heavy': heavy, 'backends': backends, "
        "'af3_dialect': af3['dialect'], 'b2_ok': 'version: 1' in b2, 'px_len': len(px)}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["heavy"] == [], f"Unexpected eager heavy imports: {payload['heavy']}"
    assert payload["dt_ms"] < 5.0, f"Cold import took {payload['dt_ms']:.2f} ms (limit < 5 ms)"
    assert set(payload["backends"]) >= {
        "alphafold3",
        "boltz2",
        "esmfold_api",
        "nanofold3",
        "protenix",
    }
    assert payload["af3_dialect"] == "alphafold3"
    assert payload["b2_ok"] is True
    assert payload["px_len"] == 1


def test_folding_lazy_dir_and_unknown_attr() -> None:
    """`geneuclidean.folding` exposes both runner & NanoFold3 symbols via PEP 562."""
    exported = set(dir(folding))
    for sym in (
        "FOLDING_BACKEND_REGISTRY",
        "FoldedComplexResult",
        "FoldingJobSpec",
        "MissingBackendError",
        "fold_complex",
        "list_folding_backends",
        "export_alphafold3_json",
        "export_boltz2_yaml",
        "export_protenix_json",
        "NanoFold3",
        "NanoFold3Config",
        "build_nanofold3",
        "PairformerStack",
        "PairformerBlock",
        "SimpleFoldTrunk",
        "TriangleMultiplicativeUpdate",
        "TriangleAttention",
        "PairBiasedAttention",
        "AllAtomDiffusionModule",
        "EDMDiffusionSchedule",
        "FlowMatchingSchedule",
        "KabschWeightedRigidAlign",
        "CoFoldingLoss",
        "ConfidenceHead",
        "AffinityHead",
    ):
        assert sym in exported, f"Missing expected lazy export {sym!r}"
    with pytest.raises(AttributeError):
        _ = folding.nonexistent_folding_symbol


# ---------------------------------------------------------------------------
# 2. Schema Exporters (AlphaFold 3 JSON v2/v3/v4, Boltz-2 YAML v1, Protenix JSON)
# ---------------------------------------------------------------------------


def test_exporters_alphafold3_boltz2_protenix_schemas(tmp_path: Path) -> None:
    """All three schema exporters produce spec-compliant round-trippable files."""
    ecopipam = "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21"
    job = FoldingJobSpec(
        name="drd1_ecopipam",
        protein_chains=(("A", "MRTLNTSAMDGTGLVVERDFSVRILTACFLSLLILSTLLGNTLVCAAVIRFR"),),
        ligand_smiles=(("B", ecopipam),),
        ligand_ccds=(("C", "ATP"),),
        pocket_contact_residues=(("A", 103), ("A", 288)),
        bonded_atom_pairs=((("A", 103, "OD1"), ("B", 1, "N1")),),
        msa_paths={"A": tmp_path / "drd1.a3m"},
        seeds=(42, 1337),
        predict_affinity=True,
        user_ccd="data_MYLIG\n#",
    )

    # AlphaFold 3 JSON v2 / v3 / v4
    for ver in (2, 3, 4):
        af3_file = tmp_path / f"af3_v{ver}.json"
        af3_doc = export_alphafold3_json(job, af3_file, dialect_version=ver)
        reloaded = json.loads(af3_file.read_text(encoding="utf-8"))
        assert reloaded == af3_doc
        assert af3_doc["dialect"] == "alphafold3"
        assert af3_doc["version"] == ver
        assert af3_doc["name"] == "drd1_ecopipam"
        assert af3_doc["modelSeeds"] == [42, 1337]
        assert af3_doc["sequences"][0]["protein"]["id"] == "A"
        assert "unpairedMsaPath" in af3_doc["sequences"][0]["protein"]
        assert af3_doc["sequences"][1]["ligand"] == {"id": "B", "smiles": ecopipam}
        assert af3_doc["sequences"][2]["ligand"] == {"id": "C", "ccdCodes": ["ATP"]}
        assert af3_doc["bondedAtomPairs"] == [[["A", 103, "OD1"], ["B", 1, "N1"]]]
        assert af3_doc["userCCD"] == "data_MYLIG\n#"

    with pytest.raises(ValueError, match="dialect version"):
        export_alphafold3_json(job, dialect_version=1)

    # Boltz-2 YAML v1 (verify strict YAML parsing + safe chiral SMILES quoting!)
    b2_file = tmp_path / "boltz2_job.yaml"
    b2_yaml = export_boltz2_yaml(job, b2_file)
    assert b2_file.read_text(encoding="utf-8") == b2_yaml
    parsed_yaml = yaml.safe_load(b2_yaml)
    assert parsed_yaml["version"] == 1
    assert parsed_yaml["sequences"][0]["protein"]["id"] == "A"
    assert parsed_yaml["sequences"][1]["ligand"]["smiles"] == ecopipam
    assert parsed_yaml["sequences"][2]["ligand"]["ccd"] == "ATP"
    assert parsed_yaml["constraints"][0]["pocket"] == {
        "binder": "B",
        "contacts": [["A", 103], ["A", 288]],
    }
    assert parsed_yaml["constraints"][1]["bond"] == {
        "atom1": ["A", 103, "OD1"],
        "atom2": ["B", 1, "N1"],
    }
    assert parsed_yaml["properties"] == [{"affinity": {"binder": "B"}}]

    # Protenix JSON
    px_file = tmp_path / "protenix_job.json"
    px_doc = export_protenix_json(job, px_file)
    assert json.loads(px_file.read_text(encoding="utf-8")) == px_doc
    assert len(px_doc) == 1
    entry = px_doc[0]
    assert entry["name"] == "drd1_ecopipam"
    assert entry["sequences"][0]["proteinChain"]["count"] == 1
    assert "msa" in entry["sequences"][0]["proteinChain"]
    assert entry["sequences"][1]["ligand"] == {"ligand": ecopipam, "count": 1}
    assert entry["sequences"][2]["ligand"] == {"ligand": "CCD_ATP", "count": 1}
    assert entry["covalent_bonds"][0]["entity1"] == 1
    assert entry["covalent_bonds"][0]["entity2"] == 2


# ---------------------------------------------------------------------------
# 3. B-Factor pLDDT Extraction (0..1 vs 0..100) & Confidence JSON Sidecars
# ---------------------------------------------------------------------------


def test_bfactor_plddt_extraction_and_autonormalization_pdb_and_cif(tmp_path: Path) -> None:
    """Auto-normalizes 0..1 vs 0..100 B-factor pLDDT cleanly for both PDB and mmCIF."""
    assert np.allclose(normalize_plddt_values([0.92, 0.85, 0.78]), [92.0, 85.0, 78.0], atol=1e-4)
    assert np.allclose(normalize_plddt_values([92.5, 85.0, 78.3]), [92.5, 85.0, 78.3], atol=1e-4)

    # Unit-scale PDB (ESMFold convention: 0.00..1.00)
    unit_pdb = tmp_path / "esmfold_unit.pdb"
    unit_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.91           N",
                "ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.84           C",
                "ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.76           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    unit_map = extract_bfactor_plddt(unit_pdb)
    assert np.allclose(unit_map["all"], [91.0, 84.0, 76.0], atol=1e-3)
    assert np.allclose(unit_map["protein"], [91.0, 84.0, 76.0], atol=1e-3)
    assert unit_map["ligand"].size == 0

    text_map = extract_bfactor_plddt_from_text(unit_pdb.read_text(encoding="utf-8"), fmt="pdb")
    assert np.allclose(text_map["all"], unit_map["all"], atol=1e-5)

    # Percent-scale mmCIF with protein + ligand + ZN cofactor
    lig = _make_clean_ligand_3d()
    cif_path = _write_mock_complex_mmcif(
        tmp_path / "cofold.cif",
        lig,
        protein_bfactors=(91.5, 88.2),
        ligand_bfactor=84.6,
        include_zn_cofactor=True,
    )
    cif_map = extract_bfactor_plddt(cif_path, ligand_resname="LIG")
    assert np.allclose(cif_map["protein"], [91.5, 88.2], atol=1e-3)
    assert cif_map["ligand"].size == lig.GetNumAtoms()
    assert np.allclose(cif_map["ligand"], np.full(lig.GetNumAtoms(), 84.6), atol=1e-3)
    assert "chain:A" in cif_map and "chain:B" in cif_map

    # Cofactor retention & preserved B-factors in load_cofolded_complex
    _, poc_with_zn = load_cofolded_complex(
        cif_path,
        ligand_resname="LIG",
        reference_smiles="CNC(=O)c1ccccc1",
        pocket_cutoff=15.0,
        keep_cofactors=("ZN",),
        preserve_bfactors=True,
    )
    assert "HETATM" in poc_with_zn and " ZN " in poc_with_zn
    assert " 91.50" in poc_with_zn


def test_confidence_json_parsers_af3_boltz2_protenix(tmp_path: Path) -> None:
    """Parse official AF3, Boltz-2 (+ affinity), and Protenix confidence JSONs cleanly."""
    # 1. AlphaFold 3 summary_confidences.json + confidences.json
    af3_sum = tmp_path / "job_summary_confidences.json"
    af3_sum.write_text(
        json.dumps(
            {
                "ptm": 0.87,
                "iptm": 0.81,
                "ranking_score": 0.822,
                "fraction_disordered": 0.04,
                "has_clash": 0.0,
                "chain_ptm": [0.89, 0.78],
                "chain_iptm": [0.81, 0.81],
                "chain_pair_iptm": [[0.89, 0.81], [0.81, 0.78]],
            }
        ),
        encoding="utf-8",
    )
    af3_full = tmp_path / "job_confidences.json"
    af3_full.write_text(
        json.dumps(
            {
                "atom_plddts": [92.0, 90.0, 86.0, 84.0],
                "pae": [[0.5, 1.8], [2.1, 0.6]],
            }
        ),
        encoding="utf-8",
    )
    s_af3 = load_cofold_confidence_json(af3_sum, confidences_path=af3_full)
    assert isinstance(s_af3, ConfidenceSummary)
    assert s_af3.source_format == "alphafold3"
    assert s_af3.ptm == pytest.approx(0.87)
    assert s_af3.iptm == pytest.approx(0.81)
    assert s_af3.ranking_score == pytest.approx(0.822)
    assert s_af3.plddt_mean == pytest.approx(88.0)
    assert s_af3.has_clash is False
    assert s_af3.pae_matrix is not None and s_af3.pae_matrix.shape == (2, 2)

    # 2. Boltz-2 confidence_*.json (0..1 pLDDT!) + affinity_*.json
    b2_conf = tmp_path / "confidence_job_model_0.json"
    b2_conf.write_text(
        json.dumps(
            {
                "confidence_score": 0.854,
                "ptm": 0.91,
                "iptm": 0.84,
                "ligand_iptm": 0.86,
                "protein_iptm": 0.90,
                "complex_plddt": 0.885,
                "complex_iplddt": 0.870,
                "complex_pde": 0.92,
                "chains_ptm": {"0": 0.91, "1": 0.85},
                "pair_chains_iptm": {"0": {"0": 0.91, "1": 0.84}, "1": {"0": 0.84, "1": 0.85}},
            }
        ),
        encoding="utf-8",
    )
    b2_aff = tmp_path / "affinity_job.json"
    b2_aff.write_text(
        json.dumps(
            {
                "affinity_pred_value": -1.45,
                "affinity_probability_binary": 0.93,
            }
        ),
        encoding="utf-8",
    )
    s_b2 = load_cofold_confidence_json(b2_conf, affinity_path=b2_aff)
    assert s_b2.source_format == "boltz2"
    assert s_b2.plddt_mean == pytest.approx(88.5, abs=1e-3)
    assert s_b2.ligand_plddt == pytest.approx(87.0, abs=1e-3)
    assert s_b2.affinity_pred_value == pytest.approx(-1.45)
    assert s_b2.predicted_pkd == pytest.approx(7.45)
    assert s_b2.affinity_probability_binary == pytest.approx(0.93)
    assert s_b2.chain_ptm == pytest.approx((0.91, 0.85))

    # 3. Protenix summary_confidence_sample_0.json
    px_conf = tmp_path / "summary_confidence_sample_0.json"
    px_conf.write_text(
        json.dumps(
            {
                "plddt": 0.832,
                "gpde": 1.15,
                "ptm": 0.84,
                "iptm": 0.79,
                "ranking_score": 0.80,
                "disorder": 0.02,
            }
        ),
        encoding="utf-8",
    )
    s_px = load_cofold_confidence_json(px_conf)
    assert s_px.source_format == "protenix"
    assert s_px.plddt_mean == pytest.approx(83.2, abs=1e-3)
    assert s_px.complex_pde == pytest.approx(1.15)


# ---------------------------------------------------------------------------
# 4. Roundtrip Backend parse_output -> FoldedComplexResult -> PoseRealityAuditor
# ---------------------------------------------------------------------------


def test_roundtrip_parse_output_and_pose_audit_af3_boltz2_protenix(tmp_path: Path) -> None:
    """All external backends parse mmCIF + sidecar JSONs and bridge into PoseRealityAuditor."""
    smiles = "CNC(=O)c1ccccc1"
    lig = _make_clean_ligand_3d(smiles)

    # AlphaFold 3 mock output directory
    af3_dir = tmp_path / "af3_out"
    af3_dir.mkdir()
    _write_mock_complex_mmcif(af3_dir / "pred_model.cif", lig)
    (af3_dir / "pred_summary_confidences.json").write_text(
        json.dumps({"ptm": 0.88, "iptm": 0.82, "ranking_score": 0.832, "has_clash": 0.0}),
        encoding="utf-8",
    )
    (af3_dir / "pred_confidences.json").write_text(
        json.dumps(
            {
                "atom_plddts": [90.0] * (2 + lig.GetNumAtoms()),
                "pae": [[0.4, 1.2], [1.3, 0.5]],
            }
        ),
        encoding="utf-8",
    )
    res_af3 = fold_complex(
        protein_sequence="FF",
        ligand_smiles=smiles,
        backend="alphafold3",
        name="af3_roundtrip",
        output_dir=af3_dir,
        mode="parse_output",
        run_audit=True,
        pocket_cutoff=15.0,
    )
    assert res_af3.backend == "alphafold3"
    assert res_af3.ptm == pytest.approx(0.88)
    assert res_af3.iptm == pytest.approx(0.82)
    assert res_af3.ranking_score == pytest.approx(0.832)
    assert res_af3.plddt_mean == pytest.approx(90.0)
    assert res_af3.pae_matrix is not None and res_af3.pae_matrix.shape == (2, 2)
    assert len(res_af3.provenance_sha256) == 64
    assert res_af3.pose_audit is not None
    assert res_af3.pose_audit.passed_posebusters is True
    assert res_af3.audit_report is res_af3.pose_audit

    # Boltz-2 mock output directory
    b2_dir = tmp_path / "boltz2_out"
    b2_dir.mkdir()
    _write_mock_complex_mmcif(
        b2_dir / "pred_model_0.cif",
        lig,
        protein_bfactors=(0.91, 0.89),
        ligand_bfactor=0.86,
    )
    (b2_dir / "confidence_pred_model_0.json").write_text(
        json.dumps(
            {
                "confidence_score": 0.86,
                "ptm": 0.90,
                "iptm": 0.85,
                "complex_plddt": 0.88,
            }
        ),
        encoding="utf-8",
    )
    (b2_dir / "affinity_pred.json").write_text(
        json.dumps({"affinity_pred_value": -1.2, "affinity_probability_binary": 0.91}),
        encoding="utf-8",
    )
    res_b2 = fold_complex(
        protein_sequence="FF",
        ligand_smiles=smiles,
        backend="boltz2",
        name="b2_roundtrip",
        output_dir=b2_dir,
        mode="parse_output",
        run_audit=True,
        pocket_cutoff=15.0,
    )
    assert res_b2.backend == "boltz2"
    assert res_b2.plddt_mean == pytest.approx(88.0, abs=1e-3)
    assert res_b2.affinity_pred == pytest.approx(-1.2)
    assert res_b2.predicted_pkd == pytest.approx(7.2)
    assert res_b2.binding_prob == pytest.approx(0.91)
    assert res_b2.pose_audit is not None and res_b2.pose_audit.passed_posebusters is True

    # Protenix mock output directory + export_only smoke
    px_dir = tmp_path / "protenix_out"
    px_dir.mkdir()
    _write_mock_complex_mmcif(px_dir / "pred_sample_0.cif", lig)
    (px_dir / "summary_confidence_sample_0.json").write_text(
        json.dumps({"plddt": 86.5, "gpde": 0.9, "ptm": 0.85, "iptm": 0.80, "ranking_score": 0.81}),
        encoding="utf-8",
    )
    res_px = fold_complex(
        protein_sequence="FF",
        ligand_smiles=smiles,
        backend="protenix",
        name="px_roundtrip",
        output_dir=px_dir,
        mode="parse_output",
        run_audit=True,
        pocket_cutoff=15.0,
    )
    assert res_px.backend == "protenix"
    assert res_px.plddt_mean == pytest.approx(86.5)
    assert res_px.pose_audit is not None and res_px.pose_audit.passed_posebusters is True

    # Export-only mode writes job bundle without CLI and cleanly skips audit
    exp_res = fold_complex(
        protein_sequence="MKTAYIAK",
        ligand_smiles="CCO",
        backend="boltz2",
        name="export_demo",
        output_dir=tmp_path / "exported",
        mode="export_only",
        run_audit=True,
    )
    assert exp_res.pose_audit is None
    assert (tmp_path / "exported" / "export_demo.yaml").is_file()


# ---------------------------------------------------------------------------
# 5. Registry Extensibility & Graceful MissingBackendError
# ---------------------------------------------------------------------------


def test_registry_custom_backend_and_missing_cli_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom backends register cleanly and absent CLIs raise MissingBackendError with hints."""
    assert list_folding_backends() == [
        "alphafold3",
        "boltz2",
        "esmfold_api",
        "nanofold3",
        "protenix",
    ]
    catalog = folding_backend_catalog()
    assert "boltz2" in catalog and "boltz" in catalog["boltz2"]["aliases"]

    # Custom backend registration via scoped .temporary()
    def _mock_custom_backend(job: FoldingJobSpec, **kwargs: Any) -> FoldedComplexResult:
        del kwargs
        return FoldedComplexResult(
            backend="mock_custom",
            name=job.name,
            plddt_mean=93.4,
            plddt_per_atom=(93.4, 93.4),
            ptm=0.91,
            iptm=0.87,
            ranking_score=0.878,
            ligand_smiles=job.primary_ligand_smiles,
            provenance_sha256="a" * 64,
            metadata={"custom": True},
        )

    with FOLDING_BACKEND_REGISTRY.temporary("mock_custom", _mock_custom_backend):
        assert "mock_custom" in list_folding_backends()
        res = fold_complex(
            protein_sequence="MKTAYIAK",
            ligand_smiles="CCO",
            backend="mock_custom",
            name="custom_run",
            run_audit=False,
        )
        assert res.backend == "mock_custom"
        assert res.plddt_mean == pytest.approx(93.4)
        assert res.metadata["custom"] is True
    assert "mock_custom" not in list_folding_backends()

    # Graceful MissingBackendError when external CLIs are absent
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    for backend_name, expected_hint in (
        ("boltz2", "pip install boltz"),
        ("protenix", "pip install protenix"),
        ("alphafold3", "export_only"),
    ):
        with pytest.raises(MissingBackendError) as exc_info:
            fold_complex(
                protein_sequence="MKTAYIAK",
                ligand_smiles="CCO",
                backend=backend_name,
                run_audit=False,
            )
        assert expected_hint in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Offline Network Safety for ESMFold
# ---------------------------------------------------------------------------


def test_esmfold_offline_network_safety_and_mocked_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`esmfold_api` blocks HTTP calls when `allow_network=False` and caps API at 400 aa."""
    import urllib.request

    def _forbidden_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Network call attempted while allow_network=False or over 400 aa!")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden_urlopen)

    # Default allow_network=False raises clean NetworkDisabledError without touching network
    with pytest.raises(NetworkDisabledError, match="allow_network=False"):
        fold_complex(
            protein_sequence="MKTAYIAKQRQISFVKSHFSRQ",
            backend="esmfold_api",
            run_audit=False,
        )

    # Reject > 400 aa even with allow_network=True before touching network
    with pytest.raises(ValueError, match="<= 400 aa"):
        fold_complex(
            protein_sequence="A" * 401,
            backend="esmfold_api",
            allow_network=True,
            run_audit=False,
        )

    # Reject ligand co-folding on protein-only ESMFold
    with pytest.raises(ValueError, match="protein-only"):
        fold_complex(
            protein_sequence="MKTAYIAK",
            ligand_smiles="CCO",
            backend="esmfold_api",
            allow_network=True,
            run_audit=False,
        )

    # Mocked opt-in HTTP call parses PDB & auto-normalizes 0..1 B-factors to [0, 100]
    mock_pdb = "\n".join(
        [
            "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.92           N",
            "ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.88           C",
            "END",
        ]
    )

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_exc: Any) -> None:
            self.close()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _req, timeout=30.0: _FakeResp(mock_pdb.encode("utf-8")),
    )
    res = fold_complex(
        protein_sequence="M",
        backend="esmfold_api",
        name="esm_demo",
        output_dir=tmp_path / "esm_out",
        allow_network=True,
        run_audit=True,
    )
    assert res.backend == "esmfold_api"
    assert res.plddt_mean == pytest.approx(90.0, abs=1e-3)
    assert res.plddt_per_atom == pytest.approx((92.0, 88.0), abs=1e-3)
    assert res.pose_audit is None
    assert res.structure_path is not None and res.structure_path.is_file()


# ---------------------------------------------------------------------------
# 7. Security & Input Sanitization Invariants
# ---------------------------------------------------------------------------


def test_runners_security_ast_and_input_sanitization() -> None:
    """Verify zero `shell=True`/`eval`/`exec` in runners.py and strict input sanitization."""
    runners_src = (
        Path(__file__).resolve().parents[1] / "src" / "geneuclidean" / "folding" / "runners.py"
    )
    tree = ast.parse(runners_src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec"), f"Forbidden call {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr != "system", "Forbidden os.system call"
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False

    # Path traversal & CLI option injection blocked
    for bad_name in ("../escape", "/etc/passwd", "-rf", "bad name", "a\x00b"):
        with pytest.raises(ValueError, match="Invalid folding job name"):
            export_alphafold3_json(name=bad_name, protein_sequence="MKTAYIAK")

    # Unsafe SMILES characters blocked
    for bad_smi in ("CCO; rm -rf /", "CCO\nMALICIOUS", "CCO `id`", ""):
        with pytest.raises(ValueError, match=r"Invalid or unsafe ligand SMILES|single-line string"):
            export_boltz2_yaml(protein_sequence="MKTAYIAK", ligand_smiles=bad_smi)

    # Invalid AA sequence blocked
    with pytest.raises(ValueError, match="Invalid amino acid"):
        export_protenix_json(protein_sequence="MKT123!@#")


def test_nanofold3_backend_adapter_with_mock_model(tmp_path: Path) -> None:
    """`NanoFold3Backend` cleanly adapts a NanoFold3Result into FoldedComplexResult."""
    from types import SimpleNamespace

    mock_pdb = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 88.50           C\nEND\n"
    )

    class _FakeNanoFold3:
        def fold_sequence(
            self,
            sequence: str,
            *,
            num_samples: int = 1,
            num_steps: int | None = None,
            seed: int = 42,
        ) -> Any:
            del num_samples, num_steps
            return SimpleNamespace(
                backend="nanofold3",
                name=f"seq_{len(sequence)}_{seed}",
                pdb_text=mock_pdb,
                pocket_pdb_text=None,
                ligand_mol_3d=None,
                ligand_smiles=None,
                plddt_per_atom=(88.5,),
                plddt_mean=88.5,
                ligand_plddt=float("nan"),
                pae_matrix=np.zeros((1, 1), dtype=np.float32),
                pde_matrix=np.zeros((1, 1), dtype=np.float32),
                ptm=0.81,
                iptm=0.0,
                ranking_score=0.162,
                predicted_pkd=float("nan"),
                binding_prob=float("nan"),
                runtime_sec=0.05,
                provenance_sha256="",
                pose_audit=None,
                metadata={"mock": True},
            )

    res = fold_complex(
        protein_sequence="MKTAYIAK",
        backend="nanofold3",
        name="nf3_mock_seq",
        output_dir=tmp_path / "nf3_out",
        run_audit=False,
        model=_FakeNanoFold3(),
    )

    assert res.backend == "nanofold3"
    assert res.plddt_mean == pytest.approx(88.5)
    assert res.ptm == pytest.approx(0.81)
    assert res.affinity_pred is None
    assert res.predicted_pkd is None
    assert len(res.provenance_sha256) == 64
    assert res.structure_path is not None and res.structure_path.is_file()
    assert res.to_dict()["backend"] == "nanofold3"


# ---------------------------------------------------------------------------
# 8. Verifier-Caught Regression Tests (Literal AF3 Files, YAML 1.1 Quoting,
#    Bonded Atom & pocket_pdb Sanitization, NanoFold3 pocket_center)
# ---------------------------------------------------------------------------


def test_af3_literal_summary_and_confidences_filenames(tmp_path: Path) -> None:
    """Literal `summary_confidences.json` + `confidences.json` are both loaded cleanly."""
    smiles = "CNC(=O)c1ccccc1"
    lig = _make_clean_ligand_3d(smiles)
    af3_dir = tmp_path / "af3_literal"
    af3_dir.mkdir()
    _write_mock_complex_mmcif(
        af3_dir / "model.cif", lig, protein_bfactors=(50.0, 50.0), ligand_bfactor=50.0
    )
    (af3_dir / "summary_confidences.json").write_text(
        json.dumps({"ptm": 0.92, "iptm": 0.89, "ranking_score": 0.896, "has_clash": 0.0}),
        encoding="utf-8",
    )
    (af3_dir / "confidences.json").write_text(
        json.dumps(
            {
                "atom_plddts": [95.0] * (2 + lig.GetNumAtoms()),
                "pae": [[0.2, 0.9], [1.0, 0.3]],
            }
        ),
        encoding="utf-8",
    )
    res = fold_complex(
        protein_sequence="FF",
        ligand_smiles=smiles,
        backend="alphafold3",
        name="af3_literal_test",
        output_dir=af3_dir,
        mode="parse_output",
        run_audit=False,
    )
    assert res.metadata["summary_json"] is not None
    assert res.metadata["confidences_json"] is not None
    assert Path(res.metadata["confidences_json"]).name == "confidences.json"
    assert res.plddt_mean == pytest.approx(95.0)
    assert res.pae_matrix is not None and res.pae_matrix.shape == (2, 2)
    assert float(res.pae_matrix[0, 0]) == pytest.approx(0.2)


def test_boltz2_yaml_quotes_yaml11_ambiguous_chain_ids() -> None:
    """YAML 1.1 ambiguous chain IDs (NO, ON, 1, 2, TRUE, OFF, NULL, 1e2) roundtrip as exact str."""
    cases = (
        ("NO", "ON"),
        ("1", "2"),
        ("TRUE", "OFF"),
        ("NULL", "1e2"),
    )
    for prot_cid, lig_cid in cases:
        job = FoldingJobSpec(
            name="yaml_quote_test",
            protein_chains=((prot_cid, "MKTAYIAK"),),
            ligand_smiles=((lig_cid, "CCO"),),
            pocket_contact_residues=((prot_cid, 3),),
            bonded_atom_pairs=(((prot_cid, 3, "CA"), (lig_cid, 1, "C1")),),
            predict_affinity=True,
        )
        parsed = yaml.safe_load(export_boltz2_yaml(job))
        assert parsed["sequences"][0]["protein"]["id"] == prot_cid
        assert isinstance(parsed["sequences"][0]["protein"]["id"], str)
        assert parsed["sequences"][1]["ligand"]["id"] == lig_cid
        assert isinstance(parsed["sequences"][1]["ligand"]["id"], str)
        assert parsed["constraints"][0]["pocket"]["binder"] == lig_cid
        assert isinstance(parsed["constraints"][0]["pocket"]["binder"], str)
        assert parsed["constraints"][0]["pocket"]["contacts"][0][0] == prot_cid
        assert isinstance(parsed["constraints"][0]["pocket"]["contacts"][0][0], str)
        assert parsed["constraints"][1]["bond"]["atom1"][0] == prot_cid
        assert isinstance(parsed["constraints"][1]["bond"]["atom1"][0], str)
        assert parsed["constraints"][1]["bond"]["atom2"][0] == lig_cid
        assert isinstance(parsed["constraints"][1]["bond"]["atom2"][0], str)
        assert parsed["properties"][0]["affinity"]["binder"] == lig_cid
        assert isinstance(parsed["properties"][0]["affinity"]["binder"], str)


def test_sanitization_bonded_atom_pairs_pocket_pdb_and_pocket_center() -> None:
    """Validate bonded_atom_pairs, reject NUL in pocket_pdb, and forward pocket_center."""
    from types import SimpleNamespace

    # Invalid residue indices (<= 0) and unsafe/empty atom names blocked
    with pytest.raises(ValueError, match="residue index"):
        FoldingJobSpec(
            protein_chains=(("A", "MKTAYIAK"),),
            ligand_smiles=(("B", "CCO"),),
            bonded_atom_pairs=((("A", 0, "CA"), ("B", 1, "C1")),),
        ).validated()

    with pytest.raises(ValueError, match="residue index"):
        FoldingJobSpec(
            protein_chains=(("A", "MKTAYIAK"),),
            ligand_smiles=(("B", "CCO"),),
            bonded_atom_pairs=((("A", 1, "CA"), ("B", -5, "C1")),),
        ).validated()

    for bad_atom in ("", "   ", "C\nBAD", "CA\x00", "C;rm"):
        with pytest.raises(ValueError, match="Invalid atom name"):
            FoldingJobSpec(
                protein_chains=(("A", "MKTAYIAK"),),
                ligand_smiles=(("B", "CCO"),),
                bonded_atom_pairs=((("A", 1, bad_atom), ("B", 1, "C1")),),
            ).validated()

    # NUL byte in pocket_pdb rejected; valid multi-line PDB text still allowed
    with pytest.raises(ValueError, match="NUL"):
        FoldingJobSpec(
            ligand_smiles=(("B", "CCO"),),
            pocket_pdb="/tmp/evil\x00.pdb",
        ).validated()

    multiline_pdb = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    )
    ok_spec = FoldingJobSpec(
        ligand_smiles=(("B", "CCO"),),
        pocket_pdb=multiline_pdb,
    ).validated()
    assert ok_spec.pocket_pdb == multiline_pdb

    # Verify pocket_center forwarding in NanoFold3Backend.run -> model.cofold
    captured: dict[str, Any] = {}

    class _CapturingNanoFold3:
        def cofold(self, pocket_pdb: Any, ligand_smiles: Any, **kwargs: Any) -> Any:
            captured["pocket_pdb"] = pocket_pdb
            captured["ligand_smiles"] = ligand_smiles
            captured.update(kwargs)
            return SimpleNamespace(
                backend="nanofold3",
                name="cap",
                pdb_text=multiline_pdb,
                pocket_pdb_text=multiline_pdb,
                ligand_mol_3d=None,
                ligand_smiles=ligand_smiles,
                plddt_per_atom=(85.0,),
                plddt_mean=85.0,
                ligand_plddt=85.0,
                pae_matrix=None,
                pde_matrix=None,
                ptm=0.8,
                iptm=0.75,
                ranking_score=0.76,
                predicted_pkd=7.1,
                binding_prob=0.88,
                runtime_sec=0.01,
                provenance_sha256="",
                pose_audit=None,
                metadata={},
            )

    res = fold_complex(
        pocket_pdb=multiline_pdb,
        ligand_smiles="CCO",
        backend="nanofold3",
        run_audit=False,
        pocket_center=(1.5, -2.0, 3.25),
        model=_CapturingNanoFold3(),
    )
    assert captured["pocket_center"] == (1.5, -2.0, 3.25)
    assert res.predicted_pkd == pytest.approx(7.1)
    assert res.binding_prob == pytest.approx(0.88)
