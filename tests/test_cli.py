"""Offline CI-safe CLI tests for GenEuclidean v3.1 CNS and 3D visualization subcommands."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from typer.testing import CliRunner

from geneuclidean.cli import app
from geneuclidean.cns import REFERENCE_CNS_DRUGS, REFERENCE_NON_CNS_DRUGS, CNSAdmetSuite

runner = CliRunner()


def _write_mini_3d_complex(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal 3D pocket PDB + 3D ligand SDF with Asp/Phe contacts for CLI viz3d."""
    pocket_pdb = tmp_path / "mini_drd1_pocket.pdb"
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
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mol = Chem.AddHs(Chem.MolFromSmiles("CNCCc1ccccc1"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    mol = Chem.RemoveHs(mol)
    ligand_sdf = tmp_path / "mini_ligand.sdf"
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(mol)
    writer.close()
    return pocket_pdb, ligand_sdf


def _fit_tiny_admet_bundle(path: Path) -> Path:
    """Train a tiny deterministic CNSAdmetSuite in <1s for CLI ADMET integration testing."""
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


def test_cli_cns_targets_catalog_detail_and_json(tmp_path: Path) -> None:
    """`geneuclidean cns-targets` renders all 12 targets, detail view, and valid JSON."""
    res_all = runner.invoke(app, ["cns-targets", "--structures-dir", str(tmp_path)])
    assert res_all.exit_code == 0, res_all.stdout
    assert "DRD1" in res_all.stdout
    assert "CHEMBL1893" in res_all.stdout
    assert "CHEMBL5857" in res_all.stdout
    assert "5SJX" in res_all.stdout

    res_detail = runner.invoke(
        app,
        ["cns-targets", "--target", "DRD1", "--structures-dir", str(tmp_path)],
    )
    assert res_detail.exit_code == 0, res_detail.stdout
    assert "P21728" in res_detail.stdout
    assert "CHEMBL2056" in res_detail.stdout
    assert "Asp103" in res_detail.stdout
    assert "Ecopipam" in res_detail.stdout
    # Brackets in isomeric SMILES ([C@H]) must survive Rich rendering unstripped
    assert "[C@H]" in res_detail.stdout

    res_json = runner.invoke(
        app,
        [
            "cns-targets",
            "--indication",
            "Tourette",
            "--structures-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert res_json.exit_code == 0, res_json.stdout
    data = json.loads(res_json.stdout)
    assert len(data["targets"]) >= 6
    target_ids = {t["target_id"] for t in data["targets"]}
    assert "DRD1" in target_ids and "VMAT2" in target_ids

    res_bad = runner.invoke(app, ["cns-targets", "--target", "UNKNOWN_TARGET_99"])
    assert res_bad.exit_code != 0


def test_cli_cns_admet_offline_mpo_only_and_with_oracle(tmp_path: Path) -> None:
    """`geneuclidean cns-admet` works gracefully offline (MPO-only) and with ADMET joblib."""
    missing_oracle = tmp_path / "missing_admet.joblib"
    out_json = tmp_path / "admet_out.json"
    smiles_file = tmp_path / "query.smi"
    smiles_file.write_text(
        f"# Query drugs\n{REFERENCE_CNS_DRUGS['ecopipam']} ecopipam\n"
        f"{REFERENCE_NON_CNS_DRUGS['metformin']} metformin\n",
        encoding="utf-8",
    )

    # 1. Offline CI fallback (no .joblib present) — Rich table + file output
    res_table = runner.invoke(
        app,
        [
            "cns-admet",
            REFERENCE_CNS_DRUGS["xanomeline"],
            "--smiles-file",
            str(smiles_file),
            "--oracle-path",
            str(missing_oracle),
            "--out-json",
            str(out_json),
        ],
    )
    assert res_table.exit_code == 0, res_table.stdout
    assert "[C@H]" in res_table.stdout
    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["admet_oracles_loaded"] is False
    assert saved["n_molecules"] == 3
    assert saved["results"][0]["cns_mpo_v2"] > 5.0
    assert saved["results"][0]["bbb_prob"] is None

    # 2. Full ADMET mode with tiny synthetic bundle + --json stdout
    bundle_path = _fit_tiny_admet_bundle(tmp_path / "mini_admet.joblib")
    res_full = runner.invoke(
        app,
        [
            "cns-admet",
            REFERENCE_CNS_DRUGS["xanomeline"],
            "NOT_A_SMILES",
            "--oracle-path",
            str(bundle_path),
            "--json",
        ],
    )
    assert res_full.exit_code == 0, res_full.stdout
    parsed = json.loads(res_full.stdout)
    assert parsed["admet_oracles_loaded"] is True
    assert parsed["results"][0]["valid"] is True
    assert 0.0 <= parsed["results"][0]["bbb_prob"] <= 1.0
    assert 0.0 <= parsed["results"][0]["pgp_prob"] <= 1.0
    assert 0.0 <= parsed["results"][0]["herg_prob"] <= 1.0
    assert 0.0 <= parsed["results"][0]["ad_tanimoto_min"] <= 1.0
    assert parsed["results"][1]["valid"] is False
    assert parsed["results"][1]["composite_reward"] == 0.0

    # 3. Error when no SMILES provided
    res_empty = runner.invoke(app, ["cns-admet"])
    assert res_empty.exit_code != 0


def test_cli_viz3d_interactive_html_and_static_png(tmp_path: Path) -> None:
    """`geneuclidean viz3d` profiles 3D interactions and writes interactive HTML + 300-DPI PNG."""
    pocket_pdb, ligand_sdf = _write_mini_3d_complex(tmp_path)
    out_html = tmp_path / "complex.html"
    out_png = tmp_path / "complex_light.png"

    res = runner.invoke(
        app,
        [
            "viz3d",
            "--pocket",
            str(pocket_pdb),
            "--ligand",
            str(ligand_sdf),
            "--out-html",
            str(out_html),
            "--out-png",
            str(out_png),
            "--theme",
            "light",
            "--title",
            "DRD1 Mini Complex",
            "--pdb-id",
            "9LLG",
            "--disease",
            "Tourette Syndrome",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out_html.is_file()
    html_text = out_html.read_text(encoding="utf-8")
    assert "3Dmol" in html_text
    assert "DRD1 Mini Complex" in html_text

    assert out_png.is_file()
    png_bytes = out_png.read_bytes()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 5_000

    # Invalid theme rejected cleanly
    res_bad = runner.invoke(
        app,
        [
            "viz3d",
            "--pocket",
            str(pocket_pdb),
            "--ligand",
            str(ligand_sdf),
            "--out-html",
            str(tmp_path / "bad.html"),
            "--theme",
            "neon",
        ],
    )
    assert res_bad.exit_code != 0
