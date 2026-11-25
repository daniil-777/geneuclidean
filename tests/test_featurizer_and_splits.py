"""Unit tests for 52-dim E(3)-invariant pocket featurization, leak-proof splits, and datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from geneuclidean.data.dataset import (
    PocketLigandDataset,
    ZincSMILESDataset,
    collate_pocket_batch,
    collate_smiles_batch,
)
from geneuclidean.data.featurizer import (
    LIGAND_FP_DIM,
    POCKET_NODE_DIM,
    extract_pocket_from_files,
    featurize_pdb_pocket_inference,
)
from geneuclidean.data.splits import (
    compute_murcko_scaffold,
    make_casf2016_split,
    make_lp_pdbbind_split,
    make_random_split,
    make_scaffold_split,
    verify_split_integrity,
)
from geneuclidean.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, MolecularTokenizer


def _write_synthetic_pdb(
    path: Path,
    atoms: list[tuple[str, str, int, np.ndarray, str]],
) -> Path:
    """Write a minimal valid PDB file from (atom_name, resname, resseq, xyz, element)."""
    lines: list[str] = []
    for serial, (aname, resname, resseq, xyz, elem) in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {serial:5d} {aname:<4s} {resname:>3s} A{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {elem:>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_synthetic_ligand_sdf(
    path: Path,
    smiles: str,
    transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Generate a 3D ligand SDF, optionally rigidly transformed by (R, t)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    mol = Chem.RemoveHs(mol)
    if transform is not None:
        rot, trans = transform
        conf = mol.GetConformer()
        for idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(idx)
            vec = np.array([pos.x, pos.y, pos.z], dtype=np.float64) @ rot.T + trans
            conf.SetAtomPosition(idx, vec.tolist())
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _make_synthetic_pocket_atoms() -> list[tuple[str, str, int, np.ndarray, str]]:
    """Construct a diverse multi-residue 3D pocket around the origin."""
    rng = np.random.default_rng(7)
    specs = [
        ("N", "ASP", 10, "N"),
        ("CA", "ASP", 10, "C"),
        ("C", "ASP", 10, "C"),
        ("O", "ASP", 10, "O"),
        ("CG", "ASP", 10, "C"),
        ("OD1", "ASP", 10, "O"),
        ("OD2", "ASP", 10, "O"),
        ("N", "PHE", 11, "N"),
        ("CA", "PHE", 11, "C"),
        ("CG", "PHE", 11, "C"),
        ("CD1", "PHE", 11, "C"),
        ("CE1", "PHE", 11, "C"),
        ("CZ", "PHE", 11, "C"),
        ("N", "LYS", 12, "N"),
        ("NZ", "LYS", 12, "N"),
        ("SD", "MET", 13, "S"),
        ("OH", "TYR", 14, "O"),
        ("ZN", "ZN", 99, "ZN"),
        ("CA", "ALA", 15, "C"),
        ("CB", "VAL", 16, "C"),
    ]
    atoms: list[tuple[str, str, int, np.ndarray, str]] = []
    for i, (aname, resname, resseq, elem) in enumerate(specs):
        # Mix close contact atoms (<4.5 A) and shell atoms (5..9 A) with fractional coordinates
        radius = 2.8 + (i % 7) * 0.95
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        xyz = (direction * radius).astype(np.float64)
        atoms.append((aname, resname, resseq, xyz, elem))
    return atoms


@pytest.mark.equivariance
def test_pocket_featurizer_52d_e3_invariance_and_float32(tmp_path: Path) -> None:
    """All 52 node features must be strictly E(3)-invariant and coords exact float32 (not int)."""
    atoms = _make_synthetic_pocket_atoms()
    pdb_orig = _write_synthetic_pdb(tmp_path / "orig_pocket.pdb", atoms)
    sdf_orig = _write_synthetic_ligand_sdf(tmp_path / "orig_ligand.sdf", "CC(=O)Oc1ccccc1C(=O)O")

    # Random proper rotation R in SO(3) + large translation t
    rng = np.random.default_rng(123)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    trans = np.array([123.456, -78.912, 45.678], dtype=np.float64)

    rot_atoms = [
        (aname, resname, resseq, xyz @ q.T + trans, elem)
        for aname, resname, resseq, xyz, elem in atoms
    ]
    pdb_rot = _write_synthetic_pdb(tmp_path / "rot_pocket.pdb", rot_atoms)
    sdf_rot = _write_synthetic_ligand_sdf(
        tmp_path / "rot_ligand.sdf",
        "CC(=O)Oc1ccccc1C(=O)O",
        transform=(q, trans),
    )

    f1 = extract_pocket_from_files(pdb_orig, ligand_sdf=sdf_orig, pkd=7.25)
    f2 = extract_pocket_from_files(pdb_rot, ligand_sdf=sdf_rot, pkd=7.25)

    assert f1.node_feats.shape == (len(atoms), POCKET_NODE_DIM)
    assert f1.coords.dtype == np.float32
    assert f1.node_feats.dtype == np.float32
    assert np.all(np.isfinite(f1.node_feats))
    # Verify non-integer fractional coordinates (regression check against 2021 .long() bug!)
    assert np.any(np.abs(f1.coords - np.round(f1.coords)) > 0.05)

    # One-hot block sanity: elements [0:10], residues [10:31], backbone roles [31:36]
    np.testing.assert_allclose(f1.node_feats[:, 0:10].sum(axis=1), 1.0)
    np.testing.assert_allclose(f1.node_feats[:, 10:31].sum(axis=1), 1.0)
    np.testing.assert_allclose(f1.node_feats[:, 31:36].sum(axis=1), 1.0)

    # E(3) invariance of node_feats & contact_mask; SO(3) equivariance of centered coords
    # (PDB coordinates round to 0.001 A, so tolerance 2e-3 accounts for PDB format rounding)
    np.testing.assert_allclose(f2.node_feats, f1.node_feats, atol=2e-3)
    np.testing.assert_array_equal(f2.contact_mask, f1.contact_mask)
    np.testing.assert_allclose(f2.coords, f1.coords @ q.T.astype(np.float32), atol=3e-3)

    assert f1.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert f1.ligand_fp is not None and f1.ligand_fp.shape == (LIGAND_FP_DIM,)
    assert f1.ligand_props is not None and f1.ligand_props.shape == (6,)
    assert f1.pkd == pytest.approx(7.25)


def test_elongated_ligand_shell_and_fallback_smiles(tmp_path: Path) -> None:
    """Pocket shell checks distance to ANY ligand atom (not centroid) with fallback SMILES."""
    # Two protein clusters at x=-12 and x=+12 (both >10A from origin, but <3A from ligand ends!)
    atoms: list[tuple[str, str, int, np.ndarray, str]] = []
    for i in range(10):
        atoms.append(("CA", "ALA", i + 1, np.array([-12.0, i * 0.5, 0.0]), "C"))
        atoms.append(("CA", "GLY", i + 20, np.array([12.0, i * 0.5, 0.0]), "C"))
    # Far decoy atom 30A away should be excluded by pocket_cutoff=10.0
    atoms.append(("CA", "LEU", 99, np.array([0.0, 40.0, 0.0]), "C"))
    pdb_path = _write_synthetic_pdb(tmp_path / "elong_pocket.pdb", atoms)

    # Create a MOL2 with two heavy atoms at x=-10 and x=+10 (centroid at origin!)
    mol2_path = tmp_path / "elong_ligand.mol2"
    mol2_path.write_text(
        "@<TRIPOS>MOLECULE\nELONG\n 2 1 0 0 0\nSMALL\nGASTEIGER\n\n"
        "@<TRIPOS>ATOM\n"
        "  1 C1 -10.000 0.000 0.000 C.3 1 LIG 0.000\n"
        "  2 C2  10.000 0.000 0.000 C.3 1 LIG 0.000\n"
        "@<TRIPOS>BOND\n"
        "  1 1 2 1\n",
        encoding="utf-8",
    )

    feat = extract_pocket_from_files(
        pdb_path,
        ligand_sdf=tmp_path / "nonexistent.sdf",
        ligand_mol2=mol2_path,
        fallback_smiles="CCCCCC",
        pocket_cutoff=10.0,
        max_atoms=18,
    )
    # Far atom at y=40 must be excluded; capping at max_atoms=18 retains closest 18 atoms
    assert feat.coords.shape == (18, 3)
    assert np.all(feat.coords[:, 1] < 10.0)
    assert feat.contact_mask.any()
    assert feat.canonical_smiles is not None


def test_ligand_free_inference_featurizer(tmp_path: Path) -> None:
    """featurize_pdb_pocket_inference works with explicit center and with center=None."""
    atoms = _make_synthetic_pocket_atoms()
    pdb_path = _write_synthetic_pdb(tmp_path / "novel_target.pdb", atoms)

    f_auto = featurize_pdb_pocket_inference(pdb_path, center=None, pocket_cutoff=10.0)
    assert f_auto.coords.shape[1] == 3
    assert f_auto.node_feats.shape == (len(f_auto.coords), POCKET_NODE_DIM)
    assert f_auto.canonical_smiles is None
    assert f_auto.ligand_fp is None

    f_centered = featurize_pdb_pocket_inference(
        pdb_path, center=[0.0, 0.0, 0.0], pocket_cutoff=10.0
    )
    assert f_centered.node_feats.shape[1] == POCKET_NODE_DIM


def test_raw_holo_pdb_strips_organic_hetatms_keeps_mse_and_metals(tmp_path: Path) -> None:
    """Raw holo PDB parsing must strip organic ligand/lipid/buffer HETATMs without leakage."""
    protein_atoms = _make_synthetic_pocket_atoms()
    # Keep polymer residues only (exclude helper's ATOM ZN so we test HETATM ZN directly)
    polymer_atoms = [a for a in protein_atoms if a[1] != "ZN"]
    lines: list[str] = []
    serial = 1
    for aname, resname, resseq, xyz, elem in polymer_atoms:
        lines.append(
            f"ATOM  {serial:5d} {aname:<4s} {resname:>3s} A{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {elem:>2s}"
        )
        serial += 1
    # Modified polymer residue selenomethionine (MSE) as HETATM -> MUST BE KEPT
    lines.append(
        f"HETATM{serial:5d} SE   MSE A 101    "
        f"{2.100:8.3f}{0.500:8.3f}{-0.500:8.3f}  1.00 20.00          SE"
    )
    serial += 1
    # Catalytic zinc ion (ZN) as HETATM -> MUST BE KEPT
    lines.append(
        f"HETATM{serial:5d} ZN   ZN  A 201    "
        f"{1.800:8.3f}{-0.400:8.3f}{0.600:8.3f}  1.00 20.00          ZN"
    )
    serial += 1
    # Co-crystal organic ligand (8NU, e.g. risperidone), cholesterol lipid (CLR), glycerol (GOL),
    # and water (HOH) right inside the binding pocket -> ALL MUST BE STRIPPED!
    for aname, resname, resseq, xyz, elem in [
        ("C1", "8NU", 301, (0.0, 0.0, 0.0), "C"),
        ("N1", "8NU", 301, (1.2, 0.0, 0.0), "N"),
        ("F1", "8NU", 301, (0.0, 1.2, 0.0), "F"),
        ("C1", "CLR", 302, (2.0, 2.0, 0.0), "C"),
        ("O1", "GOL", 303, (1.5, -1.5, 0.0), "O"),
        ("O", "HOH", 401, (0.5, 0.5, 0.5), "O"),
    ]:
        lines.append(
            f"HETATM{serial:5d} {aname:<4s} {resname:>3s} A{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {elem:>2s}"
        )
        serial += 1
    lines.append("END")

    holo_pdb = tmp_path / "raw_holo_drd2.pdb"
    holo_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    feat = featurize_pdb_pocket_inference(holo_pdb, center=[0.0, 0.0, 0.0], pocket_cutoff=15.0)
    # Expect exactly polymer_atoms + 1 MSE atom + 1 ZN atom (0 ligand/lipid/buffer/water atoms!)
    assert feat.coords.shape == (len(polymer_atoms) + 2, 3)
    # Element index 5 is Fluorine (F from 8NU ligand) -> must be strictly zero
    assert float(feat.node_feats[:, 5].sum()) == 0.0
    # Element index 8 is Metal (ZN HETATM) -> must be exactly 1
    assert float(feat.node_feats[:, 8].sum()) == 1.0
    # MSE maps via _RESIDUE_ALIASES to MET (residue index 12 -> col 10 + 12 = 22);
    # polymer_atoms already has 1 MET atom + 1 MSE atom = 2 MET atoms
    assert float(feat.node_feats[:, 22].sum()) == 2.0


def test_leak_proof_splits(tmp_path: Path) -> None:
    """Scaffold, CASF-2016, LP-PDBBind, and random splits guarantee zero leakage."""
    smiles_map = {
        "p001": "c1ccccc1C(=O)O",
        "p002": "c1ccccc1CC(=O)O",
        "p003": "c1ccncc1C",
        "p004": "c1ccncc1CC",
        "p005": "c1ccc2ccccc2c1C",
        "p006": "c1ccc2[nH]ccc2c1C",
        "p007": "C1CCCCC1O",
        "p008": "C1CCNCC1C",
        "p009": "c1cncnc1N",
        "p010": "c1ccoc1C(=O)O",
    }
    ids = list(smiles_map.keys())

    scaf = make_scaffold_split(ids, smiles_map, train_frac=0.6, val_frac=0.2, seed=42)
    counts = verify_split_integrity(scaf, smiles_map=smiles_map, check_scaffold_disjoint=True)
    assert sum(counts.values()) == len(ids)
    # p001 and p002 share benzene scaffold -> must stay in the exact same split
    assert compute_murcko_scaffold(smiles_map["p001"]) == compute_murcko_scaffold(
        smiles_map["p002"]
    )
    for part in scaf.values():
        if "p001" in part:
            assert "p002" in part

    casf_ids = ["p009", "p010"]
    casf_split = make_casf2016_split(ids, casf_ids, smiles_map, val_frac=0.25, seed=42)
    verify_split_integrity(casf_split, casf2016_ids=casf_ids)
    assert set(casf_split["test"]) == set(casf_ids)

    lp_df = pd.DataFrame(
        {
            "new_split": [
                "train",
                "train",
                "train",
                "train",
                "val",
                "val",
                "test",
                "test",
                "train",  # CASF ID intentionally placed in train to verify purging!
                "val",
            ],
            "value": [6.0] * 10,
        },
        index=ids,
    )
    lp_csv = tmp_path / "LP_PDBBind.csv"
    lp_df.to_csv(lp_csv)
    lp_split = make_lp_pdbbind_split(
        ids, lp_csv, casf2016_ids=casf_ids, exclude_casf2016_from_train_val=True
    )
    verify_split_integrity(lp_split, casf2016_ids=casf_ids)
    assert "p009" not in lp_split["train"]
    assert "p010" not in lp_split["val"]

    rand_a = make_random_split(ids, seed=99)
    rand_b = make_random_split(ids, seed=99)
    assert rand_a == rand_b
    verify_split_integrity(rand_a)


def test_pocket_and_zinc_datasets_and_collate_contract(tmp_path: Path) -> None:
    """PocketLigandDataset and ZincSMILESDataset emit the shared GenEuclidean v2 batch dict."""
    smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "O=C([O-])c1cc(Cl)ccc1Br"]
    tok = MolecularTokenizer("smiles").fit(smiles_list, randomized_passes=1)

    records = {}
    for idx, smi in enumerate(smiles_list):
        pid = f"1a0{idx}"
        n_a = 20 + idx * 5
        contact = np.zeros(n_a, dtype=np.bool_)
        contact[:8] = True
        fp_packed = np.packbits(np.zeros(LIGAND_FP_DIM, dtype=np.uint8))
        records[pid] = {
            "pdb_id": pid,
            "coords": np.random.default_rng(idx).normal(size=(n_a, 3)).astype(np.float32),
            "node_feats": np.ones((n_a, POCKET_NODE_DIM), dtype=np.float32),
            "contact_mask": contact,
            "center": np.zeros(3, dtype=np.float32),
            "canonical_smiles": smi,
            "ligand_fp_packed": fp_packed,
            "ligand_props": np.array([180.0, 1.2, 60.0, 0.6, 1.0, 4.0], dtype=np.float32),
            "pkd": 6.5 + idx,
        }

    cache_path = tmp_path / "pdbbind_pockets.pt"
    torch.save(
        {
            "prop_mean": np.zeros(6, dtype=np.float32),
            "prop_std": np.ones(6, dtype=np.float32),
            "records": records,
        },
        cache_path,
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "lp_pdbbind": {"train": list(records.keys()), "val": ["1a00"], "test": ["1a02"]},
                "casf2016_core_ids": ["1a02"],
            }
        ),
        encoding="utf-8",
    )

    ds = PocketLigandDataset(
        cache_path,
        splits_path,
        tok,
        split_type="lp_pdbbind",
        split_name="train",
        randomize_smiles=True,
        coord_jitter_std=0.1,
        atom_drop_prob=0.2,
        seed=42,
    )
    batch = collate_pocket_batch([ds[0], ds[1], ds[2]])

    expected_keys = {
        "pdb_ids",
        "coords",
        "node_feats",
        "atom_mask",
        "contact_mask",
        "input_ids",
        "target_ids",
        "token_mask",
        "pkd",
        "ligand_fp",
        "ligand_props",
        "canonical_smiles",
    }
    assert set(batch.keys()) == expected_keys
    bsz, n_atoms, _ = batch["coords"].shape
    seq_len = batch["input_ids"].shape[1]
    assert bsz == 3
    assert batch["coords"].dtype == torch.float32
    assert batch["node_feats"].shape == (3, n_atoms, POCKET_NODE_DIM)
    assert batch["atom_mask"].dtype == torch.bool
    assert batch["contact_mask"].dtype == torch.bool
    # Peripheral-only dropout must retain all 8 contact atoms per pocket
    assert int(batch["contact_mask"].sum()) == 24
    # Padded atom slots must have atom_mask == False and contact_mask == False
    assert not bool(batch["contact_mask"][~batch["atom_mask"]].any())
    assert batch["input_ids"].shape == (3, seq_len)
    assert batch["target_ids"].shape == (3, seq_len)
    assert batch["token_mask"].shape == (3, seq_len)
    assert torch.all(batch["input_ids"][:, 0] == BOS_ID)
    assert torch.all(batch["target_ids"][~batch["token_mask"]] == PAD_ID)
    # Every active sequence must end with EOS_ID
    for i in range(3):
        last_pos = int(batch["token_mask"][i].sum().item()) - 1
        assert int(batch["target_ids"][i, last_pos].item()) == EOS_ID
    assert batch["pkd"].shape == (3,)
    assert batch["ligand_fp"].shape == (3, LIGAND_FP_DIM)
    assert batch["ligand_props"].shape == (3, 6)

    # Unconditional ZincSMILESDataset contract
    zds = ZincSMILESDataset(smiles_list, tok, max_seq_len=64, randomize_smiles=False)
    zbatch = collate_smiles_batch([zds[0], zds[1]])
    assert set(zbatch.keys()) == {"input_ids", "target_ids", "token_mask"}
    assert zbatch["input_ids"].shape == zbatch["target_ids"].shape == zbatch["token_mask"].shape


def test_real_pdbbind_complex_if_present() -> None:
    """Smoke-test extract_pocket_from_files on real PDBbind complex 10gs when downloaded."""
    pdir = Path("data/raw/pbpp2020/10gs")
    if not pdir.exists():
        pytest.skip("data/raw/pbpp2020/10gs not present")
    feat = extract_pocket_from_files(
        pdir / "10gs_pocket.pdb",
        ligand_sdf=pdir / "10gs_ligand.sdf",
        ligand_mol2=pdir / "10gs_ligand.mol2",
        pdb_id="10gs",
        pkd=6.40,
    )
    assert feat.coords.shape[1] == 3
    assert feat.node_feats.shape == (len(feat.coords), POCKET_NODE_DIM)
    assert 30 <= len(feat.coords) <= 384
    assert feat.contact_mask.sum() >= 10
    assert feat.canonical_smiles is not None
    assert Chem.MolFromSmiles(feat.canonical_smiles) is not None
