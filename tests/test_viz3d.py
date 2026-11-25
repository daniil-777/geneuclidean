"""Comprehensive unit, geometry, live-PDB, and rendering tests for interactions3d & viz3d."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from geneuclidean.evaluation.interactions3d import (
    ALL_INTERACTION_KINDS,
    InteractionProfile,
    InteractionRecord,
    compute_heavy_atom_rmsd,
    profile_3d_interactions,
    profile_3d_interactions_from_pdb_text,
)
from geneuclidean.evaluation.viz3d import (
    ComplexShowcaseSpec,
    render_interactive_3d_gallery_html,
    render_interactive_3d_html,
    render_static_3d_complex_png,
    render_static_3d_molecule_png,
)

# ---------------------------------------------------------------------------
# Synthetic 3D Molecular & PDB Fixture Builders
# ---------------------------------------------------------------------------


def _pdb_atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    """Format a strictly fixed-width PDB ATOM record."""
    atom_field = f" {name:<3s}" if len(name) < 4 else name[:4]
    return (
        f"ATOM  {serial:5d} {atom_field} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def _mol_with_exact_coords(smiles: str, coords: list[tuple[float, float, float]]) -> Chem.Mol:
    """Create an RDKit 3D Mol with exact heavy-atom coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumAtoms() == len(coords)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


def _regular_hexagon_coords(
    center: tuple[float, float, float],
    radius: float = 1.39,
    plane: str = "xy",
) -> list[tuple[float, float, float]]:
    """Generate 6 vertices of a regular benzene ring in the xy or xz plane."""
    cx, cy, cz = center
    pts: list[tuple[float, float, float]] = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        u = radius * math.cos(ang)
        v = radius * math.sin(ang)
        if plane == "xy":
            pts.append((cx + u, cy + v, cz))
        elif plane == "xz":
            pts.append((cx + u, cy, cz + v))
        else:
            raise ValueError(f"Unsupported plane: {plane}")
    return pts


def _make_phe_pdb(center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    """Build a synthetic PHE288:A PDB ring in the xy plane centered at ``center``."""
    hex_pts = _regular_hexagon_coords(center, radius=1.39, plane="xy")
    # Order matches PHE ring connectivity: CG, CD1, CE1, CZ, CE2, CD2
    names = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    lines = [
        _pdb_atom_line(1, "CA", "PHE", "A", 288, (center[0] - 3.6, center[1], center[2]), "C"),
        _pdb_atom_line(2, "CB", "PHE", "A", 288, (center[0] - 2.5, center[1], center[2]), "C"),
    ]
    for i, (nm, pt) in enumerate(zip(names, hex_pts, strict=True), start=3):
        lines.append(_pdb_atom_line(i, nm, "PHE", "A", 288, pt, "C"))
    lines.append("END\n")
    return "".join(lines)


def _embed_smiles_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    """Generate a deterministic 3D conformer using RDKit ETKDGv3."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _translate_mol(mol: Chem.Mol, shift: tuple[float, float, float]) -> Chem.Mol:
    """Return a copy of ``mol`` rigidly translated by ``shift``."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    dx, dy, dz = shift
    for idx in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(idx)
        conf.SetAtomPosition(idx, (p.x + dx, p.y + dy, p.z + dz))
    return out


def _rotate_mol(mol: Chem.Mol, rot: np.ndarray) -> Chem.Mol:
    """Return a copy of ``mol`` rigidly rotated by 3x3 matrix ``rot``."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for idx in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(idx)
        vec = rot @ np.array([p.x, p.y, p.z], dtype=np.float64)
        conf.SetAtomPosition(idx, (float(vec[0]), float(vec[1]), float(vec[2])))
    return out


# ---------------------------------------------------------------------------
# 1. Exact Geometric Unit Tests for interactions3d.py
# ---------------------------------------------------------------------------


def test_hbond_detection_explicit_h_angle_and_unprotonated_fallback() -> None:
    """Verify directional D-H...A H-bond detection, angle rejection, and no-H fallback."""
    # Formaldehyde C=O oxygen acceptor at (2.85, 0, 0)
    lig = _mol_with_exact_coords("C=O", [(4.05, 0.0, 0.0), (2.85, 0.0, 0.0)])

    # Case A: Linear explicit H-bond (SER10:A OG -> HG -> lig O, angle = 180 deg, dist = 2.85 A)
    pdb_linear = "".join(
        [
            _pdb_atom_line(1, "CB", "SER", "A", 10, (-1.42, 0.0, 0.0), "C"),
            _pdb_atom_line(2, "OG", "SER", "A", 10, (0.0, 0.0, 0.0), "O"),
            _pdb_atom_line(3, "HG", "SER", "A", 10, (0.96, 0.0, 0.0), "H"),
            "END\n",
        ]
    )
    prof_ok = profile_3d_interactions_from_pdb_text(pdb_linear, lig)
    assert prof_ok.counts_by_type["hbond"] == 1
    rec = prof_ok.interactions[0]
    assert rec.kind == "hbond"
    assert rec.protein_residue == "SER10:A"
    assert rec.protein_atom == "OG"
    assert rec.distance_angstrom == pytest.approx(2.85, abs=1e-3)
    assert rec.angle_deg == pytest.approx(180.0, abs=1e-2)

    # Case B: Explicit H pointing directly away (angle = 0 deg < 120 deg) -> rejected
    pdb_bad_angle = "".join(
        [
            _pdb_atom_line(1, "CB", "SER", "A", 10, (0.0, -1.42, 0.0), "C"),
            _pdb_atom_line(2, "OG", "SER", "A", 10, (0.0, 0.0, 0.0), "O"),
            _pdb_atom_line(3, "HG", "SER", "A", 10, (-0.96, 0.0, 0.0), "H"),
            "END\n",
        ]
    )
    prof_bad = profile_3d_interactions_from_pdb_text(pdb_bad_angle, lig)
    assert prof_bad.counts_by_type["hbond"] == 0

    # Case C: Unprotonated PDB without explicit H -> heavy-atom geometric fallback (<= 3.5 A)
    pdb_no_h = "".join(
        [
            _pdb_atom_line(1, "CB", "SER", "A", 10, (-1.42, 0.0, 0.0), "C"),
            _pdb_atom_line(2, "OG", "SER", "A", 10, (0.0, 0.0, 0.0), "O"),
            "END\n",
        ]
    )
    prof_noh = profile_3d_interactions_from_pdb_text(pdb_no_h, lig)
    assert prof_noh.counts_by_type["hbond"] == 1
    assert prof_noh.interactions[0].angle_deg is None


def test_salt_bridge_asp332_amine_anchor_and_amide_exclusion() -> None:
    """Verify physiological pH 7.4 ASP carboxylate <-> basic amine salt bridge detection."""
    # Conserved aminergic CNS GPCR Asp103 carboxylate anchor (both OD1 and OD2 within 4.2 A)
    pdb_asp = "".join(
        [
            _pdb_atom_line(1, "CB", "ASP", "A", 103, (-2.50, 0.0, 0.0), "C"),
            _pdb_atom_line(2, "CG", "ASP", "A", 103, (-1.00, 0.0, 0.0), "C"),
            _pdb_atom_line(3, "OD1", "ASP", "A", 103, (0.00, 0.0, 0.0), "O"),
            _pdb_atom_line(4, "OD2", "ASP", "A", 103, (-0.50, 1.15, 0.0), "O"),
            "END\n",
        ]
    )

    # Ethylamine (neutral SMILES, protonated cationic at pH 7.4) with N at (3.0, 0, 0)
    lig_amine = _mol_with_exact_coords("CCN", [(5.5, 0.0, 0.0), (4.4, 0.0, 0.0), (3.0, 0.0, 0.0)])
    prof = profile_3d_interactions_from_pdb_text(pdb_asp, lig_amine)

    sb = [r for r in prof.interactions if r.kind == "salt_bridge"]
    assert len(sb) == 1  # deduplicated across OD1/OD2 to closest oxygen
    assert sb[0].protein_residue == "ASP103:A"
    assert sb[0].protein_atom == "OD1"
    assert sb[0].ligand_element == "N"
    assert sb[0].distance_angstrom == pytest.approx(3.0, abs=1e-3)
    assert sb[0].subtype == "protein_anion"

    # Neutral acetamide CC(=O)N must NOT form a false-positive salt bridge
    lig_amide = _mol_with_exact_coords(
        "CC(=O)N",
        [(5.6, 0.0, 0.0), (4.3, 0.0, 0.0), (4.3, 1.2, 0.0), (3.0, 0.0, 0.0)],
    )
    prof_amide = profile_3d_interactions_from_pdb_text(pdb_asp, lig_amide)
    assert prof_amide.counts_by_type["salt_bridge"] == 0

    # Reverse polarity: Cationic ARG50:A <-> Ligand carboxylic acid CC(=O)O
    pdb_arg = "".join(
        [
            _pdb_atom_line(1, "CD", "ARG", "A", 50, (-3.2, 0.0, 0.0), "C"),
            _pdb_atom_line(2, "NE", "ARG", "A", 50, (-2.0, 0.0, 0.0), "N"),
            _pdb_atom_line(3, "CZ", "ARG", "A", 50, (-1.0, 0.0, 0.0), "C"),
            _pdb_atom_line(4, "NH1", "ARG", "A", 50, (0.0, 0.0, 0.0), "N"),
            _pdb_atom_line(5, "NH2", "ARG", "A", 50, (-0.8, 1.2, 0.0), "N"),
            "END\n",
        ]
    )
    lig_acid = _mol_with_exact_coords(
        "CC(=O)O",
        [(5.5, 0.0, 0.0), (4.2, 0.0, 0.0), (4.1, 1.2, 0.0), (3.1, 0.0, 0.0)],
    )
    prof_arg = profile_3d_interactions_from_pdb_text(pdb_arg, lig_acid)
    assert prof_arg.counts_by_type["salt_bridge"] >= 1
    assert any(
        r.kind == "salt_bridge" and r.protein_residue == "ARG50:A" for r in prof_arg.interactions
    )


def test_pi_stacking_parallel_t_shaped_and_coplanar_rejection() -> None:
    """Verify parallel-displaced & T-shaped π–π stacking and rejection of coplanar rings."""
    pdb_phe = _make_phe_pdb((0.0, 0.0, 0.0))

    # 1) Parallel-displaced benzene at (0.8, 0.0, 3.7)
    lig_parallel = _mol_with_exact_coords(
        "c1ccccc1", _regular_hexagon_coords((0.8, 0.0, 3.7), plane="xy")
    )
    prof_par = profile_3d_interactions_from_pdb_text(pdb_phe, lig_parallel)
    assert prof_par.counts_by_type["pi_stacking"] == 1
    rec_par = [r for r in prof_par.interactions if r.kind == "pi_stacking"][0]
    assert rec_par.protein_residue == "PHE288:A"
    assert rec_par.subtype == "parallel"
    assert rec_par.angle_deg == pytest.approx(0.0, abs=1e-2)
    assert rec_par.distance_angstrom == pytest.approx(math.hypot(0.8, 3.7), abs=1e-2)

    # 2) T-shaped (edge-to-face) benzene in xz plane at (0.0, 0.0, 4.9)
    lig_t = _mol_with_exact_coords("c1ccccc1", _regular_hexagon_coords((0.0, 0.0, 4.9), plane="xz"))
    prof_t = profile_3d_interactions_from_pdb_text(pdb_phe, lig_t)
    assert prof_t.counts_by_type["pi_stacking"] == 1
    rec_t = [r for r in prof_t.interactions if r.kind == "pi_stacking"][0]
    assert rec_t.subtype == "t_shaped"
    assert rec_t.angle_deg == pytest.approx(90.0, abs=1e-2)

    # 3) Coplanar side-by-side benzene rings (lateral offset 5.0 A > 2.8 A) -> rejected
    lig_coplanar = _mol_with_exact_coords(
        "c1ccccc1", _regular_hexagon_coords((5.0, 0.0, 0.0), plane="xy")
    )
    prof_flat = profile_3d_interactions_from_pdb_text(pdb_phe, lig_coplanar)
    assert prof_flat.counts_by_type["pi_stacking"] == 0


def test_cation_pi_hydrophobic_and_halogen_bond_geometry() -> None:
    """Verify cation–π, deduplicated hydrophobic contacts, and sigma-hole halogen bond rules."""
    # 1) Cation-π: Basic amine nitrogen 4.0 A directly over PHE288:A aromatic centroid
    pdb_phe = _make_phe_pdb((0.0, 0.0, 0.0))
    lig_cation = _mol_with_exact_coords("CN", [(0.0, 0.0, 5.45), (0.0, 0.0, 4.0)])
    prof_cat = profile_3d_interactions_from_pdb_text(pdb_phe, lig_cation)
    assert prof_cat.counts_by_type["cation_pi"] == 1
    cat_rec = [r for r in prof_cat.interactions if r.kind == "cation_pi"][0]
    assert cat_rec.protein_residue == "PHE288:A"
    assert cat_rec.distance_angstrom == pytest.approx(4.0, abs=1e-2)

    # 2) Hydrophobic: VAL20:A with two apolar sidechain carbons (CG1 at 3.4 A, CG2 at 3.8 A)
    #    both close to the same ligand methyl carbon -> deduplicated to closest (CG1)
    pdb_val = "".join(
        [
            _pdb_atom_line(1, "CA", "VAL", "A", 20, (-2.8, 0.0, 0.0), "C"),
            _pdb_atom_line(2, "CB", "VAL", "A", 20, (-1.4, 0.0, 0.0), "C"),
            _pdb_atom_line(3, "CG1", "VAL", "A", 20, (0.0, 0.0, 0.0), "C"),
            _pdb_atom_line(4, "CG2", "VAL", "A", 20, (-0.4, 1.3, 0.0), "C"),
            "END\n",
        ]
    )
    lig_alk = _mol_with_exact_coords("CC", [(3.4, 0.0, 0.0), (4.9, 0.0, 0.0)])
    prof_hyd = profile_3d_interactions_from_pdb_text(pdb_val, lig_alk)
    assert prof_hyd.counts_by_type["hydrophobic"] == 1
    assert prof_hyd.interactions[0].protein_atom == "CG1"
    assert prof_hyd.interactions[0].distance_angstrom == pytest.approx(3.4, abs=1e-2)

    # 3) Halogen bond: C-Cl...O linear (180 deg, 3.2 A) passes; bent (90 deg) fails
    pdb_ala = "".join(
        [
            _pdb_atom_line(1, "C", "ALA", "A", 30, (-1.23, 0.0, 0.0), "C"),
            _pdb_atom_line(2, "O", "ALA", "A", 30, (0.0, 0.0, 0.0), "O"),
            "END\n",
        ]
    )
    lig_cl_linear = _mol_with_exact_coords("CCl", [(4.95, 0.0, 0.0), (3.20, 0.0, 0.0)])
    prof_xb = profile_3d_interactions_from_pdb_text(pdb_ala, lig_cl_linear)
    assert prof_xb.counts_by_type["halogen_bond"] == 1
    xb_rec = prof_xb.interactions[0]
    assert xb_rec.protein_residue == "ALA30:A"
    assert xb_rec.ligand_element == "Cl"
    assert xb_rec.distance_angstrom == pytest.approx(3.20, abs=1e-2)
    assert xb_rec.angle_deg == pytest.approx(180.0, abs=1e-2)

    lig_cl_bent = _mol_with_exact_coords("CCl", [(3.20, 1.75, 0.0), (3.20, 0.0, 0.0)])
    prof_xb_bad = profile_3d_interactions_from_pdb_text(pdb_ala, lig_cl_bent)
    assert prof_xb_bad.counts_by_type["halogen_bond"] == 0


def test_plif_tanimoto_and_residue_recovery() -> None:
    """Verify exact PLIF Tanimoto self=1.0, disjoint=0.0, and partial overlap math."""
    r1 = InteractionRecord(
        kind="salt_bridge",
        protein_residue="ASP103:A",
        protein_atom="OD1",
        ligand_atom_idx=0,
        ligand_element="N",
        distance_angstrom=2.9,
        angle_deg=None,
        protein_xyz=(0.0, 0.0, 0.0),
        ligand_xyz=(2.9, 0.0, 0.0),
    )
    r2 = InteractionRecord(
        kind="hbond",
        protein_residue="SER193:A",
        protein_atom="OG",
        ligand_atom_idx=4,
        ligand_element="O",
        distance_angstrom=2.8,
        angle_deg=155.0,
        protein_xyz=(1.0, 0.0, 0.0),
        ligand_xyz=(3.8, 0.0, 0.0),
    )
    r3 = InteractionRecord(
        kind="pi_stacking",
        protein_residue="PHE288:A",
        protein_atom="RING",
        ligand_atom_idx=2,
        ligand_element="RING",
        distance_angstrom=4.1,
        angle_deg=12.0,
        protein_xyz=(0.0, 4.1, 0.0),
        ligand_xyz=(0.0, 0.0, 0.0),
    )

    p_ref = InteractionProfile(interactions=[r1, r2])
    p_gen = InteractionProfile(interactions=[r1, r3])
    p_disjoint = InteractionProfile(interactions=[r3])
    p_empty = InteractionProfile()

    assert p_ref.tanimoto_to(p_ref) == pytest.approx(1.0)
    assert p_empty.tanimoto_to(p_empty) == pytest.approx(1.0)
    assert p_ref.tanimoto_to(p_empty) == pytest.approx(0.0)
    assert p_ref.tanimoto_to(p_disjoint) == pytest.approx(0.0)
    # Intersection = {r1}, Union = {r1, r2, r3} -> 1/3
    assert p_ref.tanimoto_to(p_gen) == pytest.approx(1.0 / 3.0)
    assert p_gen.residue_recovery_rate(p_ref) == pytest.approx(0.5)


def test_heavy_atom_rmsd_self_translation_rotation_and_symmetry() -> None:
    """Verify heavy-atom RMSD is exact under identity, translation, rotation, and ring flip."""
    mol = _embed_smiles_3d("COc1ccc(C(=O)N2CCN(C)CC2)cc1", seed=7)

    # 1) Self RMSD = 0.0 (both in-place and aligned)
    assert compute_heavy_atom_rmsd(mol, mol) == pytest.approx(0.0, abs=1e-8)
    assert compute_heavy_atom_rmsd(mol, mol, align=True) == pytest.approx(0.0, abs=1e-6)

    # 2) Pure translation by (3, 4, 0) -> exact 5.0 A in-place, 0.0 A aligned
    shifted = _translate_mol(mol, (3.0, 4.0, 0.0))
    assert compute_heavy_atom_rmsd(mol, shifted, align=False) == pytest.approx(5.0, abs=1e-5)
    assert compute_heavy_atom_rmsd(mol, shifted, align=True) == pytest.approx(0.0, abs=1e-5)

    # 3) Exact rigid rotation on an asymmetric molecule (ethanol with distinct C-C-O)
    asym = _mol_with_exact_coords("CCO", [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.2, 1.2, 0.0)])
    theta = math.radians(45.0)
    rot_z = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    asym_rot = _rotate_mol(asym, rot_z)
    p0 = np.asarray(asym.GetConformer().GetPositions(), dtype=np.float64)
    p1 = p0 @ rot_z.T
    expected_rmsd = float(np.sqrt(np.mean(np.sum((p1 - p0) ** 2, axis=1))))
    assert compute_heavy_atom_rmsd(asym, asym_rot, align=False) == pytest.approx(
        expected_rmsd, abs=1e-6
    )
    assert compute_heavy_atom_rmsd(asym, asym_rot, align=True) == pytest.approx(0.0, abs=1e-6)

    # 4) Symmetry correction: 180-degree flip of a benzene ring around the x-axis swaps
    #    top/bottom carbons in atom order, yet symmetry-corrected in-place RMSD MUST be 0.0 A!
    benz = _mol_with_exact_coords("c1ccccc1", _regular_hexagon_coords((0.0, 0.0, 0.0), plane="xy"))
    rot_x_180 = np.diag([1.0, -1.0, -1.0])
    benz_flipped = _rotate_mol(benz, rot_x_180)
    assert compute_heavy_atom_rmsd(benz, benz_flipped, align=False) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Live Real Crystal Complex Test (Auto-Skipped Cleanly in CI Without data/raw)
# ---------------------------------------------------------------------------


def test_live_10gs_crystal_complex_interactions() -> None:
    """Verify real crystallographic contacts on PDBbind 10gs (GSTP1-1) when available."""
    pocket_path = Path("data/raw/pbpp2020/10gs/10gs_pocket.pdb")
    ligand_path = Path("data/raw/pbpp2020/10gs/10gs_ligand.sdf")
    if not (pocket_path.is_file() and ligand_path.is_file()):
        pytest.skip("data/raw/pbpp2020/10gs not available in this environment")

    suppl = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
    lig_mol = suppl[0]
    assert lig_mol is not None

    profile = profile_3d_interactions(pocket_path, lig_mol)
    assert set(profile.counts_by_type.keys()) == set(ALL_INTERACTION_KINDS)
    assert profile.counts_by_type["hbond"] >= 4
    assert profile.counts_by_type["salt_bridge"] >= 1
    assert profile.counts_by_type["hydrophobic"] >= 1
    assert "TYR7:A" in profile.key_residues  # catalytic Tyr7
    assert "ASP98:B" in profile.key_residues  # canonical Asp98 salt bridge
    assert profile.tanimoto_to(profile) == pytest.approx(1.0)
    assert compute_heavy_atom_rmsd(lig_mol, lig_mol) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. Visualizer Smoke & Integrity Tests (Static 3D PNG + Interactive WebGL HTML)
# ---------------------------------------------------------------------------


def _write_synthetic_pocket_and_ligand(tmp_path: Path) -> tuple[Path, Chem.Mol, Chem.Mol]:
    """Create a rich mini synthetic pocket PDB + 3D dopamine-like ligand & shifted native pose."""
    lines = [
        # Mini Cα backbone + ASP103:A carboxylate + PHE288:A aromatic ring + SER193:A hydroxyl
        _pdb_atom_line(1, "N", "ASP", "A", 103, (-4.1, -0.8, -1.2), "N"),
        _pdb_atom_line(2, "CA", "ASP", "A", 103, (-3.2, 0.1, -0.5), "C"),
        _pdb_atom_line(3, "C", "ASP", "A", 103, (-3.8, 1.4, -0.8), "C"),
        _pdb_atom_line(4, "O", "ASP", "A", 103, (-4.8, 1.5, -1.4), "O"),
        _pdb_atom_line(5, "CB", "ASP", "A", 103, (-1.8, 0.0, -0.2), "C"),
        _pdb_atom_line(6, "CG", "ASP", "A", 103, (-0.9, 0.1, 0.1), "C"),
        _pdb_atom_line(7, "OD1", "ASP", "A", 103, (0.2, 0.0, 0.0), "O"),
        _pdb_atom_line(8, "OD2", "ASP", "A", 103, (-1.1, 1.2, 0.5), "O"),
        _pdb_atom_line(9, "N", "VAL", "A", 104, (-3.1, 2.4, -0.2), "N"),
        _pdb_atom_line(10, "CA", "VAL", "A", 104, (-3.4, 3.8, -0.3), "C"),
        _pdb_atom_line(11, "CB", "VAL", "A", 104, (-2.3, 4.6, 0.3), "C"),
        _pdb_atom_line(12, "CG1", "VAL", "A", 104, (-1.1, 4.2, -0.3), "C"),
        _pdb_atom_line(13, "CG2", "VAL", "A", 104, (-2.7, 6.0, 0.1), "C"),
        _pdb_atom_line(14, "CA", "SER", "A", 193, (9.8, 1.8, -1.0), "C"),
        _pdb_atom_line(15, "CB", "SER", "A", 193, (8.6, 1.1, -0.4), "C"),
        _pdb_atom_line(16, "OG", "SER", "A", 193, (7.6, 0.5, 0.0), "O"),
    ]
    # Add PHE288:A aromatic ring stacked behind the ligand aromatic ring
    phe_pts = _regular_hexagon_coords((5.0, 0.2, -3.7), radius=1.39, plane="xy")
    lines.append(_pdb_atom_line(17, "CA", "PHE", "A", 288, (2.2, 0.2, -4.4), "C"))
    lines.append(_pdb_atom_line(18, "CB", "PHE", "A", 288, (3.2, 0.2, -4.1), "C"))
    for idx, (nm, pt) in enumerate(
        zip(["CG", "CD1", "CE1", "CZ", "CE2", "CD2"], phe_pts, strict=True), start=19
    ):
        lines.append(_pdb_atom_line(idx, nm, "PHE", "A", 288, pt, "C"))
    lines.append("END\n")

    pdb_path = tmp_path / "mini_drd2_pocket.pdb"
    pdb_path.write_text("".join(lines), encoding="utf-8")

    # Dopamine-like 3D ligand positioned inside the synthetic pocket
    lig = _embed_smiles_3d("NCCc1ccc(O)c(O)c1", seed=11)
    # Center ligand so its amine sits near (3.0, 0.0, 0.0) and catechol sits near (6.0, 0.0, 0.0)
    conf = lig.GetConformer()
    n_idx = next(a.GetIdx() for a in lig.GetAtoms() if a.GetAtomicNum() == 7)
    n_pos = np.array(
        [
            conf.GetAtomPosition(n_idx).x,
            conf.GetAtomPosition(n_idx).y,
            conf.GetAtomPosition(n_idx).z,
        ],
        dtype=np.float64,
    )
    lig_shifted = _translate_mol(
        lig, (3.0 - float(n_pos[0]), 0.0 - float(n_pos[1]), 0.0 - float(n_pos[2]))
    )
    native_ghost = _translate_mol(lig_shifted, (0.35, -0.25, 0.2))
    return pdb_path, lig_shifted, native_ghost


def test_render_static_3d_molecule_and_complex_dark_and_light(tmp_path: Path) -> None:
    """Smoke-test 300-DPI static 3D molecule & complex PNG renderers (dark and light themes)."""
    pdb_path, lig, native = _write_synthetic_pocket_and_ligand(tmp_path)
    metrics = {
        "vina_kcal": -9.45,
        "native_vina_kcal": -8.10,
        "cns_mpo": 5.35,
        "bbb_prob": 0.94,
        "pgp_minus_prob": 0.87,
        "herg_prob": 0.12,
        "qed": 0.84,
        "sa": 2.15,
        "posebusters_pass": True,
    }
    attn = {"ASP103:A": 0.96, "SER193:A": 0.81, "PHE288:A": 0.74, "VAL104:A": 0.42}

    mol_png = render_static_3d_molecule_png(
        lig,
        tmp_path / "mol_dark.png",
        title="De Novo Dopaminergic Hit",
        subtitle="ETKDGv3 + MMFF94s 3D Conformer",
        theme="dark",
        metrics=metrics,
    )
    assert mol_png.is_file()
    assert mol_png.stat().st_size > 20_000
    assert mol_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    for theme in ("dark", "light"):
        out_png = render_static_3d_complex_png(
            pdb_path,
            lig,
            tmp_path / f"complex_{theme}.png",
            native_ligand_mol=native,
            title="Human DRD2 Binding Complex",
            pdb_id="6CM4",
            disease="Tourette Syndrome / CNS",
            theme=theme,
            metrics=metrics,
            residue_attention=attn,
        )
        assert out_png.is_file()
        assert out_png.stat().st_size > 20_000
        assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_interactive_3d_html_and_gallery(tmp_path: Path) -> None:
    """Verify interactive single & gallery HTML files contain valid JSON, 3Dmol calls, and SVG."""
    pdb_path, lig, native = _write_synthetic_pocket_and_ligand(tmp_path)
    metrics = {
        "vina_kcal": -9.50,
        "native_vina_kcal": -8.29,
        "cns_mpo": 5.4,
        "bbb_prob": 0.93,
        "pgp_minus_prob": 0.89,
        "herg_prob": 0.09,
        "qed": 0.83,
        "sa": 2.15,
        "posebusters_pass": True,
    }
    attn = {"ASP103:A": 0.95, "PHE288:A": 0.78}

    # 1) Single-target interactive WebGL HTML
    single_html = render_interactive_3d_html(
        pdb_path,
        lig,
        tmp_path / "drd2_studio.html",
        native_ligand_mol=native,
        title="Dopamine D2 Receptor (DRD2)",
        pdb_id="6CM4",
        disease="Tourette Syndrome",
        metrics=metrics,
        residue_attention=attn,
        notes="Conserved Asp3.32 salt bridge + aromatic sandwich verified.",
    )
    assert single_html.is_file()
    html_text = single_html.read_text(encoding="utf-8")
    assert len(html_text.encode("utf-8")) > 5_000
    assert "3Dmol-min.js" in html_text
    assert "$3Dmol.createViewer" in html_text
    assert "addCylinder" in html_text

    # Parse embedded JSON payload and verify structural integrity
    match = re.search(
        r'<script id="geneuclidean-studio-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert len(payload["specs"]) == 1
    sp0 = payload["specs"][0]
    assert sp0["pdb_id"] == "6CM4"
    assert "ATOM" in sp0["pdb"]
    assert "M  END" in sp0["ligand_sdf"]
    assert "M  END" in sp0["native_sdf"]
    assert "<svg" in sp0["svg_2d"]
    assert sp0["profile"]["counts_by_type"]["salt_bridge"] >= 1

    # 2) Multi-target interactive gallery HTML
    specs = [
        ComplexShowcaseSpec(
            target_id="drd2_6cm4",
            title="Dopamine D2 Receptor (DRD2)",
            pdb_id="6CM4",
            disease="Tourette Syndrome",
            pocket_pdb_path=pdb_path,
            ligand_mol=lig,
            native_ligand_mol=native,
            residue_attention=attn,
            metrics=metrics,
        ),
        ComplexShowcaseSpec(
            target_id="htr2a_6a93",
            title="Serotonin 5-HT2A Receptor (HTR2A)",
            pdb_id="6A93",
            disease="Psychiatry / Tourette",
            pocket_pdb_path=pdb_path,
            ligand_mol=native,
            metrics={"vina_kcal": -8.85, "cns_mpo": 5.1, "qed": 0.79, "posebusters_pass": True},
        ),
    ]
    gallery_html = render_interactive_3d_gallery_html(
        specs,
        tmp_path / "tourette_gallery.html",
        gallery_title="Tourette & Psychiatric CNS Discovery Gallery",
    )
    assert gallery_html.is_file()
    g_text = gallery_html.read_text(encoding="utf-8")
    assert len(g_text.encode("utf-8")) > 5_000
    g_match = re.search(
        r'<script id="geneuclidean-studio-data" type="application/json">(.*?)</script>',
        g_text,
        re.DOTALL,
    )
    assert g_match is not None
    g_payload = json.loads(g_match.group(1).replace("<\\/", "</"))
    assert len(g_payload["specs"]) == 2
    assert g_payload["specs"][1]["pdb_id"] == "6A93"
