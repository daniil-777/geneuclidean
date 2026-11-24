"""3D Conformer Generation, Native AutoDock Vina Docking & PoseBusters Physical Checks.

Provides an ARM64-native 3D evaluation pipeline with zero external MGLTools or
OpenBabel dependencies:
- RDKit ETKDGv3 3D conformer generation + MMFF94s (UFF fallback) energy minimization,
  internal torsional strain energy, and van der Waals molecular volume.
- Meeko-based ligand PDBQT preparation and pure-Python AutoDock4-typed receptor
  PDBQT preparation for cut protein binding pockets.
- Subprocess execution of local AutoDock Vina (`./bin/vina`) with pose parsing back
  into RDKit 3D molecules.
- PoseBusters-inspired physical plausibility audits (covalent bond lengths/angles,
  aromatic ring planarity, internal steric clashes, and protein-ligand clashes).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL"})

_AROMATIC_PROTEIN_ATOMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("PHE", "CG"),
        ("PHE", "CD1"),
        ("PHE", "CD2"),
        ("PHE", "CE1"),
        ("PHE", "CE2"),
        ("PHE", "CZ"),
        ("TYR", "CG"),
        ("TYR", "CD1"),
        ("TYR", "CD2"),
        ("TYR", "CE1"),
        ("TYR", "CE2"),
        ("TYR", "CZ"),
        ("TRP", "CG"),
        ("TRP", "CD1"),
        ("TRP", "CD2"),
        ("TRP", "CE2"),
        ("TRP", "CE3"),
        ("TRP", "CZ2"),
        ("TRP", "CZ3"),
        ("TRP", "CH2"),
        ("HIS", "CG"),
        ("HIS", "CD2"),
        ("HIS", "CE1"),
        ("HID", "CG"),
        ("HID", "CD2"),
        ("HID", "CE1"),
        ("HIE", "CG"),
        ("HIE", "CD2"),
        ("HIE", "CE1"),
        ("HIP", "CG"),
        ("HIP", "CD2"),
        ("HIP", "CE1"),
    }
)

_VALID_AD4_TYPES: frozenset[str] = frozenset(
    {
        "C",
        "A",
        "N",
        "NA",
        "NS",
        "OA",
        "OS",
        "SA",
        "S",
        "H",
        "HD",
        "HS",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
        "Mg",
        "Mn",
        "Zn",
        "Ca",
        "Fe",
        "Cu",
        "Co",
        "Ni",
        "Na",
        "K",
    }
)

_PERIODIC_TABLE = Chem.GetPeriodicTable()
_VINA_REMARK_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s+([-+]?\d+(?:\.\d+)?)")
_VINA_TABLE_RE = re.compile(r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?)\s+\d")


@dataclass
class ConformerResult:
    """Output of RDKit ETKDGv3 3D embedding and MMFF94s minimization."""

    smiles: str
    success: bool
    mol_3d: Chem.Mol | None = None
    energy_init: float = float("nan")
    energy_min: float = float("nan")
    strain_energy: float = float("nan")
    mol_volume: float = float("nan")
    force_field: str = "none"
    error: str | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize numeric fields without the RDKit Mol object."""
        return {
            "smiles": self.smiles,
            "success": self.success,
            "energy_init": self.energy_init,
            "energy_min": self.energy_min,
            "strain_energy": self.strain_energy,
            "mol_volume": self.mol_volume,
            "force_field": self.force_field,
            "error": self.error,
        }


@dataclass
class PoseBustersChecks:
    """PoseBusters-inspired intramolecular and intermolecular 3D geometry checks."""

    bond_lengths_ok: bool
    bond_angles_ok: bool
    aromatic_ring_planarity_ok: bool
    max_ring_planarity_dev: float
    internal_clashes_ok: bool
    internal_clash_count: int
    protein_clashes_ok: bool
    protein_clash_count: int
    min_protein_ligand_dist: float
    passed_all: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize checks to a plain dictionary."""
        return asdict(self)


@dataclass
class DockingResult:
    """Output of local AutoDock Vina docking and 3D physical pose verification."""

    smiles: str
    success: bool
    vina_affinity: float = float("nan")
    all_affinities: list[float] = field(default_factory=list)
    docked_mol: Chem.Mol | None = None
    posebusters: PoseBustersChecks | None = None
    error: str | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize docking summary without the RDKit Mol object."""
        return {
            "smiles": self.smiles,
            "success": self.success,
            "vina_affinity": self.vina_affinity,
            "all_affinities": list(self.all_affinities),
            "posebusters": self.posebusters.to_dict() if self.posebusters is not None else None,
            "error": self.error,
        }


def _get_vdw_radius(atomic_num: int) -> float:
    """Return van der Waals radius (Å) for an atomic number with safe fallback."""
    if atomic_num <= 0:
        return 1.70
    try:
        rad = float(_PERIODIC_TABLE.GetRvdw(atomic_num))
        return rad if rad > 0.0 else 1.70
    except Exception:
        return 1.70


def _get_covalent_radius(atomic_num: int) -> float:
    """Return covalent radius (Å) for an atomic number with safe fallback."""
    if atomic_num <= 0:
        return 0.77
    try:
        rad = float(_PERIODIC_TABLE.GetRcovalent(atomic_num))
        return rad if rad > 0.0 else 0.77
    except Exception:
        return 0.77


def _symbol_to_atomic_num(symbol: str) -> int:
    """Convert PDB element symbol to atomic number (fallback 6 for carbon)."""
    cleaned = symbol.strip()
    if not cleaned:
        return 6
    norm = cleaned[0].upper() + cleaned[1:].lower()
    try:
        return int(_PERIODIC_TABLE.GetAtomicNumber(norm))
    except Exception:
        return 6


def embed_and_minimize_3d(
    smiles: str,
    max_iters: int = 200,
    seed: int = 42,
) -> ConformerResult:
    """Generate a 3D conformer via RDKit ETKDGv3 and minimize with MMFF94s.

    Falls back to UFF if MMFF94s lacks force-field parameters for an exotic atom.
    Computes initial energy, minimized energy, internal torsional strain energy
    (``E_init - E_min``), and van der Waals molecular volume.
    """
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return ConformerResult(smiles=str(smiles), success=False, error="Invalid SMILES")

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = int(seed)
    params.useRandomCoords = False

    embed_status = AllChem.EmbedMolecule(mol_h, params)  # type: ignore[attr-defined]
    if embed_status != 0:
        params.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(mol_h, params)  # type: ignore[attr-defined]
    if embed_status != 0 or mol_h.GetNumConformers() == 0:
        return ConformerResult(
            smiles=smiles,
            success=False,
            error="ETKDGv3 3D conformer embedding failed",
        )

    force_field_name = "none"
    e_init = float("nan")
    e_min = float("nan")
    strain = float("nan")

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(  # type: ignore[attr-defined]
            mol_h, mmffVariant="MMFF94s"
        )
        if mmff_props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props)  # type: ignore[attr-defined]
            if ff is not None:
                force_field_name = "MMFF94s"
                e_init = float(ff.CalcEnergy())
                ff.Minimize(maxIts=max_iters)
                e_min = float(ff.CalcEnergy())
                strain = float(e_init - e_min)

        if force_field_name == "none" and AllChem.UFFHasAllMoleculeParams(mol_h):  # type: ignore[attr-defined]
            uff = AllChem.UFFGetMoleculeForceField(mol_h)  # type: ignore[attr-defined]
            if uff is not None:
                force_field_name = "UFF"
                e_init = float(uff.CalcEnergy())
                uff.Minimize(maxIts=max_iters)
                e_min = float(uff.CalcEnergy())
                strain = float(e_init - e_min)
    except Exception as exc:
        return ConformerResult(
            smiles=smiles,
            success=False,
            error=f"Force-field minimization error: {exc}",
        )

    try:
        mol_vol = float(AllChem.ComputeMolVolume(mol_h))
    except Exception:
        mol_vol = float("nan")

    return ConformerResult(
        smiles=smiles,
        success=True,
        mol_3d=mol_h,
        energy_init=e_init,
        energy_min=e_min,
        strain_energy=strain,
        mol_volume=mol_vol,
        force_field=force_field_name,
    )


def prepare_ligand_pdbqt(mol_3d: Chem.Mol) -> str | None:
    """Prepare a 3D RDKit molecule into an AutoDock Vina ligand PDBQT string via Meeko."""
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return None
    try:
        import meeko
    except ImportError:
        return None

    try:
        has_hs = any(atom.GetAtomicNum() == 1 for atom in mol_3d.GetAtoms())
        mol_prep = mol_3d if has_hs else Chem.AddHs(mol_3d, addCoords=True)
        preparator = meeko.MoleculePreparation()
        setups = preparator.prepare(mol_prep)
        if not setups:
            return None
        pdbqt_str, is_ok, _err = meeko.PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok or not pdbqt_str:
            return None
        return str(pdbqt_str)
    except Exception:
        return None


def prepare_receptor_pdbqt(
    pocket_pdb_path: str | Path,
    out_pdbqt_path: str | Path | None = None,
    *,
    keep_waters: bool = False,
) -> Path | None:
    """Convert a protein pocket PDB into an AutoDock Vina-compatible receptor PDBQT file.

    Lightweight pure-Python AutoDock4 receptor typer designed to handle cut protein
    binding pockets cleanly on macOS ARM64 with zero MGLTools/OpenBabel dependencies:
    - Filters water molecules (unless ``keep_waters=True``) and alternate conformations.
    - Retains polar hydrogens (within 1.35 Å of N/O/S) as ``HD`` and omits nonpolar H.
    - Assigns AutoDock4 atom types (``A`` aromatic C, ``C`` aliphatic C, ``NA``/``N``,
      ``OA``, ``SA``, halogens, catalytic metals) in exact PDBQT columns 78-79.
    """
    in_path = Path(pocket_pdb_path)
    if not in_path.exists():
        return None

    out_path = Path(out_pdbqt_path) if out_pdbqt_path is not None else in_path.with_suffix(".pdbqt")

    parsed: list[tuple[str, str, str, str, NDArray[np.float64]]] = []
    with in_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "A", "1"):
                continue
            resname = line[17:20].strip().upper()
            if not keep_waters and resname in _WATER_RESNAMES:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            elem = line[76:78].strip() if len(line) >= 78 else ""
            if not elem:
                letters = "".join(ch for ch in atom_name if ch.isalpha())
                elem = letters[:1] if letters else "C"
            parsed.append((line, resname, atom_name, elem, np.array([x, y, z], dtype=np.float64)))

    if not parsed:
        return None

    polar_heavy_coords = np.asarray(
        [rec[4] for rec in parsed if rec[3].upper() in {"N", "O", "S"}],
        dtype=np.float64,
    )
    h_coords = np.asarray(
        [rec[4] for rec in parsed if rec[3].upper() == "H"],
        dtype=np.float64,
    )

    lines_out: list[str] = []
    for line, resname, atom_name, elem, coord in parsed:
        elem_u = elem.upper()
        if elem_u == "H":
            if (
                len(polar_heavy_coords) > 0
                and float(np.min(np.linalg.norm(polar_heavy_coords - coord, axis=1))) <= 1.35
            ):
                ad_type = "HD"
            else:
                # Skip nonpolar hydrogens (united-atom AutoDock/Vina representation)
                continue
        elif elem_u == "C":
            ad_type = "A" if (resname, atom_name) in _AROMATIC_PROTEIN_ATOMS else "C"
        elif elem_u == "N":
            if resname in {"HIS", "HID", "HIE"} and atom_name in {"ND1", "NE2"}:
                has_h = (
                    len(h_coords) > 0
                    and float(np.min(np.linalg.norm(h_coords - coord, axis=1))) <= 1.35
                )
                ad_type = "N" if has_h else "NA"
            else:
                ad_type = "N"
        elif elem_u == "O":
            ad_type = "OA"
        elif elem_u == "S":
            ad_type = "SA"
        else:
            cand = elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem_u
            ad_type = cand if cand in _VALID_AD4_TYPES else "C"

        # Exact PDBQT column layout: cols 1-66 PDB prefix, 67-70 spaces, 71-76 charge, 78-79 type
        base = line[:66].rstrip("\r\n").ljust(66)
        charge = 0.0
        lines_out.append(f"{base}    {charge:6.3f} {ad_type:<2s}\n")

    if not lines_out:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines_out), encoding="utf-8")
    return out_path


def extract_pocket_heavy_atoms(
    pocket_pdb_path: str | Path,
    *,
    keep_waters: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Parse heavy-atom coordinates ``(N, 3)`` and atomic numbers from a pocket PDB."""
    coords: list[list[float]] = []
    atomic_nums: list[int] = []
    with Path(pocket_pdb_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "A", "1"):
                continue
            resname = line[17:20].strip().upper()
            if not keep_waters and resname in _WATER_RESNAMES:
                continue
            elem = line[76:78].strip() if len(line) >= 78 else ""
            if not elem:
                atom_name = line[12:16].strip()
                letters = "".join(ch for ch in atom_name if ch.isalpha())
                elem = letters[:1] if letters else "C"
            if elem.upper() == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
            atomic_nums.append(_symbol_to_atomic_num(elem))
    if not coords:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int32)
    return np.asarray(coords, dtype=np.float64), np.asarray(atomic_nums, dtype=np.int32)


def compute_ligand_box_center(ligand_path: str | Path) -> tuple[float, float, float]:
    """Compute the 3D heavy-atom geometric centroid of a crystal ligand SDF/MOL2/PDB."""
    path = Path(ligand_path)
    mol: Chem.Mol | None = None
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)

    if mol is None or mol.GetNumConformers() == 0:
        msg = f"Failed to read 3D ligand coordinates from {path}"
        raise ValueError(msg)

    conf = mol.GetConformer(0)
    coords = np.asarray(
        [
            list(conf.GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ],
        dtype=np.float64,
    )
    if len(coords) == 0:
        coords = np.asarray(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float64,
        )
    center = coords.mean(axis=0)
    return (float(center[0]), float(center[1]), float(center[2]))


def check_pose_geometry(
    mol_3d: Chem.Mol,
    pocket_pdb_path: str | Path | None = None,
    *,
    conf_id: int = 0,
    protein_clash_ratio: float = 0.75,
    internal_clash_ratio: float = 0.70,
    planarity_tol: float = 0.25,
) -> PoseBustersChecks:
    """Run PoseBusters-inspired 3D physical plausibility checks on a conformer/pose.

    Checks:
    1. Covalent bond length sanity against covalent radii sum (``[0.70, 1.30]``).
    2. Bond angle sanity for all heavy-atom triplets (``>= 45°``).
    3. Aromatic ring planarity (max out-of-plane deviation ``<= 0.25 Å`` via SVD).
    4. Internal steric clashes for heavy atoms separated by ``>= 3`` bonds.
    5. Protein-ligand steric clashes where ``d_ij < 0.75 * (r_vdw_i + r_vdw_j)``.
    """
    mol_heavy = Chem.RemoveHs(mol_3d)
    if mol_heavy.GetNumConformers() == 0:
        return PoseBustersChecks(
            bond_lengths_ok=False,
            bond_angles_ok=False,
            aromatic_ring_planarity_ok=False,
            max_ring_planarity_dev=float("nan"),
            internal_clashes_ok=False,
            internal_clash_count=0,
            protein_clashes_ok=False,
            protein_clash_count=0,
            min_protein_ligand_dist=float("nan"),
            passed_all=False,
        )

    conf = mol_heavy.GetConformer(conf_id)
    n_atoms = mol_heavy.GetNumAtoms()
    coords = np.asarray([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=np.float64)
    atomic_nums = np.asarray(
        [mol_heavy.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)], dtype=np.int32
    )
    vdw_radii = np.asarray([_get_vdw_radius(int(z)) for z in atomic_nums], dtype=np.float64)

    # 1. Covalent bond length sanity
    bond_lengths_ok = True
    for bond in mol_heavy.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        expected = _get_covalent_radius(int(atomic_nums[i])) + _get_covalent_radius(
            int(atomic_nums[j])
        )
        ratio = dist / max(expected, 1e-6)
        if ratio < 0.70 or ratio > 1.30:
            bond_lengths_ok = False
            break

    # 2. Heavy-atom bond angle sanity
    bond_angles_ok = True
    for center_atom in mol_heavy.GetAtoms():
        j = center_atom.GetIdx()
        nbrs = [n.GetIdx() for n in center_atom.GetNeighbors()]
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                angle_deg = float(rdMolTransforms.GetAngleDeg(conf, nbrs[a_idx], j, nbrs[b_idx]))
                if not np.isfinite(angle_deg) or angle_deg < 45.0:
                    bond_angles_ok = False
                    break
            if not bond_angles_ok:
                break
        if not bond_angles_ok:
            break

    # 3. Aromatic ring planarity via best-fit SVD plane
    max_ring_dev = 0.0
    ring_info = mol_heavy.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) < 4:
            continue
        if all(mol_heavy.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            pts = coords[list(ring)]
            centered = pts - pts.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            normal = vt[-1]
            devs = np.abs(centered @ normal)
            max_ring_dev = max(max_ring_dev, float(np.max(devs)))
    aromatic_planarity_ok = max_ring_dev <= planarity_tol

    # 4. Internal steric clashes (heavy atoms >= 3 topological bonds apart)
    internal_clash_count = 0
    if n_atoms >= 4:
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        topo_dist = Chem.GetDistanceMatrix(mol_heavy)
        vdw_sum = vdw_radii[:, None] + vdw_radii[None, :]
        tri_i, tri_j = np.triu_indices(n_atoms, k=1)
        nonbonded = topo_dist[tri_i, tri_j] >= 3.0
        too_close = dist_mat[tri_i, tri_j] < internal_clash_ratio * vdw_sum[tri_i, tri_j]
        internal_clash_count = int(np.sum(nonbonded & too_close))
    internal_clashes_ok = internal_clash_count == 0

    # 5. Protein-ligand steric clashes
    protein_clash_count = 0
    min_pl_dist = float("nan")
    if pocket_pdb_path is not None and Path(pocket_pdb_path).exists() and n_atoms > 0:
        p_coords, p_nums = extract_pocket_heavy_atoms(pocket_pdb_path)
        if len(p_coords) > 0:
            p_vdw = np.asarray([_get_vdw_radius(int(z)) for z in p_nums], dtype=np.float64)
            pl_dists = np.linalg.norm(coords[:, None, :] - p_coords[None, :, :], axis=-1)
            min_pl_dist = float(np.min(pl_dists))
            pl_thresh = protein_clash_ratio * (vdw_radii[:, None] + p_vdw[None, :])
            protein_clash_count = int(np.sum(pl_dists < pl_thresh))
    protein_clashes_ok = protein_clash_count == 0

    passed_all = (
        bond_lengths_ok
        and bond_angles_ok
        and aromatic_planarity_ok
        and internal_clashes_ok
        and protein_clashes_ok
    )
    return PoseBustersChecks(
        bond_lengths_ok=bond_lengths_ok,
        bond_angles_ok=bond_angles_ok,
        aromatic_ring_planarity_ok=aromatic_planarity_ok,
        max_ring_planarity_dev=max_ring_dev,
        internal_clashes_ok=internal_clashes_ok,
        internal_clash_count=internal_clash_count,
        protein_clashes_ok=protein_clashes_ok,
        protein_clash_count=protein_clash_count,
        min_protein_ligand_dist=min_pl_dist,
        passed_all=passed_all,
    )


def _resolve_vina_binary(vina_bin: str | Path) -> Path | None:
    """Resolve executable AutoDock Vina binary path or return ``None``."""
    candidate = Path(vina_bin)
    if candidate.is_file() and candidate.stat().st_mode & 0o111:
        return candidate
    found = shutil.which(str(vina_bin))
    return Path(found) if found is not None else None


def _parse_vina_affinities(out_pdbqt_text: str, stdout_text: str) -> list[float]:
    """Parse docked affinities (kcal/mol) from Vina output PDBQT or stdout table."""
    affinities = [float(m.group(1)) for m in _VINA_REMARK_RE.finditer(out_pdbqt_text)]
    if affinities:
        return affinities
    for line in stdout_text.splitlines():
        match = _VINA_TABLE_RE.match(line)
        if match:
            affinities.append(float(match.group(2)))
    return affinities


def _reconstruct_docked_mol(out_pdbqt_text: str) -> Chem.Mol | None:
    """Convert Vina output PDBQT text back to an RDKit 3D Mol via Meeko."""
    try:
        import meeko

        pmol = meeko.PDBQTMolecule(out_pdbqt_text, is_dlg=False, skip_typing=True)
        rdmols = meeko.RDKitMolCreate.from_pdbqt_mol(pmol)
        if rdmols and rdmols[0] is not None and rdmols[0].GetNumConformers() > 0:
            return cast(Chem.Mol, rdmols[0])
    except Exception:
        return None
    return None


def dock_ligand_vina(
    smiles: str,
    pocket_pdb_path: str | Path,
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 4,
    *,
    num_modes: int = 9,
    seed: int = 42,
    timeout: int = 120,
    receptor_pdbqt_path: str | Path | None = None,
) -> DockingResult:
    """Dock a SMILES ligand into a protein pocket via native AutoDock Vina.

    Workflow:
    1. Generate 3D conformer via ETKDGv3 + MMFF94s.
    2. Prepare ligand PDBQT via Meeko and receptor PDBQT (or reuse cached path).
    3. Run ``vina_bin`` in an isolated temporary directory and parse binding affinities.
    4. Reconstruct the best docked 3D pose and audit it with PoseBusters checks.
    """
    vina_exe = _resolve_vina_binary(vina_bin)
    if vina_exe is None:
        return DockingResult(
            smiles=smiles,
            success=False,
            error=f"AutoDock Vina binary not found or not executable: {vina_bin}",
        )

    pocket_path = Path(pocket_pdb_path)
    if not pocket_path.exists():
        return DockingResult(
            smiles=smiles,
            success=False,
            error=f"Pocket PDB file not found: {pocket_path}",
        )

    conf_res = embed_and_minimize_3d(smiles, seed=seed)
    if not conf_res.success or conf_res.mol_3d is None:
        return DockingResult(
            smiles=smiles,
            success=False,
            error=conf_res.error or "3D conformer generation failed",
        )

    lig_pdbqt = prepare_ligand_pdbqt(conf_res.mol_3d)
    if lig_pdbqt is None:
        return DockingResult(
            smiles=smiles,
            success=False,
            error="Meeko ligand PDBQT preparation failed",
        )

    with tempfile.TemporaryDirectory(prefix="geneuc_vina_") as tmp_dir:
        tmp = Path(tmp_dir)
        lig_path = tmp / "ligand.pdbqt"
        out_path = tmp / "docked.pdbqt"
        lig_path.write_text(lig_pdbqt, encoding="utf-8")

        if receptor_pdbqt_path is not None and Path(receptor_pdbqt_path).exists():
            rec_path = Path(receptor_pdbqt_path)
        else:
            rec_prepared = prepare_receptor_pdbqt(pocket_path, tmp / "receptor.pdbqt")
            if rec_prepared is None:
                return DockingResult(
                    smiles=smiles,
                    success=False,
                    error="Receptor PDBQT preparation failed",
                )
            rec_path = rec_prepared

        cmd = [
            str(vina_exe),
            "--receptor",
            str(rec_path),
            "--ligand",
            str(lig_path),
            "--center_x",
            f"{box_center[0]:.3f}",
            "--center_y",
            f"{box_center[1]:.3f}",
            "--center_z",
            f"{box_center[2]:.3f}",
            "--size_x",
            f"{box_size[0]:.3f}",
            "--size_y",
            f"{box_size[1]:.3f}",
            "--size_z",
            f"{box_size[2]:.3f}",
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--seed",
            str(seed),
            "--out",
            str(out_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return DockingResult(
                smiles=smiles,
                success=False,
                error=f"AutoDock Vina timed out after {timeout}s",
            )
        except OSError as exc:
            return DockingResult(
                smiles=smiles,
                success=False,
                error=f"Failed to launch AutoDock Vina: {exc}",
            )

        if proc.returncode != 0 or not out_path.exists():
            err_msg = (proc.stderr or proc.stdout).strip()[:400]
            return DockingResult(
                smiles=smiles,
                success=False,
                error=f"Vina exited with code {proc.returncode}: {err_msg}",
            )

        out_text = out_path.read_text(encoding="utf-8", errors="ignore")
        affinities = _parse_vina_affinities(out_text, proc.stdout)
        if not affinities:
            return DockingResult(
                smiles=smiles,
                success=False,
                error="Failed to parse Vina affinity scores from output",
            )

        docked_mol = _reconstruct_docked_mol(out_text)
        pb_checks = (
            check_pose_geometry(docked_mol, pocket_pdb_path=pocket_path, conf_id=0)
            if docked_mol is not None
            else None
        )

        return DockingResult(
            smiles=smiles,
            success=True,
            vina_affinity=float(affinities[0]),
            all_affinities=affinities,
            docked_mol=docked_mol,
            posebusters=pb_checks,
        )


def evaluate_3d_plausibility(
    smiles_list: Sequence[str],
    pocket_pdb_path: str | Path | None = None,
    native_ligand_sdf: str | Path | None = None,
    *,
    run_vina: bool = False,
    vina_bin: str | Path = "bin/vina",
    max_dock: int = 10,
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Batch 3D conformer minimization, optional AutoDock Vina docking, and PoseBusters audit.

    Computes ETKDGv3 conformer success rate, MMFF94s strain energy, molecular volume,
    intramolecular PoseBusters pass rate, and (when ``run_vina=True``) Vina docking
    scores, High-Affinity % (fraction outperforming redocked native ligand), and
    bound-pose PoseBusters pass rate.
    """
    n_input = len(smiles_list)
    conformer_results: list[ConformerResult] = []
    strain_energies: list[float] = []
    mol_volumes: list[float] = []
    intra_pb_passes: list[bool] = []
    valid_embedded_smiles: list[str] = []

    for idx, smi in enumerate(smiles_list):
        res = embed_and_minimize_3d(smi, seed=seed + idx)
        conformer_results.append(res)
        if res.success and res.mol_3d is not None:
            valid_embedded_smiles.append(smi)
            if np.isfinite(res.strain_energy):
                strain_energies.append(res.strain_energy)
            if np.isfinite(res.mol_volume):
                mol_volumes.append(res.mol_volume)
            pb_intra = check_pose_geometry(res.mol_3d, pocket_pdb_path=None)
            intra_pb_passes.append(pb_intra.passed_all)

    n_embedded = len(valid_embedded_smiles)
    summary: dict[str, Any] = {
        "n_input": n_input,
        "n_embedded": n_embedded,
        "conformer_success_rate": (n_embedded / n_input) if n_input > 0 else 0.0,
        "strain_energy_mean": float(np.mean(strain_energies)) if strain_energies else float("nan"),
        "strain_energy_median": (
            float(np.median(strain_energies)) if strain_energies else float("nan")
        ),
        "mol_volume_mean": float(np.mean(mol_volumes)) if mol_volumes else float("nan"),
        "mol_volume_std": float(np.std(mol_volumes)) if mol_volumes else float("nan"),
        "intra_posebusters_pass_rate": (
            float(np.mean(intra_pb_passes)) if intra_pb_passes else 0.0
        ),
        "vina_ran": False,
    }

    if not run_vina or pocket_pdb_path is None:
        return summary

    pocket_path = Path(pocket_pdb_path)
    if not pocket_path.exists() or _resolve_vina_binary(vina_bin) is None:
        return summary

    # Determine docking box center from native crystal ligand or pocket heavy centroid
    if native_ligand_sdf is not None and Path(native_ligand_sdf).exists():
        box_center = compute_ligand_box_center(native_ligand_sdf)
    else:
        p_coords, _ = extract_pocket_heavy_atoms(pocket_path)
        if len(p_coords) == 0:
            return summary
        c = p_coords.mean(axis=0)
        box_center = (float(c[0]), float(c[1]), float(c[2]))

    with tempfile.TemporaryDirectory(prefix="geneuc_rec_") as rec_tmp:
        rec_pdbqt = prepare_receptor_pdbqt(pocket_path, Path(rec_tmp) / "receptor.pdbqt")
        if rec_pdbqt is None:
            return summary

        native_vina_affinity = float("nan")
        if native_ligand_sdf is not None and Path(native_ligand_sdf).exists():
            suppl = Chem.SDMolSupplier(str(native_ligand_sdf), removeHs=False)
            nat_mol = suppl[0] if len(suppl) > 0 else None
            if nat_mol is not None:
                nat_smi = Chem.MolToSmiles(Chem.RemoveHs(nat_mol), canonical=True)
                nat_dock = dock_ligand_vina(
                    nat_smi,
                    pocket_path,
                    box_center=box_center,
                    box_size=box_size,
                    vina_bin=vina_bin,
                    exhaustiveness=exhaustiveness,
                    seed=seed,
                    receptor_pdbqt_path=rec_pdbqt,
                )
                if nat_dock.success:
                    native_vina_affinity = nat_dock.vina_affinity

        to_dock = valid_embedded_smiles[:max_dock]
        dock_results: list[DockingResult] = []
        for idx, smi in enumerate(to_dock):
            d_res = dock_ligand_vina(
                smi,
                pocket_path,
                box_center=box_center,
                box_size=box_size,
                vina_bin=vina_bin,
                exhaustiveness=exhaustiveness,
                seed=seed + idx,
                receptor_pdbqt_path=rec_pdbqt,
            )
            dock_results.append(d_res)

    affinities = [
        d.vina_affinity for d in dock_results if d.success and np.isfinite(d.vina_affinity)
    ]
    pb_list = [d.posebusters for d in dock_results if d.success and d.posebusters is not None]

    high_affinity_rate = float("nan")
    if affinities and np.isfinite(native_vina_affinity):
        high_affinity_rate = float(np.mean([a <= native_vina_affinity for a in affinities]))

    summary.update(
        {
            "vina_ran": True,
            "n_dock_attempted": len(to_dock),
            "n_dock_succeeded": len(affinities),
            "vina_success_rate": (len(affinities) / len(to_dock)) if to_dock else 0.0,
            "native_vina_affinity": native_vina_affinity,
            "vina_affinity_mean": float(np.mean(affinities)) if affinities else float("nan"),
            "vina_affinity_std": float(np.std(affinities)) if affinities else float("nan"),
            "vina_affinity_best": float(np.min(affinities)) if affinities else float("nan"),
            "vina_affinities": affinities,
            "high_affinity_rate": high_affinity_rate,
            "posebusters_pass_rate": (
                float(np.mean([pb.passed_all for pb in pb_list])) if pb_list else float("nan")
            ),
            "protein_clash_free_rate": (
                float(np.mean([pb.protein_clashes_ok for pb in pb_list]))
                if pb_list
                else float("nan")
            ),
            "mean_protein_clashes": (
                float(np.mean([pb.protein_clash_count for pb in pb_list]))
                if pb_list
                else float("nan")
            ),
            "docking_results": [d.to_summary_dict() for d in dock_results],
        }
    )
    return summary


__all__ = [
    "ConformerResult",
    "DockingResult",
    "PoseBustersChecks",
    "check_pose_geometry",
    "compute_ligand_box_center",
    "dock_ligand_vina",
    "embed_and_minimize_3d",
    "evaluate_3d_plausibility",
    "extract_pocket_heavy_atoms",
    "prepare_ligand_pdbqt",
    "prepare_receptor_pdbqt",
]
