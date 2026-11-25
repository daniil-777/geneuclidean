"""E(3)-invariant 52-dim 3D protein pocket featurization and ligand descriptor extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from geneuclidean.data.tokenizer import canonicalize_smiles

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

POCKET_NODE_DIM: int = 52
LIGAND_FP_DIM: int = 1024
LIGAND_PROP_NAMES: tuple[str, ...] = ("MW", "LogP", "TPSA", "QED", "HBD", "HBA")

# 10 element classes: C, N, O, S, P, F, Cl, Br, Metal, Other
_ELEMENT_TO_IDX: dict[str, int] = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "P": 4,
    "F": 5,
    "CL": 6,
    "BR": 7,
}
_METAL_ELEMENTS: frozenset[str] = frozenset(
    {"ZN", "MG", "CA", "FE", "MN", "CU", "NA", "K", "NI", "CO"}
)

# 20 standard amino acids + OTHER (index 20)
_STANDARD_RESIDUES: tuple[str, ...] = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
_RESIDUE_TO_IDX: dict[str, int] = {res: idx for idx, res in enumerate(_STANDARD_RESIDUES)}
_RESIDUE_ALIASES: dict[str, str] = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "MSE": "MET",
    "LYN": "LYS",
}

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "SOL"})
_MODIFIED_POLYMER_RESNAMES: frozenset[str] = frozenset({"MSE"})
_BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})
_BACKBONE_ROLE_TO_IDX: dict[str, int] = {"N": 0, "CA": 1, "C": 2, "O": 3, "OXT": 3}

# Physics-grounded atomic masses (Da) and Bondi van der Waals radii (Angstroms)
_ATOMIC_MASS: dict[str, float] = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "F": 18.998,
    "CL": 35.45,
    "BR": 79.904,
    "I": 126.904,
    "SE": 78.971,
    "ZN": 65.38,
    "MG": 24.305,
    "CA": 40.078,
    "FE": 55.845,
    "MN": 54.938,
    "CU": 63.546,
    "NA": 22.990,
    "K": 39.098,
    "NI": 58.693,
    "CO": 58.933,
}
_VDW_RADIUS: dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 2.31,
    "FE": 1.56,
    "MN": 1.61,
    "CU": 1.40,
    "NA": 2.27,
    "K": 2.75,
    "NI": 1.63,
    "CO": 1.52,
}

# Residue-specific pharmacophore lookup sets (standard PDB atom names)
_AROMATIC_ATOMS: dict[str, frozenset[str]] = {
    "PHE": frozenset({"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "TYR": frozenset({"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "TRP": frozenset({"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}),
    "HIS": frozenset({"CG", "ND1", "CD2", "CE1", "NE2"}),
}
_HBA_SIDECHAIN: dict[str, frozenset[str]] = {
    "ASP": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
    "ASN": frozenset({"OD1"}),
    "GLN": frozenset({"OE1"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TYR": frozenset({"OH"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "CYS": frozenset({"SG"}),
    "MET": frozenset({"SD"}),
}
_HBD_SIDECHAIN: dict[str, frozenset[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "LYS": frozenset({"NZ"}),
    "ASN": frozenset({"ND2"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "TRP": frozenset({"NE1"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TYR": frozenset({"OH"}),
    "CYS": frozenset({"SG"}),
}
_POS_IONIZABLE: dict[str, frozenset[str]] = {
    "ARG": frozenset({"NE", "CZ", "NH1", "NH2"}),
    "LYS": frozenset({"NZ"}),
    "HIS": frozenset({"ND1", "CE1", "NE2"}),
}
_NEG_IONIZABLE: dict[str, frozenset[str]] = {
    "ASP": frozenset({"CG", "OD1", "OD2"}),
    "GLU": frozenset({"CD", "OE1", "OE2"}),
}

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=LIGAND_FP_DIM)


@dataclass(frozen=True)
class PocketFeatures:
    """Centered 3D protein pocket atom graph and associated ligand descriptors."""

    pdb_id: str
    coords: np.ndarray  # [N, 3] float32, centered at ligand/pocket centroid
    node_feats: np.ndarray  # [N, 52] float32, strictly E(3)-invariant
    contact_mask: np.ndarray  # [N] bool, True within 4.5 A of any ligand heavy atom
    center: np.ndarray  # [3] float32, original world-space center used for centering
    canonical_smiles: str | None = None
    ligand_fp: np.ndarray | None = None  # [1024] float32
    ligand_props: np.ndarray | None = None  # [6] float32: [MW, LogP, TPSA, QED, HBD, HBA]
    pkd: float = float("nan")


def _infer_element(atom: Any) -> str:
    """Resolve uppercase chemical element symbol from a Biopython Atom."""
    elem = str(getattr(atom, "element", "") or "").strip().upper()
    if elem and elem != "X":
        return elem
    name = str(atom.get_name()).strip().upper()
    if not name:
        return "C"
    if len(name) >= 2 and name[:2] in ("CL", "BR", "ZN", "MG", "CA", "FE", "MN", "CU", "NA", "SE"):
        return name[:2]
    for ch in name:
        if ch.isalpha():
            return ch
    return "C"


def _partial_charge_proxy(resname: str, atom_name: str, element: str) -> float:
    """Return a physics-grounded Amber/Gasteiger partial charge proxy in [-1.0, +1.0]."""
    if element in _METAL_ELEMENTS:
        return 1.0
    if atom_name == "N":
        return -0.42
    if atom_name == "CA":
        return 0.05
    if atom_name == "C":
        return 0.60
    if atom_name == "O":
        return -0.57
    if atom_name == "OXT":
        return -0.80
    if atom_name in _NEG_IONIZABLE.get(resname, frozenset()):
        return -0.80 if element == "O" else 0.70
    if atom_name in _POS_IONIZABLE.get(resname, frozenset()):
        return 0.50 if element == "N" else 0.60
    if element == "O":
        return -0.50
    if element == "N":
        return -0.35
    if element == "S":
        return -0.20
    if element == "P":
        return 0.80
    if element in ("F", "CL", "BR"):
        return -0.20
    return 0.0


def _compute_52d_node_features(
    atoms_meta: list[tuple[str, str, str]],
    centered_coords: np.ndarray,
    pocket_cutoff: float,
) -> np.ndarray:
    """Compute the canonical strictly E(3)-invariant 52-dim per-atom descriptor matrix.

    Layout (total = 52 floats, ``float32``):
      - ``[0:10]``: Element one-hot (C, N, O, S, P, F, Cl, Br, Metal, Other)
      - ``[10:31]``: Residue one-hot (20 standard amino acids + OTHER)
      - ``[31:36]``: Backbone/sidechain role one-hot (N, CA, C, O, sidechain)
      - ``[36:52]``: 16 pharmacophore & E(3)-invariant geometric scalars:
        ``[is_hydrophobic, is_aromatic, is_hba, is_hbd, is_pos_ionizable, is_neg_ionizable,
        is_backbone, is_polar, normalized_atomic_mass, vdw_radius_norm, partial_charge_proxy,
        dist_to_centroid_norm, dist_min_to_centroid_shell, radial_rbf_0, radial_rbf_1,
        radial_rbf_2]``.
    """
    n_atoms = len(atoms_meta)
    feats = np.zeros((n_atoms, POCKET_NODE_DIM), dtype=np.float32)
    if n_atoms == 0:
        return feats

    radii = np.linalg.norm(centered_coords, axis=-1).astype(np.float32)
    r_min = float(radii.min())
    r_max = float(radii.max())
    r_span = max(r_max - r_min, 1e-6)
    inv_cutoff = 1.0 / max(float(pocket_cutoff), 1e-6)

    for idx, (element, raw_resname, atom_name) in enumerate(atoms_meta):
        # 1. Element one-hot (10)
        if element in _ELEMENT_TO_IDX:
            feats[idx, _ELEMENT_TO_IDX[element]] = 1.0
        elif element in _METAL_ELEMENTS:
            feats[idx, 8] = 1.0
        else:
            feats[idx, 9] = 1.0

        # 2. Residue one-hot (21)
        resname = _RESIDUE_ALIASES.get(raw_resname, raw_resname)
        res_idx = _RESIDUE_TO_IDX.get(resname, 20)
        feats[idx, 10 + res_idx] = 1.0

        # 3. Backbone / sidechain role (5)
        role_idx = _BACKBONE_ROLE_TO_IDX.get(atom_name, 4)
        feats[idx, 31 + role_idx] = 1.0

        # 4. Pharmacophore & E(3)-invariant scalars (16)
        is_backbone = 1.0 if atom_name in _BACKBONE_ATOMS else 0.0
        is_aromatic = 1.0 if atom_name in _AROMATIC_ATOMS.get(resname, frozenset()) else 0.0
        is_hba = (
            1.0
            if (
                atom_name in ("O", "OXT")
                or atom_name in _HBA_SIDECHAIN.get(resname, frozenset())
                or (res_idx == 20 and element in ("O", "N"))
            )
            else 0.0
        )
        is_hbd = (
            1.0
            if (
                (atom_name == "N" and resname != "PRO")
                or atom_name in _HBD_SIDECHAIN.get(resname, frozenset())
            )
            else 0.0
        )
        is_pos = (
            1.0
            if (atom_name in _POS_IONIZABLE.get(resname, frozenset()) or element in _METAL_ELEMENTS)
            else 0.0
        )
        is_neg = (
            1.0
            if (atom_name == "OXT" or atom_name in _NEG_IONIZABLE.get(resname, frozenset()))
            else 0.0
        )
        is_polar = (
            1.0
            if (
                element in ("N", "O", "S", "P")
                or is_hba > 0.0
                or is_hbd > 0.0
                or is_pos > 0.0
                or is_neg > 0.0
            )
            else 0.0
        )
        is_hydrophobic = (
            1.0
            if (
                (element == "C" and atom_name != "C" and is_pos == 0.0 and is_neg == 0.0)
                or (element == "S" and resname in ("MET", "CYS"))
                or element in ("F", "CL", "BR")
            )
            else 0.0
        )

        mass_norm = min(_ATOMIC_MASS.get(element, 12.011) / 32.0, 4.0)
        vdw_norm = _VDW_RADIUS.get(element, 1.70) / 2.0
        charge_proxy = _partial_charge_proxy(resname, atom_name, element)

        r_val = float(radii[idx])
        dist_norm = r_val * inv_cutoff
        shell_pos = (r_val - r_min) / r_span
        rbf0 = float(np.exp(-(((r_val - 3.0) / 2.0) ** 2)))
        rbf1 = float(np.exp(-(((r_val - 6.0) / 2.0) ** 2)))
        rbf2 = float(np.exp(-(((r_val - 9.0) / 2.0) ** 2)))

        feats[idx, 36] = is_hydrophobic
        feats[idx, 37] = is_aromatic
        feats[idx, 38] = is_hba
        feats[idx, 39] = is_hbd
        feats[idx, 40] = is_pos
        feats[idx, 41] = is_neg
        feats[idx, 42] = is_backbone
        feats[idx, 43] = is_polar
        feats[idx, 44] = mass_norm
        feats[idx, 45] = vdw_norm
        feats[idx, 46] = charge_proxy
        feats[idx, 47] = dist_norm
        feats[idx, 48] = shell_pos
        feats[idx, 49] = rbf0
        feats[idx, 50] = rbf1
        feats[idx, 51] = rbf2

    return feats


def _load_protein_heavy_atoms(
    structure_path: Path,
    *,
    include_metals: bool = True,
) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Load heavy (non-H, non-water, non-ligand) protein atoms from a PDB or mmCIF file.

    Filters out non-protein HETATM records (co-crystal organic ligands, membrane lipids,
    buffers, cryoprotectants, and solvents) to prevent ligand-atom leakage when parsing raw
    holo PDB/mmCIF structures. Retains standard polymer amino-acid residues
    (``residue.id[0] == " "``), selenomethionine (``MSE``), and optionally catalytic metal
    ions (``ZN, MG, CA, FE, MN, CU, CO, NI, K, NA``).
    """
    suffix = structure_path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser: Any = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(structure_path.stem, str(structure_path))
    model = structure[0]

    coords_list: list[np.ndarray] = []
    meta_list: list[tuple[str, str, str]] = []
    for chain in model:
        for residue in chain:
            resname = str(residue.get_resname()).strip().upper()
            if resname in _WATER_RESNAMES:
                continue
            hetflag = str(residue.id[0])
            is_polymer = hetflag == " " or resname in _MODIFIED_POLYMER_RESNAMES
            is_metal_ion = include_metals and resname in _METAL_ELEMENTS
            if not (is_polymer or is_metal_ion):
                continue
            for atom in residue.get_atoms():
                element = _infer_element(atom)
                if element == "H" or element == "D":
                    continue
                if is_metal_ion and not is_polymer and element not in _METAL_ELEMENTS:
                    continue
                coord = np.asarray(atom.get_coord(), dtype=np.float32)
                if not np.all(np.isfinite(coord)):
                    continue
                atom_name = str(atom.get_name()).strip().upper()
                coords_list.append(coord)
                meta_list.append((element, resname, atom_name))

    if not coords_list:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.stack(coords_list, axis=0).astype(np.float32), meta_list


def _extract_heavy_coords_from_mol(mol: Any) -> np.ndarray | None:
    """Extract 3D heavy-atom coordinates from an RDKit Mol conformer."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords: list[list[float]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def _parse_mol2_heavy_coords(mol2_path: Path) -> np.ndarray | None:
    """Fallback TRIPOS MOL2 heavy-atom coordinate parser independent of RDKit sanitization."""
    coords: list[list[float]] = []
    in_atom = False
    with mol2_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            s = line.strip()
            if s.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if s.startswith("@<TRIPOS>"):
                if in_atom:
                    break
                continue
            if in_atom and s:
                parts = s.split()
                if len(parts) >= 6:
                    atom_type = parts[5].split(".")[0].upper()
                    if atom_type == "H":
                        continue
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def _parse_sdf_heavy_coords(sdf_path: Path) -> np.ndarray | None:
    """Fallback V2000 SDF heavy-atom coordinate parser when RDKit kekulization fails."""
    lines = sdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 5:
        return None
    counts_line = lines[3]
    try:
        n_atoms = int(counts_line[:3].strip())
    except ValueError:
        return None
    coords: list[list[float]] = []
    for line in lines[4 : 4 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            elem = parts[3].strip().upper()
            if elem == "H":
                continue
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def featurize_ligand_mol(
    mol: Any,
    *,
    fp_size: int = LIGAND_FP_DIM,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Compute canonical SMILES, ECFP4 fingerprint (float32), and 6 descriptors from a Mol."""
    can_smiles = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    if fp_size == LIGAND_FP_DIM:
        fp_arr = np.asarray(_MORGAN_GEN.GetFingerprintAsNumPy(mol), dtype=np.float32)
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        fp_arr = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)

    props = np.asarray(
        [
            float(Descriptors.MolWt(mol)),  # type: ignore[attr-defined]
            float(Crippen.MolLogP(mol)),  # type: ignore[attr-defined]
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(QED.qed(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
        ],
        dtype=np.float32,
    )
    return can_smiles, fp_arr, props


def featurize_ligand_smiles(
    smiles: str,
    *,
    fp_size: int = LIGAND_FP_DIM,
    strip_salts: bool = True,
) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Canonicalize a SMILES string and compute its ECFP4 fingerprint and 6 properties."""
    can = canonicalize_smiles(smiles, strip_salts=strip_salts)
    if can is None:
        return None
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return None
    return featurize_ligand_mol(mol, fp_size=fp_size)


def _load_ligand_and_coords(
    ligand_sdf: Path | None,
    ligand_mol2: Path | None,
    fallback_smiles: str | None,
) -> tuple[Any | None, np.ndarray | None]:
    """Load both a sanitized RDKit Mol and 3D heavy-atom coordinates via 3-tier fallback.

    Separates 3D crystal coordinates from RDKit kekulization so complexes whose PDBbind SDF
    fails aromaticity sanitization still receive exact 3D pocket centering and 4.5 A contact
    masks alongside valid RDKit descriptors from ``.mol2`` or ``LP_PDBBind.csv`` SMILES.
    """
    sanitized_mol: Any | None = None
    lig_coords: np.ndarray | None = None

    if ligand_sdf is not None and ligand_sdf.exists():
        suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=True, removeHs=True)
        if len(suppl) > 0 and suppl[0] is not None:
            sanitized_mol = suppl[0]
            lig_coords = _extract_heavy_coords_from_mol(sanitized_mol)
        if lig_coords is None:
            raw_suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False, removeHs=False)
            if len(raw_suppl) > 0 and raw_suppl[0] is not None:
                lig_coords = _extract_heavy_coords_from_mol(raw_suppl[0])
        if lig_coords is None:
            lig_coords = _parse_sdf_heavy_coords(ligand_sdf)

    if (
        (sanitized_mol is None or lig_coords is None)
        and ligand_mol2 is not None
        and ligand_mol2.exists()
    ):
        mol2_san = Chem.MolFromMol2File(str(ligand_mol2), sanitize=True, removeHs=True)
        if sanitized_mol is None and mol2_san is not None:
            sanitized_mol = mol2_san
        if lig_coords is None and mol2_san is not None:
            lig_coords = _extract_heavy_coords_from_mol(mol2_san)
        if lig_coords is None:
            lig_coords = _parse_mol2_heavy_coords(ligand_mol2)

    if sanitized_mol is None and fallback_smiles:
        can = canonicalize_smiles(fallback_smiles, strip_salts=True)
        if can is not None:
            sanitized_mol = Chem.MolFromSmiles(can)

    return sanitized_mol, lig_coords


def extract_pocket_from_files(
    pocket_pdb: str | Path,
    *,
    ligand_sdf: str | Path | None = None,
    ligand_mol2: str | Path | None = None,
    fallback_smiles: str | None = None,
    pdb_id: str | None = None,
    pkd: float = float("nan"),
    pocket_cutoff: float = 10.0,
    contact_cutoff: float = 4.5,
    max_atoms: int = 384,
) -> PocketFeatures:
    """Extract centered 3D coordinates, 52-dim E(3)-invariant features, and ligand targets.

    Selects all protein heavy atoms within ``pocket_cutoff`` Angstroms of ANY ligand heavy
    atom (capped at ``max_atoms`` closest atoms), centers coordinates on the ligand centroid
    in ``float32``, and computes ``contact_mask`` (within ``contact_cutoff`` = 4.5 A).
    """
    pocket_path = Path(pocket_pdb)
    pid = (pdb_id or pocket_path.stem.split("_")[0]).strip().lower()
    prot_coords, prot_meta = _load_protein_heavy_atoms(pocket_path)
    if len(prot_coords) == 0:
        raise ValueError(f"No heavy protein atoms found in {pocket_path}")

    sdf_path = Path(ligand_sdf) if ligand_sdf is not None else None
    mol2_path = Path(ligand_mol2) if ligand_mol2 is not None else None
    lig_mol, lig_coords = _load_ligand_and_coords(sdf_path, mol2_path, fallback_smiles)

    if lig_coords is not None and len(lig_coords) > 0:
        center = lig_coords.mean(axis=0).astype(np.float32)
        # Pairwise distance from every pocket heavy atom to every ligand heavy atom
        diff = prot_coords[:, None, :] - lig_coords[None, :, :]
        min_dist = np.sqrt(np.sum(diff * diff, axis=-1).min(axis=1)).astype(np.float32)

        in_shell = np.nonzero(min_dist <= float(pocket_cutoff))[0]
        if len(in_shell) < 16:
            # Guard against degenerate/distant structures by keeping nearest available atoms
            order = np.argsort(min_dist)[: min(max_atoms, len(min_dist))]
        elif len(in_shell) > max_atoms:
            sub_order = np.argsort(min_dist[in_shell])[:max_atoms]
            order = in_shell[sub_order]
        else:
            order = in_shell

        sel_coords = prot_coords[order]
        sel_meta = [prot_meta[int(i)] for i in order.tolist()]
        contact_mask = (min_dist[order] <= float(contact_cutoff)).astype(bool)
    else:
        center = prot_coords.mean(axis=0).astype(np.float32)
        dists = np.linalg.norm(prot_coords - center[None, :], axis=-1).astype(np.float32)
        in_shell = np.nonzero(dists <= float(pocket_cutoff))[0]
        if len(in_shell) < 16:
            order = np.argsort(dists)[: min(max_atoms, len(dists))]
        elif len(in_shell) > max_atoms:
            order = in_shell[np.argsort(dists[in_shell])[:max_atoms]]
        else:
            order = in_shell
        sel_coords = prot_coords[order]
        sel_meta = [prot_meta[int(i)] for i in order.tolist()]
        contact_mask = np.zeros(len(sel_coords), dtype=bool)

    centered_coords = (sel_coords - center[None, :]).astype(np.float32)
    node_feats = _compute_52d_node_features(sel_meta, centered_coords, pocket_cutoff=pocket_cutoff)

    can_smiles: str | None = None
    ligand_fp: np.ndarray | None = None
    ligand_props: np.ndarray | None = None
    if lig_mol is not None:
        can_smiles, ligand_fp, ligand_props = featurize_ligand_mol(lig_mol)

    return PocketFeatures(
        pdb_id=pid,
        coords=centered_coords,
        node_feats=node_feats,
        contact_mask=contact_mask,
        center=center,
        canonical_smiles=can_smiles,
        ligand_fp=ligand_fp,
        ligand_props=ligand_props,
        pkd=float(pkd),
    )


def featurize_pdb_pocket_inference(
    pdb_path: str | Path,
    center: Sequence[float] | np.ndarray | None = None,
    *,
    pocket_cutoff: float = 10.0,
    max_atoms: int = 384,
    pdb_id: str | None = None,
) -> PocketFeatures:
    """Featurize a novel protein PDB/mmCIF pocket for ligand-free de novo inference.

    Args:
        pdb_path: Path to a protein or pre-cut pocket PDB/mmCIF structure.
        center: Optional 3D binding-site center ``(x, y, z)`` in Angstroms. If ``None``,
            the heavy-atom centroid of ``pdb_path`` is used.
        pocket_cutoff: Inclusion radius in Angstroms from ``center``.
        max_atoms: Maximum number of heavy atoms retained (closest to ``center``).
        pdb_id: Optional identifier label.

    Returns:
        ``PocketFeatures`` with centered ``float32`` coordinates and 52-dim E(3)-invariant
        node features (ligand fields set to ``None``).
    """
    path = Path(pdb_path)
    pid = (pdb_id or path.stem.split("_")[0]).strip().lower()
    prot_coords, prot_meta = _load_protein_heavy_atoms(path)
    if len(prot_coords) == 0:
        raise ValueError(f"No heavy protein atoms found in {path}")

    if center is None:
        center_arr = prot_coords.mean(axis=0).astype(np.float32)
    else:
        center_arr = np.asarray(center, dtype=np.float32).reshape(3)

    dists = np.linalg.norm(prot_coords - center_arr[None, :], axis=-1).astype(np.float32)
    in_shell = np.nonzero(dists <= float(pocket_cutoff))[0]
    if len(in_shell) < 16:
        order = np.argsort(dists)[: min(max_atoms, len(dists))]
    elif len(in_shell) > max_atoms:
        order = in_shell[np.argsort(dists[in_shell])[:max_atoms]]
    else:
        order = in_shell

    sel_coords = prot_coords[order]
    sel_meta = [prot_meta[int(i)] for i in order.tolist()]
    centered_coords = (sel_coords - center_arr[None, :]).astype(np.float32)
    node_feats = _compute_52d_node_features(sel_meta, centered_coords, pocket_cutoff=pocket_cutoff)

    return PocketFeatures(
        pdb_id=pid,
        coords=centered_coords,
        node_feats=node_feats,
        contact_mask=np.zeros(len(sel_coords), dtype=bool),
        center=center_arr,
        canonical_smiles=None,
        ligand_fp=None,
        ligand_props=None,
        pkd=float("nan"),
    )


__all__ = [
    "LIGAND_FP_DIM",
    "LIGAND_PROP_NAMES",
    "POCKET_NODE_DIM",
    "PocketFeatures",
    "extract_pocket_from_files",
    "featurize_ligand_mol",
    "featurize_ligand_smiles",
    "featurize_pdb_pocket_inference",
]
