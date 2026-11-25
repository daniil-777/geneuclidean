"""Exact Geometric 3D Non-Covalent Protein-Ligand Interaction Profiler & RMSD Suite.

Implements a pure-Python, deterministic, PLIP-inspired (Salentin et al., *Nucleic Acids
Research* 2015) geometric profiler for 3D protein-ligand complexes using only RDKit,
BioPython, and NumPy — zero external binaries or license-restricted toolkits required.

Supported non-covalent interaction channels:
1. **Hydrogen Bonds** (``hbond``): Heavy-atom donor-acceptor distance ``<= 3.5 Å`` with
   explicit ``D-H···A`` angle validation (``>= 120°``) whenever donor hydrogens are
   present, falling back cleanly to heavy-atom geometry on unprotonated crystal structures.
2. **Salt Bridges** (``salt_bridge``): Oppositely charged protein residues (anionic
   ``ASP``/``GLU`` carboxylates; cationic ``ARG``/``LYS``/``HIS``) and physiological-pH 7.4
   ionizable ligand groups (protonated/basic amines, amidines, guanidines, carboxylates,
   sulfonates, phosphonates, tetrazoles) within ``<= 4.2 Å``. Crucially captures the
   conserved ``Asp^{3.32}`` ionic anchor in aminergic CNS GPCRs.
3. **π–π Stacking** (``pi_stacking``): Aromatic ring centroids (``PHE``, ``TYR``, ``TRP``
   5- & 6-membered rings, ``HIS`` vs. ligand aromatic rings) within ``<= 5.5 Å`` with
   SVD plane-normal angles distinguishing parallel-displaced and T-shaped geometries.
4. **Cation–π Interactions** (``cation_pi``): Cationic center to aromatic ring centroid
   within ``<= 5.5 Å`` and bounded lateral offset above the π-electron cloud.
5. **Hydrophobic Contacts** (``hydrophobic``): Apolar protein carbon to apolar ligand
   carbon contacts within ``<= 4.0 Å``, deduplicated per ``(protein_residue, ligand_atom)``
   pair to the single closest contact.
6. **Halogen Bonds** (``halogen_bond``): Ligand ``C-Cl``/``C-Br``/``C-I`` sigma-hole to
   protein ``O``/``N``/``S`` lone-pair acceptors within ``<= 3.7 Å`` with ``C-X···A >= 120°``.

Also provides:
- **3D Protein-Ligand Interaction Fingerprint (PLIF) Tanimoto Recovery** via
  :meth:`InteractionProfile.tanimoto_to` for comparing generated poses against native
  co-crystal binding modes.
- **Symmetry-Corrected Heavy-Atom RMSD** (:func:`compute_heavy_atom_rmsd`) supporting
  both in-place docking pose RMSD (``align=False``, default) and Kabsch-superimposed
  conformer RMSD (``align=True`` via RDKit ``rdMolAlign.GetBestRMS``).
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdMolAlign

# ---------------------------------------------------------------------------
# Public Constants & Type Aliases
# ---------------------------------------------------------------------------

InteractionKind = Literal[
    "hbond",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "cation_pi",
    "halogen_bond",
]

ALL_INTERACTION_KINDS: tuple[InteractionKind, ...] = (
    "hbond",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "cation_pi",
    "halogen_bond",
)

# Importance ordering for deterministic reporting and residue ranking
_KIND_PRIORITY: dict[str, int] = {
    "salt_bridge": 0,
    "hbond": 1,
    "pi_stacking": 2,
    "cation_pi": 3,
    "halogen_bond": 4,
    "hydrophobic": 5,
}

# ---------------------------------------------------------------------------
# Protein Structural Biochemistry Tables
# ---------------------------------------------------------------------------

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL"})

_PROTEIN_RESNAMES: frozenset[str] = frozenset(
    {
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
        # Common protonation / post-translational variants in PDB structures
        "HID",
        "HIE",
        "HIP",
        "ASH",
        "GLH",
        "LYN",
        "CYX",
        "CYM",
        "MSE",
        "SEP",
        "TPO",
        "PTR",
        "CSO",
        "KCX",
    }
)

_TWO_LETTER_ELEMENTS: frozenset[str] = frozenset(
    {"SE", "CL", "BR", "FE", "ZN", "MG", "CA", "MN", "CU", "NA", "CO", "NI"}
)

# Sidechain H-bond donors by residue name
_PROTEIN_SIDECHAIN_DONORS: dict[str, frozenset[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "ASN": frozenset({"ND2"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "HID": frozenset({"ND1"}),
    "HIE": frozenset({"NE2"}),
    "HIP": frozenset({"ND1", "NE2"}),
    "LYS": frozenset({"NZ"}),
    "LYN": frozenset({"NZ"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TRP": frozenset({"NE1"}),
    "TYR": frozenset({"OH"}),
    "CYS": frozenset({"SG"}),
    "ASH": frozenset({"OD1", "OD2"}),
    "GLH": frozenset({"OE1", "OE2"}),
}

# Sidechain H-bond acceptors by residue name
_PROTEIN_SIDECHAIN_ACCEPTORS: dict[str, frozenset[str]] = {
    "ASP": frozenset({"OD1", "OD2"}),
    "ASH": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
    "GLH": frozenset({"OE1", "OE2"}),
    "ASN": frozenset({"OD1"}),
    "GLN": frozenset({"OE1"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "HID": frozenset({"NE2"}),
    "HIE": frozenset({"ND1"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TYR": frozenset({"OH"}),
    "MET": frozenset({"SD"}),
    "CYS": frozenset({"SG"}),
    "CYM": frozenset({"SG"}),
    "LYN": frozenset({"NZ"}),
}

# Anionic (negatively charged at physiological pH 7.4) sidechain atoms
_PROTEIN_ANIONIC_ATOMS: dict[str, frozenset[str]] = {
    "ASP": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
}

# Cationic (positively charged / ionizable at physiological pH 7.4) sidechain atoms
_PROTEIN_CATIONIC_ATOMS: dict[str, frozenset[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "LYS": frozenset({"NZ"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "HIP": frozenset({"ND1", "NE2"}),
}

# Protein aromatic ring definitions (each residue can have multiple fused rings, e.g. TRP)
_PROTEIN_AROMATIC_RINGS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "PTR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HID": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HIE": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HIP": (("CG", "ND1", "CD2", "CE1", "NE2"),),
}

# Halogen-bond lone-pair acceptors on protein residues (plus backbone O / OXT)
_PROTEIN_HALOGEN_ACCEPTORS: dict[str, frozenset[str]] = {
    "ASP": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
    "ASN": frozenset({"OD1"}),
    "GLN": frozenset({"OE1"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TYR": frozenset({"OH"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "HID": frozenset({"NE2"}),
    "HIE": frozenset({"ND1"}),
    "MET": frozenset({"SD"}),
    "CYS": frozenset({"SG"}),
}

# ---------------------------------------------------------------------------
# Pre-compiled RDKit SMARTS for Physiological pH 7.4 Ligand Chemistry
# ---------------------------------------------------------------------------

# Basic aliphatic amines (primary/secondary/tertiary), protonated ammoniums, and
# basic amidines/guanidines (e.g. dopamine, haloperidol, clonidine, guanfacine).
# Excludes amides, sulfonamides, nitro groups, and weakly basic anilines.
_SMARTS_LIGAND_BASIC_AMINE = Chem.MolFromSmarts("[NX3,NX4+;!$(NC=O);!$(NS=O);!$(N=O);!$(N[a])]")
_SMARTS_LIGAND_AMIDINE_GUANIDINE = Chem.MolFromSmarts("[NX2]=[CX3]([NX3])[#6,N]")

# Acidic functional groups deprotonated at physiological pH 7.4
_SMARTS_LIGAND_CARBOXYLATE = Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]")
_SMARTS_LIGAND_SULFO_PHOSPHO_ACID = Chem.MolFromSmarts("[SX4,PX4](=O)(=O)[OX1H0-,OX2H1]")
_SMARTS_LIGAND_TETRAZOLE = Chem.MolFromSmarts("c1nnn[nH,n-]1")

# Nitro groups carry formal +1/-1 charges in Kekulé representation but are neutral dipoles
_SMARTS_LIGAND_NITRO = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")

# Standard H-bond acceptor SMARTS (lone-pair bearing N/O/S excluding pyrrole NH and amides)
_SMARTS_LIGAND_HBA = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)


# ---------------------------------------------------------------------------
# Public Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionRecord:
    """Single geometrically verified 3D protein-ligand non-covalent contact.

    Attributes
    ----------
    kind:
        One of ``"hbond"``, ``"salt_bridge"``, ``"hydrophobic"``, ``"pi_stacking"``,
        ``"cation_pi"``, or ``"halogen_bond"``.
    protein_residue:
        Canonical residue identifier ``"<RESNAME><RESSEQ>:<CHAIN>"`` (e.g. ``"ASP103:A"``).
    protein_atom:
        PDB atom name (e.g. ``"OD1"``, ``"OG"``) or ``"RING"`` for aromatic ring centroids.
    ligand_atom_idx:
        Atom index in ``ligand_mol_3d`` (for aromatic ring contacts, the lowest heavy-atom
        index belonging to the ligand ring).
    ligand_element:
        Element symbol of the ligand atom (e.g. ``"N"``, ``"O"``, ``"C"``, ``"Cl"``) or
        ``"RING"`` for ligand aromatic ring centroids.
    distance_angstrom:
        Exact 3D Euclidean distance between interaction anchors in Ångströms.
    angle_deg:
        Characteristic interaction geometry angle in degrees when applicable:
        ``D-H···A`` angle for H-bonds (if explicit H present), acute ring-normal angle
        (``0°–90°``) for π–π stacking, or ``C-X···A`` sigma-hole angle for halogen bonds.
    protein_xyz:
        3D Cartesian coordinates ``(x, y, z)`` of the protein interaction anchor (Å).
    ligand_xyz:
        3D Cartesian coordinates ``(x, y, z)`` of the ligand interaction anchor (Å).
    subtype:
        Optional geometric/directional annotation (e.g. ``"parallel"``, ``"t_shaped"``,
        ``"donor_protein"``, ``"donor_ligand"``, ``"protein_anion"``, ``"protein_cation"``).
    """

    kind: InteractionKind
    protein_residue: str
    protein_atom: str
    ligand_atom_idx: int
    ligand_element: str
    distance_angstrom: float
    angle_deg: float | None
    protein_xyz: tuple[float, float, float]
    ligand_xyz: tuple[float, float, float]
    subtype: str = ""

    def fingerprint_key(self) -> tuple[str, str]:
        """Return the canonical ``(protein_residue, kind)`` PLIF bit tuple."""
        return (self.protein_residue, self.kind)

    def to_dict(self) -> dict[str, Any]:
        """Serialize interaction record to a JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "protein_residue": self.protein_residue,
            "protein_atom": self.protein_atom,
            "ligand_atom_idx": int(self.ligand_atom_idx),
            "ligand_element": self.ligand_element,
            "distance_angstrom": round(float(self.distance_angstrom), 3),
            "angle_deg": round(float(self.angle_deg), 2) if self.angle_deg is not None else None,
            "protein_xyz": [round(float(v), 4) for v in self.protein_xyz],
            "ligand_xyz": [round(float(v), 4) for v in self.ligand_xyz],
            "subtype": self.subtype,
        }


@dataclass
class InteractionProfile:
    """Aggregated 3D non-covalent interaction profile for a protein-ligand complex.

    Attributes
    ----------
    interactions:
        Deterministically ordered list of detected :class:`InteractionRecord` entries.
    counts_by_type:
        Total interaction count per interaction kind (always contains all 6 keys).
    key_residues:
        Unique interacting protein residues ordered by interaction importance and
        shortest contact distance.
    """

    interactions: list[InteractionRecord] = field(default_factory=list)
    counts_by_type: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in ALL_INTERACTION_KINDS}
    )
    key_residues: list[str] = field(default_factory=list)

    def fingerprint_set(self, *, include_hydrophobic: bool = True) -> frozenset[tuple[str, str]]:
        """Extract the binary 3D Protein-Ligand Interaction Fingerprint (PLIF) set.

        Each bit corresponds to a unique ``(protein_residue, interaction_kind)`` pair.
        """
        return frozenset(
            rec.fingerprint_key()
            for rec in self.interactions
            if include_hydrophobic or rec.kind != "hydrophobic"
        )

    def tanimoto_to(
        self,
        other: InteractionProfile,
        *,
        include_hydrophobic: bool = True,
    ) -> float:
        """Compute quantitative 3D PLIF Tanimoto (Jaccard) similarity to ``other``.

        Compares the set of ``(protein_residue, interaction_kind)`` contacts between
        two binding poses in the same protein pocket (e.g., generated pose vs. native
        co-crystal ligand).

        Returns
        -------
        float
            Similarity in ``[0.0, 1.0]``. Returns ``1.0`` when both profiles are empty,
            and ``0.0`` when exactly one profile is empty.
        """
        fp_a = self.fingerprint_set(include_hydrophobic=include_hydrophobic)
        fp_b = other.fingerprint_set(include_hydrophobic=include_hydrophobic)
        if not fp_a and not fp_b:
            return 1.0
        union = fp_a | fp_b
        if not union:
            return 0.0
        return float(len(fp_a & fp_b)) / float(len(union))

    def residue_recovery_rate(
        self,
        reference: InteractionProfile,
        *,
        include_hydrophobic: bool = True,
    ) -> float:
        """Compute fraction of ``reference`` PLIF interactions recovered by ``self``."""
        fp_ref = reference.fingerprint_set(include_hydrophobic=include_hydrophobic)
        if not fp_ref:
            return 1.0
        fp_self = self.fingerprint_set(include_hydrophobic=include_hydrophobic)
        return float(len(fp_self & fp_ref)) / float(len(fp_ref))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete interaction profile to a JSON-compatible dictionary."""
        return {
            "counts_by_type": dict(self.counts_by_type),
            "key_residues": list(self.key_residues),
            "total_interactions": len(self.interactions),
            "interactions": [rec.to_dict() for rec in self.interactions],
        }


# ---------------------------------------------------------------------------
# Internal Geometric & Chemical Feature Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProteinAtomFeat:
    """Parsed protein heavy atom with chemical role flags and attached hydrogens."""

    residue_id: str
    resname: str
    atom_name: str
    element: str
    xyz: NDArray[np.float64]
    is_backbone: bool
    is_donor: bool
    is_acceptor: bool
    is_anionic: bool
    is_cationic: bool
    is_hydrophobic: bool
    is_halogen_acceptor: bool
    attached_h_xyz: tuple[NDArray[np.float64], ...] = ()


@dataclass(frozen=True)
class _AromaticRingFeat:
    """Planar aromatic ring centroid and SVD-derived unit normal vector."""

    label: str  # protein_residue for protein rings, or "LIG" for ligand rings
    atom_idx: int  # 0 for protein, lowest heavy-atom index for ligand
    centroid: NDArray[np.float64]
    normal: NDArray[np.float64]


@dataclass(frozen=True)
class _LigandAtomFeat:
    """Parsed ligand heavy atom with 3D coordinates and physiological pH 7.4 role flags."""

    atom_idx: int
    element: str
    xyz: NDArray[np.float64]
    is_donor: bool
    is_acceptor: bool
    is_cationic: bool
    is_anionic: bool
    is_hydrophobic: bool
    attached_h_xyz: tuple[NDArray[np.float64], ...] = ()
    halogen_carbon_xyz: NDArray[np.float64] | None = None  # bonded C for Cl/Br/I


@dataclass
class _ParsedProteinPocket:
    """All extracted protein interaction anchors from a PDB pocket."""

    atoms: list[_ProteinAtomFeat]
    aromatic_rings: list[_AromaticRingFeat]


@dataclass
class _ParsedLigand:
    """All extracted ligand interaction anchors from an RDKit 3D Mol."""

    atoms: list[_LigandAtomFeat]
    aromatic_rings: list[_AromaticRingFeat]


# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------


def _angle_deg_at_vertex(
    p1: NDArray[np.float64],
    vertex: NDArray[np.float64],
    p2: NDArray[np.float64],
) -> float:
    """Compute the angle ``p1-vertex-p2`` in degrees (range ``[0, 180]``)."""
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_val = float(np.dot(v1, v2) / (n1 * n2))
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def _compute_ring_plane(
    coords: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact ring centroid and unit plane normal via SVD."""
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64) if norm < 1e-8 else normal / norm
    return centroid.astype(np.float64), normal.astype(np.float64)


def _lateral_offset(
    vec: NDArray[np.float64],
    ring_normal: NDArray[np.float64],
) -> float:
    """Return the in-plane lateral offset of ``vec`` perpendicular to ``ring_normal``."""
    proj = float(np.dot(vec, ring_normal)) * ring_normal
    return float(np.linalg.norm(vec - proj))


def _to_xyz_tuple(arr: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert a 3-element numpy vector to a clean float 3-tuple."""
    return (float(arr[0]), float(arr[1]), float(arr[2]))


# ---------------------------------------------------------------------------
# Protein Pocket Parsing & Feature Classification
# ---------------------------------------------------------------------------


def _infer_pdb_element(raw_element: str, atom_name: str) -> str:
    """Infer canonical uppercase element symbol from PDB element column or atom name."""
    el = raw_element.strip().upper()
    if el and el.isalpha():
        return el
    cleaned = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if len(cleaned) >= 2 and cleaned[:2] in _TWO_LETTER_ELEMENTS:
        return cleaned[:2]
    return cleaned[:1] if cleaned else "C"


def _format_residue_id(residue: Residue, chain_id: str) -> str:
    """Format canonical residue identifier ``<RESNAME><RESSEQ>[<ICODE>]:<CHAIN>``."""
    resname = residue.get_resname().strip().upper()
    _, resseq, icode = residue.id
    icode_clean = icode.strip() if isinstance(icode, str) else ""
    chain_clean = chain_id.strip() or "A"
    return f"{resname}{resseq}{icode_clean}:{chain_clean}"


def _parse_protein_pocket_from_handle(handle: io.StringIO | str | Path) -> _ParsedProteinPocket:
    """Parse protein pocket heavy atoms, hydrogens, and aromatic rings with BioPython."""
    parser = PDBParser(QUIET=True)
    if isinstance(handle, (str, Path)):
        structure = parser.get_structure("pocket", str(handle))
    else:
        structure = parser.get_structure("pocket", handle)

    models = list(structure.get_models())
    if not models:
        return _ParsedProteinPocket(atoms=[], aromatic_rings=[])
    model = models[0]

    parsed_atoms: list[_ProteinAtomFeat] = []
    parsed_rings: list[_AromaticRingFeat] = []

    for chain in model:
        chain_id = str(chain.id)
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            if resname in _WATER_RESNAMES:
                continue
            hetflag = residue.id[0]
            if hetflag != " " and resname not in _PROTEIN_RESNAMES:
                continue

            res_id = _format_residue_id(residue, chain_id)
            heavy_records: list[tuple[str, str, NDArray[np.float64]]] = []
            hydrogen_coords: list[NDArray[np.float64]] = []

            for atom in residue.get_atoms():
                atom_name = atom.get_name().strip().upper()
                element = _infer_pdb_element(getattr(atom, "element", ""), atom_name)
                xyz = np.asarray(atom.get_coord(), dtype=np.float64)
                if element in {"H", "D"}:
                    hydrogen_coords.append(xyz)
                else:
                    heavy_records.append((atom_name, element, xyz))

            if not heavy_records:
                continue

            heavy_by_name: dict[str, tuple[str, NDArray[np.float64]]] = {
                name: (el, xyz) for name, el, xyz in heavy_records
            }

            # Map each explicit hydrogen to its nearest heavy atom within covalent bond distance
            attached_hs: dict[str, list[NDArray[np.float64]]] = {
                name: [] for name, _, _ in heavy_records
            }
            if hydrogen_coords:
                heavy_names = [r[0] for r in heavy_records]
                heavy_xyz_mat = np.stack([r[2] for r in heavy_records], axis=0)
                for h_xyz in hydrogen_coords:
                    dists = np.linalg.norm(heavy_xyz_mat - h_xyz, axis=1)
                    nearest_idx = int(np.argmin(dists))
                    if float(dists[nearest_idx]) <= 1.30:
                        attached_hs[heavy_names[nearest_idx]].append(h_xyz)

            # Determine intra-residue heteroatom adjacency for accurate apolar carbon typing
            heavy_xyz_mat = np.stack([r[2] for r in heavy_records], axis=0)
            dist_mat = np.linalg.norm(
                heavy_xyz_mat[:, None, :] - heavy_xyz_mat[None, :, :], axis=-1
            )

            sc_donors = _PROTEIN_SIDECHAIN_DONORS.get(resname, frozenset())
            sc_acceptors = _PROTEIN_SIDECHAIN_ACCEPTORS.get(resname, frozenset())
            sc_anions = _PROTEIN_ANIONIC_ATOMS.get(resname, frozenset())
            sc_cations = _PROTEIN_CATIONIC_ATOMS.get(resname, frozenset())
            sc_hal_acc = _PROTEIN_HALOGEN_ACCEPTORS.get(resname, frozenset())

            # For HIS with explicit sidechain hydrogens, distinguish donor vs acceptor N
            his_has_sc_h = resname == "HIS" and bool(
                attached_hs.get("ND1") or attached_hs.get("NE2")
            )

            for idx, (atom_name, element, xyz) in enumerate(heavy_records):
                is_backbone = atom_name in {"N", "CA", "C", "O", "OXT"}
                atom_hs = tuple(attached_hs[atom_name])

                # H-bond donor classification
                is_donor = False
                if atom_name == "N":
                    # Proline backbone N has no amide proton unless N-terminal with explicit H
                    is_donor = resname != "PRO" or len(atom_hs) > 0
                elif atom_name in sc_donors:
                    is_donor = len(atom_hs) > 0 if his_has_sc_h else True

                # H-bond acceptor classification
                is_acceptor = False
                if atom_name in {"O", "OXT"}:
                    is_acceptor = True
                elif atom_name in sc_acceptors:
                    is_acceptor = len(atom_hs) == 0 if his_has_sc_h else True

                # Ionic groups (pH 7.4)
                is_anionic = (atom_name in sc_anions) or (atom_name == "OXT")
                is_cationic = atom_name in sc_cations

                # Halogen-bond lone-pair acceptors
                is_halogen_acceptor = (atom_name in {"O", "OXT"}) or (atom_name in sc_hal_acc)

                # Apolar hydrophobic carbon: non-backbone C not bonded to any N, O, S, or P
                is_hydrophobic = False
                if element == "C" and atom_name not in {"C", "CA"}:
                    bonded_to_hetero = False
                    for j, (_, nbr_el, _) in enumerate(heavy_records):
                        if j == idx:
                            continue
                        if nbr_el in {"N", "O", "S", "P"}:
                            bond_cutoff = 1.95 if nbr_el in {"S", "P"} else 1.75
                            if float(dist_mat[idx, j]) <= bond_cutoff:
                                bonded_to_hetero = True
                                break
                    is_hydrophobic = not bonded_to_hetero

                parsed_atoms.append(
                    _ProteinAtomFeat(
                        residue_id=res_id,
                        resname=resname,
                        atom_name=atom_name,
                        element=element,
                        xyz=xyz,
                        is_backbone=is_backbone,
                        is_donor=is_donor,
                        is_acceptor=is_acceptor,
                        is_anionic=is_anionic,
                        is_cationic=is_cationic,
                        is_hydrophobic=is_hydrophobic,
                        is_halogen_acceptor=is_halogen_acceptor,
                        attached_h_xyz=atom_hs,
                    )
                )

            # Extract aromatic ring planes (e.g., PHE, TYR, HIS, and both rings of TRP)
            for ring_atom_names in _PROTEIN_AROMATIC_RINGS.get(resname, ()):
                if all(name in heavy_by_name for name in ring_atom_names):
                    ring_coords = np.stack(
                        [heavy_by_name[name][1] for name in ring_atom_names], axis=0
                    )
                    centroid, normal = _compute_ring_plane(ring_coords)
                    parsed_rings.append(
                        _AromaticRingFeat(
                            label=res_id,
                            atom_idx=0,
                            centroid=centroid,
                            normal=normal,
                        )
                    )

    return _ParsedProteinPocket(atoms=parsed_atoms, aromatic_rings=parsed_rings)


# ---------------------------------------------------------------------------
# Ligand 3D Chemical & Geometric Feature Extraction
# ---------------------------------------------------------------------------


def _match_atom_indices(mol: Chem.Mol, query: Chem.Mol | None) -> set[int]:
    """Return all atom indices matched by a compiled SMARTS query."""
    if query is None:
        return set()
    indices: set[int] = set()
    for match in mol.GetSubstructMatches(query):
        indices.update(match)
    return indices


def _parse_ligand_3d(ligand_mol_3d: Chem.Mol) -> _ParsedLigand:
    """Extract 3D coordinates, pH 7.4 ionizable roles, rings, and donors/acceptors."""
    if ligand_mol_3d is None or ligand_mol_3d.GetNumConformers() == 0:
        raise ValueError("ligand_mol_3d must be a valid RDKit Mol with a 3D conformer.")

    conf = ligand_mol_3d.GetConformer()

    # Pre-match SMARTS patterns for physiological pH 7.4 ionization and H-bonding
    nitro_atoms = _match_atom_indices(ligand_mol_3d, _SMARTS_LIGAND_NITRO)
    basic_amine_atoms = _match_atom_indices(ligand_mol_3d, _SMARTS_LIGAND_BASIC_AMINE)
    amidine_matches = (
        ligand_mol_3d.GetSubstructMatches(_SMARTS_LIGAND_AMIDINE_GUANIDINE)
        if _SMARTS_LIGAND_AMIDINE_GUANIDINE is not None
        else ()
    )
    for match in amidine_matches:
        # Nitrogen atoms in amidine / guanidine resonance system are cationic at pH 7.4
        for idx in match:
            if ligand_mol_3d.GetAtomWithIdx(idx).GetAtomicNum() == 7:
                basic_amine_atoms.add(idx)

    anionic_atoms: set[int] = set()
    for query in (_SMARTS_LIGAND_CARBOXYLATE, _SMARTS_LIGAND_SULFO_PHOSPHO_ACID):
        if query is not None:
            for match in ligand_mol_3d.GetSubstructMatches(query):
                for idx in match:
                    if ligand_mol_3d.GetAtomWithIdx(idx).GetAtomicNum() == 8:
                        anionic_atoms.add(idx)
    if _SMARTS_LIGAND_TETRAZOLE is not None:
        for match in ligand_mol_3d.GetSubstructMatches(_SMARTS_LIGAND_TETRAZOLE):
            for idx in match:
                if ligand_mol_3d.GetAtomWithIdx(idx).GetAtomicNum() == 7:
                    anionic_atoms.add(idx)

    hba_atoms = _match_atom_indices(ligand_mol_3d, _SMARTS_LIGAND_HBA)

    parsed_atoms: list[_LigandAtomFeat] = []
    for atom in ligand_mol_3d.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            continue
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        xyz = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        element = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()

        # Gather explicit 3D hydrogens covalently attached to this heavy atom
        attached_hs: list[NDArray[np.float64]] = []
        bonded_heavy_nums: list[int] = []
        bonded_carbon_xyz: NDArray[np.float64] | None = None

        for nbr in atom.GetNeighbors():
            nbr_num = nbr.GetAtomicNum()
            nbr_pos = conf.GetAtomPosition(nbr.GetIdx())
            nbr_xyz = np.array([nbr_pos.x, nbr_pos.y, nbr_pos.z], dtype=np.float64)
            if nbr_num == 1:
                attached_hs.append(nbr_xyz)
            else:
                bonded_heavy_nums.append(nbr_num)
                if nbr_num == 6 and bonded_carbon_xyz is None:
                    bonded_carbon_xyz = nbr_xyz

        # H-bond donor: N/O/S bearing >= 1 hydrogen (explicit or implicit)
        is_donor = atomic_num in {7, 8, 16} and atom.GetTotalNumHs(includeNeighbors=True) > 0

        # H-bond acceptor: SMARTS-matched lone-pair heteroatoms or anionic O/N/S
        is_acceptor = (idx in hba_atoms) or (
            atomic_num in {7, 8, 16} and formal_charge < 0 and idx not in nitro_atoms
        )

        # Cationic at pH 7.4: basic amine/amidine/guanidine or formal positive N/P (non-nitro)
        is_cationic = (idx not in nitro_atoms) and (
            (idx in basic_amine_atoms) or (formal_charge > 0 and atomic_num in {7, 15})
        )

        # Anionic at pH 7.4: carboxylate/sulfonate/phosphonate/tetrazole or formal negative O/S/N
        is_anionic = (idx not in nitro_atoms) and (
            (idx in anionic_atoms) or (formal_charge < 0 and atomic_num in {7, 8, 16})
        )

        # Apolar hydrophobic carbon (PLIP definition: C bonded solely to C and/or H)
        is_hydrophobic = atomic_num == 6 and all(n == 6 for n in bonded_heavy_nums)

        # Halogen bond donor (Cl, Br, I bonded to carbon)
        halogen_c = bonded_carbon_xyz if atomic_num in {17, 35, 53} else None

        parsed_atoms.append(
            _LigandAtomFeat(
                atom_idx=idx,
                element=element,
                xyz=xyz,
                is_donor=is_donor,
                is_acceptor=is_acceptor,
                is_cationic=is_cationic,
                is_anionic=is_anionic,
                is_hydrophobic=is_hydrophobic,
                attached_h_xyz=tuple(attached_hs),
                halogen_carbon_xyz=halogen_c,
            )
        )

    # Extract ligand aromatic rings (size >= 5 where all ring members are aromatic)
    parsed_rings: list[_AromaticRingFeat] = []
    ring_info = ligand_mol_3d.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) < 5:
            continue
        if all(ligand_mol_3d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            ring_coords = np.stack(
                [
                    np.array(
                        [
                            conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z,
                        ],
                        dtype=np.float64,
                    )
                    for i in ring
                ],
                axis=0,
            )
            centroid, normal = _compute_ring_plane(ring_coords)
            parsed_rings.append(
                _AromaticRingFeat(
                    label="LIG",
                    atom_idx=min(ring),
                    centroid=centroid,
                    normal=normal,
                )
            )

    return _ParsedLigand(atoms=parsed_atoms, aromatic_rings=parsed_rings)


# ---------------------------------------------------------------------------
# Interaction Detectors
# ---------------------------------------------------------------------------


def _detect_hbonds(
    protein: _ParsedProteinPocket,
    ligand: _ParsedLigand,
    *,
    max_dist: float,
    min_angle: float,
) -> list[InteractionRecord]:
    """Detect directional hydrogen bonds (protein->ligand and ligand->protein)."""
    # Deduplicate per (protein_residue, protein_atom, ligand_atom_idx) keeping best geometry
    best_by_pair: dict[tuple[str, str, int], InteractionRecord] = {}

    def _evaluate_pair(
        p_atom: _ProteinAtomFeat,
        l_atom: _LigandAtomFeat,
        donor_xyz: NDArray[np.float64],
        acceptor_xyz: NDArray[np.float64],
        donor_hs: Sequence[NDArray[np.float64]],
        subtype: str,
    ) -> None:
        dist = float(np.linalg.norm(p_atom.xyz - l_atom.xyz))
        if dist > max_dist or dist < 1e-6:
            return

        angle_deg: float | None = None
        if donor_hs:
            best_angle = max(
                _angle_deg_at_vertex(donor_xyz, h_xyz, acceptor_xyz) for h_xyz in donor_hs
            )
            if best_angle < min_angle:
                return
            angle_deg = best_angle

        key = (p_atom.residue_id, p_atom.atom_name, l_atom.atom_idx)
        rec = InteractionRecord(
            kind="hbond",
            protein_residue=p_atom.residue_id,
            protein_atom=p_atom.atom_name,
            ligand_atom_idx=l_atom.atom_idx,
            ligand_element=l_atom.element,
            distance_angstrom=dist,
            angle_deg=angle_deg,
            protein_xyz=_to_xyz_tuple(p_atom.xyz),
            ligand_xyz=_to_xyz_tuple(l_atom.xyz),
            subtype=subtype,
        )
        prev = best_by_pair.get(key)
        if prev is None or (
            rec.angle_deg is not None and (prev.angle_deg is None or rec.angle_deg > prev.angle_deg)
        ):
            best_by_pair[key] = rec

    p_donors = [a for a in protein.atoms if a.is_donor]
    p_acceptors = [a for a in protein.atoms if a.is_acceptor]
    l_donors = [a for a in ligand.atoms if a.is_donor]
    l_acceptors = [a for a in ligand.atoms if a.is_acceptor]

    for p_atom in p_donors:
        for l_atom in l_acceptors:
            _evaluate_pair(
                p_atom, l_atom, p_atom.xyz, l_atom.xyz, p_atom.attached_h_xyz, "donor_protein"
            )

    for l_atom in l_donors:
        for p_atom in p_acceptors:
            _evaluate_pair(
                p_atom, l_atom, l_atom.xyz, p_atom.xyz, l_atom.attached_h_xyz, "donor_ligand"
            )

    return list(best_by_pair.values())


def _detect_salt_bridges(
    protein: _ParsedProteinPocket,
    ligand: _ParsedLigand,
    *,
    max_dist: float,
) -> list[InteractionRecord]:
    """Detect electrostatic salt bridges (deduplicated per residue-ligand atom pair)."""
    best_by_pair: dict[tuple[str, int], InteractionRecord] = {}

    def _record_if_closer(
        p_atom: _ProteinAtomFeat,
        l_atom: _LigandAtomFeat,
        subtype: str,
    ) -> None:
        dist = float(np.linalg.norm(p_atom.xyz - l_atom.xyz))
        if dist > max_dist or dist < 1e-6:
            return
        key = (p_atom.residue_id, l_atom.atom_idx)
        prev = best_by_pair.get(key)
        if prev is None or dist < prev.distance_angstrom:
            best_by_pair[key] = InteractionRecord(
                kind="salt_bridge",
                protein_residue=p_atom.residue_id,
                protein_atom=p_atom.atom_name,
                ligand_atom_idx=l_atom.atom_idx,
                ligand_element=l_atom.element,
                distance_angstrom=dist,
                angle_deg=None,
                protein_xyz=_to_xyz_tuple(p_atom.xyz),
                ligand_xyz=_to_xyz_tuple(l_atom.xyz),
                subtype=subtype,
            )

    # Anionic protein (ASP/GLU) <-> Cationic ligand (e.g., Asp3.32 <-> protonated amine)
    for p_atom in (a for a in protein.atoms if a.is_anionic):
        for l_atom in (a for a in ligand.atoms if a.is_cationic):
            _record_if_closer(p_atom, l_atom, "protein_anion")

    # Cationic protein (ARG/LYS/HIS) <-> Anionic ligand (e.g., carboxylate)
    for p_atom in (a for a in protein.atoms if a.is_cationic):
        for l_atom in (a for a in ligand.atoms if a.is_anionic):
            _record_if_closer(p_atom, l_atom, "protein_cation")

    return list(best_by_pair.values())


def _detect_pi_stacking(
    protein: _ParsedProteinPocket,
    ligand: _ParsedLigand,
    *,
    max_dist: float,
    max_offset: float = 2.8,
) -> list[InteractionRecord]:
    """Detect parallel-displaced and T-shaped π–π stacking between aromatic rings."""
    results: list[InteractionRecord] = []

    for p_ring in protein.aromatic_rings:
        for l_ring in ligand.aromatic_rings:
            vec = l_ring.centroid - p_ring.centroid
            dist = float(np.linalg.norm(vec))
            if dist > max_dist or dist < 1e-6:
                continue

            # Acute angle between ring normals (0° = parallel, 90° = T-shaped)
            cos_theta = min(1.0, abs(float(np.dot(p_ring.normal, l_ring.normal))))
            angle_deg = math.degrees(math.acos(cos_theta))

            # Lateral offset onto each ring plane rejects coplanar side-by-side rings
            offset_p = _lateral_offset(vec, p_ring.normal)
            offset_l = _lateral_offset(vec, l_ring.normal)
            min_offset = min(offset_p, offset_l)
            if min_offset > max_offset:
                continue

            if angle_deg <= 35.0:
                subtype = "parallel"
            elif angle_deg >= 55.0:
                subtype = "t_shaped"
            elif dist <= 5.0 and min_offset <= 2.4:
                subtype = "tilted"
            else:
                continue

            results.append(
                InteractionRecord(
                    kind="pi_stacking",
                    protein_residue=p_ring.label,
                    protein_atom="RING",
                    ligand_atom_idx=l_ring.atom_idx,
                    ligand_element="RING",
                    distance_angstrom=dist,
                    angle_deg=angle_deg,
                    protein_xyz=_to_xyz_tuple(p_ring.centroid),
                    ligand_xyz=_to_xyz_tuple(l_ring.centroid),
                    subtype=subtype,
                )
            )

    return results


def _detect_cation_pi(
    protein: _ParsedProteinPocket,
    ligand: _ParsedLigand,
    *,
    max_dist: float,
    max_offset: float = 2.5,
) -> list[InteractionRecord]:
    """Detect cation–π contacts (both protein-cation/ligand-ring and ligand-cation/protein-ring)."""
    best_by_key: dict[tuple[str, int, str], InteractionRecord] = {}

    # 1) Protein cation (ARG/LYS/HIS) <-> Ligand aromatic ring
    for p_atom in (a for a in protein.atoms if a.is_cationic):
        for l_ring in ligand.aromatic_rings:
            vec = p_atom.xyz - l_ring.centroid
            dist = float(np.linalg.norm(vec))
            if dist > max_dist or dist < 1e-6:
                continue
            if _lateral_offset(vec, l_ring.normal) > max_offset:
                continue
            cos_off = min(1.0, abs(float(np.dot(vec / dist, l_ring.normal))))
            offset_angle = math.degrees(math.acos(cos_off))
            key = (p_atom.residue_id, l_ring.atom_idx, "protein_cation")
            prev = best_by_key.get(key)
            if prev is None or dist < prev.distance_angstrom:
                best_by_key[key] = InteractionRecord(
                    kind="cation_pi",
                    protein_residue=p_atom.residue_id,
                    protein_atom=p_atom.atom_name,
                    ligand_atom_idx=l_ring.atom_idx,
                    ligand_element="RING",
                    distance_angstrom=dist,
                    angle_deg=offset_angle,
                    protein_xyz=_to_xyz_tuple(p_atom.xyz),
                    ligand_xyz=_to_xyz_tuple(l_ring.centroid),
                    subtype="protein_cation",
                )

    # 2) Ligand cation (basic amine/ammonium) <-> Protein aromatic ring (PHE/TYR/TRP/HIS)
    for p_ring in protein.aromatic_rings:
        for l_atom in (a for a in ligand.atoms if a.is_cationic):
            vec = l_atom.xyz - p_ring.centroid
            dist = float(np.linalg.norm(vec))
            if dist > max_dist or dist < 1e-6:
                continue
            if _lateral_offset(vec, p_ring.normal) > max_offset:
                continue
            cos_off = min(1.0, abs(float(np.dot(vec / dist, p_ring.normal))))
            offset_angle = math.degrees(math.acos(cos_off))
            key = (p_ring.label, l_atom.atom_idx, "ligand_cation")
            prev = best_by_key.get(key)
            if prev is None or dist < prev.distance_angstrom:
                best_by_key[key] = InteractionRecord(
                    kind="cation_pi",
                    protein_residue=p_ring.label,
                    protein_atom="RING",
                    ligand_atom_idx=l_atom.atom_idx,
                    ligand_element=l_atom.element,
                    distance_angstrom=dist,
                    angle_deg=offset_angle,
                    protein_xyz=_to_xyz_tuple(p_ring.centroid),
                    ligand_xyz=_to_xyz_tuple(l_atom.xyz),
                    subtype="ligand_cation",
                )

    return list(best_by_key.values())


def _detect_hydrophobic(
    protein: _ParsedProteinPocket,
    ligand: _ParsedLigand,
    *,
    max_dist: float,
) -> list[InteractionRecord]:
    """Detect apolar C–C contacts, deduplicated per (protein_residue, ligand_atom_idx)."""
    best_by_pair: dict[tuple[str, int], InteractionRecord] = {}

    p_hydro = [a for a in protein.atoms if a.is_hydrophobic]
    l_hydro = [a for a in ligand.atoms if a.is_hydrophobic]

    for p_atom in p_hydro:
        for l_atom in l_hydro:
            dist = float(np.linalg.norm(p_atom.xyz - l_atom.xyz))
            if dist > max_dist or dist < 1e-6:
                continue
            key = (p_atom.residue_id, l_atom.atom_idx)
            prev = best_by_pair.get(key)
            if prev is None or dist < prev.distance_angstrom:
                best_by_pair[key] = InteractionRecord(
                    kind="hydrophobic",
                    protein_residue=p_atom.residue_id,
                    protein_atom=p_atom.atom_name,
                    ligand_atom_idx=l_atom.atom_idx,
                    ligand_element=l_atom.element,
                    distance_angstrom=dist,
                    angle_deg=None,
                    protein_xyz=_to_xyz_tuple(p_atom.xyz),
                    ligand_xyz=_to_xyz_tuple(l_atom.xyz),
                    subtype="apolar",
                )

    return list(best_by_pair.values())


def _detect_halogen_bonds(
    protein: _ParsedProteinPocket,
    ligand: _ParsedLigand,
    *,
    max_dist: float,
    min_angle: float,
) -> list[InteractionRecord]:
    """Detect directional C-X···A halogen bonds (X in Cl, Br, I; A in O, N, S)."""
    results: list[InteractionRecord] = []

    p_acceptors = [a for a in protein.atoms if a.is_halogen_acceptor]
    l_halogens = [a for a in ligand.atoms if a.halogen_carbon_xyz is not None]

    for l_atom in l_halogens:
        assert l_atom.halogen_carbon_xyz is not None
        for p_atom in p_acceptors:
            dist = float(np.linalg.norm(p_atom.xyz - l_atom.xyz))
            if dist > max_dist or dist < 1e-6:
                continue
            angle_deg = _angle_deg_at_vertex(l_atom.halogen_carbon_xyz, l_atom.xyz, p_atom.xyz)
            if angle_deg < min_angle:
                continue
            results.append(
                InteractionRecord(
                    kind="halogen_bond",
                    protein_residue=p_atom.residue_id,
                    protein_atom=p_atom.atom_name,
                    ligand_atom_idx=l_atom.atom_idx,
                    ligand_element=l_atom.element,
                    distance_angstrom=dist,
                    angle_deg=angle_deg,
                    protein_xyz=_to_xyz_tuple(p_atom.xyz),
                    ligand_xyz=_to_xyz_tuple(l_atom.xyz),
                    subtype="sigma_hole",
                )
            )

    return results


def _build_profile(records: list[InteractionRecord]) -> InteractionProfile:
    """Sort interaction records deterministically and compute summary counts & key residues."""
    sorted_records = sorted(
        records,
        key=lambda r: (
            _KIND_PRIORITY.get(r.kind, 99),
            round(r.distance_angstrom, 6),
            r.protein_residue,
            r.protein_atom,
            r.ligand_atom_idx,
        ),
    )

    counts: dict[str, int] = {k: 0 for k in ALL_INTERACTION_KINDS}
    residue_rank: dict[str, tuple[int, float]] = {}

    for rec in sorted_records:
        counts[rec.kind] = counts.get(rec.kind, 0) + 1
        prio = _KIND_PRIORITY.get(rec.kind, 99)
        prev = residue_rank.get(rec.protein_residue)
        if prev is None or (prio, rec.distance_angstrom) < prev:
            residue_rank[rec.protein_residue] = (prio, rec.distance_angstrom)

    key_residues = sorted(
        residue_rank.keys(),
        key=lambda res: (residue_rank[res][0], round(residue_rank[res][1], 6), res),
    )

    return InteractionProfile(
        interactions=sorted_records,
        counts_by_type=counts,
        key_residues=key_residues,
    )


# ---------------------------------------------------------------------------
# Public Profiling API
# ---------------------------------------------------------------------------


def profile_3d_interactions_from_pdb_text(
    pocket_pdb_text: str,
    ligand_mol_3d: Chem.Mol,
    *,
    hbond_max_dist: float = 3.5,
    hbond_min_angle: float = 120.0,
    salt_bridge_max_dist: float = 4.2,
    pi_stacking_max_dist: float = 5.5,
    cation_pi_max_dist: float = 5.5,
    hydrophobic_max_dist: float = 4.0,
    halogen_max_dist: float = 3.7,
    halogen_min_angle: float = 120.0,
) -> InteractionProfile:
    """Profile exact 3D non-covalent interactions from an in-memory PDB string and 3D ligand."""
    protein = _parse_protein_pocket_from_handle(io.StringIO(pocket_pdb_text))
    ligand = _parse_ligand_3d(ligand_mol_3d)

    all_records: list[InteractionRecord] = []
    all_records.extend(
        _detect_hbonds(protein, ligand, max_dist=hbond_max_dist, min_angle=hbond_min_angle)
    )
    all_records.extend(_detect_salt_bridges(protein, ligand, max_dist=salt_bridge_max_dist))
    all_records.extend(_detect_pi_stacking(protein, ligand, max_dist=pi_stacking_max_dist))
    all_records.extend(_detect_cation_pi(protein, ligand, max_dist=cation_pi_max_dist))
    all_records.extend(_detect_hydrophobic(protein, ligand, max_dist=hydrophobic_max_dist))
    all_records.extend(
        _detect_halogen_bonds(
            protein, ligand, max_dist=halogen_max_dist, min_angle=halogen_min_angle
        )
    )

    return _build_profile(all_records)


def profile_3d_interactions(
    pocket_pdb_path: str | Path,
    ligand_mol_3d: Chem.Mol,
    *,
    hbond_max_dist: float = 3.5,
    hbond_min_angle: float = 120.0,
    salt_bridge_max_dist: float = 4.2,
    pi_stacking_max_dist: float = 5.5,
    cation_pi_max_dist: float = 5.5,
    hydrophobic_max_dist: float = 4.0,
    halogen_max_dist: float = 3.7,
    halogen_min_angle: float = 120.0,
) -> InteractionProfile:
    """Profile exact 3D non-covalent interactions between a PDB pocket file and a 3D ligand.

    Parameters
    ----------
    pocket_pdb_path:
        Path to the receptor or cut binding-pocket PDB file.
    ligand_mol_3d:
        RDKit ``Mol`` with at least one 3D conformer positioned in the pocket coordinate frame.
    hbond_max_dist:
        Maximum heavy-atom donor-acceptor distance (Å) for hydrogen bonds. Default ``3.5``.
    hbond_min_angle:
        Minimum ``D-H···A`` angle (degrees) enforced when donor hydrogens are explicit.
        Default ``120.0``.
    salt_bridge_max_dist:
        Maximum distance (Å) between oppositely charged anchors for salt bridges. Default ``4.2``.
    pi_stacking_max_dist:
        Maximum centroid-to-centroid distance (Å) for aromatic π–π stacking. Default ``5.5``.
    cation_pi_max_dist:
        Maximum distance (Å) between cationic center and aromatic ring centroid. Default ``5.5``.
    hydrophobic_max_dist:
        Maximum apolar C–C distance (Å) for hydrophobic contacts. Default ``4.0``.
    halogen_max_dist:
        Maximum ``X···A`` distance (Å) for ``C-Cl``/``C-Br``/``C-I`` halogen bonds. Default ``3.7``.
    halogen_min_angle:
        Minimum ``C-X···A`` sigma-hole angle (degrees) for halogen bonds. Default ``120.0``.

    Returns
    -------
    InteractionProfile
        Deterministic summary of all detected 3D contacts, counts per channel, and key residues.
    """
    path = Path(pocket_pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"Protein pocket PDB not found: {path}")

    protein = _parse_protein_pocket_from_handle(path)
    ligand = _parse_ligand_3d(ligand_mol_3d)

    all_records: list[InteractionRecord] = []
    all_records.extend(
        _detect_hbonds(protein, ligand, max_dist=hbond_max_dist, min_angle=hbond_min_angle)
    )
    all_records.extend(_detect_salt_bridges(protein, ligand, max_dist=salt_bridge_max_dist))
    all_records.extend(_detect_pi_stacking(protein, ligand, max_dist=pi_stacking_max_dist))
    all_records.extend(_detect_cation_pi(protein, ligand, max_dist=cation_pi_max_dist))
    all_records.extend(_detect_hydrophobic(protein, ligand, max_dist=hydrophobic_max_dist))
    all_records.extend(
        _detect_halogen_bonds(
            protein, ligand, max_dist=halogen_max_dist, min_angle=halogen_min_angle
        )
    )

    return _build_profile(all_records)


# ---------------------------------------------------------------------------
# Symmetry-Corrected Heavy-Atom RMSD (Self-Docking & Conformer Verification)
# ---------------------------------------------------------------------------


def _bond_order_agnostic_copy(mol: Chem.Mol) -> Chem.Mol:
    """Create a neutral, single-bond graph copy preserving heavy-atom elements & connectivity.

    Used as a fallback when comparing poses reconstructed from AutoDock Vina PDBQT files
    whose Kekulé bond orders or formal protonation state differ slightly from the crystal SDF.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumRadicalElectrons(0)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def _kabsch_rmsd(coords_probe: NDArray[np.float64], coords_ref: NDArray[np.float64]) -> float:
    """Compute minimum RMSD between two ordered point clouds after optimal rigid alignment."""
    p_centered = coords_probe - np.mean(coords_probe, axis=0)
    r_centered = coords_ref - np.mean(coords_ref, axis=0)
    cov = p_centered.T @ r_centered
    u_mat, _, vt_mat = np.linalg.svd(cov)
    det_sign = 1.0 if float(np.linalg.det(vt_mat.T @ u_mat.T)) >= 0.0 else -1.0
    diag = np.diag(np.array([1.0, 1.0, det_sign], dtype=np.float64))
    rot = vt_mat.T @ diag @ u_mat.T
    aligned = p_centered @ rot.T
    diff = aligned - r_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_heavy_atom_rmsd(
    mol_ref: Chem.Mol,
    mol_probe: Chem.Mol,
    *,
    align: bool = False,
    max_matches: int = 10000,
) -> float:
    """Compute symmetry-corrected heavy-atom 3D RMSD (Å) between two ligand poses.

    By default (``align=False``), evaluates the **in-place docking pose RMSD** in the
    shared protein-pocket coordinate frame across all graph-automorphism atom mappings
    (``GetSubstructMatches(..., uniquify=False)``), so symmetric functional groups
    (e.g. flipped phenyl, tert-butyl, or carboxylate groups) incur zero spurious penalty
    while true spatial shifts/rotations in the binding pocket are measured exactly.

    When ``align=True``, computes the Kabsch-superimposed conformational RMSD using
    RDKit ``rdMolAlign.GetBestRMS`` (with symmetry-aware Kabsch fallback if PDBQT
    round-tripping altered bond orders).

    Parameters
    ----------
    mol_ref:
        Reference 3D ligand (e.g. crystallographic co-crystal pose).
    mol_probe:
        Probe 3D ligand (e.g. AutoDock Vina self-docked pose).
    align:
        If ``False`` (default), compute in-place docking pose RMSD without superposition.
        If ``True``, superimpose ``mol_probe`` onto ``mol_ref`` before computing RMSD.
    max_matches:
        Maximum symmetry automorphisms evaluated via RDKit substructure matching.

    Returns
    -------
    float
        Minimum heavy-atom RMSD in Ångströms across all symmetry-equivalent mappings.
    """
    if mol_ref is None or mol_probe is None:
        raise ValueError("Both mol_ref and mol_probe must be non-None RDKit Mol objects.")
    if mol_ref.GetNumConformers() == 0 or mol_probe.GetNumConformers() == 0:
        raise ValueError("Both mol_ref and mol_probe must contain a 3D conformer.")

    ref_noh = Chem.RemoveHs(Chem.Mol(mol_ref), sanitize=False)
    probe_noh = Chem.RemoveHs(Chem.Mol(mol_probe), sanitize=False)
    ref_noh.UpdatePropertyCache(strict=False)
    probe_noh.UpdatePropertyCache(strict=False)

    n_ref = ref_noh.GetNumAtoms()
    n_probe = probe_noh.GetNumAtoms()
    if n_ref == 0 or n_ref != n_probe:
        raise ValueError(
            f"Heavy-atom count mismatch between mol_ref ({n_ref}) and mol_probe ({n_probe})."
        )

    if align:
        try:
            # Copy to avoid mutating caller coordinates during rdMolAlign superposition
            return float(
                rdMolAlign.GetBestRMS(
                    Chem.Mol(probe_noh),
                    Chem.Mol(ref_noh),
                    maxMatches=max_matches,
                )
            )
        except Exception:
            # Fall through to bond-order-agnostic Kabsch alignment below
            pass

    # 1) Exact molecular-graph automorphisms (captures phenyl/carboxylate/t-Bu symmetries)
    matches = probe_noh.GetSubstructMatches(ref_noh, uniquify=False, maxMatches=max_matches)

    # 2) Fallback: bond-order-agnostic connectivity match (handles PDBQT Kekulé/charge shifts)
    if not matches:
        ref_skel = _bond_order_agnostic_copy(ref_noh)
        probe_skel = _bond_order_agnostic_copy(probe_noh)
        matches = probe_skel.GetSubstructMatches(ref_skel, uniquify=False, maxMatches=max_matches)

    # 3) Final fallback: direct 1-to-1 atom index order if element sequence matches identically
    if not matches:
        ref_els = [a.GetAtomicNum() for a in ref_noh.GetAtoms()]
        probe_els = [a.GetAtomicNum() for a in probe_noh.GetAtoms()]
        if ref_els != probe_els:
            raise ValueError(
                "Could not establish a symmetry-equivalent heavy-atom mapping between "
                "mol_ref and mol_probe."
            )
        matches = (tuple(range(n_ref)),)

    ref_coords = np.asarray(ref_noh.GetConformer().GetPositions(), dtype=np.float64)
    probe_coords = np.asarray(probe_noh.GetConformer().GetPositions(), dtype=np.float64)

    best_rmsd = float("inf")
    for match in matches:
        mapped_probe = probe_coords[list(match)]
        if align:
            rmsd = _kabsch_rmsd(mapped_probe, ref_coords)
        else:
            diff = mapped_probe - ref_coords
            rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            if best_rmsd <= 1e-10:
                return 0.0

    return float(best_rmsd)
