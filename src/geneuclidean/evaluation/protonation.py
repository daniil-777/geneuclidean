"""Deterministic physiological pH 7.4 ligand protonation & ionization engine.

Promoted from GenEuclidean's Gate 1 & Iteration 3 structural validation pipelines into a
reusable, zero-external-binary cheminformatics module grounded in physical organic chemistry.

Rules applied at physiological ``pH = 7.4``:

1. **Amidines & Guanidines** (conjugate-acid ``pKa ~ 7.8–13.5``):
   Protonate the ``sp2`` imine nitrogen (``=N-`` → ``=[NH+]-``), including acylguanidines
   (e.g., guanfacine, clinical ``pKa ~ 8.1``) and 2-aminoimidazolines (e.g., brimonidine,
   clonidine). Prevents unphysical diprotonation of conjugated biguanides (e.g., metformin,
   ``pKa1 = 12.4, pKa2 = 2.8`` → monocation ``+1``).
2. **Basic Aliphatic Amines** (``pKa ~ 7.5–10.5``):
   Protonate primary, secondary, and tertiary aliphatic amines (``[NX3]``) while keeping
   amides, ureas, carbamates, sulfonamides, nitro groups, anilines/arylamines, imines,
   nitriles, and heteroaromatic nitrogens strictly neutral. Prevents unphysical
   diprotonation of proximal diamines and 1,4-piperazines (``<= 3`` topological bonds apart,
   where electrostatic repulsion depresses ``pKa2 < 6``). In asymmetric piperazines
   (e.g., flupentixol), preferentially protonates the inductively more basic nitrogen with
   fewer nearby electron-withdrawing atoms (O, F, Cl, Br, I, S within 2–3 bonds).
3. **Permanent Quaternary Cations**:
   Preserves existing formal ``+1`` quaternary ammonium centers (e.g., trospium) without
   altering valence or double-protonating nearby amines.
4. **Strong Organic Acids** (``pKa < 5.0``):
   Deprotonates carboxylic acids, sulfonic/phosphonic acids, and 5-substituted tetrazoles.
5. **Idempotency**:
   Running :func:`protonate_mol_ph74` twice produces the exact same ionization state and
   canonical isomeric SMILES.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

PHYSIOLOGICAL_PH: float = 7.4

# Electronegative atoms inducing inductive basicity reduction at 2–3 bonds
_INDUCTIVE_EWG_ATOMIC_NUMS: frozenset[int] = frozenset({8, 9, 16, 17, 35, 53})

# SMARTS patterns for deterministic pH 7.4 microspecies assignment
# Excludes amides, sulfonamides, nitro, anilines, imines, nitriles, and resonance-conjugated
# amidine/guanidine amino nitrogens (which protonate strictly at their sp2 imine nitrogen).
_SMARTS_STR_BASIC_ALIPHATIC_AMINE = (
    "[NX3;H0,H1,H2;+0;!$(NC=O);!$(NS=O);!$(N=O);!$(N[a]);!$(N=*);!$(N#*);!$(N[CX3]=[NX2,NX3+])]"
)
_SMARTS_STR_BASIC_AMINE_ANY_STATE = (
    "[NX3,NX4+;!$(NC=O);!$(NS=O);!$(N=O);!$(N[a]);!$(N=*);!$(N#*);!$(N[CX3]=[NX2,NX3+])]"
)
_SMARTS_STR_AMIDINE_GUANIDINE_IMINE = "[NX2;+0]=[CX3]([NX3])[#6,N]"
_SMARTS_STR_AMIDINE_GUANIDINE_ANY = "[NX2,NX3+;!$([N+](=O)[O-])]=[CX3]([NX3])[#6,N]"
_SMARTS_STR_CARBOXYLIC_OH = "[OX2H1;+0][CX3]=O"
_SMARTS_STR_SULFO_PHOSPHO_OH = "[OX2H1;+0][SX4,PX4](=O)=O"
_SMARTS_STR_TETRAZOLE_NH = "[nH1;+0]1nnnn1"
_SMARTS_STR_NITRO_N = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"


@lru_cache(maxsize=1)
def _compiled_smarts() -> dict[str, Chem.Mol]:
    """Compile and cache all SMARTS queries lazily on first use."""
    raw = {
        "basic_amine": _SMARTS_STR_BASIC_ALIPHATIC_AMINE,
        "basic_amine_any": _SMARTS_STR_BASIC_AMINE_ANY_STATE,
        "amidine_imine": _SMARTS_STR_AMIDINE_GUANIDINE_IMINE,
        "amidine_any": _SMARTS_STR_AMIDINE_GUANIDINE_ANY,
        "carboxylic_oh": _SMARTS_STR_CARBOXYLIC_OH,
        "sulfo_phospho_oh": _SMARTS_STR_SULFO_PHOSPHO_OH,
        "tetrazole_nh": _SMARTS_STR_TETRAZOLE_NH,
        "nitro_n": _SMARTS_STR_NITRO_N,
    }
    compiled: dict[str, Chem.Mol] = {}
    for key, smarts in raw.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise RuntimeError(f"Failed to compile internal protonation SMARTS for {key!r}")
        compiled[key] = pat
    return compiled


@dataclass(frozen=True)
class ProtonationResult:
    """Structured summary of physiological pH 7.4 ionization analysis.

    Attributes:
        input_smiles: Canonical isomeric SMILES of the input structure.
        protonated_smiles: Canonical isomeric SMILES at physiological pH 7.4.
        net_charge: Formal net molecular charge at pH 7.4.
        n_cationic_sites: Count of positively charged non-nitro atoms at pH 7.4.
        n_anionic_sites: Count of negatively charged non-nitro atoms at pH 7.4.
        has_basic_amine: Whether the molecule carries a basic aliphatic amine,
            amidine/guanidine, or cationic ammonium center capable of GPCR
            ``Asp^{3.32}`` salt-bridge engagement.
        is_zwitterion: ``True`` if both cationic and anionic sites exist simultaneously.
        protonated_atom_indices: Atom indices assigned formal ``+1`` charge.
        deprotonated_atom_indices: Atom indices assigned formal ``-1`` charge.
    """

    input_smiles: str
    protonated_smiles: str
    net_charge: int
    n_cationic_sites: int
    n_anionic_sites: int
    has_basic_amine: bool
    is_zwitterion: bool
    protonated_atom_indices: tuple[int, ...]
    deprotonated_atom_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize ionization result to a JSON-compatible dictionary."""
        return {
            "input_smiles": self.input_smiles,
            "protonated_smiles": self.protonated_smiles,
            "net_charge": self.net_charge,
            "n_cationic_sites": self.n_cationic_sites,
            "n_anionic_sites": self.n_anionic_sites,
            "has_basic_amine": self.has_basic_amine,
            "is_zwitterion": self.is_zwitterion,
            "protonated_atom_indices": list(self.protonated_atom_indices),
            "deprotonated_atom_indices": list(self.deprotonated_atom_indices),
        }


def _coerce_mol(mol_or_smiles: str | Chem.Mol) -> Chem.Mol:
    """Convert a SMILES string or RDKit ``Mol`` into a valid sanitized ``Chem.Mol``."""
    if isinstance(mol_or_smiles, Chem.Mol):
        return mol_or_smiles
    if not isinstance(mol_or_smiles, str) or not mol_or_smiles.strip():
        raise ValueError(f"Expected non-empty SMILES string or RDKit Mol, got {mol_or_smiles!r}")
    mol = Chem.MolFromSmiles(mol_or_smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {mol_or_smiles!r}")
    return mol


def extract_canonical_isomeric_smiles(mol_3d: Chem.Mol) -> str:
    """Extract canonical isomeric SMILES preserving both R/S and E/Z stereochemistry from 3D.

    Args:
        mol_3d: Sanitized RDKit ``Mol`` (optionally carrying a 3D conformer).

    Returns:
        Canonical isomeric SMILES with explicit hydrogens removed.
    """
    mol_copy = Chem.Mol(mol_3d)
    for bond in mol_copy.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    if mol_copy.GetNumConformers() > 0:
        Chem.DetectBondStereochemistry(mol_copy)
        Chem.AssignStereochemistryFrom3D(mol_copy)
    Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
    return str(Chem.MolToSmiles(Chem.RemoveHs(mol_copy), canonical=True, isomericSmiles=True))


def protonate_mol_ph74(mol: Chem.Mol, *, preserve_coords: bool = False) -> Chem.Mol:
    """Deterministically assign physiological pH 7.4 formal charges to an RDKit ``Mol``.

    Args:
        mol: Input RDKit ``Mol`` (2D or 3D).
        preserve_coords: If ``True``, retains existing 3D heavy-atom conformer coordinates
            (useful for profiling native co-crystal ligands). If ``False`` (default),
            strips conformers so downstream 2D-to-3D conformer generation is strictly unbiased.

    Returns:
        New sanitized RDKit ``Mol`` (heavy atoms only; call ``Chem.AddHs`` before 3D
        embedding) with physiological pH 7.4 formal charges applied.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCN(CC)CC")
        >>> Chem.GetFormalCharge(protonate_mol_ph74(mol))
        1
    """
    if mol is None:
        raise ValueError("Input mol must not be None")

    pats = _compiled_smarts()
    base = Chem.RemoveHs(Chem.Mol(mol))
    if not preserve_coords:
        base.RemoveAllConformers()
    rw = Chem.RWMol(base)
    n_atoms = rw.GetNumAtoms()
    dist_mat = Chem.GetDistanceMatrix(rw)

    nitro_indices: set[int] = set()
    for match in rw.GetSubstructMatches(pats["nitro_n"]):
        nitro_indices.add(int(match[0]))

    # Seed with any pre-existing non-nitro cationic nitrogen centers (quaternary [N+] or
    # already-protonated [NH+]) so this function is strictly idempotent.
    protonated: set[int] = set()
    protonated_imines: set[int] = set()
    for atom in rw.GetAtoms():
        idx = atom.GetIdx()
        if idx in nitro_indices:
            continue
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0:
            protonated.add(idx)
            if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
                protonated_imines.add(idx)

    # 1. Amidines, guanidines & acylguanidines: protonate sp2 imine N (=N- -> =[NH+]-),
    #    blocking conjugated biguanide diprotonation (two imines <= 4 bonds apart, e.g. metformin).
    imine_candidates = sorted({int(m[0]) for m in rw.GetSubstructMatches(pats["amidine_imine"])})
    for n_idx in imine_candidates:
        if n_idx in protonated:
            continue
        if any(int(round(float(dist_mat[n_idx, p]))) <= 4 for p in protonated_imines):
            continue
        if any(int(round(float(dist_mat[n_idx, p]))) <= 3 for p in protonated):
            continue
        rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
        protonated.add(n_idx)
        protonated_imines.add(n_idx)

    # 2. Aliphatic primary / secondary / tertiary amines ranked by inductive basicity
    cand_amines = [int(m[0]) for m in rw.GetSubstructMatches(pats["basic_amine"])]

    def _basicity_sort_key(atom_idx: int) -> tuple[int, int, int]:
        """Rank more basic amines first (fewer nearby EWGs at 2-3 bonds, higher alkylation)."""
        ewg_2 = 0
        ewg_3 = 0
        for j in range(n_atoms):
            if j == atom_idx:
                continue
            if rw.GetAtomWithIdx(j).GetAtomicNum() in _INDUCTIVE_EWG_ATOMIC_NUMS:
                d = int(round(float(dist_mat[atom_idx, j])))
                if d == 2:
                    ewg_2 += 1
                elif d == 3:
                    ewg_3 += 1
        atom = rw.GetAtomWithIdx(atom_idx)
        return (ewg_2 * 3 + ewg_3, atom.GetTotalNumHs(), atom_idx)

    for n_idx in sorted(set(cand_amines), key=_basicity_sort_key):
        if n_idx in protonated:
            continue
        # Prevent unphysical diprotonation of 1,2-diamines / 1,4-piperazines (<= 3 bonds apart)
        if any(int(round(float(dist_mat[n_idx, p]))) <= 3 for p in protonated):
            continue
        rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
        protonated.add(n_idx)

    # 3. Deprotonate strong organic acids (pKa < 5.0)
    for key in ("carboxylic_oh", "sulfo_phospho_oh", "tetrazole_nh"):
        for match in rw.GetSubstructMatches(pats[key]):
            a_idx = int(match[0])
            atom = rw.GetAtomWithIdx(a_idx)
            if atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(-1)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def protonate_smiles_ph74(smiles: str, *, isomeric: bool = True) -> str:
    """Deterministically protonate a SMILES string to its dominant pH 7.4 microspecies.

    Args:
        smiles: Input SMILES string.
        isomeric: Whether to preserve tetrahedral and E/Z stereochemistry in output SMILES.

    Returns:
        Canonical (optionally isomeric) SMILES at physiological pH 7.4.

    Raises:
        ValueError: If ``smiles`` cannot be parsed by RDKit.

    Examples:
        >>> protonate_smiles_ph74("CC(=O)O")
        'CC(=O)[O-]'
        >>> protonate_smiles_ph74("CCN(CC)CC")
        'CC[NH+](CC)CC'
    """
    mol = _coerce_mol(smiles)
    prot = protonate_mol_ph74(mol, preserve_coords=False)
    return str(Chem.MolToSmiles(prot, canonical=True, isomericSmiles=isomeric))


def has_basic_amine(mol_or_smiles: str | Chem.Mol) -> bool:
    """Return ``True`` if the molecule has a basic aliphatic amine, amidine, or guanidine.

    Excludes amides, ureas, sulfonamides, nitro groups, anilines, nitriles, and neutral
    aromatic heterocycles. Recognizes both neutral and pre-protonated basic amines and
    permanent quaternary ammonium ions.

    Args:
        mol_or_smiles: SMILES string or RDKit ``Mol``.

    Returns:
        ``True`` if a basic / cationic amine center is present.
    """
    mol = _coerce_mol(mol_or_smiles)
    pats = _compiled_smarts()
    if mol.HasSubstructMatch(pats["basic_amine_any"]):
        return True
    return bool(mol.HasSubstructMatch(pats["amidine_any"]))


def estimate_physiological_charge(mol_or_smiles: str | Chem.Mol) -> int:
    """Estimate the integer net formal charge of a ligand at physiological pH 7.4.

    Args:
        mol_or_smiles: SMILES string or RDKit ``Mol``.

    Returns:
        Net integer formal charge at pH 7.4 (e.g., ``+1`` for typical aminergic CNS drugs,
        ``0`` for neutral PDE10A inhibitors or zwitterions, ``-1`` for carboxylic acids).
    """
    mol = _coerce_mol(mol_or_smiles)
    prot = protonate_mol_ph74(mol, preserve_coords=False)
    return int(Chem.GetFormalCharge(prot))


def analyze_protonation_ph74(mol_or_smiles: str | Chem.Mol) -> ProtonationResult:
    """Return full physiological pH 7.4 protonation metadata for a SMILES or RDKit ``Mol``.

    Args:
        mol_or_smiles: SMILES string or RDKit ``Mol``.

    Returns:
        :class:`ProtonationResult` with protonated SMILES, formal net charge, cationic and
        anionic site counts, basic-amine flag, zwitterion flag, and ionized atom indices.
    """
    mol = _coerce_mol(mol_or_smiles)
    input_smi = str(Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True))
    prot = protonate_mol_ph74(mol, preserve_coords=False)
    prot_smi = str(Chem.MolToSmiles(prot, canonical=True, isomericSmiles=True))

    pats = _compiled_smarts()
    nitro_atoms: set[int] = set()
    for match in prot.GetSubstructMatches(pats["nitro_n"]):
        nitro_atoms.update(int(i) for i in match)

    pos_idxs = tuple(
        a.GetIdx()
        for a in prot.GetAtoms()
        if a.GetFormalCharge() > 0 and a.GetIdx() not in nitro_atoms
    )
    neg_idxs = tuple(
        a.GetIdx()
        for a in prot.GetAtoms()
        if a.GetFormalCharge() < 0 and a.GetIdx() not in nitro_atoms
    )
    net_charge = int(Chem.GetFormalCharge(prot))
    return ProtonationResult(
        input_smiles=input_smi,
        protonated_smiles=prot_smi,
        net_charge=net_charge,
        n_cationic_sites=len(pos_idxs),
        n_anionic_sites=len(neg_idxs),
        has_basic_amine=has_basic_amine(prot),
        is_zwitterion=(len(pos_idxs) > 0 and len(neg_idxs) > 0),
        protonated_atom_indices=pos_idxs,
        deprotonated_atom_indices=neg_idxs,
    )


__all__ = [
    "PHYSIOLOGICAL_PH",
    "ProtonationResult",
    "analyze_protonation_ph74",
    "estimate_physiological_charge",
    "extract_canonical_isomeric_smiles",
    "has_basic_amine",
    "protonate_mol_ph74",
    "protonate_smiles_ph74",
]
