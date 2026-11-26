"""52-D compatible E(3)-invariant 3D small-molecule ligand featurization for GenEuclidean.

Produces heavy-atom 3D coordinates and 52-dimensional per-atom descriptors aligned channel-for-
channel with GenEuclidean's canonical protein pocket layout
(:func:`geneuclidean.data.featurizer._compute_52d_node_features`):

- ``[0:10]``: Element one-hot (``C, N, O, S, P, F, Cl, Br, Metal, Other``)
- ``[10:31]``: Residue one-hot (ligands activate slot ``20`` = ``OTHER``, column index ``30``)
- ``[31:36]``: Ligand hybridization & ring role (``sp, sp2, sp3, aromatic_ring, aliphatic_ring``)
- ``[36:52]``: 16 biophysical & E(3)-invariant radial scalars:
  ``[is_hydrophobic(36), is_aromatic(37), is_hba(38), is_hbd(39), is_pos(40), is_neg(41),
  is_backbone=0(42), is_polar(43), mass_norm(44), vdw_radius_norm(45), partial_charge(46),
  dist_to_center_norm(47), dist_shell_pos(48), radial_rbf_0(49), radial_rbf_1(50),
  radial_rbf_2(51)]``.

This shared 52-D feature space allows ANY GenEuclidean 3D pocket encoder (``PaiNN-SE3``,
``CartFormer-SE3``, ``AtomGS-SE3``, ``SubPocketVN-JEPA``) to encode both protein pockets AND
3D ligand conformers natively with identical input channels — including Bondi van der Waals
radii at ``dim 45`` and RDKit Gasteiger partial charges at ``dim 46``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Lipinski
from torch import Tensor

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

LIGAND_3D_NODE_DIM: int = 52

# Exact match with geneuclidean.data.featurizer channel conventions
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


def _safe_gasteiger_charges(mol: Any) -> dict[int, float]:
    """Compute finite RDKit Gasteiger partial charges clamped to ``[-1.0, 1.0]``."""
    charges: dict[int, float] = {}
    try:
        AllChem.ComputeGasteigerCharges(mol)  # type: ignore[attr-defined]
        for atom in mol.GetAtoms():
            idx = int(atom.GetIdx())
            if atom.HasProp("_GasteigerCharge"):
                val = float(atom.GetProp("_GasteigerCharge"))
                if math.isfinite(val):
                    charges[idx] = max(-1.0, min(1.0, val))
                    continue
            charges[idx] = 0.0
    except Exception:
        for atom in mol.GetAtoms():
            charges[int(atom.GetIdx())] = 0.0
    return charges


def featurize_ligand_3d(
    mol: Any,
    *,
    pocket_center: Sequence[float] | np.ndarray | None = None,
    pocket_cutoff: float = 10.0,
    conformer_id: int = -1,
    embed_if_missing: bool = False,
    seed: int = 42,
    center_coords: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize a 3D RDKit ligand conformer into GenEuclidean's canonical 52-D layout.

    Never mutates the caller's ``mol``. Heavy atoms (``atomic_num > 1``) are extracted in their
    molecule atom order; if ``pocket_center`` is supplied, radial features ``[47:52]`` (and
    centered coordinates when ``center_coords=True``) share the exact pocket coordinate frame.

    Args:
        mol: RDKit ``Chem.Mol`` object (must possess at least one 3D conformer unless
            ``embed_if_missing=True``).
        pocket_center: Optional 3D pocket reference center ``(x, y, z)`` in Angstroms.
            Defaults to the ligand heavy-atom centroid.
        pocket_cutoff: Normalization radius in Angstroms for radial distance features.
        conformer_id: RDKit conformer index (default ``-1`` for active conformer).
        embed_if_missing: If ``True`` and ``mol`` has no conformer, generates one via ETKDGv3.
        seed: Deterministic RNG seed used only when ``embed_if_missing=True``.
        center_coords: If ``True`` (default), subtract ``pocket_center`` (or ligand centroid)
            from returned coordinates; otherwise return raw world coordinates.

    Returns:
        Tuple ``(coords [M, 3], node_feats [M, 52], atom_mask [M])`` with dtypes
        ``(float32, float32, bool)``.
    """
    if mol is None:
        raise ValueError("mol cannot be None.")
    if pocket_cutoff <= 0.0:
        raise ValueError(f"pocket_cutoff must be positive, got {pocket_cutoff}")

    work_mol = Chem.Mol(mol)
    if work_mol.GetNumConformers() == 0:
        if not embed_if_missing:
            raise ValueError(
                "RDKit Mol has no 3D conformer. Embed a 3D conformer first or pass "
                "embed_if_missing=True."
            )
        mol_h = Chem.AddHs(work_mol)
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = int(seed)
        status = int(AllChem.EmbedMolecule(mol_h, params))  # type: ignore[attr-defined]
        if status != 0:
            raise ValueError("Failed to generate a 3D conformer via ETKDGv3.")
        work_mol = Chem.RemoveHs(mol_h)
        conformer_id = 0
    else:
        # Strip explicit hydrogens while preserving conformer IDs so explicit-H SDFs
        # (e.g., PDBbind / Boltz-2) and H-suppressed mols produce identical Lipinski
        # HBA/HBD SMARTS matches and Gasteiger partial charges.
        try:
            work_mol = Chem.RemoveHs(work_mol, sanitize=True)
        except Exception:
            work_mol = Chem.RemoveHs(work_mol, sanitize=False)

    conf = work_mol.GetConformer(int(conformer_id))
    heavy_atoms = [atom for atom in work_mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy_atoms:
        raise ValueError("Ligand contains no heavy atoms (atomic number > 1).")

    raw_coords = np.empty((len(heavy_atoms), 3), dtype=np.float32)
    for row, atom in enumerate(heavy_atoms):
        pos = conf.GetAtomPosition(atom.GetIdx())
        raw_coords[row, 0] = float(pos.x)
        raw_coords[row, 1] = float(pos.y)
        raw_coords[row, 2] = float(pos.z)

    if not np.all(np.isfinite(raw_coords)):
        raise ValueError("Ligand 3D conformer contains non-finite coordinates.")

    if pocket_center is not None:
        ref_center = np.asarray(pocket_center, dtype=np.float32).reshape(3)
    else:
        ref_center = raw_coords.mean(axis=0).astype(np.float32)

    rel_coords = (raw_coords - ref_center[None, :]).astype(np.float32)
    radii = np.linalg.norm(rel_coords, axis=-1).astype(np.float32)
    r_min = float(radii.min())
    r_max = float(radii.max())
    r_span = max(r_max - r_min, 1e-6)
    inv_cutoff = 1.0 / max(float(pocket_cutoff), 1e-6)

    # Pharmacophore SMARTS & Gasteiger partial charges on the copied molecule
    hba_indices: set[int] = {
        int(match[0]) for match in work_mol.GetSubstructMatches(Lipinski.HAcceptorSmarts)
    }
    hbd_indices: set[int] = {
        int(match[0]) for match in work_mol.GetSubstructMatches(Lipinski.HDonorSmarts)
    }
    gasteiger = _safe_gasteiger_charges(work_mol)

    feats = np.zeros((len(heavy_atoms), LIGAND_3D_NODE_DIM), dtype=np.float32)
    for row, atom in enumerate(heavy_atoms):
        a_idx = int(atom.GetIdx())
        elem = str(atom.GetSymbol()).strip().upper()
        formal_charge = int(atom.GetFormalCharge())

        # [0:10] Element one-hot
        if elem in _ELEMENT_TO_IDX:
            feats[row, _ELEMENT_TO_IDX[elem]] = 1.0
        elif elem in _METAL_ELEMENTS:
            feats[row, 8] = 1.0
        else:
            feats[row, 9] = 1.0

        # [10:31] Residue one-hot -> OTHER (index 20 -> col 30)
        feats[row, 30] = 1.0

        # [31:36] Hybridization & ring role (sp, sp2, sp3, aromatic_ring, aliphatic_ring)
        hyb = atom.GetHybridization()
        is_aromatic = bool(atom.GetIsAromatic())
        in_ring = bool(atom.IsInRing())
        if hyb == Chem.HybridizationType.SP:
            feats[row, 31] = 1.0
        elif hyb == Chem.HybridizationType.SP2:
            feats[row, 32] = 1.0
        elif hyb == Chem.HybridizationType.SP3:
            feats[row, 33] = 1.0
        if is_aromatic and in_ring:
            feats[row, 34] = 1.0
        elif in_ring and not is_aromatic:
            feats[row, 35] = 1.0

        # [36:52] 16 biophysical & E(3)-invariant scalars matching featurizer.py
        is_hba = 1.0 if a_idx in hba_indices else 0.0
        is_hbd = (
            1.0
            if (a_idx in hbd_indices or (elem in ("N", "O", "S") and atom.GetTotalNumHs() > 0))
            else 0.0
        )
        is_pos = 1.0 if (formal_charge > 0 or elem in _METAL_ELEMENTS) else 0.0
        is_neg = 1.0 if formal_charge < 0 else 0.0
        is_polar = (
            1.0
            if (
                elem in ("N", "O", "S", "P")
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
                (elem == "C" and is_pos == 0.0 and is_neg == 0.0)
                or (elem == "S" and is_hba == 0.0 and is_hbd == 0.0)
                or elem in ("F", "CL", "BR", "I")
            )
            else 0.0
        )

        mass = _ATOMIC_MASS.get(elem, float(atom.GetMass()) or 12.011)
        mass_norm = min(mass / 32.0, 4.0)
        vdw_norm = _VDW_RADIUS.get(elem, 1.70) / 2.0
        charge_val = gasteiger.get(a_idx, 0.0)
        if charge_val == 0.0 and formal_charge != 0:
            charge_val = max(-1.0, min(1.0, float(formal_charge)))

        r_val = float(radii[row])
        dist_norm = r_val * inv_cutoff
        shell_pos = (r_val - r_min) / r_span
        rbf0 = float(math.exp(-(((r_val - 3.0) / 2.0) ** 2)))
        rbf1 = float(math.exp(-(((r_val - 6.0) / 2.0) ** 2)))
        rbf2 = float(math.exp(-(((r_val - 9.0) / 2.0) ** 2)))

        feats[row, 36] = is_hydrophobic
        feats[row, 37] = 1.0 if is_aromatic else 0.0
        feats[row, 38] = is_hba
        feats[row, 39] = is_hbd
        feats[row, 40] = is_pos
        feats[row, 41] = is_neg
        feats[row, 42] = 0.0  # is_backbone = 0 for small-molecule ligands
        feats[row, 43] = is_polar
        feats[row, 44] = mass_norm
        feats[row, 45] = vdw_norm
        feats[row, 46] = charge_val
        feats[row, 47] = dist_norm
        feats[row, 48] = shell_pos
        feats[row, 49] = rbf0
        feats[row, 50] = rbf1
        feats[row, 51] = rbf2

    out_coords = rel_coords if center_coords else raw_coords
    atom_mask = np.ones(len(heavy_atoms), dtype=bool)
    return out_coords, feats, atom_mask


def collate_ligand_3d_batch(
    mols: Sequence[Any],
    *,
    pocket_centers: Sequence[Sequence[float] | np.ndarray | None] | None = None,
    pocket_cutoff: float = 10.0,
    embed_if_missing: bool = False,
    seed: int = 42,
    center_coords: bool = True,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Featurize and pad a batch of 3D RDKit ligands into ``[B, M_max, ...]`` PyTorch tensors."""
    if not mols:
        raise ValueError("mols must contain at least one RDKit Mol.")
    if pocket_centers is not None and len(pocket_centers) != len(mols):
        raise ValueError(
            f"pocket_centers length ({len(pocket_centers)}) != len(mols) ({len(mols)})"
        )

    items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for idx, mol in enumerate(mols):
        center = pocket_centers[idx] if pocket_centers is not None else None
        items.append(
            featurize_ligand_3d(
                mol,
                pocket_center=center,
                pocket_cutoff=pocket_cutoff,
                embed_if_missing=embed_if_missing,
                seed=seed + idx,
                center_coords=center_coords,
            )
        )

    batch_size = len(items)
    max_atoms = max(c.shape[0] for c, _, _ in items)
    coords_np = np.zeros((batch_size, max_atoms, 3), dtype=np.float32)
    feats_np = np.zeros((batch_size, max_atoms, LIGAND_3D_NODE_DIM), dtype=np.float32)
    mask_np = np.zeros((batch_size, max_atoms), dtype=bool)

    for b, (c, f, m) in enumerate(items):
        n = c.shape[0]
        coords_np[b, :n] = c
        feats_np[b, :n] = f
        mask_np[b, :n] = m

    dev = torch.device(device) if device is not None else torch.device("cpu")
    return (
        torch.from_numpy(coords_np).to(dev),
        torch.from_numpy(feats_np).to(dev),
        torch.from_numpy(mask_np).to(dev),
    )


__all__ = [
    "LIGAND_3D_NODE_DIM",
    "collate_ligand_3d_batch",
    "featurize_ligand_3d",
]
