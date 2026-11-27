"""Differentiable 3D Protein–Pocket–Ligand Biophysical & Electrical Force-Field Engine.

Implements a pure-PyTorch ``torch.float32`` biophysical parameterization and multi-term
molecular recognition Hamiltonian compatible with Apple Silicon MPS, CUDA, and CPU
(strictly **zero** ``torch.float64`` parameters or buffers):

1. **Atomic Biophysical Parameterization** (:class:`BiophysicalAtomParams`,
   :func:`parameterize_ligand_biophysics`, :func:`parameterize_pocket_biophysics`):
   - Electronegativity Equalization / Gasteiger–Marsili partial charges :math:`q_i` with
     strict net formal-charge conservation weighted by atomic dipole polarizability.
   - AMBER ff14SB-inspired protein backbone and side-chain partial charges (properly
     charging anionic ``Asp-``/``Glu-``, cationic ``Lys+``/``Arg+``/``Hip+``, polar
     ``Ser``/``Thr``/``Tyr``/``Asn``/``Gln``/``Cys``, aromatic ``Trp``/``Phe``/``Tyr``/``His``,
     and catalytic divalent metals ``Zn2+``/``Mg2+``/``Ca2+``/``Fe2+``).
   - GAFF2/Bondi van der Waals parameters :math:`(\\sigma_i, \\epsilon_i)`, mbondi2
     intrinsic Born radii :math:`\\rho_i`, OBC2 screening factors :math:`S_i`, isotropic
     atomic polarizabilities :math:`\\alpha_i`, H-bond donor/acceptor topology, ordered
     aromatic ring perimeters for Newell cross-product plane normals, and covalent
     halogen :math:`\\sigma`-hole :math:`\\text{C–X}` (:math:`\\text{Cl, Br, I}`) axes.

2. **Differentiable Physical & Electrical Hamiltonian** (:class:`BiophysicalHamiltonian`,
   :class:`ComplexEnergyBreakdown`):
   - **Coulomb Electrostatics** (:math:`332.06371 \\cdot q_i q_j / (\\varepsilon(r_{ij}) r_{ij})`
     in :math:`\\text{kcal/mol}`) with Mehler–Solmajer sigmoidal distance-dependent
     dielectric screening :math:`\\varepsilon(r)` and **Generalized Born (OBC2-style)
     Polar Solvation / Desolvation** (:math:`\\Delta G_{\\text{GB}}` via differentiable
     effective Born radii :math:`R_i` capturing buried vs. solvent-exposed microenvironments).
   - **Soft-Core / Buffered Lennard-Jones 12-6 Dispersion & Pauli Steric Repulsion**
     parameterized by :math:`\\lambda \\in [0, 1]` so clashed generative poses produce
     smooth, finite repulsive forces without ``inf`` or ``NaN``.
   - **Directional Hydrogen Bonds & Salt Bridges** (12-10 radial potential with smooth
     :math:`C^3` cosine-power donor–acceptor angular gating).
   - **Anisotropic :math:`\\sigma`-Hole Halogen Bonding**
     (:math:`\\text{C–Cl/Br/I}\\cdots\\text{O/N/S}` crown attraction peaking at
     :math:`\\theta_{\\text{C–X}\\cdots\\text{B}} = 180^\\circ` plus equatorial
     :math:`p`-belt lateral clash penalty around :math:`90^\\circ`).
   - **Cation–:math:`\\pi` & :math:`\\pi`–:math:`\\pi` Aromatic Quadrupole Interactions**
     (above-ring cation–:math:`\\pi` attraction + Hunter–Sanders parallel-displaced and
     T-shaped aromatic multipole interactions).
   - **Non-polar Hydrophobic SASA Burial** (:math:`\\Delta G_{\\text{np}}`) and
     **Rotatable-Bond Conformational Entropy Penalty** (:math:`T\\Delta S_{\\text{rot}}`).
   - **Exact Analytical Autograd 3D Forces & Electric Fields**: exact total forces
     :math:`\\vec{F}_i = -\\nabla_{\\mathbf{r}_i} U_{\\text{total}}`
     (:math:`\\text{kcal}\\cdot\\text{mol}^{-1}\\cdot\\text{\\AA}^{-1}`), separated electrical
     forces :math:`\\vec{F}_i^{\\text{elec}}`, 3D electric field vectors :math:`\\vec{E}_i`
     (:math:`\\text{V/\\AA}`), steric/vdW forces :math:`\\vec{F}_i^{\\text{vdW}}`, per-residue
     energy decomposition :math:`\\Delta E_r`, and estimated :math:`\\Delta G_{\\text{bind}}`.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch import Tensor

from geneuclidean.evaluation.geometry3d import embed_and_minimize_3d
from geneuclidean.exceptions import (
    InvalidMoleculeError,
    PocketFeaturizationError,
    StructureNotFoundError,
)

# ---------------------------------------------------------------------------
# Fundamental Physical Constants (CODATA / AMBER / OBC2, float32-safe)
# ---------------------------------------------------------------------------

COULOMB_KCAL_ANGSTROM: float = 332.06371  # kcal * A / (mol * e^2)
KCAL_MOL_E_TO_VOLT: float = 0.043364115  # 1 kcal/(mol*e) = 1 / 23.060548 V
KB_KCAL_MOL_K: float = 0.001987204259  # Boltzmann constant in kcal/(mol*K)
PROBE_RADIUS_WATER: float = 1.40  # Solvent probe radius (A)

DielectricMode = Literal["hybrid", "mehler_solmajer", "gb_obc2"]

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
        "HID",
        "HIE",
        "HIP",
        "ASH",
        "GLH",
        "LYN",
        "CYX",
        "CYM",
        "MSE",
    }
)

_METAL_FORMAL_CHARGES: dict[str, float] = {
    "ZN": 2.0,
    "MG": 2.0,
    "CA": 2.0,
    "FE": 2.0,
    "MN": 2.0,
    "CU": 2.0,
    "CO": 2.0,
    "NI": 2.0,
    "NA": 1.0,
    "K": 1.0,
}

# Bondi / GAFF2 van der Waals (sigma in A, epsilon in kcal/mol), mbondi2 Born radius (A),
# OBC2 scaling factor S_i, and isotropic dipole polarizability alpha_i (A^3).
# Keyed by atomic number Z with fallback for uncommon elements.
_ELEMENT_BIOPHYS_TABLE: dict[int, tuple[float, float, float, float, float]] = {
    # Z: (sigma_A, epsilon_kcal_mol, born_radius_A, obc_scale, polarizability_A3)
    1: (2.471, 0.0157, 1.25, 0.85, 0.387),  # H
    5: (3.450, 0.0950, 1.80, 0.80, 1.850),  # B
    6: (3.399, 0.0860, 1.70, 0.72, 1.200),  # C
    7: (3.250, 0.1700, 1.55, 0.79, 0.964),  # N
    8: (2.960, 0.2100, 1.50, 0.85, 0.637),  # O
    9: (3.118, 0.0610, 1.47, 0.88, 0.410),  # F
    11: (2.439, 0.0874, 1.80, 0.80, 0.240),  # Na+
    12: (1.640, 0.8750, 1.45, 0.80, 0.120),  # Mg2+
    14: (3.826, 0.3000, 2.10, 0.80, 3.800),  # Si
    15: (3.742, 0.2000, 1.85, 0.86, 3.350),  # P
    16: (3.564, 0.2500, 1.80, 0.96, 2.900),  # S
    17: (3.471, 0.2650, 1.75, 0.80, 2.315),  # Cl
    19: (3.038, 0.1937, 2.10, 0.80, 0.830),  # K+
    20: (2.360, 0.4500, 1.60, 0.80, 0.550),  # Ca2+
    25: (1.950, 0.0500, 1.40, 0.80, 0.650),  # Mn2+
    26: (1.950, 0.0500, 1.40, 0.80, 0.650),  # Fe2+
    29: (1.950, 0.0500, 1.40, 0.80, 0.600),  # Cu2+
    30: (1.960, 0.0125, 1.40, 0.80, 0.500),  # Zn2+
    34: (3.750, 0.2910, 1.90, 0.85, 3.770),  # Se
    35: (3.596, 0.3200, 1.85, 0.80, 3.013),  # Br
    53: (3.860, 0.4000, 1.98, 0.80, 4.690),  # I
}

_SYMBOL_TO_Z: dict[str, int] = {
    "H": 1,
    "D": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}

# Sanderson electronegativity & hardness parameters for clean Gasteiger-style fallback
_PAULING_CHI: dict[int, float] = {
    1: 2.20,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    34: 2.55,
    35: 2.96,
    53: 2.66,
}

# AMBER ff14SB-inspired protein partial charges (heavy-atom united-charge aware).
# When explicit hydrogens ARE present in the PDB, polar H gets +0.30..+0.42 and heavy atom
# gets the all-atom AMBER charge; whatever hydrogens are omitted in an X-ray PDB have their
# charge folded onto their parent heavy atom so every residue strictly conserves its
# physiological pH 7.4 formal charge (-1 ASP/GLU/CYM, +1 LYS/ARG/HIP, 0 neutral).
_RESIDUE_NET_FORMAL_CHARGE: dict[str, float] = {
    "ASP": -1.0,
    "GLU": -1.0,
    "CYM": -1.0,
    "LYS": 1.0,
    "ARG": 1.0,
    "HIP": 1.0,
}

_AMBER_UNITED_ATOM_CHARGES: dict[tuple[str, str], float] = {
    # Anionic carboxylates
    ("ASP", "CB"): -0.08,
    ("ASP", "CG"): 0.68,
    ("ASP", "OD1"): -0.80,
    ("ASP", "OD2"): -0.80,
    ("GLU", "CB"): 0.00,
    ("GLU", "CG"): -0.08,
    ("GLU", "CD"): 0.68,
    ("GLU", "OE1"): -0.80,
    ("GLU", "OE2"): -0.80,
    # Cationic basic sidechains
    ("LYS", "CB"): 0.02,
    ("LYS", "CG"): 0.03,
    ("LYS", "CD"): 0.05,
    ("LYS", "CE"): 0.15,
    ("LYS", "NZ"): 0.75,
    ("ARG", "CB"): 0.00,
    ("ARG", "CG"): 0.04,
    ("ARG", "CD"): 0.12,
    ("ARG", "NE"): -0.16,
    ("ARG", "CZ"): 0.50,
    ("ARG", "NH1"): 0.25,
    ("ARG", "NH2"): 0.25,
    ("HIP", "CB"): 0.05,
    ("HIP", "CG"): 0.15,
    ("HIP", "ND1"): 0.25,
    ("HIP", "CE1"): 0.30,
    ("HIP", "NE2"): 0.25,
    ("HIP", "CD2"): 0.00,
    # Neutral histidine tautomers
    ("HIS", "CG"): 0.12,
    ("HIS", "ND1"): -0.38,
    ("HIS", "CE1"): 0.28,
    ("HIS", "NE2"): -0.18,
    ("HIS", "CD2"): 0.16,
    ("HID", "CG"): 0.12,
    ("HID", "ND1"): -0.18,
    ("HID", "CE1"): 0.28,
    ("HID", "NE2"): -0.42,
    ("HID", "CD2"): 0.20,
    ("HIE", "CG"): 0.12,
    ("HIE", "ND1"): -0.42,
    ("HIE", "CE1"): 0.28,
    ("HIE", "NE2"): -0.18,
    ("HIE", "CD2"): 0.20,
    # Polar hydroxyl / thiol / amide sidechains
    ("SER", "CB"): 0.25,
    ("SER", "OG"): -0.25,
    ("THR", "CB"): 0.26,
    ("THR", "OG1"): -0.26,
    ("THR", "CG2"): 0.00,
    ("TYR", "CG"): -0.02,
    ("TYR", "CD1"): -0.04,
    ("TYR", "CD2"): -0.04,
    ("TYR", "CE1"): -0.06,
    ("TYR", "CE2"): -0.06,
    ("TYR", "CZ"): 0.36,
    ("TYR", "OH"): -0.14,
    ("ASN", "CB"): -0.04,
    ("ASN", "CG"): 0.58,
    ("ASN", "OD1"): -0.56,
    ("ASN", "ND2"): 0.02,
    ("GLN", "CG"): -0.04,
    ("GLN", "CD"): 0.58,
    ("GLN", "OE1"): -0.56,
    ("GLN", "NE2"): 0.02,
    ("CYS", "CB"): 0.12,
    ("CYS", "SG"): -0.12,
    ("CYM", "CB"): -0.12,
    ("CYM", "SG"): -0.88,
    ("MET", "CG"): 0.10,
    ("MET", "SD"): -0.22,
    ("MET", "CE"): 0.12,
    # Aromatic quadrupole-bearing sidechains
    ("PHE", "CG"): 0.06,
    ("PHE", "CD1"): -0.012,
    ("PHE", "CD2"): -0.012,
    ("PHE", "CE1"): -0.012,
    ("PHE", "CE2"): -0.012,
    ("PHE", "CZ"): -0.012,
    ("TRP", "CG"): -0.06,
    ("TRP", "CD1"): 0.08,
    ("TRP", "NE1"): -0.12,
    ("TRP", "CE2"): 0.12,
    ("TRP", "CD2"): 0.04,
    ("TRP", "CE3"): -0.015,
    ("TRP", "CZ2"): -0.015,
    ("TRP", "CZ3"): -0.015,
    ("TRP", "CH2"): -0.015,
}

_PROTEIN_AROMATIC_RING_SPECS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
    ),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HID": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HIE": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HIP": (("CG", "ND1", "CE1", "NE2", "CD2"),),
}

_PROTEIN_SIDECHAIN_DONORS: dict[str, frozenset[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "ASN": frozenset({"ND2"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "HID": frozenset({"ND1"}),
    "HIE": frozenset({"NE2"}),
    "HIP": frozenset({"ND1", "NE2"}),
    "LYS": frozenset({"NZ"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TRP": frozenset({"NE1"}),
    "TYR": frozenset({"OH"}),
    "CYS": frozenset({"SG"}),
    "ASH": frozenset({"OD1", "OD2"}),
    "GLH": frozenset({"OE1", "OE2"}),
}

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
}

_SMARTS_BASIC_AMINE = Chem.MolFromSmarts("[NX3,NX4+;!$(NC=O);!$(NS=O);!$(N=O);!$(N[a])]")
_SMARTS_AMIDINE_GUANIDINE = Chem.MolFromSmarts("[NX2]=[CX3]([NX3])[#6,N]")
_SMARTS_CARBOXYLATE = Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]")
_SMARTS_SULFO_PHOSPHO = Chem.MolFromSmarts("[SX4,PX4](=O)(=O)[OX1H0-,OX2H1]")
_SMARTS_TETRAZOLE = Chem.MolFromSmarts("c1nnn[nH,n-]1")
_SMARTS_NITRO = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
_SMARTS_HBA = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0]),$([F;v1])]"
)


# ---------------------------------------------------------------------------
# Differentiable Pure-Float32 Geometry Helpers
# ---------------------------------------------------------------------------


def _smooth_switch_c2(dist: Tensor, r_on: float, r_off: float) -> Tensor:
    """Exact CHARMM/OpenMM ``C^2`` potential switching envelope in ``[0, 1]``."""
    r2 = dist * dist
    on2 = float(r_on * r_on)
    off2 = float(r_off * r_off)
    denom = max((off2 - on2) ** 3, 1e-8)
    num = (off2 - r2).clamp(min=0.0).pow(2) * (off2 + 2.0 * r2 - 3.0 * on2)
    switched = (num / denom).clamp(min=0.0, max=1.0)
    return torch.where(dist <= r_on, torch.ones_like(dist), switched)


def compute_ring_geometry_from_coords(
    coords: Tensor,
    ring_atom_indices: Tensor,
    ring_atom_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Differentiably compute aromatic ring centroids and Newell unit normals.

    Uses Newell's cyclic polygon cross-product formula rather than SVD so regular
    6-membered rings with degenerate in-plane singular values remain ``C^infty``
    differentiable on CPU, CUDA, and Apple Silicon MPS without ``NaN`` backward passes.

    Args:
        coords: Atom coordinates ``[N, 3]`` (``float32``).
        ring_atom_indices: Ordered ring vertex indices ``[R, 6]`` (``long``).
        ring_atom_mask: Valid vertex mask ``[R, 6]`` (``bool``).

    Returns:
        Tuple ``(centroids [R, 3], normals [R, 3])`` in ``torch.float32``.
    """
    num_rings = ring_atom_indices.shape[0]
    if num_rings == 0 or coords.shape[0] == 0:
        empty = torch.zeros((0, 3), dtype=torch.float32, device=coords.device)
        return empty, empty

    safe_idx = ring_atom_indices.clamp(min=0, max=max(coords.shape[0] - 1, 0))
    pts = coords[safe_idx]  # [R, 6, 3]
    mask_f = ring_atom_mask.to(dtype=torch.float32).unsqueeze(-1)  # [R, 6, 1]
    counts = mask_f.sum(dim=1).clamp(min=1.0)  # [R, 1]
    centroids = (pts * mask_f).sum(dim=1) / counts  # [R, 3]

    rel = (pts - centroids.unsqueeze(1)) * mask_f  # [R, 6, 3]
    # Cyclic next valid vertex: for 5-membered rings slot 5 wraps to slot 0 cleanly
    ring_sizes = ring_atom_mask.sum(dim=1, keepdim=True).clamp(min=3)  # [R, 1]
    slot_arange = torch.arange(6, device=coords.device).unsqueeze(0).expand(num_rings, 6)
    next_slot = (slot_arange + 1) % ring_sizes  # [R, 6]
    rel_next = torch.gather(rel, dim=1, index=next_slot.unsqueeze(-1).expand(-1, -1, 3))

    cross_sum = (torch.linalg.cross(rel, rel_next, dim=-1) * mask_f).sum(dim=1)  # [R, 3]
    normals = cross_sum * torch.rsqrt((cross_sum * cross_sum).sum(dim=-1, keepdim=True) + 1e-8)
    return centroids, normals


def compute_halogen_axes_from_coords(
    coords: Tensor,
    halogen_bond_pairs: Tensor,
    fallback_axes: Tensor | None = None,
) -> Tensor:
    """Differentiably compute ``C->X`` unit sigma-hole vectors ``[N, 3]`` from ``coords``."""
    n_atoms = coords.shape[0]
    axes = (
        fallback_axes.to(dtype=torch.float32, device=coords.device).clone()
        if fallback_axes is not None and fallback_axes.shape == (n_atoms, 3)
        else torch.zeros((n_atoms, 3), dtype=torch.float32, device=coords.device)
    )
    if halogen_bond_pairs.shape[0] == 0 or n_atoms == 0:
        return axes

    c_idx = halogen_bond_pairs[:, 0]
    x_idx = halogen_bond_pairs[:, 1]
    vec_cx = coords[x_idx] - coords[c_idx]
    unit_cx = vec_cx * torch.rsqrt((vec_cx * vec_cx).sum(dim=-1, keepdim=True) + 1e-8)
    # Build out-of-place tensor via scatter to preserve autograd graph cleanly
    out = torch.zeros((n_atoms, 3), dtype=torch.float32, device=coords.device)
    if fallback_axes is not None and fallback_axes.shape == (n_atoms, 3):
        out = fallback_axes.to(dtype=torch.float32, device=coords.device).clone()
    out = out.index_put((x_idx,), unit_cx)
    return out


def compute_donor_axes_from_coords(
    coords: Tensor,
    is_donor: Tensor,
    donor_h_pairs: Tensor,
    bond_pairs: Tensor,
    fallback_axes: Tensor | None = None,
) -> Tensor:
    """Differentiably compute outward donor unit axes ``[N, 3]`` from ``coords``.

    For heavy donor atoms bearing an explicit polar hydrogen ``(D, H)``, points directly
    along ``D -> H``. For heavy-atom crystal donors lacking explicit hydrogens, points
    outward opposite the sum of covalent heavy-neighbor bond vectors ``-(sum_k (r_k - r_D))``
    so donor orientation remains strictly ``SO(3)``-equivariant and differentiable.
    """
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=coords.device)

    accum = torch.zeros((n_atoms, 3), dtype=torch.float32, device=coords.device)

    # 1. Covalent heavy-neighbor outward frame fallback: -(r_nbr - r_self)
    if bond_pairs.shape[0] > 0:
        src = bond_pairs[:, 0]
        dst = bond_pairs[:, 1]
        v_sd = coords[dst] - coords[src]
        u_sd = v_sd * torch.rsqrt((v_sd * v_sd).sum(dim=-1, keepdim=True) + 1e-8)
        # Avoid index_add_ for MPS determinism: dense mask or index_put accumulate
        eye_n = torch.eye(n_atoms, dtype=torch.float32, device=coords.device)
        outward = -(eye_n[src].T @ u_sd) - (eye_n[dst].T @ (-u_sd))
        accum = accum + outward

    # 2. Override with explicit D -> H vector wherever polar hydrogens exist
    if donor_h_pairs.shape[0] > 0:
        d_idx = donor_h_pairs[:, 0]
        h_idx = donor_h_pairs[:, 1]
        v_dh = coords[h_idx] - coords[d_idx]
        u_dh = v_dh * torch.rsqrt((v_dh * v_dh).sum(dim=-1, keepdim=True) + 1e-8)
        eye_n = torch.eye(n_atoms, dtype=torch.float32, device=coords.device)
        dh_sum = eye_n[d_idx].T @ u_dh
        has_h = (eye_n[d_idx].sum(dim=0) > 0.5).unsqueeze(-1)
        accum = torch.where(has_h, dh_sum, accum)

    sq = (accum * accum).sum(dim=-1, keepdim=True)
    unit = accum * torch.rsqrt(sq + 1e-8)
    if fallback_axes is not None and fallback_axes.shape == (n_atoms, 3):
        fb = fallback_axes.to(dtype=torch.float32, device=coords.device)
        unit = torch.where(sq > 1e-6, unit, fb)
    donor_mask = is_donor.unsqueeze(-1)
    return torch.where(donor_mask, unit, torch.zeros_like(unit))


# ---------------------------------------------------------------------------
# Public Dataclasses: BiophysicalAtomParams & ComplexEnergyBreakdown
# ---------------------------------------------------------------------------


@dataclass
class BiophysicalAtomParams:
    """Pure ``torch.float32`` biophysical force-field parameterization for a molecule or pocket.

    All floating-point tensors are strictly ``torch.float32`` for Apple Silicon MPS, CUDA,
    and CPU portability. Topological index tensors enable ``BiophysicalHamiltonian`` to
    recompute aromatic ring planes, halogen :math:`\\sigma`-hole axes, and H-bond donor
    vectors differentiably as exact ``SO(3)``-equivariant functions of current 3D coordinates.
    """

    coords: Tensor  # [N, 3] float32
    atomic_nums: Tensor  # [N] long
    partial_charges: Tensor  # [N] float32 (e)
    vdw_sigma: Tensor  # [N] float32 (A)
    vdw_epsilon: Tensor  # [N] float32 (kcal/mol)
    born_radii: Tensor  # [N] float32 (A)
    obc_scaling: Tensor  # [N] float32
    polarizability: Tensor  # [N] float32 (A^3)
    is_donor: Tensor  # [N] bool
    is_acceptor: Tensor  # [N] bool
    is_cationic: Tensor  # [N] bool
    is_anionic: Tensor  # [N] bool
    is_hydrophobic: Tensor  # [N] bool
    is_backbone: Tensor  # [N] bool
    is_halogen: Tensor  # [N] bool
    halogen_strength: Tensor  # [N] float32
    donor_axes: Tensor  # [N, 3] float32
    halogen_sigma_axes: Tensor  # [N, 3] float32
    ring_centroids: Tensor  # [R, 3] float32
    ring_normals: Tensor  # [R, 3] float32
    ring_atom_indices: Tensor  # [R, 6] long
    ring_atom_mask: Tensor  # [R, 6] bool
    halogen_bond_pairs: Tensor  # [X, 2] long: (c_idx, x_idx)
    donor_h_pairs: Tensor  # [H_d, 2] long: (donor_heavy_idx, h_idx)
    bond_pairs: Tensor  # [E, 2] long
    bond_eq_lengths: Tensor  # [E] float32 (A)
    angle_triplets: Tensor  # [A, 3] long: (i, center_j, k)
    angle_eq_cos: Tensor  # [A] float32: cos(theta_0)
    residue_indices: Tensor  # [N] long
    residue_labels: list[str] = field(default_factory=list)
    atom_names: list[str] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    rotatable_bonds: int = 0
    net_formal_charge: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        """Enforce ``torch.float32`` discipline on all floating-point tensors."""
        float_fields = (
            "coords",
            "partial_charges",
            "vdw_sigma",
            "vdw_epsilon",
            "born_radii",
            "obc_scaling",
            "polarizability",
            "halogen_strength",
            "donor_axes",
            "halogen_sigma_axes",
            "ring_centroids",
            "ring_normals",
            "bond_eq_lengths",
            "angle_eq_cos",
        )
        for fname in float_fields:
            val: Tensor = getattr(self, fname)
            if val.dtype != torch.float32:
                setattr(self, fname, val.to(dtype=torch.float32))

    @property
    def num_atoms(self) -> int:
        """Return total atom count ``N``."""
        return int(self.coords.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the PyTorch device hosting this parameter bundle."""
        return self.coords.device

    def to(self, device: torch.device | str) -> BiophysicalAtomParams:
        """Move all tensors to ``device`` while preserving ``float32`` types."""
        dev = torch.device(device)
        return BiophysicalAtomParams(
            coords=self.coords.to(device=dev, dtype=torch.float32),
            atomic_nums=self.atomic_nums.to(device=dev, dtype=torch.long),
            partial_charges=self.partial_charges.to(device=dev, dtype=torch.float32),
            vdw_sigma=self.vdw_sigma.to(device=dev, dtype=torch.float32),
            vdw_epsilon=self.vdw_epsilon.to(device=dev, dtype=torch.float32),
            born_radii=self.born_radii.to(device=dev, dtype=torch.float32),
            obc_scaling=self.obc_scaling.to(device=dev, dtype=torch.float32),
            polarizability=self.polarizability.to(device=dev, dtype=torch.float32),
            is_donor=self.is_donor.to(device=dev, dtype=torch.bool),
            is_acceptor=self.is_acceptor.to(device=dev, dtype=torch.bool),
            is_cationic=self.is_cationic.to(device=dev, dtype=torch.bool),
            is_anionic=self.is_anionic.to(device=dev, dtype=torch.bool),
            is_hydrophobic=self.is_hydrophobic.to(device=dev, dtype=torch.bool),
            is_backbone=self.is_backbone.to(device=dev, dtype=torch.bool),
            is_halogen=self.is_halogen.to(device=dev, dtype=torch.bool),
            halogen_strength=self.halogen_strength.to(device=dev, dtype=torch.float32),
            donor_axes=self.donor_axes.to(device=dev, dtype=torch.float32),
            halogen_sigma_axes=self.halogen_sigma_axes.to(device=dev, dtype=torch.float32),
            ring_centroids=self.ring_centroids.to(device=dev, dtype=torch.float32),
            ring_normals=self.ring_normals.to(device=dev, dtype=torch.float32),
            ring_atom_indices=self.ring_atom_indices.to(device=dev, dtype=torch.long),
            ring_atom_mask=self.ring_atom_mask.to(device=dev, dtype=torch.bool),
            halogen_bond_pairs=self.halogen_bond_pairs.to(device=dev, dtype=torch.long),
            donor_h_pairs=self.donor_h_pairs.to(device=dev, dtype=torch.long),
            bond_pairs=self.bond_pairs.to(device=dev, dtype=torch.long),
            bond_eq_lengths=self.bond_eq_lengths.to(device=dev, dtype=torch.float32),
            angle_triplets=self.angle_triplets.to(device=dev, dtype=torch.long),
            angle_eq_cos=self.angle_eq_cos.to(device=dev, dtype=torch.float32),
            residue_indices=self.residue_indices.to(device=dev, dtype=torch.long),
            residue_labels=list(self.residue_labels),
            atom_names=list(self.atom_names),
            elements=list(self.elements),
            rotatable_bonds=int(self.rotatable_bonds),
            net_formal_charge=float(self.net_formal_charge),
            name=self.name,
        )

    def with_coords(self, new_coords: Tensor) -> BiophysicalAtomParams:
        """Return an updated copy with refreshed coordinates and derived directional vectors."""
        coords_f = new_coords.detach().to(device=self.device, dtype=torch.float32)
        rc, rn = compute_ring_geometry_from_coords(
            coords_f, self.ring_atom_indices, self.ring_atom_mask
        )
        h_axes = compute_halogen_axes_from_coords(
            coords_f, self.halogen_bond_pairs, self.halogen_sigma_axes
        )
        d_axes = compute_donor_axes_from_coords(
            coords_f, self.is_donor, self.donor_h_pairs, self.bond_pairs, self.donor_axes
        )
        cloned = self.to(self.device)
        cloned.coords = coords_f
        cloned.ring_centroids = rc
        cloned.ring_normals = rn
        cloned.halogen_sigma_axes = h_axes
        cloned.donor_axes = d_axes
        return cloned


@dataclass
class ComplexEnergyBreakdown:
    """Full biophysical energy decomposition, analytical 3D forces, and electric fields.

    Energies are reported in :math:`\\text{kcal/mol}`, forces in
    :math:`\\text{kcal}\\cdot\\text{mol}^{-1}\\cdot\\text{\\AA}^{-1}`, and electric fields
    in :math:`\\text{V/\\AA}`.
    """

    e_coulomb: float
    dg_gb_solvation: float
    e_vdw: float
    e_hbond: float
    e_halogen: float
    e_cation_pi: float
    e_pi_stacking: float
    dg_hydrophobic: float
    tds_rotational_penalty: float
    e_internal_strain: float
    u_total: float
    dg_bind_est: float
    ligand_forces_total: Tensor  # [N_lig, 3] float32
    pocket_forces_total: Tensor  # [N_rec, 3] float32
    ligand_forces_elec: Tensor  # [N_lig, 3] float32
    pocket_forces_elec: Tensor  # [N_rec, 3] float32
    ligand_forces_vdw: Tensor  # [N_lig, 3] float32
    pocket_forces_vdw: Tensor  # [N_rec, 3] float32
    ligand_electric_field: Tensor  # [N_lig, 3] float32 (V/A)
    pocket_electric_field: Tensor  # [N_rec, 3] float32 (V/A)
    ligand_born_radii: Tensor  # [N_lig] float32 (A)
    pocket_born_radii: Tensor  # [N_rec] float32 (A)
    per_residue_energy: dict[str, float] = field(default_factory=dict)
    per_residue_components: dict[str, dict[str, float]] = field(default_factory=dict)
    total_energy_tensor: Tensor | None = field(default=None, repr=False)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize scalar energy terms and per-residue hotspot breakdown to plain JSON dict."""
        return {
            "e_coulomb": round(self.e_coulomb, 4),
            "dg_gb_solvation": round(self.dg_gb_solvation, 4),
            "e_vdw": round(self.e_vdw, 4),
            "e_hbond": round(self.e_hbond, 4),
            "e_halogen": round(self.e_halogen, 4),
            "e_cation_pi": round(self.e_cation_pi, 4),
            "e_pi_stacking": round(self.e_pi_stacking, 4),
            "dg_hydrophobic": round(self.dg_hydrophobic, 4),
            "tds_rotational_penalty": round(self.tds_rotational_penalty, 4),
            "e_internal_strain": round(self.e_internal_strain, 4),
            "u_total": round(self.u_total, 4),
            "dg_bind_est": round(self.dg_bind_est, 4),
            "max_ligand_force_kcal_mol_a": round(
                float(torch.linalg.vector_norm(self.ligand_forces_total, dim=-1).max().item())
                if self.ligand_forces_total.numel() > 0
                else 0.0,
                4,
            ),
            "max_ligand_efield_v_a": round(
                float(torch.linalg.vector_norm(self.ligand_electric_field, dim=-1).max().item())
                if self.ligand_electric_field.numel() > 0
                else 0.0,
                4,
            ),
            "per_residue_energy": {k: round(v, 4) for k, v in self.per_residue_energy.items()},
        }


# ---------------------------------------------------------------------------
# Atomic Parameterization Functions (Ligand & Protein Pocket)
# ---------------------------------------------------------------------------


def _lookup_element_biophys(
    atomic_num: int,
    *,
    is_aromatic: bool = False,
    is_polar_h: bool = False,
    is_carbonyl_o: bool = False,
) -> tuple[float, float, float, float, float]:
    """Return ``(sigma_A, epsilon_kcal_mol, born_radius_A, obc_scale, polarizability_A3)``."""
    base = _ELEMENT_BIOPHYS_TABLE.get(atomic_num, (3.400, 0.1000, 1.70, 0.80, 1.500))
    sigma, eps, born_r, obc_s, alpha = base
    if atomic_num == 1 and is_polar_h:
        return (1.960, 0.0157, 1.30, 0.85, 0.350)
    if atomic_num == 6 and is_aromatic:
        return (3.399, 0.0860, 1.70, 0.72, 1.352)
    if atomic_num == 8 and is_carbonyl_o:
        return (2.960, 0.2100, 1.50, 0.85, 0.680)
    return (sigma, eps, born_r, obc_s, alpha)


def _match_indices(mol: Chem.Mol, query: Chem.Mol | None) -> set[int]:
    """Collect all atom indices matching a SMARTS pattern."""
    if query is None:
        return set()
    out: set[int] = set()
    for match in mol.GetSubstructMatches(query):
        out.update(match)
    return out


def _compute_conserved_gasteiger_charges(
    mol: Chem.Mol,
    polarizabilities: NDArray[np.float32],
) -> tuple[NDArray[np.float32], float]:
    """Compute Gasteiger–Marsili / EEM partial charges strictly conserving formal charge."""
    n_atoms = mol.GetNumAtoms()
    formal_charges = np.array(
        [float(mol.GetAtomWithIdx(i).GetFormalCharge()) for i in range(n_atoms)],
        dtype=np.float32,
    )
    net_formal = float(np.sum(formal_charges))

    AllChem.ComputeGasteigerCharges(mol, nIter=12, throwOnParamFailure=False)  # type: ignore[attr-defined]
    raw_q = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        val = float("nan")
        if atom.HasProp("_GasteigerCharge"):
            try:
                val = float(atom.GetDoubleProp("_GasteigerCharge"))
            except Exception:
                val = float("nan")
        if not math.isfinite(val):
            # Electronegativity equalization fallback across immediate covalent neighborhood
            chi_i = _PAULING_CHI.get(atom.GetAtomicNum(), 2.50)
            nbr_chis = [_PAULING_CHI.get(nb.GetAtomicNum(), 2.50) for nb in atom.GetNeighbors()]
            dq = sum((nb_c - chi_i) * 0.14 for nb_c in nbr_chis)
            val = float(formal_charges[i]) + dq
        else:
            # Blend formal charge localization on ionized heteroatoms (RDKit Gasteiger
            # damps formal charges slightly on resonance systems)
            if abs(formal_charges[i]) > 0.5:
                val = 0.55 * val + 0.45 * float(formal_charges[i])
        raw_q[i] = np.float32(val)

    # Strict formal-charge conservation weighted by atomic softness (polarizability)
    weights = np.maximum(polarizabilities, np.float32(0.20))
    w_sum = float(np.sum(weights))
    residual = net_formal - float(np.sum(raw_q))
    if w_sum > 1e-8:
        corr = float(residual / w_sum)
        raw_q = np.asarray(raw_q + corr * weights, dtype=np.float32)
    return raw_q, net_formal


def parameterize_ligand_biophysics(
    mol_or_smiles: Chem.Mol | str,
    *,
    conf_id: int = 0,
    keep_hydrogens: bool = True,
    device: torch.device | str | None = None,
) -> BiophysicalAtomParams:
    """Parameterize an RDKit 3D molecule or SMILES string into :class:`BiophysicalAtomParams`.

    Args:
        mol_or_smiles: 3D RDKit ``Chem.Mol`` (embedded automatically if lacking a conformer)
            or SMILES string.
        conf_id: Conformer index to read coordinates from when ``mol_or_smiles`` is a ``Mol``.
        keep_hydrogens: Retain/add explicit 3D hydrogens for directional H-bond and
            Gasteiger–Marsili charge accuracy.
        device: Optional target PyTorch device (``cpu``, ``mps``, ``cuda``).

    Returns:
        :class:`BiophysicalAtomParams` in pure ``torch.float32``.

    Raises:
        InvalidMoleculeError: If SMILES parsing or 3D conformer generation fails.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")

    if isinstance(mol_or_smiles, str):
        smi = mol_or_smiles.strip()
        if not smi:
            raise InvalidMoleculeError("Empty SMILES string provided to biophysics parameterizer.")
        conf_res = embed_and_minimize_3d(smi, max_iters=150, seed=42)
        if not conf_res.success or conf_res.mol_3d is None:
            raise InvalidMoleculeError(
                f"Failed to generate 3D conformer for ligand {smi!r}: {conf_res.error}",
                hint="Pass a chemically valid SMILES string or pre-embedded 3D RDKit Mol.",
            )
        mol = conf_res.mol_3d
        mol_name = smi
        conf_id = 0
    elif isinstance(mol_or_smiles, Chem.Mol):
        mol = Chem.Mol(mol_or_smiles)
        mol_name = Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol.GetNumAtoms() > 0 else "ligand"
        if mol.GetNumAtoms() == 0:
            raise InvalidMoleculeError("Ligand RDKit Mol contains zero atoms.")
        if mol.GetNumConformers() == 0:
            mol_h = Chem.AddHs(mol)
            status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())  # type: ignore[attr-defined]
            if status != 0:
                raise InvalidMoleculeError("Failed to embed 3D coordinates for conformerless Mol.")
            mol = mol_h
            conf_id = 0
    else:
        raise InvalidMoleculeError(f"Expected RDKit Mol or SMILES str, got {type(mol_or_smiles)!r}")

    has_explicit_h = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    if keep_hydrogens and not has_explicit_h:
        mol = Chem.AddHs(mol, addCoords=True)
    elif not keep_hydrogens and has_explicit_h:
        mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    coords_np = np.zeros((n_atoms, 3), dtype=np.float32)
    atomic_nums_np = np.zeros(n_atoms, dtype=np.int64)
    sigma_np = np.zeros(n_atoms, dtype=np.float32)
    eps_np = np.zeros(n_atoms, dtype=np.float32)
    born_np = np.zeros(n_atoms, dtype=np.float32)
    obc_np = np.zeros(n_atoms, dtype=np.float32)
    alpha_np = np.zeros(n_atoms, dtype=np.float32)
    is_donor_np = np.zeros(n_atoms, dtype=bool)
    is_acceptor_np = np.zeros(n_atoms, dtype=bool)
    is_cat_np = np.zeros(n_atoms, dtype=bool)
    is_ani_np = np.zeros(n_atoms, dtype=bool)
    is_hyd_np = np.zeros(n_atoms, dtype=bool)
    is_hal_np = np.zeros(n_atoms, dtype=bool)
    hal_str_np = np.zeros(n_atoms, dtype=np.float32)

    nitro_set = _match_indices(mol, _SMARTS_NITRO)
    basic_set = _match_indices(mol, _SMARTS_BASIC_AMINE)
    if _SMARTS_AMIDINE_GUANIDINE is not None:
        for match in mol.GetSubstructMatches(_SMARTS_AMIDINE_GUANIDINE):
            for idx in match:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7:
                    basic_set.add(idx)

    anionic_set: set[int] = set()
    for q_pat in (_SMARTS_CARBOXYLATE, _SMARTS_SULFO_PHOSPHO):
        if q_pat is not None:
            for match in mol.GetSubstructMatches(q_pat):
                for idx in match:
                    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8:
                        anionic_set.add(idx)
    if _SMARTS_TETRAZOLE is not None:
        for match in mol.GetSubstructMatches(_SMARTS_TETRAZOLE):
            for idx in match:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7:
                    anionic_set.add(idx)

    hba_set = _match_indices(mol, _SMARTS_HBA)

    elements: list[str] = []
    atom_names: list[str] = []
    donor_h_list: list[tuple[int, int]] = []
    hal_pair_list: list[tuple[int, int]] = []

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        z = int(atom.GetAtomicNum())
        pos = conf.GetAtomPosition(i)
        coords_np[i] = (np.float32(pos.x), np.float32(pos.y), np.float32(pos.z))
        atomic_nums_np[i] = z
        sym = atom.GetSymbol()
        elements.append(sym)
        atom_names.append(f"{sym}{i + 1}")

        nbrs = list(atom.GetNeighbors())
        is_polar_h = z == 1 and any(nb.GetAtomicNum() in {7, 8, 16} for nb in nbrs)
        is_carbonyl_o = z == 8 and any(
            nb.GetAtomicNum() in {6, 15, 16}
            and mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondTypeAsDouble() >= 1.5
            for nb in nbrs
        )
        sig, eps, br, obc_s, alp = _lookup_element_biophys(
            z,
            is_aromatic=atom.GetIsAromatic(),
            is_polar_h=is_polar_h,
            is_carbonyl_o=is_carbonyl_o,
        )
        sigma_np[i] = np.float32(sig)
        eps_np[i] = np.float32(eps)
        born_np[i] = np.float32(br)
        obc_np[i] = np.float32(obc_s)
        alpha_np[i] = np.float32(alp)

        f_chg = atom.GetFormalCharge()
        has_h = atom.GetTotalNumHs(includeNeighbors=True) > 0
        if z in {7, 8, 16} and has_h:
            is_donor_np[i] = True
            for nb in nbrs:
                if nb.GetAtomicNum() == 1:
                    donor_h_list.append((i, nb.GetIdx()))

        if (i in hba_set) or (z in {7, 8, 16} and f_chg < 0 and i not in nitro_set):
            is_acceptor_np[i] = True

        if i not in nitro_set and ((i in basic_set) or (f_chg > 0 and z in {7, 15})):
            is_cat_np[i] = True
        if i not in nitro_set and ((i in anionic_set) or (f_chg < 0 and z in {7, 8, 16})):
            is_ani_np[i] = True

        heavy_nbr_z = [nb.GetAtomicNum() for nb in nbrs if nb.GetAtomicNum() > 1]
        if z == 6 and all(nz == 6 for nz in heavy_nbr_z):
            is_hyd_np[i] = True

        if z in {17, 35, 53}:
            c_nbrs = [nb for nb in nbrs if nb.GetAtomicNum() == 6]
            if c_nbrs:
                c_atom = c_nbrs[0]
                is_hal_np[i] = True
                hal_pair_list.append((c_atom.GetIdx(), i))
                base_hal = {17: 1.0, 35: 1.65, 53: 2.35}[z]
                # Electron-withdrawing aromatic / fluorinated ring boosts sigma-hole potential
                ewg_boost = 1.15 if c_atom.GetIsAromatic() else 1.0
                hal_str_np[i] = np.float32(base_hal * ewg_boost)

    charges_np, net_formal = _compute_conserved_gasteiger_charges(mol, alpha_np)

    # Covalent bonds & harmonic equilibrium lengths
    bond_list: list[tuple[int, int]] = []
    bond_len_list: list[float] = []
    for bond in mol.GetBonds():
        bi = bond.GetBeginAtomIdx()
        bj = bond.GetEndAtomIdx()
        bond_list.append((bi, bj))
        d0 = float(np.linalg.norm(coords_np[bi] - coords_np[bj]))
        bond_len_list.append(max(d0, 0.85))

    # Valence bond angle triplets (i, center_j, k)
    angle_list: list[tuple[int, int, int]] = []
    angle_cos_list: list[float] = []
    for j in range(n_atoms):
        nbr_indices = [nb.GetIdx() for nb in mol.GetAtomWithIdx(j).GetNeighbors()]
        for a_pos in range(len(nbr_indices)):
            for b_pos in range(a_pos + 1, len(nbr_indices)):
                ia = nbr_indices[a_pos]
                ib = nbr_indices[b_pos]
                v1 = coords_np[ia] - coords_np[j]
                v2 = coords_np[ib] - coords_np[j]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 > 1e-5 and n2 > 1e-5:
                    cos0 = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    angle_list.append((ia, j, ib))
                    angle_cos_list.append(cos0)

    # Ordered 5- and 6-membered aromatic rings
    ring_idx_rows: list[list[int]] = []
    ring_mask_rows: list[list[bool]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) in {5, 6} and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            padded_idx = list(ring) + [ring[0]] * (6 - len(ring))
            padded_mask = [True] * len(ring) + [False] * (6 - len(ring))
            ring_idx_rows.append(padded_idx)
            ring_mask_rows.append(padded_mask)

    coords_t = torch.from_numpy(coords_np).to(device=dev, dtype=torch.float32)
    ring_idx_t = (
        torch.tensor(ring_idx_rows, dtype=torch.long, device=dev)
        if ring_idx_rows
        else torch.zeros((0, 6), dtype=torch.long, device=dev)
    )
    ring_mask_t = (
        torch.tensor(ring_mask_rows, dtype=torch.bool, device=dev)
        if ring_mask_rows
        else torch.zeros((0, 6), dtype=torch.bool, device=dev)
    )
    hal_pairs_t = (
        torch.tensor(hal_pair_list, dtype=torch.long, device=dev)
        if hal_pair_list
        else torch.zeros((0, 2), dtype=torch.long, device=dev)
    )
    donor_h_t = (
        torch.tensor(donor_h_list, dtype=torch.long, device=dev)
        if donor_h_list
        else torch.zeros((0, 2), dtype=torch.long, device=dev)
    )
    bond_pairs_t = (
        torch.tensor(bond_list, dtype=torch.long, device=dev)
        if bond_list
        else torch.zeros((0, 2), dtype=torch.long, device=dev)
    )
    bond_len_t = (
        torch.tensor(bond_len_list, dtype=torch.float32, device=dev)
        if bond_len_list
        else torch.zeros((0,), dtype=torch.float32, device=dev)
    )
    angle_trip_t = (
        torch.tensor(angle_list, dtype=torch.long, device=dev)
        if angle_list
        else torch.zeros((0, 3), dtype=torch.long, device=dev)
    )
    angle_cos_t = (
        torch.tensor(angle_cos_list, dtype=torch.float32, device=dev)
        if angle_cos_list
        else torch.zeros((0,), dtype=torch.float32, device=dev)
    )
    is_donor_t = torch.from_numpy(is_donor_np).to(device=dev, dtype=torch.bool)

    ring_centroids_t, ring_normals_t = compute_ring_geometry_from_coords(
        coords_t, ring_idx_t, ring_mask_t
    )
    hal_axes_t = compute_halogen_axes_from_coords(coords_t, hal_pairs_t)
    donor_axes_t = compute_donor_axes_from_coords(coords_t, is_donor_t, donor_h_t, bond_pairs_t)

    mol_heavy = Chem.RemoveHs(mol)
    n_rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol_heavy))

    return BiophysicalAtomParams(
        coords=coords_t,
        atomic_nums=torch.from_numpy(atomic_nums_np).to(device=dev, dtype=torch.long),
        partial_charges=torch.from_numpy(charges_np).to(device=dev, dtype=torch.float32),
        vdw_sigma=torch.from_numpy(sigma_np).to(device=dev, dtype=torch.float32),
        vdw_epsilon=torch.from_numpy(eps_np).to(device=dev, dtype=torch.float32),
        born_radii=torch.from_numpy(born_np).to(device=dev, dtype=torch.float32),
        obc_scaling=torch.from_numpy(obc_np).to(device=dev, dtype=torch.float32),
        polarizability=torch.from_numpy(alpha_np).to(device=dev, dtype=torch.float32),
        is_donor=is_donor_t,
        is_acceptor=torch.from_numpy(is_acceptor_np).to(device=dev, dtype=torch.bool),
        is_cationic=torch.from_numpy(is_cat_np).to(device=dev, dtype=torch.bool),
        is_anionic=torch.from_numpy(is_ani_np).to(device=dev, dtype=torch.bool),
        is_hydrophobic=torch.from_numpy(is_hyd_np).to(device=dev, dtype=torch.bool),
        is_backbone=torch.zeros(n_atoms, dtype=torch.bool, device=dev),
        is_halogen=torch.from_numpy(is_hal_np).to(device=dev, dtype=torch.bool),
        halogen_strength=torch.from_numpy(hal_str_np).to(device=dev, dtype=torch.float32),
        donor_axes=donor_axes_t,
        halogen_sigma_axes=hal_axes_t,
        ring_centroids=ring_centroids_t,
        ring_normals=ring_normals_t,
        ring_atom_indices=ring_idx_t,
        ring_atom_mask=ring_mask_t,
        halogen_bond_pairs=hal_pairs_t,
        donor_h_pairs=donor_h_t,
        bond_pairs=bond_pairs_t,
        bond_eq_lengths=bond_len_t,
        angle_triplets=angle_trip_t,
        angle_eq_cos=angle_cos_t,
        residue_indices=torch.zeros(n_atoms, dtype=torch.long, device=dev),
        residue_labels=["LIG1:L"],
        atom_names=atom_names,
        elements=elements,
        rotatable_bonds=n_rot,
        net_formal_charge=net_formal,
        name=mol_name,
    )


def _infer_pdb_element(raw_element: str, atom_name: str) -> str:
    """Resolve uppercase element symbol from PDB element column or atom name."""
    el = raw_element.strip().upper()
    if el and el in _SYMBOL_TO_Z:
        return el
    letters = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if len(letters) >= 2 and letters[:2] in _SYMBOL_TO_Z and letters[:2] not in {"CA", "CB", "CD"}:
        # Note: CA/CB/CD in standard amino acids are C-alpha/beta/delta carbons unless residue is CA
        return letters[:2]
    return letters[:1] if letters else "C"


def parameterize_pocket_biophysics(
    pdb_path_or_text: str | Path | Any,
    *,
    keep_waters: bool = False,
    keep_hydrogens: bool = True,
    device: torch.device | str | None = None,
) -> BiophysicalAtomParams:
    """Parameterize a 3D protein binding pocket into AMBER-inspired :class:`BiophysicalAtomParams`.

    Accepts a ``Path`` to a ``.pdb`` file, an inline PDB string containing ``ATOM``/``HETATM``
    records, or a :class:`~geneuclidean.sdk.Pocket` instance.

    Args:
        pdb_path_or_text: Pocket PDB path, raw PDB string, or ``Pocket`` object.
        keep_waters: Retain crystallographic water molecules when ``True``.
        keep_hydrogens: Retain explicit PDB hydrogens when ``True`` (automatically folds
            missing proton charges onto parent heavy atoms so residue formal charges hold).
        device: Optional target PyTorch device.

    Returns:
        :class:`BiophysicalAtomParams` in pure ``torch.float32``.

    Raises:
        StructureNotFoundError: If a file path does not exist.
        PocketFeaturizationError: If no valid protein/cofactor atoms are parsed.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")

    # Support geneuclidean.sdk.Pocket duck-typed or direct objects without eager SDK import
    if hasattr(pdb_path_or_text, "pocket_pdb"):
        pocket_pdb_attr = getattr(pdb_path_or_text, "pocket_pdb", None)
        if pocket_pdb_attr is None:
            raise StructureNotFoundError("Pocket object has no local pocket_pdb path attached.")
        pdb_path_or_text = pocket_pdb_attr

    parser = PDBParser(QUIET=True)
    pocket_name = "pocket"
    if isinstance(pdb_path_or_text, Path) or (
        isinstance(pdb_path_or_text, str)
        and "\n" not in pdb_path_or_text
        and not pdb_path_or_text.lstrip().startswith(("ATOM", "HETATM", "HEADER", "REMARK"))
    ):
        p = Path(pdb_path_or_text)
        if not p.is_file():
            raise StructureNotFoundError(f"Pocket PDB file not found: {p}")
        pocket_name = p.stem
        structure = parser.get_structure("pocket", str(p))
    elif isinstance(pdb_path_or_text, str):
        structure = parser.get_structure("pocket", io.StringIO(pdb_path_or_text))
    else:
        raise PocketFeaturizationError(f"Unsupported pocket input type: {type(pdb_path_or_text)!r}")

    models = list(structure.get_models())
    if not models:
        raise PocketFeaturizationError("PDB structure contains zero models.")

    coords_list: list[tuple[float, float, float]] = []
    z_list: list[int] = []
    q_list: list[float] = []
    sig_list: list[float] = []
    eps_list: list[float] = []
    born_list: list[float] = []
    obc_list: list[float] = []
    alpha_list: list[float] = []
    is_donor_list: list[bool] = []
    is_acc_list: list[bool] = []
    is_cat_list: list[bool] = []
    is_ani_list: list[bool] = []
    is_hyd_list: list[bool] = []
    is_bb_list: list[bool] = []
    res_idx_list: list[int] = []
    res_labels: list[str] = []
    atom_names: list[str] = []
    elements: list[str] = []

    bond_list: list[tuple[int, int]] = []
    bond_len_list: list[float] = []
    angle_list: list[tuple[int, int, int]] = []
    angle_cos_list: list[float] = []
    donor_h_list: list[tuple[int, int]] = []
    ring_idx_rows: list[list[int]] = []
    ring_mask_rows: list[list[bool]] = []

    net_formal_total = 0.0
    prev_c_idx_by_chain: dict[str, tuple[int, int, NDArray[np.float32]]] = {}

    for chain in models[0]:
        chain_id = str(chain.id).strip() or "A"
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            if not keep_waters and resname in _WATER_RESNAMES:
                continue
            hetflag, resseq, icode = residue.id
            is_metal = resname in _METAL_FORMAL_CHARGES
            if (
                hetflag != " "
                and resname not in _PROTEIN_RESNAMES
                and not is_metal
                and not (keep_waters and resname in _WATER_RESNAMES)
            ):
                continue

            icode_s = icode.strip() if isinstance(icode, str) else ""
            res_label = f"{resname}{int(resseq)}{icode_s}:{chain_id}"

            # Collect residue atoms
            raw_res_atoms: list[tuple[str, str, NDArray[np.float32]]] = []
            for atom in residue.get_atoms():
                aname = atom.get_name().strip().upper()
                raw_el = str(getattr(atom, "element", "")).strip().upper()
                el = resname if is_metal and aname == resname else _infer_pdb_element(raw_el, aname)
                if el in {"H", "D"} and not keep_hydrogens:
                    continue
                xyz = np.asarray(atom.get_coord(), dtype=np.float32)
                raw_res_atoms.append((aname, el, xyz))

            if not raw_res_atoms:
                continue

            res_idx = len(res_labels)
            res_labels.append(res_label)
            start_global = len(coords_list)

            # Separate heavy vs H within residue to map polar hydrogens & intra-residue bonds
            heavy_local: list[tuple[int, str, str, NDArray[np.float32]]] = []
            h_local: list[tuple[int, str, NDArray[np.float32]]] = []
            for loc_i, (aname, el, xyz) in enumerate(raw_res_atoms):
                if el in {"H", "D"}:
                    h_local.append((loc_i, aname, xyz))
                else:
                    heavy_local.append((loc_i, aname, el, xyz))

            attached_h_by_local: dict[int, list[int]] = {
                loc_i: [] for loc_i, _, _, _ in heavy_local
            }
            if heavy_local and h_local:
                heavy_xyz = np.stack([item[3] for item in heavy_local], axis=0)
                for h_loc, _, h_xyz in h_local:
                    dists = np.linalg.norm(heavy_xyz - h_xyz[None, :], axis=1)
                    nearest = int(np.argmin(dists))
                    if float(dists[nearest]) <= 1.32:
                        parent_loc = heavy_local[nearest][0]
                        attached_h_by_local[parent_loc].append(h_loc)

            # Assign residue charges with exact residue net formal charge conservation
            res_target_q = (
                _METAL_FORMAL_CHARGES[resname]
                if is_metal
                else _RESIDUE_NET_FORMAL_CHARGE.get(resname, 0.0)
            )
            if any(aname == "OXT" for aname, _, _ in raw_res_atoms):
                res_target_q -= 1.0

            res_q = np.zeros(len(raw_res_atoms), dtype=np.float32)
            for loc_i, (aname, el, _) in enumerate(raw_res_atoms):
                if is_metal:
                    res_q[loc_i] = np.float32(res_target_q)
                elif resname in _WATER_RESNAMES:
                    res_q[loc_i] = np.float32(-0.834 if el == "O" else 0.417)
                elif el in {"H", "D"}:
                    # Explicit H: polar amide/sidechain H gets +0.28, aliphatic H gets +0.06
                    is_polar_name = aname.startswith(
                        ("H", "HN", "HG", "HH", "HD", "HE", "HZ")
                    ) and (
                        aname in {"H", "HN"}
                        or any(
                            loc_i in attached_h_by_local.get(h_idx_p, [])
                            and p_el in {"N", "O", "S"}
                            for h_idx_p, _, p_el, _ in heavy_local
                        )
                    )
                    res_q[loc_i] = np.float32(0.28 if is_polar_name else 0.06)
                elif aname == "N":
                    has_hn = len(attached_h_by_local.get(loc_i, [])) > 0
                    base_n = -0.2548 if resname == "PRO" else (-0.4157 if has_hn else -0.1438)
                    res_q[loc_i] = np.float32(base_n)
                elif aname == "CA":
                    has_ha = len(attached_h_by_local.get(loc_i, [])) > 0
                    res_q[loc_i] = np.float32(0.0337 if has_ha else 0.1160)
                elif aname == "C":
                    res_q[loc_i] = np.float32(0.5973)
                elif aname == "O":
                    res_q[loc_i] = np.float32(-0.5679)
                elif aname == "OXT":
                    res_q[loc_i] = np.float32(-0.7000)
                else:
                    united_q = _AMBER_UNITED_ATOM_CHARGES.get((resname, aname), 0.0)
                    # Subtract charge already carried by explicit attached hydrogens
                    h_children = attached_h_by_local.get(loc_i, [])
                    h_q_est = sum(0.28 if el in {"N", "O", "S"} else 0.06 for _ in h_children)
                    res_q[loc_i] = np.float32(united_q - h_q_est)

            # Conserve exact target charge across standard complete amino acids
            if resname in _PROTEIN_RESNAMES and len(heavy_local) >= 4:
                q_diff = res_target_q - float(np.sum(res_q))
                res_q = res_q + np.float32(q_diff / len(raw_res_atoms))
                net_formal_total += res_target_q
            elif is_metal:
                net_formal_total += res_target_q

            sc_donors = _PROTEIN_SIDECHAIN_DONORS.get(resname, frozenset())
            sc_acceptors = _PROTEIN_SIDECHAIN_ACCEPTORS.get(resname, frozenset())
            name_to_global: dict[str, int] = {}

            # Precompute intra-residue heavy atom distance matrix for covalent bonds & apolar typing
            n_res = len(raw_res_atoms)
            res_xyz_mat = np.stack([item[2] for item in raw_res_atoms], axis=0)
            res_dmat = np.linalg.norm(res_xyz_mat[:, None, :] - res_xyz_mat[None, :, :], axis=-1)

            for loc_i, (aname, el, xyz) in enumerate(raw_res_atoms):
                g_idx = start_global + loc_i
                name_to_global[aname] = g_idx
                z = _SYMBOL_TO_Z.get(el, 6)
                is_bb = aname in {"N", "CA", "C", "O", "OXT", "H", "HN", "HA", "HA2", "HA3"}
                is_arom = resname in _PROTEIN_AROMATIC_RING_SPECS and not is_bb and el in {"C", "N"}
                is_polar_h = z == 1 and any(
                    loc_i in attached_h_by_local.get(p_loc, []) and p_el in {"N", "O", "S"}
                    for p_loc, _, p_el, _ in heavy_local
                )
                sig, eps, br, obc_s, alp = _lookup_element_biophys(
                    z,
                    is_aromatic=is_arom,
                    is_polar_h=is_polar_h,
                    is_carbonyl_o=(aname in {"O", "OXT", "OD1", "OD2", "OE1", "OE2"}),
                )

                is_don = (
                    (aname == "N" and resname != "PRO")
                    or (aname in sc_donors)
                    or (resname in _WATER_RESNAMES and el == "O")
                )
                is_acc = (
                    (aname in {"O", "OXT"})
                    or (aname in sc_acceptors)
                    or (resname in _WATER_RESNAMES and el == "O")
                )
                is_ani = (
                    (resname == "ASP" and aname in {"OD1", "OD2"})
                    or (resname == "GLU" and aname in {"OE1", "OE2"})
                    or (resname == "CYM" and aname == "SG")
                    or aname == "OXT"
                )
                is_cat = (
                    (resname == "LYS" and aname == "NZ")
                    or (resname == "ARG" and aname in {"NE", "NH1", "NH2"})
                    or (resname == "HIP" and aname in {"ND1", "NE2"})
                    or is_metal
                )

                is_hyd = False
                if z == 6 and aname not in {"C", "CA"}:
                    bonded_hetero = any(
                        raw_res_atoms[j][1] in {"N", "O", "S"} and float(res_dmat[loc_i, j]) <= 1.92
                        for j in range(n_res)
                        if j != loc_i
                    )
                    is_hyd = not bonded_hetero

                coords_list.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                z_list.append(z)
                q_list.append(float(res_q[loc_i]))
                sig_list.append(sig)
                eps_list.append(eps)
                born_list.append(br)
                obc_list.append(obc_s)
                alpha_list.append(alp)
                is_donor_list.append(is_don)
                is_acc_list.append(is_acc)
                is_cat_list.append(is_cat)
                is_ani_list.append(is_ani)
                is_hyd_list.append(is_hyd)
                is_bb_list.append(is_bb)
                res_idx_list.append(res_idx)
                atom_names.append(aname)
                elements.append(el)

                if is_don:
                    for h_loc in attached_h_by_local.get(loc_i, []):
                        donor_h_list.append((g_idx, start_global + h_loc))

            # Intra-residue covalent bonds & angles from 3D bond cutoffs
            res_adj: dict[int, list[int]] = {i: [] for i in range(n_res)}
            for ia in range(n_res):
                el_a = raw_res_atoms[ia][1]
                for ib in range(ia + 1, n_res):
                    el_b = raw_res_atoms[ib][1]
                    if el_a in {"H", "D"} and el_b in {"H", "D"}:
                        continue
                    cutoff = (
                        1.30
                        if (el_a in {"H", "D"} or el_b in {"H", "D"})
                        else (1.98 if ("S" in {el_a, el_b}) else 1.72)
                    )
                    d_ab = float(res_dmat[ia, ib])
                    if 0.65 <= d_ab <= cutoff:
                        ga = start_global + ia
                        gb = start_global + ib
                        bond_list.append((ga, gb))
                        bond_len_list.append(d_ab)
                        res_adj[ia].append(ib)
                        res_adj[ib].append(ia)

            for jc, nbrs_loc in res_adj.items():
                for p1 in range(len(nbrs_loc)):
                    for p2 in range(p1 + 1, len(nbrs_loc)):
                        la = nbrs_loc[p1]
                        lb = nbrs_loc[p2]
                        v1 = res_xyz_mat[la] - res_xyz_mat[jc]
                        v2 = res_xyz_mat[lb] - res_xyz_mat[jc]
                        n1 = float(np.linalg.norm(v1))
                        n2 = float(np.linalg.norm(v2))
                        if n1 > 1e-5 and n2 > 1e-5:
                            cos0 = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                            angle_list.append(
                                (start_global + la, start_global + jc, start_global + lb)
                            )
                            angle_cos_list.append(cos0)

            # Inter-residue peptide bond C(i-1) -- N(i)
            if "N" in name_to_global and chain_id in prev_c_idx_by_chain:
                prev_seq, prev_c_gidx, prev_c_xyz = prev_c_idx_by_chain[chain_id]
                cur_n_gidx = name_to_global["N"]
                cur_n_xyz = np.asarray(coords_list[cur_n_gidx], dtype=np.float32)
                d_pep = float(np.linalg.norm(cur_n_xyz - prev_c_xyz))
                if int(resseq) - prev_seq == 1 and d_pep <= 1.65:
                    bond_list.append((prev_c_gidx, cur_n_gidx))
                    bond_len_list.append(d_pep)
            if "C" in name_to_global:
                c_gidx = name_to_global["C"]
                prev_c_idx_by_chain[chain_id] = (
                    int(resseq),
                    c_gidx,
                    np.asarray(coords_list[c_gidx], dtype=np.float32),
                )

            # Aromatic rings in residue
            for ring_spec in _PROTEIN_AROMATIC_RING_SPECS.get(resname, ()):
                if all(rname in name_to_global for rname in ring_spec):
                    r_gidxs = [name_to_global[rname] for rname in ring_spec]
                    padded_idx = r_gidxs + [r_gidxs[0]] * (6 - len(r_gidxs))
                    padded_mask = [True] * len(r_gidxs) + [False] * (6 - len(r_gidxs))
                    ring_idx_rows.append(padded_idx)
                    ring_mask_rows.append(padded_mask)

    n_total = len(coords_list)
    if n_total == 0:
        raise PocketFeaturizationError("No protein or cofactor atoms found in pocket structure.")

    coords_t = torch.tensor(coords_list, dtype=torch.float32, device=dev)
    ring_idx_t = (
        torch.tensor(ring_idx_rows, dtype=torch.long, device=dev)
        if ring_idx_rows
        else torch.zeros((0, 6), dtype=torch.long, device=dev)
    )
    ring_mask_t = (
        torch.tensor(ring_mask_rows, dtype=torch.bool, device=dev)
        if ring_mask_rows
        else torch.zeros((0, 6), dtype=torch.bool, device=dev)
    )
    donor_h_t = (
        torch.tensor(donor_h_list, dtype=torch.long, device=dev)
        if donor_h_list
        else torch.zeros((0, 2), dtype=torch.long, device=dev)
    )
    bond_pairs_t = (
        torch.tensor(bond_list, dtype=torch.long, device=dev)
        if bond_list
        else torch.zeros((0, 2), dtype=torch.long, device=dev)
    )
    bond_len_t = (
        torch.tensor(bond_len_list, dtype=torch.float32, device=dev)
        if bond_len_list
        else torch.zeros((0,), dtype=torch.float32, device=dev)
    )
    angle_trip_t = (
        torch.tensor(angle_list, dtype=torch.long, device=dev)
        if angle_list
        else torch.zeros((0, 3), dtype=torch.long, device=dev)
    )
    angle_cos_t = (
        torch.tensor(angle_cos_list, dtype=torch.float32, device=dev)
        if angle_cos_list
        else torch.zeros((0,), dtype=torch.float32, device=dev)
    )
    is_donor_t = torch.tensor(is_donor_list, dtype=torch.bool, device=dev)

    ring_centroids_t, ring_normals_t = compute_ring_geometry_from_coords(
        coords_t, ring_idx_t, ring_mask_t
    )
    donor_axes_t = compute_donor_axes_from_coords(coords_t, is_donor_t, donor_h_t, bond_pairs_t)

    return BiophysicalAtomParams(
        coords=coords_t,
        atomic_nums=torch.tensor(z_list, dtype=torch.long, device=dev),
        partial_charges=torch.tensor(q_list, dtype=torch.float32, device=dev),
        vdw_sigma=torch.tensor(sig_list, dtype=torch.float32, device=dev),
        vdw_epsilon=torch.tensor(eps_list, dtype=torch.float32, device=dev),
        born_radii=torch.tensor(born_list, dtype=torch.float32, device=dev),
        obc_scaling=torch.tensor(obc_list, dtype=torch.float32, device=dev),
        polarizability=torch.tensor(alpha_list, dtype=torch.float32, device=dev),
        is_donor=is_donor_t,
        is_acceptor=torch.tensor(is_acc_list, dtype=torch.bool, device=dev),
        is_cationic=torch.tensor(is_cat_list, dtype=torch.bool, device=dev),
        is_anionic=torch.tensor(is_ani_list, dtype=torch.bool, device=dev),
        is_hydrophobic=torch.tensor(is_hyd_list, dtype=torch.bool, device=dev),
        is_backbone=torch.tensor(is_bb_list, dtype=torch.bool, device=dev),
        is_halogen=torch.zeros(n_total, dtype=torch.bool, device=dev),
        halogen_strength=torch.zeros(n_total, dtype=torch.float32, device=dev),
        donor_axes=donor_axes_t,
        halogen_sigma_axes=torch.zeros((n_total, 3), dtype=torch.float32, device=dev),
        ring_centroids=ring_centroids_t,
        ring_normals=ring_normals_t,
        ring_atom_indices=ring_idx_t,
        ring_atom_mask=ring_mask_t,
        halogen_bond_pairs=torch.zeros((0, 2), dtype=torch.long, device=dev),
        donor_h_pairs=donor_h_t,
        bond_pairs=bond_pairs_t,
        bond_eq_lengths=bond_len_t,
        angle_triplets=angle_trip_t,
        angle_eq_cos=angle_cos_t,
        residue_indices=torch.tensor(res_idx_list, dtype=torch.long, device=dev),
        residue_labels=res_labels,
        atom_names=atom_names,
        elements=elements,
        rotatable_bonds=0,
        net_formal_charge=net_formal_total,
        name=pocket_name,
    )


# ---------------------------------------------------------------------------
# Full Differentiable Physical & Electrical Hamiltonian
# ---------------------------------------------------------------------------


class BiophysicalHamiltonian(nn.Module):
    """Differentiable ``torch.float32`` Protein–Pocket–Ligand Biophysical Hamiltonian.

    Evaluates exact ``SE(3)``-invariant molecular recognition energies and analytical
    ``SO(3)``-equivariant 3D forces and electric fields on MPS, CUDA, and CPU.
    """

    weights_buf: Tensor

    def __init__(
        self,
        *,
        dielectric_mode: DielectricMode = "hybrid",
        interior_dielectric: float = 4.0,
        solvent_dielectric: float = 78.5,
        use_gb_solvation: bool = True,
        gb_weight: float = 0.30,
        nonbonded_cutoff: float = 12.0,
        switch_on: float = 10.0,
        softcore_alpha: float = 0.50,
        hbond_well_kcal: float = 2.60,
        salt_bridge_boost: float = 1.55,
        halogen_crown_kcal: float = 2.80,
        halogen_belt_kcal: float = 1.15,
        cation_pi_kcal: float = 3.20,
        pi_pd_kcal: float = 2.20,
        pi_t_kcal: float = 1.85,
        sasa_surface_tension: float = -0.0072,
        rotatable_bond_penalty: float = 0.45,
    ) -> None:
        super().__init__()
        self.dielectric_mode: DielectricMode = dielectric_mode
        self.interior_dielectric = float(interior_dielectric)
        self.solvent_dielectric = float(solvent_dielectric)
        self.use_gb_solvation = bool(use_gb_solvation)
        self.gb_weight = float(gb_weight)
        self.nonbonded_cutoff = float(nonbonded_cutoff)
        self.switch_on = float(switch_on)
        self.softcore_alpha = float(softcore_alpha)
        self.hbond_well_kcal = float(hbond_well_kcal)
        self.salt_bridge_boost = float(salt_bridge_boost)
        self.halogen_crown_kcal = float(halogen_crown_kcal)
        self.halogen_belt_kcal = float(halogen_belt_kcal)
        self.cation_pi_kcal = float(cation_pi_kcal)
        self.pi_pd_kcal = float(pi_pd_kcal)
        self.pi_t_kcal = float(pi_t_kcal)
        self.sasa_surface_tension = float(sasa_surface_tension)
        self.rotatable_bond_penalty = float(rotatable_bond_penalty)

        # Register a pure float32 buffer of physical constants for MPS/device inspection
        self.register_buffer(
            "weights_buf",
            torch.tensor(
                [
                    COULOMB_KCAL_ANGSTROM,
                    KCAL_MOL_E_TO_VOLT,
                    self.interior_dielectric,
                    self.solvent_dielectric,
                    self.gb_weight,
                ],
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def mehler_solmajer_dielectric(dist: Tensor, min_eps: float = 4.0) -> Tensor:
        """Smooth Mehler–Solmajer sigmoidal dielectric :math:`\\varepsilon(r)`."""
        a_const = -8.5525
        b_const = 78.5 - a_const  # 87.0525
        kappa = 7.7839
        lam = 0.003627
        raw_eps = a_const + b_const / (1.0 + kappa * torch.exp(-lam * b_const * dist))
        # Smoothly floor at interior_dielectric (4.0) without any derivative kink
        return min_eps + torch.nn.functional.softplus(raw_eps - min_eps, beta=2.5)

    @staticmethod
    def compute_effective_born_radii(
        coords: Tensor,
        intrinsic_radii: Tensor,
        obc_scaling: Tensor,
        *,
        r_max: float = 30.0,
        dielectric_offset: float = 0.09,
    ) -> Tensor:
        r"""Compute differentiable OBC2 effective Born radii :math:`R_i` (``[N]``, Angstrom).

        Implements the Hawkins-Cramer-Truhlar pairwise descreening integral as used by
        AMBER ``igb=2`` and OpenMM's ``GBSAOBCForce``, closed with the
        Onufriev-Bashford-Case II rescaling

        .. math::
            R_i^{-1} = \tilde{\rho}_i^{-1}
                     - \rho_i^{-1}\tanh\!\big(\alpha\Psi - \beta\Psi^2 + \gamma\Psi^3\big),
            \qquad \Psi = \tilde{\rho}_i I_i,

        with :math:`\alpha = 1.0`, :math:`\beta = 0.8`, :math:`\gamma = 4.85` and the
        dielectric offset :math:`\tilde{\rho}_i = \rho_i - 0.09\,\text{\AA}`. Isolated atoms
        keep :math:`R_i \approx \tilde{\rho}_i`; buried atoms grow smoothly.

        .. note::
           **This replaces an ad-hoc burial kernel that silently disabled all polar
           solvation.** The previous version summed
           ``2.15 * rho_i * rho_j**3 / (r**2 + 0.35 * rho_j**2)**2``, which over-counts
           burial by roughly an order of magnitude. Measured on a real complex (PDB
           ``10gs``, 61 ligand atoms) it produced :math:`\Psi` with mean **4.20** (range
           2.23-7.93) where OBC2 expects :math:`O(0.1\text{-}3)`. The
           :math:`4.85\Psi^3` term then reached mean **486** (max 2371), saturating the old
           ``clamp(max=12)`` so ``tanh`` returned exactly ``1.0`` and **every atom
           collapsed to** :math:`R_i = r_{\max} = 12`\ Angstrom. Apo and complex radii became
           identical, so :meth:`_intra_gb_desolvation` returned identically ``0.0`` and
           ``ComplexEnergyBreakdown.dg_gb_solvation`` was ``0.0`` for every complex --
           indistinguishable from "this complex has no net desolvation".

           The unit tests missed it because they use sparse synthetic geometries: a
           12-atom shell at 6 Angstrom yields :math:`R_i = 1.82`\ Angstrom and behaves
           correctly, and the burial assertion ``R_bound > R_free`` was satisfied
           *trivially* by the degenerate value (12 > 1.70). Saturation appears only at
           realistic atom density.

           ``r_max`` is now a numerical safety ceiling rather than the operating point: the
           closing formula already bounds :math:`R_i` by
           :math:`\tilde{\rho}_i\rho_i/0.09 \approx 23`\ Angstrom.

        Args:
            coords: Atom coordinates ``[N, 3]``.
            intrinsic_radii: Per-atom intrinsic (Bondi/mbondi) radii ``[N]`` in Angstrom.
            obc_scaling: Per-atom HCT descreening scale factors ``[N]``.
            r_max: Safety ceiling on the returned radii, in Angstrom.
            dielectric_offset: AMBER dielectric offset subtracted from the intrinsic radii.

        Returns:
            Effective Born radii ``[N]`` as ``float32``.
        """
        n_atoms = coords.shape[0]
        rho_single = (intrinsic_radii.clamp(min=0.15) - float(dielectric_offset)).clamp(min=0.10)
        if n_atoms <= 1:
            # Match the formula's own isolated-atom limit: with no descreening neighbour
            # Psi = 0, tanh(0) = 0 and R_i = rho_tilde. Returning the *intrinsic* radius
            # here instead would make a lone atom disagree with two atoms held far apart.
            return rho_single.to(dtype=torch.float32)

        rho = intrinsic_radii.clamp(min=0.15)
        rho_t = (rho - float(dielectric_offset)).clamp(min=0.10)  # offset radii
        scaled = obc_scaling * rho_t

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        eye = torch.eye(n_atoms, dtype=torch.bool, device=coords.device)
        # Keep the diagonal at a harmless finite distance; it is masked out below.
        dist = torch.where(eye, torch.ones_like(dist), dist)

        rho_i = rho_t.unsqueeze(1)  # [N, 1] offset radius of the atom being descreened
        sr_j = scaled.unsqueeze(0)  # [1, N] scaled offset radius of the descreener

        upper = dist + sr_j
        lower = torch.maximum(rho_i, (dist - sr_j).abs())
        inv_l = 1.0 / lower.clamp(min=1e-6)
        inv_u = 1.0 / upper.clamp(min=1e-6)
        inv_rho_i = 1.0 / rho_i

        # HCT analytic descreening integral for one ordered pair, matching the reference
        # loop in OpenMM's GBSAOBCForce: the leading pair term is (1/L - 1/U), NOT
        # (1/rho_i - 1/L) -- the latter is only the engulfment correction below.
        bracket = (
            inv_l
            - inv_u
            + 0.25 * (dist - sr_j * sr_j / dist) * (inv_u * inv_u - inv_l * inv_l)
            + 0.5 * torch.log(lower / upper) / dist
        )
        # Atom i lies entirely inside the descreening sphere of atom j. The 0.5 prefactor
        # must multiply the engulfment addend TOO: OpenMM accumulates the bare pair terms
        # and scales the whole sum by 0.5*offsetRadiusI at the end. Adding 2.0*(...) after
        # the 0.5 instead of inside it makes this branch twice too large -- a latent
        # factor-2 that only bites once an atom is fully engulfed, which is why ordinary
        # bound poses never exposed it.
        engulfed = rho_i < (sr_j - dist)
        term = 0.5 * (
            bracket + torch.where(engulfed, 2.0 * (inv_rho_i - inv_l), torch.zeros_like(bracket))
        )
        # No contribution once j is too far to descreen i at all, and never self-descreen.
        term = torch.where(rho_i < upper, term, torch.zeros_like(term))
        term = torch.where(eye, torch.zeros_like(term), term)

        integral = term.sum(dim=-1).clamp(min=0.0)
        psi = integral * rho_t

        # OBC2 polynomial: alpha*Psi - beta*Psi^2 + gamma*Psi^3; alpha=1, beta=0.8, gamma=4.85
        obc_arg = psi * (1.0 - 0.8 * psi + 4.85 * psi * psi)
        inv_r_eff = (1.0 / rho_t) - torch.tanh(obc_arg) / rho
        inv_r_eff = inv_r_eff.clamp(min=1.0 / float(r_max))
        return (1.0 / inv_r_eff).to(dtype=torch.float32)

    @staticmethod
    def _still_f_gb(coords: Tensor, born_r: Tensor, *, soft_r2: float = 0.0) -> Tensor:
        """Pairwise Still Generalized Born effective distance denominator ``f_GB [N, N]``."""
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        sq_d = (diff * diff).sum(dim=-1) + soft_r2
        ri_rj = born_r.unsqueeze(1) * born_r.unsqueeze(0)
        return torch.sqrt(sq_d + ri_rj * torch.exp(-sq_d / (4.0 * ri_rj + 1e-8)) + 1e-8)

    def _intra_gb_desolvation(
        self,
        coords: Tensor,
        charges: Tensor,
        born_r_comp: Tensor,
        born_r_apo: Tensor,
        *,
        soft_r2: float = 0.0,
    ) -> Tensor:
        """Cancellation-free ``float32`` intramolecular OBC2 Born desolvation difference.

        Evaluates ``(1 / f_comp - 1 / f_apo) = (f_apo - f_comp) / (f_comp * f_apo)``
        pairwise before summing so large constant self-solvation terms never suffer
        ``float32`` catastrophic cancellation.
        """
        if coords.shape[0] == 0:
            return torch.zeros((), dtype=torch.float32, device=coords.device)
        f_comp = self._still_f_gb(coords, born_r_comp, soft_r2=soft_r2)
        f_apo = self._still_f_gb(coords, born_r_apo, soft_r2=soft_r2)
        delta_inv_f = (f_apo - f_comp) / (f_comp * f_apo + 1e-12)
        prefac = (
            -0.5
            * COULOMB_KCAL_ANGSTROM
            * ((1.0 / self.interior_dielectric) - (1.0 / self.solvent_dielectric))
        )
        qi_qj = charges.unsqueeze(1) * charges.unsqueeze(0)
        return prefac * (qi_qj * delta_inv_f).sum()

    def _compute_sasa_exposed(
        self,
        coords: Tensor,
        vdw_sigma: Tensor,
        is_hydrophobic: Tensor,
    ) -> Tensor:
        """Differentiable Gaussian-exclusion apolar solvent-accessible surface area (A^2)."""
        n_atoms = coords.shape[0]
        if n_atoms == 0:
            return torch.zeros((0,), dtype=torch.float32, device=coords.device)

        # Convert LJ sigma back to vdW radius r_vdw = 0.5 * 2^(1/6) * sigma
        r_vdw = 0.561231 * vdw_sigma
        r_solv = r_vdw + PROBE_RADIUS_WATER  # [N]
        max_area = 4.0 * math.pi * (r_solv * r_solv)

        if n_atoms == 1:
            return max_area * is_hydrophobic.to(dtype=torch.float32)

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        sq_d = (diff * diff).sum(dim=-1)
        dist = torch.sqrt(sq_d + 1e-8)
        not_self = ~torch.eye(n_atoms, dtype=torch.bool, device=coords.device)

        sum_rad = r_solv.unsqueeze(1) + r_solv.unsqueeze(0)
        # Smooth cubic overlap fraction when dist < sum_rad
        gap = ((sum_rad - dist) / sum_rad.clamp(min=1.0)).clamp(min=0.0, max=1.0)
        occ = 0.42 * gap.pow(3)
        occ = torch.where(not_self, occ, torch.zeros_like(occ))
        exposed = max_area * torch.exp(-occ.sum(dim=-1))
        # Weight apolar carbons full strength and polar heteroatoms weakly (15% cavitation)
        apolar_w = torch.where(
            is_hydrophobic,
            torch.ones_like(exposed),
            torch.full_like(exposed, 0.15),
        )
        return exposed * apolar_w

    def _compute_energy_terms(
        self,
        lig_coords: Tensor,
        rec_coords: Tensor,
        lig: BiophysicalAtomParams,
        rec: BiophysicalAtomParams,
        *,
        softcore_lambda: float = 1.0,
    ) -> dict[str, Any]:
        """Compute all differentiable intermolecular energy terms & per-pocket-atom attribution."""
        lam = float(max(0.05, min(1.0, softcore_lambda)))
        soft_r2 = 0.60 * (1.0 - lam)

        # Shift into ligand centroid shared frame to prevent float32 cancellation at large |x|
        origin = (
            lig_coords.mean(dim=0, keepdim=True)
            if lig_coords.shape[0] > 0
            else torch.zeros((1, 3), dtype=torch.float32, device=lig_coords.device)
        )
        lc = lig_coords - origin  # [NL, 3]
        rc = rec_coords - origin  # [NP, 3]

        nl = lc.shape[0]
        np_atoms = rc.shape[0]
        dev = lc.device

        if nl == 0 or np_atoms == 0:
            z0 = (lc.sum() + rc.sum()) * 0.0
            return {
                "e_coulomb": z0,
                "dg_gb": z0,
                "e_vdw": z0,
                "e_hbond": z0,
                "e_halogen": z0,
                "e_cation_pi": z0,
                "e_pi_stacking": z0,
                "dg_hydrophobic": z0,
                "lig_born_r": lig.born_radii,
                "rec_born_r": rec.born_radii,
                "lig_efield": torch.zeros_like(lc),
                "rec_efield": torch.zeros_like(rc),
                "rec_atom_energy": torch.zeros((np_atoms,), dtype=torch.float32, device=dev),
                "rec_atom_elec": torch.zeros((np_atoms,), dtype=torch.float32, device=dev),
                "rec_atom_vdw": torch.zeros((np_atoms,), dtype=torch.float32, device=dev),
                "rec_atom_hb": torch.zeros((np_atoms,), dtype=torch.float32, device=dev),
                "rec_atom_spec": torch.zeros((np_atoms,), dtype=torch.float32, device=dev),
            }

        # Pairwise ligand-pocket geometry [NL, NP, 3]
        diff_lp = lc.unsqueeze(1) - rc.unsqueeze(0)  # r_i - r_j
        sq_lp = (diff_lp * diff_lp).sum(dim=-1)  # [NL, NP]
        dist_lp = torch.sqrt(sq_lp + 1e-8)
        sw_cut = _smooth_switch_c2(dist_lp, self.switch_on, self.nonbonded_cutoff)

        # ------------------------------------------------------------------
        # 1. Coulomb Electrostatics + Exact 3D Electric Field Vectors (V/A)
        # ------------------------------------------------------------------
        r_elec = torch.sqrt(sq_lp + soft_r2 + 1e-8)
        if self.dielectric_mode == "gb_obc2":
            eps_ij = torch.full_like(dist_lp, self.interior_dielectric)
        else:
            eps_ij = self.mehler_solmajer_dielectric(dist_lp, min_eps=self.interior_dielectric)

        qi = lig.partial_charges.unsqueeze(1)  # [NL, 1]
        qj = rec.partial_charges.unsqueeze(0)  # [1, NP]
        pair_coulomb = (COULOMB_KCAL_ANGSTROM * qi * qj / (eps_ij * r_elec)) * sw_cut
        e_coulomb = pair_coulomb.sum()

        # Analytical 3D electric field E_i (V/A) produced by partner charges:
        # E(r_i) = sum_j (k_e * q_j / (eps(r_ij) * r_elec^3)) * (r_i - r_j) * V_conv
        efield_kernel = (
            COULOMB_KCAL_ANGSTROM
            * KCAL_MOL_E_TO_VOLT
            * sw_cut
            / (eps_ij * r_elec * r_elec * r_elec + 1e-8)
        )  # [NL, NP]
        lig_efield = (efield_kernel.unsqueeze(-1) * qj.unsqueeze(-1) * diff_lp).sum(dim=1)
        rec_efield = -(efield_kernel.unsqueeze(-1) * qi.unsqueeze(-1) * diff_lp).sum(dim=0)

        # ------------------------------------------------------------------
        # 2. Generalized Born (OBC2) Polar Solvation / Desolvation (dG_GB)
        # ------------------------------------------------------------------
        comp_coords = torch.cat([lc, rc], dim=0)
        comp_rho = torch.cat([lig.born_radii, rec.born_radii], dim=0)
        comp_obc = torch.cat([lig.obc_scaling, rec.obc_scaling], dim=0)

        comp_born_r = self.compute_effective_born_radii(comp_coords, comp_rho, comp_obc)
        lig_born_r = comp_born_r[:nl]
        rec_born_r = comp_born_r[nl:]

        if self.use_gb_solvation and self.dielectric_mode != "mehler_solmajer":
            lig_born_apo = self.compute_effective_born_radii(lc, lig.born_radii, lig.obc_scaling)
            rec_born_apo = self.compute_effective_born_radii(rc, rec.born_radii, rec.obc_scaling)

            dg_desolv_l = self._intra_gb_desolvation(
                lc, lig.partial_charges, lig_born_r, lig_born_apo, soft_r2=soft_r2
            )
            dg_desolv_r = self._intra_gb_desolvation(
                rc, rec.partial_charges, rec_born_r, rec_born_apo, soft_r2=soft_r2
            )

            if self.dielectric_mode == "gb_obc2":
                # In pure OBC2 mode, direct Coulomb uses constant interior_dielectric and Still's
                # cross-GB reaction field provides intermolecular solvent screening.
                ri_rj_lp = lig_born_r.unsqueeze(1) * rec_born_r.unsqueeze(0)
                sq_s = sq_lp + soft_r2
                f_gb_lp = torch.sqrt(
                    sq_s + ri_rj_lp * torch.exp(-sq_s / (4.0 * ri_rj_lp + 1e-8)) + 1e-8
                )
                cross_prefac = -COULOMB_KCAL_ANGSTROM * (
                    (1.0 / self.interior_dielectric) - (1.0 / self.solvent_dielectric)
                )
                g_cross = (cross_prefac * (qi * qj / f_gb_lp) * sw_cut).sum()
                dg_gb = dg_desolv_l + dg_desolv_r + g_cross
                gb_scale = 1.0
            else:
                # In hybrid mode, Mehler-Solmajer eps(r_ij) already screens cross-pair Coulomb,
                # while OBC2 captures the exact Born polar desolvation upon pocket burial.
                gb_scale = self.gb_weight
                dg_gb = gb_scale * (dg_desolv_l + dg_desolv_r)

            prefac_gb = (
                -0.5
                * COULOMB_KCAL_ANGSTROM
                * ((1.0 / self.interior_dielectric) - (1.0 / self.solvent_dielectric))
            )
            rec_self_desolv = (
                gb_scale
                * prefac_gb
                * (rec.partial_charges * rec.partial_charges)
                * ((rec_born_apo - rec_born_r) / (rec_born_r * rec_born_apo + 1e-12))
            )
        else:
            dg_gb = lc.sum() * 0.0
            rec_self_desolv = torch.zeros((np_atoms,), dtype=torch.float32, device=dev)

        # ------------------------------------------------------------------
        # 3. Soft-Core / Buffered Lennard-Jones 12-6 Dispersion & Repulsion
        # ------------------------------------------------------------------
        sig_ij = 0.5 * (lig.vdw_sigma.unsqueeze(1) + rec.vdw_sigma.unsqueeze(0))
        eps_ij_lj = torch.sqrt(lig.vdw_epsilon.unsqueeze(1) * rec.vdw_epsilon.unsqueeze(0) + 1e-12)
        r_over_sig2 = sq_lp / (sig_ij * sig_ij + 1e-8)
        r_over_sig6 = r_over_sig2 * r_over_sig2 * r_over_sig2
        u_sc = self.softcore_alpha * (1.0 - lam) + r_over_sig6 + 1e-4
        inv_u = 1.0 / u_sc
        pair_vdw = 4.0 * lam * eps_ij_lj * (inv_u * inv_u - inv_u) * sw_cut
        e_vdw = pair_vdw.sum()

        # ------------------------------------------------------------------
        # 4. Directional Hydrogen Bonds & Charge-Assisted Salt Bridges
        # ------------------------------------------------------------------
        lig_d_axes = compute_donor_axes_from_coords(
            lc, lig.is_donor, lig.donor_h_pairs, lig.bond_pairs, lig.donor_axes
        )
        rec_d_axes = compute_donor_axes_from_coords(
            rc, rec.is_donor, rec.donor_h_pairs, rec.bond_pairs, rec.donor_axes
        )

        # Unit vector from L_i -> P_j is -diff_lp / dist_lp
        u_l_to_p = -diff_lp / dist_lp.unsqueeze(-1)  # [NL, NP, 3]
        u_p_to_l = -u_l_to_p

        # Case A: Ligand donor -> Pocket acceptor
        da_mask_lp = lig.is_donor.unsqueeze(1) & rec.is_acceptor.unsqueeze(0)
        cos_lp = (lig_d_axes.unsqueeze(1) * u_l_to_p).sum(dim=-1)
        ang_lp = torch.relu((cos_lp - 0.15) / 0.85).pow(3)

        # Case B: Pocket donor -> Ligand acceptor
        da_mask_pl = rec.is_donor.unsqueeze(0) & lig.is_acceptor.unsqueeze(1)
        cos_pl = (rec_d_axes.unsqueeze(0) * u_p_to_l).sum(dim=-1)
        ang_pl = torch.relu((cos_pl - 0.15) / 0.85).pow(3)

        ang_gate = torch.where(da_mask_lp, ang_lp, torch.zeros_like(ang_lp)) + torch.where(
            da_mask_pl, ang_pl, torch.zeros_like(ang_pl)
        )
        ang_gate = ang_gate.clamp(max=1.25)

        is_salt = (lig.is_cationic.unsqueeze(1) & rec.is_anionic.unsqueeze(0)) | (
            lig.is_anionic.unsqueeze(1) & rec.is_cationic.unsqueeze(0)
        )
        hb_strength = self.hbond_well_kcal * torch.where(
            is_salt,
            torch.full_like(dist_lp, self.salt_bridge_boost),
            torch.ones_like(dist_lp),
        )

        # Smooth 12-10 Morse-like radial well centered at r0 = 2.85 A (soft-clamped inner wall)
        r0_hb = 2.85
        r_hb = torch.sqrt(sq_lp + 0.35 * (1.0 - lam) + 0.09)
        ratio2 = (r0_hb * r0_hb) / (r_hb * r_hb)
        ratio10 = ratio2.pow(5)
        ratio12 = ratio10 * ratio2
        # Softly cap short-range 12-10 wall so LJ owns Pauli repulsion cleanly
        v_12_10 = (5.0 * ratio12 - 6.0 * ratio10).clamp(min=-1.0, max=2.0)
        sw_hb = _smooth_switch_c2(dist_lp, 3.8, 4.8)
        pair_hbond = hb_strength * v_12_10 * ang_gate * sw_hb
        e_hbond = pair_hbond.sum()

        # ------------------------------------------------------------------
        # 5. Anisotropic Sigma-Hole Halogen Bonding (C-Cl/Br/I ... O/N/S)
        # ------------------------------------------------------------------
        pair_halogen = torch.zeros_like(dist_lp)
        if bool(lig.is_halogen.any()) and bool(rec.is_acceptor.any()):
            hal_axes = compute_halogen_axes_from_coords(
                lc, lig.halogen_bond_pairs, lig.halogen_sigma_axes
            )  # [NL, 3] unit C -> X
            xb_mask = lig.is_halogen.unsqueeze(1) & rec.is_acceptor.unsqueeze(0)
            # cos_alpha = u_{C->X} . u_{X->B} = -cos(theta_{C-X...B})
            # theta = 180 deg => cos_alpha = +1.0; theta = 90 deg => cos_alpha = 0.0
            cos_alpha = (hal_axes.unsqueeze(1) * u_l_to_p).sum(dim=-1).clamp(min=-1.0, max=1.0)
            crown = torch.relu(2.0 * cos_alpha - 1.0).pow(3)  # peaks at 180 deg, 0 for <= 120 deg
            belt = (1.0 - cos_alpha * cos_alpha).pow(2)  # sin^4(alpha), peaks at 90 deg

            # Element-specific equilibrium sigma-hole distance r0_xb (Cl: 3.20, Br: 3.30, I: 3.45)
            z_lig = lig.atomic_nums
            r0_x = torch.where(
                z_lig == 53,
                torch.full_like(lig.halogen_strength, 3.45),
                torch.where(
                    z_lig == 35,
                    torch.full_like(lig.halogen_strength, 3.30),
                    torch.full_like(lig.halogen_strength, 3.20),
                ),
            ).unsqueeze(1)

            rad_crown = torch.exp(-0.5 * ((dist_lp - r0_x) / 0.55).pow(2))
            rad_belt = torch.exp(-0.5 * ((dist_lp - (r0_x - 0.10)) / 0.65).pow(2))
            sw_xb = _smooth_switch_c2(dist_lp, 4.2, 5.2)

            v_xb = (
                lig.halogen_strength.unsqueeze(1)
                * (
                    -self.halogen_crown_kcal * crown * rad_crown
                    + self.halogen_belt_kcal * belt * rad_belt
                )
                * sw_xb
            )
            pair_halogen = torch.where(xb_mask, v_xb, torch.zeros_like(v_xb))

        e_halogen = pair_halogen.sum()

        # ------------------------------------------------------------------
        # 6. Cation-Pi & Pi-Pi Aromatic Quadrupole Interactions
        # ------------------------------------------------------------------
        lig_rc, lig_rn = compute_ring_geometry_from_coords(
            lc, lig.ring_atom_indices, lig.ring_atom_mask
        )
        rec_rc, rec_rn = compute_ring_geometry_from_coords(
            rc, rec.ring_atom_indices, rec.ring_atom_mask
        )

        e_cation_pi = lc.sum() * 0.0
        rec_atom_spec = pair_halogen.sum(dim=0)

        # 6a. Ligand cation -> Pocket aromatic ring
        if bool(lig.is_cationic.any()) and rec_rc.shape[0] > 0:
            cat_coords = lc[lig.is_cationic]  # [NC_l, 3]
            d_cr = cat_coords.unsqueeze(1) - rec_rc.unsqueeze(0)  # [NC_l, R_p, 3]
            r_cr = torch.sqrt((d_cr * d_cr).sum(dim=-1) + 1e-8)
            u_cr = d_cr / r_cr.unsqueeze(-1)
            cos2_phi = ((u_cr * rec_rn.unsqueeze(0)).sum(dim=-1)).pow(2)
            v_cp = (
                -self.cation_pi_kcal
                * cos2_phi.pow(2)
                * torch.exp(-0.5 * ((r_cr - 4.0) / 0.85).pow(2))
                * _smooth_switch_c2(r_cr, 5.2, 6.2)
            )
            e_cation_pi = e_cation_pi + v_cp.sum()
            # Attribute pocket ring energy onto first ring atom for per-residue rollup
            ring_first_atom = rec.ring_atom_indices[:, 0].clamp(min=0, max=np_atoms - 1)
            eye_p = torch.eye(np_atoms, dtype=torch.float32, device=dev)
            rec_atom_spec = rec_atom_spec + (eye_p[ring_first_atom].T @ v_cp.sum(dim=0))

        # 6b. Pocket cation -> Ligand aromatic ring
        if bool(rec.is_cationic.any()) and lig_rc.shape[0] > 0:
            p_cat_idx = torch.nonzero(rec.is_cationic, as_tuple=False).squeeze(-1)
            p_cat_coords = rc[p_cat_idx]  # [NC_p, 3]
            d_cr2 = p_cat_coords.unsqueeze(1) - lig_rc.unsqueeze(0)  # [NC_p, R_l, 3]
            r_cr2 = torch.sqrt((d_cr2 * d_cr2).sum(dim=-1) + 1e-8)
            u_cr2 = d_cr2 / r_cr2.unsqueeze(-1)
            cos2_phi2 = ((u_cr2 * lig_rn.unsqueeze(0)).sum(dim=-1)).pow(2)
            v_cp2 = (
                -self.cation_pi_kcal
                * cos2_phi2.pow(2)
                * torch.exp(-0.5 * ((r_cr2 - 4.0) / 0.85).pow(2))
                * _smooth_switch_c2(r_cr2, 5.2, 6.2)
            )
            e_cation_pi = e_cation_pi + v_cp2.sum()
            eye_p = torch.eye(np_atoms, dtype=torch.float32, device=dev)
            rec_atom_spec = rec_atom_spec + (eye_p[p_cat_idx].T @ v_cp2.sum(dim=1))

        # 6c. Pi-Pi Stacking (Hunter-Sanders parallel-displaced + T-shaped quadrupole)
        e_pi_stacking = lc.sum() * 0.0
        if lig_rc.shape[0] > 0 and rec_rc.shape[0] > 0:
            d_rr = lig_rc.unsqueeze(1) - rec_rc.unsqueeze(0)  # [R_l, R_p, 3]
            r_rr = torch.sqrt((d_rr * d_rr).sum(dim=-1) + 1e-8)
            u_rr = d_rr / r_rr.unsqueeze(-1)
            c12 = ((lig_rn.unsqueeze(1) * rec_rn.unsqueeze(0)).sum(dim=-1)).pow(2)
            p1 = ((lig_rn.unsqueeze(1) * u_rr).sum(dim=-1)).pow(2)
            p2 = ((rec_rn.unsqueeze(0) * u_rr).sum(dim=-1)).pow(2)
            p_bar = 0.5 * (p1 + p2)

            v_pd = (
                -self.pi_pd_kcal
                * c12
                * (1.35 * p_bar * (1.20 - 0.50 * p_bar)).clamp(min=0.0, max=1.2)
                * torch.exp(-0.5 * ((r_rr - 3.95) / 0.75).pow(2))
            )
            v_t = (
                -self.pi_t_kcal
                * (1.0 - c12)
                * (0.5 * (p1 + p2))
                * torch.exp(-0.5 * ((r_rr - 4.95) / 0.80).pow(2))
            )
            v_pipi = (v_pd + v_t) * _smooth_switch_c2(r_rr, 5.5, 6.6)
            e_pi_stacking = v_pipi.sum()
            ring_first_atom = rec.ring_atom_indices[:, 0].clamp(min=0, max=np_atoms - 1)
            eye_p = torch.eye(np_atoms, dtype=torch.float32, device=dev)
            rec_atom_spec = rec_atom_spec + (eye_p[ring_first_atom].T @ v_pipi.sum(dim=0))

        # ------------------------------------------------------------------
        # 7. Non-polar Hydrophobic SASA Burial (dG_np)
        # ------------------------------------------------------------------
        comp_sig = torch.cat([lig.vdw_sigma, rec.vdw_sigma], dim=0)
        comp_hyd = torch.cat([lig.is_hydrophobic, rec.is_hydrophobic], dim=0)
        sasa_comp = self._compute_sasa_exposed(comp_coords, comp_sig, comp_hyd)
        sasa_lig_apo = self._compute_sasa_exposed(lc, lig.vdw_sigma, lig.is_hydrophobic)
        sasa_rec_apo = self._compute_sasa_exposed(rc, rec.vdw_sigma, rec.is_hydrophobic)

        buried_lig = (sasa_lig_apo - sasa_comp[:nl]).clamp(min=0.0)
        buried_rec = (sasa_rec_apo - sasa_comp[nl:]).clamp(min=0.0)
        dg_hydrophobic = self.sasa_surface_tension * (buried_lig.sum() + buried_rec.sum())

        # Per-pocket-atom energy attribution for exact per-residue decomposition
        rec_elec_attr = pair_coulomb.sum(dim=0) + rec_self_desolv
        rec_vdw_attr = pair_vdw.sum(dim=0) + self.sasa_surface_tension * buried_rec
        rec_hb_attr = pair_hbond.sum(dim=0)
        rec_total_attr = rec_elec_attr + rec_vdw_attr + rec_hb_attr + rec_atom_spec

        return {
            "e_coulomb": e_coulomb,
            "dg_gb": dg_gb,
            "e_vdw": e_vdw,
            "e_hbond": e_hbond,
            "e_halogen": e_halogen,
            "e_cation_pi": e_cation_pi,
            "e_pi_stacking": e_pi_stacking,
            "dg_hydrophobic": dg_hydrophobic,
            "lig_born_r": lig_born_r,
            "rec_born_r": rec_born_r,
            "lig_efield": lig_efield,
            "rec_efield": rec_efield,
            "rec_atom_energy": rec_total_attr,
            "rec_atom_elec": rec_elec_attr,
            "rec_atom_vdw": rec_vdw_attr,
            "rec_atom_hb": rec_hb_attr,
            "rec_atom_spec": rec_atom_spec,
        }

    @staticmethod
    def compute_internal_restraint_energy(
        coords: Tensor,
        params: BiophysicalAtomParams,
        *,
        bond_k: float = 150.0,
        angle_k: float = 40.0,
    ) -> Tensor:
        """Differentiable harmonic covalent bond-length and bond-angle strain energy."""
        e_val = coords.sum() * 0.0
        if params.bond_pairs.shape[0] > 0 and bond_k > 0.0:
            bi = params.bond_pairs[:, 0]
            bj = params.bond_pairs[:, 1]
            diff = coords[bi] - coords[bj]
            dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
            e_val = e_val + 0.5 * float(bond_k) * ((dist - params.bond_eq_lengths).pow(2)).sum()

        if params.angle_triplets.shape[0] > 0 and angle_k > 0.0:
            ia = params.angle_triplets[:, 0]
            jc = params.angle_triplets[:, 1]
            ib = params.angle_triplets[:, 2]
            v1 = coords[ia] - coords[jc]
            v2 = coords[ib] - coords[jc]
            u1 = v1 * torch.rsqrt((v1 * v1).sum(dim=-1, keepdim=True) + 1e-8)
            u2 = v2 * torch.rsqrt((v2 * v2).sum(dim=-1, keepdim=True) + 1e-8)
            cos_cur = (u1 * u2).sum(dim=-1)
            e_val = e_val + 0.5 * float(angle_k) * ((cos_cur - params.angle_eq_cos).pow(2)).sum()
        return e_val

    def potential_energy(
        self,
        lig_coords: Tensor,
        rec_coords: Tensor,
        lig: BiophysicalAtomParams,
        rec: BiophysicalAtomParams,
        *,
        softcore_lambda: float = 1.0,
    ) -> Tensor:
        """Return differentiable scalar intermolecular potential ``U_total`` (``float32``)."""
        lc = lig_coords.to(dtype=torch.float32)
        rc = rec_coords.to(dtype=torch.float32)
        terms = self._compute_energy_terms(lc, rc, lig, rec, softcore_lambda=softcore_lambda)
        u_tot: Tensor = (
            terms["e_coulomb"]
            + terms["dg_gb"]
            + terms["e_vdw"]
            + terms["e_hbond"]
            + terms["e_halogen"]
            + terms["e_cation_pi"]
            + terms["e_pi_stacking"]
            + terms["dg_hydrophobic"]
        )
        return u_tot

    def evaluate_complex(
        self,
        lig: BiophysicalAtomParams,
        rec: BiophysicalAtomParams,
        *,
        lig_coords: Tensor | None = None,
        rec_coords: Tensor | None = None,
        softcore_lambda: float = 1.0,
        include_internal_strain: bool = False,
        create_graph: bool = False,
    ) -> ComplexEnergyBreakdown:
        """Evaluate full energy breakdown, exact autograd 3D forces, and electric fields.

        Computes exact analytical gradients ``F_i = -grad_{r_i} U_total`` alongside separated
        electrical forces ``F_i^{elec} = -grad_{r_i} (E_Coulomb + dG_GB)``, steric/vdW forces
        ``F_i^{vdW} = -grad_{r_i} E_vdW``, 3D electric field vectors ``E_i`` (``V/A``),
        and per-residue interaction energy decomposition ``dE_r``.
        """
        base_lc = (lig_coords if lig_coords is not None else lig.coords).to(dtype=torch.float32)
        base_rc = (rec_coords if rec_coords is not None else rec.coords).to(dtype=torch.float32)

        with torch.enable_grad():
            lc = base_lc.clone().detach().requires_grad_(True) if not create_graph else base_lc
            rc = base_rc.clone().detach().requires_grad_(True) if not create_graph else base_rc
            if not lc.requires_grad:
                lc = lc.detach().requires_grad_(True)
            if not rc.requires_grad:
                rc = rc.detach().requires_grad_(True)

            terms = self._compute_energy_terms(lc, rc, lig, rec, softcore_lambda=softcore_lambda)
            e_elec_tot: Tensor = terms["e_coulomb"] + terms["dg_gb"]
            e_vdw_tot: Tensor = terms["e_vdw"]
            e_strain = (
                self.compute_internal_restraint_energy(lc, lig)
                if include_internal_strain
                else lc.sum() * 0.0
            )
            u_total: Tensor = (
                e_elec_tot
                + e_vdw_tot
                + terms["e_hbond"]
                + terms["e_halogen"]
                + terms["e_cation_pi"]
                + terms["e_pi_stacking"]
                + terms["dg_hydrophobic"]
                + e_strain
            )

            g_elec_l, g_elec_r = torch.autograd.grad(
                e_elec_tot,
                (lc, rc),
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
            g_vdw_l, g_vdw_r = torch.autograd.grad(
                e_vdw_tot,
                (lc, rc),
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
            g_tot_l, g_tot_r = torch.autograd.grad(
                u_total,
                (lc, rc),
                retain_graph=create_graph,
                create_graph=create_graph,
                allow_unused=True,
            )

        f_tot_l = -(g_tot_l if g_tot_l is not None else torch.zeros_like(base_lc))
        f_tot_r = -(g_tot_r if g_tot_r is not None else torch.zeros_like(base_rc))
        f_elec_l = -(g_elec_l if g_elec_l is not None else torch.zeros_like(base_lc))
        f_elec_r = -(g_elec_r if g_elec_r is not None else torch.zeros_like(base_rc))
        f_vdw_l = -(g_vdw_l if g_vdw_l is not None else torch.zeros_like(base_lc))
        f_vdw_r = -(g_vdw_r if g_vdw_r is not None else torch.zeros_like(base_rc))

        if not create_graph:
            f_tot_l = f_tot_l.detach()
            f_tot_r = f_tot_r.detach()
            f_elec_l = f_elec_l.detach()
            f_elec_r = f_elec_r.detach()
            f_vdw_l = f_vdw_l.detach()
            f_vdw_r = f_vdw_r.detach()

        tds_rot = float(self.rotatable_bond_penalty * max(lig.rotatable_bonds, 0))
        u_val = float(u_total.detach().item())
        # Empirical MM/GBSA enthalpy damping (0.35 on raw molecular-mechanics enthalpy + T*dS_rot)
        # yields realistic experimental-scale kcal/mol binding free energies while preserving
        # repulsive clashes when u_val > 0.
        dg_bind = (0.35 * u_val + tds_rot) if u_val < 0.0 else (u_val + tds_rot)

        # Aggregate per-residue energy decomposition sorted by interaction energy
        rec_res_idx = rec.residue_indices.detach().cpu().numpy()
        atom_tot_np = terms["rec_atom_energy"].detach().cpu().numpy()
        atom_elec_np = terms["rec_atom_elec"].detach().cpu().numpy()
        atom_vdw_np = terms["rec_atom_vdw"].detach().cpu().numpy()
        atom_hb_np = terms["rec_atom_hb"].detach().cpu().numpy()
        atom_spec_np = terms["rec_atom_spec"].detach().cpu().numpy()

        raw_res_e: dict[str, float] = {}
        raw_res_comp: dict[str, dict[str, float]] = {}
        for r_i, label in enumerate(rec.residue_labels):
            mask = rec_res_idx == r_i
            if not np.any(mask):
                continue
            tot_r = float(np.sum(atom_tot_np[mask]))
            raw_res_e[label] = tot_r
            raw_res_comp[label] = {
                "electrostatic": float(np.sum(atom_elec_np[mask])),
                "vdw_hydrophobic": float(np.sum(atom_vdw_np[mask])),
                "hbond_salt": float(np.sum(atom_hb_np[mask])),
                "halogen_pi": float(np.sum(atom_spec_np[mask])),
                "total": tot_r,
            }

        sorted_res = dict(sorted(raw_res_e.items(), key=lambda kv: kv[1]))
        sorted_comp = {k: raw_res_comp[k] for k in sorted_res}

        return ComplexEnergyBreakdown(
            e_coulomb=float(terms["e_coulomb"].detach().item()),
            dg_gb_solvation=float(terms["dg_gb"].detach().item()),
            e_vdw=float(terms["e_vdw"].detach().item()),
            e_hbond=float(terms["e_hbond"].detach().item()),
            e_halogen=float(terms["e_halogen"].detach().item()),
            e_cation_pi=float(terms["e_cation_pi"].detach().item()),
            e_pi_stacking=float(terms["e_pi_stacking"].detach().item()),
            dg_hydrophobic=float(terms["dg_hydrophobic"].detach().item()),
            tds_rotational_penalty=tds_rot,
            e_internal_strain=float(e_strain.detach().item()),
            u_total=u_val,
            dg_bind_est=float(dg_bind),
            ligand_forces_total=f_tot_l.to(dtype=torch.float32),
            pocket_forces_total=f_tot_r.to(dtype=torch.float32),
            ligand_forces_elec=f_elec_l.to(dtype=torch.float32),
            pocket_forces_elec=f_elec_r.to(dtype=torch.float32),
            ligand_forces_vdw=f_vdw_l.to(dtype=torch.float32),
            pocket_forces_vdw=f_vdw_r.to(dtype=torch.float32),
            ligand_electric_field=terms["lig_efield"].detach().to(dtype=torch.float32),
            pocket_electric_field=terms["rec_efield"].detach().to(dtype=torch.float32),
            ligand_born_radii=terms["lig_born_r"].detach().to(dtype=torch.float32),
            pocket_born_radii=terms["rec_born_r"].detach().to(dtype=torch.float32),
            per_residue_energy=sorted_res,
            per_residue_components=sorted_comp,
            total_energy_tensor=u_total if create_graph else u_total.detach(),
        )

    def forward(
        self,
        lig: BiophysicalAtomParams,
        rec: BiophysicalAtomParams,
        *,
        lig_coords: Tensor | None = None,
        rec_coords: Tensor | None = None,
        softcore_lambda: float = 1.0,
    ) -> ComplexEnergyBreakdown:
        """PyTorch ``nn.Module`` forward entry point delegating to :meth:`evaluate_complex`."""
        return self.evaluate_complex(
            lig,
            rec,
            lig_coords=lig_coords,
            rec_coords=rec_coords,
            softcore_lambda=softcore_lambda,
            create_graph=torch.is_grad_enabled(),
        )


__all__ = [
    "COULOMB_KCAL_ANGSTROM",
    "KB_KCAL_MOL_K",
    "KCAL_MOL_E_TO_VOLT",
    "BiophysicalAtomParams",
    "BiophysicalHamiltonian",
    "ComplexEnergyBreakdown",
    "DielectricMode",
    "compute_donor_axes_from_coords",
    "compute_halogen_axes_from_coords",
    "compute_ring_geometry_from_coords",
    "parameterize_ligand_biophysics",
    "parameterize_pocket_biophysics",
]
