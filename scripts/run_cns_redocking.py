#!/usr/bin/env python3
"""Gate 1: Multi-Seed Unbiased Native Redocking & 3D PLIF Validation for Human CNS Targets.

Validates the AutoDock Vina v1.2.7 + Meeko + pure-Python receptor PDBQT + 3D PLIF pipeline
across all 12 curated *Homo sapiens* Tourette Syndrome and psychiatric CNS targets
(13 primary structures including both DRD1 inactive templates 9LLG & 9URL, plus 4 key
secondary conformations = 17 human CNS complexes total).

Protocol (100% unbiased 2D-to-3D redocking, zero crystal coordinate leakage):
1. Extract canonical isomeric SMILES (preserving tetrahedral & E/Z double-bond stereochemistry)
   from each native CCD-bonded crystal SDF, then discard all 3D crystal coordinates.
2. Apply deterministic physiological pH 7.4 protonation via RDKit SMARTS:
   - Protonate basic aliphatic primary/secondary/tertiary amines, amidines, and guanidines
     (avoiding unphysical diprotonation of 1,2-diamines / 1,4-piperazines <= 3 bonds apart
     and preferring the inductively less-withdrawn nitrogen in asymmetric piperazines).
   - Keep amides, sulfonamides, nitro groups, anilines, and aromatic heterocycles neutral.
   - Deprotonate strong acids (carboxylates, sulfonates, phosphonates, tetrazoles).
3. Regenerate unbiased 3D conformers strictly from 2D SMILES using RDKit ETKDGv3 + MMFF94s
   (UFF fallback). Cluster 48 unbiased conformers by per-fused-ring-system internal SO(3)
   Kabsch RMSD (>= 0.28 A, within 6.0 kcal/mol, up to 4 conformers) to cover rigid aliphatic
   ring puckers, tricyclic butterfly folds, and pyramidal [NH+;R] invertomers that AutoDock
   Vina holds rigid; pool and rank all returned poses strictly by Vina dG.
4. Dock with AutoDock Vina v1.2.7 (exhaustiveness=8, num_modes=9, box centered on native
   ligand centroid, cubic size min(26.0, max(20.0, native_span + 8.0)) A) across 3 independent
   random seeds (42, 137, 2026) using ProcessPoolExecutor(max_workers=4, cpu=2 per Vina).
5. Evaluate exact symmetry-corrected heavy-atom RMSD (align=False), Murcko core RMSD,
   Ligand Efficiency (LE = -dG / n_heavy), PoseBusters checks, 3D PLIF Tanimoto recovery,
   residue recovery rate, and canonical pharmacophore anchor recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import meeko
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from geneuclidean.evaluation.geometry3d import (
    _parse_vina_affinities,
    _resolve_vina_binary,
    check_pose_geometry,
    compute_ligand_box_center,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
)
from geneuclidean.evaluation.interactions3d import (
    InteractionProfile,
    compute_heavy_atom_rmsd,
    profile_3d_interactions,
)

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Pre-compiled SMARTS for Deterministic Physiological pH 7.4 Protonation
# ---------------------------------------------------------------------------

# Basic aliphatic primary/secondary/tertiary amines (excludes amides, sulfonamides,
# nitro, anilines/arylamines, imines, nitriles, and aromatic nitrogens).
_SMARTS_BASIC_ALIPHATIC_AMINE = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;+0;!$(NC=O);!$(NS=O);!$(N=O);!$(N[a]);!$(N=*);!$(N#*)]"
)
# Amidines & guanidines (protonate the sp2 imine nitrogen =N- to =[NH+]-)
_SMARTS_AMIDINE_GUANIDINE_IMINE = Chem.MolFromSmarts("[NX2;+0]=[CX3]([NX3])[#6,N]")
# Strong acids deprotonated at pH 7.4
_SMARTS_CARBOXYLIC_OH = Chem.MolFromSmarts("[OX2H1;+0][CX3]=O")
_SMARTS_SULFO_PHOSPHO_OH = Chem.MolFromSmarts("[OX2H1;+0][SX4,PX4](=O)=O")
_SMARTS_TETRAZOLE_NH = Chem.MolFromSmarts("[nH1;+0]1nnnn1")


@dataclass(frozen=True)
class ComplexSpec:
    """Curated *Homo sapiens* CNS protein-ligand complex for Gate 1 redocking."""

    target: str
    pdb_id: str
    ligand_code: str
    drug_name: str
    state: str
    resolution_angstrom: float
    tier: str  # "primary_core", "primary_alt", "primary_ext", "secondary"
    canonical_anchors: tuple[str, ...]
    anchor_label: str
    expected_anchor_kinds: tuple[str, ...] = ("salt_bridge", "hbond")
    notes: str = ""

    @property
    def stem(self) -> str:
        """Return filename stem inside ``clean_pockets/``."""
        return f"{self.pdb_id}_{self.target}_{self.ligand_code}"

    @property
    def complex_id(self) -> str:
        """Return compact ``TARGET:PDB`` identifier."""
        return f"{self.target}:{self.pdb_id}"


# All 17 verified *Homo sapiens* CNS complexes (13 primary across 12 targets + 4 secondary)
GATE1_COMPLEXES: tuple[ComplexSpec, ...] = (
    # --- Primary 12 Human CNS Targets (DRD1 includes both 9LLG & 9URL) ---
    ComplexSpec(
        target="DRD1",
        pdb_id="9LLG",
        ligand_code="A1EKL",
        drug_name="Flupentixol",
        state="Inactive",
        resolution_angstrom=2.77,
        tier="primary_core",
        canonical_anchors=("ASP103",),
        anchor_label="Asp103 (3.32)",
        notes="Primary inactive DRD1 antagonist template (cis-flupentixol).",
    ),
    ComplexSpec(
        target="DRD1",
        pdb_id="9URL",
        ligand_code="A1EP3",
        drug_name="SKF-83566",
        state="Inactive",
        resolution_angstrom=3.04,
        tier="primary_alt",
        canonical_anchors=("ASP103",),
        anchor_label="Asp103 (3.32)",
        notes="Inactive DRD1 benzazepine template (direct bromo-congener of ecopipam).",
    ),
    ComplexSpec(
        target="DRD2",
        pdb_id="6CM4",
        ligand_code="8NU",
        drug_name="Risperidone",
        state="Inactive",
        resolution_angstrom=2.87,
        tier="primary_core",
        canonical_anchors=("ASP114",),
        anchor_label="Asp114 (3.32)",
        notes="Primary inactive DRD2 anti-EPS counter-target template.",
    ),
    ComplexSpec(
        target="DRD3",
        pdb_id="3PBL",
        ligand_code="ETQ",
        drug_name="Eticlopride",
        state="Inactive",
        resolution_angstrom=2.89,
        tier="primary_core",
        canonical_anchors=("ASP110",),
        anchor_label="Asp110 (3.32)",
        notes="Primary inactive DRD3 reference complex.",
    ),
    ComplexSpec(
        target="VMAT2",
        pdb_id="8T69",
        ligand_code="YHL",
        drug_name="Tetrabenazine",
        state="Lumen-occluded",
        resolution_angstrom=2.89,
        tier="primary_core",
        canonical_anchors=("GLU312", "ASP399"),
        anchor_label="Glu312/Asp399",
        notes="Primary human VMAT2 inhibited complex (SLC18A2).",
    ),
    ComplexSpec(
        target="ADRA2A",
        pdb_id="7EJ8",
        ligand_code="J59",
        drug_name="Brimonidine",
        state="Active",
        resolution_angstrom=3.00,
        tier="primary_core",
        canonical_anchors=("ASP113", "ASP128"),
        anchor_label="Asp128 (3.32)",
        notes="Active Gi-coupled ADRA2A (+15 cryo-EM numbering: Asp128 = Asp113^3.32).",
    ),
    ComplexSpec(
        target="HRH3",
        pdb_id="7F61",
        ligand_code="1IB",
        drug_name="PF-03654746",
        state="Inactive",
        resolution_angstrom=2.60,
        tier="primary_core",
        canonical_anchors=("ASP114", "GLU206"),
        anchor_label="Asp114 (3.32)/Glu206",
        notes="Inactive human HRH3 antagonist complex.",
    ),
    ComplexSpec(
        target="PDE10A",
        pdb_id="5SJX",
        ligand_code="PF9",
        drug_name="Mardepodect (MP-10)",
        state="Catalytic (Q2)",
        resolution_angstrom=2.06,
        tier="primary_core",
        canonical_anchors=("GLN726", "TYR693"),
        anchor_label="Gln726/Tyr693",
        expected_anchor_kinds=("hbond",),
        notes="Verified Homo sapiens Q9Y233 MP-10 structure (excludes rat 3HR1); Zn/Mg > 6.6 A.",
    ),
    ComplexSpec(
        target="CHRM4",
        pdb_id="8FX5",
        ligand_code="XNO",
        drug_name="Xanomeline",
        state="Active",
        resolution_angstrom=2.45,
        tier="primary_core",
        canonical_anchors=("ASP112",),
        anchor_label="Asp112 (3.32)",
        notes="Active CHRM4 Cobenfy orthosteric complex (flexible hexyloxy aliphatic tail).",
    ),
    ComplexSpec(
        target="HTR2A",
        pdb_id="7WC8",
        ligand_code="92S",
        drug_name="Lumateperone",
        state="Inactive",
        resolution_angstrom=2.45,
        tier="primary_core",
        canonical_anchors=("ASP155",),
        anchor_label="Asp155 (3.32)",
        notes="Inactive human 5-HT2A complex with Caplyta.",
    ),
    ComplexSpec(
        target="SERT",
        pdb_id="5I6X",
        ligand_code="8PR",
        drug_name="Paroxetine",
        state="Outward-open",
        resolution_angstrom=3.14,
        tier="primary_core",
        canonical_anchors=("ASP98",),
        anchor_label="Asp98 (S1)",
        notes="Outward-open human SERT S1 orthosteric complex.",
    ),
    ComplexSpec(
        target="TAAR1",
        pdb_id="8W88",
        ligand_code="UJL",
        drug_name="Ulotaront",
        state="Active",
        resolution_angstrom=2.60,
        tier="primary_ext",
        canonical_anchors=("ASP103",),
        anchor_label="Asp103 (3.32)",
        notes="Extended target: active human TAAR1 Gs-coupled complex (compact 12-HA ligand).",
    ),
    ComplexSpec(
        target="DAT",
        pdb_id="9JKI",
        ligand_code="1XR",
        drug_name="Bupropion",
        state="Inhibited",
        resolution_angstrom=2.69,
        tier="primary_ext",
        canonical_anchors=("ASP79",),
        anchor_label="Asp79 (S1)",
        notes="Extended target: inhibited human DAT complex with bupropion.",
    ),
    # --- Key Secondary Conformations (4 additional = 17 total) ---
    ComplexSpec(
        target="DRD2",
        pdb_id="6LUQ",
        ligand_code="GMJ",
        drug_name="Haloperidol",
        state="Inactive",
        resolution_angstrom=3.10,
        tier="secondary",
        canonical_anchors=("ASP114",),
        anchor_label="Asp114 (3.32)",
        notes="Secondary DRD2 conformation with classical TS antipsychotic haloperidol.",
    ),
    ComplexSpec(
        target="VMAT2",
        pdb_id="9KQ8",
        ligand_code="XW7",
        drug_name="Valbenazine",
        state="Lumen-occluded",
        resolution_angstrom=3.38,
        tier="secondary",
        canonical_anchors=("GLU312", "ASP399"),
        anchor_label="Glu312/Asp399",
        notes="Secondary VMAT2 conformation with FDA-approved Ingrezza (flexible L-valine ester).",
    ),
    ComplexSpec(
        target="PDE10A",
        pdb_id="3WYM",
        ligand_code="3K9",
        drug_name="Balipodect (TAK-063)",
        state="Catalytic (Q2)",
        resolution_angstrom=2.00,
        tier="secondary",
        canonical_anchors=("GLN726", "TYR693", "GLN716", "TYR683"),
        anchor_label="Gln716/Tyr683 (-10 iso)",
        expected_anchor_kinds=("hbond",),
        notes="Secondary human PDE10A Q9Y233 (-10 isoform offset: Gln716=Gln726, Tyr683=Tyr693).",
    ),
    ComplexSpec(
        target="ADRA2A",
        pdb_id="9CBL",
        ligand_code="ALE",
        drug_name="L-Epinephrine",
        state="Active",
        resolution_angstrom=2.80,
        tier="secondary",
        canonical_anchors=("ASP113", "ASP128"),
        anchor_label="Asp128 (3.32)",
        notes="Secondary active ADRA2A conformation with endogenous catecholamine agonist.",
    ),
)


# ---------------------------------------------------------------------------
# Deterministic Physiological pH 7.4 Protonation & Stereochemistry Helpers
# ---------------------------------------------------------------------------


def extract_canonical_isomeric_smiles_from_3d(mol_3d: Chem.Mol) -> str:
    """Extract exact canonical isomeric SMILES preserving both R/S and E/Z from 3D coords."""
    mol_copy = Chem.Mol(mol_3d)
    for bond in mol_copy.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    if mol_copy.GetNumConformers() > 0:
        Chem.DetectBondStereochemistry(mol_copy)
        Chem.AssignStereochemistryFrom3D(mol_copy)
    Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
    return str(Chem.MolToSmiles(Chem.RemoveHs(mol_copy), canonical=True, isomericSmiles=True))


def protonate_ligand_ph74(mol: Chem.Mol, *, preserve_coords: bool = False) -> Chem.Mol:
    """Deterministically protonate/deprotonate an RDKit Mol for physiological pH 7.4.

    Rules grounded in physical organic chemistry:
    - Protonates amidine/guanidine sp2 imine nitrogens (``=N-`` -> ``=[NH+]-``).
    - Protonates aliphatic primary, secondary, and tertiary amines while keeping amides,
      sulfonamides, nitro groups, anilines, imines, nitriles, and aromatic N neutral.
    - Prevents unphysical diprotonation of close diamines / 1,4-piperazines (``<= 3``
      topological bonds apart, where electrostatic repulsion lowers ``pKa2 < 6``).
      In asymmetric piperazines (e.g., flupentixol), prefers the inductively more basic
      nitrogen with fewer nearby electron-withdrawing atoms (O, F, Cl, Br, S within 3 bonds).
    - Deprotonates strong acids (carboxylic, sulfonic, phosphonic acids, tetrazoles).
    """
    base = Chem.RemoveHs(Chem.Mol(mol))
    if not preserve_coords:
        base.RemoveAllConformers()
    rw = Chem.RWMol(base)
    n_atoms = rw.GetNumAtoms()
    dist_mat = Chem.GetDistanceMatrix(rw)

    protonated: set[int] = set()

    # 1. Amidines and guanidines (pKa ~ 7.8 - 13.5): protonate sp2 imine N
    if _SMARTS_AMIDINE_GUANIDINE_IMINE is not None:
        for match in rw.GetSubstructMatches(_SMARTS_AMIDINE_GUANIDINE_IMINE):
            n_idx = int(match[0])
            if n_idx not in protonated:
                rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
                protonated.add(n_idx)

    # 2. Aliphatic primary / secondary / tertiary amines (pKa ~ 7.5 - 10.5)
    cand_amines: list[int] = []
    if _SMARTS_BASIC_ALIPHATIC_AMINE is not None:
        cand_amines = [int(m[0]) for m in rw.GetSubstructMatches(_SMARTS_BASIC_ALIPHATIC_AMINE)]

    def _basicity_sort_key(atom_idx: int) -> tuple[int, int, int]:
        """Rank more basic amines first (fewer nearby inductive EWGs, higher alkylation)."""
        # Count electronegative inductive withdrawers (O, F, Cl, Br, I, S) at 2-3 bonds
        ewg_2 = 0
        ewg_3 = 0
        for j in range(n_atoms):
            if j == atom_idx:
                continue
            z_j = rw.GetAtomWithIdx(j).GetAtomicNum()
            if z_j in {8, 9, 16, 17, 35, 53}:
                d = int(round(float(dist_mat[atom_idx, j])))
                if d == 2:
                    ewg_2 += 1
                elif d == 3:
                    ewg_3 += 1
        atom = rw.GetAtomWithIdx(atom_idx)
        return (ewg_2 * 3 + ewg_3, atom.GetTotalNumHs(), atom_idx)

    for n_idx in sorted(set(cand_amines), key=_basicity_sort_key):
        # Skip if within 3 bonds of an already-protonated nitrogen (e.g., 1,4-piperazine)
        if any(int(round(float(dist_mat[n_idx, p]))) <= 3 for p in protonated):
            continue
        rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
        protonated.add(n_idx)

    # 3. Deprotonate strong acids (pKa < 5.0)
    for acid_smarts in (_SMARTS_CARBOXYLIC_OH, _SMARTS_SULFO_PHOSPHO_OH, _SMARTS_TETRAZOLE_NH):
        if acid_smarts is None:
            continue
        for match in rw.GetSubstructMatches(acid_smarts):
            a_idx = int(match[0])
            atom = rw.GetAtomWithIdx(a_idx)
            if atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(-1)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def verify_protonation_unit_tests() -> None:
    """Self-test deterministic pH 7.4 protonation helper on canonical functional groups."""
    test_cases: list[tuple[str, str, int]] = [
        ("aliphatic_tertiary_amine", "CCN(CC)CC", 1),
        ("aliphatic_secondary_amine", "CNC[C@@H]1OCCc2ccsc21", 1),
        ("amide_neutral", "CC(=O)NCC", 0),
        ("aniline_neutral", "Nc1ccccc1", 0),
        ("sulfonamide_neutral", "CS(=O)(=O)NC", 0),
        ("pyridine_neutral", "c1ccncc1", 0),
        ("guanidine_amidine_brimonidine", "Brc1c(NC2=NCCN2)ccc2nccnc12", 1),
        (
            "piperazine_monoprotonated_flupentixol",
            "OCCN1CCN(CC/C=C2/c3ccccc3Sc3ccc(C(F)(F)F)cc32)CC1",
            1,
        ),
        ("pde10a_mp10_neutral", "Cn1cc(-c2ccncc2)c(-c2ccc(OCc3ccc4ccccc4n3)cc2)n1", 0),
        (
            "lumateperone_only_aliphatic_N",
            "CN1CCN2c3c(cccc31)[C@@H]1CN(CCCC(=O)c3ccc(F)cc3)CC[C@@H]12",
            1,
        ),
        ("carboxylic_acid_anion", "CC(=O)O", -1),
    ]
    for name, smi, expected_net_charge in test_cases:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Failed to parse test SMILES for {name}"
        prot = protonate_ligand_ph74(mol)
        net_chg = Chem.GetFormalCharge(prot)
        assert net_chg == expected_net_charge, (
            f"pH 7.4 protonation unit test failed for {name}: "
            f"expected net charge {expected_net_charge}, got {net_chg} "
            f"({Chem.MolToSmiles(prot, isomericSmiles=True)})"
        )


# ---------------------------------------------------------------------------
# Unbiased 2D-to-3D Conformer Ensemble (Per-Fused-Ring-System SO(3) Diversity)
# ---------------------------------------------------------------------------


def _extract_fused_ring_systems(mol_h: Chem.Mol) -> list[list[int]]:
    """Group ring heavy atoms into fused/spiro ring systems with local stereo probes.

    For each independent fused/spiro ring system, includes:
    1. All heavy atoms belonging to that ring system.
    2. Explicit polar hydrogens attached to protonated ring nitrogens (``[NH+;R]``),
       separating axial vs. equatorial pyramidal ammonium invertomers.
    3. Immediate exocyclic heavy neighbors of non-aromatic ring atoms, separating
       chair/boat ring puckers and tricyclic butterfly folds (e.g., thioxanthene).
    """
    ring_info = mol_h.GetRingInfo()
    atom_rings = [set(r) for r in ring_info.AtomRings()]
    if not atom_rings:
        return []

    # Merge rings sharing >= 1 atom into fused/spiro ring systems
    systems: list[set[int]] = []
    for ring in atom_rings:
        merged = set(ring)
        remaining: list[set[int]] = []
        for sys_set in systems:
            if merged & sys_set:
                merged |= sys_set
            else:
                remaining.append(sys_set)
        remaining.append(merged)
        systems = remaining

    enriched: list[list[int]] = []
    for sys_set in systems:
        probe = set(sys_set)
        for idx in list(sys_set):
            atom = mol_h.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 1:
                        probe.add(nbr.GetIdx())
            if not atom.GetIsAromatic():
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in sys_set:
                        probe.add(nbr.GetIdx())
        if len(probe) >= 4:
            enriched.append(sorted(probe))
    return enriched


def _so3_kabsch_rmsd(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Proper SO(3) (reflection-free) Kabsch RMSD between two ordered 3D point sets."""
    a_c = pts_a - pts_a.mean(axis=0, keepdims=True)
    b_c = pts_b - pts_b.mean(axis=0, keepdims=True)
    cov = a_c.T @ b_c
    u_mat, _, vt_mat = np.linalg.svd(cov)
    det_sign = 1.0 if float(np.linalg.det(vt_mat.T @ u_mat.T)) >= 0.0 else -1.0
    rot = vt_mat.T @ np.diag(np.array([1.0, 1.0, det_sign], dtype=np.float64)) @ u_mat.T
    diff = (a_c @ rot.T) - b_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _max_ring_system_rmsd(
    mol_h: Chem.Mol,
    cid_a: int,
    cid_b: int,
    ring_systems: list[list[int]],
) -> float:
    """Maximum proper-SO(3) internal Kabsch RMSD across independent fused ring systems.

    Decouples rigid ring pucker / butterfly fold / pyramidal [NH+] invertomer differences
    from acyclic rotatable-bond torsions (which AutoDock Vina samples freely).
    """
    conf_a = mol_h.GetConformer(cid_a)
    conf_b = mol_h.GetConformer(cid_b)
    max_rmsd = 0.0
    for idxs in ring_systems:
        pts_a = np.asarray([list(conf_a.GetAtomPosition(i)) for i in idxs], dtype=np.float64)
        pts_b = np.asarray([list(conf_b.GetAtomPosition(i)) for i in idxs], dtype=np.float64)
        max_rmsd = max(max_rmsd, _so3_kabsch_rmsd(pts_a, pts_b))
    return max_rmsd


def generate_unbiased_3d_conformers(
    smiles_ph74: str,
    *,
    seed: int,
    num_confs: int = 48,
    max_ring_confs: int = 4,
    ring_rmsd_cutoff: float = 0.28,
    energy_window_kcal: float = 6.0,
) -> list[tuple[Chem.Mol, float, str]]:
    """Generate unbiased 3D conformers strictly from 2D SMILES via ETKDGv3 + MMFF94s.

    AutoDock Vina samples acyclic torsions freely while holding aliphatic rings, tricyclic
    butterfly folds, and pyramidal protonated tertiary amines ([NH+;R]) rigid. Starting
    purely from 2D SMILES, we embed ``num_confs`` conformers, minimize all with MMFF94s
    (UFF fallback), and select up to ``max_ring_confs`` lowest-energy conformers whose
    internal fused-ring-system SO(3) Kabsch RMSD differs by ``>= ring_rmsd_cutoff`` within
    ``energy_window_kcal`` of the MMFF94s minimum. Planar aromatic scaffolds yield a
    single conformer automatically; flexible aliphatic/tricyclic scaffolds cover their
    genuine low-energy ring puckers and [NH+] invertomers without crystal leakage.
    """
    mol_2d = Chem.MolFromSmiles(smiles_ph74)
    if mol_2d is None:
        return []
    mol_h = Chem.AddHs(mol_2d)

    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    params.numThreads = 1

    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))  # type: ignore[attr-defined]
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))  # type: ignore[attr-defined]
    if not cids:
        return []

    energies: list[tuple[int, float]] = []
    if AllChem.MMFFHasAllMoleculeParams(mol_h):  # type: ignore[attr-defined]
        opt_res = AllChem.MMFFOptimizeMoleculeConfs(  # type: ignore[attr-defined]
            mol_h, numThreads=1, maxIters=400, mmffVariant="MMFF94s"
        )
        for cid, (_conv, energy) in zip(cids, opt_res, strict=True):
            energies.append((int(cid), float(energy)))
    elif AllChem.UFFHasAllMoleculeParams(mol_h):  # type: ignore[attr-defined]
        opt_res = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=1, maxIters=400)  # type: ignore[attr-defined]
        for cid, (_conv, energy) in zip(cids, opt_res, strict=True):
            energies.append((int(cid), float(energy)))
    else:
        energies = [(int(cid), 0.0) for cid in cids]

    energies.sort(key=lambda item: item[1])
    min_energy = energies[0][1]
    ring_systems = _extract_fused_ring_systems(mol_h)

    chosen: list[tuple[int, float, str]] = [(energies[0][0], energies[0][1], "ring_conf_0")]
    if ring_systems and max_ring_confs > 1:
        for cid, energy in energies[1:]:
            if energy - min_energy > energy_window_kcal:
                break
            if all(
                _max_ring_system_rmsd(mol_h, cid, prev_cid, ring_systems) >= ring_rmsd_cutoff
                for prev_cid, _, _ in chosen
            ):
                chosen.append((cid, energy, f"ring_conf_{len(chosen)}"))
                if len(chosen) >= max_ring_confs:
                    break

    results: list[tuple[Chem.Mol, float, str]] = []
    for cid, energy, tag in chosen:
        single = Chem.Mol(mol_h)
        single.RemoveAllConformers()
        single.AddConformer(mol_h.GetConformer(cid), assignId=True)
        results.append((single, energy, tag))
    return results


# ---------------------------------------------------------------------------
# Multi-Mode Vina Execution & Exact 3D Coordinate Reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_all_vina_modes(
    out_pdbqt_text: str,
    affinities: list[float],
    template_mol_prot: Chem.Mol,
) -> list[tuple[float, Chem.Mol]]:
    """Reconstruct ALL docked Vina modes into individual 3D RDKit Mol objects.

    Uses Meeko ``RDKitMolCreate.from_pdbqt_mol`` (preserving exact SMILES bond orders,
    stereochemistry, formal charges, and docked coordinates). Falls back to heavy-atom
    PDBQT coordinate transfer onto ``template_mol_prot`` if Meeko reconstruction fails.
    """
    modes: list[tuple[float, Chem.Mol]] = []
    try:
        pmol = meeko.PDBQTMolecule(out_pdbqt_text, is_dlg=False, skip_typing=True)
        rdmols = meeko.RDKitMolCreate.from_pdbqt_mol(pmol)
        if rdmols and rdmols[0] is not None and rdmols[0].GetNumConformers() > 0:
            multi = rdmols[0]
            n_modes = min(len(affinities), multi.GetNumConformers())
            for cid in range(n_modes):
                m_one = Chem.Mol(multi)
                m_one.RemoveAllConformers()
                m_one.AddConformer(multi.GetConformer(cid), assignId=True)
                modes.append((float(affinities[cid]), m_one))
            if modes:
                return modes
    except Exception:
        pass

    # Fallback: parse MODEL blocks and verify atom count against template
    models_text = out_pdbqt_text.split("ENDMDL")
    n_expected = Chem.RemoveHs(template_mol_prot).GetNumAtoms()
    for idx, block in enumerate(models_text):
        if idx >= len(affinities):
            break
        pdb_lines = [
            line[:66] + "\n" for line in block.splitlines() if line.startswith(("ATOM  ", "HETATM"))
        ]
        if not pdb_lines:
            continue
        raw_mol = Chem.MolFromPDBBlock("".join(pdb_lines), removeHs=True, sanitize=False)
        if raw_mol is None or raw_mol.GetNumAtoms() != n_expected:
            continue
        try:
            mapped = AllChem.AssignBondOrdersFromTemplate(  # type: ignore[attr-defined]
                Chem.RemoveHs(template_mol_prot), raw_mol
            )
            modes.append((float(affinities[idx]), mapped))
        except Exception:
            continue
    return modes


def _run_vina_single_conformer(
    mol_3d: Chem.Mol,
    template_mol_prot: Chem.Mol,
    rec_pdbqt: Path,
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    *,
    vina_exe: Path,
    exhaustiveness: int,
    num_modes: int,
    cpu: int,
    seed: int,
    timeout: int = 180,
) -> list[tuple[float, Chem.Mol]]:
    """Prepare ligand PDBQT via Meeko, execute Vina v1.2.7, and return all reconstructed modes."""
    lig_pdbqt = prepare_ligand_pdbqt(mol_3d)
    if lig_pdbqt is None:
        return []

    with tempfile.TemporaryDirectory(prefix="geneuc_g1_") as tmp_dir:
        tmp = Path(tmp_dir)
        lig_path = tmp / "ligand.pdbqt"
        out_path = tmp / "docked.pdbqt"
        lig_path.write_text(lig_pdbqt, encoding="utf-8")

        cmd = [
            str(vina_exe),
            "--receptor",
            str(rec_pdbqt),
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
            "--cpu",
            str(cpu),
            "--seed",
            str(seed),
            "--out",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode != 0 or not out_path.exists():
            return []
        out_text = out_path.read_text(encoding="utf-8", errors="ignore")
        affinities = _parse_vina_affinities(out_text, proc.stdout)
        if not affinities:
            return []
        return _reconstruct_all_vina_modes(out_text, affinities, template_mol_prot)


def _deduplicate_ranked_modes(
    modes: list[tuple[float, Chem.Mol, str]],
    *,
    rmsd_dedup_tol: float = 0.5,
    max_keep: int = 9,
) -> list[tuple[float, Chem.Mol, str]]:
    """Rank pooled unbiased modes strictly by Vina dG and deduplicate near-identical poses."""
    sorted_modes = sorted(modes, key=lambda item: item[0])
    kept: list[tuple[float, Chem.Mol, str]] = []
    for aff, mol, tag in sorted_modes:
        is_dup = False
        for _, prev_mol, _ in kept:
            try:
                if compute_heavy_atom_rmsd(prev_mol, mol, align=False) < rmsd_dedup_tol:
                    is_dup = True
                    break
            except Exception:
                continue
        if not is_dup:
            kept.append((aff, mol, tag))
            if len(kept) >= max_keep:
                break
    return kept


# ---------------------------------------------------------------------------
# Murcko Core RMSD & Canonical Pharmacophore Anchor Analysis
# ---------------------------------------------------------------------------


def compute_murcko_core_rmsd(mol_ref: Chem.Mol, mol_probe: Chem.Mol) -> float | None:
    """Compute symmetry-corrected in-place heavy-atom RMSD over the Murcko ring core.

    Diagnostic metric distinguishing global binding-mode success from flexible aliphatic
    tail drift (e.g., xanomeline's hexyloxy chain in CHRM4 8FX5 or valbenazine's ester).
    """
    ref_noh = Chem.RemoveHs(Chem.Mol(mol_ref))
    probe_noh = Chem.RemoveHs(Chem.Mol(mol_probe))
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(ref_noh)
    except Exception:
        return None
    if scaffold is None or scaffold.GetNumHeavyAtoms() < 5:
        return None
    if scaffold.GetNumHeavyAtoms() >= ref_noh.GetNumHeavyAtoms():
        return None

    ref_matches = ref_noh.GetSubstructMatches(scaffold, uniquify=False, maxMatches=512)
    probe_matches = probe_noh.GetSubstructMatches(scaffold, uniquify=False, maxMatches=512)
    if not ref_matches or not probe_matches:
        return None

    ref_coords = np.asarray(ref_noh.GetConformer().GetPositions(), dtype=np.float64)
    probe_coords = np.asarray(probe_noh.GetConformer().GetPositions(), dtype=np.float64)
    ref_core = ref_coords[list(ref_matches[0])]

    best_rmsd = float("inf")
    for p_match in probe_matches:
        diff = probe_coords[list(p_match)] - ref_core
        rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
    return float(best_rmsd) if math.isfinite(best_rmsd) else None


def check_canonical_anchor(
    profile: InteractionProfile,
    spec: ComplexSpec,
) -> dict[str, Any]:
    """Check recovery of canonical pharmacophore anchor residue(s) in a 3D PLIF."""
    anchor_set = {a.upper() for a in spec.canonical_anchors}
    allowed_kinds = set(spec.expected_anchor_kinds)

    polar_hits: list[dict[str, Any]] = []
    any_hits: list[dict[str, Any]] = []
    for rec in profile.interactions:
        res_base = rec.protein_residue.split(":")[0].upper()
        if res_base in anchor_set:
            hit = {
                "residue": rec.protein_residue,
                "kind": rec.kind,
                "protein_atom": rec.protein_atom,
                "distance_angstrom": round(float(rec.distance_angstrom), 3),
            }
            any_hits.append(hit)
            if rec.kind in allowed_kinds:
                polar_hits.append(hit)

    polar_hits.sort(key=lambda h: (0 if h["kind"] == "salt_bridge" else 1, h["distance_angstrom"]))
    min_dist = polar_hits[0]["distance_angstrom"] if polar_hits else None
    return {
        "recovered": len(polar_hits) > 0,
        "recovered_any_contact": len(any_hits) > 0,
        "min_distance_angstrom": min_dist,
        "hits": polar_hits,
    }


# ---------------------------------------------------------------------------
# Per-Complex Multi-Seed Redocking Worker
# ---------------------------------------------------------------------------


def evaluate_complex_redocking(
    spec: ComplexSpec,
    clean_pockets_dir: Path,
    vina_bin: Path,
    seeds: tuple[int, ...],
    exhaustiveness: int,
    num_modes: int,
    cpu_per_vina: int,
) -> dict[str, Any]:
    """Run full 3-seed unbiased redocking, PoseBusters, and 3D PLIF audit for one complex."""
    t_start = time.perf_counter()
    pocket_pdb = clean_pockets_dir / f"{spec.stem}_pocket.pdb"
    lig_sdf = clean_pockets_dir / f"{spec.stem}_lig.sdf"

    if not pocket_pdb.exists() or not lig_sdf.exists():
        raise FileNotFoundError(f"Missing clean pocket or SDF for {spec.stem}")

    suppl = Chem.SDMolSupplier(str(lig_sdf), removeHs=False, sanitize=True)
    native_mol = suppl[0] if len(suppl) > 0 else None
    if native_mol is None or native_mol.GetNumConformers() == 0:
        raise ValueError(f"Failed to load valid 3D crystal ligand from {lig_sdf}")

    # 1. Canonical isomeric SMILES (strips all 3D coords) & deterministic pH 7.4 protonation
    canonical_smiles = extract_canonical_isomeric_smiles_from_3d(native_mol)
    mol_2d = Chem.MolFromSmiles(canonical_smiles)
    assert mol_2d is not None, f"Round-trip SMILES failed for {spec.stem}: {canonical_smiles}"
    mol_prot_2d = protonate_ligand_ph74(mol_2d, preserve_coords=False)
    smiles_ph74 = str(Chem.MolToSmiles(mol_prot_2d, canonical=True, isomericSmiles=True))

    # Apples-to-apples native pH 7.4 crystal reference (exact crystal heavy-atom 3D coords)
    native_mol_ph74 = protonate_ligand_ph74(native_mol, preserve_coords=True)
    n_heavy = int(native_mol_ph74.GetNumHeavyAtoms())
    n_rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol_prot_2d))
    mw = float(Descriptors.MolWt(mol_2d))
    clogp = float(Descriptors.MolLogP(mol_2d))
    net_charge_ph74 = int(Chem.GetFormalCharge(mol_prot_2d))

    # 2. Compute docking box centered on native ligand heavy-atom centroid
    box_center = compute_ligand_box_center(lig_sdf)
    nat_conf = native_mol.GetConformer(0)
    heavy_coords = np.asarray(
        [
            list(nat_conf.GetAtomPosition(i))
            for i in range(native_mol.GetNumAtoms())
            if native_mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ],
        dtype=np.float64,
    )
    axis_spans = heavy_coords.max(axis=0) - heavy_coords.min(axis=0)
    native_span = float(np.max(axis_spans))
    box_edge = float(min(26.0, max(20.0, native_span + 8.0)))
    box_size = (box_edge, box_edge, box_edge)

    # 3. Profile native crystal complex 3D PLIF & PoseBusters baseline
    native_plif = profile_3d_interactions(pocket_pdb, native_mol_ph74)
    native_anchor = check_canonical_anchor(native_plif, spec)
    native_pb = check_pose_geometry(native_mol_ph74, pocket_pdb_path=pocket_pdb)

    # 4. Prepare receptor PDBQT once per complex and run all seeds
    seed_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"geneuc_rec_{spec.pdb_id}_") as rec_tmp:
        rec_pdbqt = prepare_receptor_pdbqt(pocket_pdb, Path(rec_tmp) / f"{spec.pdb_id}.pdbqt")
        if rec_pdbqt is None:
            raise RuntimeError(f"Receptor PDBQT preparation failed for {pocket_pdb}")

        for seed in seeds:
            t_seed = time.perf_counter()
            conformers = generate_unbiased_3d_conformers(smiles_ph74, seed=seed)
            pooled_modes: list[tuple[float, Chem.Mol, str]] = []
            for conf_mol, _mmff_e, conf_tag in conformers:
                conf_modes = _run_vina_single_conformer(
                    conf_mol,
                    mol_prot_2d,
                    rec_pdbqt,
                    box_center,
                    box_size,
                    vina_exe=vina_bin,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    cpu=cpu_per_vina,
                    seed=seed,
                )
                for aff, m_dock in conf_modes:
                    pooled_modes.append((aff, m_dock, conf_tag))

            ranked = _deduplicate_ranked_modes(pooled_modes, rmsd_dedup_tol=0.5, max_keep=num_modes)
            if not ranked:
                seed_records.append(
                    {
                        "seed": seed,
                        "success": False,
                        "error": "No valid docked poses returned",
                        "elapsed_sec": round(time.perf_counter() - t_seed, 2),
                    }
                )
                continue

            mode_summaries: list[dict[str, Any]] = []
            for mode_rank, (aff, m_dock, conf_tag) in enumerate(ranked, start=1):
                rmsd = compute_heavy_atom_rmsd(native_mol_ph74, m_dock, align=False)
                core_rmsd = compute_murcko_core_rmsd(native_mol_ph74, m_dock)
                mode_summaries.append(
                    {
                        "mode_rank": mode_rank,
                        "vina_affinity": round(float(aff), 3),
                        "le": round(float(-aff / max(n_heavy, 1)), 4),
                        "rmsd_sym": round(float(rmsd), 3),
                        "core_rmsd": round(float(core_rmsd), 3) if core_rmsd is not None else None,
                        "conformer_source": conf_tag,
                    }
                )

            # Top-1 pose evaluation
            top1_aff, top1_mol, top1_tag = ranked[0]
            top1_rmsd = float(mode_summaries[0]["rmsd_sym"])
            top1_core = mode_summaries[0]["core_rmsd"]
            top1_pb = check_pose_geometry(top1_mol, pocket_pdb_path=pocket_pdb)
            top1_plif = profile_3d_interactions(pocket_pdb, top1_mol)
            top1_anchor = check_canonical_anchor(top1_plif, spec)

            # Best-of-Top-3 pose evaluation (lowest heavy-atom RMSD among top 3 Vina modes)
            top3_slice = mode_summaries[: min(3, len(mode_summaries))]
            best3_idx = int(min(range(len(top3_slice)), key=lambda i: top3_slice[i]["rmsd_sym"]))
            best3_aff, best3_mol, _best3_tag = ranked[best3_idx]
            best3_rmsd = float(top3_slice[best3_idx]["rmsd_sym"])
            best3_core = top3_slice[best3_idx]["core_rmsd"]
            if best3_idx == 0:
                best3_pb = top1_pb
                best3_plif = top1_plif
                best3_anchor = top1_anchor
            else:
                best3_pb = check_pose_geometry(best3_mol, pocket_pdb_path=pocket_pdb)
                best3_plif = profile_3d_interactions(pocket_pdb, best3_mol)
                best3_anchor = check_canonical_anchor(best3_plif, spec)

            # Best-of-All-9 diagnostic
            best_all_idx = int(
                min(range(len(mode_summaries)), key=lambda i: mode_summaries[i]["rmsd_sym"])
            )
            best_all_rmsd = float(mode_summaries[best_all_idx]["rmsd_sym"])

            seed_records.append(
                {
                    "seed": seed,
                    "success": True,
                    "n_unbiased_conformers_docked": len(conformers),
                    "n_modes_parsed": len(ranked),
                    "elapsed_sec": round(time.perf_counter() - t_seed, 2),
                    "top1": {
                        "vina_affinity": round(float(top1_aff), 3),
                        "le": round(float(-top1_aff / max(n_heavy, 1)), 4),
                        "rmsd_sym": top1_rmsd,
                        "core_rmsd": top1_core,
                        "conformer_source": top1_tag,
                        "posebusters_passed": bool(top1_pb.passed_all),
                        "protein_clash_count": int(top1_pb.protein_clash_count),
                        "internal_clash_count": int(top1_pb.internal_clash_count),
                        "plif_tanimoto": round(
                            top1_plif.tanimoto_to(native_plif, include_hydrophobic=True), 4
                        ),
                        "plif_tanimoto_polar": round(
                            top1_plif.tanimoto_to(native_plif, include_hydrophobic=False), 4
                        ),
                        "residue_recovery_rate": round(
                            top1_plif.residue_recovery_rate(native_plif, include_hydrophobic=True),
                            4,
                        ),
                        "anchor": top1_anchor,
                        "interaction_counts": dict(top1_plif.counts_by_type),
                    },
                    "best_of_3": {
                        "mode_rank": best3_idx + 1,
                        "vina_affinity": round(float(best3_aff), 3),
                        "le": round(float(-best3_aff / max(n_heavy, 1)), 4),
                        "rmsd_sym": best3_rmsd,
                        "core_rmsd": best3_core,
                        "posebusters_passed": bool(best3_pb.passed_all),
                        "protein_clash_count": int(best3_pb.protein_clash_count),
                        "plif_tanimoto": round(
                            best3_plif.tanimoto_to(native_plif, include_hydrophobic=True), 4
                        ),
                        "plif_tanimoto_polar": round(
                            best3_plif.tanimoto_to(native_plif, include_hydrophobic=False), 4
                        ),
                        "residue_recovery_rate": round(
                            best3_plif.residue_recovery_rate(native_plif, include_hydrophobic=True),
                            4,
                        ),
                        "anchor": best3_anchor,
                        "interaction_counts": dict(best3_plif.counts_by_type),
                    },
                    "best_of_all_rmsd": best_all_rmsd,
                    "best_of_all_mode_rank": best_all_idx + 1,
                    "modes": mode_summaries,
                }
            )

    # 5. Aggregate across all 3 seeds
    ok = [r for r in seed_records if r.get("success")]
    if not ok:
        raise RuntimeError(f"All redocking seeds failed for {spec.complex_id}")

    def _agg(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 3),
            "std": round(float(np.std(arr, ddof=0)), 3),
            "median": round(float(np.median(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
        }

    top1_rmsds = [float(r["top1"]["rmsd_sym"]) for r in ok]
    best3_rmsds = [float(r["best_of_3"]["rmsd_sym"]) for r in ok]
    bestall_rmsds = [float(r["best_of_all_rmsd"]) for r in ok]
    top1_affs = [float(r["top1"]["vina_affinity"]) for r in ok]
    top1_les = [float(r["top1"]["le"]) for r in ok]
    top1_plifs = [float(r["top1"]["plif_tanimoto"]) for r in ok]
    best3_plifs = [float(r["best_of_3"]["plif_tanimoto"]) for r in ok]
    top1_res_rec = [float(r["top1"]["residue_recovery_rate"]) for r in ok]
    best3_res_rec = [float(r["best_of_3"]["residue_recovery_rate"]) for r in ok]
    top1_pb_rate = float(np.mean([1.0 if r["top1"]["posebusters_passed"] else 0.0 for r in ok]))
    best3_pb_rate = float(
        np.mean([1.0 if r["best_of_3"]["posebusters_passed"] else 0.0 for r in ok])
    )
    top1_anchor_rate = float(
        np.mean([1.0 if r["top1"]["anchor"]["recovered"] else 0.0 for r in ok])
    )
    best3_anchor_rate = float(
        np.mean([1.0 if r["best_of_3"]["anchor"]["recovered"] else 0.0 for r in ok])
    )

    top1_cores = [float(r["top1"]["core_rmsd"]) for r in ok if r["top1"]["core_rmsd"] is not None]
    best3_cores = [
        float(r["best_of_3"]["core_rmsd"]) for r in ok if r["best_of_3"]["core_rmsd"] is not None
    ]

    # Diagnostic classification for Gate 1 transparency
    med_top1 = float(np.median(top1_rmsds))
    med_best3 = float(np.median(best3_rmsds))
    diag_flags: list[str] = []
    if med_best3 <= 2.0 and med_top1 <= 2.0:
        status = "PASS_TOP1"
    elif med_best3 <= 2.0:
        status = "PASS_TOP3"
        diag_flags.append(
            f"Top-1 median {med_top1:.2f} A vs Best-of-3 median {med_best3:.2f} A "
            "(near-degenerate scoring rank; native pose recovered in Top-3)"
        )
    elif med_best3 <= 2.5:
        status = "BORDERLINE"
        diag_flags.append(f"Best-of-3 median {med_best3:.2f} A (within 2.0-2.5 A tolerance)")
    else:
        status = "DIAGNOSTIC_WARN"
        core_msg = (
            f"; Murcko core RMSD={float(np.median(best3_cores)):.2f} A" if best3_cores else ""
        )
        diag_flags.append(
            f"Exceeds 2.5 A (Top-1 median={med_top1:.2f} A, Best-of-3 median={med_best3:.2f} A"
            f"{core_msg}; RotB={n_rot})"
        )

    if top1_anchor_rate < 1.0:
        diag_flags.append(
            f"Top-1 anchor recovery {top1_anchor_rate * 100:.0f}% "
            f"(Best-of-3: {best3_anchor_rate * 100:.0f}%)"
        )
    if top1_pb_rate < 1.0:
        diag_flags.append(f"Top-1 PoseBusters pass rate {top1_pb_rate * 100:.0f}%")

    return {
        "complex_id": spec.complex_id,
        "target": spec.target,
        "pdb_id": spec.pdb_id,
        "ligand_code": spec.ligand_code,
        "drug_name": spec.drug_name,
        "state": spec.state,
        "resolution_angstrom": spec.resolution_angstrom,
        "tier": spec.tier,
        "canonical_anchors": list(spec.canonical_anchors),
        "anchor_label": spec.anchor_label,
        "notes": spec.notes,
        "ligand_properties": {
            "canonical_smiles": canonical_smiles,
            "smiles_ph74": smiles_ph74,
            "n_heavy_atoms": n_heavy,
            "n_rotatable_bonds": n_rot,
            "mw": round(mw, 2),
            "clogp": round(clogp, 2),
            "net_charge_ph74": net_charge_ph74,
        },
        "docking_box": {
            "center": [round(c, 3) for c in box_center],
            "size": [round(s, 3) for s in box_size],
            "native_span_angstrom": round(native_span, 3),
        },
        "native_reference": {
            "posebusters_passed": bool(native_pb.passed_all),
            "protein_clash_count": int(native_pb.protein_clash_count),
            "interaction_counts": dict(native_plif.counts_by_type),
            "key_residues": list(native_plif.key_residues[:10]),
            "anchor": native_anchor,
        },
        "summary": {
            "status": status,
            "n_seeds_succeeded": len(ok),
            "top1_rmsd": _agg(top1_rmsds),
            "best3_rmsd": _agg(best3_rmsds),
            "best_all_rmsd": _agg(bestall_rmsds),
            "top1_core_rmsd": _agg(top1_cores) if top1_cores else None,
            "best3_core_rmsd": _agg(best3_cores) if best3_cores else None,
            "top1_vina_affinity": _agg(top1_affs),
            "top1_le": _agg(top1_les),
            "top1_plif_tanimoto": _agg(top1_plifs),
            "best3_plif_tanimoto": _agg(best3_plifs),
            "top1_residue_recovery": _agg(top1_res_rec),
            "best3_residue_recovery": _agg(best3_res_rec),
            "top1_posebusters_pass_rate": round(top1_pb_rate, 4),
            "best3_posebusters_pass_rate": round(best3_pb_rate, 4),
            "top1_anchor_recovery_rate": round(top1_anchor_rate, 4),
            "best3_anchor_recovery_rate": round(best3_anchor_rate, 4),
            "diagnostics": diag_flags,
        },
        "seeds": seed_records,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }


def _add_receptor_polar_hs_pdb(in_pdb: Path, out_pdb: Path) -> Path:
    """Add ideal polar hydrogens to protein N/O/S atoms (keeping catalytic metals intact)."""
    raw_lines = in_pdb.read_text(encoding="utf-8", errors="ignore").splitlines()
    het_lines = [line for line in raw_lines if line.startswith("HETATM")]
    atom_block = (
        "\n".join(line for line in raw_lines if line.startswith(("ATOM  ", "TER"))) + "\nEND\n"
    )
    mol = Chem.MolFromPDBBlock(atom_block, sanitize=False, removeHs=False)
    if mol is None:
        out_pdb.write_text(in_pdb.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return out_pdb
    mol.UpdatePropertyCache(strict=False)
    polar_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in {7, 8, 16}]
    mol_h = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=polar_idxs)
    pdb_h = Chem.MolToPDBBlock(mol_h)
    out_lines = [line for line in pdb_h.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    out_lines.extend(het_lines)
    out_lines.append("END")
    out_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_pdb


def run_sensitivity_diagnostic(
    spec: ComplexSpec,
    clean_pockets_dir: Path,
    vina_bin: Path,
    *,
    exhaustiveness: int,
    num_modes: int,
    cpu_per_vina: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Gate 1 Diagnostic Transparency probe for any complex exceeding 2.5 A Best-of-3 RMSD.

    Checks (1) crystal-pose ``--local_only`` Vina energy (detects rigid cryo-EM steric
    squeezes), (2) tight per-axis orthosteric box ``clip(axis_span + 6.0, 14.0, 24.0)`` A
    (detects cut-pocket exterior lipid-groove escape on small fragments), and (3) receptor
    polar-H addition (detects acceptor-only ligands requiring receptor H-bond donor ``HD``).
    Never replaces canonical standard-box Gate 1 results.
    """
    pocket_pdb = clean_pockets_dir / f"{spec.stem}_pocket.pdb"
    lig_sdf = clean_pockets_dir / f"{spec.stem}_lig.sdf"
    native_mol = Chem.SDMolSupplier(str(lig_sdf), removeHs=False, sanitize=True)[0]
    canonical_smiles = extract_canonical_isomeric_smiles_from_3d(native_mol)
    mol_prot_2d = protonate_ligand_ph74(Chem.MolFromSmiles(canonical_smiles), preserve_coords=False)
    smiles_ph74 = str(Chem.MolToSmiles(mol_prot_2d, canonical=True, isomericSmiles=True))
    native_mol_ph74 = protonate_ligand_ph74(native_mol, preserve_coords=True)

    box_center = compute_ligand_box_center(lig_sdf)
    nat_conf = native_mol.GetConformer(0)
    heavy_coords = np.asarray(
        [
            list(nat_conf.GetAtomPosition(i))
            for i in range(native_mol.GetNumAtoms())
            if native_mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ],
        dtype=np.float64,
    )
    ax_span = heavy_coords.max(axis=0) - heavy_coords.min(axis=0)
    std_edge = float(min(26.0, max(20.0, float(np.max(ax_span)) + 8.0)))
    std_box = (std_edge, std_edge, std_edge)
    tight_box = tuple(float(x) for x in np.clip(ax_span + 6.0, 14.0, 24.0))

    conformers = generate_unbiased_3d_conformers(smiles_ph74, seed=seed)

    diag: dict[str, Any] = {
        "seed": seed,
        "tight_box_size": [round(x, 2) for x in tight_box],
    }

    with tempfile.TemporaryDirectory(prefix=f"geneuc_sens_{spec.pdb_id}_") as td_str:
        td = Path(td_str)
        rec_std = prepare_receptor_pdbqt(pocket_pdb, td / "rec_std.pdbqt")
        pdb_h = _add_receptor_polar_hs_pdb(pocket_pdb, td / "rec_h.pdb")
        rec_h = prepare_receptor_pdbqt(pdb_h, td / "rec_h.pdbqt")
        assert rec_std is not None and rec_h is not None

        # 1. Crystal pose local_only Vina affinity (measures rigid-receptor steric fit)
        nat_h = Chem.AddHs(native_mol_ph74, addCoords=True)
        nat_pdbqt = prepare_ligand_pdbqt(nat_h)
        if nat_pdbqt:
            lp_nat = td / "native.pdbqt"
            op_loc = td / "native_loc.pdbqt"
            lp_nat.write_text(nat_pdbqt, encoding="utf-8")
            proc_loc = subprocess.run(
                [
                    str(vina_bin),
                    "--receptor",
                    str(rec_std),
                    "--ligand",
                    str(lp_nat),
                    "--center_x",
                    f"{box_center[0]:.3f}",
                    "--center_y",
                    f"{box_center[1]:.3f}",
                    "--center_z",
                    f"{box_center[2]:.3f}",
                    "--size_x",
                    f"{std_edge:.2f}",
                    "--size_y",
                    f"{std_edge:.2f}",
                    "--size_z",
                    f"{std_edge:.2f}",
                    "--local_only",
                    "--out",
                    str(op_loc),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            loc_aff = None
            for line in proc_loc.stdout.splitlines():
                if "Estimated Free Energy of Binding" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        loc_aff = float(parts[1].strip().split()[0])
                        break
            diag["crystal_local_only_dg"] = round(loc_aff, 3) if loc_aff is not None else None

        # 2. Tight per-axis orthosteric box probe (standard receptor)
        # 3. Receptor polar-H probe (standard cubic box)
        for probe_name, rec_path, box_use in (
            ("tight_box", rec_std, tight_box),
            ("receptor_polar_h", rec_h, std_box),
        ):
            pooled: list[tuple[float, Chem.Mol, str]] = []
            for cmol, _, ctag in conformers:
                c_modes = _run_vina_single_conformer(
                    cmol,
                    mol_prot_2d,
                    rec_path,
                    box_center,
                    box_use,
                    vina_exe=vina_bin,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    cpu=cpu_per_vina,
                    seed=seed,
                )
                for aff, md in c_modes:
                    pooled.append((aff, md, ctag))
            dedup = _deduplicate_ranked_modes(pooled, rmsd_dedup_tol=0.5, max_keep=num_modes)
            if not dedup:
                diag[probe_name] = {"success": False}
                continue
            rmsds = [compute_heavy_atom_rmsd(native_mol_ph74, m, align=False) for _, m, _ in dedup]
            b3_i = int(min(range(min(3, len(rmsds))), key=lambda i: rmsds[i]))
            t1_anc = check_canonical_anchor(profile_3d_interactions(pocket_pdb, dedup[0][1]), spec)
            b3_anc = check_canonical_anchor(
                profile_3d_interactions(pocket_pdb, dedup[b3_i][1]), spec
            )
            diag[probe_name] = {
                "success": True,
                "top1_vina_dg": round(float(dedup[0][0]), 3),
                "top1_rmsd": round(float(rmsds[0]), 3),
                "best3_rmsd": round(float(rmsds[b3_i]), 3),
                "best3_mode_rank": b3_i + 1,
                "top1_anchor_recovered": bool(t1_anc["recovered"]),
                "best3_anchor_recovered": bool(b3_anc["recovered"]),
            }

    return diag


# ---------------------------------------------------------------------------
# Cohort Rollup & Publication Table 1 Markdown Generation
# ---------------------------------------------------------------------------


def _summarize_cohort(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Compute Gate 1 aggregate statistics over a cohort of complexes."""
    n = len(records)
    top1_meds = [float(r["summary"]["top1_rmsd"]["median"]) for r in records]
    top1_means = [float(r["summary"]["top1_rmsd"]["mean"]) for r in records]
    best3_meds = [float(r["summary"]["best3_rmsd"]["median"]) for r in records]
    best3_means = [float(r["summary"]["best3_rmsd"]["mean"]) for r in records]
    vina_meds = [float(r["summary"]["top1_vina_affinity"]["median"]) for r in records]
    le_meds = [float(r["summary"]["top1_le"]["median"]) for r in records]
    plif_top1 = [float(r["summary"]["top1_plif_tanimoto"]["mean"]) for r in records]
    plif_best3 = [float(r["summary"]["best3_plif_tanimoto"]["mean"]) for r in records]
    res_top1 = [float(r["summary"]["top1_residue_recovery"]["mean"]) for r in records]
    res_best3 = [float(r["summary"]["best3_residue_recovery"]["mean"]) for r in records]
    pb_top1 = [float(r["summary"]["top1_posebusters_pass_rate"]) for r in records]
    pb_best3 = [float(r["summary"]["best3_posebusters_pass_rate"]) for r in records]
    anc_top1 = [float(r["summary"]["top1_anchor_recovery_rate"]) for r in records]
    anc_best3 = [float(r["summary"]["best3_anchor_recovery_rate"]) for r in records]

    return {
        "cohort": label,
        "n_complexes": n,
        "median_top1_rmsd_angstrom": round(float(np.median(top1_meds)), 3),
        "mean_top1_rmsd_angstrom": round(float(np.mean(top1_means)), 3),
        "median_best3_rmsd_angstrom": round(float(np.median(best3_meds)), 3),
        "mean_best3_rmsd_angstrom": round(float(np.mean(best3_means)), 3),
        "pct_top1_le_2A": round(100.0 * float(np.mean([x <= 2.0 for x in top1_meds])), 1),
        "pct_best3_le_2A": round(100.0 * float(np.mean([x <= 2.0 for x in best3_meds])), 1),
        "pct_best3_le_2_5A": round(100.0 * float(np.mean([x <= 2.5 for x in best3_meds])), 1),
        "median_vina_dg_kcal_mol": round(float(np.median(vina_meds)), 3),
        "median_ligand_efficiency": round(float(np.median(le_meds)), 4),
        "mean_top1_plif_tanimoto": round(float(np.mean(plif_top1)), 4),
        "mean_best3_plif_tanimoto": round(float(np.mean(plif_best3)), 4),
        "mean_top1_residue_recovery": round(float(np.mean(res_top1)), 4),
        "mean_best3_residue_recovery": round(float(np.mean(res_best3)), 4),
        "mean_top1_posebusters_pass_pct": round(100.0 * float(np.mean(pb_top1)), 1),
        "mean_best3_posebusters_pass_pct": round(100.0 * float(np.mean(pb_best3)), 1),
        "mean_top1_anchor_recovery_pct": round(100.0 * float(np.mean(anc_top1)), 1),
        "mean_best3_anchor_recovery_pct": round(100.0 * float(np.mean(anc_best3)), 1),
        "n_complexes_exceeding_2_5A_best3": int(sum(1 for x in best3_meds if x > 2.5)),
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render publication-ready Gate 1 Table 1 and transparent diagnostic commentary."""
    meta = payload["protocol"]
    cohorts = payload["cohort_summaries"]
    results = payload["complexes"]

    core = cohorts["core_10_primary"]
    prim13 = cohorts["primary_13_all_12_targets"]
    all17 = cohorts["all_17_human_complexes"]

    rollup_hdr = (
        "| Cohort | N | Median Top-1 RMSD (Å) | Median Best-of-3 RMSD (Å) | "
        "Best-of-3 <= 2.0 Å | Median Vina ΔG (kcal/mol) | Median LE | "
        "Mean Top-1 / Best-3 PLIF Tanimoto | Mean Top-1 / Best-3 Res. Rec. | "
        "Anchor Recovery (Top-1 / Best-3) | PoseBusters Pass (Top-1 / Best-3) |"
    )
    rollup_sep = (
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |"
    )
    detail_hdr = (
        "| Tier | Target | PDB (Res) | Native Ligand (HA, RotB, q7.4) | Box (Å) | "
        "Vina ΔG (kcal/mol) | LE | Top-1 RMSD Med [Range] (Mean±SD, Å) | "
        "Best-of-3 RMSD Med [Range] (Mean±SD, Å) | "
        "PLIF Tan. (T1 / B3) | Res. Rec. (T1 / B3) | Canonical Anchor (T1 / B3) | "
        "PB Pass (T1) | Gate Status |"
    )
    detail_sep = (
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | "
        ":---: | :---: | :--- | ---: | :--- |"
    )

    lines: list[str] = [
        "# Table 1 — Gate 1 Multi-Seed Native Redocking & 3D PLIF Validation "
        "(17 *Homo sapiens* CNS Complexes)",
        "",
        "## Protocol Summary",
        f"- **Docking Engine:** `{meta['vina_version']}` "
        f"(`exhaustiveness={meta['exhaustiveness']}`, `num_modes={meta['num_modes']}`, "
        f"`seeds={meta['seeds']}`, `cpu={meta['cpu_per_vina']}`/run)",
        "- **Unbiased 2D-to-3D Protocol:** Canonical isomeric SMILES (crystal 3D coordinates "
        "stripped) -> deterministic physiological pH 7.4 protonation -> RDKit `ETKDGv3` + "
        "`MMFF94s` conformer generation (48 conformers clustered by per-fused-ring-system "
        "internal SO(3) Kabsch RMSD >= 0.28 Å, up to 4 low-energy ring/butterfly/`[NH+;R]` "
        "conformers, pooled and ranked strictly by Vina $\\Delta G$) -> Meeko PDBQT -> "
        "AutoDock Vina.",
        "- **Box Definition:** Centered on native crystal ligand heavy-atom centroid; cubic edge "
        "`min(26.0, max(20.0, native_span + 8.0))` Å.",
        "- **Species Integrity:** **17 / 17 (100%) *Homo sapiens*** cleaned receptors (`PDE10A` "
        "uses human `5SJX`/`3WYM`, excluding rat `3HR1`; catalytic `Zn2+`/`Mg2+` sit `> 5.6 Å` "
        "from ligand).",
        "",
        "## Gate 1 Aggregate Rollup",
        "",
        rollup_hdr,
        rollup_sep,
    ]

    for c in (core, prim13, all17):
        anc_t1 = c["mean_top1_anchor_recovery_pct"]
        anc_b3 = c["mean_best3_anchor_recovery_pct"]
        pb_t1 = c["mean_top1_posebusters_pass_pct"]
        pb_b3 = c["mean_best3_posebusters_pass_pct"]
        lines.append(
            f"| **{c['cohort']}** | {c['n_complexes']} | "
            f"**{c['median_top1_rmsd_angstrom']:.2f}** | "
            f"**{c['median_best3_rmsd_angstrom']:.2f}** | "
            f"**{c['pct_best3_le_2A']:.1f}%** | {c['median_vina_dg_kcal_mol']:.2f} | "
            f"{c['median_ligand_efficiency']:.3f} | "
            f"{c['mean_top1_plif_tanimoto']:.3f} / {c['mean_best3_plif_tanimoto']:.3f} | "
            f"{c['mean_top1_residue_recovery']:.3f} / {c['mean_best3_residue_recovery']:.3f} | "
            f"**{anc_t1:.1f}% / {anc_b3:.1f}%** | {pb_t1:.1f}% / {pb_b3:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Per-Complex Gate 1 Validation Table "
            "(3-Seed Median [Range] & Mean±SD: Seeds 42, 137, 2026)",
            "",
            detail_hdr,
            detail_sep,
        ]
    )

    tier_labels = {
        "primary_core": "Core",
        "primary_alt": "Core-Alt",
        "primary_ext": "Ext",
        "secondary": "Secondary",
    }

    for r in results:
        s = r["summary"]
        lp = r["ligand_properties"]
        t_lbl = tier_labels.get(r["tier"], r["tier"])
        chg = f"{lp['net_charge_ph74']:+d}" if lp["net_charge_ph74"] != 0 else "0"
        t1 = s["top1_rmsd"]
        b3 = s["best3_rmsd"]
        t1_rmsd_str = (
            f"**{t1['median']:.2f}** [{t1['min']:.2f}–{t1['max']:.2f}] "
            f"({t1['mean']:.2f}±{t1['std']:.2f})"
        )
        b3_rmsd_str = (
            f"**{b3['median']:.2f}** [{b3['min']:.2f}–{b3['max']:.2f}] "
            f"({b3['mean']:.2f}±{b3['std']:.2f})"
        )
        vina_str = f"{s['top1_vina_affinity']['mean']:.2f} ± {s['top1_vina_affinity']['std']:.2f}"
        le_str = f"{s['top1_le']['mean']:.3f}"
        plif_str = f"{s['top1_plif_tanimoto']['mean']:.3f} / {s['best3_plif_tanimoto']['mean']:.3f}"
        res_str = (
            f"{s['top1_residue_recovery']['mean']:.3f} / {s['best3_residue_recovery']['mean']:.3f}"
        )
        anc_str = (
            f"{r['anchor_label']} "
            f"({s['top1_anchor_recovery_rate'] * 100:.0f}% / "
            f"{s['best3_anchor_recovery_rate'] * 100:.0f}%)"
        )
        pb_str = f"{s['top1_posebusters_pass_rate'] * 100:.0f}%"
        lines.append(
            f"| {t_lbl} | **{r['target']}** | `{r['pdb_id']}` ({r['resolution_angstrom']:.2f} Å) | "
            f"{r['drug_name']} ({lp['n_heavy_atoms']}, {lp['n_rotatable_bonds']}, {chg}) | "
            f"{r['docking_box']['size'][0]:.1f} | {vina_str} | {le_str} | "
            f"{t1_rmsd_str} | {b3_rmsd_str} | {plif_str} | {res_str} | {anc_str} | "
            f"{pb_str} | `{s['status']}` |"
        )

    # Diagnostic transparency section (explicitly report every complex > 2.0 A or flagged)
    flagged = [r for r in results if r["summary"]["diagnostics"]]
    lines.extend(["", "## Diagnostic Transparency & Pockets Requiring Care in Iteration 3", ""])
    if not flagged:
        lines.append(
            "All 17 complexes passed Top-1 RMSD <= 2.0 Å with 100% canonical anchor "
            "and PoseBusters recovery."
        )
    else:
        for r in flagged:
            s = r["summary"]
            core_note = ""
            if s["best3_core_rmsd"] is not None:
                c_t1 = s["top1_core_rmsd"]["median"]
                c_b3 = s["best3_core_rmsd"]["median"]
                core_note = f" | Murcko Core RMSD (T1/B3): `{c_t1:.2f} Å / {c_b3:.2f} Å`"
            sens = r.get("sensitivity_diagnostic")
            sens_note = ""
            if isinstance(sens, dict):
                parts: list[str] = []
                if sens.get("crystal_local_only_dg") is not None:
                    parts.append(f"crystal local-only ΔG={sens['crystal_local_only_dg']:.2f}")
                tb = sens.get("tight_box", {})
                if tb.get("success"):
                    parts.append(
                        f"tight-box {sens['tight_box_size']} Best-3={tb['best3_rmsd']:.2f} Å "
                        f"(Anc={tb['best3_anchor_recovered']})"
                    )
                rh = sens.get("receptor_polar_h", {})
                if rh.get("success"):
                    parts.append(
                        f"receptor-polar-H Best-3={rh['best3_rmsd']:.2f} Å "
                        f"(Anc={rh['best3_anchor_recovered']})"
                    )
                if parts:
                    sens_note = " | Sensitivity probe (seed 42): " + "; ".join(parts)
            lines.append(
                f"- **`{r['complex_id']}` ({r['drug_name']}, `{s['status']}`):** "
                f"{'; '.join(s['diagnostics'])}{core_note}. "
                f"Best-of-9 median RMSD = `{s['best_all_rmsd']['median']:.2f} Å`"
                f"{sens_note}. Context: {r['notes']}"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Gate 1 CNS redocking."""
    parser = argparse.ArgumentParser(
        description="Gate 1 Multi-Seed Unbiased Native Redocking & 3D PLIF Validation."
    )
    parser.add_argument(
        "--clean-pockets-dir",
        type=Path,
        default=Path("data/cns_neuro.nosync/structures/clean_pockets"),
        help="Directory containing cleaned *_pocket.pdb and *_lig.sdf files.",
    )
    parser.add_argument(
        "--vina-bin",
        type=Path,
        default=Path("bin/vina"),
        help="Path to AutoDock Vina v1.2.7 executable.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 137, 2026],
        help="Independent random seeds per complex.",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
        help="AutoDock Vina search exhaustiveness.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=9,
        help="Maximum binding modes requested from AutoDock Vina.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel ProcessPoolExecutor workers (leaves CPU headroom for concurrent training).",
    )
    parser.add_argument(
        "--cpu-per-vina",
        type=int,
        default=2,
        help="CPU threads per Vina subprocess.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/tables/cns_gate1_redocking.json"),
        help="Output path for structured JSON Gate 1 results.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/tables/cns_gate1_redocking.md"),
        help="Output path for Markdown Table 1 summary.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute Gate 1 multi-seed redocking across all 17 human CNS complexes."""
    args = parse_args()
    t0 = time.perf_counter()

    # 1. Run deterministic pH 7.4 protonation self-tests before docking
    verify_protonation_unit_tests()

    vina_exe = _resolve_vina_binary(args.vina_bin)
    if vina_exe is None:
        msg = f"AutoDock Vina binary not found or not executable: {args.vina_bin}"
        raise FileNotFoundError(msg)

    ver_proc = subprocess.run([str(vina_exe), "--version"], capture_output=True, text=True)
    vina_version = (ver_proc.stdout or ver_proc.stderr).strip() or "AutoDock Vina"
    seeds = tuple(int(s) for s in args.seeds)

    print(
        f"[Gate 1] Starting unbiased {len(seeds)}-seed redocking on {len(GATE1_COMPLEXES)} "
        f"Homo sapiens CNS complexes ({vina_version}, exh={args.exhaustiveness}, "
        f"workers={args.workers}, cpu/vina={args.cpu_per_vina})...",
        flush=True,
    )

    results_by_id: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        future_map = {
            pool.submit(
                evaluate_complex_redocking,
                spec,
                args.clean_pockets_dir,
                vina_exe,
                seeds,
                int(args.exhaustiveness),
                int(args.num_modes),
                int(args.cpu_per_vina),
            ): spec
            for spec in GATE1_COMPLEXES
        }
        for future in as_completed(future_map):
            spec = future_map[future]
            res = future.result()
            results_by_id[spec.complex_id] = res
            s = res["summary"]
            anc_t1 = s["top1_anchor_recovery_rate"] * 100.0
            anc_b3 = s["best3_anchor_recovery_rate"] * 100.0
            print(
                f"  [{len(results_by_id):2d}/{len(GATE1_COMPLEXES)}] {spec.complex_id:12s} "
                f"({spec.drug_name:20s}) "
                f"dG={s['top1_vina_affinity']['mean']:6.2f} "
                f"LE={s['top1_le']['mean']:.3f} "
                f"Top1={s['top1_rmsd']['mean']:4.2f}±{s['top1_rmsd']['std']:4.2f}A "
                f"Best3={s['best3_rmsd']['mean']:4.2f}±{s['best3_rmsd']['std']:4.2f}A "
                f"PLIF={s['top1_plif_tanimoto']['mean']:.3f} "
                f"Anc={anc_t1:3.0f}%/{anc_b3:3.0f}% "
                f"[{s['status']}] ({res['elapsed_sec']:.1f}s)",
                flush=True,
            )

    # Preserve canonical Gate 1 ordering
    ordered_results = [results_by_id[spec.complex_id] for spec in GATE1_COMPLEXES]

    # Run transparent one-time sensitivity diagnostics on any complex with Best-of-3 > 2.5 A
    spec_by_id = {spec.complex_id: spec for spec in GATE1_COMPLEXES}
    warn_recs = [r for r in ordered_results if float(r["summary"]["best3_rmsd"]["median"]) > 2.5]
    if warn_recs:
        print(
            f"\n[Gate 1] Running transparent sensitivity diagnostics on {len(warn_recs)} "
            f"complexes with Best-of-3 median RMSD > 2.5 A...",
            flush=True,
        )
        for rec in warn_recs:
            spec = spec_by_id[rec["complex_id"]]
            sens = run_sensitivity_diagnostic(
                spec,
                args.clean_pockets_dir,
                vina_exe,
                exhaustiveness=int(args.exhaustiveness),
                num_modes=int(args.num_modes),
                cpu_per_vina=int(args.cpu_per_vina),
                seed=seeds[0],
            )
            rec["sensitivity_diagnostic"] = sens
            tb_r = sens.get("tight_box", {}).get("best3_rmsd")
            rh_r = sens.get("receptor_polar_h", {}).get("best3_rmsd")
            loc_dg = sens.get("crystal_local_only_dg")
            print(
                f"  [Diag] {spec.complex_id:12s} local_only_dG={loc_dg} "
                f"tight_box_B3={tb_r}A rec_polarH_B3={rh_r}A",
                flush=True,
            )

    core_10 = [r for r in ordered_results if r["tier"] == "primary_core"]
    prim_13 = [r for r in ordered_results if r["tier"] != "secondary"]

    cohort_summaries = {
        "core_10_primary": _summarize_cohort(core_10, "Core 10 Primary Targets"),
        "primary_13_all_12_targets": _summarize_cohort(
            prim_13, "Primary 13 (All 12 Targets incl. DRD1 9URL)"
        ),
        "all_17_human_complexes": _summarize_cohort(
            ordered_results, "All 17 Human CNS Complexes (Primary + Secondary)"
        ),
    }

    # Gate 1 quantitative acceptance check (publishable_science_plan.md §2 Gate 1)
    core_sum = cohort_summaries["core_10_primary"]
    drd1_9llg = results_by_id["DRD1:9LLG"]["summary"]
    drd2_6cm4 = results_by_id["DRD2:6CM4"]["summary"]
    gate1_checks = {
        "core_median_top1_rmsd_le_2_0A": core_sum["median_top1_rmsd_angstrom"] <= 2.0,
        "core_best3_le_2_0A_ge_75pct": core_sum["pct_best3_le_2A"] >= 75.0,
        "core_anchor_recovery_ge_80pct": (
            max(
                core_sum["mean_top1_anchor_recovery_pct"],
                core_sum["mean_best3_anchor_recovery_pct"],
            )
            >= 80.0
        ),
        "drd1_9llg_asp103_100pct": drd1_9llg["best3_anchor_recovery_rate"] == 1.0,
        "drd2_6cm4_asp114_100pct": drd2_6cm4["best3_anchor_recovery_rate"] == 1.0,
    }
    gate1_checks["all_gates_passed"] = all(gate1_checks.values())

    payload: dict[str, Any] = {
        "protocol": {
            "vina_version": vina_version,
            "exhaustiveness": int(args.exhaustiveness),
            "num_modes": int(args.num_modes),
            "seeds": list(seeds),
            "cpu_per_vina": int(args.cpu_per_vina),
            "workers": int(args.workers),
            "protonation": "deterministic_rdkit_smarts_ph74",
            "conformer_generation": (
                "unbiased_2d_smiles_etkdgv3_mmff94s_48confs_per_ring_system_so3_rmsd_0.28A_max4"
            ),
            "box_rule": "centroid_centered_cubic_min(26.0,max(20.0,native_span+8.0))",
            "species": "Homo sapiens (100% verified; rat 3HR1 excluded)",
            "total_wall_sec": round(time.perf_counter() - t0, 2),
        },
        "gate1_acceptance_checks": gate1_checks,
        "cohort_summaries": cohort_summaries,
        "complexes": ordered_results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_text = render_markdown_report(payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")

    print(f"\n[Gate 1] Finished in {payload['protocol']['total_wall_sec']:.1f}s")
    print(f"  JSON saved: {args.out_json}")
    print(f"  MD   saved: {args.out_md}")
    for key, c in cohort_summaries.items():
        print(
            f"  {key}: Top1_med={c['median_top1_rmsd_angstrom']:.2f}A, "
            f"Best3_med={c['median_best3_rmsd_angstrom']:.2f}A "
            f"(<=2A: {c['pct_best3_le_2A']:.1f}%), "
            f"dG_med={c['median_vina_dg_kcal_mol']:.2f}, "
            f"LE_med={c['median_ligand_efficiency']:.3f}, "
            f"PLIF={c['mean_top1_plif_tanimoto']:.3f}/{c['mean_best3_plif_tanimoto']:.3f}, "
            f"Anc={c['mean_top1_anchor_recovery_pct']:.1f}%/{c['mean_best3_anchor_recovery_pct']:.1f}%"
        )
    print("  Gate 1 checks:", gate1_checks)


if __name__ == "__main__":
    main()
