#!/usr/bin/env python3
"""Iteration 3: Independent 3D Physics, 10x10 Cross-Target Selectivity & DRD1 vs DRD2 Study.

Executes the held-out AutoDock Vina v1.2.7 + PoseBusters + 3D PLIF validation suite for
GenEuclidean v3.1 across all 12 curated *Homo sapiens* Tourette Syndrome & psychiatric CNS
targets (plus dual DRD2:6LUQ conformational cross-validation = 13 pockets total), strictly
following the Gate 1 scientific sign-off directives (#156):

1. Universal Tight Per-Axis Orthosteric Box:
   ``box_axis_i = clamp(native_ligand_span_i + 6.0, 14.0, 22.0)`` A centered on the native
   crystal ligand heavy-atom centroid across ALL pockets (eliminates exterior lipid-groove /
   adjacent allosteric-cleft escape while guaranteeing matched orthosteric volumes).
2. Primary *Homo sapiens* Templates:
   Core-10 (DRD1:9LLG, DRD2:6CM4, DRD3:3PBL, VMAT2:8T69, ADRA2A:7EJ8, HRH3:7F61,
   PDE10A:5SJX, CHRM4:8FX5, HTR2A:7WC8, SERT:5I6X) + Extended-2 (TAAR1:8W88, DAT:9JKI) +
   Dual DRD2 Cross-Validation (DRD2:6LUQ).
3. Dual Reporting (Unconstrained Top-1 AND Canonical-Anchor-Verified Modes) across:
   (a) On-Target Physics Audit (all 7 arms x 12 targets = 840 candidates + native/clinical)
   (b) 10x10 Cross-Target Selectivity Matrix (50 Iter-2 hits x 10 core pockets + 10k perms)
   (c) Flagship Tourette DRD1 (9LLG) vs Off-Target DRD2 (6CM4 & 6LUQ) Selectivity Study
   (d) Top-12 Prioritized Candidate Dossiers + Saved 3D Docked SDFs for Studio 3D gallery.

All designed molecules are strictly *in silico* research hypotheses requiring experimental
synthesis, functional pharmacology, and preclinical ADMET/safety validation.
"""

from __future__ import annotations

import argparse
import hashlib
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
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
from scipy import stats

from geneuclidean.cns.admet import CNSAdmetSuite
from geneuclidean.cns.mpo import compute_cns_mpo
from geneuclidean.cns.targets import CNS_TARGET_REGISTRY
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

# Thermodynamic conversion RT * ln(10) at 298.15 K (kcal/mol per log10 Kd unit)
_RT_LN10 = 1.364

# Pre-compiled SMARTS for deterministic physiological pH 7.4 protonation (matching Gate 1)
_SMARTS_BASIC_ALIPHATIC_AMINE = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;+0;!$(NC=O);!$(NS=O);!$(N=O);!$(N[a]);!$(N=*);!$(N#*)]"
)
_SMARTS_AMIDINE_GUANIDINE_IMINE = Chem.MolFromSmarts("[NX2;+0]=[CX3]([NX3])[#6,N]")
_SMARTS_CARBOXYLIC_OH = Chem.MolFromSmarts("[OX2H1;+0][CX3]=O")
_SMARTS_SULFO_PHOSPHO_OH = Chem.MolFromSmarts("[OX2H1;+0][SX4,PX4](=O)=O")
_SMARTS_TETRAZOLE_NH = Chem.MolFromSmarts("[nH1;+0]1nnnn1")


@dataclass(frozen=True)
class PocketSpec:
    """Verified *Homo sapiens* CNS pocket specification for Iteration 3 docking."""

    pocket_key: str
    target: str
    pdb_id: str
    ligand_code: str
    drug_name: str
    tier: str  # "core", "extended", "drd2_cv"
    canonical_anchors: tuple[str, ...]
    anchor_label: str
    expected_anchor_kinds: tuple[str, ...] = ("salt_bridge", "hbond")

    @property
    def stem(self) -> str:
        """Filename stem in ``clean_pockets/``."""
        return f"{self.pdb_id}_{self.target}_{self.ligand_code}"


CORE_TARGETS: tuple[str, ...] = (
    "DRD1",
    "DRD2",
    "DRD3",
    "VMAT2",
    "ADRA2A",
    "HRH3",
    "PDE10A",
    "CHRM4",
    "HTR2A",
    "SERT",
)
EXTENDED_TARGETS: tuple[str, ...] = ("TAAR1", "DAT")
ALL_TARGETS: tuple[str, ...] = (*CORE_TARGETS, *EXTENDED_TARGETS)

POCKET_SPECS: dict[str, PocketSpec] = {
    "DRD1": PocketSpec(
        "DRD1", "DRD1", "9LLG", "A1EKL", "Flupentixol", "core", ("ASP103",), "Asp103 (3.32)"
    ),
    "DRD2": PocketSpec(
        "DRD2", "DRD2", "6CM4", "8NU", "Risperidone", "core", ("ASP114",), "Asp114 (3.32)"
    ),
    "DRD3": PocketSpec(
        "DRD3", "DRD3", "3PBL", "ETQ", "Eticlopride", "core", ("ASP110",), "Asp110 (3.32)"
    ),
    "VMAT2": PocketSpec(
        "VMAT2",
        "VMAT2",
        "8T69",
        "YHL",
        "Tetrabenazine",
        "core",
        ("GLU312", "ASP399"),
        "Glu312/Asp399",
    ),
    "ADRA2A": PocketSpec(
        "ADRA2A",
        "ADRA2A",
        "7EJ8",
        "J59",
        "Brimonidine",
        "core",
        ("ASP113", "ASP128"),
        "Asp128 (3.32)",
    ),
    "HRH3": PocketSpec(
        "HRH3",
        "HRH3",
        "7F61",
        "1IB",
        "PF-03654746",
        "core",
        ("ASP114", "GLU206"),
        "Asp114 (3.32)/Glu206",
    ),
    "PDE10A": PocketSpec(
        "PDE10A",
        "PDE10A",
        "5SJX",
        "PF9",
        "Mardepodect (MP-10)",
        "core",
        ("GLN726", "TYR693"),
        "Gln726/Tyr693",
        expected_anchor_kinds=("hbond",),
    ),
    "CHRM4": PocketSpec(
        "CHRM4", "CHRM4", "8FX5", "XNO", "Xanomeline", "core", ("ASP112",), "Asp112 (3.32)"
    ),
    "HTR2A": PocketSpec(
        "HTR2A", "HTR2A", "7WC8", "92S", "Lumateperone", "core", ("ASP155",), "Asp155 (3.32)"
    ),
    "SERT": PocketSpec(
        "SERT", "SERT", "5I6X", "8PR", "Paroxetine", "core", ("ASP98",), "Asp98 (S1)"
    ),
    "TAAR1": PocketSpec(
        "TAAR1", "TAAR1", "8W88", "UJL", "Ulotaront", "extended", ("ASP103",), "Asp103 (3.32)"
    ),
    "DAT": PocketSpec(
        "DAT", "DAT", "9JKI", "1XR", "Bupropion", "extended", ("ASP79",), "Asp79 (S1)"
    ),
    "DRD2_6LUQ": PocketSpec(
        "DRD2_6LUQ",
        "DRD2",
        "6LUQ",
        "GMJ",
        "Haloperidol",
        "drd2_cv",
        ("ASP114",),
        "Asp114 (3.32)",
    ),
}

# Clinical reference drugs to dock alongside generated candidates
CLINICAL_CONTROLS: list[tuple[str, str, str, tuple[str, ...]]] = [
    # (slug, display_name, target_id, pockets_to_dock)
    ("ecopipam", "Ecopipam", "DRD1", ("DRD1", "DRD2", "DRD2_6LUQ")),
    ("sch23390", "SCH-23390", "DRD1", ("DRD1", "DRD2", "DRD2_6LUQ")),
    ("risperidone", "Risperidone", "DRD2", ("DRD1", "DRD2", "DRD2_6LUQ")),
    ("haloperidol", "Haloperidol", "DRD2", ("DRD1", "DRD2", "DRD2_6LUQ")),
    ("valbenazine", "Valbenazine", "VMAT2", ("VMAT2",)),
    ("tetrabenazine", "Tetrabenazine", "VMAT2", ("VMAT2",)),
    ("guanfacine", "Guanfacine", "ADRA2A", ("ADRA2A",)),
    ("pitolisant", "Pitolisant", "HRH3", ("HRH3",)),
    ("balipodect", "Balipodect (TAK-063)", "PDE10A", ("PDE10A",)),
    ("mp10", "Mardepodect (MP-10)", "PDE10A", ("PDE10A",)),
    ("xanomeline", "Xanomeline", "CHRM4", ("CHRM4",)),
    ("lumateperone", "Lumateperone", "HTR2A", ("HTR2A",)),
    ("escitalopram", "Escitalopram", "SERT", ("SERT",)),
    ("eticlopride", "Eticlopride", "DRD3", ("DRD3",)),
    ("ulotaront", "Ulotaront", "TAAR1", ("TAAR1",)),
    ("bupropion", "Bupropion", "DAT", ("DAT",)),
]

ARM_ORDER: tuple[str, ...] = (
    "Ctrl-0A",
    "Ctrl-0B",
    "Iter-0",
    "Iter-1",
    "Iter-2",
    "Iter-2_Shuffled",
    "A4_NoSel",
)

DRD1_SELECTIVITY_ARMS: tuple[str, ...] = (
    "Iter-2",
    "A4_NoSel",
    "Iter-1",
    "Iter-0",
    "Ctrl-0B",
    "Iter-2_Shuffled",
)

# ---------------------------------------------------------------------------
# Chemistry Helpers (pH 7.4 Protonation, Ring-System Conformer Generation)
# ---------------------------------------------------------------------------


def extract_canonical_isomeric_smiles_from_3d(mol_3d: Chem.Mol) -> str:
    """Extract canonical isomeric SMILES preserving both R/S and E/Z stereochemistry."""
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
    """Deterministic physiological pH 7.4 SMARTS protonation (consistent with Gate 1)."""
    base = Chem.RemoveHs(Chem.Mol(mol))
    if not preserve_coords:
        base.RemoveAllConformers()
    rw = Chem.RWMol(base)
    n_atoms = rw.GetNumAtoms()
    dist_mat = Chem.GetDistanceMatrix(rw)

    protonated: set[int] = set()
    if _SMARTS_AMIDINE_GUANIDINE_IMINE is not None:
        for match in rw.GetSubstructMatches(_SMARTS_AMIDINE_GUANIDINE_IMINE):
            n_idx = int(match[0])
            if n_idx not in protonated:
                rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
                protonated.add(n_idx)

    cand_amines: list[int] = []
    if _SMARTS_BASIC_ALIPHATIC_AMINE is not None:
        cand_amines = [int(m[0]) for m in rw.GetSubstructMatches(_SMARTS_BASIC_ALIPHATIC_AMINE)]

    def _basicity_key(atom_idx: int) -> tuple[int, int, int]:
        ewg_2 = 0
        ewg_3 = 0
        for j in range(n_atoms):
            if j == atom_idx:
                continue
            if rw.GetAtomWithIdx(j).GetAtomicNum() in {8, 9, 16, 17, 35, 53}:
                d = int(round(float(dist_mat[atom_idx, j])))
                if d == 2:
                    ewg_2 += 1
                elif d == 3:
                    ewg_3 += 1
        return (ewg_2 * 3 + ewg_3, rw.GetAtomWithIdx(atom_idx).GetTotalNumHs(), atom_idx)

    for n_idx in sorted(set(cand_amines), key=_basicity_key):
        if any(int(round(float(dist_mat[n_idx, p]))) <= 3 for p in protonated):
            continue
        rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
        protonated.add(n_idx)

    for acid_smarts in (_SMARTS_CARBOXYLIC_OH, _SMARTS_SULFO_PHOSPHO_OH, _SMARTS_TETRAZOLE_NH):
        if acid_smarts is None:
            continue
        for match in rw.GetSubstructMatches(acid_smarts):
            a_idx = int(match[0])
            if rw.GetAtomWithIdx(a_idx).GetFormalCharge() == 0:
                rw.GetAtomWithIdx(a_idx).SetFormalCharge(-1)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _extract_fused_ring_systems(mol_h: Chem.Mol) -> list[list[int]]:
    """Group ring atoms into independent fused/spiro ring systems + local stereo probes."""
    atom_rings = [set(r) for r in mol_h.GetRingInfo().AtomRings()]
    if not atom_rings:
        return []
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
    """Proper SO(3) (det=+1) Kabsch RMSD between two ordered point clouds."""
    a_c = pts_a - pts_a.mean(axis=0, keepdims=True)
    b_c = pts_b - pts_b.mean(axis=0, keepdims=True)
    u_mat, _, vt_mat = np.linalg.svd(a_c.T @ b_c)
    det_sign = 1.0 if float(np.linalg.det(vt_mat.T @ u_mat.T)) >= 0.0 else -1.0
    rot = vt_mat.T @ np.diag(np.array([1.0, 1.0, det_sign], dtype=np.float64)) @ u_mat.T
    diff = (a_c @ rot.T) - b_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _max_ring_system_rmsd(
    mol_h: Chem.Mol, cid_a: int, cid_b: int, ring_systems: list[list[int]]
) -> float:
    """Max internal SO(3) Kabsch RMSD across independent fused ring systems."""
    conf_a = mol_h.GetConformer(cid_a)
    conf_b = mol_h.GetConformer(cid_b)
    max_rmsd = 0.0
    for idxs in ring_systems:
        pts_a = np.asarray([list(conf_a.GetAtomPosition(i)) for i in idxs], dtype=np.float64)
        pts_b = np.asarray([list(conf_b.GetAtomPosition(i)) for i in idxs], dtype=np.float64)
        max_rmsd = max(max_rmsd, _so3_kabsch_rmsd(pts_a, pts_b))
    return max_rmsd


def prepare_ligand_bundle(
    canonical_smiles: str,
    *,
    seed: int = 42,
    num_confs: int = 24,
    max_ring_confs: int = 2,
    ring_rmsd_cutoff: float = 0.28,
    energy_window_kcal: float = 5.0,
) -> dict[str, Any]:
    """Prepare one unique SMILES once: pH 7.4 -> ETKDGv3 + MMFF94s -> Meeko PDBQTs.

    Returns PDBQT strings for up to ``max_ring_confs`` distinct low-energy ring/invertomer
    conformers plus MMFF94s torsional strain energy (``E_init - E_min``), ``n_heavy``,
    ``clogp``, ``mw``, and ``smiles_ph74``.
    """
    mol_2d = Chem.MolFromSmiles(canonical_smiles)
    if mol_2d is None:
        return {"smiles": canonical_smiles, "success": False, "error": "Invalid SMILES"}

    try:
        mol_prot = protonate_ligand_ph74(mol_2d, preserve_coords=False)
    except Exception as exc:
        return {"smiles": canonical_smiles, "success": False, "error": f"Protonation: {exc}"}

    smiles_ph74 = str(Chem.MolToSmiles(mol_prot, canonical=True, isomericSmiles=True))
    n_heavy = int(mol_prot.GetNumHeavyAtoms())
    clogp = float(Descriptors.MolLogP(mol_2d))
    mw = float(Descriptors.MolWt(mol_2d))
    net_charge = int(Chem.GetFormalCharge(mol_prot))

    mol_h = Chem.AddHs(Chem.MolFromSmiles(smiles_ph74))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    params.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))  # type: ignore[attr-defined]
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))  # type: ignore[attr-defined]
    if not cids:
        return {"smiles": canonical_smiles, "success": False, "error": "ETKDGv3 embedding failed"}

    # Compute initial energy on conf 0 before minimization for strain tracking
    init_energies: dict[int, float] = {}
    energies: list[tuple[int, float]] = []
    force_field = "none"

    if AllChem.MMFFHasAllMoleculeParams(mol_h):  # type: ignore[attr-defined]
        force_field = "MMFF94s"
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
        if mmff_props is not None:
            for cid in cids:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=int(cid))  # type: ignore[attr-defined]
                if ff is not None:
                    e0 = float(ff.CalcEnergy())
                    ff.Minimize(maxIts=350)
                    e1 = float(ff.CalcEnergy())
                    init_energies[int(cid)] = e0
                    energies.append((int(cid), e1))
    if not energies and AllChem.UFFHasAllMoleculeParams(mol_h):  # type: ignore[attr-defined]
        force_field = "UFF"
        for cid in cids:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=int(cid))  # type: ignore[attr-defined]
            if ff is not None:
                e0 = float(ff.CalcEnergy())
                ff.Minimize(maxIts=350)
                e1 = float(ff.CalcEnergy())
                init_energies[int(cid)] = e0
                energies.append((int(cid), e1))
    if not energies:
        energies = [(int(cid), 0.0) for cid in cids]

    energies.sort(key=lambda item: item[1])
    min_cid, min_e = energies[0]
    strain_kcal = float(max(0.0, init_energies.get(min_cid, min_e) - min_e))

    ring_systems = _extract_fused_ring_systems(mol_h)
    chosen: list[int] = [min_cid]
    if ring_systems and max_ring_confs > 1:
        for cid, energy in energies[1:]:
            if energy - min_e > energy_window_kcal:
                break
            if all(
                _max_ring_system_rmsd(mol_h, cid, prev_cid, ring_systems) >= ring_rmsd_cutoff
                for prev_cid in chosen
            ):
                chosen.append(cid)
                if len(chosen) >= max_ring_confs:
                    break

    pdbqts: list[str] = []
    for cid in chosen:
        m_one = Chem.Mol(mol_h)
        m_one.RemoveAllConformers()
        m_one.AddConformer(mol_h.GetConformer(cid), assignId=True)
        pdbqt_str = prepare_ligand_pdbqt(m_one)
        if pdbqt_str:
            pdbqts.append(pdbqt_str)

    if not pdbqts:
        return {"smiles": canonical_smiles, "success": False, "error": "Meeko PDBQT failed"}

    return {
        "smiles": canonical_smiles,
        "smiles_ph74": smiles_ph74,
        "success": True,
        "n_heavy": n_heavy,
        "clogp": round(clogp, 4),
        "mw": round(mw, 3),
        "net_charge_ph74": net_charge,
        "force_field": force_field,
        "mmff_strain_kcal": round(strain_kcal, 3),
        "n_ring_confs": len(pdbqts),
        "pdbqts": pdbqts,
    }


# ---------------------------------------------------------------------------
# Receptor Box & Docking Execution Worker
# ---------------------------------------------------------------------------


def _reconstruct_all_modes(
    out_pdbqt_text: str, affinities: list[float], smiles_ph74: str
) -> list[tuple[float, Chem.Mol]]:
    """Reconstruct all Vina modes into RDKit 3D Mols via Meeko with PDB fallback."""
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

    template = Chem.MolFromSmiles(smiles_ph74)
    if template is None:
        return []
    n_expected = template.GetNumAtoms()
    for idx, block in enumerate(out_pdbqt_text.split("ENDMDL")):
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
            mapped = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)  # type: ignore[attr-defined]
            modes.append((float(affinities[idx]), mapped))
        except Exception:
            continue
    return modes


def _deduplicate_modes(
    modes: list[tuple[float, Chem.Mol]],
    *,
    rmsd_tol: float = 0.5,
    max_keep: int = 9,
) -> list[tuple[float, Chem.Mol]]:
    """Rank pooled conformer modes strictly by Vina dG and deduplicate near-duplicates."""
    ordered = sorted(modes, key=lambda x: x[0])
    kept: list[tuple[float, Chem.Mol]] = []
    for aff, mol in ordered:
        is_dup = False
        for _, prev_mol in kept:
            try:
                if compute_heavy_atom_rmsd(prev_mol, mol, align=False) < rmsd_tol:
                    is_dup = True
                    break
            except Exception:
                continue
        if not is_dup:
            kept.append((aff, mol))
            if len(kept) >= max_keep:
                break
    return kept


def _check_anchor(profile: InteractionProfile, spec: PocketSpec) -> dict[str, Any]:
    """Check canonical pharmacophore anchor engagement in a 3D interaction profile."""
    anchor_set = {a.upper() for a in spec.canonical_anchors}
    allowed = set(spec.expected_anchor_kinds)
    hits: list[dict[str, Any]] = []
    for rec in profile.interactions:
        res_base = rec.protein_residue.split(":")[0].upper()
        if res_base in anchor_set and rec.kind in allowed:
            hits.append(
                {
                    "residue": rec.protein_residue,
                    "kind": rec.kind,
                    "protein_atom": rec.protein_atom,
                    "distance_angstrom": round(float(rec.distance_angstrom), 3),
                }
            )
    hits.sort(key=lambda h: (0 if h["kind"] == "salt_bridge" else 1, h["distance_angstrom"]))
    return {
        "engaged": len(hits) > 0,
        "min_dist": hits[0]["distance_angstrom"] if hits else None,
        "hits": hits[:4],
    }


def _compact_contacts(profile: InteractionProfile, max_items: int = 16) -> list[dict[str, Any]]:
    """Extract compact non-hydrophobic + key hydrophobic 3D contacts for dossiers."""
    polar = [r for r in profile.interactions if r.kind != "hydrophobic"]
    hydro = [r for r in profile.interactions if r.kind == "hydrophobic"][:6]
    selected = (polar + hydro)[:max_items]
    return [
        {
            "residue": r.protein_residue,
            "kind": r.kind,
            "protein_atom": r.protein_atom,
            "ligand_element": r.ligand_element,
            "distance_angstrom": round(float(r.distance_angstrom), 3),
            "subtype": r.subtype,
        }
        for r in selected
    ]


def _mol_to_sdf_block(mol: Chem.Mol) -> str:
    """Serialize an RDKit 3D Mol to a V2000 SDF block string."""
    return str(Chem.MolToMolBlock(mol))


def dock_single_pair_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Dock one unique (pocket_key, canonical_smiles) pair and profile Top-1 + Anchor pose."""
    pocket_key: str = task["pocket_key"]
    spec = POCKET_SPECS[pocket_key]
    bundle: dict[str, Any] = task["bundle"]
    smiles: str = bundle["smiles"]
    pocket_pdb = Path(task["pocket_pdb"])
    rec_pdbqt = Path(task["rec_pdbqt"])
    native_sdf = Path(task["native_sdf"])
    box_center: tuple[float, float, float] = tuple(task["box_center"])
    box_size: tuple[float, float, float] = tuple(task["box_size"])
    vina_exe = Path(task["vina_exe"])
    exhaustiveness = int(task["exhaustiveness"])
    num_modes = int(task["num_modes"])
    cpu_per_vina = int(task["cpu_per_vina"])
    seed = int(task["seed"])
    store_sdf = bool(task.get("store_sdf", False))

    if not bundle.get("success") or not bundle.get("pdbqts"):
        return {
            "pocket_key": pocket_key,
            "smiles": smiles,
            "success": False,
            "error": bundle.get("error", "Ligand preparation failed"),
        }

    # Native crystal PLIF reference (loaded fast from pre-cleaned pocket + SDF)
    nat_mol = Chem.SDMolSupplier(str(native_sdf), removeHs=False, sanitize=True)[0]
    nat_ph74 = protonate_ligand_ph74(nat_mol, preserve_coords=True)
    native_plif = profile_3d_interactions(pocket_pdb, nat_ph74)

    pooled: list[tuple[float, Chem.Mol]] = []
    with tempfile.TemporaryDirectory(prefix="geneuc_i3_") as td_str:
        td = Path(td_str)
        for c_idx, lig_pdbqt in enumerate(bundle["pdbqts"]):
            lp = td / f"lig_{c_idx}.pdbqt"
            op = td / f"out_{c_idx}.pdbqt"
            lp.write_text(lig_pdbqt, encoding="utf-8")
            cmd = [
                str(vina_exe),
                "--receptor",
                str(rec_pdbqt),
                "--ligand",
                str(lp),
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
                str(cpu_per_vina),
                "--seed",
                str(seed),
                "--out",
                str(op),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
            except subprocess.TimeoutExpired:
                continue
            if proc.returncode != 0 or not op.exists():
                continue
            out_text = op.read_text(encoding="utf-8", errors="ignore")
            affs = _parse_vina_affinities(out_text, proc.stdout)
            if not affs:
                continue
            pooled.extend(_reconstruct_all_modes(out_text, affs, bundle["smiles_ph74"]))

    ranked = _deduplicate_modes(pooled, rmsd_tol=0.5, max_keep=num_modes)
    if not ranked:
        return {
            "pocket_key": pocket_key,
            "smiles": smiles,
            "success": False,
            "error": "AutoDock Vina returned no valid modes",
        }

    n_heavy = max(int(bundle["n_heavy"]), 1)
    clogp = float(bundle["clogp"])

    # 1. Unconstrained Top-1 pose
    top1_aff, top1_mol = ranked[0]
    top1_pb = check_pose_geometry(top1_mol, pocket_pdb_path=pocket_pdb)
    top1_plif = profile_3d_interactions(pocket_pdb, top1_mol)
    top1_anc = _check_anchor(top1_plif, spec)
    top1_le = float(-top1_aff / n_heavy)
    top1_lle_vina = float((-top1_aff / _RT_LN10) - clogp)

    # 2. Best canonical-anchor-engaging mode across all returned modes 1..9
    anchor_mode_rank: int | None = None
    anchor_aff: float | None = None
    anchor_mol: Chem.Mol | None = None
    anchor_pb = None
    anchor_plif: InteractionProfile | None = None
    anchor_info: dict[str, Any] = {"engaged": False, "min_dist": None, "hits": []}

    if top1_anc["engaged"]:
        anchor_mode_rank = 1
        anchor_aff = top1_aff
        anchor_mol = top1_mol
        anchor_pb = top1_pb
        anchor_plif = top1_plif
        anchor_info = top1_anc
    else:
        for m_idx, (aff_m, mol_m) in enumerate(ranked[1:], start=2):
            plif_m = profile_3d_interactions(pocket_pdb, mol_m)
            anc_m = _check_anchor(plif_m, spec)
            if anc_m["engaged"]:
                anchor_mode_rank = m_idx
                anchor_aff = aff_m
                anchor_mol = mol_m
                anchor_pb = check_pose_geometry(mol_m, pocket_pdb_path=pocket_pdb)
                anchor_plif = plif_m
                anchor_info = anc_m
                break

    # Representative pose for SDF storage: anchor-engaging pose if found & PB-pass, else Top-1
    chosen_pose_mol = top1_mol
    chosen_pose_plif = top1_plif
    chosen_pose_mode = 1
    if anchor_mol is not None and anchor_pb is not None and anchor_pb.passed_all:
        chosen_pose_mol = anchor_mol
        chosen_pose_plif = anchor_plif if anchor_plif is not None else top1_plif
        chosen_pose_mode = anchor_mode_rank or 1

    rec_out: dict[str, Any] = {
        "pocket_key": pocket_key,
        "target": spec.target,
        "pdb_id": spec.pdb_id,
        "smiles": smiles,
        "smiles_ph74": bundle["smiles_ph74"],
        "success": True,
        "n_heavy": n_heavy,
        "clogp": round(clogp, 3),
        "mw": bundle["mw"],
        "net_charge_ph74": bundle["net_charge_ph74"],
        "mmff_strain_kcal": bundle["mmff_strain_kcal"],
        "n_modes": len(ranked),
        "top1_vina_dg": round(float(top1_aff), 3),
        "top1_le": round(top1_le, 4),
        "top1_lle_vina": round(top1_lle_vina, 3),
        "top1_posebusters_passed": bool(top1_pb.passed_all),
        "top1_zero_clash": bool(top1_pb.protein_clash_count == 0),
        "top1_protein_clashes": int(top1_pb.protein_clash_count),
        "top1_internal_clashes": int(top1_pb.internal_clash_count),
        "top1_plif_tanimoto": round(
            top1_plif.tanimoto_to(native_plif, include_hydrophobic=True), 4
        ),
        "top1_plif_tanimoto_polar": round(
            top1_plif.tanimoto_to(native_plif, include_hydrophobic=False), 4
        ),
        "top1_residue_recovery": round(
            top1_plif.residue_recovery_rate(native_plif, include_hydrophobic=True), 4
        ),
        "top1_anchor_engaged": bool(top1_anc["engaged"]),
        "top1_anchor_min_dist": top1_anc["min_dist"],
        "top1_anchor_hits": top1_anc["hits"],
        "top1_interaction_counts": dict(top1_plif.counts_by_type),
        "top1_key_residues": list(top1_plif.key_residues[:12]),
        "top1_contacts": _compact_contacts(top1_plif),
        "anchor_any_mode_engaged": bool(anchor_info["engaged"]),
        "anchor_mode_rank": anchor_mode_rank,
        "anchor_vina_dg": round(float(anchor_aff), 3) if anchor_aff is not None else None,
        "anchor_le": round(float(-anchor_aff / n_heavy), 4) if anchor_aff is not None else None,
        "anchor_posebusters_passed": bool(anchor_pb.passed_all) if anchor_pb is not None else None,
        "anchor_plif_tanimoto": (
            round(anchor_plif.tanimoto_to(native_plif, include_hydrophobic=True), 4)
            if anchor_plif is not None
            else None
        ),
        "anchor_min_dist": anchor_info["min_dist"],
        "anchor_hits": anchor_info["hits"],
        "anchor_contacts": _compact_contacts(chosen_pose_plif),
        "preferred_pose_mode": chosen_pose_mode,
    }
    if store_sdf:
        rec_out["preferred_sdf_block"] = _mol_to_sdf_block(chosen_pose_mol)
        rec_out["top1_sdf_block"] = _mol_to_sdf_block(top1_mol)
    return rec_out


# ---------------------------------------------------------------------------
# Statistical Helpers (Wilcoxon, Mann-Whitney U, Cliff's delta, Bootstrap, Holm)
# ---------------------------------------------------------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute exact Cliff's delta effect size in [-1, 1] (positive => x > y)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    n_gt = float(np.sum(diff > 0))
    n_lt = float(np.sum(diff < 0))
    return float((n_gt - n_lt) / (len(x) * len(y)))


def bootstrap_mean_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    paired: bool = False,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean(x) - mean(y) and 1,000x percentile bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    diff_obs = float(np.mean(x) - np.mean(y))
    if len(x) == 0 or len(y) == 0:
        return diff_obs, float("nan"), float("nan")
    if paired and len(x) == len(y):
        d = x - y
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot_vals = np.mean(d[idx], axis=1)
    else:
        ix = rng.integers(0, len(x), size=(n_boot, len(x)))
        iy = rng.integers(0, len(y), size=(n_boot, len(y)))
        boot_vals = np.mean(x[ix], axis=1) - np.mean(y[iy], axis=1)
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return round(diff_obs, 4), round(float(lo), 4), round(float(hi), 4)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Compute Holm-Bonferroni FWER-adjusted p-values preserving original ordering."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adj = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        p_clean = 1.0 if not math.isfinite(p_val) else float(p_val)
        val = min(1.0, p_clean * (m - rank))
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj


def safe_wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value handling all-zero differences."""
    if len(x) != len(y) or len(x) < 2:
        return 1.0
    diff = x - y
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    try:
        res = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return 1.0


def safe_mwu_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value handling identical constant arrays."""
    if len(x) == 0 or len(y) == 0:
        return 1.0
    if np.allclose(x, x[0]) and np.allclose(y, y[0]) and abs(float(x[0] - y[0])) < 1e-12:
        return 1.0
    try:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# ChEMBL Nearest-Analog Novelty Lookup (ECFP4 2048-bit)
# ---------------------------------------------------------------------------


class ChEMBLReferenceIndex:
    """Fast ECFP4 nearest-analog index over curated 12-target human ChEMBL actives."""

    def __init__(self, csv_path: Path) -> None:
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        df = pd.read_csv(csv_path)
        self.all_fps: list[Any] = []
        self.all_meta: list[dict[str, Any]] = []
        self.by_target_fps: dict[str, list[Any]] = {}
        self.by_target_meta: dict[str, list[dict[str, Any]]] = {}

        seen_global: dict[str, int] = {}
        for row in df.itertuples(index=False):
            smi = str(row.canonical_smiles)
            target = str(row.target_symbol)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = self._gen.GetFingerprint(mol)
            meta = {
                "chembl_id": str(row.molecule_chembl_id),
                "target": target,
                "best_pchembl": round(float(row.best_pchembl), 2),
                "canonical_smiles": smi,
            }
            self.by_target_fps.setdefault(target, []).append(fp)
            self.by_target_meta.setdefault(target, []).append(meta)
            if smi not in seen_global:
                seen_global[smi] = len(self.all_fps)
                self.all_fps.append(fp)
                self.all_meta.append(meta)

    def nearest_analog(self, smiles: str, *, target: str | None = None) -> dict[str, Any]:
        """Return nearest ChEMBL analog (overall and optionally within target)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"max_tanimoto": float("nan"), "nearest": None}
        qfp = self._gen.GetFingerprint(mol)

        sims_all = DataStructs.BulkTanimotoSimilarity(qfp, self.all_fps)
        best_i = int(np.argmax(sims_all))
        out: dict[str, Any] = {
            "max_tanimoto_all_chembl": round(float(sims_all[best_i]), 4),
            "nearest_all_chembl": self.all_meta[best_i],
        }
        if target and target in self.by_target_fps:
            t_fps = self.by_target_fps[target]
            sims_t = DataStructs.BulkTanimotoSimilarity(qfp, t_fps)
            best_t = int(np.argmax(sims_t))
            out["max_tanimoto_same_target"] = round(float(sims_t[best_t]), 4)
            out["nearest_same_target"] = self.by_target_meta[target][best_t]
        return out


# ---------------------------------------------------------------------------
# Analysis Builders (a, b, c, d) and Publication Summary Renderer
# ---------------------------------------------------------------------------


def _pair_key(pocket_key: str, smiles: str) -> str:
    """Stable hashable cache key for a (pocket_key, canonical_smiles) pair."""
    digest = hashlib.sha1(f"{pocket_key}||{smiles}".encode()).hexdigest()[:16]
    return f"{pocket_key}:{digest}"


def _build_arm_summary(
    arm_records: list[dict[str, Any]],
    targets_subset: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate unconstrained + anchor-verified physics metrics for one arm."""
    rows = [r for r in arm_records if r["target"] in targets_subset and r.get("success")]
    n_total = sum(1 for r in arm_records if r["target"] in targets_subset)
    if not rows:
        return {"n_attempted": n_total, "n_succeeded": 0}

    dg = np.asarray([r["top1_vina_dg"] for r in rows], dtype=np.float64)
    le = np.asarray([r["top1_le"] for r in rows], dtype=np.float64)
    lle_v = np.asarray([r["top1_lle_vina"] for r in rows], dtype=np.float64)
    lle_p = np.asarray(
        [
            float(r["cand"]["pred_pkd"]) - float(r["clogp"])
            for r in rows
            if r.get("cand") and r["cand"].get("pred_pkd") is not None
        ],
        dtype=np.float64,
    )
    plif = np.asarray([r["top1_plif_tanimoto"] for r in rows], dtype=np.float64)
    res_rec = np.asarray([r["top1_residue_recovery"] for r in rows], dtype=np.float64)
    strain = np.asarray([r["mmff_strain_kcal"] for r in rows], dtype=np.float64)
    pb = np.asarray([1.0 if r["top1_posebusters_passed"] else 0.0 for r in rows], dtype=np.float64)
    zclash = np.asarray([1.0 if r["top1_zero_clash"] else 0.0 for r in rows], dtype=np.float64)
    anc_t1 = np.asarray([1.0 if r["top1_anchor_engaged"] else 0.0 for r in rows], dtype=np.float64)
    anc_any = np.asarray(
        [1.0 if r["anchor_any_mode_engaged"] else 0.0 for r in rows], dtype=np.float64
    )
    ha_rate = np.asarray([1.0 if r["high_affinity_top1"] else 0.0 for r in rows], dtype=np.float64)
    joint_phys_safe = np.asarray(
        [
            1.0
            if (
                r["top1_posebusters_passed"]
                and r["top1_le"] >= 0.30
                and bool(r.get("cand", {}).get("joint_cns_safe", False))
            )
            else 0.0
            for r in rows
        ],
        dtype=np.float64,
    )
    joint_phys_anc_safe = np.asarray(
        [
            1.0
            if (
                r["top1_posebusters_passed"]
                and r["top1_le"] >= 0.30
                and r["anchor_any_mode_engaged"]
                and bool(r.get("cand", {}).get("joint_cns_safe", False))
            )
            else 0.0
            for r in rows
        ],
        dtype=np.float64,
    )

    anc_rows = [r for r in rows if r["anchor_any_mode_engaged"] and r["anchor_vina_dg"] is not None]
    anc_dg = np.asarray([r["anchor_vina_dg"] for r in anc_rows], dtype=np.float64)
    anc_le = np.asarray([r["anchor_le"] for r in anc_rows], dtype=np.float64)
    anc_pb = np.asarray(
        [1.0 if r["anchor_posebusters_passed"] else 0.0 for r in anc_rows], dtype=np.float64
    )
    anc_plif = np.asarray([r["anchor_plif_tanimoto"] for r in anc_rows], dtype=np.float64)

    return {
        "n_attempted": n_total,
        "n_succeeded": len(rows),
        "dock_success_pct": round(100.0 * len(rows) / max(n_total, 1), 1),
        "top1_vina_dg_mean": round(float(np.mean(dg)), 3),
        "top1_vina_dg_median": round(float(np.median(dg)), 3),
        "top1_vina_dg_std": round(float(np.std(dg, ddof=0)), 3),
        "top1_le_mean": round(float(np.mean(le)), 4),
        "top1_le_median": round(float(np.median(le)), 4),
        "top1_lle_vina_mean": round(float(np.mean(lle_v)), 3),
        "top1_lle_painn_mean": round(float(np.mean(lle_p)), 3) if len(lle_p) else None,
        "high_affinity_within_0_5_pct": round(100.0 * float(np.mean(ha_rate)), 1),
        "top1_posebusters_pass_pct": round(100.0 * float(np.mean(pb)), 1),
        "top1_zero_clash_pct": round(100.0 * float(np.mean(zclash)), 1),
        "mmff_strain_mean_kcal": round(float(np.mean(strain)), 2),
        "mmff_strain_median_kcal": round(float(np.median(strain)), 2),
        "top1_anchor_engaged_pct": round(100.0 * float(np.mean(anc_t1)), 1),
        "top1_plif_tanimoto_mean": round(float(np.mean(plif)), 4),
        "top1_residue_recovery_mean": round(float(np.mean(res_rec)), 4),
        "anchor_any_mode_engaged_pct": round(100.0 * float(np.mean(anc_any)), 1),
        "anchor_verified_n": len(anc_rows),
        "anchor_verified_vina_dg_mean": (round(float(np.mean(anc_dg)), 3) if len(anc_dg) else None),
        "anchor_verified_le_mean": round(float(np.mean(anc_le)), 4) if len(anc_le) else None,
        "anchor_verified_posebusters_pct": (
            round(100.0 * float(np.mean(anc_pb)), 1) if len(anc_pb) else None
        ),
        "anchor_verified_plif_mean": (
            round(float(np.mean(anc_plif)), 4) if len(anc_plif) else None
        ),
        "joint_physics_cns_safe_pct": round(100.0 * float(np.mean(joint_phys_safe)), 1),
        "joint_physics_anchor_cns_safe_pct": round(100.0 * float(np.mean(joint_phys_anc_safe)), 1),
    }


def _compute_arm_comparisons(
    records_by_arm: dict[str, list[dict[str, Any]]],
    targets_subset: tuple[str, ...],
) -> dict[str, Any]:
    """Compute paired Wilcoxon, MWU, Cliff's delta, 95% CIs, and Holm-Bonferroni vs Iter-2."""
    comparators = ["Ctrl-0A", "Ctrl-0B", "Iter-0", "Iter-1", "Iter-2_Shuffled", "A4_NoSel"]
    metrics_spec = [
        ("top1_vina_dg", False),
        ("top1_le", True),
        ("top1_lle_vina", True),
        ("top1_plif_tanimoto", True),
        ("top1_anchor_engaged", True),
        ("anchor_any_mode_engaged", True),
        ("top1_posebusters_passed", True),
        ("joint_physics_cns_safe", True),
    ]

    def _extract_val(rec: dict[str, Any], key: str) -> float:
        if key == "joint_physics_cns_safe":
            ok = (
                rec["top1_posebusters_passed"]
                and rec["top1_le"] >= 0.30
                and bool(rec.get("cand", {}).get("joint_cns_safe", False))
            )
            return 1.0 if ok else 0.0
        val = rec.get(key)
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        return float(val)

    i2_rows = [
        r for r in records_by_arm["Iter-2"] if r["target"] in targets_subset and r["success"]
    ]

    out: dict[str, dict[str, Any]] = {}
    for m_key, _higher_better in metrics_spec:
        raw_wilc_ps: list[float] = []
        raw_mwu_ps: list[float] = []
        entries: list[tuple[str, dict[str, Any]]] = []

        # Target-level paired means for Iter-2
        i2_by_t = {
            t: float(np.mean([_extract_val(r, m_key) for r in i2_rows if r["target"] == t]))
            for t in targets_subset
        }
        x_pool = np.asarray([_extract_val(r, m_key) for r in i2_rows], dtype=np.float64)
        x_paired = np.asarray([i2_by_t[t] for t in targets_subset], dtype=np.float64)

        for comp in comparators:
            c_rows = [
                r for r in records_by_arm[comp] if r["target"] in targets_subset and r["success"]
            ]
            c_by_t = {
                t: float(np.mean([_extract_val(r, m_key) for r in c_rows if r["target"] == t]))
                for t in targets_subset
            }
            y_pool = np.asarray([_extract_val(r, m_key) for r in c_rows], dtype=np.float64)
            y_paired = np.asarray([c_by_t[t] for t in targets_subset], dtype=np.float64)

            p_w = safe_wilcoxon_p(x_paired, y_paired)
            p_u = safe_mwu_p(x_pool, y_pool)
            delta = cliffs_delta(x_pool, y_pool)
            diff_m, ci_lo, ci_hi = bootstrap_mean_diff_ci(
                x_pool, y_pool, paired=False, n_boot=1000, seed=42
            )
            p_diff, p_lo, p_hi = bootstrap_mean_diff_ci(
                x_paired, y_paired, paired=True, n_boot=1000, seed=42
            )
            raw_wilc_ps.append(p_w)
            raw_mwu_ps.append(p_u)
            entries.append(
                (
                    comp,
                    {
                        "iter2_mean": round(float(np.mean(x_pool)), 4),
                        "comp_mean": round(float(np.mean(y_pool)), 4),
                        "mean_diff": diff_m,
                        "boot95_ci": [ci_lo, ci_hi],
                        "paired_target_diff": p_diff,
                        "paired_boot95_ci": [p_lo, p_hi],
                        "cliffs_delta": round(delta, 4),
                        "wilcoxon_p": round(p_w, 6),
                        "mwu_p": round(p_u, 6),
                    },
                )
            )

        adj_w = holm_bonferroni(raw_wilc_ps)
        adj_u = holm_bonferroni(raw_mwu_ps)
        m_dict: dict[str, Any] = {}
        for (comp, item), pw_adj, pu_adj in zip(entries, adj_w, adj_u, strict=True):
            item["wilcoxon_p_holm"] = round(pw_adj, 6)
            item["mwu_p_holm"] = round(pu_adj, 6)
            m_dict[comp] = item
        out[m_key] = m_dict
    return out


def build_cross_target_matrix(
    candidates_data: dict[str, Any],
    dock_cache: dict[str, dict[str, Any]],
    *,
    top_k: int = 5,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build 10x10 cross-target selectivity matrix and run 10,000x permutation test."""
    n = len(CORE_TARGETS)
    mean_dg = np.full((n, n), np.nan, dtype=np.float64)
    mean_le = np.full((n, n), np.nan, dtype=np.float64)
    mean_plif = np.full((n, n), np.nan, dtype=np.float64)
    anchor_rate = np.full((n, n), np.nan, dtype=np.float64)

    for i, design_t in enumerate(CORE_TARGETS):
        ligs = candidates_data["arms"]["Iter-2"][design_t][:top_k]
        for j, pocket_t in enumerate(CORE_TARGETS):
            dgs: list[float] = []
            les: list[float] = []
            plifs: list[float] = []
            ancs: list[float] = []
            for it in ligs:
                rec = dock_cache.get(_pair_key(pocket_t, it["canonical_smiles"]))
                if rec and rec.get("success"):
                    dgs.append(float(rec["top1_vina_dg"]))
                    les.append(float(rec["top1_le"]))
                    plifs.append(float(rec["top1_plif_tanimoto"]))
                    ancs.append(1.0 if rec["anchor_any_mode_engaged"] else 0.0)
            if dgs:
                mean_dg[i, j] = float(np.mean(dgs))
                mean_le[i, j] = float(np.mean(les))
                mean_plif[i, j] = float(np.mean(plifs))
                anchor_rate[i, j] = float(np.mean(ancs))

    # Column-normalized z-score selectivity (positive = stronger binding than pocket average)
    col_mean = np.nanmean(mean_dg, axis=0, keepdims=True)
    col_std = np.nanstd(mean_dg, axis=0, keepdims=True)
    col_std_safe = np.where(col_std > 1e-8, col_std, 1.0)
    z_col = -(mean_dg - col_mean) / col_std_safe

    # Row-normalized z-score (positive = ligand prefers this pocket over other pockets)
    row_mean = np.nanmean(mean_dg, axis=1, keepdims=True)
    row_std = np.nanstd(mean_dg, axis=1, keepdims=True)
    row_std_safe = np.where(row_std > 1e-8, row_std, 1.0)
    z_row = -(mean_dg - row_mean) / row_std_safe

    # Double-centered interaction residual (removes BOTH pocket-size & ligand-MW main effects!)
    grand_mean = float(np.nanmean(mean_dg))
    interaction_dg = -(mean_dg - row_mean - col_mean + grand_mean)
    inter_std = float(np.nanstd(interaction_dg))
    z_double = interaction_dg / max(inter_std, 1e-8)

    # 10,000x exact permutation test of diagonal enrichment (permuting column labels)
    rng = np.random.default_rng(seed)

    def _perm_test(mat: np.ndarray, *, higher_is_better: bool) -> dict[str, Any]:
        diag_vals = np.diag(mat)
        mask_off = ~np.eye(n, dtype=bool)
        off_vals = mat[mask_off]
        obs_diag = float(np.mean(diag_vals))
        obs_off = float(np.mean(off_vals))
        obs_enrichment = obs_diag - obs_off if higher_is_better else obs_off - obs_diag

        null_diag = np.empty(n_permutations, dtype=np.float64)
        idx_rows = np.arange(n)
        for p_idx in range(n_permutations):
            perm = rng.permutation(n)
            null_diag[p_idx] = float(np.mean(mat[idx_rows, perm]))

        if higher_is_better:
            p_val = float((1 + np.sum(null_diag >= obs_diag - 1e-12)) / (n_permutations + 1))
        else:
            p_val = float((1 + np.sum(null_diag <= obs_diag + 1e-12)) / (n_permutations + 1))

        # Count how many targets rank in Top-3 on their own diagonal
        n_diag_positive = (
            int(np.sum(diag_vals > 0.0))
            if higher_is_better
            else int(np.sum(diag_vals < np.mean(mat, axis=0)))
        )
        return {
            "diagonal_mean": round(obs_diag, 4),
            "off_diagonal_mean": round(obs_off, 4),
            "enrichment_margin": round(obs_enrichment, 4),
            "permutation_p_10k": round(p_val, 6),
            "n_targets_above_pocket_mean": n_diag_positive,
        }

    perm_results = {
        "raw_vina_dg": _perm_test(mean_dg, higher_is_better=False),
        "col_zscore_selectivity": _perm_test(z_col, higher_is_better=True),
        "double_centered_interaction_z": _perm_test(z_double, higher_is_better=True),
        "ligand_efficiency": _perm_test(mean_le, higher_is_better=True),
        "plif_tanimoto": _perm_test(mean_plif, higher_is_better=True),
    }

    per_target_diag = []
    for i, t in enumerate(CORE_TARGETS):
        col_rank = int(1 + np.sum(mean_dg[:, i] < mean_dg[i, i]))
        per_target_diag.append(
            {
                "target": t,
                "diag_vina_dg": round(float(mean_dg[i, i]), 3),
                "col_mean_vina_dg": round(float(col_mean[0, i]), 3),
                "col_zscore": round(float(z_col[i, i]), 3),
                "double_centered_z": round(float(z_double[i, i]), 3),
                "double_centered_ddg_kcal": round(float(-interaction_dg[i, i]), 3),
                "diag_plif_tanimoto": round(float(mean_plif[i, i]), 4),
                "col_rank_of_10": col_rank,
            }
        )

    return {
        "targets": list(CORE_TARGETS),
        "top_k_per_target": top_k,
        "n_permutations": n_permutations,
        "mean_vina_dg_matrix": np.round(mean_dg, 3).tolist(),
        "mean_le_matrix": np.round(mean_le, 4).tolist(),
        "mean_plif_tanimoto_matrix": np.round(mean_plif, 4).tolist(),
        "anchor_engagement_rate_matrix": np.round(anchor_rate, 3).tolist(),
        "col_zscore_selectivity_matrix": np.round(z_col, 3).tolist(),
        "row_zscore_selectivity_matrix": np.round(z_row, 3).tolist(),
        "double_centered_z_matrix": np.round(z_double, 3).tolist(),
        "permutation_tests": perm_results,
        "per_target_diagonal": per_target_diag,
    }


def build_drd1_vs_drd2_study(
    candidates_data: dict[str, Any],
    dock_cache: dict[str, dict[str, Any]],
    native_refs: dict[str, dict[str, Any]],
    clinical_smiles: dict[str, str],
) -> dict[str, Any]:
    """Flagship Tourette DRD1 (9LLG) vs Off-Target DRD2 (6CM4 & 6LUQ) Selectivity Study."""
    nat_d1 = float(native_refs["DRD1"]["top1_vina_dg"])
    nat_d2_6cm4 = float(native_refs["DRD2"]["top1_vina_dg"])
    nat_d2_6luq = float(native_refs["DRD2_6LUQ"]["top1_vina_dg"])

    drd1_selectivity_residues = {
        "ASP103",
        "LYS81",
        "SER107",
        "SER198",
        "SER199",
        "SER202",
        "TRP285",
        "PHE288",
        "PHE289",
        "ASN292",
        "LEU190",
    }
    drd2_selectivity_residues = {
        "ASP114",
        "CYS118",
        "SER193",
        "SER197",
        "TRP386",
        "PHE389",
        "PHE390",
        "HIS393",
        "ILE184",
        "TYR408",
    }

    def _score_ligand(label: str, smi: str, cand_meta: dict[str, Any] | None) -> dict[str, Any]:
        r_d1 = dock_cache.get(_pair_key("DRD1", smi))
        r_6cm4 = dock_cache.get(_pair_key("DRD2", smi))
        r_6luq = dock_cache.get(_pair_key("DRD2_6LUQ", smi))
        if not (
            r_d1
            and r_6cm4
            and r_6luq
            and r_d1.get("success")
            and r_6cm4.get("success")
            and r_6luq.get("success")
        ):
            return {"label": label, "smiles": smi, "success": False}

        dg_d1 = float(r_d1["top1_vina_dg"])
        dg_6cm4 = float(r_6cm4["top1_vina_dg"])
        dg_6luq = float(r_6luq["top1_vina_dg"])
        dg_d2_mean = 0.5 * (dg_6cm4 + dg_6luq)

        ddg_6cm4 = dg_d1 - dg_6cm4  # Negative => DRD1-selective
        ddg_6luq = dg_d1 - dg_6luq
        ddg_mean = dg_d1 - dg_d2_mean
        ddg_rel_6cm4 = (dg_d1 - nat_d1) - (dg_6cm4 - nat_d2_6cm4)
        ddg_rel_6luq = (dg_d1 - nat_d1) - (dg_6luq - nat_d2_6luq)

        d1_contacts = [
            c
            for c in r_d1.get("top1_contacts", [])
            if c["residue"].split(":")[0].upper() in drd1_selectivity_residues
        ]
        d2_contacts = [
            c
            for c in r_6cm4.get("top1_contacts", [])
            if c["residue"].split(":")[0].upper() in drd2_selectivity_residues
        ]

        return {
            "label": label,
            "smiles": smi,
            "success": True,
            "dg_drd1_9llg": round(dg_d1, 3),
            "le_drd1_9llg": r_d1["top1_le"],
            "dg_drd2_6cm4": round(dg_6cm4, 3),
            "dg_drd2_6luq": round(dg_6luq, 3),
            "dg_drd2_mean": round(dg_d2_mean, 3),
            "ddg_d1_vs_6cm4": round(ddg_6cm4, 3),
            "ddg_d1_vs_6luq": round(ddg_6luq, 3),
            "ddg_d1_vs_d2_mean": round(ddg_mean, 3),
            "ddg_native_rel_6cm4": round(ddg_rel_6cm4, 3),
            "ddg_native_rel_6luq": round(ddg_rel_6luq, 3),
            "drd1_asp103_top1": bool(r_d1["top1_anchor_engaged"]),
            "drd1_asp103_any": bool(r_d1["anchor_any_mode_engaged"]),
            "drd1_plif_tanimoto": r_d1["top1_plif_tanimoto"],
            "drd1_posebusters": bool(r_d1["top1_posebusters_passed"]),
            "drd2_6cm4_asp114_top1": bool(r_6cm4["top1_anchor_engaged"]),
            "drd1_hotspot_contacts": d1_contacts[:10],
            "drd2_hotspot_contacts": d2_contacts[:10],
            "qsar_delta_sel": (
                round(float(cand_meta["delta_sel"]), 4)
                if cand_meta and cand_meta.get("delta_sel") is not None
                else None
            ),
            "cns_mpo_v2": (
                round(float(cand_meta["cns_mpo_v2"]), 3)
                if cand_meta and cand_meta.get("cns_mpo_v2") is not None
                else None
            ),
            "joint_cns_safe": bool(cand_meta.get("joint_cns_safe", False)) if cand_meta else None,
        }

    arm_entries: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in DRD1_SELECTIVITY_ARMS:
        rows = [
            _score_ligand(f"{arm}_DRD1_{idx}", it["canonical_smiles"], it)
            for idx, it in enumerate(candidates_data["arms"][arm]["DRD1"], start=1)
        ]
        ok = [r for r in rows if r.get("success")]
        arm_entries[arm] = rows
        if ok:
            ddg_6cm4 = np.asarray([r["ddg_d1_vs_6cm4"] for r in ok], dtype=np.float64)
            ddg_6luq = np.asarray([r["ddg_d1_vs_6luq"] for r in ok], dtype=np.float64)
            ddg_mean = np.asarray([r["ddg_d1_vs_d2_mean"] for r in ok], dtype=np.float64)
            ddg_rel = np.asarray([r["ddg_native_rel_6cm4"] for r in ok], dtype=np.float64)
            dg_d1 = np.asarray([r["dg_drd1_9llg"] for r in ok], dtype=np.float64)
            dg_d2 = np.asarray([r["dg_drd2_mean"] for r in ok], dtype=np.float64)
            arm_summaries[arm] = {
                "n": len(ok),
                "mean_dg_drd1": round(float(np.mean(dg_d1)), 3),
                "mean_dg_drd2": round(float(np.mean(dg_d2)), 3),
                "mean_ddg_6cm4": round(float(np.mean(ddg_6cm4)), 3),
                "median_ddg_6cm4": round(float(np.median(ddg_6cm4)), 3),
                "mean_ddg_6luq": round(float(np.mean(ddg_6luq)), 3),
                "median_ddg_6luq": round(float(np.median(ddg_6luq)), 3),
                "mean_ddg_d2_mean": round(float(np.mean(ddg_mean)), 3),
                "median_ddg_d2_mean": round(float(np.median(ddg_mean)), 3),
                "mean_ddg_native_rel_6cm4": round(float(np.mean(ddg_rel)), 3),
                "pct_ddg_6cm4_lt_0": round(100.0 * float(np.mean(ddg_6cm4 < 0.0)), 1),
                "pct_ddg_6luq_lt_0": round(100.0 * float(np.mean(ddg_6luq < 0.0)), 1),
                "pct_ddg_mean_lt_neg0_5": round(100.0 * float(np.mean(ddg_mean <= -0.5)), 1),
                "pct_ddg_mean_lt_neg1_0": round(100.0 * float(np.mean(ddg_mean <= -1.0)), 1),
                "pct_drd1_asp103_any": round(
                    100.0 * float(np.mean([1.0 if r["drd1_asp103_any"] else 0.0 for r in ok])), 1
                ),
                "pct_drd1_posebusters": round(
                    100.0 * float(np.mean([1.0 if r["drd1_posebusters"] else 0.0 for r in ok])), 1
                ),
                "best_ddg_d2_mean": round(float(np.min(ddg_mean)), 3),
            }

    # Clinical positive (ecopipam, SCH-23390) & negative D2 (haloperidol, risperidone) controls
    clinical_rows = []
    for slug, disp in (
        ("ecopipam", "Ecopipam (D1-selective Ph3 positive ctrl)"),
        ("sch23390", "SCH-23390 (D1-selective tool ctrl)"),
        ("haloperidol", "Haloperidol (D2-selective negative ctrl)"),
        ("risperidone", "Risperidone (D2/5-HT2A negative ctrl)"),
    ):
        smi = clinical_smiles.get(slug)
        if smi:
            clinical_rows.append(_score_ligand(disp, smi, None))

    # Statistical comparisons: Iter-2 vs A4_NoSel, Iter-1, Iter-0, Ctrl-0B, Iter-2_Shuffled
    comps = ["A4_NoSel", "Iter-1", "Iter-0", "Ctrl-0B", "Iter-2_Shuffled"]
    i2_ok = [r for r in arm_entries["Iter-2"] if r.get("success")]
    stat_tests: dict[str, dict[str, Any]] = {}
    for field in ("ddg_d1_vs_6cm4", "ddg_d1_vs_6luq", "ddg_d1_vs_d2_mean", "dg_drd1_9llg"):
        x = np.asarray([r[field] for r in i2_ok], dtype=np.float64)
        ps: list[float] = []
        items: list[tuple[str, dict[str, Any]]] = []
        for comp in comps:
            y = np.asarray(
                [r[field] for r in arm_entries[comp] if r.get("success")], dtype=np.float64
            )
            p_u = safe_mwu_p(x, y)
            delta = cliffs_delta(x, y)
            diff_m, lo, hi = bootstrap_mean_diff_ci(x, y, paired=False, n_boot=1000, seed=42)
            ps.append(p_u)
            items.append(
                (
                    comp,
                    {
                        "iter2_mean": round(float(np.mean(x)), 3),
                        "comp_mean": round(float(np.mean(y)), 3),
                        "mean_diff": round(diff_m, 3),
                        "boot95_ci": [round(lo, 3), round(hi, 3)],
                        "cliffs_delta": round(delta, 4),
                        "mwu_p": round(p_u, 6),
                    },
                )
            )
        adj = holm_bonferroni(ps)
        f_out: dict[str, Any] = {}
        for (comp, item), p_adj in zip(items, adj, strict=True):
            item["mwu_p_holm"] = round(p_adj, 6)
            f_out[comp] = item
        stat_tests[field] = f_out

    return {
        "native_baselines": {
            "DRD1_9LLG_flupentixol_dg": nat_d1,
            "DRD2_6CM4_risperidone_dg": nat_d2_6cm4,
            "DRD2_6LUQ_haloperidol_dg": nat_d2_6luq,
        },
        "arm_summaries": arm_summaries,
        "clinical_controls": clinical_rows,
        "statistical_comparisons_vs_iter2": stat_tests,
        "per_arm_ligands": arm_entries,
    }


def _write_annotated_sdf(sdf_path: Path, sdf_block: str, props: dict[str, Any]) -> None:
    """Write a 3D SDF file with standardized molecular properties for 3Dmol/PyMOL/viz3d."""
    mol = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=False)
    if mol is None:
        sdf_path.write_text(sdf_block, encoding="utf-8")
        return
    for k, v in props.items():
        if v is not None:
            mol.SetProp(str(k), str(v))
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()


def build_top_hits_dossier(
    candidates_data: dict[str, Any],
    dock_cache: dict[str, dict[str, Any]],
    chembl_index: ChEMBLReferenceIndex,
    poses_dir: Path,
    clinical_smiles: dict[str, str],
) -> dict[str, Any]:
    """Select Top-3 prioritized Iter-2 hits per target, build dossier JSON, and save 3D SDFs."""
    poses_dir.mkdir(parents=True, exist_ok=True)
    dossiers_by_target: dict[str, list[dict[str, Any]]] = {}
    flagship_top12: list[dict[str, Any]] = []

    for target in ALL_TARGETS:
        spec = POCKET_SPECS[target]
        cands = candidates_data["arms"]["Iter-2"][target]
        scored: list[tuple[tuple[int, int, int, int, float, float], dict[str, Any]]] = []

        for orig_idx, cand in enumerate(cands, start=1):
            smi = cand["canonical_smiles"]
            dock = dock_cache.get(_pair_key(target, smi))
            if not dock or not dock.get("success"):
                continue
            analog = chembl_index.nearest_analog(smi, target=target)
            max_tan = float(analog.get("max_tanimoto_all_chembl", 1.0))
            is_novel_lt60 = max_tan < 0.60

            # Also attach cross-target DRD2 dG if already docked (e.g. DRD1 hits or Top-5 core)
            d2_6cm4 = dock_cache.get(_pair_key("DRD2", smi)) if target != "DRD2" else None
            d2_6luq = dock_cache.get(_pair_key("DRD2_6LUQ", smi)) if target != "DRD2" else None
            ddg_vs_d2 = None
            if d2_6cm4 and d2_6cm4.get("success"):
                ddg_vs_d2 = round(float(dock["top1_vina_dg"]) - float(d2_6cm4["top1_vina_dg"]), 3)
            ddg_vs_6luq = None
            if d2_6luq and d2_6luq.get("success"):
                ddg_vs_6luq = round(float(dock["top1_vina_dg"]) - float(d2_6luq["top1_vina_dg"]), 3)

            # Priority sort: PB-pass -> anchor-engaged -> CNS-safe -> novel(<0.60) -> LE & dG
            sort_key = (
                1 if dock["top1_posebusters_passed"] else 0,
                1 if dock["anchor_any_mode_engaged"] else 0,
                1 if bool(cand.get("joint_cns_safe", False)) else 0,
                1 if is_novel_lt60 else 0,
                round(float(dock["top1_le"]), 4),
                -round(float(dock["top1_vina_dg"]), 3),
            )
            entry = {
                "target": target,
                "pdb_id": spec.pdb_id,
                "anchor_label": spec.anchor_label,
                "iter2_selection_rank": orig_idx,
                "canonical_smiles": smi,
                "smiles_ph74": dock["smiles_ph74"],
                "murcko_scaffold": cand.get("murcko_scaffold", ""),
                "physics_3d": {
                    "top1_vina_dg_kcal_mol": dock["top1_vina_dg"],
                    "top1_le": dock["top1_le"],
                    "top1_lle_vina": dock["top1_lle_vina"],
                    "anchor_mode_rank": dock["anchor_mode_rank"],
                    "anchor_vina_dg_kcal_mol": dock["anchor_vina_dg"],
                    "anchor_engaged_top1": dock["top1_anchor_engaged"],
                    "anchor_engaged_any": dock["anchor_any_mode_engaged"],
                    "anchor_min_dist_angstrom": dock["anchor_min_dist"],
                    "anchor_hits": dock["anchor_hits"],
                    "posebusters_passed": dock["top1_posebusters_passed"],
                    "protein_clashes": dock["top1_protein_clashes"],
                    "mmff_strain_kcal": dock["mmff_strain_kcal"],
                    "plif_tanimoto": dock["top1_plif_tanimoto"],
                    "residue_recovery": dock["top1_residue_recovery"],
                    "ddg_vs_drd2_6cm4": ddg_vs_d2,
                    "ddg_vs_drd2_6luq": ddg_vs_6luq,
                    "interacting_contacts": dock["anchor_contacts"],
                    "key_residues": dock["top1_key_residues"],
                },
                "cns_admet": {
                    "cns_mpo_v1": round(float(cand["cns_mpo_v1"]), 3),
                    "cns_mpo_v2": round(float(cand["cns_mpo_v2"]), 3),
                    "bbb_prob": round(float(cand["bbb_prob"]), 4),
                    "pgp_inhib_prob": round(float(cand["pgp_prob"]), 4),
                    "herg_prob": round(float(cand["herg_prob"]), 4),
                    "min_ad_sim": round(float(cand["min_ad_sim"]), 4),
                    "in_domain_all": bool(cand["in_domain_all"]),
                    "joint_cns_safe": bool(cand["joint_cns_safe"]),
                    "mw": round(float(cand["mw"]), 2),
                    "clogp": round(float(cand["clogp"]), 2),
                    "tpsa": round(float(cand["tpsa"]), 2),
                    "hbd": int(cand["hbd"]),
                    "most_basic_pka": round(float(cand["most_basic_pka"]), 2),
                    "qed": round(float(cand["qed"]), 3),
                    "sa_score": round(float(cand["sa_score"]), 2),
                    "pains_free": bool(cand["pains_free"]),
                    "pred_pkd_painn": round(float(cand["pred_pkd"]), 3),
                    "qsar_p_on_target": round(float(cand["p_on_target"]), 4),
                    "qsar_delta_sel": round(float(cand["delta_sel"]), 4),
                },
                "novelty_chembl": {
                    "is_strict_novel": bool(cand.get("is_strict_novel", True)),
                    "novel_lt_0_60_chembl": is_novel_lt60,
                    "max_tanimoto_all_chembl": analog.get("max_tanimoto_all_chembl"),
                    "nearest_all_chembl": analog.get("nearest_all_chembl"),
                    "max_tanimoto_same_target": analog.get("max_tanimoto_same_target"),
                    "nearest_same_target": analog.get("nearest_same_target"),
                },
                "_sdf_block": dock.get("preferred_sdf_block") or dock.get("top1_sdf_block"),
            }
            scored.append((sort_key, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        top3: list[dict[str, Any]] = []
        for rank_idx, (_, entry) in enumerate(scored[:3], start=1):
            sdf_block = entry.pop("_sdf_block", None)
            sdf_rel = Path(
                f"runs_local.nosync/geneuclidean_v31_cns_dpo/docked_poses_3d/{target}_{rank_idx}.sdf"
            )
            if sdf_block:
                _write_annotated_sdf(
                    poses_dir / f"{target}_{rank_idx}.sdf",
                    sdf_block,
                    {
                        "TARGET": target,
                        "PDB_ID": spec.pdb_id,
                        "DOSSIER_RANK": rank_idx,
                        "CANONICAL_SMILES": entry["canonical_smiles"],
                        "VINA_DG": entry["physics_3d"]["top1_vina_dg_kcal_mol"],
                        "LE": entry["physics_3d"]["top1_le"],
                        "ANCHOR_ENGAGED": entry["physics_3d"]["anchor_engaged_any"],
                        "CNS_MPO_V2": entry["cns_admet"]["cns_mpo_v2"],
                        "BBB_PROB": entry["cns_admet"]["bbb_prob"],
                        "PGP_PROB": entry["cns_admet"]["pgp_inhib_prob"],
                        "HERG_PROB": entry["cns_admet"]["herg_prob"],
                        "NEAREST_CHEMBL_TANIMOTO": entry["novelty_chembl"][
                            "max_tanimoto_all_chembl"
                        ],
                    },
                )
            entry["dossier_rank"] = rank_idx
            entry["saved_sdf_path"] = str(sdf_rel)
            top3.append(entry)
            if rank_idx == 1:
                flagship_top12.append(entry)

        dossiers_by_target[target] = top3

    # Save clinical control docked SDFs for side-by-side 3D studio comparisons
    saved_clinical_sdfs: list[str] = []
    for slug, disp_name, _t_id, pockets in CLINICAL_CONTROLS:
        smi = clinical_smiles.get(slug)
        if not smi:
            continue
        for p_key in pockets:
            dock = dock_cache.get(_pair_key(p_key, smi))
            if not dock or not dock.get("success"):
                continue
            sdf_block = dock.get("preferred_sdf_block") or dock.get("top1_sdf_block")
            if not sdf_block:
                continue
            fname = f"CLINICAL_{p_key}_{slug}.sdf"
            _write_annotated_sdf(
                poses_dir / fname,
                sdf_block,
                {
                    "DRUG_NAME": disp_name,
                    "POCKET_KEY": p_key,
                    "PDB_ID": POCKET_SPECS[p_key].pdb_id,
                    "CANONICAL_SMILES": smi,
                    "VINA_DG": dock["top1_vina_dg"],
                    "LE": dock["top1_le"],
                    "ANCHOR_ENGAGED": dock["anchor_any_mode_engaged"],
                },
            )
            saved_clinical_sdfs.append(
                f"runs_local.nosync/geneuclidean_v31_cns_dpo/docked_poses_3d/{fname}"
            )

    return {
        "disclaimer": (
            "All prioritized de novo compounds are strictly in silico computational hypotheses "
            "requiring stereopure synthesis, functional GPCR/transporter/enzyme assays, broad "
            "off-target panels (PDSP/SafetyScreen44), MDCK-MDR1 Kp,uu, and patch-clamp hERG."
        ),
        "flagship_top12": flagship_top12,
        "top3_by_target": dossiers_by_target,
        "saved_clinical_sdfs": saved_clinical_sdfs,
    }


def render_iter3_markdown(
    docking_payload: dict[str, Any],
    matrix_payload: dict[str, Any],
    drd1_payload: dict[str, Any],
    dossier_payload: dict[str, Any],
) -> str:
    """Render concise, publication-grade Iteration 3 Markdown summary."""
    proto = docking_payload["protocol"]
    arm_c10 = docking_payload["arm_summaries_core10"]
    arm_a12 = docking_payload["arm_summaries_all12"]
    stats_c10 = docking_payload["statistical_comparisons_core10"]
    native_refs = docking_payload["native_references"]
    perm = matrix_payload["permutation_tests"]
    d1_arms = drd1_payload["arm_summaries"]
    d1_clin = drd1_payload["clinical_controls"]
    d1_stats = drd1_payload["statistical_comparisons_vs_iter2"]

    lines: list[str] = [
        "# GenEuclidean v3.1 — Iteration 3 Independent 3D Physics, 10×10 Cross-Target "
        "Selectivity & Flagship DRD1 vs. DRD2 Summary",
        "",
        "## 1. Protocol & Anti-Goodhart Separation",
        f"- **Docking Engine:** `{proto['vina_version']}` "
        f"(`exhaustiveness={proto['exhaustiveness']}`, `num_modes={proto['num_modes']}`, "
        f"`seed={proto['seed']}`, `cpu={proto['cpu_per_vina']}`/run; Vina was **100% held out** "
        "from all Iter-1 SFT & Iter-2 CNS-RPO-DPO training).",
        "- **Universal Tight Per-Axis Orthosteric Box (Directive #156):** "
        "`box_axis_i = clamp(native_span_i + 6.0, 14.0, 22.0)` Å centered on native centroid "
        "across ALL 13 *Homo sapiens* pockets (10 core + 2 extended + `DRD2:6LUQ`).",
        "- **Ligand Chemistry:** Deterministic physiological pH 7.4 SMARTS protonation -> unbiased "
        "RDKit `ETKDGv3` + `MMFF94s` (`24` confs, per-fused-ring-system SO(3) clustering, up to 2 "
        "ring/invertomer conformers pooled & ranked strictly by Vina $\\Delta G$) -> Meeko PDBQT.",
        f"- **Unique `(pocket, smiles)` Dockings Executed:** `{proto['n_unique_pairs_docked']}` "
        f"(total wall time: `{proto['total_wall_sec']:.1f} s`).",
        "",
        "## 2. Analysis (a) — On-Target 3D Physics Audit (Core-10 Primary Targets)",
        "",
        "| Arm | N | Vina ΔG Mean (Med) | LE Mean | LLE_Vina | High-Aff (≤Nat+0.5) | PB Pass | "
        "Zero-Clash | MMFF Strain | Anchor (T1 / Any) | Anchor ΔG | PLIF Tan. (T1) | "
        "Joint Phys+CNS-Safe |",
        "| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | "
        "---: | ---: |",
    ]

    for arm in ARM_ORDER:
        s = arm_c10[arm]
        anc_dg = (
            f"{s['anchor_verified_vina_dg_mean']:.2f}"
            if s.get("anchor_verified_vina_dg_mean") is not None
            else "—"
        )
        lines.append(
            f"| **{arm}** | {s['n_succeeded']} | "
            f"{s['top1_vina_dg_mean']:.2f} ({s['top1_vina_dg_median']:.2f}) | "
            f"**{s['top1_le_mean']:.3f}** | {s['top1_lle_vina_mean']:.2f} | "
            f"{s['high_affinity_within_0_5_pct']:.1f}% | "
            f"{s['top1_posebusters_pass_pct']:.1f}% | {s['top1_zero_clash_pct']:.1f}% | "
            f"{s['mmff_strain_mean_kcal']:.1f} | "
            f"{s['top1_anchor_engaged_pct']:.1f}% / {s['anchor_any_mode_engaged_pct']:.1f}% | "
            f"{anc_dg} | {s['top1_plif_tanimoto_mean']:.3f} | "
            f"**{s['joint_physics_cns_safe_pct']:.1f}%** |"
        )

    # All-12 rollup row for Iter-2 vs Ctrl-0B & Iter-0
    lines.extend(
        [
            "",
            "### All-12 Targets Rollup (Core-10 + Extended TAAR1 & DAT)",
            "",
            "| Arm | N | Vina ΔG Mean | LE Mean | LLE_Vina | PB Pass | "
            "Anchor (T1 / Any) | PLIF Tan. (T1) | Joint Phys+CNS-Safe |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for arm in ARM_ORDER:
        s = arm_a12[arm]
        lines.append(
            f"| **{arm}** | {s['n_succeeded']} | {s['top1_vina_dg_mean']:.2f} | "
            f"{s['top1_le_mean']:.3f} | {s['top1_lle_vina_mean']:.2f} | "
            f"{s['top1_posebusters_pass_pct']:.1f}% | "
            f"{s['top1_anchor_engaged_pct']:.1f}% / {s['anchor_any_mode_engaged_pct']:.1f}% | "
            f"{s['top1_plif_tanimoto_mean']:.3f} | "
            f"**{s['joint_physics_cns_safe_pct']:.1f}%** |"
        )

    lines.extend(
        [
            "",
            "### Statistical Comparisons (`Iter-2` vs. Comparators — Core-10, Holm FWER)",
            "",
            "| Comparator | ΔVina ΔG [95% CI] (MWU p_H / δ) | ΔLE [95% CI] (MWU p_H / δ) | "
            "ΔPLIF [95% CI] (MWU p_H / δ) | ΔJoint Phys+Safe (Wilc p_H / MWU p_H / δ) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for comp in ("Ctrl-0A", "Ctrl-0B", "Iter-0", "Iter-1", "Iter-2_Shuffled", "A4_NoSel"):
        sdg = stats_c10["top1_vina_dg"][comp]
        sle = stats_c10["top1_le"][comp]
        spl = stats_c10["top1_plif_tanimoto"][comp]
        sjs = stats_c10["joint_physics_cns_safe"][comp]
        lines.append(
            f"| **vs {comp}** | "
            f"{sdg['mean_diff']:+.2f} [{sdg['boot95_ci'][0]:+.2f},{sdg['boot95_ci'][1]:+.2f}] "
            f"(p={sdg['mwu_p_holm']:.2e}, δ={sdg['cliffs_delta']:+.2f}) | "
            f"{sle['mean_diff']:+.3f} [{sle['boot95_ci'][0]:+.3f},{sle['boot95_ci'][1]:+.3f}] "
            f"(p={sle['mwu_p_holm']:.2e}, δ={sle['cliffs_delta']:+.2f}) | "
            f"{spl['mean_diff']:+.3f} (p={spl['mwu_p_holm']:.2e}, δ={spl['cliffs_delta']:+.2f}) | "
            f"{sjs['mean_diff'] * 100:+.1f} pp "
            f"(p_W={sjs['wilcoxon_p_holm']:.2e}, p_U={sjs['mwu_p_holm']:.2e}, "
            f"δ={sjs['cliffs_delta']:+.2f}) |"
        )

    # Native reference table
    lines.extend(
        [
            "",
            "### Native Crystal Baselines in Universal Tight Per-Axis Orthosteric Box",
            "",
            "| Pocket | PDB | Native Drug | Tight Box (Å) | Native ΔG (kcal/mol) | LE | "
            "Native Anchor Engaged |",
            "| :--- | :--- | :--- | :---: | ---: | ---: | :---: |",
        ]
    )
    for p_key, spec in POCKET_SPECS.items():
        nr = native_refs[p_key]
        bx = docking_payload["pocket_boxes"][p_key]["box_size"]
        lines.append(
            f"| **{p_key}** | `{spec.pdb_id}` | {spec.drug_name} | "
            f"{bx[0]:.1f}×{bx[1]:.1f}×{bx[2]:.1f} | {nr['top1_vina_dg']:.2f} | "
            f"{nr['top1_le']:.3f} | {nr['anchor_any_mode_engaged']} |"
        )

    # Analysis (b) 10x10 Cross-Target Selectivity Matrix
    p_raw = perm["raw_vina_dg"]
    p_col = perm["col_zscore_selectivity"]
    p_dbl = perm["double_centered_interaction_z"]
    p_plif = perm["plif_tanimoto"]
    lines.extend(
        [
            "",
            "## 3. Analysis (b) — 10×10 Cross-Target Selectivity Matrix (`Iter-2` Top-5/Target)",
            f"- **Raw Vina ΔG Diagonal vs. Off-Diagonal:** `{p_raw['diagonal_mean']:.3f}` vs "
            f"`{p_raw['off_diagonal_mean']:.3f}` kcal/mol "
            f"(enrichment `{p_raw['enrichment_margin']:+.3f}` kcal/mol, "
            f"10,000× perm $p = {p_raw['permutation_p_10k']:.5f}$).",
            f"- **Column-Normalized Z-Score Selectivity:** "
            f"Diagonal `{p_col['diagonal_mean']:+.3f}` vs "
            f"Off-Diagonal `{p_col['off_diagonal_mean']:+.3f}` "
            f"(10,000× perm $p = {p_col['permutation_p_10k']:.5f}$, "
            f"`{p_col['n_targets_above_pocket_mean']}/10` targets above pocket mean).",
            f"- **Double-Centered Interaction Z-Score (Controls Pocket-Size & Ligand-MW):** "
            f"Diagonal **`{p_dbl['diagonal_mean']:+.3f}`** vs Off-Diagonal "
            f"`{p_dbl['off_diagonal_mean']:+.3f}` "
            f"(10,000× perm **$p = {p_dbl['permutation_p_10k']:.5f}$**).",
            f"- **3D PLIF Tanimoto Diagonal vs. Off-Diagonal:** `{p_plif['diagonal_mean']:.3f}` vs "
            f"`{p_plif['off_diagonal_mean']:.3f}` "
            f"(10,000× perm $p = {p_plif['permutation_p_10k']:.5f}$).",
            "",
            "| Design \\ Docked | " + " | ".join(CORE_TARGETS) + " |",
            "| :--- | " + " | ".join("---:" for _ in CORE_TARGETS) + " |",
        ]
    )
    mat_dg = matrix_payload["mean_vina_dg_matrix"]
    for i, t in enumerate(CORE_TARGETS):
        row_cells = [
            f"**{mat_dg[i][j]:.2f}**" if i == j else f"{mat_dg[i][j]:.2f}"
            for j in range(len(CORE_TARGETS))
        ]
        lines.append(f"| **{t}** | " + " | ".join(row_cells) + " |")

    # Analysis (c) Flagship DRD1 vs DRD2 Study
    lines.extend(
        [
            "",
            "## 4. Analysis (c) — Flagship Tourette `DRD1 (9LLG)` vs. Off-Target "
            "`DRD2 (6CM4 & 6LUQ)` Selectivity",
            "",
            "| Arm / Clinical Control | N | DRD1 (9LLG) ΔG | DRD2 (6CM4) ΔG | DRD2 (6LUQ) ΔG | "
            "ΔΔG (9LLG−6CM4) | ΔΔG (9LLG−6LUQ) | ΔΔG Mean | ΔΔG_rel (6CM4) | "
            "% ΔΔG_mean ≤ −0.5 | DRD1 Asp103 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in DRD1_SELECTIVITY_ARMS:
        s = d1_arms[arm]
        dg_6cm4 = s["mean_dg_drd1"] - s["mean_ddg_6cm4"]
        dg_6luq = s["mean_dg_drd1"] - s["mean_ddg_6luq"]
        lines.append(
            f"| **{arm}** | {s['n']} | {s['mean_dg_drd1']:.2f} | {dg_6cm4:.2f} | "
            f"{dg_6luq:.2f} | **{s['mean_ddg_6cm4']:+.2f}** | "
            f"**{s['mean_ddg_6luq']:+.2f}** | **{s['mean_ddg_d2_mean']:+.2f}** | "
            f"{s['mean_ddg_native_rel_6cm4']:+.2f} | {s['pct_ddg_mean_lt_neg0_5']:.0f}% | "
            f"{s['pct_drd1_asp103_any']:.0f}% |"
        )
    for cr in d1_clin:
        if not cr.get("success"):
            continue
        lines.append(
            f"| *{cr['label']}* | 1 | {cr['dg_drd1_9llg']:.2f} | {cr['dg_drd2_6cm4']:.2f} | "
            f"{cr['dg_drd2_6luq']:.2f} | {cr['ddg_d1_vs_6cm4']:+.2f} | "
            f"{cr['ddg_d1_vs_6luq']:+.2f} | **{cr['ddg_d1_vs_d2_mean']:+.2f}** | "
            f"{cr['ddg_native_rel_6cm4']:+.2f} | — | {cr['drd1_asp103_any']} |"
        )

    lines.extend(["", "### DRD1 vs. DRD2 Statistical Tests (`Iter-2` vs. Comparators on ΔΔG_mean)"])
    for comp in ("A4_NoSel", "Iter-1", "Iter-0", "Ctrl-0B", "Iter-2_Shuffled"):
        st = d1_stats["ddg_d1_vs_d2_mean"][comp]
        lines.append(
            f"- **vs `{comp}`:** Δ = `{st['mean_diff']:+.3f}` kcal/mol "
            f"(95% CI `[{st['boot95_ci'][0]:+.3f}, {st['boot95_ci'][1]:+.3f}]`, "
            f"Cliff's δ = `{st['cliffs_delta']:+.3f}`, MWU $p = {st['mwu_p']:.4f}$, "
            f"Holm $p = {st['mwu_p_holm']:.4f}$)."
        )

    # Analysis (d) Top-12 Prioritized Candidate Dossiers
    lines.extend(
        [
            "",
            "## 5. Analysis (d) — Top-12 Prioritized *De Novo* CNS Candidate Dossiers "
            "(#1 Lead Per Target)",
            "> **Translational Disclaimer:** All prioritized candidates are strictly *in silico* "
            "computational hypotheses requiring synthesis and experimental validation.",
            "",
            "| Target (PDB) | Vina ΔG | LE | Anchor (Dist Å) | PLIF | PB | MPO v2 | "
            "BBB / Pgp / hERG | AD | Max ChEMBL Tan. (Nearest ID) | SMILES |",
            "| :--- | ---: | ---: | :--- | ---: | :---: | ---: | :---: | ---: | :--- | :--- |",
        ]
    )
    for d in dossier_payload["flagship_top12"]:
        p3 = d["physics_3d"]
        ad = d["cns_admet"]
        nv = d["novelty_chembl"]
        anc_str = (
            f"Yes ({p3['anchor_min_dist_angstrom']:.2f})"
            if p3["anchor_engaged_any"] and p3["anchor_min_dist_angstrom"] is not None
            else "No"
        )
        near = nv.get("nearest_all_chembl") or {}
        near_str = f"{nv['max_tanimoto_all_chembl']:.2f} (`{near.get('chembl_id', '—')}`)"
        lines.append(
            f"| **{d['target']}** (`{d['pdb_id']}`) | {p3['top1_vina_dg_kcal_mol']:.2f} | "
            f"{p3['top1_le']:.3f} | {anc_str} | {p3['plif_tanimoto']:.3f} | "
            f"{'PASS' if p3['posebusters_passed'] else 'FAIL'} | {ad['cns_mpo_v2']:.2f} | "
            f"{ad['bbb_prob']:.2f} / {ad['pgp_inhib_prob']:.2f} / {ad['herg_prob']:.2f} | "
            f"{ad['min_ad_sim']:.2f} | {near_str} | `{d['canonical_smiles']}` |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Iteration 3 docking & selectivity study."""
    parser = argparse.ArgumentParser(
        description="Iteration 3 Independent 3D Physics, 10x10 Matrix & DRD1/DRD2 Study."
    )
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=Path("runs_local.nosync/geneuclidean_v31_cns_dpo/candidates_for_docking.json"),
    )
    parser.add_argument(
        "--clean-pockets-dir",
        type=Path,
        default=Path("data/cns_neuro.nosync/structures/clean_pockets"),
    )
    parser.add_argument(
        "--chembl-csv",
        type=Path,
        default=Path("data/cns_neuro.nosync/chembl/human_cns_actives_12targets.csv"),
    )
    parser.add_argument(
        "--admet-oracles",
        type=Path,
        default=Path("data/cns_neuro.nosync/admet_oracles.joblib"),
    )
    parser.add_argument(
        "--cache-jsonl",
        type=Path,
        default=Path("runs_local.nosync/geneuclidean_v31_cns_dpo/iter3_vina_cache.jsonl"),
    )
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=Path("runs_local.nosync/geneuclidean_v31_cns_dpo/docked_poses_3d"),
    )
    parser.add_argument("--vina-bin", type=Path, default=Path("bin/vina"))
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cpu-per-vina", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def _resolve_clinical_smiles() -> dict[str, str]:
    """Pull verified canonical SMILES for clinical reference controls from targets registry."""
    name_map: dict[str, str] = {}
    for spec in CNS_TARGET_REGISTRY.values():
        for ref in spec.reference_drugs:
            key = ref.name.lower()
            name_map[key] = ref.canonical_smiles
    resolved: dict[str, str] = {}
    lookup_hints = {
        "ecopipam": "ecopipam",
        "sch23390": "sch-23390",
        "risperidone": "risperidone",
        "haloperidol": "haloperidol",
        "valbenazine": "valbenazine",
        "tetrabenazine": "tetrabenazine",
        "guanfacine": "guanfacine",
        "pitolisant": "pitolisant",
        "balipodect": "balipodect (tak-063)",
        "mp10": "mardepodect (mp-10 / pf-02545920)",
        "xanomeline": "xanomeline",
        "lumateperone": "lumateperone",
        "escitalopram": "escitalopram",
        "eticlopride": "eticlopride",
        "ulotaront": "ulotaront (sep-363856)",
        "bupropion": "bupropion",
    }
    for slug, key in lookup_hints.items():
        if key in name_map:
            mol = Chem.MolFromSmiles(name_map[key])
            if mol is not None:
                resolved[slug] = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return resolved


def main() -> None:
    """Execute full Iteration 3 independent 3D physics & selectivity suite."""
    args = parse_args()
    t0 = time.perf_counter()

    vina_exe = _resolve_vina_binary(args.vina_bin)
    if vina_exe is None:
        raise FileNotFoundError(f"AutoDock Vina binary not found: {args.vina_bin}")
    ver_proc = subprocess.run([str(vina_exe), "--version"], capture_output=True, text=True)
    vina_version = (ver_proc.stdout or ver_proc.stderr).strip() or "AutoDock Vina"

    candidates_data = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    clinical_smiles = _resolve_clinical_smiles()

    # 1. Compute universal tight per-axis orthosteric box + receptor PDBQT for all 13 pockets
    rec_cache_dir = args.poses_dir.parent / "receptor_pdbqts_iter3"
    rec_cache_dir.mkdir(parents=True, exist_ok=True)
    pocket_meta: dict[str, dict[str, Any]] = {}
    native_smiles_by_pocket: dict[str, str] = {}

    for p_key, spec in POCKET_SPECS.items():
        pocket_pdb = args.clean_pockets_dir / f"{spec.stem}_pocket.pdb"
        native_sdf = args.clean_pockets_dir / f"{spec.stem}_lig.sdf"
        if not pocket_pdb.exists() or not native_sdf.exists():
            raise FileNotFoundError(f"Missing clean pocket or SDF for {spec.stem}")

        nat_mol = Chem.SDMolSupplier(str(native_sdf), removeHs=False, sanitize=True)[0]
        nat_smi = extract_canonical_isomeric_smiles_from_3d(nat_mol)
        native_smiles_by_pocket[p_key] = nat_smi

        box_center = compute_ligand_box_center(native_sdf)
        conf = nat_mol.GetConformer(0)
        h_coords = np.asarray(
            [
                list(conf.GetAtomPosition(i))
                for i in range(nat_mol.GetNumAtoms())
                if nat_mol.GetAtomWithIdx(i).GetAtomicNum() > 1
            ],
            dtype=np.float64,
        )
        span_xyz = h_coords.max(axis=0) - h_coords.min(axis=0)
        # Universal tight per-axis orthosteric box per #156 Directive #1
        box_size = tuple(float(x) for x in np.clip(span_xyz + 6.0, 14.0, 22.0))
        rec_pdbqt = prepare_receptor_pdbqt(pocket_pdb, rec_cache_dir / f"{p_key}.pdbqt")
        if rec_pdbqt is None:
            raise RuntimeError(f"Failed receptor PDBQT for {p_key}")

        pocket_meta[p_key] = {
            "pocket_pdb": str(pocket_pdb),
            "native_sdf": str(native_sdf),
            "rec_pdbqt": str(rec_pdbqt),
            "box_center": [round(c, 3) for c in box_center],
            "box_size": [round(s, 3) for s in box_size],
            "native_span_xyz": [round(float(x), 3) for x in span_xyz],
            "native_smiles": nat_smi,
        }

    # 2. Collect ALL unique (pocket_key, smiles) pairs needed across Analyses (a), (b), (c), (d)
    # Mark which pairs need 3D SDF blocks retained (all Iter-2 on-target + clinical controls)
    required_pairs: dict[str, tuple[str, str, bool]] = {}

    def _add_pair(p_key: str, smi: str, store_sdf: bool = False) -> None:
        pk = _pair_key(p_key, smi)
        if pk in required_pairs:
            prev_p, prev_s, prev_sdf = required_pairs[pk]
            required_pairs[pk] = (prev_p, prev_s, prev_sdf or store_sdf)
        else:
            required_pairs[pk] = (p_key, smi, store_sdf)

    # Native crystal references (all 13 pockets)
    for p_key, nat_smi in native_smiles_by_pocket.items():
        _add_pair(p_key, nat_smi, store_sdf=False)

    # Analysis (a): all 7 arms x 12 targets x 10 candidates
    for arm in ARM_ORDER:
        for target in ALL_TARGETS:
            for item in candidates_data["arms"][arm][target]:
                _add_pair(target, item["canonical_smiles"], store_sdf=(arm == "Iter-2"))

    # Analysis (b): 10x10 cross-target matrix (Top-5 Iter-2 per core target x 10 core pockets)
    for design_t in CORE_TARGETS:
        for item in candidates_data["arms"]["Iter-2"][design_t][:5]:
            for pocket_t in CORE_TARGETS:
                _add_pair(pocket_t, item["canonical_smiles"], store_sdf=False)

    # Analysis (c): Flagship DRD1 vs DRD2 (6CM4 & 6LUQ) across 6 arms (Top-10 DRD1 candidates)
    for arm in DRD1_SELECTIVITY_ARMS:
        for item in candidates_data["arms"][arm]["DRD1"]:
            _add_pair("DRD1", item["canonical_smiles"], store_sdf=(arm == "Iter-2"))
            _add_pair("DRD2", item["canonical_smiles"], store_sdf=False)
            _add_pair("DRD2_6LUQ", item["canonical_smiles"], store_sdf=False)

    # Clinical controls
    for slug, _disp, _t_id, pockets in CLINICAL_CONTROLS:
        smi = clinical_smiles.get(slug)
        if smi:
            for p_key in pockets:
                _add_pair(p_key, smi, store_sdf=True)

    # 3. Load any cached results from previous partial runs
    dock_cache: dict[str, dict[str, Any]] = {}
    if args.cache_jsonl.exists():
        for line in args.cache_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                pk = _pair_key(rec["pocket_key"], rec["smiles"])
                dock_cache[pk] = rec
            except Exception:
                continue

    missing_keys = [
        pk
        for pk, (_, _, need_sdf) in required_pairs.items()
        if pk not in dock_cache
        or (
            need_sdf
            and dock_cache[pk].get("success")
            and "preferred_sdf_block" not in dock_cache[pk]
        )
    ]
    print(
        f"[Iter 3] {vina_version} | Total unique (pocket, ligand) pairs={len(required_pairs)} | "
        f"Cached={len(required_pairs) - len(missing_keys)} | To dock={len(missing_keys)} "
        f"(workers={args.workers}, cpu/vina={args.cpu_per_vina}, exh={args.exhaustiveness})",
        flush=True,
    )

    # 4. Pre-prepare each unique SMILES ONCE in parallel (ETKDGv3 + MMFF94s + Meeko PDBQT)
    if missing_keys:
        unique_smiles = sorted({required_pairs[pk][1] for pk in missing_keys})
        t_prep = time.perf_counter()
        print(
            f"[Iter 3] Preparing {len(unique_smiles)} unique ligands "
            "(pH 7.4 + ETKDGv3 + MMFF94s + Meeko)...",
            flush=True,
        )
        bundles: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            fmap = {
                pool.submit(
                    prepare_ligand_bundle,
                    smi,
                    seed=int(args.seed),
                    num_confs=24,
                    max_ring_confs=2,
                ): smi
                for smi in unique_smiles
            }
            for fut in as_completed(fmap):
                b = fut.result()
                bundles[b["smiles"]] = b
        n_ok_b = sum(1 for b in bundles.values() if b.get("success"))
        avg_confs = float(
            np.mean([b["n_ring_confs"] for b in bundles.values() if b.get("success")])
        )
        print(
            f"[Iter 3] Prepared {n_ok_b}/{len(unique_smiles)} ligands in "
            f"{time.perf_counter() - t_prep:.1f}s (avg ring conformers={avg_confs:.2f})",
            flush=True,
        )

        # 5. Run AutoDock Vina v1.2.7 across ProcessPoolExecutor with live cache checkpointing
        t_dock = time.perf_counter()
        args.cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
        tasks = []
        for pk in missing_keys:
            p_key, smi, store_sdf = required_pairs[pk]
            pm = pocket_meta[p_key]
            tasks.append(
                {
                    "pocket_key": p_key,
                    "bundle": bundles[smi],
                    "pocket_pdb": pm["pocket_pdb"],
                    "rec_pdbqt": pm["rec_pdbqt"],
                    "native_sdf": pm["native_sdf"],
                    "box_center": pm["box_center"],
                    "box_size": pm["box_size"],
                    "vina_exe": str(vina_exe),
                    "exhaustiveness": int(args.exhaustiveness),
                    "num_modes": int(args.num_modes),
                    "cpu_per_vina": int(args.cpu_per_vina),
                    "seed": int(args.seed),
                    "store_sdf": store_sdf,
                }
            )

        done_cnt = 0
        with (
            args.cache_jsonl.open("a", encoding="utf-8") as cache_fh,
            ProcessPoolExecutor(max_workers=int(args.workers)) as pool,
        ):
            futures = [pool.submit(dock_single_pair_worker, t_item) for t_item in tasks]
            for fut in as_completed(futures):
                rec = fut.result()
                pk = _pair_key(rec["pocket_key"], rec["smiles"])
                dock_cache[pk] = rec
                cache_fh.write(json.dumps(rec) + "\n")
                done_cnt += 1
                if done_cnt % 50 == 0 or done_cnt == len(tasks):
                    cache_fh.flush()
                    elapsed = time.perf_counter() - t_dock
                    rate = done_cnt / max(elapsed, 1e-6)
                    eta = (len(tasks) - done_cnt) / max(rate, 1e-6)
                    pct_done = 100.0 * done_cnt / len(tasks)
                    print(
                        f"  [Vina] {done_cnt:4d}/{len(tasks)} ({pct_done:5.1f}%) "
                        f"elapsed={elapsed:6.1f}s rate={rate:.2f}/s ETA={eta:5.1f}s",
                        flush=True,
                    )

    # 6. Assemble Analysis (a): On-Target Physics Audit across all 7 arms x 12 targets
    native_refs: dict[str, dict[str, Any]] = {}
    for p_key, nat_smi in native_smiles_by_pocket.items():
        nr = dock_cache[_pair_key(p_key, nat_smi)]
        native_refs[p_key] = {k: v for k, v in nr.items() if not k.endswith("_sdf_block")}

    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        arm_recs: list[dict[str, Any]] = []
        for target in ALL_TARGETS:
            nat_dg = float(native_refs[target]["top1_vina_dg"])
            for item in candidates_data["arms"][arm][target]:
                smi = item["canonical_smiles"]
                rec = dock_cache.get(_pair_key(target, smi), {"success": False})
                clean = {k: v for k, v in rec.items() if not k.endswith("_sdf_block")}
                clean["arm"] = arm
                clean["target"] = target
                clean["cand"] = item
                clean["high_affinity_top1"] = bool(
                    clean.get("success") and float(clean["top1_vina_dg"]) <= nat_dg + 0.5
                )
                arm_recs.append(clean)
        records_by_arm[arm] = arm_recs

    arm_summaries_core10 = {
        arm: _build_arm_summary(records_by_arm[arm], CORE_TARGETS) for arm in ARM_ORDER
    }
    arm_summaries_all12 = {
        arm: _build_arm_summary(records_by_arm[arm], ALL_TARGETS) for arm in ARM_ORDER
    }
    per_target_arm_summary: dict[str, dict[str, Any]] = {}
    for target in ALL_TARGETS:
        per_target_arm_summary[target] = {
            arm: _build_arm_summary(records_by_arm[arm], (target,)) for arm in ARM_ORDER
        }

    stats_core10 = _compute_arm_comparisons(records_by_arm, CORE_TARGETS)
    stats_all12 = _compute_arm_comparisons(records_by_arm, ALL_TARGETS)

    # Clinical control on-target benchmark table (including CNS MPO & ADMET)
    admet_suite = CNSAdmetSuite.load(args.admet_oracles) if args.admet_oracles.exists() else None
    clinical_benchmarks: list[dict[str, Any]] = []
    for slug, disp, t_id, _pockets in CLINICAL_CONTROLS:
        smi = clinical_smiles.get(slug)
        if not smi:
            continue
        dock = dock_cache.get(_pair_key(t_id, smi))
        if not dock or not dock.get("success"):
            continue
        mpo = compute_cns_mpo(smi)
        ad_pred = admet_suite.predict([smi])[0] if admet_suite is not None else None
        clinical_benchmarks.append(
            {
                "drug_name": disp,
                "target": t_id,
                "pdb_id": POCKET_SPECS[t_id].pdb_id,
                "smiles": smi,
                "top1_vina_dg": dock["top1_vina_dg"],
                "top1_le": dock["top1_le"],
                "top1_lle_vina": dock["top1_lle_vina"],
                "top1_posebusters_passed": dock["top1_posebusters_passed"],
                "anchor_any_mode_engaged": dock["anchor_any_mode_engaged"],
                "top1_plif_tanimoto": dock["top1_plif_tanimoto"],
                "cns_mpo_v2": round(float(mpo.cns_mpo_v2), 3) if mpo.valid else None,
                "bbb_prob": round(float(ad_pred.bbb_prob), 4) if ad_pred else None,
                "pgp_prob": round(float(ad_pred.pgp_prob), 4) if ad_pred else None,
                "herg_prob": round(float(ad_pred.herg_prob), 4) if ad_pred else None,
            }
        )

    docking_payload: dict[str, Any] = {
        "protocol": {
            "vina_version": vina_version,
            "exhaustiveness": int(args.exhaustiveness),
            "num_modes": int(args.num_modes),
            "seed": int(args.seed),
            "workers": int(args.workers),
            "cpu_per_vina": int(args.cpu_per_vina),
            "box_rule": "universal_tight_per_axis_clamp(native_span_i+6.0,14.0,22.0)",
            "protonation": "deterministic_rdkit_smarts_ph74",
            "conformer_generation": "etkdgv3_mmff94s_24confs_per_ring_system_max2",
            "anti_goodhart_vina_held_out": True,
            "n_unique_pairs_docked": len(required_pairs),
            "total_wall_sec": round(time.perf_counter() - t0, 2),
        },
        "pocket_boxes": {
            k: {
                "pdb_id": POCKET_SPECS[k].pdb_id,
                "box_center": v["box_center"],
                "box_size": v["box_size"],
                "native_span_xyz": v["native_span_xyz"],
            }
            for k, v in pocket_meta.items()
        },
        "native_references": native_refs,
        "clinical_benchmarks": clinical_benchmarks,
        "arm_summaries_core10": arm_summaries_core10,
        "arm_summaries_all12": arm_summaries_all12,
        "per_target_arm_summary": per_target_arm_summary,
        "statistical_comparisons_core10": stats_core10,
        "statistical_comparisons_all12": stats_all12,
    }

    # 7. Assemble Analysis (b): 10x10 Cross-Target Selectivity Matrix
    matrix_payload = build_cross_target_matrix(
        candidates_data, dock_cache, top_k=5, n_permutations=10000, seed=int(args.seed)
    )

    # 8. Assemble Analysis (c): Flagship Tourette DRD1 (9LLG) vs DRD2 (6CM4 & 6LUQ)
    drd1_payload = build_drd1_vs_drd2_study(
        candidates_data, dock_cache, native_refs, clinical_smiles
    )

    # 9. Assemble Analysis (d): Top-12 Prioritized Dossiers + Saved 3D SDFs
    print("[Iter 3] Indexing human ChEMBL actives for nearest-analog novelty audit...", flush=True)
    chembl_index = ChEMBLReferenceIndex(args.chembl_csv)
    dossier_payload = build_top_hits_dossier(
        candidates_data, dock_cache, chembl_index, args.poses_dir, clinical_smiles
    )

    # 10. Save all 5 owned results files
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dock = args.out_dir / "cns_iter3_docking.json"
    out_mat = args.out_dir / "cns_cross_target_matrix.json"
    out_d1 = args.out_dir / "cns_drd1_vs_drd2_selectivity.json"
    out_dos = args.out_dir / "cns_iter3_top_hits_dossier.json"
    out_md = args.out_dir / "cns_iter3_summary.md"

    out_dock.write_text(json.dumps(docking_payload, indent=2) + "\n", encoding="utf-8")
    out_mat.write_text(json.dumps(matrix_payload, indent=2) + "\n", encoding="utf-8")
    out_d1.write_text(json.dumps(drd1_payload, indent=2) + "\n", encoding="utf-8")
    out_dos.write_text(json.dumps(dossier_payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        render_iter3_markdown(docking_payload, matrix_payload, drd1_payload, dossier_payload),
        encoding="utf-8",
    )

    print(f"\n[Iter 3] COMPLETE in {time.perf_counter() - t0:.1f}s!", flush=True)
    print(f"  Saved: {out_dock}")
    print(f"  Saved: {out_mat}")
    print(f"  Saved: {out_d1}")
    print(f"  Saved: {out_dos}")
    print(f"  Saved: {out_md}")
    print(f"  3D SDFs saved in: {args.poses_dir}")

    i2_c10 = arm_summaries_core10["Iter-2"]
    cb_c10 = arm_summaries_core10["Ctrl-0B"]
    print(
        f"  Core-10 Iter-2 vs Ctrl-0B: "
        f"dG={i2_c10['top1_vina_dg_mean']:.2f} vs {cb_c10['top1_vina_dg_mean']:.2f} | "
        f"LE={i2_c10['top1_le_mean']:.3f} vs {cb_c10['top1_le_mean']:.3f} | "
        f"PB={i2_c10['top1_posebusters_pass_pct']:.1f}% | "
        f"Anc(any)={i2_c10['anchor_any_mode_engaged_pct']:.1f}% | "
        f"JointPhysSafe={i2_c10['joint_physics_cns_safe_pct']:.1f}% vs "
        f"{cb_c10['joint_physics_cns_safe_pct']:.1f}%"
    )
    pt = matrix_payload["permutation_tests"]
    d1_i2 = drd1_payload["arm_summaries"]["Iter-2"]
    print(
        f"  10x10 Matrix perm p: raw={pt['raw_vina_dg']['permutation_p_10k']}, "
        f"col_z={pt['col_zscore_selectivity']['permutation_p_10k']}, "
        f"dbl_z={pt['double_centered_interaction_z']['permutation_p_10k']}"
    )
    print(
        f"  DRD1 vs DRD2 Iter-2: ddG_6CM4={d1_i2['mean_ddg_6cm4']:+.2f}, "
        f"ddG_6LUQ={d1_i2['mean_ddg_6luq']:+.2f}, "
        f"ddG_mean={d1_i2['mean_ddg_d2_mean']:+.2f}"
    )


if __name__ == "__main__":
    main()
