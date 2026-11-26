"""Reusable 3D Structural Docking, Native Redocking & Cross-Target Selectivity Engine.

Promotes GenEuclidean's Gate 1 multi-seed redocking benchmark and Iteration 3 cross-target
selectivity physics pipeline into a clean, typed, modular library API:

- **Universal Orthosteric Box Computation** (:func:`compute_orthosteric_box`):
  Supports both the Iteration 3 universal tight per-axis orthosteric box
  ``clamp(native_span_i + 6.0, 14.0, 22.0)`` Å (preventing cut-pocket exterior groove
  escape across matched pocket volumes) and the Gate 1 standard cubic box
  ``clamp(max_span + 8.0, 20.0, 26.0)`` Å.
- **Unbiased Ring- & Invertomer-Diverse 3D Conformer Generation**
  (:func:`generate_ring_diverse_conformers`):
  Embeds conformers strictly from 2D SMILES with RDKit ``ETKDGv3`` + ``MMFF94s`` (``UFF``
  fallback), then clusters by per-fused-ring-system proper-``SO(3)`` (``det = +1``) Kabsch
  RMSD so aliphatic ring puckers, tricyclic butterfly folds, and pyramidal ``[NH+;R]``
  invertomers (held rigid by AutoDock Vina) are systematically covered with zero crystal
  coordinate leakage.
- **Dual Top-1 & Canonical-Anchor-Verified Docking** (:func:`dock_ligand_to_pocket`):
  Pools and deduplicates modes across ring conformers ranked strictly by AutoDock Vina
  ``ΔG``, runs PoseBusters physical checks (:func:`check_pose_geometry`) and exact 3D PLIF
  profiling (:func:`profile_3d_interactions`), and reports both unconstrained Top-1 and
  canonical-anchor-engaging poses alongside Ligand Efficiency (``LE``) and ``LLE_Vina``.
- **Multi-Seed Unbiased Native Redocking** (:func:`redock_native_complex`):
  Computes symmetry-corrected heavy-atom RMSD, Bemis-Murcko core RMSD, PoseBusters pass
  rate, 3D PLIF Tanimoto recovery, and canonical anchor recovery across independent seeds.
- **Cross-Target ΔΔG Selectivity Matrix & Permutation Test**
  (:func:`compute_cross_target_matrix`, :func:`compute_selectivity_ddg`):
  Computes column z-scores, row z-scores, and double-centered interaction residuals
  (removing both pocket-size and ligand-MW main effects) with exact Monte Carlo diagonal
  permutation p-values.

Gracefully raises actionable :class:`RuntimeError` messages if the optional ``meeko``
package or AutoDock Vina executable is unavailable on a minimal CPU install.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from geneuclidean.evaluation.geometry3d import (
    PoseBustersChecks,
    _parse_vina_affinities,
    _resolve_vina_binary,
    check_pose_geometry,
    extract_pocket_heavy_atoms,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
)
from geneuclidean.evaluation.interactions3d import (
    InteractionProfile,
    compute_heavy_atom_rmsd,
    profile_3d_interactions,
)
from geneuclidean.evaluation.protonation import (
    extract_canonical_isomeric_smiles,
    protonate_mol_ph74,
)

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

_ALLCHEM: Any = AllChem
_DESCRIPTORS: Any = Descriptors

# RT * ln(10) at 298.15 K (kcal/mol per log10 Kd unit)
RT_LN10_KCAL: float = 1.364

BoxMode = Literal["tight_per_axis", "cubic_standard"]


# ---------------------------------------------------------------------------
# Structured Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrthostericBox:
    """3D AutoDock Vina orthosteric search box centered on a ligand or binding site."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    native_span_xyz: tuple[float, float, float]
    mode: BoxMode

    def to_dict(self) -> dict[str, Any]:
        """Serialize docking box specification to a JSON-compatible dictionary."""
        return {
            "center": [round(float(v), 3) for v in self.center],
            "size": [round(float(v), 3) for v in self.size],
            "native_span_xyz": [round(float(v), 3) for v in self.native_span_xyz],
            "mode": self.mode,
        }


@dataclass
class RingConformer:
    """Single low-energy 3D conformer representing a distinct ring/invertomer state."""

    mol_3d: Chem.Mol
    energy_kcal: float
    strain_kcal: float
    force_field: str
    tag: str


@dataclass(frozen=True)
class AnchorCheckResult:
    """Canonical 3D pharmacophore anchor engagement summary."""

    engaged: bool
    any_contact: bool
    min_distance_angstrom: float | None
    hits: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize anchor engagement check to a JSON-compatible dictionary."""
        return {
            "engaged": self.engaged,
            "any_contact": self.any_contact,
            "min_distance_angstrom": self.min_distance_angstrom,
            "hits": [dict(h) for h in self.hits],
        }


@dataclass
class DockingPoseResult:
    """Comprehensive AutoDock Vina docking, PoseBusters, and 3D PLIF result for one ligand."""

    smiles: str
    smiles_ph74: str
    success: bool
    vina_affinity: float = float("nan")
    all_affinities: list[float] = field(default_factory=list)
    ligand_efficiency: float = float("nan")
    lle_vina: float = float("nan")
    n_heavy: int = 0
    clogp: float = float("nan")
    mw: float = float("nan")
    net_charge_ph74: int = 0
    mmff_strain_kcal: float = float("nan")
    n_ring_confs: int = 0
    n_modes: int = 0
    docked_mol: Chem.Mol | None = None
    all_modes: list[tuple[float, Chem.Mol]] = field(default_factory=list)
    posebusters: PoseBustersChecks | None = None
    interactions: InteractionProfile | None = None
    plif_tanimoto: float | None = None
    plif_tanimoto_polar: float | None = None
    residue_recovery: float | None = None
    anchor_top1: AnchorCheckResult | None = None
    anchor_any_mode: AnchorCheckResult | None = None
    anchor_mode_rank: int | None = None
    anchor_vina_affinity: float | None = None
    anchor_mol: Chem.Mol | None = None
    box: OrthostericBox | None = None
    error: str | None = None

    @property
    def preferred_mol(self) -> Chem.Mol | None:
        """Return anchor-verified pose if available, otherwise Top-1 docked pose."""
        return self.anchor_mol if self.anchor_mol is not None else self.docked_mol

    def save_sdf(self, out_path: str | Path, *, prefer_anchor_pose: bool = True) -> Path:
        """Write the docked 3D pose with annotated metadata to an SDF file."""
        mol = self.preferred_mol if prefer_anchor_pose else self.docked_mol
        if mol is None or mol.GetNumConformers() == 0:
            raise ValueError("Cannot save SDF: no valid 3D docked pose is available.")
        mol_copy = Chem.Mol(mol)
        mol_copy.SetProp("SMILES", self.smiles)
        mol_copy.SetProp("SMILES_PH74", self.smiles_ph74)
        if math.isfinite(self.vina_affinity):
            mol_copy.SetProp("VINA_DG_KCAL_MOL", f"{self.vina_affinity:.3f}")
        if math.isfinite(self.ligand_efficiency):
            mol_copy.SetProp("LIGAND_EFFICIENCY", f"{self.ligand_efficiency:.4f}")
        if self.posebusters is not None:
            mol_copy.SetProp("POSEBUSTERS_PASSED", str(bool(self.posebusters.passed_all)))
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(dest))
        writer.write(mol_copy)
        writer.close()
        return dest

    def to_dict(self) -> dict[str, Any]:
        """Serialize docking summary (without raw RDKit Mol objects) to a plain dict."""
        return {
            "smiles": self.smiles,
            "smiles_ph74": self.smiles_ph74,
            "success": self.success,
            "vina_affinity": round(self.vina_affinity, 3)
            if math.isfinite(self.vina_affinity)
            else None,
            "all_affinities": [round(float(a), 3) for a in self.all_affinities],
            "ligand_efficiency": round(self.ligand_efficiency, 4)
            if math.isfinite(self.ligand_efficiency)
            else None,
            "lle_vina": round(self.lle_vina, 3) if math.isfinite(self.lle_vina) else None,
            "n_heavy": self.n_heavy,
            "clogp": round(self.clogp, 3) if math.isfinite(self.clogp) else None,
            "mw": round(self.mw, 2) if math.isfinite(self.mw) else None,
            "net_charge_ph74": self.net_charge_ph74,
            "mmff_strain_kcal": round(self.mmff_strain_kcal, 3)
            if math.isfinite(self.mmff_strain_kcal)
            else None,
            "n_ring_confs": self.n_ring_confs,
            "n_modes": self.n_modes,
            "posebusters": self.posebusters.to_dict() if self.posebusters is not None else None,
            "interactions": self.interactions.to_dict() if self.interactions is not None else None,
            "plif_tanimoto": self.plif_tanimoto,
            "plif_tanimoto_polar": self.plif_tanimoto_polar,
            "residue_recovery": self.residue_recovery,
            "anchor_top1": self.anchor_top1.to_dict() if self.anchor_top1 is not None else None,
            "anchor_any_mode": (
                self.anchor_any_mode.to_dict() if self.anchor_any_mode is not None else None
            ),
            "anchor_mode_rank": self.anchor_mode_rank,
            "anchor_vina_affinity": self.anchor_vina_affinity,
            "box": self.box.to_dict() if self.box is not None else None,
            "error": self.error,
        }


@dataclass
class RedockingSeedResult:
    """Single-seed unbiased native redocking metrics."""

    seed: int
    success: bool
    top1_rmsd: float = float("nan")
    best3_rmsd: float = float("nan")
    best_all_rmsd: float = float("nan")
    top1_core_rmsd: float | None = None
    best3_core_rmsd: float | None = None
    top1_vina_affinity: float = float("nan")
    top1_le: float = float("nan")
    top1_posebusters_passed: bool = False
    top1_plif_tanimoto: float = float("nan")
    best3_plif_tanimoto: float = float("nan")
    top1_anchor_recovered: bool = False
    best3_anchor_recovered: bool = False
    n_ring_confs: int = 0
    n_modes: int = 0
    top1_mol: Chem.Mol | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize seed redocking metrics to a JSON-compatible dictionary."""
        return {
            "seed": self.seed,
            "success": self.success,
            "top1_rmsd": round(self.top1_rmsd, 3) if math.isfinite(self.top1_rmsd) else None,
            "best3_rmsd": round(self.best3_rmsd, 3) if math.isfinite(self.best3_rmsd) else None,
            "best_all_rmsd": (
                round(self.best_all_rmsd, 3) if math.isfinite(self.best_all_rmsd) else None
            ),
            "top1_core_rmsd": (
                round(self.top1_core_rmsd, 3) if self.top1_core_rmsd is not None else None
            ),
            "best3_core_rmsd": (
                round(self.best3_core_rmsd, 3) if self.best3_core_rmsd is not None else None
            ),
            "top1_vina_affinity": (
                round(self.top1_vina_affinity, 3)
                if math.isfinite(self.top1_vina_affinity)
                else None
            ),
            "top1_le": round(self.top1_le, 4) if math.isfinite(self.top1_le) else None,
            "top1_posebusters_passed": self.top1_posebusters_passed,
            "top1_plif_tanimoto": (
                round(self.top1_plif_tanimoto, 4)
                if math.isfinite(self.top1_plif_tanimoto)
                else None
            ),
            "best3_plif_tanimoto": (
                round(self.best3_plif_tanimoto, 4)
                if math.isfinite(self.best3_plif_tanimoto)
                else None
            ),
            "top1_anchor_recovered": self.top1_anchor_recovered,
            "best3_anchor_recovered": self.best3_anchor_recovered,
            "n_ring_confs": self.n_ring_confs,
            "n_modes": self.n_modes,
            "error": self.error,
        }


@dataclass
class RedockingBenchmarkResult:
    """Multi-seed Gate 1 unbiased native redocking validation summary."""

    pocket_pdb: str
    native_ligand_sdf: str
    canonical_smiles: str
    smiles_ph74: str
    box: OrthostericBox
    seeds: tuple[int, ...]
    seed_results: list[RedockingSeedResult]
    median_top1_rmsd: float
    median_best3_rmsd: float
    median_best_all_rmsd: float
    median_vina_affinity: float
    median_le: float
    mean_top1_plif_tanimoto: float
    top1_posebusters_pass_rate: float
    top1_anchor_recovery_rate: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize redocking benchmark summary to a JSON-compatible dictionary."""
        return {
            "pocket_pdb": self.pocket_pdb,
            "native_ligand_sdf": self.native_ligand_sdf,
            "canonical_smiles": self.canonical_smiles,
            "smiles_ph74": self.smiles_ph74,
            "box": self.box.to_dict(),
            "seeds": list(self.seeds),
            "median_top1_rmsd": round(self.median_top1_rmsd, 3),
            "median_best3_rmsd": round(self.median_best3_rmsd, 3),
            "median_best_all_rmsd": round(self.median_best_all_rmsd, 3),
            "median_vina_affinity": round(self.median_vina_affinity, 3),
            "median_le": round(self.median_le, 4),
            "mean_top1_plif_tanimoto": round(self.mean_top1_plif_tanimoto, 4),
            "top1_posebusters_pass_rate": round(self.top1_posebusters_pass_rate, 4),
            "top1_anchor_recovery_rate": round(self.top1_anchor_recovery_rate, 4),
            "status": self.status,
            "seeds_detail": [sr.to_dict() for sr in self.seed_results],
        }


@dataclass
class CrossTargetSelectivityResult:
    """Cross-target ΔΔG docking matrix & diagonal-enrichment permutation test."""

    target_names: tuple[str, ...]
    mean_dg_matrix: list[list[float]]
    col_zscore_matrix: list[list[float]]
    row_zscore_matrix: list[list[float]]
    double_centered_z_matrix: list[list[float]]
    diagonal_mean_dg: float
    off_diagonal_mean_dg: float
    raw_permutation_p: float
    col_z_permutation_p: float
    double_centered_permutation_p: float
    n_permutations: int
    per_target: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize selectivity matrix analysis to a JSON-compatible dictionary."""
        return {
            "target_names": list(self.target_names),
            "mean_dg_matrix": self.mean_dg_matrix,
            "col_zscore_matrix": self.col_zscore_matrix,
            "row_zscore_matrix": self.row_zscore_matrix,
            "double_centered_z_matrix": self.double_centered_z_matrix,
            "diagonal_mean_dg": round(self.diagonal_mean_dg, 4),
            "off_diagonal_mean_dg": round(self.off_diagonal_mean_dg, 4),
            "raw_permutation_p": round(self.raw_permutation_p, 6),
            "col_z_permutation_p": round(self.col_z_permutation_p, 6),
            "double_centered_permutation_p": round(self.double_centered_permutation_p, 6),
            "n_permutations": self.n_permutations,
            "per_target": self.per_target,
        }


# ---------------------------------------------------------------------------
# Universal Orthosteric Box Computation
# ---------------------------------------------------------------------------


def _load_3d_ligand(ligand_or_path: str | Path | Chem.Mol) -> Chem.Mol:
    """Load a 3D ligand RDKit Mol from an SDF/MOL2/PDB path or existing RDKit Mol."""
    if isinstance(ligand_or_path, Chem.Mol):
        if ligand_or_path.GetNumConformers() == 0:
            raise ValueError("Supplied RDKit Mol has no 3D conformer.")
        return ligand_or_path

    path = Path(ligand_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"3D ligand structure file not found: {path}")

    suffix = path.suffix.lower()
    mol: Chem.Mol | None = None
    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        mol = suppl[0] if len(suppl) > 0 else None
        if mol is None:
            raw_suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
            mol = raw_suppl[0] if len(raw_suppl) > 0 else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    else:
        raise ValueError(f"Unsupported 3D ligand format {suffix!r}; expected .sdf, .mol2, or .pdb")

    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Failed to read 3D coordinates from ligand file: {path}")
    return mol


def compute_orthosteric_box(
    ligand_or_path: str | Path | Chem.Mol,
    *,
    mode: BoxMode = "tight_per_axis",
    padding: float | None = None,
    min_edge: float | None = None,
    max_edge: float | None = None,
) -> OrthostericBox:
    """Compute a standardized AutoDock Vina orthosteric docking box from a 3D ligand.

    Two audited protocols are supported:

    - ``mode="tight_per_axis"`` (Iteration 3 universal default):
      ``size_i = clamp(span_i + 6.0, 14.0, 22.0)`` Å independently per Cartesian axis,
      centered on the heavy-atom geometric centroid. Prevents cut-pocket exterior
      lipid-groove escape while ensuring matched orthosteric volumes.
    - ``mode="cubic_standard"`` (Gate 1 native redocking standard):
      ``edge = clamp(max(span_xyz) + 8.0, 20.0, 26.0)`` Å cubic box ``(edge, edge, edge)``.

    Args:
        ligand_or_path: Path to a 3D ligand file (``.sdf``, ``.mol2``, ``.pdb``) or an
            RDKit ``Mol`` with a 3D conformer.
        mode: Box sizing protocol (``"tight_per_axis"`` or ``"cubic_standard"``).
        padding: Optional padding override in Ångströms.
        min_edge: Optional minimum box axis length override in Ångströms.
        max_edge: Optional maximum box axis length override in Ångströms.

    Returns:
        :class:`OrthostericBox` specifying ``center``, ``size``, and ``native_span_xyz``.
    """
    mol = _load_3d_ligand(ligand_or_path)
    conf = mol.GetConformer(0)
    heavy_pts = [
        list(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]
    if not heavy_pts:
        heavy_pts = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    coords = np.asarray(heavy_pts, dtype=np.float64)

    center_arr = coords.mean(axis=0)
    span_arr = coords.max(axis=0) - coords.min(axis=0)
    center = (float(center_arr[0]), float(center_arr[1]), float(center_arr[2]))
    span_xyz = (float(span_arr[0]), float(span_arr[1]), float(span_arr[2]))

    if mode == "tight_per_axis":
        pad = 6.0 if padding is None else float(padding)
        lo = 14.0 if min_edge is None else float(min_edge)
        hi = 22.0 if max_edge is None else float(max_edge)
        clipped = np.clip(span_arr + pad, lo, hi)
        size = (float(clipped[0]), float(clipped[1]), float(clipped[2]))
    elif mode == "cubic_standard":
        pad = 8.0 if padding is None else float(padding)
        lo = 20.0 if min_edge is None else float(min_edge)
        hi = 26.0 if max_edge is None else float(max_edge)
        edge = float(min(hi, max(lo, float(np.max(span_arr)) + pad)))
        size = (edge, edge, edge)
    else:
        raise ValueError(
            f"Unknown box mode {mode!r}; expected 'tight_per_axis' or 'cubic_standard'"
        )

    return OrthostericBox(center=center, size=size, native_span_xyz=span_xyz, mode=mode)


# ---------------------------------------------------------------------------
# Unbiased 2D-to-3D Ring- & Invertomer-Diverse Conformer Generation
# ---------------------------------------------------------------------------


def _extract_fused_ring_systems(mol_h: Chem.Mol) -> list[list[int]]:
    """Group ring heavy atoms into fused/spiro ring systems with local stereo probes.

    Includes explicit polar hydrogens attached to protonated ring nitrogens (``[NH+;R]``)
    to distinguish axial vs. equatorial ammonium invertomers, and immediate exocyclic
    neighbors of non-aromatic ring atoms to separate chair/boat puckers and tricyclic
    butterfly folds.
    """
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


def _so3_kabsch_rmsd(pts_a: NDArray[np.float64], pts_b: NDArray[np.float64]) -> float:
    """Compute proper ``SO(3)`` (reflection-free, ``det = +1``) Kabsch RMSD."""
    a_c = pts_a - pts_a.mean(axis=0, keepdims=True)
    b_c = pts_b - pts_b.mean(axis=0, keepdims=True)
    u_mat, _, vt_mat = np.linalg.svd(a_c.T @ b_c)
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
    """Maximum proper-``SO(3)`` internal Kabsch RMSD across fused ring systems."""
    conf_a = mol_h.GetConformer(cid_a)
    conf_b = mol_h.GetConformer(cid_b)
    max_rmsd = 0.0
    for idxs in ring_systems:
        pts_a = np.asarray([list(conf_a.GetAtomPosition(i)) for i in idxs], dtype=np.float64)
        pts_b = np.asarray([list(conf_b.GetAtomPosition(i)) for i in idxs], dtype=np.float64)
        max_rmsd = max(max_rmsd, _so3_kabsch_rmsd(pts_a, pts_b))
    return max_rmsd


def generate_ring_diverse_conformers(
    smiles_or_mol: str | Chem.Mol,
    *,
    protonate_ph74: bool = True,
    seed: int = 42,
    num_confs: int = 24,
    max_ring_confs: int = 4,
    ring_rmsd_cutoff: float = 0.28,
    energy_window_kcal: float = 6.0,
    max_iters: int = 350,
) -> list[RingConformer]:
    """Generate unbiased low-energy 3D conformers covering distinct ring/invertomer states.

    AutoDock Vina samples acyclic torsions freely while holding aliphatic rings, tricyclic
    butterfly folds, and pyramidal ``[NH+;R]`` invertomers rigid. Starting strictly from 2D
    connectivity, this routine embeds ``num_confs`` conformers with ``ETKDGv3``, minimizes
    every conformer with ``MMFF94s`` (``UFF`` fallback), and selects up to ``max_ring_confs``
    lowest-energy conformers separated by at least ``ring_rmsd_cutoff`` Å internal proper
    ``SO(3)`` ring-system Kabsch RMSD within ``energy_window_kcal`` of the global minimum.

    Args:
        smiles_or_mol: Input SMILES string or RDKit ``Mol`` (any existing 3D coordinates are
            discarded to guarantee unbiased 2D-to-3D generation).
        protonate_ph74: Whether to apply deterministic physiological pH 7.4 protonation before
            adding explicit hydrogens.
        seed: Deterministic RNG seed for RDKit ``ETKDGv3``.
        num_confs: Number of initial 3D conformers embedded prior to ring clustering.
        max_ring_confs: Maximum number of distinct ring/invertomer conformers returned.
        ring_rmsd_cutoff: Minimum internal proper-``SO(3)`` ring-system Kabsch RMSD (Å).
        energy_window_kcal: Maximum MMFF94s/UFF energy window (kcal/mol) above minimum.
        max_iters: Force-field minimization iteration cap per conformer.

    Returns:
        List of :class:`RingConformer` instances sorted by ascending force-field energy.

    Raises:
        ValueError: If the input SMILES or ``Mol`` is invalid.
    """
    if isinstance(smiles_or_mol, str):
        mol_2d = Chem.MolFromSmiles(smiles_or_mol.strip())
        if mol_2d is None:
            raise ValueError(f"Invalid SMILES: {smiles_or_mol!r}")
    else:
        mol_2d = Chem.RemoveHs(Chem.Mol(smiles_or_mol))
        mol_2d.RemoveAllConformers()

    mol_prep = protonate_mol_ph74(mol_2d, preserve_coords=False) if protonate_ph74 else mol_2d
    mol_h = Chem.AddHs(mol_prep)

    params = _ALLCHEM.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    params.numThreads = 1

    cids = [int(c) for c in _ALLCHEM.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)]
    if not cids:
        params.useRandomCoords = True
        cids = [
            int(c) for c in _ALLCHEM.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)
        ]
    if not cids:
        return []

    init_energies: dict[int, float] = {}
    energies: list[tuple[int, float]] = []
    force_field = "none"

    if _ALLCHEM.MMFFHasAllMoleculeParams(mol_h):
        mmff_props = _ALLCHEM.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if mmff_props is not None:
            force_field = "MMFF94s"
            for cid in cids:
                ff = _ALLCHEM.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=cid)
                if ff is not None:
                    e0 = float(ff.CalcEnergy())
                    ff.Minimize(maxIts=int(max_iters))
                    e1 = float(ff.CalcEnergy())
                    init_energies[cid] = e0
                    energies.append((cid, e1))

    if not energies and _ALLCHEM.UFFHasAllMoleculeParams(mol_h):
        force_field = "UFF"
        for cid in cids:
            ff = _ALLCHEM.UFFGetMoleculeForceField(mol_h, confId=cid)
            if ff is not None:
                e0 = float(ff.CalcEnergy())
                ff.Minimize(maxIts=int(max_iters))
                e1 = float(ff.CalcEnergy())
                init_energies[cid] = e0
                energies.append((cid, e1))

    if not energies:
        energies = [(cid, 0.0) for cid in cids]

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

    results: list[RingConformer] = []
    for cid, energy, tag in chosen:
        single = Chem.Mol(mol_h)
        single.RemoveAllConformers()
        single.AddConformer(mol_h.GetConformer(cid), assignId=True)
        strain = float(max(0.0, init_energies.get(cid, energy) - energy))
        results.append(
            RingConformer(
                mol_3d=single,
                energy_kcal=energy,
                strain_kcal=strain,
                force_field=force_field,
                tag=tag,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Murcko Core RMSD & Canonical Pharmacophore Anchor Analysis
# ---------------------------------------------------------------------------


def compute_murcko_core_rmsd(mol_ref: Chem.Mol, mol_probe: Chem.Mol) -> float | None:
    """Compute symmetry-corrected in-place heavy-atom RMSD over the Bemis-Murcko ring core.

    Distinguishes global binding-mode accuracy from flexible aliphatic tail drift (e.g.,
    xanomeline's hexyloxy chain in ``CHRM4 8FX5``). Returns ``None`` if the molecule has no
    distinct Murcko scaffold smaller than the full heavy-atom graph.
    """
    ref_noh = Chem.RemoveHs(Chem.Mol(mol_ref))
    probe_noh = Chem.RemoveHs(Chem.Mol(mol_probe))
    if ref_noh.GetNumConformers() == 0 or probe_noh.GetNumConformers() == 0:
        return None
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


def check_canonical_anchors(
    profile: InteractionProfile,
    canonical_anchors: Sequence[str],
    *,
    expected_kinds: Sequence[str] = ("salt_bridge", "hbond"),
) -> AnchorCheckResult:
    """Verify whether canonical pharmacological anchor residue(s) are engaged in 3D.

    Args:
        profile: Computed :class:`InteractionProfile` for a docked or crystal complex.
        canonical_anchors: Residue labels such as ``("ASP103",)`` or ``("GLN726", "TYR693")``.
        expected_kinds: Interaction channels qualifying as canonical anchor engagement.

    Returns:
        :class:`AnchorCheckResult` summarizing engagement status and shortest contact distance.
    """
    anchor_set = {a.strip().split(":")[0].upper() for a in canonical_anchors if a.strip()}
    allowed = {k.strip().lower() for k in expected_kinds}

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
            if rec.kind in allowed:
                polar_hits.append(hit)

    polar_hits.sort(key=lambda h: (0 if h["kind"] == "salt_bridge" else 1, h["distance_angstrom"]))
    min_dist = float(polar_hits[0]["distance_angstrom"]) if polar_hits else None
    return AnchorCheckResult(
        engaged=len(polar_hits) > 0,
        any_contact=len(any_hits) > 0,
        min_distance_angstrom=min_dist,
        hits=tuple(polar_hits[:4]),
    )


# ---------------------------------------------------------------------------
# Internal Vina Execution & Multi-Mode Reconstruction
# ---------------------------------------------------------------------------


def _require_meeko() -> Any:
    """Lazily import ``meeko`` or raise an actionable :class:`OptionalDependencyError`."""
    from geneuclidean.exceptions import OptionalDependencyError

    try:
        import meeko

        return meeko
    except ImportError as exc:
        raise OptionalDependencyError(
            "The optional 'meeko' package is required for AutoDock Vina ligand preparation "
            "and pose reconstruction.",
            hint="Install it via: pip install 'geneuclidean[docking]'",
        ) from exc


def _resolve_required_vina(vina_bin: str | Path) -> Path:
    """Resolve AutoDock Vina binary or raise an actionable :class:`OptionalDependencyError`."""
    from geneuclidean.exceptions import OptionalDependencyError

    candidates: list[str | Path] = [vina_bin]
    if str(vina_bin) != "vina":
        candidates.append("vina")
    for cand in candidates:
        found = _resolve_vina_binary(cand)
        if found is not None:
            return found
    raise OptionalDependencyError(
        f"AutoDock Vina executable not found (checked {vina_bin!r} and system PATH).",
        hint="Run `geneuclidean download --include-vina` or install AutoDock Vina v1.2+ on PATH.",
    )


def _reconstruct_all_vina_modes(
    out_pdbqt_text: str,
    affinities: list[float],
    template_mol_prot: Chem.Mol,
) -> list[tuple[float, Chem.Mol]]:
    """Reconstruct ALL docked Vina modes into individual 3D RDKit Mol objects."""
    modes: list[tuple[float, Chem.Mol]] = []
    try:
        meeko = _require_meeko()
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

    # Fallback: parse MODEL blocks and assign bond orders from template
    template_noh = Chem.RemoveHs(template_mol_prot)
    n_expected = template_noh.GetNumAtoms()
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
            mapped = _ALLCHEM.AssignBondOrdersFromTemplate(template_noh, raw_mol)
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
    """Run AutoDock Vina for one prepared 3D conformer and return reconstructed modes."""
    lig_pdbqt = prepare_ligand_pdbqt(mol_3d)
    if lig_pdbqt is None:
        return []

    with tempfile.TemporaryDirectory(prefix="geneuc_dock_") as tmp_dir:
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
            str(int(exhaustiveness)),
            "--num_modes",
            str(int(num_modes)),
            "--cpu",
            str(int(cpu)),
            "--seed",
            str(int(seed)),
            "--out",
            str(out_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return []
        if proc.returncode != 0 or not out_path.exists():
            return []
        out_text = out_path.read_text(encoding="utf-8", errors="ignore")
        affinities = _parse_vina_affinities(out_text, proc.stdout)
        if not affinities:
            return []
        return _reconstruct_all_vina_modes(out_text, affinities, template_mol_prot)


def _deduplicate_modes(
    modes: list[tuple[float, Chem.Mol]],
    *,
    rmsd_tol: float = 0.5,
    max_keep: int = 9,
) -> list[tuple[float, Chem.Mol]]:
    """Rank pooled conformer modes strictly by Vina ΔG and deduplicate near-identical poses."""
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


# ---------------------------------------------------------------------------
# Public High-Level Docking & Redocking Pipelines
# ---------------------------------------------------------------------------


def dock_ligand_to_pocket(
    smiles: str,
    pocket_pdb: str | Path,
    *,
    native_ligand_sdf: str | Path | None = None,
    box_center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
    box_mode: BoxMode = "tight_per_axis",
    canonical_anchors: Sequence[str] = (),
    expected_anchor_kinds: Sequence[str] = ("salt_bridge", "hbond"),
    protonate: bool = True,
    num_confs: int = 24,
    max_ring_confs: int = 2,
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 1,
    seed: int = 42,
    timeout: int = 180,
    receptor_pdbqt_path: str | Path | None = None,
    require_vina: bool = True,
) -> DockingPoseResult:
    """Dock a SMILES ligand into a 3D protein pocket with pH 7.4, PoseBusters & 3D PLIF.

    Args:
        smiles: Ligand SMILES string.
        pocket_pdb: Path to cleaned protein binding-pocket PDB file.
        native_ligand_sdf: Optional native co-crystal ligand SDF used both for automatic
            orthosteric box calculation and 3D PLIF Tanimoto reference comparison.
        box_center: Explicit docking box center ``(x, y, z)`` in Ångströms (optional when
            ``native_ligand_sdf`` is supplied; otherwise defaults to pocket heavy centroid).
        box_size: Explicit docking box dimensions ``(sx, sy, sz)`` in Ångströms.
        box_mode: Orthosteric box protocol when deriving box from ``native_ligand_sdf``.
        canonical_anchors: Optional residue identifiers (e.g., ``("ASP103",)``) audited on
            both Top-1 and all returned modes.
        expected_anchor_kinds: Interaction channels qualifying as anchor engagement.
        protonate: Whether to apply deterministic physiological pH 7.4 protonation.
        num_confs: Initial RDKit ``ETKDGv3`` conformers embedded prior to ring clustering.
        max_ring_confs: Maximum distinct ring/invertomer conformers docked and pooled.
        vina_bin: Path to AutoDock Vina executable (also checks ``PATH``).
        exhaustiveness: AutoDock Vina search exhaustiveness.
        num_modes: Maximum binding modes retained after multi-conformer deduplication.
        cpu: CPU threads passed to AutoDock Vina.
        seed: Deterministic random seed.
        timeout: Per-conformer Vina subprocess timeout in seconds.
        receptor_pdbqt_path: Optional pre-generated receptor ``.pdbqt`` path to reuse.
        require_vina: If ``True`` (default), raises :class:`RuntimeError` immediately when
            AutoDock Vina or ``meeko`` is unavailable; if ``False``, returns a failed
            :class:`DockingPoseResult` with ``error`` populated.

    Returns:
        :class:`DockingPoseResult` with Top-1 and anchor-verified poses, Vina ``ΔG``,
        ``LE``, ``LLE_Vina``, PoseBusters audit, and 3D PLIF metrics.

    Raises:
        FileNotFoundError: If ``pocket_pdb`` does not exist.
        RuntimeError: If ``require_vina=True`` and AutoDock Vina or ``meeko`` is unavailable.
    """
    pocket_path = Path(pocket_pdb)
    if not pocket_path.is_file():
        raise FileNotFoundError(f"Protein pocket PDB file not found: {pocket_path}")

    try:
        _require_meeko()
        vina_exe = _resolve_required_vina(vina_bin)
    except RuntimeError as exc:
        if require_vina:
            raise
        return DockingPoseResult(smiles=smiles, smiles_ph74=smiles, success=False, error=str(exc))

    mol_2d = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol_2d is None:
        return DockingPoseResult(
            smiles=str(smiles), smiles_ph74="", success=False, error="Invalid SMILES string"
        )

    canon_smiles = str(Chem.MolToSmiles(mol_2d, canonical=True, isomericSmiles=True))
    mol_prot = protonate_mol_ph74(mol_2d, preserve_coords=False) if protonate else Chem.Mol(mol_2d)
    smiles_ph74 = str(Chem.MolToSmiles(mol_prot, canonical=True, isomericSmiles=True))
    n_heavy = int(mol_prot.GetNumHeavyAtoms())
    clogp = float(_DESCRIPTORS.MolLogP(mol_2d))
    mw = float(_DESCRIPTORS.MolWt(mol_2d))
    net_charge = int(Chem.GetFormalCharge(mol_prot))

    # Determine orthosteric search box
    native_plif: InteractionProfile | None = None
    computed_box: OrthostericBox
    if native_ligand_sdf is not None and Path(native_ligand_sdf).is_file():
        nat_mol = _load_3d_ligand(native_ligand_sdf)
        computed_box = compute_orthosteric_box(nat_mol, mode=box_mode)
        nat_prot = protonate_mol_ph74(nat_mol, preserve_coords=True) if protonate else nat_mol
        native_plif = profile_3d_interactions(pocket_path, nat_prot)
        eff_center = box_center if box_center is not None else computed_box.center
        eff_size = box_size if box_size is not None else computed_box.size
        computed_box = OrthostericBox(
            center=eff_center,
            size=eff_size,
            native_span_xyz=computed_box.native_span_xyz,
            mode=box_mode,
        )
    else:
        if box_center is None:
            p_coords, _ = extract_pocket_heavy_atoms(pocket_path)
            if len(p_coords) == 0:
                return DockingPoseResult(
                    smiles=canon_smiles,
                    smiles_ph74=smiles_ph74,
                    success=False,
                    error="No heavy protein atoms found in pocket PDB",
                )
            c_arr = p_coords.mean(axis=0)
            eff_center = (float(c_arr[0]), float(c_arr[1]), float(c_arr[2]))
        else:
            eff_center = box_center
        eff_size = box_size if box_size is not None else (20.0, 20.0, 20.0)
        computed_box = OrthostericBox(
            center=eff_center,
            size=eff_size,
            native_span_xyz=(0.0, 0.0, 0.0),
            mode=box_mode,
        )

    conformers = generate_ring_diverse_conformers(
        mol_prot,
        protonate_ph74=False,
        seed=seed,
        num_confs=num_confs,
        max_ring_confs=max_ring_confs,
    )
    if not conformers:
        return DockingPoseResult(
            smiles=canon_smiles,
            smiles_ph74=smiles_ph74,
            success=False,
            n_heavy=n_heavy,
            clogp=clogp,
            mw=mw,
            net_charge_ph74=net_charge,
            box=computed_box,
            error="ETKDGv3 3D conformer generation failed",
        )

    min_strain = float(conformers[0].strain_kcal)
    pooled: list[tuple[float, Chem.Mol]] = []

    with tempfile.TemporaryDirectory(prefix="geneuc_rec_") as rec_tmp:
        if receptor_pdbqt_path is not None and Path(receptor_pdbqt_path).is_file():
            rec_pdbqt = Path(receptor_pdbqt_path)
        else:
            rec_prepared = prepare_receptor_pdbqt(pocket_path, Path(rec_tmp) / "receptor.pdbqt")
            if rec_prepared is None:
                return DockingPoseResult(
                    smiles=canon_smiles,
                    smiles_ph74=smiles_ph74,
                    success=False,
                    n_heavy=n_heavy,
                    clogp=clogp,
                    mw=mw,
                    net_charge_ph74=net_charge,
                    mmff_strain_kcal=min_strain,
                    n_ring_confs=len(conformers),
                    box=computed_box,
                    error="Receptor PDBQT preparation failed",
                )
            rec_pdbqt = rec_prepared

        for conf in conformers:
            modes = _run_vina_single_conformer(
                conf.mol_3d,
                mol_prot,
                rec_pdbqt,
                computed_box.center,
                computed_box.size,
                vina_exe=vina_exe,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                cpu=cpu,
                seed=seed,
                timeout=timeout,
            )
            pooled.extend(modes)

    ranked = _deduplicate_modes(pooled, rmsd_tol=0.5, max_keep=num_modes)
    if not ranked:
        return DockingPoseResult(
            smiles=canon_smiles,
            smiles_ph74=smiles_ph74,
            success=False,
            n_heavy=n_heavy,
            clogp=clogp,
            mw=mw,
            net_charge_ph74=net_charge,
            mmff_strain_kcal=min_strain,
            n_ring_confs=len(conformers),
            box=computed_box,
            error="AutoDock Vina returned no valid docked poses",
        )

    top1_aff, top1_mol = ranked[0]
    top1_pb = check_pose_geometry(top1_mol, pocket_pdb_path=pocket_path)
    top1_plif = profile_3d_interactions(pocket_path, top1_mol)
    top1_le = float(-top1_aff / max(n_heavy, 1))
    top1_lle = float((-top1_aff / RT_LN10_KCAL) - clogp)

    plif_tan: float | None = None
    plif_tan_polar: float | None = None
    res_rec: float | None = None
    if native_plif is not None:
        plif_tan = round(top1_plif.tanimoto_to(native_plif, include_hydrophobic=True), 4)
        plif_tan_polar = round(top1_plif.tanimoto_to(native_plif, include_hydrophobic=False), 4)
        res_rec = round(top1_plif.residue_recovery_rate(native_plif, include_hydrophobic=True), 4)

    anchor_top1: AnchorCheckResult | None = None
    anchor_any: AnchorCheckResult | None = None
    anchor_rank: int | None = None
    anchor_aff: float | None = None
    anchor_mol: Chem.Mol | None = None

    if canonical_anchors:
        anchor_top1 = check_canonical_anchors(
            top1_plif, canonical_anchors, expected_kinds=expected_anchor_kinds
        )
        if anchor_top1.engaged:
            anchor_any = anchor_top1
            anchor_rank = 1
            anchor_aff = float(top1_aff)
            anchor_mol = top1_mol
        else:
            anchor_any = anchor_top1
            for m_rank, (aff_m, mol_m) in enumerate(ranked[1:], start=2):
                plif_m = profile_3d_interactions(pocket_path, mol_m)
                chk_m = check_canonical_anchors(
                    plif_m, canonical_anchors, expected_kinds=expected_anchor_kinds
                )
                if chk_m.engaged:
                    anchor_any = chk_m
                    anchor_rank = m_rank
                    anchor_aff = float(aff_m)
                    anchor_mol = mol_m
                    break

    return DockingPoseResult(
        smiles=canon_smiles,
        smiles_ph74=smiles_ph74,
        success=True,
        vina_affinity=float(top1_aff),
        all_affinities=[float(a) for a, _ in ranked],
        ligand_efficiency=top1_le,
        lle_vina=top1_lle,
        n_heavy=n_heavy,
        clogp=clogp,
        mw=mw,
        net_charge_ph74=net_charge,
        mmff_strain_kcal=min_strain,
        n_ring_confs=len(conformers),
        n_modes=len(ranked),
        docked_mol=top1_mol,
        all_modes=ranked,
        posebusters=top1_pb,
        interactions=top1_plif,
        plif_tanimoto=plif_tan,
        plif_tanimoto_polar=plif_tan_polar,
        residue_recovery=res_rec,
        anchor_top1=anchor_top1,
        anchor_any_mode=anchor_any,
        anchor_mode_rank=anchor_rank,
        anchor_vina_affinity=anchor_aff,
        anchor_mol=anchor_mol,
        box=computed_box,
    )


def redock_native_complex(
    pocket_pdb: str | Path,
    native_ligand_sdf: str | Path,
    *,
    seeds: Sequence[int] = (42, 137, 2026),
    canonical_anchors: Sequence[str] = (),
    expected_anchor_kinds: Sequence[str] = ("salt_bridge", "hbond"),
    box_mode: BoxMode = "cubic_standard",
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 2,
    num_confs: int = 48,
    max_ring_confs: int = 4,
    timeout: int = 180,
) -> RedockingBenchmarkResult:
    """Run unbiased 2D-to-3D multi-seed native redocking validation on a crystal complex.

    Strictly strips all 3D coordinates from ``native_ligand_sdf`` (retaining only canonical
    isomeric SMILES), applies deterministic physiological pH 7.4 protonation, regenerates
    ring-diverse 3D conformers from scratch, docks with AutoDock Vina across ``seeds``, and
    measures symmetry-corrected heavy-atom RMSD, Murcko core RMSD, PoseBusters, 3D PLIF
    Tanimoto recovery, and canonical anchor recovery.

    Args:
        pocket_pdb: Path to cleaned ligand-free receptor pocket PDB file.
        native_ligand_sdf: Path to native co-crystal 3D ligand SDF file.
        seeds: Independent RNG seeds to evaluate.
        canonical_anchors: Optional anchor residues (e.g., ``("ASP103",)``) to audit.
        expected_anchor_kinds: Interaction channels qualifying as anchor recovery.
        box_mode: Box sizing protocol (``"cubic_standard"`` or ``"tight_per_axis"``).
        vina_bin: Path to AutoDock Vina executable.
        exhaustiveness: Vina exhaustiveness.
        num_modes: Modes returned per seed.
        cpu: CPU threads per Vina run.
        num_confs: Initial ``ETKDGv3`` conformers per seed.
        max_ring_confs: Maximum ring/invertomer conformers pooled per seed.
        timeout: Subprocess timeout in seconds.

    Returns:
        :class:`RedockingBenchmarkResult` aggregating all seed metrics and Gate 1 status.

    Raises:
        FileNotFoundError: If ``pocket_pdb`` or ``native_ligand_sdf`` is missing.
        RuntimeError: If AutoDock Vina or ``meeko`` is unavailable, or all seeds fail.
    """
    pocket_path = Path(pocket_pdb)
    sdf_path = Path(native_ligand_sdf)
    if not pocket_path.is_file():
        raise FileNotFoundError(f"Pocket PDB file not found: {pocket_path}")
    if not sdf_path.is_file():
        raise FileNotFoundError(f"Native ligand SDF file not found: {sdf_path}")

    _require_meeko()
    _resolve_required_vina(vina_bin)

    native_mol = _load_3d_ligand(sdf_path)
    canonical_smiles = extract_canonical_isomeric_smiles(native_mol)
    native_mol_ph74 = protonate_mol_ph74(native_mol, preserve_coords=True)
    mol_2d = Chem.MolFromSmiles(canonical_smiles)
    if mol_2d is None:
        raise ValueError(f"Round-trip SMILES failed for native ligand: {canonical_smiles!r}")
    smiles_ph74 = str(
        Chem.MolToSmiles(
            protonate_mol_ph74(mol_2d, preserve_coords=False), canonical=True, isomericSmiles=True
        )
    )
    box = compute_orthosteric_box(native_mol, mode=box_mode)
    native_plif = profile_3d_interactions(pocket_path, native_mol_ph74)

    seed_results: list[RedockingSeedResult] = []
    with tempfile.TemporaryDirectory(prefix="geneuc_redock_") as rec_tmp:
        rec_pdbqt = prepare_receptor_pdbqt(pocket_path, Path(rec_tmp) / "receptor.pdbqt")
        if rec_pdbqt is None:
            raise RuntimeError(f"Receptor PDBQT preparation failed for {pocket_path}")

        for seed in seeds:
            dock_res = dock_ligand_to_pocket(
                canonical_smiles,
                pocket_path,
                native_ligand_sdf=sdf_path,
                box_center=box.center,
                box_size=box.size,
                box_mode=box_mode,
                canonical_anchors=canonical_anchors,
                expected_anchor_kinds=expected_anchor_kinds,
                protonate=True,
                num_confs=num_confs,
                max_ring_confs=max_ring_confs,
                vina_bin=vina_bin,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                cpu=cpu,
                seed=int(seed),
                timeout=timeout,
                receptor_pdbqt_path=rec_pdbqt,
                require_vina=True,
            )
            if not dock_res.success or not dock_res.all_modes:
                seed_results.append(
                    RedockingSeedResult(
                        seed=int(seed),
                        success=False,
                        error=dock_res.error or "No valid docked poses returned",
                    )
                )
                continue

            rmsds = [
                compute_heavy_atom_rmsd(native_mol_ph74, m, align=False)
                for _, m in dock_res.all_modes
            ]
            top3_n = min(3, len(rmsds))
            best3_idx = int(min(range(top3_n), key=lambda i: rmsds[i]))
            best_all_idx = int(min(range(len(rmsds)), key=lambda i: rmsds[i]))

            top1_mol = dock_res.all_modes[0][1]
            best3_mol = dock_res.all_modes[best3_idx][1]
            top1_core = compute_murcko_core_rmsd(native_mol_ph74, top1_mol)
            best3_core = (
                top1_core
                if best3_idx == 0
                else compute_murcko_core_rmsd(native_mol_ph74, best3_mol)
            )

            top1_plif = dock_res.interactions or profile_3d_interactions(pocket_path, top1_mol)
            best3_plif = (
                top1_plif if best3_idx == 0 else profile_3d_interactions(pocket_path, best3_mol)
            )
            top1_anc = (
                check_canonical_anchors(
                    top1_plif, canonical_anchors, expected_kinds=expected_anchor_kinds
                ).engaged
                if canonical_anchors
                else True
            )
            best3_anc = (
                check_canonical_anchors(
                    best3_plif, canonical_anchors, expected_kinds=expected_anchor_kinds
                ).engaged
                if canonical_anchors
                else True
            )

            seed_results.append(
                RedockingSeedResult(
                    seed=int(seed),
                    success=True,
                    top1_rmsd=float(rmsds[0]),
                    best3_rmsd=float(rmsds[best3_idx]),
                    best_all_rmsd=float(rmsds[best_all_idx]),
                    top1_core_rmsd=top1_core,
                    best3_core_rmsd=best3_core,
                    top1_vina_affinity=float(dock_res.vina_affinity),
                    top1_le=float(dock_res.ligand_efficiency),
                    top1_posebusters_passed=bool(
                        dock_res.posebusters.passed_all if dock_res.posebusters else False
                    ),
                    top1_plif_tanimoto=float(
                        top1_plif.tanimoto_to(native_plif, include_hydrophobic=True)
                    ),
                    best3_plif_tanimoto=float(
                        best3_plif.tanimoto_to(native_plif, include_hydrophobic=True)
                    ),
                    top1_anchor_recovered=top1_anc,
                    best3_anchor_recovered=best3_anc,
                    n_ring_confs=dock_res.n_ring_confs,
                    n_modes=dock_res.n_modes,
                    top1_mol=top1_mol,
                )
            )

    ok = [sr for sr in seed_results if sr.success]
    if not ok:
        raise RuntimeError(f"All redocking seeds failed for {pocket_path.name}")

    med_top1 = float(np.median([r.top1_rmsd for r in ok]))
    med_best3 = float(np.median([r.best3_rmsd for r in ok]))
    med_all = float(np.median([r.best_all_rmsd for r in ok]))
    med_vina = float(np.median([r.top1_vina_affinity for r in ok]))
    med_le = float(np.median([r.top1_le for r in ok]))
    mean_plif = float(np.mean([r.top1_plif_tanimoto for r in ok]))
    pb_rate = float(np.mean([1.0 if r.top1_posebusters_passed else 0.0 for r in ok]))
    anc_rate = float(np.mean([1.0 if r.top1_anchor_recovered else 0.0 for r in ok]))

    if med_top1 <= 2.0 and med_best3 <= 2.0:
        status = "PASS_TOP1"
    elif med_best3 <= 2.0:
        status = "PASS_TOP3"
    elif med_best3 <= 2.5:
        status = "BORDERLINE"
    else:
        status = "DIAGNOSTIC_WARN"

    return RedockingBenchmarkResult(
        pocket_pdb=str(pocket_path),
        native_ligand_sdf=str(sdf_path),
        canonical_smiles=canonical_smiles,
        smiles_ph74=smiles_ph74,
        box=box,
        seeds=tuple(int(s) for s in seeds),
        seed_results=seed_results,
        median_top1_rmsd=med_top1,
        median_best3_rmsd=med_best3,
        median_best_all_rmsd=med_all,
        median_vina_affinity=med_vina,
        median_le=med_le,
        mean_top1_plif_tanimoto=mean_plif,
        top1_posebusters_pass_rate=pb_rate,
        top1_anchor_recovery_rate=anc_rate,
        status=status,
    )


# ---------------------------------------------------------------------------
# Cross-Target ΔΔG Selectivity Matrix & Statistical Permutation Engine
# ---------------------------------------------------------------------------


def compute_selectivity_ddg(
    on_target_dg: float,
    off_target_dgs: Sequence[float],
) -> dict[str, float]:
    """Compute thermodynamic binding selectivity ``ΔΔG`` and fold-selectivity ratio.

    Negative ``ΔΔG = ΔG_on - ΔG_off`` indicates preferred binding to the on-target pocket.
    Converts ``ΔΔG`` to an approximate thermodynamic ``Kd_off / Kd_on`` fold-selectivity
    ratio via ``10^(-ΔΔG / 1.364)`` at ``298.15 K``.

    Args:
        on_target_dg: On-target docking free energy (kcal/mol, more negative = stronger).
        off_target_dgs: Sequence of off-target docking free energies (kcal/mol).

    Returns:
        Dictionary containing ``on_target_dg``, ``mean_off_target_dg``,
        ``strongest_off_target_dg``, ``ddg_vs_mean``, ``ddg_vs_strongest_off_target``,
        and ``estimated_fold_selectivity_mean``.
    """
    off_arr = np.asarray([float(v) for v in off_target_dgs if math.isfinite(float(v))], dtype=float)
    if len(off_arr) == 0:
        raise ValueError("off_target_dgs must contain at least one finite value.")
    mean_off = float(np.mean(off_arr))
    best_off = float(np.min(off_arr))
    ddg_mean = float(on_target_dg - mean_off)
    ddg_worst = float(on_target_dg - best_off)
    fold_mean = float(10.0 ** np.clip(-ddg_mean / RT_LN10_KCAL, -12.0, 12.0))
    return {
        "on_target_dg": round(float(on_target_dg), 3),
        "mean_off_target_dg": round(mean_off, 3),
        "strongest_off_target_dg": round(best_off, 3),
        "ddg_vs_mean": round(ddg_mean, 3),
        "ddg_vs_strongest_off_target": round(ddg_worst, 3),
        "estimated_fold_selectivity_mean": round(fold_mean, 2),
    }


def _diagonal_permutation_p(
    mat: NDArray[np.float64],
    *,
    higher_is_better: bool,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Exact Monte Carlo column-label permutation test for diagonal selectivity enrichment."""
    n = mat.shape[0]
    obs_diag = float(np.nanmean(np.diag(mat)))
    idx_rows = np.arange(n)
    null_diag = np.empty(n_permutations, dtype=np.float64)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        null_diag[k] = float(np.nanmean(mat[idx_rows, perm]))

    if higher_is_better:
        exceed = int(np.sum(null_diag >= obs_diag - 1e-12))
    else:
        exceed = int(np.sum(null_diag <= obs_diag + 1e-12))
    return float((1 + exceed) / (n_permutations + 1))


def compute_cross_target_matrix(
    scores: ArrayLike | Mapping[tuple[str, str], Sequence[float] | float],
    target_names: Sequence[str] | None = None,
    *,
    n_permutations: int = 10000,
    seed: int = 42,
) -> CrossTargetSelectivityResult:
    """Analyze a ``T x T`` cross-target docking ``ΔG`` selectivity matrix.

    Row ``i`` corresponds to ligands designed for target ``i``; column ``j`` corresponds to
    docking into pocket ``j`` (more negative ``ΔG`` = stronger binding). Computes:

    1. **Column z-score selectivity** ``Z_col = -(ΔG - μ_col) / σ_col`` (positive = ligand
       binds pocket ``j`` stronger than competing ligands).
    2. **Row z-score selectivity** ``Z_row = -(ΔG - μ_row) / σ_row`` (positive = ligand
       prefers pocket ``j`` over other pockets).
    3. **Double-centered interaction residual** ``Z_dbl`` removing **both** pocket-size
       and ligand-MW main effects simultaneously:
       ``ΔΔG_int = -(ΔG - μ_row - μ_col + μ_grand)``.
    4. Exact ``n_permutations`` Monte Carlo diagonal-enrichment permutation p-values.

    Args:
        scores: Either a square ``(T, T)`` numeric array of mean Vina ``ΔG`` values, or a
            mapping ``{(design_target, pocket_target): dg_or_list_of_dgs}``.
        target_names: Ordered sequence of ``T`` target identifiers (inferred automatically
            when ``scores`` is a mapping, or defaulted to ``T0..T{n-1}`` for raw arrays).
        n_permutations: Number of random column-label permutations for p-value estimation.
        seed: Deterministic RNG seed for permutation testing.

    Returns:
        :class:`CrossTargetSelectivityResult` containing normalized matrices, diagonal vs.
        off-diagonal summary, permutation p-values, and per-target ranks.
    """
    if isinstance(scores, Mapping):
        if target_names is None:
            names = tuple(sorted({str(k[0]) for k in scores} | {str(k[1]) for k in scores}))
        else:
            names = tuple(str(t) for t in target_names)
        n = len(names)
        mat = np.full((n, n), np.nan, dtype=np.float64)
        for i, design_t in enumerate(names):
            for j, pocket_t in enumerate(names):
                val = scores.get((design_t, pocket_t))
                if val is None:
                    continue
                if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                    arr = [float(x) for x in val if math.isfinite(float(x))]
                    if arr:
                        mat[i, j] = float(np.mean(arr))
                else:
                    fv = float(val)
                    if math.isfinite(fv):
                        mat[i, j] = fv
    else:
        mat = np.asarray(scores, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 2:
            raise ValueError(f"Expected square (T, T) matrix with T >= 2, got shape {mat.shape}")
        n = mat.shape[0]
        if target_names is None:
            names = tuple(f"T{i}" for i in range(n))
        else:
            names = tuple(str(t) for t in target_names)
            if len(names) != n:
                raise ValueError(f"len(target_names) ({len(names)}) != matrix size ({n})")

    if n < 2:
        raise ValueError(f"Cross-target selectivity requires at least 2 targets, got {n}")
    if np.isnan(mat).all():
        raise ValueError("Cross-target matrix contains no finite docking scores.")

    # Fill isolated missing cells with grand mean for stable centering if sparse
    grand_mean = float(np.nanmean(mat))
    clean_mat = np.where(np.isfinite(mat), mat, grand_mean)

    col_mean = np.mean(clean_mat, axis=0, keepdims=True)
    col_std = np.std(clean_mat, axis=0, keepdims=True)
    z_col = -(clean_mat - col_mean) / np.where(col_std > 1e-8, col_std, 1.0)

    row_mean = np.mean(clean_mat, axis=1, keepdims=True)
    row_std = np.std(clean_mat, axis=1, keepdims=True)
    z_row = -(clean_mat - row_mean) / np.where(row_std > 1e-8, row_std, 1.0)

    interaction_dg = -(clean_mat - row_mean - col_mean + grand_mean)
    inter_std = float(np.std(interaction_dg))
    z_double = interaction_dg / max(inter_std, 1e-8)

    diag_vals = np.diag(clean_mat)
    off_mask = ~np.eye(n, dtype=bool)
    diag_mean = float(np.mean(diag_vals))
    off_mean = float(np.mean(clean_mat[off_mask]))

    rng = np.random.default_rng(int(seed))
    p_raw = _diagonal_permutation_p(
        clean_mat, higher_is_better=False, n_permutations=int(n_permutations), rng=rng
    )
    p_col = _diagonal_permutation_p(
        z_col, higher_is_better=True, n_permutations=int(n_permutations), rng=rng
    )
    p_dbl = _diagonal_permutation_p(
        z_double, higher_is_better=True, n_permutations=int(n_permutations), rng=rng
    )

    per_target: list[dict[str, Any]] = []
    for i, t_name in enumerate(names):
        col_rank = int(1 + np.sum(clean_mat[:, i] < clean_mat[i, i]))
        per_target.append(
            {
                "target": t_name,
                "diag_vina_dg": round(float(clean_mat[i, i]), 3),
                "col_mean_vina_dg": round(float(col_mean[0, i]), 3),
                "col_zscore": round(float(z_col[i, i]), 3),
                "double_centered_z": round(float(z_double[i, i]), 3),
                "double_centered_ddg_kcal": round(float(-interaction_dg[i, i]), 3),
                "col_rank": col_rank,
            }
        )

    return CrossTargetSelectivityResult(
        target_names=names,
        mean_dg_matrix=np.round(clean_mat, 3).tolist(),
        col_zscore_matrix=np.round(z_col, 3).tolist(),
        row_zscore_matrix=np.round(z_row, 3).tolist(),
        double_centered_z_matrix=np.round(z_double, 3).tolist(),
        diagonal_mean_dg=diag_mean,
        off_diagonal_mean_dg=off_mean,
        raw_permutation_p=p_raw,
        col_z_permutation_p=p_col,
        double_centered_permutation_p=p_dbl,
        n_permutations=int(n_permutations),
        per_target=per_target,
    )


__all__ = [
    "RT_LN10_KCAL",
    "AnchorCheckResult",
    "BoxMode",
    "CrossTargetSelectivityResult",
    "DockingPoseResult",
    "OrthostericBox",
    "RedockingBenchmarkResult",
    "RedockingSeedResult",
    "RingConformer",
    "check_canonical_anchors",
    "compute_cross_target_matrix",
    "compute_murcko_core_rmsd",
    "compute_orthosteric_box",
    "compute_selectivity_ddg",
    "dock_ligand_to_pocket",
    "generate_ring_diverse_conformers",
    "redock_native_complex",
]
