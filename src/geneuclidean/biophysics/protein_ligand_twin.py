"""Stateful 3D Protein–Pocket–Ligand Biophysical Digital Twin, Induced-Fit Relaxer,
Room-Temperature Langevin Ensemble Sampler, Generative Force Guidance & OpenMM Bridge.

Provides :class:`ProteinLigandDigitalTwin` on top of :mod:`geneuclidean.biophysics.forcefield`
for end-to-end physical and electrical modeling of 3D protein-ligand complexes in pure
``torch.float32`` (Apple Silicon MPS, CUDA, and CPU compatible; zero ``torch.float64``).
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch import Tensor

from geneuclidean.biophysics.forcefield import (
    KB_KCAL_MOL_K,
    BiophysicalAtomParams,
    BiophysicalHamiltonian,
    ComplexEnergyBreakdown,
    parameterize_ligand_biophysics,
    parameterize_pocket_biophysics,
)
from geneuclidean.evaluation.geometry3d import embed_and_minimize_3d
from geneuclidean.exceptions import (
    InvalidMoleculeError,
    OptionalDependencyError,
    PocketFeaturizationError,
    StructureNotFoundError,
)

# ---------------------------------------------------------------------------
# Public Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TwinPhysicsInspection:
    """Comprehensive biophysical & electrical diagnostic snapshot of a 3D complex."""

    energy_breakdown: ComplexEnergyBreakdown
    hotspot_table: list[dict[str, Any]]
    clash_diagnostics: dict[str, Any]
    max_ligand_force_norm: float
    rms_ligand_force: float
    max_ligand_efield_v_a: float
    min_intermolecular_distance_angstrom: float

    @property
    def u_total(self) -> float:
        """Total intermolecular potential energy (kcal/mol)."""
        return self.energy_breakdown.u_total

    @property
    def dg_bind_est(self) -> float:
        """Estimated binding free energy dG_bind (kcal/mol)."""
        return self.energy_breakdown.dg_bind_est

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete physics inspection report to a JSON-compatible dict."""
        return {
            "energy_breakdown": self.energy_breakdown.to_summary_dict(),
            "hotspot_table": list(self.hotspot_table),
            "clash_diagnostics": dict(self.clash_diagnostics),
            "max_ligand_force_norm": round(self.max_ligand_force_norm, 4),
            "rms_ligand_force": round(self.rms_ligand_force, 4),
            "max_ligand_efield_v_a": round(self.max_ligand_efield_v_a, 4),
            "min_intermolecular_distance_angstrom": round(
                self.min_intermolecular_distance_angstrom, 4
            ),
        }


@dataclass
class TwinRelaxationResult:
    """Summary of differentiable induced-fit 3D complex micro-relaxation."""

    initial_energy: float
    final_energy: float
    delta_energy: float
    initial_clash_count: int
    final_clash_count: int
    initial_min_dist_angstrom: float
    final_min_dist_angstrom: float
    ligand_rmsd_angstrom: float
    sidechain_rmsd_angstrom: float
    backbone_rmsd_angstrom: float
    steps_run: int
    converged: bool
    energy_trajectory: list[float] = field(default_factory=list)
    final_breakdown: ComplexEnergyBreakdown | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize relaxation summary to plain JSON-ready dictionary."""
        return {
            "initial_energy": round(self.initial_energy, 4),
            "final_energy": round(self.final_energy, 4),
            "delta_energy": round(self.delta_energy, 4),
            "initial_clash_count": int(self.initial_clash_count),
            "final_clash_count": int(self.final_clash_count),
            "initial_min_dist_angstrom": round(self.initial_min_dist_angstrom, 4),
            "final_min_dist_angstrom": round(self.final_min_dist_angstrom, 4),
            "ligand_rmsd_angstrom": round(self.ligand_rmsd_angstrom, 4),
            "sidechain_rmsd_angstrom": round(self.sidechain_rmsd_angstrom, 4),
            "backbone_rmsd_angstrom": round(self.backbone_rmsd_angstrom, 4),
            "steps_run": int(self.steps_run),
            "converged": bool(self.converged),
            "energy_trajectory": [round(v, 4) for v in self.energy_trajectory],
        }


@dataclass
class TwinEnsembleSummary:
    """Room-temperature overdamped Langevin / Brownian conformational ensemble metrics."""

    temperature_k: float
    n_frames: int
    ligand_rmsf: NDArray[np.float32]  # [N_lig] (A)
    pocket_sidechain_rmsf: NDArray[np.float32]  # [N_rec] (A)
    mean_ligand_rmsf: float
    mean_sidechain_rmsf: float
    mean_energy_kcal_mol: float
    std_energy_kcal_mol: float
    hbond_occupancy: dict[str, float]
    salt_bridge_occupancy: dict[str, float]
    halogen_sigma_hole_occupancy: dict[str, float]
    pi_interaction_occupancy: dict[str, float]
    interaction_persistence: dict[str, float]  # contact occupancy % in [0, 100]
    dynamic_stability_score: float  # [0, 1] kinetic stability index
    residence_time_proxy: float  # dimensionless log10(tau / tau_0) barrier proxy

    def to_dict(self) -> dict[str, Any]:
        """Serialize ensemble thermodynamics & persistence metrics to plain JSON dict."""
        return {
            "temperature_k": round(self.temperature_k, 2),
            "n_frames": int(self.n_frames),
            "mean_ligand_rmsf": round(self.mean_ligand_rmsf, 4),
            "mean_sidechain_rmsf": round(self.mean_sidechain_rmsf, 4),
            "mean_energy_kcal_mol": round(self.mean_energy_kcal_mol, 4),
            "std_energy_kcal_mol": round(self.std_energy_kcal_mol, 4),
            "hbond_occupancy": {k: round(v, 4) for k, v in self.hbond_occupancy.items()},
            "salt_bridge_occupancy": {
                k: round(v, 4) for k, v in self.salt_bridge_occupancy.items()
            },
            "halogen_sigma_hole_occupancy": {
                k: round(v, 4) for k, v in self.halogen_sigma_hole_occupancy.items()
            },
            "pi_interaction_occupancy": {
                k: round(v, 4) for k, v in self.pi_interaction_occupancy.items()
            },
            "interaction_persistence": {
                k: round(v, 2) for k, v in self.interaction_persistence.items()
            },
            "dynamic_stability_score": round(self.dynamic_stability_score, 4),
            "residence_time_proxy": round(self.residence_time_proxy, 4),
        }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _smooth_clip_vectors(vecs: Tensor, max_norm: float) -> Tensor:
    """Smoothly norm-bound 3D vectors ``[..., 3]`` while preserving exact ``SO(3)`` equivariance."""
    if max_norm <= 0.0:
        return vecs
    sq = (vecs * vecs).sum(dim=-1, keepdim=True)
    scale = torch.rsqrt(1.0 + sq / (max_norm * max_norm))
    return vecs * scale


def _load_ligand_mol(ligand_mol_or_path: Chem.Mol | str | Path) -> Chem.Mol:
    """Resolve an RDKit 3D ``Chem.Mol`` from a ``Mol``, SDF/MOL2/PDB path, or SMILES."""
    if isinstance(ligand_mol_or_path, Chem.Mol):
        return Chem.Mol(ligand_mol_or_path)

    if isinstance(ligand_mol_or_path, Path) or (
        isinstance(ligand_mol_or_path, str)
        and Path(ligand_mol_or_path).suffix.lower() in {".sdf", ".mol", ".mol2", ".pdb"}
    ):
        p = Path(ligand_mol_or_path)
        if not p.is_file():
            raise StructureNotFoundError(f"Ligand 3D file not found: {p}")
        suf = p.suffix.lower()
        mol: Chem.Mol | None = None
        if suf in {".sdf", ".mol"}:
            suppl = Chem.SDMolSupplier(str(p), removeHs=False)
            mol = suppl[0] if len(suppl) > 0 else None
        elif suf == ".mol2":
            mol = Chem.MolFromMol2File(str(p), removeHs=False)
        elif suf == ".pdb":
            mol = Chem.MolFromPDBFile(str(p), removeHs=False)
        if mol is None or mol.GetNumConformers() == 0:
            raise InvalidMoleculeError(f"Failed to read 3D ligand molecule from {p}")
        return mol

    if isinstance(ligand_mol_or_path, str):
        res = embed_and_minimize_3d(ligand_mol_or_path, max_iters=150, seed=42)
        if not res.success or res.mol_3d is None:
            raise InvalidMoleculeError(
                f"Failed to generate 3D ligand conformer from SMILES {ligand_mol_or_path!r}"
            )
        return res.mol_3d

    raise InvalidMoleculeError(f"Unsupported ligand input: {type(ligand_mol_or_path)!r}")


def _count_pauli_clashes(
    lig: BiophysicalAtomParams,
    rec: BiophysicalAtomParams,
    lig_coords: Tensor,
    rec_coords: Tensor,
    *,
    clash_ratio: float = 0.72,
) -> tuple[int, float, list[dict[str, Any]]]:
    """Count intermolecular Pauli steric clashes and return ``(count, min_dist, clash_rows)``."""
    if lig_coords.shape[0] == 0 or rec_coords.shape[0] == 0:
        return 0, float("nan"), []

    diff = lig_coords.unsqueeze(1) - rec_coords.unsqueeze(0)
    dists = torch.sqrt((diff * diff).sum(dim=-1) + 1e-10)

    # Focus clash audit on heavy-atom pairs (and polar hydrogens below 1.45 A)
    lig_heavy = lig.atomic_nums > 1
    rec_heavy = rec.atomic_nums > 1
    heavy_pair = lig_heavy.unsqueeze(1) & rec_heavy.unsqueeze(0)

    if bool(heavy_pair.any()):
        min_dist = float(dists[heavy_pair].min().item())
    else:
        min_dist = float(dists.min().item())

    r_vdw_l = 0.561231 * lig.vdw_sigma  # [NL]
    r_vdw_r = 0.561231 * rec.vdw_sigma  # [NP]
    thresh = clash_ratio * (r_vdw_l.unsqueeze(1) + r_vdw_r.unsqueeze(0))

    # H-bond donor-acceptor heavy pairs can physically approach down to ~2.50 A
    hb_pair = (lig.is_donor.unsqueeze(1) & rec.is_acceptor.unsqueeze(0)) | (
        lig.is_acceptor.unsqueeze(1) & rec.is_donor.unsqueeze(0)
    )
    eff_thresh = torch.where(hb_pair, torch.minimum(thresh, torch.full_like(thresh, 2.35)), thresh)

    clash_mask = heavy_pair & (dists < eff_thresh)
    clash_indices = torch.nonzero(clash_mask, as_tuple=False)
    rows: list[dict[str, Any]] = []
    rec_res_np = rec.residue_indices.detach().cpu().numpy()
    for pair in clash_indices[:25]:
        li = int(pair[0].item())
        rj = int(pair[1].item())
        r_label = (
            rec.residue_labels[int(rec_res_np[rj])]
            if int(rec_res_np[rj]) < len(rec.residue_labels)
            else f"RES{rj}"
        )
        rows.append(
            {
                "ligand_atom_idx": li,
                "ligand_atom": lig.atom_names[li] if li < len(lig.atom_names) else f"L{li}",
                "pocket_residue": r_label,
                "pocket_atom": rec.atom_names[rj] if rj < len(rec.atom_names) else f"P{rj}",
                "distance_angstrom": round(float(dists[li, rj].item()), 3),
                "threshold_angstrom": round(float(eff_thresh[li, rj].item()), 3),
            }
        )
    return int(clash_indices.shape[0]), min_dist, rows


# ---------------------------------------------------------------------------
# Stateful 3D Protein–Pocket–Ligand Digital Twin
# ---------------------------------------------------------------------------


class ProteinLigandDigitalTwin:
    """Stateful, differentiable 3D Digital Twin of a Protein Pocket + Ligand complex.

    Supports analytical biophysical inspection (:meth:`inspect_physics`), induced-fit
    all-atom coordinate micro-relaxation (:meth:`relax_complex`), room-temperature
    overdamped Langevin conformational ensemble sampling (:meth:`sample_thermal_ensemble`),
    and plug-and-play ``torch.float32`` force guidance (:meth:`guidance_force_fn`) for
    3D Flow Matching and Diffusion *de novo* molecular decoders.
    """

    def __init__(
        self,
        ligand_params: BiophysicalAtomParams,
        pocket_params: BiophysicalAtomParams,
        *,
        hamiltonian: BiophysicalHamiltonian | None = None,
        ligand_mol: Chem.Mol | None = None,
        pocket_pdb_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        dev = ligand_params.device
        self.ligand_params = ligand_params.to(dev)
        self.pocket_params = pocket_params.to(dev)
        self.hamiltonian = (
            hamiltonian.to(dev) if hamiltonian is not None else BiophysicalHamiltonian().to(dev)
        )
        self.ligand_mol = Chem.Mol(ligand_mol) if ligand_mol is not None else None
        self.pocket_pdb_text = pocket_pdb_text
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}

    @property
    def device(self) -> torch.device:
        """Active PyTorch compute device."""
        return self.ligand_params.device

    def to(self, device: torch.device | str) -> ProteinLigandDigitalTwin:
        """Transfer the entire digital twin and Hamiltonian onto ``device``."""
        dev = torch.device(device)
        return ProteinLigandDigitalTwin(
            self.ligand_params.to(dev),
            self.pocket_params.to(dev),
            hamiltonian=self.hamiltonian.to(dev),
            ligand_mol=self.ligand_mol,
            pocket_pdb_text=self.pocket_pdb_text,
            metadata=self.metadata,
        )

    def clone(self) -> ProteinLigandDigitalTwin:
        """Create a deep state copy of this digital twin."""
        return ProteinLigandDigitalTwin(
            self.ligand_params.with_coords(self.ligand_params.coords.clone()),
            self.pocket_params.with_coords(self.pocket_params.coords.clone()),
            hamiltonian=self.hamiltonian,
            ligand_mol=Chem.Mol(self.ligand_mol) if self.ligand_mol is not None else None,
            pocket_pdb_text=self.pocket_pdb_text,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_tensors(
        cls,
        ligand_params: BiophysicalAtomParams,
        pocket_params: BiophysicalAtomParams,
        *,
        hamiltonian: BiophysicalHamiltonian | None = None,
        ligand_mol: Chem.Mol | None = None,
        pocket_pdb_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProteinLigandDigitalTwin:
        """Construct a :class:`ProteinLigandDigitalTwin` directly from parameterized tensors."""
        return cls(
            ligand_params=ligand_params,
            pocket_params=pocket_params,
            hamiltonian=hamiltonian,
            ligand_mol=ligand_mol,
            pocket_pdb_text=pocket_pdb_text,
            metadata=metadata,
        )

    @classmethod
    def from_pdb_and_mol(
        cls,
        pocket_pdb: str | Path,
        ligand_mol: Chem.Mol | str | Path,
        *,
        hamiltonian: BiophysicalHamiltonian | None = None,
        keep_waters: bool = False,
        keep_hydrogens: bool = True,
        device: torch.device | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProteinLigandDigitalTwin:
        """Instantiate a digital twin from a pocket PDB (file or text) and 3D RDKit ligand."""
        mol = _load_ligand_mol(ligand_mol)
        has_h = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
        if keep_hydrogens and not has_h:
            mol = Chem.AddHs(mol, addCoords=True)
        elif not keep_hydrogens and has_h:
            mol = Chem.RemoveHs(mol)

        pdb_text: str | None = None
        if isinstance(pocket_pdb, Path) or (
            isinstance(pocket_pdb, str)
            and "\n" not in pocket_pdb
            and not pocket_pdb.lstrip().startswith(("ATOM", "HETATM", "HEADER", "REMARK"))
        ):
            p = Path(pocket_pdb)
            if not p.is_file():
                raise StructureNotFoundError(f"Pocket PDB file not found: {p}")
            pdb_text = p.read_text(encoding="utf-8", errors="ignore")
        else:
            pdb_text = str(pocket_pdb)

        lig_params = parameterize_ligand_biophysics(
            mol, keep_hydrogens=keep_hydrogens, device=device
        )
        rec_params = parameterize_pocket_biophysics(
            pdb_text,
            keep_waters=keep_waters,
            keep_hydrogens=keep_hydrogens,
            device=device,
        )
        meta = dict(metadata) if metadata is not None else {}
        meta.setdefault("smiles", Chem.MolToSmiles(Chem.RemoveHs(mol)))
        return cls(
            ligand_params=lig_params,
            pocket_params=rec_params,
            hamiltonian=hamiltonian,
            ligand_mol=mol,
            pocket_pdb_text=pdb_text,
            metadata=meta,
        )

    @classmethod
    def from_pocket_and_smiles(
        cls,
        pocket: Any,
        smiles: str,
        *,
        center: Sequence[float] | None = None,
        seed: int = 42,
        hamiltonian: BiophysicalHamiltonian | None = None,
        keep_waters: bool = False,
        keep_hydrogens: bool = True,
        device: torch.device | str | None = None,
    ) -> ProteinLigandDigitalTwin:
        """Build a 3D complex twin from a ``Pocket`` (or PDB path/text) and a SMILES string.

        Generates a low-strain ETKDGv3 + MMFF94s 3D conformer and positions its centroid
        at ``center`` (or the pocket's orthosteric box / geometric cavity centroid).
        """
        conf_res = embed_and_minimize_3d(smiles, max_iters=200, seed=seed)
        if not conf_res.success or conf_res.mol_3d is None:
            raise InvalidMoleculeError(
                f"Failed to embed 3D conformer for SMILES {smiles!r}: {conf_res.error}"
            )
        mol = conf_res.mol_3d
        if not keep_hydrogens:
            mol = Chem.RemoveHs(mol)

        pdb_source: str | Path
        meta: dict[str, Any] = {"smiles": smiles}
        inferred_center: tuple[float, float, float] | None = None

        if hasattr(pocket, "pocket_pdb"):
            p_path = getattr(pocket, "pocket_pdb", None)
            if p_path is None or not Path(p_path).is_file():
                raise StructureNotFoundError(
                    "Pocket has no local PDB structure file attached.",
                    hint="Use Pocket.from_pdb(...) or Pocket.from_target(..., allow_fetch=True).",
                )
            pdb_source = Path(p_path)
            meta["target_id"] = getattr(pocket, "target_id", None)
            meta["pdb_id"] = getattr(pocket, "pdb_id", None)
            if center is None and hasattr(pocket, "orthosteric_box"):
                try:
                    box = pocket.orthosteric_box()
                    inferred_center = (
                        float(box.center[0]),
                        float(box.center[1]),
                        float(box.center[2]),
                    )
                except Exception:
                    inferred_center = None
        elif isinstance(pocket, (str, Path)):
            pdb_source = pocket
        else:
            raise PocketFeaturizationError(f"Unsupported pocket argument: {type(pocket)!r}")

        rec_params = parameterize_pocket_biophysics(
            pdb_source,
            keep_waters=keep_waters,
            keep_hydrogens=keep_hydrogens,
            device=device,
        )

        target_center_t: Tensor
        if center is not None:
            if len(center) != 3:
                raise PocketFeaturizationError("center must contain exactly 3 coordinates.")
            target_center_t = torch.tensor(
                [float(center[0]), float(center[1]), float(center[2])],
                dtype=torch.float32,
                device=rec_params.device,
            )
        elif inferred_center is not None:
            target_center_t = torch.tensor(
                list(inferred_center), dtype=torch.float32, device=rec_params.device
            )
        else:
            target_center_t = rec_params.coords.mean(dim=0)

        lig_params = parameterize_ligand_biophysics(
            mol, keep_hydrogens=keep_hydrogens, device=rec_params.device
        )
        shift = target_center_t - lig_params.coords.mean(dim=0)
        shifted_coords = lig_params.coords + shift.unsqueeze(0)
        lig_params = lig_params.with_coords(shifted_coords)

        # Sync RDKit Mol conformer positions to the pocket-centered coordinates
        conf = mol.GetConformer(0)
        coords_cpu = shifted_coords.detach().cpu().numpy()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(float(coords_cpu[i, 0]), float(coords_cpu[i, 1]), float(coords_cpu[i, 2])),
            )

        pdb_text = (
            Path(pdb_source).read_text(encoding="utf-8", errors="ignore")
            if isinstance(pdb_source, Path)
            or (
                isinstance(pdb_source, str)
                and "\n" not in pdb_source
                and Path(pdb_source).is_file()
            )
            else str(pdb_source)
        )

        return cls(
            ligand_params=lig_params,
            pocket_params=rec_params,
            hamiltonian=hamiltonian,
            ligand_mol=mol,
            pocket_pdb_text=pdb_text,
            metadata=meta,
        )

    def _sync_mol_coords(self) -> None:
        """Synchronize attached RDKit ``ligand_mol`` conformer with ``ligand_params.coords``."""
        if self.ligand_mol is None or self.ligand_mol.GetNumConformers() == 0:
            return
        if self.ligand_mol.GetNumAtoms() != self.ligand_params.num_atoms:
            return
        conf = self.ligand_mol.GetConformer(0)
        arr = self.ligand_params.coords.detach().cpu().numpy()
        for i in range(arr.shape[0]):
            conf.SetAtomPosition(i, Point3D(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])))

    def evaluate_complex(
        self,
        *,
        softcore_lambda: float = 1.0,
        include_internal_strain: bool = False,
        create_graph: bool = False,
    ) -> ComplexEnergyBreakdown:
        """Evaluate the full differentiable biophysical Hamiltonian on current coordinates."""
        return self.hamiltonian.evaluate_complex(
            self.ligand_params,
            self.pocket_params,
            softcore_lambda=softcore_lambda,
            include_internal_strain=include_internal_strain,
            create_graph=create_graph,
        )

    def inspect_physics(self, *, softcore_lambda: float = 1.0) -> TwinPhysicsInspection:
        """Compute full energy breakdown, per-residue hotspot table, and clash diagnostics."""
        breakdown = self.evaluate_complex(softcore_lambda=softcore_lambda)
        n_clash, min_dist, clash_rows = _count_pauli_clashes(
            self.ligand_params,
            self.pocket_params,
            self.ligand_params.coords,
            self.pocket_params.coords,
        )

        hotspot_table: list[dict[str, Any]] = []
        for res_id, tot_e in breakdown.per_residue_energy.items():
            comp = breakdown.per_residue_components.get(res_id, {})
            role = (
                "stabilizing_hotspot"
                if tot_e <= -0.50
                else ("steric_or_electrostatic_clash" if tot_e >= 1.0 else "neutral_shell")
            )
            hotspot_table.append(
                {
                    "residue_id": res_id,
                    "delta_e_kcal_mol": round(tot_e, 4),
                    "electrostatic": round(comp.get("electrostatic", 0.0), 4),
                    "vdw_hydrophobic": round(comp.get("vdw_hydrophobic", 0.0), 4),
                    "hbond_salt": round(comp.get("hbond_salt", 0.0), 4),
                    "halogen_pi": round(comp.get("halogen_pi", 0.0), 4),
                    "classification": role,
                }
            )

        f_norms = torch.linalg.vector_norm(breakdown.ligand_forces_total, dim=-1)
        e_norms = torch.linalg.vector_norm(breakdown.ligand_electric_field, dim=-1)
        max_f = float(f_norms.max().item()) if f_norms.numel() > 0 else 0.0
        rms_f = float(torch.sqrt((f_norms * f_norms).mean()).item()) if f_norms.numel() > 0 else 0.0
        max_e = float(e_norms.max().item()) if e_norms.numel() > 0 else 0.0

        clash_diag = {
            "pauli_clash_count": n_clash,
            "clash_free": n_clash == 0,
            "min_heavy_distance_angstrom": round(min_dist, 4),
            "clashing_pairs": clash_rows,
        }
        return TwinPhysicsInspection(
            energy_breakdown=breakdown,
            hotspot_table=hotspot_table,
            clash_diagnostics=clash_diag,
            max_ligand_force_norm=max_f,
            rms_ligand_force=rms_f,
            max_ligand_efield_v_a=max_e,
            min_intermolecular_distance_angstrom=min_dist,
        )

    def relax_complex(
        self,
        *,
        steps: int = 60,
        lr: float = 0.08,
        optimizer: Literal["lbfgs", "adam"] = "lbfgs",
        backbone_k: float = 30.0,
        sidechain_k: float = 2.0,
        bond_k: float = 160.0,
        angle_k: float = 45.0,
        flexible_sidechains: bool = True,
        sidechain_cutoff: float = 6.0,
        softcore_schedule: bool = True,
        inplace: bool = True,
    ) -> TwinRelaxationResult:
        """Perform differentiable induced-fit micro-relaxation of ligand & pocket sidechains.

        Combines a soft-core clash-relief stage (:math:`\\lambda: 0.60 \\to 1.0`) with
        harmonic covalent bond/angle preservation and protein backbone tethers, followed
        by strong-Wolfe L-BFGS (or force-clipped Adam) polish at :math:`\\lambda = 1.0`.
        """
        target = self if inplace else self.clone()
        lig = target.ligand_params
        rec = target.pocket_params
        dev = target.device

        lc0 = lig.coords.detach().clone()
        rc0 = rec.coords.detach().clone()

        init_bd = target.hamiltonian.evaluate_complex(lig, rec, softcore_lambda=1.0)
        init_energy = init_bd.u_total
        init_clashes, init_min_d, _ = _count_pauli_clashes(lig, rec, lc0, rc0)

        # Determine flexible pocket sidechain atoms near the ligand binding site
        diff0 = lc0.unsqueeze(1) - rc0.unsqueeze(0)
        min_d_rec = torch.sqrt((diff0 * diff0).sum(dim=-1) + 1e-8).min(dim=0).values
        flex_rec_mask = (
            (~rec.is_backbone) & (min_d_rec <= float(sidechain_cutoff))
            if flexible_sidechains
            else torch.zeros_like(rec.is_backbone)
        )
        tether_weights = torch.where(
            flex_rec_mask,
            torch.full((rec.num_atoms,), float(sidechain_k), dtype=torch.float32, device=dev),
            torch.full((rec.num_atoms,), float(backbone_k), dtype=torch.float32, device=dev),
        )

        lc_var = lc0.clone().detach().requires_grad_(True)
        rc_var = rc0.clone().detach().requires_grad_(flexible_sidechains)

        best_energy = float(init_energy)
        best_lc = lc0.clone()
        best_rc = rc0.clone()
        trajectory: list[float] = [float(init_energy)]

        def _total_objective(lc_in: Tensor, rc_in: Tensor, lam_val: float) -> Tensor:
            u_inter = target.hamiltonian.potential_energy(
                lc_in, rc_in, lig, rec, softcore_lambda=lam_val
            )
            u_lig_val = target.hamiltonian.compute_internal_restraint_energy(
                lc_in, lig, bond_k=bond_k, angle_k=angle_k
            )
            if flexible_sidechains:
                u_rec_val = target.hamiltonian.compute_internal_restraint_energy(
                    rc_in, rec, bond_k=bond_k, angle_k=angle_k
                )
                disp_sq = ((rc_in - rc0) * (rc_in - rc0)).sum(dim=-1)
                u_teth = 0.5 * (tether_weights * disp_sq).sum()
            else:
                u_rec_val = lc_in.sum() * 0.0
                u_teth = lc_in.sum() * 0.0
            return u_inter + u_lig_val + u_rec_val + u_teth

        total_steps = max(int(steps), 4)
        opt_params = [lc_var, rc_var] if flexible_sidechains else [lc_var]

        # Stage 1: Clipped Adam with soft-core lambda ramp to smoothly relieve steric clashes
        warmup_steps = (
            max(total_steps // 2, 12) if (softcore_schedule or optimizer == "adam") else 0
        )
        if optimizer == "adam":
            warmup_steps = total_steps

        if warmup_steps > 0:
            adam = torch.optim.Adam(opt_params, lr=float(lr))
            for s_idx in range(warmup_steps):
                lam_s = (
                    0.60 + 0.40 * (float(s_idx) / max(warmup_steps - 1, 1))
                    if softcore_schedule and s_idx < (3 * warmup_steps) // 4
                    else 1.0
                )
                adam.zero_grad()
                with torch.enable_grad():
                    loss = _total_objective(lc_var, rc_var, lam_s)
                    loss.backward()

                with torch.no_grad():
                    if lc_var.grad is not None:
                        lc_var.grad.copy_(_smooth_clip_vectors(lc_var.grad, 20.0))
                    if flexible_sidechains and rc_var.grad is not None:
                        # Freeze backbone atoms strictly during explicit step updates as well
                        rc_grad = _smooth_clip_vectors(rc_var.grad, 15.0)
                        rc_grad = torch.where(
                            flex_rec_mask.unsqueeze(-1), rc_grad, torch.zeros_like(rc_grad)
                        )
                        rc_var.grad.copy_(rc_grad)

                adam.step()

                with torch.no_grad():
                    phys_u = float(
                        target.hamiltonian.potential_energy(
                            lc_var, rc_var, lig, rec, softcore_lambda=1.0
                        ).item()
                    )
                    if math.isfinite(phys_u):
                        trajectory.append(phys_u)
                        if phys_u < best_energy:
                            best_energy = phys_u
                            best_lc = lc_var.detach().clone()
                            best_rc = rc_var.detach().clone()

        # Stage 2: Strong-Wolfe L-BFGS local minimum polish at physical lambda = 1.0
        lbfgs_iters = max(total_steps - warmup_steps, 10) if optimizer == "lbfgs" else 0
        if lbfgs_iters > 0:
            with torch.no_grad():
                lc_var.copy_(best_lc)
                rc_var.copy_(best_rc)

            lbfgs = torch.optim.LBFGS(
                opt_params,
                lr=0.5,
                max_iter=lbfgs_iters,
                history_size=10,
                line_search_fn="strong_wolfe",
            )

            def _closure() -> Tensor:
                lbfgs.zero_grad()
                with torch.enable_grad():
                    rc_eff = (
                        torch.where(flex_rec_mask.unsqueeze(-1), rc_var, rc0)
                        if flexible_sidechains
                        else rc0
                    )
                    obj = _total_objective(lc_var, rc_eff, 1.0)
                    obj.backward()
                if lc_var.grad is not None:
                    lc_var.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if flexible_sidechains and rc_var.grad is not None:
                    rc_var.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                return obj

            with contextlib.suppress(RuntimeError):
                lbfgs.step(_closure)

            with torch.no_grad():
                rc_cand = (
                    torch.where(flex_rec_mask.unsqueeze(-1), rc_var.detach(), rc0)
                    if flexible_sidechains
                    else rc0
                )
                if bool(torch.isfinite(lc_var).all()) and bool(torch.isfinite(rc_cand).all()):
                    cand_u = float(
                        target.hamiltonian.potential_energy(
                            lc_var.detach(), rc_cand, lig, rec, softcore_lambda=1.0
                        ).item()
                    )
                    if math.isfinite(cand_u):
                        trajectory.append(cand_u)
                        if cand_u <= best_energy:
                            best_energy = cand_u
                            best_lc = lc_var.detach().clone()
                            best_rc = rc_cand.clone()

        target.ligand_params = lig.with_coords(best_lc)
        target.pocket_params = rec.with_coords(best_rc)
        target._sync_mol_coords()

        final_bd = target.hamiltonian.evaluate_complex(
            target.ligand_params, target.pocket_params, softcore_lambda=1.0
        )
        final_clashes, final_min_d, _ = _count_pauli_clashes(
            target.ligand_params, target.pocket_params, best_lc, best_rc
        )

        lig_rmsd = float(torch.sqrt(((best_lc - lc0).pow(2).sum(dim=-1)).mean()).item())
        if bool(flex_rec_mask.any()):
            sc_rmsd = float(
                torch.sqrt(
                    ((best_rc[flex_rec_mask] - rc0[flex_rec_mask]).pow(2).sum(dim=-1)).mean()
                ).item()
            )
        else:
            sc_rmsd = 0.0
        if bool(rec.is_backbone.any()):
            bb_rmsd = float(
                torch.sqrt(
                    ((best_rc[rec.is_backbone] - rc0[rec.is_backbone]).pow(2).sum(dim=-1)).mean()
                ).item()
            )
        else:
            bb_rmsd = 0.0

        f_max_after = (
            float(torch.linalg.vector_norm(final_bd.ligand_forces_total, dim=-1).max().item())
            if final_bd.ligand_forces_total.numel() > 0
            else 0.0
        )
        converged = (final_bd.u_total <= init_energy + 1e-4) and math.isfinite(f_max_after)

        return TwinRelaxationResult(
            initial_energy=float(init_energy),
            final_energy=float(final_bd.u_total),
            delta_energy=float(final_bd.u_total - init_energy),
            initial_clash_count=init_clashes,
            final_clash_count=final_clashes,
            initial_min_dist_angstrom=init_min_d,
            final_min_dist_angstrom=final_min_d,
            ligand_rmsd_angstrom=lig_rmsd,
            sidechain_rmsd_angstrom=sc_rmsd,
            backbone_rmsd_angstrom=bb_rmsd,
            steps_run=len(trajectory) - 1,
            converged=converged,
            energy_trajectory=trajectory,
            final_breakdown=final_bd,
        )

    def sample_thermal_ensemble(
        self,
        *,
        n_steps: int = 48,
        step_size_a: float = 0.025,
        temperature_k: float = 300.0,
        burn_in: int = 8,
        stride: int = 2,
        backbone_k: float = 35.0,
        sidechain_k: float = 4.0,
        bond_k: float = 180.0,
        angle_k: float = 50.0,
        sidechain_cutoff: float = 5.5,
        seed: int = 42,
    ) -> TwinEnsembleSummary:
        """Sample a room-temperature (300 K) overdamped Langevin conformational ensemble.

        Integrates Einstein-consistent Brownian dynamics
        ``dr = (sigma^2 / (2 k_B T)) F(r) + sigma * xi`` in ``torch.float32`` without
        mutating the twin's equilibrium coordinates, returning per-atom ligand & pocket
        sidechain RMSFs, H-bond / salt-bridge / :math:`\\sigma`-hole / :math:`\\pi`
        persistence probabilities, and a dynamic stability / residence-time proxy.
        """
        from geneuclidean.biophysics.interaction_viz import extract_typed_interaction_network

        lig = self.ligand_params
        rec = self.pocket_params
        dev = self.device

        kbt = max(float(temperature_k) * KB_KCAL_MOL_K, 1e-4)
        sigma = float(max(step_size_a, 1e-4))
        mobility_dt = (sigma * sigma) / (2.0 * kbt)

        lc_eq = lig.coords.detach().clone()
        rc_eq = rec.coords.detach().clone()

        diff0 = lc_eq.unsqueeze(1) - rc_eq.unsqueeze(0)
        min_d_rec = torch.sqrt((diff0 * diff0).sum(dim=-1) + 1e-8).min(dim=0).values
        flex_mask = (~rec.is_backbone) & (min_d_rec <= float(sidechain_cutoff))

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        lc_cur = lc_eq.clone()
        rc_cur = rc_eq.clone()

        lig_frames: list[NDArray[np.float32]] = []
        rec_frames: list[NDArray[np.float32]] = []
        energies: list[float] = []

        hb_counts: dict[str, int] = {}
        sb_counts: dict[str, int] = {}
        xb_counts: dict[str, int] = {}
        pi_counts: dict[str, int] = {}
        all_contact_counts: dict[str, int] = {}

        total_steps = max(int(n_steps), int(burn_in) + 2)
        eff_stride = max(int(stride), 1)

        for step_i in range(total_steps):
            with torch.enable_grad():
                lc_g = lc_cur.detach().requires_grad_(True)
                rc_g = rc_cur.detach().requires_grad_(True)
                u_inter = self.hamiltonian.potential_energy(
                    lc_g, rc_g, lig, rec, softcore_lambda=0.95
                )
                u_val_l = self.hamiltonian.compute_internal_restraint_energy(
                    lc_g, lig, bond_k=bond_k, angle_k=angle_k
                )
                u_val_r = self.hamiltonian.compute_internal_restraint_energy(
                    rc_g, rec, bond_k=bond_k, angle_k=angle_k
                )
                disp_r = ((rc_g - rc_eq) * (rc_g - rc_eq)).sum(dim=-1)
                tether_w = torch.where(
                    flex_mask,
                    torch.full_like(disp_r, float(sidechain_k)),
                    torch.full_like(disp_r, float(backbone_k)),
                )
                u_tot = u_inter + u_val_l + u_val_r + 0.5 * (tether_w * disp_r).sum()
                gl, gr = torch.autograd.grad(u_tot, (lc_g, rc_g))

            with torch.no_grad():
                fl = _smooth_clip_vectors(-gl, 25.0)
                fr = _smooth_clip_vectors(-gr, 20.0)

                noise_l = torch.randn(lc_cur.shape, dtype=torch.float32, device=dev, generator=gen)
                noise_r = torch.randn(rc_cur.shape, dtype=torch.float32, device=dev, generator=gen)

                lc_cur = lc_cur + mobility_dt * fl + sigma * noise_l
                rc_step = rc_cur + mobility_dt * fr + sigma * noise_r
                rc_cur = torch.where(flex_mask.unsqueeze(-1), rc_step, rc_eq)

                if step_i >= burn_in and ((step_i - burn_in) % eff_stride == 0):
                    lig_frames.append(lc_cur.detach().cpu().numpy().astype(np.float32))
                    rec_frames.append(rc_cur.detach().cpu().numpy().astype(np.float32))
                    energies.append(float(u_inter.detach().item()))

                    frame_twin = ProteinLigandDigitalTwin(
                        lig.with_coords(lc_cur),
                        rec.with_coords(rc_cur),
                        hamiltonian=self.hamiltonian,
                    )
                    net = extract_typed_interaction_network(frame_twin)
                    seen_hb: set[str] = set()
                    seen_sb: set[str] = set()
                    seen_xb: set[str] = set()
                    seen_pi: set[str] = set()
                    seen_any: set[str] = set()

                    for edge in net:
                        if edge.kind == "pauli_clash":
                            continue
                        tag = f"{edge.kind}:{edge.protein_residue}"
                        seen_any.add(tag)
                        if edge.kind == "hbond":
                            seen_hb.add(edge.protein_residue)
                        elif edge.kind == "salt_bridge":
                            seen_sb.add(edge.protein_residue)
                        elif edge.kind == "halogen_sigma_hole":
                            seen_xb.add(edge.protein_residue)
                        elif edge.kind in {"cation_pi", "pi_stacking"}:
                            seen_pi.add(edge.protein_residue)

                    for k in seen_hb:
                        hb_counts[k] = hb_counts.get(k, 0) + 1
                    for k in seen_sb:
                        sb_counts[k] = sb_counts.get(k, 0) + 1
                    for k in seen_xb:
                        xb_counts[k] = xb_counts.get(k, 0) + 1
                    for k in seen_pi:
                        pi_counts[k] = pi_counts.get(k, 0) + 1
                    for k in seen_any:
                        all_contact_counts[k] = all_contact_counts.get(k, 0) + 1

        n_frames = len(lig_frames)
        lig_stack = np.stack(lig_frames, axis=0)  # [M, NL, 3]
        rec_stack = np.stack(rec_frames, axis=0)  # [M, NP, 3]

        lig_mean = np.mean(lig_stack, axis=0, keepdims=True)
        rec_mean = np.mean(rec_stack, axis=0, keepdims=True)
        lig_rmsf = np.sqrt(np.mean(np.sum((lig_stack - lig_mean) ** 2, axis=-1), axis=0)).astype(
            np.float32
        )
        rec_rmsf = np.sqrt(np.mean(np.sum((rec_stack - rec_mean) ** 2, axis=-1), axis=0)).astype(
            np.float32
        )

        flex_np = flex_mask.detach().cpu().numpy()
        mean_lig_rmsf = float(np.mean(lig_rmsf)) if lig_rmsf.size > 0 else 0.0
        mean_sc_rmsf = (
            float(np.mean(rec_rmsf[flex_np]))
            if np.any(flex_np)
            else (float(np.mean(rec_rmsf)) if rec_rmsf.size > 0 else 0.0)
        )

        mean_e = float(np.mean(energies))
        std_e = float(np.std(energies))

        inv_m = 1.0 / max(n_frames, 1)
        hb_occ = {k: float(v * inv_m) for k, v in sorted(hb_counts.items())}
        sb_occ = {k: float(v * inv_m) for k, v in sorted(sb_counts.items())}
        xb_occ = {k: float(v * inv_m) for k, v in sorted(xb_counts.items())}
        pi_occ = {k: float(v * inv_m) for k, v in sorted(pi_counts.items())}
        persistence_pct = {
            k: float(100.0 * v * inv_m)
            for k, v in sorted(all_contact_counts.items(), key=lambda kv: -kv[1])
        }

        # Dynamic stability score in [0, 1] & residence-time proxy log10(tau / tau_0)
        rmsf_factor = math.exp(-max(mean_lig_rmsf - 0.05, 0.0) / 0.85)
        enthalpy_factor = 1.0 / (1.0 + math.exp((mean_e + 2.0) / 6.0))
        top_occ = (
            float(np.mean(list(persistence_pct.values())[:3])) / 100.0 if persistence_pct else 0.35
        )
        stability = float(
            min(1.0, max(0.0, 0.40 * rmsf_factor + 0.35 * enthalpy_factor + 0.25 * top_occ))
        )
        # Eyring-inspired kinetic barrier proxy from favorable enthalpy + persistent polar anchors
        anchor_bonus = (
            0.65 * sum(sb_occ.values())
            + 0.45 * sum(hb_occ.values())
            + 0.40 * sum(xb_occ.values())
            + 0.30 * sum(pi_occ.values())
        )
        res_time_proxy = float(max(0.0, (-0.18 * min(mean_e, 0.0)) + anchor_bonus) * rmsf_factor)

        return TwinEnsembleSummary(
            temperature_k=float(temperature_k),
            n_frames=n_frames,
            ligand_rmsf=lig_rmsf,
            pocket_sidechain_rmsf=rec_rmsf,
            mean_ligand_rmsf=mean_lig_rmsf,
            mean_sidechain_rmsf=mean_sc_rmsf,
            mean_energy_kcal_mol=mean_e,
            std_energy_kcal_mol=std_e,
            hbond_occupancy=hb_occ,
            salt_bridge_occupancy=sb_occ,
            halogen_sigma_hole_occupancy=xb_occ,
            pi_interaction_occupancy=pi_occ,
            interaction_persistence=persistence_pct,
            dynamic_stability_score=stability,
            residence_time_proxy=res_time_proxy,
        )

    def guidance_force_fn(
        self,
        ligand_coords: Tensor,
        *,
        softcore_lambda: float = 0.85,
        max_force_norm: float = 25.0,
    ) -> Tensor:
        """Plug-and-play ``torch.float32`` analytical force-guidance callback for 3D decoders.

        Accepts either unbatched ``[M, 3]`` or batched ``[B, M, 3]`` generated ligand
        coordinates during 3D Flow Matching / Diffusion ODE integration and returns
        smoothly norm-clipped ``SO(3)``-equivariant forces ``-grad_x U_phys`` steering
        generated atoms toward pocket electrostatic complementarity and out of Pauli clashes.
        """
        is_unbatched = ligand_coords.ndim == 2
        if is_unbatched:
            coords_b = ligand_coords.unsqueeze(0)
        elif ligand_coords.ndim == 3 and ligand_coords.shape[-1] == 3:
            coords_b = ligand_coords
        else:
            msg = f"Expected ligand_coords [M, 3] or [B, M, 3], got {tuple(ligand_coords.shape)}"
            raise ValueError(msg)

        dev = coords_b.device
        rec = self.pocket_params.to(dev)
        ham = self.hamiltonian.to(dev)
        m_atoms = coords_b.shape[1]

        if m_atoms == self.ligand_params.num_atoms:
            lig_template = self.ligand_params.to(dev)
        else:
            # Generic neutral/polarizable carbon-sized probe template when M differs during decoding
            lig_template = BiophysicalAtomParams(
                coords=torch.zeros((m_atoms, 3), dtype=torch.float32, device=dev),
                atomic_nums=torch.full((m_atoms,), 6, dtype=torch.long, device=dev),
                partial_charges=torch.zeros((m_atoms,), dtype=torch.float32, device=dev),
                vdw_sigma=torch.full((m_atoms,), 3.399, dtype=torch.float32, device=dev),
                vdw_epsilon=torch.full((m_atoms,), 0.086, dtype=torch.float32, device=dev),
                born_radii=torch.full((m_atoms,), 1.70, dtype=torch.float32, device=dev),
                obc_scaling=torch.full((m_atoms,), 0.72, dtype=torch.float32, device=dev),
                polarizability=torch.full((m_atoms,), 1.20, dtype=torch.float32, device=dev),
                is_donor=torch.zeros((m_atoms,), dtype=torch.bool, device=dev),
                is_acceptor=torch.zeros((m_atoms,), dtype=torch.bool, device=dev),
                is_cationic=torch.zeros((m_atoms,), dtype=torch.bool, device=dev),
                is_anionic=torch.zeros((m_atoms,), dtype=torch.bool, device=dev),
                is_hydrophobic=torch.ones((m_atoms,), dtype=torch.bool, device=dev),
                is_backbone=torch.zeros((m_atoms,), dtype=torch.bool, device=dev),
                is_halogen=torch.zeros((m_atoms,), dtype=torch.bool, device=dev),
                halogen_strength=torch.zeros((m_atoms,), dtype=torch.float32, device=dev),
                donor_axes=torch.zeros((m_atoms, 3), dtype=torch.float32, device=dev),
                halogen_sigma_axes=torch.zeros((m_atoms, 3), dtype=torch.float32, device=dev),
                ring_centroids=torch.zeros((0, 3), dtype=torch.float32, device=dev),
                ring_normals=torch.zeros((0, 3), dtype=torch.float32, device=dev),
                ring_atom_indices=torch.zeros((0, 6), dtype=torch.long, device=dev),
                ring_atom_mask=torch.zeros((0, 6), dtype=torch.bool, device=dev),
                halogen_bond_pairs=torch.zeros((0, 2), dtype=torch.long, device=dev),
                donor_h_pairs=torch.zeros((0, 2), dtype=torch.long, device=dev),
                bond_pairs=torch.zeros((0, 2), dtype=torch.long, device=dev),
                bond_eq_lengths=torch.zeros((0,), dtype=torch.float32, device=dev),
                angle_triplets=torch.zeros((0, 3), dtype=torch.long, device=dev),
                angle_eq_cos=torch.zeros((0,), dtype=torch.float32, device=dev),
                residue_indices=torch.zeros((m_atoms,), dtype=torch.long, device=dev),
            )

        out_forces: list[Tensor] = []
        for b_idx in range(coords_b.shape[0]):
            with torch.enable_grad():
                xb = coords_b[b_idx].detach().to(dtype=torch.float32).requires_grad_(True)
                u_b = ham.potential_energy(
                    xb, rec.coords, lig_template, rec, softcore_lambda=softcore_lambda
                )
                (grad_x,) = torch.autograd.grad(u_b, (xb,))
            fb = _smooth_clip_vectors(-grad_x.detach(), max_force_norm)
            out_forces.append(fb.to(dtype=torch.float32))

        stacked = torch.stack(out_forces, dim=0)
        return stacked.squeeze(0) if is_unbatched else stacked

    def make_flow_guidance_callback(
        self,
        *,
        lambda_start: float = 0.45,
        lambda_end: float = 0.95,
        max_force_norm: float = 25.0,
    ) -> Callable[[Tensor, float], Tensor]:
        """Return a time-dependent ``(coords_t, t) -> force`` callback for 3D Flow Matching."""

        def _callback(coords_t: Tensor, t: float) -> Tensor:
            t_clamped = max(0.0, min(1.0, float(t)))
            lam_t = lambda_start + (lambda_end - lambda_start) * t_clamped
            return self.guidance_force_fn(
                coords_t, softcore_lambda=lam_t, max_force_norm=max_force_norm
            )

        return _callback


# ---------------------------------------------------------------------------
# Optional C++ / CUDA / OpenCL OpenMM Bridge
# ---------------------------------------------------------------------------


class OpenMMTwinAdapter:
    """Clean optional C++/CUDA/OpenCL OpenMM bridge for long-timescale MD trajectories.

    Uses ``importlib.util.find_spec("openmm")`` for zero-overhead runtime detection and
    raises :class:`~geneuclidean.exceptions.OptionalDependencyError` with an actionable
    installation hint when ``openmm`` is not installed in the active environment.
    """

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` if ``openmm`` is importable in the current Python environment."""
        return importlib.util.find_spec("openmm") is not None

    @classmethod
    def require_openmm(cls) -> Any:
        """Import and return ``openmm`` or raise :class:`OptionalDependencyError`."""
        if not cls.is_available():
            raise OptionalDependencyError(
                "OpenMM is required for C++/CUDA/OpenCL molecular dynamics execution.",
                hint=(
                    "Install OpenMM via `conda install -c conda-forge openmm` "
                    "or `pip install openmm`."
                ),
            )
        import importlib

        return importlib.import_module("openmm")

    @staticmethod
    def export_nonbonded_spec(twin: ProteinLigandDigitalTwin) -> dict[str, Any]:
        """Export OpenMM-unit (``nm``, ``kJ/mol``, ``e``) Nonbonded + GBSA-OBC parameters."""
        lig = twin.ligand_params
        rec = twin.pocket_params
        coords_nm = (
            torch.cat([lig.coords, rec.coords], dim=0).detach().cpu().numpy() * 0.1
        ).astype(np.float32)
        charges_e = (
            torch.cat([lig.partial_charges, rec.partial_charges], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        sigmas_nm = (
            torch.cat([lig.vdw_sigma, rec.vdw_sigma], dim=0).detach().cpu().numpy() * 0.1
        ).astype(np.float32)
        epsilons_kj = (
            torch.cat([lig.vdw_epsilon, rec.vdw_epsilon], dim=0).detach().cpu().numpy() * 4.184
        ).astype(np.float32)
        born_radii_nm = (
            torch.cat([lig.born_radii, rec.born_radii], dim=0).detach().cpu().numpy() * 0.1
        ).astype(np.float32)
        obc_scales = (
            torch.cat([lig.obc_scaling, rec.obc_scaling], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return {
            "num_ligand_atoms": lig.num_atoms,
            "num_pocket_atoms": rec.num_atoms,
            "positions_nm": coords_nm,
            "charges_e": charges_e,
            "sigmas_nm": sigmas_nm,
            "epsilons_kj_mol": epsilons_kj,
            "born_radii_nm": born_radii_nm,
            "obc_scales": obc_scales,
            "solute_dielectric": twin.hamiltonian.interior_dielectric,
            "solvent_dielectric": twin.hamiltonian.solvent_dielectric,
        }

    @classmethod
    def build_openmm_context(
        cls,
        twin: ProteinLigandDigitalTwin,
        *,
        temperature_k: float = 300.0,
        friction_inv_ps: float = 1.0,
        step_size_ps: float = 0.002,
        platform_name: str | None = None,
    ) -> Any:
        """Construct a live ``openmm.Context`` with ``NonbondedForce`` + ``GBSAOBCForce``."""
        omm = cls.require_openmm()
        spec = cls.export_nonbonded_spec(twin)

        system = omm.System()
        nonbonded = omm.NonbondedForce()
        nonbonded.setNonbondedMethod(omm.NonbondedForce.CutoffNonPeriodic)
        nonbonded.setCutoffDistance(twin.hamiltonian.nonbonded_cutoff * 0.1)

        gbsa = omm.GBSAOBCForce()
        gbsa.setSoluteDielectric(spec["solute_dielectric"])
        gbsa.setSolventDielectric(spec["solvent_dielectric"])

        lig_z = twin.ligand_params.atomic_nums.detach().cpu().tolist()
        rec_z = twin.pocket_params.atomic_nums.detach().cpu().tolist()
        rec_bb = twin.pocket_params.is_backbone.detach().cpu().tolist()
        all_z = lig_z + rec_z
        is_frozen = [False] * len(lig_z) + list(rec_bb)

        for idx, z_val in enumerate(all_z):
            mass = 0.0 if is_frozen[idx] else (1.008 if z_val == 1 else float(z_val) * 2.0)
            system.addParticle(mass)
            nonbonded.addParticle(
                float(spec["charges_e"][idx]),
                float(spec["sigmas_nm"][idx]),
                float(spec["epsilons_kj_mol"][idx]),
            )
            gbsa.addParticle(
                float(spec["charges_e"][idx]),
                float(spec["born_radii_nm"][idx]),
                float(spec["obc_scales"][idx]),
            )

        system.addForce(nonbonded)
        system.addForce(gbsa)

        integrator = omm.LangevinMiddleIntegrator(
            float(temperature_k), float(friction_inv_ps), float(step_size_ps)
        )
        if platform_name is not None:
            platform = omm.Platform.getPlatformByName(platform_name)
            ctx = omm.Context(system, integrator, platform)
        else:
            ctx = omm.Context(system, integrator)

        positions = [omm.Vec3(float(p[0]), float(p[1]), float(p[2])) for p in spec["positions_nm"]]
        ctx.setPositions(positions)
        return ctx


__all__ = [
    "OpenMMTwinAdapter",
    "ProteinLigandDigitalTwin",
    "TwinEnsembleSummary",
    "TwinPhysicsInspection",
    "TwinRelaxationResult",
]
