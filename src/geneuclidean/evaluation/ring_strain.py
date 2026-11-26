"""Saturated-ring conformational strain & Cremer-Pople ring-pucker auditing for 3D SBDD.

Scientific motivation:
    Standard docking geometry checks (e.g., PoseBusters) verify bond lengths, bond angles,
    internal steric clashes, and aromatic ring planarity, yet frequently miss high-energy
    conformational distortions in *saturated* ring systems (such as strained boat/twist-boat
    piperidines, eclipsed pyrrolidines, or axially clashing bridged/fused aliphatic cores).
    Recent work from ETH Zurich MODLAB / Roche (Atz, Bo, Minot, Schneider & Kuhn, *ChemRxiv*
    2026, ``RingAnalyzer``) demonstrated that such high-energy saturated ring artifacts are
    pervasive across docking poses, conformer generators, and 3D generative models.

Architecture:
    1. **Cremer-Pople Puckering Coordinates (Cremer & Pople, *JACS* 1975):** Exact closed-form
       ring-pucker amplitude ``Q`` (Angstroms), polar angle ``theta`` (degrees for 6-membered
       rings: ``0/180`` = chair, ``54.7/125.3`` = half-chair/envelope, ``90`` = boat/twist-boat),
       and azimuthal phase ``phi`` computed directly from 3D coordinates.
    2. **Ring-Centric Subsystem Strain Energy (``delta_e_kcal``):** Extracts each saturated or
       partially saturated ring system (including fused/spiro/bridged aliphatic ring assemblies),
       caps exocyclic attachment bonds with hydrogens, evaluates the bound conformer energy via
       ``MMFF94s``, and compares it against the global minimum across deterministic ``ETKDGv3``
       + ``MMFF94s`` reference conformers of the isolated ring system.
    3. **Pluggable Strain Backends (``RING_STRAIN_BACKEND_REGISTRY``):** Ships with the zero-QM
       ``"mmff94s"`` physics backend out of the box and allows researchers to register equivariant
       neural DFT estimators (e.g., ``RingAnalyzer``) via ``register_ring_strain_backend(...)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from geneuclidean.registry import FILTER_REGISTRY, SCORER_REGISTRY, Registry

if TYPE_CHECKING:
    from geneuclidean.sdk.scoring import MoleculeReport

RING_STRAIN_BACKEND_REGISTRY: Registry[Callable[..., Any]] = Registry("ring_strain_backend")


@dataclass(frozen=True)
class SaturatedRingReport:
    """Conformational strain and Cremer-Pople pucker analysis for one saturated ring system."""

    ring_index: int
    atom_indices: tuple[int, ...]
    ring_size: int
    n_sp3_atoms: int
    n_rings_in_system: int
    ring_smiles: str
    bound_energy_kcal: float
    global_min_energy_kcal: float
    strain_energy_kcal: float
    pucker_amplitude_q: float | None
    pucker_theta_deg: float | None
    pucker_phi_deg: float | None
    pucker_class: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize single ring report to a JSON-compatible dictionary."""
        return {
            "ring_index": self.ring_index,
            "atom_indices": list(self.atom_indices),
            "ring_size": self.ring_size,
            "n_sp3_atoms": self.n_sp3_atoms,
            "n_rings_in_system": self.n_rings_in_system,
            "ring_smiles": self.ring_smiles,
            "bound_energy_kcal": round(self.bound_energy_kcal, 3),
            "global_min_energy_kcal": round(self.global_min_energy_kcal, 3),
            "strain_energy_kcal": round(self.strain_energy_kcal, 3),
            "pucker_amplitude_q": (
                round(self.pucker_amplitude_q, 4) if self.pucker_amplitude_q is not None else None
            ),
            "pucker_theta_deg": (
                round(self.pucker_theta_deg, 2) if self.pucker_theta_deg is not None else None
            ),
            "pucker_phi_deg": (
                round(self.pucker_phi_deg, 2) if self.pucker_phi_deg is not None else None
            ),
            "pucker_class": self.pucker_class,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RingStrainProfile:
    """Complete 3D saturated-ring & full-ligand conformational strain audit for a 3D pose."""

    smiles: str
    backend: str
    n_saturated_ring_systems: int
    max_ring_strain_kcal: float
    mean_ring_strain_kcal: float
    total_ring_strain_kcal: float
    full_ligand_strain_kcal: float | None
    strain_threshold_kcal: float
    passed: bool
    rings: tuple[SaturatedRingReport, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete ring-strain profile to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "backend": self.backend,
            "n_saturated_ring_systems": self.n_saturated_ring_systems,
            "max_ring_strain_kcal": round(self.max_ring_strain_kcal, 3),
            "mean_ring_strain_kcal": round(self.mean_ring_strain_kcal, 3),
            "total_ring_strain_kcal": round(self.total_ring_strain_kcal, 3),
            "full_ligand_strain_kcal": (
                round(self.full_ligand_strain_kcal, 3)
                if self.full_ligand_strain_kcal is not None
                else None
            ),
            "strain_threshold_kcal": round(self.strain_threshold_kcal, 3),
            "passed": self.passed,
            "rings": [r.to_dict() for r in self.rings],
        }


# ---------------------------------------------------------------------------
# Exact Cremer-Pople Puckering Coordinates (5- and 6-Membered Rings)
# ---------------------------------------------------------------------------


def compute_cremer_pople(
    ring_coords: np.ndarray,
) -> tuple[float, float | None, float | None, str]:
    """Compute exact Cremer-Pople puckering coordinates from ordered 3D ring coordinates.

    Args:
        ring_coords: Array of shape ``[N, 3]`` in cyclic bond-traversal order around a
            single ``N``-membered ring (``N >= 4``).

    Returns:
        Tuple ``(Q, theta_deg, phi_deg, pucker_class)`` where ``Q`` is total puckering
        amplitude in Angstroms, ``theta_deg`` is the polar coordinate in ``[0, 180]``
        (populated for 6-membered rings), ``phi_deg`` is the phase angle in ``[0, 360)``,
        and ``pucker_class`` is a canonical conformational label.
    """
    pts = np.asarray(ring_coords, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 4 or pts.ndim != 2 or pts.shape[1] != 3:
        return 0.0, None, None, "non_canonical"

    # Translate ring geometric center to origin
    r = pts - pts.mean(axis=0, keepdims=True)

    # Cremer-Pople mean-plane vectors R' and R'' (Eqs. 5-7, Cremer & Pople 1975)
    angles = 2.0 * math.pi * np.arange(n, dtype=np.float64) / float(n)
    r_prime = (r * np.sin(angles)[:, None]).sum(axis=0)
    r_double = (r * np.cos(angles)[:, None]).sum(axis=0)
    normal = np.cross(r_prime, r_double)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-10:
        return 0.0, None, None, "degenerate"
    n_hat = normal / norm_len

    # Signed out-of-plane displacements z_j along the Cremer-Pople mean-plane normal
    z = r @ n_hat
    q_total = float(math.sqrt(float(np.sum(z * z))))

    if q_total < 0.08:
        return q_total, 0.0 if n == 6 else None, 0.0, "planar"

    if n == 5:
        c2 = float(math.sqrt(2.0 / 5.0) * np.sum(z * np.cos(2.0 * angles)))
        s2 = float(-math.sqrt(2.0 / 5.0) * np.sum(z * np.sin(2.0 * angles)))
        phi = math.degrees(math.atan2(s2, c2)) % 360.0
        # Envelope every 36 deg (0, 36, 72, ...), twist/half-chair halfway between (18, 54, ...)
        mod36 = phi % 36.0
        dist_env = min(mod36, 36.0 - mod36)
        label = "envelope" if dist_env <= 9.0 else "twist_half_chair"
        return q_total, None, phi, label

    if n == 6:
        c2 = float(math.sqrt(2.0 / 6.0) * np.sum(z * np.cos(2.0 * angles)))
        s2 = float(-math.sqrt(2.0 / 6.0) * np.sum(z * np.sin(2.0 * angles)))
        q2 = float(math.hypot(c2, s2))
        q3 = float(math.sqrt(1.0 / 6.0) * np.sum(z * ((-1.0) ** np.arange(6))))
        theta = math.degrees(math.atan2(q2, q3))
        phi = math.degrees(math.atan2(s2, c2)) % 360.0

        # Standard Cremer-Pople 6-ring globe regions (polar symmetry around 0/180 deg)
        polar_dev = min(theta, 180.0 - theta)
        if polar_dev <= 25.0:
            label = "chair"
        elif polar_dev <= 67.5:
            label = "half_chair_or_envelope"
        else:
            # Equatorial belt around theta = 90 deg: boat at phi = 0,60,...; twist-boat at 30,90,...
            mod60 = phi % 60.0
            dist_boat = min(mod60, 60.0 - mod60)
            label = "boat" if dist_boat <= 15.0 else "twist_boat"
        return q_total, theta, phi, label

    return q_total, None, None, f"puckered_{n}ring"


# ---------------------------------------------------------------------------
# MMFF94s Ring-System Extraction & Conformational Strain Engine
# ---------------------------------------------------------------------------


def _mmff_or_uff_energy(mol_with_h: Chem.Mol, conf_id: int = -1) -> float | None:
    """Evaluate single-point MMFF94s (or UFF fallback) potential energy in kcal/mol."""
    if AllChem.MMFFHasAllMoleculeParams(mol_with_h):  # type: ignore[attr-defined]
        props = AllChem.MMFFGetMoleculeProperties(  # type: ignore[attr-defined]
            mol_with_h, mmffVariant="MMFF94s"
        )
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(  # type: ignore[attr-defined]
                mol_with_h, props, confId=conf_id
            )
            if ff is not None:
                return float(ff.CalcEnergy())
    uff = AllChem.UFFGetMoleculeForceField(mol_with_h, confId=conf_id)  # type: ignore[attr-defined]
    if uff is not None:
        return float(uff.CalcEnergy())
    return None


def _relax_hydrogens_only(mol_with_h: Chem.Mol, heavy_indices: Sequence[int]) -> Chem.Mol:
    """Relax added capping hydrogens while strictly freezing heavy-atom 3D ring geometry."""
    copy_mol = Chem.Mol(mol_with_h)
    if AllChem.MMFFHasAllMoleculeParams(copy_mol):  # type: ignore[attr-defined]
        props = AllChem.MMFFGetMoleculeProperties(  # type: ignore[attr-defined]
            copy_mol, mmffVariant="MMFF94s"
        )
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(copy_mol, props)  # type: ignore[attr-defined]
            if ff is not None:
                for idx in heavy_indices:
                    ff.AddFixedPoint(int(idx))
                ff.Minimize(maxIts=200)
                return copy_mol
    uff = AllChem.UFFGetMoleculeForceField(copy_mol)  # type: ignore[attr-defined]
    if uff is not None:
        for idx in heavy_indices:
            uff.AddFixedPoint(int(idx))
        uff.Minimize(maxIts=200)
    return copy_mol


def _extract_isolated_ring_3d(
    mol_3d: Chem.Mol,
    atom_indices: Sequence[int],
) -> tuple[Chem.Mol, tuple[int, ...]] | None:
    """Extract an isolated 3D ring-system fragment with exocyclic conjugation preserved.

    Following Atz & Schneider's ring-centric fragmentation (``RingAnalyzer`` 2026), retains:
    - All heavy atoms in the saturated ring system ``atom_indices``,
    - Any exocyclic double-bonded heteroatom/carbon (``=O``, ``=S``, ``=N``, ``=C``) directly
      attached to a ring atom (preserving lactam/ketone/enamine hybridization), and
    - Any exocyclic acyl/sulfonyl group (``-C(=O)`` or ``-S(=O)=O``) attached to a ring
      nitrogen so amide/sulfonamide nitrogens keep true sp2 geometry rather than suffering
      artificial NH pyramidalization strain.
    """
    ring_set = set(int(i) for i in atom_indices)
    keep_set = set(ring_set)
    for r_idx in ring_set:
        r_atom = mol_3d.GetAtomWithIdx(r_idx)
        for bond in r_atom.GetBonds():
            nb = bond.GetOtherAtom(r_atom)
            nb_idx = nb.GetIdx()
            if nb_idx in ring_set or nb.GetAtomicNum() <= 1:
                continue
            # Keep exocyclic double bonds directly on the ring (e.g., =O, =N, =C)
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                keep_set.add(nb_idx)
                continue
            # Keep immediate acyl/sulfonyl group (-C(=O), -S(=O)=O) on ring N (sp2 amide N)
            if r_atom.GetAtomicNum() == 7 and nb.GetAtomicNum() in (6, 16):
                has_exo_double_o = any(
                    b2.GetBondType() == Chem.BondType.DOUBLE
                    and b2.GetOtherAtom(nb).GetAtomicNum() in (8, 16)
                    for b2 in nb.GetBonds()
                )
                if has_exo_double_o:
                    keep_set.add(nb_idx)
                    for b2 in nb.GetBonds():
                        o_atom = b2.GetOtherAtom(nb)
                        if b2.GetBondType() == Chem.BondType.DOUBLE and o_atom.GetAtomicNum() in (
                            8,
                            16,
                        ):
                            keep_set.add(o_atom.GetIdx())

    src_conf = mol_3d.GetConformer()

    rw = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old_idx in sorted(keep_set):
        src_atom = mol_3d.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(src_atom.GetAtomicNum())
        new_atom.SetFormalCharge(src_atom.GetFormalCharge())
        new_atom.SetIsAromatic(src_atom.GetIsAromatic())
        # Preserve pyrrole/indole [nH] H-count so RDKit aromatic Kekulization succeeds cleanly
        if src_atom.GetIsAromatic() and src_atom.GetAtomicNum() == 7:
            n_h = int(src_atom.GetTotalNumHs())
            if n_h > 0:
                new_atom.SetNumExplicitHs(n_h)
        new_idx = rw.AddAtom(new_atom)
        old_to_new[old_idx] = new_idx

    for bond in mol_3d.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u in keep_set and v in keep_set:
            rw.AddBond(old_to_new[u], old_to_new[v], bond.GetBondType())

    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        return None

    conf = Chem.Conformer(sub.GetNumAtoms())
    for old_idx, new_idx in old_to_new.items():
        pos = src_conf.GetAtomPosition(old_idx)
        conf.SetAtomPosition(new_idx, pos)
    sub.AddConformer(conf, assignId=True)
    # Lock exact 3D stereochemistry (bridgeheads, ring junctions) onto the extracted fragment so
    # global-minimum conformer search never compares against a different diastereomer.
    Chem.AssignStereochemistryFrom3D(sub)
    sub_ring_indices = tuple(old_to_new[old_i] for old_i in sorted(ring_set))
    return sub, sub_ring_indices


def _global_min_mmff_energy(
    mol_base: Chem.Mol,
    *,
    n_confs: int = 24,
    seed: int = 42,
) -> float | None:
    """Find the lowest MMFF94s conformer energy of an isolated molecule/ring system.

    Includes an unconstrained minimization of the input 3D geometry itself (if present)
    alongside multi-conformer ``ETKDGv3`` sampling, with automatic ``useRandomCoords=True``
    fallback and bounded per-call timeout to guard against RDKit BFGS distance-geometry
    invariant violations or slowdowns on rigid bridged/polycyclic cages.
    """
    best_e: float | None = None

    # 1. Always evaluate unconstrained minimization starting from the input 3D conformer
    if mol_base.GetNumConformers() > 0:
        from_bound = Chem.AddHs(Chem.Mol(mol_base), addCoords=True)
        if AllChem.MMFFHasAllMoleculeParams(from_bound):  # type: ignore[attr-defined]
            p0 = AllChem.MMFFGetMoleculeProperties(from_bound, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
            ff0 = AllChem.MMFFGetMoleculeForceField(from_bound, p0) if p0 is not None else None  # type: ignore[attr-defined]
        else:
            ff0 = AllChem.UFFGetMoleculeForceField(from_bound)  # type: ignore[attr-defined]
        if ff0 is not None:
            ff0.Minimize(maxIts=400)
            best_e = float(ff0.CalcEnergy())

    # 2. Multi-conformer ETKDGv3 global minimum search (with random-coords fallback)
    mol_h = Chem.AddHs(Chem.Mol(mol_base))
    mol_h.RemoveAllConformers()
    cids: list[int] = []
    for use_random_coords in (False, True):
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = int(seed)
        params.numThreads = 1
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.enforceChirality = True
        params.timeout = 3
        params.useRandomCoords = use_random_coords
        try:
            cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(n_confs), params=params))  # type: ignore[attr-defined]
        except RuntimeError:
            cids = []
        if cids:
            break

    if not cids:
        return best_e

    use_mmff = bool(AllChem.MMFFHasAllMoleculeParams(mol_h))  # type: ignore[attr-defined]
    props = (
        AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
        if use_mmff
        else None
    )
    for cid in cids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=int(cid))  # type: ignore[attr-defined]
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=int(cid))  # type: ignore[attr-defined]
        if ff is None:
            continue
        ff.Minimize(maxIts=300)
        energy = float(ff.CalcEnergy())
        if best_e is None or energy < best_e:
            best_e = energy
    return best_e


def _group_saturated_ring_systems(
    mol: Chem.Mol,
) -> list[tuple[tuple[int, ...], list[tuple[int, ...]]]]:
    """Group connected fused/spiro/bridged ring assemblies containing >= 1 saturated ring.

    Retains fused aromatic rings sharing atoms with a saturated ring (e.g., tetralin,
    benzazepine, indole-azepine in ibogaine) so ring-fusion geometry and aromaticity are
    preserved rather than cleaved into artificial non-aromatic cycloalkenes, while
    returning the member saturated rings separately for Cremer-Pople puckering analysis.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = [tuple(int(i) for i in r) for r in ring_info.AtomRings()]
    if not atom_rings:
        return []

    def _is_saturated_ring(ring: tuple[int, ...]) -> bool:
        if not (3 <= len(ring) <= 8):
            return False
        n_sp3 = sum(
            1
            for idx in ring
            if not mol.GetAtomWithIdx(idx).GetIsAromatic()
            and mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
        )
        return n_sp3 >= 2

    # Build connected components over ALL rings sharing >= 1 atom (fused/spiro/bridged)
    visited = [False] * len(atom_rings)
    ring_sets = [set(r) for r in atom_rings]
    systems: list[tuple[tuple[int, ...], list[tuple[int, ...]]]] = []
    for i in range(len(atom_rings)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp_all_rings: list[tuple[int, ...]] = []
        comp_atoms: set[int] = set()
        while stack:
            cur = stack.pop()
            comp_all_rings.append(atom_rings[cur])
            comp_atoms |= ring_sets[cur]
            for j in range(len(atom_rings)):
                if not visited[j] and (ring_sets[cur] & ring_sets[j]):
                    visited[j] = True
                    stack.append(j)
        sat_members = [r for r in comp_all_rings if _is_saturated_ring(r)]
        if sat_members:
            systems.append((tuple(sorted(comp_atoms)), sat_members))
    return systems


def _mmff94s_ring_strain_backend(
    mol_3d: Chem.Mol,
    *,
    strain_threshold_kcal: float = 3.0,
    n_ref_confs: int = 24,
    include_full_ligand_strain: bool = True,
    seed: int = 42,
) -> RingStrainProfile:
    """Built-in deterministic MMFF94s ring-centric & full-ligand strain backend."""
    mol_clean = Chem.RemoveHs(Chem.Mol(mol_3d))
    can_smi = Chem.MolToSmiles(mol_clean, canonical=True, isomericSmiles=True)
    conf = mol_clean.GetConformer()
    systems = _group_saturated_ring_systems(mol_clean)

    ring_reports: list[SaturatedRingReport] = []
    for sys_idx, (atom_indices, member_rings) in enumerate(systems):
        extracted = _extract_isolated_ring_3d(mol_clean, atom_indices)
        if extracted is None:
            continue
        sub_3d, sub_ring_indices = extracted
        ring_smi = Chem.MolToSmiles(sub_3d, canonical=True)
        n_sp3 = sum(
            1
            for idx in atom_indices
            if mol_clean.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
        )

        # Freeze ONLY the endocyclic ring coordinates; let exocyclic cap atoms & Hs relax cleanly
        sub_h = Chem.AddHs(sub_3d, addCoords=True)
        sub_h_relaxed = _relax_hydrogens_only(sub_h, sub_ring_indices)
        bound_e = _mmff_or_uff_energy(sub_h_relaxed)
        min_e = _global_min_mmff_energy(sub_3d, n_confs=n_ref_confs, seed=seed + sys_idx)

        if bound_e is None or min_e is None:
            bound_val = 0.0
            min_val = 0.0
            strain = 0.0
        else:
            bound_val = float(bound_e)
            # Also allow bound pose itself to establish lower bound if it beats ETKDGv3 sampling
            min_val = min(float(min_e), bound_val)
            strain = max(0.0, bound_val - min_val)

        # Cremer-Pople pucker on the primary member ring (largest 5/6-ring in system)
        primary_ring = max(member_rings, key=lambda r: (len(r) in (5, 6), len(r)))
        ring_xyz = np.asarray(
            [
                [
                    float(conf.GetAtomPosition(i).x),
                    float(conf.GetAtomPosition(i).y),
                    float(conf.GetAtomPosition(i).z),
                ]
                for i in primary_ring
            ],
            dtype=np.float64,
        )
        q_val, theta_val, phi_val, pucker_cls = compute_cremer_pople(ring_xyz)

        ring_reports.append(
            SaturatedRingReport(
                ring_index=sys_idx,
                atom_indices=atom_indices,
                ring_size=len(primary_ring),
                n_sp3_atoms=n_sp3,
                n_rings_in_system=len(member_rings),
                ring_smiles=ring_smi,
                bound_energy_kcal=bound_val,
                global_min_energy_kcal=min_val,
                strain_energy_kcal=strain,
                pucker_amplitude_q=q_val,
                pucker_theta_deg=theta_val,
                pucker_phi_deg=phi_val,
                pucker_class=pucker_cls,
                passed=strain <= float(strain_threshold_kcal),
            )
        )

    full_strain: float | None = None
    if include_full_ligand_strain and mol_clean.GetNumAtoms() <= 80:
        lig_h = Chem.AddHs(mol_clean, addCoords=True)
        heavy_lig = [a.GetIdx() for a in lig_h.GetAtoms() if a.GetAtomicNum() > 1]
        lig_h_rel = _relax_hydrogens_only(lig_h, heavy_lig)
        e_bound_full = _mmff_or_uff_energy(lig_h_rel)
        e_min_full = _global_min_mmff_energy(mol_clean, n_confs=min(16, n_ref_confs), seed=seed)
        if e_bound_full is not None and e_min_full is not None:
            full_strain = max(
                0.0, float(e_bound_full) - min(float(e_min_full), float(e_bound_full))
            )

    strains = [r.strain_energy_kcal for r in ring_reports]
    max_s = max(strains) if strains else 0.0
    mean_s = float(sum(strains) / len(strains)) if strains else 0.0
    total_s = float(sum(strains)) if strains else 0.0
    return RingStrainProfile(
        smiles=can_smi,
        backend="mmff94s",
        n_saturated_ring_systems=len(ring_reports),
        max_ring_strain_kcal=max_s,
        mean_ring_strain_kcal=mean_s,
        total_ring_strain_kcal=total_s,
        full_ligand_strain_kcal=full_strain,
        strain_threshold_kcal=float(strain_threshold_kcal),
        passed=all(r.passed for r in ring_reports),
        rings=tuple(ring_reports),
    )


RING_STRAIN_BACKEND_REGISTRY.register(
    "mmff94s",
    _mmff94s_ring_strain_backend,
    override=True,
    metadata={
        "type": "force_field",
        "description": "Deterministic MMFF94s ring-system strain + Cremer-Pople",
    },
)


def register_ring_strain_backend(
    name: str,
    fn: Callable[..., RingStrainProfile] | None = None,
    *,
    override: bool = False,
) -> Any:
    """Register a custom 3D ring-strain backend (e.g., neural DFT ``RingAnalyzer``)."""
    return RING_STRAIN_BACKEND_REGISTRY.register(name, fn, override=override)


def analyze_ring_strain(
    mol_or_sdf: Chem.Mol | str,
    *,
    backend: str = "mmff94s",
    strain_threshold_kcal: float = 3.0,
    n_ref_confs: int = 24,
    include_full_ligand_strain: bool = True,
    seed: int = 42,
) -> RingStrainProfile:
    """Audit saturated-ring conformational strain and Cremer-Pople puckering of a 3D ligand.

    Args:
        mol_or_sdf: 3D RDKit ``Chem.Mol`` (with conformer) or SDF molblock / path string.
        backend: Registered strain backend name (default ``"mmff94s"``).
        strain_threshold_kcal: Maximum allowed per-ring-system strain in kcal/mol (``3.0``).
        n_ref_confs: Number of ``ETKDGv3`` reference conformers for isolated ring minimum.
        include_full_ligand_strain: Also compute full-ligand bound vs. unbound MMFF94s strain.
        seed: Deterministic conformer RNG seed.

    Returns:
        :class:`RingStrainProfile` detailing per-ring strain, Cremer-Pople pucker, and pass flag.
    """
    parsed_mol: Chem.Mol | None
    if isinstance(mol_or_sdf, str):
        if "\n" in mol_or_sdf or "V2000" in mol_or_sdf or "V3000" in mol_or_sdf:
            parsed_mol = Chem.MolFromMolBlock(mol_or_sdf, sanitize=True, removeHs=False)
        else:
            suppl = Chem.SDMolSupplier(mol_or_sdf, sanitize=True, removeHs=False)
            parsed_mol = suppl[0] if len(suppl) > 0 else None
        if parsed_mol is None:
            raise ValueError("Failed to parse 3D ligand from supplied SDF block/path.")
        mol = parsed_mol
    else:
        mol = mol_or_sdf

    if mol.GetNumConformers() == 0:
        raise ValueError("analyze_ring_strain requires a 3D Chem.Mol with at least one conformer.")

    fn = RING_STRAIN_BACKEND_REGISTRY.get(backend)
    profile: RingStrainProfile = fn(
        mol,
        strain_threshold_kcal=strain_threshold_kcal,
        n_ref_confs=n_ref_confs,
        include_full_ligand_strain=include_full_ligand_strain,
        seed=seed,
    )
    return profile


def _embed_smiles_for_strain(smiles: str, seed: int = 42, n_confs: int = 8) -> Chem.Mol | None:
    """Generate the lowest-energy MMFF94s 3D conformer from ETKDGv3 for 2D->3D strain checks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    cids: list[int] = []
    for use_random_coords in (False, True):
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = int(seed)
        params.numThreads = 1
        params.useSmallRingTorsions = True
        params.enforceChirality = True
        params.timeout = 3
        params.useRandomCoords = use_random_coords
        try:
            cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(n_confs), params=params))  # type: ignore[attr-defined]
        except RuntimeError:
            cids = []
        if cids:
            break
    if not cids:
        return None
    best_cid = int(cids[0])
    best_e: float | None = None
    use_mmff = bool(AllChem.MMFFHasAllMoleculeParams(mol_h))  # type: ignore[attr-defined]
    props = (
        AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
        if use_mmff
        else None
    )
    for cid in cids:
        ff = (
            AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=int(cid))  # type: ignore[attr-defined]
            if props is not None
            else AllChem.UFFGetMoleculeForceField(mol_h, confId=int(cid))  # type: ignore[attr-defined]
        )
        if ff is None:
            continue
        ff.Minimize(maxIts=250)
        e = float(ff.CalcEnergy())
        if best_e is None or e < best_e:
            best_e = e
            best_cid = int(cid)
    out = Chem.Mol(mol_h)
    best_conf = Chem.Conformer(mol_h.GetConformer(best_cid))
    out.RemoveAllConformers()
    out.AddConformer(best_conf, assignId=True)
    return Chem.RemoveHs(out)


@dataclass(frozen=True)
class RingStrainFilter:
    """Composable :class:`MoleculeFilterProtocol` rejecting high-strain saturated rings."""

    max_strain_kcal: float = 3.0
    n_ref_confs: int = 16
    seed: int = 42
    name: str = "ring_strain_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when the candidate's 3D saturated rings are below ``max_strain_kcal``."""
        if not report.valid or not report.canonical_smiles:
            return False
        mol_3d = _embed_smiles_for_strain(report.canonical_smiles, seed=self.seed)
        if mol_3d is None:
            return False
        prof = analyze_ring_strain(
            mol_3d,
            strain_threshold_kcal=self.max_strain_kcal,
            n_ref_confs=self.n_ref_confs,
            include_full_ligand_strain=False,
            seed=self.seed,
        )
        return prof.passed


@dataclass(frozen=True)
class RingStrainScorer:
    """Composable :class:`MoleculeScorerProtocol` scoring 3D ring-conformational plausibility."""

    scale_kcal: float = 3.0
    n_ref_confs: int = 16
    seed: int = 42
    name: str = "ring_strain_plausibility"

    def score(self, reports: Sequence[MoleculeReport]) -> list[float]:
        """Map ``max_ring_strain_kcal`` via ``exp(-strain / scale_kcal)`` into ``(0, 1]``."""
        out: list[float] = []
        for idx, rep in enumerate(reports):
            if not rep.valid or not rep.canonical_smiles:
                out.append(0.0)
                continue
            mol_3d = _embed_smiles_for_strain(rep.canonical_smiles, seed=self.seed + idx)
            if mol_3d is None:
                out.append(0.0)
                continue
            prof = analyze_ring_strain(
                mol_3d,
                strain_threshold_kcal=self.scale_kcal,
                n_ref_confs=self.n_ref_confs,
                include_full_ligand_strain=False,
                seed=self.seed + idx,
            )
            out.append(float(math.exp(-prof.max_ring_strain_kcal / max(self.scale_kcal, 1e-6))))
        return out


FILTER_REGISTRY.register("ring_strain", RingStrainFilter, override=True)
SCORER_REGISTRY.register("ring_strain", RingStrainScorer, override=True)

__all__ = [
    "RING_STRAIN_BACKEND_REGISTRY",
    "RingStrainFilter",
    "RingStrainProfile",
    "RingStrainScorer",
    "SaturatedRingReport",
    "analyze_ring_strain",
    "compute_cremer_pople",
    "register_ring_strain_backend",
]
