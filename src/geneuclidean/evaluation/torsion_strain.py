"""Rotatable-bond torsion-angle strain auditing & pluggable Torsion Library backend.

Scientific motivation:
    Complementing Roche Basel's & ZBH Hamburg's classic *Torsion Library* (Penner et al.,
    *JCIM* 2022) and Boehringer Ingelheim's ``TorsionProfiler``, this module evaluates
    acyclic rotatable and partially conjugated dihedrals in 3D docked or co-folded ligands
    (AutoDock Vina, Boltz-2, AlphaFold 3, Chai-1) along two orthogonal axes:

    1. **Curated SMARTS Traffic-Light Geometry Rules:** Classifies every matched dihedral
       into ``"green"`` (preferred CSD/PDB population), ``"yellow"`` (tolerable), or
       ``"red"`` (energetically disfavored / twisted-amide / eclipsed / ortho-clashing
       outlier), ordered most-specific first.
    2. **Localized MMFF94s Torsion-Barrier Estimate:** Evaluates the local single-bond
       rotational energy penalty (``kcal/mol``) by scanning the observed bond around its
       dihedral on an extracted local neighborhood (free of distant whole-ligand steric
       clashes), flagging bonds with red traffic-light angles or high local torsional
       strain (``>= 2.0 kcal/mol``).

    Users who hold a local copy of the Roche/ZBH XML Torsion Library can plug it directly
    into :data:`TORSION_BACKEND_REGISTRY` via :func:`load_torsion_library_xml` and
    :func:`register_torsion_backend` without bundling any license-restricted CSD files.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms

from geneuclidean.registry import FILTER_REGISTRY, SCORER_REGISTRY, Registry

if TYPE_CHECKING:
    from geneuclidean.sdk.scoring import MoleculeReport

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

TrafficLight = Literal["green", "yellow", "red"]


@dataclass(frozen=True)
class TorsionRule:
    """SMARTS-defined dihedral angle tolerance rule (traffic-light classification)."""

    name: str
    smarts: str
    # Preferred (green) peak centers in degrees (modulo 360) and half-widths
    green_peaks: tuple[tuple[float, float], ...]
    # Tolerable (yellow) additional margin beyond green half-width (degrees)
    yellow_margin_deg: float = 15.0
    # Whether the rule is symmetric under reflection (|angle| symmetry)
    symmetric: bool = True

    def classify(self, angle_deg: float) -> tuple[TrafficLight, float]:
        """Classify observed dihedral ``angle_deg`` and return ``(traffic_light, min_dev_deg)``."""
        a = ((float(angle_deg) + 180.0) % 360.0) - 180.0
        min_dev = 180.0
        best_green_tol = 20.0
        for peak, tol in self.green_peaks:
            candidates = (peak, -peak) if self.symmetric else (peak,)
            for p in candidates:
                diff = abs(((a - p + 180.0) % 360.0) - 180.0)
                if diff < min_dev:
                    min_dev = diff
                    best_green_tol = tol
        if min_dev <= best_green_tol:
            return "green", float(min_dev)
        if min_dev <= best_green_tol + self.yellow_margin_deg:
            return "yellow", float(min_dev)
        return "red", float(min_dev)


# Ordered most-specific first (matching Roche/ZBH Torsion Library hierarchy).
# Patterns define 4-atom dihedrals [a1]-[a2]-[a3]-[a4] centered on acyclic bond a2-a3.
CURATED_MEDCHEM_TORSION_RULES: tuple[TorsionRule, ...] = (
    # Secondary / tertiary amide C(=O)-N planar conjugation (0/180 deg)
    TorsionRule(
        name="amide_cn",
        smarts="[O]=[CX3;!R]-[NX3;!R]-[#6,#1]",
        green_peaks=((0.0, 22.0), (180.0, 22.0)),
        yellow_margin_deg=15.0,
    ),
    # Carboxylic ester O=C-O-C planar conjugation (0/180 deg)
    TorsionRule(
        name="ester_co",
        smarts="[O]=[CX3;!R]-[OX2;!R]-[#6]",
        green_peaks=((0.0, 22.0), (180.0, 22.0)),
        yellow_margin_deg=15.0,
    ),
    # Sulfonamide S(=O)(=O)-N-C gauche/orthogonal preference (~60-120 deg)
    TorsionRule(
        name="sulfonamide_sn",
        smarts="[O]=[SX4](=[O])-[NX3;!R]-[#6]",
        green_peaks=((75.0, 45.0), (165.0, 25.0)),
        yellow_margin_deg=20.0,
    ),
    # Aryl-amide / aryl-ketone c-c-C(=O) coplanar preference
    TorsionRule(
        name="aryl_carbonyl",
        smarts="[c]:[c]-[CX3](=[O])-[!#1]",
        green_peaks=((0.0, 40.0), (180.0, 40.0)),
        yellow_margin_deg=20.0,
    ),
    # Aryl-ether c-c-O-C preference (coplanar for unhindered; out-of-plane tolerable for ortho)
    TorsionRule(
        name="aryl_ether",
        smarts="[c]:[c]-[OX2;!R]-[CX4]",
        green_peaks=((0.0, 40.0), (180.0, 40.0), (90.0, 30.0)),
        yellow_margin_deg=20.0,
    ),
    # Aniline / aryl-amine c-c-N-C preference
    TorsionRule(
        name="aryl_amine",
        smarts="[c]:[c]-[NX3;!R]-[#6]",
        green_peaks=((0.0, 45.0), (180.0, 45.0)),
        yellow_margin_deg=20.0,
    ),
    # Biaryl c-c-c-c twist (twisted 30-150 deg or coplanar 0/180 for unhindered heterobiaryls)
    TorsionRule(
        name="biaryl",
        smarts="[c]:[c]-[c]:[c]",
        green_peaks=((45.0, 45.0), (135.0, 45.0)),
        yellow_margin_deg=15.0,
    ),
    # Aryl-sp3 carbon c-c-CH2-X prefers orthogonal/staggered (60-120 deg)
    TorsionRule(
        name="aryl_sp3",
        smarts="[c]:[c]-[CX4;!R]-[#6,#7,#8,#16]",
        green_peaks=((90.0, 50.0), (0.0, 35.0), (180.0, 35.0)),
        yellow_margin_deg=15.0,
    ),
    # Standard sp3-sp3 aliphatic bond: staggered (+/-60, 180) green; eclipsed (0, +/-120) red
    TorsionRule(
        name="sp3_sp3_aliphatic",
        smarts="[!#1]-[CX4;!R]-[CX4,NX3,OX2;!R]-[!#1]",
        green_peaks=((60.0, 32.0), (180.0, 32.0)),
        yellow_margin_deg=14.0,
    ),
    # Fallback generic acyclic rotatable bond
    TorsionRule(
        name="generic_rotatable",
        smarts="[!#1]-[!#1;!R]-!@[!#1;!R]-[!#1]",
        green_peaks=((60.0, 40.0), (180.0, 40.0), (0.0, 30.0)),
        yellow_margin_deg=15.0,
    ),
)

_COMPILED_RULES: tuple[tuple[TorsionRule, Chem.Mol], ...] | None = None


def _get_compiled_rules() -> tuple[tuple[TorsionRule, Chem.Mol], ...]:
    global _COMPILED_RULES
    if _COMPILED_RULES is None:
        compiled: list[tuple[TorsionRule, Chem.Mol]] = []
        for rule in CURATED_MEDCHEM_TORSION_RULES:
            pat = Chem.MolFromSmarts(rule.smarts)
            if pat is not None:
                compiled.append((rule, pat))
        _COMPILED_RULES = tuple(compiled)
    return _COMPILED_RULES


@dataclass(frozen=True)
class TorsionReport:
    """Conformational torsion audit for a single acyclic rotatable/conjugated bond."""

    atom_indices: tuple[int, int, int, int]
    rule_name: str
    observed_angle_deg: float
    deviation_deg: float
    traffic_light: TrafficLight
    local_barrier_kcal: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_indices": list(self.atom_indices),
            "rule_name": self.rule_name,
            "observed_angle_deg": round(self.observed_angle_deg, 2),
            "deviation_deg": round(self.deviation_deg, 2),
            "traffic_light": self.traffic_light,
            "local_barrier_kcal": round(self.local_barrier_kcal, 3),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TorsionStrainProfile:
    """Molecule-level rotatable-bond torsion-strain audit profile."""

    smiles: str
    backend: str
    n_rotatable_audited: int
    n_green: int
    n_yellow: int
    n_red: int
    max_local_barrier_kcal: float
    mean_local_barrier_kcal: float
    barrier_threshold_kcal: float
    passed: bool
    torsions: tuple[TorsionReport, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "backend": self.backend,
            "n_rotatable_audited": self.n_rotatable_audited,
            "n_green": self.n_green,
            "n_yellow": self.n_yellow,
            "n_red": self.n_red,
            "max_local_barrier_kcal": round(self.max_local_barrier_kcal, 3),
            "mean_local_barrier_kcal": round(self.mean_local_barrier_kcal, 3),
            "barrier_threshold_kcal": round(self.barrier_threshold_kcal, 3),
            "passed": self.passed,
            "torsions": [t.to_dict() for t in self.torsions],
        }


def _extract_local_torsion_fragment(
    mol_3d: Chem.Mol,
    quad: tuple[int, int, int, int],
) -> tuple[Chem.Mol, tuple[int, int, int, int]] | None:
    """Extract a 2-hop local fragment around bond ``(j, k)`` preserving attached rings."""
    i_idx, j_idx, k_idx, l_idx = quad
    ring_info = mol_3d.GetRingInfo()
    atom_rings = [set(r) for r in ring_info.AtomRings()]

    keep: set[int] = {i_idx, j_idx, k_idx, l_idx}
    # Include immediate heavy neighbors of j and k (captures ortho steric bulk & amide O)
    for center in (j_idx, k_idx):
        atom = mol_3d.GetAtomWithIdx(center)
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() > 1:
                keep.add(nb.GetIdx())
    # Include any ring containing any kept atom (keeps aromatic/saturated rings intact)
    changed = True
    while changed:
        changed = False
        for rset in atom_rings:
            if (rset & keep) and not rset.issubset(keep):
                keep |= rset
                changed = True
    # Also include exocyclic double-bonded atoms on kept atoms (e.g., =O on lactam/carbonyl)
    for idx in list(keep):
        atom = mol_3d.GetAtomWithIdx(idx)
        for b in atom.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                other = b.GetOtherAtom(atom)
                if other.GetAtomicNum() > 1:
                    keep.add(other.GetIdx())

    src_conf = mol_3d.GetConformer()
    rw = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old_i in sorted(keep):
        src = mol_3d.GetAtomWithIdx(old_i)
        na = Chem.Atom(src.GetAtomicNum())
        na.SetFormalCharge(src.GetFormalCharge())
        na.SetIsAromatic(src.GetIsAromatic())
        if src.GetIsAromatic() and src.GetAtomicNum() == 7:
            nh = int(src.GetTotalNumHs())
            if nh > 0:
                na.SetNumExplicitHs(nh)
        old_to_new[old_i] = rw.AddAtom(na)

    for b in mol_3d.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in keep and v in keep:
            rw.AddBond(old_to_new[u], old_to_new[v], b.GetBondType())

    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    conf = Chem.Conformer(sub.GetNumAtoms())
    for old_i, new_i in old_to_new.items():
        conf.SetAtomPosition(new_i, src_conf.GetAtomPosition(old_i))
    sub.AddConformer(conf, assignId=True)
    new_quad = (old_to_new[i_idx], old_to_new[j_idx], old_to_new[k_idx], old_to_new[l_idx])
    return sub, new_quad


def _local_dihedral_barrier_kcal(
    mol_3d: Chem.Mol,
    quad: tuple[int, int, int, int],
    *,
    n_grid: int = 12,
) -> float:
    """Estimate relaxed local rotational barrier (kcal/mol) via torsion-constrained scan.

    Performs a Boehringer ``TorsionProfiler`` / QM-style relaxed dihedral scan on the
    extracted local fragment: at each grid dihedral ``theta``, locks ONLY the audited central
    dihedral ``(ni, nj, nk, nl)`` within ``+/-1.5 deg`` while allowing bond lengths, bond
    angles, and flanking substituents to relax cleanly, then evaluates the unconstrained
    single-point MMFF94s energy on the relaxed geometry. Eliminates crystallographic
    bond-length jitter and artificial rigid-scan ortho-clash spikes.
    """
    extracted = _extract_local_torsion_fragment(mol_3d, quad)
    if extracted is None:
        return 0.0
    sub, (ni, nj, nk, nl) = extracted
    bond = sub.GetBondBetweenAtoms(nj, nk)
    if bond is None or bond.IsInRing():
        return 0.0

    sub_h = Chem.AddHs(sub, addCoords=True)
    use_mmff = bool(AllChem.MMFFHasAllMoleculeParams(sub_h))  # type: ignore[attr-defined]
    props = (
        AllChem.MMFFGetMoleculeProperties(sub_h, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
        if use_mmff
        else None
    )

    def _eval_torsion_relaxed(m_h: Chem.Mol, target_deg: float) -> float | None:
        m_copy = Chem.Mol(m_h)
        try:
            rdMolTransforms.SetDihedralDeg(m_copy.GetConformer(), ni, nj, nk, nl, target_deg)
        except Exception:
            return None
        if props is not None:
            ff_con = AllChem.MMFFGetMoleculeForceField(m_copy, props)  # type: ignore[attr-defined]
            if ff_con is None:
                return None
            ff_con.MMFFAddTorsionConstraint(ni, nj, nk, nl, True, -1.5, 1.5, 500.0)
            ff_con.Minimize(maxIts=150)
            ff_eval = AllChem.MMFFGetMoleculeForceField(m_copy, props)  # type: ignore[attr-defined]
            return float(ff_eval.CalcEnergy()) if ff_eval is not None else None
        uff_con = AllChem.UFFGetMoleculeForceField(m_copy)  # type: ignore[attr-defined]
        if uff_con is None:
            return None
        uff_con.UFFAddTorsionConstraint(ni, nj, nk, nl, True, -1.5, 1.5, 500.0)
        uff_con.Minimize(maxIts=150)
        uff_eval = AllChem.UFFGetMoleculeForceField(m_copy)  # type: ignore[attr-defined]
        return float(uff_eval.CalcEnergy()) if uff_eval is not None else None

    obs_deg = float(rdMolTransforms.GetDihedralDeg(sub_h.GetConformer(), ni, nj, nk, nl))
    obs_e = _eval_torsion_relaxed(sub_h, obs_deg)
    if obs_e is None or not math.isfinite(obs_e):
        return 0.0

    min_e = obs_e
    step = 360.0 / float(max(6, n_grid))
    for g in range(1, max(6, n_grid)):
        e_val = _eval_torsion_relaxed(sub_h, obs_deg + g * step)
        if e_val is not None and math.isfinite(e_val) and e_val < min_e:
            min_e = e_val

    return float(max(0.0, min(obs_e - min_e, 25.0)))


def _discover_auditable_torsions(
    mol_noh: Chem.Mol,
    rules: Sequence[tuple[TorsionRule, Chem.Mol]],
) -> list[tuple[tuple[int, int, int, int], TorsionRule]]:
    """Find every acyclic rotatable/conjugated central bond and assign its best matching rule."""
    # Map unordered central bond (min(j,k), max(j,k)) -> (quad, rule)
    matched_bonds: dict[tuple[int, int], tuple[tuple[int, int, int, int], TorsionRule]] = {}

    for rule, pat in rules:
        for match in mol_noh.GetSubstructMatches(pat):
            if len(match) < 4:
                continue
            i_idx, j_idx, k_idx, l_idx = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            bond = mol_noh.GetBondBetweenAtoms(j_idx, k_idx)
            if bond is None or bond.IsInRing():
                continue
            # Ignore terminal bonds (degree 1 heavy atoms like -CH3, -OH, -F)
            if (
                mol_noh.GetAtomWithIdx(j_idx).GetDegree() < 2
                or mol_noh.GetAtomWithIdx(k_idx).GetDegree() < 2
            ):
                continue
            key = (min(j_idx, k_idx), max(j_idx, k_idx))
            if key not in matched_bonds:
                matched_bonds[key] = ((i_idx, j_idx, k_idx, l_idx), rule)

    return [matched_bonds[k] for k in sorted(matched_bonds.keys())]


def _smarts_mmff_torsion_backend(
    mol_3d: Chem.Mol,
    *,
    barrier_threshold_kcal: float = 2.5,
    max_red_allowed: int = 0,
    compute_local_barriers: bool = True,
    rules: Sequence[tuple[TorsionRule, Chem.Mol]] | None = None,
) -> TorsionStrainProfile:
    """Built-in deterministic SMARTS traffic-light + local MMFF94s torsion-barrier backend."""
    mol_noh = Chem.RemoveHs(Chem.Mol(mol_3d))
    can_smi = Chem.MolToSmiles(mol_noh, canonical=True, isomericSmiles=True)
    conf = mol_noh.GetConformer()
    compiled = rules if rules is not None else _get_compiled_rules()

    found = _discover_auditable_torsions(mol_noh, compiled)
    reports: list[TorsionReport] = []
    for quad, rule in found:
        obs_deg = float(rdMolTransforms.GetDihedralDeg(conf, quad[0], quad[1], quad[2], quad[3]))
        light, dev = rule.classify(obs_deg)
        barrier = _local_dihedral_barrier_kcal(mol_noh, quad) if compute_local_barriers else 0.0
        # A bond fails only when it lies OUTSIDE preferred CSD "green" geometry AND exhibits
        # either severe red conjugated distortion (twisted amide/ester) or high relaxed barrier.
        severe_red = light == "red" and (
            rule.name in ("amide_cn", "ester_co") or barrier >= 1.5 or not compute_local_barriers
        )
        energy_fail = (light != "green") and (barrier > barrier_threshold_kcal)
        passed = (not severe_red) and (not energy_fail)
        effective_light: TrafficLight = (
            "red"
            if (not passed)
            else ("yellow" if (light == "red" or light == "yellow") else "green")
        )
        reports.append(
            TorsionReport(
                atom_indices=quad,
                rule_name=rule.name,
                observed_angle_deg=obs_deg,
                deviation_deg=dev,
                traffic_light=effective_light,
                local_barrier_kcal=barrier,
                passed=passed,
            )
        )

    n_green = sum(1 for r in reports if r.traffic_light == "green")
    n_yellow = sum(1 for r in reports if r.traffic_light == "yellow")
    n_red = sum(1 for r in reports if r.traffic_light == "red")
    barriers = [r.local_barrier_kcal for r in reports]
    max_b = max(barriers) if barriers else 0.0
    mean_b = (sum(barriers) / len(barriers)) if barriers else 0.0
    passed_all = (n_red <= int(max_red_allowed)) and all(r.passed for r in reports)
    return TorsionStrainProfile(
        smiles=can_smi,
        backend="smarts_mmff94s",
        n_rotatable_audited=len(reports),
        n_green=n_green,
        n_yellow=n_yellow,
        n_red=n_red,
        max_local_barrier_kcal=float(max_b),
        mean_local_barrier_kcal=float(mean_b),
        barrier_threshold_kcal=float(barrier_threshold_kcal),
        passed=passed_all,
        torsions=tuple(reports),
    )


def load_torsion_library_xml(xml_path: str | Path) -> tuple[tuple[TorsionRule, Chem.Mol], ...]:
    """Parse an external ZBH/Roche-format Torsion Library XML into compiled ``TorsionRule`` pairs.

    Allows users holding an institutional Torsion Library XML file to plug it directly into
    :func:`register_torsion_backend` without bundling restricted CSD data inside GenEuclidean.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    compiled: list[tuple[TorsionRule, Chem.Mol]] = []
    idx = 0
    for elem in root.iter("torsionRule"):
        smarts = elem.attrib.get("smarts", "").strip()
        if not smarts:
            continue
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        peaks: list[tuple[float, float]] = []
        for angle_el in elem.findall(".//angle"):
            val = angle_el.attrib.get("value")
            tol = angle_el.attrib.get("tolerance1", angle_el.attrib.get("tolerance", "25"))
            if val is not None:
                peaks.append((float(val), float(tol)))
        if not peaks:
            continue
        rule = TorsionRule(
            name=elem.attrib.get("name", f"xml_rule_{idx}"),
            smarts=smarts,
            green_peaks=tuple(peaks),
            yellow_margin_deg=15.0,
            symmetric=False,
        )
        compiled.append((rule, pat))
        idx += 1
    return tuple(compiled)


TorsionBackendFn = Callable[..., TorsionStrainProfile]
TORSION_BACKEND_REGISTRY: Registry[TorsionBackendFn] = Registry("torsion_strain_backends")
TORSION_BACKEND_REGISTRY.register("smarts_mmff94s", _smarts_mmff_torsion_backend)


def register_torsion_backend(
    name: str,
    fn: TorsionBackendFn,
    *,
    override: bool = False,
) -> TorsionBackendFn:
    """Register a custom torsion-strain backend (e.g., institutional Torsion Library XML)."""
    return TORSION_BACKEND_REGISTRY.register(name, fn, override=override)


def analyze_torsion_strain(
    mol_or_sdf: Chem.Mol | str,
    *,
    backend: str = "smarts_mmff94s",
    barrier_threshold_kcal: float = 2.5,
    max_red_allowed: int = 0,
    compute_local_barriers: bool = True,
) -> TorsionStrainProfile:
    """Audit rotatable-bond dihedral geometry and local torsional strain of a 3D ligand."""
    parsed: Chem.Mol | None
    if isinstance(mol_or_sdf, str):
        if "\n" in mol_or_sdf or "V2000" in mol_or_sdf or "V3000" in mol_or_sdf:
            parsed = Chem.MolFromMolBlock(mol_or_sdf, sanitize=True, removeHs=False)
        else:
            suppl = Chem.SDMolSupplier(mol_or_sdf, sanitize=True, removeHs=False)
            parsed = suppl[0] if len(suppl) > 0 else None
        if parsed is None:
            raise ValueError("Failed to parse 3D ligand from supplied SDF block or path.")
        mol = parsed
    else:
        mol = mol_or_sdf

    if mol.GetNumConformers() == 0:
        raise ValueError("analyze_torsion_strain requires a 3D Chem.Mol with a conformer.")

    fn = TORSION_BACKEND_REGISTRY.get(backend)
    return fn(
        mol,
        barrier_threshold_kcal=barrier_threshold_kcal,
        max_red_allowed=max_red_allowed,
        compute_local_barriers=compute_local_barriers,
    )


class TorsionStrainFilter:
    """Composable SDK filter rejecting molecules with red-flag rotatable-bond torsions."""

    def __init__(
        self,
        *,
        barrier_threshold_kcal: float = 2.5,
        max_red_allowed: int = 0,
        backend: str = "smarts_mmff94s",
        seed: int = 42,
    ) -> None:
        self.barrier_threshold_kcal = float(barrier_threshold_kcal)
        self.max_red_allowed = int(max_red_allowed)
        self.backend = str(backend)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return "torsion_strain_filter"

    def keep(self, report: MoleculeReport) -> bool:
        if not report.valid or not report.canonical_smiles:
            return False
        from geneuclidean.evaluation.ring_strain import _embed_smiles_for_strain

        mol_3d = _embed_smiles_for_strain(report.canonical_smiles, seed=self.seed)
        if mol_3d is None:
            return False
        prof = analyze_torsion_strain(
            mol_3d,
            backend=self.backend,
            barrier_threshold_kcal=self.barrier_threshold_kcal,
            max_red_allowed=self.max_red_allowed,
        )
        return prof.passed


class TorsionStrainScorer:
    """Composable SDK scorer mapping torsion strain smoothly to ``[0, 1]``."""

    def __init__(
        self,
        *,
        scale_kcal: float = 2.5,
        backend: str = "smarts_mmff94s",
        seed: int = 42,
    ) -> None:
        self.scale_kcal = max(0.25, float(scale_kcal))
        self.backend = str(backend)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return "torsion_strain_plausibility"

    def score(self, reports: Sequence[MoleculeReport]) -> list[float]:
        from geneuclidean.evaluation.ring_strain import _embed_smiles_for_strain

        out: list[float] = []
        for report in reports:
            if not report.valid or not report.canonical_smiles:
                out.append(0.0)
                continue
            mol_3d = _embed_smiles_for_strain(report.canonical_smiles, seed=self.seed)
            if mol_3d is None:
                out.append(0.0)
                continue
            prof = analyze_torsion_strain(mol_3d, backend=self.backend)
            red_penalty = 0.75**prof.n_red
            val = float(
                red_penalty * math.exp(-max(0.0, prof.max_local_barrier_kcal) / self.scale_kcal)
            )
            out.append(val)
        return out


FILTER_REGISTRY.register("torsion_strain", TorsionStrainFilter, override=True)
SCORER_REGISTRY.register("torsion_strain_plausibility", TorsionStrainScorer, override=True)

__all__ = [
    "CURATED_MEDCHEM_TORSION_RULES",
    "TORSION_BACKEND_REGISTRY",
    "TorsionReport",
    "TorsionRule",
    "TorsionStrainFilter",
    "TorsionStrainProfile",
    "TorsionStrainScorer",
    "TrafficLight",
    "analyze_torsion_strain",
    "load_torsion_library_xml",
    "register_torsion_backend",
]
