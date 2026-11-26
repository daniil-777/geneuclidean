"""Indication-agnostic medicinal-chemistry Property Profile packs.

Elevates GenEuclidean beyond CNS-only scoring by providing pluggable therapeutic property
profiles (``"cns_mpo_v2"``, ``"oral_ro5"``, ``"kinase_selectivity"``, ``"beyond_ro5"``)
backed by :data:`PROPERTY_PROFILE_REGISTRY`. Any oncology, immunology, cardiometabolic, or
neuroscience team can evaluate molecules or register custom project-specific MPO profiles
in a single line.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED as RDKitQED
from rdkit.Chem import Descriptors, rdMolDescriptors

from geneuclidean.cns.mpo import compute_cns_mpo
from geneuclidean.registry import Registry


@dataclass(frozen=True)
class PropertyProfileResult:
    """Standardized indication-profile evaluation result in ``[0, 1]``."""

    profile_name: str
    smiles: str
    score: float
    passed: bool
    metrics: dict[str, float]
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "smiles": self.smiles,
            "score": round(self.score, 4),
            "passed": self.passed,
            "metrics": {k: round(float(v), 3) for k, v in self.metrics.items()},
            "violations": list(self.violations),
        }


def _ramp(val: float, good_hi: float, bad_hi: float) -> float:
    """Piecewise linear desirability: 1.0 when <= good_hi, 0.0 when >= bad_hi."""
    if val <= good_hi:
        return 1.0
    if val >= bad_hi:
        return 0.0
    return float((bad_hi - val) / (bad_hi - good_hi))


def _band(val: float, lo_bad: float, lo_good: float, hi_good: float, hi_bad: float) -> float:
    """Trapezoidal desirability: 1.0 inside [lo_good, hi_good], tapering to 0 outside."""
    if lo_good <= val <= hi_good:
        return 1.0
    if val < lo_good:
        return 0.0 if val <= lo_bad else float((val - lo_bad) / (lo_good - lo_bad))
    return 0.0 if val >= hi_bad else float((hi_bad - val) / (hi_bad - hi_good))


def _basic_desc(mol: Chem.Mol) -> dict[str, float]:
    return {
        "mw": float(Descriptors.MolWt(mol)),  # type: ignore[attr-defined]
        "clogp": float(Descriptors.MolLogP(mol)),  # type: ignore[attr-defined]
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "rot_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "qed": float(RDKitQED.qed(mol)),
    }


def _eval_cns_mpo_v2(mol: Chem.Mol) -> PropertyProfileResult:
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mpo = compute_cns_mpo(mol)
    violations: list[str] = []
    if mpo.cns_mpo_v2 < 4.0:
        violations.append(f"CNS MPO v2 ({mpo.cns_mpo_v2:.2f}) < 4.0")
    if mpo.tpsa > 90.0:
        violations.append(f"TPSA ({mpo.tpsa:.1f}) > 90 A^2")
    if mpo.mw > 450.0:
        violations.append(f"MW ({mpo.mw:.1f}) > 450 Da")
    return PropertyProfileResult(
        profile_name="cns_mpo_v2",
        smiles=smi,
        score=float(max(0.0, min(1.0, mpo.cns_mpo_v2 / 6.0))),
        passed=(mpo.cns_mpo_v2 >= 4.0 and mpo.tpsa <= 90.0),
        metrics={
            "cns_mpo_v2": mpo.cns_mpo_v2,
            "mw": mpo.mw,
            "clogp": mpo.clogp,
            "tpsa": mpo.tpsa,
            "hbd": float(mpo.hbd),
            "sa_score": mpo.sa_score,
        },
        violations=tuple(violations),
    )


def _eval_oral_ro5(mol: Chem.Mol) -> PropertyProfileResult:
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    d = _basic_desc(mol)
    violations: list[str] = []
    ro5_v = 0
    if d["mw"] > 500.0:
        ro5_v += 1
        violations.append(f"MW ({d['mw']:.1f}) > 500")
    if d["clogp"] > 5.0:
        ro5_v += 1
        violations.append(f"cLogP ({d['clogp']:.2f}) > 5.0")
    if d["hbd"] > 5.0:
        ro5_v += 1
        violations.append(f"HBD ({int(d['hbd'])}) > 5")
    if d["hba"] > 10.0:
        ro5_v += 1
        violations.append(f"HBA ({int(d['hba'])}) > 10")
    if d["tpsa"] > 140.0:
        violations.append(f"Veber TPSA ({d['tpsa']:.1f}) > 140")
    if d["rot_bonds"] > 10.0:
        violations.append(f"Veber RotB ({int(d['rot_bonds'])}) > 10")
    score = (
        0.25 * _ramp(d["mw"], 420.0, 580.0)
        + 0.25 * _band(d["clogp"], -1.0, 0.5, 4.2, 5.8)
        + 0.20 * _ramp(d["tpsa"], 120.0, 160.0)
        + 0.15 * _ramp(d["rot_bonds"], 8.0, 13.0)
        + 0.15 * d["qed"]
    )
    return PropertyProfileResult(
        profile_name="oral_ro5",
        smiles=smi,
        score=float(max(0.0, min(1.0, score))),
        passed=(ro5_v <= 1 and d["tpsa"] <= 140.0 and d["rot_bonds"] <= 10.0),
        metrics={**d, "ro5_violations": float(ro5_v)},
        violations=tuple(violations),
    )


def _eval_kinase_selectivity(mol: Chem.Mol) -> PropertyProfileResult:
    """Oncology/immunology kinase inhibitor profile (hinge HBD/HBA + escape-from-flatland)."""
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    d = _basic_desc(mol)
    violations: list[str] = []
    if d["mw"] > 520.0:
        violations.append(f"MW ({d['mw']:.1f}) > 520")
    if d["clogp"] > 4.5:
        violations.append(f"cLogP ({d['clogp']:.2f}) > 4.5")
    if d["hba"] < 2.0:
        violations.append("Lacks hinge HBA (>=2)")
    if d["fsp3"] < 0.15:
        violations.append(f"Low Fsp3 ({d['fsp3']:.2f}) < 0.15 (flatness/solubility risk)")
    score = (
        0.22 * _ramp(d["mw"], 450.0, 560.0)
        + 0.22 * _band(d["clogp"], 0.0, 1.5, 4.0, 5.2)
        + 0.20 * _band(d["tpsa"], 35.0, 60.0, 110.0, 145.0)
        + 0.18 * _band(d["hbd"], 0.0, 1.0, 3.0, 5.0)
        + 0.18 * min(1.0, d["fsp3"] / 0.35)
    )
    return PropertyProfileResult(
        profile_name="kinase_selectivity",
        smiles=smi,
        score=float(max(0.0, min(1.0, score))),
        passed=(d["mw"] <= 520.0 and d["clogp"] <= 4.5 and d["hba"] >= 2.0 and d["tpsa"] <= 130.0),
        metrics=d,
        violations=tuple(violations),
    )


def _eval_beyond_ro5(mol: Chem.Mol) -> PropertyProfileResult:
    """AbbVie/Kihlberg beyond-Rule-of-5 oral macrocycle & PROTAC-like profile."""
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    d = _basic_desc(mol)
    violations: list[str] = []
    if d["mw"] > 1000.0:
        violations.append(f"bRo5 MW ({d['mw']:.1f}) > 1000")
    if d["hbd"] > 6.0:
        violations.append(f"bRo5 HBD ({int(d['hbd'])}) > 6")
    if d["tpsa"] > 250.0:
        violations.append(f"bRo5 TPSA ({d['tpsa']:.1f}) > 250")
    if d["rot_bonds"] > 20.0:
        violations.append(f"bRo5 RotB ({int(d['rot_bonds'])}) > 20")
    score = (
        0.25 * _ramp(d["mw"], 750.0, 1050.0)
        + 0.25 * _band(d["clogp"], -2.0, 1.0, 6.0, 8.0)
        + 0.25 * _ramp(d["tpsa"], 180.0, 260.0)
        + 0.25 * _ramp(d["hbd"], 4.0, 7.0)
    )
    return PropertyProfileResult(
        profile_name="beyond_ro5",
        smiles=smi,
        score=float(max(0.0, min(1.0, score))),
        passed=(
            d["mw"] <= 1000.0 and d["hbd"] <= 6.0 and d["tpsa"] <= 250.0 and d["rot_bonds"] <= 20.0
        ),
        metrics=d,
        violations=tuple(violations),
    )


ProfileEvaluatorFn = Callable[[Chem.Mol], PropertyProfileResult]
PROPERTY_PROFILE_REGISTRY: Registry[ProfileEvaluatorFn] = Registry("property_profiles")
PROPERTY_PROFILE_REGISTRY.register("cns_mpo_v2", _eval_cns_mpo_v2, aliases=("cns",))
PROPERTY_PROFILE_REGISTRY.register("oral_ro5", _eval_oral_ro5, aliases=("oral", "ro5"))
PROPERTY_PROFILE_REGISTRY.register(
    "kinase_selectivity", _eval_kinase_selectivity, aliases=("kinase",)
)
PROPERTY_PROFILE_REGISTRY.register("beyond_ro5", _eval_beyond_ro5, aliases=("bro5", "macrocycle"))


def register_property_profile(
    name: str,
    fn: ProfileEvaluatorFn,
    *,
    aliases: tuple[str, ...] = (),
    override: bool = False,
) -> ProfileEvaluatorFn:
    """Register a custom indication/project property profile."""
    return PROPERTY_PROFILE_REGISTRY.register(name, fn, aliases=aliases, override=override)


def list_property_profiles() -> tuple[str, ...]:
    """Return sorted canonical names of registered therapeutic property profiles."""
    return tuple(PROPERTY_PROFILE_REGISTRY.names())


def evaluate_property_profile(
    mol_or_smiles: str | Chem.Mol,
    profile: str = "oral_ro5",
) -> PropertyProfileResult:
    """Evaluate a molecule against any registered therapeutic property profile."""
    mol = (
        Chem.MolFromSmiles(mol_or_smiles.strip())
        if isinstance(mol_or_smiles, str)
        else Chem.RemoveHs(Chem.Mol(mol_or_smiles))
    )
    if mol is None:
        raise ValueError(f"Invalid molecule for property profile {profile!r}.")
    fn = PROPERTY_PROFILE_REGISTRY.get(profile)
    return fn(mol)


__all__ = [
    "PROPERTY_PROFILE_REGISTRY",
    "PropertyProfileResult",
    "evaluate_property_profile",
    "list_property_profiles",
    "register_property_profile",
]
