"""Offline compound assays for aqueous mixtures and hepatic disposition.

Concentrations in the equilibrium model are micromolar. Model inputs describe
one compound, solid form, buffer and temperature. The structure-only solubility
estimate is a separate endpoint. See docs/compound_assays.md for sources.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SolubilityEstimate:
    """RDKit ESOL estimate of log10(mol/L) for the submitted neutral molecule."""

    canonical_smiles: str
    log_s_mol_l: float


def estimate_aqueous_solubility(smiles: str) -> SolubilityEstimate:
    """Estimate pH-unspecified aqueous logS using Walters' RDKit refit of ESOL.

    Accept a single net-neutral organic molecule. Salt forms, mixtures,
    radicals and incomplete structures lie outside this screening model.
    """

    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Expected a valid, nonempty molecular SMILES")
    elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if (
        len(Chem.GetMolFrags(mol)) != 1
        or Chem.GetFormalCharge(mol) != 0
        or 6 not in elements
        or not elements <= {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        or any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    ):
        raise ValueError("ESOL expects one complete, net-neutral organic molecule")
    aromatic_fraction = sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / mol.GetNumAtoms()
    log_s = (
        0.26121066137801696
        - 0.0066138847738667125 * Descriptors.MolWt(mol)  # type: ignore[attr-defined]
        - 0.7416739523408995 * Crippen.MolLogP(mol)  # type: ignore[attr-defined]
        + 0.003451545565957996 * Lipinski.NumRotatableBonds(mol)  # type: ignore[attr-defined]
        - 0.42624840441316975 * aromatic_fraction
    )
    return SolubilityEstimate(str(Chem.MolToSmiles(mol)), float(log_s))


@dataclass(frozen=True)
class AqueousEquilibrium:
    """Equilibrium inventory in µM relative to the original liquid volume."""

    free_solubility_um: float
    free_um: float
    bound_um: float
    solid_um: float
    partner_free_um: float

    @property
    def dissolved_um(self) -> float:
        """Total compound in the liquid, including the soluble complex."""
        return self.free_um + self.bound_um


def aqueous_equilibrium(
    *,
    total_um: float,
    intrinsic_solubility_um: float,
    ph: float = 7.0,
    ionization: Literal["neutral", "acid", "base"] = "neutral",
    pka: float | None = None,
    partner_total_um: float = 0.0,
    kd_um: float | None = None,
) -> AqueousEquilibrium:
    """Partition a finite compound load among free solute, 1:1 complex and solid.

    Supply neutral-species intrinsic solubility and, for a monoprotic solute,
    the relevant pKa. A soluble partner binds total free solute with a conditional
    Kd at this pH. Excess solid stays in contact with the strongly buffered liquid.
    """

    _nonnegative("total_um", total_um)
    _positive("intrinsic_solubility_um", intrinsic_solubility_um)
    _nonnegative("partner_total_um", partner_total_um)
    if not 0 <= ph <= 14:
        raise ValueError("ph must be in [0, 14] for this aqueous model")
    limit = intrinsic_solubility_um
    if ionization in ("acid", "base"):
        if pka is None or not math.isfinite(pka):
            raise ValueError("Monoprotic solutes require a finite pka")
        exponent = ph - pka if ionization == "acid" else pka - ph
        try:
            limit *= 1.0 + 10.0**exponent
        except OverflowError as exc:
            raise ValueError("Solubility exceeds the representable concentration range") from exc
        _positive("pH-dependent solubility", limit)
    elif ionization != "neutral" or pka is not None:
        raise ValueError("Use neutral without pka, or acid/base with pka")

    if kd_um is not None:
        _positive("kd_um", kd_um)
    if partner_total_um == 0:
        free = min(total_um, limit)
        return AqueousEquilibrium(limit, free, 0.0, total_um - free, 0.0)
    if kd_um is None:
        raise ValueError("A nonzero soluble partner requires a conditional kd_um")

    # For free concentration f, bound = P*f/(Kd+f). When solid is present
    # f is fixed at the saturation limit; binding can dissolve additional solid.
    bound_at_limit = partner_total_um * (limit / (kd_um + limit))
    if total_um >= limit + bound_at_limit:
        free = limit
        solid = (total_um - limit) - bound_at_limit
    else:
        # Stable positive root of f² + (Kd + P - T) f - Kd*T = 0.
        # Rationalize the root when subtraction would cancel significant digits.
        scale = max(total_um, partner_total_um, kd_um)
        t, p, k = total_um / scale, partner_total_um / scale, kd_um / scale
        b = k + (p - t)
        root = math.hypot(b, 2.0 * math.sqrt(k) * math.sqrt(t))
        free = total_um * (2.0 * k / (root + b)) if b > 0 else scale * ((root - b) / 2.0)
        solid = 0.0
    bound = partner_total_um * (free / (kd_um + free))
    partner_free = partner_total_um * (kd_um / (kd_um + free))
    return AqueousEquilibrium(limit, free, bound, max(0.0, solid), partner_free)


@dataclass(frozen=True)
class MicrosomalDepletion:
    """First-order fit to parent disappearance in a microsomal incubation."""

    n_observations: int
    log_slope_per_min: float
    r_squared: float | None
    half_life_min: float | None
    apparent_clint_ul_min_mg: float | None


def fit_microsomal_depletion(
    times_min: Sequence[float], parent_um: Sequence[float], *, protein_mg_per_ml: float
) -> MicrosomalDepletion:
    """Fit log parent concentration against time for one microsomal preparation.

    Use positive quantified observations from the log-linear range, including
    replicates if available. A flat or rising trace has no resolved depletion
    half-life or clearance. Apparent clearance is per mg of microsomal protein.
    """

    _positive("protein_mg_per_ml", protein_mg_per_ml)
    if len(times_min) != len(parent_um) or len(set(times_min)) < 3:
        raise ValueError("Provide aligned observations at three or more distinct times")
    for time, concentration in zip(times_min, parent_um, strict=True):
        _nonnegative("time", time)
        _positive("quantified parent concentration", concentration)

    logs = [math.log(c) for c in parent_um]
    mean_time = math.fsum(times_min) / len(times_min)
    mean_log = math.fsum(logs) / len(logs)
    dt = [t - mean_time for t in times_min]
    dy = [y - mean_log for y in logs]
    ss_time = math.fsum(t * t for t in dt)
    ss_log = math.fsum(y * y for y in dy)
    slope = math.fsum(t * y for t, y in zip(dt, dy, strict=True)) / ss_time
    residual_ss = math.fsum((y - slope * t) ** 2 for t, y in zip(dt, dy, strict=True))
    return MicrosomalDepletion(
        len(logs),
        slope,
        1.0 - residual_ss / ss_log if ss_log > 0 else None,
        math.log(2.0) / -slope if slope < 0 else None,
        -slope * 1000.0 / protein_mg_per_ml if slope < 0 else None,
    )


def well_stirred_hepatic_clearance(
    *, blood_flow_ml_min: float, fraction_unbound_blood: float, unbound_clint_ml_min: float
) -> float:
    """Return hepatic blood clearance (mL/min) in the well-stirred liver model.

    Supply whole-organ unbound intrinsic clearance in mL/min and the blood
    unbound fraction. Microsomal clearance needs independent binding and
    physiological scaling before use here.
    """

    _positive("blood_flow_ml_min", blood_flow_ml_min)
    _nonnegative("unbound_clint_ml_min", unbound_clint_ml_min)
    if not 0 <= fraction_unbound_blood <= 1:
        raise ValueError("fraction_unbound_blood must be in [0, 1]")
    intrinsic = fraction_unbound_blood * unbound_clint_ml_min
    smaller, larger = sorted((blood_flow_ml_min, intrinsic))
    return smaller / (1.0 + smaller / larger)


__all__ = [
    "AqueousEquilibrium",
    "MicrosomalDepletion",
    "SolubilityEstimate",
    "aqueous_equilibrium",
    "estimate_aqueous_solubility",
    "fit_microsomal_depletion",
    "well_stirred_hepatic_clearance",
]
