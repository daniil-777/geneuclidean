"""Amount-conserving oral parent-drug PK with buffered gut and reversible BBB transport."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from geneuclidean.assays import aqueous_equilibrium, well_stirred_hepatic_clearance

Vector = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class OralMedium:
    """A fixed, buffered lumen. The solubility cap is apparent at the specified pH.

    The optional partner is retained at constant total concentration and binds
    free solute reversibly in a 1:1 complex. Only free solute is absorbed.
    """

    ph: float
    volume_l: float
    free_solubility_um: float
    partner_total_um: float = 0.0
    partner_kd_um: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.volume_l) or self.volume_l <= 0:
            raise ValueError("volume_l must be finite and positive")
        self.free_amount_umol(0.0)

    def free_amount_umol(self, amount_umol: float) -> float:
        mixture = aqueous_equilibrium(
            total_um=amount_umol / self.volume_l,
            intrinsic_solubility_um=self.free_solubility_um,
            ph=self.ph,
            partner_total_um=self.partner_total_um,
            kd_um=self.partner_kd_um,
        )
        return mixture.free_um * self.volume_l


@dataclass(frozen=True, kw_only=True)
class Disposition:
    """Explicit scenario parameters. Clearances use L/h and volumes use L.

    central_volume_l relates central amount to total plasma concentration.
    peripheral_volume_l is plasma-equivalent. brain_unbound_volume_l relates
    brain amount to unbound interstitial concentration. These are apparent
    distribution volumes. BBB clearances act on unbound drug on each side.
    """

    absorption_per_h: float
    gut_transit_per_h: float
    gut_escape_fraction: float
    central_volume_l: float
    peripheral_volume_l: float
    peripheral_exchange_l_h: float
    brain_unbound_volume_l: float
    bbb_in_l_h: float
    bbb_out_l_h: float
    fraction_unbound_plasma: float
    blood_to_plasma_ratio: float
    liver_blood_flow_l_h: float
    liver_unbound_clint_l_h: float
    other_plasma_clearance_l_h: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        for name in (
            "central_volume_l",
            "peripheral_volume_l",
            "brain_unbound_volume_l",
            "blood_to_plasma_ratio",
            "liver_blood_flow_l_h",
        ):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.gut_escape_fraction > 1 or self.fraction_unbound_plasma > 1:
            raise ValueError("Escape and unbound fractions must be in [0, 1]")
        if self.fraction_unbound_plasma > self.blood_to_plasma_ratio:
            raise ValueError("Implied fraction unbound in whole blood exceeds 1")

    @property
    def hepatic_blood_clearance_l_h(self) -> float:
        ml_min_per_l_h = 1000.0 / 60.0
        return (
            well_stirred_hepatic_clearance(
                blood_flow_ml_min=self.liver_blood_flow_l_h * ml_min_per_l_h,
                fraction_unbound_blood=self.fraction_unbound_plasma / self.blood_to_plasma_ratio,
                unbound_clint_ml_min=self.liver_unbound_clint_l_h * ml_min_per_l_h,
            )
            / ml_min_per_l_h
        )


@dataclass(frozen=True, kw_only=True)
class CompetitiveInhibition:
    """Constant unbound liver inhibitor concentration acting on one clearance pathway.

    The low-substrate competitive model assumes linear intrinsic clearance and
    independent unaffected pathways. Inputs refer to the same enzyme system.
    """

    affected_clint_fraction: float
    unbound_inhibitor_um: float
    unbound_ki_um: float

    def __post_init__(self) -> None:
        if not 0 <= self.affected_clint_fraction <= 1:
            raise ValueError("affected_clint_fraction must be in [0, 1]")
        if not math.isfinite(self.unbound_inhibitor_um) or self.unbound_inhibitor_um < 0:
            raise ValueError("unbound_inhibitor_um must be finite and nonnegative")
        if not math.isfinite(self.unbound_ki_um) or self.unbound_ki_um <= 0:
            raise ValueError("unbound_ki_um must be finite and positive")

    @property
    def remaining_clint_fraction(self) -> float:
        return (
            1
            - self.affected_clint_fraction
            + self.affected_clint_fraction / (1 + self.unbound_inhibitor_um / self.unbound_ki_um)
        )


@dataclass(frozen=True)
class ExposureProfile:
    """Concentrations in µM, AUC in µM·h and all seven inventory fields in µmol."""

    time_h: Vector
    gut_umol: Vector
    central_umol: Vector
    peripheral_umol: Vector
    brain_umol: Vector
    transit_loss_umol: Vector
    first_pass_loss_umol: Vector
    systemic_loss_umol: Vector
    total_plasma_um: Vector
    unbound_plasma_um: Vector
    unbound_brain_um: Vector
    unbound_plasma_auc_um_h: Vector
    unbound_brain_auc_um_h: Vector

    @property
    def accounted_umol(self) -> Vector:
        return (
            self.gut_umol
            + self.central_umol
            + self.peripheral_umol
            + self.brain_umol
            + self.transit_loss_umol
            + self.first_pass_loss_umol
            + self.systemic_loss_umol
        )


def simulate_oral(
    disposition: Disposition,
    medium: OralMedium,
    times_h: Sequence[float],
    *,
    amount_umol: float,
    inhibition: CompetitiveInhibition | None = None,
) -> ExposureProfile:
    """Simulate one initial oral load of unchanged parent at t=0.

    The seven inventory states close the mass balance. Two additional integrals
    accumulate free plasma and brain exposure independently of sampling density.
    The solver works in fractions of the input for scale-independent tolerances.
    """
    t = np.array(times_h, dtype=np.float64)
    if t.ndim != 1 or not t.size or not np.isfinite(t).all() or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times_h must be finite, nonnegative and strictly increasing")
    if not math.isfinite(amount_umol) or amount_umol <= 0:
        raise ValueError("amount_umol must be finite and positive")
    p = disposition
    if inhibition is not None:
        p = replace(
            p,
            liver_unbound_clint_l_h=p.liver_unbound_clint_l_h * inhibition.remaining_clint_fraction,
        )
    hepatic_cl = p.hepatic_blood_clearance_l_h
    available = p.gut_escape_fraction * (1 - hepatic_cl / p.liver_blood_flow_l_h)
    systemic_cl = hepatic_cl * p.blood_to_plasma_ratio + p.other_plasma_clearance_l_h

    def rates(_time: float, y: Vector) -> Vector:
        gut, central, periph, brain = y[:4]
        # Concentrations below are per unit input during integration.
        cp = central / p.central_volume_l
        cu_p = p.fraction_unbound_plasma * cp
        cu_b = brain / p.brain_unbound_volume_l
        absorb = (
            p.absorption_per_h * medium.free_amount_umol(max(gut, 0) * amount_umol) / amount_umol
        )
        transit = p.gut_transit_per_h * gut
        tissue_flux = p.peripheral_exchange_l_h * (cp - periph / p.peripheral_volume_l)
        brain_flux = p.bbb_in_l_h * cu_p - p.bbb_out_l_h * cu_b
        loss = systemic_cl * cp
        return np.array(
            [
                -absorb - transit,
                absorb * available - tissue_flux - brain_flux - loss,
                tissue_flux,
                brain_flux,
                transit,
                absorb * (1 - available),
                loss,
                cu_p,
                cu_b,
            ]
        )

    initial = np.zeros(9)
    initial[0] = 1.0
    if t[-1] == 0:
        values = initial[:, None]
    else:
        sol = solve_ivp(rates, (0, t[-1]), initial, t_eval=t, method="Radau", rtol=1e-8, atol=1e-11)
        if not sol.success:
            raise RuntimeError(f"Exposure integration failed: {sol.message}")
        values = np.asarray(sol.y)
    if (
        not np.isfinite(values).all()
        or values.min() < -1e-8
        or np.max(np.abs(values[:7].sum(axis=0) - 1.0)) > 1e-7
    ):
        raise ArithmeticError("Integration failed the nonnegative mass-balance check")
    gut, central, periph, brain, transit, pre, systemic, auc_p, auc_b = (
        np.maximum(values, 0) * amount_umol
    )
    cp = central / p.central_volume_l
    return ExposureProfile(
        t,
        gut,
        central,
        periph,
        brain,
        transit,
        pre,
        systemic,
        cp,
        cp * p.fraction_unbound_plasma,
        brain / p.brain_unbound_volume_l,
        auc_p,
        auc_b,
    )
