"""Concentration-driven competitive receptor engagement and microsomal clearance scaling."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from statistics import median
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.integrate import solve_ivp

from geneuclidean.assays import fit_microsomal_depletion
from geneuclidean.human_twin.dialysis import Compound, Fraction, Nonnegative, Positive, Text
from geneuclidean.human_twin.exposure import BrainExposureDriver


class _Input(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=True
    )


class MicrosomalTrace(_Input):
    """Paired active and control parent observations at matching times in one preparation."""

    replicate_id: Text
    times_min: tuple[Nonnegative, ...]
    active_parent_um: tuple[Positive, ...]
    control_parent_um: tuple[Positive, ...]

    @model_validator(mode="after")
    def paired_times(self) -> MicrosomalTrace:
        if not (
            len(self.times_min) == len(self.active_parent_um) == len(self.control_parent_um) >= 3
        ):
            raise ValueError("Supply three or more aligned active and control observations")
        if any(b <= a for a, b in pairwise(self.times_min)):
            raise ValueError("Sampling times must increase")
        return self


class MicrosomalExperiment(_Input):
    """Parent depletion in human liver microsomes plus explicit assumptions for IVIVE."""

    compound: Compound
    matrix: Literal["human liver microsomes"]
    material_lot: Text
    microsome_lot: Text
    active_and_control_conditions: Text
    raw_data_uri: Text
    protein_mg_per_ml: Positive
    quantification_limit_um: Positive
    minimum_observable_turnover: Fraction
    minimum_fit_r_squared: Annotated[float, Field(ge=0, le=1)]
    fraction_unbound_incubation: Fraction
    microsomal_protein_mg_per_g_liver: Positive
    liver_mass_g: Positive
    additional_unbound_clint_l_h: Nonnegative
    scaling_source: Text
    replicates: tuple[MicrosomalTrace, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_replicates(self) -> MicrosomalExperiment:
        if len({r.replicate_id for r in self.replicates}) != len(self.replicates):
            raise ValueError("Duplicate microsomal replicate ID")
        return self


def scale_microsomal_clearance(run: MicrosomalExperiment) -> dict[str, Any]:
    """Subtract matched first-order background loss and scale the median apparent CLint.

    For each replicate log(active/control) cancels common first-order loss.
    Dividing apparent CLint by fu,inc converts to unbound intrinsic clearance.
    Microsomal abundance times liver mass converts to the whole-organ rate.
    """
    estimates: list[float] = []
    fits = []
    for rep in run.replicates:
        if min(*rep.active_parent_um, *rep.control_parent_um) < run.quantification_limit_um:
            raise ValueError(f"{rep.replicate_id} includes parent below quantification")
        corrected = [
            a * (rep.control_parent_um[0] / c)
            for a, c in zip(rep.active_parent_um, rep.control_parent_um, strict=True)
        ]
        fit = fit_microsomal_depletion(
            rep.times_min, corrected, protein_mg_per_ml=run.protein_mg_per_ml
        )
        # Require observed change across the experiment as well as a linear trend.
        turnover = 1 - corrected[-1] / corrected[0]
        if turnover < run.minimum_observable_turnover:
            raise ValueError(f"{rep.replicate_id} has unresolved control-corrected turnover")
        apparent = fit.apparent_clint_ul_min_mg
        if apparent is None or fit.r_squared is None or fit.r_squared < run.minimum_fit_r_squared:
            raise ValueError(
                f"{rep.replicate_id} has unresolved or poorly fitted corrected depletion"
            )
        estimates.append(apparent)
        fits.append(
            {
                "replicate_id": rep.replicate_id,
                "half_life_min": fit.half_life_min,
                "observed_turnover": turnover,
                "r_squared": fit.r_squared,
                "apparent_clint_ul_min_mg": apparent,
            }
        )
    clint = (
        median(estimates)
        / run.fraction_unbound_incubation
        * run.microsomal_protein_mg_per_g_liver
        * run.liver_mass_g
        * 60
        / 1_000_000
    )
    total = clint + run.additional_unbound_clint_l_h
    if not math.isfinite(total):
        raise ValueError("Scaled intrinsic clearance exceeds the representable range")
    return {
        "fits": fits,
        "scaled_microsomal_unbound_clint_l_h": clint,
        "total_liver_unbound_clint_l_h": total,
        "experiment": run.model_dump(mode="json"),
    }


class BindingRates(_Input):
    association_per_nm_h: Positive
    dissociation_per_h: Positive
    source: Text


class D1CompetitionExperiment(_Input):
    """Parent and dopamine compete for a single human D1 receptor site."""

    compound: Compound
    target: Literal["human DRD1"]
    parent_rates: BindingRates
    dopamine_rates: BindingRates
    parent_material_lot: Text
    assay_conditions: Text
    dopamine_time_h: tuple[Nonnegative, ...]
    unbound_dopamine_nm: tuple[Nonnegative, ...]
    dopamine_profile_source: Text

    @model_validator(mode="after")
    def validate_driver(self) -> D1CompetitionExperiment:
        if (
            len(self.dopamine_time_h) != len(self.unbound_dopamine_nm)
            or len(self.dopamine_time_h) < 2
            or self.dopamine_time_h[0] != 0
            or any(b <= a for a, b in pairwise(self.dopamine_time_h))
        ):
            raise ValueError("Dopamine samples must align at increasing times starting at zero")
        return self


def simulate_d1_competition(
    times_h: Sequence[float],
    parent_brain_nm: Sequence[float] | BrainExposureDriver,
    experiment: D1CompetitionExperiment,
) -> dict[str, list[float]]:
    """Sample competitive binding driven by free parent and dopamine concentrations.

    Simulated PK is evaluated on its continuous trajectory. Alternatively supply
    measured parent concentrations at each reporting time, starting at zero,
    for linear interpolation between observations. At time zero only dopamine
    is bound, in equilibrium with its first concentration.
    """
    t = np.asarray(times_h, dtype=float)
    da_t, da_c = experiment.dopamine_time_h, experiment.unbound_dopamine_nm
    if (
        t.ndim != 1
        or t.size == 0
        or not np.isfinite(t).all()
        or t[0] < 0
        or np.any(np.diff(t) <= 0)
        or t[-1] > da_t[-1]
    ):
        raise ValueError("Supply increasing finite report times within the dopamine time range")
    concentration: Callable[[float], float]
    if isinstance(parent_brain_nm, BrainExposureDriver):
        knots = parent_brain_nm.breakpoints_h
        if knots[0] != 0 or t[-1] > knots[-1]:
            raise ValueError("Brain exposure must cover the interval from zero to the last report")
        concentration = parent_brain_nm.unbound_nm
    else:
        parent = np.asarray(parent_brain_nm, dtype=float)
        if (
            t[0] != 0
            or parent.shape != t.shape
            or not np.isfinite(parent).all()
            or np.any(parent < 0)
        ):
            raise ValueError("Supply aligned nonnegative finite parent samples starting at zero")
        knots = t

        def sampled_parent(time: float) -> float:
            return float(np.interp(time, t, parent))

        concentration = sampled_parent

    pr, dr = experiment.parent_rates, experiment.dopamine_rates
    initial_da = da_c[0] / (da_c[0] + dr.dissociation_per_h / dr.association_per_nm_h)
    state = np.array([0.0, initial_da])
    off = np.array([pr.dissociation_per_h, dr.dissociation_per_h])
    occupied = np.empty((len(t), 2))
    if t[0] == 0:
        occupied[0] = state

    def association(time: float) -> np.ndarray:
        return np.array(
            [
                concentration(time) * pr.association_per_nm_h,
                np.interp(time, da_t, da_c) * dr.association_per_nm_h,
            ]
        )

    def rates(time: float, bound: np.ndarray) -> np.ndarray:
        return np.asarray(association(time) * (1 - bound.sum()) - off * bound, dtype=float)

    def jacobian(time: float, bound: np.ndarray) -> np.ndarray:
        parent_on, da_on = association(time)
        return np.array([[-parent_on - off[0], -parent_on], [-da_on, -da_on - off[1]]])

    # Split at genuine driver knots so short exposure or dopamine pulses cannot
    # fall between integrator evaluations. Output sampling does not set step size.
    grid = np.union1d(np.union1d(knots, da_t), t[-1])
    for start, stop in pairwise(grid[grid <= t[-1]]):
        left, right = np.searchsorted(t, [start, stop], side="right")
        report_times = t[left:right]
        evaluate_at = np.union1d(report_times, stop)
        sol = solve_ivp(
            rates,
            (start, stop),
            state,
            method="Radau",
            jac=jacobian,
            t_eval=evaluate_at,
            rtol=1e-8,
            atol=1e-11,
        )
        if not sol.success:
            raise ArithmeticError(sol.message)
        state = sol.y[:, -1]
        occupied[left:right] = sol.y[:, : len(report_times)].T
    fractions = np.column_stack((occupied, 1 - occupied.sum(axis=1)))
    if not np.isfinite(fractions).all() or fractions.min() < -1e-8:
        raise ArithmeticError("Receptor site inventory failed")
    fractions = np.clip(fractions, 0, 1)
    return {
        "parent_fraction": fractions[:, 0].tolist(),
        "dopamine_fraction": fractions[:, 1].tolist(),
        "free_fraction": fractions[:, 2].tolist(),
    }
