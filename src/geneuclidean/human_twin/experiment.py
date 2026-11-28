"""Couple a solid formulation and identity-matched laboratory evidence to a PK scenario."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .barrier import BarrierRun, evaluate_barrier
from .dialysis import Compound, Nonnegative, Positive, Text
from .exposure import (
    Disposition,
    GastricStage,
    OralFormulation,
    OralMedium,
    fit_sink_dissolution,
    simulate_oral,
)
from .kinetics import (
    D1CompetitionExperiment,
    MicrosomalExperiment,
    scale_microsomal_clearance,
    simulate_d1_competition,
)


class FormulationInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=True
    )
    compound: Compound
    parameters: OralFormulation
    source: Text
    basis: Literal["assumed", "in vitro fit"]


class SinkDissolutionExperiment(BaseModel):
    """Closed in-vitro dissolution assay and provenance for its transfer to the lumen."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=True
    )
    compound: Compound
    material_lot: Text
    solid_form: Text
    raw_data_uri: Text
    assay_medium: OralMedium
    temperature_c: float
    max_rmse_fraction_of_load: Positive
    amount_umol: Positive
    initial_dissolved_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.0
    sample_time_h: tuple[Nonnegative, ...]
    dissolved_umol: tuple[Nonnegative, ...]
    precipitation_per_h: Nonnegative
    precipitation_source: Text
    transfer_assumption: Text


class GastricInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    compound: Compound
    parameters: GastricStage
    source: Text


class MechanisticInputs(BaseModel):
    """Evidence for one material evaluated under a normalized research exposure."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    formulation: FormulationInput | SinkDissolutionExperiment | None = None
    gastric: GastricInput | None = None
    microsomes: MicrosomalExperiment | None = None
    barrier: BarrierRun | None = None
    d1_competition: D1CompetitionExperiment | None = None


def _intestinal_formulations(
    record: FormulationInput | SinkDissolutionExperiment | None,
) -> tuple[dict[str, OralFormulation | None], dict[str, Any] | None]:
    release: dict[str, OralFormulation | None] = {"rapid_equilibrium": None}
    fit_diagnostics = None
    if record is None:
        release.update(
            {
                "faster_solid_release": OralFormulation(
                    dissolution_per_h=2.0, precipitation_per_h=1.0
                ),
                "slower_solid_release": OralFormulation(
                    dissolution_per_h=0.25, precipitation_per_h=1.0
                ),
            }
        )
    elif isinstance(record, SinkDissolutionExperiment):
        assay = record
        fit = fit_sink_dissolution(
            assay.assay_medium,
            assay.sample_time_h,
            assay.dissolved_umol,
            amount_umol=assay.amount_umol,
            initial_dissolved_fraction=assay.initial_dissolved_fraction,
        )
        residuals = [
            assay.amount_umol
            * (
                assay.initial_dissolved_fraction
                - (1 - assay.initial_dissolved_fraction) * math.expm1(-fit.dissolution_per_h * t)
            )
            - d
            for t, d in zip(assay.sample_time_h, assay.dissolved_umol, strict=True)
        ]
        normalized_rmse = fit.rmse_umol / assay.amount_umol
        accepted = normalized_rmse <= assay.max_rmse_fraction_of_load
        fit_diagnostics = {
            **asdict(fit),
            "residuals_umol": residuals,
            "rmse_fraction_of_load": normalized_rmse,
            "accepted": accepted,
            "reason": None if accepted else "Residual error exceeds the supplied assay limit",
        }
        if accepted:
            release["supplied_formulation"] = OralFormulation(
                dissolution_per_h=fit.dissolution_per_h,
                precipitation_per_h=assay.precipitation_per_h,
                initial_dissolved_fraction=assay.initial_dissolved_fraction,
            )
    else:
        release["supplied_formulation"] = record.parameters
    return release, fit_diagnostics


def _gastric_stage(inputs: MechanisticInputs) -> tuple[GastricStage | None, dict[str, Any] | None]:
    gastric = None
    gastric_source: dict[str, Any] | None = None
    if inputs.gastric is not None:
        gastric = inputs.gastric.parameters
        gastric_source = inputs.gastric.model_dump(mode="json")
    elif inputs.formulation is None:
        # Effective first-order rates matching the mean half-emptying times of a
        # labeled mixed meal. The formulation, pH and cap remain scenario assumptions.
        gastric = GastricStage(
            medium=OralMedium(ph=2.0, volume_l=0.25, free_solubility_um=100.0),
            formulation=OralFormulation(dissolution_per_h=2.0, precipitation_per_h=1.0),
            solid_emptying_per_h=math.log(2) * 60 / 58,
            liquid_emptying_per_h=math.log(2) * 60 / 24,
        )
        gastric_source = {
            "source": "https://pubmed.ncbi.nlm.nih.gov/3731973/",
            "emptying_basis": "effective rates matching the reported mixed-meal mean half-times",
            "solid_half_emptying_min": 58,
            "liquid_half_emptying_min": 24,
            "medium_and_release_basis": "hypothetical buffered gastric formulation",
        }
    return gastric, gastric_source


def run_formulation_experiment(
    compound: Compound,
    disposition: Disposition,
    medium: OralMedium,
    times: list[float],
    *,
    amount_umol: float,
    inputs: MechanisticInputs,
) -> dict[str, Any]:
    """Compare oral release models with matched systemic parameters and input mass."""
    for record in (inputs.formulation, inputs.gastric, inputs.microsomes, inputs.d1_competition):
        if record is not None and record.compound != compound:
            raise ValueError("Mechanistic evidence belongs to a different compound ID or structure")
    barrier_data = None
    if inputs.barrier is not None:
        if inputs.barrier.candidate != compound:
            raise ValueError("Barrier run belongs to a different compound ID or structure")
        result = evaluate_barrier(inputs.barrier)
        barrier_data = {
            "run": inputs.barrier.model_dump(mode="json"),
            "analysis": result.model_dump(mode="json"),
            "scope": "in vitro directional permeability and efflux",
        }
    liver_data = None
    if inputs.microsomes is not None:
        liver_data = scale_microsomal_clearance(inputs.microsomes)
        disposition = replace(
            disposition, liver_unbound_clint_l_h=liver_data["total_liver_unbound_clint_l_h"]
        )
    release, fit_diagnostics = _intestinal_formulations(inputs.formulation)
    gastric, gastric_source = _gastric_stage(inputs)
    comparison = (
        "supplied_formulation" if inputs.formulation is not None else "faster_solid_release"
    )
    if gastric is not None and comparison in release:
        release["gastric_to_intestine"] = release[comparison]
    cases: dict[str, Any] = {}
    for name, formulation in release.items():
        stage = gastric if name == "gastric_to_intestine" else None
        profile = simulate_oral(
            disposition,
            medium,
            times,
            amount_umol=amount_umol,
            formulation=formulation,
            gastric_stage=stage,
            dense_brain=True,
        )
        assert profile.gut_dissolved_umol is not None and profile.gut_solid_umol is not None
        assert profile.brain_driver is not None
        trajectory_peak, trajectory_tmax = profile.brain_driver.peak()
        brain_nm = profile.unbound_brain_um * 1000
        peak = int(np.argmax(brain_nm))
        engagement = None
        if inputs.d1_competition is not None:
            engagement = simulate_d1_competition(times, profile.brain_driver, inputs.d1_competition)
        cases[name] = {
            "parameters": asdict(formulation) if formulation else None,
            "gut_solid_umol": profile.gut_solid_umol.round(9).tolist(),
            "gut_dissolved_umol": profile.gut_dissolved_umol.round(9).tolist(),
            "unbound_plasma_nm": (1000 * profile.unbound_plasma_um).round(8).tolist(),
            "unbound_brain_nm": brain_nm.round(8).tolist(),
            "trajectory_brain_cmax_nm": trajectory_peak,
            "trajectory_brain_tmax_h": trajectory_tmax,
            "sampled_brain_cmax_nm": float(brain_nm[peak]),
            "sampled_brain_tmax_h": times[peak],
            "brain_auc_nm_h": 1000 * float(profile.unbound_brain_auc_um_h[-1]),
            "unabsorbed_transit_umol": float(profile.transit_loss_umol[-1]),
            "max_mass_error_umol": float(np.max(np.abs(profile.accounted_umol - amount_umol))),
            "d1_site_fractions": engagement,
        }
        if stage is not None:
            for field in (
                "gastric_solid_umol",
                "gastric_dissolved_umol",
                "intestinal_solid_umol",
                "intestinal_dissolved_umol",
            ):
                cases[name][field] = getattr(profile, field).round(9).tolist()
            cases[name]["gastric_parameters"] = asdict(stage)
            cases[name]["matched_intestine_only_case"] = comparison
    return {
        "gastric_source": gastric_source,
        "input_amount_umol": amount_umol,
        "time_h": times,
        "medium": asdict(medium),
        "disposition": asdict(disposition),
        "cases": cases,
        "formulation_source": inputs.formulation.model_dump(mode="json")
        if inputs.formulation
        else {"basis": "assumed", "source": "illustrative finite release rates"},
        "sink_dissolution_fit": fit_diagnostics,
        "liver_ivive": liver_data,
        "barrier_assay": barrier_data,
        "d1_kinetic_evidence": inputs.d1_competition.model_dump(mode="json")
        if inputs.d1_competition
        else None,
    }
