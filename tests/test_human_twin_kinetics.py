"""Analytical solutions and end-to-end evidence ingestion for mechanistic exposure studies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from geneuclidean.cli import app
from geneuclidean.human_twin import (
    BindingRates,
    D1CompetitionExperiment,
    Disposition,
    FormulationInput,
    MechanisticInputs,
    MicrosomalExperiment,
    MicrosomalTrace,
    OralFormulation,
    OralMedium,
    run_formulation_experiment,
    run_tourette_study,
    scale_microsomal_clearance,
    simulate_d1_competition,
    tourette_candidate,
)


def receptor(da: float) -> D1CompetitionExperiment:
    rates = BindingRates(association_per_nm_h=0.2, dissociation_per_h=0.5, source="test")
    return D1CompetitionExperiment(
        compound=tourette_candidate(),
        target="human DRD1",
        parent_rates=rates,
        dopamine_rates=rates,
        assay_conditions="synthetic",
        parent_material_lot="x",
        dopamine_time_h=(0, 48),
        unbound_dopamine_nm=(da, da),
        dopamine_profile_source="test",
    )


def test_single_ligand_has_finite_washin_and_correct_equilibrium() -> None:
    t = [0, 0.5, 10, 48]
    result = simulate_d1_competition(t, [5] * 4, receptor(0))
    np.testing.assert_allclose(
        result["parent_fraction"], (1 - np.exp(-1.5 * np.asarray(t))) / 1.5, atol=1e-8
    )
    np.testing.assert_allclose(result["dopamine_fraction"], 0)
    np.testing.assert_allclose(np.sum(list(result.values()), axis=0), 1)


def test_competitor_pre_equilibration_and_pulse_displacement() -> None:
    run = receptor(2.5)
    result = simulate_d1_competition([0, 48], [5, 5], run)
    np.testing.assert_allclose(result["parent_fraction"], [0, 0.5], atol=1e-8)
    np.testing.assert_allclose(result["dopamine_fraction"], [0.5, 0.25], atol=1e-8)
    pulse = run.model_copy(
        update={
            "dopamine_time_h": (0, 47.75, 47.9, 48),
            "unbound_dopamine_nm": (2.5, 2.5, 100.0, 2.5),
        }
    )
    displaced = simulate_d1_competition([0, 48], [5, 5], pulse)
    assert displaced["parent_fraction"][-1] < result["parent_fraction"][-1]
    with pytest.raises(ValueError):
        simulate_d1_competition([0, 49], [5, 5], run)
    with pytest.raises(ValidationError):
        BindingRates(association_per_nm_h=-1, dissociation_per_h=1, source="test")


def test_report_frequency_preserves_d1_binding_after_brief_brain_exposure() -> None:
    drug = tourette_candidate()
    run = receptor(0)
    run = run.model_copy(
        update={"parent_rates": run.parent_rates.model_copy(update={"dissociation_per_h": 0.03})}
    )
    inputs = MechanisticInputs(
        d1_competition=run,
        formulation=FormulationInput(
            compound=drug,
            parameters=OralFormulation(dissolution_per_h=2, precipitation_per_h=1),
            basis="assumed",
            source="synthetic",
        ),
    )
    pk = Disposition(
        absorption_per_h=8,
        gut_transit_per_h=0,
        gut_escape_fraction=1,
        central_volume_l=1,
        peripheral_volume_l=1,
        peripheral_exchange_l_h=0,
        brain_unbound_volume_l=1,
        bbb_in_l_h=4,
        bbb_out_l_h=4,
        fraction_unbound_plasma=1,
        blood_to_plasma_ratio=1,
        liver_blood_flow_l_h=90,
        liver_unbound_clint_l_h=0,
        other_plasma_clearance_l_h=8,
    )
    medium = OralMedium(ph=6.8, volume_l=1, free_solubility_um=100)

    def assess(times: list[float]) -> dict:
        return run_formulation_experiment(drug, pk, medium, times, amount_umol=1, inputs=inputs)[
            "cases"
        ]

    sparse = assess([12.0])
    dense = assess(np.linspace(0, 12, 241).tolist())
    for name, case in sparse.items():
        occupancy = case["d1_site_fractions"]["parent_fraction"][-1]
        assert case["trajectory_brain_cmax_nm"] > case["sampled_brain_cmax_nm"]
        assert case["trajectory_brain_tmax_h"] == pytest.approx(
            dense[name]["trajectory_brain_tmax_h"]
        )
        assert occupancy > 0.5
        assert occupancy == pytest.approx(
            dense[name]["d1_site_fractions"]["parent_fraction"][-1], abs=1e-7
        )
        assert case["brain_auc_nm_h"] == pytest.approx(dense[name]["brain_auc_nm_h"])
    assert sparse["rapid_equilibrium"]["unbound_brain_nm"][-1] < 1e-5


def liver_run() -> MicrosomalExperiment:
    t = np.array([0.0, 10, 20, 30])
    control = 5 * np.exp(-0.012 * t)
    active = 3 * np.exp(-(0.04 + 0.012) * t)
    rep = MicrosomalTrace(
        replicate_id="a",
        times_min=tuple(t),
        active_parent_um=tuple(active),
        control_parent_um=tuple(control),
    )
    return MicrosomalExperiment(
        compound=tourette_candidate(),
        matrix="human liver microsomes",
        material_lot="test",
        microsome_lot="test",
        active_and_control_conditions="synthetic",
        raw_data_uri="fixture://depletion",
        protein_mg_per_ml=0.5,
        quantification_limit_um=0.1,
        minimum_fit_r_squared=0.95,
        minimum_observable_turnover=0.10,
        fraction_unbound_incubation=0.5,
        microsomal_protein_mg_per_g_liver=40,
        liver_mass_g=1000,
        additional_unbound_clint_l_h=2,
        scaling_source="synthetic",
        replicates=(rep,),
    )


def test_control_corrected_hepatic_scaling_and_unresolved_losses() -> None:
    run = liver_run()
    report = scale_microsomal_clearance(run)
    assert report["fits"][0]["half_life_min"] == pytest.approx(math.log(2) / 0.04)
    assert report["fits"][0]["apparent_clint_ul_min_mg"] == pytest.approx(80)
    assert report["total_liver_unbound_clint_l_h"] == pytest.approx(386)
    with pytest.raises(ValueError, match="quantification"):
        scale_microsomal_clearance(run.model_copy(update={"quantification_limit_um": 3.1}))
    trace = run.replicates[0].model_copy(update={"active_parent_um": (1.0, 1.0, 1.0, 1.0)})
    with pytest.raises(ValueError, match="unresolved"):
        scale_microsomal_clearance(run.model_copy(update={"replicates": (trace,)}))
    ref = run.replicates[0]
    tiny_turnover = np.asarray(ref.control_parent_um) * np.exp(-1e-5 * np.asarray(ref.times_min))
    trace = ref.model_copy(update={"active_parent_um": tuple(tiny_turnover)})
    with pytest.raises(ValueError, match="unresolved control-corrected turnover"):
        scale_microsomal_clearance(run.model_copy(update={"replicates": (trace,)}))
    with pytest.raises(ValueError, match="different compound"):
        run_tourette_study("DRD1_1", experiment=MechanisticInputs(microsomes=run))


def test_cli_uses_same_measured_clearance_for_matched_formulations(tmp_path: Path) -> None:
    source, output = tmp_path / "inputs.json", tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "microsomes": liver_run().model_dump(mode="json"),
                "formulation": {
                    "compound": tourette_candidate().model_dump(),
                    "source": "synthetic fixture",
                    "basis": "assumed",
                    "parameters": {"dissolution_per_h": 0.25, "precipitation_per_h": 1},
                },
                "d1_competition": receptor(2.5).model_dump(mode="json"),
            }
        )
    )
    invocation = CliRunner().invoke(
        app, ["human-twin", "--experiment", str(source), "--output", str(output)]
    )
    assert invocation.exit_code == 0, invocation.exception
    report = json.loads(output.read_text())
    exp = report["formulation_experiment"]
    assert exp["disposition"]["liver_unbound_clint_l_h"] == pytest.approx(386)
    ref, solid = exp["cases"]["rapid_equilibrium"], exp["cases"]["supplied_formulation"]
    assert solid["sampled_brain_cmax_nm"] < ref["sampled_brain_cmax_nm"]
    assert solid["gut_solid_umol"][0] == 1 and ref["gut_solid_umol"][0] == 0
    assert solid["max_mass_error_umol"] < 1e-7
    assert max(solid["d1_site_fractions"]["parent_fraction"]) > 0
    assert report["clinical_endpoints"]["tic_response"] is None


def test_measured_sink_trace_drives_the_model_and_preserves_assay_geometry() -> None:
    curve = {
        "compound": tourette_candidate().model_dump(),
        "material_lot": "test",
        "solid_form": "free base",
        "raw_data_uri": "fixture://dissolution",
        "assay_medium": {"ph": 6.8, "volume_l": 0.01, "free_solubility_um": 100.0},
        "temperature_c": 37.0,
        "max_rmse_fraction_of_load": 0.05,
        "amount_umol": 0.1,
        "sample_time_h": [0, 0.5, 1, 3],
        "dissolved_umol": [0.1 * -math.expm1(-0.7 * t) for t in (0, 0.5, 1, 3)],
        "precipitation_per_h": 1.0,
        "precipitation_source": "assumed",
        "transfer_assumption": "Same solid material in matched buffered media",
    }
    study = run_tourette_study(experiment=MechanisticInputs.model_validate({"formulation": curve}))
    experiment = study["formulation_experiment"]
    assert experiment["sink_dissolution_fit"]["dissolution_per_h"] == pytest.approx(0.7)
    assert experiment["cases"]["supplied_formulation"]["parameters"][
        "dissolution_per_h"
    ] == pytest.approx(0.7)
    assert experiment["medium"]["volume_l"] == 0.25
    assert experiment["input_amount_umol"] == 1.0
    poor = {**curve, "dissolved_umol": [0.0, 0.09, 0.05, 0.01]}
    rejected = run_tourette_study(
        experiment=MechanisticInputs.model_validate({"formulation": poor})
    )["formulation_experiment"]
    assert not rejected["sink_dissolution_fit"]["accepted"]
    assert "supplied_formulation" not in rejected["cases"]
