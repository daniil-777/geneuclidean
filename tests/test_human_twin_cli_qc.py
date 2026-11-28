"""Command-line decisions for supplied dissolution and directional barrier observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from geneuclidean.cli import app
from geneuclidean.human_twin import tourette_candidate


def _experiment(passes: bool) -> dict[str, Any]:
    candidate = tourette_candidate().model_dump()
    reference = {"compound_id": "reference", "smiles": "CCO"}
    times = [0, 10, 20, 30]
    layout, traces = [], []
    for role, analyte in (("candidate", candidate), ("reference", reference)):
        for direction, rate in (("A_to_B", 1), ("B_to_A", 2)):
            for rep in (1, 2):
                well = f"{role}-{direction}-{rep}"
                layout.append({"well": well, "role": role, "direction": direction})
                traces.append(
                    {
                        "well": well,
                        "direction": direction,
                        "analyte": analyte,
                        "material_lot": "test-lot",
                        "initial_donor_ul": 100,
                        "initial_receiver_ul": 100,
                        "net_teer_before_ohm_cm2": 150,
                        "net_teer_after_ohm_cm2": 150,
                        "samples": [
                            {
                                "time_min": t,
                                "donor_um": 10 - rate * t / 100,
                                "receiver_um": rate * t / 100,
                            }
                            for t in times
                        ],
                    }
                )
        well = f"{role}-blank"
        layout.append({"well": well, "role": f"{role}_blank", "direction": "A_to_B"})
        traces.append(
            {
                "well": well,
                "direction": "A_to_B",
                "analyte": analyte,
                "material_lot": None,
                "initial_donor_ul": 100,
                "initial_receiver_ul": 100,
                "samples": [
                    {"time_min": t, "donor_um": None, "receiver_um": None} for t in (0, 30)
                ],
            }
        )
    if not passes:
        traces[0]["net_teer_after_ohm_cm2"] = 20
    dissolution_times = [0, 0.5, 1, 3]
    return {
        "formulation": {
            "compound": candidate,
            "material_lot": "test-lot",
            "solid_form": "free base",
            "raw_data_uri": "fixture://dissolution",
            "assay_medium": {
                "ph": 6.8,
                "volume_l": 0.01,
                "free_solubility_um": 100.0,
            },
            "temperature_c": 37,
            "max_rmse_fraction_of_load": 0.05,
            "amount_umol": 0.1,
            "sample_time_h": dissolution_times,
            "dissolved_umol": [-0.1 * math.expm1(-0.7 * t) for t in dissolution_times]
            if passes
            else [0, 0.09, 0.05, 0.01],
            "precipitation_per_h": 1,
            "precipitation_source": "fixture",
            "transfer_assumption": "Matched test buffer and solid",
        },
        "barrier": {
            "run_id": "synthetic-plate-run",
            "plate_id": "p1",
            "observed_plate_id": "p1",
            "protocol_version": "QC-v1",
            "instrument_id": "fixture",
            "completed_at": "2026-01-01T00:00:00Z",
            "raw_data_uri": "fixture://barrier",
            "cell_model": "synthetic insert",
            "cell_lot": "cells",
            "apical_medium": "A",
            "basolateral_medium": "B",
            "apical_ph": 7.4,
            "basolateral_ph": 7.4,
            "temperature_c": 37,
            "area_cm2": 1,
            "fit_start_min": 0,
            "fit_end_min": 30,
            "candidate": candidate,
            "candidate_lot": "test-lot",
            "candidate_nominal_donor_um": 10,
            "reference": reference,
            "reference_lot": "test-lot",
            "reference_nominal_donor_um": 10,
            "limits": {
                "source": "validated fixture ranges",
                "candidate_lloq_um": 0.01,
                "reference_lloq_um": 0.01,
                "aqueous_recovery": {"low": 0.95, "high": 1.05},
                "reference_ab_cm_s": {"low": 1.6e-6, "high": 1.7e-6},
                "reference_efflux_ratio": {"low": 1.9, "high": 2.1},
                "min_net_teer_ohm_cm2": 100,
                "max_loading_error_fraction": 0.1,
                "max_donor_change_fraction": 0.2,
                "max_sink_ratio": 0.2,
                "min_fit_r_squared": 0.95,
                "max_replicate_cv": 0.1,
            },
            "layout": layout,
            "traces": traces,
        },
    }


@pytest.mark.parametrize("passes", [True, False], ids=["accepted", "rejected"])
def test_cli_explains_evidence_and_exposure_cases(tmp_path: Path, passes: bool) -> None:
    source, output = tmp_path / "experiment.json", tmp_path / "result.json"
    source.write_text(json.dumps(_experiment(passes)), encoding="utf-8")
    cli = CliRunner().invoke(
        app, ["human-twin", "--experiment", str(source), "--output", str(output)]
    )
    assert cli.exit_code == 0, cli.exception
    report = json.loads(output.read_text(encoding="utf-8"))["formulation_experiment"]
    fit, barrier = report["sink_dissolution_fit"], report["barrier_assay"]["analysis"]
    assert fit["accepted"] is passes
    assert ("supplied_formulation" in report["cases"]) is passes
    assert (
        f"RMSE was {fit['rmse_fraction_of_load']:.2%} of the assay load (limit 5.00%)" in cli.output
    )
    assert "rapid equilibrium" in cli.output
    if passes:
        assert "Dissolution fit accepted for lot test-lot" in cli.output
        assert "dissolution rate of 0.7 h⁻¹" in cli.output
        assert "Barrier assay synthetic-plate-run (synthetic insert) passed QC" in cli.output
        assert f"A→B {barrier['papp_ab_cm_s']:.3g} cm/s" in cli.output
        assert f"B→A {barrier['papp_ba_cm_s']:.3g} cm/s" in cli.output
        assert "apparent efflux ratio 2." in cli.output
    else:
        assert "Dissolution fit rejected for lot test-lot" in cli.output
        assert fit["reason"] in cli.output
        assert "excludes the supplied formulation" in cli.output
        assert "Barrier assay synthetic-plate-run (synthetic insert) failed QC" in cli.output
        assert all(reason in cli.output for reason in barrier["reasons"])
        assert barrier["papp_ab_cm_s"] is None and "In vitro Papp" not in cli.output
        assert "validated fixture ranges" in cli.output
        assert "fixture://dissolution" in cli.output and "fixture://barrier" in cli.output
