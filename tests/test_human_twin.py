"""Analytical limits, identity checks and offline plate-to-PK integration."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import inchi
from typer.testing import CliRunner

import geneuclidean as ge
from geneuclidean.cli import app
from geneuclidean.human_twin import (
    AqueousSolubilityMeasurement,
    CompetitiveInhibition,
    Compound,
    D1BindingMeasurement,
    DialysisLimits,
    DialysisReadout,
    DialysisRun,
    Disposition,
    OralMedium,
    compare_with_human_data,
    evaluate_dialysis,
    plan_dialysis,
    run_tourette_study,
    simulate_oral,
    tourette_candidate,
)

MEDIUM = OralMedium(ph=6.8, volume_l=1.0, free_solubility_um=100)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def disposition() -> Disposition:
    return Disposition(
        absorption_per_h=1.3,
        gut_transit_per_h=0,
        gut_escape_fraction=1,
        central_volume_l=5,
        peripheral_volume_l=20,
        peripheral_exchange_l_h=0,
        brain_unbound_volume_l=2,
        bbb_in_l_h=0,
        bbb_out_l_h=0,
        fraction_unbound_plasma=0.5,
        blood_to_plasma_ratio=1,
        liver_blood_flow_l_h=90,
        liver_unbound_clint_l_h=0,
        other_plasma_clearance_l_h=1,
    )


def test_bateman_trajectory_concentration_and_auc(disposition: Disposition) -> None:
    t = np.array([0, 0.01, 0.3, 1.7, 10, 100])
    x = simulate_oral(disposition, MEDIUM, t.tolist(), amount_umol=2)
    ka, kel = 1.3, 0.2
    factor = 2 / 5 * ka / (ka - kel)
    expected = factor * (np.exp(-kel * t) - np.exp(-ka * t))
    auc = factor * (-np.expm1(-kel * t) / kel + np.expm1(-ka * t) / ka)
    np.testing.assert_allclose(x.total_plasma_um, expected, atol=1e-10)
    np.testing.assert_allclose(x.unbound_plasma_auc_um_h, 0.5 * auc, atol=1e-9)
    np.testing.assert_allclose(x.accounted_umol, 2, atol=1e-10)
    np.testing.assert_array_equal(x.brain_umol, 0)
    zero = simulate_oral(disposition, MEDIUM, [0], amount_umol=2)
    assert zero.gut_umol[0] == zero.accounted_umol[0] == 2


def test_gut_partner_and_solubility_limit_free_parent(disposition: Disposition) -> None:
    p = replace(disposition, gut_transit_per_h=0.2)
    saturated = replace(MEDIUM, free_solubility_um=0.5)
    partner = replace(saturated, partner_total_um=8, partner_kd_um=0.25)
    assert 0 < partner.free_amount_umol(4) < saturated.free_amount_umol(4) < 4
    a, b = (
        simulate_oral(p, medium, [0.01, 2, 96], amount_umol=4) for medium in (saturated, partner)
    )
    assert a.central_umol[0] > b.central_umol[0]
    assert b.transit_loss_umol[-1] > a.transit_loss_umol[-1]
    for trace in (a, b):
        np.testing.assert_allclose(trace.accounted_umol, 4, atol=1e-8)


def test_first_pass_converts_blood_and_plasma_once(disposition: Disposition) -> None:
    p = replace(
        disposition, blood_to_plasma_ratio=2, liver_unbound_clint_l_h=120, gut_escape_fraction=0.6
    )
    # fu,b=0.25, CLblood=90*30/(90+30), Fh=0.75 and CLplasma=45 L/h.
    assert p.hepatic_blood_clearance_l_h == pytest.approx(22.5)
    result = simulate_oral(p, MEDIUM, [120], amount_umol=1)
    assert result.first_pass_loss_umol[-1] == pytest.approx(0.55)
    assert result.systemic_loss_umol[-1] == pytest.approx(0.45)
    assert result.unbound_plasma_auc_um_h[-1] == pytest.approx(0.5 * 0.45 / 46)


def test_reversible_bbb_full_auc_and_finite_time_identity(disposition: Disposition) -> None:
    p = replace(disposition, bbb_in_l_h=2, bbb_out_l_h=5, peripheral_exchange_l_h=3)
    profile = simulate_oral(p, MEDIUM, [0.2, 3, 1200], amount_umol=1)
    np.testing.assert_allclose(
        2 * profile.unbound_plasma_auc_um_h - 5 * profile.unbound_brain_auc_um_h,
        profile.brain_umol,
        atol=1e-9,
    )
    assert profile.unbound_brain_auc_um_h[-1] / profile.unbound_plasma_auc_um_h[
        -1
    ] == pytest.approx(0.4)
    assert profile.peripheral_umol[1] > 0
    np.testing.assert_allclose(profile.accounted_umol, 1, atol=1e-9)


def test_competition_affects_first_pass_and_systemic_clearance(disposition: Disposition) -> None:
    p = replace(disposition, liver_unbound_clint_l_h=30)
    inhibitor = CompetitiveInhibition(
        affected_clint_fraction=0.5, unbound_inhibitor_um=2, unbound_ki_um=2
    )
    assert inhibitor.remaining_clint_fraction == 0.75
    inhibited = simulate_oral(p, MEDIUM, [120], amount_umol=1, inhibition=inhibitor)
    baseline = simulate_oral(p, MEDIUM, [120], amount_umol=1)
    assert inhibited.first_pass_loss_umol[-1] < baseline.first_pass_loss_umol[-1]
    assert inhibited.unbound_plasma_auc_um_h[-1] > baseline.unbound_plasma_auc_um_h[-1]
    null = simulate_oral(
        p, MEDIUM, [120], amount_umol=1, inhibition=replace(inhibitor, unbound_inhibitor_um=0)
    )
    np.testing.assert_allclose(null.central_umol, baseline.central_umol)
    with pytest.raises(ValueError):
        replace(inhibitor, unbound_ki_um=0)


@pytest.mark.parametrize("times", [[], [-1], [0, 0], [1, 0.9], [float("inf")], [float("nan")]])
def test_invalid_sampling(disposition: Disposition, times: list[float]) -> None:
    with pytest.raises(ValueError, match="times_h"):
        simulate_oral(disposition, MEDIUM, times, amount_umol=1)


def test_invalid_parameter_units(disposition: Disposition) -> None:
    with pytest.raises(ValueError, match="whole blood"):
        replace(disposition, blood_to_plasma_ratio=0.25)
    with pytest.raises(ValueError):
        replace(MEDIUM, partner_total_um=1)
    with pytest.raises(ValueError, match="amount_umol"):
        simulate_oral(disposition, MEDIUM, [0], amount_umol=0)


@pytest.fixture
def dialysis() -> DialysisRun:
    candidate = tourette_candidate()
    reference = Compound(compound_id="QA-reference", smiles="CCO")
    plan = plan_dialysis(candidate, reference, plate_id="fixture-01", seed=7)
    rows = []
    for well in plan.wells:
        # Different final volumes require amount-based recovery. All samples recover 100%.
        material, cp, cb = {
            "candidate": (candidate.compound_id, 8, 2),
            "reference": (reference.compound_id, 6, 3),
            "blank": (None, 0, 0),
        }[well.role]
        rows.append(
            DialysisReadout(
                well=well.well,
                compound_id=material,
                initial_plasma_um=(90 * cp + 110 * cb) / 100,
                final_plasma_um=cp,
                final_buffer_um=cb,
                initial_plasma_ul=100,
                final_plasma_ul=90,
                final_buffer_ul=110,
            )
        )
    return DialysisRun(
        plan=plan,
        observed_plate_id=plan.plate_id,
        run_id="synthetic-unit-test",
        protocol_version="fixture-v1",
        instrument_id="test-LCMS",
        completed_at="2026-01-01T12:00:00+00:00",
        raw_data_uri="fixture://observations",
        candidate_lot="testA",
        reference_lot="testB",
        plasma_lot="testP",
        matrix="human_plasma",
        ph=7.4,
        temperature_c=37,
        equilibrium_verified=True,
        limits=DialysisLimits(
            source="test fixture",
            quantification_limit_um=0.1,
            recovery_low=0.8,
            recovery_high=1.1,
            reference_fu_low=0.47,
            reference_fu_high=0.53,
            max_replicate_cv=0.15,
        ),
        observations=tuple(rows),
    )


def test_worklist_is_reproducible_and_executable_offline(dialysis: DialysisRun) -> None:
    plan = dialysis.plan
    assert plan_dialysis(plan.candidate, plan.reference, plate_id=plan.plate_id, seed=7) == plan
    assert plan_dialysis(plan.candidate, plan.reference, plate_id=plan.plate_id, seed=8) != plan
    assert len({w.well for w in plan.wells}) == 9
    with pytest.raises(ValueError, match="distinct compound"):
        plan_dialysis(plan.candidate, plan.candidate, plate_id="a")
    result = evaluate_dialysis(DialysisRun.model_validate_json(dialysis.model_dump_json()))
    assert result.fraction_unbound_plasma == pytest.approx(0.25)
    assert not result.reasons and result.candidate_cv == result.reference_cv == 0
    assert all(r.recovered_fraction == pytest.approx(1) for r in result.pairs if r.buffer_to_plasma)


@pytest.mark.parametrize(
    ("role", "change", "reason"),
    [
        ("blank", {"final_buffer_um": 0.1}, "blank contains"),
        ("candidate", {"final_plasma_ul": 35}, "mass recovery"),
        ("candidate", {"compound_id": "different"}, "identity"),
        ("candidate", {"final_buffer_um": 0}, "quantification"),
        ("reference", {"final_buffer_um": 2.5}, "outside acceptance"),
        ("candidate", {"initial_plasma_um": 9.5, "final_buffer_um": 3}, "replicate precision"),
    ],
)
def test_bad_robot_observation_blocks_release(
    dialysis: DialysisRun, role: str, change: dict, reason: str
) -> None:
    pos = next(w.well for w in dialysis.plan.wells if w.role == role)
    rows = tuple(o.model_copy(update=change) if o.well == pos else o for o in dialysis.observations)
    result = evaluate_dialysis(dialysis.model_copy(update={"observations": rows}))
    assert result.fraction_unbound_plasma is None
    assert reason in " ".join(result.reasons)


def test_incomplete_duplicate_or_unverified_plate_is_rejected(dialysis: DialysisRun) -> None:
    for change in (
        {"observed_plate_id": "other"},
        {"observations": dialysis.observations[:-1]},
        {"observations": dialysis.observations + dialysis.observations[:1]},
        {"equilibrium_verified": False},
    ):
        result = evaluate_dialysis(dialysis.model_copy(update=change))
        assert result.reasons and result.fraction_unbound_plasma is None
    with pytest.raises(ValidationError):
        DialysisLimits.model_validate(
            {**dialysis.limits.model_dump(), "quantification_limit_um": math.nan}
        )


def test_both_dossier_structures_and_measured_input_provenance(dialysis: DialysisRun) -> None:
    assert ge.human_twin.tourette_candidate() == dialysis.plan.candidate
    cases = [
        ("DRD1_v53_L1", "ts_scz_realistic_ligands_v53.json", "flagship_leads", "smiles"),
        ("DRD1_1", "cns_iter3_top_hits_dossier.json", "flagship_top12", "canonical_smiles"),
    ]
    for candidate, file, key, smiles in cases:
        row = json.loads((ROOT / "results/tables" / file).read_text())[key][0]
        assert row[smiles] == tourette_candidate(candidate).smiles
    binding = D1BindingMeasurement(
        compound=dialysis.plan.candidate,
        target="human DRD1",
        endpoint="equilibrium dissociation constant",
        kd_nm=5,
        material_lot="test",
        assay_conditions="synthetic test fixture",
        raw_data_uri="fixture://binding",
    )
    result = run_tourette_study(dialysis=dialysis, binding=binding)
    assert result["public_record"]["pubchem_cid"] == 81540663
    assert result["public_record"]["standard_inchikey"] == inchi.MolToInchiKey(
        Chem.MolFromSmiles(dialysis.plan.candidate.smiles)
    )
    fit = result["structure_informed_scenario"]
    assert fit["exposure"]["disposition"]["fraction_unbound_plasma"] == 0.25
    assert result["plasma_binding_evidence"]["run"]["matrix"] == "human_plasma"
    for s in [*result["scenarios"], fit["exposure"]]:
        brain = np.asarray(s["unbound_brain_nm"])
        np.testing.assert_allclose(
            s["equilibrium_d1_occupancy_fraction"], brain / (brain + 5), atol=1e-8
        )
    with pytest.raises(ValueError, match="different compound"):
        run_tourette_study("DRD1_1", dialysis=dialysis)
    with pytest.raises(ValueError, match="different compound"):
        run_tourette_study("DRD1_1", binding=binding)
    with pytest.raises(ValueError, match="QC failed"):
        run_tourette_study(dialysis=dialysis.model_copy(update={"equilibrium_verified": False}))


def test_supplied_solubility_replaces_only_matching_assay(dialysis: DialysisRun) -> None:
    sample = AqueousSolubilityMeasurement(
        compound=tourette_candidate(),
        endpoint="equilibrium solubility in buffered water",
        solubility_um=0.25,
        ph=6.8,
        temperature_c=37,
        material_lot="fixture",
        solid_form="free base",
        buffer="synthetic fixture",
        raw_data_uri="fixture://solubility",
    )
    with pytest.raises(ValueError, match="pH"):
        run_tourette_study(solubility=sample.model_copy(update={"ph": 7.4}))
    with pytest.raises(ValueError, match="different compound"):
        run_tourette_study("DRD1_1", solubility=sample)
    report = run_tourette_study(solubility=sample, dialysis=dialysis)
    informed = report["structure_informed_scenario"]
    assert informed["exposure"]["medium"]["free_solubility_um"] == 0.25
    assert report["solubility_evidence"]["material_lot"] == "fixture"
    assert len(informed["assay_sensitivity"]["corners"]) == 1
    assert informed["assay_sensitivity"]["corners"][0]["fraction_unbound_plasma"] == 0.25


def test_external_comparisons_reveal_model_errors() -> None:
    checks = compare_with_human_data()
    oral = checks["ecopipam_parent_total_plasma"]
    assert oral["fit"]["source"] != oral["external"]["source"]
    assert oral["modeled_cmax_ng_ml"] == pytest.approx(30.219, rel=1e-4)
    assert oral["relative_auc_error"] == pytest.approx(736 / 725 - 1)
    assert -0.50 < oral["relative_peak_error"] < -0.48
    brain = checks["linezolid_unbound_auc_ratio"]
    assert brain["passive_model_brain_to_plasma"] == pytest.approx(1)
    assert brain["relative_error_against_brain_isf"] == pytest.approx(1)


def test_cli_report_figure_and_plate(tmp_path: Path) -> None:
    runner = CliRunner()
    report, figure, plate = (tmp_path / n for n in ("out.json", "brain.png", "plan.json"))
    cmd = runner.invoke(
        app,
        ["human-twin", "--candidate", "DRD1_1", "--output", str(report), "--figure", str(figure)],
    )
    assert cmd.exit_code == 0, cmd.exception
    data = json.loads(report.read_text())
    assert len(data["scenarios"]) == 8 and data["d1_binding_evidence"] is None
    assert (
        data["clinical_endpoints"]["tic_response"]
        is data["clinical_endpoints"]["human_safety"]
        is None
    )
    assert all(s["equilibrium_d1_occupancy_fraction"] is None for s in data["scenarios"])
    assert data["public_record"]["pubchem_cid"] == 1863793
    assert data["public_record"]["standard_inchikey"] == inchi.MolToInchiKey(
        Chem.MolFromSmiles(data["compound"]["smiles"])
    )
    scenario = data["structure_informed_scenario"]
    assert scenario["fraction_unbound_plasma_basis"] == "structure estimate"
    corners = scenario["assay_sensitivity"]["corners"]
    assert len(corners) == 4 and max(c["fraction_unbound_plasma"] for c in corners) == 1.0
    score = scenario["assay_estimates"]["Biogen_log_Human_Unbound_Percent"]["estimate"]
    assert scenario["exposure"]["disposition"]["fraction_unbound_plasma"] == pytest.approx(
        10 ** score["y_pred"] / 100
    )
    assert (
        data["hepatic_competition_scenario"]["plasma_auc_ratio_to_structure_informed_scenario"] > 1
    )
    assert figure.read_bytes().startswith(b"\x89PNG")
    cmd = runner.invoke(
        app,
        [
            "dialysis-plan",
            "--plate-id",
            "example",
            "--reference-id",
            "ref",
            "--reference-smiles",
            "CCO",
            "--output",
            str(plate),
        ],
    )
    assert cmd.exit_code == 0 and len(json.loads(plate.read_text())["wells"]) == 9
    assert runner.invoke(app, ["human-twin", "--candidate", "unregistered"]).exit_code != 0
