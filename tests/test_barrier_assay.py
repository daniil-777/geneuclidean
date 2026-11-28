"""Assay-level acceptance, sample inventories, and measured directional transport."""

from __future__ import annotations

import math
from hashlib import sha256
from statistics import mean

import pytest
from pydantic import ValidationError

from geneuclidean.human_twin.barrier import (
    BarrierLimits,
    BarrierRange,
    BarrierRun,
    BarrierSample,
    BarrierTrace,
    BarrierWell,
    evaluate_barrier,
)
from geneuclidean.human_twin.dialysis import Compound


def _samples(rate: float) -> tuple[BarrierSample, ...]:
    # A known amount enters the receiver every ten minutes. Donor samples at
    # 0 and 20 min are discarded. Receiver aliquots at 10 and 20 min remove
    # respectively 25/150 and 15/135 of the then-present amount.
    d20 = (1900 - 20 * rate) / 190
    r10 = 10 * rate / 150
    r20 = (20 * rate - 25 * r10) / 135
    return (
        BarrierSample(time_min=0, donor_um=10, receiver_um=None, donor_sample_ul=10),
        BarrierSample(
            time_min=10,
            donor_um=(1900 - 10 * rate) / 190,
            receiver_um=r10,
            receiver_sample_ul=25,
            receiver_replacement_ul=10,
        ),
        BarrierSample(
            time_min=20,
            donor_um=d20,
            receiver_um=r20,
            donor_sample_ul=10,
            receiver_sample_ul=15,
            receiver_replacement_ul=15,
        ),
        BarrierSample(
            time_min=30,
            donor_um=(1900 - 10 * d20 - 30 * rate) / 180,
            receiver_um=(30 * rate - 25 * r10 - 15 * r20) / 135,
            donor_sample_ul=180,
            receiver_sample_ul=135,
        ),
    )


@pytest.fixture
def run() -> BarrierRun:
    candidate = Compound(compound_id="fixture-A", smiles="C[C@H](O)C(=O)O")
    reference = Compound(compound_id="fixture-R", smiles="CCO")
    layout, traces = [], []
    for role, compound, lot, ab, ba in (
        ("candidate", candidate, "A-lot", 1.2, 3.6),
        ("reference", reference, "R-lot", 0.8, 1.6),
    ):
        for direction, rate in (("A_to_B", ab), ("B_to_A", ba)):
            for i in range(2):
                well = f"{role}-{direction}-{i}"
                layout.append(BarrierWell(well=well, role=role, direction=direction))
                traces.append(
                    BarrierTrace(
                        well=well,
                        direction=direction,
                        analyte=compound,
                        material_lot=lot,
                        initial_donor_ul=200,
                        initial_receiver_ul=150,
                        net_teer_before_ohm_cm2=180,
                        net_teer_after_ohm_cm2=160,
                        samples=_samples(rate),
                    )
                )
        well = f"{role}-matrix-blank"
        layout.append(BarrierWell(well=well, role=f"{role}_blank", direction="A_to_B"))
        traces.append(
            BarrierTrace(
                well=well,
                direction="A_to_B",
                analyte=compound,
                material_lot=None,
                initial_donor_ul=200,
                initial_receiver_ul=150,
                samples=(
                    BarrierSample(time_min=0, donor_um=None, receiver_um=0),
                    BarrierSample(time_min=30, donor_um=0, receiver_um=None),
                ),
            )
        )
    return BarrierRun(
        run_id="synthetic-run",
        plate_id="plate-1",
        observed_plate_id="plate-1",
        protocol_version="fixture-v1",
        instrument_id="LC-MS-fixture",
        completed_at="2026-01-01T12:00:00Z",
        raw_data_uri="fixture://parent-peaks",
        raw_data_sha256="1" * 64,
        cell_model="synthetic barrier insert",
        cell_lot="cells-1",
        apical_medium="A1",
        basolateral_medium="B1",
        apical_ph=7.4,
        basolateral_ph=7.4,
        temperature_c=37,
        area_cm2=0.5,
        fit_start_min=0,
        fit_end_min=30,
        candidate=candidate,
        candidate_lot="A-lot",
        candidate_nominal_donor_um=10,
        reference=reference,
        reference_lot="R-lot",
        reference_nominal_donor_um=10,
        limits=BarrierLimits(
            source="fixture absolute and relative performance windows",
            candidate_lloq_um=0.01,
            reference_lloq_um=0.01,
            aqueous_recovery=BarrierRange(low=0.95, high=1.05),
            reference_ab_cm_s=BarrierRange(low=2.5e-6, high=2.8e-6),
            reference_efflux_ratio=BarrierRange(low=1.8, high=2.2),
            min_net_teer_ohm_cm2=100,
            max_loading_error_fraction=0.1,
            max_donor_change_fraction=0.2,
            max_sink_ratio=0.2,
            min_fit_r_squared=0.99,
            max_replicate_cv=0.1,
        ),
        layout=layout,
        traces=traces,
    )


def _trace(run: BarrierRun, trace: BarrierTrace, **changes) -> BarrierRun:
    updated = BarrierTrace.model_validate({**trace.model_dump(), **changes})
    rows = tuple(updated if t.well == trace.well else t for t in run.traces)
    return run.model_copy(update={"traces": rows})


def _point(run: BarrierRun, trace: BarrierTrace, i: int, **changes) -> BarrierRun:
    samples = list(trace.samples)
    samples[i] = BarrierSample.model_validate({**samples[i].model_dump(), **changes})
    return _trace(run, trace, samples=samples)


def _failed(run: BarrierRun, reason: str) -> None:
    result = evaluate_barrier(run)
    assert result.papp_ab_cm_s is result.papp_ba_cm_s is result.efflux_ratio is None
    assert any(reason in text for text in result.reasons), result.reasons


def test_bidirectional_flux_units_recovery_and_provenance(run: BarrierRun) -> None:
    result = evaluate_barrier(BarrierRun.model_validate_json(run.model_dump_json()))
    assert not result.reasons
    assert result.papp_ab_cm_s == pytest.approx(4e-6)
    assert result.papp_ba_cm_s == pytest.approx(12e-6)
    assert result.efflux_ratio == pytest.approx(3)
    assert result.reference_efflux_ratio == pytest.approx(2)
    assert result.assay_conditions.cell_lot == run.cell_lot
    assert result.assay_conditions.fit_end_min == 30
    assert "traces" not in result.assay_conditions.model_dump()
    assert result.qc_source == run.limits.source
    assert result.raw_data_sha256 == "1" * 64
    assert result.input_sha256 == sha256(run.model_dump_json().encode()).hexdigest()
    insert = result.inserts[0]
    assert insert.fit_point_count == 3
    assert insert.r_squared == pytest.approx(1)
    assert insert.slope_pmol_min == pytest.approx(1.2)
    assert 0 < insert.max_receiver_to_donor_ratio < 0.1
    assert 0 < insert.max_donor_change_fraction < 0.1
    assert [p.cumulative_receiver_pmol for p in insert.mass_points] == pytest.approx(
        [0, 12, 24, 36]
    )
    assert [p.donor_volume_ul for p in insert.mass_points] == [200, 190, 190, 180]
    assert [p.receiver_volume_ul for p in insert.mass_points] == [150, 150, 135, 135]
    assert all(
        fit.aqueous_recovery_fraction == pytest.approx(1)
        for fit in result.inserts
        if fit.mass_points
    )
    assert all(group.n_inserts == 2 and group.replicate_cv == 0 for group in result.directions)


def test_ratio_of_mean_independent_permeabilities(run: BarrierRun) -> None:
    # A candidate pair with a small real difference in permeability.
    row = run.traces[0]
    run = _trace(run, row, samples=_samples(1.26))
    run = run.model_copy(update={"fit_start_min": 10.0})
    result = evaluate_barrier(run)
    assert not result.reasons
    assert result.efflux_ratio == pytest.approx(3.6 / mean([1.2, 1.26]))
    assert result.inserts[0].fit_point_count == 3
    assert result.directions[0].replicate_cv > 0
    assert result.inserts[0].aqueous_recovery_fraction == pytest.approx(1)


@pytest.mark.parametrize("fit_start", [0, 10])
def test_prespecified_window_keeps_lag_out_of_free_intercept_fit(
    run: BarrierRun, fit_start: float
) -> None:
    # Use three delayed amounts with no draws. A zero-intercept regression over
    # the selected three times would underestimate each true slope.
    rows = []
    for t, well in zip(run.traces, run.layout, strict=True):
        if well.role.endswith("_blank"):
            rows.append(t)
            continue
        rate = {
            ("candidate", "A_to_B"): 1.2,
            ("candidate", "B_to_A"): 3.6,
            ("reference", "A_to_B"): 0.8,
            ("reference", "B_to_A"): 1.6,
        }[well.role, well.direction]
        pts = tuple(
            BarrierSample(time_min=m, donor_um=(2000 - q) / 200, receiver_um=q / 150)
            for m, q in ((0, 0), (10, 5 * rate), (20, 15 * rate), (30, 25 * rate))
        )
        rows.append(t.model_copy(update={"samples": pts}))
    result = evaluate_barrier(
        run.model_copy(update={"fit_start_min": fit_start, "traces": tuple(rows)})
    )
    assert not result.reasons
    assert result.papp_ab_cm_s == pytest.approx(4e-6)
    assert result.papp_ba_cm_s == pytest.approx(12e-6)


def test_no_censored_concentration_enters_a_released_slope(run: BarrierRun) -> None:
    # All time-zero observations are inventory boundaries. Neither a reported
    # zero nor a sub-LLOQ peak supplies another point for the slope.
    row = run.traces[0]
    for baseline in (None, 0, 0.009):
        observed = _point(run, row, 0, receiver_um=baseline)
        result = evaluate_barrier(observed)
        assert not result.reasons
        assert result.papp_ab_cm_s == pytest.approx(4e-6)
        assert result.inserts[0].fit_point_count == 3
        _failed(observed.model_copy(update={"fit_end_min": 20}), "three quantified postdose")
    _failed(_point(run, row, 2, receiver_um=0.009), "unquantifiable")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"net_teer_before_ohm_cm2": None}, "integrity"),
        ({"net_teer_after_ohm_cm2": 99}, "integrity"),
        ({"material_lot": "other"}, "lot"),
        ({"direction": "B_to_A"}, "direction"),
        ({"analyte": {"compound_id": "fixture-A", "smiles": "C[C@@H](O)C(=O)O"}}, "identity"),
    ],
)
def test_insert_identity_direction_and_integrity(
    run: BarrierRun, changes: dict, reason: str
) -> None:
    _failed(_trace(run, run.traces[0], **changes), reason)


@pytest.mark.parametrize(
    ("i", "changes", "reason"),
    [
        (0, {"receiver_um": 0.01}, "baseline"),
        (0, {"donor_um": None}, "Initial donor"),
        (0, {"donor_um": 12}, "loading"),
        (1, {"receiver_um": None}, "unquantifiable"),
        (1, {"donor_um": 6}, "Donor concentration changed"),
        (1, {"receiver_um": 4}, "sink limit"),
        (2, {"receiver_um": 0.30}, "linear receiver flux"),
        (3, {"donor_um": 7}, "mass recovery"),
    ],
)
def test_censored_mass_and_flux_qc(run: BarrierRun, i: int, changes: dict, reason: str) -> None:
    _failed(_point(run, run.traces[0], i, **changes), reason)


def test_censored_draw_outside_fit_propagates_to_recovery(run: BarrierRun) -> None:
    # The final fit has three quantitative receiver readings; the unknown amount
    # removed from the donor before that window still prevents mass recovery.
    row = run.traces[0]
    samples = list(row.samples)
    samples.insert(
        1,
        BarrierSample(
            time_min=5, donor_um=None, receiver_um=None, donor_sample_ul=1, donor_replacement_ul=1
        ),
    )
    bad = _trace(run, row, samples=samples).model_copy(update={"fit_start_min": 10.0})
    _failed(bad, "mass recovery is unquantifiable")
    fit = evaluate_barrier(bad).inserts[0]
    assert fit.fit_point_count == 3 and fit.aqueous_recovery_fraction is None
    samples[1] = samples[1].model_copy(update={"donor_sample_ul": 0, "donor_replacement_ul": 0})
    good = _trace(run, row, samples=samples).model_copy(update={"fit_start_min": 10.0})
    assert not evaluate_barrier(good).reasons


def test_censored_receiver_aliquot_before_fit_leaves_transport_unquantifiable(
    run: BarrierRun,
) -> None:
    trace = run.traces[0]
    samples = list(trace.samples)
    samples.insert(
        1,
        BarrierSample(
            time_min=5,
            donor_um=9.98,
            receiver_um=0.009,
            receiver_sample_ul=2,
            receiver_replacement_ul=2,
        ),
    )
    altered = _trace(run, trace, samples=samples).model_copy(update={"fit_start_min": 10})
    _failed(altered, "unquantifiable")
    fit = evaluate_barrier(altered).inserts[0]
    assert fit.papp_cm_s is None
    assert all(p.cumulative_receiver_pmol is None for p in fit.mass_points[1:])


def test_reference_blank_and_replicate_qc(run: BarrierRun) -> None:
    blank = next(t for t in run.traces if t.well == "reference-matrix-blank")
    _failed(_point(run, blank, 1, receiver_um=0.01), "blank contains")
    _failed(_trace(run, blank, material_lot="spiked"), "lot")
    _failed(_trace(run, run.traces[0], samples=_samples(0.5)), "replicate precision")
    _failed(run.model_copy(update={"area_cm2": 2.0}), "Reference A-to-B")
    new_limits = run.limits.model_copy(
        update={"reference_efflux_ratio": BarrierRange(low=3, high=5)}
    )
    _failed(run.model_copy(update={"limits": new_limits}), "Reference efflux")
    _failed(run.model_copy(update={"fit_start_min": 20}), "Fewer than three")


def test_wrong_plate_missing_duplicate_and_unplanned_inserts_cannot_release(
    run: BarrierRun,
) -> None:
    _failed(run.model_copy(update={"observed_plate_id": "wrong"}), "plate identity")
    for rows in (
        run.traces[:-1],
        run.traces + run.traces[:1],
        run.traces + (run.traces[0].model_copy(update={"well": "unknown"}),),
    ):
        _failed(run.model_copy(update={"traces": rows}), "exactly once")
    duplicate = evaluate_barrier(run.model_copy(update={"traces": run.traces + run.traces[:1]}))
    assert duplicate.directions[0].n_inserts == 1


def test_impossible_inventory_and_invalid_input(run: BarrierRun) -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        _point(run, run.traces[0], 1, receiver_sample_ul=151)
    with pytest.raises(ValidationError, match="contain liquid"):
        _point(run, run.traces[0], 0, donor_sample_ul=200)
    with pytest.raises(ValidationError, match="increase"):
        _point(run, run.traces[0], 1, time_min=0)
    with pytest.raises(ValidationError, match="time-zero"):
        _trace(run, run.traces[0], samples=run.traces[0].samples[1:])
    with pytest.raises(ValidationError):
        _point(run, run.traces[0], 1, receiver_um=math.nan)
    with pytest.raises(ValidationError):
        BarrierSample(time_min=0, donor_um=0)
    with pytest.raises(ValidationError, match="distinct compound"):
        BarrierRun.model_validate({**run.model_dump(), "reference": run.candidate.model_dump()})
    with pytest.raises(ValidationError, match="independent"):
        BarrierRun.model_validate({**run.model_dump(), "layout": run.layout[1:]})
    with pytest.raises(ValidationError):
        BarrierRange(low=2, high=1)
