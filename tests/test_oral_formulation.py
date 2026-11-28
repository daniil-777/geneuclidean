"""Closed-form finite-rate gut inventories and their coupling to systemic exposure."""

from dataclasses import replace

import numpy as np
import pytest

from geneuclidean.human_twin.exposure import (
    BrainExposureDriver,
    Disposition,
    GastricStage,
    OralFormulation,
    OralMedium,
    fit_sink_dissolution,
    simulate_oral,
)

MEDIUM = OralMedium(ph=6.8, volume_l=0.5, free_solubility_um=4.0)
NO_SINK = Disposition(
    absorption_per_h=0.0,
    gut_transit_per_h=0.0,
    gut_escape_fraction=1.0,
    central_volume_l=5.0,
    peripheral_volume_l=10.0,
    peripheral_exchange_l_h=0.0,
    brain_unbound_volume_l=1.0,
    bbb_in_l_h=0.0,
    bbb_out_l_h=0.0,
    fraction_unbound_plasma=0.5,
    blood_to_plasma_ratio=1.0,
    liver_blood_flow_l_h=90.0,
    liver_unbound_clint_l_h=0.0,
    other_plasma_clearance_l_h=0.0,
)
TIMES = [0.0, 0.1, 0.5, 2.0, 6.0, 24.0]


@pytest.mark.parametrize("initial_liquid", [0.0, 0.25, 0.95, 1.0])
def test_solid_and_supersaturated_liquid_relax_to_the_same_capacity(initial_liquid: float) -> None:
    medium = replace(MEDIUM, partner_total_um=8.0, partner_kd_um=4.0)
    # At saturation, free = 4 µM and half of the 8 µM partner pool is bound.
    capacity_umol = 4.0
    assert medium.dissolved_capacity_umol == pytest.approx(capacity_umol)
    load = 10.0
    form = OralFormulation(
        dissolution_per_h=2.0, precipitation_per_h=0.3, initial_dissolved_fraction=initial_liquid
    )
    run = simulate_oral(NO_SINK, medium, TIMES, amount_umol=load, formulation=form)
    rate = (
        form.dissolution_per_h
        if load * initial_liquid < capacity_umol
        else form.precipitation_per_h
    )
    expected = capacity_umol + (load * initial_liquid - capacity_umol) * np.exp(
        -rate * np.array(TIMES)
    )
    np.testing.assert_allclose(run.gut_dissolved_umol, expected, atol=1e-8)
    np.testing.assert_allclose(run.gut_solid_umol, load - expected, atol=1e-8)
    np.testing.assert_allclose(run.gut_umol, load, atol=1e-9)
    np.testing.assert_allclose(run.accounted_umol, load, atol=1e-9)

    equilibrium = simulate_oral(NO_SINK, medium, [0.0], amount_umol=load)
    assert equilibrium.gut_dissolved_umol is not None and equilibrium.gut_solid_umol is not None
    assert equilibrium.gut_dissolved_umol[0] == pytest.approx(capacity_umol)
    assert equilibrium.gut_solid_umol[0] == pytest.approx(load - capacity_umol)


def test_dissolution_absorption_and_transit_close_for_equal_rates() -> None:
    # Below saturation, S -> D -> central is two serial first-order processes.
    # With equal rates k and shared transit loss q, D(t) = S(0) k t exp(-(k+q)t).
    k, q, load = 1.4, 0.3, 1.0
    p = replace(NO_SINK, absorption_per_h=k, gut_transit_per_h=q, gut_escape_fraction=0.6)
    form = OralFormulation(dissolution_per_h=k, precipitation_per_h=0.5)
    run = simulate_oral(p, MEDIUM, TIMES, amount_umol=load, formulation=form)
    t = np.array(TIMES)
    kt = (k + q) * t
    solid = load * np.exp(-kt)
    dissolved = solid * k * t
    absorbed = load * (k / (k + q)) ** 2 * (1.0 - (1.0 + kt) * np.exp(-kt))
    np.testing.assert_allclose(run.gut_solid_umol, solid, atol=1e-9)
    np.testing.assert_allclose(run.gut_dissolved_umol, dissolved, atol=1e-9)
    np.testing.assert_allclose(run.central_umol, absorbed * 0.6, atol=1e-9)
    np.testing.assert_allclose(run.first_pass_loss_umol, absorbed * 0.4, atol=1e-9)
    np.testing.assert_allclose(
        run.transit_loss_umol, load - solid - dissolved - absorbed, atol=1e-9
    )
    np.testing.assert_allclose(run.accounted_umol, load, atol=1e-10)


def test_liquid_partner_binding_preserves_transient_supersaturation() -> None:
    # Zero phase-exchange rates leave the initially predissolved load in solution.
    # For free concentration f, total dissolved is f + P*f/(Kd + f) and
    # d(total dissolved)/dt = -ka*f. Integrating analytically gives the check below.
    medium = OralMedium(
        ph=7.0, volume_l=1.0, free_solubility_um=0.1, partner_total_um=1, partner_kd_um=1
    )
    form = OralFormulation(dissolution_per_h=0, precipitation_per_h=0, initial_dissolved_fraction=1)
    ka = 1.3
    t = np.array([0.0, 0.03, 0.4, 1.0, 2.0])
    run = simulate_oral(
        replace(NO_SINK, absorption_per_h=ka), medium, t.tolist(), amount_umol=3, formulation=form
    )
    assert run.gut_dissolved_umol is not None
    dissolved = run.gut_dissolved_umol
    free = (dissolved - 2.0 + np.sqrt((2.0 - dissolved) ** 2 + 4.0 * dissolved)) / 2.0
    primitive = np.log(free) + np.log(free / (1.0 + free)) + 1.0 / (1.0 + free)
    np.testing.assert_allclose(primitive[0] - primitive, ka * t, atol=1e-8)
    np.testing.assert_array_equal(run.gut_solid_umol, 0.0)
    np.testing.assert_allclose(run.accounted_umol, 3.0, atol=1e-9)


@pytest.mark.parametrize(
    "rates",
    [
        {"dissolution_per_h": -1, "precipitation_per_h": 1},
        {"dissolution_per_h": float("nan"), "precipitation_per_h": 1},
        {"dissolution_per_h": 1, "precipitation_per_h": float("inf")},
        {"dissolution_per_h": 1, "precipitation_per_h": 1, "initial_dissolved_fraction": 1.01},
        {
            "dissolution_per_h": 1,
            "precipitation_per_h": 1,
            "initial_dissolved_fraction": float("nan"),
        },
    ],
)
def test_formulation_parameters_must_be_physical(rates: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        OralFormulation(**rates)


def test_sink_fit_recovers_the_rate_used_by_the_oral_formulation() -> None:
    times = np.array([0.0, 0.07, 0.4, 0.4, 1.7, 2.5])
    rate, load, initial = 1.75, 0.35, 0.2
    observed = load * (1.0 - (1.0 - initial) * np.exp(-rate * times))
    fit = fit_sink_dissolution(
        MEDIUM,
        times.tolist(),
        observed.tolist(),
        amount_umol=load,
        initial_dissolved_fraction=initial,
    )
    assert fit.dissolution_per_h == pytest.approx(rate, abs=1e-9)
    assert fit.rmse_umol < 1e-9
    assert fit.capacity_to_load_ratio == pytest.approx(2.0 / load)
    assert fit.n_observations == len(times)
    form = OralFormulation(
        dissolution_per_h=fit.dissolution_per_h,
        precipitation_per_h=0.0,
        initial_dissolved_fraction=initial,
    )
    run = simulate_oral(
        NO_SINK, MEDIUM, np.unique(times).tolist(), amount_umol=load, formulation=form
    )
    np.testing.assert_allclose(run.gut_dissolved_umol, np.unique(observed), atol=1e-9)
    observed[3] += 0.02
    noisy = fit_sink_dissolution(
        MEDIUM,
        times.tolist(),
        observed.tolist(),
        amount_umol=load,
        initial_dissolved_fraction=initial,
    )
    assert rate < noisy.dissolution_per_h < 1.15 * rate
    assert noisy.rmse_umol > 0


def test_measured_paracetamol_tablet_dissolution_with_held_out_final_sample() -> None:
    # Amidol 500 mg, 900 mL phosphate buffer, 37 °C, 50 rpm paddle, n=3,
    # sampling-corrected means in Table 7. DOI: 10.5935/2595-0118.20240044-en.
    times = np.array([5, 10, 15, 30, 45, 60]) / 60
    percent_dissolved = np.array([29.34, 38.07, 65.07, 85.77, 90.72, 91.62])
    molecular_weight = 151.16  # g/mol
    load = 500 * 1000 / molecular_weight  # µmol from mg of paracetamol
    # Shaw et al. measured ~20.3 mg/mL across pH 1.2–8 (10.1080/03639040500215784).
    medium = OralMedium(ph=5.8, volume_l=0.9, free_solubility_um=20.3e6 / molecular_weight)
    fit = fit_sink_dissolution(
        medium,
        times[:-1].tolist(),
        (load * percent_dissolved[:-1] / 100).tolist(),
        amount_umol=load,
    )
    assert fit.dissolution_per_h == pytest.approx(3.667, abs=0.001)
    assert fit.capacity_to_load_ratio == pytest.approx(36.54)
    assert 4.5 < 100 * fit.rmse_umol / load < 4.7  # percentage points of the load
    held_out_percent = -100 * np.expm1(-fit.dissolution_per_h * times[-1])
    assert held_out_percent - percent_dissolved[-1] == pytest.approx(5.82, abs=0.02)


def test_sink_fit_includes_measured_over_recovery_in_residuals() -> None:
    fit = fit_sink_dissolution(MEDIUM, [0.1, 2.0, 10.0], [0.2, 0.9, 1.02], amount_umol=1)
    assert fit.rmse_umol > 0.02 / np.sqrt(3)


@pytest.mark.parametrize(
    ("times", "dissolved", "amount", "initial"),
    [
        ([0.0, 1.0, 2.0], [0.0, 0.2, 0.4], 2.0, 0.0),  # load reaches capacity
        ([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], 1.0, 0.0),
        ([0.0, 1.0, 2.0], [0.0, 0.2, 0.2], 1.0, 0.0),
        ([0.0, 1.0, 1.0], [0.0, 0.2, 0.4], 1.0, 0.0),
        ([0.0, 1.0, 2.0], [0.0, 0.2], 1.0, 0.0),
        ([0.0, 1.0, 2.0], [0.0, -0.5, 1.01], 1.0, 0.0),
        ([0.0, 1.0, 2.0], [0.0, 0.5, float("nan")], 1.0, 0.0),
        ([-1.0, 1.0, 2.0], [0.0, 0.5, 0.8], 1.0, 0.0),
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], 1.0, 1.0),
    ],
)
def test_sink_fit_rejects_invalid_or_unresolved_measurements(
    times: list[float], dissolved: list[float], amount: float, initial: float
) -> None:
    with pytest.raises(ValueError):
        fit_sink_dissolution(
            MEDIUM, times, dissolved, amount_umol=amount, initial_dissolved_fraction=initial
        )


@pytest.mark.parametrize("solid_emptying", [0.0, 0.35])
def test_solid_and_liquid_empty_independently_into_matching_intestinal_pools(
    solid_emptying: float,
) -> None:
    fraction, liquid_emptying, load = 0.35, 1.3, 4.0
    stage = GastricStage(
        medium=MEDIUM,
        formulation=OralFormulation(
            dissolution_per_h=0, precipitation_per_h=0, initial_dissolved_fraction=fraction
        ),
        solid_emptying_per_h=solid_emptying,
        liquid_emptying_per_h=liquid_emptying,
    )
    run = simulate_oral(
        NO_SINK,
        MEDIUM,
        TIMES,
        amount_umol=load,
        gastric_stage=stage,
        formulation=OralFormulation(dissolution_per_h=0, precipitation_per_h=0),
    )
    t = np.array(TIMES)
    gastric_liquid = load * fraction * np.exp(-liquid_emptying * t)
    gastric_solid = load * (1 - fraction) * np.exp(-solid_emptying * t)
    np.testing.assert_allclose(run.gastric_dissolved_umol, gastric_liquid, atol=1e-8)
    np.testing.assert_allclose(run.gastric_solid_umol, gastric_solid, atol=1e-8)
    np.testing.assert_allclose(
        run.intestinal_dissolved_umol, load * fraction - gastric_liquid, atol=1e-8
    )
    np.testing.assert_allclose(
        run.intestinal_solid_umol, load * (1 - fraction) - gastric_solid, atol=1e-8
    )
    np.testing.assert_allclose(run.gut_dissolved_umol, load * fraction, atol=1e-9)
    np.testing.assert_allclose(run.gut_solid_umol, load * (1 - fraction), atol=1e-9)
    np.testing.assert_allclose(run.accounted_umol, load, atol=1e-9)


def test_gastric_transfer_into_lower_capacity_intestine_preserves_supersaturation() -> None:
    load, capacity, emptying, precipitation = 4.0, 1.0, 1.3, 0.4
    stage = GastricStage(
        medium=OralMedium(ph=2.0, volume_l=0.25, free_solubility_um=20.0),
        formulation=OralFormulation(
            dissolution_per_h=1, precipitation_per_h=1, initial_dissolved_fraction=1
        ),
        solid_emptying_per_h=0,
        liquid_emptying_per_h=emptying,
    )
    medium = OralMedium(ph=6.8, volume_l=0.5, free_solubility_um=2.0)
    t = np.array(TIMES)
    gastric = load * np.exp(-emptying * t)
    # The initially empty intestine crosses capacity when A*(1-exp(-ke*t)) = C.
    crossed_at = -np.log1p(-capacity / load) / emptying
    after = t > crossed_at
    tx = t[after]
    expected = load - gastric
    expected[after] = capacity + load * emptying / (precipitation - emptying) * (
        np.exp(-emptying * tx) - np.exp(-emptying * crossed_at - precipitation * (tx - crossed_at))
    )
    run = simulate_oral(
        NO_SINK,
        medium,
        TIMES,
        amount_umol=load,
        gastric_stage=stage,
        formulation=OralFormulation(dissolution_per_h=1, precipitation_per_h=precipitation),
    )
    np.testing.assert_allclose(run.intestinal_dissolved_umol, expected, atol=2e-8)
    np.testing.assert_allclose(run.intestinal_solid_umol, load - gastric - expected, atol=2e-8)
    np.testing.assert_allclose(run.gastric_dissolved_umol, gastric, atol=1e-8)
    np.testing.assert_allclose(run.accounted_umol, load, atol=1e-9)
    instant = simulate_oral(NO_SINK, medium, TIMES, amount_umol=load, gastric_stage=stage)
    np.testing.assert_allclose(
        instant.intestinal_dissolved_umol, np.minimum(load - gastric, capacity), atol=1e-8
    )
    np.testing.assert_allclose(instant.accounted_umol, load, atol=1e-9)


def test_gastric_dissolution_uses_its_own_capacity_and_cannot_absorb_drug() -> None:
    load = 6.0
    gastric_rate = 0.75
    stage = GastricStage(
        medium=OralMedium(ph=1.5, volume_l=0.25, free_solubility_um=3.0),
        formulation=OralFormulation(dissolution_per_h=gastric_rate, precipitation_per_h=0),
        solid_emptying_per_h=0,
        liquid_emptying_per_h=0,
    )
    run = simulate_oral(
        replace(NO_SINK, absorption_per_h=2, gut_transit_per_h=1),
        MEDIUM,
        TIMES,
        amount_umol=load,
        gastric_stage=stage,
    )
    expected = 0.75 * -np.expm1(-gastric_rate * np.array(TIMES))
    np.testing.assert_allclose(run.gastric_dissolved_umol, expected, atol=1e-8)
    np.testing.assert_allclose(run.gastric_solid_umol, load - expected, atol=1e-8)
    np.testing.assert_array_equal(run.intestinal_dissolved_umol, 0)
    np.testing.assert_array_equal(run.central_umol, 0)
    np.testing.assert_array_equal(run.transit_loss_umol, 0)


def test_predissolved_dose_reaches_plasma_after_emptying_and_absorption() -> None:
    k, load = 1.3, 0.4
    stage = GastricStage(
        medium=MEDIUM,
        formulation=OralFormulation(
            dissolution_per_h=0, precipitation_per_h=0, initial_dissolved_fraction=1
        ),
        solid_emptying_per_h=0,
        liquid_emptying_per_h=k,
    )
    run = simulate_oral(
        replace(NO_SINK, absorption_per_h=k), MEDIUM, TIMES, amount_umol=load, gastric_stage=stage
    )
    kt = k * np.array(TIMES)
    gastric = load * np.exp(-kt)
    intestinal = gastric * kt
    np.testing.assert_allclose(run.intestinal_dissolved_umol, intestinal, atol=1e-9)
    np.testing.assert_allclose(run.central_umol, load - gastric - intestinal, atol=1e-9)
    np.testing.assert_allclose(run.accounted_umol, load, atol=1e-9)


@pytest.mark.parametrize(("solid", "liquid"), [(-1.0, 1), (1, float("inf")), (float("nan"), 1)])
def test_gastric_emptying_rates_must_be_finite_and_nonnegative(solid: float, liquid: float) -> None:
    with pytest.raises(ValueError, match="emptying_per_h"):
        GastricStage(
            medium=MEDIUM,
            formulation=OralFormulation(dissolution_per_h=1, precipitation_per_h=1),
            solid_emptying_per_h=solid,
            liquid_emptying_per_h=liquid,
        )


def test_continuous_brain_driver_resolves_unsampled_exposure_on_the_solver_mesh() -> None:
    # Equal irreversible parent transfer rates in gut -> central -> brain.
    k, load = 1.8, 1.2
    p = replace(
        NO_SINK,
        absorption_per_h=k,
        central_volume_l=1,
        brain_unbound_volume_l=2,
        bbb_in_l_h=2 * k,
    )
    profile = simulate_oral(p, MEDIUM, [6.0], amount_umol=load, dense_brain=True)
    driver = profile.brain_driver
    assert driver is not None and len(driver.breakpoints_h) > 2
    assert driver.breakpoints_h[0] == 0 and driver.breakpoints_h[-1] == 6
    assert np.all(np.diff(driver.breakpoints_h) > 0)
    t = np.array([0.0, 0.03, 0.5, 1.7, 3.8, 6.0])
    expected = 1000 * load / 2 * (1 - (1 + k * t) * np.exp(-k * t))
    np.testing.assert_allclose([driver.unbound_nm(x) for x in t], expected, atol=2e-6)
    assert driver.unbound_nm(6) == pytest.approx(1000 * profile.unbound_brain_um[-1])
    for outside in [-0.1, 6.01, np.nan]:
        with pytest.raises(ValueError):
            driver.unbound_nm(outside)
    baseline = simulate_oral(p, MEDIUM, [0.0], amount_umol=load)
    assert baseline.brain_driver is None
    np.testing.assert_array_equal(baseline.gastric_dissolved_umol, 0)
    np.testing.assert_array_equal(baseline.gastric_solid_umol, 0)
    np.testing.assert_array_equal(baseline.intestinal_dissolved_umol, baseline.gut_dissolved_umol)
    zero = simulate_oral(p, MEDIUM, [0.0], amount_umol=load, dense_brain=True).brain_driver
    assert zero is not None and zero.unbound_nm(0) == 0
    np.testing.assert_array_equal(zero.breakpoints_h, [0.0])
    assert zero.peak() == (0.0, 0.0)


def test_brain_peak_resolves_internal_maximum_between_sparse_reporting_times() -> None:
    # Plasma/brain exchange and clearance yield eigenvalues -1 and -3 per hour.
    # With absorption at 2/h, brain fraction = exp(-t)*(1-exp(-t))²/2.
    p = replace(
        NO_SINK,
        absorption_per_h=2,
        central_volume_l=1,
        brain_unbound_volume_l=1,
        bbb_in_l_h=1,
        bbb_out_l_h=2,
        other_plasma_clearance_l_h=1.5,
    )
    profile = simulate_oral(p, MEDIUM, [0, 8], amount_umol=1, dense_brain=True)
    assert profile.brain_driver is not None
    peak_nm, peak_h = profile.brain_driver.peak()
    assert peak_h == pytest.approx(np.log(3), abs=2e-6)
    assert peak_nm == pytest.approx(2000 / 27, abs=2e-6)
    assert peak_nm > 100 * 1000 * profile.unbound_brain_um.max()
    assert profile.brain_driver.unbound_nm(peak_h) == pytest.approx(peak_nm)

    rising = simulate_oral(p, MEDIUM, [0.25], amount_umol=1, dense_brain=True)
    assert rising.brain_driver is not None
    assert rising.brain_driver.peak() == pytest.approx((1000 * rising.unbound_brain_um[-1], 0.25))
    unexposed = simulate_oral(NO_SINK, MEDIUM, [3.0], amount_umol=1, dense_brain=True)
    assert unexposed.brain_driver is not None and unexposed.brain_driver.peak() == (0.0, 0.0)


def test_brain_peak_finds_both_extrema_in_a_step_and_selects_the_first_equal_peak() -> None:
    # Two adjacent copies of a cubic, mirrored so concentrations agree at the join.
    # Each contains two extrema. Both maxima exceed the left and right endpoints.
    def state(t: float | np.ndarray) -> np.ndarray:
        times = np.asarray(t)
        x = np.where(times <= 0.2, times / 0.2, (3.2 - times) / 3.0)
        result = np.zeros((4, *times.shape))
        result[3] = 0.5 + x**3 / 3 - x**2 / 2 + 0.16 * x
        return result

    driver = BrainExposureDriver(np.array([0.0, 0.2, 3.2]), state, 1000)
    peak_nm, peak_h = driver.peak()
    assert peak_h == pytest.approx(0.2 * 0.2, abs=1e-12)
    assert peak_nm == pytest.approx(driver.unbound_nm(3.2 - 0.2 * 3.0), abs=1e-10)
