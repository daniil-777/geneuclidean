"""Physical limits and measured reference cases for the compound assay API."""

from __future__ import annotations

import math

import pytest

import geneuclidean as ge


def test_esol_refit_and_structure_domain() -> None:
    ethanol = ge.assays.estimate_aqueous_solubility("OCC")
    assert ethanol.canonical_smiles == "CCO"
    # Independent substitution of ethanol's descriptors (MW 46.069, Crippen logP -0.0014).
    assert ethanol.log_s_mol_l == pytest.approx(-0.0423, abs=0.001)
    for smi in ("", "invalid", "*CC", "CC.O", "[Na+]", "C[NH3+]", "[CH3]"):
        with pytest.raises(ValueError):
            ge.assays.estimate_aqueous_solubility(smi)


@pytest.mark.parametrize(("mode", "ph"), [("acid", 6.0), ("base", 4.0)])
def test_monoprotic_solubility(mode, ph) -> None:
    result = ge.assays.aqueous_equilibrium(
        total_um=250, intrinsic_solubility_um=10, ionization=mode, pka=5, ph=ph
    )
    assert (result.free_solubility_um, result.free_um, result.solid_um) == (110, 110, 140)


@pytest.mark.parametrize("total", [0, 1e-10, 0.5, 1, 1.5, 2, 1e5])
@pytest.mark.parametrize("kd", [1e-8, 1, 1e8])
def test_drug_and_partner_mass_balances(total: float, kd: float) -> None:
    res = ge.assays.aqueous_equilibrium(
        total_um=total, intrinsic_solubility_um=1, partner_total_um=1, kd_um=kd
    )
    assert res.dissolved_um + res.solid_um == pytest.approx(total, rel=2e-12, abs=1e-19)
    assert res.partner_free_um + res.bound_um == pytest.approx(1)
    assert res.free_um * res.partner_free_um == pytest.approx(
        kd * res.bound_um, rel=2e-12, abs=1e-19
    )
    assert 0 <= res.free_um <= res.free_solubility_um
    assert res.solid_um == 0 or res.free_um == res.free_solubility_um


def test_one_site_analytic_and_published_phase_solubility() -> None:
    eq = ge.assays.aqueous_equilibrium
    one = eq(total_um=1, intrinsic_solubility_um=1, partner_total_um=1, kd_um=1)
    assert one.bound_um == pytest.approx((3 - math.sqrt(5)) / 2)
    saturated = eq(total_um=4, intrinsic_solubility_um=1, partner_total_um=1, kd_um=1)
    assert (saturated.free_um, saturated.bound_um, saturated.solid_um) == (1, 0.5, 2.5)
    # Tzanova et al. (2023), Tables 2–3. Same-study fitted Ka, Kd expressed in µM.
    hc = eq(total_um=20000, intrinsic_solubility_um=940, partner_total_um=10000, kd_um=1e6 / 1438)
    assert hc.dissolved_um == pytest.approx(6688, abs=2)
    assert abs(hc.dissolved_um - 6270) / 6270 < 0.07


@pytest.mark.parametrize(
    "bad",
    [
        {"total_um": math.nan},
        {"total_um": -1},
        {"ph": math.nan},
        {"pka": 4},
        {"ionization": "acid"},
        {"ionization": "ampholyte"},
        {"intrinsic_solubility_um": 0},
        {"partner_total_um": 1},
        {"kd_um": math.inf},
        {"kd_um": 0},
    ],
)
def test_invalid_assay_conditions(bad) -> None:
    with pytest.raises(ValueError):
        ge.assays.aqueous_equilibrium(**({"total_um": 10, "intrinsic_solubility_um": 1} | bad))


def test_depletion_regression_and_hepatic_limits() -> None:
    fit = ge.assays.fit_microsomal_depletion
    trace = fit([20, 0, 10, 10], [2.5, 10, 5, 5], protein_mg_per_ml=0.5)
    assert trace.half_life_min == pytest.approx(10)
    assert trace.apparent_clint_ul_min_mg == pytest.approx(math.log(2) * 200)
    assert trace.r_squared == pytest.approx(1)
    for concentrations in ([5, 5, 5], [5, 6, 7]):
        flat = fit([0, 10, 20], concentrations, protein_mg_per_ml=1)
        assert flat.half_life_min is flat.apparent_clint_ul_min_mg is None
    for times, conc in [([0, 1], [2, 1]), ([0, 1, 2], [2, 1, 0]), ([0, 1, 2], [1, 2])]:
        with pytest.raises(ValueError):
            fit(times, conc, protein_mg_per_ml=1)
    cl = ge.assays.well_stirred_hepatic_clearance
    assert cl(blood_flow_ml_min=100, fraction_unbound_blood=0.1, unbound_clint_ml_min=1000) == 50
    assert cl(blood_flow_ml_min=100, fraction_unbound_blood=0, unbound_clint_ml_min=1000) == 0
    assert cl(
        blood_flow_ml_min=100, fraction_unbound_blood=1, unbound_clint_ml_min=1e15
    ) == pytest.approx(100)
