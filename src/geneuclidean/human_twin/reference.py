"""Measured human plasma profiles and reference checks for restricted PK models."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from geneuclidean.human_twin import _indomethacin
from geneuclidean.human_twin.exposure import Disposition, OralMedium, simulate_oral


def _fit_iv_models(
    time_h: NDArray[np.float64], training_mg_l: NDArray[np.float64]
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Fit a common post-bolus plasma curve to all training participants."""
    # Sampling times are identical. Averaging log measurements at each time preserves the
    # minimizer of the sum of squared log residuals across all training observations.
    log_conc = np.log(training_mg_l).mean(axis=0)
    design = np.column_stack((np.ones_like(time_h), -time_h))
    intercept, rate = np.linalg.lstsq(design, log_conc, rcond=None)[0]

    def two_compartment_residuals(log_parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        log_a, log_b, log_rate_gap, log_slow = log_parameters
        slow = np.exp(log_slow)
        fast = slow + np.exp(log_rate_gap)
        log_predicted = np.logaddexp(log_a - fast * time_h, log_b - slow * time_h)
        return np.asarray(log_predicted - log_conc, dtype=np.float64)

    # Fixed initialization independent of held-out data. Positive amplitudes and ordered,
    # positive rates rule out exchanging the fast and slow terms during optimization.
    fit = least_squares(
        two_compartment_residuals, np.log([2.0, 0.25, 1.5, 0.2]), ftol=1e-12, xtol=1e-12, gtol=1e-12
    )
    if not fit.success:
        raise RuntimeError(f"Indomethacin two-compartment fit failed: {fit.message}")
    a, b, rate_gap, slow = np.exp(fit.x)
    return {
        "one_compartment": (np.array([np.exp(intercept)]), np.array([rate])),
        "two_compartment": (np.array([a, b]), np.array([slow + rate_gap, slow])),
    }


def _concentration_errors(
    predicted: NDArray[np.float64], observed: NDArray[np.float64]
) -> dict[str, float]:
    residual = predicted - observed
    return {
        "mae_mg_l": float(np.abs(residual).mean()),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(residual)))),
        "geometric_mean_absolute_fold_error": float(
            np.exp(np.mean(np.abs(np.log(predicted / observed))))
        ),
    }


def _indomethacin_iv_profiles(observed: NDArray[np.float64]) -> dict[str, Any]:
    """Predict all samples of one participant using only the five other profiles."""
    time_h = np.asarray(_indomethacin.TIME_H)
    subjects = np.asarray(_indomethacin.SUBJECTS)
    predictions = {
        "one_compartment": np.empty_like(observed),
        "two_compartment": np.empty_like(observed),
    }
    folds = []
    for index, subject in enumerate(subjects):
        train = subjects != subject
        models = {}
        for name, (amplitudes, rates) in _fit_iv_models(time_h, observed[train]).items():
            predicted = amplitudes @ np.exp(-np.outer(rates, time_h))
            predictions[name][index] = predicted
            models[name] = {
                "amplitudes_mg_l": amplitudes.tolist(),
                "rates_per_h": rates.tolist(),
                "predicted_mg_l": predicted.tolist(),
                "errors": _concentration_errors(predicted, observed[index]),
            }
        folds.append(
            {
                "held_out_subject": int(subject),
                "training_subjects": subjects[train].tolist(),
                "observed_mg_l": observed[index].tolist(),
                "models": models,
            }
        )
    return {
        "source": _indomethacin.SOURCE,
        "data_source": _indomethacin.DATA_SOURCE,
        "data_source_sha256": _indomethacin.DATA_SHA256,
        "data_license": "MIT",
        "data_license_notice": _indomethacin.DATA_LICENSE_NOTICE,
        "cohort": "Six healthy volunteers in one intravenous indomethacin study",
        "endpoint": "Total plasma indomethacin concentration (mg/L)",
        "validation": "Leave-one-subject-out, with one shared curve per training set",
        "fit_objective": "Unweighted sum of squared natural-log concentration residuals",
        "model_assumptions": (
            "Instantaneous IV input and first-order elimination from the central compartment. "
            "The two-compartment model adds reversible, first-order distribution."
        ),
        "model_equations": {
            "one_compartment": "C(t) = A exp(-k t)",
            "two_compartment": "C(t) = A exp(-alpha t) + B exp(-beta t), alpha > beta > 0",
        },
        "scope": (
            "Same-study prediction over the sampled 0.25 to 8 hours. The two-compartment "
            "fit summarizes early and late decline. Enterohepatic recycling described by the "
            "original investigators is outside these equations. Prediction in another cohort, "
            "brain exposure and clinical response require independent evidence."
        ),
        "n_subjects": int(subjects.size),
        "n_observations": int(observed.size),
        "time_h": time_h.tolist(),
        "subjects_with_lower_two_compartment_rmse": subjects[
            np.square(predictions["two_compartment"] - observed).mean(axis=1)
            < np.square(predictions["one_compartment"] - observed).mean(axis=1)
        ].tolist(),
        "pooled_held_out_errors": {
            name: _concentration_errors(values, observed) for name, values in predictions.items()
        },
        "folds": folds,
    }


def compare_with_human_data() -> dict[str, Any]:
    """Return independent report copies. All fixed clinical comparisons run once per process."""
    return deepcopy(_human_reference_checks())


@cache
def _human_reference_checks() -> dict[str, Any]:
    """Compare the oral and passive brain models and cross-validate IV plasma curves.

    The ecopipam studies used the same 179.2 mg free-base equivalent. A per-unit
    concentration times the study mass yields mg/L, independent of molecular weight.
    Apparent V/F and CL/F are fitted parameters confined to the reference baseline.
    """
    study_mass_mg = 179.2
    auc_ng_h_ml, tmax_h, half_life_h = 736.0, 1.51, 15.8
    kel = math.log(2) / half_life_h
    ka = brentq(lambda k: math.log(k / kel) / (k - kel) - tmax_h, kel * 1.001, 100)
    cl_over_f = study_mass_mg / (auc_ng_h_ml / 1000)
    model = Disposition(
        absorption_per_h=ka,
        gut_transit_per_h=0,
        gut_escape_fraction=1,
        central_volume_l=cl_over_f / kel,
        peripheral_volume_l=1,
        peripheral_exchange_l_h=0,
        brain_unbound_volume_l=1,
        bbb_in_l_h=0,
        bbb_out_l_h=0,
        fraction_unbound_plasma=1,
        blood_to_plasma_ratio=1,
        liver_blood_flow_l_h=90,
        liver_unbound_clint_l_h=0,
        other_plasma_clearance_l_h=cl_over_f,
    )
    medium = OralMedium(ph=7, volume_l=1, free_solubility_um=10)
    # In this apparent one-compartment limit the analytic peak time is known and fu is unity.
    profile = simulate_oral(model, medium, [tmax_h, 600], amount_umol=1)
    unit_conversion = study_mass_mg * 1000
    modeled_peak = float(profile.total_plasma_um[0]) * unit_conversion
    modeled_auc = float(profile.unbound_plasma_auc_um_h[-1]) * unit_conversion

    # Full-course AUC ratio equals the steady-state interval ratio in this linear model.
    passive = replace(
        model,
        central_volume_l=10,
        fraction_unbound_plasma=0.5,
        bbb_in_l_h=4,
        bbb_out_l_h=4,
        other_plasma_clearance_l_h=10,
    )
    paired = simulate_oral(passive, medium, [240], amount_umol=1)
    passive_ratio = float(paired.unbound_brain_auc_um_h[-1] / paired.unbound_plasma_auc_um_h[-1])
    return {
        "indomethacin_iv_held_out_profiles": _indomethacin_iv_profiles(
            np.asarray(_indomethacin.PLASMA_MG_L)
        ),
        "ecopipam_parent_total_plasma": {
            "model": "one-compartment oral limit with apparent parameters",
            "fit": {
                "source": "https://doi.org/10.1111/cts.70495",
                "cohort": "A, day 1",
                "n": 18,
                "auc_inf_ng_h_ml": auc_ng_h_ml,
                "terminal_half_life_h": half_life_h,
                "median_tmax_h": tmax_h,
                "cl_over_f_l_h": cl_over_f,
                "ka_per_h": ka,
            },
            "external": {
                "source": "https://doi.org/10.1111/cts.70587",
                "n": 8,
                "cmax_ng_ml": 59.4,
                "auc_inf_ng_h_ml": 725.0,
                "terminal_half_life_h": 17.3,
                "median_tmax_h": 2.0,
            },
            "modeled_cmax_ng_ml": modeled_peak,
            "modeled_auc_0_600_ng_h_ml": modeled_auc,
            "relative_peak_error": modeled_peak / 59.4 - 1,
            "relative_auc_error": modeled_auc / 725 - 1,
        },
        "linezolid_unbound_auc_ratio": {
            "source": "https://doi.org/10.1093/jac/dkae025",
            "cohort": "neurocritical care, paired samples at steady state",
            "n_paired": 4,
            "observed_median_brain_isf_to_plasma": 0.50,
            "observed_range": [0.25, 0.81],
            "observed_median_ventricular_csf_to_plasma": 0.92,
            "passive_model_brain_to_plasma": passive_ratio,
            "relative_error_against_brain_isf": passive_ratio / 0.50 - 1,
        },
    }
