"""Human IV reference data, participant holdout and compartment-equation equivalence."""

import json

import numpy as np
import pytest
from scipy.linalg import expm

from geneuclidean.human_twin import _indomethacin, compare_with_human_data
from geneuclidean.human_twin.reference import _indomethacin_iv_profiles

KEY = "indomethacin_iv_held_out_profiles"


def test_whole_participant_validation_and_pooled_metrics() -> None:
    data = compare_with_human_data()[KEY]
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["n_subjects"] == 6 and data["n_observations"] == 66
    assert data["source"] == "https://doi.org/10.1007/BF01063617"
    assert data["data_license"] == "MIT" and "medicaldata authors" in data["data_license_notice"]
    assert [f["held_out_subject"] for f in data["folds"]] == [1, 2, 3, 4, 5, 6]
    for fold in data["folds"]:
        assert set(fold["training_subjects"]) == set(range(1, 7)) - {fold["held_out_subject"]}
        assert len(fold["observed_mg_l"]) == len(data["time_h"]) == 11

    observed = np.array([f["observed_mg_l"] for f in data["folds"]])
    for name in ("one_compartment", "two_compartment"):
        predicted = np.array([f["models"][name]["predicted_mg_l"] for f in data["folds"]])
        assert predicted.shape == observed.shape and np.all(predicted > 0)
        absolute_error = np.abs(predicted.ravel() - observed.ravel())
        fold_error = np.maximum(predicted / observed, observed / predicted)
        assert data["pooled_held_out_errors"][name] == pytest.approx(
            {
                "rmse_mg_l": np.linalg.norm(absolute_error) / np.sqrt(66),
                "mae_mg_l": absolute_error.sum() / 66,
                "geometric_mean_absolute_fold_error": np.prod(fold_error) ** (1 / 66),
            }
        )
    assert data["pooled_held_out_errors"]["one_compartment"]["rmse_mg_l"] == pytest.approx(
        0.40755, abs=3e-5
    )
    assert data["pooled_held_out_errors"]["two_compartment"]["rmse_mg_l"] == pytest.approx(
        0.20432, abs=3e-5
    )
    assert data["subjects_with_lower_two_compartment_rmse"] == [2, 3, 4, 5, 6]


def test_held_out_measurements_do_not_influence_fitted_curves() -> None:
    original = compare_with_human_data()[KEY]
    changed = np.array(_indomethacin.PLASMA_MG_L)
    changed[-1] *= 3
    updated = _indomethacin_iv_profiles(changed)
    for name in original["pooled_held_out_errors"]:
        before = original["folds"][-1]["models"][name]
        after = updated["folds"][-1]["models"][name]
        assert before["predicted_mg_l"] == after["predicted_mg_l"]
        assert before["errors"]["rmse_mg_l"] != after["errors"]["rmse_mg_l"]
        # Subject 6 participates in every other training set.
        assert (
            original["folds"][0]["models"][name]["predicted_mg_l"]
            != (updated["folds"][0]["models"][name]["predicted_mg_l"])
        )


def test_biexponential_curves_equal_independent_compartment_amount_propagation() -> None:
    data = compare_with_human_data()[KEY]
    for fold in data["folds"]:
        model = fold["models"]["two_compartment"]
        a, b = model["amplitudes_mg_l"]
        fast, slow = model["rates_per_h"]
        assert a > 0 and b > 0 and fast > slow > 0
        # Standard mammillary model with unit central volume and loss only from central.
        c0 = a + b
        k21 = (a * slow + b * fast) / c0
        k10 = fast * slow / k21
        k12 = fast + slow - k21 - k10
        assert min(k10, k12, k21) > 0
        generator = np.array([[-(k10 + k12), k21], [k12, -k21]])
        central = [(expm(generator * t) @ [c0, 0])[0] for t in data["time_h"]]
        np.testing.assert_allclose(central, model["predicted_mg_l"], rtol=1e-11)


def test_reference_report_is_an_independent_copy() -> None:
    first = compare_with_human_data()
    first[KEY]["folds"][0]["observed_mg_l"][0] = -1
    assert compare_with_human_data()[KEY]["folds"][0]["observed_mg_l"][0] == 1.50
