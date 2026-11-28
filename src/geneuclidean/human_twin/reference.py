"""External human reference checks for two restricted cases of the exposure solver."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from scipy.optimize import brentq

from geneuclidean.human_twin.exposure import Disposition, OralMedium, simulate_oral


def compare_with_human_data() -> dict[str, Any]:
    """Compare an independently calibrated oral baseline and a passive BBB null to observations.

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
