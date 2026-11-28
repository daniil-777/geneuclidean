"""Reproducible exposure scenarios for the repository's two nominated DRD1 structures."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from rdkit.Chem import Descriptors

from geneuclidean.assays import aqueous_equilibrium, estimate_aqueous_solubility
from geneuclidean.human_twin.dialysis import Compound, DialysisRun, evaluate_dialysis
from geneuclidean.human_twin.exposure import (
    CompetitiveInhibition,
    Disposition,
    OralMedium,
    simulate_oral,
)
from geneuclidean.human_twin.reference import compare_with_human_data

DEFAULT_CANDIDATE = "DRD1_v53_L1"

# Stable dossier identities are available in installed wheels without downloading campaign outputs.
_DOSSIERS = {
    DEFAULT_CANDIDATE: (
        "CCN1CCN(Cc2cc(F)cc(C#N)c2)CC1",
        "results/tables/ts_scz_realistic_ligands_v53.json",
        "flagship_leads",
    ),
    "DRD1_1": (
        "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1",
        "results/tables/cns_iter3_top_hits_dossier.json",
        "flagship_top12",
    ),
}


def tourette_candidate(compound_id: str = DEFAULT_CANDIDATE) -> Compound:
    """Retrieve the exact free-base structure used in the selected campaign."""
    if compound_id not in _DOSSIERS:
        raise ValueError(f"Unknown candidate {compound_id!r}. Choose from {', '.join(_DOSSIERS)}")
    return Compound(compound_id=compound_id, smiles=_DOSSIERS[compound_id][0])


class AqueousSolubilityMeasurement(BaseModel):
    """Equilibrated free parent in buffered water in contact with the specified solid form."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )
    compound: Compound
    endpoint: Literal["equilibrium solubility in buffered water"]
    solubility_um: float = Field(gt=0)
    ph: float = Field(ge=0, le=14)
    temperature_c: float
    material_lot: str = Field(min_length=1)
    solid_form: str = Field(min_length=1)
    buffer: str = Field(min_length=1)
    raw_data_uri: str = Field(min_length=1)


class D1BindingMeasurement(BaseModel):
    """A lab-supplied parent-specific equilibrium Kd for human DRD1, in nM."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )
    compound: Compound
    target: Literal["human DRD1"]
    endpoint: Literal["equilibrium dissociation constant"]
    kd_nm: float = Field(gt=0)
    material_lot: str = Field(min_length=1)
    assay_conditions: str = Field(min_length=1)
    raw_data_uri: str = Field(min_length=1)


def _summarize_exposure(
    p: Disposition,
    medium: OralMedium,
    times: list[float],
    amount: float,
    binding: D1BindingMeasurement | None,
) -> dict[str, Any]:
    profile = simulate_oral(p, medium, times, amount_umol=amount)
    brain_nm = 1000 * profile.unbound_brain_um
    plasma_nm = 1000 * profile.unbound_plasma_um
    peak_index = int(np.argmax(brain_nm))
    return {
        "medium": asdict(medium),
        "disposition": asdict(p),
        "unbound_plasma_nm": plasma_nm.round(8).tolist(),
        "unbound_brain_nm": brain_nm.round(8).tolist(),
        "sampled_brain_cmax_nm": float(brain_nm[peak_index]),
        "sampled_brain_tmax_h": times[peak_index],
        "unbound_brain_auc_0_48_nm_h": 1000 * float(profile.unbound_brain_auc_um_h[-1]),
        "unbound_plasma_auc_0_48_nm_h": 1000 * float(profile.unbound_plasma_auc_um_h[-1]),
        "inventory_48h_umol": {
            k: float(getattr(profile, k)[-1])
            for k in (
                "gut_umol",
                "central_umol",
                "peripheral_umol",
                "brain_umol",
                "transit_loss_umol",
                "first_pass_loss_umol",
                "systemic_loss_umol",
            )
        },
        "max_mass_error_umol": float(np.max(np.abs(profile.accounted_umol - amount))),
        "equilibrium_d1_occupancy_fraction": None
        if binding is None
        else (brain_nm / (brain_nm + binding.kd_nm)).round(8).tolist(),
    }


def run_tourette_study(
    compound_id: str = DEFAULT_CANDIDATE,
    *,
    dialysis: DialysisRun | None = None,
    solubility: AqueousSolubilityMeasurement | None = None,
    binding: D1BindingMeasurement | None = None,
) -> dict[str, Any]:
    """Run the 2×2×2 sensitivity design, optionally using matched assay evidence.

    A ninth scenario uses structure-specific solubility and plasma-binding estimates.
    A QC-passing measured unbound fraction supersedes the estimated or assumed value.
    The same normalized one-µmol input is used in every case.
    """
    compound = tourette_candidate(compound_id)
    fu = 0.20
    binding_evidence = None
    if dialysis is not None:
        if dialysis.plan.candidate != compound:
            raise ValueError("Dialysis run is for a different compound ID or structure")
        if dialysis.ph != 7.4 or dialysis.temperature_c != 37.0:
            raise ValueError("This study uses a nominal dialysis protocol of pH 7.4 and 37 °C")
        measured = evaluate_dialysis(dialysis)
        if measured.fraction_unbound_plasma is None:
            raise ValueError("Dialysis QC failed: " + " | ".join(measured.reasons))
        fu = measured.fraction_unbound_plasma
        binding_evidence = {
            "result": measured.model_dump(mode="json"),
            "run": dialysis.model_dump(mode="json"),
        }
    if solubility is not None:
        if solubility.compound != compound:
            raise ValueError("Solubility measurement is for a different compound")
        if solubility.ph != 6.8 or solubility.temperature_c != 37:
            raise ValueError("The buffered lumen scenario uses nominal pH 6.8 and 37 °C")
    if binding is not None and binding.compound != compound:
        raise ValueError("D1 equilibrium binding measurement is for a different compound")

    # These values define counterfactual conditions. No structure score determines a rate.
    base = Disposition(
        absorption_per_h=1.0,
        gut_transit_per_h=0.2,
        gut_escape_fraction=1,
        central_volume_l=5,
        peripheral_volume_l=30,
        peripheral_exchange_l_h=5,
        brain_unbound_volume_l=10,
        bbb_in_l_h=2,
        bbb_out_l_h=2,
        fraction_unbound_plasma=fu,
        blood_to_plasma_ratio=1,
        liver_blood_flow_l_h=90,
        liver_unbound_clint_l_h=10,
        other_plasma_clearance_l_h=1,
    )
    times = np.linspace(0, 48, 97).tolist()
    amount = 1.0
    scenarios = []
    for cap, out_cl, clint in product((0.5, 100.0), (2.0, 10.0), (10.0, 80.0)):
        p = replace(base, bbb_out_l_h=out_cl, liver_unbound_clint_l_h=clint)
        medium = OralMedium(ph=6.8, volume_l=0.25, free_solubility_um=cap)
        scenarios.append(_summarize_exposure(p, medium, times, amount, binding))

    weight = float(Descriptors.MolWt(Chem.MolFromSmiles(compound.smiles)))  # type: ignore[attr-defined]
    assay_data = json.loads(
        files(__package__).joinpath("assay_estimates.json").read_text(encoding="utf-8")
    )
    assays = {}
    for endpoint, calibration in assay_data["endpoints"].items():
        estimate = calibration["predictions"][compound_id]
        if estimate["smiles"] != compound.smiles or not estimate["valid"]:
            raise ValueError(f"Packaged assay estimate does not match {compound_id}")
        assays[endpoint] = {
            **{k: v for k, v in calibration.items() if k != "predictions"},
            "estimate": estimate,
        }
    # µg/mL equals mg/L. Multiply by 1000/MW for µmol/L. The plasma endpoint is
    # log10(percent unbound), so converting to a fraction also requires division by 100.
    sol_est = assays["Biogen_log_Solubility_pH68"]["estimate"]
    fu_est = assays["Biogen_log_Human_Unbound_Percent"]["estimate"]
    cap_um = solubility.solubility_um if solubility else 10 ** sol_est["y_pred"] * 1000 / weight
    estimated_fu = 10 ** fu_est["y_pred"] / 100
    informed_pk = replace(base, fraction_unbound_plasma=fu if dialysis else estimated_fu)
    informed_medium = OralMedium(ph=6.8, volume_l=0.25, free_solubility_um=cap_um)
    informed = _summarize_exposure(informed_pk, informed_medium, times, amount, binding)
    # Vary only the two assay predictions. Marginal assay intervals do not imply joint PK coverage.
    bounds = ("interval_lo", "interval_hi")
    sol_values = [cap_um] if solubility else [10 ** sol_est[k] * 1000 / weight for k in bounds]
    fu_values = [fu] if dialysis else [min(1.0, 10 ** fu_est[k] / 100) for k in bounds]
    corners = []
    for sol_value, fu_value in product(sol_values, fu_values):
        curve = simulate_oral(
            replace(informed_pk, fraction_unbound_plasma=fu_value),
            replace(informed_medium, free_solubility_um=sol_value),
            times,
            amount_umol=amount,
        )
        corners.append(
            {
                "free_solubility_um": sol_value,
                "fraction_unbound_plasma": fu_value,
                "sampled_brain_cmax_nm": float(curve.unbound_brain_um.max()) * 1000,
                "unbound_brain_auc_0_48_nm_h": float(curve.unbound_brain_auc_um_h[-1]) * 1000,
            }
        )
    inhibitor = CompetitiveInhibition(
        affected_clint_fraction=0.5, unbound_inhibitor_um=1, unbound_ki_um=1
    )
    coexposure = simulate_oral(
        informed_pk, informed_medium, [48.0], amount_umol=amount, inhibition=inhibitor
    )
    inhibited_auc = float(coexposure.unbound_plasma_auc_um_h[-1]) * 1000

    # A separate finite vial illustrates reversible complexation with a hypothetical
    # soluble partner. No measured pKa or intrinsic solubility is assumed for a piperazine.
    mixtures = []
    for partner in (0.0, 8.0):
        mixture = aqueous_equilibrium(
            total_um=4.0, intrinsic_solubility_um=0.5, partner_total_um=partner, kd_um=0.25, ph=6.8
        )
        mixtures.append(
            {
                "fixed_ph_apparent_cap_um": 0.5,
                "total_um": 4,
                "ph": 6.8,
                "partner_total_um": partner,
                "assumed_partner_kd_um": 0.25,
                "free_um": mixture.free_um,
                "bound_um": mixture.bound_um,
                "solid_um": mixture.solid_um,
                "dissolved_um": mixture.dissolved_um,
            }
        )
    estimate = estimate_aqueous_solubility(compound.smiles)
    _, dossier, key = _DOSSIERS[compound_id]
    return {
        "schema_version": 1,
        "compound": compound.model_dump(),
        "target": "human DRD1",
        "dossier": {"path": dossier, "collection": key},
        "material_status": "computational nomination",
        "calculated_molecular_weight_g_mol": round(weight, 3),
        "esol_log10_mol_l": estimate.log_s_mol_l,
        "esol_context": "structure estimate, pH unspecified",
        "input_amount_umol": amount,
        "time_h": times,
        "scenario_design": "deterministic factorial sensitivity",
        "parameter_basis": {
            "medium_and_disposition": "assumed except for any identified assay below",
            "fraction_unbound_plasma": "assumed" if dialysis is None else "measured in vitro",
        },
        "solubility_evidence": solubility.model_dump(mode="json") if solubility else None,
        "plasma_binding_evidence": binding_evidence,
        "d1_binding_evidence": None if binding is None else binding.model_dump(mode="json"),
        "clinical_endpoints": {
            "tic_response": None,
            "human_safety": None,
            "reason": "No validated exposure-to-clinical-outcome model for this lead",
        },
        "aqueous_binding_scenarios": mixtures,
        "scenarios": scenarios,
        "structure_informed_scenario": {
            "model": "Fang assay predictions with assumed human disposition",
            "fraction_unbound_plasma_basis": "measured in vitro"
            if dialysis
            else "structure estimate",
            "cap_basis": "measured buffer solubility transferred to lumen"
            if solubility
            else "assumed transfer of predicted pH 6.8 assay solubility to buffered lumen",
            "assay_sensitivity": {
                "design": "marginal assay interval corners at fixed disposition",
                "fraction_bounds_intersected_with_unit_interval": True,
                "corners": corners,
            },
            "assay_source": assay_data["source"],
            "assay_table_sha256": assay_data["sha256"],
            "assay_estimates": assays,
            "exposure": informed,
        },
        "hepatic_competition_scenario": {
            "evidence": "hypothetical constant inhibitor and assumed pathway contribution",
            "inhibition": asdict(inhibitor),
            "remaining_clint_fraction": inhibitor.remaining_clint_fraction,
            "unbound_plasma_auc_0_48_nm_h": inhibited_auc,
            "plasma_auc_ratio_to_structure_informed_scenario": inhibited_auc
            / informed["unbound_plasma_auc_0_48_nm_h"],
        },
        "human_reference_checks": compare_with_human_data(),
    }


def plot_study(report: dict[str, Any], path: Path) -> None:
    """Draw the same normalized input under each solubility, transport and clearance assumption."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    figure = Figure(figsize=(12, 5.6), layout="constrained")
    axes = figure.subplots(1, 3, sharex=True)
    axes[1].sharey(axes[0])
    colors = {2.0: "#00736f", 10.0: "#aa4284"}
    styles = {10.0: "-", 80.0: "--"}
    for ax, solubility in zip(axes[:2], (0.5, 100.0), strict=True):
        for scenario in report["scenarios"]:
            if scenario["medium"]["free_solubility_um"] == solubility:
                p = scenario["disposition"]
                ax.plot(
                    report["time_h"],
                    scenario["unbound_brain_nm"],
                    lw=2.1,
                    color=colors[p["bbb_out_l_h"]],
                    ls=styles[p["liver_unbound_clint_l_h"]],
                )
        ax.set_title(f"Solubility cap {solubility:g} µM")
        ax.set_xlabel("Time after model input (h)")
        ax.set_xlim(0, 48)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    peak = max(s["sampled_brain_cmax_nm"] for s in report["scenarios"])
    axes[0].set_ylim(0, 1.08 * peak if peak > 0 else 1)
    curve = report["structure_informed_scenario"]["exposure"]
    for key, label, color in (
        ("unbound_plasma_nm", "Plasma", "#375b9a"),
        ("unbound_brain_nm", "Brain ISF", "#aa650a"),
    ):
        axes[2].plot(report["time_h"], curve[key], label=label, color=color, lw=2.1)
    axes[2].set(
        title="Assay-informed scenario",
        xlabel="Time after model input (h)",
        ylabel="Unbound concentration (nM)",
        ylim=(0, None),
    )
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False)
    axes[0].set_ylabel("Unbound brain interstitial concentration (nM)")
    figure.suptitle(
        f"{report['compound']['compound_id']}  |  1 µmol oral scenario input", fontsize=15
    )
    handles = [
        Line2D([], [], color=color, lw=2, label=f"BBB out {cl:g} L/h")
        for cl, color in colors.items()
    ]
    handles += [
        Line2D([], [], color="#313c43", lw=2, ls=style, label=f"Unbound liver CLint {cl:g} L/h")
        for cl, style in styles.items()
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        title="Assumed model inputs",
    )
    figure.savefig(path, dpi=155)
