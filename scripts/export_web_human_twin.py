"""Rebuild the studio's input parameters and independent SciPy parity fixtures.

Run from a checkout with the Python library installed. Browser simulations use
only the input parameters. Reference trajectories are shipped to tests alone.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from geneuclidean.human_twin import (
    Disposition,
    GastricStage,
    OralFormulation,
    OralMedium,
    run_tourette_study,
    simulate_oral,
)

ROOT = Path(__file__).resolve().parents[1]
TIMES_H = (0, 0.5, 2, 4, 8, 12, 24, 36, 48)


def export() -> None:
    leads: dict[str, Any] = {}
    references = []
    common: dict[str, Any] = {}
    for lead in ("DRD1_v53_L1", "DRD1_1"):
        study = run_tourette_study(lead)
        experiment = study["formulation_experiment"]
        cases = experiment["cases"]
        pk = Disposition(**experiment["disposition"])
        medium = OralMedium(**experiment["medium"])
        leads[lead] = {
            "smiles": study["compound"]["smiles"],
            "pubchemCid": study["public_record"]["pubchem_cid"],
            "solubilityUm": medium.free_solubility_um,
            "fractionUnboundPlasma": pk.fraction_unbound_plasma,
        }
        gastric = cases["gastric_to_intestine"]["gastric_parameters"]
        if medium.partner_total_um or gastric["medium"]["partner_total_um"]:
            raise ValueError("The browser solver currently supports partner-free buffers")
        params = {
            "disposition": {
                k: v for k, v in experiment["disposition"].items() if k != "fraction_unbound_plasma"
            },
            "intestinalMedium": {
                k: v for k, v in experiment["medium"].items() if k != "free_solubility_um"
            },
            "gastric": gastric,
            "gastricSource": experiment["gastric_source"]["source"],
            "formulations": {name: case["parameters"] for name, case in cases.items()},
            "amountUmol": experiment["input_amount_umol"],
            "durationH": experiment["time_h"][-1],
            "assaySource": study["structure_informed_scenario"]["assay_source"],
        }
        if common and params != common:
            raise ValueError("The shared study defaults differ between candidate structures")
        common = params
        stage = GastricStage(
            **{k: v for k, v in gastric.items() if k not in {"medium", "formulation"}},
            medium=OralMedium(**gastric["medium"]),
            formulation=OralFormulation(**gastric["formulation"]),
        )
        # Include a high-clearance gastric case for both candidates to verify all sliders
        # change the actual dynamics even away from the preset formulations.
        for name, custom in [(name, False) for name in cases] + [("gastric_to_intestine", True)]:
            form = (
                OralFormulation(**cases[name]["parameters"]) if cases[name]["parameters"] else None
            )
            disposition = pk
            if custom:
                assert form is not None
                form = replace(form, dissolution_per_h=3.8)
                disposition = replace(pk, gut_transit_per_h=0.9, liver_unbound_clint_l_h=78)
            profile = simulate_oral(
                disposition,
                medium,
                TIMES_H,
                amount_umol=experiment["input_amount_umol"],
                formulation=form,
                gastric_stage=stage if name == "gastric_to_intestine" else None,
                dense_brain=True,
            )
            assert profile.brain_driver is not None
            assert profile.gut_solid_umol is not None and profile.gut_dissolved_umol is not None
            peak, peak_time = profile.brain_driver.peak()
            references.append(
                {
                    "lead": lead,
                    "formulation": name,
                    "custom": custom,
                    "transitPerH": disposition.gut_transit_per_h,
                    "clearanceLPerH": disposition.liver_unbound_clint_l_h,
                    "dissolutionPerH": form.dissolution_per_h if form else 2.0,
                    "peakBrainNm": round(peak, 8),
                    "timeToPeakH": round(peak_time, 8),
                    "brainAucNmH": round(float(profile.unbound_brain_auc_um_h[-1]) * 1000, 8),
                    "transitLossUmol": round(float(profile.transit_loss_umol[-1]), 10),
                    "brainNm": (profile.unbound_brain_um * 1000).round(8).tolist(),
                    "plasmaNm": (profile.unbound_plasma_um * 1000).round(8).tolist(),
                    "solidUmol": profile.gut_solid_umol.round(10).tolist(),
                    "dissolvedUmol": profile.gut_dissolved_umol.round(10).tolist(),
                }
            )
    for relative, data in (
        ("web/src/lib/humanTwinParameters.json", {**common, "leads": leads}),
        ("web/tests/humanTwinReference.json", {"timesH": TIMES_H, "cases": references}),
    ):
        (ROOT / relative).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    export()
