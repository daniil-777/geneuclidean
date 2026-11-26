#!/usr/bin/env python3
"""Phase 3 & 4 Real Benchmark: Makeable Synthesis, Avoid-ome MMP, OpenFE & OpenADMET Conformal DMPK.

Runs three real end-to-end industrial benchmarks and writes
``results/tables/phase3_phase4_leadopt_dmpk_benchmark.json``:

1. **Reaction-Grounded Makeable Enumeration & Avoid-ome MMP Rescue:**
   - Synthesizes 1-step makeable piperazide/morpholine analogs around quinoline-2-acid (`DRD1`),
   - Runs `AvoidomeMMPEngine` (with real 12-endpoint `v4_admet_avoidome_oracles.joblib`) on
     high-hERG **Donepezil** (`ACHE`) and **(-)-Ibogaine** (`SERT`) to propose minimal 1-cut
     bioisostere swaps quantifying exact `Delta P(hERG)`, `Delta P(MDR1)`, `Delta P(5-HT2B)`.
2. **OpenFE RBFE Perturbation Network Export:**
   - Exports a charge-matched (`Delta q == 0`), ring-strain-clean, MCS-aligned congeneric
     `DRD1` piperazide series directly into a ready-to-run OpenFE RBFE perturbation graph.
3. **OpenADMET-ExpansionRx & Biogen Continuous DMPK Regression (5-Fold Scaffold CV):**
   - Evaluates `ConformalDMPKRegressor` on real OpenADMET-ExpansionRx (`LogD`, `KSOL`, `HLM CLint`)
     and Biogen (`LOG SOLUBILITY PH 6.8`, `LOG HLM_CLint`, `LOG MDR1-MDCK ER`) reporting
     `Spearman rho`, `R2`, `MAE`, `RMSE`, and empirical 90% Split-Conformal coverage.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from geneuclidean.cns.admet import CNSAdmetSuite
from geneuclidean.cns.dmpk_regression import evaluate_scaffold_cv_conformal_dmpk
from geneuclidean.cns.scaffold_hop import IBOGAINE_SMILES
from geneuclidean.interop import export_openfe_rbfe_network
from geneuclidean.leadopt import AvoidomeMMPEngine
from geneuclidean.synthesis import enumerate_makeable_analogs

OUT_JSON = Path("results/tables/phase3_phase4_leadopt_dmpk_benchmark.json")
V4_ADMET_PATH = Path("data/cns_neuro.nosync/v4_admet_avoidome_oracles.joblib")
OPENADMET_TRAIN = Path("data/cns_neuro.nosync/v4_expansion/admet/openadmet_expansionrx_train.csv")
BIOGEN_CSV = Path("data/cns_neuro.nosync/v4_expansion/admet/ADME_public_set_3521.csv")
DONEPEZIL_SMILES = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"


def _load_continuous_endpoint(
    csv_path: Path,
    smiles_col: str,
    target_col: str,
    *,
    log10_transform: bool = False,
    max_n: int = 1200,
) -> tuple[list[str], list[float]]:
    smis: list[str] = []
    vals: list[float] = []
    if not csv_path.is_file():
        return smis, vals
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            s = (row.get(smiles_col) or "").strip()
            raw = (row.get(target_col) or "").strip()
            if not s or not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            if log10_transform:
                if v <= 0.0:
                    continue
                v = math.log10(v)
            smis.append(s)
            vals.append(v)
            if len(smis) >= max_n:
                break
    return smis, vals


def main() -> None:
    t0 = time.perf_counter()
    v4_admet = CNSAdmetSuite.try_load(V4_ADMET_PATH)

    print("=== [1/3] Reaction-Grounded Synthesis & Avoid-ome 1-Cut MMP Rescues ===")
    core_acid = "O=C(O)c1ccc2ccccc2n1"
    amine_bbs = [
        "CN1CCNCC1",
        "CCN1CCNCC1",
        "C1COCCN1",
        "C1CCNCC1",
        "Fc1ccc(CN2CCNCC2)cc1",
        "CC1CN(C)CCN1",
        "NCC2CC2",
        "OC1CCNCC1",
    ]
    makeable = enumerate_makeable_analogs(
        core_acid, amine_bbs, reactions=("amide_coupling_acid_amine",), triage_filter=True
    )
    print(
        f"  Enumerated {len(makeable)} clean 1-step makeable amide analogs from quinoline-2-acid."
    )

    mmp_engine = AvoidomeMMPEngine(admet_suite=v4_admet, property_profile="cns_mpo_v2")
    donepezil_rescues = mmp_engine.propose_rescues(DONEPEZIL_SMILES, max_proposals=10)
    ibogaine_rescues = mmp_engine.propose_rescues(IBOGAINE_SMILES, max_proposals=10)
    if donepezil_rescues:
        top_d = donepezil_rescues[0]
        print(
            f"  Donepezil top 1-cut MMP rescue ({top_d.transform_name}): "
            f"hERG={top_d.herg_prob} (d={top_d.delta_herg_reduction}), MPO={top_d.cns_mpo_v2:.2f}"
        )
    if ibogaine_rescues:
        top_i = ibogaine_rescues[0]
        print(
            f"  Ibogaine top 1-cut MMP rescue ({top_i.transform_name}): "
            f"hERG={top_i.herg_prob} (d={top_i.delta_herg_reduction}), MPO={top_i.cns_mpo_v2:.2f}"
        )

    print("=== [2/3] Exporting Charge-Matched, Ring-Strain-Clean OpenFE RBFE Network ===")
    drd1_series = [(f"DRD1_Makeable_{i + 1}", c.product_smiles) for i, c in enumerate(makeable[:6])]
    openfe_bundle = export_openfe_rbfe_network(
        drd1_series,
        pocket_pdb="data/cns_neuro.nosync/structures/clean_pockets/9LLG_DRD1_A1EKL_pocket.pdb",
        out_dir="results/openfe_drd1_demo",
        require_same_charge=True,
        filter_ring_strain=True,
        max_ring_strain_kcal=3.0,
        seed=42,
    )
    n_skip = len(openfe_bundle.skipped_ligands)
    print(
        f"  Exported OpenFE RBFE network: {openfe_bundle.n_ligands} ligands, "
        f"{openfe_bundle.n_edges} charge-matched edges (skipped={n_skip})"
    )

    print("=== [3/3] OpenADMET-ExpansionRx & Biogen 5-Fold Scaffold-CV Conformal DMPK ===")
    dmpk_specs = [
        ("OpenADMET_LogD", OPENADMET_TRAIN, "SMILES", "LogD", False),
        ("OpenADMET_log10_KSOL", OPENADMET_TRAIN, "SMILES", "KSOL", True),
        ("OpenADMET_log10_HLM_CLint", OPENADMET_TRAIN, "SMILES", "HLM CLint", True),
        ("Biogen_log_HLM_CLint", BIOGEN_CSV, "SMILES", "LOG HLM_CLint (mL/min/kg)", False),
        (
            "Biogen_log_Solubility_pH68",
            BIOGEN_CSV,
            "SMILES",
            "LOG SOLUBILITY PH 6.8 (ug/mL)",
            False,
        ),
    ]
    dmpk_results: list[dict[str, Any]] = []
    for ep_name, path, scol, tcol, do_log in dmpk_specs:
        smis, vals = _load_continuous_endpoint(path, scol, tcol, log10_transform=do_log, max_n=1000)
        if len(smis) < 40:
            continue
        res = evaluate_scaffold_cv_conformal_dmpk(
            smis, vals, endpoint=ep_name, n_splits=5, alpha=0.10, n_estimators=60, seed=42
        )
        dmpk_results.append(res)
        print(
            f"  {ep_name:28s} (n={res['n_valid']:4d}): rho={res['spearman_rho']:.3f}, "
            f"R2={res['r2']:.3f}, MAE={res['mae']:.3f}, 90%-Cov={res['empirical_coverage']:.1%}"
        )

    payload = {
        "generated_by": "scripts/run_makeable_leadopt_and_openfe_demo.py",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "makeable_drd1_analogs": [c.to_dict() for c in makeable],
        "donepezil_mmp_rescues": [r.to_dict() for r in donepezil_rescues[:5]],
        "ibogaine_mmp_rescues": [r.to_dict() for r in ibogaine_rescues[:5]],
        "openfe_drd1_rbfe_network": openfe_bundle.to_dict(),
        "conformal_dmpk_scaffold_cv": dmpk_results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Phase 3 & 4 benchmark -> {OUT_JSON} ({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
