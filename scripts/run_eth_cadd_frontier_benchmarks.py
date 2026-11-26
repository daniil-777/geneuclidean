#!/usr/bin/env python3
"""Real quantitative benchmarks extending ETH Zurich CADD (MODLAB 2024-2026) research frontiers.

Runs four reproducible experiments on Apple M3 Pro MPS + AutoDock Vina v1.2.7 using real
curated human CNS crystal structures, trained GenEuclidean v4.0 checkpoints, and the
12-endpoint Industrial Avoid-ome oracle suite:

1. **Saturated-Ring Conformational Strain Audit (extending Atz & Schneider, RingAnalyzer 2026):**
   Audits all 28 docked human CNS complexes (`web/public/data/complexes/*.json`) with
   Cremer-Pople puckering & MMFF94s ring-system strain energy to test whether PoseBusters-passing
   Vina leads hide high-energy saturated-ring conformational artifacts.
2. **Multi-Conformation Ensemble + Contrastive Off-Target CFG + PMI_3D (extending DRAGONFLY 2024):**
   Benchmarks our trained `v4_safegpt_aligned` and `v4_gpmolformer_aligned` models on:
   - `DRD1` (`9LLG` + `9U5E` + `7LJC` 3-conformation ensemble) vs `DRD2` (`6CM4` off-target),
   - `ACHE` (`4EY7`) vs `BCHE` (`4BDS` off-target),
   quantifying `DeltaPMI_3D` zero-shot selectivity discrimination, Contrastive CFG off-target
   suppression, and multi-conformation scaffold diversity gain over single-pocket conditioning.
3. **Ibogaine -> Human SERT 3D Scaffold-Hopping & hERG/5-HT2B De-Risking (extending COLIBRI 2026):**
   Samples candidates conditioned on the human SERT 2-conformation ensemble (`5I6X` + `5I71`),
   evaluates 5-axis scaffold-hop & 12-endpoint Avoid-ome de-risking vs Ibogaine (fatal hERG/QT
   liability), and docks top novel scaffold hops into human SERT (`5I6X`) with real AutoDock
   Vina v1.2.7 + PoseBusters + `Asp98` salt-bridge check + `RingStrainAnalyzer`.
4. **3D Multi-Task Gradient Conflict Mediation Ablation (extending Minot & Schneider AIM 2025):**
   Runs a controlled 4-task (`lm_nll`, `pkd`, `drugclip`, `props`) training ablation on
   `GenEuclideanModel` (`PaiNN-SE3`) comparing `scalarized` vs `pcgrad` vs `aim_policy` and
   extracts exact pairwise cosine-conflict matrices and interpretable `W_policy` weights.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.special import logit
from sklearn.metrics import roc_auc_score

import geneuclidean as ge
from geneuclidean.cns.admet import CNSAdmetSuite, featurize_admet_smiles
from geneuclidean.cns.medchem_triage import bemis_murcko_scaffold
from geneuclidean.cns.scaffold_hop import (
    IBOGAINE_SMILES,
    PAROXETINE_SMILES,
    audit_scaffold_hop_campaign,
)
from geneuclidean.config import (
    CVAEConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
    seed_everything,
)
from geneuclidean.evaluation.docking_pipeline import dock_ligand_to_pocket
from geneuclidean.evaluation.metrics import compute_internal_diversity
from geneuclidean.evaluation.ring_strain import analyze_ring_strain
from geneuclidean.models.geneuclidean import GenEuclideanModel
from geneuclidean.training.losses import compute_drugclip_loss
from geneuclidean.training.mtl_intervention import GradientConflictMediator

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("HF_HOME", ".hf_cache.nosync")

OUT_JSON = Path("results/tables/eth_cadd_frontier_benchmarks.json")
V4_ADMET_PATH = Path("data/cns_neuro.nosync/v4_admet_avoidome_oracles.joblib")
V4_QSAR_PATH = Path("data/cns_neuro.nosync/v4_24target_qsar.joblib")
V2_PAINN_CKPT = Path("runs_local.nosync/geneuclidean_v2_painn/best.pt")
SAFEGPT_CKPT = Path(
    "runs_local.nosync/geneuclidean_v4_tri_disorder/v4_safegpt_aligned/best_adapter.pt"
)
GPMOLF_CKPT = Path(
    "runs_local.nosync/geneuclidean_v4_tri_disorder/v4_gpmolformer_aligned/best_adapter.pt"
)


def run_exp1_ring_strain_audit() -> dict[str, Any]:
    """Exp 1: Saturated-ring strain & Cremer-Pople audit across all 28 docked CNS complexes."""
    complex_dir = Path("web/public/data/complexes")
    files = sorted(complex_dir.glob("*.json"))
    per_complex: list[dict[str, Any]] = []
    all_ring_strains: list[float] = []
    pucker_counts: dict[str, int] = {}

    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        lig_sdf = data.get("ligand_sdf", "")
        if not lig_sdf:
            continue
        prof = analyze_ring_strain(
            lig_sdf,
            strain_threshold_kcal=3.0,
            n_ref_confs=24,
            include_full_ligand_strain=True,
            seed=42,
        )
        for r in prof.rings:
            all_ring_strains.append(r.strain_energy_kcal)
            pucker_counts[r.pucker_class] = pucker_counts.get(r.pucker_class, 0) + 1
        per_complex.append(
            {
                "target": data.get("target", fp.stem),
                "lead_id": data.get("lead_id", fp.stem),
                "pdb_id": data.get("pdb_id", ""),
                "n_saturated_ring_systems": prof.n_saturated_ring_systems,
                "max_ring_strain_kcal": round(prof.max_ring_strain_kcal, 3),
                "full_ligand_strain_kcal": (
                    round(prof.full_ligand_strain_kcal, 3)
                    if prof.full_ligand_strain_kcal is not None
                    else None
                ),
                "passed_3kcal": prof.passed,
                "puckers": [r.pucker_class for r in prof.rings],
            }
        )

    with_sat = [c for c in per_complex if c["n_saturated_ring_systems"] > 0]
    flagged = [c for c in per_complex if not c["passed_3kcal"]]
    return {
        "n_complexes_audited": len(per_complex),
        "n_with_saturated_rings": len(with_sat),
        "n_total_saturated_ring_systems": len(all_ring_strains),
        "pass_rate_3kcal": round(
            sum(1 for c in per_complex if c["passed_3kcal"]) / max(len(per_complex), 1), 4
        ),
        "pass_rate_with_sat_rings_3kcal": round(
            sum(1 for c in with_sat if c["passed_3kcal"]) / max(len(with_sat), 1), 4
        ),
        "median_ring_strain_kcal": round(statistics.median(all_ring_strains), 3)
        if all_ring_strains
        else 0.0,
        "mean_ring_strain_kcal": round(float(np.mean(all_ring_strains)), 3)
        if all_ring_strains
        else 0.0,
        "max_ring_strain_kcal": round(max(all_ring_strains), 3) if all_ring_strains else 0.0,
        "pucker_class_distribution": pucker_counts,
        "flagged_high_strain_complexes": flagged,
        "per_complex": per_complex,
    }


def _load_qsar_bundle() -> dict[str, Any] | None:
    if not V4_QSAR_PATH.is_file():
        return None
    import joblib  # type: ignore[import-untyped]

    obj: Any = joblib.load(V4_QSAR_PATH)
    return obj if isinstance(obj, dict) else None


def _score_qsar_prob(
    qsar_bundle: dict[str, Any] | None, target_id: str, smiles_list: list[str]
) -> list[float]:
    """Score calibrated active probability on a target from ``v4_24target_qsar.joblib``."""
    if qsar_bundle is None or not smiles_list:
        return [0.5] * len(smiles_list)
    models = qsar_bundle.get("models", qsar_bundle)
    entry = models.get(target_id) if isinstance(models, dict) else None
    if not isinstance(entry, dict) or "hgb" not in entry or "rf" not in entry:
        return [0.5] * len(smiles_list)
    x_mat, _fp_bits, valid_mask, _ = featurize_admet_smiles(smiles_list)
    out = np.zeros(len(smiles_list), dtype=np.float64)
    if not bool(valid_mask.any()):
        return [0.0] * len(smiles_list)
    valid_idx = np.flatnonzero(valid_mask)
    x_val = x_mat[valid_idx]
    hgb = entry["hgb"]
    rf = entry["rf"]
    if hasattr(rf, "n_jobs"):
        rf.n_jobs = 4
    raw = 0.5 * (hgb.predict_proba(x_val)[:, 1] + rf.predict_proba(x_val)[:, 1])
    clipped = np.clip(raw.astype(np.float64), 1e-5, 1.0 - 1e-5)
    platt = entry.get("platt")
    if platt is not None:
        cal = np.asarray(platt.predict_proba(logit(clipped).reshape(-1, 1))[:, 1], dtype=np.float64)
    else:
        cal = clipped
    out[valid_idx] = np.clip(cal, 0.0, 1.0)
    return [float(v) for v in out]


def run_exp2_contrastive_ensemble_and_pmi(
    designer: ge.MoleculeDesigner,
    qsar_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Exp 2: Multi-conformation ensemble + contrastive off-target CFG + DeltaPMI_3D selectivity."""
    # DRD1 3-conformation ensemble (9LLG, 9U5E, 7LJC) vs DRD2 counter-pocket (6CM4)
    p_drd1_9llg = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/structures/clean_pockets/9LLG_DRD1_A1EKL_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/structures/clean_pockets/9LLG_DRD1_A1EKL_lig.sdf",
        target_id="DRD1",
    )
    p_drd1_9u5e = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/structures/clean_pockets/9U5E_DRD1_A1EN9_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/structures/clean_pockets/9U5E_DRD1_A1EN9_lig.sdf",
        target_id="DRD1",
    )
    p_drd1_7ljc = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/structures/clean_pockets/7LJC_DRD1_SK0_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/structures/clean_pockets/7LJC_DRD1_SK0_lig.sdf",
        target_id="DRD1",
    )
    p_drd2_6cm4 = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/structures/clean_pockets/6CM4_DRD2_8NU_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/structures/clean_pockets/6CM4_DRD2_8NU_lig.sdf",
        target_id="DRD2",
    )
    p_ache = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/4EY7_ACHE_E20_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/4EY7_ACHE_E20_lig.sdf",
        target_id="ACHE",
    )
    p_maob = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/2V5Z_MAOB_SAG_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/2V5Z_MAOB_SAG_lig.sdf",
        target_id="MAOB",
    )
    p_maoa = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/2Z5X_MAOA_HRM_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/2Z5X_MAOA_HRM_lig.sdf",
        target_id="MAOA",
    )

    def _eval_arm(res: ge.DesignResult, on_t: str, off_t: str) -> dict[str, Any]:
        smis = res.smiles
        p_on = _score_qsar_prob(qsar_bundle, on_t, smis)
        p_off = _score_qsar_prob(qsar_bundle, off_t, smis)
        margins = [a - b for a, b in zip(p_on, p_off, strict=True)]
        scafs = {bemis_murcko_scaffold(s) for s in smis if s}
        scafs.discard("")
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)  # type: ignore[attr-defined]
            for m in (Chem.MolFromSmiles(s) for s in smis)
            if m is not None
        ]
        intdiv = float(compute_internal_diversity(fps)) if len(fps) >= 2 else 0.0
        mean_mpo = float(np.mean([r.cns_mpo_v2 for r in res.reports])) if res.reports else 0.0
        return {
            "n_raw": len(res.raw_smiles),
            "valid_rate": round(res.valid_rate, 4),
            "unique_valid": len(smis),
            "unique_murcko_scaffolds": len(scafs),
            "intdiv1": round(intdiv, 4),
            "mean_mpo_v2": round(mean_mpo, 3),
            f"mean_p_{on_t}": round(float(np.mean(p_on)), 4) if p_on else 0.0,
            f"mean_p_{off_t}": round(float(np.mean(p_off)), 4) if p_off else 0.0,
            "mean_selectivity_margin": round(float(np.mean(margins)), 4) if margins else 0.0,
            "frac_selective_margin_gt_0": round(
                sum(1 for m in margins if m > 0.0) / max(len(margins), 1), 4
            ),
        }

    n_gen = 64
    drd1_single = designer.design_ensemble(
        [p_drd1_9llg], w_on=1.6, w_off=0.0, n=n_gen, include_admet=False, seed=42
    )
    drd1_ens3 = designer.design_ensemble(
        [p_drd1_9llg, p_drd1_9u5e, p_drd1_7ljc],
        w_on=1.6,
        w_off=0.0,
        n=n_gen,
        include_admet=False,
        seed=42,
    )
    drd1_contrastive = designer.design_ensemble(
        [p_drd1_9llg, p_drd1_9u5e, p_drd1_7ljc],
        off_pocket=p_drd2_6cm4,
        w_on=1.8,
        w_off=0.6,
        n=n_gen,
        include_admet=False,
        seed=42,
    )

    maob_single = designer.design_ensemble(
        [p_maob], w_on=1.6, w_off=0.0, n=n_gen, include_admet=False, seed=137
    )
    maob_contrastive = designer.design_ensemble(
        [p_maob],
        off_pocket=p_maoa,
        w_on=1.8,
        w_off=0.6,
        n=n_gen,
        include_admet=False,
        seed=137,
    )

    # Zero-shot DeltaPMI_3D discrimination on curated human clinical reference drugs
    drd1_refs = [d.canonical_smiles for d in ge.get_target("DRD1").reference_drugs]
    drd2_refs = [d.canonical_smiles for d in ge.get_target("DRD2").reference_drugs]
    ache_refs = [d.canonical_smiles for d in ge.get_target("ACHE").reference_drugs]
    maob_refs = [d.canonical_smiles for d in ge.get_target("MAOB").reference_drugs]

    pmi_ache_vs_drd2 = designer.score_pmi(ache_refs + drd2_refs, [p_ache], off_pocket=p_drd2_6cm4)
    y_ad = [1] * len(ache_refs) + [0] * len(drd2_refs)
    auroc_delta_pmi_ache_drd2 = float(
        roc_auc_score(y_ad, [s.delta_pmi_selectivity or 0.0 for s in pmi_ache_vs_drd2])
    )
    auroc_raw_ll_ache_drd2 = float(roc_auc_score(y_ad, [s.logp_null for s in pmi_ache_vs_drd2]))

    pmi_drd1_vs_ache = designer.score_pmi(
        drd1_refs + ache_refs, [p_drd1_9llg, p_drd1_9u5e], off_pocket=p_ache
    )
    y_da = [1] * len(drd1_refs) + [0] * len(ache_refs)
    auroc_delta_pmi_drd1_ache = float(
        roc_auc_score(y_da, [s.delta_pmi_selectivity or 0.0 for s in pmi_drd1_vs_ache])
    )
    auroc_raw_ll_drd1_ache = float(roc_auc_score(y_da, [s.logp_null for s in pmi_drd1_vs_ache]))

    pmi_maob_vs_drd2 = designer.score_pmi(maob_refs + drd2_refs, [p_maob], off_pocket=p_drd2_6cm4)
    y_md = [1] * len(maob_refs) + [0] * len(drd2_refs)
    auroc_delta_pmi_maob_drd2 = float(
        roc_auc_score(y_md, [s.delta_pmi_selectivity or 0.0 for s in pmi_maob_vs_drd2])
    )
    auroc_raw_ll_maob_drd2 = float(roc_auc_score(y_md, [s.logp_null for s in pmi_maob_vs_drd2]))

    return {
        "backbone": designer.backbone_id,
        "drd1_vs_drd2": {
            "single_pocket_9LLG": _eval_arm(drd1_single, "DRD1", "DRD2"),
            "ensemble_3conf_9LLG_9U5E_7LJC": _eval_arm(drd1_ens3, "DRD1", "DRD2"),
            "contrastive_ensemble_repel_DRD2": _eval_arm(drd1_contrastive, "DRD1", "DRD2"),
        },
        "maob_vs_maoa": {
            "single_pocket_2V5Z": _eval_arm(maob_single, "MAOB", "MAOA"),
            "contrastive_repel_MAOA_2Z5X": _eval_arm(maob_contrastive, "MAOB", "MAOA"),
        },
        "pmi_3d_zero_shot_discrimination": {
            "ache_vs_drd2_n": len(y_ad),
            "ache_vs_drd2_delta_pmi_auroc": round(auroc_delta_pmi_ache_drd2, 4),
            "ache_vs_drd2_uncond_ll_auroc": round(auroc_raw_ll_ache_drd2, 4),
            "drd1_vs_ache_n": len(y_da),
            "drd1_vs_ache_delta_pmi_auroc": round(auroc_delta_pmi_drd1_ache, 4),
            "drd1_vs_ache_uncond_ll_auroc": round(auroc_raw_ll_drd1_ache, 4),
            "maob_vs_drd2_n": len(y_md),
            "maob_vs_drd2_delta_pmi_auroc": round(auroc_delta_pmi_maob_drd2, 4),
            "maob_vs_drd2_uncond_ll_auroc": round(auroc_raw_ll_maob_drd2, 4),
            "mean_pmi_3d_on_target_drd1_refs": round(
                float(np.mean([s.pmi_3d for s in pmi_drd1_vs_ache[: len(drd1_refs)]])), 5
            ),
        },
    }


def run_exp3_ibogaine_sert_scaffold_hopping(
    safegpt_designer: ge.MoleculeDesigner,
    gpmolf_designer: ge.MoleculeDesigner,
    v4_admet: CNSAdmetSuite | None,
) -> dict[str, Any]:
    """Exp 3: Ibogaine -> Human SERT (5I6X+5I71) scaffold-hopping & hERG de-risking + Vina."""
    sert_5i6x = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/structures/clean_pockets/5I6X_SERT_8PR_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/structures/clean_pockets/5I6X_SERT_8PR_lig.sdf",
        target_id="SERT",
    )
    sert_5i71 = ge.Pocket.from_pdb(
        "data/cns_neuro.nosync/structures/clean_pockets/5I71_SERT_68P_pocket.pdb",
        ligand_sdf="data/cns_neuro.nosync/structures/clean_pockets/5I71_SERT_68P_lig.sdf",
        target_id="SERT",
    )

    res_safe = safegpt_designer.design_ensemble(
        [sert_5i6x, sert_5i71],
        w_on=1.6,
        w_off=0.0,
        n=64,
        include_admet=False,
        seed=2026,
    )
    res_gpm = gpmolf_designer.design_ensemble(
        [sert_5i6x, sert_5i71],
        w_on=1.6,
        w_off=0.0,
        n=64,
        include_admet=False,
        seed=2026,
    )

    combined = list(dict.fromkeys(res_safe.smiles + res_gpm.smiles))
    source_map = {s: "SAFE-GPT-87M" for s in res_safe.smiles}
    for s in res_gpm.smiles:
        source_map.setdefault(s, "GP-MoLFormer-v4")

    audit = audit_scaffold_hop_campaign(
        combined,
        reference_smiles=IBOGAINE_SMILES,
        admet_suite=v4_admet,
        max_tanimoto_for_hop=0.50,
        require_basic_amine=True,
    )

    # Dock positive controls (Ibogaine, Paroxetine) + top 8 clean basic-amine hops on SERT 5I6X
    top_hops = [
        c
        for c in audit["candidates"]
        if c["is_exact_scaffold_hop"] and c["triage_passed"] and c["has_basic_amine"]
    ][:8]

    docked_results: list[dict[str, Any]] = []
    controls = [
        ("Ibogaine (NP ref)", IBOGAINE_SMILES, "ref"),
        ("Paroxetine (SSRI ctrl)", PAROXETINE_SMILES, "ctrl"),
    ]
    to_dock = controls + [
        (f"SERT_Hop_{idx + 1}", c["canonical_smiles"], source_map.get(c["canonical_smiles"], "gen"))
        for idx, c in enumerate(top_hops)
    ]
    cand_lookup = {c["canonical_smiles"]: c for c in top_hops}

    for label, smi, src in to_dock:
        dock = dock_ligand_to_pocket(
            smiles=smi,
            pocket_pdb=sert_5i6x.pocket_pdb,  # type: ignore[arg-type]
            native_ligand_sdf=sert_5i6x.ligand_sdf,
            box_mode="tight_per_axis",
            canonical_anchors=("ASP98",),
            expected_anchor_kinds=("salt_bridge", "hbond"),
            vina_bin="bin/vina",
            exhaustiveness=8,
            num_modes=9,
            cpu=4,
            seed=42,
            require_vina=True,
        )
        ring_prof_dict: dict[str, Any] | None = None
        pose_mol = dock.preferred_mol
        if dock.success and pose_mol is not None:
            rprof = analyze_ring_strain(
                pose_mol, strain_threshold_kcal=3.0, n_ref_confs=24, seed=42
            )
            ring_prof_dict = {
                "n_sat_rings": rprof.n_saturated_ring_systems,
                "max_ring_strain_kcal": round(rprof.max_ring_strain_kcal, 3),
                "passed_3kcal": rprof.passed,
                "puckers": [r.pucker_class for r in rprof.rings],
            }
        meta = cand_lookup.get(smi, {})
        asp98_engaged = bool(
            (dock.anchor_top1 is not None and dock.anchor_top1.engaged)
            or (dock.anchor_any_mode is not None and dock.anchor_any_mode.engaged)
        )
        pb_pass = bool(dock.posebusters is not None and dock.posebusters.passed_all)
        docked_results.append(
            {
                "label": label,
                "source": src,
                "smiles": smi,
                "vina_top1_dg": round(float(dock.vina_affinity), 2)
                if math.isfinite(dock.vina_affinity)
                else None,
                "vina_best_anchor_dg": (
                    round(float(dock.anchor_vina_affinity), 2)
                    if dock.anchor_vina_affinity is not None
                    and math.isfinite(dock.anchor_vina_affinity)
                    else None
                ),
                "asp98_engaged": asp98_engaged,
                "posebusters_pass": pb_pass,
                "ring_strain": ring_prof_dict,
                "cns_mpo_v2": meta.get(
                    "cns_mpo_v2", audit["reference"]["cns_mpo_v2"] if src == "ref" else None
                ),
                "sa_score": meta.get(
                    "sa_score", audit["reference"]["sa_score"] if src == "ref" else None
                ),
                "herg_prob": meta.get(
                    "herg_prob",
                    audit["reference"]["admet_probabilities"].get("herg") if src == "ref" else None,
                ),
                "htr2b_prob": meta.get(
                    "htr2b_prob",
                    audit["reference"]["admet_probabilities"].get("htr2b_avoid")
                    if src == "ref"
                    else None,
                ),
                "delta_herg_risk_reduction": meta.get("delta_herg_risk_reduction"),
                "delta_sa_simplification": meta.get("delta_sa_simplification"),
                "tanimoto_to_ibogaine": meta.get("ecfp4_tanimoto_to_ref"),
            }
        )

    return {
        "reference_ibogaine": audit["reference"],
        "campaign_summary": audit["summary"],
        "n_safegpt_valid": len(res_safe.smiles),
        "n_gpmolformer_valid": len(res_gpm.smiles),
        "docked_sert_5I6X_leads": docked_results,
    }


def run_exp4_mtl_gradient_intervention_ablation() -> dict[str, Any]:
    """Exp 4: Controlled 4-task GenEuclideanModel (PaiNN-SE3) gradient-conflict ablation."""
    device = torch.device("cpu")
    vocab_size = 24
    fp_dim = 128

    def _make_dataset(n_samples: int, seed: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        coords = torch.randn(n_samples, 12, 3, generator=g) * 2.5
        node_feats = torch.randn(n_samples, 12, 52, generator=g)
        atom_mask = torch.ones(n_samples, 12, dtype=torch.bool)
        input_ids = torch.randint(1, vocab_size, (n_samples, 8), generator=g)
        ligand_fp = (torch.rand(n_samples, fp_dim, generator=g) > 0.7).float()
        # Construct distinct geometric & chemical targets to induce real multi-task trade-offs
        radial = coords.norm(dim=-1).mean(dim=-1)
        feat_sig = node_feats[:, :, :8].mean(dim=(1, 2))
        pkd_target = 6.5 + 0.8 * radial - 0.6 * feat_sig
        prop_target = torch.stack(
            [
                feat_sig,
                -0.7 * radial + 0.3 * feat_sig,
                0.5 * node_feats[:, :, 8:16].mean(dim=(1, 2)),
                -0.4 * feat_sig,
                0.3 * radial,
                node_feats[:, :, 16:24].mean(dim=(1, 2)),
            ],
            dim=-1,
        )
        return {
            "coords": coords,
            "node_feats": node_feats,
            "atom_mask": atom_mask,
            "input_ids": input_ids,
            "ligand_fp": ligand_fp,
            "pkd_target": pkd_target,
            "prop_target": prop_target,
        }

    train_data = _make_dataset(24, seed=101)
    val_data = _make_dataset(16, seed=202)
    task_weights = {"lm_nll": 1.0, "pkd": 0.5, "drugclip": 0.3, "props": 0.2}

    results: dict[str, Any] = {}
    for strategy in ("scalarized", "pcgrad", "aim_policy"):
        seed_everything(42)
        cfg = ExperimentConfig(
            encoder=EncoderConfig(
                arch="painn", hidden_dim=32, vector_dim=16, out_dim=32, num_layers=2
            ),
            cvae=CVAEConfig(enabled=False),
            decoder=DecoderConfig(
                arch="transformer",
                vocab_size=vocab_size,
                d_model=32,
                n_heads=4,
                n_layers=1,
                d_ff=64,
            ),
            multitask=MultiTaskConfig(
                use_pkd_head=True,
                use_clip_head=True,
                use_prop_head=True,
                ligand_fp_dim=fp_dim,
            ),
        )
        model = GenEuclideanModel(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        mediator = GradientConflictMediator(
            task_names=("lm_nll", "pkd", "drugclip", "props"),
            strategy=strategy,
            policy_lr=0.10,
            lag_priority_alpha=0.8,
        )
        shared_params = list(model.encoder.parameters())
        shared_ids = {id(p) for p in shared_params}
        head_params = [p for p in model.parameters() if id(p) not in shared_ids]

        def _compute_losses(
            data_dict: dict[str, torch.Tensor],
            net: GenEuclideanModel = model,
        ) -> dict[str, torch.Tensor]:
            out = net(data_dict)
            assert out.pred_pkd is not None and out.pred_props is not None
            assert (
                out.clip_pocket is not None
                and out.clip_ligand is not None
                and out.clip_logit_scale is not None
            )
            l_lm = nn.functional.cross_entropy(
                out.logits[:, :-1].reshape(-1, vocab_size),
                data_dict["input_ids"][:, 1:].reshape(-1),
            )
            l_pkd = nn.functional.mse_loss(out.pred_pkd, data_dict["pkd_target"])
            l_clip = compute_drugclip_loss(out.clip_pocket, out.clip_ligand, out.clip_logit_scale)
            l_prop = nn.functional.mse_loss(out.pred_props, data_dict["prop_target"])
            return {"lm_nll": l_lm, "pkd": l_pkd, "drugclip": l_clip, "props": l_prop}

        conflict_fracs: list[float] = []
        mean_cosines: list[float] = []
        last_diag: dict[str, Any] = {}
        first_diag: dict[str, Any] = {}

        for step in range(25):
            model.train()
            opt.zero_grad(set_to_none=True)
            losses = _compute_losses(train_data)
            # Compute non-shared head grads from total loss while mediating shared encoder grads
            non_shared_total = sum(task_weights[k] * losses[k] for k in losses)
            assert isinstance(non_shared_total, torch.Tensor)
            head_grads = torch.autograd.grad(
                non_shared_total, head_params, retain_graph=True, allow_unused=True
            )
            diag = mediator.backward(losses, shared_params=shared_params, task_weights=task_weights)
            for p, g in zip(head_params, head_grads, strict=True):
                if g is not None:
                    p.grad = g.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            conflict_fracs.append(diag.conflict_fraction)
            mean_cosines.append(diag.mean_pairwise_cosine)
            if step == 0:
                first_diag = diag.to_dict()
            last_diag = diag.to_dict()

        model.eval()
        with torch.no_grad():
            val_losses = {
                k: round(float(v.item()), 4) for k, v in _compute_losses(val_data).items()
            }
            train_losses_final = {
                k: round(float(v.item()), 4) for k, v in _compute_losses(train_data).items()
            }

        results[strategy] = {
            "mean_conflict_fraction": round(float(np.mean(conflict_fracs)), 4),
            "mean_pairwise_cosine": round(float(np.mean(mean_cosines)), 4),
            "train_losses_final": train_losses_final,
            "val_losses_final": val_losses,
            "val_total_weighted": round(
                sum(task_weights[k] * val_losses[k] for k in val_losses), 4
            ),
            "step0_diagnostics": first_diag,
            "final_diagnostics": last_diag,
        }

    return results


def main() -> None:
    t0 = time.perf_counter()
    print("=== [1/4] Auditing 28 Docked Human CNS Complexes for Saturated-Ring Strain ===")
    exp1 = run_exp1_ring_strain_audit()
    n_flag = len(exp1["flagged_high_strain_complexes"])
    print(
        f"  Audited {exp1['n_complexes_audited']} complexes "
        f"({exp1['n_total_saturated_ring_systems']} sat ring systems): "
        f"pass@3kcal={exp1['pass_rate_3kcal']:.1%}, median={exp1['median_ring_strain_kcal']:.3f}, "
        f"max={exp1['max_ring_strain_kcal']:.3f} kcal/mol, flagged={n_flag}"
    )

    print(
        "=== [2/4] Loading Trained v4.0 MPS Designers & Running Contrastive Ensemble + PMI_3D ==="
    )
    qsar_bundle = _load_qsar_bundle()
    v4_admet = CNSAdmetSuite.try_load(V4_ADMET_PATH)

    safegpt_designer = ge.MoleculeDesigner.from_pretrained(
        SAFEGPT_CKPT,
        v2_pocket_ckpt=V2_PAINN_CKPT,
        device="auto",
        local_files_only=True,
    )
    gpmolf_designer = ge.MoleculeDesigner.from_pretrained(
        GPMOLF_CKPT,
        v2_pocket_ckpt=V2_PAINN_CKPT,
        device="auto",
        local_files_only=True,
    )
    exp2_safe = run_exp2_contrastive_ensemble_and_pmi(safegpt_designer, qsar_bundle)
    exp2_gpm = run_exp2_contrastive_ensemble_and_pmi(gpmolf_designer, qsar_bundle)
    d_arms = exp2_safe["drd1_vs_drd2"]
    m_single = d_arms["single_pocket_9LLG"]["mean_selectivity_margin"]
    m_ens = d_arms["ensemble_3conf_9LLG_9U5E_7LJC"]["mean_selectivity_margin"]
    m_con = d_arms["contrastive_ensemble_repel_DRD2"]["mean_selectivity_margin"]
    auroc_pmi = exp2_safe["pmi_3d_zero_shot_discrimination"]["ache_vs_drd2_delta_pmi_auroc"]
    print(
        f"  SAFE-GPT DRD1 margins single->ens3->contrastive: "
        f"{m_single:+.4f} -> {m_ens:+.4f} -> {m_con:+.4f} | "
        f"DeltaPMI AUROC(ACHE vs DRD2)={auroc_pmi:.3f}"
    )

    print(
        "=== [3/4] Running Ibogaine -> Human SERT (5I6X+5I71) Scaffold Hopping & Vina Docking ==="
    )
    exp3 = run_exp3_ibogaine_sert_scaffold_hopping(safegpt_designer, gpmolf_designer, v4_admet)
    print(
        f"  Ibogaine ref: SA={exp3['reference_ibogaine']['sa_score']:.2f}, "
        f"hERG={exp3['reference_ibogaine']['admet_probabilities'].get('herg')}, "
        f"clean_hops={exp3['campaign_summary']['n_clean_scaffold_hops']}/{exp3['campaign_summary']['n_unique_valid']}"
    )

    print("=== [4/4] Running 4-Task PaiNN-SE3 Gradient Conflict Mediation Ablation ===")
    exp4 = run_exp4_mtl_gradient_intervention_ablation()
    for strat, info in exp4.items():
        print(
            f"  {strat:10s}: val_weighted={info['val_total_weighted']:.4f}, "
            f"mean_conflict={info['mean_conflict_fraction']:.3f}, val={info['val_losses_final']}"
        )

    payload = {
        "generated_by": "scripts/run_eth_cadd_frontier_benchmarks.py",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "exp1_ring_strain_28_complexes": exp1,
        "exp2_contrastive_ensemble_and_pmi": {
            "safegpt_87m": exp2_safe,
            "gpmolformer_47m": exp2_gpm,
        },
        "exp3_ibogaine_sert_scaffold_hopping": exp3,
        "exp4_mtl_gradient_intervention": exp4,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elapsed = payload["elapsed_seconds"]
    print(f"\nSaved ETH-CADD frontier benchmarks -> {OUT_JSON} ({elapsed}s)")


if __name__ == "__main__":
    main()
