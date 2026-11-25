#!/usr/bin/env python3
"""Reproducible Tourette CSTC Basal Ganglia & GABAergic Pocket Benchmark Study.

Runs end-to-end empirical validation across the 4 newly curated *Homo sapiens*
GABAergic impulse-braking & Basal Ganglia signal-filtering pockets:
  - ``GABRA5`` (``8BHG``, 2.39 A X-ray multi-chain alpha5+/gamma2- Bretazenil partial PAM)
  - ``GAT1`` (``SLC6A1``, ``7Y7Z``, 3.20 A cryo-EM 100% WT human Tiagabine inhibitor)
  - ``ADORA2A`` (``9EE8``, 2.63 A cryo-EM 2025 GPCRdb=Active miniGs + Adenosine agonist)
  - ``GRM5`` (``mGluR5``, ``4OO9``, 2.60 A X-ray GPCRdb=Inactive Mavoglurant NAM)
  (plus native full-length ``GABRA1`` ``6X3X`` Diazepam as multi-chain alpha1+/gamma2- control)

Stages executed:
  1. Pocket featurization & canonical hotspot audit (``(N, 52)``, ``0 NaN``, anchors < 4.5 A).
  2. Gate 1 Unbiased 3-Seed AutoDock Vina v1.2.7 native redocking + 3D PLIF recovery
     via ``geneuclidean.evaluation.docking_pipeline.redock_native_complex``.
  3. Zero-Shot 3D-Conditioned De Novo Generation using frozen
     ``PaiNN-SE3 + GP-MoLFormer v3.1 CNS-DPO`` (N=64/pocket) vs Random ZINC-250k control
     (N=64/pocket), evaluated with ``medchem_triage.py`` (0 DCU/PAINS), Pfizer CNS MPO v2,
     12-endpoint v4.0 Avoid-ome/ADMET suite (``v4_admet_avoidome_oracles.joblib``),
     ChEMBL 37 human scaffold-hop novelty, and held-out Vina v1.2.7 docking + 3D PLIF on
     Top-5 MaxMin-diverse hygiene-passing leads + clinical reference drugs.
  4. Saves verified 3D docked SDFs under
     ``data/cns_neuro.nosync/ts_circuit_expansion/docked_poses_3d/`` and writes compact
     (<500 KB) summary JSON to ``results/tables/ts_circuit_audit_and_benchmark.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import MaxMinPicker
from scipy.stats import mannwhitneyu

from geneuclidean.cns.admet import (
    DEFAULT_V4_ADMET_MODEL_PATH,
    CNSAdmetSuite,
)
from geneuclidean.cns.medchem_triage import (
    compute_scaffold_hop_metrics,
    triage_molecule,
)
from geneuclidean.cns.mpo import compute_cns_mpo
from geneuclidean.cns.targets import audit_pocket_hotspots
from geneuclidean.cns.tourette_circuit import (
    DEFAULT_TS_CIRCUIT_DIR,
    DEFAULT_TS_CIRCUIT_STRUCTURES_DIR,
    TOURETTE_CIRCUIT_EXPANSION_REGISTRY,
    TOURETTE_CSTC_CIRCUIT_AUDIT,
    find_local_ts_circuit_pocket_files,
    load_tourette_circuit_pocket,
    score_tourette_circuit_candidate,
)
from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything, select_device
from geneuclidean.data.featurizer import LIGAND_FP_DIM, POCKET_NODE_DIM, PocketFeatures
from geneuclidean.evaluation.docking_pipeline import (
    DockingPoseResult,
    dock_ligand_to_pocket,
    redock_native_complex,
)
from geneuclidean.evaluation.geometry3d import prepare_receptor_pdbqt
from geneuclidean.evaluation.interactions3d import profile_3d_interactions
from geneuclidean.evaluation.protonation import protonate_mol_ph74
from geneuclidean.models.encoders.rbf import PocketEncoderOutput
from geneuclidean.models.hf_backbones import load_hf_backbone
from geneuclidean.models.llm_bridge import PocketConditionedLLM
from geneuclidean.sampling.llm_sampler import sample_llm_molecules

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

PRIMARY_TS_EXPANSION_IDS: tuple[str, ...] = ("GABRA5", "GAT1", "ADORA2A", "GRM5")

V3_CONFIG_PATH = Path("configs/geneuclidean_v3_gpmolformer.yaml")
V2_PAINN_CKPT = Path("runs_local.nosync/geneuclidean_v2_painn/best.pt")
V31_DPO_ADAPTER_CKPT = Path("runs_local.nosync/geneuclidean_v31_cns_dpo/best_adapter.pt")
ZINC_SMILES_PATH = Path("data/processed_local.nosync/zinc250k_smiles.txt")
CHEMBL_ACTIVES_CSV = DEFAULT_TS_CIRCUIT_DIR / "chembl" / "ts_circuit_4targets_chembl37_actives.csv"
CHEMBL_SUMMARY_JSON = DEFAULT_TS_CIRCUIT_DIR / "chembl" / "ts_circuit_chembl37_summary.json"
DOCKED_POSES_DIR = DEFAULT_TS_CIRCUIT_DIR / "docked_poses_3d"
OUTPUT_JSON_PATH = Path("results/tables/ts_circuit_audit_and_benchmark.json")

# Target-specific canonical 3D anchor residues & interaction channels
TARGET_ANCHOR_CONFIG: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "GABRA5": (
        ("HIS105", "PHE77", "THR142", "SER209", "TYR163", "TYR213"),
        ("hbond", "pi_stacking", "salt_bridge", "halogen_bond"),
    ),
    "GAT1": (
        ("GLY65", "SER302", "TYR60", "PHE294", "SER396"),
        ("hbond", "salt_bridge", "pi_stacking", "cation_pi"),
    ),
    "ADORA2A": (
        ("ASN253", "SER277", "HIS278", "PHE168", "GLU169", "THR88"),
        ("hbond", "pi_stacking", "salt_bridge"),
    ),
    "GRM5": (
        ("ASN747", "SER809", "SER805", "TYR659", "TRP785", "PHE788"),
        ("hbond", "pi_stacking", "halogen_bond"),
    ),
    "GABRA1_COMP": (
        ("HIS102", "PHE77", "SER205", "TYR160", "TYR210"),
        ("hbond", "pi_stacking", "halogen_bond"),
    ),
}

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_MORGAN_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=LIGAND_FP_DIM)


def _smiles_to_fp_1024(smiles_list: Sequence[str]) -> torch.Tensor:
    """Compute [N, 1024] Morgan FP tensor for frozen PaiNN-SE3 DrugCLIP/pKd scoring."""
    arr = np.zeros((len(smiles_list), LIGAND_FP_DIM), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = _MORGAN_1024.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr[i])
    return torch.from_numpy(arr)


def _pocket_to_batch(
    pf: PocketFeatures,
    model: PocketConditionedLLM,
    device: torch.device,
) -> tuple[PocketEncoderOutput, torch.Tensor]:
    """Encode a ``PocketFeatures`` graph into model-ready batched tensors."""
    coords = torch.from_numpy(pf.coords).float().unsqueeze(0).to(device)
    feats = torch.from_numpy(pf.node_feats).float().unsqueeze(0).to(device)
    mask = torch.ones((1, int(pf.coords.shape[0])), dtype=torch.bool, device=device)
    with torch.no_grad():
        enc = model.encode_pocket(coords, feats, mask)
    return enc, mask


def _load_v31_dpo_model(device: torch.device) -> PocketConditionedLLM:
    """Load frozen PaiNN-SE3 + GP-MoLFormer-Uniq + v3.1 CNS-DPO LoRA/Perceiver adapter."""
    cfg = ExperimentConfig.from_yaml(V3_CONFIG_PATH)
    cfg.data.randomize_smiles = False
    llm_cfg = cfg.llm if cfg.llm is not None else LLMConfig()
    os.environ["HF_HOME"] = str(Path(llm_cfg.cache_dir).resolve())

    backbone = load_hf_backbone(
        backbone_id=llm_cfg.backbone_id,
        cache_dir=llm_cfg.cache_dir,
        dtype=llm_cfg.dtype,
        lora_r=llm_cfg.lora_r,
        lora_alpha=llm_cfg.lora_alpha,
        lora_dropout=llm_cfg.lora_dropout,
        use_lora=True,
        local_files_only=True,
    )
    model = PocketConditionedLLM(
        v2_cfg=cfg,
        backbone=backbone,
        num_queries=llm_cfg.num_queries,
        perceiver_layers=llm_cfg.perceiver_layers,
        perceiver_heads=llm_cfg.perceiver_heads,
        num_global_tokens=llm_cfg.num_global_tokens,
        cfg_drop_prob=llm_cfg.cfg_drop_prob,
        freeze_pocket_encoder=True,
    )
    model.load_v2_pocket_checkpoint(V2_PAINN_CKPT)
    model.load_adapter_checkpoint(V31_DPO_ADAPTER_CKPT, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def _load_chembl_actives_by_target(csv_path: Path) -> dict[str, list[str]]:
    """Load downloaded ChEMBL 37 human actives grouped by target_id."""
    out: dict[str, list[str]] = {tid: [] for tid in PRIMARY_TS_EXPANSION_IDS}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tid = str(row["target_id"]).strip().upper()
            smi = str(row["canonical_smiles"]).strip()
            if tid in out and smi:
                out[tid].append(smi)
    return out


def _load_zinc_pool(exclude_smiles: set[str], max_load: int = 40000) -> list[str]:
    """Load canonical ZINC-250k SMILES excluding any ChEMBL actives."""
    path = ZINC_SMILES_PATH
    if not path.exists():
        alt = Path("data/processed/zinc250k_smiles.txt")
        if alt.exists():
            path = alt
    pool: list[str] = []
    if not path.exists():
        return pool
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s not in exclude_smiles:
                pool.append(s)
            if len(pool) >= max_load:
                break
    return pool


def _select_top_k_maxmin_diverse(
    records: list[dict[str, Any]],
    *,
    k: int = 5,
    pool_top_n: int = 24,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select Top-k MaxMin-diverse hygiene-passing unique molecules ranked by composite priority."""
    clean = [
        r
        for r in records
        if r["valid"]
        and r["hygiene_gate"] >= 0.70
        and r["pains_free"]
        and r["reagent_quarantine_free"]
    ]
    if len(clean) < k:
        clean = [r for r in records if r["valid"] and r["hygiene_gate"] > 0.0]

    # Deduplicate by canonical SMILES keeping highest pre-docking score
    dedup: dict[str, dict[str, Any]] = {}
    for r in sorted(clean, key=lambda x: -float(x["pre_dock_priority"])):
        smi = str(r["canonical_smiles"])
        if smi not in dedup:
            dedup[smi] = r
    shortlist = list(dedup.values())[: max(k, pool_top_n)]
    if len(shortlist) <= k:
        return shortlist

    fps = [_MORGAN_GEN.GetFingerprint(Chem.MolFromSmiles(r["canonical_smiles"])) for r in shortlist]
    picker = MaxMinPicker()

    def _dist(i: int, j: int) -> float:
        return float(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    # Seed with the #1 highest-scoring molecule, then greedily maximize scaffold diversity
    picked = list(picker.LazyPick(_dist, len(shortlist), k, firstPicks=[0], seed=seed))
    return [shortlist[int(idx)] for idx in picked]


def _mean_pairwise_tanimoto_div(smiles_list: Sequence[str]) -> float:
    """Compute IntDiv1 (mean pairwise Tanimoto distance) across unique valid molecules."""
    fps = [
        _MORGAN_GEN.GetFingerprint(m)
        for s in dict.fromkeys(smiles_list)
        if (m := Chem.MolFromSmiles(s)) is not None
    ]
    if len(fps) < 2:
        return 0.0
    dists: list[float] = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        dists.extend(1.0 - float(v) for v in sims)
    return float(np.mean(dists)) if dists else 0.0


def _compact_pose_summary(
    res: DockingPoseResult,
    *,
    label: str,
    target_id: str,
    admet_suite: CNSAdmetSuite | None,
    chembl_refs: Sequence[str],
    pre_record: dict[str, Any] | None = None,
    sdf_rel_path: str | None = None,
) -> dict[str, Any]:
    """Serialize one docked ligand + mechanism-aware score + scaffold-hop metrics."""
    anch = res.anchor_any_mode or res.anchor_top1
    score = score_tourette_circuit_candidate(
        res.smiles,
        target_id,
        vina_dg_kcal=res.vina_affinity if math.isfinite(res.vina_affinity) else None,
        anchor_engaged=bool(anch.engaged) if anch is not None else None,
        posebusters_pass=(
            bool(res.posebusters.passed_all) if res.posebusters is not None else None
        ),
        plif_tanimoto=res.plif_tanimoto,
        admet_suite=admet_suite,
    )
    hop = compute_scaffold_hop_metrics(res.smiles, reference_smiles=chembl_refs)
    admet_probs = (
        {k: round(float(v), 4) for k, v in score.admet_v4.probabilities.items()}
        if score.admet_v4 is not None
        else {}
    )
    top_contacts: list[dict[str, Any]] = []
    if res.interactions is not None:
        for rec in res.interactions.interactions[:10]:
            top_contacts.append(
                {
                    "residue": rec.protein_residue,
                    "kind": rec.kind,
                    "distance_A": round(float(rec.distance_angstrom), 2),
                }
            )
    return {
        "label": label,
        "canonical_smiles": res.smiles,
        "smiles_ph74": res.smiles_ph74,
        "net_charge_ph74": res.net_charge_ph74,
        "docking_success": res.success,
        "vina_dg_kcal": round(float(res.vina_affinity), 3)
        if math.isfinite(res.vina_affinity)
        else None,
        "ligand_efficiency": round(float(res.ligand_efficiency), 4)
        if math.isfinite(res.ligand_efficiency)
        else None,
        "lle_vina": round(float(res.lle_vina), 3) if math.isfinite(res.lle_vina) else None,
        "posebusters_passed": (
            bool(res.posebusters.passed_all) if res.posebusters is not None else False
        ),
        "plif_tanimoto": res.plif_tanimoto,
        "plif_tanimoto_polar": res.plif_tanimoto_polar,
        "residue_recovery": res.residue_recovery,
        "anchor_top1_engaged": bool(res.anchor_top1.engaged) if res.anchor_top1 else False,
        "anchor_any_mode_engaged": bool(anch.engaged) if anch else False,
        "anchor_mode_rank": res.anchor_mode_rank,
        "anchor_min_dist_A": anch.min_distance_angstrom if anch else None,
        "cns_mpo_v2": round(float(score.cns_mpo_v2), 3),
        "hygiene_gate": round(float(score.hygiene_gate), 3),
        "lilly_demerit_score": score.triage.lilly_demerit_score if score.triage else None,
        "pains_free": bool(score.triage.pains_free) if score.triage else False,
        "reagent_quarantine_free": (
            bool(score.triage.reagent_quarantine_free) if score.triage else False
        ),
        "circuit_readiness_score": round(float(score.circuit_readiness_score), 4),
        "mechanism_flags": list(score.mechanism_flags),
        "admet_v4_probs": admet_probs,
        "chembl_max_tanimoto": round(float(hop.max_tanimoto), 4),
        "is_novel_scaffold": bool(hop.is_novel_scaffold),
        "fto_pre_screen_flag": hop.fto_pre_screen_flag,
        "pred_pkd_painn": (
            round(float(pre_record["pred_pkd"]), 3)
            if pre_record and math.isfinite(float(pre_record.get("pred_pkd", float("nan"))))
            else None
        ),
        "clip_sim_painn": (
            round(float(pre_record["clip_sim"]), 4)
            if pre_record and math.isfinite(float(pre_record.get("clip_sim", float("nan"))))
            else None
        ),
        "top_3d_contacts": top_contacts,
        "docked_sdf_path": sdf_rel_path,
    }


def _evaluate_candidate_pool(
    smiles_list: Sequence[str],
    target_id: str,
    *,
    p_enc: PocketEncoderOutput,
    model: PocketConditionedLLM,
    admet_suite: CNSAdmetSuite | None,
    chembl_set: set[str],
    zinc_set: set[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate N generated or control molecules with PaiNN-SE3, MedChem triage, MPO & v4 ADMET."""
    # Frozen PaiNN-SE3 3D DrugCLIP + pKd scoring
    with torch.no_grad():
        fps_t = _smiles_to_fp_1024(smiles_list).to(device)
        sc_3d = model.score_candidates(p_enc, fps_t) if smiles_list else {}
        pkd_arr = (
            sc_3d["pkd"].detach().cpu().numpy().tolist()
            if "pkd" in sc_3d
            else [float("nan")] * len(smiles_list)
        )
        clip_arr = (
            sc_3d["clip_sim"].detach().cpu().numpy().tolist()
            if "clip_sim" in sc_3d
            else [float("nan")] * len(smiles_list)
        )

    valid_canon: list[str] = []
    records: list[dict[str, Any]] = []
    admet_preds = (
        admet_suite.predict_v4(list(smiles_list))
        if admet_suite is not None and len(smiles_list) > 0
        else [None] * len(smiles_list)
    )

    for idx, raw_smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(raw_smi) if isinstance(raw_smi, str) and raw_smi.strip() else None
        if mol is None:
            records.append(
                {
                    "raw_smiles": raw_smi,
                    "canonical_smiles": "",
                    "valid": False,
                    "hygiene_gate": 0.0,
                    "pains_free": False,
                    "reagent_quarantine_free": False,
                    "lilly_pass": False,
                    "cns_mpo_v2": 0.0,
                    "pred_pkd": float("nan"),
                    "clip_sim": float("nan"),
                    "pre_dock_priority": 0.0,
                }
            )
            continue
        can = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        valid_canon.append(can)
        tr = triage_molecule(can)
        mpo = compute_cns_mpo(can)
        ap = admet_preds[idx]
        probs = ap.probabilities if (ap is not None and ap.valid) else {}
        bbb = float(probs.get("bbb", mpo.cns_mpo_v2 / 6.0))
        herg = float(probs.get("herg", 0.30))
        pgp = float(probs.get("pgp", probs.get("mdr1_efflux", 0.30)))
        htr2b = float(probs.get("htr2b_avoid", 0.15))
        hrh1 = float(probs.get("hrh1_avoid", 0.20))
        pkd_val = float(pkd_arr[idx]) if math.isfinite(float(pkd_arr[idx])) else 6.5
        clip_val = float(clip_arr[idx]) if math.isfinite(float(clip_arr[idx])) else 0.0

        pkd_norm = float(np.clip((pkd_val - 4.5) / 5.5, 0.0, 1.0))
        clip_norm = float(np.clip((clip_val + 0.1) / 0.8, 0.0, 1.0))
        admet_safe = float(
            np.clip(
                0.35 * bbb
                + 0.25 * (1.0 - herg)
                + 0.15 * (1.0 - pgp)
                + 0.15 * (1.0 - htr2b)
                + 0.10 * (1.0 - hrh1),
                0.0,
                1.0,
            )
        )
        priority = float(
            tr.hygiene_gate
            * (
                0.35 * pkd_norm
                + 0.20 * clip_norm
                + 0.25 * (mpo.cns_mpo_v2 / 6.0)
                + 0.20 * admet_safe
            )
        )
        records.append(
            {
                "raw_smiles": raw_smi,
                "canonical_smiles": can,
                "valid": True,
                "hygiene_gate": float(tr.hygiene_gate),
                "pains_free": bool(tr.pains_free),
                "reagent_quarantine_free": bool(tr.reagent_quarantine_free),
                "lilly_pass": bool(tr.lilly_pass),
                "lilly_demerit_score": int(tr.lilly_demerit_score),
                "cns_mpo_v2": float(mpo.cns_mpo_v2),
                "bbb_prob": bbb,
                "herg_prob": herg,
                "pgp_prob": pgp,
                "htr2b_prob": htr2b,
                "hrh1_prob": hrh1,
                "novel_vs_chembl": can not in chembl_set,
                "novel_vs_chembl_zinc": (can not in chembl_set) and (can not in zinc_set),
                "pred_pkd": pkd_val,
                "clip_sim": clip_val,
                "pre_dock_priority": priority,
            }
        )

    valid_recs = [r for r in records if r["valid"]]
    n_tot = max(len(smiles_list), 1)
    n_val = max(len(valid_recs), 1)
    uniq_valid = list(dict.fromkeys(valid_canon))
    summary = {
        "n_total": len(smiles_list),
        "n_valid": len(valid_recs),
        "validity": round(len(valid_recs) / n_tot, 4),
        "uniqueness": round(len(uniq_valid) / n_val, 4) if valid_recs else 0.0,
        "int_div_1": round(_mean_pairwise_tanimoto_div(uniq_valid), 4),
        "novelty_vs_chembl": round(sum(1 for r in valid_recs if r["novel_vs_chembl"]) / n_val, 4)
        if valid_recs
        else 0.0,
        "strict_novelty_vs_chembl_zinc": round(
            sum(1 for r in valid_recs if r["novel_vs_chembl_zinc"]) / n_val, 4
        )
        if valid_recs
        else 0.0,
        "pains_free_rate": round(sum(1 for r in valid_recs if r["pains_free"]) / n_val, 4)
        if valid_recs
        else 0.0,
        "reagent_quarantine_free_rate": round(
            sum(1 for r in valid_recs if r["reagent_quarantine_free"]) / n_val, 4
        )
        if valid_recs
        else 0.0,
        "dcu_count": sum(
            1 for r in valid_recs if r["canonical_smiles"] == "O=C(NC1CCCCC1)NC1CCCCC1"
        ),
        "lilly_pass_rate": round(sum(1 for r in valid_recs if r["lilly_pass"]) / n_val, 4)
        if valid_recs
        else 0.0,
        "hygiene_pass_rate": round(
            sum(1 for r in valid_recs if r["hygiene_gate"] >= 0.70) / n_val, 4
        )
        if valid_recs
        else 0.0,
        "cns_mpo_v2_mean": round(float(np.mean([r["cns_mpo_v2"] for r in valid_recs])), 3)
        if valid_recs
        else 0.0,
        "bbb_prob_mean": round(float(np.mean([r["bbb_prob"] for r in valid_recs])), 4)
        if valid_recs
        else 0.0,
        "herg_prob_mean": round(float(np.mean([r["herg_prob"] for r in valid_recs])), 4)
        if valid_recs
        else 0.0,
        "pgp_prob_mean": round(float(np.mean([r["pgp_prob"] for r in valid_recs])), 4)
        if valid_recs
        else 0.0,
        "htr2b_avoid_prob_mean": round(float(np.mean([r["htr2b_prob"] for r in valid_recs])), 4)
        if valid_recs
        else 0.0,
        "hrh1_avoid_prob_mean": round(float(np.mean([r["hrh1_prob"] for r in valid_recs])), 4)
        if valid_recs
        else 0.0,
        "pred_pkd_mean": round(float(np.mean([r["pred_pkd"] for r in valid_recs])), 3)
        if valid_recs
        else 0.0,
        "clip_sim_mean": round(float(np.mean([r["clip_sim"] for r in valid_recs])), 4)
        if valid_recs
        else 0.0,
    }
    return summary, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tourette CSTC Basal Ganglia & GABAergic Pocket Audit + Docking Study."
    )
    parser.add_argument("--num-samples", type=int, default=64, help="Generated mols per pocket.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k MaxMin-diverse mols to dock.")
    parser.add_argument(
        "--exhaustiveness", type=int, default=8, help="AutoDock Vina exhaustiveness."
    )
    parser.add_argument("--vina-cpu", type=int, default=4, help="CPU threads per Vina run.")
    parser.add_argument("--seed", type=int, default=42, help="Master deterministic seed.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON_PATH, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    seed_everything(args.seed)
    device = select_device("auto")
    print(
        f"[ts-circuit-study] Starting study on {device} "
        f"(N={args.num_samples}/pocket, Top-{args.top_k} docked, "
        f"exhaustiveness={args.exhaustiveness})",
        flush=True,
    )

    DOCKED_POSES_DIR.mkdir(parents=True, exist_ok=True)
    rec_cache_dir = DEFAULT_TS_CIRCUIT_DIR / "receptor_pdbqt_cache"
    rec_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load ChEMBL 37 human actives & ZINC-250k control pool
    chembl_by_target = _load_chembl_actives_by_target(CHEMBL_ACTIVES_CSV)
    all_chembl_set = {s for smis in chembl_by_target.values() for s in smis}
    chembl_summary = (
        json.loads(CHEMBL_SUMMARY_JSON.read_text(encoding="utf-8"))
        if CHEMBL_SUMMARY_JSON.exists()
        else {}
    )
    zinc_pool = _load_zinc_pool(exclude_smiles=all_chembl_set, max_load=40000)
    zinc_set = set(zinc_pool)
    print(
        f"[ts-circuit-study] Loaded {len(all_chembl_set)} unique ChEMBL 37 human actives & "
        f"{len(zinc_pool)} ZINC-250k control molecules.",
        flush=True,
    )

    # Load 12-endpoint v4.0 Avoid-ome / ADMET suite
    admet_suite = CNSAdmetSuite.try_load(DEFAULT_V4_ADMET_MODEL_PATH)
    if admet_suite is not None:
        print(
            f"[ts-circuit-study] Loaded v4.0 ADMET/Avoid-ome suite "
            f"({len(admet_suite.fitted_endpoints)} endpoints: {admet_suite.fitted_endpoints}).",
            flush=True,
        )
    else:
        print("[ts-circuit-study] WARNING: v4.0 ADMET suite not found; using MPO fallback.")

    # Load frozen PaiNN-SE3 + GP-MoLFormer v3.1 CNS-DPO model
    model = _load_v31_dpo_model(device)
    print("[ts-circuit-study] Loaded frozen PaiNN-SE3 + GP-MoLFormer v3.1 CNS-DPO.", flush=True)

    # ------------------------------------------------------------------
    # Stage 1: Pocket Featurization, Hotspot Audit & Gate 1 Redocking
    # ------------------------------------------------------------------
    pocket_audits: dict[str, Any] = {}
    gate1_results: dict[str, Any] = {}
    pocket_files_map: dict[str, tuple[Path, Path]] = {}
    receptor_pdbqt_map: dict[str, Path] = {}
    pocket_feats_map: dict[str, PocketFeatures] = {}

    for tid in PRIMARY_TS_EXPANSION_IDS:
        spec = TOURETTE_CIRCUIT_EXPANSION_REGISTRY[tid]
        found = find_local_ts_circuit_pocket_files(tid)
        assert found is not None and found[1] is not None, f"Curated pocket missing for {tid}"
        pocket_pdb, lig_sdf = found
        pocket_files_map[tid] = (pocket_pdb, lig_sdf)

        # Featurize & assert (N, 52) + 0 NaN
        pf = load_tourette_circuit_pocket(tid)
        assert pf.node_feats.shape[1] == POCKET_NODE_DIM
        assert bool(np.all(np.isfinite(pf.node_feats))) and bool(np.all(np.isfinite(pf.coords)))
        pocket_feats_map[tid] = pf

        h_audit = audit_pocket_hotspots(pocket_pdb, spec, ligand_sdf=lig_sdf, contact_cutoff=4.5)
        nat_mol = next(Chem.SDMolSupplier(str(lig_sdf), removeHs=False))
        nat_prot = protonate_mol_ph74(nat_mol, preserve_coords=True)
        nat_plif = profile_3d_interactions(pocket_pdb, nat_prot)

        plif_counts: dict[str, int] = {}
        for rec in nat_plif.interactions:
            plif_counts[rec.kind] = plif_counts.get(rec.kind, 0) + 1

        pocket_audits[tid] = {
            "pdb_id": spec.primary_pdb_id,
            "uniprot_id": spec.uniprot_id,
            "chembl_id": spec.chembl_id,
            "featurizer_shape": list(pf.node_feats.shape),
            "nan_count": int(np.isnan(pf.node_feats).sum()),
            "n_contact_atoms_4p5A": int(pf.contact_mask.sum()),
            "hotspot_audit": h_audit,
            "native_plif_counts": plif_counts,
            "native_top_contacts": [
                {
                    "residue": rec.protein_residue,
                    "kind": rec.kind,
                    "distance_A": round(float(rec.distance_angstrom), 2),
                }
                for rec in nat_plif.interactions[:12]
            ],
        }

        # Prepare receptor PDBQT once for fast reuse across all docking calls
        rec_pdbqt = rec_cache_dir / f"{spec.primary_pdb_id}_{tid}_receptor.pdbqt"
        if not rec_pdbqt.exists():
            prep = prepare_receptor_pdbqt(pocket_pdb, rec_pdbqt)
            assert prep is not None, f"Failed to prepare receptor PDBQT for {tid}"
        receptor_pdbqt_map[tid] = rec_pdbqt

        # Gate 1 Unbiased 3-Seed AutoDock Vina v1.2.7 Native Redocking
        anchors, kinds = TARGET_ANCHOR_CONFIG[tid]
        t_g1 = time.perf_counter()
        g1 = redock_native_complex(
            pocket_pdb,
            lig_sdf,
            seeds=(42, 137, 2026),
            canonical_anchors=anchors,
            expected_anchor_kinds=kinds,
            box_mode="tight_per_axis",
            vina_bin="bin/vina",
            exhaustiveness=args.exhaustiveness,
            num_modes=9,
            cpu=args.vina_cpu,
            num_confs=48,
            max_ring_confs=4,
        )
        gate1_results[tid] = {
            **g1.to_dict(),
            "elapsed_sec": round(time.perf_counter() - t_g1, 2),
        }
        print(
            f"  [Gate1 {tid:7s} {spec.primary_pdb_id}] "
            f"Top1_RMSD={g1.median_top1_rmsd:.2f} A | Best3_RMSD={g1.median_best3_rmsd:.2f} A | "
            f"Vina={g1.median_vina_affinity:.2f} kcal/mol | "
            f"PB={g1.top1_posebusters_pass_rate * 100:.0f}% | "
            f"PLIF={g1.mean_top1_plif_tanimoto:.3f} | "
            f"Anchor={g1.top1_anchor_recovery_rate * 100:.0f}% "
            f"({time.perf_counter() - t_g1:.1f}s)",
            flush=True,
        )

    # Also run Gate 1 on secondary native pentamer GABA-A comparator (6X3X Diazepam alpha1+/gamma2-)
    comp_pdb = (
        DEFAULT_TS_CIRCUIT_STRUCTURES_DIR / "clean_pockets" / "6X3X_GABRA1_COMP_DZP_pocket.pdb"
    )
    comp_sdf = DEFAULT_TS_CIRCUIT_STRUCTURES_DIR / "clean_pockets" / "6X3X_GABRA1_COMP_DZP_lig.sdf"
    if comp_pdb.exists() and comp_sdf.exists():
        t_c = time.perf_counter()
        c_anch, c_kinds = TARGET_ANCHOR_CONFIG["GABRA1_COMP"]
        g1_comp = redock_native_complex(
            comp_pdb,
            comp_sdf,
            seeds=(42, 137, 2026),
            canonical_anchors=c_anch,
            expected_anchor_kinds=c_kinds,
            box_mode="tight_per_axis",
            vina_bin="bin/vina",
            exhaustiveness=args.exhaustiveness,
            num_modes=9,
            cpu=args.vina_cpu,
            num_confs=48,
            max_ring_confs=4,
        )
        gate1_results["GABRA1_COMP_6X3X"] = {
            **g1_comp.to_dict(),
            "elapsed_sec": round(time.perf_counter() - t_c, 2),
        }
        print(
            f"  [Gate1 GABRA1_6X3X] Top1_RMSD={g1_comp.median_top1_rmsd:.2f} A | "
            f"Best3_RMSD={g1_comp.median_best3_rmsd:.2f} A | "
            f"Vina={g1_comp.median_vina_affinity:.2f} | "
            f"PB={g1_comp.top1_posebusters_pass_rate * 100:.0f}% "
            f"({time.perf_counter() - t_c:.1f}s)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Stage 2: Zero-Shot 3D-Conditioned De Novo Generation vs ZINC-250k
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed + 2026)
    gen_summaries: dict[str, Any] = {}
    ctrl_summaries: dict[str, Any] = {}
    docking_benchmark: dict[str, Any] = {}

    all_gen_dgs: list[float] = []
    all_ctrl_dgs: list[float] = []
    all_gen_readiness: list[float] = []
    all_ctrl_readiness: list[float] = []

    for t_idx, tid in enumerate(PRIMARY_TS_EXPANSION_IDS):
        spec = TOURETTE_CIRCUIT_EXPANSION_REGISTRY[tid]
        pocket_pdb, lig_sdf = pocket_files_map[tid]
        rec_pdbqt = receptor_pdbqt_map[tid]
        pf = pocket_feats_map[tid]
        anchors, kinds = TARGET_ANCHOR_CONFIG[tid]
        chembl_refs = chembl_by_target.get(tid, [])

        t_gen = time.perf_counter()
        p_enc, p_mask = _pocket_to_batch(pf, model, device)
        gen_smiles = sample_llm_molecules(
            model,
            p_enc,
            p_mask,
            num_samples=args.num_samples,
            max_new_tokens=120,
            strategy="min_p",
            temperature=0.85,
            min_p=0.08,
            top_p=0.92,
            top_k=30,
            cfg_scale=1.3,
            latent_temp=1.0,
            repetition_penalty=1.0,
            batch_size=min(64, args.num_samples),
            seed=args.seed + 1000 * t_idx,
        )
        ctrl_idx = rng.choice(len(zinc_pool), size=args.num_samples, replace=False)
        ctrl_smiles = [zinc_pool[int(i)] for i in ctrl_idx]

        gen_sum, gen_recs = _evaluate_candidate_pool(
            gen_smiles,
            tid,
            p_enc=p_enc,
            model=model,
            admet_suite=admet_suite,
            chembl_set=all_chembl_set,
            zinc_set=zinc_set,
            device=device,
        )
        ctrl_sum, ctrl_recs = _evaluate_candidate_pool(
            ctrl_smiles,
            tid,
            p_enc=p_enc,
            model=model,
            admet_suite=admet_suite,
            chembl_set=all_chembl_set,
            zinc_set=zinc_set,
            device=device,
        )
        gen_summaries[tid] = gen_sum
        ctrl_summaries[tid] = ctrl_sum
        print(
            f"  [Gen {tid:7s}] Valid={gen_sum['validity'] * 100:.1f}% "
            f"Uniq={gen_sum['uniqueness'] * 100:.1f}% "
            f"Nov={gen_sum['strict_novelty_vs_chembl_zinc'] * 100:.1f}% "
            f"PAINS_free={gen_sum['pains_free_rate'] * 100:.1f}% DCU={gen_sum['dcu_count']} "
            f"MPOv2={gen_sum['cns_mpo_v2_mean']:.2f} BBB={gen_sum['bbb_prob_mean']:.3f} "
            f"hERG={gen_sum['herg_prob_mean']:.3f} ({time.perf_counter() - t_gen:.1f}s)",
            flush=True,
        )

        # Select Top-5 MaxMin-diverse hygiene-passing leads for GenEuclidean & ZINC-250k
        top_gen = _select_top_k_maxmin_diverse(gen_recs, k=args.top_k, seed=args.seed + t_idx)
        top_ctrl = _select_top_k_maxmin_diverse(
            ctrl_recs, k=args.top_k, seed=args.seed + 100 + t_idx
        )

        # Dock Clinical Reference Drugs
        t_dock = time.perf_counter()
        clinical_docked: list[dict[str, Any]] = []
        for drug in spec.reference_drugs:
            dres = dock_ligand_to_pocket(
                drug.canonical_smiles,
                pocket_pdb,
                native_ligand_sdf=lig_sdf,
                box_mode="tight_per_axis",
                canonical_anchors=anchors,
                expected_anchor_kinds=kinds,
                vina_bin="bin/vina",
                exhaustiveness=args.exhaustiveness,
                num_modes=9,
                cpu=args.vina_cpu,
                seed=args.seed,
                num_confs=24,
                max_ring_confs=2,
                receptor_pdbqt_path=rec_pdbqt,
            )
            clinical_docked.append(
                _compact_pose_summary(
                    dres,
                    label=drug.name,
                    target_id=tid,
                    admet_suite=admet_suite,
                    chembl_refs=chembl_refs,
                )
            )

        # Dock Top-5 GenEuclidean Zero-Shot Leads & save 3D SDFs!
        gen_docked: list[dict[str, Any]] = []
        for rank_i, rec in enumerate(top_gen, start=1):
            dres = dock_ligand_to_pocket(
                rec["canonical_smiles"],
                pocket_pdb,
                native_ligand_sdf=lig_sdf,
                box_mode="tight_per_axis",
                canonical_anchors=anchors,
                expected_anchor_kinds=kinds,
                vina_bin="bin/vina",
                exhaustiveness=args.exhaustiveness,
                num_modes=9,
                cpu=args.vina_cpu,
                seed=args.seed + rank_i,
                num_confs=24,
                max_ring_confs=2,
                receptor_pdbqt_path=rec_pdbqt,
            )
            sdf_rel: str | None = None
            if dres.success and dres.preferred_mol is not None:
                sdf_dest = DOCKED_POSES_DIR / f"{tid}_geneuclidean_{rank_i}.sdf"
                dres.save_sdf(sdf_dest, prefer_anchor_pose=True)
                sdf_rel = str(sdf_dest)
            comp = _compact_pose_summary(
                dres,
                label=f"{tid}_Gen_{rank_i}",
                target_id=tid,
                admet_suite=admet_suite,
                chembl_refs=chembl_refs,
                pre_record=rec,
                sdf_rel_path=sdf_rel,
            )
            gen_docked.append(comp)
            if comp["vina_dg_kcal"] is not None:
                all_gen_dgs.append(float(comp["vina_dg_kcal"]))
                all_gen_readiness.append(float(comp["circuit_readiness_score"]))

        # Dock Top-5 ZINC-250k Random Control molecules
        ctrl_docked: list[dict[str, Any]] = []
        for rank_i, rec in enumerate(top_ctrl, start=1):
            dres = dock_ligand_to_pocket(
                rec["canonical_smiles"],
                pocket_pdb,
                native_ligand_sdf=lig_sdf,
                box_mode="tight_per_axis",
                canonical_anchors=anchors,
                expected_anchor_kinds=kinds,
                vina_bin="bin/vina",
                exhaustiveness=args.exhaustiveness,
                num_modes=9,
                cpu=args.vina_cpu,
                seed=args.seed + 50 + rank_i,
                num_confs=24,
                max_ring_confs=2,
                receptor_pdbqt_path=rec_pdbqt,
            )
            comp = _compact_pose_summary(
                dres,
                label=f"{tid}_ZINC_{rank_i}",
                target_id=tid,
                admet_suite=admet_suite,
                chembl_refs=chembl_refs,
                pre_record=rec,
            )
            ctrl_docked.append(comp)
            if comp["vina_dg_kcal"] is not None:
                all_ctrl_dgs.append(float(comp["vina_dg_kcal"]))
                all_ctrl_readiness.append(float(comp["circuit_readiness_score"]))

        # Sort GenEuclidean Top-5 by circuit_readiness_score descending
        gen_docked.sort(key=lambda x: -float(x["circuit_readiness_score"]))
        ctrl_docked.sort(key=lambda x: -float(x["circuit_readiness_score"]))

        g_dgs = [float(x["vina_dg_kcal"]) for x in gen_docked if x["vina_dg_kcal"] is not None]
        c_dgs = [float(x["vina_dg_kcal"]) for x in ctrl_docked if x["vina_dg_kcal"] is not None]
        r_dgs = [float(x["vina_dg_kcal"]) for x in clinical_docked if x["vina_dg_kcal"] is not None]

        docking_benchmark[tid] = {
            "primary_pdb_id": spec.primary_pdb_id,
            "elapsed_sec": round(time.perf_counter() - t_dock, 2),
            "summary": {
                "geneuclidean_mean_vina_dg": round(float(np.mean(g_dgs)), 3) if g_dgs else None,
                "geneuclidean_best_vina_dg": round(float(np.min(g_dgs)), 3) if g_dgs else None,
                "geneuclidean_pb_pass_rate": round(
                    sum(1 for x in gen_docked if x["posebusters_passed"]) / max(len(gen_docked), 1),
                    3,
                ),
                "geneuclidean_anchor_rate": round(
                    sum(1 for x in gen_docked if x["anchor_any_mode_engaged"])
                    / max(len(gen_docked), 1),
                    3,
                ),
                "geneuclidean_mean_readiness": round(
                    float(np.mean([x["circuit_readiness_score"] for x in gen_docked])), 4
                )
                if gen_docked
                else None,
                "zinc_ctrl_mean_vina_dg": round(float(np.mean(c_dgs)), 3) if c_dgs else None,
                "zinc_ctrl_best_vina_dg": round(float(np.min(c_dgs)), 3) if c_dgs else None,
                "zinc_ctrl_mean_readiness": round(
                    float(np.mean([x["circuit_readiness_score"] for x in ctrl_docked])), 4
                )
                if ctrl_docked
                else None,
                "clinical_ref_mean_vina_dg": round(float(np.mean(r_dgs)), 3) if r_dgs else None,
                "clinical_ref_best_vina_dg": round(float(np.min(r_dgs)), 3) if r_dgs else None,
                "delta_vina_gen_minus_zinc": (
                    round(float(np.mean(g_dgs) - np.mean(c_dgs)), 3) if (g_dgs and c_dgs) else None
                ),
            },
            "geneuclidean_top5": gen_docked,
            "zinc_control_top5": ctrl_docked,
            "clinical_references": clinical_docked,
        }
        b_sum = docking_benchmark[tid]["summary"]
        print(
            f"  [Dock {tid:7s}] Gen_mean={b_sum['geneuclidean_mean_vina_dg']} "
            f"(best={b_sum['geneuclidean_best_vina_dg']}) vs "
            f"ZINC_mean={b_sum['zinc_ctrl_mean_vina_dg']} | "
            f"Clin_best={b_sum['clinical_ref_best_vina_dg']} | "
            f"PB={b_sum['geneuclidean_pb_pass_rate'] * 100:.0f}% "
            f"Anchor={b_sum['geneuclidean_anchor_rate'] * 100:.0f}% "
            f"({time.perf_counter() - t_dock:.1f}s)",
            flush=True,
        )

    # Aggregate statistics across all 4 unseen GABA/Basal-Ganglia pockets
    mwu_dg_p: float | None = None
    mwu_readiness_p: float | None = None
    if len(all_gen_dgs) >= 4 and len(all_ctrl_dgs) >= 4:
        mwu_dg_p = float(mannwhitneyu(all_gen_dgs, all_ctrl_dgs, alternative="less").pvalue)
        mwu_readiness_p = float(
            mannwhitneyu(all_gen_readiness, all_ctrl_readiness, alternative="greater").pvalue
        )

    g1_primary = [gate1_results[tid] for tid in PRIMARY_TS_EXPANSION_IDS]
    aggregate = {
        "gate1_4target_median_top1_rmsd_A": round(
            float(np.median([r["median_top1_rmsd"] for r in g1_primary])), 3
        ),
        "gate1_4target_median_best3_rmsd_A": round(
            float(np.median([r["median_best3_rmsd"] for r in g1_primary])), 3
        ),
        "gate1_4target_median_best_all_rmsd_A": round(
            float(np.median([r["median_best_all_rmsd"] for r in g1_primary])), 3
        ),
        "gate1_4target_pass_lt2A_top1_count": sum(
            1 for r in g1_primary if float(r["median_top1_rmsd"]) <= 2.0
        ),
        "gate1_4target_pass_lt2A_best3_count": sum(
            1 for r in g1_primary if float(r["median_best3_rmsd"]) <= 2.0
        ),
        "gate1_4target_mean_posebusters_rate": round(
            float(np.mean([r["top1_posebusters_pass_rate"] for r in g1_primary])), 4
        ),
        "gate1_4target_mean_plif_tanimoto": round(
            float(np.mean([r["mean_top1_plif_tanimoto"] for r in g1_primary])), 4
        ),
        "gate1_4target_mean_anchor_recovery": round(
            float(np.mean([r["top1_anchor_recovery_rate"] for r in g1_primary])), 4
        ),
        "denovo_4target_geneuclidean_mean_vina_dg": round(float(np.mean(all_gen_dgs)), 3)
        if all_gen_dgs
        else None,
        "denovo_4target_zinc_ctrl_mean_vina_dg": round(float(np.mean(all_ctrl_dgs)), 3)
        if all_ctrl_dgs
        else None,
        "denovo_4target_delta_vina_gen_minus_zinc": (
            round(float(np.mean(all_gen_dgs) - np.mean(all_ctrl_dgs)), 3)
            if (all_gen_dgs and all_ctrl_dgs)
            else None
        ),
        "denovo_4target_mwu_vina_p_one_sided": (
            round(mwu_dg_p, 6) if mwu_dg_p is not None else None
        ),
        "denovo_4target_geneuclidean_mean_readiness": round(float(np.mean(all_gen_readiness)), 4)
        if all_gen_readiness
        else None,
        "denovo_4target_zinc_ctrl_mean_readiness": round(float(np.mean(all_ctrl_readiness)), 4)
        if all_ctrl_readiness
        else None,
        "denovo_4target_mwu_readiness_p_one_sided": (
            round(mwu_readiness_p, 6) if mwu_readiness_p is not None else None
        ),
        "denovo_4target_mean_mpo_v2_gen": round(
            float(np.mean([gen_summaries[t]["cns_mpo_v2_mean"] for t in PRIMARY_TS_EXPANSION_IDS])),
            3,
        ),
        "denovo_4target_mean_mpo_v2_zinc": round(
            float(
                np.mean([ctrl_summaries[t]["cns_mpo_v2_mean"] for t in PRIMARY_TS_EXPANSION_IDS])
            ),
            3,
        ),
        "denovo_4target_total_dcu_gen": sum(
            int(gen_summaries[t]["dcu_count"]) for t in PRIMARY_TS_EXPANSION_IDS
        ),
        "denovo_4target_mean_pains_free_gen": round(
            float(np.mean([gen_summaries[t]["pains_free_rate"] for t in PRIMARY_TS_EXPANSION_IDS])),
            4,
        ),
        "denovo_4target_mean_reagent_free_gen": round(
            float(
                np.mean(
                    [
                        gen_summaries[t]["reagent_quarantine_free_rate"]
                        for t in PRIMARY_TS_EXPANSION_IDS
                    ]
                )
            ),
            4,
        ),
        "denovo_4target_mean_bbb_gen": round(
            float(np.mean([gen_summaries[t]["bbb_prob_mean"] for t in PRIMARY_TS_EXPANSION_IDS])),
            4,
        ),
        "denovo_4target_mean_herg_gen": round(
            float(np.mean([gen_summaries[t]["herg_prob_mean"] for t in PRIMARY_TS_EXPANSION_IDS])),
            4,
        ),
        "denovo_4target_mean_bbb_zinc": round(
            float(np.mean([ctrl_summaries[t]["bbb_prob_mean"] for t in PRIMARY_TS_EXPANSION_IDS])),
            4,
        ),
        "denovo_4target_mean_herg_zinc": round(
            float(np.mean([ctrl_summaries[t]["herg_prob_mean"] for t in PRIMARY_TS_EXPANSION_IDS])),
            4,
        ),
    }

    payload = {
        "metadata": {
            "generated_utc": datetime.now(UTC).isoformat(),
            "model_checkpoint": str(V31_DPO_ADAPTER_CKPT),
            "admet_suite_path": str(DEFAULT_V4_ADMET_MODEL_PATH),
            "transfer_regime": (
                "Zero-Shot 3D Transfer to 4 Unseen GABAergic & Basal Ganglia Pockets"
            ),
            "num_samples_per_pocket": args.num_samples,
            "top_k_docked_per_arm": args.top_k,
            "vina_exhaustiveness": args.exhaustiveness,
            "seed": args.seed,
            "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
            "disclaimer": (
                "All generated ligands and docking poses are strictly in silico computational "
                "research hypotheses and are NOT clinically validated medicines or medical advice."
            ),
        },
        "cstc_circuit_audit_13targets": {
            tid: entry.to_dict() for tid, entry in TOURETTE_CSTC_CIRCUIT_AUDIT.items()
        },
        "excluded_or_caveated_candidates_audit": {
            "GABRA6": {
                "uniprot_id": "Q16445",
                "chembl_id": "CHEMBL2579",
                "rcsb_human_pdb_count": 0,
                "chembl37_pchembl_all": 13,
                "chembl37_pchembl_ge6": 5,
                "verdict": (
                    "Preclinically compelling (DK-I-56-1 / LAU-176 alpha6-PAM Tourette models), "
                    "but ZERO experimental human PDB structures exist; requires homology modeling "
                    "from human alpha5/gamma2 (8BHG) or alpha1/beta2/gamma2 (6X3X)."
                ),
            },
            "ABAT": {
                "uniprot_id": "P80404",
                "chembl_id": "CHEMBL2044",
                "rcsb_human_pdb_count": 0,
                "rcsb_pig_pdb_count": 9,
                "pig_uniprot_id": "P80147",
                "chembl37_pchembl_all": 17,
                "verdict": (
                    "SPECIES TRAP: 0 Homo sapiens PDB structures exist; all 9 RCSB entries "
                    "(4Y0H, 1OHV, etc.) are Sus scrofa (pig P80147). Moreover, irreversible "
                    "vigabatrin carries black-box irreversible retinal/visual-field toxicity."
                ),
            },
            "GRIN2B": {
                "uniprot_id": "Q13224",
                "chembl_id": "CHEMBL1904",
                "chembl37_pchembl_all": 2255,
                "chembl37_pchembl_ge6": 1770,
                "verdict": (
                    "SPECIES-CHIMERA TRAP: High-res ifenprodil ATD complex 5EWJ (2.77 A) is a "
                    "Xenopus laevis GluN1 (A0A1L8F5J9) + Homo sapiens GluN2B chimera, 4PE5 is "
                    "Rattus norvegicus, and full-human 7EU8 is 4.07 A."
                ),
            },
            "GABBR1_GABBR2": {
                "uniprot_ids": ["Q9UBS5", "O75899"],
                "chembl_ids": ["CHEMBL2064", "CHEMBL2111463"],
                "primary_pdb_id": "7C7Q",
                "resolution_A": 3.0,
                "chembl37_pchembl_all_complex": 81,
                "verdict": (
                    "Valid human cryo-EM (7C7Q 3.0 A active baclofen + rac-BHFF PAM; clinical "
                    "pediatric TS crossover RCT evidence for baclofen), but sparse human ChEMBL "
                    "bioactivity (81 complex / 8 GABBR1 records) vs GAT1/GABRA5/ADORA2A/GRM5."
                ),
            },
            "CNR1": {
                "uniprot_id": "P21554",
                "chembl_id": "CHEMBL218",
                "primary_pdb_id": "5XRA",
                "resolution_A": 2.8,
                "chembl37_pchembl_all": 8681,
                "verdict": (
                    "Rich human structural & ChEMBL data (5XRA 2.8 A active agonist, 6N4B 3.0 A "
                    "Gi-coupled; nabiximols/THC clinical TS signals), deprioritized behind MGLL "
                    "already in atlas due to direct CB1 psychoactivity/tolerance liability."
                ),
            },
            "GRM4": {
                "uniprot_id": "Q14833",
                "chembl_id": "CHEMBL2736",
                "primary_pdb_id": "7E9H",
                "resolution_A": 4.0,
                "chembl37_pchembl_all": 1766,
                "verdict": (
                    "Mechanistically relevant presynaptic GPe indirect-pathway brake, but only "
                    "low-resolution 4.0 A apo/phosphoserine human cryo-EM (7E9H, no 7TM PAM)."
                ),
            },
        },
        "chembl37_expansion_summary": chembl_summary,
        "pocket_featurization_and_hotspots": pocket_audits,
        "gate1_native_redocking": gate1_results,
        "denovo_generation_metrics": {
            "geneuclidean_v31_dpo_zeroshot": gen_summaries,
            "zinc250k_random_control": ctrl_summaries,
        },
        "heldout_vina_docking_and_plif": docking_benchmark,
        "aggregate_summary": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024.0
    print(
        f"[ts-circuit-study] DONE in {time.perf_counter() - t_start:.1f}s! "
        f"Saved {args.output} ({size_kb:.1f} KB).",
        flush=True,
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
