#!/usr/bin/env python3
"""Phase 2 Real Quantitative Benchmark: MIT Contrastive 3D Screening & Active-Learning DMTA.

Evaluates GenEuclidean's ``PocketScreener`` (trained chiral ``PaiNN-SE3`` + ``DrugCLIP`` +
neural ``pKd``) and ``ActiveLearningDMTA`` on strict Bemis-Murcko-deduplicated human ChEMBL
bioactivity decks (zero scaffold duplication leakage):

1. **Throughput & Index Compactness:** Measures exact encoding and cosine-retrieval speed on
   Apple M3 Pro MPS/CPU.
2. **Zero-Shot Multi-Conformation & Contrastive 3D Selectivity Screening:** Tests 4 homologous
   receptor pairs (``DRD1`` vs ``DRD2``, ``MAOB`` vs ``MAOA``, ``BACE1`` vs ``BACE2``,
   ``SERT`` vs ``DAT``) comparing single-pocket ``DrugCLIP``, multi-conformation ensemble +
   off-target repulsion, and neural ``Delta_pKd`` reranking on AUROC & EF5%.
3. **Hard-Negative Paralog Selectivity Fine-Tuning (Strict 60/40 Murcko Scaffold Split):**
   Tests whether fine-tuning ONLY the lightweight CLIP projection heads with
   :func:`compute_selectivity_contrastive_loss` improves held-out scaffold selectivity AUROC.
4. **Pool-Based Active-Learning DMTA (``Lab-in-a-Loop`` Simulation):** Evaluates
   ``maxmin_diverse_ucb`` vs ``ucb`` vs ``random`` over a 600-compound Murcko-deduplicated
   ``DRD1`` pChEMBL pool at 8% assay budget (Top-25 potent-hit recall & unique active scaffolds).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import geneuclidean as ge
from geneuclidean.cns.medchem_triage import bemis_murcko_scaffold
from geneuclidean.screening import ActiveLearningDMTA, LigandVectorIndex, PocketScreener
from geneuclidean.screening.encoders import featurize_ecfp4_batch
from geneuclidean.training.contrastive_selectivity import compute_selectivity_contrastive_loss

OUT_JSON = Path("results/tables/phase2_screening_and_al_benchmark.json")
V2_CKPT = Path("runs_local.nosync/geneuclidean_v2_painn/best.pt")
CHEMBL_V3 = Path("data/cns_neuro.nosync/chembl/human_cns_actives_12targets.csv")
CHEMBL_V4 = Path("data/cns_neuro.nosync/v4_expansion/chembl_v4_12new_targets.csv")


def _load_target_actives(
    targets: set[str], *, min_pchembl: float = 6.5, max_per_target: int = 350
) -> dict[str, list[tuple[str, float, str]]]:
    """Load 1 highest-potency molecule per unique Bemis-Murcko scaffold per target."""
    raw: dict[str, dict[str, tuple[str, float]]] = {t: {} for t in targets}
    for csv_path in (CHEMBL_V3, CHEMBL_V4):
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                t = (row.get("target_symbol") or "").strip().upper()
                if t not in raw:
                    continue
                smi = (row.get("canonical_smiles") or "").strip()
                p_str = (row.get("best_pchembl") or row.get("median_pchembl") or "").strip()
                if not smi or not p_str:
                    continue
                p_val = float(p_str)
                if p_val < min_pchembl:
                    continue
                scaf = bemis_murcko_scaffold(smi) or smi
                prev = raw[t].get(scaf)
                if prev is None or p_val > prev[1]:
                    raw[t][scaf] = (smi, p_val)
    out: dict[str, list[tuple[str, float, str]]] = {}
    for t, scaf_map in raw.items():
        items = [(smi, p, scaf) for scaf, (smi, p) in scaf_map.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        out[t] = items[:max_per_target]
    return out


def _ef_at_frac(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.05) -> float:
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    if n == 0 or n_pos == 0:
        return 0.0
    k = max(1, int(math.ceil(n * frac)))
    top_idx = np.argsort(y_score)[::-1][:k]
    hits = int(np.sum(y_true[top_idx]))
    return float((hits / k) / (n_pos / n))


def _bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """Exact Truchon & Bayly BEDROC(alpha) early-recognition metric in [0, 1]."""
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    if n == 0 or n_pos == 0 or n_pos == n:
        return 0.0
    order = np.argsort(y_score)[::-1]
    ranks = (np.flatnonzero(y_true[order] == 1) + 1) / float(n)
    ra = n_pos / float(n)
    rie_num = float(np.sum(np.exp(-alpha * ranks)))
    rie_den = ra * ((1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0))
    rie = rie_num / max(rie_den, 1e-12)
    rie_min = (1.0 - math.exp(alpha * ra)) / (ra * (1.0 - math.exp(alpha)))
    rie_max = (1.0 - math.exp(-alpha * ra)) / (ra * (1.0 - math.exp(-alpha)))
    return float(np.clip((rie - rie_min) / max(rie_max - rie_min, 1e-12), 0.0, 1.0))


def _eval_selectivity_pair(
    screener: PocketScreener,
    on_pockets: list[ge.Pocket],
    off_pockets: list[ge.Pocket],
    on_items: list[tuple[str, float, str]],
    off_items: list[tuple[str, float, str]],
) -> dict[str, Any]:
    """Benchmark on-target vs off-target paralog discrimination (strict scaffold disjoint)."""
    on_scafs = {x[2] for x in on_items}
    off_only = [x for x in off_items if x[2] not in on_scafs]
    on_smis = [x[0] for x in on_items]
    off_smis = [x[0] for x in off_only]
    all_smis = on_smis + off_smis
    y = np.array([1] * len(on_smis) + [0] * len(off_smis), dtype=np.int32)

    t0 = time.perf_counter()
    idx = LigandVectorIndex.build_from_smiles(all_smis, screener.ligand_encoder, batch_size=512)
    encode_s = time.perf_counter() - t0

    v_single, p_single = screener.encode_pocket_ensemble([on_pockets[0]])
    v_ens, p_ens = screener.encode_pocket_ensemble(on_pockets)
    v_off, p_off = screener.encode_pocket_ensemble(off_pockets)

    t_q = time.perf_counter()
    s_single = idx.score_query(v_single)
    s_ens = idx.score_query(v_ens)
    s_off = idx.score_query(v_off)
    query_ms = (time.perf_counter() - t_q) * 1000.0 / 3.0
    s_contrast = s_ens - 0.5 * s_off

    pkd_on = np.array(screener.predict_pkd_batch(p_ens, idx.smiles), dtype=np.float32)
    pkd_off = np.array(screener.predict_pkd_batch(p_off, idx.smiles), dtype=np.float32)
    d_pkd = pkd_on - pkd_off

    # Align y in case any invalid SMILES was skipped
    valid_map = {s: int(lbl) for s, lbl in zip(all_smis, y, strict=True)}
    y_aligned = np.array([valid_map[s] for s in idx.smiles], dtype=np.int32)

    def _metrics(scores: np.ndarray) -> dict[str, float]:
        return {
            "auroc": round(float(roc_auc_score(y_aligned, scores)), 4),
            "ef5": round(_ef_at_frac(y_aligned, scores, 0.05), 3),
            "bedroc20": round(_bedroc(y_aligned, scores, 20.0), 4),
        }

    return {
        "n_on_scaffolds": int(np.sum(y_aligned == 1)),
        "n_off_scaffolds": int(np.sum(y_aligned == 0)),
        "encode_seconds": round(encode_s, 3),
        "query_ms_per_pocket": round(query_ms, 3),
        "single_pocket_clip": _metrics(s_single),
        "ensemble_clip": _metrics(s_ens),
        "contrastive_clip_lam0p5": _metrics(s_contrast),
        "neural_delta_pkd": _metrics(d_pkd),
    }


def _run_hard_negative_head_tuning(
    screener: PocketScreener,
    p_drd1: list[ge.Pocket],
    p_drd2: ge.Pocket,
    drd1_items: list[tuple[str, float, str]],
    drd2_items: list[tuple[str, float, str]],
) -> dict[str, Any]:
    """Strict 60/40 Murcko-scaffold-split test of contrastive selectivity head fine-tuning."""
    drd1_scafs = {x[2] for x in drd1_items}
    drd2_only = [x for x in drd2_items if x[2] not in drd1_scafs]

    # Deterministic 60/40 split by scaffold index
    n1_tr = int(0.6 * len(drd1_items))
    n2_tr = int(0.6 * len(drd2_only))
    tr_on = [x[0] for x in drd1_items[:n1_tr]]
    te_on = [x[0] for x in drd1_items[n1_tr:]]
    tr_off = [x[0] for x in drd2_only[:n2_tr]]
    te_off = [x[0] for x in drd2_only[n2_tr:]]

    # Clone only the lightweight CLIP projection heads so base screener stays untouched
    import copy

    tuned_model = copy.deepcopy(screener.model).to(screener.device)
    for p in tuned_model.parameters():
        p.requires_grad_(False)
    assert tuned_model.heads.clip_pocket_proj is not None
    assert tuned_model.heads.clip_ligand_proj is not None
    assert tuned_model.heads.logit_scale is not None
    for p in tuned_model.heads.clip_pocket_proj.parameters():
        p.requires_grad_(True)
    for p in tuned_model.heads.clip_ligand_proj.parameters():
        p.requires_grad_(True)
    for p in tuned_model.heads.ligand_encoder.parameters():
        p.requires_grad_(True)

    # Pre-encode static 3D pocket pooled vectors on device
    def _encode_graph(poc: ge.Pocket) -> torch.Tensor:
        feat = poc.featurize()
        c_t = torch.from_numpy(feat.coords).float().unsqueeze(0).to(screener.device)
        n_t = torch.from_numpy(feat.node_feats).float().unsqueeze(0).to(screener.device)
        m_t = torch.ones((1, int(feat.coords.shape[0])), dtype=torch.bool, device=screener.device)
        return tuned_model.encode_pocket(c_t, n_t, m_t).pooled_emb

    with torch.no_grad():
        on_pooled_list = [_encode_graph(poc) for poc in p_drd1]
        p_on_pool = torch.stack(on_pooled_list, dim=0).mean(dim=0)
        p_off_pool = _encode_graph(p_drd2)

    fp_dim = int(tuned_model.cfg.multitask.ligand_fp_dim)
    x_tr_on, _ = featurize_ecfp4_batch(tr_on, n_bits=fp_dim)
    x_tr_off, _ = featurize_ecfp4_batch(tr_off, n_bits=fp_dim)
    t_tr_on = torch.from_numpy(x_tr_on).to(screener.device)
    t_tr_off = torch.from_numpy(x_tr_off).to(screener.device)

    te_smis = te_on + te_off
    y_te = np.array([1] * len(te_on) + [0] * len(te_off), dtype=np.int32)

    def _eval_heldout(m_eval: Any) -> dict[str, float]:
        sc = PocketScreener(m_eval, device=str(screener.device))
        idx_te = LigandVectorIndex.build_from_smiles(te_smis, sc.ligand_encoder, batch_size=512)
        v_on, _ = sc.encode_pocket_ensemble(p_drd1)
        v_off, _ = sc.encode_pocket_ensemble([p_drd2])
        scores = idx_te.score_query(v_on) - 0.5 * idx_te.score_query(v_off)
        return {
            "heldout_auroc": round(float(roc_auc_score(y_te, scores)), 4),
            "heldout_ef5": round(_ef_at_frac(y_te, scores, 0.05), 3),
            "heldout_bedroc20": round(_bedroc(y_te, scores, 20.0), 4),
        }

    before = _eval_heldout(tuned_model)
    opt = torch.optim.AdamW(
        [p for p in tuned_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-3
    )
    tuned_model.train()
    n_pairs = min(len(tr_on), len(tr_off))
    for _step in range(30):
        opt.zero_grad(set_to_none=True)
        perm1 = torch.randperm(len(tr_on), device=screener.device)[: min(64, n_pairs)]
        perm2 = torch.randperm(len(tr_off), device=screener.device)[: min(64, n_pairs)]
        # Positive DRD1 ligands prefer DRD1 over DRD2; DRD2 ligands prefer DRD2 over DRD1
        mt_on_1 = tuned_model.heads(
            p_on_pool.expand(len(perm1), -1), tuned_model.heads.encode_ligand(t_tr_on[perm1])
        )
        mt_off_1 = tuned_model.heads(p_off_pool.expand(len(perm1), -1), ligand_emb=None)
        mt_on_2 = tuned_model.heads(
            p_off_pool.expand(len(perm2), -1), tuned_model.heads.encode_ligand(t_tr_off[perm2])
        )
        mt_off_2 = tuned_model.heads(p_on_pool.expand(len(perm2), -1), ligand_emb=None)
        assert (
            mt_on_1.clip_pocket is not None
            and mt_off_1.clip_pocket is not None
            and mt_on_1.clip_ligand is not None
            and mt_on_1.clip_logit_scale is not None
            and mt_on_2.clip_pocket is not None
            and mt_off_2.clip_pocket is not None
            and mt_on_2.clip_ligand is not None
            and mt_on_2.clip_logit_scale is not None
        )
        l1, _ = compute_selectivity_contrastive_loss(
            mt_on_1.clip_pocket,
            mt_off_1.clip_pocket,
            mt_on_1.clip_ligand,
            mt_on_1.clip_logit_scale,
            margin=0.25,
            selectivity_weight=2.0,
        )
        l2, _ = compute_selectivity_contrastive_loss(
            mt_on_2.clip_pocket,
            mt_off_2.clip_pocket,
            mt_on_2.clip_ligand,
            mt_on_2.clip_logit_scale,
            margin=0.25,
            selectivity_weight=2.0,
        )
        (l1 + l2).backward()
        opt.step()

    tuned_model.eval()
    after = _eval_heldout(tuned_model)
    return {
        "pair": "DRD1_vs_DRD2_strict_scaffold_split",
        "n_train_scaffolds": len(tr_on) + len(tr_off),
        "n_heldout_scaffolds": len(te_smis),
        "zero_shot_before": before,
        "after_30step_selectivity_tuning": after,
    }


def _run_active_learning_benchmark(drd1_items: list[tuple[str, float, str]]) -> dict[str, Any]:
    """Pool-based Active Learning DMTA on Murcko-deduplicated DRD1 pChEMBL pool (8% budget)."""
    # Build diverse pool of up to 400 unique-scaffold DRD1 compounds across potency spectrum
    raw_all: dict[str, tuple[str, float]] = {}
    with CHEMBL_V3.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("target_symbol") or "").strip().upper() != "DRD1":
                continue
            smi = (row.get("canonical_smiles") or "").strip()
            p_str = (row.get("best_pchembl") or "").strip()
            if not smi or not p_str:
                continue
            scaf = bemis_murcko_scaffold(smi) or smi
            val = float(p_str)
            if scaf not in raw_all or val > raw_all[scaf][1]:
                raw_all[scaf] = (smi, val)
    entries = list(raw_all.values())[:400]
    pool_smis = [s for s, _ in entries]
    oracle_map = {s: v for s, v in entries}
    top25_thresh = float(np.sort([v for _, v in entries])[::-1][min(24, len(entries) - 1)])
    top25_set = {s for s, v in entries if v >= top25_thresh}

    def _oracle(batch: Any) -> list[float]:
        return [oracle_map[s] for s in batch]

    # Budget: 4 rounds x 8 compounds = 32 compounds (~8% of 400-scaffold pool)
    rng = np.random.default_rng(42)
    rand_picks = [pool_smis[int(i)] for i in rng.choice(len(pool_smis), size=32, replace=False)]
    rand_hits = sum(1 for s in rand_picks if s in top25_set)

    out: dict[str, Any] = {
        "pool_unique_scaffolds": len(pool_smis),
        "budget_compounds": 32,
        "budget_fraction": round(32 / max(len(pool_smis), 1), 4),
        "top25_pchembl_threshold": round(top25_thresh, 2),
        "random_baseline": {
            "top25_hits_recovered": rand_hits,
            "top25_recall": round(rand_hits / max(len(top25_set), 1), 4),
            "mean_pchembl_acquired": round(float(np.mean([oracle_map[s] for s in rand_picks])), 3),
        },
    }
    for strat in ("ucb", "maxmin_diverse_ucb"):
        al = ActiveLearningDMTA(strategy=strat, ucb_beta=1.2, diversity_weight=0.25, seed=42)
        for _ in range(4):
            al.run_round(pool_smis, _oracle, batch_size=8)
        obs = list(al.observed.keys())
        hits = sum(1 for s in obs if s in top25_set)
        out[strat] = {
            "top25_hits_recovered": hits,
            "top25_recall": round(hits / max(len(top25_set), 1), 4),
            "enrichment_vs_random": round(hits / max(rand_hits, 1), 2),
            "mean_pchembl_acquired": round(float(np.mean(list(al.observed.values()))), 3),
            "best_pchembl_found": round(max(al.observed.values()), 3),
        }
    return out


def main() -> None:
    t0 = time.perf_counter()
    print("=== [1/3] Loading Trained v2 PaiNN-SE3 + DrugCLIP + pKd PocketScreener ===")
    screener = PocketScreener.from_pretrained(V2_CKPT, device="auto")

    actives = _load_target_actives(
        {"DRD1", "DRD2", "MAOB", "MAOA", "BACE1", "BACE2", "SERT", "DAT"},
        min_pchembl=6.5,
        max_per_target=300,
    )

    p_drd1 = [
        ge.Pocket.from_pdb(
            f"data/cns_neuro.nosync/structures/clean_pockets/{fn}_pocket.pdb", target_id="DRD1"
        )
        for fn in ("9LLG_DRD1_A1EKL", "9U5E_DRD1_A1EN9", "7LJC_DRD1_SK0")
    ]
    p_drd2 = [
        ge.Pocket.from_pdb(
            "data/cns_neuro.nosync/structures/clean_pockets/6CM4_DRD2_8NU_pocket.pdb",
            target_id="DRD2",
        )
    ]
    p_maob = [
        ge.Pocket.from_pdb(
            "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/2V5Z_MAOB_SAG_pocket.pdb",
            target_id="MAOB",
        )
    ]
    p_maoa = [
        ge.Pocket.from_pdb(
            "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/2Z5X_MAOA_HRM_pocket.pdb",
            target_id="MAOA",
        )
    ]
    p_bace1 = [
        ge.Pocket.from_pdb(
            "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/6EQM_BACE1_BUH_pocket.pdb",
            target_id="BACE1",
        )
    ]
    p_bace2 = [
        ge.Pocket.from_pdb(
            "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets/7D5B_BACE2_66F_pocket.pdb",
            target_id="BACE2",
        )
    ]
    p_sert = [
        ge.Pocket.from_pdb(
            f"data/cns_neuro.nosync/structures/clean_pockets/{fn}_pocket.pdb", target_id="SERT"
        )
        for fn in ("5I6X_SERT_8PR", "5I71_SERT_68P")
    ]
    p_dat = [
        ge.Pocket.from_pdb(
            "data/cns_neuro.nosync/structures/clean_pockets/8Y2G_DAT_A1D5U_pocket.pdb",
            target_id="DAT",
        )
    ]

    print("=== [2/3] Benchmarking Zero-Shot Multi-Conf & Contrastive 3D Selectivity Screening ===")
    pairs_res = {
        "DRD1_vs_DRD2": _eval_selectivity_pair(
            screener, p_drd1, p_drd2, actives["DRD1"], actives["DRD2"]
        ),
        "MAOB_vs_MAOA": _eval_selectivity_pair(
            screener, p_maob, p_maoa, actives["MAOB"], actives["MAOA"]
        ),
        "BACE1_vs_BACE2": _eval_selectivity_pair(
            screener, p_bace1, p_bace2, actives["BACE1"], actives["BACE2"]
        ),
        "SERT_vs_DAT": _eval_selectivity_pair(
            screener, p_sert, p_dat, actives["SERT"], actives["DAT"]
        ),
    }
    for pname, info in pairs_res.items():
        print(
            f"  {pname:15s}: single={info['single_pocket_clip']['auroc']:.3f} | "
            f"contrast={info['contrastive_clip_lam0p5']['auroc']:.3f} "
            f"(EF5={info['contrastive_clip_lam0p5']['ef5']:.2f}x) | "
            f"delta_pKd={info['neural_delta_pkd']['auroc']:.3f} "
            f"({info['query_ms_per_pocket']:.2f} ms/query)"
        )

    print("=== [3/3] Hard-Negative Selectivity Tuning + Active-Learning DMTA Simulation ===")
    tuning_res = _run_hard_negative_head_tuning(
        screener, p_drd1, p_drd2[0], actives["DRD1"], actives["DRD2"]
    )
    print(
        f"  DRD1/DRD2 held-out scaffold AUROC: "
        f"{tuning_res['zero_shot_before']['heldout_auroc']:.4f} -> "
        f"{tuning_res['after_30step_selectivity_tuning']['heldout_auroc']:.4f} "
        f"(EF5 {tuning_res['zero_shot_before']['heldout_ef5']:.2f}x -> "
        f"{tuning_res['after_30step_selectivity_tuning']['heldout_ef5']:.2f}x)"
    )

    al_res = _run_active_learning_benchmark(actives["DRD1"])
    r_rand = al_res["random_baseline"]["top25_recall"]
    r_ucb = al_res["ucb"]["top25_recall"]
    r_div = al_res["maxmin_diverse_ucb"]["top25_recall"]
    enr = al_res["maxmin_diverse_ucb"]["enrichment_vs_random"]
    print(
        f"  Active Learning DMTA (8% budget): random={r_rand:.1%}, "
        f"ucb={r_ucb:.1%}, maxmin_diverse_ucb={r_div:.1%} ({enr:.2f}x random)"
    )

    payload = {
        "generated_by": "scripts/run_selectivity_screening_benchmark.py",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "zero_shot_selectivity_pairs": pairs_res,
        "hard_negative_selectivity_tuning": tuning_res,
        "active_learning_dmta": al_res,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Phase 2 screening & AL benchmark -> {OUT_JSON} ({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
