#!/usr/bin/env python3
"""5-Tier Zero-Leakage CASF-2016 Benchmark Driver for GenEuclidean v3.0 LLM Models.

Runs the exact same 5-tier evaluation suite as ``scripts/run_benchmark.py`` so v3.0 models
are directly comparable to the 2021 ETH Zurich thesis baseline and all GenEuclidean v2.0
models, AND outputs ``samples_v3_<name>.json`` in the exact schema expected by
``scripts/run_benchmark.py --mode docking`` (zero changes required!):

  - Tier 1: 2D quality, uniqueness, novelty, IntDiv1/2, Murcko scaffold diversity, QED,
    SA, Lipinski, PAINS, FFD, W1, JSD (both raw and MPO+MaxMin Top-10 reranked).
  - Tier 2: Causal pocket specificity (Matched vs. Shuffled derangement vs. Null pocket)
    with True ECFP4 Tanimoto, 2021-thesis Count Dice, Wilcoxon signed-rank, Mann-Whitney U,
    Cliff's delta, 1,000x bootstrap 95% CIs, and MW/LogP/TPSA cross-pocket Pearson r.
  - Tier 3: ``<4.5 A`` Perceiver cross-attention contact atom enrichment + residue heatmaps.
  - Tier 4: Full 285-complex CASF-2016 ``pKd`` affinity & DrugCLIP retrieval + pocket
    embeddings (``.npz``).
  - Optional ``--cfg-sweep``: CFG scale Pareto sweep (``w = 1.0, 1.3, 1.6, 2.0``) +
    Hit-to-Lead CVAE seed refinement.
  - Unified comparison table: ``results/tables/model_comparison_v3.{csv,md}`` combining
    2021 Thesis + v2.0 + v3.0 models side-by-side.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything, select_device
from geneuclidean.data.featurizer import LIGAND_FP_DIM
from geneuclidean.evaluation import (
    dice_similarity,
    evaluate_affinity_metrics,
    evaluate_attention_enrichment,
    evaluate_clip_retrieval,
    evaluate_generation_metrics,
    evaluate_pocket_specificity,
    get_morgan_bit_fp,
    get_morgan_count_fp,
    parse_mol,
    tanimoto_similarity,
)
from geneuclidean.models.hf_backbones import load_hf_backbone
from geneuclidean.models.llm_bridge import PocketConditionedLLM
from geneuclidean.sampling.llm_sampler import sample_llm_molecules, sample_llm_with_attention
from geneuclidean.sampling.rerank import rerank_and_select

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# Exact 2021 ETH Zurich thesis baseline (Report_denovo.pdf, Tables 4.3, 4.4 & Fig. A.1)
THESIS_2021_BASELINE: dict[str, Any] = {
    "model": "2021 Thesis Baseline (e3nn+LSTM)",
    "run_id": "thesis_2021_baseline",
    "num_params": 632_292,
    "validity": 0.65,
    "uniqueness": 0.70,
    "novelty": 0.65,
    "int_div_1": float("nan"),
    "int_div_2": float("nan"),
    "murcko_scaffold_diversity": float("nan"),
    "QED": 0.45,
    "SA": 4.00,
    "SA_norm": (10.0 - 4.00) / 9.0,
    "lipinski_pass_rate": float("nan"),
    "pains_free_rate": float("nan"),
    "True Tanimoto": 0.17,
    "Count Dice": 0.29,
    "shuffled_tanimoto": 0.10,
    "cliffs_delta": float("nan"),
    "attn_enrichment": float("nan"),
    "pKd Pearson r": 0.5025,
    "pKd RMSE": float("nan"),
    "DrugCLIP R@1": float("nan"),
    "DrugCLIP AUC": float("nan"),
}

_V2_MODEL_ORDER: tuple[str, ...] = (
    "e3nn_lstm_v15",
    "painn_pocketformer_scratch",
    "painn_pocketformer",
    "geneuclidean_v2_schnet",
    "geneuclidean_v2_egnn",
    "geneuclidean_v2_e3nn",
    "geneuclidean_v2_selfies",
    "geneuclidean_v2_painn",
    "geneuclidean_v2_painn_lp",
    "geneuclidean_v2_painn_dpo",
)

_V3_MODEL_ORDER: tuple[str, ...] = (
    "qwen25",
    "gpt2zinc",
    "gpmolformer",
    "gpmolformer_dpo",
    "gpt2zinc_dpo",
    "qwen25_dpo",
)

_V3_DISPLAY_NAMES: dict[str, str] = {
    "gpmolformer": "GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer-Uniq)",
    "gpt2zinc": "GenEuclidean v3 (PaiNN-SE3 + GPT2-ZINC-87M)",
    "qwen25": "GenEuclidean v3 (PaiNN-SE3 + Qwen2.5-0.5B)",
    "gpmolformer_dpo": "GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer + RPO-DPO)",
    "gpt2zinc_dpo": "GenEuclidean v3 (PaiNN-SE3 + GPT2-ZINC-87M + RPO-DPO)",
    "qwen25_dpo": "GenEuclidean v3 (PaiNN-SE3 + Qwen2.5-0.5B + RPO-DPO)",
}

_STANDARD_RESIDUES: tuple[str, ...] = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "OTH",
)

_LIG_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=LIGAND_FP_DIM)


def _json_safe(obj: Any) -> Any:
    """Recursively convert NumPy, Path, and non-finite floats into strict JSON types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: Any) -> Path:
    """Write strictly valid JSON to ``path`` creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _short_v3_name(raw_name: str) -> str:
    """Normalize run directory name into short v3 tag (e.g. ``gpmolformer``)."""
    cleaned = raw_name.strip()
    for prefix in ("geneuclidean_v3_", "geneuclidean-v3-", "v3_"):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :]
    return cleaned


def _smiles_to_fp_tensor(smiles_list: Sequence[str], fp_dim: int = LIGAND_FP_DIM) -> torch.Tensor:
    """Compute float32 Morgan bit fingerprints ``[M, fp_dim]`` for model scoring heads."""
    gen = (
        _LIG_FP_GEN
        if fp_dim == LIGAND_FP_DIM
        else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    )
    arr = np.zeros((len(smiles_list), fp_dim), dtype=np.float32)
    for idx, smi in enumerate(smiles_list):
        mol = parse_mol(smi)
        if mol is None:
            continue
        bv = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(bv, arr[idx])
    return torch.from_numpy(arr)


def _load_cache_and_splits(
    cache_path: Path,
    splits_path: Path,
    split_type: str,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Load preprocessed pocket records plus train/test PDB IDs for ``split_type``."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records: dict[str, dict[str, Any]] = payload["records"]
    splits_data = json.loads(splits_path.read_text(encoding="utf-8"))

    if split_type == "casf2016_core":
        train_ids = [pid for pid in splits_data["casf2016"]["train"] if pid in records]
        test_ids = [pid for pid in splits_data["casf2016_core_ids"] if pid in records]
    else:
        group = splits_data[split_type]
        train_ids = [pid for pid in group["train"] if pid in records]
        test_ids = [pid for pid in group["test"] if pid in records]
    return records, train_ids, test_ids


def _record_to_pocket_batch(rec: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert a single cached pocket record into batch-size-1 model input tensors."""
    coords = torch.as_tensor(rec["coords"], dtype=torch.float32).unsqueeze(0)
    node_feats = torch.as_tensor(rec["node_feats"], dtype=torch.float32).unsqueeze(0)
    contact_mask = torch.as_tensor(rec["contact_mask"], dtype=torch.bool).unsqueeze(0)
    atom_mask = torch.ones((1, coords.shape[1]), dtype=torch.bool)
    fp_unpacked = np.unpackbits(rec["ligand_fp_packed"])[:LIGAND_FP_DIM].astype(np.float32)
    return {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "contact_mask": contact_mask,
        "ligand_fp": torch.from_numpy(fp_unpacked).unsqueeze(0),
    }


def _find_raw_complex_files(raw_dir: Path, pdb_id: str) -> tuple[Path | None, Path | None]:
    """Locate ``<pdb_id>_pocket.pdb`` and ``<pdb_id>_ligand.sdf`` under ``raw_dir``."""
    pid = pdb_id.strip().lower()
    for subdir in ("pbpp2020", "casf2016_extra"):
        cdir = raw_dir / subdir / pid
        pocket = cdir / f"{pid}_pocket.pdb"
        if pocket.is_file():
            sdf = cdir / f"{pid}_ligand.sdf"
            return pocket, (sdf if sdf.is_file() else None)
    return None, None


def _aggregate_atom_attention_to_residues(
    rec: Mapping[str, Any],
    attn_map: np.ndarray,
    raw_dir: Path,
    max_residues: int = 24,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate query-to-atom attention ``[Q, N_atoms]`` into interpretable pocket residues."""
    n_q, n_atoms = attn_map.shape
    contact_atoms = np.asarray(rec["contact_mask"], dtype=bool)[:n_atoms]
    coords_local = np.asarray(rec["coords"], dtype=np.float64)[:n_atoms]
    center = np.asarray(rec.get("center", np.zeros(3)), dtype=np.float64)
    world_coords = coords_local + center[None, :]
    pdb_id = str(rec.get("pdb_id", "pocket"))

    atom_res_labels: list[str] = []
    pocket_pdb, _ = _find_raw_complex_files(raw_dir, pdb_id)
    if pocket_pdb is not None:
        pdb_coords: list[list[float]] = []
        pdb_labels: list[str] = []
        with pocket_pdb.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                resname = line[17:20].strip().upper()
                if resname in {"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL"}:
                    continue
                elem = line[76:78].strip().upper() if len(line) >= 78 else ""
                if elem == "H":
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                resseq = line[22:26].strip()
                chain = line[21].strip()
                lbl = f"{resname}{resseq}" + (f":{chain}" if chain else "")
                pdb_coords.append([x, y, z])
                pdb_labels.append(lbl)
        if pdb_coords:
            pdb_arr = np.asarray(pdb_coords, dtype=np.float64)
            dists = np.linalg.norm(world_coords[:, None, :] - pdb_arr[None, :, :], axis=-1)
            nearest = np.argmin(dists, axis=1)
            atom_res_labels = [pdb_labels[int(idx)] for idx in nearest]

    if len(atom_res_labels) != n_atoms:
        nf = np.asarray(rec["node_feats"], dtype=np.float32)[:n_atoms]
        res_idx = np.argmax(nf[:, 10:31], axis=1) if nf.shape[1] >= 31 else np.zeros(n_atoms, int)
        atom_res_labels = []
        cur_group = 0
        prev_r = -1
        for r_i in res_idx.tolist():
            if r_i != prev_r:
                cur_group += 1
                prev_r = r_i
            name = (
                _STANDARD_RESIDUES[int(r_i)] if 0 <= int(r_i) < len(_STANDARD_RESIDUES) else "OTH"
            )
            atom_res_labels.append(f"{name}{cur_group}")

    res_to_atoms: dict[str, list[int]] = {}
    for atom_idx, lbl in enumerate(atom_res_labels):
        res_to_atoms.setdefault(lbl, []).append(atom_idx)

    res_labels = list(res_to_atoms.keys())
    res_attn = np.zeros((n_q, len(res_labels)), dtype=np.float64)
    res_contact = np.zeros(len(res_labels), dtype=bool)
    for r_i, lbl in enumerate(res_labels):
        idxs = res_to_atoms[lbl]
        res_attn[:, r_i] = attn_map[:, idxs].sum(axis=1)
        res_contact[r_i] = bool(np.any(contact_atoms[idxs]))

    if len(res_labels) > max_residues:
        mean_w = res_attn.mean(axis=0)
        contact_idx = np.flatnonzero(res_contact)
        periph_idx = np.flatnonzero(~res_contact)
        keep_c = contact_idx[np.argsort(-mean_w[contact_idx])][: max_residues // 2 + 4]
        rem = max_residues - len(keep_c)
        keep_p = periph_idx[np.argsort(-mean_w[periph_idx])][: max(0, rem)]
        keep = np.sort(np.concatenate([keep_c, keep_p]))
        res_labels = [res_labels[int(i)] for i in keep]
        res_attn = res_attn[:, keep]
        res_contact = res_contact[keep]

    row_sums = res_attn.sum(axis=1, keepdims=True)
    res_attn = np.divide(res_attn, row_sums, out=np.zeros_like(res_attn), where=row_sums > 1e-12)
    return res_labels, res_attn, res_contact


def _build_shuffled_derangement(
    pocket_ids: Sequence[str],
    native_smiles: Mapping[str, str],
) -> dict[str, str]:
    """Map each pocket to a distinct decoy pocket (avoiding identical native ligands)."""
    n = len(pocket_ids)
    if n < 2:
        return {}
    offset = max(1, n // 2)
    mapping: dict[str, str] = {}
    for i, pid in enumerate(pocket_ids):
        chosen = pocket_ids[(i + offset) % n]
        for shift in range(offset, offset + n):
            cand = pocket_ids[(i + shift) % n]
            if cand != pid and native_smiles.get(cand) != native_smiles.get(pid):
                chosen = cand
                break
        mapping[pid] = chosen
    return mapping


def _discover_v3_runs(
    runs_dir: Path,
    explicit_models: Sequence[str] | None,
) -> list[tuple[str, Path]]:
    """Resolve ``(short_name, best_adapter_path)`` pairs to evaluate."""
    if explicit_models:
        resolved: list[tuple[str, Path]] = []
        for item in explicit_models:
            cand = Path(item)
            if cand.is_file():
                resolved.append((_short_v3_name(cand.parent.name), cand))
            elif (cand / "best_adapter.pt").is_file():
                resolved.append((_short_v3_name(cand.name), cand / "best_adapter.pt"))
            elif (runs_dir / item / "best_adapter.pt").is_file():
                resolved.append((_short_v3_name(item), runs_dir / item / "best_adapter.pt"))
            elif (runs_dir / f"geneuclidean_v3_{item}" / "best_adapter.pt").is_file():
                resolved.append(
                    (
                        _short_v3_name(item),
                        runs_dir / f"geneuclidean_v3_{item}" / "best_adapter.pt",
                    )
                )
            else:
                raise FileNotFoundError(f"Could not resolve v3 adapter checkpoint for {item!r}")
        return resolved

    if not runs_dir.is_dir():
        return []

    found: dict[str, Path] = {}
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("geneuclidean_v3_"):
            continue
        ckpt = entry / "best_adapter.pt"
        if ckpt.is_file():
            found[_short_v3_name(entry.name)] = ckpt

    ordered: list[tuple[str, Path]] = []
    for name in _V3_MODEL_ORDER:
        if name in found:
            ordered.append((name, found.pop(name)))
    for name in sorted(found.keys()):
        ordered.append((name, found[name]))
    return ordered


def _load_v3_model(
    adapter_ckpt: Path,
    *,
    device: torch.device,
    local_files_only: bool = False,
) -> tuple[PocketConditionedLLM, ExperimentConfig, int]:
    """Reconstruct ``PocketConditionedLLM``, load v2 frozen encoder + trained v3 adapter."""
    payload = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
    sibling_cfg = adapter_ckpt.parent / "config.yaml"
    if sibling_cfg.is_file():
        cfg = ExperimentConfig.from_yaml(sibling_cfg)
    elif isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping):
        meta = payload["metadata"]
        raw_cfg = meta.get("config") or meta.get("v2_cfg")
        cfg = (
            ExperimentConfig.model_validate(raw_cfg)
            if isinstance(raw_cfg, dict)
            else ExperimentConfig()
        )
    elif isinstance(payload, Mapping) and isinstance(payload.get("config"), dict):
        cfg = ExperimentConfig.model_validate(payload["config"])
    else:
        cfg = ExperimentConfig()

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
        local_files_only=local_files_only,
    )
    model = PocketConditionedLLM(
        v2_cfg=cfg,
        backbone=backbone,
        num_queries=llm_cfg.num_queries,
        perceiver_layers=llm_cfg.perceiver_layers,
        perceiver_heads=llm_cfg.perceiver_heads,
        num_global_tokens=llm_cfg.num_global_tokens,
        cfg_drop_prob=llm_cfg.cfg_drop_prob,
        freeze_pocket_encoder=llm_cfg.freeze_pocket_encoder,
    )

    v2_ckpt = Path(llm_cfg.v2_pocket_ckpt)
    if v2_ckpt.is_file():
        model.load_v2_pocket_checkpoint(v2_ckpt)
    model.load_adapter_checkpoint(adapter_ckpt, map_location="cpu")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, n_params


@torch.no_grad()
def _evaluate_tier4_full_split(
    model: PocketConditionedLLM,
    records: Mapping[str, dict[str, Any]],
    test_ids: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate binding affinity (pKd) and DrugCLIP retrieval over ALL test complexes."""
    all_pooled: list[np.ndarray] = []
    all_pkd_true: list[float] = []
    all_pkd_pred: list[float] = []
    all_clip_p: list[np.ndarray] = []
    all_clip_l: list[np.ndarray] = []
    used_ids: list[str] = []

    for start in range(0, len(test_ids), batch_size):
        chunk_ids = list(test_ids[start : start + batch_size])
        chunk_recs = [records[pid] for pid in chunk_ids]
        bsz = len(chunk_recs)
        max_atoms = max(int(np.asarray(r["coords"]).shape[0]) for r in chunk_recs)
        node_dim = int(np.asarray(chunk_recs[0]["node_feats"]).shape[-1])

        coords = torch.zeros((bsz, max_atoms, 3), dtype=torch.float32)
        node_feats = torch.zeros((bsz, max_atoms, node_dim), dtype=torch.float32)
        atom_mask = torch.zeros((bsz, max_atoms), dtype=torch.bool)
        ligand_fps = torch.zeros((bsz, LIGAND_FP_DIM), dtype=torch.float32)

        for i, rec in enumerate(chunk_recs):
            c = np.asarray(rec["coords"], dtype=np.float32)
            nf = np.asarray(rec["node_feats"], dtype=np.float32)
            n_a = int(c.shape[0])
            coords[i, :n_a] = torch.from_numpy(c)
            node_feats[i, :n_a] = torch.from_numpy(nf)
            atom_mask[i, :n_a] = True
            fp = np.unpackbits(rec["ligand_fp_packed"])[:LIGAND_FP_DIM].astype(np.float32)
            ligand_fps[i] = torch.from_numpy(fp)

        pocket_out = model.encode_pocket(
            coords.to(device), node_feats.to(device), atom_mask.to(device)
        )
        all_pooled.append(pocket_out.pooled_emb.detach().cpu().numpy())

        lig_emb = model.heads.encode_ligand(ligand_fps.to(device))
        mt_out = model.heads(pocket_out.pooled_emb, ligand_emb=lig_emb)

        used_ids.extend(chunk_ids)
        all_pkd_true.extend(float(r["pkd"]) for r in chunk_recs)
        if mt_out.pred_pkd is not None:
            all_pkd_pred.extend(mt_out.pred_pkd.detach().cpu().numpy().tolist())
        if mt_out.clip_pocket is not None and mt_out.clip_ligand is not None:
            all_clip_p.append(mt_out.clip_pocket.detach().cpu().numpy())
            all_clip_l.append(mt_out.clip_ligand.detach().cpu().numpy())

    return {
        "n_test_complexes": len(used_ids),
        "pdb_ids": used_ids,
        "pkd_y_true": all_pkd_true,
        "pkd_y_pred": all_pkd_pred if len(all_pkd_pred) == len(all_pkd_true) else [],
        "affinity": (
            evaluate_affinity_metrics(all_pkd_true, all_pkd_pred)
            if len(all_pkd_pred) == len(all_pkd_true) and all_pkd_pred
            else None
        ),
        "clip_retrieval": (
            evaluate_clip_retrieval(
                np.concatenate(all_clip_p, axis=0), np.concatenate(all_clip_l, axis=0)
            )
            if all_clip_p and all_clip_l
            else None
        ),
        "pocket_embeddings": np.concatenate(all_pooled, axis=0) if all_pooled else np.empty((0, 0)),
    }


def _run_cfg_and_h2l_sweep(
    model: PocketConditionedLLM,
    cfg: ExperimentConfig,
    short_name: str,
    *,
    records: Mapping[str, dict[str, Any]],
    train_smiles: Sequence[str],
    test_ids: Sequence[str],
    output_dir: Path,
    device: torch.device,
    n_pockets: int,
    num_samples: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Run CFG scale sweep (w = 1.0, 1.3, 1.6, 2.0) + Hit-to-Lead CVAE seed refinement."""
    pids = list(test_ids[:n_pockets]) if n_pockets > 0 else list(test_ids)
    native_map = {pid: str(records[pid]["canonical_smiles"]) for pid in pids if pid in records}

    sweep_specs: list[tuple[str, float, float, bool]] = [
        ("CFG w=1.0", 1.0, 1.0, False),
        ("CFG w=1.3", 1.3, 1.0, False),
        ("CFG w=1.6", 1.6, 1.0, False),
        ("CFG w=2.0", 2.0, 1.0, False),
        ("Hit-to-Lead CVAE", 1.6, 0.5, True),
    ]

    rows: list[dict[str, Any]] = []
    for s_idx, (label, w_cfg, l_temp, use_seed_fp) in enumerate(sweep_specs):
        t0 = time.perf_counter()
        per_pocket_gen: dict[str, list[str]] = {}
        all_smis: list[str] = []
        pred_pkds: list[float] = []

        for p_idx, pid in enumerate(pids):
            p_batch = _record_to_pocket_batch(records[pid])
            with torch.no_grad():
                p_out = model.encode_pocket(
                    p_batch["coords"].to(device),
                    p_batch["node_feats"].to(device),
                    p_batch["atom_mask"].to(device),
                )
            a_mask = p_batch["atom_mask"].to(device)
            seed_fp = p_batch["ligand_fp"].to(device) if use_seed_fp else None
            smis = sample_llm_molecules(
                model,
                p_out,
                a_mask,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                strategy=cfg.sampling.strategy,
                temperature=cfg.sampling.temperature,
                min_p=cfg.sampling.min_p,
                top_p=cfg.sampling.top_p,
                top_k=cfg.sampling.top_k,
                cfg_scale=w_cfg,
                latent_temp=l_temp,
                repetition_penalty=cfg.sampling.repetition_penalty,
                seed_ligand_fp=seed_fp,
                batch_size=num_samples,
                seed=seed + 10_000 * s_idx + p_idx,
            )
            per_pocket_gen[pid] = smis
            all_smis.extend(smis)

            valid_smis = [s for s in smis if parse_mol(s) is not None]
            if valid_smis:
                with torch.no_grad():
                    fps_t = _smiles_to_fp_tensor(valid_smis, fp_dim=cfg.multitask.ligand_fp_dim).to(
                        device
                    )
                    sc = model.score_candidates(p_out, fps_t)
                    if "pkd" in sc:
                        pred_pkds.extend(sc["pkd"].detach().cpu().numpy().tolist())

        t1 = evaluate_generation_metrics(
            all_smis,
            train_smiles=train_smiles,
            reference_smiles=list(native_map.values()),
            seed=seed,
        )
        t2 = evaluate_pocket_specificity(
            per_pocket_generated=per_pocket_gen,
            per_pocket_native_smiles=native_map,
            n_bootstrap=400,
            seed=seed,
        )
        row = {
            "label": label,
            "cfg_scale": w_cfg,
            "latent_temp": l_temp,
            "use_seed_refine": use_seed_fp,
            "validity": t1["validity"],
            "uniqueness": t1["uniqueness"],
            "novelty": t1["novelty"],
            "int_div_1": t1["int_div_1"],
            "intra_int_div_1": t2["intra_int_div_1_mean"],
            "QED": t1["QED_mean"],
            "SA": t1["SA_mean"],
            "SA_norm": t1["SA_norm_mean"],
            "tanimoto": t2["matched_tanimoto_mean"],
            "count_dice": t2["matched_count_dice_mean"],
            "pred_pkd": float(np.mean(pred_pkds)) if pred_pkds else float("nan"),
            "elapsed_sec": time.perf_counter() - t0,
        }
        rows.append(row)
        print(
            f"[cfg-sweep:{short_name}] {label:<18s} | Valid={row['validity']:.3f} "
            f"IntDiv1={row['int_div_1']:.3f} QED={row['QED']:.3f} "
            f"Tan={row['tanimoto']:.3f} ({row['elapsed_sec']:.1f}s)",
            flush=True,
        )

    payload = {
        "model_name": f"geneuclidean_v3_{short_name}",
        "n_pockets": len(pids),
        "num_samples": num_samples,
        "results": rows,
    }
    _write_json(output_dir / f"cfg_sweep_v3_{short_name}.json", payload)
    pd.DataFrame(rows).to_csv(output_dir / f"cfg_sweep_v3_{short_name}.csv", index=False)
    return payload


def evaluate_single_v3_model(
    short_name: str,
    adapter_ckpt: Path,
    *,
    records: Mapping[str, dict[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    split_name: str,
    raw_dir: Path,
    output_dir: Path,
    device: torch.device,
    num_samples: int,
    max_pockets: int,
    attention_pockets: int,
    max_new_tokens: int,
    cfg_scale_override: float | None,
    temperature_override: float | None,
    min_p_override: float | None,
    seed: int,
    run_cfg_sweep: bool,
    sweep_pockets: int,
    local_files_only: bool,
) -> dict[str, Any]:
    """Run full 5-tier CASF-2016 evaluation for one trained v3.0 LLM model."""
    t0 = time.perf_counter()
    seed_everything(seed)
    model, cfg, n_params = _load_v3_model(
        adapter_ckpt, device=device, local_files_only=local_files_only
    )
    llm_cfg = cfg.llm if cfg.llm is not None else LLMConfig()

    train_smiles = [str(records[pid]["canonical_smiles"]) for pid in train_ids if pid in records]
    test_native = {pid: str(records[pid]["canonical_smiles"]) for pid in test_ids if pid in records}
    sample_pids = list(test_ids[:max_pockets]) if max_pockets > 0 else list(test_ids)

    eff_cfg_scale = (
        float(cfg_scale_override)
        if cfg_scale_override is not None
        else (cfg.sampling.cfg_scale if llm_cfg.cfg_drop_prob > 0.0 else 1.0)
    )
    eff_temp = (
        float(temperature_override)
        if temperature_override is not None
        else cfg.sampling.temperature
    )
    eff_min_p = float(min_p_override) if min_p_override is not None else cfg.sampling.min_p
    eff_latent_temp = cfg.sampling.latent_temp if cfg.cvae.enabled else 0.0

    per_pocket_matched: dict[str, list[str]] = {}
    per_pocket_null: dict[str, list[str]] = {}
    per_pocket_reranked: dict[str, list[dict[str, Any]]] = {}
    all_raw_matched: list[str] = []
    all_reranked_smiles: list[str] = []

    print(
        f"[v3:{short_name}] Sampling {len(sample_pids)} pockets x {num_samples} mols "
        f"(backbone={llm_cfg.backbone_id}, cfg={eff_cfg_scale}, T={eff_temp}, min_p={eff_min_p}) "
        f"on {device}...",
        flush=True,
    )

    for idx, pid in enumerate(sample_pids):
        p_batch = _record_to_pocket_batch(records[pid])
        with torch.no_grad():
            pocket_out = model.encode_pocket(
                p_batch["coords"].to(device),
                p_batch["node_feats"].to(device),
                p_batch["atom_mask"].to(device),
            )
        atom_mask_dev = p_batch["atom_mask"].to(device)

        matched_smis = sample_llm_molecules(
            model,
            pocket_out,
            atom_mask_dev,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            strategy=cfg.sampling.strategy,
            temperature=eff_temp,
            min_p=eff_min_p,
            top_p=cfg.sampling.top_p,
            top_k=cfg.sampling.top_k,
            cfg_scale=eff_cfg_scale,
            latent_temp=eff_latent_temp,
            repetition_penalty=cfg.sampling.repetition_penalty,
            batch_size=num_samples,
            seed=seed + idx,
        )
        null_smis = sample_llm_molecules(
            model,
            pocket_out,
            atom_mask_dev,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            strategy=cfg.sampling.strategy,
            temperature=eff_temp,
            min_p=eff_min_p,
            top_p=cfg.sampling.top_p,
            top_k=cfg.sampling.top_k,
            cfg_scale=0.0,
            latent_temp=0.0,
            repetition_penalty=cfg.sampling.repetition_penalty,
            batch_size=num_samples,
            seed=seed + 100_000 + idx,
        )

        per_pocket_matched[pid] = matched_smis
        per_pocket_null[pid] = null_smis
        all_raw_matched.extend(matched_smis)

        with torch.no_grad():
            cand_fps = _smiles_to_fp_tensor(matched_smis, fp_dim=cfg.multitask.ligand_fp_dim).to(
                device
            )
            cand_scores = model.score_candidates(pocket_out, cand_fps)

        pred_pkd_arr = cand_scores["pkd"].detach().cpu().numpy() if "pkd" in cand_scores else None
        clip_sim_arr = (
            cand_scores["clip_sim"].detach().cpu().numpy() if "clip_sim" in cand_scores else None
        )
        selected = rerank_and_select(
            matched_smis,
            k=min(10, num_samples),
            pred_pkd=pred_pkd_arr,
            clip_sim=clip_sim_arr,
            min_tanimoto_dist=0.35,
            relax_if_needed=True,
        )

        nat_mol = parse_mol(test_native[pid])
        nat_bv = get_morgan_bit_fp(nat_mol) if nat_mol is not None else None
        nat_cfp = get_morgan_count_fp(nat_mol) if nat_mol is not None else None

        ranked_records: list[dict[str, Any]] = []
        for entry in selected:
            smi_c = entry.canonical_smiles or entry.smiles
            all_reranked_smiles.append(smi_c)
            cmol = parse_mol(smi_c)
            tan = (
                tanimoto_similarity(get_morgan_bit_fp(cmol), nat_bv)
                if (cmol is not None and nat_bv is not None)
                else float("nan")
            )
            cdice = (
                dice_similarity(get_morgan_count_fp(cmol), nat_cfp)
                if (cmol is not None and nat_cfp is not None)
                else float("nan")
            )
            ranked_records.append(
                {
                    "smiles": smi_c,
                    "qed": entry.qed,
                    "sa_score": entry.sa_score,
                    "sa_norm": entry.sa_norm,
                    "lipinski_pass": entry.lipinski_pass,
                    "pains_free": entry.pains_free,
                    "pred_pkd": entry.pred_pkd,
                    "clip_sim": entry.clip_sim,
                    "mpo_score": entry.mpo_score,
                    "tanimoto": tan,
                    "count_dice": cdice,
                }
            )
        per_pocket_reranked[pid] = ranked_records

    shuf_map = _build_shuffled_derangement(sample_pids, test_native)
    per_pocket_shuffled = {
        pid: per_pocket_matched[decoy_pid] for pid, decoy_pid in shuf_map.items()
    }

    ref_smiles = [test_native[pid] for pid in sample_pids if pid in test_native]
    tier1_raw = evaluate_generation_metrics(
        all_raw_matched, train_smiles=train_smiles, reference_smiles=ref_smiles, seed=seed
    )
    tier1_reranked = evaluate_generation_metrics(
        all_reranked_smiles, train_smiles=train_smiles, reference_smiles=ref_smiles, seed=seed
    )
    tier2 = evaluate_pocket_specificity(
        per_pocket_generated=per_pocket_matched,
        per_pocket_native_smiles=test_native,
        per_pocket_shuffled_generated=per_pocket_shuffled if per_pocket_shuffled else None,
        per_pocket_null_generated=per_pocket_null,
        n_bootstrap=1000,
        seed=seed,
    )

    # Tier 3: <4.5 A Perceiver Contact Atom Attention Enrichment
    attn_pids = sample_pids[: min(attention_pockets, len(sample_pids))]
    attn_records: list[dict[str, float]] = []
    attn_examples: list[dict[str, Any]] = []

    for idx, pid in enumerate(attn_pids):
        rec = records[pid]
        p_batch = _record_to_pocket_batch(rec)
        with torch.no_grad():
            pocket_out = model.encode_pocket(
                p_batch["coords"].to(device),
                p_batch["node_feats"].to(device),
                p_batch["atom_mask"].to(device),
            )
        atom_mask_dev = p_batch["atom_mask"].to(device)
        smis, perceiver_attn = sample_llm_with_attention(
            model,
            pocket_out,
            atom_mask_dev,
            num_samples=4,
            max_new_tokens=max_new_tokens,
            strategy=cfg.sampling.strategy,
            temperature=eff_temp,
            min_p=eff_min_p,
            top_p=cfg.sampling.top_p,
            top_k=cfg.sampling.top_k,
            cfg_scale=eff_cfg_scale,
            latent_temp=0.0,
            repetition_penalty=cfg.sampling.repetition_penalty,
            batch_size=4,
            seed=seed + 200_000 + idx,
        )
        atom_mask_np = p_batch["atom_mask"].squeeze(0).numpy()
        contact_mask_np = p_batch["contact_mask"].squeeze(0).numpy()
        if perceiver_attn is None or perceiver_attn.numel() == 0:
            continue

        # perceiver_attn: [B, H, Q, N] or [H, Q, N] or [Q, N] -> mean over batch & heads -> [Q, N]
        attn_np = perceiver_attn.detach().cpu().float().numpy()
        if attn_np.ndim == 4:
            q_atom_map = attn_np.mean(axis=(0, 1))
        elif attn_np.ndim == 3:
            q_atom_map = attn_np.mean(axis=0)
        else:
            q_atom_map = attn_np

        audit = evaluate_attention_enrichment(q_atom_map, atom_mask_np, contact_mask_np)
        if math.isfinite(audit["enrichment_ratio"]):
            attn_records.append(audit)
            valid_smi = next((s for s in smis if parse_mol(s) is not None), smis[0] if smis else "")
            res_labels, res_attn, res_contact = _aggregate_atom_attention_to_residues(
                rec, q_atom_map, raw_dir=raw_dir, max_residues=24
            )
            attn_examples.append(
                {
                    "pdb_id": pid,
                    "smiles": valid_smi,
                    "tokens": [f"Q{q}" for q in range(q_atom_map.shape[0])],
                    "residue_labels": res_labels,
                    "attn_matrix": res_attn.tolist(),
                    "contact_mask": res_contact.tolist(),
                    "enrichment_ratio": audit["enrichment_ratio"],
                    "enrichment_vs_uniform": audit["enrichment_vs_uniform"],
                    "normalized_entropy": audit["normalized_entropy"],
                }
            )

    attn_examples.sort(key=lambda d: float(d["enrichment_ratio"]), reverse=True)
    tier3_attn: dict[str, Any] = {
        "n_molecules_audited": len(attn_records),
        "enrichment_ratio_mean": (
            float(np.mean([r["enrichment_ratio"] for r in attn_records]))
            if attn_records
            else float("nan")
        ),
        "enrichment_vs_uniform_mean": (
            float(np.mean([r["enrichment_vs_uniform"] for r in attn_records]))
            if attn_records
            else float("nan")
        ),
        "contact_attn_mass_mean": (
            float(np.mean([r["contact_attn_mass"] for r in attn_records]))
            if attn_records
            else float("nan")
        ),
        "normalized_entropy_mean": (
            float(np.mean([r["normalized_entropy"] for r in attn_records]))
            if attn_records
            else float("nan")
        ),
        "examples": attn_examples[:3],
    }

    # Tier 4: Full-split pKd Affinity & DrugCLIP Retrieval + Pocket Embeddings
    tier4 = _evaluate_tier4_full_split(model, records, test_ids, device=device, batch_size=32)
    emb_arr = tier4.pop("pocket_embeddings")
    emb_path = output_dir / f"pocket_embeddings_v3_{short_name}.npz"
    if isinstance(emb_arr, np.ndarray) and emb_arr.size > 0:
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            emb_path,
            embeddings=emb_arr.astype(np.float32),
            pkd=np.asarray(tier4["pkd_y_true"], dtype=np.float32),
            pdb_ids=np.asarray(tier4["pdb_ids"]),
        )

    cfg_sweep_payload: dict[str, Any] | None = None
    if run_cfg_sweep:
        cfg_sweep_payload = _run_cfg_and_h2l_sweep(
            model,
            cfg,
            short_name,
            records=records,
            train_smiles=train_smiles,
            test_ids=test_ids,
            output_dir=output_dir,
            device=device,
            n_pockets=sweep_pockets,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    elapsed = time.perf_counter() - t0
    full_model_name = f"geneuclidean_v3_{short_name}"
    eval_payload: dict[str, Any] = {
        "model_name": full_model_name,
        "display_name": _V3_DISPLAY_NAMES.get(short_name, f"GenEuclidean v3 ({short_name})"),
        "checkpoint": str(adapter_ckpt),
        "backbone_id": llm_cfg.backbone_id,
        "split": split_name,
        "num_params": n_params,
        "encoder_arch": cfg.encoder.arch,
        "decoder_arch": llm_cfg.backbone_id,
        "representation": cfg.data.representation,
        "cvae_enabled": cfg.cvae.enabled,
        "cfg_scale": eff_cfg_scale,
        "n_pockets_sampled": len(sample_pids),
        "samples_per_pocket": num_samples,
        "elapsed_sec": elapsed,
        "tier1_raw": tier1_raw,
        "tier1_reranked": tier1_reranked,
        "tier2_specificity": tier2,
        "tier3_attention": tier3_attn,
        "tier4_affinity_clip": tier4,
        "pocket_embeddings_path": str(emb_path),
    }
    if cfg_sweep_payload is not None:
        eval_payload["cfg_sweep"] = cfg_sweep_payload

    samples_payload = {
        "model_name": full_model_name,
        "split": split_name,
        "native_smiles": {pid: test_native[pid] for pid in sample_pids if pid in test_native},
        "per_pocket_matched": per_pocket_matched,
        "per_pocket_reranked": per_pocket_reranked,
    }

    _write_json(output_dir / f"eval_v3_{short_name}.json", eval_payload)
    _write_json(output_dir / f"samples_v3_{short_name}.json", samples_payload)
    pkd_r = tier4["affinity"]["pearson_r"] if tier4.get("affinity") else float("nan")
    print(
        f"[v3:{short_name}] Done in {elapsed:.1f}s | Valid={tier1_raw['validity']:.3f} "
        f"Uniq={tier1_raw['uniqueness']:.3f} IntDiv1={tier1_raw['int_div_1']:.3f} "
        f"QED={tier1_raw['QED_mean']:.3f} Tan={tier2['matched_tanimoto_mean']:.3f} "
        f"(Dice={tier2['matched_count_dice_mean']:.3f}) "
        f"AttnEnr={tier3_attn['enrichment_ratio_mean']:.2f}x pKd_r={pkd_r:.3f}",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return eval_payload


def _summary_row_from_eval(eval_data: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ``eval_*.json`` dictionary into a unified comparison table row."""
    t1 = eval_data.get("tier1_raw", {})
    t2 = eval_data.get("tier2_specificity", {})
    t3 = eval_data.get("tier3_attention", {})
    t4 = eval_data.get("tier4_affinity_clip", {})
    aff = t4.get("affinity") or {}
    clip = t4.get("clip_retrieval") or {}
    vs_shuf = t2.get("vs_shuffled") or {}

    return {
        "model": eval_data.get("display_name", eval_data.get("model_name", "unknown")),
        "run_id": eval_data.get("model_name", "unknown"),
        "num_params": eval_data.get("num_params"),
        "validity": t1.get("validity"),
        "uniqueness": t1.get("uniqueness"),
        "novelty": t1.get("novelty"),
        "int_div_1": t1.get("int_div_1"),
        "int_div_2": t1.get("int_div_2"),
        "murcko_scaffold_diversity": t1.get("murcko_scaffold_diversity"),
        "QED": t1.get("QED_mean"),
        "SA": t1.get("SA_mean"),
        "SA_norm": t1.get("SA_norm_mean"),
        "lipinski_pass_rate": t1.get("lipinski_pass_rate"),
        "pains_free_rate": t1.get("pains_free_rate"),
        "True Tanimoto": t2.get("matched_tanimoto_mean"),
        "Count Dice": t2.get("matched_count_dice_mean"),
        "shuffled_tanimoto": vs_shuf.get("control_tanimoto_mean"),
        "cliffs_delta": vs_shuf.get("cliffs_delta"),
        "attn_enrichment": t3.get("enrichment_ratio_mean"),
        "pKd Pearson r": aff.get("pearson_r"),
        "pKd RMSE": aff.get("rmse"),
        "DrugCLIP R@1": clip.get("recall@1"),
        "DrugCLIP AUC": clip.get("roc_auc"),
    }


def write_v3_comparison_tables(output_dir: Path) -> tuple[Path, Path]:
    """Aggregate 2021 Thesis + all v2.0 + all v3.0 eval JSONs into unified CSV & Markdown."""
    rows: list[dict[str, Any]] = [dict(THESIS_2021_BASELINE)]

    v2_files: dict[str, Path] = {}
    v3_files: dict[str, Path] = {}
    for path in sorted(output_dir.glob("eval_*.json")):
        stem = path.stem.removeprefix("eval_")
        if stem.startswith("v3_"):
            v3_files[stem.removeprefix("v3_")] = path
        else:
            v2_files[stem] = path

    ordered_v2 = [n for n in _V2_MODEL_ORDER if n in v2_files] + [
        n for n in sorted(v2_files.keys()) if n not in _V2_MODEL_ORDER
    ]
    for name in ordered_v2:
        data = json.loads(v2_files[name].read_text(encoding="utf-8"))
        rows.append(_summary_row_from_eval(data))

    ordered_v3 = [n for n in _V3_MODEL_ORDER if n in v3_files] + [
        n for n in sorted(v3_files.keys()) if n not in _V3_MODEL_ORDER
    ]
    for name in ordered_v3:
        data = json.loads(v3_files[name].read_text(encoding="utf-8"))
        rows.append(_summary_row_from_eval(data))

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "model_comparison_v3.csv"
    md_path = output_dir / "model_comparison_v3.md"
    df.to_csv(csv_path, index=False)

    md_cols = [
        "model",
        "validity",
        "uniqueness",
        "novelty",
        "int_div_1",
        "QED",
        "SA_norm",
        "True Tanimoto",
        "Count Dice",
        "cliffs_delta",
        "attn_enrichment",
        "pKd Pearson r",
    ]
    md_lines = [
        "| " + " | ".join(md_cols) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(md_cols) - 1)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = [str(row["model"])]
        for col in md_cols[1:]:
            val = row.get(col)
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                cells.append("—")
            else:
                cells.append(f"{float(val):.3f}")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the GenEuclidean v3.0 5-tier benchmark driver."""
    parser = argparse.ArgumentParser(
        description="GenEuclidean v3.0 — 5-Tier Zero-Leakage CASF-2016 Benchmark Driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        nargs="*",
        default=None,
        help="v3 model names (e.g. gpmolformer gpt2zinc qwen25 gpmolformer_dpo) or adapter paths.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Trained runs root.")
    parser.add_argument(
        "--split",
        choices=("casf2016", "lp_pdbbind", "scaffold", "random"),
        default="casf2016",
        help="Evaluation split name from splits.json.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/processed/pdbbind_pockets.pt"),
        help="Path to preprocessed pocket cache.",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("data/processed/splits.json"),
        help="Path to leak-proof splits JSON.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw PDBbind dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Output directory for JSON/CSV/Markdown tables.",
    )
    parser.add_argument("--device", default="auto", help="Execution device: auto|mps|cuda|cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-samples", type=int, default=32, help="Samples per pocket (K).")
    parser.add_argument(
        "--max-pockets",
        type=int,
        default=100,
        help="Max test pockets to sample (0 = all test pockets).",
    )
    parser.add_argument(
        "--attention-pockets",
        type=int,
        default=30,
        help="Number of test pockets for Tier 3 Perceiver attention audit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens per molecule.",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=None, help="Override CFG guidance scale."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override sampling temperature."
    )
    parser.add_argument("--min-p", type=float, default=None, help="Override min-p threshold.")
    parser.add_argument(
        "--cfg-sweep",
        action="store_true",
        help="Also run CFG scale sweep (w=1.0, 1.3, 1.6, 2.0) + Hit-to-Lead CVAE refinement.",
    )
    parser.add_argument(
        "--sweep-pockets",
        type=int,
        default=50,
        help="Number of test pockets for optional --cfg-sweep.",
    )
    parser.add_argument(
        "--resume-skip",
        action="store_true",
        help="Skip models whose eval_v3_<name>.json already exists.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load HF weights strictly from local cache.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only regenerate unified model_comparison_v3.{csv,md} from existing eval JSONs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for GenEuclidean v3.0 5-tier evaluation suite."""
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.comparison_only:
        csv_path, md_path = write_v3_comparison_tables(args.output_dir)
        print(f"Wrote unified v3 comparison tables -> {csv_path} & {md_path}", flush=True)
        return 0

    device = select_device(args.device)
    records, train_ids, test_ids = _load_cache_and_splits(
        args.cache_path, args.splits_path, args.split
    )

    runs = _discover_v3_runs(args.runs_dir, args.models)
    if not runs:
        print(f"No v3 adapter checkpoints found under {args.runs_dir}.", flush=True)

    for short_name, adapter_ckpt in runs:
        out_eval = args.output_dir / f"eval_v3_{short_name}.json"
        if args.resume_skip and out_eval.is_file():
            print(f"[v3:{short_name}] Skipping (already exists at {out_eval})", flush=True)
            continue
        evaluate_single_v3_model(
            short_name,
            adapter_ckpt,
            records=records,
            train_ids=train_ids,
            test_ids=test_ids,
            split_name=args.split,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            device=device,
            num_samples=args.num_samples,
            max_pockets=args.max_pockets,
            attention_pockets=args.attention_pockets,
            max_new_tokens=args.max_new_tokens,
            cfg_scale_override=args.cfg_scale,
            temperature_override=args.temperature,
            min_p_override=args.min_p,
            seed=args.seed,
            run_cfg_sweep=args.cfg_sweep,
            sweep_pockets=args.sweep_pockets,
            local_files_only=args.local_files_only,
        )

    csv_path, md_path = write_v3_comparison_tables(args.output_dir)
    print(f"Wrote unified v3 comparison tables -> {csv_path} & {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
