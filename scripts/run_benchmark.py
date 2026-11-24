#!/usr/bin/env python3
"""Comprehensive, resumable 5-tier SBDD benchmark driver for GenEuclidean v2.

Supports three evaluation modes (or ``--mode all``):
  - ``models``: Tier 1 (2D quality/diversity/FFD/W1/JSD, raw & MPO-reranked),
    Tier 2 (Causal Matched vs. Shuffled vs. Null pocket conditioning with Tanimoto &
    2021-thesis Count-Dice, Wilcoxon/MWU/Cliff's delta/bootstrap CIs), Tier 3 (3D
    cross-attention <4.5 A contact enrichment + residue-level heatmap payloads), and
    Tier 4 (full-split pKd affinity & DrugCLIP retrieval + pocket embeddings for t-SNE).
  - ``decoding``: High-variation decoding & CFG/CVAE Pareto frontier sweep (Greedy,
    Temperature, Top-p, Min-p, Locally Typical, Diverse Beam Search, CFG scale sweep,
    CVAE latent-temp sweep, and Hit-to-Lead CVAE seed refinement).
  - ``docking``: Batch 3D ETKDGv3 + MMFF94s conformer generation, native Apple Silicon
    AutoDock Vina v1.2.7 docking (top-K MPO hits + redocked native ligand), High-Affinity %,
    and PoseBusters-inspired physical checks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from geneuclidean.config import ExperimentConfig, SamplingConfig, seed_everything, select_device
from geneuclidean.data.featurizer import LIGAND_FP_DIM
from geneuclidean.data.tokenizer import MolecularTokenizer
from geneuclidean.evaluation import (
    dice_similarity,
    evaluate_3d_plausibility,
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
from geneuclidean.models.geneuclidean import GenEuclideanModel
from geneuclidean.sampling.decoder_strategies import sample_molecules, sample_with_attention
from geneuclidean.sampling.rerank import rerank_and_select

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

_MODEL_ORDER: tuple[str, ...] = (
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

_DISPLAY_NAMES: dict[str, str] = {
    "e3nn_lstm_v15": "GenEuclidean v1.5 (e3nn+LSTM)",
    "painn_pocketformer_scratch": "PaiNN-PocketFormer (Scratch)",
    "painn_pocketformer": "PaiNN-PocketFormer (Pretrained)",
    "geneuclidean_v2_schnet": "GenEuclidean v2 (SchNet)",
    "geneuclidean_v2_egnn": "GenEuclidean v2 (EGNN)",
    "geneuclidean_v2_e3nn": "GenEuclidean v2 (e3nn)",
    "geneuclidean_v2_selfies": "GenEuclidean v2 (SELFIES)",
    "geneuclidean_v2_painn": "GenEuclidean v2 (PaiNN-SE3)",
    "geneuclidean_v2_painn_lp": "GenEuclidean v2 (PaiNN-SE3, LP)",
    "geneuclidean_v2_painn_dpo": "GenEuclidean v2 (PaiNN-SE3+DPO)",
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


def _load_checkpoint_model(
    ckpt_path: Path,
    *,
    processed_dir: Path,
    device: torch.device,
) -> tuple[GenEuclideanModel, ExperimentConfig, MolecularTokenizer, int]:
    """Load trained ``GenEuclideanModel``, config, tokenizer, and parameter count."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain 'model_state_dict'")

    raw_cfg = payload.get("config")
    if isinstance(raw_cfg, dict):
        cfg = ExperimentConfig.model_validate(raw_cfg)
    else:
        cfg_yaml = ckpt_path.parent / "config.yaml"
        cfg = ExperimentConfig.from_yaml(cfg_yaml) if cfg_yaml.is_file() else ExperimentConfig()

    rep = cfg.data.representation
    vocab_candidates = [
        processed_dir / f"vocab_{rep}.json",
        Path(cfg.data.processed_dir) / f"vocab_{rep}.json",
        ckpt_path.parent / f"vocab_{rep}.json",
    ]
    vocab_path = next((p for p in vocab_candidates if p.is_file()), None)
    if vocab_path is None:
        raise FileNotFoundError(
            f"Could not find vocab_{rep}.json for {ckpt_path} in {vocab_candidates}"
        )

    tokenizer = MolecularTokenizer.load(vocab_path)
    cfg = cfg.model_copy(
        update={"decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})}
    )

    model = GenEuclideanModel(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, tokenizer, n_params


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
    """Aggregate token-to-atom attention ``[T, N_atoms]`` into interpretable pocket residues.

    Matches cached world-space atom coordinates to crystallographic PDB residue labels
    when the raw pocket PDB is available; otherwise falls back to residue-type one-hots
    from ``node_feats[:, 10:31]``.
    """
    n_tok, n_atoms = attn_map.shape
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
        # Fallback: decode residue type one-hot in node_feats[:, 10:31]
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

    # Group atom indices by unique residue label (preserving encounter order)
    res_to_atoms: dict[str, list[int]] = {}
    for atom_idx, lbl in enumerate(atom_res_labels):
        res_to_atoms.setdefault(lbl, []).append(atom_idx)

    res_labels = list(res_to_atoms.keys())
    res_attn = np.zeros((n_tok, len(res_labels)), dtype=np.float64)
    res_contact = np.zeros(len(res_labels), dtype=bool)
    for r_i, lbl in enumerate(res_labels):
        idxs = res_to_atoms[lbl]
        res_attn[:, r_i] = attn_map[:, idxs].sum(axis=1)
        res_contact[r_i] = bool(np.any(contact_atoms[idxs]))

    # Keep top-attended + contact residues if pocket is very large for crisp plotting
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

    # Renormalize rows over displayed residues
    row_sums = res_attn.sum(axis=1, keepdims=True)
    res_attn = np.divide(res_attn, row_sums, out=np.zeros_like(res_attn), where=row_sums > 1e-12)
    return res_labels, res_attn, res_contact


def _discover_model_runs(
    runs_dir: Path, explicit_models: Sequence[str] | None
) -> list[tuple[str, Path]]:
    """Resolve list of ``(run_name, checkpoint_path)`` pairs to evaluate."""
    discovered: list[tuple[str, Path]] = []
    if explicit_models:
        for item in explicit_models:
            cand = Path(item)
            if cand.is_file():
                name = cand.parent.name
                discovered.append((name, cand))
            elif (runs_dir / item / "best.pt").is_file():
                discovered.append((item, runs_dir / item / "best.pt"))
            elif (cand / "best.pt").is_file():
                discovered.append((cand.name, cand / "best.pt"))
            else:
                raise FileNotFoundError(f"Could not resolve model checkpoint for {item!r}")
        return discovered

    if not runs_dir.is_dir():
        return []

    found_map: dict[str, Path] = {}
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("prior_"):
            continue
        ckpt = entry / "best.pt"
        if ckpt.is_file():
            found_map[entry.name] = ckpt

    ordered: list[tuple[str, Path]] = []
    for name in _MODEL_ORDER:
        if name in found_map:
            ordered.append((name, found_map.pop(name)))
    for name in sorted(found_map.keys()):
        ordered.append((name, found_map[name]))
    return ordered


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


@torch.no_grad()
def _evaluate_tier4_full_split(
    model: GenEuclideanModel,
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
            n_a = c.shape[0]
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

    result: dict[str, Any] = {
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
    return result


def evaluate_single_model(
    model_name: str,
    ckpt_path: Path,
    *,
    records: Mapping[str, dict[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    split_name: str,
    processed_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    device: torch.device,
    num_samples: int,
    max_pockets: int,
    attention_pockets: int,
    seed: int,
) -> dict[str, Any]:
    """Run full 4-tier GPU/CPU evaluation for one trained model and persist artifacts."""
    t0 = time.perf_counter()
    seed_everything(seed)
    model, cfg, tokenizer, n_params = _load_checkpoint_model(
        ckpt_path, processed_dir=processed_dir, device=device
    )

    train_smiles = [str(records[pid]["canonical_smiles"]) for pid in train_ids if pid in records]
    test_native = {pid: str(records[pid]["canonical_smiles"]) for pid in test_ids if pid in records}
    sample_pids = list(test_ids[:max_pockets]) if max_pockets > 0 else list(test_ids)

    use_cfg = cfg.decoder.cfg_drop_prob > 0.0
    matched_cfg = SamplingConfig(
        strategy="min_p",
        num_samples=num_samples,
        max_len=min(120, cfg.decoder.max_seq_len),
        temperature=1.05,
        min_p=0.08,
        cfg_scale=1.75 if use_cfg else 1.0,
        latent_temp=1.0 if cfg.cvae.enabled else 0.0,
        repetition_penalty=1.05,
    )
    null_cfg = matched_cfg.model_copy(
        update={
            "cfg_scale": 0.0,
            "latent_temp": 0.0,
        }
    )

    per_pocket_matched: dict[str, list[str]] = {}
    per_pocket_null: dict[str, list[str]] = {}
    per_pocket_reranked: dict[str, list[dict[str, Any]]] = {}
    all_raw_matched: list[str] = []
    all_reranked_smiles: list[str] = []

    print(
        f"[{model_name}] Sampling {len(sample_pids)} pockets x {num_samples} mols "
        f"(strategy={matched_cfg.strategy}, cfg={matched_cfg.cfg_scale}, "
        f"cvae={cfg.cvae.enabled}) on {device}..."
    )

    for idx, pid in enumerate(sample_pids):
        p_batch = _record_to_pocket_batch(records[pid])
        gen_m = torch.Generator(device="cpu").manual_seed(seed + idx)
        gen_n = torch.Generator(device="cpu").manual_seed(seed + 100_000 + idx)

        matched_smis = sample_molecules(
            model, p_batch, tokenizer, matched_cfg, generator=gen_m, device=device
        )
        null_smis = sample_molecules(
            model, p_batch, tokenizer, null_cfg, generator=gen_n, device=device
        )
        per_pocket_matched[pid] = matched_smis
        per_pocket_null[pid] = null_smis
        all_raw_matched.extend(matched_smis)

        # Score matched candidates against pocket and select diverse top-10 via MPO+MaxMin
        with torch.no_grad():
            pocket_out = model.encode_pocket(
                p_batch["coords"].to(device),
                p_batch["node_feats"].to(device),
                p_batch["atom_mask"].to(device),
            )
            cand_fps = _smiles_to_fp_tensor(matched_smis, fp_dim=cfg.multitask.ligand_fp_dim)
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

    # Exact derangement shuffled-pocket control
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

    # Tier 3: 3D Cross-Attention Interpretability Audit
    attn_pids = sample_pids[: min(attention_pockets, len(sample_pids))]
    attn_cfg = matched_cfg.model_copy(update={"num_samples": 4, "latent_temp": 0.0})
    attn_records: list[dict[str, float]] = []
    attn_examples: list[dict[str, Any]] = []

    for idx, pid in enumerate(attn_pids):
        rec = records[pid]
        p_batch = _record_to_pocket_batch(rec)
        gen_a = torch.Generator(device="cpu").manual_seed(seed + 200_000 + idx)
        smis, tok_lists, attn_maps = sample_with_attention(
            model, p_batch, tokenizer, attn_cfg, generator=gen_a, device=device
        )
        atom_mask_np = p_batch["atom_mask"].squeeze(0).numpy()
        contact_mask_np = p_batch["contact_mask"].squeeze(0).numpy()
        for smi, toks, amap in zip(smis, tok_lists, attn_maps, strict=True):
            if amap is None or amap.numel() == 0 or len(toks) == 0:
                continue
            amap_np = amap.numpy()
            audit = evaluate_attention_enrichment(amap_np, atom_mask_np, contact_mask_np)
            if math.isfinite(audit["enrichment_ratio"]):
                attn_records.append(audit)
                if parse_mol(smi) is not None and len(toks) >= 8:
                    res_labels, res_attn, res_contact = _aggregate_atom_attention_to_residues(
                        rec, amap_np, raw_dir=raw_dir, max_residues=24
                    )
                    attn_examples.append(
                        {
                            "pdb_id": pid,
                            "smiles": smi,
                            "tokens": toks[:36],
                            "residue_labels": res_labels,
                            "attn_matrix": res_attn[:36, :].tolist(),
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
    emb_path = output_dir / f"pocket_embeddings_{model_name}.npz"
    if isinstance(emb_arr, np.ndarray) and emb_arr.size > 0:
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            emb_path,
            embeddings=emb_arr.astype(np.float32),
            pkd=np.asarray(tier4["pkd_y_true"], dtype=np.float32),
            pdb_ids=np.asarray(tier4["pdb_ids"]),
        )

    elapsed = time.perf_counter() - t0
    eval_payload: dict[str, Any] = {
        "model_name": model_name,
        "display_name": _DISPLAY_NAMES.get(model_name, model_name),
        "checkpoint": str(ckpt_path),
        "split": split_name,
        "num_params": n_params,
        "encoder_arch": cfg.encoder.arch,
        "decoder_arch": cfg.decoder.arch,
        "representation": cfg.data.representation,
        "cvae_enabled": cfg.cvae.enabled,
        "cfg_scale": matched_cfg.cfg_scale,
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

    samples_payload = {
        "model_name": model_name,
        "split": split_name,
        "native_smiles": {pid: test_native[pid] for pid in sample_pids if pid in test_native},
        "per_pocket_matched": per_pocket_matched,
        "per_pocket_reranked": per_pocket_reranked,
    }

    _write_json(output_dir / f"eval_{model_name}.json", eval_payload)
    _write_json(output_dir / f"samples_{model_name}.json", samples_payload)
    print(
        f"[{model_name}] Done in {elapsed:.1f}s | Valid={tier1_raw['validity']:.3f} "
        f"Uniq={tier1_raw['uniqueness']:.3f} IntDiv1={tier1_raw['int_div_1']:.3f} "
        f"Tan={tier2['matched_tanimoto_mean']:.3f} (Dice={tier2['matched_count_dice_mean']:.3f}) "
        f"pKd_r={(tier4['affinity']['pearson_r'] if tier4['affinity'] else float('nan')):.3f}"
    )
    return eval_payload


def _summary_row_from_eval(eval_data: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ``eval_<model>.json`` dictionary into a comparison table row."""
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


def _write_comparison_tables(output_dir: Path) -> Path:
    """Aggregate all ``eval_*.json`` files + 2021 Thesis Baseline into CSV and Markdown."""
    rows: list[dict[str, Any]] = [dict(THESIS_2021_BASELINE)]
    eval_files = {p.stem.removeprefix("eval_"): p for p in sorted(output_dir.glob("eval_*.json"))}

    ordered_names = [n for n in _MODEL_ORDER if n in eval_files] + [
        n for n in sorted(eval_files.keys()) if n not in _MODEL_ORDER
    ]
    for name in ordered_names:
        data = json.loads(eval_files[name].read_text(encoding="utf-8"))
        rows.append(_summary_row_from_eval(data))

    df = pd.DataFrame(rows)
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Render clean GitHub-flavored Markdown comparison table
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
    (output_dir / "model_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path


def run_decoding_sweep(
    ckpt_path: Path,
    *,
    records: Mapping[str, dict[str, Any]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    processed_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_pockets: int,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run Decoding, CFG, CVAE latent temperature, and Hit-to-Lead refinement Pareto sweep."""
    seed_everything(seed)
    model, cfg, tokenizer, _ = _load_checkpoint_model(
        ckpt_path, processed_dir=processed_dir, device=device
    )
    train_smiles = [str(records[pid]["canonical_smiles"]) for pid in train_ids if pid in records]
    pids = list(test_ids[:n_pockets]) if n_pockets > 0 else list(test_ids)
    native_map = {pid: str(records[pid]["canonical_smiles"]) for pid in pids if pid in records}
    max_len = min(120, cfg.decoder.max_seq_len)
    has_cvae = model.cvae is not None
    default_tz = 1.0 if has_cvae else 0.0

    sweep_specs: list[tuple[str, str, SamplingConfig, bool]] = [
        (
            "Greedy",
            "Greedy",
            SamplingConfig(
                strategy="greedy",
                num_samples=1,
                max_len=max_len,
                cfg_scale=1.0,
                latent_temp=0.0,
            ),
            False,
        ),
        (
            "Temp",
            "Temp T=0.8",
            SamplingConfig(
                strategy="temperature",
                num_samples=num_samples,
                max_len=max_len,
                temperature=0.8,
                cfg_scale=1.0,
                latent_temp=default_tz,
            ),
            False,
        ),
        (
            "Temp",
            "Temp T=1.0",
            SamplingConfig(
                strategy="temperature",
                num_samples=num_samples,
                max_len=max_len,
                temperature=1.0,
                cfg_scale=1.0,
                latent_temp=default_tz,
            ),
            False,
        ),
        (
            "Temp",
            "Temp T=1.2",
            SamplingConfig(
                strategy="temperature",
                num_samples=num_samples,
                max_len=max_len,
                temperature=1.2,
                cfg_scale=1.0,
                latent_temp=default_tz,
            ),
            False,
        ),
        (
            "Top-p",
            "Top-p 0.92",
            SamplingConfig(
                strategy="top_p",
                num_samples=num_samples,
                max_len=max_len,
                temperature=1.0,
                top_p=0.92,
                cfg_scale=1.0,
                latent_temp=default_tz,
            ),
            False,
        ),
        (
            "Typical",
            "Typical 0.90",
            SamplingConfig(
                strategy="typical",
                num_samples=num_samples,
                max_len=max_len,
                temperature=1.0,
                top_p=0.90,
                cfg_scale=1.0,
                latent_temp=default_tz,
            ),
            False,
        ),
        (
            "DBS",
            "DBS B=16 G=8",
            SamplingConfig(
                strategy="diverse_beam",
                num_samples=min(16, num_samples),
                max_len=max_len,
                beam_width=16,
                num_beam_groups=8,
                diversity_penalty=0.7,
                cfg_scale=1.5 if cfg.decoder.cfg_drop_prob > 0 else 1.0,
                latent_temp=0.0,
            ),
            False,
        ),
    ]

    # CFG scale sweep with Min-p=0.08
    for w in (1.0, 1.5, 1.75, 2.25):
        sweep_specs.append(
            (
                "Min-p+CFG",
                f"Min-p CFG w={w:g}",
                SamplingConfig(
                    strategy="min_p",
                    num_samples=num_samples,
                    max_len=max_len,
                    temperature=1.05,
                    min_p=0.08,
                    cfg_scale=w,
                    latent_temp=default_tz,
                    repetition_penalty=1.05,
                ),
                False,
            )
        )

    if has_cvae:
        for tz in (0.5, 1.0, 1.3):
            sweep_specs.append(
                (
                    "CVAE-Temp",
                    f"CVAE tau_z={tz:g}",
                    SamplingConfig(
                        strategy="min_p",
                        num_samples=num_samples,
                        max_len=max_len,
                        temperature=1.05,
                        min_p=0.08,
                        cfg_scale=1.75 if cfg.decoder.cfg_drop_prob > 0 else 1.0,
                        latent_temp=tz,
                        repetition_penalty=1.05,
                    ),
                    False,
                )
            )
        sweep_specs.append(
            (
                "CVAE-Refine",
                "Hit-to-Lead Refine",
                SamplingConfig(
                    strategy="min_p",
                    num_samples=num_samples,
                    max_len=max_len,
                    temperature=1.0,
                    min_p=0.08,
                    cfg_scale=1.5 if cfg.decoder.cfg_drop_prob > 0 else 1.0,
                    latent_temp=0.5,
                    repetition_penalty=1.05,
                ),
                True,
            )
        )

    results_list: list[dict[str, Any]] = []
    for s_idx, (family, label, scfg, use_seed_refine) in enumerate(sweep_specs):
        t0 = time.perf_counter()
        per_pocket_gen: dict[str, list[str]] = {}
        all_smis: list[str] = []
        pred_pkds: list[float] = []

        for p_idx, pid in enumerate(pids):
            p_batch = _record_to_pocket_batch(records[pid])
            gen_rng = torch.Generator(device="cpu").manual_seed(seed + 10_000 * s_idx + p_idx)
            seed_fp = p_batch["ligand_fp"] if use_seed_refine else None
            smis = sample_molecules(
                model,
                p_batch,
                tokenizer,
                scfg,
                seed_ligand_fp=seed_fp,
                generator=gen_rng,
                device=device,
            )
            per_pocket_gen[pid] = smis
            all_smis.extend(smis)

            valid_smis = [s for s in smis if parse_mol(s) is not None]
            if valid_smis and cfg.multitask.use_pkd_head:
                with torch.no_grad():
                    pout = model.encode_pocket(
                        p_batch["coords"].to(device),
                        p_batch["node_feats"].to(device),
                        p_batch["atom_mask"].to(device),
                    )
                    fps_t = _smiles_to_fp_tensor(valid_smis, fp_dim=cfg.multitask.ligand_fp_dim)
                    sc = model.score_candidates(pout, fps_t)
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
        elapsed = time.perf_counter() - t0
        row = {
            "strategy": family,
            "label": label,
            "use_seed_refine": use_seed_refine,
            "validity": t1["validity"],
            "uniqueness": t1["uniqueness"],
            "novelty": t1["novelty"],
            "int_div_1": t1["int_div_1"],
            "intra_int_div_1": t2["intra_int_div_1_mean"],
            "murcko_scaffold_diversity": t1["murcko_scaffold_diversity"],
            "QED": t1["QED_mean"],
            "SA_norm": t1["SA_norm_mean"],
            "tanimoto": t2["matched_tanimoto_mean"],
            "count_dice": t2["matched_count_dice_mean"],
            "pred_pkd": float(np.mean(pred_pkds)) if pred_pkds else float("nan"),
            "elapsed_sec": elapsed,
        }
        results_list.append(row)
        print(
            f"[decoding] {label:<20s} | Valid={row['validity']:.3f} "
            f"IntDiv1={row['int_div_1']:.3f} Tan={row['tanimoto']:.3f} "
            f"pKd={row['pred_pkd']:.2f} ({elapsed:.1f}s)"
        )

    payload = {
        "checkpoint": str(ckpt_path),
        "n_pockets": len(pids),
        "num_samples": num_samples,
        "results": results_list,
    }
    _write_json(output_dir / "decoding_ablation.json", payload)
    pd.DataFrame(results_list).to_csv(output_dir / "decoding_ablation.csv", index=False)
    return payload


def _dock_single_pocket_worker(
    task: tuple[str, list[str], str, str | None, str, int, int, int],
) -> dict[str, Any]:
    """Process-safe worker running ETKDGv3 + AutoDock Vina + PoseBusters on one pocket."""
    pid, smis, pocket_pdb, native_sdf, vina_bin, max_dock, exhaustiveness, seed = task
    res = evaluate_3d_plausibility(
        smiles_list=smis,
        pocket_pdb_path=pocket_pdb,
        native_ligand_sdf=native_sdf,
        run_vina=True,
        vina_bin=vina_bin,
        max_dock=max_dock,
        exhaustiveness=exhaustiveness,
        seed=seed,
    )
    res["pdb_id"] = pid
    return res


def run_docking_benchmark(
    samples_path: Path,
    *,
    records: Mapping[str, dict[str, Any]],
    raw_dir: Path,
    vina_bin: Path,
    output_dir: Path,
    split_name: str,
    n_pockets: int,
    top_k: int,
    exhaustiveness: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Run batch 3D conformer + native AutoDock Vina docking + PoseBusters audit."""
    samples_data = json.loads(samples_path.read_text(encoding="utf-8"))
    per_pocket_reranked: dict[str, list[dict[str, Any]]] = samples_data.get(
        "per_pocket_reranked", {}
    )
    per_pocket_matched: dict[str, list[str]] = samples_data.get("per_pocket_matched", {})

    # Collect pockets that have raw pocket PDB files on disk, stratified by pKd
    eligible: list[tuple[float, str, Path, Path | None, list[str]]] = []
    candidate_pids = sorted(set(per_pocket_reranked.keys()) | set(per_pocket_matched.keys()))
    for pid in candidate_pids:
        pocket_pdb, lig_sdf = _find_raw_complex_files(raw_dir, pid)
        if pocket_pdb is None:
            continue
        if pid in per_pocket_reranked and per_pocket_reranked[pid]:
            smis = [str(item["smiles"]) for item in per_pocket_reranked[pid][:top_k]]
        else:
            smis = [str(s) for s in per_pocket_matched.get(pid, []) if parse_mol(s) is not None][
                :top_k
            ]
        if not smis:
            continue
        pkd_val = float(records[pid]["pkd"]) if pid in records else 6.5
        eligible.append(
            (pkd_val if math.isfinite(pkd_val) else 6.5, pid, pocket_pdb, lig_sdf, smis)
        )

    eligible.sort(key=lambda item: item[0])
    if len(eligible) > n_pockets > 0:
        indices = np.linspace(0, len(eligible) - 1, num=n_pockets, dtype=int)
        selected = [eligible[int(i)] for i in indices]
    else:
        selected = eligible

    tasks = [
        (
            pid,
            smis,
            str(pocket_pdb),
            str(lig_sdf) if lig_sdf is not None else None,
            str(vina_bin),
            top_k,
            exhaustiveness,
            seed + idx,
        )
        for idx, (_pkd, pid, pocket_pdb, lig_sdf, smis) in enumerate(selected)
    ]

    print(
        f"[docking] Docking {len(tasks)} pockets x top-{top_k} ligands "
        f"(exhaustiveness={exhaustiveness}, workers={workers})..."
    )
    t0 = time.perf_counter()
    per_pocket_results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_dock_single_pocket_worker, tasks):
                per_pocket_results.append(res)
    else:
        for task in tasks:
            per_pocket_results.append(_dock_single_pocket_worker(task))

    all_vina: list[float] = []
    all_native_vina: list[float] = []
    high_aff_flags: list[float] = []
    strains: list[float] = []
    pb_passes: list[float] = []
    zero_clash_flags: list[float] = []

    for r in per_pocket_results:
        if r.get("vina_affinities"):
            all_vina.extend(float(v) for v in r["vina_affinities"])
        nat_v = r.get("native_vina_affinity")
        if nat_v is not None and math.isfinite(float(nat_v)):
            all_native_vina.append(float(nat_v))
            for v in r.get("vina_affinities", []):
                high_aff_flags.append(float(float(v) <= float(nat_v)))
        st = r.get("strain_energy_mean")
        if st is not None and math.isfinite(float(st)):
            strains.append(float(st))
        for d in r.get("docking_results", []):
            pb = d.get("posebusters")
            if pb is not None:
                pb_passes.append(float(bool(pb.get("passed_all"))))
                zero_clash_flags.append(float(int(pb.get("protein_clash_count", 1)) == 0))

    summary = {
        "samples_file": str(samples_path),
        "model_name": samples_data.get("model_name", samples_path.stem),
        "split": split_name,
        "n_pockets": len(per_pocket_results),
        "top_k_per_pocket": top_k,
        "exhaustiveness": exhaustiveness,
        "n_ligands_docked": len(all_vina),
        "vina_affinity_mean": float(np.mean(all_vina)) if all_vina else float("nan"),
        "vina_affinity_median": float(np.median(all_vina)) if all_vina else float("nan"),
        "vina_affinity_std": float(np.std(all_vina)) if all_vina else float("nan"),
        "vina_affinity_best": float(np.min(all_vina)) if all_vina else float("nan"),
        "native_vina_affinity_mean": (
            float(np.mean(all_native_vina)) if all_native_vina else float("nan")
        ),
        "high_affinity_rate": float(np.mean(high_aff_flags)) if high_aff_flags else float("nan"),
        "strain_energy_mean": float(np.mean(strains)) if strains else float("nan"),
        "zero_clash_rate": float(np.mean(zero_clash_flags)) if zero_clash_flags else float("nan"),
        "posebusters_pass_rate": float(np.mean(pb_passes)) if pb_passes else float("nan"),
        "elapsed_sec": time.perf_counter() - t0,
        "per_pocket": per_pocket_results,
    }
    out_path = output_dir / f"docking_{split_name}.json"
    _write_json(out_path, summary)
    print(
        f"[docking] Done in {summary['elapsed_sec']:.1f}s | "
        f"Vina mean={summary['vina_affinity_mean']:.2f} kcal/mol "
        f"(native={summary['native_vina_affinity_mean']:.2f}) | "
        f"HighAff={summary['high_affinity_rate']:.1%} | "
        f"PB-Pass={summary['posebusters_pass_rate']:.1%}"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the 5-tier benchmark driver."""
    parser = argparse.ArgumentParser(
        description="GenEuclidean v2 — 5-Tier SBDD Evaluation, Decoding Sweep & Vina Docking Driver"
    )
    parser.add_argument(
        "--mode",
        choices=("models", "decoding", "docking", "all"),
        default="models",
        help="Evaluation workflow to execute.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Trained runs root.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional run names or checkpoint paths for --mode models (default: auto-discover).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for --mode decoding (defaults to flagship PaiNN run).",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=None,
        help="Samples JSON path for --mode docking (defaults to flagship samples JSON).",
    )
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
    parser.add_argument(
        "--resume-skip",
        action="store_true",
        help="Skip models/modes whose output JSON files already exist.",
    )
    parser.add_argument("--num-samples", type=int, default=32, help="Samples per pocket (K).")
    parser.add_argument(
        "--max-pockets",
        type=int,
        default=100,
        help="Max test pockets to sample in --mode models (0 = all test pockets).",
    )
    parser.add_argument(
        "--attention-pockets",
        type=int,
        default=30,
        help="Number of test pockets for Tier 3 attention interpretability audit.",
    )
    parser.add_argument(
        "--decoding-pockets",
        type=int,
        default=50,
        help="Number of test pockets for --mode decoding Pareto sweep.",
    )
    parser.add_argument(
        "--dock-pockets",
        type=int,
        default=30,
        help="Number of test pockets for --mode docking.",
    )
    parser.add_argument(
        "--dock-top-k",
        type=int,
        default=5,
        help="Top-K MPO molecules docked per pocket in --mode docking.",
    )
    parser.add_argument(
        "--dock-exhaustiveness",
        type=int,
        default=4,
        help="AutoDock Vina exhaustiveness.",
    )
    parser.add_argument(
        "--dock-workers",
        type=int,
        default=3,
        help="Parallel CPU worker processes for AutoDock Vina docking.",
    )
    parser.add_argument(
        "--vina-bin",
        type=Path,
        default=Path("bin/vina"),
        help="Path to native AutoDock Vina executable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for GenEuclidean v2 benchmark suite."""
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    processed_dir = args.cache_path.parent

    records, train_ids, test_ids = _load_cache_and_splits(
        args.cache_path, args.splits_path, args.split
    )

    modes = ("models", "decoding", "docking") if args.mode == "all" else (args.mode,)

    if "models" in modes:
        runs = _discover_model_runs(args.runs_dir, args.models)
        if not runs:
            print(f"No model checkpoints found under {args.runs_dir}.")
        for name, ckpt in runs:
            out_eval = args.output_dir / f"eval_{name}.json"
            if args.resume_skip and out_eval.is_file():
                print(f"[{name}] Skipping (already exists at {out_eval})")
                continue
            evaluate_single_model(
                name,
                ckpt,
                records=records,
                train_ids=train_ids,
                test_ids=test_ids,
                split_name=args.split,
                processed_dir=processed_dir,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                device=device,
                num_samples=args.num_samples,
                max_pockets=args.max_pockets,
                attention_pockets=args.attention_pockets,
                seed=args.seed,
            )
        csv_path = _write_comparison_tables(args.output_dir)
        print(f"Wrote combined comparison table -> {csv_path}")

    if "decoding" in modes:
        out_dec = args.output_dir / "decoding_ablation.json"
        if args.resume_skip and out_dec.is_file():
            print(f"[decoding] Skipping (already exists at {out_dec})")
        else:
            ckpt = args.checkpoint
            if ckpt is None:
                default_flagship = args.runs_dir / "geneuclidean_v2_painn" / "best.pt"
                if default_flagship.is_file():
                    ckpt = default_flagship
                else:
                    runs = _discover_model_runs(args.runs_dir, args.models)
                    ckpt = runs[-1][1] if runs else None
            if ckpt is None or not ckpt.is_file():
                raise FileNotFoundError(
                    "No checkpoint found for --mode decoding; pass --checkpoint"
                )
            run_decoding_sweep(
                ckpt,
                records=records,
                train_ids=train_ids,
                test_ids=test_ids,
                processed_dir=processed_dir,
                output_dir=args.output_dir,
                device=device,
                n_pockets=args.decoding_pockets,
                num_samples=args.num_samples,
                seed=args.seed,
            )

    if "docking" in modes:
        out_dock = args.output_dir / f"docking_{args.split}.json"
        if args.resume_skip and out_dock.is_file():
            print(f"[docking] Skipping (already exists at {out_dock})")
        else:
            s_path = args.samples_file
            if s_path is None:
                for cand_name in (
                    "samples_geneuclidean_v2_painn_dpo.json",
                    "samples_geneuclidean_v2_painn.json",
                ):
                    cand = args.output_dir / cand_name
                    if cand.is_file():
                        s_path = cand
                        break
                if s_path is None:
                    avail = sorted(args.output_dir.glob("samples_*.json"))
                    s_path = avail[-1] if avail else None
            if s_path is None or not s_path.is_file():
                raise FileNotFoundError(
                    "No samples JSON found for --mode docking; "
                    "run --mode models first or pass --samples-file"
                )
            run_docking_benchmark(
                s_path,
                records=records,
                raw_dir=args.raw_dir,
                vina_bin=args.vina_bin,
                output_dir=args.output_dir,
                split_name=args.split,
                n_pockets=args.dock_pockets,
                top_k=args.dock_top_k,
                exhaustiveness=args.dock_exhaustiveness,
                workers=args.dock_workers,
                seed=args.seed,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
