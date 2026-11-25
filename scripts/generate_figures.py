#!/usr/bin/env python3
"""Render all 9 publication-grade 300-DPI figures for GenEuclidean v2.

Reads training histories from ``runs/*/history.jsonl`` and benchmark artifacts from
``results/tables/``, producing:
  1. ``01_model_comparison.png``
  2. ``02_training_curves.png``
  3. ``03_matched_vs_shuffled_specificity.png``
  4. ``04_property_distributions.png``
  5. ``05_pkd_parity_casf2016.png``
  6. ``06_pocket_attention_heatmap.png``
  7. ``07_generated_molecules_grid.png``
  8. ``08_decoding_pareto.png``
  9. ``09_pocket_latent_tsne.png``
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geneuclidean.evaluation import (
    compute_properties_batch,
    plot_decoding_pareto,
    plot_generated_molecule_grid,
    plot_matched_vs_shuffled_similarity,
    plot_model_comparison_bars,
    plot_pkd_parity,
    plot_pocket_attention_heatmap,
    plot_pocket_embedding_tsne,
    plot_property_distributions,
    plot_training_curves,
)

_DISPLAY_NAMES: dict[str, str] = {
    "pretrain_zinc": "Stage-1 Prior (SMILES)",
    "pretrain_zinc_selfies": "Stage-1 Prior (SELFIES)",
    "e3nn_lstm_v15": "v1.5 (e3nn+LSTM)",
    "painn_pocketformer_scratch": "PocketFormer (Scratch)",
    "painn_pocketformer": "PaiNN-PocketFormer",
    "geneuclidean_v2_schnet": "v2 (SchNet)",
    "geneuclidean_v2_egnn": "v2 (EGNN)",
    "geneuclidean_v2_e3nn": "v2 (e3nn)",
    "geneuclidean_v2_selfies": "v2 (SELFIES)",
    "geneuclidean_v2_painn": "v2 (PaiNN-SE3)",
    "geneuclidean_v2_painn_lp": "v2 (PaiNN-SE3, LP)",
    "geneuclidean_v2_painn_dpo": "v2 (PaiNN-SE3+DPO)",
}

_CURVE_ORDER: tuple[str, ...] = (
    "pretrain_zinc",
    "pretrain_zinc_selfies",
    "painn_pocketformer",
    "geneuclidean_v2_schnet",
    "geneuclidean_v2_egnn",
    "geneuclidean_v2_selfies",
    "geneuclidean_v2_painn",
)

_PROP_OVERLAY_PRIORITY: tuple[str, ...] = (
    "e3nn_lstm_v15",
    "painn_pocketformer",
    "geneuclidean_v2_painn",
    "geneuclidean_v2_painn_dpo",
    "geneuclidean_v2_e3nn",
    "geneuclidean_v2_selfies",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a ``.jsonl`` file, deduplicating repeated epochs."""
    rows: list[dict[str, Any]] = []
    by_epoch: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        obj = json.loads(cleaned)
        ep = obj.get("epoch")
        if isinstance(ep, int):
            if ep in by_epoch:
                rows[by_epoch[ep]] = obj
            else:
                by_epoch[ep] = len(rows)
                rows.append(obj)
        else:
            rows.append(obj)
    return rows


def _resolve_flagship_eval(tables_dir: Path, flagship: str) -> tuple[str, Path] | None:
    """Locate flagship ``eval_<name>.json`` or gracefully fall back to best available."""
    preferred = [
        flagship,
        "geneuclidean_v2_painn_dpo",
        "geneuclidean_v2_painn",
        "geneuclidean_v2_e3nn",
        "painn_pocketformer",
    ]
    for name in preferred:
        cand = tables_dir / f"eval_{name}.json"
        if cand.is_file():
            return name, cand
    avail = sorted(tables_dir.glob("eval_*.json"))
    if avail:
        return avail[-1].stem.removeprefix("eval_"), avail[-1]
    return None


def _select_showcase_pocket(
    samples_data: dict[str, Any],
    docking_data: dict[str, Any] | None,
) -> tuple[str, str, list[str], list[dict[str, Any]]] | None:
    """Pick the richest showcase pocket (preferring real Vina scores + diverse high-MPO hits)."""
    native_map: dict[str, str] = samples_data.get("native_smiles", {})
    reranked_map: dict[str, list[dict[str, Any]]] = samples_data.get("per_pocket_reranked", {})

    vina_lookup: dict[str, dict[str, float]] = {}
    if docking_data is not None:
        for pocket_entry in docking_data.get("per_pocket", []):
            pid = str(pocket_entry.get("pdb_id", ""))
            for dres in pocket_entry.get("docking_results", []):
                if dres.get("success") and dres.get("vina_affinity") is not None:
                    vina_lookup.setdefault(pid, {})[str(dres["smiles"])] = float(
                        dres["vina_affinity"]
                    )

    best_pid: str | None = None
    best_score = -1e9
    for pid, hits in reranked_map.items():
        if pid not in native_map or len(hits) < 3:
            continue
        has_vina = 1.0 if pid in vina_lookup and len(vina_lookup[pid]) >= 2 else 0.0
        mean_mpo = float(np.mean([float(h.get("mpo_score") or 0.0) for h in hits[:7]]))
        mean_tan = float(
            np.mean(
                [
                    float(h["tanimoto"])
                    for h in hits[:7]
                    if h.get("tanimoto") is not None and math.isfinite(float(h["tanimoto"]))
                ]
                or [0.0]
            )
        )
        score = 2.0 * has_vina + mean_mpo + mean_tan
        if score > best_score:
            best_score = score
            best_pid = pid

    if best_pid is None:
        return None

    hits = reranked_map[best_pid][:7]
    gen_smiles: list[str] = []
    annotations: list[dict[str, Any]] = []
    p_vina = vina_lookup.get(best_pid, {})
    for h in hits:
        smi = str(h["smiles"])
        gen_smiles.append(smi)
        ann: dict[str, Any] = {}
        if h.get("qed") is not None:
            ann["QED"] = float(h["qed"])
        if h.get("sa_score") is not None:
            ann["SA"] = float(h["sa_score"])
        if h.get("tanimoto") is not None:
            ann["Tanimoto"] = float(h["tanimoto"])
        if h.get("pred_pkd") is not None:
            ann["pKd"] = float(h["pred_pkd"])
        if smi in p_vina:
            ann["Vina"] = p_vina[smi]
        annotations.append(ann)

    return best_pid, str(native_map[best_pid]), gen_smiles, annotations


def generate_all_figures(
    *,
    runs_dir: Path,
    tables_dir: Path,
    output_dir: Path,
    flagship: str = "geneuclidean_v2_painn",
    split_name: str = "casf2016",
) -> dict[str, Path]:
    """Render all available publication figures and return mapping of figure ID -> Path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 1. Model comparison multi-panel bar chart
    comp_csv = tables_dir / "model_comparison.csv"
    if comp_csv.is_file():
        df = pd.read_csv(comp_csv)
        out = plot_model_comparison_bars(df, output_dir / "01_model_comparison.png")
        generated["01_model_comparison"] = out
        print(f"[fig 01] Rendered -> {out}")
    else:
        print(f"[fig 01] Skipped (missing {comp_csv})")

    # 2. Multi-model training curves from runs/*/history.jsonl
    _stage2_order: tuple[str, ...] = (
        "geneuclidean_v2_painn",
        "geneuclidean_v2_egnn",
        "geneuclidean_v2_schnet",
        "geneuclidean_v2_selfies",
        "painn_pocketformer",
        "geneuclidean_v2_e3nn",
        "e3nn_lstm_v15",
    )
    histories: dict[str, list[dict[str, Any]]] = {}
    if runs_dir.is_dir():
        run_entries = {
            e.name: e
            for e in runs_dir.iterdir()
            if e.is_dir()
            and not e.name.startswith(("prior_", "pretrain_"))
            and not e.name.endswith("_dpo")
        }
        ordered_names = [n for n in _stage2_order if n in run_entries] + sorted(
            n for n in run_entries if n not in _stage2_order
        )
        for name in ordered_names:
            hist_path = run_entries[name] / "history.jsonl"
            if hist_path.is_file():
                rows = _read_jsonl(hist_path)
                if len(rows) >= 2:
                    label = _DISPLAY_NAMES.get(name, name)
                    histories[label] = rows
    if histories:
        out = plot_training_curves(histories, output_dir / "02_training_curves.png")
        generated["02_training_curves"] = out
        print(f"[fig 02] Rendered ({len(histories)} runs) -> {out}")
    else:
        print(f"[fig 02] Skipped (no Stage-2 history.jsonl files under {runs_dir})")

    flagship_found = _resolve_flagship_eval(tables_dir, flagship)
    flagship_name: str | None = flagship_found[0] if flagship_found else None
    flagship_eval: dict[str, Any] | None = (
        json.loads(flagship_found[1].read_text(encoding="utf-8")) if flagship_found else None
    )

    # 3. Matched vs. Shuffled vs. Null pocket similarity
    if flagship_eval is not None:
        t2 = flagship_eval.get("tier2_specificity", {})
        matched_sims = t2.get("pooled_matched_tanimoto") or []
        shuffled_sims = t2.get("pooled_shuffled_tanimoto") or []
        null_sims = t2.get("pooled_null_tanimoto") or None
        if matched_sims and shuffled_sims:
            out = plot_matched_vs_shuffled_similarity(
                matched_sims,
                shuffled_sims,
                null_sims,
                output_dir / "03_matched_vs_shuffled_specificity.png",
            )
            generated["03_matched_vs_shuffled_specificity"] = out
            print(f"[fig 03] Rendered ({flagship_name}) -> {out}")
        else:
            print("[fig 03] Skipped (missing pooled matched/shuffled similarities)")
    else:
        print("[fig 03] Skipped (no eval_*.json found)")

    # 4. 6-panel physicochemical property distributions (+ Wasserstein-1)
    sample_files = {
        p.stem.removeprefix("samples_"): p for p in sorted(tables_dir.glob("samples_*.json"))
    }
    if sample_files:
        chosen_models = [m for m in _PROP_OVERLAY_PRIORITY if m in sample_files][:4]
        if not chosen_models:
            chosen_models = sorted(sample_files.keys())[:4]

        ref_smiles_list: list[str] = []
        gen_props_dict: dict[str, dict[str, np.ndarray]] = {}
        for m_name in chosen_models:
            s_data = json.loads(sample_files[m_name].read_text(encoding="utf-8"))
            if not ref_smiles_list and s_data.get("native_smiles"):
                ref_smiles_list = list(s_data["native_smiles"].values())
            flat_gen = [
                smi
                for smi_list in s_data.get("per_pocket_matched", {}).values()
                for smi in smi_list
            ]
            if flat_gen:
                gen_props_dict[_DISPLAY_NAMES.get(m_name, m_name)] = compute_properties_batch(
                    flat_gen
                )

        if ref_smiles_list and gen_props_dict:
            ref_props = compute_properties_batch(ref_smiles_list)
            out = plot_property_distributions(
                ref_props, gen_props_dict, output_dir / "04_property_distributions.png"
            )
            generated["04_property_distributions"] = out
            print(f"[fig 04] Rendered ({len(gen_props_dict)} models) -> {out}")
        else:
            print("[fig 04] Skipped (no valid generated/reference SMILES in samples_*.json)")
    else:
        print("[fig 04] Skipped (no samples_*.json found)")

    # 5. pKd parity scatter plot on CASF-2016
    if flagship_eval is not None:
        t4 = flagship_eval.get("tier4_affinity_clip", {})
        yt = t4.get("pkd_y_true") or []
        yp = t4.get("pkd_y_pred") or []
        if yt and yp and len(yt) == len(yp):
            raw_split = str(flagship_eval.get("split", split_name)).lower()
            split_label = "CASF-2016" if raw_split == "casf2016" else raw_split.upper()
            model_label = _DISPLAY_NAMES.get(flagship_name or "", flagship_name or "Flagship")
            out = plot_pkd_parity(
                yt,
                yp,
                f"{split_label} · {model_label}",
                output_dir / f"05_pkd_parity_{split_name}.png",
            )
            generated["05_pkd_parity"] = out
            print(f"[fig 05] Rendered -> {out}")
        else:
            print("[fig 05] Skipped (flagship run has no pKd predictions)")
    else:
        print("[fig 05] Skipped (no eval_*.json found)")

    # 6. 3D Pocket-Residue vs. SMILES-Token Cross-Attention Heatmap
    if flagship_eval is not None:
        examples = flagship_eval.get("tier3_attention", {}).get("examples") or []
        if examples:
            ex = examples[0]
            out = plot_pocket_attention_heatmap(
                tokens=ex["tokens"],
                residue_labels=ex["residue_labels"],
                attn_matrix=np.asarray(ex["attn_matrix"], dtype=np.float64),
                contact_mask=np.asarray(ex["contact_mask"], dtype=bool),
                pdb_id=str(ex["pdb_id"]),
                out_path=output_dir / "06_pocket_attention_heatmap.png",
            )
            generated["06_pocket_attention_heatmap"] = out
            print(f"[fig 06] Rendered ({ex['pdb_id']}) -> {out}")
        else:
            print("[fig 06] Skipped (no attention examples stored in flagship eval JSON)")
    else:
        print("[fig 06] Skipped (no eval_*.json found)")

    # 7. Generated Molecule Grid (native crystal ligand + top diverse generated hits)
    dock_path = tables_dir / f"docking_{split_name}.json"
    docking_data = (
        json.loads(dock_path.read_text(encoding="utf-8")) if dock_path.is_file() else None
    )
    grid_samples_path = (
        tables_dir / f"samples_{flagship_name}.json"
        if flagship_name and (tables_dir / f"samples_{flagship_name}.json").is_file()
        else (next(iter(sample_files.values())) if sample_files else None)
    )
    if grid_samples_path is not None and grid_samples_path.is_file():
        s_data = json.loads(grid_samples_path.read_text(encoding="utf-8"))
        showcase = _select_showcase_pocket(s_data, docking_data)
        if showcase is not None:
            pid, nat_smi, gen_smis, anns = showcase
            out = plot_generated_molecule_grid(
                native_smiles=nat_smi,
                gen_smiles_list=gen_smis,
                annotations=anns,
                pdb_id=pid,
                out_path=output_dir / "07_generated_molecules_grid.png",
            )
            generated["07_generated_molecules_grid"] = out
            print(f"[fig 07] Rendered ({pid}) -> {out}")
        else:
            print("[fig 07] Skipped (no showcase pocket with >=3 reranked hits)")
    else:
        print("[fig 07] Skipped (no samples JSON found)")

    # 8. Decoding & CFG Pareto Frontier
    dec_path = tables_dir / "decoding_ablation.json"
    if dec_path.is_file():
        dec_data = json.loads(dec_path.read_text(encoding="utf-8"))
        rows = dec_data.get("results", dec_data) if isinstance(dec_data, dict) else dec_data
        if rows:
            out = plot_decoding_pareto(rows, output_dir / "08_decoding_pareto.png")
            generated["08_decoding_pareto"] = out
            print(f"[fig 08] Rendered -> {out}")
        else:
            print("[fig 08] Skipped (empty decoding_ablation.json)")
    else:
        print(f"[fig 08] Skipped (missing {dec_path})")

    # 9. 3D Equivariant Pocket Latent Space t-SNE Colored by pKd
    emb_path: Path | None = None
    if flagship_eval is not None and flagship_eval.get("pocket_embeddings_path"):
        cand = Path(str(flagship_eval["pocket_embeddings_path"]))
        if cand.is_file():
            emb_path = cand
    if emb_path is None and flagship_name:
        cand = tables_dir / f"pocket_embeddings_{flagship_name}.npz"
        if cand.is_file():
            emb_path = cand
    if emb_path is None:
        avail_npz = sorted(tables_dir.glob("pocket_embeddings_*.npz"))
        if avail_npz:
            emb_path = avail_npz[-1]

    if emb_path is not None and emb_path.is_file():
        npz = np.load(emb_path)
        embs = npz["embeddings"]
        pkds = npz["pkd"]
        valid_mask = np.isfinite(pkds)
        if int(valid_mask.sum()) >= 3:
            out = plot_pocket_embedding_tsne(
                embs[valid_mask],
                pkds[valid_mask],
                output_dir / "09_pocket_latent_tsne.png",
                label_name="Experimental Binding Affinity (pKd)",
            )
            generated["09_pocket_latent_tsne"] = out
            print(f"[fig 09] Rendered ({int(valid_mask.sum())} pockets) -> {out}")
        else:
            print("[fig 09] Skipped (<3 finite pKd pocket embeddings)")
    else:
        print("[fig 09] Skipped (no pocket_embeddings_*.npz found)")

    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for publication figure generation."""
    parser = argparse.ArgumentParser(
        description="GenEuclidean v2 — Render all 9 publication-grade 300-DPI figures"
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Runs directory.")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Benchmark tables & JSON artifacts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Destination directory for 300-DPI PNG figures.",
    )
    parser.add_argument(
        "--flagship",
        default="geneuclidean_v2_painn",
        help="Primary flagship run name for single-model figures.",
    )
    parser.add_argument(
        "--split",
        default="casf2016",
        help="Split suffix used for pKd parity & docking filenames.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for rendering all GenEuclidean v2 figures."""
    args = build_arg_parser().parse_args(argv)
    figs = generate_all_figures(
        runs_dir=args.runs_dir,
        tables_dir=args.tables_dir,
        output_dir=args.output_dir,
        flagship=args.flagship,
        split_name=args.split,
    )
    print(f"Generated {len(figs)}/9 figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
