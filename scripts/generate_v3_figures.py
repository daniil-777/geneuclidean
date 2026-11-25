#!/usr/bin/env python3
"""Render publication-grade 300-DPI Okabe-Ito comparison & docking figures for GenEuclidean v3.0.

Reads available evaluation JSONs, CFG/decoding sweep tables, and AutoDock Vina v1.2.7
docking summaries from ``results/tables/`` (gracefully tolerating missing optional files
so figures can render incrementally during training sweeps) and writes:

  - ``results/figures/10_v3_llm_comparison.png``: 4-panel progression across 2021 Thesis ->
    v2.0 PaiNN-SE3 -> v2.0 RPO-DPO -> v3.0 GP-MoLFormer (Base & RPO-DPO) -> v3.0 GPT2-ZINC ->
    v3.0 Qwen2.5-0.5B on Raw Validity/QED/SA_norm/IntDiv1, Matched vs. Shuffled Count Dice &
    True Tanimoto, Cross-Pocket MW/TPSA/LogP Pearson r, and <4.5 A Contact Attention Enrichment.
  - ``results/figures/11_v3_cfg_and_docking.png``: 2-panel figure showing (A) v3.0 CFG scale
    (w=1.0..2.0) + Hit-to-Lead CVAE Pareto frontier and (B) 30-pocket AutoDock Vina v1.2.7
    Top-1 affinity, High-Affinity %, and PoseBusters Pass-All progression.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Colorblind-safe Okabe-Ito palette
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "black": "#1A1A1A",
    "gray": "#7F7F7F",
}

_DPI = 300

# Verified 2021 ETH Zurich thesis baseline (Report_denovo.pdf Tables 4.3, 4.4 & Fig. A.1)
_THESIS_2021_RECORD: dict[str, Any] = {
    "short_label": "2021\nThesis\n(e3nn+LSTM)",
    "validity": 0.650,
    "qed": 0.450,
    "sa_norm": (10.0 - 4.00) / 9.0,
    "int_div_1": float("nan"),
    "matched_dice": 0.290,
    "shuffled_dice": float("nan"),
    "matched_tan": 0.170,
    "shuffled_tan": 0.100,
    "cliffs_delta": float("nan"),
    "mw_r": float("nan"),
    "tpsa_r": float("nan"),
    "logp_r": float("nan"),
    "attn_enrichment": float("nan"),
}

_MODEL_SLOTS: tuple[tuple[str, str], ...] = (
    ("eval_geneuclidean_v2_painn.json", "v2.0\nPaiNN-SE3"),
    ("eval_geneuclidean_v2_painn_dpo.json", "v2.0\nRPO-DPO"),
    ("eval_v3_gpmolformer_iter1_rand.json", "v3.0\nGP-MoLF\n(Iter1 Rand)"),
    ("eval_v3_gpmolformer.json", "v3.0\nGP-MoLF\n(Canon)"),
    ("eval_v3_gpmolformer_dpo.json", "v3.0\nGP-MoLF\n+ RPO-DPO"),
    ("eval_v3_gpt2zinc.json", "v3.0\nGPT2-ZINC"),
    ("eval_v3_qwen25.json", "v3.0\nQwen2.5\n0.5B"),
)

_SWEEP_TITLE_MAP: dict[str, str] = {
    "geneuclidean_v3_gpmolformer": "v3.0 GP-MoLFormer-Uniq",
    "geneuclidean_v3_gpmolformer_dpo": "v3.0 GP-MoLFormer + RPO-DPO",
    "geneuclidean_v3_gpt2zinc": "v3.0 GPT2-ZINC-87M",
    "geneuclidean_v3_qwen25": "v3.0 Qwen2.5-0.5B",
}


def _apply_pub_style() -> None:
    """Apply clean publication typography and subtle axes styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_fig(fig: Figure, out_path: Path) -> Path:
    """Save figure at 300 DPI and release matplotlib memory."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _safe_float(val: Any) -> float:
    """Coerce value to finite float or NaN."""
    if val is None:
        return float("nan")
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _extract_model_record(eval_data: Mapping[str, Any], short_label: str) -> dict[str, Any]:
    """Extract normalized multi-tier metrics from an ``eval_*.json`` payload."""
    t1 = eval_data.get("tier1_raw") or {}
    t2 = eval_data.get("tier2_specificity") or {}
    t3 = eval_data.get("tier3_attention") or {}
    vs_shuf = t2.get("vs_shuffled") or {}
    props = t2.get("property_correlations") or {}

    return {
        "short_label": short_label,
        "validity": _safe_float(t1.get("validity")),
        "qed": _safe_float(t1.get("QED_mean")),
        "sa_norm": _safe_float(t1.get("SA_norm_mean")),
        "int_div_1": _safe_float(t1.get("int_div_1")),
        "matched_dice": _safe_float(t2.get("matched_count_dice_mean")),
        "shuffled_dice": _safe_float(vs_shuf.get("control_count_dice_mean")),
        "matched_tan": _safe_float(t2.get("matched_tanimoto_mean")),
        "shuffled_tan": _safe_float(vs_shuf.get("control_tanimoto_mean")),
        "cliffs_delta": _safe_float(vs_shuf.get("cliffs_delta")),
        "mw_r": _safe_float((props.get("MW") or {}).get("pearson_r")),
        "tpsa_r": _safe_float((props.get("TPSA") or {}).get("pearson_r")),
        "logp_r": _safe_float((props.get("LogP") or {}).get("pearson_r")),
        "attn_enrichment": _safe_float(t3.get("enrichment_ratio_mean")),
    }


def _load_comparison_records(tables_dir: Path) -> list[dict[str, Any]]:
    """Load 2021 thesis baseline + all available v2/v3 evaluation JSONs in canonical order."""
    records: list[dict[str, Any]] = [dict(_THESIS_2021_RECORD)]
    seen_files: set[str] = set()

    for fname, label in _MODEL_SLOTS:
        path = tables_dir / fname
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(_extract_model_record(data, label))
            seen_files.add(fname)

    # Gracefully include any extra v3 eval JSONs (e.g., Qwen2.5-1.5B, Qwen3-0.6B, SmolLM2)
    for path in sorted(tables_dir.glob("eval_v3_*.json")):
        if path.name in seen_files:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tag = path.stem.removeprefix("eval_v3_")
        records.append(_extract_model_record(data, f"v3.0\n{tag}"))

    return records


def _annotate_bars(
    ax: Axes,
    bars: Any,
    fmt: str = "{:.2f}",
    dy: float = 0.015,
    fontsize: float = 7.2,
) -> None:
    """Place crisp numeric labels above finite positive bars."""
    for bar in bars:
        height = float(bar.get_height())
        if not math.isfinite(height) or height <= 0.0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def _annotate_grouped_bars(
    ax: Axes,
    bar_groups: Sequence[Any],
    fmt: str = "{:.2f}",
    dy: float = 0.012,
    stagger_thresh: float = 0.045,
    stagger_step: float = 0.036,
    fontsize: float = 6.6,
) -> None:
    """Annotate grouped bars while vertically staggering close adjacent labels."""
    if not bar_groups:
        return
    n_slots = len(bar_groups[0])
    for slot_idx in range(n_slots):
        prev_y: float | None = None
        for grp in bar_groups:
            bar = grp[slot_idx]
            height = float(bar.get_height())
            if not math.isfinite(height) or height <= 0.0:
                prev_y = None
                continue
            y_pos = height + dy
            if prev_y is not None and abs(y_pos - prev_y) < stagger_thresh:
                y_pos = max(y_pos, prev_y + stagger_step)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#222222",
            )
            prev_y = y_pos


def plot_v3_llm_comparison(tables_dir: Path, out_path: Path) -> Path:
    """Render ``10_v3_llm_comparison.png`` across Thesis -> v2.0 -> v3.0 LLM models."""
    _apply_pub_style()
    records = _load_comparison_records(tables_dir)
    labels = [r["short_label"] for r in records]
    x = np.arange(len(records), dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.2), constrained_layout=True)
    fig.suptitle(
        "GenEuclidean v3.0 — Chemical Foundation Transformers & Local LLMs "
        "vs. v2.0 & 2021 Thesis (CASF-2016)",
        fontsize=14.0,
        fontweight="bold",
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(h_pad=0.12, w_pad=0.10, hspace=0.06, wspace=0.04)

    # --- Panel A: Chemical Quality & Diversity ---
    ax_a = axes[0, 0]
    w_a = 0.20
    metrics_a = [
        ("validity", "Raw Validity", OKABE_ITO["blue"]),
        ("qed", "Raw QED", OKABE_ITO["green"]),
        ("sa_norm", "SA_norm", OKABE_ITO["orange"]),
        ("int_div_1", "IntDiv1", OKABE_ITO["purple"]),
    ]
    bar_groups_a: list[Any] = []
    for offset_idx, (key, title, color) in enumerate(metrics_a):
        vals = np.array([r[key] for r in records], dtype=np.float64)
        plot_vals = np.where(np.isfinite(vals), vals, 0.0)
        shift = (offset_idx - 1.5) * w_a
        bars = ax_a.bar(
            x + shift, plot_vals, width=w_a, label=title, color=color, edgecolor="white"
        )
        bar_groups_a.append(bars)
    _annotate_grouped_bars(
        ax_a, bar_groups_a, fmt="{:.2f}", dy=0.012, stagger_thresh=0.055, stagger_step=0.042
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=8.0)
    ax_a.set_ylim(0.0, 1.18)
    ax_a.set_ylabel("Score (0 – 1, higher is better)")
    ax_a.set_title("A. Chemical Prior Quality & Structural Diversity", pad=8)
    ax_a.legend(ncol=2, loc="upper left", frameon=True, framealpha=0.9)

    # --- Panel B: Causal 3D Pocket Specificity (Matched vs Shuffled Dice + Tanimoto) ---
    ax_b = axes[0, 1]
    w_b = 0.25
    m_dice = np.array([r["matched_dice"] for r in records], dtype=np.float64)
    s_dice = np.array([r["shuffled_dice"] for r in records], dtype=np.float64)
    m_tan = np.array([r["matched_tan"] for r in records], dtype=np.float64)
    bars_md = ax_b.bar(
        x - w_b,
        np.where(np.isfinite(m_dice), m_dice, 0.0),
        width=w_b,
        color=OKABE_ITO["blue"],
        label="Matched Count Dice",
        edgecolor="white",
    )
    bars_sd = ax_b.bar(
        x,
        np.where(np.isfinite(s_dice), s_dice, 0.0),
        width=w_b,
        color=OKABE_ITO["gray"],
        label="Shuffled Count Dice",
        edgecolor="white",
    )
    bars_mt = ax_b.bar(
        x + w_b,
        np.where(np.isfinite(m_tan), m_tan, 0.0),
        width=w_b,
        color=OKABE_ITO["vermilion"],
        label="Matched True Tanimoto",
        edgecolor="white",
    )
    _annotate_grouped_bars(
        ax_b,
        [bars_md, bars_sd, bars_mt],
        fmt="{:.3f}",
        dy=0.008,
        stagger_thresh=0.022,
        stagger_step=0.017,
        fontsize=6.8,
    )
    top_b = max(
        0.44,
        float(np.nanmax(np.concatenate([m_dice, s_dice, m_tan]))) * 1.30
        if np.any(np.isfinite(m_dice))
        else 0.44,
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=8.0)
    ax_b.set_ylim(0.0, top_b)
    ax_b.set_ylabel("Fingerprint Similarity to Native Ligand")
    ax_b.set_title("B. Causal 3D Pocket Conditioning (Matched vs. Shuffled)", pad=8)
    ax_b.legend(loc="upper left", frameon=True, framealpha=0.9)

    # --- Panel C: Cross-Pocket Physicochemical Alignment (MW / TPSA / LogP Pearson r) ---
    ax_c = axes[1, 0]
    w_c = 0.25
    prop_specs = [
        ("mw_r", "MW Pearson r", OKABE_ITO["blue"]),
        ("tpsa_r", "TPSA Pearson r", OKABE_ITO["green"]),
        ("logp_r", "LogP Pearson r", OKABE_ITO["orange"]),
    ]
    bar_groups_c: list[Any] = []
    for offset_idx, (key, title, color) in enumerate(prop_specs):
        vals = np.array([r[key] for r in records], dtype=np.float64)
        plot_vals = np.where(np.isfinite(vals), vals, 0.0)
        shift = (offset_idx - 1.0) * w_c
        bars = ax_c.bar(
            x + shift, plot_vals, width=w_c, label=title, color=color, edgecolor="white"
        )
        bar_groups_c.append(bars)
    _annotate_grouped_bars(
        ax_c,
        bar_groups_c,
        fmt="{:.2f}",
        dy=0.014,
        stagger_thresh=0.045,
        stagger_step=0.040,
        fontsize=6.8,
    )
    ax_c.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=8.0)
    ax_c.set_ylim(-0.05, 1.15)
    ax_c.set_ylabel("Cross-Pocket Pearson r (Generated vs. Native)")
    ax_c.set_title("C. Cavity Size & Polarity Alignment Across CASF-2016 Pockets", pad=8)
    ax_c.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9)

    # --- Panel D: <4.5 A Contact Atom Attention Enrichment ---
    ax_d = axes[1, 1]
    enr_vals = np.array([r["attn_enrichment"] for r in records], dtype=np.float64)
    bar_colors = [
        OKABE_ITO["green"] if (math.isfinite(v) and v >= 1.5) else OKABE_ITO["skyblue"]
        for v in enr_vals
    ]
    bars_d = ax_d.bar(
        x,
        np.where(np.isfinite(enr_vals), enr_vals, 0.0),
        width=0.55,
        color=bar_colors,
        edgecolor="white",
    )
    _annotate_bars(ax_d, bars_d, fmt="{:.2f}×", dy=0.10, fontsize=7.8)
    ax_d.axhline(
        1.0,
        color=OKABE_ITO["vermilion"],
        linestyle="--",
        linewidth=1.4,
        label="Uniform Baseline (1.00×)",
    )
    top_d = max(
        2.0,
        float(np.nanmax(enr_vals)) * 1.22 if np.any(np.isfinite(enr_vals)) else 2.0,
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=8.0)
    ax_d.set_ylim(0.0, top_d)
    ax_d.set_ylabel("Contact / Peripheral Attention Enrichment (×)")
    ax_d.set_title("D. 3D Interpretability: <4.5 Å Contact Atom Attention Enrichment", pad=8)
    ax_d.legend(loc="upper left", frameon=True, framealpha=0.9)

    return _save_fig(fig, out_path)


def _load_v3_cfg_sweep(tables_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """Discover best available v3 CFG / decoding sweep table (JSON or CSV)."""
    candidates = [
        tables_dir / "cfg_sweep_v3_gpmolformer_dpo.json",
        tables_dir / "cfg_sweep_v3_gpmolformer.json",
        tables_dir / "decoding_v3_gpmolformer_dpo.json",
        tables_dir / "decoding_v3_gpmolformer.json",
        tables_dir / "cfg_sweep_v3_gpmolformer_dpo.csv",
        tables_dir / "cfg_sweep_v3_gpmolformer.csv",
        tables_dir / "decoding_v3_gpmolformer.csv",
    ]
    for pattern in (
        "cfg_sweep_v3_*.json",
        "decoding_v3_*.json",
        "cfg_sweep_v3_*.csv",
        "decoding_v3_*.csv",
    ):
        candidates.extend(sorted(tables_dir.glob(pattern)))

    for cand in candidates:
        if not cand.is_file():
            continue
        if cand.suffix == ".json":
            payload = json.loads(cand.read_text(encoding="utf-8"))
            rows = payload.get("results", payload) if isinstance(payload, dict) else payload
            if isinstance(rows, list) and rows:
                name = (
                    str(payload.get("model_name", cand.stem))
                    if isinstance(payload, dict)
                    else cand.stem
                )
                return name, rows
        elif cand.suffix == ".csv":
            df = pd.read_csv(cand)
            if not df.empty:
                return cand.stem, df.to_dict(orient="records")
    return "", []


def _summarize_docking_json(path: Path, label: str) -> dict[str, Any]:
    """Compute Top-1 Vina mean, Top-1 pocket High-Affinity %, and PoseBusters Pass-All %."""
    data = json.loads(path.read_text(encoding="utf-8"))
    per_pocket = data.get("per_pocket") or []

    top1_affs: list[float] = []
    nat_affs: list[float] = []
    top1_beats: list[float] = []
    for entry in per_pocket:
        best_v = _safe_float(entry.get("vina_affinity_best"))
        if not math.isfinite(best_v) and entry.get("vina_affinities"):
            best_v = min(_safe_float(v) for v in entry["vina_affinities"])
        nat_v = _safe_float(entry.get("native_vina_affinity"))
        if math.isfinite(best_v):
            top1_affs.append(best_v)
        if math.isfinite(nat_v):
            nat_affs.append(nat_v)
        if math.isfinite(best_v) and math.isfinite(nat_v):
            top1_beats.append(1.0 if best_v <= nat_v else 0.0)

    top1_mean = (
        float(np.mean(top1_affs)) if top1_affs else _safe_float(data.get("vina_affinity_mean"))
    )
    nat_mean = (
        float(np.mean(nat_affs)) if nat_affs else _safe_float(data.get("native_vina_affinity_mean"))
    )
    ha_pct = (
        100.0 * float(np.mean(top1_beats))
        if top1_beats
        else 100.0 * _safe_float(data.get("high_affinity_rate"))
    )
    pb_pct = 100.0 * _safe_float(data.get("posebusters_pass_rate"))
    return {
        "label": label,
        "top1_vina_mean": top1_mean,
        "native_vina_mean": nat_mean,
        "top1_ha_pct": ha_pct,
        "posebusters_pct": pb_pct,
        "n_pockets": len(top1_affs) or int(data.get("n_pockets", 0)),
    }


def _load_docking_progression(tables_dir: Path) -> list[dict[str, Any]]:
    """Discover v2 and v3 AutoDock Vina docking JSONs and prepend Native Crystal reference."""
    dock_specs: list[tuple[tuple[str, ...], str]] = [
        (("docking_casf2016.json",), "v2.0\nPaiNN Base"),
        (("docking_casf2016_dpo.json",), "v2.0\nRPO-DPO"),
        (
            ("docking_casf2016_v3_gpmolformer.json", "docking_v3_gpmolformer.json"),
            "v3.0\nGP-MoLF Base",
        ),
        (
            ("docking_casf2016_v3_gpmolformer_dpo.json", "docking_v3_gpmolformer_dpo.json"),
            "v3.0\nGP-MoLF+DPO",
        ),
        (
            ("docking_casf2016_v3_gpt2zinc.json", "docking_v3_gpt2zinc.json"),
            "v3.0\nGPT2-ZINC",
        ),
    ]

    model_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for filenames, label in dock_specs:
        for fname in filenames:
            path = tables_dir / fname
            if path.is_file():
                model_rows.append(_summarize_docking_json(path, label))
                seen.add(fname)
                break

    for path in sorted(tables_dir.glob("docking_*v3*.json")):
        if path.name in seen:
            continue
        short = path.stem.removeprefix("docking_casf2016_").removeprefix("docking_")
        model_rows.append(_summarize_docking_json(path, f"v3.0\n{short}"))

    if not model_rows:
        return []

    native_vals = [
        r["native_vina_mean"] for r in model_rows if math.isfinite(r["native_vina_mean"])
    ]
    native_ref = native_vals[0] if native_vals else float("nan")
    native_row = {
        "label": "Native\nCrystal",
        "top1_vina_mean": native_ref,
        "native_vina_mean": native_ref,
        "top1_ha_pct": float("nan"),
        "posebusters_pct": float("nan"),
        "n_pockets": model_rows[0]["n_pockets"],
        "is_native": True,
    }
    return [native_row, *model_rows]


def plot_v3_cfg_and_docking(tables_dir: Path, out_path: Path) -> Path:
    """Render ``11_v3_cfg_and_docking.png`` (CFG/CVAE Pareto + AutoDock Vina progression)."""
    _apply_pub_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.4, 6.2), constrained_layout=True)
    fig.suptitle(
        "GenEuclidean v3.0 — Autoregressive CFG / Hit-to-Lead Pareto Frontier "
        "& 3D AutoDock Vina v1.2.7 Progression",
        fontsize=13.2,
        fontweight="bold",
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(h_pad=0.14, w_pad=0.12)

    # --- Panel A: CFG Scale & Hit-to-Lead CVAE Pareto Curve ---
    sweep_name, sweep_rows = _load_v3_cfg_sweep(tables_dir)
    if sweep_rows:
        cfg_pts = [r for r in sweep_rows if not bool(r.get("use_seed_refine", False))]
        h2l_pts = [r for r in sweep_rows if bool(r.get("use_seed_refine", False))]
        all_xs: list[float] = []
        all_ys: list[float] = []

        if cfg_pts:
            cfg_pts.sort(key=lambda r: _safe_float(r.get("cfg_scale")))
            xs = np.array([_safe_float(r.get("int_div_1")) for r in cfg_pts], dtype=np.float64)
            ys = np.array([_safe_float(r.get("tanimoto")) for r in cfg_pts], dtype=np.float64)
            qeds = np.array([_safe_float(r.get("QED")) for r in cfg_pts], dtype=np.float64)
            vals = np.array([_safe_float(r.get("validity")) for r in cfg_pts], dtype=np.float64)
            valid = np.isfinite(xs) & np.isfinite(ys)
            all_xs.extend(xs[valid].tolist())
            all_ys.extend(ys[valid].tolist())
            ax_l.plot(
                xs[valid],
                ys[valid],
                color=OKABE_ITO["blue"],
                linewidth=2.3,
                marker="o",
                markersize=9,
                label="CFG Scale Sweep (w = 1.0 → 2.0)",
                zorder=3,
            )
            n_cfg = len(cfg_pts)
            for idx, (r, xv, yv, qv, vv) in enumerate(
                zip(cfg_pts, xs, ys, qeds, vals, strict=True)
            ):
                if not (math.isfinite(xv) and math.isfinite(yv)):
                    continue
                lbl = str(r.get("label", f"w={r.get('cfg_scale')}"))
                sub_parts: list[str] = []
                if math.isfinite(vv):
                    sub_parts.append(f"Val {100.0 * vv:.1f}%")
                if math.isfinite(qv):
                    sub_parts.append(f"QED {qv:.2f}")
                sub_txt = f"\n{' · '.join(sub_parts)}" if sub_parts else ""
                if idx == 0:
                    xytext, ha, va = (8, -22), "left", "top"
                elif idx == n_cfg - 1:
                    xytext, ha, va = (-10, 8), "right", "bottom"
                elif idx == 1:
                    xytext, ha, va = (0, -14), "center", "top"
                else:
                    xytext, ha, va = (10, 6), "left", "bottom"
                ax_l.annotate(
                    f"{lbl}{sub_txt}",
                    xy=(xv, yv),
                    xytext=xytext,
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=7.8,
                    fontweight="bold",
                    color=OKABE_ITO["black"],
                )

        for r in h2l_pts:
            xv = _safe_float(r.get("int_div_1"))
            yv = _safe_float(r.get("tanimoto"))
            vv = _safe_float(r.get("validity"))
            qv = _safe_float(r.get("QED"))
            if math.isfinite(xv) and math.isfinite(yv):
                all_xs.append(xv)
                all_ys.append(yv)
                ax_l.scatter(
                    [xv],
                    [yv],
                    s=230,
                    marker="*",
                    color=OKABE_ITO["vermilion"],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                    label=str(r.get("label", "Hit-to-Lead CVAE")),
                )
                h2l_sub = (
                    f"\nVal {100.0 * vv:.1f}% · QED {qv:.2f}"
                    if math.isfinite(vv) and math.isfinite(qv)
                    else ""
                )
                ax_l.annotate(
                    f"{r.get('label', 'Hit-to-Lead CVAE')}{h2l_sub}",
                    xy=(xv, yv),
                    xytext=(12, -4),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                    color=OKABE_ITO["vermilion"],
                )

        if all_xs and all_ys:
            x_span = max(max(all_xs) - min(all_xs), 0.005)
            y_span = max(max(all_ys) - min(all_ys), 0.005)
            ax_l.set_xlim(min(all_xs) - 0.14 * x_span, max(all_xs) + 0.24 * x_span)
            ax_l.set_ylim(min(all_ys) - 0.14 * y_span, max(all_ys) + 0.18 * y_span)

        display_sweep = _SWEEP_TITLE_MAP.get(sweep_name, sweep_name)
        ax_l.set_xlabel("Internal Structural Diversity (IntDiv1 ↑)")
        ax_l.set_ylabel("Matched Pocket True Tanimoto (↑)")
        ax_l.set_title(f"A. CFG Scale & Hit-to-Lead CVAE Frontier ({display_sweep})", pad=8)
        ax_l.legend(loc="lower left", frameon=True, framealpha=0.92)
    else:
        ax_l.text(
            0.5,
            0.5,
            "No v3 CFG sweep table found yet.\nRun scripts/eval_v3_llm.py --cfg-sweep",
            ha="center",
            va="center",
            fontsize=10,
            color=OKABE_ITO["gray"],
            transform=ax_l.transAxes,
        )
        ax_l.set_title("A. v3.0 CFG Scale & Hit-to-Lead CVAE Frontier", pad=8)

    # --- Panel B: 30-Pocket AutoDock Vina v1.2.7 + PoseBusters Progression ---
    dock_rows = _load_docking_progression(tables_dir)
    if dock_rows:
        d_labels = [r["label"] for r in dock_rows]
        dx = np.arange(len(dock_rows), dtype=np.float64)
        aff_vals = np.array([r["top1_vina_mean"] for r in dock_rows], dtype=np.float64)
        ha_vals = np.array([r["top1_ha_pct"] for r in dock_rows], dtype=np.float64)
        pb_vals = np.array([r["posebusters_pct"] for r in dock_rows], dtype=np.float64)

        colors = [
            OKABE_ITO["gray"]
            if r.get("is_native")
            else (OKABE_ITO["green"] if "v3.0" in r["label"] else OKABE_ITO["blue"])
            for r in dock_rows
        ]
        # Plot binding strength magnitude (-kcal/mol, taller = stronger affinity)
        strength = np.where(np.isfinite(aff_vals), -aff_vals, 0.0)
        bars = ax_r.bar(dx, strength, width=0.52, color=colors, edgecolor="white", alpha=0.92)

        native_strength = strength[0] if len(strength) > 0 and strength[0] > 0 else float("nan")
        if math.isfinite(native_strength):
            ax_r.axhline(
                native_strength,
                color=OKABE_ITO["vermilion"],
                linestyle="--",
                linewidth=1.6,
                zorder=4,
                label=f"Native Crystal ({-native_strength:.3f} kcal/mol)",
            )

        for idx, (bar, raw_aff, ha, pb) in enumerate(
            zip(bars, aff_vals, ha_vals, pb_vals, strict=True)
        ):
            h = float(bar.get_height())
            if not math.isfinite(raw_aff):
                continue
            badge = f"{raw_aff:.3f}"
            if idx > 0 and math.isfinite(ha):
                badge += f"\nHA {ha:.1f}%"
            if math.isfinite(pb):
                badge += f"\nPB {pb:.1f}%"
            y_anchor = (
                max(h, native_strength) + 0.22
                if math.isfinite(native_strength) and abs(h - native_strength) < 0.7
                else h + 0.22
            )
            ax_r.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_anchor,
                badge,
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold" if "v3.0" in d_labels[idx] else "normal",
                color="#1A1A1A",
            )

        ax_r.set_xticks(dx)
        ax_r.set_xticklabels(d_labels, fontsize=8.5)
        top_r = max(11.0, float(np.nanmax(strength)) * 1.34 if np.any(strength > 0) else 11.0)
        ax_r.set_ylim(0.0, top_r)
        ax_r.set_ylabel("Top-1 AutoDock Vina Binding Strength (−kcal/mol ↑)")
        ax_r.set_title("B. 30-Pocket AutoDock Vina v1.2.7 Top-1 Affinity, HA% & PoseBusters", pad=8)
        ax_r.legend(loc="lower right", frameon=True, framealpha=0.92)
    else:
        ax_r.text(
            0.5,
            0.5,
            "No AutoDock Vina docking JSONs found yet.\n"
            "Run scripts/run_benchmark.py --mode docking",
            ha="center",
            va="center",
            fontsize=10,
            color=OKABE_ITO["gray"],
            transform=ax_r.transAxes,
        )
        ax_r.set_title("B. 30-Pocket AutoDock Vina v1.2.7 Progression", pad=8)

    return _save_fig(fig, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for v3.0 publication figure rendering."""
    parser = argparse.ArgumentParser(
        description="Render 300-DPI Okabe-Ito publication figures for GenEuclidean v3.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory holding eval_*.json, cfg_sweep_v3_*.{json,csv}, and docking_*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Destination directory for 300-DPI PNG figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point rendering ``10_v3_llm_comparison.png`` and ``11_v3_cfg_and_docking.png``."""
    args = build_arg_parser().parse_args(argv)
    fig10 = plot_v3_llm_comparison(args.tables_dir, args.output_dir / "10_v3_llm_comparison.png")
    fig11 = plot_v3_cfg_and_docking(args.tables_dir, args.output_dir / "11_v3_cfg_and_docking.png")
    print(f"Rendered v3 figures:\n  - {fig10}\n  - {fig11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
