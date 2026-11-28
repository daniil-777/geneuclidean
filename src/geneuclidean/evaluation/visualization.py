"""Publication-Grade Visualization Suite for GenEuclidean v2.

All plotting routines use the non-interactive Matplotlib ``Agg`` backend, 300 DPI
output, a colorblind-safe Okabe-Ito palette, and clean publication styling suitable
for papers, portfolio reports, and automated CI artifact generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
from rdkit import Chem
from rdkit.Chem import Draw
from scipy import stats

from geneuclidean.evaluation.metrics import (
    cliffs_delta,
    compute_wasserstein_1,
    evaluate_affinity_metrics,
)

# Colorblind-safe Okabe-Ito palette
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#000000",  # Black
    "#7F7F7F",  # Neutral gray
)

_DPI = 300


def _apply_pub_style() -> None:
    """Apply clean, consistent publication typography and spine styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
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


def _save_figure(fig: Figure, out_path: str | Path) -> Path:
    """Save figure at 300 DPI with tight bounding box and close cleanly."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _safe_kde_plot(
    ax: Axes,
    values: NDArray[np.float64],
    *,
    color: str,
    label: str,
    fill: bool = False,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha_fill: float = 0.20,
    clip: tuple[float, float] | None = None,
) -> None:
    """Draw a smooth KDE curve with safe fallback for low-variance or small arrays."""
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return
    if len(clean) < 3 or np.isclose(np.std(clean), 0.0):
        ax.axvline(
            float(np.mean(clean)), color=color, linestyle=linestyle, lw=linewidth, label=label
        )
        return
    try:
        kde_kwargs: dict[str, Any] = {
            "x": clean,
            "ax": ax,
            "color": color,
            "label": label,
            "fill": fill,
            "alpha": alpha_fill if fill else 1.0,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "warn_singular": False,
        }
        if clip is not None:
            kde_kwargs["clip"] = clip
        sns.kdeplot(**kde_kwargs)
    except Exception:
        ax.hist(clean, bins=20, density=True, color=color, alpha=0.35, label=label)


# ---------------------------------------------------------------------------
# 1. Training Curves
# ---------------------------------------------------------------------------


def _normalize_history(
    hist: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    """Normalize either a columnar dict-of-lists or row-wise list-of-dicts history."""
    if isinstance(hist, Mapping):
        return {str(k): [float(v) for v in vals] for k, vals in hist.items()}
    columns: dict[str, list[float]] = {}
    for row in hist:
        for key, val in row.items():
            if isinstance(val, (int, float, np.number)):
                columns.setdefault(str(key), []).append(float(val))
    return columns


def _first_finite_series(
    hist: Mapping[str, Sequence[float]],
    keys: Sequence[str],
) -> list[float] | None:
    """Return the first non-empty metric series containing at least one finite value."""
    for key in keys:
        vals = hist.get(key)
        if vals:
            arr = [float(v) for v in vals]
            if any(np.isfinite(v) for v in arr):
                return arr
    return None


def plot_training_curves(
    histories_dict: Mapping[str, Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]]],
    out_path: str | Path,
) -> Path:
    """Plot multi-model training curves (Loss, Perplexity, pKd Pearson r, CVAE KL).

    Supports both columnar histories (``dict[metric, list[float]]``) and row-wise
    epoch records directly from ``PocketTrainer.fit`` / ``history.jsonl``.
    """
    _apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    ax_loss, ax_ppl, ax_pkd, ax_kl = axes.ravel()

    for idx, (model_name, raw_hist) in enumerate(histories_dict.items()):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        hist = _normalize_history(raw_hist)

        # Panel 1: Train & Val Chemical LM Loss (falls back to total/objective loss)
        train_loss = _first_finite_series(hist, ("train_lm_loss", "train_loss", "train_total_loss"))
        val_loss = _first_finite_series(
            hist, ("val_lm_loss", "val_loss", "val_objective", "val_total_loss")
        )
        if train_loss:
            epochs = np.arange(1, len(train_loss) + 1)
            ax_loss.plot(
                epochs,
                train_loss,
                color=color,
                linestyle="--",
                alpha=0.55,
                lw=1.4,
            )
        if val_loss:
            epochs = np.arange(1, len(val_loss) + 1)
            ax_loss.plot(
                epochs,
                val_loss,
                color=color,
                linestyle="-",
                lw=2.2,
                label=model_name,
            )

        # Panel 2: Validation Perplexity
        val_ppl = _first_finite_series(hist, ("val_perplexity", "val_ppl"))
        if val_ppl is None:
            lm_loss = _first_finite_series(hist, ("val_lm_loss",)) or val_loss
            if lm_loss:
                val_ppl = [float(np.exp(min(float(v), 20.0))) for v in lm_loss]
        if val_ppl:
            epochs = np.arange(1, len(val_ppl) + 1)
            ax_ppl.plot(epochs, val_ppl, color=color, lw=2.2, label=model_name)

        # Panel 3: pKd Pearson r
        val_pkd_r = _first_finite_series(
            hist, ("val_pkd_r", "val_pkd_pearson", "val_pearson_r", "pkd_r")
        )
        if val_pkd_r and any(abs(v) > 1e-8 for v in val_pkd_r):
            epochs = np.arange(1, len(val_pkd_r) + 1)
            ax_pkd.plot(epochs, val_pkd_r, color=color, lw=2.2, marker="o", ms=3, label=model_name)

        # Panel 4: CVAE KL Divergence (nats)
        kl_vals = _first_finite_series(
            hist, ("val_kl", "val_kl_nats", "train_kl", "train_kl_nats", "kl_nats")
        )
        if kl_vals and any(abs(v) > 1e-8 for v in kl_vals):
            epochs = np.arange(1, len(kl_vals) + 1)
            ax_kl.plot(epochs, kl_vals, color=color, lw=2.2, label=model_name)

    ax_loss.set_title("Chemical LM Training & Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss (↓)")
    if ax_loss.lines:
        handles, labels = ax_loss.get_legend_handles_labels()
        handles.extend(
            [
                Line2D([0], [0], color="#444444", lw=2.0, linestyle="-", label="Validation"),
                Line2D(
                    [0], [0], color="#444444", lw=1.4, linestyle="--", alpha=0.65, label="Train"
                ),
            ]
        )
        labels.extend(["Validation (solid)", "Train (dashed)"])
        ax_loss.legend(
            handles=handles,
            labels=labels,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            fontsize=8.0,
        )

    ax_ppl.set_title("Validation Chemical Language Perplexity")
    ax_ppl.set_xlabel("Epoch")
    ax_ppl.set_ylabel("Perplexity (↓)")
    if ax_ppl.lines:
        ax_ppl.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    ax_pkd.set_title("Validation Binding Affinity Scoring (pKd Pearson r)")
    ax_pkd.set_xlabel("Epoch")
    ax_pkd.set_ylabel("Pearson r (↑)")
    ax_pkd.set_ylim(0.35, 0.80)
    if ax_pkd.lines:
        ymin_pkd = min(float(np.nanmin(ln.get_ydata())) for ln in ax_pkd.lines)
        ymax_pkd = max(float(np.nanmax(ln.get_ydata())) for ln in ax_pkd.lines)
        ax_pkd.set_ylim(
            max(-0.05, min(0.35, ymin_pkd - 0.06)), min(1.02, max(0.76, ymax_pkd + 0.08))
        )
        ax_pkd.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    else:
        ax_pkd.text(
            0.5, 0.5, "No pKd head logged", ha="center", va="center", transform=ax_pkd.transAxes
        )

    ax_kl.set_title("CVAE Latent KL Divergence (nats)")
    ax_kl.set_xlabel("Epoch")
    ax_kl.set_ylabel("KL Divergence (nats)")
    if ax_kl.lines:
        ax_kl.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    else:
        ax_kl.text(
            0.5, 0.5, "No CVAE KL logged", ha="center", va="center", transform=ax_kl.transAxes
        )

    fig.suptitle("GenEuclidean v2 — Multi-Model Training Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Multi-Panel Model Comparison Bars
# ---------------------------------------------------------------------------


def _resolve_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    """Find first matching column name in ``df`` (case-insensitive)."""
    cols = [str(c) for c in df.columns]
    lower_map: dict[str, str] = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias in cols:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def plot_model_comparison_bars(
    results_df: pd.DataFrame | Sequence[Mapping[str, Any]],
    out_path: str | Path,
) -> Path:
    """Multi-panel bar chart comparing architectures across 8 core SBDD metrics."""
    _apply_pub_style()
    df = results_df.copy() if isinstance(results_df, pd.DataFrame) else pd.DataFrame(results_df)
    model_col = _resolve_column(df, ("model", "Model", "name", "architecture"))
    if model_col is None:
        df = df.reset_index()
        model_col = str(df.columns[0])

    metric_specs: list[tuple[str, tuple[str, ...]]] = [
        ("Validity (↑)", ("validity", "Validity")),
        ("Uniqueness (↑)", ("uniqueness", "Uniqueness")),
        ("Novelty (↑)", ("novelty", "Novelty")),
        ("IntDiv1 (↑)", ("int_div_1", "IntDiv1", "intdiv1")),
        ("QED (↑)", ("QED_mean", "QED", "qed")),
        ("SA_norm (↑)", ("SA_norm_mean", "SA_norm", "sa_norm")),
        (
            "True Tanimoto (↑)",
            ("matched_tanimoto_mean", "tanimoto", "True Tanimoto", "true_tanimoto"),
        ),
        ("pKd Pearson r (↑)", ("pkd_pearson_r", "pearson_r", "pKd Pearson r", "pkd_r")),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.8))
    models = [str(m) for m in df[model_col].tolist()]
    x_pos = np.arange(len(models))
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(models))]

    for ax, (title, aliases) in zip(axes.ravel(), metric_specs, strict=True):
        col = _resolve_column(df, aliases)
        if col is None:
            ax.set_title(title)
            ax.text(
                0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, color="#7F7F7F"
            )
            ax.set_xticks([])
            continue

        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        plot_vals = np.where(np.isfinite(vals), vals, 0.0)
        bars = ax.bar(
            x_pos, plot_vals, color=colors, width=0.65, edgecolor="#222222", linewidth=0.6
        )
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=8)
        upper = max(1.08, float(np.nanmax(plot_vals)) * 1.18) if len(plot_vals) > 0 else 1.08
        ax.set_ylim(0.0, upper)

        for bar, raw_v in zip(bars, vals, strict=True):
            label_txt = f"{raw_v:.2f}" if np.isfinite(raw_v) else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02 * upper,
                label_txt,
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    fig.suptitle(
        "GenEuclidean Benchmark Comparison — 2021 Thesis Baseline vs. Modern v2 Architectures",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Matched vs. Shuffled vs. Null Pocket Similarity
# ---------------------------------------------------------------------------


def plot_matched_vs_shuffled_similarity(
    matched_sims: ArrayLike,
    shuffled_sims: ArrayLike,
    null_sims: ArrayLike | None,
    out_path: str | Path,
    *,
    metric_name: str = "ECFP4 Tanimoto Similarity",
) -> Path:
    """Plot KDE + Violin/Strip comparison of Matched vs. Shuffled (vs. Null) pocket controls."""
    _apply_pub_style()
    m_arr = np.asarray(matched_sims, dtype=np.float64)
    m_arr = m_arr[np.isfinite(m_arr)]
    s_arr = np.asarray(shuffled_sims, dtype=np.float64)
    s_arr = s_arr[np.isfinite(s_arr)]
    n_arr: NDArray[np.float64] | None = None
    if null_sims is not None:
        tmp = np.asarray(null_sims, dtype=np.float64)
        tmp = tmp[np.isfinite(tmp)]
        if len(tmp) > 0:
            n_arr = tmp

    fig, (ax_kde, ax_vio) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # Left panel: KDE overlays
    _safe_kde_plot(
        ax_kde,
        m_arr,
        color=OKABE_ITO[0],
        label=f"Matched Pocket (μ={np.mean(m_arr):.3f})",
        fill=True,
    )
    _safe_kde_plot(
        ax_kde,
        s_arr,
        color=OKABE_ITO[1],
        label=f"Shuffled Pocket (μ={np.mean(s_arr):.3f})",
        fill=True,
    )
    if n_arr is not None:
        _safe_kde_plot(
            ax_kde,
            n_arr,
            color=OKABE_ITO[7],
            label=f"Null Pocket (μ={np.mean(n_arr):.3f})",
            fill=True,
            linestyle="--",
        )

    if len(m_arr) > 0:
        ax_kde.axvline(float(np.mean(m_arr)), color=OKABE_ITO[0], linestyle=":", lw=1.6)
    if len(s_arr) > 0:
        ax_kde.axvline(float(np.mean(s_arr)), color=OKABE_ITO[1], linestyle=":", lw=1.6)

    ax_kde.set_title("Similarity Distribution to Native Crystal Ligand")
    ax_kde.set_xlabel(metric_name)
    ax_kde.set_ylabel("Density")
    ax_kde.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Right panel: Violin + strip plot
    groups: list[str] = ["Matched\nPocket", "Shuffled\nPocket"]
    arrays: list[NDArray[np.float64]] = [m_arr, s_arr]
    palette = [OKABE_ITO[0], OKABE_ITO[1]]
    if n_arr is not None:
        groups.append("Null\nPocket")
        arrays.append(n_arr)
        palette.append(OKABE_ITO[7])

    rows = [
        {"Condition": grp, "Similarity": float(val)}
        for grp, arr in zip(groups, arrays, strict=True)
        for val in arr
    ]
    plot_df = pd.DataFrame(rows)
    if not plot_df.empty:
        sns.violinplot(
            data=plot_df,
            x="Condition",
            y="Similarity",
            hue="Condition",
            palette=palette,
            inner="quartile",
            cut=0,
            alpha=0.55,
            legend=False,
            ax=ax_vio,
        )
        # Subsample strip overlay if huge to keep plot crisp and fast
        strip_df = plot_df.sample(n=min(len(plot_df), 400), random_state=42)
        sns.stripplot(
            data=strip_df,
            x="Condition",
            y="Similarity",
            color="#222222",
            alpha=0.35,
            size=2.8,
            jitter=0.18,
            ax=ax_vio,
        )

    # Statistical significance annotation (Matched vs Shuffled)
    if len(m_arr) > 0 and len(s_arr) > 0:
        mwu = stats.mannwhitneyu(m_arr, s_arr, alternative="two-sided")
        delta = cliffs_delta(m_arr, s_arr)
        p_str = f"{mwu.pvalue:.2e}" if mwu.pvalue < 1e-3 else f"{mwu.pvalue:.4f}"
        ann = f"Matched vs. Shuffled:\nMann-Whitney U p = {p_str}\nCliff's δ = {delta:+.3f}"
        ax_vio.text(
            0.97,
            0.96,
            ann,
            transform=ax_vio.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8F9FA", "edgecolor": "#CCCCCC"},
        )

    ax_vio.set_title("Causal 3D Pocket Conditioning Audit")
    ax_vio.set_xlabel("")
    ax_vio.set_ylabel(metric_name)

    fig.suptitle(
        "Causal Proof of 3D Pocket Conditioning (Matched vs. Controls)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 4. 6-Panel Physicochemical Property Distributions
# ---------------------------------------------------------------------------


def plot_property_distributions(
    ref_props: Mapping[str, ArrayLike],
    gen_props_dict: Mapping[str, Mapping[str, ArrayLike]],
    out_path: str | Path,
) -> Path:
    """Plot 6-panel KDE overlays (MW, LogP, QED, SA, TPSA, NP_score) with W1 distances."""
    _apply_pub_style()
    panels: tuple[tuple[str, str], ...] = (
        ("MW", "Molecular Weight (Da)"),
        ("LogP", "Crippen LogP"),
        ("QED", "Drug-Likeness (QED)"),
        ("SA", "Synthetic Accessibility (SA ↓)"),
        ("TPSA", "Topological Polar Surface Area (Å²)"),
        ("NP_score", "Natural Product-Likeness (NP)"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    non_negative_keys = {"MW", "QED", "SA", "TPSA"}
    shared_handles: list[Any] = []
    shared_labels: list[str] = []

    for ax, (key, xlabel) in zip(axes.ravel(), panels, strict=True):
        pooled_chunks: list[NDArray[np.float64]] = []
        ref_vals = np.asarray(ref_props.get(key, []), dtype=np.float64)
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if len(ref_vals) > 0:
            pooled_chunks.append(ref_vals)
            _safe_kde_plot(
                ax,
                ref_vals,
                color="#4D4D4D",
                label="Reference Crystal",
                fill=True,
                linestyle="--",
                linewidth=2.0,
                alpha_fill=0.22,
            )

        w1_lines: list[str] = []
        for idx, (model_name, props_map) in enumerate(gen_props_dict.items()):
            color = OKABE_ITO[idx % len(OKABE_ITO)]
            g_vals = np.asarray(props_map.get(key, []), dtype=np.float64)
            g_vals = g_vals[np.isfinite(g_vals)]
            if len(g_vals) == 0:
                continue
            pooled_chunks.append(g_vals)
            _safe_kde_plot(ax, g_vals, color=color, label=model_name, fill=False, linewidth=2.0)
            if len(ref_vals) > 0:
                w1 = compute_wasserstein_1(g_vals, ref_vals)
                w1_lines.append(f"{model_name}: W₁={w1:.2f}")

        if pooled_chunks:
            pooled = np.concatenate(pooled_chunks)
            if len(pooled) >= 20:
                q_lo, q_hi = np.percentile(pooled, [0.5, 99.5])
                span = max(float(q_hi - q_lo), 1e-6)
                x_lo = float(q_lo) - 0.08 * span
                x_hi = float(q_hi) + 0.10 * span
                if key in non_negative_keys:
                    x_lo = max(0.0, x_lo)
                if key == "QED":
                    x_lo, x_hi = 0.0, 1.04
                ax.set_xlim(x_lo, x_hi)

        # Add top headroom so the W1 annotation box never overlaps KDE peaks
        y_lo, y_hi = ax.get_ylim()
        if y_hi > y_lo:
            ax.set_ylim(y_lo, y_hi * 1.24)

        ax.set_title(xlabel)
        ax.set_xlabel(key)
        ax.set_ylabel("Density")
        if w1_lines:
            ax.text(
                0.98,
                0.96,
                "\n".join(w1_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#DDDDDD",
                    "alpha": 0.92,
                },
            )
        if not shared_handles:
            h, lbls = ax.get_legend_handles_labels()
            if h:
                shared_handles, shared_labels = list(h), list(lbls)

    fig.suptitle(
        "Physicochemical & Medicinal Chemistry Property Fidelity",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.942),
            ncol=min(5, len(shared_handles)),
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            fontsize=8.8,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.895))
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Binding Affinity (pKd) Parity Plot
# ---------------------------------------------------------------------------


def plot_pkd_parity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    split_name: str,
    out_path: str | Path,
) -> Path:
    """Plot experimental vs. predicted pKd parity scatter with regression & metrics box."""
    _apply_pub_style()
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    metrics = evaluate_affinity_metrics(yt, yp)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    if len(yt) > 0:
        err = np.abs(yp - yt)
        sc = ax.scatter(
            yt,
            yp,
            c=err,
            cmap="viridis_r",
            alpha=0.78,
            s=30,
            edgecolors="white",
            linewidth=0.35,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("|Error| (pKd units)")

        lo = float(min(np.min(yt), np.min(yp))) - 0.4
        hi = float(max(np.max(yt), np.max(yp))) + 0.4
        ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", lw=1.4, label="Ideal (y = x)")

        if len(yt) >= 2 and np.std(yt) > 0.0:
            slope, intercept = np.polyfit(yt, yp, 1)
            xs = np.linspace(lo, hi, 100)
            ax.plot(
                xs,
                slope * xs + intercept,
                color=OKABE_ITO[1],
                lw=2.0,
                label=f"OLS Fit (y = {slope:.2f}x + {intercept:.2f})",
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    stats_txt = (
        f"N = {int(metrics['n'])}\n"
        f"Pearson r = {metrics['pearson_r']:.3f}\n"
        f"Spearman ρ = {metrics['spearman_rho']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"R² = {metrics['r2']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        stats_txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F8F9FA", "edgecolor": "#CCCCCC"},
    )
    ax.set_xlabel("Experimental pKd")
    ax.set_ylabel("Predicted pKd")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle(
        f"Binding Affinity (pKd) Prediction — {split_name}",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 6. 3D Pocket-Residue vs. SMILES-Token Attention Heatmap
# ---------------------------------------------------------------------------


def plot_pocket_attention_heatmap(
    tokens: Sequence[str],
    residue_labels: Sequence[str],
    attn_matrix: ArrayLike,
    contact_mask: ArrayLike,
    pdb_id: str,
    out_path: str | Path,
) -> Path:
    """Plot cross-attention heatmap between generated SMILES tokens and 3D pocket residues.

    Highlights true ``<4.5 Å`` ligand-contacting binding-site residues with an indicator
    banner and star markers, and annotates the contact attention enrichment ratio.
    """
    _apply_pub_style()
    mat = np.asarray(attn_matrix, dtype=np.float64)
    c_mask = np.asarray(contact_mask, dtype=bool).ravel()

    # Align orientation to (n_tokens, n_residues)
    if mat.ndim != 2:
        msg = f"attn_matrix must be 2D, got shape {mat.shape}"
        raise ValueError(msg)
    if mat.shape == (len(residue_labels), len(tokens)) and mat.shape[0] != len(tokens):
        mat = mat.T

    n_tok, n_res = mat.shape
    tok_labels = list(tokens)[:n_tok]
    res_labels = [
        f"{lbl} ★" if (i < len(c_mask) and c_mask[i]) else str(lbl)
        for i, lbl in enumerate(list(residue_labels)[:n_res])
    ]

    # Compute contact enrichment ratio for title
    mean_res_attn = mat.mean(axis=0)
    c_aligned = c_mask[:n_res]
    n_contact = int(c_aligned.sum())
    n_periph = n_res - n_contact
    if n_contact > 0 and n_periph > 0:
        c_mean = float(mean_res_attn[c_aligned].mean())
        p_mean = float(mean_res_attn[~c_aligned].mean())
        enrich_txt = f" | Contact Enrichment = {c_mean / max(p_mean, 1e-12):.2f}×"
    else:
        enrich_txt = ""

    fig_w = max(8.0, min(18.0, 0.38 * n_res + 2.8))
    fig_h = max(4.8, min(12.0, 0.32 * n_tok + 2.4))
    fig, (ax_bar, ax_hm) = plt.subplots(
        2,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [0.08, 0.92], "hspace": 0.04},
    )

    # Top strip showing true <4.5 Å contact residues
    strip = c_aligned.astype(np.float64)[None, :]
    contact_cmap = matplotlib.colors.ListedColormap(["#E5E7EB", OKABE_ITO[1]])
    ax_bar.imshow(strip, aspect="auto", cmap=contact_cmap, vmin=0.0, vmax=1.0)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([0])
    ax_bar.set_yticklabels(["<4.5 Å"], fontsize=8, fontweight="bold")
    ax_bar.set_title(
        f"3D Pocket Cross-Attention Interpretability — {pdb_id.upper()}{enrich_txt}",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    im = ax_hm.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest")
    ax_hm.set_xticks(np.arange(n_res))
    ax_hm.set_xticklabels(res_labels, rotation=55, ha="right", fontsize=8)
    for tick_label, is_contact in zip(ax_hm.get_xticklabels(), c_aligned, strict=True):
        if is_contact:
            tick_label.set_color(OKABE_ITO[1])
            tick_label.set_fontweight("bold")

    ax_hm.set_yticks(np.arange(n_tok))
    ax_hm.set_yticklabels(tok_labels, fontsize=8.5, family="monospace")
    ax_hm.set_xlabel("3D Binding Pocket Residues (★ = True <4.5 Å Crystal Ligand Contact)")
    ax_hm.set_ylabel("Generated SMILES Tokens")
    ax_hm.grid(False)

    cbar = fig.colorbar(im, ax=[ax_bar, ax_hm], pad=0.015, fraction=0.035)
    cbar.set_label("Cross-Attention Weight")

    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Generated Molecule Grid (RDKit)
# ---------------------------------------------------------------------------


def _format_mol_annotation(ann: str | Mapping[str, Any]) -> str:
    """Format a molecule legend string from a raw string or metric mapping."""
    if isinstance(ann, str):
        return ann
    parts: list[str] = []
    if "QED" in ann:
        parts.append(f"QED={float(ann['QED']):.2f}")
    if "SA" in ann:
        parts.append(f"SA={float(ann['SA']):.2f}")
    if "Tanimoto" in ann:
        parts.append(f"Tan={float(ann['Tanimoto']):.2f}")
    elif "tanimoto" in ann:
        parts.append(f"Tan={float(ann['tanimoto']):.2f}")
    if "pKd" in ann:
        parts.append(f"pKd={float(ann['pKd']):.2f}")
    elif "pred_pkd" in ann:
        parts.append(f"pKd={float(ann['pred_pkd']):.2f}")
    if "Vina" in ann:
        parts.append(f"Vina={float(ann['Vina']):.1f}")
    elif "vina" in ann:
        parts.append(f"Vina={float(ann['vina']):.1f}")
    return " | ".join(parts) if parts else ""


def plot_generated_molecule_grid(
    native_smiles: str,
    gen_smiles_list: Sequence[str],
    annotations: Sequence[str | Mapping[str, Any]] | None,
    pdb_id: str,
    out_path: str | Path,
    *,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (340, 280),
) -> Path:
    """Render reference crystal ligand alongside generated molecules via RDKit."""
    _apply_pub_style()
    mols: list[Chem.Mol] = []
    legends: list[str] = []

    ref_mol = Chem.MolFromSmiles(native_smiles)
    if ref_mol is not None:
        mols.append(ref_mol)
        legends.append(f"Crystal Ref ({pdb_id.upper()})")

    for idx, smi in enumerate(gen_smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        if annotations is not None and idx < len(annotations):
            ann_str = _format_mol_annotation(annotations[idx])
            legends.append(f"Gen #{idx + 1}: {ann_str}" if ann_str else f"Gen #{idx + 1}")
        else:
            legends.append(f"Gen #{idx + 1}")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not mols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No valid molecules to render for {pdb_id}", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, path)

    grid_img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(mols_per_row, len(mols)),
        subImgSize=sub_img_size,
        legends=legends,
        returnPNG=False,
    )
    img_arr = np.asarray(grid_img)

    fig_w = max(8.0, img_arr.shape[1] / 120.0)
    fig_h = max(3.5, img_arr.shape[0] / 120.0 + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_arr)
    ax.axis("off")
    ax.set_title(
        f"De Novo Ligands Generated for Target Pocket {pdb_id.upper()}",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    return _save_figure(fig, path)


# ---------------------------------------------------------------------------
# 8. Decoding Strategy Pareto Frontier
# ---------------------------------------------------------------------------


def _compute_pareto_front(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.intp]:
    """Return sorted indices of non-dominated points maximizing both ``x`` and ``y``."""
    from geneuclidean.active_learning.acquisitions import pareto_front

    front, _ = pareto_front(np.column_stack((xs, ys)))
    return front[np.argsort(xs[front], kind="stable")]


def plot_decoding_pareto(
    decoding_results: pd.DataFrame | Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    x_metric: str | None = None,
    y_metric: str | None = None,
) -> Path:
    """Plot decoding Pareto frontier: Structural Variation vs. Target Specificity."""
    _apply_pub_style()
    df = (
        decoding_results.copy()
        if isinstance(decoding_results, pd.DataFrame)
        else pd.DataFrame(decoding_results)
    )

    x_col = x_metric or _resolve_column(df, ("int_div_1", "IntDiv1", "uniqueness", "diversity"))
    y_col = y_metric or _resolve_column(
        df, ("matched_tanimoto_mean", "tanimoto", "Tanimoto", "pred_pkd", "pKd", "specificity")
    )
    label_col = _resolve_column(df, ("label", "config", "strategy", "method", "name"))
    group_col = _resolve_column(df, ("strategy", "method", "family", "group"))

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    if x_col is None or y_col is None or df.empty:
        ax.text(0.5, 0.5, "Insufficient decoding metrics for Pareto plot", ha="center", va="center")
        return _save_figure(fig, out_path)

    xs = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys)

    if group_col is not None:
        unique_groups = list(dict.fromkeys(str(g) for g in df[group_col].tolist()))
        color_map = {g: OKABE_ITO[i % len(OKABE_ITO)] for i, g in enumerate(unique_groups)}
        for grp in unique_groups:
            mask = valid & (df[group_col].astype(str).to_numpy() == grp)
            if not np.any(mask):
                continue
            ax.scatter(
                xs[mask],
                ys[mask],
                color=color_map[grp],
                label=grp,
                s=80,
                edgecolors="#222222",
                linewidth=0.7,
                zorder=3,
            )
    else:
        ax.scatter(
            xs[valid],
            ys[valid],
            color=OKABE_ITO[0],
            s=80,
            edgecolors="#222222",
            linewidth=0.7,
            zorder=3,
        )

    # Highlight non-dominated Pareto frontier
    valid_idx = np.where(valid)[0]
    if len(valid_idx) >= 2:
        front_local = _compute_pareto_front(xs[valid_idx], ys[valid_idx])
        front_idx = valid_idx[front_local]
        ax.plot(
            xs[front_idx],
            ys[front_idx],
            color=OKABE_ITO[1],
            linestyle="--",
            lw=2.0,
            marker="*",
            ms=12,
            label="Pareto Frontier",
            zorder=4,
        )

    # Pad axis limits cleanly so callout labels never clip at margins
    if len(valid_idx) > 0:
        x_min, x_max = float(np.min(xs[valid_idx])), float(np.max(xs[valid_idx]))
        y_min, y_max = float(np.min(ys[valid_idx])), float(np.max(ys[valid_idx]))
        x_span = max(x_max - x_min, 1e-4)
        y_span = max(y_max - y_min, 1e-4)
        ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.16 * x_span)
        ax.set_ylim(y_min - 0.09 * y_span, y_max + 0.12 * y_span)

    # Non-overlapping label placement with clean leader lines for crowded clusters
    if label_col is not None and len(valid_idx) > 0:
        labels = df[label_col].astype(str).tolist()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx_lim = max(x1 - x0, 1e-9)
        dy_lim = max(y1 - y0, 1e-9)

        # Normalize points to [0, 1]^2 for scale-invariant box repulsion
        pts_u = (xs[valid_idx] - x0) / dx_lim
        pts_v = (ys[valid_idx] - y0) / dy_lim
        n_pts = len(valid_idx)
        box_w = np.array([max(0.09, 0.0082 * len(labels[i]) + 0.018) for i in valid_idx])
        box_h = np.full(n_pts, 0.044, dtype=np.float64)

        # Deterministic initial spread away from dense centroid
        pos_u = pts_u.copy()
        pos_v = pts_v.copy()
        order = np.argsort(pts_v)
        for rank, k in enumerate(order):
            # Alternate slight left/right nudge for tightly packed vertical points
            close_neighbors = int(
                np.sum((np.abs(pts_u - pts_u[k]) < 0.06) & (np.abs(pts_v - pts_v[k]) < 0.06))
            )
            if close_neighbors >= 3:
                side = -1.0 if (rank % 2 == 0) else 1.0
                pos_u[k] = pts_u[k] + side * (0.11 + 0.015 * (rank % 3))
                pos_v[k] = pts_v[k] + 0.035 * ((rank % 5) - 2)
            elif pts_u[k] > 0.85:
                pos_u[k] = pts_u[k] - 0.04
                pos_v[k] = pts_v[k] + 0.04
            else:
                pos_u[k] = pts_u[k] + 0.035
                pos_v[k] = pts_v[k] + 0.032

        # Iterative repulsion separating label boxes from each other, data points, and legend
        for _ in range(180):
            # Label-label separation
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    du = pos_u[i] - pos_u[j]
                    dv = pos_v[i] - pos_v[j]
                    min_u = 0.5 * (box_w[i] + box_w[j]) + 0.012
                    min_v = 0.5 * (box_h[i] + box_h[j]) + 0.012
                    if abs(du) < min_u and abs(dv) < min_v:
                        ov_u = min_u - abs(du)
                        ov_v = min_v - abs(dv)
                        if ov_v <= ov_u * 1.15:
                            push = 0.52 * ov_v * (1.0 if dv >= 0.0 else -1.0)
                            pos_v[i] += push
                            pos_v[j] -= push
                        else:
                            push = 0.52 * ov_u * (1.0 if du >= 0.0 else -1.0)
                            pos_u[i] += push
                            pos_u[j] -= push
            # Gentle spring back toward anchor point first, then enforce hard non-overlap
            for i in range(n_pts):
                pos_u[i] += 0.02 * (pts_u[i] - pos_u[i])
                pos_v[i] += 0.02 * (pts_v[i] - pos_v[i])
            # Label-point separation so text boxes never occlude scatter markers
            for i in range(n_pts):
                for j in range(n_pts):
                    du = pos_u[i] - pts_u[j]
                    dv = pos_v[i] - pts_v[j]
                    min_u = 0.5 * box_w[i] + 0.026
                    min_v = 0.5 * box_h[i] + 0.032
                    if abs(du) < min_u and abs(dv) < min_v:
                        ov_u = min_u - abs(du)
                        ov_v = min_v - abs(dv)
                        if ov_v <= ov_u * 1.2:
                            pos_v[i] += 0.55 * ov_v * (1.0 if dv >= 0.0 else -1.0)
                        else:
                            pos_u[i] += 0.55 * ov_u * (1.0 if du >= 0.0 else -1.0)
            # Keep outside lower-left legend region and within frame margins
            for i in range(n_pts):
                half_w = 0.5 * box_w[i]
                half_h = 0.5 * box_h[i]
                if pos_u[i] - half_w < 0.25 and pos_v[i] - half_h < 0.44:
                    pos_u[i] = max(pos_u[i], 0.26 + half_w)
                pos_u[i] = float(np.clip(pos_u[i], half_w + 0.015, 0.985 - half_w))
                pos_v[i] = float(np.clip(pos_v[i], half_h + 0.025, 0.975 - half_h))

        for k, idx_i in enumerate(valid_idx):
            tx = x0 + pos_u[k] * dx_lim
            ty = y0 + pos_v[k] * dy_lim
            dist_norm = float(np.hypot(pos_u[k] - pts_u[k], pos_v[k] - pts_v[k]))
            arrow_kw: dict[str, Any] | None = (
                {"arrowstyle": "-", "color": "#777777", "lw": 0.75, "alpha": 0.8, "shrinkA": 2}
                if dist_norm > 0.035
                else None
            )
            ax.annotate(
                labels[idx_i],
                xy=(xs[idx_i], ys[idx_i]),
                xytext=(tx, ty),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=7.8,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#E0E0E0",
                    "alpha": 0.88,
                },
                arrowprops=arrow_kw,
                zorder=5,
            )

    axis_labels: dict[str, str] = {
        "int_div_1": "Internal Diversity (IntDiv1 ↑)",
        "tanimoto": "ECFP4 Tanimoto Similarity (↑)",
        "matched_tanimoto_mean": "ECFP4 Tanimoto Similarity (↑)",
        "pred_pkd": "Predicted Binding Affinity pKd (↑)",
    }
    ax.set_title("Decoding Pareto Frontier: Target Specificity vs. Structural Diversity")
    ax.set_xlabel(axis_labels.get(x_col, f"Structural Variation ({x_col} ↑)"))
    ax.set_ylabel(axis_labels.get(y_col, f"Target Specificity ({y_col} ↑)"))
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92)
    fig.tight_layout()
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 9. Learned 3D Equivariant Pocket Latent Space (PCA / t-SNE)
# ---------------------------------------------------------------------------


def _pca_project(x: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
    """Deterministic economy-SVD PCA projection to ``n_components`` dimensions."""
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return centered @ vt[:k].T


def _exact_tsne_2d(
    x: NDArray[np.float64],
    *,
    perplexity: float = 30.0,
    n_iter: int = 250,
    learning_rate: float = 150.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Deterministic exact van der Maaten & Hinton (2008) 2D t-SNE in pure NumPy."""
    n = x.shape[0]
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T), 0.0)
    np.fill_diagonal(dist_sq, 0.0)

    # Binary search per-point Gaussian precision to match target perplexity
    target_h = float(np.log(max(perplexity, 1.5)))
    p_cond = np.zeros((n, n), dtype=np.float64)
    idx_all = np.arange(n)
    for i in range(n):
        d_i = dist_sq[i, idx_all != i]
        beta, lo, hi = 1.0, -np.inf, np.inf
        p_row = np.full(n - 1, 1.0 / max(n - 1, 1), dtype=np.float64)
        for _ in range(40):
            scaled = -d_i * beta
            scaled -= np.max(scaled)
            exp_d = np.exp(scaled)
            sum_exp = float(np.sum(exp_d))
            if sum_exp <= 1e-15:
                p_row = np.full(n - 1, 1.0 / max(n - 1, 1), dtype=np.float64)
                break
            p_row = exp_d / sum_exp
            h_val = float(-np.sum(p_row * np.log(np.maximum(p_row, 1e-12))))
            diff = h_val - target_h
            if abs(diff) < 1e-4:
                break
            if diff > 0.0:
                lo = beta
                beta = beta * 2.0 if not np.isfinite(hi) else 0.5 * (beta + hi)
            else:
                hi = beta
                beta = beta * 0.5 if not np.isfinite(lo) else 0.5 * (beta + lo)
        p_cond[i, idx_all != i] = p_row

    p_sym = (p_cond + p_cond.T) / (2.0 * n)
    p_sym = np.maximum(p_sym, 1e-12)
    np.fill_diagonal(p_sym, 0.0)

    # PCA-aligned initialization with tiny deterministic jitter
    rng = np.random.default_rng(seed)
    init = _pca_project(x, n_components=2)
    if init.shape[1] < 2:
        init = np.column_stack([init[:, 0], np.zeros(n, dtype=np.float64)])
    scale = max(float(np.std(init)), 1e-6)
    y = (init / scale) * 1e-4 + rng.normal(0.0, 1e-5, size=(n, 2))
    vel = np.zeros_like(y)

    for step in range(n_iter):
        exag = 4.0 if step < 60 else 1.0
        momentum = 0.5 if step < 60 else 0.8
        y_sq = np.sum(y * y, axis=1)
        inv_dist = 1.0 / (1.0 + np.maximum(y_sq[:, None] + y_sq[None, :] - 2.0 * (y @ y.T), 0.0))
        np.fill_diagonal(inv_dist, 0.0)
        q_sym = np.maximum(inv_dist / max(float(np.sum(inv_dist)), 1e-12), 1e-12)
        pq_diff = (exag * p_sym - q_sym) * inv_dist
        # Vectorized gradient: 4 * sum_j pq_diff_ij * (y_i - y_j)
        grad = 4.0 * (np.sum(pq_diff, axis=1, keepdims=True) * y - pq_diff @ y)
        vel = momentum * vel - learning_rate * grad
        y = y + vel
        y -= np.mean(y, axis=0, keepdims=True)

    return y


def plot_pocket_embedding_tsne(
    embeddings: ArrayLike,
    labels: ArrayLike,
    out_path: str | Path,
    *,
    label_name: str = "Binding Affinity (pKd)",
    method: str = "tsne",
    seed: int = 42,
) -> Path:
    """Visualize learned 3D equivariant pocket embeddings via t-SNE (or PCA) colored by pKd."""
    _apply_pub_style()
    emb = np.asarray(embeddings, dtype=np.float64)
    lbl = np.asarray(labels)
    if emb.ndim != 2:
        msg = f"embeddings must be 2D (N, D), got shape {emb.shape}"
        raise ValueError(msg)

    n_samples, n_dims = emb.shape
    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    if n_samples < 3:
        ax.text(0.5, 0.5, "Need >= 3 pocket embeddings for projection", ha="center", va="center")
        return _save_figure(fig, out_path)

    # Optional PCA pre-reduction for speed and denoising
    reduced = _pca_project(emb, n_components=50) if (n_dims > 50 and n_samples > 50) else emb

    use_tsne = method.lower() == "tsne" and n_samples >= 6
    if use_tsne:
        perp = float(min(30, max(2, (n_samples - 1) // 3)))
        coords_2d = _exact_tsne_2d(reduced, perplexity=perp, seed=seed)
        axis_prefix = "t-SNE"
    else:
        proj = _pca_project(reduced, n_components=2)
        coords_2d = (
            np.column_stack([proj[:, 0], np.zeros(n_samples)]) if proj.shape[1] == 1 else proj
        )
        axis_prefix = "PCA"

    # Continuous colormap (e.g. pKd) vs discrete categorical legend
    if np.issubdtype(lbl.dtype, np.number):
        vals = lbl.astype(np.float64)
        sc = ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=vals,
            cmap="viridis",
            s=36,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.35,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(label_name)
    else:
        cats = list(dict.fromkeys(str(v) for v in lbl.tolist()))
        for idx, cat in enumerate(cats):
            mask = np.asarray([str(v) == cat for v in lbl.tolist()], dtype=bool)
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                color=OKABE_ITO[idx % len(OKABE_ITO)],
                label=cat,
                s=38,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.35,
            )
        ax.legend(title=label_name, frameon=True, facecolor="white", framealpha=0.9)

    ax.set_title(f"3D Equivariant Pocket Latent Space ({axis_prefix}) Colored by {label_name}")
    ax.set_xlabel(f"{axis_prefix} Dimension 1")
    ax.set_ylabel(f"{axis_prefix} Dimension 2")
    fig.tight_layout()
    return _save_figure(fig, out_path)


__all__ = [
    "OKABE_ITO",
    "plot_decoding_pareto",
    "plot_generated_molecule_grid",
    "plot_matched_vs_shuffled_similarity",
    "plot_model_comparison_bars",
    "plot_pkd_parity",
    "plot_pocket_attention_heatmap",
    "plot_pocket_embedding_tsne",
    "plot_property_distributions",
    "plot_training_curves",
]
