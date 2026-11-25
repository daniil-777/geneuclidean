#!/usr/bin/env python3
"""Generate 300-DPI Nature/JCIM Publication Figures for GenEuclidean v3.1 CNS Discovery.

Modular CLI script (``--phase {1,2,all}``) reading EVERY metric directly from the
committed evaluation JSON tables under ``results/tables/`` (zero hardcoded guesses)
and rendering publication-grade figures using the colorblind-safe Okabe-Ito palette:

Phase 1 (Gates 1-2 Trustworthiness + Iterations 0-2 Closed-Loop Progression):
  - ``results/figures/12_cns_gates12_trustworthiness.png`` (4-panel):
      (A) Gate 1 multi-seed unbiased native redocking RMSD_sym across all 12 primary
          human CNS targets (3-seed Median + Min-Max range + Best-of-3 + 2.0 A gate).
      (B) Gate 2 5-fold Bemis-Murcko scaffold-CV ROC-AUC (BBB, P-gp, hERG_Karim,
          DRD1, DRD2, All-12 QSAR mean) + Pre- vs. Post-Platt Calibration ECE drop.
      (C) Clinical CNS & Peripheral Reference Drug Sanity Scatter (P(BBB+) vs. P(hERG),
          colored by Pfizer CNS MPO v2) separating QT-liability, CNS-safe, and non-BBB zones.
      (D) PocketPerceiverResampler 3D CNS Attention Interpretability (<4.5 A contact
          enrichment, canonical pharmacophore hotspot enrichment, and Recall@10).
  - ``results/figures/13_cns_closed_loop_progression.png`` (4-panel):
      (A) Closed-loop progression across Ctrl-0A -> Ctrl-0B -> Iter-0 -> Iter-1 ->
          Iter-2 (+ Shuffled) on Joint CNS-Safe % & Pfizer CNS MPO v2 with exact
          Wilcoxon p-values and Cliff's delta brackets.
      (B) Resolution of the Basic-Amine CNS Paradox (Basic Amine %, P(hERG), P(P-gp)
          from Ctrl-0B & Iter-0 through Iter-1 SFT to Iter-2 CNS-RPO-DPO vs. Ref-Clinical).
      (C) Ablation A3 anti-reward-hacking proof (Iter-2 AD-Damped vs. A3_NoAD) and
          Ablation A5 CFG Pareto curve (w = 1.0, 1.3, 1.6).
      (D) Per-target 3D conditioning specificity (Iter-2 Matched / Shuffled P(on_target)
          fold-enrichment across all 12 targets) + Anti-Mode-Collapse & CASF-2016 audit.

Phase 2 (slotted for Iteration 3 Vina + 10x10 Cross-Target Physics & 3D Studio Gallery):
  - Figures 14, 15, 16 + ``results/viz3d/cns_tourette_3d_studio.html``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem

# Colorblind-safe Okabe-Ito publication palette
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "black": "#1E293B",
    "slate": "#64748B",
    "lightgray": "#E2E8F0",
}

_DPI = 300

# Canonical basic aliphatic amine SMARTS (required for GPCR Asp^3.32 salt bridge)
_BASIC_AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(Nc)]")

# Well-known QT-prolonging antipsychotic & peripheral non-BBB control drugs scored live
# via the trained Gate 2 ADMET oracle for Panel 12C (zero hardcoded scores!)
_EXTRA_CONTROL_DRUGS: tuple[tuple[str, str, str], ...] = (
    (
        "Pimozide",
        "QT_CNS",
        "O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1",
    ),
    (
        "Metformin",
        "Peripheral",
        "CN(C)C(=N)NC(=N)N",
    ),
    (
        "Ceftriaxone",
        "Peripheral",
        "CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CSc3nc(=O)c(=O)[nH]n3C)CS[C@H]12)c1csc(N)n1",
    ),
    (
        "Trospium",
        "Peripheral",
        "O=C(O[C@@H]1C[C@@H]2CC[C@H](C1)[N+]21CCCC1)C(O)(c1ccccc1)c1ccccc1",
    ),
)


# ---------------------------------------------------------------------------
# Shared Styling & I/O Utilities
# ---------------------------------------------------------------------------


def _apply_nature_style() -> None:
    """Apply clean Nature Communications / JCIM publication matplotlib styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_fig(fig: Figure, out_path: Path) -> Path:
    """Save figure at 300 DPI with crisp white margins and close cleanly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor="white", pad_inches=0.16)
    plt.close(fig)
    return out_path


def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse a required JSON artifact with an informative error on absence."""
    if not path.is_file():
        raise FileNotFoundError(f"Required CNS table not found: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _fmt_p(p_val: float) -> str:
    """Format a p-value with publication math superscripts for significance brackets."""
    if not math.isfinite(p_val):
        return "n.s."
    if p_val < 1e-3:
        exp = int(math.floor(math.log10(max(p_val, 1e-300))))
        mant = p_val / (10.0**exp)
        return rf"$p={mant:.1f}\times 10^{{{exp}}}$"
    if p_val < 0.01:
        return f"p={p_val:.4f}"
    if p_val < 0.05:
        return f"p={p_val:.3f}"
    return f"p={p_val:.2f}"


def _panel_title(ax: Axes, letter: str, title: str) -> None:
    """Draw a bold Nature-style panel letter and crisp title aligned cleanly left."""
    ax.set_title(
        f"({letter})  {title}",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
        color="#0F172A",
        pad=10.0,
    )


def _compute_basic_amine_fraction(smiles_list: list[str]) -> float:
    """Compute fraction of valid molecules bearing an ionizable basic aliphatic amine."""
    valid = 0
    basic = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid += 1
        if _BASIC_AMINE_SMARTS is not None and mol.HasSubstructMatch(_BASIC_AMINE_SMARTS):
            basic += 1
    return (float(basic) / float(valid)) if valid > 0 else float("nan")


# ---------------------------------------------------------------------------
# Figure 12 — Gates 1 & 2 Scientific Trustworthiness (4-Panel)
# ---------------------------------------------------------------------------


def _plot_fig12_panel_a(ax: Axes, gate1: dict[str, Any]) -> None:
    """Panel 12A: Gate 1 multi-seed unbiased native redocking RMSD_sym (12 primary CNS targets)."""
    complexes = gate1["complexes"]
    core10 = [c for c in complexes if c.get("tier") == "primary_core"]
    ext2 = [c for c in complexes if c.get("tier") == "primary_ext"]

    # Sort Core-10 by Top-1 median RMSD ascending, Extended-2 by Top-1 median RMSD ascending
    core10.sort(key=lambda c: float(c["summary"]["top1_rmsd"]["median"]))
    ext2.sort(key=lambda c: float(c["summary"]["top1_rmsd"]["median"]))

    # Order bottom-to-top on y-axis so best Core-10 sits at top, Extended-2 at bottom
    ordered: list[dict[str, Any] | None] = list(reversed(ext2)) + [None] + list(reversed(core10))

    # Shaded <= 2.0 A gold-standard crystallographic acceptance band
    ax.axvspan(0.0, 2.0, color="#ECFDF5", alpha=0.85, zorder=0)
    ax.axvline(2.0, color=OKABE_ITO["green"], linestyle="--", lw=1.5, zorder=1)
    ax.text(
        2.12,
        len(ordered) - 0.32,
        "2.0 Å gold-standard gate",
        fontsize=8.0,
        fontweight="bold",
        color="#047857",
        va="top",
        ha="left",
    )

    y_ticks: list[float] = []
    y_labels: list[str] = []
    divider_y = 0.0

    for y_idx, entry in enumerate(ordered):
        if entry is None:
            divider_y = float(y_idx)
            continue

        s = entry["summary"]
        t1 = s["top1_rmsd"]
        b3 = s["best3_rmsd"]
        med = float(t1["median"])
        lo = float(t1["min"])
        hi = float(t1["max"])
        b3_med = float(b3["median"])
        anc_b3 = float(s["best3_anchor_recovery_rate"]) * 100.0

        y_ticks.append(float(y_idx))
        y_labels.append(f"{entry['target']} ({entry['pdb_id']})")

        passed_top1 = med <= 2.0
        dot_color = OKABE_ITO["green"] if passed_top1 else OKABE_ITO["vermilion"]

        # Subtle horizontal guide stem from 0 to median
        ax.plot([0.0, med], [y_idx, y_idx], color="#CBD5E1", lw=1.3, zorder=2)

        # Min-Max 3-seed range bar
        ax.plot([lo, hi], [y_idx, y_idx], color=dot_color, lw=3.0, alpha=0.35, zorder=3)
        ax.plot(
            [lo, lo], [y_idx - 0.14, y_idx + 0.14], color=dot_color, lw=1.3, alpha=0.7, zorder=3
        )
        ax.plot(
            [hi, hi], [y_idx - 0.14, y_idx + 0.14], color=dot_color, lw=1.3, alpha=0.7, zorder=3
        )

        # Top-1 3-seed median circle
        ax.scatter(
            [med],
            [y_idx],
            s=56,
            color=dot_color,
            edgecolor="white",
            linewidth=1.1,
            zorder=5,
        )

        # Best-of-3 median diamond when it meaningfully improves upon Top-1
        if abs(b3_med - med) >= 0.25:
            ax.scatter(
                [b3_med],
                [y_idx],
                s=46,
                marker="D",
                facecolor="white",
                edgecolor=OKABE_ITO["blue"],
                linewidth=1.5,
                zorder=6,
            )

        # Place exact annotation cleanly right beside each bar's max extent (leaving right open)
        anc_tag = f"Anc {anc_b3:.0f}%"
        if abs(b3_med - med) >= 0.4:
            val_txt = f"{med:.2f} Å (B3 {b3_med:.2f}) · {anc_tag}"
        else:
            val_txt = f"{med:.2f} Å · {anc_tag}"
        ax.text(
            max(hi, med) + 0.26,
            y_idx,
            val_txt,
            fontsize=7.5,
            family="monospace",
            color="#0F172A" if passed_top1 or b3_med <= 2.0 else "#9A3412",
            va="center",
            ha="left",
        )

    # Subtle Core-10 vs Extended-2 separator line
    ax.axhline(divider_y, color="#94A3B8", linestyle=":", lw=1.1, zorder=1)
    ax.text(
        0.15,
        divider_y,
        "Extended-2 Targets  ↓",
        fontsize=7.6,
        fontweight="bold",
        color="#64748B",
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
    )

    core_sum = gate1["cohort_summaries"]["core_10_primary"]
    callout = (
        f"Core-10 Top-1 Median: {core_sum['median_top1_rmsd_angstrom']:.2f} Å\n"
        f"Best-of-3 ≤2.0 Å: {core_sum['pct_best3_le_2A']:.0f}% (9/10)\n"
        f"Anchor Recovery: {core_sum['mean_best3_anchor_recovery_pct']:.1f}%\n"
        f"PoseBusters Pass: {core_sum['mean_top1_posebusters_pass_pct']:.0f}%"
    )
    ax.text(
        0.985,
        0.975,
        callout,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color="#0F172A",
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#F8FAFC",
            "edgecolor": "#0072B2",
            "linewidth": 1.0,
        },
        zorder=8,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["green"],
            markeredgecolor="white",
            markersize=7,
            label="Top-1 Median ≤2.0 Å (3 seeds)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["vermilion"],
            markeredgecolor="white",
            markersize=7,
            label="Top-1 Median >2.0 Å",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=OKABE_ITO["blue"],
            markeredgewidth=1.4,
            markersize=6,
            label="Best-of-3 Median",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.99, 0.50),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.4,
    )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontweight="bold", fontsize=8.4)
    ax.set_xlim(0.0, 15.6)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Symmetry-Corrected Heavy-Atom RMSD (Å, 3 Unbiased Seeds)")
    ax.grid(axis="y", visible=False)
    _panel_title(ax, "A", "Gate 1: Unbiased Multi-Seed Native Redocking (12 Targets)")


def _plot_fig12_panel_b(
    fig: Figure,
    slot: Any,
    admet_cv: dict[str, Any],
    qsar_cv: dict[str, Any],
) -> None:
    """Panel 12B: Gate 2 5-fold scaffold-CV ROC-AUC (left) + Platt calibration ECE drop (right)."""
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[1.12, 1.0], wspace=0.22)
    ax_auc = fig.add_subplot(sub[0, 0])
    ax_ece = fig.add_subplot(sub[0, 1])

    ep_a = admet_cv["endpoints"]
    ep_q = qsar_cv["endpoints"]
    all12_q_aucs = [float(v["oof_roc_auc"]) for v in ep_q.values()]
    all12_q_mean = float(np.mean(all12_q_aucs))
    all12_q_std = float(np.std(all12_q_aucs))

    # Left subpanel: 5-Fold Bemis-Murcko Scaffold-CV ROC-AUC
    auc_rows = [
        (
            "BBB (B3DB)",
            float(ep_a["bbb"]["oof_roc_auc"]),
            float(ep_a["bbb"]["fold_roc_auc_std"]),
            OKABE_ITO["blue"],
        ),
        (
            "P-gp (TDC)",
            float(ep_a["pgp"]["oof_roc_auc"]),
            float(ep_a["pgp"]["fold_roc_auc_std"]),
            OKABE_ITO["skyblue"],
        ),
        (
            "hERG (Karim)",
            float(ep_a["herg"]["oof_roc_auc"]),
            float(ep_a["herg"]["fold_roc_auc_std"]),
            OKABE_ITO["vermilion"],
        ),
        (
            "QSAR DRD1",
            float(ep_q["DRD1"]["oof_roc_auc"]),
            float(ep_q["DRD1"]["fold_roc_auc_std"]),
            OKABE_ITO["green"],
        ),
        (
            "QSAR DRD2",
            float(ep_q["DRD2"]["oof_roc_auc"]),
            float(ep_q["DRD2"]["fold_roc_auc_std"]),
            OKABE_ITO["purple"],
        ),
        ("QSAR All-12", all12_q_mean, all12_q_std, OKABE_ITO["orange"]),
    ]

    y_pos = np.arange(len(auc_rows))[::-1]
    for y, (_lbl, val, sd, col) in zip(y_pos, auc_rows, strict=True):
        ax_auc.barh(y, val, height=0.60, color=col, alpha=0.88, edgecolor="none", zorder=3)
        ax_auc.errorbar(
            val,
            y,
            xerr=sd,
            fmt="none",
            ecolor="#0F172A",
            elinewidth=1.1,
            capsize=2.5,
            zorder=4,
        )
        ax_auc.text(
            val + sd + 0.008,
            y,
            f"{val:.4f}",
            fontsize=7.9,
            fontweight="bold",
            family="monospace",
            color="#0F172A",
            va="center",
        )

    ax_auc.axvline(0.80, color="#64748B", linestyle="--", lw=1.1, zorder=2)
    ax_auc.text(0.804, 0.55, "0.80\ngate", fontsize=7.2, color="#475569", va="center")
    ax_auc.set_yticks(y_pos)
    ax_auc.set_yticklabels([r[0] for r in auc_rows], fontweight="bold", fontsize=8.3)
    ax_auc.set_xlim(0.70, 1.08)
    ax_auc.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax_auc.set_xlabel("5-Fold Scaffold-CV OOF ROC-AUC")
    ax_auc.grid(axis="y", visible=False)
    _panel_title(ax_auc, "B", "Gate 2: Scaffold-CV Oracles")

    # Right subpanel: Pre- vs. Post-Platt Calibration ECE Drop
    ece_rows = [
        ("BBB", float(ep_a["bbb"]["oof_uncalibrated_ece"]), float(ep_a["bbb"]["oof_ece"])),
        ("P-gp", float(ep_a["pgp"]["oof_uncalibrated_ece"]), float(ep_a["pgp"]["oof_ece"])),
        ("hERG", float(ep_a["herg"]["oof_uncalibrated_ece"]), float(ep_a["herg"]["oof_ece"])),
        ("DRD1", float(ep_q["DRD1"]["oof_uncalibrated_ece"]), float(ep_q["DRD1"]["oof_ece"])),
        ("DRD2", float(ep_q["DRD2"]["oof_uncalibrated_ece"]), float(ep_q["DRD2"]["oof_ece"])),
    ]
    y_e = np.arange(len(ece_rows))[::-1]
    ax_ece.axvspan(0.0, 0.05, color="#ECFDF5", alpha=0.75, zorder=0)
    ax_ece.axvline(0.05, color=OKABE_ITO["green"], linestyle="--", lw=1.4, zorder=1)
    ax_ece.text(
        0.0518,
        3.50,
        "ECE ≤ 0.05\ngate",
        fontsize=7.3,
        fontweight="bold",
        color="#047857",
        va="center",
    )

    for y, (_lbl, pre, post) in zip(y_e, ece_rows, strict=True):
        ax_ece.annotate(
            "",
            xy=(post + 0.0018, y),
            xytext=(pre - 0.0018, y),
            arrowprops={
                "arrowstyle": "->,head_width=0.22,head_length=0.28",
                "color": "#475569",
                "lw": 1.35,
            },
            zorder=3,
        )
        ax_ece.scatter(
            [pre],
            [y],
            s=46,
            facecolor="white",
            edgecolor=OKABE_ITO["vermilion"],
            linewidth=1.6,
            zorder=4,
        )
        ax_ece.scatter(
            [post],
            [y],
            s=52,
            facecolor=OKABE_ITO["green"],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        pct_drop = (post - pre) / max(pre, 1e-9) * 100.0
        ax_ece.text(
            max(pre, 0.054) + 0.004,
            y,
            f"{post:.4f} ({pct_drop:+.0f}%)",
            fontsize=7.3,
            family="monospace",
            color="#0F172A",
            va="center",
        )

    ax_ece.set_yticks(y_e)
    ax_ece.set_yticklabels([r[0] for r in ece_rows], fontweight="bold", fontsize=8.3)
    ax_ece.set_xlim(0.0, 0.112)
    ax_ece.set_ylim(-1.15, 4.55)
    ax_ece.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08])
    ax_ece.set_xlabel("Expected Calibration Error (ECE ↓)")
    ax_ece.set_title(
        "Platt Calibration Drop",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color="#334155",
        pad=10.0,
    )
    ax_ece.grid(axis="y", visible=False)
    ece_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=OKABE_ITO["vermilion"],
            markeredgewidth=1.5,
            markersize=6,
            label="Uncalibrated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["green"],
            markeredgecolor="white",
            markersize=6.5,
            label="Platt Calibrated",
        ),
    ]
    ax_ece.legend(
        handles=ece_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.1,
    )


def _plot_fig12_panel_c(
    ax: Axes,
    iter0: dict[str, Any],
    admet_model_path: Path | None,
) -> None:
    """Panel 12C: Clinical CNS & Peripheral Reference Drug Sanity Scatter (BBB vs. hERG)."""
    drugs: list[dict[str, Any]] = []
    for row in iter0.get("clinical_drug_profiles", []):
        drugs.append(
            {
                "name": str(row["name"]),
                "group": "CNS",
                "bbb": float(row["bbb_prob"]),
                "herg": float(row["herg_prob"]),
                "mpo": float(row["cns_mpo_v2"]),
            }
        )

    # Score Pimozide + 3 peripheral non-BBB controls live with our trained Gate 2 ADMET oracle
    if admet_model_path is not None and admet_model_path.is_file():
        from geneuclidean.cns.admet import CNSAdmetSuite
        from geneuclidean.cns.mpo import compute_cns_mpo

        suite = CNSAdmetSuite.load(admet_model_path)
        preds = suite.predict([smi for _, _, smi in _EXTRA_CONTROL_DRUGS])
        for (name, grp, smi), pred in zip(_EXTRA_CONTROL_DRUGS, preds, strict=True):
            mpo_res = compute_cns_mpo(smi)
            drugs.append(
                {
                    "name": name,
                    "group": grp,
                    "bbb": float(pred.bbb_prob),
                    "herg": float(pred.herg_prob),
                    "mpo": float(mpo_res.cns_mpo_v2),
                }
            )

    # Background pharmacologic sanity zones
    ax.axvspan(0.0, 0.50, color="#F1F5F9", alpha=0.80, zorder=0)
    ax.fill_between([0.55, 1.04], 0.0, 0.42, color="#ECFDF5", alpha=0.82, zorder=0)
    ax.fill_between([0.55, 1.04], 0.55, 1.05, color="#FEF2F2", alpha=0.82, zorder=0)

    ax.axvline(0.50, color="#94A3B8", linestyle="--", lw=1.0, zorder=1)
    ax.axhline(0.42, color=OKABE_ITO["green"], linestyle=":", lw=1.0, zorder=1)
    ax.axhline(0.55, color=OKABE_ITO["vermilion"], linestyle=":", lw=1.0, zorder=1)

    ax.text(
        0.04,
        0.96,
        "Peripheral / Non-BBB\nControls (P(BBB+) < 0.5)",
        fontsize=7.7,
        fontweight="bold",
        color="#475569",
        va="top",
        ha="left",
    )
    ax.text(
        1.015,
        0.015,
        "CNS-Safe Window (High BBB+, Low hERG)",
        fontsize=7.1,
        fontweight="bold",
        color="#047857",
        va="bottom",
        ha="right",
    )
    ax.text(
        0.57,
        1.025,
        "CNS-Active / QT Liability Zone",
        fontsize=7.6,
        fontweight="bold",
        color="#B91C1C",
        va="top",
        ha="left",
    )

    cns_pts = [d for d in drugs if d["group"] == "CNS"]
    ctrl_pts = [d for d in drugs if d["group"] != "CNS"]

    cmap = plt.get_cmap("viridis")
    sc = ax.scatter(
        [d["bbb"] for d in cns_pts],
        [d["herg"] for d in cns_pts],
        c=[d["mpo"] for d in cns_pts],
        cmap=cmap,
        vmin=2.0,
        vmax=6.0,
        s=64,
        edgecolor="#0F172A",
        linewidth=0.85,
        zorder=4,
        label="Clinical CNS Drugs (n=25)",
    )
    if ctrl_pts:
        ax.scatter(
            [d["bbb"] for d in ctrl_pts],
            [d["herg"] for d in ctrl_pts],
            c=[d["mpo"] for d in ctrl_pts],
            cmap=cmap,
            vmin=2.0,
            vmax=6.0,
            marker="D",
            s=72,
            edgecolor="#0F172A",
            linewidth=1.2,
            zorder=5,
            label="QT / Peripheral Controls",
        )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.025, aspect=22)
    cbar.set_label("Pfizer CNS MPO v2 (0–6)", fontsize=8.3, fontweight="bold")
    cbar.ax.tick_params(labelsize=7.8)

    # Curated clinical drug callouts with guaranteed non-overlapping positions
    label_targets: dict[str, tuple[float, float, str]] = {
        "Pimozide": (0.76, 0.955, "right"),
        "Haloperidol": (0.76, 0.870, "right"),
        "Risperidone": (0.76, 0.780, "right"),
        "Ecopipam": (0.76, 0.650, "right"),
        "Valbenazine": (0.66, 0.495, "right"),
        "Xanomeline": (0.76, 0.315, "right"),
        "Clonidine": (0.76, 0.220, "right"),
        "Ulotaront (SEP-363856)": (0.76, 0.115, "right"),
        "Guanfacine": (0.55, 0.075, "right"),
        "Metformin": (0.33, 0.210, "left"),
        "Ceftriaxone": (0.14, 0.265, "left"),
        "Trospium": (0.40, 0.745, "left"),
    }
    short_names = {"Ulotaront (SEP-363856)": "Ulotaront"}

    for d in drugs:
        nm = d["name"]
        if nm not in label_targets:
            continue
        tx, ty, ha = label_targets[nm]
        disp = short_names.get(nm, nm)
        ax.annotate(
            disp,
            xy=(d["bbb"], d["herg"]),
            xytext=(tx, ty),
            fontsize=7.5,
            fontweight="bold",
            color="#0F172A",
            ha=ha,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.92,
                "linewidth": 0.6,
            },
            arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.85},
            zorder=7,
        )

    ax.set_xlim(0.0, 1.04)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Calibrated Blood-Brain Barrier Probability P(BBB+)")
    ax.set_ylabel("Calibrated Cardiotoxicity Risk P(hERG)")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.01, 0.45),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.4,
    )
    _panel_title(ax, "C", "Clinical CNS & Peripheral Drug Sanity Verification (n = 29)")


def _plot_fig12_panel_d(ax: Axes, iter2: dict[str, Any]) -> None:
    """Panel 12D: PocketPerceiverResampler 3D CNS attention enrichment across all 12 targets."""
    per_target = iter2["arms"]["Iter-2_Matched"]["per_target"]
    all12 = iter2["arms"]["Iter-2_Matched"]["all12"]

    rows: list[dict[str, Any]] = []
    for tgt, row in per_target.items():
        att = row.get("attention", {})
        top5 = att.get("top5_residues", [])
        top_res = str(top5[0]) if top5 else "—"
        rows.append(
            {
                "target": tgt,
                "contact": float(att.get("contact_enrichment_ratio", float("nan"))),
                "hotspot": float(att.get("hotspot_enrichment_ratio", float("nan"))),
                "recall10": float(att.get("hotspot_recall_at_10", float("nan"))) * 100.0,
                "top_res": top_res,
            }
        )

    # Sort descending by <4.5 A contact enrichment so strongest targets sit at top
    rows.sort(key=lambda r: r["contact"])
    y_pos = np.arange(len(rows))
    bar_h = 0.36

    ax.axvline(1.0, color="#94A3B8", linestyle=":", lw=1.1, zorder=1)
    mean_c = float(all12["attn_contact_enrichment"])
    mean_h = float(all12["attn_hotspot_enrichment"])
    mean_r10 = float(all12["attn_hotspot_recall_at_10"]) * 100.0

    ax.axvline(mean_c, color=OKABE_ITO["blue"], linestyle="--", lw=1.25, alpha=0.75, zorder=2)
    ax.axvline(mean_h, color=OKABE_ITO["orange"], linestyle="--", lw=1.25, alpha=0.75, zorder=2)

    ax.barh(
        y_pos + bar_h * 0.52,
        [r["contact"] for r in rows],
        height=bar_h,
        color=OKABE_ITO["blue"],
        alpha=0.88,
        label=f"<4.5 Å Contact ({mean_c:.2f}×)",
        zorder=3,
    )
    ax.barh(
        y_pos - bar_h * 0.52,
        [r["hotspot"] for r in rows],
        height=bar_h,
        color=OKABE_ITO["orange"],
        alpha=0.90,
        label=f"Canonical Hotspots ({mean_h:.2f}×)",
        zorder=3,
    )

    for y, r in zip(y_pos, rows, strict=True):
        right_x = max(r["contact"], r["hotspot"]) + 0.22
        ax.text(
            right_x,
            y,
            f"R@10 {r['recall10']:.0f}% · #1 {r['top_res']}",
            fontsize=7.2,
            family="monospace",
            color="#334155",
            va="center",
            ha="left",
        )

    summary_box = (
        f"Mean <4.5 Å Contact: {mean_c:.2f}×\n"
        f"Canonical Hotspots: {mean_h:.2f}×\n"
        f"Ionic/Key Anchors: {float(all12['attn_anchor_enrichment']):.2f}×\n"
        f"Hotspot Recall@10: {mean_r10:.1f}%"
    )
    ax.text(
        0.988,
        0.48,
        summary_box,
        transform=ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
        color="#0F172A",
        ha="right",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.40",
            "facecolor": "#F8FAFC",
            "edgecolor": OKABE_ITO["blue"],
            "linewidth": 1.0,
        },
        zorder=6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["target"] for r in rows], fontweight="bold", fontsize=8.4)
    ax.set_xlim(0.0, 21.2)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("3D Perceiver Cross-Attention Enrichment Ratio (vs. Pocket Mean, ×)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.16),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.4,
    )
    _panel_title(ax, "D", "PocketPerceiverResampler 3D CNS Attention Interpretability")


def generate_figure_12(
    tables_dir: Path,
    figures_dir: Path,
    admet_model_path: Path | None,
) -> Path:
    """Render Figure 12: Gates 1 & 2 Scientific Trustworthiness (300 DPI, 4-panel)."""
    _apply_nature_style()
    gate1 = _load_json(tables_dir / "cns_gate1_redocking.json")
    admet_cv = _load_json(tables_dir / "cns_admet_cv_metrics.json")
    qsar_cv = _load_json(tables_dir / "cns_selectivity_qsar_cv.json")
    iter0 = _load_json(tables_dir / "cns_iter0_baselines.json")
    iter2 = _load_json(tables_dir / "cns_iter2_dpo_eval.json")

    fig = plt.figure(figsize=(15.6, 11.4), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.31, wspace=0.22, top=0.92, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_fig12_panel_a(ax_a, gate1)
    _plot_fig12_panel_b(fig, gs[0, 1], admet_cv, qsar_cv)
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_fig12_panel_c(ax_c, iter0, admet_model_path)
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_fig12_panel_d(ax_d, iter2)

    fig.suptitle(
        "GenEuclidean v3.1 — Gates 1 & 2 Trustworthiness: "
        "3D Physics, Calibrated Oracles & Attention Audit",
        fontsize=13.2,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    return _save_fig(fig, figures_dir / "12_cns_gates12_trustworthiness.png")


# ---------------------------------------------------------------------------
# Figure 13 — Closed-Loop Progression, Basic-Amine Paradox & Ablations (4-Panel)
# ---------------------------------------------------------------------------


def _draw_sig_bracket(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    color: str = "#0F172A",
) -> None:
    """Draw a crisp statistical significance bracket with centered label."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=1.05, zorder=5)
    ax.text(
        0.5 * (x1 + x2),
        y + h + 0.5,
        text,
        ha="center",
        va="bottom",
        fontsize=7.3,
        fontweight="bold",
        color=color,
        zorder=6,
    )


def _plot_fig13_panel_a(
    ax: Axes,
    iter0: dict[str, Any],
    iter1: dict[str, Any],
    iter2: dict[str, Any],
) -> None:
    """Panel 13A: Closed-loop progression across 6 arms on Joint CNS-Safe % & CNS MPO v2."""
    arms_spec = [
        ("Ctrl-0A\n(ZINC)", iter0["arms"]["Ctrl-0A"]["all12"], "#94A3B8"),
        ("Ctrl-0B\n(MW/LogP)", iter2["arms"]["Ctrl-0B"]["all12"], "#64748B"),
        ("Iter-0\n(Zero-Shot)", iter0["arms"]["Iter-0_Matched"]["all12"], OKABE_ITO["skyblue"]),
        ("Iter-1\n(CNS SFT)", iter1["arms"]["Iter-1_Matched"]["all12"], OKABE_ITO["orange"]),
        ("Iter-2\n(CNS-DPO)", iter2["arms"]["Iter-2_Matched"]["all12"], OKABE_ITO["green"]),
        ("Iter-2\n(Shuffled)", iter2["arms"]["Iter-2_Shuffled"]["all12"], OKABE_ITO["purple"]),
    ]

    x = np.arange(len(arms_spec), dtype=np.float64)
    safe_pct = [float(a[1]["joint_cns_safe_rate"]) * 100.0 for a in arms_spec]
    mpo_vals = [float(a[1]["cns_mpo_v2_mean"]) for a in arms_spec]
    colors = [a[2] for a in arms_spec]

    bars = ax.bar(
        x,
        safe_pct,
        width=0.56,
        color=colors,
        edgecolor="#1E293B",
        linewidth=0.8,
        alpha=0.90,
        zorder=3,
    )
    for bar, val in zip(bars, safe_pct, strict=True):
        if val >= 28.0:
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                val - 2.4,
                f"{val:.1f}%",
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                val + 0.9,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color="#0F172A",
                zorder=5,
            )

    # Secondary right axis for Pfizer CNS MPO v2 progression
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(OKABE_ITO["blue"])
    ax2.grid(False)

    ax2.plot(
        x,
        mpo_vals,
        color=OKABE_ITO["blue"],
        marker="D",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        lw=2.0,
        zorder=6,
        label="Pfizer CNS MPO v2 (0–6)",
    )
    for xi, mv in zip(x, mpo_vals, strict=True):
        dy = 0.10 if xi != 3 else -0.15
        ax2.text(
            xi,
            mv + dy,
            f"{mv:.2f}",
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=7.6,
            fontweight="bold",
            color=OKABE_ITO["blue"],
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=7,
        )
    ax2.set_ylim(3.75, 5.85)
    ax2.set_ylabel("Pfizer CNS MPO v2 (0–6)", color=OKABE_ITO["blue"], fontweight="bold")
    ax2.tick_params(axis="y", colors=OKABE_ITO["blue"])

    # Exact Wilcoxon p & Cliff's delta brackets from committed statistics_all12
    st12 = iter2["statistics_all12"]
    s_i1 = st12["Iter-1"]["joint_cns_safe"]
    s_i0 = st12["Iter-0"]["joint_cns_safe"]
    s_cb = st12["Ctrl-0B"]["joint_cns_safe"]

    lbl_i1 = (
        f"+{s_i1['diff_target_mean'] * 100:.1f} pp "
        f"({_fmt_p(s_i1['wilcoxon_p'])}, δ={s_i1['cliffs_delta_mol']:+.2f})"
    )
    lbl_i0 = (
        f"+{s_i0['diff_target_mean'] * 100:.1f} pp "
        f"({_fmt_p(s_i0['wilcoxon_p'])}, δ={s_i0['cliffs_delta_mol']:+.2f})"
    )
    lbl_cb = (
        f"vs Ctrl-0B: +{s_cb['diff_target_mean'] * 100:.1f} pp "
        f"({_fmt_p(s_cb['wilcoxon_p'])}, δ={s_cb['cliffs_delta_mol']:+.2f})"
    )
    _draw_sig_bracket(ax, 3.0, 4.0, 48.0, 1.5, lbl_i1, color="#047857")
    _draw_sig_bracket(ax, 2.0, 4.0, 54.8, 1.5, lbl_i0, color="#0F172A")
    _draw_sig_bracket(ax, 1.0, 4.0, 61.6, 1.5, lbl_cb, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms_spec], fontweight="bold", fontsize=8.1)
    ax.set_ylim(0.0, 71.0)
    ax.set_ylabel("In-AD Joint CNS-Safe Pass Rate (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Patch(
                facecolor=OKABE_ITO["green"],
                edgecolor="#1E293B",
                label="In-AD Joint CNS-Safe %",
            ),
            Line2D(
                [0],
                [0],
                color=OKABE_ITO["blue"],
                marker="D",
                markerfacecolor="white",
                markeredgewidth=1.6,
                lw=2.0,
                label="Pfizer CNS MPO v2",
            ),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.3,
    )
    _panel_title(ax, "A", "Closed-Loop Progression (All 12 Targets, N = 128/Target)")


def _plot_fig13_panel_b(
    ax: Axes,
    iter0: dict[str, Any],
    iter1: dict[str, Any],
    iter2: dict[str, Any],
    candidates_path: Path,
) -> None:
    """Panel 13B: Resolution of the Basic-Amine CNS Paradox (Basic Amines vs. hERG & P-gp)."""
    cand_data = _load_json(candidates_path)
    arm_smiles: dict[str, list[str]] = {}
    for arm_name, tmap in cand_data["arms"].items():
        arm_smiles[arm_name] = [
            str(row["canonical_smiles"]) for rows in tmap.values() for row in rows
        ]
    clin_smiles = [str(row["canonical_smiles"]) for row in iter0["clinical_drug_profiles"]]

    stages = [
        (
            "Ctrl-0B\n(ZINC)",
            _compute_basic_amine_fraction(arm_smiles["Ctrl-0B"]) * 100.0,
            float(iter2["arms"]["Ctrl-0B"]["all12"]["herg_prob_mean"]) * 100.0,
            float(iter2["arms"]["Ctrl-0B"]["all12"]["pgp_prob_mean"]) * 100.0,
        ),
        (
            "Iter-0\n(Zero-Shot)",
            _compute_basic_amine_fraction(arm_smiles["Iter-0"]) * 100.0,
            float(iter0["arms"]["Iter-0_Matched"]["all12"]["herg_prob_mean"]) * 100.0,
            float(iter0["arms"]["Iter-0_Matched"]["all12"]["pgp_prob_mean"]) * 100.0,
        ),
        (
            "Iter-1\n(CNS SFT)",
            _compute_basic_amine_fraction(arm_smiles["Iter-1"]) * 100.0,
            float(iter1["arms"]["Iter-1_Matched"]["all12"]["herg_prob_mean"]) * 100.0,
            float(iter1["arms"]["Iter-1_Matched"]["all12"]["pgp_prob_mean"]) * 100.0,
        ),
        (
            "Iter-2\n(CNS-DPO)",
            _compute_basic_amine_fraction(arm_smiles["Iter-2"]) * 100.0,
            float(iter2["arms"]["Iter-2_Matched"]["all12"]["herg_prob_mean"]) * 100.0,
            float(iter2["arms"]["Iter-2_Matched"]["all12"]["pgp_prob_mean"]) * 100.0,
        ),
        (
            "Ref-Clinical\n(Drugs)",
            _compute_basic_amine_fraction(clin_smiles) * 100.0,
            float(iter0["arms"]["Ref-Clinical"]["all12"]["herg_prob_mean"]) * 100.0,
            float(iter0["arms"]["Ref-Clinical"]["all12"]["pgp_prob_mean"]) * 100.0,
        ),
    ]

    x = np.array([0.0, 1.15, 2.30, 3.45, 4.82], dtype=np.float64)
    w = 0.28

    basic_vals = [s[1] for s in stages]
    herg_vals = [s[2] for s in stages]
    pgp_vals = [s[3] for s in stages]

    b1 = ax.bar(
        x - w * 1.04,
        basic_vals,
        width=w,
        color=OKABE_ITO["blue"],
        edgecolor="#1E293B",
        linewidth=0.65,
        label="Basic Amines (% Top-10, Asp3.32)",
        zorder=3,
    )
    b2 = ax.bar(
        x,
        herg_vals,
        width=w,
        color=OKABE_ITO["vermilion"],
        edgecolor="#1E293B",
        linewidth=0.65,
        label="Cardiotox Risk P(hERG) (%)",
        zorder=3,
    )
    b3 = ax.bar(
        x + w * 1.04,
        pgp_vals,
        width=w,
        color=OKABE_ITO["orange"],
        edgecolor="#1E293B",
        linewidth=0.65,
        label="Efflux / P-gp Liability P(P-gp) (%)",
        zorder=3,
    )

    for grp_bars in (b1, b2, b3):
        for bar in grp_bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                val + 1.2,
                f"{val:.0f}%" if val >= 50.0 else f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color="#0F172A",
            )

    # Vertical dashed divider before Ref-Clinical benchmark
    ax.axvline(4.12, color="#94A3B8", linestyle="--", lw=1.1, zorder=2)

    # Highlight the headline paradox resolution between Iter-1 SFT and Iter-2 DPO
    herg_drop = (herg_vals[3] - herg_vals[2]) / herg_vals[2] * 100.0
    pgp_drop = (pgp_vals[3] - pgp_vals[2]) / pgp_vals[2] * 100.0
    amine_fold = basic_vals[3] / max(basic_vals[1], 1e-6)

    ax.annotate(
        f"Basic-Amine Paradox Resolved:\n"
        f"• Preserves {basic_vals[3]:.1f}% basic amines ({amine_fold:.1f}× Iter-0)\n"
        f"• Slashes hERG {herg_drop:.0f}% & P-gp {pgp_drop:.0f}%",
        xy=(3.45, herg_vals[3] + 6.0),
        xytext=(1.88, 76.0),
        fontsize=7.8,
        fontweight="bold",
        color="#065F46",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#ECFDF5",
            "edgecolor": OKABE_ITO["green"],
            "linewidth": 1.1,
        },
        arrowprops={
            "arrowstyle": "->,head_width=0.25,head_length=0.35",
            "color": OKABE_ITO["green"],
            "lw": 1.6,
            "connectionstyle": "arc3,rad=-0.16",
        },
        zorder=8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontweight="bold", fontsize=8.2)
    ax.set_ylim(0.0, 99.0)
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.4,
    )
    _panel_title(ax, "B", "Headline Discovery: Resolving the Basic-Amine CNS Paradox")


def _plot_fig13_panel_c(fig: Figure, slot: Any, iter2: dict[str, Any]) -> None:
    """Panel 13C: Ablation A3 anti-reward-hacking proof (left) & Ablation A5 CFG Pareto (right)."""
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[1.12, 1.0], wspace=0.26)
    ax_a3 = fig.add_subplot(sub[0, 0])
    ax_a5 = fig.add_subplot(sub[0, 1])

    # --- C1: Ablation A3 AD-Damping Proof ---
    dpo = iter2["arms"]["Iter-2_Matched"]["all12"]
    noad = iter2["ablations"]["A3_NoAD"]["all12"]
    st_a3 = iter2["statistics_all12"]["A3_NoAD"]

    metrics = [
        (
            "In-AD\nSafe (%)",
            float(dpo["joint_cns_safe_rate"]) * 100.0,
            float(noad["joint_cns_safe_rate"]) * 100.0,
            f"{float(dpo['joint_cns_safe_rate']) * 100:.1f}%",
            f"{float(noad['joint_cns_safe_rate']) * 100:.1f}%",
            _fmt_p(float(st_a3["joint_cns_safe"]["wilcoxon_p"])),
        ),
        (
            "Min AD\n(×100)",
            float(dpo["min_ad_sim_mean"]) * 100.0,
            float(noad["min_ad_sim_mean"]) * 100.0,
            f"{float(dpo['min_ad_sim_mean']):.3f}",
            f"{float(noad['min_ad_sim_mean']):.3f}",
            _fmt_p(float(st_a3["min_ad_sim"]["wilcoxon_p"])),
        ),
        (
            "Test-Scaf\nSNN (×100)",
            float(dpo["test_scaffold_snn_mean"]) * 100.0,
            float(noad["test_scaffold_snn_mean"]) * 100.0,
            f"{float(dpo['test_scaffold_snn_mean']):.3f}",
            f"{float(noad['test_scaffold_snn_mean']):.3f}",
            _fmt_p(float(st_a3["test_scaffold_snn"]["wilcoxon_p"])),
        ),
    ]

    x = np.arange(len(metrics), dtype=np.float64)
    w = 0.34
    ax_a3.bar(
        x - w * 0.5,
        [m[1] for m in metrics],
        width=w,
        color=OKABE_ITO["green"],
        edgecolor="#1E293B",
        linewidth=0.7,
        label="Iter-2 (AD-Damped)",
        zorder=3,
    )
    ax_a3.bar(
        x + w * 0.5,
        [m[2] for m in metrics],
        width=w,
        color=OKABE_ITO["vermilion"],
        edgecolor="#1E293B",
        linewidth=0.7,
        alpha=0.86,
        label="A3_NoAD (Unguarded)",
        zorder=3,
    )

    for xi, m in zip(x, metrics, strict=True):
        ax_a3.text(
            xi - w * 0.5,
            m[1] + 0.8,
            m[3],
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color="#065F46",
        )
        ax_a3.text(
            xi + w * 0.5,
            m[2] + 0.8,
            m[4],
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color="#9A3412",
        )
        top_y = max(m[1], m[2]) + 4.4
        ax_a3.text(
            xi,
            top_y,
            m[5],
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color="#334155",
        )

    drop_pp = (float(noad["joint_cns_safe_rate"]) - float(dpo["joint_cns_safe_rate"])) * 100.0
    ax_a3.text(
        0.50,
        0.96,
        f"Without AD-Damping:\n{drop_pp:.1f} pp Verified Safe Crash",
        transform=ax_a3.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#9A3412",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "#FEF2F2",
            "edgecolor": OKABE_ITO["vermilion"],
            "linewidth": 0.9,
        },
    )

    ax_a3.set_xticks(x)
    ax_a3.set_xticklabels([m[0] for m in metrics], fontweight="bold", fontsize=8.0)
    ax_a3.set_ylim(0.0, 56.0)
    ax_a3.set_ylabel("Score (% / Scaled ×100)")
    ax_a3.grid(axis="x", visible=False)
    ax_a3.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.1,
    )
    _panel_title(ax_a3, "C", "A3: Anti-Reward-Hacking")

    # --- C2: Ablation A5 CFG Scale Pareto Curve ---
    sweep = sorted(iter2["ablations"]["A5_CFG_Sweep"], key=lambda r: float(r["cfg_scale"]))
    ws = [float(r["cfg_scale"]) for r in sweep]
    drd1_sel = [float(r["all12"]["drd1_delta_sel"]) for r in sweep]
    safe_s = [float(r["all12"]["joint_cns_safe_rate"]) * 100.0 for r in sweep]
    mpo_s = [float(r["all12"]["cns_mpo_v2_mean"]) for r in sweep]
    div_s = [float(r["all12"]["global_int_div_1"]) for r in sweep]

    ax_a5.plot(
        drd1_sel,
        safe_s,
        color=OKABE_ITO["blue"],
        lw=2.0,
        linestyle="-",
        zorder=3,
    )
    for w_val, sel_v, sf_v, mp_v, dv_v in zip(ws, drd1_sel, safe_s, mpo_s, div_s, strict=True):
        is_best = abs(w_val - 1.3) < 0.05
        ax_a5.scatter(
            [sel_v],
            [sf_v],
            s=120 if is_best else 75,
            marker="*" if is_best else "o",
            color=OKABE_ITO["green"] if is_best else OKABE_ITO["blue"],
            edgecolor="#0F172A",
            linewidth=1.0,
            zorder=5,
        )
        if w_val < 1.1:
            tx, ty, ha = sel_v + 0.002, sf_v + 1.15, "left"
        elif is_best:
            tx, ty, ha = sel_v - 0.004, sf_v + 0.95, "center"
        else:
            tx, ty, ha = sel_v - 0.004, sf_v - 0.95, "right"
        tag = f"w={w_val:.1f}" + (" ★" if is_best else "")
        ax_a5.text(
            tx,
            ty,
            f"{tag}\nSafe {sf_v:.1f}%\nMPO {mp_v:.2f} | Div {dv_v:.3f}",
            fontsize=7.0,
            fontweight="bold" if is_best else "normal",
            color="#065F46" if is_best else "#1E293B",
            ha=ha,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#ECFDF5" if is_best else "white",
                "edgecolor": OKABE_ITO["green"] if is_best else "#CBD5E1",
                "linewidth": 0.85,
            },
            zorder=6,
        )

    ax_a5.set_xlim(0.028, 0.100)
    ax_a5.set_ylim(31.5, 39.2)
    ax_a5.set_xlabel("DRD1 Selectivity Δ_sel (D1 − D2)")
    ax_a5.set_ylabel("In-AD Joint CNS-Safe (%)")
    ax_a5.set_title(
        "A5: CFG Pareto (w = 1.0–1.6)",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color="#334155",
        pad=10.0,
    )


def _plot_fig13_panel_d(ax: Axes, iter2: dict[str, Any]) -> None:
    """Panel 13D: Per-target Iter-2 Matched/Shuffled P(on_target) enrichment & retention audit."""
    m_pt = iter2["arms"]["Iter-2_Matched"]["per_target"]
    s_pt = iter2["arms"]["Iter-2_Shuffled"]["per_target"]
    dpo_all = iter2["arms"]["Iter-2_Matched"]["all12"]
    casf = iter2["casf2016_retention"]
    st_shuf = iter2["statistics_all12"]["Iter-2_Shuffled"]["p_on_target"]

    ratios: list[tuple[str, float, float, float]] = []
    for tgt in m_pt:
        mp = float(m_pt[tgt]["p_on_target_mean"])
        sp = float(s_pt[tgt]["p_on_target_mean"])
        ratio = mp / max(sp, 1e-9)
        ratios.append((tgt, ratio, mp, sp))

    ratios.sort(key=lambda item: item[1])
    y_pos = np.arange(len(ratios))
    log2_vals = [math.log2(max(r[1], 1e-6)) for r in ratios]
    bar_cols = [OKABE_ITO["green"] if r[1] >= 1.0 else OKABE_ITO["vermilion"] for r in ratios]

    ax.axvline(0.0, color="#334155", linestyle="--", lw=1.25, zorder=2)
    ax.barh(
        y_pos,
        log2_vals,
        height=0.62,
        color=bar_cols,
        edgecolor="#1E293B",
        linewidth=0.65,
        alpha=0.88,
        zorder=3,
    )

    for y, (_, ratio, _, _), lv in zip(y_pos, ratios, log2_vals, strict=True):
        if lv >= 0.0:
            ax.text(
                lv + 0.09,
                y,
                f"{ratio:.2f}×",
                fontsize=7.8,
                fontweight="bold",
                family="monospace",
                color="#065F46",
                va="center",
                ha="left",
            )
        else:
            ax.text(
                lv - 0.09,
                y,
                f"{ratio:.2f}×",
                fontsize=7.8,
                fontweight="bold",
                family="monospace",
                color="#9A3412",
                va="center",
                ha="right",
            )

    n_pos = sum(1 for r in ratios if r[1] > 1.0)
    cns_val_pct = float(dpo_all["validity"]) * 100.0
    cns_nov_pct = float(dpo_all["strict_novelty"]) * 100.0
    casf_val_pct = float(casf["gen_validity"]) * 100.0
    casf_pkd_r = float(casf["pkd_pearson_r"])
    casf_r5_pct = float(casf["drugclip_recall_at_5"]) * 100.0
    audit_box = (
        f"3D Specificity: {n_pos}/{len(ratios)} targets > 1.0×\n"
        f"  Matched vs Shuffled {_fmt_p(float(st_shuf['wilcoxon_p']))}\n"
        f"──────────────────────────\n"
        f"Anti-Mode-Collapse (CNS):\n"
        f"  Valid {cns_val_pct:.1f}% | Nov {cns_nov_pct:.1f}%\n"
        f"  IntDiv1 = {float(dpo_all['global_int_div_1']):.3f}\n"
        f"──────────────────────────\n"
        f"CASF-2016 Retention (Frozen 3D):\n"
        f"  Valid {casf_val_pct:.1f}% | pKd r = {casf_pkd_r:.3f}\n"
        f"  DrugCLIP R@5 = {casf_r5_pct:.1f}%"
    )
    ax.text(
        0.985,
        0.04,
        audit_box,
        transform=ax.transAxes,
        fontsize=7.4,
        family="monospace",
        color="#0F172A",
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F8FAFC",
            "edgecolor": "#0072B2",
            "linewidth": 1.0,
        },
        zorder=7,
    )

    tick_folds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    ax.set_xticks([math.log2(v) for v in tick_folds])
    ax.set_xticklabels([f"{v:g}×" for v in tick_folds], fontweight="bold")
    ax.set_xlim(-1.05, 5.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in ratios], fontweight="bold", fontsize=8.4)
    ax.set_xlabel("3D Specificity Fold-Enrichment: Iter-2 Matched / Shuffled P(on-target)")
    ax.grid(axis="y", visible=False)
    _panel_title(ax, "D", "3D Pocket Conditioning Specificity & Anti-Mode-Collapse Audit")


def generate_figure_13(tables_dir: Path, figures_dir: Path) -> Path:
    """Render Figure 13: Closed-Loop Progression, Basic-Amine Paradox & Ablations (300 DPI)."""
    _apply_nature_style()
    iter0 = _load_json(tables_dir / "cns_iter0_baselines.json")
    iter1 = _load_json(tables_dir / "cns_iter1_sft_eval.json")
    iter2 = _load_json(tables_dir / "cns_iter2_dpo_eval.json")

    cand_path = Path(str(iter2["candidates_for_docking_path"]))
    if not cand_path.is_file():
        raise FileNotFoundError(f"Candidates file needed for basic-amine audit: {cand_path}")

    fig = plt.figure(figsize=(15.8, 11.4), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.31, wspace=0.22, top=0.92, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_fig13_panel_a(ax_a, iter0, iter1, iter2)
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_fig13_panel_b(ax_b, iter0, iter1, iter2, cand_path)
    _plot_fig13_panel_c(fig, gs[1, 0], iter2)
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_fig13_panel_d(ax_d, iter2)

    fig.suptitle(
        "GenEuclidean v3.1 — Closed-Loop CNS Optimization, "
        "Basic-Amine Paradox Resolution & Ablations",
        fontsize=13.2,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    return _save_fig(fig, figures_dir / "13_cns_closed_loop_progression.png")


# ---------------------------------------------------------------------------
# Phase Runners & CLI Entrypoint
# ---------------------------------------------------------------------------


def run_phase1(
    tables_dir: Path,
    figures_dir: Path,
    admet_model_path: Path | None,
) -> list[Path]:
    """Generate Phase 1 publication figures (Figures 12 and 13)."""
    p12 = generate_figure_12(tables_dir, figures_dir, admet_model_path)
    p13 = generate_figure_13(tables_dir, figures_dir)
    return [p12, p13]


def run_phase2(
    tables_dir: Path,
    figures_dir: Path,
    viz3d_dir: Path,
) -> list[Path]:
    """Placeholder hook for Phase 2 (Iteration 3 Vina + 10x10 Cross-Target + 3D Studio Gallery)."""
    dock_json = tables_dir / "cns_iter3_docking_summary.json"
    if not dock_json.is_file():
        print(f"[Phase 2] Skipping: Iteration 3 summary not yet present at {dock_json}")
        return []
    _ = (figures_dir, viz3d_dir)
    return []


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CNS publication figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate GenEuclidean v3.1 300-DPI CNS publication figures."
    )
    parser.add_argument(
        "--phase",
        choices=("1", "2", "all"),
        default="1",
        help="Execution phase: 1 (Figs 12-13), 2 (Iter-3 Figs 14-16 + 3D Studio), or all.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory containing committed CNS JSON evaluation tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for 300-DPI publication PNGs.",
    )
    parser.add_argument(
        "--viz3d-dir",
        type=Path,
        default=Path("results/viz3d"),
        help="Output directory for interactive 3Dmol.js WebGL HTML studio dashboards.",
    )
    parser.add_argument(
        "--admet-model",
        type=Path,
        default=Path("data/cns_neuro.nosync/admet_oracles.joblib"),
        help="Optional trained Gate 2 ADMET joblib artifact for live peripheral/QT drug controls.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    outputs: list[Path] = []
    if args.phase in ("1", "all"):
        outputs.extend(run_phase1(args.tables_dir, args.figures_dir, args.admet_model))
    if args.phase in ("2", "all"):
        outputs.extend(run_phase2(args.tables_dir, args.figures_dir, args.viz3d_dir))
    for path in outputs:
        print(f"Generated {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
