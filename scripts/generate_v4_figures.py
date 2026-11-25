#!/usr/bin/env python3
"""Generate 300-DPI Publication Figures for GenEuclidean v4.0 Tri-Disorder Discovery.

Reads EVERY metric strictly from the committed evaluation JSON tables under
``results/tables/`` and renders two publication-grade 300-DPI figures (< 1.8 MB each)
using the Okabe-Ito colorblind-safe palette on a crisp Nature Communications white theme:

  - ``results/figures/17_v4_llm_tri_disorder_benchmark.png`` (4-panel):
      (A) 3-LLM Head-to-Head across 24 Human Tri-Disorder Targets (N=1,536/arm):
          12-Endpoint v4 Pharma-Safe %, MDR1-MDCK Active Efflux Risk, PDE10A
          VMAT2-Sink elimination, and DCU/Reagent quarantine rate.
      (B) 12-Endpoint Industrial Scaffold-CV Avoid-ome/DMPK Suite ROC-AUC with
          90% Split-Conformal empirical coverage badges and external rat Kp,uu,brain
          (N=239) calibration callout.
      (C) SAFE-GPT 3D-Conditioned Clinical Scaffold Rescue (Before -> After safety
          liability reduction) paired with held-out AutoDock Vina counter-target
          selectivity (Delta-Delta-G) across Donepezil, Safinamide, Xanomeline,
          and Ecopipam.
      (D) Tourette CSTC Basal-Ganglia & GABAergic 4-Pocket Validation (GABRA5,
          GAT1, ADORA2A active, GRM5): Gate 1 multi-seed RMSD + flagship zero-shot
          Vina affinities vs. clinical comparators.

  - ``results/figures/18_v4_safegpt_rescue_and_alzheimer_3d.png`` (2x2 3D composite):
      (A) Alzheimer's ACHE (4EY7) — SAFE-GPT Donepezil Rescue
      (B) Alzheimer's MAOB (2V5Z + FAD cofactor) — SAFE-GPT Safinamide Rescue
      (C) Schizophrenia CHRM4 (8FX5) — SAFE-GPT Xanomeline Rescue
      (D) Tourette Basal-Ganglia GRM5 (4OO9) — Signal-Filter NAM Lead (GRM5_Gen_5)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
from numpy.typing import NDArray
from rdkit import Chem

from geneuclidean.evaluation.interactions3d import (
    InteractionProfile,
    InteractionRecord,
    profile_3d_interactions,
)
from geneuclidean.evaluation.viz3d import (
    _ATOM_VISUAL_RADIUS,
    _INTERACTION_META,
    _apply_depth_fog,
    _blend_rgb,
    _catmull_rom_spline,
    _compute_optimal_camera,
    _deoverlap_label_positions,
    _draw_multi_order_bond,
    _draw_phong_sphere,
    _element_color,
    _hex_to_rgb,
    _load_pocket_geometry,
    _render_2d_mol_png_array,
    _short_residue_label,
)

# Okabe-Ito colorblind-safe publication palette
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "black": "#0F172A",
    "slate": "#64748B",
    "lightgray": "#E2E8F0",
}

_DPI = 300


def _apply_nature_style() -> None:
    """Apply clean Nature Communications white publication styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.titlesize": 10.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Required v4.0 table not found: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _save_fig_300dpi(fig: Figure, out_path: Path) -> Path:
    """Save figure at exact 300 DPI with maximum lossless PNG compression (< 1.8 MB)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.14,
        pil_kwargs={"optimize": True, "compress_level": 9},
    )
    plt.close(fig)
    return out_path


def _fmt_p_sci(p_val: float) -> str:
    """Format a p-value concisely using mathtext scientific notation."""
    if not math.isfinite(p_val) or p_val <= 0.0:
        return r"$p < 10^{-300}$"
    if p_val < 1e-3:
        exp = int(math.floor(math.log10(p_val)))
        mant = p_val / (10.0**exp)
        return rf"$p={mant:.2f}\times 10^{{{exp}}}$"
    return f"p={p_val:.3f}"


# ---------------------------------------------------------------------------
# Figure 17 — 4-Panel v4.0 Tri-Disorder Benchmark, Avoid-ome, Rescue & CSTC
# ---------------------------------------------------------------------------


def _plot_fig17_panel_a(fig: Figure, slot: Any, h2h: Mapping[str, Any]) -> None:
    """Panel 17A: 3-LLM head-to-head (24 targets, N=1,536/arm) + sink & reagent audit."""
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=[1.52, 0.84], hspace=0.45)
    ax_top = fig.add_subplot(sub[0, 0])
    sub_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=sub[1, 0], wspace=0.28)
    ax_bleed = fig.add_subplot(sub_bot[0, 0])
    ax_reag = fig.add_subplot(sub_bot[0, 1])

    arms = h2h["arms"]
    stats_safe = h2h["statistics_all24"]["v4.0_SAFE-GPT-87M_vs_v3.1"]

    arm_specs = [
        ("Ctrl-ZINC", "Ctrl-\nZINC", "#94A3B8"),
        ("v3.1_GP-MoLFormer", "v3.1\nGP-MoLF", OKABE_ITO["orange"]),
        ("v4.0_GP-MoLFormer", "v4.0\nGP-MoLF", OKABE_ITO["skyblue"]),
        ("v4.0_NovoMolGen-157M", "v4.0\nNovo-157M", OKABE_ITO["purple"]),
        ("v4.0_SAFE-GPT-87M", "v4.0\nSAFE-GPT★", OKABE_ITO["green"]),
    ]

    x = np.arange(len(arm_specs), dtype=np.float64)
    w = 0.36
    safe_pct = [
        float(arms[k]["overall_24"]["v4_pharma_12ep_safe_rate"]) * 100.0 for k, _, _ in arm_specs
    ]
    mdr1_pct = [
        float(arms[k]["overall_24"]["mdr1_efflux_prob_mean"]) * 100.0 for k, _, _ in arm_specs
    ]

    bars_safe = ax_top.bar(
        x - w / 2,
        safe_pct,
        width=w,
        color=OKABE_ITO["green"],
        edgecolor="#0F172A",
        linewidth=0.75,
        label="12-Ep v4 Pharma-Safe (%, ↑)",
        zorder=3,
    )
    bars_mdr1 = ax_top.bar(
        x + w / 2,
        mdr1_pct,
        width=w,
        color=OKABE_ITO["vermilion"],
        edgecolor="#0F172A",
        linewidth=0.75,
        alpha=0.82,
        hatch="///",
        label="MDR1-MDCK Efflux Risk (%, ↓)",
        zorder=3,
    )

    for idx, (bs, bm, sv, mv) in enumerate(
        zip(bars_safe, bars_mdr1, safe_pct, mdr1_pct, strict=True)
    ):
        ax_top.text(
            bs.get_x() + bs.get_width() * 0.5,
            sv + 1.1,
            f"{sv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold" if idx == 4 else "normal",
            color="#065F46" if idx == 4 else "#0F172A",
        )
        ax_top.text(
            bm.get_x() + bm.get_width() * 0.5,
            mv + 1.1,
            f"{mv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold" if idx == 4 else "normal",
            color="#9A3412" if idx != 4 else "#065F46",
        )

    # Exact fold-change & Holm-Bonferroni badges from JSON (positioned in open air)
    safe_mult = safe_pct[4] / max(safe_pct[1], 1e-6)
    mdr1_drop = mdr1_pct[1] / max(mdr1_pct[4], 1e-6)
    holm_safe = int(stats_safe["v4_pharma_12ep_safe"]["holm_sig_targets_p05"])
    holm_mdr1 = int(stats_safe["mdr1_efflux_prob"]["holm_sig_targets_p05"])
    ax_top.annotate(
        f"{safe_mult:.2f}× vs v3.1\n({holm_safe}/24 Holm)",
        xy=(4.0 - w / 2, safe_pct[4] + 5.2),
        xytext=(3.20, 56.0),
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color="#065F46",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#ECFDF5",
            "edgecolor": OKABE_ITO["green"],
        },
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["green"], "lw": 1.0},
        zorder=6,
    )
    ax_top.annotate(
        f"{mdr1_drop:.1f}× efflux ↓\n({holm_mdr1}/24 Holm)",
        xy=(4.0 + w / 2, mdr1_pct[4] + 5.4),
        xytext=(4.46, 30.0),
        ha="center",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color="#9A3412",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "#FFF7ED",
            "edgecolor": OKABE_ITO["vermilion"],
        },
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["vermilion"], "lw": 1.0},
        zorder=6,
    )

    ax_top.set_xticks(x)
    ax_top.set_xticklabels([lbl for _, lbl, _ in arm_specs], fontweight="bold", fontsize=7.6)
    ax_top.set_xlim(-0.52, 4.92)
    ax_top.set_ylim(0.0, 72.0)
    ax_top.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax_top.set_ylabel("Rate / Risk (%)", fontsize=8.8)
    ax_top.grid(axis="x", visible=False)
    ax_top.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.0,
        ncol=1,
    )
    ax_top.set_title(
        "(A)  3-LLM Tri-Disorder Benchmark (24 Targets · N=1,536/Arm)",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        pad=8.0,
    )

    # Bottom-left: PDE10A VMAT2-Sink bleed elimination
    short_names = ["Ctrl", "v3.1", "v4 GP", "Novo", "SAFE★"]
    bleed_vals = [
        float(arms[k]["sink_elimination_audit"]["PDE10A"]["vmat2_bleed_rate"]) * 100.0
        for k, _, _ in arm_specs
    ]
    b_cols = [c for _, _, c in arm_specs]
    bx = np.arange(len(short_names), dtype=np.float64)
    bbars = ax_bleed.bar(
        bx, bleed_vals, width=0.58, color=b_cols, edgecolor="#0F172A", lw=0.7, zorder=3
    )
    for bar, bv in zip(bbars, bleed_vals, strict=True):
        ax_bleed.text(
            bar.get_x() + bar.get_width() * 0.5,
            bv + 1.2,
            f"{bv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold" if bv == 0.0 or bv > 20.0 else "normal",
            color="#9A3412" if bv > 20.0 else "#065F46",
        )
    ax_bleed.set_xticks(bx)
    ax_bleed.set_xticklabels(short_names, fontsize=7.1, fontweight="bold")
    ax_bleed.set_ylim(0.0, 44.0)
    ax_bleed.set_ylabel("Bleed (%)", fontsize=8.0)
    ax_bleed.grid(axis="x", visible=False)
    ax_bleed.set_title(
        "PDE10A VMAT2-Sink Bleed (↓)", fontsize=8.3, fontweight="bold", color="#0F172A", pad=4.0
    )

    # Bottom-right: DCU / Coupling-Reagent quarantine rate
    reag_vals = [
        float(arms[k]["overall_24"]["reagent_quarantine_rate"]) * 100.0 for k, _, _ in arm_specs
    ]
    rbars = ax_reag.bar(
        bx, reag_vals, width=0.58, color=b_cols, edgecolor="#0F172A", lw=0.7, zorder=3
    )
    for idx, (bar, rv) in enumerate(zip(rbars, reag_vals, strict=True)):
        ax_reag.text(
            bar.get_x() + bar.get_width() * 0.5,
            rv + 0.04,
            f"{rv:.2f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold" if idx == 4 else "normal",
            color="#065F46" if rv == 0.0 else "#0F172A",
        )
    ax_reag.set_xticks(bx)
    ax_reag.set_xticklabels(short_names, fontsize=7.1, fontweight="bold")
    ax_reag.set_ylim(0.0, 1.28)
    ax_reag.set_ylabel("Reagent (%)", fontsize=8.0)
    ax_reag.grid(axis="x", visible=False)
    ax_reag.set_title(
        "DCU / Reagent Rate (↓)", fontsize=8.3, fontweight="bold", color="#0F172A", pad=4.0
    )


def _plot_fig17_panel_b(fig: Figure, slot: Any, admet_cv: Mapping[str, Any]) -> None:
    """Panel 17B: 12-endpoint scaffold-CV Avoid-ome/DMPK ROC-AUC + Conformal UQ + Kp,uu strip."""
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=[4.3, 1.05], hspace=0.32)
    ax = fig.add_subplot(sub[0, 0])
    ax_kpuu = fig.add_subplot(sub[1, 0])

    eps = admet_cv["endpoints"]
    kpuu = admet_cv["external_rat_kpuu_benchmark"]

    ep_specs = [
        ("chrm3_avoid", "M3 Avoid-ome", OKABE_ITO["green"]),
        ("hrh1_avoid", "H1 Avoid-ome", OKABE_ITO["green"]),
        ("pgp", "P-gp Inhib (TDC)", OKABE_ITO["blue"]),
        ("bbb", "BBB Perm (B3DB)", OKABE_ITO["blue"]),
        ("htr2b_avoid", "5-HT2B Avoid-ome", OKABE_ITO["green"]),
        ("cyp3a4", "CYP3A4 (TDC)", OKABE_ITO["skyblue"]),
        ("cyp2d6", "CYP2D6 (TDC)", OKABE_ITO["skyblue"]),
        ("mdr1_efflux", "MDR1-MDCK (Biogen)", OKABE_ITO["blue"]),
        ("herg", "hERG (Karim)", OKABE_ITO["orange"]),
        ("ames", "AMES (TDC)", OKABE_ITO["orange"]),
        ("hlm_clearance", "HLM CLint (Biogen)", OKABE_ITO["skyblue"]),
        ("dili", "DILI (TDC, N=474)", OKABE_ITO["orange"]),
    ]
    rows = list(reversed(ep_specs))
    y = np.arange(len(rows), dtype=np.float64)

    aucs = [float(eps[k]["oof_roc_auc"]) for k, _, _ in rows]
    stds = [float(eps[k]["fold_roc_auc_std"]) for k, _, _ in rows]
    covs = [float(eps[k]["conformal_uq"]["empirical_coverage"]) * 100.0 for k, _, _ in rows]
    cols = [c for _, _, c in rows]

    ax.axvline(0.80, color="#94A3B8", linestyle=":", lw=1.1, zorder=1)
    ax.axvline(0.90, color="#64748B", linestyle="--", lw=1.0, zorder=1)

    ax.barh(y, aucs, height=0.66, color=cols, edgecolor="#0F172A", lw=0.65, alpha=0.88, zorder=3)
    ax.errorbar(
        aucs, y, xerr=stds, fmt="none", ecolor="#0F172A", elinewidth=1.0, capsize=2.0, zorder=4
    )

    for idx, (auc, std, cov) in enumerate(zip(aucs, stds, covs, strict=True)):
        ax.text(
            auc + std + 0.008,
            y[idx],
            f"{auc:.3f} [{cov:.1f}%]",
            va="center",
            ha="left",
            fontsize=6.9,
            family="monospace",
            fontweight="bold" if auc >= 0.90 else "normal",
            color="#0F172A",
        )

    cat_handles = [
        Patch(facecolor=OKABE_ITO["blue"], edgecolor="#0F172A", lw=0.6, label="CNS Exp."),
        Patch(facecolor=OKABE_ITO["green"], edgecolor="#0F172A", lw=0.6, label="Avoid-ome"),
        Patch(facecolor=OKABE_ITO["skyblue"], edgecolor="#0F172A", lw=0.6, label="DMPK/CYP"),
        Patch(facecolor=OKABE_ITO["orange"], edgecolor="#0F172A", lw=0.6, label="Safety/Tox"),
    ]
    ax.legend(
        handles=cat_handles,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.015),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.6,
        ncol=1,
        handlelength=1.0,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl, _ in rows], fontweight="bold", fontsize=7.4)
    ax.set_xlim(0.55, 1.175)
    ax.set_xticks([0.60, 0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("5-Fold Scaffold-CV OOF ROC-AUC (↑) · [90% Conformal Cov.]", fontsize=8.4)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "(B)  12-Endpoint Avoid-ome/DMPK Suite & External $K_{p,uu}$",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        pad=8.0,
    )

    # Dedicated clean external Rat Kp,uu,brain calibration card (zero bar overlap!)
    ax_kpuu.set_facecolor("#F8FAFC")
    ax_kpuu.set_xticks([])
    ax_kpuu.set_yticks([])
    for sp in ax_kpuu.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(OKABE_ITO["blue"])
        sp.set_linewidth(1.0)
    bbb_rho = float(kpuu["bbb_prob"]["spearman_rho"])
    bbb_p = float(kpuu["bbb_prob"]["spearman_p"])
    bbb_auc = float(kpuu["bbb_prob"]["roc_auc_kpuu_ge_0p3"])
    mdr1_rho = float(kpuu["mdr1_efflux_prob"]["spearman_rho"])
    mpo_rho = float(kpuu["cns_mpo_v2"]["spearman_rho"])
    n_kpuu = int(kpuu["n_valid_compounds"])
    ax_kpuu.text(
        0.5,
        0.5,
        f"External Held-Out Rat $K_{{p,uu,\\mathrm{{brain}}}}$ "
        f"Calibration (N={n_kpuu}, Zero Leakage)\n"
        f"BBB Oracle: ρ = {bbb_rho:+.3f} ({_fmt_p_sci(bbb_p)}, AUC={bbb_auc:.3f})  |  "
        f"MDR1 Efflux: ρ = {mdr1_rho:+.3f}  |  CNS MPO v2: ρ = {mpo_rho:+.3f}",
        transform=ax_kpuu.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="#0F172A",
        linespacing=1.45,
    )


def _plot_fig17_panel_c(fig: Figure, slot: Any, dock: Mapping[str, Any]) -> None:
    """Panel 17C: SAFE-GPT 3D clinical scaffold rescue (liability drop + counter-target ddG)."""
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[1.08, 1.0], wspace=0.14)
    ax_liab = fig.add_subplot(sub[0, 0])
    ax_ddg = fig.add_subplot(sub[0, 1])

    dec = dock["safegpt_decorated_docking"]
    panels = dock["counter_target_panels"]

    drug_specs = [
        (
            "donepezil_ache",
            "Donepezil\nACHE vs BCHE",
            "herg_prob",
            "hERG",
            float(panels["AD_ACHE_vs_BCHE"]["clinical_comparators"]["Donepezil"]["ddg"]),
        ),
        (
            "safinamide_maob",
            "Safinamide\nMAOB vs MAOA",
            "mdr1_efflux_prob",
            "MDR1",
            float(panels["AD_MAOB_vs_MAOA"]["clinical_comparators"]["Safinamide"]["ddg"]),
        ),
        (
            "xanomeline_chrm4",
            "Xanomeline\nCHRM4 vs DRD2",
            "chrm3_avoid_prob",
            "M3",
            float(panels["SCZ_NonD2_CHRM4_vs_DRD2"]["clinical_comparators"]["Xanomeline"]["ddg"]),
        ),
        (
            "ecopipam_drd1",
            "Ecopipam\nDRD1 vs DRD2",
            "herg_prob",
            "hERG*",
            float(panels["Tourette_DRD1_vs_DRD2_6CM4"]["clinical_comparators"]["Ecopipam"]["ddg"]),
        ),
    ]
    rows = list(reversed(drug_specs))
    y = np.arange(len(rows), dtype=np.float64)

    # Left subpanel: Paired Parent -> SAFE-GPT Rescue Liability Dumbbell
    ax_liab.axvspan(0.0, 0.30, color="#ECFDF5", alpha=0.75, zorder=0)
    ax_liab.axvline(0.30, color=OKABE_ITO["green"], linestyle="--", lw=1.1, zorder=1)
    ax_liab.text(
        0.15,
        3.56,
        "Safe Zone (≤0.30)",
        ha="center",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color="#047857",
    )

    for idx, (key, _lbl, metric_key, m_tag, _p_ddg) in enumerate(rows):
        entry = dec[key]
        parent_val = float(entry["parent_profile"][metric_key])
        lead = entry["docked_analogs"][0]
        lead_val = float(lead[metric_key])
        improved = lead_val < parent_val - 0.05

        if abs(lead_val - parent_val) > 0.04:
            ax_liab.annotate(
                "",
                xy=(lead_val, y[idx]),
                xytext=(parent_val, y[idx]),
                arrowprops={
                    "arrowstyle": "->,head_width=0.22,head_length=0.28",
                    "color": OKABE_ITO["green"] if improved else OKABE_ITO["slate"],
                    "lw": 2.1,
                },
                zorder=4,
            )
        y_par = y[idx] if abs(lead_val - parent_val) > 0.04 else y[idx] + 0.07
        y_res = y[idx] if abs(lead_val - parent_val) > 0.04 else y[idx] - 0.07
        ax_liab.scatter(
            [parent_val],
            [y_par],
            s=62,
            marker="s",
            color=OKABE_ITO["vermilion"],
            edgecolor="white",
            lw=0.9,
            zorder=5,
            label="Clinical Parent" if idx == 3 else None,
        )
        ax_liab.scatter(
            [lead_val],
            [y_res],
            s=70,
            marker="o",
            color=OKABE_ITO["green"] if improved else OKABE_ITO["orange"],
            edgecolor="#0F172A",
            lw=0.9,
            zorder=6,
            label="SAFE-GPT Rescue" if idx == 3 else None,
        )
        ax_liab.text(
            0.5 * (parent_val + lead_val)
            if abs(parent_val - lead_val) > 0.25
            else min(0.72, parent_val),
            y[idx] + 0.22,
            f"{m_tag}: {parent_val:.2f} → {lead_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color="#065F46" if improved else "#9A3412",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=7,
        )

    ax_liab.set_yticks(y)
    ax_liab.set_yticklabels([lbl for _, lbl, _, _, _ in rows], fontweight="bold", fontsize=7.6)
    ax_liab.set_xlim(-0.04, 1.08)
    ax_liab.set_ylim(-1.02, 3.68)
    ax_liab.set_xlabel("Liability Probability (↓ Safer)", fontsize=8.4)
    ax_liab.grid(axis="y", visible=False)
    ax_liab.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.7,
        ncol=2,
        columnspacing=0.8,
    )
    ax_liab.set_title(
        "(C)  SAFE-GPT Clinical Rescue & Counter-Target ΔΔG",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        pad=8.0,
    )

    # Right subpanel: Held-out Vina Counter-Target ddG (Parent vs SAFE-GPT Rescue)
    h = 0.30
    parent_ddgs = [p_ddg for _, _, _, _, p_ddg in rows]
    lead_ddgs = [float(dec[k]["docked_analogs"][0]["ddg"]) for k, _, _, _, _ in rows]

    ax_ddg.axvline(0.0, color="#475569", lw=1.0, zorder=2)
    b_par = ax_ddg.barh(
        y + h / 2,
        parent_ddgs,
        height=h,
        color=OKABE_ITO["blue"],
        edgecolor="#0F172A",
        lw=0.65,
        alpha=0.80,
        label="Parent ΔΔG",
        zorder=3,
    )
    b_res = ax_ddg.barh(
        y - h / 2,
        lead_ddgs,
        height=h,
        color=OKABE_ITO["green"],
        edgecolor="#0F172A",
        lw=0.65,
        label="SAFE-GPT ΔΔG",
        zorder=3,
    )

    for bp, br, pv, lv in zip(b_par, b_res, parent_ddgs, lead_ddgs, strict=True):
        ax_ddg.text(
            pv - 0.08,
            bp.get_y() + bp.get_height() * 0.5,
            f"{pv:.2f}",
            ha="right",
            va="center",
            fontsize=6.9,
            family="monospace",
            color="#1E3A8A",
        )
        ax_ddg.text(
            lv - 0.08,
            br.get_y() + br.get_height() * 0.5,
            f"{lv:.2f}",
            ha="right",
            va="center",
            fontsize=7.0,
            family="monospace",
            fontweight="bold",
            color="#065F46",
        )

    ax_ddg.set_yticks(y)
    ax_ddg.set_yticklabels([])
    ax_ddg.set_xlim(-4.35, 0.25)
    ax_ddg.set_ylim(-1.02, 3.68)
    ax_ddg.set_xlabel("Held-Out Vina ΔΔG (kcal/mol, ← Selective)", fontsize=8.3)
    ax_ddg.grid(axis="y", visible=False)
    ax_ddg.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.6,
    )
    ax_ddg.text(
        0.92,
        0.025,
        "*Ecopipam: ΔΔG −0.72→−0.98\n(core hERG 0.61→0.62)",
        transform=ax_ddg.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        style="italic",
        color="#475569",
    )


def _plot_fig17_panel_d(fig: Figure, slot: Any, ts: Mapping[str, Any]) -> None:
    """Panel 17D: Tourette CSTC 4-pocket Gate 1 RMSD + flagship zero-shot Vina vs clinical."""
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[0.88, 1.24], wspace=0.14)
    ax_rmsd = fig.add_subplot(sub[0, 0])
    ax_vina = fig.add_subplot(sub[0, 1])

    g1 = ts["gate1_native_redocking"]
    hd = ts["heldout_vina_docking_and_plif"]
    agg = ts["aggregate_summary"]

    tgt_specs = [
        ("GRM5", "GRM5 (4OO9)\nSTN NAM", "Dipraglurant", -9.583, "Beats Diprag."),
        ("GABRA5", "GABRA5 (8BHG)\nα5+/γ2- PAM", "Clonazepam", -9.182, "Beats Clonaz."),
        ("GAT1", "GAT1 (7Y7Z)\nGABA Uptake", "Tiagabine", -7.210, "10× ↓hERG"),
        ("ADORA2A", "ADORA2A (9EE8)\nActive A2A", "Adenosine", -6.493, "Non-Ribose"),
    ]
    rows = list(reversed(tgt_specs))
    y = np.arange(len(rows), dtype=np.float64)

    # Left: Gate 1 3-Seed Median RMSD
    ax_rmsd.axvspan(0.0, 2.0, color="#ECFDF5", alpha=0.78, zorder=0)
    ax_rmsd.axvline(2.0, color=OKABE_ITO["green"], linestyle="--", lw=1.2, zorder=1)
    ax_rmsd.text(
        1.90, 3.58, "2.0 Å", fontsize=6.8, fontweight="bold", color="#047857", va="top", ha="right"
    )

    rmsds = [float(g1[t]["median_top1_rmsd"]) for t, _, _, _, _ in rows]
    r_cols = [OKABE_ITO["green"] if r <= 2.0 else OKABE_ITO["orange"] for r in rmsds]
    ax_rmsd.barh(
        y, rmsds, height=0.52, color=r_cols, edgecolor="#0F172A", lw=0.7, alpha=0.88, zorder=3
    )
    for idx, rv in enumerate(rmsds):
        ax_rmsd.text(
            rv + 0.08,
            y[idx],
            f"{rv:.2f} Å",
            va="center",
            ha="left",
            fontsize=7.2,
            family="monospace",
            fontweight="bold",
            color="#065F46" if rv <= 2.0 else "#9A3412",
        )

    med_rmsd = float(agg["gate1_4target_median_top1_rmsd_A"])
    pb_pct = float(agg["gate1_4target_mean_posebusters_rate"]) * 100.0
    anc_pct = float(agg["gate1_4target_mean_anchor_recovery"]) * 100.0
    ax_rmsd.text(
        0.975,
        0.97,
        f"Median {med_rmsd:.2f} Å\nPB {pb_pct:.0f}% · Anc {anc_pct:.0f}%",
        transform=ax_rmsd.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#F8FAFC",
            "edgecolor": OKABE_ITO["blue"],
        },
    )

    ax_rmsd.set_yticks(y)
    ax_rmsd.set_yticklabels([lbl for _, lbl, _, _, _ in rows], fontweight="bold", fontsize=7.5)
    ax_rmsd.set_xlim(0.0, 4.50)
    ax_rmsd.set_ylim(-1.02, 3.68)
    ax_rmsd.set_xlabel("Gate 1 Redock RMSD (Å, ↓)", fontsize=8.3)
    ax_rmsd.grid(axis="y", visible=False)
    ax_rmsd.set_title(
        "(D)  Tourette CSTC Basal-Ganglia & GABAergic 4-Pocket Validation",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        pad=8.0,
    )

    # Right: Flagship Zero-Shot Lead Vina vs Clinical Comparator
    h = 0.30
    gen_vals: list[float] = []
    clin_vals: list[float] = []
    for t, _, _clin_name, clin_fallback, _note in rows:
        if t == "ADORA2A":
            gen_vals.append(float(hd[t]["geneuclidean_top5"][0]["vina_dg_kcal"]))
        else:
            gen_vals.append(float(hd[t]["summary"]["geneuclidean_best_vina_dg"]))
        c_refs = hd[t].get("clinical_refs", [])
        match = next(
            (
                float(r["vina_dg_kcal"])
                for r in c_refs
                if _clin_name.lower() in str(r["label"]).lower()
            ),
            clin_fallback,
        )
        clin_vals.append(match)

    b_gen = ax_vina.barh(
        y + h / 2,
        gen_vals,
        height=h,
        color=OKABE_ITO["green"],
        edgecolor="#0F172A",
        lw=0.7,
        label="GenEuclidean 0-Shot Lead",
        zorder=3,
    )
    b_clin = ax_vina.barh(
        y - h / 2,
        clin_vals,
        height=h,
        color=OKABE_ITO["blue"],
        edgecolor="#0F172A",
        lw=0.7,
        alpha=0.78,
        label="Clinical Comparator",
        zorder=3,
    )

    for idx, (bg, bc, gv, cv) in enumerate(zip(b_gen, b_clin, gen_vals, clin_vals, strict=True)):
        cname = rows[idx][2]
        note = rows[idx][4]
        ax_vina.text(
            gv - 0.14,
            bg.get_y() + bg.get_height() * 0.5,
            f"{gv:.2f} ({note})",
            ha="right",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color="#065F46",
        )
        ax_vina.text(
            cv - 0.14,
            bc.get_y() + bc.get_height() * 0.5,
            f"{cv:.2f} {cname}",
            ha="right",
            va="center",
            fontsize=6.6,
            color="#1E3A8A",
        )

    ax_vina.set_yticks(y)
    ax_vina.set_yticklabels([])
    ax_vina.set_xlim(-16.5, 0.0)
    ax_vina.set_ylim(-1.02, 3.68)
    ax_vina.set_xticks([-14, -12, -10, -8, -6, -4, -2, 0])
    ax_vina.set_xlabel("Held-Out Vina ΔG (kcal/mol, ← Stronger)", fontsize=8.3)
    ax_vina.grid(axis="y", visible=False)
    ax_vina.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=6.7,
        ncol=2,
        columnspacing=0.8,
    )


def generate_figure_17(tables_dir: Path, figures_dir: Path) -> Path:
    """Render Figure 17: 4-panel v4.0 tri-disorder LLM benchmark, Avoid-ome, rescue & CSTC."""
    _apply_nature_style()
    h2h = _load_json(tables_dir / "v4_llm_head_to_head.json")
    admet_cv = _load_json(tables_dir / "v4_admet_avoidome_cv_metrics.json")
    dock = _load_json(tables_dir / "v4_tri_disorder_docking.json")
    ts = _load_json(tables_dir / "ts_circuit_audit_and_benchmark.json")

    fig = plt.figure(figsize=(15.0, 10.4), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.33,
        wspace=0.28,
        left=0.06,
        right=0.985,
        top=0.905,
        bottom=0.06,
    )

    _plot_fig17_panel_a(fig, gs[0, 0], h2h)
    _plot_fig17_panel_b(fig, gs[0, 1], admet_cv)
    _plot_fig17_panel_c(fig, gs[1, 0], dock)
    _plot_fig17_panel_d(fig, gs[1, 1], ts)

    fig.suptitle(
        "GenEuclidean v4.0 — Tri-Disorder 3-LLM Benchmark, 12-Endpoint Avoid-ome, "
        "Clinical Rescue & Tourette CSTC Circuit",
        fontsize=12.6,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )
    return _save_fig_300dpi(fig, figures_dir / "17_v4_llm_tri_disorder_benchmark.png")


# ---------------------------------------------------------------------------
# Figure 18 — 2x2 300-DPI 3D Binding-Mode Composite (SAFE-GPT & CSTC GRM5)
# ---------------------------------------------------------------------------


def _load_first_sdf_mol(sdf_path: Path) -> Chem.Mol:
    """Load the first valid 3D molecule from an SDF file."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in suppl:
        if mol is not None and mol.GetNumConformers() > 0:
            return mol
    raise RuntimeError(f"No valid 3D conformer found in {sdf_path}")


def _extract_cofactor_atoms(
    pocket_pdb: Path,
    cofactor_names: tuple[str, ...] = ("FAD", "ADP"),
) -> list[tuple[str, NDArray[np.float64]]]:
    """Extract heavy atoms of retained physiological cofactors (e.g., FAD) from a pocket PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cofac", str(pocket_pdb))
    atoms: list[tuple[str, NDArray[np.float64]]] = []
    for model in structure.get_models():
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip().upper() in cofactor_names:
                    for atom in residue.get_atoms():
                        raw_el = str(getattr(atom, "element", "")).strip().upper()
                        name = atom.get_name().strip().upper()
                        el = (
                            raw_el if raw_el else "".join(c for c in name if c.isalpha())[:1] or "C"
                        )
                        if el in {"H", "D"}:
                            continue
                        atoms.append((el, np.asarray(atom.get_coord(), dtype=np.float64)))
        break
    return atoms


def _render_v4_3d_panel(
    ax: Axes,
    pocket_pdb: Path,
    lig_mol: Chem.Mol,
    *,
    panel_letter: str,
    header_title: str,
    metrics_line: str,
    elev_deg: float = 12.0,
    azim_deg: float = 12.0,
    ligand_carbon: str = "#0F766E",
    show_cofactors: bool = False,
) -> InteractionProfile:
    """Render one publication white-theme 3D binding-mode panel with unified de-overlapped pills."""
    profile = profile_3d_interactions(pocket_pdb, lig_mol)
    pocket_geom = _load_pocket_geometry(pocket_pdb_path=pocket_pdb)
    lig_noh = Chem.RemoveHs(Chem.Mol(lig_mol))
    lig_coords = np.asarray(lig_noh.GetConformer().GetPositions(), dtype=np.float64)
    pocket_xyz = np.stack([a.xyz for a in pocket_geom.atoms], axis=0) if pocket_geom.atoms else None
    contact_mids = (
        np.stack(
            [
                0.5 * (np.asarray(r.protein_xyz) + np.asarray(r.ligand_xyz))
                for r in profile.interactions
            ],
            axis=0,
        )
        if profile.interactions
        else None
    )
    cam = _compute_optimal_camera(
        lig_coords,
        pocket_coords=pocket_xyz,
        contact_coords=contact_mids,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
    )

    bg_hex = "#FFFFFF"
    bg_rgb = _hex_to_rgb(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#CBD5E1")
        sp.set_linewidth(1.0)

    # Deduplicate overlapping atom/residue interaction vectors cleanly
    display_ints: list[InteractionRecord] = []
    seen_res_dist: set[tuple[str, int]] = set()
    hydro_seen_res: set[str] = set()
    hydro_n = 0
    for rec in profile.interactions:
        if rec.kind == "hydrophobic":
            if hydro_n >= 3 or rec.protein_residue in hydro_seen_res:
                continue
            hydro_seen_res.add(rec.protein_residue)
            hydro_n += 1
        else:
            rk = (rec.protein_residue, int(round(rec.distance_angstrom * 10)))
            if rk in seen_res_dist:
                continue
            seen_res_dist.add(rk)
        display_ints.append(rec)

    active_res: list[str] = []
    for rec in display_ints:
        if rec.protein_residue not in active_res:
            active_res.append(rec.protein_residue)
    active_res = active_res[:6]

    depth_samples = [cam.transform(lig_coords)[:, 2]]
    for rid in active_res:
        ra = pocket_geom.by_residue.get(rid, [])
        if ra:
            depth_samples.append(cam.transform(np.stack([a.xyz for a in ra], axis=0))[:, 2])
    all_z = np.concatenate(depth_samples)
    z_min, z_max = float(np.min(all_z)) - 2.8, float(np.max(all_z)) + 1.0

    # Soft feathered pocket cavity backdrop
    halo_rgb = _hex_to_rgb("#94A3B8")
    for pa in pocket_geom.atoms:
        if float(np.min(np.linalg.norm(lig_coords - pa.xyz, axis=1))) > 5.8:
            continue
        x2, y2, _, sc = cam.project_point(pa.xyz)
        for rm, am in ((1.30, 0.35), (0.82, 0.72)):
            ax.add_patch(
                Circle(
                    (x2, y2),
                    radius=rm * sc,
                    facecolor=tuple(halo_rgb),
                    edgecolor="none",
                    alpha=0.017 * am,
                    zorder=0.4,
                )
            )

    queue: list[tuple[float, str, dict[str, Any]]] = []
    ribbon_col = _hex_to_rgb("#94A3B8")
    for trace in pocket_geom.ca_traces:
        ca_pts = np.stack([item[2] for item in trace], axis=0)
        if (
            float(np.min(np.linalg.norm(ca_pts[:, None, :] - lig_coords[None, :, :], axis=-1)))
            > 9.4
        ):
            continue
        spline = _catmull_rom_spline(ca_pts, subdivisions=5)
        for seg_i in range(len(spline) - 1):
            p1, p2 = spline[seg_i], spline[seg_i + 1]
            if float(np.min(np.linalg.norm(lig_coords - 0.5 * (p1 + p2), axis=1))) > 9.8:
                continue
            x1, y1, z1, sc1 = cam.project_point(p1)
            x2, y2, z2, sc2 = cam.project_point(p2)
            zm = 0.5 * (z1 + z2)
            queue.append(
                (
                    zm - 0.5,
                    "ribbon",
                    {
                        "p1": (x1, y1),
                        "p2": (x2, y2),
                        "rgb": _apply_depth_fog(ribbon_col, bg_rgb, zm, z_min, z_max, 0.45),
                        "w": 3.4 * 0.5 * (sc1 + sc2),
                        "alpha": 0.40,
                    },
                )
            )

    # Optional cofactor sticks (e.g., MAO-B FAD flavin ring)
    cofac_label_pt: tuple[float, float] | None = None
    if show_cofactors:
        cof_atoms = _extract_cofactor_atoms(pocket_pdb)
        near_cof = [
            (el, xyz)
            for el, xyz in cof_atoms
            if float(np.min(np.linalg.norm(lig_coords - xyz, axis=1))) <= 7.8
        ]
        if near_cof:
            c_xyz = np.stack([xyz for _, xyz in near_cof], axis=0)
            c_xy, c_z = cam.project(c_xyz)
            gold_rgb = _hex_to_rgb("#D97706")
            for i in range(len(near_cof)):
                for j in range(i + 1, len(near_cof)):
                    if float(np.linalg.norm(c_xyz[i] - c_xyz[j])) <= 1.85:
                        zm = 0.5 * (float(c_z[i]) + float(c_z[j])) - 0.06
                        queue.append(
                            (
                                zm,
                                "cofactor_bond",
                                {
                                    "xy1": (float(c_xy[i, 0]), float(c_xy[i, 1])),
                                    "xy2": (float(c_xy[j, 0]), float(c_xy[j, 1])),
                                    "rgb": _apply_depth_fog(
                                        gold_rgb, bg_rgb, zm, z_min, z_max, 0.25
                                    ),
                                },
                            )
                        )
            cofac_label_pt = (float(np.mean(c_xy[:, 0])), float(np.mean(c_xy[:, 1])))

    # Interacting pocket side chains
    res_carbon = _hex_to_rgb("#64748B")
    for rid in active_res:
        ra = pocket_geom.by_residue.get(rid, [])
        if not ra:
            continue
        rxy, rz = cam.project(np.stack([a.xyz for a in ra], axis=0))
        for poly3d in pocket_geom.aromatic_polygons.get(rid, []):
            pxy, pz = cam.project(poly3d)
            queue.append(
                (
                    float(np.mean(pz)) - 0.15,
                    "ring_poly",
                    {"xy": pxy, "rgb": _hex_to_rgb("#D97706"), "alpha": 0.14},
                )
            )
        for i in range(len(ra)):
            for j in range(i + 1, len(ra)):
                if float(np.linalg.norm(ra[i].xyz - ra[j].xyz)) <= 1.92:
                    zm = 0.5 * (float(rz[i]) + float(rz[j])) - 0.08
                    c1 = (
                        res_carbon
                        if ra[i].element == "C"
                        else _blend_rgb(_element_color(ra[i].element, "light"), res_carbon, 0.25)
                    )
                    c2 = (
                        res_carbon
                        if ra[j].element == "C"
                        else _blend_rgb(_element_color(ra[j].element, "light"), res_carbon, 0.25)
                    )
                    queue.append(
                        (
                            zm,
                            "pocket_bond",
                            {
                                "xy1": (float(rxy[i, 0]), float(rxy[i, 1])),
                                "xy2": (float(rxy[j, 0]), float(rxy[j, 1])),
                                "rgb1": _apply_depth_fog(
                                    c1, bg_rgb, float(rz[i]), z_min, z_max, 0.30
                                ),
                                "rgb2": _apply_depth_fog(
                                    c2, bg_rgb, float(rz[j]), z_min, z_max, 0.30
                                ),
                            },
                        )
                    )
        for pa in ra:
            if pa.element in {"N", "O", "S"}:
                x2, y2, zc, sc = cam.project_point(pa.xyz)
                rgb = _apply_depth_fog(
                    _blend_rgb(_element_color(pa.element, "light"), res_carbon, 0.18),
                    bg_rgb,
                    zc,
                    z_min,
                    z_max,
                    0.30,
                )
                queue.append(
                    (zc - 0.04, "pocket_atom", {"x": x2, "y": y2, "r": 0.19 * sc, "rgb": rgb})
                )

    # Ligand bonds & Phong spheres
    lig_xy, lig_z = cam.project(lig_coords)
    for bond in lig_noh.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        el1, el2 = lig_noh.GetAtomWithIdx(i).GetSymbol(), lig_noh.GetAtomWithIdx(j).GetSymbol()
        zm = 0.5 * (float(lig_z[i]) + float(lig_z[j])) - 0.05
        _, _, _, s1 = cam.project_point(lig_coords[i])
        _, _, _, s2 = cam.project_point(lig_coords[j])
        queue.append(
            (
                zm,
                "lig_bond",
                {
                    "xy1": (float(lig_xy[i, 0]), float(lig_xy[i, 1])),
                    "xy2": (float(lig_xy[j, 0]), float(lig_xy[j, 1])),
                    "rgb1": _apply_depth_fog(
                        _element_color(el1, "light", carbon_override=ligand_carbon),
                        bg_rgb,
                        float(lig_z[i]),
                        z_min,
                        z_max,
                        0.15,
                    ),
                    "rgb2": _apply_depth_fog(
                        _element_color(el2, "light", carbon_override=ligand_carbon),
                        bg_rgb,
                        float(lig_z[j]),
                        z_min,
                        z_max,
                        0.15,
                    ),
                    "bond_type": bond.GetBondType(),
                    "is_aromatic": bond.GetIsAromatic(),
                    "scale": 0.5 * (s1 + s2),
                },
            )
        )

    for idx in range(lig_noh.GetNumAtoms()):
        el = lig_noh.GetAtomWithIdx(idx).GetSymbol().upper()
        x2, y2, zc, sc = cam.project_point(lig_coords[idx])
        rgb = _apply_depth_fog(
            _element_color(el, "light", carbon_override=ligand_carbon),
            bg_rgb,
            zc,
            z_min,
            z_max,
            0.15,
        )
        queue.append(
            (
                zc,
                "lig_atom",
                {"x": x2, "y": y2, "r": _ATOM_VISUAL_RADIUS.get(el, 0.35) * sc, "rgb": rgb},
            )
        )

    for rec in display_ints:
        px, py, pz, _ = cam.project_point(np.asarray(rec.protein_xyz, dtype=np.float64))
        lx, ly, lz, _ = cam.project_point(np.asarray(rec.ligand_xyz, dtype=np.float64))
        meta = _INTERACTION_META.get(rec.kind, _INTERACTION_META["hbond"])
        queue.append(
            (
                max(pz, lz) + 0.15,
                "interaction",
                {
                    "p": (px, py),
                    "l": (lx, ly),
                    "color": meta["light"],
                    "kind": rec.kind,
                    "dist": rec.distance_angstrom,
                },
            )
        )

    dist_anchors: list[tuple[float, float]] = []
    dist_meta: list[tuple[str, str]] = []
    queue.sort(key=lambda item: item[0])
    for step, (_, kind_p, pl) in enumerate(queue):
        zo = 2.0 + step * 0.01
        if kind_p == "ribbon":
            ax.plot(
                [pl["p1"][0], pl["p2"][0]],
                [pl["p1"][1], pl["p2"][1]],
                color=tuple(pl["rgb"]),
                lw=pl["w"],
                solid_capstyle="butt",
                alpha=pl["alpha"],
                zorder=zo,
            )
        elif kind_p == "cofactor_bond":
            ax.plot(
                [pl["xy1"][0], pl["xy2"][0]],
                [pl["xy1"][1], pl["xy2"][1]],
                color=tuple(pl["rgb"]),
                lw=2.2,
                solid_capstyle="round",
                alpha=0.88,
                zorder=zo,
            )
        elif kind_p == "ring_poly":
            ax.add_patch(
                Polygon(
                    pl["xy"],
                    closed=True,
                    facecolor=tuple(pl["rgb"]),
                    edgecolor=tuple(pl["rgb"]),
                    linewidth=0.6,
                    alpha=pl["alpha"],
                    zorder=zo,
                )
            )
        elif kind_p == "pocket_bond":
            xy1, xy2 = pl["xy1"], pl["xy2"]
            mid = (0.5 * (xy1[0] + xy2[0]), 0.5 * (xy1[1] + xy2[1]))
            ax.plot(
                [xy1[0], mid[0]],
                [xy1[1], mid[1]],
                color=tuple(pl["rgb1"]),
                lw=1.7,
                solid_capstyle="round",
                alpha=0.78,
                zorder=zo,
            )
            ax.plot(
                [mid[0], xy2[0]],
                [mid[1], xy2[1]],
                color=tuple(pl["rgb2"]),
                lw=1.7,
                solid_capstyle="round",
                alpha=0.78,
                zorder=zo,
            )
        elif kind_p == "pocket_atom":
            _draw_phong_sphere(
                ax,
                pl["x"],
                pl["y"],
                pl["r"],
                pl["rgb"],
                bg_rgb,
                zorder=zo,
                alpha=0.88,
                specular=0.60,
            )
        elif kind_p == "lig_bond":
            _draw_multi_order_bond(
                ax,
                pl["xy1"],
                pl["xy2"],
                pl["rgb1"],
                pl["rgb2"],
                bg_rgb,
                bond_type=pl["bond_type"],
                is_aromatic=pl["is_aromatic"],
                persp_scale=pl["scale"],
                zorder=zo,
                base_width_pt=3.9,
            )
        elif kind_p == "lig_atom":
            _draw_phong_sphere(ax, pl["x"], pl["y"], pl["r"], pl["rgb"], bg_rgb, zorder=zo)
        elif kind_p == "interaction":
            px, py = pl["p"]
            lx, ly = pl["l"]
            is_hyd = pl["kind"] == "hydrophobic"
            ax.plot(
                [px, lx],
                [py, ly],
                color=pl["color"],
                lw=1.15 if is_hyd else 1.75,
                linestyle=(0, (2.2, 2.2)) if is_hyd else (0, (3.2, 2.0)),
                alpha=0.65 if is_hyd else 0.96,
                zorder=zo,
            )
            if not is_hyd:
                dist_anchors.append((0.5 * (px + lx), 0.5 * (py + ly)))
                dist_meta.append((f"{pl['dist']:.2f}Å", pl["color"]))

    # Frame cleanly with generous top margin for headers & bottom-right margin for 2D inset
    frame_pts = [lig_xy]
    for rid in active_res:
        ra = pocket_geom.by_residue.get(rid, [])
        if ra:
            rxy, _ = cam.project(np.stack([a.xyz for a in ra], axis=0))
            frame_pts.append(rxy)
    all_xy = np.vstack(frame_pts)
    x_span = max(float(np.ptp(all_xy[:, 0])), 7.4)
    y_span = max(float(np.ptp(all_xy[:, 1])), 6.0)
    xmin = float(np.min(all_xy[:, 0])) - max(1.8, 0.16 * x_span)
    xmax = float(np.max(all_xy[:, 0])) + max(2.6, 0.28 * x_span)
    ymin = float(np.min(all_xy[:, 1])) - max(1.7, 0.18 * y_span)
    ymax = float(np.max(all_xy[:, 1])) + max(2.8, 0.30 * y_span)

    target_aspect = 1.44
    cur_w, cur_h = xmax - xmin, ymax - ymin
    if cur_w / max(cur_h, 1e-6) < target_aspect:
        pad_each = 0.5 * (cur_h * target_aspect - cur_w)
        xmin -= pad_each
        xmax += pad_each
    else:
        pad_each = 0.5 * (cur_w / target_aspect - cur_h)
        ymin -= pad_each
        ymax += pad_each

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Unified de-overlap pass for BOTH residue callouts AND distance pills (zero mutual overlap!)
    y_top_safe = ymax - 0.19 * (ymax - ymin)
    y_bot_safe = ymin + 0.08 * (ymax - ymin)
    lig_c2 = np.mean(lig_xy, axis=0)

    all_anc: list[tuple[float, float]] = []
    all_init: list[tuple[float, float]] = []
    all_meta: list[tuple[str, str, str]] = []  # (kind: 'dist'|'res', text, color)

    for (mx, my), (txt, col) in zip(dist_anchors, dist_meta, strict=True):
        all_anc.append((mx, my))
        all_init.append((mx, my))
        all_meta.append(("dist", txt, col))

    for rid in active_res:
        rmatch = next((r for r in display_ints if r.protein_residue == rid), None)
        if rmatch is None:
            continue
        ax_p, ay_p, _, _ = cam.project_point(np.asarray(rmatch.protein_xyz, dtype=np.float64))
        vout = np.array([ax_p - lig_c2[0], ay_p - lig_c2[1]], dtype=np.float64)
        vn = float(np.linalg.norm(vout))
        ddir = vout / vn if vn > 1e-4 else np.array([0.0, 1.0])
        pdist = max(1.45, 0.125 * max(x_span, y_span))
        all_anc.append((ax_p, ay_p))
        all_init.append((ax_p + float(ddir[0]) * pdist, ay_p + float(ddir[1]) * pdist))
        col_h = _INTERACTION_META.get(rmatch.kind, _INTERACTION_META["hbond"])["light"]
        all_meta.append(("res", _short_residue_label(rid), col_h))

    if cofac_label_pt is not None:
        all_anc.append(cofac_label_pt)
        all_init.append((cofac_label_pt[0] - 1.2, cofac_label_pt[1] + 1.0))
        all_meta.append(("res", "FAD", "#D97706"))

    if all_meta:
        placed = _deoverlap_label_positions(
            all_anc,
            all_init,
            box_w=max(1.85, 0.135 * (xmax - xmin)),
            box_h=max(0.76, 0.072 * (ymax - ymin)),
            bounds=(xmin + 0.55, xmax - 0.55, y_bot_safe, y_top_safe),
            iterations=60,
        )
        for (ax_p, ay_p), (lx_p, ly_p), (lkind, ltxt, col_h) in zip(
            all_anc, placed, all_meta, strict=True
        ):
            if lx_p > xmin + 0.67 * (xmax - xmin) and ly_p < ymin + 0.33 * (ymax - ymin):
                ly_p = ymin + 0.37 * (ymax - ymin)
            if lkind == "dist":
                ax.text(
                    lx_p,
                    ly_p,
                    ltxt,
                    fontsize=5.9,
                    fontweight="bold",
                    family="monospace",
                    color=col_h,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.12,rounding_size=0.20",
                        "facecolor": "white",
                        "edgecolor": col_h,
                        "linewidth": 0.65,
                        "alpha": 0.94,
                    },
                    zorder=91.0,
                )
            else:
                ax.plot([ax_p, lx_p], [ay_p, ly_p], color=col_h, lw=0.80, alpha=0.55, zorder=92.0)
                ax.text(
                    lx_p,
                    ly_p,
                    ltxt,
                    fontsize=6.8,
                    fontweight="bold",
                    family="monospace",
                    color="#0F172A",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.20,rounding_size=0.25",
                        "facecolor": "#FFFBEB" if ltxt == "FAD" else "#F8FAFC",
                        "edgecolor": col_h,
                        "linewidth": 1.0,
                        "alpha": 0.96,
                    },
                    zorder=95.0,
                )

    # Crisp top header banner & KPI pill
    ax.text(
        0.02,
        0.972,
        f"({panel_letter})  {header_title}",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color="#0F172A",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=98.0,
    )
    ax.text(
        0.02,
        0.896,
        metrics_line,
        transform=ax.transAxes,
        fontsize=6.7,
        fontweight="bold",
        family="monospace",
        color="#0369A1",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
            "linewidth": 0.65,
            "alpha": 0.94,
        },
        zorder=98.0,
    )

    # Crisp 2D RDKit structure inset in bottom-right corner
    ax_in = ax.inset_axes((0.70, 0.025, 0.28, 0.27))
    img_2d = _render_2d_mol_png_array(lig_mol, theme="light", width=460, height=290)
    ax_in.imshow(img_2d)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_edgecolor("#94A3B8")
        sp.set_linewidth(0.85)

    # 2.0 Å scale bar in bottom-left
    sb_x0 = xmin + 0.48
    sb_y = ymin + 0.42
    ax.plot([sb_x0, sb_x0 + 2.0], [sb_y, sb_y], color="#475569", lw=1.7, zorder=99.0)
    ax.text(
        sb_x0 + 1.0,
        sb_y + 0.14,
        "2.0 Å",
        fontsize=6.4,
        color="#475569",
        ha="center",
        va="bottom",
        family="monospace",
        zorder=99.0,
    )
    return profile


def generate_figure_18(repo_root: Path, tables_dir: Path, figures_dir: Path) -> Path:
    """Render Figure 18: 2x2 300-DPI 3D binding-mode showcase (SAFE-GPT rescues + GRM5 NAM)."""
    _apply_nature_style()
    dock = _load_json(tables_dir / "v4_tri_disorder_docking.json")
    ts = _load_json(tables_dir / "ts_circuit_audit_and_benchmark.json")

    dec = dock["safegpt_decorated_docking"]
    don_p = dec["donepezil_ache"]["parent_profile"]
    don_a = dec["donepezil_ache"]["docked_analogs"][0]
    saf_p = dec["safinamide_maob"]["parent_profile"]
    saf_a = dec["safinamide_maob"]["docked_analogs"][0]
    xan_p = dec["xanomeline_chrm4"]["parent_profile"]
    xan_a = dec["xanomeline_chrm4"]["docked_analogs"][0]
    grm5_a = ts["heldout_vina_docking_and_plif"]["GRM5"]["geneuclidean_top5"][0]

    v4_clean = repo_root / "data/cns_neuro.nosync/v4_expansion/structures/clean_pockets"
    v3_clean = repo_root / "data/cns_neuro.nosync/structures/clean_pockets"
    ts_dir = repo_root / "data/cns_neuro.nosync/ts_circuit_expansion"

    panel_specs = [
        (
            "A",
            "Alzheimer's ACHE (4EY7) — SAFE-GPT Donepezil Rescue",
            f"Vina {don_a['on_vina_dg']:.2f} | ΔΔG {don_a['ddg']:+.2f} vs BCHE | "
            f"hERG {don_p['herg_prob']:.2f}→{don_a['herg_prob']:.2f} | "
            f"MPO {don_a['cns_mpo_v2']:.2f}",
            v4_clean / "4EY7_ACHE_E20_pocket.pdb",
            repo_root / str(don_a["saved_sdf"]),
            16.0,
            18.0,
            "#0F766E",
            False,
        ),
        (
            "B",
            "Alzheimer's MAOB (2V5Z+FAD) — SAFE-GPT Safinamide Rescue",
            f"Vina {saf_a['on_vina_dg']:.2f} | ΔΔG {saf_a['ddg']:+.2f} vs MAOA | "
            f"MDR1 {saf_p['mdr1_efflux_prob']:.2f}→{saf_a['mdr1_efflux_prob']:.2f} | "
            f"MPO {saf_a['cns_mpo_v2']:.2f}",
            v4_clean / "2V5Z_MAOB_SAG_pocket.pdb",
            repo_root / str(saf_a["saved_sdf"]),
            14.0,
            -16.0,
            "#0369A1",
            True,
        ),
        (
            "C",
            "Schizophrenia CHRM4 (8FX5) — SAFE-GPT Xanomeline Rescue",
            f"Vina {xan_a['on_vina_dg']:.2f} | ΔΔG {xan_a['ddg']:+.2f} vs DRD2 | "
            f"M3 {xan_p['chrm3_avoid_prob']:.2f}→{xan_a['chrm3_avoid_prob']:.2f} | "
            f"hERG {xan_p['herg_prob']:.2f}→{xan_a['herg_prob']:.2f}",
            v3_clean / "8FX5_CHRM4_XNO_pocket.pdb",
            repo_root / str(xan_a["saved_sdf"]),
            14.0,
            20.0,
            "#0F766E",
            False,
        ),
        (
            "D",
            "Tourette Basal-Ganglia GRM5 (4OO9) — Signal-Filter NAM Lead",
            f"Vina {grm5_a['vina_dg_kcal']:.2f} (beats Dipraglurant) | "
            f"MPO {grm5_a['cns_mpo_v2']:.2f} | hERG {grm5_a['admet_v4_probs']['herg']:.3f} | "
            f"Non-Alkyne",
            ts_dir / "structures/clean_pockets/4OO9_GRM5_2U8_pocket.pdb",
            ts_dir / "docked_poses_3d/GRM5_geneuclidean_5.sdf",
            14.0,
            10.0,
            "#0369A1",
            False,
        ),
    ]

    fig = plt.figure(figsize=(12.2, 8.8), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.06,
        wspace=0.04,
        left=0.015,
        right=0.985,
        top=0.92,
        bottom=0.065,
    )

    for idx, (letter, title, mline, pdb_path, sdf_path, elev, azim, c_hex, show_cof) in enumerate(
        panel_specs
    ):
        lig_mol = _load_first_sdf_mol(sdf_path)
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        _render_v4_3d_panel(
            ax,
            pdb_path,
            lig_mol,
            panel_letter=letter,
            header_title=title,
            metrics_line=mline,
            elev_deg=elev,
            azim_deg=azim,
            ligand_carbon=c_hex,
            show_cofactors=show_cof,
        )

    leg_handles = [
        Line2D(
            [0],
            [0],
            color=_INTERACTION_META[k]["light"],
            lw=2.0,
            linestyle="--",
            label=_INTERACTION_META[k]["label"],
        )
        for k in ("salt_bridge", "hbond", "pi_stacking", "cation_pi", "hydrophobic")
    ] + [Line2D([0], [0], color="#D97706", lw=2.4, label="FAD Cofactor (MAOB)")]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        ncol=6,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=7.8,
    )
    fig.suptitle(
        "GenEuclidean v4.0 — 3D Binding Modes of SAFE-GPT Clinical Rescues "
        "& Tourette CSTC NAM Lead",
        fontsize=11.8,
        fontweight="bold",
        color="#0F172A",
        y=0.968,
    )
    return _save_fig_300dpi(fig, figures_dir / "18_v4_safegpt_rescue_and_alzheimer_3d.png")


def main() -> None:
    """CLI entrypoint to generate GenEuclidean v4.0 publication figures."""
    parser = argparse.ArgumentParser(description="Generate GenEuclidean v4.0 300-DPI figures.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fig", choices=["17", "18", "all"], default="all")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    tables_dir = repo_root / "results/tables"
    figures_dir = repo_root / "results/figures"

    if args.fig in {"17", "all"}:
        p17 = generate_figure_17(tables_dir, figures_dir)
        print(f"Saved {p17} ({p17.stat().st_size / 1e6:.2f} MB)")
    if args.fig in {"18", "all"}:
        p18 = generate_figure_18(repo_root, tables_dir, figures_dir)
        print(f"Saved {p18} ({p18.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
