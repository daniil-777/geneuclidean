"""FDA Jan 2025 / Joint FDA-EMA Jan 2026 AI Credibility Dossier & Split-Conformal UQ.

Implements:

1. **``SplitConformalBinaryCalibrator``:** Distribution-free, finite-sample Split-Conformal
   prediction sets (Vovk / Angelopoulos-Bates) wrapping any calibrated binary probability
   oracle ``p_cal in [0, 1]`` with exact ``1 - alpha`` (default ``90%``, ``alpha = 0.10``)
   finite-sample coverage guarantee under exchangeability, plus optional class-conditional
   (Mondrian) conformal calibration for imbalanced ADMET endpoints.
2. **``build_credibility_dossier`` & ``render_credibility_markdown``:** Automated builder
   emitting a structured **FDA Jan 2025 (``FDA-2024-D-4689``) 7-Step Risk-Based AI
   Credibility Dossier** aligned with the **Joint FDA-EMA 10 Guiding Principles of Good AI
   Practice (Jan 2026)** and the **FDA NAMs Roadmap (2025-2026)** — including explicit
   Context of Use (COU), SHA-256 data/model provenance, 5-fold scaffold-CV calibration
   (ROC-AUC, PR-AUC, Brier, ECE), Split-Conformal UQ coverage, ECFP4 Applicability Domain
   summary, held-out physics Gate 1 metrics, and explicit *in silico* / human-in-the-loop
   safety boundaries.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

CREDIBILITY_SCHEMA_VERSION: str = "geneuclidean-v4.0-fda-ema-credibility-dossier"
DEFAULT_CONTEXT_OF_USE: str = (
    "Exploratory preclinical in silico hypothesis generation, 3D pocket-conditioned lead "
    "prioritization, and CNS ADMET/Avoid-ome triage (NON-REGULATORY). All candidate "
    "molecules and oracle predictions are research hypotheses requiring chemical synthesis, "
    "functional pharmacology, SafetyScreen44/CiPA profiling, and GLP preclinical validation "
    "before clinical translation."
)
HUMAN_IN_THE_LOOP_DISCLAIMER: str = (
    "MANDATORY HUMAN-IN-THE-LOOP & NON-CLINICAL BOUNDARY: GenEuclidean outputs are strictly "
    "in silico computational hypotheses and are NOT clinically validated medicines, regulatory "
    "filings, or medical advice. Every shortlisted series must undergo medicinal-chemistry "
    "review, formal Markush/CAS MARPAT patent-attorney FTO clearance, retrosynthetic route "
    "verification, and experimental wet-lab potency, selectivity, DMPK, and toxicology testing."
)


# ---------------------------------------------------------------------------
# Distribution-Free Finite-Sample Split-Conformal Binary Classification UQ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformalPredictionResult:
    """Split-Conformal prediction set and uncertainty indicators for one binary prediction."""

    prob: float
    prediction_set: tuple[int, ...]
    uncertain_flag: bool
    coverage_bound: float
    threshold_q: float
    singleton_prediction: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize conformal prediction result to a JSON-compatible dictionary."""
        return {
            "prob": self.prob,
            "set": list(self.prediction_set),
            "prediction_set": list(self.prediction_set),
            "uncertain_flag": self.uncertain_flag,
            "coverage_bound": self.coverage_bound,
            "threshold_q": self.threshold_q,
            "singleton_prediction": self.singleton_prediction,
        }


def _finite_sample_conformal_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Compute exact finite-sample upper quantile ``ceil((n + 1)(1 - alpha)) / n``."""
    n = len(scores)
    if n == 0:
        return 1.0
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank >= n:
        return float(np.max(scores))
    if rank <= 1:
        return float(np.min(scores))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1])


class SplitConformalBinaryCalibrator:
    """Finite-sample Split-Conformal wrapper around any calibrated binary probability oracle.

    Uses the standard probability nonconformity score ``s(p, y) = 1 - P(Y = y | X)``
    (so ``s(p, 1) = 1 - p`` and ``s(p, 0) = p``). Given ``n`` exchangeable calibration
    samples, taking the ``ceil((n + 1)(1 - alpha))``-th order statistic ``q_hat`` guarantees:

        ``P(Y_test in C(X_test)) >= 1 - alpha``

    without any parametric distributional assumptions. Supports optional class-conditional
    (Mondrian) calibration (``mondrian=True``) for imbalanced safety endpoints.
    """

    def __init__(self, alpha: float = 0.10, *, mondrian: bool = False) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
        self.alpha: float = float(alpha)
        self.mondrian: bool = bool(mondrian)
        self.q_hat: float = 1.0
        self.q_hat_neg: float = 1.0
        self.q_hat_pos: float = 1.0
        self.n_calibration: int = 0
        self.is_fitted: bool = False

    @property
    def coverage_bound(self) -> float:
        """Nominal finite-sample lower coverage guarantee ``1 - alpha``."""
        return float(1.0 - self.alpha)

    def fit(
        self,
        probs: ArrayLike,
        y_true: ArrayLike,
    ) -> SplitConformalBinaryCalibrator:
        """Fit finite-sample nonconformity quantile(s) on calibrated probabilities."""
        p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
        y = np.asarray(y_true, dtype=np.int64)
        mask = np.isfinite(p) & np.isin(y, [0, 1])
        p = p[mask]
        y = y[mask]
        if len(y) == 0:
            raise ValueError("SplitConformalBinaryCalibrator.fit requires >= 1 valid sample")

        # True-class nonconformity scores: s_i = 1 - p_i if y_i == 1 else p_i
        scores = np.where(y == 1, 1.0 - p, p)
        self.q_hat = _finite_sample_conformal_quantile(scores, self.alpha)
        if self.mondrian:
            neg_scores = scores[y == 0]
            pos_scores = scores[y == 1]
            self.q_hat_neg = (
                _finite_sample_conformal_quantile(neg_scores, self.alpha)
                if len(neg_scores) > 0
                else self.q_hat
            )
            self.q_hat_pos = (
                _finite_sample_conformal_quantile(pos_scores, self.alpha)
                if len(pos_scores) > 0
                else self.q_hat
            )
        else:
            self.q_hat_neg = self.q_hat
            self.q_hat_pos = self.q_hat

        self.n_calibration = int(len(y))
        self.is_fitted = True
        return self

    def predict_one(self, prob: float) -> ConformalPredictionResult:
        """Return conformal set ``{set, uncertain_flag, coverage_bound}`` for ``prob``."""
        if not self.is_fitted:
            raise RuntimeError("SplitConformalBinaryCalibrator must be fitted before prediction.")
        if not math.isfinite(float(prob)):
            return ConformalPredictionResult(
                prob=float("nan"),
                prediction_set=(0, 1),
                uncertain_flag=True,
                coverage_bound=self.coverage_bound,
                threshold_q=self.q_hat,
                singleton_prediction=None,
            )
        p = float(np.clip(prob, 0.0, 1.0))
        q0 = self.q_hat_neg if self.mondrian else self.q_hat
        q1 = self.q_hat_pos if self.mondrian else self.q_hat
        members: list[int] = []
        # s(p, 0) = p <= q0 ; s(p, 1) = 1 - p <= q1 (with tiny float epsilon)
        if p <= q0 + 1e-12:
            members.append(0)
        if (1.0 - p) <= q1 + 1e-12:
            members.append(1)
        pred_set = tuple(members)
        singleton = pred_set[0] if len(pred_set) == 1 else None
        uncertain = len(pred_set) != 1
        return ConformalPredictionResult(
            prob=p,
            prediction_set=pred_set,
            uncertain_flag=uncertain,
            coverage_bound=self.coverage_bound,
            threshold_q=self.q_hat,
            singleton_prediction=singleton,
        )

    def predict_batch(self, probs: ArrayLike) -> list[ConformalPredictionResult]:
        """Compute conformal prediction sets across a batch of calibrated probabilities."""
        arr = np.asarray(probs, dtype=np.float64)
        return [self.predict_one(float(val)) for val in arr.tolist()]

    def evaluate_coverage(self, probs: ArrayLike, y_true: ArrayLike) -> dict[str, float]:
        """Evaluate empirical coverage, singleton rate, ambiguous rate, and mean set size."""
        p = np.asarray(probs, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.int64)
        mask = np.isfinite(p) & np.isin(y, [0, 1])
        p = p[mask]
        y = y[mask]
        n = len(y)
        if n == 0:
            return {
                "n_samples": 0.0,
                "alpha": self.alpha,
                "target_coverage": self.coverage_bound,
                "empirical_coverage": float("nan"),
                "singleton_rate": float("nan"),
                "ambiguous_rate": float("nan"),
                "empty_rate": float("nan"),
                "avg_set_size": float("nan"),
                "q_hat": self.q_hat,
            }

        results = self.predict_batch(p)
        covered = sum(
            1 for res, lbl in zip(results, y.tolist(), strict=True) if lbl in res.prediction_set
        )
        singletons = sum(1 for res in results if len(res.prediction_set) == 1)
        ambiguous = sum(1 for res in results if len(res.prediction_set) == 2)
        empty = sum(1 for res in results if len(res.prediction_set) == 0)
        mean_size = sum(len(res.prediction_set) for res in results) / n

        return {
            "n_samples": float(n),
            "alpha": self.alpha,
            "target_coverage": self.coverage_bound,
            "empirical_coverage": float(covered / n),
            "singleton_rate": float(singletons / n),
            "ambiguous_rate": float(ambiguous / n),
            "empty_rate": float(empty / n),
            "avg_set_size": float(mean_size),
            "q_hat": float(self.q_hat),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibrator state to a JSON-compatible dictionary."""
        return {
            "alpha": self.alpha,
            "coverage_bound": self.coverage_bound,
            "mondrian": self.mondrian,
            "q_hat": self.q_hat,
            "q_hat_neg": self.q_hat_neg,
            "q_hat_pos": self.q_hat_pos,
            "n_calibration": self.n_calibration,
            "is_fitted": self.is_fitted,
        }


# ---------------------------------------------------------------------------
# Provenance Checksums & FDA Jan 2025 / Joint FDA-EMA Jan 2026 Credibility Dossier
# ---------------------------------------------------------------------------


def compute_file_sha256(path: str | Path) -> str | None:
    """Compute SHA-256 hex digest of a file, returning ``None`` if absent (CI-safe)."""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_provenance_hashes(paths: Mapping[str, str | Path] | None) -> dict[str, dict[str, Any]]:
    """Build SHA-256 + byte-size provenance ledger for a mapping of named artifacts."""
    ledger: dict[str, dict[str, Any]] = {}
    if not paths:
        return ledger
    for label, raw_path in sorted(paths.items()):
        p = Path(raw_path)
        sha = compute_file_sha256(p)
        ledger[str(label)] = {
            "path": str(p),
            "exists": sha is not None,
            "size_bytes": int(p.stat().st_size) if p.exists() and p.is_file() else None,
            "sha256": sha,
        }
    return ledger


def build_credibility_dossier(
    *,
    model_name: str = "GenEuclidean v4.0 CNS Discovery & ADMET/Avoid-ome Suite",
    model_version: str = "4.0.0",
    question_of_interest: str = (
        "Can 3D SE(3)-equivariant pocket-conditioned generation paired with calibrated "
        "scaffold-CV ADMET/Avoid-ome oracles, Split-Conformal UQ, and industrial MedChem "
        "triage prioritize novel, synthesizable, CNS-penetrant lead hypotheses?"
    ),
    context_of_use: str = DEFAULT_CONTEXT_OF_USE,
    decision_consequence: str = (
        "Low (exploratory preclinical triage; no direct clinical dosing decision)"
    ),
    model_influence: str = (
        "Moderate (prioritizes molecules for medicinal-chemistry review & wet-lab synthesis)"
    ),
    risk_tier: str = "Low-to-Moderate (Preclinical Research Decision Support — Non-Regulatory)",
    admet_cv_metrics: Mapping[str, Any] | None = None,
    conformal_uq_metrics: Mapping[str, Any] | None = None,
    applicability_domain_summary: Mapping[str, Any] | None = None,
    physics_gate1_metrics: Mapping[str, Any] | None = None,
    provenance_files: Mapping[str, str | Path] | None = None,
    deviations_and_limitations: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build an FDA Jan 2025 (``FDA-2024-D-4689``) / Joint FDA-EMA Jan 2026 Credibility Dossier.

    Returns a clean JSON-serializable dictionary covering all 7 risk-based credibility
    steps, the 10 Joint FDA-EMA Guiding Principles, SHA-256 artifact provenance, scaffold-CV
    calibration (ECE/Brier), Split-Conformal coverage, ECFP4 Applicability Domain metrics,
    held-out physics Gate 1 RMSD/PoseBusters checks, and explicit human-in-the-loop scope.
    """
    timestamp = generated_at_utc or datetime.datetime.now(datetime.UTC).isoformat(
        timespec="seconds"
    )
    provenance = _collect_provenance_hashes(provenance_files)

    # Extract concise per-endpoint calibration & conformal summaries if provided
    endpoints_summary: dict[str, Any] = {}
    conformal_summary: dict[str, Any] = dict(conformal_uq_metrics or {})
    if admet_cv_metrics:
        raw_eps = (
            admet_cv_metrics.get("endpoints", admet_cv_metrics)
            if isinstance(admet_cv_metrics, Mapping)
            else {}
        )
        if isinstance(raw_eps, Mapping):
            for ep_name, ep_data in raw_eps.items():
                if not isinstance(ep_data, Mapping):
                    continue
                endpoints_summary[str(ep_name)] = {
                    "n_samples": ep_data.get("n_samples"),
                    "n_unique_scaffolds": ep_data.get("n_unique_scaffolds"),
                    "scaffold_leakage_groups": ep_data.get("scaffold_leakage_groups", 0),
                    "oof_roc_auc": ep_data.get("oof_roc_auc"),
                    "oof_pr_auc": ep_data.get("oof_pr_auc"),
                    "oof_brier_score": ep_data.get("oof_brier_score"),
                    "oof_ece": ep_data.get("oof_ece"),
                    "calibration_method": ep_data.get("calibration_method"),
                }
                if "conformal_uq" in ep_data and str(ep_name) not in conformal_summary:
                    conformal_summary[str(ep_name)] = dict(ep_data["conformal_uq"])

    default_limitations = [
        (
            "Strictly exploratory in silico research tool; NOT qualified for regulatory "
            "IND safety waivers."
        ),
        (
            "TDC Pgp_Broccatelli measures P-gp inhibition rather than transcellular substrate "
            "efflux; v4.0 adds Biogen MDR1-MDCK Efflux Ratio (ER >= 2.0) as primary efflux oracle."
        ),
        "TDC DILI has modest dataset size (N=475) and serves as an advisory hepatotoxicity screen.",
        (
            "Automated Bemis-Murcko / ECFP4 scaffold-hop checks do NOT adjudicate "
            "Markush patent claims."
        ),
        (
            "Equilibrium docking scores (AutoDock Vina) do not measure binding kinetics "
            "(k_off / residence time) or biased GPCR signaling; Tier-2/3 co-folding, "
            "FEP/OpenFE, and wet-lab BRET/SPR are required."
        ),
    ]
    all_limitations = list(default_limitations)
    if deviations_and_limitations:
        all_limitations.extend(str(item) for item in deviations_and_limitations)

    ad_summary: dict[str, Any] = {
        "metric": "Nearest-neighbor 1024-bit ECFP4 Tanimoto similarity to training set",
        "in_domain_threshold": 0.30,
        "ood_policy": (
            "Out-of-domain predictions (AD Tanimoto < 0.30) are explicitly flagged and linearly "
            "damped toward a neutral 0.50 prior during reward optimization (ad_damp=True)."
        ),
        **(dict(applicability_domain_summary) if applicability_domain_summary else {}),
    }

    physics_summary: dict[str, Any] = (
        dict(physics_gate1_metrics)
        if physics_gate1_metrics
        else {
            "status": "not_supplied_in_this_dossier_snapshot",
            "protocol": (
                "Held-out AutoDock Vina redocking RMSD + PoseBusters + 3D hotspot anchor audit"
            ),
        }
    )

    dossier: dict[str, Any] = {
        "schema_version": CREDIBILITY_SCHEMA_VERSION,
        "generated_at_utc": timestamp,
        "model_identity": {
            "name": model_name,
            "version": model_version,
            "runtime_python": platform.python_version(),
            "runtime_platform": platform.platform(),
        },
        "regulatory_alignment": [
            (
                "FDA Draft Guidance: Considerations for the Use of AI to Support Regulatory "
                "Decision-Making (Docket FDA-2024-D-4689, Jan 2025)"
            ),
            (
                "Joint FDA-EMA 10 Guiding Principles of Good AI Practice in Drug Development "
                "(Jan 2026)"
            ),
            "FDA Modernization Acts 2.0/3.0 & NAMs Preclinical Safety Roadmap (2025-2026)",
        ],
        "step_1_question_of_interest": question_of_interest,
        "step_2_context_of_use": {
            "statement": context_of_use,
            "regulatory_submission_status": (
                "Non-regulatory exploratory preclinical hypothesis generation"
            ),
            "orthogonal_experimental_evidence_required": True,
            "human_in_the_loop_required": True,
        },
        "step_3_model_risk_assessment": {
            "decision_consequence": decision_consequence,
            "model_influence": model_influence,
            "overall_risk_tier": risk_tier,
            "primary_risk_mitigations": [
                (
                    "Hard MedChem structural triage & reagent quarantine "
                    "(PAINS/BRENK/NIH/Lilly/DCU=0.0)"
                ),
                "Multiplicative safety-gated reward eliminating compensatory ADMET-sink bleed",
                (
                    "Zero-leakage 5-fold Bemis-Murcko scaffold CV with Platt calibration "
                    "& Split-Conformal 90% UQ"
                ),
                "ECFP4 Tanimoto Applicability Domain enforcement & OOD probability damping",
                (
                    "Mandatory wet-lab synthesis, functional assay, SafetyScreen44, "
                    "and GLP validation gate"
                ),
            ],
        },
        "step_4_credibility_assessment_plan": {
            "data_splitting": (
                "5-fold Bemis-Murcko scaffold-disjoint cross-validation "
                "(strictly 0 overlapping scaffolds)"
            ),
            "probability_calibration": (
                "Nested scaffold-disjoint Platt sigmoid calibration (target ECE <= 0.05)"
            ),
            "uncertainty_quantification": (
                "Distribution-free finite-sample Split-Conformal prediction sets "
                "(alpha=0.10, >=90% coverage)"
            ),
            "applicability_domain": ("Bit-packed 1024-bit ECFP4 nearest-neighbor Tanimoto >= 0.30"),
            "structural_hygiene": (
                "RDKit PAINS + BRENK + NIH + Lilly-inspired demerits <= 100 + "
                "coupling-reagent quarantine"
            ),
            "physics_verification": (
                "Held-out AutoDock Vina redocking RMSD (<= 2.0 A), PoseBusters validity, "
                "and 3D anchor contacts"
            ),
        },
        "step_5_provenance_and_execution": {
            "artifact_sha256_ledger": provenance,
            "n_tracked_artifacts": len(provenance),
            "deterministic_seeding": True,
        },
        "step_6_validation_results_and_deviations": {
            "admet_scaffold_cv": endpoints_summary,
            "conformal_uq": conformal_summary,
            "applicability_domain": ad_summary,
            "physics_gate1": physics_summary,
            "known_limitations_and_deviations": all_limitations,
        },
        "step_7_lifecycle_maintenance": {
            "drift_monitoring_triggers": {
                "min_in_domain_rate": 0.70,
                "max_calibration_ece": 0.05,
                "min_conformal_empirical_coverage": 0.85,
            },
            "retraining_and_change_control": (
                "Re-verify 0-scaffold-leakage CV, Platt ECE, Split-Conformal coverage, and "
                "offline unit tests before promoting any updated oracle artifact or target pocket."
            ),
        },
        "fda_ema_10_principles_checklist": {
            "1_human_centric_by_design": True,
            "2_risk_based_approach": True,
            "3_adherence_to_standards": True,
            "4_clear_context_of_use": True,
            "5_multidisciplinary_expertise": True,
            "6_data_governance_and_provenance": True,
            "7_model_design_and_reproducibility": True,
            "8_risk_based_performance_uq_and_ad": True,
            "9_lifecycle_management": True,
            "10_clear_essential_information": True,
        },
        "human_in_the_loop_boundary": HUMAN_IN_THE_LOOP_DISCLAIMER,
    }
    return dossier


def render_credibility_markdown(dossier: Mapping[str, Any]) -> str:
    """Render a structured FDA/EMA Credibility Dossier dict as clean GitHub-flavored Markdown."""
    ident = dossier.get("model_identity", {})
    cou = dossier.get("step_2_context_of_use", {})
    risk = dossier.get("step_3_model_risk_assessment", {})
    val = dossier.get("step_6_validation_results_and_deviations", {})
    endpoints = val.get("admet_scaffold_cv", {})
    conformal = val.get("conformal_uq", {})

    lines: list[str] = [
        f"# AI Credibility Dossier — {ident.get('name', 'GenEuclidean')}",
        "",
        f"- **Schema:** `{dossier.get('schema_version', CREDIBILITY_SCHEMA_VERSION)}`",
        (
            f"- **Version:** `{ident.get('version', '4.0.0')}` | "
            f"**Generated (UTC):** `{dossier.get('generated_at_utc', '')}`"
        ),
        f"- **Risk Tier:** {risk.get('overall_risk_tier', 'N/A')}",
        "",
        "> **Human-in-the-Loop & Non-Clinical Boundary:** "
        + str(dossier.get("human_in_the_loop_boundary", HUMAN_IN_THE_LOOP_DISCLAIMER)),
        "",
        "## Step 1–2: Question of Interest & Context of Use (COU)",
        f"- **Question of Interest:** {dossier.get('step_1_question_of_interest', '')}",
        f"- **Context of Use:** {cou.get('statement', '')}",
        "",
    ]

    if endpoints:
        header = (
            "| Endpoint | N | Scaffolds | Leakage | ROC-AUC | PR-AUC | "
            "Brier | ECE | Conformal Cov (90%) | Avg Set Size |"
        )
        lines.extend(
            [
                "## Step 6: 5-Fold Bemis-Murcko Scaffold-CV & Split-Conformal UQ Summary",
                "",
                header,
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for ep, stats in endpoints.items():
            uq = conformal.get(ep, {})
            cov = uq.get("empirical_coverage")
            sz = uq.get("avg_set_size")
            cov_str = f"{cov:.3f}" if isinstance(cov, (int, float)) and math.isfinite(cov) else "—"
            sz_str = f"{sz:.2f}" if isinstance(sz, (int, float)) and math.isfinite(sz) else "—"
            roc = stats.get("oof_roc_auc")
            pr = stats.get("oof_pr_auc")
            br = stats.get("oof_brier_score")
            ece = stats.get("oof_ece")
            n_val = stats.get("n_samples", "—")
            n_scaf = stats.get("n_unique_scaffolds", "—")
            lines.append(
                f"| `{ep}` | {n_val} | {n_scaf} | "
                f"{stats.get('scaffold_leakage_groups', 0)} | "
                f"{f'{roc:.4f}' if isinstance(roc, (int, float)) else '—'} | "
                f"{f'{pr:.4f}' if isinstance(pr, (int, float)) else '—'} | "
                f"{f'{br:.4f}' if isinstance(br, (int, float)) else '—'} | "
                f"{f'{ece:.4f}' if isinstance(ece, (int, float)) else '—'} | "
                f"{cov_str} | {sz_str} |"
            )
        lines.append("")

    limitations = val.get("known_limitations_and_deviations", [])
    if limitations:
        lines.append("## Known Limitations & Protocol Boundaries")
        for item in limitations:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def export_credibility_dossier(
    dossier: Mapping[str, Any],
    json_path: str | Path,
    *,
    md_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Write a Credibility Dossier to JSON (and optionally Markdown)."""
    j_out = Path(json_path)
    j_out.parent.mkdir(parents=True, exist_ok=True)
    j_out.write_text(json.dumps(dossier, indent=2) + "\n", encoding="utf-8")

    m_out: Path | None = None
    if md_path is not None:
        m_out = Path(md_path)
        m_out.parent.mkdir(parents=True, exist_ok=True)
        m_out.write_text(render_credibility_markdown(dossier), encoding="utf-8")
    return j_out, m_out


__all__ = [
    "CREDIBILITY_SCHEMA_VERSION",
    "DEFAULT_CONTEXT_OF_USE",
    "HUMAN_IN_THE_LOOP_DISCLAIMER",
    "ConformalPredictionResult",
    "SplitConformalBinaryCalibrator",
    "build_credibility_dossier",
    "compute_file_sha256",
    "export_credibility_dossier",
    "render_credibility_markdown",
]
