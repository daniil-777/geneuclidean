"""FDA Jan 2025 / Joint FDA-EMA Jan 2026 regulatory AI credibility dossier SDK."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CredibilityDossier:
    """Structured FDA Jan 2025 (``FDA-2024-D-4689``) / EMA Jan 2026 AI Credibility Dossier."""

    payload: dict[str, Any]

    @property
    def schema_version(self) -> str:
        """Return credibility dossier schema identifier."""
        return str(self.payload.get("schema_version", ""))

    @property
    def context_of_use(self) -> str:
        """Return Context of Use (COU) statement."""
        cou = self.payload.get("step_2_context_of_use", {})
        return str(cou.get("statement", "")) if isinstance(cou, Mapping) else str(cou)

    @property
    def disclaimer(self) -> str:
        """Return explicit *in silico* human-in-the-loop disclaimer."""
        return str(self.payload.get("human_in_the_loop_boundary", ""))

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the JSON-serializable dossier payload."""
        return dict(self.payload)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the dossier to formatted JSON text."""
        return json.dumps(self.payload, indent=indent) + "\n"

    def to_markdown(self) -> str:
        """Render the dossier to GitHub-flavored Markdown."""
        from geneuclidean.cns.credibility import render_credibility_markdown

        return render_credibility_markdown(self.payload)

    def save(
        self,
        json_path: str | Path,
        *,
        md_path: str | Path | None = None,
    ) -> tuple[Path, Path | None]:
        """Persist dossier to JSON (and optionally Markdown)."""
        from geneuclidean.cns.credibility import export_credibility_dossier

        return export_credibility_dossier(self.payload, json_path=json_path, md_path=md_path)


def build_credibility_dossier(
    *,
    model_name: str = "GenEuclidean CNS Discovery & ADMET Suite",
    model_version: str = "4.0.0",
    question_of_interest: str | None = None,
    context_of_use: str | None = None,
    decision_consequence: str | None = None,
    model_influence: str | None = None,
    risk_tier: str | None = None,
    admet_cv_metrics: Mapping[str, Any] | None = None,
    conformal_uq_metrics: Mapping[str, Any] | None = None,
    applicability_domain_summary: Mapping[str, Any] | None = None,
    physics_gate1_metrics: Mapping[str, Any] | None = None,
    provenance_files: Mapping[str, str | Path] | None = None,
    deviations_and_limitations: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> CredibilityDossier:
    """Build an FDA Jan 2025 (``FDA-2024-D-4689``) / EMA Jan 2026 7-Step Credibility Dossier."""
    from geneuclidean.cns.credibility import build_credibility_dossier as _build

    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "admet_cv_metrics": admet_cv_metrics,
        "conformal_uq_metrics": conformal_uq_metrics,
        "applicability_domain_summary": applicability_domain_summary,
        "physics_gate1_metrics": physics_gate1_metrics,
        "provenance_files": provenance_files,
        "deviations_and_limitations": deviations_and_limitations,
        "generated_at_utc": generated_at_utc,
    }
    if question_of_interest is not None:
        kwargs["question_of_interest"] = question_of_interest
    if context_of_use is not None:
        kwargs["context_of_use"] = context_of_use
    if decision_consequence is not None:
        kwargs["decision_consequence"] = decision_consequence
    if model_influence is not None:
        kwargs["model_influence"] = model_influence
    if risk_tier is not None:
        kwargs["risk_tier"] = risk_tier

    return CredibilityDossier(payload=_build(**kwargs))


def export_credibility_json(
    dossier: CredibilityDossier | Mapping[str, Any],
    out_path: str | Path,
) -> Path:
    """Write a :class:`CredibilityDossier` (or raw dossier dict) to a JSON file."""
    payload = dossier.to_dict() if isinstance(dossier, CredibilityDossier) else dict(dossier)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def export_credibility_markdown(
    dossier: CredibilityDossier | Mapping[str, Any],
    out_path: str | Path,
) -> Path:
    """Render and write a :class:`CredibilityDossier` (or raw dict) to a Markdown file."""
    from geneuclidean.cns.credibility import render_credibility_markdown

    md_text = (
        dossier.to_markdown()
        if isinstance(dossier, CredibilityDossier)
        else render_credibility_markdown(dossier)
    )
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md_text, encoding="utf-8")
    return dest


__all__ = [
    "CredibilityDossier",
    "build_credibility_dossier",
    "export_credibility_json",
    "export_credibility_markdown",
]
