"""Unified molecular scoring, physiological ionization, MedChem triage, and composable filters.

Design goals:
    1. **Exact numerical parity:** ``score_molecules(..., admet_profile="v3")`` preserves
       bit-exact Pfizer CNS MPO v1/v2, pH 7.4 ionization, MedChem triage, and hygiene-gated
       composite reward values from ``v3.1`` / ``v4.0``.
    2. **v4.0 Avoid-ome opt-in:** Passing ``admet_profile="v4"`` seamlessly activates the
       12-endpoint scaffold-CV Industrial Avoid-ome/DMPK oracle and Multiplicative Safety-Gated
       Reward (with zero breaking changes to existing defaults).
    3. **First-class extensibility:** Callers can plug in custom :class:`MoleculeScorerProtocol`
       scorers (or wrap any callable with :class:`FunctionScorer`) and chain reusable
       :class:`MoleculeFilterProtocol` predicates (:class:`TriageFilter`, :class:`MPOFilter`,
       :class:`ADMETFilter`, :class:`BasicAmineFilter`, :class:`ScaffoldHopFilter`,
       :class:`QEDFilter`, :class:`PredicateFilter`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, overload

from geneuclidean.exceptions import InvalidMoleculeError
from geneuclidean.protocols import MoleculeFilterProtocol, MoleculeScorerProtocol
from geneuclidean.registry import FILTER_REGISTRY
from geneuclidean.sdk.registries import get_target

if TYPE_CHECKING:
    from rdkit import Chem

    from geneuclidean.cns.medchem_triage import MedChemTriageResult
    from geneuclidean.cns.tourette_circuit import TouretteCircuitScore


# ---------------------------------------------------------------------------
# Physiological pH 7.4 Protonation & Industrial MedChem Triage Facades
# ---------------------------------------------------------------------------


@overload
def protonate_ph74(
    mol_or_smiles: str,
    *,
    preserve_coords: bool = False,
    isomeric: bool = True,
) -> str: ...


@overload
def protonate_ph74(
    mol_or_smiles: Chem.Mol,
    *,
    preserve_coords: bool = False,
    isomeric: bool = True,
) -> Chem.Mol: ...


def protonate_ph74(
    mol_or_smiles: str | Chem.Mol,
    *,
    preserve_coords: bool = False,
    isomeric: bool = True,
) -> str | Chem.Mol:
    """Deterministically protonate a SMILES string or RDKit ``Mol`` at physiological pH 7.4.

    Args:
        mol_or_smiles: SMILES string (returns protonated canonical SMILES) or RDKit ``Mol``
            (returns protonated sanitized ``Chem.Mol``).
        preserve_coords: Preserve 3D coordinates when ``mol_or_smiles`` is a 3D ``Chem.Mol``.
        isomeric: Preserve stereochemical markers when returning SMILES.

    Returns:
        Protonated canonical SMILES string or RDKit ``Mol`` matching input type.

    Raises:
        InvalidMoleculeError: Subclass of ``ValueError`` when a SMILES string cannot be parsed.
    """
    from geneuclidean.evaluation.protonation import protonate_mol_ph74, protonate_smiles_ph74

    if isinstance(mol_or_smiles, str):
        try:
            return protonate_smiles_ph74(mol_or_smiles, isomeric=isomeric)
        except ValueError as exc:
            raise InvalidMoleculeError(str(exc)) from exc
    return protonate_mol_ph74(mol_or_smiles, preserve_coords=preserve_coords)


def triage_molecule(mol_or_smiles: str | Chem.Mol) -> MedChemTriageResult:
    """Run 2026 industrial MedChem structural triage, reagent quarantine & Lilly demerits.

    Screens against RDKit ``PAINS``, ``BRENK``, and ``NIH`` catalogs, explicitly quarantines
    synthesis coupling reagents/byproducts (DCU, DIU, HOBt, HOAt, DMAP, Ph3P=O, symmetric
    dialkyl ureas, reactive warheads), computes Lilly-inspired demerits and Bemis-Murcko
    scaffolds, and outputs a smooth ``hygiene_gate`` in ``[0.0, 1.0]``.
    """
    from geneuclidean.cns.medchem_triage import triage_molecule as _triage_one

    return _triage_one(mol_or_smiles)


def triage_molecules(smiles_list: Sequence[str | Chem.Mol]) -> list[MedChemTriageResult]:
    """Run :func:`triage_molecule` across a batch of SMILES strings or RDKit ``Mol`` objects."""
    from geneuclidean.cns.medchem_triage import triage_batch

    return triage_batch(smiles_list)


# ---------------------------------------------------------------------------
# Unified Molecular Report & Scoring Engine
# ---------------------------------------------------------------------------


def _admet_pass(admet: Mapping[str, Any], *, missing: bool) -> bool:
    for key in ("admet_v4_pass", "admet_pass", "v4_pharma_safe"):
        if key in admet:
            return bool(admet[key])
    return missing


@dataclass(frozen=True)
class MoleculeReport:
    """Comprehensive physicochemical, CNS MPO, MedChem triage, pH 7.4 & ADMET scorecard."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    target_id: str | None = None
    # Physiological pH 7.4 ionization
    smiles_ph74: str | None = None
    net_charge_ph74: int | None = None
    has_basic_amine: bool = False
    # Physicochemical properties & Pfizer CNS MPO v1/v2
    mw: float = float("nan")
    clogp: float = float("nan")
    clogd_74: float = float("nan")
    tpsa: float = float("nan")
    hbd: int = 0
    hba: int = 0
    most_basic_pka: float = float("nan")
    cns_mpo_v1: float = 0.0
    cns_mpo_v2: float = 0.0
    bbb_clark_pass: bool = False
    qed: float = 0.0
    sa_score: float = 10.0
    sa_norm: float = 0.0
    lipinski_violations: int = 4
    lipinski_pass: bool = False
    # Industrial MedChem Triage
    pains_free: bool = False
    brenk_free: bool = False
    nih_free: bool = True
    reagent_quarantine_free: bool = True
    lilly_demerit_score: int = 0
    triage_passed: bool = False
    hygiene_gate: float = 1.0
    triage_alerts: tuple[str, ...] = ()
    murcko_scaffold: str = ""
    generic_scaffold: str = ""
    # Optional calibrated CNS ADMET oracle predictions, ML heads, custom scorers & reward
    admet: dict[str, Any] | None = None
    pred_pkd: float | None = None
    clip_sim: float | None = None
    composite_reward: float = 0.0
    custom_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce deep immutability so custom_scores cannot be mutated after construction.
        if not isinstance(self.custom_scores, MappingProxyType):
            object.__setattr__(
                self,
                "custom_scores",
                MappingProxyType({str(k): float(v) for k, v in self.custom_scores.items()}),
            )

    @property
    def is_cns_lead(self) -> bool:
        """Return ``True`` if valid, triage-passing, CNS MPO v2 >= 4.0, and ADMET-passing."""
        if not self.valid or not self.triage_passed or not self.cns_mpo_v2 >= 4.0:
            return False
        if self.admet is not None:
            return _admet_pass(self.admet, missing=True)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a flat JSON-compatible dictionary."""
        out: dict[str, Any] = {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "target_id": self.target_id,
            "smiles_ph74": self.smiles_ph74,
            "net_charge_ph74": self.net_charge_ph74,
            "has_basic_amine": self.has_basic_amine,
            "mw": round(self.mw, 2) if math.isfinite(self.mw) else None,
            "clogp": round(self.clogp, 3) if math.isfinite(self.clogp) else None,
            "clogd_74": round(self.clogd_74, 3) if math.isfinite(self.clogd_74) else None,
            "tpsa": round(self.tpsa, 2) if math.isfinite(self.tpsa) else None,
            "hbd": self.hbd,
            "hba": self.hba,
            "most_basic_pka": (
                round(self.most_basic_pka, 2) if math.isfinite(self.most_basic_pka) else None
            ),
            "cns_mpo_v1": round(self.cns_mpo_v1, 3),
            "cns_mpo_v2": round(self.cns_mpo_v2, 3),
            "bbb_clark_pass": self.bbb_clark_pass,
            "qed": round(self.qed, 3),
            "sa_score": round(self.sa_score, 3),
            "sa_norm": round(self.sa_norm, 3),
            "lipinski_violations": self.lipinski_violations,
            "lipinski_pass": self.lipinski_pass,
            "pains_free": self.pains_free,
            "brenk_free": self.brenk_free,
            "nih_free": self.nih_free,
            "reagent_quarantine_free": self.reagent_quarantine_free,
            "lilly_demerit_score": self.lilly_demerit_score,
            "triage_passed": self.triage_passed,
            "hygiene_gate": round(self.hygiene_gate, 4),
            "triage_alerts": list(self.triage_alerts),
            "murcko_scaffold": self.murcko_scaffold,
            "generic_scaffold": self.generic_scaffold,
            "admet": dict(self.admet) if self.admet is not None else None,
            "pred_pkd": round(self.pred_pkd, 3) if self.pred_pkd is not None else None,
            "clip_sim": round(self.clip_sim, 4) if self.clip_sim is not None else None,
            "composite_reward": round(self.composite_reward, 4),
        }
        if self.custom_scores:
            out["custom_scores"] = {k: round(float(v), 4) for k, v in self.custom_scores.items()}
        return out


# ---------------------------------------------------------------------------
# Pluggable Custom Scorers & Composable Filters (Protocols Implementations)
# ---------------------------------------------------------------------------


class FunctionScorer:
    """Adapter turning any plain callable into a :class:`MoleculeScorerProtocol` plugin.

    Supports both per-report callables ``fn(report) -> float`` (default) and
    batch callables ``fn(reports) -> Sequence[float]`` (when ``batch=True``).

    Examples:
        >>> import geneuclidean as ge
        >>> low_mw = ge.FunctionScorer("low_mw_bonus", lambda r: 1.0 if r.mw <= 360 else 0.5)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        batch: bool = False,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Scorer name cannot be empty.")
        self._name = cleaned
        self._fn = fn
        self._batch = bool(batch)

    @property
    def name(self) -> str:
        """Unique scorer identifier."""
        return self._name

    def score(self, reports: Sequence[MoleculeReport]) -> list[float]:
        """Evaluate this scorer across an ordered sequence of reports."""
        if not reports:
            return []
        if self._batch:
            values = list(self._fn(reports))
            if len(values) != len(reports):
                raise ValueError(
                    f"Batch scorer {self._name!r} returned {len(values)} scores for "
                    f"{len(reports)} reports."
                )
            return [float(v) if r.valid else 0.0 for r, v in zip(reports, values, strict=True)]
        return [float(self._fn(r)) if r.valid else 0.0 for r in reports]


@dataclass(frozen=True)
class TriageFilter:
    """Keep only valid molecules passing industrial MedChem structural triage."""

    require_triage_pass: bool = True
    min_hygiene_gate: float = 0.0
    max_lilly_demerits: int | None = None
    name: str = "triage_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when the report satisfies MedChem hygiene thresholds."""
        if not report.valid:
            return False
        if self.require_triage_pass and not report.triage_passed:
            return False
        if report.hygiene_gate < self.min_hygiene_gate:
            return False
        return not (
            self.max_lilly_demerits is not None
            and report.lilly_demerit_score > self.max_lilly_demerits
        )


@dataclass(frozen=True)
class MPOFilter:
    """Keep molecules meeting Pfizer CNS MPO v2 and optional physicochemical bounds."""

    min_mpo_v2: float = 4.0
    max_mw: float | None = None
    max_tpsa: float | None = None
    require_bbb_clark: bool = False
    name: str = "mpo_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when the report satisfies CNS MPO / physchem bounds."""
        if not report.valid or report.cns_mpo_v2 < self.min_mpo_v2:
            return False
        if self.max_mw is not None and (not math.isfinite(report.mw) or report.mw > self.max_mw):
            return False
        if self.max_tpsa is not None and (
            not math.isfinite(report.tpsa) or report.tpsa > self.max_tpsa
        ):
            return False
        return not (self.require_bbb_clark and not report.bbb_clark_pass)


@dataclass(frozen=True)
class ADMETFilter:
    """Keep molecules passing calibrated CNS ADMET / Avoid-ome probability criteria."""

    require_admet_pass: bool = True
    min_bbb_prob: float | None = None
    max_herg_prob: float | None = None
    max_pgp_prob: float | None = None
    pass_when_missing: bool = True
    name: str = "admet_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when ADMET oracle criteria are met (or absent if allowed)."""
        if not report.valid:
            return False
        if report.admet is None:
            return self.pass_when_missing
        if self.require_admet_pass and not _admet_pass(
            report.admet, missing=self.pass_when_missing
        ):
            return False
        probs = report.admet.get("probabilities", report.admet)
        for name, lower, upper in (
            ("bbb", self.min_bbb_prob, None),
            ("herg", None, self.max_herg_prob),
            ("pgp", None, self.max_pgp_prob),
        ):
            if lower is None and upper is None:
                continue
            value = probs.get(name, report.admet.get(f"{name}_prob"))
            if value is None:
                if self.pass_when_missing:
                    continue
                return False
            prob = float(value)
            if not 0.0 <= prob <= 1.0:
                return False
            if (lower is not None and prob < lower) or (upper is not None and prob > upper):
                return False
        return True


@dataclass(frozen=True)
class BasicAmineFilter:
    """Filter molecules by physiological pH 7.4 basic-amine salt-bridge readiness."""

    require_basic_amine: bool = True
    allowed_charges: tuple[int, ...] | None = None
    name: str = "basic_amine_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when pH 7.4 basicity and formal charge match requirements."""
        if not report.valid:
            return False
        if self.require_basic_amine and not report.has_basic_amine:
            return False
        return not (
            self.allowed_charges is not None and report.net_charge_ph74 not in self.allowed_charges
        )


@dataclass(frozen=True)
class QEDFilter:
    """Filter molecules by quantitative drug-likeness (QED) and synthetic accessibility."""

    min_qed: float = 0.50
    max_sa_score: float = 4.5
    name: str = "qed_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when QED and synthetic accessibility satisfy thresholds."""
        return report.valid and report.qed >= self.min_qed and report.sa_score <= self.max_sa_score


@dataclass(frozen=True)
class ScaffoldHopFilter:
    """Keep only candidates that represent genuine Bemis-Murcko scaffold hops from references."""

    reference_scaffolds: frozenset[str]
    max_tanimoto: float = 0.55
    name: str = "scaffold_hop_filter"

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` when the molecule hops outside ``reference_scaffolds``."""
        if not report.valid or not report.canonical_smiles:
            return False
        from geneuclidean.cns.medchem_triage import is_scaffold_hop

        return is_scaffold_hop(
            report.canonical_smiles,
            self.reference_scaffolds,
            max_tanimoto=self.max_tanimoto,
        )


class PredicateFilter:
    """Wrap any user predicate ``fn(report) -> bool`` as a named :class:`MoleculeFilterProtocol`."""

    def __init__(self, name: str, predicate: Callable[[MoleculeReport], bool]) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Filter name cannot be empty.")
        self._name = cleaned
        self._predicate = predicate

    @property
    def name(self) -> str:
        """Human-readable filter identifier."""
        return self._name

    def keep(self, report: MoleculeReport) -> bool:
        """Evaluate the wrapped user predicate."""
        return bool(self._predicate(report))


# Register built-in composable filter classes so configs/pipelines can instantiate by name.
FILTER_REGISTRY.register("triage", TriageFilter, override=True)
FILTER_REGISTRY.register("mpo", MPOFilter, override=True)
FILTER_REGISTRY.register("admet", ADMETFilter, override=True)
FILTER_REGISTRY.register("basic_amine", BasicAmineFilter, override=True)
FILTER_REGISTRY.register("qed", QEDFilter, override=True)
FILTER_REGISTRY.register("scaffold_hop", ScaffoldHopFilter, override=True)


def score_tourette_circuit(
    smiles: str,
    target_id: str,
    *,
    vina_dg_kcal: float | None = None,
    anchor_engaged: bool | None = None,
    posebusters_pass: bool | None = None,
    plif_tanimoto: float | None = None,
    oracle_path: str | Path | None = None,
) -> TouretteCircuitScore:
    """Score a candidate molecule on Tourette CSTC basal-ganglia & GABAergic circuit readiness.

    Combines 3D Vina affinity, target mechanism-of-action polarity fit (e.g., ``GABRA5``
    positive allosteric modulation, ``GAT1`` GABA reuptake inhibition, ``ADORA2A`` active
    agonism, ``GRM5`` negative allosteric modulation), industrial MedChem hygiene gating,
    and optional v4.0 12-endpoint Avoid-ome safety probabilities.
    """
    from geneuclidean.cns.admet import DEFAULT_V4_ADMET_MODEL_PATH, CNSAdmetSuite
    from geneuclidean.cns.tourette_circuit import score_tourette_circuit_candidate

    suite = (
        CNSAdmetSuite.load(oracle_path)
        if oracle_path is not None
        else CNSAdmetSuite.try_load(DEFAULT_V4_ADMET_MODEL_PATH)
    )
    return score_tourette_circuit_candidate(
        smiles=smiles,
        target_id=target_id,
        vina_dg_kcal=vina_dg_kcal,
        anchor_engaged=anchor_engaged,
        posebusters_pass=posebusters_pass,
        plif_tanimoto=plif_tanimoto,
        admet_suite=suite,
    )


# ---------------------------------------------------------------------------
# Core Scoring Function
# ---------------------------------------------------------------------------


def apply_custom_scorers(
    reports: Sequence[MoleculeReport],
    extra_scorers: Sequence[MoleculeScorerProtocol | tuple[MoleculeScorerProtocol, float]],
) -> list[MoleculeReport]:
    """Evaluate one or more :class:`MoleculeScorerProtocol` plugins onto ``reports``.

    Each entry in ``extra_scorers`` may be either:
    - a ``scorer`` instance (records ``custom_scores[scorer.name]`` without changing
      ``composite_reward``), or
    - a ``(scorer, weight)`` tuple (records ``custom_scores[scorer.name]`` AND blends
      ``weight * score * hygiene_gate`` into ``composite_reward`` so hygiene vetoes
      cannot be overridden by auxiliary bonuses).
    """
    if not reports or not extra_scorers:
        return list(reports)

    normalized: list[tuple[MoleculeScorerProtocol, float]] = []
    for entry in extra_scorers:
        if isinstance(entry, tuple):
            scorer, weight = entry
            normalized.append((scorer, float(weight)))
        else:
            normalized.append((entry, 0.0))

    per_report_scores: list[dict[str, float]] = [dict(r.custom_scores) for r in reports]
    reward_deltas: list[float] = [0.0] * len(reports)
    total_extra_weight = sum(max(0.0, w) for _, w in normalized)

    for scorer, weight in normalized:
        vals = list(scorer.score(reports))
        if len(vals) != len(reports):
            raise ValueError(
                f"Scorer {scorer.name!r} returned {len(vals)} scores for {len(reports)} reports."
            )
        for idx, (rep, val) in enumerate(zip(reports, vals, strict=True)):
            fval = float(val) if (rep.valid and math.isfinite(float(val))) else 0.0
            per_report_scores[idx][scorer.name] = fval
            if weight != 0.0 and rep.valid:
                reward_deltas[idx] += weight * fval * rep.hygiene_gate

    updated: list[MoleculeReport] = []
    for idx, rep in enumerate(reports):
        if not rep.valid:
            updated.append(replace(rep, custom_scores=per_report_scores[idx]))
            continue
        if total_extra_weight > 0.0:
            # Convexly blend base reward (weight 1.0) with weighted custom scorers.
            new_reward = (rep.composite_reward + reward_deltas[idx]) / (1.0 + total_extra_weight)
        else:
            new_reward = rep.composite_reward + reward_deltas[idx]
        updated.append(
            replace(
                rep,
                composite_reward=float(new_reward),
                custom_scores=per_report_scores[idx],
            )
        )
    return updated


def score_molecules(
    smiles_list: Sequence[str] | str,
    *,
    target: str | None = None,
    include_triage: bool = True,
    include_admet: bool = True,
    admet_profile: Literal["v3", "v4", "auto"] = "v3",
    oracle_path: str | Path | None = None,
    ad_damp: bool = True,
    pred_pkd: Sequence[float] | None = None,
    clip_sim: Sequence[float] | None = None,
    extra_scorers: Sequence[MoleculeScorerProtocol | tuple[MoleculeScorerProtocol, float]]
    | None = None,
) -> list[MoleculeReport]:
    """Score one or more SMILES across Pfizer CNS MPO, pH 7.4 ionization, triage & ADMET.

    Args:
        smiles_list: Single SMILES string or sequence of SMILES strings.
        target: Optional CNS/custom target identifier recorded on each report.
        include_triage: Run 2026 MedChem structural triage (PAINS/BRENK/NIH, reagent
            quarantine, Lilly demerits, Murcko scaffolds, smooth ``hygiene_gate``).
        include_admet: Evaluate calibrated ADMET oracles when available on disk.
        admet_profile: ``"v3"`` (default — exact v3.1 3-endpoint BBB/P-gp/hERG parity),
            ``"v4"`` (12-endpoint Industrial Avoid-ome/DMPK + Multiplicative Safety-Gated
            Reward), or ``"auto"`` (prefers v4 bundle when available, else falls back to v3).
        oracle_path: Explicit path to a trained ``CNSAdmetSuite`` ``.joblib`` artifact.
        ad_damp: Damp out-of-domain ADMET probabilities toward 0.50 in composite reward.
        pred_pkd: Optional per-molecule predicted binding affinity ``pKd`` scores.
        clip_sim: Optional per-molecule DrugCLIP cosine similarity scores.
        extra_scorers: Optional custom :class:`MoleculeScorerProtocol` instances or
            ``(scorer, weight)`` tuples evaluated on all reports.

    Returns:
        List of :class:`MoleculeReport` entries aligned 1-to-1 with ``smiles_list``.
    """
    from geneuclidean.cns.admet import (
        DEFAULT_ADMET_MODEL_PATH,
        DEFAULT_V4_ADMET_MODEL_PATH,
        CNSAdmetSuite,
        compute_cns_composite_reward,
        compute_multiplicative_cns_reward,
    )
    from geneuclidean.cns.medchem_triage import triage_batch
    from geneuclidean.evaluation.protonation import analyze_protonation_ph74

    items: list[str] = [smiles_list] if isinstance(smiles_list, str) else list(smiles_list)
    if not items:
        return []

    resolved_target: str | None = None
    if target is not None:
        resolved_target = get_target(target).target_id

    admet_suite: CNSAdmetSuite | None = None
    use_v4_reward = False
    if include_admet:
        if oracle_path is not None:
            admet_suite = CNSAdmetSuite.load(oracle_path)
            use_v4_reward = admet_profile == "v4"
        elif admet_profile == "v4":
            admet_suite = CNSAdmetSuite.try_load(DEFAULT_V4_ADMET_MODEL_PATH)
            use_v4_reward = admet_suite is not None
        elif admet_profile == "auto":
            admet_suite = CNSAdmetSuite.try_load(DEFAULT_V4_ADMET_MODEL_PATH)
            if admet_suite is not None:
                use_v4_reward = True
            else:
                admet_suite = CNSAdmetSuite.try_load(DEFAULT_ADMET_MODEL_PATH)
        else:
            admet_suite = CNSAdmetSuite.try_load(DEFAULT_ADMET_MODEL_PATH)

    if use_v4_reward:
        mult_scores = compute_multiplicative_cns_reward(
            items,
            pred_pkd=pred_pkd,
            clip_sim=clip_sim,
            admet_suite=admet_suite,
            ad_damp=ad_damp,
        )
        triage_results = triage_batch(items) if include_triage else None
        reports_v4: list[MoleculeReport] = []
        for idx, smi in enumerate(items):
            ms = mult_scores[idx]
            mpo = ms.mpo
            tr = triage_results[idx] if triage_results is not None else ms.triage
            if not mpo.valid:
                reports_v4.append(
                    MoleculeReport(
                        smiles=smi,
                        canonical_smiles=None,
                        valid=False,
                        target_id=resolved_target,
                        triage_passed=False,
                        hygiene_gate=0.0,
                        triage_alerts=("invalid_smiles",),
                        composite_reward=0.0,
                    )
                )
                continue
            prot = analyze_protonation_ph74(mpo.canonical_smiles or smi)
            admet_dict = ms.admet_v4.to_dict() if ms.admet_v4 is not None else None
            alerts = (
                tuple(
                    list(tr.reagent_alerts)
                    + list(tr.pains_alerts)
                    + list(tr.brenk_alerts)
                    + list(tr.nih_alerts)
                    + list(tr.lilly_demerit_reasons)
                )
                if tr is not None
                else ()
            )
            reports_v4.append(
                MoleculeReport(
                    smiles=smi,
                    canonical_smiles=mpo.canonical_smiles,
                    valid=True,
                    target_id=resolved_target,
                    smiles_ph74=prot.protonated_smiles,
                    net_charge_ph74=prot.net_charge,
                    has_basic_amine=prot.has_basic_amine,
                    mw=mpo.mw,
                    clogp=mpo.clogp,
                    clogd_74=mpo.clogd_74,
                    tpsa=mpo.tpsa,
                    hbd=mpo.hbd,
                    hba=mpo.hba,
                    most_basic_pka=mpo.most_basic_pka,
                    cns_mpo_v1=mpo.cns_mpo_v1,
                    cns_mpo_v2=mpo.cns_mpo_v2,
                    bbb_clark_pass=mpo.bbb_clark_pass,
                    qed=mpo.qed,
                    sa_score=mpo.sa_score,
                    sa_norm=mpo.sa_norm,
                    lipinski_violations=mpo.lipinski_violations,
                    lipinski_pass=mpo.lipinski_pass,
                    pains_free=mpo.pains_free,
                    brenk_free=mpo.brenk_free,
                    nih_free=bool(tr.nih_free) if tr is not None else True,
                    reagent_quarantine_free=(
                        bool(tr.reagent_quarantine_free) if tr is not None else True
                    ),
                    lilly_demerit_score=int(tr.lilly_demerit_score) if tr is not None else 0,
                    triage_passed=bool(tr.triage_pass) if tr is not None else mpo.pains_free,
                    hygiene_gate=float(ms.hygiene_gate),
                    triage_alerts=alerts,
                    murcko_scaffold=tr.exact_murcko_scaffold if tr is not None else "",
                    generic_scaffold=tr.generic_murcko_scaffold if tr is not None else "",
                    admet=admet_dict,
                    pred_pkd=ms.pred_pkd,
                    clip_sim=ms.clip_sim,
                    composite_reward=float(ms.total_reward),
                )
            )
        return apply_custom_scorers(reports_v4, extra_scorers or ())

    comp_scores = compute_cns_composite_reward(
        items,
        pred_pkd=pred_pkd,
        clip_sim=clip_sim,
        admet_suite=admet_suite,
        ad_damp=ad_damp,
    )
    triage_results = triage_batch(items) if include_triage else None

    reports: list[MoleculeReport] = []
    for idx, smi in enumerate(items):
        comp = comp_scores[idx]
        mpo = comp.mpo
        tr_item: MedChemTriageResult | None = (
            triage_results[idx] if triage_results is not None else None
        )

        if not mpo.valid:
            reports.append(
                MoleculeReport(
                    smiles=smi,
                    canonical_smiles=None,
                    valid=False,
                    target_id=resolved_target,
                    triage_passed=False,
                    hygiene_gate=0.0,
                    triage_alerts=("invalid_smiles",),
                    composite_reward=0.0,
                )
            )
            continue

        prot = analyze_protonation_ph74(mpo.canonical_smiles or smi)
        admet_dict = comp.admet.to_dict() if comp.admet is not None else None

        if tr_item is not None:
            alerts = tuple(
                list(tr_item.reagent_alerts)
                + list(tr_item.pains_alerts)
                + list(tr_item.brenk_alerts)
                + list(tr_item.nih_alerts)
                + list(tr_item.lilly_demerit_reasons)
            )
            hygiene_gate = float(tr_item.hygiene_gate)
            triage_passed = bool(tr_item.triage_pass)
            nih_free = bool(tr_item.nih_free)
            reagent_free = bool(tr_item.reagent_quarantine_free)
            lilly_score = int(tr_item.lilly_demerit_score)
            murcko = tr_item.exact_murcko_scaffold
            generic_scaf = tr_item.generic_murcko_scaffold
        else:
            alerts = ()
            hygiene_gate = 1.0 if mpo.pains_free else 0.0
            triage_passed = bool(mpo.pains_free and mpo.brenk_free)
            nih_free = True
            reagent_free = True
            lilly_score = 0
            murcko = ""
            generic_scaf = ""

        # Multiplicative hygiene gate guarantees DCU/reagent/PAINS artifacts receive 0.0.
        gated_reward = float(comp.composite_reward * hygiene_gate)

        reports.append(
            MoleculeReport(
                smiles=smi,
                canonical_smiles=mpo.canonical_smiles,
                valid=True,
                target_id=resolved_target,
                smiles_ph74=prot.protonated_smiles,
                net_charge_ph74=prot.net_charge,
                has_basic_amine=prot.has_basic_amine,
                mw=mpo.mw,
                clogp=mpo.clogp,
                clogd_74=mpo.clogd_74,
                tpsa=mpo.tpsa,
                hbd=mpo.hbd,
                hba=mpo.hba,
                most_basic_pka=mpo.most_basic_pka,
                cns_mpo_v1=mpo.cns_mpo_v1,
                cns_mpo_v2=mpo.cns_mpo_v2,
                bbb_clark_pass=mpo.bbb_clark_pass,
                qed=mpo.qed,
                sa_score=mpo.sa_score,
                sa_norm=mpo.sa_norm,
                lipinski_violations=mpo.lipinski_violations,
                lipinski_pass=mpo.lipinski_pass,
                pains_free=mpo.pains_free,
                brenk_free=mpo.brenk_free,
                nih_free=nih_free,
                reagent_quarantine_free=reagent_free,
                lilly_demerit_score=lilly_score,
                triage_passed=triage_passed,
                hygiene_gate=hygiene_gate,
                triage_alerts=alerts,
                murcko_scaffold=murcko,
                generic_scaffold=generic_scaf,
                admet=admet_dict,
                pred_pkd=comp.pred_pkd,
                clip_sim=comp.clip_sim,
                composite_reward=gated_reward,
            )
        )

    return apply_custom_scorers(reports, extra_scorers or ())


__all__ = [
    "ADMETFilter",
    "BasicAmineFilter",
    "FunctionScorer",
    "MPOFilter",
    "MoleculeFilterProtocol",
    "MoleculeReport",
    "MoleculeScorerProtocol",
    "PredicateFilter",
    "QEDFilter",
    "ScaffoldHopFilter",
    "TriageFilter",
    "apply_custom_scorers",
    "protonate_ph74",
    "score_molecules",
    "score_tourette_circuit",
    "triage_molecule",
    "triage_molecules",
]
