"""Composable multi-stage drug-design workflow pipelines and lifecycle hooks.

Inspired by modern scientific ML pipelines (scikit-learn ``Pipeline`` / Optax
chaining), :class:`DesignPipeline` lets computational chemists compose:

1. 3D pocket-conditioned generation (``de_novo`` or ``scaffold_decoration``),
2. Arbitrary custom :class:`~geneuclidean.protocols.MoleculeScorerProtocol` plugins,
3. Ordered :class:`~geneuclidean.protocols.MoleculeFilterProtocol` attrition stages,
4. Deterministic reranking, top-k selection, and optional multi-target batching,
5. Non-intrusive :class:`~geneuclidean.protocols.DesignCallbackProtocol` telemetry hooks.

Every builder method returns a **new** :class:`DesignPipeline` copy (immutable
fluent builder pattern) so base pipelines can be branched cleanly for A/B ablations
without shared mutable state bugs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geneuclidean.protocols import (
    DesignCallbackProtocol,
    MoleculeFilterProtocol,
    MoleculeScorerProtocol,
)
from geneuclidean.sdk.designer import DesignResult, MoleculeDesigner
from geneuclidean.sdk.pocket import Pocket
from geneuclidean.sdk.scoring import FunctionScorer, MoleculeReport, PredicateFilter

if TYPE_CHECKING:
    from geneuclidean.data.featurizer import PocketFeatures


class BaseDesignCallback:
    """No-op base implementation of :class:`DesignCallbackProtocol`.

    Subclass and override only the hooks you care about (e.g., ``on_design_end``
    for MLflow/W&B logging or JSON audit persistence).
    """

    def on_design_start(self, pocket: Pocket | Any, config: Mapping[str, Any]) -> None:
        """Called immediately before generation starts."""

    def on_candidates_generated(
        self,
        pocket: Pocket | Any,
        raw_smiles: Sequence[str],
        reports: Sequence[MoleculeReport],
    ) -> None:
        """Called after sampling & scoring, before pipeline filter stages execute."""

    def on_design_end(self, pocket: Pocket | Any, result: DesignResult) -> None:
        """Called after filtering and reranking complete."""


@dataclass(frozen=True)
class DesignPipeline:
    """Immutable fluent builder for reproducible 3D drug-design screening workflows.

    Examples:
        >>> import geneuclidean as ge
        >>> pipe = (
        ...     ge.DesignPipeline(designer, name="tourette_drd1")
        ...     .add_filter(ge.TriageFilter(max_lilly_demerits=100))
        ...     .add_filter(ge.MPOFilter(min_mpo_v2=4.5))
        ...     .add_scorer(
        ...         ge.FunctionScorer("low_mw", lambda r: 1.0 if r.mw <= 380 else 0.4),
        ...         weight=0.25,
        ...     )
        ...     .with_top_k(20)
        ... )
        >>> result = pipe.run(pocket, n=64, seed=42)
    """

    designer: MoleculeDesigner
    name: str = "geneuclidean_pipeline"
    filters: tuple[MoleculeFilterProtocol, ...] = ()
    scorers: tuple[tuple[MoleculeScorerProtocol, float], ...] = ()
    callbacks: tuple[DesignCallbackProtocol, ...] = ()
    cns_safe_only: bool = False
    include_triage: bool = True
    include_admet: bool = True
    admet_profile: Literal["v3", "v4", "auto"] = "v3"
    oracle_path: str | Path | None = None
    rerank: bool = True
    top_k_hits: int | None = None
    default_sampling_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def add_filter(
        self,
        mol_filter: MoleculeFilterProtocol | Callable[[MoleculeReport], bool],
        *,
        name: str | None = None,
    ) -> DesignPipeline:
        """Return a new pipeline with an appended post-generation filter stage."""
        resolved: MoleculeFilterProtocol
        if isinstance(mol_filter, MoleculeFilterProtocol):
            resolved = mol_filter
        elif callable(mol_filter):
            filter_label = str(name or getattr(mol_filter, "__name__", "custom_filter"))
            resolved = PredicateFilter(filter_label, mol_filter)
        else:
            raise TypeError(
                "add_filter expects a MoleculeFilterProtocol or callable(MoleculeReport) -> bool."
            )
        return replace(self, filters=(*self.filters, resolved))

    def add_scorer(
        self,
        scorer: MoleculeScorerProtocol | Callable[[MoleculeReport], float],
        *,
        weight: float = 0.0,
        name: str | None = None,
    ) -> DesignPipeline:
        """Return a new pipeline with an appended custom scoring plugin.

        Args:
            scorer: A :class:`MoleculeScorerProtocol` instance or a per-report callable
                ``fn(report) -> float``.
            weight: Non-negative blend weight into ``composite_reward`` (``0.0`` records
                ``report.custom_scores[name]`` purely for telemetry without altering ranking).
            name: Optional name required only when wrapping an anonymous lambda callable.
        """
        resolved: MoleculeScorerProtocol
        if isinstance(scorer, MoleculeScorerProtocol):
            resolved = scorer
        elif callable(scorer):
            scorer_label = str(name or getattr(scorer, "__name__", "custom_scorer"))
            resolved = FunctionScorer(scorer_label, scorer)
        else:
            raise TypeError(
                "add_scorer expects a MoleculeScorerProtocol or callable(MoleculeReport) -> float."
            )
        return replace(self, scorers=(*self.scorers, (resolved, float(weight))))

    def add_callback(self, callback: DesignCallbackProtocol) -> DesignPipeline:
        """Return a new pipeline with an attached lifecycle callback hook."""
        return replace(self, callbacks=(*self.callbacks, callback))

    def with_admet(
        self,
        *,
        enabled: bool = True,
        profile: Literal["v3", "v4", "auto"] = "v4",
        oracle_path: str | Path | None = None,
    ) -> DesignPipeline:
        """Configure calibrated CNS ADMET / Avoid-ome oracle evaluation."""
        return replace(
            self,
            include_admet=bool(enabled),
            admet_profile=profile,
            oracle_path=oracle_path,
        )

    def with_top_k(self, k: int | None, *, rerank: bool = True) -> DesignPipeline:
        """Configure final reranking and top-k hit cap."""
        return replace(self, top_k_hits=k, rerank=bool(rerank))

    def with_cns_safe_only(self, enabled: bool = True) -> DesignPipeline:
        """Toggle built-in strict CNS lead gate (valid + triage + MPO v2 >= 4.0 + ADMET)."""
        return replace(self, cns_safe_only=bool(enabled))

    def with_sampling(self, **kwargs: Any) -> DesignPipeline:
        """Attach default decoding hyperparameters (e.g., ``strategy``, ``cfg_scale``)."""
        merged = {**dict(self.default_sampling_kwargs), **kwargs}
        return replace(self, default_sampling_kwargs=merged)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable specification of this pipeline for audit logs & dossiers."""
        return {
            "name": self.name,
            "backend": f"{self.designer.backend}:{self.designer.backbone_id}",
            "filters": [flt.name for flt in self.filters],
            "scorers": [{"name": sc.name, "weight": wt} for sc, wt in self.scorers],
            "n_callbacks": len(self.callbacks),
            "cns_safe_only": self.cns_safe_only,
            "include_triage": self.include_triage,
            "include_admet": self.include_admet,
            "admet_profile": self.admet_profile,
            "rerank": self.rerank,
            "top_k_hits": self.top_k_hits,
            "default_sampling_kwargs": dict(self.default_sampling_kwargs),
        }

    def run(
        self,
        pocket: Pocket | PocketFeatures,
        *,
        n: int = 32,
        core_smiles_with_dummy: str | None = None,
        seed: int = 42,
        **sampling_overrides: Any,
    ) -> DesignResult:
        """Execute the composed pipeline on a single 3D binding pocket.

        When ``core_smiles_with_dummy`` is supplied, executes 3D scaffold decoration
        (:meth:`MoleculeDesigner.decorate`); otherwise runs *de novo* generation
        (:meth:`MoleculeDesigner.design`).
        """
        merged_sampling = {**dict(self.default_sampling_kwargs), **sampling_overrides}
        if core_smiles_with_dummy is not None:
            res = self.designer.decorate(
                pocket,
                core_smiles_with_dummy,
                n=n,
                cns_safe_only=self.cns_safe_only,
                include_triage=self.include_triage,
                include_admet=self.include_admet,
                admet_profile=self.admet_profile,
                oracle_path=self.oracle_path,
                extra_scorers=self.scorers,
                filters=self.filters,
                callbacks=self.callbacks,
                rerank=self.rerank,
                top_k_hits=self.top_k_hits,
                seed=seed,
                **merged_sampling,
            )
        else:
            res = self.designer.design(
                pocket,
                n=n,
                cns_safe_only=self.cns_safe_only,
                include_triage=self.include_triage,
                include_admet=self.include_admet,
                admet_profile=self.admet_profile,
                oracle_path=self.oracle_path,
                extra_scorers=self.scorers,
                filters=self.filters,
                callbacks=self.callbacks,
                rerank=self.rerank,
                top_k_hits=self.top_k_hits,
                seed=seed,
                **merged_sampling,
            )
        res.metadata["pipeline_name"] = self.name
        return res

    def run_batch(
        self,
        pockets: Sequence[Pocket | PocketFeatures],
        *,
        n: int = 32,
        seed: int = 42,
        **sampling_overrides: Any,
    ) -> dict[str, DesignResult]:
        """Run this pipeline deterministically across a panel of 3D pockets.

        Each pocket receives a deterministic offset seed ``seed + idx`` so multi-target
        campaigns are reproducible while avoiding identical RNG trajectories.
        """
        results: dict[str, DesignResult] = {}
        for idx, pocket in enumerate(pockets):
            res = self.run(pocket, n=n, seed=int(seed) + idx, **sampling_overrides)
            key = res.pocket_id
            if key in results:
                key = f"{key}_{idx}"
            results[key] = res
        return results


__all__ = ["BaseDesignCallback", "DesignPipeline"]
