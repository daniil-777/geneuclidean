"""Unified Pool-Based & Async Delayed-Batch Active Learning DMTA Engine.

Orchestrates end-to-end Design-Make-Test-Analyze (DMTA) cycles with:

- Pluggable surrogates (:data:`SURROGATE_REGISTRY`: ``tanimoto_gp``, ``minmax_gp``,
  ``ensemble``, ``conformal_tanimoto_gp``, ``conformal_ensemble``).
- Pluggable acquisitions (:data:`ACQUISITION_REGISTRY`: 14 single-objective, batch-diverse,
  multi-objective Pareto, and Avoid-ome feasibility-constrained strategies).
- Cold-start cluster seeding (``butina_medoids``, ``maxmin``, ``random``).
- **Asynchronous delayed-batch execution** (:class:`PendingBatchManager`): proposals
  can be marked ``pending`` while syntheses/assays are in flight; subsequent batches
  exclude pending molecules, condition GP/DPP/MaxMin diversity on pending structures,
  handle partial completions, and blacklist failed syntheses cleanly.
- **Chained SHA-256 Provenance Ledger** (:class:`DMTALedger`): tamper-evident audit
  chain across every query and round compatible with GenEuclidean's FDA/EMA Credibility
  Dossier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from geneuclidean.active_learning.acquisitions import (
    AcquisitionContext,
    AcquisitionFn,
    KBBaseAcq,
    LiarMode,
    compute_expected_improvement,
    compute_hypervolume,
    run_acquisition,
)
from geneuclidean.active_learning.diagnostics import (
    ColdStartStrategy,
    StoppingAdvisor,
    StoppingReport,
    compute_batch_internal_tanimoto,
    select_cold_start,
)
from geneuclidean.active_learning.surrogates import (
    SurrogatePrediction,
    SurrogateProtocol,
    build_surrogate,
)

QueryStatus = Literal["pending", "partial", "completed", "failed"]


@dataclass
class DMTAQuery:
    """Tracked DMTA batch proposal supporting synchronous and async delayed workflows."""

    query_id: str
    round_index: int
    strategy: str
    smiles: tuple[str, ...]
    acquisition_scores: tuple[float, ...]
    predicted_means: tuple[float, ...]
    predicted_stds: tuple[float, ...]
    interval_lo: tuple[float | None, ...] = ()
    interval_hi: tuple[float | None, ...] = ()
    status: QueryStatus = "pending"
    completed_smiles: list[str] = field(default_factory=list)
    failed_smiles: list[str] = field(default_factory=list)
    created_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "round_index": self.round_index,
            "strategy": self.strategy,
            "smiles": list(self.smiles),
            "acquisition_scores": [
                round(float(v), 5) if math.isfinite(v) else None for v in self.acquisition_scores
            ],
            "predicted_means": [
                round(float(v), 4) if math.isfinite(v) else None for v in self.predicted_means
            ],
            "predicted_stds": [
                round(float(v), 4) if math.isfinite(v) else None for v in self.predicted_stds
            ],
            "interval_lo": [
                round(float(v), 4) if v is not None and math.isfinite(v) else None
                for v in self.interval_lo
            ],
            "interval_hi": [
                round(float(v), 4) if v is not None and math.isfinite(v) else None
                for v in self.interval_hi
            ],
            "status": self.status,
            "completed_smiles": list(self.completed_smiles),
            "failed_smiles": list(self.failed_smiles),
            "created_sha256": self.created_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DMTARoundSummary:
    """Immutable telemetry record for one completed DMTA active-learning round."""

    round_index: int
    strategy: str
    surrogate_name: str
    n_acquired: int
    n_total_observed: int
    n_pending: int
    mean_acquired_value: float
    max_acquired_value: float
    best_overall_value: float
    best_overall_smiles: str
    mean_batch_internal_tanimoto: float
    round_sha256: str
    chain_sha256: str
    acquired_smiles: tuple[str, ...]
    acquired_values: tuple[float, ...]
    stopping_report: dict[str, Any] | None = None
    pareto_hypervolume: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "strategy": self.strategy,
            "surrogate_name": self.surrogate_name,
            "n_acquired": self.n_acquired,
            "n_total_observed": self.n_total_observed,
            "n_pending": self.n_pending,
            "mean_acquired_value": (
                round(self.mean_acquired_value, 4)
                if math.isfinite(self.mean_acquired_value)
                else None
            ),
            "max_acquired_value": (
                round(self.max_acquired_value, 4)
                if math.isfinite(self.max_acquired_value)
                else None
            ),
            "best_overall_value": (
                round(self.best_overall_value, 4)
                if math.isfinite(self.best_overall_value)
                else None
            ),
            "best_overall_smiles": self.best_overall_smiles,
            "mean_batch_internal_tanimoto": round(self.mean_batch_internal_tanimoto, 4),
            "round_sha256": self.round_sha256,
            "chain_sha256": self.chain_sha256,
            "acquired_smiles": list(self.acquired_smiles),
            "acquired_values": [round(float(v), 4) for v in self.acquired_values],
            "stopping_report": self.stopping_report,
            "pareto_hypervolume": (
                round(self.pareto_hypervolume, 6)
                if self.pareto_hypervolume is not None and math.isfinite(self.pareto_hypervolume)
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _LedgerEntry:
    seq: int
    kind: Literal["query", "round"]
    payload: dict[str, Any]
    prev_chain_sha256: str
    chain_sha256: str


class DMTALedger:
    """Append-only SHA-256 hash-chained provenance ledger for DMTA campaigns."""

    GENESIS_HASH: str = "0" * 64

    def __init__(self) -> None:
        self._entries: list[_LedgerEntry] = []
        self.queries: list[DMTAQuery] = []
        self.rounds: list[DMTARoundSummary] = []

    @property
    def head_sha256(self) -> str:
        return self._entries[-1].chain_sha256 if self._entries else self.GENESIS_HASH

    def _compute_chain_hash(self, prev_hash: str, kind: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"prev": prev_hash, "kind": kind, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def record_query(self, query: DMTAQuery) -> str:
        """Append a batch proposal query to the provenance chain and return its SHA-256."""
        payload = {
            "query_id": query.query_id,
            "round_index": query.round_index,
            "strategy": query.strategy,
            "smiles": list(query.smiles),
        }
        prev = self.head_sha256
        digest = self._compute_chain_hash(prev, "query", payload)
        query.created_sha256 = digest
        self._entries.append(
            _LedgerEntry(
                seq=len(self._entries) + 1,
                kind="query",
                payload=payload,
                prev_chain_sha256=prev,
                chain_sha256=digest,
            )
        )
        self.queries.append(query)
        return digest

    def record_round(self, summary: DMTARoundSummary) -> str:
        """Append a completed round summary to the provenance chain."""
        payload = {
            "round_index": summary.round_index,
            "strategy": summary.strategy,
            "surrogate_name": summary.surrogate_name,
            "acquired_smiles": list(summary.acquired_smiles),
            "acquired_values": [round(float(v), 6) for v in summary.acquired_values],
            "round_sha256": summary.round_sha256,
        }
        prev = self.head_sha256
        digest = self._compute_chain_hash(prev, "round", payload)
        self._entries.append(
            _LedgerEntry(
                seq=len(self._entries) + 1,
                kind="round",
                payload=payload,
                prev_chain_sha256=prev,
                chain_sha256=digest,
            )
        )
        self.rounds.append(summary)
        return digest

    def verify_chain(self) -> bool:
        """Recompute and verify every link in the SHA-256 provenance chain."""
        prev = self.GENESIS_HASH
        for entry in self._entries:
            if entry.prev_chain_sha256 != prev:
                return False
            expected = self._compute_chain_hash(prev, entry.kind, entry.payload)
            if expected != entry.chain_sha256:
                return False
            prev = entry.chain_sha256
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "geneuclidean-v5.1-dmta-ledger",
            "n_entries": len(self._entries),
            "head_sha256": self.head_sha256,
            "chain_valid": self.verify_chain(),
            "queries": [q.to_dict() for q in self.queries],
            "rounds": [r.to_dict() for r in self.rounds],
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return out


class PendingBatchManager:
    """Tracks asynchronous in-flight DMTA syntheses/assays and failed-synthesis blacklists."""

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}
        self._failed_blacklist: set[str] = set()
        self._queries_by_id: dict[str, DMTAQuery] = {}

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    @property
    def active_smiles(self) -> tuple[str, ...]:
        return tuple(self._pending.keys())

    @property
    def failed_blacklist(self) -> frozenset[str]:
        return frozenset(self._failed_blacklist)

    def excluded_set(self) -> set[str]:
        """Return union of in-flight pending and permanently failed-synthesis SMILES."""
        return set(self._pending.keys()) | self._failed_blacklist

    def register(self, query: DMTAQuery) -> None:
        self._queries_by_id[query.query_id] = query
        for smi in query.smiles:
            s = smi.strip()
            if s:
                self._pending[s] = query.query_id

    def complete(self, smiles: Sequence[str]) -> int:
        """Mark a (possibly partial) subset of pending SMILES as assay-completed."""
        cleared = 0
        touched_qids: set[str] = set()
        for raw in smiles:
            s = raw.strip()
            qid = self._pending.pop(s, None)
            if qid is not None:
                cleared += 1
                touched_qids.add(qid)
                q = self._queries_by_id.get(qid)
                if q is not None and s not in q.completed_smiles:
                    q.completed_smiles.append(s)
        self._refresh_query_statuses(touched_qids)
        return cleared

    def mark_failed(self, smiles: Sequence[str], *, blacklist: bool = True) -> int:
        """Remove failed syntheses from pending and optionally blacklist them from re-proposal."""
        failed_n = 0
        touched_qids: set[str] = set()
        for raw in smiles:
            s = raw.strip()
            if not s:
                continue
            if blacklist:
                self._failed_blacklist.add(s)
            qid = self._pending.pop(s, None)
            if qid is not None:
                failed_n += 1
                touched_qids.add(qid)
                q = self._queries_by_id.get(qid)
                if q is not None and s not in q.failed_smiles:
                    q.failed_smiles.append(s)
        self._refresh_query_statuses(touched_qids)
        return failed_n

    def _refresh_query_statuses(self, query_ids: set[str]) -> None:
        for qid in query_ids:
            q = self._queries_by_id.get(qid)
            if q is None:
                continue
            total = len(q.smiles)
            n_comp = len(q.completed_smiles)
            n_fail = len(q.failed_smiles)
            if n_comp + n_fail >= total:
                q.status = "failed" if n_comp == 0 and n_fail > 0 else "completed"
            elif n_comp > 0 or n_fail > 0:
                q.status = "partial"
            else:
                q.status = "pending"


class ActiveLearningEngine:
    """Flagship synchronous & asynchronous DMTA Active Learning orchestrator."""

    def __init__(
        self,
        *,
        surrogate: str | SurrogateProtocol = "tanimoto_gp",
        surrogate_kwargs: Mapping[str, Any] | None = None,
        acquisition: str | AcquisitionFn = "kriging_believer",
        cold_start: ColdStartStrategy = "butina_medoids",
        objective_names: Sequence[str] | None = None,
        reference_point: Sequence[float] | None = None,
        objective_weights: Sequence[float] | None = None,
        beta: float = 1.5,
        xi: float = 0.01,
        diversity_weight: float = 0.25,
        dpp_quality_temp: float = 1.0,
        liar_mode: LiarMode = "kb_mean",
        kb_base: KBBaseAcq = "ucb",
        qmc_samples: int = 256,
        min_obs_for_surrogate: int = 4,
        stopping_advisor: StoppingAdvisor | None = None,
        seed: int = 42,
    ) -> None:
        self._surrogate_spec = surrogate
        self._surrogate_kwargs = dict(surrogate_kwargs or {})
        self.acquisition = acquisition
        self.acquisition_name = (
            acquisition
            if isinstance(acquisition, str)
            else getattr(acquisition, "__name__", "custom")
        )
        self.cold_start: ColdStartStrategy = cold_start
        self.objective_names: tuple[str, ...] = (
            tuple(str(n) for n in objective_names) if objective_names is not None else ()
        )
        self.is_multi_objective: bool = len(self.objective_names) >= 2
        self.reference_point: NDArray[np.float64] | None = (
            np.asarray(reference_point, dtype=np.float64) if reference_point is not None else None
        )
        self.objective_weights: NDArray[np.float64] | None = (
            np.asarray(objective_weights, dtype=np.float64)
            if objective_weights is not None
            else None
        )
        self.beta = float(beta)
        self.xi = float(xi)
        self.diversity_weight = float(diversity_weight)
        self.dpp_quality_temp = float(dpp_quality_temp)
        self.liar_mode: LiarMode = liar_mode
        self.kb_base: KBBaseAcq = kb_base
        self.qmc_samples = int(qmc_samples)
        self.min_obs_for_surrogate = max(2, int(min_obs_for_surrogate))
        self.stopping_advisor = stopping_advisor or StoppingAdvisor()
        self.seed = int(seed)

        self.observed: dict[str, float] = {}
        self.observed_multi: dict[str, tuple[float, ...]] = {}
        self.observed_feasible: dict[str, bool] = {}
        self.pending = PendingBatchManager()
        self.ledger = DMTALedger()
        self.history: list[DMTARoundSummary] = []

        self._surrogates: list[SurrogateProtocol] = []
        self._query_counter: int = 0
        self._last_stopping_report: StoppingReport | None = None

    @property
    def n_observed(self) -> int:
        return len(self.observed_multi) if self.is_multi_objective else len(self.observed)

    @property
    def surrogate(self) -> SurrogateProtocol | None:
        return self._surrogates[0] if self._surrogates else None

    @property
    def last_stopping_report(self) -> StoppingReport | None:
        return self._last_stopping_report

    def _build_fresh_surrogate(self, offset: int = 0) -> SurrogateProtocol:
        if isinstance(self._surrogate_spec, str):
            kw = dict(self._surrogate_kwargs)
            kw.setdefault("seed", self.seed + offset)
            return build_surrogate(self._surrogate_spec, **kw)
        if offset == 0:
            return self._surrogate_spec
        if hasattr(self._surrogate_spec, "_clone_unfitted"):
            return self._surrogate_spec._clone_unfitted()  # type: ignore[no-any-return]
        return self._surrogate_spec

    def ingest_observations(
        self,
        observations: (
            Mapping[str, float] | Mapping[str, Sequence[float]] | Mapping[str, Mapping[str, float]]
        ),
        *,
        feasible_flags: Mapping[str, bool] | None = None,
    ) -> int:
        """Ingest single- or multi-objective observations and clear any matching pending entries."""
        added = 0
        completed_smis: list[str] = []
        for raw_smi, raw_val in observations.items():
            s = str(raw_smi).strip()
            if not s:
                continue
            if self.is_multi_objective:
                if isinstance(raw_val, Mapping):
                    missing = [k for k in self.objective_names if k not in raw_val]
                    if missing:
                        raise ValueError(
                            f"Missing objective keys {missing} for {s!r}; "
                            f"expected {list(self.objective_names)}."
                        )
                    vec = tuple(float(raw_val[k]) for k in self.objective_names)
                elif isinstance(raw_val, (int, float)):
                    raise ValueError(
                        f"Engine configured with {len(self.objective_names)} objectives; "
                        f"expected sequence or mapping of floats for {s!r}."
                    )
                else:
                    vec = tuple(float(x) for x in raw_val)
                if len(vec) != len(self.objective_names) or not all(math.isfinite(x) for x in vec):
                    continue
                self.observed_multi[s] = vec
                self.observed[s] = vec[0]
            else:
                if not isinstance(raw_val, (int, float)):
                    raise ValueError(
                        f"Single-objective engine expected float for {s!r}, got {type(raw_val)}."
                    )
                fval = float(raw_val)
                if not math.isfinite(fval):
                    continue
                self.observed[s] = fval

            if feasible_flags is not None and s in feasible_flags:
                self.observed_feasible[s] = bool(feasible_flags[s])
            completed_smis.append(s)
            added += 1

        if completed_smis:
            self.pending.complete(completed_smis)
        return added

    def complete_observations(
        self,
        observations: (
            Mapping[str, float] | Mapping[str, Sequence[float]] | Mapping[str, Mapping[str, float]]
        ),
        *,
        feasible_flags: Mapping[str, bool] | None = None,
    ) -> int:
        """Alias for :meth:`ingest_observations` in asynchronous delayed-batch workflows."""
        return self.ingest_observations(observations, feasible_flags=feasible_flags)

    def cancel_pending(self, failed_smiles: Sequence[str], *, blacklist: bool = True) -> int:
        """Mark in-flight syntheses as failed and remove them from pending queue."""
        return self.pending.mark_failed(failed_smiles, blacklist=blacklist)

    def ingest_assay_csv(
        self,
        csv_path: str | Path,
        *,
        smiles_col: str = "smiles",
        value_col: str = "value",
        negate_values: bool = False,
    ) -> int:
        """Ingest wet-lab CSV measurements (e.g., ``pIC50`` or negated docking ``DeltaG``)."""
        src = Path(csv_path)
        obs: dict[str, float] = {}
        with src.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                s = (row.get(smiles_col) or "").strip()
                raw_v = (row.get(value_col) or "").strip()
                if not s or not raw_v:
                    continue
                val = float(raw_v)
                obs[s] = -val if negate_values else val
        return self.ingest_observations(obs)

    def _fit_surrogates(self) -> list[SurrogateProtocol]:
        if self.n_observed < self.min_obs_for_surrogate:
            self._surrogates = []
            return []
        if self.is_multi_objective:
            smis = list(self.observed_multi.keys())
            y_mat = np.array([self.observed_multi[s] for s in smis], dtype=np.float64)
            models: list[SurrogateProtocol] = []
            for m_idx in range(len(self.objective_names)):
                mdl = self._build_fresh_surrogate(offset=len(self.history) * 10 + m_idx)
                mdl.fit(smis, y_mat[:, m_idx].tolist())
                models.append(mdl)
            self._surrogates = models
            return models

        smis = list(self.observed.keys())
        ys = [self.observed[s] for s in smis]
        mdl = self._build_fresh_surrogate(offset=len(self.history))
        mdl.fit(smis, ys)
        self._surrogates = [mdl]
        return self._surrogates

    def predict(self, pool_smiles: Sequence[str]) -> list[SurrogatePrediction]:
        """Predict surrogate posterior across ``pool_smiles`` (1 per objective)."""
        if not self._surrogates:
            self._fit_surrogates()
        if not self._surrogates:
            n = len(pool_smiles)
            return [
                SurrogatePrediction(
                    mean=np.zeros(n, dtype=np.float64),
                    std=np.ones(n, dtype=np.float64),
                    epistemic_std=np.ones(n, dtype=np.float64),
                    aleatoric_std=np.zeros(n, dtype=np.float64),
                    valid=np.ones(n, dtype=np.bool_),
                )
            ]
        return [m.predict(pool_smiles) for m in self._surrogates]

    def _best_observed_for_acquisition(self, use_feasibility_best: bool) -> float:
        if not self.observed:
            return float("-inf")
        if use_feasibility_best and self.observed_feasible:
            feas_vals = [
                v for s, v in self.observed.items() if self.observed_feasible.get(s, False)
            ]
            return float(max(feas_vals)) if feas_vals else float("-inf")
        return float(max(self.observed.values()))

    def propose_batch(
        self,
        pool_smiles: Sequence[str],
        *,
        batch_size: int = 16,
        feasibility_probs: Sequence[float] | NDArray[np.float64] | None = None,
        prior_scores: Sequence[float] | None = None,
        mark_pending: bool = False,
    ) -> DMTAQuery:
        """Select the next ``batch_size`` unobserved, non-pending compounds from ``pool_smiles``."""
        excluded = set(self.observed.keys()) | self.pending.excluded_set()
        unseen_idx: list[int] = []
        seen_pool: set[str] = set()
        for i, raw_s in enumerate(pool_smiles):
            s = raw_s.strip()
            if s and s not in excluded and s not in seen_pool:
                seen_pool.add(s)
                unseen_idx.append(i)
        self._query_counter += 1
        qid = f"dmta_q{self._query_counter:04d}"
        next_round = len(self.history) + 1

        if not unseen_idx:
            empty_q = DMTAQuery(
                query_id=qid,
                round_index=next_round,
                strategy=self.acquisition_name,
                smiles=(),
                acquisition_scores=(),
                predicted_means=(),
                predicted_stds=(),
                status="completed",
            )
            self.ledger.record_query(empty_q)
            return empty_q

        unseen_smis = tuple(pool_smiles[i].strip() for i in unseen_idx)
        k = min(len(unseen_smis), max(1, int(batch_size)))

        models = self._fit_surrogates()
        # Cold-start phase (insufficient observations for surrogate fitting)
        if not models:
            if prior_scores is not None and len(prior_scores) == len(pool_smiles):
                p_arr = np.array([float(prior_scores[i]) for i in unseen_idx], dtype=np.float64)
                chosen_local = np.argsort(-p_arr, kind="mergesort")[:k].tolist()
                acq_sc = tuple(float(p_arr[i]) for i in chosen_local)
            else:
                chosen_local = select_cold_start(
                    unseen_smis,
                    k,
                    strategy=self.cold_start,
                    seed=self.seed + self._query_counter,
                )
                acq_sc = tuple(0.0 for _ in chosen_local)
            chosen_smis = tuple(unseen_smis[i] for i in chosen_local)
            query = DMTAQuery(
                query_id=qid,
                round_index=next_round,
                strategy=f"cold_start:{self.cold_start}",
                smiles=chosen_smis,
                acquisition_scores=acq_sc,
                predicted_means=tuple(float("nan") for _ in chosen_local),
                predicted_stds=tuple(float("nan") for _ in chosen_local),
                status="pending" if mark_pending else "completed",
                metadata={"cold_start": True, "n_pending_excluded": self.pending.n_pending},
            )
            self.ledger.record_query(query)
            if mark_pending:
                self.pending.register(query)
            return query

        preds = [m.predict(unseen_smis) for m in models]
        if self.is_multi_objective:
            mean_arr = np.column_stack([p.mean for p in preds])
            std_arr = np.column_stack([p.std for p in preds])
            obs_obj = np.array(
                [self.observed_multi[s] for s in self.observed_multi], dtype=np.float64
            )
            surrogate_for_ctx: SurrogateProtocol | None = None
        else:
            mean_arr = preds[0].mean
            std_arr = preds[0].std
            obs_obj = None
            surrogate_for_ctx = models[0]

        feas_unseen: NDArray[np.float64] | None = None
        if feasibility_probs is not None:
            if len(feasibility_probs) != len(pool_smiles):
                raise ValueError(
                    f"feasibility_probs len {len(feasibility_probs)} "
                    f"!= pool len {len(pool_smiles)}."
                )
            feas_unseen = np.array(
                [float(feasibility_probs[i]) for i in unseen_idx], dtype=np.float64
            )

        is_constrained_ei = isinstance(self.acquisition, str) and self.acquisition.lower() in (
            "constrained_ei",
            "cei",
        )
        best_obs = self._best_observed_for_acquisition(use_feasibility_best=is_constrained_ei)

        ctx = AcquisitionContext(
            pool_smiles=unseen_smis,
            mean=mean_arr,
            std=std_arr,
            best_observed=best_obs,
            batch_size=k,
            beta=self.beta,
            xi=self.xi,
            diversity_weight=self.diversity_weight,
            dpp_quality_temp=self.dpp_quality_temp,
            surrogate=surrogate_for_ctx,
            observed_objectives=obs_obj,
            reference_point=self.reference_point,
            objective_weights=self.objective_weights,
            feasibility_probs=feas_unseen,
            pending_smiles=self.pending.active_smiles,
            liar_mode=self.liar_mode,
            kb_base=self.kb_base,
            qmc_samples=self.qmc_samples,
            seed=self.seed + self._query_counter * 31,
        )
        sel = run_acquisition(self.acquisition, ctx)

        p0 = preds[0]
        lo_vals: tuple[float | None, ...] = (
            tuple(
                float(p0.interval_lo[i]) if math.isfinite(float(p0.interval_lo[i])) else None
                for i in sel.indices
            )
            if p0.interval_lo is not None
            else ()
        )
        hi_vals: tuple[float | None, ...] = (
            tuple(
                float(p0.interval_hi[i]) if math.isfinite(float(p0.interval_hi[i])) else None
                for i in sel.indices
            )
            if p0.interval_hi is not None
            else ()
        )

        query = DMTAQuery(
            query_id=qid,
            round_index=next_round,
            strategy=sel.strategy,
            smiles=sel.smiles,
            acquisition_scores=sel.scores,
            predicted_means=tuple(float(p0.mean[i]) for i in sel.indices),
            predicted_stds=tuple(float(p0.std[i]) for i in sel.indices),
            interval_lo=lo_vals,
            interval_hi=hi_vals,
            status="pending" if mark_pending else "completed",
            metadata={
                **sel.metadata,
                "surrogate": p0.metadata.get("surrogate", models[0].name),
                "n_pending_excluded": self.pending.n_pending,
            },
        )
        self.ledger.record_query(query)
        if mark_pending:
            self.pending.register(query)
        return query

    def run_round(
        self,
        pool_smiles: Sequence[str],
        oracle_fn: Callable[[Sequence[str]], Sequence[Any]],
        *,
        batch_size: int = 16,
        feasibility_probs: Sequence[float] | NDArray[np.float64] | None = None,
        feasible_check_fn: Callable[[Sequence[str]], Sequence[bool]] | None = None,
        prior_scores: Sequence[float] | None = None,
    ) -> DMTARoundSummary:
        """Execute one full propose -> oracle-evaluate -> ingest -> stopping-check round."""
        query = self.propose_batch(
            pool_smiles,
            batch_size=batch_size,
            feasibility_probs=feasibility_probs,
            prior_scores=prior_scores,
            mark_pending=True,
        )
        if not query.smiles:
            raise ValueError("No unobserved compounds remain in pool_smiles.")

        raw_vals = list(oracle_fn(query.smiles))
        if len(raw_vals) != len(query.smiles):
            raise ValueError(
                f"oracle_fn returned {len(raw_vals)} values for {len(query.smiles)} SMILES."
            )

        feas_map: dict[str, bool] | None = None
        if feasible_check_fn is not None:
            flags = [bool(f) for f in feasible_check_fn(query.smiles)]
            if len(flags) != len(query.smiles):
                raise ValueError("feasible_check_fn returned mismatched length.")
            feas_map = dict(zip(query.smiles, flags, strict=True))

        if self.is_multi_objective:
            obs_dict: dict[str, tuple[float, ...]] = {}
            for s, v in zip(query.smiles, raw_vals, strict=True):
                if isinstance(v, Mapping):
                    obs_dict[s] = tuple(float(v[k]) for k in self.objective_names)
                else:
                    obs_dict[s] = tuple(float(x) for x in v)
            self.ingest_observations(obs_dict, feasible_flags=feas_map)
            scalar_vals = [v[0] for v in obs_dict.values()]
        else:
            obs_scalar = {s: float(v) for s, v in zip(query.smiles, raw_vals, strict=True)}
            self.ingest_observations(obs_scalar, feasible_flags=feas_map)
            scalar_vals = list(obs_scalar.values())

        best_smi, best_val = max(self.observed.items(), key=lambda kv: kv[1])
        pareto_hv: float | None = None
        if self.is_multi_objective and self.observed_multi:
            y_all = np.array(list(self.observed_multi.values()), dtype=np.float64)
            ref = (
                self.reference_point
                if self.reference_point is not None
                else np.min(y_all, axis=0) - 0.1
            )
            pareto_hv = compute_hypervolume(y_all, ref, seed=self.seed)

        # Refit and run stopping advisor diagnostics on remaining pool
        models = self._fit_surrogates()
        stop_rep: StoppingReport | None = None
        if models:
            excluded = set(self.observed.keys()) | self.pending.excluded_set()
            rem_smis = [s.strip() for s in pool_smiles if s.strip() and s.strip() not in excluded]
            if rem_smis:
                pred_rem = models[0].predict(rem_smis)
                ei_rem = compute_expected_improvement(
                    pred_rem.mean, pred_rem.std, best_val, xi=self.xi
                )
                fin_ei = ei_rem[np.isfinite(ei_rem)]
                max_ei = float(np.max(fin_ei)) if len(fin_ei) else 0.0
                obs_std = float(np.std(list(self.observed.values())))
                stop_metric = pareto_hv if pareto_hv is not None else best_val
                stop_rep = self.stopping_advisor.update(
                    best_metric=stop_metric,
                    max_ei=max_ei,
                    observed_std=obs_std,
                    pool_smiles=rem_smis,
                    pool_pred_mean=pred_rem.mean,
                )
                self._last_stopping_report = stop_rep

        int_tan = compute_batch_internal_tanimoto(query.smiles)
        round_idx = len(self.history) + 1
        round_digest = hashlib.sha256(
            json.dumps(
                {
                    "round": round_idx,
                    "smiles": list(query.smiles),
                    "values": [round(float(v), 6) for v in scalar_vals],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]

        surrogate_name = models[0].name if models else str(self._surrogate_spec)
        # Pre-compute summary, then record on chained SHA-256 ledger
        pre_summary = DMTARoundSummary(
            round_index=round_idx,
            strategy=query.strategy,
            surrogate_name=surrogate_name,
            n_acquired=len(query.smiles),
            n_total_observed=self.n_observed,
            n_pending=self.pending.n_pending,
            mean_acquired_value=float(np.mean(scalar_vals)) if scalar_vals else float("nan"),
            max_acquired_value=float(np.max(scalar_vals)) if scalar_vals else float("nan"),
            best_overall_value=float(best_val),
            best_overall_smiles=best_smi,
            mean_batch_internal_tanimoto=int_tan,
            round_sha256=round_digest,
            chain_sha256="",
            acquired_smiles=query.smiles,
            acquired_values=tuple(float(v) for v in scalar_vals),
            stopping_report=stop_rep.to_dict() if stop_rep is not None else None,
            pareto_hypervolume=pareto_hv,
            metadata={"query_id": query.query_id},
        )
        chain_hash = self.ledger.record_round(pre_summary)
        final_summary = DMTARoundSummary(
            round_index=pre_summary.round_index,
            strategy=pre_summary.strategy,
            surrogate_name=pre_summary.surrogate_name,
            n_acquired=pre_summary.n_acquired,
            n_total_observed=pre_summary.n_total_observed,
            n_pending=pre_summary.n_pending,
            mean_acquired_value=pre_summary.mean_acquired_value,
            max_acquired_value=pre_summary.max_acquired_value,
            best_overall_value=pre_summary.best_overall_value,
            best_overall_smiles=pre_summary.best_overall_smiles,
            mean_batch_internal_tanimoto=pre_summary.mean_batch_internal_tanimoto,
            round_sha256=pre_summary.round_sha256,
            chain_sha256=chain_hash,
            acquired_smiles=pre_summary.acquired_smiles,
            acquired_values=pre_summary.acquired_values,
            stopping_report=pre_summary.stopping_report,
            pareto_hypervolume=pre_summary.pareto_hypervolume,
            metadata=pre_summary.metadata,
        )
        # Keep ledger.rounds synchronized with the final chain_sha256
        self.ledger.rounds[-1] = final_summary
        self.history.append(final_summary)
        return final_summary


__all__ = [
    "ActiveLearningEngine",
    "DMTALedger",
    "DMTAQuery",
    "DMTARoundSummary",
    "PendingBatchManager",
    "QueryStatus",
]
