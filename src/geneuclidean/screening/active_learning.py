"""Pool-based Active-Learning DMTA (Lab-in-a-Loop) engine with uncertainty & diversity.

Preserves 100% backward compatibility with GenEuclidean v5.0's :class:`ActiveLearningDMTA`,
:class:`ActiveLearningRoundSummary`, and :data:`AcquisitionStrategy` while cleanly
upgrading and delegating to the v5.1 :mod:`geneuclidean.active_learning` flagship engine
(:class:`~geneuclidean.active_learning.engine.ActiveLearningEngine`,
:class:`~geneuclidean.active_learning.surrogates.TanimotoGP`,
:class:`~geneuclidean.active_learning.acquisitions.ACQUISITION_REGISTRY`,
:class:`DMTAQuery`, :class:`DMTARoundSummary`, :class:`DMTALedger`).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from geneuclidean.active_learning.acquisitions import (
    AcquisitionContext,
    run_acquisition,
)
from geneuclidean.active_learning.engine import (
    DMTALedger,
    DMTAQuery,
    DMTARoundSummary,
)
from geneuclidean.active_learning.surrogates import (
    SurrogateProtocol,
    build_surrogate,
)
from geneuclidean.screening.encoders import featurize_ecfp4_batch

AcquisitionStrategy = Literal[
    "ucb",
    "thompson",
    "greedy",
    "maxmin_diverse_ucb",
    "maxmin_ucb",
    "ei",
    "log_ei",
    "pi",
    "uncertainty",
    "kriging_believer",
    "dpp_ucb",
    "constrained_ei",
    "avoidome_feasibility_ucb",
]


@dataclass(frozen=True)
class ActiveLearningRoundSummary:
    """Telemetry summary for a single DMTA active-learning iteration."""

    round_index: int
    strategy: AcquisitionStrategy
    n_acquired: int
    n_total_observed: int
    mean_acquired_value: float
    max_acquired_value: float
    best_overall_value: float
    best_overall_smiles: str
    round_sha256: str
    acquired_smiles: tuple[str, ...]
    acquired_values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "strategy": self.strategy,
            "n_acquired": self.n_acquired,
            "n_total_observed": self.n_total_observed,
            "mean_acquired_value": round(self.mean_acquired_value, 4),
            "max_acquired_value": round(self.max_acquired_value, 4),
            "best_overall_value": round(self.best_overall_value, 4),
            "best_overall_smiles": self.best_overall_smiles,
            "round_sha256": self.round_sha256,
            "acquired_smiles": list(self.acquired_smiles),
            "acquired_values": [round(float(v), 4) for v in self.acquired_values],
        }


class ActiveLearningDMTA:
    """Iterative pool-based active learning for virtual screening & wet-lab DMTA cycles.

    Backward-compatible v5.0 facade enhanced with optional v5.1 ``surrogate``
    (``"rf_legacy"`` default preserves exact v5.0 RF behavior; pass ``"tanimoto_gp"``,
    ``"minmax_gp"``, ``"ensemble"``, ``"conformal_tanimoto_gp"``, or any
    :class:`SurrogateProtocol`) and full :data:`ACQUISITION_REGISTRY` strategy support.
    """

    def __init__(
        self,
        *,
        strategy: AcquisitionStrategy = "maxmin_diverse_ucb",
        ucb_beta: float = 1.5,
        diversity_weight: float = 0.25,
        n_estimators: int = 64,
        fp_bits: int = 1024,
        surrogate: str | SurrogateProtocol = "rf_legacy",
        seed: int = 42,
    ) -> None:
        self.strategy: AcquisitionStrategy = strategy
        self.ucb_beta = float(ucb_beta)
        self.diversity_weight = float(diversity_weight)
        self.n_estimators = int(n_estimators)
        self.fp_bits = int(fp_bits)
        self.surrogate_spec: str | SurrogateProtocol = surrogate
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.observed: dict[str, float] = {}
        self.history: list[ActiveLearningRoundSummary] = []
        self.ledger = DMTALedger()
        self._surrogate: RandomForestRegressor | None = None
        self._v51_surrogate: SurrogateProtocol | None = None

    def ingest_observations(self, observations: Mapping[str, float]) -> int:
        """Record experimental or oracle values ``{smiles: higher_is_better_score}``."""
        added = 0
        for smi, val in observations.items():
            s = str(smi).strip()
            if s and np.isfinite(float(val)):
                self.observed[s] = float(val)
                added += 1
        return added

    def ingest_assay_csv(
        self,
        csv_path: str | Path,
        *,
        smiles_col: str = "smiles",
        value_col: str = "value",
        negate_values: bool = False,
    ) -> int:
        """Ingest wet-lab assay results (e.g., ``pIC50`` or negated Vina ``DeltaG``) from CSV."""
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

    def _fit_surrogate(self) -> RandomForestRegressor | SurrogateProtocol | None:
        if len(self.observed) < 4:
            return None
        smis = list(self.observed.keys())
        ys = [self.observed[s] for s in smis]

        if self.surrogate_spec != "rf_legacy":
            mdl = (
                build_surrogate(self.surrogate_spec, seed=self.seed + len(self.history))
                if isinstance(self.surrogate_spec, str)
                else self.surrogate_spec
            )
            mdl.fit(smis, ys)
            self._v51_surrogate = mdl
            return mdl

        ys_arr = np.array(ys, dtype=np.float32)
        x_mat, valid = featurize_ecfp4_batch(smis, n_bits=self.fp_bits)
        if int(valid.sum()) < 4:
            return None
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            random_state=self.seed + len(self.history),
            n_jobs=1,
        )
        rf.fit(x_mat[valid], ys_arr[valid])
        self._surrogate = rf
        return rf

    def predict_mean_and_std(
        self, pool_smiles: Sequence[str]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Predict surrogate mean and epistemic/total uncertainty across ``pool_smiles``."""
        n = len(pool_smiles)
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        if self._v51_surrogate is not None:
            pred = self._v51_surrogate.predict(pool_smiles)
            means = np.where(pred.valid & np.isfinite(pred.mean), pred.mean, -1e9).astype(
                np.float32
            )
            stds = np.where(pred.valid & np.isfinite(pred.std), pred.std, 0.0).astype(np.float32)
            return means, stds

        if self._surrogate is None:
            return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
        x_pool, valid = featurize_ecfp4_batch(pool_smiles, n_bits=self.fp_bits)
        means = np.full(n, -1e9, dtype=np.float32)
        stds = np.zeros(n, dtype=np.float32)
        if not bool(valid.any()):
            return means, stds
        xv = x_pool[valid]
        tree_preds = np.stack([est.predict(xv) for est in self._surrogate.estimators_], axis=0)
        means[valid] = np.mean(tree_preds, axis=0).astype(np.float32)
        stds[valid] = np.std(tree_preds, axis=0).astype(np.float32)
        return means, stds

    def propose_batch(
        self,
        pool_smiles: Sequence[str],
        *,
        batch_size: int = 16,
        prior_scores: Sequence[float] | None = None,
        feasibility_probs: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> list[str]:
        """Select the next ``batch_size`` unobserved compounds from ``pool_smiles``."""
        unseen_idx = [i for i, s in enumerate(pool_smiles) if s.strip() not in self.observed]
        if not unseen_idx:
            return []
        unseen_smis = [pool_smiles[i].strip() for i in unseen_idx]
        k = min(len(unseen_smis), max(1, int(batch_size)))

        fitted = self._fit_surrogate()
        if fitted is None:
            if prior_scores is not None and len(prior_scores) == len(pool_smiles):
                p_arr = np.array([float(prior_scores[i]) for i in unseen_idx], dtype=np.float64)
                order = np.argsort(p_arr)[::-1][:k]
                return [unseen_smis[int(i)] for i in order]
            chosen = self._rng.choice(len(unseen_smis), size=k, replace=False)
            return [unseen_smis[int(i)] for i in chosen]

        mean32, std32 = self.predict_mean_and_std(unseen_smis)
        mean64 = np.where(mean32 > -1e8, mean32.astype(np.float64), float("nan"))
        std64 = std32.astype(np.float64)
        feas_unseen = (
            np.array([float(feasibility_probs[i]) for i in unseen_idx], dtype=np.float64)
            if feasibility_probs is not None and len(feasibility_probs) == len(pool_smiles)
            else None
        )
        best_obs = max(self.observed.values()) if self.observed else float("-inf")
        ctx = AcquisitionContext(
            pool_smiles=tuple(unseen_smis),
            mean=mean64,
            std=std64,
            best_observed=best_obs,
            batch_size=k,
            beta=self.ucb_beta,
            diversity_weight=self.diversity_weight,
            surrogate=self._v51_surrogate,
            feasibility_probs=feas_unseen,
            seed=self.seed + len(self.history) * 17,
        )
        sel = run_acquisition(self.strategy, ctx)
        return list(sel.smiles)

    def run_round(
        self,
        pool_smiles: Sequence[str],
        oracle_fn: Callable[[Sequence[str]], Sequence[float]],
        *,
        batch_size: int = 16,
        prior_scores: Sequence[float] | None = None,
        feasibility_probs: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> ActiveLearningRoundSummary:
        """Execute one full propose -> oracle-evaluate -> update DMTA round."""
        proposed = self.propose_batch(
            pool_smiles,
            batch_size=batch_size,
            prior_scores=prior_scores,
            feasibility_probs=feasibility_probs,
        )
        if not proposed:
            raise ValueError("No unobserved compounds remain in pool_smiles.")
        values = [float(v) for v in oracle_fn(proposed)]
        if len(values) != len(proposed):
            raise ValueError("oracle_fn returned mismatched length vs proposed batch.")
        self.ingest_observations(dict(zip(proposed, values, strict=True)))

        best_smi, best_val = max(self.observed.items(), key=lambda kv: kv[1])
        digest = hashlib.sha256(
            json.dumps(
                {"round": len(self.history) + 1, "smiles": proposed, "values": values},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        summary = ActiveLearningRoundSummary(
            round_index=len(self.history) + 1,
            strategy=self.strategy,
            n_acquired=len(proposed),
            n_total_observed=len(self.observed),
            mean_acquired_value=float(np.mean(values)),
            max_acquired_value=float(np.max(values)),
            best_overall_value=float(best_val),
            best_overall_smiles=best_smi,
            round_sha256=digest,
            acquired_smiles=tuple(proposed),
            acquired_values=tuple(values),
        )
        self.history.append(summary)
        # Synchronize the chained v5.1 SHA-256 DMTALedger alongside legacy history
        pre_round = DMTARoundSummary(
            round_index=summary.round_index,
            strategy=str(summary.strategy),
            surrogate_name=(
                self._v51_surrogate.name
                if self._v51_surrogate is not None
                else str(self.surrogate_spec)
            ),
            n_acquired=summary.n_acquired,
            n_total_observed=summary.n_total_observed,
            n_pending=0,
            mean_acquired_value=summary.mean_acquired_value,
            max_acquired_value=summary.max_acquired_value,
            best_overall_value=summary.best_overall_value,
            best_overall_smiles=summary.best_overall_smiles,
            mean_batch_internal_tanimoto=0.0,
            round_sha256=summary.round_sha256,
            chain_sha256="",
            acquired_smiles=summary.acquired_smiles,
            acquired_values=summary.acquired_values,
        )
        chain_hash = self.ledger.record_round(pre_round)
        self.ledger.rounds[-1] = DMTARoundSummary(
            round_index=pre_round.round_index,
            strategy=pre_round.strategy,
            surrogate_name=pre_round.surrogate_name,
            n_acquired=pre_round.n_acquired,
            n_total_observed=pre_round.n_total_observed,
            n_pending=pre_round.n_pending,
            mean_acquired_value=pre_round.mean_acquired_value,
            max_acquired_value=pre_round.max_acquired_value,
            best_overall_value=pre_round.best_overall_value,
            best_overall_smiles=pre_round.best_overall_smiles,
            mean_batch_internal_tanimoto=pre_round.mean_batch_internal_tanimoto,
            round_sha256=pre_round.round_sha256,
            chain_sha256=chain_hash,
            acquired_smiles=pre_round.acquired_smiles,
            acquired_values=pre_round.acquired_values,
        )
        return summary


__all__ = [
    "AcquisitionStrategy",
    "ActiveLearningDMTA",
    "ActiveLearningRoundSummary",
    "DMTALedger",
    "DMTAQuery",
    "DMTARoundSummary",
]
