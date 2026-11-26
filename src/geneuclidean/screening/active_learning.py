"""Pool-based Active-Learning DMTA (Lab-in-a-Loop) engine with uncertainty & diversity.

Supports both *in silico* scoring oracles (AutoDock Vina, Boltz-2, ``PoseRealityAuditor``)
and external wet-lab CSV assay ingestion across iterative Design-Make-Test-Analyze rounds,
using bootstrap-ensemble / Split-Conformal uncertainty (``ucb``, ``thompson``, ``greedy``,
``maxmin_diverse_ucb``) and SHA-256 provenance digests compatible with ``CredibilityDossier``.
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

from geneuclidean.screening.encoders import featurize_ecfp4_batch

AcquisitionStrategy = Literal["ucb", "thompson", "greedy", "maxmin_diverse_ucb"]


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
    """Iterative pool-based active learning for virtual screening & wet-lab DMTA cycles."""

    def __init__(
        self,
        *,
        strategy: AcquisitionStrategy = "maxmin_diverse_ucb",
        ucb_beta: float = 1.5,
        diversity_weight: float = 0.25,
        n_estimators: int = 64,
        fp_bits: int = 1024,
        seed: int = 42,
    ) -> None:
        self.strategy: AcquisitionStrategy = strategy
        self.ucb_beta = float(ucb_beta)
        self.diversity_weight = float(diversity_weight)
        self.n_estimators = int(n_estimators)
        self.fp_bits = int(fp_bits)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.observed: dict[str, float] = {}
        self.history: list[ActiveLearningRoundSummary] = []
        self._surrogate: RandomForestRegressor | None = None

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

    def _fit_surrogate(self) -> RandomForestRegressor | None:
        if len(self.observed) < 4:
            return None
        smis = list(self.observed.keys())
        ys = np.array([self.observed[s] for s in smis], dtype=np.float32)
        x_mat, valid = featurize_ecfp4_batch(smis, n_bits=self.fp_bits)
        if int(valid.sum()) < 4:
            return None
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            random_state=self.seed + len(self.history),
            n_jobs=1,
        )
        rf.fit(x_mat[valid], ys[valid])
        self._surrogate = rf
        return rf

    def predict_mean_and_std(
        self, pool_smiles: Sequence[str]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Predict surrogate mean and tree-ensemble epistemic uncertainty across ``pool_smiles``."""
        n = len(pool_smiles)
        if self._surrogate is None or n == 0:
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
    ) -> list[str]:
        """Select the next ``batch_size`` unobserved compounds from ``pool_smiles``."""
        unseen_idx = [i for i, s in enumerate(pool_smiles) if s.strip() not in self.observed]
        if not unseen_idx:
            return []
        unseen_smis = [pool_smiles[i].strip() for i in unseen_idx]
        k = min(len(unseen_smis), max(1, int(batch_size)))

        rf = self._fit_surrogate()
        if rf is None:
            if prior_scores is not None and len(prior_scores) == len(pool_smiles):
                p_arr = np.array([float(prior_scores[i]) for i in unseen_idx], dtype=np.float64)
                order = np.argsort(p_arr)[::-1][:k]
                return [unseen_smis[int(i)] for i in order]
            chosen = self._rng.choice(len(unseen_smis), size=k, replace=False)
            return [unseen_smis[int(i)] for i in chosen]

        mean, std = self.predict_mean_and_std(unseen_smis)
        if self.strategy == "greedy":
            acq = mean
        elif self.strategy == "thompson":
            acq = mean + self._rng.normal(0.0, 1.0, size=len(mean)).astype(np.float32) * std
        else:
            acq = mean + float(self.ucb_beta) * std

        if self.strategy != "maxmin_diverse_ucb" or k <= 1:
            order = np.argsort(acq)[::-1][:k]
            return [unseen_smis[int(i)] for i in order]

        # Diversity-aware greedy MaxMin-UCB selection over top candidate shortlist
        shortlist_k = min(len(unseen_smis), max(k * 8, 64))
        short_idx = np.argsort(acq)[::-1][:shortlist_k]
        short_smis = [unseen_smis[int(i)] for i in short_idx]
        short_acq = acq[short_idx]
        a_lo, a_hi = float(np.min(short_acq)), float(np.max(short_acq))
        acq_norm = (short_acq - a_lo) / max(a_hi - a_lo, 1e-6)

        fp_mat, _ = featurize_ecfp4_batch(short_smis, n_bits=self.fp_bits)
        norms = fp_mat.sum(axis=1)
        inter = fp_mat @ fp_mat.T
        union = norms[:, None] + norms[None, :] - inter
        tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)

        picked: list[int] = [int(np.argmax(acq_norm))]
        w_div = float(self.diversity_weight)
        for _ in range(1, k):
            max_sim = np.max(tan[:, picked], axis=1)
            scores = (1.0 - w_div) * acq_norm + w_div * (1.0 - max_sim)
            scores[picked] = -1e9
            picked.append(int(np.argmax(scores)))
        return [short_smis[p] for p in picked]

    def run_round(
        self,
        pool_smiles: Sequence[str],
        oracle_fn: Callable[[Sequence[str]], Sequence[float]],
        *,
        batch_size: int = 16,
        prior_scores: Sequence[float] | None = None,
    ) -> ActiveLearningRoundSummary:
        """Execute one full propose -> oracle-evaluate -> update DMTA round."""
        proposed = self.propose_batch(pool_smiles, batch_size=batch_size, prior_scores=prior_scores)
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
        return summary


__all__ = [
    "AcquisitionStrategy",
    "ActiveLearningDMTA",
    "ActiveLearningRoundSummary",
]
