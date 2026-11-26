"""Cold-start seeding, uncertainty calibration diagnostics, and stopping advisors.

Provides production-grade DMTA loop diagnostics:

1. **Cold-start seeding** (:func:`select_cold_start`, :class:`ColdStartSelector`):
   - ``"butina_medoids"``: Butina (1999) Tanimoto sphere-exclusion clustering picking true
     medoids of the largest distinct scaffold/chemical clusters (with chunked leader-follower
     scaling for large pools ``N > 2500``).
   - ``"maxmin"``: Greedy MaxMin Tanimoto dissimilarity seeding starting from the pool medoid.
   - ``"random"``: Reproducible seeded uniform baseline.
2. **Uncertainty calibration & AD-stratified evaluation** (:func:`evaluate_calibration`,
   alias :func:`evaluate_uq_calibration`):
   - Computes RMSE, MAE, :math:`R^2`, Spearman :math:`\\rho`, Gaussian NLL, quantile ECE,
     error-vs-std rank correlation :math:`\\rho_s(|y-\\hat{\\mu}|, \\hat{\\sigma})`,
     conformal coverage/width, AND similarity-stratified coverage across three ECFP4
     Applicability Domain bins (``< 0.35`` novel scaffold, ``0.35-0.55`` moderate shift,
     ``>= 0.55`` in-domain).
3. **Principled stopping diagnostics** (:class:`StoppingAdvisor`, :class:`StoppingReport`):
   - Combines EI saturation, Top-:math:`K` prediction Jaccard stability, and incumbent
     improvement plateau detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import spearmanr

from geneuclidean.active_learning.featurizers import (
    featurize_pool,
    tanimoto_kernel,
    tanimoto_max_similarity,
)

ColdStartStrategy = Literal["butina_medoids", "maxmin", "random"]


def compute_batch_internal_tanimoto(
    smiles: Sequence[str],
    *,
    n_bits: int = 1024,
    radius: int = 2,
) -> float:
    """Return mean pairwise off-diagonal ECFP4 Tanimoto similarity within a batch.

    Lower values indicate higher structural diversity within the proposed DMTA batch.
    Returns ``0.0`` when fewer than 2 valid molecules are provided.
    """
    if len(smiles) < 2:
        return 0.0
    fps, valid = featurize_pool(smiles, kind="ecfp4_binary", n_bits=n_bits, radius=radius)
    xv = fps[valid]
    m = int(xv.shape[0])
    if m < 2:
        return 0.0
    k_mat = tanimoto_kernel(xv)
    triu = k_mat[np.triu_indices(m, k=1)]
    return float(np.mean(triu))


def _greedy_maxmin_select(
    fps: NDArray[np.float64],
    k: int,
    *,
    already_picked: Sequence[int] = (),
    seed: int = 42,
) -> list[int]:
    """Exact greedy MaxMin Tanimoto dissimilarity picker on dense fingerprint rows."""
    n = int(fps.shape[0])
    target_k = min(n, max(0, int(k)))
    if target_k == 0:
        return []

    rng = np.random.default_rng(seed)
    picked: list[int] = [int(i) for i in already_picked]
    taken = np.zeros(n, dtype=np.bool_)
    if picked:
        taken[picked] = True

    if not picked:
        # Start from the exact pool medoid (highest average similarity to all pool molecules)
        if n <= 2000:
            sim_full = tanimoto_kernel(fps)
            avg_sim = np.mean(sim_full, axis=1)
        else:
            ref_idx = rng.choice(n, size=min(512, n), replace=False)
            avg_sim = np.mean(tanimoto_kernel(fps, fps[ref_idx]), axis=1)
        # Tiny seeded noise only breaks exact floating-point ties deterministically
        tie_eps = rng.uniform(0.0, 1e-12, size=n)
        first = int(np.argmax(avg_sim + tie_eps))
        picked.append(first)
        taken[first] = True

    # Initialize max similarity to already-picked anchors
    max_sim = np.max(tanimoto_kernel(fps, fps[picked]), axis=1)
    while len(picked) < target_k:
        dist = 1.0 - max_sim
        dist[taken] = -1.0
        # Deterministic tie-breaking
        tie_eps = rng.uniform(0.0, 1e-12, size=n)
        cand = int(np.argmax(np.where(taken, -1.0, dist + tie_eps)))
        picked.append(cand)
        taken[cand] = True
        sim_new = tanimoto_kernel(fps, fps[cand : cand + 1]).ravel()
        max_sim = np.maximum(max_sim, sim_new)

    return picked[:target_k]


def _butina_medoids_select(
    fps: NDArray[np.float64],
    k: int,
    *,
    tanimoto_cutoff: float = 0.45,
    seed: int = 42,
) -> list[int]:
    """Butina (1999) sphere-exclusion clustering picking true medoids of largest clusters."""
    n = int(fps.shape[0])
    target_k = min(n, max(0, int(k)))
    if target_k == 0:
        return []

    if n <= 2500:
        sim = tanimoto_kernel(fps)
        nbrs = [np.flatnonzero(sim[i] >= tanimoto_cutoff) for i in range(n)]
        order = sorted(range(n), key=lambda i: (-len(nbrs[i]), i))
        assigned = np.zeros(n, dtype=np.bool_)
        clusters: list[NDArray[np.intp]] = []
        for idx in order:
            if assigned[idx]:
                continue
            members = nbrs[idx][~assigned[nbrs[idx]]]
            if len(members) == 0:
                continue
            assigned[members] = True
            clusters.append(members.astype(np.intp))
        clusters.sort(key=lambda c: (-len(c), int(c[0])))

        medoids: list[int] = []
        for cl in clusters[:target_k]:
            if len(cl) == 1:
                medoids.append(int(cl[0]))
            else:
                sub = sim[np.ix_(cl, cl)]
                best_local = int(np.argmax(np.mean(sub, axis=1)))
                medoids.append(int(cl[best_local]))
    else:
        # Scalable chunked leader-follower clustering for large pools N > 2500
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        leaders: list[int] = []
        leader_counts: list[int] = []
        for idx_np in perm:
            idx = int(idx_np)
            if not leaders:
                leaders.append(idx)
                leader_counts.append(1)
                continue
            sims = tanimoto_kernel(fps[idx : idx + 1], fps[leaders]).ravel()
            best_l = int(np.argmax(sims))
            if float(sims[best_l]) >= tanimoto_cutoff:
                leader_counts[best_l] += 1
            else:
                leaders.append(idx)
                leader_counts.append(1)
        ranked = sorted(range(len(leaders)), key=lambda i: (-leader_counts[i], leaders[i]))
        medoids = [leaders[i] for i in ranked[:target_k]]

    if len(medoids) < target_k:
        return _greedy_maxmin_select(fps, target_k, already_picked=medoids, seed=seed)
    return medoids[:target_k]


def select_cold_start(
    pool_smiles: Sequence[str],
    n_select: int,
    *,
    strategy: ColdStartStrategy = "butina_medoids",
    tanimoto_cutoff: float = 0.45,
    n_bits: int = 1024,
    radius: int = 2,
    seed: int = 42,
) -> list[int]:
    """Select an initial Round-0 DMTA seed batch from ``pool_smiles``.

    Returns unique valid indices into ``pool_smiles``.
    """
    n = len(pool_smiles)
    if n == 0 or n_select <= 0:
        return []
    fps, valid = featurize_pool(pool_smiles, kind="ecfp4_binary", n_bits=n_bits, radius=radius)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return []
    k = min(len(valid_idx), int(n_select))

    if strategy == "random":
        rng = np.random.default_rng(seed)
        chosen_local = rng.choice(len(valid_idx), size=k, replace=False).tolist()
        return [int(valid_idx[int(i)]) for i in chosen_local]

    xv = fps[valid_idx]
    if strategy == "maxmin":
        picked_local = _greedy_maxmin_select(xv, k, seed=seed)
        return [int(valid_idx[i]) for i in picked_local]
    if strategy == "butina_medoids":
        picked_local = _butina_medoids_select(xv, k, tanimoto_cutoff=tanimoto_cutoff, seed=seed)
        return [int(valid_idx[i]) for i in picked_local]
    raise ValueError(
        f"Unknown cold-start strategy {strategy!r}; "
        "expected 'butina_medoids', 'maxmin', or 'random'."
    )


class ColdStartSelector:
    """Configurable cold-start batch selector for Round-0 DMTA initialization."""

    def __init__(
        self,
        *,
        strategy: ColdStartStrategy = "butina_medoids",
        tanimoto_cutoff: float = 0.45,
        n_bits: int = 1024,
        radius: int = 2,
        seed: int = 42,
    ) -> None:
        self.strategy: ColdStartStrategy = strategy
        self.tanimoto_cutoff = float(tanimoto_cutoff)
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        self.seed = int(seed)

    def select(self, pool_smiles: Sequence[str], n_select: int) -> list[int]:
        """Return selected Round-0 indices into ``pool_smiles``."""
        return select_cold_start(
            pool_smiles,
            n_select,
            strategy=self.strategy,
            tanimoto_cutoff=self.tanimoto_cutoff,
            n_bits=self.n_bits,
            radius=self.radius,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Uncertainty Calibration & Applicability-Domain Stratified Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationMetrics:
    """Comprehensive predictive accuracy, Gaussian UQ, and conformal coverage metrics."""

    n_eval: int
    rmse: float
    mae: float
    r2: float
    spearman_rho: float
    gaussian_nll: float
    ece_regression: float
    error_std_spearman: float
    conformal_empirical_coverage: float | None = None
    conformal_mean_width: float | None = None
    conformal_target_coverage: float | None = None
    similarity_stratified_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _r(v: float | None) -> float | None:
            if v is None or not math.isfinite(v):
                return None
            return round(float(v), 4)

        return {
            "n_eval": self.n_eval,
            "rmse": _r(self.rmse),
            "mae": _r(self.mae),
            "r2": _r(self.r2),
            "spearman_rho": _r(self.spearman_rho),
            "gaussian_nll": _r(self.gaussian_nll),
            "ece_regression": _r(self.ece_regression),
            "error_std_spearman": _r(self.error_std_spearman),
            "conformal_empirical_coverage": _r(self.conformal_empirical_coverage),
            "conformal_mean_width": _r(self.conformal_mean_width),
            "conformal_target_coverage": _r(self.conformal_target_coverage),
            "similarity_stratified_coverage": self.similarity_stratified_coverage,
        }


def evaluate_calibration(
    y_true: Sequence[float] | NDArray[np.float64],
    mean: Sequence[float] | NDArray[np.float64],
    std: Sequence[float] | NDArray[np.float64],
    *,
    interval_lo: Sequence[float] | NDArray[np.float64] | None = None,
    interval_hi: Sequence[float] | NDArray[np.float64] | None = None,
    alpha: float = 0.10,
    query_smiles: Sequence[str] | None = None,
    train_smiles: Sequence[str] | None = None,
    max_tanimoto: Sequence[float] | NDArray[np.float64] | None = None,
) -> CalibrationMetrics:
    """Evaluate regression accuracy, Gaussian NLL/ECE, error-std rank correlation & AD coverage."""
    yt = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    if not (len(yt) == len(mu) == len(sd)):
        raise ValueError("y_true, mean, and std must have equal length.")

    ok = np.isfinite(yt) & np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return CalibrationMetrics(
            n_eval=0,
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            spearman_rho=float("nan"),
            gaussian_nll=float("nan"),
            ece_regression=float("nan"),
            error_std_spearman=float("nan"),
        )

    y_v = yt[ok]
    m_v = mu[ok]
    s_v = np.maximum(sd[ok], 1e-6)
    err = np.abs(y_v - m_v)

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((y_v - m_v) ** 2)))
    ss_res = float(np.sum((y_v - m_v) ** 2))
    ss_tot = float(np.sum((y_v - float(np.mean(y_v))) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if n_ok >= 2 else float("nan")

    rho = (
        float(spearmanr(y_v, m_v).statistic)
        if n_ok >= 3 and float(np.std(y_v)) > 0 and float(np.std(m_v)) > 0
        else float("nan")
    )
    err_std_rho = (
        float(spearmanr(err, s_v).statistic)
        if n_ok >= 3 and float(np.std(err)) > 0 and float(np.std(s_v)) > 0
        else float("nan")
    )

    nll = float(0.5 * np.mean(np.log(2.0 * math.pi * s_v * s_v) + ((y_v - m_v) ** 2) / (s_v * s_v)))

    # Regression Expected Calibration Error across symmetric Gaussian confidence levels
    nom_levels = (0.50, 0.68, 0.80, 0.90, 0.95)
    ece_gaps: list[float] = []
    for p_nom in nom_levels:
        z_crit = float(ndtri(0.5 + 0.5 * p_nom))
        emp_cov = float(np.mean(err <= z_crit * s_v))
        ece_gaps.append(abs(emp_cov - p_nom))
    ece_reg = float(np.mean(ece_gaps))

    # Prediction intervals (explicit conformal intervals, or Gaussian (1-alpha) fallback)
    has_explicit_intervals = interval_lo is not None and interval_hi is not None
    if has_explicit_intervals:
        lo_arr = np.asarray(interval_lo, dtype=np.float64)[ok]
        hi_arr = np.asarray(interval_hi, dtype=np.float64)[ok]
    else:
        z_alpha = float(ndtri(1.0 - 0.5 * float(alpha)))
        lo_arr = m_v - z_alpha * s_v
        hi_arr = m_v + z_alpha * s_v

    int_ok = np.isfinite(lo_arr) & np.isfinite(hi_arr)
    conf_cov: float | None = None
    conf_width: float | None = None
    if bool(int_ok.any()):
        covered = (y_v[int_ok] >= lo_arr[int_ok]) & (y_v[int_ok] <= hi_arr[int_ok])
        conf_cov = float(np.mean(covered))
        conf_width = float(np.mean(hi_arr[int_ok] - lo_arr[int_ok]))

    # Similarity-stratified coverage across ECFP4 Applicability Domain buckets
    strat_cov: dict[str, dict[str, Any]] = {}
    sim_vec: NDArray[np.float64] | None = None
    if max_tanimoto is not None:
        sim_vec = np.asarray(max_tanimoto, dtype=np.float64)[ok]
    elif query_smiles is not None and train_smiles is not None and len(train_smiles) > 0:
        q_fps, q_val = featurize_pool(query_smiles, kind="ecfp4_binary", n_bits=1024, radius=2)
        t_fps, t_val = featurize_pool(train_smiles, kind="ecfp4_binary", n_bits=1024, radius=2)
        if bool(t_val.any()):
            sims_all = np.zeros(len(query_smiles), dtype=np.float64)
            if bool(q_val.any()):
                sims_all[q_val] = tanimoto_max_similarity(q_fps[q_val], t_fps[t_val])
            sim_vec = sims_all[ok]

    if sim_vec is not None and bool(int_ok.any()):
        buckets: tuple[tuple[str, NDArray[np.bool_]], ...] = (
            ("lt_0.35", (sim_vec < 0.35) & int_ok),
            ("0.35_to_0.55", (sim_vec >= 0.35) & (sim_vec < 0.55) & int_ok),
            ("ge_0.55", (sim_vec >= 0.55) & int_ok),
        )
        for label, bmask in buckets:
            bn = int(bmask.sum())
            if bn == 0:
                strat_cov[label] = {
                    "n": 0,
                    "coverage": None,
                    "mean_width": None,
                    "mae": None,
                    "rmse": None,
                }
            else:
                b_cov = float(
                    np.mean((y_v[bmask] >= lo_arr[bmask]) & (y_v[bmask] <= hi_arr[bmask]))
                )
                b_w = float(np.mean(hi_arr[bmask] - lo_arr[bmask]))
                b_mae = float(np.mean(np.abs(y_v[bmask] - m_v[bmask])))
                b_rmse = float(np.sqrt(np.mean((y_v[bmask] - m_v[bmask]) ** 2)))
                strat_cov[label] = {
                    "n": bn,
                    "coverage": round(b_cov, 4),
                    "mean_width": round(b_w, 4),
                    "mae": round(b_mae, 4),
                    "rmse": round(b_rmse, 4),
                }

    return CalibrationMetrics(
        n_eval=n_ok,
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman_rho=rho,
        gaussian_nll=nll,
        ece_regression=ece_reg,
        error_std_spearman=err_std_rho,
        conformal_empirical_coverage=conf_cov,
        conformal_mean_width=conf_width,
        conformal_target_coverage=round(1.0 - float(alpha), 4),
        similarity_stratified_coverage=strat_cov,
    )


evaluate_uq_calibration = evaluate_calibration


# ---------------------------------------------------------------------------
# Principled Stopping Advisor (EI Saturation + Top-K Jaccard + Plateau)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoppingReport:
    """Multi-signal convergence telemetry emitted at the end of each DMTA round."""

    should_stop: bool
    triggered_reasons: tuple[str, ...]
    max_rel_ei: float
    topk_jaccard: float | None
    rounds_since_improvement: int
    round_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "triggered_reasons": list(self.triggered_reasons),
            "max_rel_ei": round(self.max_rel_ei, 6) if math.isfinite(self.max_rel_ei) else None,
            "topk_jaccard": (
                round(self.topk_jaccard, 4)
                if self.topk_jaccard is not None and math.isfinite(self.topk_jaccard)
                else None
            ),
            "rounds_since_improvement": self.rounds_since_improvement,
            "round_index": self.round_index,
        }


class StoppingAdvisor:
    """Evaluates three independent DMTA convergence signals across rounds.

    Triggers ``should_stop = True`` when any configured criterion persists for
    ``patience`` consecutive rounds:
    1. ``ei_saturated``: relative max Expected Improvement ``max(EI) / std(y_obs) < ei_threshold``.
    2. ``rank_stable``: Top-``top_k`` predicted pool Jaccard overlap ``>= jaccard_threshold``.
    3. ``improvement_plateau``: best observed score (or Pareto HV) improved by less than
       ``improvement_tol`` for ``patience`` consecutive rounds.
    """

    def __init__(
        self,
        *,
        ei_threshold: float = 0.01,
        jaccard_threshold: float = 0.85,
        top_k: int = 25,
        improvement_tol: float = 1e-3,
        patience: int = 3,
    ) -> None:
        self.ei_threshold = float(ei_threshold)
        self.jaccard_threshold = float(jaccard_threshold)
        self.top_k = max(1, int(top_k))
        self.improvement_tol = float(improvement_tol)
        self.patience = max(1, int(patience))

        self._round_count: int = 0
        self._best_metric: float = float("-inf")
        self._rounds_since_improvement: int = 0
        self._ei_sat_streak: int = 0
        self._rank_stable_streak: int = 0
        self._prev_topk: set[str] | None = None

    def update(
        self,
        *,
        best_metric: float,
        max_ei: float,
        observed_std: float,
        pool_smiles: Sequence[str] | None = None,
        pool_pred_mean: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> StoppingReport:
        """Record end-of-round telemetry and return a :class:`StoppingReport`."""
        self._round_count += 1
        if math.isfinite(best_metric):
            if not math.isfinite(self._best_metric) or (
                best_metric > self._best_metric + self.improvement_tol
            ):
                self._best_metric = float(best_metric)
                self._rounds_since_improvement = 0
            else:
                self._rounds_since_improvement += 1

        scale = max(float(observed_std), 1e-6) if math.isfinite(observed_std) else 1.0
        max_rel_ei = float(max(0.0, max_ei) / scale) if math.isfinite(max_ei) else float("inf")
        if max_rel_ei < self.ei_threshold:
            self._ei_sat_streak += 1
        else:
            self._ei_sat_streak = 0

        topk_jaccard: float | None = None
        if pool_smiles is not None and pool_pred_mean is not None and len(pool_smiles) > 0:
            mu = np.asarray(pool_pred_mean, dtype=np.float64)
            fin = np.flatnonzero(np.isfinite(mu))
            if len(fin) > 0:
                k = min(self.top_k, len(fin))
                top_idx = fin[np.argsort(-mu[fin], kind="mergesort")[:k]]
                cur_topk = {pool_smiles[int(i)] for i in top_idx}
                if self._prev_topk is not None:
                    union = len(cur_topk | self._prev_topk)
                    topk_jaccard = float(len(cur_topk & self._prev_topk) / max(union, 1))
                    if topk_jaccard >= self.jaccard_threshold:
                        self._rank_stable_streak += 1
                    else:
                        self._rank_stable_streak = 0
                self._prev_topk = cur_topk

        reasons: list[str] = []
        if self._ei_sat_streak >= self.patience:
            reasons.append("ei_saturated")
        if self._rank_stable_streak >= self.patience:
            reasons.append("rank_stable")
        if self._rounds_since_improvement >= self.patience:
            reasons.append("improvement_plateau")

        return StoppingReport(
            should_stop=len(reasons) > 0,
            triggered_reasons=tuple(reasons),
            max_rel_ei=max_rel_ei,
            topk_jaccard=topk_jaccard,
            rounds_since_improvement=self._rounds_since_improvement,
            round_index=self._round_count,
        )


__all__ = [
    "CalibrationMetrics",
    "ColdStartSelector",
    "ColdStartStrategy",
    "StoppingAdvisor",
    "StoppingReport",
    "compute_batch_internal_tanimoto",
    "evaluate_calibration",
    "evaluate_uq_calibration",
    "select_cold_start",
]
