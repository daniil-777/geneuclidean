"""Acquisition functions for single-objective, batch-diverse, Pareto & constrained AL.

Provides :data:`ACQUISITION_REGISTRY` with 14 zero-PyTorch acquisition strategies:

- **Pointwise single-objective:** ``"greedy"``, ``"uncertainty"``, ``"ucb"``, ``"pi"``,
  ``"ei"``, ``"log_ei"`` (Ament et al., NeurIPS 2023 tail-stabilized asymptotic Mills
  expansion preventing underflow tie-collapse), ``"thompson"`` (correlated joint GP
  posterior Cholesky sampling with ensemble fallback).
- **Batch diversity & async decorrelation:** ``"maxmin_ucb"`` (alias
  ``"maxmin_diverse_ucb"``), ``"kriging_believer"`` (exact :math:`O(M^2)` Schur-complement
  fantasy conditioning with ``kb_mean`` / ``cl_min`` / ``cl_mean`` / ``cl_pessimistic``
  modes and async ``pending_smiles`` support), ``"dpp_ucb"`` (Chen et al., NeurIPS 2018
  fast greedy MAP Determinantal Point Process in :math:`O(B^2 M)`).
- **Multi-objective Pareto:** ``"qehvi"`` (exact analytical 2D EHVI + sequential greedy
  Sobol-QMC ``qEHVI`` for batches & :math:`M \\ge 2`), ``"pareto_scalarized_ucb"``
  (ParEGO augmented Chebyshev scalarization across :math:`M \\ge 2` objectives).
- **Feasibility-constrained:** ``"constrained_ei"`` (with Phase-1 feasibility recovery
  when zero feasible compounds have been observed) and ``"avoidome_feasibility_ucb"``
  (sign-inversion-safe shifted utility weighted by :func:`build_feasibility_vector`,
  integrating ``CNSAdmetSuite``, ``PROPERTY_PROFILE_REGISTRY``, ``PoseRealityAuditor``,
  and ``triage_molecule``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, eigh
from scipy.special import erfcx, expit, log_ndtr, ndtr
from scipy.stats import qmc

from geneuclidean.active_learning.featurizers import (
    featurize_pool,
    tanimoto_kernel,
    tanimoto_max_similarity,
)
from geneuclidean.active_learning.surrogates import (
    ConformalSurrogate,
    SurrogateProtocol,
    TanimotoGP,
)
from geneuclidean.registry import Registry

if TYPE_CHECKING:
    from geneuclidean.audit.pose_auditor import PoseAuditReport
    from geneuclidean.cns.admet import CNSAdmetSuite

LiarMode = Literal["kb_mean", "cl_min", "cl_mean", "cl_pessimistic"]
KBBaseAcq = Literal["ucb", "ei", "log_ei"]

_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)
_INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class AcquisitionContext:
    """Immutable input bundle passed to every registered acquisition strategy."""

    pool_smiles: tuple[str, ...]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    best_observed: float = float("-inf")
    batch_size: int = 16
    beta: float = 1.5
    xi: float = 0.01
    diversity_weight: float = 0.25
    dpp_quality_temp: float = 1.0
    shortlist_multiplier: int = 8
    surrogate: SurrogateProtocol | None = None
    observed_objectives: NDArray[np.float64] | None = None
    reference_point: NDArray[np.float64] | None = None
    objective_weights: NDArray[np.float64] | None = None
    feasibility_probs: NDArray[np.float64] | None = None
    pending_smiles: tuple[str, ...] = ()
    liar_mode: LiarMode = "kb_mean"
    kb_base: KBBaseAcq = "ucb"
    qmc_samples: int = 256
    seed: int = 42


@dataclass(frozen=True)
class AcquisitionSelection:
    """Structured output returned by every acquisition function."""

    indices: tuple[int, ...]
    smiles: tuple[str, ...]
    scores: tuple[float, ...]
    all_scores: NDArray[np.float64]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


AcquisitionFn = Callable[[AcquisitionContext], AcquisitionSelection]


# ---------------------------------------------------------------------------
# Analytical Gaussian Acquisition Primitives (Tail-Stabilized)
# ---------------------------------------------------------------------------


def _norm_pdf(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * z * z), dtype=np.float64)


def _norm_cdf(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(ndtr(z), dtype=np.float64)


def compute_log_h(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute :math:`\\log(z\\Phi(z) + \\phi(z))` stably across the entire real line.

    Uses direct evaluation for :math:`z > -1.0`, scaled complementary error function
    ``erfcx`` for :math:`-12.0 < z \\le -1.0`, and the 5-term asymptotic Mills-ratio
    expansion (Ament et al., NeurIPS 2023) for :math:`z \\le -12.0` (continuous to
    :math:`1.6 \\times 10^{-7}` at the transition and strictly monotonic for all :math:`z`).
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr, dtype=np.float64)
    pos_region = z_arr > -1.0
    mid_region = (z_arr <= -1.0) & (z_arr > -12.0)
    deep_region = z_arr <= -12.0

    if bool(np.any(pos_region)):
        zp = z_arr[pos_region]
        val = zp * _norm_cdf(zp) + _norm_pdf(zp)
        out[pos_region] = np.log(np.maximum(val, 1e-300))

    if bool(np.any(mid_region)):
        zm = z_arr[mid_region]
        um = -zm
        # Exact identity: h(z) = phi(z) * (1 - u * sqrt(pi/2) * erfcx(u / sqrt(2)))
        bracket = 1.0 - um * math.sqrt(0.5 * math.pi) * np.asarray(
            erfcx(um / math.sqrt(2.0)), dtype=np.float64
        )
        out[mid_region] = -0.5 * zm * zm - _LOG_SQRT_2PI + np.log(np.maximum(bracket, 1e-300))

    if bool(np.any(deep_region)):
        zd = z_arr[deep_region]
        u = -zd
        inv_u2 = 1.0 / (u * u)
        poly = 1.0 - 3.0 * inv_u2 + 15.0 * (inv_u2**2) - 105.0 * (inv_u2**3) + 945.0 * (inv_u2**4)
        out[deep_region] = (
            -0.5 * zd * zd - _LOG_SQRT_2PI - 2.0 * np.log(u) + np.log(np.maximum(poly, 0.5))
        )

    return out


def compute_expected_improvement(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    best_observed: float,
    *,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """Compute exact Expected Improvement :math:`\\text{EI}(x) \\ge 0`."""
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    out = np.full_like(mu, float("-inf"), dtype=np.float64)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    if not bool(fin.any()):
        return out

    delta = mu[fin] - float(best_observed) - float(xi)
    s_fin = sd[fin]
    ei = np.zeros_like(delta, dtype=np.float64)
    pos = s_fin > 1e-12
    if bool(pos.any()):
        z = delta[pos] / s_fin[pos]
        ei[pos] = np.maximum(0.0, s_fin[pos] * (z * _norm_cdf(z) + _norm_pdf(z)))
    zero_sd = ~pos
    if bool(zero_sd.any()):
        ei[zero_sd] = np.maximum(0.0, delta[zero_sd])
    out[fin] = ei
    return out


def compute_log_expected_improvement(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    best_observed: float,
    *,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """Compute numerically stable :math:`\\log\\text{EI}(x)` without negative-tail underflow."""
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    out = np.full_like(mu, float("-inf"), dtype=np.float64)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    if not bool(fin.any()):
        return out

    delta = mu[fin] - float(best_observed) - float(xi)
    s_fin = sd[fin]
    lei = np.full_like(delta, -1e30, dtype=np.float64)
    pos = s_fin > 1e-12
    if bool(pos.any()):
        z = delta[pos] / s_fin[pos]
        lei[pos] = np.log(s_fin[pos]) + compute_log_h(z)
    zero_sd = (~pos) & (delta > 0.0)
    if bool(zero_sd.any()):
        lei[zero_sd] = np.log(delta[zero_sd])
    out[fin] = lei
    return out


def compute_probability_of_improvement(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    best_observed: float,
    *,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """Compute exact Probability of Improvement :math:`\\text{PI}(x) \\in [0, 1]`."""
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    out = np.full_like(mu, float("-inf"), dtype=np.float64)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    if not bool(fin.any()):
        return out

    delta = mu[fin] - float(best_observed) - float(xi)
    s_fin = sd[fin]
    pi = np.zeros_like(delta, dtype=np.float64)
    pos = s_fin > 1e-12
    if bool(pos.any()):
        pi[pos] = _norm_cdf(delta[pos] / s_fin[pos])
    zero_sd = ~pos
    if bool(zero_sd.any()):
        pi[zero_sd] = np.where(delta[zero_sd] > 0.0, 1.0, 0.0)
    out[fin] = pi
    return out


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _as_1d_mean_std(ctx: AcquisitionContext) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu = np.asarray(ctx.mean, dtype=np.float64)
    sd = np.asarray(ctx.std, dtype=np.float64)
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu[:, 0]
        sd = sd[:, 0]
    if mu.ndim != 1 or sd.ndim != 1 or len(mu) != len(ctx.pool_smiles) or len(sd) != len(mu):
        raise ValueError(
            f"Single-objective acquisition expected 1D mean/std of len {len(ctx.pool_smiles)}, "
            f"got mean={mu.shape}, std={sd.shape}."
        )
    return mu, sd


def _select_top_k(
    ctx: AcquisitionContext,
    scores: NDArray[np.float64],
    strategy: str,
    *,
    tie_break: NDArray[np.float64] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AcquisitionSelection:
    n = len(ctx.pool_smiles)
    fin = np.isfinite(scores)
    n_valid = int(fin.sum())
    if n == 0 or n_valid == 0:
        return AcquisitionSelection(
            indices=(),
            smiles=(),
            scores=(),
            all_scores=scores,
            strategy=strategy,
            metadata=dict(metadata or {}),
        )
    k = min(n_valid, max(1, int(ctx.batch_size)))
    valid_idx = np.flatnonzero(fin)
    if tie_break is not None:
        # Lexicographic sort: primary descending scores, secondary descending tie_break
        tb = np.where(np.isfinite(tie_break[valid_idx]), tie_break[valid_idx], float("-inf"))
        order = np.lexsort((-tb, -scores[valid_idx]))[:k]
    else:
        order = np.argsort(-scores[valid_idx], kind="mergesort")[:k]
    chosen = [int(valid_idx[int(o)]) for o in order]
    return AcquisitionSelection(
        indices=tuple(chosen),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen),
        scores=tuple(float(scores[i]) for i in chosen),
        all_scores=scores,
        strategy=strategy,
        metadata=dict(metadata or {}),
    )


def _unwrap_tanimoto_gp(surrogate: SurrogateProtocol | None) -> TanimotoGP | None:
    if isinstance(surrogate, TanimotoGP):
        return surrogate
    if isinstance(surrogate, ConformalSurrogate) and isinstance(
        surrogate.base_surrogate, TanimotoGP
    ):
        return surrogate.base_surrogate
    return None


def _shortlist_indices(
    scores: NDArray[np.float64], batch_size: int, multiplier: int, min_shortlist: int = 128
) -> NDArray[np.intp]:
    fin = np.flatnonzero(np.isfinite(scores))
    if len(fin) == 0:
        return np.zeros(0, dtype=np.intp)
    m = min(len(fin), max(int(batch_size) * max(2, int(multiplier)), int(min_shortlist)))
    order = np.argsort(-scores[fin], kind="mergesort")[:m]
    return fin[order]


# ---------------------------------------------------------------------------
# 1. Pointwise Single-Objective Acquisitions
# ---------------------------------------------------------------------------


def acquire_greedy(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Pure exploitation: maximize posterior mean :math:`\\mu(x)`."""
    mu, sd = _as_1d_mean_std(ctx)
    scores = np.where(np.isfinite(mu) & np.isfinite(sd), mu, float("-inf"))
    return _select_top_k(ctx, scores, "greedy")


def acquire_uncertainty(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Pure exploration: maximize predictive uncertainty :math:`\\sigma(x)`."""
    mu, sd = _as_1d_mean_std(ctx)
    scores = np.where(np.isfinite(mu) & np.isfinite(sd), sd, float("-inf"))
    return _select_top_k(ctx, scores, "uncertainty")


def acquire_ucb(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Upper Confidence Bound: :math:`\\mu(x) + \\beta\\,\\sigma(x)`."""
    mu, sd = _as_1d_mean_std(ctx)
    scores = np.where(
        np.isfinite(mu) & np.isfinite(sd),
        mu + float(ctx.beta) * sd,
        float("-inf"),
    )
    return _select_top_k(ctx, scores, "ucb", metadata={"beta": float(ctx.beta)})


def acquire_pi(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Probability of Improvement with log-CDF tail tie-breaker."""
    mu, sd = _as_1d_mean_std(ctx)
    best = ctx.best_observed if math.isfinite(ctx.best_observed) else float(np.nanmax(mu))
    pi = compute_probability_of_improvement(mu, sd, best, xi=ctx.xi)
    # Log-CDF tie-breaker prevents flat 0.0 ties when all candidates are far below f*
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-12)
    log_pi = np.full_like(mu, float("-inf"))
    if bool(fin.any()):
        log_pi[fin] = log_ndtr((mu[fin] - best - float(ctx.xi)) / sd[fin])
    return _select_top_k(ctx, pi, "pi", tie_break=log_pi, metadata={"xi": float(ctx.xi)})


def acquire_ei(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Expected Improvement with ``log_ei`` tie-breaker when raw EI underflows."""
    mu, sd = _as_1d_mean_std(ctx)
    best = ctx.best_observed if math.isfinite(ctx.best_observed) else float(np.nanmax(mu))
    ei = compute_expected_improvement(mu, sd, best, xi=ctx.xi)
    log_ei = compute_log_expected_improvement(mu, sd, best, xi=ctx.xi)
    return _select_top_k(ctx, ei, "ei", tie_break=log_ei, metadata={"xi": float(ctx.xi)})


def acquire_log_ei(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Log Expected Improvement (Ament et al., NeurIPS 2023) for smooth deep-tail ranking."""
    mu, sd = _as_1d_mean_std(ctx)
    best = ctx.best_observed if math.isfinite(ctx.best_observed) else float(np.nanmax(mu))
    log_ei = compute_log_expected_improvement(mu, sd, best, xi=ctx.xi)
    return _select_top_k(ctx, log_ei, "log_ei", metadata={"xi": float(ctx.xi)})


def acquire_thompson(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Thompson Sampling (correlated joint GP posterior draws or independent ensemble draws)."""
    mu, sd = _as_1d_mean_std(ctx)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    n_valid = int(fin.sum())
    if n_valid == 0:
        return _select_top_k(ctx, np.full_like(mu, float("-inf")), "thompson")

    k = min(n_valid, max(1, int(ctx.batch_size)))
    rng = np.random.default_rng(ctx.seed)
    gp = _unwrap_tanimoto_gp(ctx.surrogate)

    # Correlated joint posterior sampling when backed by an exact fitted TanimotoGP
    if gp is not None and gp.is_fitted:
        ucb = np.where(fin, mu + float(ctx.beta) * sd, float("-inf"))
        short_idx = _shortlist_indices(ucb, k, ctx.shortlist_multiplier, min_shortlist=128)
        short_smis = [ctx.pool_smiles[int(i)] for i in short_idx]
        gp_eff = gp.condition_on_fantasies(ctx.pending_smiles) if ctx.pending_smiles else gp
        mu_s, cov_s, valid_s = gp_eff.predict_covariance(short_smis)
        ok_local = np.flatnonzero(valid_s & np.isfinite(mu_s))
        if len(ok_local) >= k:
            mu_ok = mu_s[ok_local]
            cov_ok = cov_s[np.ix_(ok_local, ok_local)]
            m_ok = len(ok_local)
            # Robust PSD square root via Cholesky with eigh fallback
            chol_mat: NDArray[np.float64] | None = None
            diag_scale = max(float(np.max(np.diag(cov_ok))), 1e-8)
            for eps in (1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    c_fac, _ = cho_factor(
                        cov_ok + (eps * diag_scale) * np.eye(m_ok), lower=True, check_finite=False
                    )
                    chol_mat = np.tril(np.asarray(c_fac, dtype=np.float64))
                    break
                except np.linalg.LinAlgError:
                    continue
            if chol_mat is None:
                vals, vecs = eigh(cov_ok, check_finite=False)
                chol_mat = np.asarray(
                    vecs * np.sqrt(np.maximum(vals, 0.0))[None, :], dtype=np.float64
                )

            picked_local: list[int] = []
            picked_scores: list[float] = []
            taken = np.zeros(m_ok, dtype=np.bool_)
            all_scores = np.full(len(ctx.pool_smiles), float("-inf"), dtype=np.float64)
            for b in range(k):
                draw = mu_ok + chol_mat @ rng.standard_normal(m_ok)
                if b == 0:
                    for loc_pos, orig_short in enumerate(ok_local):
                        all_scores[int(short_idx[int(orig_short)])] = float(draw[loc_pos])
                masked = np.where(taken, float("-inf"), draw)
                j_loc = int(np.argmax(masked))
                taken[j_loc] = True
                picked_local.append(j_loc)
                picked_scores.append(float(draw[j_loc]))

            chosen = [int(short_idx[int(ok_local[j])]) for j in picked_local]
            return AcquisitionSelection(
                indices=tuple(chosen),
                smiles=tuple(ctx.pool_smiles[i] for i in chosen),
                scores=tuple(picked_scores),
                all_scores=all_scores,
                strategy="thompson",
                metadata={"correlated_gp_draws": True, "shortlist_size": int(m_ok)},
            )

    # Independent Gaussian posterior draws (Ensemble / generic surrogate fallback)
    valid_idx = np.flatnonzero(fin)
    taken_v = np.zeros(len(valid_idx), dtype=np.bool_)
    chosen_global: list[int] = []
    chosen_scores: list[float] = []
    all_scores_ind = np.full(len(ctx.pool_smiles), float("-inf"), dtype=np.float64)
    for b in range(k):
        draw_v = mu[valid_idx] + sd[valid_idx] * rng.standard_normal(len(valid_idx))
        if b == 0:
            all_scores_ind[valid_idx] = draw_v
        masked_v = np.where(taken_v, float("-inf"), draw_v)
        j = int(np.argmax(masked_v))
        taken_v[j] = True
        chosen_global.append(int(valid_idx[j]))
        chosen_scores.append(float(draw_v[j]))

    return AcquisitionSelection(
        indices=tuple(chosen_global),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen_global),
        scores=tuple(chosen_scores),
        all_scores=all_scores_ind,
        strategy="thompson",
        metadata={"correlated_gp_draws": False},
    )


# ---------------------------------------------------------------------------
# 2. Batch Diversity & Sequential Decorrelation Acquisitions
# ---------------------------------------------------------------------------


def acquire_maxmin_ucb(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Greedy MaxMin-Diverse UCB balancing acquisition utility and ECFP4 dissimilarity."""
    mu, sd = _as_1d_mean_std(ctx)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    ucb = np.where(fin, mu + float(ctx.beta) * sd, float("-inf"))
    n_valid = int(fin.sum())
    if n_valid == 0:
        return _select_top_k(ctx, ucb, "maxmin_ucb")

    k = min(n_valid, max(1, int(ctx.batch_size)))
    if k == 1 and not ctx.pending_smiles:
        return _select_top_k(
            ctx, ucb, "maxmin_ucb", metadata={"diversity_weight": ctx.diversity_weight}
        )

    short_idx = _shortlist_indices(ucb, k, ctx.shortlist_multiplier, min_shortlist=64)
    short_smis = [ctx.pool_smiles[int(i)] for i in short_idx]
    short_ucb = ucb[short_idx]
    u_lo, u_hi = float(np.min(short_ucb)), float(np.max(short_ucb))
    u_norm = (short_ucb - u_lo) / max(u_hi - u_lo, 1e-9)

    fps, _ = featurize_pool(short_smis, kind="ecfp4_binary", n_bits=1024, radius=2)
    sim_mat = tanimoto_kernel(fps)

    w_div = float(np.clip(ctx.diversity_weight, 0.0, 1.0))
    max_sim = np.zeros(len(short_idx), dtype=np.float64)
    if ctx.pending_smiles:
        p_fps, p_val = featurize_pool(
            ctx.pending_smiles, kind="ecfp4_binary", n_bits=1024, radius=2
        )
        if bool(p_val.any()):
            max_sim = tanimoto_max_similarity(fps, p_fps[p_val])

    picked_local: list[int] = []
    picked_scores: list[float] = []
    taken = np.zeros(len(short_idx), dtype=np.bool_)

    for step in range(k):
        if step == 0 and not ctx.pending_smiles:
            cand_scores = u_norm.copy()
        else:
            cand_scores = (1.0 - w_div) * u_norm + w_div * (1.0 - max_sim)
        cand_scores[taken] = float("-inf")
        j = int(np.argmax(cand_scores))
        taken[j] = True
        picked_local.append(j)
        picked_scores.append(float(cand_scores[j]))
        max_sim = np.maximum(max_sim, sim_mat[:, j])

    chosen = [int(short_idx[j]) for j in picked_local]
    return AcquisitionSelection(
        indices=tuple(chosen),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen),
        scores=tuple(picked_scores),
        all_scores=ucb,
        strategy="maxmin_ucb",
        metadata={
            "beta": float(ctx.beta),
            "diversity_weight": w_div,
            "shortlist_size": len(short_idx),
            "n_pending_anchored": len(ctx.pending_smiles),
        },
    )


def _kb_step_scores(
    mu: NDArray[np.float64],
    std: NDArray[np.float64],
    *,
    base: KBBaseAcq,
    beta: float,
    best_observed: float,
    xi: float,
) -> NDArray[np.float64]:
    if base == "ei":
        best = best_observed if math.isfinite(best_observed) else float(np.nanmax(mu))
        ei = compute_expected_improvement(mu, std, best, xi=xi)
        if float(np.nanmax(ei)) <= 1e-12:
            return compute_log_expected_improvement(mu, std, best, xi=xi)
        return ei
    if base == "log_ei":
        best = best_observed if math.isfinite(best_observed) else float(np.nanmax(mu))
        return compute_log_expected_improvement(mu, std, best, xi=xi)
    return mu + float(beta) * std


def acquire_kriging_believer(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Sequential Kriging Believer / Constant Liar batch acquisition with pending conditioning.

    When backed by :class:`TanimotoGP`, computes the exact :math:`M \\times M` posterior
    covariance matrix on the top-:math:`M` shortlist (pre-conditioned on any in-flight
    ``pending_smiles``) and applies exact :math:`O(M^2)` rank-1 Schur-complement Gaussian
    updates after each pick, provably deflating variance at the chosen molecule and all
    structurally correlated analogs.
    """
    mu, sd = _as_1d_mean_std(ctx)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    base_all = np.where(fin, mu + float(ctx.beta) * sd, float("-inf"))
    n_valid = int(fin.sum())
    if n_valid == 0:
        return _select_top_k(ctx, base_all, "kriging_believer")

    k = min(n_valid, max(1, int(ctx.batch_size)))
    gp = _unwrap_tanimoto_gp(ctx.surrogate)

    if gp is not None and gp.is_fitted:
        gp_eff = gp.condition_on_fantasies(ctx.pending_smiles) if ctx.pending_smiles else gp
        if ctx.pending_smiles:
            pred_eff = gp_eff.predict(ctx.pool_smiles)
            eff_scores = np.where(
                pred_eff.valid & np.isfinite(pred_eff.mean),
                pred_eff.mean + float(ctx.beta) * pred_eff.epistemic_std,
                float("-inf"),
            )
        else:
            eff_scores = base_all

        short_idx = _shortlist_indices(eff_scores, k, ctx.shortlist_multiplier, min_shortlist=128)
        short_smis = [ctx.pool_smiles[int(i)] for i in short_idx]
        mu_cur, cov_cur, valid_s = gp_eff.predict_covariance(short_smis)
        mu_cur = mu_cur.copy()
        cov_cur = cov_cur.copy()
        noise_var = max(gp_eff.noise_variance_raw, 1e-8)

        # Determine constant-liar target value if requested
        valid_mu = mu_cur[valid_s & np.isfinite(mu_cur)]
        if len(valid_mu) == 0:
            return _select_top_k(ctx, base_all, "kriging_believer")

        init_epi = np.sqrt(np.maximum(np.diag(cov_cur), 0.0))
        cl_target: float | None = None
        if ctx.liar_mode == "cl_min":
            cl_target = float(np.min(valid_mu))
        elif ctx.liar_mode == "cl_mean":
            cl_target = float(np.mean(valid_mu))
        elif ctx.liar_mode == "cl_pessimistic":
            cl_target = float(np.min(mu_cur[valid_s] - init_epi[valid_s]))

        taken = ~valid_s
        picked_local: list[int] = []
        picked_scores: list[float] = []
        best_obs = ctx.best_observed
        initial_mean_var = float(np.mean(np.maximum(np.diag(cov_cur)[valid_s], 0.0)))

        for _ in range(k):
            epi_std = np.sqrt(np.maximum(np.diag(cov_cur), 0.0))
            step_sc = _kb_step_scores(
                mu_cur,
                epi_std,
                base=ctx.kb_base,
                beta=ctx.beta,
                best_observed=best_obs,
                xi=ctx.xi,
            )
            step_sc[taken] = float("-inf")
            if not bool(np.any(np.isfinite(step_sc))):
                break
            j = int(np.argmax(step_sc))
            taken[j] = True
            picked_local.append(j)
            picked_scores.append(float(step_sc[j]))

            # Exact O(M^2) rank-1 Gaussian Schur-complement conditioning
            v = cov_cur[:, j].copy()
            d = float(cov_cur[j, j]) + noise_var
            if d > 1e-12:
                if cl_target is not None:
                    delta = cl_target - float(mu_cur[j])
                    mu_cur = mu_cur + (delta / d) * v
                    best_obs = max(best_obs, cl_target) if math.isfinite(best_obs) else cl_target
                else:
                    # Pure Kriging Believer: fantasy_y = mu_cur[j] -> delta = 0, mean unchanged
                    if math.isfinite(best_obs):
                        best_obs = max(best_obs, float(mu_cur[j]))
                cov_cur = cov_cur - np.outer(v, v) / d
                cov_cur = 0.5 * (cov_cur + cov_cur.T)
                diag_i = np.diag_indices(cov_cur.shape[0])
                cov_cur[diag_i] = np.maximum(cov_cur[diag_i], 0.0)

        final_mean_var = float(np.mean(np.maximum(np.diag(cov_cur)[valid_s], 0.0)))
        chosen = [int(short_idx[j]) for j in picked_local]
        return AcquisitionSelection(
            indices=tuple(chosen),
            smiles=tuple(ctx.pool_smiles[i] for i in chosen),
            scores=tuple(picked_scores),
            all_scores=eff_scores,
            strategy="kriging_believer",
            metadata={
                "exact_gp_schur": True,
                "liar_mode": ctx.liar_mode,
                "kb_base": ctx.kb_base,
                "shortlist_size": len(short_idx),
                "n_pending_conditioned": len(ctx.pending_smiles),
                "mean_var_before": round(initial_mean_var, 6),
                "mean_var_after": round(final_mean_var, 6),
            },
        )

    # Non-GP fallback: Tanimoto-kernel variance deflation on shortlist
    short_idx = _shortlist_indices(base_all, k, ctx.shortlist_multiplier, min_shortlist=128)
    short_smis = [ctx.pool_smiles[int(i)] for i in short_idx]
    fps, _ = featurize_pool(short_smis, kind="ecfp4_binary", n_bits=1024, radius=2)
    sim_mat = tanimoto_kernel(fps)

    mu_cur_fb = mu[short_idx].copy()
    sd_cur_fb = sd[short_idx].copy()
    if ctx.pending_smiles:
        p_fps, p_val = featurize_pool(
            ctx.pending_smiles, kind="ecfp4_binary", n_bits=1024, radius=2
        )
        if bool(p_val.any()):
            p_sim = tanimoto_max_similarity(fps, p_fps[p_val])
            sd_cur_fb = sd_cur_fb * np.sqrt(np.maximum(0.0, 1.0 - 0.95 * (p_sim**2)))

    taken_fb = np.zeros(len(short_idx), dtype=np.bool_)
    picked_fb: list[int] = []
    scores_fb: list[float] = []
    for _ in range(k):
        sc = _kb_step_scores(
            mu_cur_fb,
            sd_cur_fb,
            base=ctx.kb_base,
            beta=ctx.beta,
            best_observed=ctx.best_observed,
            xi=ctx.xi,
        )
        sc[taken_fb] = float("-inf")
        j = int(np.argmax(sc))
        taken_fb[j] = True
        picked_fb.append(j)
        scores_fb.append(float(sc[j]))
        sd_cur_fb = sd_cur_fb * np.sqrt(np.maximum(0.0, 1.0 - 0.95 * (sim_mat[:, j] ** 2)))

    chosen_fb = [int(short_idx[j]) for j in picked_fb]
    return AcquisitionSelection(
        indices=tuple(chosen_fb),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen_fb),
        scores=tuple(scores_fb),
        all_scores=base_all,
        strategy="kriging_believer",
        metadata={
            "exact_gp_schur": False,
            "liar_mode": ctx.liar_mode,
            "shortlist_size": len(short_idx),
            "n_pending_conditioned": len(ctx.pending_smiles),
        },
    )


def acquire_dpp_ucb(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Fast Greedy MAP Determinantal Point Process (Chen, Zhang & Zhou, NeurIPS 2018).

    Forms the L-ensemble quality-diversity kernel
    :math:`L = \\operatorname{diag}(q)\\, K\\, \\operatorname{diag}(q)` on the top-:math:`M`
    UCB shortlist (where :math:`K = 0.999\\,K_{\\text{Tan}} + 0.001\\,I` is strictly PD)
    and greedily maximizes :math:`\\log\\det(L_S)` via incremental Cholesky in
    :math:`O(B^2 M)` time with zero duplicate picks.
    """
    mu, sd = _as_1d_mean_std(ctx)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    ucb = np.where(fin, mu + float(ctx.beta) * sd, float("-inf"))
    n_valid = int(fin.sum())
    if n_valid == 0:
        return _select_top_k(ctx, ucb, "dpp_ucb")

    k = min(n_valid, max(1, int(ctx.batch_size)))
    short_idx = _shortlist_indices(ucb, k, ctx.shortlist_multiplier, min_shortlist=128)
    m = len(short_idx)
    short_smis = [ctx.pool_smiles[int(i)] for i in short_idx]
    short_ucb = ucb[short_idx]

    u_mean = float(np.mean(short_ucb))
    u_std = max(float(np.std(short_ucb)), 1e-6)
    z = np.clip((short_ucb - u_mean) / u_std, -5.0, 5.0)
    eta = max(0.05, float(ctx.dpp_quality_temp))
    q = np.exp(eta * z)

    fps, _ = featurize_pool(short_smis, kind="ecfp4_binary", n_bits=1024, radius=2)
    # Pending-batch down-weighting when asynchronous compounds are in flight
    if ctx.pending_smiles:
        p_fps, p_val = featurize_pool(
            ctx.pending_smiles, kind="ecfp4_binary", n_bits=1024, radius=2
        )
        if bool(p_val.any()):
            p_sim = tanimoto_max_similarity(fps, p_fps[p_val])
            q = q * np.sqrt(np.maximum(1e-6, 1.0 - 0.95 * (p_sim**2)))

    k_reg = 0.999 * tanimoto_kernel(fps) + 0.001 * np.eye(m, dtype=np.float64)
    l_mat = (q[:, None] * k_reg) * q[None, :]

    # Chen et al. (NeurIPS 2018) Algorithm 1: Exact incremental Cholesky greedy MAP DPP
    d2 = np.diag(l_mat).copy()
    c_basis = np.zeros((k, m), dtype=np.float64)
    taken = np.zeros(m, dtype=np.bool_)
    picked_local: list[int] = []
    log_det_gains: list[float] = []

    for t in range(k):
        log_d2 = np.where(taken, float("-inf"), np.log(np.maximum(d2, 1e-300)))
        j = int(np.argmax(log_d2))
        taken[j] = True
        picked_local.append(j)
        log_det_gains.append(float(log_d2[j]))
        if t == k - 1:
            break
        dj = math.sqrt(max(float(d2[j]), 1e-30))
        e_vec = l_mat[j] / dj if t == 0 else (l_mat[j] - c_basis[:t, j] @ c_basis[:t]) / dj
        c_basis[t] = e_vec
        d2 = np.maximum(0.0, d2 - e_vec * e_vec)
        d2[j] = 0.0

    chosen = [int(short_idx[j]) for j in picked_local]
    return AcquisitionSelection(
        indices=tuple(chosen),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen),
        scores=tuple(log_det_gains),
        all_scores=ucb,
        strategy="dpp_ucb",
        metadata={
            "beta": float(ctx.beta),
            "dpp_quality_temp": eta,
            "shortlist_size": m,
            "total_log_det": round(float(sum(log_det_gains)), 4),
            "n_pending_conditioned": len(ctx.pending_smiles),
        },
    )


# ---------------------------------------------------------------------------
# 3. Multi-Objective Pareto Primitives & Acquisitions (qEHVI, ParEGO)
# ---------------------------------------------------------------------------


def pareto_front(
    Y: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Return exact non-dominated ``(indices, points)`` under maximization convention.

    Point ``a`` dominates point ``b`` iff ``all(a >= b)`` and ``any(a > b)``.
    Non-finite rows are ignored.
    """
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D objective matrix [N, M], got shape {arr.shape}.")
    n = arr.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp), np.zeros((0, arr.shape[1]), dtype=np.float64)

    fin_idx = np.flatnonzero(np.all(np.isfinite(arr), axis=1))
    if len(fin_idx) == 0:
        return np.zeros(0, dtype=np.intp), np.zeros((0, arr.shape[1]), dtype=np.float64)

    pts = arr[fin_idx]
    m_pts = len(pts)
    is_nd = np.ones(m_pts, dtype=np.bool_)
    for i in range(m_pts):
        if not is_nd[i]:
            continue
        # Any point j >= pts[i] everywhere and > somewhere dominates i
        ge_all = np.all(pts >= pts[i], axis=1)
        gt_any = np.any(pts > pts[i], axis=1)
        if bool(np.any(ge_all & gt_any)):
            is_nd[i] = False
        else:
            # Prune points strictly dominated by i right away
            dom_by_i = np.all(pts[i] >= pts, axis=1) & np.any(pts[i] > pts, axis=1)
            is_nd[dom_by_i] = False

    nd_idx = fin_idx[is_nd]
    return nd_idx.astype(np.intp), arr[nd_idx]


def compute_hypervolume_2d(
    front: NDArray[np.float64],
    ref: NDArray[np.float64],
) -> float:
    """Compute exact 2D dominated hypervolume (area) above ``ref`` in :math:`O(P\\log P)`."""
    pts = np.asarray(front, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return 0.0
    above = (pts[:, 0] > r[0]) & (pts[:, 1] > r[1]) & np.all(np.isfinite(pts), axis=1)
    if not bool(above.any()):
        return 0.0
    _, nd = pareto_front(pts[above])
    if len(nd) == 0:
        return 0.0
    # Sort descending in objective 0, descending in objective 1
    order = np.lexsort((-nd[:, 1], -nd[:, 0]))
    sorted_nd = nd[order]
    hv = 0.0
    prev_y1 = float(r[1])
    r0 = float(r[0])
    for row in sorted_nd:
        y0 = float(row[0])
        y1 = float(row[1])
        if y1 > prev_y1:
            hv += (y0 - r0) * (y1 - prev_y1)
            prev_y1 = y1
    return float(max(0.0, hv))


def compute_hypervolume_3d(
    front: NDArray[np.float64],
    ref: NDArray[np.float64],
) -> float:
    """Compute exact 3D dominated hypervolume above ``ref`` via Z-slice sweeping."""
    pts = np.asarray(front, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return 0.0
    above = np.all(pts > r[None, :], axis=1) & np.all(np.isfinite(pts), axis=1)
    if not bool(above.any()):
        return 0.0
    _, nd = pareto_front(pts[above])
    if len(nd) == 0:
        return 0.0
    order = np.argsort(-nd[:, 2], kind="mergesort")
    s = nd[order]
    hv = 0.0
    p_len = len(s)
    for i in range(p_len):
        z_hi = float(s[i, 2])
        z_lo = float(s[i + 1, 2]) if i + 1 < p_len else float(r[2])
        dz = z_hi - z_lo
        if dz > 0.0:
            area_2d = compute_hypervolume_2d(s[: i + 1, :2], r[:2])
            hv += area_2d * dz
    return float(max(0.0, hv))


def compute_hypervolume(
    front: NDArray[np.float64],
    ref: NDArray[np.float64],
    *,
    n_mc: int = 4096,
    seed: int = 42,
) -> float:
    """Compute dominated hypervolume above ``ref`` (exact for M<=3, Sobol QMC for M>=4)."""
    pts = np.asarray(front, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return 0.0
    m = int(pts.shape[1])
    if len(r) != m:
        raise ValueError(f"ref length {len(r)} does not match objective dim {m}.")
    if m == 1:
        fin = pts[np.isfinite(pts[:, 0]), 0]
        return float(max(0.0, float(np.max(fin)) - float(r[0]))) if len(fin) else 0.0
    if m == 2:
        return compute_hypervolume_2d(pts, r)
    if m == 3:
        return compute_hypervolume_3d(pts, r)

    above = np.all(pts > r[None, :], axis=1) & np.all(np.isfinite(pts), axis=1)
    if not bool(above.any()):
        return 0.0
    _, nd = pareto_front(pts[above])
    if len(nd) == 0:
        return 0.0
    hi = np.max(nd, axis=0)
    span = hi - r
    box_vol = float(np.prod(span))
    if box_vol <= 0.0:
        return 0.0
    # Power-of-2 Sobol sequence for optimal QMC balance
    pow2 = max(8, int(math.ceil(math.log2(max(256, int(n_mc))))))
    sampler = qmc.Sobol(d=m, scramble=True, seed=int(seed))
    u = sampler.random_base2(m=pow2)
    samples = r[None, :] + u * span[None, :]
    dominated = np.zeros(len(samples), dtype=np.bool_)
    for p in nd:
        dominated |= np.all(samples <= p[None, :], axis=1)
    return float(box_vol * float(np.mean(dominated)))


def _psi_interval(
    a: float,
    b: float,
    mu: NDArray[np.float64],
    sd: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute exact :math:`\\Psi(a, b, \\mu, \\sigma) = \\int_a^b (y-a)N(y|\\mu,\\sigma^2)dy`."""
    out = np.zeros_like(mu, dtype=np.float64)
    pos = sd > 1e-12
    if bool(pos.any()):
        m_p = mu[pos]
        s_p = sd[pos]
        ta = (a - m_p) / s_p
        if math.isinf(b):
            val = (m_p - a) * (1.0 - _norm_cdf(ta)) + s_p * _norm_pdf(ta)
        else:
            tb = (b - m_p) / s_p
            val = (m_p - a) * (_norm_cdf(tb) - _norm_cdf(ta)) + s_p * (
                _norm_pdf(ta) - _norm_pdf(tb)
            )
        out[pos] = np.maximum(0.0, val)
    det = ~pos
    if bool(det.any()):
        m_d = mu[det]
        in_interval = (m_d > a) & (m_d <= b)
        out[det] = np.where(in_interval, m_d - a, 0.0)
    return out


def compute_exact_2d_ehvi(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    front: NDArray[np.float64],
    ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute exact analytical 2D Expected Hypervolume Improvement (Emmerich / Yang)."""
    n = mean.shape[0]
    ehvi = np.full(n, float("-inf"), dtype=np.float64)
    fin = (
        np.all(np.isfinite(mean), axis=1)
        & np.all(np.isfinite(std), axis=1)
        & np.all(std >= 0.0, axis=1)
    )
    if not bool(fin.any()):
        return ehvi

    r0, r1 = float(ref[0]), float(ref[1])
    mu1 = mean[fin, 0]
    mu2 = mean[fin, 1]
    s1 = std[fin, 0]
    s2 = std[fin, 1]

    above = (
        (front[:, 0] > r0) & (front[:, 1] > r1) & np.all(np.isfinite(front), axis=1)
        if len(front) > 0
        else np.zeros(0, dtype=np.bool_)
    )
    if bool(np.any(above)):
        _, nd = pareto_front(front[above])
        # Sort ascending in obj 0 (hence strictly descending in obj 1)
        nd = nd[np.argsort(nd[:, 0], kind="mergesort")]
    else:
        nd = np.zeros((0, 2), dtype=np.float64)

    p_len = len(nd)
    xs = [r0, *[float(nd[i, 0]) for i in range(p_len)], float("inf")]
    ys = [float("inf"), *[float(nd[i, 1]) for i in range(p_len)], r1]

    total = np.zeros(len(mu1), dtype=np.float64)
    for i in range(p_len + 1):
        l_i = xs[i]
        u_i = xs[i + 1]
        h_i = ys[i + 1]
        y_term = _psi_interval(h_i, float("inf"), mu2, s2)
        x_term = _psi_interval(l_i, u_i, mu1, s1)
        if not math.isinf(u_i):
            # Tail probability P(Y1 > u_i) * (u_i - l_i)
            prob_gt = np.where(
                s1 > 1e-12,
                1.0 - _norm_cdf((u_i - mu1) / np.maximum(s1, 1e-12)),
                np.where(mu1 > u_i, 1.0, 0.0),
            )
            x_term = x_term + (u_i - l_i) * prob_gt
        total += x_term * y_term

    ehvi[fin] = np.maximum(0.0, total)
    return ehvi


def _infer_reference_point(
    observed: NDArray[np.float64] | None,
    mean: NDArray[np.float64],
) -> NDArray[np.float64]:
    parts: list[NDArray[np.float64]] = []
    fin_m = mean[np.all(np.isfinite(mean), axis=1)]
    if len(fin_m) > 0:
        parts.append(fin_m)
    if observed is not None and len(observed) > 0:
        fin_o = observed[np.all(np.isfinite(observed), axis=1)]
        if len(fin_o) > 0:
            parts.append(fin_o)
    stacked = np.vstack(parts) if parts else np.zeros((1, mean.shape[1]), dtype=np.float64)
    lo = np.min(stacked, axis=0)
    hi = np.max(stacked, axis=0)
    margin = np.maximum(0.1 * (hi - lo), 0.1)
    return np.asarray(lo - margin, dtype=np.float64)


def acquire_qehvi(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Exact 2D analytical EHVI + sequential greedy Sobol-QMC batch ``qEHVI``."""
    mean = np.asarray(ctx.mean, dtype=np.float64)
    std = np.asarray(ctx.std, dtype=np.float64)
    if mean.ndim != 2 or std.ndim != 2 or mean.shape != std.shape or mean.shape[1] < 2:
        raise ValueError(
            f"qehvi requires 2D [N, M>=2] mean/std arrays, got mean={mean.shape}, std={std.shape}."
        )
    n, m_obj = mean.shape
    fin = (
        np.all(np.isfinite(mean), axis=1)
        & np.all(np.isfinite(std), axis=1)
        & np.all(std >= 0.0, axis=1)
    )
    n_valid = int(fin.sum())
    if n_valid == 0:
        return _select_top_k(ctx, np.full(n, float("-inf"), dtype=np.float64), "qehvi")

    obs = (
        np.asarray(ctx.observed_objectives, dtype=np.float64)
        if ctx.observed_objectives is not None and len(ctx.observed_objectives) > 0
        else np.zeros((0, m_obj), dtype=np.float64)
    )
    ref = (
        np.asarray(ctx.reference_point, dtype=np.float64)
        if ctx.reference_point is not None
        else _infer_reference_point(obs, mean)
    )
    _, base_front = pareto_front(obs) if len(obs) > 0 else (np.zeros(0, dtype=np.intp), obs)
    k = min(n_valid, max(1, int(ctx.batch_size)))

    # Tie-breaker: standardized average UCB so zero-EHVI extreme tails still rank sensibly
    ucb_mat = mean + float(ctx.beta) * std
    u_min = np.nanmin(ucb_mat[fin], axis=0)
    u_span = np.maximum(np.nanmax(ucb_mat[fin], axis=0) - u_min, 1e-6)
    fallback_rank = np.full(n, float("-inf"), dtype=np.float64)
    fallback_rank[fin] = np.mean((ucb_mat[fin] - u_min) / u_span, axis=1)

    if m_obj == 2:
        ehvi_1 = compute_exact_2d_ehvi(mean, std, base_front, ref)
        if k == 1:
            return _select_top_k(
                ctx,
                ehvi_1,
                "qehvi",
                tie_break=fallback_rank,
                metadata={"exact_2d": True, "reference_point": ref.tolist()},
            )
        rank_for_short = np.where(ehvi_1 > 0.0, ehvi_1, -1e9 + fallback_rank)
    else:
        ehvi_1 = fallback_rank.copy()
        rank_for_short = fallback_rank

    # Sequential greedy QMC batch qEHVI over shortlist
    short_idx = _shortlist_indices(
        rank_for_short, k, max(4, ctx.shortlist_multiplier // 2), min_shortlist=64
    )
    n_s = len(short_idx)
    pow2 = max(6, int(math.ceil(math.log2(max(64, int(ctx.qmc_samples))))))
    n_draws = 1 << pow2
    sampler = qmc.Sobol(d=m_obj, scramble=True, seed=int(ctx.seed))
    u_qmc = np.clip(sampler.random_base2(m=pow2), 1e-6, 1.0 - 1e-6)
    from scipy.special import ndtri

    z_qmc = np.asarray(ndtri(u_qmc), dtype=np.float64)  # [S, M]
    # Decorrelate Sobol draws across distinct candidates via per-candidate row permutations
    # so every candidate keeps exact Sobol marginal stratification while joint batch HV
    # evaluates independent draws rather than perfectly correlated (+1) draws.
    mu_s = mean[short_idx]
    sd_s = std[short_idx]
    rng_perm = np.random.default_rng(int(ctx.seed) + 101)
    samples = np.empty((n_s, n_draws, m_obj), dtype=np.float64)
    for c_idx in range(n_s):
        z_c = z_qmc[rng_perm.permutation(n_draws)]
        samples[c_idx] = mu_s[c_idx][None, :] + sd_s[c_idx][None, :] * z_c

    base_hv = compute_hypervolume(base_front, ref, n_mc=512, seed=ctx.seed)
    # Track per-draw augmented fronts (base_front + already picked batch points)
    cur_hvs = np.full(n_draws, base_hv, dtype=np.float64)
    chosen_draws: list[NDArray[np.float64]] = []
    taken = np.zeros(n_s, dtype=np.bool_)
    picked_local: list[int] = []
    picked_gains: list[float] = []

    for step in range(k):
        if step == 0 and m_obj == 2:
            step_scores = ehvi_1[short_idx].copy()
            step_scores[taken] = float("-inf")
            if float(np.max(step_scores)) <= 1e-12:
                tb = np.where(taken, float("-inf"), fallback_rank[short_idx])
                j = int(np.argmax(tb))
            else:
                j = int(np.argmax(step_scores))
            gain = float(max(0.0, ehvi_1[int(short_idx[j])]))
        else:
            gains = np.full(n_s, float("-inf"), dtype=np.float64)
            for cand in range(n_s):
                if taken[cand]:
                    continue
                cand_pts = samples[cand]  # [S, M]
                delta_sum = 0.0
                for s_i in range(n_draws):
                    pt = cand_pts[s_i : s_i + 1]
                    if bool(np.any(pt[0] <= ref)):
                        continue
                    if chosen_draws:
                        extra = np.vstack([cd[s_i : s_i + 1] for cd in chosen_draws] + [pt])
                        aug = np.vstack([base_front, extra]) if len(base_front) else extra
                    else:
                        aug = np.vstack([base_front, pt]) if len(base_front) else pt
                    new_hv = compute_hypervolume(aug, ref, n_mc=512, seed=ctx.seed + s_i)
                    if new_hv > cur_hvs[s_i]:
                        delta_sum += new_hv - cur_hvs[s_i]
                gains[cand] = delta_sum / float(n_draws)

            if float(np.max(gains)) <= 1e-12:
                tb = np.where(taken, float("-inf"), fallback_rank[short_idx])
                j = int(np.argmax(tb))
                gain = float(max(0.0, gains[j]))
            else:
                j = int(np.argmax(gains))
                gain = float(gains[j])

        taken[j] = True
        picked_local.append(j)
        picked_gains.append(gain)
        chosen_draws.append(samples[j])

        # Update per-draw baseline hypervolumes after committing pick j
        for s_i in range(n_draws):
            extra = np.vstack([cd[s_i : s_i + 1] for cd in chosen_draws])
            aug = np.vstack([base_front, extra]) if len(base_front) else extra
            cur_hvs[s_i] = compute_hypervolume(aug, ref, n_mc=512, seed=ctx.seed + s_i)

    chosen = [int(short_idx[j]) for j in picked_local]
    all_sc = ehvi_1.copy()
    return AcquisitionSelection(
        indices=tuple(chosen),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen),
        scores=tuple(picked_gains),
        all_scores=all_sc,
        strategy="qehvi",
        metadata={
            "exact_2d_init": m_obj == 2,
            "n_objectives": m_obj,
            "qmc_samples": n_draws,
            "base_hypervolume": round(base_hv, 6),
            "reference_point": [round(float(x), 4) for x in ref],
        },
    )


def acquire_pareto_scalarized_ucb(ctx: AcquisitionContext) -> AcquisitionSelection:
    """ParEGO Augmented Chebyshev Scalarized UCB across :math:`M \\ge 2` objectives."""
    mean = np.asarray(ctx.mean, dtype=np.float64)
    std = np.asarray(ctx.std, dtype=np.float64)
    if mean.ndim != 2 or std.ndim != 2 or mean.shape != std.shape or mean.shape[1] < 2:
        raise ValueError(
            "pareto_scalarized_ucb requires 2D [N, M>=2] mean/std arrays, "
            f"got mean={mean.shape}, std={std.shape}."
        )
    n, m_obj = mean.shape
    fin = (
        np.all(np.isfinite(mean), axis=1)
        & np.all(np.isfinite(std), axis=1)
        & np.all(std >= 0.0, axis=1)
    )
    n_valid = int(fin.sum())
    if n_valid == 0:
        return _select_top_k(
            ctx, np.full(n, float("-inf"), dtype=np.float64), "pareto_scalarized_ucb"
        )

    k = min(n_valid, max(1, int(ctx.batch_size)))
    ucb = mean + float(ctx.beta) * std
    u_lo = np.min(ucb[fin], axis=0)
    u_hi = np.max(ucb[fin], axis=0)
    u_norm = np.zeros_like(ucb, dtype=np.float64)
    u_norm[fin] = (ucb[fin] - u_lo) / np.maximum(u_hi - u_lo, 1e-6)

    rng = np.random.default_rng(ctx.seed)
    weights_list: list[NDArray[np.float64]] = []
    if ctx.objective_weights is not None:
        w = np.asarray(ctx.objective_weights, dtype=np.float64)
        if len(w) != m_obj or float(np.sum(w)) <= 0.0:
            raise ValueError(f"Invalid objective_weights of shape {w.shape} for M={m_obj}.")
        w_norm = np.maximum(w, 0.0) / float(np.sum(np.maximum(w, 0.0)))
        weights_list = [w_norm for _ in range(k)]
    elif m_obj == 2:
        grid = np.linspace(0.08, 0.92, k) if k > 1 else np.array([0.5])
        weights_list = [np.array([float(g), 1.0 - float(g)], dtype=np.float64) for g in grid]
    else:
        for _ in range(k):
            w_raw = rng.dirichlet(np.ones(m_obj, dtype=np.float64))
            weights_list.append(np.asarray(w_raw, dtype=np.float64))

    taken = ~fin
    chosen: list[int] = []
    chosen_scores: list[float] = []
    first_scores = np.full(n, float("-inf"), dtype=np.float64)

    for b, w_vec in enumerate(weights_list):
        weighted = u_norm * w_vec[None, :]
        # Augmented Chebyshev scalarization: min_m (w_m * u_m) + rho * sum_m (w_m * u_m)
        sc = np.min(weighted, axis=1) + 0.05 * np.sum(weighted, axis=1)
        sc = np.where(fin, sc, float("-inf"))
        if b == 0:
            first_scores = sc.copy()
        sc[taken] = float("-inf")
        j = int(np.argmax(sc))
        taken[j] = True
        chosen.append(j)
        chosen_scores.append(float(sc[j]))

    return AcquisitionSelection(
        indices=tuple(chosen),
        smiles=tuple(ctx.pool_smiles[i] for i in chosen),
        scores=tuple(chosen_scores),
        all_scores=first_scores,
        strategy="pareto_scalarized_ucb",
        metadata={
            "n_objectives": m_obj,
            "beta": float(ctx.beta),
            "user_weighted": ctx.objective_weights is not None,
        },
    )


# ---------------------------------------------------------------------------
# 4. Feasibility-Constrained Acquisitions (CNSAdmetSuite / Profile / PoseReality)
# ---------------------------------------------------------------------------


def build_feasibility_vector(
    smiles: Sequence[str],
    *,
    property_profile: str | None = None,
    admet_suite: CNSAdmetSuite | None = None,
    herg_max: float = 0.50,
    bbb_min: float | None = None,
    pgp_max: float | None = None,
    avoidome_max: float | None = None,
    triage_hard_gate: bool = True,
    pose_audit_reports: Sequence[PoseAuditReport] | None = None,
    custom_constraint_fns: Sequence[Callable[[Sequence[str]], Sequence[float]]] = (),
    gate_temperature: float = 0.08,
) -> NDArray[np.float64]:
    """Compute smooth feasibility probability :math:`P_{\\text{feas}}(x) \\in [0, 1]`.

    Integrates GenEuclidean's ``triage_molecule`` (PAINS/BRENK/DCU quarantine),
    ``PROPERTY_PROFILE_REGISTRY`` (``cns_mpo_v2``, ``oral_ro5``, ``kinase_selectivity``,
    ``beyond_ro5``), calibrated 12-endpoint ``CNSAdmetSuite`` Avoid-ome probabilities,
    7-axis ``PoseRealityAuditor`` reports, and arbitrary probabilistic constraint functions.
    """
    n = len(smiles)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    probs = np.ones(n, dtype=np.float64)
    temp = max(1e-3, float(gate_temperature))

    if triage_hard_gate:
        from geneuclidean.cns.medchem_triage import triage_molecule

        for i, smi in enumerate(smiles):
            tr = triage_molecule(smi)
            if not tr.valid:
                probs[i] = 0.0
            else:
                probs[i] *= float(np.clip(tr.hygiene_gate, 0.0, 1.0))

    if property_profile is not None:
        from geneuclidean.sdk.profiles import evaluate_property_profile

        for i, smi in enumerate(smiles):
            if probs[i] <= 0.0:
                continue
            try:
                res = evaluate_property_profile(smi, profile=property_profile)
                probs[i] *= float(np.clip(res.score, 0.0, 1.0))
            except ValueError:
                probs[i] = 0.0

    if admet_suite is not None:
        fitted = set(admet_suite.fitted_endpoints)
        if len(fitted) > 3:
            preds_v4 = admet_suite.predict_v4(smiles)
            for i, p4 in enumerate(preds_v4):
                if not p4.valid:
                    probs[i] = 0.0
                    continue
                if "herg" in p4.probabilities:
                    probs[i] *= float(expit((float(herg_max) - p4.probabilities["herg"]) / temp))
                if bbb_min is not None and "bbb" in p4.probabilities:
                    probs[i] *= float(expit((p4.probabilities["bbb"] - float(bbb_min)) / temp))
                if pgp_max is not None and "pgp" in p4.probabilities:
                    probs[i] *= float(expit((float(pgp_max) - p4.probabilities["pgp"]) / temp))
                if avoidome_max is not None:
                    for ep in ("htr2b_avoid", "hrh1_avoid", "ames"):
                        if ep in p4.probabilities:
                            probs[i] *= float(
                                expit((float(avoidome_max) - p4.probabilities[ep]) / temp)
                            )
        else:
            preds_v3 = admet_suite.predict(smiles)
            for i, p3 in enumerate(preds_v3):
                if not p3.valid:
                    probs[i] = 0.0
                    continue
                probs[i] *= float(expit((float(herg_max) - p3.herg_prob) / temp))
                if bbb_min is not None:
                    probs[i] *= float(expit((p3.bbb_prob - float(bbb_min)) / temp))
                if pgp_max is not None:
                    probs[i] *= float(expit((float(pgp_max) - p3.pgp_prob) / temp))

    if pose_audit_reports is not None:
        if len(pose_audit_reports) != n:
            raise ValueError(
                f"pose_audit_reports length {len(pose_audit_reports)} != len(smiles) {n}."
            )
        for i, rep in enumerate(pose_audit_reports):
            if rep.passed_physical_beyond_pb:
                probs[i] *= float(np.clip(rep.reality_score, 0.0, 1.0))
            else:
                probs[i] *= 0.05 * float(np.clip(rep.reality_score, 0.0, 1.0))

    for fn in custom_constraint_fns:
        vals = np.asarray(fn(smiles), dtype=np.float64)
        if len(vals) != n:
            raise ValueError(f"Custom constraint fn returned length {len(vals)}, expected {n}.")
        probs *= np.clip(np.nan_to_num(vals, nan=0.0), 0.0, 1.0)

    return np.clip(probs, 0.0, 1.0)


def acquire_constrained_ei(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Feasibility-Constrained EI with Phase-1 Feasibility Recovery fallback.

    When at least one feasible observation exists (``math.isfinite(ctx.best_observed)``),
    scores :math:`\\text{EI}(x) \\cdot P_{\\text{feas}}(x)` with log-space tie-breaking.
    When **zero** feasible observations exist (``not math.isfinite(ctx.best_observed)``),
    automatically enters **Phase-1 Feasibility Recovery** maximizing :math:`P_{\\text{feas}}(x)`
    tie-broken by UCB so the loop finds a feasible anchor first.
    """
    mu, sd = _as_1d_mean_std(ctx)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    if ctx.feasibility_probs is not None:
        if len(ctx.feasibility_probs) != len(mu):
            raise ValueError("feasibility_probs length must match pool_smiles.")
        p_feas = np.clip(
            np.nan_to_num(np.asarray(ctx.feasibility_probs, dtype=np.float64), nan=0.0), 0.0, 1.0
        )
    else:
        p_feas = np.ones_like(mu, dtype=np.float64)

    ucb = np.where(fin, mu + float(ctx.beta) * sd, float("-inf"))

    # Phase-1 Feasibility Recovery: 0 feasible observations recorded so far
    if not math.isfinite(ctx.best_observed):
        phase1_scores = np.where(fin, p_feas, float("-inf"))
        return _select_top_k(
            ctx,
            phase1_scores,
            "constrained_ei",
            tie_break=ucb,
            metadata={"phase1_feasibility_recovery": True},
        )

    ei = compute_expected_improvement(mu, sd, ctx.best_observed, xi=ctx.xi)
    log_ei = compute_log_expected_improvement(mu, sd, ctx.best_observed, xi=ctx.xi)
    cei = np.where(fin, ei * p_feas, float("-inf"))
    log_cei = np.where(fin, log_ei + np.log(np.maximum(p_feas, 1e-12)), float("-inf"))
    return _select_top_k(
        ctx,
        cei,
        "constrained_ei",
        tie_break=log_cei,
        metadata={"phase1_feasibility_recovery": False, "xi": float(ctx.xi)},
    )


def acquire_avoidome_feasibility_ucb(ctx: AcquisitionContext) -> AcquisitionSelection:
    """Sign-inversion-safe Avoid-ome / ADMET Feasibility-Weighted UCB.

    Shifts raw UCB onto strictly positive utility
    :math:`\\text{UCB}(x) - \\min_i \\text{UCB}(x_i) + 10^{-6}`
    before multiplying by :math:`P_{\\text{feas}}(x) \\in [0, 1]`, guaranteeing that
    negative objective scales (e.g., ``logBB`` or standardized energies) never invert
    feasibility preference.
    """
    mu, sd = _as_1d_mean_std(ctx)
    fin = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0.0)
    if not bool(fin.any()):
        return _select_top_k(ctx, np.full_like(mu, float("-inf")), "avoidome_feasibility_ucb")

    if ctx.feasibility_probs is not None:
        if len(ctx.feasibility_probs) != len(mu):
            raise ValueError("feasibility_probs length must match pool_smiles.")
        p_feas = np.clip(
            np.nan_to_num(np.asarray(ctx.feasibility_probs, dtype=np.float64), nan=0.0), 0.0, 1.0
        )
    else:
        p_feas = np.ones_like(mu, dtype=np.float64)

    raw_ucb = mu + float(ctx.beta) * sd
    u_min = float(np.min(raw_ucb[fin]))
    shifted_pos = (raw_ucb - u_min) + 1e-6
    scores = np.where(fin, shifted_pos * p_feas, float("-inf"))
    return _select_top_k(
        ctx,
        scores,
        "avoidome_feasibility_ucb",
        tie_break=p_feas,
        metadata={"beta": float(ctx.beta), "ucb_shift_floor": round(u_min, 6)},
    )


# ---------------------------------------------------------------------------
# Global Acquisition Plugin Registry
# ---------------------------------------------------------------------------

ACQUISITION_REGISTRY: Registry[AcquisitionFn] = Registry("active_learning_acquisitions")

ACQUISITION_REGISTRY.register("greedy", acquire_greedy, metadata={"family": "pointwise"})
ACQUISITION_REGISTRY.register("uncertainty", acquire_uncertainty, metadata={"family": "pointwise"})
ACQUISITION_REGISTRY.register("ucb", acquire_ucb, metadata={"family": "pointwise"})
ACQUISITION_REGISTRY.register("pi", acquire_pi, metadata={"family": "pointwise"})
ACQUISITION_REGISTRY.register("ei", acquire_ei, metadata={"family": "pointwise"})
ACQUISITION_REGISTRY.register("log_ei", acquire_log_ei, metadata={"family": "pointwise"})
ACQUISITION_REGISTRY.register(
    "thompson", acquire_thompson, aliases=("ts",), metadata={"family": "posterior_sampling"}
)
ACQUISITION_REGISTRY.register(
    "maxmin_ucb",
    acquire_maxmin_ucb,
    aliases=("maxmin_diverse_ucb",),
    metadata={"family": "batch_diverse"},
)
ACQUISITION_REGISTRY.register(
    "kriging_believer",
    acquire_kriging_believer,
    aliases=("kb", "constant_liar"),
    metadata={"family": "batch_diverse", "supports_pending": True},
)
ACQUISITION_REGISTRY.register(
    "dpp_ucb",
    acquire_dpp_ucb,
    aliases=("dpp",),
    metadata={"family": "batch_diverse", "supports_pending": True},
)
ACQUISITION_REGISTRY.register(
    "qehvi",
    acquire_qehvi,
    aliases=("ehvi",),
    metadata={"family": "multi_objective"},
)
ACQUISITION_REGISTRY.register(
    "pareto_scalarized_ucb",
    acquire_pareto_scalarized_ucb,
    aliases=("parego", "pareto_ucb"),
    metadata={"family": "multi_objective"},
)
ACQUISITION_REGISTRY.register(
    "constrained_ei",
    acquire_constrained_ei,
    aliases=("cei",),
    metadata={"family": "constrained"},
)
ACQUISITION_REGISTRY.register(
    "avoidome_feasibility_ucb",
    acquire_avoidome_feasibility_ucb,
    aliases=("feasible_ucb", "constrained_ucb"),
    metadata={"family": "constrained"},
)


def register_acquisition(
    name: str,
    fn: AcquisitionFn,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
) -> AcquisitionFn:
    """Register a custom acquisition strategy in :data:`ACQUISITION_REGISTRY`."""
    return ACQUISITION_REGISTRY.register(name, fn, aliases=aliases, override=override)


def list_acquisitions() -> list[str]:
    """Return sorted canonical names of registered acquisition strategies."""
    return ACQUISITION_REGISTRY.names()


def run_acquisition(strategy: str | AcquisitionFn, ctx: AcquisitionContext) -> AcquisitionSelection:
    """Execute a registered acquisition strategy name or custom callable."""
    fn = ACQUISITION_REGISTRY.get(strategy) if isinstance(strategy, str) else strategy
    return fn(ctx)


__all__ = [
    "ACQUISITION_REGISTRY",
    "AcquisitionContext",
    "AcquisitionFn",
    "AcquisitionSelection",
    "KBBaseAcq",
    "LiarMode",
    "acquire_avoidome_feasibility_ucb",
    "acquire_constrained_ei",
    "acquire_dpp_ucb",
    "acquire_ei",
    "acquire_greedy",
    "acquire_kriging_believer",
    "acquire_log_ei",
    "acquire_maxmin_ucb",
    "acquire_pareto_scalarized_ucb",
    "acquire_pi",
    "acquire_qehvi",
    "acquire_thompson",
    "acquire_ucb",
    "acquire_uncertainty",
    "build_feasibility_vector",
    "compute_exact_2d_ehvi",
    "compute_expected_improvement",
    "compute_hypervolume",
    "compute_hypervolume_2d",
    "compute_hypervolume_3d",
    "compute_log_expected_improvement",
    "compute_log_h",
    "compute_probability_of_improvement",
    "list_acquisitions",
    "pareto_front",
    "register_acquisition",
    "run_acquisition",
]
