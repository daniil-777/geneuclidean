"""Cost-aware, correlation-gated Multi-Fidelity Active Learning DMTA router.

Routes candidate evaluations across a hierarchy of scoring tiers — for example:
Tier 0 (fast 2D / ``DrugCLIP`` screening, cost ~ 1), Tier 1 (AutoDock Vina +
``PoseRealityAuditor`` 7-axis physical gate, cost ~ 20), and Tier 2 (OpenFE RBFE or
wet-lab experimental assay, cost ~ 200) — using an autoregressive Kennedy-O'Hagan
linear-residual ``TanimotoGP`` surrogate:

.. math::
    \\hat{f}_H(x) = \\hat{\\alpha} + \\hat{\\rho}\\, \\hat{f}_L(x) + \\delta(x)

**Critical Anti-Hype Safeguard (Sabanza-Gil et al., 2024-2025):**
Multi-fidelity Bayesian optimization performs *worse* than single-fidelity BO when the
low-fidelity proxy correlates poorly with high-fidelity reality (:math:`\\rho_s < 0.25`).
:class:`MultiFidelityDMTA` measures empirical cross-fidelity Spearman :math:`\\hat{\\rho}_s`
online on paired observations and automatically disables untrusted lower tiers, routing
all remaining budget cleanly to the high-fidelity ``TanimotoGP``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from geneuclidean.active_learning.acquisitions import (
    compute_expected_improvement,
    compute_log_expected_improvement,
)
from geneuclidean.active_learning.diagnostics import select_cold_start
from geneuclidean.active_learning.surrogates import TanimotoGP


@dataclass(frozen=True)
class FidelityTierSpec:
    """Specification of a single oracle fidelity level (higher ``level`` = higher fidelity)."""

    name: str
    level: int
    cost: float
    oracle_fn: Callable[[Sequence[str]], Sequence[float]]
    hard_gate_fn: Callable[[Sequence[str]], Sequence[bool]] | None = None

    def __post_init__(self) -> None:
        if self.cost <= 0.0:
            raise ValueError(f"Tier {self.name!r} cost must be > 0, got {self.cost}.")


@dataclass(frozen=True)
class MultiFidelityObservation:
    """Single evaluated compound record at a specific fidelity tier."""

    smiles: str
    tier_level: int
    tier_name: str
    value: float
    passed_gate: bool
    cost: float
    round_index: int


@dataclass(frozen=True)
class MultiFidelityRoundSummary:
    """Per-round routing, cost, correlation guardrail, and hit discovery summary."""

    round_index: int
    tier_evaluated_counts: dict[str, int]
    cost_this_round: float
    cumulative_cost: float
    paired_spearman_rho: dict[str, float | None]
    low_fidelity_trusted: dict[str, bool]
    n_high_fidelity_total: int
    best_high_fidelity_value: float
    best_high_fidelity_smiles: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "tier_evaluated_counts": dict(self.tier_evaluated_counts),
            "cost_this_round": round(self.cost_this_round, 4),
            "cumulative_cost": round(self.cumulative_cost, 4),
            "paired_spearman_rho": {
                k: round(float(v), 4) if v is not None and math.isfinite(v) else None
                for k, v in self.paired_spearman_rho.items()
            },
            "low_fidelity_trusted": dict(self.low_fidelity_trusted),
            "n_high_fidelity_total": self.n_high_fidelity_total,
            "best_high_fidelity_value": (
                round(self.best_high_fidelity_value, 4)
                if math.isfinite(self.best_high_fidelity_value)
                else None
            ),
            "best_high_fidelity_smiles": self.best_high_fidelity_smiles,
            "warnings": list(self.warnings),
        }


@dataclass
class _AutoregressiveBridge:
    """Internal fitted Kennedy-O'Hagan bridge from lower tier L to high tier H."""

    source_level: int | None
    alpha: float
    rho: float
    spearman_rho: float
    residual_gp: TanimotoGP
    low_gp: TanimotoGP | None
    low_obs: dict[str, float] = field(default_factory=dict)


class MultiFidelityDMTA:
    """Cost-aware, correlation-gated Multi-Fidelity DMTA router across ``K >= 2`` tiers."""

    def __init__(
        self,
        tiers: Sequence[FidelityTierSpec],
        *,
        min_correlation: float = 0.25,
        min_paired_for_gate: int = 8,
        enable_correlation_gate: bool = True,
        ridge_alpha: float = 1e-2,
        n_bits: int = 2048,
        seed: int = 42,
    ) -> None:
        if len(tiers) < 2:
            raise ValueError("MultiFidelityDMTA requires at least 2 FidelityTierSpec levels.")
        ordered = sorted(tiers, key=lambda t: t.level)
        if len({t.level for t in ordered}) != len(ordered):
            raise ValueError("FidelityTierSpec levels must be distinct integers.")

        self.tiers: tuple[FidelityTierSpec, ...] = tuple(ordered)
        self._tier_by_level: dict[int, FidelityTierSpec] = {t.level: t for t in ordered}
        self.high_tier: FidelityTierSpec = ordered[-1]
        self.low_tiers: tuple[FidelityTierSpec, ...] = tuple(ordered[:-1])

        self.min_correlation = float(min_correlation)
        self.min_paired_for_gate = max(4, int(min_paired_for_gate))
        self.enable_correlation_gate = bool(enable_correlation_gate)
        self.ridge_alpha = float(ridge_alpha)
        self.n_bits = int(n_bits)
        self.seed = int(seed)

        self.observations: dict[int, dict[str, float]] = {t.level: {} for t in ordered}
        self.gate_passed: dict[int, dict[str, bool]] = {t.level: {} for t in ordered}
        self.ledger: list[MultiFidelityObservation] = []
        self.history: list[MultiFidelityRoundSummary] = []
        self.total_cost_spent: float = 0.0
        self.low_fidelity_trusted: dict[int, bool] = {t.level: True for t in self.low_tiers}
        self.paired_spearman_rho: dict[int, float] = {t.level: float("nan") for t in self.low_tiers}
        self._active_bridge: _AutoregressiveBridge | None = None

    @property
    def high_observations(self) -> dict[str, float]:
        """Mapping of ``smiles -> high_fidelity_score`` recorded on the top tier."""
        return self.observations[self.high_tier.level]

    def _evaluate_tier(
        self,
        tier: FidelityTierSpec,
        smiles: Sequence[str],
        *,
        round_index: int,
    ) -> tuple[int, float]:
        """Evaluate unobserved SMILES on ``tier`` and record costs, values, and gates."""
        todo = [s for s in smiles if s not in self.observations[tier.level]]
        if not todo:
            return 0, 0.0
        vals = [float(v) for v in tier.oracle_fn(todo)]
        if len(vals) != len(todo):
            raise ValueError(
                f"Tier {tier.name!r} oracle_fn returned {len(vals)} values for {len(todo)} inputs."
            )
        if tier.hard_gate_fn is not None:
            gates = [bool(g) for g in tier.hard_gate_fn(todo)]
            if len(gates) != len(todo):
                raise ValueError(
                    f"Tier {tier.name!r} hard_gate_fn returned {len(gates)} flags for {len(todo)}."
                )
        else:
            gates = [True] * len(todo)

        for smi, val, ok_gate in zip(todo, vals, gates, strict=True):
            if math.isfinite(val):
                self.observations[tier.level][smi] = val
                self.gate_passed[tier.level][smi] = ok_gate
                self.ledger.append(
                    MultiFidelityObservation(
                        smiles=smi,
                        tier_level=tier.level,
                        tier_name=tier.name,
                        value=val,
                        passed_gate=ok_gate,
                        cost=tier.cost,
                        round_index=round_index,
                    )
                )
        spent = float(len(todo)) * tier.cost
        self.total_cost_spent += spent
        return len(todo), spent

    def _update_correlation_guardrails(self) -> list[str]:
        """Recompute online paired Spearman correlations and gate low-correlation proxies."""
        warnings_list: list[str] = []
        hi_obs = self.observations[self.high_tier.level]

        for tier in self.low_tiers:
            lo_obs = self.observations[tier.level]
            paired = [s for s in hi_obs if s in lo_obs]
            if len(paired) < 4:
                self.paired_spearman_rho[tier.level] = float("nan")
                continue

            x_lo = np.array([lo_obs[s] for s in paired], dtype=np.float64)
            y_hi = np.array([hi_obs[s] for s in paired], dtype=np.float64)
            if float(np.std(x_lo)) < 1e-12 or float(np.std(y_hi)) < 1e-12:
                rho_s = 0.0
            else:
                rho_s = float(spearmanr(x_lo, y_hi).statistic)
                if not math.isfinite(rho_s):
                    rho_s = 0.0
            self.paired_spearman_rho[tier.level] = rho_s

            if self.enable_correlation_gate and len(paired) >= self.min_paired_for_gate:
                if rho_s < self.min_correlation:
                    self.low_fidelity_trusted[tier.level] = False
                    warnings_list.append(
                        f"Tier {tier.name!r} gated off: paired Spearman rho={rho_s:.3f} "
                        f"< min_correlation={self.min_correlation:.2f} (n={len(paired)})."
                    )
                else:
                    self.low_fidelity_trusted[tier.level] = True
            elif not self.enable_correlation_gate:
                self.low_fidelity_trusted[tier.level] = True
        return warnings_list

    def _fit_autoregressive_bridge(self) -> _AutoregressiveBridge | None:
        """Fit Kennedy-O'Hagan linear-residual TanimotoGP from best trusted low tier."""
        hi_obs = self.observations[self.high_tier.level]
        if len(hi_obs) < 4:
            return None

        # Choose the highest-level trusted lower tier with >= 6 paired observations
        best_low: FidelityTierSpec | None = None
        for tier in reversed(self.low_tiers):
            if not self.low_fidelity_trusted.get(tier.level, False):
                continue
            lo_obs = self.observations[tier.level]
            n_paired = sum(1 for s in hi_obs if s in lo_obs)
            if n_paired >= 6:
                best_low = tier
                break

        if best_low is None:
            gp_direct = TanimotoGP(
                kernel="tanimoto",
                n_bits=self.n_bits,
                optimize_hyperparams=True,
                seed=self.seed + len(self.history),
            )
            hi_smis = list(hi_obs.keys())
            gp_direct.fit(hi_smis, [hi_obs[s] for s in hi_smis])
            return _AutoregressiveBridge(
                source_level=None,
                alpha=0.0,
                rho=0.0,
                spearman_rho=0.0,
                residual_gp=gp_direct,
                low_gp=None,
            )

        lo_obs = self.observations[best_low.level]
        paired_smis = [s for s in hi_obs if s in lo_obs]
        x_lo = np.array([lo_obs[s] for s in paired_smis], dtype=np.float64)
        y_hi = np.array([hi_obs[s] for s in paired_smis], dtype=np.float64)

        x_mean = float(np.mean(x_lo))
        y_mean = float(np.mean(y_hi))
        xc = x_lo - x_mean
        yc = y_hi - y_mean
        denom = float(np.dot(xc, xc)) + self.ridge_alpha
        rho_hat = float(np.dot(xc, yc) / max(denom, 1e-12))
        alpha_hat = y_mean - rho_hat * x_mean

        # Fit residual GP on r_i = f_H(x_i) - (alpha_hat + rho_hat * f_L(x_i))
        # For high-tier points without low-tier obs, predict f_L first via low_gp
        low_gp = TanimotoGP(
            kernel="tanimoto",
            n_bits=self.n_bits,
            optimize_hyperparams=True,
            seed=self.seed + 100 + len(self.history),
        )
        lo_smis = list(lo_obs.keys())
        low_gp.fit(lo_smis, [lo_obs[s] for s in lo_smis])

        all_hi_smis = list(hi_obs.keys())
        lo_pred_for_hi = low_gp.predict(all_hi_smis).mean
        residuals: list[float] = []
        for idx, smi in enumerate(all_hi_smis):
            fl_val = lo_obs[smi] if smi in lo_obs else float(lo_pred_for_hi[idx])
            residuals.append(hi_obs[smi] - (alpha_hat + rho_hat * fl_val))

        res_gp = TanimotoGP(
            kernel="tanimoto",
            n_bits=self.n_bits,
            optimize_hyperparams=True,
            seed=self.seed + len(self.history),
        )
        res_gp.fit(all_hi_smis, residuals)

        r_s = self.paired_spearman_rho.get(best_low.level, 0.0)
        return _AutoregressiveBridge(
            source_level=best_low.level,
            alpha=alpha_hat,
            rho=rho_hat,
            spearman_rho=r_s if math.isfinite(r_s) else 0.0,
            residual_gp=res_gp,
            low_gp=low_gp,
            low_obs=dict(lo_obs),
        )

    def predict_high_fidelity(
        self, smiles: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Predict high-fidelity ``(mean, std, valid)``, fusing exact low-tier values."""
        bridge = self._active_bridge
        if bridge is None:
            n = len(smiles)
            return (
                np.zeros(n, dtype=np.float64),
                np.ones(n, dtype=np.float64),
                np.ones(n, dtype=np.bool_),
            )

        res_pred = bridge.residual_gp.predict(smiles)
        if bridge.source_level is None or bridge.low_gp is None:
            return res_pred.mean, res_pred.std, res_pred.valid

        lo_pred = bridge.low_gp.predict(smiles)
        f_l = lo_pred.mean.copy()
        s_l = lo_pred.std.copy()
        for idx, smi in enumerate(smiles):
            if smi in bridge.low_obs:
                f_l[idx] = bridge.low_obs[smi]
                s_l[idx] = 0.0  # Exact low-fidelity measurement -> zero proxy uncertainty!

        mu_h = bridge.alpha + bridge.rho * f_l + res_pred.mean
        sd_h = np.sqrt(res_pred.std**2 + (bridge.rho**2) * (s_l**2))
        return mu_h, sd_h, res_pred.valid & lo_pred.valid

    def _is_gate_eligible(self, smi: str) -> bool:
        for tier in self.low_tiers:
            if smi in self.gate_passed[tier.level] and not self.gate_passed[tier.level][smi]:
                return False
        return True

    def run_round(
        self,
        pool_smiles: Sequence[str],
        *,
        high_batch_size: int = 8,
        low_funnel_multiplier: int = 4,
        remaining_budget: float | None = None,
        xi: float = 0.01,
    ) -> MultiFidelityRoundSummary:
        """Execute one cost-aware multi-fidelity DMTA round with online correlation gating."""
        round_idx = len(self.history) + 1
        unseen_hi = [s for s in pool_smiles if s not in self.high_observations]
        if not unseen_hi:
            raise ValueError("All pool compounds have already been evaluated on the high tier.")

        tier_counts: dict[str, int] = {t.name: 0 for t in self.tiers}
        round_cost = 0.0
        budget_left = float("inf") if remaining_budget is None else float(remaining_budget)

        # Round 1 cold-start: evaluate a diverse seed batch across low + high tiers to calibrate rho
        if len(self.high_observations) < 6:
            low_cost_sum = sum(t.cost for t in self.low_tiers)
            joint_unit_cost = self.high_tier.cost + low_cost_sum
            max_hi_affordable = int(budget_left // self.high_tier.cost)
            max_joint_affordable = int(budget_left // joint_unit_cost)
            if max_joint_affordable >= min(len(unseen_hi), max(1, int(high_batch_size))):
                n_hi = min(len(unseen_hi), max(1, int(high_batch_size)))
            elif max_joint_affordable >= 1:
                n_hi = min(len(unseen_hi), max_joint_affordable)
            else:
                n_hi = min(len(unseen_hi), max(1, int(high_batch_size)), max_hi_affordable)
            if n_hi <= 0:
                raise ValueError(
                    f"Remaining budget {budget_left:.2f} is below high-tier cost "
                    f"{self.high_tier.cost:.2f}."
                )
            seed_idx = select_cold_start(
                unseen_hi, n_hi, strategy="butina_medoids", seed=self.seed + round_idx
            )
            seed_smis = [unseen_hi[i] for i in seed_idx]

            # Evaluate low tiers on the same seed molecules only when strictly affordable
            for tier in self.low_tiers:
                needed_cost = float(len(seed_smis)) * (tier.cost + self.high_tier.cost)
                if budget_left + 1e-9 < needed_cost:
                    continue
                cnt, spent = self._evaluate_tier(tier, seed_smis, round_index=round_idx)
                tier_counts[tier.name] += cnt
                round_cost += spent
                budget_left -= spent

            cnt_h, spent_h = self._evaluate_tier(self.high_tier, seed_smis, round_index=round_idx)
            tier_counts[self.high_tier.name] += cnt_h
            round_cost += spent_h
            warnings_list = self._update_correlation_guardrails()
            self._active_bridge = self._fit_autoregressive_bridge()
        else:
            warnings_list = self._update_correlation_guardrails()
            self._active_bridge = self._fit_autoregressive_bridge()

            # Stage 1: Cost-aware VOI screening on trusted lower tiers
            best_hi = max(self.high_observations.values())
            for tier in self.low_tiers:
                if not self.low_fidelity_trusted.get(tier.level, False):
                    continue
                if tier.cost >= self.high_tier.cost:
                    continue
                unseen_lo = [s for s in unseen_hi if s not in self.observations[tier.level]]
                if not unseen_lo:
                    continue
                # Reserve at least 1 high-fidelity evaluation from remaining budget
                spendable_lo = max(0.0, budget_left - self.high_tier.cost)
                max_lo_budget = int(spendable_lo // tier.cost)
                if max_lo_budget <= 0:
                    continue

                mu_lo_cand, sd_lo_cand, valid_lo = self.predict_high_fidelity(unseen_lo)
                ei_lo = compute_expected_improvement(mu_lo_cand, sd_lo_cand, best_hi, xi=xi)
                log_ei_lo = compute_log_expected_improvement(mu_lo_cand, sd_lo_cand, best_hi, xi=xi)
                rho_cur = self.paired_spearman_rho.get(tier.level, 0.5)
                info_gain = max(0.25, float(rho_cur**2)) if math.isfinite(rho_cur) else 0.5

                # Cost-normalized VOI check: screen lower tier when VOI_L > VOI_H
                max_ei_val = float(np.nanmax(ei_lo[valid_lo])) if bool(valid_lo.any()) else 0.0
                voi_low = (max_ei_val * info_gain) / tier.cost
                voi_high = max_ei_val / self.high_tier.cost
                if voi_low > voi_high or max_ei_val <= 1e-12:
                    n_funnel = min(
                        len(unseen_lo),
                        max(1, int(high_batch_size) * max(1, int(low_funnel_multiplier))),
                        max_lo_budget,
                    )
                    # Rank by EI with log_ei tie-break
                    fin_i = np.flatnonzero(valid_lo & np.isfinite(ei_lo))
                    if len(fin_i) > 0:
                        order = np.lexsort((-log_ei_lo[fin_i], -ei_lo[fin_i]))[:n_funnel]
                        chosen_lo = [unseen_lo[int(fin_i[int(o)])] for o in order]
                        cnt, spent = self._evaluate_tier(tier, chosen_lo, round_index=round_idx)
                        tier_counts[tier.name] += cnt
                        round_cost += spent
                        budget_left -= spent

            # Stage 2: Re-fit bridge with fresh low-tier obs, then promote top high-VOI
            self._active_bridge = self._fit_autoregressive_bridge()
            max_hi_affordable = int(budget_left // self.high_tier.cost)
            n_hi = min(len(unseen_hi), max(1, int(high_batch_size)), max_hi_affordable)
            if n_hi > 0:
                eligible = [s for s in unseen_hi if self._is_gate_eligible(s)]
                cand_pool = eligible if len(eligible) >= n_hi else unseen_hi
                mu_h, sd_h, valid_h = self.predict_high_fidelity(cand_pool)
                ei_h = compute_expected_improvement(mu_h, sd_h, best_hi, xi=xi)
                log_ei_h = compute_log_expected_improvement(mu_h, sd_h, best_hi, xi=xi)
                fin_h = np.flatnonzero(valid_h & np.isfinite(ei_h))
                if len(fin_h) > 0:
                    order_h = np.lexsort((-log_ei_h[fin_h], -ei_h[fin_h]))[:n_hi]
                    chosen_hi = [cand_pool[int(fin_h[int(o)])] for o in order_h]
                    cnt_h, spent_h = self._evaluate_tier(
                        self.high_tier, chosen_hi, round_index=round_idx
                    )
                    tier_counts[self.high_tier.name] += cnt_h
                    round_cost += spent_h
                    budget_left -= spent_h

            warnings_list.extend(self._update_correlation_guardrails())
            self._active_bridge = self._fit_autoregressive_bridge()

        if self.high_observations:
            best_smi, best_val = max(self.high_observations.items(), key=lambda kv: kv[1])
        else:
            best_smi, best_val = "", float("nan")

        summary = MultiFidelityRoundSummary(
            round_index=round_idx,
            tier_evaluated_counts=tier_counts,
            cost_this_round=round_cost,
            cumulative_cost=self.total_cost_spent,
            paired_spearman_rho={
                t.name: (
                    self.paired_spearman_rho[t.level]
                    if math.isfinite(self.paired_spearman_rho[t.level])
                    else None
                )
                for t in self.low_tiers
            },
            low_fidelity_trusted={
                t.name: self.low_fidelity_trusted[t.level] for t in self.low_tiers
            },
            n_high_fidelity_total=len(self.high_observations),
            best_high_fidelity_value=float(best_val),
            best_high_fidelity_smiles=best_smi,
            warnings=tuple(dict.fromkeys(warnings_list)),
        )
        self.history.append(summary)
        return summary

    def run_budgeted_campaign(
        self,
        pool_smiles: Sequence[str],
        *,
        total_cost_budget: float,
        high_batch_size: int = 8,
        low_funnel_multiplier: int = 4,
        max_rounds: int = 50,
    ) -> list[MultiFidelityRoundSummary]:
        """Run multi-fidelity DMTA rounds until ``total_cost_budget`` is exhausted."""
        summaries: list[MultiFidelityRoundSummary] = []
        for _ in range(int(max_rounds)):
            rem = float(total_cost_budget) - self.total_cost_spent
            if rem < self.high_tier.cost:
                break
            unseen = [s for s in pool_smiles if s not in self.high_observations]
            if not unseen:
                break
            summaries.append(
                self.run_round(
                    pool_smiles,
                    high_batch_size=high_batch_size,
                    low_funnel_multiplier=low_funnel_multiplier,
                    remaining_budget=rem,
                )
            )
        return summaries

    def export_top_to_openfe(
        self,
        pocket_pdb: str | Path,
        out_dir: str | Path,
        *,
        top_k: int = 8,
    ) -> Any:
        """Export top-ranked high-fidelity hits to a charge-consistent OpenFE RBFE network."""
        from geneuclidean.interop.openfe_export import export_openfe_rbfe_network

        ranked = sorted(self.high_observations.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        if len(ranked) < 2:
            raise ValueError("Need at least 2 high-fidelity observations to export an RBFE graph.")
        labeled: list[tuple[str, str]] = [
            (f"mf_hit_{i + 1}", smi) for i, (smi, _) in enumerate(ranked)
        ]
        return export_openfe_rbfe_network(
            labeled,
            pocket_pdb=pocket_pdb,
            out_dir=out_dir,
        )


__all__ = [
    "FidelityTierSpec",
    "MultiFidelityDMTA",
    "MultiFidelityObservation",
    "MultiFidelityRoundSummary",
]
