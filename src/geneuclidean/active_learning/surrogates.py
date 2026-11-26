"""Probabilistic molecular surrogates & finite-sample conformal UQ for Active Learning.

Implements three complementary zero-PyTorch uncertainty-aware surrogates registered in
:data:`SURROGATE_REGISTRY`:

1. :class:`TanimotoGP`: Exact Gaussian Process with analytical binary Tanimoto or
   continuous MinMax (Ruzicka) kernel, eigendecomposition-accelerated Type-II marginal
   log-likelihood hyperparameter tuning (:math:`O(n)` per L-BFGS-B step), adaptive
   Cholesky jitter ladder for near-singular molecular Gram matrices, exact posterior
   covariance (:meth:`~TanimotoGP.predict_covariance`), and :math:`O(n^2)` rank-1
   fantasy conditioning (:meth:`~TanimotoGP.condition_on_fantasies`).
2. :class:`EnsembleSurrogate`: Scalable ``RandomForestRegressor`` + bootstrap-bagged
   ``HistGradientBoostingRegressor`` hybrid with exact Law-of-Total-Variance epistemic
   vs. out-of-bag aleatoric uncertainty decomposition
   (:math:`\\sigma_{\\text{total}}^2 = \\sigma_{\\text{epi}}^2 + \\sigma_{\\text{alea}}^2`).
3. :class:`ConformalSurrogate`: Distribution-free finite-sample Split-Conformal wrapper
   supporting ``"absolute"`` (homoscedastic), ``"normalized"`` (heteroscedastic adaptive
   width), and ``"weighted"`` (Tibshirani et al. 2019 covariate-shift Tanimoto-weighted
   with Kish :math:`N_{\\text{eff}}` diagnostics) modes, scaffold-grouped calibration
   splitting, and leak-free out-of-fold CV+ fallback when ``n_obs < min_split_n``.

Note:
    Standalone Evidential Deep Learning (EDL) is deliberately excluded following
    Bengs et al. (NeurIPS 2023), which proved EDL epistemic uncertainty fails to vanish
    as data grows and yields overconfident OOD estimates on novel scaffolds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from geneuclidean.active_learning.featurizers import (
    FeaturizerKind,
    compute_murcko_scaffold_cached,
    featurize_pool,
    minmax_kernel,
    tanimoto_kernel,
)
from geneuclidean.registry import Registry

KernelKind = Literal["tanimoto", "minmax"]
ConformalMode = Literal[
    "absolute",
    "homoscedastic",
    "normalized",
    "weighted",
    "weighted_covariate_shift",
]

_TWO_PI: float = 2.0 * math.pi


@dataclass(frozen=True)
class SurrogatePrediction:
    """Vectorized surrogate prediction with epistemic/aleatoric & conformal intervals."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    epistemic_std: NDArray[np.float64]
    aleatoric_std: NDArray[np.float64]
    valid: NDArray[np.bool_]
    interval_lo: NDArray[np.float64] | None = None
    interval_hi: NDArray[np.float64] | None = None
    alpha: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SurrogateProtocol(Protocol):
    """Structural typing protocol (PEP 544) for pluggable Active Learning surrogates."""

    @property
    def name(self) -> str:
        """Canonical identifier of the surrogate model."""
        ...

    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> dict[str, float]:
        """Fit the surrogate on ``(smiles, y)`` and return scalar training diagnostics."""
        ...

    def predict(self, smiles: Sequence[str]) -> SurrogatePrediction:
        """Predict posterior mean and uncertainty across ``smiles``."""
        ...


# ---------------------------------------------------------------------------
# 1. Exact Analytical Tanimoto / MinMax Gaussian Process
# ---------------------------------------------------------------------------


class TanimotoGP:
    """Exact analytical Gaussian Process with PSD Tanimoto or MinMax molecular kernel.

    Target values are internally standardized :math:`\\tilde{y} = (y - \\bar{y}) / s_y`
    so the empirical prior mean is :math:`m_0 = \\bar{y}` and out-of-domain molecules
    orthogonal to all training fingerprints revert smoothly to
    :math:`(\\mu = \\bar{y},\\; \\sigma_{\\text{epi}} = s_y \\sigma_f)`.
    """

    def __init__(
        self,
        *,
        kernel: KernelKind = "tanimoto",
        n_bits: int = 2048,
        radius: int = 2,
        use_chirality: bool = True,
        signal_variance: float = 1.0,
        noise_variance: float = 0.05,
        optimize_hyperparams: bool = True,
        jitter: float = 1e-6,
        seed: int = 42,
    ) -> None:
        if kernel not in ("tanimoto", "minmax"):
            raise ValueError(f"kernel must be 'tanimoto' or 'minmax', got {kernel!r}.")
        if signal_variance <= 0.0:
            raise ValueError(f"signal_variance must be > 0, got {signal_variance}.")
        if noise_variance <= 0.0:
            raise ValueError(f"noise_variance must be > 0, got {noise_variance}.")
        self.kernel: KernelKind = kernel
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        self.use_chirality = bool(use_chirality)
        self.init_signal_variance = float(signal_variance)
        self.init_noise_variance = float(noise_variance)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)
        self.optimize_hyperparams = bool(optimize_hyperparams)
        self.jitter = float(jitter)
        self.seed = int(seed)

        # Fitted internal state
        self._x_train: NDArray[np.float64] | None = None
        self._y_train_norm: NDArray[np.float64] | None = None
        self._l_factor: NDArray[np.float64] | None = None
        self._alpha: NDArray[np.float64] | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._jitter_used: float = float(jitter)
        self._log_marginal_likelihood: float = float("nan")

    @property
    def name(self) -> str:
        return "tanimoto_gp" if self.kernel == "tanimoto" else "minmax_gp"

    @property
    def featurizer_kind(self) -> FeaturizerKind:
        return "ecfp4_binary" if self.kernel == "tanimoto" else "ecfp4_count"

    @property
    def is_fitted(self) -> bool:
        return self._l_factor is not None and self._x_train is not None

    @property
    def n_train(self) -> int:
        return 0 if self._x_train is None else int(self._x_train.shape[0])

    @property
    def noise_variance_raw(self) -> float:
        """Observation noise variance in raw (unstandardized) target units."""
        return float((self._y_std**2) * self.noise_variance)

    def _compute_kernel(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if self.kernel == "minmax":
            return minmax_kernel(X, Y)
        return tanimoto_kernel(X, Y)

    def _clone_unfitted(self) -> TanimotoGP:
        return TanimotoGP(
            kernel=self.kernel,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
            signal_variance=self.init_signal_variance,
            noise_variance=self.init_noise_variance,
            optimize_hyperparams=self.optimize_hyperparams,
            jitter=self.jitter,
            seed=self.seed,
        )

    def _optimize_mll_spectral(
        self, k0: NDArray[np.float64], y_norm: NDArray[np.float64]
    ) -> tuple[float, float]:
        """Maximize exact log marginal likelihood in O(n) per step via eigh(K0)."""
        eigvals, eigvecs = eigh(k0, check_finite=False)
        lam = np.maximum(np.asarray(eigvals, dtype=np.float64), 0.0)
        proj = np.asarray(eigvecs.T @ y_norm, dtype=np.float64)
        proj_sq = proj * proj
        n = float(len(y_norm))
        const = 0.5 * n * math.log(_TWO_PI)

        def _nlml_and_grad(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            sf2 = math.exp(float(theta[0]))
            sn2 = math.exp(float(theta[1]))
            d = np.maximum(sf2 * lam + sn2, 1e-12)
            inv_d = 1.0 / d
            nlml = float(0.5 * np.sum(proj_sq * inv_d + np.log(d)) + const)
            coeff = inv_d - proj_sq * (inv_d * inv_d)
            g_sf2 = float(0.5 * sf2 * np.sum(lam * coeff))
            g_sn2 = float(0.5 * sn2 * np.sum(coeff))
            return nlml, np.array([g_sf2, g_sn2], dtype=np.float64)

        bounds = [(math.log(1e-2), math.log(1e2)), (math.log(1e-4), math.log(2.0))]
        starts = (
            (self.init_signal_variance, self.init_noise_variance),
            (0.8, 0.15),
            (1.5, 0.01),
        )
        best_val = float("inf")
        best_pair = (self.init_signal_variance, self.init_noise_variance)
        for s0, n0 in starts:
            x0 = np.array(
                [
                    math.log(max(1e-2, min(1e2, s0))),
                    math.log(max(1e-4, min(2.0, n0))),
                ],
                dtype=np.float64,
            )
            res = minimize(
                _nlml_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 60, "ftol": 1e-9},
            )
            if math.isfinite(float(res.fun)) and float(res.fun) < best_val:
                best_val = float(res.fun)
                best_pair = (math.exp(float(res.x[0])), math.exp(float(res.x[1])))
        return best_pair

    def _factorize_with_jitter(
        self,
        k_base: NDArray[np.float64],
        sf2: float,
        sn2: float,
    ) -> tuple[NDArray[np.float64], float]:
        """Cholesky-factorize ``sf2 * k_base + (sn2 + eps) * I`` via an adaptive ladder."""
        n = int(k_base.shape[0])
        ladder = sorted({self.jitter, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2})
        ladder = [eps for eps in ladder if eps >= self.jitter]
        if not ladder:
            ladder = [self.jitter, 1e-2]

        last_err: Exception | None = None
        for eps in ladder:
            k_y = sf2 * k_base
            k_y[np.diag_indices(n)] += sn2 + eps
            try:
                l_fac, _ = cho_factor(k_y, lower=True, check_finite=False)
                return np.tril(np.asarray(l_fac, dtype=np.float64)), float(eps)
            except np.linalg.LinAlgError as exc:
                last_err = exc
        raise RuntimeError(
            f"TanimotoGP Cholesky failed across jitter ladder {ladder}."
        ) from last_err

    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> dict[str, float]:
        """Fit exact Tanimoto/MinMax GP on ``(smiles, y)`` with optional MLL tuning."""
        if len(smiles) != len(y):
            raise ValueError(f"Length mismatch: {len(smiles)} smiles vs {len(y)} targets.")
        x_all, valid_smi = featurize_pool(
            smiles,
            kind=self.featurizer_kind,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
        )
        y_all = np.asarray(y, dtype=np.float64)
        mask = valid_smi & np.isfinite(y_all)
        n_valid = int(mask.sum())
        if n_valid == 0:
            raise ValueError("TanimotoGP.fit requires at least 1 valid molecule with finite y.")

        x_tr = x_all[mask]
        y_tr = y_all[mask]
        y_mean = float(np.mean(y_tr))
        y_std_raw = float(np.std(y_tr))
        y_std = y_std_raw if y_std_raw > 1e-8 else 1.0
        y_norm = (y_tr - y_mean) / y_std

        k0 = self._compute_kernel(x_tr)
        if self.optimize_hyperparams and n_valid >= 4 and y_std_raw > 1e-8:
            sf2, sn2 = self._optimize_mll_spectral(k0, y_norm)
        else:
            sf2, sn2 = self.init_signal_variance, self.init_noise_variance

        l_fac, eps_used = self._factorize_with_jitter(k0, sf2, sn2)
        alpha = np.asarray(cho_solve((l_fac, True), y_norm, check_finite=False), dtype=np.float64)
        lml = float(
            -0.5 * float(np.dot(y_norm, alpha))
            - float(np.sum(np.log(np.maximum(np.diag(l_fac), 1e-300))))
            - 0.5 * n_valid * math.log(_TWO_PI)
        )

        self._x_train = x_tr
        self._y_train_norm = y_norm
        self._l_factor = l_fac
        self._alpha = alpha
        self._y_mean = y_mean
        self._y_std = y_std
        self.signal_variance = float(sf2)
        self.noise_variance = float(sn2)
        self._jitter_used = float(eps_used)
        self._log_marginal_likelihood = lml

        return {
            "n_train": float(n_valid),
            "signal_variance": round(self.signal_variance, 6),
            "noise_variance": round(self.noise_variance, 6),
            "log_marginal_likelihood": round(lml, 4),
            "jitter_used": float(eps_used),
        }

    def predict(self, smiles: Sequence[str]) -> SurrogatePrediction:
        """Predict posterior mean, epistemic std, aleatoric std, and total std."""
        n = len(smiles)
        if n == 0:
            empty_f = np.zeros(0, dtype=np.float64)
            return SurrogatePrediction(
                mean=empty_f,
                std=empty_f.copy(),
                epistemic_std=empty_f.copy(),
                aleatoric_std=empty_f.copy(),
                valid=np.zeros(0, dtype=np.bool_),
                metadata={"surrogate": self.name},
            )

        x_q, valid = featurize_pool(
            smiles,
            kind=self.featurizer_kind,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
        )
        mean = np.full(n, float("nan"), dtype=np.float64)
        std_tot = np.full(n, float("nan"), dtype=np.float64)
        std_epi = np.full(n, float("nan"), dtype=np.float64)
        std_ale = np.full(n, float("nan"), dtype=np.float64)

        if not bool(valid.any()):
            return SurrogatePrediction(
                mean=mean,
                std=std_tot,
                epistemic_std=std_epi,
                aleatoric_std=std_ale,
                valid=valid,
                metadata={"surrogate": self.name},
            )

        if not self.is_fitted:
            epi_prior = math.sqrt(self.signal_variance)
            ale_prior = math.sqrt(self.noise_variance)
            tot_prior = math.sqrt(self.signal_variance + self.noise_variance)
            mean[valid] = 0.0
            std_epi[valid] = epi_prior
            std_ale[valid] = ale_prior
            std_tot[valid] = tot_prior
            return SurrogatePrediction(
                mean=mean,
                std=std_tot,
                epistemic_std=std_epi,
                aleatoric_std=std_ale,
                valid=valid,
                metadata={"surrogate": self.name, "fitted": False},
            )

        assert self._x_train is not None
        assert self._l_factor is not None
        assert self._alpha is not None

        k_star = self.signal_variance * self._compute_kernel(x_q[valid], self._x_train)
        mu_norm = k_star @ self._alpha
        v_mat = np.asarray(
            solve_triangular(self._l_factor, k_star.T, lower=True, check_finite=False),
            dtype=np.float64,
        )
        var_epi_norm = np.maximum(0.0, self.signal_variance - np.einsum("ij,ij->j", v_mat, v_mat))

        epi_vals = self._y_std * np.sqrt(var_epi_norm)
        ale_val = self._y_std * math.sqrt(self.noise_variance)
        tot_vals = np.sqrt(epi_vals * epi_vals + ale_val * ale_val)

        mean[valid] = self._y_mean + self._y_std * mu_norm
        std_epi[valid] = epi_vals
        std_ale[valid] = ale_val
        std_tot[valid] = tot_vals

        return SurrogatePrediction(
            mean=mean,
            std=std_tot,
            epistemic_std=std_epi,
            aleatoric_std=std_ale,
            valid=valid,
            metadata={
                "surrogate": self.name,
                "kernel": self.kernel,
                "signal_variance": self.signal_variance,
                "noise_variance": self.noise_variance,
                "n_train": self.n_train,
            },
        )

    def predict_covariance(
        self, smiles: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Return exact latent posterior ``(mean [N], cov [N, N], valid [N])`` in raw units."""
        n = len(smiles)
        if n == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                np.zeros(0, dtype=np.bool_),
            )

        x_q, valid = featurize_pool(
            smiles,
            kind=self.featurizer_kind,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
        )
        mean = np.full(n, float("nan"), dtype=np.float64)
        cov = np.zeros((n, n), dtype=np.float64)
        if not bool(valid.any()):
            return mean, cov, valid

        xv = x_q[valid]
        k_qq = self.signal_variance * self._compute_kernel(xv)
        if not self.is_fitted:
            mean[valid] = 0.0
            idx = np.flatnonzero(valid)
            cov[np.ix_(idx, idx)] = k_qq
            return mean, cov, valid

        assert self._x_train is not None
        assert self._l_factor is not None
        assert self._alpha is not None

        k_star = self.signal_variance * self._compute_kernel(xv, self._x_train)
        mu_norm = k_star @ self._alpha
        v_mat = np.asarray(
            solve_triangular(self._l_factor, k_star.T, lower=True, check_finite=False),
            dtype=np.float64,
        )
        cov_norm = k_qq - (v_mat.T @ v_mat)
        cov_norm = 0.5 * (cov_norm + cov_norm.T)
        # Guard numerical negative diagonal entries from floating-point cancellation
        diag_idx = np.diag_indices(cov_norm.shape[0])
        cov_norm[diag_idx] = np.maximum(cov_norm[diag_idx], 0.0)

        mean[valid] = self._y_mean + self._y_std * mu_norm
        idx = np.flatnonzero(valid)
        cov[np.ix_(idx, idx)] = (self._y_std**2) * cov_norm
        return mean, cov, valid

    def condition_on_fantasies(
        self,
        fantasy_smiles: Sequence[str],
        fantasy_y: Sequence[float] | None = None,
    ) -> TanimotoGP:
        """Return a new GP conditioned on fantasy/pending observations via O(n^2) rank-1 Cholesky.

        Hyperparameters ``(signal_variance, noise_variance, y_mean, y_std)`` are kept fixed,
        providing exact Kriging Believer / asynchronous pending-batch variance deflation
        without an :math:`O(n^3)` refit.
        """
        if not self.is_fitted:
            raise RuntimeError("TanimotoGP must be fitted before conditioning on fantasies.")
        if not fantasy_smiles:
            return self
        if fantasy_y is not None and len(fantasy_y) != len(fantasy_smiles):
            raise ValueError(
                f"Length mismatch: {len(fantasy_smiles)} fantasy_smiles vs {len(fantasy_y)} y."
            )

        x_fan, valid_fan = featurize_pool(
            fantasy_smiles,
            kind=self.featurizer_kind,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
        )
        if fantasy_y is None:
            pred_fan = self.predict(fantasy_smiles)
            y_fan_raw = pred_fan.mean
        else:
            y_fan_raw = np.asarray(fantasy_y, dtype=np.float64)

        keep = valid_fan & np.isfinite(y_fan_raw)
        if not bool(keep.any()):
            return self

        x_add = x_fan[keep]
        y_add_norm = (y_fan_raw[keep] - self._y_mean) / self._y_std

        assert self._x_train is not None
        assert self._y_train_norm is not None
        assert self._l_factor is not None

        n_old = int(self._x_train.shape[0])
        m = int(x_add.shape[0])
        x_ext = np.vstack([self._x_train, x_add])
        y_ext = np.concatenate([self._y_train_norm, y_add_norm])

        # Incremental rank-1 Cholesky append: O(n^2) per fantasy molecule
        l_ext = np.zeros((n_old + m, n_old + m), dtype=np.float64)
        l_ext[:n_old, :n_old] = self._l_factor
        diag_const = self.signal_variance + self.noise_variance + self._jitter_used

        for step in range(m):
            cur_n = n_old + step
            k_vec = (
                self.signal_variance * self._compute_kernel(x_ext[:cur_n], x_ext[cur_n : cur_n + 1])
            ).ravel()
            ell = np.asarray(
                solve_triangular(l_ext[:cur_n, :cur_n], k_vec, lower=True, check_finite=False),
                dtype=np.float64,
            )
            rem = diag_const - float(np.dot(ell, ell))
            l_ext[cur_n, :cur_n] = ell
            l_ext[cur_n, cur_n] = math.sqrt(max(rem, self._jitter_used))

        alpha_ext = np.asarray(
            cho_solve((l_ext, True), y_ext, check_finite=False), dtype=np.float64
        )

        cloned = self._clone_unfitted()
        cloned.signal_variance = self.signal_variance
        cloned.noise_variance = self.noise_variance
        cloned._x_train = x_ext
        cloned._y_train_norm = y_ext
        cloned._l_factor = l_ext
        cloned._alpha = alpha_ext
        cloned._y_mean = self._y_mean
        cloned._y_std = self._y_std
        cloned._jitter_used = self._jitter_used
        cloned._log_marginal_likelihood = self._log_marginal_likelihood
        return cloned


# ---------------------------------------------------------------------------
# 2. Scalable HGB + RandomForest Ensemble with Epistemic/Aleatoric Split
# ---------------------------------------------------------------------------


class EnsembleSurrogate:
    """RandomForest + bootstrap-bagged HGB surrogate with exact variance decomposition.

    Combines 50% ``RandomForestRegressor`` and 50% bootstrap-bagged
    ``HistGradientBoostingRegressor`` predictions. Applies the Law of Total Variance
    for the epistemic component and out-of-bag residual excess MSE for the aleatoric
    component so
    :math:`\\sigma_{\\text{total}}^2 = \\sigma_{\\text{epi}}^2 + \\sigma_{\\text{alea}}^2`
    holds identically.
    """

    def __init__(
        self,
        *,
        n_rf_estimators: int = 64,
        n_hgb_members: int = 5,
        featurizer: FeaturizerKind = "ecfp4_physchem",
        n_bits: int = 1024,
        radius: int = 2,
        use_chirality: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_rf_estimators = max(4, int(n_rf_estimators))
        self.n_hgb_members = max(2, int(n_hgb_members))
        self.featurizer: FeaturizerKind = featurizer
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        self.use_chirality = bool(use_chirality)
        self.seed = int(seed)

        self._rf: RandomForestRegressor | None = None
        self._hgbs: list[HistGradientBoostingRegressor] = []
        self._aleatoric_var: float = 0.05
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._n_train: int = 0

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def is_fitted(self) -> bool:
        return self._rf is not None and len(self._hgbs) > 0

    def _clone_unfitted(self) -> EnsembleSurrogate:
        return EnsembleSurrogate(
            n_rf_estimators=self.n_rf_estimators,
            n_hgb_members=self.n_hgb_members,
            featurizer=self.featurizer,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
            seed=self.seed,
        )

    def _predict_components(
        self, x_mat: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, epistemic_var)`` for valid feature rows."""
        assert self._rf is not None and len(self._hgbs) > 0
        x32 = x_mat.astype(np.float32, copy=False)
        rf_preds = np.stack(
            [np.asarray(est.predict(x32), dtype=np.float64) for est in self._rf.estimators_],
            axis=0,
        )
        hgb_preds = np.stack(
            [np.asarray(m.predict(x32), dtype=np.float64) for m in self._hgbs],
            axis=0,
        )
        mu_rf = np.mean(rf_preds, axis=0)
        var_rf = np.var(rf_preds, axis=0)
        mu_hgb = np.mean(hgb_preds, axis=0)
        var_hgb = np.var(hgb_preds, axis=0)

        mean = 0.5 * (mu_rf + mu_hgb)
        # Exact Law of Total Variance for an equal-weight two-family mixture
        epi_var = np.maximum(0.0, 0.5 * var_rf + 0.5 * var_hgb + 0.25 * (mu_rf - mu_hgb) ** 2)
        return mean, epi_var

    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> dict[str, float]:
        """Fit RF + bootstrap-bagged HGB ensemble and estimate OOB aleatoric noise."""
        if len(smiles) != len(y):
            raise ValueError(f"Length mismatch: {len(smiles)} smiles vs {len(y)} targets.")
        x_all, valid_smi = featurize_pool(
            smiles,
            kind=self.featurizer,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
        )
        y_all = np.asarray(y, dtype=np.float64)
        mask = valid_smi & np.isfinite(y_all)
        n = int(mask.sum())
        if n == 0:
            raise ValueError("EnsembleSurrogate.fit requires >= 1 valid molecule with finite y.")

        x_tr = x_all[mask].astype(np.float32, copy=False)
        y_tr = y_all[mask]
        self._y_mean = float(np.mean(y_tr))
        self._y_std = float(np.std(y_tr)) if float(np.std(y_tr)) > 1e-8 else 1.0
        self._n_train = n

        use_oob = n >= 16
        rf = RandomForestRegressor(
            n_estimators=self.n_rf_estimators,
            max_features="sqrt",
            bootstrap=True,
            oob_score=use_oob,
            random_state=self.seed,
            n_jobs=1,
        )
        rf.fit(x_tr, y_tr)
        self._rf = rf

        rng = np.random.default_rng(self.seed)
        min_leaf = max(1, min(8, n // 5))
        max_depth = 6 if n >= 20 else 3
        hgbs: list[HistGradientBoostingRegressor] = []
        oob_sum = np.zeros(n, dtype=np.float64)
        oob_cnt = np.zeros(n, dtype=np.int64)

        for m_idx in range(self.n_hgb_members):
            if n >= 6:
                boot_idx = rng.integers(0, n, size=n)
                if len(np.unique(boot_idx)) < 2:
                    boot_idx = np.arange(n)
            else:
                boot_idx = np.arange(n)
            hgb = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.06,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=self.seed + 17 * (m_idx + 1),
            )
            hgb.fit(x_tr[boot_idx], y_tr[boot_idx])
            hgbs.append(hgb)

            if n >= 8:
                in_bag = np.zeros(n, dtype=np.bool_)
                in_bag[boot_idx] = True
                oob_mask = ~in_bag
                if bool(oob_mask.any()):
                    oob_sum[oob_mask] += np.asarray(hgb.predict(x_tr[oob_mask]), dtype=np.float64)
                    oob_cnt[oob_mask] += 1

        self._hgbs = hgbs

        # Estimate aleatoric variance from OOB MSE minus mean in-bag epistemic variance
        _, in_epi_var = self._predict_components(x_all[mask])
        mean_epi_var = float(np.mean(in_epi_var))
        oob_mse = float("nan")

        if use_oob and hasattr(rf, "oob_prediction_"):
            rf_oob = np.asarray(rf.oob_prediction_, dtype=np.float64)
            has_hgb_oob = oob_cnt > 0
            combined_oob = rf_oob.copy()
            if bool(has_hgb_oob.any()):
                hgb_oob = oob_sum[has_hgb_oob] / oob_cnt[has_hgb_oob]
                combined_oob[has_hgb_oob] = 0.5 * (rf_oob[has_hgb_oob] + hgb_oob)
            fin = np.isfinite(combined_oob)
            if bool(fin.any()):
                oob_mse = float(np.mean((y_tr[fin] - combined_oob[fin]) ** 2))
                self._aleatoric_var = max(1e-4, oob_mse - mean_epi_var)
            else:
                self._aleatoric_var = max(1e-4, 0.1 * (self._y_std**2))
        else:
            self._aleatoric_var = max(1e-4, 0.1 * (self._y_std**2))

        return {
            "n_train": float(n),
            "oob_mse": round(oob_mse, 6) if math.isfinite(oob_mse) else float("nan"),
            "mean_epistemic_std": round(math.sqrt(max(0.0, mean_epi_var)), 6),
            "aleatoric_std": round(math.sqrt(self._aleatoric_var), 6),
        }

    def predict(self, smiles: Sequence[str]) -> SurrogatePrediction:
        """Predict ensemble mean and decomposed epistemic / aleatoric / total std."""
        n = len(smiles)
        if n == 0:
            empty_f = np.zeros(0, dtype=np.float64)
            return SurrogatePrediction(
                mean=empty_f,
                std=empty_f.copy(),
                epistemic_std=empty_f.copy(),
                aleatoric_std=empty_f.copy(),
                valid=np.zeros(0, dtype=np.bool_),
                metadata={"surrogate": self.name},
            )

        x_q, valid = featurize_pool(
            smiles,
            kind=self.featurizer,
            n_bits=self.n_bits,
            radius=self.radius,
            use_chirality=self.use_chirality,
        )
        mean = np.full(n, float("nan"), dtype=np.float64)
        std_tot = np.full(n, float("nan"), dtype=np.float64)
        std_epi = np.full(n, float("nan"), dtype=np.float64)
        std_ale = np.full(n, float("nan"), dtype=np.float64)

        if not bool(valid.any()):
            return SurrogatePrediction(
                mean=mean,
                std=std_tot,
                epistemic_std=std_epi,
                aleatoric_std=std_ale,
                valid=valid,
                metadata={"surrogate": self.name},
            )

        if not self.is_fitted:
            mean[valid] = 0.0
            std_epi[valid] = 1.0
            std_ale[valid] = math.sqrt(self._aleatoric_var)
            std_tot[valid] = math.sqrt(1.0 + self._aleatoric_var)
            return SurrogatePrediction(
                mean=mean,
                std=std_tot,
                epistemic_std=std_epi,
                aleatoric_std=std_ale,
                valid=valid,
                metadata={"surrogate": self.name, "fitted": False},
            )

        mu_v, epi_var_v = self._predict_components(x_q[valid])
        epi_v = np.sqrt(epi_var_v)
        ale_v = math.sqrt(self._aleatoric_var)
        tot_v = np.sqrt(epi_var_v + self._aleatoric_var)

        mean[valid] = mu_v
        std_epi[valid] = epi_v
        std_ale[valid] = ale_v
        std_tot[valid] = tot_v
        return SurrogatePrediction(
            mean=mean,
            std=std_tot,
            epistemic_std=std_epi,
            aleatoric_std=std_ale,
            valid=valid,
            metadata={"surrogate": self.name, "n_train": self._n_train},
        )


# ---------------------------------------------------------------------------
# 3. Distribution-Free Finite-Sample Split-Conformal & Weighted Conformal UQ
# ---------------------------------------------------------------------------


def _canonical_conformal_mode(mode: ConformalMode) -> Literal["absolute", "normalized", "weighted"]:
    if mode in ("absolute", "homoscedastic"):
        return "absolute"
    if mode == "normalized":
        return "normalized"
    if mode in ("weighted", "weighted_covariate_shift"):
        return "weighted"
    raise ValueError(f"Unknown conformal mode {mode!r}.")


def _finite_sample_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Exact finite-sample split-conformal quantile ``ceil((n + 1)(1 - alpha))``-th order stat."""
    n = len(scores)
    if n == 0:
        return 1.0
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    sorted_scores = np.sort(scores)
    if rank >= n:
        return float(sorted_scores[-1])
    if rank <= 1:
        return float(sorted_scores[0])
    return float(sorted_scores[rank - 1])


def _clone_surrogate(surrogate: SurrogateProtocol, seed_offset: int = 0) -> SurrogateProtocol:
    """Create a fresh unfitted copy of ``surrogate`` for leak-free CV+ folds."""
    if isinstance(surrogate, TanimotoGP):
        clone = surrogate._clone_unfitted()
        clone.seed = surrogate.seed + seed_offset
        return clone
    if isinstance(surrogate, EnsembleSurrogate):
        clone_e = surrogate._clone_unfitted()
        clone_e.seed = surrogate.seed + seed_offset
        return clone_e
    if hasattr(surrogate, "_clone_unfitted"):
        return surrogate._clone_unfitted()  # type: ignore[no-any-return]
    raise TypeError(
        f"Cannot clone custom surrogate {type(surrogate).__name__} for CV+; "
        "implement _clone_unfitted() or call fit_calibrated() with an explicit split."
    )


def _split_train_cal_indices(
    smiles: Sequence[str],
    *,
    cal_fraction: float,
    scaffold_split: bool,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Partition ``range(len(smiles))`` into strictly disjoint ``(I_train, I_cal)``."""
    n = len(smiles)
    target_cal = max(4, min(n - 4, int(round(cal_fraction * n))))
    rng = np.random.default_rng(seed)

    if scaffold_split:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, smi in enumerate(smiles):
            groups[compute_murcko_scaffold_cached(smi)].append(idx)
        if len(groups) >= 5:
            scaf_keys = list(groups.keys())
            order = rng.permutation(len(scaf_keys))
            cal_idx: list[int] = []
            tr_idx: list[int] = []
            for pos in order:
                members = groups[scaf_keys[int(pos)]]
                if len(cal_idx) < target_cal and (n - (len(cal_idx) + len(members))) >= 4:
                    cal_idx.extend(members)
                else:
                    tr_idx.extend(members)
            if len(cal_idx) >= 4 and len(tr_idx) >= 4:
                return sorted(tr_idx), sorted(cal_idx)

    perm = rng.permutation(n).tolist()
    return sorted(perm[target_cal:]), sorted(perm[:target_cal])


class ConformalSurrogate:
    """Distribution-free Split-Conformal & Covariate-Shift Weighted UQ wrapper.

    Supports three calibration modes:
    - ``"absolute"`` / ``"homoscedastic"``: constant width ``mean ± q_hat``.
    - ``"normalized"`` *(default)*: adaptive width ``mean ± q_hat * max(std, min_sigma)``
      so out-of-domain scaffolds automatically receive wider intervals.
    - ``"weighted"`` / ``"weighted_covariate_shift"``: Tibshirani et al. (2019)
      Tanimoto-similarity-weighted normalized conformal intervals with Kish effective
      sample size :math:`N_{\\text{eff}}` telemetry.

    Enforces strict ``I_train ∩ I_cal == ∅`` (never evaluating nonconformity residuals on
    in-sample training points). When ``len(smiles) < min_split_n`` inside early DMTA rounds,
    automatically uses leak-free out-of-fold CV+ calibration (``calibration_mode="cv_plus"``).
    """

    def __init__(
        self,
        base_surrogate: SurrogateProtocol,
        *,
        alpha: float = 0.10,
        mode: ConformalMode = "normalized",
        cal_fraction: float = 0.20,
        scaffold_split_cal: bool = True,
        weight_gamma: float = 5.0,
        min_sigma: float = 1e-3,
        min_split_n: int = 40,
        seed: int = 42,
    ) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")
        self.base_surrogate = base_surrogate
        self.alpha = float(alpha)
        self.mode: Literal["absolute", "normalized", "weighted"] = _canonical_conformal_mode(mode)
        self.cal_fraction = float(cal_fraction)
        self.scaffold_split_cal = bool(scaffold_split_cal)
        self.weight_gamma = float(weight_gamma)
        self.min_sigma = float(min_sigma)
        self.min_split_n = max(12, int(min_split_n))
        self.seed = int(seed)

        self.q_hat: float = 1.0
        self.cal_size: int = 0
        self.calibration_mode: Literal["split", "cv_plus", "unfitted"] = "unfitted"
        self._cal_scores_sorted: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._cal_fps_sorted: NDArray[np.float64] = np.zeros((0, 1024), dtype=np.float64)
        self._cal_fp_bits: int = 1024

    @property
    def name(self) -> str:
        return f"conformal_{self.base_surrogate.name}"

    def _compute_scores(
        self,
        y_cal: NDArray[np.float64],
        pred_cal: SurrogatePrediction,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        ok = pred_cal.valid & np.isfinite(pred_cal.mean) & np.isfinite(y_cal)
        residuals = np.abs(y_cal[ok] - pred_cal.mean[ok])
        if self.mode == "absolute":
            return residuals, ok
        denom = np.maximum(np.nan_to_num(pred_cal.std[ok], nan=self.min_sigma), self.min_sigma)
        return residuals / denom, ok

    def _store_calibration(
        self,
        cal_smiles: Sequence[str],
        scores: NDArray[np.float64],
        *,
        cal_mode: Literal["split", "cv_plus"],
    ) -> None:
        self.cal_size = int(len(scores))
        self.calibration_mode = cal_mode
        if self.cal_size == 0:
            self.q_hat = 1.0
            self._cal_scores_sorted = np.zeros(0, dtype=np.float64)
            self._cal_fps_sorted = np.zeros((0, self._cal_fp_bits), dtype=np.float64)
            return

        order = np.argsort(scores, kind="mergesort")
        self._cal_scores_sorted = np.asarray(scores[order], dtype=np.float64)
        self.q_hat = _finite_sample_quantile(self._cal_scores_sorted, self.alpha)

        if self.mode == "weighted":
            ordered_smis = [cal_smiles[int(i)] for i in order]
            fps, _ = featurize_pool(
                ordered_smis, kind="ecfp4_binary", n_bits=self._cal_fp_bits, radius=2
            )
            self._cal_fps_sorted = fps

    def fit_calibrated(
        self,
        smiles_train: Sequence[str],
        y_train: Sequence[float],
        smiles_cal: Sequence[str],
        y_cal: Sequence[float],
    ) -> dict[str, float]:
        """Fit ``base_surrogate`` strictly on train and calibrate on disjoint calibration data."""
        base_stats = self.base_surrogate.fit(smiles_train, y_train)
        pred_cal = self.base_surrogate.predict(smiles_cal)
        y_cal_arr = np.asarray(y_cal, dtype=np.float64)
        scores, ok = self._compute_scores(y_cal_arr, pred_cal)
        kept_cal_smis = [s for s, flag in zip(smiles_cal, ok, strict=True) if bool(flag)]
        self._store_calibration(kept_cal_smis, scores, cal_mode="split")
        return {
            **base_stats,
            "n_cal": float(self.cal_size),
            "q_hat": round(self.q_hat, 6),
            "target_coverage": round(1.0 - self.alpha, 4),
        }

    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> dict[str, float]:
        """Fit and calibrate with zero leakage (split-conformal or early-round OOF CV+)."""
        if len(smiles) != len(y):
            raise ValueError(f"Length mismatch: {len(smiles)} smiles vs {len(y)} targets.")
        n = len(smiles)
        if n < 6:
            base_stats = self.base_surrogate.fit(smiles, y)
            self.q_hat = 1.6448536 if self.mode != "absolute" else 1.0
            self.cal_size = 0
            self.calibration_mode = "cv_plus"
            return {**base_stats, "n_cal": 0.0, "q_hat": round(self.q_hat, 6)}

        if n >= self.min_split_n:
            tr_idx, cal_idx = _split_train_cal_indices(
                smiles,
                cal_fraction=self.cal_fraction,
                scaffold_split=self.scaffold_split_cal,
                seed=self.seed,
            )
            return self.fit_calibrated(
                [smiles[i] for i in tr_idx],
                [float(y[i]) for i in tr_idx],
                [smiles[i] for i in cal_idx],
                [float(y[i]) for i in cal_idx],
            )

        # Leak-free K-fold OOF CV+ calibration when n < min_split_n
        k_folds = min(5, max(3, n // 3))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        folds = [arr.tolist() for arr in np.array_split(perm, k_folds)]

        oof_scores: list[float] = []
        oof_smiles: list[str] = []
        for f_idx, val_idx in enumerate(folds):
            val_set = set(val_idx)
            tr_idx = [i for i in range(n) if i not in val_set]
            if len(tr_idx) < 2 or not val_idx:
                continue
            fold_model = _clone_surrogate(self.base_surrogate, seed_offset=f_idx + 1)
            fold_model.fit([smiles[i] for i in tr_idx], [float(y[i]) for i in tr_idx])
            val_smis = [smiles[i] for i in val_idx]
            val_y = np.array([float(y[i]) for i in val_idx], dtype=np.float64)
            pred_val = fold_model.predict(val_smis)
            sc, ok = self._compute_scores(val_y, pred_val)
            kept = [s for s, flag in zip(val_smis, ok, strict=True) if bool(flag)]
            oof_scores.extend(sc.tolist())
            oof_smiles.extend(kept)

        base_stats = self.base_surrogate.fit(smiles, y)
        self._store_calibration(
            oof_smiles, np.asarray(oof_scores, dtype=np.float64), cal_mode="cv_plus"
        )
        return {
            **base_stats,
            "n_cal": float(self.cal_size),
            "q_hat": round(self.q_hat, 6),
            "target_coverage": round(1.0 - self.alpha, 4),
        }

    def _weighted_quantiles(
        self, query_smiles: Sequence[str], valid_mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute Tibshirani et al. (2019) Tanimoto-weighted quantiles & Kish N_eff."""
        n_q = len(query_smiles)
        qs = np.full(n_q, self.q_hat, dtype=np.float64)
        n_eff = np.full(n_q, float(self.cal_size), dtype=np.float64)
        if self.cal_size == 0 or not bool(valid_mask.any()) or self._cal_fps_sorted.shape[0] == 0:
            return qs, n_eff

        valid_idx = np.flatnonzero(valid_mask)
        valid_smis = [query_smiles[int(i)] for i in valid_idx]
        q_fps, _ = featurize_pool(
            valid_smis, kind="ecfp4_binary", n_bits=self._cal_fp_bits, radius=2
        )
        sims = tanimoto_kernel(q_fps, self._cal_fps_sorted)
        weights = np.exp(self.weight_gamma * sims)

        w_sum = np.sum(weights, axis=1)
        w_sq_sum = np.sum(weights * weights, axis=1)
        kish = np.divide(w_sum * w_sum, w_sq_sum, out=np.zeros_like(w_sum), where=w_sq_sum > 0.0)

        # Kish-consistent test mass w_test = sum(w^2) / sum(w) => p_inf = 1 / (N_eff + 1),
        # recovering exact unweighted Split-Conformal (1 / (n_cal + 1)) when weights are uniform.
        w_test = np.divide(w_sq_sum, w_sum, out=np.ones_like(w_sum), where=w_sum > 0.0)
        denom = w_sum + w_test
        cdf = np.cumsum(weights, axis=1) / denom[:, None]
        target = 1.0 - self.alpha - 1e-12

        for local_i, global_i in enumerate(valid_idx):
            n_eff[int(global_i)] = float(kish[local_i])
            pos = int(np.searchsorted(cdf[local_i], target, side="left"))
            if pos >= self.cal_size:
                qs[int(global_i)] = float(self._cal_scores_sorted[-1])
            else:
                qs[int(global_i)] = float(self._cal_scores_sorted[pos])
        return qs, n_eff

    def predict(self, smiles: Sequence[str]) -> SurrogatePrediction:
        """Predict point estimates, uncertainty, and finite-sample ``(1 - alpha)`` intervals."""
        base_pred = self.base_surrogate.predict(smiles)
        n = len(smiles)
        lo = np.full(n, float("nan"), dtype=np.float64)
        hi = np.full(n, float("nan"), dtype=np.float64)
        meta: dict[str, Any] = {
            **dict(base_pred.metadata),
            "conformal_mode": self.mode,
            "calibration_mode": self.calibration_mode,
            "q_hat": self.q_hat,
            "n_cal": self.cal_size,
            "alpha": self.alpha,
        }

        ok = base_pred.valid & np.isfinite(base_pred.mean)
        if bool(ok.any()):
            if self.mode == "absolute":
                half_w = np.full(n, self.q_hat, dtype=np.float64)
            elif self.mode == "normalized":
                scale = np.maximum(np.nan_to_num(base_pred.std, nan=self.min_sigma), self.min_sigma)
                half_w = self.q_hat * scale
            else:
                q_vec, kish_vec = self._weighted_quantiles(smiles, ok)
                scale = np.maximum(np.nan_to_num(base_pred.std, nan=self.min_sigma), self.min_sigma)
                half_w = q_vec * scale
                meta["mean_kish_n_eff"] = float(np.mean(kish_vec[ok]))
                meta["min_kish_n_eff"] = float(np.min(kish_vec[ok]))

            lo[ok] = base_pred.mean[ok] - half_w[ok]
            hi[ok] = base_pred.mean[ok] + half_w[ok]

        return SurrogatePrediction(
            mean=base_pred.mean,
            std=base_pred.std,
            epistemic_std=base_pred.epistemic_std,
            aleatoric_std=base_pred.aleatoric_std,
            valid=base_pred.valid,
            interval_lo=lo,
            interval_hi=hi,
            alpha=self.alpha,
            metadata=meta,
        )

    def predict_covariance(
        self, smiles: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Delegate exact covariance computation if ``base_surrogate`` supports it."""
        if isinstance(self.base_surrogate, TanimotoGP):
            return self.base_surrogate.predict_covariance(smiles)
        raise TypeError(
            f"Wrapped surrogate {type(self.base_surrogate).__name__} does not expose "
            "predict_covariance()."
        )


# ---------------------------------------------------------------------------
# Global Surrogate Plugin Registry
# ---------------------------------------------------------------------------

SurrogateFactory = Callable[..., SurrogateProtocol]
SURROGATE_REGISTRY: Registry[SurrogateFactory] = Registry("active_learning_surrogates")


def _build_tanimoto_gp(**kwargs: Any) -> TanimotoGP:
    return TanimotoGP(kernel="tanimoto", **kwargs)


def _build_minmax_gp(**kwargs: Any) -> TanimotoGP:
    return TanimotoGP(kernel="minmax", **kwargs)


def _build_ensemble(**kwargs: Any) -> EnsembleSurrogate:
    return EnsembleSurrogate(**kwargs)


def _build_conformal_tanimoto_gp(
    *,
    alpha: float = 0.10,
    mode: ConformalMode = "normalized",
    cal_fraction: float = 0.20,
    scaffold_split_cal: bool = True,
    weight_gamma: float = 5.0,
    min_split_n: int = 40,
    seed: int = 42,
    **gp_kwargs: Any,
) -> ConformalSurrogate:
    gp = TanimotoGP(kernel="tanimoto", seed=seed, **gp_kwargs)
    return ConformalSurrogate(
        gp,
        alpha=alpha,
        mode=mode,
        cal_fraction=cal_fraction,
        scaffold_split_cal=scaffold_split_cal,
        weight_gamma=weight_gamma,
        min_split_n=min_split_n,
        seed=seed,
    )


def _build_conformal_ensemble(
    *,
    alpha: float = 0.10,
    mode: ConformalMode = "normalized",
    cal_fraction: float = 0.20,
    scaffold_split_cal: bool = True,
    weight_gamma: float = 5.0,
    min_split_n: int = 40,
    seed: int = 42,
    **ens_kwargs: Any,
) -> ConformalSurrogate:
    ens = EnsembleSurrogate(seed=seed, **ens_kwargs)
    return ConformalSurrogate(
        ens,
        alpha=alpha,
        mode=mode,
        cal_fraction=cal_fraction,
        scaffold_split_cal=scaffold_split_cal,
        weight_gamma=weight_gamma,
        min_split_n=min_split_n,
        seed=seed,
    )


SURROGATE_REGISTRY.register(
    "tanimoto_gp",
    _build_tanimoto_gp,
    aliases=("gp", "tanimoto"),
    metadata={"family": "gaussian_process", "kernel": "tanimoto"},
)
SURROGATE_REGISTRY.register(
    "minmax_gp",
    _build_minmax_gp,
    aliases=("minmax", "ruzicka_gp"),
    metadata={"family": "gaussian_process", "kernel": "minmax"},
)
SURROGATE_REGISTRY.register(
    "ensemble",
    _build_ensemble,
    aliases=("hgb_rf", "rf_hgb"),
    metadata={"family": "tree_ensemble", "uq": "epistemic_aleatoric"},
)
SURROGATE_REGISTRY.register(
    "conformal_tanimoto_gp",
    _build_conformal_tanimoto_gp,
    aliases=("conformal_gp",),
    metadata={"family": "conformal", "base": "tanimoto_gp"},
)
SURROGATE_REGISTRY.register(
    "conformal_ensemble",
    _build_conformal_ensemble,
    metadata={"family": "conformal", "base": "ensemble"},
)


def register_surrogate(
    name: str,
    factory: SurrogateFactory,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
) -> SurrogateFactory:
    """Register a custom surrogate factory in :data:`SURROGATE_REGISTRY`."""
    return SURROGATE_REGISTRY.register(name, factory, aliases=aliases, override=override)


def list_surrogates() -> list[str]:
    """Return sorted canonical names of registered Active Learning surrogates."""
    return SURROGATE_REGISTRY.names()


def build_surrogate(
    surrogate: str | SurrogateProtocol,
    **kwargs: Any,
) -> SurrogateProtocol:
    """Resolve a surrogate name via :data:`SURROGATE_REGISTRY` or return an instance as-is."""
    if isinstance(surrogate, str):
        factory = SURROGATE_REGISTRY.get(surrogate)
        return factory(**kwargs)
    return surrogate


__all__ = [
    "SURROGATE_REGISTRY",
    "ConformalMode",
    "ConformalSurrogate",
    "EnsembleSurrogate",
    "KernelKind",
    "SurrogateFactory",
    "SurrogatePrediction",
    "SurrogateProtocol",
    "TanimotoGP",
    "build_surrogate",
    "list_surrogates",
    "register_surrogate",
]
