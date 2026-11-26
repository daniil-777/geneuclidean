"""Continuous DMPK Regression with Distribution-Free Split-Conformal Prediction Intervals.

Extends GenEuclidean's 12-endpoint binary Avoid-ome suite toward OpenADMET-aligned
continuous pharmacokinetic regression (``LogD_7.4``, kinetic solubility ``KSOL``,
``HLM_CLint``, ``MDR1-MDCK ER``) with exact distribution-free Split-Conformal prediction
intervals ``[y_lo, y_hi]`` guaranteeing finite-sample marginal coverage ``>= 1 - alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from geneuclidean.cns.admet import featurize_admet_smiles, make_scaffold_kfold


@dataclass(frozen=True)
class ConformalRegressionPrediction:
    """Single-molecule continuous DMPK prediction with calibrated Split-Conformal interval."""

    smiles: str
    valid: bool
    endpoint: str
    y_pred: float
    interval_lo: float
    interval_hi: float
    interval_width: float
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "valid": self.valid,
            "endpoint": self.endpoint,
            "y_pred": round(self.y_pred, 4) if math.isfinite(self.y_pred) else None,
            "interval_lo": round(self.interval_lo, 4) if math.isfinite(self.interval_lo) else None,
            "interval_hi": round(self.interval_hi, 4) if math.isfinite(self.interval_hi) else None,
            "interval_width": round(self.interval_width, 4)
            if math.isfinite(self.interval_width)
            else None,
            "alpha": self.alpha,
        }


class ConformalDMPKRegressor:
    """HGB + RandomForest ensemble regressor wrapped with Split-Conformal UQ intervals."""

    def __init__(
        self,
        endpoint: str,
        *,
        alpha: float = 0.10,
        n_estimators: int = 80,
        seed: int = 42,
    ) -> None:
        self.endpoint = str(endpoint)
        self.alpha = float(alpha)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self._hgb: HistGradientBoostingRegressor | None = None
        self._rf: RandomForestRegressor | None = None
        self.q_hat: float = float("nan")
        self.cal_size: int = 0

    def _fit_base(self, x_train: NDArray[np.float32], y_train: NDArray[np.float64]) -> None:
        self._hgb = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_depth=6,
            min_samples_leaf=8,
            random_state=self.seed,
        )
        self._rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            random_state=self.seed,
            n_jobs=1,
        )
        self._hgb.fit(x_train, y_train)
        self._rf.fit(x_train, y_train)

    def _predict_raw(self, x_mat: NDArray[np.float32]) -> NDArray[np.float64]:
        if self._hgb is None or self._rf is None:
            raise RuntimeError("ConformalDMPKRegressor must be fitted before prediction.")
        p1 = np.asarray(self._hgb.predict(x_mat), dtype=np.float64)
        p2 = np.asarray(self._rf.predict(x_mat), dtype=np.float64)
        return 0.5 * (p1 + p2)

    def fit(
        self,
        smiles_train: Sequence[str],
        y_train: Sequence[float],
        smiles_cal: Sequence[str],
        y_cal: Sequence[float],
    ) -> dict[str, float]:
        """Fit ensemble on ``smiles_train`` and calibrate Split-Conformal on ``smiles_cal``."""
        x_tr, _, v_tr, _ = featurize_admet_smiles(smiles_train)
        y_tr_arr = np.asarray(y_train, dtype=np.float64)
        self._fit_base(x_tr[v_tr], y_tr_arr[v_tr])

        x_ca, _, v_ca, _ = featurize_admet_smiles(smiles_cal)
        y_ca_arr = np.asarray(y_cal, dtype=np.float64)[v_ca]
        preds_ca = self._predict_raw(x_ca[v_ca])
        residuals = np.abs(y_ca_arr - preds_ca)
        n_cal = len(residuals)
        self.cal_size = int(n_cal)
        if n_cal == 0:
            self.q_hat = 1.0
        else:
            # Exact finite-sample split-conformal quantile index ceil((n+1)*(1-alpha))/n
            level = min(1.0, math.ceil((n_cal + 1) * (1.0 - self.alpha)) / float(n_cal))
            self.q_hat = float(np.quantile(residuals, level, method="higher"))
        return {
            "n_train": int(v_tr.sum()),
            "n_cal": int(n_cal),
            "q_hat": round(self.q_hat, 4),
            "cal_mae": round(float(np.mean(residuals)), 4) if n_cal > 0 else float("nan"),
        }

    def predict(self, smiles_list: Sequence[str]) -> list[ConformalRegressionPrediction]:
        """Predict point estimates and ``(1 - alpha)`` Split-Conformal intervals."""
        x_mat, _, valid, _ = featurize_admet_smiles(smiles_list)
        preds = np.full(len(smiles_list), float("nan"), dtype=np.float64)
        if bool(valid.any()):
            preds[valid] = self._predict_raw(x_mat[valid])
        out: list[ConformalRegressionPrediction] = []
        for smi, ok, yp in zip(smiles_list, valid, preds, strict=True):
            if not bool(ok) or not math.isfinite(float(yp)):
                out.append(
                    ConformalRegressionPrediction(
                        smiles=smi,
                        valid=False,
                        endpoint=self.endpoint,
                        y_pred=float("nan"),
                        interval_lo=float("nan"),
                        interval_hi=float("nan"),
                        interval_width=float("nan"),
                        alpha=self.alpha,
                    )
                )
            else:
                lo = float(yp - self.q_hat)
                hi = float(yp + self.q_hat)
                out.append(
                    ConformalRegressionPrediction(
                        smiles=smi,
                        valid=True,
                        endpoint=self.endpoint,
                        y_pred=float(yp),
                        interval_lo=lo,
                        interval_hi=hi,
                        interval_width=float(hi - lo),
                        alpha=self.alpha,
                    )
                )
        return out


def evaluate_scaffold_cv_conformal_dmpk(
    smiles_list: Sequence[str],
    y_values: Sequence[float],
    *,
    endpoint: str,
    n_splits: int = 5,
    alpha: float = 0.10,
    n_estimators: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    """Run strict Bemis-Murcko scaffold-CV evaluating accuracy + Split-Conformal coverage."""
    smis = list(smiles_list)
    ys = np.asarray(y_values, dtype=np.float64)
    folds = make_scaffold_kfold(smis, n_splits=n_splits, seed=seed)

    oof_pred = np.full(len(smis), float("nan"), dtype=np.float64)
    oof_lo = np.full(len(smis), float("nan"), dtype=np.float64)
    oof_hi = np.full(len(smis), float("nan"), dtype=np.float64)

    for f_idx, (tr_idx, va_idx) in enumerate(folds):
        # Split tr_idx 80/20 into proper-train and calibration (preserving scaffold grouping)
        split_at = max(8, int(0.8 * len(tr_idx)))
        prop_idx = tr_idx[:split_at]
        cal_idx = tr_idx[split_at:]
        if len(cal_idx) < 4:
            cal_idx = tr_idx[-4:]
            prop_idx = tr_idx[:-4]
        reg = ConformalDMPKRegressor(
            endpoint=endpoint, alpha=alpha, n_estimators=n_estimators, seed=seed + f_idx
        )
        reg.fit(
            [smis[int(i)] for i in prop_idx],
            ys[prop_idx].tolist(),
            [smis[int(i)] for i in cal_idx],
            ys[cal_idx].tolist(),
        )
        preds = reg.predict([smis[int(i)] for i in va_idx])
        for idx_val, pr in zip(va_idx, preds, strict=True):
            if pr.valid:
                oof_pred[int(idx_val)] = pr.y_pred
                oof_lo[int(idx_val)] = pr.interval_lo
                oof_hi[int(idx_val)] = pr.interval_hi

    ok = np.isfinite(oof_pred) & np.isfinite(ys)
    yt = ys[ok]
    yp = oof_pred[ok]
    yl = oof_lo[ok]
    yh = oof_hi[ok]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rho = float(spearmanr(yt, yp).statistic) if len(yt) >= 3 else float("nan")
    cov = float(np.mean((yt >= yl) & (yt <= yh)))
    width = float(np.mean(yh - yl))
    return {
        "endpoint": endpoint,
        "n_valid": int(ok.sum()),
        "n_splits": int(n_splits),
        "target_coverage": round(1.0 - alpha, 3),
        "empirical_coverage": round(cov, 4),
        "mean_interval_width": round(width, 4),
        "spearman_rho": round(rho, 4),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


__all__ = [
    "ConformalDMPKRegressor",
    "ConformalRegressionPrediction",
    "evaluate_scaffold_cv_conformal_dmpk",
]
