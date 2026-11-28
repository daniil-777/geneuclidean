"""Continuous DMPK Regression with Distribution-Free Split-Conformal Prediction Intervals.

Extends GenEuclidean's 12-endpoint binary Avoid-ome suite toward OpenADMET-aligned
continuous pharmacokinetic regression (``LogD_7.4``, kinetic solubility ``KSOL``,
``HLM_CLint``, ``MDR1-MDCK ER``) with exact distribution-free Split-Conformal prediction
intervals with marginal coverage under exchangeability. Scaffold-shift coverage is empirical.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from geneuclidean.cns.admet import featurize_admet_smiles, make_scaffold_kfold
from geneuclidean.cns.credibility import _finite_sample_conformal_quantile
from geneuclidean.data.splits import compute_murcko_scaffold

_OPENADMET = "openadmet_expansionrx_train.csv"
_BIOGEN = "ADME_public_set_3521.csv"
REFERENCE_ENDPOINTS = {
    "OpenADMET_LogD": (_OPENADMET, "LogD", False),
    "OpenADMET_log10_KSOL": (_OPENADMET, "KSOL", True),
    "OpenADMET_log10_HLM_CLint": (_OPENADMET, "HLM CLint", True),
    "Biogen_log_Human_Unbound_Percent": (
        _BIOGEN,
        "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
        False,
    ),
    "Biogen_log_HLM_CLint": (_BIOGEN, "LOG HLM_CLint (mL/min/kg)", False),
    "Biogen_log_Solubility_pH68": (_BIOGEN, "LOG SOLUBILITY PH 6.8 (ug/mL)", False),
}


def load_reference_endpoint(
    endpoint: str, data_dir: str | Path, *, max_rows: int | None = None
) -> tuple[list[str], list[float]]:
    """Read uncensored, quantified observations from the named reference assay CSV."""
    if endpoint not in REFERENCE_ENDPOINTS:
        raise ValueError(f"Unknown reference endpoint {endpoint!r}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    filename, column, use_log = REFERENCE_ENDPOINTS[endpoint]
    smiles, labels = [], []
    with (Path(data_dir) / filename).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"SMILES", column}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{filename} requires SMILES and {column!r} columns")
        for row in reader:
            smi = (row["SMILES"] or "").strip()
            try:
                value = float(row[column] or "")
            except ValueError:
                continue
            if not smi or not math.isfinite(value) or (use_log and value <= 0):
                continue
            smiles.append(smi)
            labels.append(math.log10(value) if use_log else value)
            if len(labels) == max_rows:
                break
    return smiles, labels


def _split_calibration(
    smiles: Sequence[str], seed: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    groups = [compute_murcko_scaffold(s) for s in smiles]
    if len(set(groups)) < 2:
        raise ValueError("Training and calibration need at least two distinct scaffolds")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, calibration = next(splitter.split(groups, groups=groups))
    return np.asarray(train, dtype=np.intp), np.asarray(calibration, dtype=np.intp)


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
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.endpoint = str(endpoint)
        self.alpha = float(alpha)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self._hgb: HistGradientBoostingRegressor | None = None
        self._rf: RandomForestRegressor | None = None
        self.q_hat: float = float("nan")
        self.cal_size: int = 0

    @classmethod
    def from_reference_data(
        cls,
        endpoint: str,
        *,
        data_dir: str | Path = "data/cns_neuro/v4_expansion/admet",
        max_rows: int | None = None,
        alpha: float = 0.10,
        n_estimators: int = 80,
        seed: int = 42,
    ) -> ConformalDMPKRegressor:
        """Fit from local endpoint-specific measurements using disjoint calibration scaffolds."""
        smiles, labels = load_reference_endpoint(endpoint, data_dir, max_rows=max_rows)
        tr, ca = _split_calibration(smiles, seed)
        model = cls(endpoint, alpha=alpha, n_estimators=n_estimators, seed=seed)
        model.fit(
            [smiles[i] for i in tr],
            [labels[i] for i in tr],
            [smiles[i] for i in ca],
            [labels[i] for i in ca],
        )
        return model

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
        y_tr_arr = np.asarray(y_train, dtype=np.float64)
        y_ca_arr = np.asarray(y_cal, dtype=np.float64)
        for smis, labels in ((smiles_train, y_tr_arr), (smiles_cal, y_ca_arr)):
            if labels.shape != (len(smis),) or not np.isfinite(labels).all():
                raise ValueError("Each SMILES requires one finite endpoint label")
        x_tr, _, v_tr, _ = featurize_admet_smiles(smiles_train)
        x_ca, _, v_ca, _ = featurize_admet_smiles(smiles_cal)
        if not v_tr.any():
            raise ValueError("Training requires at least one valid molecule")
        self._fit_base(x_tr[v_tr], y_tr_arr[v_tr])
        residuals = (
            np.abs(y_ca_arr[v_ca] - self._predict_raw(x_ca[v_ca]))
            if v_ca.any()
            else np.empty(0, dtype=np.float64)
        )
        n_cal = self.cal_size = len(residuals)
        self.q_hat = _finite_sample_conformal_quantile(residuals, self.alpha, upper_bound=math.inf)
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
    if ys.shape != (len(smis),) or not np.isfinite(ys).all():
        raise ValueError("Each SMILES requires one finite endpoint label")
    folds = make_scaffold_kfold(smis, n_splits=n_splits, seed=seed)

    oof_pred = np.full(len(smis), float("nan"), dtype=np.float64)
    oof_lo = np.full(len(smis), float("nan"), dtype=np.float64)
    oof_hi = np.full(len(smis), float("nan"), dtype=np.float64)

    for f_idx, (tr_idx, va_idx) in enumerate(folds):
        if not len(va_idx):
            raise ValueError("Each evaluation fold requires at least one scaffold")
        proper, calibration = _split_calibration([smis[i] for i in tr_idx], seed + f_idx)
        prop_idx, cal_idx = tr_idx[proper], tr_idx[calibration]
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
        "n_finite_intervals": int((np.isfinite(yl) & np.isfinite(yh)).sum()),
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
    "REFERENCE_ENDPOINTS",
    "evaluate_scaffold_cv_conformal_dmpk",
    "load_reference_endpoint",
]
