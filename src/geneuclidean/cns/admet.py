"""Calibrated Bemis-Murcko scaffold-CV CNS ADMET & cardiotoxicity oracles with AD.

Implements pure scikit-learn calibrated ensemble oracles
(``HistGradientBoostingClassifier`` + ``RandomForestClassifier`` + scaffold-disjoint
Platt/isotonic probability calibration) trained on 1024-bit ECFP4 fingerprints plus 12
RDKit 2D physicochemical descriptors across three critical CNS drug-design endpoints:

1. ``bbb``: Blood-Brain Barrier permeability ``P(BBB+)`` (primary: B3DB 7,805 compounds;
   cross-checked on non-overlapping TDC ``BBB_Martins``).
2. ``pgp``: P-glycoprotein (ABCB1/MDR1) **inhibition** ``P(P-gp inhibitor)`` (TDC
   ``Pgp_Broccatelli``, 1,212 unique compounds). Note: Broccatelli ``Y=1`` measures P-gp
   inhibition rather than pure MDCK-MDR1 efflux ratio; combine with Pfizer CNS MPO v2's
   2x HBD penalty for brain-efflux risk mitigation.
3. ``herg``: hERG K+ channel blockade cardiotoxicity ``P(hERG IC50 < 10 uM)`` (primary:
   TDC ``hERG_Karim``, 13,445 balanced compounds; cross-checked on TDC ``hERG``).

Every prediction carries an explicit ECFP4 nearest-neighbor **Applicability Domain (AD)**
Tanimoto similarity (``ad_tanimoto_max`` and ``in_domain = ad_tanimoto_max >= 0.30``).
"""

from __future__ import annotations

import importlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from geneuclidean.cns.mpo import CNSMPOResult, compute_cns_mpo, estimate_most_basic_pka
from geneuclidean.data.splits import compute_murcko_scaffold
from geneuclidean.evaluation.metrics import canonicalize_smiles

_DESCRIPTORS: Any = Descriptors
_CRIPPEN: Any = Crippen
_MORGAN_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

DEFAULT_ADMET_DATA_DIR = Path("data/cns_neuro/admet")
DEFAULT_ADMET_MODEL_PATH = Path("data/cns_neuro.nosync/admet_oracles.joblib")
DEFAULT_ADMET_CV_REPORT_PATH = Path("results/tables/cns_admet_cv_metrics.json")

ADMET_FP_BITS: int = 1024
ADMET_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MW",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "FractionCSP3",
    "RingCount",
    "AromaticRings",
    "HeavyAtomCount",
    "FormalCharge",
    "MostBasicPKa",
)

ORACLE_SEMANTICS: dict[str, str] = {
    "bbb": (
        "Blood-Brain Barrier permeability P(BBB+) trained on B3DB (Meng et al. 2021, "
        "7,805 valid compounds); higher is better for CNS penetration."
    ),
    "pgp": (
        "P-glycoprotein (ABCB1/MDR1) inhibition P(P-gp inhibitor) trained on TDC "
        "Pgp_Broccatelli (Broccatelli et al. 2011, 1,212 unique compounds). NOTE: Y=1 "
        "reflects P-gp inhibition rather than pure MDCK-MDR1 efflux substrate ratio; "
        "interpret alongside Pfizer CNS MPO v2's 2x HBD term for brain-efflux control."
    ),
    "herg": (
        "hERG K+ channel blockade cardiotoxicity P(hERG IC50 < 10 uM) trained on TDC "
        "hERG_Karim (Karim et al. 2021, 13,445 balanced compounds); lower is safer "
        "(critical QT-prolongation gate for psychiatric and pediatric Tourette drugs)."
    ),
}

DEFAULT_CNS_REWARD_WEIGHTS: dict[str, float] = {
    "cns_mpo_v2": 0.22,
    "bbb": 0.14,
    "herg_safe": 0.16,
    "pgp_safe": 0.08,
    "qed": 0.08,
    "sa": 0.08,
    "pains": 0.04,
    "pkd": 0.10,
    "clip": 0.05,
    "vina": 0.05,
}


def _get_joblib() -> Any:
    """Dynamically import joblib (bundled with scikit-learn) without static stub issues."""
    return importlib.import_module("joblib")


# ---------------------------------------------------------------------------
# Molecular Featurization (1024-bit ECFP4 + 12 RDKit 2D Descriptors)
# ---------------------------------------------------------------------------


def _mol_to_descriptors(mol: Chem.Mol) -> NDArray[np.float32]:
    """Compute the 12 standardized RDKit 2D physicochemical descriptors for one Mol."""
    vals = np.array(
        [
            float(_DESCRIPTORS.MolWt(mol)),
            float(_CRIPPEN.MolLogP(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(Chem.GetFormalCharge(mol)),
            float(estimate_most_basic_pka(mol)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def featurize_admet_smiles(
    smiles_list: Sequence[str],
) -> tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.bool_], list[str | None]]:
    """Convert SMILES into ``(X [N, 1036], fp_bits [N, 1024], valid_mask, canonical_smiles)``.

    Invalid SMILES produce zero rows with ``valid_mask[i] = False`` and
    ``canonical_smiles[i] = None``.
    """
    n = len(smiles_list)
    n_desc = len(ADMET_DESCRIPTOR_NAMES)
    x_mat = np.zeros((n, ADMET_FP_BITS + n_desc), dtype=np.float32)
    fp_bits = np.zeros((n, ADMET_FP_BITS), dtype=np.uint8)
    valid_mask = np.zeros(n, dtype=np.bool_)
    canonical: list[str | None] = [None] * n

    for idx, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            continue
        fp_arr = np.asarray(_MORGAN_1024.GetFingerprintAsNumPy(mol), dtype=np.uint8)
        desc = _mol_to_descriptors(mol)
        fp_bits[idx] = fp_arr
        x_mat[idx, :ADMET_FP_BITS] = fp_arr.astype(np.float32)
        x_mat[idx, ADMET_FP_BITS:] = desc
        valid_mask[idx] = True
        canonical[idx] = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

    return x_mat, fp_bits, valid_mask, canonical


# ---------------------------------------------------------------------------
# Strict Bemis-Murcko Scaffold K-Fold & Calibration Metrics
# ---------------------------------------------------------------------------


def make_scaffold_kfold(
    smiles_list: Sequence[str],
    n_splits: int = 5,
    *,
    seed: int = 42,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Build deterministic K-fold Bemis-Murcko scaffold splits with zero scaffold overlap.

    Assigns scaffold clusters greedily (largest first, deterministic tie-break) to the
    currently smallest fold, guaranteeing balanced fold sizes and zero shared Murcko
    scaffolds between train and validation in every fold.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    n = len(smiles_list)
    if n < n_splits:
        raise ValueError(f"Cannot build {n_splits}-fold scaffold CV on {n} samples")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = compute_murcko_scaffold(smi)
        groups[scaf].append(idx)

    rng = np.random.default_rng(seed)
    sorted_scafs = sorted(groups.keys())
    tie_breakers = {scaf: float(rng.random()) for scaf in sorted_scafs}
    ordered_clusters = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), tie_breakers[item[0]]),
    )

    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for _scaf, members in ordered_clusters:
        target_fold = int(np.argmin(fold_sizes))
        fold_indices[target_fold].extend(members)
        fold_sizes[target_fold] += len(members)

    all_idx = np.arange(n, dtype=np.intp)
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    for k in range(n_splits):
        val_idx = np.asarray(sorted(fold_indices[k]), dtype=np.intp)
        mask = np.ones(n, dtype=np.bool_)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        folds.append((train_idx, val_idx))
    return folds


def compute_ece(
    y_true: NDArray[np.int64] | Sequence[int],
    y_prob: NDArray[np.float64] | Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over ``n_bins`` equal-width probability bins."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = len(yt)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (yp >= lo) & (yp <= hi) if b == n_bins - 1 else (yp >= lo) & (yp < hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        acc = float(yt[in_bin].mean())
        conf = float(yp[in_bin].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def _classification_metrics(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC, Brier Score, and 10-bin ECE."""
    yp = np.clip(y_prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    has_both = len(np.unique(y_true)) >= 2
    return {
        "roc_auc": float(roc_auc_score(y_true, yp)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, yp)) if has_both else float("nan"),
        "brier_score": float(brier_score_loss(y_true, yp)),
        "ece": compute_ece(y_true, yp, n_bins=10),
    }


# ---------------------------------------------------------------------------
# Calibrated Single-Endpoint Oracle with Bit-Packed Applicability Domain
# ---------------------------------------------------------------------------


class _ProbabilityCalibrator:
    """Compact Platt-sigmoid or isotonic post-hoc calibrator for ensemble probabilities."""

    def __init__(self, method: Literal["sigmoid", "isotonic"] = "sigmoid") -> None:
        self.method = method
        self._platt: Any | None = None
        self._iso: Any | None = None

    def fit(self, raw_probs: NDArray[np.float64], y: NDArray[np.int64]) -> _ProbabilityCalibrator:
        clipped = np.clip(raw_probs.astype(np.float64), 1e-5, 1.0 - 1e-5)
        if len(np.unique(y)) < 2 or len(y) < 10:
            self._platt = None
            self._iso = None
            return self
        if self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            iso.fit(clipped, y.astype(np.float64))
            self._iso = iso
        else:
            logits = logit(clipped).reshape(-1, 1)
            lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=200)
            lr.fit(logits, y)
            self._platt = lr
        return self

    def transform(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(raw_probs.astype(np.float64), 1e-5, 1.0 - 1e-5)
        if self._iso is not None:
            return np.asarray(self._iso.predict(clipped), dtype=np.float64)
        if self._platt is not None:
            logits = logit(clipped).reshape(-1, 1)
            return np.asarray(self._platt.predict_proba(logits)[:, 1], dtype=np.float64)
        return clipped


class CalibratedADMETOracle:
    """Single calibrated HGB + RF CNS ADMET classifier with ECFP4 applicability domain."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        hgb_max_iter: int = 160,
        rf_n_estimators: int = 160,
        ad_threshold: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.description = description or ORACLE_SEMANTICS.get(name, "")
        self.calibration_method: Literal["sigmoid", "isotonic"] = calibration_method
        self.hgb_max_iter = int(hgb_max_iter)
        self.rf_n_estimators = int(rf_n_estimators)
        self.ad_threshold = float(ad_threshold)
        self.seed = int(seed)

        self._hgb: Any | None = None
        self._rf: Any | None = None
        self._calibrator: _ProbabilityCalibrator = _ProbabilityCalibrator(calibration_method)
        self._ref_fp_packed: NDArray[np.uint8] = np.zeros((0, ADMET_FP_BITS // 8), dtype=np.uint8)
        self._ref_fp_float_cache: NDArray[np.float32] | None = None
        self._ref_fp_norms_cache: NDArray[np.float32] | None = None
        self.cv_metrics: dict[str, Any] = {}
        self.n_train: int = 0

    def _build_base_models(self, seed_offset: int = 0) -> tuple[Any, Any]:
        s = self.seed + seed_offset
        hgb = HistGradientBoostingClassifier(
            max_iter=self.hgb_max_iter,
            max_depth=6,
            learning_rate=0.06,
            min_samples_leaf=15,
            l2_regularization=1.0,
            max_bins=128,
            random_state=s,
        )
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=s,
        )
        return hgb, rf

    @staticmethod
    def _predict_ensemble_raw(hgb: Any, rf: Any, x_mat: NDArray[np.float32]) -> NDArray[np.float64]:
        p_hgb = np.asarray(hgb.predict_proba(x_mat)[:, 1], dtype=np.float64)
        p_rf = np.asarray(rf.predict_proba(x_mat)[:, 1], dtype=np.float64)
        return 0.5 * (p_hgb + p_rf)

    def fit_and_evaluate(
        self,
        smiles_list: Sequence[str],
        labels: Sequence[int],
        *,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Run strict K-fold Bemis-Murcko scaffold CV, then fit final calibrated oracle."""
        x_all, fp_bits_all, valid_mask, canon_list = featurize_admet_smiles(smiles_list)
        y_raw = np.asarray(labels, dtype=np.int64)
        keep = valid_mask & np.isin(y_raw, [0, 1])

        x_mat = x_all[keep]
        fp_bits = fp_bits_all[keep]
        y_arr = y_raw[keep]
        valid_smiles: list[str] = [
            s for i in np.flatnonzero(keep).tolist() if (s := canon_list[i]) is not None
        ]
        n_samples = len(valid_smiles)
        if n_samples < max(n_splits * 2, 10):
            raise ValueError(
                f"Oracle {self.name!r} requires at least {max(n_splits * 2, 10)} valid molecules, "
                f"got {n_samples}"
            )

        # Verify scaffold-CV partitioning and zero leakage
        folds = make_scaffold_kfold(valid_smiles, n_splits=n_splits, seed=self.seed)
        scaffolds = [compute_murcko_scaffold(s) for s in valid_smiles]
        unique_scaffolds = len(set(scaffolds))

        oof_raw = np.zeros(n_samples, dtype=np.float64)
        oof_cal = np.zeros(n_samples, dtype=np.float64)
        fold_reports: list[dict[str, float]] = []
        leakage_count = 0

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            tr_scafs = {scaffolds[i] for i in tr_idx.tolist()}
            va_scafs = {scaffolds[i] for i in va_idx.tolist()}
            leakage_count += len(tr_scafs & va_scafs)

            # Inner scaffold-disjoint calibration split (80% fit / 20% calibrate)
            tr_smiles = [valid_smiles[i] for i in tr_idx.tolist()]
            inner_folds = make_scaffold_kfold(
                tr_smiles,
                n_splits=5,
                seed=self.seed + 100 + fold_idx,
            )
            in_fit_rel, in_cal_rel = inner_folds[0]
            fit_idx = tr_idx[in_fit_rel]
            cal_idx = tr_idx[in_cal_rel]

            # Guard against single-class inner split on very small synthetic sets
            if len(np.unique(y_arr[fit_idx])) < 2 or len(np.unique(y_arr[cal_idx])) < 2:
                fit_idx = tr_idx
                cal_idx = tr_idx

            hgb_f, rf_f = self._build_base_models(seed_offset=fold_idx + 1)
            hgb_f.fit(x_mat[fit_idx], y_arr[fit_idx])
            rf_f.fit(x_mat[fit_idx], y_arr[fit_idx])

            cal_raw = self._predict_ensemble_raw(hgb_f, rf_f, x_mat[cal_idx])
            cal_model = _ProbabilityCalibrator(self.calibration_method).fit(cal_raw, y_arr[cal_idx])

            va_raw = self._predict_ensemble_raw(hgb_f, rf_f, x_mat[va_idx])
            va_cal = cal_model.transform(va_raw)

            oof_raw[va_idx] = va_raw
            oof_cal[va_idx] = va_cal
            fold_reports.append(_classification_metrics(y_arr[va_idx], va_cal))

        if leakage_count != 0:
            raise AssertionError(
                f"Scaffold leakage detected in {self.name} CV: {leakage_count} overlapping groups"
            )

        # Final deployed model trained on 100% of data, calibrated on scaffold-OOF probabilities
        hgb_final, rf_final = self._build_base_models(seed_offset=0)
        hgb_final.fit(x_mat, y_arr)
        rf_final.fit(x_mat, y_arr)
        self._hgb = hgb_final
        self._rf = rf_final
        self._calibrator = _ProbabilityCalibrator(self.calibration_method).fit(oof_raw, y_arr)

        # Store bit-packed training fingerprints for fast Applicability Domain queries
        self._ref_fp_packed = np.packbits(fp_bits, axis=1)
        self._ref_fp_float_cache = None
        self._ref_fp_norms_cache = None
        self.n_train = n_samples

        overall_cal = _classification_metrics(y_arr, oof_cal)
        overall_raw = _classification_metrics(y_arr, oof_raw)

        def _fold_stat(key: str) -> tuple[float, float]:
            vals = [f[key] for f in fold_reports if math.isfinite(f[key])]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        roc_m, roc_s = _fold_stat("roc_auc")
        pr_m, pr_s = _fold_stat("pr_auc")
        brier_m, brier_s = _fold_stat("brier_score")
        ece_m, ece_s = _fold_stat("ece")

        self.cv_metrics = {
            "endpoint": self.name,
            "description": self.description,
            "split_protocol": f"{n_splits}-fold Bemis-Murcko scaffold CV (zero scaffold overlap)",
            "calibration_method": self.calibration_method,
            "n_samples": int(n_samples),
            "n_positive": int(y_arr.sum()),
            "n_negative": int((y_arr == 0).sum()),
            "positive_rate": float(y_arr.mean()),
            "n_unique_scaffolds": int(unique_scaffolds),
            "scaffold_leakage_groups": int(leakage_count),
            "oof_roc_auc": overall_cal["roc_auc"],
            "oof_pr_auc": overall_cal["pr_auc"],
            "oof_brier_score": overall_cal["brier_score"],
            "oof_ece": overall_cal["ece"],
            "oof_uncalibrated_brier_score": overall_raw["brier_score"],
            "oof_uncalibrated_ece": overall_raw["ece"],
            "fold_roc_auc_mean": roc_m,
            "fold_roc_auc_std": roc_s,
            "fold_pr_auc_mean": pr_m,
            "fold_pr_auc_std": pr_s,
            "fold_brier_mean": brier_m,
            "fold_brier_std": brier_s,
            "fold_ece_mean": ece_m,
            "fold_ece_std": ece_s,
            "ad_threshold": self.ad_threshold,
        }
        return dict(self.cv_metrics)

    def _get_ref_fp_matrix(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Lazily unpack bit-packed training ECFP4 fingerprints into float32 for BLAS Tanimoto."""
        if self._ref_fp_float_cache is None or self._ref_fp_norms_cache is None:
            if len(self._ref_fp_packed) == 0:
                self._ref_fp_float_cache = np.zeros((0, ADMET_FP_BITS), dtype=np.float32)
                self._ref_fp_norms_cache = np.zeros((0,), dtype=np.float32)
            else:
                unpacked = np.unpackbits(self._ref_fp_packed, axis=1)[:, :ADMET_FP_BITS]
                mat = unpacked.astype(np.float32)
                self._ref_fp_float_cache = mat
                self._ref_fp_norms_cache = mat.sum(axis=1).astype(np.float32)
        return self._ref_fp_float_cache, self._ref_fp_norms_cache

    def compute_ad_tanimoto_from_bits(
        self,
        fp_bits: NDArray[np.uint8],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Compute nearest-neighbor ECFP4 Tanimoto similarity to the training set."""
        n = len(fp_bits)
        max_sim = np.zeros(n, dtype=np.float64)
        ref_mat, ref_norms = self._get_ref_fp_matrix()
        if n == 0 or len(ref_mat) == 0 or not bool(valid_mask.any()):
            return max_sim

        valid_idx = np.flatnonzero(valid_mask)
        q_mat = fp_bits[valid_idx].astype(np.float32)
        q_norms = q_mat.sum(axis=1).astype(np.float32)

        chunk = 256
        out_valid = np.zeros(len(valid_idx), dtype=np.float64)
        for start in range(0, len(valid_idx), chunk):
            end = min(start + chunk, len(valid_idx))
            inter = q_mat[start:end] @ ref_mat.T
            union = q_norms[start:end, None] + ref_norms[None, :] - inter
            tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
            out_valid[start:end] = np.max(tan, axis=1).astype(np.float64)

        max_sim[valid_idx] = np.clip(out_valid, 0.0, 1.0)
        return max_sim

    def predict_from_features(
        self,
        x_mat: NDArray[np.float32],
        fp_bits: NDArray[np.uint8],
        valid_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Predict calibrated probabilities and Applicability Domain from precomputed features."""
        if self._hgb is None or self._rf is None:
            raise RuntimeError(f"Oracle {self.name!r} has not been fitted.")
        n = len(x_mat)
        probs = np.full(n, float("nan"), dtype=np.float64)
        ad_tan = np.zeros(n, dtype=np.float64)
        in_dom = np.zeros(n, dtype=np.bool_)

        if n == 0 or not bool(valid_mask.any()):
            return probs, ad_tan, in_dom

        valid_idx = np.flatnonzero(valid_mask)
        raw = self._predict_ensemble_raw(self._hgb, self._rf, x_mat[valid_idx])
        cal = np.clip(self._calibrator.transform(raw), 0.0, 1.0)
        probs[valid_idx] = cal

        ad_tan = self.compute_ad_tanimoto_from_bits(fp_bits, valid_mask)
        in_dom = valid_mask & (ad_tan >= self.ad_threshold)
        return probs, ad_tan, in_dom

    def predict(
        self,
        smiles_list: Sequence[str],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Predict calibrated probabilities, AD Tanimoto max, and in-domain flags for SMILES."""
        x_mat, fp_bits, valid_mask, _ = featurize_admet_smiles(smiles_list)
        return self.predict_from_features(x_mat, fp_bits, valid_mask)

    def evaluate_external(
        self,
        smiles_list: Sequence[str],
        labels: Sequence[int],
        *,
        training_canonical_set: set[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the fitted oracle on an external benchmark (full & non-overlapping)."""
        x_all, fp_bits_all, valid_mask, canon_list = featurize_admet_smiles(smiles_list)
        y_raw = np.asarray(labels, dtype=np.int64)
        keep = valid_mask & np.isin(y_raw, [0, 1])

        probs, ad_tan, in_dom = self.predict_from_features(
            x_all[keep], fp_bits_all[keep], np.ones(int(keep.sum()), dtype=np.bool_)
        )
        y_keep = y_raw[keep]
        can_keep: list[str] = [
            s for i in np.flatnonzero(keep).tolist() if (s := canon_list[i]) is not None
        ]

        report: dict[str, Any] = {
            "n_total": int(len(y_keep)),
            "in_domain_rate": float(in_dom.mean()) if len(in_dom) > 0 else 0.0,
            "mean_ad_tanimoto": float(ad_tan.mean()) if len(ad_tan) > 0 else 0.0,
            "full": _classification_metrics(y_keep, probs),
        }
        if training_canonical_set is not None:
            novel_mask = np.asarray(
                [smi not in training_canonical_set for smi in can_keep], dtype=np.bool_
            )
            n_novel = int(novel_mask.sum())
            report["n_non_overlapping"] = n_novel
            report["n_overlapping"] = int(len(y_keep) - n_novel)
            if n_novel >= 10:
                report["non_overlapping"] = _classification_metrics(
                    y_keep[novel_mask], probs[novel_mask]
                )
        return report

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Never serialize unpacked float32 caches — keep joblib artifact ultra-compact!
        state["_ref_fp_float_cache"] = None
        state["_ref_fp_norms_cache"] = None
        return state


# ---------------------------------------------------------------------------
# 3-Oracle CNS ADMET Suite & Dataset Loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CNSAdmetPrediction:
    """Calibrated BBB / P-gp / hERG probabilities and Applicability Domain for one ligand."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    bbb_prob: float
    bbb_ad_tanimoto: float
    bbb_in_domain: bool
    pgp_prob: float
    pgp_ad_tanimoto: float
    pgp_in_domain: bool
    herg_prob: float
    herg_ad_tanimoto: float
    herg_in_domain: bool
    in_domain_all: bool
    admet_pass: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize prediction to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "bbb_prob": self.bbb_prob,
            "bbb_ad_tanimoto": self.bbb_ad_tanimoto,
            "bbb_in_domain": self.bbb_in_domain,
            "pgp_prob": self.pgp_prob,
            "pgp_ad_tanimoto": self.pgp_ad_tanimoto,
            "pgp_in_domain": self.pgp_in_domain,
            "herg_prob": self.herg_prob,
            "herg_ad_tanimoto": self.herg_ad_tanimoto,
            "herg_in_domain": self.herg_in_domain,
            "in_domain_all": self.in_domain_all,
            "admet_pass": self.admet_pass,
        }


def _deduplicate_binary_dataset(
    smiles_list: Sequence[str],
    labels: Sequence[int],
) -> tuple[list[str], list[int], dict[str, int]]:
    """Canonicalize SMILES (largest organic fragment), deduplicate, and drop label conflicts."""
    by_canon: dict[str, set[int]] = defaultdict(set)
    n_invalid = 0
    for smi, lbl in zip(smiles_list, labels, strict=True):
        can = canonicalize_smiles(str(smi), isomeric=True, keep_largest_fragment=True)
        if can is None:
            n_invalid += 1
            continue
        by_canon[can].add(int(lbl))

    clean_smiles: list[str] = []
    clean_labels: list[int] = []
    n_conflicts = 0
    for can, lbl_set in by_canon.items():
        if len(lbl_set) > 1:
            n_conflicts += 1
            continue
        clean_smiles.append(can)
        clean_labels.append(next(iter(lbl_set)))

    stats_dict = {
        "n_raw": len(smiles_list),
        "n_invalid": n_invalid,
        "n_label_conflicts_dropped": n_conflicts,
        "n_clean_unique": len(clean_smiles),
    }
    return clean_smiles, clean_labels, stats_dict


def load_b3db_classification(
    path: Path,
) -> tuple[list[str], list[int], dict[str, int]]:
    """Load and clean ``B3DB_classification.tsv`` (``BBB+ -> 1``, ``BBB- -> 0``)."""
    df = pd.read_csv(path, sep="\t")
    smiles = df["SMILES"].astype(str).tolist()
    labels = [1 if str(v).strip() == "BBB+" else 0 for v in df["BBB+/BBB-"].tolist()]
    return _deduplicate_binary_dataset(smiles, labels)


def load_tdc_binary_csv(
    path: Path,
    *,
    smiles_col: str = "Drug",
    label_col: str = "Y",
) -> tuple[list[str], list[int], dict[str, int]]:
    """Load and deduplicate a TDC binary ADMET CSV/TSV file."""
    df = pd.read_csv(path)
    smiles = df[smiles_col].astype(str).tolist()
    labels = [int(round(float(v))) for v in df[label_col].tolist()]
    return _deduplicate_binary_dataset(smiles, labels)


class CNSAdmetSuite:
    """Unified 3-oracle CNS ADMET & Safety Suite (BBB+, P-gp inhibition, hERG blockade)."""

    def __init__(
        self,
        *,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        hgb_max_iter: int = 160,
        rf_n_estimators: int = 160,
        ad_threshold: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.ad_threshold = float(ad_threshold)
        self.oracles: dict[str, CalibratedADMETOracle] = {
            name: CalibratedADMETOracle(
                name=name,
                description=ORACLE_SEMANTICS[name],
                calibration_method=calibration_method,
                hgb_max_iter=hgb_max_iter,
                rf_n_estimators=rf_n_estimators,
                ad_threshold=ad_threshold,
                seed=seed + offset,
            )
            for offset, name in enumerate(("bbb", "pgp", "herg"))
        }
        self.cv_report: dict[str, Any] = {}

    def fit_endpoint(
        self,
        endpoint: Literal["bbb", "pgp", "herg"],
        smiles_list: Sequence[str],
        labels: Sequence[int],
        *,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Train and 5-fold scaffold-CV evaluate one ADMET oracle."""
        if endpoint not in self.oracles:
            raise KeyError(f"Unknown ADMET endpoint {endpoint!r}; expected 'bbb', 'pgp', or 'herg'")
        metrics = self.oracles[endpoint].fit_and_evaluate(smiles_list, labels, n_splits=n_splits)
        self.cv_report[endpoint] = metrics
        return metrics

    def fit_from_directory(
        self,
        admet_dir: str | Path = DEFAULT_ADMET_DATA_DIR,
        *,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Train all 3 CNS oracles from ``data/cns_neuro/admet/`` with external cross-checks."""
        root = Path(admet_dir)
        b3db_path = root / "B3DB_classification.tsv"
        martins_path = root / "tdc_BBB_Martins.csv"
        pgp_path = root / "tdc_Pgp_Broccatelli.csv"
        herg_karim_path = root / "tdc_hERG_Karim.csv"
        herg_wang_path = root / "tdc_hERG.csv"

        for required in (b3db_path, pgp_path, herg_karim_path):
            if not required.exists():
                raise FileNotFoundError(f"Required CNS ADMET dataset not found: {required}")

        # 1. BBB+ (B3DB primary + TDC BBB_Martins cross-check)
        bbb_smi, bbb_y, bbb_clean_stats = load_b3db_classification(b3db_path)
        bbb_cv = self.fit_endpoint("bbb", bbb_smi, bbb_y, n_splits=n_splits)
        bbb_cv["data_hygiene"] = bbb_clean_stats
        if martins_path.exists():
            m_smi, m_y, m_stats = load_tdc_binary_csv(martins_path)
            bbb_cv["external_tdc_bbb_martins"] = {
                "data_hygiene": m_stats,
                **self.oracles["bbb"].evaluate_external(
                    m_smi, m_y, training_canonical_set=set(bbb_smi)
                ),
            }

        # 2. P-gp inhibition (TDC Pgp_Broccatelli)
        pgp_smi, pgp_y, pgp_clean_stats = load_tdc_binary_csv(pgp_path)
        pgp_cv = self.fit_endpoint("pgp", pgp_smi, pgp_y, n_splits=n_splits)
        pgp_cv["data_hygiene"] = pgp_clean_stats

        # 3. hERG cardiotoxicity (TDC hERG_Karim primary + TDC hERG cross-check)
        herg_smi, herg_y, herg_clean_stats = load_tdc_binary_csv(herg_karim_path)
        herg_cv = self.fit_endpoint("herg", herg_smi, herg_y, n_splits=n_splits)
        herg_cv["data_hygiene"] = herg_clean_stats
        if herg_wang_path.exists():
            hw_smi, hw_y, hw_stats = load_tdc_binary_csv(herg_wang_path)
            herg_cv["external_tdc_herg_wang"] = {
                "data_hygiene": hw_stats,
                **self.oracles["herg"].evaluate_external(
                    hw_smi, hw_y, training_canonical_set=set(herg_smi)
                ),
            }

        self.cv_report = {
            "schema_version": "geneuclidean-v3.1-cns-admet-cv",
            "features": {
                "ecfp4_bits": ADMET_FP_BITS,
                "descriptors": list(ADMET_DESCRIPTOR_NAMES),
                "total_dim": ADMET_FP_BITS + len(ADMET_DESCRIPTOR_NAMES),
            },
            "model_family": (
                "Calibrated 50/50 HistGradientBoostingClassifier + RandomForestClassifier "
                "ensemble with scaffold-disjoint probability calibration and ECFP4 AD"
            ),
            "endpoints": {
                "bbb": bbb_cv,
                "pgp": pgp_cv,
                "herg": herg_cv,
            },
        }
        return dict(self.cv_report)

    def predict(
        self,
        smiles_list: Sequence[str],
        *,
        bbb_min: float = 0.70,
        herg_max: float = 0.50,
        pgp_max: float = 0.75,
    ) -> list[CNSAdmetPrediction]:
        """Predict calibrated BBB+, P-gp, hERG probabilities and AD for candidate SMILES."""
        x_mat, fp_bits, valid_mask, canonical = featurize_admet_smiles(smiles_list)
        bbb_p, bbb_ad, bbb_in = self.oracles["bbb"].predict_from_features(
            x_mat, fp_bits, valid_mask
        )
        pgp_p, pgp_ad, pgp_in = self.oracles["pgp"].predict_from_features(
            x_mat, fp_bits, valid_mask
        )
        herg_p, herg_ad, herg_in = self.oracles["herg"].predict_from_features(
            x_mat, fp_bits, valid_mask
        )

        results: list[CNSAdmetPrediction] = []
        for idx, raw_smi in enumerate(smiles_list):
            is_valid = bool(valid_mask[idx])
            if not is_valid:
                results.append(
                    CNSAdmetPrediction(
                        smiles=raw_smi,
                        canonical_smiles=None,
                        valid=False,
                        bbb_prob=0.0,
                        bbb_ad_tanimoto=0.0,
                        bbb_in_domain=False,
                        pgp_prob=1.0,
                        pgp_ad_tanimoto=0.0,
                        pgp_in_domain=False,
                        herg_prob=1.0,
                        herg_ad_tanimoto=0.0,
                        herg_in_domain=False,
                        in_domain_all=False,
                        admet_pass=False,
                    )
                )
                continue

            b_prob = float(bbb_p[idx])
            p_prob = float(pgp_p[idx])
            h_prob = float(herg_p[idx])
            b_in = bool(bbb_in[idx])
            p_in = bool(pgp_in[idx])
            h_in = bool(herg_in[idx])
            admet_ok = bool(b_prob >= bbb_min and h_prob < herg_max and p_prob < pgp_max)

            results.append(
                CNSAdmetPrediction(
                    smiles=raw_smi,
                    canonical_smiles=canonical[idx],
                    valid=True,
                    bbb_prob=b_prob,
                    bbb_ad_tanimoto=float(bbb_ad[idx]),
                    bbb_in_domain=b_in,
                    pgp_prob=p_prob,
                    pgp_ad_tanimoto=float(pgp_ad[idx]),
                    pgp_in_domain=p_in,
                    herg_prob=h_prob,
                    herg_ad_tanimoto=float(herg_ad[idx]),
                    herg_in_domain=h_in,
                    in_domain_all=bool(b_in and p_in and h_in),
                    admet_pass=admet_ok,
                )
            )
        return results

    def save(self, path: str | Path = DEFAULT_ADMET_MODEL_PATH, *, compress: int = 3) -> Path:
        """Serialize the fitted suite cleanly to a compact compressed joblib artifact."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib = _get_joblib()
        joblib.dump(self, out_path, compress=compress)
        return out_path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ADMET_MODEL_PATH) -> CNSAdmetSuite:
        """Load a previously serialized ``CNSAdmetSuite`` from disk."""
        src_path = Path(path)
        if not src_path.exists():
            raise FileNotFoundError(
                f"CNS ADMET oracle bundle not found at {src_path}. "
                "Train it via train_default_cns_admet_suite() when data/cns_neuro/admet is present."
            )
        joblib = _get_joblib()
        loaded = joblib.load(src_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected CNSAdmetSuite in {src_path}, got {type(loaded).__name__}")
        return loaded

    @classmethod
    def try_load(cls, path: str | Path = DEFAULT_ADMET_MODEL_PATH) -> CNSAdmetSuite | None:
        """Load the oracle suite if present on disk, otherwise return ``None`` (CI-safe)."""
        src_path = Path(path)
        if not src_path.exists():
            return None
        return cls.load(src_path)


def train_default_cns_admet_suite(
    admet_dir: str | Path = DEFAULT_ADMET_DATA_DIR,
    *,
    model_out: str | Path = DEFAULT_ADMET_MODEL_PATH,
    report_out: str | Path = DEFAULT_ADMET_CV_REPORT_PATH,
    n_splits: int = 5,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    seed: int = 42,
) -> tuple[CNSAdmetSuite, dict[str, Any]]:
    """Train, scaffold-CV evaluate, and persist the default 3-oracle CNS ADMET suite."""
    suite = CNSAdmetSuite(calibration_method=calibration_method, seed=seed)
    report = suite.fit_from_directory(admet_dir, n_splits=n_splits)
    saved_model = suite.save(model_out)
    report["model_artifact"] = str(saved_model)
    report["model_size_mb"] = round(saved_model.stat().st_size / (1024.0 * 1024.0), 3)

    report_path = Path(report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return suite, report


# ---------------------------------------------------------------------------
# Composite CNS Multi-Objective Reward (MPO v2 + ADMET + MedChem + 3D Binding)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CNSCompositeScore:
    """Unified CNS drug-design desirability combining Pfizer MPO v2, ADMET, and 3D binding."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    mpo: CNSMPOResult
    admet: CNSAdmetPrediction | None
    pred_pkd: float | None
    clip_sim: float | None
    vina_score: float | None
    composite_reward: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize composite score to a flat JSON-friendly dict."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "cns_mpo_v1": self.mpo.cns_mpo_v1,
            "cns_mpo_v2": self.mpo.cns_mpo_v2,
            "qed": self.mpo.qed,
            "sa_score": self.mpo.sa_score,
            "sa_norm": self.mpo.sa_norm,
            "pains_free": self.mpo.pains_free,
            "bbb_prob": self.admet.bbb_prob if self.admet is not None else None,
            "pgp_prob": self.admet.pgp_prob if self.admet is not None else None,
            "herg_prob": self.admet.herg_prob if self.admet is not None else None,
            "in_domain_all": self.admet.in_domain_all if self.admet is not None else None,
            "pred_pkd": self.pred_pkd,
            "clip_sim": self.clip_sim,
            "vina_score": self.vina_score,
            "composite_reward": self.composite_reward,
        }


def compute_cns_composite_reward(
    smiles_list: Sequence[str],
    pred_pkd: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    clip_sim: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    vina_scores: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    *,
    admet_suite: CNSAdmetSuite | None = None,
    weights: Mapping[str, float] | None = None,
    ad_damp: bool = False,
) -> list[CNSCompositeScore]:
    """Compute CNS multi-objective composite reward in ``[0.0, 1.0]`` for candidate SMILES.

    Combines Pfizer CNS MPO v2 (``cns_mpo_v2 / 6.0``), calibrated ``P(BBB+)``,
    ``1 - P(hERG)``, ``1 - P(P-gp)``, ``QED``, ``SA_norm``, ``PAINS-free``, and optional
    3D pocket binding terms (predicted ``pKd``, DrugCLIP similarity, AutoDock Vina score).
    When ``admet_suite`` or 3D scores are omitted, weights renormalize cleanly across the
    available terms; PAINS-flagged molecules receive a strict ``0.5x`` penalty. Setting
    ``ad_damp=True`` linearly shrinks out-of-domain ADMET oracle predictions toward a
    neutral ``0.5`` prior proportional to ``min(1.0, ad_tanimoto / ad_threshold)`` to
    prevent reward hacking outside each oracle's Applicability Domain.
    """
    n = len(smiles_list)
    if pred_pkd is not None and len(pred_pkd) != n:
        raise ValueError(f"pred_pkd length ({len(pred_pkd)}) != smiles_list length ({n})")
    if clip_sim is not None and len(clip_sim) != n:
        raise ValueError(f"clip_sim length ({len(clip_sim)}) != smiles_list length ({n})")
    if vina_scores is not None and len(vina_scores) != n:
        raise ValueError(f"vina_scores length ({len(vina_scores)}) != smiles_list length ({n})")

    eff_weights = dict(DEFAULT_CNS_REWARD_WEIGHTS)
    if weights is not None:
        eff_weights.update({k: float(v) for k, v in weights.items()})

    mpo_results = [compute_cns_mpo(s) for s in smiles_list]
    admet_results = admet_suite.predict(smiles_list) if admet_suite is not None else None

    scores_out: list[CNSCompositeScore] = []
    for idx, smi in enumerate(smiles_list):
        mpo = mpo_results[idx]
        admet = admet_results[idx] if admet_results is not None else None

        pkd_val: float | None = None
        if pred_pkd is not None:
            raw_v = float(pred_pkd[idx])
            if math.isfinite(raw_v):
                pkd_val = raw_v

        clip_val: float | None = None
        if clip_sim is not None:
            raw_v = float(clip_sim[idx])
            if math.isfinite(raw_v):
                clip_val = raw_v

        vina_val: float | None = None
        if vina_scores is not None:
            raw_v = float(vina_scores[idx])
            if math.isfinite(raw_v):
                vina_val = raw_v

        if not mpo.valid:
            scores_out.append(
                CNSCompositeScore(
                    smiles=smi,
                    canonical_smiles=None,
                    valid=False,
                    mpo=mpo,
                    admet=admet,
                    pred_pkd=pkd_val,
                    clip_sim=clip_val,
                    vina_score=vina_val,
                    composite_reward=0.0,
                )
            )
            continue

        components: list[tuple[float, float]] = [
            (eff_weights.get("cns_mpo_v2", 0.0), mpo.cns_mpo_v2 / 6.0),
            (eff_weights.get("qed", 0.0), mpo.qed),
            (eff_weights.get("sa", 0.0), mpo.sa_norm),
            (eff_weights.get("pains", 0.0), 1.0 if mpo.pains_free else 0.0),
        ]
        if admet is not None and admet.valid:
            if ad_damp and admet_suite is not None and admet_suite.ad_threshold > 0.0:
                thr = admet_suite.ad_threshold
                w_b = min(1.0, admet.bbb_ad_tanimoto / thr)
                w_h = min(1.0, admet.herg_ad_tanimoto / thr)
                w_p = min(1.0, admet.pgp_ad_tanimoto / thr)
                eff_bbb = 0.5 + w_b * (admet.bbb_prob - 0.5)
                eff_herg_safe = 0.5 + w_h * ((1.0 - admet.herg_prob) - 0.5)
                eff_pgp_safe = 0.5 + w_p * ((1.0 - admet.pgp_prob) - 0.5)
            else:
                eff_bbb = admet.bbb_prob
                eff_herg_safe = 1.0 - admet.herg_prob
                eff_pgp_safe = 1.0 - admet.pgp_prob
            components.append((eff_weights.get("bbb", 0.0), eff_bbb))
            components.append((eff_weights.get("herg_safe", 0.0), eff_herg_safe))
            components.append((eff_weights.get("pgp_safe", 0.0), eff_pgp_safe))
        if pkd_val is not None:
            pkd_norm = float(np.clip((pkd_val - 4.0) / 6.0, 0.0, 1.0))
            components.append((eff_weights.get("pkd", 0.0), pkd_norm))
        if clip_val is not None:
            clip_norm = float(np.clip((clip_val + 1.0) / 2.0, 0.0, 1.0))
            components.append((eff_weights.get("clip", 0.0), clip_norm))
        if vina_val is not None:
            # More negative Vina kcal/mol = stronger affinity; maps [-4, -12] -> [0, 1]
            vina_norm = float(np.clip((-vina_val - 4.0) / 8.0, 0.0, 1.0))
            components.append((eff_weights.get("vina", 0.0), vina_norm))

        w_sum = sum(max(0.0, w) for w, _ in components)
        reward = sum(max(0.0, w) * val for w, val in components) / w_sum if w_sum > 0.0 else 0.0
        if not mpo.pains_free:
            reward *= 0.5
        # Softly dampen reward when hERG cardiotoxicity probability is critically elevated
        if admet is not None and admet.valid and admet.herg_prob >= 0.70:
            reward *= float(expit(-6.0 * (admet.herg_prob - 0.70))) * 2.0

        scores_out.append(
            CNSCompositeScore(
                smiles=smi,
                canonical_smiles=mpo.canonical_smiles,
                valid=True,
                mpo=mpo,
                admet=admet,
                pred_pkd=pkd_val,
                clip_sim=clip_val,
                vina_score=vina_val,
                composite_reward=float(np.clip(reward, 0.0, 1.0)),
            )
        )

    return scores_out


__all__ = [
    "ADMET_DESCRIPTOR_NAMES",
    "ADMET_FP_BITS",
    "DEFAULT_ADMET_CV_REPORT_PATH",
    "DEFAULT_ADMET_DATA_DIR",
    "DEFAULT_ADMET_MODEL_PATH",
    "DEFAULT_CNS_REWARD_WEIGHTS",
    "ORACLE_SEMANTICS",
    "CNSAdmetPrediction",
    "CNSAdmetSuite",
    "CNSCompositeScore",
    "CalibratedADMETOracle",
    "compute_cns_composite_reward",
    "compute_ece",
    "featurize_admet_smiles",
    "load_b3db_classification",
    "load_tdc_binary_csv",
    "make_scaffold_kfold",
    "train_default_cns_admet_suite",
]
