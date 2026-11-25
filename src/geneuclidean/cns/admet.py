"""Calibrated Bemis-Murcko scaffold-CV CNS ADMET, Avoid-ome, Conformal UQ & Safety-Gated Reward.

Implements pure scikit-learn calibrated ensemble oracles
(``HistGradientBoostingClassifier`` + ``RandomForestClassifier`` + scaffold-disjoint
Platt/isotonic probability calibration + finite-sample Split-Conformal UQ) trained on
1024-bit ECFP4 fingerprints plus 12 RDKit 2D physicochemical descriptors across both the
v3.1 3-endpoint core and the expanded v4.0 12-endpoint industrial CNS Pharma & Avoid-ome
suite:

Core v3.1 endpoints:
1. ``bbb``: Blood-Brain Barrier permeability ``P(BBB+)`` (B3DB, 7,805 compounds).
2. ``pgp``: P-glycoprotein (ABCB1) inhibition ``P(P-gp inhibitor)`` (TDC Pgp_Broccatelli).
3. ``herg``: hERG K+ channel cardiotoxicity ``P(hERG IC50 < 10 uM)`` (TDC hERG_Karim).

New v4.0 industrial DMPK, DDI, safety & Avoid-ome endpoints:
4. ``mdr1_efflux``: True active BBB efflux ``P(MDR1-MDCK ER >= 2.0)`` (Biogen Fang 2023).
5. ``hlm_clearance``: Human liver microsomal clearance ``P(HLM CLint >= 30)`` (Biogen + OpenADMET).
6. ``cyp2d6``: CYP2D6 reversible inhibition ``P(CYP2D6 inhibitor)`` (TDC CYP2D6_Veith).
7. ``cyp3a4``: CYP3A4 reversible inhibition ``P(CYP3A4 inhibitor)`` (TDC CYP3A4_Veith).
8. ``htr2b_avoid``: Human 5-HT2B cardiac valvulopathy kill-switch (ChEMBL 37 CHEMBL1833).
9. ``hrh1_avoid``: Human H1 sedation & weight-gain kill-switch (ChEMBL 37 CHEMBL231).
10. ``dili``: Drug-Induced Liver Injury advisory screen (TDC DILI).
11. ``ames``: ICH M7 Ames mutagenicity screen (TDC AMES).
12. ``chrm3_avoid``: Human M3 peripheral muscarinic counter-screen (ChEMBL 37 CHEMBL245).

Also provides GenEuclidean v4.0's **Multiplicative Safety-Gated CNS Reward**
(``compute_multiplicative_cns_reward``) eliminating additive ADMET-sink bleed and
coupling-reagent artifacts, alongside backward-compatible ``compute_cns_composite_reward``.
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
from scipy import stats
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from geneuclidean.cns.credibility import (
    ConformalPredictionResult,
    SplitConformalBinaryCalibrator,
    build_credibility_dossier,
)
from geneuclidean.cns.medchem_triage import MedChemTriageResult, triage_molecule
from geneuclidean.cns.mpo import CNSMPOResult, compute_cns_mpo, estimate_most_basic_pka
from geneuclidean.data.splits import compute_murcko_scaffold
from geneuclidean.evaluation.metrics import canonicalize_smiles

_DESCRIPTORS: Any = Descriptors
_CRIPPEN: Any = Crippen
_MORGAN_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

DEFAULT_ADMET_DATA_DIR = Path("data/cns_neuro/admet")
DEFAULT_ADMET_MODEL_PATH = Path("data/cns_neuro.nosync/admet_oracles.joblib")
DEFAULT_ADMET_CV_REPORT_PATH = Path("results/tables/cns_admet_cv_metrics.json")

DEFAULT_V4_EXPANSION_DIR = Path("data/cns_neuro.nosync/v4_expansion")
DEFAULT_V4_ADMET_MODEL_PATH = Path("data/cns_neuro.nosync/v4_admet_avoidome_oracles.joblib")
DEFAULT_V4_ADMET_CV_REPORT_PATH = Path("results/tables/v4_admet_avoidome_cv_metrics.json")

V3_ADMET_ENDPOINTS: tuple[str, ...] = ("bbb", "pgp", "herg")
V4_ADMET_ENDPOINTS: tuple[str, ...] = (
    "bbb",
    "pgp",
    "herg",
    "mdr1_efflux",
    "hlm_clearance",
    "cyp2d6",
    "cyp3a4",
    "htr2b_avoid",
    "hrh1_avoid",
    "dili",
    "ames",
    "chrm3_avoid",
)

# Direction: True means higher probability is desirable (BBB+); False means lower is safer.
ENDPOINT_HIGHER_IS_BETTER: dict[str, bool] = {
    "bbb": True,
    "pgp": False,
    "herg": False,
    "mdr1_efflux": False,
    "hlm_clearance": False,
    "cyp2d6": False,
    "cyp3a4": False,
    "htr2b_avoid": False,
    "hrh1_avoid": False,
    "dili": False,
    "ames": False,
    "chrm3_avoid": False,
}

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
        "use mdr1_efflux for active BBB substrate efflux control."
    ),
    "herg": (
        "hERG K+ channel blockade cardiotoxicity P(hERG IC50 < 10 uM) trained on TDC "
        "hERG_Karim (Karim et al. 2021, 13,445 balanced compounds); lower is safer "
        "(critical QT-prolongation gate for psychiatric, pediatric TS, and geriatric AD drugs)."
    ),
    "mdr1_efflux": (
        "True active transcellular BBB efflux P(MDR1-MDCK ER >= 2.0) trained on Biogen "
        "Industrial ADME (Fang et al., JCIM 2023, MIT/CC-BY-4.0); lower is better for "
        "unbound CNS exposure (Kp,uu,brain)."
    ),
    "hlm_clearance": (
        "Human liver microsomal metabolic instability P(HLM CLint >= 30 mL/min/kg) "
        "trained on Biogen Industrial ADME (Fang et al. 2023); lower is metabolically stabler."
    ),
    "cyp2d6": (
        "CYP2D6 inhibition P(CYP2D6 inhibitor) trained on TDC CYP2D6_Veith (13,130 compounds); "
        "lower prevents poor-metabolizer QT/DDI caps on valbenazine, antipsychotics & donepezil."
    ),
    "cyp3a4": (
        "CYP3A4 inhibition P(CYP3A4 inhibitor) trained on TDC CYP3A4_Veith (12,328 compounds); "
        "lower is safer (prevents geriatric AD and psychiatric polypharmacy DDIs)."
    ),
    "htr2b_avoid": (
        "Human 5-HT2B (HTR2B, CHEMBL1833) Avoid-ome kill-switch P(5-HT2B active); "
        "lower is safer (guards against fatal drug-induced cardiac valvulopathy)."
    ),
    "hrh1_avoid": (
        "Human Histamine H1 (HRH1, CHEMBL231) Avoid-ome kill-switch P(H1 active); "
        "lower is safer (guards against sedation and antipsychotic metabolic weight gain)."
    ),
    "dili": (
        "Drug-Induced Liver Injury advisory screen P(DILI+) trained on TDC DILI (475 drugs); "
        "lower is safer (advisory hepatotoxicity filter; compact N requires cautious use)."
    ),
    "ames": (
        "ICH M7 Ames bacterial mutagenicity screen P(Ames+) trained on TDC AMES (7,278 compounds); "
        "lower is safer (mandatory chronic CNS genotoxicity gate paired with Brenk/Lilly alerts)."
    ),
    "chrm3_avoid": (
        "Human Muscarinic M3 (CHRM3, CHEMBL245) Avoid-ome counter-screen P(M3 active); "
        "lower spares peripheral cholinergic GI/cardiac toxicity for CHRM4/CHRM1 programs."
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
    """Build deterministic K-fold Bemis-Murcko scaffold splits with zero scaffold overlap."""
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
# Calibrated Single-Endpoint Oracle with Split-Conformal UQ & Applicability Domain
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
    """Single calibrated HGB + RF CNS ADMET classifier with Split-Conformal UQ & ECFP4 AD."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        conformal_alpha: float = 0.10,
        hgb_max_iter: int = 160,
        rf_n_estimators: int = 160,
        ad_threshold: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.description = description or ORACLE_SEMANTICS.get(name, "")
        self.calibration_method: Literal["sigmoid", "isotonic"] = calibration_method
        self.conformal_alpha: float = float(conformal_alpha)
        self.hgb_max_iter = int(hgb_max_iter)
        self.rf_n_estimators = int(rf_n_estimators)
        self.ad_threshold = float(ad_threshold)
        self.seed = int(seed)

        self._hgb: Any | None = None
        self._rf: Any | None = None
        self._calibrator: _ProbabilityCalibrator = _ProbabilityCalibrator(calibration_method)
        self._conformal_calibrator: SplitConformalBinaryCalibrator | None = None
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
        """Run strict K-fold Bemis-Murcko scaffold CV, fit Platt + Split-Conformal UQ."""
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

        folds = make_scaffold_kfold(valid_smiles, n_splits=n_splits, seed=self.seed)
        scaffolds = [compute_murcko_scaffold(s) for s in valid_smiles]
        unique_scaffolds = len(set(scaffolds))

        oof_raw = np.zeros(n_samples, dtype=np.float64)
        oof_cal = np.zeros(n_samples, dtype=np.float64)
        fold_reports: list[dict[str, float]] = []
        fold_conformal_covs: list[float] = []
        fold_conformal_sizes: list[float] = []
        fold_conformal_ambig: list[float] = []
        leakage_count = 0

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            tr_scafs = {scaffolds[i] for i in tr_idx.tolist()}
            va_scafs = {scaffolds[i] for i in va_idx.tolist()}
            leakage_count += len(tr_scafs & va_scafs)

            tr_smiles = [valid_smiles[i] for i in tr_idx.tolist()]
            inner_folds = make_scaffold_kfold(
                tr_smiles,
                n_splits=5,
                seed=self.seed + 100 + fold_idx,
            )
            in_fit_rel, in_cal_rel = inner_folds[0]
            fit_idx = tr_idx[in_fit_rel]
            cal_idx = tr_idx[in_cal_rel]

            if len(np.unique(y_arr[fit_idx])) < 2 or len(np.unique(y_arr[cal_idx])) < 2:
                fit_idx = tr_idx
                cal_idx = tr_idx

            hgb_f, rf_f = self._build_base_models(seed_offset=fold_idx + 1)
            hgb_f.fit(x_mat[fit_idx], y_arr[fit_idx])
            rf_f.fit(x_mat[fit_idx], y_arr[fit_idx])

            cal_raw = self._predict_ensemble_raw(hgb_f, rf_f, x_mat[cal_idx])
            cal_model = _ProbabilityCalibrator(self.calibration_method).fit(cal_raw, y_arr[cal_idx])
            cal_probs = cal_model.transform(cal_raw)

            va_raw = self._predict_ensemble_raw(hgb_f, rf_f, x_mat[va_idx])
            va_cal = cal_model.transform(va_raw)

            # Leak-free nested Split-Conformal evaluation on scaffold-held-out fold
            alpha = getattr(self, "conformal_alpha", 0.10)
            fold_conf = SplitConformalBinaryCalibrator(alpha=alpha).fit(cal_probs, y_arr[cal_idx])
            conf_eval = fold_conf.evaluate_coverage(va_cal, y_arr[va_idx])
            if math.isfinite(conf_eval["empirical_coverage"]):
                fold_conformal_covs.append(conf_eval["empirical_coverage"])
                fold_conformal_sizes.append(conf_eval["avg_set_size"])
                fold_conformal_ambig.append(conf_eval["ambiguous_rate"])

            oof_raw[va_idx] = va_raw
            oof_cal[va_idx] = va_cal
            fold_reports.append(_classification_metrics(y_arr[va_idx], va_cal))

        if leakage_count != 0:
            raise AssertionError(
                f"Scaffold leakage detected in {self.name} CV: {leakage_count} overlapping groups"
            )

        # Final deployed model trained on 100% of data, Platt + Split-Conformal calibrated on OOF
        hgb_final, rf_final = self._build_base_models(seed_offset=0)
        hgb_final.fit(x_mat, y_arr)
        rf_final.fit(x_mat, y_arr)
        self._hgb = hgb_final
        self._rf = rf_final
        self._calibrator = _ProbabilityCalibrator(self.calibration_method).fit(oof_raw, y_arr)

        alpha = getattr(self, "conformal_alpha", 0.10)
        oof_cal_final = self._calibrator.transform(oof_raw)
        self._conformal_calibrator = SplitConformalBinaryCalibrator(alpha=alpha).fit(
            oof_cal_final, y_arr
        )

        # Evaluate cross-conformal OOF coverage using leave-one-fold-out conformal quantiles
        # (guaranteed exchangeable across scaffold folds for exact OOF coverage reporting)
        oof_conf_covered = 0
        oof_conf_singletons = 0
        oof_conf_ambig = 0
        oof_conf_total_size = 0
        for _fold_idx, (tr_idx, va_idx) in enumerate(folds):
            lofo_conf = SplitConformalBinaryCalibrator(alpha=alpha).fit(
                oof_cal[tr_idx], y_arr[tr_idx]
            )
            preds_c = lofo_conf.predict_batch(oof_cal[va_idx])
            for res, lbl in zip(preds_c, y_arr[va_idx].tolist(), strict=True):
                if lbl in res.prediction_set:
                    oof_conf_covered += 1
                if len(res.prediction_set) == 1:
                    oof_conf_singletons += 1
                elif len(res.prediction_set) == 2:
                    oof_conf_ambig += 1
                oof_conf_total_size += len(res.prediction_set)

        conformal_summary = {
            "alpha": float(alpha),
            "target_coverage": float(1.0 - alpha),
            "empirical_coverage": float(oof_conf_covered / n_samples),
            "singleton_rate": float(oof_conf_singletons / n_samples),
            "ambiguous_rate": float(oof_conf_ambig / n_samples),
            "avg_set_size": float(oof_conf_total_size / n_samples),
            "nested_fold_coverage_mean": (
                float(np.mean(fold_conformal_covs)) if fold_conformal_covs else float("nan")
            ),
            "nested_fold_avg_set_size_mean": (
                float(np.mean(fold_conformal_sizes)) if fold_conformal_sizes else float("nan")
            ),
            "nested_fold_ambiguous_rate_mean": (
                float(np.mean(fold_conformal_ambig)) if fold_conformal_ambig else float("nan")
            ),
            "deployed_q_hat": float(self._conformal_calibrator.q_hat),
        }

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
            "higher_is_better": bool(ENDPOINT_HIGHER_IS_BETTER.get(self.name, False)),
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
            "conformal_uq": conformal_summary,
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

    def predict_conformal_from_features(
        self,
        x_mat: NDArray[np.float32],
        fp_bits: NDArray[np.uint8],
        valid_mask: NDArray[np.bool_],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        list[ConformalPredictionResult | None],
    ]:
        """Predict probabilities, AD, and Split-Conformal prediction sets when calibrated."""
        probs, ad_tan, in_dom = self.predict_from_features(x_mat, fp_bits, valid_mask)
        conf_cal: SplitConformalBinaryCalibrator | None = getattr(
            self, "_conformal_calibrator", None
        )
        conf_out: list[ConformalPredictionResult | None] = []
        for idx in range(len(probs)):
            if not bool(valid_mask[idx]) or conf_cal is None or not conf_cal.is_fitted:
                conf_out.append(None)
            else:
                conf_out.append(conf_cal.predict_one(float(probs[idx])))
        return probs, ad_tan, in_dom, conf_out

    def predict(
        self,
        smiles_list: Sequence[str],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Predict calibrated probabilities, AD Tanimoto max, and in-domain flags for SMILES."""
        x_mat, fp_bits, valid_mask, _ = featurize_admet_smiles(smiles_list)
        return self.predict_from_features(x_mat, fp_bits, valid_mask)

    def predict_conformal(
        self,
        smiles_list: Sequence[str],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        list[ConformalPredictionResult | None],
    ]:
        """Predict calibrated probabilities, AD, and Split-Conformal sets for SMILES."""
        x_mat, fp_bits, valid_mask, _ = featurize_admet_smiles(smiles_list)
        return self.predict_conformal_from_features(x_mat, fp_bits, valid_mask)

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
        conf_cal: SplitConformalBinaryCalibrator | None = getattr(
            self, "_conformal_calibrator", None
        )
        if conf_cal is not None and conf_cal.is_fitted and len(y_keep) > 0:
            report["conformal_external"] = conf_cal.evaluate_coverage(probs, y_keep)

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
        state["_ref_fp_float_cache"] = None
        state["_ref_fp_norms_cache"] = None
        return state


# ---------------------------------------------------------------------------
# Dataset Loaders (v3.1 B3DB/TDC + v4.0 Biogen, OpenADMET, Avoid-ome, Kp,uu)
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


@dataclass(frozen=True)
class PharmaAdmetPrediction:
    """Expanded v4.0 multi-endpoint ADMET & Avoid-ome prediction with AD & Split-Conformal UQ."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    probabilities: dict[str, float]
    ad_tanimoto: dict[str, float]
    in_domain: dict[str, bool]
    conformal: dict[str, ConformalPredictionResult | None]
    in_domain_all: bool
    admet_v4_pass: bool

    @property
    def conformal_sets(self) -> dict[str, ConformalPredictionResult | None]:
        """Alias for ``conformal`` mapping endpoint name to its Split-Conformal set."""
        return self.conformal

    def to_dict(self) -> dict[str, Any]:
        """Serialize v4.0 prediction to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "probabilities": dict(self.probabilities),
            "ad_tanimoto": dict(self.ad_tanimoto),
            "in_domain": dict(self.in_domain),
            "conformal": {
                k: (v.to_dict() if v is not None else None) for k, v in self.conformal.items()
            },
            "in_domain_all": self.in_domain_all,
            "admet_v4_pass": self.admet_v4_pass,
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
    smiles_col: str | None = None,
    label_col: str = "Y",
) -> tuple[list[str], list[int], dict[str, int]]:
    """Load and deduplicate a TDC binary ADMET CSV/TSV file (handles ``Drug`` or ``X``)."""
    df = pd.read_csv(path)
    if smiles_col is None:
        if "Drug" in df.columns:
            smiles_col = "Drug"
        elif "X" in df.columns:
            smiles_col = "X"
        elif "SMILES" in df.columns:
            smiles_col = "SMILES"
        else:
            raise KeyError(f"Could not locate SMILES column in {path}; columns={list(df.columns)}")
    smiles = df[smiles_col].astype(str).tolist()
    labels = [int(round(float(v))) for v in df[label_col].tolist()]
    return _deduplicate_binary_dataset(smiles, labels)


def load_biogen_binary_endpoint(
    path: Path,
    *,
    value_col: str,
    log10_threshold: float,
    smiles_col: str = "SMILES",
) -> tuple[list[str], list[int], dict[str, int]]:
    """Load and binarize a continuous log10 endpoint from Biogen ``ADME_public_set_3521.csv``."""
    df = pd.read_csv(path)
    sub = df[[smiles_col, value_col]].dropna()
    smiles = sub[smiles_col].astype(str).tolist()
    vals = pd.to_numeric(sub[value_col], errors="coerce")
    keep = vals.notna()
    clean_smi = [s for s, k in zip(smiles, keep.tolist(), strict=True) if k]
    labels = [1 if float(v) >= float(log10_threshold) else 0 for v in vals[keep].tolist()]
    return _deduplicate_binary_dataset(clean_smi, labels)


def load_avoidome_binary_dataset(
    avoidome_csv: Path,
    *,
    background_cns_csv: Path | None = None,
    exclude_background_targets: Sequence[str] = (),
    seed: int = 42,
) -> tuple[list[str], list[int], dict[str, int]]:
    """Load a ChEMBL human Avoid-ome target CSV and balance negatives with orthogonal CNS decoys.

    Drops ambiguous/discordant rows (``binary_label == -1``), keeps all confirmed ChEMBL
    actives (``binary_label == 1``) and confirmed ChEMBL inactives (``binary_label == 0``),
    and optionally augments the negative class with a deterministic sample of orthogonal
    human CNS actives from unrelated target families (strictly excluding any compound
    present in the Avoid-ome file) so binary classifiers achieve calibrated discrimination.
    """
    df = pd.read_csv(avoidome_csv)
    valid_df = df[df["binary_label"].isin([0, 1])].copy()
    smiles = valid_df["canonical_smiles"].astype(str).tolist()
    labels = [int(v) for v in valid_df["binary_label"].tolist()]

    # Track ALL SMILES mentioned in avoidome file (including discordant -1) so they are never
    # added as background negatives.
    all_seen_can: set[str] = set()
    for raw in df["canonical_smiles"].astype(str).tolist():
        c = canonicalize_smiles(raw, isomeric=True, keep_largest_fragment=True)
        if c is not None:
            all_seen_can.add(c)

    base_smi, base_y, base_stats = _deduplicate_binary_dataset(smiles, labels)
    n_pos = sum(1 for y in base_y if y == 1)
    n_explicit_neg = sum(1 for y in base_y if y == 0)
    n_needed_bg = max(0, n_pos - n_explicit_neg)
    n_bg_added = 0

    if background_cns_csv is not None and background_cns_csv.exists() and n_needed_bg > 0:
        bg_df = pd.read_csv(background_cns_csv)
        excl = {t.strip().upper() for t in exclude_background_targets}
        if "target_symbol" in bg_df.columns and excl:
            bg_df = bg_df[~bg_df["target_symbol"].astype(str).str.upper().isin(excl)]
        smi_col = "canonical_smiles" if "canonical_smiles" in bg_df.columns else "SMILES"
        bg_candidates: list[str] = []
        seen_bg: set[str] = set()
        for raw in bg_df[smi_col].astype(str).tolist():
            c = canonicalize_smiles(raw, isomeric=True, keep_largest_fragment=True)
            if c is None or c in all_seen_can or c in seen_bg:
                continue
            seen_bg.add(c)
            bg_candidates.append(c)

        bg_candidates.sort()
        rng = np.random.default_rng(seed)
        if len(bg_candidates) > n_needed_bg:
            pick_idx = rng.choice(len(bg_candidates), size=n_needed_bg, replace=False)
            chosen_bg = [bg_candidates[int(i)] for i in sorted(pick_idx.tolist())]
        else:
            chosen_bg = bg_candidates
        base_smi.extend(chosen_bg)
        base_y.extend([0] * len(chosen_bg))
        n_bg_added = len(chosen_bg)

    clean_smi, clean_y, final_stats = _deduplicate_binary_dataset(base_smi, base_y)
    final_stats["n_explicit_chembl_negatives"] = int(n_explicit_neg)
    final_stats["n_orthogonal_cns_negatives_added"] = int(n_bg_added)
    final_stats["n_raw_avoidome_rows"] = int(base_stats["n_raw"])
    return clean_smi, clean_y, final_stats


# ---------------------------------------------------------------------------
# Unified CNS ADMET & Avoid-ome Suite (v3.1 3-Endpoint & v4.0 10-12 Endpoint)
# ---------------------------------------------------------------------------


class CNSAdmetSuite:
    """Unified calibrated CNS ADMET & Avoid-ome Suite (v3.1 3-endpoint & v4.0 expanded)."""

    def __init__(
        self,
        *,
        endpoints: Sequence[str] = V3_ADMET_ENDPOINTS,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        conformal_alpha: float = 0.10,
        hgb_max_iter: int = 160,
        rf_n_estimators: int = 160,
        ad_threshold: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.ad_threshold = float(ad_threshold)
        self.conformal_alpha = float(conformal_alpha)
        self.calibration_method: Literal["sigmoid", "isotonic"] = calibration_method
        self.hgb_max_iter = int(hgb_max_iter)
        self.rf_n_estimators = int(rf_n_estimators)
        self.seed = int(seed)
        self.oracles: dict[str, CalibratedADMETOracle] = {
            name: CalibratedADMETOracle(
                name=name,
                description=ORACLE_SEMANTICS.get(name, f"Calibrated CNS ADMET oracle for {name}"),
                calibration_method=calibration_method,
                conformal_alpha=conformal_alpha,
                hgb_max_iter=hgb_max_iter,
                rf_n_estimators=rf_n_estimators,
                ad_threshold=ad_threshold,
                seed=seed + offset,
            )
            for offset, name in enumerate(endpoints)
        }
        self.cv_report: dict[str, Any] = {}

    @property
    def fitted_endpoints(self) -> tuple[str, ...]:
        """Return tuple of endpoint names whose underlying ensemble models are fitted."""
        return tuple(name for name, oracle in self.oracles.items() if oracle._hgb is not None)

    @property
    def endpoint_names(self) -> tuple[str, ...]:
        """Return ordered tuple of all registered endpoint names in this suite."""
        return tuple(self.oracles.keys())

    def fit_endpoint(
        self,
        endpoint: str,
        smiles_list: Sequence[str],
        labels: Sequence[int],
        *,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Train and K-fold Bemis-Murcko scaffold-CV evaluate one ADMET/Avoid-ome oracle."""
        if endpoint not in self.oracles:
            if endpoint in ORACLE_SEMANTICS:
                self.oracles[endpoint] = CalibratedADMETOracle(
                    name=endpoint,
                    description=ORACLE_SEMANTICS[endpoint],
                    calibration_method=getattr(self, "calibration_method", "sigmoid"),
                    conformal_alpha=getattr(self, "conformal_alpha", 0.10),
                    hgb_max_iter=getattr(self, "hgb_max_iter", 160),
                    rf_n_estimators=getattr(self, "rf_n_estimators", 160),
                    ad_threshold=self.ad_threshold,
                    seed=getattr(self, "seed", 42) + len(self.oracles),
                )
            else:
                known = ", ".join(sorted(ORACLE_SEMANTICS.keys()))
                raise KeyError(f"Unknown ADMET endpoint {endpoint!r}; expected one of: {known}")
        metrics = self.oracles[endpoint].fit_and_evaluate(smiles_list, labels, n_splits=n_splits)
        self.cv_report[endpoint] = metrics
        return metrics

    def fit_from_directory(
        self,
        admet_dir: str | Path = DEFAULT_ADMET_DATA_DIR,
        *,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Train the core 3 v3.1 CNS oracles from ``data/cns_neuro/admet/``."""
        root = Path(admet_dir)
        b3db_path = root / "B3DB_classification.tsv"
        martins_path = root / "tdc_BBB_Martins.csv"
        pgp_path = root / "tdc_Pgp_Broccatelli.csv"
        herg_karim_path = root / "tdc_hERG_Karim.csv"
        herg_wang_path = root / "tdc_hERG.csv"

        for required in (b3db_path, pgp_path, herg_karim_path):
            if not required.exists():
                raise FileNotFoundError(f"Required CNS ADMET dataset not found: {required}")

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

        pgp_smi, pgp_y, pgp_clean_stats = load_tdc_binary_csv(pgp_path)
        pgp_cv = self.fit_endpoint("pgp", pgp_smi, pgp_y, n_splits=n_splits)
        pgp_cv["data_hygiene"] = pgp_clean_stats

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
                "ensemble with scaffold-disjoint Platt calibration, Split-Conformal UQ & ECFP4 AD"
            ),
            "endpoints": {
                "bbb": bbb_cv,
                "pgp": pgp_cv,
                "herg": herg_cv,
            },
        }
        return dict(self.cv_report)

    def fit_v4_from_directories(
        self,
        *,
        v3_admet_dir: str | Path = DEFAULT_ADMET_DATA_DIR,
        v4_expansion_dir: str | Path = DEFAULT_V4_EXPANSION_DIR,
        background_cns_csv: str | Path = Path(
            "data/cns_neuro/chembl/human_cns_actives_12targets.csv"
        ),
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Train & calibrate the full v4.0 10-12 endpoint Pharma ADMET & Avoid-ome suite."""
        v3_report = self.fit_from_directory(v3_admet_dir, n_splits=n_splits)
        endpoints_report: dict[str, Any] = dict(v3_report["endpoints"])

        exp_root = Path(v4_expansion_dir)
        admet_v4 = exp_root / "admet"
        avoid_v4 = exp_root / "avoidome"
        bg_path = Path(background_cns_csv)

        biogen_csv = admet_v4 / "ADME_public_set_3521.csv"
        if biogen_csv.exists():
            # 4. MDR1-MDCK active efflux (ER >= 2.0 -> log10 >= 0.30103)
            mdr_smi, mdr_y, mdr_stats = load_biogen_binary_endpoint(
                biogen_csv,
                value_col="LOG MDR1-MDCK ER (B-A/A-B)",
                log10_threshold=math.log10(2.0),
            )
            mdr_cv = self.fit_endpoint("mdr1_efflux", mdr_smi, mdr_y, n_splits=n_splits)
            mdr_cv["data_hygiene"] = mdr_stats
            mdr_cv["cutoff_definition"] = "MDR1-MDCK Efflux Ratio (B-A/A-B) >= 2.0"
            endpoints_report["mdr1_efflux"] = mdr_cv

            # 5. HLM metabolic clearance (CLint >= 30 mL/min/kg -> log10 >= 1.47712)
            hlm_smi, hlm_y, hlm_stats = load_biogen_binary_endpoint(
                biogen_csv,
                value_col="LOG HLM_CLint (mL/min/kg)",
                log10_threshold=math.log10(30.0),
            )
            hlm_cv = self.fit_endpoint("hlm_clearance", hlm_smi, hlm_y, n_splits=n_splits)
            hlm_cv["data_hygiene"] = hlm_stats
            hlm_cv["cutoff_definition"] = "Biogen HLM CLint >= 30 mL/min/kg"
            endpoints_report["hlm_clearance"] = hlm_cv

        # 6-9. TDC CYP2D6, CYP3A4, DILI, AMES
        tdc_map: tuple[tuple[str, str], ...] = (
            ("cyp2d6", "tdc_CYP2D6_Veith.csv"),
            ("cyp3a4", "tdc_CYP3A4_Veith.csv"),
            ("dili", "tdc_DILI.csv"),
            ("ames", "tdc_AMES.csv"),
        )
        for ep_key, fname in tdc_map:
            fpath = admet_v4 / fname
            if not fpath.exists():
                continue
            smi, y, hygiene = load_tdc_binary_csv(fpath)
            ep_cv = self.fit_endpoint(ep_key, smi, y, n_splits=n_splits)
            ep_cv["data_hygiene"] = hygiene
            endpoints_report[ep_key] = ep_cv

        # 10-12. ChEMBL 37 Human Avoid-ome Kill-Switches (HTR2B, HRH1, CHRM3)
        avoid_specs: tuple[tuple[str, str, tuple[str, ...]], ...] = (
            ("htr2b_avoid", "avoidome_HTR2B_CHEMBL1833.csv", ("HTR2A", "DRD2", "DRD3")),
            ("hrh1_avoid", "avoidome_HRH1_CHEMBL231.csv", ("HRH3", "HTR2A", "DRD2")),
            ("chrm3_avoid", "avoidome_CHRM3_CHEMBL245.csv", ("CHRM4",)),
        )
        for ep_key, fname, excl_targets in avoid_specs:
            fpath = avoid_v4 / fname
            if not fpath.exists():
                continue
            smi, y, hygiene = load_avoidome_binary_dataset(
                fpath,
                background_cns_csv=bg_path if bg_path.exists() else None,
                exclude_background_targets=excl_targets,
                seed=self.seed + len(endpoints_report),
            )
            ep_cv = self.fit_endpoint(ep_key, smi, y, n_splits=n_splits)
            ep_cv["data_hygiene"] = hygiene
            endpoints_report[ep_key] = ep_cv

        # External Rat Kp,uu,brain calibration benchmark (239 compounds, academic eval only)
        kpuu_csv = admet_v4 / "rat_kpuu_external_calibration.csv"
        kpuu_eval: dict[str, Any] | None = None
        if kpuu_csv.exists():
            kpuu_eval = evaluate_kpuu_external_benchmark(self, kpuu_csv)

        self.cv_report = {
            "schema_version": "geneuclidean-v4.0-admet-avoidome-cv",
            "n_endpoints": len(endpoints_report),
            "endpoint_names": list(endpoints_report.keys()),
            "features": {
                "ecfp4_bits": ADMET_FP_BITS,
                "descriptors": list(ADMET_DESCRIPTOR_NAMES),
                "total_dim": ADMET_FP_BITS + len(ADMET_DESCRIPTOR_NAMES),
            },
            "model_family": (
                "Calibrated 50/50 HistGradientBoostingClassifier + RandomForestClassifier "
                "ensemble with 5-fold Bemis-Murcko scaffold-CV (0 leakage), Platt sigmoid "
                "calibration, Split-Conformal UQ (90%), and ECFP4 Applicability Domain"
            ),
            "endpoints": endpoints_report,
            "external_rat_kpuu_benchmark": kpuu_eval,
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

    def predict_v4(
        self,
        smiles_list: Sequence[str],
        *,
        thresholds: Mapping[str, float] | None = None,
    ) -> list[PharmaAdmetPrediction]:
        """Predict all fitted v4.0 endpoints with ECFP4 AD and Split-Conformal prediction sets."""
        x_mat, fp_bits, valid_mask, canonical = featurize_admet_smiles(smiles_list)
        fitted = self.fitted_endpoints

        per_ep_probs: dict[str, NDArray[np.float64]] = {}
        per_ep_ad: dict[str, NDArray[np.float64]] = {}
        per_ep_in: dict[str, NDArray[np.bool_]] = {}
        per_ep_conf: dict[str, list[ConformalPredictionResult | None]] = {}

        for ep in fitted:
            p, ad, ind, conf = self.oracles[ep].predict_conformal_from_features(
                x_mat, fp_bits, valid_mask
            )
            per_ep_probs[ep] = p
            per_ep_ad[ep] = ad
            per_ep_in[ep] = ind
            per_ep_conf[ep] = conf

        default_gates: dict[str, float] = {
            "bbb": 0.70,
            "herg": 0.40,
            "mdr1_efflux": 0.45,
            "pgp": 0.75,
            "hlm_clearance": 0.50,
            "cyp2d6": 0.40,
            "cyp3a4": 0.45,
            "htr2b_avoid": 0.30,
            "hrh1_avoid": 0.35,
            "dili": 0.65,
            "ames": 0.50,
            "chrm3_avoid": 0.40,
        }
        if thresholds is not None:
            default_gates.update({k: float(v) for k, v in thresholds.items()})

        out: list[PharmaAdmetPrediction] = []
        for idx, raw_smi in enumerate(smiles_list):
            if not bool(valid_mask[idx]):
                out.append(
                    PharmaAdmetPrediction(
                        smiles=raw_smi,
                        canonical_smiles=None,
                        valid=False,
                        probabilities={},
                        ad_tanimoto={},
                        in_domain={},
                        conformal={},
                        in_domain_all=False,
                        admet_v4_pass=False,
                    )
                )
                continue

            probs_d = {ep: float(per_ep_probs[ep][idx]) for ep in fitted}
            ad_d = {ep: float(per_ep_ad[ep][idx]) for ep in fitted}
            in_d = {ep: bool(per_ep_in[ep][idx]) for ep in fitted}
            conf_d = {ep: per_ep_conf[ep][idx] for ep in fitted}

            gate_pass = True
            for ep, val in probs_d.items():
                thr = default_gates.get(ep)
                if thr is None:
                    continue
                if ENDPOINT_HIGHER_IS_BETTER.get(ep, False):
                    if val < thr:
                        gate_pass = False
                elif val > thr:
                    gate_pass = False

            out.append(
                PharmaAdmetPrediction(
                    smiles=raw_smi,
                    canonical_smiles=canonical[idx],
                    valid=True,
                    probabilities=probs_d,
                    ad_tanimoto=ad_d,
                    in_domain=in_d,
                    conformal=conf_d,
                    in_domain_all=all(in_d.values()) if in_d else False,
                    admet_v4_pass=gate_pass,
                )
            )
        return out

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
                "Train it via train_default_cns_admet_suite() or train_v4_pharma_admet_suite()."
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


def evaluate_kpuu_external_benchmark(
    suite: CNSAdmetSuite,
    kpuu_csv: str | Path,
) -> dict[str, Any]:
    """Evaluate ``mdr1_efflux``, ``bbb``, and ``CNS MPO v2`` vs external rat ``Kp,uu,brain``."""
    df = pd.read_csv(kpuu_csv)
    smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
    smiles = df[smi_col].astype(str).tolist()
    log_kpuu = pd.to_numeric(df["logKpuu"], errors="coerce").to_numpy(dtype=np.float64)
    y_high_kpuu = (
        pd.to_numeric(df["kpuu_ge_0p3"], errors="coerce").to_numpy(dtype=np.int64)
        if "kpuu_ge_0p3" in df.columns
        else (np.power(10.0, log_kpuu) >= 0.30).astype(np.int64)
    )

    mpos = [compute_cns_mpo(s) for s in smiles]
    valid_idx = [i for i, m in enumerate(mpos) if m.valid and math.isfinite(float(log_kpuu[i]))]
    valid_smiles = [smiles[i] for i in valid_idx]
    valid_log_kpuu = log_kpuu[valid_idx]
    valid_y_high = y_high_kpuu[valid_idx]
    mpo_v2 = np.asarray([mpos[i].cns_mpo_v2 for i in valid_idx], dtype=np.float64)

    preds = suite.predict_v4(valid_smiles)

    def _corr_block(
        scores: NDArray[np.float64],
        higher_means_higher_kpuu: bool,
    ) -> dict[str, float]:
        sp_r, sp_p = stats.spearmanr(scores, valid_log_kpuu)
        pe_r, pe_p = stats.pearsonr(scores, valid_log_kpuu)
        auc_score = scores if higher_means_higher_kpuu else (1.0 - scores)
        auc = (
            float(roc_auc_score(valid_y_high, auc_score))
            if len(np.unique(valid_y_high)) >= 2
            else float("nan")
        )
        return {
            "spearman_rho": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
            "roc_auc_kpuu_ge_0p3": auc,
        }

    report: dict[str, Any] = {
        "benchmark_file": str(kpuu_csv),
        "role": "Academic external calibration benchmark only (NOT used in model training)",
        "n_valid_compounds": int(len(valid_smiles)),
        "n_high_kpuu_ge_0p3": int(valid_y_high.sum()),
        "cns_mpo_v2": _corr_block(mpo_v2 / 6.0, higher_means_higher_kpuu=True),
    }

    if "bbb" in suite.fitted_endpoints:
        bbb_p = np.asarray([p.probabilities["bbb"] for p in preds], dtype=np.float64)
        report["bbb_prob"] = _corr_block(bbb_p, higher_means_higher_kpuu=True)
    if "mdr1_efflux" in suite.fitted_endpoints:
        mdr_p = np.asarray([p.probabilities["mdr1_efflux"] for p in preds], dtype=np.float64)
        report["mdr1_efflux_prob"] = _corr_block(mdr_p, higher_means_higher_kpuu=False)
        report["mdr1_non_efflux_prob"] = _corr_block(1.0 - mdr_p, higher_means_higher_kpuu=True)
    if "bbb" in suite.fitted_endpoints and "mdr1_efflux" in suite.fitted_endpoints:
        bbb_p = np.asarray([p.probabilities["bbb"] for p in preds], dtype=np.float64)
        mdr_safe = 1.0 - np.asarray(
            [p.probabilities["mdr1_efflux"] for p in preds], dtype=np.float64
        )
        joint = np.cbrt(
            np.clip(mpo_v2 / 6.0, 1e-4, 1.0)
            * np.clip(bbb_p, 1e-4, 1.0)
            * np.clip(mdr_safe, 1e-4, 1.0)
        )
        report["joint_mpo_bbb_mdr1_exposure_index"] = _corr_block(
            joint, higher_means_higher_kpuu=True
        )
    return report


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


def train_v4_pharma_admet_suite(
    *,
    v3_admet_dir: str | Path = DEFAULT_ADMET_DATA_DIR,
    v4_expansion_dir: str | Path = DEFAULT_V4_EXPANSION_DIR,
    background_cns_csv: str | Path = Path("data/cns_neuro/chembl/human_cns_actives_12targets.csv"),
    model_out: str | Path = DEFAULT_V4_ADMET_MODEL_PATH,
    report_out: str | Path = DEFAULT_V4_ADMET_CV_REPORT_PATH,
    n_splits: int = 5,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    conformal_alpha: float = 0.10,
    seed: int = 42,
) -> tuple[CNSAdmetSuite, dict[str, Any]]:
    """Train, scaffold-CV calibrate, Conformal-verify, and persist the v4.0 Pharma ADMET suite."""
    suite = CNSAdmetSuite(
        endpoints=V4_ADMET_ENDPOINTS,
        calibration_method=calibration_method,
        conformal_alpha=conformal_alpha,
        seed=seed,
    )
    report = suite.fit_v4_from_directories(
        v3_admet_dir=v3_admet_dir,
        v4_expansion_dir=v4_expansion_dir,
        background_cns_csv=background_cns_csv,
        n_splits=n_splits,
    )
    saved_model = suite.save(model_out)
    report["model_artifact"] = str(saved_model)
    report["model_size_mb"] = round(saved_model.stat().st_size / (1024.0 * 1024.0), 3)

    # Attach FDA Jan 2025 / Joint FDA-EMA Jan 2026 Credibility Dossier metadata
    report["credibility_dossier"] = build_credibility_dossier(
        model_name="GenEuclidean v4.0 Pharma ADMET & Avoid-ome Oracle Suite",
        model_version="4.0.0",
        admet_cv_metrics=report,
        provenance_files={"v4_admet_oracles_joblib": saved_model},
    )

    report_path = Path(report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return suite, report


# ---------------------------------------------------------------------------
# Composite CNS Reward (v3.1 Additive — Kept 100% Backward-Compatible)
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
    """Compute v3.1 additive CNS composite reward in ``[0.0, 1.0]`` (backward-compatible)."""
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
            vina_norm = float(np.clip((-vina_val - 4.0) / 8.0, 0.0, 1.0))
            components.append((eff_weights.get("vina", 0.0), vina_norm))

        w_sum = sum(max(0.0, w) for w, _ in components)
        reward = sum(max(0.0, w) * val for w, val in components) / w_sum if w_sum > 0.0 else 0.0
        if not mpo.pains_free:
            reward *= 0.5
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


# ---------------------------------------------------------------------------
# GenEuclidean v4.0 Multiplicative Safety-Gated CNS Reward (Anti-ADMET-Sink)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiplicativeCNSReward:
    """Decomposed v4.0 Multiplicative Safety-Gated CNS reward in ``[0.0, 1.0]``.

    Implements:
        ``R_mult(m | P_t) = G_hygiene * G_safety^gamma_s * G_target_sel^gamma_t * G_3D^gamma_g``

    Hard-vetoes coupling reagents / PAINS / reactive warheads (``G_hygiene == 0.0``, e.g.,
    DCU -> ``0.0``) and suppresses cross-target ADMET sinks (e.g., high-MPO VMAT2 scaffolds
    with low on-target ``P(PDE10A)`` score ``<= 0.15``).
    """

    smiles: str
    canonical_smiles: str | None
    valid: bool
    hygiene_gate: float
    safety_gate: float
    target_sel_gate: float
    geometry_3d_gate: float
    total_reward: float
    mpo: CNSMPOResult
    triage: MedChemTriageResult
    admet_v4: PharmaAdmetPrediction | None
    on_target_prob: float | None
    off_target_max_prob: float | None
    pred_pkd: float | None
    clip_sim: float | None
    vina_score: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize multiplicative reward breakdown to a JSON-compatible dict."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "hygiene_gate": self.hygiene_gate,
            "safety_gate": self.safety_gate,
            "target_sel_gate": self.target_sel_gate,
            "geometry_3d_gate": self.geometry_3d_gate,
            "total_reward": self.total_reward,
            "cns_mpo_v2": self.mpo.cns_mpo_v2,
            "lilly_demerit_score": self.triage.lilly_demerit_score,
            "reagent_quarantine_free": self.triage.reagent_quarantine_free,
            "pains_free": self.triage.pains_free,
            "on_target_prob": self.on_target_prob,
            "off_target_max_prob": self.off_target_max_prob,
            "pred_pkd": self.pred_pkd,
            "clip_sim": self.clip_sim,
            "vina_score": self.vina_score,
            "admet_probabilities": (
                dict(self.admet_v4.probabilities) if self.admet_v4 is not None else None
            ),
        }


def _damped_endpoint_desirability(
    ep_name: str,
    prob: float,
    ad_tan: float,
    ad_threshold: float,
    *,
    ad_damp: bool,
) -> float:
    """Map an oracle probability (with optional AD damping) to desirability in ``[0.05, 1.0]``."""
    raw_desir = prob if ENDPOINT_HIGHER_IS_BETTER.get(ep_name, False) else (1.0 - prob)
    if ad_damp and ad_threshold > 0.0:
        w = min(1.0, max(0.0, ad_tan) / ad_threshold)
        eff = 0.5 + w * (raw_desir - 0.5)
    else:
        eff = raw_desir
    # Apply extra steepening for critical cardiac kill-switches (hERG & 5-HT2B)
    if ep_name in ("herg", "htr2b_avoid") and eff < 0.50:
        eff *= float(expit(6.0 * (eff - 0.35)))
    return float(np.clip(eff, 0.05, 1.0))


def compute_multiplicative_cns_reward(
    smiles_list: Sequence[str],
    *,
    on_target_prob: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    off_target_max_prob: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    pred_pkd: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    clip_sim: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    vina_scores: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    admet_suite: CNSAdmetSuite | None = None,
    ad_damp: bool = True,
    gamma_safety: float = 0.55,
    gamma_target: float = 1.00,
    gamma_3d: float = 0.35,
) -> list[MultiplicativeCNSReward]:
    """Compute GenEuclidean v4.0 Multiplicative Safety-Gated CNS Reward in ``[0.0, 1.0]``.

    Formula:
        ``R_mult(m | P_t) = G_hygiene * G_safety^gamma_s * G_target_sel^gamma_t * G_3D^gamma_g``

    - ``G_hygiene(m)``: From ``medchem_triage.triage_molecule(m).hygiene_gate`` (exact ``0.0``
      for DCU/DIU/reagents, PAINS, reactive warheads, or invalid SMILES).
    - ``G_safety(m)``: Geometric mean of Pfizer CNS MPO v2, QED/SA, and all fitted AD-damped
      ADMET & Avoid-ome safety gates (``bbb``, ``herg``, ``mdr1_efflux``/``pgp``,
      ``hlm_clearance``, ``cyp2d6``, ``cyp3a4``, ``htr2b_avoid``, ``hrh1_avoid``, ``ames``).
    - ``G_target_sel(m, t)``: On-target probability gated by smooth off-target selectivity
      margin ``clip(p_on, 0.02, 1.0) * (0.20 + 0.80 * expit(4.0 * (p_on - p_off)))``,
      preventing compensatory high-MPO off-target ADMET sinks (e.g. a VMAT2 scaffold with
      ``MPO=5.94``, ``P(PDE10A)=0.01``, ``P(VMAT2)=0.90`` collapses to ``<= 0.05`` on PDE10A!).
    - ``G_3D(m, P_t)``: Mean normalized 3D pocket score from ``pred_pkd``, ``clip_sim``,
      and/or ``vina_scores`` (defaults to ``1.0`` when omitted).
    """
    n = len(smiles_list)
    for label, seq in (
        ("on_target_prob", on_target_prob),
        ("off_target_max_prob", off_target_max_prob),
        ("pred_pkd", pred_pkd),
        ("clip_sim", clip_sim),
        ("vina_scores", vina_scores),
    ):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{label} length ({len(seq)}) != smiles_list length ({n})")

    mpo_results = [compute_cns_mpo(s) for s in smiles_list]
    triage_results = [triage_molecule(s) for s in smiles_list]
    admet_v4_results = admet_suite.predict_v4(smiles_list) if admet_suite is not None else None
    ad_thr = admet_suite.ad_threshold if admet_suite is not None else 0.30

    out: list[MultiplicativeCNSReward] = []
    for idx, smi in enumerate(smiles_list):
        mpo = mpo_results[idx]
        triage = triage_results[idx]
        ad_pred = admet_v4_results[idx] if admet_v4_results is not None else None

        p_on = (
            float(on_target_prob[idx])
            if on_target_prob is not None and math.isfinite(float(on_target_prob[idx]))
            else None
        )
        p_off = (
            float(off_target_max_prob[idx])
            if off_target_max_prob is not None and math.isfinite(float(off_target_max_prob[idx]))
            else None
        )
        pkd_v = (
            float(pred_pkd[idx])
            if pred_pkd is not None and math.isfinite(float(pred_pkd[idx]))
            else None
        )
        clip_v = (
            float(clip_sim[idx])
            if clip_sim is not None and math.isfinite(float(clip_sim[idx]))
            else None
        )
        vina_v = (
            float(vina_scores[idx])
            if vina_scores is not None and math.isfinite(float(vina_scores[idx]))
            else None
        )

        if not mpo.valid or not triage.valid:
            out.append(
                MultiplicativeCNSReward(
                    smiles=smi,
                    canonical_smiles=None,
                    valid=False,
                    hygiene_gate=0.0,
                    safety_gate=0.0,
                    target_sel_gate=0.0,
                    geometry_3d_gate=0.0,
                    total_reward=0.0,
                    mpo=mpo,
                    triage=triage,
                    admet_v4=ad_pred,
                    on_target_prob=p_on,
                    off_target_max_prob=p_off,
                    pred_pkd=pkd_v,
                    clip_sim=clip_v,
                    vina_score=vina_v,
                )
            )
            continue

        # 1. Hygiene gate G_hygiene(m) in [0.0, 1.0]
        g_hygiene = float(triage.hygiene_gate)

        # 2. Safety & DMPK gate G_safety(m) via weighted geometric mean in [0.05, 1.0]
        safety_factors: list[tuple[float, float]] = [
            (2.0, float(np.clip(mpo.cns_mpo_v2 / 6.0, 0.05, 1.0))),
            (0.75, float(np.clip(mpo.qed, 0.05, 1.0))),
            (0.75, float(np.clip(mpo.sa_norm, 0.05, 1.0))),
        ]
        if ad_pred is not None and ad_pred.valid:
            # Prefer true MDR1-MDCK efflux over Broccatelli P-gp inhibition when both exist
            has_mdr1 = "mdr1_efflux" in ad_pred.probabilities
            ep_weights: dict[str, float] = {
                "bbb": 1.6,
                "herg": 2.0,
                "mdr1_efflux": 1.4,
                "pgp": 0.4 if has_mdr1 else 1.0,
                "hlm_clearance": 1.0,
                "cyp2d6": 1.0,
                "cyp3a4": 0.9,
                "htr2b_avoid": 1.8,
                "hrh1_avoid": 1.0,
                "ames": 1.0,
                "dili": 0.5,
                "chrm3_avoid": 0.6,
            }
            for ep_name, ep_prob in ad_pred.probabilities.items():
                w = ep_weights.get(ep_name, 0.5)
                desir = _damped_endpoint_desirability(
                    ep_name,
                    ep_prob,
                    ad_pred.ad_tanimoto.get(ep_name, 0.0),
                    ad_thr,
                    ad_damp=ad_damp,
                )
                safety_factors.append((w, desir))

        w_tot = sum(w for w, _ in safety_factors)
        log_geom = sum(w * math.log(max(val, 1e-6)) for w, val in safety_factors) / max(w_tot, 1e-6)
        g_safety = float(np.clip(math.exp(log_geom), 0.05, 1.0))

        # 3. Target & Selectivity gate G_target_sel(m, t)
        if p_on is not None:
            p_on_c = float(np.clip(p_on, 0.0, 1.0))
            on_term = float(np.clip(p_on_c, 0.02, 1.0))
            if p_off is not None:
                p_off_c = float(np.clip(p_off, 0.0, 1.0))
                margin_gate = float(0.20 + 0.80 * expit(4.0 * (p_on_c - p_off_c)))
                g_target = float(np.clip(on_term * margin_gate, 0.004, 1.0))
            else:
                g_target = on_term
        else:
            g_target = 1.0

        # 4. 3D Geometry / Physics gate G_3D(m, P_t)
        geom_terms: list[float] = []
        if pkd_v is not None:
            geom_terms.append(float(np.clip((pkd_v - 4.0) / 6.0, 0.05, 1.0)))
        if clip_v is not None:
            geom_terms.append(float(np.clip((clip_v + 1.0) / 2.0, 0.05, 1.0)))
        if vina_v is not None:
            geom_terms.append(float(np.clip((-vina_v - 4.0) / 8.0, 0.05, 1.0)))
        g_3d = float(np.clip(float(np.mean(geom_terms)), 0.05, 1.0)) if geom_terms else 1.0

        total = (
            g_hygiene
            * (g_safety ** float(gamma_safety))
            * (g_target ** float(gamma_target))
            * (g_3d ** float(gamma_3d))
        )
        out.append(
            MultiplicativeCNSReward(
                smiles=smi,
                canonical_smiles=mpo.canonical_smiles,
                valid=True,
                hygiene_gate=g_hygiene,
                safety_gate=g_safety,
                target_sel_gate=g_target,
                geometry_3d_gate=g_3d,
                total_reward=float(np.clip(total, 0.0, 1.0)),
                mpo=mpo,
                triage=triage,
                admet_v4=ad_pred,
                on_target_prob=p_on,
                off_target_max_prob=p_off,
                pred_pkd=pkd_v,
                clip_sim=clip_v,
                vina_score=vina_v,
            )
        )
    return out


__all__ = [
    "ADMET_DESCRIPTOR_NAMES",
    "ADMET_FP_BITS",
    "DEFAULT_ADMET_CV_REPORT_PATH",
    "DEFAULT_ADMET_DATA_DIR",
    "DEFAULT_ADMET_MODEL_PATH",
    "DEFAULT_CNS_REWARD_WEIGHTS",
    "DEFAULT_V4_ADMET_CV_REPORT_PATH",
    "DEFAULT_V4_ADMET_MODEL_PATH",
    "DEFAULT_V4_EXPANSION_DIR",
    "ENDPOINT_HIGHER_IS_BETTER",
    "ORACLE_SEMANTICS",
    "V3_ADMET_ENDPOINTS",
    "V4_ADMET_ENDPOINTS",
    "CNSAdmetPrediction",
    "CNSAdmetSuite",
    "CNSCompositeScore",
    "CalibratedADMETOracle",
    "MultiplicativeCNSReward",
    "PharmaAdmetPrediction",
    "compute_cns_composite_reward",
    "compute_ece",
    "compute_multiplicative_cns_reward",
    "evaluate_kpuu_external_benchmark",
    "featurize_admet_smiles",
    "load_avoidome_binary_dataset",
    "load_b3db_classification",
    "load_biogen_binary_endpoint",
    "load_tdc_binary_csv",
    "make_scaffold_kfold",
    "train_default_cns_admet_suite",
    "train_v4_pharma_admet_suite",
]
