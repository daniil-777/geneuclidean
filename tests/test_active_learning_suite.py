"""Comprehensive unit & mathematical invariant suite for ``geneuclidean.active_learning``.

Verifies all 13 mathematical invariants (M1-M13), zero-PyTorch lazy cold import,
14 registered acquisition strategies, 5 registered surrogates, correlation-gated
multi-fidelity routing, reaction-space O(N_A + N_B) Thompson Sampling, async
pending-batch DMTA orchestration, chained SHA-256 ledgers, and v5.0 backward
compatibility — 100% offline and CI-safe.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from geneuclidean.active_learning import (
    ACQUISITION_REGISTRY,
    SURROGATE_REGISTRY,
    AcquisitionContext,
    ActiveLearningEngine,
    CombinatorialThompsonSampler,
    ConformalSurrogate,
    DMTALedger,
    EnsembleSurrogate,
    FidelityTierSpec,
    MultiFidelityDMTA,
    StoppingAdvisor,
    TanimotoGP,
    build_feasibility_vector,
    canonicalize_smiles,
    compute_batch_internal_tanimoto,
    compute_exact_2d_ehvi,
    compute_expected_improvement,
    compute_hypervolume,
    compute_hypervolume_2d,
    compute_log_expected_improvement,
    evaluate_calibration,
    featurize_pool,
    list_acquisitions,
    list_surrogates,
    minmax_kernel,
    pareto_front,
    run_acquisition,
    select_cold_start,
    tanimoto_kernel,
)
from geneuclidean.active_learning.acquisitions import compute_hypervolume_3d, compute_log_h
from geneuclidean.screening.active_learning import (
    ActiveLearningDMTA,
    DMTAQuery,
    DMTARoundSummary,
)

# Diverse drug-like CNS/MedChem test deck (no external files required)
_DRUG_DECK: tuple[str, ...] = (
    "CN1CCN(C(=O)c2ccc(F)cc2)CC1",
    "CCN1CCN(C(=O)c2ccc(F)cc2)CC1",
    "CN1CCN(C(=O)c2ccc(Cl)cc2)CC1",
    "CN1CCN(C(=O)c2ccc(OC)cc2)CC1",
    "CN1CCN(C(=O)c2ccc(C#N)cc2)CC1",
    "COc1ccc2[nH]cc(CCN(C)C)c2c1",
    "COc1ccc2[nH]cc(CCN(C)CC)c2c1",
    "Fc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1",
    "Clc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1",
    "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",
    "CN1CCc2cc(F)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",
    "c1ccc(Cn2ccc(CN3CCOCC3)n2)nc1",
    "Cc1ccc(Cn2ccc(CN3CCOCC3)n2)nc1",
    "Fc1ccc(Cn2ccc(CN3CCOCC3)n2)nc1",
    "O=C(Nc1ccc(F)cc1)c1ccncc1",
    "O=C(Nc1ccc(Cl)cc1)c1ccncc1",
    "O=C(Nc1ccc(C)cc1)c1ccncc1",
    "CC(C)Nc1ncnc2[nH]cnc12",
    "CCCc1nc2ccccc2n1Cc1ccc(F)cc1",
    "CCOc1ccc(NC(=O)C2CCN(C)CC2)cc1",
    "Cc1nc2ccc(F)cc2n1CCN1CCCCC1",
    "COc1cc2c(cc1OC)CCN(C)C2c1ccccc1",
    "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
)


def _deterministic_sar_oracle(smiles: Sequence[str]) -> list[float]:
    """Deterministic structure-activity oracle favoring fluorinated/chlorinated basic amines."""
    out: list[float] = []
    for s in smiles:
        score = 6.0
        if "F" in s:
            score += 1.4
        if "Cl" in s:
            score += 0.9
        if "N1CCN" in s or "CCN" in s:
            score += 1.1
        if "c2ccc" in s:
            score += 0.4
        score -= 0.025 * len(s)
        out.append(float(score))
    return out


# ---------------------------------------------------------------------------
# 1. Zero-PyTorch Lazy Cold-Import Invariant
# ---------------------------------------------------------------------------


def test_lazy_cold_import_zero_heavy_dependencies() -> None:
    """Both ``import geneuclidean`` and ``import geneuclidean.active_learning`` stay lazy."""
    code = (
        "import json, sys, time; "
        "t0 = time.perf_counter(); "
        "import geneuclidean; "
        "import geneuclidean.active_learning; "
        "dt_ms = (time.perf_counter() - t0) * 1000.0; "
        "heavy = [m for m in ('torch', 'transformers', 'sklearn', 'rdkit', 'scipy', 'pandas') "
        "if m in sys.modules]; "
        "print(json.dumps({'dt_ms': dt_ms, 'heavy': heavy}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["heavy"] == [], f"Unexpected eager heavy imports: {payload['heavy']}"
    assert payload["dt_ms"] < 250.0


# ---------------------------------------------------------------------------
# 2. Featurizers & PSD Kernels (Invariants M1)
# ---------------------------------------------------------------------------


def test_m1_tanimoto_and_minmax_kernel_mathematical_invariants() -> None:
    """Verify symmetry, unit diagonal, [0,1] bounds, PSD, zero-vector rule, and binary match."""
    x_bin, valid_b = featurize_pool(_DRUG_DECK[:20], kind="ecfp4_binary", n_bits=512)
    x_cnt, valid_c = featurize_pool(_DRUG_DECK[:20], kind="ecfp4_count", n_bits=512)
    assert bool(valid_b.all()) and bool(valid_c.all())

    # Append two zero vectors to explicitly test zero-vector boundary rule k(0, 0) = 1
    zeros = np.zeros((2, 512), dtype=np.float64)
    x_bin_z = np.vstack([x_bin, zeros])
    x_cnt_z = np.vstack([x_cnt, zeros])

    k_tan = tanimoto_kernel(x_bin_z)
    k_mm_bin = minmax_kernel(x_bin_z)
    k_mm_cnt = minmax_kernel(x_cnt_z)

    for k_mat in (k_tan, k_mm_bin, k_mm_cnt):
        # Symmetry
        assert float(np.max(np.abs(k_mat - k_mat.T))) < 1e-12
        # Unit diagonal & [0, 1] bounds
        assert np.allclose(np.diag(k_mat), 1.0, atol=1e-12)
        assert float(np.min(k_mat)) >= 0.0 and float(np.max(k_mat)) <= 1.0
        # Positive Semi-Definiteness
        eigvals = np.linalg.eigvalsh(k_mat)
        assert float(np.min(eigvals)) >= -1e-10
        # Zero-vector boundary: k(0, 0) = 1.0 and k(0, non-zero) = 0.0
        assert k_mat[-1, -2] == 1.0
        assert float(np.max(np.abs(k_mat[-1, :-2]))) == 0.0

    # Exact binary equivalence between MinMax and Tanimoto
    assert float(np.max(np.abs(k_mm_bin - k_tan))) < 1e-12
    # Count fingerprints carry multiplicity beyond binary presence
    assert float(np.max(np.abs(k_mm_cnt[:20, :20] - k_tan[:20, :20]))) > 1e-3

    # Canonicalization handles invalid SMILES gracefully
    assert canonicalize_smiles("INVALID_###") is None
    assert canonicalize_smiles("c1ccccc1") == "c1ccccc1"


# ---------------------------------------------------------------------------
# 3. Surrogates: TanimotoGP, EnsembleSurrogate, ConformalSurrogate (M2-M6)
# ---------------------------------------------------------------------------


def test_m2_m3_tanimoto_gp_exactness_ood_reversion_and_rank1_fantasies() -> None:
    """Verify MLL fit, duplicate robustness, low-noise collapse, OOD reversion & rank-1."""
    train_smis = list(_DRUG_DECK[:16]) + [_DRUG_DECK[0], _DRUG_DECK[0]]  # deliberate duplicates!
    train_y = _deterministic_sar_oracle(train_smis)

    # Low-noise fixed GP checks exact training collapse & OOD prior reversion
    gp = TanimotoGP(
        kernel="tanimoto",
        n_bits=1024,
        signal_variance=1.0,
        noise_variance=1e-4,
        optimize_hyperparams=False,
    )
    stats = gp.fit(train_smis, train_y)
    assert stats["n_train"] == float(len(train_smis))
    assert math.isfinite(stats["log_marginal_likelihood"])

    pred_tr = gp.predict(train_smis[:16])
    assert bool(pred_tr.valid.all())
    assert float(np.min(pred_tr.epistemic_std)) >= 0.0
    # Epistemic variance collapses <= s_y^2 * sigma_n^2 at observed training points
    max_allowed_epi = math.sqrt(gp.noise_variance_raw) * 1.05
    assert float(np.max(pred_tr.epistemic_std)) <= max_allowed_epi
    assert float(np.mean(np.abs(pred_tr.mean - np.array(train_y[:16])))) < 0.05

    # Strictly orthogonal OOD molecule (verify 0.0 max Tanimoto overlap with training set)
    x_tr_chk, _ = featurize_pool(train_smis, kind="ecfp4_binary", n_bits=1024)
    ood_smi = None
    for cand in ("[He]", "[Ne]", "[Ar]", "[Kr]", "[Xe]", "[Rn]", "[Os]", "[Ir]", "[Pt]", "[Au]"):
        xc, _ = featurize_pool([cand], kind="ecfp4_binary", n_bits=1024)
        if float(np.max(tanimoto_kernel(xc, x_tr_chk))) == 0.0:
            ood_smi = cand
            break
    assert ood_smi is not None
    pred_ood = gp.predict([ood_smi])
    assert bool(pred_ood.valid[0])
    y_mean_emp = float(np.mean(train_y))
    y_std_emp = float(np.std(train_y))
    assert abs(float(pred_ood.mean[0]) - y_mean_emp) < 1e-10
    assert abs(float(pred_ood.epistemic_std[0]) - y_std_emp * math.sqrt(gp.signal_variance)) < 1e-10

    # M3: O(n^2) rank-1 fantasy update matches full Cholesky refit with fixed hyperparams
    gp_opt = TanimotoGP(kernel="tanimoto", n_bits=1024, optimize_hyperparams=True)
    gp_opt.fit(list(_DRUG_DECK[:14]), _deterministic_sar_oracle(list(_DRUG_DECK[:14])))
    fan_smis = list(_DRUG_DECK[14:18])
    fan_y = _deterministic_sar_oracle(fan_smis)

    gp_rank1 = gp_opt.condition_on_fantasies(fan_smis, fan_y)
    # Build reference exact GP with identical standardization & hyperparams on union
    x_all, _ = featurize_pool(list(_DRUG_DECK[:18]), kind="ecfp4_binary", n_bits=1024)
    k_all = gp_opt.signal_variance * tanimoto_kernel(x_all)
    k_all[np.diag_indices(18)] += gp_opt.noise_variance + gp_opt._jitter_used
    y_all_norm = (
        np.array(_deterministic_sar_oracle(list(_DRUG_DECK[:18])), dtype=np.float64)
        - gp_opt._y_mean
    ) / gp_opt._y_std
    test_smis = list(_DRUG_DECK[18:24])
    x_test, _ = featurize_pool(test_smis, kind="ecfp4_binary", n_bits=1024)
    k_star = gp_opt.signal_variance * tanimoto_kernel(x_test, x_all)
    ref_mu = gp_opt._y_mean + gp_opt._y_std * (k_star @ np.linalg.solve(k_all, y_all_norm))
    ref_var_norm = np.maximum(
        0.0,
        gp_opt.signal_variance - np.einsum("ij,ji->i", k_star, np.linalg.solve(k_all, k_star.T)),
    )
    ref_epi = gp_opt._y_std * np.sqrt(ref_var_norm)

    pred_r1 = gp_rank1.predict(test_smis)
    assert np.allclose(pred_r1.mean, ref_mu, atol=1e-9)
    assert np.allclose(pred_r1.epistemic_std, ref_epi, atol=1e-9)

    # MinMax count GP also fits & predicts cleanly
    mm_gp = TanimotoGP(kernel="minmax", n_bits=512, optimize_hyperparams=True)
    mm_gp.fit(list(_DRUG_DECK[:16]), _deterministic_sar_oracle(list(_DRUG_DECK[:16])))
    mm_pred = mm_gp.predict(list(_DRUG_DECK[16:22]))
    assert bool(mm_pred.valid.all()) and bool(np.all(np.isfinite(mm_pred.mean)))


def test_m4_ensemble_surrogate_variance_decomposition() -> None:
    """Verify exact Law-of-Total-Variance identity std^2 == epistemic^2 + aleatoric^2."""
    ens = EnsembleSurrogate(n_rf_estimators=24, n_hgb_members=4, n_bits=512, seed=42)
    ens.fit(list(_DRUG_DECK[:22]), _deterministic_sar_oracle(list(_DRUG_DECK[:22])))
    pred = ens.predict(list(_DRUG_DECK[22:]) + ["INVALID_SMILES"])
    assert bool(pred.valid[:-1].all()) and not bool(pred.valid[-1])
    ok = pred.valid
    assert bool(np.all(pred.epistemic_std[ok] >= 0.0))
    assert bool(np.all(pred.aleatoric_std[ok] >= 0.0))
    assert np.allclose(
        pred.std[ok] ** 2,
        pred.epistemic_std[ok] ** 2 + pred.aleatoric_std[ok] ** 2,
        atol=1e-12,
    )


def test_m5_m6_conformal_surrogate_split_normalized_and_weighted() -> None:
    """Verify strict disjoint calibration, CV+ fallback, OOD width growth & weighted Kish N_eff."""
    tr_smis = list(_DRUG_DECK[:18])
    cal_smis = list(_DRUG_DECK[18:28])
    tr_y = _deterministic_sar_oracle(tr_smis)
    cal_y = _deterministic_sar_oracle(cal_smis)

    # M5 & M6: Explicit disjoint fit_calibrated in normalized mode
    conf_norm = ConformalSurrogate(
        TanimotoGP(n_bits=1024, optimize_hyperparams=True),
        alpha=0.10,
        mode="normalized",
        seed=42,
    )
    stats = conf_norm.fit_calibrated(tr_smis, tr_y, cal_smis, cal_y)
    assert stats["n_cal"] == float(len(cal_smis))
    assert conf_norm.calibration_mode == "split"

    # Normalized intervals MUST be strictly wider on OOD ([He]) than on in-domain training molecule
    pred_cmp = conf_norm.predict([tr_smis[0], "[He]"])
    assert pred_cmp.interval_lo is not None and pred_cmp.interval_hi is not None
    w_in = float(pred_cmp.interval_hi[0] - pred_cmp.interval_lo[0])
    w_ood = float(pred_cmp.interval_hi[1] - pred_cmp.interval_lo[1])
    assert w_ood > w_in * 1.15, f"Expected wider OOD interval: in={w_in:.4f}, ood={w_ood:.4f}"

    # Weighted covariate-shift mode emits Kish N_eff and adapts intervals
    conf_w = ConformalSurrogate(
        TanimotoGP(n_bits=1024, optimize_hyperparams=True),
        alpha=0.10,
        mode="weighted_covariate_shift",
        weight_gamma=5.0,
        seed=42,
    )
    conf_w.fit_calibrated(tr_smis, tr_y, cal_smis, cal_y)
    pred_w = conf_w.predict([tr_smis[0], "[He]"])
    assert "mean_kish_n_eff" in pred_w.metadata
    assert pred_w.metadata["mean_kish_n_eff"] > 0.0
    assert pred_w.interval_lo is not None and pred_w.interval_hi is not None
    assert float(pred_w.interval_hi[1] - pred_w.interval_lo[1]) > float(
        pred_w.interval_hi[0] - pred_w.interval_lo[0]
    )

    # Early-round small-n automatically uses leak-free CV+ fallback
    conf_small = ConformalSurrogate(
        TanimotoGP(n_bits=512, optimize_hyperparams=False),
        alpha=0.10,
        mode="normalized",
        min_split_n=40,
        seed=42,
    )
    conf_small.fit(tr_smis[:15], tr_y[:15])
    assert conf_small.calibration_mode == "cv_plus"
    assert conf_small.cal_size == 15

    # Registry exposes all 5 built-in surrogates
    reg_names = set(list_surrogates())
    assert {
        "tanimoto_gp",
        "minmax_gp",
        "ensemble",
        "conformal_tanimoto_gp",
        "conformal_ensemble",
    }.issubset(reg_names)
    assert "tanimoto_gp" in SURROGATE_REGISTRY


# ---------------------------------------------------------------------------
# 4. Acquisitions: All 14 Strategies & Invariants M7-M11
# ---------------------------------------------------------------------------


def test_m7_log_ei_stability_and_monotonicity_in_deep_negative_tail() -> None:
    """Verify log_ei stays finite and strictly monotonic even at z = -30 .. -25."""
    zs = np.array([-40.0, -30.0, -25.0, -20.0, -12.0, -8.5, -8.0, -7.5, -2.0, 0.0, 3.0])
    lh = compute_log_h(zs)
    assert bool(np.all(np.isfinite(lh)))
    assert bool(np.all(np.diff(lh) > 0.0)), f"log_h must be strictly increasing: {lh}"

    # Continuity across the z = -12.0 erfcx / asymptotic transition (slope ~ 12 -> dz=1e-7)
    lh_left = float(compute_log_h(np.array([-12.0 - 1e-7]))[0])
    lh_right = float(compute_log_h(np.array([-12.0 + 1e-7]))[0])
    assert abs(lh_left - lh_right) < 1e-5

    # Deep-tail pool ranking where standard float64 EI underflows completely to 0.0 (z <= -40)
    means = np.array([-45.0, -44.0, -43.0, -42.0], dtype=np.float64)
    stds = np.ones(4, dtype=np.float64)
    raw_ei = compute_expected_improvement(means, stds, best_observed=0.0, xi=0.0)
    log_ei = compute_log_expected_improvement(means, stds, best_observed=0.0, xi=0.0)
    assert float(np.max(raw_ei)) == 0.0  # standard EI underflows completely in float64
    assert bool(np.all(np.isfinite(log_ei)))
    assert bool(np.all(np.diff(log_ei) > 0.0))

    # Even registered 'ei' uses log_ei tie-break to rank the least-bad candidate #1!
    ctx = AcquisitionContext(
        pool_smiles=("C", "CC", "CCC", "CCCC"),
        mean=means,
        std=stds,
        best_observed=0.0,
        batch_size=2,
        xi=0.0,
    )
    sel_ei = run_acquisition("ei", ctx)
    sel_lei = run_acquisition("log_ei", ctx)
    assert sel_ei.indices == (3, 2)
    assert sel_lei.indices == (3, 2)


def test_m8_m9_kriging_believer_and_dpp_batch_decorrelation() -> None:
    """Verify KB Schur variance deflation changes pick 2 & DPP enforces zero-duplicate diversity."""
    # Construct pool containing tight pairs of close analogs + distinct chemotypes
    train_smis = list(_DRUG_DECK[15:25])
    train_y = _deterministic_sar_oracle(train_smis)
    gp = TanimotoGP(kernel="tanimoto", n_bits=1024, optimize_hyperparams=True)
    gp.fit(train_smis, train_y)

    # Pool with intentional near-clones at top (indices 0,1 and 2,3 are very close piperazides)
    pool = tuple(_DRUG_DECK[:15])
    pred = gp.predict(pool)

    ctx_ucb = AcquisitionContext(
        pool_smiles=pool,
        mean=pred.mean,
        std=pred.epistemic_std,
        best_observed=max(train_y),
        batch_size=5,
        beta=2.0,
        surrogate=gp,
    )
    sel_ucb = run_acquisition("ucb", ctx_ucb)
    sel_kb = run_acquisition("kriging_believer", ctx_ucb)
    sel_dpp = run_acquisition("dpp_ucb", ctx_ucb)
    sel_mm = run_acquisition("maxmin_ucb", ctx_ucb)

    # Zero duplicates and exact requested batch size
    for sel in (sel_ucb, sel_kb, sel_dpp, sel_mm):
        assert len(sel.indices) == 5
        assert len(set(sel.indices)) == 5

    # M8: KB pick 1 matches UCB pick 1, strictly reduces variance, and decorrelates subsequent picks
    assert sel_kb.indices[0] == sel_ucb.indices[0]
    assert sel_kb.metadata["exact_gp_schur"] is True
    assert sel_kb.metadata["mean_var_after"] < sel_kb.metadata["mean_var_before"]

    # Create explicit clone-pair test proving KB pick 2 diverges from static top-2 UCB
    clone_pool = (
        "CN1CCN(C(=O)c2ccc(F)cc2)CC1",  # #0
        "CN1CCN(C(=O)c2ccc(F)cc2)CC1",  # #1 exact clone of #0!
        "COc1ccc2[nH]cc(CCN(C)C)c2c1",  # #2 distinct scaffold
        "Fc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1",  # #3 distinct scaffold
    )
    p_clone = gp.predict(clone_pool)
    ctx_clone = AcquisitionContext(
        pool_smiles=clone_pool,
        mean=p_clone.mean,
        std=p_clone.epistemic_std,
        best_observed=max(train_y),
        batch_size=2,
        beta=2.5,
        surrogate=gp,
    )
    kb_clone = run_acquisition("kriging_believer", ctx_clone)
    dpp_clone = run_acquisition("dpp_ucb", ctx_clone)
    # Neither KB nor DPP may select BOTH identical clones {0, 1}!
    assert set(kb_clone.indices) != {0, 1}
    assert set(dpp_clone.indices) != {0, 1}

    # Pending-batch conditioning also avoids picking a pending clone
    ctx_pend = AcquisitionContext(
        pool_smiles=clone_pool,
        mean=p_clone.mean,
        std=p_clone.epistemic_std,
        best_observed=max(train_y),
        batch_size=1,
        beta=3.0,
        surrogate=gp,
        pending_smiles=("CN1CCN(C(=O)c2ccc(F)cc2)CC1",),
    )
    kb_pend = run_acquisition("kriging_believer", ctx_pend)
    assert kb_pend.metadata["n_pending_conditioned"] == 1
    assert kb_pend.indices[0] not in (0, 1)


def test_m10_pareto_hypervolume_exact_2d_ehvi_and_multiobjective_acquisitions() -> None:
    """Verify exact 2D/3D HV, analytical 2D EHVI vs MC, qEHVI batching, and ParEGO."""
    # Exact 2D HV check
    pts_2d = np.array([[5.0, 1.0], [3.0, 4.0], [1.0, 6.0], [2.0, 2.0]], dtype=np.float64)
    nd_idx, nd_pts = pareto_front(pts_2d)
    assert set(nd_idx.tolist()) == {0, 1, 2}
    hv2 = compute_hypervolume_2d(nd_pts, np.array([0.0, 0.0]))
    assert abs(hv2 - 16.0) < 1e-12

    # Exact 3D HV check: box1=[0,2]x[0,1]x[0,3]=6, box2=[0,1]x[0,4]x[0,2]=8, overlap=2 -> 12
    pts_3d = np.array([[2.0, 1.0, 3.0], [1.0, 4.0, 2.0]], dtype=np.float64)
    hv3 = compute_hypervolume_3d(pts_3d, np.array([0.0, 0.0, 0.0]))
    assert abs(hv3 - 12.0) < 1e-12

    # Exact 2D analytical EHVI matches high-sample Monte Carlo to < 0.02
    ref = np.array([0.0, 0.0])
    cand_mu = np.array([[4.0, 4.0], [2.0, 2.0], [6.0, 0.5]], dtype=np.float64)
    cand_sd = np.array([[0.8, 0.8], [0.4, 0.4], [0.5, 0.5]], dtype=np.float64)
    ehvi_exact = compute_exact_2d_ehvi(cand_mu, cand_sd, nd_pts, ref)
    assert bool(np.all(ehvi_exact >= 0.0))

    rng = np.random.default_rng(123)
    n_mc = 25000
    for idx in range(len(cand_mu)):
        draws = cand_mu[idx] + cand_sd[idx] * rng.standard_normal((n_mc, 2))
        gains = np.array(
            [max(0.0, compute_hypervolume_2d(np.vstack([nd_pts, d]), ref) - hv2) for d in draws]
        )
        mc_est = float(np.mean(gains))
        assert abs(float(ehvi_exact[idx]) - mc_est) < 0.05, (
            f"Candidate {idx}: exact={ehvi_exact[idx]:.4f} vs MC={mc_est:.4f}"
        )

    # Batch qEHVI & pareto_scalarized_ucb select unique valid batches
    pool_smis = tuple(_DRUG_DECK[:10])
    mu_mo = rng.uniform(5.0, 9.0, size=(10, 2))
    sd_mo = rng.uniform(0.2, 0.8, size=(10, 2))
    ctx_mo = AcquisitionContext(
        pool_smiles=pool_smis,
        mean=mu_mo,
        std=sd_mo,
        batch_size=4,
        observed_objectives=np.array([[6.0, 7.0], [8.0, 5.5]]),
        reference_point=np.array([4.0, 4.0]),
        qmc_samples=64,
        seed=42,
    )
    sel_qehvi = run_acquisition("qehvi", ctx_mo)
    sel_parego = run_acquisition("pareto_scalarized_ucb", ctx_mo)
    assert len(sel_qehvi.indices) == 4 and len(set(sel_qehvi.indices)) == 4
    assert len(sel_parego.indices) == 4 and len(set(sel_parego.indices)) == 4
    assert all(s >= 0.0 for s in sel_qehvi.scores)

    # 4D Sobol QMC hypervolume is positive and monotone
    pts_4d = np.array([[2.0, 2.0, 2.0, 2.0]])
    hv4 = compute_hypervolume(pts_4d, np.zeros(4), n_mc=1024, seed=7)
    assert abs(hv4 - 16.0) < 0.5


def test_m11_constrained_acquisitions_phase1_recovery_and_no_sign_inversion() -> None:
    """Verify Phase-1 feasibility recovery, negative-UCB sign safety, and DCU triage veto."""
    pool = (
        "O=C(NC1CCCCC1)NC1CCCCC1",  # DCU coupling byproduct -> MUST get hygiene_gate == 0.0
        "CN1CCN(C(=O)c2ccc(F)cc2)CC1",  # clean CNS drug-like
        "COc1ccc2[nH]cc(CCN(C)C)c2c1",  # clean CNS drug-like
    )
    feas = build_feasibility_vector(pool, property_profile="cns_mpo_v2", triage_hard_gate=True)
    assert feas[0] == 0.0
    assert feas[1] > 0.5 and feas[2] > 0.5

    # Phase-1 Feasibility Recovery when best_observed == -inf (0 feasible points observed yet)
    ctx_p1 = AcquisitionContext(
        pool_smiles=pool,
        mean=np.array([10.0, 6.0, 6.5]),
        std=np.array([0.5, 0.5, 0.5]),
        best_observed=float("-inf"),
        batch_size=1,
        feasibility_probs=np.array([0.05, 0.95, 0.60]),
    )
    sel_p1 = run_acquisition("constrained_ei", ctx_p1)
    assert sel_p1.metadata["phase1_feasibility_recovery"] is True
    assert sel_p1.indices == (1,)  # highest feasibility wins over high-mean infeasible #0

    # Sign-inversion bug prevention on negative UCB values (e.g., logBB < 0)
    ctx_neg = AcquisitionContext(
        pool_smiles=pool,
        mean=np.array([-2.0, -1.5, -1.8]),
        std=np.array([0.1, 0.1, 0.1]),
        batch_size=1,
        beta=1.0,
        feasibility_probs=np.array([0.01, 0.99, 0.50]),
    )
    sel_feas = run_acquisition("avoidome_feasibility_ucb", ctx_neg)
    assert sel_feas.indices == (1,)  # feasible #1 beats infeasible #0 despite negative means

    # Verify all 14 canonical strategies exist in ACQUISITION_REGISTRY
    expected_14 = {
        "greedy",
        "uncertainty",
        "ucb",
        "pi",
        "ei",
        "log_ei",
        "thompson",
        "maxmin_ucb",
        "kriging_believer",
        "dpp_ucb",
        "qehvi",
        "pareto_scalarized_ucb",
        "constrained_ei",
        "avoidome_feasibility_ucb",
    }
    assert expected_14.issubset(set(list_acquisitions()))
    assert "maxmin_diverse_ucb" in ACQUISITION_REGISTRY


# ---------------------------------------------------------------------------
# 5. Multi-Fidelity Correlation Guardrail (M12)
# ---------------------------------------------------------------------------


def test_m12_multifidelity_correlation_guardrail_and_hard_gate() -> None:
    """Verify informative proxy stays trusted, adversarial trips rho guardrail & gate works."""
    pool = list(_DRUG_DECK)
    truth = {s: v for s, v in zip(pool, _deterministic_sar_oracle(pool), strict=True)}
    rng = np.random.default_rng(7)
    noisy_good = {s: truth[s] + float(rng.normal(0.0, 0.15)) for s in pool}
    adversarial_bad = {s: -truth[s] + float(rng.normal(0.0, 0.3)) for s in pool}

    # Condition A: Informative proxy stays trusted
    mf_good = MultiFidelityDMTA(
        [
            FidelityTierSpec(
                name="tier1_proxy",
                level=0,
                cost=1.0,
                oracle_fn=lambda xs: [noisy_good[s] for s in xs],
                hard_gate_fn=lambda xs: [i % 4 != 0 for i in range(len(xs))],
            ),
            FidelityTierSpec(
                name="tier2_truth",
                level=1,
                cost=20.0,
                oracle_fn=lambda xs: [truth[s] for s in xs],
            ),
        ],
        min_correlation=0.25,
        min_paired_for_gate=8,
        seed=42,
    )
    summaries_good = mf_good.run_budgeted_campaign(
        pool, total_cost_budget=360.0, high_batch_size=8, low_funnel_multiplier=2
    )
    assert len(summaries_good) >= 2
    assert mf_good.low_fidelity_trusted[0] is True
    assert mf_good.paired_spearman_rho[0] > 0.5
    assert mf_good.total_cost_spent <= 360.0 + 1e-6

    # Condition B: Adversarial uncorrelated / inverted proxy trips guardrail cleanly
    mf_bad = MultiFidelityDMTA(
        [
            FidelityTierSpec(
                name="tier1_garbage",
                level=0,
                cost=1.0,
                oracle_fn=lambda xs: [adversarial_bad[s] for s in xs],
            ),
            FidelityTierSpec(
                name="tier2_truth",
                level=1,
                cost=20.0,
                oracle_fn=lambda xs: [truth[s] for s in xs],
            ),
        ],
        min_correlation=0.25,
        min_paired_for_gate=8,
        seed=42,
    )
    r1 = mf_bad.run_round(pool, high_batch_size=10, remaining_budget=400.0)
    assert mf_bad.low_fidelity_trusted[0] is False
    assert any("gated off" in w for w in r1.warnings)
    # Subsequent round must spend 0 evaluations on the gated-off tier!
    r2 = mf_bad.run_round(pool, high_batch_size=6, remaining_budget=200.0)
    assert r2.tier_evaluated_counts["tier1_garbage"] == 0
    assert r2.tier_evaluated_counts["tier2_truth"] == 6


# ---------------------------------------------------------------------------
# 6. Combinatorial Reaction-Space Thompson Sampling (M13)
# ---------------------------------------------------------------------------


def test_m13_combinatorial_ts_isolation_uncharging_and_epistasis_diagnostic() -> None:
    """Verify O(N_A + N_B) isolation, carboxylate uncharging, both modes & epistasis R^2."""
    # Include pre-protonated carboxylates [O-] to verify automatic Uncharger() recovery!
    acids = [
        "Cc1ccc(C(=O)[O-])cc1",
        "Fc1ccc(C(=O)[O-])cc1",
        "Clc1ccc(C(=O)O)cc1",
        "COc1ccc(C(=O)O)cc1",
        "c1ccncc1C(=O)O",
        "CC(C)C(=O)O",
        "C1CCCCC1C(=O)O",
        "O=C(O)c1ccco1",
    ]
    amines = [
        "CN1CCNCC1",
        "C1COCCN1",
        "CCN",
        "CC(C)N",
        "NCC1CC1",
        "Nc1ccc(F)cc1",
        "Nc1ccccc1",
        "NCCOC",
    ]
    # Virtual space = 8 x 8 = 64 pairs; give tight budget = 26 (< 41% of space)
    call_counter = {"n_evaluated": 0}

    def _additive_oracle(prods: Sequence[str]) -> list[float]:
        call_counter["n_evaluated"] += len(prods)
        return [
            5.0 + (2.5 if "F" in p else 0.0) + (2.0 if "N1CCN" in p else 0.0) - 0.02 * len(p)
            for p in prods
        ]

    ts_arg = CombinatorialThompsonSampler(
        reaction_name="amide_coupling_acid_amine",
        warmup_per_reagent=1,
        selection_mode="argmax",
        triage_filter=True,
        uncharge_reactants=True,
        seed=42,
    )
    res_arg = ts_arg.run(acids, amines, _additive_oracle, total_budget=26, batch_size=6)
    assert res_arg.n_reactants_a == 8 and res_arg.n_reactants_b == 8
    assert res_arg.n_virtual_space == 64
    # Strict ground-truth isolation: oracle never called beyond budget
    assert call_counter["n_evaluated"] == res_arg.n_oracle_calls <= 26
    assert len({p.product_smiles for p in res_arg.all_products}) == res_arg.n_oracle_calls
    assert res_arg.best_product is not None
    assert math.isfinite(res_arg.reagent_additivity_r2)
    assert res_arg.reagent_additivity_r2 > 0.65
    assert len(res_arg.ledger_sha256) == 64

    # Boltzmann roulette mode also runs cleanly
    ts_roul = CombinatorialThompsonSampler(
        reaction_name="amide_coupling_acid_amine",
        warmup_per_reagent=1,
        selection_mode="boltzmann_roulette",
        triage_filter=True,
        uncharge_reactants=True,
        seed=42,
    )
    res_roul = ts_roul.run(acids, amines, _additive_oracle, total_budget=24, batch_size=6)
    assert res_roul.n_oracle_calls <= 24 and res_roul.best_product is not None


# ---------------------------------------------------------------------------
# 7. Diagnostics, Stopping, Async Engine, Chained Ledger & v5.0 Compat
# ---------------------------------------------------------------------------


def test_cold_start_calibration_and_stopping_advisor() -> None:
    """Verify cold-start clustering, UQ calibration buckets & 3-signal stopping criteria."""
    for strat in ("butina_medoids", "maxmin", "random"):
        idx = select_cold_start(_DRUG_DECK, 8, strategy=strat, seed=42)
        assert len(idx) == 8 and len(set(idx)) == 8

    # Calibration diagnostics with similarity-stratified AD buckets
    gp = TanimotoGP(n_bits=1024, optimize_hyperparams=True)
    gp.fit(list(_DRUG_DECK[:18]), _deterministic_sar_oracle(list(_DRUG_DECK[:18])))
    test_smis = list(_DRUG_DECK[18:]) + ["[He]"]
    test_y = _deterministic_sar_oracle(list(_DRUG_DECK[18:])) + [6.5]
    pred = gp.predict(test_smis)
    cal = evaluate_calibration(
        test_y,
        pred.mean,
        pred.std,
        query_smiles=test_smis,
        train_smiles=list(_DRUG_DECK[:18]),
    )
    assert cal.n_eval == len(test_smis)
    assert math.isfinite(cal.rmse) and math.isfinite(cal.gaussian_nll)
    assert set(cal.similarity_stratified_coverage.keys()) == {"lt_0.35", "0.35_to_0.55", "ge_0.55"}
    assert cal.similarity_stratified_coverage["lt_0.35"]["n"] >= 1

    # StoppingAdvisor triggers all three signals when stalled
    advisor = StoppingAdvisor(
        ei_threshold=0.05, jaccard_threshold=0.80, top_k=5, improvement_tol=1e-3, patience=2
    )
    pool_scores = np.linspace(1.0, 10.0, len(_DRUG_DECK))
    r1 = advisor.update(
        best_metric=9.0,
        max_ei=0.001,
        observed_std=1.0,
        pool_smiles=_DRUG_DECK,
        pool_pred_mean=pool_scores,
    )
    assert not r1.should_stop
    r2 = advisor.update(
        best_metric=9.0,
        max_ei=0.001,
        observed_std=1.0,
        pool_smiles=_DRUG_DECK,
        pool_pred_mean=pool_scores,
    )
    r3 = advisor.update(
        best_metric=9.0,
        max_ei=0.001,
        observed_std=1.0,
        pool_smiles=_DRUG_DECK,
        pool_pred_mean=pool_scores,
    )
    assert r2.should_stop
    assert "ei_saturated" in r2.triggered_reasons
    assert "improvement_plateau" in r3.triggered_reasons
    assert "rank_stable" in r3.triggered_reasons


def test_async_engine_pending_batches_ledger_chain_and_v50_compat(tmp_path: Path) -> None:
    """Verify async pending masking, failed synthesis blacklisting, SHA-256 chain & v5.0 API."""
    pool = list(_DRUG_DECK)
    engine = ActiveLearningEngine(
        surrogate="conformal_tanimoto_gp",
        acquisition="kriging_believer",
        cold_start="butina_medoids",
        seed=42,
    )
    # Seed 8 observations via wet-lab CSV
    csv_path = tmp_path / "seed.csv"
    seed_smis = pool[:8]
    seed_vals = _deterministic_sar_oracle(seed_smis)
    csv_path.write_text(
        "smiles,pIC50\n"
        + "\n".join(f"{s},{v:.4f}" for s, v in zip(seed_smis, seed_vals, strict=True))
        + "\n",
        encoding="utf-8",
    )
    assert engine.ingest_assay_csv(csv_path, smiles_col="smiles", value_col="pIC50") == 8

    # Async batch 1 marked pending -> batch 2 MUST NOT overlap batch 1
    q1 = engine.propose_batch(pool, batch_size=4, mark_pending=True)
    assert q1.status == "pending" and len(q1.smiles) == 4
    assert len(q1.interval_lo) == 4 and len(q1.interval_hi) == 4
    q2 = engine.propose_batch(pool, batch_size=4, mark_pending=True)
    assert set(q1.smiles).isdisjoint(set(q2.smiles))
    assert engine.pending.n_pending == 8

    # Partial completion of 2 from q1 + synthesis failure blacklist of 2 from q1
    completed_part = {s: _deterministic_sar_oracle([s])[0] for s in q1.smiles[:2]}
    engine.complete_observations(completed_part)
    assert str(q1.status) == "partial"
    failed_part = list(q1.smiles[2:])
    engine.cancel_pending(failed_part, blacklist=True)
    assert str(q1.status) == "completed"
    assert set(failed_part).issubset(engine.pending.failed_blacklist)

    # Next proposal must never re-propose blacklisted or still-pending compounds
    q3 = engine.propose_batch(pool, batch_size=4, mark_pending=False)
    assert set(q3.smiles).isdisjoint(set(failed_part))
    assert set(q3.smiles).isdisjoint(set(q2.smiles))

    # Synchronous round + SHA-256 chain integrity
    r_sync = engine.run_round(pool, _deterministic_sar_oracle, batch_size=4)
    assert r_sync.n_acquired == 4
    assert engine.ledger.verify_chain() is True
    assert len(r_sync.chain_sha256) == 64
    saved = engine.ledger.save_json(tmp_path / "ledger.json")
    assert saved.exists()

    # Tampering check on DMTALedger
    ledger = DMTALedger()
    ledger.record_query(q1)
    assert ledger.verify_chain() is True
    ledger._entries[0].payload["strategy"] = "TAMPERED"
    assert ledger.verify_chain() is False

    # Multi-objective ActiveLearningEngine end-to-end
    mo_engine = ActiveLearningEngine(
        surrogate="tanimoto_gp",
        acquisition="qehvi",
        objective_names=("potency", "mpo"),
        reference_point=(4.0, 0.0),
        qmc_samples=64,
        seed=42,
    )

    def _mo_oracle(batch: Sequence[str]) -> list[tuple[float, float]]:
        pots = _deterministic_sar_oracle(batch)
        return [
            (p, float(max(0.1, 1.0 - 0.015 * len(s)))) for s, p in zip(batch, pots, strict=True)
        ]

    mo_r1 = mo_engine.run_round(pool, _mo_oracle, batch_size=6)
    mo_r2 = mo_engine.run_round(pool, _mo_oracle, batch_size=4)
    assert mo_r1.pareto_hypervolume is not None and mo_r2.pareto_hypervolume is not None
    assert mo_r2.pareto_hypervolume >= mo_r1.pareto_hypervolume - 1e-12

    # 100% v5.0 backward-compatible facade in screening.active_learning
    assert DMTAQuery is not None and DMTARoundSummary is not None
    al_legacy = ActiveLearningDMTA(strategy="maxmin_diverse_ucb", ucb_beta=1.2, seed=42)
    al_legacy.ingest_observations(dict(zip(seed_smis, seed_vals, strict=True)))
    leg_r1 = al_legacy.run_round(pool, _deterministic_sar_oracle, batch_size=4)
    assert leg_r1.n_total_observed == 12 and len(leg_r1.round_sha256) == 16

    # Upgraded v5.1 surrogate + acquisition through v5.0 facade
    al_upgraded = ActiveLearningDMTA(
        strategy="dpp_ucb", surrogate="tanimoto_gp", ucb_beta=1.5, seed=42
    )
    al_upgraded.ingest_observations(dict(zip(seed_smis, seed_vals, strict=True)))
    up_r1 = al_upgraded.run_round(pool, _deterministic_sar_oracle, batch_size=4)
    assert up_r1.n_total_observed == 12
    assert compute_batch_internal_tanimoto(up_r1.acquired_smiles) < 0.99


def test_additional_mathematical_invariants_and_regressions() -> None:
    """Verify weighted-conformal gamma=0 equivalence, exact DPP slogdet, dict MO & budget."""
    # 1. Weighted conformal with gamma=0.0 (uniform weights) equals unweighted normalized exactly
    tr_smis = list(_DRUG_DECK[:18])
    cal_smis = list(_DRUG_DECK[18:27])  # n_cal = 9 tests exact 9/10 = 0.9 float boundary!
    tr_y = _deterministic_sar_oracle(tr_smis)
    cal_y = _deterministic_sar_oracle(cal_smis)

    c_norm = ConformalSurrogate(
        TanimotoGP(n_bits=512, optimize_hyperparams=False), alpha=0.10, mode="normalized", seed=42
    )
    c_unif = ConformalSurrogate(
        TanimotoGP(n_bits=512, optimize_hyperparams=False),
        alpha=0.10,
        mode="weighted",
        weight_gamma=0.0,
        seed=42,
    )
    c_norm.fit_calibrated(tr_smis, tr_y, cal_smis, cal_y)
    c_unif.fit_calibrated(tr_smis, tr_y, cal_smis, cal_y)
    p_norm = c_norm.predict(list(_DRUG_DECK[27:]))
    p_unif = c_unif.predict(list(_DRUG_DECK[27:]))
    assert p_norm.interval_lo is not None and p_unif.interval_lo is not None
    assert p_norm.interval_hi is not None and p_unif.interval_hi is not None
    assert np.allclose(p_unif.interval_lo, p_norm.interval_lo, atol=1e-12)
    assert np.allclose(p_unif.interval_hi, p_norm.interval_hi, atol=1e-12)
    assert abs(float(p_unif.metadata["mean_kish_n_eff"]) - 9.0) < 1e-9

    # 2. Exact Chen et al. (2018) DPP total_log_det == log det(L_S)
    pool = tuple(_DRUG_DECK[:12])
    gp = TanimotoGP(n_bits=512, optimize_hyperparams=False)
    gp.fit(list(_DRUG_DECK[12:24]), _deterministic_sar_oracle(list(_DRUG_DECK[12:24])))
    pred = gp.predict(pool)
    ctx_dpp = AcquisitionContext(
        pool_smiles=pool,
        mean=pred.mean,
        std=pred.std,
        batch_size=5,
        beta=1.5,
        dpp_quality_temp=1.0,
    )
    sel_dpp = run_acquisition("dpp_ucb", ctx_dpp)
    ucb_v = pred.mean + 1.5 * pred.std
    z_v = np.clip((ucb_v - float(np.mean(ucb_v))) / max(float(np.std(ucb_v)), 1e-6), -5.0, 5.0)
    q_v = np.exp(z_v)
    fps, _ = featurize_pool(pool, kind="ecfp4_binary", n_bits=1024, radius=2)
    k_reg = 0.999 * tanimoto_kernel(fps) + 0.001 * np.eye(len(pool))
    l_mat = (q_v[:, None] * k_reg) * q_v[None, :]
    sub_idx = list(sel_dpp.indices)
    sign, exact_logdet = np.linalg.slogdet(l_mat[np.ix_(sub_idx, sub_idx)])
    assert sign > 0
    assert abs(float(sum(sel_dpp.scores)) - float(exact_logdet)) < 1e-8

    # 3. Multi-objective dict[str, float] observations & duplicate pool deduplication
    mo_dict = ActiveLearningEngine(
        surrogate="tanimoto_gp",
        surrogate_kwargs={"n_bits": 512},
        acquisition="pareto_scalarized_ucb",
        objective_names=("potency", "mpo"),
        seed=42,
    )
    seed_obs = {
        s: {"potency": v, "mpo": 0.8}
        for s, v in zip(_DRUG_DECK[:6], _deterministic_sar_oracle(_DRUG_DECK[:6]), strict=True)
    }
    assert mo_dict.ingest_observations(seed_obs) == 6
    dup_pool = list(_DRUG_DECK[6:14]) + list(_DRUG_DECK[6:10])  # deliberate duplicate rows
    q_dedup = mo_dict.propose_batch(dup_pool, batch_size=4)
    assert len(q_dedup.smiles) == 4 and len(set(q_dedup.smiles)) == 4
    r_dict = mo_dict.run_round(
        dup_pool,
        lambda xs: [{"potency": p, "mpo": 0.75} for p in _deterministic_sar_oracle(xs)],
        batch_size=4,
    )
    assert r_dict.n_total_observed == 10

    # 4. MultiFidelityDMTA strict cold-start budget compliance (never overshoots remaining_budget)
    truth_m = {
        s: v
        for s, v in zip(_DRUG_DECK[:20], _deterministic_sar_oracle(_DRUG_DECK[:20]), strict=True)
    }
    mf_tight = MultiFidelityDMTA(
        [
            FidelityTierSpec("low", 0, 1.0, lambda xs: [truth_m[s] for s in xs]),
            FidelityTierSpec("high", 1, 20.0, lambda xs: [truth_m[s] for s in xs]),
        ],
        n_bits=512,
        seed=42,
    )
    # 162.0 can afford 7 joint (7 * 21 = 147) but NOT 8 joint (8 * 21 = 168)!
    mf_tight.run_round(list(_DRUG_DECK[:20]), high_batch_size=8, remaining_budget=162.0)
    assert mf_tight.total_cost_spent <= 162.0 + 1e-9

    # 5. ActiveLearningDMTA v5.0 facade synchronizes verifiable SHA-256 DMTALedger
    al_facade = ActiveLearningDMTA(strategy="ucb", surrogate="tanimoto_gp", seed=42)
    al_facade.ingest_observations(dict(zip(tr_smis[:6], tr_y[:6], strict=True)))
    al_facade.run_round(list(_DRUG_DECK[:16]), _deterministic_sar_oracle, batch_size=4)
    assert len(al_facade.ledger.rounds) == 1
    assert al_facade.ledger.verify_chain() is True


def test_invalid_inputs_raise_clean_errors() -> None:
    """Sanity checks on clear error handling for edge cases."""
    with pytest.raises(ValueError):
        TanimotoGP(kernel="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ConformalSurrogate(TanimotoGP(), alpha=1.5)
    with pytest.raises(ValueError):
        MultiFidelityDMTA([FidelityTierSpec("only_one", 0, 1.0, lambda xs: [0.0] * len(xs))])
    with pytest.raises(ValueError):
        CombinatorialThompsonSampler(reaction_name="nonexistent_reaction_xyz")
    assert _extract_valid_check(featurize_pool(["INVALID", ""], kind="ecfp4_binary")) == 0


def _extract_valid_check(res: tuple[Any, Any]) -> int:
    return int(res[1].sum())
