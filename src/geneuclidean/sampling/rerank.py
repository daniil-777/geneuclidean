"""Multi-parameter optimization (MPO) reranking and MaxMin sphere-exclusion selection."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import (
    QED,
    Crippen,
    Descriptors,
    FilterCatalog,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

_SAS_MODULE: Any | None = None
_PAINS_CATALOG: Any | None = None

DEFAULT_MPO_WEIGHTS: dict[str, float] = {
    "qed": 0.25,
    "sa": 0.20,
    "lipinski": 0.10,
    "pains": 0.10,
    "pkd": 0.25,
    "clip": 0.10,
}


@dataclass(frozen=True)
class MPOScore:
    """Multi-parameter medicinal chemistry & binding desirability profile for one ligand."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    qed: float
    sa_score: float
    sa_norm: float
    lipinski_violations: int
    lipinski_pass: bool
    pains_free: bool
    pred_pkd: float | None
    clip_sim: float | None
    mpo_score: float


def _load_sascorer() -> Any:
    """Lazily load RDKit's built-in ``SA_Score/sascorer`` from ``RDConfig.RDContribDir``."""
    global _SAS_MODULE
    if _SAS_MODULE is None:
        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.insert(0, sa_dir)
        _SAS_MODULE = importlib.import_module("sascorer")
    return _SAS_MODULE


def _get_pains_catalog() -> Any:
    """Lazily construct the RDKit PAINS structural alert filter catalog."""
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def _compute_sa_score(mol: Chem.Mol) -> float:
    """Compute Ertl synthetic accessibility score in ``[1.0, 10.0]`` (lower is easier)."""
    sascorer = _load_sascorer()
    return float(sascorer.calculateScore(mol))


def _lipinski_violations(mol: Chem.Mol) -> int:
    """Count Lipinski Rule-of-5 violations (MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10)."""
    descriptors_mod: Any = Descriptors
    crippen_mod: Any = Crippen
    mw = float(descriptors_mod.MolWt(mol))
    logp = float(crippen_mod.MolLogP(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    violations = 0
    if mw > 500.0:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def compute_mpo_scores(
    smiles_list: Sequence[str],
    pred_pkd: Sequence[float] | np.ndarray | None = None,
    clip_sim: Sequence[float] | np.ndarray | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[MPOScore]:
    """Score candidate SMILES by medicinal chemistry + binding Multi-Parameter Optimization.

    Combines QED, normalized Ertl Synthetic Accessibility (``(10 - SA) / 9``), Lipinski
    Rule-of-5 compliance, RDKit PAINS-free indicator, optional predicted binding affinity
    ``pKd``, and optional DrugCLIP cosine similarity into a normalized desirability score
    in ``[0.0, 1.0]``. Invalid SMILES receive ``mpo_score = 0.0``; PAINS-flagged molecules
    receive a strict multiplicative penalty.

    Args:
        smiles_list: Candidate SMILES strings.
        pred_pkd: Optional per-ligand predicted binding affinities (pKd).
        clip_sim: Optional per-ligand DrugCLIP cosine similarities in ``[-1, 1]``.
        weights: Optional overrides for ``{"qed", "sa", "lipinski", "pains", "pkd", "clip"}``.

    Returns:
        List of ``MPOScore`` objects aligned with ``smiles_list``.
    """
    n = len(smiles_list)
    if pred_pkd is not None and len(pred_pkd) != n:
        raise ValueError(f"pred_pkd length ({len(pred_pkd)}) != smiles_list length ({n})")
    if clip_sim is not None and len(clip_sim) != n:
        raise ValueError(f"clip_sim length ({len(clip_sim)}) != smiles_list length ({n})")

    eff_weights = dict(DEFAULT_MPO_WEIGHTS)
    if weights is not None:
        eff_weights.update({k: float(v) for k, v in weights.items()})

    pains_catalog = _get_pains_catalog()
    results: list[MPOScore] = []

    for idx, smi in enumerate(smiles_list):
        pkd_val: float | None = None
        if pred_pkd is not None:
            raw_pkd = float(pred_pkd[idx])
            if math.isfinite(raw_pkd):
                pkd_val = raw_pkd

        clip_val: float | None = None
        if clip_sim is not None:
            raw_clip = float(clip_sim[idx])
            if math.isfinite(raw_clip):
                clip_val = raw_clip

        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        if mol is None:
            results.append(
                MPOScore(
                    smiles=smi,
                    canonical_smiles=None,
                    valid=False,
                    qed=0.0,
                    sa_score=10.0,
                    sa_norm=0.0,
                    lipinski_violations=4,
                    lipinski_pass=False,
                    pains_free=False,
                    pred_pkd=pkd_val,
                    clip_sim=clip_val,
                    mpo_score=0.0,
                )
            )
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        qed_val = float(QED.qed(mol))
        sa_raw = _compute_sa_score(mol)
        sa_norm = float(np.clip((10.0 - sa_raw) / 9.0, 0.0, 1.0))
        violations = _lipinski_violations(mol)
        ro5_pass = violations <= 1
        ro5_score = float(4 - violations) / 4.0
        pains_free = not bool(pains_catalog.HasMatch(mol))

        components: list[tuple[float, float]] = [
            (eff_weights.get("qed", 0.0), qed_val),
            (eff_weights.get("sa", 0.0), sa_norm),
            (eff_weights.get("lipinski", 0.0), ro5_score),
            (eff_weights.get("pains", 0.0), 1.0 if pains_free else 0.0),
        ]
        if pkd_val is not None:
            pkd_norm = float(np.clip((pkd_val - 4.0) / 6.0, 0.0, 1.0))
            components.append((eff_weights.get("pkd", 0.0), pkd_norm))
        if clip_val is not None:
            clip_norm = float(np.clip((clip_val + 1.0) / 2.0, 0.0, 1.0))
            components.append((eff_weights.get("clip", 0.0), clip_norm))

        weight_sum = sum(max(0.0, w) for w, _ in components)
        if weight_sum <= 0.0:
            mpo = 0.0
        else:
            mpo = sum(max(0.0, w) * val for w, val in components) / weight_sum
        if not pains_free:
            mpo *= 0.5

        results.append(
            MPOScore(
                smiles=smi,
                canonical_smiles=canonical,
                valid=True,
                qed=qed_val,
                sa_score=sa_raw,
                sa_norm=sa_norm,
                lipinski_violations=violations,
                lipinski_pass=ro5_pass,
                pains_free=pains_free,
                pred_pkd=pkd_val,
                clip_sim=clip_val,
                mpo_score=float(np.clip(mpo, 0.0, 1.0)),
            )
        )

    return results


def maxmin_diversity_select(
    smiles_list: Sequence[str],
    scores: Sequence[float] | np.ndarray,
    k: int,
    min_tanimoto_dist: float = 0.35,
    *,
    radius: int = 2,
    fp_size: int = 2048,
    quality_weight: float = 0.7,
    relax_if_needed: bool = False,
) -> list[int]:
    """Greedy quality-weighted MaxMin / sphere-exclusion selector for diverse top-K hits.

    Deduplicates canonical SMILES (retaining the highest-scoring occurrence), skips invalid
    molecules, selects the highest-scoring valid molecule first, and iteratively selects
    candidates satisfying ``min_{j in S} (1 - Tanimoto(i, j)) >= min_tanimoto_dist`` that
    maximize ``quality_weight * norm_score + (1 - quality_weight) * min_dist``.

    Args:
        smiles_list: Candidate SMILES strings.
        scores: Quality / MPO desirability scores aligned with ``smiles_list``.
        k: Maximum number of diverse hits to select.
        min_tanimoto_dist: Minimum pairwise Morgan Tanimoto distance in ``[0, 1]``.
        radius: Morgan fingerprint radius (default 2 = ECFP4).
        fp_size: Morgan fingerprint bit vector size.
        quality_weight: Trade-off in ``[0, 1]`` between quality and MaxMin distance among
            sphere-excluded candidates.
        relax_if_needed: If ``True`` and fewer than ``k`` molecules satisfy
            ``min_tanimoto_dist``, backfill remaining slots by pure MaxMin distance.

    Returns:
        List of selected indices (into ``smiles_list``) of length ``<= k``.
    """
    n = len(smiles_list)
    if len(scores) != n:
        raise ValueError(f"scores length ({len(scores)}) != smiles_list length ({n})")
    if k <= 0 or n == 0:
        return []

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Parse valid molecules and deduplicate canonical SMILES keeping highest score
    best_by_canonical: dict[str, tuple[int, float, Any]] = {}
    for idx, smi in enumerate(smiles_list):
        score_val = float(scores[idx])
        if not math.isfinite(score_val):
            continue
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        existing = best_by_canonical.get(canonical)
        if existing is None or score_val > existing[1]:
            fp = fp_gen.GetFingerprint(mol)
            best_by_canonical[canonical] = (idx, score_val, fp)

    if not best_by_canonical:
        return []

    pool = list(best_by_canonical.values())
    orig_indices = [item[0] for item in pool]
    pool_scores = np.array([item[1] for item in pool], dtype=np.float64)
    pool_fps = [item[2] for item in pool]
    m = len(pool)

    # Min-max normalize scores for quality-weighted MaxMin tie-breaking
    s_min = float(pool_scores.min())
    s_max = float(pool_scores.max())
    if s_max - s_min > 1e-12:
        norm_scores = (pool_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(pool_scores)

    # First hit is always the highest-scoring valid molecule
    first_pos = int(np.argmax(pool_scores))
    selected_positions: list[int] = [first_pos]
    remaining = np.ones(m, dtype=bool)
    remaining[first_pos] = False

    # Incremental min Tanimoto distance to the selected set
    sims = np.array(
        DataStructs.BulkTanimotoSimilarity(pool_fps[first_pos], pool_fps),
        dtype=np.float64,
    )
    min_dists = 1.0 - sims
    min_dists[first_pos] = -1.0

    qw = float(np.clip(quality_weight, 0.0, 1.0))

    while len(selected_positions) < k and bool(remaining.any()):
        eligible = remaining & (min_dists >= float(min_tanimoto_dist) - 1e-12)
        if bool(eligible.any()):
            eligible_idx = np.flatnonzero(eligible)
            combined = qw * norm_scores[eligible_idx] + (1.0 - qw) * min_dists[eligible_idx]
            next_pos = int(eligible_idx[int(np.argmax(combined))])
        elif relax_if_needed:
            rem_idx = np.flatnonzero(remaining)
            next_pos = int(rem_idx[int(np.argmax(min_dists[rem_idx]))])
        else:
            break

        selected_positions.append(next_pos)
        remaining[next_pos] = False
        new_sims = np.array(
            DataStructs.BulkTanimotoSimilarity(pool_fps[next_pos], pool_fps),
            dtype=np.float64,
        )
        min_dists = np.minimum(min_dists, 1.0 - new_sims)
        min_dists[next_pos] = -1.0

    return [orig_indices[pos] for pos in selected_positions]


def rerank_and_select(
    smiles_list: Sequence[str],
    k: int,
    *,
    pred_pkd: Sequence[float] | np.ndarray | None = None,
    clip_sim: Sequence[float] | np.ndarray | None = None,
    min_tanimoto_dist: float = 0.35,
    weights: Mapping[str, float] | None = None,
    quality_weight: float = 0.7,
    relax_if_needed: bool = False,
) -> list[MPOScore]:
    """Compute MPO desirability scores and return the top-K diverse hits."""
    mpo_results = compute_mpo_scores(
        smiles_list,
        pred_pkd=pred_pkd,
        clip_sim=clip_sim,
        weights=weights,
    )
    scores = [entry.mpo_score for entry in mpo_results]
    chosen_indices = maxmin_diversity_select(
        smiles_list,
        scores=scores,
        k=k,
        min_tanimoto_dist=min_tanimoto_dist,
        quality_weight=quality_weight,
        relax_if_needed=relax_if_needed,
    )
    return [mpo_results[i] for i in chosen_indices]


__all__ = [
    "DEFAULT_MPO_WEIGHTS",
    "MPOScore",
    "compute_mpo_scores",
    "maxmin_diversity_select",
    "rerank_and_select",
]
