"""Checkpoint-backed co-folding shared by Studio and its exported Python programs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geneuclidean.folding.runners import FoldedComplexResult


def cofold_candidate(
    trunk: Mapping[str, Any],
    molecules: Sequence[str],
    *,
    seed: int,
    num_steps: int,
    num_samples: int,
    flex_mode: str,
    checkpoint_path: str | Path | None = None,
) -> FoldedComplexResult | None:
    """Co-fold the first candidate with the source pocket and local NanoFold3 weights.

    The checkpoint supplies the trained architecture. The local server and exported
    scripts read GENEUCLIDEAN_NANOFOLD3_CHECKPOINT unless a path is supplied explicitly.
    A missing input yields no result. Invalid or incompatible checkpoints raise an error.
    """
    from geneuclidean.folding import fold_complex

    checkpoint = checkpoint_path or os.environ.get("GENEUCLIDEAN_NANOFOLD3_CHECKPOINT")
    pocket = trunk.get("pocket")
    pdb = (
        pocket.get("pocket_pdb")
        if isinstance(pocket, Mapping)
        else getattr(pocket, "pocket_pdb", None)
    )
    candidates = molecules or trunk.get("molecules") or ()
    if not checkpoint or not pdb or not Path(pdb).is_file() or not candidates:
        return None
    return fold_complex(
        pocket_pdb=pdb,
        ligand_smiles=candidates[0],
        backend="nanofold3",
        checkpoint_path=checkpoint,
        seeds=(seed,),
        num_steps=num_steps,
        num_samples=num_samples,
        flex_mode=flex_mode,
        predict_affinity=True,
    )


def confidence_scores(
    result: FoldedComplexResult | None, *, include_affinity: bool
) -> dict[str, float]:
    """Read finite confidence metrics from the same co-folded candidate."""
    if result is None or result.metadata.get("weight_source") == "random_initialization":
        return {}
    scores = {
        "plddt_mean": result.plddt_mean,
        "ptm": result.ptm,
        "iptm": result.iptm,
        "ranking_score": result.ranking_score,
    }
    if include_affinity and result.predicted_pkd is not None:
        scores["predicted_pkd"] = result.predicted_pkd
    return {name: score for name, score in scores.items() if math.isfinite(score)}


def passes_confidence(scores: Mapping[str, float], *, min_iptm: float, min_plddt: float) -> bool:
    """Require both reported confidence metrics to pass the configured cutoffs."""
    return (
        scores.get("iptm", -math.inf) >= min_iptm
        and scores.get("plddt_mean", -math.inf) >= min_plddt
    )
