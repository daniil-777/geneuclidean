"""Autoregressive sampling strategies, CFG/CVAE decoding, MPO scoring, and MaxMin selection."""

from __future__ import annotations

from geneuclidean.sampling.decoder_strategies import (
    SamplingResult,
    apply_repetition_penalty,
    apply_smiles_syntax_guardrails,
    combine_cfg_logits,
    compute_strategy_probs,
    compute_temperature_probs,
    diverse_beam_search,
    filter_min_p_probs,
    filter_top_k,
    filter_top_p_probs,
    filter_typical_probs,
    sample_molecules,
    sample_molecules_detailed,
    sample_with_attention,
    select_next_tokens,
)
from geneuclidean.sampling.rerank import (
    DEFAULT_MPO_WEIGHTS,
    MPOScore,
    compute_mpo_scores,
    maxmin_diversity_select,
    rerank_and_select,
)

__all__ = [
    "DEFAULT_MPO_WEIGHTS",
    "MPOScore",
    "SamplingResult",
    "apply_repetition_penalty",
    "apply_smiles_syntax_guardrails",
    "combine_cfg_logits",
    "compute_mpo_scores",
    "compute_strategy_probs",
    "compute_temperature_probs",
    "diverse_beam_search",
    "filter_min_p_probs",
    "filter_top_k",
    "filter_top_p_probs",
    "filter_typical_probs",
    "maxmin_diversity_select",
    "rerank_and_select",
    "sample_molecules",
    "sample_molecules_detailed",
    "sample_with_attention",
    "select_next_tokens",
]
