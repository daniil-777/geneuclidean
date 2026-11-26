"""Foundational Chemical NLP & Multimodal Pocket-to-Ligand Operations Library.

Implements PEP 562 lazy loading (``__getattr__`` / ``__dir__``) with full PEP 561 static type
checking (``if TYPE_CHECKING:``) so ``import geneuclidean.nlp as nlp`` completes in ``< 2 ms``
with **zero** eager ``torch``, ``rdkit``, ``transformers``, or ``peft`` imports.

All underlying tensor primitives execute strictly in ``torch.float32`` natively on Apple
Silicon MPS (``PYTORCH_ENABLE_MPS_FALLBACK=0``), CUDA, and CPU without heavy external
frameworks.

Cookbook — Composing a Pocket-Conditioned Chemical Generator & RL Aligner
-------------------------------------------------------------------------

.. code-block:: python

    import torch
    import geneuclidean.nlp as nlp

    # 1. Atom-level SMILES / SELFIES / SAFE tokenization + stateful valence grammar mask
    tok = nlp.AtomLevelSMILESTokenizer()
    ids, attn_mask = tok.batch_encode(["NCCc1ccc(O)c(O)c1", "c1ccc2cc1.CC2"])
    grammar = nlp.SMILESValenceGrammarMask(tok)

    # 2. Pocket-conditioned RoPE + GQA + SwiGLU + Perceiver Chemical LM + pure-PyTorch LoRA
    lm = nlp.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
        d_pocket=192, conditioning_mode="cross_attn", use_perceiver_bridge=True,
        rope_scaling_type="yarn", rope_scaling_factor=2.0,
    )
    nlp.inject_lora_adapters(lm, target_modules=("q_proj", "v_proj"), rank=8, alpha=16.0)
    out = lm(ids, pocket_tokens=pocket_emb, pocket_key_padding_mask=~atom_mask, labels=ids)
    nlp.merge_lora_adapters(lm, unwrap=True)

    # 3. Contrastive multi-conformation CFG + deterministic KV-cached grammar beam search
    guided = nlp.apply_contrastive_cfg_logits(
        [on_conf1_logits, on_conf2_logits], uncond_logits,
        off_target_logits=herg_logits, w_on=1.6, w_off=0.5,
    )
    filtered = nlp.apply_temperature_top_k_top_p(guided, temperature=0.9, top_k=30, top_p=0.92)
    beams = nlp.beam_search_decode(
        lm, pocket_tokens=pocket_emb[:1], beam_width=4, grammar_mask=grammar, use_cache=True,
    )

    # 4. Critic-free molecular RL alignment (DPO / cDPO / IPO & GRPO)
    dpo = nlp.dpo_preference_loss(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1)
    grpo = nlp.grpo_policy_loss(policy_lp, old_lp, rewards, ref_logps=ref_lp, clip_eps=0.2)

    # 5. Non-autoregressive Discrete Masked Diffusion (MDLM) & SAFE-fragment inpainting
    mdlm = nlp.DiscreteMaskedDiffusionStep(mask_token_id=tok.mask_id, schedule="cosine")
    corrupted = mdlm.forward_corrupt(ids, t=0.5, fixed_mask=scaffold_pin_mask)
    inpainted = mdlm.sample_inpaint(lm, corrupted.noisy_ids, num_steps=8, pocket_tokens=pocket_emb)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geneuclidean.models.nlp_ops import (
        BOS_ID as BOS_ID,
    )
    from geneuclidean.models.nlp_ops import (
        BOS_TOKEN as BOS_TOKEN,
    )
    from geneuclidean.models.nlp_ops import (
        CHEMICAL_TOKEN_REGEX as CHEMICAL_TOKEN_REGEX,
    )
    from geneuclidean.models.nlp_ops import (
        DEFAULT_CHEMICAL_VOCAB as DEFAULT_CHEMICAL_VOCAB,
    )
    from geneuclidean.models.nlp_ops import (
        EOS_ID as EOS_ID,
    )
    from geneuclidean.models.nlp_ops import (
        EOS_TOKEN as EOS_TOKEN,
    )
    from geneuclidean.models.nlp_ops import (
        MASK_ID as MASK_ID,
    )
    from geneuclidean.models.nlp_ops import (
        MASK_TOKEN as MASK_TOKEN,
    )
    from geneuclidean.models.nlp_ops import (
        PAD_ID as PAD_ID,
    )
    from geneuclidean.models.nlp_ops import (
        PAD_TOKEN as PAD_TOKEN,
    )
    from geneuclidean.models.nlp_ops import (
        SEP_ID as SEP_ID,
    )
    from geneuclidean.models.nlp_ops import (
        SEP_TOKEN as SEP_TOKEN,
    )
    from geneuclidean.models.nlp_ops import (
        SPECIAL_TOKENS as SPECIAL_TOKENS,
    )
    from geneuclidean.models.nlp_ops import (
        UNK_ID as UNK_ID,
    )
    from geneuclidean.models.nlp_ops import (
        UNK_TOKEN as UNK_TOKEN,
    )
    from geneuclidean.models.nlp_ops import (
        AtomLevelSMILESTokenizer as AtomLevelSMILESTokenizer,
    )
    from geneuclidean.models.nlp_ops import (
        BeamSearchResult as BeamSearchResult,
    )
    from geneuclidean.models.nlp_ops import (
        ChemicalLMOutput as ChemicalLMOutput,
    )
    from geneuclidean.models.nlp_ops import (
        ChemicalTransformerBlock as ChemicalTransformerBlock,
    )
    from geneuclidean.models.nlp_ops import (
        DiscreteMaskedDiffusionStep as DiscreteMaskedDiffusionStep,
    )
    from geneuclidean.models.nlp_ops import (
        DPOLossOutput as DPOLossOutput,
    )
    from geneuclidean.models.nlp_ops import (
        GroupedQueryAttention as GroupedQueryAttention,
    )
    from geneuclidean.models.nlp_ops import (
        GRPOLossOutput as GRPOLossOutput,
    )
    from geneuclidean.models.nlp_ops import (
        KVCache as KVCache,
    )
    from geneuclidean.models.nlp_ops import (
        LayerKVCache as LayerKVCache,
    )
    from geneuclidean.models.nlp_ops import (
        LoRALinear as LoRALinear,
    )
    from geneuclidean.models.nlp_ops import (
        MaskedDiffusionCorruption as MaskedDiffusionCorruption,
    )
    from geneuclidean.models.nlp_ops import (
        PocketConditionedChemicalLM as PocketConditionedChemicalLM,
    )
    from geneuclidean.models.nlp_ops import (
        PocketCrossAttention as PocketCrossAttention,
    )
    from geneuclidean.models.nlp_ops import (
        PocketPerceiverBridge as PocketPerceiverBridge,
    )
    from geneuclidean.models.nlp_ops import (
        RMSNorm as RMSNorm,
    )
    from geneuclidean.models.nlp_ops import (
        RotaryPositionEmbedding as RotaryPositionEmbedding,
    )
    from geneuclidean.models.nlp_ops import (
        SMILESValenceGrammarMask as SMILESValenceGrammarMask,
    )
    from geneuclidean.models.nlp_ops import (
        SwiGLUFeedForward as SwiGLUFeedForward,
    )
    from geneuclidean.models.nlp_ops import (
        apply_contrastive_cfg_logits as apply_contrastive_cfg_logits,
    )
    from geneuclidean.models.nlp_ops import (
        apply_rotary_pos_emb as apply_rotary_pos_emb,
    )
    from geneuclidean.models.nlp_ops import (
        apply_temperature_top_k_top_p as apply_temperature_top_k_top_p,
    )
    from geneuclidean.models.nlp_ops import (
        beam_search_decode as beam_search_decode,
    )
    from geneuclidean.models.nlp_ops import (
        dpo_preference_loss as dpo_preference_loss,
    )
    from geneuclidean.models.nlp_ops import (
        grpo_policy_loss as grpo_policy_loss,
    )
    from geneuclidean.models.nlp_ops import (
        inject_lora_adapters as inject_lora_adapters,
    )
    from geneuclidean.models.nlp_ops import (
        merge_lora_adapters as merge_lora_adapters,
    )
    from geneuclidean.models.nlp_ops import (
        sequence_logprobs_from_logits as sequence_logprobs_from_logits,
    )

_NLP_OPS_MOD = "geneuclidean.models.nlp_ops"

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Tokenization & Grammar Constraints
    "BOS_ID": (_NLP_OPS_MOD, "BOS_ID"),
    "BOS_TOKEN": (_NLP_OPS_MOD, "BOS_TOKEN"),
    "CHEMICAL_TOKEN_REGEX": (_NLP_OPS_MOD, "CHEMICAL_TOKEN_REGEX"),
    "DEFAULT_CHEMICAL_VOCAB": (_NLP_OPS_MOD, "DEFAULT_CHEMICAL_VOCAB"),
    "EOS_ID": (_NLP_OPS_MOD, "EOS_ID"),
    "EOS_TOKEN": (_NLP_OPS_MOD, "EOS_TOKEN"),
    "MASK_ID": (_NLP_OPS_MOD, "MASK_ID"),
    "MASK_TOKEN": (_NLP_OPS_MOD, "MASK_TOKEN"),
    "PAD_ID": (_NLP_OPS_MOD, "PAD_ID"),
    "PAD_TOKEN": (_NLP_OPS_MOD, "PAD_TOKEN"),
    "SEP_ID": (_NLP_OPS_MOD, "SEP_ID"),
    "SEP_TOKEN": (_NLP_OPS_MOD, "SEP_TOKEN"),
    "SPECIAL_TOKENS": (_NLP_OPS_MOD, "SPECIAL_TOKENS"),
    "UNK_ID": (_NLP_OPS_MOD, "UNK_ID"),
    "UNK_TOKEN": (_NLP_OPS_MOD, "UNK_TOKEN"),
    "AtomLevelSMILESTokenizer": (_NLP_OPS_MOD, "AtomLevelSMILESTokenizer"),
    "SMILESValenceGrammarMask": (_NLP_OPS_MOD, "SMILESValenceGrammarMask"),
    # Champion Transformer & Multimodal Pocket Bridge Primitives
    "RMSNorm": (_NLP_OPS_MOD, "RMSNorm"),
    "RotaryPositionEmbedding": (_NLP_OPS_MOD, "RotaryPositionEmbedding"),
    "apply_rotary_pos_emb": (_NLP_OPS_MOD, "apply_rotary_pos_emb"),
    "SwiGLUFeedForward": (_NLP_OPS_MOD, "SwiGLUFeedForward"),
    "LayerKVCache": (_NLP_OPS_MOD, "LayerKVCache"),
    "KVCache": (_NLP_OPS_MOD, "KVCache"),
    "GroupedQueryAttention": (_NLP_OPS_MOD, "GroupedQueryAttention"),
    "PocketCrossAttention": (_NLP_OPS_MOD, "PocketCrossAttention"),
    "PocketPerceiverBridge": (_NLP_OPS_MOD, "PocketPerceiverBridge"),
    "ChemicalTransformerBlock": (_NLP_OPS_MOD, "ChemicalTransformerBlock"),
    "ChemicalLMOutput": (_NLP_OPS_MOD, "ChemicalLMOutput"),
    "PocketConditionedChemicalLM": (_NLP_OPS_MOD, "PocketConditionedChemicalLM"),
    "LoRALinear": (_NLP_OPS_MOD, "LoRALinear"),
    "inject_lora_adapters": (_NLP_OPS_MOD, "inject_lora_adapters"),
    "merge_lora_adapters": (_NLP_OPS_MOD, "merge_lora_adapters"),
    # Decoding, Contrastive Steering, Molecular RL Alignment & Masked Diffusion
    "apply_temperature_top_k_top_p": (_NLP_OPS_MOD, "apply_temperature_top_k_top_p"),
    "apply_contrastive_cfg_logits": (_NLP_OPS_MOD, "apply_contrastive_cfg_logits"),
    "BeamSearchResult": (_NLP_OPS_MOD, "BeamSearchResult"),
    "beam_search_decode": (_NLP_OPS_MOD, "beam_search_decode"),
    "sequence_logprobs_from_logits": (_NLP_OPS_MOD, "sequence_logprobs_from_logits"),
    "DPOLossOutput": (_NLP_OPS_MOD, "DPOLossOutput"),
    "dpo_preference_loss": (_NLP_OPS_MOD, "dpo_preference_loss"),
    "GRPOLossOutput": (_NLP_OPS_MOD, "GRPOLossOutput"),
    "grpo_policy_loss": (_NLP_OPS_MOD, "grpo_policy_loss"),
    "MaskedDiffusionCorruption": (_NLP_OPS_MOD, "MaskedDiffusionCorruption"),
    "DiscreteMaskedDiffusionStep": (_NLP_OPS_MOD, "DiscreteMaskedDiffusionStep"),
}

__all__ = [
    *sorted(_LAZY_ATTRS.keys()),
    "list_nlp_primitives",
]


def list_nlp_primitives() -> list[str]:
    """Return sorted names of all public Chemical NLP operations without importing PyTorch."""
    return sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    """Lazily resolve Chemical NLP primitives on first attribute access (PEP 562)."""
    if name in _LAZY_ATTRS:
        mod_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose complete lazy Chemical NLP library surface to ``dir(geneuclidean.nlp)``."""
    return sorted(set(globals().keys()) | set(__all__))
