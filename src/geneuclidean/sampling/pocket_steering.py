"""Multi-conformation 3D pocket ensemble steering, contrastive off-target CFG, and PMI_3D.

Scientific motivation:
    Addresses three fundamental limitations in current 3D pocket-conditioned chemical language
    models such as ETH Zurich MODLAB's ``DRAGONFLY`` (Atz et al., *Nat. Commun.* 2024) and
    model-intrinsic CLM beam-search scoring (Moret, Grisoni & Schneider):

    1. **Rigid Single-Pocket Limitation -> Multi-Conformation Pocket Ensembling:**
       Conditioning on a single static PDB crystal structure ignores induced-fit side-chain
       rearrangements. :func:`sample_contrastive_ensemble_llm` simultaneously conditions on an
       ensemble of ``K`` experimental holo/apo conformations ``{P_on^(1), ..., P_on^(K)}`` by
       combining per-conformation next-token logit distributions inside a single KV-cached loop.
    2. **Zero-Retraining Off-Target Repulsion -> Contrastive 3D Pocket CFG:**
       Extends standard Classifier-Free Guidance with an optional counter-target pocket stream
       ``P_off`` (e.g., ``DRD2`` when designing ``DRD1`` leads, or ``BCHE`` when designing
       ``ACHE`` leads):

       .. math::
           \\ell_{\\text{guided}} = \\ell_{\\varnothing}
           + w_{\\text{on}} (\\bar{\\ell}_{\\text{on}} - \\ell_{\\varnothing})
           - w_{\\text{off}} (\\ell_{\\text{off}} - \\ell_{\\varnothing})

    3. **Model-Intrinsic 3D Specificity Scoring -> Pointwise Mutual Information (``PMI_3D``):**
       :func:`score_pocket_pmi` computes exact length-normalized 3D pocket-conditional
       Pointwise Mutual Information ``log P(m | P_on) - log P(m | P_null)`` and differential
       selectivity ``DeltaPMI_3D = log P(m | P_on) - log P(m | P_off)`` for zero-shot
       target-specificity ranking without any external QSAR model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.models.encoders import PocketEncoderOutput
from geneuclidean.models.safe_codec import smiles_to_safe
from geneuclidean.sampling.decoder_strategies import apply_repetition_penalty, select_next_tokens

if TYPE_CHECKING:
    from geneuclidean.models.llm_bridge import PocketConditionedLLM


@dataclass(frozen=True)
class PocketPMIScore:
    """Information-theoretic 3D pocket specificity & differential selectivity score."""

    smiles: str
    n_tokens: int
    logp_on: float
    logp_null: float
    pmi_3d: float
    logp_off: float | None = None
    delta_pmi_selectivity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize PMI score to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "n_tokens": self.n_tokens,
            "logp_on": round(self.logp_on, 5),
            "logp_null": round(self.logp_null, 5),
            "pmi_3d": round(self.pmi_3d, 5),
            "logp_off": round(self.logp_off, 5) if self.logp_off is not None else None,
            "delta_pmi_selectivity": (
                round(self.delta_pmi_selectivity, 5)
                if self.delta_pmi_selectivity is not None
                else None
            ),
        }


def combine_contrastive_ensemble_logits(
    on_logits_list: Sequence[Tensor],
    uncond_logits: Tensor,
    *,
    on_weights: Sequence[float] | None = None,
    w_on: float = 1.6,
    off_logits: Tensor | None = None,
    w_off: float = 0.0,
) -> Tensor:
    """Combine multi-conformation on-target, null, and optional off-target logits.

    Implements:
        ``logits_out = uncond + w_on * (sum_k alpha_k * on_k - uncond)
                       - w_off * (off - uncond)``
    where ``alpha`` is normalized to sum to ``1.0``.
    """
    if not on_logits_list:
        raise ValueError("on_logits_list must contain at least one on-target logit tensor.")
    if on_weights is None:
        alpha = [1.0 / len(on_logits_list)] * len(on_logits_list)
    else:
        if len(on_weights) != len(on_logits_list):
            raise ValueError("on_weights length must match on_logits_list length.")
        total_w = float(sum(on_weights))
        if total_w <= 0.0:
            raise ValueError("Sum of on_weights must be strictly positive.")
        alpha = [float(w) / total_w for w in on_weights]

    on_ens = sum(a * lg for a, lg in zip(alpha, on_logits_list, strict=True))
    assert isinstance(on_ens, Tensor)
    guided = uncond_logits + float(w_on) * (on_ens - uncond_logits)
    if off_logits is not None and abs(float(w_off)) > 1e-8:
        guided = guided - float(w_off) * (off_logits - uncond_logits)
    return guided


def _prepare_target_token_ids(model: PocketConditionedLLM, smiles: str) -> list[int]:
    """Tokenize a SMILES string in the backbone's native format (SMILES or canonical SAFE)."""
    text = smiles.strip()
    if model.format_mode == "safe":
        encoded_safe = smiles_to_safe(text)
        if encoded_safe is not None:
            text = encoded_safe
    full_text = f"{model.prompt_prefix}{text}" if model.prompt_prefix else text
    ids = list(model.tokenizer.encode(full_text, add_special_tokens=False))
    return [int(model.bos_token_id), *[int(t) for t in ids], int(model.eos_token_id)]


@torch.no_grad()
def _mean_conditional_logprob(
    model: PocketConditionedLLM,
    soft_prefix: Tensor,
    token_ids_list: Sequence[list[int]],
) -> tuple[list[float], list[int]]:
    """Compute exact per-token mean log-likelihood ``(1/T) sum_t log P(x_t | P, x_<t)``."""
    from geneuclidean.models.hf_backbones import get_input_embeddings_tensor

    device = soft_prefix.device
    pad_id = int(model.pad_token_id)
    max_len = max(len(seq) for seq in token_ids_list)
    bsz = len(token_ids_list)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(token_ids_list):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, : len(seq)] = 1

    tok_embeds = get_input_embeddings_tensor(model.llm, input_ids)
    prefix_exp = soft_prefix.expand(bsz, -1, -1).to(
        device=tok_embeds.device, dtype=tok_embeds.dtype
    )
    prefix_len = int(prefix_exp.shape[1])

    full_embeds = torch.cat([prefix_exp, tok_embeds], dim=1)
    prefix_mask = torch.ones((bsz, prefix_len), dtype=torch.long, device=tok_embeds.device)
    full_mask = torch.cat([prefix_mask, attn_mask], dim=1)

    outputs = model.llm(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        use_cache=False,
        return_dict=True,
    )
    # Token logits align so position ``prefix_len + t`` predicts ``input_ids[:, t + 1]``.
    tok_logits = outputs.logits[:, prefix_len : prefix_len + max_len - 1, :].float()
    log_probs = F.log_softmax(tok_logits, dim=-1)
    targets = input_ids[:, 1:]
    valid_targets = attn_mask[:, 1:].to(dtype=torch.float32)

    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * valid_targets
    counts = valid_targets.sum(dim=-1).clamp(min=1.0)
    mean_lp = (gathered.sum(dim=-1) / counts).detach().cpu().tolist()
    n_toks = [int(c) for c in counts.detach().cpu().tolist()]
    return [float(x) for x in mean_lp], n_toks


@torch.no_grad()
def score_pocket_pmi(
    model: PocketConditionedLLM,
    smiles_list: Sequence[str],
    on_pockets: Sequence[tuple[PocketEncoderOutput, Tensor]] | tuple[PocketEncoderOutput, Tensor],
    *,
    on_weights: Sequence[float] | None = None,
    off_pocket: tuple[PocketEncoderOutput, Tensor] | None = None,
    batch_size: int = 32,
) -> list[PocketPMIScore]:
    """Compute exact 3D Pocket-Conditional Pointwise Mutual Information (``PMI_3D``).

    Quantifies how strongly the 3D pocket geometry increases a candidate molecule's
    likelihood relative to the unconditional chemical prior (``null_prefix``), plus
    optional differential selectivity ``DeltaPMI_3D`` against an off-target pocket.

    Args:
        model: Trained :class:`PocketConditionedLLM` in eval mode.
        smiles_list: Candidate SMILES strings to score.
        on_pockets: Single ``(pocket_out, atom_mask)`` tuple OR sequence of ``K`` crystal
            conformations of the on-target pocket.
        on_weights: Optional ensemble weights across the ``K`` on-target conformations.
        off_pocket: Optional counter-target ``(pocket_out, atom_mask)`` for selectivity.
        batch_size: Evaluation chunk size.

    Returns:
        List of :class:`PocketPMIScore` aligned 1-to-1 with ``smiles_list``.
    """
    if not smiles_list:
        return []

    if (
        isinstance(on_pockets, tuple)
        and len(on_pockets) == 2
        and isinstance(on_pockets[0], PocketEncoderOutput)
    ):
        on_list: list[tuple[PocketEncoderOutput, Tensor]] = [on_pockets]
    else:
        on_list = list(on_pockets)

    if on_weights is None:
        alpha = [1.0 / len(on_list)] * len(on_list)
    else:
        total = float(sum(on_weights))
        alpha = [float(w) / total for w in on_weights]

    was_training = model.training
    model.eval()
    try:
        # Build deterministic single-item soft prefixes for each stream
        on_prefixes: list[Tensor] = []
        for p_out, p_mask in on_list:
            pref, _, _ = model.build_soft_prefix(
                pocket_out=p_out,
                atom_mask=p_mask,
                use_null_pocket=False,
                latent_temp=0.0,
                deterministic_z=True,
                return_attn=False,
            )
            on_prefixes.append(pref[:1])

        null_prefix, _, _ = model.build_soft_prefix(
            pocket_out=on_list[0][0],
            atom_mask=on_list[0][1],
            use_null_pocket=True,
            batch_size=1,
        )
        off_prefix: Tensor | None = None
        if off_pocket is not None:
            off_pref, _, _ = model.build_soft_prefix(
                pocket_out=off_pocket[0],
                atom_mask=off_pocket[1],
                use_null_pocket=False,
                latent_temp=0.0,
                deterministic_z=True,
                return_attn=False,
            )
            off_prefix = off_pref[:1]

        results: list[PocketPMIScore] = []
        for start in range(0, len(smiles_list), max(1, int(batch_size))):
            chunk_smi = list(smiles_list[start : start + batch_size])
            tok_rows = [_prepare_target_token_ids(model, s) for s in chunk_smi]

            lp_null, n_toks = _mean_conditional_logprob(model, null_prefix, tok_rows)
            lp_on_acc = [0.0] * len(chunk_smi)
            for a_k, pref_k in zip(alpha, on_prefixes, strict=True):
                lp_k, _ = _mean_conditional_logprob(model, pref_k, tok_rows)
                for i, val in enumerate(lp_k):
                    lp_on_acc[i] += a_k * val

            lp_off: list[float] | None = None
            if off_prefix is not None:
                lp_off, _ = _mean_conditional_logprob(model, off_prefix, tok_rows)

            for i, smi in enumerate(chunk_smi):
                off_val = lp_off[i] if lp_off is not None else None
                results.append(
                    PocketPMIScore(
                        smiles=smi,
                        n_tokens=n_toks[i],
                        logp_on=lp_on_acc[i],
                        logp_null=lp_null[i],
                        pmi_3d=lp_on_acc[i] - lp_null[i],
                        logp_off=off_val,
                        delta_pmi_selectivity=(
                            lp_on_acc[i] - off_val if off_val is not None else None
                        ),
                    )
                )
        return results
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def sample_contrastive_ensemble_llm(
    model: PocketConditionedLLM,
    on_pockets: Sequence[tuple[PocketEncoderOutput, Tensor]],
    *,
    on_weights: Sequence[float] | None = None,
    off_pocket: tuple[PocketEncoderOutput, Tensor] | None = None,
    w_on: float = 1.6,
    w_off: float = 0.0,
    num_samples: int = 32,
    max_new_tokens: int = 96,
    strategy: str = "min_p",
    temperature: float = 1.0,
    min_p: float = 0.08,
    top_p: float = 0.92,
    top_k: int = 30,
    latent_temp: float = 1.0,
    repetition_penalty: float = 1.05,
    batch_size: int = 16,
    seed: int | None = 42,
) -> list[str]:
    """Generate molecules via KV-cached multi-conformation ensemble & contrastive off-target CFG.

    Packs all ``K`` on-target crystal conformations, the unconditional ``null`` stream,
    and the optional off-target counter-pocket stream into a single batched KV-cached
    forward pass so inference speed scales linearly with ``K`` while eliminating rigid
    single-pocket bias and actively steering away from off-target selectivity traps.
    """
    if not on_pockets:
        raise ValueError("on_pockets must contain at least one (pocket_out, atom_mask) pair.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    from geneuclidean.models.hf_backbones import get_input_embeddings_tensor
    from geneuclidean.sampling.llm_sampler import (
        _build_initial_prompt_ids,
        _collect_stop_token_ids,
        _decode_generated_smiles,
        _expand_pocket_for_batch,
        _mask_safe_control_tokens,
    )

    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
    stop_token_ids = _collect_stop_token_ids(model)
    n_on = len(on_pockets)
    has_off = off_pocket is not None and abs(float(w_off)) > 1e-8
    n_streams = n_on + 1 + (1 if has_off else 0)  # [on_0..on_{K-1}, uncond, (off)]

    all_smiles: list[str] = []
    try:
        remaining = int(num_samples)
        while remaining > 0:
            cur_bs = min(remaining, max(1, int(batch_size)))
            ref_device = on_pockets[0][0].node_emb.device
            prompt_ids = _build_initial_prompt_ids(model, cur_bs, ref_device)
            prompt_embeds = get_input_embeddings_tensor(model.llm, prompt_ids)

            stream_inputs: list[Tensor] = []
            for p_out, p_mask in on_pockets:
                exp_p, exp_m = _expand_pocket_for_batch(p_out, p_mask, cur_bs)
                pref, _, _ = model.build_soft_prefix(
                    pocket_out=exp_p,
                    atom_mask=exp_m,
                    use_null_pocket=False,
                    latent_temp=latent_temp,
                    deterministic_z=(latent_temp <= 0.0),
                    generator=generator,
                    return_attn=False,
                )
                pref = pref.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                stream_inputs.append(torch.cat([pref, prompt_embeds], dim=1))

            # Unconditional stream (null pocket)
            null_pref, _, _ = model.build_soft_prefix(
                pocket_out=on_pockets[0][0],
                atom_mask=on_pockets[0][1],
                use_null_pocket=True,
                batch_size=cur_bs,
            )
            null_pref = null_pref.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            stream_inputs.append(torch.cat([null_pref, prompt_embeds], dim=1))

            # Optional off-target repulsion stream
            if has_off and off_pocket is not None:
                exp_off, exp_off_m = _expand_pocket_for_batch(off_pocket[0], off_pocket[1], cur_bs)
                off_pref, _, _ = model.build_soft_prefix(
                    pocket_out=exp_off,
                    atom_mask=exp_off_m,
                    use_null_pocket=False,
                    latent_temp=latent_temp,
                    deterministic_z=(latent_temp <= 0.0),
                    generator=generator,
                    return_attn=False,
                )
                off_pref = off_pref.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                stream_inputs.append(torch.cat([off_pref, prompt_embeds], dim=1))

            prefill_inputs = torch.cat(stream_inputs, dim=0)
            llm_device = prefill_inputs.device
            total_len = int(prefill_inputs.shape[1])
            prefill_mask = torch.ones(
                (n_streams * cur_bs, total_len), dtype=torch.long, device=llm_device
            )

            outputs = model.llm(
                inputs_embeds=prefill_inputs,
                attention_mask=prefill_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            step_logits = outputs.logits[:, -1, :].float()

            finished = torch.zeros(cur_bs, dtype=torch.bool, device=llm_device)
            generated_cols: list[Tensor] = []
            running_attn_mask = prefill_mask

            for _ in range(int(max_new_tokens)):
                chunks = step_logits.split(cur_bs, dim=0)
                on_chunks = chunks[:n_on]
                uncond_chunk = chunks[n_on]
                off_chunk = chunks[n_on + 1] if has_off else None

                logits = combine_contrastive_ensemble_logits(
                    on_chunks,
                    uncond_chunk,
                    on_weights=on_weights,
                    w_on=w_on,
                    off_logits=off_chunk,
                    w_off=w_off if has_off else 0.0,
                )
                if repetition_penalty > 1.0 and generated_cols:
                    history = torch.stack(generated_cols, dim=1)
                    logits = apply_repetition_penalty(
                        logits,
                        history,
                        penalty=repetition_penalty,
                        ignore_ids=(
                            int(model.pad_token_id),
                            int(model.bos_token_id),
                            int(model.eos_token_id),
                        ),
                    )
                logits = _mask_safe_control_tokens(
                    logits,
                    pad_token_id=int(model.pad_token_id),
                    bos_token_id=int(model.bos_token_id),
                    eos_token_id=int(model.eos_token_id),
                )
                next_tok = select_next_tokens(
                    logits,
                    strategy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    generator=generator,
                )
                eos_fill = torch.full_like(next_tok, int(model.eos_token_id))
                next_tok = torch.where(finished, eos_fill, next_tok)
                generated_cols.append(next_tok)

                is_stop = torch.zeros_like(finished)
                for sid in stop_token_ids:
                    is_stop |= next_tok == sid
                finished = finished | is_stop
                if bool(finished.all()):
                    break

                step_ids = next_tok.unsqueeze(1).repeat(n_streams, 1)
                running_attn_mask = torch.cat(
                    [
                        running_attn_mask,
                        torch.ones((n_streams * cur_bs, 1), dtype=torch.long, device=llm_device),
                    ],
                    dim=1,
                )
                step_out = model.llm(
                    input_ids=step_ids,
                    attention_mask=running_attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = step_out.past_key_values
                step_logits = step_out.logits[:, -1, :].float()

            gen_rows = torch.stack(generated_cols, dim=1).detach().cpu().tolist()
            all_smiles.extend(_decode_generated_smiles(model, gen_rows, stop_token_ids))
            remaining -= cur_bs
    finally:
        if was_training:
            model.train()

    return all_smiles


__all__ = [
    "PocketPMIScore",
    "combine_contrastive_ensemble_logits",
    "sample_contrastive_ensemble_llm",
    "score_pocket_pmi",
]
