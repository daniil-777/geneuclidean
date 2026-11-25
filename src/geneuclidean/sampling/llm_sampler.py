"""KV-cached vectorized batched CFG sampling for GenEuclidean v3 PocketConditionedLLM.

Supports both ``GP-MoLFormer-Uniq`` recurrent linear-attention tuple KV-caches and HuggingFace
``DynamicCache`` (GPT-2, Qwen2.5, SmolLM2, Qwen3) with:

- Single-shot prefix + prompt prefill followed by O(1)-token incremental decoding steps.
- Exact batched Autoregressive Classifier-Free Guidance (CFG):
  ``logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)``.
- Single-softmax Min-p, Top-p, Top-k, Temperature, Locally Typical, and Greedy decoding.
- Optional seed-ligand CVAE posterior refinement for hit-to-lead analog generation.
- Perceiver cross-attention extraction for Tier-3 ``<4.5 A`` pocket contact enrichment.
"""

from __future__ import annotations

import torch
from torch import Tensor

from geneuclidean.models.encoders import PocketEncoderOutput
from geneuclidean.models.hf_backbones import get_input_embeddings_tensor
from geneuclidean.models.llm_bridge import PocketConditionedLLM
from geneuclidean.sampling.decoder_strategies import (
    apply_repetition_penalty,
    combine_cfg_logits,
    select_next_tokens,
)


def _expand_pocket_for_batch(
    pocket_out: PocketEncoderOutput,
    atom_mask: Tensor,
    batch_size: int,
) -> tuple[PocketEncoderOutput, Tensor]:
    """Expand a single-pocket ``[1, ...]`` or matching ``[B, ...]`` encoding to ``batch_size``."""
    cur_b = int(pocket_out.node_emb.shape[0])
    if cur_b == batch_size:
        return pocket_out, atom_mask
    if cur_b != 1:
        raise ValueError(f"Expected pocket_out batch size 1 or {batch_size}, got {cur_b}.")
    expanded_out = PocketEncoderOutput(
        node_emb=pocket_out.node_emb.expand(batch_size, -1, -1).contiguous(),
        pooled_emb=pocket_out.pooled_emb.expand(batch_size, -1).contiguous(),
        vec_emb=(
            pocket_out.vec_emb.expand(batch_size, -1, -1, -1).contiguous()
            if pocket_out.vec_emb is not None
            else None
        ),
    )
    expanded_mask = atom_mask.expand(batch_size, -1).contiguous()
    return expanded_out, expanded_mask


def _build_initial_prompt_ids(
    model: PocketConditionedLLM,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Construct the initial token prompt ``[B, P0]`` fed immediately after the 3D soft prefix."""
    if model.is_smiles_native or not model.prompt_prefix:
        return torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)

    encoded: list[int] = list(model.tokenizer.encode(model.prompt_prefix, add_special_tokens=False))
    if not encoded:
        encoded = [model.bos_token_id]
    return (
        torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    )


def _collect_stop_token_ids(model: PocketConditionedLLM) -> set[int]:
    """Return token IDs that terminate SMILES generation (EOS plus newline for general LLMs)."""
    stop_ids: set[int] = {int(model.eos_token_id)}
    if not model.is_smiles_native:
        for text in ("\n", "\r\n", "\n\n"):
            try:
                ids = model.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                ids = []
            for tid in ids:
                decoded = model.tokenizer.decode([int(tid)], skip_special_tokens=False)
                if "\n" in decoded:
                    stop_ids.add(int(tid))
    return stop_ids


def _mask_safe_control_tokens(
    logits: Tensor,
    *,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> Tensor:
    """Mask PAD/BOS tokens unless they alias EOS (as in Qwen2.5 and SmolLM2)."""
    vocab_size = logits.size(-1)
    to_mask = [
        tid for tid in {pad_token_id, bos_token_id} if tid != eos_token_id and 0 <= tid < vocab_size
    ]
    if not to_mask:
        return logits
    masked = logits.clone()
    for tid in to_mask:
        masked[:, tid] = float("-inf")
    has_finite = torch.isfinite(masked).any(dim=-1, keepdim=True)
    return torch.where(has_finite, masked, logits)


def _decode_generated_smiles(
    model: PocketConditionedLLM,
    generated_rows: list[list[int]],
    stop_token_ids: set[int],
) -> list[str]:
    """Decode generated token ID rows into clean SMILES strings."""
    decoded_smiles: list[str] = []
    prompt_tag = model.prompt_prefix.strip()
    for row in generated_rows:
        trimmed: list[int] = []
        for tid in row:
            if tid in stop_token_ids:
                break
            trimmed.append(tid)
        text = str(model.tokenizer.decode(trimmed, skip_special_tokens=True))
        if model.is_smiles_native:
            smi = "".join(text.split())
        else:
            cleaned = text
            if prompt_tag and cleaned.lstrip().startswith(prompt_tag):
                cleaned = cleaned.lstrip()[len(prompt_tag) :]
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            first_line = lines[0] if lines else ""
            smi = first_line.split()[0] if first_line else ""
        decoded_smiles.append(smi)
    return decoded_smiles


@torch.no_grad()
def _sample_chunk_kv_cached(
    model: PocketConditionedLLM,
    pocket_out: PocketEncoderOutput,
    atom_mask: Tensor,
    *,
    chunk_size: int,
    max_new_tokens: int,
    strategy: str,
    temperature: float,
    min_p: float,
    top_p: float,
    top_k: int,
    cfg_scale: float,
    latent_temp: float,
    repetition_penalty: float,
    seed_ligand_fp: Tensor | None,
    generator: torch.Generator | None,
    return_attn: bool,
) -> tuple[list[str], Tensor | None]:
    """Run one KV-cached batched CFG sampling chunk of size ``chunk_size``."""
    device = pocket_out.node_emb.device
    exp_pocket, exp_mask = _expand_pocket_for_batch(pocket_out, atom_mask, chunk_size)

    exp_seed_fp: Tensor | None = None
    if seed_ligand_fp is not None:
        fp = seed_ligand_fp.to(device=device, dtype=torch.float32)
        if fp.ndim == 1:
            fp = fp.unsqueeze(0)
        if fp.shape[0] == 1 and chunk_size > 1:
            fp = fp.expand(chunk_size, -1).contiguous()
        elif fp.shape[0] != chunk_size:
            raise ValueError(
                f"Expected seed_ligand_fp batch size 1 or {chunk_size}, got {fp.shape[0]}."
            )
        exp_seed_fp = fp

    cond_prefix, attn_weights, _ = model.build_soft_prefix(
        pocket_out=exp_pocket,
        atom_mask=exp_mask,
        ligand_fp=exp_seed_fp,
        use_null_pocket=False,
        latent_temp=latent_temp,
        use_posterior=(exp_seed_fp is not None),
        deterministic_z=(latent_temp <= 0.0),
        return_attn=return_attn,
    )

    prompt_ids = _build_initial_prompt_ids(model, batch_size=chunk_size, device=device)
    prompt_embeds = get_input_embeddings_tensor(model.llm, prompt_ids)
    cond_prefix = cond_prefix.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    cond_inputs = torch.cat([cond_prefix, prompt_embeds], dim=1)

    use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
    if use_cfg:
        uncond_prefix, _, _ = model.build_soft_prefix(
            pocket_out=exp_pocket,
            atom_mask=exp_mask,
            use_null_pocket=True,
            batch_size=chunk_size,
        )
        uncond_prefix = uncond_prefix.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        uncond_inputs = torch.cat([uncond_prefix, prompt_embeds], dim=1)
        prefill_inputs = torch.cat([cond_inputs, uncond_inputs], dim=0)
    else:
        prefill_inputs = cond_inputs

    llm_device = prefill_inputs.device
    eff_batch = int(prefill_inputs.shape[0])
    total_len = int(prefill_inputs.shape[1])
    prefill_mask = torch.ones((eff_batch, total_len), dtype=torch.long, device=llm_device)

    # Step 0: single prefill of [prefix_embeds, prompt_tokens] -> caches all K/V states
    outputs = model.llm(
        inputs_embeds=prefill_inputs,
        attention_mask=prefill_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    step_logits: Tensor = outputs.logits[:, -1, :].float()

    stop_token_ids = _collect_stop_token_ids(model)
    stop_tensor = torch.tensor(sorted(stop_token_ids), dtype=torch.long, device=llm_device)
    ignore_rep_ids = tuple({model.pad_token_id, model.bos_token_id, model.eos_token_id})

    finished = torch.zeros(chunk_size, dtype=torch.bool, device=llm_device)
    generated_cols: list[Tensor] = []

    for step in range(max_new_tokens):
        if use_cfg:
            logits_cond, logits_uncond = step_logits.chunk(2, dim=0)
            next_logits = combine_cfg_logits(logits_cond, logits_uncond, cfg_scale)
        else:
            next_logits = step_logits

        next_logits = _mask_safe_control_tokens(
            next_logits,
            pad_token_id=model.pad_token_id,
            bos_token_id=model.bos_token_id,
            eos_token_id=model.eos_token_id,
        )
        if generated_cols and repetition_penalty > 1.0:
            history = torch.stack(generated_cols, dim=1)
            next_logits = apply_repetition_penalty(
                next_logits,
                history,
                penalty=repetition_penalty,
                ignore_ids=ignore_rep_ids,
            )

        next_tok = select_next_tokens(
            next_logits,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            generator=generator,
        )
        next_tok = torch.where(
            finished,
            torch.full_like(next_tok, model.eos_token_id),
            next_tok,
        )
        generated_cols.append(next_tok)

        hit_stop = (next_tok.unsqueeze(-1) == stop_tensor.unsqueeze(0)).any(dim=-1)
        finished = finished | hit_stop
        if bool(finished.all().item()) or step == max_new_tokens - 1:
            break

        # Incremental 1-token KV-cached decode step
        total_len += 1
        step_ids = (
            torch.cat([next_tok, next_tok], dim=0).unsqueeze(1)
            if use_cfg
            else next_tok.unsqueeze(1)
        )
        step_mask = torch.ones((eff_batch, total_len), dtype=torch.long, device=llm_device)
        outputs = model.llm(
            input_ids=step_ids,
            attention_mask=step_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        step_logits = outputs.logits[:, -1, :].float()

    generated_matrix = (
        torch.stack(generated_cols, dim=1).detach().cpu().tolist()
        if generated_cols
        else [[] for _ in range(chunk_size)]
    )
    smiles = _decode_generated_smiles(model, generated_matrix, stop_token_ids)
    return smiles, attn_weights


@torch.no_grad()
def sample_llm_with_attention(
    model: PocketConditionedLLM,
    pocket_out: PocketEncoderOutput,
    atom_mask: Tensor,
    num_samples: int = 32,
    max_new_tokens: int = 96,
    strategy: str = "min_p",
    temperature: float = 1.0,
    min_p: float = 0.08,
    top_p: float = 0.92,
    top_k: int = 30,
    cfg_scale: float = 1.6,
    latent_temp: float = 1.0,
    repetition_penalty: float = 1.05,
    seed_ligand_fp: Tensor | None = None,
    batch_size: int = 32,
    seed: int | None = None,
) -> tuple[list[str], Tensor]:
    """Generate SMILES with KV-cached CFG decoding and return Perceiver cross-attention maps.

    Args:
        model: ``PocketConditionedLLM`` instance (temporarily placed in eval mode).
        pocket_out: Pre-encoded 3D pocket representation (batch size ``1`` or ``num_samples``).
        atom_mask: Boolean pocket atom mask ``[1, N]`` or ``[num_samples, N]``.
        num_samples: Total number of molecules to generate.
        max_new_tokens: Maximum newly generated tokens per molecule.
        strategy: Decoding filter (``"min_p"``, ``"top_p"``, ``"top_k"``, ``"greedy"``,
            ``"temperature"``, ``"typical"``).
        temperature: Softmax sampling temperature ``> 0``.
        min_p: Min-p threshold (Nguyen et al., ICLR 2025 Oral).
        top_p: Nucleus probability threshold.
        top_k: Top-k cutoff.
        cfg_scale: Autoregressive Classifier-Free Guidance weight (``1.0`` disables CFG).
        latent_temp: CVAE latent sampling temperature (``0.0`` uses deterministic mean).
        repetition_penalty: CTRL repetition penalty ``>= 1.0``.
        seed_ligand_fp: Optional ECFP4 fingerprint ``[fp_dim]`` or ``[1, fp_dim]`` for
            posterior hit-to-lead refinement.
        batch_size: Maximum parallel sequences per KV-cached batch chunk.
        seed: Optional random seed for deterministic generation.

    Returns:
        Tuple ``(smiles_list, perceiver_attn_weights)`` where ``perceiver_attn_weights`` has shape
        ``[num_samples, H, Q, N]`` with exact ``0.0`` weight on masked pocket atoms.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    was_training = model.training
    model.eval()
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    all_smiles: list[str] = []
    all_attns: list[Tensor] = []
    pocket_batch = int(pocket_out.node_emb.shape[0])

    try:
        remaining = int(num_samples)
        offset = 0
        while remaining > 0:
            cur_bs = min(remaining, int(batch_size))
            if pocket_batch == num_samples:
                chunk_pocket = PocketEncoderOutput(
                    node_emb=pocket_out.node_emb[offset : offset + cur_bs],
                    pooled_emb=pocket_out.pooled_emb[offset : offset + cur_bs],
                    vec_emb=(
                        pocket_out.vec_emb[offset : offset + cur_bs]
                        if pocket_out.vec_emb is not None
                        else None
                    ),
                )
                chunk_mask = atom_mask[offset : offset + cur_bs]
                chunk_fp = (
                    seed_ligand_fp[offset : offset + cur_bs]
                    if seed_ligand_fp is not None
                    and seed_ligand_fp.ndim == 2
                    and seed_ligand_fp.shape[0] == num_samples
                    else seed_ligand_fp
                )
            else:
                chunk_pocket = pocket_out
                chunk_mask = atom_mask
                chunk_fp = seed_ligand_fp

            chunk_smiles, chunk_attn = _sample_chunk_kv_cached(
                model=model,
                pocket_out=chunk_pocket,
                atom_mask=chunk_mask,
                chunk_size=cur_bs,
                max_new_tokens=max_new_tokens,
                strategy=strategy,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                top_k=top_k,
                cfg_scale=cfg_scale,
                latent_temp=latent_temp,
                repetition_penalty=repetition_penalty,
                seed_ligand_fp=chunk_fp,
                generator=generator,
                return_attn=True,
            )
            all_smiles.extend(chunk_smiles)
            assert chunk_attn is not None
            all_attns.append(chunk_attn.detach())
            remaining -= cur_bs
            offset += cur_bs
    finally:
        if was_training:
            model.train()

    return all_smiles, torch.cat(all_attns, dim=0)


@torch.no_grad()
def sample_llm_molecules(
    model: PocketConditionedLLM,
    pocket_out: PocketEncoderOutput,
    atom_mask: Tensor,
    num_samples: int = 32,
    max_new_tokens: int = 96,
    strategy: str = "min_p",
    temperature: float = 1.0,
    min_p: float = 0.08,
    top_p: float = 0.92,
    top_k: int = 30,
    cfg_scale: float = 1.6,
    latent_temp: float = 1.0,
    repetition_penalty: float = 1.05,
    seed_ligand_fp: Tensor | None = None,
    batch_size: int = 32,
    seed: int | None = None,
) -> list[str]:
    """Generate SMILES strings conditioned on a 3D pocket via KV-cached CFG decoding."""
    smiles, _ = sample_llm_with_attention(
        model=model,
        pocket_out=pocket_out,
        atom_mask=atom_mask,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        strategy=strategy,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=top_k,
        cfg_scale=cfg_scale,
        latent_temp=latent_temp,
        repetition_penalty=repetition_penalty,
        seed_ligand_fp=seed_ligand_fp,
        batch_size=batch_size,
        seed=seed,
    )
    return smiles


__all__ = [
    "sample_llm_molecules",
    "sample_llm_with_attention",
]
