"""Vectorized autoregressive sampling strategies, CFG, CVAE sampling, and attention extraction.

Implements mathematically exact, single-softmax decoding strategies with zero legacy string
truncation:
- Greedy decoding
- Temperature sampling (direct ``softmax(logits / T)`` — fixes 2021 double-softmax bug)
- Top-k sampling
- Top-p (Nucleus) sampling
- Min-p sampling (Nguyen et al., ICLR 2025 Oral — flagship default)
- Locally Typical sampling (Meister et al., TACL 2023)
- Autoregressive Classifier-Free Guidance (CFG) via batched ``[cond; uncond]`` forward
- Stochastic CVAE prior sampling and TamGen-style hit-to-lead posterior refinement
- Diverse Beam Search (Vijayakumar et al., AAAI 2018) with length & Hamming diversity penalties
- Optional CTRL repetition penalty and optional SMILES syntax guardrails (off by default)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from geneuclidean.config import SamplingConfig
from geneuclidean.models.cvae import PocketCVAE
from geneuclidean.registry import (
    SAMPLING_STRATEGY_REGISTRY,
    list_sampling_strategies,
    register_sampling_strategy,
)

if TYPE_CHECKING:
    from geneuclidean.models.encoders.rbf import PocketEncoderOutput


_BOND_TOKENS = frozenset({"=", "#", "/", "\\", ":"})


@dataclass
class SamplingResult:
    """Structured output from autoregressive sampling."""

    smiles: list[str]
    token_ids: list[list[int]]
    tokens: list[list[str]]
    scores: list[float] | None = None
    attn_maps: list[Tensor | None] | None = None


def _get_id_to_token(tokenizer: Any) -> dict[int, str]:
    """Extract an ``id -> token`` mapping from a tokenizer if exposed."""
    for attr in ("id_to_token", "itos", "idx_to_token"):
        mapping = getattr(tokenizer, attr, None)
        if isinstance(mapping, Mapping):
            return {int(k): str(v) for k, v in mapping.items()}
        if isinstance(mapping, (list, tuple)):
            return {idx: str(tok) for idx, tok in enumerate(mapping)}
    vocab = getattr(tokenizer, "token_to_id", None) or getattr(tokenizer, "stoi", None)
    if isinstance(vocab, Mapping):
        return {int(v): str(k) for k, v in vocab.items()}
    return {}


def _extract_content_positions(
    token_ids: Sequence[int],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    """Return 0-based positions into ``token_ids[1:]`` for emitted non-special tokens."""
    positions: list[int] = []
    for pos, raw_tid in enumerate(token_ids[1:]):
        tid = int(raw_tid)
        if tid in (eos_id, pad_id):
            break
        if tid == bos_id:
            continue
        positions.append(pos)
    return positions


def _ids_to_tokens(token_ids: Sequence[int], tokenizer: Any) -> list[str]:
    """Convert generated token IDs (excluding BOS/PAD/EOS) into token strings."""
    pad_id = int(getattr(tokenizer, "PAD_ID", 0))
    bos_id = int(getattr(tokenizer, "BOS_ID", 1))
    eos_id = int(getattr(tokenizer, "EOS_ID", 2))
    id_to_tok = _get_id_to_token(tokenizer)
    positions = _extract_content_positions(token_ids, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    return [id_to_tok.get(int(token_ids[pos + 1]), str(token_ids[pos + 1])) for pos in positions]


def _mask_control_tokens(logits: Tensor, pad_id: int, bos_id: int) -> Tensor:
    """Prevent emitting structural control tokens (PAD/BOS) when other tokens are finite."""
    vocab_size = logits.size(-1)
    masked = logits.clone()
    for tid in (pad_id, bos_id):
        if 0 <= tid < vocab_size:
            masked[:, tid] = float("-inf")
    has_finite = torch.isfinite(masked).any(dim=-1, keepdim=True)
    return torch.where(has_finite, masked, logits)


def apply_repetition_penalty(
    logits: Tensor,
    generated_ids: Tensor,
    penalty: float = 1.0,
    ignore_ids: tuple[int, ...] = (0, 1, 2),
) -> Tensor:
    """Apply Keskar et al. (CTRL) repetition penalty to previously generated tokens.

    For every token previously emitted in sequence ``b`` (excluding ``ignore_ids``),
    positive logits are divided by ``penalty`` and negative logits are multiplied by
    ``penalty``.

    Args:
        logits: Next-token logits of shape ``[B, V]``.
        generated_ids: Previously generated token IDs of shape ``[B, T]``.
        penalty: Repetition penalty factor ``>= 1.0`` (``1.0`` is a no-op).
        ignore_ids: Special token IDs exempt from repetition penalization.

    Returns:
        Penalized logits tensor of shape ``[B, V]``.
    """
    if penalty <= 1.0 or generated_ids.numel() == 0:
        return logits

    vocab_size = logits.size(-1)
    seen = torch.zeros_like(logits, dtype=torch.bool)
    clamped_ids = generated_ids.clamp(min=0, max=vocab_size - 1)
    seen.scatter_(1, clamped_ids, True)
    for special_id in ignore_ids:
        if 0 <= special_id < vocab_size:
            seen[:, special_id] = False

    penalized = torch.where(logits > 0.0, logits / float(penalty), logits * float(penalty))
    return torch.where(seen, penalized, logits)


def apply_smiles_syntax_guardrails(
    logits: Tensor,
    generated_ids: Tensor,
    id_to_token: Mapping[int, str],
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
) -> Tensor:
    """Optionally mask syntactically impossible SMILES continuations (off by default).

    Tracks parenthesis balance, open ring-closure digits, and illegal consecutive bond
    tokens. If a sequence has no remaining unmasked tokens, leaves its logits unchanged
    to avoid NaN probabilities.

    Args:
        logits: Next-token logits of shape ``[B, V]``.
        generated_ids: Generated token IDs so far of shape ``[B, T]``.
        id_to_token: Mapping from token ID to token string.
        pad_id: PAD token ID.
        bos_id: BOS token ID.
        eos_id: EOS token ID.

    Returns:
        Guarded logits of shape ``[B, V]``.
    """
    if not id_to_token:
        return logits

    token_to_id = {tok: idx for idx, tok in id_to_token.items()}
    close_paren_id = token_to_id.get(")")
    bond_ids = [idx for tok, idx in token_to_id.items() if tok in _BOND_TOKENS]
    ring_ids = [
        idx
        for tok, idx in token_to_id.items()
        if (len(tok) == 1 and tok.isdigit()) or (tok.startswith("%") and tok[1:].isdigit())
    ]

    masked = logits.clone()
    vocab_size = logits.size(-1)
    neg_inf = float("-inf")

    for row_idx in range(generated_ids.size(0)):
        seq = generated_ids[row_idx].tolist()
        content_tokens = [
            id_to_token.get(int(t), "") for t in seq if int(t) not in (pad_id, bos_id, eos_id)
        ]

        paren_depth = 0
        open_rings: set[str] = set()
        for tok in content_tokens:
            if tok == "(":
                paren_depth += 1
            elif tok == ")":
                paren_depth = max(0, paren_depth - 1)
            elif (len(tok) == 1 and tok.isdigit()) or (tok.startswith("%") and tok[1:].isdigit()):
                if tok in open_rings:
                    open_rings.remove(tok)
                else:
                    open_rings.add(tok)

        row_mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
        if 0 <= pad_id < vocab_size:
            row_mask[pad_id] = True
        if 0 <= bos_id < vocab_size:
            row_mask[bos_id] = True

        # Prevent unmatched ')' and premature EOS while branches or rings are open
        if paren_depth == 0 and close_paren_id is not None and close_paren_id < vocab_size:
            row_mask[close_paren_id] = True
        if (paren_depth > 0 or len(open_rings) > 0 or len(content_tokens) == 0) and (
            0 <= eos_id < vocab_size
        ):
            row_mask[eos_id] = True

        # Prevent leading bond/ring digits or illegal double bonds / empty branches
        last_tok = content_tokens[-1] if content_tokens else ""
        if len(content_tokens) == 0:
            for idx in bond_ids:
                if idx < vocab_size:
                    row_mask[idx] = True
            for idx in ring_ids:
                if idx < vocab_size:
                    row_mask[idx] = True
        elif last_tok in _BOND_TOKENS or last_tok == "(":
            for idx in bond_ids:
                if idx < vocab_size:
                    row_mask[idx] = True
            if last_tok == "(" and close_paren_id is not None and close_paren_id < vocab_size:
                row_mask[close_paren_id] = True

        # Apply only if at least one finite candidate remains
        candidate = masked[row_idx].masked_fill(row_mask, neg_inf)
        if bool(torch.isfinite(candidate).any().item()):
            masked[row_idx] = candidate

    return masked


def compute_temperature_probs(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Compute exact temperature-scaled probabilities ``softmax(logits / T)``."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive, got {temperature}")
    return F.softmax(logits / float(temperature), dim=-1)


def filter_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Mask all logits outside the top-``k`` entries to ``-inf``."""
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p_probs(probs: Tensor, top_p: float) -> Tensor:
    """Apply Nucleus (top-p) truncation on a probability distribution and renormalize."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    shifted_cum = cum_probs - sorted_probs
    remove_sorted = shifted_cum >= float(top_p)
    remove_sorted[..., 0] = False

    kept_sorted = sorted_probs.masked_fill(remove_sorted, 0.0)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, kept_sorted)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def filter_min_p_probs(probs: Tensor, min_p: float) -> Tensor:
    """Apply Min-p truncation (Nguyen et al., ICLR 2025 Oral): keep ``p >= min_p * max(p)``."""
    if min_p <= 0.0:
        return probs
    p_max = probs.max(dim=-1, keepdim=True).values
    threshold = float(min_p) * p_max
    keep = probs >= threshold
    argmax_idx = probs.argmax(dim=-1, keepdim=True)
    keep.scatter_(-1, argmax_idx, True)
    filtered = probs * keep.to(dtype=probs.dtype)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def filter_typical_probs(
    logits: Tensor,
    temperature: float = 1.0,
    mass: float = 0.92,
) -> Tensor:
    """Apply Locally Typical Sampling (Meister et al., TACL 2023).

    Ranks tokens by ``| -log p(x) - H(p) |`` (absolute deviation of token surprisal from
    conditional entropy) and keeps the smallest prefix covering cumulative mass ``mass``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive, got {temperature}")
    scaled = logits / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
    probs = log_probs.exp()

    # Safe entropy computation avoiding 0 * (-inf) = NaN
    neg_log_probs = -log_probs
    entropy_terms = torch.where(probs > 0.0, probs * neg_log_probs, torch.zeros_like(probs))
    entropy = entropy_terms.sum(dim=-1, keepdim=True)
    deviation = (neg_log_probs - entropy).abs()
    deviation = torch.where(torch.isfinite(deviation), deviation, torch.full_like(deviation, 1e9))

    _, sorted_idx = torch.sort(deviation, dim=-1, descending=False)
    sorted_probs = probs.gather(-1, sorted_idx)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    shifted_cum = cum_probs - sorted_probs
    remove_sorted = shifted_cum >= float(mass)
    remove_sorted[..., 0] = False

    kept_sorted = sorted_probs.masked_fill(remove_sorted, 0.0)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, kept_sorted)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def combine_cfg_logits(
    logits_cond: Tensor,
    logits_uncond: Tensor,
    cfg_scale: float,
) -> Tensor:
    """Apply Classifier-Free Guidance: ``logits_uncond + w * (logits_cond - logits_uncond)``."""
    return logits_uncond + float(cfg_scale) * (logits_cond - logits_uncond)


def _strategy_greedy(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    del temperature, top_k, top_p, min_p
    argmax_idx = logits.argmax(dim=-1, keepdim=True)
    probs = torch.zeros_like(logits)
    probs.scatter_(-1, argmax_idx, 1.0)
    return probs


def _strategy_temperature(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    del top_k, top_p, min_p
    return compute_temperature_probs(logits, temperature=temperature)


def _strategy_top_k(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    del top_p, min_p
    filtered_logits = filter_top_k(logits, top_k=top_k)
    return compute_temperature_probs(filtered_logits, temperature=temperature)


def _strategy_top_p(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    del top_k, min_p
    probs = compute_temperature_probs(logits, temperature=temperature)
    return filter_top_p_probs(probs, top_p=top_p)


def _strategy_min_p(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    del top_k, top_p
    probs = compute_temperature_probs(logits, temperature=temperature)
    return filter_min_p_probs(probs, min_p=min_p)


def _strategy_typical(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    del top_k, min_p
    return filter_typical_probs(logits, temperature=temperature, mass=top_p)


SAMPLING_STRATEGY_REGISTRY.register("greedy", _strategy_greedy, override=True)
SAMPLING_STRATEGY_REGISTRY.register("temperature", _strategy_temperature, override=True)
SAMPLING_STRATEGY_REGISTRY.register("top_k", _strategy_top_k, override=True)
SAMPLING_STRATEGY_REGISTRY.register("top_p", _strategy_top_p, override=True)
SAMPLING_STRATEGY_REGISTRY.register("min_p", _strategy_min_p, override=True)
SAMPLING_STRATEGY_REGISTRY.register("typical", _strategy_typical, override=True)


def compute_strategy_probs(
    logits: Tensor,
    strategy: str,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
) -> Tensor:
    """Compute exact next-token sampling probabilities for any built-in or custom strategy.

    Dispatches through :data:`~geneuclidean.registry.SAMPLING_STRATEGY_REGISTRY` so users
    can register custom probability shapers via :func:`geneuclidean.register_sampling_strategy`.
    """
    if strategy not in SAMPLING_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unsupported stochastic decoding strategy: {strategy!r}. "
            f"Registered: {list_sampling_strategies()}"
        )
    fn = SAMPLING_STRATEGY_REGISTRY.get(strategy)
    probs: Tensor = fn(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    return probs


def select_next_tokens(
    logits: Tensor,
    strategy: str,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.92,
    min_p: float = 0.08,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample next token IDs of shape ``[B]`` from ``logits[B, V]``."""
    if strategy == "greedy":
        return logits.argmax(dim=-1)

    probs = compute_strategy_probs(
        logits,
        strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    # Handle generator device compatibility cleanly across CPU / MPS
    if generator is not None and generator.device != probs.device:
        sampled = torch.multinomial(probs.to(generator.device), num_samples=1, generator=generator)
        return sampled.squeeze(-1).to(device=probs.device)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _pad_and_concat_memories(
    mem_a: Tensor,
    mask_a: Tensor,
    mem_b: Tensor,
    mask_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pad two memories to equal sequence length along dim=1 and stack along batch dim=0."""
    m_a = mem_a.size(1)
    m_b = mem_b.size(1)
    max_m = max(m_a, m_b)

    if m_a < max_m:
        pad_len = max_m - m_a
        mem_a = F.pad(mem_a, (0, 0, 0, pad_len), value=0.0)
        mask_a = F.pad(mask_a, (0, pad_len), value=False)
    if m_b < max_m:
        pad_len = max_m - m_b
        mem_b = F.pad(mem_b, (0, 0, 0, pad_len), value=0.0)
        mask_b = F.pad(mask_b, (0, pad_len), value=False)

    return torch.cat([mem_a, mem_b], dim=0), torch.cat([mask_a, mask_b], dim=0)


def _extract_z_tensor(z_out: Any) -> Tensor:
    """Unpack ``z_tokens[B, K, D]`` from CVAE output tensor, tuple, or dataclass."""
    if isinstance(z_out, Tensor):
        return z_out
    if isinstance(z_out, tuple) and isinstance(z_out[0], Tensor):
        return z_out[0]
    if hasattr(z_out, "z_tokens") and isinstance(z_out.z_tokens, Tensor):
        return z_out.z_tokens
    raise TypeError(f"Unsupported CVAE return type: {type(z_out).__name__}")


def _sample_cvae_tokens(
    model: nn.Module,
    pooled_emb: Tensor,
    num_samples: int,
    latent_temp: float = 1.0,
    seed_ligand_fp: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor | None:
    """Sample per-molecule CVAE prior or posterior refinement tokens ``[num_samples, K, D]``."""
    cvae = getattr(model, "cvae", None)
    if cvae is None:
        return None

    pooled_exp = (
        pooled_emb.expand(num_samples, -1).contiguous() if pooled_emb.size(0) == 1 else pooled_emb
    )

    # External CVAE implementations can still use the original sampling signature.
    rng_kwargs = {"generator": generator} if isinstance(cvae, PocketCVAE) else {}

    if seed_ligand_fp is not None:
        fp = seed_ligand_fp.to(device=pooled_exp.device, dtype=pooled_exp.dtype)
        if fp.ndim == 1:
            fp = fp.unsqueeze(0)
        if fp.size(0) == 1 and num_samples > 1:
            fp = fp.expand(num_samples, -1).contiguous()

        if hasattr(cvae, "refine"):
            try:
                return _extract_z_tensor(
                    cvae.refine(pooled_exp, fp, latent_temp=latent_temp, **rng_kwargs)
                )
            except (RuntimeError, ValueError):
                ligand_enc = getattr(model, "ligand_encoder", None)
                if ligand_enc is None and hasattr(model, "heads"):
                    ligand_enc = getattr(model.heads, "encode_ligand", None) or getattr(
                        model.heads, "ligand_encoder", None
                    )
                if ligand_enc is not None:
                    lig_emb = ligand_enc(fp)
                    return _extract_z_tensor(
                        cvae.refine(pooled_exp, lig_emb, latent_temp=latent_temp, **rng_kwargs)
                    )
                raise
        raise AttributeError("Model CVAE does not implement .refine(...) for hit-to-lead seeding.")

    if hasattr(cvae, "sample_prior"):
        return _extract_z_tensor(
            cvae.sample_prior(pooled_exp, latent_temp=latent_temp, **rng_kwargs)
        )
    return _extract_z_tensor(
        cvae(pooled_exp, ligand_emb=None, latent_temp=latent_temp, **rng_kwargs)
    )


def _prepare_pocket_tensors(
    pocket_batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize single-pocket ``(coords[1,N,3], node_feats[1,N,F], atom_mask[1,N])``."""
    coords = pocket_batch["coords"].to(device=device, dtype=torch.float32)
    node_feats = pocket_batch["node_feats"].to(device=device, dtype=torch.float32)
    atom_mask = pocket_batch["atom_mask"].to(device=device, dtype=torch.bool)

    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    if node_feats.ndim == 2:
        node_feats = node_feats.unsqueeze(0)
    if atom_mask.ndim == 1:
        atom_mask = atom_mask.unsqueeze(0)

    # Use the first pocket if a larger batch was passed
    if coords.size(0) > 1:
        coords = coords[:1]
        node_feats = node_feats[:1]
        atom_mask = atom_mask[:1]

    return coords, node_feats, atom_mask


def _repeat_pocket_out(pocket_out: Any, num_samples: int) -> Any:
    """Repeat a single-pocket ``PocketEncoderOutput`` to batch size ``num_samples``."""
    if num_samples == 1:
        return pocket_out
    node_emb = pocket_out.node_emb.expand(num_samples, -1, -1).contiguous()
    pooled_emb = pocket_out.pooled_emb.expand(num_samples, -1).contiguous()
    vec_emb = (
        pocket_out.vec_emb.expand(num_samples, -1, -1, -1).contiguous()
        if pocket_out.vec_emb is not None
        else None
    )
    cls = type(pocket_out)
    return cls(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=vec_emb)


def _decoder_step_logits(
    decoder: nn.Module,
    input_ids: Tensor,
    mem_cond: Tensor,
    mask_cond: Tensor,
    mem_uncond: Tensor | None,
    mask_uncond: Tensor | None,
    cfg_scale: float,
) -> Tensor:
    """Run one autoregressive decoder step (with batched ``[cond; uncond]`` pass if CFG active)."""
    use_cfg = mem_uncond is not None and mask_uncond is not None and abs(cfg_scale - 1.0) > 1e-6
    if not use_cfg:
        dec_out = decoder(input_ids, memory=mem_cond, memory_mask=mask_cond, return_attn=False)
        logits: Tensor = dec_out[0] if isinstance(dec_out, tuple) else dec_out
        return logits[:, -1, :]

    assert mem_uncond is not None and mask_uncond is not None
    if abs(cfg_scale) <= 1e-9:
        dec_out = decoder(input_ids, memory=mem_uncond, memory_mask=mask_uncond, return_attn=False)
        logits_u: Tensor = dec_out[0] if isinstance(dec_out, tuple) else dec_out
        return logits_u[:, -1, :]

    bsz = input_ids.size(0)
    ids_both = torch.cat([input_ids, input_ids], dim=0)
    mem_both, mask_both = _pad_and_concat_memories(mem_cond, mask_cond, mem_uncond, mask_uncond)
    dec_out = decoder(ids_both, memory=mem_both, memory_mask=mask_both, return_attn=False)
    logits_both: Tensor = dec_out[0] if isinstance(dec_out, tuple) else dec_out
    last_both = logits_both[:, -1, :]
    logits_cond, logits_uncond = last_both[:bsz], last_both[bsz:]
    return combine_cfg_logits(logits_cond, logits_uncond, cfg_scale=cfg_scale)


def diverse_beam_search(
    model: nn.Module,
    mem_cond: Tensor,
    mask_cond: Tensor,
    mem_uncond: Tensor | None,
    mask_uncond: Tensor | None,
    tokenizer: Any,
    sampling_cfg: SamplingConfig,
    length_penalty_alpha: float = 0.6,
) -> SamplingResult:
    """Diverse Beam Search (Vijayakumar et al., AAAI 2018) with Hamming diversity & length penalty.

    Partitions ``beam_width`` beams into ``num_beam_groups`` groups and applies a Hamming
    diversity penalty against tokens chosen by earlier groups at each decoding step, plus
    Wu et al. length normalization ``((5 + L) / 6)^alpha``.

    Args:
        model: ``GenEuclideanModel`` exposing ``.decoder``.
        mem_cond: Single-pocket conditional memory ``[1, M_c, D]``.
        mask_cond: Single-pocket conditional memory mask ``[1, M_c]``.
        mem_uncond: Optional unconditional null-pocket memory ``[1, M_u, D]`` for CFG.
        mask_uncond: Optional unconditional memory mask ``[1, M_u]``.
        tokenizer: ``MolecularTokenizer``.
        sampling_cfg: ``SamplingConfig`` with ``beam_width``, ``num_beam_groups``,
            ``diversity_penalty``, ``max_len``, ``cfg_scale``, ``temperature``,
            ``repetition_penalty``, and ``syntax_guardrails``.
        length_penalty_alpha: Exponent ``alpha`` in ``((5 + L) / 6)^alpha``.

    Returns:
        ``SamplingResult`` containing ranked diverse SMILES, token IDs, tokens, and scores.
    """
    device = mem_cond.device
    pad_id = int(getattr(tokenizer, "PAD_ID", 0))
    bos_id = int(getattr(tokenizer, "BOS_ID", 1))
    eos_id = int(getattr(tokenizer, "EOS_ID", 2))
    use_smiles_guardrails = (
        sampling_cfg.syntax_guardrails
        and getattr(tokenizer, "representation", "smiles") == "smiles"
    )
    id_to_token = _get_id_to_token(tokenizer) if use_smiles_guardrails else {}

    beam_width = sampling_cfg.beam_width
    num_groups = sampling_cfg.num_beam_groups
    if beam_width % num_groups != 0:
        raise ValueError(
            f"beam_width ({beam_width}) must be divisible by num_beam_groups ({num_groups})"
        )
    group_size = beam_width // num_groups
    diversity_penalty = float(sampling_cfg.diversity_penalty)
    max_len = sampling_cfg.max_len
    decoder: nn.Module = model.decoder  # type: ignore[assignment]

    # Expand single-pocket memory to group_size once
    g_mem_cond = mem_cond[:1].expand(group_size, -1, -1).contiguous()
    g_mask_cond = mask_cond[:1].expand(group_size, -1).contiguous()
    g_mem_uncond = (
        mem_uncond[:1].expand(group_size, -1, -1).contiguous() if mem_uncond is not None else None
    )
    g_mask_uncond = (
        mask_uncond[:1].expand(group_size, -1).contiguous() if mask_uncond is not None else None
    )

    # State per group: track both search scores and true unpenalized log-probs for fair final rank
    group_seqs: list[list[list[int]]] = [
        [[bos_id] for _ in range(group_size)] for _ in range(num_groups)
    ]
    group_scores: list[list[float]] = [[0.0] + [-1e9] * (group_size - 1) for _ in range(num_groups)]
    group_raw_scores: list[list[float]] = [
        [0.0] + [-1e9] * (group_size - 1) for _ in range(num_groups)
    ]
    group_finished: list[list[bool]] = [
        [False for _ in range(group_size)] for _ in range(num_groups)
    ]

    for step in range(max_len):
        if all(all(flags) for flags in group_finished):
            break

        prev_group_token_counts: Tensor | None = None

        for g in range(num_groups):
            seq_tensor = torch.tensor(group_seqs[g], dtype=torch.long, device=device)
            logits = _decoder_step_logits(
                decoder,
                seq_tensor,
                g_mem_cond,
                g_mask_cond,
                g_mem_uncond,
                g_mask_uncond,
                cfg_scale=sampling_cfg.cfg_scale,
            )  # [group_size, V]

            if sampling_cfg.repetition_penalty > 1.0:
                logits = apply_repetition_penalty(
                    logits,
                    seq_tensor,
                    penalty=sampling_cfg.repetition_penalty,
                    ignore_ids=(pad_id, bos_id, eos_id),
                )
            if sampling_cfg.syntax_guardrails and id_to_token:
                logits = apply_smiles_syntax_guardrails(
                    logits,
                    seq_tensor,
                    id_to_token=id_to_token,
                    pad_id=pad_id,
                    bos_id=bos_id,
                    eos_id=eos_id,
                )
            logits = _mask_control_tokens(logits, pad_id=pad_id, bos_id=bos_id)

            vocab_size = logits.size(-1)
            log_probs = F.log_softmax(logits / float(sampling_cfg.temperature), dim=-1)
            if prev_group_token_counts is None:
                prev_group_token_counts = torch.zeros(
                    vocab_size, dtype=log_probs.dtype, device=device
                )

            # Build candidate score table [group_size, V]
            prev_scores_t = torch.tensor(group_scores[g], dtype=log_probs.dtype, device=device)
            cand_scores = prev_scores_t.unsqueeze(1) + log_probs

            # Apply Hamming diversity penalty against tokens selected by groups 0..g-1 at step
            if g > 0 and diversity_penalty > 0.0:
                cand_scores = cand_scores - diversity_penalty * prev_group_token_counts.unsqueeze(0)

            # Finished beams can only emit PAD with zero incremental cost
            for b_idx, is_done in enumerate(group_finished[g]):
                if is_done:
                    cand_scores[b_idx].fill_(-1e9)
                    cand_scores[b_idx, pad_id] = prev_scores_t[b_idx]

            # At step 0, only expand beam 0 so duplicates aren't cloned before branching
            if step == 0 and group_size > 1:
                cand_scores[1:].fill_(-1e9)

            flat_scores = cand_scores.reshape(-1)
            top_vals, top_flat_idx = torch.topk(flat_scores, k=group_size, dim=-1)

            new_seqs: list[list[int]] = []
            new_scores: list[float] = []
            new_raw_scores: list[float] = []
            new_finished: list[bool] = []

            for k_idx in range(group_size):
                flat_i = int(top_flat_idx[k_idx].item())
                parent_b = flat_i // vocab_size
                tok_id = flat_i % vocab_size
                parent_done = group_finished[g][parent_b]

                if parent_done:
                    next_search_score = group_scores[g][parent_b]
                    next_raw_score = group_raw_scores[g][parent_b]
                    tok_id = pad_id
                    next_done = True
                else:
                    next_search_score = float(top_vals[k_idx].item())
                    next_raw_score = group_raw_scores[g][parent_b] + float(
                        log_probs[parent_b, tok_id].item()
                    )
                    next_done = tok_id == eos_id
                    prev_group_token_counts[tok_id] += 1.0

                new_seqs.append(group_seqs[g][parent_b] + [tok_id])
                new_scores.append(next_search_score)
                new_raw_scores.append(next_raw_score)
                new_finished.append(next_done)

            group_seqs[g] = new_seqs
            group_scores[g] = new_scores
            group_raw_scores[g] = new_raw_scores
            group_finished[g] = new_finished

    # Gather all hypotheses across groups and rank with Wu et al. length penalty on true log-probs
    candidates: list[tuple[float, list[int]]] = []
    for g in range(num_groups):
        for seq, raw_score in zip(group_seqs[g], group_raw_scores[g], strict=True):
            gen_len = sum(1 for t in seq[1:] if t != pad_id)
            lp = ((5.0 + float(max(1, gen_len))) / 6.0) ** float(length_penalty_alpha)
            candidates.append((raw_score / lp, seq))

    candidates.sort(key=lambda item: item[0], reverse=True)
    limit = min(len(candidates), sampling_cfg.num_samples)
    selected = candidates[:limit]

    token_ids = [seq for _, seq in selected]
    scores = [score for score, _ in selected]
    smiles = [str(tokenizer.decode(seq, to_smiles=True)) for seq in token_ids]
    tokens = [_ids_to_tokens(seq, tokenizer) for seq in token_ids]
    return SamplingResult(smiles=smiles, token_ids=token_ids, tokens=tokens, scores=scores)


@torch.no_grad()
def _extract_attention_maps(
    model: nn.Module,
    token_ids_list: list[list[int]],
    mem_cond: Tensor,
    mask_cond: Tensor,
    n_pocket_atoms: int,
    pad_id: int,
    eos_id: int,
) -> list[Tensor | None]:
    """Compute head-averaged pocket cross-attention maps aligned with generated tokens."""
    if not token_ids_list:
        return []

    device = mem_cond.device
    decoder: nn.Module = model.decoder  # type: ignore[assignment]
    bsz = len(token_ids_list)

    # Prefix inputs predicting each generated token (seq[:-1])
    prefixes = [seq[:-1] if len(seq) > 1 else seq for seq in token_ids_list]
    max_t = max(len(p) for p in prefixes)
    padded = torch.full((bsz, max_t), fill_value=pad_id, dtype=torch.long, device=device)
    for i, pref in enumerate(prefixes):
        padded[i, : len(pref)] = torch.tensor(pref, dtype=torch.long, device=device)

    mem_eval = (
        mem_cond[:1].expand(bsz, -1, -1).contiguous()
        if mem_cond.size(0) == 1 and bsz > 1
        else mem_cond[:bsz]
    )
    mask_eval = (
        mask_cond[:1].expand(bsz, -1).contiguous()
        if mask_cond.size(0) == 1 and bsz > 1
        else mask_cond[:bsz]
    )

    dec_out = decoder(padded, memory=mem_eval, memory_mask=mask_eval, return_attn=True)
    attn_weights = dec_out[1] if isinstance(dec_out, tuple) and len(dec_out) > 1 else None
    if attn_weights is None:
        return [None for _ in range(bsz)]

    # Average across attention heads if shape is [B, H, L, M]; accept [B, L, M] directly too
    attn_avg = attn_weights.mean(dim=1) if attn_weights.ndim == 4 else attn_weights
    pocket_slice = min(n_pocket_atoms, attn_avg.size(-1))

    maps: list[Tensor | None] = []
    bos_id = 1
    for i, seq in enumerate(token_ids_list):
        if len(seq) > 0:
            bos_id = int(seq[0])
        positions = [
            p
            for p in _extract_content_positions(seq, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
            if p < attn_avg.size(1)
        ]
        if positions:
            pos_tensor = torch.tensor(positions, dtype=torch.long, device=attn_avg.device)
            maps.append(attn_avg[i, pos_tensor, :pocket_slice].detach().cpu())
        else:
            maps.append(
                torch.empty((0, pocket_slice), dtype=attn_avg.dtype, device=torch.device("cpu"))
            )
    return maps


@torch.no_grad()
def sample_molecules_detailed(
    model: nn.Module,
    pocket_batch: Mapping[str, Any],
    tokenizer: Any,
    sampling_cfg: SamplingConfig,
    *,
    seed_ligand_fp: Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    return_attn: bool = False,
    length_penalty_alpha: float = 0.6,
    pocket_output: PocketEncoderOutput | None = None,
) -> SamplingResult:
    """Generate molecules conditioned on a 3D protein pocket with full diagnostics.

    Encodes the 3D pocket ONCE, optionally draws per-sample CVAE prior or posterior
    refinement latent codes, and runs fully vectorized batched decoding with early EOS
    stopping and optional Classifier-Free Guidance.

    Args:
        model: ``GenEuclideanModel`` instance.
        pocket_batch: Pocket tensors (``coords``, ``node_feats``, ``atom_mask``).
        tokenizer: ``MolecularTokenizer`` exposing ``PAD_ID``, ``BOS_ID``, ``EOS_ID``, ``decode``.
        sampling_cfg: ``SamplingConfig`` controlling decoding strategy and hyperparameters.
        seed_ligand_fp: Optional seed ligand fingerprint for TamGen-style CVAE refinement.
        generator: Optional ``torch.Generator`` for deterministic sampling.
        device: Optional execution device (defaults to first model parameter's device).
        pocket_output: Optional encoding of this pocket, reused for generation and scoring.
        return_attn: Whether to extract per-molecule cross-attention maps over pocket atoms.
        length_penalty_alpha: Length penalty exponent for ``diverse_beam``.

    Returns:
        ``SamplingResult`` with ``smiles``, ``token_ids``, ``tokens``, and optional ``attn_maps``.
    """
    model.eval()
    eff_device = device if device is not None else next(model.parameters(), torch.empty(0)).device
    pad_id = int(getattr(tokenizer, "PAD_ID", 0))
    bos_id = int(getattr(tokenizer, "BOS_ID", 1))
    eos_id = int(getattr(tokenizer, "EOS_ID", 2))
    use_smiles_guardrails = (
        sampling_cfg.syntax_guardrails
        and getattr(tokenizer, "representation", "smiles") == "smiles"
    )
    id_to_token = _get_id_to_token(tokenizer) if use_smiles_guardrails else {}

    # 1. Encode the 3D pocket ONCE
    coords, node_feats, atom_mask = _prepare_pocket_tensors(pocket_batch, eff_device)
    n_pocket_atoms = int(coords.size(1))
    pocket_out_single = pocket_output
    if pocket_out_single is None:
        pocket_out_single = model.encode_pocket(coords, node_feats, atom_mask)  # type: ignore[operator]

    # 2. Dispatch Diverse Beam Search
    if sampling_cfg.strategy == "diverse_beam":
        z_single = _sample_cvae_tokens(
            model,
            pocket_out_single.pooled_emb,
            num_samples=1,
            latent_temp=sampling_cfg.latent_temp,
            seed_ligand_fp=seed_ligand_fp,
            generator=generator,
        )
        mem_cond, mask_cond = model.build_memory(  # type: ignore[operator]
            pocket_out_single,
            atom_mask,
            z_tokens=z_single,
            use_null_pocket=False,
        )
        mem_uncond: Tensor | None = None
        mask_uncond: Tensor | None = None
        if abs(sampling_cfg.cfg_scale - 1.0) > 1e-6:
            mem_uncond, mask_uncond = model.build_memory(  # type: ignore[operator]
                pocket_out_single,
                atom_mask,
                z_tokens=None,
                use_null_pocket=True,
            )

        dbs_result = diverse_beam_search(
            model,
            mem_cond=mem_cond,
            mask_cond=mask_cond,
            mem_uncond=mem_uncond,
            mask_uncond=mask_uncond,
            tokenizer=tokenizer,
            sampling_cfg=sampling_cfg,
            length_penalty_alpha=length_penalty_alpha,
        )
        if return_attn:
            dbs_result.attn_maps = _extract_attention_maps(
                model,
                dbs_result.token_ids,
                mem_cond=mem_cond,
                mask_cond=mask_cond,
                n_pocket_atoms=n_pocket_atoms,
                pad_id=pad_id,
                eos_id=eos_id,
            )
        return dbs_result

    # 3. Parallel batched sampling for all stochastic & greedy strategies
    num_samples = sampling_cfg.num_samples
    pocket_out_batched = _repeat_pocket_out(pocket_out_single, num_samples)
    atom_mask_batched = atom_mask.expand(num_samples, -1).contiguous()

    z_tokens = _sample_cvae_tokens(
        model,
        pocket_out_single.pooled_emb,
        num_samples=num_samples,
        latent_temp=sampling_cfg.latent_temp,
        seed_ligand_fp=seed_ligand_fp,
        generator=generator,
    )
    mem_cond, mask_cond = model.build_memory(  # type: ignore[operator]
        pocket_out_batched,
        atom_mask_batched,
        z_tokens=z_tokens,
        use_null_pocket=False,
    )

    mem_uncond = None
    mask_uncond = None
    if abs(sampling_cfg.cfg_scale - 1.0) > 1e-6:
        mem_uncond, mask_uncond = model.build_memory(  # type: ignore[operator]
            pocket_out_batched,
            atom_mask_batched,
            z_tokens=None,
            use_null_pocket=True,
        )

    decoder: nn.Module = model.decoder  # type: ignore[assignment]
    generated = torch.full(
        (num_samples, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=eff_device,
    )
    finished = torch.zeros(num_samples, dtype=torch.bool, device=eff_device)

    for _ in range(sampling_cfg.max_len):
        logits = _decoder_step_logits(
            decoder,
            generated,
            mem_cond=mem_cond,
            mask_cond=mask_cond,
            mem_uncond=mem_uncond,
            mask_uncond=mask_uncond,
            cfg_scale=sampling_cfg.cfg_scale,
        )

        if sampling_cfg.repetition_penalty > 1.0:
            logits = apply_repetition_penalty(
                logits,
                generated,
                penalty=sampling_cfg.repetition_penalty,
                ignore_ids=(pad_id, bos_id, eos_id),
            )
        if sampling_cfg.syntax_guardrails and id_to_token:
            logits = apply_smiles_syntax_guardrails(
                logits,
                generated,
                id_to_token=id_to_token,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )
        logits = _mask_control_tokens(logits, pad_id=pad_id, bos_id=bos_id)

        next_tokens = select_next_tokens(
            logits,
            strategy=sampling_cfg.strategy,
            temperature=sampling_cfg.temperature,
            top_k=sampling_cfg.top_k,
            top_p=sampling_cfg.top_p,
            min_p=sampling_cfg.min_p,
            generator=generator,
        )

        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, pad_id),
            next_tokens,
        )
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        finished = finished | (next_tokens == eos_id)
        if bool(finished.all().item()):
            break

    token_ids_list = [row.tolist() for row in generated.detach().cpu()]
    smiles_list = [str(tokenizer.decode(ids, to_smiles=True)) for ids in token_ids_list]
    token_lists = [_ids_to_tokens(ids, tokenizer) for ids in token_ids_list]

    attn_maps: list[Tensor | None] | None = None
    if return_attn:
        attn_maps = _extract_attention_maps(
            model,
            token_ids_list,
            mem_cond=mem_cond,
            mask_cond=mask_cond,
            n_pocket_atoms=n_pocket_atoms,
            pad_id=pad_id,
            eos_id=eos_id,
        )

    return SamplingResult(
        smiles=smiles_list,
        token_ids=token_ids_list,
        tokens=token_lists,
        attn_maps=attn_maps,
    )


@torch.no_grad()
def sample_molecules(
    model: nn.Module,
    pocket_batch: Mapping[str, Any],
    tokenizer: Any,
    sampling_cfg: SamplingConfig,
    *,
    seed_ligand_fp: Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    length_penalty_alpha: float = 0.6,
) -> list[str]:
    """Sample SMILES strings conditioned on a 3D protein pocket."""
    result = sample_molecules_detailed(
        model=model,
        pocket_batch=pocket_batch,
        tokenizer=tokenizer,
        sampling_cfg=sampling_cfg,
        seed_ligand_fp=seed_ligand_fp,
        generator=generator,
        device=device,
        return_attn=False,
        length_penalty_alpha=length_penalty_alpha,
    )
    return result.smiles


@torch.no_grad()
def sample_with_attention(
    model: nn.Module,
    pocket_batch: Mapping[str, Any],
    tokenizer: Any,
    cfg: SamplingConfig,
    *,
    seed_ligand_fp: Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[list[str], list[list[str]], list[Tensor | None]]:
    """Sample molecules and return aligned 3D pocket cross-attention maps for visualization.

    Args:
        model: ``GenEuclideanModel`` instance.
        pocket_batch: Pocket dictionary (``coords``, ``node_feats``, ``atom_mask``).
        tokenizer: ``MolecularTokenizer``.
        cfg: ``SamplingConfig``.
        seed_ligand_fp: Optional seed ligand fingerprint for hit-to-lead CVAE refinement.
        generator: Optional ``torch.Generator``.
        device: Optional execution device.

    Returns:
        Tuple ``(smiles_list, token_lists, attn_maps)`` where each ``attn_maps[i]`` has
        shape ``[len(token_lists[i]), N_pocket]`` (or ``None`` if decoder lacks cross-attention).
    """
    result = sample_molecules_detailed(
        model=model,
        pocket_batch=pocket_batch,
        tokenizer=tokenizer,
        sampling_cfg=cfg,
        seed_ligand_fp=seed_ligand_fp,
        generator=generator,
        device=device,
        return_attn=True,
    )
    attn_maps: list[Tensor | None] = (
        result.attn_maps
        if result.attn_maps is not None
        else [None for _ in range(len(result.smiles))]
    )
    return result.smiles, result.tokens, attn_maps


__all__ = [
    "SAMPLING_STRATEGY_REGISTRY",
    "SamplingResult",
    "apply_repetition_penalty",
    "apply_smiles_syntax_guardrails",
    "combine_cfg_logits",
    "compute_strategy_probs",
    "compute_temperature_probs",
    "diverse_beam_search",
    "filter_min_p_probs",
    "filter_top_k",
    "filter_top_p_probs",
    "filter_typical_probs",
    "list_sampling_strategies",
    "register_sampling_strategy",
    "sample_molecules",
    "sample_molecules_detailed",
    "sample_with_attention",
    "select_next_tokens",
]
