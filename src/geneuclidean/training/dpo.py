"""Stage-3 MPO-guided Direct Preference Optimization (DPO) alignment."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from geneuclidean.config import ExperimentConfig, select_device
from geneuclidean.training.losses import compute_dpo_loss


@dataclass(frozen=True)
class PreferencePair:
    """Single pocket-conditioned (chosen, rejected) preference pair for DPO."""

    pocket_batch: Mapping[str, Any]
    chosen_smiles: str
    rejected_smiles: str
    chosen_score: float = 0.0
    rejected_score: float = 0.0


def build_preference_pairs_from_mpo(
    pocket_batch: Mapping[str, Any],
    smiles_list: Sequence[str],
    mpo_scores: Sequence[float],
    min_margin: float = 0.05,
    max_pairs: int = 8,
) -> list[PreferencePair]:
    """Construct top-vs-bottom preference pairs from MPO-scored candidate molecules.

    Args:
        pocket_batch: Single-pocket batch dictionary.
        smiles_list: Candidate SMILES generated for this pocket.
        mpo_scores: Multi-parameter optimization desirability scores aligned with ``smiles_list``.
        min_margin: Minimum MPO score margin required between chosen and rejected ligands.
        max_pairs: Maximum number of preference pairs to extract for this pocket.

    Returns:
        List of ``PreferencePair`` instances ordered by descending score margin.
    """
    if len(smiles_list) != len(mpo_scores):
        raise ValueError("smiles_list and mpo_scores must have equal length.")
    if len(smiles_list) < 2 or max_pairs <= 0:
        return []

    ranked = sorted(
        zip(smiles_list, mpo_scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    pairs: list[PreferencePair] = []
    left = 0
    right = len(ranked) - 1
    while left < right and len(pairs) < max_pairs:
        pos_smi, pos_score = ranked[left]
        neg_smi, neg_score = ranked[right]
        if pos_smi != neg_smi and (float(pos_score) - float(neg_score)) >= min_margin:
            pairs.append(
                PreferencePair(
                    pocket_batch=pocket_batch,
                    chosen_smiles=pos_smi,
                    rejected_smiles=neg_smi,
                    chosen_score=float(pos_score),
                    rejected_score=float(neg_score),
                )
            )
        left += 1
        right -= 1
    return pairs


def _extract_z_tokens_deterministic(model: nn.Module, pooled_emb: Tensor) -> Tensor | None:
    """Sample deterministic CVAE prior mean (``latent_temp=0.0``) if model has a CVAE."""
    cvae = getattr(model, "cvae", None)
    if cvae is None:
        return None
    if hasattr(cvae, "sample_prior"):
        z_out = cvae.sample_prior(pooled_emb, latent_temp=0.0)
        if isinstance(z_out, Tensor):
            return z_out
        if isinstance(z_out, tuple) and isinstance(z_out[0], Tensor):
            return z_out[0]
        if hasattr(z_out, "z_tokens") and isinstance(z_out.z_tokens, Tensor):
            return z_out.z_tokens
    z_out = cvae(pooled_emb, ligand_emb=None, latent_temp=0.0)
    if isinstance(z_out, Tensor):
        return z_out
    if isinstance(z_out, tuple) and isinstance(z_out[0], Tensor):
        return z_out[0]
    if hasattr(z_out, "z_tokens") and isinstance(z_out.z_tokens, Tensor):
        return z_out.z_tokens
    return None


def _encode_sequence_to_io(
    smiles: str,
    tokenizer: Any,
    max_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Encode SMILES into teacher-forced ``(input_ids[1, L], target_ids[1, L])``."""
    ids = list(tokenizer.encode(smiles, randomize=False, max_len=max_len))
    if len(ids) < 2:
        bos_id = int(getattr(tokenizer, "BOS_ID", 1))
        eos_id = int(getattr(tokenizer, "EOS_ID", 2))
        ids = [bos_id, eos_id]
    inp = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    tgt = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    return inp, tgt


def _compute_decoder_logps(
    decoder: nn.Module,
    memory: Tensor,
    memory_mask: Tensor,
    input_ids: Tensor,
    target_ids: Tensor,
    pad_id: int = 0,
    length_normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute sequence log-probs and per-token mean log-probs from pocket memory."""
    dec_out = decoder(input_ids, memory=memory, memory_mask=memory_mask, return_attn=False)
    logits: Tensor = dec_out[0] if isinstance(dec_out, tuple) else dec_out

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    valid_mask = (target_ids != pad_id).to(dtype=token_logps.dtype)
    summed = (token_logps * valid_mask).sum(dim=-1)
    lengths = valid_mask.sum(dim=-1).clamp(min=1.0)
    per_token = summed / lengths
    return (per_token if length_normalize else summed), per_token


def compute_sequence_logps(
    model: nn.Module,
    pocket_batch: Mapping[str, Any],
    input_ids: Tensor,
    target_ids: Tensor,
    pad_id: int = 0,
    length_normalize: bool = True,
) -> Tensor:
    """Compute conditional sequence log-probabilities ``log p(y | pocket)``.

    Args:
        model: ``GenEuclideanModel`` exposing ``encode_pocket``, ``build_memory``, ``decoder``.
        pocket_batch: Pocket tensors (``coords``, ``node_feats``, ``atom_mask``).
        input_ids: Teacher-forced prefix token IDs ``[B, L]``.
        target_ids: Next-token target IDs ``[B, L]``.
        pad_id: Padding token ID excluded from summation.
        length_normalize: Whether to divide summed log-probs by non-pad sequence length
            (default ``True`` to keep DPO gradients scale-invariant across SMILES lengths).

    Returns:
        Sequence log-probabilities of shape ``[B]``.
    """
    device = input_ids.device
    coords = pocket_batch["coords"].to(device=device)
    node_feats = pocket_batch["node_feats"].to(device=device)
    atom_mask = pocket_batch["atom_mask"].to(device=device)

    if coords.size(0) == 1 and input_ids.size(0) > 1:
        b = input_ids.size(0)
        coords = coords.expand(b, -1, -1)
        node_feats = node_feats.expand(b, -1, -1)
        atom_mask = atom_mask.expand(b, -1)

    pocket_out = model.encode_pocket(coords, node_feats, atom_mask)  # type: ignore[operator]
    z_tokens = _extract_z_tokens_deterministic(model, pocket_out.pooled_emb)
    memory, memory_mask = model.build_memory(  # type: ignore[operator]
        pocket_out,
        atom_mask,
        z_tokens=z_tokens,
        use_null_pocket=False,
    )
    decoder: nn.Module = model.decoder  # type: ignore[assignment]
    seq_logps, _ = _compute_decoder_logps(
        decoder,
        memory,
        memory_mask,
        input_ids,
        target_ids,
        pad_id=pad_id,
        length_normalize=length_normalize,
    )
    return seq_logps


def _encode_pocket_memory(
    model: nn.Module,
    pocket_batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Encode a single pocket once into deterministic ``(memory, memory_mask)`` tensors."""
    coords = pocket_batch["coords"].to(device=device)
    node_feats = pocket_batch["node_feats"].to(device=device)
    atom_mask = pocket_batch["atom_mask"].to(device=device)
    pocket_out = model.encode_pocket(coords, node_feats, atom_mask)  # type: ignore[operator]
    z_tokens = _extract_z_tokens_deterministic(model, pocket_out.pooled_emb)
    memory, memory_mask = model.build_memory(  # type: ignore[operator]
        pocket_out,
        atom_mask,
        z_tokens=z_tokens,
        use_null_pocket=False,
    )
    return memory.detach(), memory_mask.detach()


def _unpack_pair(
    pair: PreferencePair | tuple[Mapping[str, Any], str, str] | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    """Normalize tuple, dict, or ``PreferencePair`` into ``(pocket_batch, chosen, rejected)``."""
    if isinstance(pair, PreferencePair):
        return pair.pocket_batch, pair.chosen_smiles, pair.rejected_smiles
    if isinstance(pair, tuple):
        if len(pair) < 3:
            raise ValueError(f"Expected (pocket_batch, chosen_smiles, rejected_smiles), got {pair}")
        return pair[0], str(pair[1]), str(pair[2])
    if isinstance(pair, Mapping):
        return (
            pair["pocket_batch"],
            str(pair["chosen_smiles"]),
            str(pair["rejected_smiles"]),
        )
    raise TypeError(f"Unsupported preference pair type: {type(pair).__name__}")


def run_dpo_alignment(
    model: nn.Module,
    ref_model: nn.Module,
    preference_pairs: Sequence[
        PreferencePair | tuple[Mapping[str, Any], str, str] | Mapping[str, Any]
    ],
    cfg: ExperimentConfig,
    device: torch.device | str | None = None,
    *,
    tokenizer: Any | None = None,
    epochs: int = 1,
    beta: float = 0.5,
    lr: float = 1e-5,
    weight_decay: float = 1e-2,
    grad_clip: float | None = None,
    length_normalize: bool = True,
    freeze_non_decoder: bool = True,
    sft_nll_weight: float = 0.25,
    grad_accum_steps: int = 8,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run stabilized Stage-3 MPO-guided RPO-DPO preference alignment.

    Updates ``model`` to prefer high-MPO (potent, drug-like, PAINS-free) ligands over
    low-MPO rejected ligands while preventing likelihood displacement via length-normalized
    log-probabilities, an RPO/SFT NLL anchor on chosen sequences, gradient accumulation,
    and optional freezing of the 3D encoder/CVAE/multi-task heads.

    Args:
        model: Policy ``GenEuclideanModel`` to align in place.
        ref_model: Frozen reference ``GenEuclideanModel`` copy (Stage-2 checkpoint).
        preference_pairs: Sequence of ``PreferencePair``, ``(pocket_batch, chosen, rejected)``
            tuples, or mappings with keys ``pocket_batch``, ``chosen_smiles``, ``rejected_smiles``.
        cfg: Experiment configuration.
        device: Execution device (defaults to ``select_device(cfg.train.device)``).
        tokenizer: ``MolecularTokenizer`` used to encode SMILES strings.
        epochs: Number of alignment epochs (default ``1``).
        beta: DPO KL-regularization strength (default ``0.5``).
        lr: Fine-tuning learning rate (default ``1e-5``).
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm (defaults to ``cfg.train.grad_clip``).
        length_normalize: Divide sequence log-probs by non-pad token count (default ``True``).
        freeze_non_decoder: Freeze all non-``decoder`` parameters (encoder, CVAE, multi-task
            heads) during DPO so binding affinity and DrugCLIP representations remain intact,
            restoring original ``requires_grad`` flags upon exit (default ``True``).
        sft_nll_weight: RPO/SFT NLL anchor weight on length-normalized chosen log-probs
            (default ``0.25``).
        grad_accum_steps: Number of preference pairs accumulated per optimizer step
            (default ``8``).
        save_dir: Optional directory where ``dpo_aligned.pt`` and ``dpo_history.jsonl`` are saved.

    Returns:
        Dictionary containing per-epoch DPO alignment metrics and optional checkpoint path.
    """
    eff_device = (
        select_device(cfg.train.device)
        if device is None
        else (select_device(device) if isinstance(device, str) else device)
    )
    eff_tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    if eff_tokenizer is None:
        raise ValueError("A tokenizer must be passed to run_dpo_alignment (or attached to model).")

    pad_id = int(getattr(eff_tokenizer, "PAD_ID", 0))
    max_len = cfg.data.max_seq_len
    eff_grad_clip = cfg.train.grad_clip if grad_clip is None else grad_clip
    eff_accum = max(1, int(grad_accum_steps))
    eff_sft_weight = max(0.0, float(sft_nll_weight))

    model = model.to(eff_device)
    ref_model = ref_model.to(eff_device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if len(preference_pairs) == 0:
        return {"epochs": 0, "history": [], "ckpt_path": None}

    orig_requires_grad: dict[str, bool] = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    history: list[dict[str, float | int]] = []

    try:
        if freeze_non_decoder:
            for name, param in model.named_parameters():
                if not name.startswith("decoder."):
                    param.requires_grad = False
                    param.grad = None

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters available for DPO alignment.")

        # Pre-encode preference pairs, reference log-probs, and (when frozen) pocket memory once
        cached_items: list[
            tuple[
                Mapping[str, Any],
                Tensor | None,
                Tensor | None,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
            ]
        ] = []
        ref_decoder: nn.Module = ref_model.decoder  # type: ignore[assignment]
        with torch.no_grad():
            for raw_pair in preference_pairs:
                pocket_batch, chosen_smi, rejected_smi = _unpack_pair(raw_pair)
                pos_in, pos_tgt = _encode_sequence_to_io(
                    chosen_smi, eff_tokenizer, max_len, eff_device
                )
                neg_in, neg_tgt = _encode_sequence_to_io(
                    rejected_smi, eff_tokenizer, max_len, eff_device
                )
                mem, mem_mask = _encode_pocket_memory(ref_model, pocket_batch, eff_device)
                ref_pos, _ = _compute_decoder_logps(
                    ref_decoder,
                    mem,
                    mem_mask,
                    pos_in,
                    pos_tgt,
                    pad_id=pad_id,
                    length_normalize=length_normalize,
                )
                ref_neg, _ = _compute_decoder_logps(
                    ref_decoder,
                    mem,
                    mem_mask,
                    neg_in,
                    neg_tgt,
                    pad_id=pad_id,
                    length_normalize=length_normalize,
                )
                cached_items.append(
                    (
                        pocket_batch,
                        mem.cpu() if freeze_non_decoder else None,
                        mem_mask.cpu() if freeze_non_decoder else None,
                        pos_in,
                        pos_tgt,
                        neg_in,
                        neg_tgt,
                        ref_pos.detach(),
                        ref_neg.detach(),
                    )
                )

        optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        policy_decoder: nn.Module = model.decoder  # type: ignore[assignment]
        n_total = len(cached_items)

        for epoch in range(1, max(1, epochs) + 1):
            t0 = time.perf_counter()
            if freeze_non_decoder:
                model.eval()
                policy_decoder.train()
            else:
                model.train()

            dpo_loss_sum = 0.0
            sft_nll_sum = 0.0
            total_loss_sum = 0.0
            pos_reward_sum = 0.0
            neg_reward_sum = 0.0
            margin_sum = 0.0
            correct = 0
            opt_steps = 0

            optimizer.zero_grad(set_to_none=True)
            for idx, (
                pocket_batch,
                cached_mem,
                cached_mem_mask,
                pos_in,
                pos_tgt,
                neg_in,
                neg_tgt,
                ref_pos,
                ref_neg,
            ) in enumerate(cached_items):
                if freeze_non_decoder and cached_mem is not None and cached_mem_mask is not None:
                    mem_dev = cached_mem.to(device=eff_device)
                    mask_dev = cached_mem_mask.to(device=eff_device)
                    pi_pos, pi_pos_norm = _compute_decoder_logps(
                        policy_decoder,
                        mem_dev,
                        mask_dev,
                        pos_in,
                        pos_tgt,
                        pad_id=pad_id,
                        length_normalize=length_normalize,
                    )
                    pi_neg, _ = _compute_decoder_logps(
                        policy_decoder,
                        mem_dev,
                        mask_dev,
                        neg_in,
                        neg_tgt,
                        pad_id=pad_id,
                        length_normalize=length_normalize,
                    )
                else:
                    pi_pos = compute_sequence_logps(
                        model,
                        pocket_batch,
                        pos_in,
                        pos_tgt,
                        pad_id=pad_id,
                        length_normalize=length_normalize,
                    )
                    pi_neg = compute_sequence_logps(
                        model,
                        pocket_batch,
                        neg_in,
                        neg_tgt,
                        pad_id=pad_id,
                        length_normalize=length_normalize,
                    )
                    if length_normalize:
                        pi_pos_norm = pi_pos
                    else:
                        pos_len = (pos_tgt != pad_id).to(pi_pos.dtype).sum(dim=-1).clamp(min=1.0)
                        pi_pos_norm = pi_pos / pos_len

                dpo_loss = compute_dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=beta)
                sft_nll = -pi_pos_norm.mean()
                loss = dpo_loss + eff_sft_weight * sft_nll

                window_start = (idx // eff_accum) * eff_accum
                window_size = min(eff_accum, n_total - window_start)
                (loss / float(window_size)).backward()

                if ((idx + 1) % eff_accum == 0) or ((idx + 1) == n_total):
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=eff_grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_steps += 1

                chosen_r = float((beta * (pi_pos.detach() - ref_pos)).mean().item())
                rejected_r = float((beta * (pi_neg.detach() - ref_neg)).mean().item())
                margin = chosen_r - rejected_r
                dpo_loss_sum += float(dpo_loss.detach().item())
                sft_nll_sum += float(sft_nll.detach().item())
                total_loss_sum += float(loss.detach().item())
                pos_reward_sum += chosen_r
                neg_reward_sum += rejected_r
                margin_sum += margin
                if margin > 0.0:
                    correct += 1

            n_pairs = float(n_total)
            history.append(
                {
                    "epoch": epoch,
                    "dpo_loss": dpo_loss_sum / n_pairs,
                    "sft_nll": sft_nll_sum / n_pairs,
                    "total_loss": total_loss_sum / n_pairs,
                    "chosen_reward": pos_reward_sum / n_pairs,
                    "rejected_reward": neg_reward_sum / n_pairs,
                    "reward_margin": margin_sum / n_pairs,
                    "preference_accuracy": float(correct) / n_pairs,
                    "optimizer_steps": opt_steps,
                    "elapsed_sec": time.perf_counter() - t0,
                }
            )
    finally:
        for name, param in model.named_parameters():
            if name in orig_requires_grad:
                param.requires_grad = orig_requires_grad[name]

    ckpt_path_str: str | None = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "dpo_aligned.pt"
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": cfg.model_dump(mode="python"),
                "dpo_history": history,
            },
            ckpt_path,
        )
        ckpt_path_str = str(ckpt_path)
        hist_path = out_dir / "dpo_history.jsonl"
        with hist_path.open("w", encoding="utf-8") as handle:
            for row in history:
                handle.write(json.dumps(row) + "\n")

    return {
        "epochs": len(history),
        "history": history,
        "ckpt_path": ckpt_path_str,
    }


__all__ = [
    "PreferencePair",
    "build_preference_pairs_from_mpo",
    "compute_sequence_logps",
    "run_dpo_alignment",
]
