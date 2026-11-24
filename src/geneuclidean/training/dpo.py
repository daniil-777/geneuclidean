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


def compute_sequence_logps(
    model: nn.Module,
    pocket_batch: Mapping[str, Any],
    input_ids: Tensor,
    target_ids: Tensor,
    pad_id: int = 0,
    length_normalize: bool = False,
) -> Tensor:
    """Compute conditional sequence log-probabilities ``log p(y | pocket)``.

    Args:
        model: ``GenEuclideanModel`` exposing ``encode_pocket``, ``build_memory``, ``decoder``.
        pocket_batch: Pocket tensors (``coords``, ``node_feats``, ``atom_mask``).
        input_ids: Teacher-forced prefix token IDs ``[B, L]``.
        target_ids: Next-token target IDs ``[B, L]``.
        pad_id: Padding token ID excluded from summation.
        length_normalize: Whether to divide summed log-probs by non-pad sequence length.

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
    dec_out = decoder(input_ids, memory=memory, memory_mask=memory_mask, return_attn=False)
    logits: Tensor = dec_out[0] if isinstance(dec_out, tuple) else dec_out

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    valid_mask = (target_ids != pad_id).to(dtype=token_logps.dtype)
    seq_logps = (token_logps * valid_mask).sum(dim=-1)
    if length_normalize:
        lengths = valid_mask.sum(dim=-1).clamp(min=1.0)
        seq_logps = seq_logps / lengths
    return seq_logps


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
    epochs: int = 2,
    beta: float = 0.1,
    lr: float = 5e-5,
    weight_decay: float = 1e-2,
    grad_clip: float | None = None,
    length_normalize: bool = False,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run lightweight Stage-3 MPO-guided DPO preference alignment (1–3 epochs).

    Updates ``model`` to increase likelihood of high-MPO (potent, drug-like, PAINS-free)
    ligands relative to low-MPO rejected ligands while remaining KL-regularized to the
    frozen ``ref_model``.

    Args:
        model: Policy ``GenEuclideanModel`` to align in place.
        ref_model: Frozen reference ``GenEuclideanModel`` copy (Stage-2 checkpoint).
        preference_pairs: Sequence of ``PreferencePair``, ``(pocket_batch, chosen, rejected)``
            tuples, or mappings with keys ``pocket_batch``, ``chosen_smiles``, ``rejected_smiles``.
        cfg: Experiment configuration.
        device: Execution device (defaults to ``select_device(cfg.train.device)``).
        tokenizer: ``MolecularTokenizer`` used to encode SMILES strings.
        epochs: Number of alignment epochs (typically 1–3).
        beta: DPO KL-regularization strength.
        lr: Fine-tuning learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm (defaults to ``cfg.train.grad_clip``).
        length_normalize: Whether to length-normalize sequence log-probs.
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

    model = model.to(eff_device)
    ref_model = ref_model.to(eff_device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if len(preference_pairs) == 0:
        return {"epochs": 0, "history": [], "ckpt_path": None}

    # Pre-encode preference pairs and cache frozen reference log-probabilities once
    cached_items: list[
        tuple[Mapping[str, Any], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    ] = []
    with torch.no_grad():
        for raw_pair in preference_pairs:
            pocket_batch, chosen_smi, rejected_smi = _unpack_pair(raw_pair)
            pos_in, pos_tgt = _encode_sequence_to_io(chosen_smi, eff_tokenizer, max_len, eff_device)
            neg_in, neg_tgt = _encode_sequence_to_io(
                rejected_smi, eff_tokenizer, max_len, eff_device
            )
            ref_pos = compute_sequence_logps(
                ref_model,
                pocket_batch,
                pos_in,
                pos_tgt,
                pad_id=pad_id,
                length_normalize=length_normalize,
            ).detach()
            ref_neg = compute_sequence_logps(
                ref_model,
                pocket_batch,
                neg_in,
                neg_tgt,
                pad_id=pad_id,
                length_normalize=length_normalize,
            ).detach()
            cached_items.append((pocket_batch, pos_in, pos_tgt, neg_in, neg_tgt, ref_pos, ref_neg))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, max(1, epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum = 0.0
        pos_reward_sum = 0.0
        neg_reward_sum = 0.0
        margin_sum = 0.0
        correct = 0

        for pocket_batch, pos_in, pos_tgt, neg_in, neg_tgt, ref_pos, ref_neg in cached_items:
            optimizer.zero_grad(set_to_none=True)
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
            loss = compute_dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=eff_grad_clip,
            )
            optimizer.step()

            chosen_r = float((beta * (pi_pos.detach() - ref_pos)).mean().item())
            rejected_r = float((beta * (pi_neg.detach() - ref_neg)).mean().item())
            margin = chosen_r - rejected_r
            loss_sum += float(loss.detach().item())
            pos_reward_sum += chosen_r
            neg_reward_sum += rejected_r
            margin_sum += margin
            if margin > 0.0:
                correct += 1

        n_pairs = float(len(cached_items))
        history.append(
            {
                "epoch": epoch,
                "dpo_loss": loss_sum / n_pairs,
                "chosen_reward": pos_reward_sum / n_pairs,
                "rejected_reward": neg_reward_sum / n_pairs,
                "reward_margin": margin_sum / n_pairs,
                "preference_accuracy": float(correct) / n_pairs,
                "elapsed_sec": time.perf_counter() - t0,
            }
        )

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
