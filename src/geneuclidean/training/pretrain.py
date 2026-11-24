"""Stage-1 unconditional chemical language model pretraining on ZINC-250k."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from geneuclidean.config import ExperimentConfig, seed_everything, select_device
from geneuclidean.training.losses import compute_lm_loss


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.08,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Build a linear-warmup + cosine-decay learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer whose LR groups will be scheduled.
        total_steps: Total number of optimization steps.
        warmup_ratio: Fraction of total steps used for linear warmup.
        min_lr_ratio: Floor multiplier at the end of cosine decay.

    Returns:
        Configured ``LambdaLR`` scheduler.
    """
    eff_total = max(1, int(total_steps))
    warmup_steps = int(eff_total * max(0.0, min(warmup_ratio, 0.99)))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(1, eff_total - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_adamw_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split parameters into weight-decayed and non-decayed groups (biases & norms)."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    return groups


def _extract_logits(output: Any) -> Tensor:
    """Extract logits tensor from ``(logits, attn)``, ``ModelOutput``, or bare ``Tensor``."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple):
        first = output[0]
        if not isinstance(first, Tensor):
            raise TypeError(f"Expected Tensor logits at index 0, got {type(first).__name__}")
        return first
    if hasattr(output, "logits"):
        logits = output.logits
        if not isinstance(logits, Tensor):
            raise TypeError(f"Expected Tensor output.logits, got {type(logits).__name__}")
        return logits
    raise TypeError(f"Unsupported decoder output type: {type(output).__name__}")


def _forward_unconditional(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Invoke a chemical LM decoder unconditionally (``memory=None``)."""
    try:
        out = model(input_ids, memory=None, memory_mask=None, return_attn=False)
    except TypeError:
        out = model(input_ids)
    return _extract_logits(out)


def _instantiate_default_decoder(cfg: ExperimentConfig, tokenizer: Any) -> nn.Module:
    """Lazily instantiate the configured chemical decoder for Stage-1 pretraining."""
    vocab_size = int(getattr(tokenizer, "vocab_size", cfg.decoder.vocab_size))
    dec_cfg = cfg.decoder.model_copy(update={"vocab_size": vocab_size})
    if dec_cfg.arch == "transformer":
        from geneuclidean.models.decoders.transformer import PocketTransformerDecoder

        return PocketTransformerDecoder(dec_cfg)
    from geneuclidean.models.decoders.lstm import PocketLSTMDecoder

    return PocketLSTMDecoder(dec_cfg)


@torch.no_grad()
def evaluate_chemical_prior(
    model: nn.Module,
    val_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    pad_id: int = 0,
) -> dict[str, float]:
    """Evaluate unconditional chemical LM token-weighted NLL and perplexity.

    Uses ``label_smoothing=0.0`` so reported validation perplexity reflects exact NLL.

    Args:
        model: Chemical LM decoder module.
        val_loader: Validation data loader yielding ``input_ids`` and ``target_ids``.
        device: Execution device.
        pad_id: Padding token ID ignored in the loss.

    Returns:
        Dictionary with ``val_loss``, ``val_ppl``, and ``num_tokens``.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
        target_ids = batch["target_ids"].to(device=device, dtype=torch.long)
        valid_tokens = int((target_ids != pad_id).sum().item())
        if valid_tokens == 0:
            continue
        logits = _forward_unconditional(model, input_ids)
        loss = compute_lm_loss(logits, target_ids, pad_id=pad_id, label_smoothing=0.0)
        total_nll += float(loss.item()) * valid_tokens
        total_tokens += valid_tokens

    mean_nll = total_nll / float(max(1, total_tokens))
    return {
        "val_loss": mean_nll,
        "val_ppl": math.exp(min(mean_nll, 20.0)),
        "num_tokens": float(total_tokens),
    }


def pretrain_chemical_prior(
    cfg: ExperimentConfig,
    train_loader: Iterable[Mapping[str, Any]],
    val_loader: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    save_dir: str | Path,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Run Stage-1 unconditional chemical LM pretraining on ZINC-250k.

    Trains with AdamW, cosine LR decay with linear warmup, gradient norm clipping,
    exact validation perplexity tracking, JSONL epoch history logging, and best-val
    checkpoint saving (``best_prior.pt``).

    Args:
        cfg: Experiment configuration (uses ``cfg.train`` and ``cfg.decoder``).
        train_loader: Training dataloader yielding dicts with ``input_ids`` and ``target_ids``.
        val_loader: Validation dataloader yielding dicts with ``input_ids`` and ``target_ids``.
        tokenizer: ``MolecularTokenizer`` (exposing ``PAD_ID`` and ``vocab_size``).
        save_dir: Directory where ``best_prior.pt`` and ``history.jsonl`` are written.
        model: Optional pre-instantiated decoder module (if omitted, instantiates the
            configured decoder from ``geneuclidean.models.decoders``).

    Returns:
        Summary dictionary containing best validation metrics, checkpoint path, and history.
    """
    seed_everything(cfg.train.seed)
    device = select_device(cfg.train.device)
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pad_id = int(getattr(tokenizer, "PAD_ID", 0))
    vocab_size = int(getattr(tokenizer, "vocab_size", cfg.decoder.vocab_size))

    decoder = model if model is not None else _instantiate_default_decoder(cfg, tokenizer)
    decoder = decoder.to(device)

    steps_per_epoch = len(train_loader) if isinstance(train_loader, Sized) else 1
    total_steps = max(1, steps_per_epoch * cfg.train.epochs)

    param_groups = _build_adamw_param_groups(
        decoder,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    optimizer = AdamW(param_groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg.train.warmup_ratio,
    )

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = 0
    best_ckpt_path = out_dir / "best_prior.pt"
    history_path = out_dir / "history.jsonl"
    history: list[dict[str, Any]] = []

    with history_path.open("w", encoding="utf-8") as hist_file:
        for epoch in range(1, cfg.train.epochs + 1):
            t0 = time.perf_counter()
            decoder.train()
            train_loss_sum = 0.0
            train_tokens = 0
            grad_norm_sum = 0.0
            num_steps = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
                target_ids = batch["target_ids"].to(device=device, dtype=torch.long)
                valid_tokens = int((target_ids != pad_id).sum().item())
                if valid_tokens == 0:
                    continue

                optimizer.zero_grad(set_to_none=True)
                logits = _forward_unconditional(decoder, input_ids)
                loss = compute_lm_loss(
                    logits,
                    target_ids,
                    pad_id=pad_id,
                    label_smoothing=cfg.train.label_smoothing,
                )
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(),
                        max_norm=cfg.train.grad_clip,
                    ).item()
                )
                optimizer.step()
                scheduler.step()

                train_loss_sum += float(loss.detach().item()) * valid_tokens
                train_tokens += valid_tokens
                grad_norm_sum += grad_norm
                num_steps += 1

            train_loss = train_loss_sum / float(max(1, train_tokens))
            train_ppl = math.exp(min(train_loss, 20.0))
            mean_grad_norm = grad_norm_sum / float(max(1, num_steps))

            val_metrics = evaluate_chemical_prior(decoder, val_loader, device=device, pad_id=pad_id)
            val_loss = val_metrics["val_loss"]
            val_ppl = val_metrics["val_ppl"]
            current_lr = float(optimizer.param_groups[0]["lr"])
            elapsed_sec = time.perf_counter() - t0

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": {
                            k: v.detach().cpu() for k, v in decoder.state_dict().items()
                        },
                        "config": cfg.model_dump(mode="python"),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "vocab_size": vocab_size,
                    },
                    best_ckpt_path,
                )

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "lr": current_lr,
                "grad_norm": mean_grad_norm,
                "elapsed_sec": elapsed_sec,
                "is_best": is_best,
            }
            history.append(record)
            hist_file.write(json.dumps(record) + "\n")
            hist_file.flush()

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
        "best_ckpt_path": str(best_ckpt_path),
        "history_path": str(history_path),
        "history": history,
    }


__all__ = [
    "build_cosine_warmup_scheduler",
    "evaluate_chemical_prior",
    "pretrain_chemical_prior",
]
