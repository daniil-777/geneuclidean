"""Stage-2 pocket-conditioned multi-task trainer for GenEuclidean v2."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from geneuclidean.config import ExperimentConfig, select_device
from geneuclidean.training.losses import (
    compute_cvae_kl_loss,
    compute_cvae_kl_nats,
    compute_drugclip_loss,
    compute_lm_loss,
    compute_multitask_loss,
    compute_pkd_loss,
    compute_property_loss,
    cyclical_kl_beta,
)
from geneuclidean.training.pretrain import build_cosine_warmup_scheduler

_CROSS_ATTN_KEYWORDS = (
    "cross_attn",
    "cross_attention",
    "cross_norm",
    "cross_gate",
    "memory_proj",
    "pocket_proj",
    "pocket_attn",
)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move all tensor fields in a collated pocket batch to ``device``."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _compute_pearson_and_rmse(preds: list[Tensor], targets: list[Tensor]) -> tuple[float, float]:
    """Compute Pearson r and RMSE over finite pKd predictions."""
    if not preds:
        return 0.0, 0.0
    y_pred = torch.cat(preds, dim=0).detach().cpu().numpy().astype(np.float64)
    y_true = torch.cat(targets, dim=0).detach().cpu().numpy().astype(np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_pred = y_pred[finite]
    y_true = y_true[finite]
    if y_true.size == 0:
        return 0.0, 0.0

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if y_true.size < 2:
        return 0.0, rmse

    pred_std = float(np.std(y_pred))
    true_std = float(np.std(y_true))
    if pred_std < 1e-12 or true_std < 1e-12:
        return 0.0, rmse
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    if not math.isfinite(corr):
        corr = 0.0
    return corr, rmse


def _compute_clip_recall(
    pocket_embs: list[Tensor],
    ligand_embs: list[Tensor],
) -> tuple[float, float]:
    """Compute symmetric DrugCLIP Recall@1 and Recall@5 over all validation complexes."""
    if not pocket_embs:
        return 0.0, 0.0
    p_all = F.normalize(torch.cat(pocket_embs, dim=0).detach().cpu().float(), p=2, dim=-1, eps=1e-8)
    l_all = F.normalize(torch.cat(ligand_embs, dim=0).detach().cpu().float(), p=2, dim=-1, eps=1e-8)
    n = p_all.size(0)
    if n < 2:
        return 0.0, 0.0

    sim = p_all @ l_all.transpose(0, 1)  # [N, N]
    targets = torch.arange(n, dtype=torch.long)

    def _recall_at_k(matrix: Tensor, k: int) -> float:
        eff_k = min(k, matrix.size(1))
        topk = torch.topk(matrix, k=eff_k, dim=-1).indices
        hits = (topk == targets.unsqueeze(1)).any(dim=-1)
        return float(hits.float().mean().item())

    r1 = 0.5 * (_recall_at_k(sim, 1) + _recall_at_k(sim.transpose(0, 1), 1))
    r5 = 0.5 * (_recall_at_k(sim, 5) + _recall_at_k(sim.transpose(0, 1), 5))
    return r1, r5


def _is_decoder_backbone_param(name: str) -> bool:
    """Identify pretrained decoder self-attention/embedding/FFN backbone parameters."""
    if not name.startswith("decoder."):
        return False
    lower = name.lower()
    return not any(keyword in lower for keyword in _CROSS_ATTN_KEYWORDS)


class PocketTrainer:
    """Stage-2 trainer for 3D pocket-conditioned molecular generation and affinity prediction.

    Features:
        - Clean loading of Stage-1 ZINC-250k pretrained decoder checkpoints.
        - Differential learning rates (``encoder_lr`` vs ``decoder_lr``) with proper
          weight-decay exclusion on biases and normalization parameters.
        - Gradual decoder unfreezing (``freeze_decoder_epochs``) so the 3D encoder and
          cross-attention gates warm up before fine-tuning the pretrained chemical backbone.
        - Cyclical KL annealing (Fu et al., NAACL 2019) + Classifier-Free Guidance pocket
          dropout during training.
        - Full validation pass every epoch tracking LM NLL, perplexity, pKd Pearson r & RMSE,
          DrugCLIP Recall@1/5, and KL nats; ``best.pt`` is saved strictly on validation
          objective (never single-batch train loss).
        - Full checkpoint save/resume via ``state_dict`` / ``load_state_dict``.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        tokenizer: Any,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.pad_id = int(getattr(tokenizer, "PAD_ID", 0))
        self.device = (
            select_device(cfg.train.device)
            if device is None
            else (select_device(device) if isinstance(device, str) else device)
        )
        self.model = model.to(self.device)

        self._decoder_backbone_params: list[nn.Parameter] = []
        self._other_params: list[nn.Parameter] = []
        self._validate_decoder_vocab_size()
        self._load_pretrained_decoder_if_configured()
        self._partition_parameters()

        self.optimizer: AdamW | None = None
        self.scheduler: LambdaLR | None = None
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_val_objective: float = float("inf")
        self.history: list[dict[str, Any]] = []

    def _validate_decoder_vocab_size(self) -> None:
        """Ensure decoder vocabulary matches active tokenizer when both are exposed."""
        decoder = getattr(self.model, "decoder", None)
        dec_vocab = getattr(decoder, "vocab_size", None)
        tok_vocab = getattr(self.tokenizer, "vocab_size", None)
        if dec_vocab is not None and tok_vocab is not None and int(dec_vocab) != int(tok_vocab):
            raise ValueError(
                f"Decoder vocab_size ({int(dec_vocab)}) does not match tokenizer.vocab_size "
                f"({int(tok_vocab)}). Sync cfg.decoder.vocab_size before building the model."
            )

    def _load_pretrained_decoder_if_configured(self) -> None:
        """Load Stage-1 pretrained decoder weights when ``pretrained_decoder_ckpt`` is set."""
        ckpt_ref = self.cfg.train.pretrained_decoder_ckpt
        if not ckpt_ref:
            return
        ckpt_path = Path(ckpt_ref)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Configured pretrained_decoder_ckpt not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict: dict[str, Tensor]
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unexpected checkpoint payload type in {ckpt_path}: {type(payload)}")

        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            raise AttributeError(
                "Model has no 'decoder' attribute for pretrained backbone loading."
            )

        dec_vocab = getattr(decoder, "vocab_size", None)
        if isinstance(payload, dict) and payload.get("vocab_size") is not None:
            ckpt_vocab = int(payload["vocab_size"])
            if dec_vocab is not None and ckpt_vocab != int(dec_vocab):
                raise ValueError(
                    f"Pretrained checkpoint vocab_size ({ckpt_vocab}) does not match "
                    f"decoder vocab_size ({int(dec_vocab)}) in {ckpt_path}."
                )

        normalized_sd = {
            (k[len("decoder.") :] if k.startswith("decoder.") else k): v
            for k, v in state_dict.items()
        }
        decoder_sd = decoder.state_dict()
        for key in ("token_emb.weight", "tok_emb.weight", "lm_head.weight"):
            if key in decoder_sd:
                if key not in normalized_sd:
                    if key in ("token_emb.weight", "tok_emb.weight"):
                        raise ValueError(
                            f"Pretrained decoder checkpoint {ckpt_path} is missing '{key}'."
                        )
                    continue
                if tuple(normalized_sd[key].shape) != tuple(decoder_sd[key].shape):
                    raise ValueError(
                        f"Pretrained checkpoint shape mismatch for '{key}': "
                        f"checkpoint={tuple(normalized_sd[key].shape)} vs "
                        f"decoder={tuple(decoder_sd[key].shape)}."
                    )

        if hasattr(decoder, "load_pretrained_backbone"):
            report = decoder.load_pretrained_backbone(state_dict)
            if isinstance(report, Mapping):
                skipped = report.get("skipped") or report.get("skipped_keys") or []
                if "token_emb.weight" in skipped or "tok_emb.weight" in skipped:
                    raise ValueError(
                        f"Pretrained decoder loading dropped token embedding weights: {skipped}"
                    )
        else:
            decoder.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

    def _partition_parameters(self) -> None:
        """Separate pretrained decoder backbone parameters from encoder/CVAE/heads/cross-attn."""
        self._decoder_backbone_params = []
        self._other_params = []
        for name, param in self.model.named_parameters():
            if _is_decoder_backbone_param(name):
                self._decoder_backbone_params.append(param)
            else:
                self._other_params.append(param)

    def build_optimizer(self) -> AdamW:
        """Construct AdamW with differential LRs and bias/norm weight-decay exclusions."""
        enc_decay: list[nn.Parameter] = []
        enc_no_decay: list[nn.Parameter] = []
        dec_decay: list[nn.Parameter] = []
        dec_no_decay: list[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            no_decay = param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
            if _is_decoder_backbone_param(name):
                (dec_no_decay if no_decay else dec_decay).append(param)
            else:
                (enc_no_decay if no_decay else enc_decay).append(param)

        param_groups: list[dict[str, Any]] = []
        if enc_decay:
            param_groups.append(
                {
                    "name": "encoder_decay",
                    "params": enc_decay,
                    "lr": self.cfg.train.encoder_lr,
                    "weight_decay": self.cfg.train.weight_decay,
                }
            )
        if enc_no_decay:
            param_groups.append(
                {
                    "name": "encoder_no_decay",
                    "params": enc_no_decay,
                    "lr": self.cfg.train.encoder_lr,
                    "weight_decay": 0.0,
                }
            )
        if dec_decay:
            param_groups.append(
                {
                    "name": "decoder_decay",
                    "params": dec_decay,
                    "lr": self.cfg.train.decoder_lr,
                    "weight_decay": self.cfg.train.weight_decay,
                }
            )
        if dec_no_decay:
            param_groups.append(
                {
                    "name": "decoder_no_decay",
                    "params": dec_no_decay,
                    "lr": self.cfg.train.decoder_lr,
                    "weight_decay": 0.0,
                }
            )

        return AdamW(param_groups)

    def set_decoder_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the pretrained decoder self-attention backbone."""
        for param in self._decoder_backbone_params:
            param.requires_grad = not frozen
            if frozen:
                param.grad = None

    def state_dict(self) -> dict[str, Any]:
        """Serialize full trainer state (model, optimizer, scheduler, counters, history)."""
        return {
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.cfg.model_dump(mode="python"),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_objective": self.best_val_objective,
            "history": list(self.history),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore model, optimizer, scheduler, counters, and history from checkpoint state."""
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        if self.optimizer is not None and state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            for opt_state in self.optimizer.state.values():
                for key, val in opt_state.items():
                    if isinstance(val, Tensor):
                        opt_state[key] = val.to(self.device)
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.current_epoch = int(state.get("current_epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_val_objective = float(state.get("best_val_objective", float("inf")))
        raw_history = state.get("history", [])
        self.history = list(raw_history) if isinstance(raw_history, list) else []

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save trainer state dictionary to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_path)
        return out_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load trainer state dictionary from disk."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.load_state_dict(payload)

    def train_epoch(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        epoch: int,
        total_steps: int,
    ) -> dict[str, float]:
        """Run one training epoch with gradual unfreezing, cyclical KL, and CFG dropout."""
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer and scheduler must be initialized before train_epoch.")

        should_freeze = (
            epoch <= self.cfg.train.freeze_decoder_epochs
            and len(self._decoder_backbone_params) > 0
            and len(self._other_params) > 0
        )
        self.set_decoder_frozen(should_freeze)
        self.model.train()

        sums: dict[str, float] = {
            "total_loss": 0.0,
            "lm_loss": 0.0,
            "kl_loss": 0.0,
            "kl_nats": 0.0,
            "pkd_loss": 0.0,
            "clip_loss": 0.0,
            "prop_loss": 0.0,
            "grad_norm": 0.0,
            "kl_beta": 0.0,
        }
        num_batches = 0

        for raw_batch in train_loader:
            batch = _move_batch_to_device(raw_batch, self.device)
            beta = (
                cyclical_kl_beta(
                    step=self.global_step,
                    total_steps=total_steps,
                    max_beta=self.cfg.cvae.kl_max_beta,
                    n_cycles=self.cfg.cvae.kl_cycles,
                )
                if self.cfg.cvae.enabled
                else 0.0
            )

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch, cfg_drop=True, return_attn=False)
            loss, step_metrics = compute_multitask_loss(
                out,
                batch,
                self.cfg,
                kl_beta=beta,
                pad_id=self.pad_id,
            )
            loss.backward()

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=self.cfg.train.grad_clip,
                ).item()
            )
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            for key in (
                "total_loss",
                "lm_loss",
                "kl_loss",
                "kl_nats",
                "pkd_loss",
                "clip_loss",
                "prop_loss",
                "kl_beta",
            ):
                sums[key] += step_metrics[key]
            sums["grad_norm"] += grad_norm
            num_batches += 1

        denom = float(max(1, num_batches))
        avg = {f"train_{k}": v / denom for k, v in sums.items()}
        avg["train_ppl"] = math.exp(min(avg["train_lm_loss"], 20.0))
        avg["decoder_frozen"] = 1.0 if should_freeze else 0.0
        return avg

    @torch.no_grad()
    def evaluate(self, val_loader: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        """Evaluate full validation set: LM NLL/PPL, pKd Pearson r & RMSE, DrugCLIP R@1/5, KL."""
        self.model.eval()

        lm_nll_sum = 0.0
        total_tokens = 0
        kl_nats_sum = 0.0
        kl_loss_sum = 0.0
        pkd_loss_sum = 0.0
        clip_loss_sum = 0.0
        prop_loss_sum = 0.0
        num_batches = 0

        pkd_preds: list[Tensor] = []
        pkd_targets: list[Tensor] = []
        clip_pockets: list[Tensor] = []
        clip_ligands: list[Tensor] = []

        for raw_batch in val_loader:
            batch = _move_batch_to_device(raw_batch, self.device)
            out = self.model(batch, cfg_drop=False, return_attn=False)

            target_ids = batch["target_ids"]
            valid_tokens = int((target_ids != self.pad_id).sum().item())
            if valid_tokens > 0:
                lm_nll = compute_lm_loss(
                    out.logits,
                    target_ids,
                    pad_id=self.pad_id,
                    label_smoothing=0.0,
                )
                lm_nll_sum += float(lm_nll.item()) * valid_tokens
                total_tokens += valid_tokens

            if (
                self.cfg.cvae.enabled
                and out.mu_q is not None
                and out.logvar_q is not None
                and out.mu_p is not None
                and out.logvar_p is not None
            ):
                kl_nats_sum += float(
                    compute_cvae_kl_nats(out.mu_q, out.logvar_q, out.mu_p, out.logvar_p).item()
                )
                kl_loss_sum += float(
                    compute_cvae_kl_loss(
                        out.mu_q,
                        out.logvar_q,
                        out.mu_p,
                        out.logvar_p,
                        free_bits=self.cfg.cvae.free_bits,
                    ).item()
                )

            if (
                self.cfg.multitask.use_pkd_head
                and out.pred_pkd is not None
                and "pkd" in batch
                and batch["pkd"] is not None
            ):
                pkd_loss_sum += float(compute_pkd_loss(out.pred_pkd, batch["pkd"]).item())
                pkd_preds.append(out.pred_pkd.reshape(-1).detach().cpu())
                pkd_targets.append(batch["pkd"].reshape(-1).detach().cpu())

            if (
                self.cfg.multitask.use_clip_head
                and out.clip_pocket is not None
                and out.clip_ligand is not None
                and out.clip_logit_scale is not None
            ):
                clip_loss_sum += float(
                    compute_drugclip_loss(
                        out.clip_pocket,
                        out.clip_ligand,
                        out.clip_logit_scale,
                    ).item()
                )
                clip_pockets.append(out.clip_pocket.detach().cpu())
                clip_ligands.append(out.clip_ligand.detach().cpu())

            if (
                self.cfg.multitask.use_prop_head
                and out.pred_props is not None
                and "ligand_props" in batch
                and batch["ligand_props"] is not None
            ):
                prop_loss_sum += float(
                    compute_property_loss(out.pred_props, batch["ligand_props"]).item()
                )

            num_batches += 1

        denom = float(max(1, num_batches))
        val_lm_loss = lm_nll_sum / float(max(1, total_tokens))
        val_ppl = math.exp(min(val_lm_loss, 20.0))
        val_kl_nats = kl_nats_sum / denom
        val_kl_loss = kl_loss_sum / denom
        val_pkd_loss = pkd_loss_sum / denom
        val_clip_loss = clip_loss_sum / denom
        val_prop_loss = prop_loss_sum / denom

        pkd_pearson, pkd_rmse = _compute_pearson_and_rmse(pkd_preds, pkd_targets)
        clip_r1, clip_r5 = _compute_clip_recall(clip_pockets, clip_ligands)

        # Cycle-invariant ELBO + multi-task validation objective (fixed kl_max_beta)
        val_objective = val_lm_loss
        if self.cfg.cvae.enabled and self.cfg.cvae.kl_max_beta > 0.0:
            val_objective += self.cfg.cvae.kl_max_beta * val_kl_loss
        if self.cfg.multitask.use_pkd_head and self.cfg.multitask.lambda_pkd > 0.0:
            val_objective += self.cfg.multitask.lambda_pkd * val_pkd_loss
        if self.cfg.multitask.use_clip_head and self.cfg.multitask.lambda_clip > 0.0:
            val_objective += self.cfg.multitask.lambda_clip * val_clip_loss
        if self.cfg.multitask.use_prop_head and self.cfg.multitask.lambda_prop > 0.0:
            val_objective += self.cfg.multitask.lambda_prop * val_prop_loss

        return {
            "val_objective": val_objective,
            "val_lm_loss": val_lm_loss,
            "val_ppl": val_ppl,
            "val_kl_nats": val_kl_nats,
            "val_kl_loss": val_kl_loss,
            "val_pkd_loss": val_pkd_loss,
            "val_pkd_pearson": pkd_pearson,
            "val_pkd_rmse": pkd_rmse,
            "val_clip_loss": val_clip_loss,
            "val_clip_r1": clip_r1,
            "val_clip_r5": clip_r5,
            "val_prop_loss": val_prop_loss,
        }

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Iterable[Mapping[str, Any]],
        output_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run full Stage-2 training loop, checkpointing ``best.pt`` on validation objective.

        Args:
            train_loader: Pocket-ligand training dataloader.
            val_loader: Pocket-ligand validation dataloader.
            output_dir: Directory for checkpoints and ``history.jsonl`` (defaults to
                ``cfg.train.output_dir``).
            resume_from: Optional checkpoint path to resume training state.

        Returns:
            Summary dictionary with best validation objective, checkpoint paths, and history.
        """
        out_dir = Path(output_dir if output_dir is not None else self.cfg.train.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = out_dir / "best.pt"
        last_ckpt_path = out_dir / "last.pt"
        history_path = out_dir / "history.jsonl"

        steps_per_epoch = len(train_loader) if isinstance(train_loader, Sized) else 1
        total_steps = max(1, steps_per_epoch * self.cfg.train.epochs)

        if self.optimizer is None:
            self.optimizer = self.build_optimizer()
        if self.scheduler is None:
            self.scheduler = build_cosine_warmup_scheduler(
                self.optimizer,
                total_steps=total_steps,
                warmup_ratio=self.cfg.train.warmup_ratio,
            )

        if resume_from is not None:
            self.load_checkpoint(resume_from)

        start_epoch = self.current_epoch + 1
        file_mode = "a" if resume_from is not None and history_path.exists() else "w"

        with history_path.open(file_mode, encoding="utf-8") as hist_file:
            for epoch in range(start_epoch, self.cfg.train.epochs + 1):
                t0 = time.perf_counter()
                train_metrics = self.train_epoch(
                    train_loader,
                    epoch=epoch,
                    total_steps=total_steps,
                )
                val_metrics = self.evaluate(val_loader)
                elapsed_sec = time.perf_counter() - t0

                self.current_epoch = epoch
                is_best = val_metrics["val_objective"] < self.best_val_objective
                if is_best:
                    self.best_val_objective = val_metrics["val_objective"]

                lr_by_group = {
                    str(g.get("name", f"group_{i}")): float(g["lr"])
                    for i, g in enumerate(self.optimizer.param_groups)
                }

                record: dict[str, Any] = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "elapsed_sec": elapsed_sec,
                    "is_best": is_best,
                    "lrs": lr_by_group,
                    **train_metrics,
                    **val_metrics,
                }
                self.history.append(record)
                hist_file.write(json.dumps(record) + "\n")
                hist_file.flush()

                if is_best:
                    self.save_checkpoint(best_ckpt_path)
                self.save_checkpoint(last_ckpt_path)

        return {
            "best_val_objective": self.best_val_objective,
            "best_ckpt_path": str(best_ckpt_path),
            "last_ckpt_path": str(last_ckpt_path),
            "history_path": str(history_path),
            "history": list(self.history),
        }


__all__ = ["PocketTrainer"]
