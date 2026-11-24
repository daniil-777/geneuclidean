"""Multi-task, variational, contrastive, and preference-alignment loss functions.

Implements all training objectives for GenEuclidean v2:
- Autoregressive chemical language modeling loss with label smoothing and PAD masking.
- Free-bits conditional VAE KL divergence with Fu et al. (NAACL 2019) cyclical annealing.
- NaN-safe Huber loss for binding affinity (pKd) and auxiliary physicochemical properties.
- Symmetric DrugCLIP InfoNCE contrastive loss between 3D pockets and ligand fingerprints.
- Direct Preference Optimization (DPO) loss for Stage-3 MPO preference alignment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.config import ExperimentConfig

_MAX_LOG_SCALE = math.log(100.0)
_LOGVAR_CLAMP_MIN = -20.0
_LOGVAR_CLAMP_MAX = 20.0


def compute_lm_loss(
    logits: Tensor,
    target_ids: Tensor,
    pad_id: int = 0,
    label_smoothing: float = 0.05,
) -> Tensor:
    """Compute token-level autoregressive cross-entropy loss ignoring ``pad_id``.

    Args:
        logits: Unnormalized vocabulary logits of shape ``[B, L, V]`` (or ``[N, V]``).
        target_ids: Integer target token IDs of shape ``[B, L]`` (or ``[N]``).
        pad_id: Padding token index excluded from the loss.
        label_smoothing: Label smoothing factor in ``[0, 1)``.

    Returns:
        Scalar mean cross-entropy loss tensor on ``logits.device``.
    """
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1).to(device=logits.device, dtype=torch.long)

    valid_mask = flat_targets != pad_id
    if not bool(valid_mask.any().item()):
        return flat_logits.sum() * 0.0

    return F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
        reduction="mean",
    )


def _kl_per_dim(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
) -> Tensor:
    """Compute analytic per-dimension KL(N(mu_q, var_q) || N(mu_p, var_p))."""
    lv_q = logvar_q.clamp(min=_LOGVAR_CLAMP_MIN, max=_LOGVAR_CLAMP_MAX)
    lv_p = logvar_p.clamp(min=_LOGVAR_CLAMP_MIN, max=_LOGVAR_CLAMP_MAX)
    kl = 0.5 * (
        lv_p - lv_q + torch.exp(lv_q - lv_p) + (mu_q - mu_p).pow(2) * torch.exp(-lv_p) - 1.0
    )
    return kl.clamp(min=0.0)


def compute_cvae_kl_nats(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
) -> Tensor:
    """Compute raw (un-clamped) KL divergence in nats summed across latent dimensions.

    Args:
        mu_q: Posterior mean ``[B, D]``.
        logvar_q: Posterior log-variance ``[B, D]``.
        mu_p: Prior mean ``[B, D]``.
        logvar_p: Prior log-variance ``[B, D]``.

    Returns:
        Scalar mean KL divergence in nats across the batch.
    """
    kl = _kl_per_dim(mu_q, logvar_q, mu_p, logvar_p)
    return kl.sum(dim=-1).mean()


def compute_cvae_kl_loss(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
    free_bits: float = 0.25,
) -> Tensor:
    """Compute free-bits regularized Gaussian KL divergence between posterior and prior.

    Applies Kingma et al. (2016) free-bits clamping per latent dimension averaged over
    the batch so no latent dimension collapses while preserving flexibility across
    individual pockets.

    Args:
        mu_q: Posterior mean tensor of shape ``[B, D]`` (or ``[D]``).
        logvar_q: Posterior log-variance tensor of shape ``[B, D]`` (or ``[D]``).
        mu_p: Prior mean tensor of shape ``[B, D]`` (or ``[D]``).
        logvar_p: Prior log-variance tensor of shape ``[B, D]`` (or ``[D]``).
        free_bits: Minimum KL budget (in nats) per latent dimension.

    Returns:
        Scalar free-bits KL loss tensor.
    """
    kl_dim = _kl_per_dim(mu_q, logvar_q, mu_p, logvar_p)
    kl_dim_mean = kl_dim.mean(dim=0) if kl_dim.ndim >= 2 else kl_dim
    if free_bits > 0.0:
        kl_dim_mean = torch.clamp(kl_dim_mean, min=free_bits)
    return kl_dim_mean.sum()


def cyclical_kl_beta(
    step: int,
    total_steps: int,
    max_beta: float = 0.3,
    n_cycles: int = 4,
    ratio: float = 0.5,
) -> float:
    """Compute Fu et al. (NAACL 2019) cyclical KL annealing weight ``beta``.

    Splits training into ``n_cycles`` equal cycles. Within each cycle, ``beta`` increases
    linearly from ``0.0`` to ``max_beta`` over the first ``ratio`` fraction of the cycle,
    then remains constant at ``max_beta`` for the remainder of the cycle.

    Args:
        step: Current 0-indexed optimization step.
        total_steps: Total number of optimization steps across training.
        max_beta: Maximum KL weight plateau.
        n_cycles: Number of annealing cycles.
        ratio: Fraction of each cycle spent linearly increasing ``beta`` in ``(0, 1]``.

    Returns:
        KL weight ``beta`` in ``[0.0, max_beta]``.
    """
    if max_beta <= 0.0:
        return 0.0
    if total_steps <= 0 or n_cycles <= 0:
        return float(max_beta)
    if step <= 0:
        return 0.0
    if step >= total_steps:
        return float(max_beta)

    eff_ratio = min(max(ratio, 1e-6), 1.0)
    period = total_steps / float(n_cycles)
    cycle_pos = (step % period) / period
    if cycle_pos >= eff_ratio:
        return float(max_beta)
    return float(max_beta * (cycle_pos / eff_ratio))


def compute_pkd_loss(
    pred_pkd: Tensor,
    target_pkd: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Compute NaN-safe Huber loss for binding affinity (pKd) regression.

    Missing labels (``NaN`` or ``inf``) are masked out *before* loss computation so
    ``NaN`` gradients can never propagate into model parameters.

    Args:
        pred_pkd: Predicted affinity values of shape ``[B]`` or ``[B, 1]``.
        target_pkd: Ground-truth pKd values of shape ``[B]`` or ``[B, 1]`` (may hold NaNs).
        delta: Huber transition threshold.

    Returns:
        Scalar Huber loss tensor (exact ``0.0`` with valid grad graph if all targets are NaN).
    """
    pred = pred_pkd.reshape(-1)
    target = target_pkd.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    valid = torch.isfinite(target)
    if not bool(valid.any().item()):
        return pred.sum() * 0.0
    return F.huber_loss(pred[valid], target[valid], reduction="mean", delta=delta)


def compute_drugclip_loss(
    clip_pocket: Tensor,
    clip_ligand: Tensor,
    logit_scale: Tensor,
) -> Tensor:
    """Compute symmetric DrugCLIP InfoNCE contrastive loss across a batch of complexes.

    Args:
        clip_pocket: Pocket projection embeddings of shape ``[B, D]``.
        clip_ligand: Ligand projection embeddings of shape ``[B, D]``.
        logit_scale: Learnable log-temperature scalar tensor (clamped to ``log(100)``).

    Returns:
        Scalar symmetric InfoNCE loss tensor.
    """
    batch_size = clip_pocket.size(0)
    if batch_size < 2:
        return (clip_pocket.sum() + clip_ligand.sum() + logit_scale.sum()) * 0.0

    p_norm = F.normalize(clip_pocket, p=2, dim=-1, eps=1e-8)
    l_norm = F.normalize(clip_ligand, p=2, dim=-1, eps=1e-8)

    scale = (
        logit_scale.to(device=p_norm.device, dtype=p_norm.dtype)
        .clamp(
            min=-_MAX_LOG_SCALE,
            max=_MAX_LOG_SCALE,
        )
        .exp()
    )

    logits_p2l = scale * (p_norm @ l_norm.transpose(0, 1))
    logits_l2p = logits_p2l.transpose(0, 1)
    labels = torch.arange(batch_size, device=logits_p2l.device, dtype=torch.long)

    loss_p2l = F.cross_entropy(logits_p2l, labels)
    loss_l2p = F.cross_entropy(logits_l2p, labels)
    return 0.5 * (loss_p2l + loss_l2p)


def compute_property_loss(
    pred_props: Tensor,
    target_props: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Compute NaN-safe Huber loss over auxiliary standardized ligand properties.

    Args:
        pred_props: Predicted standardized ligand properties ``[B, P]``.
        target_props: Target standardized ligand properties ``[B, P]`` (may contain NaNs).
        delta: Huber transition threshold.

    Returns:
        Scalar Huber loss tensor (exact ``0.0`` with valid grad graph if no valid entries).
    """
    target = target_props.to(device=pred_props.device, dtype=pred_props.dtype)
    valid = torch.isfinite(target)
    if not bool(valid.any().item()):
        return pred_props.sum() * 0.0
    return F.huber_loss(pred_props[valid], target[valid], reduction="mean", delta=delta)


def compute_multitask_loss(
    out: Any,
    batch: Mapping[str, Any],
    cfg: ExperimentConfig,
    kl_beta: float,
    pad_id: int = 0,
) -> tuple[Tensor, dict[str, float]]:
    """Combine LM, CVAE KL, pKd, DrugCLIP, and auxiliary property losses.

    Args:
        out: ``ModelOutput``-compatible object from ``GenEuclideanModel.forward``.
        batch: Collated pocket-ligand training batch mapping.
        cfg: Full ``ExperimentConfig``.
        kl_beta: Current cyclical KL annealing weight.
        pad_id: Padding token ID for LM cross-entropy.

    Returns:
        Tuple ``(total_loss, metrics_dict)`` where ``total_loss`` is a differentiable scalar
        tensor and ``metrics_dict`` contains detached Python floats for logging.
    """
    lm_loss = compute_lm_loss(
        out.logits,
        batch["target_ids"],
        pad_id=pad_id,
        label_smoothing=cfg.train.label_smoothing,
    )
    total_loss = lm_loss
    lm_val = float(lm_loss.detach().item())
    metrics: dict[str, float] = {
        "lm_loss": lm_val,
        "ppl": math.exp(min(lm_val, 20.0)),
        "kl_beta": float(kl_beta),
        "kl_loss": 0.0,
        "kl_nats": 0.0,
        "pkd_loss": 0.0,
        "clip_loss": 0.0,
        "prop_loss": 0.0,
    }

    if (
        cfg.cvae.enabled
        and out.mu_q is not None
        and out.logvar_q is not None
        and out.mu_p is not None
        and out.logvar_p is not None
    ):
        kl_loss = compute_cvae_kl_loss(
            out.mu_q,
            out.logvar_q,
            out.mu_p,
            out.logvar_p,
            free_bits=cfg.cvae.free_bits,
        )
        kl_nats = compute_cvae_kl_nats(out.mu_q, out.logvar_q, out.mu_p, out.logvar_p)
        total_loss = total_loss + float(kl_beta) * kl_loss
        metrics["kl_loss"] = float(kl_loss.detach().item())
        metrics["kl_nats"] = float(kl_nats.detach().item())

    if (
        cfg.multitask.use_pkd_head
        and cfg.multitask.lambda_pkd > 0.0
        and out.pred_pkd is not None
        and "pkd" in batch
        and batch["pkd"] is not None
    ):
        pkd_loss = compute_pkd_loss(out.pred_pkd, batch["pkd"])
        total_loss = total_loss + cfg.multitask.lambda_pkd * pkd_loss
        metrics["pkd_loss"] = float(pkd_loss.detach().item())

    if (
        cfg.multitask.use_clip_head
        and cfg.multitask.lambda_clip > 0.0
        and out.clip_pocket is not None
        and out.clip_ligand is not None
        and out.clip_logit_scale is not None
    ):
        clip_loss = compute_drugclip_loss(
            out.clip_pocket,
            out.clip_ligand,
            out.clip_logit_scale,
        )
        total_loss = total_loss + cfg.multitask.lambda_clip * clip_loss
        metrics["clip_loss"] = float(clip_loss.detach().item())

    if (
        cfg.multitask.use_prop_head
        and cfg.multitask.lambda_prop > 0.0
        and out.pred_props is not None
        and "ligand_props" in batch
        and batch["ligand_props"] is not None
    ):
        prop_loss = compute_property_loss(out.pred_props, batch["ligand_props"])
        total_loss = total_loss + cfg.multitask.lambda_prop * prop_loss
        metrics["prop_loss"] = float(prop_loss.detach().item())

    metrics["total_loss"] = float(total_loss.detach().item())
    return total_loss, metrics


def compute_dpo_loss(
    policy_logps_pos: Tensor,
    policy_logps_neg: Tensor,
    ref_logps_pos: Tensor,
    ref_logps_neg: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Compute Direct Preference Optimization (DPO) loss (Rafailov et al., NeurIPS 2023).

    Args:
        policy_logps_pos: Sequence log-probabilities of chosen ligands under policy ``[B]``.
        policy_logps_neg: Sequence log-probabilities of rejected ligands under policy ``[B]``.
        ref_logps_pos: Sequence log-probabilities of chosen ligands under frozen ref ``[B]``.
        ref_logps_neg: Sequence log-probabilities of rejected ligands under frozen ref ``[B]``.
        beta: KL regularization temperature controlling deviation from the reference model.

    Returns:
        Scalar DPO loss tensor.
    """
    pi_logratios = policy_logps_pos - policy_logps_neg
    ref_logratios = ref_logps_pos - ref_logps_neg
    logits = float(beta) * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


__all__ = [
    "compute_cvae_kl_loss",
    "compute_cvae_kl_nats",
    "compute_dpo_loss",
    "compute_drugclip_loss",
    "compute_lm_loss",
    "compute_multitask_loss",
    "compute_pkd_loss",
    "compute_property_loss",
    "cyclical_kl_beta",
]
