"""Multi-task gradient conflict diagnostics & adaptive intervention for 3D SBDD training.

Scientific motivation:
    Jointly training a shared 3D equivariant pocket encoder (``PaiNN-SE3``) on heterogeneous
    objectives — autoregressive molecule generation (``lm_nll``), binding affinity regression
    (``pkd``), contrastive pocket-ligand retrieval (``drugclip``), and physicochemical
    property supervision (``props``) — frequently suffers from **destructive gradient
    interference** (negative pairwise cosine similarity between task gradients).

    Directly inspired by ETH Zurich MODLAB's *NeurIPS* 2025 framework **AIM**
    (*Adaptive Intervention for Deep Multi-task Learning of Molecular Properties*,
    Minot & Schneider, ``arXiv:2509.25955``), this module provides:

    1. **Exact Task-Gradient Conflict Diagnostics:** Computes the ``T x T`` pairwise cosine
       similarity matrix ``C_ij = cos(g_i, g_j)``, negative-conflict pair fraction, and task
       gradient-norm imbalance ratio on shared encoder parameters.
    2. **Pluggable Gradient-Mediation Policies (``MTL_MEDIATOR_REGISTRY``):**
       - ``"scalarized"``: Standard fixed weighted sum baseline.
       - ``"pcgrad"``: Surgical projection of conflicting task gradients onto each other's
         normal planes (Yu et al., *NeurIPS* 2020).
       - ``"aim_policy"``: Learnable dynamic task-intervention policy matrix ``W_policy``
         coupled with lagging-task progress prioritization and geometric-stability entropy
         regularization, exposing ``W_policy`` as an interpretable inter-task diagnostic matrix.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.registry import Registry

MTL_MEDIATOR_REGISTRY: Registry[Callable[..., Any]] = Registry("mtl_mediator")


@dataclass(frozen=True)
class TaskGradientDiagnostics:
    """Interpretable snapshot of multi-task gradient conflicts and intervention weights."""

    task_names: tuple[str, ...]
    strategy: str
    grad_norms: tuple[float, ...]
    norm_imbalance_ratio: float
    cosine_matrix: tuple[tuple[float, ...], ...]
    conflict_pair_count: int
    total_pair_count: int
    conflict_fraction: float
    mean_pairwise_cosine: float
    policy_matrix: tuple[tuple[float, ...], ...] | None = None
    effective_task_weights: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics to a JSON-compatible dictionary."""
        return {
            "task_names": list(self.task_names),
            "strategy": self.strategy,
            "grad_norms": [round(x, 6) for x in self.grad_norms],
            "norm_imbalance_ratio": round(self.norm_imbalance_ratio, 4),
            "cosine_matrix": [[round(v, 4) for v in row] for row in self.cosine_matrix],
            "conflict_pair_count": self.conflict_pair_count,
            "total_pair_count": self.total_pair_count,
            "conflict_fraction": round(self.conflict_fraction, 4),
            "mean_pairwise_cosine": round(self.mean_pairwise_cosine, 4),
            "policy_matrix": (
                [[round(v, 4) for v in row] for row in self.policy_matrix]
                if self.policy_matrix is not None
                else None
            ),
            "effective_task_weights": [round(w, 4) for w in self.effective_task_weights],
        }


def flatten_task_grads(
    losses: Mapping[str, Tensor],
    shared_params: Sequence[nn.Parameter],
) -> tuple[list[str], list[Tensor]]:
    """Compute flat shared-parameter gradient vectors ``g_i`` for each active task loss."""
    params = [p for p in shared_params if p.requires_grad]
    if not params:
        raise ValueError(
            "shared_params must contain at least one parameter with requires_grad=True."
        )

    names = list(losses.keys())
    flat_grads: list[Tensor] = []
    for idx, name in enumerate(names):
        loss_val = losses[name]
        retain = idx < len(names) - 1
        grads = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=retain,
            allow_unused=True,
        )
        chunks = [
            (
                g.detach().reshape(-1).float()
                if g is not None
                else torch.zeros(p.numel(), device=p.device)
            )
            for g, p in zip(grads, params, strict=True)
        ]
        flat_grads.append(torch.cat(chunks, dim=0))
    return names, flat_grads


def compute_gradient_conflict_matrix(
    task_grads: Sequence[Tensor],
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, int, int]:
    """Compute exact pairwise cosine similarity matrix ``C[T, T]`` and conflict pair counts."""
    stacked = torch.stack([g.float() for g in task_grads], dim=0)
    norms = stacked.norm(dim=-1)
    unit = stacked / norms.unsqueeze(-1).clamp(min=eps)
    cos_mat = (unit @ unit.T).clamp(min=-1.0, max=1.0)

    n_tasks = len(task_grads)
    conflicts = 0
    total_pairs = 0
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            total_pairs += 1
            if float(cos_mat[i, j].item()) < 0.0:
                conflicts += 1
    return cos_mat, norms, conflicts, total_pairs


def mediate_scalarized(
    task_grads: Sequence[Tensor],
    task_weights: Sequence[float] | None = None,
) -> tuple[Tensor, list[float], Tensor | None]:
    """Standard weighted sum of task gradients (baseline with no conflict surgery)."""
    weights = (
        [float(w) for w in task_weights] if task_weights is not None else [1.0] * len(task_grads)
    )
    merged = sum(w * g for w, g in zip(weights, task_grads, strict=True))
    assert isinstance(merged, Tensor)
    return merged, weights, None


def mediate_pcgrad(
    task_grads: Sequence[Tensor],
    task_weights: Sequence[float] | None = None,
    eps: float = 1e-12,
) -> tuple[Tensor, list[float], Tensor | None]:
    """Surgical gradient conflict projection (PCGrad, Yu et al. NeurIPS 2020)."""
    weights = (
        [float(w) for w in task_weights] if task_weights is not None else [1.0] * len(task_grads)
    )
    scaled = [w * g.clone() for w, g in zip(weights, task_grads, strict=True)]
    n_tasks = len(scaled)
    projected: list[Tensor] = []
    for i in range(n_tasks):
        g_i = scaled[i].clone()
        for j in range(n_tasks):
            if i == j:
                continue
            g_j = scaled[j]
            dot = torch.dot(g_i, g_j)
            if float(dot.item()) < 0.0:
                denom = torch.dot(g_j, g_j).clamp(min=eps)
                g_i = g_i - (dot / denom) * g_j
        projected.append(g_i)
    merged = torch.stack(projected, dim=0).sum(dim=0)
    return merged, weights, None


class AdaptiveInterventionPolicy(nn.Module):
    """Learnable AIM-inspired gradient-intervention policy matrix (Minot & Schneider 2025).

    Maintains a differentiable ``T x T`` intervention logit matrix ``M`` whose row-softmax
    policy ``W_policy`` dynamically mediates inter-task gradient transfer while:
    - attenuating negative-cosine conflicting entries via conflict-aware masking,
    - prioritizing lagging tasks using EMA-tracked relative loss progress, and
    - exposing ``W_policy`` directly as an interpretable diagnostic matrix of task synergies.
    """

    def __init__(
        self,
        task_names: Sequence[str],
        *,
        temperature: float = 1.0,
        policy_lr: float = 0.05,
        lag_priority_alpha: float = 0.5,
        ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.task_names = tuple(task_names)
        n_tasks = len(self.task_names)
        if n_tasks < 1:
            raise ValueError("AdaptiveInterventionPolicy requires at least one task.")
        self.temperature = float(temperature)
        self.policy_lr = float(policy_lr)
        self.lag_priority_alpha = float(lag_priority_alpha)
        self.ema_decay = float(ema_decay)

        # Initialize near identity so early training preserves clean per-task gradients.
        init_logits = torch.eye(n_tasks, dtype=torch.float32) * 2.0
        self.policy_logits = nn.Parameter(init_logits)
        self.register_buffer("initial_losses", torch.zeros(n_tasks, dtype=torch.float32))
        self.register_buffer("ema_losses", torch.zeros(n_tasks, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def current_policy_matrix(self) -> Tensor:
        """Return row-stochastic ``[T, T]`` intervention policy matrix ``W_policy``."""
        return F.softmax(self.policy_logits / max(self.temperature, 1e-4), dim=-1)

    def step_and_mediate(
        self,
        task_grads: Sequence[Tensor],
        task_losses: Sequence[float],
        task_weights: Sequence[float] | None = None,
        eps: float = 1e-12,
    ) -> tuple[Tensor, list[float], Tensor]:
        """Update the adaptive policy using current conflict geometry & lag signals, then merge."""
        n_tasks = len(task_grads)
        device = task_grads[0].device
        loss_vec = torch.tensor([float(x) for x in task_losses], dtype=torch.float32, device=device)

        init_losses_buf: Tensor = self.initial_losses  # type: ignore[assignment]
        ema_losses_buf: Tensor = self.ema_losses  # type: ignore[assignment]
        initialized_buf: Tensor = self.initialized  # type: ignore[assignment]

        if not bool(initialized_buf.item()):
            init_losses_buf.copy_(loss_vec.clamp(min=1e-6))
            ema_losses_buf.copy_(loss_vec.clamp(min=1e-6))
            initialized_buf.fill_(True)
        else:
            ema_losses_buf.mul_(self.ema_decay).add_(loss_vec * (1.0 - self.ema_decay))

        # Relative lagging ratio: tasks that dropped less from initial loss get higher priority
        rel_progress = (ema_losses_buf / init_losses_buf.clamp(min=1e-6)).clamp(min=0.1, max=5.0)
        lag_weights = F.softmax(self.lag_priority_alpha * rel_progress, dim=0) * float(n_tasks)
        base_w = (
            torch.tensor([float(w) for w in task_weights], dtype=torch.float32, device=device)
            if task_weights is not None
            else torch.ones(n_tasks, dtype=torch.float32, device=device)
        )
        eff_weights = base_w * lag_weights

        # Conflict-aware policy update: reward positive synergy (C_ij > 0), suppress conflicts
        cos_mat, norms, _, _ = compute_gradient_conflict_matrix(task_grads, eps=eps)
        with torch.no_grad():
            synergy_signal = cos_mat.to(device=self.policy_logits.device)
            # Small geometric-stability pull toward identity prevents policy collapse
            identity = torch.eye(n_tasks, device=self.policy_logits.device)
            delta = synergy_signal + 0.15 * identity
            self.policy_logits.add_(self.policy_lr * delta)
            self.policy_logits.clamp_(min=-4.0, max=6.0)

        w_policy = self.current_policy_matrix().to(device=device)
        # Project conflicting components out before applying learned synergy mediation
        stacked = torch.stack([g.float() for g in task_grads], dim=0)
        unit_grads = stacked / norms.unsqueeze(-1).clamp(min=eps)
        mean_norm = norms.mean().clamp(min=eps)

        mediated_rows: list[Tensor] = []
        for i in range(n_tasks):
            g_i = stacked[i].clone()
            for j in range(n_tasks):
                if i == j:
                    continue
                if float(cos_mat[i, j].item()) < 0.0:
                    # Remove destructive opposing component along task j
                    dot_ij = torch.dot(g_i, stacked[j])
                    if float(dot_ij.item()) < 0.0:
                        g_i = (
                            g_i
                            - (dot_ij / torch.dot(stacked[j], stacked[j]).clamp(min=eps))
                            * stacked[j]
                        )
                else:
                    # Transfer constructive synergistic signal weighted by learned W_policy[i, j]
                    g_i = g_i + w_policy[i, j] * mean_norm * unit_grads[j]
            mediated_rows.append(eff_weights[i] * g_i)

        merged = torch.stack(mediated_rows, dim=0).sum(dim=0)
        return merged, [float(x) for x in eff_weights.detach().cpu().tolist()], w_policy.detach()


MTL_MEDIATOR_REGISTRY.register("scalarized", mediate_scalarized, override=True)
MTL_MEDIATOR_REGISTRY.register("pcgrad", mediate_pcgrad, override=True)


class GradientConflictMediator:
    """High-level multi-task gradient conflict analyzer & mediator for ``GenEuclideanModel``.

    Usage inside any PyTorch training step::

        mediator = GradientConflictMediator(
            task_names=("lm_nll", "pkd", "drugclip", "props"),
            strategy="aim_policy",
        )
        diag = mediator.backward(
            losses={"lm_nll": l_lm, "pkd": l_pkd, "drugclip": l_clip, "props": l_prop},
            shared_params=list(model.encoder.parameters()),
            task_weights={"lm_nll": 1.0, "pkd": 0.5, "drugclip": 0.3, "props": 0.2},
        )
    """

    def __init__(
        self,
        task_names: Sequence[str],
        *,
        strategy: str = "aim_policy",
        policy_lr: float = 0.05,
        lag_priority_alpha: float = 0.5,
    ) -> None:
        self.task_names = tuple(task_names)
        self.strategy = strategy.strip().lower()
        self.policy: AdaptiveInterventionPolicy | None = (
            AdaptiveInterventionPolicy(
                self.task_names,
                policy_lr=policy_lr,
                lag_priority_alpha=lag_priority_alpha,
            )
            if self.strategy == "aim_policy"
            else None
        )
        if self.strategy != "aim_policy" and self.strategy not in MTL_MEDIATOR_REGISTRY:
            raise ValueError(
                f"Unknown MTL mediation strategy {strategy!r}. "
                f"Expected 'aim_policy' or one of {MTL_MEDIATOR_REGISTRY.names()}."
            )

    def backward(
        self,
        losses: Mapping[str, Tensor],
        shared_params: Sequence[nn.Parameter],
        *,
        task_weights: Mapping[str, float] | None = None,
        non_shared_loss: Tensor | None = None,
    ) -> TaskGradientDiagnostics:
        """Compute task gradients, mediate conflicts on ``shared_params``, and set ``.grad``."""
        params = [p for p in shared_params if p.requires_grad]
        names, flat_grads = flatten_task_grads(losses, params)
        weights = [float(task_weights.get(n, 1.0)) if task_weights else 1.0 for n in names]
        loss_floats = [float(losses[n].detach().item()) for n in names]

        cos_mat, norms, conflicts, total_pairs = compute_gradient_conflict_matrix(flat_grads)
        if self.strategy == "aim_policy":
            assert self.policy is not None
            merged, eff_w, policy_mat = self.policy.step_and_mediate(
                flat_grads,
                loss_floats,
                task_weights=weights,
            )
        else:
            fn = MTL_MEDIATOR_REGISTRY.get(self.strategy)
            merged, eff_w, policy_mat = fn(flat_grads, task_weights=weights)

        if non_shared_loss is not None:
            non_shared_loss.backward()

        # Scatter mediated shared-parameter gradient back onto each parameter's ``.grad`` tensor
        offset = 0
        for p in params:
            numel = p.numel()
            chunk = merged[offset : offset + numel].view_as(p).to(device=p.device, dtype=p.dtype)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            offset += numel

        norm_list = [float(x) for x in norms.detach().cpu().tolist()]
        min_pos = min((x for x in norm_list if x > 1e-12), default=1e-12)
        imbalance = max(norm_list, default=0.0) / max(min_pos, 1e-12)
        off_diag_cos = [
            float(cos_mat[i, j].item()) for i in range(len(names)) for j in range(i + 1, len(names))
        ]
        mean_cos = sum(off_diag_cos) / max(len(off_diag_cos), 1)
        if math.isnan(mean_cos):
            mean_cos = 0.0

        return TaskGradientDiagnostics(
            task_names=tuple(names),
            strategy=self.strategy,
            grad_norms=tuple(norm_list),
            norm_imbalance_ratio=float(imbalance),
            cosine_matrix=tuple(
                tuple(float(v) for v in row) for row in cos_mat.detach().cpu().tolist()
            ),
            conflict_pair_count=conflicts,
            total_pair_count=total_pairs,
            conflict_fraction=float(conflicts / max(total_pairs, 1)),
            mean_pairwise_cosine=float(mean_cos),
            policy_matrix=(
                tuple(tuple(float(v) for v in row) for row in policy_mat.detach().cpu().tolist())
                if policy_mat is not None
                else None
            ),
            effective_task_weights=tuple(eff_w),
        )


__all__ = [
    "MTL_MEDIATOR_REGISTRY",
    "AdaptiveInterventionPolicy",
    "GradientConflictMediator",
    "TaskGradientDiagnostics",
    "compute_gradient_conflict_matrix",
    "flatten_task_grads",
    "mediate_pcgrad",
    "mediate_scalarized",
]
