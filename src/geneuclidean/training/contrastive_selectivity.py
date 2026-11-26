"""Hard-negative selectivity-aware contrastive loss for 3D pocket-ligand dual encoders.

Extends standard symmetric InfoNCE (``DrugCLIP``) with an explicit paralog/off-target
selectivity margin loss so a shared 3D pocket encoder learns to separate close homologous
receptors (e.g., ``DRD1`` vs ``DRD2``, ``MAOB`` vs ``MAOA``, ``BACE1`` vs ``BACE2``):

.. math::
    \\mathcal{L} = \\mathcal{L}_{\\text{InfoNCE}}(P_{\\text{on}}, L)
    + \\alpha \\cdot \\mathbb{E}[\\max(0, m - (v_{\\text{on}}^\\top u - v_{\\text{off}}^\\top u))]
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from geneuclidean.training.losses import compute_drugclip_loss


def compute_selectivity_contrastive_loss(
    clip_pocket_on: Tensor,
    clip_pocket_off: Tensor,
    clip_ligand: Tensor,
    logit_scale: Tensor,
    *,
    margin: float = 0.15,
    selectivity_weight: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Compute InfoNCE + hard-negative paralog selectivity margin loss."""
    p_on = F.normalize(clip_pocket_on, p=2, dim=-1, eps=1e-8)
    p_off = F.normalize(clip_pocket_off, p=2, dim=-1, eps=1e-8)
    lig = F.normalize(clip_ligand, p=2, dim=-1, eps=1e-8)

    base_clip = compute_drugclip_loss(p_on, lig, logit_scale)
    sim_on = (p_on * lig).sum(dim=-1)
    sim_off = (p_off * lig).sum(dim=-1)
    gap = sim_on - sim_off
    sel_loss = F.relu(float(margin) - gap).mean()
    total = base_clip + float(selectivity_weight) * sel_loss
    metrics = {
        "base_drugclip_loss": float(base_clip.detach().item()),
        "selectivity_margin_loss": float(sel_loss.detach().item()),
        "mean_sim_on": float(sim_on.detach().mean().item()),
        "mean_sim_off": float(sim_off.detach().mean().item()),
        "mean_selectivity_gap": float(gap.detach().mean().item()),
    }
    return total, metrics


__all__ = ["compute_selectivity_contrastive_loss"]
