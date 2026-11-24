"""Training objectives, Stage-1 prior pretraining, Stage-2 pocket trainer, and Stage-3 DPO."""

from __future__ import annotations

from geneuclidean.training.dpo import (
    PreferencePair,
    build_preference_pairs_from_mpo,
    compute_sequence_logps,
    run_dpo_alignment,
)
from geneuclidean.training.losses import (
    compute_cvae_kl_loss,
    compute_cvae_kl_nats,
    compute_dpo_loss,
    compute_drugclip_loss,
    compute_lm_loss,
    compute_multitask_loss,
    compute_pkd_loss,
    compute_property_loss,
    cyclical_kl_beta,
)
from geneuclidean.training.pretrain import (
    build_cosine_warmup_scheduler,
    evaluate_chemical_prior,
    pretrain_chemical_prior,
)
from geneuclidean.training.trainer import PocketTrainer

__all__ = [
    "PocketTrainer",
    "PreferencePair",
    "build_cosine_warmup_scheduler",
    "build_preference_pairs_from_mpo",
    "compute_cvae_kl_loss",
    "compute_cvae_kl_nats",
    "compute_dpo_loss",
    "compute_drugclip_loss",
    "compute_lm_loss",
    "compute_multitask_loss",
    "compute_pkd_loss",
    "compute_property_loss",
    "compute_sequence_logps",
    "cyclical_kl_beta",
    "evaluate_chemical_prior",
    "pretrain_chemical_prior",
    "run_dpo_alignment",
]
