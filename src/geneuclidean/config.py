"""Typed Pydantic v2 configuration schemas, device selection, and deterministic seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictBaseModel(BaseModel):
    """Base schema rejecting unknown keys and validating field updates."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(_StrictBaseModel):
    """Dataset paths, 3D pocket graph construction, sequence format, and augmentation."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    metadata_dir: str = "data/metadata"
    pocket_cutoff: float = Field(
        default=10.0,
        gt=0.0,
        description="Pocket extraction shell radius in Angstroms from any ligand heavy atom.",
    )
    graph_cutoff: float = Field(
        default=5.5,
        gt=0.0,
        description="3D radius graph cutoff in Angstroms for equivariant message passing.",
    )
    max_neighbors: int = Field(default=24, ge=1)
    max_pocket_atoms: int = Field(default=384, ge=16)
    representation: Literal["smiles", "selfies", "safe"] = "smiles"
    randomize_smiles: bool = True
    max_seq_len: int = Field(default=128, ge=8)
    split_type: Literal["lp_pdbbind", "casf2016", "scaffold", "random"] = "lp_pdbbind"
    coord_jitter_std: float = Field(default=0.10, ge=0.0)
    atom_drop_prob: float = Field(default=0.05, ge=0.0, lt=1.0)


class EncoderConfig(_StrictBaseModel):
    """3D SE(3)/E(3)-equivariant protein pocket encoder configuration.

    Default ``node_in_dim = 52`` corresponds to the canonical per-atom pocket descriptor:
    element one-hot (10) + residue type one-hot (21) + backbone/sidechain role (5) +
    pharmacophore & invariant geometric scalars (16).
    """

    arch: Literal["painn", "egnn", "e3nn", "schnet"] = "painn"
    node_in_dim: int = Field(default=52, ge=1)
    hidden_dim: int = Field(default=192, ge=16)
    vector_dim: int = Field(default=64, ge=8)
    out_dim: int = Field(default=256, ge=16)
    num_layers: int = Field(default=4, ge=1)
    num_rbf: int = Field(default=24, ge=4)
    cutoff: float = Field(default=5.5, gt=0.0)
    e3nn_lmax: int = Field(default=1, ge=0, le=3)
    use_se3_chirality: bool = True
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class CVAEConfig(_StrictBaseModel):
    """Pocket-conditioned variational latent bridge for high structural diversity."""

    enabled: bool = True
    latent_dim: int = Field(default=64, ge=4)
    num_latent_tokens: int = Field(default=4, ge=1)
    kl_max_beta: float = Field(default=0.3, ge=0.0)
    kl_cycles: int = Field(default=4, ge=1)
    free_bits: float = Field(default=0.25, ge=0.0)


class DecoderConfig(_StrictBaseModel):
    """Autoregressive chemical language decoder configuration."""

    arch: Literal["transformer", "lstm"] = "transformer"
    vocab_size: int = Field(
        default=96,
        ge=4,
        description="Placeholder default overridden at runtime by the active tokenizer.",
    )
    d_model: int = Field(default=256, ge=16)
    n_heads: int = Field(default=8, ge=1)
    n_layers: int = Field(default=6, ge=1)
    d_ff: int = Field(default=688, ge=16)
    max_seq_len: int = Field(default=128, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_cross_attn: bool = True
    gated_cross_attn: bool = True
    cfg_drop_prob: float = Field(default=0.15, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_heads(self) -> DecoderConfig:
        if self.arch == "transformer" and self.d_model % self.n_heads != 0:
            msg = f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        return self


class MultiTaskConfig(_StrictBaseModel):
    """Joint binding affinity (pKd), DrugCLIP contrastive, and property supervision."""

    use_pkd_head: bool = True
    use_clip_head: bool = True
    use_prop_head: bool = True
    lambda_pkd: float = Field(default=0.5, ge=0.0)
    lambda_clip: float = Field(default=0.3, ge=0.0)
    lambda_prop: float = Field(default=0.2, ge=0.0)
    clip_dim: int = Field(default=128, ge=8)
    ligand_fp_dim: int = Field(default=1024, ge=64)


class TrainConfig(_StrictBaseModel):
    """Optimization, scheduling, differential learning rates, and checkpointing."""

    seed: int = 42
    device: str = "auto"
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=40, ge=1)
    lr: float = Field(default=5e-4, gt=0.0)
    encoder_lr: float = Field(default=5e-4, gt=0.0)
    decoder_lr: float = Field(default=1.5e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    warmup_ratio: float = Field(default=0.08, ge=0.0, lt=1.0)
    label_smoothing: float = Field(default=0.05, ge=0.0, lt=1.0)
    freeze_decoder_epochs: int = Field(default=2, ge=0)
    num_workers: int = Field(default=0, ge=0)
    pretrained_decoder_ckpt: str | None = None
    output_dir: str = "runs/default"


class SamplingConfig(_StrictBaseModel):
    """Inference decoding strategy, classifier-free guidance, and diversity controls."""

    strategy: Literal[
        "min_p",
        "top_p",
        "top_k",
        "temperature",
        "greedy",
        "typical",
        "diverse_beam",
    ] = "min_p"
    num_samples: int = Field(default=64, ge=1)
    max_len: int = Field(default=120, ge=4)
    temperature: float = Field(default=1.05, gt=0.0)
    min_p: float = Field(default=0.08, ge=0.0, le=1.0)
    top_p: float = Field(default=0.92, gt=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0)
    cfg_scale: float = Field(default=1.75, ge=0.0)
    latent_temp: float = Field(default=1.0, ge=0.0)
    beam_width: int = Field(default=16, ge=1)
    num_beam_groups: int = Field(default=8, ge=1)
    diversity_penalty: float = Field(default=0.7, ge=0.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0)
    syntax_guardrails: bool = False

    @model_validator(mode="after")
    def _validate_diverse_beam(self) -> SamplingConfig:
        if self.strategy == "diverse_beam" and self.beam_width % self.num_beam_groups != 0:
            msg = (
                f"beam_width ({self.beam_width}) must be divisible by "
                f"num_beam_groups ({self.num_beam_groups}) for diverse_beam sampling"
            )
            raise ValueError(msg)
        return self


class ExperimentConfig(_StrictBaseModel):
    """Top-level experiment configuration composing all GenEuclidean subsystems."""

    name: str = "geneuclidean-v2-default"
    description: str = "PaiNN-SE3 + Pretrained RoPE/SwiGLU Transformer + CVAE + Multi-Task SBDD"
    data: DataConfig = Field(default_factory=DataConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    cvae: CVAEConfig = Field(default_factory=CVAEConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    multitask: MultiTaskConfig = Field(default_factory=MultiTaskConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialize configuration to a clean YAML file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="python")
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
        return out_path

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an ExperimentConfig from a YAML file."""
        in_path = Path(path)
        with in_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            msg = f"Expected a YAML mapping in {in_path}, got {type(loaded).__name__}"
            raise ValueError(msg)
        return cls.model_validate(loaded)


def select_device(preferred: str = "auto") -> torch.device:
    """Select execution device (mps > cuda > cpu) and enforce float32 default dtype.

    Apple Silicon MPS does not support ``float64``; enforcing ``torch.float32`` globally
    prevents silent precision mismatches and guarantees cross-device numerical parity.
    """
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    choice = preferred.strip().lower()
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available on this host.")
        return torch.device("mps")

    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{preferred}' is not available on this host.")
        return torch.device(choice)

    if choice == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device specification: {preferred!r}")


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU, CUDA, MPS) RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


__all__ = [
    "CVAEConfig",
    "DataConfig",
    "DecoderConfig",
    "EncoderConfig",
    "ExperimentConfig",
    "MultiTaskConfig",
    "SamplingConfig",
    "TrainConfig",
    "seed_everything",
    "select_device",
]
