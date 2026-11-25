"""Two-phase 3D-to-LLM trainer, pocket feature caching, and LoRA RPO-DPO alignment.

Implements the GenEuclidean v3.0 training pipeline for conditioning pretrained chemical
foundation transformers (``GP-MoLFormer-Uniq``, ``entropy/gpt2_zinc_87m``) and compact
local general LLMs (``Qwen2.5-0.5B``, ``SmolLM2-360M``) on frozen ``PaiNN-SE3`` 3D
protein pocket representations:

1. ``precompute_pocket_cache``: Runs the frozen 3D pocket encoder ONCE over train/val
   pockets and stores compact CPU tensors (trimmed to valid atoms), eliminating 3D GNN
   overhead during Phase A and Phase B LLM fine-tuning.
2. ``PocketLLMTrainer``:
   - Dynamic SMILES randomization on every epoch fetch for SMILES-native backbones
     (``is_smiles_native=True``), and ``"SMILES: <smi>\\n"`` formatting with strict loss
     masking on prompt tokens for general LLMs (``is_smiles_native=False``).
   - Phase A (Projector Warmup): freezes LoRA & base LLM; trains ONLY the soft-prefix
     bridge (``perceiver``, ``cvae_proj``, ``global_proj``, ``prefix_norm``,
     ``null_prefix``) with cosine warmup and gradient clipping.
   - Phase B (Joint Perceiver + LoRA Fine-Tuning): unfreezes LoRA with differential
     learning rates (``lora_lr`` vs ``phase_b_projector_lr``), CFG pocket dropout,
     optional light CVAE KL regularization, per-epoch validation perplexity tracking,
     and compact adapter-only checkpointing (``best_adapter.pt``).
3. ``run_llm_rpo_dpo``: On-policy MPO preference pair sampling with fast KV-cached
   generation, frozen reference log-prob precomputation, strict non-LoRA parameter
   freezing (preserving ``PaiNN-SE3``, ``Perceiver``, ``pKd``, and ``DrugCLIP``
   bit-for-bit), and length-normalized RPO-DPO alignment.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything, select_device
from geneuclidean.data.featurizer import LIGAND_FP_DIM
from geneuclidean.data.tokenizer import randomize_smiles
from geneuclidean.models.encoders.rbf import PocketEncoderOutput
from geneuclidean.sampling.rerank import compute_mpo_scores
from geneuclidean.training.dpo import PreferencePair, build_preference_pairs_from_mpo
from geneuclidean.training.losses import compute_cvae_kl_loss, compute_dpo_loss
from geneuclidean.training.pretrain import build_cosine_warmup_scheduler

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

_BRIDGE_PREFIXES: tuple[str, ...] = (
    "perceiver.",
    "cvae_proj.",
    "global_proj.",
    "prefix_norm.",
    "null_prefix",
)

_FROZEN_3D_PREFIXES: tuple[str, ...] = (
    "encoder.",
    "cvae.",
    "heads.",
)

_LLM_BACKBONE_PREFIXES: tuple[str, ...] = (
    "llm.",
    "backbone.",
)


@dataclass(frozen=True)
class CachedPocketRecord:
    """Compact CPU representation of one pre-encoded 3D protein pocket and ligand."""

    pdb_id: str
    node_emb: Tensor  # [N_valid, D] float32 CPU
    pooled_emb: Tensor  # [D] float32 CPU
    atom_mask: Tensor  # [N_valid] bool CPU (all True)
    ligand_fp: Tensor | None  # [fp_dim] float32 CPU
    canonical_smiles: str
    pkd: float = float("nan")


def _is_lora_param(name: str) -> bool:
    """Return True if parameter name belongs to a PEFT LoRA adapter weight."""
    lower = name.lower()
    return "lora_" in lower or "modules_to_save" in lower


def _is_bridge_param(name: str) -> bool:
    """Return True if parameter belongs to the 3D-to-LLM soft-prefix bridge."""
    return name.startswith(_BRIDGE_PREFIXES)


def _is_llm_backbone_param(name: str) -> bool:
    """Return True if parameter resides inside the HuggingFace LLM backbone."""
    return name.startswith(_LLM_BACKBONE_PREFIXES)


def _extract_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Safely read attribute or mapping key ``name`` from a model output object."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_backbone_meta(model: Any) -> dict[str, Any]:
    """Extract tokenizer and backbone metadata from ``PocketConditionedLLM``."""
    backbone = getattr(model, "backbone_meta", getattr(model, "backbone", None))
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None and backbone is not None:
        tokenizer = getattr(backbone, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Model must expose a HuggingFace tokenizer via model.tokenizer.")

    is_smiles_native = bool(
        getattr(
            model,
            "is_smiles_native",
            getattr(backbone, "is_smiles_native", True) if backbone is not None else True,
        )
    )
    prompt_prefix = str(
        getattr(
            model,
            "prompt_prefix",
            getattr(backbone, "prompt_prefix", "") if backbone is not None else "",
        )
    )
    bos_id = getattr(
        model,
        "bos_token_id",
        getattr(backbone, "bos_token_id", getattr(tokenizer, "bos_token_id", None))
        if backbone is not None
        else getattr(tokenizer, "bos_token_id", None),
    )
    eos_id = getattr(
        model,
        "eos_token_id",
        getattr(backbone, "eos_token_id", getattr(tokenizer, "eos_token_id", None))
        if backbone is not None
        else getattr(tokenizer, "eos_token_id", None),
    )
    pad_id = getattr(
        model,
        "pad_token_id",
        getattr(backbone, "pad_token_id", getattr(tokenizer, "pad_token_id", None))
        if backbone is not None
        else getattr(tokenizer, "pad_token_id", None),
    )

    if eos_id is None:
        eos_id = getattr(tokenizer, "EOS_ID", 0)
    eos_int = int(eos_id) if eos_id is not None else 0
    pad_int = int(pad_id) if pad_id is not None else eos_int

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "is_smiles_native": is_smiles_native,
        "prompt_prefix": prompt_prefix,
        "bos_token_id": int(bos_id) if bos_id is not None else None,
        "eos_token_id": eos_int,
        "pad_token_id": pad_int,
    }


def _hf_encode_no_special(tokenizer: Any, text: str) -> list[int]:
    """Encode raw text to token IDs without inserting automatic special tokens."""
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [int(t) for t in ids]
        except TypeError:
            ids = tokenizer.encode(text)
            return [int(t) for t in ids]
    encoded = tokenizer(text, add_special_tokens=False)
    raw = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return [int(t) for t in raw]


def tokenize_smiles_for_llm(
    smiles: str,
    *,
    tokenizer: Any,
    is_smiles_native: bool,
    prompt_prefix: str = "",
    bos_token_id: int | None = None,
    eos_token_id: int = 1,
    max_seq_len: int = 128,
) -> tuple[list[int], list[int]]:
    """Format and tokenize a SMILES string into ``(input_ids, labels)`` for prefix LM training.

    - For ``is_smiles_native=True`` backbones (e.g. ``GP-MoLFormer-Uniq``, ``gpt2_zinc_87m``):
      builds ``[BOS, *smi_tokens, EOS]`` with ``labels = [-100, *smi_tokens, EOS]`` so the
      loss is computed over every SMILES token + EOS starting right after ``[prefix, BOS]``.
    - For ``is_smiles_native=False`` general LLMs (e.g. ``Qwen2.5-0.5B``, ``SmolLM2-360M``):
      formats as ``"<prompt_prefix><smiles>\\n"`` + ``EOS`` (default prompt ``"SMILES: "``),
      encoding prompt and SMILES separately to avoid BPE merges across the prompt boundary
      and masking all prompt tokens with ``-100`` in ``labels``.
    """
    cleaned = smiles.strip()
    eff_max = max(4, int(max_seq_len))

    if is_smiles_native:
        start_id = int(bos_token_id) if bos_token_id is not None else int(eos_token_id)
        smi_ids = _hf_encode_no_special(tokenizer, cleaned)
        max_inner = max(1, eff_max - 2)
        if len(smi_ids) > max_inner:
            smi_ids = smi_ids[:max_inner]
        input_ids = [start_id, *smi_ids, int(eos_token_id)]
        labels = [-100, *smi_ids, int(eos_token_id)]
        return input_ids, labels

    eff_prompt = prompt_prefix if prompt_prefix else "SMILES: "
    prompt_ids = _hf_encode_no_special(tokenizer, eff_prompt)
    if not prompt_ids:
        start_id = int(bos_token_id) if bos_token_id is not None else int(eos_token_id)
        prompt_ids = [start_id]

    smi_ids = _hf_encode_no_special(tokenizer, f"{cleaned}\n")
    max_inner = max(1, eff_max - len(prompt_ids) - 1)
    if len(smi_ids) > max_inner:
        smi_ids = smi_ids[:max_inner]

    input_ids = [*prompt_ids, *smi_ids, int(eos_token_id)]
    labels = [*([-100] * len(prompt_ids)), *smi_ids, int(eos_token_id)]
    return input_ids, labels


def build_pocket_batches_from_records(
    records: Mapping[str, Mapping[str, Any]],
    pdb_ids: Sequence[str],
    *,
    batch_size: int = 32,
    fp_dim: int = LIGAND_FP_DIM,
) -> list[dict[str, Any]]:
    """Build padded pocket batches directly from ``pdbbind_pockets.pt`` cache records.

    Avoids requiring a v2 ``MolecularTokenizer`` when precomputing frozen ``PaiNN-SE3``
    pocket embeddings for v3 LLM training.
    """
    valid_ids = [pid for pid in pdb_ids if pid in records]
    batches: list[dict[str, Any]] = []
    step = max(1, int(batch_size))

    for start in range(0, len(valid_ids), step):
        chunk_ids = valid_ids[start : start + step]
        chunk_recs = [records[pid] for pid in chunk_ids]
        bsz = len(chunk_recs)
        max_atoms = max(int(np.asarray(r["coords"]).shape[0]) for r in chunk_recs)
        node_dim = int(np.asarray(chunk_recs[0]["node_feats"]).shape[-1])

        coords = torch.zeros((bsz, max_atoms, 3), dtype=torch.float32)
        node_feats = torch.zeros((bsz, max_atoms, node_dim), dtype=torch.float32)
        atom_mask = torch.zeros((bsz, max_atoms), dtype=torch.bool)
        ligand_fp = torch.zeros((bsz, fp_dim), dtype=torch.float32)
        pkd = torch.full((bsz,), float("nan"), dtype=torch.float32)
        smiles_list: list[str] = []

        for i, rec in enumerate(chunk_recs):
            c = np.asarray(rec["coords"], dtype=np.float32)
            nf = np.asarray(rec["node_feats"], dtype=np.float32)
            n_a = int(c.shape[0])
            coords[i, :n_a] = torch.from_numpy(c)
            node_feats[i, :n_a] = torch.from_numpy(nf)
            atom_mask[i, :n_a] = True

            if "ligand_fp" in rec and rec["ligand_fp"] is not None:
                fp_t = torch.as_tensor(rec["ligand_fp"], dtype=torch.float32).reshape(-1)
                ligand_fp[i, : min(fp_dim, fp_t.numel())] = fp_t[:fp_dim]
            elif "ligand_fp_packed" in rec and rec["ligand_fp_packed"] is not None:
                fp_unpacked = np.unpackbits(rec["ligand_fp_packed"])[:fp_dim].astype(np.float32)
                ligand_fp[i] = torch.from_numpy(fp_unpacked)

            if "pkd" in rec and rec["pkd"] is not None:
                pkd[i] = float(rec["pkd"])
            smiles_list.append(str(rec["canonical_smiles"]))

        batches.append(
            {
                "pdb_ids": list(chunk_ids),
                "coords": coords,
                "node_feats": node_feats,
                "atom_mask": atom_mask,
                "ligand_fp": ligand_fp,
                "pkd": pkd,
                "canonical_smiles": smiles_list,
            }
        )
    return batches


@torch.no_grad()
def precompute_pocket_cache(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
) -> list[CachedPocketRecord]:
    """Run the frozen 3D pocket encoder once over all pockets and cache CPU embeddings.

    Stores valid-atom-trimmed ``float32`` CPU tensors ``(node_emb, pooled_emb, atom_mask,
    ligand_fp, canonical_smiles)`` so Phase A and Phase B incur zero 3D GNN overhead.
    """
    eff_device = select_device(device) if isinstance(device, str) else device
    was_training = model.training
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        encoder.to(eff_device)
        encoder.eval()
    else:
        model.to(eff_device)
        model.eval()

    cached: list[CachedPocketRecord] = []
    try:
        for batch in loader:
            coords = torch.as_tensor(batch["coords"], dtype=torch.float32)
            node_feats = torch.as_tensor(batch["node_feats"], dtype=torch.float32)
            unbatched = coords.ndim == 2
            if unbatched:
                coords = coords.unsqueeze(0)
                node_feats = node_feats.unsqueeze(0)

            bsz, max_atoms = int(coords.shape[0]), int(coords.shape[1])
            if "atom_mask" in batch and batch["atom_mask"] is not None:
                atom_mask = torch.as_tensor(batch["atom_mask"], dtype=torch.bool)
                if atom_mask.ndim == 1:
                    atom_mask = atom_mask.unsqueeze(0)
            else:
                atom_mask = torch.ones((bsz, max_atoms), dtype=torch.bool)

            pocket_out: PocketEncoderOutput = model.encode_pocket(  # type: ignore[operator]
                coords.to(device=eff_device, dtype=torch.float32),
                node_feats.to(device=eff_device, dtype=torch.float32),
                atom_mask.to(device=eff_device, dtype=torch.bool),
            )
            node_emb_cpu = pocket_out.node_emb.detach().cpu().float()
            pooled_cpu = pocket_out.pooled_emb.detach().cpu().float()
            mask_cpu = atom_mask.detach().cpu().bool()

            raw_fp = batch.get("ligand_fp")
            fp_cpu: Tensor | None = None
            if raw_fp is not None:
                fp_t = torch.as_tensor(raw_fp, dtype=torch.float32)
                if fp_t.ndim == 1:
                    fp_t = fp_t.unsqueeze(0)
                fp_cpu = fp_t.detach().cpu().float()

            raw_smiles = batch.get("canonical_smiles")
            if isinstance(raw_smiles, str):
                smiles_seq = [raw_smiles]
            elif raw_smiles is not None:
                smiles_seq = [str(s) for s in raw_smiles]
            else:
                smiles_seq = [""] * bsz

            raw_ids = batch.get("pdb_ids", batch.get("pdb_id"))
            if isinstance(raw_ids, str):
                id_seq = [raw_ids]
            elif raw_ids is not None:
                id_seq = [str(p) for p in raw_ids]
            else:
                id_seq = [f"pocket_{len(cached) + i}" for i in range(bsz)]

            raw_pkd = batch.get("pkd")
            pkd_vals: list[float]
            if raw_pkd is None:
                pkd_vals = [float("nan")] * bsz
            elif isinstance(raw_pkd, (int, float)):
                pkd_vals = [float(raw_pkd)] * bsz
            else:
                pkd_t = torch.as_tensor(raw_pkd, dtype=torch.float32).reshape(-1)
                pkd_vals = [float(v.item()) for v in pkd_t]

            for i in range(bsz):
                valid = mask_cpu[i]
                trimmed_nodes = node_emb_cpu[i][valid].contiguous()
                n_valid = int(trimmed_nodes.shape[0])
                cached.append(
                    CachedPocketRecord(
                        pdb_id=id_seq[i],
                        node_emb=trimmed_nodes,
                        pooled_emb=pooled_cpu[i].contiguous(),
                        atom_mask=torch.ones((n_valid,), dtype=torch.bool),
                        ligand_fp=fp_cpu[i].contiguous() if fp_cpu is not None else None,
                        canonical_smiles=smiles_seq[i],
                        pkd=pkd_vals[i] if i < len(pkd_vals) else float("nan"),
                    )
                )
    finally:
        if was_training:
            model.train()

    return cached


class CachedPocketLLMDataset(Dataset[dict[str, Any]]):
    """Dataset over pre-encoded pockets with per-epoch dynamic SMILES randomization."""

    def __init__(
        self,
        records: Sequence[CachedPocketRecord],
        *,
        tokenizer: Any,
        is_smiles_native: bool,
        prompt_prefix: str = "",
        bos_token_id: int | None = None,
        eos_token_id: int = 1,
        max_seq_len: int = 128,
        randomize: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.records = list(records)
        self.tokenizer = tokenizer
        self.is_smiles_native = bool(is_smiles_native)
        self.prompt_prefix = str(prompt_prefix)
        self.bos_token_id = bos_token_id
        self.eos_token_id = int(eos_token_id)
        self.max_seq_len = int(max_seq_len)
        self.randomize = bool(randomize and is_smiles_native)
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        smi = rec.canonical_smiles
        if self.randomize:
            smi = randomize_smiles(smi, rng=self._rng)

        input_ids, labels = tokenize_smiles_for_llm(
            smi,
            tokenizer=self.tokenizer,
            is_smiles_native=self.is_smiles_native,
            prompt_prefix=self.prompt_prefix,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_seq_len=self.max_seq_len,
        )
        return {
            "pdb_id": rec.pdb_id,
            "node_emb": rec.node_emb,
            "pooled_emb": rec.pooled_emb,
            "atom_mask": rec.atom_mask,
            "ligand_fp": rec.ligand_fp,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "canonical_smiles": rec.canonical_smiles,
        }


def collate_cached_pocket_batch(
    samples: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    """Collate pre-encoded pocket records and tokenized SMILES into padded batch tensors."""
    bsz = len(samples)
    max_atoms = max(int(s["node_emb"].shape[0]) for s in samples)
    emb_dim = int(samples[0]["node_emb"].shape[-1])
    max_seq = max(int(s["input_ids"].shape[0]) for s in samples)

    node_emb = torch.zeros((bsz, max_atoms, emb_dim), dtype=torch.float32)
    pooled_emb = torch.stack([s["pooled_emb"] for s in samples], dim=0).float()
    atom_mask = torch.zeros((bsz, max_atoms), dtype=torch.bool)

    input_ids = torch.full((bsz, max_seq), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_seq), dtype=torch.long)
    labels = torch.full((bsz, max_seq), -100, dtype=torch.long)

    has_fp = all(s.get("ligand_fp") is not None for s in samples)
    ligand_fp: Tensor | None = None
    if has_fp:
        ligand_fp = torch.stack([s["ligand_fp"] for s in samples], dim=0).float()

    pdb_ids: list[str] = []
    canonical_smiles: list[str] = []
    for i, item in enumerate(samples):
        n_a = int(item["node_emb"].shape[0])
        n_t = int(item["input_ids"].shape[0])
        node_emb[i, :n_a] = item["node_emb"]
        atom_mask[i, :n_a] = item["atom_mask"][:n_a]
        input_ids[i, :n_t] = item["input_ids"]
        attention_mask[i, :n_t] = 1
        labels[i, :n_t] = item["labels"]
        pdb_ids.append(str(item["pdb_id"]))
        canonical_smiles.append(str(item["canonical_smiles"]))

    pocket_out = PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=None)
    return {
        "pdb_ids": pdb_ids,
        "pocket_out": pocket_out,
        "atom_mask": atom_mask,
        "ligand_fp": ligand_fp,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "canonical_smiles": canonical_smiles,
    }


def _split_decay_groups(
    named_params: Sequence[tuple[str, nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    group_prefix: str,
) -> list[dict[str, Any]]:
    """Create AdamW decay / no-decay parameter groups (excluding biases and norms)."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "name": f"{group_prefix}_decay",
                "params": decay,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            }
        )
    if no_decay:
        groups.append(
            {
                "name": f"{group_prefix}_no_decay",
                "params": no_decay,
                "lr": float(lr),
                "weight_decay": 0.0,
            }
        )
    return groups


def _compute_llm_sequence_logps(
    model: nn.Module,
    *,
    pocket_out: PocketEncoderOutput,
    atom_mask: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    length_normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute exact float32 sequence log-probabilities on valid SMILES target tokens.

    Works seamlessly whether ``model.forward(...).logits`` has shape ``[B, L, V]`` or
    ``[B, M + L, V]`` (prefixed), and operates in ``float32`` even for ``bfloat16`` LLMs.

    Returns:
        Tuple ``(selected_logps[B], per_token_logps[B])``.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        pocket_out=pocket_out,
        atom_mask=atom_mask,
        ligand_fp=None,
        cfg_drop=False,
        use_posterior=False,
        return_attn=False,
    )
    logits = _extract_attr_or_key(out, "logits")
    if not isinstance(logits, Tensor):
        raise TypeError(f"Expected Tensor logits from model forward, got {type(logits).__name__}")

    seq_len = int(input_ids.size(1))
    if int(logits.size(1)) > seq_len:
        prefix_len = int(logits.size(1)) - seq_len
        tok_logits = logits[:, prefix_len:, :]
    else:
        tok_logits = logits

    # Standard autoregressive next-token shift: tok_logits[:, t] predicts labels[:, t + 1]
    shift_logits = tok_logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(device=shift_logits.device, dtype=torch.long)

    valid_mask = shift_labels != -100
    safe_targets = shift_labels.clamp(min=0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

    valid_f = valid_mask.to(dtype=token_logps.dtype)
    summed = (token_logps * valid_f).sum(dim=-1)
    lengths = valid_f.sum(dim=-1).clamp(min=1.0)
    per_token = summed / lengths
    return (per_token if length_normalize else summed), per_token


class PocketLLMTrainer:
    """Two-phase trainer for ``PocketConditionedLLM`` (Projector Warmup -> Joint LoRA).

    - **Phase A (Projector Warmup):** Freezes LoRA & base LLM weights; trains ONLY the
      soft-prefix bridge (``perceiver``, ``cvae_proj``, ``global_proj``, ``prefix_norm``,
      ``null_prefix``) at ``llm.projector_lr`` with cosine warmup and gradient clipping.
    - **Phase B (Joint Perceiver + LoRA Fine-Tuning):** Unfreezes LoRA at ``llm.lora_lr``
      and continues updating the bridge at ``llm.phase_b_projector_lr`` with CFG pocket
      dropout (``cfg_drop=True``) and optional light CVAE KL regularization.
    - Evaluates token-weighted ``val_lm_loss`` & ``val_ppl`` every epoch and saves compact
      adapter-only checkpoints (``best_adapter.pt``), streaming ``history.jsonl``, and
      ``train_summary.json``.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm_cfg: LLMConfig = cfg.llm if cfg.llm is not None else LLMConfig()
        self.device = (
            select_device(cfg.train.device)
            if device is None
            else (select_device(device) if isinstance(device, str) else device)
        )
        self.model = model.to(self.device)
        self._meta = _get_backbone_meta(self.model)

        self._bridge_named: list[tuple[str, nn.Parameter]] = []
        self._lora_named: list[tuple[str, nn.Parameter]] = []
        self._frozen_named: list[tuple[str, nn.Parameter]] = []
        self._partition_parameters()

        self.best_val_lm_loss: float = float("inf")
        self.best_val_ppl: float = float("inf")
        self.best_epoch: int = 0
        self.global_step: int = 0
        self.history: list[dict[str, Any]] = []

    def _partition_parameters(self) -> None:
        """Partition model parameters into bridge, LoRA, and permanently frozen sets."""
        self._bridge_named = []
        self._lora_named = []
        self._frozen_named = []

        for name, param in self.model.named_parameters():
            if _is_lora_param(name):
                self._lora_named.append((name, param))
            elif _is_bridge_param(name):
                self._bridge_named.append((name, param))
            elif _is_llm_backbone_param(name):
                # Base pretrained LLM weights stay strictly frozen
                self._frozen_named.append((name, param))
            elif name.startswith(_FROZEN_3D_PREFIXES):
                if self.llm_cfg.freeze_pocket_encoder:
                    self._frozen_named.append((name, param))
                else:
                    self._bridge_named.append((name, param))
            elif param.requires_grad:
                self._bridge_named.append((name, param))
            else:
                self._frozen_named.append((name, param))

    def configure_phase(self, phase: str) -> None:
        """Set exact ``requires_grad`` flags for ``phase='A'`` or ``phase='B'``."""
        phase_upper = phase.strip().upper()
        if phase_upper not in {"A", "B"}:
            raise ValueError(f"Unsupported training phase: {phase!r} (expected 'A' or 'B')")

        for _, param in self._frozen_named:
            param.requires_grad = False
            param.grad = None

        for _, param in self._bridge_named:
            param.requires_grad = True

        unfreeze_lora = phase_upper == "B"
        for _, param in self._lora_named:
            param.requires_grad = unfreeze_lora
            if not unfreeze_lora:
                param.grad = None

    def _build_phase_optimizer(self, phase: str) -> AdamW:
        """Construct phase-specific AdamW optimizer with differential learning rates."""
        wd = self.cfg.train.weight_decay
        groups: list[dict[str, Any]] = []
        if phase == "A":
            groups.extend(
                _split_decay_groups(
                    self._bridge_named,
                    lr=self.llm_cfg.projector_lr,
                    weight_decay=wd,
                    group_prefix="projector",
                )
            )
        else:
            groups.extend(
                _split_decay_groups(
                    self._bridge_named,
                    lr=self.llm_cfg.phase_b_projector_lr,
                    weight_decay=wd,
                    group_prefix="projector",
                )
            )
            groups.extend(
                _split_decay_groups(
                    self._lora_named,
                    lr=self.llm_cfg.lora_lr,
                    weight_decay=wd,
                    group_prefix="lora",
                )
            )

        if not groups:
            raise RuntimeError(f"No trainable parameters configured for Phase {phase}.")
        return AdamW(groups)

    def _make_dataloader(
        self,
        records: Sequence[CachedPocketRecord],
        *,
        shuffle: bool,
        randomize: bool,
        seed: int,
    ) -> DataLoader[dict[str, Any]]:
        """Build DataLoader over cached pocket embeddings with dynamic SMILES tokenization."""
        dataset = CachedPocketLLMDataset(
            records,
            tokenizer=self._meta["tokenizer"],
            is_smiles_native=self._meta["is_smiles_native"],
            prompt_prefix=self._meta["prompt_prefix"],
            bos_token_id=self._meta["bos_token_id"],
            eos_token_id=self._meta["eos_token_id"],
            max_seq_len=self.cfg.data.max_seq_len,
            randomize=randomize,
            seed=seed,
        )
        pad_id = int(self._meta["pad_token_id"])
        gen = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda batch: collate_cached_pocket_batch(batch, pad_token_id=pad_id),
            generator=gen,
            drop_last=False,
        )

    def _move_cached_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move collated cached pocket batch tensors to ``self.device``."""
        p_out: PocketEncoderOutput = batch["pocket_out"]
        pocket_dev = PocketEncoderOutput(
            node_emb=p_out.node_emb.to(device=self.device, dtype=torch.float32),
            pooled_emb=p_out.pooled_emb.to(device=self.device, dtype=torch.float32),
            vec_emb=(
                p_out.vec_emb.to(device=self.device, dtype=torch.float32)
                if p_out.vec_emb is not None
                else None
            ),
        )
        ligand_fp = batch.get("ligand_fp")
        return {
            "pocket_out": pocket_dev,
            "atom_mask": batch["atom_mask"].to(device=self.device, dtype=torch.bool),
            "ligand_fp": (
                ligand_fp.to(device=self.device, dtype=torch.float32)
                if isinstance(ligand_fp, Tensor)
                else None
            ),
            "input_ids": batch["input_ids"].to(device=self.device, dtype=torch.long),
            "attention_mask": batch["attention_mask"].to(device=self.device, dtype=torch.long),
            "labels": batch["labels"].to(device=self.device, dtype=torch.long),
        }

    def _set_train_modes(self, phase: str) -> None:
        """Keep frozen 3D encoder/heads deterministic while training active submodules."""
        self.model.train()
        if self.llm_cfg.freeze_pocket_encoder:
            for attr in ("encoder", "cvae", "heads"):
                sub = getattr(self.model, attr, None)
                if isinstance(sub, nn.Module):
                    sub.eval()
        if phase == "A":
            for attr in ("llm", "backbone"):
                sub = getattr(self.model, attr, None)
                if isinstance(sub, nn.Module):
                    sub.eval()

    @torch.no_grad()
    def evaluate(
        self,
        val_records_or_loader: Sequence[CachedPocketRecord] | Iterable[Mapping[str, Any]],
    ) -> dict[str, float]:
        """Evaluate token-weighted ``val_lm_loss`` and ``val_ppl`` using prior pocket prefix."""
        self.model.eval()
        if isinstance(val_records_or_loader, Sequence) and (
            not val_records_or_loader or isinstance(val_records_or_loader[0], CachedPocketRecord)
        ):
            loader: Iterable[Mapping[str, Any]] = self._make_dataloader(
                cast(Sequence[CachedPocketRecord], val_records_or_loader),
                shuffle=False,
                randomize=False,
                seed=self.cfg.train.seed + 10_000,
            )
        else:
            loader = cast(Iterable[Mapping[str, Any]], val_records_or_loader)

        total_nll = 0.0
        total_tokens = 0

        for raw_batch in loader:
            batch = self._move_cached_batch(raw_batch)
            valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            if valid_tokens <= 0:
                continue

            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pocket_out=batch["pocket_out"],
                atom_mask=batch["atom_mask"],
                ligand_fp=None,
                cfg_drop=False,
                use_posterior=False,
                return_attn=False,
            )
            loss_t = _extract_attr_or_key(out, "loss")
            if loss_t is None:
                _, per_tok = _compute_llm_sequence_logps(
                    self.model,
                    pocket_out=batch["pocket_out"],
                    atom_mask=batch["atom_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    length_normalize=True,
                )
                loss_val = float((-per_tok.mean()).item())
            else:
                loss_val = float(loss_t.detach().float().item())

            if math.isfinite(loss_val):
                total_nll += loss_val * valid_tokens
                total_tokens += valid_tokens

        mean_loss = total_nll / float(max(1, total_tokens))
        return {
            "val_lm_loss": mean_loss,
            "val_ppl": math.exp(min(mean_loss, 20.0)),
            "val_tokens": float(total_tokens),
        }

    def _run_phase(
        self,
        phase: str,
        *,
        num_epochs: int,
        start_epoch: int,
        train_cache: Sequence[CachedPocketRecord],
        val_cache: Sequence[CachedPocketRecord],
        out_dir: Path,
        best_ckpt_path: Path,
        hist_handle: Any,
        log_every: int = 50,
    ) -> int:
        """Run all epochs of Phase A or Phase B and update best adapter checkpoint."""
        if num_epochs <= 0:
            return start_epoch

        self.configure_phase(phase)
        optimizer = self._build_phase_optimizer(phase)
        grad_accum = max(1, int(self.llm_cfg.grad_accum_steps))
        num_batches = max(1, math.ceil(len(train_cache) / max(1, self.cfg.train.batch_size)))
        opt_steps_per_epoch = max(1, math.ceil(num_batches / grad_accum))
        total_opt_steps = max(1, opt_steps_per_epoch * num_epochs)
        scheduler = build_cosine_warmup_scheduler(
            optimizer,
            total_steps=total_opt_steps,
            warmup_ratio=self.cfg.train.warmup_ratio,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        use_cfg_drop = self.llm_cfg.cfg_drop_prob > 0.0
        kl_weight = float(self.llm_cfg.cvae_kl_weight) if phase == "B" else 0.0

        epoch_idx = start_epoch
        for local_ep in range(1, num_epochs + 1):
            epoch_idx += 1
            t0 = time.perf_counter()
            self._set_train_modes(phase)

            train_loader = self._make_dataloader(
                train_cache,
                shuffle=True,
                randomize=self.cfg.data.randomize_smiles,
                seed=self.cfg.train.seed + 100 * epoch_idx,
            )

            lm_loss_sum = 0.0
            kl_loss_sum = 0.0
            total_tokens = 0
            grad_norm_sum = 0.0
            n_opt_steps = 0

            optimizer.zero_grad(set_to_none=True)
            total_batches = len(train_loader)
            for b_idx, raw_batch in enumerate(train_loader):
                batch = self._move_cached_batch(raw_batch)
                valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                if valid_tokens <= 0:
                    continue

                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    pocket_out=batch["pocket_out"],
                    atom_mask=batch["atom_mask"],
                    ligand_fp=batch["ligand_fp"],
                    cfg_drop=use_cfg_drop,
                    use_posterior=True,
                    return_attn=False,
                )
                lm_loss = _extract_attr_or_key(out, "loss")
                if not isinstance(lm_loss, Tensor):
                    raise RuntimeError("PocketConditionedLLM.forward returned no loss tensor.")

                total_loss = lm_loss
                kl_val = 0.0
                cvae_kl = _extract_attr_or_key(out, "cvae_kl_loss")
                if cvae_kl is None and self.cfg.cvae.enabled:
                    mu_q = _extract_attr_or_key(out, "mu_q")
                    logvar_q = _extract_attr_or_key(out, "logvar_q")
                    mu_p = _extract_attr_or_key(out, "mu_p")
                    logvar_p = _extract_attr_or_key(out, "logvar_p")
                    if (
                        isinstance(mu_q, Tensor)
                        and isinstance(logvar_q, Tensor)
                        and isinstance(mu_p, Tensor)
                        and isinstance(logvar_p, Tensor)
                    ):
                        cvae_kl = compute_cvae_kl_loss(
                            mu_q,
                            logvar_q,
                            mu_p,
                            logvar_p,
                            free_bits=self.cfg.cvae.free_bits,
                        )
                if isinstance(cvae_kl, Tensor):
                    kl_val = float(cvae_kl.detach().float().item())
                    if kl_weight > 0.0 and cvae_kl.requires_grad:
                        total_loss = total_loss + kl_weight * cvae_kl

                window_start = (b_idx // grad_accum) * grad_accum
                window_size = min(grad_accum, total_batches - window_start)
                (total_loss / float(max(1, window_size))).backward()

                if ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == total_batches):
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            max_norm=self.cfg.train.grad_clip,
                        ).item()
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    grad_norm_sum += grad_norm
                    n_opt_steps += 1

                lm_val = float(lm_loss.detach().float().item())
                lm_loss_sum += lm_val * valid_tokens
                kl_loss_sum += kl_val * valid_tokens
                total_tokens += valid_tokens

                if log_every > 0 and ((b_idx + 1) % log_every == 0):
                    running_lm = lm_loss_sum / float(max(1, total_tokens))
                    print(
                        f"  [Phase {phase} ep {local_ep}/{num_epochs}] "
                        f"step {b_idx + 1}/{total_batches} | train_lm={running_lm:.4f}",
                        flush=True,
                    )

            train_lm = lm_loss_sum / float(max(1, total_tokens))
            train_ppl = math.exp(min(train_lm, 20.0))
            train_kl = kl_loss_sum / float(max(1, total_tokens))
            mean_gn = grad_norm_sum / float(max(1, n_opt_steps))

            val_metrics = self.evaluate(val_cache)
            val_lm = val_metrics["val_lm_loss"]
            val_ppl = val_metrics["val_ppl"]
            elapsed = time.perf_counter() - t0

            is_best = val_lm < self.best_val_lm_loss
            if is_best:
                self.best_val_lm_loss = val_lm
                self.best_val_ppl = val_ppl
                self.best_epoch = epoch_idx
                extra_meta = {
                    "epoch": epoch_idx,
                    "phase": phase,
                    "global_step": self.global_step,
                    "val_lm_loss": val_lm,
                    "val_ppl": val_ppl,
                    "config": self.cfg.model_dump(mode="python"),
                }
                self.model.save_adapter_checkpoint(  # type: ignore[operator]
                    best_ckpt_path,
                    extra_meta=extra_meta,
                )

            lr_map = {
                str(g.get("name", i)): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)
            }
            row: dict[str, Any] = {
                "epoch": epoch_idx,
                "phase": phase,
                "phase_epoch": local_ep,
                "global_step": self.global_step,
                "train_lm_loss": train_lm,
                "train_ppl": train_ppl,
                "train_kl_loss": train_kl,
                "val_lm_loss": val_lm,
                "val_ppl": val_ppl,
                "grad_norm": mean_gn,
                "lrs": lr_map,
                "elapsed_sec": round(elapsed, 3),
                "is_best": is_best,
            }
            self.history.append(row)
            hist_handle.write(json.dumps(row) + "\n")
            hist_handle.flush()

            print(
                f"[Phase {phase} | ep {epoch_idx:02d}] "
                f"train_lm={train_lm:.4f} (ppl={train_ppl:.2f}) | "
                f"val_lm={val_lm:.4f} (ppl={val_ppl:.2f}) | "
                f"gn={mean_gn:.2f} | {elapsed:.1f}s{' *BEST*' if is_best else ''}",
                flush=True,
            )

        return epoch_idx

    def fit(
        self,
        train_cache: Sequence[CachedPocketRecord],
        val_cache: Sequence[CachedPocketRecord],
        output_dir: str | Path,
        *,
        log_every: int = 50,
    ) -> dict[str, Any]:
        """Run full Phase A + Phase B training over precomputed pocket caches."""
        seed_everything(self.cfg.train.seed)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.cfg.to_yaml(out_dir / "config.yaml")

        best_ckpt_path = out_dir / "best_adapter.pt"
        history_path = out_dir / "history.jsonl"
        summary_path = out_dir / "train_summary.json"

        t0 = time.perf_counter()
        self.best_val_lm_loss = float("inf")
        self.best_val_ppl = float("inf")
        self.best_epoch = 0
        self.global_step = 0
        self.history = []

        with history_path.open("w", encoding="utf-8") as hist_file:
            ep = self._run_phase(
                "A",
                num_epochs=int(self.llm_cfg.phase_a_epochs),
                start_epoch=0,
                train_cache=train_cache,
                val_cache=val_cache,
                out_dir=out_dir,
                best_ckpt_path=best_ckpt_path,
                hist_handle=hist_file,
                log_every=log_every,
            )
            ep = self._run_phase(
                "B",
                num_epochs=int(self.llm_cfg.phase_b_epochs),
                start_epoch=ep,
                train_cache=train_cache,
                val_cache=val_cache,
                out_dir=out_dir,
                best_ckpt_path=best_ckpt_path,
                hist_handle=hist_file,
                log_every=log_every,
            )

        # Ensure a checkpoint always exists even if both phases had 0 epochs or non-finite val
        if not best_ckpt_path.is_file():
            self.model.save_adapter_checkpoint(  # type: ignore[operator]
                best_ckpt_path,
                extra_meta={
                    "epoch": ep,
                    "phase": "B" if self.llm_cfg.phase_b_epochs > 0 else "A",
                    "global_step": self.global_step,
                    "val_lm_loss": self.best_val_lm_loss,
                    "val_ppl": self.best_val_ppl,
                    "config": self.cfg.model_dump(mode="python"),
                },
            )

        wall_time = time.perf_counter() - t0
        total_params = sum(p.numel() for p in self.model.parameters())
        bridge_params = sum(p.numel() for _, p in self._bridge_named)
        lora_params = sum(p.numel() for _, p in self._lora_named)
        ckpt_size_mb = best_ckpt_path.stat().st_size / (1024.0 * 1024.0)

        summary: dict[str, Any] = {
            "name": self.cfg.name,
            "backbone_id": self.llm_cfg.backbone_id,
            "dtype": self.llm_cfg.dtype,
            "split_type": self.cfg.data.split_type,
            "device": self.device.type,
            "seed": self.cfg.train.seed,
            "num_total_params": int(total_params),
            "num_bridge_params": int(bridge_params),
            "num_lora_params": int(lora_params),
            "num_trainable_phase_a": int(bridge_params),
            "num_trainable_phase_b": int(bridge_params + lora_params),
            "train_pockets": len(train_cache),
            "val_pockets": len(val_cache),
            "phase_a_epochs": int(self.llm_cfg.phase_a_epochs),
            "phase_b_epochs": int(self.llm_cfg.phase_b_epochs),
            "epochs_completed": len(self.history),
            "global_steps": int(self.global_step),
            "best_epoch": int(self.best_epoch),
            "best_val_lm_loss": (
                float(self.best_val_lm_loss) if math.isfinite(self.best_val_lm_loss) else None
            ),
            "best_val_ppl": (
                float(self.best_val_ppl) if math.isfinite(self.best_val_ppl) else None
            ),
            "wall_time_sec": round(wall_time, 3),
            "adapter_ckpt_mb": round(ckpt_size_mb, 3),
            "best_adapter_path": str(best_ckpt_path),
            "config_path": str(config_path),
            "history_path": str(history_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return {
            **summary,
            "history": list(self.history),
        }


def _compute_ecfp4_tensor(smiles_list: Sequence[str], fp_dim: int = LIGAND_FP_DIM) -> Tensor:
    """Compute float32 Morgan bit fingerprints ``[M, fp_dim]`` for candidate scoring."""
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    rows: list[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append(np.zeros(fp_dim, dtype=np.float32))
        else:
            rows.append(fp_gen.GetFingerprintAsNumPy(mol).astype(np.float32))
    if not rows:
        return torch.zeros((0, fp_dim), dtype=torch.float32)
    return torch.from_numpy(np.stack(rows, axis=0))


def _record_to_pocket_out_and_mask(
    record: CachedPocketRecord | Mapping[str, Any],
    model: nn.Module,
    device: torch.device,
) -> tuple[PocketEncoderOutput, Tensor]:
    """Convert either a ``CachedPocketRecord`` or raw pocket dict into device tensors."""
    if isinstance(record, CachedPocketRecord):
        node_emb = record.node_emb.unsqueeze(0).to(device=device, dtype=torch.float32)
        pooled = record.pooled_emb.unsqueeze(0).to(device=device, dtype=torch.float32)
        mask = record.atom_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled, vec_emb=None), mask

    if "node_emb" in record and "pooled_emb" in record:
        node_emb = torch.as_tensor(record["node_emb"], dtype=torch.float32)
        if node_emb.ndim == 2:
            node_emb = node_emb.unsqueeze(0)
        pooled = torch.as_tensor(record["pooled_emb"], dtype=torch.float32)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        if "atom_mask" in record and record["atom_mask"] is not None:
            mask = torch.as_tensor(record["atom_mask"], dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.ones((node_emb.shape[0], node_emb.shape[1]), dtype=torch.bool)
        return (
            PocketEncoderOutput(
                node_emb=node_emb.to(device=device),
                pooled_emb=pooled.to(device=device),
                vec_emb=None,
            ),
            mask.to(device=device),
        )

    coords = torch.as_tensor(record["coords"], dtype=torch.float32)
    node_feats = torch.as_tensor(record["node_feats"], dtype=torch.float32)
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
        node_feats = node_feats.unsqueeze(0)
    if "atom_mask" in record and record["atom_mask"] is not None:
        mask = torch.as_tensor(record["atom_mask"], dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
    else:
        mask = torch.ones((coords.shape[0], coords.shape[1]), dtype=torch.bool)

    with torch.no_grad():
        pocket_out: PocketEncoderOutput = model.encode_pocket(  # type: ignore[operator]
            coords.to(device=device),
            node_feats.to(device=device),
            mask.to(device=device),
        )
    return pocket_out, mask.to(device=device)


def run_llm_rpo_dpo(
    model: nn.Module,
    train_records: Sequence[CachedPocketRecord | Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]],
    cfg: ExperimentConfig,
    device: torch.device | str | None = None,
    *,
    dpo_pockets: int = 250,
    samples_per_pocket: int = 16,
    epochs: int = 1,
    beta: float = 0.5,
    lr: float = 1e-5,
    sft_nll_weight: float = 0.25,
    grad_accum_steps: int = 8,
    min_margin: float = 0.05,
    max_pairs_per_pocket: int = 8,
    max_new_tokens: int = 96,
    cfg_scale: float | None = None,
    temperature: float | None = None,
    min_p: float | None = None,
    seed: int | None = None,
    preference_pairs: Sequence[PreferencePair] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run stabilized LoRA-only RPO-DPO preference alignment on ``PocketConditionedLLM``.

    Pipeline:
      1. Samples ``M = samples_per_pocket`` candidate molecules per pocket via fast
         KV-cached ``sample_llm_molecules``.
      2. Scores valid unique candidates with ``model.score_candidates`` (``pred_pkd`` +
         ``clip_sim``) and ``compute_mpo_scores`` (QED, SA, Lipinski, PAINS).
      3. Constructs MPO preference pairs via ``build_preference_pairs_from_mpo``.
      4. Precomputes frozen reference log-probabilities ``ref_logps`` before any weight
         updates (avoiding a second copy of the LLM in memory).
      5. Freezes EVERY parameter except LoRA adapter weights (preserving ``PaiNN-SE3``,
         ``CVAE``, ``heads``, ``Perceiver``, ``prefix_norm``, and base LLM bit-for-bit!).
      6. Optimizes length-normalized RPO-DPO (``dpo_loss + sft_nll_weight * sft_nll``)
         with gradient accumulation and clipping.
    """
    from geneuclidean.sampling.llm_sampler import sample_llm_molecules

    eff_device = (
        select_device(cfg.train.device)
        if device is None
        else (select_device(device) if isinstance(device, str) else device)
    )
    eff_seed = cfg.train.seed if seed is None else int(seed)
    seed_everything(eff_seed)
    model = model.to(eff_device)
    meta = _get_backbone_meta(model)

    eff_cfg_scale = cfg.sampling.cfg_scale if cfg_scale is None else float(cfg_scale)
    eff_temp = cfg.sampling.temperature if temperature is None else float(temperature)
    eff_min_p = cfg.sampling.min_p if min_p is None else float(min_p)
    eff_accum = max(1, int(grad_accum_steps))
    eff_sft_weight = max(0.0, float(sft_nll_weight))

    t0 = time.perf_counter()
    pairs: list[PreferencePair] = []
    total_generated = 0
    unique_valid_total = 0
    n_pockets_used = 0

    if preference_pairs is not None:
        pairs = list(preference_pairs)
    else:
        record_list: list[CachedPocketRecord | Mapping[str, Any]]
        if isinstance(train_records, Mapping):
            record_list = [train_records[k] for k in sorted(train_records.keys())]
        else:
            record_list = list(train_records)

        n_avail = len(record_list)
        n_pockets_used = min(int(dpo_pockets), n_avail)
        if n_pockets_used > 0:
            rng = np.random.default_rng(eff_seed)
            indices = rng.choice(n_avail, size=n_pockets_used, replace=False).tolist()
        else:
            indices = []

        model.eval()
        for step_idx, r_idx in enumerate(indices):
            rec = record_list[int(r_idx)]
            pocket_out, atom_mask = _record_to_pocket_out_and_mask(rec, model, eff_device)
            sampled = sample_llm_molecules(
                cast(Any, model),
                pocket_out,
                atom_mask,
                num_samples=int(samples_per_pocket),
                max_new_tokens=int(max_new_tokens),
                strategy=cfg.sampling.strategy,
                temperature=eff_temp,
                min_p=eff_min_p,
                top_p=cfg.sampling.top_p,
                top_k=cfg.sampling.top_k,
                cfg_scale=eff_cfg_scale,
                latent_temp=cfg.sampling.latent_temp,
                repetition_penalty=cfg.sampling.repetition_penalty,
                batch_size=int(samples_per_pocket),
                seed=eff_seed + step_idx,
            )
            total_generated += len(sampled)

            unique_valid: list[str] = []
            seen_can: set[str] = set()
            for smi in sampled:
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None:
                    continue
                can = Chem.MolToSmiles(mol, canonical=True)
                if can not in seen_can:
                    seen_can.add(can)
                    unique_valid.append(can)
            unique_valid_total += len(unique_valid)
            if len(unique_valid) < 2:
                continue

            with torch.no_grad():
                fps_t = _compute_ecfp4_tensor(
                    unique_valid,
                    fp_dim=cfg.multitask.ligand_fp_dim,
                ).to(device=eff_device)
                cand_scores = model.score_candidates(pocket_out, fps_t)  # type: ignore[operator]
                pred_pkd = (
                    cand_scores["pkd"].detach().cpu().numpy().tolist()
                    if "pkd" in cand_scores
                    else None
                )
                clip_sim = (
                    cand_scores["clip_sim"].detach().cpu().numpy().tolist()
                    if "clip_sim" in cand_scores
                    else None
                )

            mpo_entries = compute_mpo_scores(unique_valid, pred_pkd=pred_pkd, clip_sim=clip_sim)
            mpo_vals = [entry.mpo_score for entry in mpo_entries]

            compact_pocket = {
                "node_emb": pocket_out.node_emb.detach().cpu(),
                "pooled_emb": pocket_out.pooled_emb.detach().cpu(),
                "atom_mask": atom_mask.detach().cpu(),
            }
            pocket_pairs = build_preference_pairs_from_mpo(
                compact_pocket,
                unique_valid,
                mpo_vals,
                min_margin=float(min_margin),
                max_pairs=int(max_pairs_per_pocket),
            )
            pairs.extend(pocket_pairs)

    pair_gen_sec = time.perf_counter() - t0
    if not pairs:
        return {
            "epochs": 0,
            "num_pairs": 0,
            "dpo_pockets_sampled": n_pockets_used,
            "total_generated": total_generated,
            "unique_valid_generated": unique_valid_total,
            "pair_gen_sec": round(pair_gen_sec, 3),
            "history": [],
            "adapter_ckpt_path": None,
        }

    # Precompute frozen reference log-probabilities before any LoRA update
    cached_dpo_items: list[
        tuple[PocketEncoderOutput, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    ] = []
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            p_out_dev, a_mask_dev = _record_to_pocket_out_and_mask(
                pair.pocket_batch, model, eff_device
            )
            pos_ids_list, pos_lbl_list = tokenize_smiles_for_llm(
                pair.chosen_smiles,
                tokenizer=meta["tokenizer"],
                is_smiles_native=meta["is_smiles_native"],
                prompt_prefix=meta["prompt_prefix"],
                bos_token_id=meta["bos_token_id"],
                eos_token_id=meta["eos_token_id"],
                max_seq_len=cfg.data.max_seq_len,
            )
            neg_ids_list, neg_lbl_list = tokenize_smiles_for_llm(
                pair.rejected_smiles,
                tokenizer=meta["tokenizer"],
                is_smiles_native=meta["is_smiles_native"],
                prompt_prefix=meta["prompt_prefix"],
                bos_token_id=meta["bos_token_id"],
                eos_token_id=meta["eos_token_id"],
                max_seq_len=cfg.data.max_seq_len,
            )
            pos_ids = torch.tensor(pos_ids_list, dtype=torch.long, device=eff_device).unsqueeze(0)
            pos_lbl = torch.tensor(pos_lbl_list, dtype=torch.long, device=eff_device).unsqueeze(0)
            pos_attn = torch.ones_like(pos_ids)

            neg_ids = torch.tensor(neg_ids_list, dtype=torch.long, device=eff_device).unsqueeze(0)
            neg_lbl = torch.tensor(neg_lbl_list, dtype=torch.long, device=eff_device).unsqueeze(0)
            neg_attn = torch.ones_like(neg_ids)

            ref_pos, _ = _compute_llm_sequence_logps(
                model,
                pocket_out=p_out_dev,
                atom_mask=a_mask_dev,
                input_ids=pos_ids,
                attention_mask=pos_attn,
                labels=pos_lbl,
                length_normalize=True,
            )
            ref_neg, _ = _compute_llm_sequence_logps(
                model,
                pocket_out=p_out_dev,
                atom_mask=a_mask_dev,
                input_ids=neg_ids,
                attention_mask=neg_attn,
                labels=neg_lbl,
                length_normalize=True,
            )
            p_out_cpu = PocketEncoderOutput(
                node_emb=p_out_dev.node_emb.detach().cpu(),
                pooled_emb=p_out_dev.pooled_emb.detach().cpu(),
                vec_emb=None,
            )
            cached_dpo_items.append(
                (
                    p_out_cpu,
                    a_mask_dev.detach().cpu(),
                    pos_ids.detach().cpu(),
                    pos_lbl.detach().cpu(),
                    neg_ids.detach().cpu(),
                    neg_lbl.detach().cpu(),
                    ref_pos.detach().cpu(),
                    ref_neg.detach().cpu(),
                )
            )

    orig_requires_grad: dict[str, bool] = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    history: list[dict[str, float | int]] = []

    try:
        # Freeze EVERYTHING except LoRA adapter weights bit-for-bit!
        lora_named: list[tuple[str, nn.Parameter]] = []
        for name, param in model.named_parameters():
            if _is_lora_param(name):
                param.requires_grad = True
                lora_named.append((name, param))
            else:
                param.requires_grad = False
                param.grad = None

        if not lora_named:
            raise RuntimeError("No LoRA parameters found in PocketConditionedLLM for RPO-DPO.")

        lora_params = [p for _, p in lora_named]
        optimizer = AdamW(
            _split_decay_groups(
                lora_named,
                lr=lr,
                weight_decay=cfg.train.weight_decay,
                group_prefix="dpo_lora",
            )
        )
        n_total = len(cached_dpo_items)
        llm_sub = getattr(model, "llm", None)

        for epoch in range(1, max(1, int(epochs)) + 1):
            ep_t0 = time.perf_counter()
            model.eval()
            if isinstance(llm_sub, nn.Module):
                llm_sub.train()

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
                p_out_cpu,
                a_mask_cpu,
                pos_ids_c,
                pos_lbl_c,
                neg_ids_c,
                neg_lbl_c,
                ref_pos_c,
                ref_neg_c,
            ) in enumerate(cached_dpo_items):
                p_out_dev = PocketEncoderOutput(
                    node_emb=p_out_cpu.node_emb.to(device=eff_device),
                    pooled_emb=p_out_cpu.pooled_emb.to(device=eff_device),
                    vec_emb=None,
                )
                a_mask_dev = a_mask_cpu.to(device=eff_device)
                pos_ids = pos_ids_c.to(device=eff_device)
                pos_lbl = pos_lbl_c.to(device=eff_device)
                neg_ids = neg_ids_c.to(device=eff_device)
                neg_lbl = neg_lbl_c.to(device=eff_device)
                ref_pos = ref_pos_c.to(device=eff_device)
                ref_neg = ref_neg_c.to(device=eff_device)

                pi_pos, pi_pos_norm = _compute_llm_sequence_logps(
                    model,
                    pocket_out=p_out_dev,
                    atom_mask=a_mask_dev,
                    input_ids=pos_ids,
                    attention_mask=torch.ones_like(pos_ids),
                    labels=pos_lbl,
                    length_normalize=True,
                )
                pi_neg, _ = _compute_llm_sequence_logps(
                    model,
                    pocket_out=p_out_dev,
                    atom_mask=a_mask_dev,
                    input_ids=neg_ids,
                    attention_mask=torch.ones_like(neg_ids),
                    labels=neg_lbl,
                    length_normalize=True,
                )

                dpo_loss = compute_dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=beta)
                sft_nll = -pi_pos_norm.mean()
                loss = dpo_loss + eff_sft_weight * sft_nll

                window_start = (idx // eff_accum) * eff_accum
                window_size = min(eff_accum, n_total - window_start)
                (loss / float(max(1, window_size))).backward()

                if ((idx + 1) % eff_accum == 0) or ((idx + 1) == n_total):
                    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=cfg.train.grad_clip)
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

            n_pairs_f = float(n_total)
            history.append(
                {
                    "epoch": epoch,
                    "dpo_loss": dpo_loss_sum / n_pairs_f,
                    "sft_nll": sft_nll_sum / n_pairs_f,
                    "total_loss": total_loss_sum / n_pairs_f,
                    "chosen_reward": pos_reward_sum / n_pairs_f,
                    "rejected_reward": neg_reward_sum / n_pairs_f,
                    "reward_margin": margin_sum / n_pairs_f,
                    "preference_accuracy": float(correct) / n_pairs_f,
                    "optimizer_steps": opt_steps,
                    "elapsed_sec": round(time.perf_counter() - ep_t0, 3),
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
        ckpt_path = out_dir / "best_adapter.pt"
        model.save_adapter_checkpoint(  # type: ignore[operator]
            ckpt_path,
            extra_meta={
                "stage": "rpo_dpo",
                "config": cfg.model_dump(mode="python"),
                "dpo_history": history,
                "num_pairs": len(pairs),
            },
        )
        ckpt_path_str = str(ckpt_path)
        hist_path = out_dir / "history.jsonl"
        with hist_path.open("w", encoding="utf-8") as handle:
            for row in history:
                handle.write(json.dumps(row) + "\n")

    return {
        "epochs": len(history),
        "num_pairs": len(pairs),
        "dpo_pockets_sampled": n_pockets_used,
        "total_generated": total_generated,
        "unique_valid_generated": unique_valid_total,
        "pair_gen_sec": round(pair_gen_sec, 3),
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "history": history,
        "adapter_ckpt_path": ckpt_path_str,
    }


__all__ = [
    "CachedPocketLLMDataset",
    "CachedPocketRecord",
    "PocketLLMTrainer",
    "build_pocket_batches_from_records",
    "collate_cached_pocket_batch",
    "precompute_pocket_cache",
    "run_llm_rpo_dpo",
    "tokenize_smiles_for_llm",
]
