"""Resumable Stage-2 & Stage-3 experimental campaign orchestrator for GenEuclidean v2.

Runs any individual model, subset, or the full 9-model benchmark suite (from modernized
v1.5 E3NN-LSTM through flagship GenEuclidean v2 PaiNN-SE3 + Stage-3 MPO-Guided DPO)
against the cached PDBbind/CASF-2016/LP-PDBBind splits.

Usage examples::

    # Inspect available suite runs
    .venv/bin/python scripts/run_stage2_suite.py --list-runs

    # Train flagship GenEuclidean v2 (PaiNN-SE3) on CASF-2016 split
    .venv/bin/python scripts/run_stage2_suite.py \\
        --run geneuclidean_v2_painn --split casf2016 --epochs 30 --device mps

    # Train flagship on leak-proof LP-PDBBind split (-> runs/geneuclidean_v2_painn_lp/)
    .venv/bin/python scripts/run_stage2_suite.py \\
        --run geneuclidean_v2_painn --split lp_pdbbind --epochs 30 --device mps

    # Run Stage-3 MPO-Guided DPO initialized from flagship best.pt
    .venv/bin/python scripts/run_stage2_suite.py \\
        --run geneuclidean_v2_painn_dpo --split casf2016 --device mps

    # Run full campaign with automatic skip of completed runs
    .venv/bin/python scripts/run_stage2_suite.py --all --split casf2016 --resume-skip
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, Dataset, Subset

from geneuclidean.config import ExperimentConfig, SamplingConfig, seed_everything, select_device
from geneuclidean.data.dataset import PocketLigandDataset, collate_pocket_batch
from geneuclidean.data.tokenizer import MolecularTokenizer
from geneuclidean.models import GenEuclideanModel
from geneuclidean.sampling.decoder_strategies import sample_molecules
from geneuclidean.sampling.rerank import compute_mpo_scores
from geneuclidean.training.dpo import (
    PreferencePair,
    build_preference_pairs_from_mpo,
    run_dpo_alignment,
)
from geneuclidean.training.trainer import PocketTrainer

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RunSpec:
    """Declarative specification for one named Stage-2 or Stage-3 experiment."""

    name: str
    config_path: str
    stage: str  # "stage2" or "stage3_dpo"
    default_pretrained: str | None
    from_scratch: bool = False
    description: str = ""


RUN_SPECS: dict[str, RunSpec] = {
    "e3nn_lstm_v15": RunSpec(
        name="e3nn_lstm_v15",
        config_path="configs/e3nn_lstm_v15.yaml",
        stage="stage2",
        default_pretrained=None,
        from_scratch=True,
        description="Model 1: Bug-fixed e3nn (lmax=1) + 2L Attention-LSTM from scratch",
    ),
    "painn_pocketformer_scratch": RunSpec(
        name="painn_pocketformer_scratch",
        config_path="configs/painn_pocketformer.yaml",
        stage="stage2",
        default_pretrained=None,
        from_scratch=True,
        description="Model 2a: PaiNN-SE3 + RoPE/SwiGLU Transformer from scratch (no prior/CVAE/MT)",
    ),
    "painn_pocketformer": RunSpec(
        name="painn_pocketformer",
        config_path="configs/painn_pocketformer.yaml",
        stage="stage2",
        default_pretrained="runs/prior_smiles/best_prior.pt",
        from_scratch=False,
        description="Model 2b: PaiNN-SE3 + Stage-1 SMILES prior (no CVAE/MultiTask)",
    ),
    "geneuclidean_v2_painn": RunSpec(
        name="geneuclidean_v2_painn",
        config_path="configs/geneuclidean_v2_painn.yaml",
        stage="stage2",
        default_pretrained="runs/prior_smiles/best_prior.pt",
        from_scratch=False,
        description="Model 3 Flagship: PaiNN-SE3 + Prior + CVAE + MultiTask (pKd/CLIP/Props) + CFG",
    ),
    "geneuclidean_v2_e3nn": RunSpec(
        name="geneuclidean_v2_e3nn",
        config_path="configs/geneuclidean_v2_e3nn.yaml",
        stage="stage2",
        default_pretrained="runs/prior_smiles/best_prior.pt",
        from_scratch=False,
        description="Model 3b: Modern e3nn (lmax=1) + Prior + CVAE + MultiTask + CFG",
    ),
    "geneuclidean_v2_egnn": RunSpec(
        name="geneuclidean_v2_egnn",
        config_path="configs/geneuclidean_v2_egnn.yaml",
        stage="stage2",
        default_pretrained="runs/prior_smiles/best_prior.pt",
        from_scratch=False,
        description="Model 3c: EGNN encoder ablation + Prior + CVAE + MultiTask + CFG",
    ),
    "geneuclidean_v2_schnet": RunSpec(
        name="geneuclidean_v2_schnet",
        config_path="configs/geneuclidean_v2_schnet.yaml",
        stage="stage2",
        default_pretrained="runs/prior_smiles/best_prior.pt",
        from_scratch=False,
        description="Model 3d: SchNet distance-only ablation + Prior + CVAE + MultiTask + CFG",
    ),
    "geneuclidean_v2_selfies": RunSpec(
        name="geneuclidean_v2_selfies",
        config_path="configs/geneuclidean_v2_selfies.yaml",
        stage="stage2",
        default_pretrained="runs/prior_selfies/best_prior.pt",
        from_scratch=False,
        description="Model 3e: SELFIES ablation with PaiNN-SE3 + SELFIES Prior + CVAE + MultiTask",
    ),
    "geneuclidean_v2_painn_dpo": RunSpec(
        name="geneuclidean_v2_painn_dpo",
        config_path="configs/geneuclidean_v2_painn.yaml",
        stage="stage3_dpo",
        default_pretrained=None,
        from_scratch=False,
        description="Model 4 (Stage-3): MPO-Guided DPO alignment from flagship PaiNN-SE3 best.pt",
    ),
}

DEFAULT_RUN_ORDER: tuple[str, ...] = tuple(RUN_SPECS.keys())


def _split_suffix(split_type: str) -> str:
    """Return canonical directory suffix for non-default benchmark splits."""
    if split_type == "casf2016":
        return ""
    if split_type == "lp_pdbbind":
        return "_lp"
    return f"_{split_type}"


def _resolve_output_dir(
    run_name: str,
    split_type: str,
    runs_root: Path,
    explicit_output_dir: str | None,
) -> Path:
    """Resolve output directory for a run, applying split suffix when not overridden."""
    if explicit_output_dir is not None:
        return Path(explicit_output_dir)
    return runs_root / f"{run_name}{_split_suffix(split_type)}"


def _resolve_vocab_path(processed_dir: Path, representation: str) -> Path:
    """Resolve vocabulary JSON matching representation (``smiles`` or ``selfies``)."""
    filename = "vocab_selfies.json" if representation == "selfies" else "vocab_smiles.json"
    vocab_path = processed_dir / filename
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    return vocab_path


def _load_and_configure(
    spec: RunSpec,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[ExperimentConfig, MolecularTokenizer]:
    """Load base YAML, apply spec adjustments and CLI overrides, and bind tokenizer vocab."""
    cfg = ExperimentConfig.from_yaml(spec.config_path)
    cfg.data.processed_dir = str(args.processed_dir)
    cfg.data.split_type = args.split
    cfg.train.device = args.device
    cfg.train.output_dir = str(out_dir)

    if spec.name == "painn_pocketformer_scratch":
        cfg.name = "painn-pocketformer-scratch"
        cfg.description = spec.description
        cfg.train.pretrained_decoder_ckpt = None
        cfg.train.freeze_decoder_epochs = 0
        cfg.train.decoder_lr = cfg.train.encoder_lr
    elif spec.from_scratch:
        cfg.train.pretrained_decoder_ckpt = None
        cfg.train.freeze_decoder_epochs = 0
    elif spec.stage == "stage2":
        cfg.train.pretrained_decoder_ckpt = spec.default_pretrained

    # Apply explicit CLI overrides
    if args.pretrained_ckpt is not None:
        if args.pretrained_ckpt.strip().lower() in {"none", "null", ""}:
            cfg.train.pretrained_decoder_ckpt = None
            cfg.train.freeze_decoder_epochs = 0
        else:
            cfg.train.pretrained_decoder_ckpt = args.pretrained_ckpt

    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.encoder_lr is not None:
        cfg.train.encoder_lr = float(args.encoder_lr)
        if spec.from_scratch and args.decoder_lr is None:
            cfg.train.decoder_lr = float(args.encoder_lr)
    if args.decoder_lr is not None:
        cfg.train.decoder_lr = float(args.decoder_lr)
    if args.kl_max_beta is not None:
        cfg.cvae.kl_max_beta = float(args.kl_max_beta)
    if args.lambda_pkd is not None:
        cfg.multitask.lambda_pkd = float(args.lambda_pkd)
    if args.lambda_clip is not None:
        cfg.multitask.lambda_clip = float(args.lambda_clip)
    if args.lambda_prop is not None:
        cfg.multitask.lambda_prop = float(args.lambda_prop)
    if args.freeze_decoder_epochs is not None:
        cfg.train.freeze_decoder_epochs = int(args.freeze_decoder_epochs)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)

    vocab_path = _resolve_vocab_path(Path(args.processed_dir), cfg.data.representation)
    tokenizer = MolecularTokenizer.load(vocab_path)
    # CRITICAL: always bind exact tokenizer vocab size before constructing GenEuclideanModel
    cfg.decoder.vocab_size = tokenizer.vocab_size
    return cfg, tokenizer


def _build_datasets(
    cfg: ExperimentConfig,
    tokenizer: MolecularTokenizer,
    processed_dir: Path,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[Dataset[dict[str, Any]], Dataset[dict[str, Any]]]:
    """Instantiate train/val ``PocketLigandDataset`` objects with optional dry-run subsetting."""
    cache_path = processed_dir / "pdbbind_pockets.pt"
    splits_path = processed_dir / "splits.json"
    if not cache_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(
            f"Processed cache missing in {processed_dir} "
            "(expected pdbbind_pockets.pt and splits.json)."
        )

    train_base = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type=cfg.data.split_type,
        split_name="train",
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=cfg.data.randomize_smiles,
        coord_jitter_std=cfg.data.coord_jitter_std,
        atom_drop_prob=cfg.data.atom_drop_prob,
        seed=cfg.train.seed,
    )
    val_base = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type=cfg.data.split_type,
        split_name="val",
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=False,
        coord_jitter_std=0.0,
        atom_drop_prob=0.0,
        seed=cfg.train.seed + 1,
    )

    train_ds: Dataset[dict[str, Any]] = train_base
    val_ds: Dataset[dict[str, Any]] = val_base
    if max_train_samples is not None and max_train_samples < len(train_base):
        train_ds = Subset(train_base, list(range(max_train_samples)))
    if max_val_samples is not None and max_val_samples < len(val_base):
        val_ds = Subset(val_base, list(range(max_val_samples)))
    return train_ds, val_ds


def _strip_optimizer_from_best_ckpt(best_ckpt_path: Path) -> None:
    """Remove bulky optimizer/scheduler states from finalized ``best.pt`` to save disk space."""
    if not best_ckpt_path.is_file():
        return
    payload = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and (
        "optimizer_state_dict" in payload or "scheduler_state_dict" in payload
    ):
        slim = {
            k: v
            for k, v in payload.items()
            if k not in ("optimizer_state_dict", "scheduler_state_dict")
        }
        torch.save(slim, best_ckpt_path)


def _extract_best_history_row(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the epoch history row corresponding to the saved ``best.pt`` checkpoint."""
    if not history:
        return {}
    best_flagged = [row for row in history if row.get("is_best") is True]
    if best_flagged:
        return best_flagged[-1]
    return min(history, key=lambda row: float(row.get("val_objective", float("inf"))))


def _clear_accelerator_cache(device: torch.device) -> None:
    """Release cached allocator memory cleanly between sequential suite runs."""
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_stage2(
    spec: RunSpec,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    """Execute one Stage-2 pocket-conditioned training run and emit artifacts."""
    cfg, tokenizer = _load_and_configure(spec, args, out_dir)
    seed_everything(cfg.train.seed)
    device = select_device(cfg.train.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard against missing Stage-1 prior before loading dataset
    if cfg.train.pretrained_decoder_ckpt is not None:
        prior_path = Path(cfg.train.pretrained_decoder_ckpt)
        if not prior_path.is_file():
            raise FileNotFoundError(
                f"Stage-1 pretrained prior not found for {spec.name!r}: {prior_path}. "
                "Wait for pretraining to finish or pass --pretrained-ckpt <path|none>."
            )

    train_ds, val_ds = _build_datasets(
        cfg,
        tokenizer,
        processed_dir=Path(args.processed_dir),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    train_gen = torch.Generator().manual_seed(cfg.train.seed)
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pocket_batch,
        generator=train_gen,
        drop_last=False,
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pocket_batch,
        drop_last=False,
    )

    model = GenEuclideanModel(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config_path = cfg.to_yaml(out_dir / "config.yaml")

    print(
        f"[{spec.name}] split={cfg.data.split_type} device={device.type} "
        f"arch={cfg.encoder.arch}+{cfg.decoder.arch} cvae={cfg.cvae.enabled} "
        f"vocab={tokenizer.vocab_size} params={num_params:,} "
        f"train={len(train_ds)} val={len(val_ds)} epochs={cfg.train.epochs} "
        f"prior={cfg.train.pretrained_decoder_ckpt}",
        flush=True,
    )

    trainer = PocketTrainer(model=model, cfg=cfg, tokenizer=tokenizer, device=device)
    resume_from: Path | None = None
    if args.resume is not None:
        candidate = out_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if candidate.is_file():
            resume_from = candidate
        elif args.resume != "auto":
            raise FileNotFoundError(f"Requested --resume checkpoint not found: {candidate}")

    t0 = time.perf_counter()
    fit_res = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        resume_from=resume_from,
    )
    wall_time = time.perf_counter() - t0

    best_ckpt = Path(fit_res["best_ckpt_path"])
    last_ckpt = Path(fit_res["last_ckpt_path"])
    history: list[dict[str, Any]] = list(fit_res["history"])

    # Clean up redundant last.pt and bulky optimizer state in best.pt after clean completion
    _strip_optimizer_from_best_ckpt(best_ckpt)
    if not args.keep_last_ckpt and last_ckpt.is_file():
        last_ckpt.unlink()

    best_row = _extract_best_history_row(history)
    summary: dict[str, Any] = {
        "run_name": spec.name,
        "stage": spec.stage,
        "split_type": cfg.data.split_type,
        "representation": cfg.data.representation,
        "encoder_arch": cfg.encoder.arch,
        "decoder_arch": cfg.decoder.arch,
        "cvae_enabled": cfg.cvae.enabled,
        "pretrained_decoder_ckpt": cfg.train.pretrained_decoder_ckpt,
        "device": device.type,
        "seed": cfg.train.seed,
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs_completed": len(history),
        "wall_time_sec": round(wall_time, 3),
        "best_epoch": int(best_row.get("epoch", 0)),
        "best_val_objective": float(best_row.get("val_objective", float("nan"))),
        "best_val_lm_loss": float(best_row.get("val_lm_loss", float("nan"))),
        "best_val_ppl": float(best_row.get("val_ppl", float("nan"))),
        "best_val_kl_nats": float(best_row.get("val_kl_nats", 0.0)),
        "best_val_pkd_pearson": float(best_row.get("val_pkd_pearson", 0.0)),
        "best_val_pkd_rmse": float(best_row.get("val_pkd_rmse", 0.0)),
        "best_val_clip_r1": float(best_row.get("val_clip_r1", 0.0)),
        "best_val_clip_r5": float(best_row.get("val_clip_r5", 0.0)),
        "best_ckpt_path": str(best_ckpt),
        "config_path": str(config_path),
        "history_path": str(fit_res["history_path"]),
    }

    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{spec.name}] DONE in {wall_time:.1f}s | best_ep={summary['best_epoch']} "
        f"val_lm={summary['best_val_lm_loss']:.4f} val_ppl={summary['best_val_ppl']:.2f} "
        f"pkd_r={summary['best_val_pkd_pearson']:.3f} pkd_rmse={summary['best_val_pkd_rmse']:.3f} "
        f"clip_r1={summary['best_val_clip_r1']:.3f} clip_r5={summary['best_val_clip_r5']:.3f}",
        flush=True,
    )

    del trainer, model
    _clear_accelerator_cache(device)
    return summary


def _compute_ecfp4_tensor(smiles_list: Sequence[str], fp_dim: int = 1024) -> torch.Tensor:
    """Compute float32 ECFP4 fingerprint matrix ``[M, fp_dim]`` matching ``featurizer.py``."""
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    rows: list[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(fp_dim, dtype=np.float32))
        else:
            rows.append(fp_gen.GetFingerprintAsNumPy(mol).astype(np.float32))
    if not rows:
        return torch.zeros((0, fp_dim), dtype=torch.float32)
    return torch.from_numpy(np.stack(rows, axis=0))


def _load_parent_config_and_weights(
    init_ckpt_path: Path,
    fallback_yaml: str,
) -> tuple[ExperimentConfig, dict[str, torch.Tensor]]:
    """Load parent Stage-2 ``ExperimentConfig`` and model state dict from ``best.pt``."""
    if not init_ckpt_path.is_file():
        raise FileNotFoundError(
            f"Stage-3 DPO parent checkpoint not found: {init_ckpt_path}. "
            "Train geneuclidean_v2_painn first or pass --init-ckpt <path>."
        )
    payload = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Expected dict with 'model_state_dict' in {init_ckpt_path}")

    sibling_cfg = init_ckpt_path.parent / "config.yaml"
    if sibling_cfg.is_file():
        cfg = ExperimentConfig.from_yaml(sibling_cfg)
    elif isinstance(payload.get("config"), dict):
        cfg = ExperimentConfig.model_validate(payload["config"])
    else:
        cfg = ExperimentConfig.from_yaml(fallback_yaml)

    state_dict: dict[str, torch.Tensor] = payload["model_state_dict"]
    return cfg, state_dict


def _run_stage3_dpo(
    spec: RunSpec,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    """Execute Stage-3 MPO-Guided DPO preference alignment from a trained flagship checkpoint."""
    runs_root = Path(args.runs_root)
    default_parent = runs_root / f"geneuclidean_v2_painn{_split_suffix(args.split)}" / "best.pt"
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt is not None else default_parent

    cfg, parent_state_dict = _load_parent_config_and_weights(init_ckpt, spec.config_path)
    cfg.name = "geneuclidean-v2-painn-dpo"
    cfg.description = spec.description
    cfg.data.processed_dir = str(args.processed_dir)
    cfg.data.split_type = args.split
    cfg.train.device = args.device
    cfg.train.pretrained_decoder_ckpt = None
    cfg.train.output_dir = str(out_dir)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)

    seed_everything(cfg.train.seed)
    device = select_device(cfg.train.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = _resolve_vocab_path(Path(args.processed_dir), cfg.data.representation)
    tokenizer = MolecularTokenizer.load(vocab_path)
    cfg.decoder.vocab_size = tokenizer.vocab_size

    model = GenEuclideanModel(cfg)
    model.load_state_dict(parent_state_dict)
    model = model.to(device)
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds, val_ds = _build_datasets(
        cfg,
        tokenizer,
        processed_dir=Path(args.processed_dir),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_ds,
        batch_size=args.batch_size if args.batch_size is not None else cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pocket_batch,
        drop_last=False,
    )

    # Select deterministic subset of training pockets for on-policy candidate generation
    n_train = len(train_ds)
    n_pockets = min(int(args.dpo_pockets), n_train)
    rng = np.random.default_rng(cfg.train.seed)
    pocket_indices = rng.choice(n_train, size=n_pockets, replace=False).tolist()

    dpo_sampling_cfg = SamplingConfig(
        strategy="min_p",
        num_samples=int(args.dpo_samples_per_pocket),
        max_len=int(args.dpo_max_len),
        temperature=cfg.sampling.temperature,
        min_p=cfg.sampling.min_p,
        top_p=cfg.sampling.top_p,
        top_k=cfg.sampling.top_k,
        cfg_scale=cfg.sampling.cfg_scale,
        latent_temp=cfg.sampling.latent_temp,
        repetition_penalty=cfg.sampling.repetition_penalty,
        syntax_guardrails=False,
    )

    dpo_epochs = int(args.epochs) if args.epochs is not None else int(args.dpo_epochs)
    config_path = cfg.to_yaml(out_dir / "config.yaml")

    print(
        f"[{spec.name}] split={cfg.data.split_type} device={device.type} "
        f"init={init_ckpt} pockets={n_pockets} M={dpo_sampling_cfg.num_samples} "
        f"epochs={dpo_epochs} beta={args.dpo_beta} lr={args.dpo_lr} "
        f"sft_w={args.dpo_sft_weight} accum={args.dpo_grad_accum}",
        flush=True,
    )

    t0 = time.perf_counter()
    preference_pairs: list[PreferencePair] = []
    valid_generated_total = 0
    total_generated = 0

    model.eval()
    for p_idx in pocket_indices:
        raw_item = train_ds[int(p_idx)]
        pocket_batch = collate_pocket_batch([raw_item])
        sampled = sample_molecules(
            model,
            pocket_batch,
            tokenizer,
            dpo_sampling_cfg,
            device=device,
        )
        total_generated += len(sampled)

        # Keep valid unique canonical molecules for preference ranking
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
        valid_generated_total += len(unique_valid)
        if len(unique_valid) < 2:
            continue

        # Score candidates via model's pKd & DrugCLIP heads + medicinal chemistry MPO
        with torch.no_grad():
            coords = pocket_batch["coords"].to(device=device)
            node_feats = pocket_batch["node_feats"].to(device=device)
            atom_mask = pocket_batch["atom_mask"].to(device=device)
            pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
            fps_tensor = _compute_ecfp4_tensor(
                unique_valid,
                fp_dim=cfg.multitask.ligand_fp_dim,
            ).to(device=device)
            cand_scores = model.score_candidates(pocket_out, fps_tensor)
            pred_pkd = (
                cand_scores["pkd"].detach().cpu().numpy().tolist() if "pkd" in cand_scores else None
            )
            clip_sim = (
                cand_scores["clip_sim"].detach().cpu().numpy().tolist()
                if "clip_sim" in cand_scores
                else None
            )

        mpo_entries = compute_mpo_scores(unique_valid, pred_pkd=pred_pkd, clip_sim=clip_sim)
        mpo_vals = [entry.mpo_score for entry in mpo_entries]

        # Store lightweight pocket tensors on CPU to avoid retaining large GPU caches
        compact_pocket = {
            "coords": pocket_batch["coords"].cpu(),
            "node_feats": pocket_batch["node_feats"].cpu(),
            "atom_mask": pocket_batch["atom_mask"].cpu(),
        }
        pocket_pairs = build_preference_pairs_from_mpo(
            compact_pocket,
            unique_valid,
            mpo_vals,
            min_margin=float(args.dpo_min_margin),
            max_pairs=int(args.dpo_max_pairs_per_pocket),
        )
        preference_pairs.extend(pocket_pairs)

    pair_gen_time = time.perf_counter() - t0
    print(
        f"[{spec.name}] Collected {len(preference_pairs)} preference pairs from {n_pockets} "
        f"pockets ({valid_generated_total}/{total_generated} unique valid) in {pair_gen_time:.1f}s",
        flush=True,
    )

    dpo_res = run_dpo_alignment(
        model=model,
        ref_model=ref_model,
        preference_pairs=preference_pairs,
        cfg=cfg,
        device=device,
        tokenizer=tokenizer,
        epochs=dpo_epochs,
        beta=float(args.dpo_beta),
        lr=float(args.dpo_lr),
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.grad_clip,
        length_normalize=True,
        freeze_non_decoder=True,
        sft_nll_weight=float(args.dpo_sft_weight),
        grad_accum_steps=int(args.dpo_grad_accum),
        save_dir=None,
    )

    # Evaluate post-DPO model on validation split for unified comparison metrics
    eval_trainer = PocketTrainer(model=model, cfg=cfg, tokenizer=tokenizer, device=device)
    val_metrics = eval_trainer.evaluate(val_loader)
    wall_time = time.perf_counter() - t0

    dpo_history: list[dict[str, Any]] = list(dpo_res["history"])
    best_ckpt = out_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": cfg.model_dump(mode="python"),
            "current_epoch": len(dpo_history),
            "best_val_objective": float(val_metrics["val_objective"]),
            "val_objective_version": 2,
            "dpo_history": dpo_history,
            "val_metrics": val_metrics,
        },
        best_ckpt,
    )

    history_path = out_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in dpo_history:
            enriched = {**row, **val_metrics}
            handle.write(json.dumps(enriched) + "\n")

    last_dpo = dpo_history[-1] if dpo_history else {}
    dpo_final_loss = (
        float(last_dpo["dpo_loss"])
        if "dpo_loss" in last_dpo and math.isfinite(float(last_dpo["dpo_loss"]))
        else None
    )
    summary: dict[str, Any] = {
        "run_name": spec.name,
        "stage": spec.stage,
        "split_type": cfg.data.split_type,
        "representation": cfg.data.representation,
        "encoder_arch": cfg.encoder.arch,
        "decoder_arch": cfg.decoder.arch,
        "cvae_enabled": cfg.cvae.enabled,
        "init_ckpt_path": str(init_ckpt),
        "device": device.type,
        "seed": cfg.train.seed,
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable),
        "dpo_pockets_sampled": n_pockets,
        "dpo_samples_per_pocket": dpo_sampling_cfg.num_samples,
        "dpo_num_pairs": len(preference_pairs),
        "dpo_final_loss": dpo_final_loss,
        "dpo_final_margin": float(last_dpo.get("reward_margin", 0.0)),
        "dpo_final_accuracy": float(last_dpo.get("preference_accuracy", 0.0)),
        "epochs_completed": len(dpo_history),
        "wall_time_sec": round(wall_time, 3),
        "best_epoch": len(dpo_history),
        "best_val_objective": float(val_metrics["val_objective"]),
        "best_val_lm_loss": float(val_metrics["val_lm_loss"]),
        "best_val_ppl": float(val_metrics["val_ppl"]),
        "best_val_kl_nats": float(val_metrics["val_kl_nats"]),
        "best_val_pkd_pearson": float(val_metrics["val_pkd_pearson"]),
        "best_val_pkd_rmse": float(val_metrics["val_pkd_rmse"]),
        "best_val_clip_r1": float(val_metrics["val_clip_r1"]),
        "best_val_clip_r5": float(val_metrics["val_clip_r5"]),
        "best_ckpt_path": str(best_ckpt),
        "config_path": str(config_path),
        "history_path": str(history_path),
    }

    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    loss_str = f"{dpo_final_loss:.4f}" if dpo_final_loss is not None else "n/a"
    print(
        f"[{spec.name}] DONE in {wall_time:.1f}s | pairs={len(preference_pairs)} "
        f"dpo_loss={loss_str} margin={summary['dpo_final_margin']:.4f} "
        f"val_ppl={summary['best_val_ppl']:.2f} pkd_r={summary['best_val_pkd_pearson']:.3f}",
        flush=True,
    )

    del eval_trainer, ref_model, model
    _clear_accelerator_cache(device)
    return summary


def _parse_requested_runs(args: argparse.Namespace) -> list[str]:
    """Expand ``--all`` and repeated/comma-separated ``--run`` flags into validated names."""
    if args.all:
        return list(DEFAULT_RUN_ORDER)
    requested: list[str] = []
    for raw in args.run or []:
        for token in raw.split(","):
            name = token.strip()
            if not name:
                continue
            if name not in RUN_SPECS:
                valid = ", ".join(DEFAULT_RUN_ORDER)
                raise ValueError(f"Unknown run name {name!r}. Valid runs: {valid}")
            if name not in requested:
                requested.append(name)
    return requested


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the Stage-2 & Stage-3 suite orchestrator."""
    parser = argparse.ArgumentParser(
        description="Resumable Stage-2 & Stage-3 experimental suite runner for GenEuclidean v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help=(
            "Named run(s) to execute (repeatable or comma-separated). "
            f"Choices: {', '.join(DEFAULT_RUN_ORDER)}."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all 9 Stage-2 & Stage-3 runs in dependency order.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all configured suite runs with base configs and defaults, then exit.",
    )
    parser.add_argument(
        "--split",
        choices=["casf2016", "lp_pdbbind", "scaffold", "random"],
        default="casf2016",
        help="Dataset split to train on (lp_pdbbind appends '_lp' to default run dir).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--resume-skip",
        action="store_true",
        help="Skip any run that already has both best.pt and train_summary.json.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume an interrupted Stage-2 run from last.pt (or an explicit checkpoint path).",
    )
    parser.add_argument(
        "--keep-last-ckpt",
        action="store_true",
        help="Keep last.pt after training finishes (deleted by default to save disk space).",
    )

    # Paths & hyperparameter overrides
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing pdbbind_pockets.pt, splits.json, and vocab_*.json.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory where per-model run subdirectories are created.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory override (only valid when executing a single --run).",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        default=None,
        help="Override Stage-1 pretrained decoder checkpoint (pass 'none' to train from scratch).",
    )
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="Parent Stage-2 best.pt checkpoint used to initialize Stage-3 DPO.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--encoder-lr", type=float, default=None, help="Override encoder LR.")
    parser.add_argument("--decoder-lr", type=float, default=None, help="Override decoder LR.")
    parser.add_argument(
        "--kl-max-beta", type=float, default=None, help="Override CVAE kl_max_beta."
    )
    parser.add_argument("--lambda-pkd", type=float, default=None, help="Override pKd loss weight.")
    parser.add_argument(
        "--lambda-clip", type=float, default=None, help="Override DrugCLIP loss weight."
    )
    parser.add_argument(
        "--lambda-prop", type=float, default=None, help="Override property loss weight."
    )
    parser.add_argument(
        "--freeze-decoder-epochs",
        type=int,
        default=None,
        help="Override gradual unfreezing warmup epochs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training dataset size (useful for fast smoke/dry-run verification).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap validation dataset size (useful for fast smoke/dry-run verification).",
    )

    # Stage-3 MPO-Guided DPO options
    parser.add_argument(
        "--dpo-pockets",
        type=int,
        default=500,
        help="Number of training pockets sampled to mine MPO preference pairs for Stage-3 DPO.",
    )
    parser.add_argument(
        "--dpo-samples-per-pocket",
        type=int,
        default=16,
        help="Candidate molecules sampled per pocket for MPO scoring in Stage-3 DPO.",
    )
    parser.add_argument(
        "--dpo-epochs",
        type=int,
        default=1,
        help="Default number of DPO alignment epochs (when --epochs is not set).",
    )
    parser.add_argument("--dpo-beta", type=float, default=0.5, help="DPO KL regularization beta.")
    parser.add_argument("--dpo-lr", type=float, default=1e-5, help="DPO alignment learning rate.")
    parser.add_argument(
        "--dpo-sft-weight",
        type=float,
        default=0.25,
        help="RPO/SFT NLL anchor weight on length-normalized chosen log-probs.",
    )
    parser.add_argument(
        "--dpo-grad-accum",
        type=int,
        default=8,
        help="Number of preference pairs accumulated per optimizer step during DPO.",
    )
    parser.add_argument(
        "--dpo-min-margin",
        type=float,
        default=0.05,
        help="Minimum MPO score gap required between chosen and rejected pairs.",
    )
    parser.add_argument(
        "--dpo-max-pairs-per-pocket",
        type=int,
        default=4,
        help="Maximum preference pairs extracted per pocket.",
    )
    parser.add_argument(
        "--dpo-max-len",
        type=int,
        default=96,
        help="Maximum token length during on-policy DPO candidate generation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Stage-2 & Stage-3 experimental orchestrator."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_runs:
        print(f"{'RUN NAME':<28} {'STAGE':<11} {'PRIOR':<34} DESCRIPTION")
        print("-" * 115)
        for name in DEFAULT_RUN_ORDER:
            spec = RUN_SPECS[name]
            prior = spec.default_pretrained or (
                "parent best.pt" if spec.stage == "stage3_dpo" else "none (scratch)"
            )
            print(f"{spec.name:<28} {spec.stage:<11} {prior:<34} {spec.description}")
        return 0

    run_names = _parse_requested_runs(args)
    if not run_names:
        parser.error("Specify --run <name> or --all (or --list-runs to inspect options).")
    if args.output_dir is not None and len(run_names) > 1:
        parser.error("--output-dir can only be used with a single --run.")

    runs_root = Path(args.runs_root)
    results: list[dict[str, Any]] = []

    for name in run_names:
        spec = RUN_SPECS[name]
        out_dir = _resolve_output_dir(name, args.split, runs_root, args.output_dir)
        best_ckpt = out_dir / "best.pt"
        summary_file = out_dir / "train_summary.json"

        if args.resume_skip and best_ckpt.is_file() and summary_file.is_file():
            existing = json.loads(summary_file.read_text(encoding="utf-8"))
            print(
                f"[{name}] SKIP (--resume-skip): found {best_ckpt} & {summary_file} "
                f"(best_val_ppl={existing.get('best_val_ppl', math.nan):.2f})",
                flush=True,
            )
            results.append(existing)
            continue

        if spec.stage == "stage3_dpo":
            summary = _run_stage3_dpo(spec, args, out_dir)
        else:
            summary = _run_stage2(spec, args, out_dir)
        results.append(summary)

    # Write combined suite manifest when running multiple experiments
    if len(results) > 1:
        manifest_path = runs_root / f"stage2_suite_summary{_split_suffix(args.split)}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved suite manifest to {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
