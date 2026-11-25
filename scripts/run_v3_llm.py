#!/usr/bin/env python3
"""Resumable CLI runner for GenEuclidean v3.0 3D-to-LLM training & LoRA RPO-DPO.

Supports Track A Chemical Foundation Transformers (``gpmolformer``, ``gpt2zinc``),
Track B General LLM Control (``qwen25``), and Stage-3 LoRA RPO-DPO alignment
(``gpmolformer_dpo``, ``gpt2zinc_dpo``, ``qwen25_dpo``).

Usage examples::

    # Train Track A primary (GP-MoLFormer-Uniq) on zero-leakage CASF-2016 split
    .venv/bin/python scripts/run_v3_llm.py --model gpmolformer --split casf2016 --device mps

    # Train Track A2 (GPT2-ZINC-87M)
    .venv/bin/python scripts/run_v3_llm.py --model gpt2zinc --split casf2016 --device mps

    # Train Track B general LLM control (Qwen2.5-0.5B in bfloat16)
    .venv/bin/python scripts/run_v3_llm.py --model qwen25 --split casf2016 --device mps

    # Run LoRA-only RPO-DPO alignment on trained GP-MoLFormer adapter
    .venv/bin/python scripts/run_v3_llm.py --model gpmolformer_dpo --split casf2016 --device mps
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything, select_device
from geneuclidean.models.hf_backbones import load_hf_backbone
from geneuclidean.models.llm_bridge import PocketConditionedLLM
from geneuclidean.training.llm_trainer import (
    PocketLLMTrainer,
    build_pocket_batches_from_records,
    precompute_pocket_cache,
    run_llm_rpo_dpo,
)


@dataclass(frozen=True)
class V3RunSpec:
    """Declarative specification for one GenEuclidean v3.0 model run."""

    key: str
    run_name: str
    config_path: str
    stage: str  # "sft" or "dpo"
    parent_key: str | None = None
    description: str = ""


V3_SPECS: dict[str, V3RunSpec] = {
    "gpmolformer": V3RunSpec(
        key="gpmolformer",
        run_name="geneuclidean_v3_gpmolformer",
        config_path="configs/geneuclidean_v3_gpmolformer.yaml",
        stage="sft",
        description="Track A Primary: Frozen PaiNN-SE3 + Perceiver + GP-MoLFormer-Uniq LoRA",
    ),
    "gpt2zinc": V3RunSpec(
        key="gpt2zinc",
        run_name="geneuclidean_v3_gpt2zinc",
        config_path="configs/geneuclidean_v3_gpt2zinc.yaml",
        stage="sft",
        description="Track A2: Frozen PaiNN-SE3 + Perceiver + GPT2-ZINC-87M LoRA",
    ),
    "qwen25": V3RunSpec(
        key="qwen25",
        run_name="geneuclidean_v3_qwen25",
        config_path="configs/geneuclidean_v3_qwen25.yaml",
        stage="sft",
        description="Track B Control: Frozen PaiNN-SE3 + Perceiver + Qwen2.5-0.5B LoRA",
    ),
    "gpmolformer_dpo": V3RunSpec(
        key="gpmolformer_dpo",
        run_name="geneuclidean_v3_gpmolformer_dpo",
        config_path="configs/geneuclidean_v3_gpmolformer.yaml",
        stage="dpo",
        parent_key="gpmolformer",
        description="Track A + Stage-3 LoRA RPO-DPO: PaiNN-SE3 + GP-MoLFormer-Uniq + RPO-DPO",
    ),
    "gpt2zinc_dpo": V3RunSpec(
        key="gpt2zinc_dpo",
        run_name="geneuclidean_v3_gpt2zinc_dpo",
        config_path="configs/geneuclidean_v3_gpt2zinc.yaml",
        stage="dpo",
        parent_key="gpt2zinc",
        description="Track A2 + Stage-3 LoRA RPO-DPO: PaiNN-SE3 + GPT2-ZINC-87M + RPO-DPO",
    ),
    "qwen25_dpo": V3RunSpec(
        key="qwen25_dpo",
        run_name="geneuclidean_v3_qwen25_dpo",
        config_path="configs/geneuclidean_v3_qwen25.yaml",
        stage="dpo",
        parent_key="qwen25",
        description="Track B + Stage-3 LoRA RPO-DPO: PaiNN-SE3 + Qwen2.5-0.5B + RPO-DPO",
    ),
}


def _split_suffix(split_type: str) -> str:
    """Return canonical directory suffix for non-default benchmark splits."""
    if split_type == "casf2016":
        return ""
    if split_type == "lp_pdbbind":
        return "_lp"
    return f"_{split_type}"


def _clear_accelerator_cache(device: torch.device) -> None:
    """Release cached allocator memory cleanly after a run."""
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_cache_and_split_ids(
    processed_dir: Path,
    split_type: str,
    max_train: int | None = None,
    max_val: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Load preprocessed ``pdbbind_pockets.pt`` records plus train/val PDB IDs."""
    cache_path = processed_dir / "pdbbind_pockets.pt"
    splits_path = processed_dir / "splits.json"
    if not cache_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(
            f"Processed PDBbind cache missing in {processed_dir} "
            "(expected pdbbind_pockets.pt and splits.json)."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records: dict[str, dict[str, Any]] = payload["records"]
    splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
    group = splits_data[split_type]
    train_ids = [pid for pid in group["train"] if pid in records]
    val_ids = [pid for pid in group["val"] if pid in records]
    if max_train is not None and max_train > 0:
        train_ids = train_ids[:max_train]
    if max_val is not None and max_val > 0:
        val_ids = val_ids[:max_val]
    return records, train_ids, val_ids


def _build_v3_model(
    cfg: ExperimentConfig,
    *,
    local_files_only: bool = False,
) -> PocketConditionedLLM:
    """Instantiate HF backbone + ``PocketConditionedLLM`` and warm-start frozen v2 encoder."""
    llm_cfg = cfg.llm if cfg.llm is not None else LLMConfig()
    cache_dir = str(Path(llm_cfg.cache_dir).resolve())
    os.environ["HF_HOME"] = cache_dir

    backbone = load_hf_backbone(
        backbone_id=llm_cfg.backbone_id,
        cache_dir=llm_cfg.cache_dir,
        dtype=llm_cfg.dtype,
        lora_r=llm_cfg.lora_r,
        lora_alpha=llm_cfg.lora_alpha,
        lora_dropout=llm_cfg.lora_dropout,
        use_lora=True,
        local_files_only=local_files_only,
    )
    model = PocketConditionedLLM(
        v2_cfg=cfg,
        backbone=backbone,
        num_queries=llm_cfg.num_queries,
        perceiver_layers=llm_cfg.perceiver_layers,
        perceiver_heads=llm_cfg.perceiver_heads,
        num_global_tokens=llm_cfg.num_global_tokens,
        cfg_drop_prob=llm_cfg.cfg_drop_prob,
        freeze_pocket_encoder=llm_cfg.freeze_pocket_encoder,
    )

    v2_ckpt = Path(llm_cfg.v2_pocket_ckpt)
    if v2_ckpt.is_file():
        model.load_v2_pocket_checkpoint(v2_ckpt)
    else:
        raise FileNotFoundError(
            f"v2 pocket encoder checkpoint not found: {v2_ckpt}. "
            "Pass --v2-ckpt <path> or ensure runs/geneuclidean_v2_painn/best.pt exists."
        )
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for GenEuclidean v3.0 training and RPO-DPO runner."""
    parser = argparse.ArgumentParser(
        description="Resumable CLI trainer & LoRA RPO-DPO runner for GenEuclidean v3.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=tuple(V3_SPECS.keys()),
        required=True,
        help="Target v3.0 model or RPO-DPO alignment run.",
    )
    parser.add_argument(
        "--split",
        choices=("casf2016", "lp_pdbbind", "scaffold", "random"),
        default="casf2016",
        help="Zero-leakage benchmark split from splits.json.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto|mps|cuda|cpu.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config override.")
    parser.add_argument(
        "--v2-ckpt",
        type=Path,
        default=None,
        help="Optional v2.0 PaiNN-SE3 checkpoint override.",
    )
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Parent best_adapter.pt checkpoint for *_dpo runs.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory holding pdbbind_pockets.pt and splits.json.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit run output directory override.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory override.",
    )
    parser.add_argument(
        "--phase-a-epochs",
        type=int,
        default=None,
        help="Override Phase A (projector warmup) epochs.",
    )
    parser.add_argument(
        "--phase-b-epochs",
        type=int,
        default=None,
        help="Override Phase B (joint Perceiver + LoRA) epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument(
        "--dpo-pockets",
        type=int,
        default=250,
        help="Number of training pockets sampled for on-policy RPO-DPO pairs.",
    )
    parser.add_argument(
        "--dpo-samples-per-pocket",
        type=int,
        default=16,
        help="Candidate molecules generated per pocket for RPO-DPO.",
    )
    parser.add_argument("--dpo-epochs", type=int, default=1, help="RPO-DPO epochs.")
    parser.add_argument("--dpo-beta", type=float, default=0.5, help="RPO-DPO KL beta.")
    parser.add_argument("--dpo-lr", type=float, default=1e-5, help="RPO-DPO LoRA learning rate.")
    parser.add_argument(
        "--dpo-sft-weight",
        type=float,
        default=0.25,
        help="RPO SFT NLL anchor weight on chosen SMILES.",
    )
    parser.add_argument(
        "--dpo-grad-accum",
        type=int,
        default=8,
        help="RPO-DPO preference pair gradient accumulation window.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training pockets (for fast smoke runs).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on validation pockets (for fast smoke runs).",
    )
    parser.add_argument(
        "--resume-skip",
        action="store_true",
        help="Skip run cleanly if best_adapter.pt and train_summary.json already exist.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require HuggingFace weights to load strictly from local cache.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for GenEuclidean v3.0 LLM training and RPO-DPO CLI."""
    args = build_arg_parser().parse_args(argv)
    spec = V3_SPECS[args.model]
    suffix = _split_suffix(args.split)
    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.runs_root / f"{spec.run_name}{suffix}"
    )

    if (
        args.resume_skip
        and (out_dir / "best_adapter.pt").is_file()
        and (out_dir / "train_summary.json").is_file()
    ):
        print(f"[{spec.key}] Skipping (already completed in {out_dir}).", flush=True)
        return 0

    cfg_path = args.config if args.config is not None else Path(spec.config_path)
    cfg = ExperimentConfig.from_yaml(cfg_path)
    llm_cfg = cfg.llm if cfg.llm is not None else LLMConfig()

    cfg.data.processed_dir = str(args.processed_dir)
    cfg.data.split_type = args.split
    cfg.train.device = args.device
    cfg.train.output_dir = str(out_dir)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.v2_ckpt is not None:
        llm_cfg.v2_pocket_ckpt = str(args.v2_ckpt)
    if args.cache_dir is not None:
        llm_cfg.cache_dir = str(args.cache_dir)
    if args.phase_a_epochs is not None:
        llm_cfg.phase_a_epochs = int(args.phase_a_epochs)
    if args.phase_b_epochs is not None:
        llm_cfg.phase_b_epochs = int(args.phase_b_epochs)
    if spec.stage == "dpo":
        cfg.name = f"{cfg.name}-dpo"
        cfg.description = spec.description
    cfg.llm = llm_cfg

    seed_everything(cfg.train.seed)
    device = select_device(cfg.train.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, train_ids, val_ids = _load_cache_and_split_ids(
        args.processed_dir,
        args.split,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
    )

    print(
        f"[{spec.key}] stage={spec.stage} backbone={llm_cfg.backbone_id} dtype={llm_cfg.dtype} "
        f"split={args.split} device={device.type} train={len(train_ids)} val={len(val_ids)}",
        flush=True,
    )

    model = _build_v3_model(cfg, local_files_only=args.local_files_only)

    # Precompute frozen 3D PaiNN-SE3 pocket embeddings ONCE on device -> compact CPU cache
    t_cache = time.perf_counter()
    train_batches = build_pocket_batches_from_records(
        records, train_ids, batch_size=32, fp_dim=cfg.multitask.ligand_fp_dim
    )
    val_batches = build_pocket_batches_from_records(
        records, val_ids, batch_size=32, fp_dim=cfg.multitask.ligand_fp_dim
    )
    train_cache = precompute_pocket_cache(model, train_batches, device=device)
    val_cache = precompute_pocket_cache(model, val_batches, device=device)
    print(
        f"[{spec.key}] Precomputed frozen 3D pocket cache ({len(train_cache)} train, "
        f"{len(val_cache)} val) in {time.perf_counter() - t_cache:.1f}s",
        flush=True,
    )

    if spec.stage == "sft":
        trainer = PocketLLMTrainer(model=model, cfg=cfg, device=device)
        summary = trainer.fit(train_cache=train_cache, val_cache=val_cache, output_dir=out_dir)
        print(
            f"[{spec.key}] DONE in {summary['wall_time_sec']:.1f}s | "
            f"best_ep={summary['best_epoch']} val_lm={summary['best_val_lm_loss']} "
            f"val_ppl={summary['best_val_ppl']} ckpt={summary['adapter_ckpt_mb']:.2f}MB",
            flush=True,
        )
    else:
        assert spec.parent_key is not None
        parent_spec = V3_SPECS[spec.parent_key]
        default_parent = args.runs_root / f"{parent_spec.run_name}{suffix}" / "best_adapter.pt"
        init_adapter = args.init_adapter if args.init_adapter is not None else default_parent
        if not init_adapter.is_file():
            raise FileNotFoundError(
                f"Parent SFT adapter checkpoint not found for {spec.key}: {init_adapter}. "
                f"Train --model {spec.parent_key} first or pass --init-adapter <path>."
            )
        model.load_adapter_checkpoint(init_adapter, map_location="cpu")
        model.to(device)
        cfg_path_out = cfg.to_yaml(out_dir / "config.yaml")

        dpo_res = run_llm_rpo_dpo(
            model=model,
            train_records=train_cache,
            cfg=cfg,
            device=device,
            dpo_pockets=args.dpo_pockets,
            samples_per_pocket=args.dpo_samples_per_pocket,
            epochs=args.dpo_epochs,
            beta=args.dpo_beta,
            lr=args.dpo_lr,
            sft_nll_weight=args.dpo_sft_weight,
            grad_accum_steps=args.dpo_grad_accum,
            seed=cfg.train.seed,
            save_dir=out_dir,
        )
        eval_trainer = PocketLLMTrainer(model=model, cfg=cfg, device=device)
        val_metrics = eval_trainer.evaluate(val_cache)
        last_row = dpo_res["history"][-1] if dpo_res["history"] else {}
        summary = {
            "name": cfg.name,
            "stage": "rpo_dpo",
            "backbone_id": llm_cfg.backbone_id,
            "dtype": llm_cfg.dtype,
            "split_type": cfg.data.split_type,
            "device": device.type,
            "seed": cfg.train.seed,
            "init_adapter_path": str(init_adapter),
            "dpo_pockets_sampled": dpo_res["dpo_pockets_sampled"],
            "dpo_num_pairs": dpo_res["num_pairs"],
            "dpo_final_loss": last_row.get("dpo_loss"),
            "dpo_final_margin": last_row.get("reward_margin"),
            "dpo_final_accuracy": last_row.get("preference_accuracy"),
            "best_val_lm_loss": val_metrics["val_lm_loss"],
            "best_val_ppl": val_metrics["val_ppl"],
            "wall_time_sec": dpo_res.get("wall_time_sec"),
            "best_adapter_path": dpo_res.get("adapter_ckpt_path"),
            "config_path": str(cfg_path_out),
            "history_path": str(out_dir / "history.jsonl"),
        }
        (out_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{spec.key}] DPO DONE | pairs={summary['dpo_num_pairs']} "
            f"margin={summary['dpo_final_margin']} val_ppl={summary['best_val_ppl']:.2f}",
            flush=True,
        )

    del model
    _clear_accelerator_cache(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
