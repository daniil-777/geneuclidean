"""Command-line interface for GenEuclidean v3.0 (Typer + Rich)."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

import geneuclidean

if TYPE_CHECKING:
    import torch

    from geneuclidean.config import ExperimentConfig

app = typer.Typer(
    name="geneuclidean",
    help=(
        "GenEuclidean v3.0 — SE(3)/E(3)-equivariant 3D protein pocket-conditioned "
        "Chemical Foundation Transformer & Local LLM de novo drug design."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)
console = Console()


def _json_safe(obj: Any) -> Any:
    """Recursively convert NumPy/Path/non-finite floats into strict JSON-serializable values."""
    import numpy as np

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _resolve_vocab_path(cfg: ExperimentConfig, vocab_path: Path | None) -> Path:
    """Resolve JSON vocabulary path from explicit argument or processed cache directory."""
    if vocab_path is not None:
        if not vocab_path.is_file():
            raise typer.BadParameter(f"Vocabulary file not found: {vocab_path}")
        return vocab_path
    rep = cfg.data.representation
    candidate = Path(cfg.data.processed_dir) / f"vocab_{rep}.json"
    if not candidate.is_file():
        raise typer.BadParameter(
            f"Expected vocabulary at {candidate} (pass --vocab or run `geneuclidean prepare`)."
        )
    return candidate


def _load_checkpoint_bundle(
    checkpoint: Path,
    vocab_path: Path | None,
    device: torch.device,
) -> tuple[Any, ExperimentConfig, Any]:
    """Load a trained ``GenEuclideanModel``, validated ``ExperimentConfig``, and tokenizer."""
    import torch

    from geneuclidean.config import ExperimentConfig
    from geneuclidean.data.tokenizer import MolecularTokenizer
    from geneuclidean.models.geneuclidean import GenEuclideanModel

    if not checkpoint.is_file():
        raise typer.BadParameter(f"Checkpoint file not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise typer.BadParameter(
            f"Checkpoint {checkpoint} does not contain a 'model_state_dict' key."
        )

    raw_cfg = payload.get("config")
    cfg = (
        ExperimentConfig.model_validate(raw_cfg)
        if isinstance(raw_cfg, dict)
        else ExperimentConfig()
    )
    resolved_vocab = _resolve_vocab_path(cfg, vocab_path)
    tokenizer = MolecularTokenizer.load(resolved_vocab)
    cfg = cfg.model_copy(
        update={"decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})}
    )

    model = GenEuclideanModel(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, tokenizer


def _parse_center_arg(center: str | None) -> tuple[float, float, float] | None:
    """Parse a comma-separated ``x,y,z`` coordinate triplet."""
    if center is None:
        return None
    parts = [p.strip() for p in center.split(",") if p.strip()]
    if len(parts) != 3:
        raise typer.BadParameter(f"--center must have 3 comma-separated floats, got: {center!r}")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid numeric --center value: {center!r}") from exc


def _resolve_pocket_for_sampling(
    *,
    pdb_id: str | None,
    pdb_file: Path | None,
    center: tuple[float, float, float] | None,
    cfg: ExperimentConfig,
) -> tuple[dict[str, Any], str, Path | None, Path | None, str | None]:
    """Build single-pocket tensors from a local PDB file, processed cache, raw dir, or RCSB."""
    import torch

    from geneuclidean.data.download import fetch_rcsb_complex
    from geneuclidean.data.featurizer import (
        extract_pocket_from_files,
        featurize_pdb_pocket_inference,
    )

    if pdb_file is None and not pdb_id:
        raise typer.BadParameter("Provide either --pdb-file or --pdb-id.")

    if pdb_file is not None:
        if not pdb_file.is_file():
            raise typer.BadParameter(f"PDB/mmCIF file not found: {pdb_file}")
        feat = featurize_pdb_pocket_inference(
            pdb_file,
            center=center,
            pocket_cutoff=cfg.data.pocket_cutoff,
            max_atoms=cfg.data.max_pocket_atoms,
            pdb_id=pdb_id,
        )
        batch = {
            "coords": torch.from_numpy(feat.coords).unsqueeze(0),
            "node_feats": torch.from_numpy(feat.node_feats).unsqueeze(0),
            "atom_mask": torch.ones((1, len(feat.coords)), dtype=torch.bool),
            "contact_mask": torch.from_numpy(feat.contact_mask).unsqueeze(0),
        }
        return batch, feat.pdb_id, pdb_file, None, None

    assert pdb_id is not None
    pid = pdb_id.strip().lower()
    raw_root = Path(cfg.data.raw_dir)
    proc_root = Path(cfg.data.processed_dir)

    pocket_pdb: Path | None = None
    ligand_sdf: Path | None = None
    ligand_mol2: Path | None = None
    for subdir in ("pbpp2020", "casf2016_extra"):
        cand = raw_root / subdir / pid / f"{pid}_pocket.pdb"
        if cand.is_file():
            pocket_pdb = cand
            sdf_c = cand.parent / f"{pid}_ligand.sdf"
            mol2_c = cand.parent / f"{pid}_ligand.mol2"
            ligand_sdf = sdf_c if sdf_c.is_file() else None
            ligand_mol2 = mol2_c if mol2_c.is_file() else None
            break

    cache_path = proc_root / "pdbbind_pockets.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        records = payload.get("records", {})
        if pid in records:
            rec = records[pid]
            coords = torch.as_tensor(rec["coords"], dtype=torch.float32).unsqueeze(0)
            node_feats = torch.as_tensor(rec["node_feats"], dtype=torch.float32).unsqueeze(0)
            contact = torch.as_tensor(rec["contact_mask"], dtype=torch.bool).unsqueeze(0)
            batch = {
                "coords": coords,
                "node_feats": node_feats,
                "atom_mask": torch.ones((1, coords.shape[1]), dtype=torch.bool),
                "contact_mask": contact,
            }
            native_smi = str(rec["canonical_smiles"]) if rec.get("canonical_smiles") else None
            return batch, pid, pocket_pdb, ligand_sdf, native_smi

    if pocket_pdb is not None:
        feat = extract_pocket_from_files(
            pocket_pdb,
            ligand_sdf=ligand_sdf,
            ligand_mol2=ligand_mol2,
            pdb_id=pid,
            pocket_cutoff=cfg.data.pocket_cutoff,
            max_atoms=cfg.data.max_pocket_atoms,
        )
        batch = {
            "coords": torch.from_numpy(feat.coords).unsqueeze(0),
            "node_feats": torch.from_numpy(feat.node_feats).unsqueeze(0),
            "atom_mask": torch.ones((1, len(feat.coords)), dtype=torch.bool),
            "contact_mask": torch.from_numpy(feat.contact_mask).unsqueeze(0),
        }
        return batch, pid, pocket_pdb, ligand_sdf, feat.canonical_smiles

    # Fallback: fetch novel target live from RCSB PDB
    console.print(f"[cyan]PDB '{pid}' not in local cache; fetching live from RCSB...[/cyan]")
    fetched = fetch_rcsb_complex(
        pid,
        out_dir=raw_root / "rcsb_cache",
        pocket_cutoff=cfg.data.pocket_cutoff,
    )
    fetched_pdb = Path(fetched["pocket_pdb"])
    fetched_center = center if center is not None else fetched.get("ligand_center")
    feat = featurize_pdb_pocket_inference(
        fetched_pdb,
        center=fetched_center,
        pocket_cutoff=cfg.data.pocket_cutoff,
        max_atoms=cfg.data.max_pocket_atoms,
        pdb_id=pid,
    )
    batch = {
        "coords": torch.from_numpy(feat.coords).unsqueeze(0),
        "node_feats": torch.from_numpy(feat.node_feats).unsqueeze(0),
        "atom_mask": torch.ones((1, len(feat.coords)), dtype=torch.bool),
        "contact_mask": torch.from_numpy(feat.contact_mask).unsqueeze(0),
    }
    return (
        batch,
        pid,
        fetched_pdb,
        Path(fetched["ligand_sdf"]) if fetched.get("ligand_sdf") else None,
        fetched.get("ligand_smiles"),
    )


@app.command()
def download(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Destination directory for raw datasets."),
    bin_dir: Path = typer.Option(Path("bin"), help="Destination directory for AutoDock Vina."),
    include_vina: bool = typer.Option(
        True,
        "--include-vina/--no-vina",
        help="Also download precompiled AutoDock Vina v1.2.7 binary.",
    ),
    overwrite: bool = typer.Option(False, help="Re-download existing dataset files."),
) -> None:
    """Download the lean GenEuclidean v2 dataset stack (PDBbind, LP-PDBBind, CASF-2016, ZINC)."""
    from geneuclidean.data.download import download_all_datasets

    results = download_all_datasets(
        raw_dir=raw_dir,
        bin_dir=bin_dir,
        include_vina=include_vina,
        overwrite=overwrite,
    )
    table = Table(title="Downloaded GenEuclidean v2 Datasets")
    table.add_column("Asset", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Exists", style="bold")
    for name, path in results.items():
        table.add_row(name, str(path), "YES" if path.exists() else "NO")
    console.print(table)


@app.command()
def prepare(
    config: Path | None = typer.Option(None, "--config", "-c", help="Optional YAML config."),
    raw_dir: Path | None = typer.Option(None, help="Raw dataset directory override."),
    processed_dir: Path | None = typer.Option(None, help="Processed cache output override."),
    metadata_dir: Path | None = typer.Option(None, help="Metadata label CSV directory override."),
    pocket_cutoff: float | None = typer.Option(None, help="Pocket shell cutoff in Angstroms."),
    max_atoms: int | None = typer.Option(None, help="Maximum heavy pocket atoms retained."),
    seed: int | None = typer.Option(None, help="Random seed for splits and vocab fitting."),
    max_workers: int | None = typer.Option(None, help="Parallel worker processes."),
) -> None:
    """Build processed 3D pocket caches, leak-proof splits, and JSON vocabularies."""
    from geneuclidean.config import ExperimentConfig
    from geneuclidean.data.dataset import build_processed_cache

    cfg = ExperimentConfig.from_yaml(config) if config is not None else ExperimentConfig()
    summary = build_processed_cache(
        raw_dir=raw_dir if raw_dir is not None else cfg.data.raw_dir,
        processed_dir=processed_dir if processed_dir is not None else cfg.data.processed_dir,
        metadata_dir=metadata_dir if metadata_dir is not None else cfg.data.metadata_dir,
        pocket_cutoff=pocket_cutoff if pocket_cutoff is not None else cfg.data.pocket_cutoff,
        max_atoms=max_atoms if max_atoms is not None else cfg.data.max_pocket_atoms,
        seed=seed if seed is not None else cfg.train.seed,
        max_workers=max_workers,
    )

    table = Table(title="Processed Dataset Cache Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Discovered complexes", str(summary["n_discovered"]))
    table.add_row("Parsed complexes", str(summary["n_parsed"]))
    table.add_row("Failed complexes", str(summary["n_failed"]))
    table.add_row("ZINC-250k molecules", str(summary["n_zinc"]))
    table.add_row("SMILES vocab size", str(summary["smiles_vocab_size"]))
    table.add_row("SELFIES vocab size", str(summary["selfies_vocab_size"]))
    table.add_row("Cache path", str(summary["cache_path"]))
    table.add_row("Splits path", str(summary["splits_path"]))
    console.print(table)


@app.command()
def pretrain(
    config: Path = typer.Option(..., "--config", "-c", help="Stage-1 pretraining YAML config."),
    vocab: Path | None = typer.Option(None, help="Override path to JSON tokenizer vocabulary."),
    output_dir: Path | None = typer.Option(None, help="Override run output directory."),
    epochs: int | None = typer.Option(None, help="Override number of pretraining epochs."),
    batch_size: int | None = typer.Option(None, help="Override batch size."),
    max_molecules: int | None = typer.Option(None, help="Optional cap on ZINC molecules."),
    device: str | None = typer.Option(None, help="Override compute device (auto|mps|cuda|cpu)."),
) -> None:
    """Run Stage-1 unconditional chemical LM pretraining on ZINC-250k."""
    from torch.utils.data import DataLoader

    from geneuclidean.config import ExperimentConfig
    from geneuclidean.data.dataset import (
        ZincSMILESDataset,
        collate_smiles_batch,
        load_zinc_splits,
    )
    from geneuclidean.data.tokenizer import MolecularTokenizer
    from geneuclidean.training.pretrain import pretrain_chemical_prior

    cfg = ExperimentConfig.from_yaml(config)
    train_updates: dict[str, Any] = {}
    if output_dir is not None:
        train_updates["output_dir"] = str(output_dir)
    if epochs is not None:
        train_updates["epochs"] = epochs
    if batch_size is not None:
        train_updates["batch_size"] = batch_size
    if device is not None:
        train_updates["device"] = device
    if train_updates:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update=train_updates)})

    vocab_path = _resolve_vocab_path(cfg, vocab)
    tokenizer = MolecularTokenizer.load(vocab_path)
    cfg = cfg.model_copy(
        update={"decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})}
    )

    zinc_txt = Path(cfg.data.processed_dir) / "zinc250k_smiles.txt"
    if not zinc_txt.is_file():
        raise typer.BadParameter(
            f"ZINC SMILES corpus not found at {zinc_txt}. Run `geneuclidean prepare` first."
        )

    train_smiles, val_smiles = load_zinc_splits(
        zinc_txt,
        val_fraction=0.05,
        seed=cfg.train.seed,
        max_molecules=max_molecules,
    )
    train_ds = ZincSMILESDataset(
        train_smiles,
        tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=cfg.data.randomize_smiles,
        seed=cfg.train.seed,
    )
    val_ds = ZincSMILESDataset(
        val_smiles,
        tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_smiles_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_smiles_batch,
    )

    summary = pretrain_chemical_prior(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        save_dir=cfg.train.output_dir,
    )

    table = Table(title=f"Stage-1 Pretraining Complete ({cfg.name})")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Best epoch", str(summary["best_epoch"]))
    table.add_row("Best val loss", f"{summary['best_val_loss']:.4f}")
    table.add_row("Best val perplexity", f"{summary['best_val_ppl']:.3f}")
    table.add_row("Checkpoint", str(summary["best_ckpt_path"]))
    console.print(table)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Stage-2 experiment YAML config."),
    vocab: Path | None = typer.Option(None, help="Override path to JSON tokenizer vocabulary."),
    output_dir: Path | None = typer.Option(None, help="Override run output directory."),
    resume_from: Path | None = typer.Option(None, help="Checkpoint path to resume training from."),
    epochs: int | None = typer.Option(None, help="Override number of training epochs."),
    batch_size: int | None = typer.Option(None, help="Override batch size."),
    device: str | None = typer.Option(None, help="Override compute device (auto|mps|cuda|cpu)."),
) -> None:
    """Run Stage-2 3D pocket-conditioned multi-task training via PocketTrainer."""
    from torch.utils.data import DataLoader

    from geneuclidean.config import ExperimentConfig, seed_everything
    from geneuclidean.data.dataset import PocketLigandDataset, collate_pocket_batch
    from geneuclidean.data.tokenizer import MolecularTokenizer
    from geneuclidean.models.geneuclidean import GenEuclideanModel
    from geneuclidean.training.trainer import PocketTrainer

    cfg = ExperimentConfig.from_yaml(config)
    train_updates: dict[str, Any] = {}
    if output_dir is not None:
        train_updates["output_dir"] = str(output_dir)
    if epochs is not None:
        train_updates["epochs"] = epochs
    if batch_size is not None:
        train_updates["batch_size"] = batch_size
    if device is not None:
        train_updates["device"] = device
    if train_updates:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update=train_updates)})

    seed_everything(cfg.train.seed)
    vocab_path = _resolve_vocab_path(cfg, vocab)
    tokenizer = MolecularTokenizer.load(vocab_path)
    cfg = cfg.model_copy(
        update={"decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})}
    )

    proc_root = Path(cfg.data.processed_dir)
    cache_path = proc_root / "pdbbind_pockets.pt"
    splits_path = proc_root / "splits.json"
    if not cache_path.is_file() or not splits_path.is_file():
        raise typer.BadParameter(
            f"Processed cache not found in {proc_root}. Run `geneuclidean prepare` first."
        )

    train_ds = PocketLigandDataset(
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
    val_ds = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type=cfg.data.split_type,
        split_name="val",
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pocket_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pocket_batch,
    )

    model = GenEuclideanModel(cfg)
    trainer = PocketTrainer(model=model, cfg=cfg, tokenizer=tokenizer)
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=cfg.train.output_dir,
        resume_from=resume_from,
    )

    table = Table(title=f"Stage-2 Training Complete ({cfg.name})")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Encoder / Decoder", f"{cfg.encoder.arch} / {cfg.decoder.arch}")
    table.add_row("Split", f"{cfg.data.split_type} (train={len(train_ds)}, val={len(val_ds)})")
    table.add_row("Best val objective", f"{summary['best_val_objective']:.4f}")
    table.add_row("Best checkpoint", str(summary["best_ckpt_path"]))
    table.add_row("History log", str(summary["history_path"]))
    console.print(table)


@app.command()
def sample(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Trained Stage-2 checkpoint path."),
    pdb_id: str | None = typer.Option(None, "--pdb-id", help="4-char PDB ID (cache or RCSB)."),
    pdb_file: Path | None = typer.Option(None, "--pdb-file", help="Path to local PDB/mmCIF file."),
    center: str | None = typer.Option(
        None,
        help="Optional binding-site center 'x,y,z' in Angstroms (for --pdb-file).",
    ),
    vocab: Path | None = typer.Option(None, help="Override JSON tokenizer vocabulary path."),
    strategy: str | None = typer.Option(
        None,
        help="Decoding strategy (min_p|top_p|top_k|temperature|greedy|typical|diverse_beam).",
    ),
    num_samples: int | None = typer.Option(None, help="Number of molecules to generate."),
    temperature: float | None = typer.Option(None, help="Sampling temperature."),
    min_p: float | None = typer.Option(None, help="Min-p threshold."),
    cfg_scale: float | None = typer.Option(None, help="Classifier-Free Guidance scale."),
    latent_temp: float | None = typer.Option(None, help="CVAE prior sampling temperature."),
    device: str = typer.Option("auto", help="Compute device (auto|mps|cuda|cpu)."),
    seed: int = typer.Option(42, help="Sampling random seed."),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Apply MPO scoring and MaxMin diverse hit selection.",
    ),
    top_k_hits: int = typer.Option(16, help="Number of diverse reranked hits to display/save."),
    eval_3d: bool = typer.Option(
        False,
        "--eval-3d/--no-eval-3d",
        help="Run RDKit ETKDGv3 + MMFF94s + PoseBusters 3D plausibility checks.",
    ),
    run_vina: bool = typer.Option(
        False,
        "--run-vina/--no-run-vina",
        help="Run local AutoDock Vina docking during 3D evaluation.",
    ),
    vina_bin: Path = typer.Option(Path("bin/vina"), help="Path to AutoDock Vina executable."),
    out_json: Path | None = typer.Option(None, help="Optional JSON path to save sampled results."),
) -> None:
    """Sample de novo ligands for a 3D protein pocket with optional MPO reranking and 3D checks."""
    import numpy as np
    import torch

    from geneuclidean.config import seed_everything, select_device
    from geneuclidean.data.featurizer import featurize_ligand_smiles
    from geneuclidean.evaluation.geometry3d import evaluate_3d_plausibility
    from geneuclidean.sampling.decoder_strategies import sample_molecules_detailed
    from geneuclidean.sampling.rerank import compute_mpo_scores, rerank_and_select

    seed_everything(seed)
    torch_device = select_device(device)
    model, cfg, tokenizer = _load_checkpoint_bundle(checkpoint, vocab, torch_device)

    center_tuple = _parse_center_arg(center)
    pocket_batch, resolved_id, pocket_pdb, native_sdf, native_smiles = _resolve_pocket_for_sampling(
        pdb_id=pdb_id,
        pdb_file=pdb_file,
        center=center_tuple,
        cfg=cfg,
    )

    samp_updates: dict[str, Any] = {}
    if strategy is not None:
        samp_updates["strategy"] = cast(
            Literal[
                "min_p",
                "top_p",
                "top_k",
                "temperature",
                "greedy",
                "typical",
                "diverse_beam",
            ],
            strategy,
        )
    if num_samples is not None:
        samp_updates["num_samples"] = num_samples
    if temperature is not None:
        samp_updates["temperature"] = temperature
    if min_p is not None:
        samp_updates["min_p"] = min_p
    if cfg_scale is not None:
        samp_updates["cfg_scale"] = cfg_scale
    if latent_temp is not None:
        samp_updates["latent_temp"] = latent_temp
    sampling_cfg = cfg.sampling.model_copy(update=samp_updates) if samp_updates else cfg.sampling

    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)

    result = sample_molecules_detailed(
        model=model,
        pocket_batch=pocket_batch,
        tokenizer=tokenizer,
        sampling_cfg=sampling_cfg,
        generator=gen,
        device=torch_device,
        return_attn=False,
    )

    # Optional multi-task scoring (pKd & DrugCLIP) using single encoded pocket
    pred_pkd_arr: list[float] | None = None
    clip_sim_arr: list[float] | None = None
    if cfg.multitask.use_pkd_head or cfg.multitask.use_clip_head:
        fps: list[np.ndarray] = []
        valid_indices: list[int] = []
        for idx, smi in enumerate(result.smiles):
            parsed = featurize_ligand_smiles(smi, fp_size=cfg.multitask.ligand_fp_dim)
            if parsed is not None:
                valid_indices.append(idx)
                fps.append(parsed[1])
        if valid_indices:
            with torch.no_grad():
                p_out = model.encode_pocket(
                    pocket_batch["coords"].to(torch_device),
                    pocket_batch["node_feats"].to(torch_device),
                    pocket_batch["atom_mask"].to(torch_device),
                )
                fp_tensor = torch.from_numpy(np.stack(fps, axis=0)).to(
                    device=torch_device, dtype=torch.float32
                )
                head_scores = model.score_candidates(p_out, fp_tensor)
            if "pkd" in head_scores:
                pred_pkd_arr = [float("nan")] * len(result.smiles)
                pkd_vals = head_scores["pkd"].detach().cpu().tolist()
                for idx, val in zip(valid_indices, pkd_vals, strict=True):
                    pred_pkd_arr[idx] = float(val)
            if "clip_sim" in head_scores:
                clip_sim_arr = [float("nan")] * len(result.smiles)
                clip_vals = head_scores["clip_sim"].detach().cpu().tolist()
                for idx, val in zip(valid_indices, clip_vals, strict=True):
                    clip_sim_arr[idx] = float(val)

    selected_hits = (
        rerank_and_select(
            result.smiles,
            k=top_k_hits,
            pred_pkd=pred_pkd_arr,
            clip_sim=clip_sim_arr,
            relax_if_needed=True,
        )
        if rerank
        else compute_mpo_scores(result.smiles, pred_pkd=pred_pkd_arr, clip_sim=clip_sim_arr)[
            :top_k_hits
        ]
    )

    plausibility_3d: dict[str, Any] | None = None
    if eval_3d:
        eval_smiles = [h.canonical_smiles for h in selected_hits if h.canonical_smiles]
        if eval_smiles:
            plausibility_3d = evaluate_3d_plausibility(
                eval_smiles,
                pocket_pdb_path=pocket_pdb,
                native_ligand_sdf=native_sdf,
                run_vina=run_vina,
                vina_bin=vina_bin,
                seed=seed,
            )

    table = Table(
        title=(
            f"Sampled Ligands for {resolved_id.upper()} "
            f"(strategy={sampling_cfg.strategy}, cfg={sampling_cfg.cfg_scale:.2f}, "
            f"raw={len(result.smiles)}, selected={len(selected_hits)})"
        )
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Canonical SMILES", style="green")
    table.add_column("QED", justify="right")
    table.add_column("SA", justify="right")
    table.add_column("pKd", justify="right")
    table.add_column("MPO", justify="right", style="cyan")
    for rank, hit in enumerate(selected_hits, start=1):
        table.add_row(
            str(rank),
            hit.canonical_smiles or hit.smiles,
            f"{hit.qed:.3f}",
            f"{hit.sa_score:.2f}",
            f"{hit.pred_pkd:.2f}" if hit.pred_pkd is not None else "-",
            f"{hit.mpo_score:.3f}",
        )
    console.print(table)

    if out_json is not None:
        payload_out: dict[str, Any] = {
            "pdb_id": resolved_id,
            "native_smiles": native_smiles,
            "checkpoint": str(checkpoint),
            "sampling_config": sampling_cfg.model_dump(mode="python"),
            "raw_smiles": result.smiles,
            "selected_hits": [
                {
                    "smiles": h.smiles,
                    "canonical_smiles": h.canonical_smiles,
                    "valid": h.valid,
                    "qed": h.qed,
                    "sa_score": h.sa_score,
                    "sa_norm": h.sa_norm,
                    "lipinski_violations": h.lipinski_violations,
                    "lipinski_pass": h.lipinski_pass,
                    "pains_free": h.pains_free,
                    "pred_pkd": h.pred_pkd,
                    "clip_sim": h.clip_sim,
                    "mpo_score": h.mpo_score,
                }
                for h in selected_hits
            ],
            "plausibility_3d": plausibility_3d,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload_out), indent=2) + "\n", encoding="utf-8")
        console.print(f"[bold green]Saved sampling output to {out_json}[/bold green]")


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Trained Stage-2 checkpoint path."),
    split: str = typer.Option(
        "test",
        "--split",
        help="Split partition to evaluate (val|test|casf2016_core).",
    ),
    split_type: str | None = typer.Option(
        None,
        help="Override split type (lp_pdbbind|casf2016|scaffold|random).",
    ),
    processed_dir: Path | None = typer.Option(None, help="Override processed data cache dir."),
    vocab: Path | None = typer.Option(None, help="Override JSON tokenizer vocabulary path."),
    num_pockets: int | None = typer.Option(
        None,
        help="Optional cap on number of pockets evaluated (useful for quick checks).",
    ),
    num_samples: int = typer.Option(16, help="Molecules sampled per pocket."),
    strategy: str | None = typer.Option(None, help="Override decoding strategy."),
    cfg_scale: float | None = typer.Option(None, help="Override CFG guidance scale."),
    device: str = typer.Option("auto", help="Compute device (auto|mps|cuda|cpu)."),
    seed: int = typer.Option(42, help="Evaluation random seed."),
    eval_3d: bool = typer.Option(
        False,
        "--eval-3d/--no-eval-3d",
        help="Also run Tier-5 RDKit 3D conformer & PoseBusters plausibility checks.",
    ),
    run_vina: bool = typer.Option(
        False,
        "--run-vina/--no-run-vina",
        help="Run local AutoDock Vina docking inside Tier-5 evaluation.",
    ),
    vina_bin: Path = typer.Option(Path("bin/vina"), help="Path to AutoDock Vina executable."),
    out_dir: Path = typer.Option(Path("results/eval"), help="Directory for JSON & PNG outputs."),
) -> None:
    """Run the 5-tier SBDD evaluation suite and save JSON summaries and figures."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from geneuclidean.config import SamplingConfig, seed_everything, select_device
    from geneuclidean.data.dataset import PocketLigandDataset, collate_pocket_batch
    from geneuclidean.evaluation.geometry3d import evaluate_3d_plausibility
    from geneuclidean.evaluation.metrics import (
        compute_properties_batch,
        evaluate_affinity_metrics,
        evaluate_attention_enrichment,
        evaluate_clip_retrieval,
        evaluate_generation_metrics,
        evaluate_pocket_specificity,
    )
    from geneuclidean.evaluation.visualization import (
        plot_matched_vs_shuffled_similarity,
        plot_pkd_parity,
        plot_property_distributions,
    )
    from geneuclidean.sampling.decoder_strategies import sample_molecules, sample_with_attention

    seed_everything(seed)
    torch_device = select_device(device)
    model, cfg, tokenizer = _load_checkpoint_bundle(checkpoint, vocab, torch_device)

    eff_split_type = split_type if split_type is not None else cfg.data.split_type
    proc_root = processed_dir if processed_dir is not None else Path(cfg.data.processed_dir)
    cache_path = proc_root / "pdbbind_pockets.pt"
    splits_path = proc_root / "splits.json"
    if not cache_path.is_file() or not splits_path.is_file():
        raise typer.BadParameter(f"Processed cache not found in {proc_root}.")

    train_ds = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type=eff_split_type,
        split_name="train",
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=False,
    )
    eval_ds = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type=eff_split_type,
        split_name=split,
        max_seq_len=cfg.data.max_seq_len,
        randomize_smiles=False,
    )
    if len(eval_ds) == 0:
        raise typer.BadParameter(f"Evaluation split '{eff_split_type}/{split}' is empty.")

    # Tier 3: Binding affinity (pKd) & DrugCLIP retrieval across the split
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pocket_batch,
    )
    pkd_true_list: list[np.ndarray] = []
    pkd_pred_list: list[np.ndarray] = []
    clip_p_list: list[np.ndarray] = []
    clip_l_list: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in eval_loader:
            moved = {
                k: (v.to(torch_device) if isinstance(v, torch.Tensor) else v)
                for k, v in raw_batch.items()
            }
            out = model(moved, cfg_drop=False, return_attn=False)
            if out.pred_pkd is not None and "pkd" in moved:
                pkd_true_list.append(moved["pkd"].detach().cpu().numpy().reshape(-1))
                pkd_pred_list.append(out.pred_pkd.detach().cpu().numpy().reshape(-1))
            if out.clip_pocket is not None and out.clip_ligand is not None:
                clip_p_list.append(out.clip_pocket.detach().cpu().numpy())
                clip_l_list.append(out.clip_ligand.detach().cpu().numpy())

    tier3_affinity: dict[str, float] | None = None
    y_true_all = np.concatenate(pkd_true_list) if pkd_true_list else np.empty((0,))
    y_pred_all = np.concatenate(pkd_pred_list) if pkd_pred_list else np.empty((0,))
    if y_true_all.size > 0:
        tier3_affinity = evaluate_affinity_metrics(y_true_all, y_pred_all)

    tier3_clip: dict[str, float] | None = None
    if clip_p_list and clip_l_list:
        tier3_clip = evaluate_clip_retrieval(
            np.concatenate(clip_p_list, axis=0),
            np.concatenate(clip_l_list, axis=0),
        )

    # Tiers 1, 2, 4: Per-pocket generation, specificity controls & attention enrichment
    samp_updates: dict[str, Any] = {"num_samples": num_samples}
    if strategy is not None:
        samp_updates["strategy"] = cast(
            Literal[
                "min_p",
                "top_p",
                "top_k",
                "temperature",
                "greedy",
                "typical",
                "diverse_beam",
            ],
            strategy,
        )
    if cfg_scale is not None:
        samp_updates["cfg_scale"] = cfg_scale
    samp_cfg: SamplingConfig = cfg.sampling.model_copy(update=samp_updates)
    null_samp_cfg = samp_cfg.model_copy(update={"cfg_scale": 0.0})

    n_eval = len(eval_ds) if num_pockets is None else min(len(eval_ds), num_pockets)
    per_pocket_gen: dict[str, list[str]] = {}
    per_pocket_null: dict[str, list[str]] = {}
    per_pocket_native: dict[str, str] = {}
    all_generated: list[str] = []
    attn_enrichments: list[dict[str, float]] = []

    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)

    for idx in range(n_eval):
        item = eval_ds[idx]
        pid = str(item["pdb_id"])
        per_pocket_native[pid] = str(item["canonical_smiles"])
        p_batch = {
            "coords": item["coords"].unsqueeze(0),
            "node_feats": item["node_feats"].unsqueeze(0),
            "atom_mask": torch.ones((1, item["coords"].shape[0]), dtype=torch.bool),
            "contact_mask": item["contact_mask"].unsqueeze(0),
        }

        smiles_list, _, attn_maps = sample_with_attention(
            model=model,
            pocket_batch=p_batch,
            tokenizer=tokenizer,
            cfg=samp_cfg,
            generator=gen,
            device=torch_device,
        )
        per_pocket_gen[pid] = smiles_list
        all_generated.extend(smiles_list)

        # Tier-4 contact attention enrichment
        valid_maps = [m.numpy() for m in attn_maps if m is not None and m.numel() > 0]
        if valid_maps:
            mean_attn = np.mean([m.mean(axis=0) for m in valid_maps], axis=0)
            atom_m = np.ones(len(mean_attn), dtype=bool)
            contact_m = item["contact_mask"].numpy()[: len(mean_attn)].astype(bool)
            attn_enrichments.append(evaluate_attention_enrichment(mean_attn, atom_m, contact_m))

        # Null-pocket unconditional control
        per_pocket_null[pid] = sample_molecules(
            model=model,
            pocket_batch=p_batch,
            tokenizer=tokenizer,
            sampling_cfg=null_samp_cfg,
            generator=gen,
            device=torch_device,
        )

    # Cyclic derangement for shuffled-pocket control
    pocket_ids = list(per_pocket_gen.keys())
    per_pocket_shuffled: dict[str, list[str]] = {}
    if len(pocket_ids) >= 2:
        for i, pid in enumerate(pocket_ids):
            other_pid = pocket_ids[(i + 1) % len(pocket_ids)]
            per_pocket_shuffled[pid] = per_pocket_gen[other_pid]
    else:
        per_pocket_shuffled = dict(per_pocket_gen)

    train_smiles = [str(train_ds._records[pid]["canonical_smiles"]) for pid in train_ds.pdb_ids]
    ref_smiles = list(per_pocket_native.values())

    tier1_gen = evaluate_generation_metrics(
        generated_smiles=all_generated,
        train_smiles=train_smiles,
        reference_smiles=ref_smiles,
        seed=seed,
    )
    tier2_spec = evaluate_pocket_specificity(
        per_pocket_generated=per_pocket_gen,
        per_pocket_native_smiles=per_pocket_native,
        per_pocket_shuffled_generated=per_pocket_shuffled,
        per_pocket_null_generated=per_pocket_null,
        n_bootstrap=200,
        seed=seed,
    )

    tier4_attn: dict[str, float] | None = None
    if attn_enrichments:
        tier4_attn = {
            k: float(np.nanmean([row[k] for row in attn_enrichments])) for k in attn_enrichments[0]
        }

    tier5_3d: dict[str, Any] | None = None
    if eval_3d and all_generated:
        tier5_3d = evaluate_3d_plausibility(
            all_generated[: min(32, len(all_generated))],
            run_vina=run_vina,
            vina_bin=vina_bin,
            seed=seed,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_written: dict[str, str] = {}

    gen_props = compute_properties_batch(all_generated)
    ref_props = compute_properties_batch(ref_smiles)
    if len(gen_props.get("QED", [])) > 0 and len(ref_props.get("QED", [])) > 0:
        prop_fig = plot_property_distributions(
            ref_props=ref_props,
            gen_props_dict={cfg.name: gen_props},
            out_path=out_dir / "property_distributions.png",
        )
        figures_written["property_distributions"] = str(prop_fig)

    matched_tan = list(tier2_spec.get("pooled_matched_tanimoto", []))
    shuffled_tan = list(tier2_spec.get("pooled_shuffled_tanimoto", []))
    null_tan = list(tier2_spec.get("pooled_null_tanimoto", []))
    if matched_tan and shuffled_tan:
        spec_fig = plot_matched_vs_shuffled_similarity(
            matched_sims=matched_tan,
            shuffled_sims=shuffled_tan,
            null_sims=null_tan if null_tan else None,
            out_path=out_dir / "matched_vs_shuffled.png",
        )
        figures_written["matched_vs_shuffled"] = str(spec_fig)

    finite_pkd = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
    if int(finite_pkd.sum()) >= 2:
        pkd_fig = plot_pkd_parity(
            y_true=y_true_all[finite_pkd],
            y_pred=y_pred_all[finite_pkd],
            split_name=f"{eff_split_type}/{split}",
            out_path=out_dir / "pkd_parity.png",
        )
        figures_written["pkd_parity"] = str(pkd_fig)

    summary_payload: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "model_name": cfg.name,
        "split_type": eff_split_type,
        "split": split,
        "num_pockets_evaluated": n_eval,
        "num_samples_per_pocket": num_samples,
        "tier1_generation": tier1_gen,
        "tier2_specificity": tier2_spec,
        "tier3_affinity": tier3_affinity,
        "tier3_clip": tier3_clip,
        "tier4_attention": tier4_attn,
        "tier5_3d": tier5_3d,
        "figures": figures_written,
    }
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary_payload), indent=2) + "\n",
        encoding="utf-8",
    )

    table = Table(title=f"GenEuclidean v2 Evaluation ({eff_split_type}/{split}, n={n_eval})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Validity", f"{tier1_gen['validity']:.3f}")
    table.add_row("Uniqueness", f"{tier1_gen['uniqueness']:.3f}")
    table.add_row("Novelty", f"{tier1_gen['novelty']:.3f}")
    table.add_row("IntDiv1", f"{tier1_gen['int_div_1']:.3f}")
    table.add_row("Murcko Scaffold Div", f"{tier1_gen['murcko_scaffold_diversity']:.3f}")
    if tier3_affinity is not None:
        table.add_row(
            "pKd Pearson r / RMSE",
            f"{tier3_affinity['pearson_r']:.3f} / {tier3_affinity['rmse']:.3f}",
        )
    if tier3_clip is not None and "recall@1" in tier3_clip:
        table.add_row("DrugCLIP Recall@1", f"{tier3_clip['recall@1']:.3f}")
    table.add_row("Summary JSON", str(summary_path))
    console.print(table)


@app.command("train-llm")
def train_llm(
    config: Path = typer.Option(..., "--config", "-c", help="v3.0 LLM experiment YAML config."),
    v2_ckpt: Path | None = typer.Option(
        None, help="Override warm-start v2.0 PaiNN-SE3 checkpoint path."
    ),
    output_dir: Path | None = typer.Option(None, help="Override run output directory."),
    split_type: str | None = typer.Option(
        None, help="Override split type (casf2016|lp_pdbbind|scaffold|random)."
    ),
    phase_a_epochs: int | None = typer.Option(
        None, help="Override Phase A projector warmup epochs."
    ),
    phase_b_epochs: int | None = typer.Option(
        None, help="Override Phase B joint Perceiver+LoRA epochs."
    ),
    batch_size: int | None = typer.Option(None, help="Override batch size."),
    cache_dir: str | None = typer.Option(None, help="Override HuggingFace cache directory."),
    device: str | None = typer.Option(None, help="Override compute device (auto|mps|cuda|cpu)."),
    local_files_only: bool = typer.Option(
        False, help="Require HF weights to load strictly from local cache."
    ),
) -> None:
    """Run two-phase 3D-to-LLM training (Perceiver warmup + LoRA fine-tuning)."""
    import os

    import torch

    try:
        from geneuclidean.models.hf_backbones import load_hf_backbone
        from geneuclidean.models.llm_bridge import PocketConditionedLLM
        from geneuclidean.training.llm_trainer import (
            PocketLLMTrainer,
            build_pocket_batches_from_records,
            precompute_pocket_cache,
        )
    except ImportError as exc:
        raise typer.BadParameter(
            "Optional LLM dependencies are not installed. Install with: pip install -e '.[llm]'"
        ) from exc

    from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything, select_device

    cfg = ExperimentConfig.from_yaml(config)
    llm_cfg = cfg.llm if cfg.llm is not None else LLMConfig()

    if output_dir is not None:
        cfg.train.output_dir = str(output_dir)
    if split_type is not None:
        cfg.data.split_type = cast(
            Literal["lp_pdbbind", "casf2016", "scaffold", "random"], split_type
        )
    if batch_size is not None:
        cfg.train.batch_size = batch_size
    if device is not None:
        cfg.train.device = device
    if v2_ckpt is not None:
        llm_cfg.v2_pocket_ckpt = str(v2_ckpt)
    if cache_dir is not None:
        llm_cfg.cache_dir = cache_dir
    if phase_a_epochs is not None:
        llm_cfg.phase_a_epochs = phase_a_epochs
    if phase_b_epochs is not None:
        llm_cfg.phase_b_epochs = phase_b_epochs
    cfg.llm = llm_cfg

    seed_everything(cfg.train.seed)
    torch_device = select_device(cfg.train.device)
    os.environ["HF_HOME"] = str(Path(llm_cfg.cache_dir).resolve())

    proc_root = Path(cfg.data.processed_dir)
    cache_path = proc_root / "pdbbind_pockets.pt"
    splits_path = proc_root / "splits.json"
    if not cache_path.is_file() or not splits_path.is_file():
        raise typer.BadParameter(
            f"Processed cache not found in {proc_root}. Run `geneuclidean prepare` first."
        )

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records: dict[str, dict[str, Any]] = payload["records"]
    splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
    group = splits_data[cfg.data.split_type]
    train_ids = [pid for pid in group["train"] if pid in records]
    val_ids = [pid for pid in group["val"] if pid in records]

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
    v2_path = Path(llm_cfg.v2_pocket_ckpt)
    if v2_path.is_file():
        model.load_v2_pocket_checkpoint(v2_path)
    else:
        raise typer.BadParameter(f"v2 pocket checkpoint not found: {v2_path}")

    train_batches = build_pocket_batches_from_records(
        records, train_ids, batch_size=32, fp_dim=cfg.multitask.ligand_fp_dim
    )
    val_batches = build_pocket_batches_from_records(
        records, val_ids, batch_size=32, fp_dim=cfg.multitask.ligand_fp_dim
    )
    train_cache = precompute_pocket_cache(model, train_batches, device=torch_device)
    val_cache = precompute_pocket_cache(model, val_batches, device=torch_device)

    trainer = PocketLLMTrainer(model=model, cfg=cfg, device=torch_device)
    summary = trainer.fit(
        train_cache=train_cache, val_cache=val_cache, output_dir=cfg.train.output_dir
    )

    table = Table(title=f"v3.0 LLM Training Complete ({cfg.name})")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Backbone", llm_cfg.backbone_id)
    table.add_row(
        "Split", f"{cfg.data.split_type} (train={len(train_cache)}, val={len(val_cache)})"
    )
    table.add_row("Best epoch", str(summary["best_epoch"]))
    table.add_row(
        "Best val loss / ppl", f"{summary['best_val_lm_loss']} / {summary['best_val_ppl']}"
    )
    table.add_row("Adapter checkpoint", str(summary["best_adapter_path"]))
    console.print(table)


@app.command("sample-llm")
def sample_llm(
    adapter: Path = typer.Option(
        ..., "--adapter", help="Trained v3.0 adapter checkpoint (best_adapter.pt)."
    ),
    pdb_id: str | None = typer.Option(None, "--pdb-id", help="4-char PDB ID (cache or RCSB)."),
    pdb_file: Path | None = typer.Option(None, "--pdb-file", help="Path to local PDB/mmCIF file."),
    center: str | None = typer.Option(
        None, help="Optional binding-site center 'x,y,z' in Angstroms (for --pdb-file)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Optional v3 YAML override."),
    v2_ckpt: Path | None = typer.Option(None, help="Optional v2.0 PaiNN-SE3 checkpoint override."),
    strategy: str | None = typer.Option(
        None, help="Decoding strategy (min_p|top_p|top_k|temperature|greedy|typical)."
    ),
    num_samples: int | None = typer.Option(None, help="Number of molecules to generate."),
    max_new_tokens: int = typer.Option(96, help="Maximum new tokens per molecule."),
    temperature: float | None = typer.Option(None, help="Sampling temperature."),
    min_p: float | None = typer.Option(None, help="Min-p threshold."),
    cfg_scale: float | None = typer.Option(None, help="Classifier-Free Guidance scale."),
    latent_temp: float | None = typer.Option(None, help="CVAE prior sampling temperature."),
    device: str = typer.Option("auto", help="Compute device (auto|mps|cuda|cpu)."),
    seed: int = typer.Option(42, help="Sampling random seed."),
    rerank: bool = typer.Option(
        True, "--rerank/--no-rerank", help="Apply MPO scoring and MaxMin diverse hit selection."
    ),
    top_k_hits: int = typer.Option(16, help="Number of diverse reranked hits to display/save."),
    local_files_only: bool = typer.Option(False, help="Load HF weights strictly from local cache."),
    out_json: Path | None = typer.Option(None, help="Optional JSON path to save sampled results."),
) -> None:
    """Sample de novo ligands for a 3D pocket using a trained GenEuclidean v3.0 LLM adapter."""
    import os

    import numpy as np
    import torch

    try:
        from geneuclidean.models.hf_backbones import load_hf_backbone
        from geneuclidean.models.llm_bridge import PocketConditionedLLM
        from geneuclidean.sampling.llm_sampler import sample_llm_molecules
    except ImportError as exc:
        raise typer.BadParameter(
            "Optional LLM dependencies are not installed. Install with: pip install -e '.[llm]'"
        ) from exc

    from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything, select_device
    from geneuclidean.data.featurizer import featurize_ligand_smiles
    from geneuclidean.sampling.rerank import compute_mpo_scores, rerank_and_select

    if not adapter.is_file():
        raise typer.BadParameter(f"Adapter checkpoint not found: {adapter}")

    seed_everything(seed)
    torch_device = select_device(device)
    payload = torch.load(adapter, map_location="cpu", weights_only=False)

    if config is not None:
        cfg = ExperimentConfig.from_yaml(config)
    elif (adapter.parent / "config.yaml").is_file():
        cfg = ExperimentConfig.from_yaml(adapter.parent / "config.yaml")
    elif isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        meta = payload["metadata"]
        raw_cfg = meta.get("config") or meta.get("v2_cfg")
        cfg = (
            ExperimentConfig.model_validate(raw_cfg)
            if isinstance(raw_cfg, dict)
            else ExperimentConfig()
        )
    else:
        cfg = ExperimentConfig()

    llm_cfg = cfg.llm if cfg.llm is not None else LLMConfig()
    if v2_ckpt is not None:
        llm_cfg.v2_pocket_ckpt = str(v2_ckpt)
    os.environ["HF_HOME"] = str(Path(llm_cfg.cache_dir).resolve())

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
    v2_path = Path(llm_cfg.v2_pocket_ckpt)
    if v2_path.is_file():
        model.load_v2_pocket_checkpoint(v2_path)
    model.load_adapter_checkpoint(adapter, map_location="cpu")
    model.to(torch_device)
    model.eval()

    center_tuple = _parse_center_arg(center)
    pocket_batch, resolved_id, _pocket_pdb, _native_sdf, native_smiles = (
        _resolve_pocket_for_sampling(
            pdb_id=pdb_id,
            pdb_file=pdb_file,
            center=center_tuple,
            cfg=cfg,
        )
    )

    eff_strategy = strategy if strategy is not None else cfg.sampling.strategy
    eff_num = num_samples if num_samples is not None else cfg.sampling.num_samples
    eff_temp = temperature if temperature is not None else cfg.sampling.temperature
    eff_min_p = min_p if min_p is not None else cfg.sampling.min_p
    eff_cfg_scale = cfg_scale if cfg_scale is not None else cfg.sampling.cfg_scale
    eff_latent = latent_temp if latent_temp is not None else cfg.sampling.latent_temp

    with torch.no_grad():
        pocket_out = model.encode_pocket(
            pocket_batch["coords"].to(torch_device),
            pocket_batch["node_feats"].to(torch_device),
            pocket_batch["atom_mask"].to(torch_device),
        )
    raw_smiles = sample_llm_molecules(
        model=model,
        pocket_out=pocket_out,
        atom_mask=pocket_batch["atom_mask"].to(torch_device),
        num_samples=eff_num,
        max_new_tokens=max_new_tokens,
        strategy=eff_strategy,
        temperature=eff_temp,
        min_p=eff_min_p,
        top_p=cfg.sampling.top_p,
        top_k=cfg.sampling.top_k,
        cfg_scale=eff_cfg_scale,
        latent_temp=eff_latent,
        repetition_penalty=cfg.sampling.repetition_penalty,
        batch_size=min(32, eff_num),
        seed=seed,
    )

    pred_pkd_arr: list[float] | None = None
    clip_sim_arr: list[float] | None = None
    fps: list[np.ndarray] = []
    valid_indices: list[int] = []
    for idx, smi in enumerate(raw_smiles):
        parsed = featurize_ligand_smiles(smi, fp_size=cfg.multitask.ligand_fp_dim)
        if parsed is not None:
            valid_indices.append(idx)
            fps.append(parsed[1])
    if valid_indices:
        with torch.no_grad():
            fp_tensor = torch.from_numpy(np.stack(fps, axis=0)).to(
                device=torch_device, dtype=torch.float32
            )
            head_scores = model.score_candidates(pocket_out, fp_tensor)
        if "pkd" in head_scores:
            pred_pkd_arr = [float("nan")] * len(raw_smiles)
            for idx, val in zip(
                valid_indices, head_scores["pkd"].detach().cpu().tolist(), strict=True
            ):
                pred_pkd_arr[idx] = float(val)
        if "clip_sim" in head_scores:
            clip_sim_arr = [float("nan")] * len(raw_smiles)
            for idx, val in zip(
                valid_indices, head_scores["clip_sim"].detach().cpu().tolist(), strict=True
            ):
                clip_sim_arr[idx] = float(val)

    selected_hits = (
        rerank_and_select(
            raw_smiles,
            k=top_k_hits,
            pred_pkd=pred_pkd_arr,
            clip_sim=clip_sim_arr,
            relax_if_needed=True,
        )
        if rerank
        else compute_mpo_scores(raw_smiles, pred_pkd=pred_pkd_arr, clip_sim=clip_sim_arr)[
            :top_k_hits
        ]
    )

    table = Table(
        title=(
            f"v3.0 LLM Sampled Ligands for {resolved_id.upper()} "
            f"({llm_cfg.backbone_id}, cfg={eff_cfg_scale:.2f}, raw={len(raw_smiles)})"
        )
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Canonical SMILES", style="green")
    table.add_column("QED", justify="right")
    table.add_column("SA", justify="right")
    table.add_column("pKd", justify="right")
    table.add_column("MPO", justify="right", style="cyan")
    for rank, hit in enumerate(selected_hits, start=1):
        table.add_row(
            str(rank),
            hit.canonical_smiles or hit.smiles,
            f"{hit.qed:.3f}",
            f"{hit.sa_score:.2f}",
            f"{hit.pred_pkd:.2f}" if hit.pred_pkd is not None else "-",
            f"{hit.mpo_score:.3f}",
        )
    console.print(table)

    if out_json is not None:
        payload_out: dict[str, Any] = {
            "pdb_id": resolved_id,
            "native_smiles": native_smiles,
            "adapter": str(adapter),
            "backbone_id": llm_cfg.backbone_id,
            "raw_smiles": raw_smiles,
            "selected_hits": [
                {
                    "smiles": h.smiles,
                    "canonical_smiles": h.canonical_smiles,
                    "valid": h.valid,
                    "qed": h.qed,
                    "sa_score": h.sa_score,
                    "sa_norm": h.sa_norm,
                    "lipinski_pass": h.lipinski_pass,
                    "pains_free": h.pains_free,
                    "pred_pkd": h.pred_pkd,
                    "clip_sim": h.clip_sim,
                    "mpo_score": h.mpo_score,
                }
                for h in selected_hits
            ],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload_out), indent=2) + "\n", encoding="utf-8")
        console.print(f"[bold green]Saved v3 LLM sampling output to {out_json}[/bold green]")


@app.command("fetch-pdb")
def fetch_pdb(
    pdb_id: str = typer.Option(..., "--pdb-id", help="4-character RCSB PDB identifier."),
    out_dir: Path = typer.Option(Path("data/raw/rcsb"), help="Destination directory."),
    ligand_resname: str | None = typer.Option(
        None,
        help="Optional 3-letter CCD ligand code (auto-selects largest non-solvent if omitted).",
    ),
    pocket_cutoff: float = typer.Option(10.0, help="Pocket extraction radius in Angstroms."),
) -> None:
    """Fetch any novel protein-ligand complex live from RCSB PDB and extract its 3D pocket."""
    from geneuclidean.data.download import fetch_rcsb_complex

    info_dict = fetch_rcsb_complex(
        pdb_id=pdb_id,
        out_dir=out_dir,
        ligand_resname=ligand_resname,
        pocket_cutoff=pocket_cutoff,
    )
    table = Table(title=f"RCSB PDB Fetch Complete ({str(info_dict['pdb_id']).upper()})")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("mmCIF file", str(info_dict["cif_path"]))
    table.add_row("Pocket PDB", str(info_dict["pocket_pdb"]))
    table.add_row("Pocket atoms", str(info_dict["pocket_atoms"]))
    table.add_row("Detected ligand", str(info_dict.get("ligand_resname") or "None"))
    table.add_row("Ligand SMILES", str(info_dict.get("ligand_smiles") or "None"))
    table.add_row("Ligand center", str(info_dict.get("ligand_center") or "None"))
    console.print(table)


@app.command()
def info(
    processed_dir: Path = typer.Option(Path("data/processed"), help="Processed cache directory."),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw dataset directory."),
    metadata_dir: Path = typer.Option(Path("data/metadata"), help="Metadata CSV directory."),
    vina_bin: Path = typer.Option(Path("bin/vina"), help="AutoDock Vina binary path."),
    hf_cache_dir: Path = typer.Option(
        Path(".hf_cache.nosync"), help="HuggingFace local cache directory for v3.0 LLM weights."
    ),
) -> None:
    """Display GenEuclidean version, active device, model catalog, and dataset cache status."""
    import importlib.metadata

    import e3nn
    import rdkit
    import torch

    from geneuclidean.config import select_device

    device = select_device("auto")
    try:
        tf_ver = importlib.metadata.version("transformers")
        peft_ver = importlib.metadata.version("peft")
        llm_status = f"READY (transformers {tf_ver} / peft {peft_ver})"
    except importlib.metadata.PackageNotFoundError:
        llm_status = "NOT INSTALLED (pip install -e '.[llm]')"

    env_table = Table(title=f"GenEuclidean v{geneuclidean.__version__} Environment & Catalog")
    env_table.add_column("Property", style="cyan")
    env_table.add_column("Value", style="green")
    env_table.add_row("Version", geneuclidean.__version__)
    env_table.add_row("Python / Platform", f"{platform.python_version()} ({platform.machine()})")
    env_table.add_row(
        "PyTorch / Default Dtype", f"{torch.__version__} ({torch.get_default_dtype()})"
    )
    env_table.add_row("Auto-selected Device", str(device))
    env_table.add_row("RDKit / e3nn", f"{rdkit.__version__} / {e3nn.__version__}")
    env_table.add_row("LLM Extra ([llm])", llm_status)
    env_table.add_row("3D Encoders", "painn (SE3), egnn, e3nn (v0.6), schnet")
    env_table.add_row(
        "Decoders / v3 LLMs",
        "transformer, lstm | v3: GP-MoLFormer-Uniq, gpt2_zinc_87m, Qwen2.5-0.5B, SmolLM2-360M",
    )
    env_table.add_row(
        "Sampling Strategies",
        "min_p, top_p, top_k, temperature, greedy, typical, diverse_beam (+CFG, CVAE, MPO)",
    )
    console.print(env_table)

    cache_files = [
        ("Processed pockets", processed_dir / "pdbbind_pockets.pt"),
        ("Leak-proof splits", processed_dir / "splits.json"),
        ("SMILES vocabulary", processed_dir / "vocab_smiles.json"),
        ("SELFIES vocabulary", processed_dir / "vocab_selfies.json"),
        ("ZINC-250k corpus", processed_dir / "zinc250k_smiles.txt"),
        ("LP-PDBBind CSV", raw_dir / "LP_PDBBind.csv"),
        ("PDBbind v2019 labels", metadata_dir / "pdbbind_v2019_refined_labels.csv"),
        ("CASF-2016 labels", metadata_dir / "casf_2016_core_labels.csv"),
        ("AutoDock Vina binary", vina_bin),
    ]
    data_table = Table(title="Dataset & Tooling Status")
    data_table.add_column("Asset", style="cyan")
    data_table.add_column("Path", style="white")
    data_table.add_column("Status", style="bold")
    for label, path in cache_files:
        if path.is_file():
            size_mb = path.stat().st_size / (1024.0 * 1024.0)
            status = f"[green]READY ({size_mb:.1f} MB)[/green]"
        else:
            status = "[yellow]MISSING[/yellow]"
        data_table.add_row(label, str(path), status)

    hub_dir = hf_cache_dir / "hub" if (hf_cache_dir / "hub").is_dir() else hf_cache_dir
    if hf_cache_dir.is_dir():
        cached_models = list(hub_dir.glob("models--*"))
        hf_status = f"[green]READY ({len(cached_models)} cached models)[/green]"
    else:
        hf_status = "[yellow]MISSING[/yellow]"
    data_table.add_row("HF LLM cache", str(hf_cache_dir), hf_status)
    console.print(data_table)


if __name__ == "__main__":
    app()
