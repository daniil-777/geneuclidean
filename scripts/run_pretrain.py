"""Stage-1 ZINC-250k chemical prior pretraining & unconditional evaluation driver.

Invokes ``geneuclidean.training.pretrain.pretrain_chemical_prior`` on Apple Silicon MPS
(strictly ``float32``) using length-bucketed batching and parallel multi-view Randomized
SMILES tokenization (or cached SELFIES) to keep MPS saturated without CPU bottlenecks.

Usage examples::

    # 1-epoch throughput & convergence benchmark
    .venv/bin/python scripts/run_pretrain.py \\
        --config configs/pretrain_zinc.yaml \\
        --output-dir runs/_bench_1ep --epochs 1 --eval-samples 200

    # Run #1: Flagship ZINC-250k Randomized SMILES chemical prior
    .venv/bin/python scripts/run_pretrain.py \\
        --config configs/pretrain_zinc.yaml \\
        --output-dir runs/prior_smiles --epochs 6 --batch-size 128 --lr 0.0008

    # Run #2: ZINC-250k SELFIES chemical prior (caches data/processed/zinc250k_selfies.txt)
    .venv/bin/python scripts/run_pretrain.py \\
        --config configs/pretrain_zinc_selfies.yaml \\
        --output-dir runs/prior_selfies --epochs 5 --batch-size 128 --lr 0.0008
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import selfies as sf
import torch
from rdkit import Chem, RDLogger
from torch import Tensor, nn

from geneuclidean.config import ExperimentConfig, seed_everything, select_device
from geneuclidean.data.dataset import load_zinc_splits
from geneuclidean.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, MolecularTokenizer
from geneuclidean.evaluation.metrics import (
    compute_frechet_fingerprint_distance,
    compute_internal_diversity,
    compute_molecule_properties,
    compute_snn,
    get_morgan_bit_fp,
    get_murcko_scaffold,
    parse_mol,
)
from geneuclidean.models.decoders.transformer import PocketTransformerDecoder
from geneuclidean.sampling.decoder_strategies import select_next_tokens
from geneuclidean.training.pretrain import evaluate_chemical_prior, pretrain_chemical_prior

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Multiprocessing Workers (top-level for macOS spawn pickling)
# ---------------------------------------------------------------------------


def _tokenize_smiles_chunk(
    args: tuple[list[str], dict[str, int], int, bool, int],
) -> list[bytes]:
    """Tokenize a chunk of SMILES (canonical or fast RDKit randomized) to compact uint8 bytes."""
    smiles_chunk, token_to_id, max_len, randomize, worker_seed = args
    from geneuclidean.data.tokenizer import SCHWALLER_REGEX

    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    rng = random.Random(worker_seed)
    max_inner = max_len - 2
    out: list[bytes] = []

    for smi in smiles_chunk:
        work = smi
        if randomize:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.GetNumAtoms() > 1:
                # Fast deterministic-seeded random traversal without redundant re-parse
                n_atoms = mol.GetNumAtoms()
                order = list(range(n_atoms))
                rng.shuffle(order)
                root_idx = rng.randrange(n_atoms)
                try:
                    renum = Chem.RenumberAtoms(mol, order)
                    rand_smi = Chem.MolToSmiles(
                        renum,
                        canonical=False,
                        isomericSmiles=True,
                        rootedAtAtom=root_idx,
                    )
                    if rand_smi:
                        work = rand_smi
                except Exception:
                    work = smi
        toks = SCHWALLER_REGEX.findall(work)
        if len(toks) > max_inner:
            toks = toks[:max_inner]
        ids = [BOS_ID]
        ids.extend(token_to_id.get(t, UNK_ID) for t in toks)
        ids.append(EOS_ID)
        out.append(bytes(ids))
    return out


def _smiles_to_selfies_chunk(smiles_chunk: list[str]) -> list[str]:
    """Convert a chunk of SMILES strings into canonical SELFIES strings."""
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    results: list[str] = []
    for smi in smiles_chunk:
        s = smi.strip()
        encoded = ""
        try:
            res = sf.encoder(s)
            if isinstance(res, str) and res:
                encoded = res
        except Exception:
            encoded = ""
        if not encoded:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                try:
                    res2 = sf.encoder(can)
                    if isinstance(res2, str) and res2:
                        encoded = res2
                except Exception:
                    encoded = ""
        results.append(encoded)
    return results


def _tokenize_selfies_chunk(
    args: tuple[list[str], dict[str, int], int],
) -> list[bytes]:
    """Tokenize pre-cached SELFIES strings into compact uint8 byte sequences."""
    selfies_chunk, token_to_id, max_len = args
    max_inner = max_len - 2
    out: list[bytes] = []
    for s in selfies_chunk:
        if not s:
            out.append(bytes([BOS_ID, EOS_ID]))
            continue
        toks = list(sf.split_selfies(s))
        if len(toks) > max_inner:
            toks = toks[:max_inner]
        ids = [BOS_ID]
        ids.extend(token_to_id.get(t, UNK_ID) for t in toks)
        ids.append(EOS_ID)
        out.append(bytes(ids))
    return out


def _canonicalize_chunk(smiles_chunk: list[str]) -> list[str]:
    """Canonicalize a chunk of SMILES for exact novelty checking."""
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    out: list[str] = []
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out.append(str(Chem.MolToSmiles(mol, canonical=True)))
    return out


# ---------------------------------------------------------------------------
# Fast Pre-tokenized Multi-View & Length-Bucketed DataLoader
# ---------------------------------------------------------------------------


def _split_chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    if not items:
        return []
    n = max(1, min(n_chunks, len(items)))
    size = math.ceil(len(items) / n)
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_or_load_selfies_cache(
    smiles_txt: Path,
    selfies_txt: Path,
    *,
    num_workers: int = 8,
) -> list[str]:
    """Encode ZINC-250k SMILES to SELFIES once and cache 1-to-1 aligned to ``selfies_txt``."""
    smiles_lines = [
        line.strip() for line in smiles_txt.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if selfies_txt.exists():
        cached = selfies_txt.read_text(encoding="utf-8").splitlines()
        if len(cached) == len(smiles_lines):
            print(
                f"[cache] Loaded cached SELFIES ({len(cached):,} lines) from {selfies_txt}",
                flush=True,
            )
            return cached

    t0 = time.perf_counter()
    chunks = _split_chunks(smiles_lines, num_workers * 4)
    all_selfies: list[str] = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for res in pool.map(_smiles_to_selfies_chunk, chunks):
            all_selfies.extend(res)

    n_empty = sum(1 for s in all_selfies if not s)
    selfies_txt.parent.mkdir(parents=True, exist_ok=True)
    selfies_txt.write_text("\n".join(all_selfies) + "\n", encoding="utf-8")
    dt = time.perf_counter() - t0
    size_mb = selfies_txt.stat().st_size / (1024 * 1024)
    print(
        f"[cache] Encoded {len(all_selfies):,} SELFIES ({n_empty} empty) -> {selfies_txt} "
        f"({size_mb:.2f} MB) in {dt:.1f}s",
        flush=True,
    )
    return all_selfies


def pretokenize_smiles_views(
    smiles_list: list[str],
    tokenizer: MolecularTokenizer,
    *,
    max_seq_len: int,
    randomize: bool,
    num_views: int,
    seed: int,
    num_workers: int = 8,
) -> list[list[bytes]]:
    """Pre-tokenize ``num_views`` independent views (canonical or randomized SMILES) in parallel."""
    if tokenizer.vocab_size > 255:
        raise ValueError(f"uint8 packing requires vocab_size <= 255, got {tokenizer.vocab_size}")
    tok_map = tokenizer.token_to_id
    eff_views = num_views if randomize else 1
    chunks = _split_chunks(smiles_list, num_workers * 4)
    views: list[list[bytes]] = []

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for v in range(eff_views):
            tasks = [
                (chunk, tok_map, max_seq_len, randomize, seed + v * 10_000 + c_idx)
                for c_idx, chunk in enumerate(chunks)
            ]
            view_seqs: list[bytes] = []
            for res in pool.map(_tokenize_smiles_chunk, tasks):
                view_seqs.extend(res)
            views.append(view_seqs)
    dt = time.perf_counter() - t0
    ram_mb = sum(sum(len(b) for b in vw) for vw in views) / (1024 * 1024)
    mode = "randomized" if randomize else "canonical"
    print(
        f"[data] Pre-tokenized {len(views)} {mode} SMILES view(s) x {len(smiles_list):,} mols "
        f"({ram_mb:.1f} MB RAM) in {dt:.1f}s",
        flush=True,
    )
    return views


def pretokenize_selfies_view(
    selfies_list: list[str],
    tokenizer: MolecularTokenizer,
    *,
    max_seq_len: int,
    num_workers: int = 8,
) -> list[list[bytes]]:
    """Pre-tokenize cached SELFIES strings in parallel into compact uint8 byte sequences."""
    if tokenizer.vocab_size > 255:
        raise ValueError(f"uint8 packing requires vocab_size <= 255, got {tokenizer.vocab_size}")
    tok_map = tokenizer.token_to_id
    chunks = _split_chunks(selfies_list, num_workers * 4)
    t0 = time.perf_counter()
    seqs: list[bytes] = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        tasks = [(chunk, tok_map, max_seq_len) for chunk in chunks]
        for res in pool.map(_tokenize_selfies_chunk, tasks):
            seqs.extend(res)
    dt = time.perf_counter() - t0
    ram_mb = sum(len(b) for b in seqs) / (1024 * 1024)
    print(
        f"[data] Pre-tokenized {len(seqs):,} cached SELFIES ({ram_mb:.1f} MB RAM) in {dt:.1f}s",
        flush=True,
    )
    return [seqs]


class LengthBucketMultiViewLoader:
    """Zero-stall length-bucketed multi-view loader for ``pretrain_chemical_prior``.

    On each epoch iteration:
      1. Selects the next tokenized view (cycling cleanly across ``views``).
      2. Shuffles all molecule indices with an epoch-seeded RNG (when ``shuffle=True``).
      3. Sorts within megabatches of ``batch_size * megabatch_mult`` by sequence length to
         minimize wasted padding compute (~1.7x MPS speedup with <3% padding).
      4. Shuffles batch order across the full epoch so SGD sees unbiased length statistics.
    """

    def __init__(
        self,
        views: list[list[bytes]],
        *,
        batch_size: int,
        shuffle: bool = True,
        megabatch_mult: int = 32,
        seed: int = 42,
    ) -> None:
        if not views or not views[0]:
            raise ValueError("views must be non-empty")
        self.views = views
        self.n_samples = len(views[0])
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.megabatch_mult = int(megabatch_mult)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        view_idx = self._epoch % len(self.views)
        seqs = self.views[view_idx]
        rng = np.random.default_rng(self.seed + self._epoch * 7919)
        self._epoch += 1

        indices = np.arange(self.n_samples, dtype=np.int64)
        lengths = np.fromiter((len(b) for b in seqs), dtype=np.int32, count=self.n_samples)

        if self.shuffle:
            rng.shuffle(indices)
            mega_size = self.batch_size * self.megabatch_mult
            batches: list[np.ndarray] = []
            for start in range(0, self.n_samples, mega_size):
                chunk = indices[start : start + mega_size]
                order = np.argsort(lengths[chunk], kind="mergesort")
                sorted_chunk = chunk[order]
                for b_start in range(0, len(sorted_chunk), self.batch_size):
                    batches.append(sorted_chunk[b_start : b_start + self.batch_size])
            batch_perm = rng.permutation(len(batches))
            ordered_batches = [batches[int(i)] for i in batch_perm]
        else:
            order = np.argsort(lengths, kind="mergesort")
            sorted_idx = indices[order]
            ordered_batches = [
                sorted_idx[b_start : b_start + self.batch_size]
                for b_start in range(0, self.n_samples, self.batch_size)
            ]

        for batch_idx in ordered_batches:
            bsz = len(batch_idx)
            batch_seqs = [seqs[int(i)] for i in batch_idx]
            max_seq = max(len(b) - 1 for b in batch_seqs)
            inp_np = np.full((bsz, max_seq), PAD_ID, dtype=np.int64)
            tgt_np = np.full((bsz, max_seq), PAD_ID, dtype=np.int64)
            for row_i, raw in enumerate(batch_seqs):
                arr = np.frombuffer(raw, dtype=np.uint8)
                seq_len = len(arr) - 1
                inp_np[row_i, :seq_len] = arr[:-1]
                tgt_np[row_i, :seq_len] = arr[1:]
            yield {
                "input_ids": torch.from_numpy(inp_np),
                "target_ids": torch.from_numpy(tgt_np),
            }


# ---------------------------------------------------------------------------
# Unconditional Min-p Sampling & Tier-1 Chemical Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_unconditional_prior(
    decoder: nn.Module,
    tokenizer: MolecularTokenizer,
    *,
    num_samples: int = 1000,
    batch_size: int = 256,
    max_len: int = 120,
    strategy: str = "min_p",
    temperature: float = 1.0,
    min_p: float = 0.08,
    device: torch.device,
    seed: int = 42,
) -> list[str]:
    """Generate unconstrained unconditional molecules from a Stage-1 prior decoder."""
    decoder.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_smiles: list[str] = []

    remaining = num_samples
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        generated = torch.full((cur_bs, 1), fill_value=BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(cur_bs, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, _ = decoder(generated, memory=None, memory_mask=None, return_attn=False)
            step_logits = logits[:, -1, :].clone()
            step_logits[:, PAD_ID] = float("-inf")
            step_logits[:, BOS_ID] = float("-inf")
            step_logits[:, UNK_ID] = float("-inf")

            next_tokens = select_next_tokens(
                step_logits,
                strategy=strategy,
                temperature=temperature,
                min_p=min_p,
                generator=gen,
            )
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, PAD_ID),
                next_tokens,
            )
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | (next_tokens == EOS_ID)
            if bool(finished.all().item()):
                break

        for row in generated.detach().cpu().tolist():
            all_smiles.append(tokenizer.decode(row, to_smiles=True))
        remaining -= cur_bs

    return all_smiles


def evaluate_unconditional_samples(
    generated_smiles: list[str],
    train_smiles: list[str],
    val_smiles: list[str],
    *,
    seed: int = 42,
    num_workers: int = 8,
) -> dict[str, float]:
    """Evaluate 1,000 unconstrained samples against full ZINC train set & ZINC val reference."""
    t0 = time.perf_counter()
    n_gen = len(generated_smiles)
    valid_mols = []
    valid_canon: list[str] = []
    for smi in generated_smiles:
        mol = parse_mol(smi)
        if mol is not None:
            valid_mols.append(mol)
            valid_canon.append(str(Chem.MolToSmiles(mol, canonical=True)))

    n_valid = len(valid_mols)
    validity = n_valid / n_gen if n_gen > 0 else 0.0

    unique_map: dict[str, Chem.Mol] = {}
    for can, mol in zip(valid_canon, valid_mols, strict=True):
        if can not in unique_map:
            unique_map[can] = mol
    n_unique = len(unique_map)
    uniqueness = n_unique / n_valid if n_valid > 0 else 0.0

    # Canonicalize ALL 236,982 ZINC train molecules in parallel for 100% exact novelty
    chunks = _split_chunks(train_smiles, num_workers * 4)
    train_canon_set: set[str] = set()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for res in pool.map(_canonicalize_chunk, chunks):
            train_canon_set.update(res)

    n_novel = sum(1 for can in unique_map if can not in train_canon_set)
    novelty = n_novel / n_unique if n_unique > 0 else 0.0

    valid_fps = [get_morgan_bit_fp(m) for m in valid_mols]
    int_div_1 = compute_internal_diversity(valid_fps, p=1, seed=seed)
    int_div_2 = compute_internal_diversity(valid_fps, p=2, seed=seed)

    gen_scaffolds = {get_murcko_scaffold(m) for m in unique_map.values()}
    scaffold_div = len(gen_scaffolds) / n_unique if n_unique > 0 else 0.0

    # Compare SNN & FFD against 2,000 held-out ZINC validation molecules
    ref_mols = [m for s in val_smiles[:2000] if (m := parse_mol(s)) is not None]
    ref_fps = [get_morgan_bit_fp(m) for m in ref_mols]
    snn = compute_snn(valid_fps, ref_fps)
    ffd = compute_frechet_fingerprint_distance(valid_fps, ref_fps, n_components=256)

    props = [compute_molecule_properties(m) for m in valid_mols]
    if props:
        qed_mean = float(np.mean([p["QED"] for p in props]))
        qed_std = float(np.std([p["QED"] for p in props]))
        sa_mean = float(np.mean([p["SA"] for p in props]))
        sa_std = float(np.std([p["SA"] for p in props]))
        mw_mean = float(np.mean([p["MW"] for p in props]))
        logp_mean = float(np.mean([p["LogP"] for p in props]))
        tpsa_mean = float(np.mean([p["TPSA"] for p in props]))
        lipinski = float(np.mean([p["lipinski_pass"] for p in props]))
        pains_free = float(np.mean([p["pains_free"] for p in props]))
    else:
        qed_mean = qed_std = sa_mean = sa_std = mw_mean = logp_mean = tpsa_mean = float("nan")
        lipinski = pains_free = 0.0

    dt = time.perf_counter() - t0
    print(
        f"[eval] Evaluated {n_gen} samples vs {len(train_canon_set):,} train mols in {dt:.1f}s",
        flush=True,
    )
    return {
        "n_generated": float(n_gen),
        "n_valid": float(n_valid),
        "n_unique": float(n_unique),
        "n_novel": float(n_novel),
        "validity": float(validity),
        "uniqueness": float(uniqueness),
        "novelty": float(novelty),
        "int_div_1": float(int_div_1),
        "int_div_2": float(int_div_2),
        "murcko_scaffold_diversity": float(scaffold_div),
        "snn_val2k": float(snn),
        "ffd_val2k": float(ffd),
        "QED_mean": qed_mean,
        "QED_std": qed_std,
        "SA_mean": sa_mean,
        "SA_std": sa_std,
        "MW_mean": mw_mean,
        "LogP_mean": logp_mean,
        "TPSA_mean": tpsa_mean,
        "lipinski_pass_rate": lipinski,
        "pains_free_rate": pains_free,
    }


def verify_stage2_transfer(
    ckpt_path: Path, cfg: ExperimentConfig, vocab_size: int
) -> dict[str, Any]:
    """Verify ``best_prior.pt`` transfers cleanly into a Stage-2 cross-attention decoder."""
    stage2_dec_cfg = cfg.decoder.model_copy(
        update={"vocab_size": vocab_size, "use_cross_attn": True, "gated_cross_attn": True}
    )
    stage2_dec = PocketTransformerDecoder(stage2_dec_cfg)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = stage2_dec.load_pretrained_backbone(payload)
    non_cross_missing = [k for k in missing if "cross_" not in k]
    return {
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "non_cross_missing": non_cross_missing,
        "stage2_transfer_ok": len(unexpected) == 0 and len(non_cross_missing) == 0,
    }


# ---------------------------------------------------------------------------
# Main CLI Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 ZINC-250k Chemical Prior Pretraining")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--num-views", type=int, default=None, help="Number of randomized views.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="CPU worker processes for tokenization / canonicalization.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Number of unconditional samples to generate & evaluate after training.",
    )
    parser.add_argument("--sample-temp", type=float, default=1.0, help="Min-p temperature.")
    parser.add_argument("--sample-min-p", type=float, default=0.08, help="Min-p threshold.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    cfg = ExperimentConfig.from_yaml(args.config)
    train_updates: dict[str, Any] = {}
    if args.output_dir is not None:
        train_updates["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        train_updates["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_updates["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        train_updates["lr"] = float(args.lr)
        train_updates["decoder_lr"] = float(args.lr)
    if args.seed is not None:
        train_updates["seed"] = int(args.seed)
    if train_updates:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update=train_updates)})

    seed_everything(cfg.train.seed)
    device = select_device(cfg.train.device)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = Path(cfg.data.processed_dir)

    rep = cfg.data.representation
    vocab_file = proc_dir / ("vocab_selfies.json" if rep == "selfies" else "vocab_smiles.json")
    tokenizer = MolecularTokenizer.load(vocab_file)
    cfg = cfg.model_copy(
        update={"decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})}
    )

    print(
        f"=== Stage-1 Chemical Prior Pretraining ({cfg.name}) ===\n"
        f"  representation : {rep} (vocab_size={tokenizer.vocab_size})\n"
        f"  device         : {device} (default_dtype={torch.get_default_dtype()})\n"
        f"  epochs / bs / lr: {cfg.train.epochs} / {cfg.train.batch_size} / {cfg.train.lr}\n"
        f"  output_dir     : {out_dir}",
        flush=True,
    )

    # Load deterministic 95/5 ZINC-250k SMILES split
    zinc_smiles_txt = proc_dir / "zinc250k_smiles.txt"
    train_smiles, val_smiles = load_zinc_splits(
        zinc_smiles_txt, val_fraction=0.05, seed=cfg.train.seed
    )
    print(
        f"[data] ZINC-250k split: train={len(train_smiles):,}, val={len(val_smiles):,}", flush=True
    )

    # Prepare tokenized multi-view loaders
    if rep == "selfies":
        selfies_txt = proc_dir / "zinc250k_selfies.txt"
        build_or_load_selfies_cache(zinc_smiles_txt, selfies_txt, num_workers=args.workers)
        train_sf, val_sf = load_zinc_splits(selfies_txt, val_fraction=0.05, seed=cfg.train.seed)
        train_views = pretokenize_selfies_view(
            train_sf, tokenizer, max_seq_len=cfg.data.max_seq_len, num_workers=args.workers
        )
        val_views = pretokenize_selfies_view(
            val_sf, tokenizer, max_seq_len=cfg.data.max_seq_len, num_workers=args.workers
        )
    else:
        num_views = args.num_views if args.num_views is not None else cfg.train.epochs
        train_views = pretokenize_smiles_views(
            train_smiles,
            tokenizer,
            max_seq_len=cfg.data.max_seq_len,
            randomize=cfg.data.randomize_smiles,
            num_views=num_views,
            seed=cfg.train.seed,
            num_workers=args.workers,
        )
        val_views = pretokenize_smiles_views(
            val_smiles,
            tokenizer,
            max_seq_len=cfg.data.max_seq_len,
            randomize=False,
            num_views=1,
            seed=cfg.train.seed,
            num_workers=args.workers,
        )

    train_loader = LengthBucketMultiViewLoader(
        train_views,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        megabatch_mult=32,
        seed=cfg.train.seed,
    )
    val_loader = LengthBucketMultiViewLoader(
        val_views,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        seed=cfg.train.seed,
    )

    # Persist resolved config for full reproducibility
    cfg.to_yaml(out_dir / "config.yaml")

    # Instantiate decoder & log param count
    decoder = PocketTransformerDecoder(cfg.decoder)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(
        f"[model] PocketTransformerDecoder: {n_params:,} params "
        f"(L={cfg.decoder.n_layers}, d={cfg.decoder.d_model}, H={cfg.decoder.n_heads}, "
        f"d_ff={cfg.decoder.d_ff}, V={cfg.decoder.vocab_size})",
        flush=True,
    )

    # Run Stage-1 pretraining via geneuclidean.training.pretrain.pretrain_chemical_prior
    t_train = time.perf_counter()
    summary = pretrain_chemical_prior(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        save_dir=out_dir,
        model=decoder,
    )
    train_elapsed = time.perf_counter() - t_train
    if device.type == "mps":
        torch.mps.synchronize()

    for rec in summary["history"]:
        star = "*" if rec["is_best"] else " "
        print(
            f"  Epoch {rec['epoch']:2d}/{cfg.train.epochs} {star} | "
            f"train_loss={rec['train_loss']:.4f} | val_loss={rec['val_loss']:.4f} | "
            f"val_ppl={rec['val_ppl']:.4f} | lr={rec['lr']:.2e} | "
            f"gnorm={rec['grad_norm']:.2f} | {rec['elapsed_sec']:.1f}s",
            flush=True,
        )

    best_ckpt = Path(summary["best_ckpt_path"])
    ckpt_mb = best_ckpt.stat().st_size / (1024 * 1024)
    transfer_check = verify_stage2_transfer(best_ckpt, cfg, tokenizer.vocab_size)
    print(
        f"[ckpt] Best epoch={summary['best_epoch']} | val_loss={summary['best_val_loss']:.4f} | "
        f"val_ppl={summary['best_val_ppl']:.4f} | {best_ckpt} ({ckpt_mb:.2f} MB) | "
        f"stage2_transfer_ok={transfer_check['stage2_transfer_ok']}",
        flush=True,
    )

    eval_metrics: dict[str, float] = {}
    sample_elapsed = 0.0
    if not args.skip_eval and args.eval_samples > 0:
        # Reload best weights cleanly
        best_payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        decoder.load_state_dict(best_payload["model_state_dict"])
        decoder = decoder.to(device)
        decoder.eval()

        # Re-verify val_loss/val_ppl from reloaded checkpoint
        recheck = evaluate_chemical_prior(decoder, val_loader, device=device, pad_id=PAD_ID)
        print(
            f"[recheck] Reloaded best_prior.pt -> val_loss={recheck['val_loss']:.4f}, "
            f"val_ppl={recheck['val_ppl']:.4f}",
            flush=True,
        )

        t_samp = time.perf_counter()
        generated_smiles = sample_unconditional_prior(
            decoder,
            tokenizer,
            num_samples=args.eval_samples,
            batch_size=min(256, args.eval_samples),
            max_len=cfg.sampling.max_len,
            strategy="min_p",
            temperature=args.sample_temp,
            min_p=args.sample_min_p,
            device=device,
            seed=cfg.train.seed + 100,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        sample_elapsed = time.perf_counter() - t_samp
        print(
            f"[sample] Generated {len(generated_smiles)} unconditional samples "
            f"(min_p={args.sample_min_p}, temp={args.sample_temp}) in {sample_elapsed:.1f}s",
            flush=True,
        )

        # Save generated samples for reproducibility / inspection
        (out_dir / "unconditional_samples.txt").write_text(
            "\n".join(generated_smiles) + "\n", encoding="utf-8"
        )

        eval_metrics = evaluate_unconditional_samples(
            generated_smiles,
            train_smiles=train_smiles,
            val_smiles=val_smiles,
            seed=cfg.train.seed,
            num_workers=args.workers,
        )

    total_elapsed = time.perf_counter() - t_start
    report = {
        "name": cfg.name,
        "representation": rep,
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "lr": cfg.train.lr,
        "seed": cfg.train.seed,
        "device": str(device),
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"],
        "best_val_ppl": summary["best_val_ppl"],
        "best_ckpt_path": str(best_ckpt),
        "best_ckpt_mb": round(ckpt_mb, 3),
        "stage2_transfer": transfer_check,
        "train_elapsed_sec": round(train_elapsed, 2),
        "sample_elapsed_sec": round(sample_elapsed, 2),
        "total_elapsed_sec": round(total_elapsed, 2),
        "sampling_params": {
            "strategy": "min_p",
            "temperature": args.sample_temp,
            "min_p": args.sample_min_p,
            "num_samples": args.eval_samples,
        },
        "eval_metrics": eval_metrics,
    }
    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
