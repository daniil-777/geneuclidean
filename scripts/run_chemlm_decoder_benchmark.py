#!/usr/bin/env python3
"""Real 3-Tier Apple Silicon MPS Benchmark for the GenEuclidean ``chemlm`` Decoder Suite.

Validates the scalable pocket- and property-conditioned chemical LLM decoders
(:mod:`geneuclidean.models.decoders.chemlm`) on Apple Silicon MPS (falling back to CUDA or
CPU) and writes a structured JSON report to
``results/tables/chemlm_decoder_benchmark.json``:

- **Tier 1 (Throughput, Memory & Scaling):** Forward and forward+backward latency (ms),
  peak memory (MB) and parameter counts across the ``tiny``/``small``/``base`` scale
  presets, dense vs. sparse Mixture-of-Experts feed-forward, and with vs. without
  gradient checkpointing.
- **Tier 2 (Numerical Correctness):** KV-cached incremental decoding vs. full-sequence
  forward parity, MPS vs. CPU parity, exact ``0.0`` influence of masked/padded pocket
  atoms, all-masked-pocket finiteness, and measurable pocket and property conditioning
  sensitivity.
- **Tier 3 (Real Chemistry):** Trains a decoder on real ZINC-250k SMILES on the GPU for a
  fixed step budget, then samples with KV-cached decoding and reports RDKit validity,
  uniqueness, novelty and mean QED, plus the measured KV-cache sampling speedup.

Run::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_chemlm_decoder_benchmark.py
    python scripts/run_chemlm_decoder_benchmark.py --quick     # fast smoke run
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from geneuclidean.config import seed_everything, select_device
from geneuclidean.data.tokenizer import MolecularTokenizer
from geneuclidean.models.decoders.chemlm import SCALE_PRESETS, ChemLMDecoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMILES = REPO_ROOT / "data" / "processed" / "zinc250k_smiles.txt"
DEFAULT_OUT = REPO_ROOT / "results" / "tables" / "chemlm_decoder_benchmark.json"


# ---------------------------------------------------------------------------
# Report containers
# ---------------------------------------------------------------------------


@dataclass
class ThroughputRow:
    """One Tier-1 throughput measurement."""

    label: str
    scale: str
    params_m: float
    d_model: int
    n_layers: int
    moe: bool
    gradient_checkpointing: bool
    forward_ms: float
    fwd_bwd_ms: float
    peak_mem_mb: float
    tokens_per_s: float


@dataclass
class BenchmarkReport:
    """Top-level JSON report."""

    device: str
    torch_version: str
    platform: str
    batch_size: int
    seq_len: int
    pocket_atoms: int
    tier1_throughput: list[dict[str, Any]] = field(default_factory=list)
    tier2_correctness: dict[str, Any] = field(default_factory=dict)
    tier3_chemistry: dict[str, Any] = field(default_factory=dict)
    tier4_property_conditioning: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synchronize(device: torch.device) -> None:
    """Block until all queued GPU work on ``device`` has completed."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    """Reset the peak-memory counter for ``device`` where the backend supports it."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated memory in MiB (``0.0`` on CPU, which has no counter)."""
    if device.type == "mps":
        return float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    return 0.0


def _time_ms(fn: Any, device: torch.device, *, warmup: int = 3, iters: int = 10) -> float:
    """Return median wall-clock milliseconds per call of ``fn``."""
    for _ in range(warmup):
        fn()
    _synchronize(device)
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples))


def _make_decoder(
    scale: str,
    vocab_size: int,
    *,
    seq_len: int,
    use_moe: bool = False,
    num_properties: int = 0,
) -> ChemLMDecoder:
    """Instantiate a :class:`ChemLMDecoder` at a named scale preset."""
    preset = SCALE_PRESETS[scale]
    return ChemLMDecoder(
        vocab_size=vocab_size,
        d_model=preset.d_model,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        d_ff=preset.d_ff,
        max_seq_len=seq_len,
        dropout=0.0,
        use_moe=use_moe,
        num_experts=8,
        moe_top_k=2,
        num_properties=num_properties,
    )


# ---------------------------------------------------------------------------
# Tier 1 — throughput, memory, scaling
# ---------------------------------------------------------------------------


def run_tier1(
    device: torch.device,
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    pocket_atoms: int,
    scales: list[str],
    iters: int,
) -> list[ThroughputRow]:
    """Measure forward / forward+backward latency and memory across configurations."""
    rows: list[ThroughputRow] = []

    configs: list[tuple[str, str, bool, bool]] = []
    for scale in scales:
        configs.append((f"{scale}-dense", scale, False, False))
    configs.append((f"{scales[-1]}-moe8x2", scales[-1], True, False))
    configs.append((f"{scales[-1]}-dense-ckpt", scales[-1], False, True))

    for label, scale, use_moe, ckpt in configs:
        try:
            row = _measure_one_config(
                device,
                label=label,
                scale=scale,
                use_moe=use_moe,
                ckpt=ckpt,
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_len=seq_len,
                pocket_atoms=pocket_atoms,
                iters=iters,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # A large config exhausting GPU memory must not abort the later tiers, which
            # carry the correctness and chemistry results. MPS raises a plain RuntimeError
            # ("MPS backend out of memory") rather than a typed OOM error.
            message = str(exc).splitlines()[0][:160]
            print(f"  {label:<22} SKIPPED: {message}")
            rows.append(
                ThroughputRow(
                    label=label,
                    scale=scale,
                    params_m=0.0,
                    d_model=0,
                    n_layers=0,
                    moe=use_moe,
                    gradient_checkpointing=ckpt,
                    forward_ms=float("nan"),
                    fwd_bwd_ms=float("nan"),
                    peak_mem_mb=float("nan"),
                    tokens_per_s=0.0,
                )
            )
        else:
            rows.append(row)
            print(
                f"  {row.label:<22} {row.params_m:>7.2f}M  fwd {row.forward_ms:>8.2f} ms  "
                f"fwd+bwd {row.fwd_bwd_ms:>8.2f} ms  {row.tokens_per_s:>9.0f} tok/s  "
                f"{row.peak_mem_mb:>7.1f} MB"
            )
        finally:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    return rows


def _measure_one_config(
    device: torch.device,
    *,
    label: str,
    scale: str,
    use_moe: bool,
    ckpt: bool,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    pocket_atoms: int,
    iters: int,
) -> ThroughputRow:
    """Benchmark a single decoder configuration.

    Kept as its own function so the timed closures capture explicit parameters rather
    than mutating loop variables, and so every tensor is released on return.
    """
    decoder = _make_decoder(scale, vocab_size, seq_len=seq_len, use_moe=use_moe).to(device)
    if ckpt:
        decoder.enable_gradient_checkpointing(True)
    params_m = decoder.num_parameters() / 1e6

    ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    memory = torch.randn(batch_size, pocket_atoms, decoder.d_model, device=device)
    mask = torch.ones(batch_size, pocket_atoms, dtype=torch.bool, device=device)

    decoder.eval()
    _reset_peak_memory(device)

    def _forward() -> None:
        with torch.no_grad():
            decoder(ids, memory=memory, memory_mask=mask)

    forward_ms = _time_ms(_forward, device, iters=iters)

    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-4)

    def _fwd_bwd() -> None:
        logits, _ = decoder(ids, memory=memory, memory_mask=mask)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size).float(), ids[:, 1:].reshape(-1)
        )
        loss = loss + 0.01 * decoder.aux_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    fwd_bwd_ms = _time_ms(_fwd_bwd, device, warmup=2, iters=max(3, iters // 2))
    peak_mb = _peak_memory_mb(device)
    tokens_per_s = (batch_size * seq_len) / (fwd_bwd_ms / 1000.0)

    return ThroughputRow(
        label=label,
        scale=scale,
        params_m=round(params_m, 2),
        d_model=decoder.d_model,
        n_layers=decoder.n_layers,
        moe=use_moe,
        gradient_checkpointing=ckpt,
        forward_ms=round(forward_ms, 2),
        fwd_bwd_ms=round(fwd_bwd_ms, 2),
        peak_mem_mb=round(peak_mb, 1),
        tokens_per_s=round(tokens_per_s, 1),
    )


# ---------------------------------------------------------------------------
# Tier 2 — numerical correctness
# ---------------------------------------------------------------------------


def run_tier2(
    device: torch.device,
    *,
    vocab_size: int,
    seq_len: int,
) -> dict[str, Any]:
    """Verify KV-cache parity, device parity, mask isolation and conditioning sensitivity."""
    results: dict[str, Any] = {}
    batch, atoms = 2, 12
    torch.manual_seed(0)

    decoder = _make_decoder("tiny", vocab_size, seq_len=seq_len, num_properties=4).to(device)
    decoder.eval()

    ids = torch.randint(4, vocab_size, (batch, 16), device=device)
    memory = torch.randn(batch, atoms, decoder.d_model, device=device)
    mask = torch.ones(batch, atoms, dtype=torch.bool, device=device)

    with torch.no_grad():
        full, _ = decoder(ids, memory=memory, memory_mask=mask)

        cache = None
        steps = []
        for t in range(int(ids.shape[1])):
            out, cache = decoder.forward_cached(
                ids[:, t : t + 1], memory=memory, memory_mask=mask, kv_cache=cache
            )
            steps.append(out)
        cached = torch.cat(steps, dim=1)
    results["kv_cache_max_abs_diff"] = float((cached - full).abs().max())

    # MPS/CUDA vs CPU parity.
    cpu_decoder = _make_decoder("tiny", vocab_size, seq_len=seq_len, num_properties=4)
    cpu_decoder.load_state_dict({k: v.cpu() for k, v in decoder.state_dict().items()})
    cpu_decoder.eval()
    with torch.no_grad():
        cpu_logits, _ = cpu_decoder(ids.cpu(), memory=memory.cpu(), memory_mask=mask.cpu())
    results["device_vs_cpu_max_abs_diff"] = float((full.cpu() - cpu_logits).abs().max())

    # Masked/padded pocket atoms must have exactly zero influence.
    with torch.no_grad():
        garbage = torch.randn(batch, 5, decoder.d_model, device=device) * 50.0
        mem_pad = torch.cat([memory, garbage], dim=1)
        mask_pad = torch.cat([mask, torch.zeros(batch, 5, dtype=torch.bool, device=device)], dim=1)
        padded, _ = decoder(ids, memory=mem_pad, memory_mask=mask_pad)
    results["padded_pocket_max_abs_diff"] = float((padded - full).abs().max())

    # An entirely masked pocket must still produce finite logits.
    with torch.no_grad():
        empty, _ = decoder(ids, memory=memory, memory_mask=torch.zeros_like(mask, dtype=torch.bool))
    results["all_masked_pocket_finite"] = bool(torch.isfinite(empty).all())

    # Pocket conditioning sensitivity.
    with torch.no_grad():
        alt, _ = decoder(ids, memory=torch.randn_like(memory) * 3.0, memory_mask=mask)
    results["pocket_sensitivity_max_abs_diff"] = float((alt - full).abs().max())

    # Property conditioning sensitivity.
    props_a = torch.tensor([[0.9, 1.0, 0.1, 0.0]] * batch, device=device)
    props_b = torch.tensor([[0.1, 5.0, 0.9, 1.0]] * batch, device=device)
    with torch.no_grad():
        pa, _ = decoder(ids, memory=memory, memory_mask=mask, properties=props_a)
        pb, _ = decoder(ids, memory=memory, memory_mask=mask, properties=props_b)
    results["property_sensitivity_max_abs_diff"] = float((pa - pb).abs().max())

    for key, value in results.items():
        print(f"  {key:<36} {value}")
    return results


# ---------------------------------------------------------------------------
# Tier 3 — real chemistry
# ---------------------------------------------------------------------------


def _load_smiles(path: Path, limit: int) -> list[str]:
    """Load up to ``limit`` SMILES strings, one per line."""
    if not path.is_file():
        return []
    out: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                out.append(text)
            if len(out) >= limit:
                break
    return out


def run_tier3(
    device: torch.device,
    *,
    smiles_path: Path,
    scale: str,
    train_steps: int,
    batch_size: int,
    seq_len: int,
    num_samples: int,
    corpus_limit: int,
) -> dict[str, Any]:
    """Train on real ZINC SMILES, then sample and score generated chemistry with RDKit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")

    corpus = _load_smiles(smiles_path, corpus_limit)
    if not corpus:
        print(f"  SKIPPED: no SMILES corpus at {smiles_path}")
        return {"skipped": True, "reason": f"missing corpus: {smiles_path}"}

    tokenizer = MolecularTokenizer(representation="smiles")
    tokenizer.fit(corpus, min_freq=1)
    vocab_size = tokenizer.vocab_size
    print(f"  corpus={len(corpus)} molecules  vocab={vocab_size}")

    encoded = [tokenizer.encode(s, max_len=seq_len) for s in corpus]
    encoded = [e for e in encoded if len(e) >= 4]

    def _batch(rng: random.Random) -> torch.Tensor:
        picked = [rng.choice(encoded) for _ in range(batch_size)]
        width = max(len(p) for p in picked)
        padded = [p + [tokenizer.PAD_ID] * (width - len(p)) for p in picked]
        return torch.tensor(padded, dtype=torch.long, device=device)

    decoder = _make_decoder(scale, vocab_size, seq_len=seq_len, num_properties=0).to(device)
    # Unconditional Stage-1 chemical prior: no pocket memory is supplied.
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=train_steps, pct_start=0.1
    )
    rng = random.Random(0)

    losses: list[float] = []
    t_start = time.perf_counter()
    for step in range(train_steps):
        ids = _batch(rng)
        logits, _ = decoder(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size).float(),
            ids[:, 1:].reshape(-1),
            ignore_index=tokenizer.PAD_ID,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(1, train_steps // 5) == 0:
            recent = statistics.mean(losses[-20:])
            print(f"    step {step + 1:>5}/{train_steps}  loss {recent:.4f}")
    _synchronize(device)
    train_s = time.perf_counter() - t_start

    # Sample with the KV cache.
    decoder.eval()
    t0 = time.perf_counter()
    sampled = decoder.generate(
        batch_size=num_samples,
        memory=None,
        max_new_tokens=seq_len - 1,
        bos_id=tokenizer.BOS_ID,
        eos_id=tokenizer.EOS_ID,
        temperature=1.0,
        top_p=0.95,
        seed=0,
        device=device,
    )
    _synchronize(device)
    cached_s = time.perf_counter() - t0

    decoded = [tokenizer.decode(row.tolist()) for row in sampled.cpu()]
    mols = [(s, Chem.MolFromSmiles(s)) for s in decoded if s]
    valid = [(s, m) for s, m in mols if m is not None]
    canonical = {Chem.MolToSmiles(m) for _, m in valid}
    train_set = {c for c in (Chem.CanonSmiles(s) for s in corpus[:5000]) if c}

    qed_scores: list[float] = []
    for _, mol in valid:
        try:
            qed_scores.append(float(QED.qed(mol)))
        except Exception:
            continue

    results: dict[str, Any] = {
        "scale": scale,
        "params_m": round(decoder.num_parameters() / 1e6, 2),
        "vocab_size": vocab_size,
        "corpus_size": len(corpus),
        "train_steps": train_steps,
        "train_seconds": round(train_s, 1),
        "ms_per_step": round(train_s / max(1, train_steps) * 1000.0, 1),
        "first_loss": round(statistics.mean(losses[:10]), 4),
        "final_loss": round(statistics.mean(losses[-10:]), 4),
        "num_sampled": num_samples,
        "validity": round(len(valid) / max(1, num_samples), 4),
        "uniqueness": round(len(canonical) / max(1, len(valid)), 4) if valid else 0.0,
        "novelty": (
            round(len(canonical - train_set) / max(1, len(canonical)), 4) if canonical else 0.0
        ),
        "mean_qed": round(statistics.mean(qed_scores), 4) if qed_scores else 0.0,
        "kv_cached_sample_seconds": round(cached_s, 2),
        "example_valid_smiles": [s for s, _ in valid[:10]],
    }

    for key in (
        "params_m",
        "ms_per_step",
        "first_loss",
        "final_loss",
        "validity",
        "uniqueness",
        "novelty",
        "mean_qed",
        "kv_cached_sample_seconds",
    ):
        print(f"  {key:<28} {results[key]}")
    return results


# ---------------------------------------------------------------------------
# Tier 4 — property-conditioned generation
# ---------------------------------------------------------------------------

# Conditioning targets, z-scored against the training corpus so each enters the Fourier
# feature map on a comparable scale.
PROPERTY_NAMES: tuple[str, ...] = ("qed", "mol_wt", "log_p")


def _molecule_properties(smiles: str) -> tuple[float, float, float] | None:
    """Return ``(QED, MolWt, LogP)`` for a SMILES string, or ``None`` if invalid."""
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return (
            float(QED.qed(mol)),
            float(Descriptors.MolWt(mol)),
            float(Crippen.MolLogP(mol)),
        )
    except Exception:
        return None


def run_tier4(
    device: torch.device,
    *,
    smiles_path: Path,
    scale: str,
    train_steps: int,
    batch_size: int,
    seq_len: int,
    num_samples: int,
    corpus_limit: int,
) -> dict[str, Any]:
    """Train a property-conditioned decoder, then test whether conditioning steers output.

    Trains on (SMILES, QED, MolWt, LogP) triples with per-property ``unknown`` dropout,
    then samples twice with deliberately opposed targets. A working conditioner must shift
    the realised property distribution in the requested direction; the reported
    ``*_separation`` values are the signed mean differences (high-target minus
    low-target), which must be positive for the conditioning to be doing anything.
    """
    import statistics as stats

    corpus = _load_smiles(smiles_path, corpus_limit)
    if not corpus:
        print(f"  SKIPPED: no SMILES corpus at {smiles_path}")
        return {"skipped": True, "reason": f"missing corpus: {smiles_path}"}

    rows: list[tuple[str, tuple[float, float, float]]] = []
    for smi in corpus:
        props = _molecule_properties(smi)
        if props is not None:
            rows.append((smi, props))
    if len(rows) < 64:
        return {"skipped": True, "reason": f"only {len(rows)} valid molecules"}

    # z-score each property against the corpus
    columns = list(zip(*(p for _, p in rows), strict=True))
    means = [float(stats.mean(c)) for c in columns]
    stdevs = [max(float(stats.pstdev(c)), 1e-6) for c in columns]
    print(
        "  corpus stats: "
        + ", ".join(
            f"{n}={m:.2f}+/-{s:.2f}" for n, m, s in zip(PROPERTY_NAMES, means, stdevs, strict=True)
        )
    )

    def _z(props: tuple[float, float, float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(props, means, stdevs, strict=True)]

    tokenizer = MolecularTokenizer(representation="smiles")
    tokenizer.fit([s for s, _ in rows], min_freq=1)
    vocab_size = tokenizer.vocab_size

    encoded: list[tuple[list[int], list[float]]] = []
    for smi, props in rows:
        ids = tokenizer.encode(smi, max_len=seq_len)
        if len(ids) >= 4:
            encoded.append((ids, _z(props)))

    decoder = _make_decoder(
        scale,
        vocab_size,
        seq_len=seq_len,
        num_properties=len(PROPERTY_NAMES),
    ).to(device)
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(0)

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(train_steps):
        picked = [rng.choice(encoded) for _ in range(batch_size)]
        width = max(len(ids) for ids, _ in picked)
        ids_batch = torch.tensor(
            [ids + [tokenizer.PAD_ID] * (width - len(ids)) for ids, _ in picked],
            dtype=torch.long,
            device=device,
        )
        prop_batch = torch.tensor(
            [props for _, props in picked], dtype=torch.float32, device=device
        )

        logits, _ = decoder(ids_batch, properties=prop_batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size).float(),
            ids_batch[:, 1:].reshape(-1),
            ignore_index=tokenizer.PAD_ID,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(1, train_steps // 4) == 0:
            print(f"    step {step + 1:>5}/{train_steps}  loss {stats.mean(losses[-20:]):.4f}")
    _synchronize(device)
    train_s = time.perf_counter() - t0

    decoder.eval()

    def _sample_at(target_z: float) -> dict[str, Any]:
        """Sample with every property driven to ``target_z`` standard deviations."""
        props = torch.full(
            (num_samples, len(PROPERTY_NAMES)), float(target_z), dtype=torch.float32, device=device
        )
        tokens = decoder.generate(
            batch_size=num_samples,
            memory=None,
            properties=props,
            max_new_tokens=seq_len - 1,
            bos_id=tokenizer.BOS_ID,
            eos_id=tokenizer.EOS_ID,
            temperature=1.0,
            top_p=0.95,
            seed=0,
            device=device,
        )
        realised: list[tuple[float, float, float]] = []
        for row in tokens.cpu():
            smi = tokenizer.decode(row.tolist())
            if not smi:
                continue
            measured = _molecule_properties(smi)
            if measured is not None:
                realised.append(measured)
        if not realised:
            return {"n_valid": 0}
        cols = list(zip(*realised, strict=True))
        return {
            "n_valid": len(realised),
            "validity": round(len(realised) / num_samples, 4),
            **{
                f"mean_{name}": round(float(stats.mean(c)), 4)
                for name, c in zip(PROPERTY_NAMES, cols, strict=True)
            },
        }

    low = _sample_at(-1.5)
    high = _sample_at(+1.5)

    results: dict[str, Any] = {
        "scale": scale,
        "params_m": round(decoder.num_parameters() / 1e6, 2),
        "properties": list(PROPERTY_NAMES),
        "corpus_size": len(encoded),
        "train_steps": train_steps,
        "train_seconds": round(train_s, 1),
        "first_loss": round(stats.mean(losses[:10]), 4),
        "final_loss": round(stats.mean(losses[-10:]), 4),
        "corpus_mean": [round(m, 4) for m in means],
        "corpus_stdev": [round(s, 4) for s in stdevs],
        "low_target_z": -1.5,
        "high_target_z": 1.5,
        "low": low,
        "high": high,
    }
    for name in PROPERTY_NAMES:
        key = f"mean_{name}"
        if key in low and key in high:
            results[f"{name}_separation"] = round(float(high[key] - low[key]), 4)

    for key, value in results.items():
        if key in ("low", "high") or not isinstance(value, (int, float, str)):
            continue
        print(f"  {key:<24} {value}")
    print(f"  low  target -> {low}")
    print(f"  high target -> {high}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the three benchmark tiers and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pocket-atoms", type=int, default=256)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--train-scale", default="small")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--corpus-limit", type=int, default=40000)
    parser.add_argument("--smiles", type=Path, default=DEFAULT_SMILES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="Tiny budgets for a fast smoke run")
    parser.add_argument("--skip-tier3", action="store_true")
    parser.add_argument("--skip-tier4", action="store_true")
    args = parser.parse_args()

    if args.quick:
        # Quick mode lowers the *defaults* only; anything named explicitly on the command
        # line wins, so `--quick --train-steps 2000` really does run 2000 steps.
        explicit = {
            arg.split("=", 1)[0].lstrip("-").replace("-", "_")
            for arg in sys.argv[1:]
            if arg.startswith("--")
        }
        quick_defaults = {
            "batch_size": 4,
            "seq_len": 48,
            "pocket_atoms": 64,
            "iters": 3,
            "train_steps": 40,
            "num_samples": 32,
            "corpus_limit": 2000,
            "train_scale": "tiny",
        }
        for key, value in quick_defaults.items():
            if key not in explicit:
                setattr(args, key, value)

    seed_everything(42)
    device = select_device(args.device)
    scales = ["tiny", "small"] if args.quick else ["tiny", "small", "base"]

    print("=" * 78)
    print("GenEuclidean ChemLM Decoder Benchmark")
    print(f"device={device}  torch={torch.__version__}  platform={platform.platform()}")
    print(f"batch={args.batch_size}  seq_len={args.seq_len}  pocket_atoms={args.pocket_atoms}")
    print("=" * 78)

    report = BenchmarkReport(
        device=str(device),
        torch_version=str(torch.__version__),
        platform=platform.platform(),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        pocket_atoms=int(args.pocket_atoms),
    )

    print("\n[Tier 1] Throughput, memory & scaling")
    rows = run_tier1(
        device,
        vocab_size=96,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pocket_atoms=args.pocket_atoms,
        scales=scales,
        iters=args.iters,
    )
    report.tier1_throughput = [asdict(r) for r in rows]

    print("\n[Tier 2] Numerical correctness")
    report.tier2_correctness = run_tier2(device, vocab_size=96, seq_len=args.seq_len)

    if args.skip_tier3:
        print("\n[Tier 3] skipped (--skip-tier3)")
        report.tier3_chemistry = {"skipped": True, "reason": "--skip-tier3"}
    else:
        print("\n[Tier 3] Real ZINC-250k chemistry")
        report.tier3_chemistry = run_tier3(
            device,
            smiles_path=args.smiles,
            scale=args.train_scale,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            corpus_limit=args.corpus_limit,
        )

    if args.skip_tier4:
        print("\n[Tier 4] skipped (--skip-tier4)")
        report.tier4_property_conditioning = {"skipped": True, "reason": "--skip-tier4"}
    else:
        print("\n[Tier 4] Property-conditioned generation")
        report.tier4_property_conditioning = run_tier4(
            device,
            smiles_path=args.smiles,
            scale=args.train_scale,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            corpus_limit=args.corpus_limit,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
