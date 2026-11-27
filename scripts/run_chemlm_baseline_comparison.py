#!/usr/bin/env python3
"""Head-to-head: ``chemlm`` against the legacy ``transformer`` decoder it replaces.

``run_chemlm_decoder_benchmark.py`` establishes that ``chemlm`` *works*. It cannot
establish that ``chemlm`` is *better*, because it trains no baseline -- so comparing its
numbers to figures published by other papers (different corpora, sample counts and
training budgets) proves nothing. This script runs the missing control.

Both decoders get the **same** corpus, step budget, optimizer, schedule, batch size,
sequence length and sampling policy (temperature 1.0, top_p 0.95, seeded CPU multinomial,
``<pad>``/``<bos>``/``<unk>`` suppressed). The only difference is the architecture.
``chemlm`` samples through its KV cache; the legacy decoder has none, so it is sampled
with a full-sequence loop. That is not a confound for *quality*: cached and
full-sequence decoding agree to ~7e-07, far below sampling noise. It is precisely the
quantity being measured for *speed*.

The script also answers a narrower question: how much did the missing special-token
suppression inflate validity? The same trained weights are decoded twice, once with
suppression and once without, and the realised ``<pad>``/``<bos>``/``<unk>`` emissions
are counted.

What the first run of this script found (M3 Pro, MPS, 5,000 steps, batch 32, 512 samples,
recorded in ``results/tables/chemlm_baseline_comparison.json``):

* **Molecule quality is indistinguishable.** validity 0.9551 vs 0.9512 (489 vs 487 valid
  of 512), uniqueness 1.0 vs 1.0, novelty 0.998 vs 0.9979, mean QED 0.7447 vs 0.7455 --
  the legacy decoder is marginally *higher* on QED. Both simply reproduce the ZINC-250k
  distribution, whose own mean QED is 0.7318. Any claim that this architecture generates
  better molecules than the one it replaces is unsupported.
* **Sampling is 12.2x faster** (1.98 s vs 24.2 s for 512 sequences), which is what the KV
  cache buys and the reason the architecture is worth having: it is the difference
  between a usable and an unusable inner loop for RL or large-scale screening.
* **19% fewer parameters for equal quality** (27.45M vs 33.74M), from grouped-query
  attention shrinking the K/V projections.
* **Training step time is a wash** (162.8 vs 159.4 ms/step; ``chemlm`` is ~2% slower).
* **Suppression changed nothing measurable.** The trained model emitted zero ``<pad>``,
  ``<bos>`` or ``<unk>`` tokens in 512 samples, so suppressed and unsuppressed metrics
  are identical. The guard is still correct -- it makes a structural guarantee out of
  what was previously an empirical accident -- but it did not correct an inflated number.

Treat the wall-clock figures as machine-specific: the same configuration measured
162.8 ms/step here and 319.9 ms/step in ``chemlm_decoder_benchmark.json``, a 1.96x spread
driven by background load. Only ratios measured inside a single run are comparable.

Run::

    python scripts/run_chemlm_baseline_comparison.py
    python scripts/run_chemlm_baseline_comparison.py --train-steps 500 --num-samples 128
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from geneuclidean.config import seed_everything, select_device
from geneuclidean.data.chem_corpora import novelty_report
from geneuclidean.data.tokenizer import MolecularTokenizer
from geneuclidean.models.decoders.chemlm import SCALE_PRESETS, ChemLMDecoder
from geneuclidean.models.decoders.transformer import PocketTransformerDecoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMILES = REPO_ROOT / "data" / "processed" / "zinc250k_smiles.txt"
DEFAULT_OUT = REPO_ROOT / "results" / "tables" / "chemlm_baseline_comparison.json"


def _shape_probs(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    """Nucleus truncation, matching :meth:`ChemLMDecoder._shape_probs` exactly.

    Duplicated rather than imported so that the baseline is sampled by code that is
    provably independent of the model under test, while remaining numerically identical.
    """
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        drop = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    total = probs.sum(dim=-1, keepdim=True)
    safe = torch.where(total > 0, probs / total.clamp_min(1e-12), torch.ones_like(probs))
    return safe / safe.sum(dim=-1, keepdim=True)


@torch.no_grad()
def sample_without_cache(
    model: torch.nn.Module,
    *,
    num_samples: int,
    max_new_tokens: int,
    bos_id: int,
    eos_id: int,
    banned: tuple[int, ...],
    device: torch.device,
    seed: int = 0,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Autoregressive sampling via a full forward pass per step (no KV cache)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    tokens = torch.full((num_samples, 1), int(bos_id), dtype=torch.long, device=device)
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)
    ban = torch.tensor(sorted(set(banned)), dtype=torch.long, device=device) if banned else None

    for _ in range(int(max_new_tokens)):
        out = model(tokens)
        logits = out[0] if isinstance(out, tuple) else out
        nxt = logits[:, -1, :].float()
        if ban is not None:
            nxt = nxt.index_fill(-1, ban, float("-inf"))
        picked = torch.multinomial(
            _shape_probs(nxt, top_p=top_p).detach().to("cpu"), 1, generator=gen
        )
        nxt_tok = picked.squeeze(-1).to(device=device)
        nxt_tok = torch.where(finished, torch.full_like(nxt_tok, int(eos_id)), nxt_tok)
        tokens = torch.cat((tokens, nxt_tok.unsqueeze(-1)), dim=1)
        finished = finished | (nxt_tok == int(eos_id))
        if bool(finished.all().item()):
            break
    return tokens


def score_samples(decoded: list[str], corpus: list[str]) -> dict[str, Any]:
    """Score generated SMILES with RDKit against the full reference corpus."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    report = novelty_report(decoded, corpus)

    qeds: list[float] = []
    for text in decoded:
        if not text:
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        try:
            qeds.append(float(QED.qed(mol)))
        except Exception:
            continue

    return {
        "n_generated": report.n_generated,
        "n_valid": report.n_valid,
        "validity": round(report.validity, 4),
        "uniqueness": round(report.uniqueness, 4),
        "novelty": round(report.novelty, 4),
        "novelty_reference_size": report.n_reference,
        "mean_qed": round(statistics.mean(qeds), 4) if qeds else 0.0,
    }


def count_special_tokens(tokens: torch.Tensor, tok: MolecularTokenizer) -> dict[str, int]:
    """Count ``<pad>``/``<bos>``/``<unk>`` after the forced leading ``<bos>`` column."""
    body = tokens[:, 1:]
    out: dict[str, int] = {}
    for tid, name in ((tok.PAD_ID, "pad"), (tok.BOS_ID, "bos"), (tok.UNK_ID, "unk")):
        out[f"{name}_occurrences"] = int((body == tid).sum())
        out[f"{name}_sequences"] = int((body == tid).any(dim=1).sum())
    return out


def train_one(
    model: torch.nn.Module,
    encoded: list[list[int]],
    tok: MolecularTokenizer,
    device: torch.device,
    *,
    vocab_size: int,
    train_steps: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    """Train ``model`` for a fixed step budget and report the loss trajectory."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=train_steps, pct_start=0.1
    )
    rng = random.Random(0)
    losses: list[float] = []

    start = time.perf_counter()
    for step in range(train_steps):
        picked = [rng.choice(encoded) for _ in range(batch_size)]
        width = max(len(p) for p in picked)
        ids = torch.tensor(
            [p + [tok.PAD_ID] * (width - len(p)) for p in picked],
            dtype=torch.long,
            device=device,
        )
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size).float(),
            ids[:, 1:].reshape(-1),
            ignore_index=tok.PAD_ID,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(1, train_steps // 5) == 0:
            print(
                f"    step {step + 1:>5}/{train_steps}  loss {statistics.mean(losses[-20:]):.4f}",
                flush=True,
            )

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    return {
        "train_seconds": round(seconds, 1),
        "ms_per_step": round(seconds / max(1, train_steps) * 1000.0, 1),
        "first_loss": round(statistics.mean(losses[:10]), 4),
        "final_loss": round(statistics.mean(losses[-10:]), 4),
    }


def main() -> int:
    """Train both decoders on identical data and compare them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--scale", default="small", choices=sorted(SCALE_PRESETS))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-limit", type=int, default=250_000)
    parser.add_argument("--smiles", type=Path, default=DEFAULT_SMILES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seed_everything(42)
    device = select_device(args.device)

    if not args.smiles.is_file():
        print(f"No SMILES corpus at {args.smiles}; nothing to compare.")
        return 1

    corpus = [
        line.strip()
        for line in args.smiles.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.corpus_limit]

    tok = MolecularTokenizer(representation="smiles")
    tok.fit(corpus, min_freq=1)
    vocab_size = tok.vocab_size
    encoded = [e for e in (tok.encode(s, max_len=args.seq_len) for s in corpus) if len(e) >= 4]
    banned = (tok.PAD_ID, tok.BOS_ID, tok.UNK_ID)

    draws = args.train_steps * args.batch_size
    n = max(1, len(encoded))
    seen_fraction = 1.0 - math.exp(draws * math.log1p(-1.0 / n)) if n > 1 else 1.0

    print("=" * 78)
    print("ChemLM vs legacy PocketTransformerDecoder - matched-budget comparison")
    print(f"device={device} corpus={len(corpus)} vocab={vocab_size}")
    print(f"steps={args.train_steps} batch={args.batch_size} samples={args.num_samples}")
    print("=" * 78)

    results: dict[str, Any] = {
        "config": {
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_samples": args.num_samples,
            "scale": args.scale,
            "corpus_size": len(corpus),
            "vocab_size": vocab_size,
            "device": str(device),
            "optimizer": f"AdamW lr={args.lr} wd=0.01 + OneCycleLR pct_start=0.1",
            "sampling": (
                "temperature=1.0 top_p=0.95 seed=0, <pad>/<bos>/<unk> suppressed; "
                "identical for both models"
            ),
            "corpus_fraction_seen": round(seen_fraction, 4),
            "epochs_equivalent": round(draws / n, 3),
            "note": (
                "Batches are drawn WITH REPLACEMENT, so the run touches only "
                "corpus_fraction_seen of the corpus. Novelty measured against the full "
                "corpus is therefore optimistic relative to a converged model."
            ),
        },
        "machine": {
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "loadavg": list(os.getloadavg()),
        },
    }

    preset = SCALE_PRESETS[args.scale]

    # --- ChemLM ---------------------------------------------------------------
    print("\n[1/2] chemlm")
    chem = ChemLMDecoder(
        vocab_size=vocab_size,
        d_model=preset.d_model,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        d_ff=preset.d_ff,
        max_seq_len=args.seq_len,
        dropout=0.0,
    ).to(device)
    chem_stats = train_one(
        chem,
        encoded,
        tok,
        device,
        vocab_size=vocab_size,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    chem_stats["params_m"] = round(chem.num_parameters() / 1e6, 2)
    chem.eval()

    start = time.perf_counter()
    suppressed = chem.generate(
        batch_size=args.num_samples,
        memory=None,
        max_new_tokens=args.seq_len - 1,
        bos_id=tok.BOS_ID,
        eos_id=tok.EOS_ID,
        temperature=1.0,
        top_p=0.95,
        suppress_token_ids=banned,
        seed=0,
        device=device,
    )
    chem_stats["kv_cached_sample_seconds"] = round(time.perf_counter() - start, 2)

    unsuppressed = chem.generate(
        batch_size=args.num_samples,
        memory=None,
        max_new_tokens=args.seq_len - 1,
        bos_id=tok.BOS_ID,
        eos_id=tok.EOS_ID,
        temperature=1.0,
        top_p=0.95,
        suppress_token_ids=(),
        seed=0,
        device=device,
    )

    chem_stats["suppressed"] = score_samples(
        [tok.decode(r.tolist()) for r in suppressed.cpu()], corpus
    )
    chem_stats["unsuppressed"] = score_samples(
        [tok.decode(r.tolist()) for r in unsuppressed.cpu()], corpus
    )
    chem_stats["special_tokens_emitted_when_unsuppressed"] = count_special_tokens(
        unsuppressed.cpu(), tok
    )
    results["chemlm"] = chem_stats
    print(f"    {chem_stats['suppressed']}")

    del chem
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # --- legacy transformer ---------------------------------------------------
    print("\n[2/2] legacy transformer (matched budget)")
    legacy = PocketTransformerDecoder(
        vocab_size=vocab_size,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        n_layers=preset.n_layers,
        d_ff=preset.d_ff,
        max_seq_len=args.seq_len,
        dropout=0.0,
    ).to(device)
    legacy_stats = train_one(
        legacy,
        encoded,
        tok,
        device,
        vocab_size=vocab_size,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    legacy_stats["params_m"] = round(sum(p.numel() for p in legacy.parameters()) / 1e6, 2)
    legacy.eval()

    start = time.perf_counter()
    legacy_tokens = sample_without_cache(
        legacy,
        num_samples=args.num_samples,
        max_new_tokens=args.seq_len - 1,
        bos_id=tok.BOS_ID,
        eos_id=tok.EOS_ID,
        banned=banned,
        device=device,
        seed=0,
    )
    legacy_stats["uncached_sample_seconds"] = round(time.perf_counter() - start, 2)
    legacy_stats["suppressed"] = score_samples(
        [tok.decode(r.tolist()) for r in legacy_tokens.cpu()], corpus
    )
    results["legacy_transformer"] = legacy_stats
    print(f"    {legacy_stats['suppressed']}")

    # --- verdict --------------------------------------------------------------
    chem_q, legacy_q = chem_stats["suppressed"], legacy_stats["suppressed"]
    results["verdict"] = {
        "validity_delta": round(chem_q["validity"] - legacy_q["validity"], 4),
        "mean_qed_delta": round(chem_q["mean_qed"] - legacy_q["mean_qed"], 4),
        "novelty_delta": round(chem_q["novelty"] - legacy_q["novelty"], 4),
        "param_ratio": round(chem_stats["params_m"] / legacy_stats["params_m"], 3),
        "sampling_speedup": round(
            legacy_stats["uncached_sample_seconds"]
            / max(chem_stats["kv_cached_sample_seconds"], 1e-6),
            2,
        ),
        "train_step_ratio": round(
            chem_stats["ms_per_step"] / max(legacy_stats["ms_per_step"], 1e-6), 3
        ),
        "suppression_changed_metrics": (chem_stats["suppressed"] != chem_stats["unsuppressed"]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    v = results["verdict"]
    print("\n" + "=" * 78)
    print(f"  validity delta (chemlm - legacy) : {v['validity_delta']:+.4f}")
    print(f"  mean QED delta                   : {v['mean_qed_delta']:+.4f}")
    print(f"  novelty delta                    : {v['novelty_delta']:+.4f}")
    print(f"  parameter ratio                  : {v['param_ratio']}x")
    print(f"  sampling speedup (KV cache)      : {v['sampling_speedup']}x")
    print(f"  training step ratio              : {v['train_step_ratio']}x")
    print(f"  suppression changed any metric   : {v['suppression_changed_metrics']}")
    print("=" * 78)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
