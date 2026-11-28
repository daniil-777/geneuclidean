"""Compare quality-ranked and distance-constrained shortlists on one trained pocket model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from rdkit import Chem, DataStructs

from geneuclidean import MoleculeDesigner, MoleculeReport
from geneuclidean.data.featurizer import PocketFeatures
from geneuclidean.evaluation.metrics import get_morgan_bit_fp
from geneuclidean.sampling.rerank import maxmin_diversity_select


def summarize(reports: Sequence[MoleculeReport]) -> dict[str, float | int | None]:
    fps = [get_morgan_bit_fp(Chem.MolFromSmiles(r.canonical_smiles)) for r in reports]
    distances = [
        1.0 - similarity
        for i, fp in enumerate(fps)
        for similarity in DataStructs.BulkTanimotoSimilarity(fp, fps[:i])
    ]
    return {
        "count": len(reports),
        "min_distance": min(distances, default=None),
        "mean_reward": fmean(r.composite_reward for r in reports) if reports else None,
        "mean_qed": fmean(r.qed for r in reports) if reports else None,
        "scaffolds": len({r.murcko_scaffold for r in reports if r.murcko_scaffold}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/geneuclidean_v2_painn/best.pt")
    )
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--pocket", default="10gs")
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    if min(args.draws, args.batch, args.k) < 1:
        parser.error("draws, batch and k must be positive")
    torch.set_num_threads(2)
    # The project's local processed cache contains NumPy as well as Torch arrays.
    record = torch.load(
        args.processed / "pdbbind_pockets.pt", map_location="cpu", weights_only=False, mmap=True
    )["records"][args.pocket.lower()]
    pocket = PocketFeatures(
        args.pocket,
        np.asarray(record["coords"]),
        np.asarray(record["node_feats"]),
        np.asarray(record["contact_mask"]),
        np.asarray(record["center"]),
    )
    vocab = args.processed / "vocab_smiles.json"
    designer = MoleculeDesigner.from_pretrained(args.checkpoint, vocab_path=vocab, device="cpu")
    raw: list[str] = []
    best: dict[str, MoleculeReport] = {}
    for start in range(0, args.draws, args.batch):
        result = designer.design(
            pocket,
            n=min(args.batch, args.draws - start),
            seed=args.seed + start // args.batch,
            include_triage=True,
            include_admet=False,
        )
        raw.extend(result.raw_smiles)
        for report in result.reports:
            key = report.canonical_smiles or report.smiles
            if key not in best or report.composite_reward > best[key].composite_reward:
                best[key] = report

    ranked = sorted(best.values(), key=lambda r: -r.composite_reward)
    eligible = [
        r
        for r in ranked
        if r.triage_passed and r.hygiene_gate > 0 and math.isfinite(r.composite_reward)
    ]
    metrics = {}
    for name, pool in (("valid_unique", ranked), ("triage_eligible", eligible)):
        results = {"score_only": summarize(pool[: args.k])}
        for threshold in (0.5, 0.65, 0.75, 0.8, 0.85, 0.9):
            indices = maxmin_diversity_select(
                [r.canonical_smiles or r.smiles for r in pool],
                [r.composite_reward for r in pool],
                k=args.k,
                min_tanimoto_dist=threshold,
            )
            results[str(threshold)] = summarize([pool[i] for i in indices])
        metrics[name] = results
    with args.checkpoint.open("rb") as ckpt:
        checkpoint_sha256 = hashlib.file_digest(ckpt, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "pocket": args.pocket,
                "seed": args.seed,
                "batch_size": args.batch,
                "draws": len(raw),
                "valid": sum(bool(s.strip()) and Chem.MolFromSmiles(s) is not None for s in raw),
                "unique": len(ranked),
                "eligible": len(eligible),
                "shortlists": metrics,
                "raw_sha256": hashlib.sha256((json.dumps(raw) + "\n").encode()).hexdigest(),
                "checkpoint_sha256": checkpoint_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
