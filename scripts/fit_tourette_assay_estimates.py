"""Rebuild the packaged, structure-specific Fang assay estimates from the authors' public CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn import __version__ as sklearn_version

from geneuclidean.cns.admet import make_scaffold_kfold
from geneuclidean.cns.dmpk_regression import ConformalDMPKRegressor, load_reference_endpoint
from geneuclidean.human_twin import tourette_candidate

SOURCE = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/b00df003de117ce9e5b381afd886095c5f2af2d5/ADME_public_set_3521.csv"
SHA256 = "2cfabc2667740c224487876c33b23124159ef43294e0f9e4d926cb6276c95a3b"


def build(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "ADME_public_set_3521.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Expected the pinned public Fang table. See docs/tourette_human_twin.md")
    candidates = [tourette_candidate(k) for k in ("DRD1_v53_L1", "DRD1_1")]
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    endpoints = {}
    for endpoint in ("Biogen_log_Solubility_pH68", "Biogen_log_Human_Unbound_Percent"):
        smiles, y = load_reference_endpoint(endpoint, data_dir)
        # One test fold and one calibration fold share no scaffolds with proper training.
        folds = make_scaffold_kfold(smiles, n_splits=5, seed=42)
        test, cal = folds[0][1], folds[1][1]
        train = np.concatenate([fold[1] for fold in folds[2:]])
        train_smi = [smiles[i] for i in train]
        labels = np.asarray(y)
        model = ConformalDMPKRegressor(endpoint, alpha=0.10, seed=42)
        model.fit(train_smi, labels[train].tolist(), [smiles[i] for i in cal], labels[cal].tolist())
        heldout = model.predict([smiles[i] for i in test])
        if not all(p.valid and math.isfinite(p.interval_width) for p in heldout):
            raise ValueError("Invalid held-out structure or unresolved calibration interval")
        error = np.array([p.y_pred for p in heldout]) - labels[test]
        coverage = np.mean(
            [
                p.interval_lo <= obs <= p.interval_hi
                for p, obs in zip(heldout, labels[test], strict=True)
            ]
        )
        training_fps = [fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smi]
        query = model.predict([c.smiles for c in candidates])
        estimates = {}
        for c, estimate in zip(candidates, query, strict=True):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fp.GetFingerprint(Chem.MolFromSmiles(c.smiles)), training_fps
            )
            estimates[c.compound_id] = {
                **estimate.to_dict(),
                "max_training_tanimoto": round(max(similarities), 4),
            }
        endpoints[endpoint] = {
            "n_train": len(train),
            "n_calibration": len(cal),
            "n_test": len(test),
            "seed": 42,
            "heldout_mae_log10": round(float(np.mean(np.abs(error))), 4),
            "training_median_baseline_mae_log10": round(
                float(np.mean(np.abs(labels[test] - np.median(labels[train])))), 4
            ),
            "heldout_interval_coverage": round(float(coverage), 4),
            "nominal_interval_coverage": 0.9,
            "predictions": estimates,
        }
    return {
        "source": SOURCE,
        "sha256": SHA256,
        "rdkit": rdBase.rdkitVersion,
        "scikit_learn": sklearn_version,
        "endpoints": endpoints,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.data_dir), indent=2, allow_nan=False) + "\n")
