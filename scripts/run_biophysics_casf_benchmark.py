#!/usr/bin/env python3
"""CASF-2016 scoring power for the differentiable biophysics Hamiltonian.

``geneuclidean.biophysics`` ships a differentiable MM/GBSA-style Hamiltonian with a
``dg_bind_est`` in kcal/mol, but **nothing in the module, its tests or its research note
ever measured that number against an experimental affinity.** This script does, so the
simulator's accuracy becomes a measurement rather than an assumption.

Why CASF-2016: it is the standard scoring-power benchmark -- 285 protein-ligand complexes
in 57 clusters of 5, each with a curated experimental binding constant, scored in the
**crystal pose**. Scoring power is the Pearson correlation between predicted affinity and
experiment. 266 of the 285 core complexes have local structures under
``data/raw/pbpp2020/``, so this runs offline.

**Honest reference points** (read these before celebrating any number):

* ``Lin_F9`` reports Pearson **R = 0.680** on crystal poses and 0.687 on locally
  optimised poses, described as the best of 34 *classical* scoring functions.
* Machine-learned rescorers reach **R ~ 0.80-0.85** (Delta-AEScore R = 0.80 at
  RMSE 1.32 pK; PLANET v2.0 R = 0.848), but several are trained on PDBbind releases that
  overlap the CASF-2016 core, so their numbers are not all leakage-free.
* A classical, unfitted, physics-only function has no business beating 0.68. If this
  Hamiltonian scores far above that, suspect a bug or leakage before celebrating.

What this measures, and what it deliberately does not:

* **Measured:** Pearson R, Spearman rho and RMSE (in pK units, before and after a linear
  recalibration, which is the CASF convention) for the shipped ``dg_bind_est``; the same
  for the raw ``u_total``; and the correlation of *every individual energy term*, which
  shows which physics actually carries signal.
* **Reported but flagged:** an in-sample ordinary-least-squares refit over all terms. That
  is an **optimistic ceiling**, not a result -- it is fitted and evaluated on the same 266
  complexes. A leakage-controlled fit belongs on the LP-PDBBind split
  (``data/raw/LP_PDBBind.csv`` carries ``new_split`` plus CL1/CL2/CL3 cluster flags
  designed to defeat exactly this kind of leakage).
* **Not measured:** CASF *ranking* power, which is a Spearman correlation computed inside
  each 5-complex cluster and therefore needs the official cluster assignment file. The
  local metadata has only ``pdb_id,pkd``.

A note on ``dg_bind_est``. It is formed as::

    dg_bind = 0.35 * u_total + tds_rot   if u_total < 0 else u_total + tds_rot

The 0.35 is a hardcoded constant whose own comment says it "yields realistic
experimental-scale kcal/mol". Because ``u -> 0.35u`` is strictly monotone, that factor
cannot change the ranking of poses for a *fixed* ligand -- it only rescales the number.
Across different ligands only the ``tds_rot`` term (a count of rotatable bonds) can
reorder anything. This script reports the correlation of both ``dg_bind_est`` and
``u_total`` so that claim is checked rather than asserted.

Run::

    python scripts/run_biophysics_casf_benchmark.py
    python scripts/run_biophysics_casf_benchmark.py --limit 25 --include-strain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "metadata" / "casf_2016_core_labels.csv"
DEFAULT_STRUCTURES = REPO_ROOT / "data" / "raw" / "pbpp2020"
DEFAULT_OUT = REPO_ROOT / "results" / "tables" / "biophysics_casf2016_scoring_power.json"

# dG = -RT ln K; at 298.15 K, 1 pK unit = 1.364 kcal/mol.
KCAL_PER_PK = 1.3639

# Energy components recorded per complex. Each is a plain float on ComplexEnergyBreakdown.
TERM_NAMES: tuple[str, ...] = (
    "e_coulomb",
    "dg_gb_solvation",
    "e_vdw",
    "e_hbond",
    "e_halogen",
    "e_cation_pi",
    "e_pi_stacking",
    "dg_hydrophobic",
    "tds_rotational_penalty",
    "e_internal_strain",
    "u_total",
    "dg_bind_est",
)


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; ``nan`` when either series is constant."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    return pearson(ranks(xs), ranks(ys))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)`` for ``y ~ a*x + b``."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx <= 0.0:
        return 0.0, my
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / sxx
    return slope, my - slope * mx


def rmse(pred: list[float], obs: list[float]) -> float:
    """Root-mean-square error."""
    return math.sqrt(sum((p - o) ** 2 for p, o in zip(pred, obs, strict=True)) / len(pred))


def load_labels(path: Path) -> dict[str, float]:
    """Read ``pdb_id,pkd`` into a lowercase-keyed mapping."""
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("pdb_id") or "").strip().lower()
            raw = (row.get("pkd") or "").strip()
            if not code or not raw:
                continue
            try:
                out[code] = float(raw)
            except ValueError:
                continue
    return out


def score_one(
    code: str,
    struct_dir: Path,
    hamiltonian: Any,
    *,
    include_strain: bool,
) -> dict[str, float]:
    """Parameterize and score a single complex in its crystal pose.

    Raises:
        FileNotFoundError: If the pocket or ligand file is missing.
        ValueError: If RDKit cannot parse the ligand.
    """
    from rdkit import Chem

    from geneuclidean.biophysics.forcefield import (
        parameterize_ligand_biophysics,
        parameterize_pocket_biophysics,
    )

    pocket_path = struct_dir / f"{code}_pocket.pdb"
    ligand_path = struct_dir / f"{code}_ligand.sdf"
    if not pocket_path.is_file():
        raise FileNotFoundError(f"missing pocket: {pocket_path}")
    if not ligand_path.is_file():
        raise FileNotFoundError(f"missing ligand: {ligand_path}")

    # removeHs=False: the SDFs carry explicit hydrogens, which the H-bond, donor-axis and
    # Gasteiger-charge machinery needs. Stripping them silently degrades the physics.
    mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse the ligand SDF")
    if mol.GetNumConformers() == 0:
        raise ValueError("ligand SDF has no 3D conformer")

    lig = parameterize_ligand_biophysics(mol)
    rec = parameterize_pocket_biophysics(pocket_path)
    breakdown = hamiltonian.evaluate_complex(lig, rec, include_internal_strain=include_strain)

    row = {name: float(getattr(breakdown, name)) for name in TERM_NAMES}
    row["n_lig_atoms"] = float(lig.num_atoms)
    row["n_rec_atoms"] = float(rec.num_atoms)
    row["rotatable_bonds"] = float(lig.rotatable_bonds)

    # Trivial size descriptors, kept as the control every scoring function must beat.
    # Heavy-atom count alone is a notoriously strong predictor of measured affinity on
    # PDBbind-derived sets, because bigger ligands bury more surface. A physics score that
    # does not clearly beat it is mostly reporting molecular size.
    from rdkit.Chem import Descriptors

    row["n_heavy_atoms"] = float(mol.GetNumHeavyAtoms())
    row["mol_weight"] = float(Descriptors.MolWt(mol))  # type: ignore[attr-defined]
    return row


def multiterm_in_sample_fit(
    rows: list[dict[str, float]], observed: list[float], features: list[str]
) -> dict[str, Any]:
    """Ordinary least squares over several energy terms, fitted and scored in-sample.

    This is an **optimistic ceiling**, not a validation result: it is fitted on the same
    complexes it is evaluated on. It answers only "how much signal could reweighting the
    existing terms possibly extract?".
    """
    import numpy as np

    design = np.array([[row[name] for name in features] + [1.0] for row in rows], dtype=np.float64)
    target = np.array(observed, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coeffs
    names = [*features, "intercept"]
    return {
        "features": list(features),
        "coefficients": {name: round(float(c), 6) for name, c in zip(names, coeffs, strict=True)},
        "in_sample_pearson_r": round(pearson(predicted.tolist(), observed), 4),
        "in_sample_rmse_pkd": round(rmse(predicted.tolist(), observed), 4),
        "caveat": (
            "Fitted and evaluated on the same complexes. This is an upper bound on what "
            "reweighting these terms could achieve, NOT a measured accuracy. For an "
            "honest fit use the LP-PDBBind split in data/raw/LP_PDBBind.csv, which "
            "provides new_split plus CL1/CL2/CL3 cluster flags specifically to prevent "
            "train/test leakage, and exclude the CASF-2016 core from training."
        ),
    }


def main() -> int:
    """Score the CASF-2016 core set and report correlation against experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all available complexes")
    parser.add_argument(
        "--include-strain",
        action="store_true",
        help=(
            "Enable the internal strain term, which evaluate_complex disables by default. "
            "Note it currently covers only bond lengths and angles -- there is no torsion "
            "term -- so it cannot see the torsional strain that dominates pose quality."
        ),
    )
    args = parser.parse_args()

    if not args.labels.is_file():
        print(f"No CASF-2016 labels at {args.labels}")
        return 1
    if not args.structures.is_dir():
        print(f"No structure directory at {args.structures}")
        return 1

    from geneuclidean.biophysics.forcefield import BiophysicalHamiltonian

    labels = load_labels(args.labels)
    available = sorted(
        code for code in labels if (args.structures / code / f"{code}_pocket.pdb").is_file()
    )
    if args.limit > 0:
        available = available[: args.limit]

    print("=" * 78)
    print("CASF-2016 scoring power - geneuclidean.biophysics differentiable Hamiltonian")
    print(f"labelled complexes      : {len(labels)}")
    print(f"with local structures   : {len(available)}")
    print(f"internal strain enabled : {args.include_strain}")
    print("=" * 78)

    hamiltonian = BiophysicalHamiltonian()
    rows: list[dict[str, float]] = []
    observed: list[float] = []
    codes: list[str] = []
    failures: list[dict[str, str]] = []

    start = time.perf_counter()
    for idx, code in enumerate(available, 1):
        try:
            row = score_one(
                code,
                args.structures / code,
                hamiltonian,
                include_strain=args.include_strain,
            )
        except Exception as exc:  # noqa: BLE001 - every failure is catalogued, not hidden
            failures.append({"pdb_id": code, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if not all(math.isfinite(v) for v in row.values()):
            failures.append({"pdb_id": code, "error": "non-finite energy term"})
            continue
        rows.append(row)
        observed.append(labels[code])
        codes.append(code)
        if idx % 50 == 0:
            print(f"  scored {len(rows)}/{idx} (failed {len(failures)})", flush=True)

    elapsed = time.perf_counter() - start
    if len(rows) < 10:
        print(f"\nOnly {len(rows)} complexes scored; too few to report a correlation.")
        for failure in failures[:10]:
            print(f"  {failure['pdb_id']}: {failure['error']}")
        return 1

    report: dict[str, Any] = {
        "config": {
            "benchmark": "CASF-2016 core set, crystal poses",
            "labels": str(args.labels.relative_to(REPO_ROOT)),
            "structures": str(args.structures.relative_to(REPO_ROOT)),
            "include_internal_strain": bool(args.include_strain),
            "kcal_per_pk_unit": KCAL_PER_PK,
            "seconds": round(elapsed, 1),
        },
        "machine": {"platform": platform.platform(), "loadavg": list(os.getloadavg())},
        "coverage": {
            "n_labelled": len(labels),
            "n_with_structures": len(available),
            "n_scored": len(rows),
            "n_failed": len(failures),
            "failures": failures[:40],
        },
        "reference_points": {
            "Lin_F9_crystal_pose_pearson_r": 0.680,
            "Lin_F9_optimised_pose_pearson_r": 0.687,
            "best_of_34_classical_functions": "Lin_F9",
            "ml_rescorer_range_pearson_r": [0.80, 0.85],
            "note": (
                "A classical physics-only function should land well below 0.68. Several ML "
                "rescorers train on PDBbind releases overlapping the CASF-2016 core, so "
                "their figures are not all leakage-free."
            ),
        },
    }

    # Predicted affinity: a more negative dG means tighter binding, so a higher pK.
    scoring: dict[str, Any] = {}
    for key in ("dg_bind_est", "u_total"):
        series = [row[key] for row in rows]
        pred_pk = [-v / KCAL_PER_PK for v in series]
        slope, intercept = linear_fit(pred_pk, observed)
        recalibrated = [slope * v + intercept for v in pred_pk]
        scoring[key] = {
            "pearson_r": round(pearson(pred_pk, observed), 4),
            "spearman_rho": round(spearman(pred_pk, observed), 4),
            "rmse_pkd_raw": round(rmse(pred_pk, observed), 4),
            "rmse_pkd_after_linear_fit": round(rmse(recalibrated, observed), 4),
            "linear_fit": {"slope": round(slope, 6), "intercept": round(intercept, 6)},
            "n": len(rows),
        }
    report["scoring_power"] = scoring

    # Which physics actually carries signal? Correlate each term directly with pK.
    report["per_term_pearson_r"] = {
        name: round(pearson([row[name] for row in rows], observed), 4)
        for name in (*TERM_NAMES, "n_lig_atoms", "rotatable_bonds")
    }

    # The control the whole Hamiltonian has to beat. On PDBbind-derived sets, ligand size
    # alone predicts affinity well, so a physics score that merely matches heavy-atom
    # count is reporting molecular size dressed up as thermodynamics.
    trivial = {
        name: round(pearson([row[name] for row in rows], observed), 4)
        for name in ("n_heavy_atoms", "mol_weight", "n_lig_atoms", "rotatable_bonds")
    }
    best_trivial_name = max(trivial, key=lambda k: abs(trivial[k]))
    report["trivial_size_baselines"] = {
        **trivial,
        "best_trivial_descriptor": best_trivial_name,
        "best_trivial_pearson_r": trivial[best_trivial_name],
        "u_total_minus_best_trivial": round(
            abs(scoring["u_total"]["pearson_r"]) - abs(trivial[best_trivial_name]), 4
        ),
        "note": (
            "If 'u_total_minus_best_trivial' is near zero or negative, the Hamiltonian adds "
            "nothing over counting atoms and any claim of physical insight is unsupported."
        ),
    }

    fit_features = [
        n
        for n in TERM_NAMES
        if n not in ("u_total", "dg_bind_est") and len({round(row[n], 9) for row in rows}) > 1
    ]
    report["in_sample_multiterm_fit"] = multiterm_in_sample_fit(rows, observed, fit_features)

    report["per_complex"] = [
        {"pdb_id": c, "pkd_experimental": o, **{k: round(v, 4) for k, v in r.items()}}
        for c, o, r in zip(codes, observed, rows, strict=True)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    dg = scoring["dg_bind_est"]
    u_tot = scoring["u_total"]
    print(
        f"\nscored {len(rows)}/{len(available)} complexes in {elapsed:.0f}s "
        f"({len(failures)} failed)"
    )
    print("\n  SCORING POWER (vs experimental pKd)")
    print(
        f"    dg_bind_est : R = {dg['pearson_r']:+.4f}   rho = {dg['spearman_rho']:+.4f}"
        f"   RMSE = {dg['rmse_pkd_after_linear_fit']:.2f} pK (after linear fit)"
    )
    print(
        f"    u_total     : R = {u_tot['pearson_r']:+.4f}   rho = {u_tot['spearman_rho']:+.4f}"
        f"   RMSE = {u_tot['rmse_pkd_after_linear_fit']:.2f} pK (after linear fit)"
    )
    print("    reference   : Lin_F9 = 0.680 (best of 34 classical functions)")
    trivial_report = report["trivial_size_baselines"]
    print("\n  TRIVIAL SIZE CONTROL (what the physics must beat)")
    print(
        f"    best descriptor : {trivial_report['best_trivial_descriptor']} "
        f"R = {trivial_report['best_trivial_pearson_r']:+.4f}"
    )
    print(f"    |u_total| - |best trivial| = {trivial_report['u_total_minus_best_trivial']:+.4f}")
    print("\n  PER-TERM PEARSON R (which physics carries signal)")
    for name, val in sorted(
        report["per_term_pearson_r"].items(),
        key=lambda kv: -abs(kv[1]) if math.isfinite(kv[1]) else 0.0,
    ):
        print(f"    {name:<26} {val:+.4f}")
    ceiling = report["in_sample_multiterm_fit"]
    print(
        f"\n  IN-SAMPLE reweighting ceiling: R = {ceiling['in_sample_pearson_r']:+.4f} "
        f"(optimistic, fitted on the test set)"
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
