#!/usr/bin/env python3
"""Phase 1 Calibration & Empirical Study: Beyond-PoseBusters 7-Axis Pose Reality Audit.

Audits two real structural cohorts to quantify BOTH false-positive calibration on
experimental crystal structures AND hidden physicality failures missed by PoseBusters on
AutoDock Vina docked leads:

1. **Arm A — Experimental Co-Crystal Negative Controls:** All curated human CNS native
   co-crystal ligands paired with their experimental clean pockets
   (`data/cns_neuro.nosync/**/*_lig.sdf` + `*_pocket.pdb`). Verifies that high-resolution
   experimental structures pass our saturated-ring strain, torsion-strain, and microstate
   audits at high specificity (low false-positive rate).
2. **Arm B — PoseBusters-Passing Docked Leads (28 CNS Complexes):** Audits all 28 docked
   human CNS complexes (`web/public/data/complexes/*.json`) across all 7 axes (PoseBusters,
   RingStrain, TorsionStrain, pH 7.4 Microstate, 6-Ch PLIF, MedChem Hygiene, CNS MPO v2 +
   12-Endpoint Avoid-ome) to quantify exactly how many PoseBusters-passing docked poses still
   harbor physical or MedChem liabilities.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from geneuclidean.audit import PoseRealityAuditor, summarize_audit_batch
from geneuclidean.cns.admet import CNSAdmetSuite

OUT_JSON = Path("results/tables/beyond_posebusters_audit.json")
V4_ADMET_PATH = Path("data/cns_neuro.nosync/v4_admet_avoidome_oracles.joblib")


def _discover_crystal_pairs() -> list[tuple[str, Path, Path]]:
    roots = [
        Path("data/cns_neuro.nosync/structures/clean_pockets"),
        Path("data/cns_neuro.nosync/v4_expansion/structures/clean_pockets"),
        Path("data/cns_neuro.nosync/ts_circuit_expansion/structures/clean_pockets"),
    ]
    pairs: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for lig_path in sorted(root.glob("*_lig.sdf")):
            stem = lig_path.name.replace("_lig.sdf", "")
            pdb_path = root / f"{stem}_pocket.pdb"
            if pdb_path.is_file() and stem not in seen:
                seen.add(stem)
                pairs.append((stem, lig_path, pdb_path))
    return pairs


def main() -> None:
    t0 = time.perf_counter()
    v4_admet = CNSAdmetSuite.try_load(V4_ADMET_PATH)
    auditor_crystals = PoseRealityAuditor(
        ring_strain_threshold_kcal=3.0,
        ring_n_ref_confs=24,
        torsion_barrier_threshold_kcal=2.5,
        torsion_max_red=0,
        strict_input_charge=False,
        property_profile="oral_ro5",
        admet_suite=None,
        seed=42,
    )
    auditor_docked = PoseRealityAuditor(
        ring_strain_threshold_kcal=3.0,
        ring_n_ref_confs=24,
        torsion_barrier_threshold_kcal=2.5,
        torsion_max_red=0,
        strict_input_charge=False,
        property_profile="cns_mpo_v2",
        admet_suite=v4_admet,
        seed=42,
    )

    # Arm A: Native experimental co-crystal structures
    crystal_pairs = _discover_crystal_pairs()
    print(f"=== [Arm A] Auditing {len(crystal_pairs)} Native Experimental Co-Crystal Complexes ===")
    crystal_reports = []
    for stem, lig_path, pdb_path in crystal_pairs:
        try:
            rep = auditor_crystals.audit(lig_path, pocket_pdb=pdb_path, name=stem)
            crystal_reports.append(rep)
        except Exception as exc:
            print(f"  [skip {stem}]: {exc}")
    crystal_summary = summarize_audit_batch(crystal_reports)
    print(
        f"  Crystals (n={crystal_summary['n_audited']}): "
        f"PB={crystal_summary['posebusters_pass_rate']:.1%}, "
        f"Ring={crystal_summary['ring_strain_pass_rate']:.1%}, "
        f"Torsion={crystal_summary['torsion_strain_pass_rate']:.1%}, "
        f"Microstate={crystal_summary['microstate_pass_rate']:.1%}, "
        f"BeyondPB={crystal_summary['beyond_pb_physical_pass_rate']:.1%}"
    )

    # Arm B: 28 Docked Human CNS Complexes
    complex_files = sorted(Path("web/public/data/complexes").glob("*.json"))
    print(f"=== [Arm B] Auditing {len(complex_files)} Docked Human CNS Complexes ===")
    docked_reports = []
    for fp in complex_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        lig_sdf = data.get("ligand_sdf", "")
        poc_pdb = data.get("pocket_pdb", "")
        if not lig_sdf:
            continue
        rep = auditor_docked.audit(
            lig_sdf,
            pocket_pdb=poc_pdb if poc_pdb else None,
            name=f"{data.get('target', fp.stem)}_{data.get('lead_id', fp.stem)}",
        )
        docked_reports.append(rep)
    docked_summary = summarize_audit_batch(docked_reports)
    print(
        f"  Docked CNS (n={docked_summary['n_audited']}): "
        f"PB={docked_summary['posebusters_pass_rate']:.1%}, "
        f"Ring={docked_summary['ring_strain_pass_rate']:.1%}, "
        f"Torsion={docked_summary['torsion_strain_pass_rate']:.1%}, "
        f"Microstate={docked_summary['microstate_pass_rate']:.1%}, "
        f"BeyondPB={docked_summary['beyond_pb_physical_pass_rate']:.1%}, "
        f"PB-pass-but-BeyondPB-fail="
        f"{docked_summary['pb_passing_but_failed_beyond_pb_count']}/"
        f"{docked_summary['n_audited']} "
        f"({docked_summary['pb_passing_but_failed_beyond_pb_rate']:.1%})"
    )

    payload: dict[str, Any] = {
        "generated_by": "scripts/run_beyond_posebusters_audit.py",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "arm_a_native_crystals": {
            "summary": crystal_summary,
            "flagged": [r.to_dict() for r in crystal_reports if not r.passed_physical_beyond_pb],
            "reports": [r.to_dict() for r in crystal_reports],
        },
        "arm_b_docked_cns_28": {
            "summary": docked_summary,
            "flagged": [r.to_dict() for r in docked_reports if not r.passed_physical_beyond_pb],
            "reports": [r.to_dict() for r in docked_reports],
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elapsed = payload["elapsed_seconds"]
    print(f"Saved Beyond-PoseBusters calibration audit -> {OUT_JSON} ({elapsed}s)")


if __name__ == "__main__":
    main()
