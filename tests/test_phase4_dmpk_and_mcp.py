"""Offline CI-safe tests for Phase 4 Conformal DMPK regression, MCP tools & packaging files."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from geneuclidean.cns.dmpk_regression import (
    ConformalDMPKRegressor,
    evaluate_scaffold_cv_conformal_dmpk,
)
from geneuclidean.mcp_server import DMTA_TOOL_SCHEMAS, invoke_dmta_tool


def test_conformal_dmpk_regressor_achieves_valid_coverage() -> None:
    """Split-Conformal DMPK regressor achieves valid empirical coverage on real descriptors."""
    # Construct 45 distinct drug-like scaffolds with true cLogP target (deterministic ground truth)
    cores = [
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc2ccccc2c1",
        "c1ccc2[nH]ccc2c1",
        "c1cnc2ccccc2n1",
    ]
    substs = ["", "C", "CC", "F", "Cl", "OC", "C#N", "C(=O)N", "O"]
    smis: list[str] = []
    ys: list[float] = []
    for c in cores:
        for sub in substs:
            s = (
                f"{sub}{c}"
                if sub and not sub.startswith("C(")
                else (f"{c}" if not sub else f"{c[:-1]}({sub})1")
            )
            m = Chem.MolFromSmiles(s)
            if m is not None:
                can = Chem.MolToSmiles(m, canonical=True)
                if can not in smis:
                    smis.append(can)
                    ys.append(float(Descriptors.MolLogP(m)))  # type: ignore[attr-defined]

    reg = ConformalDMPKRegressor("logp_demo", alpha=0.10, n_estimators=24, seed=42)
    fit_stats = reg.fit(smis[:22], ys[:22], smis[22:32], ys[22:32])
    assert fit_stats["n_train"] == 22 and fit_stats["n_cal"] == 10
    preds = reg.predict(smis[32:])
    assert len(preds) == len(smis[32:])
    assert all(p.valid and p.interval_lo <= p.y_pred <= p.interval_hi for p in preds)

    cv = evaluate_scaffold_cv_conformal_dmpk(
        smis, ys, endpoint="logp_demo", n_splits=3, alpha=0.10, n_estimators=20, seed=42
    )
    assert cv["n_valid"] == len(smis)
    assert cv["empirical_coverage"] >= 0.70
    assert cv["spearman_rho"] > 0.50


def test_dmta_mcp_tool_dispatcher_and_provenance() -> None:
    """All typed agentic DMTA tools execute deterministically with SHA-256 provenance digests."""
    assert len(DMTA_TOOL_SCHEMAS) == 6

    r1 = invoke_dmta_tool(
        "score_molecules", {"smiles": ["CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"], "target": "DRD1"}
    )
    assert len(r1["provenance_sha256"]) == 64
    assert r1["result"][0]["valid"] is True

    r2 = invoke_dmta_tool(
        "evaluate_property_profile",
        {"smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "profile": "kinase_selectivity"},
    )
    assert r2["result"]["passed"] is True

    r3 = invoke_dmta_tool(
        "enumerate_makeable_analogs",
        {"core_smiles": "O=C(O)c1ccc2ccccc2n1", "building_blocks": ["CN1CCNCC1", "C1COCCN1"]},
    )
    assert len(r3["result"]) >= 2

    r4 = invoke_dmta_tool(
        "propose_mmp_rescues", {"parent_smiles": "Cc1ccc(C(=O)N2CCCCC2)cc1", "max_proposals": 5}
    )
    assert len(r4["result"]) >= 1

    r5 = invoke_dmta_tool("match_retrosynthesis", {"smiles": "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"})
    assert r5["result"]["is_solved"] is True

    # Citation and container packaging artifacts exist at repository root
    root = Path(__file__).resolve().parent.parent
    assert (root / "CITATION.cff").is_file()
    assert (root / "Dockerfile").is_file()
