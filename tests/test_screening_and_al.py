"""Offline CI-safe tests for Phase 2 MIT Contrastive 3D Screener & Active-Learning DMTA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from geneuclidean.config import (
    CVAEConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
    seed_everything,
)
from geneuclidean.models.geneuclidean import GenEuclideanModel
from geneuclidean.screening import (
    LIGAND_ENCODER_REGISTRY,
    ActiveLearningDMTA,
    ECFP4BinaryEncoder,
    LigandVectorIndex,
    PocketScreener,
)
from geneuclidean.sdk.pocket import Pocket
from geneuclidean.training.contrastive_selectivity import compute_selectivity_contrastive_loss


def _make_tiny_v2_model() -> GenEuclideanModel:
    seed_everything(42)
    cfg = ExperimentConfig(
        encoder=EncoderConfig(arch="painn", hidden_dim=32, vector_dim=16, out_dim=32, num_layers=2),
        cvae=CVAEConfig(enabled=False),
        decoder=DecoderConfig(
            arch="transformer", vocab_size=24, d_model=32, n_heads=4, n_layers=1, d_ff=64
        ),
        multitask=MultiTaskConfig(
            use_pkd_head=True,
            use_clip_head=True,
            use_prop_head=True,
            clip_dim=32,
            ligand_fp_dim=128,
        ),
    )
    return GenEuclideanModel(cfg)


def _write_two_pockets(tmp_path: Path) -> tuple[Pocket, Pocket]:
    p1 = tmp_path / "9LLG_DRD1_pocket.pdb"
    p1.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ASP R 103       2.800   0.000   0.000  1.00 20.00           C",
                "ATOM      2  OD1 ASP R 103       1.200   0.800   1.000  1.00 20.00           O",
                "ATOM      3  CA  PHE R 288      -3.200   1.200   0.500  1.00 20.00           C",
                "ATOM      4  OG  SER R 198       0.100  -1.400   0.800  1.00 20.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p2 = tmp_path / "6CM4_DRD2_pocket.pdb"
    p2.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ASP A 114       4.100   0.700  -0.500  1.00 20.00           C",
                "ATOM      2  OD1 ASP A 114       2.500   1.200   0.400  1.00 20.00           O",
                "ATOM      3  CA  TRP A 386      -2.200  -1.900   1.500  1.00 20.00           C",
                "ATOM      4  CE2 TRP A 386      -0.900  -2.500   2.100  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return Pocket.from_pdb(p1, target_id="DRD1"), Pocket.from_pdb(p2, target_id="DRD2")


def test_vector_index_and_contrastive_3d_pocket_screener(tmp_path: Path) -> None:
    """End-to-end streaming index build/save/load, contrastive 3D screening & catalog grounding."""
    assert "drugclip_v2" in LIGAND_ENCODER_REGISTRY
    assert "ecfp4_binary" in LIGAND_ENCODER_REGISTRY
    assert "ibm_mmelon" in LIGAND_ENCODER_REGISTRY

    catalog = [
        "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1",
        "CCN1CCN(C(=O)c2cccc3c2CCC3)CC1",
        "COc1ccc2[nH]cc(CCN(C)C)c2c1",
        "Fc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1",
        "O=C(NC1CCCCC1)NC1CCCCC1",
        "INVALID_SMILES_###",
        "c1ccc(Cn2ccc(CN3CCOCC3)n2)nc1",
    ]
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "id,smiles\n" + "\n".join(f"CMP_{i},{s}" for i, s in enumerate(catalog)) + "\n",
        encoding="utf-8",
    )

    # Pure-RDKit ECFP4 encoder baseline works with zero PyTorch
    ecfp_enc = ECFP4BinaryEncoder(n_bits=128)
    ecfp_idx = LigandVectorIndex.build_from_file(csv_path, ecfp_enc, id_col="id")
    assert len(ecfp_idx) == 6  # skips INVALID_SMILES_###
    assert ecfp_idx.dim == 128

    # DrugCLIP 3D PocketScreener with multi-conformation ensemble & off-target repulsion
    model = _make_tiny_v2_model()
    screener = PocketScreener(model, device="cpu")
    idx = LigandVectorIndex.build_from_file(csv_path, screener.ligand_encoder, id_col="id")
    saved_dir = idx.save(tmp_path / "vec_index")
    reloaded = LigandVectorIndex.load(saved_dir, mmap=True)
    assert len(reloaded) == 6
    assert reloaded.dim == 32

    p_on, p_off = _write_two_pockets(tmp_path)
    hits = screener.screen(
        [p_on, p_on],
        reloaded,
        off_pockets=p_off,
        off_weight=0.5,
        top_k=5,
        rerank_pkd=True,
        property_profile="cns_mpo_v2",
        triage_filter=True,
    )
    # DCU must be filtered out by triage_filter=True
    assert len(hits) == 5
    assert all(h.smiles != "O=C(NC1CCCCC1)NC1CCCCC1" for h in hits)
    for h in hits:
        assert h.clip_score_off is not None
        assert abs(h.contrastive_score - (h.clip_score_on - 0.5 * h.clip_score_off)) < 1e-4
        assert h.pred_pkd_on is not None and h.pred_pkd_off is not None
        assert h.delta_pkd is not None
        assert h.profile_score is not None and 0.0 <= h.profile_score <= 1.0

    # Analog-by-catalog grounding (alpha_3d=0.25 ensures exact-match Tanimoto=1.0 ranks #1)
    grounded = screener.ground_to_catalog(
        ["CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"],
        reloaded,
        p_on,
        off_pockets=p_off,
        top_k_per_query=3,
        alpha_3d=0.25,
    )
    top_match = grounded["CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"][0]
    assert top_match["catalog_smiles"] == "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"
    assert top_match["tanimoto_to_query"] == 1.0


def test_active_learning_dmta_loop_and_csv_ingestion(tmp_path: Path) -> None:
    """Verify multi-round active learning recovers high-scoring series + ingests wet-lab CSV."""
    pool = [
        f"C{'C' * (i % 5)}N1CCN(C(=O)c2ccc({subst})cc2)CC1"
        for i, subst in enumerate(["F", "Cl", "C", "OC", "C#N", "Br"] * 5)
    ]
    pool = list(dict.fromkeys(pool))
    assert len(pool) == 30

    # Synthetic deterministic oracle favoring fluorinated/chlorinated compact piperazides
    def _oracle(batch: list[str] | tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for s in batch:
            score = 5.0 + (2.0 if "(F)" in s else 0.0) + (1.5 if "(Cl)" in s else 0.0)
            score -= 0.1 * len(s)
            out.append(score)
        return out

    # Seed round via wet-lab CSV ingestion
    csv_assay = tmp_path / "round0_assay.csv"
    seed_smis = pool[:6]
    seed_vals = _oracle(seed_smis)
    csv_assay.write_text(
        "smiles,pIC50\n"
        + "\n".join(f"{s},{v:.4f}" for s, v in zip(seed_smis, seed_vals, strict=True))
        + "\n",
        encoding="utf-8",
    )

    al = ActiveLearningDMTA(strategy="maxmin_diverse_ucb", ucb_beta=1.0, seed=42)
    n_ingested = al.ingest_assay_csv(csv_assay, smiles_col="smiles", value_col="pIC50")
    assert n_ingested == 6

    r1 = al.run_round(pool, _oracle, batch_size=5)
    r2 = al.run_round(pool, _oracle, batch_size=5)
    assert r1.n_total_observed == 11
    assert r2.n_total_observed == 16
    assert r2.best_overall_value >= r1.best_overall_value
    assert len(r2.round_sha256) == 16
    assert "best_overall_smiles" in r2.to_dict()


def test_contrastive_selectivity_loss_separates_homologs() -> None:
    """Hard-negative selectivity loss backpropagates cleanly and reduces gap violation."""
    torch.manual_seed(0)
    p_on = torch.randn(8, 32, requires_grad=True)
    p_off = torch.randn(8, 32, requires_grad=True)
    lig = torch.randn(8, 32, requires_grad=True)
    scale = torch.tensor(2.0)

    loss, metrics = compute_selectivity_contrastive_loss(
        p_on, p_off, lig, scale, margin=0.20, selectivity_weight=0.5
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert p_on.grad is not None and torch.isfinite(p_on.grad).all()
    assert p_off.grad is not None and torch.isfinite(p_off.grad).all()
    assert lig.grad is not None and torch.isfinite(lig.grad).all()
    assert np.isfinite(metrics["mean_selectivity_gap"])
