"""CPU-only CLI and end-to-end integration tests across all GenEuclidean v2 subsystems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from typer.testing import CliRunner

from geneuclidean.cli import app
from geneuclidean.config import (
    CVAEConfig,
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
    SamplingConfig,
    TrainConfig,
    seed_everything,
)
from geneuclidean.data.dataset import PocketLigandDataset, collate_pocket_batch
from geneuclidean.data.featurizer import (
    POCKET_NODE_DIM,
    extract_pocket_from_files,
    featurize_ligand_smiles,
    featurize_pdb_pocket_inference,
)
from geneuclidean.data.tokenizer import MolecularTokenizer
from geneuclidean.evaluation.metrics import (
    compute_properties_batch,
    evaluate_affinity_metrics,
    evaluate_attention_enrichment,
    evaluate_generation_metrics,
    evaluate_pocket_specificity,
)
from geneuclidean.evaluation.visualization import (
    plot_matched_vs_shuffled_similarity,
    plot_property_distributions,
)
from geneuclidean.models.geneuclidean import GenEuclideanModel
from geneuclidean.sampling.decoder_strategies import sample_molecules, sample_with_attention
from geneuclidean.sampling.rerank import rerank_and_select
from geneuclidean.training.trainer import PocketTrainer

runner = CliRunner()

_SYNTHETIC_SMILES: list[str] = [
    "CCO",
    "CCN",
    "CCCO",
    "CCCN",
    "CC(C)O",
    "CC(=O)O",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "COc1ccccc1",
]


def _write_synthetic_pocket_pdb(
    path: Path, shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Path:
    """Create a minimal realistic 24-heavy-atom 4-residue protein binding pocket PDB."""
    sx, sy, sz = shift
    residues = [
        (
            "ALA",
            1,
            [
                ("N", "N", -3.0, 0.0, 0.0),
                ("CA", "C", -2.2, 1.1, 0.2),
                ("C", "C", -1.1, 1.4, -0.7),
                ("O", "O", -0.8, 2.5, -1.1),
                ("CB", "C", -3.0, 2.3, 0.6),
            ],
        ),
        (
            "ASP",
            2,
            [
                ("N", "N", 2.5, 0.1, 0.3),
                ("CA", "C", 3.2, 1.3, 0.5),
                ("C", "C", 4.1, 1.6, -0.6),
                ("O", "O", 4.7, 2.7, -0.7),
                ("CG", "C", 2.2, 2.4, 1.4),
                ("OD1", "O", 1.5, 2.0, 2.3),
                ("OD2", "O", 2.2, 3.6, 1.1),
            ],
        ),
        (
            "PHE",
            3,
            [
                ("N", "N", 0.2, -3.2, 0.4),
                ("CA", "C", 0.5, -4.1, 1.5),
                ("CG", "C", -0.3, -3.5, 2.7),
                ("CD1", "C", -1.6, -3.0, 2.6),
                ("CD2", "C", 0.3, -3.4, 3.9),
                ("CE1", "C", -2.2, -2.5, 3.7),
                ("CZ", "C", -1.5, -2.5, 4.9),
            ],
        ),
        (
            "LYS",
            4,
            [
                ("N", "N", -0.4, 0.5, -3.4),
                ("CA", "C", 0.4, 1.2, -4.4),
                ("C", "C", 1.8, 0.8, -4.4),
                ("O", "O", 2.5, 1.3, -5.2),
                ("NZ", "N", -0.2, 2.9, -2.4),
            ],
        ),
    ]
    lines: list[str] = []
    serial = 1
    for resname, resseq, atoms in residues:
        for atom_name, elem, x, y, z in atoms:
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4s} {resname:>3s} A{resseq:4d}    "
                f"{x + sx:8.3f}{y + sy:8.3f}{z + sz:8.3f}  1.00 20.00          {elem:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_synthetic_ligand_sdf(path: Path, smiles: str) -> Path:
    """Embed a 3D conformer of ``smiles`` centered near the origin and write SDF."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol_no_h = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(str(path))
    writer.write(mol_no_h)
    writer.close()
    return path


def _make_tiny_config(
    *,
    encoder_arch: Literal["painn", "egnn", "e3nn", "schnet"],
    decoder_arch: Literal["transformer", "lstm"],
    vocab_size: int,
    cvae_enabled: bool = True,
    processed_dir: str = "data/processed",
    output_dir: str = "runs/test_tiny",
) -> ExperimentConfig:
    """Create a tiny fast CPU-only ExperimentConfig for integration testing."""
    return ExperimentConfig(
        name=f"tiny-{encoder_arch}-{decoder_arch}",
        data=DataConfig(
            processed_dir=processed_dir,
            max_pocket_atoms=64,
            max_seq_len=32,
            randomize_smiles=False,
            coord_jitter_std=0.0,
            atom_drop_prob=0.0,
        ),
        encoder=EncoderConfig(
            arch=encoder_arch,
            node_in_dim=POCKET_NODE_DIM,
            hidden_dim=32,
            vector_dim=16,
            out_dim=32,
            num_layers=2,
            num_rbf=8,
            cutoff=5.5,
            e3nn_lmax=1,
            use_se3_chirality=True,
            dropout=0.0,
        ),
        cvae=CVAEConfig(
            enabled=cvae_enabled,
            latent_dim=16,
            num_latent_tokens=2,
            kl_max_beta=0.2,
            kl_cycles=1,
            free_bits=0.1,
        ),
        decoder=DecoderConfig(
            arch=decoder_arch,
            vocab_size=vocab_size,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout=0.0,
            use_cross_attn=True,
            gated_cross_attn=True,
            cfg_drop_prob=0.15,
        ),
        multitask=MultiTaskConfig(
            use_pkd_head=True,
            use_clip_head=True,
            use_prop_head=True,
            clip_dim=16,
            ligand_fp_dim=1024,
        ),
        train=TrainConfig(
            seed=42,
            device="cpu",
            batch_size=4,
            epochs=2,
            lr=1e-3,
            encoder_lr=1e-3,
            decoder_lr=5e-4,
            freeze_decoder_epochs=0,
            num_workers=0,
            output_dir=output_dir,
        ),
        sampling=SamplingConfig(
            strategy="min_p",
            num_samples=6,
            max_len=20,
            temperature=1.0,
            min_p=0.08,
            cfg_scale=1.5,
            latent_temp=0.8,
        ),
    )


def _build_synthetic_cache(
    tmp_path: Path,
) -> tuple[Path, Path, Path, MolecularTokenizer, Path]:
    """Build a synthetic 8-complex processed cache, splits.json, and vocab_smiles.json."""
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MolecularTokenizer("smiles").fit(_SYNTHETIC_SMILES)
    tokenizer.save(proc_dir / "vocab_smiles.json")

    records: dict[str, dict[str, Any]] = {}
    first_pdb = tmp_path / "p000_pocket.pdb"
    for idx in range(8):
        pid = f"p00{idx}"
        smi = _SYNTHETIC_SMILES[idx]
        pdb_path = _write_synthetic_pocket_pdb(
            tmp_path / f"{pid}_pocket.pdb",
            shift=(float(idx) * 0.3, float(idx) * -0.2, 0.1 * float(idx)),
        )
        sdf_path = _write_synthetic_ligand_sdf(tmp_path / f"{pid}_ligand.sdf", smi)
        feat = extract_pocket_from_files(
            pdb_path,
            ligand_sdf=sdf_path,
            pdb_id=pid,
            pkd=5.5 + 0.4 * idx,
            pocket_cutoff=10.0,
            max_atoms=64,
        )
        assert feat.canonical_smiles is not None
        assert feat.ligand_fp is not None
        assert feat.ligand_props is not None
        records[pid] = {
            "pdb_id": pid,
            "coords": feat.coords.astype(np.float32),
            "node_feats": feat.node_feats.astype(np.float32),
            "contact_mask": feat.contact_mask.astype(np.bool_),
            "center": feat.center.astype(np.float32),
            "canonical_smiles": feat.canonical_smiles,
            "ligand_fp_packed": np.packbits((feat.ligand_fp > 0.5).astype(np.uint8)),
            "ligand_props": feat.ligand_props.astype(np.float32),
            "pkd": float(feat.pkd),
        }

    all_props = np.stack([rec["ligand_props"] for rec in records.values()], axis=0)
    prop_mean = all_props.mean(axis=0).astype(np.float32)
    prop_std = np.maximum(all_props.std(axis=0), 1e-6).astype(np.float32)

    cache_path = proc_dir / "pdbbind_pockets.pt"
    torch.save(
        {
            "version": "2.0.0",
            "pocket_cutoff": 10.0,
            "max_atoms": 64,
            "node_dim": POCKET_NODE_DIM,
            "fp_dim": 1024,
            "prop_names": ["MW", "LogP", "TPSA", "QED", "HBD", "HBA"],
            "prop_mean": prop_mean,
            "prop_std": prop_std,
            "records": records,
        },
        cache_path,
    )

    splits_payload = {
        "lp_pdbbind": {
            "train": ["p000", "p001", "p002", "p003"],
            "val": ["p004", "p005"],
            "test": ["p006", "p007"],
        },
        "casf2016": {
            "train": ["p000", "p001", "p002", "p003"],
            "val": ["p004", "p005"],
            "test": ["p006", "p007"],
        },
        "scaffold": {
            "train": ["p000", "p001", "p002", "p003"],
            "val": ["p004", "p005"],
            "test": ["p006", "p007"],
        },
        "random": {
            "train": ["p000", "p001", "p002", "p003"],
            "val": ["p004", "p005"],
            "test": ["p006", "p007"],
        },
        "casf2016_core_ids": ["p006", "p007"],
    }
    splits_path = proc_dir / "splits.json"
    splits_path.write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")
    return proc_dir, cache_path, splits_path, tokenizer, first_pdb


def test_cli_help_and_subcommands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in (
        "download",
        "prepare",
        "pretrain",
        "train",
        "train-llm",
        "sample",
        "sample-llm",
        "evaluate",
        "fetch-pdb",
        "info",
    ):
        assert cmd in res.stdout
        sub = runner.invoke(app, [cmd, "--help"])
        assert sub.exit_code == 0, f"{cmd} --help failed: {sub.stdout}"


def test_cli_llm_help_and_info_status(tmp_path: Path) -> None:
    """Verify v3.0 LLM subcommand help pages and LLM/HF cache reporting in `info`."""
    train_help = runner.invoke(app, ["train-llm", "--help"])
    assert train_help.exit_code == 0
    assert "--config" in train_help.stdout
    assert "--phase-a-epochs" in train_help.stdout

    sample_help = runner.invoke(app, ["sample-llm", "--help"])
    assert sample_help.exit_code == 0
    assert "--adapter" in sample_help.stdout
    assert "--pdb-id" in sample_help.stdout
    assert "--pdb-file" in sample_help.stdout

    info_res = runner.invoke(
        app,
        [
            "info",
            "--processed-dir",
            str(tmp_path / "missing_proc"),
            "--raw-dir",
            str(tmp_path / "missing_raw"),
            "--metadata-dir",
            "data/metadata",
            "--hf-cache-dir",
            str(tmp_path / "missing_hf"),
        ],
    )
    assert info_res.exit_code == 0
    assert "LLM Extra ([llm])" in info_res.stdout
    assert "HF LLM cache" in info_res.stdout


def test_cli_info(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "info",
            "--processed-dir",
            str(tmp_path / "missing_proc"),
            "--raw-dir",
            str(tmp_path / "missing_raw"),
            "--metadata-dir",
            "data/metadata",
        ],
    )
    assert res.exit_code == 0
    assert "3.0.0" in res.stdout
    assert "painn" in res.stdout
    assert "MISSING" in res.stdout


def test_end_to_end_pipeline_train_sample_eval_viz_and_cli(tmp_path: Path) -> None:
    """Full CPU smoke test: featurize -> train -> checkpoint -> Min-p+CFG sample -> eval -> plot."""
    seed_everything(42)
    proc_dir, cache_path, splits_path, tokenizer, sample_pdb = _build_synthetic_cache(tmp_path)

    # Verify ligand-free inference featurization contract
    inf_feat = featurize_pdb_pocket_inference(sample_pdb, pocket_cutoff=10.0, max_atoms=64)
    assert inf_feat.coords.dtype == np.float32
    assert inf_feat.node_feats.dtype == np.float32
    assert inf_feat.node_feats.shape == (len(inf_feat.coords), POCKET_NODE_DIM)

    cfg = _make_tiny_config(
        encoder_arch="painn",
        decoder_arch="transformer",
        vocab_size=tokenizer.vocab_size,
        cvae_enabled=True,
        processed_dir=str(proc_dir),
        output_dir=str(tmp_path / "run_painn"),
    )

    train_ds = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type="lp_pdbbind",
        split_name="train",
        max_seq_len=cfg.data.max_seq_len,
    )
    val_ds = PocketLigandDataset(
        cache_path=cache_path,
        splits_path=splits_path,
        tokenizer=tokenizer,
        split_type="lp_pdbbind",
        split_name="val",
        max_seq_len=cfg.data.max_seq_len,
    )
    train_loader = [collate_pocket_batch([train_ds[i] for i in range(len(train_ds))])]
    val_loader = [collate_pocket_batch([val_ds[i] for i in range(len(val_ds))])]

    model = GenEuclideanModel(cfg)
    trainer = PocketTrainer(model=model, cfg=cfg, tokenizer=tokenizer, device="cpu")
    fit_summary = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    best_ckpt = Path(fit_summary["best_ckpt_path"])
    assert best_ckpt.is_file()
    assert len(fit_summary["history"]) == 2

    # Sample via Min-p + CFG with attention maps
    pocket_batch = {
        "coords": torch.from_numpy(inf_feat.coords).unsqueeze(0),
        "node_feats": torch.from_numpy(inf_feat.node_feats).unsqueeze(0),
        "atom_mask": torch.ones((1, len(inf_feat.coords)), dtype=torch.bool),
    }
    gen = torch.Generator(device="cpu").manual_seed(42)
    smiles_list, token_lists, attn_maps = sample_with_attention(
        model=model,
        pocket_batch=pocket_batch,
        tokenizer=tokenizer,
        cfg=cfg.sampling,
        generator=gen,
        device=torch.device("cpu"),
    )
    assert len(smiles_list) == cfg.sampling.num_samples
    assert len(token_lists) == cfg.sampling.num_samples
    assert len(attn_maps) == cfg.sampling.num_samples
    assert any(m is not None for m in attn_maps)

    # Attention enrichment check
    first_attn = next(m for m in attn_maps if m is not None)
    assert first_attn.shape[1] == len(inf_feat.coords)
    contact_mask = np.zeros(len(inf_feat.coords), dtype=bool)
    contact_mask[:6] = True
    if first_attn.shape[0] > 0:
        enr = evaluate_attention_enrichment(
            first_attn.mean(dim=0).numpy(),
            np.ones(len(inf_feat.coords), dtype=bool),
            contact_mask,
        )
        assert "enrichment_ratio" in enr

    # Evaluate Tier 1, Tier 2, Tier 3 metrics + MPO reranking + publication figures
    eval_gen_smiles = smiles_list + ["CCO", "CCCO", "c1ccccc1", "Cc1ccccc1"]
    tier1 = evaluate_generation_metrics(
        generated_smiles=eval_gen_smiles,
        train_smiles=_SYNTHETIC_SMILES[:4],
        reference_smiles=_SYNTHETIC_SMILES[4:8],
    )
    assert 0.0 <= tier1["validity"] <= 1.0
    assert tier1["n_valid"] >= 4

    reranked = rerank_and_select(eval_gen_smiles, k=3, relax_if_needed=True)
    assert len(reranked) >= 1

    tier2 = evaluate_pocket_specificity(
        per_pocket_generated={
            "p006": ["CCO", "CCCO", "c1ccccc1"],
            "p007": ["Cc1ccccc1", "COc1ccccc1"],
        },
        per_pocket_native_smiles={"p006": "c1ccccc1", "p007": "Cc1ccccc1"},
        per_pocket_shuffled_generated={"p006": ["CCN", "CCCN"], "p007": ["CCO", "CC(=O)O"]},
        n_bootstrap=50,
    )
    assert "matched_tanimoto_mean" in tier2 and "vs_shuffled" in tier2

    aff = evaluate_affinity_metrics(np.array([5.5, 6.2, 7.1, 8.0]), np.array([5.6, 6.0, 7.3, 7.8]))
    assert aff["pearson_r"] > 0.9

    fig_prop = plot_property_distributions(
        ref_props=compute_properties_batch(_SYNTHETIC_SMILES[:6]),
        gen_props_dict={"PaiNN-v2": compute_properties_batch(eval_gen_smiles)},
        out_path=tmp_path / "figs" / "props.png",
    )
    assert fig_prop.is_file() and fig_prop.stat().st_size > 1000

    fig_sim = plot_matched_vs_shuffled_similarity(
        matched_sims=[0.42, 0.38, 0.51],
        shuffled_sims=[0.18, 0.21, 0.15],
        null_sims=[0.16, 0.19, 0.14],
        out_path=tmp_path / "figs" / "sim.png",
    )
    assert fig_sim.is_file() and fig_sim.stat().st_size > 1000

    # CLI `sample` end-to-end against saved checkpoint & PDB file
    out_json = tmp_path / "cli_sample.json"
    sample_res = runner.invoke(
        app,
        [
            "sample",
            "--checkpoint",
            str(best_ckpt),
            "--pdb-file",
            str(sample_pdb),
            "--vocab",
            str(proc_dir / "vocab_smiles.json"),
            "--strategy",
            "min_p",
            "--num-samples",
            "4",
            "--cfg-scale",
            "1.5",
            "--device",
            "cpu",
            "--out-json",
            str(out_json),
        ],
    )
    assert sample_res.exit_code == 0, sample_res.stdout
    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(saved["raw_smiles"]) == 4

    # CLI `evaluate` end-to-end against saved checkpoint & synthetic cache
    eval_out = tmp_path / "cli_eval"
    eval_res = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(best_ckpt),
            "--split",
            "test",
            "--processed-dir",
            str(proc_dir),
            "--vocab",
            str(proc_dir / "vocab_smiles.json"),
            "--num-pockets",
            "2",
            "--num-samples",
            "4",
            "--device",
            "cpu",
            "--out-dir",
            str(eval_out),
        ],
    )
    assert eval_res.exit_code == 0, eval_res.stdout
    eval_summary = json.loads((eval_out / "eval_summary.json").read_text(encoding="utf-8"))
    assert eval_summary["num_pockets_evaluated"] == 2
    assert "tier1_generation" in eval_summary
    assert "tier3_affinity" in eval_summary


@pytest.mark.parametrize(
    ("encoder_arch", "decoder_arch", "cvae_enabled"),
    [
        ("painn", "transformer", True),
        ("e3nn", "lstm", True),
        ("e3nn", "lstm", False),
        ("egnn", "transformer", False),
        ("schnet", "lstm", False),
    ],
)
def test_cross_module_trainer_and_sampler_matrix(
    tmp_path: Path,
    encoder_arch: Literal["painn", "egnn", "e3nn", "schnet"],
    decoder_arch: Literal["transformer", "lstm"],
    cvae_enabled: bool,
) -> None:
    """Verify PocketTrainer + sample_molecules + sample_with_attention across model combinations."""
    seed_everything(7)
    tokenizer = MolecularTokenizer("smiles").fit(_SYNTHETIC_SMILES[:6])
    cfg = _make_tiny_config(
        encoder_arch=encoder_arch,
        decoder_arch=decoder_arch,
        vocab_size=tokenizer.vocab_size,
        cvae_enabled=cvae_enabled,
        output_dir=str(tmp_path / f"run_{encoder_arch}_{decoder_arch}_{cvae_enabled}"),
    )
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"epochs": 1})})

    # Construct minimal in-memory collated batch (exact shared batch contract)
    samples: list[dict[str, Any]] = []
    for i in range(4):
        smi = _SYNTHETIC_SMILES[i]
        ids = tokenizer.encode(smi, max_len=32)
        feat_lig = featurize_ligand_smiles(smi, fp_size=1024)
        assert feat_lig is not None
        _, fp, props = feat_lig
        n_atoms = 18 + i
        samples.append(
            {
                "pdb_id": f"m{i:03d}",
                "coords": torch.randn(n_atoms, 3, dtype=torch.float32) * 3.0,
                "node_feats": torch.randn(n_atoms, POCKET_NODE_DIM, dtype=torch.float32),
                "contact_mask": torch.arange(n_atoms) < 6,
                "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                "target_ids": torch.tensor(ids[1:], dtype=torch.long),
                "pkd": 6.0 + 0.5 * i,
                "ligand_fp": torch.from_numpy(fp),
                "ligand_props": torch.from_numpy(props),
                "canonical_smiles": smi,
            }
        )
    batch = collate_pocket_batch(samples)

    model = GenEuclideanModel(cfg)
    trainer = PocketTrainer(model=model, cfg=cfg, tokenizer=tokenizer, device="cpu")
    summary = trainer.fit(train_loader=[batch], val_loader=[batch])
    assert Path(summary["best_ckpt_path"]).is_file()
    assert np.isfinite(summary["best_val_objective"])

    single_pocket = {
        "coords": batch["coords"][:1],
        "node_feats": batch["node_feats"][:1],
        "atom_mask": batch["atom_mask"][:1],
    }
    samp_cfg = SamplingConfig(
        strategy="min_p",
        num_samples=3,
        max_len=12,
        temperature=1.0,
        min_p=0.1,
        cfg_scale=1.5,
        latent_temp=0.8,
    )
    gen = torch.Generator(device="cpu").manual_seed(7)
    sampled = sample_molecules(
        model=model,
        pocket_batch=single_pocket,
        tokenizer=tokenizer,
        sampling_cfg=samp_cfg,
        generator=gen,
        device=torch.device("cpu"),
    )
    assert len(sampled) == 3

    smiles_a, tokens_a, attns_a = sample_with_attention(
        model=model,
        pocket_batch=single_pocket,
        tokenizer=tokenizer,
        cfg=samp_cfg,
        generator=gen,
        device=torch.device("cpu"),
    )
    assert len(smiles_a) == 3
    assert len(tokens_a) == 3
    assert len(attns_a) == 3
    # Verify padded atoms receive exact 0.0 cross-attention weight
    pad_positions = ~single_pocket["atom_mask"][0]
    if bool(pad_positions.any().item()):
        for attn in attns_a:
            if attn is not None and attn.numel() > 0:
                assert float(attn[:, pad_positions].abs().max().item()) == 0.0
