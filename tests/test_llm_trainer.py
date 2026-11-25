"""Offline CPU unit & integration tests for GenEuclidean v3.0 LLM training & RPO-DPO."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

pytest.importorskip("transformers")
pytest.importorskip("peft")

from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything  # noqa: E402
from geneuclidean.data.featurizer import LIGAND_FP_DIM, POCKET_NODE_DIM  # noqa: E402
from geneuclidean.models.hf_backbones import (  # noqa: E402
    MOCK_BACKBONE_ID,
    MockSmilesTokenizer,
    load_hf_backbone,
)
from geneuclidean.models.llm_bridge import PocketConditionedLLM  # noqa: E402
from geneuclidean.training.dpo import PreferencePair  # noqa: E402
from geneuclidean.training.llm_trainer import (  # noqa: E402
    CachedPocketLLMDataset,
    CachedPocketRecord,
    PocketLLMTrainer,
    build_pocket_batches_from_records,
    precompute_pocket_cache,
    run_llm_rpo_dpo,
    tokenize_smiles_for_llm,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_SAMPLE_SMILES: tuple[str, ...] = (
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1ccc(cc1)C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccc2c(c1)ccc3ccccc23",
    "CCN(CC)CC",
    "c1ccncc1C(=O)N",
)


def _make_tiny_v2_cfg(seed: int = 7) -> ExperimentConfig:
    """Create a tiny fast CPU ExperimentConfig compatible with mock-tiny-gpt2 (d=64)."""
    cfg = ExperimentConfig(
        name="test-v3-mock",
        llm=LLMConfig(
            backbone_id=MOCK_BACKBONE_ID,
            dtype="float32",
            freeze_pocket_encoder=True,
            num_queries=4,
            perceiver_layers=1,
            perceiver_heads=4,
            num_global_tokens=2,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            phase_a_epochs=1,
            phase_b_epochs=2,
            projector_lr=5e-3,
            phase_b_projector_lr=2e-3,
            lora_lr=5e-3,
            cfg_drop_prob=0.25,
            cvae_kl_weight=0.05,
            grad_accum_steps=1,
        ),
    )
    cfg.data.max_seq_len = 48
    cfg.data.randomize_smiles = True
    cfg.encoder.hidden_dim = 32
    cfg.encoder.vector_dim = 16
    cfg.encoder.out_dim = 32
    cfg.encoder.num_layers = 2
    cfg.encoder.num_rbf = 8
    cfg.encoder.dropout = 0.0
    cfg.cvae.enabled = True
    cfg.cvae.latent_dim = 16
    cfg.cvae.num_latent_tokens = 2
    cfg.decoder.d_model = 32
    cfg.decoder.n_heads = 4
    cfg.decoder.n_layers = 2
    cfg.decoder.d_ff = 64
    cfg.multitask.clip_dim = 16
    cfg.multitask.ligand_fp_dim = LIGAND_FP_DIM
    cfg.train.seed = seed
    cfg.train.device = "cpu"
    cfg.train.batch_size = 4
    cfg.train.warmup_ratio = 0.0
    cfg.sampling.max_len = 24
    cfg.sampling.num_samples = 4
    return cfg


def _make_synthetic_pocket_records(n: int = 8, seed: int = 11) -> dict[str, dict[str, Any]]:
    """Generate deterministic synthetic 3D pocket records matching pdbbind_pockets.pt schema."""
    rng = np.random.default_rng(seed)
    records: dict[str, dict[str, Any]] = {}
    for i in range(n):
        n_atoms = 18 + (i % 5) * 3
        coords = rng.normal(0.0, 3.0, size=(n_atoms, 3)).astype(np.float32)
        node_feats = rng.normal(0.0, 0.5, size=(n_atoms, POCKET_NODE_DIM)).astype(np.float32)
        fp_bits = np.packbits((rng.random(LIGAND_FP_DIM) > 0.85).astype(np.uint8))
        pid = f"t{i:03d}"
        records[pid] = {
            "pdb_id": pid,
            "coords": coords,
            "node_feats": node_feats,
            "contact_mask": (np.arange(n_atoms) < 8),
            "center": np.zeros(3, dtype=np.float32),
            "canonical_smiles": _SAMPLE_SMILES[i % len(_SAMPLE_SMILES)],
            "ligand_fp_packed": fp_bits,
            "pkd": float(5.5 + 0.4 * (i % 6)),
        }
    return records


def _build_mock_model(cfg: ExperimentConfig) -> PocketConditionedLLM:
    """Instantiate an offline mock-tiny-gpt2 PocketConditionedLLM on CPU."""
    assert cfg.llm is not None
    backbone = load_hf_backbone(
        backbone_id=MOCK_BACKBONE_ID,
        dtype="float32",
        lora_r=cfg.llm.lora_r,
        lora_alpha=cfg.llm.lora_alpha,
        lora_dropout=cfg.llm.lora_dropout,
        use_lora=True,
    )
    return PocketConditionedLLM(
        v2_cfg=cfg,
        backbone=backbone,
        num_queries=cfg.llm.num_queries,
        perceiver_layers=cfg.llm.perceiver_layers,
        perceiver_heads=cfg.llm.perceiver_heads,
        num_global_tokens=cfg.llm.num_global_tokens,
        cfg_drop_prob=cfg.llm.cfg_drop_prob,
        freeze_pocket_encoder=cfg.llm.freeze_pocket_encoder,
    )


def test_llm_config_defaults_validation_and_v3_yaml_configs(tmp_path: Path) -> None:
    """Verify LLMConfig schema, backwards compatibility, and all three v3.0 YAML configs."""
    base_cfg = ExperimentConfig()
    assert base_cfg.llm is None

    llm_cfg = LLMConfig()
    assert llm_cfg.backbone_id == "ibm-research/GP-MoLFormer-Uniq"
    assert llm_cfg.cache_dir == ".hf_cache.nosync"
    assert llm_cfg.freeze_pocket_encoder is True
    assert llm_cfg.num_queries == 16
    assert llm_cfg.phase_a_epochs == 3
    assert llm_cfg.phase_b_epochs == 12

    with pytest.raises(ValidationError):
        LLMConfig.model_validate({"unknown_llm_key": 1})
    with pytest.raises(ValidationError):
        LLMConfig(dtype="float16")  # type: ignore[arg-type]

    cfg_with_llm = ExperimentConfig(llm=llm_cfg)
    yaml_out = cfg_with_llm.to_yaml(tmp_path / "roundtrip_v3.yaml")
    assert ExperimentConfig.from_yaml(yaml_out) == cfg_with_llm

    expected_configs = [
        (
            "configs/geneuclidean_v3_gpmolformer.yaml",
            "ibm-research/GP-MoLFormer-Uniq",
            "float32",
            16,
            1,
            3,
            12,
        ),
        (
            "configs/geneuclidean_v3_gpt2zinc.yaml",
            "entropy/gpt2_zinc_87m",
            "float32",
            16,
            1,
            2,
            10,
        ),
        (
            "configs/geneuclidean_v3_qwen25.yaml",
            "Qwen/Qwen2.5-0.5B",
            "bfloat16",
            8,
            2,
            1,
            4,
        ),
    ]
    for rel_path, backbone_id, dtype, bs, accum, ep_a, ep_b in expected_configs:
        loaded = ExperimentConfig.from_yaml(REPO_ROOT / rel_path)
        assert loaded.llm is not None
        assert loaded.llm.backbone_id == backbone_id
        assert loaded.llm.dtype == dtype
        assert loaded.train.batch_size == bs
        assert loaded.llm.grad_accum_steps == accum
        assert loaded.llm.phase_a_epochs == ep_a
        assert loaded.llm.phase_b_epochs == ep_b
        assert loaded.llm.v2_pocket_ckpt == "runs/geneuclidean_v2_painn/best.pt"


def test_tokenization_masking_and_dynamic_smiles_randomization() -> None:
    """Verify native SMILES randomization and general-LLM strict prompt loss masking."""
    tok = MockSmilesTokenizer(vocab_size=96)

    # 1. Native SMILES backbone: [BOS, *smi, EOS], labels=[-100, *smi, EOS]
    inp_nat, lbl_nat = tokenize_smiles_for_llm(
        "CCO",
        tokenizer=tok,
        is_smiles_native=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        max_seq_len=32,
    )
    assert inp_nat[0] == tok.bos_token_id
    assert inp_nat[-1] == tok.eos_token_id
    assert lbl_nat[0] == -100
    assert lbl_nat[1:] == inp_nat[1:]

    # 2. General LLM backbone: masks all tokens in "SMILES: " strictly to -100
    class _CharTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(ch) for ch in text]

    gen_tok = _CharTokenizer()
    prompt = "SMILES: "
    inp_gen, lbl_gen = tokenize_smiles_for_llm(
        "CCO",
        tokenizer=gen_tok,
        is_smiles_native=False,
        prompt_prefix=prompt,
        bos_token_id=None,
        eos_token_id=999,
        max_seq_len=64,
    )
    assert len(inp_gen) == len(prompt) + len("CCO\n") + 1
    assert all(val == -100 for val in lbl_gen[: len(prompt)])
    assert lbl_gen[len(prompt) :] == [*[ord(c) for c in "CCO\n"], 999]

    # 3. Dynamic SMILES randomization on native dataset produces valid varied encodings
    rec = CachedPocketRecord(
        pdb_id="p0",
        node_emb=torch.randn(6, 32),
        pooled_emb=torch.randn(32),
        atom_mask=torch.ones(6, dtype=torch.bool),
        ligand_fp=torch.zeros(LIGAND_FP_DIM),
        canonical_smiles="CC(=O)Oc1ccccc1C(=O)O",
    )
    ds = CachedPocketLLMDataset(
        [rec],
        tokenizer=tok,
        is_smiles_native=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        max_seq_len=64,
        randomize=True,
        seed=42,
    )
    seen_seqs = {tuple(ds[0]["input_ids"].tolist()) for _ in range(8)}
    assert len(seen_seqs) >= 2


def test_precompute_pocket_cache_exact_parity() -> None:
    """Verify precompute_pocket_cache trims padding and matches live encode_pocket bit-for-bit."""
    seed_everything(13)
    cfg = _make_tiny_v2_cfg(seed=13)
    model = _build_mock_model(cfg)
    raw_records = _make_synthetic_pocket_records(n=6, seed=19)
    pids = sorted(raw_records.keys())
    batches = build_pocket_batches_from_records(raw_records, pids, batch_size=4)

    cached = precompute_pocket_cache(model, batches, device="cpu")
    assert len(cached) == len(pids)
    for rec_c, pid in zip(cached, pids, strict=True):
        raw = raw_records[pid]
        expected_n = int(raw["coords"].shape[0])
        assert rec_c.pdb_id == pid
        assert rec_c.node_emb.device.type == "cpu"
        assert rec_c.node_emb.dtype == torch.float32
        assert rec_c.node_emb.shape == (expected_n, cfg.encoder.out_dim)
        assert rec_c.pooled_emb.shape == (cfg.encoder.out_dim,)
        assert bool(rec_c.atom_mask.all().item())
        assert rec_c.ligand_fp is not None and rec_c.ligand_fp.shape == (LIGAND_FP_DIM,)
        assert rec_c.canonical_smiles == raw["canonical_smiles"]

        # Compare single-pocket live encode_pocket against cached tensors
        with torch.no_grad():
            live = model.encode_pocket(
                torch.from_numpy(raw["coords"]).unsqueeze(0),
                torch.from_numpy(raw["node_feats"]).unsqueeze(0),
                torch.ones((1, expected_n), dtype=torch.bool),
            )
        assert torch.allclose(rec_c.node_emb, live.node_emb.squeeze(0), atol=1e-5)
        assert torch.allclose(rec_c.pooled_emb, live.pooled_emb.squeeze(0), atol=1e-5)


def test_two_phase_pocket_llm_trainer_phase_a_and_phase_b(tmp_path: Path) -> None:
    """Verify Phase A freezes LoRA, Phase B updates LoRA, and best_adapter.pt reloads cleanly."""
    seed_everything(23)
    cfg = _make_tiny_v2_cfg(seed=23)
    assert cfg.llm is not None
    model = _build_mock_model(cfg)

    raw_records = _make_synthetic_pocket_records(n=8, seed=29)
    pids = sorted(raw_records.keys())
    train_cache = precompute_pocket_cache(
        model, build_pocket_batches_from_records(raw_records, pids[:6], batch_size=3), device="cpu"
    )
    val_cache = precompute_pocket_cache(
        model, build_pocket_batches_from_records(raw_records, pids[6:], batch_size=2), device="cpu"
    )

    init_snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    lora_names = {k for k in init_snap if "lora_" in k.lower()}
    bridge_names = {
        k
        for k in init_snap
        if k.startswith(("perceiver.", "cvae_proj.", "global_proj.", "prefix_norm.", "null_prefix"))
    }
    frozen_3d_names = {k for k in init_snap if k.startswith(("encoder.", "cvae.", "heads."))}
    base_llm_names = {k for k in init_snap if k.startswith("llm.") and k not in lora_names}
    assert lora_names and bridge_names and frozen_3d_names and base_llm_names

    # --- Phase A ONLY: LoRA MUST remain bit-for-bit frozen, bridge MUST update ---
    cfg.llm.phase_a_epochs = 2
    cfg.llm.phase_b_epochs = 0
    trainer_a = PocketLLMTrainer(model=model, cfg=cfg, device="cpu")
    res_a = trainer_a.fit(train_cache, val_cache, output_dir=tmp_path / "phase_a", log_every=0)
    assert res_a["epochs_completed"] == 2
    assert math.isfinite(float(res_a["best_val_lm_loss"]))
    assert math.isfinite(float(res_a["best_val_ppl"]))

    cur_after_a = dict(model.named_parameters())
    for name in lora_names | frozen_3d_names | base_llm_names:
        assert torch.equal(cur_after_a[name].detach(), init_snap[name]), (
            f"Parameter {name} changed unexpectedly during Phase A!"
        )
    bridge_changed_a = any(
        not torch.equal(cur_after_a[name].detach(), init_snap[name]) for name in bridge_names
    )
    assert bridge_changed_a, "Bridge/projector weights did not update during Phase A!"

    # --- Phase A + Phase B: LoRA MUST update, frozen 3D + base LLM MUST stay identical ---
    snap_before_b = {k: v.detach().clone() for k, v in model.named_parameters()}
    cfg.llm.phase_a_epochs = 1
    cfg.llm.phase_b_epochs = 2
    trainer_ab = PocketLLMTrainer(model=model, cfg=cfg, device="cpu")
    out_ab = tmp_path / "phase_ab"
    res_ab = trainer_ab.fit(train_cache, val_cache, output_dir=out_ab, log_every=0)

    assert res_ab["epochs_completed"] == 3
    assert [row["phase"] for row in res_ab["history"]] == ["A", "B", "B"]
    assert (out_ab / "best_adapter.pt").is_file()
    assert (out_ab / "history.jsonl").is_file()
    assert (out_ab / "train_summary.json").is_file()
    assert (out_ab / "config.yaml").is_file()

    cur_after_b = dict(model.named_parameters())
    for name in frozen_3d_names | base_llm_names:
        assert torch.equal(cur_after_b[name].detach(), snap_before_b[name]), (
            f"Frozen parameter {name} drifted during Phase B!"
        )
    lora_changed_b = any(
        not torch.equal(cur_after_b[name].detach(), snap_before_b[name]) for name in lora_names
    )
    assert lora_changed_b, "LoRA weights did not update during Phase B!"

    # Verify compact adapter-only checkpoint reloads bit-exact into a fresh model
    seed_everything(23)
    reloaded = _build_mock_model(cfg)
    reloaded.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if k.startswith(("encoder.", "cvae.", "heads.", "llm."))
        },
        strict=False,
    )
    reloaded.load_adapter_checkpoint(out_ab / "best_adapter.pt", map_location="cpu")
    reloaded.eval()
    # Reload best weights into model too and verify exact logit parity
    model.load_adapter_checkpoint(out_ab / "best_adapter.pt", map_location="cpu")
    model.eval()

    sample_ids = torch.tensor([[1, 4, 4, 8, 2]], dtype=torch.long)
    sample_mask = torch.ones_like(sample_ids)
    rec0 = val_cache[0]
    from geneuclidean.models.encoders.rbf import PocketEncoderOutput

    p_out = PocketEncoderOutput(
        node_emb=rec0.node_emb.unsqueeze(0),
        pooled_emb=rec0.pooled_emb.unsqueeze(0),
        vec_emb=None,
    )
    a_mask = rec0.atom_mask.unsqueeze(0)
    with torch.no_grad():
        out_orig = model(
            input_ids=sample_ids,
            attention_mask=sample_mask,
            pocket_out=p_out,
            atom_mask=a_mask,
            use_posterior=False,
        )
        out_rel = reloaded(
            input_ids=sample_ids,
            attention_mask=sample_mask,
            pocket_out=p_out,
            atom_mask=a_mask,
            use_posterior=False,
        )
    assert torch.allclose(out_orig.logits, out_rel.logits, atol=1e-6)


def test_run_llm_rpo_dpo_freezes_all_non_lora_params_bit_exact(tmp_path: Path) -> None:
    """Verify run_llm_rpo_dpo updates ONLY LoRA weights, freezing all non-LoRA params bit-exact."""
    seed_everything(37)
    cfg = _make_tiny_v2_cfg(seed=37)
    model = _build_mock_model(cfg)

    raw_records = _make_synthetic_pocket_records(n=4, seed=41)
    pids = sorted(raw_records.keys())
    train_cache = precompute_pocket_cache(
        model, build_pocket_batches_from_records(raw_records, pids, batch_size=4), device="cpu"
    )

    # First smoke-test on-policy sampling path (small pocket/sample count on untrained mock model)
    smoke_res = run_llm_rpo_dpo(
        model=model,
        train_records=train_cache,
        cfg=cfg,
        device="cpu",
        dpo_pockets=2,
        samples_per_pocket=6,
        epochs=1,
        beta=0.5,
        lr=1e-3,
        sft_nll_weight=0.25,
        grad_accum_steps=2,
        max_new_tokens=10,
        seed=37,
    )
    assert "num_pairs" in smoke_res

    # Now run deterministic preference pairs to guarantee gradient steps and verify bit-exact freeze
    before_snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    before_grad_flags = {k: p.requires_grad for k, p in model.named_parameters()}

    explicit_pairs = [
        PreferencePair(
            pocket_batch={
                "node_emb": rec.node_emb.unsqueeze(0),
                "pooled_emb": rec.pooled_emb.unsqueeze(0),
                "atom_mask": rec.atom_mask.unsqueeze(0),
            },
            chosen_smiles="CC(=O)Oc1ccccc1C(=O)O",
            rejected_smiles="CC",
            chosen_score=0.82,
            rejected_score=0.25,
        )
        for rec in train_cache
    ]

    dpo_dir = tmp_path / "dpo_out"
    dpo_res = run_llm_rpo_dpo(
        model=model,
        train_records=train_cache,
        cfg=cfg,
        device="cpu",
        epochs=2,
        beta=0.5,
        lr=2e-3,
        sft_nll_weight=0.25,
        grad_accum_steps=2,
        seed=37,
        preference_pairs=explicit_pairs,
        save_dir=dpo_dir,
    )

    assert dpo_res["epochs"] == 2
    assert dpo_res["num_pairs"] == len(explicit_pairs)
    assert (dpo_dir / "best_adapter.pt").is_file()
    assert (dpo_dir / "history.jsonl").is_file()
    assert math.isfinite(float(dpo_res["history"][-1]["dpo_loss"]))
    assert math.isfinite(float(dpo_res["history"][-1]["sft_nll"]))

    after_params = dict(model.named_parameters())
    lora_changed = False
    for name, param in after_params.items():
        assert param.requires_grad == before_grad_flags[name], (
            f"requires_grad was not restored for {name}"
        )
        if "lora_" in name.lower():
            if not torch.equal(param.detach(), before_snap[name]):
                lora_changed = True
        else:
            assert torch.equal(param.detach(), before_snap[name]), (
                f"Non-LoRA parameter {name} was modified during RPO-DPO!"
            )
    assert lora_changed, "LoRA weights did not change during RPO-DPO!"


def test_v3_scripts_parsers_and_comparison_table(tmp_path: Path) -> None:
    """Verify run_v3_llm.py & eval_v3_llm.py CLI parsers and unified v3 comparison table builder."""
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import eval_v3_llm
    import run_v3_llm

    run_args = run_v3_llm.build_arg_parser().parse_args(
        [
            "--model",
            "gpmolformer",
            "--split",
            "casf2016",
            "--device",
            "cpu",
            "--phase-a-epochs",
            "1",
            "--phase-b-epochs",
            "2",
            "--batch-size",
            "8",
            "--dpo-pockets",
            "10",
            "--resume-skip",
        ]
    )
    assert run_args.model == "gpmolformer"
    assert run_args.phase_a_epochs == 1
    assert run_args.phase_b_epochs == 2
    assert run_args.resume_skip is True

    eval_args = eval_v3_llm.build_arg_parser().parse_args(
        ["--models", "gpmolformer", "--cfg-sweep", "--device", "cpu", "--max-pockets", "5"]
    )
    assert eval_args.models == ["gpmolformer"]
    assert eval_args.cfg_sweep is True

    # Verify write_v3_comparison_tables merges 2021 thesis + v2 + v3 JSONs cleanly
    stub_v2 = {
        "model_name": "geneuclidean_v2_painn_dpo",
        "display_name": "GenEuclidean v2 (PaiNN-SE3+DPO)",
        "num_params": 9_455_631,
        "tier1_raw": {
            "validity": 0.916,
            "uniqueness": 0.970,
            "novelty": 0.982,
            "int_div_1": 0.904,
            "QED_mean": 0.521,
            "SA_mean": 3.36,
            "SA_norm_mean": 0.737,
        },
        "tier2_specificity": {
            "matched_tanimoto_mean": 0.157,
            "matched_count_dice_mean": 0.317,
            "vs_shuffled": {"control_tanimoto_mean": 0.095, "cliffs_delta": 0.457},
        },
        "tier3_attention": {"enrichment_ratio_mean": 1.657},
        "tier4_affinity_clip": {
            "affinity": {"pearson_r": 0.746, "rmse": 1.448},
            "clip_retrieval": {"recall@1": 0.305, "roc_auc": 0.937},
        },
    }
    stub_v3 = {
        "model_name": "geneuclidean_v3_gpmolformer",
        "display_name": "GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer-Uniq)",
        "num_params": 51_000_000,
        "tier1_raw": {
            "validity": 0.990,
            "uniqueness": 0.985,
            "novelty": 0.990,
            "int_div_1": 0.895,
            "QED_mean": 0.680,
            "SA_mean": 2.95,
            "SA_norm_mean": 0.783,
        },
        "tier2_specificity": {
            "matched_tanimoto_mean": 0.185,
            "matched_count_dice_mean": 0.355,
            "vs_shuffled": {"control_tanimoto_mean": 0.100, "cliffs_delta": 0.550},
        },
        "tier3_attention": {"enrichment_ratio_mean": 1.720},
        "tier4_affinity_clip": {
            "affinity": {"pearson_r": 0.746, "rmse": 1.448},
            "clip_retrieval": {"recall@1": 0.305, "roc_auc": 0.937},
        },
    }
    (tmp_path / "eval_geneuclidean_v2_painn_dpo.json").write_text(
        json.dumps(stub_v2), encoding="utf-8"
    )
    (tmp_path / "eval_v3_gpmolformer.json").write_text(json.dumps(stub_v3), encoding="utf-8")

    csv_path, md_path = eval_v3_llm.write_v3_comparison_tables(tmp_path)
    assert csv_path.is_file() and md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "2021 Thesis Baseline" in md_text
    assert "GenEuclidean v2 (PaiNN-SE3+DPO)" in md_text
    assert "GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer-Uniq)" in md_text
