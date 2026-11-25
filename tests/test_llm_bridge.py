"""Comprehensive offline + cached-HF unit tests for GenEuclidean v3 3D-to-LLM bridge & sampler.

Runs 100% offline in CI (~2-3s) using ``mock-tiny-gpt2`` while automatically verifying live
cached HuggingFace backbones (``ibm-research/GP-MoLFormer-Uniq``, ``entropy/gpt2_zinc_87m``,
``Qwen/Qwen2.5-0.5B``) and v2.0 ``PaiNN-SE3`` warm-start when ``.hf_cache.nosync`` is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers")
pytest.importorskip("peft")

from geneuclidean.config import (  # noqa: E402
    CVAEConfig,
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
)
from geneuclidean.models.hf_backbones import (  # noqa: E402
    DEFAULT_HF_CACHE_DIR,
    MOCK_BACKBONE_ID,
    MOLFORMER_REPO_ID,
    get_input_embeddings_tensor,
    load_hf_backbone,
)
from geneuclidean.models.llm_bridge import (  # noqa: E402
    PocketConditionedLLM,
    PocketPerceiverResampler,
    ScaleMatchRMSNorm,
)
from geneuclidean.sampling.decoder_strategies import combine_cfg_logits  # noqa: E402
from geneuclidean.sampling.llm_sampler import (  # noqa: E402
    _build_initial_prompt_ids,
    sample_llm_molecules,
    sample_llm_with_attention,
)

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
HF_CACHE_ROOT = Path(DEFAULT_HF_CACHE_DIR)
V2_CKPT_PATH = Path("runs/geneuclidean_v2_painn/best.pt")
V2_CFG_PATH = Path("runs/geneuclidean_v2_painn/config.yaml")


def _has_cached_repo(repo_id: str) -> bool:
    slug = "models--" + repo_id.replace("/", "--")
    return (HF_CACHE_ROOT / "hub" / slug / "snapshots").is_dir()


def _random_so3_matrix(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a Haar-uniform proper rotation matrix R in SO(3) (det(R) = +1)."""
    mat = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    if float(torch.det(q).item()) < 0.0:
        q[:, 0] = -q[:, 0]
    return q.to(device=device, dtype=torch.float32)


def _make_pocket_batch(
    batch_size: int = 2,
    num_atoms: int = 28,
    node_in_dim: int = 52,
    device: torch.device | str = "cpu",
    seed: int = 77,
) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32) * 2.2
    node_feats = torch.randn(
        (batch_size, num_atoms, node_in_dim), generator=gen, dtype=torch.float32
    )
    atom_mask = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
    for b in range(batch_size):
        valid_count = max(8, num_atoms - 6 - 2 * b)
        atom_mask[b, :valid_count] = True
    dev = torch.device(device)
    return coords.to(dev), node_feats.to(dev), atom_mask.to(dev)


def _small_v2_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(max_neighbors=24),
        encoder=EncoderConfig(
            arch="painn",
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=3,
            num_rbf=16,
            cutoff=5.5,
            use_se3_chirality=True,
            dropout=0.0,
        ),
        cvae=CVAEConfig(enabled=True, latent_dim=16, num_latent_tokens=4),
        decoder=DecoderConfig(
            arch="transformer",
            vocab_size=96,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,
        ),
        multitask=MultiTaskConfig(
            use_pkd_head=True,
            use_clip_head=True,
            use_prop_head=True,
            clip_dim=32,
            ligand_fp_dim=128,
        ),
    )


def _build_mock_bridge(
    device: str = "cpu",
    use_lora: bool = True,
    freeze_pocket_encoder: bool = True,
) -> PocketConditionedLLM:
    torch.manual_seed(42)
    backbone = load_hf_backbone(
        MOCK_BACKBONE_ID,
        dtype="float32",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        use_lora=use_lora,
    )
    model = PocketConditionedLLM(
        v2_cfg=_small_v2_cfg(),
        backbone=backbone,
        num_queries=8,
        perceiver_layers=2,
        perceiver_heads=4,
        num_global_tokens=4,
        cfg_drop_prob=0.2,
        freeze_pocket_encoder=freeze_pocket_encoder,
    )
    return model.to(torch.device(device)).eval()


@pytest.mark.parametrize("init_rms", [1.0002, 0.1207, 0.0156])
def test_scale_match_rmsnorm_initialization_and_scale(init_rms: float) -> None:
    """ScaleMatchRMSNorm initializes weight to init_rms and outputs vectors with matching RMS."""
    norm = ScaleMatchRMSNorm(dim=128, init_rms=init_rms)
    assert torch.allclose(norm.weight, torch.full((128,), init_rms, dtype=torch.float32))

    x = torch.randn(4, 16, 128, dtype=torch.float32) * 7.5 + 3.0
    y = norm(x)
    empirical_rms = torch.sqrt(y.pow(2).mean(dim=-1))
    assert torch.allclose(empirical_rms, torch.full_like(empirical_rms, init_rms), atol=1e-4)

    # Verify bfloat16 dtype preservation
    y_bf16 = norm(x.to(dtype=torch.bfloat16))
    assert y_bf16.dtype == torch.bfloat16
    assert torch.isfinite(y_bf16).all()


@pytest.mark.parametrize("device", DEVICES)
def test_perceiver_zero_mask_attention_and_permutation_invariance(device: str) -> None:
    """PocketPerceiverResampler assigns exact 0.0 weight on dummy atoms and is perm-invariant."""
    torch.manual_seed(7)
    dev = torch.device(device)
    perceiver = (
        PocketPerceiverResampler(
            pocket_dim=64,
            llm_dim=64,
            num_queries=8,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )

    node_emb = torch.randn(2, 24, 64, device=dev, dtype=torch.float32)
    atom_mask = torch.zeros(2, 24, device=dev, dtype=torch.bool)
    atom_mask[0, :15] = True
    atom_mask[1, :11] = True

    with torch.no_grad():
        ref_tokens, ref_attn = perceiver(node_emb, atom_mask)

    # Exact 0.0 attention weight on every masked dummy atom
    assert ref_attn.shape == (2, 4, 8, 24)
    dummy_weights = ref_attn.masked_select(~atom_mask[:, None, None, :])
    assert float(dummy_weights.abs().max().item()) == 0.0
    assert torch.allclose(
        ref_attn.sum(dim=-1),
        torch.ones(2, 4, 8, device=dev),
        atol=1e-5,
    )

    # Extreme dummy corruption -> exact 0.0 sensitivity
    corrupted = node_emb.clone()
    corrupted[~atom_mask] = 1e6
    with torch.no_grad():
        corrupt_tokens, corrupt_attn = perceiver(corrupted, atom_mask)
    assert float((corrupt_tokens - ref_tokens).abs().max().item()) == 0.0
    assert float((corrupt_attn - ref_attn).abs().max().item()) == 0.0

    # Random atom permutation -> exact invariance of query tokens & equivariance of attention
    perm = torch.randperm(24, device=dev)
    with torch.no_grad():
        perm_tokens, perm_attn = perceiver(node_emb[:, perm, :], atom_mask[:, perm])
    assert float((perm_tokens - ref_tokens).abs().max().item()) < 1e-5
    assert float((perm_attn - ref_attn[..., perm]).abs().max().item()) < 1e-5

    # All-masked row edge case must stay finite and exact 0.0
    empty_mask = torch.zeros(1, 10, device=dev, dtype=torch.bool)
    empty_nodes = torch.randn(1, 10, 64, device=dev, dtype=torch.float32)
    with torch.no_grad():
        empty_tokens, empty_attn = perceiver(empty_nodes, empty_mask)
    assert torch.isfinite(empty_tokens).all()
    assert float(empty_attn.abs().max().item()) == 0.0


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_pocket_conditioned_llm_se3_translation_permutation_dummy_invariance(device: str) -> None:
    """Verify <1e-5 SO(3), R^3 translation, permutation, and dummy-atom invariance end-to-end."""
    dev = torch.device(device)
    model = _build_mock_bridge(device=device)
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=26, device=dev)
    tok_batch = model.tokenizer(["CCO", "c1ccccc1"], padding=True, return_tensors="pt")
    input_ids = tok_batch["input_ids"].to(dev)
    attention_mask = tok_batch["attention_mask"].to(dev)

    with torch.no_grad():
        ref_pocket = model.encode_pocket(coords, node_feats, atom_mask)
        ref_prefix, ref_attn, _ = model.build_soft_prefix(
            ref_pocket, atom_mask, deterministic_z=True, return_attn=True
        )
        ref_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords,
            node_feats=node_feats,
            atom_mask=atom_mask,
            return_attn=True,
        )
        cached_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pocket_out=ref_pocket,
            atom_mask=atom_mask,
        )

    assert ref_attn is not None
    assert float(ref_attn.masked_select(~atom_mask[:, None, None, :]).abs().max().item()) == 0.0
    # Cached pocket_out vs live encoder matches bit-exact on CPU and <1e-5 (scatter ULP) on MPS
    tol_cached = 0.0 if device == "cpu" else 1e-5
    assert float((cached_out.logits - ref_out.logits).abs().max().item()) <= tol_cached

    # 1. Pure SO(3) rotation (<1e-5) + rigid R^3 translation
    gen = torch.Generator(device="cpu").manual_seed(2026)
    rot = _random_so3_matrix(gen, dev)
    trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 100.0 - 50.0).to(dev)

    with torch.no_grad():
        rot_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords @ rot.T,
            node_feats=node_feats,
            atom_mask=atom_mask,
        )
        rot_prefix, _, _ = model.build_soft_prefix(
            rot_out.pocket_out, atom_mask, deterministic_z=True
        )
        tr_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=(coords @ rot.T) + trans,
            node_feats=node_feats,
            atom_mask=atom_mask,
        )
        tr_prefix, _, _ = model.build_soft_prefix(
            tr_out.pocket_out, atom_mask, deterministic_z=True
        )

    tol_rot = 1e-5 if device == "cpu" else 5e-5
    tol_trans = 1e-4
    assert float((rot_prefix - ref_prefix).abs().max().item()) < tol_rot
    assert float((rot_out.logits - ref_out.logits).abs().max().item()) < tol_rot
    assert float((tr_prefix - ref_prefix).abs().max().item()) < tol_trans
    assert float((tr_out.logits - ref_out.logits).abs().max().item()) < tol_trans

    # 2. Atom permutation invariance
    perm = torch.randperm(coords.shape[1], device=dev)
    with torch.no_grad():
        perm_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords[:, perm, :],
            node_feats=node_feats[:, perm, :],
            atom_mask=atom_mask[:, perm],
        )
        perm_prefix, _, _ = model.build_soft_prefix(
            perm_out.pocket_out, atom_mask[:, perm], deterministic_z=True
        )
    assert float((perm_prefix - ref_prefix).abs().max().item()) < tol_rot
    assert float((perm_out.logits - ref_out.logits).abs().max().item()) < tol_rot

    # 3. Dummy-atom coordinate & feature corruption invariance
    bad_coords = coords.clone()
    bad_feats = node_feats.clone()
    bad_coords[~atom_mask] = 9999.0
    bad_feats[~atom_mask] = -9999.0
    with torch.no_grad():
        dum_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=bad_coords,
            node_feats=bad_feats,
            atom_mask=atom_mask,
        )
    assert float((dum_out.logits - ref_out.logits).abs().max().item()) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_kv_cache_exact_parity_cfg_shifts_and_sampling_smoke(device: str) -> None:
    """Verify <1e-4 KV-cache vs non-cached logit parity, CFG w=0/1/1.6 math, and sampler API."""
    dev = torch.device(device)
    model = _build_mock_bridge(device=device)
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=1, num_atoms=20, device=dev)

    with torch.no_grad():
        pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
        cond_prefix, _, _ = model.build_soft_prefix(pocket_out, atom_mask, deterministic_z=True)
        uncond_prefix, _, _ = model.build_soft_prefix(pocket_out, atom_mask, use_null_pocket=True)

        # Compare 6-step incremental KV-cached decoding against full non-cached forward
        prompt_ids = _build_initial_prompt_ids(model, batch_size=1, device=dev)
        prompt_embeds = get_input_embeddings_tensor(model.llm, prompt_ids)
        prefill_embeds = torch.cat([cond_prefix, prompt_embeds], dim=1)
        prefill_mask = torch.ones((1, prefill_embeds.shape[1]), dtype=torch.long, device=dev)

        cached_out = model.llm(
            inputs_embeds=prefill_embeds,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
        past = cached_out.past_key_values
        seq_ids = prompt_ids.clone()
        cur_logits = cached_out.logits[:, -1, :].float()

        max_kv_diff = 0.0
        for _ in range(5):
            next_tok = cur_logits.argmax(dim=-1, keepdim=True)
            seq_ids = torch.cat([seq_ids, next_tok], dim=1)

            # Non-cached full forward on [cond_prefix, seq_ids]
            full_tok_embeds = get_input_embeddings_tensor(model.llm, seq_ids)
            full_embeds = torch.cat([cond_prefix, full_tok_embeds], dim=1)
            full_mask = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=dev)
            full_logits = (
                model.llm(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    use_cache=False,
                    return_dict=True,
                )
                .logits[:, -1, :]
                .float()
            )

            # Incremental 1-token KV-cached step
            cached_out = model.llm(
                input_ids=next_tok,
                attention_mask=full_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = cached_out.past_key_values
            cur_logits = cached_out.logits[:, -1, :].float()
            max_kv_diff = max(max_kv_diff, float((cur_logits - full_logits).abs().max().item()))

        assert max_kv_diff < 1e-4, f"KV-cache parity drift too large: {max_kv_diff:.2e}"

        # CFG shift identity checks (w = 0.0, 1.0, 1.6)
        uncond_embeds = torch.cat([uncond_prefix, prompt_embeds], dim=1)
        cond_step0 = (
            model.llm(
                inputs_embeds=prefill_embeds,
                attention_mask=prefill_mask,
                use_cache=False,
                return_dict=True,
            )
            .logits[:, -1, :]
            .float()
        )
        uncond_step0 = (
            model.llm(
                inputs_embeds=uncond_embeds,
                attention_mask=prefill_mask,
                use_cache=False,
                return_dict=True,
            )
            .logits[:, -1, :]
            .float()
        )

        assert float((cond_step0 - uncond_step0).abs().max().item()) > 1e-4
        assert torch.allclose(
            combine_cfg_logits(cond_step0, uncond_step0, 0.0), uncond_step0, atol=1e-6
        )
        assert torch.allclose(
            combine_cfg_logits(cond_step0, uncond_step0, 1.0), cond_step0, atol=1e-6
        )
        expected_16 = uncond_step0 + 1.6 * (cond_step0 - uncond_step0)
        assert torch.allclose(
            combine_cfg_logits(cond_step0, uncond_step0, 1.6), expected_16, atol=1e-5
        )

        # Sampling smoke test across strategies, multi-chunk batching, and posterior refinement
        for strategy in ("min_p", "top_p", "top_k", "greedy"):
            smiles, attn = sample_llm_with_attention(
                model=model,
                pocket_out=pocket_out,
                atom_mask=atom_mask,
                num_samples=4,
                max_new_tokens=16,
                strategy=strategy,
                cfg_scale=1.6,
                latent_temp=0.5,
                batch_size=2,
                seed=123,
            )
            assert len(smiles) == 4
            assert all(isinstance(s, str) for s in smiles)
            assert attn.shape == (4, 4, 8, 20)
            assert float(attn.masked_select(~atom_mask[:, None, None, :]).abs().max().item()) == 0.0

        seed_fp = torch.rand(128, device=dev)
        refined = sample_llm_molecules(
            model=model,
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            num_samples=3,
            max_new_tokens=12,
            cfg_scale=1.0,
            seed_ligand_fp=seed_fp,
            seed=7,
        )
        assert len(refined) == 3

        # score_candidates parity with GenEuclideanModel API
        scores = model.score_candidates(pocket_out, torch.rand(5, 128, device=dev))
        assert set(scores.keys()) == {"pkd", "clip_sim"}
        assert scores["pkd"].shape == (5,)
        assert scores["clip_sim"].shape == (5,)


def test_loss_masking_training_grads_and_adapter_checkpoint_bit_exact(tmp_path: Path) -> None:
    """Verify pad/prefix loss masking, bridge+LoRA gradients, and bit-exact adapter save/load."""
    model = _build_mock_bridge(device="cpu", use_lora=True, freeze_pocket_encoder=True)
    model.train()
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=20, device="cpu")
    tok_batch = model.tokenizer(["CCO", "c1ccccc1"], padding=True, return_tensors="pt")
    input_ids = tok_batch["input_ids"]
    attention_mask = tok_batch["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    ligand_fp = torch.rand(2, 128)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        coords=coords,
        node_feats=node_feats,
        atom_mask=atom_mask,
        ligand_fp=ligand_fp,
        cfg_drop=True,
        use_posterior=True,
    )
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.logits.shape == (2, input_ids.shape[1], model.vocab_size)
    out.loss.backward()

    # Frozen v2 pocket encoder/cvae/heads must NOT accumulate gradients
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.heads.parameters())
    # Bridge + LoRA MUST receive gradients
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.perceiver.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.lora_parameters()
    )

    # Adapter checkpoint save/load must be compact and bit-exact
    model.eval()
    ckpt_path = tmp_path / "adapter.pt"
    model.save_adapter_checkpoint(ckpt_path, extra_meta={"epoch": 3})
    saved_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "bridge_state_dict" in saved_obj
    assert "lora_state_dict" in saved_obj
    assert len(saved_obj["lora_state_dict"]) > 0
    assert saved_obj["metadata"]["epoch"] == 3
    assert ckpt_path.stat().st_size < 2 * 1024 * 1024  # tiny mock adapter << 2 MB

    with torch.no_grad():
        ref_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords,
            node_feats=node_feats,
            atom_mask=atom_mask,
        ).logits.clone()

    # Perturb bridge and LoRA parameters, then reload
    with torch.no_grad():
        for p in model.bridge_parameters():
            p.add_(1.5)
        for p in model.lora_parameters():
            p.add_(0.75)
        perturbed_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords,
            node_feats=node_feats,
            atom_mask=atom_mask,
        ).logits
    assert float((perturbed_logits - ref_logits).abs().max().item()) > 1e-2

    meta = model.load_adapter_checkpoint(ckpt_path)
    assert meta["backbone_id"] == MOCK_BACKBONE_ID
    with torch.no_grad():
        reloaded_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords,
            node_feats=node_feats,
            atom_mask=atom_mask,
        ).logits
    assert float((reloaded_logits - ref_logits).abs().max().item()) == 0.0


def test_mps_forbids_float16_backbone() -> None:
    """Reject float16 on Apple Silicon MPS to prevent NaN/Inf LoRA gradients."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    with pytest.raises(ValueError, match="float16"):
        load_hf_backbone(MOCK_BACKBONE_ID, dtype="float16")


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_cached_repo(MOLFORMER_REPO_ID),
    reason=f"{MOLFORMER_REPO_ID} not present in {DEFAULT_HF_CACHE_DIR}",
)
def test_live_gpmolformer_v2_warmstart_equivariance_and_kv_cache() -> None:
    """Verify live GP-MoLFormer-Uniq + v2 PaiNN-SE3 warm-start, SE(3) invariance, and KV-cache."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = load_hf_backbone(
        MOLFORMER_REPO_ID,
        cache_dir=DEFAULT_HF_CACHE_DIR,
        dtype="float32",
        lora_r=16,
        lora_alpha=32,
        use_lora=True,
        local_files_only=True,
    )
    assert backbone.hidden_size == 768
    assert backbone.vocab_size == 2362
    assert backbone.bos_token_id == 0
    assert backbone.eos_token_id == 1
    assert backbone.pad_token_id == 2
    assert backbone.is_smiles_native is True
    assert 0.90 <= backbone.embed_rms <= 1.15

    v2_cfg = ExperimentConfig.from_yaml(V2_CFG_PATH) if V2_CFG_PATH.is_file() else _small_v2_cfg()
    model = PocketConditionedLLM(
        v2_cfg=v2_cfg,
        backbone=backbone,
        num_queries=16,
        perceiver_layers=2,
        perceiver_heads=8,
        num_global_tokens=4,
        freeze_pocket_encoder=True,
    )
    if V2_CKPT_PATH.is_file() and V2_CFG_PATH.is_file():
        info = model.load_v2_pocket_checkpoint(V2_CKPT_PATH)
        assert info["encoder_tensors"] == 95
        assert info["cvae_tensors"] == 19
        assert info["heads_tensors"] == 29

    model = model.to(device).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=1, num_atoms=24, device=device)
    tok_batch = backbone.tokenizer(["CC(=O)Oc1ccccc1C(=O)O"], padding=True, return_tensors="pt")
    input_ids = tok_batch["input_ids"].to(device)
    attention_mask = tok_batch["attention_mask"].to(device)

    with torch.no_grad():
        ref_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords,
            node_feats=node_feats,
            atom_mask=atom_mask,
            return_attn=True,
        )
        assert torch.isfinite(ref_out.logits).all()

        # SO(3) rotation invariance check on live GP-MoLFormer-Uniq
        rot = _random_so3_matrix(torch.Generator(device="cpu").manual_seed(99), device)
        rot_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords @ rot.T,
            node_feats=node_feats,
            atom_mask=atom_mask,
        )
        rot_diff = float((rot_out.logits - ref_out.logits).abs().max().item())
        assert rot_diff < 5e-4, f"Live GP-MoLFormer SO(3) logit diff={rot_diff:.2e}"

        # Recurrent tuple KV-cache vs non-cached forward logit parity
        assert ref_out.pocket_out is not None
        cond_prefix, _, _ = model.build_soft_prefix(
            ref_out.pocket_out, atom_mask, deterministic_z=True
        )
        prompt_ids = _build_initial_prompt_ids(model, batch_size=1, device=device)
        prompt_embeds = get_input_embeddings_tensor(model.llm, prompt_ids)
        prefill_embeds = torch.cat([cond_prefix, prompt_embeds], dim=1)
        prefill_mask = torch.ones((1, prefill_embeds.shape[1]), dtype=torch.long, device=device)

        cached = model.llm(
            inputs_embeds=prefill_embeds,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
        past = cached.past_key_values
        seq_ids = prompt_ids.clone()
        cur_logits = cached.logits[:, -1, :].float()
        max_kv_diff = 0.0
        for _ in range(5):
            next_tok = cur_logits.argmax(dim=-1, keepdim=True)
            seq_ids = torch.cat([seq_ids, next_tok], dim=1)
            full_embeds = torch.cat(
                [cond_prefix, get_input_embeddings_tensor(model.llm, seq_ids)], dim=1
            )
            full_mask = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=device)
            full_logits = (
                model.llm(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    use_cache=False,
                    return_dict=True,
                )
                .logits[:, -1, :]
                .float()
            )
            cached = model.llm(
                input_ids=next_tok,
                attention_mask=full_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = cached.past_key_values
            cur_logits = cached.logits[:, -1, :].float()
            max_kv_diff = max(max_kv_diff, float((cur_logits - full_logits).abs().max().item()))

        assert max_kv_diff < 5e-4, f"GP-MoLFormer KV-cache diff={max_kv_diff:.2e}"

        smiles, attn = sample_llm_with_attention(
            model=model,
            pocket_out=ref_out.pocket_out,
            atom_mask=atom_mask,
            num_samples=4,
            max_new_tokens=32,
            strategy="min_p",
            cfg_scale=1.6,
            seed=42,
        )
        assert len(smiles) == 4
        assert attn.shape == (4, 8, 16, 24)
        assert float(attn.masked_select(~atom_mask[:, None, None, :]).abs().max().item()) == 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_cached_repo("entropy/gpt2_zinc_87m"),
    reason=f"entropy/gpt2_zinc_87m not present in {DEFAULT_HF_CACHE_DIR}",
)
def test_live_gpt2_zinc_87m_bridge_and_kv_sampler() -> None:
    """Verify native GPT-2 ZINC 87M backbone loading, forward pass, and KV-cached CFG sampling."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = load_hf_backbone(
        "entropy/gpt2_zinc_87m",
        cache_dir=DEFAULT_HF_CACHE_DIR,
        dtype="float32",
        lora_r=8,
        lora_alpha=16,
        use_lora=True,
        local_files_only=True,
    )
    assert backbone.hidden_size == 768
    assert backbone.is_smiles_native is True
    assert backbone.model_dtype == torch.float32

    model = (
        PocketConditionedLLM(
            v2_cfg=_small_v2_cfg(),
            backbone=backbone,
            num_queries=16,
            perceiver_layers=2,
            perceiver_heads=8,
            num_global_tokens=4,
        )
        .to(device)
        .eval()
    )
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=1, num_atoms=20, device=device)
    with torch.no_grad():
        pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
        smiles, attn = sample_llm_with_attention(
            model=model,
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            num_samples=4,
            max_new_tokens=24,
            strategy="min_p",
            cfg_scale=1.6,
            seed=42,
        )
    assert len(smiles) == 4
    assert attn.shape == (4, 8, 16, 20)
    assert float(attn.masked_select(~atom_mask[:, None, None, :]).abs().max().item()) == 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_cached_repo("Qwen/Qwen2.5-0.5B"),
    reason=f"Qwen/Qwen2.5-0.5B not present in {DEFAULT_HF_CACHE_DIR}",
)
def test_live_qwen25_bfloat16_bridge_and_kv_sampler() -> None:
    """Verify Qwen2.5-0.5B bfloat16 loading, prompt prefix handling, and KV-cached CFG sampling."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = load_hf_backbone(
        "Qwen/Qwen2.5-0.5B",
        cache_dir=DEFAULT_HF_CACHE_DIR,
        dtype="auto",
        lora_r=8,
        lora_alpha=16,
        use_lora=True,
        local_files_only=True,
    )
    assert backbone.hidden_size == 896
    assert backbone.is_smiles_native is False
    assert backbone.prompt_prefix == "SMILES: "
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        assert backbone.model_dtype == torch.bfloat16
    assert 0.005 <= backbone.embed_rms <= 0.05

    model = (
        PocketConditionedLLM(
            v2_cfg=_small_v2_cfg(),
            backbone=backbone,
            num_queries=16,
            perceiver_layers=2,
            perceiver_heads=8,
            num_global_tokens=4,
        )
        .to(device)
        .eval()
    )
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=1, num_atoms=20, device=device)
    tok = backbone.tokenizer(["SMILES: CCO\n"], padding=True, return_tensors="pt")
    with torch.no_grad():
        out = model(
            input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
            labels=tok["input_ids"].to(device),
            coords=coords,
            node_feats=node_feats,
            atom_mask=atom_mask,
        )
        assert out.loss is not None and torch.isfinite(out.loss)
        assert out.pocket_out is not None
        smiles = sample_llm_molecules(
            model=model,
            pocket_out=out.pocket_out,
            atom_mask=atom_mask,
            num_samples=2,
            max_new_tokens=16,
            strategy="min_p",
            cfg_scale=1.6,
            seed=42,
        )
    assert len(smiles) == 2
    assert all(isinstance(s, str) for s in smiles)
