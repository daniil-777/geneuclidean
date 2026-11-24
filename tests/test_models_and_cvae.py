"""Unit tests for decoders, masked cross-attention, PocketCVAE, CFG, heads, and MPS grads."""

from __future__ import annotations

import math
from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from geneuclidean.config import (
    CVAEConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
)
from geneuclidean.models import (
    GenEuclideanModel,
    PocketCVAE,
    PocketLSTMDecoder,
    PocketTransformerDecoder,
    build_encoder,
)

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def _make_cfg(
    encoder_arch: Literal["painn", "egnn", "e3nn", "schnet"] = "painn",
    decoder_arch: Literal["transformer", "lstm"] = "transformer",
    out_dim: int = 64,
    d_model: int = 64,
    cvae_enabled: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        encoder=EncoderConfig(
            arch=encoder_arch,
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=out_dim,
            num_layers=2,
            num_rbf=12,
            cutoff=5.5,
            e3nn_lmax=1,
            use_se3_chirality=True,
            dropout=0.0,
        ),
        cvae=CVAEConfig(enabled=cvae_enabled, latent_dim=16, num_latent_tokens=4),
        decoder=DecoderConfig(
            arch=decoder_arch,
            vocab_size=48,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            d_ff=128,
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
            clip_dim=32,
            ligand_fp_dim=128,
        ),
    )


def test_stage1_unconditional_forward_and_zero_init_backbone_transfer() -> None:
    """Stage-1 unconditional LM transfers cleanly into Stage-2 with 0.0 Step-0 drift at gate=0."""
    torch.manual_seed(0)
    stage1_cfg = DecoderConfig(
        arch="transformer",
        vocab_size=48,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        use_cross_attn=False,
    )
    stage1_dec = PocketTransformerDecoder(stage1_cfg).eval()
    input_ids = torch.randint(1, 48, (2, 12))

    with torch.no_grad():
        stage1_logits, stage1_attn = stage1_dec(input_ids, memory=None, return_attn=True)
    assert stage1_logits.shape == (2, 12, 48)
    assert stage1_attn is None

    # Transfer into Stage-2 decoder with exact zero-init Flamingo gates (gate_init=0.0).
    stage2_dec = PocketTransformerDecoder(
        vocab_size=48,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        use_cross_attn=True,
        gated_cross_attn=True,
        cross_attn_gate_init=0.0,
    ).eval()

    missing, unexpected = stage2_dec.load_pretrained_backbone(
        {"model_state_dict": {f"decoder.{k}": v for k, v in stage1_dec.state_dict().items()}}
    )
    assert len(unexpected) == 0
    assert any("cross_attn" in k for k in missing)

    memory = torch.randn(2, 18, 64)
    memory_mask = torch.ones(2, 18, dtype=torch.bool)
    with torch.no_grad():
        stage2_logits, stage2_attn = stage2_dec(
            input_ids, memory=memory, memory_mask=memory_mask, return_attn=True
        )
    assert stage2_attn is not None
    assert (stage2_logits - stage1_logits).abs().max().item() == 0.0

    # GenEuclideanModel also supports Stage-1 unconditional forward when coords are absent.
    model = GenEuclideanModel(_make_cfg()).eval()
    with torch.no_grad():
        out = model({"input_ids": input_ids})
    assert out.logits.shape == (2, 12, 48)
    assert out.pocket_out is None
    assert out.mu_p is None


@pytest.mark.parametrize("decoder_arch", ["transformer", "lstm"])
@pytest.mark.parametrize("device", DEVICES)
def test_decoders_zero_leak_on_padded_atoms_and_strict_causality(
    decoder_arch: Literal["transformer", "lstm"],
    device: str,
) -> None:
    """Padded memory positions get exact 0.0 attention weight and zero influence on logits."""
    torch.manual_seed(5)
    dev = torch.device(device)
    dec_cfg = DecoderConfig(
        arch=decoder_arch,
        vocab_size=48,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        use_cross_attn=True,
        gated_cross_attn=True,
    )
    decoder: PocketTransformerDecoder | PocketLSTMDecoder = (
        (
            PocketTransformerDecoder(dec_cfg)
            if decoder_arch == "transformer"
            else PocketLSTMDecoder(dec_cfg)
        )
        .to(dev)
        .eval()
    )

    input_ids = torch.randint(1, 48, (2, 10), device=dev)
    memory = torch.randn((2, 20, 64), device=dev)
    memory_mask = torch.zeros((2, 20), dtype=torch.bool, device=dev)
    memory_mask[0, :12] = True
    memory_mask[1, :7] = True

    memory_corrupt = memory.clone()
    memory_corrupt[~memory_mask] = 1e5

    with torch.no_grad():
        logits_a, attn_a = decoder(
            input_ids, memory=memory, memory_mask=memory_mask, return_attn=True
        )
        logits_b, attn_b = decoder(
            input_ids, memory=memory_corrupt, memory_mask=memory_mask, return_attn=True
        )

    assert attn_a is not None and attn_b is not None
    # Exact 0.0 weight on all padded pocket atoms!
    pad_mask_4d = ~memory_mask[:, None, None, :].expand_as(attn_a)
    assert attn_a[pad_mask_4d].abs().max().item() == 0.0
    assert attn_b[pad_mask_4d].abs().max().item() == 0.0
    # Valid weights sum to 1.0 at every decoding step and head.
    assert torch.allclose(attn_a.sum(dim=-1), torch.ones_like(attn_a.sum(dim=-1)), atol=1e-5)
    # Corrupted padded memory has exact 0.0 effect on logits.
    assert (logits_a - logits_b).abs().max().item() == 0.0

    # Strict autoregressive causality: changing future token t=6 never affects logits[:, :6].
    input_future = input_ids.clone()
    input_future[:, 6:] = (input_future[:, 6:] + 7) % 47 + 1
    with torch.no_grad():
        logits_future, _ = decoder(input_future, memory=memory, memory_mask=memory_mask)
    assert (logits_a[:, :6, :] - logits_future[:, :6, :]).abs().max().item() == 0.0

    # All-False mask edge case produces finite outputs (no NaN from softmax(-inf)).
    all_false = torch.zeros_like(memory_mask)
    with torch.no_grad():
        logits_empty, attn_empty = decoder(
            input_ids, memory=memory, memory_mask=all_false, return_attn=True
        )
    assert torch.isfinite(logits_empty).all()
    assert attn_empty is not None and attn_empty.abs().max().item() == 0.0


def test_cvae_prior_posterior_refine_and_fp_bridge() -> None:
    """Verify PocketCVAE prior sampling, posterior inference, and seed-ligand refinement."""
    torch.manual_seed(9)
    cvae = PocketCVAE(pocket_dim=64, d_model=64, latent_dim=16, num_latent_tokens=4, dropout=0.0)
    cvae.eval()

    pooled = torch.randn(3, 64)
    lig_emb = torch.randn(3, 64)

    out_prior = cvae(pooled, ligand_emb=None, latent_temp=1.0)
    assert out_prior.z_tokens.shape == (3, 4, 64)
    assert out_prior.mu_q is None and out_prior.logvar_q is None
    assert out_prior.mu_p.shape == (3, 16)

    out_post = cvae(pooled, ligand_emb=lig_emb, latent_temp=0.0)
    assert out_post.mu_q is not None and out_post.logvar_q is not None
    assert torch.allclose(out_post.z, out_post.mu_q)

    # Stochastic prior sampling produces distinct latent tokens across draws.
    z1 = cvae.sample_prior(pooled, latent_temp=1.0)
    z2 = cvae.sample_prior(pooled, latent_temp=1.0)
    assert z1.shape == (3, 4, 64)
    assert (z1 - z2).abs().max().item() > 1e-3

    # Deterministic prior mean when latent_temp=0.0.
    z_det1 = cvae.sample_prior(pooled, latent_temp=0.0)
    z_det2 = cvae.sample_prior(pooled, latent_temp=0.0)
    assert (z_det1 - z_det2).abs().max().item() == 0.0

    # Standalone refine with ligand embedding AND integrated GenEuclideanModel refine with raw FP.
    z_ref_emb = cvae.refine(pooled, lig_emb, latent_temp=0.0)
    assert torch.allclose(z_ref_emb, out_post.z_tokens)

    model = GenEuclideanModel(_make_cfg()).eval()
    assert model.cvae is not None
    raw_fp = (torch.rand(3, 128) > 0.75).float()
    z_ref_fp = model.cvae.refine(pooled, raw_fp, latent_temp=0.5)
    assert z_ref_fp.shape == (3, 4, 64)
    assert torch.isfinite(z_ref_fp).all()


def test_cfg_null_pocket_switch_and_training_dropout() -> None:
    """Verify CFG null_pocket_token switch and training cfg_drop equivalence."""
    torch.manual_seed(13)
    cfg = _make_cfg()
    cfg.decoder.cfg_drop_prob = 0.999999
    model = GenEuclideanModel(cfg)

    coords = torch.randn(2, 20, 3) * 2.0
    node_feats = torch.randn(2, 20, 52)
    atom_mask = torch.ones(2, 20, dtype=torch.bool)
    atom_mask[1, 14:] = False
    input_ids = torch.randint(1, 48, (2, 10))
    ligand_fp = (torch.rand(2, 128) > 0.7).float()

    model.eval()
    with torch.no_grad():
        pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
        mem_cond, mask_cond = model.build_memory(pocket_out, atom_mask, use_null_pocket=False)
        mem_null, mask_null = model.build_memory(pocket_out, atom_mask, use_null_pocket=True)

        assert mem_null.shape == (2, 1, 64)
        assert bool(mask_null.all().item())

        logits_cond, _ = model.decoder(input_ids, memory=mem_cond, memory_mask=mask_cond)
        logits_null, _ = model.decoder(input_ids, memory=mem_null, memory_mask=mask_null)
        assert (logits_cond - logits_null).abs().max().item() > 1e-3

    # In training mode with cfg_drop_prob ~ 1.0, cfg_drop=True matches null-pocket logits.
    model.train()
    batch = {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "input_ids": input_ids,
        "ligand_fp": ligand_fp,
    }
    with torch.no_grad():
        out_dropped = model(batch, cfg_drop=True)
    assert (out_dropped.logits - logits_null).abs().max().item() < 1e-5


def test_lstm_v15_no_cross_batch_leakage_and_variable_pocket_sizes() -> None:
    """Verify GenEuclidean-v1.5 LSTM handles arbitrary pocket sizes with zero batch leakage."""
    torch.manual_seed(21)
    cfg = _make_cfg(
        encoder_arch="e3nn",
        decoder_arch="lstm",
        out_dim=48,
        d_model=64,
        cvae_enabled=False,
    )
    model = GenEuclideanModel(cfg).eval()

    # Non-square arbitrary pocket atom count (N=37, no hardcoded 17x17).
    coords = torch.randn(2, 37, 3) * 2.5
    node_feats = torch.randn(2, 37, 52)
    atom_mask = torch.ones(2, 37, dtype=torch.bool)
    atom_mask[0, 25:] = False
    atom_mask[1, 31:] = False
    input_ids = torch.randint(1, 48, (2, 15))

    batch_a = {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "input_ids": input_ids,
    }
    batch_b = {
        "coords": coords.clone(),
        "node_feats": node_feats.clone(),
        "atom_mask": atom_mask.clone(),
        "input_ids": input_ids.clone(),
    }
    # Radically alter batch item 1; batch item 0 must be completely unaffected.
    batch_b["coords"][1] = torch.randn(37, 3) * 5.0
    batch_b["node_feats"][1] = torch.randn(37, 52) * 10.0
    batch_b["input_ids"][1] = torch.randint(1, 48, (15,))

    with torch.no_grad():
        out_a = model(batch_a, return_attn=True)
        out_b = model(batch_b, return_attn=True)

    assert out_a.logits.shape == (2, 15, 48)
    assert out_a.attn_weights is not None and out_a.attn_weights.shape == (2, 1, 15, 37)
    assert (out_a.logits[0] - out_b.logits[0]).abs().max().item() < 1e-6


def test_score_candidates_and_clip_scale_clamping() -> None:
    """Verify microsecond candidate scoring (pKd + DrugCLIP) and logit_scale clamping."""
    torch.manual_seed(33)
    model = GenEuclideanModel(_make_cfg()).eval()
    coords = torch.randn(1, 20, 3) * 2.0
    node_feats = torch.randn(1, 20, 52)
    atom_mask = torch.ones(1, 20, dtype=torch.bool)
    candidate_fps = (torch.rand(12, 128) > 0.7).float()

    # Force raw logit_scale parameter above log(100) and verify clamping.
    assert model.heads.logit_scale is not None
    with torch.no_grad():
        model.heads.logit_scale.fill_(10.0)
        pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
        scores = model.score_candidates(pocket_out, candidate_fps)
        mt = model.heads(pocket_out.pooled_emb, model.heads.encode_ligand(candidate_fps[:1]))

    assert mt.clip_logit_scale is not None
    assert float(mt.clip_logit_scale.item()) <= math.log(100.0) + 1e-6
    assert "pkd" in scores and scores["pkd"].shape == (12,)
    assert "clip_sim" in scores and scores["clip_sim"].shape == (12,)
    assert torch.isfinite(scores["pkd"]).all()
    assert float(scores["clip_sim"].abs().max().item()) <= 1.0 + 1e-5


@pytest.mark.parametrize("arch", ["painn", "egnn", "e3nn", "schnet"])
@pytest.mark.parametrize("device", DEVICES)
def test_end_to_end_backward_gradient_flow(
    arch: Literal["painn", "egnn", "e3nn", "schnet"],
    device: str,
) -> None:
    """Verify full multi-task forward + backward gradient flow on CPU and MPS (zero float64)."""
    torch.manual_seed(42)
    dev = torch.device(device)
    cfg = _make_cfg(encoder_arch=arch, decoder_arch="transformer")
    model = GenEuclideanModel(cfg).to(dev).train()

    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"Non-float32 param {name}: {param.dtype}"
    for name, buf in model.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer {name} breaks MPS!"

    coords = torch.randn((2, 24, 3), device=dev) * 2.0
    node_feats = torch.randn((2, 24, 52), device=dev)
    atom_mask = torch.ones((2, 24), dtype=torch.bool, device=dev)
    atom_mask[0, 18:] = False
    atom_mask[1, 20:] = False
    input_ids = torch.randint(1, 48, (2, 12), device=dev)
    target_ids = torch.randint(1, 48, (2, 12), device=dev)
    ligand_fp = (torch.rand((2, 128), device=dev) > 0.7).float()

    batch = {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "input_ids": input_ids,
        "ligand_fp": ligand_fp,
    }
    out = model(batch, cfg_drop=False, return_attn=False)
    lm_loss = F.cross_entropy(out.logits.reshape(-1, 48), target_ids.reshape(-1))
    assert out.mu_q is not None and out.logvar_q is not None
    assert out.mu_p is not None and out.logvar_p is not None
    kl_loss = (
        0.5
        * (
            out.logvar_p
            - out.logvar_q
            + (out.logvar_q.exp() + (out.mu_q - out.mu_p).pow(2)) / out.logvar_p.exp()
            - 1.0
        )
        .sum(dim=-1)
        .mean()
    )
    assert out.pred_pkd is not None and out.pred_props is not None
    assert out.clip_pocket is not None and out.clip_ligand is not None
    assert out.clip_logit_scale is not None
    pkd_loss = F.huber_loss(out.pred_pkd, torch.tensor([6.5, 8.2], device=dev))
    prop_loss = F.mse_loss(out.pred_props, torch.zeros_like(out.pred_props))
    sim = out.clip_logit_scale.exp() * (out.clip_pocket @ out.clip_ligand.T)
    clip_loss = F.cross_entropy(sim, torch.arange(2, device=dev))

    total_loss = lm_loss + 0.1 * kl_loss + 0.5 * pkd_loss + 0.3 * clip_loss + 0.2 * prop_loss
    total_loss.backward()

    # Check every major subsystem received finite, non-zero gradients.
    for subname, submod in [
        ("encoder", model.encoder),
        ("cvae", model.cvae),
        ("decoder", model.decoder),
        ("heads", model.heads),
    ]:
        assert submod is not None
        grads = [p.grad for p in submod.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, f"{arch} ({device}) {subname} received no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"Non-finite grad in {subname}"
        max_grad = max(float(g.abs().max().item()) for g in grads)
        assert max_grad > 0.0, f"{arch} ({device}) {subname} gradients are all zero"


@pytest.mark.parametrize("arch", ["painn", "egnn", "e3nn", "schnet"])
def test_empty_edge_graph_single_atom_edge_case(
    arch: Literal["painn", "egnn", "e3nn", "schnet"],
) -> None:
    """Encoders handle pockets with 0 edges (single atom or widely separated atoms) safely."""
    enc = build_encoder(
        EncoderConfig(
            arch=arch,
            node_in_dim=52,
            hidden_dim=32,
            vector_dim=8,
            out_dim=32,
            num_layers=2,
            num_rbf=8,
            cutoff=5.5,
            e3nn_lmax=1,
            dropout=0.0,
        )
    ).eval()

    # Atoms placed 50 Å apart -> 0 edges within 5.5 Å cutoff.
    coords = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1).expand(2, 3, 3) * 50.0
    feats = torch.randn(2, 3, 52)
    mask = torch.tensor([[True, False, False], [True, True, False]])

    with torch.no_grad():
        out = enc(coords, feats, mask)
    assert torch.isfinite(out.node_emb).all()
    assert torch.isfinite(out.pooled_emb).all()
    assert out.node_emb[~mask].abs().max().item() == 0.0
    if out.vec_emb is not None:
        assert torch.isfinite(out.vec_emb).all()
        assert out.vec_emb[~mask].abs().max().item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS-only regression test for SDPA dropout with frozen decoder self-attention",
)
def test_mps_frozen_decoder_self_attention_dropout_regression() -> None:
    """Verify MPS forward+backward succeeds with dropout>0 and frozen decoder backbone."""
    torch.manual_seed(42)
    dev = torch.device("mps")
    cfg = _make_cfg(encoder_arch="painn", decoder_arch="transformer")
    cfg.decoder.dropout = 0.1
    model = GenEuclideanModel(cfg).to(dev).train()

    # Freeze decoder self-attention/embedding/FFN backbone exactly like freeze_decoder_epochs.
    for name, param in model.named_parameters():
        if name.startswith("decoder.") and "cross_" not in name.lower():
            param.requires_grad = False

    coords = torch.randn((2, 16, 3), device=dev, dtype=torch.float32) * 2.0
    node_feats = torch.randn((2, 16, 52), device=dev, dtype=torch.float32)
    atom_mask = torch.ones((2, 16), dtype=torch.bool, device=dev)
    input_ids = torch.randint(1, 48, (2, 10), device=dev)
    target_ids = torch.randint(1, 48, (2, 10), device=dev)
    ligand_fp = (torch.rand((2, 128), device=dev) > 0.7).float()

    out = model(
        {
            "coords": coords,
            "node_feats": node_feats,
            "atom_mask": atom_mask,
            "input_ids": input_ids,
            "ligand_fp": ligand_fp,
        },
        cfg_drop=True,
    )
    assert torch.isfinite(out.logits).all()

    loss = F.cross_entropy(out.logits.reshape(-1, 48), target_ids.reshape(-1))
    loss.backward()

    # Frozen token embedding receives no grad; trainable encoder & cross-attn receive finite grads.
    assert model.decoder.token_emb.weight.grad is None
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0
    assert all(torch.isfinite(g).all() for g in enc_grads)
    cross_grads = [
        p.grad
        for n, p in model.decoder.named_parameters()
        if "cross_" in n.lower() and p.grad is not None
    ]
    assert len(cross_grads) > 0
    assert all(torch.isfinite(g).all() for g in cross_grads)
