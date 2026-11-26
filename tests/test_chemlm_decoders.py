"""Tests for the scalable pocket-conditioned chemical LLM decoder suite (``chemlm``).

Covers scale presets, Fourier property conditioning, sparse Mixture-of-Experts routing,
QK-norm attention, gradient checkpointing, KV-cached incremental decoding parity, exact
zero-leak padded-pocket masking, and ``MolecularDecoderProtocol`` conformance.
"""

from __future__ import annotations

import math

import pytest
import torch

from geneuclidean.config import DecoderConfig
from geneuclidean.models.decoders import build_decoder, list_decoders
from geneuclidean.models.decoders.chemlm import (
    SCALE_PRESETS,
    ChemLMDecoder,
    DecoderScale,
    FourierPropertyEmbedding,
    PropertyConditioner,
    SparseMoEFeedForward,
    build_chemlm_decoder,
    build_chemlm_moe_decoder,
    cross_attention_gate_magnitudes,
    pocket_reliance_report,
    resolve_scale,
    scaled_decoder_config,
)
from geneuclidean.protocols import MolecularDecoderProtocol

VOCAB = 48
PAD_ID = 0


def _tiny_decoder(**overrides: object) -> ChemLMDecoder:
    kwargs: dict[str, object] = {
        "vocab_size": VOCAB,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 64,
        "max_seq_len": 32,
        "dropout": 0.0,
        "pad_id": PAD_ID,
    }
    kwargs.update(overrides)
    return ChemLMDecoder(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------


def test_scale_presets_are_monotonic_and_well_formed() -> None:
    """Every preset must be internally consistent and strictly increasing in capacity."""
    order = ["tiny", "small", "base", "large", "xl"]
    assert set(order) <= set(SCALE_PRESETS)

    prev_params = -1.0
    for name in order:
        preset = SCALE_PRESETS[name]
        assert isinstance(preset, DecoderScale)
        assert preset.d_model % preset.n_heads == 0, name
        assert preset.n_heads % preset.n_kv_heads == 0, name
        assert preset.n_kv_heads >= 1
        assert preset.d_ff >= preset.d_model
        est = preset.estimated_params(VOCAB)
        assert est > prev_params, f"{name} not larger than previous preset"
        prev_params = est


def test_resolve_scale_accepts_aliases_and_rejects_unknown() -> None:
    assert resolve_scale("base").name == "base"
    assert resolve_scale("BASE").name == "base"
    assert resolve_scale(" tiny ").name == "tiny"
    with pytest.raises(ValueError, match="Unknown decoder scale"):
        resolve_scale("gigantic")


def test_scaled_decoder_config_roundtrips_through_pydantic() -> None:
    cfg = scaled_decoder_config("small", vocab_size=VOCAB, arch="chemlm")
    assert isinstance(cfg, DecoderConfig)
    preset = SCALE_PRESETS["small"]
    assert cfg.d_model == preset.d_model
    assert cfg.n_layers == preset.n_layers
    assert cfg.n_heads == preset.n_heads
    assert cfg.d_ff == preset.d_ff
    assert cfg.vocab_size == VOCAB
    assert cfg.arch == "chemlm"


def test_xl_preset_is_at_least_one_billion_parameters() -> None:
    """The flagship ``xl`` preset must genuinely be a >=1B parameter decoder."""
    est = SCALE_PRESETS["xl"].estimated_params(vocab_size=2362)
    assert est >= 1.0e9, f"xl preset only {est / 1e6:.1f}M params"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_chemlm_and_moe_are_registered_decoders() -> None:
    names = list_decoders()
    assert "chemlm" in names
    assert "chemlm_moe" in names


def test_build_decoder_dispatches_chemlm_via_registry() -> None:
    cfg = DecoderConfig(arch="chemlm", vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, n_layers=2)
    dec = build_decoder(cfg)
    assert isinstance(dec, ChemLMDecoder)
    assert not dec.is_moe


def test_build_decoder_dispatches_chemlm_moe_via_registry() -> None:
    cfg = DecoderConfig(
        arch="chemlm_moe", vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, n_layers=2
    )
    dec = build_decoder(cfg)
    assert isinstance(dec, ChemLMDecoder)
    assert dec.is_moe


def test_chemlm_satisfies_molecular_decoder_protocol() -> None:
    dec = _tiny_decoder()
    assert isinstance(dec, MolecularDecoderProtocol)


def test_chemlm_matches_legacy_decoder_call_signature() -> None:
    """``GenEuclideanModel`` calls ``decoder(ids, memory=, memory_mask=, return_attn=)``."""
    dec = _tiny_decoder().eval()
    ids = torch.randint(4, VOCAB, (2, 7))
    memory = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        logits, attn = dec(ids, memory=memory, memory_mask=mask, return_attn=True)
    assert logits.shape == (2, 7, VOCAB)
    assert attn is not None
    assert attn.shape[0] == 2 and attn.shape[-1] == 5


# ---------------------------------------------------------------------------
# Unconditional + conditional forward
# ---------------------------------------------------------------------------


def test_unconditional_forward_without_memory() -> None:
    """Stage-1 chemical prior pretraining path: ``memory=None``."""
    dec = _tiny_decoder().eval()
    ids = torch.randint(4, VOCAB, (3, 9))
    with torch.no_grad():
        logits, attn = dec(ids, memory=None)
    assert logits.shape == (3, 9, VOCAB)
    assert attn is None
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("mode", ["cross_attn", "prefix"])
def test_conditioning_modes_run_and_stay_finite(mode: str) -> None:
    dec = _tiny_decoder(conditioning_mode=mode).eval()
    ids = torch.randint(4, VOCAB, (2, 8))
    memory = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        logits, _ = dec(ids, memory=memory, memory_mask=mask)
    assert logits.shape == (2, 8, VOCAB)
    assert torch.isfinite(logits).all()


def test_pocket_conditioning_actually_changes_logits() -> None:
    """A decoder that ignores its pocket is useless: require measurable sensitivity."""
    torch.manual_seed(7)
    dec = _tiny_decoder(cross_attn_gate_init=1.0).eval()
    ids = torch.randint(4, VOCAB, (1, 6))
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        a, _ = dec(ids, memory=torch.randn(1, 5, 32), memory_mask=mask)
        b, _ = dec(ids, memory=torch.randn(1, 5, 32) * 3.0, memory_mask=mask)
    assert (a - b).abs().max().item() > 1e-5


def test_padded_pocket_atoms_have_exactly_zero_influence() -> None:
    """Appending masked-out garbage pocket atoms must not change logits at all."""
    torch.manual_seed(11)
    dec = _tiny_decoder(cross_attn_gate_init=1.0).eval()
    ids = torch.randint(4, VOCAB, (2, 6))
    real = torch.randn(2, 4, 32)

    mask_real = torch.ones(2, 4, dtype=torch.bool)
    garbage = torch.randn(2, 3, 32) * 50.0
    mem_pad = torch.cat([real, garbage], dim=1)
    mask_pad = torch.cat([mask_real, torch.zeros(2, 3, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        base, _ = dec(ids, memory=real, memory_mask=mask_real)
        padded, _ = dec(ids, memory=mem_pad, memory_mask=mask_pad)
    assert torch.allclose(base, padded, atol=1e-5), (base - padded).abs().max().item()


def _causal_leakage(dec: ChemLMDecoder, *, seq_len: int = 16, split: int = 8) -> float:
    """Perturb tokens at positions ``>= split`` and return max change before ``split``."""
    gen = torch.Generator().manual_seed(303)
    a = torch.randint(1, VOCAB, (2, seq_len), generator=gen)
    b = a.clone()
    b[:, split:] = (a[:, split:] + 5) % (VOCAB - 1) + 1
    memory = torch.randn(2, 12, dec.d_model, generator=gen)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -3:] = False

    dec.eval()
    with torch.no_grad():
        la, _ = dec(a, memory=memory, memory_mask=mask)
        lb, _ = dec(b, memory=memory, memory_mask=mask)
    return float((la[:, :split] - lb[:, :split]).abs().max())


@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("use_cross_attn", [False, True])
def test_dense_decoder_has_bitwise_zero_causal_leakage(qk_norm: bool, use_cross_attn: bool) -> None:
    """Changing future tokens must not alter earlier logits by even one bit."""
    torch.manual_seed(0)
    dec = _tiny_decoder(n_layers=2, qk_norm=qk_norm, use_cross_attn=use_cross_attn)
    assert _causal_leakage(dec) == 0.0


def test_moe_causal_leakage_is_within_float32_roundoff() -> None:
    """MoE routing is per-token, so any difference is GEMM round-off, not leakage.

    A token's expert assignment depends only on its own hidden state, so no information
    travels backwards in time. However, perturbing later tokens changes how many tokens
    each expert receives, which changes that expert's matmul shape and shifts results by
    float32 round-off. The guarantee is therefore a tight bound, not bitwise equality.
    """
    torch.manual_seed(0)
    dec = ChemLMDecoder(
        vocab_size=VOCAB,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
        use_moe=True,
        num_experts=8,
        moe_top_k=2,
    )
    assert _causal_leakage(dec) < 1e-6


def test_moe_routing_depends_only_on_each_token_own_state() -> None:
    """The expert chosen for a token must not change when other tokens change."""
    torch.manual_seed(0)
    moe = SparseMoEFeedForward(d_model=16, d_ff=32, num_experts=8, top_k=2)
    h1 = torch.randn(1, 16, 16)
    h2 = h1.clone()
    h2[:, 8:] = torch.randn(1, 8, 16)

    with torch.no_grad():
        routed1 = torch.topk(torch.softmax(moe.router(h1.reshape(-1, 16).float()), -1), 2, -1)
        routed2 = torch.topk(torch.softmax(moe.router(h2.reshape(-1, 16).float()), -1), 2, -1)
    # Unchanged tokens (positions < 8) must keep identical expert assignments.
    assert torch.equal(routed1.indices[:8], routed2.indices[:8])


def test_all_masked_pocket_row_yields_finite_logits() -> None:
    dec = _tiny_decoder().eval()
    ids = torch.randint(4, VOCAB, (2, 5))
    memory = torch.randn(2, 4, 32)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    with torch.no_grad():
        logits, _ = dec(ids, memory=memory, memory_mask=mask)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# KV-cached incremental decoding parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["cross_attn", "prefix"])
def test_kv_cached_stepwise_decoding_matches_full_forward(mode: str) -> None:
    """Incremental cached decoding must match a full-sequence forward pass."""
    torch.manual_seed(3)
    dec = _tiny_decoder(conditioning_mode=mode).eval()
    ids = torch.randint(4, VOCAB, (2, 9))
    memory = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)

    with torch.no_grad():
        full_logits, _ = dec(ids, memory=memory, memory_mask=mask)

        cache = None
        step_logits = []
        for t in range(ids.shape[1]):
            out, cache = dec.forward_cached(
                ids[:, t : t + 1],
                memory=memory,
                memory_mask=mask,
                kv_cache=cache,
            )
            step_logits.append(out)
        cached_logits = torch.cat(step_logits, dim=1)

    assert cached_logits.shape == full_logits.shape
    max_diff = (cached_logits - full_logits).abs().max().item()
    assert max_diff < 2e-4, f"KV-cache divergence {max_diff:.3e}"


def test_generate_produces_valid_token_ids_and_stops_at_eos() -> None:
    dec = _tiny_decoder().eval()
    memory = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = dec.generate(
        batch_size=2,
        memory=memory,
        memory_mask=mask,
        max_new_tokens=12,
        bos_id=1,
        eos_id=2,
        temperature=1.0,
        seed=0,
    )
    assert out.ndim == 2 and out.shape[0] == 2
    assert out.shape[1] <= 13
    assert int(out.min()) >= 0 and int(out.max()) < VOCAB


# ---------------------------------------------------------------------------
# QK-norm
# ---------------------------------------------------------------------------


def test_qk_norm_flag_is_wired_into_attention() -> None:
    torch.manual_seed(5)
    plain = _tiny_decoder(qk_norm=False).eval()
    normed = _tiny_decoder(qk_norm=True).eval()
    ids = torch.randint(4, VOCAB, (2, 8))

    with torch.no_grad():
        lp, _ = plain(ids)
        ln, _ = normed(ids)
    assert torch.isfinite(lp).all() and torch.isfinite(ln).all()
    assert normed.blocks[0].self_attn.use_qk_norm is True
    assert normed.blocks[0].self_attn.qk_norm is not None
    assert plain.blocks[0].self_attn.use_qk_norm is False
    assert plain.blocks[0].self_attn.qk_norm is None


def test_qk_norm_keeps_logits_finite_under_extreme_embedding_scale() -> None:
    """QK-norm bounds pre-softmax scores even when activations blow up."""
    torch.manual_seed(5)
    dec = _tiny_decoder(qk_norm=True).eval()
    with torch.no_grad():
        dec.token_emb.weight.mul_(200.0)
        logits, _ = dec(torch.randint(4, VOCAB, (2, 8)))
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Sparse Mixture-of-Experts
# ---------------------------------------------------------------------------


def test_moe_feedforward_shapes_and_aux_loss() -> None:
    torch.manual_seed(1)
    moe = SparseMoEFeedForward(d_model=16, d_ff=32, num_experts=4, top_k=2)
    x = torch.randn(2, 7, 16)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert moe.last_aux_loss is not None
    assert float(moe.last_aux_loss.detach()) >= 0.0


def test_moe_router_selects_exactly_top_k_experts_per_token() -> None:
    torch.manual_seed(2)
    moe = SparseMoEFeedForward(d_model=16, d_ff=32, num_experts=6, top_k=2)
    x = torch.randn(3, 5, 16)
    moe(x)
    counts = moe.last_dispatch_counts
    assert counts is not None
    assert int(counts.sum().item()) == 3 * 5 * 2


def test_moe_balanced_router_has_low_aux_loss() -> None:
    """A perfectly uniform router should sit near the ideal load-balance loss of 1.0."""
    torch.manual_seed(4)
    moe = SparseMoEFeedForward(d_model=16, d_ff=32, num_experts=4, top_k=2)
    with torch.no_grad():
        moe.router.weight.zero_()  # uniform logits -> uniform routing probabilities
    moe(torch.randn(4, 16, 16))
    assert moe.last_aux_loss is not None
    aux = float(moe.last_aux_loss.detach())
    assert aux == pytest.approx(1.0, abs=0.35), aux


def test_moe_decoder_exposes_summed_aux_loss_and_is_trainable() -> None:
    torch.manual_seed(6)
    dec = build_chemlm_moe_decoder(
        DecoderConfig(
            arch="chemlm_moe", vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, n_layers=2
        )
    )
    dec.train()
    ids = torch.randint(4, VOCAB, (2, 8))
    logits, _ = dec(ids)
    aux = dec.aux_loss()
    assert aux.requires_grad
    assert float(aux.detach()) > 0.0

    loss = logits.float().mean() + 0.01 * aux
    loss.backward()
    grads = [p.grad for p in dec.parameters() if p.grad is not None]
    assert grads, "no gradients flowed into the MoE decoder"
    assert all(torch.isfinite(g).all() for g in grads)


def test_dense_decoder_aux_loss_is_zero() -> None:
    dec = _tiny_decoder()
    assert float(dec.aux_loss()) == 0.0


def test_moe_has_more_params_than_dense_at_equal_width() -> None:
    common: dict[str, object] = {
        "vocab_size": VOCAB,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 64,
        "n_layers": 2,
        "max_seq_len": 32,
    }
    dense = ChemLMDecoder(**common)  # type: ignore[arg-type]
    moe = ChemLMDecoder(**common, use_moe=True, num_experts=4, moe_top_k=2)  # type: ignore[arg-type]
    assert moe.num_parameters() > dense.num_parameters()


# ---------------------------------------------------------------------------
# Property conditioning
# ---------------------------------------------------------------------------


def test_fourier_property_embedding_is_deterministic_and_bounded() -> None:
    emb = FourierPropertyEmbedding(num_properties=3, num_frequencies=8, d_out=16)
    x = torch.tensor([[0.5, -1.0, 2.0]])
    a = emb(x)
    b = emb(x)
    assert a.shape == (1, 3, 16)
    assert torch.allclose(a, b)
    assert torch.isfinite(a).all()


def test_property_conditioner_emits_tokens_and_honours_unknown_mask() -> None:
    torch.manual_seed(8)
    cond = PropertyConditioner(num_properties=4, d_model=16, num_tokens=3)
    props = torch.tensor([[0.7, 350.0, 2.1, 1.0], [0.3, 500.0, 3.4, 0.0]])

    known = torch.ones(2, 4, dtype=torch.bool)
    tok_known = cond(props, property_mask=known)
    assert tok_known.shape == (2, 3, 16)

    unknown = torch.zeros(2, 4, dtype=torch.bool)
    tok_unknown = cond(props, property_mask=unknown)
    # With every property unknown, the two rows must be identical (values ignored).
    assert torch.allclose(tok_unknown[0], tok_unknown[1], atol=1e-6)
    assert not torch.allclose(tok_known, tok_unknown)


def test_property_conditioned_decoder_changes_logits_with_properties() -> None:
    torch.manual_seed(9)
    dec = _tiny_decoder(num_properties=3, num_property_tokens=2).eval()
    ids = torch.randint(4, VOCAB, (1, 6))
    p1 = torch.tensor([[0.9, 1.0, 0.1]])
    p2 = torch.tensor([[0.1, 5.0, 0.9]])
    with torch.no_grad():
        a, _ = dec(ids, properties=p1)
        b, _ = dec(ids, properties=p2)
    assert (a - b).abs().max().item() > 1e-5


def test_property_decoder_works_when_properties_omitted() -> None:
    dec = _tiny_decoder(num_properties=3, num_property_tokens=2).eval()
    ids = torch.randint(4, VOCAB, (2, 6))
    with torch.no_grad():
        logits, _ = dec(ids)
    assert logits.shape == (2, 6, VOCAB)
    assert torch.isfinite(logits).all()


def test_property_tokens_do_not_shift_output_length() -> None:
    """Property prefix tokens are stripped before the LM head."""
    dec = _tiny_decoder(num_properties=2, num_property_tokens=4).eval()
    ids = torch.randint(4, VOCAB, (2, 7))
    with torch.no_grad():
        logits, _ = dec(ids, properties=torch.randn(2, 2))
    assert logits.shape == (2, 7, VOCAB)


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------


def test_gradient_checkpointing_matches_plain_forward_and_grads() -> None:
    torch.manual_seed(12)
    dec = _tiny_decoder(n_layers=3)
    ids = torch.randint(4, VOCAB, (2, 8))
    labels = torch.randint(4, VOCAB, (2, 8))

    def _loss(model: ChemLMDecoder) -> torch.Tensor:
        logits, _ = model(ids)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB).float(), labels.reshape(-1)
        )

    dec.train()
    base = _loss(dec)
    base.backward()
    base_grads = {n: p.grad.clone() for n, p in dec.named_parameters() if p.grad is not None}

    dec.zero_grad(set_to_none=True)
    dec.enable_gradient_checkpointing(True)
    assert dec.gradient_checkpointing is True
    ckpt = _loss(dec)
    ckpt.backward()

    assert ckpt.item() == pytest.approx(base.item(), rel=1e-5, abs=1e-6)
    for name, param in dec.named_parameters():
        if param.grad is None or name not in base_grads:
            continue
        torch.testing.assert_close(param.grad, base_grads[name], rtol=2e-4, atol=2e-5)


def test_gradient_checkpointing_is_inert_under_no_grad() -> None:
    dec = _tiny_decoder()
    dec.enable_gradient_checkpointing(True)
    dec.eval()
    with torch.no_grad():
        logits, _ = dec(torch.randint(4, VOCAB, (2, 6)))
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Weight tying, init, parameter counting
# ---------------------------------------------------------------------------


def test_weights_are_tied_and_pad_embedding_is_zero() -> None:
    dec = _tiny_decoder()
    assert dec.lm_head.weight is dec.token_emb.weight
    assert torch.allclose(dec.token_emb.weight[PAD_ID], torch.zeros(dec.d_model))


def test_estimated_params_tracks_actual_within_ten_percent() -> None:
    """``DecoderScale.estimated_params`` must be a usable planning number."""
    preset = DecoderScale(name="probe", d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=256)
    actual = ChemLMDecoder(
        vocab_size=VOCAB,
        d_model=preset.d_model,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        d_ff=preset.d_ff,
        max_seq_len=64,
        use_cross_attn=False,
    ).num_parameters()
    est = preset.estimated_params(VOCAB, include_cross_attn=False)
    assert abs(est - actual) / actual < 0.10, f"est={est} actual={actual}"


def test_init_std_shrinks_with_depth() -> None:
    """Residual-projection init must be depth-scaled to keep deep stacks stable."""
    shallow = _tiny_decoder(n_layers=2)
    deep = _tiny_decoder(n_layers=16)
    s_std = shallow.blocks[0].self_attn.o_proj.weight.std().item()
    d_std = deep.blocks[0].self_attn.o_proj.weight.std().item()
    assert d_std < s_std
    assert d_std == pytest.approx(0.02 / math.sqrt(2.0 * 16), rel=0.25)


# ---------------------------------------------------------------------------
# Backbone transfer
# ---------------------------------------------------------------------------


def test_load_pretrained_backbone_transfers_shape_compatible_weights() -> None:
    """Stage-1 unconditional prior weights must load into a Stage-2 cross-attn decoder."""
    torch.manual_seed(13)
    stage1 = _tiny_decoder(use_cross_attn=False)
    stage2 = _tiny_decoder(use_cross_attn=True)

    missing, unexpected = stage2.load_pretrained_backbone(stage1.state_dict())
    assert torch.allclose(stage2.token_emb.weight, stage1.token_emb.weight)
    assert stage2.lm_head.weight is stage2.token_emb.weight
    # Cross-attention layers are new and must be reported as missing, not silently zeroed.
    assert any("cross_attn" in k for k in missing)
    assert isinstance(unexpected, list)


def test_load_pretrained_backbone_unwraps_checkpoint_prefixes() -> None:
    stage1 = _tiny_decoder(use_cross_attn=False)
    wrapped = {
        "model_state_dict": {f"module.decoder.{k}": v for k, v in stage1.state_dict().items()}
    }
    stage2 = _tiny_decoder(use_cross_attn=False)
    stage2.load_pretrained_backbone(wrapped)
    assert torch.allclose(stage2.token_emb.weight, stage1.token_emb.weight)


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_moe", [False, True])
def test_decoder_overfits_a_single_batch(use_moe: bool) -> None:
    """A correct decoder must be able to drive training loss down on one fixed batch."""
    torch.manual_seed(0)
    dec = ChemLMDecoder(
        vocab_size=VOCAB,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        max_seq_len=24,
        dropout=0.0,
        use_cross_attn=False,
        use_moe=use_moe,
        num_experts=4,
        moe_top_k=2,
    )
    dec.train()
    ids = torch.randint(4, VOCAB, (4, 12))
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-3)

    first = last = 0.0
    for step in range(30):
        logits, _ = dec(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1)
        )
        total = loss + 0.01 * dec.aux_loss()
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())

    assert last < first * 0.6, f"loss did not drop: {first:.3f} -> {last:.3f}"


@pytest.mark.parametrize("arch", ["chemlm", "chemlm_moe"])
def test_runs_inside_the_full_geneuclidean_pipeline(arch: str) -> None:
    """The decoder must drop into ``GenEuclideanModel`` by config alone.

    This is the integration that was previously impossible: the strong attention
    primitives lived outside the registry, so ``build_decoder`` could not reach them and
    the real training pipeline was stuck with the cache-less ``transformer`` decoder.
    """
    from geneuclidean.config import ExperimentConfig
    from geneuclidean.models.geneuclidean import GenEuclideanModel

    cfg = ExperimentConfig()
    cfg.decoder.arch = arch
    cfg.decoder.vocab_size = VOCAB
    cfg.decoder.d_model = 64
    cfg.decoder.n_heads = 4
    cfg.decoder.n_layers = 2
    cfg.decoder.d_ff = 128
    cfg.encoder.hidden_dim = 32
    cfg.encoder.vector_dim = 16
    cfg.encoder.out_dim = 64
    cfg.encoder.num_layers = 2
    cfg.cvae.latent_dim = 16

    model = GenEuclideanModel(cfg)
    assert isinstance(model.decoder, ChemLMDecoder)
    assert model.decoder.is_moe is (arch == "chemlm_moe")

    batch_size, n_atoms, seq_len = 2, 40, 12
    batch = {
        "coords": torch.randn(batch_size, n_atoms, 3),
        "node_feats": torch.randn(batch_size, n_atoms, 52),
        "atom_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool),
        "input_ids": torch.randint(4, VOCAB, (batch_size, seq_len)),
        "ligand_fp": torch.rand(batch_size, 1024),
        "props": torch.randn(batch_size, 8),
        "pkd": torch.randn(batch_size),
    }

    model.train()
    out = model(batch, cfg_drop=True, return_attn=True)

    assert out.logits.shape == (batch_size, seq_len, VOCAB)
    assert torch.isfinite(out.logits).all()
    # Cross-attention memory is pocket atoms plus the CVAE latent tokens.
    assert out.attn_weights is not None
    assert out.attn_weights.shape[-1] == n_atoms + cfg.cvae.num_latent_tokens
    assert out.pred_pkd is not None and out.pred_pkd.shape == (batch_size,)

    out.logits.float().mean().backward()
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_pocket_reliance_detects_a_decoder_that_ignores_its_pocket() -> None:
    """The diagnostic must flag a decoder with cross-attention disabled.

    This is the dominant silent failure mode: docking score hides it (TargetDiff beats
    the crystal ligand on raw Glide while capturing 47% of its specificity), so a
    likelihood-based probe is the only cheap guard.
    """
    torch.manual_seed(0)
    deaf = _tiny_decoder(use_cross_attn=False)
    ids = torch.randint(4, VOCAB, (4, 10))
    memory = torch.randn(4, 8, 32)
    mask = torch.ones(4, 8, dtype=torch.bool)

    report = pocket_reliance_report(deaf, ids, memory, mask)
    assert report.delta_permuted == pytest.approx(0.0, abs=1e-6)
    assert report.delta_null == pytest.approx(0.0, abs=1e-6)
    assert report.uses_specific_pocket is False
    assert report.gate_magnitudes == []


def test_pocket_reliance_sees_a_decoder_that_does_use_its_pocket() -> None:
    """A decoder trained to depend on its pocket must show a positive delta."""
    torch.manual_seed(0)
    dec = _tiny_decoder(n_layers=2, cross_attn_gate_init=1.0)
    ids = torch.randint(4, VOCAB, (4, 10))
    memory = torch.randn(4, 8, 32) * 2.0
    mask = torch.ones(4, 8, dtype=torch.bool)

    # Teach it that each pocket predicts its own ligand.
    opt = torch.optim.AdamW(dec.parameters(), lr=5e-3)
    dec.train()
    for _ in range(120):
        logits, _ = dec(ids, memory=memory, memory_mask=mask)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    report = pocket_reliance_report(dec, ids, memory, mask)
    assert report.nll_true < report.nll_permuted, report.to_dict()
    assert report.delta_permuted > 0.0
    assert len(report.gate_magnitudes) == 2
    assert all(g >= 0.0 for g in report.gate_magnitudes)


def test_pocket_reliance_requires_a_permutable_batch() -> None:
    dec = _tiny_decoder()
    with pytest.raises(ValueError, match="batch_size >= 2"):
        pocket_reliance_report(dec, torch.randint(4, VOCAB, (1, 8)), torch.randn(1, 6, 32))


def test_pocket_reliance_restores_training_mode() -> None:
    """A diagnostic must not silently leave the model in eval mode mid-training."""
    dec = _tiny_decoder()
    dec.train()
    pocket_reliance_report(dec, torch.randint(4, VOCAB, (2, 8)), torch.randn(2, 6, 32))
    assert dec.training is True


def test_pocket_reliance_report_is_json_serializable() -> None:
    import json

    dec = _tiny_decoder()
    payload = pocket_reliance_report(
        dec, torch.randint(4, VOCAB, (2, 8)), torch.randn(2, 6, 32)
    ).to_dict()
    assert json.loads(json.dumps(payload))["uses_specific_pocket"] in (True, False)
    assert "delta_permuted" in payload


def test_gate_magnitudes_track_the_gate_parameter() -> None:
    """Gate magnitude is a free readout of whether a layer is using the pocket."""
    dec = _tiny_decoder(n_layers=3, cross_attn_gate_init=0.0)
    assert cross_attention_gate_magnitudes(dec) == pytest.approx([0.0, 0.0, 0.0])

    with torch.no_grad():
        for block in dec.blocks:
            block.cross_attn.gate.fill_(1.0)
    expected = float(torch.tanh(torch.tensor(1.0)))
    assert cross_attention_gate_magnitudes(dec) == pytest.approx([expected] * 3)


def test_gate_magnitudes_empty_without_cross_attention() -> None:
    assert cross_attention_gate_magnitudes(_tiny_decoder(use_cross_attn=False)) == []


def test_shipped_chemlm_config_stays_in_sync_with_its_scale_preset() -> None:
    """``configs/geneuclidean_v6_chemlm.yaml`` must not drift from ``SCALE_PRESETS``."""
    from pathlib import Path

    from geneuclidean.config import ExperimentConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / "geneuclidean_v6_chemlm.yaml"
    assert config_path.is_file(), f"missing shipped config: {config_path}"

    cfg = ExperimentConfig.from_yaml(config_path)
    assert cfg.decoder.arch == "chemlm"

    preset = SCALE_PRESETS["small"]
    assert (
        cfg.decoder.d_model,
        cfg.decoder.n_layers,
        cfg.decoder.n_heads,
        cfg.decoder.d_ff,
    ) == (preset.d_model, preset.n_layers, preset.n_heads, preset.d_ff)


def test_build_chemlm_decoder_respects_cross_attn_gate_init() -> None:
    cfg = DecoderConfig(arch="chemlm", vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, n_layers=1)
    dec = build_chemlm_decoder(cfg, cross_attn_gate_init=0.75)
    gate = dec.blocks[0].cross_attn.gate
    assert gate is not None
    assert float(gate.detach()) == pytest.approx(0.75)
