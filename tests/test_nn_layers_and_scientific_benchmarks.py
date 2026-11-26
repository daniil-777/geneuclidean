"""Google-SWE Unit & Scientific Benchmark Verification Suite for NN Layers & Benchmarks.

Verifies mathematical invariants, ``SE(3)``/``SO(3)`` equivariance, chiral enantiomer
sensitivity, padding-mask isolation, causal non-leakage, KV-cache equivalence, Apple Silicon
MPS ``float32`` compliance (zero ``torch.float64`` params/buffers), finite autograd gradients,
and end-to-end execution of :func:`run_scientific_benchmark_suite`.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import geneuclidean as ge
from geneuclidean.evaluation.scientific_benchmark import (
    ScientificBenchmarkSuiteReport,
    format_benchmark_markdown_table,
    run_scientific_benchmark_suite,
)
from geneuclidean.models.layers import (
    GEGLU,
    AdaLNZero,
    DropPath,
    EquivariantScalarVectorGate,
    GroupedQueryAttention,
    PerceiverLatentResampler,
    QKNorm,
    RMSNorm,
    SE3GeometricTransformerBlock,
    SinusoidalFourierTimeEmbedding,
    SpatialBiasedCrossAttention,
    SwiGLU,
    TriangleMultiplicativePairUpdate,
)

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def _assert_no_float64_state(module: nn.Module, name: str) -> None:
    """Assert zero ``torch.float64`` parameters or buffers on ``module``."""
    bad_params = [n for n, p in module.named_parameters() if p.dtype == torch.float64]
    bad_bufs = [n for n, b in module.named_buffers() if b.dtype == torch.float64]
    assert bad_params == [] and bad_bufs == [], (
        f"{name} contains forbidden float64 state: params={bad_params}, buffers={bad_bufs}"
    )


def _sample_so3(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a proper orthogonal matrix ``R in SO(3)`` with ``det(R) == +1``."""
    mat = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    if float(torch.det(q).item()) < 0.0:
        q[:, 0] = -q[:, 0]
    out: Tensor = q.to(device=device, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# 1. Normalization, Conditioning & Gated FFN Tests
# ---------------------------------------------------------------------------


def test_rmsnorm_qknorm_swiglu_geglu_droppath_and_time_embedding() -> None:
    """Verify RMSNorm, QKNorm, SwiGLU, GEGLU, DropPath, and Fourier time embeddings."""
    torch.manual_seed(7)
    rms = RMSNorm(dim=32)
    qk = QKNorm(head_dim=16, num_q_heads=4, num_kv_heads=2)
    swiglu = SwiGLU(d_model=32, d_ff=64, dropout=0.0)
    geglu = GEGLU(d_model=32, d_ff=64, dropout=0.0, approximate="tanh")
    dp = DropPath(drop_prob=0.35)
    t_emb = SinusoidalFourierTimeEmbedding(dim=33, out_dim=48)  # odd dim padding check

    for name, mod in (
        ("RMSNorm", rms),
        ("QKNorm", qk),
        ("SwiGLU", swiglu),
        ("GEGLU", geglu),
        ("DropPath", dp),
        ("SinusoidalFourierTimeEmbedding", t_emb),
    ):
        _assert_no_float64_state(mod, name)

    # RMSNorm preserves exact zero on padded zero vectors and normalizes non-zero unit RMS
    zeros = torch.zeros((2, 4, 32), dtype=torch.float32)
    assert torch.equal(rms(zeros), zeros)
    x = torch.randn((2, 5, 32), dtype=torch.float32) * 7.5
    y = rms(x)
    rms_per_token = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms_per_token, torch.ones_like(rms_per_token), atol=1e-5)

    # QKNorm supports GQA (H_q=4 != H_kv=2) and bounds dot products by head_dim
    q = torch.randn((2, 4, 10, 16), dtype=torch.float32) * 50.0
    k = torch.randn((2, 2, 10, 16), dtype=torch.float32) * 50.0
    q_n, k_n = qk(q, k)
    assert q_n.shape == q.shape and k_n.shape == k.shape
    assert torch.allclose(q_n.pow(2).mean(dim=-1).sqrt(), torch.ones((2, 4, 10)), atol=1e-5)
    assert torch.allclose(k_n.pow(2).mean(dim=-1).sqrt(), torch.ones((2, 2, 10)), atol=1e-5)

    # SwiGLU & GEGLU forward + backward gradient flow
    x_req = x.clone().requires_grad_(True)
    loss = (swiglu(x_req) + geglu(x_req)).pow(2).mean()
    loss.backward()
    assert x_req.grad is not None and torch.isfinite(x_req.grad).all()

    # DropPath: exact identity in eval(), preserves SO(3) vector equivariance in train()
    dp.eval()
    vecs = torch.randn((8, 6, 4, 3), dtype=torch.float32)
    assert torch.equal(dp(vecs), vecs)
    dp.train()
    gen = torch.Generator().manual_seed(19)
    rot = _sample_so3(gen, torch.device("cpu"))
    torch.manual_seed(99)
    v_dropped_rot = dp(vecs) @ rot.T
    torch.manual_seed(99)
    v_rot_dropped = dp(vecs @ rot.T)
    assert torch.allclose(v_dropped_rot, v_rot_dropped, atol=1e-6)

    # SinusoidalFourierTimeEmbedding handles continuous t in [0, 1] across shapes
    t = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float32)
    emb = t_emb(t)
    assert emb.shape == (5, 48) and emb.dtype == torch.float32
    assert (emb[0] - emb[-1]).abs().max().item() > 1e-2


def test_adaln_zero_exact_identity_at_initialization_and_gradient_flow() -> None:
    """AdaLNZero must satisfy exact residual identity at initialization and learnable updates."""
    torch.manual_seed(12)
    adaln = AdaLNZero(d_model=48, cond_dim=24, zero_init=True)
    _assert_no_float64_state(adaln, "AdaLNZero")

    x = torch.randn((3, 9, 48), dtype=torch.float32)
    cond = torch.randn((3, 24), dtype=torch.float32)
    sublayer = nn.Linear(48, 48)

    mod_x, gate = adaln(x, cond)
    expected_ln = F.layer_norm(x, (48,), eps=adaln.eps)
    assert torch.allclose(mod_x, expected_ln, atol=1e-6)
    assert torch.equal(gate, torch.zeros_like(gate))

    # Exact bit-level identity y == x at initialization for arbitrary non-zero sublayer
    y = adaln.forward_residual(x, cond, sublayer)
    assert torch.equal(y, x)

    # Gradients flow into modulation weights so subsequent optimization steps activate the branch
    x_g = x.clone().requires_grad_(True)
    cond_g = cond.clone().requires_grad_(True)
    out = adaln.forward_residual(x_g, cond_g, sublayer)
    target = torch.randn_like(out)
    F.mse_loss(out, target).backward()
    assert adaln.modulation.weight.grad is not None
    assert adaln.modulation.weight.grad.abs().max().item() > 1e-4


# ---------------------------------------------------------------------------
# 2. GroupedQueryAttention, SpatialBiasedCrossAttention & Perceiver Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_query_attention_causality_rope_shift_and_kv_cache(device: str) -> None:
    """Verify GQA causal non-leakage, RoPE shift invariance (anisotropic QKNorm), and KV cache."""
    torch.manual_seed(21)
    dev = torch.device(device)
    gqa = (
        GroupedQueryAttention(
            d_model=64,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.0,
            use_qk_norm=True,
            use_rope=True,
            is_causal=True,
        )
        .to(dev)
        .eval()
    )
    _assert_no_float64_state(gqa, "GroupedQueryAttention")

    # Perturb QKNorm weights anisotropically to prove QKNorm-before-RoPE preserves shift invariance
    with torch.no_grad():
        assert gqa.qk_norm is not None
        gqa.qk_norm.q_weight.copy_(
            torch.linspace(0.5, 1.8, steps=gqa.qk_norm.q_weight.numel(), device=dev).reshape_as(
                gqa.qk_norm.q_weight
            )
        )
        gqa.qk_norm.k_weight.copy_(
            torch.linspace(1.5, 0.6, steps=gqa.qk_norm.k_weight.numel(), device=dev).reshape_as(
                gqa.qk_norm.k_weight
            )
        )

    gen = torch.Generator(device="cpu").manual_seed(108)
    x1 = torch.randn((2, 14, 64), generator=gen, dtype=torch.float32).to(dev)
    x2 = x1.clone()
    x2[:, 7:, :] = torch.randn_like(x2[:, 7:, :]) * 10.0

    with torch.no_grad():
        out1 = gqa(x1)
        out2 = gqa(x2)
        # 1. Exact causal non-leakage: suffix perturbations at t >= 7 cannot affect t < 7
        assert (out1[:, :7, :] - out2[:, :7, :]).abs().max().item() == 0.0

        # 2. RoPE relative-position shift invariance: shifting all positions by +53
        pos_0 = torch.arange(14, device=dev)
        pos_shift = pos_0 + 53
        out_p0 = gqa(x1, position_ids=pos_0)
        out_ps = gqa(x1, position_ids=pos_shift)
        assert (out_p0 - out_ps).abs().max().item() < 1e-5

        # 3. Incremental step-by-step KV cache matches full-sequence causal forward
        cache = None
        step_outputs: list[Tensor] = []
        for t in range(x1.shape[1]):
            step_out, cache = gqa(x1[:, t : t + 1, :], kv_cache=cache, use_cache=True)
            step_outputs.append(step_out)
        cached_full = torch.cat(step_outputs, dim=1)
        assert (cached_full - out1).abs().max().item() < 1e-5


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_spatial_biased_cross_attention_se3_and_padding_invariance(device: str) -> None:
    """Verify SpatialBiasedCrossAttention SE(3) invariance, masking, and Flamingo gating."""
    torch.manual_seed(33)
    dev = torch.device(device)
    layer = (
        SpatialBiasedCrossAttention(
            d_model=64,
            num_heads=4,
            d_memory=48,
            use_coulomb_bias=True,
            gated=True,
            gate_init=0.25,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    _assert_no_float64_state(layer, "SpatialBiasedCrossAttention")

    gen = torch.Generator(device="cpu").manual_seed(77)
    x = torch.randn((2, 8, 64), generator=gen, dtype=torch.float32).to(dev)
    q_xyz = torch.randn((2, 8, 3), generator=gen, dtype=torch.float32).to(dev) * 2.5
    mem = torch.randn((2, 12, 48), generator=gen, dtype=torch.float32).to(dev)
    m_xyz = torch.randn((2, 12, 3), generator=gen, dtype=torch.float32).to(dev) * 3.0
    m_mask = torch.ones((2, 12), dtype=torch.bool, device=dev)
    m_mask[:, -3:] = False

    rot = _sample_so3(gen, dev)
    trans = torch.tensor([[[-45.0, 32.0, 19.0]]], dtype=torch.float32, device=dev)

    with torch.no_grad():
        out_ref, attn_ref = layer(
            x,
            mem,
            memory_mask=m_mask,
            query_coords=q_xyz,
            memory_coords=m_xyz,
            return_attn=True,
        )
        out_se3 = layer(
            x,
            mem,
            memory_mask=m_mask,
            query_coords=(q_xyz @ rot.T) + trans,
            memory_coords=(m_xyz @ rot.T) + trans,
        )
        assert (out_ref - out_se3).abs().max().item() < 5e-5
        assert attn_ref[:, :, :, -3:].abs().max().item() == 0.0

        # Appending extreme dummy pocket atoms leaves outputs unchanged
        pad_mem = torch.full((2, 5, 48), 99.0, device=dev)
        pad_xyz = torch.full((2, 5, 3), -120.0, device=dev)
        pad_mask = torch.zeros((2, 5), dtype=torch.bool, device=dev)
        out_padded = layer(
            x,
            torch.cat((mem, pad_mem), dim=1),
            memory_mask=torch.cat((m_mask, pad_mask), dim=1),
            query_coords=q_xyz,
            memory_coords=torch.cat((m_xyz, pad_xyz), dim=1),
        )
        assert (out_padded - out_ref).abs().max().item() < 1e-5


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_perceiver_latent_resampler_padding_permutation_and_se3_invariance(device: str) -> None:
    """PerceiverLatentResampler is invariant to atom permutation, padding corruption, and SE(3)."""
    torch.manual_seed(44)
    dev = torch.device(device)
    resampler = (
        PerceiverLatentResampler(
            d_in=52,
            d_out=64,
            num_latents=10,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    _assert_no_float64_state(resampler, "PerceiverLatentResampler")

    gen = torch.Generator(device="cpu").manual_seed(91)
    feats = torch.randn((2, 18, 52), generator=gen, dtype=torch.float32).to(dev)
    coords = torch.randn((2, 18, 3), generator=gen, dtype=torch.float32).to(dev) * 2.5
    mask = torch.ones((2, 18), dtype=torch.bool, device=dev)
    mask[0, 14:] = False
    mask[1, 16:] = False

    perm = torch.randperm(18, generator=gen).to(dev)
    rot = _sample_so3(gen, dev)
    trans = torch.tensor([[[60.0, -40.0, 25.0]]], dtype=torch.float32, device=dev)

    with torch.no_grad():
        lat_ref = resampler(feats, mask=mask, coords=coords)
        assert lat_ref.shape == (2, 10, 64)

        # Permutation invariance
        lat_perm = resampler(feats[:, perm, :], mask=mask[:, perm], coords=coords[:, perm, :])
        assert (lat_ref - lat_perm).abs().max().item() < 1e-5

        # SE(3) rigid invariance
        lat_se3 = resampler(feats, mask=mask, coords=(coords @ rot.T) + trans)
        assert (lat_ref - lat_se3).abs().max().item() < 5e-5

        # Padding-mask corruption invariance
        pad_f = torch.full((2, 7, 52), -250.0, device=dev)
        pad_c = torch.full((2, 7, 3), 500.0, device=dev)
        pad_m = torch.zeros((2, 7), dtype=torch.bool, device=dev)
        lat_pad = resampler(
            torch.cat((feats, pad_f), dim=1),
            mask=torch.cat((mask, pad_m), dim=1),
            coords=torch.cat((coords, pad_c), dim=1),
        )
        assert (lat_ref - lat_pad).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 3. SE(3)-Equivariant Gate, Triangle Pair Update & Geometric Transformer Tests
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
def test_equivariant_scalar_vector_gate_and_triangle_pair_updates() -> None:
    """Verify O(3) equivariance of ScalarVectorGate and symmetry/masking of TrianglePairUpdate."""
    torch.manual_seed(55)
    gen = torch.Generator().manual_seed(155)
    sv_gate = EquivariantScalarVectorGate(scalar_dim=48, vector_dim=12, dropout=0.0).eval()
    _assert_no_float64_state(sv_gate, "EquivariantScalarVectorGate")

    s = torch.randn((2, 10, 48), generator=gen, dtype=torch.float32)
    v = torch.randn((2, 10, 12, 3), generator=gen, dtype=torch.float32)
    mask = torch.ones((2, 10), dtype=torch.bool)
    mask[:, -2:] = False
    rot = _sample_so3(gen, torch.device("cpu"))

    with torch.no_grad():
        s0, v0 = sv_gate(s, v, mask=mask)
        s_rot, v_rot = sv_gate(s, v @ rot.T, mask=mask)
        assert (s0 - s_rot).abs().max().item() < 1e-5
        assert (v_rot - (v0 @ rot.T)).abs().max().item() < 1e-5
        assert s0[:, -2:, :].abs().max().item() == 0.0
        assert v0[:, -2:, :, :].abs().max().item() == 0.0

    # TriangleMultiplicativePairUpdate: symmetric, outgoing, incoming, and k-NN modes
    tri_sym = TriangleMultiplicativePairUpdate(
        d_pair=24, direction="symmetric", mode="dense"
    ).eval()
    tri_out = TriangleMultiplicativePairUpdate(d_pair=24, direction="outgoing", mode="dense").eval()
    tri_in = TriangleMultiplicativePairUpdate(d_pair=24, direction="incoming", mode="dense").eval()
    tri_knn = TriangleMultiplicativePairUpdate(
        d_pair=24, direction="outgoing", mode="knn", max_neighbors=6
    ).eval()
    for label, mod in (
        ("tri_sym", tri_sym),
        ("tri_out", tri_out),
        ("tri_in", tri_in),
        ("tri_knn", tri_knn),
    ):
        _assert_no_float64_state(mod, label)

    raw_z = torch.randn((2, 12, 12, 24), generator=gen, dtype=torch.float32)
    sym_z = 0.5 * (raw_z + raw_z.transpose(1, 2))
    coords = torch.randn((2, 12, 3), generator=gen, dtype=torch.float32) * 2.2

    with torch.no_grad():
        z_sym = tri_sym(sym_z, mask=torch.cat((mask, torch.zeros((2, 2), dtype=torch.bool)), dim=1))
        assert (z_sym - z_sym.transpose(1, 2)).abs().max().item() < 1e-6
        assert z_sym[:, -2:, :, :].abs().max().item() == 0.0
        assert z_sym[:, :, -2:, :].abs().max().item() == 0.0

        # Padding-mask invariance on dense outgoing/incoming AND k-NN local mode
        valid_m = torch.ones((2, 12), dtype=torch.bool)
        pad_z = torch.full((2, 16, 16, 24), 45.0, dtype=torch.float32)
        pad_z[:, :12, :12, :] = raw_z
        pad_m = torch.zeros((2, 16), dtype=torch.bool)
        pad_m[:, :12] = True
        pad_c = torch.full((2, 16, 3), -90.0, dtype=torch.float32)
        pad_c[:, :12, :] = coords

        for tri_mod in (tri_out, tri_in, tri_knn):
            ref_u = tri_mod(raw_z, mask=valid_m, coords=coords)
            pad_u = tri_mod(pad_z, mask=pad_m, coords=pad_c)
            assert (pad_u[:, :12, :12, :] - ref_u).abs().max().item() < 1e-5


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_se3_geometric_transformer_block_so3_equivariance_and_chirality(device: str) -> None:
    """Verify exact SO(3) + translation equivariance and chiral vs achiral reflection behavior."""
    torch.manual_seed(66)
    dev = torch.device(device)
    blk_chiral = (
        SE3GeometricTransformerBlock(
            scalar_dim=64,
            vector_dim=16,
            pair_dim=24,
            num_heads=4,
            cond_dim=32,
            use_se3_chirality=True,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    blk_e3 = (
        SE3GeometricTransformerBlock(
            scalar_dim=64,
            vector_dim=16,
            pair_dim=24,
            num_heads=4,
            cond_dim=32,
            use_se3_chirality=False,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    _assert_no_float64_state(blk_chiral, "SE3GeometricTransformerBlock(chiral)")
    _assert_no_float64_state(blk_e3, "SE3GeometricTransformerBlock(e3)")

    gen = torch.Generator(device="cpu").manual_seed(2026)
    s = torch.randn((2, 14, 64), generator=gen, dtype=torch.float32).to(dev)
    v = torch.randn((2, 14, 16, 3), generator=gen, dtype=torch.float32).to(dev) * 0.6
    coords = torch.randn((2, 14, 3), generator=gen, dtype=torch.float32).to(dev) * 2.2
    cond = torch.randn((2, 32), generator=gen, dtype=torch.float32).to(dev)
    mask = torch.ones((2, 14), dtype=torch.bool, device=dev)
    mask[:, -2:] = False

    with torch.no_grad():
        ref_c = blk_chiral(s, v, coords, mask=mask, cond=cond)
        ref_e = blk_e3(s, v, coords, mask=mask, cond=cond)

        for _ in range(4):
            rot = _sample_so3(gen, dev)
            trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 160.0 - 80.0).to(
                dev
            )
            out_c = blk_chiral(s, v @ rot.T, (coords @ rot.T) + trans, mask=mask, cond=cond)
            assert (out_c.scalars - ref_c.scalars).abs().max().item() < 1e-4
            assert (out_c.vectors - (ref_c.vectors @ rot.T)).abs().max().item() < 1e-4
            assert (out_c.pair_feats - ref_c.pair_feats).abs().max().item() < 1e-4

        # Improper reflection (x -> -x, v -> -v): chiral changes scalars; E(3) stays invariant
        mir_c = blk_chiral(s, -v, -coords, mask=mask, cond=cond)
        mir_e = blk_e3(s, -v, -coords, mask=mask, cond=cond)
        assert (mir_c.scalars - ref_c.scalars).abs().max().item() > 1e-3
        assert (mir_e.scalars - ref_e.scalars).abs().max().item() < 5e-5
        assert (mir_e.vectors - (-ref_e.vectors)).abs().max().item() < 5e-5

    # Zero-init AdaLN mode starts as an exact identity map on scalars & vectors
    blk_zero = (
        SE3GeometricTransformerBlock(
            scalar_dim=32,
            vector_dim=8,
            pair_dim=16,
            num_heads=4,
            cond_dim=16,
            zero_init_adaln=True,
        )
        .to(dev)
        .eval()
    )
    s_small = torch.randn((1, 8, 32), device=dev)
    v_small = torch.randn((1, 8, 8, 3), device=dev)
    c_small = torch.randn((1, 16), device=dev)
    with torch.no_grad():
        out_zero = blk_zero(s_small, v_small, coords[:1, :8, :], cond=c_small)
    assert torch.equal(out_zero.scalars, s_small)
    assert torch.equal(out_zero.vectors, v_small)


# ---------------------------------------------------------------------------
# 4. Scientific & Industrial Benchmark Validation Suite End-to-End Tests
# ---------------------------------------------------------------------------


def test_scientific_benchmark_suite_quantitative_gates_and_exports() -> None:
    """Run full scientific benchmark suite across registered models and assert QA gates."""
    report: ScientificBenchmarkSuiteReport = run_scientific_benchmark_suite(
        device="cpu",
        batch_size=2,
        num_atoms=20,
        encoder_hidden_dim=32,
        decoder_dim=64,
        num_rotations=3,
        num_generated_samples=10,
        seed=42,
        extra_metadata={"ci_profile": "google_swe_validation"},
    )

    # All registered encoders & decoders must be dynamically covered
    assert {r.arch for r in report.encoder_rows} == set(ge.list_encoders())
    assert {r.arch for r in report.decoder_rows} == set(ge.list_decoders())
    assert len(report.encoder_rows) >= 8
    assert len(report.decoder_rows) >= 2
    assert len(report.layer_rows) == 6

    # Strict quantitative scientific & numerical gates
    assert report.summary.all_zero_float64_compliant is True
    assert report.se3_invariance_max_err < 1e-4
    assert report.summary.so3_vector_equivariance_max_err < 1e-4
    assert report.summary.padding_mask_invariance_max_err < 1e-4

    # Causal masking, asserted per decoder rather than on the aggregate maximum.
    #
    # Every dense decoder must show bit-for-bit zero leakage. Sparse Mixture-of-Experts
    # decoders are held to a float32 round-off bound instead: a token's expert choice
    # depends only on its own hidden state (so no information flows backwards in time),
    # but perturbing late tokens changes how many tokens each expert receives, which
    # changes that expert's GEMM shape and perturbs results by ~1e-8. That is inherent to
    # token-choice routing, not a masking defect.
    for row in report.decoder_rows:
        if row.modality != "1d_token_autoregressive":
            continue
        if "moe" in row.arch:
            assert row.causal_leakage_err < 1e-6, (
                f"{row.arch} causal leakage {row.causal_leakage_err:.3e} exceeds the "
                "float32 round-off bound expected from MoE expert-batch composition"
            )
        else:
            assert row.causal_leakage_err == 0.0, (
                f"dense decoder {row.arch} leaked {row.causal_leakage_err:.3e}"
            )
    assert report.summary.adaln_zero_identity_err == 0.0
    assert report.summary.rope_relative_shift_err < 1e-5
    assert report.summary.se3_block_equivariance_err < 1e-4
    assert report.summary.min_pocket_conditioning_kl > 0.0
    assert report.pose_discrimination_auroc >= 0.90
    assert report.pose_discrimination.native_posebusters_pass_rate >= 0.90
    assert (
        report.pose_discrimination.native_posebusters_pass_rate
        > report.pose_discrimination.decoy_posebusters_pass_rate
    )

    for dec_row in report.decoder_rows:
        assert dec_row.tokens_per_sec > 0.0
        if dec_row.modality == "1d_token_autoregressive":
            assert dec_row.smiles_validity >= 0.80
        else:
            assert dec_row.smiles_validity >= 0.50
        assert dec_row.selfies_validity >= 0.95
        assert 0.0 < dec_row.qed_mean <= 1.0
        assert 1.0 <= dec_row.sa_mean <= 10.0

    # JSON & publication Markdown export verification
    raw_json = report.to_json()
    parsed = json.loads(raw_json)
    assert parsed["suite_version"] == ge.__version__
    assert parsed["metadata"]["ci_profile"] == "google_swe_validation"
    assert len(parsed["encoder_rows"]) == len(report.encoder_rows)

    md = report.to_markdown()
    assert md == format_benchmark_markdown_table(report)
    assert "## 1. 3D Pocket Encoder Symmetry & Throughput Benchmark" in md
    assert "## 2. Molecular Decoder Causality, Conditioning & Chemical Validity Benchmark" in md
    assert "## 3. Reusable Neural & `SE(3)` Geometric Transformer Layer Audit" in md
    assert "## 4. 3D PoseBusters & Native-vs-Decoy Pose Discrimination" in md
    for enc_name in ge.list_encoders():
        assert f"`{enc_name}`" in md
