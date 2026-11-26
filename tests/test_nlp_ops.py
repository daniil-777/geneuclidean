"""Comprehensive CPU & Apple Silicon MPS verification for GenEuclidean Chemical NLP Ops.

Proves every mathematical invariant, chemical syntax constraint, numerical equivalence bound,
and zero-fallback MPS execution contract across ``geneuclidean.nlp`` and
``geneuclidean.models.nlp_ops``:
  - Cold import latency (< 5 ms) & zero eager heavy imports (``heavy == []``) for
    ``import geneuclidean.nlp``.
  - ``AtomLevelSMILESTokenizer``: SMILES, SELFIES, SAFE dot fragments, special tokens,
    deterministic encode/decode, batch padding, and vocabulary serialization.
  - ``SMILESValenceGrammarMask``: blocks unmatched ``)``, consecutive invalid bonds (``==``,
    ``=#``, ``(#``), premature ``<eos>`` with open rings/branches, self-loop rings (``C11``),
    and halogen over-bonding while strictly permitting dopamine, haloperidol, risperidone,
    carbonyl branches, and SAFE fragments.
  - ``RotaryPositionEmbedding`` & ``apply_rotary_pos_emb``: exact relative-position inner
    product invariance (``< 1e-5``) under standard, NTK-aware, and YaRN scaling.
  - ``GroupedQueryAttention`` & ``KVCache``: step-by-step incremental decoding matches
    full-sequence forward to ``< 1e-5`` across MHA, GQA, and MQA on CPU and MPS.
  - ``LoRALinear``, ``inject_lora_adapters``, ``merge_lora_adapters``: merged weights match
    unmerged ``base + scaling * B @ A`` to ``< 1e-6``.
  - ``PocketCrossAttention``, ``PocketPerceiverBridge``, ``PocketConditionedChemicalLM``:
    zero-leak masking, permutation invariance, forward/backward on MPS, zero ``float64``
    buffers/params, contrastive CFG logits, and deterministic KV-cached beam search.
  - ``dpo_preference_loss`` (DPO / cDPO / IPO) & ``grpo_policy_loss`` (zero-mean group
    advantages, PPO clipping, Schulman k3 KL, finite gradients).
  - ``DiscreteMaskedDiffusionStep``: exact ``t=0 -> 0%`` & ``t=1 -> 100%`` mask monotonicity,
    SAFE scaffold pinning, MDLM loss grads, and reverse unmasking into valid non-mask IDs.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
from torch import Tensor

import geneuclidean.nlp as nlp

DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]

DOPAMINE_SMILES = "NCCc1ccc(O)c(O)c1"
HALOPERIDOL_SMILES = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
RISPERIDONE_SMILES = "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"


def _assert_no_float64(module: nn.Module) -> None:
    """Assert that every parameter and buffer in ``module`` avoids ``torch.float64``."""
    for name, param in module.named_parameters():
        assert param.dtype != torch.float64, f"float64 parameter leak: {name}"
    for name, buf in module.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer leak: {name}"


# ---------------------------------------------------------------------------
# 1. Cold Import Latency & PEP 562 Lazy Public Facade
# ---------------------------------------------------------------------------


def test_cold_import_geneuclidean_nlp_zero_heavy_and_fast() -> None:
    """``import geneuclidean.nlp`` completes in < 5 ms with ZERO heavy imports."""
    probe = """
import json, sys, time
import geneuclidean  # pre-warm package root so we time the nlp facade cleanly
t0 = time.perf_counter()
import geneuclidean.nlp as nlp
dt_ms = (time.perf_counter() - t0) * 1000.0
heavy_check = ("torch", "rdkit", "transformers", "peft", "sklearn", "scipy", "matplotlib", "e3nn")
loaded_heavy = [m for m in heavy_check if m in sys.modules]
print(json.dumps({
    "dt_ms": round(dt_ms, 3),
    "loaded_heavy": loaded_heavy,
    "n_primitives": len(nlp.list_nlp_primitives()),
    "has_gqa": "GroupedQueryAttention" in dir(nlp),
    "has_grpo": "grpo_policy_loss" in dir(nlp),
}))
"""
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout.strip())
    assert info["loaded_heavy"] == [], (
        f"Heavy modules leaked on cold import: {info['loaded_heavy']}"
    )
    assert info["dt_ms"] < 5.0, f"Cold import took {info['dt_ms']:.3f} ms >= 5.0 ms"
    assert info["n_primitives"] >= 35
    assert info["has_gqa"] is True
    assert info["has_grpo"] is True


def test_nlp_facade_resolves_all_exports_and_rejects_unknown() -> None:
    """Every symbol in ``geneuclidean.nlp.__all__`` resolves; unknown raises AttributeError."""
    for name in nlp.__all__:
        assert hasattr(nlp, name), f"Missing lazy nlp symbol: {name}"
    with pytest.raises(AttributeError, match="no attribute"):
        missing = "nonexistent_chemical_nlp_op_xyz"
        _ = getattr(nlp, missing)


# ---------------------------------------------------------------------------
# 2. Chemical Tokenization & Stateful SMILES Valence Grammar Constraints
# ---------------------------------------------------------------------------


def test_atom_level_smiles_tokenizer_smiles_selfies_safe_and_batch_padding() -> None:
    """Verify deterministic SMILES/SELFIES/SAFE tokenization, special tokens, and padding."""
    tok = nlp.AtomLevelSMILESTokenizer()

    # Fixed special token IDs
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.mask_id == 3
    assert tok.sep_id == 4
    assert tok.unk_id == 5

    # Exact SMILES roundtrip (halogens Cl/Br, stereocenters, aromatic NH, multi-digit ring)
    complex_smi = "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21"
    toks = tok.tokenize(complex_smi)
    assert "Cl" in toks and "[C@H]" in toks and "[C@@H]" in toks
    encoded = tok.encode(complex_smi, add_bos=True, add_eos=True)
    assert encoded[0] == tok.bos_id and encoded[-1] == tok.eos_id
    assert tok.decode(encoded) == complex_smi

    # SAFE dot-separated fragment roundtrip (including multi-digit %10 attachment)
    safe_str = "c1ccc%10cc1.CC%10"
    assert "." in tok.tokenize(safe_str) and "%10" in tok.tokenize(safe_str)
    assert tok.decode(tok.encode(safe_str)) == safe_str

    # SELFIES bracket-token support via deterministic fit()
    selfies_corpus = [
        "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
        "[C][C][=Branch1][C][=O][O]",
    ]
    sf_tok = nlp.AtomLevelSMILESTokenizer().fit(selfies_corpus)
    assert sf_tok.tokenize(selfies_corpus[0]) == [
        "[C]",
        "[=C]",
        "[C]",
        "[=C]",
        "[C]",
        "[=C]",
        "[Ring1]",
        "[=Branch1]",
    ]
    assert sf_tok.decode(sf_tok.encode(selfies_corpus[0])) == selfies_corpus[0]

    # Batch padding, pad_to_multiple_of, and JSON serialization roundtrip
    batch_texts = [DOPAMINE_SMILES, HALOPERIDOL_SMILES, RISPERIDONE_SMILES]
    ids, mask = tok.batch_encode(batch_texts, pad_to_multiple_of=8)
    assert ids.dtype == torch.long and mask.dtype == torch.bool
    assert ids.shape[0] == 3 and ids.shape[1] % 8 == 0
    assert tok.batch_decode(ids) == batch_texts
    assert bool((ids[~mask] == tok.pad_id).all().item())

    restored = nlp.AtomLevelSMILESTokenizer.from_dict(tok.to_dict())
    assert restored.tokens == tok.tokens
    assert restored.encode(RISPERIDONE_SMILES) == tok.encode(RISPERIDONE_SMILES)


@pytest.mark.parametrize("device", DEVICES)
def test_smiles_valence_grammar_mask_drugs_and_violations(device: str) -> None:
    """Block unmatched ')', invalid consecutive bonds, and open-ring EOS; allow real drugs."""
    dev = torch.device(device)
    tok = nlp.AtomLevelSMILESTokenizer()
    grammar = nlp.SMILESValenceGrammarMask(tok)

    # 1. Valid real CNS drugs, carbonyl branches, fused rings, and SAFE fragments MUST pass 100%!
    valid_molecules = [
        DOPAMINE_SMILES,
        HALOPERIDOL_SMILES,
        RISPERIDONE_SMILES,
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CS(=O)(=O)Nc1ccc(F)cc1",  # Sulfonamide
        "C(/F)=C/F",  # E/Z alkene branch
        "c1ccc2cc1.CC2",  # SAFE fragment string
    ]
    for smi in valid_molecules:
        assert grammar.validate_sequence(smi), f"Grammar mask falsely rejected valid SMILES: {smi}"

    # 2. Invalid syntax & valence sequences MUST be rejected!
    invalid_molecules = [
        "CCC)",  # Unmatched closing parenthesis
        "CC(O",  # Unclosed parenthesis before EOS
        "c1ccccc",  # Unclosed ring 1 before EOS
        "CC==CC",  # Consecutive double bond ==
        "C(#N)C",  # Forbidden (# branch triple bond
        "C()C",  # Empty branch ()
        "=CC",  # Leading bond token
        "1CCC1",  # Leading ring digit
        "C11CC",  # Same-atom ring self-loop
        "CCCl=O",  # Over-bonded monovalent halogen Cl=
        "C=F",  # Double bond to monovalent F
        "O#C",  # Triple bond on neutral unbracketed O
        "CC=",  # Dangling bond before EOS
    ]
    for bad in invalid_molecules:
        assert not grammar.validate_sequence(bad), f"Grammar mask failed to block: {bad}"

    # 3. Direct logits masking on CPU & MPS: verify exact -inf placement & stateful == history
    tid_close = tok.token_to_id[")"]
    tid_eq = tok.token_to_id["="]
    tid_triple = tok.token_to_id["#"]
    tid_eos = tok.eos_id
    tid_c = tok.token_to_id["C"]
    tid_o = tok.token_to_id["O"]

    # Prefix 0: "CC=" (must block '=', '#', ')', '<eos>'; must allow 'O' and 'C')
    # Prefix 1: "C1C(" (must block ')', '#', '<eos>'; must allow '=', 'O', 'C')
    # Prefix 2: "CC"   (must block ')' at depth 0; must allow '<eos>', 'C', '=')
    prefixes = [
        tok.encode("CC=", add_bos=True, add_eos=False),
        tok.encode("C1C(", add_bos=True, add_eos=False),
        tok.encode("CC", add_bos=True, add_eos=False),
    ]
    max_p = max(len(p) for p in prefixes)
    hist = torch.full((3, max_p), tok.pad_id, dtype=torch.long, device=dev)
    for i, p in enumerate(prefixes):
        hist[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=dev)

    logits = torch.zeros((3, tok.vocab_size), dtype=torch.float32, device=dev)
    masked_hist = grammar.mask_logits(logits, token_history=hist)

    # Row 0 ("CC="): blocks ==, =#, =), EOS; allows O, C
    assert float(masked_hist[0, tid_eq].item()) == float("-inf")
    assert float(masked_hist[0, tid_triple].item()) == float("-inf")
    assert float(masked_hist[0, tid_close].item()) == float("-inf")
    assert float(masked_hist[0, tid_eos].item()) == float("-inf")
    assert float(masked_hist[0, tid_o].item()) == 0.0
    assert float(masked_hist[0, tid_c].item()) == 0.0

    # Row 1 ("C1C("): blocks (), (#, EOS; allows (=, O, C
    assert float(masked_hist[1, tid_close].item()) == float("-inf")
    assert float(masked_hist[1, tid_triple].item()) == float("-inf")
    assert float(masked_hist[1, tid_eos].item()) == float("-inf")
    assert float(masked_hist[1, tid_eq].item()) == 0.0
    assert float(masked_hist[1, tid_o].item()) == 0.0

    # Row 2 ("CC"): blocks unmatched ')', allows EOS
    assert float(masked_hist[2, tid_close].item()) == float("-inf")
    assert float(masked_hist[2, tid_eos].item()) == 0.0

    # Verify stateful step-by-step mask matches history-based mask bit-for-bit!
    grammar.reset(batch_size=3)
    for col in range(max_p):
        grammar.step(hist[:, col])
    masked_stateful = grammar.mask_logits(logits)
    assert torch.equal(masked_stateful, masked_hist)


# ---------------------------------------------------------------------------
# 3. RMSNorm, RoPE (Standard / NTK / YaRN), SwiGLU, GQA + KV-Cache Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scaling_type", ["none", "ntk", "yarn"])
def test_rope_exact_relative_inner_product_invariance(device: str, scaling_type: str) -> None:
    """RoPE preserves <R(m+d)q, R(n+d)k> == <R(m)q, R(n)k> to < 1e-5 for all scaling modes."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(11)
    head_dim = 32
    rope = nlp.RotaryPositionEmbedding(
        head_dim=head_dim,
        max_seq_len=64,
        base=10000.0,
        scaling_type=scaling_type,  # type: ignore[arg-type]
        scaling_factor=2.0 if scaling_type != "none" else 1.0,
        original_max_seq_len=64,
    ).to(dev)
    _assert_no_float64(rope)

    # Scale q, k by head_dim^-0.25 so <q, k> equals the exact scaled attention logit (var = 1)
    scale = head_dim**-0.25
    q = (torch.randn((2, 4, 1, head_dim), generator=gen, dtype=torch.float32) * scale).to(dev)
    k = (torch.randn((2, 4, 1, head_dim), generator=gen, dtype=torch.float32) * scale).to(dev)

    # Compare relative distance (n - m = 7) across shifts (including shift > max_seq_len)
    m_base, n_base = 3, 10
    cos_m0, sin_m0 = rope(1, offset=m_base, device=dev)
    cos_n0, sin_n0 = rope(1, offset=n_base, device=dev)
    q0, _ = nlp.apply_rotary_pos_emb(q, q, cos_m0, sin_m0)
    _, k0 = nlp.apply_rotary_pos_emb(k, k, cos_n0, sin_n0)
    ip_ref = (q0 * k0).sum(dim=-1)

    for shift in (1, 17, 45, 90):  # shift=90 exceeds max_seq_len=64 -> dynamic extension
        cos_m, sin_m = rope(1, offset=m_base + shift, device=dev)
        cos_n, sin_n = rope(1, offset=n_base + shift, device=dev)
        qm, _ = nlp.apply_rotary_pos_emb(q, q, cos_m, sin_m)
        _, kn = nlp.apply_rotary_pos_emb(k, k, cos_n, sin_n)
        ip_shifted = (qm * kn).sum(dim=-1)
        err = float((ip_shifted - ip_ref).abs().max().item())
        assert err < 1e-5, (
            f"RoPE ({scaling_type}) invariance err {err:.2e} >= 1e-5 at shift={shift}"
        )

    # Verify RMSNorm fp32 stabilization & finite gradients on CPU and MPS
    norm = nlp.RMSNorm(head_dim, init_scale=1.25).to(dev)
    x = (
        torch.randn((2, 8, head_dim), generator=gen, dtype=torch.float32)
        .to(dev)
        .requires_grad_(True)
    )
    y = norm(x)
    rms_y = torch.sqrt(y.pow(2).mean(dim=-1))
    assert float((rms_y - 1.25).abs().max().item()) < 1e-4
    y.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all().item())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_kv_heads", [8, 2, 1])  # MHA, GQA, MQA
def test_gqa_incremental_kv_cache_matches_full_sequence(device: str, n_kv_heads: int) -> None:
    """Step-by-step and chunked KVCache GQA/MQA/MHA matches full forward to < 1e-5."""
    dev = torch.device(device)
    torch.manual_seed(23)
    d_model = 64
    n_heads = 8
    seq_len = 14
    bsz = 2

    rope = nlp.RotaryPositionEmbedding(head_dim=d_model // n_heads, max_seq_len=64).to(dev)
    attn = nlp.GroupedQueryAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, dropout=0.0
    ).to(dev)
    attn.eval()
    _assert_no_float64(attn)

    gen = torch.Generator(device="cpu").manual_seed(29)
    x = torch.randn((bsz, seq_len, d_model), generator=gen, dtype=torch.float32).to(dev)

    with torch.no_grad():
        full_out, full_cache = attn(x, rope=rope, is_causal=True)
        assert full_cache.seq_len == seq_len

        # 1. Token-by-token incremental decoding with LayerKVCache
        cache: nlp.LayerKVCache | None = None
        step_outs: list[Tensor] = []
        for t in range(seq_len):
            out_t, cache = attn(x[:, t : t + 1, :], rope=rope, kv_cache=cache, is_causal=True)
            step_outs.append(out_t)
        inc_out = torch.cat(step_outs, dim=1)
        err_step = float((inc_out - full_out).abs().max().item())
        assert err_step < 1e-5, (
            f"KV-cache step parity err {err_step:.2e} >= 1e-5 ({device}, kv={n_kv_heads})"
        )

        # 2. Chunked prefill (6 tokens + 8 tokens) with LayerKVCache
        out_c1, cache_c = attn(x[:, :6, :], rope=rope, is_causal=True)
        out_c2, cache_c = attn(x[:, 6:, :], rope=rope, kv_cache=cache_c, is_causal=True)
        chunk_out = torch.cat((out_c1, out_c2), dim=1)
        err_chunk = float((chunk_out - full_out).abs().max().item())
        assert err_chunk < 1e-5, f"KV-cache chunk parity err {err_chunk:.2e} >= 1e-5"

    # 3. Bidirectional mode + all-masked row gradient safety
    x_g = x.clone().requires_grad_(True)
    pad_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=dev)
    pad_mask[1, :] = False  # Entire row masked!
    bi_out, _ = attn(x_g, rope=rope, attention_mask=pad_mask, is_causal=False)
    assert bool(torch.isfinite(bi_out).all().item())
    assert float(bi_out[1].abs().max().item()) == 0.0
    bi_out.sum().backward()
    assert x_g.grad is not None and bool(torch.isfinite(x_g.grad).all().item())


# ---------------------------------------------------------------------------
# 4. LoRALinear Exact Merge Equivalence (< 1e-6) & Adapter Injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_lora_linear_merge_equivalence_and_model_injection(device: str) -> None:
    """Merged LoRALinear equals unmerged base + scaling*(B@A) to < 1e-6 on CPU & MPS."""
    dev = torch.device(device)
    torch.manual_seed(41)
    gen = torch.Generator(device="cpu").manual_seed(43)

    base = nn.Linear(48, 64, bias=True).to(dev)
    lora = nlp.LoRALinear.from_linear(base, rank=6, alpha=12.0, dropout=0.0, freeze_base=True).to(
        dev
    )
    _assert_no_float64(lora)
    assert not lora.base_layer.weight.requires_grad
    assert lora.lora_a.requires_grad and lora.lora_b.requires_grad

    # Set non-zero lora_b so adaptation delta is non-trivial!
    with torch.no_grad():
        lora.lora_b.copy_(
            torch.randn(lora.lora_b.shape, generator=gen, dtype=torch.float32).to(dev) * 0.1
        )

    x = torch.randn((4, 12, 48), generator=gen, dtype=torch.float32).to(dev)
    y_unmerged = lora(x)
    y_base_only = nn.functional.linear(x, lora.base_layer.weight, lora.base_layer.bias)
    assert float((y_unmerged - y_base_only).abs().max().item()) > 1e-2

    # Merge in-place and prove exact equivalence < 1e-6
    lora.merge()
    assert lora.merged is True
    y_merged = lora(x)
    err_merge = float((y_merged - y_unmerged).abs().max().item())
    assert err_merge < 1e-6, f"LoRA merge error {err_merge:.2e} >= 1e-6 on {device}"

    # Standalone to_linear() export matches to < 1e-6
    plain_lin = lora.to_linear()
    err_plain = float((plain_lin(x) - y_unmerged).abs().max().item())
    assert err_plain < 1e-6

    # Unmerge restores exact unmerged output and gradient flow ONLY to LoRA params
    lora.unmerge()
    assert lora.merged is False
    assert float((lora(x) - y_unmerged).abs().max().item()) < 1e-6
    lora(x).pow(2).sum().backward()
    assert lora.base_layer.weight.grad is None
    assert lora.lora_a.grad is not None and bool(torch.isfinite(lora.lora_a.grad).all().item())
    assert lora.lora_b.grad is not None and bool(torch.isfinite(lora.lora_b.grad).all().item())

    # End-to-end inject_lora_adapters & merge_lora_adapters(unwrap=True) on Chemical LM
    lm = nlp.PocketConditionedChemicalLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, conditioning_mode="none"
    ).to(dev)
    lm.eval()
    injected = nlp.inject_lora_adapters(lm, target_modules=("q_proj", "v_proj"), rank=4, alpha=8.0)
    assert len(injected) == 4  # 2 layers x (q_proj, v_proj)
    with torch.no_grad():
        for mod in lm.modules():
            if isinstance(mod, nlp.LoRALinear):
                mod.lora_b.copy_(
                    torch.randn(mod.lora_b.shape, generator=gen, dtype=torch.float32).to(dev) * 0.05
                )

    tok_ids = torch.randint(1, 64, (2, 10), generator=gen, dtype=torch.long).to(dev)
    logits_lora = lm(tok_ids).logits
    merged_paths = nlp.merge_lora_adapters(lm, unwrap=True)
    assert merged_paths == injected
    assert not any(isinstance(m, nlp.LoRALinear) for m in lm.modules())
    logits_merged = lm(tok_ids).logits
    assert float((logits_merged - logits_lora).abs().max().item()) < 1e-5


# ---------------------------------------------------------------------------
# 5. PocketCrossAttention, PocketPerceiverBridge & PocketConditionedChemicalLM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_pocket_cross_attn_and_perceiver_bridge_zero_leak_and_perm_invariance(device: str) -> None:
    """Verify zero-leak masking, SDPA/explicit parity, and Perceiver permutation invariance."""
    dev = torch.device(device)
    torch.manual_seed(51)
    gen = torch.Generator(device="cpu").manual_seed(53)

    bsz, l_lig, n_pocket, d_in, d_out, n_q = 2, 10, 18, 48, 64, 8
    q_lig = torch.randn((bsz, l_lig, d_out), generator=gen, dtype=torch.float32).to(dev)
    pocket = torch.randn((bsz, n_pocket, d_in), generator=gen, dtype=torch.float32).to(dev)
    kpm = torch.zeros((bsz, n_pocket), dtype=torch.bool, device=dev)  # True = PAD
    kpm[0, 14:] = True
    kpm[1, 10:] = True

    pocket_corrupt = pocket.clone()
    pocket_corrupt[kpm] = 9999.0

    cross = nlp.PocketCrossAttention(
        d_model=d_out, d_pocket=d_in, n_heads=4, n_kv_heads=2, gated=True
    ).to(dev)
    cross.eval()
    _assert_no_float64(cross)

    with torch.no_grad():
        out_fast = cross(q_lig, pocket, key_padding_mask=kpm, return_attn=False)
        out_corrupt = cross(q_lig, pocket_corrupt, key_padding_mask=kpm, return_attn=False)
        out_attn, attn_w = cross(q_lig, pocket, pocket_mask=~kpm, return_attn=True)
        assert float((out_fast - out_corrupt).abs().max().item()) == 0.0
        assert float((out_fast - out_attn).abs().max().item()) < 1e-5
        assert (
            float(attn_w[:, :, :, :][kpm[:, None, None, :].expand_as(attn_w)].abs().max().item())
            == 0.0
        )

    # All-padded row produces exact 0.0 output & finite gradients
    kpm_all = kpm.clone()
    kpm_all[1, :] = True
    q_g = q_lig.clone().requires_grad_(True)
    p_g = pocket.clone().requires_grad_(True)
    out_all = cross(q_g, p_g, key_padding_mask=kpm_all)
    assert bool(torch.isfinite(out_all).all().item())
    assert float(out_all[1].abs().max().item()) == 0.0
    out_all.sum().backward()
    assert q_g.grad is not None and bool(torch.isfinite(q_g.grad).all().item())
    assert p_g.grad is not None and bool(torch.isfinite(p_g.grad).all().item())

    # PocketPerceiverBridge: [B, N, D_in] -> [B, Q, D_out], zero-leak & permutation invariant!
    bridge = nlp.PocketPerceiverBridge(
        d_in=d_in, d_out=d_out, num_queries=n_q, num_layers=2, n_heads=4
    ).to(dev)
    bridge.eval()
    _assert_no_float64(bridge)

    with torch.no_grad():
        lat_ref = bridge(pocket, key_padding_mask=kpm)
        lat_corrupt = bridge(pocket_corrupt, key_padding_mask=kpm)
        assert lat_ref.shape == (bsz, n_q, d_out)
        assert float((lat_ref - lat_corrupt).abs().max().item()) == 0.0

        # Permute all pocket atoms & mask together -> exact permutation invariance!
        perm = torch.randperm(n_pocket, generator=gen).to(dev)
        lat_perm = bridge(pocket[:, perm, :], key_padding_mask=kpm[:, perm])
        err_perm = float((lat_perm - lat_ref).abs().max().item())
        assert err_perm < 1e-5, f"Perceiver permutation err {err_perm:.2e} >= 1e-5 on {device}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("conditioning_mode", ["cross_attn", "prefix"])
def test_pocket_conditioned_chemical_lm_mps_train_and_kv_cache(
    device: str, conditioning_mode: str
) -> None:
    """End-to-end PocketConditionedChemicalLM forward/backward & KV-cache parity on CPU & MPS."""
    dev = torch.device(device)
    torch.manual_seed(61)
    gen = torch.Generator(device="cpu").manual_seed(67)

    tok = nlp.AtomLevelSMILESTokenizer()
    ids, _ = tok.batch_encode([DOPAMINE_SMILES, HALOPERIDOL_SMILES], device=dev)
    bsz, seq_len = ids.shape
    pocket = torch.randn((bsz, 16, 48), generator=gen, dtype=torch.float32).to(dev)
    kpm = torch.zeros((bsz, 16), dtype=torch.bool, device=dev)
    kpm[1, 12:] = True

    lm = nlp.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_pocket=48,
        conditioning_mode=conditioning_mode,  # type: ignore[arg-type]
        use_perceiver_bridge=True,
        num_pocket_queries=6,
        perceiver_layers=1,
        rope_scaling_type="ntk",
        rope_scaling_factor=1.5,
    ).to(dev)
    _assert_no_float64(lm)

    # 1. Forward + backward pass with shifted CE loss
    lm.train()
    out = lm(ids, pocket_tokens=pocket, pocket_key_padding_mask=kpm, labels=ids)
    assert out.logits.shape == (bsz, seq_len, tok.vocab_size)
    assert out.loss is not None and bool(torch.isfinite(out.loss).item())
    out.loss.backward()
    grad_norms = [float(p.grad.abs().max().item()) for p in lm.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(torch.isfinite(torch.tensor(g)) for g in grad_norms)

    # 2. Exact incremental KV-cache decoding matches full-sequence logits to < 1e-5
    lm.eval()
    with torch.no_grad():
        full_logits = lm(ids, pocket_tokens=pocket, pocket_key_padding_mask=kpm).logits
        cache: nlp.KVCache | None = None
        step_logits: list[Tensor] = []
        for t in range(seq_len):
            step_out = lm(
                ids[:, t : t + 1],
                pocket_tokens=pocket,
                pocket_key_padding_mask=kpm,
                kv_cache=cache,
                use_cache=True,
            )
            cache = step_out.kv_cache
            step_logits.append(step_out.logits)
        inc_logits = torch.cat(step_logits, dim=1)
        err = float((inc_logits - full_logits).abs().max().item())
        assert err < 1e-5, f"ChemicalLM ({conditioning_mode}) KV-cache err {err:.2e} >= 1e-5"


# ---------------------------------------------------------------------------
# 6. Contrastive CFG, Top-k / Top-p Filtering & Deterministic Beam Search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_contrastive_cfg_top_k_top_p_and_deterministic_beam_search(device: str) -> None:
    """Verify top-k/top-p stability, contrastive CFG equation, and deterministic KV-beam search."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(71)

    # 1. Temperature + top_k + top_p + min_p numerical stability
    raw_logits = torch.randn((3, 50), generator=gen, dtype=torch.float32).to(dev) * 2.0
    filt_k = nlp.apply_temperature_top_k_top_p(raw_logits, temperature=0.8, top_k=5)
    assert bool((torch.isfinite(filt_k).sum(dim=-1) == 5).all().item())
    filt_p = nlp.apply_temperature_top_k_top_p(raw_logits, temperature=1.0, top_p=0.5, min_p=0.1)
    assert bool((torch.isfinite(filt_p).sum(dim=-1) >= 1).all().item())
    probs = torch.softmax(filt_p, dim=-1)
    assert bool(torch.isfinite(probs).all().item())
    assert float((probs.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    # 2. Contrastive multi-conformation CFG logits exact equation
    on1 = torch.randn((2, 40), generator=gen, dtype=torch.float32).to(dev)
    on2 = torch.randn((2, 40), generator=gen, dtype=torch.float32).to(dev)
    off = torch.randn((2, 40), generator=gen, dtype=torch.float32).to(dev)
    uncond = torch.randn((2, 40), generator=gen, dtype=torch.float32).to(dev)

    guided_seq = nlp.apply_contrastive_cfg_logits(
        [on1, on2], uncond, off_target_logits=off, on_weights=[0.75, 0.25], w_on=1.8, w_off=0.6
    )
    guided_stack = nlp.apply_contrastive_cfg_logits(
        torch.stack((on1, on2), dim=0),
        uncond,
        off_target_logits=off,
        on_weights=[3.0, 1.0],
        w_on=1.8,
        w_off=0.6,
    )
    expected = uncond + 1.8 * (0.75 * on1 + 0.25 * on2 - uncond) - 0.6 * (off - uncond)
    assert float((guided_seq - expected).abs().max().item()) < 1e-6
    assert float((guided_stack - expected).abs().max().item()) < 1e-6

    # 3. Deterministic KV-cached beam search + grammar mask + cached vs uncached exact parity!
    torch.manual_seed(79)
    tok = nlp.AtomLevelSMILESTokenizer()
    lm = nlp.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_pocket=32,
        conditioning_mode="cross_attn",
        use_perceiver_bridge=False,
    ).to(dev)
    lm.eval()
    pocket_1 = torch.randn((1, 10, 32), generator=gen, dtype=torch.float32).to(dev)
    grammar = nlp.SMILESValenceGrammarMask(tok)

    res_cached_a = nlp.beam_search_decode(
        lm,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        pocket_tokens=pocket_1,
        beam_width=4,
        max_new_tokens=12,
        length_penalty=0.6,
        grammar_mask=grammar,
        use_cache=True,
    )
    res_cached_b = nlp.beam_search_decode(
        lm,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        pocket_tokens=pocket_1,
        beam_width=4,
        max_new_tokens=12,
        length_penalty=0.6,
        grammar_mask=grammar,
        use_cache=True,
    )
    res_uncached = nlp.beam_search_decode(
        lm,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        pocket_tokens=pocket_1,
        beam_width=4,
        max_new_tokens=12,
        length_penalty=0.6,
        grammar_mask=grammar,
        use_cache=False,
    )
    # Determinism: exact bit-for-bit identical sequences & scores
    assert res_cached_a.sequences == res_cached_b.sequences
    assert res_cached_a.scores == res_cached_b.scores
    assert res_cached_a.scores == sorted(res_cached_a.scores, reverse=True)
    # KV-cached matches uncached beam search
    assert res_cached_a.sequences == res_uncached.sequences
    for s_c, s_u in zip(res_cached_a.scores, res_uncached.scores, strict=True):
        assert abs(s_c - s_u) < 1e-4
    # Every beam respects grammar (no illegal consecutive bonds or negative paren depth)
    for seq in res_cached_a.sequences:
        if seq[-1] == tok.eos_id:
            assert grammar.validate_sequence(seq)


# ---------------------------------------------------------------------------
# 7. Molecular RL Alignment: DPO / cDPO / IPO & Critic-Free GRPO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_dpo_and_grpo_losses_math_and_finite_grads(device: str) -> None:
    """Verify DPO monotonicity, GRPO zero-mean advantages, PPO clipping, KL, and finite grads."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(83)

    # 1. DPO / cDPO / IPO: loss strictly decreases as chosen_logp rises relative to rejected_logp
    ref_c = torch.tensor([-12.0, -10.5, -14.0, -9.0], device=dev, dtype=torch.float32)
    ref_r = torch.tensor([-12.0, -10.5, -14.0, -9.0], device=dev, dtype=torch.float32)
    pi_r = torch.tensor(
        [-12.5, -10.8, -14.2, -9.3], device=dev, dtype=torch.float32, requires_grad=True
    )

    pi_c_low = torch.tensor(
        [-13.5, -11.5, -15.0, -10.0], device=dev, dtype=torch.float32, requires_grad=True
    )
    pi_c_high = torch.tensor(
        [-10.5, -9.0, -12.0, -7.5], device=dev, dtype=torch.float32, requires_grad=True
    )

    for loss_type, ls in (("dpo", 0.0), ("cdpo", 0.1), ("ipo", 0.0)):
        out_low = nlp.dpo_preference_loss(
            pi_c_low,
            pi_r,
            ref_c,
            ref_r,
            beta=0.2,
            label_smoothing=ls,
            loss_type=loss_type,  # type: ignore[arg-type]
        )
        out_high = nlp.dpo_preference_loss(
            pi_c_high,
            pi_r,
            ref_c,
            ref_r,
            beta=0.2,
            label_smoothing=ls,
            loss_type=loss_type,  # type: ignore[arg-type]
        )
        assert float(out_high.loss.item()) < float(out_low.loss.item()), (
            f"{loss_type} loss did not decrease when chosen_logp increased"
        )
        assert float(out_high.accuracy.item()) == 1.0
        assert float(out_high.reward_margin.item()) > 0.0

    # Finite gradients through DPO loss
    dpo_grad_out = nlp.dpo_preference_loss(pi_c_high, pi_r, ref_c, ref_r, beta=0.1)
    torch.autograd.backward(dpo_grad_out.loss)
    assert pi_c_high.grad is not None and bool(torch.isfinite(pi_c_high.grad).all().item())
    assert pi_r.grad is not None and bool(torch.isfinite(pi_r.grad).all().item())
    # Chosen gradient pushes chosen_logp UP (negative dL/d_chosen_logp)
    assert bool((pi_c_high.grad < 0.0).all().item())
    assert bool((pi_r.grad > 0.0).all().item())

    # Verify sequence_logprobs_from_logits end-to-end integration with DPO
    logits_dummy = (
        torch.randn((4, 8, 32), generator=gen, dtype=torch.float32).to(dev).requires_grad_(True)
    )
    targets_dummy = torch.randint(1, 32, (4, 8), generator=gen, dtype=torch.long).to(dev)
    seq_lp, tok_lp = nlp.sequence_logprobs_from_logits(logits_dummy, targets_dummy, shift=True)
    assert seq_lp.shape == (4,) and tok_lp.shape == (4, 7)
    assert bool((seq_lp <= 0.0).all().item())

    # 2. GRPO: zero-mean group-normalized advantages + PPO clipping + Schulman k3 KL + finite grads
    bsz, group_g, seq_t = 3, 6, 10
    rewards = torch.tensor(
        [
            [0.2, 0.5, 0.9, 0.1, 0.7, 0.4],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Zero-variance constant-reward group!
        ],
        device=dev,
        dtype=torch.float32,
    )
    old_lp = (-torch.rand((bsz, group_g, seq_t), generator=gen, dtype=torch.float32) - 0.5).to(dev)
    ref_lp = old_lp.clone()
    # Perturb policy_logps far enough so importance ratios exceed 1 + clip_eps -> PPO clip triggers!
    delta = (torch.randn((bsz, group_g, seq_t), generator=gen, dtype=torch.float32) * 0.6).to(dev)
    policy_lp = (old_lp + delta).detach().requires_grad_(True)
    tok_mask = torch.ones((bsz, group_g, seq_t), dtype=torch.bool, device=dev)
    tok_mask[:, :, -2:] = False

    grpo = nlp.grpo_policy_loss(
        policy_lp,
        old_lp,
        rewards,
        ref_logps=ref_lp,
        mask=tok_mask,
        clip_eps=0.2,
        kl_beta=0.05,
    )
    # Exact zero-mean advantages per group (including constant-reward row 2!)
    group_means = grpo.advantages.mean(dim=-1)
    assert float(group_means.abs().max().item()) < 1e-6
    assert float(grpo.advantages[2].abs().max().item()) == 0.0
    # Non-constant groups have ~unit std
    group_stds = grpo.advantages[:2].std(dim=-1, unbiased=False)
    assert float((group_stds - 1.0).abs().max().item()) < 1e-3
    # PPO clipping active & Schulman k3 KL strictly positive when policy != ref
    assert float(grpo.clip_fraction.item()) > 0.1
    assert float(grpo.kl_penalty.item()) > 0.0

    # Zero KL when policy == ref
    grpo_same = nlp.grpo_policy_loss(
        old_lp.clone(), old_lp, rewards, ref_logps=ref_lp, mask=tok_mask, clip_eps=0.2, kl_beta=0.05
    )
    assert float(grpo_same.kl_penalty.item()) == 0.0
    assert float(grpo_same.clip_fraction.item()) == 0.0

    # Finite backward gradients (and zero grad on masked-out padded positions!)
    torch.autograd.backward(grpo.loss)
    assert policy_lp.grad is not None and bool(torch.isfinite(policy_lp.grad).all().item())
    assert float(policy_lp.grad[~tok_mask].abs().max().item()) == 0.0


# ---------------------------------------------------------------------------
# 8. Discrete Masked Diffusion (MDLM) & SAFE-Fragment Inpainting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_discrete_masked_diffusion_monotonicity_inpainting_and_grads(
    device: str, schedule: str
) -> None:
    """Verify t=0->0%, t=1->100% masking, scaffold pinning, MDLM loss, and reverse unmasking."""
    dev = torch.device(device)
    tok = nlp.AtomLevelSMILESTokenizer()
    mdlm = nlp.DiscreteMaskedDiffusionStep(
        mask_token_id=tok.mask_id,
        schedule=schedule,  # type: ignore[arg-type]
    ).to(dev)

    # Exact schedule boundaries & monotonicity
    ts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=dev, dtype=torch.float32)
    probs = mdlm.mask_probability(ts)
    assert float(probs[0].item()) == 0.0
    assert float(probs[-1].item()) == 1.0
    assert bool((probs[1:] > probs[:-1]).all().item())

    # Empirical forward corruption monotonicity over a large batch
    gen = torch.Generator(device="cpu").manual_seed(97)
    clean_large = torch.randint(6, tok.vocab_size, (32, 64), generator=gen, dtype=torch.long).to(
        dev
    )
    c0 = mdlm.forward_corrupt(clean_large, t=0.0, generator=gen)
    c1 = mdlm.forward_corrupt(clean_large, t=1.0, generator=gen)
    assert int((c0.noisy_ids == tok.mask_id).sum().item()) == 0
    assert bool(torch.equal(c0.noisy_ids, clean_large))
    assert int((c1.noisy_ids == tok.mask_id).sum().item()) == clean_large.numel()

    fracs = [
        float(
            (mdlm.forward_corrupt(clean_large, t=tv, generator=gen).noisy_ids == tok.mask_id)
            .float()
            .mean()
            .item()
        )
        for tv in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert fracs[0] == 0.0 and fracs[-1] == 1.0
    assert all(fracs[i] < fracs[i + 1] for i in range(len(fracs) - 1))

    # Bidirectional MDLM training loss + SAFE scaffold-pinned reverse unmasking on CPU & MPS
    torch.manual_seed(101)
    safe_ids, _ = tok.batch_encode(["c1ccc2cc1.CC2", "c1ccc2cc1.OC2"], device=dev)
    fixed_mask = torch.zeros_like(safe_ids, dtype=torch.bool)
    fixed_mask[:, :6] = True  # Pin `<bos>` + core aromatic fragment prefix!
    fixed_mask[:, -1] = True  # Pin `<eos>`

    corrupted = mdlm.forward_corrupt(safe_ids, t=1.0, fixed_mask=fixed_mask, generator=gen)
    assert bool(torch.equal(corrupted.noisy_ids[fixed_mask], safe_ids[fixed_mask]))
    assert bool((corrupted.noisy_ids[~fixed_mask] == tok.mask_id).all().item())

    lm = nlp.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        conditioning_mode="none",
    ).to(dev)
    logits = lm(corrupted.noisy_ids, is_causal=False).logits
    loss = mdlm.diffusion_loss(logits, safe_ids, corrupted.corrupt_mask, t=0.5)
    assert bool(torch.isfinite(loss).item())
    torch.autograd.backward(loss)

    # Zero-masked edge case (t=0.0): exact 0.0 loss with zero NaNs
    loss_zero = mdlm.diffusion_loss(
        logits, safe_ids, torch.zeros_like(corrupted.corrupt_mask), t=0.0
    )
    assert float(loss_zero.item()) == 0.0
    assert any(
        p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in lm.parameters()
    )

    # Reverse unmasking (confidence-guided AND exact posterior) recovers valid non-mask IDs
    for conf_mode in (True, False):
        inpainted = mdlm.sample_inpaint(
            lm,
            corrupted.noisy_ids,
            num_steps=4,
            temperature=0.9,
            top_k=15,
            confidence_remasking=conf_mode,
            forbidden_token_ids=(tok.pad_id, tok.bos_id),
            generator=gen,
        )
        # Pinned scaffold tokens remain untouched verbatim
        assert bool(torch.equal(inpainted[fixed_mask], safe_ids[fixed_mask]))
        # Zero <mask> tokens remain and all unmasked tokens are valid in-range non-forbidden IDs
        assert int((inpainted == tok.mask_id).sum().item()) == 0
        unmasked_new = inpainted[~fixed_mask]
        assert bool(((unmasked_new >= 0) & (unmasked_new < tok.vocab_size)).all().item())
        assert bool(((unmasked_new != tok.pad_id) & (unmasked_new != tok.bos_id)).all().item())


@pytest.mark.parametrize("device", DEVICES)
def test_edge_cases_empty_pocket_all_neginf_and_safe_prompt_beam_search(device: str) -> None:
    """Verify N=0 pocket, all-(-inf) logits, 1D grammar logits, ring cap, and SAFE prompt beam."""
    dev = torch.device(device)
    tok = nlp.AtomLevelSMILESTokenizer()

    # 1. Empty pocket (N=0) produces exact 0.0 cross-attention output with finite grads
    cross = nlp.PocketCrossAttention(d_model=32, d_pocket=16, n_heads=4).to(dev)
    q = torch.randn((2, 6, 32), device=dev, dtype=torch.float32, requires_grad=True)
    empty_p = torch.empty((2, 0, 16), device=dev, dtype=torch.float32)
    out_empty, attn_empty = cross(q, empty_p, return_attn=True)
    assert out_empty.shape == (2, 6, 32) and attn_empty.shape == (2, 4, 6, 0)
    assert float(out_empty.abs().max().item()) == 0.0
    torch.autograd.backward(out_empty.sum())
    assert q.grad is not None and bool(torch.isfinite(q.grad).all().item())

    # 2. All-(-inf) row in apply_temperature_top_k_top_p produces zero NaNs
    bad_logits = torch.full((2, 16), float("-inf"), device=dev, dtype=torch.float32)
    bad_logits[0, 3] = 2.5
    filt = nlp.apply_temperature_top_k_top_p(
        bad_logits, temperature=0.9, top_k=4, top_p=0.8, min_p=0.1
    )
    assert not bool(torch.isnan(filt).any().item())
    assert bool(torch.isfinite(filt[0, 3]).item())

    # 3. 1D grammar logits & max_open_rings=1 enforcement
    g_cap = nlp.SMILESValenceGrammarMask(tok, max_open_rings=1)
    g_cap.reset(batch_size=1)
    for tid in tok.encode("c1cc", add_bos=True, add_eos=False):
        g_cap.step([tid])
    logits_1d = torch.zeros((tok.vocab_size,), device=dev, dtype=torch.float32)
    masked_1d = g_cap.mask_logits(logits_1d)
    assert float(masked_1d[tok.token_to_id["1"]].item()) == 0.0  # Closing ring 1 allowed
    assert float(masked_1d[tok.token_to_id["2"]].item()) == float("-inf")  # Opening ring 2 blocked

    # 4. Multi-token SAFE scaffold prompt beam search with KV-cache vs uncached parity
    torch.manual_seed(113)
    lm = nlp.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, conditioning_mode="none"
    ).to(dev)
    lm.eval()
    safe_prompt = tok.encode("c1ccc2cc1.", add_bos=True, add_eos=False)
    g_safe = nlp.SMILESValenceGrammarMask(tok)
    b_cache = nlp.beam_search_decode(
        lm,
        prompt_ids=safe_prompt,
        beam_width=3,
        max_new_tokens=6,
        grammar_mask=g_safe,
        use_cache=True,
        device=dev,
    )
    b_nocache = nlp.beam_search_decode(
        lm,
        prompt_ids=safe_prompt,
        beam_width=3,
        max_new_tokens=6,
        grammar_mask=g_safe,
        use_cache=False,
        device=dev,
    )
    assert b_cache.sequences == b_nocache.sequences
    assert all(seq[: len(safe_prompt)] == safe_prompt for seq in b_cache.sequences)

    # 5. Dead-beam hardening: after "CC(F", ONLY ')' is grammar-valid (< beam_width=4)!
    narrow_prompt = tok.encode("CC(F", add_bos=True, add_eos=False)
    g_narrow = nlp.SMILESValenceGrammarMask(tok)
    b_narrow = nlp.beam_search_decode(
        lm,
        prompt_ids=narrow_prompt,
        beam_width=4,
        max_new_tokens=4,
        grammar_mask=g_narrow,
        use_cache=True,
        device=dev,
    )
    assert b_narrow.sequences[0][len(narrow_prompt)] == tok.token_to_id[")"]
    assert all(torch.isfinite(torch.tensor(s)) for s in b_narrow.scores)


@pytest.mark.parametrize("device", DEVICES)
def test_prefix_mode_batched_attention_mask_kv_cache_parity(device: str) -> None:
    """Batched prefix LM with 2D attention_mask + incremental KVCache matches full forward."""
    dev = torch.device(device)
    torch.manual_seed(127)
    gen = torch.Generator(device="cpu").manual_seed(131)

    tok = nlp.AtomLevelSMILESTokenizer()
    # Different lengths so row 0 has trailing <pad> (False in attention_mask) while row 1 is full
    ids, am = tok.batch_encode(["CCO", DOPAMINE_SMILES], device=dev)
    bsz, seq_len = ids.shape
    num_queries = 4
    pocket = torch.randn((bsz, 12, 32), generator=gen, dtype=torch.float32).to(dev)
    kpm = torch.zeros((bsz, 12), dtype=torch.bool, device=dev)
    kpm[0, 9:] = True

    lm_pref = nlp.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_pocket=32,
        conditioning_mode="prefix",
        num_pocket_queries=num_queries,
        perceiver_layers=1,
    ).to(dev)
    lm_pref.eval()

    with torch.no_grad():
        full_logits = lm_pref(
            ids, pocket_tokens=pocket, pocket_key_padding_mask=kpm, attention_mask=am
        ).logits

        # Prefill 3 tokens -> kv_cache.seq_len == num_queries + 3
        out0 = lm_pref(
            ids[:, :3],
            pocket_tokens=pocket,
            pocket_key_padding_mask=kpm,
            attention_mask=am[:, :3],
            use_cache=True,
        )
        assert out0.kv_cache is not None and out0.kv_cache.seq_len == num_queries + 3

        # Incremental step passing pocket_tokens AND step passing pocket_tokens=None
        chunks: list[Tensor] = [out0.logits]
        cache: nlp.KVCache | None = out0.kv_cache
        for step_idx in range(3, seq_len):
            pass_pocket = pocket if (step_idx % 2 == 1) else None
            step_out = lm_pref(
                ids[:, step_idx : step_idx + 1],
                pocket_tokens=pass_pocket,
                pocket_key_padding_mask=kpm if pass_pocket is not None else None,
                attention_mask=am[:, : step_idx + 1],
                kv_cache=cache,
                use_cache=True,
            )
            cache = step_out.kv_cache
            chunks.append(step_out.logits)

        inc_logits = torch.cat(chunks, dim=1)
        # Compare all valid (non-padded) token logits -> exact < 1e-5 parity!
        valid_err = float((inc_logits[am] - full_logits[am]).abs().max().item())
        assert valid_err < 1e-5, (
            f"Batched prefix attention_mask KV-cache err {valid_err:.2e} >= 1e-5 on {device}"
        )
