"""Offline CI-safe unit tests and live MPS verification for GenEuclidean v4.0 novel LLMs.

Covers:
1. Pure-RDKit SAFE fragment codec (21/21 clinical CNS + peripheral drug exact isomeric
   round-trip, fragment order permutation, validation, and scaffold-decoration prefixes).
2. ``SAFE-GPT`` (``datamol-io/safe-gpt``) tokenizer JSON sanitizer + ``SAFESplitter`` regex
   pre-tokenizer verification (offline synthetic JSON + live MPS).
3. ``NovoMolGen-157M`` (``chandar-lab/NovoMolGen_157M_SMILES_BPE``) fused-QKV / interleaved-RoPE /
   ``[up, gate]`` SwiGLU -> native ``LlamaForCausalLM`` remapper exact logit parity (<1e-5) on
   both MHA and GQA synthetic Llama models + live MPS verification.
4. ``PocketConditionedLLM`` ``format_mode="safe"`` encoding/decoding and KV-cached CFG
   scaffold decoration.
5. Contrastive Cross-Pocket Negative DPO (``L_XP-DPO``) exact gradient check & LoRA-only update.
6. 4-Channel Group reward-Decoupled Policy Optimization (``GDPO``, NVlabs Jan 2026) exact
   positive-affine per-channel normalization invariance & LoRA-only training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor

from geneuclidean.cns.mpo import REFERENCE_CNS_DRUGS, REFERENCE_NON_CNS_DRUGS
from geneuclidean.models.safe_codec import (
    SAFE_SPLITTER_REGEX,
    build_scaffold_prefix,
    extract_core_safe_prefix,
    is_valid_safe,
    safe_to_smiles,
    smiles_to_safe,
    split_safe_fragments,
)

# ---------------------------------------------------------------------------
# 1. Pure-RDKit SAFE Codec (0 external safe-mol / transformers dependencies)
# ---------------------------------------------------------------------------

_EXTRA_CLINICAL_SMILES: dict[str, str] = {
    "donepezil": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "donepezil_chiral": "COc1cc2c(cc1OC)C(=O)[C@H](CC1CCN(Cc3ccccc3)CC1)C2",
    "ez_chiral_ester": "CC/C=C\\C(=O)O[C@H](C)c1ccc(OC(=O)N(C)C)cc1",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
}


def test_safe_codec_21_clinical_drugs_exact_isomeric_roundtrip() -> None:
    """Verify 21/21 clinical CNS + peripheral drugs (plus extras) pass 100% isomeric round-trip."""
    clinical_21 = {**REFERENCE_CNS_DRUGS, **REFERENCE_NON_CNS_DRUGS}
    assert len(clinical_21) == 21

    all_drugs = {**clinical_21, **_EXTRA_CLINICAL_SMILES}
    rng = np.random.default_rng(2026)
    multi_frag_count = 0

    for name, smi in all_drugs.items():
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid test drug SMILES: {name}"
        ref_iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        safe_str = smiles_to_safe(smi, slicer="brics")
        assert is_valid_safe(safe_str), f"Invalid SAFE for {name}: {safe_str}"
        decoded = safe_to_smiles(safe_str, canonical=True)
        assert decoded == ref_iso, f"Round-trip mismatch for {name}: {decoded} != {ref_iso}"

        if len(split_safe_fragments(safe_str)) >= 2:
            multi_frag_count += 1
            perm_safe = smiles_to_safe(smi, slicer="brics", randomize_order=True, rng=rng)
            assert is_valid_safe(perm_safe)
            assert safe_to_smiles(perm_safe, canonical=True) == ref_iso

    assert multi_frag_count >= 18

    # Alternative slicers and edge-case validation
    rot_safe = smiles_to_safe(REFERENCE_CNS_DRUGS["haloperidol"], slicer="rotatable")
    assert safe_to_smiles(rot_safe) == Chem.MolToSmiles(
        Chem.MolFromSmiles(REFERENCE_CNS_DRUGS["haloperidol"]),
        canonical=True,
        isomericSmiles=True,
    )
    none_safe = smiles_to_safe(REFERENCE_CNS_DRUGS["ecopipam"], slicer="none")
    assert "." not in none_safe
    assert is_valid_safe("c12ccccc1.O2") is True
    assert is_valid_safe("c1ccccc1.CCO") is False
    assert is_valid_safe("c1ccccc1.CCO", allow_disconnected=True) is True
    assert is_valid_safe("c1ccccc12") is False
    assert is_valid_safe("") is False
    assert safe_to_smiles("c1ccccc1.CCO", largest_fragment=True) == "c1ccccc1"
    assert safe_to_smiles("not_a_molecule") is None

    with pytest.raises(ValueError, match="Unsupported SAFE slicer"):
        smiles_to_safe("CCO", slicer="unknown")
    with pytest.raises(ValueError, match="Invalid SMILES"):
        smiles_to_safe("C(C(C")


def test_safe_codec_build_scaffold_prefix_and_decoration_caps() -> None:
    """Verify dummy-annotated scaffolds produce valid open SAFE prefixes for R-group growth."""
    single_prefix = build_scaffold_prefix("c1ccc([*])cc1")
    assert single_prefix.endswith(".")
    capped_single = safe_to_smiles(f"{single_prefix}C2")
    assert capped_single == "Cc1ccccc1"

    double_prefix = build_scaffold_prefix("[1*]c1ccc([2*])cc1")
    assert double_prefix.endswith(".")
    capped_double = safe_to_smiles(f"{double_prefix}C2.O3")
    assert capped_double == "Cc1ccc(O)cc1"

    # Chiral VMAT2 dihydrotetrabenazine core with attachment point preserves stereocenters
    vmat2_core = "COc1cc2c(cc1OC)[C@H]1C[C@@H]([*])[C@H](CC(C)C)CN1CC2"
    vmat2_prefix = build_scaffold_prefix(vmat2_core)
    assert vmat2_prefix.endswith(".")
    assert "[C@H]" in vmat2_prefix and "[C@@H]" in vmat2_prefix
    capped_vmat2 = safe_to_smiles(f"{vmat2_prefix}O4")
    assert capped_vmat2 is not None
    assert "[C@H]" in capped_vmat2 and "[C@@H]" in capped_vmat2

    # Plain full SMILES (no dummy) extracts largest BRICS core fragment
    haloperidol_prefix = build_scaffold_prefix(REFERENCE_CNS_DRUGS["haloperidol"])
    assert haloperidol_prefix.endswith(".")
    assert len(haloperidol_prefix) >= 5

    core_1 = extract_core_safe_prefix(REFERENCE_CNS_DRUGS["risperidone"], keep_n_frags=1)
    assert core_1.endswith(".")

    with pytest.raises(ValueError, match="Invalid scaffold"):
        build_scaffold_prefix("c1ccc([*")


# ---------------------------------------------------------------------------
# 2. Backbone Registry, SAFE-GPT Tokenizer, and NovoMolGen Remapper (Offline)
# ---------------------------------------------------------------------------


def _require_llm_extras() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("tokenizers")


def test_backbone_registry_safe_gpt_and_novomolgen_specs() -> None:
    """Verify safe-gpt and novomolgen-157m registry keys, representation, and LoRA configs."""
    _require_llm_extras()
    from geneuclidean.models.hf_backbones import (
        BACKBONE_REGISTRY,
        NOVOMOLGEN_157M_ALIAS,
        NOVOMOLGEN_157M_REPO_ID,
        SAFE_GPT_ALIAS,
        SAFE_GPT_REPO_ID,
    )

    for key in (SAFE_GPT_ALIAS, SAFE_GPT_REPO_ID):
        assert key in BACKBONE_REGISTRY
        spec = BACKBONE_REGISTRY[key]
        assert spec.backbone_id == SAFE_GPT_REPO_ID
        assert spec.family == "safe_gpt"
        assert spec.representation == "safe"
        assert spec.is_safe_native is True
        assert spec.trust_remote_code is False
        assert spec.uses_tuple_kv_cache is False
        assert spec.fan_in_fan_out is True

    for key in (NOVOMOLGEN_157M_ALIAS, NOVOMOLGEN_157M_REPO_ID):
        assert key in BACKBONE_REGISTRY
        spec = BACKBONE_REGISTRY[key]
        assert spec.backbone_id == NOVOMOLGEN_157M_REPO_ID
        assert spec.family == "novomolgen"
        assert spec.representation == "smiles"
        assert spec.is_safe_native is False
        assert spec.trust_remote_code is False
        assert spec.uses_tuple_kv_cache is False
        assert "q_proj" in spec.lora_targets and "gate_proj" in spec.lora_targets


def test_safe_gpt_tokenizer_json_sanitizer_and_regex_pretokenizer(tmp_path: Path) -> None:
    """Verify offline SAFE-GPT JSON key stripping and single-token stereocenter regex splitting."""
    _require_llm_extras()
    from tokenizers import Tokenizer, models, pre_tokenizers

    from geneuclidean.models.hf_backbones import (
        SAFE_GPT_EXTRA_TOKENIZER_KEYS,
        build_safe_gpt_tokenizer,
        sanitize_safe_gpt_tokenizer_json,
    )

    vocab_tokens = [
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]",
        "C",
        "c",
        "N",
        "O",
        "F",
        "Cl",
        "Br",
        "(",
        ")",
        ".",
        "=",
        "1",
        "2",
        "3",
        "4",
        "%10",
        "%11",
        "[C@H]",
        "[C@@H]",
    ]
    vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    base_tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    base_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    raw_dict: dict[str, Any] = json.loads(base_tok.to_str())

    # Inject the exact 3 non-standard safe-mol keys that crash Rust tokenizers>=0.23
    raw_dict["custom_pre_tokenizer"] = "SAFESplitter"
    raw_dict["tokenizer_type"] = "safe"
    raw_dict["tokenizer_attrs"] = {"regex": SAFE_SPLITTER_REGEX}
    corrupted_json = json.dumps(raw_dict)

    # Confirm raw corrupted JSON indeed fails Rust Tokenizer.from_str
    with pytest.raises(Exception, match=r"expected|unknown|trailing|line"):
        Tokenizer.from_str(corrupted_json)

    cleaned = sanitize_safe_gpt_tokenizer_json(corrupted_json)
    for forbidden_key in SAFE_GPT_EXTRA_TOKENIZER_KEYS:
        assert forbidden_key not in cleaned

    # Confirm the sanitized Whitespace pre-tokenizer alone fractures stereocenters ([C@H])
    whitespace_only = Tokenizer.from_str(json.dumps(cleaned))
    sample = "[C@H]1CC[C@@H]1Cl.c1%10ccc(Br)cc1%11"
    ws_tokens = whitespace_only.encode(sample).tokens
    assert "[C@H]" not in ws_tokens
    assert "[C@@H]" not in ws_tokens
    assert "%10" not in ws_tokens

    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text(corrupted_json, encoding="utf-8")
    tok = build_safe_gpt_tokenizer(tok_file)

    assert tok.bos_token == "[CLS]"
    assert tok.eos_token == "[SEP]"
    assert tok.pad_token == "[PAD]"

    tokens = tok.tokenize(sample)
    assert "[C@H]" in tokens
    assert "[C@@H]" in tokens
    assert "Cl" in tokens
    assert "Br" in tokens
    assert "%10" in tokens
    assert "%11" in tokens
    assert "[UNK]" not in tokens
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = "".join(tok.decode(ids, skip_special_tokens=True).split())
    assert decoded == sample


def _flash_attn_llama_reference_forward(
    input_ids: Tensor,
    raw_sd: dict[str, Tensor],
    *,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_eps: float = 1e-6,
    rope_theta: float = 10000.0,
) -> Tensor:
    """Reference forward pass matching ``flash_attn`` GPTLMHeadModel Llama layout.

    Uses interleaved RoPE pairs ``(2i, 2i+1)``, fused ``Wqkv``, and ``GatedMlp``
    ``fc1 -> (x_up, gate) -> x_up * silu(gate)``.
    """

    def rms_norm(x: Tensor, weight: Tensor) -> Tensor:
        inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
        return x * inv * weight

    def apply_interleaved_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # x: [B, S, H, D]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    bsz, seq_len = input_ids.shape
    h = F.embedding(input_ids, raw_sd["transformer.embeddings.word_embeddings.weight"])

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
    )
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [S, D/2]
    cos = torch.cos(freqs)[None, :, None, :]
    sin = torch.sin(freqs)[None, :, None, :]

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    groups = num_heads // num_kv_heads
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32), diagonal=1
    )

    for i in range(num_layers):
        p = f"transformer.layers.{i}"
        xn = rms_norm(h, raw_sd[f"{p}.norm1.weight"])
        qkv = F.linear(xn, raw_sd[f"{p}.mixer.Wqkv.weight"])
        q_raw, k_raw, v_raw = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)
        q = apply_interleaved_rope(q_raw.view(bsz, seq_len, num_heads, head_dim), cos, sin)
        k = apply_interleaved_rope(k_raw.view(bsz, seq_len, num_kv_heads, head_dim), cos, sin)
        v = v_raw.view(bsz, seq_len, num_kv_heads, head_dim)

        if groups > 1:
            k = k.repeat_interleave(groups, dim=2)
            v = v.repeat_interleave(groups, dim=2)

        qt = q.transpose(1, 2)  # [B, H, S, D]
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        scores = torch.matmul(qt, kt.transpose(-2, -1)) / math.sqrt(head_dim) + causal_mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, vt).transpose(1, 2).reshape(bsz, seq_len, q_dim)
        h = h + F.linear(attn_out, raw_sd[f"{p}.mixer.out_proj.weight"])

        xn2 = rms_norm(h, raw_sd[f"{p}.norm2.weight"])
        x_up, gate = F.linear(xn2, raw_sd[f"{p}.mlp.fc1.weight"]).chunk(2, dim=-1)
        mlp_out = F.linear(x_up * F.silu(gate), raw_sd[f"{p}.mlp.fc2.weight"])
        h = h + mlp_out

    h = rms_norm(h, raw_sd["transformer.ln_f.weight"])
    return F.linear(h, raw_sd["lm_head.weight"])


@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(4, 4), (4, 2)])
def test_novomolgen_fused_qkv_remapper_exact_logit_parity(
    num_heads: int, num_kv_heads: int
) -> None:
    """Verify bit-exact (<1e-5) logit parity between flash_attn Llama and remapped HF Llama."""
    _require_llm_extras()
    from transformers import LlamaConfig, LlamaForCausalLM

    from geneuclidean.models.hf_backbones import remap_novomolgen_state_dict

    torch.manual_seed(123)
    vocab_size = 64
    hidden_size = 64
    intermediate_size = 128
    num_layers = 2
    head_dim = hidden_size // num_heads
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    raw_sd: dict[str, Tensor] = {
        "transformer.embeddings.word_embeddings.weight": torch.randn(vocab_size, hidden_size) * 0.1,
        "transformer.ln_f.weight": torch.ones(hidden_size) + torch.randn(hidden_size) * 0.05,
        "lm_head.weight": torch.randn(vocab_size, hidden_size) * 0.1,
    }
    for i in range(num_layers):
        p = f"transformer.layers.{i}"
        raw_sd[f"{p}.mixer.Wqkv.weight"] = torch.randn(q_dim + 2 * kv_dim, hidden_size) * 0.1
        raw_sd[f"{p}.mixer.out_proj.weight"] = torch.randn(hidden_size, q_dim) * 0.1
        raw_sd[f"{p}.norm1.weight"] = torch.ones(hidden_size) + torch.randn(hidden_size) * 0.05
        raw_sd[f"{p}.norm2.weight"] = torch.ones(hidden_size) + torch.randn(hidden_size) * 0.05
        raw_sd[f"{p}.mlp.fc1.weight"] = torch.randn(2 * intermediate_size, hidden_size) * 0.1
        raw_sd[f"{p}.mlp.fc2.weight"] = torch.randn(hidden_size, intermediate_size) * 0.1

    mapped = remap_novomolgen_state_dict(
        raw_sd,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
    )

    llama_cfg_cls: Any = LlamaConfig
    llama_cfg: Any = llama_cfg_cls(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
        use_cache=True,
    )
    llama_cfg._attn_implementation = "sdpa"
    hf_model = LlamaForCausalLM(llama_cfg).eval()
    hf_model.load_state_dict(mapped, strict=True)

    input_ids = torch.randint(4, vocab_size, (2, 11), dtype=torch.long)
    with torch.no_grad():
        ref_logits = _flash_attn_llama_reference_forward(
            input_ids,
            raw_sd,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        # Also verify DynamicCache KV-cached incremental decoding matches full forward
        cached = hf_model(input_ids=input_ids[:, :-1], use_cache=True)
        step_logits = hf_model(
            input_ids=input_ids[:, -1:],
            past_key_values=cached.past_key_values,
            use_cache=True,
        ).logits[:, -1, :]

    max_diff = float((hf_logits - ref_logits).abs().max().item())
    assert max_diff < 1e-5, f"NovoMolGen remapper logit drift: {max_diff:.2e}"
    kv_diff = float((step_logits - hf_logits[:, -1, :]).abs().max().item())
    assert kv_diff < 1e-5, f"Llama DynamicCache KV drift: {kv_diff:.2e}"


# ---------------------------------------------------------------------------
# 3. PocketConditionedLLM SAFE Mode, L_XP-DPO, and 4-Channel GDPO (Offline)
# ---------------------------------------------------------------------------


def _make_mock_cfg_and_model(
    *,
    format_mode: str = "smiles",
    seed: int = 17,
) -> tuple[Any, Any]:
    _require_llm_extras()
    from geneuclidean.config import ExperimentConfig, LLMConfig, seed_everything
    from geneuclidean.data.featurizer import LIGAND_FP_DIM
    from geneuclidean.models.hf_backbones import MOCK_BACKBONE_ID, load_hf_backbone
    from geneuclidean.models.llm_bridge import PocketConditionedLLM

    seed_everything(seed)
    cfg = ExperimentConfig(
        name="test-v4-novel",
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
            phase_b_epochs=1,
            grad_accum_steps=1,
        ),
    )
    cfg.data.max_seq_len = 64
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
    cfg.train.batch_size = 2

    backbone = load_hf_backbone(
        MOCK_BACKBONE_ID,
        dtype="float32",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        use_lora=True,
    )
    model = PocketConditionedLLM(
        v2_cfg=cfg,
        backbone=backbone,
        num_queries=4,
        perceiver_layers=1,
        perceiver_heads=4,
        num_global_tokens=2,
        freeze_pocket_encoder=True,
        format_mode=format_mode,
    )
    return cfg, model


def _make_pocket_dict(dim: int = 32, n_atoms: int = 12, seed: int = 1) -> dict[str, Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    node_emb = torch.randn((1, n_atoms, dim), generator=gen, dtype=torch.float32)
    pooled_emb = node_emb.mean(dim=1)
    atom_mask = torch.ones((1, n_atoms), dtype=torch.bool)
    return {"node_emb": node_emb, "pooled_emb": pooled_emb, "atom_mask": atom_mask}


def test_pocket_conditioned_llm_safe_mode_and_scaffold_decoration(tmp_path: Path) -> None:
    """Verify PocketConditionedLLM format_mode='safe', SAFE dataset, and CFG scaffold decoration."""
    cfg, model = _make_mock_cfg_and_model(format_mode="safe", seed=21)
    from geneuclidean.models.encoders.rbf import PocketEncoderOutput
    from geneuclidean.training.llm_trainer import CachedPocketLLMDataset, CachedPocketRecord

    assert model.format_mode == "safe"
    assert model.is_safe_mode is True

    halo = REFERENCE_CNS_DRUGS["haloperidol"]
    safe_str = model.encode_molecule_text(halo)
    assert "." in safe_str
    decoded = model.decode_molecule_text(safe_str)
    assert decoded == Chem.MolToSmiles(
        Chem.MolFromSmiles(halo), canonical=True, isomericSmiles=True
    )

    # CachedPocketLLMDataset automatically encodes SMILES -> SAFE when format_mode='safe'
    rec = CachedPocketRecord(
        pdb_id="drd2_test",
        node_emb=torch.randn(10, 32),
        pooled_emb=torch.randn(32),
        atom_mask=torch.ones(10, dtype=torch.bool),
        ligand_fp=None,
        canonical_smiles=halo,
    )
    ds = CachedPocketLLMDataset(
        [rec],
        tokenizer=model.tokenizer,
        is_smiles_native=True,
        bos_token_id=model.bos_token_id,
        eos_token_id=model.eos_token_id,
        max_seq_len=64,
        randomize=True,
        seed=5,
        format_mode="safe",
    )
    item = ds[0]
    decoded_from_ds = model.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
    assert "." in decoded_from_ds
    assert safe_to_smiles(decoded_from_ds) == decoded

    # Fragment-prefix prompt and KV-cached CFG scaffold decoration
    prompt_ids, prefix_str = model.build_decoration_prompt_ids(
        "c1ccc([*])cc1", batch_size=3, device="cpu"
    )
    assert prefix_str == "c1ccc2cc1."
    assert prompt_ids.shape[0] == 3
    assert int(prompt_ids[0, 0].item()) == model.bos_token_id

    p_dict = _make_pocket_dict(dim=32, n_atoms=10, seed=9)
    p_out = PocketEncoderOutput(
        node_emb=p_dict["node_emb"],
        pooled_emb=p_dict["pooled_emb"],
        vec_emb=None,
    )
    dec_results = model.decorate_scaffold(
        "c1ccc([*])cc1",
        pocket_out=p_out,
        atom_mask=p_dict["atom_mask"],
        num_samples=3,
        max_new_tokens=10,
        cfg_scale=1.5,
        seed=42,
    )
    assert len(dec_results) == 3
    for row in dec_results:
        assert row["prefix"] == "c1ccc2cc1."
        assert isinstance(row["raw_text"], str) and row["raw_text"].startswith("c1ccc2cc1.")

    ckpt = tmp_path / "safe_adapter.pt"
    model.save_adapter_checkpoint(ckpt)
    meta = model.load_adapter_checkpoint(ckpt)
    assert meta["format_mode"] == "safe"


def test_xp_dpo_loss_gradient_and_selective_lora_update(tmp_path: Path) -> None:
    """Verify L_XP-DPO exact gradients (up on p_on, down on p_off) and LoRA-only updates."""
    cfg, model = _make_mock_cfg_and_model(format_mode="smiles", seed=31)
    from geneuclidean.training.dpo import PreferencePair
    from geneuclidean.training.llm_trainer import (
        CrossPocketPreferencePair,
        compute_cross_pocket_dpo_loss,
        run_llm_rpo_dpo,
    )

    # 1. Exact closed-form value and gradient sign verification
    pi_on = torch.tensor([-1.2, -0.8], requires_grad=True)
    pi_off = torch.tensor([-1.0, -1.1], requires_grad=True)
    beta_xp = 0.75
    xp_loss = compute_cross_pocket_dpo_loss(pi_on, pi_off, beta=beta_xp)
    expected = -F.logsigmoid(beta_xp * (pi_on - pi_off)).mean()
    assert torch.allclose(xp_loss, expected, atol=1e-6)
    xp_loss.backward()
    assert pi_on.grad is not None and pi_off.grad is not None
    # Descending xp_loss increases pi_on (negative grad) and decreases pi_off (positive grad)
    assert bool((pi_on.grad < 0.0).all().item())
    assert bool((pi_off.grad > 0.0).all().item())
    assert torch.allclose(pi_on.grad, -pi_off.grad, atol=1e-6)

    # 2. End-to-end run_llm_rpo_dpo with cross_pocket_weight > 0 (both CrossPocketPreferencePair
    #    and standard PreferencePair with pocket_batch["off_target_pocket_emb"])
    p_drd1 = _make_pocket_dict(dim=32, n_atoms=12, seed=101)
    p_drd2 = _make_pocket_dict(dim=32, n_atoms=14, seed=202)
    p_pde10a = _make_pocket_dict(dim=32, n_atoms=11, seed=303)
    p_vmat2 = _make_pocket_dict(dim=32, n_atoms=13, seed=404)

    pairs = [
        CrossPocketPreferencePair(
            pocket_batch=p_drd1,
            chosen_smiles="CC(=O)Oc1ccccc1C(=O)O",
            rejected_smiles="CC",
            chosen_score=0.90,
            rejected_score=0.20,
            off_target_pocket_emb=p_drd2,
        ),
        PreferencePair(
            pocket_batch={**p_pde10a, "off_target_pocket_emb": p_vmat2},
            chosen_smiles="COc1ccccc1C(=O)N",
            rejected_smiles="CCO",
            chosen_score=0.85,
            rejected_score=0.25,
        ),
    ]

    before_snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    before_req = {k: p.requires_grad for k, p in model.named_parameters()}

    res = run_llm_rpo_dpo(
        model=model,
        train_records=[],
        cfg=cfg,
        device="cpu",
        epochs=2,
        beta=0.5,
        lr=2e-3,
        sft_nll_weight=0.25,
        grad_accum_steps=2,
        seed=31,
        preference_pairs=pairs,
        cross_pocket_weight=0.5,
        cross_pocket_beta=0.6,
        save_dir=tmp_path / "xp_dpo",
    )
    assert res["epochs"] == 2
    assert res["cross_pocket_weight"] == 0.5
    assert res["history"][0]["xp_pairs"] == 2
    assert math.isfinite(float(res["history"][-1]["xp_dpo_loss"]))
    assert math.isfinite(float(res["history"][-1]["xp_margin"]))
    assert (tmp_path / "xp_dpo" / "best_adapter.pt").is_file()

    lora_changed = False
    for name, param in model.named_parameters():
        assert param.requires_grad == before_req[name]
        if "lora_" in name.lower():
            if not torch.equal(param.detach(), before_snap[name]):
                lora_changed = True
        else:
            assert torch.equal(param.detach(), before_snap[name]), (
                f"Non-LoRA weight {name} drifted during L_XP-DPO!"
            )
    assert lora_changed

    # Backwards-compatible default cross_pocket_weight=0.0 skips XP term
    res_plain = run_llm_rpo_dpo(
        model=model,
        train_records=[],
        cfg=cfg,
        device="cpu",
        epochs=1,
        lr=1e-3,
        grad_accum_steps=2,
        preference_pairs=pairs,
    )
    assert res_plain["cross_pocket_weight"] == 0.0
    assert res_plain["history"][0]["xp_pairs"] == 0
    assert res_plain["history"][0]["xp_dpo_loss"] == 0.0


def test_gdpo_per_channel_normalization_invariance_and_lora_training(tmp_path: Path) -> None:
    """Verify GDPO affine scale invariance across 4 channels and LoRA-only training."""
    cfg, model = _make_mock_cfg_and_model(format_mode="smiles", seed=43)
    from geneuclidean.training.llm_trainer import (
        GDPOGroup,
        compute_gdpo_advantages,
        run_llm_gdpo,
    )

    # 1. Exact positive-affine scale invariance: inflating Safety by 1000x + 500 shift
    #    must produce IDENTICAL per-channel and combined advantages!
    base_rewards = torch.tensor(
        [
            [
                [0.95, 0.10, 0.40, 0.60],
                [0.85, 0.90, 0.80, 0.70],
                [0.40, 0.50, 0.30, 0.20],
                [0.20, 0.20, 0.90, 0.85],
            ],
            [
                [0.60, 0.80, 0.70, 0.50],
                [0.70, 0.20, 0.65, 0.55],
                [0.90, 0.95, 0.85, 0.90],
                [0.30, 0.15, 0.25, 0.35],
            ],
        ],
        dtype=torch.float64,
    )  # [B=2, G=4, K=4]

    scaled_rewards = base_rewards.clone()
    scaled_rewards[..., 0] = scaled_rewards[..., 0] * 1000.0 + 500.0  # Massive Safety variance
    scaled_rewards[..., 1] = scaled_rewards[..., 1] * 0.001 - 2.0  # Tiny Selectivity scale

    adv_base, per_ch_base = compute_gdpo_advantages(base_rewards, batch_normalize=True)
    adv_scaled, per_ch_scaled = compute_gdpo_advantages(scaled_rewards, batch_normalize=True)

    assert torch.allclose(per_ch_base, per_ch_scaled, atol=1e-5)
    assert torch.allclose(adv_base, adv_scaled, atol=1e-5)
    # Each group & channel has exact 0 mean and unit std
    assert torch.allclose(per_ch_base.mean(dim=-2), torch.zeros(2, 4), atol=1e-5)
    assert torch.allclose(per_ch_base.std(dim=-2, unbiased=False), torch.ones(2, 4), atol=1e-5)

    # Constant channel edge case: exact 0.0, finite, no NaN
    const_rewards = base_rewards.clone()
    const_rewards[0, :, 2] = 0.75
    adv_const, per_ch_const = compute_gdpo_advantages(const_rewards)
    assert torch.isfinite(adv_const).all()
    assert torch.allclose(per_ch_const[0, :, 2], torch.zeros(4), atol=1e-7)

    # Hygiene gate suppresses positive advantage on unhygienic artifacts (e.g., DCU)
    gates = torch.ones((2, 4), dtype=torch.float32)
    gates[0, 1] = 0.0  # Candidate with positive advantage fails hygiene
    adv_gated, _ = compute_gdpo_advantages(base_rewards, hygiene_gates=gates)
    assert float(adv_base[0, 1].item()) > 0.0
    assert float(adv_gated[0, 1].item()) == 0.0

    # 2. End-to-end LoRA-only run_llm_gdpo updates ONLY LoRA weights bit-for-bit
    groups = [
        GDPOGroup(
            pocket_batch=_make_pocket_dict(dim=32, n_atoms=10, seed=501),
            smiles_list=["CC(=O)Oc1ccccc1C(=O)O", "COc1ccccc1", "CCO", "c1ccccc1"],
            safety_rewards=[0.90, 0.85, 0.40, 0.50],
            selectivity_rewards=[0.80, 0.20, 0.10, 0.30],
            fit_3d_rewards=[0.85, 0.60, 0.25, 0.45],
            medchem_anchor_rewards=[0.75, 0.70, 0.35, 0.40],
            hygiene_gates=[1.0, 1.0, 1.0, 0.0],
        ),
        GDPOGroup(
            pocket_batch=_make_pocket_dict(dim=32, n_atoms=12, seed=502),
            smiles_list=["CCN(CC)CC", "c1ccncc1C(=O)N", "CC"],
            safety_rewards=[0.70, 0.92, 0.30],
            selectivity_rewards=[0.50, 0.88, 0.05],
            fit_3d_rewards=[0.65, 0.80, 0.20],
            medchem_anchor_rewards=[0.60, 0.85, 0.15],
        ),
    ]

    before_snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    before_req = {k: p.requires_grad for k, p in model.named_parameters()}

    gdpo_dir = tmp_path / "gdpo_out"
    res = run_llm_gdpo(
        model=model,
        train_records=[],
        cfg=cfg,
        device="cpu",
        epochs=2,
        lr=2e-3,
        clip_eps=0.2,
        kl_beta=0.05,
        rpo_nll_weight=0.15,
        grad_accum_steps=1,
        seed=43,
        gdpo_groups=groups,
        save_dir=gdpo_dir,
    )
    assert res["epochs"] == 2
    assert res["num_groups"] == 2
    assert res["num_candidates"] == 7
    assert (gdpo_dir / "best_adapter.pt").is_file()
    assert (gdpo_dir / "history.jsonl").is_file()
    assert math.isfinite(float(res["history"][-1]["total_loss"]))
    assert math.isfinite(float(res["history"][-1]["kl_loss"]))
    assert float(res["history"][-1]["kl_loss"]) >= -1e-6

    lora_changed = False
    for name, param in model.named_parameters():
        assert param.requires_grad == before_req[name]
        if "lora_" in name.lower():
            if not torch.equal(param.detach(), before_snap[name]):
                lora_changed = True
        else:
            assert torch.equal(param.detach(), before_snap[name]), (
                f"Non-LoRA parameter {name} changed during GDPO!"
            )
    assert lora_changed


# ---------------------------------------------------------------------------
# 4. Live Apple Silicon MPS Verification (@pytest.mark.slow)
# ---------------------------------------------------------------------------


def _has_cached_weights(repo_id: str, filename: str = "pytorch_model.bin") -> bool:
    slug = "models--" + repo_id.replace("/", "--")
    snaps = Path(".hf_cache.nosync") / "hub" / slug / "snapshots"
    if not snaps.is_dir():
        return False
    return any((snap / filename).exists() for snap in snaps.iterdir() if snap.is_dir())


def _small_v2_cfg_for_live() -> Any:
    from geneuclidean.config import (
        CVAEConfig,
        DataConfig,
        DecoderConfig,
        EncoderConfig,
        ExperimentConfig,
        MultiTaskConfig,
    )

    return ExperimentConfig(
        data=DataConfig(max_neighbors=24),
        encoder=EncoderConfig(
            arch="painn",
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=2,
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


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_cached_weights("datamol-io/safe-gpt"),
    reason="datamol-io/safe-gpt weights not cached in .hf_cache.nosync",
)
def test_live_safe_gpt_mps_forward_lora_backward_cfg_and_decoration() -> None:
    """Live MPS check: SAFE-GPT load, forward+LoRA backward, KV-cached CFG, and scaffold deco."""
    _require_llm_extras()
    from geneuclidean.models.hf_backbones import (
        DEFAULT_HF_CACHE_DIR,
        SAFE_GPT_ALIAS,
        load_hf_backbone,
    )
    from geneuclidean.models.llm_bridge import PocketConditionedLLM
    from geneuclidean.sampling.llm_sampler import sample_llm_molecules

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = load_hf_backbone(
        SAFE_GPT_ALIAS,
        cache_dir=DEFAULT_HF_CACHE_DIR,
        dtype="bfloat16" if device.type == "mps" else "float32",
        lora_r=8,
        lora_alpha=16,
        use_lora=True,
        local_files_only=True,
    )
    assert backbone.representation == "safe"
    assert backbone.is_safe_native is True
    assert backbone.hidden_size == 768
    assert backbone.vocab_size == 1880
    assert backbone.uses_tuple_kv_cache is False

    # Confirm single-token stereocenters and ring closures
    toks = backbone.tokenizer.tokenize("[C@H]1CC[C@@H]1Cl.c1%10ccc(Br)cc1")
    assert "[C@H]" in toks and "[C@@H]" in toks and "Cl" in toks and "%10" in toks

    model = PocketConditionedLLM(
        v2_cfg=_small_v2_cfg_for_live(),
        backbone=backbone,
        num_queries=8,
        perceiver_layers=2,
        perceiver_heads=8,
        num_global_tokens=4,
        freeze_pocket_encoder=True,
    ).to(device)
    assert model.format_mode == "safe"

    # 1. Forward + LoRA backward step
    model.train()
    safe_drug = model.encode_molecule_text(REFERENCE_CNS_DRUGS["haloperidol"])
    ids = [
        backbone.bos_token_id,
        *backbone.tokenizer.encode(safe_drug, add_special_tokens=False),
        backbone.eos_token_id,
    ]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, 0] = -100
    coords = torch.randn((1, 20, 3), device=device) * 2.5
    node_feats = torch.randn((1, 20, 52), device=device)
    atom_mask = torch.ones((1, 20), dtype=torch.bool, device=device)

    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        coords=coords,
        node_feats=node_feats,
        atom_mask=atom_mask,
    )
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.lora_parameters()
    )
    assert all(p.grad is None for p in model.encoder.parameters())

    # 2. Native MPS zero-shot SAFE-GPT generation -> >= 80% valid SAFE molecules
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        bos_in = torch.full((12, 1), backbone.bos_token_id, dtype=torch.long, device=device)
        llm_any: Any = model.llm
        gen_ids = llm_any.generate(
            bos_in,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            eos_token_id=backbone.eos_token_id,
            pad_token_id=backbone.pad_token_id,
        )
        prior_raw = [
            "".join(s.split())
            for s in backbone.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ]
        prior_valid = [d for s in prior_raw if (d := model.decode_molecule_text(s)) is not None]
        assert len(prior_valid) >= 9, (
            f"Expected >=9/12 valid zero-shot SAFE-GPT molecules on {device}, "
            f"got {len(prior_valid)}/12: {prior_raw[:4]}"
        )

        # 3. 3D-pocket-conditioned KV-cached CFG generation + SAFE decoding on MPS
        pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
        raw_samples = sample_llm_molecules(
            model=model,
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            num_samples=8,
            max_new_tokens=96,
            strategy="top_p",
            temperature=0.9,
            top_p=0.95,
            cfg_scale=1.2,
            seed=42,
        )
        assert len(raw_samples) == 8
        assert all(isinstance(s, str) and len(s) > 0 for s in raw_samples)
        decoded = [model.decode_molecule_text(s) for s in raw_samples]
        valid = [s for s in decoded if s is not None]
        assert len(valid) >= 1, f"Expected valid SAFE-GPT CFG molecules on {device}, {raw_samples}"

        # 4. 3D-conditioned fragment-prefix scaffold decoration
        deco = model.decorate_scaffold(
            "c1ccc([*])cc1",
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            num_samples=4,
            max_new_tokens=64,
            strategy="top_p",
            temperature=0.9,
            top_p=0.95,
            cfg_scale=1.2,
            seed=7,
        )
        assert len(deco) == 4
        assert all(d["prefix"] == "c1ccc2cc1." for d in deco)


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_cached_weights("chandar-lab/NovoMolGen_157M_SMILES_BPE"),
    reason="chandar-lab/NovoMolGen_157M_SMILES_BPE weights not cached in .hf_cache.nosync",
)
def test_live_novomolgen_157m_mps_forward_lora_backward_and_cfg_generation() -> None:
    """Live MPS check: NovoMolGen-157M native Llama SDPA load, LoRA backward, and CFG sampling."""
    _require_llm_extras()
    from geneuclidean.models.hf_backbones import (
        DEFAULT_HF_CACHE_DIR,
        NOVOMOLGEN_157M_ALIAS,
        load_hf_backbone,
    )
    from geneuclidean.models.llm_bridge import PocketConditionedLLM
    from geneuclidean.sampling.llm_sampler import sample_llm_molecules

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = load_hf_backbone(
        NOVOMOLGEN_157M_ALIAS,
        cache_dir=DEFAULT_HF_CACHE_DIR,
        dtype="bfloat16" if device.type == "mps" else "float32",
        lora_r=8,
        lora_alpha=16,
        use_lora=True,
        local_files_only=True,
    )
    assert backbone.hidden_size == 640
    assert backbone.vocab_size == 500
    assert backbone.representation == "smiles"
    assert backbone.uses_tuple_kv_cache is False

    model = PocketConditionedLLM(
        v2_cfg=_small_v2_cfg_for_live(),
        backbone=backbone,
        num_queries=8,
        perceiver_layers=2,
        perceiver_heads=8,
        num_global_tokens=4,
        freeze_pocket_encoder=True,
    ).to(device)

    # 1. Forward + LoRA backward step
    model.train()
    ids = [
        backbone.bos_token_id,
        *backbone.tokenizer.encode(REFERENCE_CNS_DRUGS["ecopipam"], add_special_tokens=False),
        backbone.eos_token_id,
    ]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, 0] = -100
    coords = torch.randn((1, 20, 3), device=device) * 2.5
    node_feats = torch.randn((1, 20, 52), device=device)
    atom_mask = torch.ones((1, 20), dtype=torch.bool, device=device)

    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        coords=coords,
        node_feats=node_feats,
        atom_mask=atom_mask,
    )
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.lora_parameters()
    )
    assert all(p.grad is None for p in model.encoder.parameters())

    # 2. Native MPS zero-shot Llama SDPA generation -> >= 80% valid drug-like SMILES
    #    (proves flash_attn fused-QKV + interleaved-RoPE + [up, gate] SwiGLU remap is exact!)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        bos_in = torch.full((12, 1), backbone.bos_token_id, dtype=torch.long, device=device)
        llm_any: Any = model.llm
        gen_ids = llm_any.generate(
            bos_in,
            max_new_tokens=56,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            eos_token_id=backbone.eos_token_id,
            pad_token_id=backbone.pad_token_id,
        )
        prior_smiles = [
            "".join(s.split())
            for s in backbone.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ]
        prior_valid = [s for s in prior_smiles if s and Chem.MolFromSmiles(s) is not None]
        assert len(prior_valid) >= 10, (
            f"Expected >=10/12 valid zero-shot NovoMolGen molecules on {device}, "
            f"got {len(prior_valid)}/12: {prior_smiles[:4]}"
        )

        # 3. 3D-pocket-conditioned KV-cached CFG generation smoke test on MPS
        pocket_out = model.encode_pocket(coords, node_feats, atom_mask)
        cfg_samples = sample_llm_molecules(
            model=model,
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            num_samples=4,
            max_new_tokens=32,
            strategy="top_p",
            temperature=0.9,
            top_p=0.95,
            cfg_scale=1.2,
            seed=42,
        )
        assert len(cfg_samples) == 4
        assert all(isinstance(s, str) and len(s) > 0 for s in cfg_samples)
