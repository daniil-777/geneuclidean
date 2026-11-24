"""Unit tests for losses, trainers, DPO, decoding strategies, reranking, and configs."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch import Tensor, nn

from geneuclidean.config import ExperimentConfig, SamplingConfig
from geneuclidean.sampling.decoder_strategies import (
    apply_repetition_penalty,
    apply_smiles_syntax_guardrails,
    combine_cfg_logits,
    compute_strategy_probs,
    compute_temperature_probs,
    filter_min_p_probs,
    filter_top_k,
    filter_top_p_probs,
    filter_typical_probs,
    sample_molecules,
    sample_molecules_detailed,
    sample_with_attention,
)
from geneuclidean.sampling.rerank import (
    compute_mpo_scores,
    maxmin_diversity_select,
    rerank_and_select,
)
from geneuclidean.training.dpo import (
    build_preference_pairs_from_mpo,
    run_dpo_alignment,
)
from geneuclidean.training.losses import (
    compute_cvae_kl_loss,
    compute_cvae_kl_nats,
    compute_dpo_loss,
    compute_drugclip_loss,
    compute_lm_loss,
    compute_multitask_loss,
    compute_pkd_loss,
    compute_property_loss,
    cyclical_kl_beta,
)
from geneuclidean.training.pretrain import pretrain_chemical_prior
from geneuclidean.training.trainer import PocketTrainer

# ---------------------------------------------------------------------------
# Lightweight contract-faithful mocks for independent testing
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal deterministic SMILES tokenizer matching MolecularTokenizer contract."""

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self) -> None:
        self.id_to_token: dict[int, str] = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "<unk>",
            4: "C",
            5: "N",
            6: "O",
            7: "c",
            8: "1",
            9: "(",
            10: ")",
            11: "=",
        }
        self.token_to_id: dict[str, int] = {v: k for k, v in self.id_to_token.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, smi: str, randomize: bool = False, max_len: int = 128) -> list[int]:
        del randomize
        ids = [self.BOS_ID]
        for ch in smi:
            if len(ids) >= max_len - 1:
                break
            ids.append(self.token_to_id.get(ch, self.UNK_ID))
        ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: Sequence[int], to_smiles: bool = True) -> str:
        del to_smiles
        out: list[str] = []
        for idx in ids:
            tid = int(idx)
            if tid == self.BOS_ID:
                continue
            if tid in (self.EOS_ID, self.PAD_ID):
                break
            out.append(self.id_to_token.get(tid, ""))
        return "".join(out)


class _PocketEncoderOutput(NamedTuple):
    node_emb: Tensor
    pooled_emb: Tensor
    vec_emb: Tensor | None


@dataclass
class _ModelOutput:
    logits: Tensor
    attn_weights: Tensor | None
    pocket_out: _PocketEncoderOutput
    mu_q: Tensor | None
    logvar_q: Tensor | None
    mu_p: Tensor | None
    logvar_p: Tensor | None
    pred_pkd: Tensor | None
    pred_props: Tensor | None
    clip_pocket: Tensor | None
    clip_ligand: Tensor | None
    clip_logit_scale: Tensor | None


class _ToyDecoder(nn.Module):
    """Compact decoder supporting memory=None (Stage-1), cross-attn, and backbone loading."""

    def __init__(self, vocab_size: int = 12, d_model: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.self_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_gate = nn.Parameter(torch.tensor(0.5))
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.pretrained_loaded = False

    def load_pretrained_backbone(self, state_dict: dict[str, Tensor]) -> None:
        own = self.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in own and "cross" not in k and own[k].shape == v.shape
        }
        self.load_state_dict(filtered, strict=False)
        self.pretrained_loaded = True

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        h = self.norm(self.tok_emb(input_ids) + torch.tanh(self.self_proj(self.tok_emb(input_ids))))
        attn_out: Tensor | None = None
        if memory is not None:
            bsz, seq_len, _ = h.shape
            mem_len = memory.size(1)
            q = self.cross_attn_q(h)
            k = self.cross_attn_k(memory)
            v = self.cross_attn_v(memory)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
            if memory_mask is not None:
                mask = memory_mask.unsqueeze(1).expand(bsz, seq_len, mem_len)
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            if memory_mask is not None:
                weights = weights * memory_mask.unsqueeze(1).to(dtype=weights.dtype)
            ctx = torch.matmul(weights, v)
            h = h + torch.tanh(self.cross_gate) * ctx
            if return_attn:
                attn_out = weights.unsqueeze(1).expand(bsz, self.n_heads, seq_len, mem_len)
        logits = self.lm_head(h)
        return logits, attn_out


class _ToyCVAE(nn.Module):
    """Minimal pocket CVAE implementing sample_prior and refine."""

    def __init__(
        self,
        d_model: int = 32,
        latent_dim: int = 8,
        num_tokens: int = 2,
        fp_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.prior_mu = nn.Linear(d_model, latent_dim)
        self.prior_logvar = nn.Linear(d_model, latent_dim)
        self.post_mu = nn.Linear(d_model + fp_dim, latent_dim)
        self.post_logvar = nn.Linear(d_model + fp_dim, latent_dim)
        self.z_proj = nn.Linear(latent_dim, num_tokens * d_model)

    def sample_prior(self, pooled_emb: Tensor, latent_temp: float = 1.0) -> Tensor:
        mu = self.prior_mu(pooled_emb)
        logvar = self.prior_logvar(pooled_emb).clamp(-6.0, 2.0)
        std = torch.exp(0.5 * logvar) * float(latent_temp)
        z = mu + std * torch.randn_like(std)
        return self.z_proj(z).reshape(pooled_emb.size(0), self.num_tokens, self.d_model)

    def refine(self, pooled_emb: Tensor, ligand_fp: Tensor, latent_temp: float = 0.5) -> Tensor:
        cat = torch.cat([pooled_emb, ligand_fp], dim=-1)
        mu = self.post_mu(cat)
        logvar = self.post_logvar(cat).clamp(-6.0, 2.0)
        std = torch.exp(0.5 * logvar) * float(latent_temp)
        z = mu + std * torch.randn_like(std)
        return self.z_proj(z).reshape(pooled_emb.size(0), self.num_tokens, self.d_model)


class _ToyPocketModel(nn.Module):
    """Contract-faithful mock GenEuclideanModel for fast, deterministic unit tests."""

    def __init__(
        self,
        vocab_size: int = 12,
        node_in_dim: int = 8,
        d_model: int = 32,
        fp_dim: int = 16,
        clip_dim: int = 16,
        use_cvae: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder_proj = nn.Linear(node_in_dim, d_model)
        self.decoder = _ToyDecoder(vocab_size=vocab_size, d_model=d_model)
        self.cvae: _ToyCVAE | None = (
            _ToyCVAE(d_model=d_model, latent_dim=8, num_tokens=2, fp_dim=fp_dim)
            if use_cvae
            else None
        )
        self.null_pocket_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.5)
        self.pkd_head = nn.Linear(d_model + fp_dim, 1)
        self.prop_head = nn.Linear(d_model, 6)
        self.clip_p = nn.Linear(d_model, clip_dim)
        self.clip_l = nn.Linear(fp_dim, clip_dim)
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.encode_calls = 0

    def encode_pocket(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> _PocketEncoderOutput:
        del coords
        self.encode_calls += 1
        node_emb = self.encoder_proj(node_feats) * atom_mask.unsqueeze(-1).to(node_feats.dtype)
        denom = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(node_feats.dtype)
        pooled_emb = node_emb.sum(dim=1) / denom
        return _PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=None)

    def build_memory(
        self,
        pocket_out: _PocketEncoderOutput,
        atom_mask: Tensor,
        z_tokens: Tensor | None = None,
        use_null_pocket: bool = False,
    ) -> tuple[Tensor, Tensor]:
        bsz = pocket_out.node_emb.size(0)
        if use_null_pocket:
            mem = self.null_pocket_token.expand(bsz, 1, -1).contiguous()
            mask = torch.ones((bsz, 1), dtype=torch.bool, device=mem.device)
            return mem, mask
        mem = pocket_out.node_emb
        mask = atom_mask.to(dtype=torch.bool)
        if z_tokens is not None:
            z_mask = torch.ones((bsz, z_tokens.size(1)), dtype=torch.bool, device=mem.device)
            mem = torch.cat([mem, z_tokens], dim=1)
            mask = torch.cat([mask, z_mask], dim=1)
        return mem, mask

    def forward(
        self,
        batch: dict[str, Any],
        cfg_drop: bool = False,
        return_attn: bool = False,
    ) -> _ModelOutput:
        del cfg_drop
        pocket_out = self.encode_pocket(batch["coords"], batch["node_feats"], batch["atom_mask"])
        fp = batch["ligand_fp"]

        mu_p = logvar_p = mu_q = logvar_q = z_tokens = None
        if self.cvae is not None:
            mu_p = self.cvae.prior_mu(pocket_out.pooled_emb)
            logvar_p = self.cvae.prior_logvar(pocket_out.pooled_emb)
            cat = torch.cat([pocket_out.pooled_emb, fp], dim=-1)
            mu_q = self.cvae.post_mu(cat)
            logvar_q = self.cvae.post_logvar(cat)
            z = mu_q + torch.exp(0.5 * logvar_q.clamp(-6.0, 2.0)) * torch.randn_like(mu_q)
            z_tokens = self.cvae.z_proj(z).reshape(
                pocket_out.pooled_emb.size(0),
                self.cvae.num_tokens,
                self.d_model,
            )

        mem, mask = self.build_memory(pocket_out, batch["atom_mask"], z_tokens=z_tokens)
        logits, attn = self.decoder(
            batch["input_ids"],
            memory=mem,
            memory_mask=mask,
            return_attn=return_attn,
        )
        pred_pkd = self.pkd_head(torch.cat([pocket_out.pooled_emb, fp], dim=-1)).squeeze(-1)
        pred_props = self.prop_head(pocket_out.pooled_emb)
        clip_pocket = F.normalize(self.clip_p(pocket_out.pooled_emb), p=2, dim=-1)
        clip_ligand = F.normalize(self.clip_l(fp), p=2, dim=-1)
        return _ModelOutput(
            logits=logits,
            attn_weights=attn,
            pocket_out=pocket_out,
            mu_q=mu_q,
            logvar_q=logvar_q,
            mu_p=mu_p,
            logvar_p=logvar_p,
            pred_pkd=pred_pkd,
            pred_props=pred_props,
            clip_pocket=clip_pocket,
            clip_ligand=clip_ligand,
            clip_logit_scale=self.clip_logit_scale,
        )


class _ScriptedCFGDecoder(nn.Module):
    """Deterministic decoder outputting fixed cond vs uncond logits for exact CFG tests."""

    def __init__(self, cond_logits: Tensor, uncond_logits: Tensor) -> None:
        super().__init__()
        self.register_buffer("cond_logits", cond_logits.float())
        self.register_buffer("uncond_logits", uncond_logits.float())
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        del return_attn
        assert memory is not None
        bsz, seq_len = input_ids.shape
        # Distinguish conditional (positive marker in memory[:, 0, 0]) vs null (negative marker)
        is_cond = memory[:, 0, 0] > 0.0
        cond_t = self.cond_logits.to(device=memory.device).unsqueeze(0).expand(bsz, -1)
        uncond_t = self.uncond_logits.to(device=memory.device).unsqueeze(0).expand(bsz, -1)
        step_logits = torch.where(is_cond.unsqueeze(-1), cond_t, uncond_t)
        out = step_logits.unsqueeze(1).expand(bsz, seq_len, -1).contiguous()
        attn = None
        if memory_mask is not None:
            attn = memory_mask.float().unsqueeze(1).unsqueeze(1).expand(bsz, 1, seq_len, -1)
        return out, attn


class _ScriptedPocketModel(nn.Module):
    """Model wrapping a scripted decoder with distinct cond vs null-pocket memory lengths."""

    def __init__(self, cond_logits: Tensor, uncond_logits: Tensor) -> None:
        super().__init__()
        self.decoder = _ScriptedCFGDecoder(cond_logits, uncond_logits)
        self.cvae = None

    def encode_pocket(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> _PocketEncoderOutput:
        del coords
        bsz, n_atoms, _ = node_feats.shape
        node_emb = torch.ones((bsz, n_atoms, 4), device=node_feats.device) * atom_mask.unsqueeze(-1)
        pooled = torch.ones((bsz, 4), device=node_feats.device)
        return _PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled, vec_emb=None)

    def build_memory(
        self,
        pocket_out: _PocketEncoderOutput,
        atom_mask: Tensor,
        z_tokens: Tensor | None = None,
        use_null_pocket: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del z_tokens
        bsz = pocket_out.node_emb.size(0)
        if use_null_pocket:
            mem = -torch.ones((bsz, 1, 4), device=pocket_out.node_emb.device)
            mask = torch.ones((bsz, 1), dtype=torch.bool, device=mem.device)
            return mem, mask
        return pocket_out.node_emb, atom_mask.to(dtype=torch.bool)


def _make_pocket_batch(
    batch_size: int = 4,
    n_atoms: int = 10,
    node_dim: int = 8,
    seq_len: int = 7,
    fp_dim: int = 16,
    seed: int = 0,
    include_nan_pkd: bool = False,
) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    coords = torch.randn((batch_size, n_atoms, 3), generator=gen)
    node_feats = torch.randn((batch_size, n_atoms, node_dim), generator=gen)
    atom_mask = torch.ones((batch_size, n_atoms), dtype=torch.bool)
    atom_mask[:, -2:] = False  # Two padded dummy atoms
    contact_mask = torch.zeros((batch_size, n_atoms), dtype=torch.bool)
    contact_mask[:, :3] = True
    # Valid toy SMILES token sequences: [BOS, C, C, O, EOS, PAD, PAD]
    input_ids = torch.tensor([[1, 4, 4, 6, 4, 0, 0]] * batch_size, dtype=torch.long)[:, :seq_len]
    target_ids = torch.tensor([[4, 4, 6, 4, 2, 0, 0]] * batch_size, dtype=torch.long)[:, :seq_len]
    token_mask = target_ids != 0
    pkd = torch.linspace(5.0, 9.0, steps=batch_size)
    if include_nan_pkd:
        pkd[0] = float("nan")
    ligand_fp = torch.rand((batch_size, fp_dim), generator=gen)
    ligand_props = torch.randn((batch_size, 6), generator=gen)
    return {
        "pdb_ids": [f"1ab{i}" for i in range(batch_size)],
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "contact_mask": contact_mask,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "token_mask": token_mask,
        "pkd": pkd,
        "ligand_fp": ligand_fp,
        "ligand_props": ligand_props,
        "canonical_smiles": ["CCCO"] * batch_size,
    }


# ---------------------------------------------------------------------------
# 1. Decoding strategy mathematical exactness tests
# ---------------------------------------------------------------------------


def test_temperature_exactness_and_no_double_softmax() -> None:
    logits = torch.tensor([[2.5, 1.0, -0.5, 0.2, 3.1]])

    # T=1.0 must match exact F.softmax(logits) (never double softmax!)
    probs_t1 = compute_temperature_probs(logits, temperature=1.0)
    expected_t1 = F.softmax(logits, dim=-1)
    double_softmax = F.softmax(expected_t1, dim=-1)
    assert torch.allclose(probs_t1, expected_t1, atol=1e-6)
    assert not torch.allclose(probs_t1, double_softmax, atol=1e-2)

    # T -> 0 must approach greedy one-hot argmax
    probs_cold = compute_temperature_probs(logits, temperature=1e-3)
    greedy_probs = compute_strategy_probs(logits, strategy="greedy")
    assert torch.allclose(probs_cold, greedy_probs, atol=1e-5)
    assert int(probs_cold.argmax(dim=-1).item()) == 4

    with pytest.raises(ValueError, match="strictly positive"):
        compute_temperature_probs(logits, temperature=0.0)


def test_top_k_and_top_p_truncation_exactness() -> None:
    logits = torch.tensor([[5.0, 3.0, 1.0, -1.0, -3.0]])
    top2_logits = filter_top_k(logits, top_k=2)
    assert torch.isfinite(top2_logits[0, :2]).all()
    assert torch.isneginf(top2_logits[0, 2:]).all()

    probs = torch.tensor([[0.50, 0.30, 0.12, 0.05, 0.03]])
    # top_p=0.75 must keep indices 0 (cum 0.50) and 1 (cum 0.80 >= 0.75) and zero out 2,3,4
    top_p_out = filter_top_p_probs(probs, top_p=0.75)
    assert torch.isclose(top_p_out.sum(), torch.tensor(1.0), atol=1e-6)
    assert top_p_out[0, 0] > 0.0 and top_p_out[0, 1] > 0.0
    assert torch.all(top_p_out[0, 2:] == 0.0)
    assert torch.isclose(top_p_out[0, 0], torch.tensor(0.50 / 0.80), atol=1e-6)

    # Even tiny top_p keeps at least top-1 token
    top_p_tiny = filter_top_p_probs(probs, top_p=0.01)
    assert torch.isclose(top_p_tiny[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.all(top_p_tiny[0, 1:] == 0.0)


def test_min_p_threshold_exactness() -> None:
    # Exact probabilities: max=0.60, min_p=0.25 => threshold=0.15
    # Kept: 0.60 (>=0.15), 0.20 (>=0.15), 0.15 (>=0.15); dropped: 0.04, 0.01
    probs = torch.tensor([[0.60, 0.20, 0.15, 0.04, 0.01]])
    out = filter_min_p_probs(probs, min_p=0.25)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(out[0, :3] > 0.0)
    assert torch.all(out[0, 3:] == 0.0)
    expected = torch.tensor([[0.60, 0.20, 0.15, 0.0, 0.0]]) / 0.95
    assert torch.allclose(out, expected, atol=1e-6)

    # min_p=1.0 collapses strictly to argmax
    out_max = filter_min_p_probs(probs, min_p=1.0)
    assert torch.isclose(out_max[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.all(out_max[0, 1:] == 0.0)


def test_locally_typical_sampling_prefers_entropy_matched_tokens() -> None:
    # Distribution with one ultra-high prob token (low surprisal) and medium tokens near H
    probs_raw = torch.tensor([[0.70, 0.14, 0.13, 0.02, 0.01]])
    logits = torch.log(probs_raw)
    out = filter_typical_probs(logits, temperature=1.0, mass=0.75)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)
    # Tail tokens with extreme surprisal (-log(0.01)=4.6 >> H≈0.90) must be dropped
    assert out[0, 3] == 0.0 and out[0, 4] == 0.0
    # Verify exact manual ordering by | -log p - H |
    log_p = F.log_softmax(logits, dim=-1)
    entropy = -(probs_raw * log_p).sum(dim=-1, keepdim=True)
    dev = (-log_p - entropy).abs()
    most_typical_idx = int(dev.argmin(dim=-1).item())
    assert out[0, most_typical_idx] > 0.0


def test_cfg_exact_identities_and_batched_sampling() -> None:
    cond = torch.tensor([[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    uncond = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # Exact algebraic identities
    assert torch.allclose(combine_cfg_logits(cond, uncond, 0.0), uncond)
    assert torch.allclose(combine_cfg_logits(cond, uncond, 1.0), cond)
    guided = combine_cfg_logits(cond, uncond, 2.0)
    assert torch.allclose(guided, uncond + 2.0 * (cond - uncond))
    # w > 1 strictly amplifies conditional shift relative to unconditional
    assert (guided[0, 4] - guided[0, 5]) > (cond[0, 4] - cond[0, 5])

    # End-to-end sampling with different memory lengths (cond length=10 vs null length=1)
    tokenizer = _MockTokenizer()
    model = _ScriptedPocketModel(cond_logits=cond.squeeze(0), uncond_logits=uncond.squeeze(0))
    batch = _make_pocket_batch(batch_size=1)

    # w=0.0 => purely unconditional => emits token 5 ('N')
    cfg_w0 = SamplingConfig(
        strategy="greedy", num_samples=2, max_len=4, cfg_scale=0.0, repetition_penalty=1.0
    )
    smi_w0 = sample_molecules(model, batch, tokenizer, cfg_w0)
    assert smi_w0 == ["NNNN", "NNNN"]

    # w=1.0 => purely conditional => emits token 4 ('C')
    cfg_w1 = SamplingConfig(
        strategy="greedy", num_samples=2, max_len=4, cfg_scale=1.0, repetition_penalty=1.0
    )
    smi_w1 = sample_molecules(model, batch, tokenizer, cfg_w1)
    assert smi_w1 == ["CCCC", "CCCC"]

    # w=2.0 => amplified conditional => still emits 'C' cleanly with batched [cond; uncond]
    cfg_w2 = SamplingConfig(
        strategy="greedy", num_samples=2, max_len=4, cfg_scale=2.0, repetition_penalty=1.0
    )
    smi_w2 = sample_molecules(model, batch, tokenizer, cfg_w2)
    assert smi_w2 == ["CCCC", "CCCC"]


def test_repetition_penalty_and_smiles_syntax_guardrails() -> None:
    logits = torch.tensor([[1.0, 2.0, -2.0, 4.0, -1.0, 3.0]])
    generated = torch.tensor([[1, 3, 4]])  # BOS=1 ignored, token 3 (+4) & 4 (-1) penalized
    penalized = apply_repetition_penalty(logits, generated, penalty=2.0, ignore_ids=(0, 1, 2))
    assert torch.isclose(penalized[0, 1], torch.tensor(2.0))  # BOS unchanged
    assert torch.isclose(penalized[0, 3], torch.tensor(2.0))  # +4 / 2 = +2
    assert torch.isclose(penalized[0, 4], torch.tensor(-2.0))  # -1 * 2 = -2
    assert torch.isclose(penalized[0, 5], torch.tensor(3.0))  # unseen unchanged

    tokenizer = _MockTokenizer()
    # Unclosed '(' and open ring '1' => EOS (2) must be masked; ')' allowed
    seq_open = torch.tensor([[1, 4, 8, 9, 4]])  # <bos> C 1 ( C
    raw_logits = torch.zeros((1, tokenizer.vocab_size))
    guarded = apply_smiles_syntax_guardrails(raw_logits, seq_open, tokenizer.id_to_token)
    assert torch.isneginf(guarded[0, tokenizer.EOS_ID])
    assert torch.isfinite(guarded[0, tokenizer.token_to_id[")"]])

    # Closed depth 0, no open rings => ')' masked, EOS allowed
    seq_closed = torch.tensor([[1, 4, 8, 4, 8]])  # <bos> C 1 C 1
    guarded_closed = apply_smiles_syntax_guardrails(raw_logits, seq_closed, tokenizer.id_to_token)
    assert torch.isneginf(guarded_closed[0, tokenizer.token_to_id[")"]])
    assert torch.isfinite(guarded_closed[0, tokenizer.EOS_ID])


def test_diverse_beam_search_distinctness_and_length_penalty() -> None:
    torch.manual_seed(7)
    tokenizer = _MockTokenizer()
    model = _ToyPocketModel(vocab_size=tokenizer.vocab_size)
    batch = _make_pocket_batch(batch_size=1)

    cfg = SamplingConfig(
        strategy="diverse_beam",
        num_samples=8,
        max_len=6,
        temperature=1.0,
        cfg_scale=1.5,
        beam_width=8,
        num_beam_groups=4,
        diversity_penalty=2.5,
        repetition_penalty=1.0,
    )
    res_low_alpha = sample_molecules_detailed(
        model, batch, tokenizer, cfg, length_penalty_alpha=0.0
    )
    res_high_alpha = sample_molecules_detailed(
        model, batch, tokenizer, cfg, length_penalty_alpha=1.2
    )

    assert len(res_low_alpha.smiles) == 8
    assert res_low_alpha.scores is not None
    # Scores must be strictly sorted descending
    assert res_low_alpha.scores == sorted(res_low_alpha.scores, reverse=True)
    # Diversity penalty across 4 groups guarantees multiple distinct beams
    assert len(set(res_low_alpha.smiles)) >= 4
    # Pocket encoded exactly once per call
    assert model.encode_calls == 2
    # Length penalty alters normalized scores
    assert res_high_alpha.scores is not None
    assert res_high_alpha.scores != res_low_alpha.scores


@pytest.mark.parametrize(
    "strategy",
    ["greedy", "temperature", "top_k", "top_p", "min_p", "typical"],
)
def test_all_sampling_strategies_cvae_refine_and_attention(strategy: str) -> None:
    torch.manual_seed(42)
    tokenizer = _MockTokenizer()
    model = _ToyPocketModel(vocab_size=tokenizer.vocab_size, use_cvae=True)
    batch = _make_pocket_batch(batch_size=1, n_atoms=10)

    cfg = SamplingConfig(
        strategy=strategy,  # type: ignore[arg-type]
        num_samples=6,
        max_len=8,
        temperature=1.05,
        min_p=0.08,
        top_p=0.90,
        top_k=5,
        cfg_scale=1.5,
        latent_temp=0.8,
        repetition_penalty=1.05,
    )
    gen = torch.Generator().manual_seed(123)
    smiles, token_lists, attn_maps = sample_with_attention(
        model,
        batch,
        tokenizer,
        cfg,
        generator=gen,
    )

    assert len(smiles) == 6
    assert len(token_lists) == 6
    assert len(attn_maps) == 6
    # Pocket must be encoded ONCE per generation call
    assert model.encode_calls == 1
    for toks, amap in zip(token_lists, attn_maps, strict=True):
        assert amap is not None
        assert amap.shape == (len(toks), 10)
        # Last 2 pocket atoms are masked out => attention weight must be EXACTLY 0.0
        if len(toks) > 0:
            assert float(amap[:, -2:].abs().max().item()) == 0.0

    # Test TamGen-style hit-to-lead CVAE posterior refinement
    seed_fp = torch.rand(16)
    refined = sample_molecules(model, batch, tokenizer, cfg, seed_ligand_fp=seed_fp)
    assert len(refined) == 6


# ---------------------------------------------------------------------------
# 2. Loss function unit tests
# ---------------------------------------------------------------------------


def test_cyclical_kl_beta_schedule() -> None:
    # 4 cycles over 100 steps => period=25, ratio=0.5 => ramp over [0, 12.5), flat [12.5, 25)
    assert cyclical_kl_beta(0, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(0.0)
    assert cyclical_kl_beta(6, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(
        0.3 * (6.0 / 12.5)
    )
    assert cyclical_kl_beta(13, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(0.3)
    assert cyclical_kl_beta(24, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(0.3)
    # Cycle 2 starts back at 0
    assert cyclical_kl_beta(25, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(0.0)
    assert cyclical_kl_beta(38, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(0.3)
    # Final step holds plateau
    assert cyclical_kl_beta(100, 100, max_beta=0.3, n_cycles=4, ratio=0.5) == pytest.approx(0.3)
    assert cyclical_kl_beta(10, 100, max_beta=0.0, n_cycles=4, ratio=0.5) == 0.0


def test_free_bits_cvae_kl_loss() -> None:
    bsz, dim = 4, 8
    mu_q = torch.zeros((bsz, dim), requires_grad=True)
    logvar_q = torch.zeros((bsz, dim), requires_grad=True)
    mu_p = torch.zeros((bsz, dim))
    logvar_p = torch.zeros((bsz, dim))

    # Identical distributions => raw KL = 0, free-bits KL = dim * free_bits, zero grad below floor
    raw = compute_cvae_kl_nats(mu_q, logvar_q, mu_p, logvar_p)
    fb_loss = compute_cvae_kl_loss(mu_q, logvar_q, mu_p, logvar_p, free_bits=0.25)
    assert float(raw.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(fb_loss.item()) == pytest.approx(dim * 0.25, abs=1e-6)
    fb_loss.backward()
    assert mu_q.grad is not None
    assert float(mu_q.grad.abs().max().item()) == 0.0

    # Shifted distribution above floor => matches analytic 0.5 * sum((mu_q - mu_p)^2)
    mu_q2 = torch.full((bsz, dim), 2.0, requires_grad=True)
    loss_hi = compute_cvae_kl_loss(mu_q2, torch.zeros((bsz, dim)), mu_p, logvar_p, free_bits=0.25)
    assert float(loss_hi.item()) == pytest.approx(dim * 2.0, abs=1e-5)
    loss_hi.backward()
    assert mu_q2.grad is not None
    assert float(mu_q2.grad.abs().min().item()) > 0.0


def test_drugclip_infonce_identity_optimum() -> None:
    eye = torch.eye(6, dtype=torch.float32)
    scale = torch.tensor(math.log(20.0))

    # Exact identity pairing has minimal loss compared to any cyclic permutation
    loss_identity = compute_drugclip_loss(eye, eye, scale)
    loss_shifted = compute_drugclip_loss(eye, torch.roll(eye, shifts=1, dims=0), scale)
    assert float(loss_identity.item()) < 1e-3
    assert float(loss_shifted.item()) > float(loss_identity.item()) + 10.0

    # Batch size 1 safely returns 0.0 without error
    loss_b1 = compute_drugclip_loss(eye[:1], eye[:1], scale)
    assert float(loss_b1.item()) == 0.0


def test_masked_huber_pkd_and_property_losses_handle_nans_safely() -> None:
    pred = torch.tensor([6.5, 7.2, 8.1, 5.4], requires_grad=True)
    target = torch.tensor([float("nan"), 7.0, float("nan"), 6.0])

    loss = compute_pkd_loss(pred, target)
    expected = F.huber_loss(pred[[1, 3]], target[[1, 3]])
    assert torch.isclose(loss, expected, atol=1e-6)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad[0].item()) == 0.0
    assert float(pred.grad[2].item()) == 0.0
    assert float(pred.grad[1].item()) != 0.0

    # All-NaN targets must still yield exact 0.0 loss and zero (not NaN!) gradients
    pred_all_nan = torch.tensor([5.0, 6.0], requires_grad=True)
    loss_all_nan = compute_pkd_loss(pred_all_nan, torch.tensor([float("nan"), float("nan")]))
    assert float(loss_all_nan.item()) == 0.0
    loss_all_nan.backward()
    assert pred_all_nan.grad is not None
    assert torch.isfinite(pred_all_nan.grad).all()
    assert torch.all(pred_all_nan.grad == 0.0)

    # Property loss with NaNs
    pred_props = torch.randn((3, 6), requires_grad=True)
    tgt_props = torch.randn((3, 6))
    tgt_props[1, :] = float("nan")
    prop_loss = compute_property_loss(pred_props, tgt_props)
    assert torch.isfinite(prop_loss)
    prop_loss.backward()
    assert pred_props.grad is not None and torch.isfinite(pred_props.grad).all()


def test_dpo_loss_value_and_gradient_direction() -> None:
    ref_pos = torch.tensor([-5.0, -4.0])
    ref_neg = torch.tensor([-6.0, -3.5])
    pi_pos = ref_pos.clone().detach().requires_grad_(True)
    pi_neg = ref_neg.clone().detach().requires_grad_(True)

    loss = compute_dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1)
    # When policy == ref, -logsigmoid(0) == log(2)
    assert float(loss.item()) == pytest.approx(math.log(2.0), abs=1e-6)
    loss.backward()
    assert pi_pos.grad is not None and pi_neg.grad is not None
    # Gradient descent (-grad) must INCREASE pi_pos (grad < 0) and DECREASE pi_neg (grad > 0)
    assert bool((pi_pos.grad < 0.0).all().item())
    assert bool((pi_neg.grad > 0.0).all().item())


def test_lm_and_multitask_loss() -> None:
    cfg = ExperimentConfig()
    model = _ToyPocketModel()
    batch = _make_pocket_batch(batch_size=4, include_nan_pkd=True)
    out = model(batch)

    total_loss, metrics = compute_multitask_loss(out, batch, cfg, kl_beta=0.15)
    assert torch.isfinite(total_loss)
    for key in (
        "total_loss",
        "lm_loss",
        "ppl",
        "kl_loss",
        "kl_nats",
        "pkd_loss",
        "clip_loss",
        "prop_loss",
    ):
        assert key in metrics and math.isfinite(metrics[key])

    # All-PAD targets in LM loss safely return 0.0 with finite grad
    logits = torch.randn((2, 4, 12), requires_grad=True)
    all_pad = torch.zeros((2, 4), dtype=torch.long)
    lm_zero = compute_lm_loss(logits, all_pad, pad_id=0)
    assert float(lm_zero.item()) == 0.0
    lm_zero.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# 3. Training loops (Stage-1 pretrain, Stage-2 PocketTrainer, Stage-3 DPO)
# ---------------------------------------------------------------------------


def test_stage1_pretrain_stage2_trainer_and_stage3_dpo(tmp_path: Path) -> None:
    tokenizer = _MockTokenizer()

    # Stage-1: Unconditional ZINC chemical prior pretraining
    s1_cfg = ExperimentConfig()
    s1_cfg.train.device = "cpu"
    s1_cfg.train.epochs = 2
    s1_cfg.train.lr = 1e-3
    s1_decoder = _ToyDecoder(vocab_size=tokenizer.vocab_size, d_model=32)
    s1_batches = [
        {
            "input_ids": torch.tensor([[1, 4, 4, 6, 4, 0], [1, 4, 5, 4, 0, 0]], dtype=torch.long),
            "target_ids": torch.tensor([[4, 4, 6, 4, 2, 0], [4, 5, 4, 2, 0, 0]], dtype=torch.long),
        }
    ]
    prior_dir = tmp_path / "prior"
    s1_res = pretrain_chemical_prior(
        s1_cfg,
        train_loader=s1_batches,
        val_loader=s1_batches,
        tokenizer=tokenizer,
        save_dir=prior_dir,
        model=s1_decoder,
    )
    prior_ckpt = Path(s1_res["best_ckpt_path"])
    assert prior_ckpt.is_file()
    assert (prior_dir / "history.jsonl").is_file()
    ckpt_data = torch.load(prior_ckpt, map_location="cpu", weights_only=False)
    for k in ("model_state_dict", "config", "val_loss", "val_ppl"):
        assert k in ckpt_data

    # Stage-2: PocketTrainer with Stage-1 loading, differential LRs, gradual unfreezing, resume
    s2_cfg = ExperimentConfig()
    s2_cfg.train.device = "cpu"
    s2_cfg.train.epochs = 2
    s2_cfg.train.encoder_lr = 5e-4
    s2_cfg.train.decoder_lr = 1e-4
    s2_cfg.train.freeze_decoder_epochs = 1
    s2_cfg.train.pretrained_decoder_ckpt = str(prior_ckpt)

    model = _ToyPocketModel(vocab_size=tokenizer.vocab_size, d_model=32)
    trainer = PocketTrainer(model=model, cfg=s2_cfg, tokenizer=tokenizer, device="cpu")
    assert model.decoder.pretrained_loaded is True

    opt = trainer.build_optimizer()
    lr_map = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lr_map["encoder_decay"] == pytest.approx(5e-4)
    assert lr_map["decoder_decay"] == pytest.approx(1e-4)

    train_batches = [
        _make_pocket_batch(batch_size=4, seed=1),
        _make_pocket_batch(batch_size=4, seed=2),
    ]
    val_batches = [_make_pocket_batch(batch_size=6, seed=3)]
    run_dir = tmp_path / "stage2"
    s2_res = trainer.fit(train_batches, val_batches, output_dir=run_dir)

    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()
    hist_lines = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert len(hist_lines) == 2
    assert hist_lines[0]["decoder_frozen"] == 1.0
    assert hist_lines[1]["decoder_frozen"] == 0.0
    for key in (
        "val_lm_loss",
        "val_ppl",
        "val_pkd_pearson",
        "val_pkd_rmse",
        "val_clip_r1",
        "val_clip_r5",
    ):
        assert key in hist_lines[-1]
    assert math.isfinite(s2_res["best_val_objective"])

    # Checkpoint resume from epoch 2 -> epoch 3
    s2_cfg_resume = s2_cfg.model_copy(deep=True)
    s2_cfg_resume.train.epochs = 3
    s2_cfg_resume.train.pretrained_decoder_ckpt = None
    model_resume = _ToyPocketModel(vocab_size=tokenizer.vocab_size, d_model=32)
    trainer_resume = PocketTrainer(
        model=model_resume,
        cfg=s2_cfg_resume,
        tokenizer=tokenizer,
        device="cpu",
    )
    resumed = trainer_resume.fit(
        train_batches,
        val_batches,
        output_dir=run_dir,
        resume_from=run_dir / "last.pt",
    )
    assert trainer_resume.current_epoch == 3
    assert len(resumed["history"]) == 3

    # Stage-3: MPO-guided DPO alignment
    pocket_single = _make_pocket_batch(batch_size=1, seed=9)
    pairs = build_preference_pairs_from_mpo(
        pocket_single,
        smiles_list=["CCCO", "CCN", "CCC", "C"],
        mpo_scores=[0.90, 0.70, 0.30, 0.10],
        min_margin=0.10,
        max_pairs=2,
    )
    assert len(pairs) == 2
    ref_model = copy.deepcopy(model)
    dpo_res = run_dpo_alignment(
        model=model,
        ref_model=ref_model,
        preference_pairs=pairs,
        cfg=s2_cfg,
        device="cpu",
        tokenizer=tokenizer,
        epochs=2,
        beta=0.1,
        lr=1e-3,
        save_dir=tmp_path / "dpo",
    )
    assert dpo_res["epochs"] == 2
    assert Path(dpo_res["ckpt_path"]).is_file()
    assert dpo_res["history"][-1]["dpo_loss"] <= dpo_res["history"][0]["dpo_loss"] + 1e-4


# ---------------------------------------------------------------------------
# 4. MPO scoring & MaxMin diversity selection tests
# ---------------------------------------------------------------------------


def test_mpo_scoring_and_pains_and_invalid_handling() -> None:
    # Drug-like aspirin, caffeine, classic PAINS rhodanine derivative, and invalid SMILES
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    caffeine = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    rhodanine_pains = "O=C1NC(=S)SC1=Cc1ccccc1"
    invalid = "not_a_valid_smiles(("

    scores = compute_mpo_scores(
        [aspirin, caffeine, rhodanine_pains, invalid],
        pred_pkd=[8.5, 7.2, 8.5, 9.0],
        clip_sim=[0.8, 0.5, 0.8, 0.9],
    )
    assert len(scores) == 4
    assert scores[0].valid and scores[0].pains_free and scores[0].lipinski_pass
    assert 0.0 < scores[0].qed <= 1.0
    assert 1.0 <= scores[0].sa_score <= 10.0
    assert 0.0 <= scores[0].sa_norm <= 1.0
    assert scores[1].valid and scores[1].pains_free
    # Rhodanine matches PAINS filter and receives strict downranking relative to aspirin
    assert scores[2].valid and not scores[2].pains_free
    assert scores[0].mpo_score > scores[2].mpo_score
    # Invalid SMILES strictly receives 0.0
    assert not scores[3].valid and scores[3].mpo_score == 0.0


def test_maxmin_diversity_selection_guarantees_tanimoto_distance() -> None:
    # Cluster A (near-identical ethanol/propanol/non-canonical duplicate),
    # Cluster B (aspirin & close ester), Cluster C (caffeine), Cluster D (ibuprofen), Invalid
    smiles = [
        "CCO",
        "OCC",  # Non-canonical duplicate of ethanol
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CCC(=O)Oc1ccccc1C(=O)O",  # Close aspirin analog
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # Caffeine
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        "O=C(Nc1ccccc1)c1ccncc1",  # Distinct benzamide
        "INVALID_SMILES",
    ]
    raw_scores = [0.60, 0.65, 0.95, 0.94, 0.85, 0.82, 0.78, 0.99]

    min_dist = 0.45
    chosen = maxmin_diversity_select(smiles, raw_scores, k=4, min_tanimoto_dist=min_dist)
    assert len(chosen) == 4
    # First selected hit MUST be highest-scoring valid molecule (aspirin idx 2, not invalid idx 7)
    assert chosen[0] == 2

    # Verify strict pairwise Morgan Tanimoto distance >= min_dist across ALL selected pairs
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fp_gen.GetFingerprint(Chem.MolFromSmiles(smiles[i])) for i in chosen]
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dist = 1.0 - float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            assert dist >= min_dist - 1e-9

    # Deduplication guarantees identical copies return at most 1 hit when strict
    clones = ["CCO", "OCC", "CCO"]
    assert len(maxmin_diversity_select(clones, [0.5, 0.9, 0.7], k=3, min_tanimoto_dist=0.35)) == 1

    # Integrated rerank_and_select convenience helper
    top_hits = rerank_and_select(
        smiles,
        k=3,
        pred_pkd=np.array([5.5, 5.6, 8.8, 8.7, 7.9, 8.1, 7.5, 9.9]),
        min_tanimoto_dist=0.45,
    )
    assert len(top_hits) == 3
    assert all(hit.valid for hit in top_hits)


# ---------------------------------------------------------------------------
# 5. Experiment YAML config validation tests
# ---------------------------------------------------------------------------


EXPECTED_CONFIGS = [
    "pretrain_zinc.yaml",
    "pretrain_zinc_selfies.yaml",
    "e3nn_lstm_v15.yaml",
    "painn_pocketformer.yaml",
    "geneuclidean_v2_painn.yaml",
    "geneuclidean_v2_e3nn.yaml",
    "geneuclidean_v2_egnn.yaml",
    "geneuclidean_v2_schnet.yaml",
    "geneuclidean_v2_selfies.yaml",
]


@pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
def test_all_experiment_yaml_configs_validate(filename: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "configs" / filename
    assert cfg_path.is_file(), f"Missing required config: {cfg_path}"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert cfg.name
    assert cfg.encoder.node_in_dim == 52


def test_every_yaml_in_configs_dir_is_valid() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    yaml_files = sorted((repo_root / "configs").glob("*.yaml"))
    assert len(yaml_files) >= len(EXPECTED_CONFIGS)
    for path in yaml_files:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.decoder.vocab_size >= 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS unavailable")
def test_mps_pretrain_trainer_and_sampling_smoke(tmp_path: Path) -> None:
    tokenizer = _MockTokenizer()
    cfg = ExperimentConfig()
    cfg.train.device = "mps"
    cfg.train.epochs = 1
    cfg.train.freeze_decoder_epochs = 0

    decoder = _ToyDecoder(vocab_size=tokenizer.vocab_size, d_model=32)
    s1_batches = [
        {
            "input_ids": torch.tensor([[1, 4, 4, 6, 4, 0]], dtype=torch.long),
            "target_ids": torch.tensor([[4, 4, 6, 4, 2, 0]], dtype=torch.long),
        }
    ]
    pretrain_res = pretrain_chemical_prior(
        cfg,
        train_loader=s1_batches,
        val_loader=s1_batches,
        tokenizer=tokenizer,
        save_dir=tmp_path / "mps_prior",
        model=decoder,
    )
    assert math.isfinite(pretrain_res["best_val_loss"])

    model = _ToyPocketModel(vocab_size=tokenizer.vocab_size, d_model=32)
    trainer = PocketTrainer(model=model, cfg=cfg, tokenizer=tokenizer, device="mps")
    batch = _make_pocket_batch(batch_size=4, seed=11)
    fit_res = trainer.fit([batch], [batch], output_dir=tmp_path / "mps_stage2")
    assert math.isfinite(fit_res["best_val_objective"])

    s_cfg = SamplingConfig(
        strategy="min_p",
        num_samples=4,
        max_len=6,
        cfg_scale=1.5,
        min_p=0.08,
        temperature=1.05,
    )
    smiles, token_lists, attn_maps = sample_with_attention(
        model,
        _make_pocket_batch(batch_size=1, seed=12),
        tokenizer,
        s_cfg,
        device=torch.device("mps"),
    )
    assert len(smiles) == 4
    for toks, amap in zip(token_lists, attn_maps, strict=True):
        assert amap is not None
        assert amap.shape[0] == len(toks)


def test_trainer_vocab_mismatch_and_elbo_val_objective(tmp_path: Path) -> None:
    tokenizer = _MockTokenizer()
    cfg = ExperimentConfig()
    cfg.train.device = "cpu"

    # 1. Decoder vocab_size != tokenizer.vocab_size must raise ValueError immediately
    bad_model = _ToyPocketModel(vocab_size=tokenizer.vocab_size + 5, d_model=32)
    with pytest.raises(ValueError, match="vocab_size"):
        PocketTrainer(model=bad_model, cfg=cfg, tokenizer=tokenizer, device="cpu")

    # 2. Pretrained checkpoint with mismatched embedding shape must raise ValueError
    bad_ckpt_path = tmp_path / "bad_vocab_prior.pt"
    wrong_decoder = _ToyDecoder(vocab_size=tokenizer.vocab_size + 4, d_model=32)
    torch.save(
        {"model_state_dict": wrong_decoder.state_dict(), "vocab_size": wrong_decoder.vocab_size},
        bad_ckpt_path,
    )
    cfg_bad_ckpt = cfg.model_copy(deep=True)
    cfg_bad_ckpt.train.pretrained_decoder_ckpt = str(bad_ckpt_path)
    good_model = _ToyPocketModel(vocab_size=tokenizer.vocab_size, d_model=32)
    with pytest.raises(ValueError, match="vocab_size"):
        PocketTrainer(model=good_model, cfg=cfg_bad_ckpt, tokenizer=tokenizer, device="cpu")

    # 3. Raw state_dict checkpoint missing token embedding weight must raise ValueError
    missing_emb_ckpt = tmp_path / "missing_emb.pt"
    stripped = {
        k: v
        for k, v in _ToyDecoder(vocab_size=tokenizer.vocab_size, d_model=32).state_dict().items()
        if "tok_emb" not in k
    }
    torch.save(stripped, missing_emb_ckpt)
    cfg_missing = cfg.model_copy(deep=True)
    cfg_missing.train.pretrained_decoder_ckpt = str(missing_emb_ckpt)
    with pytest.raises(ValueError, match="missing 'tok_emb.weight'"):
        PocketTrainer(model=good_model, cfg=cfg_missing, tokenizer=tokenizer, device="cpu")

    # 4. val_objective includes fixed-weight CVAE ELBO term (kl_max_beta * val_kl_loss)
    cfg_elbo = cfg.model_copy(deep=True)
    cfg_elbo.cvae.enabled = True
    cfg_elbo.cvae.kl_max_beta = 0.3
    cfg_elbo.multitask.use_pkd_head = False
    cfg_elbo.multitask.use_clip_head = False
    cfg_elbo.multitask.use_prop_head = False
    trainer = PocketTrainer(model=good_model, cfg=cfg_elbo, tokenizer=tokenizer, device="cpu")
    val_metrics = trainer.evaluate([_make_pocket_batch(batch_size=4, seed=5)])
    expected_obj = val_metrics["val_lm_loss"] + 0.3 * val_metrics["val_kl_loss"]
    assert val_metrics["val_kl_loss"] > 0.0
    assert val_metrics["val_objective"] == pytest.approx(expected_obj, rel=1e-6)


def test_cvae_refine_handles_value_error_ligand_encoder_fallback() -> None:
    tokenizer = _MockTokenizer()

    class _StrictDimCVAE(nn.Module):
        def refine(
            self, pooled_emb: Tensor, ligand_emb: Tensor, latent_temp: float = 0.5
        ) -> Tensor:
            del latent_temp
            if ligand_emb.size(-1) != pooled_emb.size(-1):
                raise ValueError("Expected pre-encoded ligand embedding dim")
            return (pooled_emb + ligand_emb).unsqueeze(1)

    class _MockHeads(nn.Module):
        def __init__(self, fp_dim: int, d_model: int) -> None:
            super().__init__()
            self.proj = nn.Linear(fp_dim, d_model)

        def encode_ligand(self, fp: Tensor) -> Tensor:
            return self.proj(fp)

    model = _ToyPocketModel(vocab_size=tokenizer.vocab_size, d_model=32, fp_dim=16, use_cvae=False)
    model.cvae = _StrictDimCVAE()  # type: ignore[assignment]
    model.heads = _MockHeads(fp_dim=16, d_model=32)  # type: ignore[assignment]

    cfg = SamplingConfig(strategy="greedy", num_samples=2, max_len=4, cfg_scale=1.0)
    smiles = sample_molecules(
        model,
        _make_pocket_batch(batch_size=1, fp_dim=16),
        tokenizer,
        cfg,
        seed_ligand_fp=torch.rand(16),
    )
    assert len(smiles) == 2
