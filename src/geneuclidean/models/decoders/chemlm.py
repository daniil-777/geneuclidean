"""Scalable pocket-conditioned chemical LLM decoders (``chemlm`` / ``chemlm_moe``).

This module closes the architectural gap between GenEuclidean's two historical decoder
stacks. The registry-dispatched
:class:`~geneuclidean.models.decoders.transformer.PocketTransformerDecoder` used by
:class:`~geneuclidean.models.geneuclidean.GenEuclideanModel` had no KV cache, no
grouped-query attention and no QK-norm, while the far stronger primitives
(:class:`~geneuclidean.models.layers.GroupedQueryAttention`,
:class:`~geneuclidean.models.layers.AttentionKVCache`,
:class:`~geneuclidean.models.nlp_ops.PocketCrossAttention`) lived outside the main
training path.

:class:`ChemLMDecoder` composes those verified primitives into a single decoder family
that is a drop-in replacement for the legacy ``transformer`` decoder — it honours the
exact ``decoder(input_ids, memory=, memory_mask=, return_attn=)`` call contract of
:class:`~geneuclidean.protocols.MolecularDecoderProtocol` — while adding:

* **Grouped-query attention + QK-norm**, so deep and ``bfloat16`` stacks train stably.
* **KV-cached incremental decoding** matching full-sequence forward to ``< 2e-4``.
* **Sparse Mixture-of-Experts feed-forward** (Switch/ST-MoE load balancing + router
  z-loss), raising parameter count without a proportional FLOP increase.
* **Continuous multi-property conditioning** via random Fourier features with an explicit
  ``unknown`` mask, enabling MolGPT-style goal-directed generation (QED, SA, MW, logP,
  TPSA, predicted affinity, BBB flag) and classifier-free guidance over properties.
* **Scale presets** from ``tiny`` (~5M) to ``xl`` (>1B parameters).
* **Gradient checkpointing**, making the larger presets trainable inside Apple Silicon
  unified memory and a single 40 GB A100.

Both ``cross_attn`` (Flamingo-style gated cross-attention over pocket memory) and
``prefix`` (Perceiver-resampled soft prefix tokens) conditioning modes are supported, plus
fully unconditional Stage-1 chemical prior pretraining via ``memory=None``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from geneuclidean.config import DecoderConfig
from geneuclidean.models.layers import (
    AttentionKVCache,
    GroupedQueryAttention,
    PerceiverLatentResampler,
    RMSNorm,
    SwiGLU,
)
from geneuclidean.models.nlp_ops import PocketCrossAttention

ConditioningMode = Literal["cross_attn", "prefix", "none"]

__all__ = [
    "SCALE_PRESETS",
    "ChemLMBlock",
    "ChemLMDecoder",
    "ConditioningMode",
    "DecoderScale",
    "FourierPropertyEmbedding",
    "PocketRelianceReport",
    "PropertyConditioner",
    "SparseMoEFeedForward",
    "build_chemlm_decoder",
    "build_chemlm_moe_decoder",
    "cross_attention_gate_magnitudes",
    "pocket_reliance_report",
    "resolve_scale",
    "scaled_decoder_config",
]


# ---------------------------------------------------------------------------
# 1. Scale presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderScale:
    """A named decoder capacity preset with an analytic parameter-count estimate.

    Attributes:
        name: Preset identifier (``tiny``, ``small``, ``base``, ``large``, ``xl``).
        d_model: Transformer hidden width.
        n_layers: Number of :class:`ChemLMBlock` layers.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads (grouped-query attention).
        d_ff: SwiGLU intermediate width (per expert when Mixture-of-Experts is enabled).
        note: Human-readable guidance on where this preset is meant to run.
    """

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"{self.name}: d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"{self.name}: n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Per-head attention dimension."""
        return self.d_model // self.n_heads

    def estimated_params(
        self,
        vocab_size: int,
        *,
        include_cross_attn: bool = True,
        num_experts: int = 1,
    ) -> float:
        """Estimate total parameter count for planning memory and compute budgets.

        Measured against the instantiated module at ``vocab_size=66``, the error is
        **under 0.1%** for every preset (tiny +0.04%, small +0.02%, base +0.01%, large
        +0.01%, xl +0.00%), so the estimate can be used directly to size a run against a
        known GPU memory budget. The residual is the handful of scalar gates and norm
        weights the closed form rounds over.

        Args:
            vocab_size: Tokenizer vocabulary size (the output head is weight-tied).
            include_cross_attn: Count gated pocket cross-attention sublayers.
            num_experts: Number of Mixture-of-Experts feed-forward experts (``1`` = dense).

        Returns:
            Estimated number of trainable parameters.
        """
        d = float(self.d_model)
        kv_dim = float(self.n_kv_heads * self.head_dim)

        attn = 2.0 * d * d + 2.0 * d * kv_dim
        qk_norm = float((self.n_heads + self.n_kv_heads) * self.head_dim)
        ffn = 3.0 * d * float(self.d_ff) * float(max(1, num_experts))
        router = d * float(num_experts) if num_experts > 1 else 0.0
        norms = 2.0 * d

        per_layer = attn + qk_norm + ffn + router + norms
        if include_cross_attn:
            per_layer += attn + qk_norm + 2.0 * d + 1.0  # + Flamingo tanh gate scalar

        return float(vocab_size) * d + float(self.n_layers) * per_layer + d


# RECOMMENDED DEFAULT: "small". Bigger is NOT better for de novo molecular design.
#
# CITE THE VERSION. arXiv:2508.13408 hosts two substantially different studies under one
# identifier: v1/v2 ("NovoMolGen: Rethinking Molecular Language Model Pretraining",
# 32M/157M/300M on Llama) and v3 ("A Large Scale Investigation of Scaling Limits in
# Chemical Language Models", 1 Oct 2026, 42 models over 0.5M-1B on Qwen3/T5Gemma). Every
# figure below is from **v3**, whose abstract reports ">30,000 experiments"; a reader who
# checks v1 or v2 will not find the 5M result and will reasonably think it invented.
#
# v3, Introduction (verbatim): "Downstream goal-directed generation largely saturates
# beyond approximately 5M parameters, despite improvements in bits per molecule and
# linear accessibility of chemical features." v3 Section 3.5 (verbatim): "Gains beyond 5M
# are marginal, the total docking score moving only from -65.8 at 5M to -66.2 at 1B."
#
# For PMO, quote v3's own framing -- Section 3.5: NovoMolGen "surpasses REINVENT ... and
# ff-RAG from 5M parameters onwards, scoring 16.45 against 16.30 for ff-RAG", and
# "[s]caling to 1B raises the total only to 16.68". (Do NOT pair the appendix numbers
# 16.48 and 16.70: those are Table I1 AtomWise-32M and Table I2 BPE-300M respectively, so
# quoting them as one 32M->300M trend silently splices two different tokenizations.
# Like-for-like it is 16.48 -> 16.64 for AtomWise and 16.68 -> 16.70 for BPE.)
#
# Xu et al. (arXiv:2601.22757, "Unveiling Scaling Behaviors in Molecular Language
# Models", 30 Jan 2026) train 300 models and fit L(P,D) = L_inf + k_P * P^-alpha +
# k_D * D^-beta per representation. For SMILES (their Table 1): alpha = 0.0171,
# beta = 0.4299. The data exponent is ~25x the parameter exponent, so per decade of
# scaling the data term falls by 10^0.4299 ~ 2.7x against 10^0.0171 ~ 1.04x for
# parameters. NOTE: that is a statement about exponents, not "25x more loss reduction" --
# each term's absolute contribution also carries k_P / k_D, which the paper does not
# report, and the ratio is our derivation rather than a claim either paper makes.
#
# Two corollaries worth acting on:
#   * Loss and sample quality can move in OPPOSITE directions. v3 Section 3.3 (verbatim):
#     "within PubChem SMILES, the FCD rises from 2.8 to 5.1 between 5M and 32M, even as
#     the loss falls from 54.1 to 48.1." Do not model-select on validation loss alone.
#     (The FCD-vs-PMO correlation r = 0.376, p = 0.358 is from **v2**, not v3.)
#   * Memorization, not invalidity, is the real failure mode. GP-MoLFormer trained on
#     1.1B SMILES reports novelty 0.323 (68% of samples are verbatim training molecules);
#     ~45% of that corpus were duplicates, and deduplicating to 650M unique lifted
#     novelty to 0.390. Deduplicate the corpus before reaching for more parameters.
#
# The "large" and "xl" presets are therefore provided for capacity experiments and
# scaling studies, not because they are expected to generate better molecules.
SCALE_PRESETS: dict[str, DecoderScale] = {
    "tiny": DecoderScale(
        name="tiny",
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        d_ff=688,
        note=(
            "~4M params. CI smoke tests and fast Mac iteration. Already past the ~5M "
            "point where goal-directed design metrics begin to saturate."
        ),
    ),
    "small": DecoderScale(
        name="small",
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        d_ff=1376,
        note=(
            "~29M params. RECOMMENDED DEFAULT: compute-optimal for de novo design and "
            "~20k tokens/s on an M3 Pro, so a ZINC-250k epoch takes minutes."
        ),
    ),
    "base": DecoderScale(
        name="base",
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        d_ff=2048,
        note=(
            "~96M params. GP-MoLFormer-class width. Comfortable on one A100; ~6k tok/s "
            "on an M3 Pro. Expect better loss but not better molecules than 'small'."
        ),
    ),
    "large": DecoderScale(
        name="large",
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ff=2752,
        note="~331M params. Capacity/scaling studies on an A100. Not a quality upgrade.",
    ),
    "xl": DecoderScale(
        name="xl",
        d_model=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,
        d_ff=5504,
        note=(
            "~1.32B params. Provided for scaling studies only. Published evidence shows "
            "no downstream design gain over ~32M; spend the compute on data and RL."
        ),
    ),
}


def resolve_scale(name: str) -> DecoderScale:
    """Look up a :class:`DecoderScale` preset by case-insensitive name.

    Args:
        name: Preset name, e.g. ``"base"`` (surrounding whitespace and case are ignored).

    Returns:
        The matching :class:`DecoderScale`.

    Raises:
        ValueError: If ``name`` is not a known preset.
    """
    key = name.strip().lower()
    if key not in SCALE_PRESETS:
        raise ValueError(f"Unknown decoder scale {name!r}. Available: {sorted(SCALE_PRESETS)}.")
    return SCALE_PRESETS[key]


def scaled_decoder_config(scale: str, **overrides: Any) -> DecoderConfig:
    """Build a validated :class:`~geneuclidean.config.DecoderConfig` from a scale preset.

    Args:
        scale: Preset name passed to :func:`resolve_scale`.
        **overrides: Any :class:`DecoderConfig` field to override (e.g. ``vocab_size``,
            ``arch``, ``max_seq_len``, ``dropout``).

    Returns:
        A fully validated :class:`DecoderConfig`.
    """
    preset = resolve_scale(scale)
    payload: dict[str, Any] = {
        "arch": "chemlm",
        "d_model": preset.d_model,
        "n_layers": preset.n_layers,
        "n_heads": preset.n_heads,
        "d_ff": preset.d_ff,
    }
    payload.update(overrides)
    return DecoderConfig(**payload)


# ---------------------------------------------------------------------------
# 2. Continuous multi-property conditioning
# ---------------------------------------------------------------------------


class FourierPropertyEmbedding(nn.Module):
    """Random Fourier feature embedding of continuous molecular property values.

    Scalar property values are poorly represented by a single linear projection, and hard
    binning throws away ordering information. Random Fourier features (Tancik et al.,
    NeurIPS 2020) give a smooth, bounded, high-frequency-capable encoding of each scalar.

    .. warning::
       **This encoding is novel in this setting and is not yet externally validated.** A
       survey of the chemical-language-model literature found continuous property
       conditioning done via property text tags (Chemlactica/Chemma), linear projection
       to prefix tokens (PolymerGPT, MolGPT) and additive injection with classifier-free
       guidance (Fraglingo), but no published use of random Fourier features, and no
       paper ablating binned against continuous property encodings. Treat this as a
       promising untested variant rather than established practice, and compare it
       against a plain linear projection before relying on it. What *is* well supported
       is the ``unknown``-dropout mechanism in :class:`PropertyConditioner`: MolGPT-style
       prefix conditioning on 7 simultaneous properties collapses to 25.7% validity,
       while a classifier-free-guidance model holds 100%.

    The projection frequencies are a fixed (non-learned) buffer drawn from a seeded
    generator, so embeddings are reproducible across processes and checkpoints. A learned
    per-property identity embedding is added so the decoder can tell *which* property a
    token encodes.

    Args:
        num_properties: Number of distinct conditioning properties ``P``.
        num_frequencies: Number of random frequencies per property.
        d_out: Output embedding width.
        sigma: Standard deviation of the random frequency matrix. Larger values encode
            finer distinctions but generalize less smoothly between property values.
        seed: Seed for the fixed random frequency buffer.
    """

    freqs: Tensor

    def __init__(
        self,
        num_properties: int,
        num_frequencies: int = 16,
        d_out: int = 256,
        *,
        sigma: float = 1.0,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if num_properties < 1 or num_frequencies < 1 or d_out < 1:
            raise ValueError(
                f"num_properties ({num_properties}), num_frequencies ({num_frequencies}) "
                f"and d_out ({d_out}) must all be >= 1"
            )
        self.num_properties = int(num_properties)
        self.num_frequencies = int(num_frequencies)
        self.d_out = int(d_out)

        gen = torch.Generator().manual_seed(int(seed))
        freqs = torch.randn(
            (self.num_properties, self.num_frequencies),
            generator=gen,
            dtype=torch.float32,
        ) * float(sigma)
        self.register_buffer("freqs", freqs, persistent=True)

        self.proj = nn.Linear(2 * self.num_frequencies, self.d_out, bias=True)
        self.property_id = nn.Parameter(torch.zeros(self.num_properties, self.d_out))

    def forward(self, values: Tensor) -> Tensor:
        """Embed property values ``[B, P]`` into per-property tokens ``[B, P, d_out]``.

        Args:
            values: Continuous property values of shape ``[B, P]``.

        Returns:
            Property embeddings of shape ``[B, P, d_out]``.
        """
        if values.ndim != 2 or int(values.shape[1]) != self.num_properties:
            raise ValueError(
                f"Expected values shape [B, {self.num_properties}], got {tuple(values.shape)}"
            )
        freqs = self.freqs.to(device=values.device, dtype=values.dtype)
        # [B, P, 1] * [1, P, F] -> [B, P, F]
        angles = 2.0 * math.pi * values.unsqueeze(-1) * freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        out: Tensor = self.proj(feats) + self.property_id.unsqueeze(0).to(dtype=feats.dtype)
        return out


class PropertyConditioner(nn.Module):
    """Compress continuous molecular property targets into decoder prefix tokens.

    Properties the caller does not wish to constrain are replaced by a learned ``unknown``
    embedding rather than an arbitrary sentinel value, so one trained decoder serves every
    subset of constraints and supports classifier-free guidance over individual
    properties.

    Args:
        num_properties: Number of conditioning properties ``P``.
        d_model: Decoder hidden width.
        num_tokens: Number of prefix tokens emitted (``K``).
        num_frequencies: Random Fourier frequencies per property.
        dropout: Dropout applied to the fused property representation.
        value_dropout_prob: Per-property probability of being forced to ``unknown`` during
            training. This is what makes partial-constraint inference work.
    """

    def __init__(
        self,
        num_properties: int,
        d_model: int,
        num_tokens: int = 4,
        *,
        num_frequencies: int = 16,
        dropout: float = 0.0,
        value_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if not 0.0 <= value_dropout_prob < 1.0:
            raise ValueError(f"value_dropout_prob must be in [0, 1), got {value_dropout_prob}")
        self.num_properties = int(num_properties)
        self.d_model = int(d_model)
        self.num_tokens = int(num_tokens)
        self.value_dropout_prob = float(value_dropout_prob)

        self.embed = FourierPropertyEmbedding(
            num_properties=self.num_properties,
            num_frequencies=num_frequencies,
            d_out=self.d_model,
        )
        self.unknown = nn.Parameter(torch.zeros(self.num_properties, self.d_model))
        nn.init.normal_(self.unknown, mean=0.0, std=0.02)

        self.norm = RMSNorm(self.d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.to_tokens = nn.Linear(self.d_model, self.num_tokens * self.d_model, bias=False)

    def forward(self, values: Tensor, property_mask: Tensor | None = None) -> Tensor:
        """Emit property prefix tokens ``[B, K, d_model]``.

        Args:
            values: Property values ``[B, P]``. Masked-out entries are ignored entirely.
            property_mask: Boolean mask ``[B, P]`` where ``True`` means "this property is
                specified". Defaults to all-specified. During training, specified
                properties are additionally dropped with probability
                ``value_dropout_prob``.

        Returns:
            Property prefix tokens of shape ``[B, K, d_model]``.
        """
        batch = int(values.shape[0])
        embedded = self.embed(values)  # [B, P, D]

        if property_mask is None:
            known = torch.ones((batch, self.num_properties), dtype=torch.bool, device=values.device)
        else:
            known = property_mask.to(device=values.device, dtype=torch.bool)
            if known.shape != (batch, self.num_properties):
                raise ValueError(
                    f"property_mask shape {tuple(known.shape)} != {(batch, self.num_properties)}"
                )

        if self.training and self.value_dropout_prob > 0.0:
            keep = torch.rand(known.shape, device=values.device) >= self.value_dropout_prob
            known = known & keep

        unknown = self.unknown.unsqueeze(0).to(dtype=embedded.dtype).expand(batch, -1, -1)
        fused = torch.where(known.unsqueeze(-1), embedded, unknown)  # [B, P, D]

        pooled = self.dropout(self.norm(fused.mean(dim=1)))  # [B, D]
        tokens: Tensor = self.to_tokens(pooled).view(batch, self.num_tokens, self.d_model)
        return tokens


# ---------------------------------------------------------------------------
# 3. Sparse Mixture-of-Experts feed-forward
# ---------------------------------------------------------------------------


class SparseMoEFeedForward(nn.Module):
    r"""Top-``k`` routed sparse Mixture-of-Experts SwiGLU feed-forward layer.

    Implements the Switch Transformer / ST-MoE auxiliary objectives:

    * **Load-balancing loss** :math:`E \sum_i f_i P_i`, where :math:`f_i` is the fraction
      of dispatch slots routed to expert :math:`i` (treated as a constant) and
      :math:`P_i` is the mean router probability for expert :math:`i`. The loss is
      minimized at ``1.0`` by a perfectly uniform router.
    * **Router z-loss** :math:`\mathbb{E}[\mathrm{logsumexp}(\ell)^2]`, which keeps router
      logits small and prevents the ``bfloat16`` round-off instability that otherwise
      collapses routing early in training.

    Dispatch uses a gather / expert-apply / ``index_add`` scatter loop over experts. This
    avoids custom kernels entirely, so it runs identically on CUDA, Apple Silicon MPS and
    CPU, at the cost of a Python loop of length ``num_experts``.

    .. warning::
       **Unvalidated for molecular generation; keep it off unless you are measuring it.**
       A survey of the literature found no evidence that sparse Mixture-of-Experts helps
       molecule generation at any scale, and no dense-vs-MoE comparison at matched FLOPs.
       The two chemistry papers that use the name are something else: Mol-MoE routes
       *densely* (every expert fires on every token) and is really model merging, while
       MoL-MoE is late fusion of frozen encoders for property prediction whose own paper
       concedes it "does not consistently outperform the best-performing uni-modal
       baseline".

       The deeper problem is that the premise runs backwards here. MoE buys parameters at
       roughly fixed FLOPs -- valuable only where parameters buy quality. For de novo
       design they demonstrably do not: goal-directed performance saturates beyond ~5M
       parameters, and for SMILES the fitted parameter exponent (0.0171) is ~25x smaller
       than the data exponent (0.4299). See :data:`SCALE_PRESETS`.

       It is provided so the question can be settled with a measurement rather than an
       assumption. Compare against a dense decoder at matched FLOPs, not matched
       parameters, and treat a win as surprising.

    Args:
        d_model: Hidden width.
        d_ff: Per-expert SwiGLU intermediate width.
        num_experts: Number of experts ``E``.
        top_k: Experts activated per token ``k``.
        dropout: Dropout inside each expert.
        router_z_loss_weight: Weight of the router z-loss term.
        normalize_top_k: Renormalize the top-``k`` gate weights to sum to ``1.0``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        router_z_loss_weight: float = 1e-3,
        normalize_top_k: bool = True,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"num_experts must be >= 2 for a MoE layer, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_z_loss_weight = float(router_z_loss_weight)
        self.normalize_top_k = bool(normalize_top_k)

        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(self.d_model, d_ff, dropout=dropout) for _ in range(self.num_experts)]
        )

        self.last_aux_loss: Tensor | None = None
        self.last_dispatch_counts: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Route ``x [B, L, d_model]`` through the top-``k`` experts per token.

        Side effects:
            Sets :attr:`last_aux_loss` (differentiable) and :attr:`last_dispatch_counts`
            (detached) for the owning decoder to collect.
        """
        in_shape = x.shape
        flat = x.reshape(-1, self.d_model)
        num_tokens = int(flat.shape[0])

        logits = self.router(flat.float())
        probs = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.normalize_top_k:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        counts = torch.zeros(self.num_experts, dtype=torch.float32, device=flat.device)

        for expert_id in range(self.num_experts):
            hit = topk_idx == expert_id  # [T, k]
            token_pos, slot_pos = hit.nonzero(as_tuple=True)
            counts[expert_id] = float(token_pos.numel())
            if token_pos.numel() == 0:
                continue
            gathered = flat.index_select(0, token_pos)
            expert_out = self.experts[expert_id](gathered)
            weight = topk_w[token_pos, slot_pos].unsqueeze(-1).to(dtype=expert_out.dtype)
            out = out.index_add(0, token_pos, expert_out * weight)

        # Switch-Transformer load balancing: f_i is a constant, P_i carries the gradient.
        denom = float(max(1, num_tokens * self.top_k))
        frac = (counts / denom).detach()
        mean_prob = probs.mean(dim=0)
        load_balance = float(self.num_experts) * torch.sum(frac * mean_prob)
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)

        self.last_aux_loss = load_balance + self.router_z_loss_weight * z_loss
        self.last_dispatch_counts = counts.detach()
        return out.reshape(in_shape)


# ---------------------------------------------------------------------------
# 4. Decoder block
# ---------------------------------------------------------------------------


class ChemLMBlock(nn.Module):
    """Pre-norm block: GQA self-attention, gated pocket cross-attention, SwiGLU or MoE FFN.

    Args:
        d_model: Hidden width.
        n_heads: Query attention heads.
        n_kv_heads: Key/value heads for grouped-query attention.
        d_ff: SwiGLU intermediate width (per expert under MoE).
        dropout: Dropout probability.
        use_cross_attn: Include a gated :class:`PocketCrossAttention` sublayer.
        d_pocket: Pocket memory width (defaults to ``d_model``).
        gated_cross_attn: Use a Flamingo ``tanh(gate)`` on the cross-attention residual.
        cross_attn_gate_init: Initial cross-attention gate value.
        qk_norm: Apply per-head QK-norm inside self-attention.
        max_seq_len: RoPE table length (must cover prefix tokens plus sequence).
        rope_base: RoPE base wavelength.
        use_moe: Replace the dense SwiGLU with :class:`SparseMoEFeedForward`.
        num_experts: Number of MoE experts.
        moe_top_k: Experts activated per token.
        router_z_loss_weight: Router z-loss weight.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        dropout: float = 0.0,
        *,
        use_cross_attn: bool = True,
        d_pocket: int | None = None,
        gated_cross_attn: bool = True,
        cross_attn_gate_init: float = 0.1,
        qk_norm: bool = True,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        router_z_loss_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_cross_attn = bool(use_cross_attn)
        self.d_pocket = self.d_model if d_pocket is None else int(d_pocket)
        self.use_moe = bool(use_moe)

        self.norm1 = RMSNorm(self.d_model)
        self.self_attn = GroupedQueryAttention(
            self.d_model,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            dropout=dropout,
            use_qk_norm=qk_norm,
            use_rope=True,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
            is_causal=True,
        )

        if self.use_cross_attn:
            self.cross_norm_q: RMSNorm | None = RMSNorm(self.d_model)
            self.cross_norm_kv: RMSNorm | None = RMSNorm(self.d_pocket)
            self.cross_attn: PocketCrossAttention | None = PocketCrossAttention(
                d_model=self.d_model,
                d_pocket=self.d_pocket,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                gated=gated_cross_attn,
                gate_init=cross_attn_gate_init,
            )
        else:
            self.cross_norm_q = None
            self.cross_norm_kv = None
            self.cross_attn = None

        self.norm2 = RMSNorm(self.d_model)
        self.ffn: nn.Module
        if self.use_moe:
            self.ffn = SparseMoEFeedForward(
                self.d_model,
                d_ff,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
                router_z_loss_weight=router_z_loss_weight,
            )
        else:
            self.ffn = SwiGLU(self.d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        *,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        kv_cache: AttentionKVCache | None = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ) -> tuple[Tensor, AttentionKVCache | None, Tensor | None]:
        """Run one block.

        Returns:
            ``(hidden, updated_cache_or_None, cross_attn_weights_or_None)``.
        """
        new_cache: AttentionKVCache | None = None
        if use_cache:
            sa_out, new_cache = self.self_attn(self.norm1(x), kv_cache=kv_cache, use_cache=True)
        else:
            sa_out = self.self_attn(self.norm1(x), kv_cache=kv_cache, use_cache=False)
        x = x + sa_out

        cross_weights: Tensor | None = None
        if self.use_cross_attn and memory is not None:
            assert (
                self.cross_norm_q is not None
                and self.cross_norm_kv is not None
                and self.cross_attn is not None
            )
            kv = self.cross_norm_kv(memory)
            valid: Tensor | None = None
            if memory_mask is not None:
                valid_mask = memory_mask.to(device=kv.device, dtype=torch.bool)
                kv = torch.where(valid_mask.unsqueeze(-1), kv, torch.zeros_like(kv))
                valid = valid_mask

            if return_attn:
                cross_out, cross_weights = self.cross_attn(
                    self.cross_norm_q(x), kv, pocket_mask=valid, return_attn=True
                )
            else:
                cross_out = self.cross_attn(
                    self.cross_norm_q(x), kv, pocket_mask=valid, return_attn=False
                )
            x = x + cross_out

        x = x + self.ffn(self.norm2(x))
        return x, new_cache, cross_weights


# ---------------------------------------------------------------------------
# 5. Decoder
# ---------------------------------------------------------------------------


class ChemLMDecoder(nn.Module):
    """Scalable pocket- and property-conditioned autoregressive chemical language decoder.

    Satisfies :class:`~geneuclidean.protocols.MolecularDecoderProtocol`, so it is a
    drop-in replacement for the legacy ``transformer`` decoder inside
    :class:`~geneuclidean.models.geneuclidean.GenEuclideanModel`.

    Args:
        vocab_size: Tokenizer vocabulary size, or a :class:`DecoderConfig` to read all
            architecture fields from.
        d_model: Hidden width.
        n_heads: Query attention heads.
        n_layers: Number of :class:`ChemLMBlock` layers.
        n_kv_heads: Key/value heads for grouped-query attention (defaults to ``n_heads``).
        d_ff: SwiGLU intermediate width (per expert under MoE).
        max_seq_len: Maximum ligand token sequence length.
        dropout: Dropout probability.
        pad_id: Padding token id (zeroed in the embedding table).
        use_cross_attn: Enable gated pocket cross-attention sublayers.
        d_pocket: Pocket memory width (defaults to ``d_model``).
        gated_cross_attn: Use Flamingo ``tanh(gate)`` cross-attention gating.
        cross_attn_gate_init: Initial cross-attention gate value.
        conditioning_mode: ``"cross_attn"``, ``"prefix"`` or ``"none"``.
        num_pocket_queries: Perceiver latent count for ``prefix`` conditioning.
        perceiver_layers: Perceiver resampler depth for ``prefix`` conditioning.
        qk_norm: Apply per-head QK-norm in self-attention.
        rope_base: RoPE base wavelength.
        use_moe: Use sparse Mixture-of-Experts feed-forward layers.
        num_experts: Number of MoE experts.
        moe_top_k: Experts activated per token.
        moe_every_n_layers: Place a MoE layer every ``n`` blocks (``1`` = all blocks).
        router_z_loss_weight: Router z-loss weight.
        num_properties: Number of continuous conditioning properties (``0`` disables).
        num_property_tokens: Property prefix tokens emitted when properties are supplied.
        property_dropout_prob: Per-property ``unknown`` dropout probability in training.
        tie_weights: Tie the LM head to the token embedding matrix.
    """

    def __init__(
        self,
        vocab_size: int | DecoderConfig = 96,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        *,
        use_cross_attn: bool = True,
        d_pocket: int | None = None,
        gated_cross_attn: bool = True,
        cross_attn_gate_init: float = 0.1,
        conditioning_mode: ConditioningMode = "cross_attn",
        num_pocket_queries: int = 16,
        perceiver_layers: int = 2,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        moe_every_n_layers: int = 1,
        router_z_loss_weight: float = 1e-3,
        num_properties: int = 0,
        num_property_tokens: int = 4,
        property_dropout_prob: float = 0.15,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(vocab_size, DecoderConfig):
            cfg = vocab_size
            resolved_vocab = int(cfg.vocab_size)
            d_model = int(cfg.d_model)
            n_heads = int(cfg.n_heads)
            n_layers = int(cfg.n_layers)
            d_ff = int(cfg.d_ff)
            max_seq_len = int(cfg.max_seq_len)
            dropout = float(cfg.dropout)
            use_cross_attn = bool(cfg.use_cross_attn)
            gated_cross_attn = bool(cfg.gated_cross_attn)
        else:
            resolved_vocab = int(vocab_size)

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if conditioning_mode not in ("cross_attn", "prefix", "none"):
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode!r}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.vocab_size = resolved_vocab
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_heads if n_kv_heads is None else n_kv_heads)
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.d_pocket = self.d_model if d_pocket is None else int(d_pocket)
        self.conditioning_mode: ConditioningMode = conditioning_mode
        self.num_pocket_queries = int(num_pocket_queries)
        self.num_properties = int(num_properties)
        self.num_property_tokens = int(num_property_tokens) if self.num_properties > 0 else 0
        self.is_moe = bool(use_moe)
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.emb_dropout = nn.Dropout(float(dropout))

        # Pocket soft-prefix bridge (prefix conditioning mode only).
        self.pocket_bridge: PerceiverLatentResampler | None = None
        if self.conditioning_mode == "prefix":
            self.pocket_bridge = PerceiverLatentResampler(
                self.d_pocket,
                self.d_model,
                num_latents=self.num_pocket_queries,
                num_layers=perceiver_layers,
                num_heads=self.n_heads,
                dropout=dropout,
            )

        # Continuous property conditioning.
        self.property_conditioner: PropertyConditioner | None = None
        if self.num_properties > 0:
            self.property_conditioner = PropertyConditioner(
                num_properties=self.num_properties,
                d_model=self.d_model,
                num_tokens=self.num_property_tokens,
                dropout=dropout,
                value_dropout_prob=property_dropout_prob,
            )

        block_cross_attn = use_cross_attn and self.conditioning_mode == "cross_attn"
        # RoPE must cover every prefix token plus the longest ligand sequence.
        prefix_budget = self.num_property_tokens + (
            self.num_pocket_queries if self.conditioning_mode == "prefix" else 0
        )
        rope_len = max(self.max_seq_len + prefix_budget, 512)

        self.blocks = nn.ModuleList(
            [
                ChemLMBlock(
                    self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_cross_attn=block_cross_attn,
                    d_pocket=self.d_pocket,
                    gated_cross_attn=gated_cross_attn,
                    cross_attn_gate_init=cross_attn_gate_init,
                    qk_norm=qk_norm,
                    max_seq_len=rope_len,
                    rope_base=rope_base,
                    use_moe=self.is_moe and (idx % max(1, int(moe_every_n_layers)) == 0),
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    router_z_loss_weight=router_z_loss_weight,
                )
                for idx in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self._init_weights()
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    # -- construction helpers ------------------------------------------------

    def _init_weights(self) -> None:
        """Apply depth-scaled residual initialization in ``float32``."""
        resid_std = 0.02 / math.sqrt(2.0 * float(self.n_layers))
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = resid_std if name.endswith(("o_proj", "w2", "w_down", "down_proj")) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        """Count parameters, de-duplicating tied weights.

        Args:
            trainable_only: Count only parameters with ``requires_grad=True``.

        Returns:
            Number of unique parameters.
        """
        seen: dict[int, int] = {}
        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue
            seen[id(param)] = param.numel()
        return int(sum(seen.values()))

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Toggle per-block gradient checkpointing to trade compute for memory.

        Off by default, because on a 36 GiB unified-memory Mac there is usually no memory
        problem to solve at these scales -- not because the saving is small. Measured on
        an M3 Pro at the ``small`` preset, batch 32, seq 96, it roughly **halves** GPU
        memory: ``torch.mps.driver_allocated_memory()`` falls 2672.9 MB -> 1360.9 MB
        (-49%). Turn it on for the ``large``/``xl`` presets, long sequences, or large
        batches on a 40 GiB A100.

        .. warning::
           Do not judge this with ``torch.mps.current_allocated_memory()``. MPS exposes
           no peak-memory counter, and the current-allocated figure sampled after a step
           has already had the activations freed, so it shows the saving as +0.3 MB --
           i.e. it cannot see the effect at all. Earlier revisions of this docstring
           quoted a 4.0x activation saving and a ~17% throughput cost with no recorded
           provenance; the throughput direction is not reproducible either, since
           ``results/tables/chemlm_decoder_benchmark.json`` has the checkpointed
           ``base`` config running *faster* (2056 vs 1640 tok/s), most likely an
           allocator-pressure artifact on unified memory. Re-measure before quoting a
           throughput number.
        """
        self.gradient_checkpointing = bool(enabled)

    def aux_loss(self) -> Tensor:
        """Sum Mixture-of-Experts auxiliary losses recorded during the last forward pass.

        Returns:
            A scalar tensor. Exactly ``0.0`` for dense decoders, or before any MoE forward
            pass has run.
        """
        device = self.token_emb.weight.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        for block in self.blocks:
            ffn = block.ffn
            if isinstance(ffn, SparseMoEFeedForward) and ffn.last_aux_loss is not None:
                total = total + ffn.last_aux_loss.to(device=device, dtype=torch.float32)
        return total

    # -- conditioning --------------------------------------------------------

    def _build_prefix(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        memory: Tensor | None,
        memory_mask: Tensor | None,
        properties: Tensor | None,
        property_mask: Tensor | None,
    ) -> Tensor | None:
        """Assemble ``[property_tokens, pocket_prefix]`` prefix embeddings, if any."""
        parts: list[Tensor] = []

        if self.property_conditioner is not None and properties is not None:
            parts.append(
                self.property_conditioner(
                    properties.to(device=device, dtype=torch.float32),
                    property_mask=property_mask,
                ).to(dtype=dtype)
            )

        if self.conditioning_mode == "prefix" and memory is not None:
            assert self.pocket_bridge is not None
            mask = None if memory_mask is None else memory_mask.to(device=device, dtype=torch.bool)
            parts.append(self.pocket_bridge(memory, mask=mask).to(dtype=dtype))

        if not parts:
            return None
        prefix = torch.cat(parts, dim=1)
        if int(prefix.shape[0]) != batch_size:
            raise ValueError(
                f"Prefix batch {int(prefix.shape[0])} does not match input batch {batch_size}"
            )
        return prefix

    def _run_blocks(
        self,
        x: Tensor,
        *,
        cross_memory: Tensor | None,
        cross_mask: Tensor | None,
        kv_caches: list[AttentionKVCache | None] | None,
        use_cache: bool,
        return_attn: bool,
    ) -> tuple[Tensor, list[AttentionKVCache | None], Tensor | None]:
        """Run the block stack, optionally with gradient checkpointing."""
        new_caches: list[AttentionKVCache | None] = []
        last_attn: Tensor | None = None
        do_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()

        def _make_runner(target: ChemLMBlock) -> Callable[[Tensor], Tensor]:
            """Bind one block into a single-tensor callable for ``checkpoint``."""

            def _run(hidden: Tensor) -> Tensor:
                out = target(
                    hidden,
                    memory=cross_memory,
                    memory_mask=cross_mask,
                    kv_cache=None,
                    use_cache=False,
                    return_attn=False,
                )
                result: Tensor = out[0]
                return result

            return _run

        for idx, raw_block in enumerate(self.blocks):
            block = cast(ChemLMBlock, raw_block)
            layer_cache = kv_caches[idx] if kv_caches is not None else None

            if do_checkpoint and not use_cache:
                # Checkpointing is incompatible with cache capture; it is a training-only
                # memory optimisation and generation never needs it.
                x = checkpoint(_make_runner(block), x, use_reentrant=False)
                new_caches.append(None)
                continue

            x, updated, attn = block(
                x,
                memory=cross_memory,
                memory_mask=cross_mask,
                kv_cache=layer_cache,
                use_cache=use_cache,
                return_attn=return_attn,
            )
            new_caches.append(updated)
            if attn is not None:
                last_attn = attn

        return x, new_caches, last_attn

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        return_attn: bool = False,
        *,
        properties: Tensor | None = None,
        property_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict next-token logits.

        Args:
            input_ids: Ligand token ids ``[B, L]`` (``torch.long``).
            memory: Optional pocket memory ``[B, M, d_pocket]``. ``None`` runs the
                unconditional Stage-1 chemical prior.
            memory_mask: Optional boolean mask ``[B, M]`` where ``True`` means valid.
            return_attn: Return final-layer pocket cross-attention weights.
            properties: Optional continuous property targets ``[B, num_properties]``.
            property_mask: Optional boolean mask ``[B, num_properties]``, ``True`` =
                "constrain this property".

        Returns:
            ``(logits [B, L, vocab_size], cross_attn_weights [B, H, L, M] | None)``.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape [B, L], got {tuple(input_ids.shape)}")

        batch_size = int(input_ids.shape[0])
        x = self.emb_dropout(self.token_emb(input_ids))

        prefix = self._build_prefix(
            batch_size,
            input_ids.device,
            x.dtype,
            memory=memory,
            memory_mask=memory_mask,
            properties=properties,
            property_mask=property_mask,
        )
        prefix_len = 0
        if prefix is not None:
            prefix_len = int(prefix.shape[1])
            x = torch.cat((prefix, x), dim=1)

        cross_memory = memory if self.conditioning_mode == "cross_attn" else None
        cross_mask = memory_mask if self.conditioning_mode == "cross_attn" else None

        x, _, last_attn = self._run_blocks(
            x,
            cross_memory=cross_memory,
            cross_mask=cross_mask,
            kv_caches=None,
            use_cache=False,
            return_attn=return_attn,
        )

        hidden = self.final_norm(x)
        if prefix_len > 0:
            hidden = hidden[:, prefix_len:, :]
            if last_attn is not None:
                last_attn = last_attn[:, :, prefix_len:, :]

        logits: Tensor = self.lm_head(hidden)
        return logits, (last_attn if return_attn else None)

    def forward_cached(
        self,
        input_ids: Tensor,
        *,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        kv_cache: list[AttentionKVCache | None] | None = None,
        properties: Tensor | None = None,
        property_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[AttentionKVCache | None]]:
        """Run one incremental decoding step with a KV cache.

        On the first call (``kv_cache is None``) any property and pocket-prefix tokens are
        prepended and enter the cache. Subsequent calls feed ligand tokens only, since the
        prefix already lives in the cache.

        Args:
            input_ids: New token ids ``[B, L_new]``.
            memory: Pocket memory ``[B, M, d_pocket]``.
            memory_mask: Boolean pocket mask ``[B, M]`` (``True`` = valid).
            kv_cache: Per-layer caches from the previous step, or ``None`` to prefill.
            properties: Optional property targets ``[B, num_properties]``.
            property_mask: Optional property mask ``[B, num_properties]``.

        Returns:
            ``(logits [B, L_new, vocab_size], updated_per_layer_caches)``.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape [B, L], got {tuple(input_ids.shape)}")

        batch_size = int(input_ids.shape[0])
        x = self.emb_dropout(self.token_emb(input_ids))

        prefix_len = 0
        if kv_cache is None:
            prefix = self._build_prefix(
                batch_size,
                input_ids.device,
                x.dtype,
                memory=memory,
                memory_mask=memory_mask,
                properties=properties,
                property_mask=property_mask,
            )
            if prefix is not None:
                prefix_len = int(prefix.shape[1])
                x = torch.cat((prefix, x), dim=1)

        cross_memory = memory if self.conditioning_mode == "cross_attn" else None
        cross_mask = memory_mask if self.conditioning_mode == "cross_attn" else None

        x, new_caches, _ = self._run_blocks(
            x,
            cross_memory=cross_memory,
            cross_mask=cross_mask,
            kv_caches=kv_cache,
            use_cache=True,
            return_attn=False,
        )

        hidden = self.final_norm(x)
        if prefix_len > 0:
            hidden = hidden[:, prefix_len:, :]
        logits: Tensor = self.lm_head(hidden)
        return logits, new_caches

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        properties: Tensor | None = None,
        property_mask: Tensor | None = None,
        max_new_tokens: int = 120,
        bos_id: int = 1,
        eos_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        cfg_scale: float = 1.0,
        suppress_token_ids: Sequence[int] | None = None,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Sample token sequences with KV-cached incremental decoding.

        Args:
            batch_size: Number of sequences to sample.
            memory: Pocket memory ``[B, M, d_pocket]``.
            memory_mask: Boolean pocket mask ``[B, M]`` (``True`` = valid).
            properties: Optional property targets ``[B, num_properties]``.
            property_mask: Optional property mask ``[B, num_properties]``.
            max_new_tokens: Maximum tokens to sample after ``<bos>``.
            bos_id: Beginning-of-sequence token id.
            eos_id: End-of-sequence token id; sampling stops when all rows emit it.
            temperature: Softmax temperature (``<= 0`` selects greedy decoding).
            top_k: Keep only the ``k`` highest-probability tokens (``0`` disables).
            top_p: Nucleus sampling threshold (``1.0`` disables).
            min_p: Minimum probability relative to the mode (``0.0`` disables).
            cfg_scale: Classifier-free guidance weight over pocket conditioning. ``1.0``
                disables guidance; requires ``memory`` when ``> 1.0``.
            suppress_token_ids: Token ids that may never be sampled. Defaults to
                ``(pad_id, bos_id)``, both of which are structurally invalid mid-sequence.
                Pass an explicit tuple to add the tokenizer's ``<unk>`` id (``3`` for
                :class:`~geneuclidean.data.tokenizer.MolecularTokenizer`), which matters
                for honest metrics: ``MolecularTokenizer.decode`` *silently drops* ``<unk>``
                and *truncates* at ``<pad>``, so an unsuppressed sample containing either
                can be silently repaired into a parseable string and inflate measured
                validity. ``eos_id`` is never suppressed -- it is the stop signal.
            seed: Optional seed for reproducible sampling.
            device: Device to sample on (defaults to the model's device).

        Returns:
            Token ids ``[B, <= max_new_tokens + 1]`` beginning with ``bos_id``.
        """
        dev = device if device is not None else self.token_emb.weight.device
        gen: torch.Generator | None = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))

        tokens = torch.full((batch_size, 1), int(bos_id), dtype=torch.long, device=dev)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=dev)

        use_cfg = cfg_scale != 1.0 and memory is not None
        cache: list[AttentionKVCache | None] | None = None
        uncond_cache: list[AttentionKVCache | None] | None = None

        # Forbid tokens that cannot legally occur mid-sequence. Without this the sampler
        # can emit <pad>/<bos>/<unk>, which the tokenizer's decoder then silently removes
        # or truncates at -- quietly turning malformed samples into parseable ones.
        banned = (self.pad_id, int(bos_id)) if suppress_token_ids is None else suppress_token_ids
        banned_ids = sorted({int(t) for t in banned if 0 <= int(t) < self.vocab_size})
        suppress_index: Tensor | None = (
            torch.tensor(banned_ids, dtype=torch.long, device=dev) if banned_ids else None
        )

        for step in range(int(max_new_tokens)):
            step_in = tokens if step == 0 else tokens[:, -1:]

            logits, cache = self.forward_cached(
                step_in,
                memory=memory,
                memory_mask=memory_mask,
                kv_cache=cache,
                properties=properties,
                property_mask=property_mask,
            )
            next_logits = logits[:, -1, :].float()

            if use_cfg:
                uncond_logits, uncond_cache = self.forward_cached(
                    step_in,
                    memory=None,
                    memory_mask=None,
                    kv_cache=uncond_cache,
                    properties=None,
                    property_mask=None,
                )
                uncond = uncond_logits[:, -1, :].float()
                next_logits = uncond + float(cfg_scale) * (next_logits - uncond)

            if suppress_index is not None:
                # After guidance, before truncation, so it binds greedy decoding too.
                next_logits = next_logits.index_fill(-1, suppress_index, float("-inf"))

            if temperature <= 0.0:
                next_tokens = next_logits.argmax(dim=-1)
            else:
                scaled = next_logits / float(temperature)
                probs = self._shape_probs(scaled, top_k=top_k, top_p=top_p, min_p=min_p)
                picked = torch.multinomial(probs.detach().to("cpu"), 1, generator=gen)
                next_tokens = picked.squeeze(-1).to(device=dev)

            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, int(eos_id)), next_tokens
            )
            tokens = torch.cat((tokens, next_tokens.unsqueeze(-1)), dim=1)
            finished = finished | (next_tokens == int(eos_id))
            if bool(finished.all().item()):
                break

        return tokens

    @staticmethod
    def _shape_probs(
        logits: Tensor,
        *,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
    ) -> Tensor:
        """Apply top-k, nucleus and min-p truncation, returning normalized probabilities."""
        work = logits
        if top_k and top_k > 0:
            k = min(int(top_k), int(work.shape[-1]))
            kth = work.topk(k, dim=-1).values[..., -1:]
            work = work.masked_fill(work < kth, float("-inf"))

        probs = torch.softmax(work, dim=-1)

        if min_p and min_p > 0.0:
            threshold = probs.max(dim=-1, keepdim=True).values * float(min_p)
            probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)

        if top_p and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            drop = cumulative - sorted_probs > float(top_p)
            sorted_probs = sorted_probs.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

        total = probs.sum(dim=-1, keepdim=True)
        safe = torch.where(total > 0, probs / total.clamp_min(1e-12), torch.ones_like(probs))
        return safe / safe.sum(dim=-1, keepdim=True)

    # -- checkpoint transfer -------------------------------------------------

    def load_pretrained_backbone(
        self,
        state_dict: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Transfer Stage-1 unconditional chemical prior weights into this decoder.

        Unwraps ``model_state_dict``, ``module.`` and ``decoder.`` wrappers, then loads
        every shape-compatible tensor. Newly introduced cross-attention layers, gates and
        property conditioners are left at their initialization and reported as missing.

        .. important::
           Construct the target decoder with ``cross_attn_gate_init=0.0`` when using this.
           The default is ``0.1``, so ``tanh(0.1) ~ 0.0997`` and the freshly initialized
           cross-attention perturbs the transferred prior from the first step. Flamingo's
           reason for a **zero**-initialized tanh gate is precisely that it makes the
           inserted cross-attention an exact no-op at initialization, leaving the
           pretrained language model intact while the gate learns to open (reported worth
           +4.2 and to fix training instability, arXiv:2204.14198 Tab. 3). Stage-1 ->
           Stage-2 transfer is the case where that matters most.

        Args:
            state_dict: A checkpoint mapping, possibly wrapped.

        Returns:
            ``(missing_keys, unexpected_keys)``.
        """
        raw: Mapping[str, Any] = state_dict
        if "model_state_dict" in raw and isinstance(raw["model_state_dict"], Mapping):
            raw = raw["model_state_dict"]

        target = self.state_dict()
        filtered: dict[str, Tensor] = {}
        unexpected: list[str] = []

        for key, value in raw.items():
            if not isinstance(value, Tensor):
                continue
            clean = key
            for prefix in ("module.", "decoder."):
                while clean.startswith(prefix):
                    clean = clean[len(prefix) :]
            if clean in target and target[clean].shape == value.shape:
                filtered[clean] = value
            else:
                unexpected.append(key)

        missing = [k for k in target if k not in filtered]
        self.load_state_dict(filtered, strict=False)
        # Weight tying must be re-asserted after a partial state_dict load.
        if self.lm_head.weight.shape == self.token_emb.weight.shape:
            self.lm_head.weight = self.token_emb.weight
        return missing, unexpected


# ---------------------------------------------------------------------------
# 6. Registry factories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PocketRelianceReport:
    """Whether a decoder actually uses *which* pocket it was given.

    Attributes:
        nll_true: Mean NLL in nats/token conditioned on the correct pocket.
        nll_permuted: Mean NLL with pockets cyclically permuted across the batch, so
            every ligand is paired with some *other* real pocket.
        nll_null: Mean NLL with no pocket at all (the unconditional prior).
        delta_permuted: ``nll_permuted - nll_true``. The headline number.
        delta_null: ``nll_null - nll_true``.
        gate_magnitudes: ``|tanh(gate)|`` per cross-attention block.
    """

    nll_true: float
    nll_permuted: float
    nll_null: float
    gate_magnitudes: list[float] = field(default_factory=list)

    @property
    def delta_permuted(self) -> float:
        """Nats/token lost by swapping in the wrong pocket. Near zero means trouble."""
        return self.nll_permuted - self.nll_true

    @property
    def delta_null(self) -> float:
        """Nats/token lost by removing the pocket entirely."""
        return self.nll_null - self.nll_true

    @property
    def uses_specific_pocket(self) -> bool:
        """``True`` when the wrong pocket costs a non-trivial amount of likelihood."""
        return self.delta_permuted > 0.01

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary including the derived deltas."""
        return {
            "nll_true": round(self.nll_true, 6),
            "nll_permuted": round(self.nll_permuted, 6),
            "nll_null": round(self.nll_null, 6),
            "delta_permuted": round(self.delta_permuted, 6),
            "delta_null": round(self.delta_null, 6),
            "uses_specific_pocket": self.uses_specific_pocket,
            "mean_gate_magnitude": (
                round(sum(self.gate_magnitudes) / len(self.gate_magnitudes), 6)
                if self.gate_magnitudes
                else None
            ),
            "gate_magnitudes": [round(g, 6) for g in self.gate_magnitudes],
        }


def cross_attention_gate_magnitudes(decoder: nn.Module) -> list[float]:
    """Read ``|tanh(gate)|`` from every gated cross-attention block.

    The Flamingo-style gate is a literal scalar readout of "is this layer using the
    pocket?". Logging it during training gives an early warning that costs nothing: if
    the gates decay toward zero, the decoder is on its way to ignoring the conditioning.
    Prefix-style conditioning has no equivalent readout.

    Args:
        decoder: Any module containing gated :class:`PocketCrossAttention` sublayers.

    Returns:
        One magnitude per gated cross-attention block, in module order.
    """
    magnitudes: list[float] = []
    for module in decoder.modules():
        if isinstance(module, PocketCrossAttention) and module.gate is not None:
            magnitudes.append(float(torch.tanh(module.gate.detach()).abs()))
    return magnitudes


@torch.no_grad()
def pocket_reliance_report(
    decoder: nn.Module,
    input_ids: Tensor,
    memory: Tensor,
    memory_mask: Tensor | None = None,
    *,
    pad_id: int = 0,
) -> PocketRelianceReport:
    """Measure whether a decoder conditions on *which* pocket it was given.

    This is the cheapest guard against the dominant failure mode in structure-based
    design, and the reason it matters is that docking scores actively hide it. On
    CrossDocked2020, TargetDiff beats the real crystal ligand on raw Glide score while
    capturing only 47% of its pocket *specificity*; Shape2Mol sits at the
    zero-conditioning floor, and a random ZINC draw beats the crystal ligand 22.4% of the
    time (Delta Score, arXiv:2311.12035). A decoder that has learned to emit
    generically-good druglike molecules while ignoring the pocket still posts a
    respectable Vina number -- TacoGFN puts the entire value of pocket conditioning at
    0.31-0.41 kcal/mol.

    So instead of asking "is the affinity good?", this asks "does the correct pocket
    raise the likelihood of the correct ligand, relative to a different real pocket?".
    Pockets are cyclically permuted across the batch, which keeps the distribution of
    pocket inputs identical and changes only the *pairing*.

    Interpretation:
        * ``delta_permuted`` near zero: the decoder is ignoring the pocket. Nothing
          downstream will fix this.
        * ``delta_permuted`` near ``delta_null``: it knows a pocket is present but not
          which one.
        * ``delta_permuted`` clearly positive: genuine pocket specificity.

    Requires a batch size of at least 2, since a permutation of one element is identity.

    Args:
        decoder: Any decoder honouring ``(input_ids, memory, memory_mask)``, so this
            works for the legacy ``transformer`` decoder too, not just :class:`ChemLMDecoder`.
        input_ids: Ground-truth ligand token ids ``[B, L]``.
        memory: Pocket memory ``[B, M, D]``, correctly paired with ``input_ids``.
        memory_mask: Optional boolean pocket mask ``[B, M]`` (``True`` = valid).
        pad_id: Padding token id, excluded from the NLL.

    Returns:
        A :class:`PocketRelianceReport`.

    Raises:
        ValueError: If the batch holds fewer than 2 examples.
    """
    batch_size = int(input_ids.shape[0])
    if batch_size < 2:
        raise ValueError(
            f"pocket_reliance_report needs batch_size >= 2 to permute pockets, got {batch_size}"
        )

    was_training = decoder.training
    decoder.eval()

    def _nll(mem: Tensor | None, mask: Tensor | None) -> float:
        out = decoder(input_ids, memory=mem, memory_mask=mask)
        logits = out[0] if isinstance(out, tuple) else out
        shift_logits = logits[:, :-1, :].reshape(-1, int(logits.shape[-1])).float()
        shift_labels = input_ids[:, 1:].reshape(-1)
        return float(F.cross_entropy(shift_logits, shift_labels, ignore_index=int(pad_id)).detach())

    # Roll by one along the batch so every ligand sees a different real pocket.
    permuted_memory = torch.roll(memory, shifts=1, dims=0)
    permuted_mask = None if memory_mask is None else torch.roll(memory_mask, shifts=1, dims=0)

    report = PocketRelianceReport(
        nll_true=_nll(memory, memory_mask),
        nll_permuted=_nll(permuted_memory, permuted_mask),
        nll_null=_nll(None, None),
        gate_magnitudes=cross_attention_gate_magnitudes(decoder),
    )

    if was_training:
        decoder.train()
    return report


def build_chemlm_decoder(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
    **kwargs: Any,
) -> ChemLMDecoder:
    """Build a dense ``chemlm`` decoder from a :class:`DecoderConfig`.

    Args:
        cfg: Decoder configuration.
        cross_attn_gate_init: Initial Flamingo cross-attention gate value.
        **kwargs: Extra :class:`ChemLMDecoder` keyword arguments (e.g. ``num_properties``,
            ``conditioning_mode``, ``qk_norm``, ``n_kv_heads``).

    Returns:
        A configured :class:`ChemLMDecoder`.
    """
    params: dict[str, Any] = {
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "d_ff": cfg.d_ff,
        "max_seq_len": cfg.max_seq_len,
        "dropout": cfg.dropout,
        "use_cross_attn": cfg.use_cross_attn,
        "gated_cross_attn": cfg.gated_cross_attn,
        "cross_attn_gate_init": cross_attn_gate_init,
    }
    # Forward grouped-query attention from the config. Before `DecoderConfig` carried
    # this field there was no way to request GQA from YAML or the CLI, so a config
    # mirroring the `small` preset built 33.76M parameters with full multi-head attention
    # instead of the 27.47M the preset intends -- 22.9% more, and without the K/V saving
    # that is the whole point of the preset. Every benchmark figure was produced by
    # constructing `ChemLMDecoder` directly with `n_kv_heads`, so the config path and the
    # measured path disagreed.
    if getattr(cfg, "n_kv_heads", None) is not None:
        params["n_kv_heads"] = cfg.n_kv_heads
    params.update(kwargs)
    return ChemLMDecoder(**params)


def build_chemlm_moe_decoder(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
    **kwargs: Any,
) -> ChemLMDecoder:
    """Build a sparse Mixture-of-Experts ``chemlm_moe`` decoder.

    Args:
        cfg: Decoder configuration.
        cross_attn_gate_init: Initial Flamingo cross-attention gate value.
        **kwargs: Extra :class:`ChemLMDecoder` keyword arguments (e.g. ``num_experts``,
            ``moe_top_k``, ``moe_every_n_layers``).

    Returns:
        A configured Mixture-of-Experts :class:`ChemLMDecoder`.
    """
    params: dict[str, Any] = {"use_moe": True, "num_experts": 8, "moe_top_k": 2}
    params.update(kwargs)
    return build_chemlm_decoder(cfg, cross_attn_gate_init=cross_attn_gate_init, **params)
