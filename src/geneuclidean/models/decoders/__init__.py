"""Registry-backed factory and exports for pluggable autoregressive molecular decoders."""

from __future__ import annotations

import torch.nn as nn

from geneuclidean.config import DecoderConfig
from geneuclidean.models.decoders.lstm import MaskedBahdanauAttention, PocketLSTMDecoder
from geneuclidean.models.decoders.transformer import (
    CausalSelfAttention,
    PocketCrossAttention,
    PocketTransformerBlock,
    PocketTransformerDecoder,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rope,
)
from geneuclidean.registry import DECODER_REGISTRY, list_decoders, register_decoder


def _build_transformer(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
) -> PocketTransformerDecoder:
    return PocketTransformerDecoder(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        use_cross_attn=cfg.use_cross_attn,
        gated_cross_attn=cfg.gated_cross_attn,
        cross_attn_gate_init=cross_attn_gate_init,
    )


def _build_lstm(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
) -> PocketLSTMDecoder:
    del cross_attn_gate_init  # LSTM cross-attention does not use Flamingo tanh gating.
    return PocketLSTMDecoder(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        use_cross_attn=cfg.use_cross_attn,
    )


DECODER_REGISTRY.register(
    "transformer",
    _build_transformer,
    override=True,
    metadata={"architecture": "RoPE + SwiGLU + Flamingo Gated Cross-Attention", "default": True},
)
DECODER_REGISTRY.register(
    "lstm",
    _build_lstm,
    override=True,
    metadata={"architecture": "Multi-layer LSTM + Masked Bahdanau Attention"},
)


def build_decoder(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
) -> nn.Module:
    """Instantiate a built-in or custom registered autoregressive decoder."""
    if cfg.arch not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder architecture: {cfg.arch!r}. Registered: {list_decoders()}"
        )
    factory = DECODER_REGISTRY.get(cfg.arch)
    module: nn.Module = factory(cfg, cross_attn_gate_init=cross_attn_gate_init)
    return module


__all__ = [
    "DECODER_REGISTRY",
    "CausalSelfAttention",
    "MaskedBahdanauAttention",
    "PocketCrossAttention",
    "PocketLSTMDecoder",
    "PocketTransformerBlock",
    "PocketTransformerDecoder",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "apply_rope",
    "build_decoder",
    "list_decoders",
    "register_decoder",
]
