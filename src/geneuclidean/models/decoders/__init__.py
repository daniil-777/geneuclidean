"""Registry and exports for autoregressive molecular decoders."""

from __future__ import annotations

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


def build_decoder(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
) -> PocketTransformerDecoder | PocketLSTMDecoder:
    """Instantiate a configured autoregressive decoder from ``DecoderConfig``."""
    if cfg.arch == "transformer":
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
    if cfg.arch == "lstm":
        return PocketLSTMDecoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            use_cross_attn=cfg.use_cross_attn,
        )
    raise ValueError(f"Unknown decoder architecture: {cfg.arch!r}")


__all__ = [
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
]
