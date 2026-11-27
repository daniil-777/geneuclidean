"""Registry-backed factory and exports for pluggable 1D and 3D molecular decoders."""

from __future__ import annotations

import importlib
from typing import Any

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


def _lazy_3d_factory(module_name: str, func_name: str) -> Any:
    """Wrap a 3D-decoder factory so its module is imported only when one is built.

    The 3D decoder modules (``autoregressive3d``, ``flow3d``, ``mol3d_recon``) are heavy
    and are imported on demand rather than at package-import time. Deferring them keeps
    ``import geneuclidean.models`` cheap and, more importantly, keeps the package
    importable when an optional 3D module is absent -- an eager import here makes the
    whole decoder registry, including the 1D SMILES decoders, unimportable.
    """

    def _build(cfg: DecoderConfig, cross_attn_gate_init: float = 0.1, **kwargs: Any) -> nn.Module:
        module = importlib.import_module(f"geneuclidean.models.decoders.{module_name}")
        factory = getattr(module, func_name)
        built: nn.Module = factory(cfg, cross_attn_gate_init=cross_attn_gate_init, **kwargs)
        return built

    _build.__name__ = f"_build_{func_name}"
    _build.__qualname__ = _build.__name__
    _build.__doc__ = f"Build via :func:`{module_name}.{func_name}`, imported lazily."
    return _build


DECODER_REGISTRY.register(
    "geom_transformer",
    _lazy_3d_factory("autoregressive3d", "build_geom_transformer_decoder"),
    aliases=("geometry_transformer",),
    override=True,
    metadata={"architecture": "QKNorm + GQA + RoPE + SwiGLU + 3D Pocket RBF Spatial Bias"},
)
DECODER_REGISTRY.register(
    "flow3d",
    _lazy_3d_factory("flow3d", "build_flow3d_decoder"),
    aliases=("pocket_flow3d", "flow_3d"),
    override=True,
    metadata={"architecture": "SE(3)-Equivariant Joint Continuous-Discrete 3D Flow Matching"},
)
DECODER_REGISTRY.register(
    "ar3d",
    _lazy_3d_factory("autoregressive3d", "build_ar3d_decoder"),
    aliases=("pocket_ar3d", "ar_3d"),
    override=True,
    metadata={"architecture": "Focal-Atom SE(3) Local-Frame Autoregressive 3D Decoder"},
)
CHEMLM_SCALE_NAMES: tuple[str, ...] = ("base", "large", "small", "tiny", "xl")


def _build_chemlm(
    cfg: DecoderConfig, cross_attn_gate_init: float = 0.1, **kwargs: Any
) -> nn.Module:
    """Build a dense ``chemlm`` decoder, importing it lazily.

    ``chemlm`` depends on :mod:`geneuclidean.models.layers`, which itself imports leaf
    modules under :mod:`geneuclidean.models.encoders`. Importing it eagerly here would
    close an import cycle (``layers -> encoders -> decoders.transformer -> decoders``),
    so the import is deferred to first use. This also keeps the cold-import budget low.
    """
    from geneuclidean.models.decoders.chemlm import build_chemlm_decoder

    return build_chemlm_decoder(cfg, cross_attn_gate_init=cross_attn_gate_init, **kwargs)


def _build_chemlm_moe(
    cfg: DecoderConfig, cross_attn_gate_init: float = 0.1, **kwargs: Any
) -> nn.Module:
    """Build a sparse Mixture-of-Experts ``chemlm_moe`` decoder, importing it lazily."""
    from geneuclidean.models.decoders.chemlm import build_chemlm_moe_decoder

    return build_chemlm_moe_decoder(cfg, cross_attn_gate_init=cross_attn_gate_init, **kwargs)


DECODER_REGISTRY.register(
    "chemlm",
    _build_chemlm,
    aliases=("chem_lm", "pocket_chemlm"),
    override=True,
    metadata={
        "architecture": (
            "GQA + QKNorm + RoPE + SwiGLU + Flamingo Gated Cross-Attention, "
            "KV-cached, property-conditioned"
        ),
        "scales": list(CHEMLM_SCALE_NAMES),
        "recommended": True,
    },
)
DECODER_REGISTRY.register(
    "chemlm_moe",
    _build_chemlm_moe,
    aliases=("chem_lm_moe", "moe_transformer"),
    override=True,
    metadata={
        "architecture": (
            "Sparse Top-k Mixture-of-Experts SwiGLU + GQA + QKNorm + RoPE, "
            "Switch load-balancing + router z-loss"
        ),
        "scales": list(CHEMLM_SCALE_NAMES),
    },
)

# Symbols re-exported from submodules that are imported on demand, mapped to the module
# that defines each one. `chemlm` is deferred to break an import cycle (see
# `_build_chemlm`); the 3D decoders are deferred so that a missing optional 3D module
# cannot make this package -- and therefore every 1D SMILES decoder -- unimportable.
_LAZY_EXPORTS: dict[str, str] = {
    name: "chemlm"
    for name in (
        "SCALE_PRESETS",
        "ChemLMBlock",
        "ChemLMDecoder",
        "DecoderScale",
        "FourierPropertyEmbedding",
        "PropertyConditioner",
        "SparseMoEFeedForward",
        "build_chemlm_decoder",
        "build_chemlm_moe_decoder",
        "resolve_scale",
        "scaled_decoder_config",
    )
}
_LAZY_EXPORTS.update(
    dict.fromkeys(
        (
            "AR3DForwardOutput",
            "AR3DLossOutput",
            "GQACausalSelfAttention",
            "GeometryBiasedCrossAttention",
            "GeometryBiasedTransformerBlock",
            "GeometryBiasedTransformerDecoder",
            "PocketFrameAR3DDecoder",
            "PocketSpatialRBFBias",
            "build_ar3d_decoder",
            "build_equivariant_local_frames",
            "build_geom_transformer_decoder",
        ),
        "autoregressive3d",
    )
)
_LAZY_EXPORTS.update(
    dict.fromkeys(
        (
            "Flow3DLossOutput",
            "Flow3DSampleOutput",
            "PocketFlow3DDecoder",
            "SE3PocketLigandFlowLayer",
            "build_flow3d_decoder",
        ),
        "flow3d",
    )
)
_LAZY_EXPORTS.update(
    dict.fromkeys(
        (
            "ATOMIC_NUM_TO_TYPE_IDX",
            "ATOM_TYPE_ATOMIC_NUMS",
            "ATOM_TYPE_SYMBOLS",
            "Generated3DMolecule",
            "NUM_ATOM_TYPES",
            "NUM_BOND_TYPES",
            "SYMBOL_TO_ATOMIC_NUM",
            "atom_types_to_class_indices",
            "reconstruct_mol_from_3d_graph",
        ),
        "mol3d_recon",
    )
)

# Retained for backwards compatibility with callers that introspected this set.
_LAZY_CHEMLM_EXPORTS: frozenset[str] = frozenset(
    name for name, module in _LAZY_EXPORTS.items() if module == "chemlm"
)


def __getattr__(name: str) -> Any:
    """Resolve a lazily re-exported submodule symbol on first attribute access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(f"geneuclidean.models.decoders.{module_name}")
        value = getattr(module, name)
        globals()[name] = value  # Cache so subsequent lookups skip __getattr__.
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily re-exported submodule symbols in ``dir()`` output."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def build_decoder(
    cfg: DecoderConfig,
    cross_attn_gate_init: float = 0.1,
) -> nn.Module:
    """Instantiate a built-in or custom registered 1D or 3D molecular decoder."""
    if cfg.arch not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder architecture: {cfg.arch!r}. Registered: {list_decoders()}"
        )
    factory = DECODER_REGISTRY.get(cfg.arch)
    module: nn.Module = factory(cfg, cross_attn_gate_init=cross_attn_gate_init)
    return module


__all__ = [
    "ATOMIC_NUM_TO_TYPE_IDX",
    "ATOM_TYPE_ATOMIC_NUMS",
    "ATOM_TYPE_SYMBOLS",
    "AR3DForwardOutput",
    "AR3DLossOutput",
    "CausalSelfAttention",
    "ChemLMBlock",
    "ChemLMDecoder",
    "DECODER_REGISTRY",
    "DecoderScale",
    "FourierPropertyEmbedding",
    "PropertyConditioner",
    "SCALE_PRESETS",
    "SparseMoEFeedForward",
    "build_chemlm_decoder",
    "build_chemlm_moe_decoder",
    "resolve_scale",
    "scaled_decoder_config",
    "Flow3DLossOutput",
    "Flow3DSampleOutput",
    "GQACausalSelfAttention",
    "Generated3DMolecule",
    "GeometryBiasedCrossAttention",
    "GeometryBiasedTransformerBlock",
    "GeometryBiasedTransformerDecoder",
    "MaskedBahdanauAttention",
    "NUM_ATOM_TYPES",
    "NUM_BOND_TYPES",
    "PocketCrossAttention",
    "PocketFlow3DDecoder",
    "PocketFrameAR3DDecoder",
    "PocketLSTMDecoder",
    "PocketSpatialRBFBias",
    "PocketTransformerBlock",
    "PocketTransformerDecoder",
    "RMSNorm",
    "RotaryEmbedding",
    "SE3PocketLigandFlowLayer",
    "SYMBOL_TO_ATOMIC_NUM",
    "SwiGLU",
    "apply_rope",
    "atom_types_to_class_indices",
    "build_ar3d_decoder",
    "build_decoder",
    "build_equivariant_local_frames",
    "build_flow3d_decoder",
    "build_geom_transformer_decoder",
    "list_decoders",
    "reconstruct_mol_from_3d_graph",
    "register_decoder",
]
