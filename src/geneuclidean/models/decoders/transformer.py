"""Modern Pre-LN RoPE + SwiGLU + RMSNorm Transformer molecular decoder.

Supports both Stage-1 unconditional chemical prior pretraining (``memory=None``) and
Stage-2 3D-pocket-conditioned generation via Flamingo-style gated cross-attention with
exact zero-weight masking over padded pocket atoms.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.config import DecoderConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) in ``float32``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        normed = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE, Su et al. 2021) for causal self-attention."""

    inv_freq: Tensor
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 256, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = int(head_dim)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._compute_tables(max_seq_len, inv_freq)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def _compute_tables(seq_len: int, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [L, head_dim // 2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, head_dim]
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables of shape ``[1, 1, seq_len, head_dim]``."""
        if seq_len <= self.cos_cached.shape[2]:
            return (
                self.cos_cached[:, :, :seq_len, :].to(device=device),
                self.sin_cached[:, :, :seq_len, :].to(device=device),
            )
        cos, sin = self._compute_tables(seq_len, self.inv_freq.to(device=device))
        return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to query and key tensors ``[B, H, L, Dh]``."""
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward network (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.w1(x)) * self.w3(x)
        out: Tensor = self.w2(self.dropout(hidden))
        return out


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        merged = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out: Tensor = self.resid_dropout(self.o_proj(merged))
        return out


class PocketCrossAttention(nn.Module):
    """Multi-head cross-attention over 3D pocket memory with exact zero-leak masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Attend from decoder tokens ``x[B, L, D]`` into ``memory[B, M, D]``.

        Padded positions (``memory_mask == False``) are masked to ``-inf`` prior to
        softmax and explicitly zeroed afterward, guaranteeing exact ``0.0`` weight.
        """
        batch_size, seq_len, _ = x.shape
        mem_len = memory.shape[1]

        if memory_mask is not None:
            mask_bool = memory_mask.to(torch.bool)
            safe_memory = memory * mask_bool.unsqueeze(-1).to(memory.dtype)
        else:
            mask_bool = None
            safe_memory = memory

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(safe_memory)
            .view(batch_size, mem_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(safe_memory)
            .view(batch_size, mem_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, M]

        if mask_bool is not None:
            key_mask = mask_bool[:, None, None, :]  # [B, 1, 1, M]
            all_masked = ~mask_bool.any(dim=-1, keepdim=True)  # [B, 1]
            all_masked_4d = all_masked[:, None, None, :]  # [B, 1, 1, 1]
            scores = scores.masked_fill(~key_mask, float("-inf"))
            scores = scores.masked_fill(all_masked_4d, 0.0)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = attn_weights.masked_fill(~key_mask, 0.0)
        else:
            attn_weights = torch.softmax(scores, dim=-1)

        attn_probs = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_probs, v)  # [B, H, L, Dh]
        merged = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out: Tensor = self.resid_dropout(self.o_proj(merged))
        return out, attn_weights


class PocketTransformerBlock(nn.Module):
    """Pre-LN Transformer block with RoPE self-attn, optional gated cross-attn, and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        gated_cross_attn: bool = True,
        cross_attn_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_cross_attn = bool(use_cross_attn)
        self.gated_cross_attn = bool(gated_cross_attn)

        self.norm1 = RMSNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        if self.use_cross_attn:
            self.cross_norm_q: RMSNorm | None = RMSNorm(d_model)
            self.cross_norm_kv: RMSNorm | None = RMSNorm(d_model)
            self.cross_attn: PocketCrossAttention | None = PocketCrossAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout
            )
            if self.gated_cross_attn:
                self.cross_gate: nn.Parameter | None = nn.Parameter(
                    torch.tensor(float(cross_attn_gate_init), dtype=torch.float32)
                )
            else:
                self.cross_gate = None
        else:
            self.cross_norm_q = None
            self.cross_norm_kv = None
            self.cross_attn = None
            self.cross_gate = None

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        x = x + self.self_attn(self.norm1(x), cos, sin)

        attn_weights: Tensor | None = None
        if self.use_cross_attn and memory is not None:
            assert (
                self.cross_norm_q is not None
                and self.cross_norm_kv is not None
                and self.cross_attn is not None
            )
            q_norm = self.cross_norm_q(x)
            kv_norm = self.cross_norm_kv(memory)
            cross_out, attn_weights = self.cross_attn(q_norm, kv_norm, memory_mask=memory_mask)
            if self.cross_gate is not None:
                x = x + torch.tanh(self.cross_gate) * cross_out
            else:
                x = x + cross_out

        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class PocketTransformerDecoder(nn.Module):
    """Autoregressive Pre-LN RoPE/SwiGLU Transformer decoder with gated pocket cross-attention.

    Calling ``forward(input_ids, memory=None)`` cleanly executes Stage-1 unconditional
    chemical language modeling; passing 3D pocket memory executes Stage-2 conditional
    generation while preserving Stage-1 fluency via gated cross-attention.
    """

    def __init__(
        self,
        vocab_size: int | DecoderConfig = 96,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 688,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        gated_cross_attn: bool = True,
        cross_attn_gate_init: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(vocab_size, DecoderConfig):
            cfg = vocab_size
            vocab_size_int = int(cfg.vocab_size)
            d_model = int(cfg.d_model)
            n_heads = int(cfg.n_heads)
            n_layers = int(cfg.n_layers)
            d_ff = int(cfg.d_ff)
            max_seq_len = int(cfg.max_seq_len)
            dropout = float(cfg.dropout)
            use_cross_attn = bool(cfg.use_cross_attn)
            gated_cross_attn = bool(cfg.gated_cross_attn)
        else:
            vocab_size_int = int(vocab_size)

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.vocab_size = vocab_size_int
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.max_seq_len = int(max_seq_len)
        self.use_cross_attn = bool(use_cross_attn)
        self.gated_cross_attn = bool(gated_cross_attn)
        self.pad_id = int(pad_id)

        self.token_emb = nn.Embedding(vocab_size_int, d_model, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(head_dim=d_model // n_heads, max_seq_len=max(max_seq_len, 256))

        self.blocks = nn.ModuleList(
            [
                PocketTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_cross_attn=use_cross_attn,
                    gated_cross_attn=gated_cross_attn,
                    cross_attn_gate_init=cross_attn_gate_init,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size_int, bias=False)

        self._init_weights()
        # Tie output LM head weights to token embedding matrix.
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self) -> None:
        residual_std = 0.02 / math.sqrt(2.0 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if name.endswith(("o_proj", "w2")) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Run autoregressive forward pass.

        Args:
            input_ids: Token indices of shape ``[B, L]`` (``torch.long``).
            memory: Optional 3D pocket (+ CVAE latent) memory of shape ``[B, M, D]``.
            memory_mask: Optional boolean mask of shape ``[B, M]`` (``True`` = valid).
            return_attn: If ``True`` and cross-attention runs, return final-layer
                cross-attention weights of shape ``[B, H, L, M]``.

        Returns:
            ``(logits[B, L, V], attn_weights[B, H, L, M] | None)``.
        """
        seq_len = input_ids.shape[1]
        x = self.emb_dropout(self.token_emb(input_ids))
        cos, sin = self.rope(seq_len, device=input_ids.device)

        last_attn: Tensor | None = None
        for block in self.blocks:
            x, block_attn = block(x, cos, sin, memory=memory, memory_mask=memory_mask)
            if block_attn is not None:
                last_attn = block_attn

        logits: Tensor = self.lm_head(self.final_norm(x))
        return logits, (last_attn if return_attn else None)

    def load_pretrained_backbone(
        self,
        state_dict: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Transfer Stage-1 pretrained chemical LM weights into a Stage-2 decoder.

        Cleanly unwraps ``model_state_dict``, ``module.``, and ``decoder.`` prefixes and
        loads all shape-compatible weights while leaving newly introduced cross-attention
        layers and gates intact.
        """
        raw: Mapping[str, Any] = state_dict
        if "model_state_dict" in raw and isinstance(raw["model_state_dict"], Mapping):
            raw = raw["model_state_dict"]

        target_state = self.state_dict()
        filtered: dict[str, Tensor] = {}
        unexpected: list[str] = []

        for key, value in raw.items():
            if not isinstance(value, Tensor):
                continue
            clean_key = key
            for prefix in ("module.", "decoder."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            if clean_key in target_state and target_state[clean_key].shape == value.shape:
                filtered[clean_key] = value
            else:
                unexpected.append(key)

        missing = [k for k in target_state if k not in filtered]
        self.load_state_dict(filtered, strict=False)
        # Re-affirm weight tying after state_dict load.
        self.lm_head.weight = self.token_emb.weight
        return missing, unexpected


__all__ = [
    "CausalSelfAttention",
    "PocketCrossAttention",
    "PocketTransformerBlock",
    "PocketTransformerDecoder",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "apply_rope",
]
