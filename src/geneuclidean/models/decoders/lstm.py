"""Bug-fixed 2-layer Attention-LSTM molecular decoder (GenEuclidean v1.5 baseline).

Faithfully modernizes the 2021 ETH Zurich thesis Attention-LSTM decoder while fixing all
legacy bugs:
- Strictly per-sample masked Bahdanau attention (exact ``0.0`` weight on padded atoms,
  zero cross-batch leakage).
- Arbitrary variable pocket atom counts (no hardcoded ``17x17`` grid).
- Clean Stage-1 unconditional support (``memory=None``) and backbone weight loading.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from geneuclidean.config import DecoderConfig


class MaskedBahdanauAttention(nn.Module):
    """Vectorized additive (Bahdanau) pocket attention with exact padded-atom masking."""

    def __init__(self, query_dim: int, memory_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(memory_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute context ``[B, L, D]`` and attention weights ``[B, 1, L, M]``."""
        if memory_mask is not None:
            mask_bool = memory_mask.to(torch.bool)
            safe_memory = memory * mask_bool.unsqueeze(-1).to(memory.dtype)
        else:
            mask_bool = None
            safe_memory = memory

        q_proj = self.w_q(queries)  # [B, L, A]
        k_proj = self.w_k(safe_memory)  # [B, M, A]
        energy = torch.tanh(q_proj.unsqueeze(2) + k_proj.unsqueeze(1))  # [B, L, M, A]
        scores = self.v(energy).squeeze(-1)  # [B, L, M]

        if mask_bool is not None:
            key_mask = mask_bool.unsqueeze(1)  # [B, 1, M]
            all_masked = ~mask_bool.any(dim=-1, keepdim=True).unsqueeze(1)  # [B, 1, 1]
            scores = scores.masked_fill(~key_mask, float("-inf"))
            scores = scores.masked_fill(all_masked, 0.0)
            weights = torch.softmax(scores, dim=-1)
            weights = weights.masked_fill(~key_mask, 0.0)
        else:
            weights = torch.softmax(scores, dim=-1)

        context = torch.bmm(weights, safe_memory)  # [B, L, D]
        return context, weights.unsqueeze(1)  # [B, 1, L, M]


class PocketLSTMDecoder(nn.Module):
    """2-layer causal Attention-LSTM decoder compatible with ``PocketTransformerDecoder``."""

    def __init__(
        self,
        vocab_size: int | DecoderConfig = 96,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(vocab_size, DecoderConfig):
            cfg = vocab_size
            vocab_size_int = int(cfg.vocab_size)
            d_model = int(cfg.d_model)
            n_layers = int(cfg.n_layers)
            dropout = float(cfg.dropout)
            use_cross_attn = bool(cfg.use_cross_attn)
        else:
            vocab_size_int = int(vocab_size)

        self.vocab_size = vocab_size_int
        self.d_model = int(d_model)
        self.n_layers = max(1, int(n_layers))
        self.use_cross_attn = bool(use_cross_attn)
        self.pad_id = int(pad_id)

        self.token_emb = nn.Embedding(vocab_size_int, d_model, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=dropout if self.n_layers > 1 else 0.0,
        )

        self.init_h = nn.Linear(d_model, self.n_layers * d_model)
        self.init_c = nn.Linear(d_model, self.n_layers * d_model)

        if self.use_cross_attn:
            self.pocket_attn: MaskedBahdanauAttention | None = MaskedBahdanauAttention(
                query_dim=d_model,
                memory_dim=d_model,
                attn_dim=d_model,
            )
            self.fuse = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        else:
            self.pocket_attn = None
            self.fuse = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )

        self.lm_head = nn.Linear(d_model, vocab_size_int, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def _init_hidden(
        self,
        batch_size: int,
        memory: Tensor | None,
        memory_mask: Tensor | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if memory is None:
            zeros = torch.zeros(
                (self.n_layers, batch_size, self.d_model),
                device=device,
                dtype=torch.float32,
            )
            return zeros, zeros.clone()

        if memory_mask is not None:
            mask_bool = memory_mask.to(torch.bool)
            mask_f = mask_bool.unsqueeze(-1).to(memory.dtype)
            counts = mask_bool.sum(dim=1, keepdim=True).clamp(min=1).to(memory.dtype)
            pooled = (memory * mask_f).sum(dim=1) / counts
        else:
            pooled = memory.mean(dim=1)

        h0 = (
            torch.tanh(self.init_h(pooled))
            .view(batch_size, self.n_layers, self.d_model)
            .transpose(0, 1)
            .contiguous()
        )
        c0 = (
            torch.tanh(self.init_c(pooled))
            .view(batch_size, self.n_layers, self.d_model)
            .transpose(0, 1)
            .contiguous()
        )
        return h0, c0

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Decode token sequences conditioned on optional 3D pocket memory."""
        batch_size = input_ids.shape[0]
        x = self.emb_dropout(self.token_emb(input_ids))
        h0, c0 = self._init_hidden(batch_size, memory, memory_mask, device=input_ids.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # [B, L, D]

        attn_weights: Tensor | None = None
        if self.use_cross_attn and self.pocket_attn is not None:
            if memory is not None:
                context, attn_weights = self.pocket_attn(lstm_out, memory, memory_mask=memory_mask)
            else:
                context = torch.zeros_like(lstm_out)
            fused = self.fuse(torch.cat([lstm_out, context], dim=-1))
        else:
            fused = self.fuse(lstm_out)

        logits: Tensor = self.lm_head(fused)
        return logits, (attn_weights if return_attn else None)

    def load_pretrained_backbone(
        self,
        state_dict: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Transfer Stage-1 pretrained LSTM weights into a Stage-2 conditional LSTM."""
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
        self.lm_head.weight = self.token_emb.weight
        return missing, unexpected


__all__ = ["MaskedBahdanauAttention", "PocketLSTMDecoder"]
