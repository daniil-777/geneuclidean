"""3D SE(3)-Equivariant Pocket-to-LLM Multimodal Bridge for GenEuclidean v3.

Connects our verified ``PaiNN-SE3`` 3D pocket encoder, TamGen-style ``PocketCVAE``, and
``MultiTaskHeads`` (``pKd r=0.746``, ``DrugCLIP R@5=62.1%``) to any pretrained HuggingFace
chemical transformer or compact local LLM via:

- ``ScaleMatchRMSNorm``: RMSNorm with learnable scale initialized to the empirical token
  embedding RMS of the target LLM (e.g. ``1.0002`` for ``GP-MoLFormer-Uniq`` vs ``0.0156``
  for ``Qwen2.5-0.5B``).
- ``PocketPerceiverResampler``: Strictly permutation- and SE(3)-invariant 2-layer Perceiver
  resampler compressing variable-length pocket atoms ``[B, N, 256]`` into ``Q=16`` soft-prefix
  tokens ``[B, 16, d_llm]`` with exact ``0.0`` attention weight on masked dummy atoms.
- ``PocketConditionedLLM``: End-to-end multimodal model assembling a 24-token soft prefix
  (16 Perceiver + 4 CVAE + 4 Global/Property tokens), supporting Classifier-Free Guidance (CFG)
  dropout, pre-cached 3D pocket features, multi-task candidate scoring, and compact ~10-25 MB
  adapter-only checkpoints.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.config import ExperimentConfig
from geneuclidean.models.cvae import PocketCVAE
from geneuclidean.models.decoders.transformer import RMSNorm, SwiGLU
from geneuclidean.models.encoders import PocketEncoderOutput, build_encoder
from geneuclidean.models.heads import MultiTaskHeads
from geneuclidean.models.hf_backbones import LoadedBackbone, get_input_embeddings_tensor


class PocketLLMOutput(NamedTuple):
    """Structured output returned by ``PocketConditionedLLM.forward``."""

    loss: Tensor | None
    logits: Tensor
    full_logits: Tensor
    attn_weights: Tensor | None
    pocket_out: PocketEncoderOutput | None
    mu_q: Tensor | None
    logvar_q: Tensor | None
    mu_p: Tensor | None
    logvar_p: Tensor | None


class ScaleMatchRMSNorm(nn.Module):
    """RMSNorm with learnable scale initialized to ``init_rms`` to match LLM embedding scale."""

    def __init__(self, dim: int, init_rms: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        if init_rms <= 0.0:
            raise ValueError(f"init_rms must be strictly positive, got {init_rms}")
        self.dim = int(dim)
        self.init_rms = float(init_rms)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.full((self.dim,), self.init_rms, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` in float32 and rescale by learnable ``weight``."""
        x_fp32 = x.to(dtype=torch.float32)
        inv_rms = torch.rsqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 * inv_rms) * self.weight
        return out.to(dtype=x.dtype)


class _PerceiverBlock(nn.Module):
    """Single Perceiver layer: masked atom cross-attention -> latent self-attention -> SwiGLU."""

    def __init__(
        self,
        llm_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if llm_dim % num_heads != 0:
            raise ValueError(f"llm_dim ({llm_dim}) must be divisible by num_heads ({num_heads})")
        self.llm_dim = int(llm_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.llm_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = float(dropout)

        # Cross-attention from learnable queries to pocket atom features
        self.norm_q = RMSNorm(self.llm_dim)
        self.norm_kv = RMSNorm(self.llm_dim)
        self.cross_q = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.cross_k = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.cross_v = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.cross_o = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.cross_resid_drop = nn.Dropout(dropout)

        # Self-attention among the Q latent tokens
        self.norm_sa = RMSNorm(self.llm_dim)
        self.sa_q = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.sa_k = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.sa_v = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.sa_o = nn.Linear(self.llm_dim, self.llm_dim, bias=False)
        self.sa_resid_drop = nn.Dropout(dropout)

        # SwiGLU feed-forward network
        self.norm_ff = RMSNorm(self.llm_dim)
        d_ff = max(self.llm_dim * 2, ((self.llm_dim * 8 // 3 + 31) // 32) * 32)
        self.ffn = SwiGLU(d_model=self.llm_dim, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        latents: Tensor,
        atom_kv: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Update ``latents[B, Q, D]`` from ``atom_kv[B, N, D]`` with exact ``atom_mask[B, N]``.

        Returns:
            ``(updated_latents[B, Q, D], cross_attn_weights[B, H, Q, N])`` where every entry at
            ``~atom_mask`` is strictly ``0.0``.
        """
        batch_size, num_q, _ = latents.shape
        num_atoms = atom_kv.shape[1]
        mask_bool = atom_mask.to(device=latents.device, dtype=torch.bool)

        # 1. Masked cross-attention (computed in float32 for exact zero-mask & <1e-5 SE(3) parity)
        q_in = self.norm_q(latents)
        kv_in = self.norm_kv(atom_kv)
        kv_in = kv_in * mask_bool.unsqueeze(-1).to(dtype=kv_in.dtype)

        q = (
            self.cross_q(q_in)
            .view(batch_size, num_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.cross_k(kv_in)
            .view(batch_size, num_atoms, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.cross_v(kv_in)
            .view(batch_size, num_atoms, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Q, N]
        key_mask = mask_bool[:, None, None, :]  # [B, 1, 1, N]
        has_valid = mask_bool.any(dim=-1)  # [B]

        # Guard all-masked rows against softmax(-inf) -> NaN
        safe_key_mask = torch.where(
            has_valid[:, None, None, None],
            key_mask,
            torch.ones_like(key_mask),
        )
        masked_scores = scores.masked_fill(~safe_key_mask, float("-inf"))
        attn_weights = torch.softmax(masked_scores, dim=-1)
        attn_weights = attn_weights * key_mask.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights * has_valid[:, None, None, None].to(dtype=attn_weights.dtype)

        attn_dropped = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        cross_ctx = torch.matmul(attn_dropped, v)  # [B, H, Q, Dh]
        cross_ctx = cross_ctx.transpose(1, 2).contiguous().view(batch_size, num_q, self.llm_dim)
        latents = latents + self.cross_resid_drop(self.cross_o(cross_ctx))

        # 2. Latent self-attention
        sa_in = self.norm_sa(latents)
        sq = self.sa_q(sa_in).view(batch_size, num_q, self.num_heads, self.head_dim).transpose(1, 2)
        sk = self.sa_k(sa_in).view(batch_size, num_q, self.num_heads, self.head_dim).transpose(1, 2)
        sv = self.sa_v(sa_in).view(batch_size, num_q, self.num_heads, self.head_dim).transpose(1, 2)
        sa_scores = torch.matmul(sq, sk.transpose(-2, -1)) * self.scale
        sa_probs = F.dropout(
            torch.softmax(sa_scores, dim=-1), p=self.dropout_p, training=self.training
        )
        sa_ctx = (
            torch.matmul(sa_probs, sv)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, num_q, self.llm_dim)
        )
        latents = latents + self.sa_resid_drop(self.sa_o(sa_ctx))

        # 3. SwiGLU FFN
        latents = latents + self.ffn(self.norm_ff(latents))
        return latents, attn_weights


class PocketPerceiverResampler(nn.Module):
    """Permutation- & SE(3)-invariant Perceiver resampler from pocket atoms to LLM prefix.

    Runs cross-attention, latent self-attention, and SwiGLU blocks in a compact pocket-aligned
    bottleneck dimension ``bridge_dim`` (default ``min(llm_dim, pocket_dim)``, e.g. 256 for
    ``PaiNN-SE3``) before projecting the ``Q`` refined tokens into ``llm_dim``. This keeps the
    3D-to-LLM bridge well-regularized on PDBbind (~2-3M params) and adapter checkpoints compact
    (~15-25 MB) while preserving exact ``0.0`` attention masking and permutation invariance.
    """

    def __init__(
        self,
        pocket_dim: int = 256,
        llm_dim: int = 768,
        num_queries: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if llm_dim % num_heads != 0:
            raise ValueError(f"llm_dim ({llm_dim}) must be divisible by num_heads ({num_heads})")

        self.pocket_dim = int(pocket_dim)
        self.llm_dim = int(llm_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        raw_bridge = min(self.llm_dim, self.pocket_dim)
        aligned_bridge = ((raw_bridge + self.num_heads - 1) // self.num_heads) * self.num_heads
        self.bridge_dim = aligned_bridge if aligned_bridge <= self.llm_dim else self.llm_dim

        self.atom_proj: nn.Module = (
            nn.Linear(self.pocket_dim, self.bridge_dim)
            if self.pocket_dim != self.bridge_dim
            else nn.Identity()
        )
        self.latents = nn.Parameter(
            torch.randn(1, self.num_queries, self.bridge_dim, dtype=torch.float32) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                _PerceiverBlock(llm_dim=self.bridge_dim, num_heads=self.num_heads, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.bridge_norm = RMSNorm(self.bridge_dim)
        self.out_proj: nn.Module = (
            nn.Linear(self.bridge_dim, self.llm_dim)
            if self.bridge_dim != self.llm_dim
            else nn.Identity()
        )
        self.out_norm = RMSNorm(self.llm_dim)

    def forward(
        self,
        node_emb: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compress ``node_emb[B, N, pocket_dim]`` to ``(queries[B, Q, D], attn[B, H, Q, N])``."""
        mask_bool = atom_mask.to(device=node_emb.device, dtype=torch.bool)
        # Zero masked dummy entries before projection so corrupted dummy values never leak.
        safe_nodes = torch.where(
            mask_bool.unsqueeze(-1),
            node_emb.to(dtype=torch.float32),
            torch.zeros_like(node_emb, dtype=torch.float32),
        )
        atom_kv: Tensor = self.atom_proj(safe_nodes)
        atom_kv = atom_kv * mask_bool.unsqueeze(-1).to(dtype=atom_kv.dtype)

        batch_size = node_emb.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)

        last_attn: Tensor | None = None
        for layer in self.layers:
            latents, last_attn = layer(latents, atom_kv, mask_bool)

        assert last_attn is not None
        projected: Tensor = self.out_proj(self.bridge_norm(latents))
        return self.out_norm(projected), last_attn


class PocketConditionedLLM(nn.Module):
    """GenEuclidean v3 3D-Pocket-Conditioned Multimodal Chemical Foundation / Local LLM."""

    def __init__(
        self,
        v2_cfg: ExperimentConfig,
        backbone: LoadedBackbone,
        num_queries: int = 16,
        perceiver_layers: int = 2,
        perceiver_heads: int = 8,
        num_global_tokens: int = 4,
        cfg_drop_prob: float = 0.15,
        freeze_pocket_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.v2_cfg = v2_cfg
        self.backbone_meta = backbone
        self.backbone_id = backbone.backbone_id
        self.tokenizer = backbone.tokenizer
        self.llm_dim = int(backbone.hidden_size)
        self.vocab_size = int(backbone.vocab_size)
        self.bos_token_id = int(backbone.bos_token_id)
        self.eos_token_id = int(backbone.eos_token_id)
        self.pad_token_id = int(backbone.pad_token_id)
        self.is_smiles_native = bool(backbone.is_smiles_native)
        self.prompt_prefix = str(backbone.prompt_prefix)
        self.uses_tuple_kv_cache = bool(backbone.uses_tuple_kv_cache)

        self.num_queries = int(num_queries)
        self.perceiver_layers = int(perceiver_layers)
        self.perceiver_heads = int(perceiver_heads)
        self.num_global_tokens = int(num_global_tokens)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.freeze_pocket_encoder = bool(freeze_pocket_encoder)
        self.v2_ckpt_path: str | None = None

        # 1. Warm-startable v2.0 3D geometric encoder, CVAE, and multi-task heads (strict fp32)
        self.encoder: nn.Module = build_encoder(
            v2_cfg.encoder,
            max_neighbors=v2_cfg.data.max_neighbors,
        )
        self.heads = MultiTaskHeads(
            pocket_dim=v2_cfg.encoder.out_dim,
            d_model=v2_cfg.decoder.d_model,
            cfg=v2_cfg.multitask,
            dropout=v2_cfg.encoder.dropout,
        )
        self.cvae: PocketCVAE | None
        if v2_cfg.cvae.enabled:
            self.cvae = PocketCVAE(
                pocket_dim=v2_cfg.encoder.out_dim,
                d_model=v2_cfg.decoder.d_model,
                latent_dim=v2_cfg.cvae.latent_dim,
                num_latent_tokens=v2_cfg.cvae.num_latent_tokens,
                ligand_dim=v2_cfg.decoder.d_model,
                dropout=v2_cfg.encoder.dropout,
                ligand_fp_encoder=self.heads.encode_ligand,
            )
            self.num_cvae_tokens = int(v2_cfg.cvae.num_latent_tokens)
            self.cvae_proj: nn.Module | None = nn.Linear(v2_cfg.decoder.d_model, self.llm_dim)
        else:
            self.cvae = None
            self.num_cvae_tokens = 0
            self.cvae_proj = None

        # 2. 3D-to-LLM multimodal soft-prefix bridge (strict fp32 before final cast to llm_dtype)
        self.perceiver = PocketPerceiverResampler(
            pocket_dim=v2_cfg.encoder.out_dim,
            llm_dim=self.llm_dim,
            num_queries=self.num_queries,
            num_layers=self.perceiver_layers,
            num_heads=self.perceiver_heads,
            dropout=v2_cfg.encoder.dropout,
        )
        global_in_dim = v2_cfg.encoder.out_dim + (
            MultiTaskHeads.NUM_LIGAND_PROPS if self.heads.prop_head is not None else 0
        )
        global_hidden = max(v2_cfg.encoder.out_dim, min(self.llm_dim, 256))
        self.global_proj = nn.Sequential(
            nn.Linear(global_in_dim, global_hidden),
            nn.SiLU(),
            nn.Linear(global_hidden, self.num_global_tokens * self.llm_dim),
        )
        self.prefix_norm = ScaleMatchRMSNorm(self.llm_dim, init_rms=backbone.embed_rms)

        self.num_prefix_tokens = self.num_queries + self.num_cvae_tokens + self.num_global_tokens
        raw_null = torch.randn(1, self.num_prefix_tokens, self.llm_dim, dtype=torch.float32)
        raw_null_rms = torch.sqrt(raw_null.pow(2).mean(dim=-1, keepdim=True)).clamp(min=1e-6)
        self.null_prefix = nn.Parameter(raw_null / raw_null_rms * float(backbone.embed_rms))

        # 3. Pretrained HF chemical transformer / compact local LLM (optionally PEFT LoRA-wrapped)
        self.llm: nn.Module = backbone.model

        if self.freeze_pocket_encoder:
            self._apply_pocket_freeze()

    @property
    def llm_dtype(self) -> torch.dtype:
        """Return the active parameter dtype of the underlying LLM backbone."""
        for param in self.llm.parameters():
            return param.dtype
        return self.backbone_meta.model_dtype

    def _apply_pocket_freeze(self) -> None:
        """Freeze v2.0 3D encoder, CVAE, and multi-task heads in eval mode."""
        for module in (self.encoder, self.heads, self.cvae):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode: bool = True) -> PocketConditionedLLM:
        """Set training mode while keeping frozen v2 pocket submodules strictly in eval mode."""
        super().train(mode)
        if self.freeze_pocket_encoder:
            for module in (self.encoder, self.heads, self.cvae):
                if module is not None:
                    module.eval()
        return self

    def bridge_parameters(self) -> list[nn.Parameter]:
        """Return all parameters belonging to the trainable 3D-to-LLM multimodal bridge."""
        params: list[nn.Parameter] = []
        for module in (self.perceiver, self.cvae_proj, self.global_proj, self.prefix_norm):
            if module is not None:
                params.extend(module.parameters())
        params.append(self.null_prefix)
        return params

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return all LoRA adapter parameters inside ``self.llm``."""
        return [
            param
            for name, param in self.llm.named_parameters()
            if "lora_" in name or "modules_to_save" in name
        ]

    def set_bridge_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients on the 3D-to-LLM bridge parameters."""
        for param in self.bridge_parameters():
            param.requires_grad_(trainable)

    def set_lora_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients on the LLM LoRA adapter parameters."""
        for param in self.lora_parameters():
            param.requires_grad_(trainable)

    def load_v2_pocket_checkpoint(
        self,
        ckpt_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Strictly warm-start ``encoder``, ``cvae``, and ``heads`` from a v2.0 checkpoint."""
        path = Path(ckpt_path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state_dict: dict[str, Tensor] = (
            payload["model_state_dict"]
            if isinstance(payload, dict) and "model_state_dict" in payload
            else payload
        )

        encoder_sd = {
            k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        heads_sd = {k[len("heads.") :]: v for k, v in state_dict.items() if k.startswith("heads.")}
        if not encoder_sd:
            raise KeyError(f"No 'encoder.*' weights found in checkpoint {path}")
        if not heads_sd:
            raise KeyError(f"No 'heads.*' weights found in checkpoint {path}")

        self.encoder.load_state_dict(encoder_sd, strict=True)
        self.heads.load_state_dict(heads_sd, strict=True)

        cvae_count = 0
        if self.cvae is not None:
            cvae_sd = {k[len("cvae.") :]: v for k, v in state_dict.items() if k.startswith("cvae.")}
            if not cvae_sd:
                raise KeyError(f"No 'cvae.*' weights found in checkpoint {path}")
            self.cvae.load_state_dict(cvae_sd, strict=True)
            cvae_count = len(cvae_sd)

        if self.freeze_pocket_encoder:
            self._apply_pocket_freeze()

        self.v2_ckpt_path = str(path)
        return {
            "ckpt_path": str(path),
            "encoder_tensors": len(encoder_sd),
            "cvae_tensors": cvae_count,
            "heads_tensors": len(heads_sd),
        }

    def encode_pocket(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode 3D protein pocket coordinates strictly in ``float32`` for exact SE(3) parity."""
        ctx = torch.no_grad() if self.freeze_pocket_encoder else contextlib.nullcontext()
        with ctx:
            out: PocketEncoderOutput = self.encoder(
                coords.to(dtype=torch.float32),
                node_feats.to(dtype=torch.float32),
                atom_mask.to(dtype=torch.bool),
            )
        return out

    def build_soft_prefix(
        self,
        pocket_out: PocketEncoderOutput | None,
        atom_mask: Tensor | None,
        ligand_fp: Tensor | None = None,
        use_null_pocket: bool = False,
        latent_temp: float = 1.0,
        use_posterior: bool = False,
        deterministic_z: bool = False,
        return_attn: bool = False,
        batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor | None]]:
        """Project 3D pocket features into ``[B, M, llm_dim]`` soft-prompt embeddings."""
        cvae_stats: dict[str, Tensor | None] = {
            "mu_q": None,
            "logvar_q": None,
            "mu_p": None,
            "logvar_p": None,
            "z": None,
        }
        target_dtype = self.llm_dtype

        if use_null_pocket:
            if batch_size is None:
                if pocket_out is not None:
                    batch_size = int(pocket_out.node_emb.shape[0])
                elif atom_mask is not None:
                    batch_size = int(atom_mask.shape[0])
                else:
                    raise ValueError(
                        "batch_size, pocket_out, or atom_mask required for null prefix"
                    )
            device = (
                pocket_out.node_emb.device if pocket_out is not None else self.null_prefix.device
            )
            null_prefix = (
                self.null_prefix.to(device=device, dtype=target_dtype)
                .expand(batch_size, -1, -1)
                .contiguous()
            )
            return null_prefix, None, cvae_stats

        if pocket_out is None or atom_mask is None:
            raise ValueError("pocket_out and atom_mask are required when use_null_pocket=False.")

        node_emb = pocket_out.node_emb.to(dtype=torch.float32)
        pooled_emb = pocket_out.pooled_emb.to(dtype=torch.float32)
        device = node_emb.device
        bsz = int(node_emb.shape[0])
        mask_bool = atom_mask.to(device=device, dtype=torch.bool)

        # 1. Perceiver resampler over pocket atoms -> [B, Q, llm_dim]
        query_tokens, attn_weights = self.perceiver(node_emb, mask_bool)

        prefix_parts: list[Tensor] = [query_tokens]

        # 2. PocketCVAE latent tokens -> [B, K, llm_dim]
        if self.cvae is not None and self.cvae_proj is not None:
            eff_temp = 0.0 if deterministic_z else float(latent_temp)
            ctx = torch.no_grad() if self.freeze_pocket_encoder else contextlib.nullcontext()
            with ctx:
                lig_emb = (
                    self.heads.encode_ligand(ligand_fp.to(device=device, dtype=torch.float32))
                    if (use_posterior and ligand_fp is not None)
                    else None
                )
                cvae_out = self.cvae(pooled_emb, ligand_emb=lig_emb, latent_temp=eff_temp)
            cvae_tokens: Tensor = self.cvae_proj(cvae_out.z_tokens.to(dtype=torch.float32))
            prefix_parts.append(cvae_tokens)
            cvae_stats = {
                "mu_q": cvae_out.mu_q,
                "logvar_q": cvae_out.logvar_q,
                "mu_p": cvae_out.mu_p,
                "logvar_p": cvae_out.logvar_p,
                "z": cvae_out.z,
            }

        # 3. Global pooled pocket (+ predicted ligand property vector) -> [B, G, llm_dim]
        if self.heads.prop_head is not None:
            ctx = torch.no_grad() if self.freeze_pocket_encoder else contextlib.nullcontext()
            with ctx:
                pred_props: Tensor = self.heads.prop_head(pooled_emb)
            global_in = torch.cat([pooled_emb, pred_props.to(dtype=torch.float32)], dim=-1)
        else:
            global_in = pooled_emb
        global_tokens = self.global_proj(global_in).view(bsz, self.num_global_tokens, self.llm_dim)
        prefix_parts.append(global_tokens)

        prefix_fp32 = torch.cat(prefix_parts, dim=1)  # [B, M, llm_dim]
        prefix_normed = self.prefix_norm(prefix_fp32)
        prefix_embeds = prefix_normed.to(dtype=target_dtype)
        return prefix_embeds, (attn_weights if return_attn else None), cvae_stats

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        coords: Tensor | None = None,
        node_feats: Tensor | None = None,
        atom_mask: Tensor | None = None,
        pocket_out: PocketEncoderOutput | None = None,
        ligand_fp: Tensor | None = None,
        cfg_drop: bool = False,
        use_posterior: bool = True,
        latent_temp: float = 1.0,
        use_null_pocket: bool = False,
        return_attn: bool = False,
    ) -> PocketLLMOutput:
        """Run pocket-conditioned causal LM forward pass with prepended 3D soft-prefix tokens.

        Supports both live 3D pocket tensors ``(coords, node_feats, atom_mask)`` and pre-cached
        ``pocket_out`` for zero encoder overhead during Phase A/B training.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device)

        if not use_null_pocket and pocket_out is None:
            if coords is None or node_feats is None or atom_mask is None:
                raise ValueError(
                    "Provide either pre-cached pocket_out + atom_mask, live (coords, node_feats, "
                    "atom_mask), or set use_null_pocket=True."
                )
            pocket_out = self.encode_pocket(coords, node_feats, atom_mask)

        deterministic_z = not self.training
        prefix_embeds, attn_weights, cvae_stats = self.build_soft_prefix(
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            ligand_fp=ligand_fp,
            use_null_pocket=use_null_pocket,
            latent_temp=latent_temp,
            use_posterior=use_posterior,
            deterministic_z=deterministic_z,
            return_attn=return_attn,
            batch_size=batch_size,
        )

        if cfg_drop and self.training and self.cfg_drop_prob > 0.0 and not use_null_pocket:
            drop_mask = torch.rand(batch_size, device=prefix_embeds.device) < self.cfg_drop_prob
            if bool(drop_mask.any().item()):
                null_expanded = self.null_prefix.to(
                    device=prefix_embeds.device, dtype=prefix_embeds.dtype
                ).expand(int(drop_mask.sum().item()), -1, -1)
                prefix_embeds = prefix_embeds.clone()
                prefix_embeds[drop_mask] = null_expanded

        token_embeds = get_input_embeddings_tensor(self.llm, input_ids)
        prefix_embeds = prefix_embeds.to(device=token_embeds.device, dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [B, M + L, D]

        prefix_len = int(prefix_embeds.shape[1])
        prefix_mask = torch.ones(
            (batch_size, prefix_len),
            dtype=attention_mask.dtype,
            device=token_embeds.device,
        )
        full_attention_mask = torch.cat(
            [prefix_mask, attention_mask.to(device=token_embeds.device)],
            dim=1,
        )

        llm_out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        full_logits: Tensor = llm_out.logits
        token_logits: Tensor = full_logits[:, prefix_len:, :]

        loss: Tensor | None = None
        if labels is not None and seq_len > 1:
            effective_labels = labels.to(device=token_logits.device, dtype=torch.long)
            pad_mask = attention_mask.to(device=token_logits.device) == 0
            effective_labels = effective_labels.masked_fill(pad_mask, -100)

            shift_logits = token_logits[:, :-1, :].float()
            shift_labels = effective_labels[:, 1:]
            if bool((shift_labels != -100).any().item()):
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                loss = shift_logits.sum() * 0.0

        return PocketLLMOutput(
            loss=loss,
            logits=token_logits,
            full_logits=full_logits,
            attn_weights=attn_weights,
            pocket_out=pocket_out,
            mu_q=cvae_stats["mu_q"],
            logvar_q=cvae_stats["logvar_q"],
            mu_p=cvae_stats["mu_p"],
            logvar_p=cvae_stats["logvar_p"],
        )

    def score_candidates(
        self,
        pocket_out: PocketEncoderOutput,
        ligand_fps: Tensor,
    ) -> dict[str, Tensor]:
        """Score candidate ligand ECFP4 fingerprints using the warm-started v2 multi-task heads.

        Identical interface to ``GenEuclideanModel.score_candidates``, returning ``"pkd"`` and/or
        ``"clip_sim"`` tensors.
        """
        pooled = pocket_out.pooled_emb.to(dtype=torch.float32)
        fps = ligand_fps.to(device=pooled.device, dtype=torch.float32)
        if fps.ndim == 1:
            fps = fps.unsqueeze(0)

        num_candidates = int(fps.shape[0])
        if pooled.shape[0] == 1 and num_candidates > 1:
            pooled = pooled.expand(num_candidates, -1)
        elif pooled.shape[0] != num_candidates:
            raise ValueError(
                f"Cannot match pocket batch size {pooled.shape[0]} with "
                f"{num_candidates} candidate ligands."
            )

        ligand_emb = self.heads.encode_ligand(fps)
        mt_out = self.heads(pooled, ligand_emb=ligand_emb)

        scores: dict[str, Tensor] = {}
        if mt_out.pred_pkd is not None:
            scores["pkd"] = mt_out.pred_pkd
        if mt_out.clip_pocket is not None and mt_out.clip_ligand is not None:
            scores["clip_sim"] = (mt_out.clip_pocket * mt_out.clip_ligand).sum(dim=-1)
        return scores

    def save_adapter_checkpoint(
        self,
        path: str | Path,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Save only the compact 3D-to-LLM bridge and trainable LoRA weights (~10-25 MB)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bridge_sd: dict[str, Any] = {
            "perceiver": {k: v.detach().cpu() for k, v in self.perceiver.state_dict().items()},
            "cvae_proj": (
                {k: v.detach().cpu() for k, v in self.cvae_proj.state_dict().items()}
                if self.cvae_proj is not None
                else None
            ),
            "global_proj": {k: v.detach().cpu() for k, v in self.global_proj.state_dict().items()},
            "prefix_norm": {k: v.detach().cpu() for k, v in self.prefix_norm.state_dict().items()},
            "null_prefix": self.null_prefix.detach().cpu(),
        }
        lora_sd: dict[str, Tensor] = {
            name: param.detach().cpu()
            for name, param in self.llm.named_parameters()
            if "lora_" in name or "modules_to_save" in name
        }
        metadata: dict[str, Any] = {
            "backbone_id": self.backbone_id,
            "v2_ckpt_path": self.v2_ckpt_path,
            "num_queries": self.num_queries,
            "perceiver_layers": self.perceiver_layers,
            "perceiver_heads": self.perceiver_heads,
            "num_global_tokens": self.num_global_tokens,
            "cfg_drop_prob": self.cfg_drop_prob,
            "embed_rms": self.backbone_meta.embed_rms,
            "v2_cfg": self.v2_cfg.model_dump(mode="python"),
        }
        if extra_meta:
            metadata.update(extra_meta)

        torch.save(
            {
                "bridge_state_dict": bridge_sd,
                "lora_state_dict": lora_sd,
                "metadata": metadata,
            },
            out_path,
        )
        return out_path

    def load_adapter_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Restore compact 3D-to-LLM bridge and LoRA weights bit-exact into ``self``."""
        in_path = Path(path)
        payload = torch.load(in_path, map_location=map_location, weights_only=False)
        bridge_sd: dict[str, Any] = payload["bridge_state_dict"]
        lora_sd: dict[str, Tensor] = payload.get("lora_state_dict", {})
        metadata: dict[str, Any] = dict(payload.get("metadata", {}))

        self.perceiver.load_state_dict(bridge_sd["perceiver"], strict=True)
        if self.cvae_proj is not None and bridge_sd.get("cvae_proj") is not None:
            self.cvae_proj.load_state_dict(bridge_sd["cvae_proj"], strict=True)
        self.global_proj.load_state_dict(bridge_sd["global_proj"], strict=True)
        self.prefix_norm.load_state_dict(bridge_sd["prefix_norm"], strict=True)
        with torch.no_grad():
            self.null_prefix.copy_(
                bridge_sd["null_prefix"].to(
                    device=self.null_prefix.device, dtype=self.null_prefix.dtype
                )
            )

        if lora_sd:
            named_params = dict(self.llm.named_parameters())
            missing = [k for k in lora_sd if k not in named_params]
            if missing:
                raise KeyError(
                    f"Adapter checkpoint {in_path} contains {len(missing)} LoRA keys absent from "
                    f"backbone {self.backbone_id!r}: {missing[:5]}"
                )
            with torch.no_grad():
                for key, tensor in lora_sd.items():
                    target = named_params[key]
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))

        return metadata


__all__ = [
    "PocketConditionedLLM",
    "PocketLLMOutput",
    "PocketPerceiverResampler",
    "ScaleMatchRMSNorm",
]
