"""Top-level GenEuclidean v2 multi-task 3D pocket-conditioned generative model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from geneuclidean.config import ExperimentConfig
from geneuclidean.models.cvae import PocketCVAE
from geneuclidean.models.decoders import PocketLSTMDecoder, PocketTransformerDecoder, build_decoder
from geneuclidean.models.encoders import PocketEncoderOutput, build_encoder
from geneuclidean.models.heads import MultiTaskHeads


class ModelOutput(NamedTuple):
    """Complete structured output from ``GenEuclideanModel.forward``."""

    logits: Tensor
    attn_weights: Tensor | None
    pocket_out: PocketEncoderOutput | None
    mu_q: Tensor | None
    logvar_q: Tensor | None
    mu_p: Tensor | None
    logvar_p: Tensor | None
    pred_pkd: Tensor | None
    pred_props: Tensor | None
    clip_pocket: Tensor | None
    clip_ligand: Tensor | None
    clip_logit_scale: Tensor | None


class GenEuclideanModel(nn.Module):
    """End-to-end SE(3)/E(3)-equivariant 3D pocket-to-molecule model.

    Unifies:
    - Pluggable 3D pocket encoder (``PaiNN-SE3``, ``EGNN``, ``E3NN``, ``SchNet``).
    - Optional TamGen-style ``PocketCVAE`` latent bridge.
    - Learnable ``null_pocket_token`` for autoregressive Classifier-Free Guidance (CFG).
    - Pluggable autoregressive decoder (RoPE/SwiGLU Transformer or Attention-LSTM).
    - Joint multi-task heads (``pKd`` affinity, ``DrugCLIP`` alignment, 6 ligand properties).
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        cross_attn_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder: nn.Module = build_encoder(
            cfg.encoder,
            max_neighbors=cfg.data.max_neighbors,
        )
        self.pocket_to_decoder: nn.Module = (
            nn.Linear(cfg.encoder.out_dim, cfg.decoder.d_model)
            if cfg.encoder.out_dim != cfg.decoder.d_model
            else nn.Identity()
        )
        self.null_pocket_token = nn.Parameter(
            torch.randn(1, 1, cfg.decoder.d_model, dtype=torch.float32) * 0.02
        )

        self.heads = MultiTaskHeads(
            pocket_dim=cfg.encoder.out_dim,
            d_model=cfg.decoder.d_model,
            cfg=cfg.multitask,
            dropout=cfg.encoder.dropout,
        )

        self.cvae: PocketCVAE | None
        if cfg.cvae.enabled:
            self.cvae = PocketCVAE(
                pocket_dim=cfg.encoder.out_dim,
                d_model=cfg.decoder.d_model,
                latent_dim=cfg.cvae.latent_dim,
                num_latent_tokens=cfg.cvae.num_latent_tokens,
                ligand_dim=cfg.decoder.d_model,
                dropout=cfg.encoder.dropout,
                ligand_fp_encoder=self.heads.encode_ligand,
            )
        else:
            self.cvae = None

        self.decoder: PocketTransformerDecoder | PocketLSTMDecoder = build_decoder(
            cfg.decoder,
            cross_attn_gate_init=cross_attn_gate_init,
        )

    def encode_pocket(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode 3D protein pocket coordinates and features into invariant/equivariant tensors."""
        out: PocketEncoderOutput = self.encoder(coords, node_feats, atom_mask)
        return out

    def build_memory(
        self,
        pocket_out: PocketEncoderOutput,
        atom_mask: Tensor,
        z_tokens: Tensor | None = None,
        use_null_pocket: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Construct cross-attention memory ``[B, M, d_model]`` and boolean mask ``[B, M]``.

        When ``use_null_pocket=True``, replaces all pocket and latent tokens with the
        learned single ``null_pocket_token`` for Classifier-Free Guidance (CFG).
        """
        batch_size = pocket_out.node_emb.shape[0]
        device = pocket_out.node_emb.device

        if use_null_pocket:
            null_mem = self.null_pocket_token.expand(batch_size, 1, -1).contiguous()
            null_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            return null_mem, null_mask

        mask_bool = atom_mask.to(torch.bool)
        mem: Tensor = self.pocket_to_decoder(pocket_out.node_emb)
        mem = mem * mask_bool.unsqueeze(-1).to(mem.dtype)

        if z_tokens is not None:
            z_mask = torch.ones((batch_size, z_tokens.shape[1]), dtype=torch.bool, device=device)
            mem = torch.cat([mem, z_tokens], dim=1)
            mask_bool = torch.cat([mask_bool, z_mask], dim=1)

        return mem, mask_bool

    def forward(
        self,
        batch: Mapping[str, Any],
        cfg_drop: bool = False,
        return_attn: bool = False,
        latent_temp: float = 1.0,
    ) -> ModelOutput:
        """Run full multi-task forward pass over a collated pocket-ligand batch.

        If ``coords`` is absent or ``None``, runs Stage-1 unconditional decoding with
        ``memory=None``. When ``cfg_drop=True`` during training and ``cfg_drop_prob > 0``,
        randomly replaces conditioning memory with ``null_pocket_token`` per batch item.
        """
        input_ids: Tensor = batch["input_ids"]
        coords: Tensor | None = batch.get("coords")

        if coords is None:
            logits, attn_weights = self.decoder(
                input_ids,
                memory=None,
                memory_mask=None,
                return_attn=return_attn,
            )
            return ModelOutput(
                logits=logits,
                attn_weights=attn_weights,
                pocket_out=None,
                mu_q=None,
                logvar_q=None,
                mu_p=None,
                logvar_p=None,
                pred_pkd=None,
                pred_props=None,
                clip_pocket=None,
                clip_ligand=None,
                clip_logit_scale=None,
            )

        node_feats: Tensor = batch["node_feats"]
        atom_mask: Tensor = batch["atom_mask"]
        pocket_out = self.encode_pocket(coords, node_feats, atom_mask)

        ligand_fp: Tensor | None = batch.get("ligand_fp")
        ligand_emb: Tensor | None = (
            self.heads.encode_ligand(ligand_fp) if ligand_fp is not None else None
        )

        mt_out = self.heads(pocket_out.pooled_emb, ligand_emb=ligand_emb)

        mu_q: Tensor | None = None
        logvar_q: Tensor | None = None
        mu_p: Tensor | None = None
        logvar_p: Tensor | None = None
        z_tokens: Tensor | None = None

        if self.cvae is not None:
            # Deterministic mean in eval() for exact equivariance and reproducible validation.
            eff_temp = float(latent_temp) if self.training else 0.0
            cvae_out = self.cvae(
                pocket_out.pooled_emb,
                ligand_emb=ligand_emb,
                latent_temp=eff_temp,
            )
            z_tokens = cvae_out.z_tokens
            mu_q = cvae_out.mu_q
            logvar_q = cvae_out.logvar_q
            mu_p = cvae_out.mu_p
            logvar_p = cvae_out.logvar_p

        memory, memory_mask = self.build_memory(
            pocket_out=pocket_out,
            atom_mask=atom_mask,
            z_tokens=z_tokens,
            use_null_pocket=False,
        )

        if cfg_drop and self.training and self.cfg.decoder.cfg_drop_prob > 0.0:
            batch_size = input_ids.shape[0]
            drop_prob = self.cfg.decoder.cfg_drop_prob
            drop_mask = torch.rand(batch_size, device=memory.device) < drop_prob
            if bool(drop_mask.any().item()):
                memory = memory.clone()
                memory_mask = memory_mask.clone()
                memory[drop_mask] = 0.0
                memory[drop_mask, 0:1, :] = self.null_pocket_token.expand(
                    int(drop_mask.sum().item()), 1, -1
                )
                memory_mask[drop_mask] = False
                memory_mask[drop_mask, 0] = True

        logits, attn_weights = self.decoder(
            input_ids,
            memory=memory,
            memory_mask=memory_mask,
            return_attn=return_attn,
        )

        return ModelOutput(
            logits=logits,
            attn_weights=attn_weights,
            pocket_out=pocket_out,
            mu_q=mu_q,
            logvar_q=logvar_q,
            mu_p=mu_p,
            logvar_p=logvar_p,
            pred_pkd=mt_out.pred_pkd,
            pred_props=mt_out.pred_props,
            clip_pocket=mt_out.clip_pocket,
            clip_ligand=mt_out.clip_ligand,
            clip_logit_scale=mt_out.clip_logit_scale,
        )

    def score_candidates(
        self,
        pocket_out: PocketEncoderOutput,
        ligand_fps_tensor: Tensor,
    ) -> dict[str, Tensor]:
        """Score a batch of candidate ligand ECFP4 fingerprints against an encoded pocket.

        Automatically broadcasts a single encoded pocket ``[1, D]`` across ``M`` candidate
        ligands ``[M, fp_dim]``. Returns a dict with keys ``"pkd"`` and/or ``"clip_sim"``
        for each enabled multi-task head.
        """
        pooled = pocket_out.pooled_emb
        fps = ligand_fps_tensor.to(device=pooled.device, dtype=torch.float32)
        if fps.ndim == 1:
            fps = fps.unsqueeze(0)

        num_candidates = fps.shape[0]
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


__all__ = ["GenEuclideanModel", "ModelOutput"]
