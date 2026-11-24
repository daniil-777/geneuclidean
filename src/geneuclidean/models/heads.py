"""Multi-task prediction heads: Ligand ECFP4 encoder, pKd affinity, DrugCLIP, and properties."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.config import MultiTaskConfig


class MultiTaskOutput(NamedTuple):
    """Structured predictions from ``MultiTaskHeads.forward``."""

    pred_pkd: Tensor | None
    pred_props: Tensor | None
    clip_pocket: Tensor | None
    clip_ligand: Tensor | None
    clip_logit_scale: Tensor | None


class LigandFingerprintEncoder(nn.Module):
    """Encode 1024-bit ECFP4 Morgan fingerprints into a dense ``d_model`` ligand embedding."""

    def __init__(
        self,
        fp_dim: int = 1024,
        out_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fp_dim = int(fp_dim)
        self.out_dim = int(out_dim)
        hidden = max(out_dim * 2, 256)
        self.net = nn.Sequential(
            nn.Linear(self.fp_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, ligand_fp: Tensor) -> Tensor:
        """Map ``ligand_fp[B, fp_dim]`` -> ``ligand_emb[B, out_dim]``."""
        out: Tensor = self.net(ligand_fp.to(dtype=torch.float32))
        return out


class MultiTaskHeads(nn.Module):
    """Joint binding affinity (pKd), DrugCLIP contrastive, and pocket property heads.

    Owns ``self.ligand_encoder`` (shared across CVAE posterior, pKd, and DrugCLIP).
    ``clip_logit_scale`` is a learnable log-temperature scalar clamped to ``log(100)``.
    """

    NUM_LIGAND_PROPS: int = 6
    _MAX_LOG_SCALE: float = math.log(100.0)

    def __init__(
        self,
        pocket_dim: int = 256,
        d_model: int = 256,
        cfg: MultiTaskConfig | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mt_cfg = cfg if cfg is not None else MultiTaskConfig()
        self.pocket_dim = int(pocket_dim)
        self.d_model = int(d_model)
        self.cfg = mt_cfg

        self.ligand_encoder = LigandFingerprintEncoder(
            fp_dim=mt_cfg.ligand_fp_dim,
            out_dim=d_model,
            dropout=dropout,
        )
        self.pocket_align: nn.Module = (
            nn.Linear(pocket_dim, d_model) if pocket_dim != d_model else nn.Identity()
        )

        self.pkd_head: nn.Module | None
        if mt_cfg.use_pkd_head:
            self.pkd_head = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.pkd_head = None

        self.clip_pocket_proj: nn.Module | None
        self.clip_ligand_proj: nn.Module | None
        self.logit_scale: nn.Parameter | None
        if mt_cfg.use_clip_head:
            clip_dim = int(mt_cfg.clip_dim)
            self.clip_pocket_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, clip_dim, bias=False),
            )
            self.clip_ligand_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, clip_dim, bias=False),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        else:
            self.clip_pocket_proj = None
            self.clip_ligand_proj = None
            self.logit_scale = None

        self.prop_head: nn.Module | None
        if mt_cfg.use_prop_head:
            self.prop_head = nn.Sequential(
                nn.Linear(pocket_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.NUM_LIGAND_PROPS),
            )
        else:
            self.prop_head = None

    def encode_ligand(self, ligand_fp: Tensor) -> Tensor:
        """Encode raw ECFP4 fingerprint tensor ``[B, fp_dim]`` -> ``[B, d_model]``."""
        out: Tensor = self.ligand_encoder(ligand_fp)
        return out

    def forward(
        self,
        pocket_pooled: Tensor,
        ligand_emb: Tensor | None = None,
    ) -> MultiTaskOutput:
        """Compute enabled multi-task outputs from pooled pocket and optional ligand embeddings."""
        p_aligned: Tensor = self.pocket_align(pocket_pooled)

        pred_props: Tensor | None = None
        if self.prop_head is not None:
            pred_props = self.prop_head(pocket_pooled)

        pred_pkd: Tensor | None = None
        if self.pkd_head is not None and ligand_emb is not None:
            interaction = torch.cat(
                [p_aligned, ligand_emb, p_aligned * ligand_emb, torch.abs(p_aligned - ligand_emb)],
                dim=-1,
            )
            pred_pkd = self.pkd_head(interaction).squeeze(-1)

        clip_pocket: Tensor | None = None
        clip_ligand: Tensor | None = None
        clip_logit_scale: Tensor | None = None
        if (
            self.clip_pocket_proj is not None
            and self.clip_ligand_proj is not None
            and self.logit_scale is not None
        ):
            clip_pocket = F.normalize(self.clip_pocket_proj(p_aligned), p=2, dim=-1, eps=1e-8)
            clip_logit_scale = self.logit_scale.clamp(
                min=-self._MAX_LOG_SCALE,
                max=self._MAX_LOG_SCALE,
            )
            if ligand_emb is not None:
                clip_ligand = F.normalize(self.clip_ligand_proj(ligand_emb), p=2, dim=-1, eps=1e-8)

        return MultiTaskOutput(
            pred_pkd=pred_pkd,
            pred_props=pred_props,
            clip_pocket=clip_pocket,
            clip_ligand=clip_ligand,
            clip_logit_scale=clip_logit_scale,
        )


__all__ = ["LigandFingerprintEncoder", "MultiTaskHeads", "MultiTaskOutput"]
