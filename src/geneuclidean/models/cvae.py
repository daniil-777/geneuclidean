"""TamGen-style Pocket-Conditioned Conditional Variational Autoencoder (PocketCVAE).

Maps a 3D pocket embedding ``p`` (prior) and optional cognate/seed ligand embedding ``l``
(posterior) to a Gaussian latent variable ``z in R^{latent_dim}``, then projects ``z``
into ``K`` cross-attention memory tokens concatenated onto the pocket atom memory.
Supports both high-diversity de novo generation (``sample_prior``) and focused
hit-to-lead analog generation around a seed ligand (``refine``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


class CVAEOutput(NamedTuple):
    """Structured output returned by ``PocketCVAE.forward``."""

    z_tokens: Tensor
    z: Tensor
    mu_q: Tensor | None
    logvar_q: Tensor | None
    mu_p: Tensor
    logvar_p: Tensor


class PocketCVAE(nn.Module):
    """Pocket-conditioned CVAE bridge producing ``K`` latent memory tokens ``[B, K, D]``."""

    _LOGVAR_MIN: float = -8.0
    _LOGVAR_MAX: float = 4.0

    def __init__(
        self,
        pocket_dim: int = 256,
        d_model: int = 256,
        latent_dim: int = 64,
        num_latent_tokens: int = 4,
        ligand_dim: int | None = None,
        dropout: float = 0.1,
        ligand_fp_encoder: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.pocket_dim = int(pocket_dim)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        self.ligand_dim = int(ligand_dim if ligand_dim is not None else d_model)
        # Store optional fingerprint encoder outside nn.Module registry to avoid duplicate keys.
        self._fp_encoder_ref: list[Callable[[Tensor], Tensor]] = (
            [ligand_fp_encoder] if ligand_fp_encoder is not None else []
        )

        hidden = max(self.pocket_dim, self.latent_dim * 2)
        self.prior_net = nn.Sequential(
            nn.Linear(self.pocket_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * self.latent_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(self.pocket_dim + self.ligand_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * self.latent_dim),
        )
        self.z_to_tokens = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_latent_tokens * self.d_model),
        )
        self.token_pos_emb = nn.Parameter(
            torch.randn(1, self.num_latent_tokens, self.d_model, dtype=torch.float32) * 0.02
        )
        self.token_norm = nn.LayerNorm(self.d_model)

    def _resolve_ligand_emb(self, ligand_input: Tensor) -> Tensor:
        """Accept either pre-encoded ``[B, ligand_dim]`` or raw fingerprint ``[B, fp_dim]``."""
        x = ligand_input.to(dtype=torch.float32)
        if x.shape[-1] == self.ligand_dim:
            return x
        if self._fp_encoder_ref:
            return self._fp_encoder_ref[0](x)
        raise ValueError(
            f"Expected ligand embedding of dim {self.ligand_dim}, got shape {tuple(x.shape)} "
            "and no ligand_fp_encoder was attached."
        )

    def encode_prior(self, pocket_pooled: Tensor) -> tuple[Tensor, Tensor]:
        """Compute prior parameters ``(mu_p, logvar_p)`` from ``pocket_pooled[B, pocket_dim]``."""
        stats: Tensor = self.prior_net(pocket_pooled)
        mu_p, logvar_p = stats.chunk(2, dim=-1)
        return mu_p, logvar_p.clamp(min=self._LOGVAR_MIN, max=self._LOGVAR_MAX)

    def encode_posterior(
        self,
        pocket_pooled: Tensor,
        ligand_emb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute posterior parameters ``(mu_q, logvar_q)`` from pocket + ligand."""
        lig = self._resolve_ligand_emb(ligand_emb)
        stats: Tensor = self.posterior_net(torch.cat([pocket_pooled, lig], dim=-1))
        mu_q, logvar_q = stats.chunk(2, dim=-1)
        return mu_q, logvar_q.clamp(min=self._LOGVAR_MIN, max=self._LOGVAR_MAX)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor, latent_temp: float = 1.0) -> Tensor:
        """Reparameterization trick ``z = mu + (latent_temp * sigma) * eps``."""
        if latent_temp <= 0.0:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (float(latent_temp) * std) * eps

    def project_latent(self, z: Tensor) -> Tensor:
        """Project ``z[B, latent_dim]`` into ``[B, num_latent_tokens, d_model]`` tokens."""
        batch_size = z.shape[0]
        tokens = self.z_to_tokens(z).view(batch_size, self.num_latent_tokens, self.d_model)
        out: Tensor = self.token_norm(tokens + self.token_pos_emb)
        return out

    def forward(
        self,
        pocket_pooled: Tensor,
        ligand_emb: Tensor | None = None,
        latent_temp: float = 1.0,
    ) -> CVAEOutput:
        """Compute prior (and optional posterior) latent tokens.

        Uses the posterior ``q(z | p, l)`` when ``ligand_emb`` is provided (training /
        evaluation / refinement), and the prior ``p(z | p)`` otherwise (de novo inference).
        """
        mu_p, logvar_p = self.encode_prior(pocket_pooled)
        if ligand_emb is not None:
            mu_q, logvar_q = self.encode_posterior(pocket_pooled, ligand_emb)
            z = self.reparameterize(mu_q, logvar_q, latent_temp=latent_temp)
        else:
            mu_q = None
            logvar_q = None
            z = self.reparameterize(mu_p, logvar_p, latent_temp=latent_temp)

        z_tokens = self.project_latent(z)
        return CVAEOutput(
            z_tokens=z_tokens,
            z=z,
            mu_q=mu_q,
            logvar_q=logvar_q,
            mu_p=mu_p,
            logvar_p=logvar_p,
        )

    def sample_prior(
        self,
        pocket_pooled: Tensor,
        latent_temp: float = 1.0,
    ) -> Tensor:
        """Sample ``K`` latent memory tokens ``[B, K, d_model]`` from prior ``p(z | p)``."""
        return self.forward(pocket_pooled, ligand_emb=None, latent_temp=latent_temp).z_tokens

    def refine(
        self,
        pocket_pooled: Tensor,
        ligand_emb: Tensor,
        latent_temp: float = 0.5,
    ) -> Tensor:
        """Sample focused analog tokens ``[B, K, d_model]`` around a seed ligand posterior."""
        return self.forward(pocket_pooled, ligand_emb=ligand_emb, latent_temp=latent_temp).z_tokens


__all__ = ["CVAEOutput", "PocketCVAE"]
