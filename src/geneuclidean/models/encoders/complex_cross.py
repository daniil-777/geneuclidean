"""SE(3)-invariant joint protein-pocket <-> 3D-ligand interaction cross-attention (RoITr PPF).

Provides :class:`SE3ComplexCrossAttention`, a shared geometric cross-attention bridge that
couples encoded protein pocket representations ``(H_P, V_P, X_P, mask_P)`` and 3D ligand
representations ``(H_L, V_L, X_L, mask_L)`` from ANY GenEuclidean 3D encoder (``PaiNN-SE3``,
``DensePaiNN-SE3``, ``CartFormer-SE3``, ``AtomGS-SE3``, ``SubPocketVN-JEPA``) in one **shared
pocket centroid frame**.

Every interaction bias is strictly invariant under joint proper rigid motions
:math:`(\\mathbf{R}, \\mathbf{t}) \\in SE(3)` (even ``100 A`` translations) while remaining
strictly sensitive to relative ligand pose shifts
:math:`\\Delta\\mathbf{t} = \\mathbf{t}_L - \\mathbf{t}_P` and (when ``use_se3_chirality=True``)
enantiomeric mirror reflections (:math:`\\det\\mathbf{P} = -1`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from geneuclidean.models.encoders.dense_knn import (
    center_masked_coords_with_offset,
    safe_masked_softmax,
)
from geneuclidean.models.encoders.electrostatics import ScreenedCoulombBias
from geneuclidean.models.encoders.rbf import BesselRBF, CosineCutoff, MaskedAttentionPool
from geneuclidean.models.encoders.tensor_ops import VectorNeuronLinear, chiral_cross_product


class SE3ComplexCrossAttention(nn.Module):
    """Bidirectional ``SE(3)``-invariant pocket <-> 3D ligand cross-attention with RoITr PPF.

    Combines three strictly rigid-invariant geometric attention biases across ``[B, N_P, N_L]``:
      1. **Bessel RBF + Smooth Cosine Cutoff** over inter-molecular atomic distances ``d_ij``.
      2. **Screened ``1/r`` Latent-Charge Electrostatics** (:class:`ScreenedCoulombBias`,
         ``LES`` lineage) capturing 6-15 Angstrom salt bridges and metal coordination.
      3. **RoITr Intrinsic Point Pair Features (PPF) + Chiral Cross-Product Invariants**:
         ``[dot(n_i^P, d_ij), dot(n_j^L, d_ij), dot(n_i^P, n_j^L),
         dot(v_i^{P,x} - v_j^{L,x}, d_ij), dot(n_i^P x n_j^L, d_ij)]``.
    """

    emb_dim: int
    vector_dim: int
    num_heads: int
    head_dim: int
    cutoff: float
    use_se3_chirality: bool
    use_coulomb_bias: bool

    def __init__(
        self,
        emb_dim: int = 256,
        vector_dim: int = 64,
        num_heads: int = 8,
        num_rbf: int = 24,
        cutoff: float = 8.0,
        *,
        use_se3_chirality: bool = True,
        use_coulomb_bias: bool = True,
        dropout: float = 0.0,
        debye_len: float = 8.0,
        coulomb_r0: float = 1.0,
    ) -> None:
        super().__init__()
        if emb_dim < 1 or num_heads < 1 or emb_dim % num_heads != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be positive and divisible by num_heads ({num_heads})"
            )
        if vector_dim < 1:
            raise ValueError(f"vector_dim must be >= 1, got {vector_dim}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.emb_dim = int(emb_dim)
        self.vector_dim = int(vector_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.emb_dim // self.num_heads
        self.cutoff = float(cutoff)
        self.use_se3_chirality = bool(use_se3_chirality)
        self.use_coulomb_bias = bool(use_coulomb_bias)

        # Pairwise distance RBF bias
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=self.cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=self.cutoff)
        self.rbf_proj = nn.Linear(num_rbf, self.num_heads)

        # Long-range screened Coulomb electrostatic bias (LES)
        self.coulomb: ScreenedCoulombBias | None
        if self.use_coulomb_bias:
            self.coulomb = ScreenedCoulombBias(
                scalar_dim=self.emb_dim,
                num_heads=self.num_heads,
                r0=coulomb_r0,
                debye_len=debye_len,
            )
        else:
            self.coulomb = None

        # RoITr PPF & chiral cross-product vector projections (bias-free O(3)-equivariant)
        self.pocket_dir_proj = VectorNeuronLinear(self.vector_dim, 1)
        self.ligand_dir_proj = VectorNeuronLinear(self.vector_dim, 1)
        self.pocket_chiral_a: VectorNeuronLinear | None
        self.pocket_chiral_b: VectorNeuronLinear | None
        self.ligand_chiral_a: VectorNeuronLinear | None
        self.ligand_chiral_b: VectorNeuronLinear | None
        if self.use_se3_chirality:
            self.pocket_chiral_a = VectorNeuronLinear(self.vector_dim, 1)
            self.pocket_chiral_b = VectorNeuronLinear(self.vector_dim, 1)
            self.ligand_chiral_a = VectorNeuronLinear(self.vector_dim, 1)
            self.ligand_chiral_b = VectorNeuronLinear(self.vector_dim, 1)
        else:
            self.pocket_chiral_a = None
            self.pocket_chiral_b = None
            self.ligand_chiral_a = None
            self.ligand_chiral_b = None

        ppf_hidden = max(16, self.num_heads * 4)
        self.ppf_mlp = nn.Sequential(
            nn.Linear(5, ppf_hidden),
            nn.SiLU(),
            nn.Linear(ppf_hidden, self.num_heads),
        )

        # Bidirectional P <-> L attention projections
        self.p_norm = nn.LayerNorm(self.emb_dim)
        self.l_norm = nn.LayerNorm(self.emb_dim)
        self.p_q = nn.Linear(self.emb_dim, self.emb_dim)
        self.p_kv = nn.Linear(self.emb_dim, 2 * self.emb_dim)
        self.l_q = nn.Linear(self.emb_dim, self.emb_dim)
        self.l_kv = nn.Linear(self.emb_dim, 2 * self.emb_dim)
        self.p_out = nn.Linear(self.emb_dim, self.emb_dim)
        self.l_out = nn.Linear(self.emb_dim, self.emb_dim)
        self.p_post_norm = nn.LayerNorm(self.emb_dim)
        self.l_post_norm = nn.LayerNorm(self.emb_dim)
        self.dropout = nn.Dropout(dropout)

        # Joint complex readout
        self.pocket_pool = MaskedAttentionPool(in_dim=self.emb_dim, out_dim=self.emb_dim)
        self.ligand_pool = MaskedAttentionPool(in_dim=self.emb_dim, out_dim=self.emb_dim)
        self.complex_proj = nn.Sequential(
            nn.Linear(3 * self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def _compute_ppf_biases(
        self,
        pocket_vec: Tensor,
        ligand_vec: Tensor,
        dir_pl: Tensor,
        pocket_mask: Tensor,
        ligand_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute RoITr PPF + chiral cross biases for both P<-L and L<-P directions."""
        batch_size, n_p, n_l, _ = dir_pl.shape
        p_raw = self.pocket_dir_proj(pocket_vec).squeeze(-2)  # [B, N_P, 3]
        l_raw = self.ligand_dir_proj(ligand_vec).squeeze(-2)  # [B, N_L, 3]
        n_p_dir = p_raw * torch.rsqrt((p_raw * p_raw).sum(dim=-1, keepdim=True) + 1.0)
        n_l_dir = l_raw * torch.rsqrt((l_raw * l_raw).sum(dim=-1, keepdim=True) + 1.0)
        n_p_dir = torch.where(pocket_mask.unsqueeze(-1), n_p_dir, torch.zeros_like(n_p_dir))
        n_l_dir = torch.where(ligand_mask.unsqueeze(-1), n_l_dir, torch.zeros_like(n_l_dir))

        p1 = (n_p_dir.unsqueeze(2) * dir_pl).sum(dim=-1)  # <n_i^P, d_ij>
        p2 = (n_l_dir.unsqueeze(1) * dir_pl).sum(dim=-1)  # <n_j^L, d_ij>
        p3 = (n_p_dir.unsqueeze(2) * n_l_dir.unsqueeze(1)).sum(dim=-1)  # <n_i^P, n_j^L>

        if self.use_se3_chirality:
            assert (
                self.pocket_chiral_a is not None
                and self.pocket_chiral_b is not None
                and self.ligand_chiral_a is not None
                and self.ligand_chiral_b is not None
            )
            p_cross = chiral_cross_product(
                self.pocket_chiral_a(pocket_vec).squeeze(-2),
                self.pocket_chiral_b(pocket_vec).squeeze(-2),
                normalize=True,
                eps=1.0,
            )
            l_cross = chiral_cross_product(
                self.ligand_chiral_a(ligand_vec).squeeze(-2),
                self.ligand_chiral_b(ligand_vec).squeeze(-2),
                normalize=True,
                eps=1.0,
            )
            p_cross = torch.where(pocket_mask.unsqueeze(-1), p_cross, torch.zeros_like(p_cross))
            l_cross = torch.where(ligand_mask.unsqueeze(-1), l_cross, torch.zeros_like(l_cross))
            p4 = ((p_cross.unsqueeze(2) - l_cross.unsqueeze(1)) * dir_pl).sum(dim=-1)

            pair_cross = chiral_cross_product(
                n_p_dir.unsqueeze(2).expand(batch_size, n_p, n_l, 3),
                n_l_dir.unsqueeze(1).expand(batch_size, n_p, n_l, 3),
                normalize=True,
                eps=1.0,
            )
            p5 = (pair_cross * dir_pl).sum(dim=-1)
        else:
            p4 = torch.zeros_like(p1)
            p5 = torch.zeros_like(p1)

        ppf_pl = torch.stack((p1, p2, p3, p4, p5), dim=-1)
        # Reversing direction d_ji = -d_ij maps (p1, p2, p3, p4, p5) -> (-p2, -p1, p3, +p4, +p5).
        ppf_lp = torch.stack((-p2, -p1, p3, p4, p5), dim=-1).transpose(1, 2)

        bias_pl: Tensor = self.ppf_mlp(ppf_pl).permute(0, 3, 1, 2)  # [B, H, N_P, N_L]
        bias_lp: Tensor = self.ppf_mlp(ppf_lp).permute(0, 3, 1, 2)  # [B, H, N_L, N_P]
        return bias_pl, bias_lp

    def forward(
        self,
        pocket_emb: Tensor,
        pocket_vec: Tensor | None,
        pocket_coords: Tensor,
        pocket_mask: Tensor,
        ligand_emb: Tensor,
        ligand_vec: Tensor | None,
        ligand_coords: Tensor,
        ligand_mask: Tensor,
        *,
        pocket_raw_feats: Tensor | None = None,
        ligand_raw_feats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run bidirectional ``SE(3)``-invariant pocket <-> ligand cross-attention.

        Args:
            pocket_emb: Invariant pocket atom embeddings ``[B, N_P, emb_dim]``.
            pocket_vec: Optional equivariant pocket vectors ``[B, N_P, vector_dim, 3]``.
            pocket_coords: World-space pocket coordinates ``[B, N_P, 3]``.
            pocket_mask: Boolean pocket atom validity mask ``[B, N_P]``.
            ligand_emb: Invariant 3D ligand atom embeddings ``[B, N_L, emb_dim]``.
            ligand_vec: Optional equivariant ligand vectors ``[B, N_L, vector_dim, 3]``.
            ligand_coords: World-space ligand coordinates ``[B, N_L, 3]``.
            ligand_mask: Boolean ligand atom validity mask ``[B, N_L]``.
            pocket_raw_feats: Optional raw 52-D pocket features for ``dim 46`` charge prior.
            ligand_raw_feats: Optional raw 52-D ligand features for ``dim 46`` charge prior.

        Returns:
            Tuple ``(pocket_emb_out [B, N_P, emb_dim], ligand_emb_out [B, N_L, emb_dim],
            complex_pooled [B, emb_dim])``.
        """
        batch_size, n_p, _ = pocket_emb.shape
        n_l = ligand_emb.shape[1]
        p_mask = pocket_mask.to(torch.bool)
        l_mask = ligand_mask.to(torch.bool)

        # Shared pocket centroid frame cancels any joint rigid translation (e.g. 100 A).
        p_centered, p_offset = center_masked_coords_with_offset(pocket_coords, p_mask)
        l_coords_f = ligand_coords.to(torch.float32)
        l_in_p_frame = torch.where(
            l_mask.unsqueeze(-1),
            l_coords_f - p_offset.unsqueeze(1),
            torch.zeros_like(l_coords_f),
        )

        # Pairwise distances and smooth unit directions [B, N_P, N_L, 3]
        diff_pl = p_centered.unsqueeze(2) - l_in_p_frame.unsqueeze(1)
        sq_d = (diff_pl * diff_pl).sum(dim=-1)
        dist_pl = torch.sqrt(sq_d + 1e-8)
        dir_pl = diff_pl * torch.rsqrt(sq_d.unsqueeze(-1) + 1e-6)
        env_pl = self.cutoff_fn(dist_pl)

        rbf_bias: Tensor = self.rbf_proj(self.rbf(dist_pl) * env_pl.unsqueeze(-1)).permute(
            0, 3, 1, 2
        )  # [B, H, N_P, N_L]

        bias_pl = rbf_bias
        bias_lp = rbf_bias.transpose(-1, -2)

        if self.coulomb is not None:
            coul_pl = self.coulomb(
                p_centered,
                pocket_emb,
                raw_feats_i=pocket_raw_feats,
                mask_i=p_mask,
                coords_j=l_in_p_frame,
                scalar_j=ligand_emb,
                raw_feats_j=ligand_raw_feats,
                mask_j=l_mask,
            )
            bias_pl = bias_pl + coul_pl
            bias_lp = bias_lp + coul_pl.transpose(-1, -2)

        if pocket_vec is not None and ligand_vec is not None:
            ppf_pl, ppf_lp = self._compute_ppf_biases(
                pocket_vec, ligand_vec, dir_pl, p_mask, l_mask
            )
            bias_pl = bias_pl + ppf_pl
            bias_lp = bias_lp + ppf_lp

        # Bidirectional multi-head cross-attention with safe_masked_softmax (all_empty safe)
        p_h = torch.where(
            p_mask.unsqueeze(-1), self.p_norm(pocket_emb), torch.zeros_like(pocket_emb)
        )
        l_h = torch.where(
            l_mask.unsqueeze(-1), self.l_norm(ligand_emb), torch.zeros_like(ligand_emb)
        )

        scale = 1.0 / math.sqrt(self.head_dim)

        # Pocket <- Ligand
        qp = (
            self.p_q(p_h)
            .reshape(batch_size, n_p, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kl_vl = self.l_kv(l_h).reshape(batch_size, n_l, 2, self.num_heads, self.head_dim)
        kl = kl_vl[:, :, 0].permute(0, 2, 1, 3)
        vl = kl_vl[:, :, 1].permute(0, 2, 1, 3)

        logits_pl = torch.matmul(qp, kl.transpose(-1, -2)) * scale + bias_pl
        attn_pl = self.dropout(
            safe_masked_softmax(logits_pl, l_mask.unsqueeze(1).unsqueeze(2), dim=-1)
        )
        ctx_p = torch.matmul(attn_pl, vl).permute(0, 2, 1, 3).reshape(batch_size, n_p, self.emb_dim)
        p_out_raw: Tensor = self.p_post_norm(pocket_emb + self.dropout(self.p_out(ctx_p)))
        pocket_emb_out = torch.where(p_mask.unsqueeze(-1), p_out_raw, torch.zeros_like(p_out_raw))

        # Ligand <- Pocket
        ql = (
            self.l_q(l_h)
            .reshape(batch_size, n_l, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kp_vp = self.p_kv(p_h).reshape(batch_size, n_p, 2, self.num_heads, self.head_dim)
        kp = kp_vp[:, :, 0].permute(0, 2, 1, 3)
        vp = kp_vp[:, :, 1].permute(0, 2, 1, 3)

        logits_lp = torch.matmul(ql, kp.transpose(-1, -2)) * scale + bias_lp
        attn_lp = self.dropout(
            safe_masked_softmax(logits_lp, p_mask.unsqueeze(1).unsqueeze(2), dim=-1)
        )
        ctx_l = torch.matmul(attn_lp, vp).permute(0, 2, 1, 3).reshape(batch_size, n_l, self.emb_dim)
        l_out_raw: Tensor = self.l_post_norm(ligand_emb + self.dropout(self.l_out(ctx_l)))
        ligand_emb_out = torch.where(l_mask.unsqueeze(-1), l_out_raw, torch.zeros_like(l_out_raw))

        # Joint pooled complex embedding
        pool_p = self.pocket_pool(pocket_emb_out, p_mask)
        pool_l = self.ligand_pool(ligand_emb_out, l_mask)
        complex_raw: Tensor = self.complex_proj(
            torch.cat((pool_p, pool_l, pool_p * pool_l), dim=-1)
        )
        has_any = (p_mask.any(dim=1) | l_mask.any(dim=1)).unsqueeze(-1)
        complex_pooled = torch.where(has_any, complex_raw, torch.zeros_like(complex_raw))

        return pocket_emb_out, ligand_emb_out, complex_pooled


__all__ = ["SE3ComplexCrossAttention"]
