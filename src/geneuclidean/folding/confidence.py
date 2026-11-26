"""AlphaFold-3 Algorithm 31 Confidence Head & Boltz-2-Style Binding Affinity Head.

Pure-PyTorch ``torch.float32`` structural confidence and affinity readouts:
- :class:`ConfidenceHead`: Predicts calibrated 50-bin per-atom ``pLDDT`` in ``[0, 100]``,
  64-bin asymmetric ``PAE`` and strictly symmetric ``PDE`` in ``[0, 32] Å``, exact
  Zhang–Skolnick ``pTM`` and cross-chain ``ipTM`` in ``[0, 1]``, and AF3 ranking score
  ``0.8 * ipTM + 0.2 * pTM``.
- :class:`AffinityHead`: Boltz-2-inspired cross-chain pocket↔ligand readout predicting
  binary binder logit / probability ``P(binder) in [0, 1]`` and continuous ``pKd``.
- :class:`ConfidenceLoss`: Joint supervised calibration loss for Stage-3 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.folding.pairformer import PairformerBlock, SimpleFoldTrunk
from geneuclidean.models.encoders.dense_knn import safe_masked_softmax


@dataclass(frozen=True)
class ConfidenceOutput:
    """Complete AF3 Algorithm 31 confidence + optional Boltz-2 binding affinity outputs."""

    plddt_logits: Tensor  # [B, N_atoms, 50]
    plddt: Tensor  # [B, N_atoms] in [0.0, 100.0]
    pae_logits: Tensor  # [B, L, L, 64]
    pae: Tensor  # [B, L, L] expected aligned error >= 0 in Angstroms
    pde_logits: Tensor  # [B, L, L, 64]
    pde: Tensor  # [B, L, L] symmetric expected distance error >= 0 in Angstroms
    ptm: Tensor  # [B] in [0.0, 1.0]
    iptm: Tensor  # [B] interface pTM (strictly cross-chain) in [0.0, 1.0]
    ranking_score: Tensor  # [B] = 0.8 * iptm + 0.2 * ptm
    ligand_plddt: Tensor  # [B] mean pLDDT over ligand heavy atoms in [0.0, 100.0]
    predicted_pkd: Tensor | None = None  # [B] continuous pKd prediction
    binding_prob: Tensor | None = None  # [B] P(binder) in [0.0, 1.0]
    binder_logit: Tensor | None = None  # [B] raw binder logit before sigmoid


def compute_tm_scores(
    pae_logits: Tensor,
    token_mask: Tensor,
    chain_id: Tensor,
    *,
    pae_bin_centers: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute exact Zhang–Skolnick ``(pae, ptm, iptm, ranking_score)`` from PAE logits.

    Uses AF3/Boltz-2 small-complex scaling:

    .. math::
        d_0(N) = \\max\\!\\left(1.24\\,(\\max(N, 19) - 15)^{1/3} - 1.8,\\; 0.5\\right), \\qquad
        f(e_k) = \\frac{1}{1 + (e_k / d_0)^2}

    ``ipTM`` strictly averages over cross-chain pairs ``chain_id[i] != chain_id[j]`` so
    intra-chain PAE logits have exact ``0.0`` influence on ``ipTM``.
    """
    if pae_logits.ndim != 4:
        raise ValueError(f"Expected pae_logits [B, L, L, K], got {tuple(pae_logits.shape)}")
    batch_size, seq_len, seq_len_2, num_bins = pae_logits.shape
    if seq_len != seq_len_2:
        raise ValueError(f"Expected square PAE grid [B, L, L, K], got {tuple(pae_logits.shape)}")
    device = pae_logits.device

    if pae_bin_centers is None:
        step = 32.0 / float(num_bins)
        bins = torch.arange(num_bins, device=device, dtype=torch.float32) * step + 0.5 * step
    else:
        bins = pae_bin_centers.to(device=device, dtype=torch.float32)

    m_b = token_mask.to(torch.bool)
    pair_valid = m_b.unsqueeze(2) & m_b.unsqueeze(1)  # [B, L, L]
    probs = torch.softmax(pae_logits.to(torch.float32), dim=-1)  # [B, L, L, K]

    pae_raw = (probs * bins.view(1, 1, 1, num_bins)).sum(dim=-1)
    pae = torch.where(pair_valid, pae_raw, torch.zeros_like(pae_raw)).clamp(min=0.0)

    # Exact Zhang-Skolnick d0(N) clamped >= 0.5 A per AF3 Alg. 31 / Boltz convention.
    n_valid = m_b.sum(dim=-1).to(torch.float32)  # [B]
    d0 = (1.24 * torch.pow(n_valid.clamp(min=19.0) - 15.0, 1.0 / 3.0) - 1.8).clamp(min=0.5)
    f_bins = 1.0 / (1.0 + (bins.view(1, num_bins) / d0.unsqueeze(-1)) ** 2)  # [B, K]

    g_ij = (probs * f_bins.view(batch_size, 1, 1, num_bins)).sum(dim=-1)  # [B, L, L]

    # Global pTM: max over aligned reference tokens i of mean over all valid j.
    ptm_sum = torch.where(pair_valid, g_ij, torch.zeros_like(g_ij)).sum(dim=-1)
    ptm_cnt = pair_valid.sum(dim=-1).clamp(min=1).to(torch.float32)
    ptm_rows = torch.where(m_b, ptm_sum / ptm_cnt, torch.zeros_like(ptm_sum))
    ptm = ptm_rows.max(dim=-1).values.clamp(min=0.0, max=1.0)

    # Interface ipTM: strictly cross-chain pairs (chain_id[i] != chain_id[j]) ONLY!
    cid = chain_id.to(torch.long)
    inter_mask = pair_valid & (cid.unsqueeze(2) != cid.unsqueeze(1))
    inter_cnt = inter_mask.sum(dim=-1)  # [B, L]
    has_inter_row = m_b & (inter_cnt > 0)
    inter_sum = torch.where(inter_mask, g_ij, torch.zeros_like(g_ij)).sum(dim=-1)
    iptm_rows = torch.where(
        has_inter_row,
        inter_sum / inter_cnt.clamp(min=1).to(torch.float32),
        torch.zeros_like(inter_sum),
    )
    iptm = iptm_rows.max(dim=-1).values.clamp(min=0.0, max=1.0)

    ranking_score = (0.8 * iptm + 0.2 * ptm).clamp(min=0.0, max=1.0)
    return pae, ptm, iptm, ranking_score


class AffinityHead(nn.Module):
    """Boltz-2-inspired pocket↔ligand binding affinity & binary binder probability head.

    Pools cross-chain pair features ``z_conf[pocket, ligand]`` via masked attention + mean
    pooling and combines them with pooled token features ``s_conf`` to predict both
    ``binder_logit`` / ``binding_prob in [0, 1]`` and continuous ``predicted_pkd``.
    """

    c_s: int
    c_z: int

    def __init__(self, c_s: int, c_z: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.c_s = int(c_s)
        self.c_z = int(c_z)
        h_dim = int(hidden_dim if hidden_dim is not None else max(64, self.c_z * 2))

        self.z_norm = nn.LayerNorm(self.c_z)
        self.z_score = nn.Sequential(
            nn.Linear(self.c_z, max(16, self.c_z // 2)),
            nn.SiLU(),
            nn.Linear(max(16, self.c_z // 2), 1),
        )
        self.s_norm = nn.LayerNorm(self.c_s)
        self.head = nn.Sequential(
            nn.Linear(2 * self.c_z + self.c_s, h_dim),
            nn.SiLU(),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim // 2),
            nn.SiLU(),
        )
        self.binder_out = nn.Linear(h_dim // 2, 1)
        self.pkd_out = nn.Linear(h_dim // 2, 1)
        # Initialize pKd bias near typical PDBbind drug-like median (~6.5).
        nn.init.constant_(self.pkd_out.bias, 6.5)

    def forward(
        self,
        s_conf: Tensor,
        z_conf: Tensor,
        token_mask: Tensor,
        chain_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(binder_logit [B], binding_prob [B], predicted_pkd [B])``."""
        batch_size, seq_len, _ = s_conf.shape
        m_b = token_mask.to(torch.bool)
        pair_valid = m_b.unsqueeze(2) & m_b.unsqueeze(1)
        cid = chain_id.to(torch.long)
        inter_mask = pair_valid & (cid.unsqueeze(2) != cid.unsqueeze(1))

        # Fall back cleanly to all valid pairs for single-chain inputs.
        has_inter = inter_mask.any(dim=(-2, -1), keepdim=True)
        pool_mask = torch.where(has_inter, inter_mask, pair_valid)  # [B, L, L]
        flat_mask = pool_mask.reshape(batch_size, seq_len * seq_len)

        z_f = self.z_norm(z_conf.to(torch.float32))
        z_flat = z_f.reshape(batch_size, seq_len * seq_len, self.c_z)
        z_flat = torch.where(flat_mask.unsqueeze(-1), z_flat, torch.zeros_like(z_flat))

        logits = self.z_score(z_flat).squeeze(-1)  # [B, L*L]
        attn = safe_masked_softmax(logits, flat_mask, dim=-1)  # [B, L*L]
        z_attn_pool = torch.bmm(attn.unsqueeze(1), z_flat).squeeze(1)  # [B, c_z]
        z_cnt = flat_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(torch.float32)
        z_mean_pool = z_flat.sum(dim=1) / z_cnt  # [B, c_z]

        s_f = self.s_norm(s_conf.to(torch.float32))
        s_f = torch.where(m_b.unsqueeze(-1), s_f, torch.zeros_like(s_f))
        s_cnt = m_b.sum(dim=-1, keepdim=True).clamp(min=1).to(torch.float32)
        s_pool = s_f.sum(dim=1) / s_cnt  # [B, c_s]

        fused = self.head(torch.cat((z_attn_pool, z_mean_pool, s_pool), dim=-1))
        binder_logit = self.binder_out(fused).squeeze(-1)
        binding_prob = torch.sigmoid(binder_logit)
        predicted_pkd = self.pkd_out(fused).squeeze(-1)
        return binder_logit, binding_prob, predicted_pkd


class ConfidenceHead(nn.Module):
    """AlphaFold-3 Algorithm 31 confidence module with optional Boltz-2 :class:`AffinityHead`."""

    c_s: int
    c_z: int
    num_dist_bins: int
    num_plddt_bins: int
    num_pae_bins: int
    trunk_mode: Literal["pairformer", "simplefold", "hybrid"]
    dist_bin_edges: Tensor
    plddt_bin_centers: Tensor
    pae_bin_centers: Tensor
    _dist_bin_step: float

    def __init__(
        self,
        c_s: int,
        c_z: int,
        *,
        num_blocks: int = 2,
        head_dim: int = 32,
        trunk_mode: Literal["pairformer", "simplefold", "hybrid"] = "pairformer",
        num_dist_bins: int = 39,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        enable_affinity_head: bool = True,
        transition_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim < 32:
            raise ValueError(f"ConfidenceHead enforces head_dim >= 32, got {head_dim}")
        self.c_s = int(c_s)
        self.c_z = int(c_z)
        self.num_dist_bins = int(num_dist_bins)
        self.num_plddt_bins = int(num_plddt_bins)
        self.num_pae_bins = int(num_pae_bins)
        self.trunk_mode = trunk_mode

        # 39 distogram bins spanning [3.25 A, 50.75 A] (AF3 Alg. 31)
        self._dist_bin_step = max((50.75 - 3.25) / max(self.num_dist_bins - 1, 1), 1e-3)
        self.register_buffer(
            "dist_bin_edges",
            torch.linspace(3.25, 50.75, self.num_dist_bins, dtype=torch.float32),
        )
        # 50 pLDDT bin centers in (0, 1) -> scaled by 100 to [1.0, 99.0] subset of [0, 100]
        self.register_buffer(
            "plddt_bin_centers",
            (torch.arange(self.num_plddt_bins, dtype=torch.float32) + 0.5)
            / float(self.num_plddt_bins),
        )
        # 64 PAE/PDE bin centers in [0.25, 31.75] A
        step = 32.0 / float(self.num_pae_bins)
        self.register_buffer(
            "pae_bin_centers",
            torch.arange(self.num_pae_bins, dtype=torch.float32) * step + 0.5 * step,
        )

        self.s_in_norm = nn.LayerNorm(self.c_s)
        self.z_in_norm = nn.LayerNorm(self.c_z)
        self.s_left = nn.Linear(self.c_s, self.c_z, bias=False)
        self.s_right = nn.Linear(self.c_s, self.c_z, bias=False)
        self.distogram_proj = nn.Linear(self.num_dist_bins, self.c_z, bias=False)

        self.pf_blocks: nn.ModuleList | None
        self.sf_trunk: SimpleFoldTrunk | None
        if trunk_mode == "simplefold":
            self.pf_blocks = None
            self.sf_trunk = SimpleFoldTrunk(
                c_s=self.c_s,
                c_z=self.c_z,
                num_layers=max(1, num_blocks),
                head_dim=head_dim,
                trunk_mode="simplefold",
                use_coulomb_bias=False,
                transition_expansion=transition_expansion,
                dropout=dropout,
            )
        else:
            self.pf_blocks = nn.ModuleList(
                [
                    PairformerBlock(
                        self.c_s,
                        self.c_z,
                        head_dim=head_dim,
                        transition_expansion=transition_expansion,
                        dropout=dropout,
                    )
                    for _ in range(max(1, num_blocks))
                ]
            )
            self.sf_trunk = None

        self.s_out_norm = nn.LayerNorm(self.c_s)
        self.z_out_norm = nn.LayerNorm(self.c_z)
        self.plddt_head = nn.Linear(self.c_s, self.num_plddt_bins)
        self.pae_head = nn.Linear(self.c_z, self.num_pae_bins)
        self.pde_head = nn.Linear(self.c_z, self.num_pae_bins)

        self.affinity_head: AffinityHead | None = (
            AffinityHead(self.c_s, self.c_z) if enable_affinity_head else None
        )

    def _embed_distogram(self, x_rep: Tensor, pair_mask: Tensor) -> Tensor:
        """Soft-bin representative token pairwise distances into 39-bin distogram."""
        diff = x_rep.unsqueeze(2) - x_rep.unsqueeze(1)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)  # [B, L, L]
        edges = self.dist_bin_edges.to(device=x_rep.device, dtype=torch.float32)
        # Smooth Gaussian binning with precomputed step (zero GPU->CPU .item() sync).
        delta = (dist.unsqueeze(-1) - edges.view(1, 1, 1, self.num_dist_bins)) / self._dist_bin_step
        bins = torch.softmax(-0.5 * (delta * delta), dim=-1)
        return torch.where(pair_mask.unsqueeze(-1), bins, torch.zeros_like(bins))

    def forward(
        self,
        s_trunk: Tensor,
        z_trunk: Tensor,
        x_rep: Tensor,
        token_mask: Tensor,
        chain_id: Tensor,
        *,
        atom_to_token: Tensor | None = None,
        atom_mask: Tensor | None = None,
        is_ligand_atom: Tensor | None = None,
        detach_trunk: bool = True,
    ) -> ConfidenceOutput:
        """Predict per-atom ``pLDDT``, ``PAE``, ``PDE``, ``pTM``, ``ipTM``, and optional ``pKd``."""
        batch_size, seq_len, _ = s_trunk.shape
        m_tok = token_mask.to(torch.bool)
        pair_mask = m_tok.unsqueeze(2) & m_tok.unsqueeze(1)

        s_in = s_trunk.detach() if detach_trunk else s_trunk
        z_in = z_trunk.detach() if detach_trunk else z_trunk
        x_in = x_rep.detach() if detach_trunk else x_rep

        s_c = self.s_in_norm(s_in.to(torch.float32))
        z_c = self.z_in_norm(z_in.to(torch.float32))
        dist_bins = self._embed_distogram(x_in.to(torch.float32), pair_mask)
        z_c = (
            z_c
            + self.s_left(s_c).unsqueeze(2)
            + self.s_right(s_c).unsqueeze(1)
            + self.distogram_proj(dist_bins)
        )
        z_c = torch.where(pair_mask.unsqueeze(-1), z_c, torch.zeros_like(z_c))
        s_c = torch.where(m_tok.unsqueeze(-1), s_c, torch.zeros_like(s_c))

        if self.sf_trunk is not None:
            s_c, z_c = self.sf_trunk(s_c, z_c, m_tok)
        else:
            assert self.pf_blocks is not None
            for block in self.pf_blocks:
                s_c, z_c = block(s_c, z_c, m_tok)

        s_final = self.s_out_norm(s_c)
        z_final = self.z_out_norm(z_c)

        # Token-level pLDDT logits broadcast to per-atom resolution when atom_to_token is given.
        tok_plddt_logits = self.plddt_head(s_final)  # [B, L, 50]
        if atom_to_token is not None:
            safe_idx = atom_to_token.to(torch.long).clamp(min=0, max=max(seq_len - 1, 0))
            idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, self.num_plddt_bins)
            plddt_logits = torch.gather(tok_plddt_logits, dim=1, index=idx_exp)  # [B, N, 50]
            a_mask = (
                atom_mask.to(torch.bool)
                if atom_mask is not None
                else torch.ones(
                    (batch_size, atom_to_token.shape[1]), device=s_trunk.device, dtype=torch.bool
                )
            )
        else:
            plddt_logits = tok_plddt_logits
            a_mask = m_tok

        plddt_probs = torch.softmax(plddt_logits, dim=-1)
        plddt_centers = self.plddt_bin_centers.to(device=s_trunk.device, dtype=torch.float32)
        plddt_raw = 100.0 * (plddt_probs * plddt_centers.view(1, 1, self.num_plddt_bins)).sum(
            dim=-1
        )
        plddt = torch.where(a_mask, plddt_raw, torch.zeros_like(plddt_raw)).clamp(
            min=0.0, max=100.0
        )

        # Asymmetric PAE logits & Zhang-Skolnick pTM / ipTM.
        pae_logits = self.pae_head(z_final)  # [B, L, L, 64]
        pae_bins = self.pae_bin_centers.to(device=s_trunk.device, dtype=torch.float32)
        pae, ptm, iptm, ranking_score = compute_tm_scores(
            pae_logits, m_tok, chain_id, pae_bin_centers=pae_bins
        )

        # Strictly symmetric PDE: symmetrize pair features BEFORE linear head.
        z_sym = 0.5 * (z_final + z_final.transpose(1, 2))
        pde_logits = self.pde_head(z_sym)  # [B, L, L, 64]
        pde_probs = torch.softmax(pde_logits, dim=-1)
        pde_raw = (pde_probs * pae_bins.view(1, 1, 1, self.num_pae_bins)).sum(dim=-1)
        pde = torch.where(pair_mask, pde_raw, torch.zeros_like(pde_raw)).clamp(min=0.0)

        # Mean ligand pLDDT (or overall valid-atom pLDDT if no ligand atoms exist).
        if is_ligand_atom is not None:
            lig_m = a_mask & is_ligand_atom.to(torch.bool)
        elif atom_to_token is not None:
            safe_idx = atom_to_token.to(torch.long).clamp(min=0, max=max(seq_len - 1, 0))
            atom_chain = torch.gather(chain_id.to(torch.long), dim=1, index=safe_idx)
            lig_m = a_mask & (atom_chain > 0)
        else:
            lig_m = m_tok & (chain_id.to(torch.long) > 0)

        has_lig = lig_m.any(dim=-1, keepdim=True)
        score_mask = torch.where(has_lig, lig_m, a_mask)
        lig_sum = torch.where(score_mask, plddt, torch.zeros_like(plddt)).sum(dim=-1)
        lig_cnt = score_mask.sum(dim=-1).clamp(min=1).to(torch.float32)
        ligand_plddt = (lig_sum / lig_cnt).clamp(min=0.0, max=100.0)

        binder_logit: Tensor | None = None
        binding_prob: Tensor | None = None
        predicted_pkd: Tensor | None = None
        if self.affinity_head is not None:
            binder_logit, binding_prob, predicted_pkd = self.affinity_head(
                s_final, z_final, m_tok, chain_id
            )

        return ConfidenceOutput(
            plddt_logits=plddt_logits,
            plddt=plddt,
            pae_logits=pae_logits,
            pae=pae,
            pde_logits=pde_logits,
            pde=pde,
            ptm=ptm,
            iptm=iptm,
            ranking_score=ranking_score,
            ligand_plddt=ligand_plddt,
            predicted_pkd=predicted_pkd,
            binding_prob=binding_prob,
            binder_logit=binder_logit,
        )


class ConfidenceLoss(nn.Module):
    """Stage-3 supervised calibration loss for :class:`ConfidenceHead` & :class:`AffinityHead`."""

    def __init__(
        self,
        *,
        w_plddt: float = 1.0,
        w_pde: float = 1.0,
        w_pae: float = 0.5,
        w_pkd: float = 0.5,
        w_binder: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_plddt = float(w_plddt)
        self.w_pde = float(w_pde)
        self.w_pae = float(w_pae)
        self.w_pkd = float(w_pkd)
        self.w_binder = float(w_binder)

    def forward(
        self,
        conf: ConfidenceOutput,
        x_pred_atoms: Tensor,
        x_true_atoms: Tensor,
        atom_mask: Tensor,
        x_pred_rep: Tensor,
        x_true_rep: Tensor,
        token_mask: Tensor,
        *,
        pkd_true: Tensor | None = None,
        is_binder_true: Tensor | None = None,
    ) -> Tensor:
        """Compute joint cross-entropy (pLDDT, PDE, PAE) + affinity calibration loss."""
        m_a = atom_mask.to(torch.bool)
        m_t = token_mask.to(torch.bool)
        zero = conf.plddt.new_zeros(())

        # 1. Per-atom ground-truth LDDT target discretized into 50 bins.
        diff_p = x_pred_atoms.detach().unsqueeze(2) - x_pred_atoms.detach().unsqueeze(1)
        diff_t = x_true_atoms.detach().unsqueeze(2) - x_true_atoms.detach().unsqueeze(1)
        d_p = torch.sqrt((diff_p * diff_p).sum(dim=-1) + 1e-8)
        d_t = torch.sqrt((diff_t * diff_t).sum(dim=-1) + 1e-8)
        eye_a = torch.eye(x_pred_atoms.shape[1], device=m_a.device, dtype=torch.bool).unsqueeze(0)
        incl_a = m_a.unsqueeze(2) & m_a.unsqueeze(1) & (~eye_a) & (d_t < 15.0)
        abs_e = torch.abs(d_p - d_t)
        lddt_hits = (
            (abs_e < 0.5).to(torch.float32)
            + (abs_e < 1.0).to(torch.float32)
            + (abs_e < 2.0).to(torch.float32)
            + (abs_e < 4.0).to(torch.float32)
        ) * 0.25
        lddt_atom = torch.where(incl_a, lddt_hits, torch.zeros_like(lddt_hits)).sum(
            dim=-1
        ) / incl_a.to(torch.float32).sum(dim=-1).clamp(min=1.0)
        n_plddt_bins = conf.plddt_logits.shape[-1]
        plddt_tgt = (lddt_atom * float(n_plddt_bins)).long().clamp(0, n_plddt_bins - 1)
        plddt_ce = F.cross_entropy(
            conf.plddt_logits.reshape(-1, n_plddt_bins),
            plddt_tgt.reshape(-1),
            reduction="none",
        ).reshape_as(plddt_tgt)
        plddt_loss = torch.where(
            m_a.any(),
            torch.where(m_a, plddt_ce, torch.zeros_like(plddt_ce)).sum()
            / m_a.to(torch.float32).sum().clamp(min=1.0),
            zero,
        )

        # 2. Token-pair PDE & PAE distance-error target discretized into 64 bins over [0, 32 A].
        pair_t = m_t.unsqueeze(2) & m_t.unsqueeze(1)
        rep_dp = torch.sqrt(
            ((x_pred_rep.detach().unsqueeze(2) - x_pred_rep.detach().unsqueeze(1)) ** 2).sum(dim=-1)
            + 1e-8
        )
        rep_dt = torch.sqrt(
            ((x_true_rep.detach().unsqueeze(2) - x_true_rep.detach().unsqueeze(1)) ** 2).sum(dim=-1)
            + 1e-8
        )
        dist_err = torch.abs(rep_dp - rep_dt)
        n_pae_bins = conf.pde_logits.shape[-1]
        err_tgt = (dist_err / (32.0 / float(n_pae_bins))).long().clamp(0, n_pae_bins - 1)

        pde_ce = F.cross_entropy(
            conf.pde_logits.reshape(-1, n_pae_bins),
            err_tgt.reshape(-1),
            reduction="none",
        ).reshape_as(err_tgt)
        pae_ce = F.cross_entropy(
            conf.pae_logits.reshape(-1, n_pae_bins),
            err_tgt.reshape(-1),
            reduction="none",
        ).reshape_as(err_tgt)
        pair_denom = pair_t.to(torch.float32).sum().clamp(min=1.0)
        pde_loss = torch.where(
            pair_t.any(),
            torch.where(pair_t, pde_ce, torch.zeros_like(pde_ce)).sum() / pair_denom,
            zero,
        )
        pae_loss = torch.where(
            pair_t.any(),
            torch.where(pair_t, pae_ce, torch.zeros_like(pae_ce)).sum() / pair_denom,
            zero,
        )

        total = self.w_plddt * plddt_loss + self.w_pde * pde_loss + self.w_pae * pae_loss

        if pkd_true is not None and conf.predicted_pkd is not None:
            valid_pkd = torch.isfinite(pkd_true)
            if valid_pkd.any():
                safe_t = torch.where(
                    valid_pkd, pkd_true.to(torch.float32), conf.predicted_pkd.detach()
                )
                pkd_l = F.huber_loss(conf.predicted_pkd, safe_t, reduction="none")
                total = total + self.w_pkd * (
                    torch.where(valid_pkd, pkd_l, torch.zeros_like(pkd_l)).sum()
                    / valid_pkd.to(torch.float32).sum().clamp(min=1.0)
                )

        if is_binder_true is not None and conf.binder_logit is not None:
            total = total + self.w_binder * F.binary_cross_entropy_with_logits(
                conf.binder_logit, is_binder_true.to(torch.float32)
            )
        return total


__all__ = [
    "AffinityHead",
    "ConfidenceHead",
    "ConfidenceLoss",
    "ConfidenceOutput",
    "compute_tm_scores",
]
