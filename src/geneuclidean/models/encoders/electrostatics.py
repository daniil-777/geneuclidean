"""SE(3)-invariant long-range electrostatics and GeoTransformer/RoITr geometric attention biases.

Implements two composable, pure-PyTorch ``float32`` attention-bias modules designed for Apple
Silicon MPS, CUDA, and CPU (zero ``float64``, zero ``index_add_``):

1. :class:`ScreenedCoulombBias` (**LES** — Latent Ewald Summation lineage, Cheng *npj Comput.
   Mater.* 2025): Predicts per-atom residual latent charges initialized from the physics-grounded
   Amber/Gasteiger partial charge prior at ``node_feats[..., 46]`` and emits head-wise screened
   Debye-Huckel :math:`1/r` electrostatic biases:

   .. math::
       \\tilde{q}_i = q_{i,\\mathrm{prior}} + \\tanh(\\mathrm{MLP}_q(\\mathbf{s}_i)), \\qquad
       b_{ij}^{\\mathrm{elec},(h)} = w_h \\,
       \\frac{\\tilde{q}_i \\tilde{q}_j}{\\sqrt{d_{ij}^2 + r_0^2}}
       \\exp\\!\\left(-\\frac{d_{ij}}{\\lambda_D}\\right).

2. :class:`GeoTransformerPairTripletBias` (**GeoTransformer** TPAMI 2024 + **RoITr** CVPR 2023
   lineage): Emits exact rigid-invariant pairwise distance RBFs, max-pooled triplet-angle
   encodings :math:`\\cos\\angle(i, k, j)` (regularized with :math:`\\delta = 0.5\\,\\text{\\AA}`
   so coincident epoch-0 centroids never divide by zero), and intrinsic Point Pair Features (PPF)
   with optional signed :math:`SE(3)` chiral dihedral/axial terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from geneuclidean.models.encoders.dense_knn import center_masked_coords_with_offset
from geneuclidean.models.encoders.rbf import BesselRBF, CosineCutoff
from geneuclidean.models.encoders.tensor_ops import chiral_cross_product


def _shared_frame_coords(
    coords_i: Tensor,
    mask_i: Tensor,
    coords_j: Tensor | None,
    mask_j: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
    """Shift ``coords_i`` (and optional ``coords_j``) into ``coords_i``'s centroid frame.

    Eliminates ``float32`` cancellation under large rigid translations (e.g. ``100 A``) while
    strictly preserving inter-molecular displacement ``coords_j - coords_i`` for cross-attention.
    """
    is_self = coords_j is None
    centered_i, offset_i = center_masked_coords_with_offset(coords_i, mask_i)
    if is_self:
        return centered_i, mask_i, centered_i, mask_i, True

    assert coords_j is not None
    if coords_j.ndim != 3 or coords_j.shape[-1] != 3:
        raise ValueError(f"Expected coords_j shape [B, M, 3], got {tuple(coords_j.shape)}")
    if coords_j.shape[0] != coords_i.shape[0]:
        raise ValueError(
            f"Batch size mismatch: coords_i {coords_i.shape[0]} vs coords_j {coords_j.shape[0]}"
        )
    if mask_j is None:
        mask_j_b = torch.ones(coords_j.shape[:2], device=coords_j.device, dtype=torch.bool)
    else:
        if mask_j.shape != coords_j.shape[:2]:
            raise ValueError(
                f"mask_j shape {tuple(mask_j.shape)} != coords_j[:2] {tuple(coords_j.shape[:2])}"
            )
        mask_j_b = mask_j.to(torch.bool)

    coords_j_f = coords_j.to(torch.float32)
    centered_j = torch.where(
        mask_j_b.unsqueeze(-1),
        coords_j_f - offset_i.unsqueeze(1),
        torch.zeros_like(coords_j_f),
    )
    return centered_i, mask_i, centered_j, mask_j_b, False


def _reduce_smooth_unit_vectors(vecs: Tensor) -> Tensor:
    """Reduce ``[B, M, 3]`` or ``[B, M, C, 3]`` vector features to bounded ``[B, M, 3]`` axes."""
    vecs_f = vecs.to(torch.float32)
    if vecs_f.ndim == 4 and vecs_f.shape[-1] == 3:
        reduced = vecs_f.mean(dim=-2)
    elif vecs_f.ndim == 3 and vecs_f.shape[-1] == 3:
        reduced = vecs_f
    else:
        raise ValueError(
            f"Expected vector tensor of shape [B, M, 3] or [B, M, C, 3], got {tuple(vecs.shape)}"
        )
    sq_norm = (reduced * reduced).sum(dim=-1, keepdim=True)
    return reduced * torch.rsqrt(sq_norm + 1.0)


class ScreenedCoulombBias(nn.Module):
    """Long-range screened ``1/r`` latent-charge electrostatic attention bias (``LES`` lineage).

    Captures 6-15 Angstrom salt bridges (e.g. ``Asp3.32``), catalytic ``Zn2+/Mg2+`` metal fields,
    and cofactor dipoles at negligible Apple Silicon MPS latency (``< 0.5 ms``). Supports both
    intra-structure self-attention and inter-molecular pocket-ligand cross-attention, and accepts
    either raw scalar embeddings (to predict latent charges on the fly) or pre-pooled charges
    (e.g. from virtual landmarks or competitive sub-pocket slots).
    """

    scalar_dim: int
    num_heads: int
    r0: float
    debye_len: float
    charge_prior_idx: int

    def __init__(
        self,
        scalar_dim: int,
        num_heads: int,
        *,
        r0: float = 1.0,
        debye_len: float = 8.0,
        charge_prior_idx: int = 46,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if scalar_dim < 1:
            raise ValueError(f"scalar_dim must be >= 1, got {scalar_dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if r0 <= 0.0:
            raise ValueError(f"r0 must be positive, got {r0}")
        if debye_len <= 0.0:
            raise ValueError(f"debye_len must be positive, got {debye_len}")

        self.scalar_dim = int(scalar_dim)
        self.num_heads = int(num_heads)
        self.r0 = float(r0)
        self.debye_len = float(debye_len)
        self.charge_prior_idx = int(charge_prior_idx)

        mid_dim = int(hidden_dim) if hidden_dim is not None else max(16, self.scalar_dim // 4)
        self.charge_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, 1),
        )
        self.head_weights = nn.Parameter(torch.full((self.num_heads,), 0.25, dtype=torch.float32))

    def predict_charges(
        self,
        scalar_feats: Tensor,
        raw_node_feats: Tensor | None = None,
        atom_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict residual latent charges ``q_tilde [B, N]`` anchored to ``dim 46`` prior."""
        raw_delta: Tensor = self.charge_mlp(scalar_feats).squeeze(-1)
        delta_q = torch.tanh(raw_delta)
        if raw_node_feats is not None and raw_node_feats.shape[-1] > self.charge_prior_idx:
            q_prior = (
                raw_node_feats[..., self.charge_prior_idx]
                .to(dtype=delta_q.dtype)
                .clamp(min=-1.0, max=1.0)
            )
            q_tilde = q_prior + delta_q
        else:
            q_tilde = delta_q

        if atom_mask is not None:
            mask_b = atom_mask.to(torch.bool)
            q_tilde = torch.where(mask_b, q_tilde, torch.zeros_like(q_tilde))
        return q_tilde

    def forward(
        self,
        coords_i: Tensor,
        scalar_i: Tensor | None = None,
        *,
        raw_feats_i: Tensor | None = None,
        mask_i: Tensor | None = None,
        coords_j: Tensor | None = None,
        scalar_j: Tensor | None = None,
        raw_feats_j: Tensor | None = None,
        mask_j: Tensor | None = None,
        charges_i: Tensor | None = None,
        charges_j: Tensor | None = None,
    ) -> Tensor:
        """Compute head-wise screened Coulomb attention bias ``[B, num_heads, N_i, N_j]``."""
        if coords_i.ndim != 3 or coords_i.shape[-1] != 3:
            raise ValueError(f"Expected coords_i shape [B, N, 3], got {tuple(coords_i.shape)}")
        batch_size, n_i, _ = coords_i.shape
        if mask_i is None:
            mask_i_b = torch.ones((batch_size, n_i), device=coords_i.device, dtype=torch.bool)
        else:
            if mask_i.shape != (batch_size, n_i):
                raise ValueError(
                    f"mask_i shape {tuple(mask_i.shape)} != coords_i[:2] {(batch_size, n_i)}"
                )
            mask_i_b = mask_i.to(torch.bool)

        ci, mi, cj, mj, is_self = _shared_frame_coords(coords_i, mask_i_b, coords_j, mask_j)

        if charges_i is not None:
            q_i_f = charges_i.to(torch.float32)
            q_i = torch.where(mi, q_i_f, torch.zeros_like(q_i_f))
        elif scalar_i is not None:
            q_i = self.predict_charges(scalar_i, raw_feats_i, mi)
        else:
            raise ValueError("Provide either scalar_i or charges_i to ScreenedCoulombBias.")

        if is_self:
            q_j = q_i
        elif charges_j is not None:
            q_j_f = charges_j.to(torch.float32)
            q_j = torch.where(mj, q_j_f, torch.zeros_like(q_j_f))
        elif scalar_j is not None:
            q_j = self.predict_charges(scalar_j, raw_feats_j, mj)
        else:
            raise ValueError(
                "Cross-attention requires either scalar_j or charges_j when coords_j is provided."
            )

        diff = ci.unsqueeze(2) - cj.unsqueeze(1)  # [B, N_i, N_j, 3]
        sq_d = (diff * diff).sum(dim=-1)  # [B, N_i, N_j]
        dist = torch.sqrt(sq_d + 1e-8)
        inv_soft_r = torch.rsqrt(sq_d + self.r0 * self.r0)
        screening = torch.exp(-dist / self.debye_len)
        phi = (q_i.unsqueeze(2) * q_j.unsqueeze(1)) * inv_soft_r * screening

        pair_mask = mi.unsqueeze(2) & mj.unsqueeze(1)
        if is_self:
            eye = torch.eye(n_i, device=coords_i.device, dtype=torch.bool).unsqueeze(0)
            pair_mask = pair_mask & (~eye)

        phi = torch.where(pair_mask, phi, torch.zeros_like(phi))
        return phi.unsqueeze(1) * self.head_weights.view(1, self.num_heads, 1, 1)


class GeoTransformerPairTripletBias(nn.Module):
    """Exact ``SE(3)``-invariant pairwise RBF + GeoTransformer triplet angle + RoITr PPF bias.

    Combines three complementary rigid-invariant geometric encodings for superpoint, landmark,
    sub-pocket, or complex-level attention (``[B, num_heads, M_i, M_j]``):
      1. **Pairwise Distance Bessel RBF** with smooth :class:`CosineCutoff`.
      2. **GeoTransformer Triplet-Angle Bias** (Qin et al., TPAMI 2024): Max-pooled smooth
         angular basis over :math:`\\cos\\angle(i, k, j)` using regularized displacements
         :math:`\\tilde{\\mathbf{u}}_{ik} = \\Delta\\boldsymbol{\\mu}_{ik} /`
         :math:`\\sqrt{\\|\\Delta\\boldsymbol{\\mu}_{ik}\\|^2 + \\delta^2}`
         (:math:`\\delta = 0.5\\,\\text{\\AA}` prevents epoch-0 coincident-centroid singularities).
      3. **RoITr Intrinsic Point Pair Features (PPF) + Signed Chirality** (Yu et al., CVPR 2023):
         When equivariant polar ``vecs`` (and optional axial ``axial``) features are supplied,
         computes ``[dot(n_i, d_ij), dot(n_j, d_ij), dot(n_i, n_j)]`` plus optional signed
         chiral terms ``dot(n_i x n_j, d_ij)`` and ``dot(omega_i - omega_j, d_ij)`` that flip
         sign under improper mirror reflections (``det(P) = -1``) when ``use_se3_chirality=True``
         and vanish identically on planar achiral geometries.
    """

    num_heads: int
    num_rbf: int
    num_angle_bins: int
    cutoff: float
    smooth_delta: float
    use_triplet_angles: bool
    use_ppf: bool
    use_se3_chirality: bool
    angle_centers: Tensor
    angle_coeff: Tensor

    def __init__(
        self,
        num_heads: int,
        *,
        num_rbf: int = 16,
        num_angle_bins: int = 8,
        cutoff: float = 20.0,
        smooth_delta: float = 0.5,
        use_triplet_angles: bool = True,
        use_ppf: bool = True,
        use_se3_chirality: bool = True,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if num_angle_bins < 2:
            raise ValueError(f"num_angle_bins must be >= 2, got {num_angle_bins}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if smooth_delta <= 0.0:
            raise ValueError(f"smooth_delta must be positive, got {smooth_delta}")

        self.num_heads = int(num_heads)
        self.num_rbf = int(num_rbf)
        self.num_angle_bins = int(num_angle_bins)
        self.cutoff = float(cutoff)
        self.smooth_delta = float(smooth_delta)
        self.use_triplet_angles = bool(use_triplet_angles)
        self.use_ppf = bool(use_ppf)
        self.use_se3_chirality = bool(use_se3_chirality)

        self.rbf = BesselRBF(num_rbf=self.num_rbf, cutoff=self.cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=self.cutoff)
        self.dist_proj = nn.Linear(self.num_rbf, self.num_heads)

        centers = torch.linspace(-1.0, 1.0, self.num_angle_bins, dtype=torch.float32)
        step = 2.0 / max(self.num_angle_bins - 1, 1)
        self.register_buffer("angle_centers", centers)
        self.register_buffer(
            "angle_coeff",
            torch.tensor(-0.5 / (step * step), dtype=torch.float32),
        )

        self.triplet_proj: nn.Linear | None
        if self.use_triplet_angles:
            self.triplet_proj = nn.Linear(self.num_angle_bins, self.num_heads)
        else:
            self.triplet_proj = None

        self.ppf_mlp: nn.Sequential | None
        if self.use_ppf:
            self.ppf_mlp = nn.Sequential(
                nn.Linear(5, int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), self.num_heads),
            )
        else:
            self.ppf_mlp = None

    def forward(
        self,
        coords_i: Tensor,
        *,
        mask_i: Tensor | None = None,
        vecs_i: Tensor | None = None,
        axial_i: Tensor | None = None,
        coords_j: Tensor | None = None,
        mask_j: Tensor | None = None,
        vecs_j: Tensor | None = None,
        axial_j: Tensor | None = None,
    ) -> Tensor:
        """Compute pairwise distance + triplet-angle + PPF bias ``[B, num_heads, M_i, M_j]``."""
        if coords_i.ndim != 3 or coords_i.shape[-1] != 3:
            raise ValueError(f"Expected coords_i shape [B, M, 3], got {tuple(coords_i.shape)}")
        batch_size, m_i, _ = coords_i.shape
        device = coords_i.device

        if mask_i is None:
            mask_i_b = torch.ones((batch_size, m_i), device=device, dtype=torch.bool)
        else:
            if mask_i.shape != (batch_size, m_i):
                raise ValueError(
                    f"mask_i shape {tuple(mask_i.shape)} != coords_i[:2] {(batch_size, m_i)}"
                )
            mask_i_b = mask_i.to(torch.bool)

        ci, mi, cj, mj, is_self = _shared_frame_coords(coords_i, mask_i_b, coords_j, mask_j)
        m_j = cj.shape[1]
        pair_mask = mi.unsqueeze(2) & mj.unsqueeze(1)  # [B, M_i, M_j]

        if batch_size == 0 or m_i == 0 or m_j == 0:
            return torch.empty((batch_size, self.num_heads, m_i, m_j), device=device)

        # 1. Smooth pairwise Bessel RBF distance bias
        diff_ij = ci.unsqueeze(2) - cj.unsqueeze(1)  # [B, M_i, M_j, 3]
        sq_d_ij = (diff_ij * diff_ij).sum(dim=-1)
        dist_ij = torch.sqrt(sq_d_ij + 1e-8)
        rbf_env = self.rbf(dist_ij) * self.cutoff_fn(dist_ij).unsqueeze(-1)
        bias_bh: Tensor = self.dist_proj(rbf_env)  # [B, M_i, M_j, H]

        delta_sq = self.smooth_delta * self.smooth_delta

        # 2. GeoTransformer triplet-angle bias: max_{k != i, j} W_angle psi(cos angle(i, k, j))
        if self.use_triplet_angles and self.triplet_proj is not None and m_j >= 2:
            # Anchor vertices k live in cj [B, M_j, 3]
            diff_ik = ci.unsqueeze(2) - cj.unsqueeze(1)  # [B, M_i, M_k, 3]
            u_ik = diff_ik * torch.rsqrt((diff_ik * diff_ik).sum(dim=-1, keepdim=True) + delta_sq)

            diff_jk = cj.unsqueeze(2) - cj.unsqueeze(1)  # [B, M_j, M_k, 3]
            u_jk = diff_jk * torch.rsqrt((diff_jk * diff_jk).sum(dim=-1, keepdim=True) + delta_sq)

            cos_ijk = torch.einsum("bikd,bjkd->bijk", u_ik, u_jk).clamp(min=-1.0, max=1.0)
            ang_diff = cos_ijk.unsqueeze(-1) - self.angle_centers
            ang_basis = torch.exp(self.angle_coeff * (ang_diff * ang_diff))
            triplet_h: Tensor = self.triplet_proj(ang_basis)  # [B, M_i, M_j, M_k, H]

            # Valid third vertex k: valid atom, k != j, and (if self-attn) k != i.
            k_valid = mj.unsqueeze(1).unsqueeze(2).expand(batch_size, m_i, m_j, m_j)
            eye_jk = torch.eye(m_j, device=device, dtype=torch.bool).view(1, 1, m_j, m_j)
            k_valid = k_valid & (~eye_jk)
            if is_self:
                eye_ik = torch.eye(m_i, device=device, dtype=torch.bool).view(1, m_i, 1, m_j)
                k_valid = k_valid & (~eye_ik)
            k_valid_exp = k_valid.unsqueeze(-1)  # [B, M_i, M_j, M_k, 1]
            has_any_k = k_valid_exp.any(dim=-2, keepdim=True)

            neg_fill = torch.full_like(triplet_h, -1e9)
            safe_trip = torch.where(k_valid_exp, triplet_h, neg_fill)
            safe_trip = torch.where(has_any_k, safe_trip, torch.zeros_like(safe_trip))
            triplet_pooled = safe_trip.max(dim=-2).values
            triplet_pooled = torch.where(
                has_any_k.squeeze(-2), triplet_pooled, torch.zeros_like(triplet_pooled)
            )
            bias_bh = bias_bh + triplet_pooled

        # 3. RoITr intrinsic Point Pair Features (PPF) + optional signed SE(3) chirality
        if self.use_ppf and self.ppf_mlp is not None and vecs_i is not None:
            v_j_src = vecs_i if (is_self and vecs_j is None) else vecs_j
            if v_j_src is not None:
                n_i = _reduce_smooth_unit_vectors(vecs_i)  # [B, M_i, 3]
                n_j = _reduce_smooth_unit_vectors(v_j_src)  # [B, M_j, 3]
                n_i = torch.where(mi.unsqueeze(-1), n_i, torch.zeros_like(n_i))
                n_j = torch.where(mj.unsqueeze(-1), n_j, torch.zeros_like(n_j))

                d_unit = diff_ij * torch.rsqrt(sq_d_ij.unsqueeze(-1) + delta_sq)  # [B, M_i, M_j, 3]
                p1 = (n_i.unsqueeze(2) * d_unit).sum(dim=-1)
                p2 = (n_j.unsqueeze(1) * d_unit).sum(dim=-1)
                p3 = (n_i.unsqueeze(2) * n_j.unsqueeze(1)).sum(dim=-1)

                if self.use_se3_chirality:
                    cross_ij = chiral_cross_product(
                        n_i.unsqueeze(2).expand(batch_size, m_i, m_j, 3),
                        n_j.unsqueeze(1).expand(batch_size, m_i, m_j, 3),
                        normalize=True,
                        eps=1.0,
                    )
                    p4 = (cross_ij * d_unit).sum(dim=-1)

                    ax_j_src = axial_i if (is_self and axial_j is None) else axial_j
                    if axial_i is not None and ax_j_src is not None:
                        w_i = _reduce_smooth_unit_vectors(axial_i)
                        w_j = _reduce_smooth_unit_vectors(ax_j_src)
                        w_i = torch.where(mi.unsqueeze(-1), w_i, torch.zeros_like(w_i))
                        w_j = torch.where(mj.unsqueeze(-1), w_j, torch.zeros_like(w_j))
                        p5 = ((w_i.unsqueeze(2) - w_j.unsqueeze(1)) * d_unit).sum(dim=-1)
                    else:
                        p5 = torch.zeros_like(p1)
                else:
                    p4 = torch.zeros_like(p1)
                    p5 = torch.zeros_like(p1)

                ppf_feats = torch.stack((p1, p2, p3, p4, p5), dim=-1)
                ppf_out: Tensor = self.ppf_mlp(ppf_feats)
                bias_bh = bias_bh + ppf_out

        bias_bh = torch.where(pair_mask.unsqueeze(-1), bias_bh, torch.zeros_like(bias_bh))
        return bias_bh.permute(0, 3, 1, 2).contiguous()


__all__ = [
    "GeoTransformerPairTripletBias",
    "ScreenedCoulombBias",
]
