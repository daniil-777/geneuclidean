"""Differentiable Anisotropic 3D/4D Gaussian Field Algebra for SE(3) Molecular Modeling.

Pure-PyTorch ``float32`` mathematical primitives powering ``AtomGS-SE3`` and external 3D
Gaussian molecular field architectures on Apple Silicon MPS, CUDA, and CPU — with **zero**
calls to ``torch.linalg.inv``, ``torch.linalg.det``, ``torch.linalg.slogdet``, or ``float64``.

Key mathematical & numerical guarantees (see ``docs/pointcloud_cv_3d_atom_encoding_plan.md`` §4.2):
  1. **Unconditional SPD Conditioning (:math:`\\kappa \\le 29.8 < 30`):**
     :func:`build_anisotropic_covariance_se3` smoothly bounds isotropic floor
     :math:`s_{i,0} \\in [0.25, 4.0]\\,\\text{\\AA}^2` and polar/axial rank-1 weights
     :math:`\\alpha_{i,c}, \\beta_{i,a} \\in (0, 1.2)\\,\\text{\\AA}^2` via scaled sigmoids,
     bounding condition numbers to :math:`\\kappa(\\mathbf{\\Sigma}_i) \\le 29.8` for default
     ``(C_g=4, C_a=2)`` while enabling out-of-plane :math:`\\pi`-cloud normal anisotropy
     (:math:`\\boldsymbol{\\omega}\\boldsymbol{\\omega}^\\top`) on strictly planar aromatic rings
     with exact :math:`O(3)` equivariance.
  2. **Exact Closed-Form 3x3 Cofactor Inverse (< 3e-6 fp32 err on MPS):**
     :func:`cofactor_inverse_and_logdet_3x3` evaluates the analytical adjugate and determinant
     in pure elementwise Metal/SIMD operations, avoiding MPS LAPACK instabilities.
  3. **Shared-Frame Analytical 3D Overlaps & OpenEye ROCS Shape Tanimoto:**
     :class:`AtomicGaussianField` stores ``centroid_offset [B, 3]`` so inter-molecular
     displacements ``d_ij = mu_i^A - (mu_j^B + (c_B - c_A))``
     cancel shared ``100 A`` rigid translations without teleporting separated complexes together.
  4. **GaussianFormer-2 Probabilistic Occupancy, 4DGS Ensemble Fusion & Super-Ellipsoid Pooling:**
     Exact volume-preserving probabilistic superposition :math:`1 - \\prod (1 - p_{ij})`,
     exact Law-of-Total-Covariance 4D MD fusion, and trace-stabilized super-ellipsoid pooling
     (``+ 0.05 * tr(Sigma_mix) * I_3``, :math:`\\kappa \\le 20`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from geneuclidean.models.encoders.dense_knn import build_dense_knn_graph, gather_neighbors

_LOG_2PI: float = math.log(2.0 * math.pi)
_LOG_2: float = math.log(2.0)
_TWO_PI_POW_1P5: float = (2.0 * math.pi) ** 1.5
_PI_POW_1P5: float = math.pi**1.5


@dataclass(frozen=True)
class AtomicGaussianField:
    """SE(3)-equivariant anisotropic 3D Gaussian field describing a pocket or 3D ligand.

    Attributes:
        centers: Centroid-subtracted atomic positions ``[B, N, 3]`` (``float32``, zeroed on
            masked dummy slots).
        centroid_offset: World-space centroid ``[B, 3]`` subtracted when building ``centers``.
            Retaining this offset prevents independent-centering teleportation during joint
            pocket-ligand overlap and ROCS Shape Tanimoto scoring.
        covariances: Symmetric positive-definite ``3x3`` atomic covariances ``[B, N, 3, 3]``
            (``float32``, identity :math:`\\mathbf{I}_3` on dummy slots).
        chem_weights: Optional multi-channel pharmacophore amplitudes ``[B, N, C_chem]``.
        dipole_vecs: Optional equivariant polar dipole vectors ``[B, N, C_dip, 3]`` (e.g.
            signed halogen :math:`\\sigma`-hole or H-bond lone-pair dipoles).
        axial_vecs: Optional bounded chiral axial vectors ``[B, N, C_ax, 3]``.
    """

    centers: Tensor
    centroid_offset: Tensor
    covariances: Tensor
    chem_weights: Tensor | None = None
    dipole_vecs: Tensor | None = None
    axial_vecs: Tensor | None = None


@dataclass(frozen=True)
class GaussianOverlapResult:
    """Complete analytical 3D anisotropic Gaussian pair-overlap bundle.

    Attributes:
        density_integral: Exact normalized 3D Gaussian product integral
            ``integral N(x; mu_i, Sigma_i) N(x; mu_j, Sigma_j) dx``.
        rocs_volume: OpenEye ROCS unnormalized volume overlap
            ``integral g_i(x) g_j(x) dx``.
        bhattacharyya: Normalized Bhattacharyya overlap coefficient ``kappa_ij in (0, 1]``.
        mahalanobis_dist: Smooth Mahalanobis distance
            ``sqrt(max(d_ij^T S_ij^{-1} d_ij, 0) + 1e-6)``.
        logdet_divergence: Non-negative covariance divergence
            ``logdet(S_ij) - 0.5*logdet(Sigma_i) - 0.5*logdet(Sigma_j) - 3*log(2) >= 0``.
        force_vec: Equivariant Mahalanobis force vector ``p_ij = S_ij^{-1} d_ij`` ``[..., 3]``.
    """

    density_integral: Tensor
    rocs_volume: Tensor
    bhattacharyya: Tensor
    mahalanobis_dist: Tensor
    logdet_divergence: Tensor
    force_vec: Tensor


def _cofactors_and_det_3x3(mat: Tensor) -> tuple[Tensor, Tensor]:
    """Compute exact closed-form 3x3 adjugate (transpose cofactor matrix) and determinant."""
    if mat.ndim < 2 or mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected mat with trailing shape (3, 3), got {tuple(mat.shape)}")

    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 0, 2]
    d = mat[..., 1, 0]
    e = mat[..., 1, 1]
    f = mat[..., 1, 2]
    g = mat[..., 2, 0]
    h = mat[..., 2, 1]
    i = mat[..., 2, 2]

    c00 = e * i - f * h
    c01 = f * g - d * i
    c02 = d * h - e * g

    c10 = c * h - b * i
    c11 = a * i - c * g
    c12 = b * g - a * h

    c20 = b * f - c * e
    c21 = c * d - a * f
    c22 = a * e - b * d

    det = a * c00 + b * c01 + c * c02

    # Adjugate is the transpose of the cofactor matrix: adj[r, s] = C[s, r].
    row0 = torch.stack((c00, c10, c20), dim=-1)
    row1 = torch.stack((c01, c11, c21), dim=-1)
    row2 = torch.stack((c02, c12, c22), dim=-1)
    adj = torch.stack((row0, row1, row2), dim=-2)
    return adj, det


def cofactor_det_3x3(mat: Tensor) -> Tensor:
    """Closed-form exact ``3x3`` determinant without ``torch.linalg.det``."""
    _, det = _cofactors_and_det_3x3(mat)
    return det


def cofactor_inverse_and_logdet_3x3(
    mat: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Closed-form exact ``3x3`` matrix inverse and ``log(det)`` via analytical cofactors.

    Avoids ``torch.linalg.inv`` and ``torch.linalg.slogdet`` entirely so Apple Silicon MPS
    executes purely in fused elementwise SIMD kernels with ``< 3e-6`` relative error versus
    ``float64`` across all SPD matrices with condition number :math:`\\kappa \\le 30`.

    Args:
        mat: Invertible ``3x3`` SPD matrix batch of shape ``[..., 3, 3]``.
        eps: Positive lower clamp on determinant before inversion / logarithm.

    Returns:
        Tuple ``(inv_mat [..., 3, 3], logdet [...])``.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    adj, det = _cofactors_and_det_3x3(mat)
    det_safe = det.clamp(min=float(eps))
    inv_mat = adj / det_safe.unsqueeze(-1).unsqueeze(-1)
    logdet = torch.log(det_safe)
    return inv_mat, logdet


def cofactor_solve_3x3(
    mat: Tensor,
    rhs: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Solve ``mat @ x = rhs`` for ``3x3`` SPD batches via exact closed-form cofactors.

    Args:
        mat: SPD ``3x3`` matrix batch of shape ``[..., 3, 3]``.
        rhs: Right-hand side vectors of shape ``[..., 3]`` matching ``mat.shape[:-1]``.
        eps: Positive determinant floor.

    Returns:
        Tuple ``(solution [..., 3], logdet [...])``.
    """
    if rhs.ndim < 1 or rhs.shape[-1] != 3:
        raise ValueError(f"Expected rhs trailing dim 3, got {tuple(rhs.shape)}")
    inv_mat, logdet = cofactor_inverse_and_logdet_3x3(mat, eps=eps)
    sol = (inv_mat * rhs.unsqueeze(-2)).sum(dim=-1)
    return sol, logdet


def vdw_radius_to_iso_logit(
    vdw_norm: Tensor,
    *,
    s0_min: float = 0.25,
    s0_max: float = 4.0,
) -> Tensor:
    """Convert normalized Bondi vdW radius (``featurizer.py`` dim 45) to isotropic sigma logit.

    In ``featurizer.py``, ``node_feats[..., 45] = r_vdw / 2.0``. Setting target isotropic
    variance to :math:`(0.5\\,r_{\\mathrm{vdW}})^2 = \\mathrm{vdw\\_norm}^2` initializes each
    atom's Gaussian envelope directly from its physical van der Waals volume.
    """
    target_var = (vdw_norm.to(torch.float32)) ** 2
    span = max(float(s0_max) - float(s0_min), 1e-6)
    prob = ((target_var - float(s0_min)) / span).clamp(min=0.05, max=0.95)
    return torch.log(prob / (1.0 - prob))


def build_anisotropic_covariance_se3(
    iso_logits: Tensor,
    polar_vecs: Tensor | None = None,
    polar_logits: Tensor | None = None,
    axial_vecs: Tensor | None = None,
    axial_logits: Tensor | None = None,
    *,
    atom_mask: Tensor | None = None,
    s0_min: float = 0.25,
    s0_max: float = 4.0,
    polar_max: float = 1.2,
    axial_max: float = 1.2,
    normalize_axes: bool = True,
    norm_eps: float = 1.0,
) -> Tensor:
    """Construct unconditionally well-conditioned, ``O(3)``-equivariant SPD 3x3 covariances.

    Evaluates:

    .. math::
        \\mathbf{\\Sigma}_i = s_{i,0}\\,\\mathbf{I}_3
            + \\sum_{c=1}^{C_g} \\alpha_{i,c}\\,\\mathbf{u}_{i,c}\\mathbf{u}_{i,c}^\\top
            + \\sum_{a=1}^{C_a} \\beta_{i,a}\\,\\boldsymbol{\\omega}_{i,a}
              \\boldsymbol{\\omega}_{i,a}^\\top

    with scaled-sigmoid bounds :math:`s_{i,0} \\in [s_{0,\\min}, s_{0,\\max}]`,
    :math:`\\alpha_{i,c} \\in (0, \\alpha_{\\max})`, :math:`\\beta_{i,a} \\in (0, \\beta_{\\max})`,
    and smooth sub-unit axis normalization ``v / sqrt(||v||^2 + eps)`` (``eps = 1.0``). With
    default ``(C_g=4, C_a=2, s0_min=0.25, polar_max=axial_max=1.2)``, the condition number is
    provably bounded by ``cond(Sigma_i) <= 1 + 7.2 / 0.25 = 29.8 < 30`` across ALL inputs!

    Because :math:`(-\\mathbf{P}\\boldsymbol{\\omega})(-\\mathbf{P}\\boldsymbol{\\omega})^\\top =
    +\\mathbf{P}\\boldsymbol{\\omega}\\boldsymbol{\\omega}^\\top\\mathbf{P}^\\top`, the axial term
    creates strong out-of-plane :math:`\\pi`-cloud normal variance on planar aromatic rings
    while remaining strictly ``O(3)``-equivariant.

    Args:
        iso_logits: Unbounded isotropic variance logits of shape ``[...]`` (e.g. ``[B, N]``).
        polar_vecs: Optional polar axes ``[..., C_g, 3]``.
        polar_logits: Optional polar rank-1 weights logits ``[..., C_g]``.
        axial_vecs: Optional axial axes ``[..., C_a, 3]`` (e.g. from :func:`chiral_cross_product`).
        axial_logits: Optional axial rank-1 weight logits ``[..., C_a]``.
        atom_mask: Optional boolean mask ``[...]``; masked slots receive :math:`\\mathbf{I}_3`.
        s0_min: Strict positive isotropic eigenvalue floor (default ``0.25 A^2``).
        s0_max: Upper isotropic variance bound (default ``4.0 A^2``).
        polar_max: Per-channel upper bound on polar rank-1 variance (default ``1.2 A^2``).
        axial_max: Per-channel upper bound on axial rank-1 variance (default ``1.2 A^2``).
        normalize_axes: If ``True`` (default), smoothly bound ``||u||_2 < 1`` via ``norm_eps``.
        norm_eps: Positive smoothing denominator for axis bounding (default ``1.0``).

    Returns:
        SPD covariance tensor of shape ``[..., 3, 3]`` in ``torch.float32``.
    """
    if s0_min <= 0.0 or s0_max <= s0_min:
        raise ValueError(f"Expected 0 < s0_min < s0_max, got s0_min={s0_min}, s0_max={s0_max}")
    if polar_max < 0.0 or axial_max < 0.0:
        raise ValueError(f"polar_max ({polar_max}) and axial_max ({axial_max}) must be >= 0")
    if norm_eps <= 0.0:
        raise ValueError(f"norm_eps must be positive, got {norm_eps}")

    iso = iso_logits.to(torch.float32)
    ref_vec = polar_vecs if polar_vecs is not None else axial_vecs
    if ref_vec is not None and iso.ndim == ref_vec.ndim - 1 and iso.shape[-1] == 1:
        iso = iso.squeeze(-1)

    s0 = float(s0_min) + (float(s0_max) - float(s0_min)) * torch.sigmoid(iso)
    eye = torch.eye(3, device=iso.device, dtype=torch.float32)
    sigma = s0.unsqueeze(-1).unsqueeze(-1) * eye  # [..., 3, 3]

    if polar_vecs is not None:
        if polar_vecs.ndim < 2 or polar_vecs.shape[-1] != 3:
            raise ValueError(
                f"Expected polar_vecs shape [..., C_g, 3], got {tuple(polar_vecs.shape)}"
            )
        u = polar_vecs.to(torch.float32)
        if normalize_axes:
            u = u * torch.rsqrt((u * u).sum(dim=-1, keepdim=True) + float(norm_eps))
        if polar_logits is not None:
            alpha = float(polar_max) * torch.sigmoid(polar_logits.to(torch.float32))
        else:
            alpha = torch.full(
                u.shape[:-1], 0.5 * float(polar_max), device=u.device, dtype=torch.float32
            )
        # Sum_c alpha_c * u_c u_c^T -> [..., 3, 3]
        sigma = sigma + torch.einsum("...c,...cp,...cq->...pq", alpha, u, u)

    if axial_vecs is not None:
        if axial_vecs.ndim < 2 or axial_vecs.shape[-1] != 3:
            raise ValueError(
                f"Expected axial_vecs shape [..., C_a, 3], got {tuple(axial_vecs.shape)}"
            )
        w = axial_vecs.to(torch.float32)
        if normalize_axes:
            w = w * torch.rsqrt((w * w).sum(dim=-1, keepdim=True) + float(norm_eps))
        if axial_logits is not None:
            beta = float(axial_max) * torch.sigmoid(axial_logits.to(torch.float32))
        else:
            beta = torch.full(
                w.shape[:-1], 0.5 * float(axial_max), device=w.device, dtype=torch.float32
            )
        sigma = sigma + torch.einsum("...a,...ap,...aq->...pq", beta, w, w)

    if atom_mask is not None:
        mask_b = atom_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1)
        eye_b = eye.expand_as(sigma)
        sigma = torch.where(mask_b, sigma, eye_b)

    return sigma


def _broadcast_shared_frame_diff(
    mi: Tensor,
    mj: Tensor,
    offset_i: Tensor | None,
    offset_j: Tensor | None,
) -> Tensor:
    """Compute ``mi - mj`` in a shared world frame when centroid offsets are provided."""
    if offset_i is None and offset_j is None:
        return mi - mj
    if offset_i is None or offset_j is None:
        raise ValueError("Provide both offset_i and offset_j together (or neither).")
    rel_offset = offset_j.to(torch.float32) - offset_i.to(torch.float32)
    if rel_offset.ndim == 1:
        rel_view = rel_offset.view(*([1] * (mi.ndim - 1)), 3)
    else:
        lead = rel_offset.shape[:-1]
        extra_ones = mi.ndim - rel_offset.ndim
        if extra_ones < 0:
            raise ValueError(f"offset ndim {rel_offset.ndim} exceeds mu ndim {mi.ndim}")
        rel_view = rel_offset.view(*lead, *([1] * extra_ones), 3)
    return mi - (mj + rel_view)


def _sym3x3_det_6(
    s00: Tensor,
    s01: Tensor,
    s02: Tensor,
    s11: Tensor,
    s12: Tensor,
    s22: Tensor,
) -> Tensor:
    """Exact determinant of a symmetric 3x3 matrix from its 6 unique entries."""
    c00 = s11 * s22 - s12 * s12
    c01 = s02 * s12 - s01 * s22
    c02 = s01 * s12 - s02 * s11
    return s00 * c00 + s01 * c01 + s02 * c02


def _rocs_volume_sym3x3_fast(
    mu_i: Tensor,
    sigma_i: Tensor,
    mu_j: Tensor,
    sigma_j: Tensor,
    *,
    offset_i: Tensor | None = None,
    offset_j: Tensor | None = None,
    pairwise: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """Fast symmetric-3x3 ROCS volume overlap without materializing ``[..., 3, 3]`` inverses.

    Exploits symmetry of :math:`\\mathbf{S}_{ij} = \\mathbf{\\Sigma}_i + \\mathbf{\\Sigma}_j`
    using only its 6 unique upper-triangular entries to evaluate
    :math:`d^\\top \\operatorname{adj}(\\mathbf{S}) d / \\det(\\mathbf{S})` directly in
    elementwise scalar ops — avoiding 9-entry adjugate/inverse stacks and unused overlap fields.
    """
    mi = mu_i.to(torch.float32)
    mj = mu_j.to(torch.float32)
    si = sigma_i.to(torch.float32)
    sj = sigma_j.to(torch.float32)
    eps_f = float(eps)

    # Precompute per-atom sqrt(det(Sigma)) BEFORE pairwise broadcasting (O(N), not O(N*M)!).
    si00, si01, si02 = si[..., 0, 0], si[..., 0, 1], si[..., 0, 2]
    si11, si12, si22 = si[..., 1, 1], si[..., 1, 2], si[..., 2, 2]
    sj00, sj01, sj02 = sj[..., 0, 0], sj[..., 0, 1], sj[..., 0, 2]
    sj11, sj12, sj22 = sj[..., 1, 1], sj[..., 1, 2], sj[..., 2, 2]

    sqrt_det_i = torch.sqrt(_sym3x3_det_6(si00, si01, si02, si11, si12, si22).clamp(min=eps_f))
    sqrt_det_j = torch.sqrt(_sym3x3_det_6(sj00, sj01, sj02, sj11, sj12, sj22).clamp(min=eps_f))

    if pairwise:
        mi = mi.unsqueeze(-2)  # [..., N_i, 1, 3]
        mj = mj.unsqueeze(-3)  # [..., 1, N_j, 3]
        si00, si01, si02 = si00.unsqueeze(-1), si01.unsqueeze(-1), si02.unsqueeze(-1)
        si11, si12, si22 = si11.unsqueeze(-1), si12.unsqueeze(-1), si22.unsqueeze(-1)
        sj00, sj01, sj02 = sj00.unsqueeze(-2), sj01.unsqueeze(-2), sj02.unsqueeze(-2)
        sj11, sj12, sj22 = sj11.unsqueeze(-2), sj12.unsqueeze(-2), sj22.unsqueeze(-2)
        sqrt_det_i = sqrt_det_i.unsqueeze(-1)
        sqrt_det_j = sqrt_det_j.unsqueeze(-2)

    d_ij = _broadcast_shared_frame_diff(mi, mj, offset_i, offset_j)
    dx, dy, dz = d_ij[..., 0], d_ij[..., 1], d_ij[..., 2]

    s00 = si00 + sj00
    s01 = si01 + sj01
    s02 = si02 + sj02
    s11 = si11 + sj11
    s12 = si12 + sj12
    s22 = si22 + sj22

    # 6 unique symmetric cofactors of S_ij
    c00 = s11 * s22 - s12 * s12
    c11 = s00 * s22 - s02 * s02
    c22 = s00 * s11 - s01 * s01
    c01 = s02 * s12 - s01 * s22
    c02 = s01 * s12 - s02 * s11
    c12 = s01 * s02 - s00 * s12

    det_s = (s00 * c00 + s01 * c01 + s02 * c02).clamp(min=eps_f)
    quad_num = (
        c00 * (dx * dx)
        + c11 * (dy * dy)
        + c22 * (dz * dz)
        + 2.0 * (c01 * (dx * dy) + c02 * (dx * dz) + c12 * (dy * dz))
    )
    quad = (quad_num / det_s).clamp(min=0.0)
    return (_TWO_PI_POW_1P5 * sqrt_det_i * sqrt_det_j) * torch.rsqrt(det_s) * torch.exp(-0.5 * quad)


def gaussian_overlap_details_3d(
    mu_i: Tensor,
    sigma_i: Tensor,
    mu_j: Tensor,
    sigma_j: Tensor,
    *,
    offset_i: Tensor | None = None,
    offset_j: Tensor | None = None,
    pairwise: bool = False,
    eps: float = 1e-8,
) -> GaussianOverlapResult:
    """Compute exact analytical 3D anisotropic Gaussian overlaps via closed-form 3x3 cofactors.

    Supports both aligned/broadcasted pairs (e.g. ``O(NK)`` dense k-NN tensors
    ``mu_i [B, N, 1, 3]`` vs ``mu_j [B, N, K, 3]``) and full cross-pair grids
    (``pairwise=True``, mapping ``[B, N_i, 3]`` and ``[B, N_j, 3]`` to ``[B, N_i, N_j]``).

    When ``offset_i`` and ``offset_j`` (world-space centroids ``[B, 3]``) are provided, relative
    displacements are formed in a shared frame: ``d_ij = mu_i - (mu_j + (c_j - c_i))``,
    canceling any shared rigid translation (even ``100 A``) before subtraction while remaining
    strictly sensitive to relative separation :math:`\\mathbf{c}_j - \\mathbf{c}_i`.
    """
    if mu_i.shape[-1] != 3 or mu_j.shape[-1] != 3:
        raise ValueError(
            f"Expected mu_i/mu_j trailing dim 3, got {tuple(mu_i.shape)} and {tuple(mu_j.shape)}"
        )
    if sigma_i.shape[-2:] != (3, 3) or sigma_j.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected sigma_i/sigma_j trailing shape (3, 3), got "
            f"{tuple(sigma_i.shape)} and {tuple(sigma_j.shape)}"
        )

    mi = mu_i.to(torch.float32)
    mj = mu_j.to(torch.float32)
    si = sigma_i.to(torch.float32)
    sj = sigma_j.to(torch.float32)

    if pairwise:
        mi = mi.unsqueeze(-2)  # [..., N_i, 1, 3]
        mj = mj.unsqueeze(-3)  # [..., 1, N_j, 3]
        si = si.unsqueeze(-3)  # [..., N_i, 1, 3, 3]
        sj = sj.unsqueeze(-4)  # [..., 1, N_j, 3, 3]

    d_ij = _broadcast_shared_frame_diff(mi, mj, offset_i, offset_j)

    s_ij = si + sj
    s_inv, logdet_s = cofactor_inverse_and_logdet_3x3(s_ij, eps=eps)
    logdet_i = torch.log(cofactor_det_3x3(si).clamp(min=float(eps)))
    logdet_j = torch.log(cofactor_det_3x3(sj).clamp(min=float(eps)))

    force_vec = (s_inv * d_ij.unsqueeze(-2)).sum(dim=-1)  # S_ij^{-1} d_ij, [..., 3]
    quad = (d_ij * force_vec).sum(dim=-1).clamp(min=0.0)
    mahalanobis_dist = torch.sqrt(quad + 1e-6)

    logdet_div = (logdet_s - 0.5 * logdet_i - 0.5 * logdet_j - 3.0 * _LOG_2).clamp(min=0.0)

    log_density = -0.5 * (quad + logdet_s + 3.0 * _LOG_2PI)
    density_integral = torch.exp(log_density)

    log_rocs = 0.5 * (logdet_i + logdet_j - logdet_s + 3.0 * _LOG_2PI) - 0.5 * quad
    rocs_volume = torch.exp(log_rocs)

    log_bhatt = 0.25 * (logdet_i + logdet_j) - 0.5 * logdet_s + 1.5 * _LOG_2 - 0.5 * quad
    bhattacharyya = torch.exp(log_bhatt).clamp(min=0.0, max=1.0)

    return GaussianOverlapResult(
        density_integral=density_integral,
        rocs_volume=rocs_volume,
        bhattacharyya=bhattacharyya,
        mahalanobis_dist=mahalanobis_dist,
        logdet_divergence=logdet_div,
        force_vec=force_vec,
    )


def gaussian_overlap_integral_3d(
    mu_i: Tensor,
    sigma_i: Tensor,
    mu_j: Tensor,
    sigma_j: Tensor,
    *,
    offset_i: Tensor | None = None,
    offset_j: Tensor | None = None,
    pairwise: bool = False,
    mode: Literal["density", "rocs", "bhattacharyya"] = "density",
    eps: float = 1e-8,
) -> Tensor:
    """Evaluate exact analytical 3D Gaussian overlap integral between anisotropic ellipsoids.

    Args:
        mu_i: Mean coordinates ``[..., 3]`` of first Gaussian(s).
        sigma_i: SPD covariances ``[..., 3, 3]`` of first Gaussian(s).
        mu_j: Mean coordinates ``[..., 3]`` of second Gaussian(s).
        sigma_j: SPD covariances ``[..., 3, 3]`` of second Gaussian(s).
        offset_i: Optional world-space centroid offset ``[B, 3]`` for ``mu_i``.
        offset_j: Optional world-space centroid offset ``[B, 3]`` for ``mu_j``.
        pairwise: If ``True``, broadcast ``[..., N_i, ...]`` against ``[..., N_j, ...]``
            to produce a ``[..., N_i, N_j]`` overlap matrix.
        mode:
            - ``"density"`` (default): exact normalized integral
              :math:`\\int \\mathcal{N}_i(\\mathbf{x})\\mathcal{N}_j(\\mathbf{x})\\,d\\mathbf{x}`.
            - ``"rocs"``: OpenEye ROCS unnormalized volume overlap
              :math:`\\int g_i(\\mathbf{x}) g_j(\\mathbf{x})\\,d\\mathbf{x}`.
            - ``"bhattacharyya"``: normalized similarity :math:`\\kappa_{ij} \\in (0, 1]`.
        eps: Positive determinant floor for cofactor inversion.

    Returns:
        Overlap tensor of shape ``[...]`` (or ``[..., N_i, N_j]`` when ``pairwise=True``).
    """
    if mode == "rocs":
        if mu_i.shape[-1] != 3 or mu_j.shape[-1] != 3:
            raise ValueError(
                f"Expected mu_i/mu_j trailing dim 3, got "
                f"{tuple(mu_i.shape)} and {tuple(mu_j.shape)}"
            )
        if sigma_i.shape[-2:] != (3, 3) or sigma_j.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected sigma_i/sigma_j trailing shape (3, 3), got "
                f"{tuple(sigma_i.shape)} and {tuple(sigma_j.shape)}"
            )
        return _rocs_volume_sym3x3_fast(
            mu_i,
            sigma_i,
            mu_j,
            sigma_j,
            offset_i=offset_i,
            offset_j=offset_j,
            pairwise=pairwise,
            eps=eps,
        )

    res = gaussian_overlap_details_3d(
        mu_i,
        sigma_i,
        mu_j,
        sigma_j,
        offset_i=offset_i,
        offset_j=offset_j,
        pairwise=pairwise,
        eps=eps,
    )
    if mode == "density":
        return res.density_integral
    if mode == "bhattacharyya":
        return res.bhattacharyya
    raise ValueError(
        f"Unsupported overlap mode {mode!r}; expected 'density', 'rocs', or 'bhattacharyya'"
    )


def rocs_self_overlap(
    field: AtomicGaussianField,
    mask: Tensor | None = None,
    weights: Tensor | None = None,
    *,
    max_neighbors: int | None = None,
    cutoff: float | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute ROCS self-volume overlap ``O(A, A) = sum_{i,j} w_i w_j V_ij^{ROCS}``.

    When ``max_neighbors`` is ``None`` (or ``N <= max_neighbors + 1``), computes the exact
    all-pairs self-overlap via :func:`_rocs_volume_sym3x3_fast`. For large protein pockets
    (``N = 384..1500``), setting ``max_neighbors`` (e.g. ``32`` or ``48``) evaluates exact
    diagonal terms :math:`w_i^2 \\pi^{3/2}\\sqrt{\\det\\mathbf{\\Sigma}_i}` plus ``O(NK)``
    local neighbor pairs within ``cutoff`` (default ``12.0 A``), eliminating ``O(N^2)`` cofactor
    blowup while preserving sub-percent physical accuracy (Gaussian tails beyond 12 A are
    ``< 1e-4``).

    Returns:
        Self-overlap volume tensor of shape ``[B]`` in ``torch.float32``.
    """
    batch_size, n_atoms, _ = field.centers.shape
    device = field.centers.device
    if mask is None:
        m_bool = torch.ones((batch_size, n_atoms), device=device, dtype=torch.bool)
    else:
        m_bool = mask.to(torch.bool)

    if weights is None:
        w = torch.ones((batch_size, n_atoms), device=device, dtype=torch.float32)
    else:
        w = weights.to(torch.float32).clamp(min=0.0)
    w = torch.where(m_bool, w, torch.zeros_like(w))

    if n_atoms == 0:
        return torch.zeros((batch_size,), device=device, dtype=torch.float32)

    if max_neighbors is None or n_atoms <= max_neighbors + 1:
        vol_aa = _rocs_volume_sym3x3_fast(
            field.centers,
            field.covariances,
            field.centers,
            field.covariances,
            pairwise=True,
            eps=eps,
        )
        return (vol_aa * (w.unsqueeze(2) * w.unsqueeze(1))).sum(dim=(-2, -1))

    # O(NK) fast self-overlap: exact self diagonal + K nearest off-diagonal pairs.
    si = field.covariances.to(torch.float32)
    det_i = _sym3x3_det_6(
        si[..., 0, 0],
        si[..., 0, 1],
        si[..., 0, 2],
        si[..., 1, 1],
        si[..., 1, 2],
        si[..., 2, 2],
    ).clamp(min=float(eps))
    diag_vol = (_PI_POW_1P5 * torch.sqrt(det_i) * (w * w)).sum(dim=1)  # [B]

    eff_cutoff = float(cutoff) if cutoff is not None else 12.0
    graph = build_dense_knn_graph(
        field.centers,
        m_bool,
        cutoff=eff_cutoff,
        max_neighbors=int(max_neighbors),
        adaptive_cutoff=False,
    )
    nbr_centers = gather_neighbors(graph.centered_coords, graph.nbr_idx)  # [B, N, K, 3]
    nbr_covs = gather_neighbors(si, graph.nbr_idx)  # [B, N, K, 3, 3]
    nbr_w = gather_neighbors(w, graph.nbr_idx)  # [B, N, K]

    vol_off = _rocs_volume_sym3x3_fast(
        graph.centered_coords.unsqueeze(2),
        si.unsqueeze(2),
        nbr_centers,
        nbr_covs,
        pairwise=False,
        eps=eps,
    )
    pair_w = torch.where(graph.nbr_mask, w.unsqueeze(2) * nbr_w, torch.zeros_like(nbr_w))
    off_vol = (vol_off * pair_w).sum(dim=(-2, -1))
    return diag_vol + off_vol


def rocs_shape_tanimoto(
    field_a: AtomicGaussianField,
    field_b: AtomicGaussianField,
    mask_a: Tensor | None = None,
    mask_b: Tensor | None = None,
    *,
    weights_a: Tensor | None = None,
    weights_b: Tensor | None = None,
    self_overlap_a: Tensor | None = None,
    self_overlap_b: Tensor | None = None,
    max_self_neighbors: int | None = None,
    self_cutoff: float | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute exact shared-frame OpenEye ROCS analytical 3D Gaussian Shape Tanimoto.

    Evaluates:

    .. math::
        T_{\\mathrm{ROCS}}(A, B) = \\frac{O(A, B)}{O(A, A) + O(B, B) - O(A, B) + \\varepsilon}
        \\in [0, 1],

    where :math:`O(A, B) = \\sum_{i \\in A, j \\in B} w_i^A w_j^B V_{ij}^{\\mathrm{ROCS}}` is
    computed in a **shared world frame** via ``field_b.centroid_offset - field_a.centroid_offset``.
    Strictly invariant under joint :math:`SE(3)` transforms (even ``100 A`` translations!) and
    decays smoothly to ``0`` when ``A`` and ``B`` are translated apart.

    For large protein pockets (``N_A = 384..1500``), pass precomputed ``self_overlap_a`` (from
    :func:`rocs_self_overlap`) or ``max_self_neighbors`` (e.g. ``32``) to avoid ``O(N_A^2)``
    self-overlap evaluation.

    Returns:
        Per-batch Shape Tanimoto tensor of shape ``[B]`` in ``[0, 1]``.
    """
    batch_size, n_a, _ = field_a.centers.shape
    n_b = field_b.centers.shape[1]
    device = field_a.centers.device

    if mask_a is None:
        ma = torch.ones((batch_size, n_a), device=device, dtype=torch.bool)
    else:
        ma = mask_a.to(torch.bool)
    if mask_b is None:
        mb = torch.ones((batch_size, n_b), device=device, dtype=torch.bool)
    else:
        mb = mask_b.to(torch.bool)

    wa = (
        torch.ones((batch_size, n_a), device=device, dtype=torch.float32)
        if weights_a is None
        else weights_a.to(torch.float32).clamp(min=0.0)
    )
    wb = (
        torch.ones((batch_size, n_b), device=device, dtype=torch.float32)
        if weights_b is None
        else weights_b.to(torch.float32).clamp(min=0.0)
    )
    wa = torch.where(ma, wa, torch.zeros_like(wa))
    wb = torch.where(mb, wb, torch.zeros_like(wb))

    vol_ab = _rocs_volume_sym3x3_fast(
        field_a.centers,
        field_a.covariances,
        field_b.centers,
        field_b.covariances,
        offset_i=field_a.centroid_offset,
        offset_j=field_b.centroid_offset,
        pairwise=True,
        eps=eps,
    )
    o_ab = (vol_ab * (wa.unsqueeze(2) * wb.unsqueeze(1))).sum(dim=(-2, -1))

    if self_overlap_a is not None:
        o_aa = self_overlap_a.to(device=device, dtype=torch.float32)
    else:
        o_aa = rocs_self_overlap(
            field_a,
            mask=ma,
            weights=wa,
            max_neighbors=max_self_neighbors,
            cutoff=self_cutoff,
            eps=eps,
        )

    if self_overlap_b is not None:
        o_bb = self_overlap_b.to(device=device, dtype=torch.float32)
    else:
        o_bb = rocs_self_overlap(
            field_b,
            mask=mb,
            weights=wb,
            max_neighbors=max_self_neighbors,
            cutoff=self_cutoff,
            eps=eps,
        )

    denom = (o_aa + o_bb - o_ab).clamp(min=0.0) + float(eps)
    tanimoto = (o_ab / denom).clamp(min=0.0, max=1.0)
    both_non_empty = ma.any(dim=1) & mb.any(dim=1)
    return torch.where(both_non_empty, tanimoto, torch.zeros_like(tanimoto))


def gaussianformer2_occupancy(
    bhattacharyya: Tensor,
    nbr_mask: Tensor,
    edge_env: Tensor | None = None,
    *,
    clip_max: float = 0.999,
) -> Tensor:
    """Compute GaussianFormer-2 (CVPR 2025) probabilistic superposition occupancy.

    Eliminates multi-neighbor volume double-counting via exact independent survival fusion:

    .. math::
        P_{\\mathrm{occ}, i} = 1 - \\exp\\!\\left(
            \\sum_{j=1}^{K} \\mathbb{1}[\\mathrm{mask}_{ij}]\\,
            \\log\\bigl(1 - \\operatorname{clip}(c_{\\max}\\,\\kappa_{ij}\\,f_{c,ij})\\bigr)
        \\right) \\in [0, 1).

    Isolated atoms with zero valid neighbors yield exactly ``0.0``; tightly buried core atoms
    approach ``1.0`` smoothly with finite gradients everywhere.
    """
    if not (0.0 < clip_max < 1.0):
        raise ValueError(f"clip_max must be strictly in (0, 1), got {clip_max}")

    mask_b = nbr_mask.to(torch.bool)
    p_ij = bhattacharyya.to(torch.float32) * float(clip_max)
    if edge_env is not None:
        p_ij = p_ij * edge_env.to(torch.float32)
    p_ij = p_ij.clamp(min=0.0, max=float(clip_max))

    log_surv = torch.where(mask_b, torch.log1p(-p_ij), torch.zeros_like(p_ij))
    total_log_surv = log_surv.sum(dim=-1)
    return (-torch.expm1(total_log_surv)).clamp(min=0.0, max=1.0)


def fuse_4d_md_covariances(
    centers: Tensor,
    covariances: Tensor,
    weights: Tensor | None = None,
    *,
    time_dim: int = 0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Fuse a ``T``-conformer MD/holo ensemble via the exact Law of Total Covariance (4DGS).

    Given conformer means :math:`\\boldsymbol{\\mu}^{(t)}` and SPD covariances
    :math:`\\mathbf{\\Sigma}^{(t)}` with normalized mixture weights :math:`w_t \\ge 0`,
    :math:`\\sum_t w_t = 1`:

    .. math::
        \\bar{\\boldsymbol{\\mu}} = \\sum_{t=1}^{T} w_t\\,\\boldsymbol{\\mu}^{(t)}, \\qquad
        \\mathbf{\\Sigma}^{\\mathrm{4D}} = \\sum_{t=1}^{T} w_t\\!\\left(
            \\mathbf{\\Sigma}^{(t)}
            + (\\boldsymbol{\\mu}^{(t)} - \\bar{\\boldsymbol{\\mu}})
              (\\boldsymbol{\\mu}^{(t)} - \\bar{\\boldsymbol{\\mu}})^\\top
        \\right).

    Guarantees exact :math:`SE(3)` equivariance and PSD dominance
    :math:`\\mathbf{\\Sigma}^{\\mathrm{4D}} \\succeq \\sum_t w_t \\mathbf{\\Sigma}^{(t)}`.

    Args:
        centers: Ensemble coordinates with conformer axis at ``time_dim`` and trailing dim ``3``.
        covariances: Ensemble SPD covariances matching ``centers.shape[:-1] + (3, 3)``.
        weights: Optional 1D conformer weights of length ``T`` (default uniform ``1/T``).
        time_dim: Dimension indexing the ``T`` conformers (default ``0``).
        eps: Numerical guard for normalizing custom weights.

    Returns:
        Tuple ``(mean_centers, fused_covariances)`` with ``time_dim`` reduced away.
    """
    if centers.ndim < 2 or centers.shape[-1] != 3:
        raise ValueError(
            f"Expected centers shape [..., 3] with ndim >= 2, got {tuple(centers.shape)}"
        )
    if covariances.shape != (*centers.shape[:-1], 3, 3):
        raise ValueError(
            f"covariances shape {tuple(covariances.shape)} != expected "
            f"{(*centers.shape[:-1], 3, 3)}"
        )

    t_axis = time_dim if time_dim >= 0 else centers.ndim - 1 + time_dim
    if not (0 <= t_axis < centers.ndim - 1):
        raise ValueError(f"Invalid time_dim={time_dim} for centers shape {tuple(centers.shape)}")

    c_f = centers.to(torch.float32)
    s_f = covariances.to(torch.float32)
    num_t = c_f.shape[t_axis]
    if num_t < 1:
        raise ValueError("Cannot fuse an empty conformer ensemble (T == 0).")

    if weights is None:
        w = torch.full((num_t,), 1.0 / num_t, device=c_f.device, dtype=torch.float32)
    else:
        w_raw = weights.to(device=c_f.device, dtype=torch.float32).reshape(-1)
        if w_raw.numel() != num_t:
            raise ValueError(f"weights length {w_raw.numel()} != T={num_t}")
        w_clamped = w_raw.clamp(min=0.0)
        w = w_clamped / w_clamped.sum().clamp(min=float(eps))

    w_shape_c = [1] * c_f.ndim
    w_shape_c[t_axis] = num_t
    w_c = w.view(*w_shape_c)

    mean_centers = (c_f * w_c).sum(dim=t_axis)
    delta = c_f - mean_centers.unsqueeze(t_axis)
    outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)  # [..., 3, 3]

    w_shape_s = [1] * s_f.ndim
    w_shape_s[t_axis] = num_t
    w_s = w.view(*w_shape_s)

    fused_cov = ((s_f + outer) * w_s).sum(dim=t_axis)
    return mean_centers, fused_cov


def fuse_multiconformer_fields(
    fields: Sequence[AtomicGaussianField],
    weights: Tensor | None = None,
    *,
    atom_mask: Tensor | None = None,
) -> AtomicGaussianField:
    """Fuse a sequence of aligned :class:`AtomicGaussianField` conformers into one 4D field.

    Args:
        fields: Non-empty sequence of aligned :class:`AtomicGaussianField` conformers.
        weights: Optional 1D mixture weights of length ``len(fields)`` (default uniform).
        atom_mask: Optional boolean atom validity mask ``[B, N]``. When provided, padded dummy
            slots stay strictly at ``centers = 0.0``, ``covariances = I_3``, and zero optional
            descriptors even when conformers have differing ``centroid_offset``s.
    """
    if not fields:
        raise ValueError("fields must contain at least one AtomicGaussianField.")
    num_t = len(fields)
    device = fields[0].centers.device

    if weights is None:
        w = torch.full((num_t,), 1.0 / num_t, device=device, dtype=torch.float32)
    else:
        w_raw = weights.to(device=device, dtype=torch.float32).reshape(-1)
        if w_raw.numel() != num_t:
            raise ValueError(f"weights length {w_raw.numel()} != len(fields)={num_t}")
        w_pos = w_raw.clamp(min=0.0)
        w = w_pos / w_pos.sum().clamp(min=1e-8)

    offsets = torch.stack([f.centroid_offset.to(torch.float32) for f in fields], dim=0)  # [T, B, 3]
    mean_offset = (offsets * w.view(num_t, 1, 1)).sum(dim=0)  # [B, 3]

    mask_b: Tensor | None = None
    if atom_mask is not None:
        mask_b = atom_mask.to(device=device, dtype=torch.bool)

    # Align every conformer's centered coordinates into the shared mean_offset frame first.
    aligned_list: list[Tensor] = []
    for f in fields:
        shifted = f.centers.to(torch.float32) + (
            f.centroid_offset.to(torch.float32) - mean_offset
        ).unsqueeze(1)
        if mask_b is not None:
            shifted = torch.where(mask_b.unsqueeze(-1), shifted, torch.zeros_like(shifted))
        aligned_list.append(shifted)
    aligned_centers = torch.stack(aligned_list, dim=0)
    covs = torch.stack([f.covariances.to(torch.float32) for f in fields], dim=0)
    mean_centers, fused_cov = fuse_4d_md_covariances(aligned_centers, covs, weights=w, time_dim=0)

    if mask_b is not None:
        eye = torch.eye(3, device=device, dtype=torch.float32).expand_as(fused_cov)
        mean_centers = torch.where(
            mask_b.unsqueeze(-1), mean_centers, torch.zeros_like(mean_centers)
        )
        fused_cov = torch.where(mask_b.unsqueeze(-1).unsqueeze(-1), fused_cov, eye)

    chem_weights: Tensor | None = None
    if all(f.chem_weights is not None for f in fields):
        stacked_chem = torch.stack(
            [f.chem_weights.to(torch.float32) for f in fields if f.chem_weights is not None], dim=0
        )
        chem_weights = (stacked_chem * w.view(num_t, 1, 1, 1)).sum(dim=0)
        if mask_b is not None:
            chem_weights = torch.where(
                mask_b.unsqueeze(-1), chem_weights, torch.zeros_like(chem_weights)
            )

    dipole_vecs: Tensor | None = None
    if all(f.dipole_vecs is not None for f in fields):
        stacked_dip = torch.stack(
            [f.dipole_vecs.to(torch.float32) for f in fields if f.dipole_vecs is not None], dim=0
        )
        dipole_vecs = (stacked_dip * w.view(num_t, 1, 1, 1, 1)).sum(dim=0)
        if mask_b is not None:
            dipole_vecs = torch.where(
                mask_b.unsqueeze(-1).unsqueeze(-1), dipole_vecs, torch.zeros_like(dipole_vecs)
            )

    axial_vecs: Tensor | None = None
    if all(f.axial_vecs is not None for f in fields):
        stacked_ax = torch.stack(
            [f.axial_vecs.to(torch.float32) for f in fields if f.axial_vecs is not None], dim=0
        )
        axial_vecs = (stacked_ax * w.view(num_t, 1, 1, 1, 1)).sum(dim=0)
        if mask_b is not None:
            axial_vecs = torch.where(
                mask_b.unsqueeze(-1).unsqueeze(-1), axial_vecs, torch.zeros_like(axial_vecs)
            )

    return AtomicGaussianField(
        centers=mean_centers,
        centroid_offset=mean_offset,
        covariances=fused_cov,
        chem_weights=chem_weights,
        dipole_vecs=dipole_vecs,
        axial_vecs=axial_vecs,
    )


def super_ellipsoid_pool(
    centers: Tensor,
    covariances: Tensor,
    assign_weights: Tensor,
    *,
    atom_mask: Tensor | None = None,
    trace_reg: float = 0.05,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Pool ``N`` atomic Gaussians into ``M`` trace-stabilized Gaussian Super-Ellipsoids.

    Uses the parallel-axis mixture covariance identity plus isotropic trace stabilization
    (Plan §4.2 ``CRIT-3.3``):

    .. math::
        \\boldsymbol{\\mu}_m = \\sum_{i=1}^N \\tilde{\\pi}_{m,i}\\boldsymbol{\\mu}_i, \\qquad
        \\tilde{\\mathbf{\\Sigma}}_m = \\sum_{i=1}^N \\tilde{\\pi}_{m,i}
            \\bigl(\\mathbf{\\Sigma}_i + (\\boldsymbol{\\mu}_i - \\boldsymbol{\\mu}_m)
            (\\boldsymbol{\\mu}_i - \\boldsymbol{\\mu}_m)^\\top\\bigr),

    .. math::
        \\mathbf{\\Sigma}_m^{\\mathrm{sup}} =
            \\tilde{\\mathbf{\\Sigma}}_m
            + \\eta\\,\\operatorname{tr}(\\tilde{\\mathbf{\\Sigma}}_m)\\,\\mathbf{I}_3
            \\quad (\\eta = 0.05).

    Even on flat 15 Angstrom cavity walls, adding ``0.05 * tr(Sigma) * I_3`` guarantees
    well-conditioned super-ellipsoid covariances (:math:`\\kappa \\le 20` on molecular pockets)
    so downstream ``M x M`` cofactor overlaps stay strictly accurate in ``float32``.

    Args:
        centers: Atomic positions ``[B, N, 3]``.
        covariances: Atomic SPD covariances ``[B, N, 3, 3]``.
        assign_weights: Non-negative slot assignment weights ``[B, M, N]``.
        atom_mask: Optional boolean atom validity mask ``[B, N]``.
        trace_reg: Isotropic trace stabilization coefficient (default ``0.05``).
        eps: Minimum slot mass guard for empty slots.

    Returns:
        Tuple ``(super_centers [B, M, 3], super_covariances [B, M, 3, 3])``.
    """
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError(f"Expected centers shape [B, N, 3], got {tuple(centers.shape)}")
    if covariances.shape != (*centers.shape[:2], 3, 3):
        raise ValueError(
            f"covariances shape {tuple(covariances.shape)} != expected {(*centers.shape[:2], 3, 3)}"
        )
    if (
        assign_weights.ndim != 3
        or assign_weights.shape[0] != centers.shape[0]
        or assign_weights.shape[2] != centers.shape[1]
    ):
        raise ValueError(
            f"assign_weights shape {tuple(assign_weights.shape)} incompatible with "
            f"centers shape {tuple(centers.shape)}"
        )
    if trace_reg < 0.0:
        raise ValueError(f"trace_reg must be >= 0, got {trace_reg}")

    c_f = centers.to(torch.float32)
    s_f = covariances.to(torch.float32)
    w_raw = assign_weights.to(torch.float32).clamp(min=0.0)  # [B, M, N]
    if atom_mask is not None:
        mask_b = atom_mask.to(torch.bool).unsqueeze(1)  # [B, 1, N]
        w_raw = torch.where(mask_b, w_raw, torch.zeros_like(w_raw))

    slot_mass = w_raw.sum(dim=-1, keepdim=True)  # [B, M, 1]
    has_mass = slot_mass > float(eps)
    pi = w_raw / slot_mass.clamp(min=float(eps))  # [B, M, N]

    super_centers = torch.einsum("bmn,bnd->bmd", pi, c_f)  # [B, M, 3]
    super_centers = torch.where(has_mass, super_centers, torch.zeros_like(super_centers))

    diff = c_f.unsqueeze(1) - super_centers.unsqueeze(2)  # [B, M, N, 3]
    cov_term = torch.einsum("bmn,bnpq->bmpq", pi, s_f)
    disp_term = torch.einsum("bmn,bmnp,bmnq->bmpq", pi, diff, diff)
    sigma_mix = cov_term + disp_term  # [B, M, 3, 3]

    tr = sigma_mix[..., 0, 0] + sigma_mix[..., 1, 1] + sigma_mix[..., 2, 2]
    eye = torch.eye(3, device=c_f.device, dtype=torch.float32)
    sigma_sup = sigma_mix + (float(trace_reg) * tr).unsqueeze(-1).unsqueeze(-1) * eye

    eye_b = eye.view(1, 1, 3, 3).expand_as(sigma_sup)
    sigma_sup = torch.where(has_mass.unsqueeze(-1), sigma_sup, eye_b)
    return super_centers, sigma_sup


__all__ = [
    "AtomicGaussianField",
    "GaussianOverlapResult",
    "build_anisotropic_covariance_se3",
    "cofactor_det_3x3",
    "cofactor_inverse_and_logdet_3x3",
    "cofactor_solve_3x3",
    "fuse_4d_md_covariances",
    "fuse_multiconformer_fields",
    "gaussian_overlap_details_3d",
    "gaussian_overlap_integral_3d",
    "gaussianformer2_occupancy",
    "rocs_self_overlap",
    "rocs_shape_tanimoto",
    "super_ellipsoid_pool",
    "vdw_radius_to_iso_logit",
]
