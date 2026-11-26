"""All-Atom Biomolecular Co-Folding Diffusion, Rectified Flow & Physics Losses.

Pure-PyTorch ``torch.float32`` all-atom coordinate generation primitives reusing
``geneuclidean.geom`` (``build_dense_knn_graph``, ``BesselRBF``, ``CosineCutoff``,
``ScreenedCoulombBias``, ``cofactor_det_3x3``, ``safe_masked_softmax``) with zero
``index_add_`` scatter contention and zero ``float64`` parameters/buffers:

- :class:`KabschWeightedRigidAlign`: AF3 Alg. 28 reflection-safe ``SO(3)`` weighted Kabsch
  alignment using native MPS ``3x3`` SVD + :func:`cofactor_det_3x3` (unconditionally
  enforces :math:`\\det(\\mathbf{R}) = +1` so chiral stereocenters are NEVER inverted).
- :class:`EDMDiffusionSchedule`: AF3 Alg. 18–20 EDM preconditioning, Karras noise grid,
  and stochastic churn sampler.
- :class:`FlowMatchingSchedule`: Apple MLR SimpleFold linear-interpolant continuous Flow
  Matching schedule with deterministic Heun / Euler integration.
- :class:`AtomAttentionEncoder`, :class:`DiffusionTransformer`, :class:`AtomAttentionDecoder`,
  and :class:`AllAtomDiffusionModule`: scatter-free ``bmm`` atom↔token denoiser supporting
  ``flexible_pocket_and_ligand``, ``rigid_pocket_flexible_ligand``, and ``de_novo_all_atom``.
- :class:`CoFoldingLoss`: 6-term physics-grounded training objective (weighted aligned MSE,
  normalized Smooth LDDT, 1–2/1–3 covalent distance, signed tetrahedral chiral volume,
  Bondi vdW steric clash, and pocket backbone tether).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.folding.pairformer import AdaLNZero, PairBiasedAttention, SwiGLUTransition
from geneuclidean.models.encoders.dense_knn import (
    DenseKNNGraph,
    build_dense_knn_graph,
    gather_neighbors,
    safe_masked_softmax,
)
from geneuclidean.models.encoders.electrostatics import ScreenedCoulombBias
from geneuclidean.models.encoders.gaussian_ops import cofactor_det_3x3
from geneuclidean.models.encoders.rbf import BesselRBF, CosineCutoff

FlexMode = Literal[
    "flexible_pocket_and_ligand",
    "rigid_pocket_flexible_ligand",
    "de_novo_all_atom",
]
DiffusionMode = Literal["edm", "flow_matching"]


def _randn_deterministic(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ``float32`` Gaussian noise on CPU when seeded so CPU & MPS match identically."""
    if generator is None:
        return torch.randn(shape, device=device, dtype=torch.float32)
    return torch.randn(
        shape, device=torch.device("cpu"), dtype=torch.float32, generator=generator
    ).to(device)


def _sample_random_so3(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample uniform Haar ``SO(3)`` rotations ``[B, 3, 3]`` via QR + ``cofactor_det_3x3``."""
    z = _randn_deterministic((batch_size, 3, 3), device, generator=generator)
    # Run QR on CPU if MPS rejects it, though PyTorch 2.15 supports QR cleanly.
    try:
        q, r = torch.linalg.qr(z)
    except RuntimeError:
        q_cpu, r_cpu = torch.linalg.qr(z.cpu())
        q, r = q_cpu.to(device), r_cpu.to(device)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(diag < 0.0, -torch.ones_like(diag), torch.ones_like(diag))
    q = q * signs.unsqueeze(-2)
    det = cofactor_det_3x3(q)
    flip = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
    q_proper = torch.cat(
        (q[..., :, :2], q[..., :, 2:3] * flip.view(batch_size, 1, 1)),
        dim=-1,
    )
    return q_proper


def _cross_product_3d(a: Tensor, b: Tensor) -> Tensor:
    """Closed-form elementwise 3D cross product (zero MPS fallback risk, pure float32)."""
    return torch.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        dim=-1,
    )


def compute_signed_tetrahedral_volume(coords: Tensor, quads: Tensor) -> Tensor:
    """Compute signed tetrahedral volume ``(r1 - r0) . ((r2 - r0) x (r3 - r0))`` for ``[B, C, 4]``.

    The scalar triple product equals ``6 * signed_volume`` and flips sign unconditionally
    under any improper mirror reflection (``det(P) = -1``), distinguishing ``(R)`` vs ``(S)``
    stereocenters directly in differentiable Cartesian coordinates.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [B, N, 3], got {tuple(coords.shape)}")
    if quads.ndim != 3 or quads.shape[-1] != 4:
        raise ValueError(f"Expected quads shape [B, C, 4], got {tuple(quads.shape)}")

    batch_size, num_quads, _ = quads.shape
    num_atoms = coords.shape[1]
    if batch_size == 0 or num_quads == 0:
        return torch.empty((batch_size, num_quads), device=coords.device, dtype=torch.float32)

    safe_q = quads.to(torch.long).clamp(min=0, max=max(num_atoms - 1, 0))
    flat_idx = safe_q.reshape(batch_size, num_quads * 4, 1).expand(-1, -1, 3)
    pts = torch.gather(coords.to(torch.float32), dim=1, index=flat_idx).reshape(
        batch_size, num_quads, 4, 3
    )
    v1 = pts[:, :, 1] - pts[:, :, 0]
    v2 = pts[:, :, 2] - pts[:, :, 0]
    v3 = pts[:, :, 3] - pts[:, :, 0]
    return (v1 * _cross_product_3d(v2, v3)).sum(dim=-1)


class KabschWeightedRigidAlign(nn.Module):
    """Reflection-safe weighted Kabsch rigid alignment (AF3 Alg. 28) in pure ``torch.float32``.

    Minimizes :math:`\\sum_i w_i \\| \\mathbf{R}\\mathbf{x}_{\\mathrm{src},i} + \\mathbf{t}
    - \\mathbf{x}_{\\mathrm{ref},i} \\|_2^2` subject strictly to
    :math:`\\mathbf{R} \\in SO(3)` (:math:`\\det(\\mathbf{R}) = +1`).

    Uses native Apple MPS ``3x3`` SVD (with automatic CPU fallback for legacy PyTorch) and
    :func:`~geneuclidean.models.encoders.gaussian_ops.cofactor_det_3x3` to guarantee proper
    rotations — preventing silent enantiomer inversion even on mirror-reflected inputs,
    collinear fragments, single-atom masks, or 100 Å world translations.
    """

    eps: float
    jitter: float
    _diag_jitter: Tensor

    def __init__(self, eps: float = 1e-8, jitter: float = 1e-7) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if jitter < 0.0:
            raise ValueError(f"jitter must be non-negative, got {jitter}")
        self.eps = float(eps)
        self.jitter = float(jitter)
        # Distinct tiny diagonal perturbation breaks degenerate singular values (s2 == s3)
        # on collinear or coincident point clouds without measurably affecting SO(3) accuracy.
        diag_j = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) * self.jitter)
        self.register_buffer("_diag_jitter", diag_j)

    def _compute_transform(
        self,
        x_src: Tensor,
        x_ref: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute proper rotation ``R [B, 3, 3]``, centroids ``mu_src, mu_ref``, and valid flag."""
        batch_size = x_src.shape[0]
        device = x_src.device

        w = weights.to(torch.float32).clamp(min=0.0)
        w_pos = w > 0.0
        w_sum = w.sum(dim=1, keepdim=True)  # [B, 1]
        has_valid = w_sum.squeeze(-1) > 0.0  # [B]
        w_norm = w / w_sum.clamp(min=self.eps)  # [B, N]

        # Zero zero-weight coordinates before centering so padded garbage never leaks.
        zeros = torch.zeros_like(x_src)
        safe_src = torch.where(w_pos.unsqueeze(-1), x_src, zeros)
        safe_ref = torch.where(w_pos.unsqueeze(-1), x_ref, zeros)

        mu_src = (w_norm.unsqueeze(-1) * safe_src).sum(dim=1, keepdim=True)  # [B, 1, 3]
        mu_ref = (w_norm.unsqueeze(-1) * safe_ref).sum(dim=1, keepdim=True)  # [B, 1, 3]

        c_src = torch.where(w_pos.unsqueeze(-1), safe_src - mu_src, zeros)
        c_ref = torch.where(w_pos.unsqueeze(-1), safe_ref - mu_ref, zeros)

        # Cross-covariance H = sum_i w_i * c_src_i * c_ref_i^T  -> [B, 3, 3]
        h_mat = torch.einsum("bn,bni,bnj->bij", w_norm, c_src, c_ref)
        h_reg = h_mat + self._diag_jitter.to(device=device, dtype=torch.float32).unsqueeze(0)

        try:
            u_mat, _, vh_mat = torch.linalg.svd(h_reg)
        except RuntimeError:
            u_cpu, _, vh_cpu = torch.linalg.svd(h_reg.detach().cpu())
            u_mat = u_cpu.to(device)
            vh_mat = vh_cpu.to(device)

        v_mat = vh_mat.transpose(-1, -2)
        u_t = u_mat.transpose(-1, -2)
        vu_t = torch.matmul(v_mat, u_t)

        # Exact det(V U^T) via closed-form 3x3 cofactor determinant (never flips chirality!).
        det = cofactor_det_3x3(vu_t)
        sign_d = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))

        # Construct D = diag(1, 1, sign_d) and R = V D U^T -> det(R) = +1 unconditionally.
        ones = torch.ones_like(sign_d)
        diag_vec = torch.stack((ones, ones, sign_d), dim=-1)  # [B, 3]
        rot = torch.matmul(v_mat * diag_vec.unsqueeze(-2), u_t)  # [B, 3, 3]

        eye = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 3, 3)
        rot = torch.where(has_valid.view(batch_size, 1, 1), rot, eye)
        return rot, mu_src, mu_ref, has_valid

    def forward(
        self,
        x_src: Tensor,
        x_ref: Tensor,
        weights: Tensor,
        *,
        detach_transform: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Align ``x_src [B, N, 3]`` onto ``x_ref [B, N, 3]``.

        Returns:
            Tuple ``(x_aligned [B, N, 3], R [B, 3, 3], t [B, 3])`` where
            ``x_aligned = x_src @ R^T + t`` and ``det(R) == +1``.
        """
        if x_src.ndim != 3 or x_src.shape[-1] != 3:
            raise ValueError(f"Expected x_src shape [B, N, 3], got {tuple(x_src.shape)}")
        if x_ref.shape != x_src.shape:
            raise ValueError(f"x_ref shape {tuple(x_ref.shape)} != x_src {tuple(x_src.shape)}")
        if weights.shape != x_src.shape[:2]:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} != x_src[:2] {tuple(x_src.shape[:2])}"
            )

        src_f = x_src.to(torch.float32)
        ref_f = x_ref.to(torch.float32)
        if src_f.shape[0] == 0 or src_f.shape[1] == 0:
            eye = (
                torch.eye(3, device=src_f.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(src_f.shape[0], 3, 3)
            )
            zeros_t = torch.zeros((src_f.shape[0], 3), device=src_f.device, dtype=torch.float32)
            return src_f, eye, zeros_t

        if detach_transform:
            with torch.no_grad():
                rot, mu_src, mu_ref, has_valid = self._compute_transform(
                    src_f.detach(), ref_f.detach(), weights.detach()
                )
        else:
            rot, mu_src, mu_ref, has_valid = self._compute_transform(src_f, ref_f, weights)

        # Row-vector transform: x_aligned = (x_src - mu_src) @ R^T + mu_ref = x_src @ R^T + t
        rot_t = rot.transpose(-1, -2)
        t_vec = (mu_ref - torch.matmul(mu_src, rot_t)).squeeze(1)  # [B, 3]
        t_vec = torch.where(has_valid.unsqueeze(-1), t_vec, torch.zeros_like(t_vec))
        x_aligned = torch.matmul(src_f - mu_src, rot_t) + mu_ref
        x_aligned = torch.where(has_valid.view(-1, 1, 1), x_aligned, src_f)
        return x_aligned, rot, t_vec


class EDMDiffusionSchedule(nn.Module):
    """AlphaFold-3 Algorithms 18–20 EDM preconditioning, noise sampling & stochastic churn."""

    sigma_data: float
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    rho: float
    gamma_0: float
    gamma_min: float
    noise_scale: float
    step_scale: float

    def __init__(
        self,
        sigma_data: float = 16.0,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        rho: float = 7.0,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
    ) -> None:
        super().__init__()
        if sigma_data <= 0.0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}")
        if sigma_min <= 0.0 or sigma_max <= sigma_min:
            raise ValueError(f"Expected 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}")
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.gamma_0 = float(gamma_0)
        self.gamma_min = float(gamma_min)
        self.noise_scale = float(noise_scale)
        self.step_scale = float(step_scale)

    def sample_sigma(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample log-normal training noise levels ``sigma [B]`` (AF3 §3.7.1)."""
        z = _randn_deterministic((batch_size,), device, generator=generator)
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * z)

    def precond_scales(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return AF3/EDM ``(c_skip, c_out, c_in, c_noise)`` tensors of shape ``sigma.shape``."""
        s = sigma.to(torch.float32).clamp(min=1e-8)
        sd = self.sigma_data
        sd2 = sd * sd
        s2 = s * s
        denom = s2 + sd2
        inv_sqrt = torch.rsqrt(denom)
        c_skip = sd2 / denom
        c_out = (s * sd) * inv_sqrt
        c_in = inv_sqrt
        c_noise = 0.25 * torch.log(s)
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigma: Tensor) -> Tensor:
        """AF3 §3.7.1 noise-level loss weight ``(sigma^2 + sigma_d^2) / (sigma + sigma_d)^2``."""
        s = sigma.to(torch.float32)
        sd = self.sigma_data
        return (s * s + sd * sd) / ((s + sd) ** 2).clamp(min=1e-8)

    def add_noise(
        self,
        x0: Tensor,
        sigma: Tensor,
        noise: Tensor,
        flex_mask: Tensor | None = None,
    ) -> Tensor:
        """Corrupt ``x0 [B, N, 3]`` on flexible atoms via ``x0 + sigma * noise``."""
        noisy = x0 + sigma.view(-1, 1, 1) * noise
        if flex_mask is not None:
            noisy = torch.where(flex_mask.unsqueeze(-1), noisy, x0)
        return noisy

    def get_sampling_sigmas(self, num_steps: int, device: torch.device) -> Tensor:
        """Return Karras noise schedule ``[num_steps + 1]`` descending to ``0.0``."""
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        t = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=torch.float32)
        inv_rho = 1.0 / self.rho
        s_max_inv = self.sigma_max**inv_rho
        s_min_inv = self.sigma_min**inv_rho
        sigmas = (s_max_inv + t * (s_min_inv - s_max_inv)) ** self.rho
        return torch.cat((sigmas, torch.zeros(1, device=device, dtype=torch.float32)), dim=0)


class FlowMatchingSchedule(nn.Module):
    """Apple MLR SimpleFold linear-interpolant rectified Flow Matching schedule."""

    sigma_fm: float
    logit_mean: float
    logit_std: float
    eps: float
    use_heun: bool

    def __init__(
        self,
        sigma_fm: float = 8.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        eps: float = 1e-4,
        use_heun: bool = True,
    ) -> None:
        super().__init__()
        if sigma_fm <= 0.0:
            raise ValueError(f"sigma_fm must be positive, got {sigma_fm}")
        self.sigma_fm = float(sigma_fm)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.eps = float(eps)
        self.use_heun = bool(use_heun)

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample continuous time ``t in [1e-3, 1 - 1e-3]`` via logit-normal distribution."""
        z = _randn_deterministic((batch_size,), device, generator=generator)
        return torch.sigmoid(self.logit_mean + self.logit_std * z).clamp(min=1e-3, max=1.0 - 1e-3)

    def interpolate(
        self,
        x0: Tensor,
        noise: Tensor,
        t: Tensor,
        flex_mask: Tensor | None = None,
    ) -> Tensor:
        """Linear interpolant ``x_t = (1 - t) * sigma_fm * noise + t * x0`` on flexible atoms."""
        t_v = t.view(-1, 1, 1)
        x_t = (1.0 - t_v) * (self.sigma_fm * noise) + t_v * x0
        if flex_mask is not None:
            x_t = torch.where(flex_mask.unsqueeze(-1), x_t, x0)
        return x_t

    def velocity_from_x0(self, x_t: Tensor, x0_pred: Tensor, t: Tensor) -> Tensor:
        """Convert clean coordinate prediction ``x0_pred`` into rectified flow velocity ``v``."""
        denom = (1.0 - t.view(-1, 1, 1)).clamp(min=self.eps)
        return (x0_pred - x_t) / denom

    def loss_weight(self, t: Tensor) -> Tensor:
        """Smooth bounded time-dependent loss weighting emphasizing clean geometry."""
        return 0.5 + 1.5 * t.to(torch.float32)

    def get_time_grid(self, num_steps: int, device: torch.device) -> Tensor:
        """Return strictly increasing time grid ``[num_steps + 1]`` from ``0.0`` to ``1.0``."""
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)


class FourierEmbedding(nn.Module):
    """AlphaFold-3 Algorithm 22 deterministic ``float32`` Fourier noise-level embedding."""

    dim: int
    w: Tensor
    b: Tensor

    def __init__(self, dim: int, out_dim: int | None = None, *, seed: int = 12345) -> None:
        super().__init__()
        self.dim = int(dim)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        w = torch.randn(self.dim, generator=gen, dtype=torch.float32)
        b = torch.randn(self.dim, generator=gen, dtype=torch.float32)
        self.register_buffer("w", w)
        self.register_buffer("b", b)
        target_dim = int(out_dim if out_dim is not None else dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, target_dim),
            nn.SiLU(),
            nn.Linear(target_dim, target_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed scalar noise levels ``t [B]`` into ``[B, out_dim]``."""
        angles = 2.0 * math.pi * (t.to(torch.float32).unsqueeze(-1) * self.w + self.b)
        feat = torch.cos(angles)
        out: Tensor = self.proj(feat)
        return out


class AtomLocalAttentionBlock(nn.Module):
    """Scatter-free ``O(N K)`` dense k-NN local atom attention block in pure ``float32``.

    Operates directly over :class:`~geneuclidean.models.encoders.dense_knn.DenseKNNGraph`
    using :func:`gather_neighbors`, :class:`BesselRBF`, :class:`CosineCutoff`, optional
    covalent edge attributes, and :func:`safe_masked_softmax` (zero ``index_add_``!).
    """

    c_atom: int
    num_heads: int
    head_dim: int

    def __init__(
        self,
        c_atom: int,
        num_heads: int = 2,
        *,
        edge_dim: int = 24,
        c_cond: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if c_atom % num_heads != 0:
            raise ValueError(f"c_atom ({c_atom}) must be divisible by num_heads ({num_heads})")
        self.c_atom = int(c_atom)
        self.num_heads = int(num_heads)
        self.head_dim = self.c_atom // self.num_heads
        c_cond_eff = int(c_cond if c_cond is not None else self.c_atom)

        self.adaln = AdaLNZero(self.c_atom, c_cond_eff)
        self.q_proj = nn.Linear(self.c_atom, self.c_atom, bias=False)
        self.k_proj = nn.Linear(self.c_atom, self.c_atom, bias=False)
        self.v_proj = nn.Linear(self.c_atom, self.c_atom, bias=False)
        self.edge_bias = nn.Linear(edge_dim, self.num_heads, bias=False)
        self.edge_val = nn.Linear(edge_dim, self.c_atom, bias=False)
        self.gate_proj = nn.Linear(self.c_atom, self.c_atom, bias=False)
        self.out_proj = nn.Linear(self.c_atom, self.c_atom, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

        self.mlp_adaln = AdaLNZero(self.c_atom, c_cond_eff)
        self.mlp = SwiGLUTransition(self.c_atom, expansion=2, dropout=dropout, use_norm=False)

    def forward(
        self,
        q_atom: Tensor,
        graph: DenseKNNGraph,
        edge_feats: Tensor,
        atom_mask: Tensor,
        *,
        cond: Tensor | None = None,
        coulomb_nbr_bias: Tensor | None = None,
    ) -> Tensor:
        """Update per-atom embeddings ``q_atom [B, N, c_atom]`` over dense k-NN graph."""
        batch_size, num_atoms, _ = q_atom.shape
        k_nbr = graph.nbr_idx.shape[-1]
        m_b = atom_mask.to(torch.bool)
        q_in = torch.where(m_b.unsqueeze(-1), q_atom, torch.zeros_like(q_atom))

        if k_nbr == 0 or num_atoms == 0:
            return q_in

        h, gate_attn = self.adaln(q_in, cond)
        q = self.q_proj(h).reshape(batch_size, num_atoms, self.num_heads, self.head_dim)
        k_all = self.k_proj(h).reshape(batch_size, num_atoms, self.num_heads, self.head_dim)
        v_all = self.v_proj(h).reshape(batch_size, num_atoms, self.num_heads, self.head_dim)

        k_nbr_t = gather_neighbors(k_all, graph.nbr_idx)  # [B, N, K, H, dh]
        v_nbr_t = gather_neighbors(v_all, graph.nbr_idx)  # [B, N, K, H, dh]
        v_edge = self.edge_val(edge_feats).reshape(
            batch_size, num_atoms, k_nbr, self.num_heads, self.head_dim
        )

        scale = 1.0 / math.sqrt(float(self.head_dim))
        logits = (q.unsqueeze(2) * k_nbr_t).sum(dim=-1) * scale  # [B, N, K, H]
        logits = logits + self.edge_bias(edge_feats)
        if coulomb_nbr_bias is not None:
            logits = logits + coulomb_nbr_bias

        nbr_mask_h = graph.nbr_mask.unsqueeze(-1)  # [B, N, K, 1]
        attn = safe_masked_softmax(logits, nbr_mask_h, dim=2)  # [B, N, K, H]
        # Envelope-modulated neighbor aggregation preserves exact tie-swap smoothness.
        weighted = attn * graph.edge_env.unsqueeze(-1)
        agg = (weighted.unsqueeze(-1) * (v_nbr_t + v_edge)).sum(dim=2)  # [B, N, H, dh]
        agg_flat = agg.reshape(batch_size, num_atoms, self.c_atom)

        delta_attn = gate_attn * self.out_proj(torch.sigmoid(self.gate_proj(h)) * agg_flat)
        delta_attn = torch.where(m_b.unsqueeze(-1), delta_attn, torch.zeros_like(delta_attn))
        q_mid = q_in + delta_attn

        h_mlp, gate_mlp = self.mlp_adaln(q_mid, cond)
        delta_mlp = gate_mlp * self.mlp(h_mlp, use_norm=False)
        delta_mlp = torch.where(m_b.unsqueeze(-1), delta_mlp, torch.zeros_like(delta_mlp))
        return q_mid + delta_mlp


def _build_membership_matrix(
    atom_to_token: Tensor,
    atom_mask: Tensor,
    num_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Build scatter-free binary token-atom membership ``M_tok [B, L, N]`` and normalized copy."""
    safe_tok = atom_to_token.to(torch.long).clamp(min=0, max=max(num_tokens - 1, 0))
    m_tok = F.one_hot(safe_tok, num_classes=num_tokens).transpose(1, 2).to(torch.float32)
    m_tok = m_tok * atom_mask.to(torch.float32).unsqueeze(1)
    counts = m_tok.sum(dim=-1, keepdim=True).clamp(min=1.0)
    m_tok_norm = m_tok / counts
    return m_tok, m_tok_norm


class AtomAttentionEncoder(nn.Module):
    """AlphaFold-3 Algorithm 5 atom-to-token encoder reusing ``geneuclidean.geom`` primitives.

    Builds a tie-safe dense k-NN graph via :func:`build_dense_knn_graph`, encodes distances
    with :class:`BesselRBF` + :class:`CosineCutoff`, injects long-range :class:`ScreenedCoulombBias`
    anchored to ``dim 46`` partial charges, runs local k-NN atom attention, and pools atoms
    to AF3 mixed tokens via scatter-free ``torch.bmm(M_tok_norm, q_atom)``.
    """

    c_atom: int
    c_token: int
    knn_cutoff: float
    knn_max_neighbors: int

    def __init__(
        self,
        atom_feat_dim: int,
        c_atom: int,
        c_token: int,
        c_s: int,
        c_cond: int,
        *,
        covalent_feat_dim: int = 6,
        num_layers: int = 2,
        num_heads: int = 2,
        num_rbf: int = 16,
        knn_cutoff: float = 8.0,
        knn_max_neighbors: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.c_atom = int(c_atom)
        self.c_token = int(c_token)
        self.knn_cutoff = float(knn_cutoff)
        self.knn_max_neighbors = int(knn_max_neighbors)
        self.covalent_feat_dim = int(covalent_feat_dim)

        # Input features: 52-D atom descriptors + 3D scaled noisy coords + 3D ref coords + flex flag
        self.atom_in_proj = nn.Linear(atom_feat_dim + 7, self.c_atom)
        self.trunk_s_proj = nn.Linear(c_s, self.c_atom, bias=False)
        self.cond_proj = nn.Linear(c_cond, self.c_atom)

        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=self.knn_cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=self.knn_cutoff)
        # Edge features: RBF + env + 3D dir + covalent features
        edge_in_dim = num_rbf + 1 + 3 + self.covalent_feat_dim
        edge_dim = max(24, self.c_atom // 2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.coulomb = ScreenedCoulombBias(scalar_dim=self.c_atom, num_heads=num_heads)
        self.layers = nn.ModuleList(
            [
                AtomLocalAttentionBlock(
                    self.c_atom,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    c_cond=self.c_atom,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.atom_to_tok_proj = nn.Sequential(
            nn.LayerNorm(self.c_atom),
            nn.Linear(self.c_atom, self.c_token),
        )

    def _build_edge_features(
        self,
        graph: DenseKNNGraph,
        covalent_adj: Tensor | None,
    ) -> Tensor:
        """Assemble local k-NN geometric + optional covalent edge embeddings ``[B, N, K, D]``."""
        batch_size, num_atoms, k_nbr = graph.nbr_idx.shape
        rbf_e = self.rbf(graph.edge_dist) * self.cutoff_fn(graph.edge_dist).unsqueeze(-1)
        if covalent_adj is not None:
            # Gather covalent_adj[b, i, nbr_idx[b, i, k], :] without sparse indexing.
            idx_exp = graph.nbr_idx.unsqueeze(-1).expand(
                batch_size, num_atoms, k_nbr, covalent_adj.shape[-1]
            )
            cov_e = torch.gather(covalent_adj.to(torch.float32), dim=2, index=idx_exp)
        else:
            cov_e = torch.zeros(
                (batch_size, num_atoms, k_nbr, self.covalent_feat_dim),
                device=graph.edge_dist.device,
                dtype=torch.float32,
            )
        raw_edge = torch.cat(
            (rbf_e, graph.edge_env.unsqueeze(-1), graph.edge_dir, cov_e),
            dim=-1,
        )
        edge_feats: Tensor = self.edge_mlp(raw_edge)
        return torch.where(graph.nbr_mask.unsqueeze(-1), edge_feats, torch.zeros_like(edge_feats))

    def forward(
        self,
        r_noisy: Tensor,
        r_scaled: Tensor,
        atom_feats: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        num_tokens: int,
        *,
        s_trunk: Tensor,
        t_emb: Tensor,
        flex_mask: Tensor | None = None,
        r_ref: Tensor | None = None,
        covalent_adj: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, DenseKNNGraph, Tensor]:
        """Encode all atoms into token representations plus skip states for the decoder."""
        m_b = atom_mask.to(torch.bool)
        m_tok, m_tok_norm = _build_membership_matrix(atom_to_token, m_b, num_tokens)

        graph = build_dense_knn_graph(
            r_noisy,
            m_b,
            cutoff=self.knn_cutoff,
            max_neighbors=self.knn_max_neighbors,
        )
        edge_feats = self._build_edge_features(graph, covalent_adj)

        r_ref_f = torch.zeros_like(r_scaled) if r_ref is None else r_ref.to(torch.float32)
        flex_f = (
            m_b.to(torch.float32).unsqueeze(-1)
            if flex_mask is None
            else flex_mask.to(torch.float32).unsqueeze(-1)
        )
        atom_in = torch.cat((atom_feats.to(torch.float32), r_scaled, r_ref_f, flex_f), dim=-1)
        q_atom = self.atom_in_proj(atom_in)

        # Broadcast trunk single representation s_trunk [B, L, c_s] -> atoms via bmm.
        s_atom = torch.bmm(m_tok.transpose(1, 2), self.trunk_s_proj(s_trunk.to(torch.float32)))
        c_atom_cond = s_atom + self.cond_proj(t_emb.to(torch.float32)).unsqueeze(1)
        q_atom = torch.where(m_b.unsqueeze(-1), q_atom + s_atom, torch.zeros_like(q_atom))

        # Long-range screened Coulomb electrostatic bias along k-NN edges (anchored to dim 46).
        coul_full = self.coulomb(
            r_noisy,
            q_atom,
            raw_feats_i=atom_feats,
            mask_i=m_b,
        )  # [B, H, N, N]
        coul_nnh = coul_full.permute(0, 2, 3, 1).contiguous()  # [B, N, N, H]
        idx_h = graph.nbr_idx.unsqueeze(-1).expand(-1, -1, -1, coul_nnh.shape[-1])
        coul_nbr = torch.gather(coul_nnh, dim=2, index=idx_h)  # [B, N, K, H]

        for layer in self.layers:
            q_atom = layer(
                q_atom,
                graph,
                edge_feats,
                m_b,
                cond=c_atom_cond,
                coulomb_nbr_bias=coul_nbr,
            )

        # Scatter-free token pooling via Metal GEMM (torch.bmm).
        a_token = torch.bmm(m_tok_norm, self.atom_to_tok_proj(q_atom))  # [B, L, c_token]
        return a_token, q_atom, c_atom_cond, m_tok, graph, edge_feats


class DiffusionTransformer(nn.Module):
    """AlphaFold-3 Algorithm 23 token-level AdaLN-Zero pair-biased diffusion transformer."""

    c_token: int
    num_layers: int

    def __init__(
        self,
        c_token: int,
        c_s: int,
        c_z: int,
        c_cond: int,
        *,
        num_layers: int = 4,
        head_dim: int = 32,
        num_heads: int | None = None,
        use_rope: bool = False,
        transition_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim < 32:
            raise ValueError(f"DiffusionTransformer enforces head_dim >= 32, got {head_dim}")
        heads = max(1, c_token // head_dim) if num_heads is None else int(num_heads)
        self.c_token = int(c_token)
        self.num_layers = int(num_layers)

        self.s_cond_proj = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_cond),
        )
        self.attn_blocks = nn.ModuleList(
            [
                PairBiasedAttention(
                    self.c_token,
                    c_z=c_z,
                    num_heads=heads,
                    use_adaln=True,
                    c_cond=c_cond,
                    use_rope=use_rope,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_adalns = nn.ModuleList(
            [AdaLNZero(self.c_token, c_cond) for _ in range(self.num_layers)]
        )
        self.mlp_blocks = nn.ModuleList(
            [
                SwiGLUTransition(
                    self.c_token, expansion=transition_expansion, dropout=dropout, use_norm=False
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        a_token: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        t_emb: Tensor,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """Refine token representations conditioned on trunk ``(s, z)`` and timestep ``t_emb``."""
        m_b = token_mask.to(torch.bool) if token_mask is not None else None
        a = a_token.to(torch.float32)
        if m_b is not None:
            a = torch.where(m_b.unsqueeze(-1), a, torch.zeros_like(a))

        cond = self.s_cond_proj(s_trunk.to(torch.float32)) + t_emb.to(torch.float32).unsqueeze(1)
        for attn, adaln, mlp in zip(
            self.attn_blocks, self.mlp_adalns, self.mlp_blocks, strict=True
        ):
            a = a + attn(a, z=z_trunk, mask=m_b, cond=cond)
            h_mlp, gate_mlp = adaln(a, cond)
            delta_mlp = gate_mlp * mlp(h_mlp, use_norm=False)
            if m_b is not None:
                delta_mlp = torch.where(m_b.unsqueeze(-1), delta_mlp, torch.zeros_like(delta_mlp))
            a = a + delta_mlp
        return a


class AtomAttentionDecoder(nn.Module):
    """AlphaFold-3 Algorithm 6 token-to-atom coordinate update decoder."""

    c_atom: int
    c_token: int

    def __init__(
        self,
        c_atom: int,
        c_token: int,
        *,
        num_layers: int = 2,
        num_heads: int = 2,
        edge_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.c_atom = int(c_atom)
        self.c_token = int(c_token)
        e_dim = int(edge_dim if edge_dim is not None else max(24, self.c_atom // 2))

        self.tok_to_atom_proj = nn.Sequential(
            nn.LayerNorm(self.c_token),
            nn.Linear(self.c_token, self.c_atom),
        )
        self.layers = nn.ModuleList(
            [
                AtomLocalAttentionBlock(
                    self.c_atom,
                    num_heads=num_heads,
                    edge_dim=e_dim,
                    c_cond=self.c_atom,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.coord_head = nn.Sequential(
            nn.LayerNorm(self.c_atom),
            nn.Linear(self.c_atom, 3, bias=False),
        )

    def forward(
        self,
        a_token: Tensor,
        q_skip: Tensor,
        c_atom_cond: Tensor,
        m_tok: Tensor,
        graph: DenseKNNGraph,
        edge_feats: Tensor,
        atom_mask: Tensor,
        *,
        flex_mask: Tensor | None = None,
    ) -> Tensor:
        """Broadcast tokens to atoms via ``bmm`` and emit coordinate updates ``[B, N, 3]``."""
        m_b = atom_mask.to(torch.bool)
        q_from_tok = torch.bmm(m_tok.transpose(1, 2), self.tok_to_atom_proj(a_token))
        q = torch.where(m_b.unsqueeze(-1), q_skip + q_from_tok, torch.zeros_like(q_skip))

        for layer in self.layers:
            q = layer(q, graph, edge_feats, m_b, cond=c_atom_cond)

        r_update: Tensor = self.coord_head(q)
        active_mask = m_b if flex_mask is None else (m_b & flex_mask.to(torch.bool))
        return torch.where(active_mask.unsqueeze(-1), r_update, torch.zeros_like(r_update))


class AllAtomDiffusionModule(nn.Module):
    """Unified All-Atom Diffusion / Rectified Flow Co-Folding Denoiser.

    Supports both ``diffusion_mode="edm"`` (AF3 Alg. 18–20) and
    ``diffusion_mode="flow_matching"`` (SimpleFold rectified flow) across all three
    flexibility regimes:
    - ``"flexible_pocket_and_ligand"``: joint induced-fit sidechain + ligand denoising.
    - ``"rigid_pocket_flexible_ligand"``: fixed receptor pocket, flexible ligand denoising.
    - ``"de_novo_all_atom"``: unconstrained all-atom denoising with random ``SO(3)``
      training augmentation.
    """

    diffusion_mode: DiffusionMode
    c_atom: int
    c_token: int

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_atom: int = 96,
        c_token: int = 192,
        *,
        atom_feat_dim: int = 52,
        covalent_feat_dim: int = 6,
        head_dim: int = 32,
        num_diff_layers: int = 4,
        num_atom_enc_layers: int = 2,
        num_atom_dec_layers: int = 2,
        knn_cutoff: float = 8.0,
        knn_max_neighbors: int = 24,
        diffusion_mode: DiffusionMode = "edm",
        sigma_data: float = 16.0,
        sigma_fm: float = 8.0,
        transition_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if diffusion_mode not in ("edm", "flow_matching"):
            raise ValueError(
                f"diffusion_mode must be 'edm' or 'flow_matching', got {diffusion_mode!r}"
            )
        self.diffusion_mode = diffusion_mode
        self.c_atom = int(c_atom)
        self.c_token = int(c_token)
        c_cond = min(128, max(64, self.c_token // 2))

        atom_heads = 4 if (self.c_atom % 4 == 0) else 2
        edge_dim = max(24, self.c_atom // 2)

        self.edm_schedule = EDMDiffusionSchedule(sigma_data=sigma_data)
        self.fm_schedule = FlowMatchingSchedule(sigma_fm=sigma_fm)
        self.time_embed = FourierEmbedding(dim=min(128, c_cond), out_dim=c_cond)
        self.encoder = AtomAttentionEncoder(
            atom_feat_dim=atom_feat_dim,
            c_atom=self.c_atom,
            c_token=self.c_token,
            c_s=c_s,
            c_cond=c_cond,
            covalent_feat_dim=covalent_feat_dim,
            num_layers=num_atom_enc_layers,
            num_heads=atom_heads,
            knn_cutoff=knn_cutoff,
            knn_max_neighbors=knn_max_neighbors,
            dropout=dropout,
        )
        self.transformer = DiffusionTransformer(
            c_token=self.c_token,
            c_s=c_s,
            c_z=c_z,
            c_cond=c_cond,
            num_layers=num_diff_layers,
            head_dim=head_dim,
            use_rope=(diffusion_mode == "flow_matching"),
            transition_expansion=transition_expansion,
            dropout=dropout,
        )
        self.decoder = AtomAttentionDecoder(
            c_atom=self.c_atom,
            c_token=self.c_token,
            num_layers=num_atom_dec_layers,
            num_heads=atom_heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

    @staticmethod
    def resolve_flex_mask(
        atom_mask: Tensor,
        atom_feats: Tensor,
        flex_mode: FlexMode,
        flex_mask: Tensor | None = None,
    ) -> Tensor:
        """Resolve which atoms are denoised under the requested ``flex_mode``."""
        m_b = atom_mask.to(torch.bool)
        if flex_mask is not None:
            return m_b & flex_mask.to(torch.bool)
        if flex_mode == "de_novo_all_atom":
            return m_b
        if flex_mode == "rigid_pocket_flexible_ligand":
            if atom_feats.shape[-1] > 42:
                is_lig = (atom_feats[..., 30] > 0.5) & (atom_feats[..., 42] < 0.5)
                has_lig = is_lig.any(dim=1, keepdim=True)
                return m_b & torch.where(has_lig, is_lig, m_b)
            return m_b
        if flex_mode == "flexible_pocket_and_ligand":
            if atom_feats.shape[-1] > 42:
                non_bb = atom_feats[..., 42] < 0.5
                has_non_bb = non_bb.any(dim=1, keepdim=True)
                return m_b & torch.where(has_non_bb, non_bb, m_b)
            return m_b
        raise ValueError(f"Unsupported flex_mode {flex_mode!r}")

    def denoise(
        self,
        r_noisy: Tensor,
        noise_level: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        atom_feats: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        *,
        token_mask: Tensor | None = None,
        flex_mask: Tensor | None = None,
        r_ref: Tensor | None = None,
        covalent_adj: Tensor | None = None,
    ) -> Tensor:
        """Predict clean all-atom coordinates ``x0_pred [B, N, 3]`` at ``noise_level [B]``."""
        m_b = atom_mask.to(torch.bool)
        r_f = torch.where(m_b.unsqueeze(-1), r_noisy.to(torch.float32), torch.zeros_like(r_noisy))
        num_tokens = s_trunk.shape[1]

        if self.diffusion_mode == "edm":
            c_skip, c_out, c_in, c_noise = self.edm_schedule.precond_scales(noise_level)
            r_scaled = c_in.view(-1, 1, 1) * r_f
            t_emb = self.time_embed(c_noise)
        else:
            r_scaled = r_f / self.fm_schedule.sigma_fm
            t_emb = self.time_embed(noise_level.to(torch.float32))
            c_skip = None
            c_out = None

        # Mask reference coords strictly onto anchored (non-flexible) atoms so the denoiser
        # can NEVER see ground-truth world coordinates of atoms it is predicting.
        safe_r_ref: Tensor | None
        if r_ref is not None:
            if flex_mask is not None:
                anchor_m = m_b & (~flex_mask.to(torch.bool))
                safe_r_ref = torch.where(
                    anchor_m.unsqueeze(-1), r_ref.to(torch.float32), torch.zeros_like(r_ref)
                )
            else:
                safe_r_ref = None
        else:
            safe_r_ref = None

        a_token, q_skip, c_atom_cond, m_tok, graph, edge_feats = self.encoder(
            r_noisy=r_f,
            r_scaled=r_scaled,
            atom_feats=atom_feats,
            atom_to_token=atom_to_token,
            atom_mask=m_b,
            num_tokens=num_tokens,
            s_trunk=s_trunk,
            t_emb=t_emb,
            flex_mask=flex_mask,
            r_ref=safe_r_ref,
            covalent_adj=covalent_adj,
        )
        a_token = self.transformer(
            a_token=a_token,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            t_emb=t_emb,
            token_mask=token_mask,
        )
        r_update = self.decoder(
            a_token=a_token,
            q_skip=q_skip,
            c_atom_cond=c_atom_cond,
            m_tok=m_tok,
            graph=graph,
            edge_feats=edge_feats,
            atom_mask=m_b,
            flex_mask=flex_mask,
        )

        if self.diffusion_mode == "edm":
            assert c_skip is not None and c_out is not None
            x0_pred = c_skip.view(-1, 1, 1) * r_f + c_out.view(-1, 1, 1) * r_update
        else:
            # Residual x0 prediction around r_f for smooth stability across t in [0, 1].
            x0_pred = r_f + self.fm_schedule.sigma_fm * r_update

        if flex_mask is not None:
            active = m_b & flex_mask.to(torch.bool)
            x0_pred = torch.where(active.unsqueeze(-1), x0_pred, r_f)
        return torch.where(m_b.unsqueeze(-1), x0_pred, torch.zeros_like(x0_pred))

    def forward_train(
        self,
        x0: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        atom_feats: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        *,
        token_mask: Tensor | None = None,
        flex_mask: Tensor | None = None,
        r_ref: Tensor | None = None,
        covalent_adj: Tensor | None = None,
        flex_mode: FlexMode = "flexible_pocket_and_ligand",
        seed: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one training forward step returning ``(x0_pred, x0_target, noise_level, weight)``."""
        batch_size = x0.shape[0]
        device = x0.device
        m_b = atom_mask.to(torch.bool)
        active_flex = self.resolve_flex_mask(m_b, atom_feats, flex_mode, flex_mask)
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None

        x0_target = torch.where(m_b.unsqueeze(-1), x0.to(torch.float32), torch.zeros_like(x0))
        if flex_mode == "de_novo_all_atom":
            # AF3 CentreRandomAugmentation: center + random Haar SO(3) + N(0, 1 A) translation.
            counts = m_b.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32).unsqueeze(-1)
            centered = x0_target - x0_target.sum(dim=1, keepdim=True) / counts
            rot = _sample_random_so3(batch_size, device, generator=gen)
            trans = _randn_deterministic((batch_size, 1, 3), device, generator=gen)
            x0_target = torch.where(
                m_b.unsqueeze(-1),
                torch.matmul(centered, rot.transpose(-1, -2)) + trans,
                torch.zeros_like(x0_target),
            )

        eps = _randn_deterministic(tuple(x0_target.shape), device, generator=gen)
        if self.diffusion_mode == "edm":
            noise_level = self.edm_schedule.sample_sigma(batch_size, device, generator=gen)
            r_noisy = self.edm_schedule.add_noise(
                x0_target, noise_level, eps, flex_mask=active_flex
            )
            loss_w = self.edm_schedule.loss_weight(noise_level)
        else:
            noise_level = self.fm_schedule.sample_time(batch_size, device, generator=gen)
            r_noisy = self.fm_schedule.interpolate(
                x0_target, eps, noise_level, flex_mask=active_flex
            )
            loss_w = self.fm_schedule.loss_weight(noise_level)

        x0_pred = self.denoise(
            r_noisy=r_noisy,
            noise_level=noise_level,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            atom_feats=atom_feats,
            atom_to_token=atom_to_token,
            atom_mask=m_b,
            token_mask=token_mask,
            flex_mask=active_flex,
            r_ref=r_ref,
            covalent_adj=covalent_adj,
        )
        return x0_pred, x0_target, noise_level, loss_w

    def sample(
        self,
        r_init: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        atom_feats: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        *,
        token_mask: Tensor | None = None,
        flex_mask: Tensor | None = None,
        r_ref: Tensor | None = None,
        covalent_adj: Tensor | None = None,
        flex_mode: FlexMode = "flexible_pocket_and_ligand",
        num_steps: int = 20,
        seed: int | None = 42,
    ) -> Tensor:
        """Deterministically sample denoised all-atom coordinates ``[B, N, 3]``."""
        batch_size = r_init.shape[0]
        device = r_init.device
        m_b = atom_mask.to(torch.bool)
        active_flex = self.resolve_flex_mask(m_b, atom_feats, flex_mode, flex_mask)
        r_anchor = torch.where(
            m_b.unsqueeze(-1), r_init.to(torch.float32), torch.zeros_like(r_init)
        )

        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
        init_eps = _randn_deterministic(tuple(r_anchor.shape), device, generator=gen)

        if self.diffusion_mode == "edm":
            sigmas = self.edm_schedule.get_sampling_sigmas(num_steps, device)
            # Initialize flexible atoms from Gaussian prior scaled by sigma_max.
            x = torch.where(
                active_flex.unsqueeze(-1),
                r_anchor + sigmas[0] * init_eps
                if flex_mode != "de_novo_all_atom"
                else sigmas[0] * init_eps,
                r_anchor,
            )
            x = torch.where(m_b.unsqueeze(-1), x, torch.zeros_like(x))

            for step_idx in range(num_steps):
                c_tau = float(sigmas[step_idx].item())
                c_next = float(sigmas[step_idx + 1].item())
                gamma = self.edm_schedule.gamma_0 if c_tau > self.edm_schedule.gamma_min else 0.0
                t_hat = c_tau * (1.0 + gamma)
                if gamma > 0.0:
                    churn_std = self.edm_schedule.noise_scale * math.sqrt(
                        max(t_hat * t_hat - c_tau * c_tau, 0.0)
                    )
                    churn_eps = _randn_deterministic(tuple(x.shape), device, generator=gen)
                    x_hat = torch.where(
                        active_flex.unsqueeze(-1), x + churn_std * churn_eps, r_anchor
                    )
                else:
                    x_hat = x

                t_tensor = torch.full((batch_size,), t_hat, device=device, dtype=torch.float32)
                x0_pred = self.denoise(
                    r_noisy=x_hat,
                    noise_level=t_tensor,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    atom_feats=atom_feats,
                    atom_to_token=atom_to_token,
                    atom_mask=m_b,
                    token_mask=token_mask,
                    flex_mask=active_flex,
                    r_ref=r_ref,
                    covalent_adj=covalent_adj,
                )
                delta = (x_hat - x0_pred) / max(t_hat, 1e-8)
                dt = c_next - t_hat
                x_next = x_hat + self.edm_schedule.step_scale * dt * delta
                x = torch.where(active_flex.unsqueeze(-1), x_next, r_anchor)
                x = torch.where(m_b.unsqueeze(-1), x, torch.zeros_like(x))
            return x

        # Rectified Flow Matching ODE integration (Heun 2nd-order or Euler)
        t_grid = self.fm_schedule.get_time_grid(num_steps, device)
        x = torch.where(
            active_flex.unsqueeze(-1),
            self.fm_schedule.sigma_fm * init_eps,
            r_anchor,
        )
        x = torch.where(m_b.unsqueeze(-1), x, torch.zeros_like(x))

        for step_idx in range(num_steps):
            t_cur = float(t_grid[step_idx].item())
            t_nxt = float(t_grid[step_idx + 1].item())
            dt = t_nxt - t_cur
            t_vec = torch.full((batch_size,), t_cur, device=device, dtype=torch.float32)
            x0_pred = self.denoise(
                r_noisy=x,
                noise_level=t_vec,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                atom_feats=atom_feats,
                atom_to_token=atom_to_token,
                atom_mask=m_b,
                token_mask=token_mask,
                flex_mask=active_flex,
                r_ref=r_ref,
                covalent_adj=covalent_adj,
            )
            v1 = self.fm_schedule.velocity_from_x0(x, x0_pred, t_vec)
            x_euler = torch.where(active_flex.unsqueeze(-1), x + dt * v1, r_anchor)
            x_euler = torch.where(m_b.unsqueeze(-1), x_euler, torch.zeros_like(x_euler))

            if self.fm_schedule.use_heun and step_idx < num_steps - 1:
                t_nxt_vec = torch.full((batch_size,), t_nxt, device=device, dtype=torch.float32)
                x0_nxt = self.denoise(
                    r_noisy=x_euler,
                    noise_level=t_nxt_vec,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    atom_feats=atom_feats,
                    atom_to_token=atom_to_token,
                    atom_mask=m_b,
                    token_mask=token_mask,
                    flex_mask=active_flex,
                    r_ref=r_ref,
                    covalent_adj=covalent_adj,
                )
                v2 = self.fm_schedule.velocity_from_x0(x_euler, x0_nxt, t_nxt_vec)
                x = torch.where(active_flex.unsqueeze(-1), x + 0.5 * dt * (v1 + v2), r_anchor)
            else:
                x = x_euler
            x = torch.where(m_b.unsqueeze(-1), x, torch.zeros_like(x))
        return x


@dataclass(frozen=True)
class CoFoldingLossOutput:
    """Breakdown of all 6 physics-grounded co-folding loss components."""

    total: Tensor
    coord: Tensor
    smooth_lddt: Tensor
    bond: Tensor
    chiral: Tensor
    clash: Tensor
    tether: Tensor


class CoFoldingLoss(nn.Module):
    """6-term physics-grounded all-atom protein–ligand co-folding loss.

    Combines:
    1. **Weighted Aligned Coordinate MSE** via reflection-safe :class:`KabschWeightedRigidAlign`,
    2. **Normalized Smooth LDDT** (AF3 Alg. 27; strictly ``0.0`` at ground truth, ``-> 1.0``
       at large distortions, differentiable everywhere),
    3. **1–2 Covalent Bond & 1–3 Geminal Distance Loss**,
    4. **Signed Tetrahedral Chiral Volume Loss** (discriminates ``(R)`` vs ``(S)`` stereocenters),
    5. **Soft Bondi vdW Steric Clash Loss** (using ``atom_feats[..., 45] * 2.0``), and
    6. **Pocket Backbone Anchor Tether Loss**.
    """

    w_coord: float
    w_lddt: float
    w_bond: float
    w_chiral: float
    w_clash: float
    w_tether: float
    clash_scale: float
    tether_tol: float
    _lddt_thresholds: Tensor
    _lddt_max_score: float

    def __init__(
        self,
        *,
        w_coord: float = 1.0,
        w_lddt: float = 1.0,
        w_bond: float = 2.0,
        w_chiral: float = 1.5,
        w_clash: float = 1.0,
        w_tether: float = 0.5,
        clash_scale: float = 0.82,
        tether_tol: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_coord = float(w_coord)
        self.w_lddt = float(w_lddt)
        self.w_bond = float(w_bond)
        self.w_chiral = float(w_chiral)
        self.w_clash = float(w_clash)
        self.w_tether = float(w_tether)
        self.clash_scale = float(clash_scale)
        self.tether_tol = float(tether_tol)
        self.kabsch = KabschWeightedRigidAlign()

        thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
        self.register_buffer("_lddt_thresholds", thresholds)
        self._lddt_max_score = float(torch.sigmoid(thresholds).mean().item())

    def _compute_smooth_lddt(
        self,
        d_pred: Tensor,
        d_true: Tensor,
        atom_mask: Tensor,
        is_ligand: Tensor | None,
    ) -> Tensor:
        """Compute normalized Smooth LDDT loss in ``[0, 1]`` (``0.0`` at exact ground truth)."""
        num_atoms = d_pred.shape[1]
        if num_atoms < 2:
            return d_pred.new_zeros(())
        m_b = atom_mask.to(torch.bool)
        eye = torch.eye(num_atoms, device=d_pred.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = m_b.unsqueeze(2) & m_b.unsqueeze(1) & (~eye)

        if is_ligand is not None:
            lig_b = is_ligand.to(torch.bool)
            inter = (lig_b.unsqueeze(2) != lig_b.unsqueeze(1)) & valid_pair
            radius_ok = torch.where(inter, d_true < 30.0, d_true < 15.0)
        else:
            radius_ok = d_true < 15.0
        incl = valid_pair & radius_ok

        # Smooth absolute distance error so exact ground truth yields zero distortion.
        diff = d_pred - d_true
        abs_err = torch.sqrt(diff * diff + 1e-10) - 1e-5
        abs_err = abs_err.clamp(min=0.0)

        thresh = self._lddt_thresholds.to(device=d_pred.device, dtype=torch.float32)
        raw_score = torch.sigmoid(thresh.view(1, 1, 1, 4) - abs_err.unsqueeze(-1)).mean(dim=-1)
        norm_score = (raw_score / self._lddt_max_score).clamp(min=0.0, max=1.0)

        pair_loss = torch.where(incl, 1.0 - norm_score, torch.zeros_like(norm_score))
        denom = incl.to(torch.float32).sum().clamp(min=1.0)
        has_any = incl.any()
        return torch.where(has_any, pair_loss.sum() / denom, d_pred.new_zeros(()))

    @staticmethod
    def _pair_distance_mse(
        coords: Tensor,
        edge_idx: Tensor,
        ref_dist: Tensor,
        edge_mask: Tensor | None,
    ) -> Tensor:
        """Masked MSE between predicted pair distances and ideal reference distances."""
        batch_size, num_edges, _ = edge_idx.shape
        if batch_size == 0 or num_edges == 0:
            return coords.new_zeros(())
        num_atoms = coords.shape[1]
        safe_idx = edge_idx.to(torch.long).clamp(min=0, max=max(num_atoms - 1, 0))
        i_idx = safe_idx[..., 0].unsqueeze(-1).expand(-1, -1, 3)
        j_idx = safe_idx[..., 1].unsqueeze(-1).expand(-1, -1, 3)
        ri = torch.gather(coords, dim=1, index=i_idx)
        rj = torch.gather(coords, dim=1, index=j_idx)
        diff = ri - rj
        d_pred = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
        sq_err = (d_pred - ref_dist.to(torch.float32)) ** 2
        if edge_mask is not None:
            m_b = edge_mask.to(torch.bool)
            sq_err = torch.where(m_b, sq_err, torch.zeros_like(sq_err))
            denom = m_b.to(torch.float32).sum().clamp(min=1.0)
            return torch.where(m_b.any(), sq_err.sum() / denom, coords.new_zeros(()))
        return sq_err.mean()

    def forward(
        self,
        x_pred: Tensor,
        x_true: Tensor,
        atom_mask: Tensor,
        *,
        atom_weights: Tensor | None = None,
        loss_weight: Tensor | None = None,
        align: bool = True,
        is_ligand: Tensor | None = None,
        atom_feats: Tensor | None = None,
        vdw_radii: Tensor | None = None,
        bond_12_idx: Tensor | None = None,
        bond_12_ref: Tensor | None = None,
        bond_12_mask: Tensor | None = None,
        bond_13_idx: Tensor | None = None,
        bond_13_ref: Tensor | None = None,
        bond_13_mask: Tensor | None = None,
        chiral_quads: Tensor | None = None,
        chiral_vol_true: Tensor | None = None,
        chiral_mask: Tensor | None = None,
        nonbonded_mask: Tensor | None = None,
        backbone_mask: Tensor | None = None,
        x_input: Tensor | None = None,
    ) -> CoFoldingLossOutput:
        """Evaluate total physics co-folding loss and per-term breakdown."""
        pred_f = x_pred.to(torch.float32)
        true_f = x_true.to(torch.float32)
        m_b = atom_mask.to(torch.bool)
        batch_size, num_atoms, _ = pred_f.shape
        device = pred_f.device
        zero = pred_f.new_zeros(())

        # 1. Weighted Aligned Coordinate MSE
        if atom_weights is None:
            if is_ligand is not None:
                w = torch.where(
                    is_ligand.to(torch.bool),
                    torch.full((batch_size, num_atoms), 10.0, device=device, dtype=torch.float32),
                    torch.ones((batch_size, num_atoms), device=device, dtype=torch.float32),
                )
            else:
                w = torch.ones((batch_size, num_atoms), device=device, dtype=torch.float32)
        else:
            w = atom_weights.to(torch.float32).clamp(min=0.0)
        w = torch.where(m_b, w, torch.zeros_like(w))

        if align:
            # Align ground truth onto prediction (detached transform, AF3 Alg. 28).
            target_aligned, _, _ = self.kabsch(true_f, pred_f, w, detach_transform=True)
        else:
            target_aligned = true_f

        sq_coord = ((pred_f - target_aligned) ** 2).sum(dim=-1)  # [B, N]
        w_sum_b = w.sum(dim=-1).clamp(min=1e-8)
        per_batch_mse = (w * sq_coord).sum(dim=-1) / (3.0 * w_sum_b)  # [B]
        if loss_weight is not None:
            per_batch_mse = per_batch_mse * loss_weight.to(torch.float32)
        has_valid_b = m_b.any(dim=-1)
        loss_coord = torch.where(
            has_valid_b, per_batch_mse, torch.zeros_like(per_batch_mse)
        ).sum() / has_valid_b.to(torch.float32).sum().clamp(min=1.0)

        # Pairwise distance matrices (reused by Smooth LDDT and Bondi vdW Clash)
        diff_p = pred_f.unsqueeze(2) - pred_f.unsqueeze(1)
        d_pred = torch.sqrt((diff_p * diff_p).sum(dim=-1) + 1e-8)
        diff_t = true_f.unsqueeze(2) - true_f.unsqueeze(1)
        d_true = torch.sqrt((diff_t * diff_t).sum(dim=-1) + 1e-8)

        # 2. Smooth LDDT Loss
        loss_lddt = self._compute_smooth_lddt(d_pred, d_true, m_b, is_ligand)

        # 3. Covalent 1-2 Bond & 1-3 Angle Distance Loss
        loss_bond = zero
        if bond_12_idx is not None and bond_12_ref is not None:
            loss_bond = loss_bond + self._pair_distance_mse(
                pred_f, bond_12_idx, bond_12_ref, bond_12_mask
            )
        if bond_13_idx is not None and bond_13_ref is not None:
            loss_bond = loss_bond + 0.5 * self._pair_distance_mse(
                pred_f, bond_13_idx, bond_13_ref, bond_13_mask
            )

        # 4. Signed Tetrahedral Chiral Volume Loss
        if chiral_quads is not None and chiral_quads.shape[1] > 0:
            vol_pred = compute_signed_tetrahedral_volume(pred_f, chiral_quads)
            vol_true = (
                compute_signed_tetrahedral_volume(true_f, chiral_quads).detach()
                if chiral_vol_true is None
                else chiral_vol_true.to(torch.float32)
            )
            # Cap sign-preservation margin by 0.25 * vol_true^2 so exact ground truth always
            # attains chiral_loss == 0.0 even for small tetrahedral volumes (|V_true| < 0.5).
            sign_margin = (0.25 * vol_true * vol_true).clamp(max=0.25)
            c_loss_raw = F.huber_loss(vol_pred, vol_true, reduction="none") + F.relu(
                -vol_true * vol_pred + sign_margin
            )
            if chiral_mask is not None:
                cm = chiral_mask.to(torch.bool)
                c_loss_raw = torch.where(cm, c_loss_raw, torch.zeros_like(c_loss_raw))
                loss_chiral = torch.where(
                    cm.any(), c_loss_raw.sum() / cm.to(torch.float32).sum().clamp(min=1.0), zero
                )
            else:
                loss_chiral = c_loss_raw.mean()
        else:
            loss_chiral = zero

        # 5. Soft Bondi vdW Steric Clash Loss
        if num_atoms >= 2:
            if vdw_radii is not None:
                r_vdw = vdw_radii.to(torch.float32).clamp(min=0.5, max=3.5)
            elif atom_feats is not None and atom_feats.shape[-1] > 45:
                r_vdw = (atom_feats[..., 45].to(torch.float32) * 2.0).clamp(min=0.5, max=3.5)
            else:
                r_vdw = torch.full(
                    (batch_size, num_atoms), 1.70, device=device, dtype=torch.float32
                )

            eye = torch.eye(num_atoms, device=device, dtype=torch.bool).unsqueeze(0)
            valid_pair = m_b.unsqueeze(2) & m_b.unsqueeze(1) & (~eye)
            if nonbonded_mask is not None:
                nb_mask = valid_pair & nonbonded_mask.to(torch.bool)
            else:
                # Exclude short covalent 1-2/1-3 pairs cleanly using ground-truth separation.
                nb_mask = valid_pair & (d_true >= 2.2)

            min_allowed = self.clash_scale * (r_vdw.unsqueeze(2) + r_vdw.unsqueeze(1))
            overlap = F.relu(min_allowed - d_pred)
            clash_sq = torch.where(nb_mask, overlap * overlap, torch.zeros_like(overlap))
            loss_clash = torch.where(
                nb_mask.any(),
                clash_sq.sum() / nb_mask.to(torch.float32).sum().clamp(min=1.0),
                zero,
            )
        else:
            loss_clash = zero

        # 6. Pocket Backbone Anchor Tether Loss
        if backbone_mask is not None and x_input is not None:
            bb_m = m_b & backbone_mask.to(torch.bool)
            disp = pred_f - x_input.to(torch.float32)
            disp_norm = torch.sqrt((disp * disp).sum(dim=-1) + 1e-8)
            excess = F.relu(disp_norm - self.tether_tol)
            tether_sq = torch.where(bb_m, excess * excess, torch.zeros_like(excess))
            loss_tether = torch.where(
                bb_m.any(),
                tether_sq.sum() / bb_m.to(torch.float32).sum().clamp(min=1.0),
                zero,
            )
        else:
            loss_tether = zero

        total = (
            self.w_coord * loss_coord
            + self.w_lddt * loss_lddt
            + self.w_bond * loss_bond
            + self.w_chiral * loss_chiral
            + self.w_clash * loss_clash
            + self.w_tether * loss_tether
        )
        return CoFoldingLossOutput(
            total=total,
            coord=loss_coord,
            smooth_lddt=loss_lddt,
            bond=loss_bond,
            chiral=loss_chiral,
            clash=loss_clash,
            tether=loss_tether,
        )


__all__ = [
    "AllAtomDiffusionModule",
    "AtomAttentionDecoder",
    "AtomAttentionEncoder",
    "AtomLocalAttentionBlock",
    "CoFoldingLoss",
    "CoFoldingLossOutput",
    "DiffusionMode",
    "DiffusionTransformer",
    "EDMDiffusionSchedule",
    "FlexMode",
    "FlowMatchingSchedule",
    "FourierEmbedding",
    "KabschWeightedRigidAlign",
    "compute_signed_tetrahedral_volume",
]
