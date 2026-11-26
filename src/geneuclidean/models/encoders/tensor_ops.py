"""Zero-e3nn Cartesian 3x3 Irrep Algebra and SE(3)/O(3) Vector-Neuron Primitives.

Implements Clebsch-Gordan-free Cartesian rank-2 tensor algebra (TensorNet / GotenNet /
CartFormer-SE3 lineage) alongside equivariant Vector-Neuron (VN) layers (Deng et al., ICCV
2021; Assaad et al., VN-Transformer TMLR 2023) strictly in native ``torch.float32`` tensor ops
for maximum Apple Silicon MPS / CUDA / CPU SIMD throughput.

Irreducible :math:`O(3)` decomposition of any :math:`3 \\times 3` Cartesian tensor
:math:`\\mathbf{T} \\in \\mathbb{R}^{3 \\times 3}`:

.. math::
    \\mathbf{T} = I(\\mathbf{T}) + A(\\mathbf{T}) + S(\\mathbf{T}),

where :math:`I(\\mathbf{T}) = \\tfrac{1}{3}\\operatorname{tr}(\\mathbf{T})\\mathbf{I}_3` (``0e``),
:math:`A(\\mathbf{T}) = \\tfrac{1}{2}(\\mathbf{T} - \\mathbf{T}^\\top)` (``1e`` axial), and
:math:`S(\\mathbf{T}) = \\tfrac{1}{2}(\\mathbf{T} + \\mathbf{T}^\\top) - I(\\mathbf{T})` (``2e``
quadrupole).

All three subspaces are mutually Frobenius-orthogonal
(:math:`\\|\\mathbf{T}\\|_F^2 = \\|I\\|_F^2 + \\|A\\|_F^2 + \\|S\\|_F^2`), equivariant under
``T -> R T R^T`` for all :math:`\\mathbf{R} \\in O(3)`,
and couple via hardware-native :math:`3 \\times 3` matrix multiplication (27 FLOPs/channel).
Under improper reflections (:math:`\\det\\mathbf{P} = -1`), the Hodge dual
:math:`\\boldsymbol{\\omega} = \\star A = [A_{21}, A_{02}, A_{10}]^\\top` and cross product
:math:`\\mathbf{a} \\times \\mathbf{b}` transform as axial pseudovectors
(:math:`\\boldsymbol{\\omega} \\mapsto -\\mathbf{P}\\boldsymbol{\\omega}`), enabling exact
:math:`SE(3)` chiral discrimination of :math:`(R)/(S)` and :math:`L/D` molecular geometries.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from geneuclidean.models.encoders.dense_knn import safe_masked_softmax


def decompose_rank2_ias(tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose rank-2 Cartesian tensor(s) ``[..., 3, 3]`` into orthogonal ``(I, A, S)`` irreps.

    Returns:
        Tuple ``(trace_part, antisym_part, sym_traceless_part)``, each of shape
        ``[..., 3, 3]``, satisfying ``trace_part + antisym_part + sym_traceless_part == tensor``:
          - ``trace_part`` (``0e``): ``(tr(T) / 3) * I_3``
          - ``antisym_part`` (``1e``): ``0.5 * (T - T^T)``
          - ``sym_traceless_part`` (``2e``): ``0.5 * (T + T^T) - I(T)``
    """
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError(f"Expected tensor with trailing shape (3, 3), got {tuple(tensor.shape)}")

    tr = tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]
    eye = torch.eye(3, device=tensor.device, dtype=tensor.dtype)
    trace_part = (tr / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
    t_trans = tensor.transpose(-1, -2)
    antisym_part = 0.5 * (tensor - t_trans)
    sym_traceless_part = 0.5 * (tensor + t_trans) - trace_part
    return trace_part, antisym_part, sym_traceless_part


def compose_rank2_ias(
    trace_part: Tensor,
    antisym_part: Tensor,
    sym_traceless_part: Tensor,
) -> Tensor:
    """Recompose ``(I, A, S)`` irreps into a full rank-2 Cartesian tensor ``[..., 3, 3]``.

    Accepts ``trace_part`` either as the full isotropic matrix ``[..., 3, 3]`` emitted by
    :func:`decompose_rank2_ias` OR as the raw scalar trace ``[...]`` (which is expanded as
    ``(trace / 3) * I_3``).
    """
    if antisym_part.ndim < 2 or antisym_part.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected antisym_part trailing shape (3, 3), got {tuple(antisym_part.shape)}"
        )
    if sym_traceless_part.shape != antisym_part.shape:
        raise ValueError(
            f"sym_traceless_part shape {tuple(sym_traceless_part.shape)} "
            f"!= antisym_part shape {tuple(antisym_part.shape)}"
        )

    if trace_part.shape == antisym_part.shape:
        iso_mat = trace_part
    elif trace_part.shape == antisym_part.shape[:-2]:
        eye = torch.eye(3, device=antisym_part.device, dtype=antisym_part.dtype)
        iso_mat = (trace_part / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
    else:
        raise ValueError(
            f"trace_part shape {tuple(trace_part.shape)} must match either "
            f"{tuple(antisym_part.shape)} or {tuple(antisym_part.shape[:-2])}"
        )
    return iso_mat + antisym_part + sym_traceless_part


def vector_to_skew_matrix(vec: Tensor) -> Tensor:
    """Map 3D axial vector(s) ``[..., 3]`` to skew-symmetric matrices ``[..., 3, 3]``.

    Follows the canonical :math:`\\mathfrak{so}(3)` hat-map convention:

    .. math::
        [\\boldsymbol{\\omega}]_\\times =
        \\begin{pmatrix}
            0 & -\\omega_z & \\omega_y \\\\
            \\omega_z & 0 & -\\omega_x \\\\
            -\\omega_y & \\omega_x & 0
        \\end{pmatrix}, \\qquad
        [\\boldsymbol{\\omega}]_\\times \\mathbf{u} = \\boldsymbol{\\omega} \\times \\mathbf{u}.

    Exact inverse of :func:`skew_matrix_to_vector` on skew-symmetric matrices.
    """
    if vec.ndim < 1 or vec.shape[-1] != 3:
        raise ValueError(f"Expected vec trailing dim 3, got {tuple(vec.shape)}")

    wx = vec[..., 0]
    wy = vec[..., 1]
    wz = vec[..., 2]
    zero = torch.zeros_like(wx)
    row0 = torch.stack((zero, -wz, wy), dim=-1)
    row1 = torch.stack((wz, zero, -wx), dim=-1)
    row2 = torch.stack((-wy, wx, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def skew_matrix_to_vector(skew: Tensor) -> Tensor:
    """Extract the Hodge-dual 3D axial vector ``[..., 3]`` from ``[..., 3, 3]`` matrix(es).

    Computes ``0.5 * [A_21 - A_12, A_02 - A_20, A_10 - A_01]``,
    which equals :math:`[A_{21}, A_{02}, A_{10}]^\\top` on skew-symmetric matrices and safely
    projects out any symmetric part on general rank-2 inputs. Under improper reflections
    (:math:`\\det\\mathbf{P} = -1`), transforms as an axial pseudovector:
    :math:`\\star(\\mathbf{P} A \\mathbf{P}^\\top) = -\\mathbf{P}(\\star A)`.
    """
    if skew.ndim < 2 or skew.shape[-2:] != (3, 3):
        raise ValueError(f"Expected skew trailing shape (3, 3), got {tuple(skew.shape)}")

    wx = 0.5 * (skew[..., 2, 1] - skew[..., 1, 2])
    wy = 0.5 * (skew[..., 0, 2] - skew[..., 2, 0])
    wz = 0.5 * (skew[..., 1, 0] - skew[..., 0, 1])
    return torch.stack((wx, wy, wz), dim=-1)


def outer_traceless(vec_a: Tensor, vec_b: Tensor | None = None) -> Tensor:
    """Construct an exact ``O(3)``-equivariant symmetric traceless (``2e`` quadrupole) tensor.

    Args:
        vec_a: Polar vector(s) of shape ``[..., 3]``.
        vec_b: Optional second polar vector(s) broadcastable to ``vec_a``. If ``None``,
            computes ``a a^T - (1/3) ||a||^2 I_3``; otherwise computes
            ``0.5 * (a b^T + b a^T) - (1/3) <a, b> I_3``.

    Returns:
        Symmetric traceless tensor of shape ``[..., 3, 3]``.
    """
    if vec_a.ndim < 1 or vec_a.shape[-1] != 3:
        raise ValueError(f"Expected vec_a trailing dim 3, got {tuple(vec_a.shape)}")

    eye = torch.eye(3, device=vec_a.device, dtype=vec_a.dtype)
    if vec_b is None:
        outer = vec_a.unsqueeze(-1) * vec_a.unsqueeze(-2)
        dot = (vec_a * vec_a).sum(dim=-1)
        return outer - (dot / 3.0).unsqueeze(-1).unsqueeze(-1) * eye

    if vec_b.ndim < 1 or vec_b.shape[-1] != 3:
        raise ValueError(f"Expected vec_b trailing dim 3, got {tuple(vec_b.shape)}")

    a_b = vec_a.unsqueeze(-1) * vec_b.unsqueeze(-2)
    sym_outer = 0.5 * (a_b + a_b.transpose(-1, -2))
    dot = (vec_a * vec_b).sum(dim=-1)
    return sym_outer - (dot / 3.0).unsqueeze(-1).unsqueeze(-1) * eye


def chiral_cross_product(
    vec_a: Tensor,
    vec_b: Tensor,
    *,
    normalize: bool = True,
    eps: float = 1.0,
) -> Tensor:
    """Compute a pure-elementwise 3D cross product with optional smooth norm bounding.

    Implemented directly via 3D Levi-Civita components so it broadcasts cleanly and never
    hits platform-specific linear-algebra fallbacks on Apple Silicon MPS.

    Under any :math:`\\mathbf{R} \\in O(3)`,
    ``(R a) x (R b) = det(R) * R (a x b)``,
    preserving exact :math:`SO(3)` equivariance while flipping sign under improper mirror
    reflections (:math:`\\det\\mathbf{P} = -1`).

    Args:
        vec_a: First 3D vector tensor of shape ``[..., 3]``.
        vec_b: Second 3D vector tensor broadcastable to ``vec_a`` with trailing dim ``3``.
        normalize: If ``True`` (default), smoothly bound norm to ``< 1`` via
            ``(a x b) / sqrt(||a x b||^2 + eps)``
            for ``C^inf`` stability on collinear inputs and bounded ``float32`` activations.
        eps: Positive smoothing denominator constant (default ``1.0``).

    Returns:
        Axial 3D vector tensor of shape ``[..., 3]``.
    """
    if vec_a.ndim < 1 or vec_a.shape[-1] != 3:
        raise ValueError(f"Expected vec_a trailing dim 3, got {tuple(vec_a.shape)}")
    if vec_b.ndim < 1 or vec_b.shape[-1] != 3:
        raise ValueError(f"Expected vec_b trailing dim 3, got {tuple(vec_b.shape)}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    a0, a1, a2 = vec_a[..., 0], vec_a[..., 1], vec_a[..., 2]
    b0, b1, b2 = vec_b[..., 0], vec_b[..., 1], vec_b[..., 2]
    cross = torch.stack(
        (a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0),
        dim=-1,
    )
    if not normalize:
        return cross
    sq_norm = (cross * cross).sum(dim=-1, keepdim=True)
    return cross * torch.rsqrt(sq_norm + float(eps))


class TensorEquivariantRMSNorm(nn.Module):
    """O(3)-equivariant Frobenius RMS normalization for rank-2 ``[..., C_t, 3, 3]`` tensors.

    Because the Frobenius norm ``||R T R^T||_F == ||T||_F``
    is strictly :math:`O(3)`-invariant, dividing by the channel RMS (or per-channel Frobenius norm)
    keeps tensor magnitudes :math:`O(1)` in ``float32`` without breaking equivariance and preserves
    exact ``0.0`` on masked dummy atoms.
    """

    num_channels: int
    eps: float
    per_channel: bool

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.per_channel = bool(per_channel)
        self.weight = nn.Parameter(torch.ones(self.num_channels, dtype=torch.float32))

    def forward(self, tensor: Tensor) -> Tensor:
        """Normalize rank-2 Cartesian tensor features ``[..., C_t, 3, 3]``."""
        if tensor.ndim < 3 or tensor.shape[-3:] != (self.num_channels, 3, 3):
            raise ValueError(
                f"Expected tensor trailing shape ({self.num_channels}, 3, 3), "
                f"got {tuple(tensor.shape)}"
            )
        sq_frob = (tensor * tensor).sum(dim=(-2, -1), keepdim=True)  # [..., C_t, 1, 1]
        if self.per_channel:
            inv_rms = torch.rsqrt(sq_frob + self.eps)
        else:
            mean_sq = sq_frob.mean(dim=-3, keepdim=True)  # [..., 1, 1, 1]
            inv_rms = torch.rsqrt(mean_sq + self.eps)
        scale = self.weight.view(*([1] * (tensor.ndim - 3)), self.num_channels, 1, 1)
        return (tensor * inv_rms) * scale


class VectorNeuronLinear(nn.Module):
    """Bias-free ``O(3)``-equivariant linear mixing over 3D vector channels ``[..., C, 3]``."""

    in_channels: int
    out_channels: int

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"in_channels ({in_channels}) and out_channels ({out_channels}) must be >= 1"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, vec: Tensor) -> Tensor:
        """Apply channel mixing ``[..., C_in, 3] -> [..., C_out, 3]`` identically across xyz."""
        if vec.ndim < 2 or vec.shape[-2:] != (self.in_channels, 3):
            raise ValueError(
                f"Expected vec trailing shape ({self.in_channels}, 3), got {tuple(vec.shape)}"
            )
        out: Tensor = self.linear(vec.transpose(-1, -2)).transpose(-1, -2)
        return out


class VectorNeuronLeakyReLU(nn.Module):
    """Exact ``O(3)``-equivariant Vector-Neuron LeakyReLU activation (Deng et al., ICCV 2021).

    Projects vector channels against learned equivariant half-space directions
    :math:`\\mathbf{k} = \\mathbf{W}_k \\mathbf{v}`:

    .. math::
        \\mathrm{VN\\text{-}LReLU}(\\mathbf{v}) =
        \\mathbf{v} - (1 - \\alpha)\\,\\frac{\\min(\\langle \\mathbf{v}, \\mathbf{k}\\rangle, 0)}
        {\\|\\mathbf{k}\\|_2^2 + \\varepsilon}\\,\\mathbf{k}.

    Strictly preserves ``O(3)`` equivariance, maps zero vectors to exact ``0.0``, and remains
    smoothly differentiable at ``k = 0``.
    """

    in_channels: int
    share_nonlinearity: bool
    negative_slope: float
    eps: float

    def __init__(
        self,
        in_channels: int,
        *,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.in_channels = int(in_channels)
        self.share_nonlinearity = bool(share_nonlinearity)
        self.negative_slope = float(negative_slope)
        self.eps = float(eps)
        dir_channels = 1 if self.share_nonlinearity else self.in_channels
        self.dir_proj = VectorNeuronLinear(self.in_channels, dir_channels)

    def forward(self, vec: Tensor) -> Tensor:
        """Apply equivariant half-space LeakyReLU to ``[..., C, 3]``."""
        k: Tensor = self.dir_proj(vec)  # [..., 1 or C, 3]
        dot = (vec * k).sum(dim=-1, keepdim=True)  # [..., C, 1]
        k_sq = (k * k).sum(dim=-1, keepdim=True)  # [..., 1 or C, 1]
        neg_part = torch.clamp(dot, max=0.0)
        return vec - (1.0 - self.negative_slope) * (neg_part / (k_sq + self.eps)) * k


class VectorNeuronLayerNorm(nn.Module):
    """Exact ``O(3)``-equivariant Vector-Neuron LayerNorm (Assaad et al., TMLR 2023).

    Computes rotation-invariant per-channel magnitudes
    :math:`n_c = \\sqrt{\\|\\mathbf{v}_c\\|_2^2 + \\varepsilon}`, normalizes them across channels
    with affine ``LayerNorm`` (initialized with ``weight = 1, bias = 1`` so uniform-norm inputs
    stay non-degenerate), and rescales each 3D vector along its exact direction while keeping
    zeroed dummy vectors strictly ``0.0``.
    """

    num_channels: int
    eps: float

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.norm: nn.LayerNorm | None
        if self.num_channels > 1:
            self.norm = nn.LayerNorm(self.num_channels, eps=self.eps)
            nn.init.ones_(self.norm.bias)
            self.scalar_weight = None
        else:
            self.norm = None
            self.scalar_weight = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, vec: Tensor) -> Tensor:
        """Apply equivariant LayerNorm to vector tensor ``[..., C, 3]``."""
        if vec.ndim < 2 or vec.shape[-2:] != (self.num_channels, 3):
            raise ValueError(
                f"Expected vec trailing shape ({self.num_channels}, 3), got {tuple(vec.shape)}"
            )
        sq_norm = (vec * vec).sum(dim=-1)  # [..., C]
        norms = torch.sqrt(sq_norm + self.eps)  # [..., C]
        unit_dir = vec / norms.unsqueeze(-1)  # exact 0 when vec == 0
        if self.norm is not None:
            normed_mag: Tensor = self.norm(norms)
            return unit_dir * normed_mag.unsqueeze(-1)
        assert self.scalar_weight is not None
        return unit_dir * self.scalar_weight.view(*([1] * (vec.ndim - 1)), 1)


class VectorNeuronAttentionBlock(nn.Module):
    """Composable ``SE(3)``/``O(3)``-equivariant Vector-Neuron Multi-Head Attention block.

    Implements Frobenius inner-product multi-head vector attention (VN-Transformer, Assaad et al.
    TMLR 2023) with optional :math:`SE(3)`-invariant external attention bias (e.g.
    :class:`~geneuclidean.models.encoders.electrostatics.ScreenedCoulombBias` or
    :class:`~geneuclidean.models.encoders.electrostatics.GeoTransformerPairTripletBias`) and an
    optional bounded chiral cross-product branch breaking mirror symmetry (``det(P) = -1``)
    when ``use_se3_chirality=True``.
    """

    vector_dim: int
    num_heads: int
    head_dim: int
    use_se3_chirality: bool

    def __init__(
        self,
        vector_dim: int,
        num_heads: int = 4,
        *,
        use_se3_chirality: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if vector_dim < 1 or num_heads < 1:
            raise ValueError(f"vector_dim ({vector_dim}) and num_heads ({num_heads}) must be >= 1")
        if vector_dim % num_heads != 0:
            raise ValueError(
                f"vector_dim ({vector_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.vector_dim = int(vector_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.vector_dim // self.num_heads
        self.use_se3_chirality = bool(use_se3_chirality)

        self.norm1 = VectorNeuronLayerNorm(self.vector_dim, eps=eps)
        self.q_proj = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.k_proj = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.v_proj = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.out_proj = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = VectorNeuronLayerNorm(self.vector_dim, eps=eps)
        self.ffn_in = VectorNeuronLinear(self.vector_dim, self.vector_dim)
        self.ffn_act = VectorNeuronLeakyReLU(self.vector_dim, eps=eps)
        self.ffn_out = VectorNeuronLinear(self.vector_dim, self.vector_dim)

        self.chiral_a: VectorNeuronLinear | None
        self.chiral_b: VectorNeuronLinear | None
        self.chiral_scale: nn.Parameter | None
        if self.use_se3_chirality:
            self.chiral_a = VectorNeuronLinear(self.vector_dim, self.vector_dim)
            self.chiral_b = VectorNeuronLinear(self.vector_dim, self.vector_dim)
            self.chiral_scale = nn.Parameter(
                torch.full((self.vector_dim,), 0.25, dtype=torch.float32)
            )
        else:
            self.chiral_a = None
            self.chiral_b = None
            self.chiral_scale = None

    def forward(
        self,
        vec: Tensor,
        mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        """Apply equivariant VN self-attention and VN-FFN to ``vec [B, N, C_v, 3]``.

        Args:
            vec: Input 3D vector features of shape ``[B, N, vector_dim, 3]``.
            mask: Optional boolean validity mask of shape ``[B, N]``.
            attn_bias: Optional ``SE(3)``-invariant attention bias broadcastable to
                ``[B, num_heads, N, N]``.

        Returns:
            Updated 3D vector features of shape ``[B, N, vector_dim, 3]`` strictly zeroed
            on masked dummy slots.
        """
        if vec.ndim != 4 or vec.shape[-2:] != (self.vector_dim, 3):
            raise ValueError(
                f"Expected vec shape [B, N, {self.vector_dim}, 3], got {tuple(vec.shape)}"
            )
        batch_size, num_tokens, _, _ = vec.shape
        if mask is None:
            mask_bool = torch.ones((batch_size, num_tokens), device=vec.device, dtype=torch.bool)
        else:
            if mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != vec[:2] {(batch_size, num_tokens)}"
                )
            mask_bool = mask.to(torch.bool)

        mask_4d = mask_bool.unsqueeze(-1).unsqueeze(-1)
        zeros = torch.zeros_like(vec)
        safe_vec = torch.where(mask_4d, vec, zeros)

        h = torch.where(mask_4d, self.norm1(safe_vec), zeros)
        q = self.q_proj(h).reshape(batch_size, num_tokens, self.num_heads, self.head_dim * 3)
        k = self.k_proj(h).reshape(batch_size, num_tokens, self.num_heads, self.head_dim * 3)
        v = self.v_proj(h).reshape(batch_size, num_tokens, self.num_heads, self.head_dim, 3)

        # Transpose to [B, H, N, ...] for Frobenius inner-product attention.
        q_h = q.permute(0, 2, 1, 3)  # [B, H, N, 3*d_v]
        k_h = k.permute(0, 2, 1, 3)  # [B, H, N, 3*d_v]
        v_h = v.permute(0, 2, 1, 3, 4)  # [B, H, N, d_v, 3]

        scale = 1.0 / math.sqrt(3.0 * self.head_dim)
        logits = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # [B, H, N, N]
        if attn_bias is not None:
            logits = logits + attn_bias

        key_mask = mask_bool.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn = safe_masked_softmax(logits, key_mask, dim=-1)
        attn = self.attn_dropout(attn)

        agg = torch.einsum("bhij,bhjda->bhida", attn, v_h)  # [B, H, N, d_v, 3]
        agg = agg.permute(0, 2, 1, 3, 4).reshape(batch_size, num_tokens, self.vector_dim, 3)
        vec_mid = torch.where(mask_4d, safe_vec + self.out_proj(agg), zeros)

        h2 = torch.where(mask_4d, self.norm2(vec_mid), zeros)
        ffn_delta = self.ffn_out(self.ffn_act(self.ffn_in(h2)))
        if self.use_se3_chirality:
            assert (
                self.chiral_a is not None
                and self.chiral_b is not None
                and self.chiral_scale is not None
            )
            v_cross = chiral_cross_product(
                self.chiral_a(h2),
                self.chiral_b(h2),
                normalize=True,
                eps=1.0,
            )
            ffn_delta = ffn_delta + self.chiral_scale.view(1, 1, self.vector_dim, 1) * v_cross

        return torch.where(mask_4d, vec_mid + ffn_delta, zeros)


__all__ = [
    "TensorEquivariantRMSNorm",
    "VectorNeuronAttentionBlock",
    "VectorNeuronLayerNorm",
    "VectorNeuronLeakyReLU",
    "VectorNeuronLinear",
    "chiral_cross_product",
    "compose_rank2_ias",
    "decompose_rank2_ias",
    "outer_traceless",
    "skew_matrix_to_vector",
    "vector_to_skew_matrix",
]
