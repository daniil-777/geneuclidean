"""Modern e3nn (v0.6+) Spherical-Harmonics Tensor-Product 3D pocket encoder.

Honors GenEuclidean's original 2021 E(3)-equivariant irreps lineage while upgrading to
modern ``e3nn==0.6.0`` (`o3.SphericalHarmonics`, `o3.FullyConnectedTensorProduct`,
`e3nn.nn.Gate`, `o3.Linear`), smooth Bessel RBF envelopes on a 5.5 Å radius graph,
equivariant per-irrep RMSNorm, optional chiral pseudoscalar/pseudovector irreps
(``0o`` / ``1e``), and strict ``float32`` buffers for Apple Silicon MPS compatibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Gate
from torch import Tensor

from geneuclidean.models.encoders.rbf import (
    BesselRBF,
    CosineCutoff,
    MaskedAttentionPool,
    PocketEncoderOutput,
    build_radius_knn_graph,
)


def _build_hidden_irreps(
    lmax: int,
    scalar_channels: int,
    vector_channels: int,
    use_se3_chirality: bool,
) -> tuple[o3.Irreps, o3.Irreps, o3.Irreps]:
    """Construct scalar, gated higher-order, and combined hidden ``o3.Irreps``."""
    c0 = max(8, min(32, scalar_channels))
    c1 = max(4, min(16, vector_channels))
    c2 = max(4, min(8, vector_channels // 2))
    c3 = max(2, min(4, vector_channels // 4))

    scalar_parts: list[tuple[int, tuple[int, int]]] = [(c0, (0, 1))]
    gated_parts: list[tuple[int, tuple[int, int]]] = []

    if lmax >= 1:
        gated_parts.append((c1, (1, -1)))  # 1o polar vectors
        if use_se3_chirality:
            c0o = max(4, c0 // 4)
            c1e = max(4, c1 // 2)
            scalar_parts.append((c0o, (0, -1)))  # 0o chiral pseudoscalars
            gated_parts.append((c1e, (1, 1)))  # 1e chiral axial vectors
    if lmax >= 2:
        gated_parts.append((c2, (2, 1)))  # 2e
    if lmax >= 3:
        gated_parts.append((c3, (3, -1)))  # 3o

    irreps_scalars = o3.Irreps(scalar_parts).simplify()
    irreps_gated = o3.Irreps(gated_parts).simplify()
    irreps_hidden = (irreps_scalars + irreps_gated).simplify()
    return irreps_scalars, irreps_gated, irreps_hidden


class IrrepsEquivariantRMSNorm(nn.Module):
    """Per-irrep O(3)-equivariant RMS normalization for arbitrary ``o3.Irreps``."""

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-4) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = float(eps)
        self._slices: list[tuple[int, int, int, int]] = []
        offset = 0
        weights: list[Tensor] = []
        for mul, ir in self.irreps:
            ir_dim = ir.dim
            width = mul * ir_dim
            self._slices.append((offset, offset + width, mul, ir_dim))
            weights.append(torch.ones(mul, dtype=torch.float32))
            offset += width
        self.weights = nn.ParameterList([nn.Parameter(w) for w in weights])

    def forward(self, x: Tensor) -> Tensor:
        """Normalize each irrep block by its invariant RMS norm across channels."""
        chunks: list[Tensor] = []
        for (start, end, mul, ir_dim), weight in zip(self._slices, self.weights, strict=True):
            block = x[..., start:end].reshape(*x.shape[:-1], mul, ir_dim)
            sq_norm = (block * block).sum(dim=-1, keepdim=True)  # [..., mul, 1]
            rms = torch.sqrt(sq_norm.mean(dim=-2, keepdim=True) + self.eps)  # [..., 1, 1]
            scale = weight.view(*([1] * (block.ndim - 2)), mul, 1)
            normed = (block / rms) * scale
            chunks.append(normed.reshape(*x.shape[:-1], mul * ir_dim))
        return torch.cat(chunks, dim=-1)


class E3NNConvBlock(nn.Module):
    """Equivariant spherical-harmonics tensor-product convolution + Gate + residual."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_scalars: o3.Irreps,
        irreps_gated: o3.Irreps,
        num_rbf: int,
        radial_hidden: int = 32,
        norm_scale: float = 1.0 / math.sqrt(12.0),
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.norm_scale = float(norm_scale)

        self.has_gated = irreps_gated.dim > 0
        if self.has_gated:
            act_scalars = [F.silu if ir.p == 1 else torch.tanh for _, ir in irreps_scalars]
            irreps_gates = o3.Irreps([(mul, (0, 1)) for mul, _ in irreps_gated]).simplify()
            act_gates = [torch.sigmoid for _ in irreps_gates]
            self.gate: Gate | None = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated,
            )
            tp_out_irreps = self.gate.irreps_in
            block_out_irreps = self.gate.irreps_out
        else:
            self.gate = None
            tp_out_irreps = irreps_scalars
            block_out_irreps = irreps_scalars

        self.irreps_out = o3.Irreps(block_out_irreps)
        self.lin_pre = o3.Linear(self.irreps_in, self.irreps_in)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            tp_out_irreps,
            shared_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_rbf, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.tp.weight_numel),
        )
        self.lin_post = o3.Linear(self.irreps_out, self.irreps_out)
        self.skip = o3.Linear(self.irreps_in, self.irreps_out)
        self.norm = IrrepsEquivariantRMSNorm(self.irreps_out)

    def forward(
        self,
        x: Tensor,
        src: Tensor,
        dst: Tensor,
        edge_sh: Tensor,
        rbf: Tensor,
        edge_env: Tensor,
        mask_flat: Tensor,
    ) -> Tensor:
        """Apply spherical-harmonics tensor-product message passing and gated update."""
        mask_f = mask_flat.to(x.dtype).unsqueeze(-1)
        skip_out: Tensor = self.skip(x)

        if src.numel() == 0:
            out_empty: Tensor = self.norm(skip_out) * mask_f
            return out_empty

        x_pre: Tensor = self.lin_pre(x)
        weights = self.radial_mlp(rbf) * edge_env.unsqueeze(-1)
        messages: Tensor = self.tp(x_pre[src], edge_sh, weights)

        agg = torch.zeros(
            (x.shape[0], messages.shape[-1]),
            device=x.device,
            dtype=x.dtype,
        )
        agg.index_add_(0, dst, messages)
        agg = agg * self.norm_scale

        activated: Tensor = self.gate(agg) if self.gate is not None else F.silu(agg)
        updated: Tensor = self.lin_post(activated)
        out: Tensor = self.norm(skip_out + updated) * mask_f
        return out


class E3NNEncoder(nn.Module):
    """Modern ``e3nn==0.6.0`` spherical-harmonics tensor-product 3D pocket encoder.

    Defaults to ``lmax=1`` (``32x0e + 16x1o``; plus ``8x0o + 8x1e`` when
    ``use_se3_chirality=True``) for fast Apple Silicon MPS execution, while supporting
    ``lmax=0, 1, 2, 3`` seamlessly. All buffers are strictly cast to ``torch.float32``.
    """

    def __init__(
        self,
        node_in_dim: int = 52,
        hidden_dim: int = 192,
        vector_dim: int = 64,
        out_dim: int = 256,
        num_layers: int = 4,
        num_rbf: int = 24,
        cutoff: float = 5.5,
        max_neighbors: int = 24,
        e3nn_lmax: int = 1,
        use_se3_chirality: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.e3nn_lmax = int(e3nn_lmax)
        self.use_se3_chirality = bool(use_se3_chirality)

        irreps_scalars, irreps_gated, irreps_hidden = _build_hidden_irreps(
            lmax=self.e3nn_lmax,
            scalar_channels=hidden_dim,
            vector_channels=vector_dim,
            use_se3_chirality=self.use_se3_chirality,
        )
        self.irreps_scalars = irreps_scalars
        self.irreps_gated = irreps_gated
        self.irreps_hidden = irreps_hidden
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.e3nn_lmax)

        c0 = next(mul for mul, ir in irreps_scalars if ir.l == 0 and ir.p == 1)
        self.irreps_in = o3.Irreps(f"{c0}x0e")

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, c0),
            nn.LayerNorm(c0),
            nn.SiLU(),
            nn.Linear(c0, c0),
        )
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)
        self.sh = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=True,
            normalization="component",
        )

        layers: list[E3NNConvBlock] = []
        curr_irreps = self.irreps_in
        for _ in range(num_layers):
            # Only allocate output irreps reachable via Clebsch-Gordan paths from curr_irreps x sh
            # so no mathematically zero channels amplify float32 rounding noise in RMSNorm.
            reachable = {
                ir_out
                for _, ir1 in curr_irreps
                for _, ir2 in self.irreps_sh
                for ir_out in ir1 * ir2
            }
            layer_scalars = o3.Irreps(
                [(mul, ir) for mul, ir in irreps_scalars if ir in reachable]
            ).simplify()
            layer_gated = o3.Irreps(
                [(mul, ir) for mul, ir in irreps_gated if ir in reachable]
            ).simplify()
            block = E3NNConvBlock(
                irreps_in=curr_irreps,
                irreps_sh=self.irreps_sh,
                irreps_scalars=layer_scalars,
                irreps_gated=layer_gated,
                num_rbf=num_rbf,
            )
            layers.append(block)
            curr_irreps = block.irreps_out
        self.layers = nn.ModuleList(layers)

        # Locate scalar (0e, and 0o if chiral) and polar vector (1o) slices in final irreps.
        self._scalar_slices: list[tuple[int, int]] = []
        self._vec1o_slice: tuple[int, int, int] | None = None
        scalar_readout_dim = 0
        offset = 0
        for mul, ir in curr_irreps:
            width = mul * ir.dim
            if ir.l == 0 and (ir.p == 1 or self.use_se3_chirality):
                self._scalar_slices.append((offset, offset + width))
                scalar_readout_dim += width
            elif ir.l == 1 and ir.p == -1:
                self._vec1o_slice = (offset, offset + width, mul)
            offset += width

        self.vec_readout: nn.Linear | None
        if self._vec1o_slice is not None:
            _, _, vec_mul = self._vec1o_slice
            self.vec_readout = nn.Linear(vec_mul, vector_dim, bias=False)
        else:
            self.vec_readout = None

        self.out_proj = nn.Sequential(
            nn.Linear(scalar_readout_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.pool = MaskedAttentionPool(in_dim=out_dim, out_dim=out_dim)

        # Ensure zero float64 buffers exist anywhere in e3nn submodules (MPS safety).
        self._cast_buffers_float32()

    def _cast_buffers_float32(self) -> None:
        for module in self.modules():
            for name, buf in list(module._buffers.items()):
                if buf is not None and buf.is_floating_point() and buf.dtype != torch.float32:
                    module._buffers[name] = buf.to(dtype=torch.float32)

    def forward(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode 3D pockets via e3nn spherical-harmonics tensor-product convolutions."""
        batch_size, num_atoms, _ = coords.shape
        mask_bool = atom_mask.to(torch.bool)
        mask_flat = mask_bool.reshape(batch_size * num_atoms)
        mask_f_flat = mask_flat.to(torch.float32)

        graph = build_radius_knn_graph(
            coords=coords,
            atom_mask=mask_bool,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

        feats_flat = node_feats.to(torch.float32).reshape(batch_size * num_atoms, self.node_in_dim)
        feats_flat = torch.where(mask_flat.unsqueeze(-1), feats_flat, torch.zeros_like(feats_flat))
        x = self.node_embed(feats_flat) * mask_f_flat.unsqueeze(-1)

        if graph.src.numel() > 0:
            edge_sh: Tensor = self.sh(graph.edge_vec).to(torch.float32)
            rbf = self.rbf(graph.edge_dist)
            edge_env = self.cutoff_fn(graph.edge_dist)
        else:
            edge_sh = torch.empty(
                (0, self.irreps_sh.dim), device=coords.device, dtype=torch.float32
            )
            rbf = torch.empty((0, self.rbf.num_rbf), device=coords.device, dtype=torch.float32)
            edge_env = torch.empty(0, device=coords.device, dtype=torch.float32)

        for layer in self.layers:
            x = layer(x, graph.src, graph.dst, edge_sh, rbf, edge_env, mask_flat)

        scalar_parts = [x[:, start:end] for start, end in self._scalar_slices]
        scalars = torch.cat(scalar_parts, dim=-1)
        node_emb = (self.out_proj(scalars) * mask_f_flat.unsqueeze(-1)).reshape(
            batch_size, num_atoms, self.out_dim
        )
        pooled_emb = self.pool(node_emb, mask_bool)

        vec_emb: Tensor | None = None
        if self._vec1o_slice is not None and self.vec_readout is not None:
            start, end, vec_mul = self._vec1o_slice
            v_1o = x[:, start:end].reshape(batch_size * num_atoms, vec_mul, 3)
            v_proj = self.vec_readout(v_1o.transpose(-1, -2)).transpose(-1, -2)
            vec_emb = (v_proj * mask_f_flat.unsqueeze(-1).unsqueeze(-1)).reshape(
                batch_size, num_atoms, self.vector_dim, 3
            )

        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled_emb, vec_emb=vec_emb)


__all__ = ["E3NNConvBlock", "E3NNEncoder", "IrrepsEquivariantRMSNorm"]
