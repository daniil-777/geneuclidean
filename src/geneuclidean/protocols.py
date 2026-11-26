"""Runtime-checkable structural typing protocols (PEP 544) for GenEuclidean extensions.

These protocols define the clean extension seams of the library so external
researchers and production teams can plug in custom 3D equivariant encoders,
autoregressive decoders, sampling strategies, molecular scorers, post-design
filters, and lifecycle callbacks without subclassing internal base classes or
modifying GenEuclidean source code.

All heavy types (``torch.Tensor``, ``PocketEncoderOutput``, ``MoleculeReport``)
live strictly under ``if TYPE_CHECKING:`` so importing ``geneuclidean.protocols``
has zero PyTorch or RDKit startup overhead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

    from geneuclidean.models.encoders.rbf import PocketEncoderOutput
    from geneuclidean.sdk.designer import DesignResult
    from geneuclidean.sdk.pocket import Pocket
    from geneuclidean.sdk.scoring import MoleculeReport


@runtime_checkable
class PocketEncoderProtocol(Protocol):
    """Contract for pluggable 3D SE(3)/E(3)-equivariant protein pocket encoders.

    Implementations accept batched heavy-atom 3D coordinates, 52-dim E(3)-invariant
    per-atom descriptors, and a boolean atom mask, and return a
    :class:`~geneuclidean.models.encoders.rbf.PocketEncoderOutput` whose
    ``node_emb`` and ``pooled_emb`` tensors are invariant under rigid 3D rotations
    and translations (``SE(3)``/``E(3)``).
    """

    def __call__(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
    ) -> PocketEncoderOutput:
        """Encode a batch of 3D pockets into invariant/equivariant representations."""
        ...


@runtime_checkable
class MolecularDecoderProtocol(Protocol):
    """Contract for pluggable autoregressive chemical language decoders."""

    def __call__(
        self,
        input_ids: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict next-token logits ``[B, T, V]`` and optional cross-attention maps."""
        ...


@runtime_checkable
class SamplingStrategyProtocol(Protocol):
    """Contract for custom next-token probability shaping strategies.

    Registered via :func:`geneuclidean.register_sampling_strategy`. Given raw
    next-token ``logits`` of shape ``[B, V]``, returns a valid normalized
    probability distribution of shape ``[B, V]`` summing to ``1.0`` along ``dim=-1``.
    """

    def __call__(
        self,
        logits: Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.92,
        min_p: float = 0.08,
    ) -> Tensor:
        """Transform raw logits ``[B, V]`` into normalized sampling probabilities ``[B, V]``."""
        ...


@runtime_checkable
class MoleculeScorerProtocol(Protocol):
    """Contract for pluggable molecular scoring functions used in SDK & pipelines.

    Scorers receive an ordered sequence of :class:`~geneuclidean.sdk.scoring.MoleculeReport`
    objects and return one numeric score per report (typically in ``[0.0, 1.0]`` when
    blended into ``composite_reward``). Invalid molecules should receive ``0.0``.
    """

    @property
    def name(self) -> str:
        """Unique identifier stored under ``MoleculeReport.custom_scores[name]``."""
        ...

    def score(self, reports: Sequence[MoleculeReport]) -> Sequence[float]:
        """Compute one numeric score per input :class:`MoleculeReport`."""
        ...


@runtime_checkable
class MoleculeFilterProtocol(Protocol):
    """Contract for composable post-generation molecular filters.

    Filters are deterministic predicates over a single
    :class:`~geneuclidean.sdk.scoring.MoleculeReport`. Used by
    :class:`~geneuclidean.sdk.pipeline.DesignPipeline` and
    :meth:`~geneuclidean.sdk.designer.DesignResult.filter`.
    """

    @property
    def name(self) -> str:
        """Human-readable filter name recorded in pipeline attrition telemetry."""
        ...

    def keep(self, report: MoleculeReport) -> bool:
        """Return ``True`` if ``report`` passes this filter criterion."""
        ...


@runtime_checkable
class DesignCallbackProtocol(Protocol):
    """Optional lifecycle observer for :class:`~geneuclidean.sdk.pipeline.DesignPipeline`.

    Implement any subset of lifecycle hooks (or use
    :class:`~geneuclidean.sdk.pipeline.BaseDesignCallback` for no-op defaults) to
    wire experiment trackers, progress bars, or structured audit logs cleanly.
    """

    def on_design_start(self, pocket: Pocket | Any, config: Mapping[str, Any]) -> None:
        """Called immediately before candidate generation starts."""
        ...

    def on_candidates_generated(
        self,
        pocket: Pocket | Any,
        raw_smiles: Sequence[str],
        reports: Sequence[MoleculeReport],
    ) -> None:
        """Called after raw sampling and scoring, prior to pipeline filtering."""
        ...

    def on_design_end(self, pocket: Pocket | Any, result: DesignResult) -> None:
        """Called once final filtering, reranking, and telemetry assembly finish."""
        ...


__all__ = [
    "DesignCallbackProtocol",
    "MolecularDecoderProtocol",
    "MoleculeFilterProtocol",
    "MoleculeScorerProtocol",
    "PocketEncoderProtocol",
    "SamplingStrategyProtocol",
]
