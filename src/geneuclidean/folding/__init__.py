"""AlphaFold 3-class Co-Folding, NanoFold3 & Open-Weights Folding Hub for GenEuclidean.

Implements PEP 562 module-level lazy loading (``__getattr__`` / ``__dir__``) with full
PEP 561 static type support (``if TYPE_CHECKING:``) so ``import geneuclidean.folding``
completes in ``< 5 ms`` with **zero** eager ``torch``, ``rdkit``, ``Bio``, or
``transformers`` imports.

Public surface spans two pillars:
1. **Open-Weights Co-Folding Hub (Zero-Torch core):**
   :data:`FOLDING_BACKEND_REGISTRY`, :class:`FoldingJobSpec`, :class:`FoldedComplexResult`,
   :func:`fold_complex`, :func:`list_folding_backends`, :func:`export_alphafold3_json`,
   :func:`export_boltz2_yaml`, and :func:`export_protenix_json`.
2. **NanoFold3 Trainable Mini-AlphaFold3 Primitives (Lazy PyTorch):**
   :class:`PairformerStack`, :class:`SimpleFoldTrunk`, :class:`AllAtomDiffusionModule`,
   :class:`KabschWeightedRigidAlign`, :class:`CoFoldingLoss`, :class:`ConfidenceHead`,
   :class:`AffinityHead`, :class:`NanoFold3`, and :func:`build_nanofold3`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geneuclidean.folding.runners import (
        FOLDING_BACKEND_REGISTRY as FOLDING_BACKEND_REGISTRY,
    )
    from geneuclidean.folding.runners import (
        AlphaFold3Backend as AlphaFold3Backend,
    )
    from geneuclidean.folding.runners import (
        BaseFoldingBackend as BaseFoldingBackend,
    )
    from geneuclidean.folding.runners import (
        Boltz2Backend as Boltz2Backend,
    )
    from geneuclidean.folding.runners import (
        ESMFoldBackend as ESMFoldBackend,
    )
    from geneuclidean.folding.runners import (
        FoldedComplexResult as FoldedComplexResult,
    )
    from geneuclidean.folding.runners import (
        FoldingBackend as FoldingBackend,
    )
    from geneuclidean.folding.runners import (
        FoldingJobSpec as FoldingJobSpec,
    )
    from geneuclidean.folding.runners import (
        MissingBackendError as MissingBackendError,
    )
    from geneuclidean.folding.runners import (
        NanoFold3Backend as NanoFold3Backend,
    )
    from geneuclidean.folding.runners import (
        NetworkDisabledError as NetworkDisabledError,
    )
    from geneuclidean.folding.runners import (
        ProtenixBackend as ProtenixBackend,
    )
    from geneuclidean.folding.runners import (
        export_alphafold3_json as export_alphafold3_json,
    )
    from geneuclidean.folding.runners import (
        export_boltz2_yaml as export_boltz2_yaml,
    )
    from geneuclidean.folding.runners import (
        export_protenix_json as export_protenix_json,
    )
    from geneuclidean.folding.runners import (
        fold_complex as fold_complex,
    )
    from geneuclidean.folding.runners import (
        folding_backend_catalog as folding_backend_catalog,
    )
    from geneuclidean.folding.runners import (
        list_folding_backends as list_folding_backends,
    )
    from geneuclidean.folding.runners import (
        register_folding_backend as register_folding_backend,
    )
    from geneuclidean.folding.runners import (
        run_folding_backend as run_folding_backend,
    )

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Open-weights co-folding hub (`runners`, zero-torch)
    "FOLDING_BACKEND_REGISTRY": ("geneuclidean.folding.runners", "FOLDING_BACKEND_REGISTRY"),
    "AlphaFold3Backend": ("geneuclidean.folding.runners", "AlphaFold3Backend"),
    "BaseFoldingBackend": ("geneuclidean.folding.runners", "BaseFoldingBackend"),
    "Boltz2Backend": ("geneuclidean.folding.runners", "Boltz2Backend"),
    "ESMFoldBackend": ("geneuclidean.folding.runners", "ESMFoldBackend"),
    "FoldedComplexResult": ("geneuclidean.folding.runners", "FoldedComplexResult"),
    "FoldingBackend": ("geneuclidean.folding.runners", "FoldingBackend"),
    "FoldingJobSpec": ("geneuclidean.folding.runners", "FoldingJobSpec"),
    "MissingBackendError": ("geneuclidean.folding.runners", "MissingBackendError"),
    "NanoFold3Backend": ("geneuclidean.folding.runners", "NanoFold3Backend"),
    "NetworkDisabledError": ("geneuclidean.folding.runners", "NetworkDisabledError"),
    "ProtenixBackend": ("geneuclidean.folding.runners", "ProtenixBackend"),
    "export_alphafold3_json": ("geneuclidean.folding.runners", "export_alphafold3_json"),
    "export_boltz2_yaml": ("geneuclidean.folding.runners", "export_boltz2_yaml"),
    "export_protenix_json": ("geneuclidean.folding.runners", "export_protenix_json"),
    "fold_complex": ("geneuclidean.folding.runners", "fold_complex"),
    "folding_backend_catalog": ("geneuclidean.folding.runners", "folding_backend_catalog"),
    "list_folding_backends": ("geneuclidean.folding.runners", "list_folding_backends"),
    "register_folding_backend": ("geneuclidean.folding.runners", "register_folding_backend"),
    "run_folding_backend": ("geneuclidean.folding.runners", "run_folding_backend"),
    # Pairformer & Apple SimpleFold trunk (`pairformer`, lazy torch)
    "PairBiasedAttention": ("geneuclidean.folding.pairformer", "PairBiasedAttention"),
    "PairformerBlock": ("geneuclidean.folding.pairformer", "PairformerBlock"),
    "PairformerStack": ("geneuclidean.folding.pairformer", "PairformerStack"),
    "SimpleFoldTransformerBlock": (
        "geneuclidean.folding.pairformer",
        "SimpleFoldTransformerBlock",
    ),
    "SimpleFoldTrunk": ("geneuclidean.folding.pairformer", "SimpleFoldTrunk"),
    "SwiGLUTransition": ("geneuclidean.folding.pairformer", "SwiGLUTransition"),
    "TriangleAttention": ("geneuclidean.folding.pairformer", "TriangleAttention"),
    "TriangleMultiplicativeUpdate": (
        "geneuclidean.folding.pairformer",
        "TriangleMultiplicativeUpdate",
    ),
    # All-atom EDM & Flow-Matching diffusion + physics losses (`diffusion`, lazy torch)
    "AllAtomDiffusionModule": ("geneuclidean.folding.diffusion", "AllAtomDiffusionModule"),
    "AtomAttentionDecoder": ("geneuclidean.folding.diffusion", "AtomAttentionDecoder"),
    "AtomAttentionEncoder": ("geneuclidean.folding.diffusion", "AtomAttentionEncoder"),
    "CoFoldingLoss": ("geneuclidean.folding.diffusion", "CoFoldingLoss"),
    "CoFoldingLossOutput": ("geneuclidean.folding.diffusion", "CoFoldingLossOutput"),
    "DiffusionTransformer": ("geneuclidean.folding.diffusion", "DiffusionTransformer"),
    "EDMDiffusionSchedule": ("geneuclidean.folding.diffusion", "EDMDiffusionSchedule"),
    "FlowMatchingSchedule": ("geneuclidean.folding.diffusion", "FlowMatchingSchedule"),
    "KabschWeightedRigidAlign": ("geneuclidean.folding.diffusion", "KabschWeightedRigidAlign"),
    # AF3 confidence & Boltz-2 affinity heads (`confidence`, lazy torch)
    "AffinityHead": ("geneuclidean.folding.confidence", "AffinityHead"),
    "ConfidenceHead": ("geneuclidean.folding.confidence", "ConfidenceHead"),
    "ConfidenceLoss": ("geneuclidean.folding.confidence", "ConfidenceLoss"),
    "ConfidenceOutput": ("geneuclidean.folding.confidence", "ConfidenceOutput"),
    # NanoFold3 end-to-end co-folder (`nanofold3`, lazy torch)
    "ComplexBatch": ("geneuclidean.folding.nanofold3", "ComplexBatch"),
    "NanoFold3": ("geneuclidean.folding.nanofold3", "NanoFold3"),
    "NanoFold3Config": ("geneuclidean.folding.nanofold3", "NanoFold3Config"),
    "NanoFold3Result": ("geneuclidean.folding.nanofold3", "NanoFold3Result"),
    "NanoFold3SampleOutput": ("geneuclidean.folding.nanofold3", "NanoFold3SampleOutput"),
    "NanoFold3TrainOutput": ("geneuclidean.folding.nanofold3", "NanoFold3TrainOutput"),
    "build_nanofold3": ("geneuclidean.folding.nanofold3", "build_nanofold3"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    """Lazily resolve folding runners and NanoFold3 neural primitives (PEP 562)."""
    if name in _LAZY_ATTRS:
        mod_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose complete lazy folding surface to ``dir(geneuclidean.folding)``."""
    return sorted(set(globals().keys()) | set(__all__))
