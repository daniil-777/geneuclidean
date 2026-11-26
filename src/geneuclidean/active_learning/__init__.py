"""GenEuclidean v5.1 Frontier Active Learning & DMTA Optimization Subpackage.

Implements PEP 562 module-level lazy loading (``__getattr__`` / ``__dir__``) so
importing ``geneuclidean.active_learning`` incurs zero eager ``torch``, ``sklearn``,
``scipy``, or ``rdkit`` startup latency while preserving full PEP 561 static typing.

Submodules:
    - ``featurizers``: Cached Morgan ECFP4 binary/count/physchem + PSD Tanimoto & MinMax kernels.
    - ``surrogates``: ``TanimotoGP``, ``EnsembleSurrogate``, ``ConformalSurrogate``,
      ``SURROGATE_REGISTRY``.
    - ``acquisitions``: 14 strategies in ``ACQUISITION_REGISTRY`` (pointwise, Kriging Believer,
      fast greedy MAP DPP, exact 2D / QMC ``qEHVI``, ParEGO, Avoid-ome constrained EI/UCB).
    - ``multifidelity``: Correlation-gated Kennedy-O'Hagan ``MultiFidelityDMTA`` router.
    - ``combinatorial_ts``: :math:`O(N_A + N_B)` reaction-space ``CombinatorialThompsonSampler``.
    - ``diagnostics``: Cold-start seeding, AD-stratified calibration metrics & ``StoppingAdvisor``.
    - ``engine``: ``ActiveLearningEngine``, ``DMTAQuery``, ``DMTARoundSummary``, ``DMTALedger``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geneuclidean.active_learning import (
        acquisitions as acquisitions,
    )
    from geneuclidean.active_learning import (
        combinatorial_ts as combinatorial_ts,
    )
    from geneuclidean.active_learning import (
        diagnostics as diagnostics,
    )
    from geneuclidean.active_learning import (
        engine as engine,
    )
    from geneuclidean.active_learning import (
        featurizers as featurizers,
    )
    from geneuclidean.active_learning import (
        multifidelity as multifidelity,
    )
    from geneuclidean.active_learning import (
        surrogates as surrogates,
    )
    from geneuclidean.active_learning.acquisitions import (
        ACQUISITION_REGISTRY as ACQUISITION_REGISTRY,
    )
    from geneuclidean.active_learning.acquisitions import (
        AcquisitionContext as AcquisitionContext,
    )
    from geneuclidean.active_learning.acquisitions import (
        AcquisitionSelection as AcquisitionSelection,
    )
    from geneuclidean.active_learning.acquisitions import (
        build_feasibility_vector as build_feasibility_vector,
    )
    from geneuclidean.active_learning.acquisitions import (
        compute_exact_2d_ehvi as compute_exact_2d_ehvi,
    )
    from geneuclidean.active_learning.acquisitions import (
        compute_expected_improvement as compute_expected_improvement,
    )
    from geneuclidean.active_learning.acquisitions import (
        compute_hypervolume as compute_hypervolume,
    )
    from geneuclidean.active_learning.acquisitions import (
        compute_hypervolume_2d as compute_hypervolume_2d,
    )
    from geneuclidean.active_learning.acquisitions import (
        compute_log_expected_improvement as compute_log_expected_improvement,
    )
    from geneuclidean.active_learning.acquisitions import (
        list_acquisitions as list_acquisitions,
    )
    from geneuclidean.active_learning.acquisitions import (
        pareto_front as pareto_front,
    )
    from geneuclidean.active_learning.acquisitions import (
        register_acquisition as register_acquisition,
    )
    from geneuclidean.active_learning.acquisitions import (
        run_acquisition as run_acquisition,
    )
    from geneuclidean.active_learning.combinatorial_ts import (
        CombinatorialThompsonSampler as CombinatorialThompsonSampler,
    )
    from geneuclidean.active_learning.combinatorial_ts import (
        CombinatorialTSProduct as CombinatorialTSProduct,
    )
    from geneuclidean.active_learning.combinatorial_ts import (
        CombinatorialTSResult as CombinatorialTSResult,
    )
    from geneuclidean.active_learning.combinatorial_ts import (
        ReagentPosterior as ReagentPosterior,
    )
    from geneuclidean.active_learning.diagnostics import (
        CalibrationMetrics as CalibrationMetrics,
    )
    from geneuclidean.active_learning.diagnostics import (
        ColdStartSelector as ColdStartSelector,
    )
    from geneuclidean.active_learning.diagnostics import (
        ColdStartStrategy as ColdStartStrategy,
    )
    from geneuclidean.active_learning.diagnostics import (
        StoppingAdvisor as StoppingAdvisor,
    )
    from geneuclidean.active_learning.diagnostics import (
        StoppingReport as StoppingReport,
    )
    from geneuclidean.active_learning.diagnostics import (
        compute_batch_internal_tanimoto as compute_batch_internal_tanimoto,
    )
    from geneuclidean.active_learning.diagnostics import (
        evaluate_calibration as evaluate_calibration,
    )
    from geneuclidean.active_learning.diagnostics import (
        evaluate_uq_calibration as evaluate_uq_calibration,
    )
    from geneuclidean.active_learning.diagnostics import (
        select_cold_start as select_cold_start,
    )
    from geneuclidean.active_learning.engine import (
        ActiveLearningEngine as ActiveLearningEngine,
    )
    from geneuclidean.active_learning.engine import (
        DMTALedger as DMTALedger,
    )
    from geneuclidean.active_learning.engine import (
        DMTAQuery as DMTAQuery,
    )
    from geneuclidean.active_learning.engine import (
        DMTARoundSummary as DMTARoundSummary,
    )
    from geneuclidean.active_learning.engine import (
        PendingBatchManager as PendingBatchManager,
    )
    from geneuclidean.active_learning.featurizers import (
        canonicalize_smiles as canonicalize_smiles,
    )
    from geneuclidean.active_learning.featurizers import (
        featurize_pool as featurize_pool,
    )
    from geneuclidean.active_learning.featurizers import (
        minmax_kernel as minmax_kernel,
    )
    from geneuclidean.active_learning.featurizers import (
        minmax_kernel_matrix as minmax_kernel_matrix,
    )
    from geneuclidean.active_learning.featurizers import (
        tanimoto_kernel as tanimoto_kernel,
    )
    from geneuclidean.active_learning.featurizers import (
        tanimoto_kernel_matrix as tanimoto_kernel_matrix,
    )
    from geneuclidean.active_learning.featurizers import (
        tanimoto_max_similarity as tanimoto_max_similarity,
    )
    from geneuclidean.active_learning.multifidelity import (
        FidelityTierSpec as FidelityTierSpec,
    )
    from geneuclidean.active_learning.multifidelity import (
        MultiFidelityDMTA as MultiFidelityDMTA,
    )
    from geneuclidean.active_learning.multifidelity import (
        MultiFidelityObservation as MultiFidelityObservation,
    )
    from geneuclidean.active_learning.multifidelity import (
        MultiFidelityRoundSummary as MultiFidelityRoundSummary,
    )
    from geneuclidean.active_learning.surrogates import (
        SURROGATE_REGISTRY as SURROGATE_REGISTRY,
    )
    from geneuclidean.active_learning.surrogates import (
        ConformalSurrogate as ConformalSurrogate,
    )
    from geneuclidean.active_learning.surrogates import (
        EnsembleSurrogate as EnsembleSurrogate,
    )
    from geneuclidean.active_learning.surrogates import (
        SurrogatePrediction as SurrogatePrediction,
    )
    from geneuclidean.active_learning.surrogates import (
        SurrogateProtocol as SurrogateProtocol,
    )
    from geneuclidean.active_learning.surrogates import (
        TanimotoGP as TanimotoGP,
    )
    from geneuclidean.active_learning.surrogates import (
        build_surrogate as build_surrogate,
    )
    from geneuclidean.active_learning.surrogates import (
        list_surrogates as list_surrogates,
    )
    from geneuclidean.active_learning.surrogates import (
        register_surrogate as register_surrogate,
    )

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        "acquisitions",
        "combinatorial_ts",
        "diagnostics",
        "engine",
        "featurizers",
        "multifidelity",
        "surrogates",
    }
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Featurizers & PSD Kernels
    "canonicalize_smiles": ("geneuclidean.active_learning.featurizers", "canonicalize_smiles"),
    "featurize_pool": ("geneuclidean.active_learning.featurizers", "featurize_pool"),
    "minmax_kernel": ("geneuclidean.active_learning.featurizers", "minmax_kernel"),
    "minmax_kernel_matrix": ("geneuclidean.active_learning.featurizers", "minmax_kernel_matrix"),
    "tanimoto_kernel": ("geneuclidean.active_learning.featurizers", "tanimoto_kernel"),
    "tanimoto_kernel_matrix": (
        "geneuclidean.active_learning.featurizers",
        "tanimoto_kernel_matrix",
    ),
    "tanimoto_max_similarity": (
        "geneuclidean.active_learning.featurizers",
        "tanimoto_max_similarity",
    ),
    # Surrogates & Registry
    "SURROGATE_REGISTRY": ("geneuclidean.active_learning.surrogates", "SURROGATE_REGISTRY"),
    "ConformalSurrogate": ("geneuclidean.active_learning.surrogates", "ConformalSurrogate"),
    "EnsembleSurrogate": ("geneuclidean.active_learning.surrogates", "EnsembleSurrogate"),
    "SurrogatePrediction": ("geneuclidean.active_learning.surrogates", "SurrogatePrediction"),
    "SurrogateProtocol": ("geneuclidean.active_learning.surrogates", "SurrogateProtocol"),
    "TanimotoGP": ("geneuclidean.active_learning.surrogates", "TanimotoGP"),
    "build_surrogate": ("geneuclidean.active_learning.surrogates", "build_surrogate"),
    "list_surrogates": ("geneuclidean.active_learning.surrogates", "list_surrogates"),
    "register_surrogate": ("geneuclidean.active_learning.surrogates", "register_surrogate"),
    # Acquisitions & Registry
    "ACQUISITION_REGISTRY": ("geneuclidean.active_learning.acquisitions", "ACQUISITION_REGISTRY"),
    "AcquisitionContext": ("geneuclidean.active_learning.acquisitions", "AcquisitionContext"),
    "AcquisitionSelection": ("geneuclidean.active_learning.acquisitions", "AcquisitionSelection"),
    "build_feasibility_vector": (
        "geneuclidean.active_learning.acquisitions",
        "build_feasibility_vector",
    ),
    "compute_exact_2d_ehvi": (
        "geneuclidean.active_learning.acquisitions",
        "compute_exact_2d_ehvi",
    ),
    "compute_expected_improvement": (
        "geneuclidean.active_learning.acquisitions",
        "compute_expected_improvement",
    ),
    "compute_hypervolume": ("geneuclidean.active_learning.acquisitions", "compute_hypervolume"),
    "compute_hypervolume_2d": (
        "geneuclidean.active_learning.acquisitions",
        "compute_hypervolume_2d",
    ),
    "compute_log_expected_improvement": (
        "geneuclidean.active_learning.acquisitions",
        "compute_log_expected_improvement",
    ),
    "list_acquisitions": ("geneuclidean.active_learning.acquisitions", "list_acquisitions"),
    "pareto_front": ("geneuclidean.active_learning.acquisitions", "pareto_front"),
    "register_acquisition": ("geneuclidean.active_learning.acquisitions", "register_acquisition"),
    "run_acquisition": ("geneuclidean.active_learning.acquisitions", "run_acquisition"),
    # Multi-Fidelity DMTA
    "FidelityTierSpec": ("geneuclidean.active_learning.multifidelity", "FidelityTierSpec"),
    "MultiFidelityDMTA": ("geneuclidean.active_learning.multifidelity", "MultiFidelityDMTA"),
    "MultiFidelityObservation": (
        "geneuclidean.active_learning.multifidelity",
        "MultiFidelityObservation",
    ),
    "MultiFidelityRoundSummary": (
        "geneuclidean.active_learning.multifidelity",
        "MultiFidelityRoundSummary",
    ),
    # Combinatorial Reaction-Space TS
    "CombinatorialTSProduct": (
        "geneuclidean.active_learning.combinatorial_ts",
        "CombinatorialTSProduct",
    ),
    "CombinatorialTSResult": (
        "geneuclidean.active_learning.combinatorial_ts",
        "CombinatorialTSResult",
    ),
    "CombinatorialThompsonSampler": (
        "geneuclidean.active_learning.combinatorial_ts",
        "CombinatorialThompsonSampler",
    ),
    "ReagentPosterior": ("geneuclidean.active_learning.combinatorial_ts", "ReagentPosterior"),
    # Diagnostics, Cold-Start & Stopping
    "CalibrationMetrics": ("geneuclidean.active_learning.diagnostics", "CalibrationMetrics"),
    "ColdStartSelector": ("geneuclidean.active_learning.diagnostics", "ColdStartSelector"),
    "ColdStartStrategy": ("geneuclidean.active_learning.diagnostics", "ColdStartStrategy"),
    "StoppingAdvisor": ("geneuclidean.active_learning.diagnostics", "StoppingAdvisor"),
    "StoppingReport": ("geneuclidean.active_learning.diagnostics", "StoppingReport"),
    "compute_batch_internal_tanimoto": (
        "geneuclidean.active_learning.diagnostics",
        "compute_batch_internal_tanimoto",
    ),
    "evaluate_calibration": ("geneuclidean.active_learning.diagnostics", "evaluate_calibration"),
    "evaluate_uq_calibration": (
        "geneuclidean.active_learning.diagnostics",
        "evaluate_uq_calibration",
    ),
    "select_cold_start": ("geneuclidean.active_learning.diagnostics", "select_cold_start"),
    # Engine & Provenance Ledger
    "ActiveLearningEngine": ("geneuclidean.active_learning.engine", "ActiveLearningEngine"),
    "DMTALedger": ("geneuclidean.active_learning.engine", "DMTALedger"),
    "DMTAQuery": ("geneuclidean.active_learning.engine", "DMTAQuery"),
    "DMTARoundSummary": ("geneuclidean.active_learning.engine", "DMTARoundSummary"),
    "PendingBatchManager": ("geneuclidean.active_learning.engine", "PendingBatchManager"),
}

__all__ = [
    *sorted(_LAZY_SUBMODULES),
    *sorted(_LAZY_ATTRS.keys()),
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        mod_path, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
