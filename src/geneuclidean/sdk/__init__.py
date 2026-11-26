"""Modular high-level SDK subpackage for GenEuclidean.

Organizes the public SDK into cohesive, single-responsibility modules:
- ``registries``: Human CNS target atlas & chemical LLM backbone introspection.
- ``pocket``: 3D binding-pocket loading, cofactor preservation, and featurization.
- ``scoring``: Unified CNS MPO, pH 7.4 ionization, triage, ADMET, and pluggable filters/scorers.
- ``designer``: 3D pocket-conditioned de novo generation & SAFE scaffold decoration.
- ``pipeline``: Immutable fluent multi-stage drug-design workflow composition.
- ``docking``: High-level AutoDock Vina docking and multi-seed native redocking wrappers.
- ``viz``: Dual-mode 3Dmol.js WebGL HTML & 300-DPI static 3D/2D renderers.
- ``credibility``: FDA Jan 2025 / EMA Jan 2026 7-step AI Credibility Dossier export.
"""

from __future__ import annotations

from geneuclidean.sdk.credibility import (
    CredibilityDossier,
    build_credibility_dossier,
    export_credibility_json,
    export_credibility_markdown,
)
from geneuclidean.sdk.designer import DesignResult, MoleculeDesigner
from geneuclidean.sdk.docking import dock_and_profile, redock_complex
from geneuclidean.sdk.pipeline import BaseDesignCallback, DesignPipeline
from geneuclidean.sdk.pocket import Pocket
from geneuclidean.sdk.registries import (
    audit_tourette_circuit,
    get_target,
    list_backbones,
    list_targets,
    register_target,
    unregister_target,
)
from geneuclidean.sdk.scoring import (
    ADMETFilter,
    BasicAmineFilter,
    FunctionScorer,
    MoleculeReport,
    MPOFilter,
    PredicateFilter,
    QEDFilter,
    ScaffoldHopFilter,
    TriageFilter,
    apply_custom_scorers,
    protonate_ph74,
    score_molecules,
    score_tourette_circuit,
    triage_molecule,
    triage_molecules,
)
from geneuclidean.sdk.viz import (
    visualize_complex,
    visualize_gallery,
    visualize_molecules_2d,
)

__all__ = [
    "ADMETFilter",
    "BaseDesignCallback",
    "BasicAmineFilter",
    "CredibilityDossier",
    "DesignPipeline",
    "DesignResult",
    "FunctionScorer",
    "MPOFilter",
    "MoleculeDesigner",
    "MoleculeReport",
    "Pocket",
    "PredicateFilter",
    "QEDFilter",
    "ScaffoldHopFilter",
    "TriageFilter",
    "apply_custom_scorers",
    "audit_tourette_circuit",
    "build_credibility_dossier",
    "dock_and_profile",
    "export_credibility_json",
    "export_credibility_markdown",
    "get_target",
    "list_backbones",
    "list_targets",
    "protonate_ph74",
    "redock_complex",
    "register_target",
    "score_molecules",
    "score_tourette_circuit",
    "triage_molecule",
    "triage_molecules",
    "unregister_target",
    "visualize_complex",
    "visualize_gallery",
    "visualize_molecules_2d",
]
