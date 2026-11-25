"""GenEuclidean: SE(3)/E(3)-equivariant 3D pocket-conditioned LLM & CNS drug design SDK.

Implements PEP 562 module-level lazy loading (``__getattr__`` and ``__dir__``) with full
PEP 561 static type support (``if TYPE_CHECKING:`` + ``py.typed``) so ``import geneuclidean as ge``
completes in milliseconds without eagerly importing ``torch``, ``transformers``, ``sklearn``,
or ``matplotlib``.

Quickstart::

    import geneuclidean as ge

    targets = ge.list_targets("tourette")
    pocket = ge.Pocket.from_target("DRD1")
    reports = ge.score_molecules(["CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21"])
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "4.0.0"

if TYPE_CHECKING:
    from geneuclidean import (
        api as api,
    )
    from geneuclidean import (
        cli as cli,
    )
    from geneuclidean import (
        cns as cns,
    )
    from geneuclidean import (
        config as config,
    )
    from geneuclidean import (
        data as data,
    )
    from geneuclidean import (
        evaluation as evaluation,
    )
    from geneuclidean import (
        models as models,
    )
    from geneuclidean import (
        sampling as sampling,
    )
    from geneuclidean import (
        training as training,
    )
    from geneuclidean.api import (
        CredibilityDossier as CredibilityDossier,
    )
    from geneuclidean.api import (
        DesignResult as DesignResult,
    )
    from geneuclidean.api import (
        MoleculeDesigner as MoleculeDesigner,
    )
    from geneuclidean.api import (
        MoleculeReport as MoleculeReport,
    )
    from geneuclidean.api import (
        Pocket as Pocket,
    )
    from geneuclidean.api import (
        build_credibility_dossier as build_credibility_dossier,
    )
    from geneuclidean.api import (
        dock_and_profile as dock_and_profile,
    )
    from geneuclidean.api import (
        export_credibility_json as export_credibility_json,
    )
    from geneuclidean.api import (
        export_credibility_markdown as export_credibility_markdown,
    )
    from geneuclidean.api import (
        get_target as get_target,
    )
    from geneuclidean.api import (
        list_backbones as list_backbones,
    )
    from geneuclidean.api import (
        list_targets as list_targets,
    )
    from geneuclidean.api import (
        protonate_ph74 as protonate_ph74,
    )
    from geneuclidean.api import (
        redock_complex as redock_complex,
    )
    from geneuclidean.api import (
        score_molecules as score_molecules,
    )
    from geneuclidean.api import (
        triage_molecule as triage_molecule,
    )
    from geneuclidean.api import (
        triage_molecules as triage_molecules,
    )
    from geneuclidean.api import (
        visualize_complex as visualize_complex,
    )
    from geneuclidean.api import (
        visualize_gallery as visualize_gallery,
    )
    from geneuclidean.api import (
        visualize_molecules_2d as visualize_molecules_2d,
    )
    from geneuclidean.cns.medchem_triage import (
        MedChemTriageResult as MedChemTriageResult,
    )
    from geneuclidean.cns.medchem_triage import (
        ScaffoldHopReport as ScaffoldHopReport,
    )
    from geneuclidean.cns.medchem_triage import (
        compute_scaffold_hop_metrics as compute_scaffold_hop_metrics,
    )
    from geneuclidean.cns.medchem_triage import (
        is_scaffold_hop as is_scaffold_hop,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        AnchorCheckResult as AnchorCheckResult,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        CrossTargetSelectivityResult as CrossTargetSelectivityResult,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        DockingPoseResult as DockingPoseResult,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        OrthostericBox as OrthostericBox,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        RedockingBenchmarkResult as RedockingBenchmarkResult,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        RedockingSeedResult as RedockingSeedResult,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        RingConformer as RingConformer,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        check_canonical_anchors as check_canonical_anchors,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        compute_cross_target_matrix as compute_cross_target_matrix,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        compute_murcko_core_rmsd as compute_murcko_core_rmsd,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        compute_orthosteric_box as compute_orthosteric_box,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        compute_selectivity_ddg as compute_selectivity_ddg,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        dock_ligand_to_pocket as dock_ligand_to_pocket,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        generate_ring_diverse_conformers as generate_ring_diverse_conformers,
    )
    from geneuclidean.evaluation.docking_pipeline import (
        redock_native_complex as redock_native_complex,
    )
    from geneuclidean.evaluation.protonation import (
        ProtonationResult as ProtonationResult,
    )
    from geneuclidean.evaluation.protonation import (
        analyze_protonation_ph74 as analyze_protonation_ph74,
    )
    from geneuclidean.evaluation.protonation import (
        estimate_physiological_charge as estimate_physiological_charge,
    )
    from geneuclidean.evaluation.protonation import (
        extract_canonical_isomeric_smiles as extract_canonical_isomeric_smiles,
    )
    from geneuclidean.evaluation.protonation import (
        has_basic_amine as has_basic_amine,
    )
    from geneuclidean.evaluation.protonation import (
        protonate_mol_ph74 as protonate_mol_ph74,
    )
    from geneuclidean.evaluation.protonation import (
        protonate_smiles_ph74 as protonate_smiles_ph74,
    )

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        "api",
        "cli",
        "cns",
        "config",
        "data",
        "evaluation",
        "models",
        "sampling",
        "training",
    }
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # High-level SDK facade (`geneuclidean.api`)
    "CredibilityDossier": ("geneuclidean.api", "CredibilityDossier"),
    "DesignResult": ("geneuclidean.api", "DesignResult"),
    "MoleculeDesigner": ("geneuclidean.api", "MoleculeDesigner"),
    "MoleculeReport": ("geneuclidean.api", "MoleculeReport"),
    "Pocket": ("geneuclidean.api", "Pocket"),
    "build_credibility_dossier": ("geneuclidean.api", "build_credibility_dossier"),
    "dock_and_profile": ("geneuclidean.api", "dock_and_profile"),
    "export_credibility_json": ("geneuclidean.api", "export_credibility_json"),
    "export_credibility_markdown": ("geneuclidean.api", "export_credibility_markdown"),
    "get_target": ("geneuclidean.api", "get_target"),
    "list_backbones": ("geneuclidean.api", "list_backbones"),
    "list_targets": ("geneuclidean.api", "list_targets"),
    "protonate_ph74": ("geneuclidean.api", "protonate_ph74"),
    "redock_complex": ("geneuclidean.api", "redock_complex"),
    "score_molecules": ("geneuclidean.api", "score_molecules"),
    "triage_molecule": ("geneuclidean.api", "triage_molecule"),
    "triage_molecules": ("geneuclidean.api", "triage_molecules"),
    "visualize_complex": ("geneuclidean.api", "visualize_complex"),
    "visualize_gallery": ("geneuclidean.api", "visualize_gallery"),
    "visualize_molecules_2d": ("geneuclidean.api", "visualize_molecules_2d"),
    # Industrial MedChem Triage (`geneuclidean.cns.medchem_triage`)
    "MedChemTriageResult": ("geneuclidean.cns.medchem_triage", "MedChemTriageResult"),
    "ScaffoldHopReport": ("geneuclidean.cns.medchem_triage", "ScaffoldHopReport"),
    "compute_scaffold_hop_metrics": (
        "geneuclidean.cns.medchem_triage",
        "compute_scaffold_hop_metrics",
    ),
    "is_scaffold_hop": ("geneuclidean.cns.medchem_triage", "is_scaffold_hop"),
    # Physiological pH 7.4 Protonation (`geneuclidean.evaluation.protonation`)
    "ProtonationResult": ("geneuclidean.evaluation.protonation", "ProtonationResult"),
    "analyze_protonation_ph74": ("geneuclidean.evaluation.protonation", "analyze_protonation_ph74"),
    "estimate_physiological_charge": (
        "geneuclidean.evaluation.protonation",
        "estimate_physiological_charge",
    ),
    "extract_canonical_isomeric_smiles": (
        "geneuclidean.evaluation.protonation",
        "extract_canonical_isomeric_smiles",
    ),
    "has_basic_amine": ("geneuclidean.evaluation.protonation", "has_basic_amine"),
    "protonate_mol_ph74": ("geneuclidean.evaluation.protonation", "protonate_mol_ph74"),
    "protonate_smiles_ph74": ("geneuclidean.evaluation.protonation", "protonate_smiles_ph74"),
    # Reusable 3D Docking & Selectivity (`geneuclidean.evaluation.docking_pipeline`)
    "AnchorCheckResult": ("geneuclidean.evaluation.docking_pipeline", "AnchorCheckResult"),
    "CrossTargetSelectivityResult": (
        "geneuclidean.evaluation.docking_pipeline",
        "CrossTargetSelectivityResult",
    ),
    "DockingPoseResult": ("geneuclidean.evaluation.docking_pipeline", "DockingPoseResult"),
    "OrthostericBox": ("geneuclidean.evaluation.docking_pipeline", "OrthostericBox"),
    "RedockingBenchmarkResult": (
        "geneuclidean.evaluation.docking_pipeline",
        "RedockingBenchmarkResult",
    ),
    "RedockingSeedResult": ("geneuclidean.evaluation.docking_pipeline", "RedockingSeedResult"),
    "RingConformer": ("geneuclidean.evaluation.docking_pipeline", "RingConformer"),
    "check_canonical_anchors": (
        "geneuclidean.evaluation.docking_pipeline",
        "check_canonical_anchors",
    ),
    "compute_cross_target_matrix": (
        "geneuclidean.evaluation.docking_pipeline",
        "compute_cross_target_matrix",
    ),
    "compute_murcko_core_rmsd": (
        "geneuclidean.evaluation.docking_pipeline",
        "compute_murcko_core_rmsd",
    ),
    "compute_orthosteric_box": (
        "geneuclidean.evaluation.docking_pipeline",
        "compute_orthosteric_box",
    ),
    "compute_selectivity_ddg": (
        "geneuclidean.evaluation.docking_pipeline",
        "compute_selectivity_ddg",
    ),
    "dock_ligand_to_pocket": ("geneuclidean.evaluation.docking_pipeline", "dock_ligand_to_pocket"),
    "generate_ring_diverse_conformers": (
        "geneuclidean.evaluation.docking_pipeline",
        "generate_ring_diverse_conformers",
    ),
    "redock_native_complex": ("geneuclidean.evaluation.docking_pipeline", "redock_native_complex"),
}

__all__ = [
    "__version__",
    *sorted(_LAZY_SUBMODULES),
    *sorted(_LAZY_ATTRS.keys()),
]


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level SDK symbols and submodules on first attribute access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        mod_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose complete lazy public SDK surface to ``dir(geneuclidean)`` and IDEs."""
    return sorted(set(globals().keys()) | set(__all__))
