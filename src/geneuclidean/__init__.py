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

Extensibility::

    @ge.register_encoder("my_se3_net")
    def build_my_se3(cfg, max_neighbors=24): ...

    pipe = (
        ge.DesignPipeline(designer)
        .add_filter(ge.TriageFilter())
        .add_filter(ge.MPOFilter(min_mpo_v2=4.5))
        .add_scorer(ge.FunctionScorer("bonus", lambda r: 1.0), weight=0.2)
    )
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
        exceptions as exceptions,
    )
    from geneuclidean import (
        models as models,
    )
    from geneuclidean import (
        protocols as protocols,
    )
    from geneuclidean import (
        registry as registry,
    )
    from geneuclidean import (
        sampling as sampling,
    )
    from geneuclidean import (
        sdk as sdk,
    )
    from geneuclidean import (
        training as training,
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
    from geneuclidean.cns.scaffold_hop import (
        IBOGAINE_SMILES as IBOGAINE_SMILES,
    )
    from geneuclidean.cns.scaffold_hop import (
        PAROXETINE_SMILES as PAROXETINE_SMILES,
    )
    from geneuclidean.cns.scaffold_hop import (
        ScaffoldHopCandidateAudit as ScaffoldHopCandidateAudit,
    )
    from geneuclidean.cns.scaffold_hop import (
        audit_scaffold_hop_campaign as audit_scaffold_hop_campaign,
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
    from geneuclidean.evaluation.ring_strain import (
        RING_STRAIN_BACKEND_REGISTRY as RING_STRAIN_BACKEND_REGISTRY,
    )
    from geneuclidean.evaluation.ring_strain import (
        RingStrainFilter as RingStrainFilter,
    )
    from geneuclidean.evaluation.ring_strain import (
        RingStrainProfile as RingStrainProfile,
    )
    from geneuclidean.evaluation.ring_strain import (
        RingStrainScorer as RingStrainScorer,
    )
    from geneuclidean.evaluation.ring_strain import (
        SaturatedRingReport as SaturatedRingReport,
    )
    from geneuclidean.evaluation.ring_strain import (
        analyze_ring_strain as analyze_ring_strain,
    )
    from geneuclidean.evaluation.ring_strain import (
        compute_cremer_pople as compute_cremer_pople,
    )
    from geneuclidean.evaluation.ring_strain import (
        register_ring_strain_backend as register_ring_strain_backend,
    )
    from geneuclidean.exceptions import (
        CheckpointError as CheckpointError,
    )
    from geneuclidean.exceptions import (
        DockingError as DockingError,
    )
    from geneuclidean.exceptions import (
        GenEuclideanError as GenEuclideanError,
    )
    from geneuclidean.exceptions import (
        InvalidMoleculeError as InvalidMoleculeError,
    )
    from geneuclidean.exceptions import (
        OptionalDependencyError as OptionalDependencyError,
    )
    from geneuclidean.exceptions import (
        PocketFeaturizationError as PocketFeaturizationError,
    )
    from geneuclidean.exceptions import (
        RegistryError as RegistryError,
    )
    from geneuclidean.exceptions import (
        StructureNotFoundError as StructureNotFoundError,
    )
    from geneuclidean.exceptions import (
        TargetNotFoundError as TargetNotFoundError,
    )
    from geneuclidean.protocols import (
        DesignCallbackProtocol as DesignCallbackProtocol,
    )
    from geneuclidean.protocols import (
        MolecularDecoderProtocol as MolecularDecoderProtocol,
    )
    from geneuclidean.protocols import (
        MoleculeFilterProtocol as MoleculeFilterProtocol,
    )
    from geneuclidean.protocols import (
        MoleculeScorerProtocol as MoleculeScorerProtocol,
    )
    from geneuclidean.protocols import (
        PocketEncoderProtocol as PocketEncoderProtocol,
    )
    from geneuclidean.protocols import (
        SamplingStrategyProtocol as SamplingStrategyProtocol,
    )
    from geneuclidean.registry import (
        DECODER_REGISTRY as DECODER_REGISTRY,
    )
    from geneuclidean.registry import (
        ENCODER_REGISTRY as ENCODER_REGISTRY,
    )
    from geneuclidean.registry import (
        FILTER_REGISTRY as FILTER_REGISTRY,
    )
    from geneuclidean.registry import (
        SAMPLING_STRATEGY_REGISTRY as SAMPLING_STRATEGY_REGISTRY,
    )
    from geneuclidean.registry import (
        SCORER_REGISTRY as SCORER_REGISTRY,
    )
    from geneuclidean.registry import (
        Registry as Registry,
    )
    from geneuclidean.registry import (
        discover_plugins as discover_plugins,
    )
    from geneuclidean.registry import (
        list_decoders as list_decoders,
    )
    from geneuclidean.registry import (
        list_encoders as list_encoders,
    )
    from geneuclidean.registry import (
        list_filters as list_filters,
    )
    from geneuclidean.registry import (
        list_sampling_strategies as list_sampling_strategies,
    )
    from geneuclidean.registry import (
        list_scorers as list_scorers,
    )
    from geneuclidean.registry import (
        register_backbone_metadata as register_backbone_metadata,
    )
    from geneuclidean.registry import (
        register_decoder as register_decoder,
    )
    from geneuclidean.registry import (
        register_encoder as register_encoder,
    )
    from geneuclidean.registry import (
        register_filter as register_filter,
    )
    from geneuclidean.registry import (
        register_sampling_strategy as register_sampling_strategy,
    )
    from geneuclidean.registry import (
        register_scorer as register_scorer,
    )
    from geneuclidean.sampling.pocket_steering import (
        PocketPMIScore as PocketPMIScore,
    )
    from geneuclidean.sampling.pocket_steering import (
        combine_contrastive_ensemble_logits as combine_contrastive_ensemble_logits,
    )
    from geneuclidean.sampling.pocket_steering import (
        sample_contrastive_ensemble_llm as sample_contrastive_ensemble_llm,
    )
    from geneuclidean.sampling.pocket_steering import (
        score_pocket_pmi as score_pocket_pmi,
    )
    from geneuclidean.sdk import (
        ADMETFilter as ADMETFilter,
    )
    from geneuclidean.sdk import (
        BaseDesignCallback as BaseDesignCallback,
    )
    from geneuclidean.sdk import (
        BasicAmineFilter as BasicAmineFilter,
    )
    from geneuclidean.sdk import (
        CredibilityDossier as CredibilityDossier,
    )
    from geneuclidean.sdk import (
        DesignPipeline as DesignPipeline,
    )
    from geneuclidean.sdk import (
        DesignResult as DesignResult,
    )
    from geneuclidean.sdk import (
        FunctionScorer as FunctionScorer,
    )
    from geneuclidean.sdk import (
        MoleculeDesigner as MoleculeDesigner,
    )
    from geneuclidean.sdk import (
        MoleculeReport as MoleculeReport,
    )
    from geneuclidean.sdk import (
        MPOFilter as MPOFilter,
    )
    from geneuclidean.sdk import (
        Pocket as Pocket,
    )
    from geneuclidean.sdk import (
        PredicateFilter as PredicateFilter,
    )
    from geneuclidean.sdk import (
        QEDFilter as QEDFilter,
    )
    from geneuclidean.sdk import (
        ScaffoldHopFilter as ScaffoldHopFilter,
    )
    from geneuclidean.sdk import (
        TriageFilter as TriageFilter,
    )
    from geneuclidean.sdk import (
        audit_tourette_circuit as audit_tourette_circuit,
    )
    from geneuclidean.sdk import (
        build_credibility_dossier as build_credibility_dossier,
    )
    from geneuclidean.sdk import (
        dock_and_profile as dock_and_profile,
    )
    from geneuclidean.sdk import (
        export_credibility_json as export_credibility_json,
    )
    from geneuclidean.sdk import (
        export_credibility_markdown as export_credibility_markdown,
    )
    from geneuclidean.sdk import (
        get_target as get_target,
    )
    from geneuclidean.sdk import (
        list_backbones as list_backbones,
    )
    from geneuclidean.sdk import (
        list_targets as list_targets,
    )
    from geneuclidean.sdk import (
        protonate_ph74 as protonate_ph74,
    )
    from geneuclidean.sdk import (
        redock_complex as redock_complex,
    )
    from geneuclidean.sdk import (
        register_target as register_target,
    )
    from geneuclidean.sdk import (
        score_molecules as score_molecules,
    )
    from geneuclidean.sdk import (
        score_tourette_circuit as score_tourette_circuit,
    )
    from geneuclidean.sdk import (
        triage_molecule as triage_molecule,
    )
    from geneuclidean.sdk import (
        triage_molecules as triage_molecules,
    )
    from geneuclidean.sdk import (
        unregister_target as unregister_target,
    )
    from geneuclidean.sdk import (
        visualize_complex as visualize_complex,
    )
    from geneuclidean.sdk import (
        visualize_gallery as visualize_gallery,
    )
    from geneuclidean.sdk import (
        visualize_molecules_2d as visualize_molecules_2d,
    )
    from geneuclidean.training.mtl_intervention import (
        MTL_MEDIATOR_REGISTRY as MTL_MEDIATOR_REGISTRY,
    )
    from geneuclidean.training.mtl_intervention import (
        AdaptiveInterventionPolicy as AdaptiveInterventionPolicy,
    )
    from geneuclidean.training.mtl_intervention import (
        GradientConflictMediator as GradientConflictMediator,
    )
    from geneuclidean.training.mtl_intervention import (
        TaskGradientDiagnostics as TaskGradientDiagnostics,
    )
    from geneuclidean.training.mtl_intervention import (
        compute_gradient_conflict_matrix as compute_gradient_conflict_matrix,
    )

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        "api",
        "cli",
        "cns",
        "config",
        "data",
        "evaluation",
        "exceptions",
        "models",
        "protocols",
        "registry",
        "sampling",
        "sdk",
        "training",
    }
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Structured Exceptions (`geneuclidean.exceptions`)
    "CheckpointError": ("geneuclidean.exceptions", "CheckpointError"),
    "DockingError": ("geneuclidean.exceptions", "DockingError"),
    "GenEuclideanError": ("geneuclidean.exceptions", "GenEuclideanError"),
    "InvalidMoleculeError": ("geneuclidean.exceptions", "InvalidMoleculeError"),
    "OptionalDependencyError": ("geneuclidean.exceptions", "OptionalDependencyError"),
    "PocketFeaturizationError": ("geneuclidean.exceptions", "PocketFeaturizationError"),
    "RegistryError": ("geneuclidean.exceptions", "RegistryError"),
    "StructureNotFoundError": ("geneuclidean.exceptions", "StructureNotFoundError"),
    "TargetNotFoundError": ("geneuclidean.exceptions", "TargetNotFoundError"),
    # Structural Typing Protocols (`geneuclidean.protocols`)
    "DesignCallbackProtocol": ("geneuclidean.protocols", "DesignCallbackProtocol"),
    "MolecularDecoderProtocol": ("geneuclidean.protocols", "MolecularDecoderProtocol"),
    "MoleculeFilterProtocol": ("geneuclidean.protocols", "MoleculeFilterProtocol"),
    "MoleculeScorerProtocol": ("geneuclidean.protocols", "MoleculeScorerProtocol"),
    "PocketEncoderProtocol": ("geneuclidean.protocols", "PocketEncoderProtocol"),
    "SamplingStrategyProtocol": ("geneuclidean.protocols", "SamplingStrategyProtocol"),
    # Plugin Registries (`geneuclidean.registry`)
    "DECODER_REGISTRY": ("geneuclidean.registry", "DECODER_REGISTRY"),
    "ENCODER_REGISTRY": ("geneuclidean.registry", "ENCODER_REGISTRY"),
    "FILTER_REGISTRY": ("geneuclidean.registry", "FILTER_REGISTRY"),
    "Registry": ("geneuclidean.registry", "Registry"),
    "SAMPLING_STRATEGY_REGISTRY": ("geneuclidean.registry", "SAMPLING_STRATEGY_REGISTRY"),
    "SCORER_REGISTRY": ("geneuclidean.registry", "SCORER_REGISTRY"),
    "discover_plugins": ("geneuclidean.registry", "discover_plugins"),
    "list_decoders": ("geneuclidean.registry", "list_decoders"),
    "list_encoders": ("geneuclidean.registry", "list_encoders"),
    "list_filters": ("geneuclidean.registry", "list_filters"),
    "list_sampling_strategies": ("geneuclidean.registry", "list_sampling_strategies"),
    "list_scorers": ("geneuclidean.registry", "list_scorers"),
    "register_backbone_metadata": ("geneuclidean.registry", "register_backbone_metadata"),
    "register_decoder": ("geneuclidean.registry", "register_decoder"),
    "register_encoder": ("geneuclidean.registry", "register_encoder"),
    "register_filter": ("geneuclidean.registry", "register_filter"),
    "register_sampling_strategy": ("geneuclidean.registry", "register_sampling_strategy"),
    "register_scorer": ("geneuclidean.registry", "register_scorer"),
    # Modular High-Level SDK (`geneuclidean.sdk.*`)
    "ADMETFilter": ("geneuclidean.sdk.scoring", "ADMETFilter"),
    "BaseDesignCallback": ("geneuclidean.sdk.pipeline", "BaseDesignCallback"),
    "BasicAmineFilter": ("geneuclidean.sdk.scoring", "BasicAmineFilter"),
    "CredibilityDossier": ("geneuclidean.sdk.credibility", "CredibilityDossier"),
    "DesignPipeline": ("geneuclidean.sdk.pipeline", "DesignPipeline"),
    "DesignResult": ("geneuclidean.sdk.designer", "DesignResult"),
    "FunctionScorer": ("geneuclidean.sdk.scoring", "FunctionScorer"),
    "MPOFilter": ("geneuclidean.sdk.scoring", "MPOFilter"),
    "MoleculeDesigner": ("geneuclidean.sdk.designer", "MoleculeDesigner"),
    "MoleculeReport": ("geneuclidean.sdk.scoring", "MoleculeReport"),
    "Pocket": ("geneuclidean.sdk.pocket", "Pocket"),
    "PredicateFilter": ("geneuclidean.sdk.scoring", "PredicateFilter"),
    "QEDFilter": ("geneuclidean.sdk.scoring", "QEDFilter"),
    "ScaffoldHopFilter": ("geneuclidean.sdk.scoring", "ScaffoldHopFilter"),
    "TriageFilter": ("geneuclidean.sdk.scoring", "TriageFilter"),
    "audit_tourette_circuit": ("geneuclidean.sdk.registries", "audit_tourette_circuit"),
    "build_credibility_dossier": ("geneuclidean.sdk.credibility", "build_credibility_dossier"),
    "dock_and_profile": ("geneuclidean.sdk.docking", "dock_and_profile"),
    "export_credibility_json": ("geneuclidean.sdk.credibility", "export_credibility_json"),
    "export_credibility_markdown": ("geneuclidean.sdk.credibility", "export_credibility_markdown"),
    "get_target": ("geneuclidean.sdk.registries", "get_target"),
    "list_backbones": ("geneuclidean.sdk.registries", "list_backbones"),
    "list_targets": ("geneuclidean.sdk.registries", "list_targets"),
    "protonate_ph74": ("geneuclidean.sdk.scoring", "protonate_ph74"),
    "redock_complex": ("geneuclidean.sdk.docking", "redock_complex"),
    "register_target": ("geneuclidean.sdk.registries", "register_target"),
    "score_molecules": ("geneuclidean.sdk.scoring", "score_molecules"),
    "score_tourette_circuit": ("geneuclidean.sdk.scoring", "score_tourette_circuit"),
    "triage_molecule": ("geneuclidean.sdk.scoring", "triage_molecule"),
    "triage_molecules": ("geneuclidean.sdk.scoring", "triage_molecules"),
    "unregister_target": ("geneuclidean.sdk.registries", "unregister_target"),
    "visualize_complex": ("geneuclidean.sdk.viz", "visualize_complex"),
    "visualize_gallery": ("geneuclidean.sdk.viz", "visualize_gallery"),
    "visualize_molecules_2d": ("geneuclidean.sdk.viz", "visualize_molecules_2d"),
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
    # Saturated-Ring Conformational Strain & Cremer-Pople (`geneuclidean.evaluation.ring_strain`)
    "RING_STRAIN_BACKEND_REGISTRY": (
        "geneuclidean.evaluation.ring_strain",
        "RING_STRAIN_BACKEND_REGISTRY",
    ),
    "RingStrainFilter": ("geneuclidean.evaluation.ring_strain", "RingStrainFilter"),
    "RingStrainProfile": ("geneuclidean.evaluation.ring_strain", "RingStrainProfile"),
    "RingStrainScorer": ("geneuclidean.evaluation.ring_strain", "RingStrainScorer"),
    "SaturatedRingReport": ("geneuclidean.evaluation.ring_strain", "SaturatedRingReport"),
    "analyze_ring_strain": ("geneuclidean.evaluation.ring_strain", "analyze_ring_strain"),
    "compute_cremer_pople": ("geneuclidean.evaluation.ring_strain", "compute_cremer_pople"),
    "register_ring_strain_backend": (
        "geneuclidean.evaluation.ring_strain",
        "register_ring_strain_backend",
    ),
    # Multi-Conformation Ensemble & Contrastive Steering (`geneuclidean.sampling.pocket_steering`)
    "PocketPMIScore": ("geneuclidean.sampling.pocket_steering", "PocketPMIScore"),
    "combine_contrastive_ensemble_logits": (
        "geneuclidean.sampling.pocket_steering",
        "combine_contrastive_ensemble_logits",
    ),
    "sample_contrastive_ensemble_llm": (
        "geneuclidean.sampling.pocket_steering",
        "sample_contrastive_ensemble_llm",
    ),
    "score_pocket_pmi": ("geneuclidean.sampling.pocket_steering", "score_pocket_pmi"),
    # Natural-Product / Clinical Scaffold-Hopping Audit (`geneuclidean.cns.scaffold_hop`)
    "IBOGAINE_SMILES": ("geneuclidean.cns.scaffold_hop", "IBOGAINE_SMILES"),
    "PAROXETINE_SMILES": ("geneuclidean.cns.scaffold_hop", "PAROXETINE_SMILES"),
    "ScaffoldHopCandidateAudit": (
        "geneuclidean.cns.scaffold_hop",
        "ScaffoldHopCandidateAudit",
    ),
    "audit_scaffold_hop_campaign": (
        "geneuclidean.cns.scaffold_hop",
        "audit_scaffold_hop_campaign",
    ),
    # Multi-Task Gradient Conflict Mediation (`geneuclidean.training.mtl_intervention`)
    "AdaptiveInterventionPolicy": (
        "geneuclidean.training.mtl_intervention",
        "AdaptiveInterventionPolicy",
    ),
    "GradientConflictMediator": (
        "geneuclidean.training.mtl_intervention",
        "GradientConflictMediator",
    ),
    "MTL_MEDIATOR_REGISTRY": (
        "geneuclidean.training.mtl_intervention",
        "MTL_MEDIATOR_REGISTRY",
    ),
    "TaskGradientDiagnostics": (
        "geneuclidean.training.mtl_intervention",
        "TaskGradientDiagnostics",
    ),
    "compute_gradient_conflict_matrix": (
        "geneuclidean.training.mtl_intervention",
        "compute_gradient_conflict_matrix",
    ),
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
