"""GenEuclidean: SE(3)-Equivariant 3D Molecular Encoding, Chemical NLP, NanoFold3 Co-Folding,
Active Learning & Apple Circuit Studio SDK for Structure-Based Drug Design.

Implements PEP 562 module-level lazy loading (``__getattr__`` and ``__dir__``) with full
PEP 561 static type support (``if TYPE_CHECKING:`` + ``py.typed``) so ``import geneuclidean as ge``
completes in ``< 10 ms`` without eagerly importing ``torch``, ``transformers``, ``peft``,
``sklearn``, ``scipy``, ``rdkit``, ``matplotlib``, or ``Bio``.

Quickstart::

    import geneuclidean as ge

    targets = ge.list_targets("tourette")
    pocket = ge.Pocket.from_target("DRD1")
    reports = ge.score_molecules(["CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21"])
    nf3 = ge.build_nanofold3("nano", trunk_mode="simplefold", diffusion_mode="flow_matching")

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

__version__ = "5.3.0"

if TYPE_CHECKING:
    from geneuclidean import (
        active_learning as active_learning,
    )
    from geneuclidean import (
        api as api,
    )
    from geneuclidean import (
        audit as audit,
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
        folding as folding,
    )
    from geneuclidean import (
        geom as geom,
    )
    from geneuclidean import (
        interop as interop,
    )
    from geneuclidean import (
        leadopt as leadopt,
    )
    from geneuclidean import (
        mcp_server as mcp_server,
    )
    from geneuclidean import (
        models as models,
    )
    from geneuclidean import (
        nlp as nlp,
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
        screening as screening,
    )
    from geneuclidean import (
        sdk as sdk,
    )
    from geneuclidean import (
        studio as studio,
    )
    from geneuclidean import (
        synthesis as synthesis,
    )
    from geneuclidean import (
        training as training,
    )
    from geneuclidean.active_learning import (
        ACQUISITION_REGISTRY as ACQUISITION_REGISTRY,
    )
    from geneuclidean.active_learning import (
        SURROGATE_REGISTRY as SURROGATE_REGISTRY,
    )
    from geneuclidean.active_learning import (
        ActiveLearningEngine as ActiveLearningEngine,
    )
    from geneuclidean.active_learning import (
        ColdStartSelector as ColdStartSelector,
    )
    from geneuclidean.active_learning import (
        CombinatorialThompsonSampler as CombinatorialThompsonSampler,
    )
    from geneuclidean.active_learning import (
        ConformalSurrogate as ConformalSurrogate,
    )
    from geneuclidean.active_learning import (
        DMTALedger as DMTALedger,
    )
    from geneuclidean.active_learning import (
        DMTAQuery as DMTAQuery,
    )
    from geneuclidean.active_learning import (
        DMTARoundSummary as DMTARoundSummary,
    )
    from geneuclidean.active_learning import (
        EnsembleSurrogate as EnsembleSurrogate,
    )
    from geneuclidean.active_learning import (
        MultiFidelityDMTA as MultiFidelityDMTA,
    )
    from geneuclidean.active_learning import (
        PendingBatchManager as PendingBatchManager,
    )
    from geneuclidean.active_learning import (
        StoppingAdvisor as StoppingAdvisor,
    )
    from geneuclidean.active_learning import (
        TanimotoGP as TanimotoGP,
    )
    from geneuclidean.active_learning import (
        evaluate_uq_calibration as evaluate_uq_calibration,
    )
    from geneuclidean.active_learning import (
        list_acquisitions as list_acquisitions,
    )
    from geneuclidean.active_learning import (
        list_surrogates as list_surrogates,
    )
    from geneuclidean.active_learning import (
        register_acquisition as register_acquisition,
    )
    from geneuclidean.active_learning import (
        register_surrogate as register_surrogate,
    )
    from geneuclidean.audit import (
        PoseAuditReport as PoseAuditReport,
    )
    from geneuclidean.audit import (
        PoseRealityAuditor as PoseRealityAuditor,
    )
    from geneuclidean.audit import (
        audit_complex as audit_complex,
    )
    from geneuclidean.audit import (
        export_audit_json as export_audit_json,
    )
    from geneuclidean.audit import (
        summarize_audit_batch as summarize_audit_batch,
    )
    from geneuclidean.cns.dmpk_regression import (
        ConformalDMPKRegressor as ConformalDMPKRegressor,
    )
    from geneuclidean.cns.dmpk_regression import (
        ConformalRegressionPrediction as ConformalRegressionPrediction,
    )
    from geneuclidean.cns.dmpk_regression import (
        evaluate_scaffold_cv_conformal_dmpk as evaluate_scaffold_cv_conformal_dmpk,
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
    from geneuclidean.evaluation.tautomer_microstate import (
        MicrostateCandidate as MicrostateCandidate,
    )
    from geneuclidean.evaluation.tautomer_microstate import (
        TautomerMicrostateAudit as TautomerMicrostateAudit,
    )
    from geneuclidean.evaluation.tautomer_microstate import (
        audit_tautomer_microstate_ph74 as audit_tautomer_microstate_ph74,
    )
    from geneuclidean.evaluation.tautomer_microstate import (
        enumerate_tautomer_microstates_ph74 as enumerate_tautomer_microstates_ph74,
    )
    from geneuclidean.evaluation.torsion_strain import (
        TORSION_BACKEND_REGISTRY as TORSION_BACKEND_REGISTRY,
    )
    from geneuclidean.evaluation.torsion_strain import (
        TorsionReport as TorsionReport,
    )
    from geneuclidean.evaluation.torsion_strain import (
        TorsionStrainFilter as TorsionStrainFilter,
    )
    from geneuclidean.evaluation.torsion_strain import (
        TorsionStrainProfile as TorsionStrainProfile,
    )
    from geneuclidean.evaluation.torsion_strain import (
        TorsionStrainScorer as TorsionStrainScorer,
    )
    from geneuclidean.evaluation.torsion_strain import (
        analyze_torsion_strain as analyze_torsion_strain,
    )
    from geneuclidean.evaluation.torsion_strain import (
        load_torsion_library_xml as load_torsion_library_xml,
    )
    from geneuclidean.evaluation.torsion_strain import (
        register_torsion_backend as register_torsion_backend,
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
    from geneuclidean.folding.nanofold3 import (
        NanoFold3 as NanoFold3,
    )
    from geneuclidean.folding.nanofold3 import (
        NanoFold3Config as NanoFold3Config,
    )
    from geneuclidean.folding.nanofold3 import (
        build_nanofold3 as build_nanofold3,
    )
    from geneuclidean.folding.runners import (
        FOLDING_BACKEND_REGISTRY as FOLDING_BACKEND_REGISTRY,
    )
    from geneuclidean.folding.runners import (
        fold_complex as fold_complex,
    )
    from geneuclidean.interop import (
        OpenFENetworkBundle as OpenFENetworkBundle,
    )
    from geneuclidean.interop import (
        OpenFEPerturbationEdge as OpenFEPerturbationEdge,
    )
    from geneuclidean.interop import (
        audit_cofolded_complex as audit_cofolded_complex,
    )
    from geneuclidean.interop import (
        export_openfe_rbfe_network as export_openfe_rbfe_network,
    )
    from geneuclidean.interop import (
        load_cofolded_complex as load_cofolded_complex,
    )
    from geneuclidean.leadopt import (
        ActivityCliffPair as ActivityCliffPair,
    )
    from geneuclidean.leadopt import (
        AvoidomeMMPEngine as AvoidomeMMPEngine,
    )
    from geneuclidean.leadopt import (
        AvoidomeMMPProposal as AvoidomeMMPProposal,
    )
    from geneuclidean.leadopt import (
        RGroupSARResult as RGroupSARResult,
    )
    from geneuclidean.leadopt import (
        analyze_rgroup_sar as analyze_rgroup_sar,
    )
    from geneuclidean.mcp_server import (
        DMTA_TOOL_SCHEMAS as DMTA_TOOL_SCHEMAS,
    )
    from geneuclidean.mcp_server import (
        invoke_dmta_tool as invoke_dmta_tool,
    )
    from geneuclidean.mcp_server import (
        run_mcp_stdio_server as run_mcp_stdio_server,
    )
    from geneuclidean.models.encoders.atomgs import (
        AtomGSSE3Encoder as AtomGSSE3Encoder,
    )
    from geneuclidean.models.encoders.cartformer import (
        CartFormerSE3Encoder as CartFormerSE3Encoder,
    )
    from geneuclidean.models.encoders.dense_painn import (
        DensePaiNNSE3Encoder as DensePaiNNSE3Encoder,
    )
    from geneuclidean.models.encoders.subpocket_vn import (
        SubPocketVNJEPAEncoder as SubPocketVNJEPAEncoder,
    )
    from geneuclidean.models.nlp_ops import (
        AtomLevelSMILESTokenizer as AtomLevelSMILESTokenizer,
    )
    from geneuclidean.models.nlp_ops import (
        DiscreteMaskedDiffusionStep as DiscreteMaskedDiffusionStep,
    )
    from geneuclidean.models.nlp_ops import (
        PocketConditionedChemicalLM as PocketConditionedChemicalLM,
    )
    from geneuclidean.models.nlp_ops import (
        SMILESValenceGrammarMask as SMILESValenceGrammarMask,
    )
    from geneuclidean.models.nlp_ops import (
        dpo_preference_loss as dpo_preference_loss,
    )
    from geneuclidean.models.nlp_ops import (
        grpo_policy_loss as grpo_policy_loss,
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
    from geneuclidean.screening import (
        LIGAND_ENCODER_REGISTRY as LIGAND_ENCODER_REGISTRY,
    )
    from geneuclidean.screening import (
        ActiveLearningDMTA as ActiveLearningDMTA,
    )
    from geneuclidean.screening import (
        ActiveLearningRoundSummary as ActiveLearningRoundSummary,
    )
    from geneuclidean.screening import (
        DrugCLIPLigandEncoder as DrugCLIPLigandEncoder,
    )
    from geneuclidean.screening import (
        ECFP4BinaryEncoder as ECFP4BinaryEncoder,
    )
    from geneuclidean.screening import (
        IBMMMelonLigandEncoder as IBMMMelonLigandEncoder,
    )
    from geneuclidean.screening import (
        LigandVectorIndex as LigandVectorIndex,
    )
    from geneuclidean.screening import (
        PocketScreener as PocketScreener,
    )
    from geneuclidean.screening import (
        ScreeningHit as ScreeningHit,
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
    from geneuclidean.sdk.profiles import (
        PROPERTY_PROFILE_REGISTRY as PROPERTY_PROFILE_REGISTRY,
    )
    from geneuclidean.sdk.profiles import (
        PropertyProfileResult as PropertyProfileResult,
    )
    from geneuclidean.sdk.profiles import (
        evaluate_property_profile as evaluate_property_profile,
    )
    from geneuclidean.sdk.profiles import (
        list_property_profiles as list_property_profiles,
    )
    from geneuclidean.sdk.profiles import (
        register_property_profile as register_property_profile,
    )
    from geneuclidean.studio import (
        compile_circuit_to_python as compile_circuit_to_python,
    )
    from geneuclidean.studio import (
        execute_circuit as execute_circuit,
    )
    from geneuclidean.studio import (
        export_catalog_payload as export_catalog_payload,
    )
    from geneuclidean.studio import (
        get_studio_catalog as get_studio_catalog,
    )
    from geneuclidean.studio import (
        run_studio_server as run_studio_server,
    )
    from geneuclidean.studio import (
        serve_studio as serve_studio,
    )
    from geneuclidean.studio import (
        validate_circuit as validate_circuit,
    )
    from geneuclidean.synthesis import (
        RETROSYNTHESIS_REGISTRY as RETROSYNTHESIS_REGISTRY,
    )
    from geneuclidean.synthesis import (
        ROBUST_MEDCHEM_REACTIONS as ROBUST_MEDCHEM_REACTIONS,
    )
    from geneuclidean.synthesis import (
        MakeableCandidate as MakeableCandidate,
    )
    from geneuclidean.synthesis import (
        enumerate_makeable_analogs as enumerate_makeable_analogs,
    )
    from geneuclidean.synthesis import (
        match_one_step_retrosynthesis as match_one_step_retrosynthesis,
    )
    from geneuclidean.synthesis import (
        register_retrosynthesis_backend as register_retrosynthesis_backend,
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
        "active_learning",
        "api",
        "audit",
        "cli",
        "cns",
        "config",
        "data",
        "evaluation",
        "exceptions",
        "folding",
        "geom",
        "interop",
        "leadopt",
        "mcp_server",
        "models",
        "nlp",
        "protocols",
        "registry",
        "sampling",
        "screening",
        "sdk",
        "studio",
        "synthesis",
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
    # Beyond-PoseBusters 7-Axis Pose Auditor, Torsion Strain, Tautomer/Microstate & Profiles
    "MicrostateCandidate": ("geneuclidean.evaluation.tautomer_microstate", "MicrostateCandidate"),
    "PROPERTY_PROFILE_REGISTRY": ("geneuclidean.sdk.profiles", "PROPERTY_PROFILE_REGISTRY"),
    "PoseAuditReport": ("geneuclidean.audit", "PoseAuditReport"),
    "PoseRealityAuditor": ("geneuclidean.audit", "PoseRealityAuditor"),
    "PropertyProfileResult": ("geneuclidean.sdk.profiles", "PropertyProfileResult"),
    "TORSION_BACKEND_REGISTRY": (
        "geneuclidean.evaluation.torsion_strain",
        "TORSION_BACKEND_REGISTRY",
    ),
    "TautomerMicrostateAudit": (
        "geneuclidean.evaluation.tautomer_microstate",
        "TautomerMicrostateAudit",
    ),
    "TorsionReport": ("geneuclidean.evaluation.torsion_strain", "TorsionReport"),
    "TorsionStrainFilter": ("geneuclidean.evaluation.torsion_strain", "TorsionStrainFilter"),
    "TorsionStrainProfile": ("geneuclidean.evaluation.torsion_strain", "TorsionStrainProfile"),
    "TorsionStrainScorer": ("geneuclidean.evaluation.torsion_strain", "TorsionStrainScorer"),
    "analyze_torsion_strain": ("geneuclidean.evaluation.torsion_strain", "analyze_torsion_strain"),
    "audit_complex": ("geneuclidean.audit", "audit_complex"),
    "audit_tautomer_microstate_ph74": (
        "geneuclidean.evaluation.tautomer_microstate",
        "audit_tautomer_microstate_ph74",
    ),
    "enumerate_tautomer_microstates_ph74": (
        "geneuclidean.evaluation.tautomer_microstate",
        "enumerate_tautomer_microstates_ph74",
    ),
    "evaluate_property_profile": ("geneuclidean.sdk.profiles", "evaluate_property_profile"),
    "export_audit_json": ("geneuclidean.audit", "export_audit_json"),
    "list_property_profiles": ("geneuclidean.sdk.profiles", "list_property_profiles"),
    "load_torsion_library_xml": (
        "geneuclidean.evaluation.torsion_strain",
        "load_torsion_library_xml",
    ),
    "register_property_profile": ("geneuclidean.sdk.profiles", "register_property_profile"),
    "register_torsion_backend": (
        "geneuclidean.evaluation.torsion_strain",
        "register_torsion_backend",
    ),
    "summarize_audit_batch": ("geneuclidean.audit", "summarize_audit_batch"),
    # MIT Contrastive 3D Virtual Screening & Active-Learning DMTA (`geneuclidean.screening`)
    "ActiveLearningDMTA": ("geneuclidean.screening", "ActiveLearningDMTA"),
    "ActiveLearningRoundSummary": ("geneuclidean.screening", "ActiveLearningRoundSummary"),
    "DrugCLIPLigandEncoder": ("geneuclidean.screening", "DrugCLIPLigandEncoder"),
    "ECFP4BinaryEncoder": ("geneuclidean.screening", "ECFP4BinaryEncoder"),
    "IBMMMelonLigandEncoder": ("geneuclidean.screening", "IBMMMelonLigandEncoder"),
    "LIGAND_ENCODER_REGISTRY": ("geneuclidean.screening", "LIGAND_ENCODER_REGISTRY"),
    "LigandVectorIndex": ("geneuclidean.screening", "LigandVectorIndex"),
    "PocketScreener": ("geneuclidean.screening", "PocketScreener"),
    "ScreeningHit": ("geneuclidean.screening", "ScreeningHit"),
    # Makeable Synthesis, Avoid-ome MMP Lead-Opt, OpenFE/Boltz-2 Interop, Conformal DMPK & MCP
    "ActivityCliffPair": ("geneuclidean.leadopt", "ActivityCliffPair"),
    "AvoidomeMMPEngine": ("geneuclidean.leadopt", "AvoidomeMMPEngine"),
    "AvoidomeMMPProposal": ("geneuclidean.leadopt", "AvoidomeMMPProposal"),
    "ConformalDMPKRegressor": ("geneuclidean.cns.dmpk_regression", "ConformalDMPKRegressor"),
    "ConformalRegressionPrediction": (
        "geneuclidean.cns.dmpk_regression",
        "ConformalRegressionPrediction",
    ),
    "DMTA_TOOL_SCHEMAS": ("geneuclidean.mcp_server", "DMTA_TOOL_SCHEMAS"),
    "MakeableCandidate": ("geneuclidean.synthesis", "MakeableCandidate"),
    "OpenFENetworkBundle": ("geneuclidean.interop", "OpenFENetworkBundle"),
    "OpenFEPerturbationEdge": ("geneuclidean.interop", "OpenFEPerturbationEdge"),
    "RETROSYNTHESIS_REGISTRY": ("geneuclidean.synthesis", "RETROSYNTHESIS_REGISTRY"),
    "RGroupSARResult": ("geneuclidean.leadopt", "RGroupSARResult"),
    "ROBUST_MEDCHEM_REACTIONS": ("geneuclidean.synthesis", "ROBUST_MEDCHEM_REACTIONS"),
    "analyze_rgroup_sar": ("geneuclidean.leadopt", "analyze_rgroup_sar"),
    "audit_cofolded_complex": ("geneuclidean.interop", "audit_cofolded_complex"),
    "enumerate_makeable_analogs": ("geneuclidean.synthesis", "enumerate_makeable_analogs"),
    "evaluate_scaffold_cv_conformal_dmpk": (
        "geneuclidean.cns.dmpk_regression",
        "evaluate_scaffold_cv_conformal_dmpk",
    ),
    "export_openfe_rbfe_network": ("geneuclidean.interop", "export_openfe_rbfe_network"),
    "invoke_dmta_tool": ("geneuclidean.mcp_server", "invoke_dmta_tool"),
    "load_cofolded_complex": ("geneuclidean.interop", "load_cofolded_complex"),
    "match_one_step_retrosynthesis": ("geneuclidean.synthesis", "match_one_step_retrosynthesis"),
    "register_retrosynthesis_backend": (
        "geneuclidean.synthesis",
        "register_retrosynthesis_backend",
    ),
    "run_mcp_stdio_server": ("geneuclidean.mcp_server", "run_mcp_stdio_server"),
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
    # Frontier Active Learning & DMTA Optimization (`geneuclidean.active_learning`)
    "ACQUISITION_REGISTRY": ("geneuclidean.active_learning.acquisitions", "ACQUISITION_REGISTRY"),
    "ActiveLearningEngine": ("geneuclidean.active_learning.engine", "ActiveLearningEngine"),
    "ColdStartSelector": ("geneuclidean.active_learning.diagnostics", "ColdStartSelector"),
    "CombinatorialThompsonSampler": (
        "geneuclidean.active_learning.combinatorial_ts",
        "CombinatorialThompsonSampler",
    ),
    "ConformalSurrogate": ("geneuclidean.active_learning.surrogates", "ConformalSurrogate"),
    "DMTALedger": ("geneuclidean.active_learning.engine", "DMTALedger"),
    "DMTAQuery": ("geneuclidean.active_learning.engine", "DMTAQuery"),
    "DMTARoundSummary": ("geneuclidean.active_learning.engine", "DMTARoundSummary"),
    "EnsembleSurrogate": ("geneuclidean.active_learning.surrogates", "EnsembleSurrogate"),
    "MultiFidelityDMTA": ("geneuclidean.active_learning.multifidelity", "MultiFidelityDMTA"),
    "PendingBatchManager": ("geneuclidean.active_learning.engine", "PendingBatchManager"),
    "SURROGATE_REGISTRY": ("geneuclidean.active_learning.surrogates", "SURROGATE_REGISTRY"),
    "StoppingAdvisor": ("geneuclidean.active_learning.diagnostics", "StoppingAdvisor"),
    "TanimotoGP": ("geneuclidean.active_learning.surrogates", "TanimotoGP"),
    "evaluate_uq_calibration": (
        "geneuclidean.active_learning.diagnostics",
        "evaluate_uq_calibration",
    ),
    "list_acquisitions": ("geneuclidean.active_learning.acquisitions", "list_acquisitions"),
    "list_surrogates": ("geneuclidean.active_learning.surrogates", "list_surrogates"),
    "register_acquisition": ("geneuclidean.active_learning.acquisitions", "register_acquisition"),
    "register_surrogate": ("geneuclidean.active_learning.surrogates", "register_surrogate"),
    # Circuit Studio Compiler, Catalog & Local HTTP Bridge (`geneuclidean.studio`)
    "compile_circuit_to_python": (
        "geneuclidean.studio.graph_compiler",
        "compile_circuit_to_python",
    ),
    "execute_circuit": ("geneuclidean.studio.graph_compiler", "execute_circuit"),
    "export_catalog_payload": ("geneuclidean.studio.catalog", "export_catalog_payload"),
    "get_studio_catalog": ("geneuclidean.studio.catalog", "get_studio_catalog"),
    "run_studio_server": ("geneuclidean.studio.server", "run_studio_server"),
    "serve_studio": ("geneuclidean.studio.server", "serve_studio"),
    "validate_circuit": ("geneuclidean.studio.graph_compiler", "validate_circuit"),
    # Flagship 3D SE(3) Molecular Encoders (`geneuclidean.models.encoders.*`)
    "AtomGSSE3Encoder": ("geneuclidean.models.encoders.atomgs", "AtomGSSE3Encoder"),
    "CartFormerSE3Encoder": ("geneuclidean.models.encoders.cartformer", "CartFormerSE3Encoder"),
    "DensePaiNNSE3Encoder": ("geneuclidean.models.encoders.dense_painn", "DensePaiNNSE3Encoder"),
    "SubPocketVNJEPAEncoder": (
        "geneuclidean.models.encoders.subpocket_vn",
        "SubPocketVNJEPAEncoder",
    ),
    # NanoFold3 & Open-Weights Co-Folding Hub (`geneuclidean.folding`)
    "FOLDING_BACKEND_REGISTRY": ("geneuclidean.folding.runners", "FOLDING_BACKEND_REGISTRY"),
    "NanoFold3": ("geneuclidean.folding.nanofold3", "NanoFold3"),
    "NanoFold3Config": ("geneuclidean.folding.nanofold3", "NanoFold3Config"),
    "build_nanofold3": ("geneuclidean.folding.nanofold3", "build_nanofold3"),
    "fold_complex": ("geneuclidean.folding.runners", "fold_complex"),
    # Chemical NLP, RL Alignment & Discrete Diffusion (`geneuclidean.nlp`)
    "AtomLevelSMILESTokenizer": ("geneuclidean.models.nlp_ops", "AtomLevelSMILESTokenizer"),
    "DiscreteMaskedDiffusionStep": ("geneuclidean.models.nlp_ops", "DiscreteMaskedDiffusionStep"),
    "PocketConditionedChemicalLM": ("geneuclidean.models.nlp_ops", "PocketConditionedChemicalLM"),
    "SMILESValenceGrammarMask": ("geneuclidean.models.nlp_ops", "SMILESValenceGrammarMask"),
    "dpo_preference_loss": ("geneuclidean.models.nlp_ops", "dpo_preference_loss"),
    "grpo_policy_loss": ("geneuclidean.models.nlp_ops", "grpo_policy_loss"),
}

__all__ = sorted({"__version__", *_LAZY_SUBMODULES, *_LAZY_ATTRS.keys()})


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
