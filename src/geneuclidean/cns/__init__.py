"""Industry-standard CNS neuroscience drug-design modules for GenEuclidean v3.1.

Provides:
- ``mpo``: Exact Pfizer CNS MPO v1 (Wager et al. 2010) and v2 (Wager et al. 2016)
  desirability scoring, physicochemical SMARTS pKa / Henderson-Hasselbalch ClogD_7.4
  estimation, and Clark/Kelder BBB heuristic filters.
- ``admet``: Calibrated 5-fold Bemis-Murcko scaffold-CV CNS ADMET & safety oracles
  (B3DB BBB+, TDC P-gp Broccatelli inhibition, TDC hERG Karim cardiotoxicity) with
  ECFP4 Tanimoto applicability domain (AD) and composite CNS reward scoring.
- ``targets``: Curated 12-target human Tourette Syndrome and neuropsychiatric 3D target
  registry with verified UniProt/ChEMBL/RCSB accessions, clinical reference drugs,
  canonical 3D hotspot residues, and 3D pocket loaders.
"""

from __future__ import annotations

from geneuclidean.cns.admet import (
    ADMET_DESCRIPTOR_NAMES,
    DEFAULT_ADMET_MODEL_PATH,
    DEFAULT_CNS_REWARD_WEIGHTS,
    CalibratedADMETOracle,
    CNSAdmetPrediction,
    CNSAdmetSuite,
    CNSCompositeScore,
    compute_cns_composite_reward,
    compute_ece,
    featurize_admet_smiles,
    make_scaffold_kfold,
    train_default_cns_admet_suite,
)
from geneuclidean.cns.mpo import (
    REFERENCE_CNS_DRUGS,
    REFERENCE_NON_CNS_DRUGS,
    CNSMPOResult,
    compute_cns_mpo,
    compute_cns_mpo_batch,
    desirability_clogd,
    desirability_clogp,
    desirability_hbd,
    desirability_mw,
    desirability_pka,
    desirability_tpsa,
    estimate_clogd_74,
    estimate_most_acidic_pka,
    estimate_most_basic_pka,
)
from geneuclidean.cns.targets import (
    CNS_TARGET_REGISTRY,
    DEFAULT_CNS_STRUCTURES_DIR,
    ClinicalReferenceDrug,
    CNSTargetSpec,
    audit_pocket_hotspots,
    fetch_rcsb_structure,
    find_local_pocket_files,
    get_cns_target,
    list_cns_targets,
    load_cns_target_pocket,
)

__all__ = [
    "ADMET_DESCRIPTOR_NAMES",
    "CNS_TARGET_REGISTRY",
    "DEFAULT_ADMET_MODEL_PATH",
    "DEFAULT_CNS_REWARD_WEIGHTS",
    "DEFAULT_CNS_STRUCTURES_DIR",
    "REFERENCE_CNS_DRUGS",
    "REFERENCE_NON_CNS_DRUGS",
    "CNSAdmetPrediction",
    "CNSAdmetSuite",
    "CNSCompositeScore",
    "CNSMPOResult",
    "CNSTargetSpec",
    "CalibratedADMETOracle",
    "ClinicalReferenceDrug",
    "audit_pocket_hotspots",
    "compute_cns_composite_reward",
    "compute_cns_mpo",
    "compute_cns_mpo_batch",
    "compute_ece",
    "desirability_clogd",
    "desirability_clogp",
    "desirability_hbd",
    "desirability_mw",
    "desirability_pka",
    "desirability_tpsa",
    "estimate_clogd_74",
    "estimate_most_acidic_pka",
    "estimate_most_basic_pka",
    "featurize_admet_smiles",
    "fetch_rcsb_structure",
    "find_local_pocket_files",
    "get_cns_target",
    "list_cns_targets",
    "load_cns_target_pocket",
    "make_scaffold_kfold",
    "train_default_cns_admet_suite",
]
