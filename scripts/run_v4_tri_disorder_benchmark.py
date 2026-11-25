#!/usr/bin/env python3
"""GenEuclidean v4.0 Flagship Head-to-Head Local LLM & 24-Target Tri-Disorder Benchmark.

Executes the end-to-end GenEuclidean v4.0 experimental pipeline across 24 curated
*Homo sapiens* Tourette Syndrome, Schizophrenia & Alzheimer's Disease targets plus
7 matched structural/Avoid-ome counter-targets on Apple Silicon MPS GPU + parallel CPU
AutoDock Vina v1.2.7:

  - ``prepare``: Builds strict zero-leakage generic Bemis-Murcko 80/10/10 scaffold splits
    across all 24 targets (excluding all clinical drugs, native co-crystal ligands, and
    their exact/generic Bemis-Murcko scaffolds from Train, plus DRD2/MAOA/BACE2 dual
    filtering) and trains calibrated 5-fold scaffold-CV QSAR classifiers for the 12 new
    targets + MAOA + BACE2 combined with the 12 v3.1 QSAR models into
    ``data/cns_neuro.nosync/v4_24target_qsar.joblib``.
  - ``gate1``: Runs unbiased 2D-to-3D multi-seed native redocking across all 12 new human
    TS/SCZ/AD targets + 4 structural AD counter-targets (16 complexes, retaining FAD/ADP/ZN/MG)
    via ``geneuclidean.evaluation.docking_pipeline.redock_native_complex``.
  - ``train``: Trains and aligns all 3 local chemical foundation LLMs on Apple MPS GPU with
    bit-exact frozen ``PaiNN-SE3`` / ``CVAE`` / ``heads`` / base LLM weights:
      1. ``v4.0 GP-MoLFormer-Uniq`` (46.8M, warm-start v3.0 -> 24-target SFT -> Mult + L_XP-DPO)
      2. ``v4.0 SAFE-GPT-87M`` (87.3M, bf16 SAFE -> Phase A+B SFT -> 4-Ch GDPO + L_XP-DPO)
      3. ``v4.0 NovoMolGen-157M`` (157M, bf16 MPS SDPA -> Phase A+B SFT -> 4-Ch GDPO + L_XP-DPO)
    Plus runs the 3D-conditioned SAFE-GPT Scaffold Decoration showcase on 4 clinical cores
    (Ecopipam/DRD1, Donepezil/ACHE hERG rescue, Safinamide/MAOB, Xanomeline/CHRM4 M3 rescue).
  - ``eval``: Evaluates ``v3.1 GP-MoLFormer``, all 3 ``v4.0`` backbones, ``Ctrl-ZINC``, and
    ``v4.0_Best_Shuffled`` (N=64/target x 24 targets = 1,536 molecules/arm) on v3.1 7-rule
    and v4.0 12-endpoint Pharma-Safe %, MedChem Triage & 0.0% DCU/reagent quarantine,
    PDE10A/DRD3 VMAT2-sink elimination, 3D specificity, Wilcoxon/Cliff's delta/1000x bootstrap
    CIs/Holm-Bonferroni FWER, and CASF-2016 retention.
  - ``dock``: Runs 100% held-out AutoDock Vina v1.2.7 across all 24 targets + 5 flagship
    counter-target delta-delta-G selectivity panels (DRD1/DRD2, PDE10A/VMAT2 & CHRM4/CHRM1/DRD2,
    MAOB/MAOA, GSK3B/CDK2, ACHE/BCHE & BACE1/BACE2 + NLRP3/OGA), exports Top-2 docked 3D SDFs
    for all 24 targets + SAFE-GPT decorated leads, builds the FDA/EMA 7-step Credibility Dossier,
    and writes ``results/tables/v4_tri_disorder_summary.md``.

Usage::

    HF_HOME=.hf_cache.nosync PYTORCH_ENABLE_MPS_FALLBACK=1 \\
        .venv/bin/python scripts/run_v4_tri_disorder_benchmark.py --stage all
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy import stats
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from geneuclidean.cns.admet import (
    ADMET_FP_BITS,
    CNSAdmetSuite,
    MultiplicativeCNSReward,
    compute_ece,
    compute_multiplicative_cns_reward,
    featurize_admet_smiles,
    make_scaffold_kfold,
)
from geneuclidean.cns.credibility import build_credibility_dossier
from geneuclidean.cns.medchem_triage import bemis_murcko_scaffold
from geneuclidean.cns.mpo import REFERENCE_CNS_DRUGS, REFERENCE_NON_CNS_DRUGS
from geneuclidean.cns.targets import (
    ALL_CNS_TARGET_REGISTRY,
    TRI_DISORDER_TARGET_REGISTRY,
    V4_COUNTER_TARGET_REGISTRY,
    find_local_pocket_files,
    get_cns_target,
    load_cns_target_pocket,
)
from geneuclidean.config import ExperimentConfig, seed_everything, select_device
from geneuclidean.data.featurizer import LIGAND_FP_DIM
from geneuclidean.data.splits import compute_murcko_scaffold
from geneuclidean.evaluation.docking_pipeline import (
    compute_selectivity_ddg,
    dock_ligand_to_pocket,
    redock_native_complex,
)
from geneuclidean.evaluation.geometry3d import prepare_receptor_pdbqt
from geneuclidean.evaluation.metrics import (
    bootstrap_ci,
    canonicalize_smiles,
    cliffs_delta,
    compute_internal_diversity,
    compute_snn,
    evaluate_affinity_metrics,
    evaluate_clip_retrieval,
    get_morgan_bit_fp,
    get_murcko_scaffold,
    is_pains_free,
    parse_mol,
    tanimoto_similarity,
)
from geneuclidean.models.encoders.rbf import PocketEncoderOutput
from geneuclidean.models.hf_backbones import (
    MOLFORMER_REPO_ID,
    NOVOMOLGEN_157M_ALIAS,
    SAFE_GPT_ALIAS,
    load_hf_backbone,
)
from geneuclidean.models.llm_bridge import PocketConditionedLLM
from geneuclidean.models.safe_codec import is_valid_safe, safe_to_smiles
from geneuclidean.sampling.llm_sampler import sample_llm_molecules
from geneuclidean.sampling.rerank import maxmin_diversity_select
from geneuclidean.training.llm_trainer import (
    CachedPocketRecord,
    CrossPocketPreferencePair,
    GDPOGroup,
    PocketLLMTrainer,
    build_pocket_batches_from_records,
    precompute_pocket_cache,
    run_llm_gdpo,
    run_llm_rpo_dpo,
)
from geneuclidean.training.pretrain import build_cosine_warmup_scheduler

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

_DESCRIPTORS: Any = Descriptors
_MORGAN_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=LIGAND_FP_DIM)

# ---------------------------------------------------------------------------
# Target Cohorts, Conformations, Contrasts & Canonical Paths
# ---------------------------------------------------------------------------

V31_12_TARGETS: tuple[str, ...] = (
    "DRD1",
    "VMAT2",
    "ADRA2A",
    "DRD2",
    "HRH3",
    "PDE10A",
    "CHRM4",
    "HTR2A",
    "SERT",
    "DRD3",
    "TAAR1",
    "DAT",
)
NEW_TS_SCZ_6_TARGETS: tuple[str, ...] = (
    "CHRM1",
    "HTR1A",
    "GLYT1",
    "MGLL",
    "PDE1B",
    "GPR52",
)
NEW_AD_6_TARGETS: tuple[str, ...] = (
    "ACHE",
    "OGA",
    "NLRP3",
    "MAOB",
    "GSK3B",
    "BACE1",
)
ALL_24_TARGETS: tuple[str, ...] = (*V31_12_TARGETS, *NEW_TS_SCZ_6_TARGETS, *NEW_AD_6_TARGETS)
NEW_12_TARGETS: tuple[str, ...] = (*NEW_TS_SCZ_6_TARGETS, *NEW_AD_6_TARGETS)
AD_STRUCTURAL_COUNTERS: tuple[str, ...] = ("BCHE", "MAOA", "CDK2", "BACE2")
GATE1_EXPANSION_16: tuple[str, ...] = (*NEW_12_TARGETS, *AD_STRUCTURAL_COUNTERS)
COUNTER_QSAR_TARGETS: tuple[str, ...] = ("MAOA", "BACE2")
LOW_N_TARGETS: frozenset[str] = frozenset({"VMAT2", "TAAR1", "GPR52", "NLRP3", "OGA"})

# Same-functional-state verified *Homo sapiens* PDB structures available on disk
SAME_STATE_PDBS_24: dict[str, tuple[str, ...]] = {
    "DRD1": ("9LLG", "9URL"),
    "DRD2": ("6CM4", "6LUQ"),
    "DRD3": ("3PBL", "9T8C"),
    "VMAT2": ("8T69", "9KQ8"),
    "ADRA2A": ("7EJ8", "9CBL", "7EJ0"),
    "HRH3": ("7F61",),
    "PDE10A": ("5SJX", "3WYM", "5SFE"),
    "CHRM4": ("8FX5", "7TRK"),
    "HTR2A": ("7WC8", "9WZO", "6A93"),
    "SERT": ("5I6X", "5I71"),
    "TAAR1": ("8W88",),
    "DAT": ("8Y2G", "9JKI"),
    "CHRM1": ("9UCP",),
    "HTR1A": ("9VJE",),
    "GLYT1": ("9J8D",),
    "MGLL": ("9F8D",),
    "PDE1B": ("5UOY",),
    "GPR52": ("8HMP",),
    "ACHE": ("4EY7",),
    "OGA": ("9BA8",),
    "NLRP3": ("9GU4", "9HG4"),
    "MAOB": ("2V5Z",),
    "GSK3B": ("7SXJ",),
    "BACE1": ("6EQM",),
}

# Anti-target 3D pocket contrast map for L_XP-DPO (target -> counter pocket key)
OFF_TARGET_POCKET_MAP: dict[str, str] = {
    "DRD1": "DRD2",
    "DRD2": "DRD1",
    "DRD3": "VMAT2",
    "VMAT2": "DRD2",
    "ADRA2A": "DRD2",
    "HRH3": "DRD2",
    "PDE10A": "VMAT2",
    "CHRM4": "DRD2",
    "HTR2A": "DRD2",
    "SERT": "DRD2",
    "TAAR1": "DRD2",
    "DAT": "DRD2",
    "CHRM1": "DRD2",
    "HTR1A": "DRD2",
    "GLYT1": "DRD2",
    "MGLL": "DRD2",
    "PDE1B": "VMAT2",
    "GPR52": "DRD2",
    "ACHE": "BCHE",
    "OGA": "BACE2",
    "NLRP3": "CDK2",
    "MAOB": "MAOA",
    "GSK3B": "CDK2",
    "BACE1": "BACE2",
}

# Off-target QSAR contrast lists for Multiplicative CNS Reward (eliminating VMAT2/DRD2 sinks)
OFF_TARGET_QSAR_MAP: dict[str, tuple[str, ...]] = {
    "DRD1": ("DRD2", "DRD3", "VMAT2"),
    "DRD2": ("DRD1", "VMAT2"),
    "DRD3": ("VMAT2", "DRD2"),
    "VMAT2": ("DRD2", "PDE10A"),
    "ADRA2A": ("DRD2", "VMAT2"),
    "HRH3": ("DRD2", "VMAT2"),
    "PDE10A": ("VMAT2", "DRD2", "PDE1B"),
    "CHRM4": ("DRD2", "VMAT2"),
    "HTR2A": ("DRD2", "VMAT2"),
    "SERT": ("DRD2", "VMAT2"),
    "TAAR1": ("DRD2", "VMAT2"),
    "DAT": ("DRD2", "VMAT2"),
    "CHRM1": ("DRD2", "VMAT2"),
    "HTR1A": ("DRD2", "VMAT2"),
    "GLYT1": ("DRD2", "VMAT2"),
    "MGLL": ("DRD2", "VMAT2"),
    "PDE1B": ("VMAT2", "PDE10A", "DRD2"),
    "GPR52": ("DRD2", "VMAT2"),
    "ACHE": ("DRD2", "VMAT2"),
    "OGA": ("DRD2", "VMAT2", "BACE2"),
    "NLRP3": ("DRD2", "VMAT2"),
    "MAOB": ("MAOA", "DRD2", "VMAT2"),
    "GSK3B": ("DRD2", "VMAT2"),
    "BACE1": ("BACE2", "DRD2", "VMAT2"),
}

# Cross-class bijection derangement over all 24 targets for Shuffled-Pocket 3D specificity control
SHUFFLED_DERANGEMENT_24: dict[str, str] = {
    "DRD1": "PDE10A",
    "PDE10A": "DRD1",
    "DRD2": "VMAT2",
    "VMAT2": "DRD2",
    "DRD3": "SERT",
    "SERT": "DRD3",
    "HTR2A": "DAT",
    "DAT": "HTR2A",
    "HRH3": "CHRM4",
    "CHRM4": "HRH3",
    "ADRA2A": "TAAR1",
    "TAAR1": "ADRA2A",
    "CHRM1": "ACHE",
    "ACHE": "CHRM1",
    "HTR1A": "MAOB",
    "MAOB": "HTR1A",
    "GLYT1": "GSK3B",
    "GSK3B": "GLYT1",
    "MGLL": "BACE1",
    "BACE1": "MGLL",
    "PDE1B": "NLRP3",
    "NLRP3": "PDE1B",
    "GPR52": "OGA",
    "OGA": "GPR52",
}

# Canonical 3D anchor specification for every pocket (Gate 1 + Stage dock)
POCKET_ANCHOR_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "DRD1": (("ASP103",), ("salt_bridge", "hbond")),
    "DRD2": (("ASP114",), ("salt_bridge", "hbond")),
    "DRD2_6LUQ": (("ASP114",), ("salt_bridge", "hbond")),
    "DRD3": (("ASP110",), ("salt_bridge", "hbond")),
    "VMAT2": (("GLU312", "ASP399"), ("salt_bridge", "hbond")),
    "ADRA2A": (("ASP113", "ASP128"), ("salt_bridge", "hbond")),
    "HRH3": (("ASP114", "GLU206"), ("salt_bridge", "hbond")),
    "PDE10A": (("GLN726", "TYR693"), ("hbond",)),
    "CHRM4": (("ASP112",), ("salt_bridge", "hbond")),
    "HTR2A": (("ASP155",), ("salt_bridge", "hbond")),
    "SERT": (("ASP98",), ("salt_bridge", "hbond")),
    "TAAR1": (("ASP103",), ("salt_bridge", "hbond")),
    "DAT": (("ASP79",), ("salt_bridge", "hbond")),
    # 12 New TS/SCZ/AD targets
    "CHRM1": (("ASP105", "TYR364", "TRP338"), ("salt_bridge", "hbond", "cation_pi", "pi_stacking")),
    "HTR1A": (("ASP116", "TYR390"), ("salt_bridge", "hbond")),
    "GLYT1": (("SER317", "TYR62", "CYS421"), ("hbond", "pi_stacking", "halogen_bond")),
    "MGLL": (("SER122", "ALA51", "MET123"), ("hbond",)),
    "PDE1B": (("GLN421", "HIS373"), ("hbond", "pi_stacking")),
    "GPR52": (("CYS40", "ASP38", "GLU191"), ("hbond",)),
    "ACHE": (("PHE295", "TRP86", "TRP286", "TYR337"), ("hbond", "pi_stacking", "cation_pi")),
    "OGA": (("ASP285", "ASP174", "ASN280"), ("salt_bridge", "hbond")),
    "NLRP3": (("ARG578", "GLU629", "ALA228"), ("salt_bridge", "hbond")),
    "MAOB": (("GLN206", "TYR435", "TYR398"), ("hbond", "pi_stacking")),
    "GSK3B": (("VAL135", "ASP133", "LYS85"), ("hbond",)),
    "BACE1": (("ASP32", "ASP228", "GLY230"), ("salt_bridge", "hbond")),
    # 4 Structural AD Counter-Targets
    "BCHE": (("TRP82", "HIS438", "GLU197"), ("hbond", "pi_stacking", "cation_pi")),
    "MAOA": (("PHE208", "TYR407", "TYR444", "GLN215"), ("hbond", "pi_stacking")),
    "CDK2": (("LEU83", "GLU81", "LYS33"), ("hbond",)),
    "BACE2": (("ASP48", "ASP241", "GLY243"), ("salt_bridge", "hbond")),
}

# SAFE-GPT 3D-Conditioned Scaffold Decoration Clinical Cores
SCAFFOLD_DECORATION_TASKS: tuple[dict[str, str], ...] = (
    {
        "slug": "ecopipam_drd1",
        "drug_name": "Ecopipam",
        "target_id": "DRD1",
        "pdb_id": "9LLG",
        "counter_target": "DRD2",
        "parent_smiles": "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",
        "core_with_dummy": "c1ccc2c(c1)CC[C@@H]1[C@H]2c2cc(O)c(Cl)cc2CCN1[*]",
        "objective": "Optimize N-substituent on Ecopipam benzazepine core inside DRD1 (9LLG)",
    },
    {
        "slug": "donepezil_ache",
        "drug_name": "Donepezil",
        "target_id": "ACHE",
        "pdb_id": "4EY7",
        "counter_target": "BCHE",
        "parent_smiles": "COc1cc2c(cc1OC)C(=O)[C@H](CC1CCN(Cc3ccccc3)CC1)C2",
        "core_with_dummy": "COc1cc2c(cc1OC)C(=O)C(CC1CCN([*])CC1)C2",
        "objective": (
            "Rescue Donepezil hERG=0.95 cardiotoxicity by replacing N-benzyl in ACHE (4EY7)"
        ),
    },
    {
        "slug": "safinamide_maob",
        "drug_name": "Safinamide",
        "target_id": "MAOB",
        "pdb_id": "2V5Z",
        "counter_target": "MAOA",
        "parent_smiles": "C[C@H](NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O",
        "core_with_dummy": "Fc1cccc(COc2ccc([*])cc2)c1",
        "objective": "Optimize polar headgroup on Safinamide benzyloxy tail inside MAOB (2V5Z+FAD)",
    },
    {
        "slug": "xanomeline_chrm4",
        "drug_name": "Xanomeline",
        "target_id": "CHRM4",
        "pdb_id": "8FX5",
        "counter_target": "DRD2",
        "parent_smiles": "CCCCCCOc1nsnc1C1=CCCN(C)C1",
        "core_with_dummy": "CN1CCC=C(c2nsnc2[*])C1",
        "objective": (
            "Lower Xanomeline peripheral M3=0.97 risk by replacing hexyloxy tail in CHRM4 (8FX5)"
        ),
    },
)

# Canonical Paths
CHEMBL_V31_CSV = Path("data/cns_neuro.nosync/chembl/human_cns_actives_12targets.csv")
CHEMBL_V4_CSV = Path("data/cns_neuro.nosync/v4_expansion/chembl_v4_12new_targets.csv")
CLEAN_POCKETS_V31_DIR = Path("data/cns_neuro.nosync/structures/clean_pockets")
CLEAN_POCKETS_V4_DIR = Path("data/cns_neuro.nosync/v4_expansion/structures/clean_pockets")
V4_ADMET_MODEL_PATH = Path("data/cns_neuro.nosync/v4_admet_avoidome_oracles.joblib")
V4_ADMET_CV_PATH = Path("results/tables/v4_admet_avoidome_cv_metrics.json")
V31_QSAR_MODEL_PATH = Path("data/cns_neuro.nosync/selectivity_qsar.joblib")
V4_QSAR_MODEL_PATH = Path("data/cns_neuro.nosync/v4_24target_qsar.joblib")
ZINC_SMILES_PATH = Path("data/processed/zinc250k_smiles.txt")
PDBBIND_CACHE_PATH = Path("data/processed/pdbbind_pockets.pt")
PDBBIND_SPLITS_PATH = Path("data/processed/splits.json")
V3_CONFIG_PATH = Path("configs/geneuclidean_v3_gpmolformer.yaml")
V2_PAINN_CKPT = Path("runs_local.nosync/geneuclidean_v2_painn/best.pt")
V3_SFT_ADAPTER_CKPT = Path("runs_local.nosync/geneuclidean_v3_gpmolformer/best_adapter.pt")
V31_DPO_ADAPTER_CKPT = Path("runs_local.nosync/geneuclidean_v31_cns_dpo/best_adapter.pt")
VINA_BIN = Path("bin/vina")

RUN_ROOT = Path("runs_local.nosync/geneuclidean_v4_tri_disorder")
FULL_SPLITS_LOCAL_PATH = RUN_ROOT / "v4_tri_disorder_splits_full.json"
DOCKED_SDF_DIR = RUN_ROOT / "docked_poses_3d"
RECEPTOR_PDBQT_DIR = RUN_ROOT / "receptor_pdbqts"
VINA_CACHE_PATH = RUN_ROOT / "v4_vina_cache.jsonl"

TABLE_SPLITS_PATH = Path("results/tables/v4_tri_disorder_splits.json")
TABLE_QSAR_CV_PATH = Path("results/tables/v4_tri_disorder_qsar_cv.json")
TABLE_GATE1_PATH = Path("results/tables/v4_gate1_expansion_redocking.json")
TABLE_HEAD_TO_HEAD_PATH = Path("results/tables/v4_llm_head_to_head.json")
TABLE_DOCKING_PATH = Path("results/tables/v4_tri_disorder_docking.json")
TABLE_CREDIBILITY_PATH = Path("results/tables/v4_credibility_dossier.json")
TABLE_SUMMARY_MD_PATH = Path("results/tables/v4_tri_disorder_summary.md")

# Standard Decoding Hyperparameters
EVAL_NUM_SAMPLES: int = 64
EVAL_TEMPERATURE: float = 0.85
EVAL_MIN_P: float = 0.08
EVAL_CFG_SCALE: float = 1.3
EVAL_REPETITION_PENALTY: float = 1.0
EVAL_MAX_NEW_TOKENS: int = 120


# ---------------------------------------------------------------------------
# Generic Helpers
# ---------------------------------------------------------------------------


def _get_joblib() -> Any:
    return importlib.import_module("joblib")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _clear_accel(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clean_isomeric_smiles(smi: str) -> str | None:
    return canonicalize_smiles(smi, isomeric=True, keep_largest_fragment=True)


def _decode_generated_molecule(raw_text: str, *, format_mode: str) -> str | None:
    """Strictly validate and decode a generated SMILES or SAFE string (single connected mol)."""
    if not isinstance(raw_text, str):
        return None
    cleaned = "".join(raw_text.split())
    if not cleaned:
        return None
    if format_mode == "safe":
        if not is_valid_safe(cleaned, allow_disconnected=False):
            return None
        return safe_to_smiles(cleaned, canonical=True, largest_fragment=False)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _load_v4_admet_suite(n_jobs: int = 4) -> CNSAdmetSuite:
    suite = CNSAdmetSuite.load(V4_ADMET_MODEL_PATH)
    for oracle in suite.oracles.values():
        rf = getattr(oracle, "_rf", None)
        if rf is not None and hasattr(rf, "n_jobs"):
            rf.n_jobs = int(n_jobs)
    return suite


def _load_zinc250k_smiles(exclude_set: set[str] | None = None) -> list[str]:
    if not ZINC_SMILES_PATH.is_file():
        raise FileNotFoundError(f"ZINC-250k file missing: {ZINC_SMILES_PATH}")
    out: list[str] = []
    with ZINC_SMILES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            if exclude_set is not None and s in exclude_set:
                continue
            out.append(s)
    return out


def _build_strict_novelty_reference_sets() -> tuple[set[str], set[str]]:
    """Build exact canonical SMILES unions across ALL 66,869 ChEMBL + PDBbind + ZINC-250k."""
    chembl_set: set[str] = set()
    for csv_path in (CHEMBL_V31_CSV, CHEMBL_V4_CSV):
        df = pd.read_csv(csv_path)
        for raw_smi in df["canonical_smiles"].astype(str).tolist():
            can = _clean_isomeric_smiles(raw_smi)
            if can is not None:
                chembl_set.add(can)

    pdbbind_set: set[str] = set()
    if PDBBIND_CACHE_PATH.is_file():
        payload = torch.load(PDBBIND_CACHE_PATH, map_location="cpu", weights_only=False)
        for rec in payload.get("records", {}).values():
            smi = rec.get("canonical_smiles")
            if smi:
                can = _clean_isomeric_smiles(str(smi))
                if can is not None:
                    pdbbind_set.add(can)

    chembl_pdbbind = chembl_set | pdbbind_set
    zinc_set: set[str] = set()
    if ZINC_SMILES_PATH.is_file():
        with ZINC_SMILES_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if s:
                    zinc_set.add(s)
    return (chembl_pdbbind | zinc_set), chembl_pdbbind


def _smiles_to_fp_tensor(smiles_list: Sequence[str], fp_dim: int = LIGAND_FP_DIM) -> torch.Tensor:
    arr = np.zeros((len(smiles_list), fp_dim), dtype=np.float32)
    for idx, smi in enumerate(smiles_list):
        mol = parse_mol(smi)
        if mol is None:
            continue
        arr[idx] = np.asarray(_MORGAN_1024.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return torch.from_numpy(arr)


def _snapshot_param_checksums(
    model: PocketConditionedLLM,
    prefixes: tuple[str, ...],
    *,
    include_base_llm: bool = False,
) -> dict[str, torch.Tensor]:
    snaps: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.startswith(prefixes) or (
            include_base_llm
            and name.startswith("llm.")
            and ("lora_" not in name.lower() and "modules_to_save" not in name.lower())
        ):
            snaps[name] = param.detach().cpu().clone()
    return snaps


def _verify_frozen_bit_exact(
    model: PocketConditionedLLM, snapshots: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    max_diff = 0.0
    n_checked = 0
    for name, param in model.named_parameters():
        if name in snapshots:
            cur = param.detach().cpu()
            ref = snapshots[name]
            diff = float((cur.float() - ref.float()).abs().max().item())
            max_diff = max(max_diff, diff)
            n_checked += 1
            if not torch.equal(cur, ref):
                raise AssertionError(f"Frozen parameter drifted bit-wise: {name} (diff={diff})")
    return {"n_tensors_verified": n_checked, "max_abs_diff": max_diff, "bit_exact": True}


def _holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: 1.0 if not math.isfinite(kv[1]) else kv[1])
    adj = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        eff_p = 1.0 if not math.isfinite(p_val) else float(p_val)
        val = min(1.0, (m - rank) * eff_p)
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj


# ---------------------------------------------------------------------------
# Stage 1: prepare — Zero-Leakage 24-Target Splits + 26-Endpoint Calibrated QSAR
# ---------------------------------------------------------------------------


def _fit_platt(raw_probs: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    clipped = np.clip(raw_probs.astype(np.float64), 1e-5, 1.0 - 1e-5)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return None
    lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=200)
    lr.fit(logit(clipped).reshape(-1, 1), y)
    return lr


def _apply_platt(cal: LogisticRegression | None, raw_probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_probs.astype(np.float64), 1e-5, 1.0 - 1e-5)
    if cal is None:
        return clipped
    return np.asarray(cal.predict_proba(logit(clipped).reshape(-1, 1))[:, 1], dtype=np.float64)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    yp = np.clip(y_prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    has_both = len(np.unique(y_true)) >= 2
    return {
        "roc_auc": float(roc_auc_score(y_true, yp)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, yp)) if has_both else float("nan"),
        "brier_score": float(brier_score_loss(y_true, yp)),
        "ece": compute_ece(y_true, yp, n_bins=10),
    }


def _train_single_qsar_endpoint(
    target_id: str,
    smiles_list: list[str],
    labels: list[int],
    *,
    n_splits: int = 5,
    n_jobs: int = 4,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_all, fp_bits_all, valid_mask, canon_list = featurize_admet_smiles(smiles_list)
    y_raw = np.asarray(labels, dtype=np.int64)
    keep = valid_mask & np.isin(y_raw, [0, 1])

    x_mat = x_all[keep]
    fp_bits = fp_bits_all[keep]
    y_arr = y_raw[keep]
    valid_smiles = [s for i in np.flatnonzero(keep).tolist() if (s := canon_list[i]) is not None]
    n_samples = len(valid_smiles)

    folds = make_scaffold_kfold(valid_smiles, n_splits=n_splits, seed=seed)
    scaffolds = [compute_murcko_scaffold(s) for s in valid_smiles]
    leakage = 0

    oof_raw = np.zeros(n_samples, dtype=np.float64)
    oof_cal = np.zeros(n_samples, dtype=np.float64)
    fold_reports: list[dict[str, float]] = []

    def _make_pair(s_off: int) -> tuple[HistGradientBoostingClassifier, RandomForestClassifier]:
        hgb = HistGradientBoostingClassifier(
            max_iter=140,
            max_depth=6,
            learning_rate=0.06,
            min_samples_leaf=12,
            l2_regularization=1.0,
            max_bins=128,
            random_state=seed + s_off,
        )
        rf = RandomForestClassifier(
            n_estimators=160,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=seed + s_off,
        )
        return hgb, rf

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr_scafs = {scaffolds[i] for i in tr_idx.tolist()}
        va_scafs = {scaffolds[i] for i in va_idx.tolist()}
        leakage += len(tr_scafs & va_scafs)

        tr_smis = [valid_smiles[i] for i in tr_idx.tolist()]
        in_folds = make_scaffold_kfold(tr_smis, n_splits=5, seed=seed + 100 + fold_idx)
        in_fit_rel, in_cal_rel = in_folds[0]
        fit_idx = tr_idx[in_fit_rel]
        cal_idx = tr_idx[in_cal_rel]
        if len(np.unique(y_arr[fit_idx])) < 2 or len(np.unique(y_arr[cal_idx])) < 2:
            fit_idx = tr_idx
            cal_idx = tr_idx

        hgb_f, rf_f = _make_pair(fold_idx + 1)
        hgb_f.fit(x_mat[fit_idx], y_arr[fit_idx])
        rf_f.fit(x_mat[fit_idx], y_arr[fit_idx])

        cal_raw = 0.5 * (
            hgb_f.predict_proba(x_mat[cal_idx])[:, 1] + rf_f.predict_proba(x_mat[cal_idx])[:, 1]
        )
        platt_f = _fit_platt(cal_raw, y_arr[cal_idx])

        va_raw = 0.5 * (
            hgb_f.predict_proba(x_mat[va_idx])[:, 1] + rf_f.predict_proba(x_mat[va_idx])[:, 1]
        )
        va_cal = _apply_platt(platt_f, va_raw)
        oof_raw[va_idx] = va_raw
        oof_cal[va_idx] = va_cal
        fold_reports.append(_binary_metrics(y_arr[va_idx], va_cal))

    if leakage != 0:
        raise AssertionError(f"Scaffold leakage detected in QSAR {target_id}: {leakage}")

    hgb_final, rf_final = _make_pair(0)
    hgb_final.fit(x_mat, y_arr)
    rf_final.fit(x_mat, y_arr)
    platt_final = _fit_platt(oof_raw, y_arr)

    pos_fp_packed = np.packbits(fp_bits[y_arr == 1], axis=1)
    all_fp_packed = np.packbits(fp_bits, axis=1)
    overall_cal = _binary_metrics(y_arr, oof_cal)
    overall_raw = _binary_metrics(y_arr, oof_raw)
    fold_aucs = [f["roc_auc"] for f in fold_reports if math.isfinite(f["roc_auc"])]

    cv_metrics: dict[str, Any] = {
        "target_id": target_id,
        "split_protocol": f"{n_splits}-fold Bemis-Murcko scaffold CV (zero scaffold overlap)",
        "n_samples": int(n_samples),
        "n_positive": int(y_arr.sum()),
        "n_negative": int((y_arr == 0).sum()),
        "n_unique_scaffolds": int(len(set(scaffolds))),
        "scaffold_leakage_groups": int(leakage),
        "oof_roc_auc": overall_cal["roc_auc"],
        "oof_pr_auc": overall_cal["pr_auc"],
        "oof_brier_score": overall_cal["brier_score"],
        "oof_ece": overall_cal["ece"],
        "oof_uncalibrated_ece": overall_raw["ece"],
        "fold_roc_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "fold_roc_auc_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        "ad_threshold": 0.30,
    }
    bundle: dict[str, Any] = {
        "target_id": target_id,
        "hgb": hgb_final,
        "rf": rf_final,
        "platt": platt_final,
        "pos_fp_packed": pos_fp_packed,
        "all_fp_packed": all_fp_packed,
        "ad_threshold": 0.30,
        "cv_metrics": cv_metrics,
    }
    return bundle, cv_metrics


def _predict_qsar_endpoint(
    bundle: Mapping[str, Any],
    x_mat: np.ndarray,
    fp_bits: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_mat)
    probs = np.full(n, float("nan"), dtype=np.float64)
    ad_sim = np.zeros(n, dtype=np.float64)
    if n == 0 or not bool(valid_mask.any()):
        return probs, ad_sim

    valid_idx = np.flatnonzero(valid_mask)
    x_val = x_mat[valid_idx]
    hgb = bundle["hgb"]
    rf = bundle["rf"]
    if hasattr(rf, "n_jobs"):
        rf.n_jobs = 4
    raw = 0.5 * (hgb.predict_proba(x_val)[:, 1] + rf.predict_proba(x_val)[:, 1])
    probs[valid_idx] = np.clip(_apply_platt(bundle["platt"], raw), 0.0, 1.0)

    tid = str(bundle["target_id"])
    if cache is not None and tid in cache:
        ref_mat, ref_norms = cache[tid]
    else:
        packed = bundle["all_fp_packed"]
        unpacked = np.unpackbits(packed, axis=1)[:, :ADMET_FP_BITS].astype(np.float32)
        ref_mat = unpacked
        ref_norms = unpacked.sum(axis=1).astype(np.float32)
        if cache is not None:
            cache[tid] = (ref_mat, ref_norms)

    q_mat = fp_bits[valid_idx].astype(np.float32)
    q_norms = q_mat.sum(axis=1).astype(np.float32)
    out_ad = np.zeros(len(valid_idx), dtype=np.float64)
    chunk = 256
    for s in range(0, len(valid_idx), chunk):
        e = min(s + chunk, len(valid_idx))
        inter = q_mat[s:e] @ ref_mat.T
        union = q_norms[s:e, None] + ref_norms[None, :] - inter
        tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
        out_ad[s:e] = np.max(tan, axis=1)
    ad_sim[valid_idx] = np.clip(out_ad, 0.0, 1.0)
    return probs, ad_sim


def score_target_and_offtarget_qsar(
    qsar_payload: Mapping[str, Any],
    smiles_list: Sequence[str],
    target_id: str,
    *,
    fp_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Score on-target P(target), off-target max P(off), P(VMAT2), P(DRD2), and AD."""
    x_mat, fp_bits, valid_mask, _ = featurize_admet_smiles(smiles_list)
    models = qsar_payload["models"]
    p_on, ad_on = _predict_qsar_endpoint(
        models[target_id], x_mat, fp_bits, valid_mask, cache=fp_cache
    )
    p_vmat2, _ = _predict_qsar_endpoint(models["VMAT2"], x_mat, fp_bits, valid_mask, cache=fp_cache)
    p_drd2, _ = _predict_qsar_endpoint(models["DRD2"], x_mat, fp_bits, valid_mask, cache=fp_cache)

    off_targets = [t for t in OFF_TARGET_QSAR_MAP.get(target_id, ("DRD2", "VMAT2")) if t in models]
    off_arrays: list[np.ndarray] = []
    for ot in off_targets:
        if ot == "VMAT2":
            off_arrays.append(p_vmat2)
        elif ot == "DRD2":
            off_arrays.append(p_drd2)
        else:
            p_ot, _ = _predict_qsar_endpoint(models[ot], x_mat, fp_bits, valid_mask, cache=fp_cache)
            off_arrays.append(p_ot)

    if off_arrays:
        p_off_max = np.nanmax(np.stack(off_arrays, axis=0), axis=0)
    else:
        p_off_max = np.zeros_like(p_on)

    primary_counter = off_targets[0] if off_targets else "DRD2"
    p_primary_off = off_arrays[0] if off_arrays else p_drd2
    return {
        "valid_mask": valid_mask,
        "p_on_target": p_on,
        "ad_on_target": ad_on,
        "p_off_max": p_off_max,
        "p_primary_off": p_primary_off,
        "primary_counter_name": np.asarray([primary_counter]),
        "p_vmat2": p_vmat2,
        "p_drd2": p_drd2,
        "delta_sel": p_on - p_primary_off,
    }


def _collect_protected_clinical_and_native_v4() -> dict[str, Any]:
    """Collect ALL clinical drugs across all 31 specs + ALL 46 native co-crystal ligands."""
    clinical_records: list[dict[str, str]] = []
    native_records: list[dict[str, str]] = []
    protected_smiles: set[str] = set()
    protected_exact_scafs: set[str] = set()
    protected_generic_scafs: set[str] = set()
    seen_clinical: set[str] = set()

    def _register(can: str) -> tuple[str, str]:
        ex = compute_murcko_scaffold(can)
        gen = bemis_murcko_scaffold(can, generic=True)
        protected_smiles.add(can)
        if ex:
            protected_exact_scafs.add(ex)
        if gen:
            protected_generic_scafs.add(gen)
        return ex, gen

    for tid, spec in ALL_CNS_TARGET_REGISTRY.items():
        for drug in spec.reference_drugs:
            can = _clean_isomeric_smiles(drug.canonical_smiles)
            if can is None:
                continue
            ex, gen = _register(can)
            if can not in seen_clinical:
                seen_clinical.add(can)
                clinical_records.append(
                    {
                        "name": drug.name,
                        "target_id": tid,
                        "canonical_smiles": can,
                        "exact_scaffold": ex,
                        "generic_scaffold": gen,
                        "role": drug.role,
                    }
                )

    for name, smi in {**REFERENCE_CNS_DRUGS, **REFERENCE_NON_CNS_DRUGS}.items():
        can = _clean_isomeric_smiles(smi)
        if can is None:
            continue
        ex, gen = _register(can)
        if name in REFERENCE_CNS_DRUGS and can not in seen_clinical:
            seen_clinical.add(can)
            clinical_records.append(
                {
                    "name": name,
                    "target_id": "CNS_REF",
                    "canonical_smiles": can,
                    "exact_scaffold": ex,
                    "generic_scaffold": gen,
                    "role": "Clinical CNS Reference",
                }
            )

    for clean_dir in (CLEAN_POCKETS_V31_DIR, CLEAN_POCKETS_V4_DIR):
        if not clean_dir.exists():
            continue
        for sdf_path in sorted(clean_dir.glob("*_lig.sdf")):
            suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
            mol = suppl[0] if len(suppl) > 0 else None
            if mol is None:
                raw_suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=True)
                mol = raw_suppl[0] if len(raw_suppl) > 0 else None
                if mol is not None:
                    for atom in mol.GetAtoms():
                        if (
                            atom.GetAtomicNum() == 7
                            and atom.GetFormalCharge() == 0
                            and atom.GetExplicitValence() == 4
                        ):
                            atom.SetFormalCharge(1)
                    with contextlib.suppress(Exception):
                        Chem.SanitizeMol(mol)
            if mol is None:
                continue
            try:
                raw_smi = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            except Exception:
                continue
            can = _clean_isomeric_smiles(raw_smi)
            if can is None:
                continue
            ex, gen = _register(can)
            parts = sdf_path.stem.split("_")
            native_records.append(
                {
                    "file": sdf_path.name,
                    "pdb_id": parts[0] if parts else sdf_path.stem,
                    "target_id": parts[1] if len(parts) > 1 else "UNK",
                    "ligand_ccd": parts[2] if len(parts) > 2 else "UNK",
                    "canonical_smiles": can,
                    "exact_scaffold": ex,
                    "generic_scaffold": gen,
                }
            )

    return {
        "clinical_records": clinical_records,
        "native_records": native_records,
        "protected_smiles": sorted(protected_smiles),
        "protected_exact_scaffolds": sorted(protected_exact_scafs),
        "protected_generic_scaffolds": sorted(protected_generic_scafs),
    }


def _partition_generic_scaffolds_80_10_10(
    smiles_list: list[str], *, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """Strict 80/10/10 generic Bemis-Murcko scaffold split (0 exact & 0 generic overlap)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for smi in smiles_list:
        g_scaf = bemis_murcko_scaffold(smi, generic=True)
        key = g_scaf if g_scaf else f"ACYCLIC::{smi}"
        groups[key].append(smi)

    rng = np.random.default_rng(seed)
    sorted_scafs = sorted(groups.keys())
    ties = {s: float(rng.random()) for s in sorted_scafs}
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), ties[kv[0]]))

    n_total = len(smiles_list)
    val_target = max(1, int(round(0.10 * n_total)))
    train_s: list[str] = []
    val_s: list[str] = []
    test_s: list[str] = []

    if len(ordered) >= 5:
        train_cap = int(round(0.80 * n_total))
        for _scaf, members in ordered:
            if len(train_s) + len(members) <= train_cap or not train_s:
                train_s.extend(members)
            elif len(val_s) < val_target:
                val_s.extend(members)
            else:
                test_s.extend(members)
        if not test_s and len(val_s) >= 2:
            last_key = bemis_murcko_scaffold(val_s[-1], generic=True) or f"ACYCLIC::{val_s[-1]}"
            moved = [
                m
                for m in val_s
                if (bemis_murcko_scaffold(m, generic=True) or f"ACYCLIC::{m}") == last_key
            ]
            val_s = [
                m
                for m in val_s
                if (bemis_murcko_scaffold(m, generic=True) or f"ACYCLIC::{m}") != last_key
            ]
            test_s.extend(moved)
    else:
        for idx, (_scaf, members) in enumerate(ordered):
            if idx == len(ordered) - 1:
                test_s.extend(members)
            elif idx == len(ordered) - 2:
                val_s.extend(members)
            else:
                train_s.extend(members)
    return sorted(train_s), sorted(val_s), sorted(test_s)


def run_stage_prepare(*, seed: int = 42, n_jobs: int = 4) -> dict[str, Any]:
    """Execute Stage ``prepare``: 24-target zero-leakage splits + 26-endpoint QSAR."""
    t0 = time.perf_counter()
    print("[prepare] Collecting clinical & native exclusion sets across 31 targets...", flush=True)
    protected = _collect_protected_clinical_and_native_v4()
    prot_smiles = set(protected["protected_smiles"])
    prot_ex_scafs = set(protected["protected_exact_scaffolds"])
    prot_gen_scafs = set(protected["protected_generic_scaffolds"])
    print(
        f"[prepare] Protected: {len(protected['clinical_records'])} clinical drugs, "
        f"{len(protected['native_records'])} native ligands -> "
        f"{len(prot_smiles)} unique SMILES, {len(prot_ex_scafs)} exact / "
        f"{len(prot_gen_scafs)} generic Murcko scaffolds.",
        flush=True,
    )

    df_v31 = pd.read_csv(CHEMBL_V31_CSV)
    df_v4 = pd.read_csv(CHEMBL_V4_CSV)
    df_all = pd.concat(
        [
            df_v31[["target_symbol", "canonical_smiles", "best_pchembl"]],
            df_v4[["target_symbol", "canonical_smiles", "best_pchembl"]],
        ],
        ignore_index=True,
    )

    # Full potency maps (down to pChEMBL >= 6.0) for counter-target dual exclusion
    counter_potency_maps: dict[str, dict[str, float]] = {
        "DRD2": {},
        "MAOA": {},
        "BACE2": {},
    }
    for ctid in counter_potency_maps:
        sub_c = df_all[df_all["target_symbol"] == ctid]
        for smi_raw, best_p in zip(
            sub_c["canonical_smiles"].astype(str).tolist(),
            sub_c["best_pchembl"].astype(float).tolist(),
            strict=True,
        ):
            can = _clean_isomeric_smiles(smi_raw)
            if can is not None and float(best_p) > counter_potency_maps[ctid].get(can, -1.0):
                counter_potency_maps[ctid][can] = float(best_p)

    all_qsar_targets = (*ALL_24_TARGETS, *COUNTER_QSAR_TARGETS)
    curated_by_target: dict[str, dict[str, float]] = {}
    filter_stats: dict[str, dict[str, Any]] = {}

    for tid in all_qsar_targets:
        sub = df_all[df_all["target_symbol"] == tid]
        p_thresh = 6.0 if tid in LOW_N_TARGETS else 6.5
        st: dict[str, Any] = {
            "n_raw_rows": int(len(sub)),
            "pchembl_threshold": p_thresh,
            "n_failed_pchembl": 0,
            "n_invalid_rdkit": 0,
            "n_failed_mw": 0,
            "n_failed_pains": 0,
            "n_clean_unique": 0,
        }
        curated: dict[str, float] = {}
        for smi_raw, best_p in zip(
            sub["canonical_smiles"].astype(str).tolist(),
            sub["best_pchembl"].astype(float).tolist(),
            strict=True,
        ):
            can = _clean_isomeric_smiles(smi_raw)
            if can is None:
                st["n_invalid_rdkit"] += 1
                continue
            if float(best_p) < p_thresh:
                st["n_failed_pchembl"] += 1
                continue
            mol = Chem.MolFromSmiles(can)
            if mol is None:
                st["n_invalid_rdkit"] += 1
                continue
            mw = float(_DESCRIPTORS.MolWt(mol))
            if mw < 150.0 or mw > 580.0:
                st["n_failed_mw"] += 1
                continue
            if not is_pains_free(mol):
                st["n_failed_pains"] += 1
                continue
            if float(best_p) > curated.get(can, -1.0):
                curated[can] = float(best_p)
        st["n_clean_unique"] = len(curated)
        curated_by_target[tid] = curated
        filter_stats[tid] = st

    # Build strict zero-leakage generic Bemis-Murcko 80/10/10 splits for all 24 primary targets
    dual_exclusion_rules: dict[str, str] = {
        "DRD1": "DRD2",
        "MAOB": "MAOA",
        "BACE1": "BACE2",
    }
    splits_full: dict[str, dict[str, Any]] = {}
    splits_summary: dict[str, Any] = {
        "schema_version": "geneuclidean-v4.0-tri-disorder-splits",
        "seed": seed,
        "n_primary_targets": len(ALL_24_TARGETS),
        "cohorts": {
            "v31_core_12": list(V31_12_TARGETS),
            "new_ts_scz_6": list(NEW_TS_SCZ_6_TARGETS),
            "new_ad_6": list(NEW_AD_6_TARGETS),
        },
        "protected_clinical_count": len(protected["clinical_records"]),
        "protected_native_count": len(protected["native_records"]),
        "protected_unique_smiles": len(prot_smiles),
        "protected_exact_scaffolds": len(prot_ex_scafs),
        "protected_generic_scaffolds": len(prot_gen_scafs),
        "dual_selectivity_curation_notes": (
            "Excluded DRD2-dominant duals from DRD1 Train, MAOA-dominant duals from MAOB Train, "
            "and BACE2-dominant duals from BACE1 Train (p_counter > p_on - 1.0). "
            "CDK2 (6Q4G) and BCHE (4BDS) selectivity is enforced structurally via L_XP-DPO "
            "cross-pocket contrast and held-out AutoDock Vina delta-delta-G panels."
        ),
        "global_leakage_audit": {
            "total_train_val_generic_scaffold_overlap": 0,
            "total_train_test_generic_scaffold_overlap": 0,
            "total_val_test_generic_scaffold_overlap": 0,
            "total_train_clinical_native_smiles_overlap": 0,
            "total_train_clinical_native_exact_scaffold_overlap": 0,
        },
        "targets": {},
    }

    for t_idx, tid in enumerate(ALL_24_TARGETS):
        curated = curated_by_target[tid]
        eligible: list[str] = []
        excl_prot: list[str] = []
        excl_dual: list[str] = []

        for smi in sorted(curated.keys()):
            ex_scaf = compute_murcko_scaffold(smi)
            if smi in prot_smiles or (ex_scaf and ex_scaf in prot_ex_scafs):
                excl_prot.append(smi)
                continue
            eligible.append(smi)

        train_s, val_s, test_s = _partition_generic_scaffolds_80_10_10(eligible, seed=seed + t_idx)

        counter_id = dual_exclusion_rules.get(tid)
        if counter_id is not None:
            c_map = counter_potency_maps[counter_id]
            kept_train: list[str] = []
            for smi in train_s:
                on_p = curated[smi]
                off_p = c_map.get(smi)
                if off_p is not None and off_p > (on_p - 1.0):
                    excl_dual.append(smi)
                else:
                    kept_train.append(smi)
            train_s = kept_train

        tr_gen = {bemis_murcko_scaffold(s, generic=True) or f"AC::{s}" for s in train_s}
        va_gen = {bemis_murcko_scaffold(s, generic=True) or f"AC::{s}" for s in val_s}
        te_gen = {bemis_murcko_scaffold(s, generic=True) or f"AC::{s}" for s in test_s}
        tr_ex = {compute_murcko_scaffold(s) for s in train_s if compute_murcko_scaffold(s)}

        ov_tr_va = len(tr_gen & va_gen)
        ov_tr_te = len(tr_gen & te_gen)
        ov_va_te = len(va_gen & te_gen)
        ov_prot_smi = len(set(train_s) & prot_smiles)
        ov_prot_scaf = len(tr_ex & prot_ex_scafs)

        assert ov_tr_va == 0, f"{tid} train/val generic scaffold leak: {ov_tr_va}"
        assert ov_tr_te == 0, f"{tid} train/test generic scaffold leak: {ov_tr_te}"
        assert ov_va_te == 0, f"{tid} val/test generic scaffold leak: {ov_va_te}"
        assert ov_prot_smi == 0, f"{tid} train clinical/native SMILES leak: {ov_prot_smi}"
        assert ov_prot_scaf == 0, f"{tid} train clinical/native scaffold leak: {ov_prot_scaf}"

        audit = splits_summary["global_leakage_audit"]
        audit["total_train_val_generic_scaffold_overlap"] += ov_tr_va
        audit["total_train_test_generic_scaffold_overlap"] += ov_tr_te
        audit["total_val_test_generic_scaffold_overlap"] += ov_va_te
        audit["total_train_clinical_native_smiles_overlap"] += ov_prot_smi
        audit["total_train_clinical_native_exact_scaffold_overlap"] += ov_prot_scaf

        cohort = (
            "v31_core_12"
            if tid in V31_12_TARGETS
            else ("new_ts_scz_6" if tid in NEW_TS_SCZ_6_TARGETS else "new_ad_6")
        )
        spec = TRI_DISORDER_TARGET_REGISTRY[tid]
        splits_summary["targets"][tid] = {
            **filter_stats[tid],
            "cohort": cohort,
            "disease_area": spec.disease_area,
            "primary_pdb_id": spec.primary_pdb_id,
            "train_conformations": list(SAME_STATE_PDBS_24[tid]),
            "n_excluded_clinical_or_native_scaffold": len(excl_prot),
            "n_excluded_counter_duals": len(excl_dual),
            "dual_counter_target": counter_id,
            "n_train": len(train_s),
            "n_val": len(val_s),
            "n_test": len(test_s),
            "n_train_generic_scaffolds": len(tr_gen),
            "n_val_generic_scaffolds": len(va_gen),
            "n_test_generic_scaffolds": len(te_gen),
            "leakage_zero_verified": True,
        }
        splits_full[tid] = {
            "train": train_s,
            "val": val_s,
            "test": test_s,
            "excluded_protected": excl_prot,
            "excluded_counter_duals": excl_dual,
            "pchembl_map": curated,
        }
        print(
            f"  [{tid:<6s} | {cohort:<12s}] clean={len(curated):4d} excl_prot={len(excl_prot):3d} "
            f"excl_dual={len(excl_dual):3d} | tr={len(train_s):4d} va={len(val_s):3d} "
            f"te={len(test_s):3d} | leak=0",
            flush=True,
        )

    _write_json(TABLE_SPLITS_PATH, splits_summary)
    _write_json(FULL_SPLITS_LOCAL_PATH, {"protected": protected, "splits": splits_full})

    # -----------------------------------------------------------------------
    # Combine Existing 12 v3.1 QSARs + Train 14 New Calibrated 5-Fold QSARs
    # -----------------------------------------------------------------------
    print(
        "\n[prepare] Loading 12 verified v3.1 QSARs & training 14 new 5-fold scaffold-CV QSARs...",
        flush=True,
    )
    joblib = _get_joblib()
    v31_qsar = joblib.load(V31_QSAR_MODEL_PATH)
    qsar_models: dict[str, dict[str, Any]] = dict(v31_qsar["models"])
    qsar_cv_endpoints: dict[str, Any] = dict(v31_qsar["cv_report"]["endpoints"])

    all_chembl_union: set[str] = set()
    for tid in all_qsar_targets:
        all_chembl_union.update(curated_by_target[tid].keys())
    zinc_clean = _load_zinc250k_smiles(exclude_set=all_chembl_union)
    rng = np.random.default_rng(seed + 404)
    zinc_pool = [
        zinc_clean[int(i)] for i in rng.choice(len(zinc_clean), size=25_000, replace=False)
    ]

    priority_neg_map: dict[str, tuple[str, ...]] = {
        "CHRM1": ("CHRM4", "DRD2", "HTR2A"),
        "HTR1A": ("HTR2A", "DRD2", "DRD3"),
        "GLYT1": ("SERT", "DAT", "VMAT2"),
        "MGLL": ("ACHE", "PDE10A"),
        "PDE1B": ("PDE10A", "VMAT2"),
        "GPR52": ("DRD2", "DRD1", "TAAR1"),
        "ACHE": ("MGLL", "CHRM4", "CHRM1"),
        "OGA": ("BACE1", "BACE2"),
        "NLRP3": ("GSK3B", "PDE10A"),
        "MAOB": ("MAOA", "DAT", "SERT"),
        "MAOA": ("MAOB", "DAT", "SERT"),
        "GSK3B": ("NLRP3", "PDE10A"),
        "BACE1": ("BACE2", "OGA"),
        "BACE2": ("BACE1", "OGA"),
    }

    new_endpoints = (*NEW_12_TARGETS, *COUNTER_QSAR_TARGETS)
    for idx, tid in enumerate(new_endpoints):
        curated_t = curated_by_target[tid]
        if tid == "MAOB":
            positives = sorted(
                s
                for s, p in curated_t.items()
                if counter_potency_maps["MAOA"].get(s, -1.0) <= p - 0.5
            )
        elif tid == "BACE1":
            positives = sorted(
                s
                for s, p in curated_t.items()
                if counter_potency_maps["BACE2"].get(s, -1.0) <= p - 0.5
            )
        else:
            positives = sorted(curated_t.keys())

        pos_set = set(positives)
        pri_targets = set(priority_neg_map.get(tid, ()))
        pri_negs: list[str] = []
        oth_negs: list[str] = []
        for other_tid in all_qsar_targets:
            if other_tid == tid:
                continue
            for smi in curated_by_target[other_tid]:
                if smi in pos_set or smi in curated_t:
                    continue
                if other_tid in pri_targets:
                    pri_negs.append(smi)
                else:
                    oth_negs.append(smi)
        pri_negs = sorted(set(pri_negs))
        oth_negs = sorted(set(oth_negs) - set(pri_negs))

        n_pos = len(positives)
        n_neg_total = min(max(n_pos * 2, 600), 6000)
        n_cross_want = n_neg_total // 2
        t_rng = np.random.default_rng(seed + 700 + idx)

        sel_cross: list[str] = []
        if pri_negs:
            take_pri = min(len(pri_negs), max(n_cross_want // 2, 200))
            sel_cross.extend(
                pri_negs[int(i)] for i in t_rng.choice(len(pri_negs), size=take_pri, replace=False)
            )
        rem = max(0, n_cross_want - len(sel_cross))
        if rem > 0 and oth_negs:
            take_oth = min(len(oth_negs), rem)
            sel_cross.extend(
                oth_negs[int(i)] for i in t_rng.choice(len(oth_negs), size=take_oth, replace=False)
            )
        n_zinc_want = max(200, n_neg_total - len(sel_cross))
        sel_zinc = [
            zinc_pool[int(i)]
            for i in t_rng.choice(
                len(zinc_pool), size=min(n_zinc_want, len(zinc_pool)), replace=False
            )
        ]
        negatives = sorted((set(sel_cross) | set(sel_zinc)) - pos_set)
        all_smis = positives + negatives
        all_labels = [1] * len(positives) + [0] * len(negatives)

        bundle, cv_m = _train_single_qsar_endpoint(
            tid, all_smis, all_labels, n_splits=5, n_jobs=n_jobs, seed=seed + 100 + idx
        )
        qsar_models[tid] = bundle
        qsar_cv_endpoints[tid] = cv_m
        print(
            f"  [QSAR {tid:<6s}] n={cv_m['n_samples']:5d} "
            f"(+{cv_m['n_positive']:4d}/-{cv_m['n_negative']:4d}) "
            f"| ROC-AUC={cv_m['oof_roc_auc']:.4f} PR-AUC={cv_m['oof_pr_auc']:.4f} "
            f"ECE={cv_m['oof_ece']:.4f} leak={cv_m['scaffold_leakage_groups']}",
            flush=True,
        )

    # Clinical sanity check across Tri-Disorder reference drugs
    clinical_checks = [
        (
            "Ecopipam",
            "DRD1",
            "DRD2",
            TRI_DISORDER_TARGET_REGISTRY["DRD1"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Donepezil",
            "ACHE",
            "MAOB",
            TRI_DISORDER_TARGET_REGISTRY["ACHE"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Safinamide",
            "MAOB",
            "MAOA",
            TRI_DISORDER_TARGET_REGISTRY["MAOB"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Harmine",
            "MAOA",
            "MAOB",
            V4_COUNTER_TARGET_REGISTRY["MAOA"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Umibecestat",
            "BACE1",
            "BACE2",
            TRI_DISORDER_TARGET_REGISTRY["BACE1"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Ceperognastat",
            "OGA",
            "BACE1",
            TRI_DISORDER_TARGET_REGISTRY["OGA"].reference_drugs[0].canonical_smiles,
        ),
        (
            "NP3-253",
            "NLRP3",
            "GSK3B",
            TRI_DISORDER_TARGET_REGISTRY["NLRP3"].reference_drugs[0].canonical_smiles,
        ),
        (
            "BRD0705",
            "GSK3B",
            "MAOB",
            TRI_DISORDER_TARGET_REGISTRY["GSK3B"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Iclepertin",
            "GLYT1",
            "SERT",
            TRI_DISORDER_TARGET_REGISTRY["GLYT1"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Buspirone",
            "HTR1A",
            "DRD2",
            TRI_DISORDER_TARGET_REGISTRY["HTR1A"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Betovumeline",
            "CHRM1",
            "DRD2",
            TRI_DISORDER_TARGET_REGISTRY["CHRM1"].reference_drugs[0].canonical_smiles,
        ),
        (
            "Xanomeline",
            "CHRM4",
            "DRD2",
            TRI_DISORDER_TARGET_REGISTRY["CHRM4"].reference_drugs[0].canonical_smiles,
        ),
    ]
    c_smis = [c[3] for c in clinical_checks]
    x_c, fp_c, m_c, _ = featurize_admet_smiles(c_smis)
    fp_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    clinical_sanity: dict[str, Any] = {}
    for idx, (dname, on_t, off_t, _smi) in enumerate(clinical_checks):
        p_on_arr, _ = _predict_qsar_endpoint(qsar_models[on_t], x_c, fp_c, m_c, cache=fp_cache)
        p_off_arr, _ = _predict_qsar_endpoint(qsar_models[off_t], x_c, fp_c, m_c, cache=fp_cache)
        clinical_sanity[dname] = {
            "on_target": on_t,
            "off_target": off_t,
            "p_on_target": float(p_on_arr[idx]),
            "p_off_target": float(p_off_arr[idx]),
            "delta": float(p_on_arr[idx] - p_off_arr[idx]),
        }

    all_aucs = [float(v["oof_roc_auc"]) for v in qsar_cv_endpoints.values()]
    new_aucs = [float(qsar_cv_endpoints[t]["oof_roc_auc"]) for t in new_endpoints]
    qsar_cv_report: dict[str, Any] = {
        "schema_version": "geneuclidean-v4.0-26endpoint-tri-disorder-qsar-cv",
        "n_endpoints": len(qsar_models),
        "primary_24_targets": list(ALL_24_TARGETS),
        "counter_qsar_targets": list(COUNTER_QSAR_TARGETS),
        "structural_counter_targets_no_chembl_csv": ["BCHE", "CDK2"],
        "avoidome_oracles_in_admet_suite": ["HTR2B", "HRH1", "CHRM3"],
        "summary": {
            "mean_roc_auc_all_26": float(np.mean(all_aucs)),
            "min_roc_auc_all_26": float(np.min(all_aucs)),
            "mean_roc_auc_new_14": float(np.mean(new_aucs)),
            "zero_scaffold_leakage_all": all(
                int(v["scaffold_leakage_groups"]) == 0 for v in qsar_cv_endpoints.values()
            ),
        },
        "endpoints": qsar_cv_endpoints,
        "clinical_sanity": clinical_sanity,
    }
    qsar_payload = {
        "schema_version": "geneuclidean-v4.0-26endpoint-tri-disorder-qsar",
        "models": qsar_models,
        "cv_report": qsar_cv_report,
    }
    V4_QSAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(qsar_payload, V4_QSAR_MODEL_PATH, compress=3)
    qsar_cv_report["model_artifact"] = str(V4_QSAR_MODEL_PATH)
    qsar_cv_report["model_size_mb"] = round(
        V4_QSAR_MODEL_PATH.stat().st_size / (1024.0 * 1024.0), 3
    )
    qsar_cv_report["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    _write_json(TABLE_QSAR_CV_PATH, qsar_cv_report)
    print(
        f"[prepare] DONE in {qsar_cv_report['elapsed_sec']:.1f}s | "
        f"26-endpoint mean ROC-AUC={qsar_cv_report['summary']['mean_roc_auc_all_26']:.4f} | "
        f"Saved {V4_QSAR_MODEL_PATH} ({qsar_cv_report['model_size_mb']} MB)",
        flush=True,
    )
    return {"splits": splits_summary, "qsar_cv": qsar_cv_report}


# ---------------------------------------------------------------------------
# Stage 2: gate1 — Unbiased Multi-Seed Native Redocking on 16 New/Counter Pockets
# ---------------------------------------------------------------------------


def _redock_one_gate1_target(args: tuple[str, tuple[int, ...], int]) -> tuple[str, dict[str, Any]]:
    tid, seeds, vina_cpu = args
    spec = get_cns_target(tid)
    loc = find_local_pocket_files(tid, pdb_id=spec.primary_pdb_id)
    if loc is None or loc[1] is None:
        raise FileNotFoundError(f"Missing pocket/ligand files for Gate-1 target {tid}")
    pocket_pdb, lig_sdf = loc
    anchors, anchor_kinds = POCKET_ANCHOR_SPEC[tid]

    # Audit retained cofactors/metals directly in receptor PDB
    het_resnames: set[str] = set()
    for line in pocket_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("HETATM"):
            het_resnames.add(line[17:20].strip().upper())

    t0 = time.perf_counter()
    res = redock_native_complex(
        pocket_pdb,
        lig_sdf,
        seeds=seeds,
        canonical_anchors=anchors,
        expected_anchor_kinds=anchor_kinds,
        box_mode="tight_per_axis",
        vina_bin=VINA_BIN,
        exhaustiveness=8,
        num_modes=9,
        cpu=vina_cpu,
        num_confs=32,
        max_ring_confs=3,
        timeout=180,
    )
    ok_seeds = [s for s in res.seed_results if s.success]
    core_top1 = [s.top1_core_rmsd for s in ok_seeds if s.top1_core_rmsd is not None]
    core_best3 = [s.best3_core_rmsd for s in ok_seeds if s.best3_core_rmsd is not None]
    entry = {
        "target_id": tid,
        "pdb_id": spec.primary_pdb_id,
        "disease_area": spec.disease_area,
        "is_counter_target": tid in AD_STRUCTURAL_COUNTERS,
        "expected_cofactors": list(spec.keep_cofactors),
        "retained_hetatms_in_pocket": sorted(het_resnames),
        "canonical_anchors": list(anchors),
        "expected_anchor_kinds": list(anchor_kinds),
        "median_top1_core_rmsd": round(float(np.median(core_top1)), 3) if core_top1 else None,
        "median_best3_core_rmsd": round(float(np.median(core_best3)), 3) if core_best3 else None,
        "best3_anchor_recovery_rate": round(
            float(np.mean([1.0 if s.best3_anchor_recovered else 0.0 for s in ok_seeds])), 4
        ),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        **res.to_dict(),
    }
    return tid, entry


def run_stage_gate1(*, max_workers: int = 4) -> dict[str, Any]:
    """Execute Stage ``gate1``: Unbiased 3-seed native redocking across 16 expansion pockets."""
    t0 = time.perf_counter()
    seeds = (42, 137, 2026)
    print(
        f"[gate1] Running unbiased 3-seed native redocking on {len(GATE1_EXPANSION_16)} "
        f"new/counter human complexes (box=tight_per_axis, workers={max_workers})...",
        flush=True,
    )
    tasks = [(tid, seeds, 2) for tid in GATE1_EXPANSION_16]
    results_map: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_redock_one_gate1_target, t) for t in tasks]
        for fut in as_completed(futs):
            tid, entry = fut.result()
            results_map[tid] = entry
            print(
                f"  [Gate1 {tid:<6s} {entry['pdb_id']}] Top1={entry['median_top1_rmsd']:.2f}A "
                f"Best3={entry['median_best3_rmsd']:.2f}A "
                f"Vina={entry['median_vina_affinity']:.2f} LE={entry['median_le']:.3f} "
                f"PLIF={entry['mean_top1_plif_tanimoto']:.3f} "
                f"PB={entry['top1_posebusters_pass_rate'] * 100:.0f}% "
                f"Anc={entry['top1_anchor_recovery_rate'] * 100:.0f}% ({entry['status']}) "
                f"het={entry['retained_hetatms_in_pocket']} [{entry['elapsed_sec']:.1f}s]",
                flush=True,
            )

    ordered = {tid: results_map[tid] for tid in GATE1_EXPANSION_16}
    vals = list(ordered.values())
    n_pass_top1 = sum(1 for v in vals if float(v["median_top1_rmsd"]) <= 2.0)
    n_pass_best3 = sum(1 for v in vals if float(v["median_best3_rmsd"]) <= 2.0)
    n_pass_25 = sum(1 for v in vals if float(v["median_best3_rmsd"]) <= 2.5)
    summary = {
        "n_complexes": len(vals),
        "n_new_primary": len(NEW_12_TARGETS),
        "n_ad_counter_targets": len(AD_STRUCTURAL_COUNTERS),
        "pass_top1_le_2p0A_count": n_pass_top1,
        "pass_top1_le_2p0A_rate": round(n_pass_top1 / len(vals), 4),
        "pass_best3_le_2p0A_count": n_pass_best3,
        "pass_best3_le_2p0A_rate": round(n_pass_best3 / len(vals), 4),
        "pass_best3_le_2p5A_count": n_pass_25,
        "median_top1_rmsd_across_16": round(
            float(np.median([v["median_top1_rmsd"] for v in vals])), 3
        ),
        "median_best3_rmsd_across_16": round(
            float(np.median([v["median_best3_rmsd"] for v in vals])), 3
        ),
        "mean_plif_tanimoto_across_16": round(
            float(np.mean([v["mean_top1_plif_tanimoto"] for v in vals])), 4
        ),
        "mean_posebusters_pass_rate": round(
            float(np.mean([v["top1_posebusters_pass_rate"] for v in vals])), 4
        ),
        "mean_top1_anchor_recovery_rate": round(
            float(np.mean([v["top1_anchor_recovery_rate"] for v in vals])), 4
        ),
        "mean_best3_anchor_recovery_rate": round(
            float(np.mean([v["best3_anchor_recovery_rate"] for v in vals])), 4
        ),
        "mean_vina_affinity": round(float(np.mean([v["median_vina_affinity"] for v in vals])), 3),
        "mean_le": round(float(np.mean([v["median_le"] for v in vals])), 4),
        "cofactors_verified": {
            "MAOB_2V5Z_FAD": "FAD" in ordered["MAOB"]["retained_hetatms_in_pocket"],
            "MAOA_2Z5X_FAD": "FAD" in ordered["MAOA"]["retained_hetatms_in_pocket"],
            "NLRP3_9GU4_ADP": "ADP" in ordered["NLRP3"]["retained_hetatms_in_pocket"],
            "PDE1B_5UOY_ZN_MG": (
                "ZN" in ordered["PDE1B"]["retained_hetatms_in_pocket"]
                and "MG" in ordered["PDE1B"]["retained_hetatms_in_pocket"]
            ),
        },
    }
    payload = {
        "schema_version": "geneuclidean-v4.0-gate1-expansion-redocking",
        "protocol": {
            "box_mode": "tight_per_axis (clamp(span_i + 6.0, 14.0, 22.0) A)",
            "protonation": "deterministic physiological pH 7.4",
            "seeds": list(seeds),
            "exhaustiveness": 8,
            "num_confs": 32,
            "max_ring_confs": 3,
            "unbiased_2d_to_3d": True,
        },
        "summary": summary,
        "complexes": ordered,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    _write_json(TABLE_GATE1_PATH, payload)
    print(
        f"[gate1] DONE in {payload['elapsed_sec']:.1f}s | "
        f"Best3<=2.0A: {n_pass_best3}/16 ({summary['pass_best3_le_2p0A_rate'] * 100:.1f}%) | "
        f"Top1<=2.0A: {n_pass_top1}/16 | "
        f"Med Best3 RMSD={summary['median_best3_rmsd_across_16']:.2f}A | "
        f"PB={summary['mean_posebusters_pass_rate'] * 100:.1f}%",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Model Factory & Shared Frozen 3D Pocket Encoding Bank
# ---------------------------------------------------------------------------


def _build_backbone_model(
    backbone_id: str,
    *,
    device: torch.device,
    adapter_ckpt: Path | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple[PocketConditionedLLM, ExperimentConfig]:
    """Build ``PocketConditionedLLM`` with frozen ``PaiNN-SE3``/``CVAE``/``heads``."""
    cfg = ExperimentConfig.from_yaml(V3_CONFIG_PATH)
    cfg.data.randomize_smiles = False
    os.environ["HF_HOME"] = str(Path(".hf_cache.nosync").resolve())

    backbone = load_hf_backbone(
        backbone_id=backbone_id,
        cache_dir=".hf_cache.nosync",
        dtype="auto",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_lora=True,
        local_files_only=True,
    )
    assert cfg.llm is not None
    cfg.llm.backbone_id = backbone.backbone_id
    cfg.llm.dtype = "bfloat16" if backbone.model_dtype == torch.bfloat16 else "float32"

    model = PocketConditionedLLM(
        v2_cfg=cfg,
        backbone=backbone,
        num_queries=cfg.llm.num_queries,
        perceiver_layers=cfg.llm.perceiver_layers,
        perceiver_heads=cfg.llm.perceiver_heads,
        num_global_tokens=cfg.llm.num_global_tokens,
        cfg_drop_prob=cfg.llm.cfg_drop_prob,
        freeze_pocket_encoder=True,
    )
    model.load_v2_pocket_checkpoint(V2_PAINN_CKPT)
    if adapter_ckpt is not None:
        model.load_adapter_checkpoint(adapter_ckpt, map_location="cpu")
    model.to(device)
    model.eval()
    return model, cfg


def _precompute_all_v4_pocket_embeddings(
    model: PocketConditionedLLM,
    device: torch.device,
    *,
    n_jitters: int = 2,
    jitter_std: float = 0.05,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Encode ALL 24 primary targets (all same-state PDBs + jitters) + 4 AD counter pockets ONCE."""
    rng = np.random.default_rng(seed)
    train_bank: dict[str, list[dict[str, Any]]] = {}
    primary_map: dict[str, dict[str, Any]] = {}

    model.encoder.to(device)
    model.encoder.eval()
    total_variants = 0

    for tid in ALL_24_TARGETS:
        spec = TRI_DISORDER_TARGET_REGISTRY[tid]
        variants: list[dict[str, Any]] = []
        for pdb_id in SAME_STATE_PDBS_24[tid]:
            pf = load_cns_target_pocket(tid, pdb_id=pdb_id)
            base_coords = pf.coords.astype(np.float32)
            base_nf = pf.node_feats.astype(np.float32)
            n_atoms = int(base_coords.shape[0])

            for j_idx in range(1 + n_jitters):
                if j_idx == 0:
                    c_arr = base_coords
                    tag = f"{tid}_{pdb_id}_base"
                else:
                    c_arr = base_coords + rng.normal(
                        0.0, jitter_std, size=base_coords.shape
                    ).astype(np.float32)
                    tag = f"{tid}_{pdb_id}_jit{j_idx}"
                with torch.no_grad():
                    p_out = model.encode_pocket(
                        torch.from_numpy(c_arr).unsqueeze(0).to(device),
                        torch.from_numpy(base_nf).unsqueeze(0).to(device),
                        torch.ones((1, n_atoms), dtype=torch.bool, device=device),
                    )
                var_entry = {
                    "tag": tag,
                    "target_id": tid,
                    "pdb_id": pdb_id,
                    "is_base": (j_idx == 0),
                    "node_emb": p_out.node_emb[0].detach().cpu().float().contiguous(),
                    "pooled_emb": p_out.pooled_emb[0].detach().cpu().float().contiguous(),
                    "atom_mask": torch.ones((n_atoms,), dtype=torch.bool),
                    "native_smiles": pf.canonical_smiles
                    or spec.reference_drugs[0].canonical_smiles,
                }
                variants.append(var_entry)
                total_variants += 1
                if pdb_id == spec.primary_pdb_id and j_idx == 0:
                    primary_map[tid] = {
                        **var_entry,
                        "pocket_out_cpu": PocketEncoderOutput(
                            node_emb=var_entry["node_emb"].unsqueeze(0),
                            pooled_emb=var_entry["pooled_emb"].unsqueeze(0),
                            vec_emb=None,
                        ),
                        "atom_mask_cpu": var_entry["atom_mask"].unsqueeze(0),
                    }
        train_bank[tid] = variants

    # Also encode the 4 structural AD counter-target pockets (BCHE, MAOA, CDK2, BACE2) for L_XP-DPO
    for ctid in AD_STRUCTURAL_COUNTERS:
        spec = V4_COUNTER_TARGET_REGISTRY[ctid]
        pf = load_cns_target_pocket(ctid, pdb_id=spec.primary_pdb_id)
        n_atoms = int(pf.coords.shape[0])
        with torch.no_grad():
            p_out = model.encode_pocket(
                torch.from_numpy(pf.coords.astype(np.float32)).unsqueeze(0).to(device),
                torch.from_numpy(pf.node_feats.astype(np.float32)).unsqueeze(0).to(device),
                torch.ones((1, n_atoms), dtype=torch.bool, device=device),
            )
        node_c = p_out.node_emb[0].detach().cpu().float().contiguous()
        pool_c = p_out.pooled_emb[0].detach().cpu().float().contiguous()
        mask_c = torch.ones((n_atoms,), dtype=torch.bool)
        primary_map[ctid] = {
            "tag": f"{ctid}_{spec.primary_pdb_id}_base",
            "target_id": ctid,
            "pdb_id": spec.primary_pdb_id,
            "is_base": True,
            "node_emb": node_c,
            "pooled_emb": pool_c,
            "atom_mask": mask_c,
            "pocket_out_cpu": PocketEncoderOutput(
                node_emb=node_c.unsqueeze(0),
                pooled_emb=pool_c.unsqueeze(0),
                vec_emb=None,
            ),
            "atom_mask_cpu": mask_c.unsqueeze(0),
            "native_smiles": pf.canonical_smiles or spec.reference_drugs[0].canonical_smiles,
        }

    print(
        f"[pockets] Pre-encoded {total_variants} primary 3D variants (24 targets) + "
        f"{len(AD_STRUCTURAL_COUNTERS)} structural counter pockets.",
        flush=True,
    )
    return train_bank, primary_map


def _pocket_entry_to_device(
    entry: Mapping[str, Any], device: torch.device
) -> tuple[PocketEncoderOutput, torch.Tensor]:
    p_cpu: PocketEncoderOutput = entry["pocket_out_cpu"]
    return (
        PocketEncoderOutput(
            node_emb=p_cpu.node_emb.to(device),
            pooled_emb=p_cpu.pooled_emb.to(device),
            vec_emb=None,
        ),
        entry["atom_mask_cpu"].to(device),
    )


def _select_maxmin_train_pool(
    smiles_list: Sequence[str], pchembl_map: Mapping[str, float], max_k: int = 300
) -> list[str]:
    if len(smiles_list) <= max_k:
        return list(smiles_list)
    scores = [float(pchembl_map.get(s, 6.5)) for s in smiles_list]
    idxs = maxmin_diversity_select(
        smiles_list,
        scores=scores,
        k=max_k,
        min_tanimoto_dist=0.25,
        quality_weight=0.55,
        relax_if_needed=True,
    )
    return [smiles_list[i] for i in idxs]


# ---------------------------------------------------------------------------
# Multiplicative Scoring & Safety Gate Definitions (v3.1 7-Rule + v4.0 12-Endpoint)
# ---------------------------------------------------------------------------


def _v31_7rule_safe_flag(r: MultiplicativeCNSReward) -> bool:
    """Exact apples-to-apples v3.1 7-rule Joint CNS-Safe gate."""
    if not r.valid or r.admet_v4 is None or not r.admet_v4.valid:
        return False
    p = r.admet_v4.probabilities
    ad = r.admet_v4.ad_tanimoto
    min_ad_v3 = min(ad.get("bbb", 0.0), ad.get("pgp", 0.0), ad.get("herg", 0.0))
    return bool(
        r.mpo.cns_mpo_v2 >= 4.0
        and p.get("bbb", 0.0) >= 0.80
        and p.get("pgp", 1.0) <= 0.40
        and p.get("herg", 1.0) <= 0.35
        and r.mpo.pains_free
        and r.mpo.sa_score <= 3.5
        and min_ad_v3 >= 0.30
    )


def _v4_pharma_12ep_safe_flag(r: MultiplicativeCNSReward) -> bool:
    """Exact v4.0 12-endpoint Industrial Pharma-Safe gate."""
    if not r.valid or r.admet_v4 is None or not r.admet_v4.valid:
        return False
    p = r.admet_v4.probabilities
    ad = r.admet_v4.ad_tanimoto
    core_ad_eps = ("bbb", "herg", "mdr1_efflux", "hlm_clearance", "cyp2d6", "cyp3a4")
    min_ad = min(ad.get(ep, 0.0) for ep in core_ad_eps if ep in ad)
    return bool(
        r.mpo.cns_mpo_v2 >= 4.0
        and p.get("bbb", 0.0) >= 0.80
        and p.get("mdr1_efflux", 1.0) < 0.45
        and p.get("hlm_clearance", 1.0) < 0.50
        and p.get("herg", 1.0) <= 0.30
        and p.get("cyp2d6", 1.0) <= 0.40
        and p.get("cyp3a4", 1.0) <= 0.45
        and p.get("htr2b_avoid", 1.0) <= 0.35
        and p.get("hrh1_avoid", 1.0) <= 0.35
        and r.triage.triage_pass
        and min_ad >= 0.30
    )


def _score_candidates_v4(
    smiles_list: Sequence[str],
    target_id: str,
    *,
    pred_pkd: Sequence[float] | None,
    clip_sim: Sequence[float] | None,
    admet_suite: CNSAdmetSuite,
    qsar_payload: Mapping[str, Any],
    qsar_fp_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[MultiplicativeCNSReward], dict[str, np.ndarray]]:
    """Compute exact v4.0 Multiplicative Safety-Gated Reward (12-ep Avoid-ome + DCU veto + QSAR)."""
    q_res = score_target_and_offtarget_qsar(
        qsar_payload, smiles_list, target_id, fp_cache=qsar_fp_cache
    )
    # AD-damp QSAR probabilities inside reward to guard against OOD reward hacking
    p_on_raw = q_res["p_on_target"]
    p_off_raw = q_res["p_off_max"]
    ad_on = q_res["ad_on_target"]
    w_on = np.minimum(1.0, np.maximum(0.0, ad_on) / 0.30)
    p_on_damped = 0.5 + w_on * (p_on_raw - 0.5)

    rewards = compute_multiplicative_cns_reward(
        smiles_list,
        on_target_prob=p_on_damped,
        off_target_max_prob=p_off_raw,
        pred_pkd=pred_pkd,
        clip_sim=clip_sim,
        vina_scores=None,  # Strictly held out!
        admet_suite=admet_suite,
        ad_damp=True,
        gamma_safety=0.65,
        gamma_target=1.00,
        gamma_3d=0.30,
    )
    return rewards, q_res


def _effective_alignment_score(
    r: MultiplicativeCNSReward,
    *,
    p_vmat2: float = 0.0,
    target_id: str = "",
) -> float:
    """Multiplicative hygiene-vetoed alignment score with 12-ep Pharma-Safe & anti-sink shaping."""
    if not r.valid or r.hygiene_gate <= 0.0:
        return 0.0
    score = float(r.hygiene_gate) * (
        0.38 * float(r.total_reward)
        + 0.37 * float(r.safety_gate)
        + 0.15 * float(r.target_sel_gate)
        + 0.10 * float(r.geometry_3d_gate)
    )
    if r.mpo.mw < 215.0:
        score *= 0.72
    elif r.mpo.mw > 500.0:
        score *= 0.80
    if target_id != "VMAT2" and p_vmat2 >= 0.50:
        score *= 0.55
    if _v4_pharma_12ep_safe_flag(r):
        score = min(1.0, score + 0.18)
    elif _v31_7rule_safe_flag(r):
        score = min(1.0, score + 0.06)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Stage 3: train — 3-Backbone Local LLM SFT + Multiplicative GDPO / L_XP-DPO
# ---------------------------------------------------------------------------


def _run_sft_for_backbone(
    arm_key: str,
    backbone_id: str,
    *,
    init_adapter_ckpt: Path | None,
    phase_a_epochs: int,
    phase_b_epochs: int,
    projector_lr: float,
    bridge_lr: float,
    lora_lr: float,
    batch_size: int,
    randomize_safe_order: bool,
    train_bank: Mapping[str, list[dict[str, Any]]],
    primary_map: Mapping[str, dict[str, Any]],
    splits_full: Mapping[str, dict[str, Any]],
    replay_cache: Sequence[CachedPocketRecord],
    device: torch.device,
    seed: int,
) -> tuple[PocketConditionedLLM, ExperimentConfig, dict[str, Any]]:
    """Train SFT (Phase A optional + Phase B) on all 24 targets + 15% PDBbind replay."""
    t0 = time.perf_counter()
    seed_everything(seed)
    save_dir = RUN_ROOT / f"{arm_key}_sft"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_adapter.pt"
    summary_path = save_dir / "train_summary.json"

    print(
        f"\n[train:SFT:{arm_key}] backbone={backbone_id} | "
        f"phaseA={phase_a_epochs} ep, phaseB={phase_b_epochs} ep | "
        f"warm_adapter={init_adapter_ckpt is not None}",
        flush=True,
    )
    model, cfg = _build_backbone_model(backbone_id, device=device, adapter_ckpt=init_adapter_ckpt)
    cfg.name = f"geneuclidean-v4-{arm_key}-sft"
    cfg.train.batch_size = batch_size
    cfg.train.seed = seed
    cfg.data.randomize_smiles = bool(randomize_safe_order and model.is_safe_mode)
    assert cfg.llm is not None
    cfg.llm.cfg_drop_prob = 0.15
    cfg.llm.phase_a_epochs = phase_a_epochs
    cfg.llm.phase_b_epochs = phase_b_epochs
    cfg.llm.projector_lr = projector_lr
    cfg.llm.phase_b_projector_lr = bridge_lr
    cfg.llm.lora_lr = lora_lr

    frozen_snap = _snapshot_param_checksums(
        model, ("encoder.", "cvae.", "heads."), include_base_llm=True
    )

    # Temperature-balanced per-epoch quotas (alpha=0.5, cap=150, min=55) across all 24 targets
    target_pools: dict[str, list[str]] = {}
    raw_counts: dict[str, int] = {}
    for tid in ALL_24_TARGETS:
        tr_smis = splits_full[tid]["train"]
        p_map = splits_full[tid]["pchembl_map"]
        target_pools[tid] = _select_maxmin_train_pool(tr_smis, p_map, max_k=300)
        raw_counts[tid] = max(1, len(tr_smis))

    weights_a = {tid: float(raw_counts[tid]) ** 0.5 for tid in ALL_24_TARGETS}
    max_w = max(weights_a.values())
    epoch_quotas: dict[str, int] = {}
    for tid in ALL_24_TARGETS:
        q = int(round(150.0 * (weights_a[tid] / max_w)))
        q = max(min(55, len(target_pools[tid]) * 2), min(150, q))
        epoch_quotas[tid] = q
    n_cns_epoch = sum(epoch_quotas.values())
    n_replay_epoch = int(round(n_cns_epoch * 0.15 / 0.85))
    print(
        f"  [SFT:{arm_key}] Epoch mix: {n_cns_epoch} CNS (24 targets, alpha=0.5) + "
        f"{n_replay_epoch} PDBbind replay (15%) = {n_cns_epoch + n_replay_epoch} samples/ep",
        flush=True,
    )

    # Compact balanced Val-scaffold cache (up to 40 val actives/target -> ~800 total)
    val_cache: list[CachedPocketRecord] = []
    for tid in ALL_24_TARGETS:
        pvar = primary_map[tid]
        for smi in splits_full[tid]["val"][:40]:
            val_cache.append(
                CachedPocketRecord(
                    pdb_id=f"{tid}_val",
                    node_emb=pvar["node_emb"],
                    pooled_emb=pvar["pooled_emb"],
                    atom_mask=pvar["atom_mask"],
                    ligand_fp=None,
                    canonical_smiles=smi,
                )
            )

    trainer = PocketLLMTrainer(model=model, cfg=cfg, device=device)
    init_val = trainer.evaluate(val_cache)
    print(
        f"  [SFT:{arm_key}] Pre-SFT Val NLL={init_val['val_lm_loss']:.4f} "
        f"(ppl={init_val['val_ppl']:.2f})",
        flush=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    global_ep = 0

    for phase, n_ep in (("A", phase_a_epochs), ("B", phase_b_epochs)):
        if n_ep <= 0:
            continue
        trainer.configure_phase(phase)
        # Strict freeze check
        for name, p in model.named_parameters():
            if name.startswith(("encoder.", "cvae.", "heads.")):
                assert not p.requires_grad, f"Frozen 3D param {name} unfrozen!"
            elif name.startswith("llm.") and (
                "lora_" not in name and "modules_to_save" not in name
            ):
                assert not p.requires_grad, f"Frozen base LLM param {name} unfrozen!"

        optimizer = trainer._build_phase_optimizer(phase)
        steps_ep = max(1, math.ceil((n_cns_epoch + n_replay_epoch) / batch_size))
        scheduler = build_cosine_warmup_scheduler(
            optimizer, total_steps=steps_ep * n_ep, warmup_ratio=0.10
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        for _local_ep in range(1, n_ep + 1):
            global_ep += 1
            ep_t0 = time.perf_counter()
            ep_rng = np.random.default_rng(seed + 1000 * global_ep)

            ep_records: list[CachedPocketRecord] = []
            for tid in ALL_24_TARGETS:
                pool = target_pools[tid]
                variants = train_bank[tid]
                q = epoch_quotas[tid]
                idxs = ep_rng.choice(len(pool), size=q, replace=(q > len(pool)))
                for idx in idxs:
                    var = variants[int(ep_rng.integers(0, len(variants)))]
                    ep_records.append(
                        CachedPocketRecord(
                            pdb_id=var["tag"],
                            node_emb=var["node_emb"],
                            pooled_emb=var["pooled_emb"],
                            atom_mask=var["atom_mask"],
                            ligand_fp=None,
                            canonical_smiles=pool[int(idx)],
                        )
                    )
            rep_idxs = ep_rng.choice(
                len(replay_cache), size=min(n_replay_epoch, len(replay_cache)), replace=False
            )
            for ri in rep_idxs:
                ep_records.append(replay_cache[int(ri)])

            loader = trainer._make_dataloader(
                ep_records,
                shuffle=True,
                randomize=bool(randomize_safe_order and model.is_safe_mode),
                seed=seed + 100 * global_ep,
            )
            trainer._set_train_modes(phase)
            lm_sum = 0.0
            tok_sum = 0
            gn_sum = 0.0
            n_steps = 0

            for raw_batch in loader:
                batch = trainer._move_cached_batch(raw_batch)
                valid_tok = int((batch["labels"][:, 1:] != -100).sum().item())
                if valid_tok <= 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    pocket_out=batch["pocket_out"],
                    atom_mask=batch["atom_mask"],
                    ligand_fp=None,
                    cfg_drop=True,
                    use_posterior=False,
                    return_attn=False,
                )
                assert out.loss is not None
                out.loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item())
                optimizer.step()
                scheduler.step()
                lm_sum += float(out.loss.detach().float().item()) * valid_tok
                tok_sum += valid_tok
                gn_sum += gn
                n_steps += 1

            train_lm = lm_sum / float(max(1, tok_sum))
            val_m = trainer.evaluate(val_cache)
            val_lm = val_m["val_lm_loss"]
            val_ppl = val_m["val_ppl"]
            is_best = val_lm < best_val_loss
            if is_best or (phase == "B" and not best_ckpt.is_file()):
                best_val_loss = min(best_val_loss, val_lm)
                best_epoch = global_ep
                model.save_adapter_checkpoint(
                    best_ckpt,
                    extra_meta={
                        "stage": "v4_sft",
                        "arm": arm_key,
                        "backbone_id": backbone_id,
                        "phase": phase,
                        "epoch": global_ep,
                        "val_lm_loss": val_lm,
                        "val_ppl": val_ppl,
                    },
                )
            row = {
                "epoch": global_ep,
                "phase": phase,
                "train_lm_loss": train_lm,
                "train_ppl": math.exp(min(train_lm, 20.0)),
                "val_lm_loss": val_lm,
                "val_ppl": val_ppl,
                "grad_norm": gn_sum / float(max(1, n_steps)),
                "elapsed_sec": round(time.perf_counter() - ep_t0, 2),
                "is_best": is_best,
            }
            history.append(row)
            print(
                f"  [SFT:{arm_key} Ph-{phase} ep {global_ep}] "
                f"train_lm={train_lm:.4f} (ppl={row['train_ppl']:.2f}) | "
                f"val_lm={val_lm:.4f} (ppl={val_ppl:.2f}) | gn={row['grad_norm']:.2f} | "
                f"{row['elapsed_sec']:.1f}s{' *BEST*' if is_best else ''}",
                flush=True,
            )

    model.load_adapter_checkpoint(best_ckpt, map_location="cpu")
    model.to(device).eval()
    freeze_audit = _verify_frozen_bit_exact(model, frozen_snap)
    summary = {
        "arm": arm_key,
        "backbone_id": backbone_id,
        "format_mode": model.format_mode,
        "dtype": str(model.llm_dtype),
        "best_epoch": best_epoch,
        "init_val_lm_loss": init_val["val_lm_loss"],
        "init_val_ppl": init_val["val_ppl"],
        "best_val_lm_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20.0)),
        "adapter_ckpt_mb": round(best_ckpt.stat().st_size / (1024.0 * 1024.0), 3),
        "frozen_audit": freeze_audit,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "history": history,
    }
    _write_json(summary_path, summary)
    print(
        f"  [SFT:{arm_key}] DONE in {summary['elapsed_sec']:.1f}s | "
        f"best_val_ppl={summary['best_val_ppl']:.2f} | "
        f"bit_exact_frozen={freeze_audit['bit_exact']}",
        flush=True,
    )
    return model, cfg, summary


def _collect_alignment_rollouts(
    model: PocketConditionedLLM,
    primary_map: Mapping[str, dict[str, Any]],
    *,
    device: torch.device,
    samples_per_target: int = 48,
    temperature: float = 0.90,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample on-policy rollouts across all 24 primary human targets for GDPO & L_XP-DPO."""
    t0 = time.perf_counter()
    model.eval()
    rollouts: list[dict[str, Any]] = []
    for t_idx, tid in enumerate(ALL_24_TARGETS):
        p_entry = primary_map[tid]
        p_dev, mask_dev = _pocket_entry_to_device(p_entry, device)
        raw_sampled = sample_llm_molecules(
            model,
            p_dev,
            mask_dev,
            num_samples=samples_per_target,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            strategy="min_p",
            temperature=temperature,
            min_p=EVAL_MIN_P,
            cfg_scale=EVAL_CFG_SCALE,
            latent_temp=1.0,
            repetition_penalty=EVAL_REPETITION_PENALTY,
            batch_size=min(48, samples_per_target),
            seed=seed + 5000 + t_idx,
        )
        unique_valid: list[str] = []
        seen: set[str] = set()
        for raw_s in raw_sampled:
            can = _decode_generated_molecule(raw_s, format_mode=model.format_mode)
            if can is not None and can not in seen:
                seen.add(can)
                unique_valid.append(can)

        pred_pkd: list[float] | None = None
        clip_sim: list[float] | None = None
        if unique_valid:
            with torch.no_grad():
                fps_t = _smiles_to_fp_tensor(unique_valid).to(device)
                sc = model.score_candidates(p_dev, fps_t)
                pred_pkd = sc["pkd"].detach().cpu().numpy().tolist() if "pkd" in sc else None
                clip_sim = (
                    sc["clip_sim"].detach().cpu().numpy().tolist() if "clip_sim" in sc else None
                )

        off_tid = OFF_TARGET_POCKET_MAP[tid]
        off_entry = primary_map[off_tid]
        rollouts.append(
            {
                "target_id": tid,
                "off_target_id": off_tid,
                "n_generated": len(raw_sampled),
                "unique_valid": unique_valid,
                "pred_pkd": pred_pkd,
                "clip_sim": clip_sim,
                "pocket_batch": {
                    "node_emb": p_entry["pocket_out_cpu"].node_emb,
                    "pooled_emb": p_entry["pocket_out_cpu"].pooled_emb,
                    "atom_mask": p_entry["atom_mask_cpu"],
                },
                "off_target_pocket_emb": {
                    "node_emb": off_entry["pocket_out_cpu"].node_emb,
                    "pooled_emb": off_entry["pocket_out_cpu"].pooled_emb,
                    "atom_mask": off_entry["atom_mask_cpu"],
                },
            }
        )
    tot_gen = sum(r["n_generated"] for r in rollouts)
    tot_uv = sum(len(r["unique_valid"]) for r in rollouts)
    print(
        f"  [rollout] Sampled {tot_gen} across 24 targets -> {tot_uv} unique valid "
        f"({100.0 * tot_uv / max(1, tot_gen):.1f}%) in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return rollouts


def _build_xp_pairs_and_gdpo_groups(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    admet_suite: CNSAdmetSuite,
    qsar_payload: Mapping[str, Any],
    qsar_fp_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    max_pairs_per_target: int = 16,
    min_margin: float = 0.06,
    seed: int = 42,
) -> tuple[list[CrossPocketPreferencePair], list[GDPOGroup]]:
    """Build Multiplicative-scored L_XP-DPO pairs AND 4-Channel GDPO groups."""
    all_pairs: list[CrossPocketPreferencePair] = []
    all_groups: list[GDPOGroup] = []

    for item in rollouts:
        tid = str(item["target_id"])
        smis: list[str] = list(item["unique_valid"])
        if len(smis) < 2:
            continue
        rewards, q_res = _score_candidates_v4(
            smis,
            tid,
            pred_pkd=item["pred_pkd"],
            clip_sim=item["clip_sim"],
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
        )
        p_vm2_arr = q_res["p_vmat2"]
        eff_scores = [
            _effective_alignment_score(r, p_vmat2=float(p_vm2_arr[i]), target_id=tid)
            for i, r in enumerate(rewards)
        ]

        # 4-Channel GDPO group
        safety_ch = [float(r.safety_gate) for r in rewards]
        sel_ch = [float(r.target_sel_gate) for r in rewards]
        fit3d_ch = [float(r.geometry_3d_gate) for r in rewards]
        medchem_ch = [
            0.55 * float(r.hygiene_gate)
            + 0.45 * max(0.0, 1.0 - r.triage.lilly_demerit_score / 300.0)
            for r in rewards
        ]
        hyg_gates = [float(r.hygiene_gate) for r in rewards]
        all_groups.append(
            GDPOGroup(
                pocket_batch=item["pocket_batch"],
                smiles_list=smis,
                safety_rewards=safety_ch,
                selectivity_rewards=sel_ch,
                fit_3d_rewards=fit3d_ch,
                medchem_anchor_rewards=medchem_ch,
                hygiene_gates=hyg_gates,
            )
        )

        # Scaffold-diverse Cross-Pocket Preference Pairs (hygiene=0.0 never chosen!)
        ranked = sorted(
            zip(smis, eff_scores, rewards, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        chosen_pool: list[tuple[str, float]] = []
        seen_scafs: set[str] = set()
        for smi, sc, rew in ranked:
            if rew.hygiene_gate <= 0.0 or sc <= 0.05:
                continue
            scaf = compute_murcko_scaffold(smi)
            if scaf not in seen_scafs:
                seen_scafs.add(scaf)
                chosen_pool.append((smi, sc))
            if len(chosen_pool) >= max_pairs_per_target * 2:
                break

        rejected_pool = list(reversed(ranked))
        used_neg: set[str] = set()
        neg_ptr = 0
        t_pairs = 0
        for pos_smi, pos_sc in chosen_pool:
            while neg_ptr < len(rejected_pool) and (
                rejected_pool[neg_ptr][0] == pos_smi or rejected_pool[neg_ptr][0] in used_neg
            ):
                neg_ptr += 1
            if neg_ptr >= len(rejected_pool):
                break
            neg_smi, neg_sc, _ = rejected_pool[neg_ptr]
            if (pos_sc - neg_sc) >= min_margin:
                all_pairs.append(
                    CrossPocketPreferencePair(
                        pocket_batch=item["pocket_batch"],
                        chosen_smiles=pos_smi,
                        rejected_smiles=neg_smi,
                        chosen_score=pos_sc,
                        rejected_score=neg_sc,
                        off_target_pocket_emb=item["off_target_pocket_emb"],
                    )
                )
                used_neg.add(neg_smi)
                neg_ptr += 1
                t_pairs += 1
            if t_pairs >= max_pairs_per_target:
                break

    rng = np.random.default_rng(seed + 999)
    perm = rng.permutation(len(all_pairs)).tolist()
    return [all_pairs[i] for i in perm], all_groups


def _run_safegpt_scaffold_decoration_showcase(
    model: PocketConditionedLLM,
    primary_map: Mapping[str, dict[str, Any]],
    *,
    admet_suite: CNSAdmetSuite,
    qsar_payload: Mapping[str, Any],
    qsar_fp_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    strict_novelty_set: set[str],
    device: torch.device,
    num_samples: int = 48,
    seed: int = 42,
) -> dict[str, Any]:
    """Run 3D-conditioned SAFE-GPT R-group scaffold decoration on 4 flagship clinical cores."""
    t0 = time.perf_counter()
    print(
        "\n[train:SAFE-GPT] Running 3D-Conditioned Scaffold Decoration on 4 clinical cores...",
        flush=True,
    )
    model.eval()
    showcase: dict[str, Any] = {}

    for idx, task in enumerate(SCAFFOLD_DECORATION_TASKS):
        slug = task["slug"]
        tid = task["target_id"]
        p_entry = primary_map[tid]
        p_dev, mask_dev = _pocket_entry_to_device(p_entry, device)

        # Score parent clinical drug
        with torch.no_grad():
            p_fp = _smiles_to_fp_tensor([task["parent_smiles"]]).to(device)
            p_sc = model.score_candidates(p_dev, p_fp)
            p_pkd = [float(p_sc["pkd"][0].item())] if "pkd" in p_sc else None
            p_clip = [float(p_sc["clip_sim"][0].item())] if "clip_sim" in p_sc else None
        parent_rews, parent_q = _score_candidates_v4(
            [task["parent_smiles"]],
            tid,
            pred_pkd=p_pkd,
            clip_sim=p_clip,
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
        )
        pr = parent_rews[0]
        p_probs = pr.admet_v4.probabilities if pr.admet_v4 is not None else {}

        # Run 3D-conditioned decorate_scaffold (attachment_start=5 avoids ring-digit collisions!)
        dec_out = model.decorate_scaffold(
            task["core_with_dummy"],
            pocket_out=p_dev,
            atom_mask=mask_dev,
            num_samples=num_samples,
            max_new_tokens=56,
            strategy="min_p",
            temperature=0.82,
            min_p=0.08,
            cfg_scale=1.2,
            repetition_penalty=1.03,
            attachment_start=5,
            seed=seed + 8800 + idx * 100,
        )
        valid_smis: list[str] = []
        seen_s: set[str] = set()
        parent_can = _clean_isomeric_smiles(task["parent_smiles"])
        core_mol = Chem.MolFromSmiles(task["core_with_dummy"].replace("[*]", "[H]"))
        for row in dec_out:
            raw_txt = str(row.get("raw_text") or "")
            smi = _decode_generated_molecule(raw_txt, format_mode="safe")
            if smi is None:
                continue
            if smi not in seen_s:
                seen_s.add(smi)
                valid_smis.append(smi)

        n_valid_total = sum(
            1
            for row in dec_out
            if _decode_generated_molecule(str(row.get("raw_text") or ""), format_mode="safe")
            is not None
        )
        analogs: list[dict[str, Any]] = []
        if valid_smis:
            with torch.no_grad():
                fps_a = _smiles_to_fp_tensor(valid_smis).to(device)
                sc_a = model.score_candidates(p_dev, fps_a)
                pkd_a = sc_a["pkd"].detach().cpu().numpy().tolist() if "pkd" in sc_a else None
                clip_a = (
                    sc_a["clip_sim"].detach().cpu().numpy().tolist() if "clip_sim" in sc_a else None
                )
            rews_a, q_a = _score_candidates_v4(
                valid_smis,
                tid,
                pred_pkd=pkd_a,
                clip_sim=clip_a,
                admet_suite=admet_suite,
                qsar_payload=qsar_payload,
                qsar_fp_cache=qsar_fp_cache,
            )
            for i, smi in enumerate(valid_smis):
                r = rews_a[i]
                if not r.valid or r.admet_v4 is None:
                    continue
                probs = r.admet_v4.probabilities
                m_obj = Chem.MolFromSmiles(smi)
                core_preserved = bool(
                    m_obj is not None
                    and core_mol is not None
                    and m_obj.HasSubstructMatch(Chem.RemoveHs(core_mol))
                )
                analogs.append(
                    {
                        "smiles": smi,
                        "is_novel": smi not in strict_novelty_set and smi != parent_can,
                        "core_preserved": core_preserved,
                        "total_reward": round(float(r.total_reward), 4),
                        "cns_mpo_v2": round(float(r.mpo.cns_mpo_v2), 3),
                        "mw": round(float(r.mpo.mw), 2),
                        "clogp": round(float(r.mpo.clogp), 2),
                        "tpsa": round(float(r.mpo.tpsa), 2),
                        "sa_score": round(float(r.mpo.sa_score), 2),
                        "triage_pass": bool(r.triage.triage_pass),
                        "v4_pharma_safe": _v4_pharma_12ep_safe_flag(r),
                        "p_on_target": round(float(q_a["p_on_target"][i]), 4),
                        "p_off_max": round(float(q_a["p_off_max"][i]), 4),
                        "bbb_prob": round(float(probs.get("bbb", float("nan"))), 4),
                        "herg_prob": round(float(probs.get("herg", float("nan"))), 4),
                        "mdr1_efflux_prob": round(float(probs.get("mdr1_efflux", float("nan"))), 4),
                        "hlm_clearance_prob": round(
                            float(probs.get("hlm_clearance", float("nan"))), 4
                        ),
                        "chrm3_avoid_prob": round(float(probs.get("chrm3_avoid", float("nan"))), 4),
                        "htr2b_avoid_prob": round(float(probs.get("htr2b_avoid", float("nan"))), 4),
                        "pred_pkd": round(float(r.pred_pkd), 3) if r.pred_pkd is not None else None,
                    }
                )

            # Rank prioritizing core-preserved, novel, triage-passing, low-liability leads
            def _rank_key(a: dict[str, Any], *, task_slug: str = slug) -> tuple[int, int, float]:
                bonus = 0.0
                if task_slug == "donepezil_ache":
                    bonus = 0.6 * (1.0 - a["herg_prob"]) + 0.4 * a["p_on_target"]
                elif task_slug == "xanomeline_chrm4":
                    bonus = 0.6 * (1.0 - a["chrm3_avoid_prob"]) + 0.4 * a["p_on_target"]
                else:
                    bonus = float(a["total_reward"])
                return (
                    int(a["core_preserved"] and a["triage_pass"] and a["is_novel"]),
                    int(a["v4_pharma_safe"]),
                    bonus + 0.5 * float(a["total_reward"]),
                )

            analogs.sort(key=_rank_key, reverse=True)

        parent_profile = {
            "name": task["drug_name"],
            "smiles": task["parent_smiles"],
            "cns_mpo_v2": round(float(pr.mpo.cns_mpo_v2), 3),
            "p_on_target": round(float(parent_q["p_on_target"][0]), 4),
            "bbb_prob": round(float(p_probs.get("bbb", float("nan"))), 4),
            "herg_prob": round(float(p_probs.get("herg", float("nan"))), 4),
            "mdr1_efflux_prob": round(float(p_probs.get("mdr1_efflux", float("nan"))), 4),
            "chrm3_avoid_prob": round(float(p_probs.get("chrm3_avoid", float("nan"))), 4),
            "triage_pass": bool(pr.triage.triage_pass),
            "v4_pharma_safe": _v4_pharma_12ep_safe_flag(pr),
            "total_reward": round(float(pr.total_reward), 4),
        }
        best_a = analogs[0] if analogs else None
        showcase[slug] = {
            "drug_name": task["drug_name"],
            "target_id": tid,
            "pdb_id": task["pdb_id"],
            "counter_target": task["counter_target"],
            "objective": task["objective"],
            "core_with_dummy": task["core_with_dummy"],
            "prefix_used": dec_out[0]["prefix"] if dec_out else "",
            "n_samples": num_samples,
            "n_valid": n_valid_total,
            "validity": round(n_valid_total / float(max(1, num_samples)), 4),
            "n_unique_valid": len(analogs),
            "n_core_preserved": sum(1 for a in analogs if a["core_preserved"]),
            "n_v4_pharma_safe": sum(1 for a in analogs if a["v4_pharma_safe"]),
            "parent_profile": parent_profile,
            "top_analogs": analogs[:8],
        }
        print(
            f"  [Decorate {task['drug_name']:<10s}->{tid:<5s}] valid={n_valid_total}/{num_samples} "
            f"uniq={len(analogs)} core_ok={showcase[slug]['n_core_preserved']} "
            f"safe={showcase[slug]['n_v4_pharma_safe']} | "
            f"parent hERG={parent_profile['herg_prob']:.2f} "
            f"M3={parent_profile['chrm3_avoid_prob']:.2f} "
            + (
                f"-> best hERG={best_a['herg_prob']:.2f} M3={best_a['chrm3_avoid_prob']:.2f} "
                f"MPO={best_a['cns_mpo_v2']:.2f} P(on)={best_a['p_on_target']:.2f}"
                if best_a
                else "-> none"
            ),
            flush=True,
        )

    showcase["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    _write_json(RUN_ROOT / "safegpt_scaffold_decoration.json", showcase)
    return showcase


def run_stage_train(
    *, seed: int = 42, arm: str = "all", align_only: bool = False
) -> dict[str, Any]:
    """Execute Stage ``train``: Train & align local v4.0 LLMs on Apple MPS GPU."""
    t0 = time.perf_counter()
    seed_everything(seed)
    device = select_device("auto")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"[train] Starting GenEuclidean v4.0 Training on {device} "
        f"(arm={arm}, align_only={align_only})...",
        flush=True,
    )

    full_splits = json.loads(FULL_SPLITS_LOCAL_PATH.read_text(encoding="utf-8"))["splits"]
    admet_suite = _load_v4_admet_suite(n_jobs=4)
    qsar_payload = _get_joblib().load(V4_QSAR_MODEL_PATH)
    qsar_fp_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    strict_nov_all, _ = _build_strict_novelty_reference_sets()

    # Pre-encode frozen 3D pocket bank & PDBbind replay cache ONCE (bit-identical PaiNN-SE3!)
    ref_model, _ = _build_backbone_model(
        MOLFORMER_REPO_ID, device=device, adapter_ckpt=V3_SFT_ADAPTER_CKPT
    )
    train_bank, primary_map = _precompute_all_v4_pocket_embeddings(
        ref_model, device, n_jitters=2, jitter_std=0.05, seed=seed
    )
    replay_cache: list[CachedPocketRecord] = []
    if not align_only:
        pdb_payload = torch.load(PDBBIND_CACHE_PATH, map_location="cpu", weights_only=False)
        pdb_records = pdb_payload["records"]
        pdb_splits = json.loads(PDBBIND_SPLITS_PATH.read_text(encoding="utf-8"))
        pdb_train_ids = [pid for pid in pdb_splits["casf2016"]["train"] if pid in pdb_records]
        rng_pdb = np.random.default_rng(seed + 77)
        replay_ids = [
            pdb_train_ids[int(i)]
            for i in rng_pdb.choice(
                len(pdb_train_ids), size=min(1500, len(pdb_train_ids)), replace=False
            )
        ]
        replay_batches = build_pocket_batches_from_records(pdb_records, replay_ids, batch_size=32)
        replay_cache = precompute_pocket_cache(ref_model, replay_batches, device=device)
        print(
            f"[train] Pre-encoded {len(replay_cache)} PDBbind refined replay pockets.",
            flush=True,
        )
    del ref_model
    _clear_accel(device)

    summary_path = RUN_ROOT / "train_all_summary.json"
    train_reports: dict[str, Any] = {}
    if arm != "all" and summary_path.is_file():
        train_reports = json.loads(summary_path.read_text(encoding="utf-8"))

    # -----------------------------------------------------------------------
    # Arm 2: v4.0 GP-MoLFormer (Warm-start v3.0 -> 24-Target SFT -> Mult + L_XP-DPO)
    # -----------------------------------------------------------------------
    if arm in ("all", "v4_gpmolformer"):
        if align_only:
            sft_ckpt = RUN_ROOT / "v4_gpmolformer_sft" / "best_adapter.pt"
            m_gpm, cfg_gpm = _build_backbone_model(
                MOLFORMER_REPO_ID, device=device, adapter_ckpt=sft_ckpt
            )
            sft_gpm = json.loads(
                (RUN_ROOT / "v4_gpmolformer_sft" / "train_summary.json").read_text(encoding="utf-8")
            )
        else:
            m_gpm, cfg_gpm, sft_gpm = _run_sft_for_backbone(
                "v4_gpmolformer",
                MOLFORMER_REPO_ID,
                init_adapter_ckpt=V3_SFT_ADAPTER_CKPT,
                phase_a_epochs=0,
                phase_b_epochs=4,
                projector_lr=3e-4,
                bridge_lr=1.5e-4,
                lora_lr=6e-5,
                batch_size=32,
                randomize_safe_order=False,
                train_bank=train_bank,
                primary_map=primary_map,
                splits_full=full_splits,
                replay_cache=replay_cache,
                device=device,
                seed=seed,
            )
        snap_gpm = _snapshot_param_checksums(
            m_gpm,
            (
                "encoder.",
                "cvae.",
                "heads.",
                "perceiver.",
                "cvae_proj.",
                "global_proj.",
                "prefix_norm.",
                "null_prefix",
            ),
            include_base_llm=True,
        )
        roll_gpm = _collect_alignment_rollouts(
            m_gpm, primary_map, device=device, samples_per_target=96, seed=seed + 10
        )
        xp_pairs_gpm, _ = _build_xp_pairs_and_gdpo_groups(
            roll_gpm,
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
            max_pairs_per_target=24,
            min_margin=0.08,
            seed=seed + 10,
        )
        print(
            f"[train:Align:v4_gpmolformer] LoRA-only Mult + L_XP-DPO "
            f"({len(xp_pairs_gpm)} pairs, xp_w=0.30)...",
            flush=True,
        )
        align_gpm = run_llm_rpo_dpo(
            model=m_gpm,
            train_records=[],
            cfg=cfg_gpm,
            device=device,
            epochs=3,
            beta=0.45,
            lr=2.4e-5,
            sft_nll_weight=0.18,
            grad_accum_steps=4,
            seed=seed + 10,
            preference_pairs=xp_pairs_gpm,
            save_dir=RUN_ROOT / "v4_gpmolformer_aligned",
            cross_pocket_weight=0.30,
            cross_pocket_beta=0.45,
        )
        fa_gpm = _verify_frozen_bit_exact(m_gpm, snap_gpm)
        train_reports["v4_gpmolformer"] = {
            "sft": sft_gpm,
            "alignment": {**align_gpm, "method": "Mult + L_XP-DPO", "frozen_audit": fa_gpm},
        }
        last_gpm = align_gpm["history"][-1] if align_gpm["history"] else {}
        print(
            f"[train:Align:v4_gpmolformer] DONE | pairs={align_gpm['num_pairs']} "
            f"last={last_gpm} | bit_exact={fa_gpm['bit_exact']}",
            flush=True,
        )
        del m_gpm
        _clear_accel(device)

    # -----------------------------------------------------------------------
    # Arm 3: v4.0 SAFE-GPT-87M (bf16 canonical SAFE -> Phase A+B SFT -> 4-Ch GDPO + L_XP-DPO)
    # -----------------------------------------------------------------------
    if arm in ("all", "v4_safegpt"):
        if align_only:
            sft_ckpt = RUN_ROOT / "v4_safegpt_sft" / "best_adapter.pt"
            m_safe, cfg_safe = _build_backbone_model(
                SAFE_GPT_ALIAS, device=device, adapter_ckpt=sft_ckpt
            )
            sft_safe = json.loads(
                (RUN_ROOT / "v4_safegpt_sft" / "train_summary.json").read_text(encoding="utf-8")
            )
        else:
            m_safe, cfg_safe, sft_safe = _run_sft_for_backbone(
                "v4_safegpt",
                SAFE_GPT_ALIAS,
                init_adapter_ckpt=None,
                phase_a_epochs=2,
                phase_b_epochs=3,
                projector_lr=6e-4,
                bridge_lr=2.0e-4,
                lora_lr=8.0e-5,
                batch_size=32,
                randomize_safe_order=False,
                train_bank=train_bank,
                primary_map=primary_map,
                splits_full=full_splits,
                replay_cache=replay_cache,
                device=device,
                seed=seed + 100,
            )
        snap_safe = _snapshot_param_checksums(
            m_safe,
            (
                "encoder.",
                "cvae.",
                "heads.",
                "perceiver.",
                "cvae_proj.",
                "global_proj.",
                "prefix_norm.",
                "null_prefix",
            ),
            include_base_llm=True,
        )
        roll_safe = _collect_alignment_rollouts(
            m_safe,
            primary_map,
            device=device,
            samples_per_target=96,
            temperature=0.84,
            seed=seed + 110,
        )
        xp_pairs_safe, gdpo_groups_safe = _build_xp_pairs_and_gdpo_groups(
            roll_safe,
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
            max_pairs_per_target=24,
            min_margin=0.08,
            seed=seed + 110,
        )
        print(
            f"[train:Align:v4_safegpt] LoRA-only 4-Ch GDPO ({len(gdpo_groups_safe)} groups) "
            f"+ L_XP-DPO ({len(xp_pairs_safe)} pairs)...",
            flush=True,
        )
        gdpo_safe = run_llm_gdpo(
            model=m_safe,
            train_records=[],
            cfg=cfg_safe,
            device=device,
            epochs=2,
            lr=2.8e-5,
            clip_eps=0.20,
            kl_beta=0.04,
            rpo_nll_weight=0.20,
            channel_weights=(0.35, 0.30, 0.15, 0.20),
            grad_accum_steps=2,
            seed=seed + 111,
            gdpo_groups=gdpo_groups_safe,
            save_dir=RUN_ROOT / "v4_safegpt_gdpo_mid",
        )
        xp_safe = run_llm_rpo_dpo(
            model=m_safe,
            train_records=[],
            cfg=cfg_safe,
            device=device,
            epochs=3,
            beta=0.45,
            lr=2.4e-5,
            sft_nll_weight=0.18,
            grad_accum_steps=4,
            seed=seed + 112,
            preference_pairs=xp_pairs_safe,
            save_dir=RUN_ROOT / "v4_safegpt_aligned",
            cross_pocket_weight=0.30,
            cross_pocket_beta=0.45,
        )
        fa_safe = _verify_frozen_bit_exact(m_safe, snap_safe)
        dec_showcase = _run_safegpt_scaffold_decoration_showcase(
            m_safe,
            primary_map,
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
            strict_novelty_set=strict_nov_all,
            device=device,
            num_samples=64,
            seed=seed + 120,
        )
        train_reports["v4_safegpt"] = {
            "sft": sft_safe,
            "alignment": {
                "method": "4-Ch GDPO + L_XP-DPO",
                "gdpo": gdpo_safe,
                "xp_dpo": xp_safe,
                "frozen_audit": fa_safe,
            },
            "scaffold_decoration_summary": {
                k: {
                    "validity": v["validity"],
                    "n_core_preserved": v["n_core_preserved"],
                    "n_v4_pharma_safe": v["n_v4_pharma_safe"],
                }
                for k, v in dec_showcase.items()
                if isinstance(v, Mapping)
            },
        }
        del m_safe
        _clear_accel(device)

    # -----------------------------------------------------------------------
    # Arm 4: v4.0 NovoMolGen-157M (bf16 native MPS SDPA -> Phase A+B SFT -> 4-Ch GDPO + L_XP-DPO)
    # -----------------------------------------------------------------------
    if arm in ("all", "v4_novomolgen157m"):
        if align_only:
            sft_ckpt = RUN_ROOT / "v4_novomolgen157m_sft" / "best_adapter.pt"
            m_novo, cfg_novo = _build_backbone_model(
                NOVOMOLGEN_157M_ALIAS, device=device, adapter_ckpt=sft_ckpt
            )
            sft_novo = json.loads(
                (RUN_ROOT / "v4_novomolgen157m_sft" / "train_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        else:
            m_novo, cfg_novo, sft_novo = _run_sft_for_backbone(
                "v4_novomolgen157m",
                NOVOMOLGEN_157M_ALIAS,
                init_adapter_ckpt=None,
                phase_a_epochs=1,
                phase_b_epochs=3,
                projector_lr=8e-4,
                bridge_lr=2.0e-4,
                lora_lr=8e-5,
                batch_size=32,
                randomize_safe_order=False,
                train_bank=train_bank,
                primary_map=primary_map,
                splits_full=full_splits,
                replay_cache=replay_cache,
                device=device,
                seed=seed + 200,
            )
        snap_novo = _snapshot_param_checksums(
            m_novo,
            (
                "encoder.",
                "cvae.",
                "heads.",
                "perceiver.",
                "cvae_proj.",
                "global_proj.",
                "prefix_norm.",
                "null_prefix",
            ),
            include_base_llm=True,
        )
        roll_novo = _collect_alignment_rollouts(
            m_novo, primary_map, device=device, samples_per_target=96, seed=seed + 210
        )
        xp_pairs_novo, gdpo_groups_novo = _build_xp_pairs_and_gdpo_groups(
            roll_novo,
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
            max_pairs_per_target=24,
            min_margin=0.08,
            seed=seed + 210,
        )
        print(
            f"[train:Align:v4_novomolgen157m] LoRA-only 4-Ch GDPO ({len(gdpo_groups_novo)} groups) "
            f"+ L_XP-DPO ({len(xp_pairs_novo)} pairs)...",
            flush=True,
        )
        gdpo_novo = run_llm_gdpo(
            model=m_novo,
            train_records=[],
            cfg=cfg_novo,
            device=device,
            epochs=2,
            lr=2.5e-5,
            clip_eps=0.20,
            kl_beta=0.04,
            rpo_nll_weight=0.20,
            channel_weights=(0.35, 0.30, 0.15, 0.20),
            grad_accum_steps=2,
            seed=seed + 211,
            gdpo_groups=gdpo_groups_novo,
            save_dir=RUN_ROOT / "v4_novomolgen157m_gdpo_mid",
        )
        xp_novo = run_llm_rpo_dpo(
            model=m_novo,
            train_records=[],
            cfg=cfg_novo,
            device=device,
            epochs=3,
            beta=0.45,
            lr=2.2e-5,
            sft_nll_weight=0.18,
            grad_accum_steps=4,
            seed=seed + 212,
            preference_pairs=xp_pairs_novo,
            save_dir=RUN_ROOT / "v4_novomolgen157m_aligned",
            cross_pocket_weight=0.30,
            cross_pocket_beta=0.45,
        )
        fa_novo = _verify_frozen_bit_exact(m_novo, snap_novo)
        train_reports["v4_novomolgen157m"] = {
            "sft": sft_novo,
            "alignment": {
                "method": "4-Ch GDPO + L_XP-DPO",
                "gdpo": gdpo_novo,
                "xp_dpo": xp_novo,
                "frozen_audit": fa_novo,
            },
        }
        del m_novo
        _clear_accel(device)

    train_reports["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    _write_json(summary_path, train_reports)
    print(
        f"[train] Finished arm='{arm}' in {train_reports['elapsed_sec']:.1f}s!",
        flush=True,
    )
    return train_reports


# ---------------------------------------------------------------------------
# Stage 4: eval — Head-to-Head 24-Target Evaluation, Sink Audit, Stats & CASF
# ---------------------------------------------------------------------------


def _sample_all_24_targets(
    model: PocketConditionedLLM,
    primary_map: Mapping[str, dict[str, Any]],
    *,
    device: torch.device,
    mode: str = "matched",
    num_samples: int = EVAL_NUM_SAMPLES,
    seed: int = 42,
) -> dict[str, list[str]]:
    t0 = time.perf_counter()
    model.eval()
    out: dict[str, list[str]] = {}
    for t_idx, tid in enumerate(ALL_24_TARGETS):
        src_tid = SHUFFLED_DERANGEMENT_24[tid] if mode == "shuffled" else tid
        p_dev, mask_dev = _pocket_entry_to_device(primary_map[src_tid], device)
        raw_list = sample_llm_molecules(
            model,
            p_dev,
            mask_dev,
            num_samples=num_samples,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            strategy="min_p",
            temperature=EVAL_TEMPERATURE,
            min_p=EVAL_MIN_P,
            cfg_scale=EVAL_CFG_SCALE,
            latent_temp=1.0,
            repetition_penalty=EVAL_REPETITION_PENALTY,
            batch_size=min(64, num_samples),
            seed=seed + 1000 * t_idx,
        )
        out[tid] = raw_list
    print(
        f"  [sample:{model.format_mode}:{mode}] 24x{num_samples} "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def _run_casf2016_retention(
    model: PocketConditionedLLM,
    device: torch.device,
    *,
    n_gen_pockets: int = 20,
    num_samples: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    payload = torch.load(PDBBIND_CACHE_PATH, map_location="cpu", weights_only=False)
    records: dict[str, dict[str, Any]] = payload["records"]
    splits_data = json.loads(PDBBIND_SPLITS_PATH.read_text(encoding="utf-8"))
    core_ids = [pid for pid in splits_data["casf2016_core_ids"] if pid in records]

    pkd_true: list[float] = []
    pkd_pred: list[float] = []
    clip_p: list[np.ndarray] = []
    clip_l: list[np.ndarray] = []
    model.eval()

    for start in range(0, len(core_ids), 32):
        chunk = [records[pid] for pid in core_ids[start : start + 32]]
        bsz = len(chunk)
        max_a = max(int(np.asarray(r["coords"]).shape[0]) for r in chunk)
        node_d = int(np.asarray(chunk[0]["node_feats"]).shape[-1])
        coords = torch.zeros((bsz, max_a, 3), dtype=torch.float32)
        nf = torch.zeros((bsz, max_a, node_d), dtype=torch.float32)
        am = torch.zeros((bsz, max_a), dtype=torch.bool)
        lfps = torch.zeros((bsz, LIGAND_FP_DIM), dtype=torch.float32)
        for i, rec in enumerate(chunk):
            c = np.asarray(rec["coords"], dtype=np.float32)
            na = int(c.shape[0])
            coords[i, :na] = torch.from_numpy(c)
            nf[i, :na] = torch.from_numpy(np.asarray(rec["node_feats"], dtype=np.float32))
            am[i, :na] = True
            lfps[i] = torch.from_numpy(
                np.unpackbits(rec["ligand_fp_packed"])[:LIGAND_FP_DIM].astype(np.float32)
            )
        with torch.no_grad():
            p_out = model.encode_pocket(coords.to(device), nf.to(device), am.to(device))
            l_emb = model.heads.encode_ligand(lfps.to(device))
            mt = model.heads(p_out.pooled_emb, ligand_emb=l_emb)
        pkd_true.extend(float(r["pkd"]) for r in chunk)
        if mt.pred_pkd is not None:
            pkd_pred.extend(mt.pred_pkd.detach().cpu().numpy().tolist())
        if mt.clip_pocket is not None and mt.clip_ligand is not None:
            clip_p.append(mt.clip_pocket.detach().cpu().numpy())
            clip_l.append(mt.clip_ligand.detach().cpu().numpy())

    aff = evaluate_affinity_metrics(pkd_true, pkd_pred)
    clip = evaluate_clip_retrieval(np.concatenate(clip_p, axis=0), np.concatenate(clip_l, axis=0))

    step = max(1, len(core_ids) // max(1, n_gen_pockets))
    sample_ids = core_ids[::step][:n_gen_pockets]
    n_valid = 0
    n_total = 0
    for idx, pid in enumerate(sample_ids):
        rec = records[pid]
        c = torch.as_tensor(rec["coords"], dtype=torch.float32).unsqueeze(0).to(device)
        nf_t = torch.as_tensor(rec["node_feats"], dtype=torch.float32).unsqueeze(0).to(device)
        am = torch.ones((1, c.shape[1]), dtype=torch.bool, device=device)
        with torch.no_grad():
            p_out = model.encode_pocket(c, nf_t, am)
        raw_s = sample_llm_molecules(
            model,
            p_out,
            am,
            num_samples=num_samples,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            strategy="min_p",
            temperature=EVAL_TEMPERATURE,
            min_p=EVAL_MIN_P,
            cfg_scale=EVAL_CFG_SCALE,
            repetition_penalty=EVAL_REPETITION_PENALTY,
            batch_size=num_samples,
            seed=seed + 777 + idx,
        )
        n_total += len(raw_s)
        n_valid += sum(
            1
            for s in raw_s
            if _decode_generated_molecule(s, format_mode=model.format_mode) is not None
        )

    return {
        "n_core_complexes": len(core_ids),
        "pkd_pearson_r": round(float(aff["pearson_r"]), 4),
        "pkd_rmse": round(float(aff["rmse"]), 4),
        "drugclip_recall_at_5": round(float(clip["recall@5"]), 4),
        "drugclip_roc_auc": round(float(clip["roc_auc"]), 4),
        "gen_pockets_checked": len(sample_ids),
        "gen_samples_total": n_total,
        "gen_validity": round(n_valid / float(max(1, n_total)), 4),
    }


def _evaluate_arm_24_targets(
    arm_name: str,
    per_target_raw: Mapping[str, Sequence[str]],
    *,
    format_mode: str,
    primary_map: Mapping[str, dict[str, Any]],
    splits_full: Mapping[str, dict[str, Any]],
    scoring_model: PocketConditionedLLM,
    admet_suite: CNSAdmetSuite,
    qsar_payload: Mapping[str, Any],
    qsar_fp_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    strict_nov_all: set[str],
    nov_cp: set[str],
    device: torch.device,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Evaluate one arm across all 24 targets on v3.1, v4.0, MedChem, Sink & Conformal metrics."""
    per_target_summary: dict[str, dict[str, Any]] = {}
    per_target_records: dict[str, list[dict[str, Any]]] = {}

    for tid in ALL_24_TARGETS:
        raw_list = list(per_target_raw.get(tid, ()))
        decoded_smis: list[str] = []
        valid_indices: list[int] = []
        for i, raw_s in enumerate(raw_list):
            can = _decode_generated_molecule(raw_s, format_mode=format_mode)
            if can is not None:
                decoded_smis.append(can)
                valid_indices.append(i)

        p_entry = primary_map[tid]
        p_dev, _ = _pocket_entry_to_device(p_entry, device)
        if decoded_smis:
            with torch.no_grad():
                fps_t = _smiles_to_fp_tensor(decoded_smis).to(device)
                sc_3d = scoring_model.score_candidates(p_dev, fps_t)
                pkd_arr = sc_3d["pkd"].detach().cpu().numpy().tolist() if "pkd" in sc_3d else None
                clip_arr = (
                    sc_3d["clip_sim"].detach().cpu().numpy().tolist()
                    if "clip_sim" in sc_3d
                    else None
                )
            rews, q_res = _score_candidates_v4(
                decoded_smis,
                tid,
                pred_pkd=pkd_arr,
                clip_sim=clip_arr,
                admet_suite=admet_suite,
                qsar_payload=qsar_payload,
                qsar_fp_cache=qsar_fp_cache,
            )
        else:
            rews = []
            q_res = {}

        test_fps = [
            get_morgan_bit_fp(m)
            for s in splits_full[tid]["test"]
            if (m := parse_mol(s)) is not None
        ]
        train_fps = [
            get_morgan_bit_fp(m)
            for s in splits_full[tid]["train"][:400]
            if (m := parse_mol(s)) is not None
        ]
        nat_mol = parse_mol(p_entry["native_smiles"])
        nat_fp = get_morgan_bit_fp(nat_mol) if nat_mol is not None else None

        mol_rows: list[dict[str, Any]] = []
        valid_fps: list[Any] = []
        for i, smi in enumerate(decoded_smis):
            r = rews[i]
            if not r.valid or r.admet_v4 is None or not r.admet_v4.valid:
                continue
            mol = parse_mol(smi)
            if mol is None:
                continue
            fp = get_morgan_bit_fp(mol)
            valid_fps.append(fp)
            probs = r.admet_v4.probabilities
            ads = r.admet_v4.ad_tanimoto
            confs = r.admet_v4.conformal
            p_on = float(q_res["p_on_target"][i])
            p_off_max = float(q_res["p_off_max"][i])
            p_vmat2 = float(q_res["p_vmat2"][i])
            p_drd2 = float(q_res["p_drd2"][i])
            delta_sel = float(q_res["delta_sel"][i])
            vmat2_bleed = bool(tid != "VMAT2" and p_vmat2 >= 0.50 and p_on < 0.30)
            core_ad = min(
                ads.get(ep, 0.0)
                for ep in ("bbb", "herg", "mdr1_efflux", "hlm_clearance", "cyp2d6", "cyp3a4")
                if ep in ads
            )
            conf_singleton_rate = float(
                np.mean(
                    [
                        1.0 if (c is not None and not c.uncertain_flag) else 0.0
                        for c in confs.values()
                        if c is not None
                    ]
                )
            )
            max_sim_tr = (
                float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
                if train_fps
                else float("nan")
            )
            mol_rows.append(
                {
                    "canonical_smiles": smi,
                    "murcko_scaffold": get_murcko_scaffold(mol),
                    "generic_scaffold": r.triage.generic_murcko_scaffold,
                    "is_strict_novel": smi not in strict_nov_all,
                    "is_novel_vs_chembl_pdbbind": smi not in nov_cp,
                    "total_reward": float(r.total_reward),
                    "hygiene_gate": float(r.hygiene_gate),
                    "safety_gate": float(r.safety_gate),
                    "target_sel_gate": float(r.target_sel_gate),
                    "geometry_3d_gate": float(r.geometry_3d_gate),
                    "cns_mpo_v2": float(r.mpo.cns_mpo_v2),
                    "qed": float(r.mpo.qed),
                    "sa_score": float(r.mpo.sa_score),
                    "mw": float(r.mpo.mw),
                    "clogp": float(r.mpo.clogp),
                    "tpsa": float(r.mpo.tpsa),
                    "pains_free": bool(r.triage.pains_free),
                    "brenk_free": bool(r.triage.brenk_free),
                    "nih_free": bool(r.triage.nih_free),
                    "reagent_quarantine_free": bool(r.triage.reagent_quarantine_free),
                    "reagent_quarantined": not bool(r.triage.reagent_quarantine_free),
                    "lilly_demerit_score": int(r.triage.lilly_demerit_score),
                    "lilly_pass": bool(r.triage.lilly_pass),
                    "triage_pass": bool(r.triage.triage_pass),
                    "v31_7rule_safe": _v31_7rule_safe_flag(r),
                    "v4_pharma_12ep_safe": _v4_pharma_12ep_safe_flag(r),
                    "bbb_prob": float(probs.get("bbb", float("nan"))),
                    "pgp_prob": float(probs.get("pgp", float("nan"))),
                    "herg_prob": float(probs.get("herg", float("nan"))),
                    "mdr1_efflux_prob": float(probs.get("mdr1_efflux", float("nan"))),
                    "hlm_clearance_prob": float(probs.get("hlm_clearance", float("nan"))),
                    "cyp2d6_prob": float(probs.get("cyp2d6", float("nan"))),
                    "cyp3a4_prob": float(probs.get("cyp3a4", float("nan"))),
                    "htr2b_avoid_prob": float(probs.get("htr2b_avoid", float("nan"))),
                    "hrh1_avoid_prob": float(probs.get("hrh1_avoid", float("nan"))),
                    "chrm3_avoid_prob": float(probs.get("chrm3_avoid", float("nan"))),
                    "dili_prob": float(probs.get("dili", float("nan"))),
                    "ames_prob": float(probs.get("ames", float("nan"))),
                    "min_ad_core": float(core_ad),
                    "in_domain_core": bool(core_ad >= 0.30),
                    "conformal_singleton_rate": conf_singleton_rate,
                    "p_on_target": p_on,
                    "p_off_max": p_off_max,
                    "p_vmat2": p_vmat2,
                    "p_drd2": p_drd2,
                    "delta_sel": delta_sel,
                    "vmat2_bleed": vmat2_bleed,
                    "pred_pkd": float(r.pred_pkd) if r.pred_pkd is not None else float("nan"),
                    "clip_sim": float(r.clip_sim) if r.clip_sim is not None else float("nan"),
                    "native_tanimoto": tanimoto_similarity(fp, nat_fp)
                    if nat_fp is not None
                    else float("nan"),
                    "max_sim_to_train": max_sim_tr,
                }
            )
        per_target_records[tid] = mol_rows

        n_gen = len(raw_list)
        n_val = len(mol_rows)
        uniq_set = {m["canonical_smiles"] for m in mol_rows}
        n_uniq = len(uniq_set)
        uniq_scafs = {m["murcko_scaffold"] for m in mol_rows}

        def _m(k: str, rows_in: list[dict[str, Any]] = mol_rows) -> float:
            vs = [
                float(d[k]) for d in rows_in if d.get(k) is not None and math.isfinite(float(d[k]))
            ]
            return float(np.mean(vs)) if vs else float("nan")

        def _r(k: str, rows_in: list[dict[str, Any]] = mol_rows) -> float:
            if not rows_in:
                return 0.0
            return float(np.mean([1.0 if d[k] else 0.0 for d in rows_in]))

        per_target_summary[tid] = {
            "target_id": tid,
            "n_generated": n_gen,
            "n_valid": n_val,
            "n_unique": n_uniq,
            "validity": n_val / n_gen if n_gen > 0 else 0.0,
            "uniqueness": n_uniq / n_val if n_val > 0 else 0.0,
            "strict_novelty": (
                sum(1 for c in uniq_set if c not in strict_nov_all) / n_uniq if n_uniq > 0 else 0.0
            ),
            "novelty_vs_chembl_pdbbind": (
                sum(1 for c in uniq_set if c not in nov_cp) / n_uniq if n_uniq > 0 else 0.0
            ),
            "int_div_1": (
                compute_internal_diversity(valid_fps, p=1, seed=seed)
                if len(valid_fps) >= 2
                else 0.0
            ),
            "scaffold_diversity": len(uniq_scafs) / n_uniq if n_uniq > 0 else 0.0,
            "triage_pass_rate": _r("triage_pass"),
            "reagent_quarantine_rate": _r("reagent_quarantined"),
            "pains_free_rate": _r("pains_free"),
            "lilly_demerit_mean": _m("lilly_demerit_score"),
            "v31_7rule_safe_rate": _r("v31_7rule_safe"),
            "v4_pharma_12ep_safe_rate": _r("v4_pharma_12ep_safe"),
            "cns_mpo_v2_mean": _m("cns_mpo_v2"),
            "qed_mean": _m("qed"),
            "sa_mean": _m("sa_score"),
            "mw_mean": _m("mw"),
            "bbb_prob_mean": _m("bbb_prob"),
            "pgp_prob_mean": _m("pgp_prob"),
            "herg_prob_mean": _m("herg_prob"),
            "mdr1_efflux_prob_mean": _m("mdr1_efflux_prob"),
            "hlm_clearance_prob_mean": _m("hlm_clearance_prob"),
            "cyp2d6_prob_mean": _m("cyp2d6_prob"),
            "cyp3a4_prob_mean": _m("cyp3a4_prob"),
            "htr2b_avoid_prob_mean": _m("htr2b_avoid_prob"),
            "hrh1_avoid_prob_mean": _m("hrh1_avoid_prob"),
            "chrm3_avoid_prob_mean": _m("chrm3_avoid_prob"),
            "dili_prob_mean": _m("dili_prob"),
            "ames_prob_mean": _m("ames_prob"),
            "min_ad_core_mean": _m("min_ad_core"),
            "in_domain_core_rate": _r("in_domain_core"),
            "conformal_singleton_rate": _m("conformal_singleton_rate"),
            "total_reward_mean": _m("total_reward"),
            "p_on_target_mean": _m("p_on_target"),
            "p_off_max_mean": _m("p_off_max"),
            "p_vmat2_mean": _m("p_vmat2"),
            "p_drd2_mean": _m("p_drd2"),
            "delta_sel_mean": _m("delta_sel"),
            "vmat2_bleed_rate": _r("vmat2_bleed"),
            "pred_pkd_mean": _m("pred_pkd"),
            "clip_sim_mean": _m("clip_sim"),
            "test_scaffold_snn": compute_snn(valid_fps, test_fps)
            if (valid_fps and test_fps)
            else float("nan"),
            "native_tanimoto_mean": _m("native_tanimoto"),
            "max_sim_to_train_mean": _m("max_sim_to_train"),
        }

    def _agg_cohort(t_list: Sequence[str]) -> dict[str, Any]:
        rows = [per_target_summary[t] for t in t_list]
        mols = [m for t in t_list for m in per_target_records[t]]
        raw_n = sum(r["n_generated"] for r in rows)
        val_n = len(mols)
        uniq_sub = {m["canonical_smiles"] for m in mols}
        n_u = len(uniq_sub)
        fps_sub = [
            get_morgan_bit_fp(p)
            for m in mols
            if (p := parse_mol(m["canonical_smiles"])) is not None
        ]

        def _tm(k: str) -> float:
            vs = [float(r[k]) for r in rows if r.get(k) is not None and math.isfinite(float(r[k]))]
            return float(np.mean(vs)) if vs else float("nan")

        return {
            "n_targets": len(t_list),
            "n_generated": raw_n,
            "n_valid": val_n,
            "n_unique": n_u,
            "validity": val_n / raw_n if raw_n > 0 else 0.0,
            "uniqueness": n_u / val_n if val_n > 0 else 0.0,
            "strict_novelty": (
                sum(1 for c in uniq_sub if c not in strict_nov_all) / n_u if n_u > 0 else 0.0
            ),
            "global_int_div_1": (
                compute_internal_diversity(fps_sub, p=1, seed=seed) if len(fps_sub) >= 2 else 0.0
            ),
            "intra_int_div_1_mean": _tm("int_div_1"),
            "triage_pass_rate": _tm("triage_pass_rate"),
            "reagent_quarantine_rate": _tm("reagent_quarantine_rate"),
            "pains_free_rate": _tm("pains_free_rate"),
            "lilly_demerit_mean": _tm("lilly_demerit_mean"),
            "v31_7rule_safe_rate": _tm("v31_7rule_safe_rate"),
            "v4_pharma_12ep_safe_rate": _tm("v4_pharma_12ep_safe_rate"),
            "cns_mpo_v2_mean": _tm("cns_mpo_v2_mean"),
            "qed_mean": _tm("qed_mean"),
            "sa_mean": _tm("sa_mean"),
            "mw_mean": _tm("mw_mean"),
            "bbb_prob_mean": _tm("bbb_prob_mean"),
            "pgp_prob_mean": _tm("pgp_prob_mean"),
            "herg_prob_mean": _tm("herg_prob_mean"),
            "mdr1_efflux_prob_mean": _tm("mdr1_efflux_prob_mean"),
            "hlm_clearance_prob_mean": _tm("hlm_clearance_prob_mean"),
            "cyp2d6_prob_mean": _tm("cyp2d6_prob_mean"),
            "cyp3a4_prob_mean": _tm("cyp3a4_prob_mean"),
            "htr2b_avoid_prob_mean": _tm("htr2b_avoid_prob_mean"),
            "hrh1_avoid_prob_mean": _tm("hrh1_avoid_prob_mean"),
            "chrm3_avoid_prob_mean": _tm("chrm3_avoid_prob_mean"),
            "dili_prob_mean": _tm("dili_prob_mean"),
            "ames_prob_mean": _tm("ames_prob_mean"),
            "min_ad_core_mean": _tm("min_ad_core_mean"),
            "in_domain_core_rate": _tm("in_domain_core_rate"),
            "conformal_singleton_rate": _tm("conformal_singleton_rate"),
            "total_reward_mean": _tm("total_reward_mean"),
            "p_on_target_mean": _tm("p_on_target_mean"),
            "p_off_max_mean": _tm("p_off_max_mean"),
            "delta_sel_mean": _tm("delta_sel_mean"),
            "vmat2_bleed_rate": _tm("vmat2_bleed_rate"),
            "pred_pkd_mean": _tm("pred_pkd_mean"),
            "clip_sim_mean": _tm("clip_sim_mean"),
            "test_scaffold_snn_mean": _tm("test_scaffold_snn"),
        }

    # Explicit PDE10A & DRD3 sink audit
    sink_audit = {}
    for stid in ("PDE10A", "DRD3"):
        pt = per_target_summary[stid]
        sink_audit[stid] = {
            "p_on_target_mean": pt["p_on_target_mean"],
            "p_vmat2_mean": pt["p_vmat2_mean"],
            "vmat2_bleed_rate": pt["vmat2_bleed_rate"],
            "reagent_quarantine_rate": pt["reagent_quarantine_rate"],
            "v4_pharma_12ep_safe_rate": pt["v4_pharma_12ep_safe_rate"],
        }

    report = {
        "arm_name": arm_name,
        "overall_24": _agg_cohort(ALL_24_TARGETS),
        "v31_core_12": _agg_cohort(V31_12_TARGETS),
        "new_ts_scz_6": _agg_cohort(NEW_TS_SCZ_6_TARGETS),
        "new_ad_6": _agg_cohort(NEW_AD_6_TARGETS),
        "sink_elimination_audit": sink_audit,
        "per_target": per_target_summary,
    }
    return report, per_target_records


def _compare_two_arms_stats(
    arm_a_recs: Mapping[str, list[dict[str, Any]]],
    arm_b_recs: Mapping[str, list[dict[str, Any]]],
    arm_a_rep: Mapping[str, Any],
    arm_b_rep: Mapping[str, Any],
    *,
    targets: Sequence[str] = ALL_24_TARGETS,
    seed: int = 42,
) -> dict[str, Any]:
    metrics = [
        ("v4_pharma_12ep_safe", "v4_pharma_12ep_safe_rate"),
        ("v31_7rule_safe", "v31_7rule_safe_rate"),
        ("triage_pass", "triage_pass_rate"),
        ("reagent_quarantined", "reagent_quarantine_rate"),
        ("cns_mpo_v2", "cns_mpo_v2_mean"),
        ("herg_prob", "herg_prob_mean"),
        ("mdr1_efflux_prob", "mdr1_efflux_prob_mean"),
        ("p_on_target", "p_on_target_mean"),
        ("delta_sel", "delta_sel_mean"),
        ("total_reward", "total_reward_mean"),
    ]
    out: dict[str, Any] = {}
    for mol_k, tgt_k in metrics:
        a_t = np.asarray([float(arm_a_rep["per_target"][t][tgt_k]) for t in targets], dtype=float)
        b_t = np.asarray([float(arm_b_rep["per_target"][t][tgt_k]) for t in targets], dtype=float)
        diffs = a_t - b_t
        if len(diffs) >= 2 and not np.allclose(diffs, 0.0):
            w_p = float(stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox").pvalue)
        else:
            w_p = 1.0

        a_m = np.asarray(
            [
                float(d[mol_k])
                for t in targets
                for d in arm_a_recs[t]
                if math.isfinite(float(d[mol_k]))
            ],
            dtype=float,
        )
        b_m = np.asarray(
            [
                float(d[mol_k])
                for t in targets
                for d in arm_b_recs[t]
                if math.isfinite(float(d[mol_k]))
            ],
            dtype=float,
        )
        mwu_p = (
            float(stats.mannwhitneyu(a_m, b_m, alternative="two-sided").pvalue)
            if len(a_m) > 0 and len(b_m) > 0
            else float("nan")
        )
        raw_ps: list[float] = []
        for t in targets:
            at = np.asarray([float(d[mol_k]) for d in arm_a_recs[t]], dtype=float)
            bt = np.asarray([float(d[mol_k]) for d in arm_b_recs[t]], dtype=float)
            if len(at) > 0 and len(bt) > 0:
                raw_ps.append(float(stats.mannwhitneyu(at, bt, alternative="two-sided").pvalue))
            else:
                raw_ps.append(1.0)
        holm_ps = _holm_bonferroni(raw_ps)
        out[mol_k] = {
            "arm_a_mean": float(np.mean(a_t)),
            "arm_b_mean": float(np.mean(b_t)),
            "diff_mean": float(np.mean(diffs)),
            "diff_ci95": bootstrap_ci(diffs, n_bootstrap=1000, seed=seed),
            "wilcoxon_p": w_p,
            "mannwhitneyu_p": mwu_p,
            "cliffs_delta_mol": cliffs_delta(a_m, b_m)
            if len(a_m) > 0 and len(b_m) > 0
            else float("nan"),
            "cliffs_delta_target": cliffs_delta(a_t, b_t),
            "holm_sig_targets_p05": sum(1 for p in holm_ps if p < 0.05),
        }

    # Test-scaffold SNN paired Wilcoxon
    a_snn = np.asarray(
        [float(arm_a_rep["per_target"][t]["test_scaffold_snn"]) for t in targets], dtype=float
    )
    b_snn = np.asarray(
        [float(arm_b_rep["per_target"][t]["test_scaffold_snn"]) for t in targets], dtype=float
    )
    d_snn = a_snn - b_snn
    snn_p = (
        float(stats.wilcoxon(d_snn, alternative="two-sided").pvalue)
        if len(d_snn) >= 2 and not np.allclose(d_snn, 0.0)
        else 1.0
    )
    out["test_scaffold_snn"] = {
        "arm_a_mean": float(np.mean(a_snn)),
        "arm_b_mean": float(np.mean(b_snn)),
        "diff_mean": float(np.mean(d_snn)),
        "diff_ci95": bootstrap_ci(d_snn, n_bootstrap=1000, seed=seed),
        "wilcoxon_p": snn_p,
        "cliffs_delta_target": cliffs_delta(a_snn, b_snn),
    }
    return out


def _select_top_k_diverse(
    mol_rows: Sequence[dict[str, Any]], *, k: int = 5, prefer_safe: bool = True
) -> list[dict[str, Any]]:
    if not mol_rows:
        return []
    # Deduplicate canonical SMILES keeping highest reward
    best_by_smi: dict[str, dict[str, Any]] = {}
    for row in mol_rows:
        smi = str(row["canonical_smiles"])
        if smi not in best_by_smi or float(row["total_reward"]) > float(
            best_by_smi[smi]["total_reward"]
        ):
            best_by_smi[smi] = row
    pool = list(best_by_smi.values())
    smis = [str(r["canonical_smiles"]) for r in pool]
    scores = [
        float(r["total_reward"])
        + (0.15 if (prefer_safe and r.get("v4_pharma_12ep_safe")) else 0.0)
        + (0.05 if r.get("triage_pass") else -0.20)
        for r in pool
    ]
    idxs = maxmin_diversity_select(
        smis,
        scores=scores,
        k=min(k, len(pool)),
        min_tanimoto_dist=0.35,
        quality_weight=0.70,
        relax_if_needed=True,
    )
    out: list[dict[str, Any]] = []
    for rank, i in enumerate(idxs, start=1):
        rec = dict(pool[i])
        rec["rank"] = rank
        out.append(rec)
    return out


def _print_v4_headline(label: str, rep: Mapping[str, Any]) -> None:
    o = rep["overall_24"]
    p_sink = rep["sink_elimination_audit"]["PDE10A"]
    d_sink = rep["sink_elimination_audit"]["DRD3"]
    print(
        f"  [{label:<20s}] Val={o['validity'] * 100:5.1f}% Nov={o['strict_novelty'] * 100:5.1f}% "
        f"Div={o['global_int_div_1']:.3f} | Triage={o['triage_pass_rate'] * 100:5.1f}% "
        f"Reag={o['reagent_quarantine_rate'] * 100:4.1f}% | "
        f"v3.1Safe={o['v31_7rule_safe_rate'] * 100:5.1f}% "
        f"v4Safe={o['v4_pharma_12ep_safe_rate'] * 100:5.1f}% MPO={o['cns_mpo_v2_mean']:.2f} "
        f"hERG={o['herg_prob_mean']:.3f} MDR1={o['mdr1_efflux_prob_mean']:.3f} | "
        f"P(on)={o['p_on_target_mean']:.3f} | "
        f"PDE10A(on={p_sink['p_on_target_mean']:.2f},"
        f"bleed={p_sink['vmat2_bleed_rate'] * 100:.0f}%) "
        f"DRD3(on={d_sink['p_on_target_mean']:.2f},"
        f"bleed={d_sink['vmat2_bleed_rate'] * 100:.0f}%)",
        flush=True,
    )


def run_stage_eval(*, seed: int = 42, num_samples: int = EVAL_NUM_SAMPLES) -> dict[str, Any]:
    """Execute Stage ``eval``: Head-to-Head 24-Target Evaluation across all arms + controls."""
    t0 = time.perf_counter()
    seed_everything(seed)
    device = select_device("auto")
    print(
        f"[eval] Running 24-Target Head-to-Head Evaluation on {device} (N={num_samples}/target)...",
        flush=True,
    )

    full_splits = json.loads(FULL_SPLITS_LOCAL_PATH.read_text(encoding="utf-8"))["splits"]
    admet_suite = _load_v4_admet_suite(n_jobs=4)
    qsar_payload = _get_joblib().load(V4_QSAR_MODEL_PATH)
    qsar_fp_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    strict_nov_all, nov_cp = _build_strict_novelty_reference_sets()

    # Load frozen 3D scoring model (v3.1 GP-MoLFormer carries identical frozen PaiNN-SE3 + heads!)
    m_v31, _ = _build_backbone_model(
        MOLFORMER_REPO_ID, device=device, adapter_ckpt=V31_DPO_ADAPTER_CKPT
    )
    _, primary_map = _precompute_all_v4_pocket_embeddings(m_v31, device, n_jitters=0, seed=seed)

    arms_config: list[tuple[str, str, Path, int]] = [
        ("v3.1_GP-MoLFormer", MOLFORMER_REPO_ID, V31_DPO_ADAPTER_CKPT, seed + 1000),
        (
            "v4.0_GP-MoLFormer",
            MOLFORMER_REPO_ID,
            RUN_ROOT / "v4_gpmolformer_aligned" / "best_adapter.pt",
            seed + 2000,
        ),
        (
            "v4.0_SAFE-GPT-87M",
            SAFE_GPT_ALIAS,
            RUN_ROOT / "v4_safegpt_aligned" / "best_adapter.pt",
            seed + 3000,
        ),
        (
            "v4.0_NovoMolGen-157M",
            NOVOMOLGEN_157M_ALIAS,
            RUN_ROOT / "v4_novomolgen157m_aligned" / "best_adapter.pt",
            seed + 4000,
        ),
    ]

    arm_reports: dict[str, dict[str, Any]] = {}
    arm_mol_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    arm_raw_samples: dict[str, dict[str, list[str]]] = {}
    casf_retention: dict[str, dict[str, Any]] = {}

    # Sample & evaluate v3.1 first (already loaded)
    raw_v31 = _sample_all_24_targets(
        m_v31, primary_map, device=device, mode="matched", num_samples=num_samples, seed=seed + 1000
    )
    arm_raw_samples["v3.1_GP-MoLFormer"] = raw_v31
    rep_v31, recs_v31 = _evaluate_arm_24_targets(
        "v3.1_GP-MoLFormer",
        raw_v31,
        format_mode="smiles",
        primary_map=primary_map,
        splits_full=full_splits,
        scoring_model=m_v31,
        admet_suite=admet_suite,
        qsar_payload=qsar_payload,
        qsar_fp_cache=qsar_fp_cache,
        strict_nov_all=strict_nov_all,
        nov_cp=nov_cp,
        device=device,
        seed=seed,
    )
    arm_reports["v3.1_GP-MoLFormer"] = rep_v31
    arm_mol_records["v3.1_GP-MoLFormer"] = recs_v31
    _print_v4_headline("v3.1 GP-MoLFormer", rep_v31)

    # Ctrl-ZINC random control (scored with frozen PaiNN-SE3 heads)
    all_chembl_set = {s for tid in ALL_24_TARGETS for s in full_splits[tid]["pchembl_map"]}
    zinc_clean = _load_zinc250k_smiles(exclude_set=all_chembl_set)
    rng_z = np.random.default_rng(seed + 55)
    pick = rng_z.choice(len(zinc_clean), size=num_samples * len(ALL_24_TARGETS), replace=False)
    ctrl_zinc_raw = {
        tid: [zinc_clean[int(i)] for i in pick[idx * num_samples : (idx + 1) * num_samples]]
        for idx, tid in enumerate(ALL_24_TARGETS)
    }
    rep_zinc, recs_zinc = _evaluate_arm_24_targets(
        "Ctrl-ZINC",
        ctrl_zinc_raw,
        format_mode="smiles",
        primary_map=primary_map,
        splits_full=full_splits,
        scoring_model=m_v31,
        admet_suite=admet_suite,
        qsar_payload=qsar_payload,
        qsar_fp_cache=qsar_fp_cache,
        strict_nov_all=strict_nov_all,
        nov_cp=nov_cp,
        device=device,
        seed=seed,
    )
    arm_reports["Ctrl-ZINC"] = rep_zinc
    arm_mol_records["Ctrl-ZINC"] = recs_zinc
    _print_v4_headline("Ctrl-ZINC", rep_zinc)

    del m_v31
    _clear_accel(device)

    # Sample & evaluate each v4.0 backbone + CASF-2016 retention
    for arm_name, b_id, ckpt_path, a_seed in arms_config[1:]:
        m_cur, _ = _build_backbone_model(b_id, device=device, adapter_ckpt=ckpt_path)
        raw_cur = _sample_all_24_targets(
            m_cur, primary_map, device=device, mode="matched", num_samples=num_samples, seed=a_seed
        )
        arm_raw_samples[arm_name] = raw_cur
        rep_cur, recs_cur = _evaluate_arm_24_targets(
            arm_name,
            raw_cur,
            format_mode=m_cur.format_mode,
            primary_map=primary_map,
            splits_full=full_splits,
            scoring_model=m_cur,
            admet_suite=admet_suite,
            qsar_payload=qsar_payload,
            qsar_fp_cache=qsar_fp_cache,
            strict_nov_all=strict_nov_all,
            nov_cp=nov_cp,
            device=device,
            seed=seed,
        )
        arm_reports[arm_name] = rep_cur
        arm_mol_records[arm_name] = recs_cur
        _print_v4_headline(arm_name, rep_cur)

        casf_cur = _run_casf2016_retention(
            m_cur, device, n_gen_pockets=20, num_samples=16, seed=seed
        )
        casf_retention[arm_name] = casf_cur
        print(
            f"  [CASF-2016 {arm_name}] pKd r={casf_cur['pkd_pearson_r']:.4f} | "
            f"DrugCLIP R@5={casf_cur['drugclip_recall_at_5'] * 100:.1f}% | "
            f"GenValid={casf_cur['gen_validity'] * 100:.1f}%",
            flush=True,
        )
        del m_cur
        _clear_accel(device)

    # Select winning v4.0 backbone by composite Pareto score (v4_Safe + Reward + P(on) + Validity)
    v4_arm_names = ("v4.0_GP-MoLFormer", "v4.0_SAFE-GPT-87M", "v4.0_NovoMolGen-157M")

    def _winner_score(name: str) -> float:
        o = arm_reports[name]["overall_24"]
        return (
            0.35 * float(o["v4_pharma_12ep_safe_rate"])
            + 0.25 * float(o["total_reward_mean"])
            + 0.20 * float(o["p_on_target_mean"])
            + 0.15 * float(o["validity"])
            + 0.05 * float(o["triage_pass_rate"])
            - 0.50 * float(o["reagent_quarantine_rate"])
        )

    best_v4_arm = max(v4_arm_names, key=_winner_score)
    best_cfg = next(x for x in arms_config if x[0] == best_v4_arm)
    print(
        f"\n[eval] Winning v4.0 backbone: {best_v4_arm} "
        "(running Shuffled-Pocket 3D specificity control)...",
        flush=True,
    )

    m_best, _ = _build_backbone_model(best_cfg[1], device=device, adapter_ckpt=best_cfg[2])
    raw_shuf = _sample_all_24_targets(
        m_best,
        primary_map,
        device=device,
        mode="shuffled",
        num_samples=num_samples,
        seed=seed + 9000,
    )
    rep_shuf, recs_shuf = _evaluate_arm_24_targets(
        f"{best_v4_arm}_Shuffled",
        raw_shuf,
        format_mode=m_best.format_mode,
        primary_map=primary_map,
        splits_full=full_splits,
        scoring_model=m_best,
        admet_suite=admet_suite,
        qsar_payload=qsar_payload,
        qsar_fp_cache=qsar_fp_cache,
        strict_nov_all=strict_nov_all,
        nov_cp=nov_cp,
        device=device,
        seed=seed,
    )
    arm_reports["v4.0_Best_Shuffled"] = rep_shuf
    arm_mol_records["v4.0_Best_Shuffled"] = recs_shuf
    _print_v4_headline("v4.0 Best Shuffled", rep_shuf)
    del m_best
    _clear_accel(device)

    # Full Statistical Testing (each v4.0 arm vs v3.1, vs Ctrl-ZINC, plus Best Matched vs Shuffled)
    statistics: dict[str, Any] = {}
    for arm_name in v4_arm_names:
        statistics[f"{arm_name}_vs_v3.1"] = _compare_two_arms_stats(
            arm_mol_records[arm_name],
            arm_mol_records["v3.1_GP-MoLFormer"],
            arm_reports[arm_name],
            arm_reports["v3.1_GP-MoLFormer"],
            targets=ALL_24_TARGETS,
            seed=seed,
        )
        statistics[f"{arm_name}_vs_Ctrl-ZINC"] = _compare_two_arms_stats(
            arm_mol_records[arm_name],
            arm_mol_records["Ctrl-ZINC"],
            arm_reports[arm_name],
            arm_reports["Ctrl-ZINC"],
            targets=ALL_24_TARGETS,
            seed=seed,
        )
    statistics["v4.0_Best_Matched_vs_Shuffled"] = _compare_two_arms_stats(
        arm_mol_records[best_v4_arm],
        arm_mol_records["v4.0_Best_Shuffled"],
        arm_reports[best_v4_arm],
        arm_reports["v4.0_Best_Shuffled"],
        targets=ALL_24_TARGETS,
        seed=seed,
    )

    # Export Top-5 MaxMin-diverse candidates per target for Stage dock
    docking_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm_name in ("v3.1_GP-MoLFormer", *v4_arm_names):
        docking_candidates[arm_name] = {
            tid: _select_top_k_diverse(arm_mol_records[arm_name][tid], k=5, prefer_safe=True)
            for tid in ALL_24_TARGETS
        }
    _write_json(
        RUN_ROOT / "v4_candidates_for_docking.json",
        {"best_v4_arm": best_v4_arm, "candidates": docking_candidates},
    )

    # Attach SAFE-GPT scaffold decoration results if present
    dec_path = RUN_ROOT / "safegpt_scaffold_decoration.json"
    dec_payload = json.loads(dec_path.read_text(encoding="utf-8")) if dec_path.is_file() else {}
    train_sum_path = RUN_ROOT / "train_all_summary.json"
    train_sum = (
        json.loads(train_sum_path.read_text(encoding="utf-8")) if train_sum_path.is_file() else {}
    )

    head_to_head = {
        "schema_version": "geneuclidean-v4.0-llm-head-to-head",
        "num_samples_per_target": num_samples,
        "total_molecules_per_arm": num_samples * len(ALL_24_TARGETS),
        "best_v4_arm": best_v4_arm,
        "v4_pharma_safe_criteria": (
            "MPO v2 >= 4.0, BBB >= 0.80, MDR1-MDCK < 0.45, HLM < 0.50, hERG <= 0.30, "
            "CYP2D6 <= 0.40, CYP3A4 <= 0.45, 5-HT2B <= 0.35, H1 <= 0.35, MedChem Triage PASS, "
            "min_AD_core >= 0.30"
        ),
        "arms": arm_reports,
        "statistics_all24": statistics,
        "casf2016_retention": casf_retention,
        "safegpt_scaffold_decoration": dec_payload,
        "frozen_3d_audits": {
            k: {
                "sft_bit_exact": v.get("sft", {}).get("frozen_audit", {}).get("bit_exact"),
                "align_bit_exact": v.get("alignment", {}).get("frozen_audit", {}).get("bit_exact"),
                "sft_max_abs_diff": v.get("sft", {}).get("frozen_audit", {}).get("max_abs_diff"),
                "align_max_abs_diff": v.get("alignment", {})
                .get("frozen_audit", {})
                .get("max_abs_diff"),
            }
            for k, v in train_sum.items()
            if isinstance(v, Mapping)
        },
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    _write_json(TABLE_HEAD_TO_HEAD_PATH, head_to_head)
    sz_kb = TABLE_HEAD_TO_HEAD_PATH.stat().st_size / 1024.0
    print(
        f"[eval] DONE in {head_to_head['elapsed_sec']:.1f}s | Best={best_v4_arm} | "
        f"Saved {TABLE_HEAD_TO_HEAD_PATH} ({sz_kb:.1f} KB)",
        flush=True,
    )
    return head_to_head


# ---------------------------------------------------------------------------
# Stage 5: dock — 24-Target Held-Out Vina + 5 Counter Panels + SDFs + Dossier
# ---------------------------------------------------------------------------


def _dock_job_key(pocket_key: str, smiles: str) -> str:
    h = hashlib.sha256(f"{pocket_key}||{smiles}".encode()).hexdigest()[:20]
    return f"{pocket_key}::{h}"


def _dock_worker_task(
    args: tuple[str, str, str, str, str, tuple[str, ...], tuple[str, ...], str | None, int],
) -> tuple[str, dict[str, Any]]:
    (
        job_key,
        pocket_key,
        smiles,
        pocket_pdb,
        lig_sdf,
        anchors,
        anchor_kinds,
        sdf_out_path,
        vina_cpu,
    ) = args
    rec_pdbqt = RECEPTOR_PDBQT_DIR / f"{pocket_key}.pdbqt"
    res = dock_ligand_to_pocket(
        smiles,
        pocket_pdb,
        native_ligand_sdf=lig_sdf,
        box_mode="tight_per_axis",
        canonical_anchors=anchors,
        expected_anchor_kinds=anchor_kinds,
        protonate=True,
        num_confs=24,
        max_ring_confs=2,
        vina_bin=VINA_BIN,
        exhaustiveness=8,
        num_modes=9,
        cpu=vina_cpu,
        seed=42,
        timeout=180,
        receptor_pdbqt_path=rec_pdbqt if rec_pdbqt.is_file() else None,
        require_vina=True,
    )
    saved_sdf: str | None = None
    if res.success and sdf_out_path is not None and res.preferred_mol is not None:
        try:
            saved_sdf = str(res.save_sdf(sdf_out_path, prefer_anchor_pose=True))
        except Exception:
            saved_sdf = None

    d_dict = res.to_dict()
    # Keep compact interactions summary for JSON budget (< 1.5 MB)
    if isinstance(d_dict.get("interactions"), Mapping):
        inter = d_dict["interactions"]
        d_dict["interactions_summary"] = {
            "total": inter.get("total_interactions"),
            "counts": inter.get("counts_by_kind"),
            "top_contacts": (inter.get("interactions") or [])[:6],
        }
        d_dict.pop("interactions", None)
    d_dict["pocket_key"] = pocket_key
    d_dict["saved_sdf"] = saved_sdf
    return job_key, d_dict


def _resolve_pocket_files_for_key(pocket_key: str) -> tuple[Path, Path]:
    if pocket_key == "DRD2_6LUQ":
        loc = find_local_pocket_files("DRD2", pdb_id="6LUQ")
    else:
        spec = get_cns_target(pocket_key)
        loc = find_local_pocket_files(pocket_key, pdb_id=spec.primary_pdb_id)
    if loc is None or loc[1] is None:
        raise FileNotFoundError(f"Missing pocket/ligand files for {pocket_key}")
    return loc[0], loc[1]


def _write_v4_summary_markdown(
    splits_sum: Mapping[str, Any],
    qsar_cv: Mapping[str, Any],
    gate1_rep: Mapping[str, Any],
    h2h_rep: Mapping[str, Any],
    dock_rep: Mapping[str, Any],
    dossier: Mapping[str, Any],
) -> None:
    """Write publication-ready GenEuclidean v4.0 Tri-Disorder Markdown report."""
    best_arm = str(h2h_rep["best_v4_arm"])
    arms = h2h_rep["arms"]
    g1 = gate1_rep["summary"]
    qsum = qsar_cv["summary"]
    best_dock = dock_rep["overall_24_summary"][best_arm]
    v31_dock = dock_rep["overall_24_summary"]["v3.1_GP-MoLFormer"]
    casf_b = h2h_rep["casf2016_retention"][best_arm]

    hdr_cols = (
        "| Arm | Valid | Uniq | Strict Nov | IntDiv1 | Triage Pass | DCU/Reag % "
        "| v3.1 7-Rule Safe | **v4.0 12-Ep Safe** | MPO v2 | P(hERG) | P(MDR1) "
        "| P(on_target) | PDE10A Bleed | DRD3 Bleed |"
    )
    hdr_sep = (
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: "
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    lines: list[str] = [
        "# GenEuclidean v4.0 Flagship Tri-Disorder & Head-to-Head Local LLM Benchmark Summary",
        "",
        (
            "> **Strictly *In Silico* Research Hypotheses:** All generated molecules, QSAR/ADMET "
            "predictions, and AutoDock Vina poses are computational research hypotheses requiring "
            "synthesis and wet-lab validation."
        ),
        "",
        "## 1. Gate 0 Zero-Leakage 24-Target Splits & 26-Endpoint Calibrated QSAR",
        (
            "- **Atlas Coverage:** `24` primary *Homo sapiens* targets (`12` v3.1 Tourette/SCZ + "
            "`6` new TS/SCZ + `6` new Alzheimer's) + `7` matched structural/Avoid-ome counters."
        ),
        (
            "- **Zero-Leakage Audit:** `0` generic Bemis-Murcko Train/Val/Test scaffold overlap "
            f"across all 24 targets; `0` clinical drug (`{splits_sum['protected_clinical_count']}`)"
            f" or native ligand (`{splits_sum['protected_native_count']}`) leakage into Train."
        ),
        (
            "- **26-Endpoint 5-Fold Scaffold-CV QSAR:** "
            f"Mean ROC-AUC = **`{qsum['mean_roc_auc_all_26']:.4f}`** "
            f"(min `{qsum['min_roc_auc_all_26']:.4f}`; 14 new endpoints mean = "
            f"`{qsum['mean_roc_auc_new_14']:.4f}`; `0` scaffold leakage across all 26 models)."
        ),
        "",
        "## 2. Gate 1 Unbiased Native Redocking (12 New TS/SCZ/AD + 4 AD Counters = 16)",
        (
            "- **Protocol:** Universal tight per-axis orthosteric box "
            "(`clamp(span + 6, 14, 22)` A), pH 7.4, 3 seeds `(42, 137, 2026)`, "
            "retaining `FAD` (`MAOB/MAOA`), `ADP` (`NLRP3`), `ZN/MG` (`PDE1B`)."
        ),
        (
            f"- **Aggregate Gate 1:** Best-3 RMSD `<= 2.0 A` = "
            f"**`{g1['pass_best3_le_2p0A_count']}/16` "
            f"({g1['pass_best3_le_2p0A_rate'] * 100:.1f}%)** | "
            f"Top-1 `<= 2.0 A` = **`{g1['pass_top1_le_2p0A_count']}/16`** | "
            f"Median Top-1 = **`{g1['median_top1_rmsd_across_16']:.2f} A`** | "
            f"Median Best-3 = **`{g1['median_best3_rmsd_across_16']:.2f} A`** | "
            f"PoseBusters = **`{g1['mean_posebusters_pass_rate'] * 100:.1f}%`** | "
            f"Top-1 Anchor = **`{g1['mean_top1_anchor_recovery_rate'] * 100:.1f}%`**."
        ),
        "",
        "## 3. Head-to-Head Local Chemical Foundation LLMs (24 Targets, N=64/Target = 1,536/Arm)",
        "",
        hdr_cols,
        hdr_sep,
    ]
    arm_order = (
        "Ctrl-ZINC",
        "v3.1_GP-MoLFormer",
        "v4.0_GP-MoLFormer",
        "v4.0_SAFE-GPT-87M",
        "v4.0_NovoMolGen-157M",
        "v4.0_Best_Shuffled",
    )
    for aname in arm_order:
        if aname not in arms:
            continue
        o = arms[aname]["overall_24"]
        ps = arms[aname]["sink_elimination_audit"]["PDE10A"]
        ds = arms[aname]["sink_elimination_audit"]["DRD3"]
        tag = f"**{aname}** ★" if aname == best_arm else f"**{aname}**"
        lines.append(
            f"| {tag} | {o['validity'] * 100:.1f}% | {o['uniqueness'] * 100:.1f}% | "
            f"{o['strict_novelty'] * 100:.1f}% | {o['global_int_div_1']:.3f} | "
            f"{o['triage_pass_rate'] * 100:.1f}% | {o['reagent_quarantine_rate'] * 100:.2f}% | "
            f"{o['v31_7rule_safe_rate'] * 100:.1f}% | "
            f"**{o['v4_pharma_12ep_safe_rate'] * 100:.1f}%** | "
            f"{o['cns_mpo_v2_mean']:.2f} | {o['herg_prob_mean']:.3f} | "
            f"{o['mdr1_efflux_prob_mean']:.3f} | {o['p_on_target_mean']:.3f} | "
            f"{ps['vmat2_bleed_rate'] * 100:.1f}% | {ds['vmat2_bleed_rate'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "### Cohort Breakdown (`v4.0 12-Ep Safe %` / `P(on_target)`)",
            "",
            "| Arm | v3.1 Core-12 | New TS/SCZ-6 | New Alzheimer's-6 | Overall 24 |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for aname in (
        "v3.1_GP-MoLFormer",
        "v4.0_GP-MoLFormer",
        "v4.0_SAFE-GPT-87M",
        "v4.0_NovoMolGen-157M",
    ):
        a = arms[aname]
        c12, cts, cad, o24 = a["v31_core_12"], a["new_ts_scz_6"], a["new_ad_6"], a["overall_24"]
        lines.append(
            f"| **{aname}** | "
            f"{c12['v4_pharma_12ep_safe_rate'] * 100:.1f}% / {c12['p_on_target_mean']:.3f} | "
            f"{cts['v4_pharma_12ep_safe_rate'] * 100:.1f}% / {cts['p_on_target_mean']:.3f} | "
            f"{cad['v4_pharma_12ep_safe_rate'] * 100:.1f}% / {cad['p_on_target_mean']:.3f} | "
            f"**{o24['v4_pharma_12ep_safe_rate'] * 100:.1f}% / {o24['p_on_target_mean']:.3f}** |"
        )

    # SAFE-GPT Decoration Table
    dec = h2h_rep.get("safegpt_scaffold_decoration", {})
    lines.extend(
        [
            "",
            "## 4. SAFE-GPT 3D-Conditioned Clinical Scaffold Decoration Showcase",
            "",
            (
                "| Clinical Core | Target (PDB) | Valid % | Core Kept "
                "| Parent Liability | Best Decorated Lead |"
            ),
            "| :--- | :---: | :---: | :---: | :--- | :--- |",
        ]
    )
    for slug in ("ecopipam_drd1", "donepezil_ache", "safinamide_maob", "xanomeline_chrm4"):
        if slug not in dec:
            continue
        item = dec[slug]
        pp = item["parent_profile"]
        top_a = item["top_analogs"][0] if item.get("top_analogs") else None
        top_str = (
            f"MPO={top_a['cns_mpo_v2']:.2f}, hERG={top_a['herg_prob']:.2f}, "
            f"M3={top_a['chrm3_avoid_prob']:.2f}, P(on)={top_a['p_on_target']:.2f}"
            if top_a
            else "N/A"
        )
        lines.append(
            f"| **{item['drug_name']}** | `{item['target_id']}:{item['pdb_id']}` | "
            f"{item['validity'] * 100:.1f}% | "
            f"{item['n_core_preserved']}/{item['n_unique_valid']} | "
            f"hERG={pp['herg_prob']:.2f}, M3={pp['chrm3_avoid_prob']:.2f}, "
            f"MPO={pp['cns_mpo_v2']:.2f} | {top_str} |"
        )

    # 5 Flagship Counter-Target Docking Selectivity Panels
    panels = dock_rep.get("counter_target_panels", {})
    lines.extend(
        [
            "",
            "## 5. Held-Out AutoDock Vina v1.2.7 Physics & 5 Counter-Target ΔΔG Selectivity Panels",
            (
                f"- **All 24 Targets (`{best_arm}` vs `v3.1`):** "
                f"`{best_arm}` Mean Top-1 Vina ΔG = **`{best_dock['mean_vina_dg']:.2f} kcal/mol`** "
                f"(PoseBusters = **`{best_dock['posebusters_pass_rate'] * 100:.1f}%`**, "
                f"Anchor = **`{best_dock['anchor_any_mode_rate'] * 100:.1f}%`**) vs "
                f"`v3.1` Mean ΔG = **`{v31_dock['mean_vina_dg']:.2f} kcal/mol`**."
            ),
            "",
            (
                "| Panel | On vs Off Pocket | v4.0 Best ΔΔG (kcal/mol) "
                "| v3.1 Baseline ΔΔG | Clinical Comparator ΔΔG |"
            ),
            "| :--- | :---: | :---: | :---: | :--- |",
        ]
    )
    for pname, pdata in panels.items():
        v4_ddg = pdata.get("v4_best_mean_ddg")
        v31_ddg = pdata.get("v31_mean_ddg")
        clin = pdata.get("clinical_comparators", {})
        clin_str = ", ".join(
            f"{cn}: {cd.get('ddg'):+.2f}" for cn, cd in clin.items() if cd.get("ddg") is not None
        )
        v4_s = f"{v4_ddg:+.2f}" if v4_ddg is not None else "—"
        v31_s = f"{v31_ddg:+.2f}" if v31_ddg is not None else "—"
        lines.append(
            f"| **{pname}** | `{pdata['on_pocket']}` vs `{pdata['off_pocket']}` | "
            f"**{v4_s}** | {v31_s} | {clin_str or '—'} |"
        )

    # CASF-2016 + Credibility Dossier
    n_art = dossier["step_5_provenance_and_execution"]["n_tracked_artifacts"]
    lines.extend(
        [
            "",
            "## 6. CASF-2016 Retention & FDA/EMA 7-Step AI Credibility Dossier",
            (
                f"- **CASF-2016 Retention (`{best_arm}`):** Frozen PaiNN-SE3 "
                f"`pKd r = {casf_b['pkd_pearson_r']:.4f}`, "
                f"DrugCLIP `R@5 = {casf_b['drugclip_recall_at_5'] * 100:.1f}%`, "
                f"Gen Validity = `{casf_b['gen_validity'] * 100:.1f}%`."
            ),
            (
                f"- **FDA/EMA Credibility Dossier:** Saved to `{TABLE_CREDIBILITY_PATH}` "
                f"(`{n_art}` SHA-256 tracked artifacts, 10/10 FDA-EMA Guiding Principles)."
            ),
            (
                "- **3D Docked SDF Exports:** Saved Top-2 leads/target (all 24 targets) + "
                f"SAFE-GPT decorated leads in `{DOCKED_SDF_DIR}/`."
            ),
        ]
    )
    TABLE_SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage_dock(*, max_workers: int = 4) -> dict[str, Any]:
    """Execute Stage ``dock``: 24-target held-out Vina, counter panels, SDFs & Dossier."""
    t0 = time.perf_counter()
    print(
        "[dock] Starting 100% held-out AutoDock Vina v1.2.7 Tri-Disorder Suite "
        f"(workers={max_workers})...",
        flush=True,
    )

    cand_payload = json.loads(
        (RUN_ROOT / "v4_candidates_for_docking.json").read_text(encoding="utf-8")
    )
    best_v4_arm: str = str(cand_payload["best_v4_arm"])
    candidates = cand_payload["candidates"]
    h2h_rep = json.loads(TABLE_HEAD_TO_HEAD_PATH.read_text(encoding="utf-8"))
    gate1_rep = json.loads(TABLE_GATE1_PATH.read_text(encoding="utf-8"))
    splits_sum = json.loads(TABLE_SPLITS_PATH.read_text(encoding="utf-8"))
    qsar_cv = json.loads(TABLE_QSAR_CV_PATH.read_text(encoding="utf-8"))
    admet_cv = json.loads(V4_ADMET_CV_PATH.read_text(encoding="utf-8"))
    dec_path = RUN_ROOT / "safegpt_scaffold_decoration.json"
    dec_payload = json.loads(dec_path.read_text(encoding="utf-8")) if dec_path.is_file() else {}

    # 1. Pre-prepare all 29 receptor PDBQTs (24 primary + DRD2_6LUQ + 4 AD counters) with cofactors!
    all_dock_pockets = (*ALL_24_TARGETS, "DRD2_6LUQ", *AD_STRUCTURAL_COUNTERS)
    RECEPTOR_PDBQT_DIR.mkdir(parents=True, exist_ok=True)
    DOCKED_SDF_DIR.mkdir(parents=True, exist_ok=True)
    pocket_files: dict[str, tuple[Path, Path]] = {}
    for pk in all_dock_pockets:
        p_pdb, l_sdf = _resolve_pocket_files_for_key(pk)
        pocket_files[pk] = (p_pdb, l_sdf)
        out_pdbqt = RECEPTOR_PDBQT_DIR / f"{pk}.pdbqt"
        if not out_pdbqt.is_file():
            prepare_receptor_pdbqt(p_pdb, out_pdbqt)

    # Load persistent Vina cache if resuming
    dock_cache: dict[str, dict[str, Any]] = {}
    if VINA_CACHE_PATH.is_file():
        for line in VINA_CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                dock_cache[obj["job_key"]] = obj["result"]
            except Exception:
                continue

    # 2. Build deduplicated job list
    jobs: dict[
        str, tuple[str, str, str, str, str, tuple[str, ...], tuple[str, ...], str | None, int]
    ] = {}

    def _enqueue(pocket_key: str, smiles: str, sdf_path: Path | None = None) -> str:
        jk = _dock_job_key(pocket_key, smiles)
        if jk in dock_cache and (sdf_path is None or sdf_path.is_file()):
            return jk
        if jk in jobs:
            if sdf_path is not None and jobs[jk][7] is None:
                old = jobs[jk]
                jobs[jk] = (*old[:7], str(sdf_path), old[8])
            return jk
        p_pdb, l_sdf = pocket_files[pocket_key]
        anchors, kinds = POCKET_ANCHOR_SPEC[pocket_key]
        jobs[jk] = (
            jk,
            pocket_key,
            smiles,
            str(p_pdb),
            str(l_sdf),
            anchors,
            kinds,
            str(sdf_path) if sdf_path is not None else None,
            2,
        )
        return jk

    # (a) All 24 targets: Top-5 for v3.1_Baseline and v4.0_Best (save Top-2 SDFs for v4.0_Best!)
    for tid in ALL_24_TARGETS:
        for rec in candidates["v3.1_GP-MoLFormer"][tid][:5]:
            _enqueue(tid, rec["canonical_smiles"])
        for rank_idx, rec in enumerate(candidates[best_v4_arm][tid][:5], start=1):
            sdf_p = DOCKED_SDF_DIR / f"{tid}_v4_lead_{rank_idx}.sdf" if rank_idx <= 2 else None
            _enqueue(tid, rec["canonical_smiles"], sdf_path=sdf_p)

    # (b) Other 2 v4.0 backbones Top-3 on 14 Flagship Targets (DRD1, PDE10A, DRD3, CHRM4 + 12 New)
    flagship_targets = ("DRD1", "PDE10A", "DRD3", "CHRM4", *NEW_12_TARGETS)
    other_v4_arms = [
        a
        for a in ("v4.0_GP-MoLFormer", "v4.0_SAFE-GPT-87M", "v4.0_NovoMolGen-157M")
        if a != best_v4_arm
    ]
    for arm_o in other_v4_arms:
        for tid in flagship_targets:
            for rec in candidates[arm_o][tid][:3]:
                _enqueue(tid, rec["canonical_smiles"])

    # (c) 5 Flagship Counter-Target Panels (cross-dock Top-5 v4_Best & v3.1 onto off-targets!)
    panel_defs: list[tuple[str, str, str, list[tuple[str, str]]]] = [
        (
            "Tourette_DRD1_vs_DRD2_6CM4",
            "DRD1",
            "DRD2",
            [
                (
                    "Ecopipam",
                    TRI_DISORDER_TARGET_REGISTRY["DRD1"].reference_drugs[0].canonical_smiles,
                ),
                (
                    "Haloperidol",
                    TRI_DISORDER_TARGET_REGISTRY["DRD2"].reference_drugs[1].canonical_smiles,
                ),
            ],
        ),
        (
            "Tourette_DRD1_vs_DRD2_6LUQ",
            "DRD1",
            "DRD2_6LUQ",
            [
                (
                    "Ecopipam",
                    TRI_DISORDER_TARGET_REGISTRY["DRD1"].reference_drugs[0].canonical_smiles,
                ),
                (
                    "Haloperidol",
                    TRI_DISORDER_TARGET_REGISTRY["DRD2"].reference_drugs[1].canonical_smiles,
                ),
            ],
        ),
        (
            "SCZ_AntiSink_PDE10A_vs_VMAT2",
            "PDE10A",
            "VMAT2",
            [
                (
                    "Balipodect",
                    TRI_DISORDER_TARGET_REGISTRY["PDE10A"].reference_drugs[0].canonical_smiles,
                ),
                (
                    "Valbenazine",
                    TRI_DISORDER_TARGET_REGISTRY["VMAT2"].reference_drugs[0].canonical_smiles,
                ),
            ],
        ),
        (
            "SCZ_NonD2_CHRM4_vs_DRD2",
            "CHRM4",
            "DRD2",
            [
                (
                    "Xanomeline",
                    TRI_DISORDER_TARGET_REGISTRY["CHRM4"].reference_drugs[0].canonical_smiles,
                )
            ],
        ),
        (
            "SCZ_NonD2_CHRM1_vs_DRD2",
            "CHRM1",
            "DRD2",
            [
                (
                    "Betovumeline",
                    TRI_DISORDER_TARGET_REGISTRY["CHRM1"].reference_drugs[0].canonical_smiles,
                )
            ],
        ),
        (
            "AD_MAOB_vs_MAOA",
            "MAOB",
            "MAOA",
            [
                (
                    "Safinamide",
                    TRI_DISORDER_TARGET_REGISTRY["MAOB"].reference_drugs[0].canonical_smiles,
                ),
                (
                    "Tisolagiline",
                    TRI_DISORDER_TARGET_REGISTRY["MAOB"].reference_drugs[1].canonical_smiles,
                ),
                ("Harmine", V4_COUNTER_TARGET_REGISTRY["MAOA"].reference_drugs[0].canonical_smiles),
            ],
        ),
        (
            "AD_GSK3B_vs_CDK2",
            "GSK3B",
            "CDK2",
            [
                (
                    "BRD0705",
                    TRI_DISORDER_TARGET_REGISTRY["GSK3B"].reference_drugs[0].canonical_smiles,
                ),
                (
                    "CDK2_Tool",
                    V4_COUNTER_TARGET_REGISTRY["CDK2"].reference_drugs[0].canonical_smiles,
                ),
            ],
        ),
        (
            "AD_ACHE_vs_BCHE",
            "ACHE",
            "BCHE",
            [
                (
                    "Donepezil",
                    TRI_DISORDER_TARGET_REGISTRY["ACHE"].reference_drugs[0].canonical_smiles,
                ),
                ("Tacrine", V4_COUNTER_TARGET_REGISTRY["BCHE"].reference_drugs[0].canonical_smiles),
            ],
        ),
        (
            "AD_BACE1_vs_BACE2",
            "BACE1",
            "BACE2",
            [
                (
                    "Umibecestat",
                    TRI_DISORDER_TARGET_REGISTRY["BACE1"].reference_drugs[0].canonical_smiles,
                ),
                (
                    "Verubecestat",
                    TRI_DISORDER_TARGET_REGISTRY["BACE1"].reference_drugs[1].canonical_smiles,
                ),
            ],
        ),
    ]
    for _pname, on_pk, off_pk, clin_list in panel_defs:
        for rec in candidates[best_v4_arm][on_pk][:5]:
            _enqueue(on_pk, rec["canonical_smiles"])
            _enqueue(off_pk, rec["canonical_smiles"])
        for rec in candidates["v3.1_GP-MoLFormer"][on_pk][:5]:
            _enqueue(on_pk, rec["canonical_smiles"])
            _enqueue(off_pk, rec["canonical_smiles"])
        for _cname, csmi in clin_list:
            _enqueue(on_pk, csmi)
            _enqueue(off_pk, csmi)

    # Extra AD clinical benchmarks (NLRP3 NP3-253 & OGA Ceperognastat)
    _enqueue("NLRP3", TRI_DISORDER_TARGET_REGISTRY["NLRP3"].reference_drugs[0].canonical_smiles)
    _enqueue("OGA", TRI_DISORDER_TARGET_REGISTRY["OGA"].reference_drugs[0].canonical_smiles)

    # (d) SAFE-GPT Decorated Clinical Leads (Top-2 SDFs saved!)
    for slug in ("ecopipam_drd1", "donepezil_ache", "safinamide_maob", "xanomeline_chrm4"):
        if slug not in dec_payload:
            continue
        item = dec_payload[slug]
        on_pk = item["target_id"]
        off_pk = item["counter_target"]
        for r_idx, analog in enumerate(item.get("top_analogs", [])[:3], start=1):
            sdf_p = DOCKED_SDF_DIR / f"safegpt_dec_{slug}_lead_{r_idx}.sdf" if r_idx <= 2 else None
            _enqueue(on_pk, analog["smiles"], sdf_path=sdf_p)
            if off_pk in pocket_files:
                _enqueue(off_pk, analog["smiles"])

    print(
        f"[dock] Executing {len(jobs)} new Vina docking jobs ({len(dock_cache)} already cached)...",
        flush=True,
    )
    if jobs:
        with (
            ProcessPoolExecutor(max_workers=max_workers) as pool,
            VINA_CACHE_PATH.open("a", encoding="utf-8") as cache_f,
        ):
            futs = [pool.submit(_dock_worker_task, args) for args in jobs.values()]
            for done_cnt, fut in enumerate(as_completed(futs), start=1):
                jk, d_res = fut.result()
                dock_cache[jk] = d_res
                cache_f.write(json.dumps(_json_safe({"job_key": jk, "result": d_res})) + "\n")
                cache_f.flush()
                if done_cnt % 40 == 0 or done_cnt == len(jobs):
                    print(
                        f"  [dock] Progress {done_cnt}/{len(jobs)} "
                        f"({time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )

    # 3. Assemble 24-Target Physics Tables & 5 Counter-Target Panels
    def _lookup(pk: str, smi: str) -> dict[str, Any]:
        return dock_cache.get(_dock_job_key(pk, smi), {"success": False})

    def _summarize_dock_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in rows if r.get("success") and r.get("vina_affinity") is not None]
        if not ok:
            return {
                "n_ok": 0,
                "mean_vina_dg": None,
                "mean_le": None,
                "posebusters_pass_rate": 0.0,
                "anchor_any_mode_rate": 0.0,
            }
        return {
            "n_ok": len(ok),
            "mean_vina_dg": round(float(np.mean([float(r["vina_affinity"]) for r in ok])), 3),
            "best_vina_dg": round(float(np.min([float(r["vina_affinity"]) for r in ok])), 3),
            "mean_le": round(
                float(
                    np.mean(
                        [
                            float(r["ligand_efficiency"])
                            for r in ok
                            if r.get("ligand_efficiency") is not None
                        ]
                    )
                ),
                4,
            ),
            "mean_plif": round(
                float(
                    np.mean(
                        [
                            float(r["plif_tanimoto"])
                            for r in ok
                            if r.get("plif_tanimoto") is not None
                        ]
                    )
                ),
                4,
            ),
            "posebusters_pass_rate": round(
                float(
                    np.mean(
                        [1.0 if (r.get("posebusters") or {}).get("passed_all") else 0.0 for r in ok]
                    )
                ),
                4,
            ),
            "anchor_top1_rate": round(
                float(
                    np.mean(
                        [1.0 if (r.get("anchor_top1") or {}).get("engaged") else 0.0 for r in ok]
                    )
                ),
                4,
            ),
            "anchor_any_mode_rate": round(
                float(
                    np.mean(
                        [
                            1.0 if (r.get("anchor_any_mode") or {}).get("engaged") else 0.0
                            for r in ok
                        ]
                    )
                ),
                4,
            ),
        }

    per_target_dock: dict[str, Any] = {}
    all_rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tid in ALL_24_TARGETS:
        t_entry: dict[str, Any] = {}
        for aname in ("v3.1_GP-MoLFormer", best_v4_arm):
            rows = []
            for rec in candidates[aname][tid][:5]:
                dr = _lookup(tid, rec["canonical_smiles"])
                rows.append(
                    {
                        "rank": rec["rank"],
                        "smiles": rec["canonical_smiles"],
                        "cns_mpo_v2": round(float(rec["cns_mpo_v2"]), 2),
                        "herg_prob": round(float(rec["herg_prob"]), 3),
                        "mdr1_efflux_prob": round(float(rec["mdr1_efflux_prob"]), 3),
                        "p_on_target": round(float(rec["p_on_target"]), 3),
                        "v4_pharma_12ep_safe": bool(rec["v4_pharma_12ep_safe"]),
                        **dr,
                    }
                )
            all_rows_by_arm[aname].extend(rows)
            t_entry[aname] = {"summary": _summarize_dock_rows(rows), "top_leads": rows[:2]}
        if tid in flagship_targets:
            for arm_o in other_v4_arms:
                rows_o = [
                    _lookup(tid, rec["canonical_smiles"]) for rec in candidates[arm_o][tid][:3]
                ]
                all_rows_by_arm[arm_o].extend(rows_o)
                t_entry[arm_o] = {"summary": _summarize_dock_rows(rows_o)}
        per_target_dock[tid] = t_entry

    # Counter-Target Panels
    counter_panels_out: dict[str, Any] = {}
    for pname, on_pk, off_pk, clin_list in panel_defs:

        def _panel_arm_ddg(
            aname: str, *, p_on: str = on_pk, p_off: str = off_pk
        ) -> tuple[float | None, list[dict[str, Any]]]:
            ddgs: list[float] = []
            details: list[dict[str, Any]] = []
            for rec in candidates[aname][p_on][:5]:
                smi = rec["canonical_smiles"]
                d_on = _lookup(p_on, smi)
                d_off = _lookup(p_off, smi)
                if d_on.get("success") and d_off.get("success"):
                    ddg_d = compute_selectivity_ddg(
                        float(d_on["vina_affinity"]), [float(d_off["vina_affinity"])]
                    )
                    ddg_val = float(ddg_d["ddg_vs_mean"])
                    ddgs.append(ddg_val)
                    details.append(
                        {
                            "smiles": smi,
                            "on_dg": d_on["vina_affinity"],
                            "off_dg": d_off["vina_affinity"],
                            "ddg": ddg_val,
                            "fold_est": ddg_d["estimated_fold_selectivity_mean"],
                            "v4_safe": bool(rec["v4_pharma_12ep_safe"]),
                            "herg_prob": round(float(rec["herg_prob"]), 3),
                        }
                    )
            return (round(float(np.mean(ddgs)), 3) if ddgs else None), details

        v4_ddg, v4_det = _panel_arm_ddg(best_v4_arm)
        v31_ddg, v31_det = _panel_arm_ddg("v3.1_GP-MoLFormer")
        clin_out: dict[str, Any] = {}
        for cname, csmi in clin_list:
            d_on = _lookup(on_pk, csmi)
            d_off = _lookup(off_pk, csmi)
            if d_on.get("success") and d_off.get("success"):
                ddg_c = compute_selectivity_ddg(
                    float(d_on["vina_affinity"]), [float(d_off["vina_affinity"])]
                )
                clin_out[cname] = {
                    "on_dg": d_on["vina_affinity"],
                    "off_dg": d_off["vina_affinity"],
                    "ddg": ddg_c["ddg_vs_mean"],
                    "fold_est": ddg_c["estimated_fold_selectivity_mean"],
                }
        counter_panels_out[pname] = {
            "on_pocket": on_pk,
            "off_pocket": off_pk,
            "v4_best_arm": best_v4_arm,
            "v4_best_mean_ddg": v4_ddg,
            "v31_mean_ddg": v31_ddg,
            "v4_vs_v31_ddg_gain": round(v4_ddg - v31_ddg, 3)
            if (v4_ddg is not None and v31_ddg is not None)
            else None,
            "clinical_comparators": clin_out,
            "v4_best_leads": sorted(v4_det, key=lambda x: x["ddg"])[:3],
            "v31_leads": sorted(v31_det, key=lambda x: x["ddg"])[:3],
        }

    # SAFE-GPT Decorated Clinical Leads Docking Summary
    dec_dock_out: dict[str, Any] = {}
    for slug in ("ecopipam_drd1", "donepezil_ache", "safinamide_maob", "xanomeline_chrm4"):
        if slug not in dec_payload:
            continue
        item = dec_payload[slug]
        on_pk = item["target_id"]
        off_pk = item["counter_target"]
        docked_analogs = []
        for analog in item.get("top_analogs", [])[:3]:
            d_on = _lookup(on_pk, analog["smiles"])
            d_off = _lookup(off_pk, analog["smiles"]) if off_pk in pocket_files else {}
            ddg_val = (
                round(float(d_on["vina_affinity"]) - float(d_off["vina_affinity"]), 3)
                if (d_on.get("success") and d_off.get("success"))
                else None
            )
            docked_analogs.append(
                {
                    **analog,
                    "on_vina_dg": d_on.get("vina_affinity"),
                    "off_vina_dg": d_off.get("vina_affinity"),
                    "ddg": ddg_val,
                    "posebusters_passed": (d_on.get("posebusters") or {}).get("passed_all"),
                    "anchor_engaged": (d_on.get("anchor_any_mode") or {}).get("engaged"),
                    "saved_sdf": d_on.get("saved_sdf"),
                }
            )
        dec_dock_out[slug] = {
            "drug_name": item["drug_name"],
            "target_id": on_pk,
            "counter_target": off_pk,
            "parent_profile": item["parent_profile"],
            "docked_analogs": docked_analogs,
        }

    dock_report = {
        "schema_version": "geneuclidean-v4.0-tri-disorder-docking",
        "best_v4_arm": best_v4_arm,
        "protocol": {
            "engine": "AutoDock Vina v1.2.7 (100% held out from training/DPO/GDPO)",
            "box_mode": "tight_per_axis (clamp(span_i + 6.0, 14.0, 22.0) A)",
            "protonation": "pH 7.4",
            "exhaustiveness": 8,
            "cofactors_retained": ["FAD", "ADP", "ZN", "MG"],
        },
        "overall_24_summary": {
            aname: _summarize_dock_rows(rows) for aname, rows in all_rows_by_arm.items()
        },
        "counter_target_panels": counter_panels_out,
        "safegpt_decorated_docking": dec_dock_out,
        "per_target": per_target_dock,
        "saved_sdf_count": len(list(DOCKED_SDF_DIR.glob("*.sdf"))),
        "saved_sdf_dir": str(DOCKED_SDF_DIR),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    _write_json(TABLE_DOCKING_PATH, dock_report)

    # 4. Build FDA Jan 2025 / Joint FDA-EMA Jan 2026 7-Step Credibility Dossier
    best_o = h2h_rep["arms"][best_v4_arm]["overall_24"]
    dossier = build_credibility_dossier(
        model_name=f"GenEuclidean v4.0 Tri-Disorder Discovery Suite ({best_v4_arm})",
        model_version="4.0.0",
        admet_cv_metrics=admet_cv,
        applicability_domain_summary={
            "best_v4_arm": best_v4_arm,
            "generated_24target_in_domain_core_rate": best_o["in_domain_core_rate"],
            "generated_24target_min_ad_mean": best_o["min_ad_core_mean"],
            "conformal_singleton_rate_mean": best_o["conformal_singleton_rate"],
        },
        physics_gate1_metrics={
            "expansion_16_complexes": gate1_rep["summary"],
            "held_out_24target_docking": dock_report["overall_24_summary"][best_v4_arm],
        },
        provenance_files={
            "v4_admet_avoidome_oracles_joblib": V4_ADMET_MODEL_PATH,
            "v4_24target_qsar_joblib": V4_QSAR_MODEL_PATH,
            "v4_tri_disorder_splits_json": TABLE_SPLITS_PATH,
            "v4_tri_disorder_qsar_cv_json": TABLE_QSAR_CV_PATH,
            "v4_gate1_expansion_redocking_json": TABLE_GATE1_PATH,
            "v4_llm_head_to_head_json": TABLE_HEAD_TO_HEAD_PATH,
            "v4_tri_disorder_docking_json": TABLE_DOCKING_PATH,
            "v4_gpmolformer_adapter": RUN_ROOT / "v4_gpmolformer_aligned" / "best_adapter.pt",
            "v4_safegpt_adapter": RUN_ROOT / "v4_safegpt_aligned" / "best_adapter.pt",
            "v4_novomolgen157m_adapter": RUN_ROOT / "v4_novomolgen157m_aligned" / "best_adapter.pt",
        },
    )
    _write_json(TABLE_CREDIBILITY_PATH, dossier)

    # 5. Write Publication-Ready Summary Markdown
    _write_v4_summary_markdown(splits_sum, qsar_cv, gate1_rep, h2h_rep, dock_report, dossier)
    print(
        f"[dock] DONE in {dock_report['elapsed_sec']:.1f}s | "
        f"SDFs saved={dock_report['saved_sdf_count']} | "
        f"Dock JSON={TABLE_DOCKING_PATH.stat().st_size / 1024:.1f} KB | "
        f"Credibility Dossier={TABLE_CREDIBILITY_PATH.stat().st_size / 1024:.1f} KB",
        flush=True,
    )
    return dock_report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GenEuclidean v4.0 Head-to-Head Local LLM & 24-Target Tri-Disorder Benchmark"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=("prepare", "gate1", "train", "eval", "dock", "all"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Polite sklearn CPU thread cap.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel Vina worker cap.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=EVAL_NUM_SAMPLES,
        help="Samples per target during eval (default: 64 -> 1,536/arm across 24 targets).",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=("all", "v4_gpmolformer", "v4_safegpt", "v4_novomolgen157m"),
        help="Optional arm filter for --stage train.",
    )
    parser.add_argument(
        "--align-only",
        action="store_true",
        help="For --stage train, load existing SFT adapter and run alignment + decoration only.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HOME", str(Path(".hf_cache.nosync").resolve()))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()
    t_all = time.perf_counter()

    if args.stage in ("prepare", "all"):
        run_stage_prepare(seed=args.seed, n_jobs=args.n_jobs)
    if args.stage in ("gate1", "all"):
        run_stage_gate1(max_workers=args.max_workers)
    if args.stage in ("train", "all"):
        run_stage_train(seed=args.seed, arm=args.arm, align_only=args.align_only)
    if args.stage in ("eval", "all"):
        run_stage_eval(seed=args.seed, num_samples=args.num_samples)
    if args.stage in ("dock", "all"):
        run_stage_dock(max_workers=args.max_workers)

    print(
        f"\n[v4-benchmark] Stage '{args.stage}' finished in {time.perf_counter() - t_all:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
