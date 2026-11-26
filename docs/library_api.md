# GenEuclidean `v4.0.0` Python Library & Extensibility Guide (`import geneuclidean as ge`)

GenEuclidean is a modular, strictly typed (**PEP 561** `py.typed`), lazily loaded (**PEP 562** `__getattr__`), plugin-extensible scientific Python library for $SE(3)/E(3)$-equivariant 3D protein pocket-conditioned *de novo* molecular generation, SAFE-GPT 3D scaffold decoration, physiological pH 7.4 preparation, industrial MedChem triage, 12-endpoint calibrated CNS ADMET/Avoid-ome scoring, AutoDock Vina 3D physics validation, dual-mode 3D/2D studio visualization, and FDA/EMA AI credibility reporting.

> **Research Boundary:** All generated molecules, scaffold decorations, docking poses, and ADMET scores produced by GenEuclidean are strictly *in silico* computational hypotheses. They are not clinically validated medicines and require chemical synthesis, biochemical/functional pharmacology, off-target profiling, and preclinical validation before translation.

---

## 1. Architecture Overview

The library separates zero-PyTorch core infrastructure from modular SDK facades and deep-learning subsystems:

| Layer | Modules | Responsibility |
|---|---|---|
| **Top-level Lazy Facade** | `geneuclidean.__init__`, `geneuclidean.api` | PEP 562 lazy exports (`< 15 ms` cold import, zero eager `torch`/`transformers`/`matplotlib` loads). |
| **Core Extensibility** | `geneuclidean.exceptions`, `geneuclidean.protocols`, `geneuclidean.registry` | Backward-compatible `GenEuclideanError` hierarchy, PEP 544 `Protocol` contracts, thread-safe `Registry[T]`, and `importlib.metadata.entry_points` plugin discovery. |
| **Modular SDK (`sdk/`)** | `pocket`, `scoring`, `designer`, `pipeline`, `docking`, `viz`, `credibility`, `registries` | Single-responsibility high-level APIs (`Pocket`, `MoleculeDesigner`, `DesignPipeline`, `DesignResult`, composable filters/scorers, SDF/CSV/JSON/DataFrame export). |
| **3D Geometry & LLMs** | `models.encoders`, `models.decoders`, `models.hf_backbones`, `models.llm_bridge`, `models.safe_codec` | Registry-backed `PaiNN-SE3`/`EGNN`/`E3NN`/`SchNet`, `GP-MoLFormer`, `SAFE-GPT`, `NovoMolGen-157M` (native MPS SDPA), and pure-RDKit SAFE codec. |
| **CNS Pharmacology** | `cns.targets`, `cns.tourette_circuit`, `cns.mpo`, `cns.medchem_triage`, `cns.admet`, `cns.credibility` | 28-target human Tri-Disorder + Tourette CSTC/GABA atlas, Pfizer MPO v1/v2, reagent quarantine, 12-endpoint Avoid-ome + Split-Conformal UQ, FDA/EMA 7-step dossier. |
| **3D Physics & Viz** | `evaluation.protonation`, `evaluation.docking_pipeline`, `evaluation.interactions3d`, `evaluation.ring_strain`, `evaluation.viz3d` | Deterministic pH 7.4 ionization, multi-conformer Vina v1.2.7 + PoseBusters + $\Delta\Delta G$ selectivity matrix, Cremer–Pople saturated-ring conformational strain auditing, 3Dmol.js HTML & 300-DPI PNG renderers. |
| **3D Steering, Hops & MTL** | `sampling.pocket_steering`, `cns.scaffold_hop`, `training.mtl_intervention` | Multi-conformation pocket ensemble & contrastive off-target CFG (`design_ensemble`, `score_pmi` $\Delta\text{PMI}_{3\text{D}}$), 5-axis natural-product scaffold-hop de-risking (`audit_scaffold_hop_campaign`), and AIM-inspired adaptive multi-task gradient conflict mediation (`GradientConflictMediator`). |

---

## 2. Installation & Modular Dependency Extras

```bash
# Core library
pip install -e .

# Optional scientific extras
pip install -e ".[cns]"       # Calibrated 3-ep / 12-ep CNS ADMET & Avoid-ome oracles
pip install -e ".[viz]"       # 300-DPI static 3D/2D matplotlib + Pillow figures (alias: .[viz3d])
pip install -e ".[docking]"   # Meeko + Gemmi for AutoDock Vina v1.2.7 structural docking
pip install -e ".[llm]"       # Hugging Face transformers, PEFT LoRA, safetensors, accelerate
pip install -e ".[safe]"      # Optional SAFE fragment toolkit integration
pip install -e ".[dev]"       # pytest, pytest-cov, ruff, mypy, type stubs
pip install -e ".[all]"       # Complete batteries-included environment
```

---

## 3. 60-Second Quickstart

```python
import geneuclidean as ge

# 1. Explore the 28-target human CNS atlas & chemical LLMs (zero PyTorch imported!)
all_targets = ge.list_targets()                  # 28 curated human targets (TS + SCZ + AD + CSTC/GABA)
ad_targets = ge.list_targets("alzheimer")        # 6 Alzheimer's targets (ACHE, MAOB, BACE1, GSK3B, NLRP3, OGA)
ts_audit = ge.audit_tourette_circuit("GRM5")     # CSTC circuit audit & clinical evidence grade
backbones = ge.list_backbones()                  # GP-MoLFormer, SAFE-GPT, NovoMolGen-157M, Qwen3, mock

# 2. Load a 3D pocket (auto-retains integral cofactors like FAD for MAOB / ADP for NLRP3!)
pocket = ge.Pocket.from_target("MAOB")
print(pocket.summary())  # includes cofactors=['FAD']

# 3. Score, protonate (pH 7.4), MedChem-triage & optionally apply custom scorers
candidates = [
    "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",  # Ecopipam (DRD1 Ph3 positive control)
    "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1",               # GenEuclidean flagship DRD1_1 lead
    "O=C(NC1CCCCC1)NC1CCCCC1",                      # DCU coupling byproduct (auto-quarantined -> 0.0!)
]
reports = ge.score_molecules(
    candidates,
    target="DRD1",
    admet_profile="v3",  # use "v4" or "auto" for 12-endpoint Industrial Avoid-ome + Multiplicative Reward
)
for rep in reports:
    print(f"{rep.canonical_smiles} | MPOv2={rep.cns_mpo_v2:.2f} | q7.4={rep.net_charge_ph74:+d} | reward={rep.composite_reward:.3f}")
```

---

## 4. Composable Drug-Design Pipelines (`ge.DesignPipeline`)

`DesignPipeline` is an immutable fluent builder that chains generation (or 3D SAFE scaffold decoration), custom scoring plugins, ordered attrition filters, reranking, telemetry callbacks, and multi-format export (`to_sdf`, `to_csv`, `to_json`, `to_dataframe`):

```python
import geneuclidean as ge

pocket = ge.Pocket.from_pdb("path/to/9LLG_DRD1_pocket.pdb", target_id="DRD1")
designer = ge.MoleculeDesigner.from_pretrained("runs/v4_safegpt_gdpo/best_adapter.pt", device="auto")

# Wrap any custom domain scorer (e.g., proprietary QSAR, pharmacophore, or MW preference)
compact_bonus = ge.FunctionScorer(
    "compact_cns_bonus",
    lambda r: 1.0 if r.mw <= 360.0 and r.tpsa <= 70.0 else 0.4,
)

pipe = (
    ge.DesignPipeline(designer, name="tourette_drd1_lead_opt")
    .with_admet(enabled=True, profile="v4")
    .add_scorer(compact_bonus, weight=0.20)
    .add_filter(ge.TriageFilter(max_lilly_demerits=100))
    .add_filter(ge.MPOFilter(min_mpo_v2=4.5, max_tpsa=85.0))
    .add_filter(ge.BasicAmineFilter(allowed_charges=(1,)))
    .with_top_k(20)
)

# De novo generation OR 3D clinical scaffold decoration (SAFE-GPT)
de_novo_hits = pipe.run(pocket, n=64, cfg_scale=1.6, seed=42)
rescue_hits = pipe.run(
    pocket,
    n=32,
    core_smiles_with_dummy="COc1cc2c(cc1OC)C(=O)C([*])C2",
    seed=137,
)

# Export directly to PyMOL/Maestro SDF, spreadsheet CSV, JSON, or pandas DataFrame
de_novo_hits.to_sdf("results/drd1_hits.sdf", embed_3d=True, protonate_ph74=True)
de_novo_hits.to_csv("results/drd1_hits.csv")
de_novo_hits.to_json("results/drd1_hits.json")
print(pipe.describe())  # JSON-serializable provenance for FDA/EMA Credibility Dossiers
```

---

## 5. Extending GenEuclidean Without Modifying Core Source

GenEuclidean provides thread-safe generic registries (`ge.Registry[T]`) backed by runtime-checkable PEP 544 protocols (`ge.protocols`) and standard Python `importlib.metadata.entry_points` discovery (`ge.discover_plugins()`).

### 5.1 Custom 3D Equivariant Pocket Encoder
```python
import torch.nn as nn
import geneuclidean as ge
from geneuclidean.config import EncoderConfig
from geneuclidean.models.encoders import PocketEncoderOutput, build_encoder

@ge.register_encoder("my_se3_net", aliases=("custom_se3",))
def build_my_se3(cfg: EncoderConfig, max_neighbors: int = 24) -> nn.Module:
    return MySE3Encoder(in_dim=cfg.node_in_dim, out_dim=cfg.out_dim)

# Works seamlessly with Pydantic validation & GenEuclideanModel!
cfg = EncoderConfig(arch="my_se3_net", out_dim=256)
encoder = build_encoder(cfg)
```

### 5.2 Custom Autoregressive Decoder & Sampling Strategy
```python
import torch
import geneuclidean as ge

@ge.register_sampling_strategy("eta_cutoff")
def eta_sampling_probs(logits, *, temperature=1.0, top_k=30, top_p=0.92, min_p=0.08):
    probs = torch.softmax(logits / temperature, dim=-1)
    keep = probs >= min_p
    keep.scatter_(-1, probs.argmax(dim=-1, keepdim=True), True)
    masked = probs * keep.to(probs.dtype)
    return masked / masked.sum(dim=-1, keepdim=True).clamp(min=1e-12)
```

### 5.3 Custom Protein Targets & Scoped Overrides (Ideal for Ablations & Tests)
```python
import geneuclidean as ge
from geneuclidean.cns.targets import CNSTargetSpec

ge.register_target(my_proprietary_target_spec, aliases=("PROJ_X",))
pocket = ge.Pocket.from_target("PROJ_X")

# Or register temporarily inside a context manager (auto-restores on exit):
with ge.ENCODER_REGISTRY.temporary("ablation_enc", my_factory):
    ...
```

### 5.4 Third-Party Pip Plugin Discovery (`pyproject.toml`)
External packages can register plugins automatically via Python entry points and load them with `ge.discover_plugins()`:

```toml
[project.entry-points."geneuclidean.encoders"]
mace = "my_pkg.encoders:build_mace"

[project.entry-points."geneuclidean.scorers"]
herg_v2 = "my_pkg.scorers:HERGV2Scorer"
```

Supported entry-point groups: `geneuclidean.encoders`, `geneuclidean.decoders`, `geneuclidean.sampling_strategies`, `geneuclidean.scorers`, `geneuclidean.filters`.

---

## 6. Structured Exception Hierarchy (`geneuclidean.exceptions`)

Every library exception inherits from **both** `ge.GenEuclideanError` and its corresponding standard Python exception, ensuring 100% backward compatibility alongside actionable `.hint` diagnostics:

- `ge.TargetNotFoundError(GenEuclideanError, KeyError)`
- `ge.RegistryError(GenEuclideanError, KeyError)`
- `ge.InvalidMoleculeError(GenEuclideanError, ValueError)`
- `ge.PocketFeaturizationError(GenEuclideanError, ValueError)`
- `ge.CheckpointError(GenEuclideanError, ValueError)`
- `ge.StructureNotFoundError(GenEuclideanError, FileNotFoundError)`
- `ge.OptionalDependencyError(GenEuclideanError, RuntimeError)`
- `ge.DockingError(GenEuclideanError, RuntimeError)`

---

## 7. Advanced Frontier Modules (3D Ring Strain, Contrastive Ensemble Steering, NP Scaffold-Hopping & Multi-Task Gradient Mediation)

Four research-grade modules extend GenEuclidean for modern 3D SBDD workflows:

### 7.1 Saturated-Ring Conformational Strain (`geneuclidean.evaluation.ring_strain`)
Computes exact Cremer–Pople puckering coordinates ($Q, \theta, \phi$) and isolated saturated-ring-system strain energies ($\Delta E_{\text{ring}}$, preserving exocyclic conjugation while relaxing capping H/exocyclic atoms) with a pluggable backend registry (`RING_STRAIN_BACKEND_REGISTRY`, `"mmff94s"` built-in + drop-in hook for neural DFT models like `RingAnalyzer`):

```python
import geneuclidean as ge

profile = ge.analyze_ring_strain("path/to/docked_pose.sdf", strain_threshold_kcal=3.0)
print(profile.max_ring_strain_kcal, profile.passed, [(r.pucker_class, r.strain_energy_kcal) for r in profile.rings])

# Composable directly inside ge.DesignPipeline:
pipe = pipe.add_filter(ge.RingStrainFilter(max_strain_kcal=3.0)).add_scorer(ge.RingStrainScorer(), weight=0.15)
```

### 7.2 Multi-Conformation Ensemble & Contrastive Off-Target 3D CFG + $\Delta\text{PMI}_{3\text{D}}$ (`geneuclidean.sampling.pocket_steering`)
Overcomes single-snapshot pocket conditioning with KV-cached multi-conformation ensemble averaging and zero-retraining off-target logit repulsion, plus exact length-normalized 3D Pocket-Conditional Pointwise Mutual Information ($\text{PMI}_{3\text{D}}$ / $\Delta\text{PMI}_{3\text{D}}$):

```python
import geneuclidean as ge

# Average logits across 3 experimental DRD1 conformations while repelling DRD2 (6CM4)
res = designer.design_ensemble(
    [drd1_9llg, drd1_9u5e, drd1_7ljc],
    off_pocket=drd2_6cm4,
    w_on=1.8,
    w_off=0.6,
    n=64,
    seed=42,
)
# Zero-shot likelihood-free selectivity scoring of arbitrary candidates
pmi_scores = designer.score_pmi(res.smiles, [drd1_9llg, drd1_9u5e], off_pocket=drd2_6cm4)
```

### 7.3 5-Axis Natural-Product Scaffold-Hop & Avoid-ome De-Risking (`geneuclidean.cns.scaffold_hop`)
Audits generated libraries against complex toxic natural products (e.g., `ge.IBOGAINE_SMILES` on human SERT) across Bemis–Murcko hop distance, $\Delta\text{SA}$ synthetic simplification, CNS MPO v2 + pH 7.4 basic-amine pharmacophore, 12-endpoint Avoid-ome risk reduction ($\Delta P(\text{hERG})$, $\Delta P(\text{5-HT}_{2\text{B}})$), and MedChem hygiene:

```python
audit = ge.audit_scaffold_hop_campaign(
    candidate_smiles,
    reference_smiles=ge.IBOGAINE_SMILES,
    admet_suite=admet_suite,
)
```

### 7.4 Adaptive Multi-Task 3D Gradient Conflict Mediation (`geneuclidean.training.mtl_intervention`)
Diagnostics (`compute_gradient_conflict_matrix`) and pluggable shared-encoder gradient mediation (`MTL_MEDIATOR_REGISTRY`: `"scalarized"`, `"pcgrad"`, `"aim_policy"`) for multi-objective 3D pocket models (`lm_nll`, `pkd`, `drugclip`, `props`):

```python
from geneuclidean.training.mtl_intervention import GradientConflictMediator

mediator = GradientConflictMediator(("lm_nll", "pkd", "drugclip", "props"), strategy="aim_policy")
diag = mediator.backward(task_losses, shared_params=list(model.encoder.parameters()))
print(diag.conflict_fraction, diag.policy_matrix)
```

---

## 8. High-Level Reference Summary

- **Registries & Circuits:** `list_targets`, `get_target`, `register_target`, `unregister_target`, `audit_tourette_circuit`, `list_backbones`, `list_encoders`, `list_decoders`, `list_sampling_strategies`, `list_filters`, `list_scorers`, `discover_plugins`, `RING_STRAIN_BACKEND_REGISTRY`, `MTL_MEDIATOR_REGISTRY`.
- **3D Pockets:** `Pocket.from_target`, `Pocket.from_pdb`, `pocket.featurize`, `pocket.audit_hotspots`, `pocket.orthosteric_box`, `pocket.dock`, `pocket.visualize`, `pocket.summary`.
- **Scoring, Ionization, Strain & Triage:** `score_molecules`, `score_tourette_circuit`, `protonate_ph74`, `analyze_protonation_ph74`, `triage_molecule`, `triage_molecules`, `is_scaffold_hop`, `compute_scaffold_hop_metrics`, `audit_scaffold_hop_campaign`, `analyze_ring_strain`, `compute_cremer_pople`.
- **Filters & Scorers:** `TriageFilter`, `MPOFilter`, `ADMETFilter`, `BasicAmineFilter`, `QEDFilter`, `ScaffoldHopFilter`, `RingStrainFilter`, `RingStrainScorer`, `PredicateFilter`, `FunctionScorer`.
- **Generation & Pipelines:** `MoleculeDesigner.from_pretrained`, `MoleculeDesigner.from_mock`, `designer.design`, `designer.design_ensemble`, `designer.score_pmi`, `designer.decorate`, `DesignPipeline`, `BaseDesignCallback`, `DesignResult` (`top`, `filter`, `summary`, `to_dict`, `to_json`, `to_csv`, `to_sdf`, `to_dataframe`).
- **3D Docking & Selectivity:** `dock_and_profile`, `redock_complex`, `compute_orthosteric_box`, `generate_ring_diverse_conformers`, `compute_murcko_core_rmsd`, `compute_selectivity_ddg`, `compute_cross_target_matrix`.
- **Studio Visualization:** `visualize_complex`, `visualize_gallery`, `visualize_molecules_2d`.
- **Regulatory Credibility (FDA Jan 2025 / EMA Jan 2026):** `build_credibility_dossier`, `CredibilityDossier`, `export_credibility_json`, `export_credibility_markdown`.

```python
import geneuclidean as ge

dossier = ge.build_credibility_dossier(
    model_name="GenEuclidean CNS Discovery & ADMET Suite",
    model_version="4.0.0",
    provenance_files={"admet_cv": "results/tables/v4_admet_avoidome_cv_metrics.json"},
)
ge.export_credibility_json(dossier, "results/credibility_dossier.json")
ge.export_credibility_markdown(dossier, "results/credibility_dossier.md")
```

---

## 9. Packaging, Static Analysis & Verification

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/mypy src
.venv/bin/pytest -q
```
