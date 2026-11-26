# GenEuclidean `v5.0` Open-Pharma CADD Suite: Technical Report & Verified Benchmarks

> **Research Boundary:** All generated, screened, makeable, MMP-modified, or audited molecules and DMPK predictions reported below are strictly ***in silico* computational hypotheses** for academic research. They are **not** experimentally synthesized compounds or clinical recommendations.

---

## 1. Architecture Overview (Phases 1–4)

GenEuclidean `v5.0` extends the chiral $SE(3)$-equivariant 3D pocket-conditioned generative core (`v4.0`) with four modular, MIT-licensed, zero-PyTorch-core-compatible CADD subsystems designed for open-pharma interoperability (`Boltz-2`, `OpenFE`, `OpenADMET`, `IBM MMELON`, `Roche Torsion/mmpdb`, `AstraZeneca AiZynthFinder`):

| Pillar | New Modules | Core Capabilities |
| :--- | :--- | :--- |
| **Phase 1 — Beyond-PoseBusters 7-Axis Pose Auditor & Profiles** | `audit/pose_auditor.py`, `evaluation/torsion_strain.py`, `evaluation/tautomer_microstate.py`, `sdk/profiles.py` | Audits any docked or co-folded (`Boltz-2`/`AF3`/`Vina`) 3D complex across 7 axes: PoseBusters geometry/clashes, torsion-constrained Cremer–Pople ring strain, QM-style relaxed SMARTS torsion strain (with external Roche/ZBH XML loader), pH 7.4 tautomer/protomer & electrostatic checks, 6-ch PLIF anchors, MedChem hygiene, and indication-agnostic property profiles (`cns_mpo_v2`, `oral_ro5`, `kinase_selectivity`, `beyond_ro5`). |
| **Phase 2 — MIT Contrastive 3D Screener, IBM `MMELON` Plugin & Active-Learning DMTA** | `screening/encoders.py`, `screening/screener.py`, `screening/active_learning.py`, `training/contrastive_selectivity.py` | Exposes the trained `PaiNN-SE3` + `DrugCLIP` + `pKd` heads (`0.02–0.04 ms/query`), multi-conformation pocket ensembling, contrastive off-target vector repulsion, Parquet/CSV streaming `float16` `LigandVectorIndex`, catalog grounding, optional IBM Research `MMELON` (`ibm-research/biomed.sm.mv-te-84m`) plugin, hard-negative paralog selectivity loss, and `ActiveLearningDMTA` (UCB / Thompson / MaxMin-Diverse-UCB + wet-lab CSV ingestion). |
| **Phase 3 — Makeable Synthesis, Avoid-ome MMP & `OpenFE`/`Boltz-2` Interop** | `synthesis/reactions.py`, `leadopt/mmp.py`, `interop/openfe_export.py`, `interop/boltz_adapter.py` | 16 robust parallel MedChem reactions (Hartenfeller/Enamine REAL SMIRKS) with explicit building-block provenance + 1-step retrosynthesis registry (`AiZynthFinder` hook), MCS R-group SAR + activity-cliff detector, Avoid-ome 1-cut bioisostere rescue engine, charge-matched (`Δq = 0`) ring-strain-clean **OpenFE RBFE** perturbation network exporter, and **Boltz-2/AF3 mmCIF/PDB** pose-audit bridge. |
| **Phase 4 — OpenADMET Conformal DMPK, Typed MCP Server & Packaging** | `cns/dmpk_regression.py`, `mcp_server.py`, `CITATION.cff`, `Dockerfile` | Continuous DMPK regression (`LogD`, `KSOL`, `HLM CLint`, solubility) with distribution-free 90% Split-Conformal prediction intervals, 6 SHA-256 provenance-stamped agentic DMTA tools (`invoke_dmta_tool` + optional MCP stdio server), CFF 1.2.0 citation metadata, and container/wheel packaging. |

---

## 2. Verified Quantitative Benchmarks (Apple M3 Pro MPS + CPU)

### 2.1 Phase 1 — Crystal-Calibrated Beyond-PoseBusters 7-Axis Audit ([`beyond_posebusters_audit.json`](../results/tables/beyond_posebusters_audit.json))

Using torsion-constrained MMFF94s relaxation (eliminating crystallographic `0.03–0.06 Å` bond-length jitter while locking exact 3D ring puckering & scanned dihedrals), we calibrated against **49 native experimental co-crystal complexes** (including `3.0–3.8 Å` cryo-EM GPCRs) and audited all **28 PoseBusters-passing AutoDock Vina CNS leads**:

| Structural Cohort | `n` | PoseBusters ↑ | Ring Strain (`≤ 3 kcal/mol`) ↑ | Torsion Strain (Relaxed Scan) ↑ | pH 7.4 Microstate ↑ | Beyond-PB Physical (Axes 1–4) ↑ | **PB-Passing Yet Beyond-PB Failed** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A: Native Experimental Co-Crystals** *(calibration)* | `49` | `89.8%` | `75.5%` | `73.5%` | `100.0%` | `49.0%` | — |
| **Arm B: Docked Human CNS Leads** *(AutoDock Vina)* | `28` | **`100.0%`** | `82.1%` | `75.0%` | `100.0%` | `60.7%` | **`11 / 28 (39.3%)`** |

- **Scientific finding:** Torsion-constrained relaxation lifted native crystal ring-strain specificity by **`+30.6 pp`** (`44.9% → 75.5%`) and relaxed torsion scans lifted crystal torsion specificity by **`+10.2 pp`** (`63.3% → 73.5%`), while exposing **`39.3%` (`11/28`)** of PoseBusters-passing docked leads carrying hidden `twist_boat` rings (e.g., `DAT_1`, `CHRM1_v4_1`, `CHRM4_1`, `GRM5_Gen_5`) or twisted conjugated torsions (e.g., `58°` twisted amide in `GAT1_Gen_5`, `56–59°` twisted esters in `GPR52_v4_1` / `MGLL_v4_1`).

---

### 2.2 Phase 2 — MIT Contrastive 3D Selectivity Screening & Active-Learning DMTA ([`phase2_screening_and_al_benchmark.json`](../results/tables/phase2_screening_and_al_benchmark.json))

Evaluated on strict Bemis–Murcko-deduplicated human ChEMBL bioactivity decks (1 representative per unique scaffold, zero scaffold duplication):

#### A. Zero-Shot Generic `DrugCLIP` vs. 30-Step Hard-Negative Paralog Selectivity Tuning

| Selectivity Task (Strict Scaffold-Disjoint) | Query Time (ms) | Single-Pocket `DrugCLIP` AUROC | Contrastive `DrugCLIP` AUROC (`EF5%`) | Neural $\Delta\widehat{\text{p}K_d}$ AUROC (`BEDROC20`) |
| :--- | :---: | :---: | :---: | :---: |
| `DRD1` vs. `DRD2` (`300` vs. `263` scafs) | `0.036` | `0.4181` | `0.3896` (`0.71×`) | `0.4869` (`0.651`) |
| `MAOB` vs. `MAOA` (`300` vs. `146` scafs) | `0.022` | `0.5094` | `0.5205` (`0.91×`) | `0.4333` (`0.590`) |
| `BACE1` vs. `BACE2` (`300` vs. `221` scafs) | `0.021` | **`0.6405`** | `0.6346` (`1.54×`) | `0.5561` (`0.723`) |
| `SERT` vs. `DAT` (`300` vs. `203` scafs) | `0.020` | `0.4331` | `0.4440` (`1.10×`) | **`0.5661`** (`0.692`) |
| **`DRD1` vs. `DRD2` — Strict 60/40 Held-Out Scaffold Split (`n_test = 226`):** | | | | |
| ↳ Zero-shot `v2` `DrugCLIP` (before tuning) | `0.02` | — | **`0.4166`** (`EF5 0.47×`, `BEDROC20 0.345`) | — |
| ↳ **After 30-step `compute_selectivity_contrastive_loss` (frozen `PaiNN-SE3`)** | `0.02` | — | **`0.7345` (`+0.318`!)** (`EF5 1.88×`, **`BEDROC20 0.919`**) | — |

- **Key scientific insight:** Generic PDBbind-trained `DrugCLIP` achieves only `0.39–0.64` close-paralog selectivity AUROC zero-shot (independently corroborating `SELECTBENCH`, Nov 2026), yet **30 steps of hard-negative selectivity tuning on the lightweight projection heads** lifts strict held-out scaffold AUROC **`0.4166 → 0.7345`** and **`BEDROC(20)` `0.345 → 0.919`** while keeping **`0.02 ms/query`** dense retrieval speed!

#### B. Pool-Based Active-Learning DMTA (`400` Unique `DRD1` Scaffolds, `8%` Assay Budget = `32` Compounds)

| Strategy | Top-25 Hits (`pChEMBL ≥ 9.06`) | Top-25 Recall ↑ | Enrichment vs. Random ↑ | Mean Acquired `pChEMBL` ↑ | Best `pChEMBL` Found ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `random` baseline | `0 / 25` | `0.0%` | `1.0×` | `7.142` | — |
| `ucb` (`beta = 1.2`) | `3 / 25` | `12.0%` | `3.0×` | `7.398` | `9.89` |
| **`maxmin_diverse_ucb` ★** | **`4 / 25`** | **`16.0%`** | **`4.0×`** | **`7.399`** | **`9.89`** |

---

### 2.3 Phases 3 & 4 — Makeable Synthesis, Avoid-ome MMP, OpenFE & OpenADMET Conformal DMPK ([`phase3_phase4_leadopt_dmpk_benchmark.json`](../results/tables/phase3_phase4_leadopt_dmpk_benchmark.json))

- **1-Step Makeable Reaction Enumeration & OpenFE RBFE Network:** Enumerated **`8/8`** MedChem-clean 1-step amide analogs around quinoline-2-acid (`DRD1`), embedded low-energy chair conformers passing `RingStrainFilter` (`0` skipped), and exported a **`6`-ligand, `7`-edge charge-matched (`Δq = 0`) Lomap RBFE perturbation graph** (`results/openfe_drd1_demo/`).
- **Avoid-ome 1-Cut Bioisostere MMP Rescues (`v4` 12-Endpoint Oracle):**
  - **(-)-Ibogaine:** Single-atom `phenyl_to_pyridine_aza` scan reduced `P(hERG)` by **`-0.207`** (`0.887 → 0.680`) and lifted `CNS MPO v2` from `4.51 → 5.40` (`+0.89`).
  - **Donepezil:** Single-atom `phenyl_to_pyridine_aza` scan lowered `P(hERG)` (`0.951 → 0.897`) and lifted `CNS MPO v2` from `5.02 → 5.38`.
- **OpenADMET-ExpansionRx & Biogen Continuous DMPK Regression (Strict 5-Fold Bemis–Murcko Scaffold CV, 90% Split-Conformal Intervals):**

| Continuous DMPK Dataset / Endpoint | `n` | Spearman $\rho$ ↑ | $R^2$ ↑ | MAE ↓ | **90% Conformal Coverage** *(target 90%)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`OpenADMET_LogD`** *(ExpansionRx)* | `1,000` | **`0.873`** | **`0.694`** | `0.441` | **`96.3%`** |
| **`OpenADMET_log10_KSOL`** *(ExpansionRx)* | `1,000` | **`0.743`** | **`0.576`** | `0.349` | **`88.4%`** |
| **`OpenADMET_log10_HLM_CLint`** *(ExpansionRx)* | `1,000` | **`0.614`** | `0.391` | `0.355` | **`93.7%`** |
| **`Biogen_log_HLM_CLint`** *(Fang et al. JCIM)* | `1,000` | **`0.608`** | `0.355` | `0.408` | **`91.1%`** |
| `Biogen_log_Solubility_pH68` *(Fang et al. JCIM)* | `1,000` | `0.332` | `0.106` | `0.251` | **`87.3%`** |

---

## 3. Reproducibility

```bash
.venv/bin/python scripts/run_beyond_posebusters_audit.py
.venv/bin/python scripts/run_selectivity_screening_benchmark.py
.venv/bin/python scripts/run_makeable_leadopt_and_openfe_demo.py
.venv/bin/pytest -q tests/test_pose_auditor.py tests/test_screening_and_al.py tests/test_synthesis_leadopt_interop.py tests/test_phase4_dmpk_and_mcp.py
```
