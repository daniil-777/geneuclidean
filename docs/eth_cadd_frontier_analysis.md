# Scientific Frontier Analysis & Library Extensions Aligned with ETH Zürich CADD (`cadd.ethz.ch` / MODLAB, 2024–2026)

> **Research Boundary:** All generated molecules, scaffold hops, docking poses, strain energies, and ADMET/Avoid-ome probabilities reported below are strictly ***in silico* computational drug-discovery hypotheses** for academic research. They are **not** experimentally synthesized compounds, validated drugs, or clinical recommendations.

---

## 1. Landscape Synthesis: ETH Zürich MODLAB (`cadd.ethz.ch`) 2024–2026 Pillars & Open Technical Gaps

Prof. Dr. Gisbert Schneider's Computer-Assisted Drug Design laboratory (**MODLAB**, D-BSSE, ETH Zürich, `cadd.ethz.ch`, GitHub `ETHmodlab`) has established four converging methodological frontiers across 2024–2026. Below is a technical mapping of each flagship publication, its remaining open methodological gap, and the concrete, tested module implemented in **`geneuclidean`** to extend it:

| ETH Zürich CADD Publication | Core Methodological Contribution | Open Technical Gap | GenEuclidean Extension Module |
| :--- | :--- | :--- | :--- |
| **`DRAGONFLY`** (Atz *et al.*, *Nat. Commun.* **2024**, `15:3408`, `ETHmodlab/dragonfly_gen`) + Model-Intrinsic CLM Scoring (`molecular_design_with_beam_search`) | 3D binding-site graph transformer (`GTNN` + `GMT` pooling) conditioning an LSTM SMILES/SELFIES decoder; prospective PPARγ hits. | Conditions on a **single static PDB snapshot** with no native off-target counter-pocket repulsion, and lacks a likelihood-free differential 3D selectivity score ($\Delta\text{PMI}_{3\text{D}}$) across receptor ensembles. | [`src/geneuclidean/sampling/pocket_steering.py`](../src/geneuclidean/sampling/pocket_steering.py) + `MoleculeDesigner.design_ensemble` & `score_pmi` |
| **`RingAnalyzer`** (Atz, Bo, Tosstorff, Minot, Guba, Rudolph, Schneider, Kramer & Kuhn, *ChemRxiv* **Jun 2026**) | Ring-centric 3D equivariant GNN trained on $\omega$B97XD/Def2TZVP DFT energies (`MAE 0.54 kcal/mol`) showing widespread high-energy saturated-ring strain artifacts in docked poses & 3D generators. | Standard pose checks (**PoseBusters**) miss local saturated-ring puckering strain (e.g., `twist_boat` rings trapped during rigid-ring Vina docking); drug-design pipelines lack a composable, backend-pluggable ring-strain filter/scorer. | [`src/geneuclidean/evaluation/ring_strain.py`](../src/geneuclidean/evaluation/ring_strain.py) (`analyze_ring_strain`, `compute_cremer_pople`, `RingStrainFilter`, `RingStrainScorer`, `RING_STRAIN_BACKEND_REGISTRY`) |
| **`COLIBRI`** (Pachon-Angona & Schneider, *ChemRxiv* **Sep 2026**, `10.26434/chemrxiv.15008337/v1`) & **`Learning from Nature`** (Schriefer Treiber & Schneider, *ChemRxiv* **Aug 2026**, `10.26434/chemrxiv.15007728/v1`) | Zero-shot interactome 3D pocket + 2D ligand graph transformer for natural-product scaffold hopping (**ibogaine → human SERT `SLC6A4`**; **mitragynine pseudoindoxyl → µ-opioid**). | Natural-product scaffold hops require simultaneous verification across **5 orthogonal axes**: topological novelty, $\Delta\text{SA}$ synthetic simplification, CNS MPO/pH 7.4 pharmacophore retention, **multi-endpoint Avoid-ome cardiotoxicity de-risking** (hERG QT & 5-HT$_{2\text{B}}$ valvulopathy), and 3D `Asp98` + ring-strain physics. | [`src/geneuclidean/cns/scaffold_hop.py`](../src/geneuclidean/cns/scaffold_hop.py) (`audit_scaffold_hop_campaign`, `IBOGAINE_SMILES`, `PAROXETINE_SMILES`) |
| **`AIM`** (Minot & Schneider, *NeurIPS* **2025**, `arXiv:2509.25955`) & **Neural MTL** (Allenspach, Hiss & Schneider, *Nat. Mach. Intell.* **2024**) | Adaptive Intervention for deep Multi-task learning: dynamic policy matrix mediating destructive inter-task gradient interference across molecular properties. | Joint 3D SBDD encoders (`PaiNN-SE3` trained simultaneously on generative `lm_nll`, binding `pkd`, `DrugCLIP`, and `props`) exhibit massive gradient-norm imbalances and cross-task conflicts that naive loss scalarization ignores. | [`src/geneuclidean/training/mtl_intervention.py`](../src/geneuclidean/training/mtl_intervention.py) (`GradientConflictMediator`, `AdaptiveInterventionPolicy`, `MTL_MEDIATOR_REGISTRY`) |

Additionally, GenEuclidean complements MODLAB's synthetic-tractability & chemistry-automation ecosystem (**`SURF`**, Nippa *et al.*, *Mol. Inf.* 2025; **Active Geometric Reaction Prediction**, Minot *et al.*, *Nat. Comput. Sci.* 2026; **LSFML**, *Nat. Chem.* 2024) via strict reagent/coupling-byproduct quarantine (`DCU = 0.0%`), pure-RDKit `SAFE` fragment decoration for modular SAR synthesis, and calibrated 12-endpoint Avoid-ome + 90% Split-Conformal uncertainty quantification.

---

## 2. Reproducible Quantitative Benchmarks (Apple M3 Pro MPS + AutoDock Vina v1.2.7)

All results below were generated end-to-end by [`scripts/run_eth_cadd_frontier_benchmarks.py`](../scripts/run_eth_cadd_frontier_benchmarks.py) (`194.86 s` wall time) and archived verbatim in [`results/tables/eth_cadd_frontier_benchmarks.json`](../results/tables/eth_cadd_frontier_benchmarks.json).

---

### Benchmark 1 — Saturated-Ring Conformational Strain Audit Across 28 Docked Human CNS Complexes *(Extending `RingAnalyzer` 2026)*

We audited all **28 PoseBusters-passing AutoDock Vina v1.2.7 docked human CNS complexes** (`web/public/data/complexes/*.json`; `19` complexes containing `23` distinct saturated ring systems) using exact Cremer–Pople puckering coordinates ($Q, \theta, \phi$) and isolated fused-ring-system `MMFF94s` conformational strain (retaining fused aromatic rings, exocyclic double bonds, ring-N acyl/sulfonyl conjugation, and exact 3D stereochemistry while freezing strictly endocyclic ring atoms during cap relaxation).

- **Overall Pass Rate ($\Delta E_{\text{ring}} \le 3.0\text{ kcal/mol}$):** **`82.1%`** (`23/28` complexes; **`73.7%`** = `14/19` among complexes containing saturated rings).
- **Ring-System Strain Distribution (`23` systems):** median **`0.495 kcal/mol`**, mean **`1.811 kcal/mol`**, max **`6.994 kcal/mol`**.
- **Cremer–Pople Pucker Breakdown:** `9` `chair`, `6` `half_chair_or_envelope`, **`4` `twist_boat` (100% flagged $> 5.2\text{ kcal/mol}$)**, `2` `envelope`, `1` `twist_half_chair`, `1` `non_canonical`.

| Docked CNS Complex (PDB) | Sat. Ring Systems | Max Ring Strain (kcal/mol) | Full-Ligand Strain (kcal/mol) | Cremer–Pople Pucker(s) | Status (`≤ 3.0`) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **`DAT_1` (`9JKI`)** | `1` | **`6.994`** | `26.988` | `twist_boat` | **FLAG** |
| **`CHRM1_v4_1` (`9UCP`)** | `2` | **`6.883`** | `42.659` | `twist_boat`, `half_chair_or_envelope` | **FLAG** |
| **`CHRM4_1` (`8FX5`)** | `1` | **`6.818`** | `24.869` | `twist_boat` | **FLAG** |
| **`GABRA5_Gen_2` (`8BHG`)** | `1` | **`5.332`** | `1.162` | `chair` (distorted) | **FLAG** |
| **`GRM5_Gen_5` (`4OO9`)** | `3` | **`5.242`** | `16.870` | `twist_boat`, `chair`, `chair` | **FLAG** |
| `DRD1_1` (`9LLG`, Flagship TS) | `1` | **`0.838`** | `0.483` | `chair` | PASS |
| `SERT_1` (`5I6X`) | `1` | **`1.318`** | `8.588` | `half_chair_or_envelope` | PASS |
| `BACE1_v4_1` (`6EQM`) | `2` | **`0.227`** | `1.843` | `chair`, `envelope` | PASS |
| `GAT1_Gen_5` (`7Y7Z`) | `1` | **`0.035`** | `33.227` | `chair` | PASS |

**Key Scientific Takeaways:**
1. **Direct Empirical Corroboration of `RingAnalyzer` (2026):** **`26.3%` (`5/19`)** of saturated-ring CNS leads that passed PoseBusters actually harbor high-energy ring conformations (`5.24–6.99 kcal/mol`), predominantly **`twist_boat` 6-rings (`4/5`)** trapped during conformer generation because Vina holds rings rigid.
2. **Orthogonality to Full-Ligand Strain:** `GABRA5_Gen_2` exhibits low full-ligand strain (`1.162 kcal/mol`) yet elevated local ring strain (`5.332 kcal/mol`), whereas `GAT1_Gen_5` exhibits `33.227 kcal/mol` full-ligand torsional strain with an unstrained `0.035 kcal/mol` chair ring — validating ring-centric decomposition over whole-molecule force-field strain alone.
3. **Drop-In DFT Extensibility:** `RING_STRAIN_BACKEND_REGISTRY` exposes `register_ring_strain_backend(...)` so a neural DFT model (such as MODLAB's $\omega$B97XD `RingAnalyzer`) can replace `mmff94s` inside `DesignPipeline` (`RingStrainFilter` / `RingStrainScorer`) with zero pipeline rewrites.

---

### Benchmark 2 — Multi-Conformation Pocket Ensemble, Contrastive Off-Target 3D CFG & Zero-Shot $\Delta\text{PMI}_{3\text{D}}$ *(Extending `DRAGONFLY` 2024)*

Using trained `v4.0` checkpoints (`SAFE-GPT-87M` and `GP-MoLFormer-47M`, zero retraining) on `N = 64` samples/arm, we evaluated KV-cached multi-conformation ensemble conditioning (`DRD1` across 3 experimental cryo-EM conformations: `9LLG`, `9U5E`, `7LJC`) and Contrastive Off-Target 3D CFG:

$$\ell_{\text{guided}} = \ell_\varnothing + w_{\text{on}}\left(\frac{1}{K}\sum_{k=1}^{K}\ell(P_{\text{on}}^{(k)}) - \ell_\varnothing\right) - w_{\text{off}}\left(\ell(P_{\text{off}}) - \ell_\varnothing\right)$$

scored by our Platt-calibrated 26-target ChEMBL QSAR ensemble (`v4_24target_qsar.joblib`):

#### 2A. Tourette Anti-EPS `DRD1` (`9LLG+9U5E+7LJC`) vs. Off-Target `DRD2` (`6CM4`)

| Backbone | Steering Mode (`w_on`, `w_off`) | Valid ↑ | `IntDiv1` ↑ | MPO v2 ↑ | `P(DRD1)` ↑ | `P(DRD2)` ↓ | **Selectivity Margin** (`ΔP`) ↑ | **% Selective** (`ΔP > 0`) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GP-MoLFormer-47M` | Single `9LLG` (`1.6`, `0.0`) | `92.2%` | `0.8531` | `4.952` | `0.0488` | `0.0998` | `-0.0510` | `37.3%` |
| `GP-MoLFormer-47M` | **3-Conf Ensemble (`1.6`, `0.0`)** | **`95.3%`** | **`0.8619`** | `5.154` | **`0.0561`** | **`0.0240`** (`4.2× ↓`) | **`+0.0321`** (`+0.0831`) | **`57.6%`** (`+20.3 pp`) |
| `GP-MoLFormer-47M` | Contrastive Ens (`1.8`, `0.6` vs `DRD2`) | `93.8%` | `0.8503` | **`5.345`** | `0.0373` | `0.0287` | `+0.0086` | `51.7%` |
| `SAFE-GPT-87M` | Single `9LLG` (`1.6`, `0.0`) | `100.0%` | `0.8564` | **`5.309`** | **`0.0492`** | `0.0319` | `+0.0173` | `50.8%` |
| `SAFE-GPT-87M` | 3-Conf Ensemble (`1.6`, `0.0`) | `100.0%` | `0.8557` | `5.293` | `0.0414` | `0.0371` | `+0.0043` | **`66.1%`** (`+15.3 pp`) |
| `SAFE-GPT-87M` | **Contrastive Ens (`1.8`, `0.6` vs `DRD2`)** | **`100.0%`** | **`0.8646`** | `5.246` | `0.0445` | **`0.0264`** (`-17%`) | **`+0.0181`** | `57.1%` |

#### 2B. Alzheimer's Isozyme Selectivity `MAOB` (`2V5Z+FAD`) vs. Off-Target `MAOA` (`2Z5X+FAD`)

| Backbone | Steering Mode | Valid ↑ | Scaffolds ↑ | `P(MAOB)` ↑ | `P(MAOA)` ↓ | **Selectivity Margin** (`ΔP`) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `SAFE-GPT-87M` | Single `2V5Z` (`1.6`, `0.0`) | `96.9%` | `55` | `0.3064` | `0.3361` | `-0.0296` |
| `SAFE-GPT-87M` | **Contrastive (`1.8`, `0.6` vs `MAOA`)** | **`98.4%`** | `51` | **`0.3188`** | **`0.3215`** | **`-0.0027`** (`+0.0269`) |
| `GP-MoLFormer-47M` | Single `2V5Z` (`1.6`, `0.0`) | `96.9%` | `50` | **`0.3188`** | `0.3752` | `-0.0563` |
| `GP-MoLFormer-47M` | **Contrastive (`1.8`, `0.6` vs `MAOA`)** | **`98.4%`** | **`56`** | `0.1654` | **`0.1968`** (`-47.5%`) | **`-0.0314`** (`+0.0249`) |

#### 2C. Zero-Shot $\Delta\text{PMI}_{3\text{D}}$ Selectivity Discrimination (`score_pmi`) on Clinical Reference Drugs

| Clinical Reference Panel | `SAFE-GPT` $\Delta\text{PMI}_{3\text{D}}$ AUROC | `SAFE-GPT` Uncond. LL AUROC | `GP-MoLFormer` $\Delta\text{PMI}_{3\text{D}}$ AUROC | `GP-MoLFormer` Uncond. LL AUROC |
| :--- | :---: | :---: | :---: | :---: |
| `ACHE` vs. `DRD2` (`n = 6`) | **`1.0000`** | `0.5556` | `0.4444` | `0.4444` |
| `DRD1` vs. `ACHE` (`n = 5`) | **`0.6667`** | `0.3333` | `0.5000` | `0.6667` |
| `MAOB` vs. `DRD2` (`n = 6`) | `0.3333` | `0.6667` | **`0.6667`** | `0.5556` |

**Honest Methodological Analysis:**
- **Where Ensemble & Contrastive CFG Clearly Win:** Averaging 3 `DRD1` cryo-EM conformations eliminated `GP-MoLFormer`'s single-snapshot `DRD2` bias (`-0.0510 → +0.0321`, `4.2×` lower `P(DRD2)`), and contrastive off-target repulsion improved the selectivity margin on **both** `MAOB` vs. `MAOA` models (`+0.0269` SAFE-GPT; `+0.0249` GP-MoLFormer) and achieved `SAFE-GPT`'s highest `DRD1` margin (`+0.0181`) with zero retraining.
- **Known Tradeoff:** On high-identity homologs (`MAOB`/`MAOA` ~70% identity + shared `FAD`), aggressive `w_off = 0.6` on `GP-MoLFormer` suppressed both off-target `MAOA` (`-47.5%`) and on-target `MAOB` (`-48.1%`) together; similarly, zero-shot $\Delta\text{PMI}_{3\text{D}}$ exhibits strong backbone/tokenization dependence (`1.000` for `SAFE-GPT` on `ACHE` vs. `DRD2`, vs. `0.333–0.667` across other small `n=5–6` panels) and warrants larger external clinical benchmarks plus adaptive per-step `w_off` annealing.

---

### Benchmark 3 — Natural-Product Scaffold Hopping: (-)-Ibogaine → Human SERT (`5I6X` + `5I71`) with 12-Endpoint Avoid-ome De-Risking & Vina Docking *(Extending `COLIBRI` Sep 2026)*

We conditioned both `SAFE-GPT-87M` and `GP-MoLFormer-47M` on the 2-conformation human SERT ensemble (`5I6X` + `5I71`), audited `116` unique valid designs against **(-)-Ibogaine** (`PubChem CID 197060` / `RCSB IBG`, `InChIKey HSIBGVUMFOSJPD-CFDPKNGZSA-N`), and docked (-)-Ibogaine, **Paroxetine** (`5I6X` native SSRI control), and the top 8 clean basic-amine scaffold hops into human SERT (`5I6X`) with AutoDock Vina v1.2.7 + `Asp98` anchor check + PoseBusters + `RingStrainAnalyzer`:

- **Campaign Funnel (`N = 116` unique valid):** **`112/116` (`96.6%`)** clean exact Bemis–Murcko scaffold hops passing full MedChem triage (`mean ECFP4 Tanimoto to Ibogaine = 0.120`), **`113/116` (`97.4%`)** reducing `P(hERG)` by `> 0.15`, mean synthetic simplification **`ΔSA = +2.70`** (`4.91 → ~2.21`).
- **Automated Rediscovery of Ibogaine's Clinical Liabilities:** Our calibrated 12-endpoint Avoid-ome independently flagged (-)-Ibogaine's real-world cardiotoxic & metabolic profile: **`P(hERG) = 0.887`** (torsades/QT), **`P(CYP2D6) = 0.785`**, **`P(HLM) = 0.592`**, **`P(5-HT2B) = 0.558`**.

| Candidate / Control | Source | Vina Top-1 (kcal/mol) ↓ | `Asp98` Anchored (kcal/mol) | PoseBusters | Ring Strain (kcal/mol) / Pucker | `SA` ↓ (`ΔSA` ↑) | MPO v2 ↑ | `P(hERG)` ↓ (`Δ` ↑) | `P(5-HT2B)` ↓ | Tanimoto to Ibogaine |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(-)-Ibogaine** *(NP ref)* | Reference | `-8.44` | `-8.44` (Yes) | PASS | `5.854` (`boat`, FLAG) | `4.91` (`—`) | `4.51` | **`0.887`** (`—`) | `0.558` | `1.000` |
| **Paroxetine** *(SSRI ctrl)* | `5I6X` Co-xtal | **`-10.69`** | **`-10.69`** (Yes) | PASS | `0.268` (`chair`, PASS) | — | — | — | — | — |
| **`SERT_Hop_4` ★** | `GP-MoLFormer-v4` | **`-8.71`** | **`-8.35`** (Yes) | **PASS** | **`0.690`** (`chair`/`env`, **PASS**) | **`1.98`** (`+2.93`) | `5.03` | **`0.180`** (`-0.707`) | **`0.226`** | `0.143` |
| **`SERT_Hop_7` ★** | `GP-MoLFormer-v4` | **`-8.36`** | **`-8.36`** (Yes) | **PASS** | **`0.328`** (`chair`, **PASS**) | **`2.78`** (`+2.12`) | **`5.57`** | **`0.201`** (`-0.686`) | **`0.061`** | `0.122` |
| `SERT_Hop_3` | `GP-MoLFormer-v4` | `-7.53` | `-7.16` (Yes) | PASS | `0.689` (`chair`, PASS) | `2.23` (`+2.68`) | `4.85` | `0.141` (`-0.747`) | `0.300` | `0.155` |
| `SERT_Hop_1` | `SAFE-GPT-87M` | `-7.50` | `-6.74` (Yes) | PASS | `0.035` (`chair`, PASS) | `2.21` (`+2.70`) | **`5.85`** | `0.161` (`-0.727`) | `0.652` *(warn)* | `0.086` |
| `SERT_Hop_6` | `GP-MoLFormer-v4` | `-7.91` | No (`—`) | PASS | `0.432` (`chair`, PASS) | `1.87` (`+3.03`) | `4.69` | `0.180` (`-0.708`) | `0.481` | `0.171` |
| `SERT_Hop_2` *(strain flag)* | `GP-MoLFormer-v4` | `-7.40` | `-6.88` (Yes) | PASS | `5.203` (`chair`, **FLAG**) | `1.55` (`+3.36`) | `4.85` | `0.174` (`-0.713`) | `0.490` | `0.134` |
| `SERT_Hop_5` *(strain flag)* | `GP-MoLFormer-v4` | `-6.96` | `-6.96` (Yes) | PASS | `4.230` (`7-ring`, **FLAG**) | `3.25` (`+1.65`) | `5.67` | `0.063` (`-0.824`) | `0.365` | `0.111` |
| `SERT_Hop_8` | `GP-MoLFormer-v4` | `-6.39` | `-6.06` (Yes) | PASS | `0.000` (acyclic, PASS) | `2.51` (`+2.40`) | `4.60` | `0.047` (`-0.840`) | `0.473` | `0.068` |

- **Top *In Silico* Hypotheses:**
  - **`SERT_Hop_4`** (`CCN1CCN(C(=O)c2cccc3c2CCC3)CC1`) surpasses (-)-Ibogaine's Vina affinity (**`-8.71`** vs. `-8.44 kcal/mol`), preserves canonical `Asp98` engagement (`-8.35 kcal/mol`), passes both PoseBusters and `RingStrainAnalyzer` (`0.690 kcal/mol`, unstrained `chair`/`envelope`), simplifies synthesis by **`-2.93` SA units** (`4.91 → 1.98`), and cuts predicted **`P(hERG)` by `-79.7%`** (`0.887 → 0.180`) and **`P(5-HT2B)` by `-59.5%`** (`0.558 → 0.226`).
  - **`SERT_Hop_7`** (`CCN(CC(=O)N1CCC(C)(O)CC1)C(C)c1ccccc1`) matches Ibogaine's `Asp98`-anchored affinity (**`-8.36 kcal/mol`**), passes all geometry & ring-strain gates (`0.328 kcal/mol`), raises `CNS MPO v2` to **`5.57`**, and cuts `P(5-HT2B)` by **`-89.1%`** (`0.558 → 0.061`) alongside `-77.3%` lower `P(hERG)`.
- **Synergy of Integrated Gating:** Notice how `RingStrainAnalyzer` caught **2 PoseBusters-passing strained poses** (`SERT_Hop_2` `5.203 kcal/mol` and `SERT_Hop_5` `4.230 kcal/mol`) while `5-HT2B` Avoid-ome auditing caught `SERT_Hop_1` (`0.652`) despite its top `CNS MPO v2 = 5.85`.

---

### Benchmark 4 — 4-Task `PaiNN-SE3` Multi-Task Gradient Conflict Mediation *(Extending `AIM`, NeurIPS 2025)*

We instrumented `GenEuclideanModel` (`PaiNN-SE3` shared 3D encoder + 4 heads: `lm_nll`, `pkd`, `drugclip`, `props`) across 25 controlled steps comparing `scalarized`, `pcgrad`, and `aim_policy` (`AdaptiveInterventionPolicy` with learnable `W_policy` + lag-priority weighting):

- **Diagnosed 3D SBDD Gradient Pathology (Step 0, Identical Across Strategies):**
  - **`4/6` (`66.7%`)** task pairs conflict on the shared `PaiNN-SE3` encoder (`lm_nll` conflicts with all three biophysical heads: `cos = -0.035` vs `pkd`, `-0.089` vs `drugclip`, `-0.049` vs `props`).
  - **Gradient-Norm Imbalance Ratio:** **`8,434.5×`** (`‖g_lm‖ = 0.00315` vs `‖g_pkd‖ = 26.56`), explaining why unweighted joint training starves the generative language head relative to scalar regression heads!

| Strategy (`MTL_MEDIATOR_REGISTRY`) | Mean Conflict Frac ↓ | Mean Pairwise Cos ↑ | Final Norm Imbalance ↓ | Val `lm_nll` ↓ | Val `pkd` ↓ | Val `drugclip` ↓ | Val `props` ↓ | Train `lm_nll` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `scalarized` | `48.7%` | `+0.0090` | **`30.8×`** | `3.4602` | `43.0011` | **`2.7995`** | `0.0712` | `2.4081` |
| `pcgrad` | `46.7%` | `+0.0189` | `221.8×` | `3.4557` | `43.0431` | `3.0685` | **`0.0699`** | `2.4015` |
| **`aim_policy`** | **`43.3%`** | **`+0.0280`** | `119.4×` | **`3.4555`** | **`42.9377`** | `3.5652` | `0.0709` | **`2.3942`** |

- **Emergent Policy Behavior & Honest Tradeoff:** `aim_policy` reduced average conflict fraction (`48.7% → 43.3%`), tripled mean pairwise cosine (`+0.0090 → +0.0280`), automatically boosted the starved lagging `lm_nll` effective weight (`1.000 → 1.200`) while tapering fast-converging `props` (`0.200 → 0.143`), and achieved the lowest validation loss on both primary objectives (`lm_nll` and `pkd`). On this compact (`N = 24`) controlled diagnostic split, faster `drugclip` optimization (`train 2.71 → 1.41`) slightly overfit `val drugclip` (`2.80 → 3.57`), motivating validation-gated early stopping and full-scale PDBbind (`N = 4,543`) runs.

---

## 3. Verification & Reproducibility

```bash
# Run unit & contract tests for all 4 ETH-CADD extension modules
.venv/bin/pytest -q tests/test_eth_cadd_extensions.py

# Reproduce all 4 MPS + AutoDock Vina benchmarks (saves results/tables/eth_cadd_frontier_benchmarks.json)
HF_HOME=.hf_cache.nosync .venv/bin/python scripts/run_eth_cadd_frontier_benchmarks.py
```
