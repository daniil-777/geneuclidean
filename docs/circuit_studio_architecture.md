# GenEuclidean Circuit Studio — Architecture & Systems Specification (`CircuitGraphV1`)

> **Document Role:** Authoritative cross-stack systems specification bridging the **Web Studio Frontend** (`web/src/**`, owned by `@studio-ui-eng`) and the **Python Circuit Compiler & Local Studio Server** (`src/geneuclidean/studio/**` + `cli.py`, owned by `@studio-api-eng`), grounded 100% in real `geneuclidean` v4.0/v5.0/v5.1 library symbols.
>
> **Research Boundary:** All generated, screened, makeable, MMP-modified, or audited molecules produced in Browser Simulation or Local Python Engine mode are strictly *in silico* computational research hypotheses — never clinically tested medicines.

---

## 1. Executive Architecture & Dual-Mode Execution Philosophy

The **GenEuclidean Circuit Constructor** (`Circuit Studio` tab) transforms GenEuclidean's modular 3D geometric deep learning, chemical foundation LLMs, 7-axis pose physics, active-learning DMTA loops, makeable synthesis, Avoid-ome lead optimization, and OpenFE/MCP interoperability into a **typed electrical-circuit / computational-graph breadboard** built to Apple macOS Sequoia / Xcode / Origami design standards.

### Core Architectural Invariants
1. **100% Real Library Grounding:** Every block on the breadboard maps 1-to-1 to real classes/functions in `src/geneuclidean/` (zero fictional backbones or unimplemented stubs). Live-generated Python scripts use `import geneuclidean as ge`, parse cleanly with `ast.parse()`, and run directly in a local GenEuclidean environment.
2. **Strict Typed Signal Grammar:** 10 scientifically meaningful signal types prevent physically illegal connections (e.g., wiring raw 2D `MOL_BATCH` SMILES directly into a 3D `POSE_3D` geometry auditor) with gentle Apple-style remediation hints, while permitting explicit **`feedback` DMTA edges** for closed-loop Active Learning.
3. **Unified Dual-Mode Execution (`RunCircuitResponse`):**
   - **Browser Simulation Mode (Default on GitHub Pages — Zero Backend):** Executes the validated graph topologically (and iteratively across `dmtaRounds` for feedback loops) in TypeScript (`web/src/lib/circuitEngine.ts`), animating signal pulses across wires and populating real verified GenEuclidean molecules, 3D metrics, 12-endpoint Avoid-ome profiles, and benchmark-calibrated DMTA curves from `web/public/data/`.
   - **Local Python Engine Mode (`geneuclidean studio --port 8765`):** Sends the exact same `CircuitGraphV1` JSON to `http://127.0.0.1:8765/api/circuit/run`, executing real RDKit / 7-Axis PoseRealityAuditor / 16-Reaction synthesis / Avoid-ome MMP / Conformal DMPK / Active Learning / MCP / OpenFE Python code safely (zero `eval`/`exec`, bounded timeout, SHA-256 provenance digest) and streaming the identical `RunCircuitResponse` schema back to the UI.
4. **Cold-Import Discipline:** `src/geneuclidean/studio/` depends strictly on Python stdlib at module import time so `import geneuclidean` stays `< 15 ms` with zero eager `torch`, `transformers`, `sklearn`, or `rdkit` imports.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GenEuclidean Web Studio (React 19)                   │
│  [Left Block Dock: 36 Blocks / 8 Cats]  [Infinite Pan/Zoom Breadboard]  │
│  [Right Drawer: Inspector | Live Python SDK | Run Telemetry + Recharts] │
└───────────────────┬─────────────────────────────────┬───────────────────┘
                    │ Zero-Backend (GitHub Pages)     │ Optional Local HTTP
                    ▼                                 ▼ (127.0.0.1:8765)
     ┌──────────────────────────────┐  ┌──────────────────────────────────┐
     │ web/src/lib/circuitEngine.ts │  │ src/geneuclidean/studio/         │
     │ • validateCircuit()          │  │ • catalog.py (36 blocks)         │
     │ • generatePythonSdk()        │  │ • graph_compiler.py (AST + run)  │
     │ • simulateCircuit()          │  │ • server.py (ThreadingHTTPServer)│
     │   (uses web/public/data/*)   │  │   GET /api/health, /api/catalog  │
     └──────────────────────────────┘  │   POST /validate, /codegen, /run │
                                       └──────────────────┬───────────────┘
                                                          ▼
                                       ┌──────────────────────────────────┐
                                       │ Real geneuclidean v4/v5.0/v5.1   │
                                       │ models • sampling • screening    │
                                       │ active_learning • audit • cns    │
                                       │ synthesis • leadopt • interop    │
                                       └──────────────────────────────────┘
```

---

## 2. Section A — Typed Electrical-Circuit Signal & Port Grammar

### 2.1 The 10 Semantic Signal Types

Every port pin on every block carries exactly one `SignalType`. Colors are calibrated for **WCAG AA/AAA legibility** in both Obsidian Dark Mode (luminous accents on `#070B14`) and Apple Alabaster Light Mode (saturated Apple-System ink on `#F5F6F8`).

| `SignalType` | Display Label | Dark Hex | Light Hex | Wire Style | Python Runtime Type / Tensor Contract |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `POCKET_3D` | 3D Pocket | `#38BDF8` | `#0284C7` | Solid `2.25px` | `ge.Pocket \| Sequence[ge.Pocket]` (`.pdb`/`.cif`, cofactors, box, `[N, 3]`, `[N, 52]`) |
| `SE3_GRAPH` | SE(3) Graph | `#818CF8` | `#4F46E5` | Solid `2.25px` | `PocketEncoderOutput` (`node_emb [B,N,256]`, `pooled_emb [B,256]`, `vec_emb [B,N,64,3]`, `atom_mask`) |
| `LATENT_VEC` | Latent Vector | `#A78BFA` | `#7C3AED` | Solid `2.25px` | `NDArray[float16/32]` or `Tensor` (`[N, 128]` DrugCLIP / `[N, 768]` MMELON / `[B, 24, d_llm]` soft prefix) |
| `MOL_BATCH` | Molecules | `#34D399` | `#059669` | Solid `2.25px` | `list[str]` canonical SMILES / `DesignResult` / `list[ScreeningHit]` / `list[MakeableCandidate]` |
| `POSE_3D` | 3D Pose | `#2DD4BF` | `#0D9488` | Solid `2.25px` | `list[DockingPoseResult]` / `list[Chem.Mol]` (3D conformers + SDF molblocks + optional pocket PDB) |
| `ORACLE_SCORES` | Oracle Scores | `#FBBF24` | `#D97706` | Solid (or Dashed if `feedback`) | `dict[str, dict[str, float]]` (`vina_dg`, `pKd`, `cns_mpo_v2`, 12-ep Avoid-ome, `reality_score`, SAR) |
| `UNCERTAINTY_DIST` | Uncertainty Dist | `#F472B6` | `#DB2777` | Solid `2.25px` | `SurrogatePrediction` (`mean [N]`, `epistemic_std [N]`, `aleatoric_std [N]`, conformal `[lo, hi]`, `q_hat`) |
| `REAGENTS` | Reagents | `#FB923C` | `#EA580C` | Solid `2.25px` | `dict(core_smiles: str, building_blocks: list[str], reactions: tuple[str, ...])` |
| `RBFE_GRAPH` | RBFE Network | `#FB7185` | `#E11D48` | Solid `2.25px` | `OpenFENetworkBundle` (`rbfe_network.json`, `aligned_ligands_ph74.sdf`, `run_openfe_rbfe.py`, `Δq = 0`) |
| `PROVENANCE_REPORT` | Provenance | `#CBD5E1` | `#475569` | Solid `2.25px` | `PoseAuditReport` / `RGroupSARResult` / `CredibilityDossier` / `StoppingReport` / SHA-256 ledger dict |

### 2.2 Strict Connection Validation Rules (Identical in TS & Python)

Both `validateCircuit()` (TypeScript) and `validate_circuit()` (Python) MUST enforce the exact same 9 rules and return deterministic diagnostic codes:

1. **Block & Port Existence (`E_UNKNOWN_BLOCK`, `E_UNKNOWN_PORT`):** Every `node.blockType` must exist in the 36-block catalog, and every edge's `sourcePort` (must be an output port) and `targetPort` (must be an input port) must exist on its referenced node.
2. **No Self-Loops (`E_SELF_LOOP`):** `edge.source !== edge.target`.
3. **Strict Signal Type Equality (`E_PORT_TYPE_MISMATCH`):** `sourcePort.signalType === targetPort.signalType`, AND `edge.signalType` must match `sourcePort.signalType`. Zero implicit unsafe coercions.
4. **No Duplicate Wires (`E_DUPLICATE_EDGE`):** `(source, sourcePort, target, targetPort)` must be unique.
5. **Input Arity (`E_MULTI_INPUT_DISALLOWED`):** If `targetPort.multi === false`, at most one incoming edge may attach to `(target, targetPort)`. *(UI behavior: dropping a new compatible wire onto an occupied single-input pin smoothly replaces the prior wire in one undoable step; the API validator flags duplicates if malformed JSON is posted directly.)* Output ports always support fan-out (`multi: true`).
6. **Feedback Edge Eligibility (`E_FEEDBACK_NOT_ALLOWED`):** An edge with `kind: "feedback"` is legal **ONLY** if the target input port declares `acceptsFeedback: true` (see Table 2.3). Any `kind: "feedback"` wire attached to a non-feedback port is rejected.
7. **Combinational DAG Enforcement (`E_COMBINATIONAL_CYCLE`):** Let $G_{\text{fwd}} = (V, E_{\text{forward}})$ be the subgraph containing **only** edges with `kind === "forward"` (and non-disabled nodes). $G_{\text{fwd}}$ MUST be acyclic (verified via Kahn's topological sort). Any cycle in $E_{\text{forward}}$ fails validation with `E_COMBINATIONAL_CYCLE` and reports the exact cycle `nodeId`s. Conversely, cycles formed by including `kind === "feedback"` edges are **valid closed-loop DMTA circuits** executed for `execution.dmtaRounds` iterations.
8. **Required Inputs (`E_MISSING_REQUIRED_INPUT`):** On any enabled node with `required: true` input ports, at least one incoming wire must be attached (unless the graph is an empty Blank Breadboard, in which case `valid: true` with an `info` diagnostic `I_EMPTY_CIRCUIT` so empty breadboards don't show red errors!).
9. **Parameter Bounds (`E_PARAM_OUT_OF_RANGE`):** Numeric params must lie within `[min, max]`, enum params within `options`, and string params must pass basic length/sanitization checks (no control chars, max 2048 chars).
10. **Disconnected Node Advisory (`W_DISCONNECTED_NODE`, severity `"warning"`):** Enabled non-empty graphs with isolated nodes produce a non-blocking warning.

### 2.3 Feedback-Capable Input Ports (`acceptsFeedback: true`)

Exactly **6 input ports** across the catalog accept closed-loop `kind: "feedback"` wires:

| `blockType` | `portId` | `signalType` | Scientific Role in Closed-Loop DMTA |
| :--- | :--- | :--- | :--- |
| `tanimoto_gp` | `observations` | `ORACLE_SCORES` | Feeds Round $t$ Vina/Audit/Wet-lab assay scores back into the Tanimoto GP posterior for Round $t+1$. |
| `ensemble_surrogate` | `observations` | `ORACLE_SCORES` | Updates RF+HGB ensemble surrogate with Round $t$ assay labels. |
| `combinatorial_ts` | `feedback_scores` | `ORACLE_SCORES` | Updates reagent-level Thompson Sampling posteriors in $O(N_A + N_B)$. |
| `safe_gpt` | `feedback_scores` | `ORACLE_SCORES` | Iterative GDPO / reward-weighted re-steering of SAFE-GPT across DMTA rounds. |
| `chemical_llm` | `feedback_scores` | `ORACLE_SCORES` | Iterative DPO/GDPO alignment feedback for GP-MoLFormer / NovoMolGen. |
| `selectivity_mtl_aligner` | `feedback_scores` | `ORACLE_SCORES` | Feeds multi-objective / paralog oracle outcomes into hard-negative & MTL conflict mediation. |

> **Smart Auto-Feedback UX Rule:** When the user drags a compatible wire from `source` to `target` whose `targetPort.acceptsFeedback === true`, and adding that wire as `forward` would create a cycle in $G_{\text{fwd}}$, the UI **automatically promotes the wire to `kind: "feedback"`** and shows a brief glowing violet badge `"↻ Connected as DMTA Feedback Loop"` instead of rejecting it!

### 2.4 Gentle Apple-Style Illegal Wiring Remediation Hints

When a user hovers an incompatible socket while wiring (or posts an invalid edge to `/api/circuit/validate`), return a concise, actionable suggestion:

| Attempted Pair (`source` → `target`) | Remediation Hint |
| :--- | :--- |
| `POCKET_3D` → `SE3_GRAPH` | Insert **PaiNN-SE(3)**, **EGNN-3D**, or **E3NN Irreps** to encode the 3D pocket into an equivariant graph. |
| `MOL_BATCH` → `POSE_3D` | Insert **AutoDock Vina 3D** (or **Boltz-2 / AF3 Co-Fold**) to generate 3D ligand poses before geometry auditing. |
| `MOL_BATCH` → `LATENT_VEC` | Insert **DrugCLIP Encoder** or **IBM MMELON (84M)** to embed SMILES into L2-normalized vectors. |
| `REAGENTS` → `MOL_BATCH` | Insert **16-Reaction Makeable & Retro** or **Combinatorial Thompson Sampler** to couple reagents into products. |
| `ORACLE_SCORES` → `UNCERTAINTY_DIST` | Insert **Tanimoto-Kernel GP** or **Ensemble Surrogate** to fit a predictive posterior with epistemic uncertainty. |
| `POSE_3D` → `RBFE_GRAPH` | Wire `POSE_3D` or `MOL_BATCH` into **OpenFE RBFE Exporter** to build a charge-matched ($\Delta q = 0$) Lomap graph. |
| *Any other mismatch* | `Signal mismatch: <SRC> cannot connect to <DST>. Look for pulsing <SRC>-colored sockets.` |

---

## 3. Section B — Complete 36-Block Authoritative GenEuclidean Library Catalog

All **36 blocks** are organized into **8 clean categories**. Every block specifies its exact `blockType`, real Python API symbol, execution backend tier (`zero_torch` runs directly on CPU/RDKit with zero PyTorch; `optional_torch` uses real PyTorch/HF checkpoints when present and falls back cleanly to deterministic RDKit/verified surrogates; `v51_al` uses `geneuclidean.active_learning` when merged and falls back seamlessly to `geneuclidean.screening.active_learning.ActiveLearningDMTA`), typed ports, and real tunable hyperparameters.

### Category Summary (8 Categories, 36 Blocks)

| # | `categoryId` | Category Display Name | Accent Token | Block Count |
| :-: | :--- | :--- | :--- | :-: |
| 1 | `inputs` | Inputs & Structures | Sky `#38BDF8` | 5 |
| 2 | `encoders` | SE(3) / 3D Neural Encoders | Indigo `#818CF8` | 5 |
| 3 | `generators` | De Novo Generators & Steering | Emerald `#34D399` | 4 |
| 4 | `screening` | Virtual Screening & Retrieval | Violet `#A78BFA` | 4 |
| 5 | `active_learning` | Active Learning DMTA | Fuchsia `#F472B6` | 6 |
| 6 | `physics_audit` | 3D Physics & 7-Axis Pose Reality | Teal `#2DD4BF` | 4 |
| 7 | `leadopt_admet` | Lead-Opt, ADMET & Synthesis | Amber `#FBBF24` | 4 |
| 8 | `outputs` | Outputs & Open-Pharma Export | Rose `#FB7185` | 4 |

---

### Category 1 — Inputs & Structures (`inputs`, 5 blocks)

#### 1. `protein_pocket` — **Protein Pocket**
- **Real Python API:** `ge.Pocket.from_target(...)` / `ge.Pocket.from_pdb(...)` (`geneuclidean.sdk.pocket`)
- **Tier:** `zero_torch`
- **Summary:** Loads any of the 28 curated human CNS/Neuro 3D pockets (preserving catalytic cofactors `FAD`/`ADP`/`ZN`/`MG`) or a local `.pdb`/`.cif` structure.
- **Inputs:** *(none — source block)*
- **Outputs:** `pocket` (`POCKET_3D`, shape `PocketFeatures[N×3, N×52]`)
- **Parameters:**
  - `target_id` (`select`, default `"DRD1"`, options: all 28 targets `DRD1`, `DRD2`, `DRD3`, `VMAT2`, `ADRA2A`, `HRH3`, `PDE10A`, `CHRM4`, `HTR2A`, `SERT`, `TAAR1`, `DAT`, `ACHE`, `MAOB`, `NLRP3`, `OGA`, `GSK3B`, `BACE1`, `CHRM1`, `HTR1A`, `GLYT1`, `GPR52`, `MGLL`, `PDE1B`, `GABRA5`, `GAT1`, `ADORA2A`, `GRM5`)
  - `pocket_cutoff` (`number`, default `10.0`, min `6.0`, max `16.0`, step `0.5`, unit `"Å"`)
  - `max_atoms` (`number`, default `384`, min `128`, max `512`, step `32`)
  - `keep_cofactors` (`boolean`, default `true`, label `"Preserve Cofactors (FAD/ADP/Zn/Mg)"`)

#### 2. `multiconformer_ensemble` — **Multi-Conformer Ensemble**
- **Real Python API:** `[ge.Pocket.from_target(on_target_id, pdb_id=p) for p in pdb_ids]` + `ge.Pocket.from_target(off_target_id)` (`geneuclidean.sdk.pocket`)
- **Tier:** `zero_torch`
- **Summary:** Bundles $K$ experimental holo/apo conformations of an on-target pocket plus an optional off-target paralog counter-pocket (`DRD2`, `BCHE`, `MAOA`, `CHRM3`, `BACE2`, `DAT`) for contrastive selectivity steering.
- **Inputs:** *(none — source block)*
- **Outputs:** `on_pockets` (`POCKET_3D`, shape `Ensemble[K×Pocket]`), `off_pocket` (`POCKET_3D`, shape `CounterPocket`)
- **Parameters:**
  - `on_target_id` (`select`, default `"DRD1"`, options: `DRD1`, `ACHE`, `MAOB`, `BACE1`, `CHRM4`, `SERT`, `GRM5`, `GABRA5`)
  - `n_conformers` (`number`, default `2`, min `1`, max `4`, step `1`)
  - `off_target_id` (`select`, default `"DRD2"`, options: `DRD2`, `BCHE`, `MAOA`, `BACE2`, `CHRM3`, `DAT`, `NONE`)
  - `weighting` (`select`, default `"uniform"`, options: `uniform`, `resolution_weighted`)

#### 3. `reagent_library` — **Reagent Library**
- **Real Python API:** `geneuclidean.synthesis.ROBUST_MEDCHEM_REACTIONS` + building-block deck
- **Tier:** `zero_torch`
- **Summary:** Supplies a medicinal-chemistry core scaffold and curated Enamine-style coupling building blocks (`REAGENTS`) for makeable analog enumeration and combinatorial Thompson Sampling.
- **Inputs:** *(none — source block)*
- **Outputs:** `reagents` (`REAGENTS`, shape `Core + N_B Building Blocks`), `core_mols` (`MOL_BATCH`, shape `1 Core SMILES`)
- **Parameters:**
  - `core_smiles` (`string`, default `"O=C(O)c1ccc2ccccc2n1"`, label `"Core Scaffold SMILES"`)
  - `reagent_pack` (`select`, default `"cns_amines"`, options: `cns_amines`, `aryl_boronics`, `sulfonyl_chlorides`, `heteroaryl_halides`)
  - `max_reagents` (`number`, default `32`, min `4`, max `256`, step `4`)

#### 4. `boltz2_cofold` — **Boltz-2 / AF3 Co-Fold**
- **Real Python API:** `ge.load_cofolded_complex(...)` (`geneuclidean.interop.boltz_adapter`)
- **Tier:** `zero_torch`
- **Summary:** Parses Boltz-2, AlphaFold 3, or Chai-1 `.cif`/`.pdb` co-folded predictions, restores exact bond orders from SMILES templates, and emits aligned `POCKET_3D`, `POSE_3D`, and extracted `MOL_BATCH`.
- **Inputs:** *(none — source block)*
- **Outputs:** `pocket` (`POCKET_3D`), `pose` (`POSE_3D`, shape `CoFolded Complex`), `ligand_mols` (`MOL_BATCH`)
- **Parameters:**
  - `structure_source` (`select`, default `"ACHE_donepezil_cofold.cif"`, options: `ACHE_donepezil_cofold.cif`, `DRD1_ecopipam_cofold.cif`, `MAOB_safinamide_cofold.cif`, `CHRM4_xanomeline_cofold.cif`, `custom_path`)
  - `ligand_resname` (`string`, default `"LIG"`)
  - `reference_smiles` (`string`, default `"COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"`)
  - `pocket_cutoff` (`number`, default `10.0`, min `6.0`, max `15.0`, step `0.5`, unit `"Å"`)

#### 5. `chembl_pool` — **ChEMBL / ZINC Pool**
- **Real Python API:** `geneuclidean.data.splits.compute_murcko_scaffold` + Bemis–Murcko deduplicated bioactivity pool
- **Tier:** `zero_torch`
- **Summary:** Streams a Bemis–Murcko scaffold-deduplicated human ChEMBL 37 or ZINC-250k compound deck for zero-leakage virtual screening and Active Learning DMTA campaigns.
- **Inputs:** *(none — source block)*
- **Outputs:** `molecules` (`MOL_BATCH`, shape `Pool[N scaffolds]`)
- **Parameters:**
  - `dataset` (`select`, default `"DRD1_chembl37_400"`, options: `DRD1_chembl37_400`, `ACHE_chembl37_300`, `MAOB_chembl37_300`, `BACE1_chembl37_300`, `SERT_chembl37_300`, `ZINC250k_diverse`)
  - `pool_size` (`number`, default `400`, min `32`, max `5000`, step `32`)
  - `dedup_murcko` (`boolean`, default `true`, label `"Strict Bemis–Murcko Dedup"`)

---

### Category 2 — SE(3) / 3D Neural Encoders (`encoders`, 5 blocks)

#### 6. `painn_se3` — **PaiNN-SE(3)**
- **Real Python API:** `geneuclidean.models.encoders.PaiNNEncoder` (via `build_encoder(EncoderConfig(arch="painn", ...))`)
- **Tier:** `optional_torch`
- **Summary:** Chiral float32 $SE(3)$-equivariant Polarizable Atom Interaction Network ($SO(3)$ error $< 8\times 10^{-6}$, frozen backbone retaining CASF-2016 $pK_d\ r = 0.746$).
- **Inputs:** `pocket` (`POCKET_3D`, required: `true`)
- **Outputs:** `pocket_graph` (`SE3_GRAPH`, shape `[B, N, 256] + [B, N, 64, 3]`), `pocket_vec` (`LATENT_VEC`, shape `[B, 256]`)
- **Parameters:**
  - `num_layers` (`number`, default `4`, min `2`, max `8`, step `1`)
  - `hidden_dim` (`select`, default `192`, options: `128`, `192`, `256`, `384`)
  - `vector_dim` (`select`, default `64`, options: `32`, `64`, `128`)
  - `out_dim` (`select`, default `256`, options: `128`, `256`, `512`)
  - `num_rbf` (`number`, default `24`, min `8`, max `64`, step `8`)
  - `cutoff` (`number`, default `5.5`, min `3.5`, max `8.0`, step `0.5`, unit `"Å"`)
  - `max_neighbors` (`number`, default `24`, min `8`, max `48`, step `4`)
  - `use_se3_chirality` (`boolean`, default `true`, label `"SE(3) Chiral Cross-Product"`)

#### 7. `egnn_3d` — **EGNN-3D**
- **Real Python API:** `geneuclidean.models.encoders.EGNNEncoder`
- **Tier:** `optional_torch`
- **Summary:** $E(n)$-equivariant graph neural network with radial Bessel edge updates and coordinate-equivariant message passing.
- **Inputs:** `pocket` (`POCKET_3D`, required: `true`)
- **Outputs:** `pocket_graph` (`SE3_GRAPH`, shape `[B, N, 256]`), `pocket_vec` (`LATENT_VEC`, shape `[B, 256]`)
- **Parameters:**
  - `num_layers` (`number`, default `4`, min `2`, max `8`, step `1`)
  - `hidden_dim` (`select`, default `192`, options: `128`, `192`, `256`)
  - `out_dim` (`select`, default `256`, options: `128`, `256`, `512`)
  - `cutoff` (`number`, default `5.5`, min `3.5`, max `8.0`, step `0.5`, unit `"Å"`)
  - `max_neighbors` (`number`, default `24`, min `8`, max `48`, step `4`)

#### 8. `e3nn_irreps` — **E3NN Irreps**
- **Real Python API:** `geneuclidean.models.encoders.E3NNEncoder`
- **Tier:** `optional_torch`
- **Summary:** Full spherical-harmonics irreducible-representation (`o3.Irreps`) tensor-product 3D pocket encoder with `IrrepsEquivariantRMSNorm`.
- **Inputs:** `pocket` (`POCKET_3D`, required: `true`)
- **Outputs:** `pocket_graph` (`SE3_GRAPH`, shape `Irreps[l≤lmax] → 256D`), `pocket_vec` (`LATENT_VEC`, shape `[B, 256]`)
- **Parameters:**
  - `num_layers` (`number`, default `4`, min `2`, max `6`, step `1`)
  - `hidden_dim` (`select`, default `192`, options: `128`, `192`, `256`)
  - `e3nn_lmax` (`select`, default `1`, options: `1`, `2`, label `"Spherical Harmonic l_max"`)
  - `use_se3_chirality` (`boolean`, default `true`)
  - `cutoff` (`number`, default `5.5`, min `3.5`, max `8.0`, step `0.5`, unit `"Å"`)

#### 9. `drugclip_encoder` — **DrugCLIP Encoder**
- **Real Python API:** `ge.DrugCLIPLigandEncoder` / `ge.ECFP4BinaryEncoder` (`geneuclidean.screening.encoders`)
- **Tier:** `optional_torch` (`zero_torch` when `backend == "ecfp4_binary"`)
- **Summary:** Projects ligands into GenEuclidean's 3D-pocket-aligned unit hypersphere (`CASF R@5 = 62.1%`, `ROC-AUC = 0.937`) or pure-RDKit L2-normalized ECFP4 ablation baseline.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`), `pocket_graph` (`SE3_GRAPH`, required: `false`)
- **Outputs:** `embeddings` (`LATENT_VEC`, shape `[N, 128] float16`)
- **Parameters:**
  - `backend` (`select`, default `"drugclip_v2"`, options: `drugclip_v2`, `ecfp4_binary`)
  - `clip_dim` (`select`, default `128`, options: `128`, `256`, `1024`)
  - `batch_size` (`number`, default `2048`, min `64`, max `8192`, step `64`)

#### 10. `ibm_mmelon` — **IBM MMELON (84M)**
- **Real Python API:** `ge.IBMMMelonLigandEncoder` (`ibm-research/biomed.sm.mv-te-84m`, `geneuclidean.screening.encoders`)
- **Tier:** `optional_torch`
- **Summary:** Apache-2.0 IBM Research multi-view molecular foundation encoder (`84M` params, `768-D` unit embeddings) for hybrid 2D/SMILES + 3D screening.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`)
- **Outputs:** `embeddings` (`LATENT_VEC`, shape `[N, 768] float32`)
- **Parameters:**
  - `model_id` (`string`, default `"ibm-research/biomed.sm.mv-te-84m"`)
  - `batch_size` (`number`, default `64`, min `16`, max `256`, step `16`)
  - `device` (`select`, default `"auto"`, options: `auto`, `mps`, `cuda`, `cpu`)

---

### Category 3 — De Novo Generators & Steering (`generators`, 4 blocks)

#### 11. `safe_gpt` — **SAFE-GPT (87M)**
- **Real Python API:** `ge.MoleculeDesigner` + `PocketConditionedLLM` (`datamol-io/safe-gpt` + `PocketPerceiverResampler` + `PocketCVAE`, `geneuclidean.models.llm_bridge`)
- **Tier:** `optional_torch`
- **Summary:** Fragment-modular SAFE chemical transformer (`87.3M`, `24`-token Perceiver bridge, `5.5×` v4 Pharma-Safe, `0%` DCU) supporting both *de novo* generation and 3D clinical scaffold decoration.
- **Inputs:** `pocket_graph` (`SE3_GRAPH`, required: `true`), `seed_mols` (`MOL_BATCH`, required: `false`), `feedback_scores` (`ORACLE_SCORES`, required: `false`, `acceptsFeedback: true`)
- **Outputs:** `molecules` (`MOL_BATCH`, shape `Batch[n SMILES]`), `latent` (`LATENT_VEC`, shape `[B, 24, 768]`)
- **Parameters:**
  - `mode` (`select`, default `"de_novo"`, options: `de_novo`, `decorate_scaffold`)
  - `num_samples` (`number`, default `32`, min `4`, max `256`, step `4`)
  - `strategy` (`select`, default `"min_p"`, options: `min_p`, `top_p`, `top_k`, `typical`, `temperature`)
  - `cfg_scale` (`number`, default `1.6`, min `0.5`, max `3.0`, step `0.05`, label `"CFG Guidance w"`)
  - `temperature` (`number`, default `0.95`, min `0.5`, max `1.4`, step `0.05`)
  - `min_p` (`number`, default `0.08`, min `0.01`, max `0.30`, step `0.01`)
  - `num_queries` (`number`, default `16`, min `4`, max `32`, step `4`, label `"Perceiver Queries Q"`)
  - `core_smiles_with_dummy` (`string`, default `""`, label `"Core Scaffold ([*] for Decorate)"`)

#### 12. `chemical_llm` — **Chemical Foundation LLM**
- **Real Python API:** `ge.MoleculeDesigner` + `PocketConditionedLLM` / `GenEuclideanModel` (`geneuclidean.models.hf_backbones`, `geneuclidean.models.decoders.transformer`)
- **Tier:** `optional_torch`
- **Summary:** Pluggable 3D-pocket-conditioned SMILES generator supporting verified real backbones: `GP-MoLFormer-Uniq` (`46.8M`, `12.5× fast_4d_tril`), `NovoMolGen-157M` (`IntDiv1=0.892`, MPS SDPA), `NovoMolGen-300M`, `gpt2_zinc_87m`, and v2 `PocketTransformerDecoder`.
- **Inputs:** `pocket_graph` (`SE3_GRAPH`, required: `true`), `feedback_scores` (`ORACLE_SCORES`, required: `false`, `acceptsFeedback: true`)
- **Outputs:** `molecules` (`MOL_BATCH`, shape `Batch[n SMILES]`), `latent` (`LATENT_VEC`, shape `[B, 24, d_llm]`)
- **Parameters:**
  - `backbone_id` (`select`, default `"ibm-research/GP-MoLFormer-Uniq"`, options: `ibm-research/GP-MoLFormer-Uniq`, `chandar-lab/NovoMolGen_157M_SMILES_BPE`, `chandar-lab/NovoMolGen_300M_SMILES_BPE`, `entropy/gpt2_zinc_87m`, `v2-PocketTransformerDecoder`, `mock-tiny-gpt2`)
  - `num_samples` (`number`, default `32`, min `4`, max `256`, step `4`)
  - `strategy` (`select`, default `"min_p"`, options: `min_p`, `top_p`, `top_k`, `typical`, `diverse_beam`)
  - `cfg_scale` (`number`, default `1.35`, min `0.5`, max `3.0`, step `0.05`)
  - `temperature` (`number`, default `0.90`, min `0.5`, max `1.4`, step `0.05`)
  - `perceiver_layers` (`number`, default `2`, min `1`, max `4`, step `1`)

#### 13. `pocket_ensemble_cfg` — **Pocket Ensemble CFG**
- **Real Python API:** `ge.sample_contrastive_ensemble_llm` + `ge.score_pocket_pmi` (`geneuclidean.sampling.pocket_steering`)
- **Tier:** `optional_torch`
- **Summary:** Multi-conformation $K$-pocket logit ensembling with zero-retraining off-target repulsion ($\ell = \ell_\varnothing + w_{\text{on}}(\bar{\ell}_{\text{on}}-\ell_\varnothing) - w_{\text{off}}(\ell_{\text{off}}-\ell_\varnothing)$) and exact $\Delta\text{PMI}_{3\text{D}}$ specificity scoring.
- **Inputs:** `pocket_graph` (`SE3_GRAPH`, required: `true`), `off_pocket` (`POCKET_3D`, required: `false`)
- **Outputs:** `molecules` (`MOL_BATCH`, shape `Steered Batch[n]`), `pmi_scores` (`ORACLE_SCORES`, shape `PMI_3D / ΔPMI`)
- **Parameters:**
  - `backbone_id` (`select`, default `"datamol-io/safe-gpt"`, options: `datamol-io/safe-gpt`, `ibm-research/GP-MoLFormer-Uniq`, `chandar-lab/NovoMolGen_157M_SMILES_BPE`)
  - `w_on` (`number`, default `1.6`, min `0.5`, max `3.0`, step `0.05`, label `"On-Target Guidance w_on"`)
  - `w_off` (`number`, default `0.65`, min `0.0`, max `2.0`, step `0.05`, label `"Off-Target Repulsion w_off"`)
  - `num_samples` (`number`, default `32`, min `4`, max `128`, step `4`)
  - `compute_pmi_3d` (`boolean`, default `true`, label `"Compute ΔPMI_3D Selectivity"`)

#### 14. `selectivity_mtl_aligner` — **Selectivity & MTL Aligner**
- **Real Python API:** `compute_selectivity_contrastive_loss` (`training.contrastive_selectivity`) + `ge.GradientConflictMediator` (`training.mtl_intervention`) + `run_llm_gdpo` (`training.llm_trainer`)
- **Tier:** `optional_torch`
- **Summary:** Fine-tunes contrastive paralog selectivity margins (lifting strict held-out `DRD1 vs DRD2` AUROC `0.417 → 0.735`, `BEDROC20 0.919`) and resolves multi-task gradient conflicts via AIM Policy / PCGrad / 4-Ch GDPO.
- **Inputs:** `pocket_graph` (`SE3_GRAPH`, required: `true`), `off_pocket` (`POCKET_3D`, required: `false`), `feedback_scores` (`ORACLE_SCORES`, required: `false`, `acceptsFeedback: true`)
- **Outputs:** `aligned_vec` (`LATENT_VEC`, shape `Tuned Selectivity Head`), `diagnostics` (`PROVENANCE_REPORT`, shape `Grad Conflict + Margin`)
- **Parameters:**
  - `margin` (`number`, default `0.15`, min `0.05`, max `0.50`, step `0.05`, label `"Paralog Margin m"`)
  - `selectivity_weight` (`number`, default `0.50`, min `0.10`, max `2.00`, step `0.05`)
  - `mtl_strategy` (`select`, default `"aim_policy"`, options: `aim_policy`, `pcgrad`, `scalarized`)
  - `gdpo_beta` (`number`, default `0.04`, min `0.01`, max `0.50`, step `0.01`, label `"GDPO KL β"`)
  - `steps` (`number`, default `30`, min `10`, max `200`, step `10`)

---

### Category 4 — Virtual Screening & Retrieval (`screening`, 4 blocks)

#### 15. `ligand_vector_index` — **Ligand Vector Index**
- **Real Python API:** `ge.LigandVectorIndex.build_from_smiles` / `.build_from_file` (`geneuclidean.screening.screener`)
- **Tier:** `zero_torch`
- **Summary:** Streaming `float16` L2-normalized vector index supporting chunked BLAS cosine retrieval over Parquet/CSV/SMILES catalogs in bounded memory.
- **Inputs:** `embeddings` (`LATENT_VEC`, required: `true`), `molecules` (`MOL_BATCH`, required: `true`)
- **Outputs:** `index_vecs` (`LATENT_VEC`, shape `Float16 Index[N×D]`), `pool_mols` (`MOL_BATCH`)
- **Parameters:**
  - `chunk_size` (`number`, default `65536`, min `1024`, max `262144`, step `1024`)
  - `dtype` (`select`, default `"float16"`, options: `float16`, `float32`)
  - `mmap` (`boolean`, default `true`, label `"Memory-Mapped Index"`)

#### 16. `pocket_screener` — **Contrastive Pocket Screener**
- **Real Python API:** `ge.PocketScreener.screen(...)` (`geneuclidean.screening.screener`)
- **Tier:** `optional_torch`
- **Summary:** Millisecond (`0.02–0.04 ms/query`) multi-conformation 3D dense retrieval with optional off-target vector repulsion $s(m) = \bar{v}_{\text{on}}^\top u_m - \lambda\,\bar{v}_{\text{off}}^\top u_m$.
- **Inputs:** `on_pockets` (`POCKET_3D`, required: `true`), `off_pocket` (`POCKET_3D`, required: `false`), `index_vecs` (`LATENT_VEC`, required: `true`), `pool_mols` (`MOL_BATCH`, required: `true`)
- **Outputs:** `hits` (`MOL_BATCH`, shape `Top-K ScreeningHits`), `scores` (`ORACLE_SCORES`, shape `CLIP + ΔpKd`)
- **Parameters:**
  - `off_weight` (`number`, default `0.50`, min `0.0`, max `1.5`, step `0.05`, label `"Off-Target Repulsion λ"`)
  - `top_k` (`number`, default `50`, min `5`, max `500`, step `5`)
  - `rerank_pkd` (`boolean`, default `true`, label `"Stage-2 Neural pKd Rerank"`)
  - `triage_filter` (`boolean`, default `true`, label `"MedChem Hygiene Gate"`)

#### 17. `neural_pkd_reranker` — **Neural pKd & MaxMin Reranker**
- **Real Python API:** `PocketScreener.predict_pkd_batch` + `geneuclidean.sampling.rerank.rerank_and_select`
- **Tier:** `optional_torch` (`zero_torch` for pure MPO + MaxMin selection)
- **Summary:** Reranks candidate molecules by neural 3D $\widehat{\text{p}K_d}$ / MPO desirability and enforces Bemis–Murcko chemical diversity via greedy MaxMin Tanimoto selection.
- **Inputs:** `pocket_graph` (`SE3_GRAPH`, required: `false`), `molecules` (`MOL_BATCH`, required: `true`), `scores` (`ORACLE_SCORES`, required: `false`)
- **Outputs:** `ranked_mols` (`MOL_BATCH`, shape `Diverse Top-K`), `scores` (`ORACLE_SCORES`, shape `pKd + MPO`)
- **Parameters:**
  - `top_k` (`number`, default `16`, min `4`, max `128`, step `4`)
  - `min_tanimoto_dist` (`number`, default `0.35`, min `0.10`, max `0.70`, step `0.05`)
  - `quality_weight` (`number`, default `0.70`, min `0.20`, max `1.00`, step `0.05`)

#### 18. `catalog_grounding` — **Catalog Grounding**
- **Real Python API:** `PocketScreener.ground_to_catalog(...)` (`geneuclidean.screening.screener`)
- **Tier:** `zero_torch` (uses pre-indexed vectors + RDKit ECFP4)
- **Summary:** Maps *de novo* generated molecules onto nearest buyable/make-on-demand catalog analogs ranked jointly by 3D contrastive pocket score ($\alpha_{3D}$) and 2D ECFP4 Tanimoto similarity.
- **Inputs:** `query_mols` (`MOL_BATCH`, required: `true`), `index_vecs` (`LATENT_VEC`, required: `true`), `on_pockets` (`POCKET_3D`, required: `true`)
- **Outputs:** `grounded_mols` (`MOL_BATCH`, shape `Buyable Analogs`), `scores` (`ORACLE_SCORES`, shape `Tanimoto + 3D Score`)
- **Parameters:**
  - `top_k_per_query` (`number`, default `5`, min `1`, max `20`, step `1`)
  - `alpha_3d` (`number`, default `0.50`, min `0.0`, max `1.0`, step `0.05`, label `"3D vs 2D Blend α_3D"`)
  - `off_weight` (`number`, default `0.50`, min `0.0`, max `1.5`, step `0.05`)

---

### Category 5 — Active Learning DMTA (`active_learning`, 6 blocks)

> **v5.1 Integration Invariant:** All blocks in Category 5 target `geneuclidean.active_learning` (built in parallel by `@al-architect`) and include a clean import-time fallback to `geneuclidean.screening.active_learning.ActiveLearningDMTA` so `@studio-api-eng`'s isolated share-branch tests pass 100% before merge and activate full v5.1 kernels automatically post-merge.

#### 19. `tanimoto_gp` — **Tanimoto-Kernel GP**
- **Real Python API:** `geneuclidean.active_learning.TanimotoGP` (fallback: `ge.ActiveLearningDMTA`)
- **Tier:** `v51_al` (`zero_torch`)
- **Summary:** Exact Gaussian Process with analytical positive-semidefinite Tanimoto / MinMax fingerprint kernel and Cholesky decomposition yielding closed-form posterior mean and variance.
- **Inputs:** `pool_mols` (`MOL_BATCH`, required: `true`), `observations` (`ORACLE_SCORES`, required: `false`, `acceptsFeedback: true`)
- **Outputs:** `posterior` (`UNCERTAINTY_DIST`, shape `GP(μ, σ²_exact)`)
- **Parameters:**
  - `kernel` (`select`, default `"tanimoto"`, options: `tanimoto`, `minmax_count`)
  - `fp_bits` (`select`, default `2048`, options: `1024`, `2048`)
  - `noise_var` (`number`, default `0.05`, min `0.001`, max `1.0`, step `0.01`, label `"Observation Noise σ²_n"`)
  - `signal_var` (`number`, default `1.0`, min `0.1`, max `5.0`, step `0.1`, label `"Kernel Variance σ²_f"`)
  - `cold_start` (`select`, default `"butina_medoids"`, options: `butina_medoids`, `maxmin`, `random`)

#### 20. `ensemble_surrogate` — **Ensemble Surrogate**
- **Real Python API:** `geneuclidean.active_learning.EnsembleSurrogate` / `ge.ActiveLearningDMTA`
- **Tier:** `v51_al` (`zero_torch`)
- **Summary:** Bootstrap RandomForest + HistGradientBoosting ensemble surrogate with explicit epistemic vs. aleatoric uncertainty decomposition over ECFP4 + 2D descriptors.
- **Inputs:** `pool_mols` (`MOL_BATCH`, required: `true`), `observations` (`ORACLE_SCORES`, required: `false`, `acceptsFeedback: true`)
- **Outputs:** `posterior` (`UNCERTAINTY_DIST`, shape `Ensemble(μ, σ_ep, σ_al)`)
- **Parameters:**
  - `n_estimators` (`number`, default `64`, min `16`, max `256`, step `16`)
  - `fp_bits` (`select`, default `1024`, options: `1024`, `2048`)
  - `include_2d_desc` (`boolean`, default `true`, label `"Append 12 RDKit Descriptors"`)

#### 21. `conformal_calibrator` — **Split-Conformal UQ**
- **Real Python API:** `geneuclidean.active_learning.surrogates.ConformalSurrogate` / `geneuclidean.cns.credibility.SplitConformalBinaryCalibrator`
- **Tier:** `v51_al` (`zero_torch`)
- **Summary:** Wraps any surrogate posterior in distribution-free finite-sample Split-Conformal intervals ($\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil / n_{\text{cal}}$) supporting `homoscedastic`, `normalized` ($|y-\hat{\mu}|/\hat{\sigma}$), and `weighted_covariate_shift` (Tanimoto-AD Kish $N_{\text{eff}}$) modes under scaffold shift.
- **Inputs:** `posterior` (`UNCERTAINTY_DIST`, required: `true`), `calibration_scores` (`ORACLE_SCORES`, required: `false`)
- **Outputs:** `calibrated_uq` (`UNCERTAINTY_DIST`, shape `Conformal[lo, hi] @ (1-α)`)
- **Parameters:**
  - `mode` (`select`, default `"normalized"`, options: `normalized`, `homoscedastic`, `weighted_covariate_shift`)
  - `alpha` (`number`, default `0.10`, min `0.02`, max `0.30`, step `0.02`, label `"Miscoverage α (90% = 0.10)"`)
  - `scaffold_grouped` (`boolean`, default `true`, label `"Bemis–Murcko Cal Split"`)

#### 22. `batch_acquisition` — **Batch Acquisition**
- **Real Python API:** `geneuclidean.active_learning.acquisitions.ACQUISITION_REGISTRY` (fallback: `ActiveLearningDMTA.propose_batch`)
- **Tier:** `v51_al` (`zero_torch`)
- **Summary:** Selects diverse, uncertainty-aware batches across 12 acquisition strategies (`KrigingBeliever`, `DPP-UCB`, `MaxMin-UCB`, `qEHVI` Pareto, `ParEGO`, `Log-EI`, `Constrained-EI`, `Avoidome-Feasibility-UCB`, `UCB`, `EI`, `PI`, `Thompson`) with `PendingBatchManager` async masking.
- **Inputs:** `posterior` (`UNCERTAINTY_DIST`, required: `true`), `pool_mols` (`MOL_BATCH`, required: `true`), `constraints` (`ORACLE_SCORES`, required: `false`, `multi: true`)
- **Outputs:** `selected_batch` (`MOL_BATCH`, shape `Query Batch[B]`), `acq_scores` (`ORACLE_SCORES`, shape `Acq + Pareto HV`)
- **Parameters:**
  - `strategy` (`select`, default `"kriging_believer"`, options: `kriging_believer`, `dpp_ucb`, `maxmin_ucb`, `qehvi`, `pareto_scalarized_ucb`, `log_ei`, `constrained_ei`, `avoidome_feasibility_ucb`, `ucb`, `ei`, `pi`, `thompson`)
  - `batch_size` (`number`, default `12`, min `2`, max `64`, step `2`)
  - `ucb_beta` (`number`, default `1.5`, min `0.1`, max `4.0`, step `0.1`, label `"Exploration β"`)
  - `diversity_weight` (`number`, default `0.30`, min `0.0`, max `0.9`, step `0.05`)
  - `mask_pending_async` (`boolean`, default `true`, label `"Async In-Flight Batch Mask"`)

#### 23. `multifidelity_router` — **Multi-Fidelity Router**
- **Real Python API:** `geneuclidean.active_learning.multifidelity.MultiFidelityDMTA`
- **Tier:** `v51_al` (`zero_torch`)
- **Summary:** Cost-aware Kennedy–O'Hagan residual GP router across Tier-0 (fast surrogate / DrugCLIP), Tier-1 (AutoDock Vina + 7-Axis PoseRealityAuditor), and Tier-2 (OpenFE RBFE / wet-lab), with an empirical Spearman $\hat{\rho} < 0.30$ negative-transfer guardrail.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`), `posterior` (`UNCERTAINTY_DIST`, required: `true`)
- **Outputs:** `tier1_batch` (`MOL_BATCH`, shape `Tier-1 Vina/Audit`), `tier2_batch` (`MOL_BATCH`, shape `Tier-2 RBFE/Wet-Lab`), `routing_report` (`PROVENANCE_REPORT`, shape `Cost/VoI Ledger`)
- **Parameters:**
  - `tier1_budget` (`number`, default `12`, min `2`, max `64`, step `2`, label `"Tier-1 Vina Quota"`)
  - `tier2_budget` (`number`, default `4`, min `1`, max `16`, step `1`, label `"Tier-2 RBFE/Assay Quota"`)
  - `cost_ratio_t2_t1` (`number`, default `20.0`, min `2.0`, max `100.0`, step `2.0`, label `"Cost Ratio (Tier2 / Tier1)"`)
  - `min_spearman_rho` (`number`, default `0.30`, min `0.0`, max `0.80`, step `0.05`, label `"Negative-Transfer Guardrail ρ"`)

#### 24. `combinatorial_ts` — **Combinatorial Thompson Sampler**
- **Real Python API:** `geneuclidean.active_learning.CombinatorialThompsonSampler` + `ge.ROBUST_MEDCHEM_REACTIONS`
- **Tier:** `v51_al` (`zero_torch`)
- **Summary:** Explores $N_A \times N_B$ reaction-grounded Enamine-style virtual spaces in $O(N_A + N_B)$ oracle calls using reagent-level Bayesian Thompson posteriors with MedChem hygiene filtering.
- **Inputs:** `reagents` (`REAGENTS`, required: `true`), `feedback_scores` (`ORACLE_SCORES`, required: `false`, `acceptsFeedback: true`)
- **Outputs:** `sampled_mols` (`MOL_BATCH`, shape `Sampled Products`), `posterior` (`UNCERTAINTY_DIST`, shape `Reagent Posteriors`)
- **Parameters:**
  - `reaction_name` (`select`, default `"amide_coupling_acid_amine"`, options: `amide_coupling_acid_amine`, `sulfonamide_coupling`, `reductive_amination_aldehyde`, `suzuki_miyaura_biaryl`, `buchwald_hartwig_amination`, `snar_heteroaryl_fluoride`, `cuaac_triazole_click`, `urea_from_isocyanate`)
  - `batch_size` (`number`, default `16`, min `4`, max `64`, step `4`)
  - `warmup_per_reagent` (`number`, default `2`, min `1`, max `5`, step `1`)
  - `thermal_scale` (`number`, default `1.0`, min `0.2`, max `3.0`, step `0.1`, label `"Exploration Temp τ"`)
  - `triage_filter` (`boolean`, default `true`)

---

### Category 6 — 3D Physics & 7-Axis Pose Reality (`physics_audit`, 4 blocks)

#### 25. `protonation_ph74` — **pH 7.4 Protonation & Tautomer**
- **Real Python API:** `ge.protonate_smiles_ph74`, `ge.analyze_protonation_ph74`, `ge.audit_tautomer_microstate_ph74` (`geneuclidean.evaluation.protonation`, `tautomer_microstate`)
- **Tier:** `zero_torch`
- **Summary:** Ionizes basic amines / carboxylic acids at physiological `pH 7.4` (critical for GPCR `ASP3.32` salt bridges) and optionally audits minor tautomer stability gaps.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`)
- **Outputs:** `protonated_mols` (`MOL_BATCH`, shape `pH 7.4 SMILES`), `microstate_report` (`PROVENANCE_REPORT`, shape `Charge + Tautomer Audit`)
- **Parameters:**
  - `audit_tautomers` (`boolean`, default `true`, label `"Audit Tautomer Microstates"`)
  - `max_tautomers` (`number`, default `16`, min `4`, max `32`, step `4`)
  - `tautomer_gap_threshold` (`number`, default `2.0`, min `0.5`, max `5.0`, step `0.25`)
  - `strict_input_charge` (`boolean`, default `false`)

#### 26. `autodock_vina` — **AutoDock Vina 3D**
- **Real Python API:** `ge.dock_ligand_to_pocket` / `ge.dock_and_profile` (`geneuclidean.evaluation.docking_pipeline`, `sdk.docking`)
- **Tier:** `zero_torch` (uses local `bin/vina` v1.2.7 when installed; deterministic ETKDGv3/MMFF94s + verified docking surrogate fallback otherwise)
- **Summary:** Generates ring-diverse ETKDGv3 conformers, docks into the orthosteric box with AutoDock Vina v1.2.7 (held out from training), and profiles 6-channel 3D PLIF & canonical anchors.
- **Inputs:** `pocket` (`POCKET_3D`, required: `true`), `molecules` (`MOL_BATCH`, required: `true`)
- **Outputs:** `poses` (`POSE_3D`, shape `3D Docked SDFs`), `scores` (`ORACLE_SCORES`, shape `Vina ΔG, LE, PLIF`)
- **Parameters:**
  - `exhaustiveness` (`number`, default `8`, min `2`, max `32`, step `2`)
  - `num_modes` (`number`, default `9`, min `1`, max `20`, step `1`)
  - `box_mode` (`select`, default `"tight_per_axis"`, options: `tight_per_axis`, `cubic_standard`)
  - `max_ring_confs` (`number`, default `4`, min `1`, max `8`, step `1`)
  - `protonate_ph74` (`boolean`, default `true`)

#### 27. `ring_torsion_strain` — **Cremer–Pople & Torsion Strain**
- **Real Python API:** `ge.analyze_ring_strain`, `ge.compute_cremer_pople`, `ge.analyze_torsion_strain`, `ge.load_torsion_library_xml` (`geneuclidean.evaluation.ring_strain`, `torsion_strain`)
- **Tier:** `zero_torch`
- **Summary:** Catches the `39.3%` of PoseBusters-passing poses hiding high-energy `twist_boat` saturated rings (torsion-constrained Cremer–Pople MMFF94s) or strained conjugated dihedrals (Roche/ZBH SMARTS traffic-light + relaxed barriers).
- **Inputs:** `poses` (`POSE_3D`, required: `true`)
- **Outputs:** `clean_poses` (`POSE_3D`, shape `Low-Strain Poses`), `strain_scores` (`ORACLE_SCORES`, shape `Ring + Torsion kcal/mol`)
- **Parameters:**
  - `ring_strain_threshold_kcal` (`number`, default `3.0`, min `1.0`, max `8.0`, step `0.5`, unit `"kcal/mol"`)
  - `ring_n_ref_confs` (`number`, default `24`, min `8`, max `48`, step `8`)
  - `torsion_barrier_threshold_kcal` (`number`, default `2.5`, min `1.0`, max `6.0`, step `0.25`, unit `"kcal/mol"`)
  - `torsion_max_red` (`number`, default `0`, min `0`, max `3`, step `1`, label `"Max Red Torsions"`)
  - `torsion_backend` (`select`, default `"smarts_mmff94s"`, options: `smarts_mmff94s`, `roche_xml_plugin`)

#### 28. `pose_reality_auditor` — **7-Axis PoseRealityAuditor**
- **Real Python API:** `ge.PoseRealityAuditor` / `ge.audit_complex` (`geneuclidean.audit.pose_auditor`)
- **Tier:** `zero_torch`
- **Summary:** Unified Beyond-PoseBusters 7-axis physical & medicinal-chemistry auditor: (1) PoseBusters geometry/clashes, (2) Cremer–Pople ring strain, (3) relaxed torsion barriers, (4) pH 7.4 microstates, (5) 6-ch PLIF anchors, (6) PAINS/BRENK/Lilly/reagent hygiene, (7) therapeutic property profile.
- **Inputs:** `poses` (`POSE_3D`, required: `true`), `pocket` (`POCKET_3D`, required: `false`)
- **Outputs:** `verified_poses` (`POSE_3D`, shape `7-Axis Verified Poses`), `reality_scores` (`ORACLE_SCORES`, shape `reality_score [0–1]`), `audit_report` (`PROVENANCE_REPORT`, shape `7-Axis PoseAuditReport`)
- **Parameters:**
  - `property_profile` (`select`, default `"cns_mpo_v2"`, options: `cns_mpo_v2`, `oral_ro5`, `kinase_selectivity`, `beyond_ro5`)
  - `ring_strain_threshold_kcal` (`number`, default `3.0`, min `1.0`, max `6.0`, step `0.5`)
  - `torsion_barrier_threshold_kcal` (`number`, default `2.5`, min `1.0`, max `6.0`, step `0.25`)
  - `torsion_max_red` (`number`, default `0`, min `0`, max `2`, step `1`)
  - `strict_input_charge` (`boolean`, default `false`)

---

### Category 7 — Lead-Opt, ADMET & Synthesis (`leadopt_admet`, 4 blocks)

#### 29. `reaction_enumerator` — **16-Reaction Makeable & Retro**
- **Real Python API:** `ge.enumerate_makeable_analogs` + `ge.match_one_step_retrosynthesis` (`geneuclidean.synthesis.reactions`)
- **Tier:** `zero_torch`
- **Summary:** Enumerates 1-step parallelizable MedChem analogs across 16 Hartenfeller / Enamine REAL SMIRKS couplings with full reactant provenance and 1-step retrosynthesis disconnection verification (`AiZynthFinder`-pluggable).
- **Inputs:** `reagents` (`REAGENTS`, required: `false`), `molecules` (`MOL_BATCH`, required: `false`)
- **Outputs:** `makeable_mols` (`MOL_BATCH`, shape `MakeableCandidates`), `synth_report` (`PROVENANCE_REPORT`, shape `SMIRKS + Retro Routes`)
- **Parameters:**
  - `mode` (`select`, default `"enumerate_and_retro"`, options: `enumerate_and_retro`, `enumerate_only`, `retrosynthesis_only`)
  - `reaction_family` (`select`, default `"all_16"`, options: `all_16`, `amide_coupling_acid_amine`, `sulfonamide_coupling`, `reductive_amination_aldehyde`, `suzuki_miyaura_biaryl`, `buchwald_hartwig_amination`, `snar_heteroaryl_fluoride`, `cuaac_triazole_click`)
  - `max_products` (`number`, default `64`, min `8`, max `500`, step `8`)
  - `triage_filter` (`boolean`, default `true`, label `"Quarantine DCU/PAINS"`)

#### 30. `leadopt_mmp_sar` — **Avoid-ome MMP & R-Group Cliffs**
- **Real Python API:** `ge.AvoidomeMMPEngine` + `ge.analyze_rgroup_sar` (`geneuclidean.leadopt.mmp`)
- **Tier:** `zero_torch`
- **Summary:** Runs Roche `mmpdb`-inspired 1-cut bioisostere rescues (15 curated hERG/MDR1/5-HT2B transforms such as `piperidine→morpholine`, `phenyl→pyridine aza-scan`) AND MCS `R1×R2` R-group activity-cliff decomposition.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`), `scores` (`ORACLE_SCORES`, required: `false`)
- **Outputs:** `rescued_mols` (`MOL_BATCH`, shape `Ranked MMP Rescues`), `mmp_scores` (`ORACLE_SCORES`, shape `ΔhERG, ΔMPO, Rescue`), `sar_report` (`PROVENANCE_REPORT`, shape `R-Group Table + Cliffs`)
- **Parameters:**
  - `mode` (`select`, default `"mmp_and_cliffs"`, options: `mmp_and_cliffs`, `mmp_rescue_only`, `rgroup_cliffs_only`)
  - `property_profile` (`select`, default `"cns_mpo_v2"`, options: `cns_mpo_v2`, `oral_ro5`, `kinase_selectivity`, `beyond_ro5`)
  - `max_proposals` (`number`, default `20`, min `5`, max `50`, step `5`)
  - `cliff_threshold` (`number`, default `1.0`, min `0.3`, max `3.0`, step `0.1`, label `"Activity Cliff Δ"`)
  - `triage_filter` (`boolean`, default `true`)

#### 31. `cns_admet_suite` — **12-Ep Avoid-ome & Property Profile**
- **Real Python API:** `geneuclidean.cns.admet.CNSAdmetSuite` + `ge.triage_molecules` + `ge.evaluate_property_profile` (`geneuclidean.sdk.profiles`)
- **Tier:** `zero_torch` (uses trained joblib oracles if present, pure-RDKit MPO + Triage + PropertyProfile always live)
- **Summary:** Scores all 12 industrial scaffold-CV Avoid-ome/DMPK endpoints (`BBB`, `P-gp`, `hERG`, `MDR1`, `HLM`, `CYP2D6/3A4`, `5-HT2B`, `H1`, `DILI`, `AMES`, `M3`), Lilly/PAINS/reagent hygiene, and pluggable therapeutic property profiles.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`, `multi: true`)
- **Outputs:** `safe_mols` (`MOL_BATCH`, shape `Safety-Gated Leads`), `admet_scores` (`ORACLE_SCORES`, shape `12-Ep Avoid-ome + MPO`)
- **Parameters:**
  - `property_profile` (`select`, default `"cns_mpo_v2"`, options: `cns_mpo_v2`, `oral_ro5`, `kinase_selectivity`, `beyond_ro5`)
  - `admet_profile` (`select`, default `"v4_12ep"`, options: `v4_12ep`, `v3_core3`, `profile_only`)
  - `min_mpo_v2` (`number`, default `4.0`, min `2.0`, max `5.8`, step `0.2`)
  - `max_herg_prob` (`number`, default `0.30`, min `0.05`, max `0.80`, step `0.05`)
  - `require_triage_pass` (`boolean`, default `true`, label `"Enforce PAINS/DCU Quarantine"`)

#### 32. `conformal_dmpk` — **Conformal DMPK Regressor**
- **Real Python API:** `ge.ConformalDMPKRegressor` / `ge.evaluate_scaffold_cv_conformal_dmpk` (`geneuclidean.cns.dmpk_regression`)
- **Tier:** `zero_torch`
- **Summary:** OpenADMET-ExpansionRx & Biogen continuous pharmacokinetic regression (`LogD_7.4`, `log10 KSOL`, `log10 HLM CLint`, `Solubility`) with distribution-free Split-Conformal 90% prediction intervals.
- **Inputs:** `molecules` (`MOL_BATCH`, required: `true`)
- **Outputs:** `dmpk_intervals` (`UNCERTAINTY_DIST`, shape `ŷ ± q̂ [lo, hi]`), `dmpk_scores` (`ORACLE_SCORES`, shape `LogD / KSOL / HLM`)
- **Parameters:**
  - `endpoint` (`select`, default `"OpenADMET_LogD"`, options: `OpenADMET_LogD`, `OpenADMET_log10_KSOL`, `OpenADMET_log10_HLM_CLint`, `Biogen_log_HLM_CLint`, `Biogen_log_Solubility_pH68`)
  - `alpha` (`number`, default `0.10`, min `0.05`, max `0.25`, step `0.05`, label `"Miscoverage α"`)
  - `n_estimators` (`number`, default `80`, min `20`, max `160`, step `20`)

---

### Category 8 — Outputs & Open-Pharma Export (`outputs`, 4 blocks)

#### 33. `openfe_rbfe_exporter` — **OpenFE RBFE Exporter**
- **Real Python API:** `ge.export_openfe_rbfe_network(...)` (`geneuclidean.interop.openfe_export`)
- **Tier:** `zero_torch`
- **Summary:** Exports a charge-consistent ($\Delta q = 0$), ring-strain-clean, MCS-aligned Lomap RBFE perturbation graph (`rbfe_network.json`, `aligned_ligands_ph74.sdf`, `run_openfe_rbfe.py`) with cycle-closure edges.
- **Inputs:** `pocket` (`POCKET_3D`, required: `true`), `molecules` (`MOL_BATCH`, required: `false`), `poses` (`POSE_3D`, required: `false`)
- **Outputs:** `rbfe_network` (`RBFE_GRAPH`, shape `Lomap Graph + OpenFE Driver`)
- **Parameters:**
  - `require_same_charge` (`boolean`, default `true`, label `"Enforce Δq = 0 Guard"`)
  - `filter_ring_strain` (`boolean`, default `true`, label `"RingStrainFilter Pre-Check"`)
  - `max_ring_strain_kcal` (`number`, default `3.0`, min `1.0`, max `6.0`, step `0.5`)
  - `add_cycle_closures` (`boolean`, default `true`, label `"Add Hysteresis Cycle Closures"`)
  - `out_dir` (`string`, default `"results/openfe_circuit_export"`)

#### 34. `studio_3d_inspector` — **3D Studio Inspector**
- **Real Python API:** `ge.visualize_complex` / `ge.visualize_gallery` (`geneuclidean.sdk.viz`) + Web Studio 3Dmol.js deep-link
- **Tier:** `zero_torch`
- **Summary:** Renders interactive 3Dmol.js WebGL protein-ligand complexes with 6-channel PLIF contacts and residue attention heatmaps, providing a 1-click jump straight into GenEuclidean's `3D & 2D Studio` tab.
- **Inputs:** `pocket` (`POCKET_3D`, required: `true`), `poses` (`POSE_3D`, required: `false`), `scores` (`ORACLE_SCORES`, required: `false`), `audit_report` (`PROVENANCE_REPORT`, required: `false`)
- **Outputs:** `viz_manifest` (`PROVENANCE_REPORT`, shape `3D WebGL + PNG Spec`)
- **Parameters:**
  - `theme` (`select`, default `"dark"`, options: `dark`, `light`)
  - `show_native_ghost` (`boolean`, default `true`, label `"Overlay Native Co-Crystal"`)
  - `highlight_anchors` (`boolean`, default `true`, label `"Highlight Canonical Anchors"`)

#### 35. `mcp_tool_export` — **Agentic MCP Tool Server**
- **Real Python API:** `ge.invoke_dmta_tool`, `ge.DMTA_TOOL_SCHEMAS`, `ge.run_mcp_stdio_server` (`geneuclidean.mcp_server`)
- **Tier:** `zero_torch`
- **Summary:** Packages pipeline endpoints as deterministic, SHA-256 provenance-stamped Model Context Protocol (MCP) tools for autonomous pharma DMTA agents.
- **Inputs:** `audit_report` (`PROVENANCE_REPORT`, required: `false`, `multi: true`), `uncertainty` (`UNCERTAINTY_DIST`, required: `false`), `molecules` (`MOL_BATCH`, required: `false`)
- **Outputs:** `mcp_manifest` (`PROVENANCE_REPORT`, shape `6 Typed MCP Tools + SHA-256`)
- **Parameters:**
  - `tool_name` (`select`, default `"audit_complex"`, options: `score_molecules`, `evaluate_property_profile`, `audit_complex`, `enumerate_makeable_analogs`, `propose_mmp_rescues`, `match_retrosynthesis`)
  - `transport` (`select`, default `"python_direct"`, options: `python_direct`, `mcp_stdio`)

#### 36. `dmta_ledger` — **DMTA Ledger & Stopping Advisor**
- **Real Python API:** `ge.build_credibility_dossier` (`sdk.credibility`) + `geneuclidean.active_learning.diagnostics.StoppingAdvisor` & `geneuclidean.active_learning.engine.DMTALedger`
- **Tier:** `zero_torch` (`v51_al` enhanced when available)
- **Summary:** Seals every circuit execution or multi-round DMTA loop into a chained SHA-256 `DMTALedger` + FDA/EMA 10-principle `CredibilityDossier` + `StoppingAdvisor` (`evaluate_uq_calibration` ECE/NLL/$\rho$, EI saturation, top-$k$ Jaccard stability).
- **Inputs:** `scores` (`ORACLE_SCORES`, required: `false`, `multi: true`), `uncertainty` (`UNCERTAINTY_DIST`, required: `false`), `reports` (`PROVENANCE_REPORT`, required: `false`, `multi: true`)
- **Outputs:** `dossier` (`PROVENANCE_REPORT`, shape `SHA-256 Ledger + Stopping Verdict`)
- **Parameters:**
  - `ei_saturation_tol` (`number`, default `0.02`, min `0.005`, max `0.10`, step `0.005`, label `"EI Saturation Threshold"`)
  - `jaccard_stability_tol` (`number`, default `0.85`, min `0.50`, max `0.98`, step `0.05`, label `"Top-K Jaccard Stop"`)
  - `export_fda_ema_dossier` (`boolean`, default `true`, label `"Generate FDA/EMA 10/10 Dossier"`)

---

## 4. Section C — 5 Flagship One-Click End-to-End Presets

All 5 presets are pre-laid-out on a clean **4-column × 2-row** architectural grid (node width `224px`, column stride `292px`, row positions `y = 64` and `y = 284`) so every preset fits comfortably inside a `1440×900` viewport at `~0.76–0.90` zoom with **zero overlapping nodes and zero cluttered wire crossings**.

### Preset 1 (Default): `se3_denovo_cns` — **SE(3) Pocket-Conditioned De Novo Circuit**
- **Story:** End-to-end Tourette `DRD1` (`9LLG`) anti-EPS *de novo* discovery — chiral `PaiNN-SE(3)` $\to$ `SAFE-GPT-87M` $\to$ parallel `12-Ep Avoid-ome` + `AutoDock Vina v1.2.7` $\to$ `7-Axis PoseRealityAuditor` + `OpenFE RBFE (Δq=0)` $\to$ `3D Studio Inspector`.
- **Execution:** `dmtaRounds: 1`, `targetContext: "DRD1"`, `seed: 42`, `budget: 32`.
- **Nodes (8, strict Left→Right columns `x = 36, 308, 580, 852, 1124, 1396`):**
  - Col 0 (`x: 36`):
    - `n_pocket` (`protein_pocket`, `y: 170`, `{ target_id: "DRD1", pocket_cutoff: 10.0, max_atoms: 384, keep_cofactors: true }`)
  - Col 1 (`x: 308`):
    - `n_painn` (`painn_se3`, `y: 64`, `{ num_layers: 4, hidden_dim: 192, vector_dim: 64, out_dim: 256, num_rbf: 24, cutoff: 5.5, max_neighbors: 24, use_se3_chirality: true }`)
  - Col 2 (`x: 580`):
    - `n_safegpt` (`safe_gpt`, `y: 64`, `{ mode: "de_novo", num_samples: 32, strategy: "min_p", cfg_scale: 1.6, temperature: 0.95, min_p: 0.08, num_queries: 16, core_smiles_with_dummy: "" }`)
  - Col 3 (`x: 852`):
    - `n_admet` (`cns_admet_suite`, `y: 64`, `{ property_profile: "cns_mpo_v2", admet_profile: "v4_12ep", min_mpo_v2: 4.5, max_herg_prob: 0.30, require_triage_pass: true }`)
    - `n_vina` (`autodock_vina`, `y: 276`, `{ exhaustiveness: 8, num_modes: 9, box_mode: "tight_per_axis", max_ring_confs: 4, protonate_ph74: true }`)
  - Col 4 (`x: 1124`):
    - `n_openfe` (`openfe_rbfe_exporter`, `y: 64`, `{ require_same_charge: true, filter_ring_strain: true, max_ring_strain_kcal: 3.0, add_cycle_closures: true, out_dir: "results/openfe_drd1_denovo" }`)
    - `n_audit` (`pose_reality_auditor`, `y: 276`, `{ property_profile: "cns_mpo_v2", ring_strain_threshold_kcal: 3.0, torsion_barrier_threshold_kcal: 2.5, torsion_max_red: 0, strict_input_charge: false }`)
  - Col 5 (`x: 1396`):
    - `n_studio3d` (`studio_3d_inspector`, `y: 170`, `{ theme: "dark", show_native_ghost: true, highlight_anchors: true }`)
- **Edges (13 forward, 0 feedback — strictly monotonic Left→Right DAG):**
  - `e1`: `n_pocket.pocket` $\to$ `n_painn.pocket` (`POCKET_3D`)
  - `e2`: `n_painn.pocket_graph` $\to$ `n_safegpt.pocket_graph` (`SE3_GRAPH`)
  - `e3`: `n_safegpt.molecules` $\to$ `n_admet.molecules` (`MOL_BATCH`)
  - `e4`: `n_safegpt.molecules` $\to$ `n_vina.molecules` (`MOL_BATCH`)
  - `e5`: `n_pocket.pocket` $\to$ `n_vina.pocket` (`POCKET_3D`)
  - `e6`: `n_admet.safe_mols` $\to$ `n_openfe.molecules` (`MOL_BATCH`)
  - `e7`: `n_pocket.pocket` $\to$ `n_openfe.pocket` (`POCKET_3D`)
  - `e8`: `n_vina.poses` $\to$ `n_audit.poses` (`POSE_3D`)
  - `e9`: `n_pocket.pocket` $\to$ `n_audit.pocket` (`POCKET_3D`)
  - `e10`: `n_pocket.pocket` $\to$ `n_studio3d.pocket` (`POCKET_3D`)
  - `e11`: `n_audit.verified_poses` $\to$ `n_studio3d.poses` (`POSE_3D`)
  - `e12`: `n_admet.admet_scores` $\to$ `n_studio3d.scores` (`ORACLE_SCORES`)
  - `e13`: `n_audit.audit_report` $\to$ `n_studio3d.audit_report` (`PROVENANCE_REPORT`)

### Preset 2: `al_dmta_closed_loop` — **Multi-Fidelity Active Learning DMTA Closed Loop**
- **Story:** Closed-loop Lab-in-a-Loop over `400` Bemis–Murcko `DRD1` scaffolds using `TanimotoGP` $\to$ `Split-Conformal UQ` $\to$ `KrigingBeliever / qEHVI` $\to$ `MultiFidelityRouter` (Tier-1 Vina / Tier-2 OpenFE) with an animated **DMTA Feedback Arc** updating the GP each round + `DMTALedger & Stopping Advisor`!
- **Execution:** `dmtaRounds: 5`, `targetContext: "DRD1"`, `seed: 42`, `budget: 48`.
- **Nodes (8, strict Left→Right columns `x = 36, 308, 580, 852, 1124, 1396, 1668`):**
  - Col 0 (`x: 36`):
    - `n_pool` (`chembl_pool`, `y: 64`, `{ dataset: "DRD1_chembl37_400", pool_size: 400, dedup_murcko: true }`)
    - `n_pocket` (`protein_pocket`, `y: 280`, `{ target_id: "DRD1", pocket_cutoff: 10.0, max_atoms: 384, keep_cofactors: true }`)
  - Col 1 (`x: 308`):
    - `n_gp` (`tanimoto_gp`, `y: 64`, `{ kernel: "tanimoto", fp_bits: 2048, noise_var: 0.05, signal_var: 1.0, cold_start: "butina_medoids" }`)
  - Col 2 (`x: 580`):
    - `n_conf` (`conformal_calibrator`, `y: 64`, `{ mode: "normalized", alpha: 0.10, scaffold_grouped: true }`)
  - Col 3 (`x: 852`):
    - `n_acq` (`batch_acquisition`, `y: 64`, `{ strategy: "kriging_believer", batch_size: 12, ucb_beta: 1.5, diversity_weight: 0.30, mask_pending_async: true }`)
  - Col 4 (`x: 1124`):
    - `n_router` (`multifidelity_router`, `y: 170`, `{ tier1_budget: 12, tier2_budget: 4, cost_ratio_t2_t1: 20.0, min_spearman_rho: 0.30 }`)
  - Col 5 (`x: 1396`):
    - `n_vina` (`autodock_vina`, `y: 64`, `{ exhaustiveness: 8, num_modes: 9, box_mode: "tight_per_axis", max_ring_confs: 4, protonate_ph74: true }`)
  - Col 6 (`x: 1668`):
    - `n_ledger` (`dmta_ledger`, `y: 170`, `{ ei_saturation_tol: 0.02, jaccard_stability_tol: 0.85, export_fda_ema_dossier: true }`)
- **Edges (11 forward + 1 explicit `feedback` wire):**
  - `e1`: `n_pool.molecules` $\to$ `n_gp.pool_mols` (`MOL_BATCH`, `forward`)
  - `e2`: `n_pool.molecules` $\to$ `n_acq.pool_mols` (`MOL_BATCH`, `forward`)
  - `e3`: `n_gp.posterior` $\to$ `n_conf.posterior` (`UNCERTAINTY_DIST`, `forward`)
  - `e4`: `n_conf.calibrated_uq` $\to$ `n_acq.posterior` (`UNCERTAINTY_DIST`, `forward`)
  - `e5`: `n_conf.calibrated_uq` $\to$ `n_router.posterior` (`UNCERTAINTY_DIST`, `forward`)
  - `e6`: `n_acq.selected_batch` $\to$ `n_router.molecules` (`MOL_BATCH`, `forward`)
  - `e7`: `n_router.tier1_batch` $\to$ `n_vina.molecules` (`MOL_BATCH`, `forward`)
  - `e8`: `n_pocket.pocket` $\to$ `n_vina.pocket` (`POCKET_3D`, `forward`)
  - `e9`: `n_vina.scores` $\to$ `n_ledger.scores` (`ORACLE_SCORES`, `forward`)
  - `e10`: `n_conf.calibrated_uq` $\to$ `n_ledger.uncertainty` (`UNCERTAINTY_DIST`, `forward`)
  - `e11`: `n_router.routing_report` $\to$ `n_ledger.reports` (`PROVENANCE_REPORT`, `forward`)
  - **`e_fb_1` (FEEDBACK LOOP):** `n_vina.scores` $\to$ `n_gp.observations` (`ORACLE_SCORES`, **`kind: "feedback"`**, `label: "↻ DMTA Round t+1 Update"`)

### Preset 3: `paralog_selectivity_mmp` — **Paralog Selectivity & Avoid-ome MMP Rescue**
- **Story:** Multi-conformation `DRD1` vs. off-target `DRD2` contrastive CFG steering + 1-step makeable reaction chemistry + Roche-style Avoid-ome 1-cut bioisostere MMP rescue + 12-endpoint Avoid-ome + charge-matched OpenFE RBFE network.
- **Execution:** `dmtaRounds: 1`, `targetContext: "DRD1"`, `seed: 42`, `budget: 32`.
- **Nodes (8, strict Left→Right columns `x = 36, 308, 580, 852, 1124`):**
  - Col 0 (`x: 36`):
    - `n_ens` (`multiconformer_ensemble`, `y: 64`, `{ on_target_id: "DRD1", n_conformers: 2, off_target_id: "DRD2", weighting: "uniform" }`)
    - `n_reag` (`reagent_library`, `y: 280`, `{ core_smiles: "O=C(O)c1ccc2ccccc2n1", reagent_pack: "cns_amines", max_reagents: 32 }`)
  - Col 1 (`x: 308`):
    - `n_painn` (`painn_se3`, `y: 64`, `{ num_layers: 4, hidden_dim: 192, vector_dim: 64, out_dim: 256, num_rbf: 24, cutoff: 5.5, max_neighbors: 24, use_se3_chirality: true }`)
    - `n_rxn` (`reaction_enumerator`, `y: 280`, `{ mode: "enumerate_and_retro", reaction_family: "amide_coupling_acid_amine", max_products: 64, triage_filter: true }`)
  - Col 2 (`x: 580`):
    - `n_cfg` (`pocket_ensemble_cfg`, `y: 64`, `{ backbone_id: "datamol-io/safe-gpt", w_on: 1.6, w_off: 0.65, num_samples: 32, compute_pmi_3d: true }`)
    - `n_mmp` (`leadopt_mmp_sar`, `y: 280`, `{ mode: "mmp_and_cliffs", property_profile: "cns_mpo_v2", max_proposals: 20, cliff_threshold: 1.0, triage_filter: true }`)
  - Col 3 (`x: 852`):
    - `n_admet` (`cns_admet_suite`, `y: 64`, `{ property_profile: "cns_mpo_v2", admet_profile: "v4_12ep", min_mpo_v2: 4.5, max_herg_prob: 0.25, require_triage_pass: true }`)
  - Col 4 (`x: 1124`):
    - `n_openfe` (`openfe_rbfe_exporter`, `y: 170`, `{ require_same_charge: true, filter_ring_strain: true, max_ring_strain_kcal: 3.0, add_cycle_closures: true, out_dir: "results/openfe_drd1_selectivity" }`)
- **Edges (9 forward):**
  - `e1`: `n_ens.on_pockets` $\to$ `n_painn.pocket` (`POCKET_3D`)
  - `e2`: `n_painn.pocket_graph` $\to$ `n_cfg.pocket_graph` (`SE3_GRAPH`)
  - `e3`: `n_ens.off_pocket` $\to$ `n_cfg.off_pocket` (`POCKET_3D`)
  - `e4`: `n_reag.reagents` $\to$ `n_rxn.reagents` (`REAGENTS`)
  - `e5`: `n_rxn.makeable_mols` $\to$ `n_mmp.molecules` (`MOL_BATCH`)
  - `e6`: `n_cfg.molecules` $\to$ `n_admet.molecules` (`MOL_BATCH`)
  - `e7`: `n_mmp.rescued_mols` $\to$ `n_admet.molecules` (`MOL_BATCH`, `multi: true`)
  - `e8`: `n_admet.safe_mols` $\to$ `n_openfe.molecules` (`MOL_BATCH`)
  - `e9`: `n_ens.on_pockets` $\to$ `n_openfe.pocket` (`POCKET_3D`)

### Preset 4: `boltz2_audit_conformal_dmpk` — **Boltz-2 / AF3 7-Axis Audit & Conformal DMPK**
- **Story:** Zero-PyTorch industrial quality-assurance circuit auditing a Boltz-2 / AlphaFold 3 co-folded `ACHE` complex (`4EY7`) across Cremer–Pople ring strain, relaxed torsion barriers, 7-Axis PoseRealityAuditor, continuous 90% Split-Conformal DMPK (`LogD`), 3D Studio WebGL, and SHA-256 Agentic MCP tools.
- **Execution:** `dmtaRounds: 1`, `targetContext: "ACHE"`, `seed: 42`, `budget: 16`.
- **Nodes (7, strict Left→Right columns `x = 36, 320, 604, 888`):**
  - Col 0 (`x: 36`):
    - `n_boltz` (`boltz2_cofold`, `y: 170`, `{ structure_source: "ACHE_donepezil_cofold.cif", ligand_resname: "LIG", reference_smiles: "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", pocket_cutoff: 10.0 }`)
  - Col 1 (`x: 320`):
    - `n_strain` (`ring_torsion_strain`, `y: 64`, `{ ring_strain_threshold_kcal: 3.0, ring_n_ref_confs: 24, torsion_barrier_threshold_kcal: 2.5, torsion_max_red: 0, torsion_backend: "smarts_mmff94s" }`)
    - `n_prot` (`protonation_ph74`, `y: 280`, `{ audit_tautomers: true, max_tautomers: 16, tautomer_gap_threshold: 2.0, strict_input_charge: false }`)
  - Col 2 (`x: 604`):
    - `n_audit` (`pose_reality_auditor`, `y: 64`, `{ property_profile: "cns_mpo_v2", ring_strain_threshold_kcal: 3.0, torsion_barrier_threshold_kcal: 2.5, torsion_max_red: 0, strict_input_charge: false }`)
    - `n_dmpk` (`conformal_dmpk`, `y: 280`, `{ endpoint: "OpenADMET_LogD", alpha: 0.10, n_estimators: 80 }`)
  - Col 3 (`x: 888`):
    - `n_studio3d` (`studio_3d_inspector`, `y: 64`, `{ theme: "dark", show_native_ghost: true, highlight_anchors: true }`)
    - `n_mcp` (`mcp_tool_export`, `y: 280`, `{ tool_name: "audit_complex", transport: "python_direct" }`)
- **Edges (10 forward):**
  - `e1`: `n_boltz.pose` $\to$ `n_strain.poses` (`POSE_3D`)
  - `e2`: `n_strain.clean_poses` $\to$ `n_audit.poses` (`POSE_3D`)
  - `e3`: `n_boltz.pocket` $\to$ `n_audit.pocket` (`POCKET_3D`)
  - `e4`: `n_boltz.ligand_mols` $\to$ `n_prot.molecules` (`MOL_BATCH`)
  - `e5`: `n_prot.protonated_mols` $\to$ `n_dmpk.molecules` (`MOL_BATCH`)
  - `e6`: `n_boltz.pocket` $\to$ `n_studio3d.pocket` (`POCKET_3D`)
  - `e7`: `n_audit.verified_poses` $\to$ `n_studio3d.poses` (`POSE_3D`)
  - `e8`: `n_audit.audit_report` $\to$ `n_studio3d.audit_report` (`PROVENANCE_REPORT`)
  - `e9`: `n_audit.audit_report` $\to$ `n_mcp.audit_report` (`PROVENANCE_REPORT`)
  - `e10`: `n_dmpk.dmpk_intervals` $\to$ `n_mcp.uncertainty` (`UNCERTAINTY_DIST`)

### Preset 5: `blank_breadboard` — **Blank Breadboard**
- **Execution:** `dmtaRounds: 1`, `targetContext: "DRD1"`, `seed: 42`, `budget: 16`.
- **Nodes:** `[]`, **Edges:** `[]`.
- **Empty State UX:** Centered frosted glass pill on the dot-matrix canvas offering quick-start hints: *"Drag any GenEuclidean block from the left dock, click `+` to insert, or select a Flagship Preset above — including Combinatorial Reaction-Space Thompson Sampling (`ReagentLibrary → CombinatorialThompsonSampler → CNSAdmetSuite12`)."*

---

## 5. Section D — Shared `CircuitGraphV1` JSON Contract, Live Codegen & Local REST API

### 5.1 Canonical TypeScript & JSON Contract (`web/src/types/circuit.ts`)

Both `@studio-ui-eng` and `@studio-api-eng` MUST implement this exact schema byte-for-byte in JSON keys:

```ts
export type CircuitSchemaVersion = 'geneuclidean.circuit.v1';

export type SignalType =
  | 'POCKET_3D'
  | 'SE3_GRAPH'
  | 'LATENT_VEC'
  | 'MOL_BATCH'
  | 'POSE_3D'
  | 'ORACLE_SCORES'
  | 'UNCERTAINTY_DIST'
  | 'REAGENTS'
  | 'RBFE_GRAPH'
  | 'PROVENANCE_REPORT';

export type BlockCategoryId =
  | 'inputs'
  | 'encoders'
  | 'generators'
  | 'screening'
  | 'active_learning'
  | 'physics_audit'
  | 'leadopt_admet'
  | 'outputs';

export type EdgeKind = 'forward' | 'feedback';
export type ExecutionEngineMode = 'browser_sim' | 'local_python';

export interface PortSpec {
  id: string;
  label: string;
  signalType: SignalType;
  direction: 'input' | 'output';
  required?: boolean;         // input only; default false
  multi?: boolean;            // default false on inputs, true on outputs
  acceptsFeedback?: boolean;  // input only; default false
  shapeHint?: string;         // e.g. "[B, N, 256]"
}

export interface ParamSpec {
  key: string;
  label: string;
  type: 'number' | 'select' | 'boolean' | 'string';
  default: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: (string | number)[];
  description?: string;
}

export interface BlockDefinition {
  blockType: string;
  name: string;
  shortName: string;
  category: BlockCategoryId;
  pythonSymbol: string;       // e.g. "geneuclidean.models.encoders.PaiNNEncoder"
  executionTier: 'zero_torch' | 'optional_torch' | 'v51_al';
  badge: string;              // e.g. "SE(3) · 4L" or "NEW v5.1"
  summary: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
}

export interface CircuitNode {
  id: string;
  blockType: string;
  label?: string;
  position: { x: number; y: number };
  params: Record<string, number | string | boolean>;
  collapsed?: boolean;
  disabled?: boolean;
}

export interface CircuitEdge {
  id: string;
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
  signalType: SignalType;
  kind: EdgeKind;
  label?: string | null;
}

export interface CircuitGraphV1 {
  schemaVersion: CircuitSchemaVersion;
  id: string;
  name: string;
  description: string;
  metadata?: {
    presetId?: string;
    author?: string;
    createdAt?: string;
    tags?: string[];
  };
  execution: {
    mode: ExecutionEngineMode;
    seed: number;
    dmtaRounds: number;       // 1 for DAGs, 1..10 for feedback circuits
    targetContext: string;    // e.g. "DRD1", "ACHE"
    budget: number;           // e.g. 32
  };
  viewport?: { x: number; y: number; zoom: number };
  nodes: CircuitNode[];
  edges: CircuitEdge[];
}

export interface ValidationDiagnostic {
  severity: 'error' | 'warning' | 'info';
  code:
    | 'E_UNKNOWN_BLOCK'
    | 'E_UNKNOWN_PORT'
    | 'E_SELF_LOOP'
    | 'E_PORT_TYPE_MISMATCH'
    | 'E_DUPLICATE_EDGE'
    | 'E_MULTI_INPUT_DISALLOWED'
    | 'E_FEEDBACK_NOT_ALLOWED'
    | 'E_COMBINATIONAL_CYCLE'
    | 'E_MISSING_REQUIRED_INPUT'
    | 'E_PARAM_OUT_OF_RANGE'
    | 'W_DISCONNECTED_NODE'
    | 'I_EMPTY_CIRCUIT';
  message: string;
  remediation?: string;
  nodeId?: string;
  edgeId?: string;
  portId?: string;
}

export interface ValidationResult {
  valid: boolean;
  topologicalOrder: string[];
  feedbackEdges: string[];
  hasFeedbackLoop: boolean;
  diagnostics: ValidationDiagnostic[];
}

export interface CodegenResult {
  valid: boolean;
  pythonCode: string;
  imports: string[];
  astValid: boolean;
  provenanceSha256: string;
  diagnostics: ValidationDiagnostic[];
}

export interface ExecutionStepTelemetry {
  round: number;
  nodeId: string;
  blockType: string;
  status: 'ok' | 'simulated_fallback' | 'skipped' | 'error';
  durationMs: number;
  executionBackend: 'real_python' | 'deterministic_surrogate';
  inputShapes: Record<string, string>;
  outputShapes: Record<string, string>;
  metrics: Record<string, number | string | boolean | null>;
  summary: string;
}

export interface CircuitMoleculeOutput {
  id: string;
  name: string;
  smiles: string;
  target: string;
  sourceNodeId: string;
  metrics: {
    vina_dg?: number | null;
    cns_mpo_v2?: number | null;
    herg_prob?: number | null;
    bbb_prob?: number | null;
    mdr1_prob?: number | null;
    reality_score?: number | null;
    posebusters_passed?: boolean | null;
    pred_pkd?: number | null;
    conformal_lo?: number | null;
    conformal_hi?: number | null;
    reaction_or_transform?: string | null;
  };
}

export interface DmtaRoundPoint {
  round: number;
  nObserved: number;
  bestValue: number;
  meanAcquired: number;
  topHitRecall: number;
  conformalCoverage: number;
  eiMax: number;
  topKJaccard: number;
}

export interface RunCircuitResponse {
  status: 'completed' | 'failed' | 'validation_error';
  engineMode: ExecutionEngineMode;
  executionId: string;
  provenanceSha256: string;
  totalDurationMs: number;
  roundsCompleted: number;
  diagnostics: ValidationDiagnostic[];
  steps: ExecutionStepTelemetry[];
  molecules: CircuitMoleculeOutput[];
  dmtaTrace: DmtaRoundPoint[];
  provenanceLedger: {
    schema: 'geneuclidean_dmta_ledger_v1';
    graphSha256: string;
    resultSha256: string;
    timestampUtc: string;
    fdaEmaPrinciples: string;
  };
  generatedCode: string;
}
```

### 5.2 Local Python Studio HTTP API (`geneuclidean studio --host 127.0.0.1 --port 8765`)

Implemented in `src/geneuclidean/studio/server.py` using pure Python stdlib `http.server.ThreadingHTTPServer` (zero FastAPI/Uvicorn dependency!) and exposed in `src/geneuclidean/cli.py` via `@app.command() def studio(...)`.

| Method & Route | Request Body | Response JSON | HTTP Status |
| :--- | :--- | :--- | :---: |
| `OPTIONS /api/*` | *(CORS preflight)* | Empty + CORS headers (`Access-Control-Allow-Origin`, `GET, POST, OPTIONS`, `Content-Type`) | `204` |
| `GET /api/health` | — | `{ status: "ok", version, python, rdkit, torch, activeLearningV51, catalogCount: 36, uptimeSec }` | `200` |
| `GET /api/catalog` | — | `{ schemaVersion: "geneuclidean.circuit.v1", signalTypes, categories, blocks, presets }` | `200` |
| `POST /api/circuit/validate` | `CircuitGraphV1` | `ValidationResult` | `200` (`400` on invalid JSON) |
| `POST /api/circuit/codegen` | `CircuitGraphV1` | `CodegenResult` (verified with `ast.parse`) | `200` (`422` if circuit invalid) |
| `POST /api/circuit/run` | `{ circuit: CircuitGraphV1, options?: { dryRun?: bool, maxSamples?: int, timeoutSec?: int } }` | `RunCircuitResponse` (real local execution + SHA-256 provenance) | `200` (`422` if circuit invalid) |

### 5.3 Security, Safety & Live Codegen Guarantees (`src/geneuclidean/studio/graph_compiler.py`)
1. **Zero `eval` / `exec` / `subprocess(shell=True)`:** The execution engine NEVER evaluates client strings or runs generated Python code via `exec()`. Each `blockType` dispatches strictly to a typed, allowlisted Python handler function.
2. **Codegen Injection-Proofing:** All string parameters embedded into emitted Python scripts MUST be serialized via Python `repr()` (and in TypeScript via strict JSON string quoting) — preventing arbitrary code injection if a user enters malformed text in a SMILES or path field. Every Python-compiled script is verified with `ast.parse(code)` before returning (`astValid: true`).
3. **Path Confinement & Request Limits:** Max HTTP JSON payload `2 MB`. Any file path parameter (`structure_path`, `out_dir`) is rejected if it escapes the workspace root (`..` traversal guard) and falls back to safe temp/output directories during `/api/circuit/run`.
4. **Deterministic Seeded Execution & Bounded Runtime:** Execution enforces `options.timeoutSec` (default `30s`, max `60s`) and caps interactive sample batches (`maxSamples`, default `8`) so local studio runs finish smoothly in `< 1–4 s`.
5. **Graceful Checkpoint-Free Execution:** Zero-PyTorch RDKit/CADD blocks (`protonation_ph74`, `ring_torsion_strain`, `pose_reality_auditor`, `reaction_enumerator`, `leadopt_mmp_sar`, `cns_admet_suite`, `conformal_dmpk`, `openfe_rbfe_exporter`, `mcp_tool_export`, `dmta_ledger`, `tanimoto_gp`, `ensemble_surrogate`, `conformal_calibrator`, `batch_acquisition`, `multifidelity_router`, `combinatorial_ts`) execute REAL GenEuclidean Python functions. Neural checkpoint blocks (`painn_se3`, `safe_gpt`, `chemical_llm`, `pocket_screener`, `autodock_vina`) invoke real weights/Vina if locally available and otherwise emit `status: "simulated_fallback"` with verified RDKit-valid seed molecules for the chosen target — zero crashes on fresh clones!

### 5.4 Zero-Backend Deterministic Browser Simulation (`web/src/lib/circuitEngine.ts`)
- Uses a deterministic Mulberry32 PRNG keyed by `(circuit.execution.seed, graphSha256)` so runs are 100% reproducible.
- Draws real verified molecules, Vina $\Delta G$, CNS MPO v2, hERG/BBB/MDR1 probabilities, and 3D contact counts directly from the already-loaded `molecules_index.json` (28 human targets) and `v4_benchmarks.json` (4 SAFE-GPT clinical rescues + 12-endpoint Avoid-ome metrics + v4.0 3-LLM stats) — **never** inventing fake SMILES!
- Reflects real hyperparameter effects deterministically in telemetry (e.g., increasing `cfg_scale` trades diversity for higher target specificity; tightening `ring_strain_threshold_kcal` increases Beyond-PB attrition; switching `BatchAcquisition.strategy` from `greedy` to `kriging_believer` / `dpp_ucb` / `qehvi` improves Top-25 scaffold recall and Pareto hypervolume across DMTA rounds; enabling `require_same_charge` enforces `Δq = 0` edges in OpenFE output).
- Computes a real browser `SHA-256` digest via `crypto.subtle.digest` (with a synchronous deterministic FNV/SHA fallback if needed) so the Provenance Ledger is genuinely hash-verified even offline on GitHub Pages.

---

## 6. Section E — Apple Craftsmanship Visual, Interaction & UX Design Spec

### 6.1 Aesthetic Direction: macOS Sequoia × Xcode × Apple Freeform
- **Zero visual slop:** No neon rainbow gradients on cards, no cramped 10px unreadable micro-badges, no jagged wire elbows, no harsh borders. Every pixel obeys generous breathing room, crisp `1px` hairlines, frosted vibrancy glass, subtle layered depth shadows, and instant figure-ground separation.
- **Typography:** Uses the existing `--font-sans` (`Inter` / `-apple-system` / `SF Pro`) with tight display tracking (`-0.015em`) for headings and `--font-mono` (`JetBrains Mono` / `SF Mono`) with `tabular-nums` for all tensor shapes, hyperparameters, Vina $\Delta G$, and SHA-256 digests.

### 6.2 Dual-Theme Surface Tokens (`web/src/index.css` additions)

| Token | Obsidian Dark (`.dark`) | Apple Alabaster Light (`:root`) | Usage |
| :--- | :--- | :--- | :--- |
| `--circuit-canvas-bg` | `#070B14` | `#F5F6F8` | Breadboard viewport background |
| `--circuit-grid-dot` | `rgba(148, 163, 184, 0.16)` | `rgba(15, 23, 42, 0.12)` | Subtle `24px` dot-matrix grid |
| `--circuit-panel-glass` | `rgba(13, 19, 34, 0.80)` | `rgba(255, 255, 255, 0.86)` | Left dock, Right drawer, Floating toolbar (`backdrop-blur-2xl`) |
| `--circuit-node-bg` | `#0F172A` | `#FFFFFF` | Node chassis body (pure white on alabaster for crisp Apple figure-ground) |
| `--circuit-node-header` | `rgba(30, 41, 59, 0.65)` | `rgba(248, 250, 252, 0.92)` | Node top strip |
| `--circuit-node-border` | `rgba(148, 163, 184, 0.18)` | `rgba(15, 23, 42, 0.11)` | `1px` node hairline |
| `--circuit-node-shadow` | `0 12px 30px -10px rgba(0,0,0,0.6), 0 2px 6px rgba(0,0,0,0.4)` | `0 10px 26px -10px rgba(15,23,42,0.14), 0 1px 3px rgba(15,23,42,0.06)` | Soft multi-layer elevation |
| `--circuit-focus-ring` | `rgba(56, 189, 248, 0.65)` | `rgba(2, 132, 199, 0.55)` | Selected node / active wire glow |

### 6.3 3-Pane Workspace Layout (`1440×900` Precision Budget)

When `activeTab === 'circuit'`, `App.tsx` hides the outer `<footer>` so `CircuitBuilderView` cleanly occupies `h-[calc(100vh-56px)]` with zero outer page scrollbars:

1. **Top Contextual Bar (`h-12`, frosted glass, `1px` bottom hairline):**
   - **Left:** 5 Flagship Preset segmented pills (`SE(3) De Novo`, `AL DMTA Loop`, `Selectivity & MMP`, `Boltz-2 7-Axis`, `Blank`).
   - **Center:** Live Circuit Status capsule (`8 blocks · 11 wires · Valid` in emerald, or `1 DMTA Loop` in fuchsia, or clickable rose badge `2 issues` that focuses the first invalid node/wire).
   - **Right:** Engine Mode toggle (`Browser Sim` | `Local :8765` with live health indicator dot), Import/Export `.json` icon buttons, and primary Apple-crafted **`Run Circuit`** CTA (`from-cyan-500 to-emerald-500` in dark, deep crisp `#0284C7` / emerald accent in light — high contrast text!).
2. **Left Collapsible Block Dock (`w-[252px]`, collapses smoothly to `w-10` icon rail):**
   - Top search bar (`Search 36 blocks… ⌘K`) + category count pill.
   - 8 collapsible category sections with category accent dot, label, and block count.
   - Each block card displays: icon squircle, `shortName`, `badge` (e.g. `SE(3)`, `87M`, `v5.1`), miniature colored port-type dots (inputs on left, outputs on right), and a crisp `+` button (supports **both HTML5 drag-to-canvas AND 1-click smart placement** without overlapping existing nodes).
3. **Center Infinite Breadboard Canvas (`flex-1`, relative overflow-hidden):**
   - Pan via Space+drag, middle-mouse drag, or dragging empty canvas background. Cursor-centered wheel zoom (`40%–180%`).
   - **Node Chassis (`w-[224px]`):**
     - `3px` category-colored top cap with smooth corner clipping (`rounded-[14px]`).
     - Header (`px-3 py-2`): `24×24` tinted icon squircle, single-line title (`12.5px` semibold), status LED dot (idle / executing pulse / passed / fallback / error).
     - Subtitle (`px-3 pb-1.5`): `10px` monospace key parameter preview (e.g. `DRD1 · 9LLG · 10 Å` or `KrigingBeliever · B=12`).
     - Port Rail (`px-2.5 py-2`): Clean two-column row of hardware-tactile `11px` circular pins with `2px` outer bezel, signal-colored core, `10.5px` port label, `↻` indicator on feedback-capable inputs, magnetic `1.35×` hover scale, and pulsing compatibility halo during wiring!
     - Live Telemetry Pill Footer (visible after Run): `10px` monospace strip inside the card showing key metric + execution time (e.g. `✓ 32 mols · MPO 5.42 · 14 ms`).
   - **SVG Wire Layer:**
     - Forward wires: Smooth horizontal cubic Bézier curves with directional midpoint chevron, `2.25px` signal-colored stroke, subtle ambient drop-glow when selected or active, and `14px` transparent hit-target path for effortless selection.
     - DMTA Feedback wires (`kind === "feedback"`): Arched smooth return Bézier (`stroke-dasharray: 6 4`), elevated gracefully above the top row (`y_arc = min(y1, y2) - 82`), carrying a frosted glass pill badge `↻ DMTA Loop` at the apex.
     - Execution Wave Particles: When running, luminous `<circle r="4">` signal pulses animate smoothly along each wire in topological round order (`~180ms` per stage) so the user watches molecular data flow through the architecture!
     - Elastic Live Wire: Dragging (or click-start → click-end for trackpad friendliness!) from any output socket draws a smooth live Bézier to the cursor; hovering an incompatible input socket turns the wire amber/rose and displays a compact floating frosted hint pill near the cursor.
   - **Floating Bottom-Center Apple Pill Toolbar** (`backdrop-blur-2xl`, `rounded-2xl`, `1px` hairline):
     - `Zoom −` · `Zoom %` · `Zoom +` · `Fit View (0)` · `Auto-Layout (L)` | divider | `Undo (⌘Z)` · `Redo (⇧⌘Z)` | divider | `Duplicate (⌘D)` · `Delete (⌫)` · `Minimap toggle`.
   - **Floating Bottom-Left Interactive Minimap (`172×108px`, toggleable):** Frosted glass card rendering scaled node rects, wires, and a clickable/draggable viewport indicator box.
4. **Right Collapsible Drawer (`w-[348px]`, collapses to `w-10` rail) with 3 Apple Segmented Tabs:**
   - **Tab 1 — `Inspector`:**
     - Selected Node: Header + Real Python API pill (`geneuclidean.models.encoders.PaiNNEncoder`), 1-sentence scientific summary, typed Input/Output Port contract table with shape hints, full interactive hyperparameter controls (range slider + tabular readout, segmented option pills, dropdown selects, toggles, SMILES input), plus `Duplicate`, `Disable/Enable`, and `Delete` actions.
     - Selected Edge: Source → Target pill, Signal Type badge, Forward vs. DMTA Feedback indicator, `Delete Wire`.
     - Canvas Overview (nothing selected): Global Execution settings (`seed`, `dmtaRounds`, `targetContext`, `budget`), live topological stage count, and clickable validation diagnostics list.
   - **Tab 2 — `Python SDK`:**
     - Live-updating, syntax-highlighted executable Python script (`import geneuclidean as ge ...`) synchronized instantaneously with every node, wire, and slider change.
     - Header actions: `Copy`, `Download .py`, `Export .json`, and live badge (`AST Verified` / `SDK v5.1`).
   - **Tab 3 — `Telemetry`:**
     - Auto-activates (or flashes emerald badge) when **Run Circuit** finishes.
     - Top Run Card: Engine badge (`Browser Sim` vs `Local Python :8765`), total runtime, rounds completed, copyable SHA-256 provenance digest, and FDA/EMA `10/10` dossier badge.
     - **DMTA Convergence Chart (Recharts):** Automatically shown whenever `dmtaTrace.length > 1` — dual-series smooth line/area chart plotting Best Observed Activity / Top-Hit Recall and 90% Conformal Coverage across rounds!
     - **Clickable Step Execution Timeline:** Ordered list of executed blocks (clicking any step pans/highlights that node on the breadboard), showing input/output shapes, backend (`Real Python` vs `Browser Sim`), and key metrics.
     - **Output Molecule Gallery:** Compact cards for top designed/screened/rescued molecules with SMILES copy, Vina $\Delta G$, CNS MPO v2, P(hERG), 7-Axis Reality badge, and a 1-click **`Inspect in 3D Studio →`** button calling `onInspectIn3D(target)`!

### 6.4 Keyboard Shortcuts & Undo/Redo Stack
- **History Stack:** 50-step immutable snapshot stack (`past[]`, `present`, `future[]`) capturing node add/move-end/delete/duplicate/param-edit, wire add/delete, preset load, and auto-layout.
- **Shortcuts** (disabled gracefully when typing inside `<input>`/`<select>`/`<textarea>`):
  - `⌘/Ctrl + Z` — Undo; `⇧ + ⌘/Ctrl + Z` — Redo
  - `Delete` / `Backspace` — Delete selected node or wire
  - `⌘/Ctrl + D` — Duplicate selected node (`+32px, +32px` offset)
  - `⌘/Ctrl + K` — Focus Block Dock search bar
  - `0` — Fit-to-View; `L` — Auto-Layout (Sugiyama/Kahn layered left-to-right topological layout)
  - `⌘/Ctrl + Enter` — Run Circuit
  - `Escape` — Cancel active wiring or clear selection

### 6.5 Seamless Navigation Integration
- `web/src/types/studio.ts`: Add `'circuit'` to `StudioTab`.
- `web/src/components/Navbar.tsx`: Add `{ id: 'circuit', label: 'Circuit Studio', shortLabel: 'Circuit', icon: Workflow, badge: 'NEW' }` right after `3D & 2D Studio` (keeps all 7 nav items on one crisp line at `1440px` without wrapping).
- `web/src/App.tsx`: Mount `<CircuitBuilderView molecules={moleculesIdx.molecules} v4Benchmarks={v4Benchmarks} theme={theme} onInspectIn3D={(target) => handleNavigate('studio3d', target, 'All')} />` when `activeTab === 'circuit'`, and omit the global `<footer>` on the `'circuit'` tab.
- `web/src/views/OverviewView.tsx`: Add a clean hero CTA button `"Circuit Studio"` navigating directly to `'circuit'`.

---

## 7. Implementation Ownership Map

| Engineer | Strict File Ownership | Verification Gate |
| :--- | :--- | :--- |
| **`@studio-api-eng`** | `src/geneuclidean/studio/__init__.py`, `catalog.py`, `graph_compiler.py`, `server.py`; `src/geneuclidean/cli.py` (`studio` subcommand); `tests/test_studio_compiler_and_server.py` | `ruff check`, `ruff format --check`, `mypy src`, `pytest`, cold `import geneuclidean` `< 15 ms` (`heavy == []`), all 5 presets pass `validate` + `ast.parse` + live HTTP tests. |
| **`@studio-ui-eng`** | `web/src/types/studio.ts`, `web/src/types/circuit.ts`, `web/src/lib/circuitCatalog.ts`, `web/src/lib/circuitEngine.ts`, `web/src/components/circuit/**`, `web/src/views/CircuitBuilderView.tsx`, `web/src/components/Navbar.tsx`, `web/src/App.tsx`, `web/src/views/OverviewView.tsx`, `web/src/index.css` | `npm run typecheck && npm run build` (0 TS errors), zero console errors, 2+ visual QA polish passes with `@ui-qa-verifier` across Obsidian Dark & Apple Alabaster Light modes at `1440×900`. |
