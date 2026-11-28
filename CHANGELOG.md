# Changelog

All notable changes to **GenEuclidean** are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/) and follows [Keep a Changelog](https://keepachangelog.com/).

---

## Unreleased

- Added offline compound assays for solubility, finite-dose binding and precipitation, microsomal depletion and hepatic blood clearance. [Models and evidence](docs/compound_assays.md)
- Added optional strict fingerprint-distance selection to the SDK and pipelines. Seeded CVAE draws now follow the caller's generator. Dense encoder neighbor search uses query tiles and generation metrics operate on packed fingerprints.
- Corrected completed-candidate triage, the v4 ADMET pass flag, canonical scaffold grouping, conformal ranks and electric fields. Studio DMPK predictions use measured endpoint labels, and the decoding Pareto plot handles ties.
- Credibility dossiers leave each guiding principle unassessed until a reviewer supplies a verdict. Studio reports observed calibration coverage when the sample supports it.

## [5.3.0] — 2026-11-26

### Added
- **Foundational 3D Geometry (`geneuclidean.geom`):** 45 lazy-loaded, pure-`float32` `SE(3)`/`O(3)` primitives (`build_dense_knn_graph`, `gather_neighbors` with `30.1×` Metal `index_select`, rank-2 `IAS` irreps, Vector Neurons, condition-bounded SPD anisotropic 3D Gaussians & differentiable ROCS overlap, `ScreenedCoulombBias`, `GeoTransformerPairTripletBias`, `SE3ComplexCrossAttention`).
- **Four Next-Generation 3D Molecular Encoders (`8` total in `ENCODER_REGISTRY`):** `DensePaiNNSE3Encoder` (`dense_painn_se3`), `CartFormerSE3Encoder` (`cartformer_se3`), `AtomGSSE3Encoder` (`atomgs_se3`), and `SubPocketVNJEPAEncoder` (`subpocket_vn_jepa`), plus `3D JEPA` self-supervised pocket pretraining (`geneuclidean.training.jepa_ssl`).
- **Pure-PyTorch Chemical NLP Library (`geneuclidean.nlp`):** 43 lazy-loaded primitives including `AtomLevelSMILESTokenizer`, stateful `SMILESValenceGrammarMask`, `PocketConditionedChemicalLM` (RoPE/YaRN + GQA + SwiGLU + Perceiver/Cross-Attn/Prefix conditioning), pure-PyTorch `LoRALinear` (`inject_lora_adapters`/`merge_lora_adapters`), contrastive CFG, KV-cached grammar beam search, `dpo_preference_loss`, `grpo_policy_loss`, and `DiscreteMaskedDiffusionStep` (MDLM).
- **`NanoFold3` & Open-Weights Co-Folding Hub (`geneuclidean.folding`):** Trainable mini-AlphaFold3 (`PairformerStack` & Apple `SimpleFoldTrunk`, EDM & Flow-Matching `AllAtomDiffusionModule`, `KabschWeightedRigidAlign`, `CoFoldingLoss`, `ConfidenceHead`, `AffinityHead`) and unified 5-backend `FOLDING_BACKEND_REGISTRY` (`nanofold3`, `boltz2`, `protenix`, `alphafold3`, `esmfold_api`) via `fold_complex`.
- **49-Block Apple-Style Circuit Studio (`geneuclidean.studio` & `web/`):** Upgraded visual drug-design breadboard from 36 to **49 blocks**, **7 flagship presets** (adding `nanofold3_cofold_audit` and `tourette_scz_champion`), and **8 feedback ports**, with telemetry-aware PCB wire routing and zero-warning WebGL cleanup.
- **Realistic Tourette & Schizophrenia Campaign (`v5.3`):** Chair-seeded pH 7.4 conformer prep, AutoDock Vina v1.2.7, and 7-axis Beyond-PoseBusters + 12-ep Avoid-ome audit across 15 human TS/SCZ targets (`30/30` unique flagship leads pass all 7 axes + canonical 3D anchors; `96.7%` 1-step makeable).
- **Google Open-Source Library Standards & CI Enforcement:** Unified `5.3.0` metadata across `pyproject.toml`, `geneuclidean.__version__`, `CITATION.cff`, `Dockerfile`, and `web/package.json`; expanded top-level PEP 562 lazy facade (`geom`, `nlp`, `folding`, plus 15 flagship symbols); added `CHANGELOG.md`, `CONTRIBUTING.md`, and automated standards test suite (`tests/test_google_library_standards.py`).

### Fixed
- Eliminated the last 2 `mypy src` `[unused-ignore]` diagnostics in `src/geneuclidean/interop/boltz_adapter.py` so `mypy src` exits `0` across all 118 source modules.
- Fixed ground-truth coordinate leakage in `NanoFold3` flexible-atom masking and small-volume chiral margin scaling in `CoFoldingLoss`.
- Hardened batched prefix KV-cache `attention_mask` handling and dead-beam grammar termination in `geneuclidean.models.nlp_ops`.

---

## [5.1.0] — 2026-11-26

### Added
- **Frontier Active Learning DMTA (`geneuclidean.active_learning`):** Exact PSD `TanimotoGP` (binary/MinMax + MLL tuning), `EnsembleSurrogate`, split/normalized/weighted `ConformalSurrogate`, 14 acquisitions in `ACQUISITION_REGISTRY` (`LogEI`, `KB`, `DPP-UCB`, `MaxMin-UCB`, exact 2D/QMC `qEHVI`, ParEGO, Avoid-ome constrained EI/UCB), correlation-gated `MultiFidelityDMTA`, $O(N_A + N_B)$ `CombinatorialThompsonSampler`, `ColdStartSelector`, `StoppingAdvisor`, `PendingBatchManager`, and SHA-256 `DMTALedger`.
- **Apple-Style Circuit Studio v1 (`geneuclidean.studio` & `geneuclidean studio` CLI):** Typed `CircuitGraphV1` validator, AST-verified Python SDK compiler, allowlisted local execution engine, and zero-dependency stdlib HTTP bridge (`serve_studio`).
- **3D Point-Cloud CV → Atom Encoding Blueprint:** 42-method literature synthesis & MPS profiling (`docs/pointcloud_cv_3d_atom_encoding_plan.md`).

---

## [5.0.0] — 2026-11-26

### Added
- **Phase 1 — Beyond-PoseBusters 7-Axis Pose Auditor (`geneuclidean.audit`, `evaluation.torsion_strain`, `evaluation.tautomer_microstate`, `sdk.profiles`):** Crystal-calibrated ring strain, relaxed SMARTS torsion scan, pH 7.4 tautomer/protomer electrostatic check, 6-channel PLIF anchors, MedChem hygiene, and indication-agnostic property profiles (`cns_mpo_v2`, `oral_ro5`, `kinase_selectivity`, `beyond_ro5`).
- **Phase 2 — MIT Contrastive 3D Screening (`geneuclidean.screening`):** Multi-conformation & off-target repulsive `PocketScreener`, streaming `LigandVectorIndex`, IBM `MMELON` plugin, and hard-negative paralog selectivity contrastive loss.
- **Phase 3 — Makeable Synthesis, Avoid-ome MMP & Interop (`synthesis`, `leadopt`, `interop`):** 16 robust MedChem SMIRKS reactions, 1-step retrosynthesis registry, R-group SAR & activity cliffs, `AvoidomeMMPEngine`, charge-matched (`Δq = 0`) `OpenFE` RBFE exporter, and `Boltz-2`/`AF3`/`Protenix` mmCIF/PDB + confidence JSON adapter.
- **Phase 4 — OpenADMET Conformal DMPK & MCP Agent Server:** `ConformalDMPKRegressor` (90% split-conformal intervals), SHA-256 typed `mcp_server` DMTA tools, `CITATION.cff`, and `Dockerfile`.

---

## [4.0.0] — 2026-11-25

### Added
- **Tri-Disorder 3-LLM CNS Platform (`28` Human Targets):** Unified Tourette CSTC/GABA × Schizophrenia × Alzheimer's pocket atlas with integral cofactor retention (`FAD`, `ADP`), `SAFE-GPT-87M` + `NovoMolGen-157M` + `GP-MoLFormer-Uniq` head-to-head, 12-endpoint scaffold-CV Avoid-ome/DMPK suite + Rat $K_{p,uu}$ calibration, FDA/EMA 7-step AI Credibility Dossiers, and the **1.79 MB INT8 Client-Side Browser AI Web Studio**.
- **ETH CADD Frontier Extensions:** Cremer–Pople saturated-ring conformational strain, multi-conformation contrastive CFG + $\Delta\text{PMI}_{3\text{D}}$, natural-product scaffold-hop de-risking, and AIM-style multi-task gradient conflict mediation.
- **Modular Python Library SDK:** PEP 562 lazy `import geneuclidean as ge`, PEP 561 `py.typed`, thread-safe `Registry[T]` with `entry_points` discovery, dual-inheritance `GenEuclideanError` hierarchy, and fluent `DesignPipeline`.

---

## [3.1.0] — 2026-11-25

### Added
- **Closed-Loop CNS Neuroscience Study (`12` Targets):** AD-damped CNS-RPO-DPO resolving the Basic-Amine CNS Paradox (`100%` PoseBusters, `92.0%` Joint Physics+Safe), 6-channel 3D PLIF profiler, 300-DPI 3D studio renders, and $10 \times 10$ cross-target Vina selectivity validation.

## [3.0.0] — 2026-11-25

### Added
- **Chemical Foundation Transformer 3D Bridge:** `PocketPerceiverResampler` (24-token $SE(3)$-invariant soft prefix), `GP-MoLFormer-Uniq` LoRA adapters, KV-cached CFG, and `6.31×` `< 4.5 Å` contact attention enrichment on CASF-2016.

## [2.0.0] — 2025-11-24

### Added
- Clean-room `float32` chiral `PaiNN-SE3` (`SO(3) < 8e-6`), `PocketCVAE`, `pKd` + `DrugCLIP` multi-task heads (`CASF-2016 pKd r = 0.746`), and RPO-DPO alignment.

## [1.0.0-thesis-2021] — 2021

- Original ETH Zürich semester thesis baseline (`e3nn 0.0.0 + LSTM`, archived for provenance).
