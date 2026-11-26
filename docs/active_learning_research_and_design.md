# Active Learning for Drug Discovery, Virtual Screening & De Novo Design: Critical Literature Synthesis & GenEuclidean `v5.1` Engineering Specification

> **Document Role:** Publication-grade research synthesis, critical pharma/academic failure-mode analysis, exact software architecture for `src/geneuclidean/active_learning/`, and zero-synthetic retrospective benchmark protocol grounded in real local datasets under `data/`.
> **Codebase Anchor:** GenEuclidean `HEAD 5fe558c` (Python 3.12; runtime stack `numpy 2.5.5`, `scipy 1.18.1`, `scikit-learn 1.9.1`, `rdkit 2026.3.6`, `torch 2.15.0`; zero `gpytorch`/`botorch`/`chemprop` required).

---

## 1. Critical Literature & Open-Source Synthesis (2023–2026)

Modern Active Learning (AL) in drug discovery has matured past proving that iterative docking beats brute-force screening. Between 2023 and 2026, industrial teams (Roche/Genentech "Lab-in-a-Loop", AstraZeneca DISC/REINVENT4/PIP, Novartis retrospective DMTA analyses, Schrödinger AL-FEP+) and academic benchmarkers (Graff/Coley, Griffiths/Tripp/Hernández-Lobato, Walters/Klaric, van Tilborg/Grisoni, Laghuvarapu/Barber) converged on a sobering lesson: **surrogate calibration under scaffold shift, batch decorrelation, multi-objective safety constraints, and reaction-space tractability determine real-world hit-to-lead ROI far more than exotic neural acquisitions.**

### 1.1 Comparative Research & OSS Synthesis Table

| Paradigm / Library | Key Citations & OSS Repos | Core Mathematical Mechanism | Genuine Strengths (Pharma & Academic) | Documented Failure Modes & Limitations | GenEuclidean `v5.1` Design Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MolPAL** *(Pool-Based AL for HTVS)* | Graff, Shakhnovich & Coley, *Chem. Sci.* 2021; Graff et al. 2022–2024 ([`coleygroup/molpal`](https://github.com/coleygroup/molpal)) | Iterative RF / NN / D-MPNN surrogate + `greedy`, `UCB`, `EI`, `PI`, `TS` over pre-enumerated pools ($10^5$–$10^8$) | Recovers ~80–95% of top-1% docking hits in 2–5% of compute; empirically proved simple **greedy** often matches UCB/EI in large noisy pools | **Scaffold collapse** under greedy; **batch myopia** at large batch sizes; 2D shortcut overfitting to docking artifacts; scales poorly past $10^8$ enumerated molecules | Adopt clean pool-based DMTA loop, **fix** batch myopia & scaffold collapse with `KrigingBeliever` + `DPP` + `MaxMin` |
| **GAUCHE & Tanimoto-Kernel GPs / TRF** | Griffiths, Klarner, Moss et al., *NeurIPS* 2023 ([`leojklarner/gauche`](https://github.com/leojklarner/gauche)); Ralaivola et al. 2005; Tripp, Bacallado, Singh & Hernández-Lobato, *NeurIPS* 2023 (*Tanimoto Random Features*) | Exact GP with PSD binary Tanimoto $k_{\text{Tan}}$ or continuous MinMax $k_{\text{MM}}$ kernel on ECFP/count fingerprints; TRF for linear-time scaling | **Gold-standard UQ in low-to-medium data ($n \le 2{,}500$)** — exact closed-form epistemic variance naturally reverts to prior variance as Tanimoto distance $\to 1$; zero PyTorch needed for exact Cholesky GP | Exact GP is $O(N^3)$ (bottlenecks past $N \approx 3{,}000$ without TRF/Nyström); 2D binary Tanimoto misses pure 3D stereochemical/activity cliffs unless chirality-aware or fused with 3D features | **Primary default surrogate** (`TanimotoGP` with binary Tanimoto + MinMax kernels, MLL hyperparameter tuning, Cholesky jitter ladder) |
| **BoTorch / Ax (`qEI`, `qNEHVI`, `qLogNEHVI`)** | Balandat et al., *NeurIPS* 2020; Daulton et al., *NeurIPS* 2021; Ament, Daulton, Eriksson, Balandat & Bakshy, *NeurIPS* 2023 Spotlight ([`pytorch/botorch`](https://github.com/pytorch/botorch)) | Parallel Monte-Carlo acquisition functions; log-space stabilization (`qLogEI`, `qLogNEHVI`) eliminates vanishing gradients/underflow in negative tails | Mathematically principled batch multi-objective Pareto optimization (`qNEHVI`/`qLogNEHVI`) under observation noise; handles constraints cleanly | Continuous-space gradient optimization (`optimize_acqf`) does not natively operate on discrete molecular graphs; heavy PyTorch/GPyTorch stack; HV box decomposition scales poorly beyond $M > 4$ objectives | Implement **zero-PyTorch** discrete-pool `log_ei` tail stabilization, **exact 2D + QMC `qEHVI`**, and **$M \ge 3$ `pareto_scalarized_ucb`** |
| **Reaction-Space Thompson Sampling** *(Enamine REAL)* | Klaric, Goldman, Kramer, Riley & Walters, *JCIM* 2024 ([`PatWalters/TS`](https://github.com/PatWalters/TS)); AstraZeneca 2024 warmup & roulette-wheel follow-up | Factorized Bayesian posteriors $(\mu_r, \sigma_r^2)$ over *reagents* $r \in \mathcal{A} \cup \mathcal{B}$ rather than $N_A \times N_B$ products | Searches $10^8$–$10^{11}$ make-on-demand spaces in **$O(N_A + N_B)$** without full enumeration; found >50% of top-100 docking hits across 335M compounds in 1% budget | **Degrades under strong epistasis / non-additivity** (e.g., 3D steric clashes between $a$ and $b$, documented by Walters & Riley 2024); requires mandatory warm-up ($w \ge 3$) and thermal exploration to avoid premature reagent lock-in | Build `CombinatorialThompsonSampler` directly on `ROBUST_MEDCHEM_REACTIONS` with conjugate Normal-Normal updates, warm-up, thermal exploration, and MedChem hygiene gating |
| **Active Learning FEP+ & Multi-Fidelity BO** | Konze et al., *JCIM* 2019; Thompson et al., *JCIM* 2022; Sabanza-Gil et al., 2024–2025 (*Best Practices in MFBO for Chemistry*) | Surrogate-guided selection of alchemical RBFE edges; autoregressive co-kriging $f_H(x) = \rho f_L(x) + \delta(x)$ with cost-normalized VOI | Cuts expensive FEP+ GPU hours 5–10× while retaining thermodynamic accuracy; bridges cheap docking/ML to rigorous $\Delta\Delta G$ | **MFBO hurts worse than single-fidelity BO when low-fidelity rank correlation $\rho_s < 0.3$** (Sabanza-Gil 2024/2025); unvetted strained poses ruin FEP convergence | Build correlation-gated `MultiFidelityDMTA` (Tier 0 `DrugCLIP`/2D $\to$ Tier 1 `Vina`/`PoseRealityAuditor` $\to$ Tier 2 `OpenFE`/wet-lab) with automatic low-$\rho$ fallback |
| **REINVENT4 + Generative Active Learning** | Loeffler et al., *J. Cheminform.* 2024 ([`MolecularAI/REINVENT4`](https://github.com/MolecularAI/REINVENT4)); van Tilborg & Grisoni, *Nat. Comput. Sci.* 2024 | Surrogate reward model updated iteratively inside RL / curriculum generative sampling + diversity filters | Explores beyond fixed catalogs; van Tilborg & Grisoni (2024) showed up to 6× hit-finding gains in extreme low-data regimes | **Reward hacking** against out-of-domain surrogate regions; generates unsynthesizable or strained structures if unconstrained; fragile on activity cliffs (`MoleculeACE`) | Couple `ActiveLearningEngine` outputs with `ConformalSurrogate` applicability bounds, `MedChem` triage, and `OpenFE`/reaction synthesis grounding |
| **UQ & Conformal Prediction under Scaffold Shift** | Chemprop v2 (Heid et al. 2024); Bengs et al., *NeurIPS* 2023 (*EDL Mirage*); Tibshirani et al. 2019; Barber et al. 2023; Laghuvarapu et al., *NeurIPS* 2023 (*CoDrug*) & 2026 (*KMM-CP*); Norinder/Alvarsson (*CPSign*) | Deep ensembles vs. Evidential DL (NIG prior) vs. distribution-free Split-Conformal / normalized / density-weighted conformal calibration | **Split-Conformal** gives finite-sample coverage $\ge 1 - \alpha$ under exchangeability; **normalized conformal** adapts width to OOD distance; aligns with FDA Jan 2025 / Joint FDA–EMA Jan 2026 Credibility Guidance | **Standalone EDL is fundamentally flawed** (non-vanishing epistemic UQ, KL coupling, OOD overconfidence — Bengs et al. 2023); **unweighted split-conformal undercovers under severe scaffold shift** where exchangeability breaks | **Reject standalone EDL**; build `ConformalSurrogate` (`absolute`, `normalized`, and Tanimoto-`weighted` modes) + scaffold-disjoint calibration & similarity-stratified coverage audits |
| **Batch Diversity: KB/CL, DPP, BADGE, MaxMin** | Ginsbourger et al. 2010 (*KB/CL*); Chen, Zhang & Zhou, *NeurIPS* 2018 (*Fast Greedy MAP DPP*); Ash et al., *ICLR* 2020 (*BADGE*) | Sequential fantasy conditioning (`KB`/`CL`), quality-diversity kernel determinant $L = \text{diag}(q) K \text{diag}(q)$ (`DPP`), gradient-space k-Means++ (`BADGE`) | Eliminates parallel-batch redundancy in 8–96 compound DMTA rounds; `KB` handles **asynchronous in-flight pending batches** naturally; fast MAP `DPP` runs in $O(B^2 M)$ | `BADGE` requires differentiable neural last-layer gradients (inapplicable to GPs/GBDTs); naive `MaxMin` ignores surrogate posterior correlations; poorly weighted diversity wastes budget on low-quality outliers | Implement `kriging_believer` (with Constant Liar modes + pending-batch conditioning), `dpp_ucb` (Chen 2018 Cholesky MAP), and `maxmin_ucb` |
| **Cold-Start, Async DMTA & Stopping Criteria** | Butina 1999; Novartis/Roche DMTA retrospectives 2023–2025; autonomous lab-in-a-loop frameworks 2024–2025 | Butina medoids / MaxMin seeding; pending-compound masking + fantasy conditioning; EI saturation & top-$K$ Jaccard stability | Prevents round-0 scaffold bias; eliminates serial wait stalls when synthesis turnaround varies from days to weeks; stops wasted assay spend | Random cold-start misses rare active scaffolds; fixed-budget loops over-spend after model ranking converges | Implement `butina_medoids`, `maxmin`, `random` cold-start; first-class `register_pending()` / `complete_pending()` async ledger; multi-signal `StoppingAdvisor` |

---

## 2. Current GenEuclidean Codebase Audit & Integration Seams

Auditing the existing `v5.0` modules reveals strong CADD foundations alongside exact gaps that `src/geneuclidean/active_learning/` must close cleanly:

| Existing Module | Current Strengths | Concrete Gaps for `v5.1` Active Learning | Exact Integration Seam |
| :--- | :--- | :--- | :--- |
| `src/geneuclidean/screening/active_learning.py` | Clean `ActiveLearningDMTA` loop, CSV assay ingestion (`ingest_assay_csv`), SHA-256 round digests, `ucb`/`thompson`/`greedy`/`maxmin_diverse_ucb` | Hard-coded `RandomForestRegressor` only (no Tanimoto GP, no aleatoric/epistemic split, no conformal bounds); no `EI`/`PI`/`KB`/`DPP`/`qEHVI`/constrained acquisitions; no cold-start clustering; no async pending batches; no stopping diagnostics; no calibration metrics; uses `RF` tree std without OOD reversion | Preserve 100% backward-compatible `ActiveLearningDMTA`, `ActiveLearningRoundSummary`, `AcquisitionStrategy` API (and export `DMTAQuery`, `DMTARoundSummary`, `DMTALedger`), delegating cleanly to `geneuclidean.active_learning` |
| `src/geneuclidean/cns/dmpk_regression.py` | Clean `ConformalDMPKRegressor` (HGB+RF + exact finite-sample split-conformal quantile `ceil((n+1)(1-alpha))/n`) & `evaluate_scaffold_cv_conformal_dmpk` | Homoskedastic intervals only (`yp ± q_hat`, constant width regardless of Tanimoto OOD distance); tightly coupled to HGB+RF rather than generic `SurrogateProtocol` | Reuse `featurize_admet_smiles` / `make_scaffold_kfold` patterns and upgrade conformal calibration to `absolute`, `normalized`, and Tanimoto-`weighted` modes in `ConformalSurrogate` |
| `src/geneuclidean/cns/credibility.py` | Exact `_finite_sample_conformal_quantile`, `SplitConformalBinaryCalibrator` (with Mondrian option), FDA/EMA Credibility Dossier | Binary classification only; not wired into iterative active-learning calibration ledgers | Match exact finite-sample quantile conventions & SHA-256 provenance schema for Credibility Dossier interoperability |
| `src/geneuclidean/screening/screener.py` & `encoders.py` | Zero-PyTorch `ECFP4BinaryEncoder`, fast `0.02 ms/query` `DrugCLIPLigandEncoder` / `PocketScreener`, `LigandVectorIndex`, `featurize_ecfp4_batch` | Static single-shot screening; no closed-loop surrogate updating or multi-fidelity promotion from `DrugCLIP` $\to$ Vina $\to$ RBFE | Wire `PocketScreener` / `LigandVectorIndex` as Tier-0 fast scoring oracle inside `MultiFidelityDMTA` and reuse `featurize_ecfp4_batch` (plus count/chiral ECFP helpers) |
| `src/geneuclidean/synthesis/reactions.py` | 16 curated high-yield `ROBUST_MEDCHEM_REACTIONS` (Hartenfeller / Enamine REAL SMIRKS) + `triage_molecule` hygiene gate | Exhaustive `enumerate_makeable_analogs` ($O(N_A N_B)$) only; no Bayesian reagent-space search | Power `CombinatorialThompsonSampler` directly with `ROBUST_MEDCHEM_REACTIONS` and on-demand RDKit `RunReactants` + `triage_molecule` gating |
| `src/geneuclidean/audit/pose_auditor.py` | Zero-PyTorch 7-axis Beyond-PoseBusters `PoseRealityAuditor` (catches 39.3% strained Vina poses) | Post-hoc audit only; not used as a multi-fidelity gate or feasibility constraint inside AL | Plug `PoseRealityAuditor` into Tier-1 of `MultiFidelityDMTA` (`passed_physical_beyond_pb` gate) and as a feasibility scorer in `constrained_ei` |
| `src/geneuclidean/interop/openfe_export.py` | Charge-consistent ($\Delta q = 0$), ring-strain-clean Lomap RBFE network exporter (`export_openfe_rbfe_network`) | Disconnected from AL batch selection | Allow `MultiFidelityDMTA` Tier-2 handoff to export selected lead-opt batches directly to an OpenFE RBFE perturbation graph |
| `src/geneuclidean/sdk/registries.py`, `profiles.py`, `registry.py`, `protocols.py` | Thread-safe `Registry[T]` (zero heavy imports), PEP 544 `Protocol`s, `PROPERTY_PROFILE_REGISTRY` (`cns_mpo_v2`, `oral_ro5`, `kinase_selectivity`, `beyond_ro5`), `< 15 ms` cold `import geneuclidean` | No `SURROGATE_REGISTRY` or `ACQUISITION_REGISTRY`; no AL protocols | Register `SURROGATE_REGISTRY` and `ACQUISITION_REGISTRY` using `geneuclidean.registry.Registry`, keeping cold `import geneuclidean` strictly `< 15 ms` with zero eager `torch`/`sklearn`/`transformers` |

---

## 3. Structured Critical Brainstorming (Two Iterations)

### Iteration 1 — Pharma & Academic Reality vs. Academic Hype

**Question:** *What genuinely accelerates hit-to-lead and lead-optimization at Roche, AstraZeneca, Novartis, and academic CADD labs vs. what looks impressive on synthetic toy papers but fails on real assays?*

1. **What genuinely works in production:**
   - **Exact Tanimoto/MinMax GPs at small-to-medium DMTA scale ($n = 16\text{–}2{,}000$):** Real wet-lab lead-opt series rarely exceed 500–2,000 assayed compounds per target. In this regime, a well-tuned Tanimoto GP with exact marginal log-likelihood hyperparameter estimation consistently matches or beats complex D-MPNNs while providing **calibrated geometric OOD variance** (uncertainty grows smoothly as max-Tanimoto to training set drops) and exact covariance for batch decorrelation.
   - **Batch decorrelation (`KrigingBeliever` & `DPP`) + Cold-start clustering (`Butina` / `MaxMin`):** Real DMTA runs in parallel batches ($B = 8\text{–}48$). Picking top-$B$ of a pointwise acquisition produces near-clone analogs that all share the same metabolic or hERG liability. Sequential posterior-variance deflation (`KrigingBeliever`) and submodular quality-diversity selection (`DPP`) maximize **unique active scaffolds discovered per assay dollar**.
   - **Multi-objective & feasibility-constrained acquisition (`qEHVI`, `constrained_ei`, `avoidome_feasibility_ucb`):** Potency-only AL is a trap: in CNS and kinase programs, the highest-potency virtual hits are overwhelmingly lipophilic hERG blockers, CYP inhibitors, or high-clearance liabilities. Embedding `CNS MPO v2`, `MedChem triage`, and Avoid-ome probability gates *inside* the acquisition function prevents wasting wet-lab cycles on dead-end chemotypes.
   - **Reagent-space Thompson Sampling for make-on-demand spaces:** Enumerating $10^4 \times 10^4$ virtual reactions is infeasible; updating factorized reagent posteriors in $O(N_A + N_B)$ delivers real purchasable/1-step makeable hits with explicit synthetic provenance.
   - **Asynchronous in-flight batch handling:** Medicinal chemistry turnaround is heterogeneous. Treating pending synthesis/FEP jobs as hallucinated fantasy observations (Kriging Believer conditioning) prevents duplicate proposals without halting the chemistry team.

2. **What is academic hype or fails under real-world assay noise & scaffold shift:**
   - **Hype #1 — Standalone Evidential Deep Learning (EDL):** Theoretically unsound as an epistemic UQ estimator (Bengs et al., *NeurIPS* 2023): epistemic uncertainty fails to vanish with data and yields dangerous false confidence on novel scaffolds. *Excluded from `v5.1`.*
   - **Hype #2 — Claiming uncertainty acquisitions always beat `greedy` on Top-$K$ recall:** In large noisy pools ($N \gg 10^3$) evaluated purely by raw Top-$K$ recall without scaffold-diversity accounting, `greedy` frequently ties or beats `UCB`/`EI` (Graff et al. 2021; van Tilborg & Grisoni 2024). We must benchmark `greedy` honestly and report **both** Top-$K$ recall **and** unique active scaffold diversity + batch internal Tanimoto so users see the true Pareto trade-off.
   - **Hype #3 — Uncritical Multi-Fidelity BO:** When low-fidelity scoring (e.g., generic docking or untuned zero-shot `DrugCLIP`, which showed `0.39–0.52` paralog AUROC in `v5_cadd_suite_report.md`) correlates weakly with high-fidelity potency ($\rho_s < 0.25$), multi-fidelity co-kriging injects bias and wastes budget. Multi-fidelity routing MUST measure cross-fidelity correlation online and fail safe to high-fidelity exploration!
   - **Hype #4 — Unweighted Split-Conformal under severe scaffold shift:** Exchangeability is violated when moving to novel Bemis–Murcko scaffolds. Homoskedastic `± q_hat` intervals undercover far-from-training scaffolds while being overly wide on close analogs. Normalized nonconformity scores + scaffold-disjoint calibration + similarity-stratified coverage auditing are mandatory.
   - **Hype #5 — Toy benchmarks (Branin, Hartmann, or duplicate-scaffold ChEMBL random pools):** Random pools containing 30 close analogs per scaffold trivially reward memorization and make every acquisition look brilliant. Only **strict Bemis–Murcko-deduplicated pools** and **real multi-target/ADMET datasets** test true discovery power.

### Iteration 2 — Hard Engineering Stress-Testing & Failure-Mode Prevention

Stress-testing our proposed architecture against mathematical edge cases, computational budgets, and codebase invariants revealed **seven critical failure modes** that we engineer out directly in the specification:

1. **Duplicate / Invalid SMILES Crashing GP Cholesky:** Real chemical catalogs contain stereoisomer duplicates, salt forms, or distinct SMILES that collide onto identical ECFP4 bit vectors, yielding singular Gram matrices ($\lambda_{\min}(K) = 0$). *Fix:* Enforce canonical SMILES deduplication, handle zero-norm bit vectors explicitly ($k(\mathbf{0}, \mathbf{0}) = 1$), and apply a deterministic adaptive Cholesky jitter ladder ($\epsilon \in \{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$).
2. **Sign-Inversion Bug in Feasibility-Weighted UCB:** Multiplying a negative acquisition value $\alpha(x) < 0$ (e.g., negative $\log\text{BB}$ or standardized UCB) by a low feasibility probability $P_{\text{feas}}(x) = 0.01$ makes an *infeasible* compound rank *higher* than a feasible compound with $\alpha(x) = -0.5$ and $P_{\text{feas}} = 0.99$! *Fix:* `constrained_ei` multiplies strictly non-negative $\text{EI}(x) \ge 0$; `avoidome_feasibility_ucb` shifts $\alpha_{\text{raw}}(x)$ by $-\min_i \alpha_{\text{raw}}(x_i) + \epsilon$ (strictly positive utility) before multiplying by $\prod_j P_{\text{feas}, j}(x)$.
3. **EI Underflow Tie-Collapse:** When all pool candidates lie far below $f^*$, standard floating-point $\text{EI}(x)$ underflows to `0.0`, degenerating to arbitrary index order. *Fix:* Implement numerically stable `log_ei` via `scipy.special.log_ndtr` and asymptotic expansion for $z < -10$ (Ament et al., *NeurIPS* 2023).
4. **Conformal Data Leakage & Small-$n$ Starvation:** Evaluating conformal residuals on training data causes catastrophic undercoverage; conversely, holding out 20% of 16 round-1 observations ruins the surrogate. *Fix:* Never fit conformal quantiles on training predictions; enforce strict held-out calibration splits (scaffold-grouped when available) for `fit(X_train, y_train, X_cal, y_cal)`, and provide automatic OOF/CV+ calibration fallback when $n_{\text{obs}} < 40$ inside an active-learning loop (explicitly tagged in metadata as `calibration_mode="cv_plus"` vs `"split"`).
5. **Pre-Protonated Catalogs Breaking SMIRKS Reactions (Verified Locally!):** Direct inspection of `data/raw/zinc250k.csv` revealed that ZINC250k stores physiological carboxylates `C(=O)[O-]`, yielding **0 matches** for neutral `[C](=O)[OH]` until uncharged via `rdMolStandardize.Uncharger()` (which recovers **141 acids, 968 amines, 120 aryl bromides/iodides** from the first 15,000 rows alone!). *Fix:* Standardize/uncharge reactant pools inside `CombinatorialThompsonSampler` (optional `uncharge_reactants: bool = True`) and handle failed reaction pairs without poisoning reagent posteriors.
6. **Heavy-Import Regression on `import geneuclidean`:** Importing `sklearn`, `scipy.optimize`, or `rdkit` at package top level would violate the repo invariant (`import geneuclidean < 15 ms`, `heavy == []`). *Fix:* Keep `src/geneuclidean/active_learning/__init__.py` strictly PEP 562 lazy (`__getattr__` + `if TYPE_CHECKING:`), and defer `sklearn`/`rdkit` imports cleanly inside module functions/methods or lazy sub-modules.
7. **Zero-PyTorch Dependency Constraint:** Neither `gpytorch` nor `botorch` is installed in `.venv`. *Fix:* Implement exact analytical `TanimotoGP`, fast greedy MAP `DPP`, exact 2D + QMC `qEHVI`, `KrigingBeliever`, `MultiFidelityDMTA`, and `CombinatorialThompsonSampler` in pure **NumPy + SciPy + scikit-learn + RDKit**.

---

## 4. Module & API Architecture: `src/geneuclidean/active_learning/`

### 4.1 Package Layout & Dependency Hygiene

```text
src/geneuclidean/active_learning/
├── __init__.py           # PEP 562 lazy public API exports (zero eager sklearn/rdkit/torch)
├── featurizers.py        # Cached canonical SMILES -> ECFP4 binary / count / chiral + 12 physchem descriptors + Tanimoto/MinMax kernels
├── surrogates.py         # SurrogateProtocol, SurrogatePrediction, TanimotoGP, EnsembleSurrogate, ConformalSurrogate, SURROGATE_REGISTRY
├── acquisitions.py       # AcquisitionContext, ACQUISITION_REGISTRY (greedy, ucb, ei, log_ei, pi, thompson, uncertainty,
│                         #   maxmin_ucb, kriging_believer, dpp_ucb, qehvi, pareto_scalarized_ucb, constrained_ei, avoidome_feasibility_ucb)
├── multifidelity.py      # FidelityTierSpec, MultiFidelityObservation, MultiFidelityDMTA (correlation-gated co-kriging & VOI router)
├── combinatorial_ts.py   # ReagentPosterior, CombinatorialTSRound, CombinatorialThompsonSampler (O(N_A + N_B) reaction-space TS)
├── diagnostics.py        # ColdStartStrategy (butina_medoids, maxmin, random), CalibrationMetrics, StoppingReport, StoppingAdvisor
└── engine.py             # DMTAQuery, DMTARoundSummary, DMTALedger, ActiveLearningEngine (sync + async pending batches + multi-objective)
```

**Backward-compatibility contract (`src/geneuclidean/screening/active_learning.py`):**
- Keep `AcquisitionStrategy`, `ActiveLearningRoundSummary`, and `ActiveLearningDMTA` 100% signature- and behavior-compatible with all existing callers (`tests/test_screening_and_al.py`, `scripts/run_selectivity_screening_benchmark.py`, `src/geneuclidean/screening/__init__.py`, `src/geneuclidean/__init__.py`).
- Upgrade `ActiveLearningDMTA` to accept optional `surrogate: str | SurrogateProtocol = "rf_legacy"` (defaulting to legacy RF behavior when instantiated with old positional/keyword args, or delegating to `ActiveLearningEngine` / `TanimotoGP` / `ConformalSurrogate` / new acquisitions when requested), and re-export `DMTAQuery`, `DMTARoundSummary`, `DMTALedger`.

---

### 4.2 Detailed Module Specifications & Mathematical Invariants

#### A. `featurizers.py` — Molecular Representations & PSD Similarity Kernels
- `canonicalize_smiles(smi: str, *, isomeric: bool = True) -> str | None`
- `featurize_pool(smiles: Sequence[str], *, kind: Literal["ecfp4_binary", "ecfp4_count", "ecfp4_physchem"] = "ecfp4_binary", n_bits: int = 2048, radius: int = 2, use_chirality: bool = True) -> tuple[NDArray[np.float64], NDArray[np.bool_]]`
  - Uses RDKit `rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=use_chirality)` (avoids RDKit deprecation warnings from `GetMorganFingerprintAsBitVect`!).
  - Supports internal LRU/dict memoization keyed by `(canonical_smiles, kind, n_bits, radius, use_chirality)` so iterative DMTA rounds never recompute fingerprints for the static pool.
- `tanimoto_kernel(X: NDArray[np.float64], Y: NDArray[np.float64] | None = None) -> NDArray[np.float64]`
  - Exact binary/non-negative Tanimoto kernel:
    $$k_{\text{Tan}}(x, y) = \frac{x^\top y}{\|x\|_2^2 + \|y\|_2^2 - x^\top y}$$
    with exact zero-vector boundary rule: if $\|x\|_2^2 = 0$ and $\|y\|_2^2 = 0$, $k = 1.0$; if only one is zero, $k = 0.0$.
- `minmax_kernel(X: NDArray[np.float64], Y: NDArray[np.float64] | None = None) -> NDArray[np.float64]`
  - Exact continuous Ruzicka / MinMax kernel for non-negative count/descriptor vectors $x, y \in \mathbb{R}_{\ge 0}^d$:
    $$k_{\text{MM}}(x, y) = \frac{\sum_{j=1}^d \min(x_j, y_j)}{\sum_{j=1}^d \max(x_j, y_j)}$$
    (with $k_{\text{MM}}(\mathbf{0}, \mathbf{0}) = 1.0$). Chunked/vectorized in float64 to avoid excessive memory blowup on large matrices.

**Mathematical Invariants (`featurizers.py`):**
1. Symmetry: $\|K(X, X) - K(X, X)^\top\|_\infty < 10^{-12}$.
2. Unit diagonal & boundedness: $\text{diag}(K(X, X)) = \mathbf{1}$ and $K_{ij} \in [0, 1]$.
3. Positive Semi-Definiteness: $\lambda_{\min}(K(X, X)) \ge -10^{-10}$.
4. Binary equivalence: for $X \in \{0, 1\}^{n \times d}$, $\|k_{\text{MM}}(X, X) - k_{\text{Tan}}(X, X)\|_\infty < 10^{-12}$.

---

#### B. `surrogates.py` — Exact GP, Decomposed Ensemble & Conformal UQ

```python
@dataclass(frozen=True)
class SurrogatePrediction:
    mean: NDArray[np.float64]                 # shape [N], raw target units
    std: NDArray[np.float64]                  # shape [N], total or epistemic std (>= 0)
    epistemic_std: NDArray[np.float64]        # shape [N], >= 0
    aleatoric_std: NDArray[np.float64]        # shape [N], >= 0
    valid: NDArray[np.bool_]                  # shape [N]
    interval_lo: NDArray[np.float64] | None = None  # populated by ConformalSurrogate
    interval_hi: NDArray[np.float64] | None = None
    alpha: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class SurrogateProtocol(Protocol):
    @property
    def name(self) -> str: ...
    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> dict[str, float]: ...
    def predict(self, smiles: Sequence[str]) -> SurrogatePrediction: ...
```

1. **`TanimotoGP` (Primary Surrogate):**
   - `__init__(self, *, kernel: Literal["tanimoto", "minmax"] = "tanimoto", n_bits: int = 2048, radius: int = 2, use_chirality: bool = True, signal_variance: float = 1.0, noise_variance: float = 0.05, optimize_hyperparams: bool = True, jitter: float = 1e-6, seed: int = 42)`
   - **Fitting algorithm:**
     1. Featurize valid training SMILES into $X \in \mathbb{R}_{\ge 0}^{n \times d}$, standardize targets $\tilde{y} = (y - \bar{y}) / s_y$ (guarding $s_y = 1.0$ if $\text{std}(y) < 10^{-8}$).
     2. Compute base kernel matrix $K_0 = k(X, X)$ once (kernel hyperparameters $\sigma_f^2, \sigma_n^2$ do NOT alter $K_0$, so MLL optimization is ultra-fast — even faster via a single eigendecomposition $K_0 = Q \Lambda Q^\top$!).
        - *Optimization speedup insight:* Because $K_y(\sigma_f^2, \sigma_n^2) = \sigma_f^2 K_0 + \sigma_n^2 I = Q (\sigma_f^2 \Lambda + \sigma_n^2 I) Q^\top$, evaluating the exact negative log marginal likelihood
          $$-\log p(\tilde{y} \mid X, \sigma_f^2, \sigma_n^2) = \frac{1}{2} \sum_{i=1}^n \frac{(Q^\top \tilde{y})_i^2}{\sigma_f^2 \lambda_i + \sigma_n^2} + \frac{1}{2} \sum_{i=1}^n \log(\sigma_f^2 \lambda_i + \sigma_n^2) + \frac{n}{2} \log(2\pi)$$
          takes **$O(n)$ per L-BFGS-B iteration** after one $O(n^3)$ `scipy.linalg.eigh` (and for $n \le 2{,}000$ takes `< 50 ms` total)! Bounds: $\sigma_f^2 \in [10^{-2}, 10^2]$, $\sigma_n^2 \in [10^{-4}, 2.0]$.
     3. Factorize final $K_y = \sigma_f^2 K_0 + (\sigma_n^2 + \epsilon) I$ via `scipy.linalg.cho_factor` with adaptive jitter ladder $\epsilon \in \{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$, and store $\alpha = K_y^{-1} \tilde{y}$ via `cho_solve`.
   - **Exact prediction & covariance:**
     - `predict(smiles)` computes $K_* = \sigma_f^2 k(X_*, X)$:
       $$\mu(x_*) = \bar{y} + s_y \cdot K_* \alpha$$
       $$V = L^{-1} K_*^\top, \quad \sigma_{\text{epi}}^2(x_*) = s_y^2 \cdot \max\left(0,\; \sigma_f^2 - \sum_{i=1}^n V_{ij}^2\right), \quad \sigma_{\text{ale}}^2 = s_y^2 \sigma_n^2$$
       $$\sigma_{\text{tot}}(x_*) = \sqrt{\sigma_{\text{epi}}^2(x_*) + \sigma_{\text{ale}}^2}$$
     - `predict_covariance(smiles) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]`: returns exact latent posterior mean $\mu_*$ and covariance $\Sigma_* = s_y^2 (\sigma_f^2 K_{**} - V^\top V)$ (symmetrized, PSD-projected for Cholesky sampling in correlated Thompson Sampling and exact fantasy updates in `KrigingBeliever`!).
     - `condition_on_fantasies(fantasy_smiles: Sequence[str], fantasy_y: Sequence[float] | None = None) -> TanimotoGP`: returns a lightweight cloned/updated GP conditioned on `(X_train ∪ X_fantasy, y_train ∪ y_fantasy)` (defaulting `fantasy_y` to current posterior mean $\hat{\mu}(X_{\text{fantasy}})$ — exact Kriging Believer / pending-batch conditioning without refitting hyperparameters!).
   - **Key Mathematical Invariants:**
     - $\sigma_{\text{epi}}(x) \ge 0$ everywhere; on training points $x_i$ with noise $\sigma_n^2$, $\sigma_{\text{epi}}^2(x_i) \le s_y^2 \sigma_n^2$ and shrinks to $0$ as $\sigma_n \to 0$.
     - For an out-of-domain molecule orthogonal to all training fingerprints ($k(x_*, x_i) = 0 \; \forall i$), $\mu(x_*) = \bar{y}$ and $\sigma_{\text{epi}}(x_*) = s_y \sigma_f$ (exact prior reversion).

2. **`EnsembleSurrogate` (Large-Pool Scalable Surrogate with Uncertainty Decomposition):**
   - `__init__(self, *, n_rf_estimators: int = 64, n_hgb_members: int = 5, featurizer: Literal["ecfp4_binary", "ecfp4_physchem"] = "ecfp4_physchem", n_bits: int = 1024, seed: int = 42)`
   - Fits a `RandomForestRegressor` (using `oob_score=True` when `n >= 16`) plus `n_hgb_members` bootstrap-bagged `HistGradientBoostingRegressor` models on `ECFP4 + 12 RDKit descriptors`.
   - **Epistemic vs. Aleatoric Decomposition:**
     - Prediction mean: $\hat{\mu}(x) = 0.5\,\bar{f}_{\text{RF}}(x) + 0.5\,\bar{f}_{\text{HGB}}(x)$.
     - Epistemic variance: weighted variance across RF trees + HGB bootstrap members ($\sigma_{\text{epi}}^2(x) = 0.5\,\text{Var}_{\text{RF}}(x) + 0.5\,\text{Var}_{\text{HGB}}(x) + 0.25\,(\bar{f}_{\text{RF}}(x) - \bar{f}_{\text{HGB}}(x))^2$ — exact law of total variance across the two model families!).
     - Aleatoric variance: estimated from OOB / bootstrap out-of-bag residual MSE minus mean in-bag epistemic variance: $\hat{\sigma}_{\text{ale}}^2 = \max(\epsilon_{\text{ale}},\, \text{MSE}_{\text{OOB}} - \overline{\sigma_{\text{epi}}^2})$.

3. **`ConformalSurrogate` (Distribution-Free Finite-Sample & Adaptive Conformal Wrapper):**
   - `__init__(self, base_surrogate: SurrogateProtocol, *, alpha: float = 0.10, mode: Literal["absolute", "normalized", "weighted"] = "normalized", cal_fraction: float = 0.20, scaffold_split_cal: bool = True, weight_gamma: float = 5.0, min_sigma: float = 1e-3, min_split_n: int = 40, seed: int = 42)`
   - **Two clean fitting entry points:**
     - `fit_calibrated(smiles_train, y_train, smiles_cal, y_cal) -> dict[str, float]`: explicit user-supplied proper-train / calibration split (**zero leakage**).
     - `fit(smiles, y) -> dict[str, float]`: automatically partitions `(smiles, y)` into proper-train and held-out calibration (grouping by Bemis–Murcko scaffold when `scaffold_split_cal=True` and $\ge 5$ unique scaffolds exist!). When `len(smiles) < min_split_n` (early DMTA rounds), falls back transparently to **5-fold out-of-fold CV+ residuals** so no data is wasted while still avoiding training-residual optimism (and records `calibration_mode="cv_plus"`).
   - **Exact Nonconformity Scores & Quantile Formula:**
     - `mode="absolute"`: $s_i = |y_i - \hat{\mu}(x_i)|$, interval $[\hat{\mu}(x) - \hat{q},\; \hat{\mu}(x) + \hat{q}]$.
     - `mode="normalized"` *(default)*: $s_i = \frac{|y_i - \hat{\mu}(x_i)|}{\max(\hat{\sigma}_{\text{tot}}(x_i), \epsilon)}$, interval $[\hat{\mu}(x) - \hat{q}\cdot\hat{\sigma}(x),\; \hat{\mu}(x) + \hat{q}\cdot\hat{\sigma}(x)]$.
     - `mode="weighted"`: computes Tanimoto similarity $k_i(x_*) = k_{\text{Tan}}(x_*, x_i^{\text{cal}})$ to calibration molecules, normalized weights $\tilde{w}_i(x_*) \propto \exp(\gamma\, k_i(x_*))$ (plus point-mass weight at $\infty$ capped at $\max(s)$ with finite-sample correction), and returns the weighted $(1 - \alpha)$ quantile of normalized scores.
     - **Exact finite-sample quantile (for unweighted `absolute` / `normalized`):**
       $$r = \lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil, \quad \hat{q}_{1-\alpha} = s_{(\min(r, n_{\text{cal}}))}$$
       (computed directly on sorted calibration scores — never touching test/pool labels!).
   - **`SURROGATE_REGISTRY: Registry[Callable[..., SurrogateProtocol]]`:** registers `"tanimoto_gp"`, `"minmax_gp"`, `"ensemble"`, `"conformal_tanimoto_gp"`, `"conformal_ensemble"`.

---

#### C. `acquisitions.py` — Single-Objective, Batch-Diverse, Multi-Objective Pareto & Constrained Acquisitions

```python
@dataclass(frozen=True)
class AcquisitionContext:
    pool_smiles: tuple[str, ...]
    mean: NDArray[np.float64]                          # [N] or [N, M] for multi-objective
    std: NDArray[np.float64]                           # [N] or [N, M]
    best_observed: float                               # f* (for single-objective EI/PI)
    batch_size: int = 16
    beta: float = 1.5                                  # UCB exploration parameter
    xi: float = 0.01                                   # EI/PI jitter margin
    diversity_weight: float = 0.25                     # MaxMin blend weight
    dpp_quality_temp: float = 1.0                      # DPP quality temperature eta
    shortlist_multiplier: int = 8                      # shortlist M = min(N, max(B*mult, 128))
    surrogate: SurrogateProtocol | None = None         # needed by kriging_believer / correlated TS
    observed_objectives: NDArray[np.float64] | None = None   # [N_obs, M] for Pareto / qEHVI
    reference_point: NDArray[np.float64] | None = None       # [M] for qEHVI
    objective_weights: NDArray[np.float64] | None = None     # [M] for scalarized Pareto UCB
    feasibility_probs: NDArray[np.float64] | None = None     # [N] in [0, 1] for constrained EI/UCB
    pending_smiles: tuple[str, ...] = ()               # async in-flight molecules!
    seed: int = 42

@dataclass(frozen=True)
class AcquisitionSelection:
    indices: tuple[int, ...]                           # selected indices into pool_smiles (len <= batch_size, unique)
    smiles: tuple[str, ...]
    scores: tuple[float, ...]                          # acquisition score of each selected compound
    all_scores: NDArray[np.float64]                    # full pool score vector [N]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Registered Strategies in `ACQUISITION_REGISTRY` (13 first-class strategies):**

1. **Pointwise Single-Objective:**
   - `"greedy"`: $\alpha(x) = \mu(x)$.
   - `"uncertainty"`: $\alpha(x) = \sigma(x)$ (pure exploration / model-improvement).
   - `"ucb"`: $\alpha(x) = \mu(x) + \beta\,\sigma(x)$.
   - `"pi"`: $\alpha(x) = \Phi\left(\frac{\mu(x) - f^* - \xi}{\sigma(x)}\right)$ (returns `1.0` if $\sigma = 0 \land \mu > f^* + \xi$, else `0.0`).
   - `"ei"` / `"log_ei"`: Exact Expected Improvement with Ament et al. (*NeurIPS* 2023) log-space numerical stabilization:
     $$\text{EI}(x) = \sigma(x)\,\big(z\,\Phi(z) + \phi(z)\big), \quad \log \text{EI}(x) = \log\sigma(x) + \log h(z)$$
     where for $z \ge -10$, $\log h(z) = \log(\max(z\Phi(z) + \phi(z), 10^{-300}))$; for $z < -10$, exact asymptotic Mills-ratio expansion $\log h(z) = -\frac{1}{2}z^2 - \frac{1}{2}\log(2\pi) - 2\log|z| + \log\left(1 - \frac{3}{z^2} + \frac{15}{z^4}\right)$ guarantees smooth strictly monotonic ranking without floating-point zero ties!
   - `"thompson"`: When `surrogate` exposes `predict_covariance` (e.g., `TanimotoGP`), draws $B$ correlated joint posterior samples on the top-$M$ shortlist via Cholesky $f^{(b)} = \mu + L_\Sigma \varepsilon^{(b)}$ (picking $\arg\max$ of each draw without replacement); falls back to independent $\mu + \sigma \odot \varepsilon$ for ensembles.

2. **Batch Diversity & Sequential Decorrelation:**
   - `"maxmin_ucb"`: Shortlists top-$M$ by UCB, normalizes acquisition to $[0, 1]$, greedily selects maximizing $(1 - w_{\text{div}})\tilde{\alpha}_i + w_{\text{div}}(1 - \max_{j \in S} k_{\text{Tan}}(x_i, x_j))$.
   - `"kriging_believer"` (plus `liar_mode: Literal["kb_mean", "cl_min", "cl_mean", "cl_pessimistic"] = "kb_mean"`):
     - **Step 0 (Async Pending Conditioning):** If `pending_smiles` is non-empty and `surrogate` is a `TanimotoGP`, immediately condition the posterior covariance on `pending_smiles` so in-flight lab compounds deflate nearby variance before picking compound 1!
     - **Steps $t = 1 \dots B$ (Fast Exact Schur-Complement Conditioning on Shortlist):**
       - For `TanimotoGP`: compute the $M \times M$ posterior covariance matrix $\Sigma^{(0)}$ and mean $\mu^{(0)}$ on the top-$M$ shortlist ONCE. At step $t$:
         1. Pick $j_t = \arg\max_{j \notin S} \alpha(\mu_j^{(t-1)}, \sqrt{\Sigma_{jj}^{(t-1)}})$ (using UCB or EI).
         2. Let fantasy residual $\Delta_t = \tilde{y}_{j_t} - \mu_{j_t}^{(t-1)}$ (note: for pure Kriging Believer $\tilde{y}_{j_t} = \mu_{j_t}^{(t-1)} \implies \Delta_t = 0$, so means stay fixed while variances deflate!).
         3. Apply exact rank-1 Gaussian conditioning in $O(M^2)$:
            $$v = \Sigma_{:, j_t}^{(t-1)}, \quad d = \Sigma_{j_t, j_t}^{(t-1)} + \sigma_n^2$$
            $$\mu^{(t)} = \mu^{(t-1)} + \frac{\Delta_t}{d}\, v, \qquad \Sigma^{(t)} = \Sigma^{(t-1)} - \frac{1}{d}\, v v^\top$$
         4. Clip $\text{diag}(\Sigma^{(t)}) \ge 0$ and mark $j_t$ selected.
       - *Mathematical Invariant:* Every pick strictly reduces posterior variance at $j_t$ AND at all structurally correlated neighbors ($k_{\text{Tan}} > 0$), provably changing subsequent picks compared to static top-$B$!
       - For non-GP surrogates (`EnsembleSurrogate`), falls back cleanly to explicit `fit` refit or Tanimoto-kernel variance deflation $\sigma_j^{(t)} \leftarrow \sigma_j^{(t-1)} \sqrt{\max(0, 1 - \gamma k_{\text{Tan}}(x_j, x_{j_t})^2)}$.
   - `"dpp_ucb"` (Fast Greedy MAP DPP — Chen, Zhang & Zhou, *NeurIPS* 2018):
     - On top-$M$ shortlist, compute strictly positive quality vector $q_i = \exp(\eta \cdot \text{clip}(z_i, -5, 5))$ where $z_i = (\alpha_i - \bar{\alpha}) / \max(s_\alpha, 10^{-6})$, and regularized PSD similarity matrix $K_{ij} = (1 - 10^{-3}) k_{\text{Tan}}(x_i, x_j) + 10^{-3} \delta_{ij}$.
     - Form L-ensemble kernel $L = \text{diag}(q)\, K\, \text{diag}(q)$.
     - Run exact incremental Cholesky greedy MAP inference ($O(B^2 M)$): maintain squared residual norms $d_i^2 = L_{ii}$ and orthogonal basis vectors $c_i \in \mathbb{R}^{t}$; at step $t$, choose $j_t = \arg\max_{i \notin S} \log(\max(d_i^2, 10^{-30}))$, update $e_i = (L_{j_t, i} - \langle c_{j_t}, c_i \rangle) / d_{j_t}$, append $e_i$ to $c_i$, and set $d_i^2 \leftarrow \max(0, d_i^2 - e_i^2)$.
     - *Mathematical Invariants:* Strictly no duplicates; any near-duplicate of an already chosen molecule has $d_i^2 \approx 0$ ($\log d_i^2 \to -\infty$), forcing maximal quality-weighted structural diversity!

3. **Multi-Objective Pareto Acquisitions:**
   - Helper `pareto_front(Y: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.float64]]`: returns exact non-dominated indices & vectors (maximization convention: $a$ dominates $b$ iff $a \ge b$ elementwise and $\exists m: a_m > b_m$).
   - Helper `compute_hypervolume_2d(front: NDArray[np.float64], ref: NDArray[np.float64]) -> float` (exact $O(|\mathcal{P}|\log|\mathcal{P}|)$ sweep-line area) and `compute_hypervolume(front, ref, n_mc=4096)` (exact 2D, exact inclusion-exclusion / WFG-slice for 3D, Sobol QMC for $M \ge 4$).
   - `"qehvi"` (Expected Hypervolume Improvement):
     - **Exact 2D analytical EHVI** (Emmerich et al. 2006 / Yang et al. 2019): Sort 2D Pareto front descending in $y_1$ ($p^{(1)}_1 > \dots > p^{(P)}_1$, $p^{(1)}_2 < \dots < p^{(P)}_2$) with sentinels $p^{(0)} = (\infty, r_2)$, $p^{(P+1)} = (r_1, \infty)$. Decompose non-dominated upper-right region into $P+1$ disjoint horizontal/vertical boxes and evaluate $\mathbb{E}[\Delta\text{HV}(Y)]$ in closed form using $\Psi(a, b, \mu, \sigma) = \int_a^b (y - a)\,\mathcal{N}(y \mid \mu, \sigma^2)\,dy$!
     - **Batch `qEHVI` ($B > 1$) & General $M \ge 2$:** Sequential greedy QMC `qEHVI` using $S = 256$ scrambled Sobol / antithetic normal draws per shortlisted candidate $Y^{(s)} = \mu + \sigma \odot Z^{(s)}$: at step $t \in \{1\dots B\}$, select $j_t = \arg\max_{j \notin S} \frac{1}{S}\sum_{s=1}^S \big[\text{HV}(\mathcal{P} \cup \{Y_{j_1}^{(s)}, \dots, Y_{j_{t-1}}^{(s)}, Y_j^{(s)}\}; r) - \text{HV}(\mathcal{P} \cup \{Y_{j_1}^{(s)}, \dots, Y_{j_{t-1}}^{(s)}\}; r)\big]$. Log-stabilized tie-break when EHVI approaches 0.
   - `"pareto_scalarized_ucb"` (ParEGO / Augmented Tchebycheff): Normalizes each objective UCB $\tilde{u}_m(x) = (\mu_m(x) + \beta\sigma_m(x) - \min_i \mu_m) / \text{range}_m$; for each batch slot $b = 1 \dots B$, uses either user weights or stratified Dirichlet weights $w^{(b)} \sim \text{Dirichlet}(\mathbf{1})$ to select $\arg\max_x \big[\min_m w_m^{(b)} \tilde{u}_m(x) + 0.05 \sum_m w_m^{(b)} \tilde{u}_m(x)\big]$ without replacement, tracing the full Pareto front smoothly even for $M = 3\text{–}6$ objectives!

4. **Feasibility-Constrained Acquisitions (Potency $\times$ Avoid-ome / Profile / PoseReality):**
   - Helper `build_feasibility_vector(smiles, *, property_profile=None, admet_suite=None, herg_max=0.5, bbb_min=None, triage_hard_gate=True, custom_constraint_fns=()) -> NDArray[np.float64]`: computes smooth feasibility probability $P_{\text{feas}}(x) \in [0, 1]$ from GenEuclidean's `triage_molecule`, `PROPERTY_PROFILE_REGISTRY`, `CNSAdmetSuite`, or probabilistic constraint surrogates $\Phi((\tau_j - \mu_{c_j})/\sigma_{c_j})$.
   - `"constrained_ei"`: $\alpha_{\text{CEI}}(x) = \text{EI}(x) \cdot P_{\text{feas}}(x)$ (with `log_ei + log(max(P_feas, 1e-12))` tie-breaker).
   - `"avoidome_feasibility_ucb"`: $\alpha(x) = (\text{UCB}(x) - \min_i \text{UCB}(x_i) + 10^{-6}) \cdot P_{\text{feas}}(x)$ — strictly preserves feasibility monotonicity even when objective values are negative!

---

#### D. `multifidelity.py` — Cost-Aware Correlation-Gated Multi-Fidelity DMTA Router

```python
@dataclass(frozen=True)
class FidelityTierSpec:
    name: str                                          # e.g. "tier0_drugclip", "tier1_vina_audit", "tier2_rbfe_wetlab"
    level: int                                         # 0 (cheapest) < 1 < ... < K-1 (ground truth)
    cost: float                                        # relative unit cost (e.g. 1.0, 20.0, 200.0)
    oracle_fn: Callable[[Sequence[str]], Sequence[float]]
    hard_gate_fn: Callable[[Sequence[str]], Sequence[bool]] | None = None  # e.g. PoseRealityAuditor physical pass
```

- **`MultiFidelityDMTA`:**
  - Maintains per-tier observation stores `observations[level]: dict[str, float]` and cumulative spent budget `total_cost_spent: float`.
  - **Autoregressive Linear-Residual Surrogate (Kennedy–O'Hagan):**
    - Highest tier $H = K - 1$ predictor uses paired observations between tier $L$ and tier $H$: fits Ridge-regularized linear bridge $\hat{f}_H^{\text{lin}}(x) = \hat{\alpha} + \hat{\rho}\, f_L(x)$ (for $n_{\text{paired}} \ge 6$), then trains a residual `TanimotoGP` $\delta(x)$ on $r_i = f_H(x_i) - \hat{f}_H^{\text{lin}}(x_i)$.
    - Predicts $\hat{\mu}_H(x) = \hat{\alpha} + \hat{\rho}\,\hat{f}_L(x) + \hat{\mu}_\delta(x)$ and $\hat{\sigma}_H(x) = \sqrt{\hat{\sigma}_\delta^2(x) + \hat{\rho}^2 \hat{\sigma}_L^2(x)}$ (where $\hat{\sigma}_L(x) = 0$ if $x$ was already evaluated at tier $L$!).
  - **Online Correlation Gate (Critical Anti-Hype Safeguard!):**
    - Computes Spearman rank correlation $\hat{r}_s$ on paired `(tier L, tier H)` observations.
    - If $n_{\text{paired}} \ge 8$ and $\hat{r}_s < \rho_{\min}$ (default `0.25`): sets `low_fidelity_trusted = False`, logs diagnostic warning, and routes remaining budget directly to Tier $H$ via pure high-fidelity `TanimotoGP` (zero negative transfer from garbage docking/proxy scores!).
    - When `low_fidelity_trusted = True`: computes cost-normalized VOI $\text{VOI}_k(x) = \frac{\text{EI}_H(x) \cdot \text{gain}_k(\hat{r}_s)}{c_k}$ (where lower tier can reduce variance by factor $\hat{r}_s^2$ at cost $c_L \ll c_H$), screening a wide Tier-0/1 funnel and promoting only top high-VOI + `hard_gate_fn`-passing compounds to Tier $H$.

---

#### E. `combinatorial_ts.py` — Reaction-Space Thompson Sampling ($O(N_A + N_B)$)

```python
@dataclass
class ReagentPosterior:
    reagent_id: str
    smiles: str
    role: Literal["A", "B"]
    mean: float
    std: float
    n_obs: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0

class CombinatorialThompsonSampler:
    def __init__(
        self,
        reaction_name: str = "amide_coupling_acid_amine",
        *,
        custom_smirks: str | None = None,
        warmup_per_reagent: int = 3,
        temperature: float = 1.0,
        selection_mode: Literal["argmax", "boltzmann_roulette"] = "argmax",
        triage_filter: bool = True,
        uncharge_reactants: bool = True,
        seed: int = 42,
    ) -> None: ...

    def run(
        self,
        reactants_a: Sequence[str],
        reactants_b: Sequence[str],
        oracle_fn: Callable[[Sequence[str]], Sequence[float]],
        *,
        total_budget: int = 400,
        batch_size: int = 16,
    ) -> CombinatorialTSResult: ...
```

- **Strict Ground-Truth Isolation Invariant:** `CombinatorialThompsonSampler` receives ONLY reactant lists `reactants_a`, `reactants_b`, and `oracle_fn`. It **never enumerates** the full $N_A \times N_B$ table and **never calls `oracle_fn`** on any product other than the $\le \text{total\_budget}$ selected pairs!
- **Exact Conjugate Normal-Normal Updates:**
  1. **Warm-up Phase:** Pair each $a \in A$ and $b \in B$ with `warmup_per_reagent` random valid partners (capped at $\le 40\%$ of `total_budget` if budget is small). Compute empirical global prior $(\mu_0, \sigma_0^2)$ and observation noise variance $\sigma_{\text{obs}}^2 = \max(\text{Var}(y_{\text{warm}}), 10^{-4})$ from warm-up scores.
  2. **Conjugate Reagent Update:** For every observed product $(a, b) \mapsto y$, update reagent $a$ and reagent $b$ independently in $O(1)$:
     $$\tau_r = \frac{1}{\sigma_0^2} + \frac{n_r}{\sigma_{\text{obs}}^2}, \quad \sigma_r = \frac{1}{\sqrt{\tau_r}}, \quad \mu_r = \sigma_r^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_{i=1}^{n_r} y_i}{\sigma_{\text{obs}}^2}\right)$$
  3. **Sampling Phase ($O(N_A + N_B)$ per proposal):** Draw $\tilde{\theta}_a \sim \mathcal{N}(\mu_a, (T\sigma_a)^2)$ for all $a \in A$ and $\tilde{\theta}_b \sim \mathcal{N}(\mu_b, (T\sigma_b)^2)$ for all $b \in B$. Either take top unattempted pair maximizing $\tilde{\theta}_a + \tilde{\theta}_b$ (`argmax`, checking only top-$k$ lists of $A$ and $B$ so selection is $O(N_A + N_B)$, never $O(N_A N_B)$!) or sample via softmax roulette weights (`boltzmann_roulette`, AstraZeneca 2024). React on-the-fly with `ROBUST_MEDCHEM_REACTIONS`, sanitize, apply optional `triage_molecule`, deduplicate canonical product SMILES, call `oracle_fn` in batches of `batch_size`, and update posteriors.
  4. **Diagnostics Returned:** `n_virtual_space = N_A * N_B`, `n_oracle_calls`, `fraction_explored`, `top_products` (with explicit `reactant_a`, `reactant_b`, `reaction_name`, `product_smiles`, `score`), `reagent_additivity_r2` (ANOVA / additive main-effects $R^2$ computed strictly on the *observed* pairs to warn users honestly if strong epistasis is limiting TS!), and SHA-256 ledger digest.

---

#### F. `diagnostics.py` & `engine.py` — Cold-Start, Calibration, Stopping & Async DMTA Engine

1. **Cold-Start Seeding (`select_cold_start`):**
   - `"butina_medoids"`: Computes pairwise ECFP4 Tanimoto matrix (or chunked leader-follower clustering for large pools $N > 2{,}500$), clusters at cutoff `tanimoto_cutoff = 0.45`, and selects medoids of the $B$ largest distinct clusters (falling back to MaxMin if fewer than $B$ clusters exist).
   - `"maxmin"`: Exact greedy MaxMin Tanimoto dissimilarity picker starting from the medoid of the pool (deterministic given `seed`).
   - `"random"`: Reproducible RNG baseline.
2. **Uncertainty Calibration & Scaffold-Stratified Diagnostics (`evaluate_calibration`):**
   - Computes: `rmse`, `mae`, `r2`, `spearman_rho`, `gaussian_nll` ($\frac{1}{2}\mathbb{E}[\log(2\pi\sigma^2) + (y-\mu)^2/\sigma^2]$), `ece_regression` (Gaussian quantile calibration error across nominal confidence levels $\{0.50, 0.68, 0.80, 0.90, 0.95\}$), `error_std_spearman` ($\rho_s(|y - \hat{\mu}|, \hat{\sigma})$ — measures whether uncertainty actually ranks error magnitude!), `conformal_empirical_coverage`, `conformal_mean_width`, AND **`similarity_stratified_coverage`** across 3 max-Tanimoto buckets (`< 0.35` novel scaffold, `0.35–0.55` moderate shift, `>= 0.55` in-domain).
3. **Principled Stopping Diagnostics (`StoppingAdvisor`):**
   - Evaluates 3 independent convergence criteria after each round:
     - `ei_saturated`: $\max_{x \in \mathcal{U}} \text{EI}(x) / \max(\text{std}(y_{\text{obs}}), 10^{-6}) < \epsilon_{\text{EI}}$ (default `0.01`) for `patience` rounds;
     - `rank_stable`: Top-$K$ pool prediction Jaccard overlap $J(\text{Top}_K^{(t)}, \text{Top}_K^{(t-1)}) \ge \tau_J$ (default `0.85`, $K = 25$);
     - `improvement_plateau`: best observed score (or Pareto HV) improved by $< \epsilon_{\text{imp}}$ over last `patience` (default `3`) rounds.
   - Emits `StoppingReport(should_stop: bool, triggered_reasons: tuple[str, ...], max_rel_ei: float, topk_jaccard: float | None, rounds_since_improvement: int)`.
4. **Unified Orchestrator (`ActiveLearningEngine` in `engine.py`):**
   - Clean immutable records: `DMTAQuery` (batch ID, round, proposed SMILES, acquisition scores, predicted means/stds, conformal intervals, status `"pending"|"completed"|"failed"`), `DMTARoundSummary`, `DMTALedger` (append-only SHA-256 hash chain across rounds for FDA/EMA Credibility Dossier auditability!).
   - **Full Async Delayed-Batch Workflow:**
     - `query = engine.propose_batch(pool_smiles, batch_size=16, mark_pending=True)` $\to$ records `query.smiles` in `self.pending` so subsequent `propose_batch` calls *before* assay completion automatically exclude `pending` SMILES AND pass them as `pending_smiles` into `KrigingBeliever` / `MaxMin` / `DPP`!
     - `engine.complete_observations(results_dict)` (or partial completion of any subset of pending molecules, plus `engine.cancel_pending(failed_smiles)` for failed syntheses!) removes completed/failed SMILES from `self.pending` and appends valid measurements to `self.observed`.
     - Synchronous helper `engine.run_round(pool_smiles, oracle_fn, batch_size=16)` runs propose $\to$ evaluate $\to$ ingest $\to$ stopping check in one call.
   - Supports single-objective (`y: float`), multi-objective (`y: dict[str, float]`, fitting one surrogate per objective for `qehvi` / `pareto_scalarized_ucb`), and feasibility-constrained modes seamlessly.

---

## 5. Exact Retrospective Benchmark Protocol (100% Real Local Data)

Every benchmark in `scripts/run_active_learning_benchmark.py` (persisting verified output to `results/tables/active_learning_v51_benchmark.json`) MUST use **exclusively real experimental datasets already verified under `data/`** — zero synthetic mathematical toy functions!

### 5.1 Verified Local Dataset Inventory

| Dataset File (relative to repo root) | Real Content | Verified Rows | Unique Murcko Scaffolds | Role in Benchmark |
| :--- | :--- | :---: | :---: | :--- |
| `data/cns_neuro.nosync/chembl/human_cns_actives_12targets.csv` | Human ChEMBL CNS bioactivities (`median_pchembl`) across 12 targets (`DRD1` `n=1,128`, `DRD2` `n=7,091`, `PDE10A` `n=5,045`, etc.) | `36,328` | `DRD1`: `424`; `DRD2`: `3,021` | **Suite 1** (`DRD1`), **Suite 2A** (MO + Avoid-ome), **Suite 2B** (`498` shared `DRD1`+`DRD2` experimental selectivity!), **Suite 3** (Multi-Fidelity) |
| `data/cns_neuro.nosync/v4_expansion/chembl_v4_12new_targets.csv` | Human ChEMBL v4 expansion (`ACHE` `n=3,051`, `MAOB` `n=2,280`, `MAOA` `n=826`, `BACE1` `n=7,468`, etc.) | `30,541` | `ACHE`: `1,495`; `MAOB`: `665` | **Suite 1** (`ACHE` enzyme + `MAOB` oxidoreductase — 3 diverse target families!), **Suite 5** (Conformal shift) |
| `data/cns_neuro.nosync/admet/B3DB_regression.tsv` | Experimental Blood-Brain Barrier `logBB` $\in [-2.69, +1.70]$ | `1,058` | `517` | **Suite 5** (Strict 5-fold scaffold-CV Conformal UQ calibration & similarity-stratified coverage) |
| `data/cns_neuro.nosync/v4_admet_avoidome_oracles.joblib` & `v4_24target_qsar.joblib` | Trained GenEuclidean `CNSAdmetSuite` (12 endpoints incl. hERG, 5-HT2B, MDR1) & 24-target QSAR models | — | — | **Suite 2A** (`P(hERG)` Avoid-ome gate), **Suite 3** (Tier-1 proxy), **Suite 4** (Make-on-demand virtual library oracle) |
| `data/raw/zinc250k.csv` | Real drug-like commercial molecules (requires `rdMolStandardize.Uncharger()` before SMIRKS matching!) | `250,000` | `141` acids & `968` amines in first `15k` (`HA <= 18`) | **Suite 4** (Real `60 x 80 = 4,800` amide-coupling combinatorial reaction space for `CombinatorialThompsonSampler`) |

### 5.2 Five-Suite Retrospective Protocol & Mandatory Reporting Metrics

1. **Suite 1 — Single-Objective Pool-Based AL under Strict Bemis–Murcko Scaffold Deduplication (3 Diverse Targets: `DRD1` GPCR `424` scafs, `MAOB` Oxidoreductase `665` scafs, `ACHE` Hydrolase `800` deterministically sampled unique scafs):**
   - **Zero-leakage pool construction:** Group by canonical Bemis–Murcko scaffold (`includeChirality=False`); keep strictly 1 molecule per scaffold (the molecule with highest `n_measurements`, tie-broken deterministically by canonical SMILES — never cherry-picking by label!).
   - **Budget:** Multi-seed (`seeds = [42, 137, 2025]`, report mean ± std). Round 0 cold-start = `16` compounds (`butina_medoids`, plus dedicated ablation of `butina_medoids` vs `maxmin` vs `random` on `DRD1`), followed by `5` DMTA rounds $\times$ `16` compounds/round = **`96` total compounds** (`12%–22.6%` of unique-scaffold pool).
   - **8 Arms:** `random`, `ensemble+greedy`, `tanimoto_gp+greedy`, `tanimoto_gp+ucb`, `tanimoto_gp+ei`, `tanimoto_gp+kriging_believer`, `tanimoto_gp+dpp_ucb`, `tanimoto_gp+maxmin_ucb`.
   - **Mandatory Metrics:** `top5pct_recall` (mean ± std), `enrichment_vs_random`, `best_pchembl`, `mean_top10_pchembl`, `mean_batch_internal_tanimoto` (lower = better batch diversity), `final_spearman_rho`, `stopping_round_triggered`.
   - **Honesty Requirement:** Explicitly report where `tanimoto_gp+greedy` matches or beats `ucb`/`ei` on raw Top-5% recall, alongside whether `kriging_believer` / `dpp_ucb` achieves strictly lower batch redundancy (`mean_batch_internal_tanimoto`) and higher diverse hit coverage!

2. **Suite 2 — Multi-Objective Pareto & Feasibility-Constrained DMTA:**
   - **Task 2A (`DRD1` `424` unique scaffolds — Potency $\times$ CNS MPO v2 + Avoid-ome `P(hERG) < 0.5` & MedChem Triage):**
     - Budget: `64` compounds (`4` rounds $\times 16$, 3 seeds).
     - Compare `random`, `tanimoto_gp+greedy` (potency-only baseline), `tanimoto_gp+qehvi`, `tanimoto_gp+pareto_scalarized_ucb`, and `tanimoto_gp+constrained_ei`.
     - Report: **Pareto Hypervolume (HV)** (at fixed reference point `(pChEMBL=5.0, mpo_norm=0.0)`), **Feasible Potent Hit Count** (`pChEMBL >= 8.0 & CNS_MPO_v2 >= 4.0 & P(hERG) < 0.5 & triage_pass`), and **Avoid-ome Failure Rate of Top Picks** (quantifying potency-only failure!).
   - **Task 2B (True Dual-Assay Paralog Selectivity — `498` shared `DRD1`/`DRD2` ChEMBL compounds):**
     - Maximize 2 real experimental objectives `(pChEMBL_DRD1, -pChEMBL_DRD2)` (selective DRD1 over DRD2), budget `64` compounds, comparing `random` vs `greedy_drd1_only` vs `qehvi` vs `pareto_scalarized_ucb` on **2D Pareto Hypervolume** and **Selective Hit Count** (`pChEMBL_DRD1 >= 8.0` and `pChEMBL_DRD1 - pChEMBL_DRD2 >= 1.5` log units $\approx 30\times$ selectivity!).

3. **Suite 3 — Multi-Fidelity Routing (Informative Proxy vs. Adversarial Low-Correlation Proxy):**
   - Pool: `DRD1` (`424` unique scaffolds). High-fidelity Tier-2 oracle = exact experimental `median_pchembl` (cost `20.0`).
   - **Condition A (Informative Tier-1 proxy, cost `1.0`):** Cross-validated OOF QSAR or real `pChEMBL` + calibrated physical noise ($\rho_s \approx 0.65\text{–}0.75$).
   - **Condition B (Adversarial/noisy Tier-1 proxy, cost `1.0`, simulating untuned docking / zero-shot `DrugCLIP`):** Low-correlation proxy ($\rho_s \approx 0.10\text{–}0.15$).
   - Fixed total cost budget `C = 800` units (= 40 Tier-2 calls alone). Compare single-fidelity `TanimotoGP+EI`, ungated MF-BO, and **correlation-gated `MultiFidelityDMTA`** on Top-5% recall per unit cost, explicitly verifying that correlation gating saves cost in Condition A AND avoids negative transfer in Condition B!

4. **Suite 4 — Reaction-Space Combinatorial Thompson Sampling (Real ZINC250k Building Blocks + `amide_coupling_acid_amine`):**
   - Extract the first `60` unique uncharged carboxylic acids and `80` unique uncharged primary/secondary amines from `data/raw/zinc250k.csv` (`HA <= 18`, `rdMolStandardize.Uncharger()`) $\to$ real **`4,800`-pair (`60 x 80`)** make-on-demand space.
   - Oracle: Real trained `v4_24target_qsar` (`ACHE` probability) $\times$ (`CNS_MPO_v2 / 6.0`) $\times$ `hygiene_gate`.
   - Ground-truth verification (strictly post-hoc, NEVER exposed to the sampler): enumerate all valid products to determine the exact true Top-50 set.
   - Budget: `360` oracle calls (`7.5%` of `4,800` space, warm-up $w = 2\text{–}3$, 3 seeds). Compare `RandomPairSampling` vs `CombinatorialThompsonSampler(argmax)` vs `CombinatorialThompsonSampler(boltzmann_roulette)` on **Top-50 Recall**, **Enrichment Factor**, **Best Score Found**, **Wall-Clock Time**, and **Empirical Reagent Additivity $R^2$**.

5. **Suite 5 — Uncertainty Calibration & Conformal Coverage under Strict Bemis–Murcko Scaffold Shift:**
   - Datasets: `B3DB_regression.tsv` (`1,058` experimental `logBB`) and `MAOB` (`2,280` experimental `median_pchembl`), strict 5-fold Bemis–Murcko scaffold CV (`alpha = 0.10`, nominal target `90.0%`).
   - Compare 5 UQ methods: (1) raw `TanimotoGP` ($\pm 1.645\sigma$), (2) raw `EnsembleSurrogate`, (3) `ConformalSurrogate(TanimotoGP, mode="absolute")`, (4) `ConformalSurrogate(TanimotoGP, mode="normalized")`, (5) `ConformalSurrogate(TanimotoGP, mode="weighted")`.
   - Report overall coverage, mean interval width, NLL, error-std Spearman $\rho$, AND **similarity-stratified empirical coverage** (`max_tan < 0.35`, `0.35–0.55`, `>= 0.55`) — honestly surfacing any residual undercoverage in the `< 0.35` structural-frontier bucket!

---

## 6. Key References

1. Graff, D. E., Shakhnovich, E. I., & Coley, C. W. (2021). *Accelerating high-throughput virtual screening through molecular pool-based active learning.* Chemical Science, 12(22), 7866–7881. (`MolPAL`)
2. Griffiths, R.-R., Klarner, L., Moss, H. B., et al. (2023). *GAUCHE: A Library for Gaussian Processes in Chemistry.* NeurIPS 2023.
3. Tripp, A., Bacallado, S., Singh, S., & Hernández-Lobato, J. M. (2023). *Tanimoto Random Features for Scalable Molecular Machine Learning.* NeurIPS 2023.
4. Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2023). *Unexpected Improvements to Expected Improvement for Bayesian Optimization.* NeurIPS 2023 (Spotlight). (`qLogEI`, `qLogNEHVI`)
5. Klaric, K., Goldman, B., Kramer, T., Riley, P., & Walters, W. P. (2024). *Thompson Sampling—An Efficient Method for Searching Ultralarge Synthesis on Demand Databases.* JCIM, 64(4), 1158–1171.
6. Konze, K. D., et al. (2019). *Reaction-Based Enumeration, Active Learning, and Free Energy Calculations to Rapidly Explore Synthetically Tractable Chemical Space.* JCIM, 59(9), 3782–3793; Thompson, J., et al. (2022). *Optimizing Active Learning for Free Energy Calculations.* Artificial Intelligence in the Life Sciences, 2, 100050.
7. van Tilborg, D., & Grisoni, F. (2024). *Traversing chemical space with active deep learning for low-data drug discovery.* Nature Computational Science, 4, 786–796; van Tilborg, D., Alenicheva, A., & Grisoni, F. (2022). *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs.* JCIM (`MoleculeACE`).
8. Loeffler, H. H., et al. (2024). *Reinvent 4: Modern AI-driven generative molecule design.* Journal of Cheminformatics, 16, 20.
9. Bengs, V., Hüllermeier, E., & Waegeman, W. (2023). *Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?* NeurIPS 2023.
10. Laghuvarapu, S., Lin, Z., & Sun, J. (2023). *CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift.* NeurIPS 2023; Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). *Conformal Prediction Beyond Exchangeability.* Annals of Statistics.
11. Chen, L., Zhang, G., & Zhou, H. (2018). *Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity.* NeurIPS 2018.
12. Ginsbourger, D., Le Riche, R., & Carraro, L. (2010). *Kriging Is Well-Suited to Parallelize Optimization.* (`Kriging Believer` / `Constant Liar`); Sabanza-Gil, V., et al. (2024–2025). *Best Practices for Multi-Fidelity Bayesian Optimization in Chemistry.*
