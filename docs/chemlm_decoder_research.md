# ChemLM Decoder Suite — Research Findings & Design Decisions (v6.0)

Research synthesis behind the `chemlm` / `chemlm_moe` decoders and the CrossDocked2020
corpus support added in `feat/molecular-llm-decoder-suite`. Conducted 2026-11-26 by four
parallel research agents; claims below are tagged with how they were established.

Legend: **[V]** verified against a primary source (paper text or released code) ·
**[M]** measured on hardware during this work · **[U]** reported but unverified, flagged.

---

## 1. The headline finding: bigger is not better here

The brief asked for "as powerful an LLM as possible". The evidence says that is the wrong
objective for de novo molecular design, and it says so unusually clearly.

**[V] NovoMolGen / "A Large Scale Investigation of Scaling Limits in Chemical Language
Models"** ([arXiv:2508.13408**v3**](https://arxiv.org/abs/2508.13408v3), >30,000
experiments) states verbatim: *"Downstream goal-directed generation largely saturates
beyond approximately 5M parameters, despite improvements in bits per molecule and linear
accessibility of chemical features."*

> **Cite the version.** This arXiv id hosts two substantially different studies. v1/v2
> ("NovoMolGen: Rethinking Molecular Language Model Pretraining") train 32M/157M/300M
> models on Llama and contain **no 5M or 1B model and no total docking score**; v3
> (1 Oct 2026, retitled) trains 42 models over 0.5M–1B on Qwen3/T5Gemma across four
> corpora. Every figure below is v3's. An earlier revision of this document cited the
> bare id, so a reader checking v1 would have found the central claim absent and
> reasonably concluded it was invented.

| Metric | 5M | 32M | 300M | 1B |
|---|---|---|---|---|
| Loss (bits/molecule, PubChem SMILES atomwise) | 54.1 | 48.1 | — | — |
| Total docking top-5% (5 targets, kcal/mol) | −65.8 | −66.3 | −66.1 | −66.2 |

v3 §3.5, verbatim: *"Gains beyond 5M are marginal, the total docking score moving only
from −65.8 at 5M to −66.2 at 1B."* 200× the parameters buys **0.4 kcal/mol**, inside the
noise.

For PMO, v3's own framing is the one to quote (§3.5): NovoMolGen *"surpasses REINVENT …
and ff-RAG from 5M parameters onwards, scoring 16.45 against 16.30 for ff-RAG"*, and
*"[s]caling to 1B raises the total only to 16.68."* REINVENT scores 14.01.

> **Correction.** This table previously read "PMO sum 16.48 (32M) → 16.70 (300M)". Those
> two numbers are real but come from *different tokenizations* — 16.48 is Table I1
> (AtomWise, 32M) and 16.70 is Table I2 (BPE, 300M) — so presenting them as one
> 32M→300M trend silently spliced two series. Like-for-like it is 16.48 → 16.64
> (AtomWise) and 16.68 → 16.70 (BPE).

**[V] Xu et al.** ([arXiv:2601.22757](https://arxiv.org/abs/2601.22757), "Unveiling
Scaling Behaviors in Molecular Language Models", 30 Jan 2026) *"train 300 models and
conduct over 10,000 experiments"*, fit `L(P,D) = L∞ + k_P·P^(−α) + k_D·D^(−β)` per
representation, and report for SMILES (their Table 1) **α = 0.0171** against
**β = 0.4299**. The other representations: DeepSMILES 0.0588/0.3624, FragLink
0.0282/0.5214, FragSeq 0.0189/0.5207, SAFE 0.0200/0.2001. β > α everywhere.

> **Correction.** This document previously concluded "*data buys ~25× more than
> parameters*". That does not follow. α and β are **exponents**: per decade of scaling
> the data term falls by 10^0.4299 ≈ 2.7× against 10^0.0171 ≈ 1.04× for parameters, but
> each term's absolute contribution also carries k_P / k_D, which the paper does not
> report. The defensible claim is the ordering — *data scaling dominates parameter
> scaling for SMILES* — not a 25× loss-reduction ratio. The ratio is ours, not either
> paper's. **[U]** The further attributions that Xu et al. "concede de novo metrics
> saturate early" and "cannot serve as primary evidence for or against scaling" could
> not be verified verbatim in the retrievable text and are demoted from quotations to
> unverified paraphrase.

The two papers look contradictory and are not: one measures loss, the other design
quality, and the mechanism below explains why they diverge.

**Mechanism [V]:** chemical *syntax* saturates early (a validity probe reaches 90% of its
final accuracy at 0.1 tokens/param) while docking/PMO *semantics* need 0.6 and
physicochemical properties 1.5.

### What this changed in the code

`SCALE_PRESETS` still spans `tiny` (4.0M) → `xl` (1.32B) so the claim can be tested rather
than assumed, but the recommended default is now **`small` (28.6M)**, and `large`/`xl`
are documented as capacity-study tools, not quality upgrades. The docstring carries the
numbers above so nobody re-derives this from scratch.

---

## 2. Two traps that will produce a misleading result

**[V] Loss and sample quality can move in opposite directions.** arXiv:2508.13408v3 §3.3,
verbatim: *"within PubChem SMILES, the FCD rises from 2.8 to 5.1 between 5M and 32M, even
as the loss falls from 54.1 to 48.1."* Never model-select on validation loss alone. The
notebook reports sample metrics separately and never selects on loss. (The related
FCD-vs-PMO correlation **r = 0.376, p = 0.358** is from **v2** of that paper, not v3.)

**[V] Loss is not comparable across corpora.** Bits/molecule at 32M: MOSES 25.6, ChEMBL
33.2, PubChem 48.1, ZINC-22 51.5. A MOSES model is not twice as good as a ZINC-22 model;
MOSES is narrower.

**[V] Memorization, not invalidity, is the real failure mode.** GP-MoLFormer trained on
1.1B SMILES reports **novelty 0.323** at 30k samples — 68% of generations are *verbatim*
training molecules — and ~45% of that corpus were duplicates. Deduplicating to 650M
unique lifted novelty to 0.390. Novelty also *decays* with sample count: 0.390 @30k →
0.340 @1B → 0.214 @10B.

**Acted on:** `data/chem_corpora.py` provides canonical-form deduplication and a novelty
report, and the notebook deduplicates before training and measures novelty against the
deduplicated set.

**[M] Measured correction — this particular win does not apply here.** ZINC-250k as
shipped in this repo is *already* clean: 20,000 molecules sampled gave **0 duplicates and
0 invalid (0.0% duplicate rate)**. The 45%-duplicate figure comes from large scraped
corpora (PubChem / ZINC-22 merges), so deduplication becomes valuable only when moving
to those. Stated explicitly so the dedup step is not credited with an improvement it
cannot produce on the current corpus.

The `novelty_report` half matters regardless, and two of its details are deliberate:
invalid strings stay in the validity denominator (scoring only the valid subset silently
inflates validity), and `n_generated` is recorded alongside novelty, because novelty
decays with sample count and a novelty figure without its sample size is not
interpretable.

---

## 3. Representation and tokenization

**[V]** NovoMolGen's 30k-experiment sweep puts **canonical SMILES with an atom-level regex
tokenizer** lowest-loss at *every* scale: 2.1 bits/molecule below SAFE and 2.8 below
SELFIES at 32M params.

SELFIES' 100% validity is **not** the win it appears to be: the guarantee is syntactic,
and it costs distribution quality. Skinnider (*Nat. Mach. Intell.* 2024) argues directly
that invalid SMILES are beneficial rather than detrimental to chemical LMs — the model
uses them as a learning signal.

This library already used the canonical Schwaller atom-level regex, so no change was
needed. SAFE remains available and is the right choice for *fragment-constrained* design
specifically, not for unconstrained generation.

---

## 4. Property conditioning — mechanism matters more than capacity

**[V] Conditioning collapse is real and measured.** MolGPT-style prefix conditioning on 7
simultaneous RDKit properties collapses validity to **25.7%**, while a classifier-free
guidance model trained with property dropout (p≈0.2) holds **100%**
([Fraglingo](https://arxiv.org/abs/2609.13519) Table 1). Notably MolGPT keeps the *better*
per-property Spearman (0.7816 vs 0.7393) — it obeys each constraint but leaves the
valid-SMILES manifold. PolymerGPT shows the same shape: validity 98%+ at ≤3 properties
falling to 64.68% at 5.

**Acted on:** `PropertyConditioner` implements per-property `unknown` dropout
(`value_dropout_prob`, default 0.15; the literature value is ≈0.2) which is the mechanism
with evidence behind it, and gives free unconditional sampling plus a tunable guidance
scale at inference.

**⚠️ Honest caveat on our own design.** The value encoder uses **random Fourier features**
(Tancik et al., NeurIPS 2020). A survey of the chemical-LM literature found property
*text tags* (Chemlactica/Chemma), *linear projection to prefix tokens* (MolGPT,
PolymerGPT) and *additive injection with CFG* (Fraglingo) — but **no published use of
random Fourier features for molecular property conditioning, and no paper ablating binned
vs continuous encodings at all.** This is therefore a promising but externally unvalidated
choice; the docstring says so, and it should be benchmarked against a plain linear
projection before being relied on.

Other evidence-backed mitigations not yet implemented, in priority order:

1. Randomize property-tag order and molecule position (Chemlactica) so the model cannot
   key on positional slots.
2. Report *partial* satisfaction, not just joint-AND (Fraglingo: 6.56/7 properties
   satisfied even when joint success is 69.65%).
3. If RL is added, include an explicit anti-collapse penalty plus experience replay.
   NovoMolGen needed a **576-run** hyperparameter search and found the REINVENT default
   λ=5000 unstable.

---

## 5. Datasets — the premises that turned out to be wrong

Four of the assumptions this work started from were false. All were established by direct
HTTP probing, not by reading summaries.

| Premise | Reality |
|---|---|
| PLINDER gives ~400k holo pockets to train on | **[V] The current release ships no structures and no splits.** `PLINDER-2026-09` contains only `index/`, `exports/`, `ligand_*`, `protein_clusters/`, `search_databases/`; `systems/`, `splits/`, `entries/` all 404. The docs confirm mmCIFs are fetched per entry at reconstruction time. Published splits live only in historical GCS releases, which returned 401 anonymously. PLINDER today is an *annotation and similarity* resource. Current counts: 611,930 PLI systems, 803,190 ligands. |
| Binding MOAD is a usable corpus | **[V] Shut down 2024-07-01, and the domain is now squatted** — `bindingmoad.org` serves an unrelated commercial page. ⚠️ **Do not download anything from it.** Only the affinity CSV survives, on GitHub. Data frozen at 41,409 structures / 15,223 with affinity. *(Verified: this repo never referenced that domain.)* |
| PDBbind + CASF-2016 are freely downloadable | **[V] Paywalled from v2021.** `pdbbind-plus.org.cn/api/download` returns `{"code":10011,"msg":"Please log in first"}`. The `wget .../CASF-2016.tar.gz` baked into years of repos now fails. *(This repo already uses a working Zenodo mirror for CASF-2016 and a HuggingFace mirror for PDBbind-pp 2020.)* |
| Pocket radius alone defines a pocket | **[V] Three incompatible conventions are in use.** TargetDiff/CrossDocked lineage: 10 Å to residue **centre of mass** (`criterion='center_of_mass'`). DiffSBDD: 8 Å **any-atom**. PLINDER: PLIP interaction + 6 Å. **This library uses 10 Å any-atom** — a fourth, more inclusive variant. |

### The reframing that matters

For a **pocket→SMILES** decoder, ligand *poses* are irrelevant — only (pocket, SMILES)
pairs are needed. That unlocks corpora the 3D-generative literature cannot use, and the
biggest usable corpus is not a download but a **join**: holo pockets ⋈ UniProt ⋈ ChEMBL 37
(24.5M activities) / BindingDB. Every activity record for a target whose pocket is known
becomes a (pocket, SMILES, pActivity) triple — tens of millions of pairs against
CrossDocked's 100k. **[U]** SAIR (SandboxAQ, 1,048,857 pairs, CC BY 4.0, commercial-OK) is
essentially a precomputed version of that join, co-folded with Boltz-1x.

### The free 1.84× nobody takes

**[V]** Luo et al. (NeurIPS 2021), the origin of the canonical CrossDocked split, filter
22.5M poses to RMSD ≤ 1 Å — yielding **184,057 pairs** — then *"randomly draw 100,000
protein-ligand pairs for training"*. **~84,000 qualifying complexes are discarded by every
baseline in that lineage.** `build_rmsd_filtered_pairs` keeps all of them by default;
`limit=100_000` reproduces the published baseline.

CrossDocked2020 is also **[V] CC0 1.0** (license file fetched directly), making it the only
major option with zero legal friction.

### Splitting honestly

Pocket-group disjointness is the floor, not the bar. **[V]** DockGen shows proteins at 22%
sequence identity can share a binding site; CleanSplit shows 53%-identity pairs reaching
TM-score 0.93 with identical sites; Runs N' Poses finds ~75% of post-Sept-2021 PDB systems
still have >50 SuCOS pocket similarity to training data, because new depositions are
overwhelmingly new ligands in old pockets. `make_pocket_disjoint_split` records a
`requires_followup` field stating that structure clustering (Foldseek / TM-align / ECOD
T-group) and ligand-Tanimoto filtering have **not** been applied.

**[V] An open gap worth knowing:** no published paper directly measures CrossDocked
test-pocket→train-pocket similarity and the resulting Vina inflation. Adjacent evidence is
strong though — PoseCheck median strain energy is 1241.7 kcal/mol for TargetDiff against
102.5 for CrossDocked reference ligands (>12×), GenBench3D finds only 0–11% of generated
molecules have valid conformations, and for several models *relaxed* molecules score
**worse** on Vina than raw ones, meaning the raw numbers are inflated by physically
invalid geometry.

---

## 6. Apple Silicon MPS — measured, not assumed

**[M]** All figures below were measured on this machine (M3 Pro, 36 GiB, torch 2.15.0).

| Finding | Detail |
|---|---|
| **Memory is not the constraint** | At 5M–200M params, optimizer state is 0.3%–11% of the ~28 GiB budget. `recommended_max_memory()` = 28.08 GiB = exactly 78% of `hw.memsize`. |
| **Gradient checkpointing: off by default, but it does save memory** | Re-measured at the `small` preset, batch 32, seq 96: `driver_allocated_memory()` falls **2672.9 MB → 1360.9 MB (−49%)**. An earlier revision of this table claimed "4.0× activation saving … ~17% throughput" with no recorded provenance, and asserted there was no memory gain — that conclusion came from reading `current_allocated_memory()`, which is sampled after the backward pass has freed its activations and reports the two configurations as identical to within 0.3 MB. **MPS exposes no peak-memory counter at all.** The throughput cost is *not* reproducible either: `chemlm_decoder_benchmark.json` has the checkpointed `base` config running *faster* (2056 vs 1640 tok/s), probably an allocator-pressure artifact. Still off by default — on a 36 GiB box there is usually no memory problem to solve — but that is a budget argument, not a "no saving" argument. |
| **The default allocator will swap your Mac instead of OOMing** | `HIGH_WATERMARK_RATIO` defaults to 1.7 → 47.7 GiB "allowed" on a 36 GiB box; 34 GiB was allocated with no error. Fix requires **both** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9` — setting HIGH alone in (0.0, 1.4) throws, and the widely-copied `HIGH=0.0` *removes* the cap. |
| **`torch.compile` needs `dynamic=False` + fixed-length padding** | Variable sequence lengths fail with a Metal shader codegen error ([pytorch#206826](https://github.com/pytorch/pytorch/issues/206826)). With defaults, Dynamo marks a dim dynamic on the *second* distinct shape, so the first batch works and then the job dies — it reads like a random flake. Worth 1.46–1.53×. |
| **Throughput** | 3.5M → 97k tok/s · 26M → ~20k tok/s · 86M → 6.2k tok/s. A ChEMBL-37 epoch (131M tokens) takes 22 min at 5M, 1.8 h at 32M, 5.9 h at 86M. |
| **Op coverage is a non-issue** | All 39 transformer-training ops probed are native with the CPU-fallback flag off, including `multinomial` up to 1,048,576 categories. |

Honest comparison: 1× A100 ≈ 5.5× an M4 Max. 5M–32M is comfortable locally *and* is the
compute-optimal size, so there is no reason to fight for 200M on a Mac.

---

## 7. Measured results for this implementation

**[M]** On M3 Pro MPS, from `scripts/run_chemlm_decoder_benchmark.py`:

| Property | Value |
|---|---|
| KV-cached vs full-sequence forward | **9.5e-07** max abs divergence |
| MPS vs CPU logit parity | **1.4e-06** |
| Padded pocket-atom influence | **exactly 0.0** |
| Dense causal leakage (all configs) | **exactly 0.0** |
| MoE causal leakage | 7.5e-08 — float32 GEMM round-off, not information flow (see below) |
| Pocket conditioning sensitivity | 0.154 |
| Property conditioning sensitivity | 1.14 |
| Parameter estimates vs instantiated | **<0.1% error** across all five presets (tiny +0.04%, small +0.02%, base +0.01%, large +0.01%, xl +0.00% at vocab 66) |

**A subtlety worth recording.** Sparse MoE cannot be bit-for-bit causally exact. A token's
expert assignment depends only on its own hidden state, so no information travels
backwards in time — but perturbing later tokens changes how many tokens each expert
receives, which changes that expert's GEMM shape and shifts results by ~1e-8. Expert
dispatch counts were observed moving `[4,3,5,2,3,4,5,6]` → `[3,2,5,3,3,7,5,4]` under such
a perturbation. The test suite therefore asserts bitwise zero for dense decoders and a
1e-6 bound for MoE, with both properties pinned by dedicated tests.

---

## 7b. Conditioning mechanism, decoding, and the benchmark that is partly an artifact

### The three-way ablation does not exist — and we are one flag from publishing it

**[V] There is no published ablation of cross-attention vs Perceiver soft-prefix vs
FiLM/adaLN within one model and one dataset.** Not for molecules, not anywhere. Two
*two-way* ablations exist and they disagree:

| Study | Backbone | Result |
|---|---|---|
| **Frag2Seq** (arXiv:2408.09730, Tab. 4) | one GPT-1, CrossDocked Luo split | **cross-attn −7.366 Vina beats prefix −6.646** (+0.72 kcal/mol); prefix diversity **0.493 ± 0.39** (mode-collapses on many pockets) |
| **f-RAG** (arXiv:2411.12078, Tab. 8–9) | frozen SAFE-GPT | deleting cross-attn costs **0.038** AUC; deleting the in-sequence channel costs **1.697** (45×); deleting the search loop costs **2.880** (76×) |

**Reconciliation:** they differ on whether the condition is expressible in the decoder's
own token language. f-RAG's signal is *fragments*, natively writable in SAFE, so
cross-attention adds nothing. Frag2Seq's signal is a *protein*, which is not — so a
learned encoder plus cross-attention wins. **A pocket is in the Frag2Seq regime, so
cross-attention is the right default**, which is what this library does.

Supporting, from outside chemistry (read with care — different scale): Flamingo
(arXiv:2204.14198, Tab. 3) settles *where* — cross-attn in **every** layer 70.7 vs once
in the middle 59.8, and **0-init tanh gating is worth +4.2 and fixes instability**;

> **⚠️ Our default does not match the result we are citing.**
> `cross_attn_gate_init` defaults to **0.1**, not 0, so `tanh(0.1) ≈ 0.0997` and
> cross-attention perturbs the decoder's output from the very first step. Flamingo's
> point is that a **0**-initialised gate makes the newly inserted cross-attention an
> exact no-op at initialisation, so a pretrained language model is left untouched while
> the gate learns to open. That matters most in exactly the path this library provides
> for it — `ChemLMDecoder.load_pretrained_backbone`, which transfers a Stage-1
> unconditional chemical prior into a Stage-2 pocket-conditioned decoder. **Pass
> `cross_attn_gate_init=0.0` for that transfer.** The default is left at 0.1 so the
> measured results in this document stay comparable, but 0.0 is what the cited evidence
> supports, and the two should be reconciled with a measurement rather than left to
> drift.

Idefics2 (arXiv:2405.02246, Tab. 3) finds cross-attn wins with a **frozen** backbone
(66.7 vs 60.3) and loses once LoRA'd (67.3 vs 69.5). DiT's adaLN-Zero result is
**diffusion-specific** and conditions on *two tokens* (class + timestep); DiT itself notes
adaLN "is the only conditioning mechanism restricted to apply the same function to all
tokens", which cannot express *this* substituent facing *that* subpocket. LlamaGen read
DiT and explicitly declined adaLN. **[V] Zero papers ablate adaLN/FiLM against
cross-attention inside an autoregressive token decoder.**

**The open experiment:** 3DSMILES-GPT (*Chem. Sci.* 2025) implements a pocket **prefix**
*and* cross-attention on one backbone, calls the encoder "a detachable modular component"
and **reports no detached-vs-attached ablation**. Since this library already ships both
`cross_attn` and `prefix` modes, that ablation is one config flag away. Frag2Seq's
confound to avoid: it did not match context length (850 vs 512).

### 🔴 The 3D leaderboard is substantially an artifact — good news for a SMILES decoder

- **[V] Generating 3D coordinates contributes ~3% of the Vina Dock rank signal.** Replacing
  the entire generated conformer leaves Vina rank correlation at **ρ = 0.971**; scrambling
  every atom type leaves ρ = 0.827 (arXiv:2408.06050, Tab. 3). Redocking washes the
  generated pose out.
- **[V] Retrieval beats every generative model.** Random retrieval of 16,384 ChEMBL
  molecules per pocket, filtered by a **23,000-parameter** surrogate, reaches
  **−8.03 kcal/mol in 2.6 s on CPU** vs TargetDiff −6.91 in 3,428 s. AutoGrow4 (2020,
  zero trainable parameters) reaches −8.33.
- **[V] Vina Dock is size-hackable.** MolCRAFT (arXiv:2404.12141, Tab. 2): size-matched at
  29.4 heavy atoms it scores −9.25 vs DecompDiff-O's −8.39, so **Decomp-O's entire lead
  was molecule size**; raising its own size prior buys 1.33 kcal/mol. The added contacts
  are the non-specific ones — vdW and hydrophobic scale, **H-bonds do not**.
- **[V] Validity collapses under 3D scrutiny.** GenBench3D: TargetDiff graph-validity
  1.000 → 3D-validity **0.106**; DiffSBDD 1.000 → **0.004**. PoseCheck median strain:
  reference 102.5, TargetDiff **1241.7**, DiffSBDD **1243.1**. No method under 2 Å median
  redock RMSD.

**Consequence:** giving up 3D coordinates costs ~3% of the ranking signal and *gains*
near-100% intramolecular validity. **A defensible target for this decoder is Vina Dock
≈ −7.9 at 22–23 heavy atoms, QED 0.52, SA 0.70, diversity 0.71, PoseBusters-valid ~92%**,
with heavy-atom count reported in the main table. Three baselines decide whether the work
is real: random ChEMBL retrieval + the 23K-param filter, AutoGrow4/graph-GA with a docking
reward, and size-matched reporting. Note we cannot report a `Vina Score` column at all
(no generated pose) — only `Vina Dock`.

⚠️ **"Vina score" means three incompatible things** (TargetDiff's 3-column convention vs
QuickVina-redock vs gnina ΔVina), differing by 1.5–2 kcal/mol. Many circulating baseline
numbers trace to a single re-evaluation in TargetDiff (ICLR 2023) Tab. 1, copied into 8+
later papers; four different numbers circulate for AR alone.

### Decoding — two of our current defaults are likely wrong

- ⚠️ **Top-p and top-k are probably harmful for SMILES.** With a ~33–66 token vocabulary,
  nucleus truncation is destructive, and **REINVENT4 exposes neither** (`sample_strategy`
  default `multinomial`). Our `SamplingConfig` defaults to `top_p=0.92, top_k=30` and the
  v6 notebook samples at `top_p=0.95`. Recommended: temperature-only at T ≈ 0.8–1.0, and
  measure before keeping truncation.
- ⚠️ **CFG at high guidance destroys physical validity.** NEAT-POCKET's sweep:
  ω=0 → PoseBusters-valid 71.0%, strain 155, Vina −4.4; **ω=2.5 → PB-valid 47.9%
  (−23 pts), strain 783 (5×)**, Vina −5.6. Turning up conditioning buys 1.2 kcal/mol and
  wrecks the molecule. Our `cfg_scale` default is **1.75**; the consensus optimum is
  **γ ≈ 1.5–2.0** with `p_uncond = 0.1`. **Sweep against the pocket-reliance delta, not
  raw docking score.** Sources disagree on whether CFG costs diversity, so measure it.
- **Best-of-n with a reranker is the strongest, simplest first move.** n=16 captures most
  of the win (>95% constraint satisfaction one-shot → ≈100% at best-of-16), and best-of-k
  buys **diversity at flat reward** (STGG+ on QM9: k=1 → 0.78/0.76, k=100 → 0.77/0.98).
- **Grammar-constrained decoding: do not reach for outlines/XGrammar/llguidance.** Every
  published overhead number targets a ~128k vocabulary; our mask is ~33–66 booleans.
  SMILES syntax is LL(1) but **ring-bond digit matching and valence are outside it**.
  Write the masker explicitly (open-ring `set[int]`, branch stack, per-atom valence) and
  validate it against `partialsmiles` on 100k training prefixes. STGG+ is the existence
  proof that correct masking is free: validity **1.000** *and* the best FCD anyway.
- ⚠️ **Beam-search folklore is wrong for conditional tasks**: beam gives 4–6× more
  scaffolds than sampling on L+M-24, while *diverse* beam raises textual diversity and
  *drops* structural (NCircles 4.562 → 2.395).

### RL — do not start with GRPO or DPO

**[V] As of Nov 2026 there is not one published paper applying GRPO to molecules that
benchmarks it at a fixed oracle budget against a REINVENT-family baseline.** GRPO exists
for critic-free group-relative advantage on multi-token reasoning rewards; a molecular
reward is a single scalar on a complete sequence — a contextual bandit, which REINVENT
already is. GRPO papers report it collapses diversity. DPO's origin paper has **no RL
baseline at all**, and on pocket-conditioned generators AliDiff's own table shows plain
DPO underperforming its E²PO on every affinity metric. This repo ships both `dpo.py` and
`grpo_policy_loss` — fine to keep, but neither is the evidence-backed starting point.

What has the largest measured effect:
1. **A diversity filter beats any algorithm choice.** DRD2, 30,000 compounds: no memory
   5,143 unique / 2,635 scaffolds → scaffold-similarity filter **22,784 / 20,712**
   (**4.4× compounds, 7.9× scaffolds**). ⚠️ Set `minscore` *below* target, or the agent
   learns to score just under the threshold to dodge being memorized.
2. **Never optimize raw docking alone.** ACEGEN produced −23.9 to −15.8 against a −8.44
   redocked reference; DOCKSTRING's logP objective produced logP 42, MW 3.1 kDa, QED
   0.014. Use a **weighted geometric mean** so a bad component zeroes the reward.
3. **ACEGEN's filter inversion** — re-scoring the same runs through chemistry filters
   *inverts* the ranking: REINVENT-MolOpt 15.65 → **7.00**, plain REINVENT 13.60 →
   **14.74**. "The best RL algorithms on MolOpt are not the best on docking."

### Two free wins worth more than any mechanism choice

Across every paper that ablated anything, **representation/data and the inference-time
search loop dominate the conditioning mechanism by roughly an order of magnitude**
(f-RAG: removing the search loop costs 76× more than removing cross-attention;
Chem3DLLM: representation is worth 5.39 kcal/mol while removing RL costs 0.18).

1. **Gaussian cross-entropy for any discretized numeric head** — replace the one-hot
   target with a Gaussian centred on the true value, so near-miss coordinate/angle bins
   are penalised less. ~10 lines, credited with ~30% regression improvement.
2. **Learned "unknown" embeddings for dropped conditions**, so pocket-conditioned and
   molecule-only rows share a batch — lets you train on millions of molecules plus ~100k
   pockets without the pocket path collapsing. `PropertyConditioner` already does this for
   properties; the same trick applies to the pocket.

⚠️ **License traps:** Lingo3DMol and DrugGPT are **GPL-3.0** (copyleft);
Chemlactica/Chemma **weights** are CC-BY-NC-4.0 despite MIT code; mCLM is CC-BY-NC-**ND**,
whose NoDerivatives term legally forbids building on those weights. Frag2Seq, f-RAG and
BindGPT released no usable code. TamGen (MIT), Token-Mol (MIT), SAFE-GPT (Apache-2.0) and
NovoMolGen (MIT) are clean.

---

## 8. Ranked next steps

1. **Build the ChEMBL ⋈ UniProt ⋈ pocket join.** Largest single data win available, and
   data is worth ~25× parameters for SMILES. The repo already carries a curated panel of
   31 CNS targets (including a 24-target Tourette / schizophrenia / Alzheimer registry)
   with verified UniProt, ChEMBL and PDB identifiers, so the target side of the join
   exists — what is missing is the activity pull and the join itself.
2. **Report novelty at the production sample count** using
   `chem_corpora.novelty_report`. Deduplication is already in place but is a no-op on
   ZINC-250k (measured 0.0% duplicates); it becomes load-bearing on PubChem / ZINC-22.
3. **Add RL / preference finetuning** (REINVENT-style Hill Climbing or GRPO) with an
   anti-collapse penalty. NovoMolGen's gains over baselines come from this stage, not from
   model size. Budget a real hyperparameter search.
4. **Make the pocket-cropping convention explicit and configurable**, and re-measure
   against the TargetDiff 10 Å centre-of-mass rule so numbers are comparable to published
   baselines.
5. **Ablate Fourier vs linear property encoding** — we are first here, so it needs
   evidence before it is trusted.
6. **Strengthen the split** with Foldseek/TM-align or ECOD T-group clustering plus ligand
   Tanimoto, then report metrics per similarity bin.

---

## 9. Primary sources

1. NovoMolGen — scaling and representation study. `arXiv:2508.13408`
2. Xu et al. — chemical LM scaling laws, 300 models. `arXiv:2601.22757`
3. Fraglingo — classifier-free guidance for joint property conditioning. `arXiv:2609.13519`
4. Chemlactica / Chemma — property tags in pretraining text. `arXiv:2407.18897`
5. GP-MoLFormer — memorization and novelty decay. `arXiv:2405.04912`
6. Luo et al. — the canonical CrossDocked split. `arXiv:2203.10446`
7. Francoeur et al. — CrossDocked2020. *JCIM* 2020, 60(9):4200–4215, doi:10.1021/acs.jcim.0c00411
8. PLINDER. bioRxiv `10.1101/2024.07.17.603955`
9. PoseCheck — physical plausibility of generated molecules. `arXiv:2308.07413`
10. GenBench3D — conformer validity of generative output. `arXiv:2407.04424`
11. Skinnider — invalid SMILES are beneficial. *Nat. Mach. Intell.* 2024, 6:437–448
12. Tancik et al. — random Fourier features. NeurIPS 2020
13. Switch Transformer / ST-MoE — load balancing and router z-loss
14. MolGPT. *JCIM*, doi:10.1021/acs.jcim.1c00600
