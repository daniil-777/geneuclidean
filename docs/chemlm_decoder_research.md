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

**[V] NovoMolGen** ([arXiv:2508.13408](https://arxiv.org/abs/2508.13408), ~30k experiments,
compute 3e15 → 1e19 FLOPs) states verbatim that *"goal-directed generation largely
saturates beyond approximately 5M parameters, despite improvements in bits per
molecule."* The supporting numbers:

| Metric | 5M | 32M | 300M | 1B |
|---|---|---|---|---|
| Loss (bits/molecule) | — | 48.1 | — | 45.0 |
| PMO sum (22 tasks, 10k oracle calls) | — | 16.48 | 16.70 | — |
| Docking top-5% (5 targets, kcal/mol) | −65.8 | −66.3 | — | −66.2 |

31× more parameters past 32M buys 3.1 bits of loss and **0.2 kcal/mol of docking score**,
which is inside the noise. For reference, REINVENT scores 14.01 on PMO and f-RAG 16.30.

**[V] Xu et al.** ([arXiv:2601.22757](https://arxiv.org/abs/2601.22757), 300 models,
1M→650M params, 5 representations) fit `L(P,D) = L∞ + k_P·P^(−α) + k_D·D^(−β)` and find
for SMILES **α = 0.0171** against **β = 0.4299**. The parameter exponent is ~25× smaller
than the data exponent: *data buys ~25× more than parameters.* The two papers look
contradictory and are not — one measures loss, the other measures design quality, and
Xu et al. concede that de novo metrics "saturate early" and "cannot serve as primary
evidence for or against scaling".

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

**[V] Loss and sample quality can move in opposite directions.** Scaling 5M→32M on PubChem
improved loss 54.1→48.1 bits/molecule while **FCD got worse, 2.8→5.1**. Never
model-select on validation loss alone. The notebook reports sample metrics separately and
never selects on loss.

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
| **Do not enable gradient checkpointing** | It works perfectly (4.0× activation saving, gradients bit-exact) but costs ~17% throughput to solve a problem you do not have. Defaults to off; the docstring explains when it earns its cost. |
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
| Parameter estimates vs instantiated | 0.0% error across all presets |

**A subtlety worth recording.** Sparse MoE cannot be bit-for-bit causally exact. A token's
expert assignment depends only on its own hidden state, so no information travels
backwards in time — but perturbing later tokens changes how many tokens each expert
receives, which changes that expert's GEMM shape and shifts results by ~1e-8. Expert
dispatch counts were observed moving `[4,3,5,2,3,4,5,6]` → `[3,2,5,3,3,7,5,4]` under such
a perturbation. The test suite therefore asserts bitwise zero for dense decoders and a
1e-6 bound for MoE, with both properties pinned by dedicated tests.

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
