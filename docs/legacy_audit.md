# Legacy Codebase Forensic Audit & GenEuclidean v2.0 Remediation

This document summarizes the forensic audit of the historical 2020–2021 GenEuclidean repository (ETH Zürich MODLAB Semester Thesis, *De novo drug design using protein structural information and Generative neural network models*, September 2021) and maps every discovered scientific, mathematical, and engineering defect to its exact fix in GenEuclidean v2.0.

---

## 1. Historical Git Archive & Preserved Data Provenance

All historical baselines remain intact as local annotated tags and remote archive branches:

| Reference | Scope | Key Content |
|---|---|---|
| `v0.1.0-legacy-binding` (`origin/master`) | 52 commits (2020-02 to 2020-09) | Early binding affinity (`pKd`) prototype on PDBbind v2019 refined & CASF-2013 |
| `origin/binding` / `origin/captioning` | 852 commits (2020-06 to 2020-10) | R&D branches adding `se3cnn`, pocket-to-SMILES captioning, and CASF-2016 label tables |
| `v1.0.0-thesis-2021` (`origin/main`) | 204 commits (2020-10 to 2021-01) | Final thesis codebase (`Report_denovo.pdf` baseline, directly in ancestry of `main`) |

Three verified label CSVs were harvested from the historical branches into clean headered files (`pdb_id,pkd`) under `data/metadata/`:

- `data/metadata/pdbbind_v2019_refined_labels.csv` — 4,852 PDBbind v2019 refined complexes (`pKd` 2.00–11.92)
- `data/metadata/casf_2016_core_labels.csv` — 285 CASF-2016 core complexes (`pKd` 2.07–11.82; 266 overlap refined)
- `data/metadata/casf_2013_core_labels.csv` — 195 CASF-2013 core complexes (`pKd` 1.40–13.96; 155 overlap refined)

Obsolete 2020/2021 source trees (`src/`), 7.8 MB of vendored RDKit `Contrib/`, 40 committed `.pyc`/`.DS_Store` binaries, and 178 redundant ETH cluster YAML/BSUB scripts were retired from `main` (preserved at `v1.0.0-thesis-2021`).

---

## 2. 2021 Thesis Baseline Benchmarks (`Report_denovo.pdf`)

Across 54 model/featurization/split combinations (PDBbind v2019 refined, 100 epochs, 900 test pockets):

| Benchmark Metric | 2021 Thesis Best Reported | Audit Caveat |
|---|---|---|
| Chemical Validity | 65% Random / 60% Morgan / 65% Chain | Uncanonicalized; invalid denominator bug (`len + 1`) |
| Uniqueness | 70% Random / 64% Morgan / 70% Chain | Computed on raw non-canonical strings (inflated) |
| Novelty | 65% Random / 60% Morgan / 65% Chain | Computed on raw non-canonical strings (inflated) |
| Ligand Similarity | 0.29 Random / 0.21 Morgan / 0.27 Chain (vs 0.18 rand) | Computed **Morgan-count Dice** ($\frac{2T}{1+T}$), ≈ **0.17 true Tanimoto** |
| Drug-Likeness (Fig. 4.5) | QED ~0.45, SA ~4.0, MW ~380 Da | Severe mode collapse on Morgan split; fold zero-padding distortion |
| Binding Affinity (`pKd`, App. A) | Test Pearson $r = 0.5025$ (Train $r = 0.7247$) | 5-fold random CV only; no CASF-2016 RMSE/MAE reported |

---

## 3. Root-Cause Audit & Exact v2.0 Fixes

### A. 3D Geometry, Equivariance & Permutation Invariance

| Legacy Defect (`v1.0.0-thesis-2021`) | Mathematical Impact | GenEuclidean v2.0 Fix |
|---|---|---|
| `geo_filt = torch.load(...).long()` (`feature.py:122`) truncated 3D Å coordinates and atomic masses to 64-bit integers | Destroyed sub-Ångström bond geometry, created `r_ij = 0` collisions, and broke $SO(3)$ equivariance | Strict `float32` 3D coordinates throughout preprocessing, storage, and forward passes |
| Fresh unregistered `nn.Linear` created inside `forward()` (`bio_e3nn.py:164,181`) | Atom features projected by untrained random noise every batch/inference step | Properly registered `nn.Module` projection layers verified by parameter & gradient tests |
| `nn.BatchNorm1d(natoms)` applied to `[B, natoms, C]` (`bio_e3nn.py:141`) | Normalized along atom-slot index instead of feature channels, breaking permutation invariance | Channel-wise `LayerNorm`/`RMSNorm` and equivariant vector norms; unit-tested permutation invariance |
| Uncentered raw crystal coordinates + dummy atoms stacked at `(99, 99, 99)` | Collisions with pockets near `(99, 99, 99)` and spurious `r_ij = 0` dummy-dummy interactions | Center pockets at centroid; sparse radius/k-NN graphs with explicit boolean masks (zero dummy coordinates) |
| `max_rad = 2.0 Å` radial cutoff & dead `rad_model` config (`CosineBasisModel` hardcoded) | Cutoff shorter than H-bonds (2.6–3.5 Å) & hydrophobic contacts; cross-residue context severed | `10.0 Å` pocket shell from **any** ligand atom; `5.5 Å` graph cutoff with `k=24` max neighbors & 24 smooth-envelope RBFs |
| Feature slice misalignment in `Bio_All_Network` (`[:7]` vs `[7:]` on `[80, 7]`) & raw `(x,y,z)` MLP in PointNet | Corrupted pharmacophore/charge modalities and broke $E(3)$ invariance | Unified, strictly invariant 52-dim atom/residue descriptor; PaiNN-SE3 (chirality-aware via cross-product), EGNN, modern `e3nn >= 0.5`, and SchNet with `< 1e-5` $SE(3)/E(3)$ unit tests |

### B. Featurization, Vocabulary & Tokenization

| Legacy Defect | Impact | GenEuclidean v2.0 Fix |
|---|---|---|
| Character-level `vocab.pkl` (split `Cl`→`C,l`, `Br`→`B,r`), missing `<unk>`, commented-out `<pad>` colliding with `<start>=0` | Forced fragile spelling of multi-char atoms; `<pad>`/`<start>` collision in `CrossEntropyLoss` | Deterministic atom-level regex SMILES tokenizer (Schwaller et al.) + SELFIES + optional SAFE, with dedicated `<pad>`, `<bos>`, `<eos>`, `<unk>` tokens |
| Off-by-one `IndexError` on `K`, `Se`, `Hg`; `H` mass set to `2` (deuterium); silent fallback to protein #2 on `FileNotFoundError` | Dropped complexes or silently paired wrong 3D pockets with ligand SMILES/`pKd` | Bounds-checked RDKit/BioPython featurization with explicit skip manifest; zero silent index substitution |
| Single 8 Å sphere around ligand centroid; inverted H-filter flags | Truncated elongated ligand binding sites and retained waters/ions | All protein heavy atoms within `10.0 Å` of any ligand heavy atom; clean water/ion exclusion |

### C. Decoder, Cross-Attention & Sampling Math

| Legacy Defect | Impact | GenEuclidean v2.0 Fix |
|---|---|---|
| Double-softmax in `sample_temp`: `softmax(softmax(logits) / T)` (`decoder.py:106-116`) | Turned `T = 0.2–0.7` into ~88% uniform noise, causing temperature sampling collapse | Exact single-softmax logit scaling `logits / T` + Min-p, Top-p, Top-k, Typical, Diverse Beam Search, and Autoregressive CFG |
| Additive raw probabilities in beam search (no `log_softmax`, no length normalization) | Monotonically favored max-length runaway strings | Log-probability accumulation with configurable length normalization and group diversity penalty |
| Hardcoded `sentence[7:-5]` slicing; unmasked dummy attention; cross-batch `[B, B, A]` broadcast leakage; `sample_prob` unable to draw token `0` | Chopped 7 real SMILES chars, leaked attention across batch items & dummy atoms | Token-ID decoding stopping cleanly at `<eos>`, explicit key-padding masks (`-inf`), batched pocket encoding once per target |
| Single-step `t=0` LSTM trained from scratch on ~3.8k pockets | Low chemical validity (~60%) and generic aromatic ring collapse | Stage-1 ZINC-250k pretraining (RoPE + SwiGLU Transformer) + zero-initialized gated cross-attention + TamGen-style CVAE (cyclical KL + free-bits) + multi-task (`pKd`, DrugCLIP, properties) |

### D. Splits, Evaluation & Modern Engineering

| Legacy Defect | Impact | GenEuclidean v2.0 Fix |
|---|---|---|
| "PFAM" split actually used `bc-95.txt` (95% sequence identity); KMeans Euclidean scaffold split; swapped `refined`/`core` labels on `master` | Homolog and scaffold leakage across train/test; misaligned `pKd` targets | Leakage-proof **LP-PDBBind** split (`CL1/CL2/CL3`), strict **CASF-2016** holdout (excluding 266 overlaps), Bemis-Murcko scaffold split, and clean random baseline |
| Dice reported as Tanimoto; uncanonicalized uniqueness/novelty; element-wise fold averaging with zero-padding | Inflated similarity/uniqueness/novelty and distorted property histograms | Canonical RDKit SMILES metrics, primary ECFP4 Tanimoto (+ Dice for thesis comparison), `IntDiv1/2`, Murcko scaffold diversity, QED/SA (via `RDConfig.RDContribDir`), Lipinski, PAINS, Vina docking, and matched-vs-shuffled pocket specificity |
| PyTorch 1.4/1.5, `e3nn==0.0.0`, forced `.double()` FP64, unconditional `py3nvml`, `set_detect_anomaly(True)`, hardcoded cluster paths | Broken on modern Python and Apple Silicon MPS | Python 3.12, `uv`, PyTorch 2.x (`mps > cuda > cpu`, strict `float32`), `e3nn >= 0.5`, `rdkit >= 2026.3, < 2026.9`, Pydantic v2 YAML config, Typer CLI, Ruff/Mypy/Pytest CI |
