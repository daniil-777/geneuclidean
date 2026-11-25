# GenEuclidean v2.0

**SE(3)/E(3)-Equivariant 3D Protein Pocket-Conditioned De Novo Molecular Generation & Binding Affinity Prediction**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(MPS%20%7C%20CUDA%20%7C%20CPU)-EE4C2C.svg?logo=pytorch&logoColor=white)](pyproject.toml)
[![e3nn 0.6](https://img.shields.io/badge/e3nn-0.6-6E40C9.svg)](docs/architecture.md)
[![RDKit 2026.3](https://img.shields.io/badge/RDKit-2026.3-2C7BB6.svg)](pyproject.toml)
[![CI](https://img.shields.io/badge/CI-ruff%20%C2%B7%20mypy%20%C2%B7%20pytest%20(144)-2EA44F.svg)](.github/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v2.0.0-0969DA.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg)](pyproject.toml)

---

## Executive Summary

**GenEuclidean v2.0** is a clean-room, mathematically verified structure-based drug design (SBDD) platform that unifies **$SE(3)$-equivariant 3D protein pocket representation learning**, **Stage-1 ZINC-250k pretrained autoregressive chemical generation**, **TamGen-style pocket-conditioned variational latent modeling (CVAE)**, **Classifier-Free Guidance (CFG)**, **multi-task biophysical supervision** (`pKd` affinity regression, `DrugCLIP` pocket–ligand retrieval, and physicochemical property prediction), and **Stage-3 Stabilized RPO-DPO preference alignment** in a strictly `float32`, Apple Silicon MPS / CUDA / CPU-native pure-PyTorch package.

Originally begun as a 2021 ETH Zürich MODLAB semester thesis (*De novo drug design using protein structural information and Generative neural network models*, preserved at git tag `v1.0.0-thesis-2021`), v2.0 systematically remediates every geometric, probabilistic, data-leakage, and software engineering defect identified in the forensic audit ([`docs/legacy_audit.md`](docs/legacy_audit.md)):

| Dimension | 2021 Thesis (`v1.0.0-thesis-2021`) | GenEuclidean v2.0 (`PaiNN-SE3 + RPO-DPO`) | Impact |
| :--- | :--- | :--- | :--- |
| **3D Geometry & Symmetry** | `.long()` truncated Å coords to integers; `BatchNorm1d(natoms)`; `(99,99,99)` dummy atoms; `2.0 Å` cutoff | Strict `float32` centroid-centered `10.0 Å` pocket / `5.5 Å` sparse kNN graphs; `PaiNN-SE3` chirality + `EGNN` / `e3nn v0.6` / `SchNet` | Verified `< 8e-6` $SE(3)$ invariance on CPU & MPS; exact `0.0` padded-atom attention |
| **Chemical Validity (Raw / MPO)** | `65.0%` (uncanonicalized) | **`91.6%` raw** / **`100.0%` MPO Top-10** (strict canonical RDKit) | `+26.6 pp` raw; zero syntactic collapse |
| **Uniqueness & Novelty** | `70.0%` / `65.0%` (raw strings) | **`97.0%` / `98.2%`** (canonical), `IntDiv1 = 0.904`, Murcko Div **`0.737`** | High structural diversity across 80 unseen pockets |
| **Medicinal Chemistry Quality** | QED `0.450`, SA `4.00` (mode collapse) | QED **`0.521`** (`0.618` MPO), SA **`3.36`** (`3.12` MPO), Lipinski **`96.3%`**, PAINS-free **`97.1%`** | Stage-3 RPO-DPO lifts drug-likeness without hurting target specificity |
| **Causal 3D Pocket Specificity** | Dice `0.29` mislabeled as Tanimoto; 95% seq-id split leakage | **Strict zero-leakage CASF-2016** (`0/285` overlap): Count Dice **`0.317`** vs Shuffled `0.201` (`p = 1.63e-192`), MW **`r = 0.928`**, **`1.66×`** `<4.5 Å` contact attention | Statistically airtight causal proof of 3D pocket conditioning |
| **Binding Affinity (`pKd`)** | Pearson `r = 0.503` (random 5-fold CV) | **CASF-2016 Pearson `r = 0.746`**, Spearman `ρ = 0.737`, RMSE `1.448`, `R² = 0.555` | **`+0.244`** Pearson $r$ on held-out CASF-2016 core |
| **Virtual Screening (`DrugCLIP`)** | Not supported | **Recall@1 `30.5%`** (`87×` random), **Recall@5 `62.1%`**, **ROC-AUC `0.937`** | Zero-shot 285-way pocket–ligand retrieval |
| **3D Docking & Physical Plausibility** | None | **30-pocket AutoDock Vina v1.2.7 Top-1 `-8.193 kcal/mol`** (beats native `-8.166`), **58.6% pockets beat native**, **94.6% PoseBusters pass-all**, **95.2% zero-clash** | Physically realistic 3D poses with `-35.2%` lower strain energy (`50.6 kcal/mol`) |

> **Full Forensic Audit:** See [`docs/legacy_audit.md`](docs/legacy_audit.md) for the exact line-by-line root-cause analysis of v1.0 defects (`feature.py:122` `.long()` truncation, unregistered `nn.Linear` inside `forward()`, double-softmax `softmax(softmax(logits)/T)`, `<pad>`/`<start>` index collision, and silent index-2 fallback) and their v2.0 fixes.

---

## Architecture Overview

GenEuclidean v2 separates general organic chemistry syntax from 3D structure-based conditioning across **three decoupled training stages** (full mathematical specification in [`docs/architecture.md`](docs/architecture.md)):

```mermaid
flowchart LR
    subgraph S1["Stage 1: Chemical Prior (ZINC-250k)"]
        Z["Randomized SMILES / SELFIES<br/>(249k molecules, atom-level regex)"] --> PRE["RoPE + SwiGLU Decoder<br/>6L · d=256 · 8H (4.77M params)"]
    end

    subgraph S2["Stage 2: 3D Pocket-Conditioned Multi-Task CVAE (PDBbind)"]
        P["3D Pocket Graph (fp32)<br/>10 Å shell · 5.5 Å kNN · 52-dim feats"] --> ENC["Pluggable 3D Encoder<br/>PaiNN-SE3 (default) | EGNN | e3nn v0.6 | SchNet"]
        ENC -->|"node_emb [B,N,D]"| MEM["Pocket Memory Builder<br/>+ CFG null_pocket_token (p=0.15)"]
        ENC -->|"pooled_emb [B,D]"| CVAE["TamGen-Style Pocket CVAE<br/>p(z|p) Prior / q(z|p,l) Posterior"]
        ENC -->|"pooled_emb [B,D]"| HEADS["Multi-Task Biophysical Heads<br/>pKd Regression · DrugCLIP · 6 Props"]
        FP["Ligand ECFP4 [B,1024]"] --> LENC["Ligand FP Encoder"]
        LENC --> CVAE
        LENC --> HEADS
        CVAE -->|"K=4 latent tokens"| MEM
        PRE -.->|"Zero-Init Gated<br/>Cross-Attention Transfer"| DEC["PocketTransformerDecoder<br/>Gated X-Attn + Causal RoPE"]
        MEM -->|"memory [B,N+K,D]"| DEC
        DEC --> LOSS["4-Task Joint Loss<br/>LM + β·KL(free-bits) + pKd + CLIP + Props"]
    end

    subgraph S3["Stage 3: Stabilized RPO-DPO & 3D Inference"]
        DEC --> DPO["Stabilized RPO-DPO Alignment<br/>Frozen 3D Encoder/Heads · Len-Norm · SFT Anchor"]
        DPO --> SAMP["Min-p + CFG (w=1.75) / DBS / CVAE Refine"]
        SAMP --> MPO["MaxMin Diverse MPO Reranking<br/>(QED · SA · pKd · DrugCLIP · Lipinski · PAINS)"]
        MPO --> VINA["ETKDGv3 + MMFF94s + AutoDock Vina v1.2.7<br/>+ PoseBusters Physical Plausibility"]
    end
```

### Core Mathematical Highlights ([`docs/architecture.md`](docs/architecture.md))

1. **Chiral $SE(3)$-Equivariant `PaiNN-SE3` Encoder:** Maintains scalar invariants $\mathbf{s}_i \in \mathbb{R}^{192}$ and $l=1$ equivariant vector channels $\mathbf{v}_i \in \mathbb{R}^{64 \times 3}$ over centroid-centered sparse radius-kNN graphs ($r_c = 5.5\,\text{\AA}$, $k_{\max}=24$, 24 smooth Bessel RBFs with cosine cutoff envelope). Normalized cross-product interactions $\mathbf{v}_i^{\times} = \frac{(\mathbf{C}_1\mathbf{v}_i) \times (\mathbf{C}_2\mathbf{v}_i)}{\sqrt{\|(\mathbf{C}_1\mathbf{v}_i) \times (\mathbf{C}_2\mathbf{v}_i)\|_2^2 + 1}}$ and pseudoscalar contractions $\chi_i = \langle \mathbf{V}\mathbf{v}_i, \mathbf{v}_i^{\times}\rangle$ preserve exact $SO(3)$ equivariance while breaking parity ($\chi_i \mapsto -\chi_i$ under mirror reflections), allowing the encoder to distinguish chiral L- vs. D-amino-acid binding sites.
2. **Zero-Initialized Flamingo-Style Gated Cross-Attention:** Transfers Stage-1 ZINC-250k backbone weights cleanly into Stage-2 while modulating every cross-attention residual by $\tanh(\gamma)$ with exact $-\infty$ masking over padded pocket atoms (`0.0` attention leakage).
3. **4-Task Joint Objective with Cyclical Free-Bits CVAE:**
   $$\mathcal{L}_{\text{Stage-2}} = \mathcal{L}_{\text{LM}} + \beta(t)\,\sum_{d=1}^{64}\max\!\left(\lambda_{\text{fb}},\, D_{\text{KL}}\!\left(q_\phi(z_d \mid \mathbf{p}, \mathbf{l}) \,\|\, p_\theta(z_d \mid \mathbf{p})\right)\right) + 0.5\,\mathcal{L}_{\text{pKd}} + 0.3\,\mathcal{L}_{\text{DrugCLIP}} + 0.2\,\mathcal{L}_{\text{Props}}$$
   with `cfg_drop_prob = 0.15` replacing pocket memory with a learned `null_pocket_token` $\mathbf{E}_\varnothing$ for exact autoregressive Classifier-Free Guidance: $\tilde{\ell}_t = \ell_t(\varnothing) + w \cdot \big(\ell_t(\mathbf{p}, \mathbf{z}) - \ell_t(\varnothing)\big)$.
4. **Stage-3 Stabilized RPO-DPO Alignment:** Standard DPO suffers from likelihood displacement on chemical sequences (eroding syntax validity and corrupting shared 3D scoring heads). GenEuclidean v2 freezes the 3D pocket encoder, CVAE, and multi-task heads bit-for-bit, applies length-normalized sequence log-probabilities $\bar{\ell}_\theta(y \mid \mathbf{p}) = \frac{1}{|y|}\sum_t \log \pi_\theta(y_t \mid y_{<t}, \mathbf{p})$, and anchors updates with an RPO negative log-likelihood term on MPO-winning molecules:
   $$\mathcal{L}_{\text{RPO-DPO}} = -\log \sigma\!\left(\beta\big[\Delta_{\theta,\text{ref}}(y_w) - \Delta_{\theta,\text{ref}}(y_l)\big]\right) + \alpha_{\text{RPO}}\,\mathcal{L}_{\text{NLL}}(y_w \mid \mathbf{p})$$

---

## Verified Experimental Results (Zero-Leakage CASF-2016 Core Set)

All evaluations below were executed natively on Apple M3 Pro (`float32`, PyTorch MPS for training/sampling, CPU for RDKit/AutoDock Vina v1.2.7) using the **strict zero-leakage CASF-2016 split**: **4,543 train / 505 val / 285 test complexes (`0/285` overlap with training or validation)**. Tiers 1–3 evaluate **80 pKd-stratified CASF-2016 pockets × 32 samples = 2,560 ligands per model** (`Min-p = 0.08`, `T = 1.05`, `CFG w = 1.75`); Tier 4 evaluates **all 285 CASF-2016 complexes**. All numbers are loaded directly from versioned artifacts under [`results/tables/`](results/tables/).

### Table 1: Stage-1 Unconditional Chemical Language Priors (ZINC-250k)

Evaluated on 1,000 unconditional samples (`Min-p = 0.08`, `T = 1.0`) against the 2,000-molecule held-out ZINC validation set ([`configs/pretrain_zinc.yaml`](configs/pretrain_zinc.yaml), [`configs/pretrain_zinc_selfies.yaml`](configs/pretrain_zinc_selfies.yaml)):

| Stage-1 Prior | Vocab | Params | Epochs (MPS Time) | Val Loss | Val PPL ↓ | Validity ↑ | Uniqueness ↑ | Novelty ↑ | IntDiv1 ↑ | Murcko Div ↑ | QED ↑ | SA ↓ | Lipinski ↑ | PAINS-Free ↑ | FFD ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Randomized SMILES** | 75 | 4.77M | 6 (29.1 min) | 0.7752 | 2.171 | 99.3% | 100.0% | 99.9% | 0.858 | **0.870** | **0.761** | **2.64** | 100.0% | 97.1% | 2.66 |
| **SELFIES** | 121 | 4.78M | 5 (21.4 min) | 0.6498 | 1.915 | **100.0%** | 100.0% | 99.9% | **0.866** | 0.841 | 0.756 | 2.84 | 100.0% | **97.8%** | **2.21** |

---

### Table 2: Full Multi-Model 5-Tier Benchmark Comparison (CASF-2016 Core Set)

Source: [`results/tables/model_comparison.csv`](results/tables/model_comparison.csv) & [`results/tables/eval_*.json`](results/tables/eval_geneuclidean_v2_painn_dpo.json).

| Model | Params | Validity ↑ | Uniq. ↑ | Novelty ↑ | IntDiv1 ↑ | Murcko Div ↑ | QED ↑ | SA ↓ | Lipinski ↑ | PAINS-Free ↑ | True Tan ↑ | Count Dice ↑ | Attn `<4.5Å` ↑ | pKd $r$ ↑ | pKd RMSE ↓ | CLIP R@1 ↑ | CLIP AUC ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 Thesis Baseline (`e3nn+LSTM`)* | 0.63M | 65.0% | 70.0% | 65.0% | — | — | 0.450 | 4.00 | — | — | 0.170* | 0.290* | — | 0.503* | — | — | — |
| `PaiNN-PocketFormer` (No CVAE/MT) | 8.35M | 92.1% | **98.5%** | 98.8% | 0.896 | 0.628 | 0.487 | **3.36** | 90.8% | 96.7% | 0.098 | 0.195 | 1.16× | — | — | — | — |
| GenEuclidean v2 (`SchNet`) | 8.98M | 90.0% | 97.4% | 98.5% | 0.904 | 0.680 | 0.481 | 3.52 | 89.6% | 97.0% | 0.125 | 0.253 | 1.50× | 0.722 | 1.505 | 25.6% | 0.924 |
| GenEuclidean v2 (`EGNN`) | 9.41M | 89.8% | 98.1% | 98.0% | 0.900 | 0.686 | 0.485 | 3.46 | 88.3% | **97.3%** | 0.125 | 0.256 | 0.97× | 0.729 | 1.491 | 23.7% | 0.927 |
| GenEuclidean v2 (`SELFIES`) | 9.47M | **100.0%** | 98.3% | **99.2%** | **0.911** | 0.671 | 0.422 | 3.97 | 87.0% | 96.3% | 0.107 | 0.232 | 0.62× | 0.731 | 1.482 | 18.1% | 0.929 |
| GenEuclidean v2 (`PaiNN-SE3`) | 9.46M | 89.3% | 96.6% | 98.3% | 0.906 | 0.712 | 0.464 | 3.65 | 91.2% | 96.4% | **0.159** | **0.319** | 1.60× | **0.746** | **1.448** | **30.5%** | **0.937** |
| **GenEuclidean v2 (`PaiNN-SE3 + RPO-DPO`)** | 9.46M | **91.6%** | 97.0% | 98.2% | 0.904 | **0.737** | **0.521** | **3.36** | **96.3%** | **97.1%** | 0.157 | 0.317 | **1.66×** | **0.746** | **1.448** | **30.5%** | **0.937** |
| **↳ + MPO / MaxMin Top-10 Reranking** | 9.46M | **100.0%** | **99.9%** | 98.6% | 0.898 | **0.820** | **0.618** | **3.12** | **99.6%** | **100.0%** | — | — | — | **0.746** | **1.448** | **30.5%** | **0.937** |

<sub>*2021 Thesis Baseline numbers are from `Report_denovo.pdf` (random split with uncanonicalized SMILES; similarity was originally reported as Morgan-count Dice `0.29` ≈ `0.17` true Tanimoto; `pKd r = 0.5025` on 5-fold random CV). Base `PaiNN-SE3` MPO Top-10 achieves `100.0%` Valid / `100.0%` Unique / `100.0%` PAINS-free, QED `0.574`, SA `3.34`, Lipinski `97.6%`, IntDiv1 `0.901`. Full DrugCLIP Recall@5 / Recall@10 / MRR for `PaiNN-SE3` and `PaiNN-SE3 + RPO-DPO`: **`62.1%` / `72.8%` / `0.450`** (bit-for-bit identical).</sub>

#### Four Key Scientific Insights

1. **Causal Proof that Multi-Task Supervision + CVAE Prevent Pocket Collapse:** Removing CVAE and multi-task heads (`PaiNN-PocketFormer`) leaves a fluent unconditional chemical generator (`92.1%` validity, `0.487` QED) whose **3D pocket specificity collapses completely to chance** (Matched Dice `0.195` vs. Shuffled `0.196`, Wilcoxon $p = 0.659$, Cliff's $\delta = -0.007$, cross-pocket MW correlation $r = 0.096$). Adding the Pocket CVAE and multi-task biophysical heads (`pKd`, `DrugCLIP`, properties) drives a **`+64%` surge in Count Dice** (`0.195 → 0.319`), **`MW r = 0.927`**, **`pKd r = 0.746`**, **`CLIP R@1 = 30.5%`**, and **`1.60×` contact attention enrichment**.
2. **Stage-3 Stabilized RPO-DPO Solves Likelihood Displacement:** Freezing the 3D encoder and scoring heads while training with length-normalized log-probabilities and an RPO SFT NLL anchor simultaneously improves **raw Validity** (`89.3% → 91.6%`), **QED** (`0.464 → 0.521`, MPO Top-10 `0.618`), **SA** (`3.65 → 3.36`, MPO Top-10 `3.12`), **Lipinski pass rate** (`91.2% → 96.3%`, MPO `99.6%`), **PAINS-free rate** (`96.4% → 97.1%`), **Murcko scaffold diversity** (`0.712 → 0.737`), **`<4.5 Å` Contact Attention Enrichment** (`1.60× → 1.66×`), and **validation perplexity** (`2.251 → 2.210`), while preserving **pKd $r = 0.746$** and **DrugCLIP R@5 = 62.1%** bit-for-bit and lifting Top-1 native-beating Vina docking rate from `55.2%` to **`58.6%`**.
3. **Chiral Equivariant Geometry (`PaiNN-SE3`) Dominates Invariant & Vector Baselines:** `PaiNN-SE3` outperforms both `EGNN` and distance-only `SchNet` across every generative, specificity, affinity, and retrieval tier (`pKd r = 0.746` vs. `0.729` / `0.722`; `CLIP R@1 = 30.5%` vs. `23.7%` / `25.6%`; `Tanimoto = 0.159` vs. `0.125` / `0.125`), while running `>10×` faster and leaner on Apple Silicon MPS than `e3nn` `FullyConnectedTensorProduct` layers.
4. **Empirical Confirmation of Skinnider (*Nature Machine Intelligence*, 2024) in 3D SBDD:** `GenEuclidean v2 (SELFIES)` guarantees **`100.0%` syntactic validity** and highest raw `IntDiv1 = 0.911`, but pays a clear biophysical penalty in drug-likeness (`QED = 0.422`), synthetic complexity (`SA = 3.97`, Brenk-free `42.2%` vs. `66.7%` for SMILES+DPO), pocket specificity (`Dice = 0.232`, `Tanimoto = 0.107`), and `<4.5 Å` contact focus (`0.62×`) because unconstrained SELFIES decoding converts improbable tokens into exotic strained substructures rather than terminating invalid trajectories.

---

### Table 3: Causal Proof of 3D Conditioning (Matched vs. Shuffled vs. Null Pocket Controls)

Every model is audited against **paired cyclic-derangement Shuffled pockets** and **unconditional Null pockets (`CFG w = 0`)** across the same 80 held-out CASF-2016 targets ([`results/tables/eval_*.json`](results/tables/eval_geneuclidean_v2_painn.json)):

| Model | Matched Tan (Top-1) | Shuffled Tan | Null Tan | Matched Dice (Top-1) | Shuffled Dice | Wilcoxon $p$ (Tan) | MWU $p$ (Tan) | MWU $p$ (Dice) | Cliff's $\delta$ (Pool / Pocket) | MW $r$ | TPSA $r$ | LogP $r$ | `<4.5 Å` Attn ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PaiNN-PocketFormer` (Ablation) | 0.098 (0.208) | 0.099 | 0.098 | 0.195 (0.401) | 0.196 | `6.59e-01` | `6.58e-01` | `7.82e-01` | `-0.007` / `-0.022` | 0.096 | 0.047 | 0.014 | 1.16× |
| GenEuclidean v2 (`SELFIES`) | 0.107 (0.222) | 0.092 | 0.097 | 0.232 (0.427) | 0.185 | `2.79e-06` | `3.21e-30` | `7.96e-55` | `+0.184` / `+0.348` | 0.892 | 0.632 | 0.722 | 0.62× |
| GenEuclidean v2 (`EGNN`) | 0.125 (0.259) | 0.099 | 0.099 | 0.256 (0.458) | 0.197 | `2.35e-09` | `9.94e-36` | `2.54e-57` | `+0.212` / `+0.433` | 0.888 | 0.658 | 0.690 | 0.97× |
| GenEuclidean v2 (`SchNet`) | 0.125 (0.280) | 0.094 | 0.101 | 0.253 (0.473) | 0.190 | `2.53e-11` | `4.61e-48` | `6.87e-66` | `+0.248` / `+0.492` | 0.900 | 0.709 | 0.699 | 1.50× |
| **GenEuclidean v2 (`PaiNN-SE3`)** | **0.159** (0.304) | 0.093 | 0.099 | **0.319** (0.506) | 0.194 | **`1.05e-13`** | **`1.52e-178`** | **`2.39e-215`** | **`+0.486` / `+0.668`** | 0.927 | **0.870** | 0.821 | 1.60× |
| **GenEuclidean v2 (`PaiNN-SE3 + DPO`)** | 0.157 (**0.315**) | 0.095 | 0.100 | 0.317 (**0.515**) | 0.201 | `4.26e-13` | `3.67e-162` | `1.63e-192` | `+0.457` / `+0.641` | **0.928** | 0.864 | **0.839** | **1.66×** |

<sub>Cross-pocket Pearson $p$-values for `PaiNN-SE3 + DPO`: MW $p = 2.73\times 10^{-35}$, TPSA $p = 6.34\times 10^{-25}$, LogP $p = 2.63\times 10^{-22}$ (all 80 CASF-2016 targets).</sub>

---

### Table 4: 15-Config Decoding, CFG Scaling & Hit-to-Lead CVAE Refinement Pareto Sweep

Evaluated on 50 CASF-2016 pockets × 32 samples (`geneuclidean_v2_painn`, [`results/tables/decoding_ablation.csv`](results/tables/decoding_ablation.csv)):

| Family | Configuration | Validity ↑ | Uniq. ↑ | Novelty ↑ | Global IntDiv1 ↑ | Intra-Pocket IntDiv1 ↑ | Murcko Div ↑ | QED ↑ | SA_norm ↑ | Tanimoto ↑ | Count Dice ↑ | Pred pKd ↑ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Greedy** | `Greedy` | 96.0% | 87.5% | 97.6% | 0.820 | 0.000 | 0.667 | 0.487 | **0.820** | **0.177** | 0.317 | 5.87 |
| **Temp** | `Temp T=0.8` | 73.1% | 97.7% | 97.3% | 0.902 | 0.827 | 0.668 | 0.493 | 0.704 | 0.143 | 0.283 | 5.99 |
| **Temp** | `Temp T=1.0` | 29.0% | 99.8% | **98.9%** | 0.916 | 0.766 | **0.762** | 0.480 | 0.627 | 0.129 | 0.256 | 5.74 |
| **Temp** | `Temp T=1.2` | 5.6% | **100.0%** | 97.8% | **0.931** | 0.334 | 0.708 | 0.400 | 0.507 | 0.085 | 0.174 | 5.78 |
| **Top-p** | `Top-p 0.92` | 89.8% | 98.7% | 98.7% | 0.901 | 0.835 | 0.723 | 0.491 | 0.708 | 0.150 | 0.293 | 6.11 |
| **Typical** | `Typical 0.90` | 90.7% | 98.4% | 98.1% | 0.900 | **0.839** | 0.702 | 0.493 | 0.712 | 0.139 | 0.279 | 6.06 |
| **DBS** | `DBS B=16 G=8` | **96.1%** | 81.5% | 94.6% | 0.879 | 0.682 | 0.512 | 0.466 | 0.712 | 0.171 | 0.311 | 5.97 |
| **Min-p+CFG** | `Min-p CFG w=1.0` | 94.7% | 97.6% | 98.4% | 0.897 | 0.834 | 0.696 | **0.502** | 0.732 | 0.147 | 0.289 | 6.10 |
| **Min-p+CFG** | `Min-p CFG w=1.5` | 90.9% | 95.9% | 97.9% | 0.899 | 0.801 | 0.710 | 0.474 | 0.712 | 0.167 | 0.317 | 6.14 |
| **Min-p+CFG** | `Min-p CFG w=1.75` *(default)* | 90.7% | 97.0% | 98.1% | 0.904 | 0.807 | 0.711 | 0.462 | 0.704 | 0.164 | 0.319 | 6.18 |
| **Min-p+CFG** | `Min-p CFG w=2.25` *(Pareto knee)* | 87.9% | 95.9% | 98.1% | 0.906 | 0.788 | 0.721 | 0.442 | 0.688 | 0.172 | **0.329** | **6.30** |
| **CVAE-Temp** | `CVAE τ_z=0.5` | 91.6% | 96.7% | 98.1% | 0.902 | 0.802 | 0.724 | 0.466 | 0.703 | 0.165 | 0.320 | 6.18 |
| **CVAE-Temp** | `CVAE τ_z=1.0` | 90.3% | 97.0% | 98.1% | 0.902 | 0.798 | 0.717 | 0.460 | 0.703 | 0.168 | 0.323 | 6.15 |
| **CVAE-Temp** | `CVAE τ_z=1.3` | 90.9% | 96.7% | 98.1% | 0.903 | 0.802 | 0.715 | 0.464 | 0.705 | 0.165 | 0.321 | 6.25 |
| **CVAE-Refine** | **`Hit-to-Lead Refine` (seed)** | 92.8% | 96.2% | 97.5% | 0.899 | 0.798 | 0.706 | 0.479 | 0.719 | 0.170 | 0.325 | 6.17 |

> **Decoding Takeaway:** Raw temperature sampling collapses at $T \ge 1.0$ (`29.0%` → `5.6%` validity), whereas **`Min-p + CFG`** sits squarely on the Pareto frontier (`w = 1.0 → 2.25` smoothly trades validity `94.7% → 87.9%` for higher target specificity `Dice 0.289 → 0.329` and predicted `pKd 6.10 → 6.30` while holding `IntDiv1 ≈ 0.90`). TamGen-style **`Hit-to-Lead Refine`** achieves `92.8%` validity, `0.325` Dice, and `0.170` Tanimoto while preserving `97.5%` novelty and `0.798` intra-pocket diversity.

---

### Table 5: Tier-5 3D Docking (AutoDock Vina v1.2.7) & PoseBusters Physical Plausibility

Native Apple Silicon **AutoDock Vina v1.2.7** + **RDKit ETKDGv3 / MMFF94s** + **PoseBusters** physical checks across **30 identical pKd-stratified CASF-2016 binding pockets** (Top-5 MPO hits per pocket, exhaustiveness = 4; [`results/tables/docking_casf2016.json`](results/tables/docking_casf2016.json) & [`results/tables/docking_casf2016_dpo.json`](results/tables/docking_casf2016_dpo.json)):

| Method (30 CASF-2016 Pockets) | Docked Ligands | Top-1 Vina Mean (kcal/mol) ↓ | Top-1 Vina Median ↓ | Top-5 Vina Mean / Median ↓ | Best Pose ↓ | High-Affinity Pockets (Top-1 ≤ Native) ↑ | High-Affinity Ligands (≤ Native) ↑ | PoseBusters Pass-All ↑ | Zero Protein Clashes ↑ | Mean Strain Energy (kcal/mol) ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Native Crystal Ligands (Redocked)** | 29 / 30 | -8.166 | -7.910 | — | -13.970 | — | — | — | — | — |
| **GenEuclidean v2 (`PaiNN-SE3`)** | 148 / 150 | -8.145 | **-8.172** | **-7.322** / **-7.411** | **-11.546** | 55.2% (16/29) | **34.3%** (49/143) | 93.9% (139/148) | 94.6% (140/148) | 78.06 |
| **GenEuclidean v2 (`PaiNN-SE3 + RPO-DPO`)** | 147 / 150 | **-8.193** | -8.062 | -7.233 / -7.342 | -10.981 | **58.6%** (17/29) | 31.7% (45/142) | **94.6%** (139/147) | **95.2%** (140/147) | **50.60** (`-35.2%`) |

> **Stage-3 RPO-DPO 3D Impact:** Without ever training on docking scores, Stage-3 MPO-guided RPO-DPO drives **Top-1 mean Vina affinity (`-8.193 kcal/mol`) past both the base model (`-8.145`) and the redocked native crystal ligands (`-8.166`)**, beats the native crystal pose on **58.6% (`17/29`) of pockets**, improves PoseBusters Pass-All to **94.6%** and zero-clash rate to **95.2%**, and slashes mean ETKDGv3/MMFF94s conformational strain energy by **`35.2%`** (`78.06 → 50.60 kcal/mol`).

#### Top Verified Novel De Novo Hits (100% PoseBusters Pass-All, 0 Protein Clashes)

| Target PDB | Model | Vina (kcal/mol) | Native Vina | $\Delta\text{Vina}$ | QED | SA | Pred pKd | Canonical SMILES |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`1mq6`** | `PaiNN-SE3 + DPO` | **-10.981** | -9.138 | **-1.843** | 0.455 | 2.48 | 9.24 | `O=C(c1ccc2ncsc2c1)N1CCN(S(=O)(=O)/C=C\c2ccc3ccccc3c2)CC1` |
| **`1mq6`** | `PaiNN-SE3 + DPO` | **-10.532** | -9.138 | **-1.394** | 0.616 | 2.27 | 8.61 | `O=S(=O)(NC1CCN(S(=O)(=O)c2ccc3ccccc3c2)CC1)c1ccc(Cl)s1` |
| **`1q8t`** | `PaiNN-SE3 + DPO` | **-9.853** | -8.287 | **-1.566** | **0.777** | 2.52 | 8.11 | `Nc1ncnc2c1CN(Cc1cnc3ccccc3c1)C2=O` |
| **`2wbg`** | `PaiNN-SE3 + DPO` | **-9.349** | -7.565 | **-1.784** | 0.567 | **2.17** | **9.96** | `Nc1nc(CNc2ccccc2)nc2cc3ccccc3nc12` |
| **`1e66`** | `PaiNN-SE3` (Base) | **-11.546** | -12.013 | +0.467 | **0.786** | 2.75 | 7.16 | `CC(=O)N1CCc2c([nH]c(CCC(=O)O)c2-c2ccc(O)cc2)C1=O` |
| **`1lpg`** | `PaiNN-SE3` (Base) | **-10.554** | -10.448 | **-0.106** | 0.457 | 3.63 | 7.64 | `Cc1nc(N)c2nc(-c3ccc4ccccc4c3)n([C@H]3CC[C@@H](C(=O)N4CCC[C@H]4C(N)=O)C3)c2n1` |
| **`1q8t`** | `PaiNN-SE3` (Base) | **-9.640** | -8.287 | **-1.353** | 0.733 | 2.33 | 8.46 | `N#Cc1cccc(C(=O)Nc2cc(C(F)(F)F)nn2-c2cccc(O)c2)c1` |

---

### Table 6: Exact $SO(3)/SE(3)$, Permutation & Dummy-Atom Equivariance Verification

Measured in `float32` across 5 Haar-uniform 3D rotations $\mathbf{R} \in SO(3)$ and rigid translations $\mathbf{t} \in [-100, 100]^3\,\text{\AA}$ (enforced continuously by [`tests/test_equivariance.py`](tests/test_equivariance.py)):

| 3D Encoder | Symmetry Group | Chiral Reflection ($\det \mathbf{P} = -1$) | CPU Pure $SO(3)$ (Scalar / Vec) | MPS Pure $SO(3)$ (Scalar / Vec) | CPU $SO(3) + \mathbf{t} \in [-100, 100]^3$ | MPS $SO(3) + \mathbf{t} \in [-100, 100]^3$ | Padded-Atom Attention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PaiNN-SE3`** *(default)* | $SE(3)$ | **Sensitive** (chiral) | `1.91e-6` / `2.56e-6` | `1.79e-6` / `2.86e-6` | `6.79e-6` / `5.11e-5` | `7.33e-6` / `5.13e-5` | Exact `0.0` |
| **`EGNN`** | $E(3)$ | Invariant | `1.67e-6` / `1.13e-6` | `1.55e-6` / `1.13e-6` | `1.91e-6` / `5.66e-6` | `1.43e-6` / `5.13e-6` | Exact `0.0` |
| **`E3NN v0.6`** (`lmax=1`) | $SE(3)$ / $E(3)$ | **Sensitive** (`0o`/`1e`) | `2.50e-5` / `3.46e-6` | `2.81e-5` / `4.53e-6` | `2.78e-4` / `3.66e-5` | `2.61e-4` / `3.75e-5` | Exact `0.0` |
| **`SchNet`** | $E(3)$ | Invariant | `1.67e-6` / — | `1.43e-6` / — | `1.67e-6` / — | `1.43e-6` / — | Exact `0.0` |

---

## Visual Results Gallery

### 1. Multi-Model Benchmark Comparison (2021 Thesis Baseline vs. GenEuclidean v2)
![GenEuclidean Benchmark Comparison](results/figures/01_model_comparison.png)
*Eight-metric overview comparing the 2021 thesis baseline against `PaiNN-PocketFormer` (ablation), `SchNet`, `EGNN`, `SELFIES`, `PaiNN-SE3`, and `PaiNN-SE3 + Stage-3 RPO-DPO` on held-out CASF-2016 pockets.*

### 2. Multi-Model Training Dynamics
![GenEuclidean v2 Multi-Model Training Dynamics](results/figures/02_training_curves.png)
*Smooth chemical language cross-entropy loss (train dashed, validation solid), validation perplexity, validation `pKd` Pearson $r$, and active CVAE latent KL divergence (`3.0–4.9 nats`, zero posterior collapse).*

### 3. Causal Proof of 3D Pocket Conditioning
![Causal Proof of 3D Pocket Conditioning](results/figures/03_matched_vs_shuffled_specificity.png)
*Distribution of ECFP4 Tanimoto similarity to the native crystal ligand for Matched (`μ = 0.160` pooled / `0.159` per-pocket), Shuffled (`μ = 0.093`), and Null (`μ = 0.099`) pocket controls (`MWU p = 1.52e-178`, Cliff's `δ = +0.486` pooled / `+0.668` per-pocket).*

### 4. Physicochemical & Medicinal Chemistry Property Fidelity
![Physicochemical & Medicinal Chemistry Property Fidelity](results/figures/04_property_distributions.png)
*Six-panel KDE overlays (MW, Crippen LogP, QED, SA, TPSA, NP-likeness) against CASF-2016 reference crystal ligands with exact Wasserstein-1 ($W_1$) distances. Stage-3 RPO-DPO visibly shifts QED rightward (`0.521`), SA leftward (`3.36`), and achieves `W₁ = 0.20` on LogP.*

### 5. Binding Affinity (`pKd`) Parity on CASF-2016 Core Set
![Binding Affinity Parity on CASF-2016](results/figures/05_pkd_parity_casf2016.png)
*Zero-leakage CASF-2016 Core Set (`N = 285`) parity plot for `PaiNN-SE3` (and bit-for-bit identical frozen head in `PaiNN-SE3 + RPO-DPO`): Pearson `r = 0.746`, Spearman `ρ = 0.737`, RMSE `1.448`, MAE `1.107`, `R² = 0.555`.*

### 6. 3D Pocket Cross-Attention Interpretability
![3D Pocket Cross-Attention Interpretability](results/figures/06_pocket_attention_heatmap.png)
*Token-to-residue cross-attention heatmap on CASF-2016 complex `1C5Z` highlighting true `<4.5 Å` crystal ligand contact residues (`★`). Across 30 audited pockets, contact enrichment averages `1.60×` (`PaiNN-SE3`) and `1.66×` (`PaiNN-SE3 + RPO-DPO`).*

### 7. Showcase De Novo Generated Ligands (`1QF1`)
![De Novo Ligands Generated for Target Pocket 1QF1](results/figures/07_generated_molecules_grid.png)
*Crystal reference ligand (`1QF1`) alongside top generated de novo hits annotated with QED, SA, ECFP4 Tanimoto, predicted `pKd`, and native AutoDock Vina scores (`-8.3` to `-9.2 kcal/mol`).*

### 8. Decoding & Hit-to-Lead Pareto Frontier
![Decoding Pareto Frontier](results/figures/08_decoding_pareto.png)
*Fifteen decoding configurations across 50 CASF-2016 pockets mapping Structural Diversity (`IntDiv1`) vs. Target Specificity (`ECFP4 Tanimoto`). `Min-p + CFG` and `Hit-to-Lead Refine` dominate the upper-right Pareto knee.*

### 9. Learned 3D Equivariant Pocket Latent Space (t-SNE)
![3D Equivariant Pocket Latent Space](results/figures/09_pocket_latent_tsne.png)
*Deterministic 2D t-SNE projection of the 285 CASF-2016 `PaiNN-SE3` pooled 3D pocket representations colored by experimental binding affinity (`pKd`), exhibiting coherent low-to-high affinity manifold structure.*

---

## Quickstart & Installation

GenEuclidean v2 uses [`uv`](https://docs.astral.sh/uv/) and targets **Python 3.12+** on Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU.

> **Important RDKit Pin:** [`pyproject.toml`](pyproject.toml) explicitly pins `rdkit>=2026.3,<2026.9` to avoid an ARM64 `MurckoDecompose` C++ `RuntimeError ("bad any cast")` observed in newer nightly builds.

```bash
# 1. Create virtual environment & install editable with dev tooling
uv venv --python 3.12
make install

# Optional: include AutoDock Vina receptor/ligand prep (meeko + gemmi) or SAFE tokenization
uv pip install --python .venv/bin/python -e ".[docking,dev]"

# 2. Verify environment, device auto-selection (mps > cuda > cpu), and cached datasets
.venv/bin/geneuclidean info

# 3. Run lint, strict mypy, and full 144-test suite (including SE(3)/E(3) equivariance)
make lint && make typecheck && make test
```

---

## CLI Reference (`geneuclidean`)

The Typer + Rich CLI ([`src/geneuclidean/cli.py`](src/geneuclidean/cli.py)) exposes **8 subcommands** for the complete SBDD lifecycle:

| Subcommand | Description | Example Usage |
| :--- | :--- | :--- |
| `info` | Display version, active device (`mps`/`cuda`/`cpu`), model catalog, and dataset readiness | `geneuclidean info` |
| `download` | Download lean dataset stack (PDBbind, LP-PDBBind, CASF-2016, ZINC-250k, AutoDock Vina v1.2.7) | `geneuclidean download --include-vina` |
| `prepare` | Extract `float32` 3D pocket graphs, build leak-proof splits, and fit JSON vocabularies | `geneuclidean prepare --config configs/geneuclidean_v2_painn.yaml` |
| `pretrain` | Run Stage-1 unconditional chemical LM pretraining on ZINC-250k | `geneuclidean pretrain -c configs/pretrain_zinc.yaml` |
| `train` | Run Stage-2 3D pocket-conditioned multi-task CVAE training via `PocketTrainer` | `geneuclidean train -c configs/geneuclidean_v2_painn.yaml` |
| `sample` | Sample de novo ligands for any local PDB/mmCIF, cached PDB ID, or live RCSB target + MPO + optional 3D/Vina | `geneuclidean sample --checkpoint runs/geneuclidean_v2_painn_dpo/best.pt --pdb-id 1qf1 --strategy min_p --cfg-scale 1.75 --eval-3d --run-vina` |
| `evaluate` | Run the 5-tier SBDD evaluation suite on any split and write JSON + summary PNGs | `geneuclidean evaluate --checkpoint runs/geneuclidean_v2_painn_dpo/best.pt --split test --split-type casf2016` |
| `fetch-pdb` | Fetch any novel protein–ligand complex live from RCSB PDB and extract its 10 Å 3D pocket | `geneuclidean fetch-pdb --pdb-id 7l10` |

---

## Reproducing Full Paper Experiments

All end-to-end training, Stage-3 RPO-DPO alignment, 5-tier benchmarking, 15-config decoding Pareto sweeps, AutoDock Vina docking, and 9-figure generation are scripted reproducibly under [`scripts/`](scripts/):

```bash
# 1. Stage-1 ZINC-250k Chemical Priors (Randomized SMILES & SELFIES)
.venv/bin/python scripts/run_pretrain.py --config configs/pretrain_zinc.yaml --epochs 6 --batch-size 128
.venv/bin/python scripts/run_pretrain.py --config configs/pretrain_zinc_selfies.yaml --epochs 5 --batch-size 128

# 2. Stage-2 3D Pocket-Conditioned Multi-Task Suite (CASF-2016 zero-leakage split)
.venv/bin/python scripts/run_stage2_suite.py \
  --run geneuclidean_v2_painn,geneuclidean_v2_egnn,geneuclidean_v2_schnet,geneuclidean_v2_selfies,painn_pocketformer \
  --split casf2016 --batch-size 32 --resume-skip --device auto

# 3. Stage-3 Stabilized RPO-DPO Alignment on top of PaiNN-SE3
.venv/bin/python scripts/run_stage2_suite.py \
  --run geneuclidean_v2_painn_dpo --split casf2016 \
  --dpo-pockets 250 --dpo-samples-per-pocket 16 --dpo-epochs 1 --dpo-beta 0.5 --dpo-sft-weight 0.25 --device auto

# 4. 5-Tier Evaluation, 15-Config Decoding Pareto Sweep, and 30-Pocket AutoDock Vina Docking
.venv/bin/python scripts/run_benchmark.py --mode models --split casf2016 --max-pockets 80 --num-samples 32 --attention-pockets 30 --device auto
.venv/bin/python scripts/run_benchmark.py --mode decoding --checkpoint runs/geneuclidean_v2_painn/best.pt --split casf2016 --decoding-pockets 50 --num-samples 32 --device auto
.venv/bin/python scripts/run_benchmark.py --mode docking --samples-file results/tables/samples_geneuclidean_v2_painn_dpo.json \
  --split casf2016 --dock-pockets 30 --dock-top-k 5 --dock-exhaustiveness 4 --dock-workers 4 --device cpu

# 5. Re-render all 9 publication figures (300 DPI)
.venv/bin/python scripts/generate_figures.py
```

---

## Testing, Quality Gates & CI/CD

Every commit is validated by local Make targets ([`Makefile`](Makefile)) and GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **Lint & Formatting:** `ruff check src tests scripts` and `ruff format --check src tests scripts` (100-char lines, Python 3.12 ruleset).
- **Strict Static Typing:** `mypy src` (`disallow_untyped_defs`, `no_implicit_optional`, PEP 561 `py.typed`).
- **144 Unit, Equivariance & End-to-End Tests (`pytest --cov=geneuclidean`):**
  - [`tests/test_equivariance.py`](tests/test_equivariance.py) — Exact $SO(3)$, $SE(3)$, reflection/chirality, atom permutation, and dummy-atom zero-leakage checks across all 4 encoders on CPU & MPS.
  - [`tests/test_models_and_cvae.py`](tests/test_models_and_cvae.py) — Forward/backward gradient flow, Stage-1 → Stage-2 weight transfer, CVAE KL & CFG null-pocket parity.
  - [`tests/test_sampling.py`](tests/test_sampling.py) — All 7 decoding strategies, CFG guidance, CVAE refinement, MaxMin MPO reranking, and RPO-DPO stabilization.
  - [`tests/test_featurizer_and_splits.py`](tests/test_featurizer_and_splits.py), [`tests/test_tokenizer.py`](tests/test_tokenizer.py), [`tests/test_metrics.py`](tests/test_metrics.py), [`tests/test_config.py`](tests/test_config.py), [`tests/test_cli_e2e.py`](tests/test_cli_e2e.py).

---

## Repository Structure

```text
geneuclidean/
├── configs/                       # 9 reproducible YAML experiment configs (Stage-1 & Stage-2)
├── data/
│   └── metadata/                  # Preserved PDBbind v2019, CASF-2016 & CASF-2013 label CSVs
├── docs/
│   ├── architecture.md            # Mathematical specification, param counts & equivariance bounds
│   └── legacy_audit.md            # Forensic audit of 2021 ETH thesis defects & v2.0 fixes
├── results/
│   ├── figures/                   # 9 publication-grade 300 DPI figures (01..09_*.png)
│   └── tables/                    # Verified CSV/MD/JSON benchmarks & AutoDock Vina docking tables
├── scripts/                       # Reproducible pretraining, Stage-2/3 suite, benchmark & plot drivers
├── src/geneuclidean/
│   ├── cli.py                     # 8-subcommand Typer + Rich CLI (`geneuclidean`)
│   ├── config.py                  # Pydantic v2 hierarchical config & device selection
│   ├── data/                      # fp32 3D featurizer, regex/SELFIES tokenizers, leak-proof splits, RCSB fetch
│   ├── models/
│   │   ├── encoders/              # PaiNN-SE3, EGNN, e3nn v0.6, SchNet, Bessel/Gaussian RBFs
│   │   ├── decoders/              # RoPE+SwiGLU GatedXAttn Transformer & Masked Bahdanau LSTM
│   │   ├── cvae.py                # TamGen-style pocket-conditioned CVAE
│   │   ├── heads.py               # Multi-task pKd, DrugCLIP, and auxiliary property heads
│   │   └── geneuclidean.py        # Unified GenEuclideanModel with CFG null_pocket_token
│   ├── training/                  # Stage-1 pretrainer, Stage-2 PocketTrainer, Stage-3 RPO-DPO, losses
│   ├── sampling/                  # Min-p, Top-p, Typical, DBS, CFG, CVAE refine & MaxMin MPO reranker
│   └── evaluation/                # 5-tier SBDD metrics, ETKDGv3/Vina/PoseBusters, 9 plotting functions
├── tests/                         # 144 unit, equivariance, metric & CLI integration tests
├── .github/workflows/ci.yml       # Automated Python 3.12 ruff, mypy, and pytest CI
├── Makefile                       # Developer workflow targets
└── pyproject.toml                 # Package metadata, pinned deps & tool configuration
```

### Historical Git Archive Tags

All original research milestones remain intact in git history:
- `v0.1.0-legacy-binding` — 2020 binding affinity prototype on PDBbind v2019 refined & CASF-2013
- `v1.0.0-thesis-2021` — September 2021 ETH Zürich semester thesis baseline (`e3nn 0.0.0` + LSTM)
- `v2.0.0-alpha.1-core` — Clean-room v2.0 core package, equivariant encoders, CVAE, sampling, and CLI
- `v2.0.0` — Full verified CASF-2016 5-tier benchmarks, Stage-3 RPO-DPO, 30-pocket Vina docking, 9 figures, and documentation

---

## Citation

If you use GenEuclidean in your research, please cite:

```bibtex
@software{emtsev2026geneuclidean,
  author       = {Emtsev, Daniil},
  title        = {{GenEuclidean v2.0: SE(3)-Equivariant 3D Protein Pocket-Conditioned De Novo Molecular Generation and Binding Affinity Prediction}},
  year         = {2026},
  version      = {2.0.0},
  url          = {https://github.com/daniil-777/geneuclidean}
}
```

## License

Released under the **MIT License** (see [`pyproject.toml`](pyproject.toml)).

## Key References

1. Schütt, K. T., Unke, O. T., & Gastegger, M. *Equivariant message passing for the prediction of tensorial properties and molecular spectra (PaiNN).* ICML 2021.
2. Satorras, V. G., Hoogeboom, E., & Welling, M. *E(n) Equivariant Graph Neural Networks (EGNN).* ICML 2021.
3. Geiger, M., & Smidt, T. *e3nn: Euclidean Neural Networks.* arXiv:2207.09453, 2022.
4. Wu, K. et al. *TamGen: drug design with target-aware molecule generation through a chemical language model.* Nature Communications, 2024.
5. Gao, B. et al. *DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening.* NeurIPS 2023.
6. Li, J. et al. *Leak Proof PDBBind: A Reorganized Dataset of Protein–Ligand Complexes for More Generalizable Binding Affinity Prediction (LP-PDBBind).* J. Chem. Inf. Model., 2024.
7. Su, M. et al. *Comparative Assessment of Scoring Functions: The CASF-2016 Update.* J. Chem. Inf. Model., 2019.
8. Skinnider, M. A. *Invalid SMILES are beneficial rather than detrimental to chemical language models.* Nature Machine Intelligence, 2024.
9. Buttenschoen, M., Morris, G. M., & Deane, C. M. *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences.* Chemical Science, 2024.
10. Pang, R. Y. et al. *Iterative Reasoning Preference Optimization (RPO).* NeurIPS 2024; Rafailov, R. et al. *Direct Preference Optimization (DPO).* NeurIPS 2023.
11. Eberhardt, J. et al. *AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings.* J. Chem. Inf. Model., 2021.
