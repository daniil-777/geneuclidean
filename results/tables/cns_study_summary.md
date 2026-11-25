# GenEuclidean v3.1 CNS Study Summary (Gates 0, 2 & 3: Iterations 0 -> 1 -> 2 + Ablations A1-A5)

## 1. Gate 0 Anti-Leakage & Gate 2 Selectivity QSAR Audit
- **Leakage Audit:** `0` Train/Val/Test Bemis-Murcko scaffold overlap across all 12 targets; `0` clinical drug or native co-crystal ligand SMILES/scaffold leakage into Train (25 clinical drugs + 29 native ligands protected).
- **DRD1 Selectivity Curation:** Excluded `138` DRD2-promiscuous duals from DRD1 Train positives.
- **Selectivity QSAR 5-Fold Scaffold-CV:** `DRD1` ROC-AUC = **`0.9901`** (ECE `0.0083`), `DRD2` ROC-AUC = **`0.9698`** (ECE `0.0187`).
- **Clinical QSAR Sanity:** Ecopipam `P(DRD1)=1.000`, `P(DRD2)=0.247` (OOS `Δ=+0.262`); Haloperidol `P(DRD2)=0.998`; Risperidone `P(DRD2)=0.968`.

## 2. Main Progression Table (All 12 Human CNS Targets, N=128/Target)

| Arm | Valid | Uniq | Strict Nov | IntDiv1 | MPO v2 | P(BBB+) | P(Pgp) | P(hERG) | Min AD | Joint CNS-Safe | DRD1 Δ_sel | Test-Scaf SNN | Pred pKd |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ctrl-0A (Random ZINC-250k)** | 100.0% | 100.0% | 1.5% | 0.878 | 4.85 | 0.621 | 0.283 | 0.215 | 0.270 | **3.5%** | -0.000 | 0.209 | 6.44 |
| **Ctrl-0B (MW/LogP/TPSA/HBD ZINC)** | 100.0% | 98.0% | 1.6% | 0.897 | 4.94 | 0.682 | 0.129 | 0.162 | 0.260 | **9.6%** | -0.001 | 0.198 | 6.28 |
| **Iter-0 (v3.0 Zero-Shot Matched)** | 98.2% | 94.2% | 98.9% | 0.856 | 4.80 | 0.697 | 0.241 | 0.177 | 0.321 | **16.2%** | +0.001 | 0.218 | 5.82 |
| **Iter-0 (v3.0 Shuffled Control)** | 98.4% | 94.9% | 98.9% | 0.861 | 4.75 | 0.704 | 0.244 | 0.192 | 0.319 | **15.4%** | -0.022 | 0.218 | 5.82 |
| **Iter-0 (v3.0 Null Control)** | 98.8% | 90.9% | 98.0% | 0.861 | 4.39 | 0.587 | 0.286 | 0.153 | 0.356 | **8.9%** | -0.006 | 0.217 | 6.05 |
| **Iter-1 (v3.1 CNS SFT Matched)** | 98.5% | 98.4% | 99.2% | 0.863 | 4.36 | 0.805 | 0.504 | 0.531 | 0.327 | **7.5%** | -0.099 | 0.276 | 6.45 |
| **Iter-1 (v3.1 CNS SFT Shuffled)** | 98.6% | 98.7% | 99.3% | 0.861 | 4.34 | 0.801 | 0.516 | 0.542 | 0.332 | **8.0%** | -0.084 | 0.261 | 6.44 |
| **Iter-2 (v3.1 CNS-RPO-DPO Matched)** | 99.4% | 90.8% | 98.9% | 0.852 | 5.00 | 0.827 | 0.121 | 0.155 | 0.326 | **37.0%** | +0.077 | 0.269 | 6.02 |
| **Iter-2 (v3.1 CNS-RPO-DPO Shuffled)** | 99.4% | 91.9% | 99.1% | 0.854 | 4.99 | 0.819 | 0.130 | 0.156 | 0.327 | **32.2%** | +0.048 | 0.251 | 6.03 |
| **Iter-2 (v3.1 CNS-RPO-DPO Null)** | 99.5% | 93.8% | 99.8% | 0.849 | 5.15 | 0.805 | 0.145 | 0.153 | 0.320 | **32.1%** | -0.008 | 0.245 | 6.23 |
| **Ablation A3 (DPO No AD-Damping)** | 98.8% | 92.2% | 99.1% | 0.874 | 5.04 | 0.841 | 0.116 | 0.147 | 0.300 | **26.5%** | +0.087 | 0.245 | 6.13 |
| **Ablation A4 (DPO No Selectivity)** | 99.7% | 87.7% | 99.0% | 0.849 | 5.03 | 0.830 | 0.119 | 0.154 | 0.336 | **37.8%** | +0.043 | 0.267 | 5.95 |
| **Ref-Clinical (Drugs + Native)** | 100.0% | 100.0% | 23.1% | 0.839 | 4.69 | 0.916 | 0.397 | 0.509 | 0.408 | **13.9%** | +0.261 | 0.525 | 6.48 |

## 3. Statistical Comparisons (Iter-2 vs Comparators — Core-10 Targets)

| Comparator | Joint CNS-Safe Δ (Wilcoxon p / Cliff δ) | CNS MPO v2 Δ (Wilcoxon p / Cliff δ) | P(hERG) Δ (Wilcoxon p / Cliff δ) | DRD1 Δ_sel (MWU p / Cliff δ) |
| :--- | :---: | :---: | :---: | :---: |
| **vs Iter-1** | +31.6 pp (p=1.95e-03, δ=+0.32) | +0.69 (p=1.95e-03, δ=+0.50) | -0.389 (p=1.95e-03, δ=-0.73) | +0.176 (p=4.25e-04, δ=+0.26) |
| **vs Iter-0** | +21.8 pp (p=1.95e-03, δ=+0.22) | +0.25 (p=1.37e-02, δ=+0.16) | -0.012 (p=2.32e-01, δ=+0.00) | +0.076 (p=7.24e-02, δ=+0.13) |
| **vs Iter-2_Shuffled** | +7.0 pp (p=1.54e-01, δ=+0.07) | +0.05 (p=6.95e-01, δ=+0.04) | +0.009 (p=6.95e-01, δ=+0.04) | +0.029 (p=7.38e-01, δ=+0.02) |
| **vs Iter-2_Null** | +5.4 pp (p=1.60e-01, δ=+0.06) | -0.13 (p=8.40e-02, δ=-0.13) | +0.012 (p=6.95e-01, δ=+0.05) | +0.085 (p=1.94e-02, δ=+0.17) |
| **vs Ctrl-0A** | +34.2 pp (p=1.95e-03, δ=+0.34) | +0.17 (p=1.95e-02, δ=+0.11) | -0.052 (p=1.95e-03, δ=-0.26) | +0.077 (p=5.23e-03, δ=+0.20) |
| **vs Ctrl-0B** | +28.0 pp (p=1.95e-03, δ=+0.28) | +0.05 (p=8.40e-02, δ=+0.03) | -0.004 (p=2.32e-01, δ=-0.07) | +0.078 (p=5.87e-03, δ=+0.20) |
| **vs A3_NoAD** | +10.6 pp (p=3.91e-03, δ=+0.11) | -0.04 (p=3.22e-01, δ=-0.02) | +0.009 (p=1.93e-01, δ=+0.06) | -0.010 (p=1.27e-01, δ=-0.11) |
| **vs A4_NoSel** | -0.3 pp (p=1.00e+00, δ=-0.00) | -0.02 (p=6.25e-01, δ=-0.02) | -0.002 (p=1.00e+00, δ=-0.02) | +0.034 (p=1.52e-01, δ=+0.10) |

## 4. Ablation A5 — CFG Scale Sweep on `v3.1_cns_dpo`

| CFG Scale | Valid | Strict Nov | IntDiv1 | CNS MPO v2 | Joint CNS-Safe | DRD1 Δ_sel | Test-Scaf SNN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w = 1.0` | 99.7% | 99.4% | 0.851 | 5.04 | **34.3%** | +0.039 | 0.264 |
| `w = 1.3` | 99.4% | 98.9% | 0.852 | 5.00 | **37.0%** | +0.077 | 0.269 |
| `w = 1.6` | 99.4% | 99.2% | 0.857 | 4.95 | **33.5%** | +0.088 | 0.273 |

## 5. Frozen 3D Geometry & CASF-2016 Catastrophic-Forgetting Retention
- **Iter-1 Frozen Check (`encoder`, `cvae`, `heads`):** Bit-exact (`max_abs_diff = 0.0`).
- **Iter-2 Frozen Check (`encoder`, `cvae`, `heads`, `perceiver`, bridge):** Bit-exact (`max_abs_diff = 0.0`).
- **CASF-2016 Retention (Iter-2):** Validity = **`95.9%`**, PaiNN-SE3 `pKd` Pearson `r` = **`0.7461`**, DrugCLIP `R@5` = **`62.1%`**.
- **Iter-2 Perceiver Attention Enrichment (12 CNS Targets):** `<4.5 Å` Contact Enrichment = **`7.12×`**, Canonical Hotspot Enrichment = **`4.25×`**, Hotspot Recall@10 = **`57.2%`**.

## 6. Iteration 3 Docking Handoff
- Exported Top-10 MaxMin-diverse (`min_tanimoto_dist=0.35`, `quality_weight=0.7`) candidates per target across all 12 targets for 7 arms (`Ctrl-0A`, `Ctrl-0B`, `Iter-0`, `Iter-1`, `Iter-2`, `Iter-2_Shuffled`, `A4_NoSel`) to `runs_local.nosync/geneuclidean_v31_cns_dpo/candidates_for_docking.json`.
