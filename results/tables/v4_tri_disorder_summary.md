# GenEuclidean v4.0 Flagship Tri-Disorder & Head-to-Head Local LLM Benchmark Summary

> **Strictly *In Silico* Research Hypotheses:** All generated molecules, QSAR/ADMET predictions, and AutoDock Vina poses are computational research hypotheses requiring synthesis and wet-lab validation.

## 1. Gate 0 Zero-Leakage 24-Target Splits & 26-Endpoint Calibrated QSAR
- **Atlas Coverage:** `24` primary *Homo sapiens* targets (`12` v3.1 Tourette/SCZ + `6` new TS/SCZ + `6` new Alzheimer's) + `7` matched structural/Avoid-ome counters.
- **Zero-Leakage Audit:** `0` generic Bemis-Murcko Train/Val/Test scaffold overlap across all 24 targets; `0` clinical drug (`59`) or native ligand (`46`) leakage into Train.
- **26-Endpoint 5-Fold Scaffold-CV QSAR:** Mean ROC-AUC = **`0.9824`** (min `0.9361`; 14 new endpoints mean = `0.9850`; `0` scaffold leakage across all 26 models).

## 2. Gate 1 Unbiased Native Redocking (12 New TS/SCZ/AD + 4 AD Counters = 16)
- **Protocol:** Universal tight per-axis orthosteric box (`clamp(span + 6, 14, 22)` A), pH 7.4, 3 seeds `(42, 137, 2026)`, retaining `FAD` (`MAOB/MAOA`), `ADP` (`NLRP3`), `ZN/MG` (`PDE1B`).
- **Aggregate Gate 1:** Best-3 RMSD `<= 2.0 A` = **`11/16` (68.8%)** | Top-1 `<= 2.0 A` = **`9/16`** | Median Top-1 = **`1.63 A`** | Median Best-3 = **`1.40 A`** | PoseBusters = **`100.0%`** | Top-1 Anchor = **`83.3%`**.

## 3. Head-to-Head Local Chemical Foundation LLMs (24 Targets, N=64/Target = 1,536/Arm)

| Arm | Valid | Uniq | Strict Nov | IntDiv1 | Triage Pass | DCU/Reag % | v3.1 7-Rule Safe | **v4.0 12-Ep Safe** | MPO v2 | P(hERG) | P(MDR1) | P(on_target) | PDE10A Bleed | DRD3 Bleed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ctrl-ZINC** | 100.0% | 100.0% | 1.6% | 0.879 | 50.3% | 0.78% | 3.8% | **1.2%** | 4.82 | 0.226 | 0.483 | 0.024 | 0.0% | 0.0% |
| **v3.1_GP-MoLFormer** | 98.3% | 93.8% | 99.2% | 0.860 | 85.6% | 0.87% | 26.8% | **6.8%** | 4.72 | 0.136 | 0.448 | 0.153 | 33.3% | 0.0% |
| **v4.0_GP-MoLFormer** | 98.6% | 88.2% | 100.0% | 0.844 | 96.7% | 0.26% | 21.4% | **10.2%** | 4.93 | 0.226 | 0.304 | 0.108 | 0.0% | 0.0% |
| **v4.0_SAFE-GPT-87M** ★ | 99.5% | 57.5% | 99.2% | 0.839 | 97.7% | 0.00% | 58.8% | **37.3%** | 5.17 | 0.127 | 0.059 | 0.092 | 0.0% | 0.0% |
| **v4.0_NovoMolGen-157M** | 90.1% | 93.2% | 99.8% | 0.892 | 85.1% | 0.94% | 20.2% | **9.5%** | 5.16 | 0.193 | 0.245 | 0.066 | 0.0% | 0.0% |
| **v4.0_Best_Shuffled** | 99.3% | 57.6% | 99.1% | 0.836 | 97.6% | 0.00% | 58.1% | **37.1%** | 5.18 | 0.125 | 0.060 | 0.090 | 0.0% | 0.0% |

### Cohort Breakdown (`v4.0 12-Ep Safe %` / `P(on_target)`)

| Arm | v3.1 Core-12 | New TS/SCZ-6 | New Alzheimer's-6 | Overall 24 |
| :--- | :---: | :---: | :---: | :---: |
| **v3.1_GP-MoLFormer** | 8.2% / 0.216 | 5.8% / 0.065 | 5.1% / 0.116 | **6.8% / 0.153** |
| **v4.0_GP-MoLFormer** | 9.4% / 0.121 | 9.2% / 0.066 | 12.7% / 0.124 | **10.2% / 0.108** |
| **v4.0_SAFE-GPT-87M** | 37.1% / 0.120 | 36.0% / 0.030 | 39.0% / 0.099 | **37.3% / 0.092** |
| **v4.0_NovoMolGen-157M** | 7.7% / 0.087 | 10.1% / 0.040 | 12.4% / 0.050 | **9.5% / 0.066** |

## 4. SAFE-GPT 3D-Conditioned Clinical Scaffold Decoration Showcase

| Clinical Core | Target (PDB) | Valid % | Core Kept | Parent Liability | Best Decorated Lead |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Ecopipam** | `DRD1:9LLG` | 100.0% | 10/10 | hERG=0.61, M3=0.04, MPO=4.25 | MPO=4.07, hERG=0.62, M3=0.09, P(on)=1.00 |
| **Donepezil** | `ACHE:4EY7` | 98.4% | 32/32 | hERG=0.95, M3=0.09, MPO=5.02 | MPO=5.95, hERG=0.14, M3=0.03, P(on)=0.95 |
| **Safinamide** | `MAOB:2V5Z` | 100.0% | 37/37 | hERG=0.38, M3=0.10, MPO=4.60 | MPO=5.30, hERG=0.66, M3=0.05, P(on)=0.95 |
| **Xanomeline** | `CHRM4:8FX5` | 100.0% | 46/46 | hERG=0.40, M3=0.97, MPO=5.80 | MPO=5.67, hERG=0.15, M3=0.22, P(on)=0.87 |

## 5. Held-Out AutoDock Vina v1.2.7 Physics & 5 Counter-Target ΔΔG Selectivity Panels
- **All 24 Targets (`v4.0_SAFE-GPT-87M` vs `v3.1`):** `v4.0_SAFE-GPT-87M` Mean Top-1 Vina ΔG = **`-6.52 kcal/mol`** (PoseBusters = **`99.2%`**, Anchor = **`46.7%`**) vs `v3.1` Mean ΔG = **`-7.31 kcal/mol`**.

| Panel | On vs Off Pocket | v4.0 Best ΔΔG (kcal/mol) | v3.1 Baseline ΔΔG | Clinical Comparator ΔΔG |
| :--- | :---: | :---: | :---: | :--- |
| **Tourette_DRD1_vs_DRD2_6CM4** | `DRD1` vs `DRD2` | **+0.51** | +0.37 | Ecopipam: -0.72, Haloperidol: +0.93 |
| **Tourette_DRD1_vs_DRD2_6LUQ** | `DRD1` vs `DRD2_6LUQ` | **+0.36** | -0.10 | Ecopipam: -1.35, Haloperidol: +0.44 |
| **SCZ_AntiSink_PDE10A_vs_VMAT2** | `PDE10A` vs `VMAT2` | **+0.91** | +0.84 | Balipodect: -0.48, Valbenazine: +1.51 |
| **SCZ_NonD2_CHRM4_vs_DRD2** | `CHRM4` vs `DRD2` | **+0.62** | +1.69 | Xanomeline: -1.82 |
| **SCZ_NonD2_CHRM1_vs_DRD2** | `CHRM1` vs `DRD2` | **+0.10** | -0.15 | Betovumeline: -0.43 |
| **AD_MAOB_vs_MAOA** | `MAOB` vs `MAOA` | **+0.01** | -0.42 | Safinamide: -1.76, Tisolagiline: -1.61, Harmine: -0.08 |
| **AD_GSK3B_vs_CDK2** | `GSK3B` vs `CDK2` | **+0.18** | +0.09 | BRD0705: -1.15, CDK2_Tool: +0.61 |
| **AD_ACHE_vs_BCHE** | `ACHE` vs `BCHE` | **-0.96** | -0.79 | Donepezil: -2.52, Tacrine: -0.78 |
| **AD_BACE1_vs_BACE2** | `BACE1` vs `BACE2` | **+0.15** | +0.33 | Umibecestat: -0.76, Verubecestat: +0.01 |

## 6. CASF-2016 Retention & FDA/EMA 7-Step AI Credibility Dossier
- **CASF-2016 Retention (`v4.0_SAFE-GPT-87M`):** Frozen PaiNN-SE3 `pKd r = 0.7461`, DrugCLIP `R@5 = 62.1%`, Gen Validity = `94.7%`.
- **FDA/EMA Credibility Dossier:** Saved to `results/tables/v4_credibility_dossier.json` (`10` SHA-256 tracked artifacts, 10/10 FDA-EMA Guiding Principles).
- **3D Docked SDF Exports:** Saved Top-2 leads/target (all 24 targets) + SAFE-GPT decorated leads in `runs_local.nosync/geneuclidean_v4_tri_disorder/docked_poses_3d/`.
