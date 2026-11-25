# GenEuclidean v3.1 — Iteration 3 Independent 3D Physics, 10×10 Cross-Target Selectivity & Flagship DRD1 vs. DRD2 Summary

## 1. Protocol & Anti-Goodhart Separation
- **Docking Engine:** `AutoDock Vina v1.2.7` (`exhaustiveness=8`, `num_modes=9`, `seed=42`, `cpu=2`/run; Vina was **100% held out** from all Iter-1 SFT & Iter-2 CNS-RPO-DPO training).
- **Universal Tight Per-Axis Orthosteric Box (Directive #156):** `box_axis_i = clamp(native_span_i + 6.0, 14.0, 22.0)` Å centered on native centroid across ALL 13 *Homo sapiens* pockets (10 core + 2 extended + `DRD2:6LUQ`).
- **Ligand Chemistry:** Deterministic physiological pH 7.4 SMARTS protonation -> unbiased RDKit `ETKDGv3` + `MMFF94s` (`24` confs, per-fused-ring-system SO(3) clustering, up to 2 ring/invertomer conformers pooled & ranked strictly by Vina $\Delta G$) -> Meeko PDBQT.
- **Unique `(pocket, smiles)` Dockings Executed:** `1423` (total wall time: `3756.7 s`).

## 2. Analysis (a) — On-Target 3D Physics Audit (Core-10 Primary Targets)

| Arm | N | Vina ΔG Mean (Med) | LE Mean | LLE_Vina | High-Aff (≤Nat+0.5) | PB Pass | Zero-Clash | MMFF Strain | Anchor (T1 / Any) | Anchor ΔG | PLIF Tan. (T1) | Joint Phys+CNS-Safe |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| **Ctrl-0A** | 100 | -8.10 (-8.24) | **0.382** | 3.61 | 25.0% | 98.0% | 98.0% | 49.8 | 15.0% / 40.0% | -7.67 | 0.307 | **34.0%** |
| **Ctrl-0B** | 100 | -7.72 (-7.68) | **0.409** | 3.63 | 19.0% | 97.0% | 97.0% | 43.4 | 11.0% / 43.0% | -7.31 | 0.264 | **76.0%** |
| **Iter-0** | 100 | -7.81 (-7.84) | **0.426** | 3.23 | 19.0% | 99.0% | 99.0% | 36.8 | 10.0% / 37.0% | -7.44 | 0.286 | **82.0%** |
| **Iter-1** | 100 | -7.87 (-7.91) | **0.399** | 3.58 | 23.0% | 97.0% | 97.0% | 51.7 | 31.0% / 71.0% | -7.37 | 0.310 | **58.0%** |
| **Iter-2** | 100 | -7.79 (-7.76) | **0.402** | 3.90 | 17.0% | 100.0% | 100.0% | 44.3 | 25.0% / 65.0% | -7.32 | 0.256 | **92.0%** |
| **Iter-2_Shuffled** | 100 | -7.42 (-7.45) | **0.385** | 3.47 | 14.0% | 94.0% | 94.0% | 46.8 | 13.0% / 58.0% | -7.09 | 0.238 | **82.0%** |
| **A4_NoSel** | 100 | -7.83 (-7.91) | **0.406** | 3.83 | 17.0% | 99.0% | 99.0% | 43.3 | 22.0% / 55.0% | -7.40 | 0.293 | **93.0%** |

### All-12 Targets Rollup (Core-10 + Extended TAAR1 & DAT)

| Arm | N | Vina ΔG Mean | LE Mean | LLE_Vina | PB Pass | Anchor (T1 / Any) | PLIF Tan. (T1) | Joint Phys+CNS-Safe |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| **Ctrl-0A** | 120 | -8.00 | 0.379 | 3.52 | 98.3% | 15.8% / 40.0% | 0.326 | **33.3%** |
| **Ctrl-0B** | 120 | -7.66 | 0.410 | 3.54 | 97.5% | 13.3% / 44.2% | 0.292 | **75.0%** |
| **Iter-0** | 120 | -7.80 | 0.427 | 3.20 | 99.2% | 11.7% / 36.7% | 0.310 | **80.0%** |
| **Iter-1** | 120 | -7.81 | 0.402 | 3.50 | 96.7% | 30.8% / 70.0% | 0.328 | **59.2%** |
| **Iter-2** | 120 | -7.79 | 0.405 | 3.82 | 100.0% | 24.2% / 63.3% | 0.282 | **93.3%** |
| **Iter-2_Shuffled** | 120 | -7.37 | 0.387 | 3.46 | 94.2% | 16.7% / 60.8% | 0.266 | **83.3%** |
| **A4_NoSel** | 120 | -7.82 | 0.413 | 3.78 | 99.2% | 22.5% / 54.2% | 0.324 | **92.5%** |

### Statistical Comparisons (`Iter-2` vs. Comparators — Core-10, Holm FWER)

| Comparator | ΔVina ΔG [95% CI] (MWU p_H / δ) | ΔLE [95% CI] (MWU p_H / δ) | ΔPLIF [95% CI] (MWU p_H / δ) | ΔJoint Phys+Safe (Wilc p_H / MWU p_H / δ) |
| :--- | :---: | :---: | :---: | :---: |
| **vs Ctrl-0A** | +0.31 [+0.00,+0.58] (p=1.25e-01, δ=+0.19) | +0.019 [+0.003,+0.036] (p=4.52e-02, δ=+0.21) | -0.051 (p=6.89e-02, δ=-0.21) | +58.0 pp (p_W=1.17e-02, p_U=0.00e+00, δ=+0.58) |
| **vs Ctrl-0B** | -0.07 [-0.37,+0.26] (p=1.00e+00, δ=-0.06) | -0.008 [-0.025,+0.009] (p=1.00e+00, δ=-0.04) | -0.008 (p=1.00e+00, δ=-0.04) | +16.0 pp (p_W=3.16e-01, p_U=8.38e-03, δ=+0.16) |
| **vs Iter-0** | +0.02 [-0.27,+0.31] (p=1.00e+00, δ=-0.00) | -0.025 [-0.042,-0.009] (p=2.95e-02, δ=-0.23) | -0.030 (p=3.19e-01, δ=-0.13) | +10.0 pp (p_W=9.38e-02, p_U=1.08e-01, δ=+0.10) |
| **vs Iter-1** | +0.08 [-0.26,+0.41] (p=1.00e+00, δ=+0.06) | +0.002 [-0.015,+0.022] (p=1.00e+00, δ=+0.02) | -0.054 (p=6.92e-02, δ=-0.20) | +34.0 pp (p_W=1.95e-02, p_U=0.00e+00, δ=+0.34) |
| **vs Iter-2_Shuffled** | -0.38 [-0.71,-0.04] (p=1.25e-01, δ=-0.19) | +0.017 [-0.002,+0.036] (p=3.33e-01, δ=+0.14) | +0.018 (p=1.00e+00, δ=+0.04) | +10.0 pp (p_W=3.16e-01, p_U=1.08e-01, δ=+0.10) |
| **vs A4_NoSel** | +0.03 [-0.25,+0.32] (p=1.00e+00, δ=+0.01) | -0.005 [-0.021,+0.011] (p=1.00e+00, δ=-0.01) | -0.037 (p=1.90e-01, δ=-0.16) | -1.0 pp (p_W=1.00e+00, p_U=7.91e-01, δ=-0.01) |

### Native Crystal Baselines in Universal Tight Per-Axis Orthosteric Box

| Pocket | PDB | Native Drug | Tight Box (Å) | Native ΔG (kcal/mol) | LE | Native Anchor Engaged |
| :--- | :--- | :--- | :---: | ---: | ---: | :---: |
| **DRD1** | `9LLG` | Flupentixol | 14.0×14.0×19.2 | -10.28 | 0.343 | True |
| **DRD2** | `6CM4` | Risperidone | 14.0×15.7×20.8 | -11.02 | 0.367 | True |
| **DRD3** | `3PBL` | Eticlopride | 14.0×15.6×14.0 | -8.15 | 0.355 | True |
| **VMAT2** | `8T69` | Tetrabenazine | 14.0×16.2×14.0 | -9.76 | 0.424 | True |
| **ADRA2A** | `7EJ8` | Brimonidine | 14.6×14.0×14.0 | -7.08 | 0.417 | True |
| **HRH3** | `7F61` | PF-03654746 | 14.0×17.6×14.0 | -10.28 | 0.447 | True |
| **PDE10A** | `5SJX` | Mardepodect (MP-10) | 20.7×14.0×14.0 | -11.29 | 0.376 | True |
| **CHRM4** | `8FX5` | Xanomeline | 16.4×14.0×14.0 | -8.58 | 0.452 | True |
| **HTR2A** | `7WC8` | Lumateperone | 14.0×14.0×20.6 | -9.78 | 0.337 | True |
| **SERT** | `5I6X` | Paroxetine | 14.0×14.1×14.0 | -10.69 | 0.446 | True |
| **TAAR1** | `8W88` | Ulotaront | 14.0×14.0×14.0 | -4.95 | 0.412 | True |
| **DAT** | `9JKI` | Bupropion | 14.0×14.6×14.0 | -7.76 | 0.485 | True |
| **DRD2_6LUQ** | `6LUQ` | Haloperidol | 14.0×14.0×22.0 | -10.25 | 0.394 | True |

## 3. Analysis (b) — 10×10 Cross-Target Selectivity Matrix (`Iter-2` Top-5/Target)
- **Raw Vina ΔG Diagonal vs. Off-Diagonal:** `-7.890` vs `-7.663` kcal/mol (enrichment `+0.228` kcal/mol, 10,000× perm $p = 0.02680$).
- **Column-Normalized Z-Score Selectivity:** Diagonal `+0.346` vs Off-Diagonal `-0.038` (10,000× perm $p = 0.00810$, `9/10` targets above pocket mean).
- **Double-Centered Interaction Z-Score (Controls Pocket-Size & Ligand-MW):** Diagonal **`+0.617`** vs Off-Diagonal `-0.069` (10,000× perm **$p = 0.02350$**).
- **3D PLIF Tanimoto Diagonal vs. Off-Diagonal:** `0.263` vs `0.257` (10,000× perm $p = 0.34427$).

| Design \ Docked | DRD1 | DRD2 | DRD3 | VMAT2 | ADRA2A | HRH3 | PDE10A | CHRM4 | HTR2A | SERT |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DRD1** | **-8.78** | -8.73 | -8.12 | -9.59 | -7.16 | -8.52 | -7.95 | -7.66 | -8.69 | -8.78 |
| **DRD2** | -8.12 | **-8.45** | -7.84 | -9.20 | -7.05 | -8.11 | -7.75 | -7.64 | -8.30 | -8.63 |
| **DRD3** | -8.09 | -8.19 | **-7.79** | -8.99 | -6.52 | -7.94 | -7.65 | -7.17 | -8.42 | -8.63 |
| **VMAT2** | -6.93 | -7.04 | -6.29 | **-7.66** | -5.72 | -6.02 | -6.93 | -5.01 | -7.11 | -7.14 |
| **ADRA2A** | -7.45 | -7.81 | -7.27 | -7.80 | **-7.09** | -7.59 | -7.28 | -7.27 | -7.69 | -7.80 |
| **HRH3** | -7.95 | -8.16 | -7.03 | -8.45 | -5.00 | **-7.60** | -7.36 | -6.24 | -7.88 | -8.14 |
| **PDE10A** | -7.88 | -7.64 | -6.77 | -8.21 | -5.36 | -6.48 | **-7.59** | -5.42 | -7.91 | -8.13 |
| **CHRM4** | -8.45 | -8.33 | -7.66 | -9.08 | -6.71 | -7.67 | -7.69 | **-7.59** | -8.44 | -8.47 |
| **HTR2A** | -8.04 | -8.45 | -7.42 | -8.73 | -7.05 | -8.38 | -7.74 | -7.71 | **-8.13** | -8.13 |
| **SERT** | -8.08 | -7.69 | -7.52 | -8.40 | -7.78 | -7.63 | -7.51 | -7.19 | -8.06 | **-8.23** |

## 4. Analysis (c) — Flagship Tourette `DRD1 (9LLG)` vs. Off-Target `DRD2 (6CM4 & 6LUQ)` Selectivity

| Arm / Clinical Control | N | DRD1 (9LLG) ΔG | DRD2 (6CM4) ΔG | DRD2 (6LUQ) ΔG | ΔΔG (9LLG−6CM4) | ΔΔG (9LLG−6LUQ) | ΔΔG Mean | ΔΔG_rel (6CM4) | % ΔΔG_mean ≤ −0.5 | DRD1 Asp103 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Iter-2** | 10 | -8.48 | -8.32 | -8.18 | **-0.16** | **-0.30** | **-0.23** | -0.90 | 30% | 50% |
| **A4_NoSel** | 10 | -8.31 | -8.54 | -8.40 | **+0.23** | **+0.09** | **+0.16** | -0.51 | 10% | 40% |
| **Iter-1** | 10 | -7.61 | -7.65 | -7.45 | **+0.05** | **-0.16** | **-0.06** | -0.69 | 10% | 50% |
| **Iter-0** | 10 | -7.38 | -7.66 | -7.66 | **+0.28** | **+0.28** | **+0.28** | -0.46 | 0% | 30% |
| **Ctrl-0B** | 10 | -7.82 | -7.73 | -7.47 | **-0.09** | **-0.35** | **-0.22** | -0.83 | 20% | 50% |
| **Iter-2_Shuffled** | 10 | -8.20 | -8.02 | -7.53 | **-0.18** | **-0.67** | **-0.42** | -0.92 | 40% | 80% |
| *Ecopipam (D1-selective Ph3 positive ctrl)* | 1 | -7.70 | -6.98 | -6.35 | -0.72 | -1.35 | **-1.04** | -1.46 | — | True |
| *SCH-23390 (D1-selective tool ctrl)* | 1 | -7.30 | -6.76 | -5.86 | -0.54 | -1.45 | **-0.99** | -1.28 | — | True |
| *Haloperidol (D2-selective negative ctrl)* | 1 | -9.81 | -10.74 | -10.25 | +0.93 | +0.44 | **+0.69** | +0.19 | — | True |
| *Risperidone (D2/5-HT2A negative ctrl)* | 1 | -10.91 | -11.02 | -10.45 | +0.12 | -0.46 | **-0.17** | -0.62 | — | True |

### DRD1 vs. DRD2 Statistical Tests (`Iter-2` vs. Comparators on ΔΔG_mean)
- **vs `A4_NoSel`:** Δ = `-0.391` kcal/mol (95% CI `[-0.873, +0.074]`, Cliff's δ = `-0.280`, MWU $p = 0.3075$, Holm $p = 1.0000$).
- **vs `Iter-1`:** Δ = `-0.175` kcal/mol (95% CI `[-0.663, +0.278]`, Cliff's δ = `-0.080`, MWU $p = 0.7913$, Holm $p = 1.0000$).
- **vs `Iter-0`:** Δ = `-0.506` kcal/mol (95% CI `[-0.916, -0.137]`, Cliff's δ = `-0.510`, MWU $p = 0.0587$, Holm $p = 0.2934$).
- **vs `Ctrl-0B`:** Δ = `-0.013` kcal/mol (95% CI `[-0.551, +0.526]`, Cliff's δ = `-0.040`, MWU $p = 0.9097$, Holm $p = 1.0000$).
- **vs `Iter-2_Shuffled`:** Δ = `+0.195` kcal/mol (95% CI `[-0.396, +0.725]`, Cliff's δ = `+0.180`, MWU $p = 0.5205$, Holm $p = 1.0000$).

## 5. Analysis (d) — Top-12 Prioritized *De Novo* CNS Candidate Dossiers (#1 Lead Per Target)
> **Translational Disclaimer:** All prioritized candidates are strictly *in silico* computational hypotheses requiring synthesis and experimental validation.

| Target (PDB) | Vina ΔG | LE | Anchor (Dist Å) | PLIF | PB | MPO v2 | BBB / Pgp / hERG | AD | Max ChEMBL Tan. (Nearest ID) | SMILES |
| :--- | ---: | ---: | :--- | ---: | :---: | ---: | :---: | ---: | :--- | :--- |
| **DRD1** (`9LLG`) | -9.26 | 0.488 | Yes (3.02) | 0.600 | PASS | 5.67 | 0.94 / 0.09 / 0.08 | 0.36 | 0.53 (`CHEMBL288591`) | `CN1CCN(C(=O)c2ccc3ccccc3n2)CC1` |
| **DRD2** (`6CM4`) | -9.84 | 0.469 | Yes (3.85) | 0.438 | PASS | 5.03 | 0.97 / 0.14 / 0.32 | 0.35 | 0.46 (`CHEMBL1081490`) | `O=C1CN(Cc2ccccc2)CCN1c1cccc(F)c1` |
| **DRD3** (`3PBL`) | -7.37 | 0.461 | Yes (2.92) | 0.231 | PASS | 5.00 | 0.85 / 0.04 / 0.03 | 0.35 | 0.36 (`CHEMBL1641821`) | `O=C(NC1CCCCC1)NC1CCCCC1` |
| **VMAT2** (`8T69`) | -9.01 | 0.409 | Yes (3.99) | 0.118 | PASS | 5.90 | 0.87 / 0.04 / 0.15 | 0.38 | 0.45 (`CHEMBL5847379`) | `COc1cc2c(cc1OC)[C@H]1CCC(C(=O)N(C)C)N1CC2` |
| **ADRA2A** (`7EJ8`) | -7.47 | 0.533 | Yes (3.01) | 0.000 | PASS | 4.60 | 0.95 / 0.04 / 0.12 | 0.30 | 0.45 (`CHEMBL2430681`) | `O=C1CNC(c2ccc(Cl)cc2)CN1` |
| **HRH3** (`7F61`) | -9.05 | 0.476 | Yes (3.63) | 0.389 | PASS | 4.58 | 0.96 / 0.09 / 0.10 | 0.35 | 0.37 (`CHEMBL78617`) | `CN1CCCC(C(=O)NCc2cccc(Cl)c2F)C1` |
| **PDE10A** (`5SJX`) | -7.55 | 0.420 | Yes (3.04) | 0.188 | PASS | 5.94 | 0.93 / 0.04 / 0.08 | 0.36 | 0.43 (`CHEMBL3753475`) | `COc1cc2c(cc1OC)C(=O)N(C1CC1)CC2` |
| **CHRM4** (`8FX5`) | -9.26 | 0.514 | Yes (3.39) | 0.389 | PASS | 4.58 | 0.96 / 0.07 / 0.12 | 0.39 | 0.48 (`CHEMBL5938223`) | `CN1CCCC(CCNC(=O)c2ccccc2)C1` |
| **HTR2A** (`7WC8`) | -8.47 | 0.471 | Yes (3.03) | 0.500 | PASS | 4.97 | 0.82 / 0.06 / 0.12 | 0.33 | 0.41 (`CHEMBL126656`) | `CC(C)(C)NC(=O)Cc1cccc2ccccc12` |
| **SERT** (`5I6X`) | -9.49 | 0.500 | Yes (2.77) | 0.357 | PASS | 5.12 | 0.98 / 0.06 / 0.06 | 0.33 | 0.41 (`CHEMBL2347703`) | `O=C1CC(c2ccccc2)(c2ccccc2)CCN1` |
| **TAAR1** (`8W88`) | -7.00 | 0.467 | Yes (3.04) | 0.400 | PASS | 5.12 | 0.91 / 0.02 / 0.07 | 0.40 | 0.40 (`CHEMBL5991741`) | `O=C(NCCC(F)(F)F)c1ccccc1` |
| **DAT** (`9JKI`) | -9.33 | 0.466 | Yes (3.31) | 0.700 | PASS | 5.79 | 0.92 / 0.05 / 0.12 | 0.31 | 0.38 (`CHEMBL541483`) | `CC(C)N1CCC(C(=O)c2ccc3c(c2)OCO3)CC1` |
