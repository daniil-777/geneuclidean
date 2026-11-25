# Table 1 — Gate 1 Multi-Seed Native Redocking & 3D PLIF Validation (17 *Homo sapiens* CNS Complexes)

## Protocol Summary
- **Docking Engine:** `AutoDock Vina v1.2.7` (`exhaustiveness=8`, `num_modes=9`, `seeds=[42, 137, 2026]`, `cpu=2`/run)
- **Unbiased 2D-to-3D Protocol:** Canonical isomeric SMILES (crystal 3D coordinates stripped) -> deterministic physiological pH 7.4 protonation -> RDKit `ETKDGv3` + `MMFF94s` conformer generation (48 conformers clustered by per-fused-ring-system internal SO(3) Kabsch RMSD >= 0.28 Å, up to 4 low-energy ring/butterfly/`[NH+;R]` conformers, pooled and ranked strictly by Vina $\Delta G$) -> Meeko PDBQT -> AutoDock Vina.
- **Box Definition:** Centered on native crystal ligand heavy-atom centroid; cubic edge `min(26.0, max(20.0, native_span + 8.0))` Å.
- **Species Integrity:** **17 / 17 (100%) *Homo sapiens*** cleaned receptors (`PDE10A` uses human `5SJX`/`3WYM`, excluding rat `3HR1`; catalytic `Zn2+`/`Mg2+` sit `> 5.6 Å` from ligand).

## Gate 1 Aggregate Rollup

| Cohort | N | Median Top-1 RMSD (Å) | Median Best-of-3 RMSD (Å) | Best-of-3 <= 2.0 Å | Median Vina ΔG (kcal/mol) | Median LE | Mean Top-1 / Best-3 PLIF Tanimoto | Mean Top-1 / Best-3 Res. Rec. | Anchor Recovery (Top-1 / Best-3) | PoseBusters Pass (Top-1 / Best-3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| **Core 10 Primary Targets** | 10 | **1.02** | **1.00** | **90.0%** | -10.64 | 0.403 | 0.624 / 0.659 | 0.750 / 0.798 | **76.7% / 96.7%** | 100.0% / 100.0% |
| **Primary 13 (All 12 Targets incl. DRD1 9URL)** | 13 | **1.21** | **1.03** | **76.9%** | -9.73 | 0.415 | 0.559 / 0.598 | 0.679 / 0.736 | **69.2% / 89.7%** | 100.0% / 100.0% |
| **All 17 Human CNS Complexes (Primary + Secondary)** | 17 | **1.52** | **1.19** | **70.6%** | -9.33 | 0.395 | 0.552 / 0.591 | 0.681 / 0.731 | **66.7% / 84.3%** | 100.0% / 100.0% |

## Per-Complex Gate 1 Validation Table (Mean ± SD Across 3 Seeds: 42, 137, 2026)

| Tier | Target | PDB (Res) | Native Ligand (HA, RotB, q7.4) | Box (Å) | Vina ΔG (kcal/mol) | LE | Top-1 RMSD (Å) | Best-of-3 RMSD (Å) | PLIF Tan. (T1 / B3) | Res. Rec. (T1 / B3) | Canonical Anchor (T1 / B3) | PB Pass (T1) | Gate Status |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :--- | ---: | :--- |
| Core | **DRD1** | `9LLG` (2.77 Å) | Flupentixol (30, 5, +1) | 21.2 | -10.64 ± 0.19 | 0.355 | 3.38 ± 3.49 | 3.13 ± 3.15 | 0.537 / 0.521 | 0.741 / 0.704 | Asp103 (3.32) (67% / 67%) | 100% | `PASS_TOP1` |
| Core-Alt | **DRD1** | `9URL` (3.04 Å) | SKF-83566 (20, 1, +1) | 20.0 | -5.93 ± 0.04 | 0.297 | 4.64 ± 0.11 | 4.53 ± 0.06 | 0.333 / 0.414 | 0.458 / 0.583 | Asp103 (3.32) (33% / 100%) | 100% | `DIAGNOSTIC_WARN` |
| Core | **DRD2** | `6CM4` (2.87 Å) | Risperidone (30, 4, +1) | 22.8 | -11.44 ± 0.45 | 0.381 | 2.62 ± 0.47 | 2.17 ± 0.69 | 0.400 / 0.448 | 0.556 / 0.639 | Asp114 (3.32) (67% / 100%) | 100% | `DIAGNOSTIC_WARN` |
| Core | **DRD3** | `3PBL` (2.89 Å) | Eticlopride (23, 6, +1) | 20.0 | -8.25 ± 0.04 | 0.359 | 1.48 ± 0.41 | 1.14 ± 0.08 | 0.599 / 0.639 | 0.778 / 0.852 | Asp110 (3.32) (67% / 100%) | 100% | `PASS_TOP1` |
| Core | **VMAT2** | `8T69` (2.89 Å) | Tetrabenazine (23, 4, +1) | 20.0 | -9.73 ± 0.02 | 0.423 | 1.05 ± 0.02 | 1.05 ± 0.02 | 0.780 / 0.780 | 0.889 / 0.889 | Glu312/Asp399 (100% / 100%) | 100% | `PASS_TOP1` |
| Core | **ADRA2A** | `7EJ8` (3.00 Å) | Brimonidine (17, 1, +1) | 20.0 | -7.05 ± 0.01 | 0.415 | 1.31 ± 0.01 | 1.31 ± 0.01 | 0.500 / 0.500 | 0.625 / 0.625 | Asp128 (3.32) (100% / 100%) | 100% | `PASS_TOP1` |
| Core | **HRH3** | `7F61` (2.60 Å) | PF-03654746 (23, 5, +1) | 20.0 | -10.57 ± 0.02 | 0.460 | 8.13 ± 0.22 | 1.84 ± 0.04 | 0.246 / 0.447 | 0.357 / 0.667 | Asp114 (3.32)/Glu206 (0% / 100%) | 100% | `PASS_TOP3` |
| Core | **PDE10A** | `5SJX` (2.06 Å) | Mardepodect (MP-10) (30, 5, 0) | 22.7 | -11.22 ± 0.02 | 0.374 | 0.85 ± 0.02 | 0.85 ± 0.02 | 0.806 / 0.806 | 0.917 / 0.917 | Gln726/Tyr693 (100% / 100%) | 100% | `PASS_TOP1` |
| Core | **CHRM4** | `8FX5` (2.45 Å) | Xanomeline (19, 7, +1) | 20.0 | -8.59 ± 0.02 | 0.452 | 0.79 ± 0.03 | 0.79 ± 0.03 | 0.750 / 0.750 | 0.923 / 0.923 | Asp112 (3.32) (100% / 100%) | 100% | `PASS_TOP1` |
| Core | **HTR2A** | `7WC8` (2.45 Å) | Lumateperone (29, 5, +1) | 22.6 | -11.29 ± 0.19 | 0.389 | 1.11 ± 0.39 | 0.77 ± 0.17 | 0.814 / 0.889 | 0.833 / 0.889 | Asp155 (3.32) (67% / 100%) | 100% | `PASS_TOP1` |
| Core | **SERT** | `5I6X` (3.14 Å) | Paroxetine (24, 4, +1) | 20.0 | -10.67 ± 0.06 | 0.445 | 0.86 ± 0.48 | 0.64 ± 0.17 | 0.806 / 0.806 | 0.879 / 0.879 | Asp98 (S1) (100% / 100%) | 100% | `PASS_TOP1` |
| Ext | **TAAR1** | `8W88` (2.60 Å) | Ulotaront (12, 2, +1) | 20.0 | -5.32 ± 0.01 | 0.443 | 9.66 ± 0.03 | 8.99 ± 0.45 | 0.000 / 0.111 | 0.000 / 0.167 | Asp103 (3.32) (0% / 0%) | 100% | `DIAGNOSTIC_WARN` |
| Ext | **DAT** | `9JKI` (2.69 Å) | Bupropion (16, 3, +1) | 20.0 | -7.78 ± 0.01 | 0.486 | 1.52 ± 0.00 | 1.05 ± 0.34 | 0.700 / 0.667 | 0.875 / 0.833 | Asp79 (S1) (100% / 100%) | 100% | `PASS_TOP1` |
| Secondary | **DRD2** | `6LUQ` (3.10 Å) | Haloperidol (26, 6, +1) | 24.3 | -10.15 ± 0.18 | 0.390 | 10.52 ± 0.21 | 3.77 ± 4.37 | 0.665 / 0.807 | 0.722 / 0.889 | Asp114 (3.32) (33% / 100%) | 100% | `PASS_TOP3` |
| Secondary | **VMAT2** | `9KQ8` (3.38 Å) | Valbenazine (30, 7, +2) | 20.0 | -9.32 ± 0.02 | 0.311 | 6.94 ± 0.01 | 2.94 ± 1.13 | 0.465 / 0.508 | 0.633 / 0.633 | Glu312/Asp399 (100% / 67%) | 100% | `BORDERLINE` |
| Secondary | **PDE10A** | `3WYM` (2.00 Å) | Balipodect (TAK-063) (32, 5, 0) | 20.0 | -8.81 ± 0.01 | 0.275 | 7.19 ± 0.02 | 6.23 ± 0.06 | 0.537 / 0.429 | 0.556 / 0.556 | Gln716/Tyr683 (-10 iso) (0% / 0%) | 100% | `DIAGNOSTIC_WARN` |
| Secondary | **ADRA2A** | `9CBL` (2.80 Å) | L-Epinephrine (13, 3, +1) | 20.0 | -6.39 ± 0.03 | 0.491 | 1.98 ± 0.00 | 1.67 ± 0.44 | 0.455 / 0.525 | 0.833 / 0.778 | Asp128 (3.32) (100% / 100%) | 100% | `PASS_TOP1` |

## Diagnostic Transparency & Pockets Requiring Care in Iteration 3

- **`DRD1:9LLG` (Flupentixol, `PASS_TOP1`):** Top-1 anchor recovery 67% (Best-of-3: 67%) | Murcko Core RMSD (T1/B3): `0.67 Å / 0.67 Å`. Best-of-9 median RMSD = `0.97 Å`. Context: Primary inactive DRD1 antagonist template (cis-flupentixol).
- **`DRD1:9URL` (SKF-83566, `DIAGNOSTIC_WARN`):** Exceeds 2.5 A (Top-1 median=4.70 A, Best-of-3 median=4.49 A; Murcko core RMSD=4.09 A; RotB=1); Top-1 anchor recovery 33% (Best-of-3: 100%) | Murcko Core RMSD (T1/B3): `4.21 Å / 4.09 Å`. Best-of-9 median RMSD = `4.39 Å` | Sensitivity probe (seed 42): crystal local-only ΔG=-4.42; tight-box [14.0, 14.0, 14.0] Best-3=4.68 Å (Anc=False); receptor-polar-H Best-3=4.68 Å (Anc=True). Context: Inactive DRD1 benzazepine template (direct bromo-congener of ecopipam).
- **`DRD2:6CM4` (Risperidone, `DIAGNOSTIC_WARN`):** Exceeds 2.5 A (Top-1 median=2.67 A, Best-of-3 median=2.65 A; Murcko core RMSD=2.57 A; RotB=4); Top-1 anchor recovery 67% (Best-of-3: 100%) | Murcko Core RMSD (T1/B3): `2.59 Å / 2.57 Å`. Best-of-9 median RMSD = `1.12 Å` | Sensitivity probe (seed 42): crystal local-only ΔG=-12.31; tight-box [14.0, 15.71, 20.81] Best-3=0.92 Å (Anc=True); receptor-polar-H Best-3=2.67 Å (Anc=True). Context: Primary inactive DRD2 anti-EPS counter-target template.
- **`DRD3:3PBL` (Eticlopride, `PASS_TOP1`):** Top-1 anchor recovery 67% (Best-of-3: 100%) | Murcko Core RMSD (T1/B3): `1.33 Å / 1.32 Å`. Best-of-9 median RMSD = `1.19 Å`. Context: Primary inactive DRD3 reference complex.
- **`HRH3:7F61` (PF-03654746, `PASS_TOP3`):** Top-1 median 8.28 A vs Best-of-3 median 1.86 A (near-degenerate scoring rank; native pose recovered in Top-3); Top-1 anchor recovery 0% (Best-of-3: 100%) | Murcko Core RMSD (T1/B3): `8.05 Å / 1.16 Å`. Best-of-9 median RMSD = `1.07 Å`. Context: Inactive human HRH3 antagonist complex.
- **`HTR2A:7WC8` (Lumateperone, `PASS_TOP1`):** Top-1 anchor recovery 67% (Best-of-3: 100%) | Murcko Core RMSD (T1/B3): `1.01 Å / 0.70 Å`. Best-of-9 median RMSD = `0.69 Å`. Context: Inactive human 5-HT2A complex with Caplyta.
- **`TAAR1:8W88` (Ulotaront, `DIAGNOSTIC_WARN`):** Exceeds 2.5 A (Top-1 median=9.65 A, Best-of-3 median=8.68 A; Murcko core RMSD=7.87 A; RotB=2); Top-1 anchor recovery 0% (Best-of-3: 0%) | Murcko Core RMSD (T1/B3): `8.89 Å / 7.87 Å`. Best-of-9 median RMSD = `0.76 Å` | Sensitivity probe (seed 42): crystal local-only ΔG=-4.38; tight-box [14.0, 14.0, 14.0] Best-3=0.56 Å (Anc=True); receptor-polar-H Best-3=8.44 Å (Anc=False). Context: Extended target: active human TAAR1 Gs-coupled complex (compact 12-HA ligand).
- **`DRD2:6LUQ` (Haloperidol, `PASS_TOP3`):** Top-1 median 10.61 A vs Best-of-3 median 0.68 A (near-degenerate scoring rank; native pose recovered in Top-3); Top-1 anchor recovery 33% (Best-of-3: 100%) | Murcko Core RMSD (T1/B3): `9.82 Å / 0.66 Å`. Best-of-9 median RMSD = `0.68 Å`. Context: Secondary DRD2 conformation with classical TS antipsychotic haloperidol.
- **`VMAT2:9KQ8` (Valbenazine, `BORDERLINE`):** Best-of-3 median 2.19 A (within 2.0-2.5 A tolerance) | Murcko Core RMSD (T1/B3): `6.97 Å / 2.12 Å`. Best-of-9 median RMSD = `2.19 Å`. Context: Secondary VMAT2 conformation with FDA-approved Ingrezza (flexible L-valine ester).
- **`PDE10A:3WYM` (Balipodect (TAK-063), `DIAGNOSTIC_WARN`):** Exceeds 2.5 A (Top-1 median=7.19 A, Best-of-3 median=6.27 A; Murcko core RMSD=5.74 A; RotB=5); Top-1 anchor recovery 0% (Best-of-3: 0%) | Murcko Core RMSD (T1/B3): `6.98 Å / 5.74 Å`. Best-of-9 median RMSD = `5.38 Å` | Sensitivity probe (seed 42): crystal local-only ΔG=-8.45; tight-box [14.0, 16.01, 15.3] Best-3=7.04 Å (Anc=True); receptor-polar-H Best-3=0.67 Å (Anc=True). Context: Secondary human PDE10A Q9Y233 (-10 isoform offset: Gln716=Gln726, Tyr683=Tyr693).
