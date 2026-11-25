# Industry Pharma & CNS Neuroscience Datasets Architecture (GenEuclidean v3.1)

> **Scope:** Why pharmaceutical R&D teams supersede legacy PDBbind v2020 + ZINC-250k stacks, the 2024–2026 commercial-safe structural/SAR/ADMET landscape, seven empirically caught industrial data-hygiene traps, live 5-fold Bemis-Murcko scaffold-CV ADMET benchmark results, and reproducible CLI/Python workflows in `geneuclidean.cns`.

---

## 1. Why Industry Pharma Teams Supersede PDBbind v2020 + ZINC-250k

GenEuclidean v3.0 established strong zero-leakage CASF-2016 performance (`PaiNN-SE3` + `PocketPerceiverResampler` + `ibm-research/GP-MoLFormer-Uniq` + LoRA RPO-DPO: `94.9%` raw validity, `96.6%` PoseBusters pass-all, `6.31×` `<4.5 Å` contact attention enrichment, `pKd r = 0.746`, `DrugCLIP R@5 = 62.1%`). However, pharmaceutical drug-discovery programs require five industrial capabilities absent from PDBbind v2020 + ZINC-250k alone:

1. **Commercial Licensing Freedom:** PDBbind imposes a dual academic/commercial license that blocks unrestricted enterprise biotech R&D without a paid commercial contract. Modern replacements (**PLINDER**, **HiQBind**, **SAIR**, **GPCRdb**, **B3DB**, **TDC**, **Biogen/Polaris**, **OpenADMET**) are released under **CC BY 4.0 / CC0 1.0 / MIT / Apache 2.0**.
2. **Leak-Proof Splits & Structural Hygiene:** Random and temporal PDBbind splits leak homologous pockets and ligands, while CrossDocked2020 suffers from synthetic docking strain and crystallization buffer/lipid contamination. Industry benchmarks require multi-axis similarity-bounded splits (**PLINDER-PL50**), automated chemical sanitization (**HiQBind**), and **paired apo–holo / MD flexibility** (**PLINDER**, **MISATO**).
3. **Clinical Safety & PK Attrition Gates (~90% Failure Reality):** Potency and QED alone do not prevent clinical failure from **hERG QT cardiotoxicity**, **P-glycoprotein (`ABCB1`) brain efflux**, **low BBB exposure ($K_{p,uu,\text{brain}}$)**, **CYP450 drug–drug interactions**, **DILI**, or **Ames mutagenicity**.
4. **Target-Family Conformational State & Polypharmacology Discipline:** Roughly 34% of FDA-approved drugs target GPCRs (475 drugs across 108 GPCRs). Conditioning a 3D generator on an *active* agonist-bound GPCR cryo-EM pocket when designing an *antagonist* (e.g., `DRD1` anti-tic antagonist vs. agonist) produces the wrong pharmacology. Real programs pair **conformational-state annotations (GPCRdb / KLIFS)** with **off-target selectivity panels (ChEMBL 35/36 + NIMH PDSP $K_i$)**.
5. **The Make-on-Demand Synthesizability Chasm:** A low RDKit SA score does not guarantee fast synthesis. Industrial pipelines couple generative models to multi-billion synthon spaces (**Enamine REAL**, **ZINC-22**) to retrieve purchasable 3–4 week analogs.

---

## 2. Verified 2024–2026 Industry Dataset & Licensing Comparison Matrix

**Commercial Licensing Traffic-Light:**
- `OPEN` (`CC0 1.0`, `CC BY 4.0`, `MIT`, `Apache 2.0`): commercial-safe for internal and deployed pharma R&D.
- `SHAREALIKE` (`CC BY-SA 3.0/4.0`): safe for internal R&D; review terms before redistributing derivative databases.
- `RESTRICTED`: academic-only, paid commercial fee, or copyleft (`PDBbind`, `BioLiP2`, `Apo2Mol GPL-2.0`).

| Dataset / Resource | Category | Scale & Content | Modality | License | Live Verified Access / Footprint | Role in GenEuclidean v3.1 / v4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PLINDER-2026** (`PL50`) | Leak-proof 3D SBDD | 449,383+ curated holo PLI systems; 500+ annotations; paired apo + AF2 | 3D Holo + Apo + PLIP + splits | `OPEN` (CC BY 4.0 / Apache 2.0) | Live mirror `cameo3d.org/plinder/PLINDER-2026-09/` (`plinder==1.0.0`); `index/` Parquet 11–384 MB | Leak-proof holo/apo benchmark & commercial 3D scaling |
| **HiQBind** | Curated 3D + Affinity | 32,275 structurally fixed complexes (18,160 PDBs) + $K_d/K_i/\text{IC}_{50}$ | 3D Sanitized + Affinity | `OPEN` (CC BY 4.0 / MIT) | Figshare (`7.53 GB` full archive) | Direct commercial-safe drop-in replacement for PDBbind |
| **SAIR** (SandboxAQ / NVIDIA) | Synthetic 3D + SAR | 5.24M Boltz-1x 3D poses across 1.05M ChEMBL/BindingDB affinity pairs | 3D Co-folded + $\text{IC}_{50}/K_i$ | `OPEN` (CC BY 4.0) | HuggingFace `SandboxAQ/SAIR` (`~2.5 TB` streamable shards) | Cluster-scale `DrugCLIP` & `pKd` pretraining |
| **MISATO** | 3D MD + Quantum | 16,972 explicit-water 10 ns MD trajectories (>170 µs) + 19,443 GFN2-xTB QM | 3D MD + QM charges/orbitals | `SHAREALIKE` (LGPL-2.1; PDBbind origin) | Zenodo (`QM.hdf5` 343 MB; `MD.hdf5` 133 GB) | Induced-fit pocket breathing augmentation + QM priors |
| **GPCRdb** | Target Family (GPCR) | 2,577 live experimental GPCR structures (~200 GPCRs); Active/Inactive tags | 3D State + BW numbering + Drugs | `OPEN` (CC BY 4.0 / Apache 2.0) | REST `gpcrdb.org/services/structure/` (HTTP 200, `<10 MB`) | Conformational-state filtering (`DRD1`, `ADRA2A`, `CHRM4`) |
| **ChEMBL 35 / 36** | Medicinal Chem SAR | 2.5–2.8M compounds; 20.8M+ activities; 13.8k patent assays | 2D SMILES + Assay $K_i/\text{IC}_{50}/\text{EC}_{50}$ | `SHAREALIKE` (CC BY-SA 3.0) | REST `ebi.ac.uk/chembl/api/data/` (3.6 MB cached 12 CNS targets) | Target-family LoRA SFT & selectivity QSAR (`DRD1` vs `DRD2`) |
| **B3DB** | CNS BBB Permeability | 7,807 binary (`BBB+`/`BBB−`, 7,805 RDKit-valid) + 1,058 numeric $\log\text{BB}$ | 2D SMILES + BBB / $\log\text{BB}$ | `OPEN` (CC0 1.0) | GitHub `theochem/B3DB` (`3.0 MB` cached locally) | **Primary BBB+ oracle** in `geneuclidean.cns.admet` |
| **TDC ADMET Core** | Safety & PK Benchmarks | `hERG_Karim` (13,445), `BBB_Martins` (1,975), `Pgp_Broccatelli` (1,212), `hERG` (648), CYP/Ames/DILI | 2D SMILES + Binary/Reg ADMET | `OPEN` (CC BY 4.0 / MIT) | Harvard Dataverse (`1.1 MB` cached locally) | **Primary hERG & P-gp inhibition oracles** + external checks |
| **Biogen / Polaris Fang ADME** | Prospective Industrial DMPK | 3,521 unified commercial Biogen compounds (`MDR1-MDCK` ER, HLM/RLM, PPB, Sol.) | 2D SMILES + Unified Assays | `OPEN` (CC BY 4.0) | `polaris-lib==0.13.0` (`polarishub.io`, HTTP 200) | True continuous `MDR1-MDCK` efflux & DMPK validation |
| **OpenADMET** | Prospective ADMET / Safety | ARPA-H AVOID-OME, ExpansionRx LogD/CL/PPB, CYP/PXR, OpenBind blind sets | 2D/3D Blind Pharma Challenges | `OPEN` (CC BY 4.0 / MIT) | HuggingFace `openadmet` & GitHub (HTTP 200) | Prospective blind ADMET generalization auditing |
| **Enamine REAL & ZINC-22** | Make-on-Demand Space | 16B–100B+ Enamine REAL; 37B+ ZINC-22 enumerated 2D/3D tranches | Purchasable SMILES & 3D Tranches | `OPEN` (Free VS / Commercial supply) | `cartblanche22.docking.org` / AWS tranches (`50–100 MB` slice) | Nearest-neighbor purchasable analog grounding |

---

## 3. Seven Live-Caught Industrial Data Hygiene Traps (Empirical Audit)

During live ingestion and verification into `data/cns_neuro.nosync/` (**17 MB total** across ADMET, 36,328 human ChEMBL target–ligand pairs, and 29 verified *Homo sapiens* 3D structures), automated validation exposed **7 critical traps** that silently corrupt naïve ML pipelines:

1. **Raw Holo PDB/mmCIF Co-Crystal Ligand & Lipid Leakage (`featurizer.py`):**
   - *Trap:* Parsing raw RCSB `6CM4.cif` (`DRD2` + risperidone) while stripping only water (`HOH`) leaked **30 risperidone (`8NU`) heavy atoms + 66 membrane lipid/buffer HETATMs** (= 96 non-protein atoms), inflating the 10 Å pocket from 177 clean protein atoms to 207 atoms and leaking the native drug into `PaiNN-SE3`.
   - *Fix:* Patched `geneuclidean.data.featurizer._load_protein_heavy_atoms` to retain only standard polymer residues (`residue.id[0] == " "`), selenomethionine (`MSE`), and optional catalytic metal ions (`ZN, MG, CA, FE, MN, CU, CO, NI, K, NA`), backed by regression test `test_raw_holo_pdb_strips_organic_hetatms_keeps_mse_and_metals` and 29 pre-cleaned pockets in `data/cns_neuro/structures/clean_pockets/`.
2. **ChEMBL Target ID Trap #1 (`CHEMBL1985` ≠ Human VMAT2):**
   - *Trap:* Secondary literature lists frequently quote `CHEMBL1985` for VMAT2; live ChEMBL API inspection verified `CHEMBL1985` is actually human **Glucagon receptor (`GCGR`, UniProt `P47871`)**.
   - *Fix:* Verified human **VMAT2 (`SLC18A2`, UniProt `Q05940`) = `CHEMBL1893`** (enforced in `geneuclidean.cns.targets` and unit-tested).
3. **ChEMBL Target ID Trap #2 (`CHEMBL4908` = Mouse Taar1, Not Human):**
   - *Trap:* Live ChEMBL verification showed `CHEMBL4908` is ***Mus musculus* `Taar1` (`Q923Y8`)** — mixing rodent SAR with human pockets introduces well-documented species shifts (e.g., human `Ile290^{7.39}` vs. rodent `Asn286`).
   - *Fix:* Verified human **TAAR1 (`Q96RJ0`) = `CHEMBL5857`** (including `EC50` agonist records -> 287 unique human actives).
4. **RCSB Species Trap #3 (`3HR1` = Rat PDE10A, Not Human):**
   - *Trap:* High-resolution `3HR1` (1.53 Å MP-10 complex) maps to ***Rattus norvegicus* `PDE10A` (`Q9QYJ6`)**.
   - *Fix:* Excluded `3HR1` from `CNS_TARGET_REGISTRY` in favor of verified ***Homo sapiens* (`Q9Y233`)** catalytic complexes **`5SJX`** (2.06 Å, exact same MP-10 `PF9` ligand, `Gln726` + `Tyr693` verified in 4.5 Å contacts), **`5SFE`** (1.86 Å), and **`3WYM`** (2.00 Å, balipodect/TAK-063).
5. **TDC `Pgp_Broccatelli` Label Semantics (`Inhibition` ≠ Pure `Efflux Substrate`):**
   - *Trap:* Many ML pipelines label `Pgp_Broccatelli` as "P-gp efflux substrate." Auditing canonical drugs in the file (`Verapamil=1`, `Haloperidol=1`, `Risperidone=1`, `Quinidine=1` vs. `Caffeine=0`) proved `Y=1` measures **P-gp inhibition** (Broccatelli et al., *J. Med. Chem.* 2011).
   - *Fix:* Documented and labeled the oracle explicitly as `P(P-gp inhibitor)` in code, CLI, and JSON outputs, and paired it with **Pfizer CNS MPO v2 (`2 × T_0(HBD)`)** (and Biogen `MDR1-MDCK` efflux ratio for prospective checks) to suppress true brain efflux.
6. **Upstream B3DB Valence Typos & TDC Duplicate Label Conflicts:**
   - *Trap:* B3DB rows 5043 (`mepenzolatebromide`) and 7737 (`tiotidine`) fail RDKit sanitization due to pentavalent `[C+]` typos; `tdc_BBB_Martins.csv` contains 11 canonical duplicate label conflicts after salt-stripping, `tdc_hERG.csv` contains 3, and `tdc_hERG_Karim.csv` contains 6 after salt-stripping.
   - *Fix:* `_deduplicate_binary_dataset()` in `src/geneuclidean/cns/admet.py` applies RDKit largest-fragment canonicalization, drops invalid SMILES, and removes conflicting duplicate labels before strict Bemis-Murcko scaffold splitting.
7. **Live Endpoint Relocations & API Quirks (2026):**
   - *PLINDER:* Legacy anonymous GCS URLs (`storage.googleapis.com/plinder/2024-06/...`) return HTTP 403/401; the live public 2026 HTTP mirror is **`https://cameo3d.org/plinder/PLINDER-2026-09/`** (HTTP 200).
   - *BindingDB REST:* `getLigandsByUniprots` returns JSON under the misspelled root key **`getLindsByUniprotsResponse`** (sic).
   - *NIMH PDSP $K_i$:* Legacy `/databases/kidb.php` returns HTTP 404; live portal relocated to **`https://pdsp.unc.edu/kidb2/kidb/web/`** (HTTP 200).

---

## 4. Verified GenEuclidean v3.1 CNS ADMET Benchmark (`results/tables/cns_admet_cv_metrics.json`)

Trained via `geneuclidean.cns.admet.train_default_cns_admet_suite()` using a calibrated 50/50 `HistGradientBoostingClassifier` + `RandomForestClassifier` ensemble on **1,024-bit ECFP4 + 12 RDKit 2D descriptors** (including SMARTS `MostBasicPKa`) with scaffold-disjoint Platt calibration and bit-packed ECFP4 Applicability Domain (`AD_sim >= 0.30`):

| Oracle | Dataset (`N` clean unique) | Unique Scaffolds | Scaffold Leakage | 5-Fold OOF ROC-AUC | 5-Fold OOF PR-AUC | Calibrated Brier | Calibrated ECE (10-bin) | Non-Overlapping External Check |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`bbb`** | `B3DB_classification` (`7,802`) | 2,437 | **0** | **0.9150** (`0.9186 ± 0.0144`) | **0.9462** | **0.1121** | **0.0304** (raw `0.0398`) | `tdc_BBB_Martins` novel (`n=227`): **0.8027** (full `0.9721`) |
| **`pgp`** | `tdc_Pgp_Broccatelli` (`1,212`) | 700 | **0** | **0.9269** (`0.9334 ± 0.0197`) | **0.9426** | **0.1035** | **0.0253** (raw `0.0481`) | Clinical check: Verapamil `0.976` vs. Caffeine `0.118` |
| **`herg`** | `tdc_hERG_Karim` (`13,268`) | 6,016 | **0** | **0.8532** (`0.8537 ± 0.0130`) | **0.8563** | **0.1565** | **0.0126** (raw `0.0551`) | `tdc_hERG` Wang novel (`n=186`): **0.9118** (full `0.9174`) |

> **Clinical Safety Calibration Verification:** Without hardcoding any clinical drug rules, the trained hERG oracle flags FDA QT-prolongation drugs (`pimozide 0.989`, `haloperidol 0.980`, `risperidone 0.976`, `escitalopram 0.979`) while clearing modern low-QT CNS leads (`xanomeline 0.296`, `ulotaront 0.123`, `guanfacine 0.056`) and rejecting peripheral/polar controls on BBB (`ceftriaxone 0.129`, `metformin 0.282`, `trospium 0.363`).

---

## 5. Reproducible CLI & Python API Usage

### CLI Commands

```bash
# 1. Inspect all 12 verified Homo sapiens Tourette & psychiatric 3D targets
geneuclidean cns-targets
geneuclidean cns-targets --target DRD1
geneuclidean cns-targets --indication Tourette --json

# 2. Score candidate SMILES with Pfizer CNS MPO v1/v2 + calibrated BBB/P-gp/hERG + AD
geneuclidean cns-admet \
  "CCCCCCOc1nsnc1C1=CCCN(C)C1" \
  "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21" \
  "CN(C)C(=N)NC(=N)N"

# 3. Profile 3D non-covalent interactions & render interactive WebGL HTML + 300-DPI PNG
geneuclidean viz3d \
  --pocket data/cns_neuro/structures/clean_pockets/9LLG_DRD1_A1EKL_pocket.pdb \
  --ligand data/cns_neuro/structures/clean_pockets/9LLG_DRD1_A1EKL_lig.sdf \
  --out-html results/figures/drd1_9llg_3d.html \
  --out-png results/figures/drd1_9llg_3d.png \
  --theme dark \
  --title "Human DRD1 (9LLG) Inactive Antagonist Complex"
```

### Python API

```python
from geneuclidean.cns import (
    CNSAdmetSuite,
    compute_cns_composite_reward,
    compute_cns_mpo,
    get_cns_target,
    load_cns_target_pocket,
)

# Pfizer CNS MPO v1 & v2 (Wager et al. 2010 & 2016)
mpo = compute_cns_mpo("CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21")  # Ecopipam
print(f"Ecopipam CNS MPO v1={mpo.cns_mpo_v1:.2f}, v2={mpo.cns_mpo_v2:.2f}, QED={mpo.qed:.3f}")

# Load calibrated scaffold-CV ADMET suite (returns None cleanly in offline CI)
suite = CNSAdmetSuite.try_load()
scores = compute_cns_composite_reward(
    ["CCCCCCOc1nsnc1C1=CCCN(C)C1"],
    admet_suite=suite,
    ad_damp=True,
)
print(scores[0].to_dict())

# Load verified human DRD1 inactive pocket (9LLG, CHEMBL2056, Asp103 anchor)
spec = get_cns_target("DRD1")
pocket = load_cns_target_pocket(spec.target_id)
print(spec.chembl_id, spec.primary_pdb_id, pocket.coords.shape, pocket.node_feats.shape)
```

> **Translational Disclaimer:** All ligands scored, ranked, or generated using GenEuclidean's CNS modules and industry datasets are strictly *in silico* computational hypotheses and are **not** clinically validated medicines or medical advice.
