# Compound assays and human-relevant lab models

Research review and implementation notes, 28 November 2026.

A generated structure can support a rough solubility estimate immediately. Predicting how much dissolves in a particular well also requires the compound's solid form, pH, and experimental conditions. Predicting liver clearance requires different measurements again. GenEuclidean provides an offline assay for each of these questions. It reports concentrations and clearance with explicit units so a researcher can compare the prediction with a measurement on the same basis.

## From a generated molecule to an assay

```python
from geneuclidean import assays

# A SMILES from design_result.smiles can be supplied here.
estimate = assays.estimate_aqueous_solubility("CC(=O)Nc1ccc(O)cc1")
print(estimate.canonical_smiles, estimate.log_s_mol_l)
```

The estimate is log₁₀ of mol/L using the published RDKit refit of ESOL. The four inputs are molecular weight, Crippen logP, rotatable-bond count and aromatic-atom fraction. Delaney reported errors of approximately a factor of five to eight on three validation sets for his original model. Those figures describe the original validation populations. A specific pH, solid form or confidence interval for a new scaffold requires additional evidence. The API accepts a single net-neutral organic structure. [Delaney's ESOL study](https://doi.org/10.1021/ci034243x) · [Walters' RDKit coefficients and descriptor definitions](https://github.com/PatWalters/solubility/blob/master/esol.py)

```python
# Illustrative buffered well. All amounts are concentrations in µM.
well = assays.aqueous_equilibrium(
    total_um=4, intrinsic_solubility_um=1,
    partner_total_um=1, kd_um=1,
)
print(well.free_um, well.bound_um, well.solid_um)  # 1.0, 0.5, 2.5
print(well.dissolved_um)                          # 1.5

# Illustrative parent concentration series in human liver microsomes.
trace = assays.fit_microsomal_depletion(
    [0, 10, 20], [10, 5, 2.5], protein_mg_per_ml=0.5,
)
print(trace.half_life_min, trace.apparent_clint_ul_min_mg)  # ~10, ~138.6
```

**Dissolution and reversible binding.** Supply the neutral-species intrinsic solubility, S₀, at the assay temperature. For an ideal monoprotic acid the saturation concentration of free solute is S₀(1 + 10^(pH − pKa)). For a base use pKa − pH. The model assumes a strongly buffered aqueous phase in contact with excess neutral solid when saturated. A soluble partner can draw additional material from that solid. At free concentration F, the bound concentration is Pₜ F / (Kd + F). Kd must describe binding of total free solute at the chosen pH. The solver closes both compound and partner balances. Specify the partner identity, solid form, temperature and source of S₀, pKa and Kd in the experiment record. Salts, multiple ionization sites, nonideal concentrated media, reaction products and dissolution rates need other models. [Avdeef on intrinsic and pH-dependent solubility](https://pmc.ncbi.nlm.nih.gov/articles/PMC11954141/) · [Equilibrium binding mass balances](https://doi.org/10.1021/cc800122m) · [Tzanova et al. on soluble complexation](https://doi.org/10.1016/j.ejps.2023.106559)

**A measured comparison.** Tzanova et al. measured hydrocortisone solubility of 0.94 mM alone and 6.27 mM with 10 mM hydroxypropyl-β-cyclodextrin at pH 7.4. With their fitted association constant of 1,438 M⁻¹ and excess solid, the module gives 6.69 mM total dissolved. The difference is 0.42 mM (6.7%). This checks consistency within the study used to fit the constant. An independent test would fix the constant first, then predict a held-out concentration series or another lab's data for the same solid and partner. [Tzanova et al., Tables 2 and 3](https://arts.units.it/bitstream/11368/3056438/1/lavoromax23.pdf)

**Human liver microsomal depletion.** The regression fits log(parent concentration) against time with a free intercept. For a negative slope −k the half-life is ln(2)/k and apparent intrinsic clearance is 1,000 k / protein concentration in µL/min/mg. Use quantified data over the first-order range. A flat or rising series yields an unresolved half-life and clearance. The result includes the fitted slope, sample count and R². Replicates and blank incubations matter because adsorption, degradation and analytical loss can also remove parent. Parent disappearance alone cannot identify a metabolite. Obach's 29-drug study compared microsomal observations with human clearance and evaluated the contribution of blood and microsomal binding. [Obach's human microsome study](https://pubmed.ncbi.nlm.nih.gov/10534321/)

**Hepatic blood clearance.** For a scenario with independently supplied hepatic blood flow Q, unbound fraction in blood fᵤ and scaled whole-organ unbound intrinsic clearance CLᵤ, `well_stirred_hepatic_clearance` evaluates Q fᵤ CLᵤ / (Q + fᵤ CLᵤ). The result tends toward blood flow at high extraction. Microsomal µL/min/mg and the body-weight-normalized mL/min/kg in some datasets require separate binding and scaling information before conversion to whole-organ mL/min. The model covers hepatic elimination under steady mixing. It gives no tissue concentration time course, renal clearance, toxicity, patient-specific effect or dosing recommendation. [Wilkinson and Shand](https://doi.org/10.1002/cpt1975184377) · [Open Systems Pharmacology model structure](https://docs.open-systems-pharmacology.org/mechanistic-modeling-of-pharmacokinetics-and-dynamics/modeling-concepts/principles-of-pbpk-modeling)

## What laboratories can automate

| Approach | Demonstrated measurement | What to carry into a twin |
| --- | --- | --- |
| Automated equilibrium solubility | Robotic dispensing, phase separation and analytical quantification were benchmarked with marketed compounds spanning four orders of magnitude. | Record pH before and after equilibration, solid identity, separation method, recovery and replicate variability. A rapid kinetic-solubility screen has a different endpoint. [Automated equilibrium assay](https://pubmed.ncbi.nlm.nih.gov/21764023/) · [Filter comparison and recovered-solid characterization](https://doi.org/10.1016/j.jpba.2006.12.024) |
| Closed-loop synthesis and testing | An integrated microfluidic platform synthesized, purified and tested compounds in an Abl kinase assay. A mobile robot ran 688 experiments on photocatalyst mixtures over eight days. The measured objectives were specific to each platform. | Link compound identity and purity to an instrument result before feeding a label back to generation. [Abl platform](https://doi.org/10.1021/jm400099d) · [Mobile robotic chemist](https://doi.org/10.1038/s41586-020-2442-2) |
| Hardware and records | PyLabRobot demonstrated portable liquid-handling and plate-reader interfaces. The Open Reaction Database schema can describe conditions, observations, analytical results, units and provenance. | Retain instrument/run ID, sample and well ID, calibration, controls, raw trace, censoring, timestamp and model version. Record measured and simulated observations with distinct origins. [PyLabRobot](https://pmc.ncbi.nlm.nih.gov/articles/PMC10369895/) · [ORD schema](https://docs.open-reaction-database.org/en/stable/schema.html) |
| Quality control | The Z′-factor uses variation and separation of positive and negative control responses to describe an assay window. | Report controls and replicate dispersion with the plate. A good control statistic supplies evidence about the assay window only. [Zhang et al.](https://doi.org/10.1177/108705719900400206) |

The local reactor and plate prototypes address synthesis conditions, transfer simulation and virtual sensor responses. Their dose-response plate requires a supplied potency. The released protein–ligand twin operates on an existing pose and energy model. The compound assays add water-phase mass balance and measured depletion. Simulated activity, force-field energy, an IC50 and an equilibrium Kd have distinct physical meanings.

## Which animal or human studies can alternatives address?

**Liver injury.** In a blinded study of 27 known drugs across 870 human Liver-Chips, the protein-binding-corrected two-donor analysis included 18 drugs. It detected 13 of 15 hepatotoxic drugs and all three negatives, giving 87% sensitivity and 100% specificity in that subset. Three negatives give limited precision, and the decision threshold was selected using these data. The work supports prioritization for this endpoint. Confidence for a new chemical series needs prospective controls and evaluation. [Ewart et al., Table 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC9727064/)

**Distribution and time-dependent exposure.** Novak et al. automated fluid handling and sampling across human organ chips and maintained eight tissue types for three weeks. Herland et al. combined measured linked-chip behavior with scaling to study human pharmacokinetics for two compounds and administration routes. Chip material can itself absorb drug. Grant et al. measured compound-dependent losses and modeled transport through PDMS. A comparison to a twin should distinguish nominal concentration, free concentration in medium and measured outlet concentration. [Automated organ chips](https://doi.org/10.1038/s41551-019-0497-x) · [Linked-chip pharmacokinetics](https://doi.org/10.1038/s41551-019-0498-9) · [Drug transport in PDMS chips](https://doi.org/10.1039/D1LC00348H)

**Cardiac and skin endpoints.** The blinded CiPA multisite study used human induced-pluripotent-stem-cell-derived cardiomyocytes, 28 reference drugs, two cell lines and five platforms. It examined proarrhythmic-risk categories using electrophysiological measurements. Performance varied by compound. OECD Guideline 497 specifies defined combinations of assays and interpretation rules for skin sensitisation. These examples support replacement decisions for a named hazard under an evaluated protocol. Each additional organ or endpoint needs its own evidence. [CiPA multisite study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226030/) · [OECD 497](https://www.oecd.org/en/publications/guideline-no-497-defined-approaches-on-skin-sensitisation_b92879a4-en.html)

**Whole-body models and clinical studies.** PK-Sim and MoBi model organs, blood flows, enzyme activity and binding. A published six-drug interaction network used 112 clinical studies to build models and 57 to evaluate interaction predictions. This level of evidence is what makes an extrapolation assessable. PK-DB supplies study-level measured concentration curves, route and subject context for external validation. FDA evaluates substitution of model evidence for particular clinical PK measurements case by case. Its acceptance of a digital liver letter of intent was a qualification-program step. Evidence for safety and efficacy in people requires endpoint-specific validation. [PBPK network](https://pmc.ncbi.nlm.nih.gov/articles/PMC6202474/) · [PK-DB](https://academic.oup.com/nar/article/49/D1/D1358/5957165) · [FDA PBPK guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/physiologically-based-pharmacokinetic-analyses-format-and-content-guidance-industry) · [FDA digital-liver announcement](https://www.fda.gov/drugs/drug-alerts-and-statements/fda-accepts-first-silico-drug-development-tool-under-istand-program-help-predict-drug-induced-liver)

## Evidence for generated candidates

Report the number drawn, parseable, canonical unique, triage-eligible and returned. Minimum pairwise fingerprint distance enforces a shortlist constraint. Mean pairwise distance, distinct ring scaffolds, reward, QED and alerts describe different tradeoffs. A parseable SMILES or favorable descriptor alone gives insufficient evidence of makeability or activity. MOSES and GuacaMol use complementary criteria, and Gao and Coley demonstrated the synthesis gap directly. [MOSES](https://pmc.ncbi.nlm.nih.gov/articles/PMC7775580/) · [GuacaMol](https://arxiv.org/abs/1811.09621) · [Synthesizability of generative models](https://arxiv.org/abs/2002.07007)

A trained PaiNN pocket encoder and SMILES decoder sampled 128 structures for the 353-atom 10GS pocket in its validation partition. Seeds 42–49 each produced 16 draws on CPU. Of those draws, 102 parsed into distinct compounds and 11 passed structural triage. The table shows how many of the eight requested positions could be filled as the distance requirement increased.

| Minimum fingerprint distance requested | Compounds returned after triage | Distinct ring scaffolds | Closest pair, observed distance |
| --- | ---: | ---: | ---: |
| Score alone | 8 | 8 | 0.710 |
| 0.65 | 8 | 8 | 0.710 |
| 0.75 | 6 | 6 | 0.769 |
| 0.80 | 5 | 5 | 0.827 |
| 0.90 | 3 | 3 | 0.906 |

The same frozen checkpoint and seeds were also evaluated on two complexes in its held-out CASF-2016 test partition. The number passing triage varied from 11 to 73 across all three pockets. Each row represents a separate 128-draw budget. The last column applies the 0.80 distance threshold after triage.

| Pocket | Split | Parseable | Canonical unique | Triage eligible | Returned / distinct ring scaffolds |
| --- | --- | ---: | ---: | ---: | ---: |
| 10GS, glutathione transferase | Validation | 102 | 102 | 11 | 5 / 5 |
| 1BZC, protein tyrosine phosphatase 1B | Test | 112 | 112 | 40 | 8 / 8 |
| 1QKT, mutant estrogen receptor α | Test | 111 | 111 | 73 | 8 / 8 |

The closest selected pair on 1BZC was 0.807 apart and on 1QKT was 0.808 apart. Binding, synthesis and assay performance remain unmeasured for these candidates. [1BZC structure](https://www.rcsb.org/structure/1BZC) · [1QKT structure](https://www.rcsb.org/structure/1QKT)

With the processed pocket cache, vocabulary and trained checkpoint available locally, run `python scripts/benchmark_design_shortlist.py --pocket 10gs` to reproduce the first row. Use `1bzc` or `1qkt` for the held-out runs. The script reports the checkpoint and raw-output SHA-256 digests. The checkpoint was `3d327423109ceb07881f85bb9473975361cabacca54bd483a7974ff6c4a869c3`. The 10GS raw digest was `7cf23cb1d9124ffbff1161b18dda49f0082b3cda4f7f13abcb5cfcd8fe4dce2d`.

For a measured-label check, five outer scaffold folds were evaluated on the first 300 usable public OpenADMET training rows for each endpoint. Each outer training fold used a separate scaffold group split for calibration. With 40 trees, seed 42 and nominal 90% coverage, log₁₀ KSOL had MAE 0.415, Spearman 0.735 and coverage 87.3%. Log₁₀ human liver microsomal CLint had MAE 0.297, Spearman 0.593 and coverage 96.0%. All 300 intervals per endpoint were finite. These are cross-validation measurements on this subset. The challenge defines KSOL in µM and its scaled HLM CLint in mL/min/kg. [ExpansionRx training data](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data) · [Challenge endpoint definitions](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge/blob/179f26532d15eaffe944b824c89add345d1ca7c8/app.py)

```python
from geneuclidean.cns.dmpk_regression import (
    evaluate_scaffold_cv_conformal_dmpk, load_reference_endpoint,
)

for name in ("OpenADMET_log10_KSOL", "OpenADMET_log10_HLM_CLint"):
    smiles, y = load_reference_endpoint(name, "data/cns_neuro/v4_expansion/admet", max_rows=300)
    print(evaluate_scaffold_cv_conformal_dmpk(
        smiles, y, endpoint=name, n_splits=5, n_estimators=40, seed=42,
    ))
```

Prospective validation would freeze the chemical inputs, units, fitted parameters and endpoint before collecting a new series. Hold out whole chemical scaffolds for property models, independent experimental runs for solubility, and entire clinical studies for physiological models. Compare prediction errors, interval coverage, failures and out-of-domain cases at that same level. FDA's ICH M15 guidance describes prospective planning, model evaluation and documentation for a stated context of use. [ICH M15](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m15-general-principles-model-informed-drug-development)
