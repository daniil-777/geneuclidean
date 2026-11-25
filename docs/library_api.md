# GenEuclidean Python Library & SDK Guide (`import geneuclidean as ge`)

GenEuclidean provides a cohesive, strictly typed (**PEP 561** `py.typed`), lazily loaded (**PEP 562** `__getattr__`) scientific Python SDK for $SE(3)/E(3)$-equivariant 3D protein pocket-conditioned *de novo* drug design, physiological pH 7.4 preparation, industrial MedChem structural triage, calibrated CNS ADMET/Avoid-ome scoring, AutoDock Vina 3D physics validation, studio-grade 3D/2D visualization, and FDA/EMA AI credibility reporting.

> **Research Boundary:** All generated molecules, docking poses, and ADMET scores produced by GenEuclidean are strictly *in silico* computational hypotheses. They are not clinically validated medicines and require chemical synthesis, functional pharmacology, off-target profiling, and preclinical validation before translation.

---

## 1. Installation & Modular Dependency Extras

```bash
# Core library (3D pocket featurization, equivariant encoders, CNS MPO, triage, 3D/2D viz)
pip install -e .

# Modular scientific extras
pip install -e ".[cns]"       # Calibrated scaffold-CV CNS ADMET / Avoid-ome oracles
pip install -e ".[viz]"       # 300-DPI static 3D/2D matplotlib + Pillow publication figures
pip install -e ".[docking]"   # Meeko + Gemmi for AutoDock Vina v1.2.7 structural docking
pip install -e ".[llm]"       # Hugging Face transformers, PEFT LoRA, safetensors, accelerate
pip install -e ".[safe]"      # Optional SAFE fragment toolkit integration
pip install -e ".[dev]"       # pytest, pytest-cov, ruff, mypy, type stubs
pip install -e ".[all]"       # Complete batteries-included environment
```

### Zero-Overhead Lazy Imports (PEP 562)
Bare `import geneuclidean as ge` completes in **`< 50 ms`** and **does not eagerly import** `torch`, `transformers`, `matplotlib`, `sklearn`, or `meeko`. Heavy ML stacks are loaded strictly when you invoke `ge.MoleculeDesigner`, while cheminformatics (`ge.score_molecules`, `ge.triage_molecules`, `ge.protonate_ph74`, `ge.list_targets`, `ge.list_backbones`) runs with zero PyTorch overhead.

---

## 2. 60-Second Quickstart

```python
import geneuclidean as ge

# 1. Discover verified human CNS targets and supported chemical LLM backbones (zero PyTorch!)
tourette_targets = ge.list_targets("tourette")
drd1 = ge.get_target("DRD1")
backbones = ge.list_backbones()
print(f"{drd1.target_id} ({drd1.uniprot_id}, PDB {drd1.primary_pdb_id}) | Anchor: {drd1.primary_anchor_residue}")

# 2. Score, protonate (pH 7.4), and MedChem-triage candidate SMILES
candidates = [
    "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",  # Ecopipam (DRD1 Ph3 positive control)
    "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1",               # GenEuclidean v3.1 flagship DRD1_1 lead
    "O=C(NC1CCCCC1)NC1CCCCC1",                      # DCU coupling byproduct (auto-quarantined!)
]
reports = ge.score_molecules(candidates, target="DRD1")
for rep in reports:
    print(
        f"{rep.canonical_smiles} | MPOv2={rep.cns_mpo_v2:.2f} | "
        f"q7.4={rep.net_charge_ph74:+d} | hygiene={rep.hygiene_gate:.2f} | "
        f"reward={rep.composite_reward:.3f}"
    )

# 3. Render a 300-DPI 2D structure gallery
ge.visualize_molecules_2d(
    candidates[:2],
    out_png="results/drd1_leads_2d.png",
    labels=["Ecopipam", "DRD1_1 (GenEuclidean)"],
    title="Tourette DRD1 Candidates",
)
```

---

## 3. High-Level Public API Reference

### 3.1 Registries (`list_targets`, `get_target`, `list_backbones`)
- **`ge.list_targets(indication=None) -> list[CNSTargetSpec]`** — Lists curated *Homo sapiens* CNS targets (filterable by keyword such as `"tourette"`, `"schizophrenia"`, `"alzheimer"`, `"adhd"`).
- **`ge.get_target(target_id) -> CNSTargetSpec`** — Resolves by target ID (`"DRD1"`), gene symbol (`"SLC18A2"`), ChEMBL ID (`"CHEMBL1893"`), or UniProt accession (`"P21728"`).
- **`ge.list_backbones() -> dict[str, dict[str, Any]]`** — Returns metadata for all registered chemical foundation & local LLM backbones (`ibm-research/GP-MoLFormer-Uniq`, `datamol-io/safe-gpt`, `chandar-lab/NovoMolGen_157M_SMILES_BPE`, `Qwen/Qwen3-0.6B-Base`, `mock-tiny-gpt2`, etc.) without importing PyTorch.

### 3.2 3D Binding Pockets (`ge.Pocket`)
- **`ge.Pocket.from_target(target_id, *, pdb_id=None, structures_dir=None, allow_fetch=False)`** — Creates a pocket handle from the curated registry, auto-locating pre-cleaned `*_pocket.pdb` / `*_lig.sdf` files when cached locally.
- **`ge.Pocket.from_pdb(pdb_path, *, center=None, ligand_sdf=None, target_id=None)`** — Wraps any local PDB or mmCIF structure.
- **`pocket.featurize(allow_fetch=False) -> PocketFeatures`** — Computes centered 3D coordinates `[N, 3]` and strictly $E(3)$-invariant 52-dim per-atom descriptors `[N, 52]`.
- **`pocket.audit_hotspots(contact_cutoff=4.5) -> dict`** — Verifies canonical pharmacophore hotspot presence and `< 4.5 Å` ligand contact engagement.
- **`pocket.orthosteric_box(mode="tight_per_axis") -> OrthostericBox`** — Computes universal tight per-axis (`clamp(span_i + 6.0, 14.0, 22.0)` Å) or standard cubic (`clamp(max_span + 8.0, 20.0, 26.0)` Å) AutoDock Vina search boxes.

### 3.3 Physiological pH 7.4 Protonation (`geneuclidean.evaluation.protonation`)
- **`ge.protonate_ph74(mol_or_smiles, *, preserve_coords=False, isomeric=True)`** — Polymorphic deterministic physiological pH 7.4 ionization (returns `str` for SMILES or `Chem.Mol` for RDKit `Mol`).
- **`ge.analyze_protonation_ph74(mol_or_smiles) -> ProtonationResult`** — Detailed microspecies summary (`protonated_smiles`, `net_charge`, `n_cationic_sites`, `n_anionic_sites`, `has_basic_amine`, `is_zwitterion`).
- **`ge.has_basic_amine(mol_or_smiles) -> bool`** & **`ge.estimate_physiological_charge(mol_or_smiles) -> int`** — Fast checks for GPCR $\text{Asp}^{3.32}$ salt-bridge readiness and physiological charge.
- **Physical-organic invariants:** Monoprotonates asymmetric 1,4-piperazines (e.g., flupentixol) at the inductively more basic nitrogen, monoprotonates biguanides (metformin `+1`), protonates acylguanidines (guanfacine `+1`), preserves permanent quaternary cations (trospium `+1`), deprotonates carboxylic/sulfonic/phosphonic acids & tetrazoles, keeps amides/ureas/anilines/sulfonamides/pyridines neutral, and is strictly idempotent.

### 3.4 Industrial MedChem Triage & Scoring (`score_molecules`, `triage_molecules`)
- **`ge.triage_molecule(mol_or_smiles) -> MedChemTriageResult`** / **`ge.triage_molecules(smiles_list)`** — Screens RDKit `PAINS`, `BRENK`, and `NIH` catalogs; explicitly quarantines coupling reagents and synthesis byproducts (`DCU`, `DIU`, `HOBt`, `HOAt`, `DMAP`, `Ph3P=O`, symmetric dialkyl ureas, reactive acyl/sulfonyl halides, epoxides, Michael acceptors); computes Lilly-inspired demerits (`0..300`, pass `<= 100`), ICH S10 phototoxicity flags, stereocenter burden, exact/generic Bemis-Murcko scaffolds, and smooth `hygiene_gate ∈ [0.0, 1.0]`.
- **`ge.is_scaffold_hop(smiles, reference_scaffolds, max_tanimoto=0.55) -> bool`** & **`ge.compute_scaffold_hop_metrics(...)`** — Automated scaffold-hop & FTO pre-screen (with explicit non-legal Markush disclaimer).
- **`ge.score_molecules(smiles_list, *, target=None, include_triage=True, include_admet=True, oracle_path=None) -> list[MoleculeReport]`** — Unified per-molecule scorecard combining Pfizer CNS MPO v1/v2, pH 7.4 ionization, MedChem triage, calibrated BBB/P-gp/hERG ADMET predictions (when oracle bundle is present), and hygiene-gated composite reward.

### 3.5 3D-Conditioned De Novo Generation (`ge.MoleculeDesigner`)
- **`ge.MoleculeDesigner.from_pretrained(checkpoint, *, mode="auto", v2_pocket_ckpt=None, backbone_id=None, device="auto")`** — Auto-detects v2 `GenEuclideanModel` (`model_state_dict`) vs. v3/v4 `PocketConditionedLLM` adapter checkpoints (`bridge_state_dict` + `lora_state_dict`).
- **`ge.MoleculeDesigner.from_mock(device="cpu", seed=42)`** — Zero-download offline 2-layer GPT-2 + `PaiNN-SE3` designer for unit tests and tutorials.
- **`designer.design(pocket, *, n=32, strategy="min_p", cfg_scale=1.6, cns_safe_only=False, rerank=True, top_k_hits=None, seed=42) -> DesignResult`** — End-to-end 3D pocket encoding, KV-cached CFG decoding, multi-task `pKd`/`DrugCLIP` head scoring, CNS MPO/triage/ADMET evaluation, deduplication, and ranking.

### 3.6 3D Structural Docking, Native Redocking & Selectivity (`geneuclidean.evaluation.docking_pipeline`)
- **`ge.compute_orthosteric_box(ligand_or_path, mode="tight_per_axis") -> OrthostericBox`**
- **`ge.generate_ring_diverse_conformers(smiles_or_mol, protonate_ph74=True, num_confs=24, max_ring_confs=4) -> list[RingConformer]`** — Unbiased 2D-to-3D `ETKDGv3` + `MMFF94s` conformers clustered by per-fused-ring-system proper-$SO(3)$ Kabsch RMSD to cover ring puckers, butterfly folds, and pyramidal `[NH+;R]` invertomers.
- **`ge.dock_and_profile(pocket, smiles, ...)`** / **`ge.dock_ligand_to_pocket(...) -> DockingPoseResult`** — Multi-conformer AutoDock Vina docking + PoseBusters + 3D PLIF profiling + dual Top-1 / canonical-anchor reporting (`RuntimeError` with install hint if `vina`/`meeko` missing).
- **`ge.redock_complex(pocket_pdb, native_ligand_sdf, ...)`** / **`ge.redock_native_complex(...) -> RedockingBenchmarkResult`** — Unbiased 2D-to-3D multi-seed native redocking with symmetry-corrected RMSD and Murcko core RMSD.
- **`ge.compute_selectivity_ddg(on_target_dg, off_target_dgs)`** & **`ge.compute_cross_target_matrix(scores, target_names, n_permutations=10000) -> CrossTargetSelectivityResult`** — Computes column z-scores, row z-scores, double-centered interaction residuals, and exact diagonal permutation p-values.

### 3.7 Studio 3D & 2D Visualizers (`visualize_complex`, `visualize_gallery`, `visualize_molecules_2d`)
- **`ge.visualize_complex(pocket_pdb, ligand_sdf, *, out_html=None, out_png=None, theme="dark", ...)`** — Profiles 3D non-covalent interactions and renders a self-contained 3Dmol.js WebGL HTML app and/or a 300-DPI pseudo-Phong PNG with HUD side panel.
- **`ge.visualize_gallery(specs, out_html, ...)`** — Builds a multi-target dark-mode 3Dmol.js WebGL gallery dashboard.
- **`ge.visualize_molecules_2d(smiles_list, out_png, *, labels=None, reference_smiles=None)`** — Renders a 300-DPI publication 2D RDKit molecular grid.

### 3.8 Regulatory AI Credibility Reporting (`build_credibility_dossier`)
- **`ge.build_credibility_dossier(...) -> CredibilityDossier`** — Emits a structured **FDA Jan 2025 (`FDA-2024-D-4689`) 7-step Risk-Based AI Credibility Dossier** aligned with the **Joint FDA-EMA Jan 2026 10 Guiding Principles of Good AI Practice**, tracking Context of Use, SHA-256 dataset/artifact provenance, scaffold-CV calibration/ECE, Split-Conformal UQ, Applicability Domain bounds, and *in silico* boundaries.
- **`ge.export_credibility_json(dossier, out_path)`** & **`ge.export_credibility_markdown(dossier, out_path)`** — Export to machine-readable JSON and audit-ready Markdown.

---

## 4. End-to-End Cookbook Recipes

### Recipe A — Design & Triage *De Novo* Ligands for a Local Pocket PDB
```python
import geneuclidean as ge

pocket = ge.Pocket.from_pdb("path/to/9LLG_DRD1_A1EKL_pocket.pdb", target_id="DRD1")
designer = ge.MoleculeDesigner.from_pretrained(
    "runs/geneuclidean_v31_cns_dpo/best_adapter.pt",
    device="auto",
)
result = designer.design(pocket, n=64, cfg_scale=1.6, cns_safe_only=True, top_k_hits=10)
for hit in result.reports:
    print(hit.canonical_smiles, f"MPO={hit.cns_mpo_v2:.2f}", f"reward={hit.composite_reward:.3f}")
```

### Recipe B — Cross-Target Selectivity Matrix & 3D Studio Export
```python
import geneuclidean as ge

# 1. Evaluate a 10x10 (or NxN) cross-target Vina ΔG matrix
sel = ge.compute_cross_target_matrix(
    dg_matrix,
    target_names=["DRD1", "DRD2", "VMAT2", "ADRA2A", "PDE10A"],
    n_permutations=10000,
)
print(f"Double-centered diagonal permutation p = {sel.double_centered_permutation_p:.5f}")

# 2. Export both interactive WebGL HTML and 300-DPI static 3D PNG
out = ge.visualize_complex(
    pocket_pdb="9LLG_DRD1_pocket.pdb",
    ligand_sdf="DRD1_1_docked.sdf",
    out_html="results/drd1_studio.html",
    out_png="results/drd1_hero.png",
    title="DRD1_1 Flagship Tourette Lead",
    pdb_id="9LLG",
    disease="Tourette Syndrome",
    theme="dark",
)
print("Detected 3D contacts:", out["interactions"].counts_by_type)
```

### Recipe C — Generate an FDA/EMA AI Credibility Audit Card
```python
import geneuclidean as ge

dossier = ge.build_credibility_dossier(
    model_name="GenEuclidean CNS Lead Discovery Suite",
    model_version="3.1.0",
    provenance_files={"admet_cv": "results/tables/cns_admet_cv_metrics.json"},
)
ge.export_credibility_json(dossier, "results/credibility_dossier.json")
ge.export_credibility_markdown(dossier, "results/credibility_dossier.md")
```

---

## 5. PyPI Packaging & Wheel Hygiene Verification

GenEuclidean uses a standard `src/` layout (`src/geneuclidean/`), PEP 621 `pyproject.toml`, and PEP 561 `src/geneuclidean/py.typed` bundled via `[tool.setuptools.package-data]`.

```bash
# Verify dependency consistency, static typing, lint, and wheel metadata
.venv/bin/python -m pip check
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/mypy src
.venv/bin/pytest -q tests/test_library_api.py tests/test_cns.py tests/test_viz3d.py tests/test_cli.py
```
