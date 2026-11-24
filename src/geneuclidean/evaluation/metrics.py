"""5-Tier Gold-Standard SBDD Evaluation Benchmark Suite for GenEuclidean v2.

Implements canonicalized cheminformatics helpers and rigorous evaluation across:
- Tier 1: 2D validity, uniqueness, novelty, MOSES internal diversity (IntDiv1/2),
  Bemis-Murcko scaffold diversity/novelty, SNN, Fréchet Fingerprint Distance (FFD),
  medicinal chemistry filters (QED, SA, NP-likeness, Lipinski Ro5, PAINS, Brenk),
  and Wasserstein-1 / Jensen-Shannon distributional fidelity.
- Tier 2: Causal proof of 3D pocket conditioning (Matched vs. Shuffled vs. Null
  pocket controls) with paired Wilcoxon signed-rank, Mann-Whitney U, Cliff's delta,
  1,000x bootstrap 95% CIs, cross-pocket property correlation, and side-by-side
  true ECFP4 Tanimoto + Morgan-count Dice (exact 2021 thesis comparison).
- Tier 3: 3D cross-attention interpretability audit (contact atom enrichment ratio
  and normalized attention entropy).
- Tier 4: Binding affinity (`pKd`) scoring power and DrugCLIP virtual screening
  retrieval metrics (Recall@K, MRR, ROC-AUC).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import (
    QED,
    Descriptors,
    FilterCatalog,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Silence verbose RDKit parse errors for invalid generated SMILES strings.
RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]

_DESCRIPTORS: Any = Descriptors
_LIPINSKI: Any = Lipinski

PROPERTY_NAMES: tuple[str, ...] = (
    "QED",
    "SA",
    "SA_norm",
    "NP_score",
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
)


# ---------------------------------------------------------------------------
# Cached RDKit Contrib Loaders (SA_Score, NP_Score), Fingerprints & Filters
# ---------------------------------------------------------------------------


def _load_contrib_module(subdir: str, module_name: str) -> Any:
    """Dynamically load a module from RDKit's built-in ``RDContribDir``."""
    contrib_dir = Path(RDConfig.RDContribDir) / subdir
    module_path = contrib_dir / f"{module_name}.py"
    if not module_path.exists():
        msg = f"RDKit Contrib module not found at {module_path}"
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location(f"rdkit_contrib_{module_name}", module_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load module spec from {module_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _get_sascorer() -> Any:
    """Return cached RDKit built-in ``sascorer`` module."""
    return _load_contrib_module("SA_Score", "sascorer")


@lru_cache(maxsize=1)
def _get_npscorer_and_model() -> tuple[Any, Any]:
    """Return cached RDKit built-in ``npscorer`` module and loaded NP score model."""
    npscorer = _load_contrib_module("NP_Score", "npscorer")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fscore = npscorer.readNPModel()
    return npscorer, fscore


@lru_cache(maxsize=4)
def _get_morgan_generator(radius: int = 2, fp_size: int = 2048) -> Any:
    """Return a cached RDKit Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


@lru_cache(maxsize=1)
def _get_pains_catalog() -> Any:
    """Return cached RDKit PAINS (A/B/C) FilterCatalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


@lru_cache(maxsize=1)
def _get_brenk_catalog() -> Any:
    """Return cached RDKit Brenk medicinal chemistry alert FilterCatalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


# ---------------------------------------------------------------------------
# Chemical Helpers
# ---------------------------------------------------------------------------


def parse_mol(smiles: str) -> Chem.Mol | None:
    """Parse and sanitize a SMILES string into an RDKit ``Mol`` or ``None``."""
    if not isinstance(smiles, str):
        return None
    cleaned = smiles.strip()
    if not cleaned:
        return None
    return Chem.MolFromSmiles(cleaned)


def canonicalize_smiles(
    smiles: str,
    *,
    isomeric: bool = True,
    keep_largest_fragment: bool = False,
) -> str | None:
    """Canonicalize a SMILES string via RDKit, returning ``None`` if invalid."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    if keep_largest_fragment:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric))


def get_sa_score(mol: Chem.Mol) -> float:
    """Compute Ertl & Schuffenhauer Synthetic Accessibility (SA) score in ``[1, 10]``."""
    sascorer = _get_sascorer()
    return float(sascorer.calculateScore(mol))


def sa_to_normalized(sa_score: float) -> float:
    """Convert raw SA score ``[1, 10]`` (lower=better) to ``SA_norm = (10 - SA) / 9``."""
    return float(np.clip((10.0 - sa_score) / 9.0, 0.0, 1.0))


def get_np_score(mol: Chem.Mol) -> float:
    """Compute Ertl Natural Product-likeness (NP) score via RDKit Contrib."""
    npscorer, fscore = _get_npscorer_and_model()
    return float(npscorer.scoreMol(mol, fscore))


def get_morgan_bit_fp(
    mol: Chem.Mol,
    radius: int = 2,
    fp_size: int = 2048,
) -> DataStructs.ExplicitBitVect:
    """Compute standard ECFP bit-vector Morgan fingerprint (default ECFP4, 2048-bit)."""
    gen = _get_morgan_generator(radius=radius, fp_size=fp_size)
    return cast(DataStructs.ExplicitBitVect, gen.GetFingerprint(mol))


def get_morgan_count_fp(mol: Chem.Mol, radius: int = 2) -> Any:
    """Compute sparse Morgan count fingerprint (exact match to 2021 thesis Dice metric)."""
    gen = _get_morgan_generator(radius=radius, fp_size=2048)
    return gen.GetSparseCountFingerprint(mol)


def tanimoto_similarity(fp1: Any, fp2: Any) -> float:
    """Compute exact Tanimoto similarity between two RDKit fingerprints."""
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def dice_similarity(fp1: Any, fp2: Any) -> float:
    """Compute exact Dice similarity between two RDKit fingerprints."""
    return float(DataStructs.DiceSimilarity(fp1, fp2))


def is_pains_free(mol: Chem.Mol) -> bool:
    """Return ``True`` if molecule triggers zero PAINS (A/B/C) structural alerts."""
    return not bool(_get_pains_catalog().HasMatch(mol))


def is_brenk_free(mol: Chem.Mol) -> bool:
    """Return ``True`` if molecule triggers zero Brenk medicinal chemistry alerts."""
    return not bool(_get_brenk_catalog().HasMatch(mol))


def lipinski_violations(mol: Chem.Mol) -> int:
    """Count Lipinski Rule-of-5 violations (MW>500, LogP>5, HBD>5, HBA>10)."""
    mw = float(_DESCRIPTORS.MolWt(mol))
    logp = float(_DESCRIPTORS.MolLogP(mol))
    hbd = int(_LIPINSKI.NumHDonors(mol))
    hba = int(_LIPINSKI.NumHAcceptors(mol))
    return int(mw > 500.0) + int(logp > 5.0) + int(hbd > 5) + int(hba > 10)


def passes_lipinski(mol: Chem.Mol, max_violations: int = 1) -> bool:
    """Return ``True`` if molecule satisfies Lipinski Rule-of-5 (``<= max_violations``)."""
    return lipinski_violations(mol) <= max_violations


def get_murcko_scaffold(mol: Chem.Mol, include_chirality: bool = False) -> str:
    """Extract canonical Bemis-Murcko scaffold SMILES (empty string for acyclic mols)."""
    return str(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality))


def compute_molecule_properties(mol: Chem.Mol) -> dict[str, float]:
    """Compute complete medicinal chemistry & physicochemical descriptors for a molecule."""
    sa_val = get_sa_score(mol)
    return {
        "QED": float(QED.qed(mol)),
        "SA": sa_val,
        "SA_norm": sa_to_normalized(sa_val),
        "NP_score": get_np_score(mol),
        "MW": float(_DESCRIPTORS.MolWt(mol)),
        "LogP": float(_DESCRIPTORS.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(_LIPINSKI.NumHDonors(mol)),
        "HBA": float(_LIPINSKI.NumHAcceptors(mol)),
        "RotBonds": float(_LIPINSKI.NumRotatableBonds(mol)),
        "lipinski_pass": float(passes_lipinski(mol, max_violations=1)),
        "pains_free": float(is_pains_free(mol)),
        "brenk_free": float(is_brenk_free(mol)),
    }


def compute_properties_batch(smiles_list: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    """Compute property arrays over all valid RDKit molecules in ``smiles_list``."""
    keys = (*PROPERTY_NAMES, "lipinski_pass", "pains_free", "brenk_free")
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for smi in smiles_list:
        mol = parse_mol(smi)
        if mol is None:
            continue
        props = compute_molecule_properties(mol)
        for k in keys:
            buckets[k].append(props[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def _fps_to_matrix(fps: Sequence[DataStructs.ExplicitBitVect]) -> NDArray[np.float32]:
    """Convert RDKit ``ExplicitBitVect`` sequence to a dense ``(N, D)`` float32 matrix."""
    if not fps:
        return np.empty((0, 0), dtype=np.float32)
    n_bits = int(fps[0].GetNumBits())
    mat = np.zeros((len(fps), n_bits), dtype=np.float32)
    for idx, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, mat[idx])
    return mat


# ---------------------------------------------------------------------------
# Tier 1: 2D Quality, High Variation & Distributional Fidelity
# ---------------------------------------------------------------------------


def compute_internal_diversity(
    fps: Sequence[DataStructs.ExplicitBitVect],
    p: int = 1,
    max_exact: int = 2000,
    seed: int = 42,
) -> float:
    """Compute exact MOSES ECFP4 internal diversity ``IntDiv_p`` in ``[0, 1]``.

    Uses exact vectorized BLAS pairwise Tanimoto for ``len(fps) <= max_exact``,
    and deterministic popcount-stratified subsampling for larger sets.
    """
    n = len(fps)
    if n < 2:
        return 0.0
    if p < 1:
        msg = f"Internal diversity order p must be >= 1, got {p}"
        raise ValueError(msg)

    mat = _fps_to_matrix(fps)
    if n > max_exact:
        # Stratified subsample across fingerprint bit-density quantiles
        rng = np.random.default_rng(seed)
        popcounts = mat.sum(axis=1)
        order = np.argsort(popcounts, kind="mergesort")
        n_strata = min(20, max_exact)
        strata = np.array_split(order, n_strata)
        chosen: list[NDArray[np.intp]] = []
        base_take = max_exact // n_strata
        rem = max_exact - base_take * n_strata
        for idx, stratum in enumerate(strata):
            take = min(len(stratum), base_take + (1 if idx < rem else 0))
            chosen.append(rng.choice(stratum, size=take, replace=False))
        sel = np.concatenate(chosen)
        mat = mat[sel]

    inter = mat @ mat.T
    norms = mat.sum(axis=1)
    union = norms[:, None] + norms[None, :] - inter
    tanimoto = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
    np.fill_diagonal(tanimoto, 1.0)
    tanimoto = np.clip(tanimoto, 0.0, 1.0)

    mean_tp = float(np.mean(np.power(tanimoto, p, dtype=np.float64)))
    return float(np.clip(1.0 - mean_tp ** (1.0 / p), 0.0, 1.0))


def compute_snn(
    gen_fps: Sequence[DataStructs.ExplicitBitVect],
    ref_fps: Sequence[DataStructs.ExplicitBitVect],
) -> float:
    """Compute Similarity to Nearest Neighbor (mean max Tanimoto to reference ligands)."""
    if not gen_fps or not ref_fps:
        return float("nan")

    g_mat = _fps_to_matrix(gen_fps)
    r_mat = _fps_to_matrix(ref_fps)
    g_norms = g_mat.sum(axis=1)
    r_norms = r_mat.sum(axis=1)

    # Chunk over generated molecules to cap peak memory on large sets
    max_sims = np.empty(len(gen_fps), dtype=np.float64)
    chunk_size = 512
    for start in range(0, len(gen_fps), chunk_size):
        end = min(start + chunk_size, len(gen_fps))
        inter = g_mat[start:end] @ r_mat.T
        union = g_norms[start:end, None] + r_norms[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
        max_sims[start:end] = np.max(sim, axis=1)
    return float(np.mean(max_sims))


def compute_frechet_fingerprint_distance(
    gen_fps: Sequence[DataStructs.ExplicitBitVect],
    ref_fps: Sequence[DataStructs.ExplicitBitVect],
    n_components: int = 256,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet Fingerprint Distance (FFD) on SVD/PCA Morgan embeddings.

    Projects both sets into a shared ``<= n_components``-dimensional centered PCA
    subspace fitted on the pooled fingerprint matrix, then computes the exact
    Fréchet Gaussian distance:

    ``||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2 (Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2})``.

    Strictly symmetric, real-valued, non-negative, and satisfies ``FFD(X, X) ~ 0``.
    """
    if not gen_fps or not ref_fps:
        return float("nan")

    x_gen = _fps_to_matrix(gen_fps).astype(np.float64)
    x_ref = _fps_to_matrix(ref_fps).astype(np.float64)
    pooled = np.vstack([x_ref, x_gen])
    pooled_mean = pooled.mean(axis=0, keepdims=True)
    pooled_centered = pooled - pooled_mean

    max_k = min(n_components, pooled_centered.shape[0], pooled_centered.shape[1])
    if max_k < 1:
        return 0.0

    # Deterministic economy SVD (exact PCA basis on pooled data)
    _, _, vt = np.linalg.svd(pooled_centered, full_matrices=False)
    basis = vt[:max_k].T  # (D, k)

    z_gen = (x_gen - pooled_mean) @ basis
    z_ref = (x_ref - pooled_mean) @ basis

    mu_gen = z_gen.mean(axis=0)
    mu_ref = z_ref.mean(axis=0)
    mean_diff_sq = float(np.sum((mu_gen - mu_ref) ** 2))

    if len(z_gen) < 2 or len(z_ref) < 2:
        return max(0.0, mean_diff_sq)

    cov_gen = np.cov(z_gen, rowvar=False)
    cov_ref = np.cov(z_ref, rowvar=False)
    cov_gen = np.atleast_2d(cov_gen)
    cov_ref = np.atleast_2d(cov_ref)

    # Regularize slightly for strict numerical PSD stability
    eye = np.eye(cov_gen.shape[0], dtype=np.float64) * eps
    s1 = 0.5 * (cov_gen + cov_gen.T) + eye
    s2 = 0.5 * (cov_ref + cov_ref.T) + eye

    # Exact symmetric PSD trace of (s1 * s2)^{1/2} = Tr((s1^{1/2} s2 s1^{1/2})^{1/2})
    eigvals_1, eigvecs_1 = np.linalg.eigh(s1)
    sqrt_s1 = (eigvecs_1 * np.sqrt(np.maximum(eigvals_1, 0.0))) @ eigvecs_1.T
    middle = sqrt_s1 @ s2 @ sqrt_s1
    middle = 0.5 * (middle + middle.T)
    eigvals_mid = np.linalg.eigvalsh(middle)
    tr_covmean = float(np.sum(np.sqrt(np.maximum(eigvals_mid, 0.0))))

    ffd = mean_diff_sq + float(np.trace(s1) + np.trace(s2)) - 2.0 * tr_covmean
    return float(max(0.0, ffd))


def compute_wasserstein_1(samples_a: ArrayLike, samples_b: ArrayLike) -> float:
    """Compute 1D Wasserstein-1 (Earth Mover's) distance between two distributions."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(stats.wasserstein_distance(a, b))


def compute_jensen_shannon_divergence(
    samples_a: ArrayLike,
    samples_b: ArrayLike,
    bins: int = 50,
) -> float:
    """Compute base-2 Jensen-Shannon Divergence in ``[0, 1]`` on shared histogram bins."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if np.isclose(lo, hi):
        return 0.0

    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)
    p = hist_a.astype(np.float64) + 1e-12
    q = hist_b.astype(np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    js_dist = float(jensenshannon(p, q, base=2.0))
    return float(np.clip(js_dist**2, 0.0, 1.0))


def evaluate_generation_metrics(
    generated_smiles: Sequence[str],
    train_smiles: Sequence[str],
    reference_smiles: Sequence[str] | None = None,
    *,
    seed: int = 42,
    ffd_components: int = 256,
) -> dict[str, float]:
    """Run Tier 1 2D generation quality, diversity, and distributional fidelity benchmark.

    Always canonicalizes all molecules with RDKit before computing uniqueness,
    novelty, scaffold metrics, and fingerprints. Validity is strictly the raw
    unconstrained fraction of parseable/sanitizable generated strings.
    """
    n_generated = len(generated_smiles)
    valid_mols: list[Chem.Mol] = []
    valid_canon: list[str] = []
    for smi in generated_smiles:
        mol = parse_mol(smi)
        if mol is not None:
            valid_mols.append(mol)
            valid_canon.append(str(Chem.MolToSmiles(mol, canonical=True)))

    n_valid = len(valid_mols)
    validity = (n_valid / n_generated) if n_generated > 0 else 0.0

    # Deduplicate by canonical SMILES (preserving order)
    unique_map: dict[str, Chem.Mol] = {}
    for canon, mol in zip(valid_canon, valid_mols, strict=True):
        if canon not in unique_map:
            unique_map[canon] = mol
    n_unique = len(unique_map)
    uniqueness = (n_unique / n_valid) if n_valid > 0 else 0.0

    # Canonicalize training set & extract training scaffolds
    train_canon_set: set[str] = set()
    train_scaffolds: set[str] = set()
    train_fps: list[DataStructs.ExplicitBitVect] = []
    for smi in train_smiles:
        mol = parse_mol(smi)
        if mol is None:
            continue
        canon = str(Chem.MolToSmiles(mol, canonical=True))
        if canon not in train_canon_set:
            train_canon_set.add(canon)
            train_scaffolds.add(get_murcko_scaffold(mol))
            if reference_smiles is None:
                train_fps.append(get_morgan_bit_fp(mol))

    novel_canon = [c for c in unique_map if c not in train_canon_set]
    n_novel = len(novel_canon)
    novelty = (n_novel / n_unique) if n_unique > 0 else 0.0

    # Fingerprints for valid generated molecules
    valid_fps = [get_morgan_bit_fp(m) for m in valid_mols]
    int_div_1 = compute_internal_diversity(valid_fps, p=1, seed=seed)
    int_div_2 = compute_internal_diversity(valid_fps, p=2, seed=seed)

    # Bemis-Murcko scaffold diversity & novelty (computed over unique valid molecules)
    gen_scaffolds = {get_murcko_scaffold(m) for m in unique_map.values()}
    murcko_scaffold_diversity = (len(gen_scaffolds) / n_unique) if n_unique > 0 else 0.0
    if gen_scaffolds:
        novel_scaffolds = gen_scaffolds - train_scaffolds
        murcko_scaffold_novelty = len(novel_scaffolds) / len(gen_scaffolds)
    else:
        murcko_scaffold_novelty = 0.0

    # Reference set for SNN, FFD, W1, and JSD
    ref_source = reference_smiles if reference_smiles is not None else train_smiles
    ref_mols: list[Chem.Mol] = []
    if reference_smiles is not None:
        for smi in reference_smiles:
            mol = parse_mol(smi)
            if mol is not None:
                ref_mols.append(mol)
        ref_fps = [get_morgan_bit_fp(m) for m in ref_mols]
    else:
        for smi in ref_source:
            mol = parse_mol(smi)
            if mol is not None:
                ref_mols.append(mol)
        ref_fps = train_fps if train_fps else [get_morgan_bit_fp(m) for m in ref_mols]

    snn = compute_snn(valid_fps, ref_fps)
    ffd = compute_frechet_fingerprint_distance(valid_fps, ref_fps, n_components=ffd_components)

    # Physicochemical & medicinal chemistry descriptors
    gen_props_list = [compute_molecule_properties(m) for m in valid_mols]
    ref_props_list = [compute_molecule_properties(m) for m in ref_mols]

    metrics: dict[str, float] = {
        "n_generated": float(n_generated),
        "n_valid": float(n_valid),
        "n_unique": float(n_unique),
        "n_novel": float(n_novel),
        "validity": float(validity),
        "uniqueness": float(uniqueness),
        "novelty": float(novelty),
        "int_div_1": float(int_div_1),
        "int_div_2": float(int_div_2),
        "murcko_scaffold_diversity": float(murcko_scaffold_diversity),
        "murcko_scaffold_novelty": float(murcko_scaffold_novelty),
        "snn": float(snn),
        "ffd": float(ffd),
    }

    if gen_props_list:
        metrics["lipinski_pass_rate"] = float(np.mean([p["lipinski_pass"] for p in gen_props_list]))
        metrics["pains_free_rate"] = float(np.mean([p["pains_free"] for p in gen_props_list]))
        metrics["brenk_free_rate"] = float(np.mean([p["brenk_free"] for p in gen_props_list]))
    else:
        metrics["lipinski_pass_rate"] = 0.0
        metrics["pains_free_rate"] = 0.0
        metrics["brenk_free_rate"] = 0.0

    for prop in PROPERTY_NAMES:
        gen_vals = np.asarray([p[prop] for p in gen_props_list], dtype=np.float64)
        if len(gen_vals) > 0:
            metrics[f"{prop}_mean"] = float(np.mean(gen_vals))
            metrics[f"{prop}_std"] = float(np.std(gen_vals))
        else:
            metrics[f"{prop}_mean"] = float("nan")
            metrics[f"{prop}_std"] = float("nan")

        if ref_props_list and len(gen_vals) > 0:
            ref_vals = np.asarray([p[prop] for p in ref_props_list], dtype=np.float64)
            metrics[f"{prop}_w1"] = compute_wasserstein_1(gen_vals, ref_vals)
            metrics[f"{prop}_jsd"] = compute_jensen_shannon_divergence(gen_vals, ref_vals)
        else:
            metrics[f"{prop}_w1"] = float("nan")
            metrics[f"{prop}_jsd"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Tier 2: Causal Proof of 3D Pocket Conditioning
# ---------------------------------------------------------------------------


def cliffs_delta(x: ArrayLike, y: ArrayLike) -> float:
    """Compute exact non-parametric Cliff's delta effect size in ``[-1, 1]``.

    Positive values indicate ``x`` tends to be larger than ``y``. Uses the exact
    Mann-Whitney U equivalence ``delta = 2 * U / (n_x * n_y) - 1`` in
    ``O((n + m) log(n + m))`` time.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    ya = ya[np.isfinite(ya)]
    nx, ny = len(xa), len(ya)
    if nx == 0 or ny == 0:
        return float("nan")
    u_stat = float(stats.mannwhitneyu(xa, ya, alternative="two-sided").statistic)
    return float(np.clip(2.0 * u_stat / (nx * ny) - 1.0, -1.0, 1.0))


def bootstrap_ci(
    values: ArrayLike,
    *,
    stat_fn: Callable[[NDArray[np.float64]], float] = lambda arr: float(np.mean(arr)),
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute deterministic percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        val = float(stat_fn(arr))
        return (val, val)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.asarray([stat_fn(arr[row]) for row in idx], dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_stats, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def _safe_wilcoxon(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Run paired Wilcoxon signed-rank test safely when all differences are zero."""
    diffs = x - y
    mask = np.isfinite(diffs)
    diffs = diffs[mask]
    if len(diffs) < 2 or np.allclose(diffs, 0.0):
        return (0.0, 1.0)
    res = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    return (float(res.statistic), float(res.pvalue))


def _safe_mannwhitneyu(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Run pooled two-sided Mann-Whitney U test safely."""
    xa = x[np.isfinite(x)]
    ya = y[np.isfinite(y)]
    if len(xa) == 0 or len(ya) == 0:
        return (float("nan"), float("nan"))
    res = stats.mannwhitneyu(xa, ya, alternative="two-sided")
    return (float(res.statistic), float(res.pvalue))


def _evaluate_condition_against_native(
    per_pocket_generated: Mapping[str, Sequence[str]],
    native_cache: Mapping[str, tuple[DataStructs.ExplicitBitVect, Any, dict[str, float]]],
    pocket_ids: Sequence[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Score one generation condition (matched, shuffled, or null) per pocket."""
    per_pocket_tan_mean: dict[str, float] = {}
    per_pocket_tan_top1: dict[str, float] = {}
    per_pocket_tan_top5: dict[str, float] = {}
    per_pocket_bit_dice_mean: dict[str, float] = {}
    per_pocket_cnt_dice_mean: dict[str, float] = {}
    per_pocket_cnt_dice_top1: dict[str, float] = {}
    per_pocket_cnt_dice_top5: dict[str, float] = {}
    intra_uniqueness: list[float] = []
    intra_int_div_1: list[float] = []
    intra_scaffold_div: list[float] = []
    pooled_tanimoto: list[float] = []
    pooled_bit_dice: list[float] = []
    pooled_count_dice: list[float] = []
    gen_prop_means: dict[str, list[float]] = {"MW": [], "LogP": [], "TPSA": []}
    native_prop_vals: dict[str, list[float]] = {"MW": [], "LogP": [], "TPSA": []}

    for pid in pocket_ids:
        gen_list = per_pocket_generated.get(pid, ())
        native_bv, native_cfp, native_props = native_cache[pid]

        valid_mols: list[Chem.Mol] = []
        valid_canon: list[str] = []
        for smi in gen_list:
            mol = parse_mol(smi)
            if mol is not None:
                valid_mols.append(mol)
                valid_canon.append(str(Chem.MolToSmiles(mol, canonical=True)))

        if not valid_mols:
            continue

        bvs = [get_morgan_bit_fp(m) for m in valid_mols]
        cfps = [get_morgan_count_fp(m) for m in valid_mols]

        tans = np.asarray([tanimoto_similarity(fp, native_bv) for fp in bvs], dtype=np.float64)
        bit_dices = np.asarray([dice_similarity(fp, native_bv) for fp in bvs], dtype=np.float64)
        cnt_dices = np.asarray([dice_similarity(cfp, native_cfp) for cfp in cfps], dtype=np.float64)

        sorted_tans = np.sort(tans)[::-1]
        sorted_cnt = np.sort(cnt_dices)[::-1]

        per_pocket_tan_mean[pid] = float(np.mean(tans))
        per_pocket_tan_top1[pid] = float(sorted_tans[0])
        per_pocket_tan_top5[pid] = float(np.mean(sorted_tans[:5]))

        per_pocket_bit_dice_mean[pid] = float(np.mean(bit_dices))
        per_pocket_cnt_dice_mean[pid] = float(np.mean(cnt_dices))
        per_pocket_cnt_dice_top1[pid] = float(sorted_cnt[0])
        per_pocket_cnt_dice_top5[pid] = float(np.mean(sorted_cnt[:5]))

        pooled_tanimoto.extend(tans.tolist())
        pooled_bit_dice.extend(bit_dices.tolist())
        pooled_count_dice.extend(cnt_dices.tolist())

        unique_mols: dict[str, Chem.Mol] = {}
        for canon, mol in zip(valid_canon, valid_mols, strict=True):
            if canon not in unique_mols:
                unique_mols[canon] = mol
        n_u = len(unique_mols)
        intra_uniqueness.append(n_u / len(valid_mols))
        intra_int_div_1.append(compute_internal_diversity(bvs, p=1, seed=seed))
        scaffolds = {get_murcko_scaffold(m) for m in unique_mols.values()}
        intra_scaffold_div.append(len(scaffolds) / n_u if n_u > 0 else 0.0)

        gen_mws = [float(_DESCRIPTORS.MolWt(m)) for m in valid_mols]
        gen_logps = [float(_DESCRIPTORS.MolLogP(m)) for m in valid_mols]
        gen_tpsas = [float(rdMolDescriptors.CalcTPSA(m)) for m in valid_mols]
        gen_prop_means["MW"].append(float(np.mean(gen_mws)))
        gen_prop_means["LogP"].append(float(np.mean(gen_logps)))
        gen_prop_means["TPSA"].append(float(np.mean(gen_tpsas)))
        native_prop_vals["MW"].append(native_props["MW"])
        native_prop_vals["LogP"].append(native_props["LogP"])
        native_prop_vals["TPSA"].append(native_props["TPSA"])

    return {
        "per_pocket_tan_mean": per_pocket_tan_mean,
        "per_pocket_tan_top1": per_pocket_tan_top1,
        "per_pocket_tan_top5": per_pocket_tan_top5,
        "per_pocket_bit_dice_mean": per_pocket_bit_dice_mean,
        "per_pocket_cnt_dice_mean": per_pocket_cnt_dice_mean,
        "per_pocket_cnt_dice_top1": per_pocket_cnt_dice_top1,
        "per_pocket_cnt_dice_top5": per_pocket_cnt_dice_top5,
        "intra_uniqueness": intra_uniqueness,
        "intra_int_div_1": intra_int_div_1,
        "intra_scaffold_div": intra_scaffold_div,
        "pooled_tanimoto": np.asarray(pooled_tanimoto, dtype=np.float64),
        "pooled_bit_dice": np.asarray(pooled_bit_dice, dtype=np.float64),
        "pooled_count_dice": np.asarray(pooled_count_dice, dtype=np.float64),
        "gen_prop_means": gen_prop_means,
        "native_prop_vals": native_prop_vals,
    }


def _compare_two_conditions(
    matched: dict[str, Any],
    control: dict[str, Any],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Wilcoxon signed-rank, Mann-Whitney U, Cliff's delta, and bootstrap CIs."""
    shared_pids = sorted(
        set(matched["per_pocket_tan_mean"].keys()) & set(control["per_pocket_tan_mean"].keys())
    )
    m_pocket_tan = np.asarray(
        [matched["per_pocket_tan_mean"][p] for p in shared_pids], dtype=np.float64
    )
    c_pocket_tan = np.asarray(
        [control["per_pocket_tan_mean"][p] for p in shared_pids], dtype=np.float64
    )
    m_pocket_dice = np.asarray(
        [matched["per_pocket_cnt_dice_mean"][p] for p in shared_pids], dtype=np.float64
    )
    c_pocket_dice = np.asarray(
        [control["per_pocket_cnt_dice_mean"][p] for p in shared_pids], dtype=np.float64
    )

    wilc_stat, wilc_p = _safe_wilcoxon(m_pocket_tan, c_pocket_tan)
    wilc_dice_stat, wilc_dice_p = _safe_wilcoxon(m_pocket_dice, c_pocket_dice)

    mwu_stat, mwu_p = _safe_mannwhitneyu(matched["pooled_tanimoto"], control["pooled_tanimoto"])
    mwu_dice_stat, mwu_dice_p = _safe_mannwhitneyu(
        matched["pooled_count_dice"], control["pooled_count_dice"]
    )

    pooled_delta = cliffs_delta(matched["pooled_tanimoto"], control["pooled_tanimoto"])
    pocket_delta = cliffs_delta(m_pocket_tan, c_pocket_tan)
    pooled_dice_delta = cliffs_delta(matched["pooled_count_dice"], control["pooled_count_dice"])

    diff_tan = m_pocket_tan - c_pocket_tan
    diff_dice = m_pocket_dice - c_pocket_dice

    return {
        "n_paired_pockets": len(shared_pids),
        "control_tanimoto_mean": (
            float(np.mean(c_pocket_tan)) if len(c_pocket_tan) > 0 else float("nan")
        ),
        "control_tanimoto_ci95": bootstrap_ci(c_pocket_tan, n_bootstrap=n_bootstrap, seed=seed),
        "control_count_dice_mean": (
            float(np.mean(c_pocket_dice)) if len(c_pocket_dice) > 0 else float("nan")
        ),
        "control_count_dice_ci95": bootstrap_ci(c_pocket_dice, n_bootstrap=n_bootstrap, seed=seed),
        "diff_tanimoto_mean": (float(np.mean(diff_tan)) if len(diff_tan) > 0 else float("nan")),
        "diff_tanimoto_ci95": bootstrap_ci(diff_tan, n_bootstrap=n_bootstrap, seed=seed),
        "diff_count_dice_mean": (float(np.mean(diff_dice)) if len(diff_dice) > 0 else float("nan")),
        "diff_count_dice_ci95": bootstrap_ci(diff_dice, n_bootstrap=n_bootstrap, seed=seed),
        "wilcoxon_stat": wilc_stat,
        "wilcoxon_p": wilc_p,
        "wilcoxon_count_dice_stat": wilc_dice_stat,
        "wilcoxon_count_dice_p": wilc_dice_p,
        "mannwhitneyu_stat": mwu_stat,
        "mannwhitneyu_p": mwu_p,
        "mannwhitneyu_count_dice_stat": mwu_dice_stat,
        "mannwhitneyu_count_dice_p": mwu_dice_p,
        "cliffs_delta": pooled_delta,
        "pocket_cliffs_delta": pocket_delta,
        "cliffs_delta_count_dice": pooled_dice_delta,
    }


def evaluate_pocket_specificity(
    per_pocket_generated: Mapping[str, Sequence[str]],
    per_pocket_native_smiles: Mapping[str, str],
    per_pocket_shuffled_generated: Mapping[str, Sequence[str]] | None = None,
    per_pocket_null_generated: Mapping[str, Sequence[str]] | None = None,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Tier 2 causal proof of 3D pocket conditioning benchmark.

    Evaluates true ECFP4 bit-vector Tanimoto AND Morgan-count Dice (exact 2021
    thesis comparison) side-by-side, intra-pocket diversity, cross-pocket
    physicochemical correlation with native ligands, and non-parametric
    hypothesis tests (Wilcoxon signed-rank, Mann-Whitney U, Cliff's delta,
    1,000x bootstrap 95% CIs) against shuffled-pocket and null-pocket controls.
    """
    native_cache: dict[str, tuple[DataStructs.ExplicitBitVect, Any, dict[str, float]]] = {}
    for pid, smi in per_pocket_native_smiles.items():
        mol = parse_mol(smi)
        if mol is None:
            continue
        native_cache[pid] = (
            get_morgan_bit_fp(mol),
            get_morgan_count_fp(mol),
            {
                "MW": float(_DESCRIPTORS.MolWt(mol)),
                "LogP": float(_DESCRIPTORS.MolLogP(mol)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            },
        )

    pocket_ids = sorted(set(per_pocket_generated.keys()) & set(native_cache.keys()))
    matched = _evaluate_condition_against_native(
        per_pocket_generated, native_cache, pocket_ids, seed=seed
    )

    m_tan = np.asarray(list(matched["per_pocket_tan_mean"].values()), dtype=np.float64)
    m_tan_top1 = np.asarray(list(matched["per_pocket_tan_top1"].values()), dtype=np.float64)
    m_tan_top5 = np.asarray(list(matched["per_pocket_tan_top5"].values()), dtype=np.float64)
    m_bit_dice = np.asarray(list(matched["per_pocket_bit_dice_mean"].values()), dtype=np.float64)
    m_cnt_dice = np.asarray(list(matched["per_pocket_cnt_dice_mean"].values()), dtype=np.float64)
    m_cnt_top1 = np.asarray(list(matched["per_pocket_cnt_dice_top1"].values()), dtype=np.float64)
    m_cnt_top5 = np.asarray(list(matched["per_pocket_cnt_dice_top5"].values()), dtype=np.float64)

    # Cross-pocket Pearson correlation between generated mean & native crystal ligand properties
    prop_corrs: dict[str, dict[str, float]] = {}
    for prop in ("MW", "LogP", "TPSA"):
        g_arr = np.asarray(matched["gen_prop_means"][prop], dtype=np.float64)
        n_arr = np.asarray(matched["native_prop_vals"][prop], dtype=np.float64)
        if len(g_arr) >= 3 and np.std(g_arr) > 0.0 and np.std(n_arr) > 0.0:
            res = stats.pearsonr(g_arr, n_arr)
            prop_corrs[prop] = {"pearson_r": float(res.statistic), "p_value": float(res.pvalue)}
        else:
            prop_corrs[prop] = {"pearson_r": float("nan"), "p_value": float("nan")}

    def _mean_or_nan(arr: NDArray[np.float64] | Sequence[float]) -> float:
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    def _std_or_nan(arr: NDArray[np.float64] | Sequence[float]) -> float:
        return float(np.std(arr)) if len(arr) > 0 else float("nan")

    result: dict[str, Any] = {
        "n_pockets": len(m_tan),
        "matched_tanimoto_mean": _mean_or_nan(m_tan),
        "matched_tanimoto_std": _std_or_nan(m_tan),
        "matched_tanimoto_ci95": bootstrap_ci(m_tan, n_bootstrap=n_bootstrap, seed=seed),
        "matched_tanimoto_top1_mean": _mean_or_nan(m_tan_top1),
        "matched_tanimoto_top5_mean": _mean_or_nan(m_tan_top5),
        "matched_bit_dice_mean": _mean_or_nan(m_bit_dice),
        "matched_count_dice_mean": _mean_or_nan(m_cnt_dice),
        "matched_count_dice_std": _std_or_nan(m_cnt_dice),
        "matched_count_dice_ci95": bootstrap_ci(m_cnt_dice, n_bootstrap=n_bootstrap, seed=seed),
        "matched_count_dice_top1_mean": _mean_or_nan(m_cnt_top1),
        "matched_count_dice_top5_mean": _mean_or_nan(m_cnt_top5),
        "intra_uniqueness_mean": _mean_or_nan(matched["intra_uniqueness"]),
        "intra_int_div_1_mean": _mean_or_nan(matched["intra_int_div_1"]),
        "intra_scaffold_diversity_mean": _mean_or_nan(matched["intra_scaffold_div"]),
        "property_correlations": prop_corrs,
        "per_pocket_tanimoto": dict(matched["per_pocket_tan_mean"]),
        "per_pocket_count_dice": dict(matched["per_pocket_cnt_dice_mean"]),
        "pooled_matched_tanimoto": matched["pooled_tanimoto"].tolist(),
        "pooled_matched_count_dice": matched["pooled_count_dice"].tolist(),
    }

    if per_pocket_shuffled_generated is not None:
        shuf_ids = sorted(set(per_pocket_shuffled_generated.keys()) & set(native_cache.keys()))
        shuffled = _evaluate_condition_against_native(
            per_pocket_shuffled_generated, native_cache, shuf_ids, seed=seed
        )
        result["vs_shuffled"] = _compare_two_conditions(
            matched, shuffled, n_bootstrap=n_bootstrap, seed=seed
        )
        result["pooled_shuffled_tanimoto"] = shuffled["pooled_tanimoto"].tolist()

    if per_pocket_null_generated is not None:
        null_ids = sorted(set(per_pocket_null_generated.keys()) & set(native_cache.keys()))
        null_cond = _evaluate_condition_against_native(
            per_pocket_null_generated, native_cache, null_ids, seed=seed
        )
        result["vs_null"] = _compare_two_conditions(
            matched, null_cond, n_bootstrap=n_bootstrap, seed=seed
        )
        result["pooled_null_tanimoto"] = null_cond["pooled_tanimoto"].tolist()

    return result


# ---------------------------------------------------------------------------
# Tier 3: 3D Attention Interpretability Audit
# ---------------------------------------------------------------------------


def evaluate_attention_enrichment(
    attn_weights: ArrayLike,
    atom_mask: ArrayLike,
    contact_mask: ArrayLike,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Audit cross-attention enrichment on true ``<4.5 Å`` ligand-contact pocket atoms.

    Supports attention shapes ``(N,)``, ``(T, N)``, ``(B, T, N)``, or ``(B, H, T, N)``
    where the last axis ``N`` indexes pocket atoms. Masks have shape ``(N,)`` (for
    single-pocket inputs) or ``(B, N)`` (for batched inputs).

    Returns contact-vs-peripheral enrichment ratio, enrichment vs. uniform baseline,
    contact attention mass, contact atom fraction, and normalized attention entropy.
    """
    attn = np.asarray(attn_weights, dtype=np.float64)
    a_mask = np.asarray(atom_mask, dtype=bool)
    c_mask = np.asarray(contact_mask, dtype=bool)

    if a_mask.ndim == 1:
        # Single pocket: collapse leading axes (e.g. heads, tokens)
        n_atoms = a_mask.shape[0]
        if attn.shape[-1] != n_atoms:
            msg = f"Attention last dim {attn.shape[-1]} != atom_mask length {n_atoms}"
            raise ValueError(msg)
        attn_batched = attn.reshape(1, -1, n_atoms)
        a_mask_b = a_mask[None, :]
        c_mask_b = c_mask[None, :]
    elif a_mask.ndim == 2:
        bsz, n_atoms = a_mask.shape
        if attn.shape[0] != bsz or attn.shape[-1] != n_atoms:
            msg = f"Attention shape {attn.shape} incompatible with atom_mask shape {a_mask.shape}"
            raise ValueError(msg)
        attn_batched = attn.reshape(bsz, -1, n_atoms)
        a_mask_b = a_mask
        c_mask_b = c_mask
    else:
        msg = f"atom_mask must be 1D or 2D, got shape {a_mask.shape}"
        raise ValueError(msg)

    enrichment_ratios: list[float] = []
    enrichment_vs_uniforms: list[float] = []
    contact_masses: list[float] = []
    contact_fractions: list[float] = []
    mean_contact_attns: list[float] = []
    mean_periph_attns: list[float] = []
    norm_entropies: list[float] = []
    raw_entropies: list[float] = []

    for b in range(attn_batched.shape[0]):
        valid = a_mask_b[b]
        contact = c_mask_b[b] & valid
        periph = valid & (~contact)
        n_valid = int(valid.sum())
        n_contact = int(contact.sum())
        n_periph = int(periph.sum())
        if n_valid == 0:
            continue

        # Mask padded atoms and renormalize per query step
        w = np.where(valid[None, :], np.maximum(attn_batched[b], 0.0), 0.0)
        row_sums = w.sum(axis=-1, keepdims=True)
        valid_rows = row_sums.squeeze(-1) > eps
        if not np.any(valid_rows):
            continue
        w = w[valid_rows] / row_sums[valid_rows]

        # Entropy per query step (averaged over valid query steps)
        p_valid = w[:, valid]
        log_p = np.log(np.maximum(p_valid, eps))
        ent = -np.sum(np.where(p_valid > eps, p_valid * log_p, 0.0), axis=-1)
        mean_ent = float(np.mean(ent))
        raw_entropies.append(mean_ent)
        max_ent = float(np.log(n_valid)) if n_valid > 1 else 1.0
        norm_entropies.append(float(np.clip(mean_ent / max_ent, 0.0, 1.0)) if n_valid > 1 else 0.0)

        # Mean attention profile over pocket atoms
        mean_profile = w.mean(axis=0)
        contact_mass = float(mean_profile[contact].sum()) if n_contact > 0 else 0.0
        contact_frac = n_contact / n_valid
        contact_masses.append(contact_mass)
        contact_fractions.append(contact_frac)

        mean_c = float(mean_profile[contact].mean()) if n_contact > 0 else 0.0
        mean_p = float(mean_profile[periph].mean()) if n_periph > 0 else 0.0
        mean_contact_attns.append(mean_c)
        mean_periph_attns.append(mean_p)

        if n_contact > 0:
            enrichment_vs_uniforms.append(contact_mass / max(contact_frac, eps))
            if n_periph > 0 and mean_p > eps:
                enrichment_ratios.append(mean_c / mean_p)
            elif n_periph == 0:
                enrichment_ratios.append(1.0)

    def _avg(xs: Sequence[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "enrichment_ratio": _avg(enrichment_ratios),
        "enrichment_vs_uniform": _avg(enrichment_vs_uniforms),
        "contact_attn_mass": _avg(contact_masses),
        "contact_atom_fraction": _avg(contact_fractions),
        "mean_contact_attn": _avg(mean_contact_attns),
        "mean_peripheral_attn": _avg(mean_periph_attns),
        "normalized_entropy": _avg(norm_entropies),
        "raw_entropy": _avg(raw_entropies),
    }


# ---------------------------------------------------------------------------
# Tier 4: Binding Affinity (pKd) & DrugCLIP Virtual Screening Retrieval
# ---------------------------------------------------------------------------


def evaluate_affinity_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Compute binding affinity (`pKd`) regression scoring power metrics."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        msg = f"Shape mismatch: y_true {yt.shape} != y_pred {yp.shape}"
        raise ValueError(msg)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    n = len(yt)
    if n == 0:
        return {
            "n": 0.0,
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "kendall_tau": float("nan"),
            "kendall_p": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))

    if n < 2 or np.std(yt) == 0.0 or np.std(yp) == 0.0:
        pearson_r = pearson_p = spearman_rho = spearman_p = kendall_tau = kendall_p = float("nan")
        r2 = float("nan")
    else:
        pr = stats.pearsonr(yt, yp)
        sr = stats.spearmanr(yt, yp)
        kt = stats.kendalltau(yt, yp)
        pearson_r, pearson_p = float(pr.statistic), float(pr.pvalue)
        spearman_rho, spearman_p = float(sr.statistic), float(sr.pvalue)
        kendall_tau, kendall_p = float(kt.statistic), float(kt.pvalue)
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {
        "n": float(n),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_clip_retrieval(
    pocket_embs: ArrayLike,
    ligand_embs: ArrayLike,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Evaluate DrugCLIP pocket-ligand symmetric retrieval (Recall@K, MRR, ROC-AUC).

    Row ``i`` of ``pocket_embs`` and row ``i`` of ``ligand_embs`` form a true
    cognate complex; off-diagonal entries ``(i, j)`` with ``i != j`` are decoys.
    """
    p = np.asarray(pocket_embs, dtype=np.float64)
    lig = np.asarray(ligand_embs, dtype=np.float64)
    if p.ndim != 2 or lig.ndim != 2 or p.shape != lig.shape:
        msg = f"Expected matching 2D arrays, got pocket {p.shape} and ligand {lig.shape}"
        raise ValueError(msg)

    n = p.shape[0]
    if n < 2:
        out: dict[str, float] = {"n": float(n), "roc_auc": float("nan"), "mrr": float("nan")}
        for k in ks:
            out[f"p2l_recall@{k}"] = float("nan")
            out[f"l2p_recall@{k}"] = float("nan")
            out[f"recall@{k}"] = float("nan")
        out["p2l_mrr"] = float("nan")
        out["l2p_mrr"] = float("nan")
        return out

    p_norm = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    l_norm = lig / np.maximum(np.linalg.norm(lig, axis=1, keepdims=True), 1e-12)
    sim = p_norm @ l_norm.T  # (N, N)

    # Compute 1-based ranks of true diagonal entry per row (p2l) and per column (l2p)
    diag = np.diag(sim)
    # Conservative tie handling: count strictly greater entries + 1
    p2l_ranks = np.sum(sim > diag[:, None], axis=1) + 1
    l2p_ranks = np.sum(diag[:, None] < sim.T, axis=1) + 1

    results: dict[str, float] = {"n": float(n)}
    for k in ks:
        p2l_rk = float(np.mean(p2l_ranks <= k))
        l2p_rk = float(np.mean(l2p_ranks <= k))
        results[f"p2l_recall@{k}"] = p2l_rk
        results[f"l2p_recall@{k}"] = l2p_rk
        results[f"recall@{k}"] = 0.5 * (p2l_rk + l2p_rk)

    p2l_mrr = float(np.mean(1.0 / p2l_ranks))
    l2p_mrr = float(np.mean(1.0 / l2p_ranks))
    results["p2l_mrr"] = p2l_mrr
    results["l2p_mrr"] = l2p_mrr
    results["mrr"] = 0.5 * (p2l_mrr + l2p_mrr)

    # Exact ROC-AUC via average-rank Mann-Whitney U equivalence (matches sklearn identically)
    labels = np.eye(n, dtype=bool).ravel()
    scores = sim.ravel()
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos
    ranks = stats.rankdata(scores, method="average")
    pos_rank_sum = float(np.sum(ranks[labels]))
    roc_auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    results["roc_auc"] = float(np.clip(roc_auc, 0.0, 1.0))

    off_diag_mask = ~np.eye(n, dtype=bool)
    pos_mean = float(np.mean(diag))
    neg_mean = float(np.mean(sim[off_diag_mask]))
    results["mean_pos_cosine"] = pos_mean
    results["mean_neg_cosine"] = neg_mean
    results["pos_neg_gap"] = pos_mean - neg_mean
    return results


__all__ = [
    "PROPERTY_NAMES",
    "bootstrap_ci",
    "canonicalize_smiles",
    "cliffs_delta",
    "compute_frechet_fingerprint_distance",
    "compute_internal_diversity",
    "compute_jensen_shannon_divergence",
    "compute_molecule_properties",
    "compute_properties_batch",
    "compute_snn",
    "compute_wasserstein_1",
    "dice_similarity",
    "evaluate_affinity_metrics",
    "evaluate_attention_enrichment",
    "evaluate_clip_retrieval",
    "evaluate_generation_metrics",
    "evaluate_pocket_specificity",
    "get_morgan_bit_fp",
    "get_morgan_count_fp",
    "get_murcko_scaffold",
    "get_np_score",
    "get_sa_score",
    "is_brenk_free",
    "is_pains_free",
    "lipinski_violations",
    "parse_mol",
    "passes_lipinski",
    "sa_to_normalized",
    "tanimoto_similarity",
]
