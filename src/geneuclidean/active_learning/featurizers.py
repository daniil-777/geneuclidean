"""Molecular featurization, caching, and positive semi-definite similarity kernels.

Implements cached RDKit Morgan ECFP4 (binary, count, and physicochemical-augmented)
representations and exact analytical molecular similarity kernels for Active Learning:

- **Tanimoto kernel** (:func:`tanimoto_kernel`): exact Jaccard/Tanimoto similarity
  :math:`k_{\\text{Tan}}(x, y) = \\frac{x^\\top y}{\\|x\\|_2^2 + \\|y\\|_2^2 - x^\\top y}`
  with explicit zero-vector boundary handling (:math:`k(\\mathbf{0}, \\mathbf{0}) = 1`),
  provably positive semi-definite on :math:`\\mathbb{R}_{\\ge 0}^d` (Ralaivola et al., 2005).
- **MinMax / Ruzicka kernel** (:func:`minmax_kernel`): exact continuous count-fingerprint
  kernel :math:`k_{\\text{MM}}(x, y) = \\frac{\\sum_j \\min(x_j, y_j)}{\\sum_j \\max(x_j, y_j)}`,
  strictly matching :func:`tanimoto_kernel` on binary vectors while preserving substructure
  multiplicity on count fingerprints (Griffiths et al., NeurIPS 2023 *GAUCHE*).

All heavy RDKit computations use modern ``rdFingerprintGenerator.GetMorganGenerator``
(avoiding deprecated ``GetMorganFingerprintAsBitVect`` warnings) and memoize per-SMILES
feature vectors so iterative DMTA rounds never re-featurize static candidate pools.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from geneuclidean.cns.mpo import estimate_most_basic_pka

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

FeaturizerKind = Literal["ecfp4_binary", "ecfp4_count", "ecfp4_physchem"]

_DESCRIPTORS: Any = Descriptors
_CRIPPEN: Any = Crippen
_MURCKO: Any = MurckoScaffold
_PHYSCHEM_DIM: int = 12


@lru_cache(maxsize=64)
def _get_morgan_generator(radius: int, n_bits: int, use_chirality: bool) -> Any:
    """Return a cached RDKit Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
        includeChirality=bool(use_chirality),
    )


@lru_cache(maxsize=65536)
def canonicalize_smiles(smi: str, *, isomeric: bool = True) -> str | None:
    """Canonicalize a SMILES string with RDKit, returning ``None`` if invalid."""
    if not isinstance(smi, str):
        return None
    cleaned = smi.strip()
    if not cleaned:
        return None
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=bool(isomeric)))


@lru_cache(maxsize=65536)
def compute_murcko_scaffold_cached(smiles: str, *, include_chirality: bool = False) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES (or canonical SMILES if acyclic)."""
    cleaned = smiles.strip() if isinstance(smiles, str) else ""
    if not cleaned:
        return ""
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return cleaned
    try:
        scaf = str(_MURCKO.MurckoScaffoldSmiles(mol=mol, includeChirality=bool(include_chirality)))
    except Exception:
        scaf = ""
    if scaf:
        return scaf
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=bool(include_chirality)))


def _mol_physchem_12d(mol: Chem.Mol) -> NDArray[np.float64]:
    """Compute the 12 standardized RDKit 2D physicochemical descriptors for one molecule."""
    vals = np.array(
        [
            float(_DESCRIPTORS.MolWt(mol)),
            float(_CRIPPEN.MolLogP(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(Chem.GetFormalCharge(mol)),
            float(estimate_most_basic_pka(mol)),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


@lru_cache(maxsize=65536)
def _featurize_single(
    cleaned_smi: str,
    kind: FeaturizerKind,
    n_bits: int,
    radius: int,
    use_chirality: bool,
) -> tuple[bool, bytes]:
    """Cached single-molecule featurizer returning ``(is_valid, float32_bytes)``."""
    mol = Chem.MolFromSmiles(cleaned_smi)
    if mol is None:
        return False, b""
    gen = _get_morgan_generator(radius, n_bits, use_chirality)
    if kind == "ecfp4_binary":
        arr = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
        return True, arr.tobytes()
    if kind == "ecfp4_count":
        arr = np.asarray(gen.GetCountFingerprintAsNumPy(mol), dtype=np.float32)
        return True, arr.tobytes()
    if kind == "ecfp4_physchem":
        fp = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
        desc = _mol_physchem_12d(mol).astype(np.float32)
        arr = np.concatenate([fp, desc], axis=0)
        return True, arr.tobytes()
    raise ValueError(f"Unsupported featurizer kind {kind!r}.")


def clear_featurizer_cache() -> None:
    """Clear all internal SMILES canonicalization, scaffold, and fingerprint caches."""
    canonicalize_smiles.cache_clear()
    compute_murcko_scaffold_cached.cache_clear()
    _featurize_single.cache_clear()


def featurize_pool(
    smiles: Sequence[str],
    *,
    kind: FeaturizerKind = "ecfp4_binary",
    n_bits: int = 2048,
    radius: int = 2,
    use_chirality: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Featurize a sequence of SMILES into ``(X [N, D], valid_mask [N])`` in ``float64``.

    Args:
        smiles: Input SMILES strings. Invalid or empty entries yield zero rows with
            ``valid_mask[i] = False``.
        kind: Representation type: ``"ecfp4_binary"`` (default, ``0/1`` Morgan bits),
            ``"ecfp4_count"`` (non-negative Morgan substructure counts for MinMax GP),
            or ``"ecfp4_physchem"`` (binary ECFP4 concatenated with 12 RDKit descriptors).
        n_bits: Morgan fingerprint vector length (default ``2048``).
        radius: Morgan neighborhood radius (default ``2`` = ECFP4).
        use_chirality: Whether to encode stereochemical chirality into Morgan invariants.

    Returns:
        Tuple ``(X, valid)`` where ``X`` has shape ``[N, D]`` (``float64``) and ``valid``
        has shape ``[N]`` (``bool``).
    """
    n = len(smiles)
    bits = int(n_bits)
    rad = int(radius)
    dim = bits + _PHYSCHEM_DIM if kind == "ecfp4_physchem" else bits
    mat = np.zeros((n, dim), dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return mat, valid

    for idx, raw in enumerate(smiles):
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip()
        if not cleaned:
            continue
        ok, payload = _featurize_single(cleaned, kind, bits, rad, bool(use_chirality))
        if ok:
            mat[idx] = np.frombuffer(payload, dtype=np.float32).astype(np.float64, copy=False)
            valid[idx] = True
    return mat, valid


def tanimoto_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute the exact PSD Tanimoto kernel matrix :math:`K(X, Y)` in ``float64``.

    For non-negative feature vectors :math:`x, y \\in \\mathbb{R}_{\\ge 0}^d`:

    .. math::
        k_{\\text{Tan}}(x, y) = \\frac{x^\\top y}{\\|x\\|_2^2 + \\|y\\|_2^2 - x^\\top y}

    with exact zero-vector boundary rule :math:`k(\\mathbf{0}, \\mathbf{0}) = 1` and
    :math:`k(\\mathbf{0}, y) = 0` for :math:`y \\neq \\mathbf{0}`.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {x_arr.shape}.")
    symmetric = Y is None
    y_arr = x_arr if symmetric else np.asarray(Y, dtype=np.float64)
    if y_arr.ndim != 2 or y_arr.shape[1] != x_arr.shape[1]:
        raise ValueError(
            f"Incompatible shapes for tanimoto_kernel: X={x_arr.shape}, Y={y_arr.shape}."
        )
    if x_arr.shape[0] == 0 or y_arr.shape[0] == 0:
        return np.zeros((x_arr.shape[0], y_arr.shape[0]), dtype=np.float64)

    nx = np.einsum("ij,ij->i", x_arr, x_arr)
    ny = nx if symmetric else np.einsum("ij,ij->i", y_arr, y_arr)
    dot = x_arr @ y_arr.T
    denom = nx[:, None] + ny[None, :] - dot

    k_mat = np.zeros_like(dot, dtype=np.float64)
    pos = denom > 0.0
    np.divide(dot, denom, out=k_mat, where=pos)

    # Exact zero-vector boundary: k(0, 0) = 1.0, k(0, non-zero) = 0.0
    both_zero = (nx[:, None] == 0.0) & (ny[None, :] == 0.0)
    k_mat[both_zero] = 1.0
    np.clip(k_mat, 0.0, 1.0, out=k_mat)

    if symmetric:
        k_mat = 0.5 * (k_mat + k_mat.T)
        np.fill_diagonal(k_mat, 1.0)
    return k_mat


def minmax_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    chunk_size: int = 64,
) -> NDArray[np.float64]:
    """Compute the exact PSD continuous Ruzicka / MinMax kernel matrix in ``float64``.

    For non-negative count/feature vectors :math:`x, y \\in \\mathbb{R}_{\\ge 0}^d`:

    .. math::
        k_{\\text{MM}}(x, y) = \\frac{\\sum_{j=1}^d \\min(x_j, y_j)}{\\sum_{j=1}^d \\max(x_j, y_j)}

    Uses active-column sparse slicing inside bounded row chunks so large count-fingerprint
    matrices remain fast and memory-bounded while remaining exact to ``float64`` precision.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {x_arr.shape}.")
    symmetric = Y is None
    y_arr = x_arr if symmetric else np.asarray(Y, dtype=np.float64)
    if y_arr.ndim != 2 or y_arr.shape[1] != x_arr.shape[1]:
        raise ValueError(
            f"Incompatible shapes for minmax_kernel: X={x_arr.shape}, Y={y_arr.shape}."
        )
    nx_rows, ny_rows = x_arr.shape[0], y_arr.shape[0]
    if nx_rows == 0 or ny_rows == 0:
        return np.zeros((nx_rows, ny_rows), dtype=np.float64)

    if float(np.min(x_arr)) < -1e-12 or (not symmetric and float(np.min(y_arr)) < -1e-12):
        raise ValueError("minmax_kernel requires non-negative feature matrices.")
    x_pos = np.maximum(x_arr, 0.0)
    y_pos = x_pos if symmetric else np.maximum(y_arr, 0.0)

    sx = np.sum(x_pos, axis=1)
    sy = sx if symmetric else np.sum(y_pos, axis=1)
    num = np.zeros((nx_rows, ny_rows), dtype=np.float64)

    step = max(1, int(chunk_size))
    for i0 in range(0, nx_rows, step):
        i1 = min(nx_rows, i0 + step)
        xb = x_pos[i0:i1]
        x_active = np.any(xb > 0.0, axis=0)
        if not bool(np.any(x_active)):
            continue
        j_start = i0 if symmetric else 0
        for j0 in range(j_start, ny_rows, step):
            j1 = min(ny_rows, j0 + step)
            yb = y_pos[j0:j1]
            # Only columns positive in BOTH blocks can contribute to min(x, y) > 0!
            shared_cols = x_active & np.any(yb > 0.0, axis=0)
            if not bool(np.any(shared_cols)):
                continue
            block = np.minimum(xb[:, None, shared_cols], yb[None, :, shared_cols]).sum(axis=2)
            num[i0:i1, j0:j1] = block
            if symmetric and j0 > i0:
                num[j0:j1, i0:i1] = block.T

    denom = sx[:, None] + sy[None, :] - num
    k_mat = np.zeros((nx_rows, ny_rows), dtype=np.float64)
    pos = denom > 0.0
    np.divide(num, denom, out=k_mat, where=pos)

    both_zero = (sx[:, None] == 0.0) & (sy[None, :] == 0.0)
    k_mat[both_zero] = 1.0
    np.clip(k_mat, 0.0, 1.0, out=k_mat)

    if symmetric:
        k_mat = 0.5 * (k_mat + k_mat.T)
        np.fill_diagonal(k_mat, 1.0)
    return k_mat


def tanimoto_max_similarity(
    X_query: NDArray[np.float64],
    X_ref: NDArray[np.float64],
    *,
    chunk_size: int = 512,
) -> NDArray[np.float64]:
    """Return the maximum Tanimoto similarity from each query row to ``X_ref``."""
    n_q = int(X_query.shape[0])
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)
    if int(X_ref.shape[0]) == 0:
        return np.zeros(n_q, dtype=np.float64)
    out = np.zeros(n_q, dtype=np.float64)
    step = max(1, int(chunk_size))
    for i0 in range(0, n_q, step):
        i1 = min(n_q, i0 + step)
        k_block = tanimoto_kernel(X_query[i0:i1], X_ref)
        out[i0:i1] = np.max(k_block, axis=1)
    return out


# Spec-friendly aliases
tanimoto_kernel_matrix = tanimoto_kernel
minmax_kernel_matrix = minmax_kernel

__all__ = [
    "FeaturizerKind",
    "canonicalize_smiles",
    "clear_featurizer_cache",
    "compute_murcko_scaffold_cached",
    "featurize_pool",
    "minmax_kernel",
    "minmax_kernel_matrix",
    "tanimoto_kernel",
    "tanimoto_kernel_matrix",
    "tanimoto_max_similarity",
]
