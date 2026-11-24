"""Leak-proof dataset splits: LP-PDBBind, CASF-2016 core, Bemis-Murcko scaffold, and random."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

SplitDict = dict[str, list[str]]


def compute_murcko_scaffold(smiles: str) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES (falling back to SMILES if acyclic)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaf = str(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    except Exception:
        return smiles
    return scaf if scaf else smiles


def make_scaffold_split(
    pdb_ids: Iterable[str],
    smiles_map: Mapping[str, str],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> SplitDict:
    """Create a deterministic Bemis-Murcko scaffold split with zero shared scaffolds.

    Larger scaffold clusters are assigned deterministically to train first, followed by
    val and test, matching standard DeepChem / MoleculeNet scaffold-splitting practice
    while guaranteeing zero scaffold leakage across splits.
    """
    ids = sorted({pid.strip().lower() for pid in pdb_ids})
    groups: dict[str, list[str]] = defaultdict(list)
    for pid in ids:
        smi = smiles_map.get(pid, pid)
        scaf = compute_murcko_scaffold(smi)
        groups[scaf].append(pid)

    # Sort scaffold buckets by descending size, then deterministic seeded hash tie-breaker
    rng = random.Random(seed)
    tie_keys = {scaf: rng.random() for scaf in sorted(groups)}
    ordered_buckets = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), tie_keys[item[0]]),
    )

    n_total = len(ids)
    train_cap = int(round(train_frac * n_total))
    val_cap = int(round(val_frac * n_total))

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for _scaf, members in ordered_buckets:
        if len(train_ids) + len(members) <= train_cap or (not train_ids and train_frac > 0):
            train_ids.extend(members)
        elif len(val_ids) + len(members) <= val_cap or (not val_ids and val_frac > 0):
            val_ids.extend(members)
        else:
            test_ids.extend(members)

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def make_random_split(
    pdb_ids: Iterable[str],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> SplitDict:
    """Create a deterministic seeded train/val/test random split."""
    ids = sorted({pid.strip().lower() for pid in pdb_ids})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def make_casf2016_split(
    pdb_ids: Iterable[str],
    casf2016_ids: Iterable[str],
    smiles_map: Mapping[str, str],
    *,
    val_frac: float = 0.10,
    seed: int = 42,
) -> SplitDict:
    """Train on non-CASF refined complexes (90/10 scaffold split) and test on CASF-2016.

    Guarantees 0/285 CASF-2016 overlap with train and val.
    """
    available = {pid.strip().lower() for pid in pdb_ids}
    casf_set = {pid.strip().lower() for pid in casf2016_ids}
    test_ids = sorted(available & casf_set)
    remaining = sorted(available - casf_set)

    sub = make_scaffold_split(
        remaining,
        smiles_map,
        train_frac=1.0 - val_frac,
        val_frac=val_frac,
        seed=seed,
    )
    # Any tiny remainder in sub["test"] belongs in val (never leaks into CASF test)
    val_ids = sorted(set(sub["val"]) | set(sub["test"]))
    train_ids = sorted(sub["train"])
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def make_lp_pdbbind_split(
    pdb_ids: Iterable[str],
    lp_csv_path: str | Path,
    casf2016_ids: Iterable[str] | None = None,
    *,
    exclude_casf2016_from_train_val: bool = True,
) -> SplitDict:
    """Build the leak-proof LP-PDBBind split (Li et al., 2024) over available complexes.

    Uses ``new_split`` (``train``/``val``/``test``) from ``LP_PDBBind.csv``, optionally
    purging all CASF-2016 core PDB IDs from ``train`` and ``val`` so CASF-2016 remains a
    zero-leakage held-out evaluation benchmark alongside LP-PDBBind's test set.
    """
    available = {pid.strip().lower() for pid in pdb_ids}
    casf_set = {pid.strip().lower() for pid in (casf2016_ids or ())}

    df = pd.read_csv(lp_csv_path, index_col=0)
    df.index = df.index.astype(str).str.strip().str.lower()

    train_set: set[str] = set()
    val_set: set[str] = set()
    test_set: set[str] = set()

    for pid in sorted(available):
        if pid not in df.index:
            continue
        raw_split = df.loc[pid, "new_split"]
        if isinstance(raw_split, pd.Series):
            raw_split = raw_split.iloc[0]
        if pd.isna(raw_split):
            continue
        split_name = str(raw_split).strip().lower()
        if split_name == "train":
            train_set.add(pid)
        elif split_name == "val":
            val_set.add(pid)
        elif split_name == "test":
            test_set.add(pid)

    if exclude_casf2016_from_train_val and casf_set:
        train_set -= casf_set
        val_set -= casf_set

    return {
        "train": sorted(train_set),
        "val": sorted(val_set),
        "test": sorted(test_set),
    }


def verify_split_integrity(
    split: SplitDict,
    *,
    casf2016_ids: Iterable[str] | None = None,
    smiles_map: Mapping[str, str] | None = None,
    check_scaffold_disjoint: bool = False,
) -> dict[str, int]:
    """Assert zero leakage across splits and return exact partition sizes.

    Raises:
        AssertionError: If any complex ID overlaps across train/val/test, if CASF-2016
            contaminates train/val (when ``casf2016_ids`` is provided), or if scaffolds
            overlap (when ``check_scaffold_disjoint=True``).
    """
    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])

    if train_set & val_set:
        raise AssertionError(f"Train/Val overlap detected: {len(train_set & val_set)}")
    if train_set & test_set:
        raise AssertionError(f"Train/Test overlap detected: {len(train_set & test_set)}")
    if val_set & test_set:
        raise AssertionError(f"Val/Test overlap detected: {len(val_set & test_set)}")

    if casf2016_ids is not None:
        casf_set = {pid.strip().lower() for pid in casf2016_ids}
        leak_tr = train_set & casf_set
        leak_va = val_set & casf_set
        if leak_tr or leak_va:
            raise AssertionError(
                f"CASF-2016 leakage detected: train={len(leak_tr)}, val={len(leak_va)}"
            )

    if check_scaffold_disjoint and smiles_map is not None:
        scaf_tr = {compute_murcko_scaffold(smiles_map[p]) for p in train_set if p in smiles_map}
        scaf_va = {compute_murcko_scaffold(smiles_map[p]) for p in val_set if p in smiles_map}
        scaf_te = {compute_murcko_scaffold(smiles_map[p]) for p in test_set if p in smiles_map}
        if (scaf_tr & scaf_va) or (scaf_tr & scaf_te) or (scaf_va & scaf_te):
            raise AssertionError(
                "Shared Murcko scaffolds detected across train/val/test splits: "
                f"tr&va={len(scaf_tr & scaf_va)}, tr&te={len(scaf_tr & scaf_te)}, "
                f"va&te={len(scaf_va & scaf_te)}"
            )

    return {"train": len(train_set), "val": len(val_set), "test": len(test_set)}


def build_all_splits(
    pdb_ids: Iterable[str],
    smiles_map: Mapping[str, str],
    lp_csv_path: str | Path,
    casf2016_csv_path: str | Path,
    *,
    seed: int = 42,
) -> dict[str, SplitDict | list[str]]:
    """Construct and verify all four benchmark splits plus the CASF-2016 held-out ID list."""
    available = sorted({pid.strip().lower() for pid in pdb_ids})
    casf_df = pd.read_csv(casf2016_csv_path)
    casf_ids = sorted(casf_df["pdb_id"].astype(str).str.strip().str.lower().unique().tolist())

    lp_split = make_lp_pdbbind_split(
        available,
        lp_csv_path,
        casf2016_ids=casf_ids,
        exclude_casf2016_from_train_val=True,
    )
    verify_split_integrity(lp_split, casf2016_ids=casf_ids)

    casf_split = make_casf2016_split(available, casf_ids, smiles_map, val_frac=0.10, seed=seed)
    verify_split_integrity(casf_split, casf2016_ids=casf_ids)

    scaf_split = make_scaffold_split(
        available, smiles_map, train_frac=0.80, val_frac=0.10, seed=seed
    )
    verify_split_integrity(scaf_split, smiles_map=smiles_map, check_scaffold_disjoint=True)

    rand_split = make_random_split(available, train_frac=0.80, val_frac=0.10, seed=seed)
    verify_split_integrity(rand_split)

    return {
        "lp_pdbbind": lp_split,
        "casf2016": casf_split,
        "scaffold": scaf_split,
        "random": rand_split,
        "casf2016_core_ids": sorted(set(available) & set(casf_ids)),
    }


__all__ = [
    "SplitDict",
    "build_all_splits",
    "compute_murcko_scaffold",
    "make_casf2016_split",
    "make_lp_pdbbind_split",
    "make_random_split",
    "make_scaffold_split",
    "verify_split_integrity",
]
