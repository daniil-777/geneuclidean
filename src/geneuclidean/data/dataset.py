"""Preprocessed dataset cache builder, PyTorch Datasets, and batch collation."""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from geneuclidean.data.featurizer import (
    LIGAND_FP_DIM,
    LIGAND_PROP_NAMES,
    POCKET_NODE_DIM,
    extract_pocket_from_files,
)
from geneuclidean.data.splits import build_all_splits
from geneuclidean.data.tokenizer import PAD_ID, MolecularTokenizer, canonicalize_smiles


def _process_single_complex(
    task: tuple[str, str, str | None, str | None, str | None, float, float, int],
) -> dict[str, Any] | None:
    """Worker-safe single-complex featurization routine for multiprocessing cache builds."""
    pdb_id, pocket_pdb, ligand_sdf, ligand_mol2, fallback_smiles, pkd, pocket_cutoff, max_atoms = (
        task
    )
    try:
        feat = extract_pocket_from_files(
            pocket_pdb=pocket_pdb,
            ligand_sdf=ligand_sdf,
            ligand_mol2=ligand_mol2,
            fallback_smiles=fallback_smiles,
            pdb_id=pdb_id,
            pkd=pkd,
            pocket_cutoff=pocket_cutoff,
            max_atoms=max_atoms,
        )
    except Exception:
        return None

    if feat.canonical_smiles is None or feat.ligand_fp is None or feat.ligand_props is None:
        return None

    # Pack 1024 binary ECFP4 bits into 128 uint8 bytes (32x smaller on disk, zero precision loss)
    fp_bits = np.packbits((feat.ligand_fp > 0.5).astype(np.uint8))
    return {
        "pdb_id": feat.pdb_id,
        "coords": feat.coords.astype(np.float32),
        "node_feats": feat.node_feats.astype(np.float32),
        "contact_mask": feat.contact_mask.astype(np.bool_),
        "center": feat.center.astype(np.float32),
        "canonical_smiles": feat.canonical_smiles,
        "ligand_fp_packed": fp_bits,
        "ligand_props": feat.ligand_props.astype(np.float32),
        "pkd": float(feat.pkd),
    }


def _clean_zinc250k_to_txt(zinc_csv: Path, out_txt: Path) -> list[str]:
    """Strip newlines, drop invalid/empty rows, and save clean ZINC-250k SMILES."""
    df = pd.read_csv(zinc_csv)
    col = "smiles" if "smiles" in df.columns else str(df.columns[0])
    raw_smiles = df[col].dropna().astype(str).str.strip().tolist()
    seen: set[str] = set()
    clean_list: list[str] = []
    for smi in raw_smiles:
        if smi and smi not in seen:
            seen.add(smi)
            clean_list.append(smi)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(clean_list) + "\n", encoding="utf-8")
    return clean_list


def build_processed_cache(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    metadata_dir: str | Path = "data/metadata",
    *,
    pocket_cutoff: float = 10.0,
    max_atoms: int = 384,
    seed: int = 42,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Build compact processed caches, leak-proof splits, and JSON vocabularies.

    Produces:
      - ``<processed_dir>/pdbbind_pockets.pt``: Compact tensor/array cache for all parsed
        PDBbind v2020 refined + extra CASF-2016 complexes, plus train-only ``prop_mean``
        and ``prop_std`` standardization vectors.
      - ``<processed_dir>/splits.json``: Verified ``lp_pdbbind``, ``casf2016``, ``scaffold``,
        and ``random`` splits plus ``casf2016_core_ids``.
      - ``<processed_dir>/zinc250k_smiles.txt``: Cleaned ZINC-250k SMILES (1 per line).
      - ``<processed_dir>/vocab_smiles.json`` & ``vocab_selfies.json``: Vocabularies fit on
        ZINC-250k union ``lp_pdbbind.train`` ligands (zero val/test leakage).
    """
    raw_root = Path(raw_dir)
    proc_root = Path(processed_dir)
    meta_root = Path(metadata_dir)
    proc_root.mkdir(parents=True, exist_ok=True)

    lp_csv = raw_root / "LP_PDBBind.csv"
    casf_csv = meta_root / "casf_2016_core_labels.csv"
    ref19_csv = meta_root / "pdbbind_v2019_refined_labels.csv"

    # Assemble authoritative pKd and fallback SMILES lookup tables
    pkd_map: dict[str, float] = {}
    fallback_smiles_map: dict[str, str] = {}

    if ref19_csv.exists():
        ref_df = pd.read_csv(ref19_csv)
        for row in ref_df.itertuples(index=False):
            pkd_map[str(row.pdb_id).strip().lower()] = float(row.pkd)

    if lp_csv.exists():
        lp_df = pd.read_csv(lp_csv, index_col=0)
        lp_df.index = lp_df.index.astype(str).str.strip().str.lower()
        for pid, row in lp_df.iterrows():
            pid_str = str(pid)
            val = row.get("value")
            if pd.notna(val):
                pkd_map[pid_str] = float(val)
            smi = row.get("smiles")
            if pd.notna(smi) and str(smi).strip():
                fallback_smiles_map[pid_str] = str(smi).strip()

    if casf_csv.exists():
        casf_df = pd.read_csv(casf_csv)
        for row in casf_df.itertuples(index=False):
            pkd_map[str(row.pdb_id).strip().lower()] = float(row.pkd)

    # Discover complexes across pbpp2020 and casf2016_extra (deduplicated by PDB ID)
    complex_dirs: dict[str, Path] = {}
    for subdir_name in ("pbpp2020", "casf2016_extra"):
        base = raw_root / subdir_name
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                pid = entry.name.strip().lower()
                if pid not in complex_dirs:
                    complex_dirs[pid] = entry

    tasks: list[tuple[str, str, str | None, str | None, str | None, float, float, int]] = []
    for pid, cdir in sorted(complex_dirs.items()):
        pocket_pdb = cdir / f"{pid}_pocket.pdb"
        if not pocket_pdb.exists():
            continue
        sdf = cdir / f"{pid}_ligand.sdf"
        mol2 = cdir / f"{pid}_ligand.mol2"
        tasks.append(
            (
                pid,
                str(pocket_pdb),
                str(sdf) if sdf.exists() else None,
                str(mol2) if mol2.exists() else None,
                fallback_smiles_map.get(pid),
                pkd_map.get(pid, float("nan")),
                float(pocket_cutoff),
                int(max_atoms),
            )
        )

    n_workers = max_workers if max_workers is not None else min(8, os.cpu_count() or 1)
    records: dict[str, dict[str, Any]] = {}
    failed_count = 0

    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    can_spawn = main_file not in (None, "<stdin>")
    if n_workers > 1 and len(tasks) > 32 and can_spawn:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for item in pool.map(_process_single_complex, tasks, chunksize=32):
                if item is None:
                    failed_count += 1
                else:
                    records[str(item["pdb_id"])] = item
    else:
        for task in tasks:
            item = _process_single_complex(task)
            if item is None:
                failed_count += 1
            else:
                records[str(item["pdb_id"])] = item

    smiles_map = {pid: str(rec["canonical_smiles"]) for pid, rec in records.items()}
    splits_payload = build_all_splits(
        records.keys(),
        smiles_map,
        lp_csv_path=lp_csv,
        casf2016_csv_path=casf_csv,
        seed=seed,
    )
    splits_path = proc_root / "splits.json"
    splits_path.write_text(json.dumps(splits_payload, indent=2) + "\n", encoding="utf-8")

    # Compute train-only ligand property standardization stats (strictly on lp_pdbbind.train)
    lp_split = splits_payload["lp_pdbbind"]
    assert isinstance(lp_split, dict)
    train_ids = lp_split["train"] if lp_split["train"] else sorted(records.keys())
    train_props = np.stack(
        [records[pid]["ligand_props"] for pid in train_ids if pid in records],
        axis=0,
    ).astype(np.float32)
    prop_mean = train_props.mean(axis=0).astype(np.float32)
    prop_std = np.maximum(train_props.std(axis=0), 1e-6).astype(np.float32)

    cache_payload: dict[str, Any] = {
        "version": "2.0.0",
        "pocket_cutoff": float(pocket_cutoff),
        "max_atoms": int(max_atoms),
        "node_dim": POCKET_NODE_DIM,
        "fp_dim": LIGAND_FP_DIM,
        "prop_names": list(LIGAND_PROP_NAMES),
        "prop_mean": prop_mean,
        "prop_std": prop_std,
        "records": records,
    }
    cache_path = proc_root / "pdbbind_pockets.pt"
    torch.save(cache_payload, cache_path)

    # Clean ZINC-250k SMILES and fit vocabularies on ZINC-250k + lp_pdbbind.train ONLY
    zinc_smiles: list[str] = []
    zinc_csv = raw_root / "zinc250k.csv"
    zinc_txt = proc_root / "zinc250k_smiles.txt"
    if zinc_csv.exists():
        zinc_smiles = _clean_zinc250k_to_txt(zinc_csv, zinc_txt)

    train_ligand_smiles = [smiles_map[pid] for pid in train_ids if pid in smiles_map]
    vocab_corpus = zinc_smiles + train_ligand_smiles

    # Fit SMILES tokenizer (include 1 randomized pass over PDBbind train ligands + subset of ZINC)
    smiles_tok = MolecularTokenizer("smiles").fit(
        vocab_corpus,
        min_freq=1,
        randomized_passes=0,
        seed=seed,
    )
    # Scan 2 randomized passes on train ligands + 25k ZINC sample to cover extra tokens
    rng_aug = random.Random(seed)
    aug_sample = list(train_ligand_smiles) + (
        rng_aug.sample(zinc_smiles, min(25000, len(zinc_smiles))) if zinc_smiles else []
    )
    extra_tok = MolecularTokenizer("smiles").fit(
        aug_sample,
        min_freq=1,
        randomized_passes=2,
        seed=seed,
    )
    merged_smiles_vocab = list(smiles_tok.tokens)
    existing_set = set(merged_smiles_vocab)
    for tok in extra_tok.tokens:
        if tok not in existing_set:
            merged_smiles_vocab.append(tok)
            existing_set.add(tok)
    smiles_tok = MolecularTokenizer("smiles", vocab=merged_smiles_vocab)
    smiles_vocab_path = smiles_tok.save(proc_root / "vocab_smiles.json")

    selfies_tok = MolecularTokenizer("selfies").fit(
        vocab_corpus,
        min_freq=1,
        randomized_passes=0,
        seed=seed,
    )
    selfies_vocab_path = selfies_tok.save(proc_root / "vocab_selfies.json")

    return {
        "n_discovered": len(tasks),
        "n_parsed": len(records),
        "n_failed": failed_count,
        "n_zinc": len(zinc_smiles),
        "smiles_vocab_size": smiles_tok.vocab_size,
        "selfies_vocab_size": selfies_tok.vocab_size,
        "splits": {
            name: ({k: len(v) for k, v in val.items()} if isinstance(val, dict) else len(val))
            for name, val in splits_payload.items()
        },
        "cache_path": str(cache_path),
        "splits_path": str(splits_path),
        "smiles_vocab_path": str(smiles_vocab_path),
        "selfies_vocab_path": str(selfies_vocab_path),
        "zinc_txt_path": str(zinc_txt) if zinc_csv.exists() else None,
        "prop_mean": prop_mean.tolist(),
        "prop_std": prop_std.tolist(),
    }


class PocketLigandDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset of centered 3D protein pockets paired with drug-like ligands.

    Supports dynamic per-access Randomized SMILES augmentation, 3D Gaussian coordinate
    jitter, and peripheral (non-contact) atom dropout during training.
    """

    def __init__(
        self,
        cache_path: str | Path,
        splits_path: str | Path,
        tokenizer: MolecularTokenizer,
        *,
        split_type: str = "lp_pdbbind",
        split_name: str = "train",
        max_seq_len: int = 128,
        randomize_smiles: bool = False,
        coord_jitter_std: float = 0.0,
        atom_drop_prob: float = 0.0,
        standardize_props: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        payload = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        splits_data = json.loads(Path(splits_path).read_text(encoding="utf-8"))

        if split_name == "casf2016_core":
            ids = list(splits_data["casf2016_core_ids"])
        else:
            split_group = splits_data[split_type]
            ids = list(split_group[split_name])

        records: dict[str, dict[str, Any]] = payload["records"]
        self.pdb_ids: list[str] = [pid for pid in ids if pid in records]
        self._records: dict[str, dict[str, Any]] = records
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.is_train = split_name == "train"
        self.randomize_smiles = bool(randomize_smiles and self.is_train)
        self.coord_jitter_std = float(coord_jitter_std if self.is_train else 0.0)
        self.atom_drop_prob = float(atom_drop_prob if self.is_train else 0.0)
        self.standardize_props = bool(standardize_props)
        self.prop_mean = np.asarray(payload["prop_mean"], dtype=np.float32)
        self.prop_std = np.asarray(payload["prop_std"], dtype=np.float32)
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pid = self.pdb_ids[index]
        rec = self._records[pid]

        coords = np.array(rec["coords"], dtype=np.float32, copy=True)
        node_feats = np.array(rec["node_feats"], dtype=np.float32, copy=True)
        contact_mask = np.array(rec["contact_mask"], dtype=np.bool_, copy=True)

        # Peripheral-only atom dropout (always retain 4.5 A contact atoms and >= 16 atoms)
        if self.atom_drop_prob > 0.0 and len(coords) > 16:
            if self._rng is not None:
                rand_vals = self._rng.random(len(coords))
            else:
                rand_vals = np.random.random(len(coords))
            keep = contact_mask | (rand_vals >= self.atom_drop_prob)
            if int(keep.sum()) >= 16:
                coords = coords[keep]
                node_feats = node_feats[keep]
                contact_mask = contact_mask[keep]

        # Isotropic 3D Gaussian coordinate jitter during training
        if self.coord_jitter_std > 0.0:
            if self._rng is not None:
                noise = self._rng.normal(0.0, self.coord_jitter_std, size=coords.shape).astype(
                    np.float32
                )
            else:
                noise = np.random.normal(0.0, self.coord_jitter_std, size=coords.shape).astype(
                    np.float32
                )
            coords = coords + noise

        canonical_smi = str(rec["canonical_smiles"])
        full_ids = self.tokenizer.encode(
            canonical_smi,
            randomize=self.randomize_smiles,
            max_len=self.max_seq_len,
            rng=self._rng,
        )
        # Autoregressive teacher forcing: input = [BOS, t1, ..., tk], target = [t1, ..., tk, EOS]
        input_ids = np.asarray(full_ids[:-1], dtype=np.int64)
        target_ids = np.asarray(full_ids[1:], dtype=np.int64)

        fp_unpacked = np.unpackbits(rec["ligand_fp_packed"])[:LIGAND_FP_DIM].astype(np.float32)
        props = np.array(rec["ligand_props"], dtype=np.float32, copy=True)
        if self.standardize_props:
            props = ((props - self.prop_mean) / self.prop_std).astype(np.float32)

        return {
            "pdb_id": pid,
            "coords": torch.from_numpy(coords),
            "node_feats": torch.from_numpy(node_feats),
            "contact_mask": torch.from_numpy(contact_mask),
            "input_ids": torch.from_numpy(input_ids),
            "target_ids": torch.from_numpy(target_ids),
            "pkd": float(rec["pkd"]),
            "ligand_fp": torch.from_numpy(fp_unpacked),
            "ligand_props": torch.from_numpy(props),
            "canonical_smiles": canonical_smi,
        }


def collate_pocket_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-sized pockets and ligand token sequences into padded batch tensors.

    Returns the shared GenEuclidean v2 batch contract:
    ``pdb_ids, coords[B,N,3], node_feats[B,N,52], atom_mask[B,N], contact_mask[B,N],
    input_ids[B,L], target_ids[B,L], token_mask[B,L], pkd[B], ligand_fp[B,1024],
    ligand_props[B,6], canonical_smiles``.
    """
    batch_size = len(samples)
    max_atoms = max(int(s["coords"].shape[0]) for s in samples)
    max_seq = max(int(s["input_ids"].shape[0]) for s in samples)
    node_dim = int(samples[0]["node_feats"].shape[-1])
    fp_dim = int(samples[0]["ligand_fp"].shape[-1])
    prop_dim = int(samples[0]["ligand_props"].shape[-1])

    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    node_feats = torch.zeros((batch_size, max_atoms, node_dim), dtype=torch.float32)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    contact_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    input_ids = torch.full((batch_size, max_seq), PAD_ID, dtype=torch.long)
    target_ids = torch.full((batch_size, max_seq), PAD_ID, dtype=torch.long)
    token_mask = torch.zeros((batch_size, max_seq), dtype=torch.bool)

    pkd = torch.zeros((batch_size,), dtype=torch.float32)
    ligand_fp = torch.zeros((batch_size, fp_dim), dtype=torch.float32)
    ligand_props = torch.zeros((batch_size, prop_dim), dtype=torch.float32)
    pdb_ids: list[str] = []
    canonical_smiles: list[str] = []

    for i, item in enumerate(samples):
        n_a = int(item["coords"].shape[0])
        n_t = int(item["input_ids"].shape[0])
        coords[i, :n_a] = item["coords"]
        node_feats[i, :n_a] = item["node_feats"]
        atom_mask[i, :n_a] = True
        contact_mask[i, :n_a] = item["contact_mask"]

        input_ids[i, :n_t] = item["input_ids"]
        target_ids[i, :n_t] = item["target_ids"]
        token_mask[i, :n_t] = True

        pkd[i] = float(item["pkd"])
        ligand_fp[i] = item["ligand_fp"]
        ligand_props[i] = item["ligand_props"]
        pdb_ids.append(str(item["pdb_id"]))
        canonical_smiles.append(str(item["canonical_smiles"]))

    return {
        "pdb_ids": pdb_ids,
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "contact_mask": contact_mask,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "token_mask": token_mask,
        "pkd": pkd,
        "ligand_fp": ligand_fp,
        "ligand_props": ligand_props,
        "canonical_smiles": canonical_smiles,
    }


def load_zinc_splits(
    smiles_txt_path: str | Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    max_molecules: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load cleaned ZINC-250k SMILES and partition deterministically into train and val."""
    lines = [
        line.strip()
        for line in Path(smiles_txt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(seed)
    rng.shuffle(lines)
    if max_molecules is not None:
        lines = lines[:max_molecules]
    n_val = max(1, int(round(len(lines) * val_fraction)))
    return lines[n_val:], lines[:n_val]


class ZincSMILESDataset(Dataset[dict[str, torch.Tensor]]):
    """Unconditional drug-like SMILES/SELFIES/SAFE dataset for Stage-1 decoder pretraining."""

    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: MolecularTokenizer,
        *,
        max_seq_len: int = 128,
        randomize_smiles: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.smiles_list = [s.strip() for s in smiles_list if s and s.strip()]
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.randomize_smiles = bool(randomize_smiles)
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        smi = self.smiles_list[index]
        full_ids = self.tokenizer.encode(
            smi,
            randomize=self.randomize_smiles,
            max_len=self.max_seq_len,
            rng=self._rng,
        )
        return {
            "input_ids": torch.tensor(full_ids[:-1], dtype=torch.long),
            "target_ids": torch.tensor(full_ids[1:], dtype=torch.long),
        }


def collate_smiles_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate unconditional Stage-1 LM sequences into ``input_ids, target_ids, token_mask``."""
    batch_size = len(samples)
    max_seq = max(int(s["input_ids"].shape[0]) for s in samples)
    input_ids = torch.full((batch_size, max_seq), PAD_ID, dtype=torch.long)
    target_ids = torch.full((batch_size, max_seq), PAD_ID, dtype=torch.long)
    token_mask = torch.zeros((batch_size, max_seq), dtype=torch.bool)
    for i, item in enumerate(samples):
        n_t = int(item["input_ids"].shape[0])
        input_ids[i, :n_t] = item["input_ids"]
        target_ids[i, :n_t] = item["target_ids"]
        token_mask[i, :n_t] = True
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "token_mask": token_mask,
    }


__all__ = [
    "PocketLigandDataset",
    "ZincSMILESDataset",
    "build_processed_cache",
    "canonicalize_smiles",
    "collate_pocket_batch",
    "collate_smiles_batch",
    "load_zinc_splits",
]

if __name__ == "__main__":
    summary = build_processed_cache()
    print(json.dumps(summary, indent=2))
