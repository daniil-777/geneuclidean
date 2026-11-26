"""NanoFold3: Compact, MIT-Licensed, Apple-MPS-Optimized Mini-AlphaFold3 Co-Folder.

Combines GenEuclidean's ``PairformerStack`` / ``SimpleFoldTrunk``, ``AllAtomDiffusionModule``
(EDM & Rectified Flow), and AF3 Alg. 31 ``ConfidenceHead`` (+ Boltz-2-style ``AffinityHead``)
into one self-contained, trainable ``nn.Module`` with exact presets:

- ``nano``  (``~1.5M`` params): Fast CI, unit testing, and rapid prototyping.
- ``small`` (``~6.1M`` params): Flagship Apple M3 Pro MPS PDBbind / CNS induced-fit co-folding.
- ``base``  (``~24.2M`` params): High-capacity pocket & de novo mini-protein folding.

Exposes high-level ``.cofold(pocket_pdb, ligand_smiles, ...)`` and ``.fold_sequence(seq, ...)``
inference APIs returning :class:`NanoFold3Result` with strict 80-column PDB export (predicted
``pLDDT`` written into B-factor columns ``61–66``) and direct 7-axis
:class:`~geneuclidean.audit.pose_auditor.PoseRealityAuditor` integration.
"""

from __future__ import annotations

import hashlib
import io
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geneuclidean.folding.confidence import ConfidenceHead, ConfidenceLoss, ConfidenceOutput
from geneuclidean.folding.diffusion import (
    AllAtomDiffusionModule,
    CoFoldingLoss,
    CoFoldingLossOutput,
    DiffusionMode,
    FlexMode,
)
from geneuclidean.folding.pairformer import PairformerStack, SimpleFoldTrunk

PresetName = Literal["nano", "small", "base"]
TrunkMode = Literal["pairformer", "simplefold", "hybrid"]

_AA1_TO_AA3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
_STANDARD_RESIDUES: tuple[str, ...] = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
_RES_TO_TOKEN_TYPE: dict[str, int] = {res: idx for idx, res in enumerate(_STANDARD_RESIDUES)}
_LIGAND_TOKEN_TYPE: int = 21  # 0..19 standard AA, 20 other residue, 21 ligand atom


@dataclass(frozen=True)
class NanoFold3Config:
    """Configuration for :class:`NanoFold3` enforcing Apple MPS ``head_dim >= 32``."""

    preset: PresetName = "small"
    trunk_mode: TrunkMode = "pairformer"
    diffusion_mode: DiffusionMode = "edm"
    c_s: int = 192
    c_z: int = 64
    c_atom: int = 96
    c_token: int = 192
    head_dim: int = 32
    num_trunk_blocks: int = 6
    num_diff_layers: int = 4
    num_atom_enc_layers: int = 2
    num_atom_dec_layers: int = 2
    num_conf_blocks: int = 2
    transition_expansion: int = 2
    default_recycles: int = 2
    default_steps: int = 20
    atom_feat_dim: int = 52
    covalent_feat_dim: int = 6
    knn_cutoff: float = 8.0
    knn_max_neighbors: int = 24
    sigma_data: float = 16.0
    sigma_fm: float = 8.0
    enable_affinity_head: bool = True
    outer_product_mean: bool = False
    grad_checkpoint: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.head_dim < 32:
            raise ValueError(
                f"NanoFold3Config requires head_dim >= 32 for Apple MPS Flash-SDPA "
                f"(got {self.head_dim})."
            )
        if self.c_s % self.head_dim != 0:
            raise ValueError(f"c_s ({self.c_s}) must be divisible by head_dim ({self.head_dim}).")
        if self.c_z % self.head_dim != 0:
            raise ValueError(f"c_z ({self.c_z}) must be divisible by head_dim ({self.head_dim}).")
        if self.c_token % self.head_dim != 0:
            raise ValueError(
                f"c_token ({self.c_token}) must be divisible by head_dim ({self.head_dim})."
            )
        if self.trunk_mode not in ("pairformer", "simplefold", "hybrid"):
            raise ValueError(f"Unsupported trunk_mode {self.trunk_mode!r}")
        if self.diffusion_mode not in ("edm", "flow_matching"):
            raise ValueError(f"Unsupported diffusion_mode {self.diffusion_mode!r}")

    @classmethod
    def from_preset(
        cls,
        preset: PresetName = "small",
        *,
        trunk_mode: TrunkMode = "pairformer",
        diffusion_mode: DiffusionMode = "edm",
        enable_affinity_head: bool = True,
        **overrides: Any,
    ) -> NanoFold3Config:
        """Instantiate one of the three canonical ``nano`` / ``small`` / ``base`` configs."""
        presets: dict[PresetName, dict[str, Any]] = {
            "nano": {
                "preset": "nano",
                "c_s": 96,
                "c_z": 64,
                "c_atom": 64,
                "c_token": 128,
                "head_dim": 32,
                "num_trunk_blocks": 3,
                "num_diff_layers": 3,
                "num_atom_enc_layers": 1,
                "num_atom_dec_layers": 1,
                "num_conf_blocks": 1,
                "transition_expansion": 2,
                "default_recycles": 1,
                "default_steps": 12,
            },
            "small": {
                "preset": "small",
                "c_s": 192,
                "c_z": 64,
                "c_atom": 96,
                "c_token": 192,
                "head_dim": 32,
                "num_trunk_blocks": 6,
                "num_diff_layers": 4,
                "num_atom_enc_layers": 2,
                "num_atom_dec_layers": 2,
                "num_conf_blocks": 2,
                "transition_expansion": 2,
                "default_recycles": 2,
                "default_steps": 20,
            },
            "base": {
                "preset": "base",
                "c_s": 256,
                "c_z": 128,
                "c_atom": 128,
                "c_token": 256,
                "head_dim": 32,
                "num_trunk_blocks": 12,
                "num_diff_layers": 6,
                "num_atom_enc_layers": 2,
                "num_atom_dec_layers": 2,
                "num_conf_blocks": 3,
                "transition_expansion": 3,
                "default_recycles": 3,
                "default_steps": 32,
            },
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(presets)}.")
        cfg = dict(presets[preset])
        cfg["trunk_mode"] = trunk_mode
        cfg["diffusion_mode"] = diffusion_mode
        cfg["enable_affinity_head"] = enable_affinity_head
        cfg.update(overrides)
        return cls(**cfg)


@dataclass(frozen=True)
class ComplexBatch:
    """Batched mixed-token (residue + ligand heavy atom) representation for :class:`NanoFold3`."""

    coords: Tensor  # [B, N, 3] float32
    atom_feats: Tensor  # [B, N, 52] float32
    atom_to_token: Tensor  # [B, N] long
    atom_mask: Tensor  # [B, N] bool
    token_mask: Tensor  # [B, L] bool
    chain_id: Tensor  # [B, L] long (0 = pocket/protein, 1 = ligand)
    token_type: Tensor  # [B, L] long (0..20 residue, 21 ligand heavy atom)
    res_index: Tensor  # [B, L] long
    flex_mask: Tensor | None = None  # [B, N] bool
    is_ligand_atom: Tensor | None = None  # [B, N] bool
    is_backbone_atom: Tensor | None = None  # [B, N] bool
    token_rep_atom_idx: Tensor | None = None  # [B, L] long
    covalent_adj: Tensor | None = None  # [B, N, N, 6] float32
    bond_12_idx: Tensor | None = None  # [B, E12, 2] long
    bond_12_ref: Tensor | None = None  # [B, E12] float32
    bond_12_mask: Tensor | None = None  # [B, E12] bool
    bond_13_idx: Tensor | None = None  # [B, E13, 2] long
    bond_13_ref: Tensor | None = None  # [B, E13] float32
    bond_13_mask: Tensor | None = None  # [B, E13] bool
    chiral_quads: Tensor | None = None  # [B, C, 4] long
    chiral_vol_true: Tensor | None = None  # [B, C] float32
    chiral_mask: Tensor | None = None  # [B, C] bool
    pkd_true: Tensor | None = None  # [B] float32
    is_binder_true: Tensor | None = None  # [B] float32
    centroid_offset: Tensor | None = None  # [B, 3] float32

    def to(self, device: torch.device | str) -> ComplexBatch:
        """Move all non-None tensors in this batch to ``device``."""
        dev = torch.device(device)

        def _mv(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(dev)

        return ComplexBatch(
            coords=self.coords.to(dev),
            atom_feats=self.atom_feats.to(dev),
            atom_to_token=self.atom_to_token.to(dev),
            atom_mask=self.atom_mask.to(dev),
            token_mask=self.token_mask.to(dev),
            chain_id=self.chain_id.to(dev),
            token_type=self.token_type.to(dev),
            res_index=self.res_index.to(dev),
            flex_mask=_mv(self.flex_mask),
            is_ligand_atom=_mv(self.is_ligand_atom),
            is_backbone_atom=_mv(self.is_backbone_atom),
            token_rep_atom_idx=_mv(self.token_rep_atom_idx),
            covalent_adj=_mv(self.covalent_adj),
            bond_12_idx=_mv(self.bond_12_idx),
            bond_12_ref=_mv(self.bond_12_ref),
            bond_12_mask=_mv(self.bond_12_mask),
            bond_13_idx=_mv(self.bond_13_idx),
            bond_13_ref=_mv(self.bond_13_ref),
            bond_13_mask=_mv(self.bond_13_mask),
            chiral_quads=_mv(self.chiral_quads),
            chiral_vol_true=_mv(self.chiral_vol_true),
            chiral_mask=_mv(self.chiral_mask),
            pkd_true=_mv(self.pkd_true),
            is_binder_true=_mv(self.is_binder_true),
            centroid_offset=_mv(self.centroid_offset),
        )


@dataclass(frozen=True)
class NanoFold3TrainOutput:
    """Outputs and loss breakdown from :meth:`NanoFold3.forward_train`."""

    loss: Tensor
    cofolding_loss: CoFoldingLossOutput
    confidence_loss: Tensor
    x0_pred: Tensor
    confidence: ConfidenceOutput


@dataclass(frozen=True)
class NanoFold3SampleOutput:
    """Raw batched tensor outputs from :meth:`NanoFold3.sample`."""

    coords: Tensor  # [S, B, N, 3]
    confidence: ConfidenceOutput  # Evaluated on best-ranked sample [B, ...]
    best_sample_idx: Tensor  # [B] long
    ranking_scores: Tensor  # [S, B]


@dataclass(frozen=True)
class NanoFold3Result:
    """High-level inference result from :meth:`NanoFold3.cofold` or :meth:`NanoFold3.fold_sequence`.

    Provides direct 7-axis :class:`~geneuclidean.audit.pose_auditor.PoseRealityAuditor`
    verification via :meth:`run_audit` and matches the field surface consumed by
    ``geneuclidean.folding.runners.FoldedComplexResult``.
    """

    backend: str = "nanofold3"
    name: str = "nanofold3_complex"
    pdb_text: str = ""
    pocket_pdb_text: str | None = None
    ligand_mol_3d: Any | None = None
    ligand_smiles: str | None = None
    coords: Any | None = None
    plddt_per_atom: tuple[float, ...] = ()
    plddt_mean: float = float("nan")
    ligand_plddt: float = float("nan")
    pae_matrix: Any | None = None
    pde_matrix: Any | None = None
    ptm: float = float("nan")
    iptm: float = float("nan")
    ranking_score: float = float("nan")
    predicted_pkd: float = float("nan")
    binding_prob: float = float("nan")
    runtime_sec: float = 0.0
    provenance_sha256: str = ""
    pose_audit: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_plddt(self) -> float:
        """Alias for ``plddt_mean`` matching design spec naming."""
        return self.plddt_mean

    @property
    def audit_report(self) -> Any | None:
        """Alias for ``pose_audit`` matching design spec naming."""
        return self.pose_audit

    def run_audit(
        self,
        *,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
    ) -> Any:
        """Execute GenEuclidean's 7-axis :class:`PoseRealityAuditor` on this folded complex."""
        if self.ligand_mol_3d is None:
            raise ValueError(
                "PoseRealityAuditor requires a 3D ligand molecule (available from .cofold(), "
                "not protein-only .fold_sequence())."
            )
        from geneuclidean.audit.pose_auditor import PoseRealityAuditor

        auditor = PoseRealityAuditor(property_profile=property_profile)
        return auditor.audit(
            self.ligand_mol_3d,
            pocket_pdb=self.pocket_pdb_text,
            name=self.name,
            canonical_anchors=canonical_anchors,
        )


@dataclass(frozen=True)
class _AtomPDBRecord:
    """Internal atom metadata for faithful 80-col PDB serialization with pLDDT B-factors."""

    record_type: Literal["ATOM", "HETATM"]
    atom_name: str
    resname: str
    chain_id: str
    resseq: int
    element: str
    is_ligand: bool


def _format_pdb_lines(
    records: Sequence[_AtomPDBRecord],
    coords_world: np.ndarray,
    plddt_per_atom: np.ndarray,
    *,
    include_ligand: bool = True,
) -> str:
    """Format strict 80-column PDB lines writing per-atom ``pLDDT`` into B-factor cols 61–66."""
    lines: list[str] = []
    serial = 1
    for idx, rec in enumerate(records):
        if rec.is_ligand and not include_ligand:
            continue
        aname = rec.atom_name.strip()[:4]
        aname_fmt = f" {aname:<3s}" if len(aname) < 4 else aname
        rname = rec.resname.strip()[:3].rjust(3)
        cid = (rec.chain_id or "A")[:1]
        elem = rec.element.strip().upper()[:2]
        x, y, z = (
            float(coords_world[idx, 0]),
            float(coords_world[idx, 1]),
            float(coords_world[idx, 2]),
        )
        bfac = max(0.0, min(100.0, float(plddt_per_atom[idx])))
        rec_tag = f"{rec.record_type:<6s}"
        lines.append(
            f"{rec_tag}{serial:5d} {aname_fmt} {rname} {cid}{rec.resseq % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {elem:>2s}"
        )
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _parse_pocket_pdb_atoms(
    pocket_pdb: str | Path,
    *,
    pocket_center: Sequence[float] | None = None,
    pocket_cutoff: float = 10.0,
    max_pocket_atoms: int = 384,
) -> tuple[np.ndarray, np.ndarray, list[_AtomPDBRecord], list[int], list[int], np.ndarray]:
    """Parse pocket PDB path or text into centered coords, 52-D feats, records & residue map."""
    from Bio.PDB.PDBParser import PDBParser as _PDBParser

    from geneuclidean.data.featurizer import (
        _BACKBONE_ATOMS,
        _RESIDUE_ALIASES,
        _RESIDUE_TO_IDX,
        _WATER_RESNAMES,
        _compute_52d_node_features,
        _infer_element,
    )

    pdb_parser_cls: Any = _PDBParser
    parser: Any = pdb_parser_cls(QUIET=True)
    if isinstance(pocket_pdb, Path) or (
        isinstance(pocket_pdb, str) and "\n" not in pocket_pdb and "ATOM  " not in pocket_pdb
    ):
        src = Path(pocket_pdb)
        if not src.is_file():
            raise FileNotFoundError(f"Pocket PDB file not found: {src}")
        structure: Any = parser.get_structure(src.stem, str(src))
    else:
        structure = parser.get_structure("pocket", io.StringIO(str(pocket_pdb)))

    model = next(structure.get_models())
    raw_residues: list[
        tuple[list[np.ndarray], list[tuple[str, str, str]], list[_AtomPDBRecord], int]
    ] = []

    for chain in model:
        cid = (chain.id or "A")[:1]
        for res in chain:
            rname = res.get_resname().strip().upper()
            if rname in _WATER_RESNAMES:
                continue
            if res.id[0] != " ":
                continue
            res_coords: list[np.ndarray] = []
            res_meta: list[tuple[str, str, str]] = []
            res_recs: list[_AtomPDBRecord] = []
            res_norm = _RESIDUE_ALIASES.get(rname, rname)
            tok_type = _RESIDUE_TO_IDX.get(res_norm, 20)
            resseq = int(res.id[1])
            for atom in res.get_atoms():
                elem = _infer_element(atom)
                if elem == "H":
                    continue
                aname = str(atom.get_name()).strip().upper()
                coord = np.asarray(atom.get_coord(), dtype=np.float32).reshape(3)
                if not np.all(np.isfinite(coord)):
                    continue
                res_coords.append(coord)
                res_meta.append((elem, rname, aname))
                res_recs.append(
                    _AtomPDBRecord(
                        record_type="ATOM",
                        atom_name=aname,
                        resname=rname[:3],
                        chain_id=cid,
                        resseq=resseq,
                        element=elem,
                        is_ligand=False,
                    )
                )
            if res_coords:
                raw_residues.append((res_coords, res_meta, res_recs, tok_type))

    if not raw_residues:
        raise ValueError("No valid heavy protein atoms found in pocket_pdb.")

    all_coords = np.vstack([np.stack(r[0], axis=0) for r in raw_residues])
    center = (
        np.asarray(pocket_center, dtype=np.float32).reshape(3)
        if pocket_center is not None
        else all_coords.mean(axis=0).astype(np.float32)
    )

    # Sort residues by minimum distance to center and keep complete residues within cutoff/cap.
    scored_res: list[tuple[float, int]] = []
    for r_idx, (rc, _, _, _) in enumerate(raw_residues):
        rc_arr = np.stack(rc, axis=0)
        min_d = float(np.linalg.norm(rc_arr - center[None, :], axis=-1).min())
        scored_res.append((min_d, r_idx))
    scored_res.sort(key=lambda x: x[0])

    selected_indices: list[int] = []
    atom_count = 0
    for min_d, r_idx in scored_res:
        n_r = len(raw_residues[r_idx][0])
        if min_d > pocket_cutoff and selected_indices:
            continue
        if atom_count + n_r > max_pocket_atoms and selected_indices:
            continue
        selected_indices.append(r_idx)
        atom_count += n_r
    selected_indices.sort()

    coords_list: list[np.ndarray] = []
    meta_list: list[tuple[str, str, str]] = []
    records: list[_AtomPDBRecord] = []
    atom_to_tok: list[int] = []
    tok_types: list[int] = []
    rep_atom_indices: list[int] = []
    is_bb_list: list[bool] = []

    for tok_idx, r_idx in enumerate(selected_indices):
        rc, rm, rr, tt = raw_residues[r_idx]
        start_atom = len(coords_list)
        ca_offset = 0
        for local_i, (c, m, rec) in enumerate(zip(rc, rm, rr, strict=True)):
            coords_list.append(c - center)
            meta_list.append(m)
            records.append(rec)
            atom_to_tok.append(tok_idx)
            is_bb_list.append(rec.atom_name in _BACKBONE_ATOMS)
            if rec.atom_name == "CA":
                ca_offset = local_i
        tok_types.append(tt)
        rep_atom_indices.append(start_atom + ca_offset)

    centered_coords = np.stack(coords_list, axis=0).astype(np.float32)
    feats = _compute_52d_node_features(meta_list, centered_coords, pocket_cutoff)
    return centered_coords, feats, records, atom_to_tok, tok_types, center


def _extract_ligand_topology(
    mol_noh: Any,
    lig_offset: int,
    total_atoms: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract ligand covalent adjacency, 1–2 / 1–3 ideal distances, and chiral quads."""
    from rdkit import Chem

    cov_adj = np.zeros((total_atoms, total_atoms, 6), dtype=np.float32)
    conf = mol_noh.GetConformer()
    n_lig = mol_noh.GetNumAtoms()
    pos = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(n_lig)
        ],
        dtype=np.float32,
    )

    adj_list: list[list[int]] = [[] for _ in range(n_lig)]
    b12_pairs: list[tuple[int, int]] = []
    b12_dists: list[float] = []

    for bond in mol_noh.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        adj_list[i].append(j)
        adj_list[j].append(i)
        gi = lig_offset + i
        gj = lig_offset + j
        bt = bond.GetBondType()
        vec = np.zeros(6, dtype=np.float32)
        if bt == Chem.BondType.SINGLE:
            vec[0] = 1.0
        elif bt == Chem.BondType.DOUBLE:
            vec[1] = 1.0
        elif bt == Chem.BondType.TRIPLE:
            vec[2] = 1.0
        elif bt == Chem.BondType.AROMATIC:
            vec[3] = 1.0
        vec[4] = 1.0 if bond.GetIsConjugated() else 0.0
        vec[5] = 1.0 if bond.IsInRing() else 0.0
        cov_adj[gi, gj] = vec
        cov_adj[gj, gi] = vec
        d_ij = float(np.linalg.norm(pos[i] - pos[j]))
        b12_pairs.append((gi, gj))
        b12_dists.append(d_ij)

    # 1-3 geminal pairs (neighbors of a shared center atom)
    seen_13: set[tuple[int, int]] = set()
    b13_pairs: list[tuple[int, int]] = []
    b13_dists: list[float] = []
    for nbrs in adj_list:
        for a_i in range(len(nbrs)):
            for b_i in range(a_i + 1, len(nbrs)):
                u, v = min(nbrs[a_i], nbrs[b_i]), max(nbrs[a_i], nbrs[b_i])
                if (u, v) not in seen_13:
                    seen_13.add((u, v))
                    b13_pairs.append((lig_offset + u, lig_offset + v))
                    b13_dists.append(float(np.linalg.norm(pos[u] - pos[v])))

    # Chiral centers with 3 or 4 heavy neighbors (3 heavy neighbors -> quad = (center, n0, n1, n2))
    chiral_quads: list[tuple[int, int, int, int]] = []
    chem_mod: Any = Chem
    chiral_centers: list[tuple[int, str]] = chem_mod.FindMolChiralCenters(
        mol_noh, includeUnassigned=False
    )
    for atom_idx, _label in chiral_centers:
        nbrs = sorted(adj_list[int(atom_idx)])
        if len(nbrs) >= 4:
            chiral_quads.append(
                (
                    lig_offset + nbrs[0],
                    lig_offset + nbrs[1],
                    lig_offset + nbrs[2],
                    lig_offset + nbrs[3],
                )
            )
        elif len(nbrs) == 3:
            chiral_quads.append(
                (
                    lig_offset + int(atom_idx),
                    lig_offset + nbrs[0],
                    lig_offset + nbrs[1],
                    lig_offset + nbrs[2],
                )
            )

    b12_idx_arr = (
        np.array(b12_pairs, dtype=np.int64) if b12_pairs else np.zeros((0, 2), dtype=np.int64)
    )
    b12_ref_arr = (
        np.array(b12_dists, dtype=np.float32) if b12_dists else np.zeros((0,), dtype=np.float32)
    )
    b13_idx_arr = (
        np.array(b13_pairs, dtype=np.int64) if b13_pairs else np.zeros((0, 2), dtype=np.int64)
    )
    b13_ref_arr = (
        np.array(b13_dists, dtype=np.float32) if b13_dists else np.zeros((0,), dtype=np.float32)
    )
    chiral_arr = (
        np.array(chiral_quads, dtype=np.int64) if chiral_quads else np.zeros((0, 4), dtype=np.int64)
    )
    return cov_adj, b12_idx_arr, b12_ref_arr, b13_idx_arr, b13_ref_arr, chiral_arr


class ComplexTokenFeaturizer:
    """Convert pocket PDB + ligand SMILES/Mol (or sequence) into :class:`ComplexBatch`."""

    def __init__(self, *, max_pocket_atoms: int = 384, pocket_cutoff: float = 10.0) -> None:
        self.max_pocket_atoms = int(max_pocket_atoms)
        self.pocket_cutoff = float(pocket_cutoff)

    def featurize_complex(
        self,
        pocket_pdb: str | Path,
        ligand: str | Any,
        *,
        pocket_center: Sequence[float] | None = None,
        pocket_cutoff: float | None = None,
        protonate_ph74: bool = True,
        seed: int = 42,
    ) -> tuple[ComplexBatch, list[_AtomPDBRecord], Any, str]:
        """Build a 1-item :class:`ComplexBatch` plus PDB metadata and sanitized 3D template Mol."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from geneuclidean.models.encoders.ligand3d import featurize_ligand_3d

        cutoff = float(self.pocket_cutoff if pocket_cutoff is None else pocket_cutoff)
        poc_coords, poc_feats, poc_records, poc_a2t, poc_tok_types, center = (
            _parse_pocket_pdb_atoms(
                pocket_pdb,
                pocket_center=pocket_center,
                pocket_cutoff=cutoff,
                max_pocket_atoms=self.max_pocket_atoms,
            )
        )

        # Parse & optionally protonate ligand at physiological pH 7.4.
        if isinstance(ligand, str):
            smi_in = ligand.strip()
            if not smi_in:
                raise ValueError("ligand_smiles cannot be empty.")
            if protonate_ph74:
                from geneuclidean.evaluation.protonation import protonate_smiles_ph74

                smi_in = str(protonate_smiles_ph74(smi_in))
            mol = Chem.MolFromSmiles(smi_in)
            if mol is None:
                raise ValueError(f"Failed to parse ligand SMILES: {ligand!r}")
        else:
            mol = Chem.Mol(ligand)
            if protonate_ph74 and mol.GetNumConformers() == 0:
                from geneuclidean.evaluation.protonation import protonate_mol_ph74

                mol = protonate_mol_ph74(mol)

        # Ensure deterministic 3D reference conformer via ETKDGv3 if none is present.
        if mol.GetNumConformers() == 0:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
            params.randomSeed = int(seed)
            status = int(AllChem.EmbedMolecule(mol_h, params))  # type: ignore[attr-defined]
            if status != 0:
                status = int(AllChem.EmbedMolecule(mol_h, randomSeed=int(seed)))  # type: ignore[attr-defined]
            if status != 0:
                raise ValueError("RDKit failed to embed a 3D reference conformer for ligand.")
            mol_3d_noh = Chem.RemoveHs(mol_h)
        else:
            try:
                mol_3d_noh = Chem.RemoveHs(mol, sanitize=True)
            except Exception:
                mol_3d_noh = Chem.RemoveHs(mol, sanitize=False)

        can_smiles = str(Chem.MolToSmiles(mol_3d_noh, canonical=True, isomericSmiles=True))
        lig_coords_raw, lig_feats, _ = featurize_ligand_3d(
            mol_3d_noh,
            pocket_center=None,
            pocket_cutoff=cutoff,
            embed_if_missing=False,
            seed=seed,
            center_coords=True,
        )
        # Keep ligand centered at the pocket frame origin (0, 0, 0) as initial reference.
        lig_coords = lig_coords_raw.astype(np.float32)
        n_poc = poc_coords.shape[0]
        n_lig = lig_coords.shape[0]
        n_tot = n_poc + n_lig
        n_poc_toks = len(poc_tok_types)
        l_tot = n_poc_toks + n_lig

        # Build atom-level PDB records for ligand atoms.
        lig_records: list[_AtomPDBRecord] = []
        for idx, atom in enumerate(mol_3d_noh.GetAtoms()):
            elem = str(atom.GetSymbol()).strip().upper()
            lig_records.append(
                _AtomPDBRecord(
                    record_type="HETATM",
                    atom_name=f"{elem}{idx + 1}"[:4],
                    resname="LIG",
                    chain_id="Z",
                    resseq=1,
                    element=elem,
                    is_ligand=True,
                )
            )
        all_records = poc_records + lig_records

        coords_np = np.concatenate((poc_coords, lig_coords), axis=0)
        feats_np = np.concatenate((poc_feats, lig_feats), axis=0)
        a2t_np = np.array(poc_a2t + [n_poc_toks + i for i in range(n_lig)], dtype=np.int64)
        chain_np = np.array([0] * n_poc_toks + [1] * n_lig, dtype=np.int64)
        tok_type_np = np.array(poc_tok_types + [_LIGAND_TOKEN_TYPE] * n_lig, dtype=np.int64)
        res_idx_np = np.array(list(range(n_poc_toks)) + list(range(n_lig)), dtype=np.int64)

        is_lig_atom_np = np.array([False] * n_poc + [True] * n_lig, dtype=bool)
        is_bb_atom_np = np.array([feats_np[i, 42] > 0.5 for i in range(n_tot)], dtype=bool)

        # Representative atom index per token: first CA (or first atom) per residue, self for ligand
        rep_idx_list: list[int] = []
        for t in range(n_poc_toks):
            atom_indices = [i for i, a_t in enumerate(poc_a2t) if a_t == t]
            ca_match = [i for i in atom_indices if poc_records[i].atom_name == "CA"]
            rep_idx_list.append(ca_match[0] if ca_match else atom_indices[0])
        rep_idx_list.extend(n_poc + i for i in range(n_lig))
        rep_idx_np = np.array(rep_idx_list, dtype=np.int64)

        cov_adj, b12_idx, b12_ref, b13_idx, b13_ref, chiral_q = _extract_ligand_topology(
            mol_3d_noh, n_poc, n_tot
        )

        batch = ComplexBatch(
            coords=torch.from_numpy(coords_np).unsqueeze(0),
            atom_feats=torch.from_numpy(feats_np).unsqueeze(0),
            atom_to_token=torch.from_numpy(a2t_np).unsqueeze(0),
            atom_mask=torch.ones((1, n_tot), dtype=torch.bool),
            token_mask=torch.ones((1, l_tot), dtype=torch.bool),
            chain_id=torch.from_numpy(chain_np).unsqueeze(0),
            token_type=torch.from_numpy(tok_type_np).unsqueeze(0),
            res_index=torch.from_numpy(res_idx_np).unsqueeze(0),
            flex_mask=torch.from_numpy(~is_bb_atom_np).unsqueeze(0),
            is_ligand_atom=torch.from_numpy(is_lig_atom_np).unsqueeze(0),
            is_backbone_atom=torch.from_numpy(is_bb_atom_np).unsqueeze(0),
            token_rep_atom_idx=torch.from_numpy(rep_idx_np).unsqueeze(0),
            covalent_adj=torch.from_numpy(cov_adj).unsqueeze(0),
            bond_12_idx=torch.from_numpy(b12_idx).unsqueeze(0),
            bond_12_ref=torch.from_numpy(b12_ref).unsqueeze(0),
            bond_12_mask=torch.ones((1, b12_idx.shape[0]), dtype=torch.bool),
            bond_13_idx=torch.from_numpy(b13_idx).unsqueeze(0),
            bond_13_ref=torch.from_numpy(b13_ref).unsqueeze(0),
            bond_13_mask=torch.ones((1, b13_idx.shape[0]), dtype=torch.bool),
            chiral_quads=torch.from_numpy(chiral_q).unsqueeze(0),
            chiral_mask=torch.ones((1, chiral_q.shape[0]), dtype=torch.bool),
            centroid_offset=torch.from_numpy(center).unsqueeze(0),
        )
        return batch, all_records, mol_3d_noh, can_smiles

    def featurize_sequence(self, sequence: str) -> tuple[ComplexBatch, list[_AtomPDBRecord]]:
        """Build a 4-backbone-atom-per-residue ``(N, CA, C, O)`` :class:`ComplexBatch`."""
        from geneuclidean.data.featurizer import _compute_52d_node_features

        seq = sequence.strip().upper()
        if not seq:
            raise ValueError("Protein sequence cannot be empty.")
        invalid = sorted({ch for ch in seq if ch not in _AA1_TO_AA3})
        if invalid:
            raise ValueError(
                f"Protein sequence contains unsupported amino-acid characters: {invalid}"
            )

        # Idealized extended backbone reference coordinates (3.8 A CA-CA spacing).
        bb_offsets = (
            ("N", "N", np.array([-1.46, 0.0, 0.0], dtype=np.float32)),
            ("CA", "C", np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            ("C", "C", np.array([1.52, 0.0, 0.0], dtype=np.float32)),
            ("O", "O", np.array([1.52, 1.23, 0.0], dtype=np.float32)),
        )
        coords_list: list[np.ndarray] = []
        meta_list: list[tuple[str, str, str]] = []
        records: list[_AtomPDBRecord] = []
        a2t: list[int] = []
        tok_types: list[int] = []
        rep_idx: list[int] = []

        for r_idx, aa1 in enumerate(seq):
            res3 = _AA1_TO_AA3[aa1]
            tok_types.append(_RES_TO_TOKEN_TYPE[res3])
            ca_origin = np.array(
                [3.8 * r_idx, 0.5 * math.sin(r_idx), 0.5 * math.cos(r_idx)], dtype=np.float32
            )
            rep_idx.append(4 * r_idx + 1)  # CA atom index
            for aname, elem, off in bb_offsets:
                coords_list.append(ca_origin + off)
                meta_list.append((elem, res3, aname))
                records.append(
                    _AtomPDBRecord(
                        record_type="ATOM",
                        atom_name=aname,
                        resname=res3,
                        chain_id="A",
                        resseq=r_idx + 1,
                        element=elem,
                        is_ligand=False,
                    )
                )
                a2t.append(r_idx)

        coords_np = np.stack(coords_list, axis=0).astype(np.float32)
        coords_np = coords_np - coords_np.mean(axis=0, keepdims=True)
        feats_np = _compute_52d_node_features(
            meta_list, coords_np, pocket_cutoff=max(10.0, 1.5 * len(seq))
        )

        n_atoms = coords_np.shape[0]
        seq_len = len(seq)
        batch = ComplexBatch(
            coords=torch.from_numpy(coords_np).unsqueeze(0),
            atom_feats=torch.from_numpy(feats_np).unsqueeze(0),
            atom_to_token=torch.tensor(a2t, dtype=torch.long).unsqueeze(0),
            atom_mask=torch.ones((1, n_atoms), dtype=torch.bool),
            token_mask=torch.ones((1, seq_len), dtype=torch.bool),
            chain_id=torch.zeros((1, seq_len), dtype=torch.long),
            token_type=torch.tensor(tok_types, dtype=torch.long).unsqueeze(0),
            res_index=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            flex_mask=torch.ones((1, n_atoms), dtype=torch.bool),
            is_ligand_atom=torch.zeros((1, n_atoms), dtype=torch.bool),
            is_backbone_atom=torch.ones((1, n_atoms), dtype=torch.bool),
            token_rep_atom_idx=torch.tensor(rep_idx, dtype=torch.long).unsqueeze(0),
            centroid_offset=torch.zeros((1, 3), dtype=torch.float32),
        )
        return batch, records


def _gather_best_confidence(
    all_confs: Sequence[ConfidenceOutput],
    best_idx: Tensor,
) -> ConfidenceOutput:
    """Gather per-batch best-ranked sample tensors across :class:`ConfidenceOutput` fields."""
    batch_size = int(best_idx.shape[0])
    idx_b = best_idx.to(torch.long)

    def _pick(tensors: Sequence[Tensor]) -> Tensor:
        stacked = torch.stack(list(tensors), dim=0)  # [S, B, ...]
        extra_dims = stacked.ndim - 2
        view_shape = (1, batch_size, *([1] * extra_dims))
        exp_shape = (1, batch_size, *stacked.shape[2:])
        g_idx = idx_b.view(*view_shape).expand(*exp_shape)
        return torch.gather(stacked, dim=0, index=g_idx).squeeze(0)

    def _pick_opt(tensors: Sequence[Tensor | None]) -> Tensor | None:
        if tensors[0] is None:
            return None
        non_null = [t for t in tensors if t is not None]
        return _pick(non_null)

    return ConfidenceOutput(
        plddt_logits=_pick([c.plddt_logits for c in all_confs]),
        plddt=_pick([c.plddt for c in all_confs]),
        pae_logits=_pick([c.pae_logits for c in all_confs]),
        pae=_pick([c.pae for c in all_confs]),
        pde_logits=_pick([c.pde_logits for c in all_confs]),
        pde=_pick([c.pde for c in all_confs]),
        ptm=_pick([c.ptm for c in all_confs]),
        iptm=_pick([c.iptm for c in all_confs]),
        ranking_score=_pick([c.ranking_score for c in all_confs]),
        ligand_plddt=_pick([c.ligand_plddt for c in all_confs]),
        predicted_pkd=_pick_opt([c.predicted_pkd for c in all_confs]),
        binding_prob=_pick_opt([c.binding_prob for c in all_confs]),
        binder_logit=_pick_opt([c.binder_logit for c in all_confs]),
    )


class NanoFold3(nn.Module):
    """Compact, trainable Mini-AlphaFold3 for Induced-Fit Co-Folding & Single-Sequence Folding."""

    config: NanoFold3Config
    _relpos_max: int
    _init_dist_edges: Tensor

    def __init__(self, config: NanoFold3Config) -> None:
        super().__init__()
        self.config = config
        self._relpos_max = 32

        # 1. Input Token & Pair Embedder
        self.atom_pool_proj = nn.Linear(config.atom_feat_dim, config.c_s)
        self.token_type_emb = nn.Embedding(24, config.c_s)
        self.chain_emb = nn.Embedding(8, config.c_s)
        self.s_init_norm = nn.LayerNorm(config.c_s)

        self.z_left = nn.Linear(config.c_s, config.c_z, bias=False)
        self.z_right = nn.Linear(config.c_s, config.c_z, bias=False)
        self.relpos_emb = nn.Embedding(2 * self._relpos_max + 2, config.c_z)
        self.same_chain_emb = nn.Embedding(2, config.c_z)
        self.init_dist_bins = 24
        self.register_buffer(
            "_init_dist_edges",
            torch.linspace(1.5, 24.5, self.init_dist_bins, dtype=torch.float32),
        )
        self.init_dist_proj = nn.Linear(self.init_dist_bins, config.c_z, bias=False)

        # Recycling projections (AF3 Alg. 1)
        self.recycle_s_norm = nn.LayerNorm(config.c_s)
        self.recycle_z_norm = nn.LayerNorm(config.c_z)
        self.recycle_s_proj = nn.Linear(config.c_s, config.c_s, bias=False)
        self.recycle_z_proj = nn.Linear(config.c_z, config.c_z, bias=False)
        nn.init.normal_(self.recycle_s_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.recycle_z_proj.weight, mean=0.0, std=0.02)

        # 2. Structural Trunk (Pairformer O(L^3), SimpleFold O(L^2), or Hybrid)
        self.trunk: PairformerStack | SimpleFoldTrunk
        if config.trunk_mode == "pairformer":
            self.trunk = PairformerStack(
                num_blocks=config.num_trunk_blocks,
                c_s=config.c_s,
                c_z=config.c_z,
                head_dim=config.head_dim,
                transition_expansion=config.transition_expansion,
                outer_product_mean=config.outer_product_mean,
                dropout=config.dropout,
                grad_checkpoint=config.grad_checkpoint,
            )
        else:
            self.trunk = SimpleFoldTrunk(
                c_s=config.c_s,
                c_z=config.c_z,
                num_layers=config.num_trunk_blocks,
                head_dim=config.head_dim,
                trunk_mode=config.trunk_mode,
                use_rope=True,
                use_coulomb_bias=True,
                transition_expansion=config.transition_expansion,
                dropout=config.dropout,
                grad_checkpoint=config.grad_checkpoint,
            )

        # 3. All-Atom Diffusion / Flow-Matching Denoiser
        self.diffusion = AllAtomDiffusionModule(
            c_s=config.c_s,
            c_z=config.c_z,
            c_atom=config.c_atom,
            c_token=config.c_token,
            atom_feat_dim=config.atom_feat_dim,
            covalent_feat_dim=config.covalent_feat_dim,
            head_dim=config.head_dim,
            num_diff_layers=config.num_diff_layers,
            num_atom_enc_layers=config.num_atom_enc_layers,
            num_atom_dec_layers=config.num_atom_dec_layers,
            knn_cutoff=config.knn_cutoff,
            knn_max_neighbors=config.knn_max_neighbors,
            diffusion_mode=config.diffusion_mode,
            sigma_data=config.sigma_data,
            sigma_fm=config.sigma_fm,
            transition_expansion=config.transition_expansion,
            dropout=config.dropout,
        )

        # 4. AF3 Alg. 31 Confidence + Optional Boltz-2 Affinity Head
        self.confidence_head = ConfidenceHead(
            c_s=config.c_s,
            c_z=config.c_z,
            num_blocks=config.num_conf_blocks,
            head_dim=config.head_dim,
            trunk_mode=config.trunk_mode,
            enable_affinity_head=config.enable_affinity_head,
            transition_expansion=config.transition_expansion,
            dropout=config.dropout,
        )

        self.cofolding_loss = CoFoldingLoss()
        self.confidence_loss = ConfidenceLoss()
        self.featurizer = ComplexTokenFeaturizer()

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_preset(
        cls,
        preset: PresetName = "small",
        *,
        trunk_mode: TrunkMode = "pairformer",
        diffusion_mode: DiffusionMode = "edm",
        enable_affinity_head: bool = True,
        **overrides: Any,
    ) -> NanoFold3:
        """Build a :class:`NanoFold3` model from ``nano``, ``small``, or ``base`` preset."""
        cfg = NanoFold3Config.from_preset(
            preset,
            trunk_mode=trunk_mode,
            diffusion_mode=diffusion_mode,
            enable_affinity_head=enable_affinity_head,
            **overrides,
        )
        return cls(cfg)

    def save_pretrained(self, checkpoint_path: str | Path) -> Path:
        """Persist config dict + ``float32`` state_dict to ``checkpoint_path``."""
        dst = Path(checkpoint_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.config), "state_dict": self.state_dict()}
        torch.save(payload, dst)
        return dst

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "auto",
    ) -> NanoFold3:
        """Load a saved :class:`NanoFold3` checkpoint safely (``weights_only=True``)."""
        src = Path(checkpoint_path)
        if not src.is_file():
            raise FileNotFoundError(f"NanoFold3 checkpoint not found: {src}")
        dev = _resolve_device(device)
        payload = torch.load(src, map_location="cpu", weights_only=True)
        cfg = NanoFold3Config(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model.to(dev)

    def _extract_token_rep_coords(self, coords: Tensor, batch: ComplexBatch) -> Tensor:
        """Gather representative token coords ``[B, L, 3]`` (CA for residues, atom for ligand)."""
        num_tokens = batch.token_mask.shape[1]
        if batch.token_rep_atom_idx is not None:
            safe_idx = batch.token_rep_atom_idx.to(torch.long).clamp(
                min=0, max=max(coords.shape[1] - 1, 0)
            )
            idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, 3)
            rep = torch.gather(coords.to(torch.float32), dim=1, index=idx_exp)
        else:
            # Scatter-free mean coordinate per token via bmm.
            safe_tok = batch.atom_to_token.to(torch.long).clamp(min=0, max=max(num_tokens - 1, 0))
            m_tok = F.one_hot(safe_tok, num_classes=num_tokens).transpose(1, 2).to(torch.float32)
            m_tok = m_tok * batch.atom_mask.to(torch.float32).unsqueeze(1)
            m_norm = m_tok / m_tok.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rep = torch.bmm(m_norm, coords.to(torch.float32))
        return torch.where(
            batch.token_mask.to(torch.bool).unsqueeze(-1), rep, torch.zeros_like(rep)
        )

    def _resolve_batch_flex_mask(
        self,
        batch: ComplexBatch,
        flex_mode: FlexMode | None = None,
    ) -> Tensor:
        """Resolve per-atom flexibility mask cleanly for any ``flex_mode``."""
        m_a = batch.atom_mask.to(torch.bool)
        if flex_mode == "de_novo_all_atom":
            return m_a
        if flex_mode == "rigid_pocket_flexible_ligand":
            if batch.is_ligand_atom is not None:
                return m_a & batch.is_ligand_atom.to(torch.bool)
            return self.diffusion.resolve_flex_mask(m_a, batch.atom_feats, flex_mode, None)
        if batch.flex_mask is not None:
            return m_a & batch.flex_mask.to(torch.bool)
        mode_eff: FlexMode = "flexible_pocket_and_ligand" if flex_mode is None else flex_mode
        return self.diffusion.resolve_flex_mask(m_a, batch.atom_feats, mode_eff, None)

    def _embed_inputs(
        self,
        batch: ComplexBatch,
        *,
        flex_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute initial single ``s0``, pair ``z0``, and anchored-only rep coords."""
        num_tokens = batch.token_mask.shape[1]
        m_tok = batch.token_mask.to(torch.bool)
        pair_mask = m_tok.unsqueeze(2) & m_tok.unsqueeze(1)

        # Scatter-free atom -> token mean feature pooling via bmm.
        safe_tok = batch.atom_to_token.to(torch.long).clamp(min=0, max=max(num_tokens - 1, 0))
        m_mat = F.one_hot(safe_tok, num_classes=num_tokens).transpose(1, 2).to(torch.float32)
        m_mat = m_mat * batch.atom_mask.to(torch.float32).unsqueeze(1)
        m_norm = m_mat / m_mat.sum(dim=-1, keepdim=True).clamp(min=1.0)
        tok_atom_feats = torch.bmm(m_norm, batch.atom_feats.to(torch.float32))  # [B, L, 52]

        tok_t = batch.token_type.to(torch.long).clamp(0, 23)
        cid = batch.chain_id.to(torch.long).clamp(0, 7)
        s0 = self.s_init_norm(
            self.atom_pool_proj(tok_atom_feats) + self.token_type_emb(tok_t) + self.chain_emb(cid)
        )
        s0 = torch.where(m_tok.unsqueeze(-1), s0, torch.zeros_like(s0))

        # Relative sequence offset clipped to [-32, 32] (plus separate bin for cross-chain pairs).
        same_chain = cid.unsqueeze(2) == cid.unsqueeze(1)
        rel = (batch.res_index.unsqueeze(2) - batch.res_index.unsqueeze(1)).clamp(
            -self._relpos_max, self._relpos_max
        ) + self._relpos_max
        cross_bin = torch.full_like(rel, 2 * self._relpos_max + 1)
        rel_idx = torch.where(same_chain, rel, cross_bin)

        # Strictly isolate anchored (non-flexible) tokens so ground-truth flexible atom
        # coordinates NEVER leak into initial distograms or SimpleFold 3D RoPE / Coulomb bias!
        eff_flex = batch.flex_mask if flex_mask is None else flex_mask
        if eff_flex is not None:
            flex_b = eff_flex.to(torch.bool)
            if batch.token_rep_atom_idx is not None:
                safe_rep = batch.token_rep_atom_idx.to(torch.long).clamp(
                    0, max(flex_b.shape[1] - 1, 0)
                )
                tok_is_anchor = m_tok & (~torch.gather(flex_b, dim=1, index=safe_rep))
            else:
                has_flex = (
                    torch.bmm(m_mat, flex_b.to(torch.float32).unsqueeze(-1)).squeeze(-1) > 0.5
                )
                tok_is_anchor = m_tok & (~has_flex)
        else:
            tok_is_anchor = torch.zeros_like(m_tok)

        has_anchors = bool(tok_is_anchor.any().item())
        if has_anchors:
            anchor_atoms = batch.atom_mask.to(torch.bool) & (
                ~eff_flex.to(torch.bool)
                if eff_flex is not None
                else torch.zeros_like(batch.atom_mask, dtype=torch.bool)
            )
            coords_anchor_atoms = torch.where(
                anchor_atoms.unsqueeze(-1),
                batch.coords.to(torch.float32),
                torch.zeros_like(batch.coords, dtype=torch.float32),
            )
            x_rep_raw = self._extract_token_rep_coords(coords_anchor_atoms, batch)
            x_rep_masked = torch.where(
                tok_is_anchor.unsqueeze(-1), x_rep_raw, torch.zeros_like(x_rep_raw)
            )
            x_rep_anchored: Tensor | None = x_rep_masked
            dist_pair_mask = pair_mask & tok_is_anchor.unsqueeze(2) & tok_is_anchor.unsqueeze(1)
            diff = x_rep_masked.unsqueeze(2) - x_rep_masked.unsqueeze(1)
            dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
            edges = self._init_dist_edges.to(device=s0.device, dtype=torch.float32)
            delta = dist.unsqueeze(-1) - edges.view(1, 1, 1, self.init_dist_bins)
            dist_soft = torch.softmax(-0.5 * (delta * delta), dim=-1)
            dist_soft = torch.where(
                dist_pair_mask.unsqueeze(-1), dist_soft, torch.zeros_like(dist_soft)
            )
        else:
            x_rep_anchored = None
            dist_soft = torch.zeros(
                (s0.shape[0], num_tokens, num_tokens, self.init_dist_bins),
                device=s0.device,
                dtype=torch.float32,
            )

        z0 = (
            self.z_left(s0).unsqueeze(2)
            + self.z_right(s0).unsqueeze(1)
            + self.relpos_emb(rel_idx)
            + self.same_chain_emb(same_chain.to(torch.long))
            + self.init_dist_proj(dist_soft)
        )
        z0 = torch.where(pair_mask.unsqueeze(-1), z0, torch.zeros_like(z0))
        return s0, z0, x_rep_anchored

    def forward_trunk(
        self,
        batch: ComplexBatch,
        *,
        recycles: int | None = None,
        flex_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run input embedder + recycled trunk returning ``(s_trunk, z_trunk)``."""
        n_rec = max(1, int(self.config.default_recycles if recycles is None else recycles))
        s0, z0, x_rep_anchored = self._embed_inputs(batch, flex_mask=flex_mask)
        m_tok = batch.token_mask.to(torch.bool)

        s_prev = torch.zeros_like(s0)
        z_prev = torch.zeros_like(z0)
        s_cur = s0
        z_cur = z0

        for rec in range(n_rec):
            is_last = rec == (n_rec - 1)
            s_in = s0 + self.recycle_s_proj(self.recycle_s_norm(s_prev))
            z_in = z0 + self.recycle_z_proj(self.recycle_z_norm(z_prev))
            if is_last:
                if isinstance(self.trunk, SimpleFoldTrunk):
                    s_cur, z_cur = self.trunk(
                        s_in,
                        z_in,
                        m_tok,
                        positions=batch.res_index,
                        coords=x_rep_anchored,
                    )
                else:
                    s_cur, z_cur = self.trunk(s_in, z_in, m_tok)
            else:
                with torch.no_grad():
                    if isinstance(self.trunk, SimpleFoldTrunk):
                        s_cur, z_cur = self.trunk(
                            s_in,
                            z_in,
                            m_tok,
                            positions=batch.res_index,
                            coords=x_rep_anchored,
                        )
                    else:
                        s_cur, z_cur = self.trunk(s_in, z_in, m_tok)
                s_prev = s_cur.detach()
                z_prev = z_cur.detach()
        return s_cur, z_cur

    def forward_train(
        self,
        batch: ComplexBatch,
        *,
        flex_mode: FlexMode = "flexible_pocket_and_ligand",
        recycles: int | None = None,
        conf_loss_weight: float = 0.2,
        seed: int | None = None,
    ) -> NanoFold3TrainOutput:
        """Execute one end-to-end training step (trunk -> diffusion -> physics & conf losses)."""
        eff_flex = self._resolve_batch_flex_mask(batch, flex_mode)
        s_trunk, z_trunk = self.forward_trunk(batch, recycles=recycles, flex_mask=eff_flex)

        # Pass anchored-only reference coords (zeroed on flexible atoms; None in de_novo_all_atom)
        # so the denoiser never receives ground-truth flexible atom coordinates.
        if flex_mode == "de_novo_all_atom":
            r_ref_safe: Tensor | None = None
        else:
            anchor_mask = batch.atom_mask.to(torch.bool) & (~eff_flex)
            r_ref_safe = torch.where(
                anchor_mask.unsqueeze(-1),
                batch.coords.to(torch.float32),
                torch.zeros_like(batch.coords, dtype=torch.float32),
            )

        x0_pred, x0_target, _noise_level, loss_w = self.diffusion.forward_train(
            x0=batch.coords,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            atom_feats=batch.atom_feats,
            atom_to_token=batch.atom_to_token,
            atom_mask=batch.atom_mask,
            token_mask=batch.token_mask,
            flex_mask=eff_flex,
            r_ref=r_ref_safe,
            covalent_adj=batch.covalent_adj,
            flex_mode=flex_mode,
            seed=seed,
        )

        align_flag = flex_mode == "de_novo_all_atom"
        cofold_loss = self.cofolding_loss(
            x_pred=x0_pred,
            x_true=x0_target,
            atom_mask=batch.atom_mask,
            loss_weight=loss_w,
            align=align_flag,
            is_ligand=batch.is_ligand_atom,
            atom_feats=batch.atom_feats,
            bond_12_idx=batch.bond_12_idx,
            bond_12_ref=batch.bond_12_ref,
            bond_12_mask=batch.bond_12_mask,
            bond_13_idx=batch.bond_13_idx,
            bond_13_ref=batch.bond_13_ref,
            bond_13_mask=batch.bond_13_mask,
            chiral_quads=batch.chiral_quads,
            chiral_vol_true=batch.chiral_vol_true,
            chiral_mask=batch.chiral_mask,
            backbone_mask=batch.is_backbone_atom
            if flex_mode == "flexible_pocket_and_ligand"
            else None,
            x_input=batch.coords if flex_mode == "flexible_pocket_and_ligand" else None,
        )

        x_pred_rep = self._extract_token_rep_coords(x0_pred, batch)
        x_true_rep = self._extract_token_rep_coords(x0_target, batch)
        conf_out = self.confidence_head(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            x_rep=x_pred_rep,
            token_mask=batch.token_mask,
            chain_id=batch.chain_id,
            atom_to_token=batch.atom_to_token,
            atom_mask=batch.atom_mask,
            is_ligand_atom=batch.is_ligand_atom,
            detach_trunk=True,
        )
        conf_l = self.confidence_loss(
            conf=conf_out,
            x_pred_atoms=x0_pred,
            x_true_atoms=x0_target,
            atom_mask=batch.atom_mask,
            x_pred_rep=x_pred_rep,
            x_true_rep=x_true_rep,
            token_mask=batch.token_mask,
            pkd_true=batch.pkd_true,
            is_binder_true=batch.is_binder_true,
        )
        total_loss = cofold_loss.total + float(conf_loss_weight) * conf_l
        return NanoFold3TrainOutput(
            loss=total_loss,
            cofolding_loss=cofold_loss,
            confidence_loss=conf_l,
            x0_pred=x0_pred,
            confidence=conf_out,
        )

    @torch.inference_mode()
    def sample(
        self,
        batch: ComplexBatch,
        *,
        num_steps: int | None = None,
        num_samples: int = 1,
        recycles: int | None = None,
        flex_mode: FlexMode = "flexible_pocket_and_ligand",
        seed: int = 42,
    ) -> NanoFold3SampleOutput:
        """Deterministically sample ``num_samples`` poses and score with ``ConfidenceHead``."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        steps = int(self.config.default_steps if num_steps is None else num_steps)
        eff_flex = self._resolve_batch_flex_mask(batch, flex_mode)
        s_trunk, z_trunk = self.forward_trunk(batch, recycles=recycles, flex_mask=eff_flex)

        if flex_mode == "de_novo_all_atom":
            r_ref_safe: Tensor | None = None
        else:
            anchor_mask = batch.atom_mask.to(torch.bool) & (~eff_flex)
            r_ref_safe = torch.where(
                anchor_mask.unsqueeze(-1),
                batch.coords.to(torch.float32),
                torch.zeros_like(batch.coords, dtype=torch.float32),
            )

        all_coords: list[Tensor] = []
        all_confs: list[ConfidenceOutput] = []
        all_scores: list[Tensor] = []
        for s_idx in range(num_samples):
            coords_s = self.diffusion.sample(
                r_init=batch.coords,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                atom_feats=batch.atom_feats,
                atom_to_token=batch.atom_to_token,
                atom_mask=batch.atom_mask,
                token_mask=batch.token_mask,
                flex_mask=eff_flex,
                r_ref=r_ref_safe,
                covalent_adj=batch.covalent_adj,
                flex_mode=flex_mode,
                num_steps=steps,
                seed=int(seed) + s_idx * 1009,
            )
            rep_s = self._extract_token_rep_coords(coords_s, batch)
            conf_s = self.confidence_head(
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                x_rep=rep_s,
                token_mask=batch.token_mask,
                chain_id=batch.chain_id,
                atom_to_token=batch.atom_to_token,
                atom_mask=batch.atom_mask,
                is_ligand_atom=batch.is_ligand_atom,
                detach_trunk=True,
            )
            all_coords.append(coords_s)
            all_confs.append(conf_s)
            all_scores.append(conf_s.ranking_score)

        coords_stack = torch.stack(all_coords, dim=0)  # [S, B, N, 3]
        scores_stack = torch.stack(all_scores, dim=0)  # [S, B]
        best_idx = scores_stack.argmax(dim=0)  # [B]
        best_conf = _gather_best_confidence(all_confs, best_idx)

        return NanoFold3SampleOutput(
            coords=coords_stack,
            confidence=best_conf,
            best_sample_idx=best_idx,
            ranking_scores=scores_stack,
        )

    @torch.inference_mode()
    def cofold(
        self,
        pocket_pdb: str | Path,
        ligand_smiles: str | Any,
        *,
        pocket_center: Sequence[float] | None = None,
        pocket_cutoff: float = 10.0,
        flex_mode: FlexMode = "flexible_pocket_and_ligand",
        protonate_ph74: bool = True,
        num_samples: int = 1,
        num_steps: int | None = None,
        seed: int = 42,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        name: str = "nanofold3_complex",
    ) -> NanoFold3Result:
        """End-to-end induced-fit pocket–ligand co-folding returning :class:`NanoFold3Result`."""
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        t0 = time.perf_counter()
        device = next(self.parameters()).device
        batch_cpu, records, mol_template, can_smiles = self.featurizer.featurize_complex(
            pocket_pdb=pocket_pdb,
            ligand=ligand_smiles,
            pocket_center=pocket_center,
            pocket_cutoff=pocket_cutoff,
            protonate_ph74=protonate_ph74,
            seed=seed,
        )
        batch = batch_cpu.to(device)
        out = self.sample(
            batch,
            num_steps=num_steps,
            num_samples=num_samples,
            flex_mode=flex_mode,
            seed=seed,
        )
        best_s = int(out.best_sample_idx[0].item())
        coords_local = out.coords[best_s, 0].detach().cpu().numpy().astype(np.float32)
        offset = (
            batch_cpu.centroid_offset[0].numpy().astype(np.float32)
            if batch_cpu.centroid_offset is not None
            else np.zeros(3, dtype=np.float32)
        )
        coords_world = coords_local + offset[None, :]

        conf = out.confidence
        plddt_np = conf.plddt[0].detach().cpu().numpy().astype(np.float32)
        pae_np = conf.pae[0].detach().cpu().numpy().astype(np.float32)
        pde_np = conf.pde[0].detach().cpu().numpy().astype(np.float32)

        # Write denoised ligand heavy coordinates back onto a clean copy of the RDKit template Mol.
        lig_mol_3d = Chem.Mol(mol_template)
        n_lig = lig_mol_3d.GetNumAtoms()
        lig_start = len(records) - n_lig
        lig_mol_3d.RemoveAllConformers()
        new_conf = Chem.Conformer(n_lig)
        for i in range(n_lig):
            x, y, z = coords_world[lig_start + i]
            new_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        lig_mol_3d.AddConformer(new_conf, assignId=True)

        full_pdb_text = _format_pdb_lines(records, coords_world, plddt_np, include_ligand=True)
        pocket_pdb_text = _format_pdb_lines(records, coords_world, plddt_np, include_ligand=False)
        prov_hash = hashlib.sha256(full_pdb_text.encode("utf-8")).hexdigest()

        runtime_sec = time.perf_counter() - t0
        result = NanoFold3Result(
            backend="nanofold3",
            name=name,
            pdb_text=full_pdb_text,
            pocket_pdb_text=pocket_pdb_text,
            ligand_mol_3d=lig_mol_3d,
            ligand_smiles=can_smiles,
            coords=coords_world,
            plddt_per_atom=tuple(float(v) for v in plddt_np),
            plddt_mean=float(plddt_np.mean()),
            ligand_plddt=float(conf.ligand_plddt[0].item()),
            pae_matrix=pae_np,
            pde_matrix=pde_np,
            ptm=float(conf.ptm[0].item()),
            iptm=float(conf.iptm[0].item()),
            ranking_score=float(conf.ranking_score[0].item()),
            predicted_pkd=(
                float(conf.predicted_pkd[0].item())
                if conf.predicted_pkd is not None
                else float("nan")
            ),
            binding_prob=(
                float(conf.binding_prob[0].item())
                if conf.binding_prob is not None
                else float("nan")
            ),
            runtime_sec=runtime_sec,
            provenance_sha256=prov_hash,
            pose_audit=None,
            metadata={
                "preset": self.config.preset,
                "trunk_mode": self.config.trunk_mode,
                "diffusion_mode": self.config.diffusion_mode,
                "flex_mode": flex_mode,
                "num_samples": num_samples,
                "num_steps": int(self.config.default_steps if num_steps is None else num_steps),
                "seed": seed,
                "device": str(device),
                "n_pocket_atoms": lig_start,
                "n_ligand_atoms": n_lig,
                "n_tokens": int(batch_cpu.token_mask.shape[1]),
            },
        )
        if run_audit:
            audit_rep = result.run_audit(
                property_profile=property_profile,
                canonical_anchors=canonical_anchors,
            )
            result = NanoFold3Result(
                backend=result.backend,
                name=result.name,
                pdb_text=result.pdb_text,
                pocket_pdb_text=result.pocket_pdb_text,
                ligand_mol_3d=result.ligand_mol_3d,
                ligand_smiles=result.ligand_smiles,
                coords=result.coords,
                plddt_per_atom=result.plddt_per_atom,
                plddt_mean=result.plddt_mean,
                ligand_plddt=result.ligand_plddt,
                pae_matrix=result.pae_matrix,
                pde_matrix=result.pde_matrix,
                ptm=result.ptm,
                iptm=result.iptm,
                ranking_score=result.ranking_score,
                predicted_pkd=result.predicted_pkd,
                binding_prob=result.binding_prob,
                runtime_sec=time.perf_counter() - t0,
                provenance_sha256=result.provenance_sha256,
                pose_audit=audit_rep,
                metadata=result.metadata,
            )
        return result

    @torch.inference_mode()
    def fold_sequence(
        self,
        sequence: str,
        *,
        num_samples: int = 1,
        num_steps: int | None = None,
        seed: int = 42,
        name: str = "nanofold3_sequence",
    ) -> NanoFold3Result:
        """Single-sequence de novo mini-protein / peptide folding (zero MSA)."""
        t0 = time.perf_counter()
        device = next(self.parameters()).device
        batch_cpu, records = self.featurizer.featurize_sequence(sequence)
        batch = batch_cpu.to(device)
        out = self.sample(
            batch,
            num_steps=num_steps,
            num_samples=num_samples,
            flex_mode="de_novo_all_atom",
            seed=seed,
        )
        best_s = int(out.best_sample_idx[0].item())
        coords_world = out.coords[best_s, 0].detach().cpu().numpy().astype(np.float32)

        conf = out.confidence
        plddt_np = conf.plddt[0].detach().cpu().numpy().astype(np.float32)
        pae_np = conf.pae[0].detach().cpu().numpy().astype(np.float32)
        pde_np = conf.pde[0].detach().cpu().numpy().astype(np.float32)
        pdb_text = _format_pdb_lines(records, coords_world, plddt_np, include_ligand=False)
        prov_hash = hashlib.sha256(pdb_text.encode("utf-8")).hexdigest()

        return NanoFold3Result(
            backend="nanofold3",
            name=name,
            pdb_text=pdb_text,
            pocket_pdb_text=pdb_text,
            ligand_mol_3d=None,
            ligand_smiles=None,
            coords=coords_world,
            plddt_per_atom=tuple(float(v) for v in plddt_np),
            plddt_mean=float(plddt_np.mean()),
            ligand_plddt=float("nan"),
            pae_matrix=pae_np,
            pde_matrix=pde_np,
            ptm=float(conf.ptm[0].item()),
            iptm=float(conf.iptm[0].item()),
            ranking_score=float(conf.ranking_score[0].item()),
            predicted_pkd=float("nan"),
            binding_prob=float("nan"),
            runtime_sec=time.perf_counter() - t0,
            provenance_sha256=prov_hash,
            pose_audit=None,
            metadata={
                "preset": self.config.preset,
                "trunk_mode": self.config.trunk_mode,
                "diffusion_mode": self.config.diffusion_mode,
                "flex_mode": "de_novo_all_atom",
                "sequence_length": len(sequence.strip()),
                "num_samples": num_samples,
                "num_steps": int(self.config.default_steps if num_steps is None else num_steps),
                "seed": seed,
                "device": str(device),
            },
        )


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve ``'auto'`` / ``None`` / explicit device string cleanly."""
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def build_nanofold3(
    preset: PresetName = "nano",
    *,
    trunk_mode: TrunkMode = "pairformer",
    diffusion_mode: DiffusionMode = "edm",
    enable_affinity_head: bool = True,
    device: torch.device | str | None = None,
    **overrides: Any,
) -> NanoFold3:
    """Convenience factory constructing a :class:`NanoFold3` model from a named preset."""
    model = NanoFold3.from_preset(
        preset,
        trunk_mode=trunk_mode,
        diffusion_mode=diffusion_mode,
        enable_affinity_head=enable_affinity_head,
        **overrides,
    )
    if device is not None:
        model = model.to(_resolve_device(device))
    return model


__all__ = [
    "ComplexBatch",
    "ComplexTokenFeaturizer",
    "NanoFold3",
    "NanoFold3Config",
    "NanoFold3Result",
    "NanoFold3SampleOutput",
    "NanoFold3TrainOutput",
    "PresetName",
    "TrunkMode",
    "build_nanofold3",
]
