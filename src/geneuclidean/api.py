"""High-level batteries-included Python SDK facade for GenEuclidean.

Provides an ergonomic, strictly typed, lazily imported interface covering the entire
GenEuclidean 3D structure-based drug design (SBDD) lifecycle:

- **3D Binding Pockets:** :class:`Pocket` (``.from_target``, ``.from_pdb``,
  ``.featurize``, ``.audit_hotspots``, ``.orthosteric_box``).
- **Molecular Scoring, Triage & Ionization:** :func:`score_molecules`,
  :func:`triage_molecule`, :func:`triage_molecules`, :func:`protonate_ph74`.
- **3D-Conditioned De Novo Generation:** :class:`MoleculeDesigner` wrapping both v2
  ``GenEuclideanModel`` (CVAE + RoPE/SwiGLU Transformer) and v3/v4
  ``PocketConditionedLLM`` (``PaiNN-SE3`` + ``PocketPerceiverResampler`` +
  ``GP-MoLFormer-Uniq`` / ``SAFE-GPT`` / ``NovoMolGen`` / ``Qwen`` / ``mock-tiny-gpt2``).
- **3D Physics Docking & Redocking:** :func:`dock_and_profile`, :func:`redock_complex`.
- **Dual-Mode 3D & 2D Studio Visualization:** :func:`visualize_complex`,
  :func:`visualize_gallery`, :func:`visualize_molecules_2d`.
- **Registries & Regulatory Credibility:** :func:`list_targets`, :func:`get_target`,
  :func:`list_backbones`, :func:`build_credibility_dossier`,
  :func:`export_credibility_json`, :func:`export_credibility_markdown`.

All heavy dependencies (``torch``, ``transformers``, ``matplotlib``, ``sklearn``,
``meeko``, ``rdkit``) are imported strictly on demand inside function bodies so bare
``import geneuclidean as ge`` completes in milliseconds with zero GPU/PyTorch overhead.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from rdkit import Chem

    from geneuclidean.cns.medchem_triage import MedChemTriageResult
    from geneuclidean.cns.targets import CNSTargetSpec
    from geneuclidean.data.featurizer import PocketFeatures
    from geneuclidean.evaluation.docking_pipeline import (
        BoxMode,
        DockingPoseResult,
        OrthostericBox,
        RedockingBenchmarkResult,
    )
    from geneuclidean.evaluation.viz3d import ComplexShowcaseSpec


# ---------------------------------------------------------------------------
# Zero-PyTorch Static Chemical Foundation & Local LLM Backbone Catalog
# ---------------------------------------------------------------------------

_STATIC_BACKBONE_CATALOG: dict[str, dict[str, Any]] = {
    "ibm-research/GP-MoLFormer-Uniq": {
        "backbone_id": "ibm-research/GP-MoLFormer-Uniq",
        "family": "molformer",
        "params": "46.8M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "ibm-research/MoLFormer-XL-both-10pct",
        "notes": "Primary GenEuclidean v3.0/v3.1 chemical foundation causal transformer.",
    },
    "datamol-io/safe-gpt": {
        "backbone_id": "datamol-io/safe-gpt",
        "family": "safe_gpt",
        "params": "87.3M",
        "license": "Apache-2.0",
        "representation": "safe",
        "is_smiles_native": True,
        "tokenizer_id": "datamol-io/safe-gpt",
        "aliases": ["safe-gpt"],
        "notes": "Fragment-modular SAFE generative model trained on 1.1B molecules.",
    },
    "chandar-lab/NovoMolGen_157M_SMILES_BPE": {
        "backbone_id": "chandar-lab/NovoMolGen_157M_SMILES_BPE",
        "family": "novomolgen",
        "params": "157M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "chandar-lab/NovoMolGen_157M_SMILES_BPE",
        "aliases": ["novomolgen-157m"],
        "notes": "Modern Llama-style SMILES BPE molecular foundation model (157M).",
    },
    "chandar-lab/NovoMolGen_300M_SMILES_BPE": {
        "backbone_id": "chandar-lab/NovoMolGen_300M_SMILES_BPE",
        "family": "novomolgen",
        "params": "300M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "chandar-lab/NovoMolGen_300M_SMILES_BPE",
        "notes": "Modern Llama-style SMILES BPE molecular foundation model (300M).",
    },
    "entropy/gpt2_zinc_87m": {
        "backbone_id": "entropy/gpt2_zinc_87m",
        "family": "gpt2",
        "params": "87.3M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "entropy/gpt2_zinc_87m",
        "notes": "Native GPT-2 ZINC SMILES causal language model fallback.",
    },
    "Qwen/Qwen2.5-0.5B": {
        "backbone_id": "Qwen/Qwen2.5-0.5B",
        "family": "general_llm",
        "params": "494M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": False,
        "tokenizer_id": "Qwen/Qwen2.5-0.5B",
        "notes": "Compact general-purpose local LLM (bfloat16 on MPS/CUDA).",
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "backbone_id": "HuggingFaceTB/SmolLM2-360M",
        "family": "general_llm",
        "params": "362M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": False,
        "tokenizer_id": "HuggingFaceTB/SmolLM2-360M",
        "notes": "Ultra-compact local LLM (bfloat16 on MPS/CUDA).",
    },
    "Qwen/Qwen3-0.6B-Base": {
        "backbone_id": "Qwen/Qwen3-0.6B-Base",
        "family": "general_llm",
        "params": "596M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": False,
        "tokenizer_id": "Qwen/Qwen3-0.6B-Base",
        "notes": "2025 compact Qwen3 base LLM (bfloat16 on MPS/CUDA).",
    },
    "mock-tiny-gpt2": {
        "backbone_id": "mock-tiny-gpt2",
        "family": "mock",
        "params": "<1M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "mock-tiny-gpt2",
        "notes": "Zero-download offline 2-layer GPT-2 + SMILES tokenizer for CI and tutorials.",
    },
}


# ---------------------------------------------------------------------------
# Target & Backbone Registries
# ---------------------------------------------------------------------------


def list_targets(indication: str | None = None) -> list[CNSTargetSpec]:
    """Return curated human CNS drug target specifications, optionally filtered by keyword.

    Args:
        indication: Optional case-insensitive keyword matching indications or mechanism of
            action (e.g., ``"tourette"``, ``"schizophrenia"``, ``"alzheimer"``, ``"adhd"``).

    Returns:
        List of :class:`~geneuclidean.cns.targets.CNSTargetSpec` instances.
    """
    from geneuclidean.cns.targets import list_cns_targets

    return list_cns_targets(indication_filter=indication)


def get_target(target_id: str) -> CNSTargetSpec:
    """Look up a verified human CNS target by symbol, gene name, ChEMBL ID, or UniProt ID.

    Args:
        target_id: Identifier such as ``"DRD1"``, ``"SLC18A2"``, ``"CHEMBL1893"``, or ``"P21728"``.

    Returns:
        Matching :class:`~geneuclidean.cns.targets.CNSTargetSpec`.

    Raises:
        KeyError: If ``target_id`` is not in the curated registry.
    """
    from geneuclidean.cns.targets import get_cns_target

    return get_cns_target(target_id)


def list_backbones() -> dict[str, dict[str, Any]]:
    """Return metadata for all supported pretrained chemical & local LLM backbones.

    Executes without importing ``torch`` or ``transformers`` so startup stays instantaneous.
    If ``geneuclidean.models.hf_backbones`` is already loaded in the current interpreter,
    any additional dynamically registered backbones are merged automatically.

    Returns:
        Dictionary mapping canonical Hugging Face / mock backbone IDs to metadata dicts.
    """
    catalog: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _STATIC_BACKBONE_CATALOG.items()}
    mod = sys.modules.get("geneuclidean.models.hf_backbones")
    if mod is not None and hasattr(mod, "BACKBONE_REGISTRY"):
        reg = mod.BACKBONE_REGISTRY
        for key, spec in reg.items():
            bid = str(getattr(spec, "backbone_id", key))
            if bid not in catalog:
                catalog[bid] = {
                    "backbone_id": bid,
                    "family": str(getattr(spec, "family", "custom")),
                    "representation": str(getattr(spec, "representation", "smiles")),
                    "is_smiles_native": bool(getattr(spec, "is_smiles_native", False)),
                    "tokenizer_id": str(getattr(spec, "tokenizer_id", bid)),
                }
    return catalog


# ---------------------------------------------------------------------------
# 3D Binding Pocket Abstraction (`Pocket`)
# ---------------------------------------------------------------------------


@dataclass
class Pocket:
    """Ergonomic 3D protein binding-pocket handle for featurization, auditing, and design.

    Create pockets either from the curated human CNS target registry via
    :meth:`Pocket.from_target` or from any local PDB/mmCIF structure via
    :meth:`Pocket.from_pdb`.

    Examples:
        >>> import geneuclidean as ge
        >>> pocket = ge.Pocket.from_target("DRD1")
        >>> pocket.pdb_id
        '9LLG'
    """

    pdb_id: str
    target_id: str | None = None
    pocket_pdb: Path | None = None
    ligand_sdf: Path | None = None
    center: tuple[float, float, float] | None = None
    structures_dir: Path | None = None
    pocket_cutoff: float = 10.0
    contact_cutoff: float = 4.5
    max_atoms: int = 384
    target_spec: CNSTargetSpec | None = None
    _cached_features: PocketFeatures | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_target(
        cls,
        target_id: str,
        *,
        pdb_id: str | None = None,
        structures_dir: str | Path | None = None,
        allow_fetch: bool = False,
        pocket_cutoff: float = 10.0,
        contact_cutoff: float = 4.5,
        max_atoms: int = 384,
    ) -> Pocket:
        """Instantiate a :class:`Pocket` for a registered human CNS target.

        Automatically discovers pre-cleaned ``*_pocket.pdb`` / ``*_lig.sdf`` files when
        present locally under ``structures_dir`` (or default ``data/cns_neuro/structures``),
        and optionally fetches the RCSB PDB structure if ``allow_fetch=True``.

        Args:
            target_id: Registered target ID, gene symbol, ChEMBL ID, or UniProt accession.
            pdb_id: Optional specific RCSB PDB accession (defaults to primary template).
            structures_dir: Optional directory containing ``clean_pockets/`` or ``raw_cif/``.
            allow_fetch: If ``True`` and no local file exists, download structure from RCSB.
            pocket_cutoff: Inclusion shell radius in Ångströms.
            contact_cutoff: Ligand contact distance threshold in Ångströms.
            max_atoms: Maximum retained heavy pocket atoms.

        Returns:
            Configured :class:`Pocket` instance.
        """
        from geneuclidean.cns.targets import (
            DEFAULT_CNS_STRUCTURES_DIR,
            fetch_rcsb_structure,
            find_local_pocket_files,
            get_cns_target,
        )

        spec = get_cns_target(target_id)
        chosen_pdb = (pdb_id or spec.primary_pdb_id).strip().upper()

        search_dirs: list[Path] = []
        if structures_dir is not None:
            search_dirs.append(Path(structures_dir))
        else:
            search_dirs.append(DEFAULT_CNS_STRUCTURES_DIR)
            alt_nosync = Path("data/cns_neuro.nosync/structures")
            if alt_nosync != DEFAULT_CNS_STRUCTURES_DIR:
                search_dirs.append(alt_nosync)

        pocket_path: Path | None = None
        lig_sdf: Path | None = None
        resolved_dir = search_dirs[0]
        for sdir in search_dirs:
            found = find_local_pocket_files(spec.target_id, pdb_id=pdb_id, structures_dir=sdir)
            if found is not None:
                pocket_path, lig_sdf = found
                resolved_dir = sdir
                break

        if pocket_path is None and allow_fetch:
            pocket_path = fetch_rcsb_structure(chosen_pdb, resolved_dir / "fetched")

        return cls(
            pdb_id=chosen_pdb,
            target_id=spec.target_id,
            pocket_pdb=pocket_path,
            ligand_sdf=lig_sdf,
            structures_dir=resolved_dir,
            pocket_cutoff=float(pocket_cutoff),
            contact_cutoff=float(contact_cutoff),
            max_atoms=int(max_atoms),
            target_spec=spec,
        )

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        *,
        center: Sequence[float] | None = None,
        ligand_sdf: str | Path | None = None,
        pdb_id: str | None = None,
        target_id: str | None = None,
        pocket_cutoff: float = 10.0,
        contact_cutoff: float = 4.5,
        max_atoms: int = 384,
    ) -> Pocket:
        """Instantiate a :class:`Pocket` directly from a local PDB or mmCIF file.

        Args:
            pdb_path: Path to a receptor or pre-cut pocket ``.pdb`` / ``.cif`` file.
            center: Optional 3D binding-site centroid ``(x, y, z)`` in Ångströms.
            ligand_sdf: Optional co-crystal 3D ligand ``.sdf`` file for centering & PLIF.
            pdb_id: Optional 4-character PDB identifier (inferred from filename if omitted).
            target_id: Optional registered CNS target ID to attach hotspot metadata.
            pocket_cutoff: Inclusion shell radius in Ångströms.
            contact_cutoff: Ligand contact distance threshold in Ångströms.
            max_atoms: Maximum retained heavy pocket atoms.

        Returns:
            Configured :class:`Pocket` instance.

        Raises:
            FileNotFoundError: If ``pdb_path`` does not exist.
            ValueError: If ``center`` does not contain exactly 3 numeric coordinates.
        """
        path = Path(pdb_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pocket structure file not found: {path}")

        center_tuple: tuple[float, float, float] | None = None
        if center is not None:
            if len(center) != 3:
                raise ValueError(f"center must have exactly 3 coordinates, got {len(center)}")
            center_tuple = (float(center[0]), float(center[1]), float(center[2]))

        sdf_path = Path(ligand_sdf) if ligand_sdf is not None else None
        if sdf_path is not None and not sdf_path.is_file():
            raise FileNotFoundError(f"Ligand SDF file not found: {sdf_path}")

        spec = get_target(target_id) if target_id is not None else None
        resolved_pdb_id = (pdb_id or path.stem.split("_")[0]).strip().upper()
        return cls(
            pdb_id=resolved_pdb_id,
            target_id=spec.target_id if spec is not None else target_id,
            pocket_pdb=path,
            ligand_sdf=sdf_path,
            center=center_tuple,
            pocket_cutoff=float(pocket_cutoff),
            contact_cutoff=float(contact_cutoff),
            max_atoms=int(max_atoms),
            target_spec=spec,
        )

    def featurize(self, *, allow_fetch: bool = False, force: bool = False) -> PocketFeatures:
        """Compute strictly E(3)-invariant 52-dim per-atom 3D pocket features.

        Args:
            allow_fetch: If ``True`` and created via :meth:`from_target` without local
                files, fetches the RCSB structure before featurizing.
            force: If ``True``, recompute even if cached features exist.

        Returns:
            :class:`~geneuclidean.data.featurizer.PocketFeatures` with centered ``[N, 3]``
            coordinates and ``[N, 52]`` E(3)-invariant node descriptors.
        """
        if self._cached_features is not None and not force:
            return self._cached_features

        from geneuclidean.data.featurizer import (
            extract_pocket_from_files,
            featurize_pdb_pocket_inference,
        )

        if self.pocket_pdb is not None and self.pocket_pdb.is_file():
            if self.ligand_sdf is not None and self.ligand_sdf.is_file():
                fallback_smi = (
                    self.target_spec.reference_drugs[0].canonical_smiles
                    if (self.target_spec is not None and self.target_spec.reference_drugs)
                    else None
                )
                feat = extract_pocket_from_files(
                    self.pocket_pdb,
                    ligand_sdf=self.ligand_sdf,
                    fallback_smiles=fallback_smi,
                    pdb_id=self.pdb_id.lower(),
                    pocket_cutoff=self.pocket_cutoff,
                    contact_cutoff=self.contact_cutoff,
                    max_atoms=self.max_atoms,
                )
            else:
                feat = featurize_pdb_pocket_inference(
                    self.pocket_pdb,
                    center=self.center,
                    pocket_cutoff=self.pocket_cutoff,
                    max_atoms=self.max_atoms,
                    pdb_id=self.pdb_id.lower(),
                )
            self._cached_features = feat
            return feat

        if self.target_id is not None:
            from geneuclidean.cns.targets import DEFAULT_CNS_STRUCTURES_DIR, load_cns_target_pocket

            feat = load_cns_target_pocket(
                self.target_id,
                pdb_id=self.pdb_id,
                structures_dir=self.structures_dir or DEFAULT_CNS_STRUCTURES_DIR,
                allow_fetch=allow_fetch,
                pocket_cutoff=self.pocket_cutoff,
                contact_cutoff=self.contact_cutoff,
                max_atoms=self.max_atoms,
            )
            self._cached_features = feat
            return feat

        raise FileNotFoundError(
            f"No 3D structure file available for pocket {self.pdb_id!r}. "
            "Pass a valid pdb_path to Pocket.from_pdb(...) or allow_fetch=True."
        )

    def audit_hotspots(
        self,
        *,
        target_id: str | None = None,
        contact_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Audit canonical pharmacological hotspot residue presence and 4.5 Å contacts.

        Args:
            target_id: Optional target ID override if pocket was loaded via :meth:`from_pdb`.
            contact_cutoff: Distance threshold in Ångströms for ligand-contact verification.

        Returns:
            Dictionary reporting ``present_hotspots``, ``contact_hotspots_4p5A``,
            ``anchor_in_contact``, and ``hotspot_coverage``.
        """
        from geneuclidean.cns.targets import audit_pocket_hotspots, get_cns_target

        spec = get_cns_target(target_id) if target_id is not None else self.target_spec
        if spec is None:
            raise ValueError(
                "Hotspot auditing requires a registered target spec. "
                "Create Pocket via Pocket.from_target(...) or pass target_id='DRD1'."
            )
        if self.pocket_pdb is None or not self.pocket_pdb.is_file():
            raise FileNotFoundError(
                f"No local pocket PDB found for {spec.target_id} ({self.pdb_id}). "
                "Supply structures_dir or Pocket.from_pdb(...)."
            )
        return audit_pocket_hotspots(
            self.pocket_pdb,
            spec,
            ligand_sdf=self.ligand_sdf,
            contact_cutoff=self.contact_cutoff if contact_cutoff is None else float(contact_cutoff),
        )

    def orthosteric_box(
        self,
        *,
        mode: BoxMode = "tight_per_axis",
        padding: float | None = None,
        min_edge: float | None = None,
        max_edge: float | None = None,
    ) -> OrthostericBox:
        """Compute an AutoDock Vina orthosteric box centered on this pocket's ligand or site."""
        from geneuclidean.evaluation.docking_pipeline import OrthostericBox, compute_orthosteric_box

        if self.ligand_sdf is not None and self.ligand_sdf.is_file():
            return compute_orthosteric_box(
                self.ligand_sdf,
                mode=mode,
                padding=padding,
                min_edge=min_edge,
                max_edge=max_edge,
            )

        if self.center is not None:
            eff_center = self.center
        elif self.pocket_pdb is not None and self.pocket_pdb.is_file():
            from geneuclidean.evaluation.geometry3d import extract_pocket_heavy_atoms

            coords, _ = extract_pocket_heavy_atoms(self.pocket_pdb)
            if len(coords) == 0:
                raise ValueError(f"No heavy protein atoms found in {self.pocket_pdb}")
            c_arr = coords.mean(axis=0)
            eff_center = (float(c_arr[0]), float(c_arr[1]), float(c_arr[2]))
        else:
            raise ValueError(
                "Cannot compute orthosteric box without ligand_sdf, center, or pocket_pdb."
            )

        edge = 20.0 if mode == "cubic_standard" else 18.0
        return OrthostericBox(
            center=eff_center,
            size=(edge, edge, edge),
            native_span_xyz=(0.0, 0.0, 0.0),
            mode=mode,
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of this pocket."""
        return {
            "target_id": self.target_id,
            "pdb_id": self.pdb_id,
            "pocket_pdb": str(self.pocket_pdb) if self.pocket_pdb is not None else None,
            "ligand_sdf": str(self.ligand_sdf) if self.ligand_sdf is not None else None,
            "center": list(self.center) if self.center is not None else None,
            "primary_anchor": (
                self.target_spec.primary_anchor_residue if self.target_spec is not None else None
            ),
            "uniprot_id": self.target_spec.uniprot_id if self.target_spec is not None else None,
            "chembl_id": self.target_spec.chembl_id if self.target_spec is not None else None,
        }


# ---------------------------------------------------------------------------
# Physiological Protonation & MedChem Triage Facades
# ---------------------------------------------------------------------------


@overload
def protonate_ph74(
    mol_or_smiles: str,
    *,
    preserve_coords: bool = False,
    isomeric: bool = True,
) -> str: ...


@overload
def protonate_ph74(
    mol_or_smiles: Chem.Mol,
    *,
    preserve_coords: bool = False,
    isomeric: bool = True,
) -> Chem.Mol: ...


def protonate_ph74(
    mol_or_smiles: str | Chem.Mol,
    *,
    preserve_coords: bool = False,
    isomeric: bool = True,
) -> str | Chem.Mol:
    """Deterministically protonate a SMILES string or RDKit ``Mol`` at physiological pH 7.4.

    Args:
        mol_or_smiles: SMILES string (returns protonated canonical SMILES) or RDKit ``Mol``
            (returns protonated sanitized ``Chem.Mol``).
        preserve_coords: Preserve 3D coordinates when ``mol_or_smiles`` is a 3D ``Chem.Mol``.
        isomeric: Preserve stereochemical markers when returning SMILES.

    Returns:
        Protonated canonical SMILES string or RDKit ``Mol`` matching input type.
    """
    from geneuclidean.evaluation.protonation import protonate_mol_ph74, protonate_smiles_ph74

    if isinstance(mol_or_smiles, str):
        return protonate_smiles_ph74(mol_or_smiles, isomeric=isomeric)
    return protonate_mol_ph74(mol_or_smiles, preserve_coords=preserve_coords)


def triage_molecule(mol_or_smiles: str | Chem.Mol) -> MedChemTriageResult:
    """Run 2026 industrial MedChem structural triage, reagent quarantine & Lilly demerits.

    Screens against RDKit ``PAINS``, ``BRENK``, and ``NIH`` catalogs, explicitly quarantines
    synthesis coupling reagents/byproducts (DCU, DIU, HOBt, HOAt, DMAP, Ph3P=O, symmetric
    dialkyl ureas, reactive warheads), computes Lilly-inspired demerits and Bemis-Murcko
    scaffolds, and outputs a smooth ``hygiene_gate`` in ``[0.0, 1.0]``.
    """
    from geneuclidean.cns.medchem_triage import triage_molecule as _triage_one

    return _triage_one(mol_or_smiles)


def triage_molecules(smiles_list: Sequence[str | Chem.Mol]) -> list[MedChemTriageResult]:
    """Run :func:`triage_molecule` across a batch of SMILES strings or RDKit ``Mol`` objects."""
    from geneuclidean.cns.medchem_triage import triage_batch

    return triage_batch(smiles_list)


# ---------------------------------------------------------------------------
# Unified Molecular Scoring (`MoleculeReport` & `score_molecules`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeReport:
    """Comprehensive physicochemical, CNS MPO, MedChem triage, pH 7.4 & ADMET report."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    target_id: str | None = None
    # Physiological pH 7.4 ionization
    smiles_ph74: str | None = None
    net_charge_ph74: int | None = None
    has_basic_amine: bool = False
    # Physicochemical properties & Pfizer CNS MPO v1/v2
    mw: float = float("nan")
    clogp: float = float("nan")
    clogd_74: float = float("nan")
    tpsa: float = float("nan")
    hbd: int = 0
    hba: int = 0
    most_basic_pka: float = float("nan")
    cns_mpo_v1: float = 0.0
    cns_mpo_v2: float = 0.0
    bbb_clark_pass: bool = False
    qed: float = 0.0
    sa_score: float = 10.0
    sa_norm: float = 0.0
    lipinski_violations: int = 4
    lipinski_pass: bool = False
    # Industrial MedChem Triage
    pains_free: bool = False
    brenk_free: bool = False
    nih_free: bool = True
    reagent_quarantine_free: bool = True
    lilly_demerit_score: int = 0
    triage_passed: bool = False
    hygiene_gate: float = 1.0
    triage_alerts: tuple[str, ...] = ()
    murcko_scaffold: str = ""
    generic_scaffold: str = ""
    # Optional calibrated CNS ADMET oracle predictions & composite score
    admet: dict[str, Any] | None = None
    pred_pkd: float | None = None
    clip_sim: float | None = None
    composite_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a flat JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "target_id": self.target_id,
            "smiles_ph74": self.smiles_ph74,
            "net_charge_ph74": self.net_charge_ph74,
            "has_basic_amine": self.has_basic_amine,
            "mw": round(self.mw, 2) if math.isfinite(self.mw) else None,
            "clogp": round(self.clogp, 3) if math.isfinite(self.clogp) else None,
            "clogd_74": round(self.clogd_74, 3) if math.isfinite(self.clogd_74) else None,
            "tpsa": round(self.tpsa, 2) if math.isfinite(self.tpsa) else None,
            "hbd": self.hbd,
            "hba": self.hba,
            "most_basic_pka": (
                round(self.most_basic_pka, 2) if math.isfinite(self.most_basic_pka) else None
            ),
            "cns_mpo_v1": round(self.cns_mpo_v1, 3),
            "cns_mpo_v2": round(self.cns_mpo_v2, 3),
            "bbb_clark_pass": self.bbb_clark_pass,
            "qed": round(self.qed, 3),
            "sa_score": round(self.sa_score, 3),
            "sa_norm": round(self.sa_norm, 3),
            "lipinski_violations": self.lipinski_violations,
            "lipinski_pass": self.lipinski_pass,
            "pains_free": self.pains_free,
            "brenk_free": self.brenk_free,
            "nih_free": self.nih_free,
            "reagent_quarantine_free": self.reagent_quarantine_free,
            "lilly_demerit_score": self.lilly_demerit_score,
            "triage_passed": self.triage_passed,
            "hygiene_gate": round(self.hygiene_gate, 4),
            "triage_alerts": list(self.triage_alerts),
            "murcko_scaffold": self.murcko_scaffold,
            "generic_scaffold": self.generic_scaffold,
            "admet": dict(self.admet) if self.admet is not None else None,
            "pred_pkd": round(self.pred_pkd, 3) if self.pred_pkd is not None else None,
            "clip_sim": round(self.clip_sim, 4) if self.clip_sim is not None else None,
            "composite_reward": round(self.composite_reward, 4),
        }


def score_molecules(
    smiles_list: Sequence[str] | str,
    *,
    target: str | None = None,
    include_triage: bool = True,
    include_admet: bool = True,
    oracle_path: str | Path | None = None,
    ad_damp: bool = True,
    pred_pkd: Sequence[float] | None = None,
    clip_sim: Sequence[float] | None = None,
) -> list[MoleculeReport]:
    """Score one or more SMILES across Pfizer CNS MPO, pH 7.4 ionization, triage & ADMET.

    Args:
        smiles_list: Single SMILES string or sequence of SMILES strings.
        target: Optional CNS target identifier recorded on each report (validated if supplied).
        include_triage: Run 2026 MedChem structural triage (PAINS/BRENK/NIH, reagent
            quarantine, Lilly demerits, Murcko scaffolds, smooth ``hygiene_gate``).
        include_admet: Evaluate calibrated BBB/P-gp/hERG ADMET oracles when available.
        oracle_path: Explicit path to a trained ``CNSAdmetSuite`` ``.joblib`` artifact.
            If ``None`` and ``include_admet=True``, loads the default bundle if present on
            disk and gracefully skips ADMET predictions in offline environments without it.
        ad_damp: Damp out-of-domain ADMET probabilities toward 0.50 in composite reward.
        pred_pkd: Optional per-molecule predicted binding affinity ``pKd`` scores.
        clip_sim: Optional per-molecule DrugCLIP cosine similarity scores.

    Returns:
        List of :class:`MoleculeReport` entries aligned 1-to-1 with ``smiles_list``.
    """
    from geneuclidean.cns.admet import (
        DEFAULT_ADMET_MODEL_PATH,
        CNSAdmetSuite,
        compute_cns_composite_reward,
    )
    from geneuclidean.cns.medchem_triage import triage_batch
    from geneuclidean.evaluation.protonation import analyze_protonation_ph74

    items: list[str] = [smiles_list] if isinstance(smiles_list, str) else list(smiles_list)
    if not items:
        return []

    resolved_target: str | None = None
    if target is not None:
        resolved_target = get_target(target).target_id

    admet_suite: CNSAdmetSuite | None = None
    if include_admet:
        if oracle_path is not None:
            admet_suite = CNSAdmetSuite.load(oracle_path)
        else:
            admet_suite = CNSAdmetSuite.try_load(DEFAULT_ADMET_MODEL_PATH)

    comp_scores = compute_cns_composite_reward(
        items,
        pred_pkd=pred_pkd,
        clip_sim=clip_sim,
        admet_suite=admet_suite,
        ad_damp=ad_damp,
    )
    triage_results = triage_batch(items) if include_triage else None

    reports: list[MoleculeReport] = []
    for idx, smi in enumerate(items):
        comp = comp_scores[idx]
        mpo = comp.mpo
        tr = triage_results[idx] if triage_results is not None else None

        if not mpo.valid:
            reports.append(
                MoleculeReport(
                    smiles=smi,
                    canonical_smiles=None,
                    valid=False,
                    target_id=resolved_target,
                    triage_passed=False,
                    hygiene_gate=0.0,
                    triage_alerts=("invalid_smiles",),
                    composite_reward=0.0,
                )
            )
            continue

        prot = analyze_protonation_ph74(mpo.canonical_smiles or smi)
        admet_dict = comp.admet.to_dict() if comp.admet is not None else None

        if tr is not None:
            alerts = tuple(
                list(tr.reagent_alerts)
                + list(tr.pains_alerts)
                + list(tr.brenk_alerts)
                + list(tr.nih_alerts)
            )
            hygiene_gate = float(tr.hygiene_gate)
            triage_passed = bool(tr.triage_pass)
            nih_free = bool(tr.nih_free)
            reagent_free = bool(tr.reagent_quarantine_free)
            lilly_score = int(tr.lilly_demerit_score)
            murcko = tr.exact_murcko_scaffold
            generic_scaf = tr.generic_murcko_scaffold
        else:
            alerts = ()
            hygiene_gate = 1.0 if mpo.pains_free else 0.0
            triage_passed = bool(mpo.pains_free and mpo.brenk_free)
            nih_free = True
            reagent_free = True
            lilly_score = 0
            murcko = ""
            generic_scaf = ""

        # Apply smooth hygiene gate (vetoes DCU/reagent/PAINS to 0.0!)
        gated_reward = float(comp.composite_reward * hygiene_gate)

        reports.append(
            MoleculeReport(
                smiles=smi,
                canonical_smiles=mpo.canonical_smiles,
                valid=True,
                target_id=resolved_target,
                smiles_ph74=prot.protonated_smiles,
                net_charge_ph74=prot.net_charge,
                has_basic_amine=prot.has_basic_amine,
                mw=mpo.mw,
                clogp=mpo.clogp,
                clogd_74=mpo.clogd_74,
                tpsa=mpo.tpsa,
                hbd=mpo.hbd,
                hba=mpo.hba,
                most_basic_pka=mpo.most_basic_pka,
                cns_mpo_v1=mpo.cns_mpo_v1,
                cns_mpo_v2=mpo.cns_mpo_v2,
                bbb_clark_pass=mpo.bbb_clark_pass,
                qed=mpo.qed,
                sa_score=mpo.sa_score,
                sa_norm=mpo.sa_norm,
                lipinski_violations=mpo.lipinski_violations,
                lipinski_pass=mpo.lipinski_pass,
                pains_free=mpo.pains_free,
                brenk_free=mpo.brenk_free,
                nih_free=nih_free,
                reagent_quarantine_free=reagent_free,
                lilly_demerit_score=lilly_score,
                triage_passed=triage_passed,
                hygiene_gate=hygiene_gate,
                triage_alerts=alerts,
                murcko_scaffold=murcko,
                generic_scaffold=generic_scaf,
                admet=admet_dict,
                pred_pkd=comp.pred_pkd,
                clip_sim=comp.clip_sim,
                composite_reward=gated_reward,
            )
        )

    return reports


# ---------------------------------------------------------------------------
# 3D-Conditioned De Novo Generator Facade (`MoleculeDesigner`)
# ---------------------------------------------------------------------------


@dataclass
class DesignResult:
    """Output of :meth:`MoleculeDesigner.design` for a single 3D protein pocket."""

    pocket_id: str
    backend: str
    raw_smiles: list[str]
    reports: list[MoleculeReport]
    valid_rate: float
    unique_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def smiles(self) -> list[str]:
        """Return canonical SMILES of ranked/filtered designed molecules."""
        return [r.canonical_smiles or r.smiles for r in self.reports]

    def to_dict(self) -> dict[str, Any]:
        """Serialize design batch and scored molecule reports to a JSON-compatible dict."""
        return {
            "pocket_id": self.pocket_id,
            "backend": self.backend,
            "n_raw": len(self.raw_smiles),
            "n_returned": len(self.reports),
            "valid_rate": round(self.valid_rate, 4),
            "unique_rate": round(self.unique_rate, 4),
            "metadata": dict(self.metadata),
            "reports": [r.to_dict() for r in self.reports],
        }


class MoleculeDesigner:
    """Unified 3D pocket-conditioned molecular generator wrapping v2 CVAE & v3/v4 LLM models.

    Lazily imports PyTorch and Hugging Face ``transformers``/``peft`` only when an instance
    is constructed via :meth:`from_pretrained` or :meth:`from_mock`.
    """

    def __init__(
        self,
        *,
        backend: Literal["v2", "llm"],
        model: Any,
        device: Any,
        cfg: Any,
        tokenizer: Any | None = None,
        backbone_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.device = device
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.backbone_id = backbone_id or (
            f"v2-{cfg.encoder.arch}-{cfg.decoder.arch}" if backend == "v2" else "llm"
        )

    @classmethod
    def from_mock(
        cls,
        *,
        device: str = "cpu",
        seed: int = 42,
    ) -> MoleculeDesigner:
        """Instantiate an offline zero-download mock 3D-to-LLM designer for testing & demos."""
        from geneuclidean.config import ExperimentConfig, seed_everything, select_device

        try:
            from geneuclidean.models.hf_backbones import MOCK_BACKBONE_ID, load_hf_backbone
            from geneuclidean.models.llm_bridge import PocketConditionedLLM
        except ImportError as exc:
            raise RuntimeError(
                "Mock LLM designer requires optional LLM dependencies. "
                "Install them via: pip install 'geneuclidean[llm]'"
            ) from exc

        seed_everything(seed)
        torch_device = select_device(device)
        v2_cfg = ExperimentConfig()
        backbone = load_hf_backbone(MOCK_BACKBONE_ID, use_lora=False, dtype="float32")
        model = PocketConditionedLLM(
            v2_cfg=v2_cfg,
            backbone=backbone,
            num_queries=8,
            perceiver_layers=1,
            perceiver_heads=4,
            num_global_tokens=2,
        )
        model.to(torch_device)
        model.eval()
        return cls(
            backend="llm",
            model=model,
            device=torch_device,
            cfg=v2_cfg,
            tokenizer=backbone.tokenizer,
            backbone_id=MOCK_BACKBONE_ID,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        mode: Literal["auto", "v2", "llm"] = "auto",
        v2_pocket_ckpt: str | Path | None = None,
        backbone_id: str | None = None,
        vocab_path: str | Path | None = None,
        cache_dir: str = ".hf_cache.nosync",
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
    ) -> MoleculeDesigner:
        """Load a trained GenEuclidean v2 model or v3/v4 3D-to-LLM adapter checkpoint.

        Args:
            checkpoint: Path to a v2 ``best.pt`` (``model_state_dict``) or v3/v4 adapter
                ``best_adapter.pt`` (``bridge_state_dict`` + ``lora_state_dict``).
            mode: ``"auto"`` (detect from checkpoint keys), ``"v2"``, or ``"llm"``.
            v2_pocket_ckpt: Optional v2 pocket encoder checkpoint path for ``llm`` mode
                (defaults to metadata path embedded inside the adapter checkpoint).
            backbone_id: Optional HF backbone override for ``llm`` checkpoints.
            vocab_path: Optional JSON vocabulary path for ``v2`` checkpoints.
            cache_dir: Hugging Face cache directory for ``llm`` backbones.
            device: Target device (``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"``).
            dtype: Backbone dtype (``"auto"``, ``"float32"``, ``"bfloat16"``).
            local_files_only: Prevent network downloads from Hugging Face Hub.

        Returns:
            Initialized :class:`MoleculeDesigner` ready for :meth:`design`.
        """
        import torch

        from geneuclidean.config import ExperimentConfig, select_device

        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

        if mode == "auto":
            if "bridge_state_dict" in payload:
                resolved_mode: Literal["v2", "llm"] = "llm"
            elif "model_state_dict" in payload:
                resolved_mode = "v2"
            else:
                raise ValueError(
                    f"Cannot infer checkpoint mode from keys {sorted(payload.keys())}; "
                    "expected 'bridge_state_dict' (llm) or 'model_state_dict' (v2)."
                )
        else:
            resolved_mode = mode

        torch_device = select_device(device)

        if resolved_mode == "v2":
            from geneuclidean.data.tokenizer import MolecularTokenizer
            from geneuclidean.models.geneuclidean import GenEuclideanModel

            raw_cfg = payload.get("config")
            cfg = (
                ExperimentConfig.model_validate(raw_cfg)
                if isinstance(raw_cfg, dict)
                else ExperimentConfig()
            )
            if vocab_path is not None:
                tok_path = Path(vocab_path)
            else:
                tok_path = Path(cfg.data.processed_dir) / f"vocab_{cfg.data.representation}.json"
            if not tok_path.is_file():
                raise FileNotFoundError(
                    f"Tokenizer vocabulary file not found at {tok_path}. Pass vocab_path=..."
                )
            tokenizer = MolecularTokenizer.load(tok_path)
            cfg = cfg.model_copy(
                update={
                    "decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})
                }
            )
            model = GenEuclideanModel(cfg)
            model.load_state_dict(payload["model_state_dict"])
            model.to(torch_device)
            model.eval()
            return cls(
                backend="v2",
                model=model,
                device=torch_device,
                cfg=cfg,
                tokenizer=tokenizer,
            )

        # v3 / v4 PocketConditionedLLM adapter loading
        try:
            from geneuclidean.models.hf_backbones import load_hf_backbone
            from geneuclidean.models.llm_bridge import PocketConditionedLLM
        except ImportError as exc:
            raise RuntimeError(
                "LLM checkpoint loading requires optional LLM dependencies. "
                "Install them via: pip install 'geneuclidean[llm]'"
            ) from exc

        meta: dict[str, Any] = dict(payload.get("metadata", {}))
        raw_v2_cfg = meta.get("v2_cfg")
        v2_cfg = (
            ExperimentConfig.model_validate(raw_v2_cfg)
            if isinstance(raw_v2_cfg, dict)
            else ExperimentConfig()
        )
        eff_backbone_id = str(
            backbone_id or meta.get("backbone_id", "ibm-research/GP-MoLFormer-Uniq")
        )
        has_lora = bool(payload.get("lora_state_dict"))
        backbone = load_hf_backbone(
            eff_backbone_id,
            cache_dir=cache_dir,
            dtype=dtype,
            use_lora=has_lora,
            local_files_only=local_files_only,
        )
        llm_model = PocketConditionedLLM(
            v2_cfg=v2_cfg,
            backbone=backbone,
            num_queries=int(meta.get("num_queries", 16)),
            perceiver_layers=int(meta.get("perceiver_layers", 2)),
            perceiver_heads=int(meta.get("perceiver_heads", 8)),
            num_global_tokens=int(meta.get("num_global_tokens", 4)),
            cfg_drop_prob=float(meta.get("cfg_drop_prob", 0.15)),
        )
        eff_v2_ckpt = v2_pocket_ckpt or meta.get("v2_ckpt_path")
        if eff_v2_ckpt is not None and Path(str(eff_v2_ckpt)).is_file():
            llm_model.load_v2_pocket_checkpoint(Path(str(eff_v2_ckpt)), map_location="cpu")
        llm_model.load_adapter_checkpoint(ckpt_path, map_location="cpu")
        llm_model.to(torch_device)
        llm_model.eval()
        return cls(
            backend="llm",
            model=llm_model,
            device=torch_device,
            cfg=v2_cfg,
            tokenizer=backbone.tokenizer,
            backbone_id=eff_backbone_id,
        )

    def design(
        self,
        pocket: Pocket | PocketFeatures,
        *,
        n: int = 32,
        strategy: str = "min_p",
        temperature: float = 1.0,
        min_p: float = 0.08,
        top_p: float = 0.92,
        top_k: int = 30,
        cfg_scale: float = 1.6,
        latent_temp: float = 1.0,
        max_new_tokens: int = 96,
        cns_safe_only: bool = False,
        include_triage: bool = True,
        include_admet: bool = True,
        oracle_path: str | Path | None = None,
        rerank: bool = True,
        top_k_hits: int | None = None,
        seed: int = 42,
    ) -> DesignResult:
        """Generate and score *de novo* candidate ligands conditioned on a 3D binding pocket.

        Args:
            pocket: :class:`Pocket` instance or precomputed :class:`PocketFeatures`.
            n: Number of raw candidate molecules to sample.
            strategy: Decoding strategy (``"min_p"``, ``"top_p"``, ``"top_k"``, ``"greedy"``,
                ``"temperature"``, ``"typical"``).
            temperature: Softmax sampling temperature.
            min_p: Min-p probability threshold.
            top_p: Nucleus sampling threshold.
            top_k: Top-k cutoff.
            cfg_scale: Autoregressive Classifier-Free Guidance scale (``1.0`` disables CFG).
            latent_temp: CVAE prior sampling temperature.
            max_new_tokens: Maximum newly generated tokens per ligand.
            cns_safe_only: If ``True``, retain only valid, MedChem-triage-passing molecules
                with Pfizer CNS MPO v2 ``>= 4.0`` (and passing ADMET when oracle available).
            include_triage: Run industrial MedChem structural triage on generated molecules.
            include_admet: Run calibrated CNS ADMET oracle when available on disk.
            oracle_path: Optional explicit ``CNSAdmetSuite`` ``.joblib`` path.
            rerank: Sort valid candidates by descending ``composite_reward``.
            top_k_hits: Optional cap on number of returned :class:`MoleculeReport` entries.
            seed: Deterministic random seed.

        Returns:
            :class:`DesignResult` containing raw SMILES, scored reports, validity, and uniqueness.
        """
        import numpy as np
        import torch

        from geneuclidean.data.featurizer import featurize_ligand_smiles

        target_id: str | None = None
        if isinstance(pocket, Pocket):
            feat = pocket.featurize()
            target_id = pocket.target_id
            pocket_id = pocket.target_id or pocket.pdb_id
        else:
            feat = pocket
            pocket_id = feat.pdb_id.upper()

        coords = (
            torch.from_numpy(feat.coords).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        )
        node_feats = (
            torch.from_numpy(feat.node_feats)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        atom_mask = torch.ones((1, len(feat.coords)), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            pocket_out = self.model.encode_pocket(coords, node_feats, atom_mask)

            if self.backend == "llm":
                from geneuclidean.sampling.llm_sampler import sample_llm_molecules

                raw_smiles = sample_llm_molecules(
                    model=self.model,
                    pocket_out=pocket_out,
                    atom_mask=atom_mask,
                    num_samples=int(n),
                    max_new_tokens=int(max_new_tokens),
                    strategy=strategy,
                    temperature=float(temperature),
                    min_p=float(min_p),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    cfg_scale=float(cfg_scale),
                    latent_temp=float(latent_temp),
                    seed=int(seed),
                )
            else:
                from geneuclidean.sampling.decoder_strategies import sample_molecules_detailed

                sampling_cfg = self.cfg.sampling.model_copy(
                    update={
                        "strategy": cast(
                            Literal[
                                "min_p",
                                "top_p",
                                "top_k",
                                "temperature",
                                "greedy",
                                "typical",
                                "diverse_beam",
                            ],
                            strategy,
                        ),
                        "num_samples": int(n),
                        "max_len": int(max_new_tokens),
                        "temperature": float(temperature),
                        "min_p": float(min_p),
                        "top_p": float(top_p),
                        "top_k": int(top_k),
                        "cfg_scale": float(cfg_scale),
                        "latent_temp": float(latent_temp),
                    }
                )
                gen = torch.Generator(device=self.device).manual_seed(int(seed))
                batch = {
                    "coords": coords,
                    "node_feats": node_feats,
                    "atom_mask": atom_mask,
                }
                samp_res = sample_molecules_detailed(
                    model=self.model,
                    pocket_batch=batch,
                    tokenizer=self.tokenizer,
                    sampling_cfg=sampling_cfg,
                    generator=gen,
                    device=self.device,
                    return_attn=False,
                )
                raw_smiles = list(samp_res.smiles)

            # Optional multi-task pKd & DrugCLIP head scoring
            pred_pkd: list[float] | None = None
            clip_sim: list[float] | None = None
            if hasattr(self.model, "score_candidates"):
                fps: list[np.ndarray] = []
                valid_idx: list[int] = []
                fp_dim = int(self.cfg.multitask.ligand_fp_dim)
                for idx, smi in enumerate(raw_smiles):
                    parsed = featurize_ligand_smiles(smi, fp_size=fp_dim)
                    if parsed is not None:
                        valid_idx.append(idx)
                        fps.append(parsed[1])
                if valid_idx:
                    fp_tensor = torch.from_numpy(np.stack(fps, axis=0)).to(
                        device=self.device, dtype=torch.float32
                    )
                    head_scores = self.model.score_candidates(pocket_out, fp_tensor)
                    if "pkd" in head_scores:
                        pred_pkd = [float("nan")] * len(raw_smiles)
                        for idx, v in zip(
                            valid_idx, head_scores["pkd"].detach().cpu().tolist(), strict=True
                        ):
                            pred_pkd[idx] = float(v)
                    if "clip_sim" in head_scores:
                        clip_sim = [float("nan")] * len(raw_smiles)
                        for idx, v in zip(
                            valid_idx, head_scores["clip_sim"].detach().cpu().tolist(), strict=True
                        ):
                            clip_sim[idx] = float(v)

        all_reports = score_molecules(
            raw_smiles,
            target=target_id,
            include_triage=include_triage,
            include_admet=include_admet,
            oracle_path=oracle_path,
            pred_pkd=pred_pkd,
            clip_sim=clip_sim,
        )

        valid_reports = [r for r in all_reports if r.valid]
        valid_rate = len(valid_reports) / max(len(raw_smiles), 1)
        unique_canon = {r.canonical_smiles for r in valid_reports if r.canonical_smiles}
        unique_rate = len(unique_canon) / max(len(valid_reports), 1) if valid_reports else 0.0

        # Deduplicate by canonical SMILES preserving highest composite reward
        dedup_map: dict[str, MoleculeReport] = {}
        for rep in valid_reports:
            key = rep.canonical_smiles or rep.smiles
            prev = dedup_map.get(key)
            if prev is None or rep.composite_reward > prev.composite_reward:
                dedup_map[key] = rep
        filtered = list(dedup_map.values())

        if cns_safe_only:
            filtered = [
                r
                for r in filtered
                if r.triage_passed
                and r.cns_mpo_v2 >= 4.0
                and (r.admet is None or bool(r.admet.get("admet_pass", True)))
            ]

        if rerank:
            filtered.sort(key=lambda r: r.composite_reward, reverse=True)
        if top_k_hits is not None:
            filtered = filtered[: int(top_k_hits)]

        return DesignResult(
            pocket_id=pocket_id,
            backend=f"{self.backend}:{self.backbone_id}",
            raw_smiles=raw_smiles,
            reports=filtered,
            valid_rate=valid_rate,
            unique_rate=unique_rate,
            metadata={
                "n_requested": int(n),
                "strategy": strategy,
                "cfg_scale": float(cfg_scale),
                "temperature": float(temperature),
                "cns_safe_only": bool(cns_safe_only),
                "seed": int(seed),
            },
        )


# ---------------------------------------------------------------------------
# 3D Docking & Native Redocking SDK Wrappers
# ---------------------------------------------------------------------------


def dock_and_profile(
    pocket: Pocket | str | Path,
    smiles: str,
    *,
    native_ligand_sdf: str | Path | None = None,
    box_center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
    box_mode: BoxMode = "tight_per_axis",
    canonical_anchors: Sequence[str] | None = None,
    expected_anchor_kinds: Sequence[str] = ("salt_bridge", "hbond"),
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 1,
    seed: int = 42,
    require_vina: bool = True,
) -> DockingPoseResult:
    """Dock a SMILES ligand into a :class:`Pocket` or PDB file and profile 3D interactions.

    Automatically resolves the orthosteric box, native PLIF reference, and canonical anchor
    residues when ``pocket`` is a :class:`Pocket` built from a registered target.
    """
    from geneuclidean.evaluation.docking_pipeline import dock_ligand_to_pocket

    eff_sdf = native_ligand_sdf
    eff_center = box_center
    eff_anchors: Sequence[str] = canonical_anchors or ()

    if isinstance(pocket, Pocket):
        if pocket.pocket_pdb is None or not pocket.pocket_pdb.is_file():
            raise FileNotFoundError(
                f"Pocket {pocket.pdb_id!r} has no local PDB file. "
                "Provide structures_dir or construct via Pocket.from_pdb(...)."
            )
        pocket_pdb: str | Path = pocket.pocket_pdb
        if eff_sdf is None:
            eff_sdf = pocket.ligand_sdf
        if eff_center is None:
            eff_center = pocket.center
        if not eff_anchors and pocket.target_spec is not None:
            eff_anchors = (
                pocket.target_spec.primary_anchor_residue.upper(),
                *(a.upper() for a in pocket.target_spec.hotspot_isoform_aliases[:1]),
            )
    else:
        pocket_pdb = pocket

    return dock_ligand_to_pocket(
        smiles=smiles,
        pocket_pdb=pocket_pdb,
        native_ligand_sdf=eff_sdf,
        box_center=eff_center,
        box_size=box_size,
        box_mode=box_mode,
        canonical_anchors=eff_anchors,
        expected_anchor_kinds=expected_anchor_kinds,
        vina_bin=vina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        cpu=cpu,
        seed=seed,
        require_vina=require_vina,
    )


def redock_complex(
    pocket_pdb: str | Path,
    native_ligand_sdf: str | Path,
    *,
    target: str | None = None,
    seeds: Sequence[int] = (42, 137, 2026),
    canonical_anchors: Sequence[str] | None = None,
    expected_anchor_kinds: Sequence[str] = ("salt_bridge", "hbond"),
    box_mode: BoxMode = "cubic_standard",
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 2,
) -> RedockingBenchmarkResult:
    """Run unbiased 2D-to-3D multi-seed native redocking validation on a crystal complex."""
    from geneuclidean.evaluation.docking_pipeline import redock_native_complex

    eff_anchors: Sequence[str] = canonical_anchors or ()
    if not eff_anchors and target is not None:
        spec = get_target(target)
        eff_anchors = (
            spec.primary_anchor_residue.upper(),
            *(a.upper() for a in spec.hotspot_isoform_aliases[:1]),
        )
    return redock_native_complex(
        pocket_pdb=pocket_pdb,
        native_ligand_sdf=native_ligand_sdf,
        seeds=seeds,
        canonical_anchors=eff_anchors,
        expected_anchor_kinds=expected_anchor_kinds,
        box_mode=box_mode,
        vina_bin=vina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        cpu=cpu,
    )


# ---------------------------------------------------------------------------
# Dual-Mode 3D & 2D Studio Visualization Wrappers
# ---------------------------------------------------------------------------


def _coerce_3d_mol(ligand: str | Path | Chem.Mol) -> Chem.Mol:
    """Load a 3D RDKit Mol from an SDF/MOL2/PDB path or existing 3D Mol."""
    from geneuclidean.evaluation.docking_pipeline import _load_3d_ligand

    return _load_3d_ligand(ligand)


def visualize_complex(
    pocket_pdb: str | Path,
    ligand_sdf: str | Path | Chem.Mol,
    *,
    out_html: str | Path | None = None,
    out_png: str | Path | None = None,
    native_ligand_sdf: str | Path | Chem.Mol | None = None,
    title: str = "3D Protein-Ligand Complex",
    pdb_id: str = "",
    disease: str = "",
    theme: Literal["dark", "light"] = "dark",
    metrics: Mapping[str, Any] | None = None,
    residue_attention: Mapping[str, float] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Render an interactive 3Dmol.js WebGL HTML viewer and/or 300-DPI 3D studio PNG.

    At least one of ``out_html`` or ``out_png`` must be supplied.

    Args:
        pocket_pdb: Path to receptor binding-pocket PDB file.
        ligand_sdf: Path to a 3D ligand file (``.sdf``/``.mol2``/``.pdb``) or 3D RDKit ``Mol``.
        out_html: Optional destination path for self-contained interactive 3Dmol.js HTML.
        out_png: Optional destination path for 300-DPI static pseudo-Phong 3D PNG.
        native_ligand_sdf: Optional native co-crystal 3D ligand for ghost overlay.
        title: Display title for the studio banner and HUD.
        pdb_id: Optional 4-char PDB code displayed in badges.
        disease: Optional therapeutic indication tag.
        theme: Color theme for static PNG rendering (``"dark"`` or ``"light"``).
        metrics: Optional KPI dictionary displayed in the side panel HUD.
        residue_attention: Optional mapping ``residue_id -> weight in [0, 1]`` for 3D
            Perceiver attention halos.
        notes: Optional scientific commentary displayed in the HTML viewer.

    Returns:
        Dictionary containing ``html_path``, ``png_path``, and the computed
        :class:`~geneuclidean.evaluation.interactions3d.InteractionProfile` under
        ``"interactions"``.
    """
    if out_html is None and out_png is None:
        raise ValueError("Provide at least one output path: out_html=... and/or out_png=...")

    from geneuclidean.evaluation.interactions3d import profile_3d_interactions
    from geneuclidean.evaluation.viz3d import (
        render_interactive_3d_html,
        render_static_3d_complex_png,
    )

    pocket_path = Path(pocket_pdb)
    if not pocket_path.is_file():
        raise FileNotFoundError(f"Pocket PDB file not found: {pocket_path}")

    lig_mol = _coerce_3d_mol(ligand_sdf)
    nat_mol = _coerce_3d_mol(native_ligand_sdf) if native_ligand_sdf is not None else None
    profile = profile_3d_interactions(pocket_path, lig_mol)

    written_html: Path | None = None
    written_png: Path | None = None

    if out_html is not None:
        written_html = render_interactive_3d_html(
            pocket_pdb_path=pocket_path,
            ligand_mol=lig_mol,
            out_path=out_html,
            native_ligand_mol=nat_mol,
            interactions=profile,
            title=title,
            pdb_id=pdb_id,
            disease=disease,
            metrics=metrics,
            residue_attention=residue_attention,
            notes=notes,
        )
    if out_png is not None:
        written_png = render_static_3d_complex_png(
            pocket_pdb_path=pocket_path,
            ligand_mol=lig_mol,
            out_path=out_png,
            interactions=profile,
            native_ligand_mol=nat_mol,
            title=title,
            pdb_id=pdb_id,
            disease=disease,
            theme=theme,
            metrics=metrics,
            residue_attention=residue_attention,
        )

    return {
        "html_path": written_html,
        "png_path": written_png,
        "interactions": profile,
    }


def visualize_gallery(
    specs: Sequence[ComplexShowcaseSpec | Mapping[str, Any]],
    out_html: str | Path,
    *,
    gallery_title: str = "GenEuclidean — 3D Molecular Studio",
    subtitle: str = "Interactive Protein-Ligand WebGL & PLIF Inspector",
) -> Path:
    """Render a multi-target interactive 3Dmol.js WebGL HTML studio gallery.

    Accepts either :class:`~geneuclidean.evaluation.viz3d.ComplexShowcaseSpec` instances or
    convenience dictionaries with keys ``target_id``, ``title``, ``pocket_pdb`` (or
    ``pocket_pdb_path``), ``ligand_sdf`` (or ``ligand_mol``), and optional ``pdb_id``,
    ``disease``, ``native_ligand_sdf``, ``metrics``, ``residue_attention``, and ``notes``.
    """
    from geneuclidean.evaluation.viz3d import (
        ComplexShowcaseSpec,
        render_interactive_3d_gallery_html,
    )

    resolved: list[ComplexShowcaseSpec] = []
    for item in specs:
        if isinstance(item, ComplexShowcaseSpec):
            resolved.append(item)
            continue
        lig_val = item.get("ligand_mol")
        if lig_val is None and item.get("ligand_sdf") is not None:
            lig_val = _coerce_3d_mol(item["ligand_sdf"])
        nat_val = item.get("native_ligand_mol")
        if nat_val is None and item.get("native_ligand_sdf") is not None:
            nat_val = _coerce_3d_mol(item["native_ligand_sdf"])
        resolved.append(
            ComplexShowcaseSpec(
                target_id=str(item.get("target_id", "complex")),
                title=str(item.get("title", item.get("target_id", "3D Complex"))),
                pdb_id=str(item.get("pdb_id", "")),
                disease=str(item.get("disease", "")),
                pocket_pdb_path=item.get("pocket_pdb_path") or item.get("pocket_pdb"),
                pocket_pdb_text=item.get("pocket_pdb_text"),
                ligand_mol=lig_val,
                native_ligand_mol=nat_val,
                smiles=str(item.get("smiles", "")),
                interactions=item.get("interactions"),
                residue_attention=item.get("residue_attention"),
                metrics=dict(item.get("metrics") or {}),
                notes=str(item.get("notes", "")),
            )
        )

    return render_interactive_3d_gallery_html(
        resolved,
        out_path=out_html,
        gallery_title=gallery_title,
        subtitle=subtitle,
    )


def visualize_molecules_2d(
    smiles_list: Sequence[str],
    out_png: str | Path,
    *,
    labels: Sequence[str | Mapping[str, Any]] | None = None,
    reference_smiles: str | None = None,
    title: str = "GenEuclidean — 2D Molecular Gallery",
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (340, 280),
) -> Path:
    """Render a 300-DPI publication-grade 2D RDKit molecular grid PNG.

    Args:
        smiles_list: Sequence of SMILES strings to render.
        out_png: Destination ``.png`` path.
        labels: Optional per-molecule legend strings or KPI dicts (e.g. ``{"QED": 0.85}``).
        reference_smiles: Optional reference ligand SMILES rendered in the first cell.
        title: Figure title displayed above the 2D grid.
        mols_per_row: Maximum number of molecules per row.
        sub_img_size: Per-molecule sub-image ``(width, height)`` in pixels.

    Returns:
        Path to the saved 300-DPI PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Draw

    from geneuclidean.evaluation.visualization import _format_mol_annotation, _save_figure

    mols: list[Chem.Mol] = []
    legends: list[str] = []

    if reference_smiles is not None:
        ref_mol = Chem.MolFromSmiles(reference_smiles.strip())
        if ref_mol is not None:
            mols.append(ref_mol)
            legends.append("Reference Ligand")

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi.strip()) if isinstance(smi, str) and smi.strip() else None
        if mol is None:
            continue
        mols.append(mol)
        if labels is not None and idx < len(labels):
            lbl = labels[idx]
            legends.append(_format_mol_annotation(lbl) if isinstance(lbl, Mapping) else str(lbl))
        else:
            legends.append(f"Mol #{idx + 1}")

    out_path = Path(out_png)
    if not mols:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, "No valid molecules to render", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, out_path)

    grid_img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(int(mols_per_row), len(mols)),
        subImgSize=sub_img_size,
        legends=legends,
        returnPNG=False,
    )
    img_arr = np.asarray(grid_img)
    fig_w = max(7.5, img_arr.shape[1] / 120.0)
    fig_h = max(3.4, img_arr.shape[0] / 120.0 + 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_arr)
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    return _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# FDA Jan 2025 / Joint FDA-EMA Jan 2026 Regulatory Credibility Dossier Facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredibilityDossier:
    """Structured FDA Jan 2025 (``FDA-2024-D-4689``) / EMA Jan 2026 AI Credibility Dossier."""

    payload: dict[str, Any]

    @property
    def schema_version(self) -> str:
        """Return credibility dossier schema identifier."""
        return str(self.payload.get("schema_version", ""))

    @property
    def context_of_use(self) -> str:
        """Return Context of Use (COU) statement."""
        cou = self.payload.get("step_2_context_of_use", {})
        return str(cou.get("statement", "")) if isinstance(cou, Mapping) else str(cou)

    @property
    def disclaimer(self) -> str:
        """Return explicit *in silico* human-in-the-loop disclaimer."""
        return str(self.payload.get("human_in_the_loop_boundary", ""))

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the JSON-serializable dossier payload."""
        return dict(self.payload)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the dossier to formatted JSON text."""
        return json.dumps(self.payload, indent=indent) + "\n"

    def to_markdown(self) -> str:
        """Render the dossier to GitHub-flavored Markdown."""
        from geneuclidean.cns.credibility import render_credibility_markdown

        return render_credibility_markdown(self.payload)

    def save(
        self,
        json_path: str | Path,
        *,
        md_path: str | Path | None = None,
    ) -> tuple[Path, Path | None]:
        """Persist dossier to JSON (and optionally Markdown)."""
        from geneuclidean.cns.credibility import export_credibility_dossier

        return export_credibility_dossier(self.payload, json_path=json_path, md_path=md_path)


def build_credibility_dossier(
    *,
    model_name: str = "GenEuclidean CNS Discovery & ADMET Suite",
    model_version: str = "3.1.0",
    question_of_interest: str | None = None,
    context_of_use: str | None = None,
    decision_consequence: str | None = None,
    model_influence: str | None = None,
    risk_tier: str | None = None,
    admet_cv_metrics: Mapping[str, Any] | None = None,
    conformal_uq_metrics: Mapping[str, Any] | None = None,
    applicability_domain_summary: Mapping[str, Any] | None = None,
    physics_gate1_metrics: Mapping[str, Any] | None = None,
    provenance_files: Mapping[str, str | Path] | None = None,
    deviations_and_limitations: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> CredibilityDossier:
    """Build an FDA Jan 2025 (``FDA-2024-D-4689``) / EMA Jan 2026 7-Step Credibility Dossier.

    Delegates directly to :func:`geneuclidean.cns.credibility.build_credibility_dossier`
    and wraps the resulting schema in :class:`CredibilityDossier` for ergonomic JSON and
    Markdown export.
    """
    from geneuclidean.cns.credibility import build_credibility_dossier as _build

    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "admet_cv_metrics": admet_cv_metrics,
        "conformal_uq_metrics": conformal_uq_metrics,
        "applicability_domain_summary": applicability_domain_summary,
        "physics_gate1_metrics": physics_gate1_metrics,
        "provenance_files": provenance_files,
        "deviations_and_limitations": deviations_and_limitations,
        "generated_at_utc": generated_at_utc,
    }
    if question_of_interest is not None:
        kwargs["question_of_interest"] = question_of_interest
    if context_of_use is not None:
        kwargs["context_of_use"] = context_of_use
    if decision_consequence is not None:
        kwargs["decision_consequence"] = decision_consequence
    if model_influence is not None:
        kwargs["model_influence"] = model_influence
    if risk_tier is not None:
        kwargs["risk_tier"] = risk_tier

    return CredibilityDossier(payload=_build(**kwargs))


def export_credibility_json(
    dossier: CredibilityDossier | Mapping[str, Any],
    out_path: str | Path,
) -> Path:
    """Write a :class:`CredibilityDossier` (or raw dossier dict) to a JSON file."""
    payload = dossier.to_dict() if isinstance(dossier, CredibilityDossier) else dict(dossier)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def export_credibility_markdown(
    dossier: CredibilityDossier | Mapping[str, Any],
    out_path: str | Path,
) -> Path:
    """Render and write a :class:`CredibilityDossier` (or raw dict) to a Markdown file."""
    from geneuclidean.cns.credibility import render_credibility_markdown

    md_text = (
        dossier.to_markdown()
        if isinstance(dossier, CredibilityDossier)
        else render_credibility_markdown(dossier)
    )
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md_text, encoding="utf-8")
    return dest


__all__ = [
    "CredibilityDossier",
    "DesignResult",
    "MoleculeDesigner",
    "MoleculeReport",
    "Pocket",
    "build_credibility_dossier",
    "dock_and_profile",
    "export_credibility_json",
    "export_credibility_markdown",
    "get_target",
    "list_backbones",
    "list_targets",
    "protonate_ph74",
    "redock_complex",
    "score_molecules",
    "triage_molecule",
    "triage_molecules",
    "visualize_complex",
    "visualize_gallery",
    "visualize_molecules_2d",
]
