"""Open-Weights Co-Folding Hub & Interop Runners for GenEuclidean.

Provides a strictly zero-PyTorch-on-import :data:`FOLDING_BACKEND_REGISTRY` uniting
five open-weights & standard biomolecular folding/co-folding ecosystems behind one
immutable :class:`FoldedComplexResult` contract directly wired into
:class:`~geneuclidean.audit.PoseRealityAuditor`:

- ``"nanofold3"``: In-process MIT-licensed Mini-AlphaFold3 (lazy-imported only when run).
- ``"boltz2"``: MIT Boltz-2 YAML v1 exporter, optional CLI runner, and mmCIF/JSON parser.
- ``"protenix"``: Apache-2.0 ByteDance Protenix JSON exporter, optional CLI runner, and parser.
- ``"alphafold3"``: Official AlphaFold 3 JSON dialect (v2/v3/v4) exporter & output parser.
- ``"esmfold_api"``: Single-sequence zero-MSA protein folding (local ``transformers`` or
  strictly opt-in Meta ESM Atlas REST API for ``<= 400 aa``, ``allow_network=False`` default).

Security invariants:
- Zero ``shell=True`` subprocess invocations, zero ``eval``/``exec``.
- Strict job-name, chain-ID, amino-acid, SMILES, CCD, and path sanitization.
- Offline-safe by default (``allow_network=False`` blocks any remote HTTP calls).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from geneuclidean.exceptions import GenEuclideanError, OptionalDependencyError
from geneuclidean.registry import Registry

if TYPE_CHECKING:
    from geneuclidean.audit.pose_auditor import PoseAuditReport
    from geneuclidean.interop.boltz_adapter import ConfidenceSummary

# ---------------------------------------------------------------------------
# Exceptions & Sanitization Constants
# ---------------------------------------------------------------------------

_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")
_CCD_CODE_RE = re.compile(r"^[A-Z0-9]{1,5}$")
_ATOM_NAME_RE = re.compile(r"^[A-Za-z0-9'_+-]{1,8}$")
_STANDARD_AA_SET: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
_AMBIGUOUS_AA_SET: frozenset[str] = _STANDARD_AA_SET | frozenset("BJOXUZ")
_DNA_SET: frozenset[str] = frozenset("ACGNT")
_RNA_SET: frozenset[str] = frozenset("ACGNU")
_SMILES_ALLOWED_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\/=#%.:*~]{1,4096}$")
_ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
_ESMFOLD_MAX_API_AA = 400
_VALID_AF3_VERSIONS: frozenset[int] = frozenset({2, 3, 4})


class MissingBackendError(OptionalDependencyError):
    """Raised when an external folding CLI or optional ML backend is not installed."""


class NetworkDisabledError(GenEuclideanError, RuntimeError):
    """Raised when a remote folding API call is attempted with ``allow_network=False``."""


def _sanitize_job_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or ".." in cleaned or "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
        raise ValueError(
            f"Invalid folding job name {name!r}: must be a safe identifier without path separators."
        )
    if not _JOB_NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid folding job name {name!r}: must start with alphanumeric and match "
            "[A-Za-z0-9_.-] (max 128 chars)."
        )
    return cleaned


def _sanitize_chain_id(chain_id: str) -> str:
    cleaned = chain_id.strip()
    if not _CHAIN_ID_RE.match(cleaned):
        raise ValueError(
            f"Invalid chain ID {chain_id!r}: expected 1-4 alphanumeric characters (e.g. 'A', 'B')."
        )
    return cleaned


def _sanitize_protein_sequence(sequence: str, *, allow_ambiguous: bool = False) -> str:
    if "\x00" in sequence:
        raise ValueError("Protein sequence must not contain NUL bytes.")
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("Protein sequence must be non-empty.")
    if len(cleaned) > 10_000:
        raise ValueError(f"Protein sequence length {len(cleaned)} exceeds safety cap (10,000 aa).")
    allowed = _AMBIGUOUS_AA_SET if allow_ambiguous else _STANDARD_AA_SET
    bad = sorted(set(cleaned) - allowed)
    if bad:
        raise ValueError(f"Invalid amino acid characters in protein sequence: {''.join(bad)!r}.")
    return cleaned


def _sanitize_nucleic_sequence(sequence: str, *, kind: Literal["dna", "rna"]) -> str:
    if "\x00" in sequence:
        raise ValueError(f"{kind.upper()} sequence must not contain NUL bytes.")
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError(f"{kind.upper()} sequence must be non-empty.")
    allowed = _DNA_SET if kind == "dna" else _RNA_SET
    bad = sorted(set(cleaned) - allowed)
    if bad:
        raise ValueError(f"Invalid characters in {kind.upper()} sequence: {''.join(bad)!r}.")
    return cleaned


def _sanitize_smiles(smiles: str) -> str:
    if "\x00" in smiles or "\n" in smiles or "\r" in smiles:
        raise ValueError("Ligand SMILES must be a single-line string without control characters.")
    cleaned = smiles.strip()
    if not cleaned or not _SMILES_ALLOWED_RE.match(cleaned):
        raise ValueError(
            f"Invalid or unsafe ligand SMILES {smiles!r}: "
            "contains disallowed characters or is empty."
        )
    return cleaned


def _sanitize_atom_name(atom_name: str) -> str:
    if any(ch in atom_name for ch in ("\x00", "\n", "\r", "\t")):
        raise ValueError(
            f"Invalid atom name {atom_name!r}: must not contain control or newline characters."
        )
    cleaned = atom_name.strip()
    if not cleaned or not _ATOM_NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid atom name {atom_name!r}: expected 1-8 safe PDB/mmCIF atom characters."
        )
    return cleaned


def _sanitize_bonded_endpoint(endpoint: tuple[str, int, str]) -> tuple[str, int, str]:
    cid = _sanitize_chain_id(endpoint[0])
    res_idx = int(endpoint[1])
    if res_idx < 1:
        raise ValueError(f"Bonded atom residue index must be >= 1, got {res_idx}.")
    atom_name = _sanitize_atom_name(endpoint[2])
    return (cid, res_idx, atom_name)


def _sanitize_ccd_code(ccd: str) -> str:
    cleaned = ccd.strip().upper()
    if not _CCD_CODE_RE.match(cleaned):
        raise ValueError(
            f"Invalid PDB CCD code {ccd!r}: expected 1-5 uppercase alphanumeric characters."
        )
    return cleaned


def _sanitize_path(path: str | Path) -> Path:
    raw = str(path)
    if "\x00" in raw:
        raise ValueError("File path must not contain NUL bytes.")
    return Path(raw).expanduser().resolve()


def _compute_provenance_sha256(
    *,
    backend: str,
    name: str,
    structure_text: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(f"backend={backend}\nname={name}\n".encode())
    if structure_text:
        h.update(structure_text.encode("utf-8"))
    if metadata:
        with_safe = json.dumps(metadata, sort_keys=True, default=str)
        h.update(with_safe.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core Data Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldingJobSpec:
    """Backend-agnostic biomolecular folding / co-folding request specification."""

    name: str = "geneuclidean_job"
    protein_chains: tuple[tuple[str, str], ...] = ()
    dna_chains: tuple[tuple[str, str], ...] = ()
    rna_chains: tuple[tuple[str, str], ...] = ()
    ligand_smiles: tuple[tuple[str, str], ...] = ()
    ligand_ccds: tuple[tuple[str, str], ...] = ()
    pocket_pdb: str | Path | None = None
    pocket_contact_residues: tuple[tuple[str, int], ...] = ()
    pocket_binder_chain: str | None = None
    bonded_atom_pairs: tuple[tuple[tuple[str, int, str], tuple[str, int, str]], ...] = ()
    msa_paths: Mapping[str, str | Path] = field(default_factory=dict)
    seeds: tuple[int, ...] = (42,)
    num_samples: int = 1
    predict_affinity: bool = False
    affinity_binder_chain: str | None = None
    user_ccd: str | None = None

    def validated(self, *, allow_ambiguous_aa: bool = False) -> FoldingJobSpec:
        """Return a sanitized, validated copy of this job spec."""
        clean_name = _sanitize_job_name(self.name)
        seen_chains: set[str] = set()

        def _claim(cid: str) -> str:
            c = _sanitize_chain_id(cid)
            if c in seen_chains:
                raise ValueError(f"Duplicate chain ID {c!r} in FoldingJobSpec.")
            seen_chains.add(c)
            return c

        clean_prot = tuple(
            (_claim(cid), _sanitize_protein_sequence(seq, allow_ambiguous=allow_ambiguous_aa))
            for cid, seq in self.protein_chains
        )
        clean_dna = tuple(
            (_claim(cid), _sanitize_nucleic_sequence(seq, kind="dna"))
            for cid, seq in self.dna_chains
        )
        clean_rna = tuple(
            (_claim(cid), _sanitize_nucleic_sequence(seq, kind="rna"))
            for cid, seq in self.rna_chains
        )
        clean_smi = tuple((_claim(cid), _sanitize_smiles(smi)) for cid, smi in self.ligand_smiles)
        clean_ccd = tuple((_claim(cid), _sanitize_ccd_code(ccd)) for cid, ccd in self.ligand_ccds)

        clean_seeds = tuple(int(s) for s in self.seeds)
        if not clean_seeds or any(s < 0 for s in clean_seeds):
            raise ValueError("FoldingJobSpec.seeds must contain one or more non-negative integers.")
        if int(self.num_samples) < 1:
            raise ValueError("FoldingJobSpec.num_samples must be >= 1.")

        clean_contacts = tuple(
            (_sanitize_chain_id(cid), int(res_idx)) for cid, res_idx in self.pocket_contact_residues
        )
        for _, res_idx in clean_contacts:
            if res_idx < 1:
                raise ValueError(f"Residue index must be >= 1, got {res_idx}.")

        clean_bonds = tuple(
            (_sanitize_bonded_endpoint(a1), _sanitize_bonded_endpoint(a2))
            for a1, a2 in self.bonded_atom_pairs
        )
        if self.pocket_pdb is not None and "\x00" in str(self.pocket_pdb):
            raise ValueError("FoldingJobSpec.pocket_pdb must not contain NUL bytes.")
        if self.user_ccd is not None and "\x00" in self.user_ccd:
            raise ValueError("FoldingJobSpec.user_ccd must not contain NUL bytes.")
        clean_msa = {
            _sanitize_chain_id(cid): str(_sanitize_path(p)) for cid, p in self.msa_paths.items()
        }
        clean_binder = (
            _sanitize_chain_id(self.pocket_binder_chain)
            if self.pocket_binder_chain is not None
            else None
        )
        clean_aff_binder = (
            _sanitize_chain_id(self.affinity_binder_chain)
            if self.affinity_binder_chain is not None
            else None
        )
        return FoldingJobSpec(
            name=clean_name,
            protein_chains=clean_prot,
            dna_chains=clean_dna,
            rna_chains=clean_rna,
            ligand_smiles=clean_smi,
            ligand_ccds=clean_ccd,
            pocket_pdb=self.pocket_pdb,
            pocket_contact_residues=clean_contacts,
            pocket_binder_chain=clean_binder,
            bonded_atom_pairs=clean_bonds,
            msa_paths=clean_msa,
            seeds=clean_seeds,
            num_samples=int(self.num_samples),
            predict_affinity=bool(self.predict_affinity),
            affinity_binder_chain=clean_aff_binder,
            user_ccd=self.user_ccd,
        )

    @property
    def primary_ligand_chain(self) -> str | None:
        if self.ligand_smiles:
            return self.ligand_smiles[0][0]
        if self.ligand_ccds:
            return self.ligand_ccds[0][0]
        return None

    @property
    def primary_ligand_smiles(self) -> str | None:
        return self.ligand_smiles[0][1] if self.ligand_smiles else None


@dataclass(frozen=True)
class FoldedComplexResult:
    """Unified immutable output contract across NanoFold3, Boltz-2, Protenix, AF3 & ESMFold."""

    backend: str
    name: str = "complex"
    pdb_text: str | None = None
    cif_text: str | None = None
    structure_path: Path | None = None
    ligand_mol_3d: Any | None = None
    pocket_pdb_text: str | None = None
    ligand_smiles: str | None = None
    plddt_mean: float = float("nan")
    plddt_per_atom: tuple[float, ...] = ()
    ligand_plddt: float = float("nan")
    ptm: float = float("nan")
    iptm: float = float("nan")
    ranking_score: float = float("nan")
    pae_matrix: Any | None = None
    pde_matrix: Any | None = None
    affinity_pred: float | None = None
    predicted_pkd: float | None = None
    binding_prob: float | None = None
    runtime_sec: float = 0.0
    pose_audit: PoseAuditReport | None = None
    confidence_summary: ConfidenceSummary | None = None
    provenance_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_plddt(self) -> float:
        """Alias for ``plddt_mean`` (design-spec compatibility)."""
        return self.plddt_mean

    @property
    def audit_report(self) -> PoseAuditReport | None:
        """Alias for ``pose_audit`` (design-spec compatibility)."""
        return self.pose_audit

    @property
    def sha256_provenance(self) -> str:
        """Alias for ``provenance_sha256`` (design-spec compatibility)."""
        return self.provenance_sha256

    def run_audit(
        self,
        *,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        ligand_resname: str | None = None,
        pocket_cutoff: float = 10.0,
    ) -> PoseAuditReport:
        """Run the 7-axis Beyond-PoseBusters :class:`PoseRealityAuditor` on this complex."""
        from geneuclidean.audit.pose_auditor import PoseRealityAuditor
        from geneuclidean.interop.boltz_adapter import audit_cofolded_complex

        if self.ligand_mol_3d is not None:
            auditor = PoseRealityAuditor(property_profile=property_profile)
            return auditor.audit(
                self.ligand_mol_3d,
                pocket_pdb=self.pocket_pdb_text or self.pdb_text,
                name=self.name,
                canonical_anchors=canonical_anchors,
            )
        if self.structure_path is not None and Path(self.structure_path).is_file():
            return audit_cofolded_complex(
                self.structure_path,
                ligand_resname=ligand_resname,
                reference_smiles=self.ligand_smiles,
                property_profile=property_profile,
                canonical_anchors=canonical_anchors,
                pocket_cutoff=pocket_cutoff,
            )
        if self.pdb_text or self.cif_text:
            import tempfile

            suffix = ".cif" if (self.cif_text and not self.pdb_text) else ".pdb"
            content = self.pdb_text if self.pdb_text else (self.cif_text or "")
            with tempfile.NamedTemporaryFile(
                "w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tf:
                tf.write(content)
                tmp_path = Path(tf.name)
            try:
                return audit_cofolded_complex(
                    tmp_path,
                    ligand_resname=ligand_resname,
                    reference_smiles=self.ligand_smiles,
                    property_profile=property_profile,
                    canonical_anchors=canonical_anchors,
                    pocket_cutoff=pocket_cutoff,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        raise ValueError("FoldedComplexResult contains no 3D structure or ligand_mol_3d to audit.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize result metadata and metrics to a JSON-friendly dictionary."""

        def _clean_float(x: float | None) -> float | None:
            if x is None:
                return None
            return round(float(x), 4) if math.isfinite(float(x)) else None

        return {
            "backend": self.backend,
            "name": self.name,
            "structure_path": str(self.structure_path) if self.structure_path else None,
            "ligand_smiles": self.ligand_smiles,
            "plddt_mean": _clean_float(self.plddt_mean),
            "ligand_plddt": _clean_float(self.ligand_plddt),
            "num_plddt_atoms": len(self.plddt_per_atom),
            "ptm": _clean_float(self.ptm),
            "iptm": _clean_float(self.iptm),
            "ranking_score": _clean_float(self.ranking_score),
            "affinity_pred": _clean_float(self.affinity_pred),
            "predicted_pkd": _clean_float(self.predicted_pkd),
            "binding_prob": _clean_float(self.binding_prob),
            "runtime_sec": round(float(self.runtime_sec), 4),
            "provenance_sha256": self.provenance_sha256,
            "pose_audit": self.pose_audit.to_dict() if self.pose_audit is not None else None,
            "confidence_summary": (
                self.confidence_summary.to_dict() if self.confidence_summary is not None else None
            ),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Zero-Torch Schema Exporters (AlphaFold 3 JSON, Boltz-2 YAML, Protenix JSON)
# ---------------------------------------------------------------------------


def _coerce_job_spec(
    job: FoldingJobSpec | None,
    *,
    name: str,
    protein_sequence: str | None = None,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
    pocket_pdb: str | Path | None = None,
    seeds: Sequence[int] = (42,),
    num_samples: int = 1,
    predict_affinity: bool = False,
) -> FoldingJobSpec:
    if job is not None:
        return job.validated()
    prot_chains: tuple[tuple[str, str], ...] = (
        (("A", protein_sequence),) if protein_sequence is not None else ()
    )
    next_lig_cid = "B" if prot_chains else "A"
    lig_smi: tuple[tuple[str, str], ...] = (
        ((next_lig_cid, ligand_smiles),) if ligand_smiles is not None else ()
    )
    ccd_cid = "C" if (prot_chains and lig_smi) else ("B" if (prot_chains or lig_smi) else "A")
    lig_ccd: tuple[tuple[str, str], ...] = (
        ((ccd_cid, ligand_ccd),) if ligand_ccd is not None else ()
    )
    return FoldingJobSpec(
        name=name,
        protein_chains=prot_chains,
        ligand_smiles=lig_smi,
        ligand_ccds=lig_ccd,
        pocket_pdb=pocket_pdb,
        seeds=tuple(int(s) for s in seeds),
        num_samples=num_samples,
        predict_affinity=predict_affinity,
    ).validated()


def export_alphafold3_json(
    job: FoldingJobSpec | None = None,
    out_path: str | Path | None = None,
    *,
    dialect_version: int = 2,
    name: str = "geneuclidean_af3",
    protein_sequence: str | None = None,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
    seeds: Sequence[int] = (42,),
) -> dict[str, Any]:
    """Export an official AlphaFold 3 JSON input payload (dialect version 2, 3, or 4).

    Executes with zero PyTorch or RDKit imports. Optionally writes pretty-printed JSON
    to ``out_path`` when provided.
    """
    if int(dialect_version) not in _VALID_AF3_VERSIONS:
        raise ValueError(
            f"Unsupported AlphaFold 3 JSON dialect version {dialect_version!r}; "
            f"expected one of {sorted(_VALID_AF3_VERSIONS)}."
        )
    spec = _coerce_job_spec(
        job,
        name=name,
        protein_sequence=protein_sequence,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
        seeds=seeds,
    )
    sequences: list[dict[str, Any]] = []
    for cid, seq in spec.protein_chains:
        p_entry: dict[str, Any] = {"id": cid, "sequence": seq}
        if cid in spec.msa_paths:
            p_entry["unpairedMsaPath"] = str(spec.msa_paths[cid])
        sequences.append({"protein": p_entry})
    for cid, seq in spec.dna_chains:
        sequences.append({"dna": {"id": cid, "sequence": seq}})
    for cid, seq in spec.rna_chains:
        sequences.append({"rna": {"id": cid, "sequence": seq}})
    for cid, smi in spec.ligand_smiles:
        sequences.append({"ligand": {"id": cid, "smiles": smi}})
    for cid, ccd in spec.ligand_ccds:
        sequences.append({"ligand": {"id": cid, "ccdCodes": [ccd]}})

    if not sequences:
        raise ValueError(
            "AlphaFold 3 JSON export requires at least one protein, nucleic, or ligand entity."
        )

    payload: dict[str, Any] = {
        "name": spec.name,
        "modelSeeds": list(spec.seeds),
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": int(dialect_version),
    }
    if spec.bonded_atom_pairs:
        payload["bondedAtomPairs"] = [
            [[a1[0], a1[1], a1[2]], [a2[0], a2[1], a2[2]]] for a1, a2 in spec.bonded_atom_pairs
        ]
    if spec.user_ccd:
        payload["userCCD"] = spec.user_ccd

    if out_path is not None:
        dest = _sanitize_path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _yaml_quote_str(value: str) -> str:
    """Safely double-quote any YAML scalar string using JSON string escaping (strict YAML 1.2)."""
    return json.dumps(value)


def _render_boltz2_yaml_stdlib(doc: Mapping[str, Any]) -> str:
    """Zero-dependency strict YAML 1.2 serializer for Boltz-2 version 1 input documents."""
    lines: list[str] = [f"version: {int(doc['version'])}", "sequences:"]
    for item in doc["sequences"]:
        if "protein" in item:
            p = item["protein"]
            lines.append("  - protein:")
            lines.append(f"      id: {_yaml_quote_str(str(p['id']))}")
            lines.append(f"      sequence: {_yaml_quote_str(p['sequence'])}")
            if "msa" in p:
                lines.append(f"      msa: {_yaml_quote_str(str(p['msa']))}")
        elif "dna" in item:
            d = item["dna"]
            lines.append("  - dna:")
            lines.append(f"      id: {_yaml_quote_str(str(d['id']))}")
            lines.append(f"      sequence: {_yaml_quote_str(d['sequence'])}")
        elif "rna" in item:
            r = item["rna"]
            lines.append("  - rna:")
            lines.append(f"      id: {_yaml_quote_str(str(r['id']))}")
            lines.append(f"      sequence: {_yaml_quote_str(r['sequence'])}")
        elif "ligand" in item:
            lig = item["ligand"]
            lines.append("  - ligand:")
            lines.append(f"      id: {_yaml_quote_str(str(lig['id']))}")
            if "smiles" in lig:
                lines.append(f"      smiles: {_yaml_quote_str(lig['smiles'])}")
            elif "ccd" in lig:
                lines.append(f"      ccd: {_yaml_quote_str(lig['ccd'])}")

    constraints = doc.get("constraints")
    if constraints:
        lines.append("constraints:")
        for c in constraints:
            if "pocket" in c:
                poc = c["pocket"]
                lines.append("  - pocket:")
                lines.append(f"      binder: {_yaml_quote_str(str(poc['binder']))}")
                lines.append("      contacts:")
                for chain_id, res_idx in poc["contacts"]:
                    lines.append(f"        - [{_yaml_quote_str(str(chain_id))}, {int(res_idx)}]")
            elif "bond" in c:
                b = c["bond"]
                a1 = b["atom1"]
                a2 = b["atom2"]
                lines.append("  - bond:")
                lines.append(
                    f"      atom1: [{_yaml_quote_str(str(a1[0]))}, {int(a1[1])}, "
                    f"{_yaml_quote_str(str(a1[2]))}]"
                )
                lines.append(
                    f"      atom2: [{_yaml_quote_str(str(a2[0]))}, {int(a2[1])}, "
                    f"{_yaml_quote_str(str(a2[2]))}]"
                )

    properties = doc.get("properties")
    if properties:
        lines.append("properties:")
        for prop in properties:
            if "affinity" in prop:
                lines.append("  - affinity:")
                lines.append(f"      binder: {_yaml_quote_str(str(prop['affinity']['binder']))}")

    return "\n".join(lines) + "\n"


def export_boltz2_yaml(
    job: FoldingJobSpec | None = None,
    out_path: str | Path | None = None,
    *,
    name: str = "geneuclidean_boltz2",
    protein_sequence: str | None = None,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
    predict_affinity: bool = False,
) -> str:
    """Export a strict Boltz-2 ``version: 1`` YAML specification string (and optional file).

    Uses a pure-stdlib YAML 1.2 emitter so cold import and schema generation require zero
    third-party dependencies while guaranteeing safe quoting of arbitrary SMILES strings.
    """
    spec = _coerce_job_spec(
        job,
        name=name,
        protein_sequence=protein_sequence,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
        predict_affinity=predict_affinity,
    )
    sequences: list[dict[str, Any]] = []
    for cid, seq in spec.protein_chains:
        p_entry: dict[str, Any] = {"id": cid, "sequence": seq}
        if cid in spec.msa_paths:
            p_entry["msa"] = str(spec.msa_paths[cid])
        sequences.append({"protein": p_entry})
    for cid, seq in spec.dna_chains:
        sequences.append({"dna": {"id": cid, "sequence": seq}})
    for cid, seq in spec.rna_chains:
        sequences.append({"rna": {"id": cid, "sequence": seq}})
    for cid, smi in spec.ligand_smiles:
        sequences.append({"ligand": {"id": cid, "smiles": smi}})
    for cid, ccd in spec.ligand_ccds:
        sequences.append({"ligand": {"id": cid, "ccd": ccd}})

    if not sequences:
        raise ValueError("Boltz-2 YAML export requires at least one polymer or ligand sequence.")

    doc: dict[str, Any] = {"version": 1, "sequences": sequences}
    constraints: list[dict[str, Any]] = []
    if spec.pocket_contact_residues:
        binder = spec.pocket_binder_chain or spec.primary_ligand_chain
        if binder is None:
            raise ValueError(
                "Boltz-2 pocket constraints require a ligand entity "
                "or explicit pocket_binder_chain."
            )
        constraints.append(
            {
                "pocket": {
                    "binder": binder,
                    "contacts": [[cid, idx] for cid, idx in spec.pocket_contact_residues],
                }
            }
        )
    for a1, a2 in spec.bonded_atom_pairs:
        constraints.append(
            {
                "bond": {
                    "atom1": [a1[0], a1[1], a1[2]],
                    "atom2": [a2[0], a2[1], a2[2]],
                }
            }
        )
    if constraints:
        doc["constraints"] = constraints

    if spec.predict_affinity:
        aff_binder = spec.affinity_binder_chain or spec.primary_ligand_chain
        if aff_binder is None:
            raise ValueError("Boltz-2 affinity prediction requires at least one ligand entity.")
        doc["properties"] = [{"affinity": {"binder": aff_binder}}]

    yaml_text = _render_boltz2_yaml_stdlib(doc)
    if out_path is not None:
        dest = _sanitize_path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml_text, encoding="utf-8")
    return yaml_text


def export_protenix_json(
    job: FoldingJobSpec | None = None,
    out_path: str | Path | None = None,
    *,
    name: str = "geneuclidean_protenix",
    protein_sequence: str | None = None,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
) -> list[dict[str, Any]]:
    """Export an Apache-2.0 ByteDance Protenix input JSON array (and optional file)."""
    spec = _coerce_job_spec(
        job,
        name=name,
        protein_sequence=protein_sequence,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
    )
    sequences: list[dict[str, Any]] = []
    entity_index_by_chain: dict[str, int] = {}

    for cid, seq in spec.protein_chains:
        p_entry: dict[str, Any] = {"sequence": seq, "count": 1}
        if cid in spec.msa_paths:
            p_entry["msa"] = {
                "precomputed_msa_dir": str(spec.msa_paths[cid]),
                "pairing_db": "uniref100",
            }
        sequences.append({"proteinChain": p_entry})
        entity_index_by_chain[cid] = len(sequences)
    for cid, seq in spec.dna_chains:
        sequences.append({"dnaSequence": {"sequence": seq, "count": 1}})
        entity_index_by_chain[cid] = len(sequences)
    for cid, seq in spec.rna_chains:
        sequences.append({"rnaSequence": {"sequence": seq, "count": 1}})
        entity_index_by_chain[cid] = len(sequences)
    for cid, smi in spec.ligand_smiles:
        sequences.append({"ligand": {"ligand": smi, "count": 1}})
        entity_index_by_chain[cid] = len(sequences)
    for cid, ccd in spec.ligand_ccds:
        sequences.append({"ligand": {"ligand": f"CCD_{ccd}", "count": 1}})
        entity_index_by_chain[cid] = len(sequences)

    if not sequences:
        raise ValueError("Protenix JSON export requires at least one entity.")

    entry: dict[str, Any] = {
        "name": spec.name,
        "sequences": sequences,
    }
    if spec.bonded_atom_pairs:
        cov_bonds: list[dict[str, Any]] = []
        for a1, a2 in spec.bonded_atom_pairs:
            if a1[0] not in entity_index_by_chain or a2[0] not in entity_index_by_chain:
                raise ValueError(f"Covalent bond references unknown chain ID in {(a1, a2)!r}.")
            cov_bonds.append(
                {
                    "entity1": entity_index_by_chain[a1[0]],
                    "copy1": 1,
                    "position1": a1[1],
                    "atom1": a1[2],
                    "entity2": entity_index_by_chain[a2[0]],
                    "copy2": 1,
                    "position2": a2[1],
                    "atom2": a2[2],
                }
            )
        entry["covalent_bonds"] = cov_bonds

    payload = [entry]
    if out_path is not None:
        dest = _sanitize_path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Shared Output Directory Parser Helper (Bridged to boltz_adapter + PoseRealityAuditor)
# ---------------------------------------------------------------------------


def _find_first_matching(
    root: Path,
    patterns: Sequence[str],
    *,
    exclude: set[Path] | None = None,
    exclude_name_substrings: Sequence[str] = (),
) -> Path | None:
    excluded = {p.resolve() for p in exclude} if exclude else set()
    excl_subs = tuple(s.lower() for s in exclude_name_substrings)
    if root.is_file():
        return None if root.resolve() in excluded else root
    for pat in patterns:
        matches = sorted(
            p
            for p in root.rglob(pat)
            if p.is_file()
            and p.resolve() not in excluded
            and not any(sub in p.name.lower() for sub in excl_subs)
        )
        if matches:
            return matches[0]
    return None


def _parse_external_structure_output(
    *,
    backend: str,
    name: str,
    output_dir: str | Path,
    structure_patterns: Sequence[str],
    summary_json_patterns: Sequence[str],
    confidences_json_patterns: Sequence[str] = (),
    affinity_json_patterns: Sequence[str] = (),
    ligand_resname: str | None = None,
    reference_smiles: str | None = None,
    pocket_cutoff: float = 10.0,
    run_audit: bool = False,
    property_profile: str = "oral_ro5",
    canonical_anchors: Sequence[str] = (),
    runtime_sec: float = 0.0,
    extra_metadata: Mapping[str, Any] | None = None,
) -> FoldedComplexResult:
    from geneuclidean.interop.boltz_adapter import (
        extract_bfactor_plddt,
        load_cofold_confidence_json,
        load_cofolded_complex,
    )

    out_root = _sanitize_path(output_dir)
    if not out_root.exists():
        raise FileNotFoundError(f"Folding output path does not exist: {out_root}")

    struct_path = (
        out_root if out_root.is_file() else _find_first_matching(out_root, structure_patterns)
    )
    if struct_path is None:
        raise FileNotFoundError(
            f"No predicted structure ({', '.join(structure_patterns)}) found in {out_root}."
        )

    search_dir = struct_path.parent if out_root.is_file() else out_root
    summary_json = _find_first_matching(search_dir, summary_json_patterns)
    conf_exclude = {summary_json} if summary_json is not None else set()
    conf_json = (
        _find_first_matching(
            search_dir,
            confidences_json_patterns,
            exclude=conf_exclude,
            exclude_name_substrings=("summary_confidences", "summary_confidence"),
        )
        if confidences_json_patterns
        else None
    )
    aff_exclude = {p for p in (summary_json, conf_json) if p is not None}
    aff_json = (
        _find_first_matching(search_dir, affinity_json_patterns, exclude=aff_exclude)
        if affinity_json_patterns
        else None
    )

    raw_text = struct_path.read_text(encoding="utf-8")
    is_cif = struct_path.suffix.lower() in (".cif", ".mmcif")
    bfactor_map = extract_bfactor_plddt(struct_path, ligand_resname=ligand_resname)
    all_plddt = bfactor_map["all"]
    lig_plddt_arr = bfactor_map["ligand"]

    if summary_json is not None:
        conf_summary = load_cofold_confidence_json(
            summary_json,
            confidences_path=conf_json,
            affinity_path=aff_json,
        )
    elif conf_json is not None:
        conf_summary = load_cofold_confidence_json(conf_json, affinity_path=aff_json)
    elif aff_json is not None:
        conf_summary = load_cofold_confidence_json(aff_json)
    else:
        conf_summary = None

    # Prefer full atom pLDDTs from AF3 confidences.json if present, else B-factors from structure
    if conf_summary is not None and conf_summary.atom_plddts:
        plddt_per_atom = conf_summary.atom_plddts
    else:
        plddt_per_atom = tuple(float(x) for x in all_plddt.tolist())

    if conf_summary is not None and conf_summary.plddt_mean is not None:
        plddt_mean = conf_summary.plddt_mean
    elif plddt_per_atom:
        plddt_mean = float(sum(plddt_per_atom) / len(plddt_per_atom))
    else:
        plddt_mean = float("nan")

    if conf_summary is not None and conf_summary.ligand_plddt is not None:
        ligand_plddt = conf_summary.ligand_plddt
    elif lig_plddt_arr.size > 0:
        ligand_plddt = float(lig_plddt_arr.mean())
    else:
        ligand_plddt = float("nan")

    # Attempt ligand + pocket extraction cleanly (optional for protein-only structures)
    lig_mol_3d: Any | None = None
    pocket_pdb_text: str | None = None
    resolved_smiles = reference_smiles
    try:
        lig_mol_3d, pocket_pdb_text = load_cofolded_complex(
            struct_path,
            ligand_resname=ligand_resname,
            reference_smiles=reference_smiles,
            pocket_cutoff=pocket_cutoff,
        )
        if resolved_smiles is None and lig_mol_3d is not None:
            from rdkit import Chem

            resolved_smiles = Chem.MolToSmiles(
                Chem.RemoveHs(Chem.Mol(lig_mol_3d)), canonical=True, isomericSmiles=True
            )
    except ValueError:
        # Protein-only fold (no small-molecule HETATM present)
        lig_mol_3d = None
        pocket_pdb_text = None

    meta: dict[str, Any] = {
        "structure_file": str(struct_path),
        "summary_json": str(summary_json) if summary_json else None,
        "confidences_json": str(conf_json) if conf_json else None,
        "affinity_json": str(aff_json) if aff_json else None,
        **(dict(extra_metadata) if extra_metadata else {}),
    }

    result = FoldedComplexResult(
        backend=backend,
        name=name,
        pdb_text=None if is_cif else raw_text,
        cif_text=raw_text if is_cif else None,
        structure_path=struct_path,
        ligand_mol_3d=lig_mol_3d,
        pocket_pdb_text=pocket_pdb_text,
        ligand_smiles=resolved_smiles,
        plddt_mean=plddt_mean,
        plddt_per_atom=plddt_per_atom,
        ligand_plddt=ligand_plddt,
        ptm=conf_summary.ptm if (conf_summary and conf_summary.ptm is not None) else float("nan"),
        iptm=conf_summary.iptm
        if (conf_summary and conf_summary.iptm is not None)
        else float("nan"),
        ranking_score=(
            conf_summary.ranking_score
            if (conf_summary and conf_summary.ranking_score is not None)
            else float("nan")
        ),
        pae_matrix=conf_summary.pae_matrix if conf_summary is not None else None,
        affinity_pred=conf_summary.affinity_pred_value if conf_summary is not None else None,
        predicted_pkd=conf_summary.predicted_pkd if conf_summary is not None else None,
        binding_prob=conf_summary.affinity_probability_binary if conf_summary is not None else None,
        runtime_sec=runtime_sec,
        confidence_summary=conf_summary,
        provenance_sha256=_compute_provenance_sha256(
            backend=backend,
            name=name,
            structure_text=raw_text,
            metadata=meta,
        ),
        metadata=meta,
    )

    if run_audit and lig_mol_3d is not None:
        audit_rep = result.run_audit(
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            ligand_resname=ligand_resname,
            pocket_cutoff=pocket_cutoff,
        )
        result = dataclasses.replace(result, pose_audit=audit_rep)
    elif run_audit and lig_mol_3d is None:
        result = dataclasses.replace(
            result,
            metadata={**result.metadata, "audit_skipped_reason": "protein_only_no_ligand"},
        )
    return result


# ---------------------------------------------------------------------------
# Backend Protocol & 5 Built-In Backends
# ---------------------------------------------------------------------------


@runtime_checkable
class FoldingBackend(Protocol):
    """Protocol implemented by pluggable folding/co-folding backends."""

    name: str

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        **kwargs: Any,
    ) -> FoldedComplexResult: ...


class BaseFoldingBackend:
    """Base class for built-in GenEuclidean folding backends."""

    name: str = "base"

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        **kwargs: Any,
    ) -> FoldedComplexResult:
        raise NotImplementedError


class NanoFold3Backend(BaseFoldingBackend):
    """In-process MIT-licensed NanoFold3 backend (strictly lazy-imports PyTorch on invocation)."""

    name = "nanofold3"

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        preset: Literal["nano", "small", "base"] = "nano",
        trunk_mode: Literal["pairformer", "simplefold", "hybrid"] = "pairformer",
        diffusion_mode: Literal["edm", "flow_matching"] = "edm",
        flex_mode: str = "flexible_pocket_and_ligand",
        num_steps: int | None = None,
        pocket_center: Sequence[float] | None = None,
        pocket_cutoff: float = 10.0,
        protonate_ph74: bool = True,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        model: Any | None = None,
        **kwargs: Any,
    ) -> FoldedComplexResult:
        del kwargs  # Reserved for forward compatibility
        t0 = time.perf_counter()
        spec = job.validated()

        if model is None:
            try:
                nf_mod = importlib.import_module("geneuclidean.folding.nanofold3")
            except ImportError as exc:
                raise MissingBackendError(
                    f"NanoFold3 backend could not be imported: {exc}",
                    hint="Ensure PyTorch is installed (`pip install torch`) and "
                    "geneuclidean.folding.nanofold3 is available.",
                ) from exc
            if checkpoint_path is not None:
                model = nf_mod.NanoFold3.from_pretrained(
                    _sanitize_path(checkpoint_path),
                    device=device or "auto",
                )
            else:
                model = nf_mod.build_nanofold3(
                    preset=preset,
                    trunk_mode=trunk_mode,
                    diffusion_mode=diffusion_mode,
                    device=device,
                )

        seed = int(spec.seeds[0])
        if spec.pocket_pdb is not None and spec.primary_ligand_smiles is not None:
            raw_res = model.cofold(
                spec.pocket_pdb,
                spec.primary_ligand_smiles,
                pocket_center=pocket_center,
                pocket_cutoff=pocket_cutoff,
                flex_mode=flex_mode,
                protonate_ph74=protonate_ph74,
                num_samples=spec.num_samples,
                num_steps=num_steps,
                seed=seed,
                run_audit=run_audit,
                property_profile=property_profile,
                canonical_anchors=canonical_anchors,
            )
        elif spec.protein_chains:
            raw_res = model.fold_sequence(
                spec.protein_chains[0][1],
                num_samples=spec.num_samples,
                num_steps=num_steps,
                seed=seed,
            )
        else:
            raise ValueError(
                "NanoFold3 backend requires either (pocket_pdb + ligand_smiles) for induced-fit "
                "co-folding or protein_chains for single-sequence mini-folding."
            )

        elapsed = time.perf_counter() - t0
        if isinstance(raw_res, FoldedComplexResult):
            result = raw_res
        else:
            # Adapt NanoFold3Result cleanly into canonical FoldedComplexResult
            pdb_text = getattr(raw_res, "pdb_text", None)
            meta = dict(getattr(raw_res, "metadata", {}) or {})
            meta.setdefault("preset", preset)
            meta.setdefault("trunk_mode", trunk_mode)
            meta.setdefault("diffusion_mode", diffusion_mode)
            prov = str(getattr(raw_res, "provenance_sha256", "") or "")
            if not prov:
                prov = _compute_provenance_sha256(
                    backend="nanofold3",
                    name=spec.name,
                    structure_text=pdb_text,
                    metadata=meta,
                )
            pred_pkd = getattr(raw_res, "predicted_pkd", None)
            if pred_pkd is not None and isinstance(pred_pkd, float) and not math.isfinite(pred_pkd):
                pred_pkd = None
            bind_prob = getattr(raw_res, "binding_prob", None)
            if (
                bind_prob is not None
                and isinstance(bind_prob, float)
                and not math.isfinite(bind_prob)
            ):
                bind_prob = None
            result = FoldedComplexResult(
                backend="nanofold3",
                name=str(getattr(raw_res, "name", spec.name) or spec.name),
                pdb_text=pdb_text,
                pocket_pdb_text=getattr(raw_res, "pocket_pdb_text", None),
                ligand_mol_3d=getattr(raw_res, "ligand_mol_3d", None),
                ligand_smiles=getattr(raw_res, "ligand_smiles", spec.primary_ligand_smiles),
                plddt_mean=float(getattr(raw_res, "plddt_mean", float("nan"))),
                plddt_per_atom=tuple(
                    float(x) for x in (getattr(raw_res, "plddt_per_atom", ()) or ())
                ),
                ligand_plddt=float(getattr(raw_res, "ligand_plddt", float("nan"))),
                ptm=float(getattr(raw_res, "ptm", float("nan"))),
                iptm=float(getattr(raw_res, "iptm", float("nan"))),
                ranking_score=float(getattr(raw_res, "ranking_score", float("nan"))),
                pae_matrix=getattr(raw_res, "pae_matrix", None),
                pde_matrix=getattr(raw_res, "pde_matrix", None),
                affinity_pred=pred_pkd,
                predicted_pkd=pred_pkd,
                binding_prob=bind_prob,
                runtime_sec=float(getattr(raw_res, "runtime_sec", elapsed) or elapsed),
                pose_audit=getattr(raw_res, "pose_audit", None),
                provenance_sha256=prov,
                metadata=meta,
            )

        if output_dir is not None and result.pdb_text:
            dest_dir = _sanitize_path(output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_pdb = dest_dir / f"{spec.name}_nanofold3.pdb"
            out_pdb.write_text(result.pdb_text, encoding="utf-8")
            result = dataclasses.replace(result, structure_path=out_pdb)
        return result


class Boltz2Backend(BaseFoldingBackend):
    """MIT Boltz-2 YAML v1 exporter, optional local CLI runner, and mmCIF/JSON parser."""

    name = "boltz2"

    def export_input(self, job: FoldingJobSpec, out_path: str | Path | None = None) -> str:
        return export_boltz2_yaml(job, out_path)

    def parse_output(
        self,
        output_dir: str | Path,
        *,
        name: str = "boltz2_complex",
        ligand_resname: str | None = None,
        reference_smiles: str | None = None,
        pocket_cutoff: float = 10.0,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        runtime_sec: float = 0.0,
    ) -> FoldedComplexResult:
        return _parse_external_structure_output(
            backend=self.name,
            name=name,
            output_dir=output_dir,
            structure_patterns=("*_model_0.cif", "*.cif", "*.mmcif", "*.pdb"),
            summary_json_patterns=("confidence_*_model_0.json", "confidence_*.json"),
            affinity_json_patterns=("affinity_*.json",),
            ligand_resname=ligand_resname,
            reference_smiles=reference_smiles,
            pocket_cutoff=pocket_cutoff,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            runtime_sec=runtime_sec,
        )

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        mode: Literal["run", "export_only", "parse_output"] = "run",
        executable: str = "boltz",
        ligand_resname: str | None = None,
        pocket_cutoff: float = 10.0,
        timeout_sec: float = 1800.0,
        extra_args: Sequence[str] = (),
        **kwargs: Any,
    ) -> FoldedComplexResult:
        del kwargs
        t0 = time.perf_counter()
        spec = job.validated()

        if mode == "parse_output":
            if output_dir is None:
                raise ValueError("Boltz2Backend(mode='parse_output') requires output_dir.")
            return self.parse_output(
                output_dir,
                name=spec.name,
                ligand_resname=ligand_resname,
                reference_smiles=spec.primary_ligand_smiles,
                pocket_cutoff=pocket_cutoff,
                run_audit=run_audit,
                property_profile=property_profile,
                canonical_anchors=canonical_anchors,
                runtime_sec=time.perf_counter() - t0,
            )

        if mode == "export_only":
            yaml_path = (
                _sanitize_path(output_dir) / f"{spec.name}.yaml" if output_dir is not None else None
            )
            yaml_text = self.export_input(spec, yaml_path)
            meta = {
                "mode": "export_only",
                "input_yaml": yaml_text,
                "input_path": str(yaml_path) if yaml_path else None,
            }
            return FoldedComplexResult(
                backend=self.name,
                name=spec.name,
                ligand_smiles=spec.primary_ligand_smiles,
                runtime_sec=time.perf_counter() - t0,
                provenance_sha256=_compute_provenance_sha256(
                    backend=self.name, name=spec.name, structure_text=yaml_text, metadata=meta
                ),
                metadata=meta,
            )

        import shutil
        import subprocess
        import tempfile

        exe_path = shutil.which(executable)
        if exe_path is None:
            raise MissingBackendError(
                f"Boltz-2 CLI executable {executable!r} is not installed or not found on PATH.",
                hint="Install Boltz-2 with `pip install boltz` (MIT license), or use "
                "mode='export_only' / mode='parse_output' for offline job preparation and parsing.",
            )

        work_dir = (
            _sanitize_path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix=f"geneuclidean_boltz2_{spec.name}_"))
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        yaml_file = work_dir / f"{spec.name}.yaml"
        self.export_input(spec, yaml_file)

        argv = [
            exe_path,
            "predict",
            str(yaml_file),
            "--out_dir",
            str(work_dir),
            "--diffusion_samples",
            str(spec.num_samples),
            *[str(a) for a in extra_args],
        ]
        proc = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Boltz-2 CLI exited with code {proc.returncode}: {proc.stderr.strip()[-500:]}"
            )

        return self.parse_output(
            work_dir,
            name=spec.name,
            ligand_resname=ligand_resname,
            reference_smiles=spec.primary_ligand_smiles,
            pocket_cutoff=pocket_cutoff,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            runtime_sec=time.perf_counter() - t0,
        )


class ProtenixBackend(BaseFoldingBackend):
    """Apache-2.0 ByteDance Protenix JSON exporter, optional CLI runner, and output parser."""

    name = "protenix"

    def export_input(
        self, job: FoldingJobSpec, out_path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        return export_protenix_json(job, out_path)

    def parse_output(
        self,
        output_dir: str | Path,
        *,
        name: str = "protenix_complex",
        ligand_resname: str | None = None,
        reference_smiles: str | None = None,
        pocket_cutoff: float = 10.0,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        runtime_sec: float = 0.0,
    ) -> FoldedComplexResult:
        return _parse_external_structure_output(
            backend=self.name,
            name=name,
            output_dir=output_dir,
            structure_patterns=("*_sample_0.cif", "*.cif", "*.mmcif", "*.pdb"),
            summary_json_patterns=(
                "summary_confidence_sample_0.json",
                "summary_confidence*.json",
                "*summary_confidences*.json",
            ),
            confidences_json_patterns=("full_data_sample_0.json", "full_data*.json"),
            ligand_resname=ligand_resname,
            reference_smiles=reference_smiles,
            pocket_cutoff=pocket_cutoff,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            runtime_sec=runtime_sec,
        )

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        mode: Literal["run", "export_only", "parse_output"] = "run",
        executable: str = "protenix",
        ligand_resname: str | None = None,
        pocket_cutoff: float = 10.0,
        timeout_sec: float = 1800.0,
        extra_args: Sequence[str] = (),
        **kwargs: Any,
    ) -> FoldedComplexResult:
        del kwargs
        t0 = time.perf_counter()
        spec = job.validated()

        if mode == "parse_output":
            if output_dir is None:
                raise ValueError("ProtenixBackend(mode='parse_output') requires output_dir.")
            return self.parse_output(
                output_dir,
                name=spec.name,
                ligand_resname=ligand_resname,
                reference_smiles=spec.primary_ligand_smiles,
                pocket_cutoff=pocket_cutoff,
                run_audit=run_audit,
                property_profile=property_profile,
                canonical_anchors=canonical_anchors,
                runtime_sec=time.perf_counter() - t0,
            )

        if mode == "export_only":
            json_path = (
                _sanitize_path(output_dir) / f"{spec.name}_protenix.json"
                if output_dir is not None
                else None
            )
            payload = self.export_input(spec, json_path)
            payload_text = json.dumps(payload, indent=2)
            meta = {
                "mode": "export_only",
                "input_json": payload,
                "input_path": str(json_path) if json_path else None,
            }
            return FoldedComplexResult(
                backend=self.name,
                name=spec.name,
                ligand_smiles=spec.primary_ligand_smiles,
                runtime_sec=time.perf_counter() - t0,
                provenance_sha256=_compute_provenance_sha256(
                    backend=self.name, name=spec.name, structure_text=payload_text, metadata=meta
                ),
                metadata=meta,
            )

        import shutil
        import subprocess
        import tempfile

        exe_path = shutil.which(executable)
        if exe_path is None:
            raise MissingBackendError(
                f"Protenix CLI executable {executable!r} is not installed or not found on PATH.",
                hint="Install Protenix with `pip install protenix` (Apache-2.0 license), or use "
                "mode='export_only' / mode='parse_output'.",
            )

        work_dir = (
            _sanitize_path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix=f"geneuclidean_protenix_{spec.name}_"))
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        json_file = work_dir / f"{spec.name}_protenix.json"
        self.export_input(spec, json_file)

        argv = [
            exe_path,
            "predict",
            "--input",
            str(json_file),
            "--out_dir",
            str(work_dir),
            *[str(a) for a in extra_args],
        ]
        proc = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Protenix CLI exited with code {proc.returncode}: {proc.stderr.strip()[-500:]}"
            )

        return self.parse_output(
            work_dir,
            name=spec.name,
            ligand_resname=ligand_resname,
            reference_smiles=spec.primary_ligand_smiles,
            pocket_cutoff=pocket_cutoff,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            runtime_sec=time.perf_counter() - t0,
        )


class AlphaFold3Backend(BaseFoldingBackend):
    """Official AlphaFold 3 JSON dialect (v2/v3/v4) exporter & mmCIF/confidence parser."""

    name = "alphafold3"

    def export_input(
        self,
        job: FoldingJobSpec,
        out_path: str | Path | None = None,
        *,
        dialect_version: int = 2,
    ) -> dict[str, Any]:
        return export_alphafold3_json(job, out_path, dialect_version=dialect_version)

    def parse_output(
        self,
        output_dir: str | Path,
        *,
        name: str = "alphafold3_complex",
        ligand_resname: str | None = None,
        reference_smiles: str | None = None,
        pocket_cutoff: float = 10.0,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        runtime_sec: float = 0.0,
    ) -> FoldedComplexResult:
        return _parse_external_structure_output(
            backend=self.name,
            name=name,
            output_dir=output_dir,
            structure_patterns=("*_model.cif", "*.cif", "*.mmcif", "*.pdb"),
            summary_json_patterns=("*summary_confidences*.json", "summary_confidences.json"),
            confidences_json_patterns=("*_confidences.json", "confidences.json"),
            ligand_resname=ligand_resname,
            reference_smiles=reference_smiles,
            pocket_cutoff=pocket_cutoff,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            runtime_sec=runtime_sec,
        )

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        mode: Literal["run", "export_only", "parse_output"] = "run",
        dialect_version: int = 2,
        executable: str = "run_alphafold.py",
        ligand_resname: str | None = None,
        pocket_cutoff: float = 10.0,
        timeout_sec: float = 3600.0,
        extra_args: Sequence[str] = (),
        **kwargs: Any,
    ) -> FoldedComplexResult:
        del kwargs
        t0 = time.perf_counter()
        spec = job.validated()

        if mode == "parse_output":
            if output_dir is None:
                raise ValueError("AlphaFold3Backend(mode='parse_output') requires output_dir.")
            return self.parse_output(
                output_dir,
                name=spec.name,
                ligand_resname=ligand_resname,
                reference_smiles=spec.primary_ligand_smiles,
                pocket_cutoff=pocket_cutoff,
                run_audit=run_audit,
                property_profile=property_profile,
                canonical_anchors=canonical_anchors,
                runtime_sec=time.perf_counter() - t0,
            )

        if mode == "export_only":
            json_path = (
                _sanitize_path(output_dir) / f"{spec.name}_af3.json"
                if output_dir is not None
                else None
            )
            payload = self.export_input(spec, json_path, dialect_version=dialect_version)
            payload_text = json.dumps(payload, indent=2)
            meta = {
                "mode": "export_only",
                "dialect_version": int(dialect_version),
                "input_json": payload,
                "input_path": str(json_path) if json_path else None,
            }
            return FoldedComplexResult(
                backend=self.name,
                name=spec.name,
                ligand_smiles=spec.primary_ligand_smiles,
                runtime_sec=time.perf_counter() - t0,
                provenance_sha256=_compute_provenance_sha256(
                    backend=self.name, name=spec.name, structure_text=payload_text, metadata=meta
                ),
                metadata=meta,
            )

        import shutil
        import subprocess
        import tempfile

        exe_path = shutil.which(executable)
        if exe_path is None:
            raise MissingBackendError(
                f"AlphaFold 3 runner {executable!r} is not installed or not found on PATH.",
                hint="Use mode='export_only' to emit an official AlphaFold 3 JSON job bundle "
                "(dialect v2/v3/v4) for AlphaFold Server or a local AF3 install, then parse "
                "outputs with mode='parse_output'.",
            )

        work_dir = (
            _sanitize_path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix=f"geneuclidean_af3_{spec.name}_"))
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        json_file = work_dir / f"{spec.name}_af3.json"
        self.export_input(spec, json_file, dialect_version=dialect_version)

        argv = [
            exe_path,
            f"--json_path={json_file}",
            f"--output_dir={work_dir}",
            *[str(a) for a in extra_args],
        ]
        proc = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"AlphaFold 3 exited with code {proc.returncode}: {proc.stderr.strip()[-500:]}"
            )

        return self.parse_output(
            work_dir,
            name=spec.name,
            ligand_resname=ligand_resname,
            reference_smiles=spec.primary_ligand_smiles,
            pocket_cutoff=pocket_cutoff,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            runtime_sec=time.perf_counter() - t0,
        )


class ESMFoldBackend(BaseFoldingBackend):
    """Single-sequence zero-MSA ESMFold runner (local ``transformers`` or opt-in ESM Atlas API).

    Offline-safe by default: ``allow_network=False`` strictly blocks remote HTTP requests
    and prevents Hugging Face network downloads during offline CI runs.
    """

    name = "esmfold_api"

    def run(
        self,
        job: FoldingJobSpec,
        *,
        output_dir: str | Path | None = None,
        run_audit: bool = False,
        property_profile: str = "oral_ro5",
        canonical_anchors: Sequence[str] = (),
        mode: Literal["api", "local"] = "api",
        allow_network: bool = False,
        model_id: str = "facebook/esmfold_v1",
        timeout_sec: float = 30.0,
        **kwargs: Any,
    ) -> FoldedComplexResult:
        del run_audit, property_profile, canonical_anchors, kwargs
        t0 = time.perf_counter()
        spec = job.validated(allow_ambiguous_aa=False)

        if spec.ligand_smiles or spec.ligand_ccds or spec.dna_chains or spec.rna_chains:
            raise ValueError(
                "ESMFold is a single-sequence protein-only folder and cannot co-fold ligands "
                "or nucleic acids. Use 'nanofold3', 'boltz2', 'protenix', or 'alphafold3'."
            )
        if len(spec.protein_chains) != 1:
            raise ValueError(
                f"ESMFold requires exactly one protein chain (got {len(spec.protein_chains)})."
            )
        sequence = spec.protein_chains[0][1]

        if mode == "api":
            if not allow_network:
                raise NetworkDisabledError(
                    "ESMFold REST API request blocked because allow_network=False.",
                    hint="Pass allow_network=True to explicitly opt in to the Meta ESM Atlas "
                    "HTTPS API (<= 400 aa), or use mode='local' with local transformers weights.",
                )
            if len(sequence) > _ESMFOLD_MAX_API_AA:
                raise ValueError(
                    f"Meta ESM Atlas public API supports sequences <= {_ESMFOLD_MAX_API_AA} aa "
                    f"(got {len(sequence)} aa). Use mode='local' or another backend."
                )
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                _ESMFOLD_API_URL,
                data=sequence.encode("ascii"),
                headers={"Content-Type": "text/plain"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:  # noqa: S310
                    pdb_text = resp.read().decode("utf-8")
            except urllib.error.URLError as exc:
                raise RuntimeError(f"ESMFold API request failed: {exc}") from exc
            source_meta: dict[str, Any] = {
                "mode": "api",
                "endpoint": _ESMFOLD_API_URL,
                "sequence_length": len(sequence),
            }
        elif mode == "local":
            try:
                import torch
                from transformers import AutoTokenizer, EsmForProteinFolding
            except ImportError as exc:
                raise MissingBackendError(
                    "Local ESMFold requires 'torch' and 'transformers'.",
                    hint="Install with `pip install 'geneuclidean[llm]'`, or pass "
                    "mode='api', allow_network=True for sequences <= 400 aa.",
                ) from exc

            try:
                tokenizer: Any = AutoTokenizer.from_pretrained(
                    model_id, local_files_only=not allow_network
                )
                esm_model: Any = EsmForProteinFolding.from_pretrained(
                    model_id, local_files_only=not allow_network
                )
            except Exception as exc:
                if not allow_network:
                    raise NetworkDisabledError(
                        f"Local ESMFold weights for {model_id!r} are not cached and "
                        "allow_network=False prevented downloading.",
                        hint="Pre-cache weights locally or pass allow_network=True.",
                    ) from exc
                raise

            esm_model.eval()
            with torch.inference_mode():
                tokenized = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
                outputs = esm_model(**tokenized)
                pdb_blocks = esm_model.output_to_pdb(outputs)
                pdb_text = str(pdb_blocks[0])
            source_meta = {
                "mode": "local",
                "model_id": model_id,
                "sequence_length": len(sequence),
            }
        else:
            raise ValueError(f"Unsupported ESMFold mode {mode!r}; expected 'api' or 'local'.")

        from geneuclidean.interop.boltz_adapter import extract_bfactor_plddt_from_text

        plddt_map = extract_bfactor_plddt_from_text(pdb_text, fmt="pdb")
        all_plddt = plddt_map["all"]
        plddt_tuple = tuple(float(x) for x in all_plddt.tolist())
        plddt_mean = float(all_plddt.mean()) if all_plddt.size > 0 else float("nan")

        struct_path: Path | None = None
        if output_dir is not None:
            dest_dir = _sanitize_path(output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            struct_path = dest_dir / f"{spec.name}_esmfold.pdb"
            struct_path.write_text(pdb_text, encoding="utf-8")

        return FoldedComplexResult(
            backend=self.name,
            name=spec.name,
            pdb_text=pdb_text,
            structure_path=struct_path,
            plddt_mean=plddt_mean,
            plddt_per_atom=plddt_tuple,
            runtime_sec=time.perf_counter() - t0,
            provenance_sha256=_compute_provenance_sha256(
                backend=self.name, name=spec.name, structure_text=pdb_text, metadata=source_meta
            ),
            metadata=source_meta,
        )


# ---------------------------------------------------------------------------
# Thread-Safe Registry & Built-In Registrations
# ---------------------------------------------------------------------------

FOLDING_BACKEND_REGISTRY: Registry[Any] = Registry("folding_backend")

FOLDING_BACKEND_REGISTRY.register(
    "nanofold3",
    NanoFold3Backend(),
    aliases=("nanofold", "mini_af3"),
    metadata={
        "license": "MIT",
        "supports_ligand": True,
        "supports_pocket_cofold": True,
        "supports_single_sequence": True,
        "requires_torch": True,
        "requires_cli": False,
        "offline_safe": True,
    },
)
FOLDING_BACKEND_REGISTRY.register(
    "boltz2",
    Boltz2Backend(),
    aliases=("boltz", "boltz-2"),
    metadata={
        "license": "MIT",
        "supports_ligand": True,
        "supports_affinity": True,
        "requires_torch": False,
        "requires_cli": True,
        "input_format": "yaml_v1",
        "offline_safe": True,
    },
)
FOLDING_BACKEND_REGISTRY.register(
    "protenix",
    ProtenixBackend(),
    aliases=("protenix_v1", "protenix_v2"),
    metadata={
        "license": "Apache-2.0",
        "supports_ligand": True,
        "requires_torch": False,
        "requires_cli": True,
        "input_format": "protenix_json",
        "offline_safe": True,
    },
)
FOLDING_BACKEND_REGISTRY.register(
    "alphafold3",
    AlphaFold3Backend(),
    aliases=("af3", "af3_json"),
    metadata={
        "license": "Apache-2.0 (code) / Non-Commercial (weights)",
        "supports_ligand": True,
        "requires_torch": False,
        "requires_cli": True,
        "input_format": "alphafold3_json_v2_v3_v4",
        "offline_safe": True,
    },
)
FOLDING_BACKEND_REGISTRY.register(
    "esmfold_api",
    ESMFoldBackend(),
    aliases=("esmfold", "esmfold_local"),
    metadata={
        "license": "MIT",
        "supports_ligand": False,
        "supports_single_sequence": True,
        "max_api_length_aa": _ESMFOLD_MAX_API_AA,
        "requires_torch": False,
        "requires_cli": False,
        "offline_safe": True,
    },
)


def register_folding_backend(
    name: str,
    backend: Any | None = None,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Register a custom folding/co-folding backend instance or callable."""
    return FOLDING_BACKEND_REGISTRY.register(
        name,
        backend,
        aliases=aliases,
        override=override,
        metadata=metadata,
    )


def list_folding_backends() -> list[str]:
    """Return sorted canonical names of all registered folding/co-folding backends."""
    return FOLDING_BACKEND_REGISTRY.names()


def folding_backend_catalog() -> dict[str, dict[str, Any]]:
    """Return JSON-serializable metadata summary of all registered folding backends."""
    return FOLDING_BACKEND_REGISTRY.summary()


def run_folding_backend(
    backend: str,
    job: FoldingJobSpec,
    *,
    output_dir: str | Path | None = None,
    run_audit: bool = False,
    property_profile: str = "oral_ro5",
    canonical_anchors: Sequence[str] = (),
    **kwargs: Any,
) -> FoldedComplexResult:
    """Dispatch a :class:`FoldingJobSpec` to any registered folding backend."""
    canonical = FOLDING_BACKEND_REGISTRY.resolve_key(backend)
    handler = FOLDING_BACKEND_REGISTRY.get(canonical)

    # Alias convenience: "esmfold_local" defaults ESMFoldBackend to mode="local"
    if backend.strip().lower() == "esmfold_local" and "mode" not in kwargs:
        kwargs["mode"] = "local"

    if hasattr(handler, "run") and callable(handler.run):
        result: FoldedComplexResult = handler.run(
            job,
            output_dir=output_dir,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            **kwargs,
        )
    elif callable(handler):
        fn: Callable[..., FoldedComplexResult] = handler
        result = fn(
            job,
            output_dir=output_dir,
            run_audit=run_audit,
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
            **kwargs,
        )
    else:
        raise TypeError(
            f"Registered folding backend {canonical!r} must implement .run(...) or be callable."
        )

    if not isinstance(result, FoldedComplexResult):
        raise TypeError(
            f"Folding backend {canonical!r} returned {type(result).__name__}, "
            "expected FoldedComplexResult."
        )
    return result


def fold_complex(
    *,
    protein_sequence: str | None = None,
    ligand_smiles: str | None = None,
    ligand_ccd: str | None = None,
    pocket_pdb: str | Path | None = None,
    backend: str = "nanofold3",
    name: str = "complex",
    seeds: Sequence[int] = (42,),
    num_samples: int = 1,
    predict_affinity: bool = False,
    run_audit: bool = True,
    property_profile: str = "oral_ro5",
    canonical_anchors: Sequence[str] = (),
    output_dir: str | Path | None = None,
    allow_network: bool = False,
    **backend_kwargs: Any,
) -> FoldedComplexResult:
    """Unified high-level entrypoint to fold/co-fold a biomolecular complex across any backend.

    Args:
        protein_sequence: Optional amino-acid sequence (single chain ``"A"``).
        ligand_smiles: Optional small-molecule ligand SMILES string.
        ligand_ccd: Optional PDB CCD code (e.g. ``"ATP"``).
        pocket_pdb: Optional 3D receptor pocket PDB path or text (used by ``nanofold3``).
        backend: Registered backend name or alias (default ``"nanofold3"``).
        name: Safe job identifier (alphanumeric / ``_.-``).
        seeds: Random seed(s) for deterministic folding.
        num_samples: Number of diffusion/sampling rollouts.
        predict_affinity: Request binding affinity prediction when supported (e.g. ``boltz2``).
        run_audit: Automatically run 7-axis :class:`PoseRealityAuditor` when a 3D ligand pose
            is present (default ``True``).
        property_profile: Therapeutic property profile for Axis 7 auditing.
        canonical_anchors: Optional required pharmacophore anchor residue tokens.
        output_dir: Optional directory for job bundles, predicted structures, or parsing.
        allow_network: Strictly opt-in network flag (default ``False`` for offline CI safety).
        **backend_kwargs: Extra backend-specific parameters (e.g. ``mode="export_only"``,
            ``preset="nano"``, ``dialect_version=2``).

    Returns:
        Unified :class:`FoldedComplexResult`.
    """
    job = _coerce_job_spec(
        None,
        name=name,
        protein_sequence=protein_sequence,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
        pocket_pdb=pocket_pdb,
        seeds=seeds,
        num_samples=num_samples,
        predict_affinity=predict_affinity,
    )

    canonical = FOLDING_BACKEND_REGISTRY.resolve_key(backend)
    call_kwargs: dict[str, Any] = dict(backend_kwargs)
    if canonical == "esmfold_api":
        call_kwargs.setdefault("allow_network", allow_network)

    result = run_folding_backend(
        backend,
        job,
        output_dir=output_dir,
        run_audit=run_audit,
        property_profile=property_profile,
        canonical_anchors=canonical_anchors,
        **call_kwargs,
    )

    # Ensure custom backends also benefit from automatic run_audit=True when returning a 3D ligand
    if (
        run_audit
        and result.pose_audit is None
        and result.metadata.get("mode") != "export_only"
        and (
            result.ligand_mol_3d is not None
            or (
                result.ligand_smiles is not None
                and (result.structure_path or result.pdb_text or result.cif_text)
            )
        )
    ):
        audit_rep = result.run_audit(
            property_profile=property_profile,
            canonical_anchors=canonical_anchors,
        )
        result = dataclasses.replace(result, pose_audit=audit_rep)

    return result


__all__ = [
    "FOLDING_BACKEND_REGISTRY",
    "AlphaFold3Backend",
    "BaseFoldingBackend",
    "Boltz2Backend",
    "ESMFoldBackend",
    "FoldedComplexResult",
    "FoldingBackend",
    "FoldingJobSpec",
    "MissingBackendError",
    "NanoFold3Backend",
    "NetworkDisabledError",
    "ProtenixBackend",
    "export_alphafold3_json",
    "export_boltz2_yaml",
    "export_protenix_json",
    "fold_complex",
    "folding_backend_catalog",
    "list_folding_backends",
    "register_folding_backend",
    "run_folding_backend",
]
