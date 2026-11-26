"""Boltz-2 / AlphaFold 3 / Protenix / Chai-1 co-folded complex loader & 7-axis PoseAuditor bridge.

Ingests standard ``.cif`` (mmCIF) or ``.pdb`` biomolecular co-folding predictions, cleanly
extracts the small-molecule ligand and surrounding protein binding pocket (preserving
per-atom B-factor ``pLDDT`` and optional structural cofactors), restores exact covalent
bond orders from the input SMILES template via ``AllChem.AssignBondOrdersFromTemplate``,
parses official AlphaFold 3 / Boltz-2 / Protenix confidence & affinity sidecar JSONs into
:class:`ConfidenceSummary`, and feeds complexes directly into
:class:`~geneuclidean.audit.PoseRealityAuditor`.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from geneuclidean.audit.pose_auditor import PoseAuditReport, PoseRealityAuditor

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

_IGNORE_HETATMS: frozenset[str] = frozenset(
    {"HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "SO4", "PO4", "GOL", "EDO"}
)


@dataclass(frozen=True)
class ConfidenceSummary:
    """Structured confidence & affinity metadata from AF3, Boltz-2, or Protenix sidecar JSONs."""

    source_format: str
    ptm: float | None = None
    iptm: float | None = None
    ranking_score: float | None = None
    plddt_mean: float | None = None
    ligand_plddt: float | None = None
    ligand_iptm: float | None = None
    protein_iptm: float | None = None
    complex_pde: float | None = None
    complex_ipde: float | None = None
    fraction_disordered: float | None = None
    has_clash: bool | None = None
    chain_ptm: tuple[float, ...] = ()
    chain_iptm: tuple[float, ...] = ()
    chain_pair_iptm: tuple[tuple[float | None, ...], ...] = ()
    atom_plddts: tuple[float, ...] = ()
    pae_matrix: np.ndarray | None = None
    affinity_pred_value: float | None = None
    affinity_probability_binary: float | None = None
    predicted_pkd: float | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to a JSON-friendly dictionary (omitting large PAE matrices)."""
        pae_shape = list(self.pae_matrix.shape) if self.pae_matrix is not None else None
        return {
            "source_format": self.source_format,
            "ptm": self.ptm,
            "iptm": self.iptm,
            "ranking_score": self.ranking_score,
            "plddt_mean": self.plddt_mean,
            "ligand_plddt": self.ligand_plddt,
            "ligand_iptm": self.ligand_iptm,
            "protein_iptm": self.protein_iptm,
            "complex_pde": self.complex_pde,
            "complex_ipde": self.complex_ipde,
            "fraction_disordered": self.fraction_disordered,
            "has_clash": self.has_clash,
            "chain_ptm": list(self.chain_ptm),
            "chain_iptm": list(self.chain_iptm),
            "chain_pair_iptm": [list(row) for row in self.chain_pair_iptm],
            "num_atom_plddts": len(self.atom_plddts),
            "pae_shape": pae_shape,
            "affinity_pred_value": self.affinity_pred_value,
            "affinity_probability_binary": self.affinity_probability_binary,
            "predicted_pkd": self.predicted_pkd,
            "raw_metadata": dict(self.raw_metadata),
        }


def normalize_plddt_values(
    values: Sequence[float] | np.ndarray,
    *,
    scale: Literal["auto", "unit", "percent"] = "auto",
) -> np.ndarray:
    """Normalize per-atom or per-residue ``pLDDT`` values cleanly onto ``[0.0, 100.0]``.

    Supports both ``[0, 1]`` (ESMFold / Boltz-2 scalar conventions) and ``[0, 100]``
    (AlphaFold 2/3 / PDB B-factor conventions) scales with automatic detection.
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    if scale == "unit":
        out = arr * np.float32(100.0)
    elif scale == "percent":
        out = arr.copy()
    elif scale == "auto":
        finite = arr[np.isfinite(arr)]
        if (
            finite.size > 0
            and float(np.max(finite)) <= 1.0 + 1e-6
            and float(np.min(finite)) >= -1e-6
        ):
            out = arr * np.float32(100.0)
        else:
            out = arr.copy()
    else:
        raise ValueError(
            f"Unsupported pLDDT scale {scale!r}; expected 'auto', 'unit', or 'percent'."
        )
    clipped: np.ndarray[Any, np.dtype[np.float32]] = np.asarray(
        np.clip(out, np.float32(0.0), np.float32(100.0)), dtype=np.float32
    )
    return clipped


def _normalize_scalar_plddt(val: float | None) -> float | None:
    if val is None or not math.isfinite(val):
        return None
    scaled = val * 100.0 if -1e-6 <= val <= 1.0 + 1e-6 else val
    return float(max(0.0, min(100.0, scaled)))


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(float(value) >= 0.5)
    return None


def _select_ligand_residue(model: Any, ligand_resname: str | None) -> Any | None:
    """Identify the primary small-molecule HETATM ligand residue in a Bio.PDB model."""
    cand_ligs: list[tuple[int, Any]] = []
    target_name = ligand_resname.strip().upper() if ligand_resname is not None else None
    for chain in model:
        for res in chain:
            hetflag = res.id[0]
            rname = res.get_resname().strip().upper()
            if hetflag == " " or rname in _IGNORE_HETATMS:
                continue
            if target_name is not None and rname != target_name:
                continue
            n_heavy = sum(1 for a in res.get_atoms() if (a.element or "").upper() != "H")
            if n_heavy >= 4:
                cand_ligs.append((n_heavy, res))
    if not cand_ligs:
        return None
    cand_ligs.sort(key=lambda x: x[0], reverse=True)
    return cand_ligs[0][1]


def _extract_bfactor_plddt_from_model(
    model: Any,
    *,
    ligand_resname: str | None = None,
    scale: Literal["auto", "unit", "percent"] = "auto",
) -> dict[str, np.ndarray]:
    """Extract and consistently normalize per-atom B-factor pLDDT arrays from a Bio.PDB model."""
    lig_res = _select_ligand_residue(model, ligand_resname)
    all_raw: list[float] = []
    protein_raw: list[float] = []
    ligand_raw: list[float] = []
    by_chain_raw: dict[str, list[float]] = {}

    for chain in model:
        cid = (chain.id or "A").strip() or "A"
        chain_vals = by_chain_raw.setdefault(cid, [])
        for res in chain:
            is_polymer = res.id[0] == " "
            is_lig = lig_res is not None and res is lig_res
            for atom in res.get_atoms():
                bfac = float(atom.get_bfactor())
                all_raw.append(bfac)
                chain_vals.append(bfac)
                if is_polymer:
                    protein_raw.append(bfac)
                elif is_lig:
                    ligand_raw.append(bfac)

    # Detect unit [0, 1] vs percent [0, 100] scale globally once for consistent subset scaling.
    resolved_scale: Literal["unit", "percent"]
    if scale == "unit":
        resolved_scale = "unit"
    elif scale == "percent":
        resolved_scale = "percent"
    elif scale == "auto":
        all_arr = np.asarray(all_raw, dtype=np.float32)
        finite = all_arr[np.isfinite(all_arr)]
        if (
            finite.size > 0
            and float(np.max(finite)) <= 1.0 + 1e-6
            and float(np.min(finite)) >= -1e-6
        ):
            resolved_scale = "unit"
        else:
            resolved_scale = "percent"
    else:
        raise ValueError(
            f"Unsupported pLDDT scale {scale!r}; expected 'auto', 'unit', or 'percent'."
        )

    result: dict[str, np.ndarray] = {
        "all": normalize_plddt_values(all_raw, scale=resolved_scale),
        "protein": normalize_plddt_values(protein_raw, scale=resolved_scale),
        "ligand": normalize_plddt_values(ligand_raw, scale=resolved_scale),
    }
    for cid, vals in by_chain_raw.items():
        result[f"chain:{cid}"] = normalize_plddt_values(vals, scale=resolved_scale)
    return result


def _parse_biopdb_first_model(source: Any, *, is_cif: bool) -> Any:
    parser: Any = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    structure: Any = parser.get_structure("cofold", source)
    return next(structure.get_models())


def extract_bfactor_plddt(
    structure_path: str | Path,
    *,
    ligand_resname: str | None = None,
    scale: Literal["auto", "unit", "percent"] = "auto",
) -> dict[str, np.ndarray]:
    """Extract per-atom ``pLDDT`` scores (normalized to ``[0, 100]``) from a PDB or mmCIF file.

    Returns a dictionary containing ``"all"``, ``"protein"``, ``"ligand"``, and per-chain
    ``"chain:<ID>"`` ``float32`` NumPy arrays.
    """
    src = Path(structure_path)
    if not src.is_file():
        raise FileNotFoundError(f"Co-folded complex file not found: {src}")
    model = _parse_biopdb_first_model(str(src), is_cif=src.suffix.lower() in (".cif", ".mmcif"))
    return _extract_bfactor_plddt_from_model(model, ligand_resname=ligand_resname, scale=scale)


def extract_bfactor_plddt_from_text(
    structure_text: str,
    *,
    fmt: Literal["pdb", "cif"] = "pdb",
    ligand_resname: str | None = None,
    scale: Literal["auto", "unit", "percent"] = "auto",
) -> dict[str, np.ndarray]:
    """Extract normalized per-atom B-factor ``pLDDT`` arrays from in-memory PDB or mmCIF text."""
    if not structure_text.strip():
        empty = np.zeros((0,), dtype=np.float32)
        return {"all": empty, "protein": empty.copy(), "ligand": empty.copy()}
    model = _parse_biopdb_first_model(io.StringIO(structure_text), is_cif=(fmt == "cif"))
    return _extract_bfactor_plddt_from_model(model, ligand_resname=ligand_resname, scale=scale)


def _extract_float_sequence(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, Mapping):
        items: list[tuple[str, float]] = []
        for k, v in raw.items():
            fv = _opt_float(v)
            if fv is not None:
                items.append((str(k), fv))
        items.sort(key=lambda kv: (int(kv[0]) if kv[0].isdigit() else 10**9, kv[0]))
        return tuple(v for _, v in items)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: list[float] = []
        for item in raw:
            fv = _opt_float(item)
            if fv is not None:
                out.append(fv)
        return tuple(out)
    return ()


def _extract_pair_matrix(raw: Any) -> tuple[tuple[float | None, ...], ...]:
    if isinstance(raw, Mapping):
        keys = sorted(
            (str(k) for k in raw),
            key=lambda s: (int(s) if s.isdigit() else 10**9, s),
        )
        rows: list[tuple[float | None, ...]] = []
        for rkey in keys:
            sub = raw.get(rkey)
            if isinstance(sub, Mapping):
                rows.append(tuple(_opt_float(sub.get(ckey)) for ckey in keys))
        return tuple(rows)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        mat_rows: list[tuple[float | None, ...]] = []
        for row in raw:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                mat_rows.append(tuple(_opt_float(x) for x in row))
        return tuple(mat_rows)
    return ()


def parse_cofold_confidence_dict(
    summary_data: Mapping[str, Any],
    *,
    confidences_data: Mapping[str, Any] | None = None,
    affinity_data: Mapping[str, Any] | None = None,
) -> ConfidenceSummary:
    """Parse AF3, Boltz-2, or Protenix confidence/affinity JSON dicts into ``ConfidenceSummary``."""
    summary = dict(summary_data)
    conf = dict(confidences_data) if confidences_data is not None else {}
    aff = dict(affinity_data) if affinity_data is not None else {}

    # Detect schema dialect cleanly
    if (
        aff
        or "complex_plddt" in summary
        or "confidence_score" in summary
        or "chains_ptm" in summary
        or "pair_chains_iptm" in summary
        or "affinity_pred_value" in summary
    ):
        source_format = "boltz2"
    elif "gpde" in summary or "chain_plddt" in summary or "disorder" in summary:
        source_format = "protenix"
    elif (
        any(
            k in summary
            for k in (
                "fraction_disordered",
                "chain_pair_iptm",
                "chain_pair_pae_min",
                "ranking_score",
            )
        )
        or "atom_plddts" in conf
        or "atom_plddts" in summary
    ):
        source_format = "alphafold3"
    else:
        source_format = "unknown"

    ptm = _opt_float(summary.get("ptm"))
    iptm = _opt_float(summary.get("iptm"))
    ligand_iptm = _opt_float(summary.get("ligand_iptm"))
    protein_iptm = _opt_float(summary.get("protein_iptm"))
    complex_pde = _opt_float(summary.get("complex_pde", summary.get("gpde")))
    complex_ipde = _opt_float(summary.get("complex_ipde"))
    fraction_disordered = _opt_float(summary.get("fraction_disordered", summary.get("disorder")))
    has_clash = _opt_bool(summary.get("has_clash"))

    chain_ptm = _extract_float_sequence(summary.get("chain_ptm", summary.get("chains_ptm")))
    chain_iptm = _extract_float_sequence(summary.get("chain_iptm"))
    chain_pair_iptm = _extract_pair_matrix(
        summary.get("chain_pair_iptm", summary.get("pair_chains_iptm"))
    )

    # Atom-level pLDDT array (AF3 confidences.json or custom sidecars)
    raw_atom_plddts = conf.get("atom_plddts", summary.get("atom_plddts"))
    atom_plddts_tuple: tuple[float, ...] = ()
    if isinstance(raw_atom_plddts, Sequence) and not isinstance(raw_atom_plddts, (str, bytes)):
        norm_atoms = normalize_plddt_values(
            [float(x) for x in raw_atom_plddts if _opt_float(x) is not None]
        )
        atom_plddts_tuple = tuple(float(x) for x in norm_atoms.tolist())

    # Scalar mean pLDDT (auto-normalizing Boltz-2 [0, 1] vs AF3/Protenix [0, 100])
    plddt_mean_raw = _opt_float(
        summary.get(
            "complex_plddt",
            summary.get("plddt_mean", summary.get("plddt")),
        )
    )
    plddt_mean = _normalize_scalar_plddt(plddt_mean_raw)
    if plddt_mean is None and atom_plddts_tuple:
        plddt_mean = float(sum(atom_plddts_tuple) / len(atom_plddts_tuple))

    ligand_plddt = _normalize_scalar_plddt(
        _opt_float(summary.get("ligand_plddt", summary.get("complex_iplddt")))
    )

    # Optional PAE matrix [L, L]
    pae_raw = conf.get(
        "pae",
        summary.get("pae", conf.get("token_pair_pae", summary.get("token_pair_pae"))),
    )
    pae_matrix: np.ndarray | None = None
    if isinstance(pae_raw, Sequence) and not isinstance(pae_raw, (str, bytes)) and len(pae_raw) > 0:
        with contextlib.suppress(TypeError, ValueError):
            arr = np.asarray(pae_raw, dtype=np.float32)
            if arr.ndim == 2:
                pae_matrix = arr

    # Ranking score (AF3 ranking_score, Boltz-2 confidence_score, or 0.8*ipTM + 0.2*pTM fallback)
    ranking_score = _opt_float(summary.get("ranking_score", summary.get("confidence_score")))
    if ranking_score is None:
        if iptm is not None and ptm is not None:
            ranking_score = 0.8 * iptm + 0.2 * ptm
        elif ptm is not None:
            ranking_score = ptm
        elif iptm is not None:
            ranking_score = iptm

    # Boltz-2 binding affinity sidecar (log10(IC50 in uM) + P(binder))
    merged_aff = {**summary, **aff}
    affinity_pred_value = _opt_float(merged_aff.get("affinity_pred_value"))
    affinity_prob = _opt_float(merged_aff.get("affinity_probability_binary"))
    explicit_pkd = _opt_float(merged_aff.get("predicted_pkd"))
    if explicit_pkd is not None:
        predicted_pkd: float | None = explicit_pkd
    elif affinity_pred_value is not None:
        # Official Boltz-2 convention (Passaro et al., 2025):
        # affinity_pred_value = log10(IC50 in uM) => pIC50 / pKd (M) = 6 - affinity_pred_value
        predicted_pkd = 6.0 - affinity_pred_value
    else:
        predicted_pkd = None

    raw_meta: dict[str, Any] = {}
    for key, val in summary.items():
        if key not in ("pae", "token_pair_pae", "atom_plddts", "contact_probs"):
            raw_meta[key] = val
    for key, val in aff.items():
        raw_meta[key] = val
    if affinity_pred_value is not None and explicit_pkd is None:
        raw_meta["affinity_pkd_conversion"] = "6.0 - log10(IC50_uM)"

    return ConfidenceSummary(
        source_format=source_format,
        ptm=ptm,
        iptm=iptm,
        ranking_score=ranking_score,
        plddt_mean=plddt_mean,
        ligand_plddt=ligand_plddt,
        ligand_iptm=ligand_iptm,
        protein_iptm=protein_iptm,
        complex_pde=complex_pde,
        complex_ipde=complex_ipde,
        fraction_disordered=fraction_disordered,
        has_clash=has_clash,
        chain_ptm=chain_ptm,
        chain_iptm=chain_iptm,
        chain_pair_iptm=chain_pair_iptm,
        atom_plddts=atom_plddts_tuple,
        pae_matrix=pae_matrix,
        affinity_pred_value=affinity_pred_value,
        affinity_probability_binary=affinity_prob,
        predicted_pkd=predicted_pkd,
        raw_metadata=raw_meta,
    )


def _read_json_mapping(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Confidence JSON file not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {src}, got {type(data).__name__}.")
    return data


def parse_cofold_confidence(
    summary_path: str | Path,
    *,
    confidences_path: str | Path | None = None,
    affinity_path: str | Path | None = None,
) -> ConfidenceSummary:
    """Load AlphaFold 3, Boltz-2, or Protenix confidence/affinity JSON sidecar files."""
    summary_data = _read_json_mapping(summary_path)
    conf_data = _read_json_mapping(confidences_path) if confidences_path is not None else None
    aff_data = _read_json_mapping(affinity_path) if affinity_path is not None else None
    return parse_cofold_confidence_dict(
        summary_data,
        confidences_data=conf_data,
        affinity_data=aff_data,
    )


def load_cofold_confidence_json(
    json_path: str | Path,
    *,
    confidences_path: str | Path | None = None,
    affinity_path: str | Path | None = None,
) -> ConfidenceSummary:
    """Alias for :func:`parse_cofold_confidence` loading AF3/Boltz-2/Protenix sidecar JSONs."""
    return parse_cofold_confidence(
        json_path,
        confidences_path=confidences_path,
        affinity_path=affinity_path,
    )


def load_cofolded_complex(
    structure_path: str | Path,
    *,
    ligand_resname: str | None = None,
    reference_smiles: str | None = None,
    pocket_cutoff: float = 10.0,
    keep_cofactors: Sequence[str] = (),
    preserve_bfactors: bool = True,
) -> tuple[Chem.Mol, str]:
    """Extract ``(ligand_mol_3d, pocket_pdb_text)`` from a Boltz-2 / AF3 / Chai-1 PDB or mmCIF."""
    src = Path(structure_path)
    if not src.is_file():
        raise FileNotFoundError(f"Co-folded complex file not found: {src}")

    model = _parse_biopdb_first_model(str(src), is_cif=src.suffix.lower() in (".cif", ".mmcif"))

    lig_res = _select_ligand_residue(model, ligand_resname)
    if lig_res is None:
        raise ValueError(
            f"No small-molecule HETATM ligand found in {src} (ligand_resname={ligand_resname!r})."
        )
    lig_coords = np.array([a.get_coord() for a in lig_res.get_atoms()], dtype=np.float64)

    # Format strict 80-col HETATM PDB block directly from extracted ligand atoms so
    # Chem.MolFromPDBBlock(..., proximityBonding=True) works identically for PDB & mmCIF.
    het_lines: list[str] = []
    for serial, atom in enumerate(lig_res.get_atoms(), start=1):
        elem = (atom.element or atom.get_name()[0]).strip().upper()[:2]
        aname = f"{elem}{serial}"[:4].ljust(4)
        x, y, z = atom.get_coord()
        bfac = max(-99.99, min(999.99, float(atom.get_bfactor()))) if preserve_bfactors else 20.00
        het_lines.append(
            f"HETATM{serial:5d} {aname} LIG Z   1    "
            f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00{bfac:6.2f}          {elem:>2s}"
        )
    het_lines.append("END")
    lig_mol = Chem.MolFromPDBBlock(
        "\n".join(het_lines) + "\n", sanitize=False, removeHs=False, proximityBonding=True
    )
    if lig_mol is None:
        raise ValueError(f"Failed to parse ligand coordinates from {src}.")

    if reference_smiles is not None:
        tmpl = Chem.MolFromSmiles(reference_smiles.strip())
        if tmpl is not None:
            lig_noh = Chem.RemoveHs(lig_mol, sanitize=False)
            with contextlib.suppress(Exception):
                lig_mol = AllChem.AssignBondOrdersFromTemplate(tmpl, lig_noh)
    try:
        Chem.SanitizeMol(lig_mol)
    except Exception as exc:
        raise ValueError(
            f"Ligand extracted from {src} failed RDKit sanitization (pass reference_smiles)."
        ) from exc

    # Format strict 80-col ATOM (and optional cofactor HETATM) PDB lines within pocket_cutoff A
    keep_cofactor_set = {c.strip().upper() for c in keep_cofactors if c.strip()}
    cutoff_sq = float(pocket_cutoff) ** 2
    poc_lines: list[str] = []
    atom_serial = 1
    for chain in model:
        cid = (chain.id or "A")[:1]
        for res in chain:
            if res is lig_res:
                continue
            rname_upper = res.get_resname().strip().upper()
            is_polymer = res.id[0] == " "
            if not is_polymer and rname_upper not in keep_cofactor_set:
                continue
            r_coords = np.array([a.get_coord() for a in res.get_atoms()], dtype=np.float64)
            if len(r_coords) == 0:
                continue
            d2 = np.sum((r_coords[:, None, :] - lig_coords[None, :, :]) ** 2, axis=-1)
            if float(np.min(d2)) > cutoff_sq:
                continue
            rname = res.get_resname().strip()[:3].rjust(3)
            resseq = int(res.id[1])
            record = "ATOM  " if is_polymer else "HETATM"
            for atom in res.get_atoms():
                aname = atom.get_name().strip()[:4]
                aname_fmt = f" {aname:<3s}" if len(aname) < 4 else aname
                elem = (atom.element or aname[0]).strip().upper()[:2]
                x, y, z = atom.get_coord()
                bfac = (
                    max(-99.99, min(999.99, float(atom.get_bfactor())))
                    if preserve_bfactors
                    else 20.00
                )
                coord_str = f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
                poc_lines.append(
                    f"{record}{atom_serial:5d} {aname_fmt} {rname} {cid}{resseq:4d}    "
                    f"{coord_str}  1.00{bfac:6.2f}          {elem:>2s}"
                )
                atom_serial += 1
    poc_lines.append("END")
    return lig_mol, "\n".join(poc_lines) + "\n"


def audit_cofolded_complex(
    structure_path: str | Path,
    *,
    ligand_resname: str | None = None,
    reference_smiles: str | None = None,
    property_profile: str = "oral_ro5",
    canonical_anchors: Sequence[str] = (),
    pocket_cutoff: float = 10.0,
    keep_cofactors: Sequence[str] = (),
) -> PoseAuditReport:
    """Load a Boltz-2 / AlphaFold 3 / Chai-1 complex and run the 7-axis Beyond-PoseBusters audit."""
    lig_mol, pocket_pdb_text = load_cofolded_complex(
        structure_path,
        ligand_resname=ligand_resname,
        reference_smiles=reference_smiles,
        pocket_cutoff=pocket_cutoff,
        keep_cofactors=keep_cofactors,
    )
    auditor = PoseRealityAuditor(property_profile=property_profile)
    return auditor.audit(
        lig_mol,
        pocket_pdb=pocket_pdb_text,
        name=Path(structure_path).stem,
        canonical_anchors=canonical_anchors,
    )


__all__ = [
    "ConfidenceSummary",
    "audit_cofolded_complex",
    "extract_bfactor_plddt",
    "extract_bfactor_plddt_from_text",
    "load_cofold_confidence_json",
    "load_cofolded_complex",
    "normalize_plddt_values",
    "parse_cofold_confidence",
    "parse_cofold_confidence_dict",
]
