"""Boltz-2 / AlphaFold 3 / Chai-1 co-folded complex loader & 7-axis PoseRealityAuditor bridge.

Ingests standard ``.cif`` (mmCIF) or ``.pdb`` biomolecular co-folding predictions, cleanly
extracts the small-molecule ligand and surrounding protein binding pocket (preserving
cofactors when requested), optionally restores exact covalent bond orders from the input
SMILES template via ``AllChem.AssignBondOrdersFromTemplate``, and feeds the complex
directly into :class:`~geneuclidean.audit.PoseRealityAuditor`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from geneuclidean.audit.pose_auditor import PoseAuditReport, PoseRealityAuditor

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

_IGNORE_HETATMS: frozenset[str] = frozenset(
    {"HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "SO4", "PO4", "GOL", "EDO"}
)


def load_cofolded_complex(
    structure_path: str | Path,
    *,
    ligand_resname: str | None = None,
    reference_smiles: str | None = None,
    pocket_cutoff: float = 10.0,
) -> tuple[Chem.Mol, str]:
    """Extract ``(ligand_mol_3d, pocket_pdb_text)`` from a Boltz-2 / AF3 / Chai-1 PDB or mmCIF."""
    src = Path(structure_path)
    if not src.is_file():
        raise FileNotFoundError(f"Co-folded complex file not found: {src}")

    parser = (
        MMCIFParser(QUIET=True)
        if src.suffix.lower() in (".cif", ".mmcif")
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("cofold", str(src))
    model = next(structure.get_models())

    # Identify target small-molecule ligand HETATM residue (largest non-water HET if unspecified)
    cand_ligs: list[tuple[int, Any]] = []
    for chain in model:
        for res in chain:
            hetflag = res.id[0]
            rname = res.get_resname().strip().upper()
            if hetflag == " " or rname in _IGNORE_HETATMS:
                continue
            if ligand_resname is not None and rname != ligand_resname.strip().upper():
                continue
            n_heavy = sum(1 for a in res.get_atoms() if (a.element or "").upper() != "H")
            if n_heavy >= 4:
                cand_ligs.append((n_heavy, res))

    if not cand_ligs:
        raise ValueError(
            f"No small-molecule HETATM ligand found in {src} (ligand_resname={ligand_resname!r})."
        )
    cand_ligs.sort(key=lambda x: x[0], reverse=True)
    lig_res = cand_ligs[0][1]
    lig_coords = np.array([a.get_coord() for a in lig_res.get_atoms()], dtype=np.float64)

    # Format strict 80-col HETATM PDB block directly from extracted ligand atoms so
    # Chem.MolFromPDBBlock(..., proximityBonding=True) works identically for PDB & mmCIF.
    het_lines: list[str] = []
    for serial, atom in enumerate(lig_res.get_atoms(), start=1):
        elem = (atom.element or atom.get_name()[0]).strip().upper()[:2]
        aname = f"{elem}{serial}"[:4].ljust(4)
        x, y, z = atom.get_coord()
        het_lines.append(
            f"HETATM{serial:5d} {aname} LIG Z   1    "
            f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00 20.00          {elem:>2s}"
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

    # Format strict 80-col ATOM PDB lines directly for pocket residues within pocket_cutoff A
    cutoff_sq = float(pocket_cutoff) ** 2
    poc_lines: list[str] = []
    atom_serial = 1
    for chain in model:
        cid = (chain.id or "A")[:1]
        for res in chain:
            if res is lig_res or res.id[0] != " ":
                continue
            r_coords = np.array([a.get_coord() for a in res.get_atoms()], dtype=np.float64)
            if len(r_coords) == 0:
                continue
            d2 = np.sum((r_coords[:, None, :] - lig_coords[None, :, :]) ** 2, axis=-1)
            if float(np.min(d2)) > cutoff_sq:
                continue
            rname = res.get_resname().strip()[:3].rjust(3)
            resseq = int(res.id[1])
            for atom in res.get_atoms():
                aname = atom.get_name().strip()[:4]
                aname_fmt = f" {aname:<3s}" if len(aname) < 4 else aname
                elem = (atom.element or aname[0]).strip().upper()[:2]
                x, y, z = atom.get_coord()
                poc_lines.append(
                    f"ATOM  {atom_serial:5d} {aname_fmt} {rname} {cid}{resseq:4d}    "
                    f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00 20.00          {elem:>2s}"
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
) -> PoseAuditReport:
    """Load a Boltz-2 / AlphaFold 3 / Chai-1 complex and run the 7-axis Beyond-PoseBusters audit."""
    lig_mol, pocket_pdb_text = load_cofolded_complex(
        structure_path,
        ligand_resname=ligand_resname,
        reference_smiles=reference_smiles,
        pocket_cutoff=pocket_cutoff,
    )
    auditor = PoseRealityAuditor(property_profile=property_profile)
    return auditor.audit(
        lig_mol,
        pocket_pdb=pocket_pdb_text,
        name=Path(structure_path).stem,
        canonical_anchors=canonical_anchors,
    )


__all__ = [
    "audit_cofolded_complex",
    "load_cofolded_complex",
]
