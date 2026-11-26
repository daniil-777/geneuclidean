"""Physiological pH 7.4 tautomer & protomer microstate auditor for 3D protein-ligand poses.

Scientific motivation:
    Recent 2025-2026 evaluations of co-folding models (Boltz-2, AlphaFold 3, Chai-1) and
    classical docking engines demonstrate two recurring electrostatic failure modes missed by
    PoseBusters:

    1. **Unprepared Protonation State / Charge Mismatch:** Supplying or predicting a neutral
       basic amine (``+0``) facing an anionic ``ASP``/``GLU`` carboxylate (or a neutral
       carboxylic acid facing ``ARG``/``LYS``), yielding a physically incorrect H-bond /
       salt-bridge geometry at physiological ``pH = 7.4``.
    2. **Unstable Minor Tautomers:** Evaluating or docking an aliphatic enol, imidic acid,
       or de-aromatized heterocycle tautomer rather than the dominant physiological keto /
       amide / aromatic tautomer.

    This pure-RDKit module enumerates canonical tautomers and pH 7.4 microstates, scores
    tautomer stability via classic Milletti/Sayle MedChem heuristics, and audits a 3D pose
    against optional pocket charged residues for electrostatic polarity mismatches.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from geneuclidean.evaluation.protonation import (
    analyze_protonation_ph74,
    protonate_mol_ph74,
)

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

# Classic medicinal-chemistry tautomer heuristic patterns (Sayle / Milletti / RDKit score)
_TAUTOMER_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("aromatic_ring", "[a]", 0.35),
    ("amide_carbonyl", "[#7;H1,H2,H0]-[CX3](=[OX1])", 2.0),
    ("urea_carbonyl", "[#7]-[CX3](=[OX1])-[#7]", 1.5),
    ("ketone_or_aldehyde", "[#6]-[CX3](=[OX1])-[#6,#1]", 1.0),
    ("enol_penalty", "[CX3]=[CX3]-[OX2H1;!$(O[c])]", -2.5),
    ("imidic_acid_penalty", "[NX2]=[CX3]-[OX2H1]", -2.5),
    ("enamine_penalty", "[CX3]=[CX3]-[NX3;H1,H2;!$(N[c])]", -1.0),
    ("charged_separation_penalty", "[+1;!$([N+](=O)[O-])]~[-1;!$([O-][N+]=O)]", -1.5),
)

_COMPILED_TAUT_PATTERNS: tuple[tuple[str, Chem.Mol, float], ...] | None = None


def _get_taut_patterns() -> tuple[tuple[str, Chem.Mol, float], ...]:
    global _COMPILED_TAUT_PATTERNS
    if _COMPILED_TAUT_PATTERNS is None:
        out: list[tuple[str, Chem.Mol, float]] = []
        for name, sm, weight in _TAUTOMER_PATTERNS:
            pat = Chem.MolFromSmarts(sm)
            if pat is not None:
                out.append((name, pat, weight))
        _COMPILED_TAUT_PATTERNS = tuple(out)
    return _COMPILED_TAUT_PATTERNS


def _score_tautomer_stability(mol: Chem.Mol) -> float:
    """Heuristic physiological tautomer stability score (higher is more favorable)."""
    score = 0.0
    for _name, pat, weight in _get_taut_patterns():
        hits = mol.GetSubstructMatches(pat)
        if hits:
            score += weight * float(len(hits))
    # Bonus for preserving complete aromatic rings (benzo/heteroaromatic stabilization)
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            score += 2.0
    return float(score)


@dataclass(frozen=True)
class MicrostateCandidate:
    """Single enumerated tautomer/protomer microstate at physiological pH 7.4."""

    canonical_smiles: str
    protonated_smiles_ph74: str
    net_charge_ph74: int
    stability_score: float
    is_canonical_tautomer: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "protonated_smiles_ph74": self.protonated_smiles_ph74,
            "net_charge_ph74": self.net_charge_ph74,
            "stability_score": round(self.stability_score, 3),
            "is_canonical_tautomer": self.is_canonical_tautomer,
        }


@dataclass(frozen=True)
class TautomerMicrostateAudit:
    """Audit result for input pose protonation, tautomer stability, and pocket electrostatics."""

    input_smiles: str
    expected_ph74_smiles: str
    input_formal_charge: int
    expected_ph74_charge: int
    charge_matches_ph74: bool
    n_tautomers_enumerated: int
    input_tautomer_score: float
    best_tautomer_score: float
    tautomer_stability_gap: float
    tautomer_plausible: bool
    n_like_charge_repulsions: int
    n_unprotonated_salt_bridge_misses: int
    electrostatic_warnings: tuple[str, ...]
    passed: bool
    top_microstates: tuple[MicrostateCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_smiles": self.input_smiles,
            "expected_ph74_smiles": self.expected_ph74_smiles,
            "input_formal_charge": self.input_formal_charge,
            "expected_ph74_charge": self.expected_ph74_charge,
            "charge_matches_ph74": self.charge_matches_ph74,
            "n_tautomers_enumerated": self.n_tautomers_enumerated,
            "input_tautomer_score": round(self.input_tautomer_score, 3),
            "best_tautomer_score": round(self.best_tautomer_score, 3),
            "tautomer_stability_gap": round(self.tautomer_stability_gap, 3),
            "tautomer_plausible": self.tautomer_plausible,
            "n_like_charge_repulsions": self.n_like_charge_repulsions,
            "n_unprotonated_salt_bridge_misses": self.n_unprotonated_salt_bridge_misses,
            "electrostatic_warnings": list(self.electrostatic_warnings),
            "passed": self.passed,
            "top_microstates": [m.to_dict() for m in self.top_microstates],
        }


def enumerate_tautomer_microstates_ph74(
    mol_or_smiles: str | Chem.Mol,
    *,
    max_tautomers: int = 16,
) -> list[MicrostateCandidate]:
    """Enumerate ranked tautomer + physiological pH 7.4 protomer microstates."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles.strip())
        if mol is None:
            raise ValueError(f"Invalid SMILES for tautomer enumeration: {mol_or_smiles!r}")
    else:
        mol = Chem.RemoveHs(Chem.Mol(mol_or_smiles))

    # Neutralize simple protonated amines/acids before enumerating tautomers cleanly
    uncharger = rdMolStandardize.Uncharger()
    neutral_base = uncharger.uncharge(Chem.Mol(mol))
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(int(max(2, max_tautomers)))
    enumerator.SetMaxTransforms(int(max(16, max_tautomers * 4)))
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)

    try:
        can_taut = enumerator.Canonicalize(neutral_base)
        can_taut_smi = Chem.MolToSmiles(can_taut, canonical=True, isomericSmiles=True)
    except Exception:
        can_taut_smi = Chem.MolToSmiles(neutral_base, canonical=True, isomericSmiles=True)

    try:
        tauts = list(enumerator.Enumerate(neutral_base))
    except Exception:
        tauts = [neutral_base]
    if not tauts:
        tauts = [neutral_base]

    seen: set[str] = set()
    candidates: list[MicrostateCandidate] = []
    for tm in tauts[: max(1, int(max_tautomers))]:
        try:
            Chem.SanitizeMol(tm)
        except Exception:
            continue
        smi_n = Chem.MolToSmiles(tm, canonical=True, isomericSmiles=True)
        if smi_n in seen:
            continue
        seen.add(smi_n)
        prot_mol = protonate_mol_ph74(tm, preserve_coords=False)
        prot_smi = Chem.MolToSmiles(prot_mol, canonical=True, isomericSmiles=True)
        net_q = int(Chem.GetFormalCharge(prot_mol))
        stab = _score_tautomer_stability(tm)
        candidates.append(
            MicrostateCandidate(
                canonical_smiles=smi_n,
                protonated_smiles_ph74=prot_smi,
                net_charge_ph74=net_q,
                stability_score=stab,
                is_canonical_tautomer=(smi_n == can_taut_smi),
            )
        )

    candidates.sort(
        key=lambda c: (round(c.stability_score, 4), int(c.is_canonical_tautomer)),
        reverse=True,
    )
    return candidates


def _extract_pocket_charged_atoms(
    pocket_pdb: str | Path,
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    """Extract anionic (ASP/GLU carboxylate O) and cationic (ARG/LYS N) 3D pocket coordinates."""
    text = (
        pocket_pdb
        if isinstance(pocket_pdb, str) and ("\n" in pocket_pdb or "ATOM  " in pocket_pdb)
        else Path(pocket_pdb).read_text(encoding="utf-8", errors="ignore")
    )
    anionic: list[tuple[str, np.ndarray]] = []
    cationic: list[tuple[str, np.ndarray]] = []
    for line in io.StringIO(text):
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        resseq = line[22:26].strip()
        try:
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
        except ValueError:
            continue
        label = f"{resname}{resseq}"
        if (resname == "ASP" and atom_name in ("OD1", "OD2")) or (
            resname == "GLU" and atom_name in ("OE1", "OE2")
        ):
            anionic.append((label, xyz))
        elif (resname == "LYS" and atom_name == "NZ") or (
            resname == "ARG" and atom_name in ("NH1", "NH2", "NE")
        ):
            cationic.append((label, xyz))
    return anionic, cationic


def audit_tautomer_microstate_ph74(
    mol_or_smiles: str | Chem.Mol,
    *,
    pocket_pdb: str | Path | None = None,
    max_tautomers: int = 16,
    tautomer_gap_threshold: float = 2.0,
    strict_input_charge: bool = False,
) -> TautomerMicrostateAudit:
    """Audit physiological pH 7.4 tautomer stability, charge state, and pocket electrostatics.

    Args:
        mol_or_smiles: Input ligand SMILES or 3D RDKit ``Chem.Mol``.
        pocket_pdb: Optional binding-pocket PDB path or PDB text for 3D electrostatic checks.
        max_tautomers: Maximum enumerated tautomers.
        tautomer_gap_threshold: Max allowed stability gap below the top tautomer before flagging
            an unstable minor tautomer (e.g., enol or imidic acid).
        strict_input_charge: When ``True``, fails if the input ligand lacks pH 7.4 formal
            charges (useful when auditing prepared docking/Boltz-2 inputs). Default ``False``
            reports ``charge_matches_ph74`` as an explicit diagnostic without falsely failing
            unprotonated crystallographic PDB depositions.
    """
    mol = (
        Chem.MolFromSmiles(mol_or_smiles.strip())
        if isinstance(mol_or_smiles, str)
        else Chem.RemoveHs(Chem.Mol(mol_or_smiles))
    )
    if mol is None:
        raise ValueError("Invalid molecule provided to audit_tautomer_microstate_ph74.")

    input_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    input_q = int(Chem.GetFormalCharge(mol))
    prot_res = analyze_protonation_ph74(mol)
    expected_q = int(prot_res.net_charge)
    charge_match = input_q == expected_q

    uncharger = rdMolStandardize.Uncharger()
    input_neutral = uncharger.uncharge(Chem.Mol(mol))
    input_taut_score = _score_tautomer_stability(input_neutral)

    microstates = enumerate_tautomer_microstates_ph74(mol, max_tautomers=max_tautomers)
    best_taut_score = (
        max(m.stability_score for m in microstates) if microstates else input_taut_score
    )
    taut_gap = float(max(0.0, best_taut_score - input_taut_score))
    taut_plausible = taut_gap <= float(tautomer_gap_threshold)

    warnings_list: list[str] = []
    n_like_repulsions = 0
    n_unprot_misses = 0

    if not charge_match:
        warnings_list.append(
            f"Input formal charge ({input_q:+d}) differs from physiological pH 7.4 "
            f"expected charge ({expected_q:+d})."
        )
    if not taut_plausible:
        warnings_list.append(
            f"Minor/unstable tautomer detected (stability gap {taut_gap:.2f} > "
            f"{tautomer_gap_threshold:.2f})."
        )

    # Optional 3D pocket electrostatic polarity audit
    if pocket_pdb is not None and mol.GetNumConformers() > 0:
        anionic_rec, cationic_rec = _extract_pocket_charged_atoms(pocket_pdb)
        conf = mol.GetConformer()
        seen_repulsions: set[tuple[int, str]] = set()
        seen_misses: set[tuple[int, str]] = set()

        for lig_idx in prot_res.protonated_atom_indices:
            if lig_idx >= mol.GetNumAtoms():
                continue
            pos = conf.GetAtomPosition(lig_idx)
            lxyz = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
            lig_q = int(mol.GetAtomWithIdx(lig_idx).GetFormalCharge())
            # Unprotonated basic amine directly facing anionic ASP/GLU carboxylate (<4.2 A)
            if lig_q <= 0:
                for res_label, rxyz in anionic_rec:
                    dist = float(np.linalg.norm(lxyz - rxyz))
                    if dist <= 4.2 and (lig_idx, res_label) not in seen_misses:
                        seen_misses.add((lig_idx, res_label))
                        n_unprot_misses += 1
                        warnings_list.append(
                            f"Basic amine atom {lig_idx} is neutral (+0) while {dist:.2f} A from "
                            f"anionic {res_label} (missing pH 7.4 salt-bridge protonation)."
                        )
            # Like-charge cationic repulsion (<3.5 A to ARG/LYS)
            for res_label, rxyz in cationic_rec:
                dist = float(np.linalg.norm(lxyz - rxyz))
                if dist <= 3.5 and (lig_idx, res_label) not in seen_repulsions:
                    seen_repulsions.add((lig_idx, res_label))
                    n_like_repulsions += 1
                    warnings_list.append(
                        f"Like-charge cationic clash: basic site {lig_idx} is {dist:.2f} A from "
                        f"cationic {res_label}."
                    )

        for lig_idx in prot_res.deprotonated_atom_indices:
            if lig_idx >= mol.GetNumAtoms():
                continue
            pos = conf.GetAtomPosition(lig_idx)
            lxyz = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
            for res_label, rxyz in anionic_rec:
                dist = float(np.linalg.norm(lxyz - rxyz))
                if dist <= 3.5 and (lig_idx, res_label) not in seen_repulsions:
                    seen_repulsions.add((lig_idx, res_label))
                    n_like_repulsions += 1
                    warnings_list.append(
                        f"Like-charge anionic clash: acidic site {lig_idx} is {dist:.2f} A from "
                        f"anionic {res_label}."
                    )

    passed = (
        taut_plausible and (n_like_repulsions == 0) and (charge_match or not strict_input_charge)
    )
    return TautomerMicrostateAudit(
        input_smiles=input_smi,
        expected_ph74_smiles=prot_res.protonated_smiles,
        input_formal_charge=input_q,
        expected_ph74_charge=expected_q,
        charge_matches_ph74=charge_match,
        n_tautomers_enumerated=len(microstates),
        input_tautomer_score=input_taut_score,
        best_tautomer_score=best_taut_score,
        tautomer_stability_gap=taut_gap,
        tautomer_plausible=taut_plausible,
        n_like_charge_repulsions=n_like_repulsions,
        n_unprotonated_salt_bridge_misses=n_unprot_misses,
        electrostatic_warnings=tuple(warnings_list),
        passed=passed,
        top_microstates=tuple(microstates[:5]),
    )


__all__ = [
    "MicrostateCandidate",
    "TautomerMicrostateAudit",
    "audit_tautomer_microstate_ph74",
    "enumerate_tautomer_microstates_ph74",
]
