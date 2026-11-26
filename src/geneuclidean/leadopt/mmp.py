"""Lead-Optimization Toolkit: R-Group SAR Matrices, Activity Cliffs & Avoid-ome MMP Swaps.

Inspired directly by Roche's ``mmpdb`` medicinal-chemistry lead-optimization workflow:
1. :func:`analyze_rgroup_sar`: Automated MCS-based R-group decomposition
   (``rdRGroupDecomposition``) returning core-aligned ``R1 x R2`` substituent tables and
   detecting 1-cut activity/toxicity cliffs across any congeneric series.
2. :class:`AvoidomeMMPEngine`: Pure-RDKit single-point bioisostere & polarity-tuning
   transformation engine that proposes minimal structural edits to cut clinical liabilities
   (``P(hERG)``, ``P(MDR1)``, ``P(5-HT2B)``) while preserving scaffold continuity and
   therapeutic property profiles.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from geneuclidean.cns.medchem_triage import triage_molecule
from geneuclidean.cns.mpo import compute_cns_mpo
from geneuclidean.sdk.profiles import evaluate_property_profile

if TYPE_CHECKING:
    from geneuclidean.cns.admet import CNSAdmetSuite

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ActivityCliffPair:
    """Pair of congeneric analogs sharing a core and differing at one R-group with large delta."""

    smiles_a: str
    smiles_b: str
    value_a: float
    value_b: float
    delta_value: float
    changed_rgroup: str
    r_a: str
    r_b: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles_a": self.smiles_a,
            "smiles_b": self.smiles_b,
            "value_a": round(self.value_a, 4),
            "value_b": round(self.value_b, 4),
            "delta_value": round(self.delta_value, 4),
            "changed_rgroup": self.changed_rgroup,
            "r_a": self.r_a,
            "r_b": self.r_b,
        }


@dataclass(frozen=True)
class RGroupSARResult:
    """Core-aligned R-group decomposition table and detected single-cut activity cliffs."""

    core_smarts: str
    n_input: int
    n_matched: int
    rgroup_columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    cliffs: tuple[ActivityCliffPair, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_smarts": self.core_smarts,
            "n_input": self.n_input,
            "n_matched": self.n_matched,
            "rgroup_columns": list(self.rgroup_columns),
            "rows": list(self.rows),
            "cliffs": [c.to_dict() for c in self.cliffs],
        }


def analyze_rgroup_sar(
    smiles_list: Sequence[str],
    values: Sequence[float],
    *,
    core_smarts: str | None = None,
    cliff_threshold: float = 1.0,
) -> RGroupSARResult:
    """Decompose a congeneric series around a shared core and detect 1-R-group SAR cliffs."""
    if len(smiles_list) != len(values):
        raise ValueError("smiles_list and values must have identical lengths.")
    mols: list[Chem.Mol] = []
    clean_smis: list[str] = []
    clean_vals: list[float] = []
    for s, v in zip(smiles_list, values, strict=True):
        m = Chem.MolFromSmiles(s.strip())
        if m is not None:
            mols.append(m)
            clean_smis.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))
            clean_vals.append(float(v))
    if len(mols) < 2:
        raise ValueError("analyze_rgroup_sar requires at least 2 valid molecules.")

    if core_smarts is None:
        mcs = rdFMCS.FindMCS(
            mols,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
            timeout=5,
        )
        resolved_smarts = mcs.smartsString or ""
        core_mol = Chem.MolFromSmarts(resolved_smarts) if resolved_smarts else None
    else:
        resolved_smarts = core_smarts
        core_mol = Chem.MolFromSmarts(core_smarts)
        if core_mol is None:
            core_mol = Chem.MolFromSmiles(core_smarts)

    if core_mol is None or core_mol.GetNumAtoms() < 3:
        return RGroupSARResult(
            core_smarts=resolved_smarts,
            n_input=len(smiles_list),
            n_matched=0,
            rgroup_columns=(),
            rows=(),
            cliffs=(),
        )

    rgd: Any = importlib.import_module("rdkit.Chem.rdRGroupDecomposition")
    res_list, _unmatched = rgd.RGroupDecompose([core_mol], mols, asSmiles=True, asRows=True)
    r_cols = sorted({k for row in res_list for k in row if k.startswith("R")})
    rows: list[dict[str, Any]] = []
    for idx, r_dict in enumerate(res_list):
        if idx >= len(clean_smis):
            break
        entry: dict[str, Any] = {
            "smiles": clean_smis[idx],
            "value": round(clean_vals[idx], 4),
            "Core": r_dict.get("Core", ""),
        }
        for col in r_cols:
            entry[col] = r_dict.get(col, "")
        rows.append(entry)

    # Detect single-R-group SAR cliffs (pairs identical at all R-positions except one)
    cliffs: list[ActivityCliffPair] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            diffs = [c for c in r_cols if rows[i][c] != rows[j][c]]
            if len(diffs) == 1:
                dv = abs(float(rows[i]["value"]) - float(rows[j]["value"]))
                if dv >= float(cliff_threshold):
                    col = diffs[0]
                    cliffs.append(
                        ActivityCliffPair(
                            smiles_a=str(rows[i]["smiles"]),
                            smiles_b=str(rows[j]["smiles"]),
                            value_a=float(rows[i]["value"]),
                            value_b=float(rows[j]["value"]),
                            delta_value=dv,
                            changed_rgroup=col,
                            r_a=str(rows[i][col]),
                            r_b=str(rows[j][col]),
                        )
                    )
    cliffs.sort(key=lambda c: c.delta_value, reverse=True)
    return RGroupSARResult(
        core_smarts=resolved_smarts,
        n_input=len(smiles_list),
        n_matched=len(rows),
        rgroup_columns=tuple(r_cols),
        rows=tuple(rows),
        cliffs=tuple(cliffs),
    )


# Curated Roche/Pfizer MedChem 1-cut bioisostere & liability-tuning MMP transforms
_MEDCHEM_MMP_TRANSFORMS: tuple[tuple[str, str, str], ...] = (
    (
        "piperidine_to_morpholine",
        "[CH2:1]1[CH2:2][N:3][CH2:4][CH2:5][CH2:6]1>>[O:1]1[CH2:2][N:3][CH2:4][CH2:5][CH2:6]1",
        "Drops basicity (~2 pKa units) & hERG",
    ),
    ("aryl_H_to_F", "[cH1:1]>>[c:1]F", "Blocks CYP metabolism & tunes lipophilicity"),
    ("aryl_H_to_CN", "[cH1:1]>>[c:1]C#N", "Reduces cLogP & hERG aromatic pi-stacking"),
    (
        "phenyl_to_pyridine_aza",
        "[cH1:1]1[c:2][c:3][c:4][c:5][c:6]1>>[n:1]1[c:2][c:3][c:4][c:5][c:6]1",
        "Aza-scan lowers LogP & hERG",
    ),
    ("aryl_Cl_to_F", "[c:1]Cl>>[c:1]F", "Reduces lipophilicity & MW"),
    ("aryl_OMe_to_OCF3", "[c:1]OC>>[c:1]OC(F)(F)F", "Blocks O-demethylation clearance"),
    ("aryl_Me_to_F", "[c:1][CH3]>>[c:1]F", "Blocks benzylic oxidation & cuts LogP"),
    ("aryl_OMe_to_F", "[c:1]OC>>[c:1]F", "Reduces TPSA/HBA & MDR1 efflux"),
    ("n_methyl_to_cyclopropyl", "[N:1][CH3]>>[N:1]C2CC2", "Modulates amine pKa & hERG"),
    ("n_ethyl_to_methyl", "[N:1][CH2][CH3]>>[N:1]C", "Trims lipophilicity & MW"),
    ("n_methyl_to_hydroxyethyl", "[N:1][CH3]>>[N:1]CCO", "Reduces lipophilicity & basicity"),
    ("aryl_H_to_OMe", "[cH1:1]>>[c:1]OC", "Modulates electronics & solubility"),
    ("aryl_F_to_CN", "[c:1]F>>[c:1]C#N", "Increases polarity / lowers hERG"),
    (
        "secondary_amide_to_n_methyl",
        "[C:1](=[O:2])[NH1:3][#6:4]>>[C:1](=[O:2])[N:3](C)[#6:4]",
        "Removes HBD to boost BBB & cut MDR1",
    ),
    ("aryl_H_to_OH", "[cH1:1]>>[c:1]O", "Adds polar H-bond handle & lowers LogP"),
)


@dataclass(frozen=True)
class AvoidomeMMPProposal:
    """Single 1-cut bioisostere/MMP transformation scored on Avoid-ome & MedChem profile."""

    parent_smiles: str
    analog_smiles: str
    transform_name: str
    rationale: str
    cns_mpo_v2: float
    delta_mpo_v2: float
    sa_score: float
    profile_score: float
    herg_prob: float | None = None
    delta_herg_reduction: float | None = None
    mdr1_prob: float | None = None
    delta_mdr1_reduction: float | None = None
    htr2b_prob: float | None = None
    delta_htr2b_reduction: float | None = None
    rescue_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_smiles": self.parent_smiles,
            "analog_smiles": self.analog_smiles,
            "transform_name": self.transform_name,
            "rationale": self.rationale,
            "cns_mpo_v2": round(self.cns_mpo_v2, 3),
            "delta_mpo_v2": round(self.delta_mpo_v2, 3),
            "sa_score": round(self.sa_score, 3),
            "profile_score": round(self.profile_score, 4),
            "herg_prob": round(self.herg_prob, 4) if self.herg_prob is not None else None,
            "delta_herg_reduction": (
                round(self.delta_herg_reduction, 4)
                if self.delta_herg_reduction is not None
                else None
            ),
            "mdr1_prob": round(self.mdr1_prob, 4) if self.mdr1_prob is not None else None,
            "delta_mdr1_reduction": (
                round(self.delta_mdr1_reduction, 4)
                if self.delta_mdr1_reduction is not None
                else None
            ),
            "htr2b_prob": round(self.htr2b_prob, 4) if self.htr2b_prob is not None else None,
            "delta_htr2b_reduction": (
                round(self.delta_htr2b_reduction, 4)
                if self.delta_htr2b_reduction is not None
                else None
            ),
            "rescue_score": round(self.rescue_score, 4),
        }


class AvoidomeMMPEngine:
    """Propose minimal 1-cut MedChem bioisostere swaps to de-risk hERG/MDR1/5-HT2B liabilities."""

    def __init__(
        self,
        *,
        admet_suite: CNSAdmetSuite | None = None,
        property_profile: str = "cns_mpo_v2",
    ) -> None:
        self.admet_suite = admet_suite
        self.property_profile = str(property_profile)
        self._compiled: list[tuple[str, Any, str]] = []
        for name, smirks, rationale in _MEDCHEM_MMP_TRANSFORMS:
            rxn = AllChem.ReactionFromSmarts(smirks)  # type: ignore[attr-defined]
            if rxn is not None:
                self._compiled.append((name, rxn, rationale))

    def propose_rescues(
        self,
        parent_smiles: str,
        *,
        max_proposals: int = 25,
        triage_filter: bool = True,
    ) -> list[AvoidomeMMPProposal]:
        """Generate and rank clean 1-cut bioisostere rescues for ``parent_smiles``."""
        parent_mol = Chem.MolFromSmiles(parent_smiles.strip())
        if parent_mol is None:
            raise ValueError(f"Invalid parent_smiles: {parent_smiles!r}")
        parent_can = Chem.MolToSmiles(parent_mol, canonical=True, isomericSmiles=True)
        parent_mpo = compute_cns_mpo(parent_mol)

        seen: set[str] = {parent_can}
        raw_analogs: list[tuple[str, str, str]] = []
        for tname, rxn, rationale in self._compiled:
            try:
                prods = rxn.RunReactants((parent_mol,))
            except Exception:
                continue
            for ptup in prods:
                if not ptup:
                    continue
                pmol = ptup[0]
                try:
                    Chem.SanitizeMol(pmol)
                    psmi = Chem.MolToSmiles(pmol, canonical=True, isomericSmiles=True)
                except Exception:
                    continue
                if not psmi or "." in psmi or psmi in seen:
                    continue
                if triage_filter and not triage_molecule(psmi).triage_pass:
                    continue
                seen.add(psmi)
                raw_analogs.append((psmi, tname, rationale))

        if not raw_analogs:
            return []

        parent_probs: dict[str, float] = {}
        analog_probs_list: list[dict[str, float]] = [{} for _ in raw_analogs]
        if (
            self.admet_suite is not None
            and len(getattr(self.admet_suite, "fitted_endpoints", ())) > 3
        ):
            all_preds = self.admet_suite.predict_v4([parent_can, *[a[0] for a in raw_analogs]])
            parent_probs = dict(all_preds[0].probabilities)
            analog_probs_list = [dict(p.probabilities) for p in all_preds[1:]]

        proposals: list[AvoidomeMMPProposal] = []
        for (asmi, tname, rationale), aprobs in zip(raw_analogs, analog_probs_list, strict=True):
            mpo = compute_cns_mpo(asmi)
            prof = evaluate_property_profile(asmi, profile=self.property_profile)
            d_mpo = float(mpo.cns_mpo_v2 - parent_mpo.cns_mpo_v2)

            herg = aprobs.get("herg")
            mdr1 = aprobs.get("mdr1_efflux")
            htr2b = aprobs.get("htr2b_avoid")
            d_herg = (
                float(parent_probs["herg"] - herg)
                if (herg is not None and "herg" in parent_probs)
                else None
            )
            d_mdr1 = (
                float(parent_probs["mdr1_efflux"] - mdr1)
                if (mdr1 is not None and "mdr1_efflux" in parent_probs)
                else None
            )
            d_htr2b = (
                float(parent_probs["htr2b_avoid"] - htr2b)
                if (htr2b is not None and "htr2b_avoid" in parent_probs)
                else None
            )

            safety_gain = 0.50 * (d_herg or 0.0) + 0.30 * (d_mdr1 or 0.0) + 0.20 * (d_htr2b or 0.0)
            rescue = float(0.55 * prof.score + 0.45 * max(-0.5, min(1.0, 0.5 + safety_gain)))
            proposals.append(
                AvoidomeMMPProposal(
                    parent_smiles=parent_can,
                    analog_smiles=asmi,
                    transform_name=tname,
                    rationale=rationale,
                    cns_mpo_v2=float(mpo.cns_mpo_v2),
                    delta_mpo_v2=d_mpo,
                    sa_score=float(mpo.sa_score),
                    profile_score=float(prof.score),
                    herg_prob=float(herg) if herg is not None else None,
                    delta_herg_reduction=d_herg,
                    mdr1_prob=float(mdr1) if mdr1 is not None else None,
                    delta_mdr1_reduction=d_mdr1,
                    htr2b_prob=float(htr2b) if htr2b is not None else None,
                    delta_htr2b_reduction=d_htr2b,
                    rescue_score=rescue,
                )
            )

        proposals.sort(key=lambda p: p.rescue_score, reverse=True)
        return proposals[: int(max_proposals)]


__all__ = [
    "ActivityCliffPair",
    "AvoidomeMMPEngine",
    "AvoidomeMMPProposal",
    "RGroupSARResult",
    "analyze_rgroup_sar",
]
