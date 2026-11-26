"""Natural-product & clinical scaffold-hopping evaluation with Avoid-ome de-risking.

Scientific motivation:
    Inspired directly by ETH Zurich MODLAB's 2026 natural-product scaffold-hopping studies
    (``COLIBRI``: Pachon-Angona & Schneider, *ChemRxiv* Sep 2026 — ibogaine -> human SERT
    scaffold hops; Schriefer Treiber & Schneider, *ChemRxiv* Aug 2026 — mitragynine
    pseudoindoxyl -> mu-opioid agonists), this module provides a quantitative 5-axis
    audit comparing *de novo* or scaffold-decorated candidates against a complex reference
    ligand:

    1. **Topological Scaffold Novelty:** Exact & generic Bemis-Murcko hop + ECFP4 Tanimoto.
    2. **Synthetic Tractability Simplification:** ``DeltaSA = SA(ref) - SA(cand)`` (positive
       means simpler synthesis than the complex natural-product cage, e.g., ibogaine ``SA~5.0``).
    3. **CNS Drug-Likeness & Basic-Amine Pharmacophore:** ``DeltaMPO_v2`` and pH 7.4 charge.
    4. **Clinical Liability De-Risking:** Exact v4.0 Avoid-ome risk reduction across ``hERG``
       (ibogaine's fatal QT-prolongation / torsades liability), ``5-HT2B`` valvulopathy,
       ``MDR1`` efflux, and ``CYP2D6``.
    5. **Industrial MedChem Hygiene:** Strict zero-PAINS / zero-reagent quarantine gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from geneuclidean.cns.admet import CNSAdmetSuite
from geneuclidean.cns.medchem_triage import bemis_murcko_scaffold, triage_molecule
from geneuclidean.cns.mpo import compute_cns_mpo
from geneuclidean.evaluation.protonation import analyze_protonation_ph74

# Verified RDKit canonical isomeric SMILES for ETH MODLAB / CNS natural-product benchmarks:
# - (-)-Ibogaine: PubChem CID 197060 / RCSB PDB IBG (InChIKey HSIBGVUMFOSJPD-CFDPKNGZSA-N)
# - Paroxetine: RCSB PDB 5I6X co-crystal ligand 8PR (InChIKey AHOUBRCZNHFOSL-YOEHRIQHSA-N)
IBOGAINE_SMILES = "CC[C@H]1C[C@@H]2C[C@H]3c4[nH]c5ccc(OC)cc5c4CCN(C2)[C@@H]13"
PAROXETINE_SMILES = "Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3c(c2)OCO3)cc1"


@dataclass(frozen=True)
class ScaffoldHopCandidateAudit:
    """Five-axis audit comparing one generated candidate against a reference scaffold."""

    smiles: str
    canonical_smiles: str
    reference_smiles: str
    valid: bool
    is_exact_scaffold_hop: bool
    is_generic_scaffold_hop: bool
    ecfp4_tanimoto_to_ref: float
    cns_mpo_v2: float
    delta_mpo_v2: float
    sa_score: float
    delta_sa_simplification: float
    net_charge_ph74: int
    has_basic_amine: bool
    triage_passed: bool
    hygiene_gate: float
    herg_prob: float | None = None
    delta_herg_risk_reduction: float | None = None
    htr2b_prob: float | None = None
    delta_htr2b_risk_reduction: float | None = None
    mdr1_prob: float | None = None
    v4_pharma_safe: bool | None = None
    hop_composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate scaffold-hop audit to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "reference_smiles": self.reference_smiles,
            "valid": self.valid,
            "is_exact_scaffold_hop": self.is_exact_scaffold_hop,
            "is_generic_scaffold_hop": self.is_generic_scaffold_hop,
            "ecfp4_tanimoto_to_ref": round(self.ecfp4_tanimoto_to_ref, 4),
            "cns_mpo_v2": round(self.cns_mpo_v2, 3),
            "delta_mpo_v2": round(self.delta_mpo_v2, 3),
            "sa_score": round(self.sa_score, 3),
            "delta_sa_simplification": round(self.delta_sa_simplification, 3),
            "net_charge_ph74": self.net_charge_ph74,
            "has_basic_amine": self.has_basic_amine,
            "triage_passed": self.triage_passed,
            "hygiene_gate": round(self.hygiene_gate, 4),
            "herg_prob": round(self.herg_prob, 4) if self.herg_prob is not None else None,
            "delta_herg_risk_reduction": (
                round(self.delta_herg_risk_reduction, 4)
                if self.delta_herg_risk_reduction is not None
                else None
            ),
            "htr2b_prob": round(self.htr2b_prob, 4) if self.htr2b_prob is not None else None,
            "delta_htr2b_risk_reduction": (
                round(self.delta_htr2b_risk_reduction, 4)
                if self.delta_htr2b_risk_reduction is not None
                else None
            ),
            "mdr1_prob": round(self.mdr1_prob, 4) if self.mdr1_prob is not None else None,
            "v4_pharma_safe": self.v4_pharma_safe,
            "hop_composite_score": round(self.hop_composite_score, 4),
        }


def _ecfp4_bv(smiles: str) -> Any | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)  # type: ignore[attr-defined]


def audit_scaffold_hop_campaign(
    candidate_smiles: Sequence[str],
    reference_smiles: str,
    *,
    admet_suite: CNSAdmetSuite | None = None,
    max_tanimoto_for_hop: float = 0.50,
    require_basic_amine: bool = True,
) -> dict[str, Any]:
    """Evaluate a batch of scaffold-hopping candidates relative to a reference drug/natural product.

    Returns a dictionary containing the reference profile, campaign-level summary metrics
    (genuine hop rate, hERG de-risking rate, mean SA simplification), and ranked per-candidate
    :class:`ScaffoldHopCandidateAudit` entries.
    """
    ref_mpo = compute_cns_mpo(reference_smiles)
    if not ref_mpo.valid or not ref_mpo.canonical_smiles:
        raise ValueError(f"Invalid reference_smiles: {reference_smiles!r}")
    ref_canon = ref_mpo.canonical_smiles
    ref_exact_scaf = bemis_murcko_scaffold(ref_canon, generic=False)
    ref_gen_scaf = bemis_murcko_scaffold(ref_canon, generic=True)
    ref_fp = _ecfp4_bv(ref_canon)
    ref_prot = analyze_protonation_ph74(ref_canon)

    ref_probs: dict[str, float] = {}
    if admet_suite is not None and len(admet_suite.fitted_endpoints) > 0:
        ref_pred = admet_suite.predict_v4([ref_canon])[0]
        ref_probs = dict(ref_pred.probabilities)

    cand_list = list(candidate_smiles)
    v4_preds = (
        admet_suite.predict_v4(cand_list)
        if (admet_suite is not None and len(admet_suite.fitted_endpoints) > 0 and cand_list)
        else None
    )

    audits: list[ScaffoldHopCandidateAudit] = []
    seen_canon: set[str] = set()
    for idx, smi in enumerate(cand_list):
        mpo = compute_cns_mpo(smi)
        if not mpo.valid or not mpo.canonical_smiles:
            continue
        if mpo.canonical_smiles in seen_canon:
            continue
        seen_canon.add(mpo.canonical_smiles)

        tr = triage_molecule(mpo.canonical_smiles)
        prot = analyze_protonation_ph74(mpo.canonical_smiles)
        cand_fp = _ecfp4_bv(mpo.canonical_smiles)
        tan = (
            float(DataStructs.TanimotoSimilarity(ref_fp, cand_fp))
            if (ref_fp is not None and cand_fp is not None)
            else 1.0
        )

        exact_hop = (
            bool(tr.exact_murcko_scaffold)
            and tr.exact_murcko_scaffold != ref_exact_scaf
            and tan <= float(max_tanimoto_for_hop)
        )
        generic_hop = (
            bool(tr.generic_murcko_scaffold)
            and tr.generic_murcko_scaffold != ref_gen_scaf
            and tan <= float(max_tanimoto_for_hop)
        )
        delta_mpo = float(mpo.cns_mpo_v2 - ref_mpo.cns_mpo_v2)
        delta_sa = float(ref_mpo.sa_score - mpo.sa_score)

        herg_p: float | None = None
        d_herg: float | None = None
        htr2b_p: float | None = None
        d_htr2b: float | None = None
        mdr1_p: float | None = None
        safe_v4: bool | None = None
        safety_term = float(mpo.cns_mpo_v2 / 6.0)

        if v4_preds is not None:
            vp = v4_preds[idx]
            probs = vp.probabilities
            herg_p = float(probs["herg"]) if "herg" in probs else None
            htr2b_p = float(probs["htr2b_avoid"]) if "htr2b_avoid" in probs else None
            mdr1_p = float(probs["mdr1_efflux"]) if "mdr1_efflux" in probs else None
            safe_v4 = bool(vp.admet_v4_pass)
            if herg_p is not None and "herg" in ref_probs:
                d_herg = float(ref_probs["herg"] - herg_p)
            if htr2b_p is not None and "htr2b_avoid" in ref_probs:
                d_htr2b = float(ref_probs["htr2b_avoid"] - htr2b_p)
            herg_safe = 1.0 - (herg_p if herg_p is not None else 0.5)
            htr2b_safe = 1.0 - (htr2b_p if htr2b_p is not None else 0.2)
            mdr1_safe = 1.0 - (mdr1_p if mdr1_p is not None else 0.3)
            safety_term = 0.45 * herg_safe + 0.30 * htr2b_safe + 0.25 * mdr1_safe

        hop_bonus = 1.0 if (exact_hop and generic_hop) else (0.75 if exact_hop else 0.25)
        amine_bonus = 1.0 if (not require_basic_amine or prot.has_basic_amine) else 0.55
        sa_norm = max(0.0, min(1.0, (5.5 - mpo.sa_score) / 4.0))
        mpo_norm = max(0.0, min(1.0, mpo.cns_mpo_v2 / 6.0))
        composite = (
            float(tr.hygiene_gate)
            * hop_bonus
            * amine_bonus
            * (0.35 * mpo_norm + 0.25 * sa_norm + 0.40 * safety_term)
        )

        audits.append(
            ScaffoldHopCandidateAudit(
                smiles=smi,
                canonical_smiles=mpo.canonical_smiles,
                reference_smiles=ref_canon,
                valid=True,
                is_exact_scaffold_hop=exact_hop,
                is_generic_scaffold_hop=generic_hop,
                ecfp4_tanimoto_to_ref=tan,
                cns_mpo_v2=float(mpo.cns_mpo_v2),
                delta_mpo_v2=delta_mpo,
                sa_score=float(mpo.sa_score),
                delta_sa_simplification=delta_sa,
                net_charge_ph74=prot.net_charge,
                has_basic_amine=prot.has_basic_amine,
                triage_passed=bool(tr.triage_pass),
                hygiene_gate=float(tr.hygiene_gate),
                herg_prob=herg_p,
                delta_herg_risk_reduction=d_herg,
                htr2b_prob=htr2b_p,
                delta_htr2b_risk_reduction=d_htr2b,
                mdr1_prob=mdr1_p,
                v4_pharma_safe=safe_v4,
                hop_composite_score=composite,
            )
        )

    audits.sort(key=lambda a: a.hop_composite_score, reverse=True)
    n_valid = len(audits)
    n_hops = sum(1 for a in audits if a.is_exact_scaffold_hop and a.triage_passed)
    n_herg_improved = sum(
        1
        for a in audits
        if a.delta_herg_risk_reduction is not None and a.delta_herg_risk_reduction > 0.15
    )

    return {
        "reference": {
            "canonical_smiles": ref_canon,
            "exact_murcko_scaffold": ref_exact_scaf,
            "generic_murcko_scaffold": ref_gen_scaf,
            "cns_mpo_v2": round(float(ref_mpo.cns_mpo_v2), 3),
            "sa_score": round(float(ref_mpo.sa_score), 3),
            "net_charge_ph74": ref_prot.net_charge,
            "has_basic_amine": ref_prot.has_basic_amine,
            "admet_probabilities": {k: round(float(v), 4) for k, v in ref_probs.items()},
        },
        "summary": {
            "n_input": len(cand_list),
            "n_unique_valid": n_valid,
            "n_clean_scaffold_hops": n_hops,
            "clean_scaffold_hop_rate": round(n_hops / max(n_valid, 1), 4),
            "n_herg_derisked_gt_0p15": n_herg_improved,
            "mean_delta_sa_simplification": round(
                sum(a.delta_sa_simplification for a in audits) / max(n_valid, 1), 3
            ),
            "mean_tanimoto_to_ref": round(
                sum(a.ecfp4_tanimoto_to_ref for a in audits) / max(n_valid, 1), 4
            ),
        },
        "candidates": [a.to_dict() for a in audits],
    }


__all__ = [
    "IBOGAINE_SMILES",
    "PAROXETINE_SMILES",
    "ScaffoldHopCandidateAudit",
    "audit_scaffold_hop_campaign",
]
