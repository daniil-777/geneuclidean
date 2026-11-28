"""2026 Industry Medicinal Chemistry Structural Triage, Reagent Quarantine & IP Pre-Screen.

Provides 100% offline, deterministic RDKit-only medicinal chemistry quality gates aligned
with 2026 pharmaceutical lead-optimization standards:

- **RDKit FilterCatalogs:** ``PAINS`` (A/B/C), ``BRENK``, and ``NIH`` structural alerts.
- **Coupling-Reagent & Workup-Byproduct Quarantine:** Explicitly detects and vetoes
  ``N,N'-dicyclohexylurea`` (DCU — fixes GenEuclidean v3.1 ``DRD3_1``),
  ``N,N'-diisopropylurea`` (DIU), EDCI urea, symmetric dialkyl ureas, ``HOBt``/``HOAt``,
  ``DMAP``, ``Ph3P=O``/``Ph3P``, carbodiimides, acyl/sulfonyl halides, reactive Michael
  acceptors, epoxides/aziridines, peroxides, and poly-nitroaromatics.
- **Lilly-Inspired MedChem Demerit Score (0..300, pass <= 100):** Penalizes excessive
  rotatable bonds, macrocycles, heavy polyhalogenation, non-CNS quaternary ammonium
  cations, ``>= 4`` fused aromatic rings, extreme ``cLogP > 5.0`` or ``MW > 500``,
  flatland low ``Fsp3``, greasy aliphatic chains, unassigned stereocenter burden, and
  ICH S10 extended polyaromatic phototoxicity chromophores.
- **Synthesizability & IP Scaffold-Hop Helpers:** Exact and generic Bemis-Murcko
  scaffolds, stereocenter/spiro/bridgehead burden, and ECFP4 scaffold-hop FTO pre-screen
  with an explicit non-legal Markush FTO disclaimer.
- **Smooth ``hygiene_gate_score`` in ``[0.0, 1.0]``:** Designed directly for GenEuclidean
  v4.0's Multiplicative Safety-Gated Reward (strict ``0.0`` veto on DCU/PAINS/reagents).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    FilterCatalog,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

_DESCRIPTORS: Any = Descriptors
_CRIPPEN: Any = Crippen
_MORGAN_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

FTO_MARKUSH_DISCLAIMER: str = (
    "Computational pre-screen only (Bemis-Murcko scaffold & ECFP4 similarity). "
    "Does NOT evaluate Markush genus patent claims or constitute legal Freedom-to-Operate "
    "(FTO) clearance; formal SureChEMBL/CAS MARPAT/PatBase searching and patent-attorney "
    "claim construction are required prior to synthesis or filing."
)

# Exact known coupling-reagent / workup-byproduct canonical SMILES quarantine table.
_EXACT_REAGENT_SMILES: tuple[tuple[str, str], ...] = (
    ("DCU_dicyclohexylurea", "O=C(NC1CCCCC1)NC1CCCCC1"),
    ("DIU_diisopropylurea", "CC(C)NC(=O)NC(C)C"),
    ("DCC_dicyclohexylcarbodiimide", "C1CCC(N=C=NC2CCCCC2)CC1"),
    ("DIC_diisopropylcarbodiimide", "CC(C)N=C=NC(C)C"),
    ("EDCI_carbodiimide", "CCN=C=NCCCN(C)C"),
    ("EDCI_urea_byproduct", "CCNC(=O)NCCCN(C)C"),
    ("HOBt_coupling_additive", "On1nnc2ccccc21"),
    ("HOAt_coupling_additive", "On1nnc2cccnc21"),
    ("DMAP_acylation_catalyst", "CN(C)c1ccncc1"),
    ("Ph3PO_wittig_byproduct", "O=P(c1ccccc1)(c1ccccc1)c1ccccc1"),
    ("Ph3P_triphenylphosphine", "c1ccc(P(c2ccccc2)c2ccccc2)cc1"),
    ("DEAD_mitsunobu_reagent", "CCOC(=O)N=NC(=O)OCC"),
    ("DIAD_mitsunobu_reagent", "CC(C)OC(=O)N=NC(=O)OC(C)C"),
    ("DIAD_hydrazine_byproduct", "CC(C)OC(=O)NNC(=O)OC(C)C"),
    ("DEAD_hydrazine_byproduct", "CCOC(=O)NNC(=O)OCC"),
    ("NHS_hydroxysuccinimide", "O=C1CCC(=O)N1O"),
)

# Hard reactive warhead / coupling-artifact SMARTS that trigger reagent/warhead quarantine.
_QUARANTINE_SMARTS: tuple[tuple[str, str], ...] = (
    ("carbodiimide", "[NX2]=[CX2]=[NX2]"),
    ("acyl_halide", "[CX3](=O)[F,Cl,Br,I]"),
    ("sulfonyl_halide", "[SX4](=O)(=O)[F,Cl,Br,I]"),
    ("isocyanate_or_isothiocyanate", "[NX2]=[CX2]=[O,S]"),
    ("anhydride", "[CX3](=O)[OX2][CX3](=O)"),
    ("peroxide", "[OX2][OX2]"),
    ("epoxide_or_aziridine", "[CX4]1[OX2,NX3][CX4]1"),
    ("triphenylphosphine_oxide_motif", "[PX4](=O)(c1ccccc1)(c1ccccc1)c1ccccc1"),
    ("triphenylphosphine_motif", "[PX3](c1ccccc1)(c1ccccc1)c1ccccc1"),
    ("hobt_hoat_core", "[OX2H1][n]1nnc2ccccc21"),
    ("hoat_aza_core", "[OX2H1][n]1nnc2cccnc21"),
    # Unhindered terminal or disubstituted Michael acceptors (acrylates, acrylamides,
    # vinyl ketones/sulfones/nitriles, maleimides)
    (
        "michael_acceptor_terminal",
        "[CX3;H2]=[CX3;H1][$([CX3]=O),$([CX2]#N),$([SX4](=O)=O)]",
    ),
    (
        "michael_acceptor_maleimide",
        "O=C1C=CC(=O)N1",
    ),
    (
        "michael_acceptor_vinyl_ketone_aldehyde",
        "[CX3;H2,H1]=[CX3;H1][CX3](=O)[#1,#6]",
    ),
    # Dialkyl/dicycloalkyl N,N'-disubstituted aliphatic urea core (checked alongside
    # symmetry/no-aromatic-ring guard in _check_reagent_quarantine to spare drugs like
    # cariprazine and pimavanserin while quarantining DCU/DIU/symmetric dialkyl ureas!)
    ("aliphatic_dialkyl_urea", "[CX4][NX3;H1]C(=O)[NX3;H1][CX4]"),
)

# Lilly-inspired demerit SMARTS (name, SMARTS, demerit_points, is_hard_reject)
_LILLY_SMARTS_RULES: tuple[tuple[str, str, int, bool], ...] = (
    ("quaternary_ammonium_non_cns", "[NX4+;!$([N+](=O)[O-]);!$([N+]=*);!$([N+]#*)]", 110, False),
    ("aldehyde_reactive", "[CX3H1](=O)[#6]", 110, True),
    ("disulfide_unstable", "[SX2][SX2]", 80, False),
    ("hydrazine_reactive", "[NX3;!R][NX3;!R]", 85, False),
    ("nitroso_mutagen", "[NX2]=O", 110, True),
    ("azide_reactive", "[NX2]=[NX2+]=[NX1-]", 110, True),
    ("thiocarbonyl_thione", "[#6]=[SX1;!$(S=C(N)N)]", 55, False),
    (
        "long_aliphatic_chain_c7",
        "[CH2;!R][CH2;!R][CH2;!R][CH2;!R][CH2;!R][CH2;!R][CH2;!R]",
        65,
        False,
    ),
    ("azo_chromophore", "[a][NX2]=[NX2][a]", 75, False),
    ("polyene_conjugated_4", "C=CC=CC=CC=C", 90, False),
    ("nitroaromatic", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][a]", 60, False),
)

# ICH S10 phototoxicity structural chromophore alerts (advisory/demerit for photosafety)
_ICH_S10_PHOTOTOX_SMARTS: tuple[tuple[str, str], ...] = (
    ("furocoumarin_psoralen", "O=c1ccc2cc3ccoc3cc2o1"),
    ("phenothiazine_core", "c1ccc2c(c1)Nc1ccccc1S2"),
    ("acridine_core", "c1ccc2nc3ccccc3cc2c1"),
    ("anthracene_or_anthraquinone", "c1ccc2cc3ccccc3cc2c1"),
    ("azo_aromatic_dye", "[a][NX2]=[NX2][a]"),
)


@lru_cache(maxsize=1)
def _get_filter_catalogs() -> tuple[Any, Any, Any]:
    """Build and cache RDKit PAINS (A/B/C), BRENK, and NIH FilterCatalog instances."""
    pains_p = FilterCatalog.FilterCatalogParams()
    pains_p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)

    brenk_p = FilterCatalog.FilterCatalogParams()
    brenk_p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)

    nih_p = FilterCatalog.FilterCatalogParams()
    nih_p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)

    return (
        FilterCatalog.FilterCatalog(pains_p),
        FilterCatalog.FilterCatalog(brenk_p),
        FilterCatalog.FilterCatalog(nih_p),
    )


@lru_cache(maxsize=1)
def _compiled_exact_reagents() -> dict[str, str]:
    """Map canonical SMILES -> reagent quarantine label."""
    out: dict[str, str] = {}
    for label, smi in _EXACT_REAGENT_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            can = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
            out[can] = label
    return out


@lru_cache(maxsize=1)
def _compiled_quarantine_smarts() -> tuple[tuple[str, Chem.Mol], ...]:
    compiled: list[tuple[str, Chem.Mol]] = []
    for name, smarts in _QUARANTINE_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled.append((name, pat))
    return tuple(compiled)


@lru_cache(maxsize=1)
def _compiled_lilly_smarts() -> tuple[tuple[str, Chem.Mol, int, bool], ...]:
    compiled: list[tuple[str, Chem.Mol, int, bool]] = []
    for name, smarts, pts, hard in _LILLY_SMARTS_RULES:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled.append((name, pat, pts, hard))
    return tuple(compiled)


@lru_cache(maxsize=1)
def _compiled_phototox_smarts() -> tuple[tuple[str, Chem.Mol], ...]:
    compiled: list[tuple[str, Chem.Mol]] = []
    for name, smarts in _ICH_S10_PHOTOTOX_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled.append((name, pat))
    return tuple(compiled)


@lru_cache(maxsize=1)
def _compiled_nitro_aromatic() -> Chem.Mol:
    pat = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])][a]")
    assert pat is not None
    return pat


def _resolve_mol(mol_or_smiles: str | Chem.Mol) -> tuple[str, Chem.Mol | None]:
    """Parse a SMILES string or return an existing RDKit Mol alongside its raw text."""
    if isinstance(mol_or_smiles, str):
        raw = mol_or_smiles
        stripped = raw.strip()
        return raw, (Chem.MolFromSmiles(stripped) if stripped else None)
    if mol_or_smiles is None:
        return "", None
    raw = str(Chem.MolToSmiles(mol_or_smiles, canonical=True, isomericSmiles=True))
    return raw, mol_or_smiles


def _catalog_matches(catalog: Any, mol: Chem.Mol) -> tuple[str, ...]:
    """Return sorted unique descriptions of all FilterCatalog alerts matching ``mol``."""
    entries = catalog.GetMatches(mol)
    if not entries:
        return ()
    return tuple(sorted({str(e.GetDescription()) for e in entries}))


def _max_fused_aromatic_ring_system(mol: Chem.Mol) -> int:
    """Return the size (ring count) of the largest fused aromatic ring system in ``mol``."""
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    aro_rings: list[set[int]] = []
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aro_rings.append(set(ring))
    if not aro_rings:
        return 0

    n = len(aro_rings)
    visited = [False] * n
    best = 0
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp_size = 0
        while stack:
            cur = stack.pop()
            comp_size += 1
            for j in range(n):
                if not visited[j] and len(aro_rings[cur] & aro_rings[j]) >= 2:
                    visited[j] = True
                    stack.append(j)
        if comp_size > best:
            best = comp_size
    return best


def _is_symmetric_dialkyl_urea(mol: Chem.Mol) -> bool:
    """Detect trivial symmetric N,N'-dialkyl/dicycloalkyl ureas (e.g., DCU, DIU).

    Requires zero aromatic rings and a central ``[NH]-C(=O)-[NH]`` where removing the
    urea carbonyl leaves two topologically equivalent (or identical-formula) aliphatic
    substituents — catching DCU/DIU/dicyclopentylurea while sparing asymmetric CNS drugs
    such as cariprazine or pimavanserin.
    """
    if int(rdMolDescriptors.CalcNumAromaticRings(mol)) > 0:
        return False
    urea_pat = Chem.MolFromSmarts("[#6;X4][NX3;H1][CX3](=O)[NX3;H1][#6;X4]")
    if urea_pat is None or not mol.HasSubstructMatch(urea_pat):
        return False
    # Check symmetry via canonical atom ranks on the two urea NH nitrogens
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    for match in mol.GetSubstructMatches(urea_pat):
        n1_idx, n2_idx = match[1], match[4]
        if ranks[n1_idx] == ranks[n2_idx]:
            return True
    return False


# ---------------------------------------------------------------------------
# Synthesizability, Stereocenter Burden & Bemis-Murcko Scaffold-Hop Helpers
# ---------------------------------------------------------------------------


def stereocenter_burden(mol_or_smiles: str | Chem.Mol) -> dict[str, int]:
    """Quantify chiral center count, undefined stereocenters, spiro, and bridgehead atoms."""
    _, mol = _resolve_mol(mol_or_smiles)
    if mol is None:
        return {
            "total_chiral_centers": 0,
            "defined_chiral_centers": 0,
            "undefined_chiral_centers": 0,
            "spiro_atoms": 0,
            "bridgehead_atoms": 0,
        }
    centers = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    total = len(centers)
    undefined = sum(1 for _, cip in centers if cip == "?")
    defined = total - undefined
    return {
        "total_chiral_centers": int(total),
        "defined_chiral_centers": int(defined),
        "undefined_chiral_centers": int(undefined),
        "spiro_atoms": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "bridgehead_atoms": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
    }


def bemis_murcko_scaffold(
    mol_or_smiles: str | Chem.Mol,
    *,
    generic: bool = False,
    include_chirality: bool = False,
) -> str:
    """Extract canonical exact or genericized Bemis-Murcko scaffold SMILES."""
    _, mol = _resolve_mol(mol_or_smiles)
    if mol is None:
        return ""
    try:
        scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf_mol is None or scaf_mol.GetNumAtoms() == 0:
            return ""
        if generic:
            scaf_mol = MurckoScaffold.MakeScaffoldGeneric(scaf_mol)
        return str(Chem.MolToSmiles(scaf_mol, canonical=True, isomericSmiles=include_chirality))
    except Exception:
        return ""


@dataclass(frozen=True)
class ScaffoldHopReport:
    """IP novelty and Bemis-Murcko scaffold-hop pre-screen report (non-legal)."""

    exact_murcko_scaffold: str
    generic_murcko_scaffold: str
    is_novel_exact_scaffold: bool
    is_novel_generic_scaffold: bool
    is_novel_scaffold: bool
    max_tanimoto: float
    fto_pre_screen_flag: Literal["GREEN", "AMBER", "RED"]
    fto_disclaimer: str = FTO_MARKUSH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaffold-hop report to a JSON-compatible dictionary."""
        return {
            "exact_murcko_scaffold": self.exact_murcko_scaffold,
            "generic_murcko_scaffold": self.generic_murcko_scaffold,
            "is_novel_exact_scaffold": self.is_novel_exact_scaffold,
            "is_novel_generic_scaffold": self.is_novel_generic_scaffold,
            "is_novel_scaffold": self.is_novel_scaffold,
            "max_tanimoto": self.max_tanimoto,
            "fto_pre_screen_flag": self.fto_pre_screen_flag,
            "fto_disclaimer": self.fto_disclaimer,
        }


def compute_scaffold_hop_metrics(
    mol_or_smiles: str | Chem.Mol,
    reference_fps: Sequence[DataStructs.ExplicitBitVect] | NDArray[np.floating[Any]] | None = None,
    *,
    reference_smiles: Sequence[str] | None = None,
    reference_exact_scaffolds: set[str] | frozenset[str] | None = None,
    reference_generic_scaffolds: set[str] | frozenset[str] | None = None,
    max_tanimoto_green: float = 0.50,
    max_tanimoto_amber: float = 0.70,
) -> ScaffoldHopReport:
    """Compute Bemis-Murcko scaffold novelty, nearest-neighbor ECFP4 Tanimoto, and FTO flag.

    Args:
        mol_or_smiles: Candidate SMILES string or RDKit ``Mol``.
        reference_fps: Optional sequence of RDKit ``ExplicitBitVect`` or 2D NumPy bit
            matrix ``(M, D)`` of known clinical/patented reference ligands.
        reference_smiles: Optional sequence of reference SMILES (used to derive reference
            scaffolds and fingerprints when ``reference_fps``/scaffold sets are omitted).
        reference_exact_scaffolds: Optional precomputed set of exact Bemis-Murcko SMILES.
        reference_generic_scaffolds: Optional precomputed set of generic Bemis-Murcko SMILES.
        max_tanimoto_green: Tanimoto ceiling for ``GREEN`` scaffold-hop pre-screen flag.
        max_tanimoto_amber: Tanimoto ceiling for ``AMBER`` pre-screen flag (above -> ``RED``).

    Returns:
        ``ScaffoldHopReport`` including an explicit non-legal Markush FTO disclaimer.
    """
    _, mol = _resolve_mol(mol_or_smiles)
    if mol is None:
        return ScaffoldHopReport(
            exact_murcko_scaffold="",
            generic_murcko_scaffold="",
            is_novel_exact_scaffold=False,
            is_novel_generic_scaffold=False,
            is_novel_scaffold=False,
            max_tanimoto=1.0,
            fto_pre_screen_flag="RED",
        )

    exact_scaf = bemis_murcko_scaffold(mol, generic=False)
    generic_scaf = bemis_murcko_scaffold(mol, generic=True)

    ref_exact: set[str] = set(reference_exact_scaffolds or ())
    ref_generic: set[str] = set(reference_generic_scaffolds or ())
    derived_fps: list[DataStructs.ExplicitBitVect] = []

    if reference_smiles is not None:
        for rsmi in reference_smiles:
            rmol = (
                Chem.MolFromSmiles(rsmi.strip()) if isinstance(rsmi, str) and rsmi.strip() else None
            )
            if rmol is None:
                continue
            if reference_exact_scaffolds is None:
                ref_exact.add(bemis_murcko_scaffold(rmol, generic=False))
            if reference_generic_scaffolds is None:
                ref_generic.add(bemis_murcko_scaffold(rmol, generic=True))
            if reference_fps is None:
                derived_fps.append(_MORGAN_2048.GetFingerprint(rmol))

    max_tan = 0.0
    if reference_fps is not None:
        if isinstance(reference_fps, np.ndarray):
            if reference_fps.size > 0:
                ref_mat = np.asarray(reference_fps, dtype=np.float32)
                n_bits = int(ref_mat.shape[1])
                gen = (
                    _MORGAN_2048
                    if n_bits == 2048
                    else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
                )
                q_arr = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
                inter = ref_mat @ q_arr
                union = ref_mat.sum(axis=1) + float(q_arr.sum()) - inter
                sims = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
                max_tan = float(np.max(sims))
        elif len(reference_fps) > 0:
            n_bits = int(reference_fps[0].GetNumBits())
            gen = (
                _MORGAN_2048
                if n_bits == 2048
                else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
            )
            q_fp = gen.GetFingerprint(mol)
            sims_list = DataStructs.BulkTanimotoSimilarity(q_fp, list(reference_fps))
            max_tan = float(max(sims_list)) if sims_list else 0.0
    elif derived_fps:
        q_fp = _MORGAN_2048.GetFingerprint(mol)
        sims_list = DataStructs.BulkTanimotoSimilarity(q_fp, derived_fps)
        max_tan = float(max(sims_list)) if sims_list else 0.0

    max_tan = float(np.clip(max_tan, 0.0, 1.0))
    is_novel_exact = bool(exact_scaf and exact_scaf not in ref_exact)
    is_novel_generic = bool(generic_scaf and generic_scaf not in ref_generic)
    # Acyclic molecules have no ring scaffold -> not a ring scaffold hop
    is_novel = bool(is_novel_generic)

    if max_tan > max_tanimoto_amber or (not exact_scaf):
        flag: Literal["GREEN", "AMBER", "RED"] = "RED"
    elif is_novel_generic and max_tan <= max_tanimoto_green:
        flag = "GREEN"
    else:
        flag = "AMBER"

    return ScaffoldHopReport(
        exact_murcko_scaffold=exact_scaf,
        generic_murcko_scaffold=generic_scaf,
        is_novel_exact_scaffold=is_novel_exact,
        is_novel_generic_scaffold=is_novel_generic,
        is_novel_scaffold=is_novel,
        max_tanimoto=max_tan,
        fto_pre_screen_flag=flag,
    )


def is_scaffold_hop(
    mol_or_smiles: str | Chem.Mol,
    reference_scaffolds: set[str] | frozenset[str] | Sequence[str],
    *,
    reference_fps: Sequence[DataStructs.ExplicitBitVect] | NDArray[np.floating[Any]] | None = None,
    max_tanimoto: float = 0.55,
) -> bool:
    """Return ``True`` if candidate hops to a novel generic Murcko core below ``max_tanimoto``."""
    ref_set = set(reference_scaffolds)
    report = compute_scaffold_hop_metrics(
        mol_or_smiles,
        reference_fps=reference_fps,
        reference_exact_scaffolds=ref_set,
        reference_generic_scaffolds=ref_set,
        max_tanimoto_green=max_tanimoto,
    )
    return bool(report.is_novel_generic_scaffold and report.max_tanimoto <= max_tanimoto)


# ---------------------------------------------------------------------------
# Complete MedChem Structural Triage & Smooth Hygiene Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MedChemTriageResult:
    """Complete 2026 industrial medicinal-chemistry structural triage profile."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    pains_free: bool
    pains_alerts: tuple[str, ...]
    brenk_free: bool
    brenk_alerts: tuple[str, ...]
    nih_free: bool
    nih_alerts: tuple[str, ...]
    reagent_quarantine_free: bool
    reagent_alerts: tuple[str, ...]
    phototox_alert: bool
    phototox_alerts: tuple[str, ...]
    lilly_demerit_score: int
    lilly_demerit_reasons: tuple[str, ...]
    lilly_pass: bool
    n_chiral_centers: int
    n_undefined_chiral_centers: int
    fsp3: float
    rotatable_bonds: int
    max_ring_size: int
    fused_aromatic_rings: int
    exact_murcko_scaffold: str
    generic_murcko_scaffold: str
    hygiene_gate: float
    triage_pass: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize triage profile to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "pains_free": self.pains_free,
            "pains_alerts": list(self.pains_alerts),
            "brenk_free": self.brenk_free,
            "brenk_alerts": list(self.brenk_alerts),
            "nih_free": self.nih_free,
            "nih_alerts": list(self.nih_alerts),
            "reagent_quarantine_free": self.reagent_quarantine_free,
            "reagent_alerts": list(self.reagent_alerts),
            "phototox_alert": self.phototox_alert,
            "phototox_alerts": list(self.phototox_alerts),
            "lilly_demerit_score": self.lilly_demerit_score,
            "lilly_demerit_reasons": list(self.lilly_demerit_reasons),
            "lilly_pass": self.lilly_pass,
            "n_chiral_centers": self.n_chiral_centers,
            "n_undefined_chiral_centers": self.n_undefined_chiral_centers,
            "fsp3": self.fsp3,
            "rotatable_bonds": self.rotatable_bonds,
            "max_ring_size": self.max_ring_size,
            "fused_aromatic_rings": self.fused_aromatic_rings,
            "exact_murcko_scaffold": self.exact_murcko_scaffold,
            "generic_murcko_scaffold": self.generic_murcko_scaffold,
            "hygiene_gate": self.hygiene_gate,
            "triage_pass": self.triage_pass,
        }


def _invalid_triage(smiles: str) -> MedChemTriageResult:
    return MedChemTriageResult(
        smiles=smiles,
        canonical_smiles=None,
        valid=False,
        pains_free=False,
        pains_alerts=("invalid_smiles",),
        brenk_free=False,
        brenk_alerts=(),
        nih_free=False,
        nih_alerts=(),
        reagent_quarantine_free=False,
        reagent_alerts=("invalid_smiles",),
        phototox_alert=False,
        phototox_alerts=(),
        lilly_demerit_score=300,
        lilly_demerit_reasons=("invalid_smiles",),
        lilly_pass=False,
        n_chiral_centers=0,
        n_undefined_chiral_centers=0,
        fsp3=0.0,
        rotatable_bonds=0,
        max_ring_size=0,
        fused_aromatic_rings=0,
        exact_murcko_scaffold="",
        generic_murcko_scaffold="",
        hygiene_gate=0.0,
        triage_pass=False,
    )


def triage_molecule(mol_or_smiles: str | Chem.Mol) -> MedChemTriageResult:
    """Run full 2026 industrial MedChem structural triage & reagent quarantine on a ligand.

    Returns a frozen ``MedChemTriageResult`` with RDKit ``PAINS``/``BRENK``/``NIH`` alert
    lists, coupling-reagent quarantine flags (explicitly blocking DCU, DIU, HOBt, DMAP,
    Ph3P=O, carbodiimides, Michael acceptors, acyl halides, and poly-nitroaromatics),
    Lilly-inspired demerit score ``0..300`` (pass ``<= 100``), ICH S10 phototoxicity flag,
    stereocenter burden, exact/generic Bemis-Murcko scaffolds, and smooth ``hygiene_gate``
    in ``[0.0, 1.0]``.
    """
    raw_smi, mol = _resolve_mol(mol_or_smiles)
    if mol is None:
        return _invalid_triage(raw_smi)

    canonical_iso = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    canonical_flat = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))

    # 1. RDKit FilterCatalogs (PAINS A/B/C, BRENK, NIH)
    pains_cat, brenk_cat, nih_cat = _get_filter_catalogs()
    pains_alerts = _catalog_matches(pains_cat, mol)
    brenk_alerts = _catalog_matches(brenk_cat, mol)
    nih_alerts = _catalog_matches(nih_cat, mol)
    pains_free = len(pains_alerts) == 0
    brenk_free = len(brenk_alerts) == 0
    nih_free = len(nih_alerts) == 0

    # 2. Coupling-Reagent, Workup-Byproduct & Hard Reactive-Warhead Quarantine
    reagent_hits: list[str] = []
    exact_map = _compiled_exact_reagents()
    if canonical_flat in exact_map:
        reagent_hits.append(f"exact_reagent:{exact_map[canonical_flat]}")

    if _is_symmetric_dialkyl_urea(mol):
        reagent_hits.append("symmetric_dialkyl_urea_byproduct")

    for name, pat in _compiled_quarantine_smarts():
        if name == "aliphatic_dialkyl_urea":
            # Only quarantine aliphatic dialkyl ureas when molecule has zero aromatic rings
            # (catches DCU/DIU/EDCI-urea analogs while sparing asymmetric aromatic drug ureas)
            if int(rdMolDescriptors.CalcNumAromaticRings(mol)) == 0 and mol.HasSubstructMatch(pat):
                reagent_hits.append("non_aromatic_dialkyl_urea_byproduct")
            continue
        if mol.HasSubstructMatch(pat):
            reagent_hits.append(name)

    nitro_matches = mol.GetSubstructMatches(_compiled_nitro_aromatic())
    if len(nitro_matches) >= 2:
        reagent_hits.append(f"poly_nitroaromatic(n={len(nitro_matches)})")

    reagent_alerts = tuple(sorted(set(reagent_hits)))
    reagent_free = len(reagent_alerts) == 0

    # 3. Physicochemical & Topological Descriptors for Lilly Demerits & ICH S10 Phototox
    mw = float(_DESCRIPTORS.MolWt(mol))
    clogp = float(_CRIPPEN.MolLogP(mol))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    n_aro_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    max_ring_size = max(ring_sizes) if ring_sizes else 0
    fused_aro = _max_fused_aromatic_ring_system(mol)
    stereo = stereocenter_burden(mol)
    n_chiral = stereo["total_chiral_centers"]
    n_undef_chiral = stereo["undefined_chiral_centers"]

    n_heavy_halogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (17, 35, 53))
    n_total_halogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))

    # 4. ICH S10 Phototoxicity Chromophores
    phototox_hits: list[str] = []
    for name, pat in _compiled_phototox_smarts():
        if mol.HasSubstructMatch(pat):
            phototox_hits.append(name)
    if fused_aro >= 3:
        phototox_hits.append(f"extended_fused_polyaromatic(rings={fused_aro})")
    phototox_alerts = tuple(sorted(set(phototox_hits)))
    phototox_flag = len(phototox_alerts) > 0

    # 5. Lilly-Inspired MedChem Demerit Score (0..300, pass <= 100)
    demerits = 0
    reasons: list[str] = []
    # These representations parse correctly, but a finished organic candidate
    # needs one connected structure with all attachment points resolved.
    if len(Chem.GetMolFrags(mol)) != 1:
        reasons.append("multiple_components")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        reasons.append("unresolved_attachment")
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        reasons.append("no_organic_parent")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        reasons.append("radical")
    hard_reject = bool(reasons)
    if hard_reject:
        demerits += 150

    if not reagent_free:
        demerits += 150
        reasons.append(f"reagent_or_warhead_quarantine({','.join(reagent_alerts[:3])})")
        hard_reject = True

    if not pains_free:
        demerits += 110
        reasons.append(f"pains_alert(n={len(pains_alerts)})")
        hard_reject = True

    for name, pat, pts, is_hard in _compiled_lilly_smarts():
        if mol.HasSubstructMatch(pat):
            demerits += pts
            reasons.append(f"{name}(+{pts})")
            if is_hard:
                hard_reject = True

    if rot_bonds > 8:
        pts = min(75, 15 * (rot_bonds - 8))
        demerits += pts
        reasons.append(f"excessive_rotatable_bonds={rot_bonds}(+{pts})")

    if max_ring_size >= 12:
        demerits += 65
        reasons.append(f"macrocycle_ring_size={max_ring_size}(+65)")
    elif max_ring_size >= 9:
        demerits += 35
        reasons.append(f"medium_ring_size={max_ring_size}(+35)")

    if n_heavy_halogens >= 3:
        pts = min(90, 35 * (n_heavy_halogens - 2))
        demerits += pts
        reasons.append(f"poly_heavy_halogenation={n_heavy_halogens}(+{pts})")
    if n_total_halogens >= 7:
        pts = min(75, 25 * (n_total_halogens - 6))
        demerits += pts
        reasons.append(f"excessive_polyhalogenation={n_total_halogens}(+{pts})")

    if fused_aro >= 4:
        demerits += 120
        reasons.append(f"fused_aromatic_rings={fused_aro}(+120)")
        hard_reject = True
    elif fused_aro == 3:
        demerits += 45
        reasons.append("three_fused_aromatic_rings(+45)")

    if clogp > 5.0:
        pts = min(90, int(math.ceil((clogp - 5.0) * 35.0)))
        demerits += pts
        reasons.append(f"extreme_clogp={clogp:.2f}(+{pts})")

    if mw > 500.0:
        pts = min(90, int(math.ceil((mw - 500.0) * 0.6)))
        demerits += pts
        reasons.append(f"high_mw={mw:.1f}(+{pts})")

    if fsp3 < 0.10 and n_aro_rings >= 3:
        demerits += 35
        reasons.append(f"flatland_low_fsp3={fsp3:.2f}(+35)")
    elif fsp3 < 0.15 and n_aro_rings >= 4:
        demerits += 25
        reasons.append(f"flatland_polyaromatic_fsp3={fsp3:.2f}(+25)")

    if n_undef_chiral >= 3:
        pts = min(60, 20 * (n_undef_chiral - 2))
        demerits += pts
        reasons.append(f"undefined_stereocenters={n_undef_chiral}(+{pts})")
    if n_chiral >= 5:
        demerits += 20
        reasons.append(f"high_stereocenter_burden={n_chiral}(+20)")

    if phototox_flag and fused_aro < 3:
        demerits += 30
        reasons.append(f"ich_s10_phototox({','.join(phototox_alerts[:2])})(+30)")

    demerit_score = int(min(300, max(0, demerits)))
    lilly_pass = bool((not hard_reject) and demerit_score <= 100)

    # 6. Smooth Hygiene Gate in [0.0, 1.0] for Multiplicative Safety-Gated Reward
    if (not reagent_free) or (not pains_free) or hard_reject:
        hygiene_gate = 0.0
    else:
        # Smooth logistic decay above demerit=40 (1.0 at <=40, ~0.50 at 110, ~0.08 at 180)
        if demerit_score <= 40:
            base_gate = 1.0
        else:
            # Smooth continuous transition anchored at 1.0 for demerit=40
            raw_sig = 1.0 / (1.0 + math.exp((demerit_score - 105.0) / 22.0))
            anchor = 1.0 / (1.0 + math.exp((40.0 - 105.0) / 22.0))
            base_gate = min(1.0, raw_sig / anchor)
        if not brenk_free:
            base_gate *= 0.70
        if not nih_free:
            base_gate *= 0.80
        hygiene_gate = float(np.clip(base_gate, 0.0, 1.0))

    exact_scaf = bemis_murcko_scaffold(mol, generic=False)
    generic_scaf = bemis_murcko_scaffold(mol, generic=True)
    triage_ok = bool(pains_free and brenk_free and nih_free and reagent_free and lilly_pass)

    return MedChemTriageResult(
        smiles=raw_smi,
        canonical_smiles=canonical_iso,
        valid=True,
        pains_free=pains_free,
        pains_alerts=pains_alerts,
        brenk_free=brenk_free,
        brenk_alerts=brenk_alerts,
        nih_free=nih_free,
        nih_alerts=nih_alerts,
        reagent_quarantine_free=reagent_free,
        reagent_alerts=reagent_alerts,
        phototox_alert=phototox_flag,
        phototox_alerts=phototox_alerts,
        lilly_demerit_score=demerit_score,
        lilly_demerit_reasons=tuple(reasons),
        lilly_pass=lilly_pass,
        n_chiral_centers=n_chiral,
        n_undefined_chiral_centers=n_undef_chiral,
        fsp3=fsp3,
        rotatable_bonds=rot_bonds,
        max_ring_size=max_ring_size,
        fused_aromatic_rings=fused_aro,
        exact_murcko_scaffold=exact_scaf,
        generic_murcko_scaffold=generic_scaf,
        hygiene_gate=hygiene_gate,
        triage_pass=triage_ok,
    )


def triage_batch(items: Sequence[str | Chem.Mol]) -> list[MedChemTriageResult]:
    """Run ``triage_molecule`` across a batch of SMILES strings or RDKit ``Mol`` objects."""
    return [triage_molecule(item) for item in items]


def hygiene_gate_score(mol_or_smiles: str | Chem.Mol) -> float:
    """Return smooth MedChem hygiene gate in ``[0.0, 1.0]`` (``0.0`` for DCU/PAINS/reagents)."""
    return triage_molecule(mol_or_smiles).hygiene_gate


__all__ = [
    "FTO_MARKUSH_DISCLAIMER",
    "MedChemTriageResult",
    "ScaffoldHopReport",
    "bemis_murcko_scaffold",
    "compute_scaffold_hop_metrics",
    "hygiene_gate_score",
    "is_scaffold_hop",
    "stereocenter_burden",
    "triage_batch",
    "triage_molecule",
]
