"""Exact Pfizer CNS MPO v1 & v2 desirability scoring and CNS BBB medicinal chemistry filters.

Implements the industry-standard Pfizer CNS Multiparameter Optimization (CNS MPO) algorithm:

- **CNS MPO v1** (Wager et al., *ACS Chem. Neurosci.* 2010, 1, 435–449):
  Sums six piecewise-linear desirability functions ``T_0`` in ``[0, 1]`` for ``ClogP``,
  ``ClogD_7.4``, ``MW``, ``TPSA`` (hump function), ``HBD``, and ``most_basic_pKa`` to yield
  a composite score in ``[0.0, 6.0]``.
- **CNS MPO v2** (Wager et al., *ACS Chem. Neurosci.* 2016, 7, 767–775):
  Replaces the collinear ``ClogD_7.4`` term with ``2 * T_0(HBD)`` to emphasize brain
  unbound exposure and P-gp efflux evasion while preserving the ``[0.0, 6.0]`` scale.

Also provides transparent SMARTS-based physicochemical ``pKa`` estimation, exact
Henderson-Hasselbalch ``ClogD_7.4`` shifting at pH 7.4, Clark (1999) and Kelder (1999)
CNS BBB heuristic flags, and integrated QED / SA / PAINS / Brenk / Lipinski metrics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors

from geneuclidean.evaluation.metrics import (
    get_sa_score,
    is_brenk_free,
    is_pains_free,
    lipinski_violations,
    passes_lipinski,
    sa_to_normalized,
)

_DESCRIPTORS: Any = Descriptors
_CRIPPEN: Any = Crippen

PHYSIOLOGICAL_PH: float = 7.4

# Ordered (name, SMARTS, estimated conjugate-acid pKa) table for basic centers.
# Rules are evaluated carefully to distinguish strongly basic alkylamines from inductively
# dampened benzylic/beta-heteroatom/piperazine amines, acylguanidines, 2-aminoimidazolines,
# heteroaromatics, anilines, and permanently charged quaternary ammonium cations.
_BASIC_PKA_SMARTS: tuple[tuple[str, str, float], ...] = (
    # Permanent quaternary ammonium cation (no neutral form at pH 7.4 -> poor passive BBB)
    ("quaternary_ammonium", "[NX4+;!$([N+](=O)[O-]);!$([N+]=*);!$([N+]#*)]", 14.0),
    # Strongly basic non-acyl, non-sulfonyl aliphatic guanidines / biguanides (metformin ~12.4)
    (
        "alkyl_guanidine",
        (
            "[NX3,NX2;!$(NC=[O,S]);!$(NS(=O)=O);!$(N[a]);!$(NC1=NCCN1)]"
            "C(=[NX2;!$(NC=[O,S]);!$(NS(=O)=O);!$(N1CCN=C1)])"
            "([NX3;!$(NC=[O,S]);!$(NS(=O)=O);!$(N[a]);!$(NC1=NCCN1)])"
        ),
        12.0,
    ),
    # Strongly basic non-acyl aliphatic amidines (excludes beta-heteroatom / fluoroalkyl-dampened
    # cyclic BACE1 warheads such as amino-oxazines and aminothiazines)
    (
        "alkyl_amidine",
        (
            "[NX3,NX2;!$(NC=[O,S]);!$(NS(=O)=O);!$(N[a]);!$(NC1=NCCN1)]"
            "C(=[NX2;!$(NC=[O,S]);!$(NS(=O)=O)])"
            "[#6;!$(C=[O,S]);!$(c);!$(C(=N)N);!$([CX4]([O,S,N,F,Cl]));!$([CX4][CX4]([F])([F]))]"
        ),
        11.5,
    ),
    # Inductively dampened cyclic sulfonyl-guanidine / iminothiadiazine dioxide
    # (e.g., BACE1 inhibitor verubecestat clinical pKa ~6.2)
    (
        "cyclic_sulfonyl_guanidine",
        "[NX3;H2,H1]C(=[NX2;R])[NX3;R]S(=O)(=O)",
        6.2,
    ),
    # Inductively dampened cyclic amino-oxazine / aminothiazine / alpha-CF3 cyclic amidine
    # (e.g., BACE1 inhibitors umibecestat & elenbecestat clinical pKa ~7.1)
    (
        "cyclic_ewg_amidine",
        "[NX3;H2,H1]C(=[NX2;R])[$([CX4;R]([O,S,N,F,Cl])),$([CX4;R][CX4]([F])([F])),$([SX2,OX2;R])]",
        7.1,
    ),
    # Acylguanidines (e.g., guanfacine clinical pKa ~8.1 — resonance-dampened by carbonyl)
    ("acyl_guanidine", "[NX3,NX2]C(=[NX2])[NX3,NX2]C(=O)", 8.1),
    ("acyl_guanidine_alt", "C(=O)[NX2]=C([NX3])[NX3]", 8.1),
    # 2-(Arylamino)-2-imidazolines (clonidine / brimonidine clinical pKa ~8.1)
    ("arylamino_imidazoline", "[a][NX3;H1]C1=NCCN1", 8.1),
    # General aliphatic primary amine (must be bonded to sp3 C; excludes guanidines/amides/anilines)
    (
        "aliphatic_primary_amine",
        "[NX3;H2;!$(N[CX3]);!$(NS(=O)=O);!$(N[a]);!$(N[CX4]C(=O)O);!$(N[CX4][c])][CX4]",
        9.8,
    ),
    # Alpha-amino ester primary amine (e.g., valbenazine L-valine ester, clinical pKa ~8.0)
    ("alpha_amino_ester", "[NX3;H2;!$(N[CX3])][CX4]C(=O)O", 8.0),
    # Benzylic primary amine
    ("benzyl_primary_amine", "[NX3;H2;!$(N[CX3])][CX4][c]", 9.0),
    # General aliphatic secondary amine (e.g., sertraline ~9.5, paroxetine ~9.8, atenolol ~9.6)
    (
        "aliphatic_secondary_amine",
        (
            "[NX3;H1;!$(N[CX3]);!$(NS(=O)=O);!$(N[a]);!$(N[CX4]C(=O));"
            "!$(N[CX4][CX4][F,Cl,Br]);!$(N[CX4][CX4]O[a,C]);!$(N[CX4][c]);!$(N1CCNCC1)]"
            "([CX4])[CX4]"
        ),
        9.6,
    ),
    # Beta-ether/halo or alpha-carbonyl secondary amine (ulotaront ~8.8, methylphenidate ~8.8)
    (
        "inductive_secondary_amine",
        ("[NX3;H1;!$(N[CX3]);!$(NS(=O)=O);!$(N[a])][$([CX4][CX4][O,N,S,F,Cl,Br]),$([CX4]C(=O))]"),
        8.8,
    ),
    # Alpha-keto secondary amine (e.g., bupropion clinical pKa ~8.2)
    (
        "alpha_keto_secondary_amine",
        "[NX3;H1;!$(N[CX3]);!$(NS(=O)=O);!$(N[a])][CX4]C(=O)[a,C]",
        8.2,
    ),
    # Benzylic secondary amine
    ("benzyl_secondary_amine", "[NX3;H1;!$(N[CX3]);!$(NS(=O)=O);!$(N[a])][CX4][c]", 8.7),
    # General aliphatic tertiary amine without beta-EWG / benzyl / piperazine / benzazepine
    # (e.g., escitalopram ~9.6, pitolisant ~9.4, pimavanserin ~9.3)
    (
        "aliphatic_tertiary_amine",
        (
            "[NX3;H0;!$(N[CX3]);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(N#*);"
            "!$(N[CX4][c]);!$(N1CCNCC1);!$(N[CX4][CX4][O,N,S]);!$(N[CX4]C=C);"
            "!$(N1CCC(O)CC1);!$(N1CCC([a])CC1);!$(N[CX4][CX4][CX4]C(=O));"
            "!$([N;R][CX4;R][CX4;R][c]);!$([N;R][CX4;R][c])]"
        ),
        9.4,
    ),
    # Beta-hydroxyl / 4-arylpiperidine / butyrophenone tertiary amine
    # (e.g., haloperidol ~8.3, risperidone ~8.2, lumateperone ~7.9)
    (
        "piperidinyl_cns_tertiary_amine",
        ("[NX3;H0;!$(N[CX3]);!$(NS(=O)=O);!$(N[a])][$([CX4][CX4][CX4]C(=O)),$([CX4][CX4][O,N,S])]"),
        8.3,
    ),
    (
        "aryl_or_hydroxy_piperidine_amine",
        "[NX3;H0;r6;!$(N[CX3]);!$(N[a])]1CCC([O,a])CC1",
        8.3,
    ),
    # Benzylic or fused benzazepine / tetrahydroisoquinoline tertiary amine
    # (e.g., ecopipam ~8.2, tetrabenazine ~6.5–8.0)
    (
        "benzyl_or_benzazepine_tertiary_amine",
        (
            "[NX3;H0;!$(N[CX3]);!$(NS(=O)=O);!$(N[a]);!$(N1CCNCC1)]"
            "[$([CX4][c]),$([CX4;R][CX4;R][c])]"
        ),
        8.2,
    ),
    # Allylic / tetrahydropyridine tertiary amine (e.g., xanomeline ~8.0)
    ("allylic_tertiary_amine", "[NX3;H0;!$(N[CX3]);!$(N[a])][CX4]C=C", 8.0),
    # Piperazine aliphatic nitrogen (e.g., aripiprazole ~7.6, cariprazine ~8.2)
    ("piperazine_amine", "[NX3;r6;!$(N[CX3]);!$(NS(=O)=O);!$(N[a])]1CCNCC1", 8.0),
    # Imidazole aromatic basic N
    ("imidazole", "[nX2]1c[nX3]cc1", 7.0),
    # Pyridine / quinoline / isoquinoline aromatic N (non-diazine)
    ("pyridine", "[nX2;r6;H0;!$(n[a][nX2]);!$(n[nX2])]", 5.2),
    # Aniline / arylamine (non-amide, non-sulfonamide)
    ("aniline", "[NX3;H2,H1,H0;!$(NC=[O,S]);!$(NS(=O)=O);!$(N=*);!$(N#*)][a]", 4.6),
    # Pyrimidine / pyrazine / pyridazine weak heteroaromatic base
    ("diazine", "[nX2;r6;H0]", 2.5),
)

# Ordered (name, SMARTS, estimated acidic pKa) table for acidic centers (used for ClogD_7.4).
_ACIDIC_PKA_SMARTS: tuple[tuple[str, str, float], ...] = (
    ("sulfonic_acid", "S(=O)(=O)[OX2H1]", 1.5),
    ("phosphonic_acid", "P(=O)([OX2H1])[OX2H1,OX2]", 2.0),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 4.2),
    ("acyl_sulfonamide", "C(=O)[NX3H1]S(=O)(=O)", 4.8),
    ("tetrazole", "c1nnn[nH]1", 4.9),
    ("sulfonamide", "S(=O)(=O)[NX3;H1,H2;!$(NC=O)]", 9.5),
    ("phenol", "[OX2H1][c]", 9.8),
)


@lru_cache(maxsize=1)
def _compiled_basic_rules() -> tuple[tuple[str, Chem.Mol, float], ...]:
    """Compile and cache SMARTS patterns for basic pKa estimation."""
    rules: list[tuple[str, Chem.Mol, float]] = []
    for name, smarts, pka in _BASIC_PKA_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            rules.append((name, pat, pka))
    return tuple(rules)


@lru_cache(maxsize=1)
def _compiled_acidic_rules() -> tuple[tuple[str, Chem.Mol, float], ...]:
    """Compile and cache SMARTS patterns for acidic pKa estimation."""
    rules: list[tuple[str, Chem.Mol, float]] = []
    for name, smarts, pka in _ACIDIC_PKA_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            rules.append((name, pat, pka))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Exact Pfizer CNS MPO Piecewise-Linear Desirability Functions T_0 (Wager 2010/2016)
# ---------------------------------------------------------------------------


def desirability_clogp(clogp: float) -> float:
    """Wager et al. (2010) ClogP desirability: ``<=3.0 -> 1.0``, ``>=5.0 -> 0.0``."""
    if clogp <= 3.0:
        return 1.0
    if clogp >= 5.0:
        return 0.0
    return float((5.0 - clogp) / 2.0)


def desirability_clogd(clogd_74: float) -> float:
    """Wager et al. (2010) ClogD_7.4 desirability: ``<=2.0 -> 1.0``, ``>=4.0 -> 0.0``."""
    if clogd_74 <= 2.0:
        return 1.0
    if clogd_74 >= 4.0:
        return 0.0
    return float((4.0 - clogd_74) / 2.0)


def desirability_mw(mw: float) -> float:
    """Wager et al. (2010) MW desirability: ``<=360 -> 1.0``, ``>=500 -> 0.0``."""
    if mw <= 360.0:
        return 1.0
    if mw >= 500.0:
        return 0.0
    return float((500.0 - mw) / 140.0)


def desirability_tpsa(tpsa: float) -> float:
    """Wager et al. (2010) TPSA hump: ``<=20 -> 0``, ``40..90 -> 1``, ``>=120 -> 0``."""
    if tpsa <= 20.0 or tpsa >= 120.0:
        return 0.0
    if 40.0 <= tpsa <= 90.0:
        return 1.0
    if tpsa < 40.0:
        return float((tpsa - 20.0) / 20.0)
    return float((120.0 - tpsa) / 30.0)


def desirability_hbd(hbd: float) -> float:
    """Wager et al. (2010) HBD desirability: ``<=0.5 -> 1.0``, ``>=3.5 -> 0.0``."""
    if hbd <= 0.5:
        return 1.0
    if hbd >= 3.5:
        return 0.0
    return float((3.5 - hbd) / 3.0)


def desirability_pka(most_basic_pka: float) -> float:
    """Wager et al. (2010) most-basic pKa desirability: ``<=8.0 -> 1.0``, ``>=10.0 -> 0.0``."""
    if most_basic_pka <= 8.0:
        return 1.0
    if most_basic_pka >= 10.0:
        return 0.0
    return float((10.0 - most_basic_pka) / 2.0)


# ---------------------------------------------------------------------------
# SMARTS Physicochemical pKa & Henderson-Hasselbalch ClogD_7.4 Estimation
# ---------------------------------------------------------------------------


def estimate_most_basic_pka(mol: Chem.Mol) -> float:
    """Estimate strongest basic ``pKa`` (conjugate acid) via curated medicinal-chemistry SMARTS.

    Covers quaternary ammonium ions (``14.0``), alkyl guanidines/amidines (``11.5–12.0``),
    aliphatic primary/secondary/tertiary amines (``9.4–9.8``), inductively dampened
    benzylic/beta-heteroatom/piperidinyl CNS amines (``8.0–9.0``), acylguanidines and
    2-aminoimidazolines (``8.1``), piperazines (``8.0``), imidazoles (``7.0``), pyridines
    (``5.2``), anilines (``4.6``), diazines (``2.5``), and neutral molecules (``2.0``).
    """
    best_pka = 2.0
    for _name, pattern, pka in _compiled_basic_rules():
        if mol.HasSubstructMatch(pattern) and pka > best_pka:
            best_pka = pka
    return float(best_pka)


def estimate_most_acidic_pka(mol: Chem.Mol) -> float | None:
    """Estimate strongest acidic ``pKa`` via curated SMARTS, or ``None`` if non-acidic."""
    min_pka: float | None = None
    for _name, pattern, pka in _compiled_acidic_rules():
        if mol.HasSubstructMatch(pattern) and (min_pka is None or pka < min_pka):
            min_pka = pka
    return float(min_pka) if min_pka is not None else None


def estimate_clogd_74(
    mol: Chem.Mol,
    *,
    clogp: float | None = None,
    most_basic_pka: float | None = None,
    most_acidic_pka: float | None = None,
    ph: float = PHYSIOLOGICAL_PH,
) -> float:
    """Compute exact Henderson-Hasselbalch ``ClogD`` at physiological ``pH = 7.4``.

    Uses:
        ``ClogD_pH = ClogP - log10(1 + 10^(pKa_base - pH) + 10^(pH - pKa_acid))``
    with a physically bounded maximum ionization shift of ``4.0`` log units for permanent
    quaternary cations and strong acids to mirror ion-pair octanol/water partitioning.
    """
    logp = float(_CRIPPEN.MolLogP(mol)) if clogp is None else float(clogp)
    pka_b = estimate_most_basic_pka(mol) if most_basic_pka is None else float(most_basic_pka)
    pka_a = estimate_most_acidic_pka(mol) if most_acidic_pka is None else most_acidic_pka

    # Permanent quaternary ammonium ions carry an un-neutralizable positive charge at pH 7.4.
    if pka_b >= 13.5:
        return float(logp - 3.5)

    base_ratio = 10.0 ** (pka_b - ph) if pka_b > 2.5 else 0.0
    acid_ratio = 10.0 ** (ph - pka_a) if pka_a is not None else 0.0
    ion_shift = min(math.log10(1.0 + base_ratio + acid_ratio), 4.0)
    return float(logp - ion_shift)


# ---------------------------------------------------------------------------
# Dataclass & Batch Evaluation API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CNSMPOResult:
    """Complete Pfizer CNS MPO v1/v2, BBB heuristic, and medicinal chemistry profile."""

    smiles: str
    canonical_smiles: str | None
    valid: bool
    # Raw physicochemical properties
    clogp: float
    clogd_74: float
    mw: float
    tpsa: float
    hbd: int
    hba: int
    most_basic_pka: float
    most_acidic_pka: float | None
    # Exact Wager piecewise-linear desirabilities T_0 in [0, 1]
    t0_clogp: float
    t0_clogd: float
    t0_mw: float
    t0_tpsa: float
    t0_hbd: float
    t0_pka: float
    # Composite Pfizer CNS MPO scores in [0, 6]
    cns_mpo_v1: float
    cns_mpo_v2: float
    # CNS BBB heuristics (Clark 1999 / Kelder 1999)
    bbb_clark_pass: bool
    bbb_kelder_tpsa_pass: bool
    # Medicinal chemistry & safety metrics
    qed: float
    sa_score: float
    sa_norm: float
    pains_free: bool
    brenk_free: bool
    lipinski_violations: int
    lipinski_pass: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "valid": self.valid,
            "clogp": self.clogp,
            "clogd_74": self.clogd_74,
            "mw": self.mw,
            "tpsa": self.tpsa,
            "hbd": self.hbd,
            "hba": self.hba,
            "most_basic_pka": self.most_basic_pka,
            "most_acidic_pka": self.most_acidic_pka,
            "t0_clogp": self.t0_clogp,
            "t0_clogd": self.t0_clogd,
            "t0_mw": self.t0_mw,
            "t0_tpsa": self.t0_tpsa,
            "t0_hbd": self.t0_hbd,
            "t0_pka": self.t0_pka,
            "cns_mpo_v1": self.cns_mpo_v1,
            "cns_mpo_v2": self.cns_mpo_v2,
            "bbb_clark_pass": self.bbb_clark_pass,
            "bbb_kelder_tpsa_pass": self.bbb_kelder_tpsa_pass,
            "qed": self.qed,
            "sa_score": self.sa_score,
            "sa_norm": self.sa_norm,
            "pains_free": self.pains_free,
            "brenk_free": self.brenk_free,
            "lipinski_violations": self.lipinski_violations,
            "lipinski_pass": self.lipinski_pass,
        }


def _invalid_cns_mpo(smiles: str) -> CNSMPOResult:
    """Return a zeroed invalid ``CNSMPOResult`` for unparseable SMILES."""
    return CNSMPOResult(
        smiles=smiles,
        canonical_smiles=None,
        valid=False,
        clogp=float("nan"),
        clogd_74=float("nan"),
        mw=float("nan"),
        tpsa=float("nan"),
        hbd=0,
        hba=0,
        most_basic_pka=float("nan"),
        most_acidic_pka=None,
        t0_clogp=0.0,
        t0_clogd=0.0,
        t0_mw=0.0,
        t0_tpsa=0.0,
        t0_hbd=0.0,
        t0_pka=0.0,
        cns_mpo_v1=0.0,
        cns_mpo_v2=0.0,
        bbb_clark_pass=False,
        bbb_kelder_tpsa_pass=False,
        qed=0.0,
        sa_score=10.0,
        sa_norm=0.0,
        pains_free=False,
        brenk_free=False,
        lipinski_violations=4,
        lipinski_pass=False,
    )


def compute_cns_mpo(
    mol_or_smiles: str | Chem.Mol,
    *,
    override_basic_pka: float | None = None,
    override_clogd_74: float | None = None,
) -> CNSMPOResult:
    """Compute exact Pfizer CNS MPO v1/v2 and CNS medicinal chemistry metrics for a ligand.

    Args:
        mol_or_smiles: SMILES string or sanitized RDKit ``Mol``.
        override_basic_pka: Optional external/experimental strongest basic ``pKa``.
        override_clogd_74: Optional external/experimental ``ClogD`` at pH 7.4.

    Returns:
        ``CNSMPOResult`` containing exact ``cns_mpo_v1`` (Wager 2010), ``cns_mpo_v2``
        (Wager 2016), all six ``T_0`` components, BBB heuristics, QED, SA, PAINS/Brenk,
        and Lipinski Ro5 flags.
    """
    if isinstance(mol_or_smiles, str):
        raw_smi = mol_or_smiles
        mol = Chem.MolFromSmiles(raw_smi.strip()) if raw_smi.strip() else None
        if mol is None:
            return _invalid_cns_mpo(raw_smi)
    else:
        mol = mol_or_smiles
        if mol is None:
            return _invalid_cns_mpo("")
        raw_smi = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

    canonical = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    clogp = float(_CRIPPEN.MolLogP(mol))
    mw = float(_DESCRIPTORS.MolWt(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))

    basic_pka = (
        float(override_basic_pka)
        if override_basic_pka is not None
        else estimate_most_basic_pka(mol)
    )
    acidic_pka = estimate_most_acidic_pka(mol)
    clogd_74 = (
        float(override_clogd_74)
        if override_clogd_74 is not None
        else estimate_clogd_74(
            mol,
            clogp=clogp,
            most_basic_pka=basic_pka,
            most_acidic_pka=acidic_pka,
        )
    )

    t0_clogp = desirability_clogp(clogp)
    t0_clogd = desirability_clogd(clogd_74)
    t0_mw = desirability_mw(mw)
    t0_tpsa = desirability_tpsa(tpsa)
    t0_hbd = desirability_hbd(float(hbd))
    t0_pka = desirability_pka(basic_pka)

    # Wager et al. (2010): 6-component sum in [0, 6]
    mpo_v1 = float(t0_clogp + t0_clogd + t0_mw + t0_tpsa + t0_hbd + t0_pka)
    # Wager et al. (2016): replaces ClogD_7.4 with 2 * T_0(HBD), preserving [0, 6] scale
    mpo_v2 = float(t0_clogp + t0_mw + t0_tpsa + t0_pka + 2.0 * t0_hbd)

    # Clark (1999) & Kelder (1999) BBB heuristics
    n_plus_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (7, 8))
    bbb_clark_pass = bool(
        tpsa <= 90.0
        and mw <= 450.0
        and hbd <= 3
        and (n_plus_o <= 5 or (clogp - float(n_plus_o)) > 0.0)
    )
    bbb_kelder_tpsa_pass = bool(tpsa <= 70.0)

    qed_val = float(QED.qed(mol))
    sa_val = get_sa_score(mol)
    sa_norm_val = sa_to_normalized(sa_val)
    pains_ok = is_pains_free(mol)
    brenk_ok = is_brenk_free(mol)
    ro5_viol = lipinski_violations(mol)
    ro5_ok = passes_lipinski(mol, max_violations=1)

    return CNSMPOResult(
        smiles=raw_smi,
        canonical_smiles=canonical,
        valid=True,
        clogp=clogp,
        clogd_74=clogd_74,
        mw=mw,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        most_basic_pka=basic_pka,
        most_acidic_pka=acidic_pka,
        t0_clogp=t0_clogp,
        t0_clogd=t0_clogd,
        t0_mw=t0_mw,
        t0_tpsa=t0_tpsa,
        t0_hbd=t0_hbd,
        t0_pka=t0_pka,
        cns_mpo_v1=float(np.clip(mpo_v1, 0.0, 6.0)),
        cns_mpo_v2=float(np.clip(mpo_v2, 0.0, 6.0)),
        bbb_clark_pass=bbb_clark_pass,
        bbb_kelder_tpsa_pass=bbb_kelder_tpsa_pass,
        qed=qed_val,
        sa_score=sa_val,
        sa_norm=sa_norm_val,
        pains_free=pains_ok,
        brenk_free=brenk_ok,
        lipinski_violations=ro5_viol,
        lipinski_pass=ro5_ok,
    )


def compute_cns_mpo_batch(smiles_list: Sequence[str]) -> list[CNSMPOResult]:
    """Compute ``CNSMPOResult`` entries for a sequence of SMILES strings."""
    return [compute_cns_mpo(smi) for smi in smiles_list]


# Verified FDA / clinical CNS reference drugs from neuro_tourette_research.md
REFERENCE_CNS_DRUGS: dict[str, str] = {
    "guanfacine": "NC(N)=NC(=O)Cc1c(Cl)cccc1Cl",
    "clonidine": "Clc1cccc(Cl)c1NC1=NCCN1",
    "haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "risperidone": "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",
    "ecopipam": "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",
    "valbenazine": "COc1cc2c(cc1OC)[C@H]1C[C@@H](OC(=O)[C@@H](N)C(C)C)[C@H](CC(C)C)CN1CC2",
    "tetrabenazine": "COc1cc2c(cc1OC)[C@H]1CC(=O)[C@H](CC(C)C)CN1CC2",
    "xanomeline": "CCCCCCOc1nsnc1C1=CCCN(C)C1",
    "escitalopram": "CN(C)CCC[C@@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
    "paroxetine": "Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3c(c2)OCO3)cc1",
    "lumateperone": "CN1CCN2c3c(cccc31)[C@@H]1CN(CCCC(=O)c3ccc(F)cc3)CC[C@@H]12",
    "dexmethylphenidate": "COC(=O)[C@H](c1ccccc1)[C@H]1CCCCN1",
    "bupropion": "C[C@H](NC(C)(C)C)C(=O)c1cccc(Cl)c1",
    "ulotaront": "CNC[C@@H]1OCCc2ccsc21",
    "balipodect": "COc1cn(-c2ccc(-n3cccn3)cc2F)nc(-c2ccnn2-c2ccccc2)c1=O",
}

# Verified non-CNS polar / peripheral control drugs (low expected CNS MPO)
REFERENCE_NON_CNS_DRUGS: dict[str, str] = {
    "metformin": "CN(C)C(=N)NC(=N)N",
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "methotrexate": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "ceftriaxone": (
        "CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CSc3nc(=O)c(=O)[nH]n3C)CS[C@H]12)c1csc(N)n1"
    ),
    "doxorubicin": (
        "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1"
    ),
    "trospium_cation": "O=C(O[C@@H]1C[C@@H]2CC[C@H](C1)[N+]21CCCC1)C(O)(c1ccccc1)c1ccccc1",
}


__all__ = [
    "PHYSIOLOGICAL_PH",
    "REFERENCE_CNS_DRUGS",
    "REFERENCE_NON_CNS_DRUGS",
    "CNSMPOResult",
    "compute_cns_mpo",
    "compute_cns_mpo_batch",
    "desirability_clogd",
    "desirability_clogp",
    "desirability_hbd",
    "desirability_mw",
    "desirability_pka",
    "desirability_tpsa",
    "estimate_clogd_74",
    "estimate_most_acidic_pka",
    "estimate_most_basic_pka",
]
