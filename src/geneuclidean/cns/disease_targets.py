"""Verified disease ontology IDs and measured ChEMBL trainability for the tri-disorder panel.

Two kinds of fact are recorded here, both established by live API queries on 2026-11-26
rather than from memory, because both are easy to get wrong in ways that fail silently.

**1. Open Targets disease IDs.** The Platform GraphQL API does **not** auto-redirect
obsolete ontology terms -- it returns ``null``. The *web UI does* redirect, so testing an
ID in a browser gives a false positive while the API call in your pipeline quietly yields
nothing. Worse, the IDs one would reasonably guess for this panel point at unrelated
diseases: ``MONDO_0005150`` is age-related macular degeneration, not Tourette syndrome,
and ``MONDO_0005260`` is autism. :data:`DISEASE_ONTOLOGY` records the verified IDs and
:data:`ONTOLOGY_TRAPS` records the wrong ones explicitly so they fail loudly.

**2. Measured ChEMBL activity counts.** Whether a target can actually train a model is an
empirical question, and the answer for this panel is uneven to the point of changing
strategy. Counts below are strict binding activities (``pchembl_value`` present,
``assay_type=B``, no validity comment, not flagged as a duplicate) from ChEMBL 37.

The headline consequence: **Tourette syndrome has no standalone medicinal-chemistry
corpus.** Histidine decarboxylase, its most TS-specific target, has **3 activities in all
of ChEMBL**. GAT1 has 309 and VMAT2 768. Tourette must therefore be modelled as a
recombination of the dopaminergic, serotonergic and GABAergic families it shares with
schizophrenia, not as an indication with its own training set. Conversely the most
important schizophrenia target of 2026 -- CHRM4, following Cobenfy's approval -- has only
609 potent actives, so it is a fine-tuning and evaluation target rather than a pretraining
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DISEASE_ONTOLOGY",
    "ONTOLOGY_TRAPS",
    "OPEN_TARGETS_GRAPHQL_URL",
    "RANKED_TARGETS_QUERY",
    "TARGET_DATA_AVAILABILITY",
    "DiseaseArea",
    "DiseaseOntology",
    "TargetDataAvailability",
    "TrainabilityTier",
    "ranked_targets_payload",
    "targets_by_tier",
    "trainable_targets",
]

DiseaseArea = Literal["schizophrenia", "alzheimer", "tourette"]
TrainabilityTier = Literal["pretrain", "finetune", "evaluation_only", "unusable"]

# POST JSON here; no auth and no API key. Note the schema field is `apiVersion`, not
# `version`. genetics.opentargets.org is retired and 301-redirects to the Platform.
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

RANKED_TARGETS_QUERY = """
query RankedTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}, orderByScore: "score") {
      count
      rows {
        score
        datatypeScores { id score }
        target {
          id
          approvedSymbol
          approvedName
          biotype
          tractability { label modality value }
        }
      }
    }
  }
}
""".strip()


@dataclass(frozen=True)
class DiseaseOntology:
    """A verified disease ontology identifier.

    Attributes:
        area: Which of the three indications this describes.
        efo_id: The identifier to pass to Open Targets as ``efoId``.
        name: The disease name the API returns for it.
        associated_targets: Number of associated targets at data release 26.09.
        top_targets: Highest-scoring associated targets, for sanity-checking a pull.
        caveat: Why the ranking should not be read naively.
    """

    area: DiseaseArea
    efo_id: str
    name: str
    associated_targets: int
    top_targets: tuple[str, ...] = ()
    caveat: str = ""


DISEASE_ONTOLOGY: dict[DiseaseArea, DiseaseOntology] = {
    "schizophrenia": DiseaseOntology(
        area="schizophrenia",
        efo_id="MONDO_0005090",
        name="schizophrenia",
        associated_targets=8161,
        top_targets=("DRD2", "SHANK3", "RTN4R", "DRD3", "MTHFR", "GRIN2A", "DRD4"),
        caveat=(
            "11 of the top 20 are serotonin receptors scored almost entirely on "
            "clinical=0.97, i.e. 'an antipsychotic binds them'. That is drug-precedence "
            "circularity, not target discovery. For non-circular selection, rank on the "
            "genetic_association datatypeScore and drop the clinical component, which "
            "promotes GRM3, SETD1A and GRIN2A and demotes the 5-HT panel."
        ),
    ),
    "alzheimer": DiseaseOntology(
        area="alzheimer",
        efo_id="MONDO_0004975",
        name="Alzheimer disease",
        associated_targets=13006,
        top_targets=("APP", "PSEN1", "PSEN2", "APOE", "ABCA7", "SORL1", "ADAM10"),
        caveat=(
            "The top of this list is dominated by genetics-driven, largely "
            "non-small-molecule-tractable targets (APP, APOE, SORL1). Filter on "
            "tractability before treating them as design targets."
        ),
    ),
    "tourette": DiseaseOntology(
        area="tourette",
        efo_id="MONDO_0007661",
        name="Tourette syndrome",
        associated_targets=0,
        top_targets=(),
        caveat=(
            "Open Targets carries essentially no Tourette association data, and ChEMBL "
            "carries no TS-specific chemistry (HDC has 3 activities). Model Tourette as a "
            "recombination of the shared dopaminergic / serotonergic / GABAergic families "
            "rather than as an indication with its own corpus."
        ),
    ),
}

# Identifiers that look right and are not. The API returns null for obsolete terms
# instead of redirecting, so an unchecked guess yields an empty pull, not an error.
ONTOLOGY_TRAPS: dict[str, str] = {
    "EFO_0000692": "obsolete term for schizophrenia; the API returns null. Use MONDO_0005090.",
    "MONDO_0005150": "age-related macular degeneration, NOT Tourette. Use MONDO_0007661.",
    "MONDO_0005260": "autism, NOT Tourette. Use MONDO_0007661.",
    "EFO_1001870": (
        "late-onset Alzheimer's disease -- a valid but much narrower term "
        "(229 targets vs 13,006). Use MONDO_0004975 unless you specifically want LOAD."
    ),
    "EFO_0004262": "returns null.",
    "EFO_0004250": "returns null.",
}


@dataclass(frozen=True)
class TargetDataAvailability:
    """Measured ChEMBL 37 data volume for one target, and what it can be used for.

    Attributes:
        symbol: HGNC gene symbol.
        uniprot: UniProt accession.
        chembl_id: Human single-protein ChEMBL target identifier.
        strict_binding: Activities with a pChEMBL value, ``assay_type=B``, no validity
            comment and not flagged as a potential duplicate.
        potent: Subset of the above with ``pchembl_value >= 7``.
        areas: Indications this target serves.
        note: Anything that would mislead a naive pull.
    """

    symbol: str
    uniprot: str
    chembl_id: str
    strict_binding: int
    potent: int
    areas: tuple[DiseaseArea, ...]
    note: str = ""

    @property
    def tier(self) -> TrainabilityTier:
        """What this volume of data can realistically support.

        Training tiers gate on *potent* actives, because a generative model learns from
        what good binding looks like. The evaluation tier gates on *total* strict binding
        records instead, because evaluating a generated series only requires a handful of
        known actives to compare against, potent or not. GPR52 is the case that forces the
        distinction: 23 strict records but only 16 potent ones -- unusable for training,
        still usable as a zero-shot generalization pocket.
        """
        if self.potent >= 2000:
            return "pretrain"
        if self.potent >= 300:
            return "finetune"
        if self.strict_binding >= 20:
            return "evaluation_only"
        return "unusable"


# Counts measured live against the ChEMBL 37 REST API on 2026-11-26.
TARGET_DATA_AVAILABILITY: dict[str, TargetDataAvailability] = {
    t.symbol: t
    for t in (
        # --- data-rich: enough for target-conditioned pretraining -------------------
        TargetDataAvailability("BACE1", "P56817", "CHEMBL4822", 13405, 6650, ("alzheimer",)),
        TargetDataAvailability(
            "DRD3", "P35462", "CHEMBL234", 7752, 5032, ("schizophrenia", "tourette")
        ),
        TargetDataAvailability(
            "DRD2", "P14416", "CHEMBL217", 11422, 4692, ("schizophrenia", "tourette")
        ),
        TargetDataAvailability("PDE10A", "Q9Y233", "CHEMBL4409", 5846, 4344, ("schizophrenia",)),
        TargetDataAvailability(
            "HTR2A", "P28223", "CHEMBL224", 6999, 4240, ("schizophrenia", "tourette")
        ),
        TargetDataAvailability("SLC6A4", "P31645", "CHEMBL228", 6938, 3561, ("tourette",)),
        TargetDataAvailability(
            "HTR6",
            "P50406",
            "CHEMBL3371",
            4549,
            3018,
            ("alzheimer",),
            note=(
                "5-HT6 failed clinically (idalopirdine, intepirdine) but is one of the "
                "richest CNS SAR sets available. Keep as training data despite the dead "
                "target -- for a generative prior that is a feature, not a bug."
            ),
        ),
        TargetDataAvailability("CNR1", "P21554", "CHEMBL218", 5354, 2596, ("tourette",)),
        TargetDataAvailability("GSK3B", "P49841", "CHEMBL262", 5451, 2330, ("alzheimer",)),
        TargetDataAvailability("ACHE", "P22303", "CHEMBL220", 7223, 2028, ("alzheimer",)),
        # --- usable for finetuning and evaluation -----------------------------------
        TargetDataAvailability("SLC6A3", "Q01959", "CHEMBL238", 4842, 1733, ("tourette",)),
        TargetDataAvailability("BCHE", "P06276", "CHEMBL1914", 4203, 1108, ("alzheimer",)),
        TargetDataAvailability(
            "CHRM1", "P11229", "CHEMBL216", 2408, 911, ("schizophrenia", "tourette")
        ),
        TargetDataAvailability(
            "DRD1", "P21728", "CHEMBL2056", 2025, 752, ("schizophrenia", "tourette")
        ),
        TargetDataAvailability(
            "CHRM4",
            "P08173",
            "CHEMBL1821",
            1591,
            609,
            ("schizophrenia", "tourette"),
            note=(
                "The most important schizophrenia target of 2026 -- Cobenfy "
                "(xanomeline-trospium) approved 2024-09-26, with direclidine in "
                "registrational Phase 3 -- yet only 609 potent actives. Finetune and "
                "evaluate here; do not expect to pretrain on it."
            ),
        ),
        TargetDataAvailability("SLC6A9", "P48067", "CHEMBL2337", 1094, 595, ("schizophrenia",)),
        TargetDataAvailability(
            "TMEM97",
            "Q5BJF2",
            "CHEMBL4105907",
            1022,
            434,
            ("alzheimer",),
            note="Sigma-2. Zervimesine/CT1812 in Phase 2. Well populated for its novelty.",
        ),
        TargetDataAvailability(
            "OGA",
            "O60502",
            "CHEMBL5921",
            788,
            362,
            ("alzheimer",),
            note="MGEA5. Real med-chem exists (MK-8719, thiamet-G, LY3372689).",
        ),
        TargetDataAvailability(
            "NLRP3",
            "Q96P20",
            "CHEMBL1741208",
            915,
            316,
            ("alzheimer",),
            note="Zero Ki records -- all functional IC50. Do not mix assay types.",
        ),
        # --- evaluation / zero-shot generalization only ------------------------------
        TargetDataAvailability(
            "SLC18A2",
            "Q05940",
            "CHEMBL1893",
            146,
            26,
            ("tourette",),
            note="VMAT2. Verified ChEMBL id is CHEMBL1893, not CHEMBL1985.",
        ),
        TargetDataAvailability(
            "SLC6A1", "P30531", "CHEMBL1903", 121, 35, ("tourette",), note="GAT1."
        ),
        TargetDataAvailability(
            "TAAR1",
            "Q96RJ0",
            "CHEMBL5857",
            71,
            38,
            ("schizophrenia",),
            note=(
                "Scientifically alive but data-poor: 71 strict records. Ulotaront failed "
                "DIAMOND 1 and 2. Verified ChEMBL id is CHEMBL5857, not CHEMBL4908."
            ),
        ),
        TargetDataAvailability(
            "CYP46A1",
            "Q9Y6A2",
            "CHEMBL4523510",
            89,
            56,
            ("alzheimer",),
            note="CH24H. Soticlestat is Dravet/LGS epilepsy, NOT Alzheimer's.",
        ),
        TargetDataAvailability(
            "GPR52",
            "Q9Y2T5",
            "CHEMBL3297639",
            23,
            16,
            ("schizophrenia",),
            note="23 strict records. Viable only as a zero-shot evaluation pocket.",
        ),
        # --- unusable ----------------------------------------------------------------
        TargetDataAvailability(
            "HDC",
            "P19113",
            "CHEMBL4523192",
            0,
            0,
            ("tourette",),
            note=(
                "3 activities in all of ChEMBL, 0 strict binding. The most "
                "Tourette-specific target in the panel has no chemistry whatsoever. This "
                "is why Tourette cannot be modelled as a standalone indication."
            ),
        ),
    )
}


def ranked_targets_payload(area: DiseaseArea, size: int = 25) -> dict[str, object]:
    """Build the Open Targets GraphQL POST body for one indication.

    Args:
        area: Which indication to query.
        size: How many associated targets to return.

    Returns:
        A JSON-serializable body for :data:`OPEN_TARGETS_GRAPHQL_URL`.

    Raises:
        ValueError: If ``area`` is unknown or ``size`` is not positive.
    """
    if area not in DISEASE_ONTOLOGY:
        raise ValueError(
            f"Unknown disease area {area!r}. Expected one of {sorted(DISEASE_ONTOLOGY)}."
        )
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return {
        "query": RANKED_TARGETS_QUERY,
        "variables": {"efoId": DISEASE_ONTOLOGY[area].efo_id, "size": int(size)},
    }


def targets_by_tier(tier: TrainabilityTier, area: DiseaseArea | None = None) -> list[str]:
    """List target symbols in a trainability tier, optionally filtered by indication.

    Args:
        tier: Which tier to return.
        area: Restrict to targets serving this indication.

    Returns:
        Target symbols, ordered by potent-activity count descending.
    """
    rows = [
        t
        for t in TARGET_DATA_AVAILABILITY.values()
        if t.tier == tier and (area is None or area in t.areas)
    ]
    rows.sort(key=lambda t: -t.potent)
    return [t.symbol for t in rows]


def trainable_targets(area: DiseaseArea, min_potent: int = 300) -> list[str]:
    """List targets for an indication with enough potent actives to train on.

    Args:
        area: Which indication.
        min_potent: Minimum activities at ``pchembl_value >= 7``. The default of 300
            corresponds to the ``finetune`` tier boundary.

    Returns:
        Target symbols, ordered by potent-activity count descending.

    Raises:
        ValueError: If ``area`` is unknown.
    """
    if area not in DISEASE_ONTOLOGY:
        raise ValueError(
            f"Unknown disease area {area!r}. Expected one of {sorted(DISEASE_ONTOLOGY)}."
        )
    rows = [
        t
        for t in TARGET_DATA_AVAILABILITY.values()
        if area in t.areas and t.potent >= int(min_potent)
    ]
    rows.sort(key=lambda t: -t.potent)
    return [t.symbol for t in rows]
