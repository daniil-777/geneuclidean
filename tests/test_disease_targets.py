"""Tests for verified disease ontology IDs and measured ChEMBL trainability tiers.

No network access: these pin facts that were established by live API queries so a future
change cannot silently reintroduce a wrong ontology ID or an optimistic activity count.
"""

from __future__ import annotations

import json

import pytest

from geneuclidean.cns.disease_targets import (
    DISEASE_ONTOLOGY,
    ONTOLOGY_TRAPS,
    OPEN_TARGETS_GRAPHQL_URL,
    RANKED_TARGETS_QUERY,
    TARGET_DATA_AVAILABILITY,
    ranked_targets_payload,
    targets_by_tier,
    trainable_targets,
)

# ---------------------------------------------------------------------------
# Ontology identifiers
# ---------------------------------------------------------------------------


def test_the_three_indications_have_verified_mondo_ids() -> None:
    assert DISEASE_ONTOLOGY["schizophrenia"].efo_id == "MONDO_0005090"
    assert DISEASE_ONTOLOGY["alzheimer"].efo_id == "MONDO_0004975"
    assert DISEASE_ONTOLOGY["tourette"].efo_id == "MONDO_0007661"


def test_known_bad_ontology_ids_are_recorded_as_traps() -> None:
    """The API returns null for obsolete terms instead of redirecting, so guesses fail
    silently. The two most dangerous point at entirely different diseases."""
    assert "MONDO_0005150" in ONTOLOGY_TRAPS
    assert "macular degeneration" in ONTOLOGY_TRAPS["MONDO_0005150"]
    assert "MONDO_0005260" in ONTOLOGY_TRAPS
    assert "autism" in ONTOLOGY_TRAPS["MONDO_0005260"]
    assert "EFO_0000692" in ONTOLOGY_TRAPS
    assert "null" in ONTOLOGY_TRAPS["EFO_0000692"]


def test_no_trap_id_is_also_a_verified_id() -> None:
    verified = {o.efo_id for o in DISEASE_ONTOLOGY.values()}
    assert verified.isdisjoint(ONTOLOGY_TRAPS)


def test_every_indication_records_a_caveat() -> None:
    """A ranking without its caveat invites drug-precedence circularity."""
    for area, onto in DISEASE_ONTOLOGY.items():
        assert onto.caveat, f"{area} has no caveat"
    assert "circularity" in DISEASE_ONTOLOGY["schizophrenia"].caveat
    assert "recombination" in DISEASE_ONTOLOGY["tourette"].caveat


# ---------------------------------------------------------------------------
# GraphQL payload
# ---------------------------------------------------------------------------


def test_graphql_endpoint_is_the_platform_api() -> None:
    assert OPEN_TARGETS_GRAPHQL_URL.startswith("https://api.platform.opentargets.org")
    # genetics.opentargets.org is retired; it must not be referenced.
    assert "genetics.opentargets.org" not in OPEN_TARGETS_GRAPHQL_URL


def test_ranked_targets_payload_is_json_serializable_and_well_formed() -> None:
    body = ranked_targets_payload("schizophrenia", size=5)
    encoded = json.loads(json.dumps(body))
    assert encoded["variables"] == {"efoId": "MONDO_0005090", "size": 5}
    assert "associatedTargets" in encoded["query"]
    assert "datatypeScores" in encoded["query"], "needed to de-circularize the ranking"


def test_query_requests_tractability_so_untractable_targets_can_be_filtered() -> None:
    assert "tractability" in RANKED_TARGETS_QUERY


@pytest.mark.parametrize("bad", ["parkinsons", "", "SCHIZOPHRENIA"])
def test_payload_rejects_unknown_area(bad: str) -> None:
    with pytest.raises(ValueError, match="Unknown disease area"):
        ranked_targets_payload(bad)  # type: ignore[arg-type]


def test_payload_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="size must be >= 1"):
        ranked_targets_payload("alzheimer", size=0)


# ---------------------------------------------------------------------------
# Measured trainability
# ---------------------------------------------------------------------------


def test_potent_counts_never_exceed_strict_binding_counts() -> None:
    for symbol, t in TARGET_DATA_AVAILABILITY.items():
        assert t.potent <= t.strict_binding, symbol


def test_tier_boundaries() -> None:
    assert TARGET_DATA_AVAILABILITY["BACE1"].tier == "pretrain"  # 6,650 potent
    assert TARGET_DATA_AVAILABILITY["CHRM4"].tier == "finetune"  # 609
    assert TARGET_DATA_AVAILABILITY["GPR52"].tier == "evaluation_only"  # 16
    assert TARGET_DATA_AVAILABILITY["HDC"].tier == "unusable"  # 0


def test_tourette_has_no_pretrainable_target_of_its_own() -> None:
    """The central finding: TS has no standalone medicinal-chemistry corpus.

    Every Tourette target that IS pretrainable is one it shares with schizophrenia
    (DRD2/DRD3/HTR2A/SLC6A4/CNR1); the TS-specific ones are empty.
    """
    assert TARGET_DATA_AVAILABILITY["HDC"].strict_binding == 0
    assert TARGET_DATA_AVAILABILITY["SLC18A2"].tier == "evaluation_only"
    assert TARGET_DATA_AVAILABILITY["SLC6A1"].tier == "evaluation_only"

    shared = set(trainable_targets("tourette")) & set(trainable_targets("schizophrenia"))
    assert shared, "Tourette must be reachable via its shared target families"


def test_the_most_important_schizophrenia_target_is_data_poor() -> None:
    """CHRM4 drives the indication post-Cobenfy yet cannot support pretraining."""
    chrm4 = TARGET_DATA_AVAILABILITY["CHRM4"]
    assert chrm4.potent < 1000
    assert chrm4.tier == "finetune"
    assert "Cobenfy" in chrm4.note


def test_verified_chembl_ids_that_are_commonly_cited_wrong() -> None:
    assert TARGET_DATA_AVAILABILITY["SLC18A2"].chembl_id == "CHEMBL1893"
    assert "CHEMBL1985" in TARGET_DATA_AVAILABILITY["SLC18A2"].note
    assert TARGET_DATA_AVAILABILITY["TAAR1"].chembl_id == "CHEMBL5857"
    assert "CHEMBL4908" in TARGET_DATA_AVAILABILITY["TAAR1"].note


def test_assay_type_and_indication_hazards_are_flagged() -> None:
    assert "IC50" in TARGET_DATA_AVAILABILITY["NLRP3"].note  # no Ki at all
    # Soticlestat is an epilepsy drug; CH24H must not be justified on an AD indication.
    assert "NOT Alzheimer" in TARGET_DATA_AVAILABILITY["CYP46A1"].note


def test_trainable_targets_is_sorted_and_filtered() -> None:
    rows = trainable_targets("alzheimer", min_potent=2000)
    assert rows[0] == "BACE1"
    counts = [TARGET_DATA_AVAILABILITY[s].potent for s in rows]
    assert counts == sorted(counts, reverse=True)
    assert all(c >= 2000 for c in counts)


def test_trainable_targets_threshold_is_respected() -> None:
    assert set(trainable_targets("schizophrenia", min_potent=5000)) <= set(
        trainable_targets("schizophrenia", min_potent=300)
    )
    assert "GPR52" not in trainable_targets("schizophrenia")


def test_trainable_targets_rejects_unknown_area() -> None:
    with pytest.raises(ValueError, match="Unknown disease area"):
        trainable_targets("parkinsons")  # type: ignore[arg-type]


def test_targets_by_tier_filters_by_indication() -> None:
    pretrain_ad = targets_by_tier("pretrain", area="alzheimer")
    assert "BACE1" in pretrain_ad
    assert "DRD2" not in pretrain_ad  # schizophrenia/tourette, not AD
    assert set(targets_by_tier("unusable")) == {"HDC"}


def test_every_target_declares_at_least_one_indication() -> None:
    for symbol, t in TARGET_DATA_AVAILABILITY.items():
        assert t.areas, symbol
        assert set(t.areas) <= set(DISEASE_ONTOLOGY), symbol


def test_dead_but_data_rich_targets_are_kept_deliberately() -> None:
    """5-HT6 failed clinically but is one of the richest CNS SAR sets available."""
    htr6 = TARGET_DATA_AVAILABILITY["HTR6"]
    assert htr6.tier == "pretrain"
    assert "failed clinically" in htr6.note
