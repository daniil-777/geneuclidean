"""Tests for SMILES corpus deduplication, novelty accounting and corpus statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

from geneuclidean.data.chem_corpora import (
    canonical_or_none,
    corpus_statistics,
    deduplicate_smiles,
    load_smiles_file,
    novelty_report,
    write_smiles_file,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ETHANOL_WRITINGS = ["CCO", "OCC", "C(O)C"]


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_canonical_or_none_accepts_valid_and_rejects_garbage() -> None:
    assert canonical_or_none(ASPIRIN) is not None
    assert canonical_or_none("not_a_molecule") is None
    assert canonical_or_none("C(((C") is None
    assert canonical_or_none("") is None
    assert canonical_or_none("   ") is None


def test_different_writings_of_one_molecule_canonicalize_identically() -> None:
    """This is the whole reason deduplication must canonicalize first."""
    canonical = {canonical_or_none(s) for s in ETHANOL_WRITINGS}
    assert len(canonical) == 1
    assert None not in canonical


def test_strip_salts_keeps_the_largest_fragment() -> None:
    salt = f"{ASPIRIN}.[Na+]"
    with_salt = canonical_or_none(salt, strip_salts=False)
    without = canonical_or_none(salt, strip_salts=True)
    assert with_salt is not None and "Na" in with_salt
    assert without is not None and "Na" not in without


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplicate_collapses_equivalent_writings() -> None:
    unique, stats = deduplicate_smiles(ETHANOL_WRITINGS)
    assert len(unique) == 1
    assert stats.n_input == 3
    assert stats.n_valid == 3
    assert stats.n_unique == 1
    assert stats.n_duplicates == 2
    assert stats.duplicate_rate == pytest.approx(2 / 3)


def test_raw_string_dedup_would_have_missed_those_duplicates() -> None:
    """Guards the design choice: a plain set() on raw strings is not enough."""
    assert len(set(ETHANOL_WRITINGS)) == 3
    unique, _ = deduplicate_smiles(ETHANOL_WRITINGS)
    assert len(unique) == 1


def test_deduplicate_counts_invalid_separately_from_duplicate() -> None:
    corpus = [ASPIRIN, ASPIRIN, "garbage", CAFFEINE, "C(((C", ""]
    unique, stats = deduplicate_smiles(corpus)
    assert stats.n_input == 5  # the empty string is not counted as input
    assert stats.n_valid == 3  # aspirin twice + caffeine
    assert stats.n_unique == 2
    assert stats.n_invalid == 2
    assert stats.n_duplicates == 1
    assert len(unique) == 2


def test_deduplicate_preserves_first_seen_order_by_default() -> None:
    unique, _ = deduplicate_smiles([CAFFEINE, ASPIRIN, CAFFEINE])
    assert unique == [canonical_or_none(CAFFEINE), canonical_or_none(ASPIRIN)]


def test_deduplicate_can_sort_instead() -> None:
    unique, _ = deduplicate_smiles([CAFFEINE, ASPIRIN], keep_order=False)
    assert unique == sorted(unique)


def test_deduplicate_is_deterministic() -> None:
    corpus = [ASPIRIN, CAFFEINE, "CCO", "OCC", "garbage", ASPIRIN]
    first, stats_a = deduplicate_smiles(corpus)
    second, stats_b = deduplicate_smiles(corpus)
    assert first == second
    assert stats_a.to_dict() == stats_b.to_dict()


def test_empty_corpus_produces_zero_rates_not_division_errors() -> None:
    unique, stats = deduplicate_smiles([])
    assert unique == []
    assert stats.duplicate_rate == 0.0
    assert stats.invalid_rate == 0.0
    assert stats.to_dict()["n_unique"] == 0


def test_corpus_statistics_matches_deduplicate() -> None:
    corpus = [ASPIRIN, "CCO", "OCC", "nope"]
    stats = corpus_statistics(corpus)
    _, same = deduplicate_smiles(corpus)
    assert stats.to_dict() == same.to_dict()


def test_stats_to_dict_is_json_serializable() -> None:
    import json

    payload = corpus_statistics([ASPIRIN, ASPIRIN]).to_dict()
    assert json.loads(json.dumps(payload))["n_duplicates"] == 1


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_report_basic_accounting() -> None:
    reference = [ASPIRIN, CAFFEINE]
    generated = [ASPIRIN, "CCO", "CCCO", "garbage"]
    rep = novelty_report(generated, reference)

    assert rep.n_generated == 4
    assert rep.n_valid == 3
    assert rep.n_unique == 3
    assert rep.n_novel == 2  # ethanol and propanol; aspirin is in the reference
    assert rep.n_reference == 2
    assert rep.validity == pytest.approx(0.75)
    assert rep.uniqueness == pytest.approx(1.0)
    assert rep.novelty == pytest.approx(2 / 3)


def test_novelty_canonicalizes_both_sides() -> None:
    """A molecule written differently from its training twin is NOT novel."""
    rep = novelty_report(["OCC"], ["CCO"])
    assert rep.n_valid == 1
    assert rep.n_novel == 0
    assert rep.novelty == 0.0


def test_novelty_includes_invalid_strings_in_the_validity_denominator() -> None:
    """Passing only the valid subset would silently inflate reported validity."""
    rep = novelty_report(["CCO", "garbage", "C(((C"], [ASPIRIN])
    assert rep.n_generated == 3
    assert rep.validity == pytest.approx(1 / 3)


def test_novelty_uniqueness_penalises_repeated_generations() -> None:
    rep = novelty_report(["CCO", "OCC", "C(O)C"], [ASPIRIN])
    assert rep.n_valid == 3
    assert rep.n_unique == 1
    assert rep.uniqueness == pytest.approx(1 / 3)
    assert rep.novelty == pytest.approx(1.0)


def test_novelty_report_records_sample_count_for_interpretability() -> None:
    """Novelty decays with sample count, so the count must travel with the number."""
    payload = novelty_report(["CCO"] * 5, [ASPIRIN]).to_dict()
    assert payload["n_generated"] == 5
    assert "novelty" in payload and "uniqueness" in payload and "validity" in payload


def test_novelty_report_surfaces_examples() -> None:
    rep = novelty_report(["CCO", "CCCO", "CCCCO"], [ASPIRIN], max_examples=2)
    assert len(rep.novel_examples) == 2
    assert all(canonical_or_none(s) == s for s in rep.novel_examples)


def test_novelty_with_empty_generation_does_not_divide_by_zero() -> None:
    rep = novelty_report([], [ASPIRIN])
    assert (rep.validity, rep.uniqueness, rep.novelty) == (0.0, 0.0, 0.0)


def test_novelty_with_empty_reference_marks_everything_novel() -> None:
    rep = novelty_report([ASPIRIN, CAFFEINE], [])
    assert rep.n_reference == 0
    assert rep.novelty == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


def test_smiles_file_roundtrip(tmp_path: Path) -> None:
    out = write_smiles_file([ASPIRIN, CAFFEINE], tmp_path / "nested" / "corpus.smi")
    assert out.is_file()
    assert load_smiles_file(out) == [ASPIRIN, CAFFEINE]


def test_load_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "c.smi"
    path.write_text(f"# header\n{ASPIRIN}\n\n   \n{CAFFEINE}\n", encoding="utf-8")
    assert load_smiles_file(path) == [ASPIRIN, CAFFEINE]


def test_load_takes_first_field_of_delimited_lines(tmp_path: Path) -> None:
    path = tmp_path / "c.smi"
    path.write_text(f"{ASPIRIN}\taspirin\n{CAFFEINE},caffeine,extra\n", encoding="utf-8")
    assert load_smiles_file(path) == [ASPIRIN, CAFFEINE]


def test_load_honours_limit(tmp_path: Path) -> None:
    path = write_smiles_file([ASPIRIN, CAFFEINE, "CCO"], tmp_path / "c.smi")
    assert load_smiles_file(path, limit=2) == [ASPIRIN, CAFFEINE]


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_smiles_file(tmp_path / "absent.smi")
