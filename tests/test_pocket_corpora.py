"""Offline tests for CrossDocked2020 corpus parsing and leakage-safe pocket splits.

Every test runs against synthetic ``.types`` content written into ``tmp_path``; nothing
here touches the network or any real dataset.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from geneuclidean.data.pocket_corpora import (
    CROSSDOCKED_BASE_URL,
    CROSSDOCKED_VARIANTS,
    POCKET_CONVENTIONS,
    CrossDockedPair,
    build_rmsd_filtered_pairs,
    crossdocked_download_plan,
    extract_pocket_group,
    group_pairs_by_pocket,
    iter_crossdocked_types,
    make_pocket_disjoint_split,
    parse_types_line,
    summarize_pairs,
    write_split_manifest,
)

# Synthetic lines in the released layout: label  pK  RMSD  receptor  ligand
# Pocket-group directory names are shortened from the real ones to keep lines readable.
SYNTHETIC_TYPES = """\
1 6.05 0.3827 TACR2_HUMAN_0/1lid_A_rec.gninatypes TACR2_HUMAN_0/1lid_dro_lig_0.gninatypes
1 6.05 2.4310 TACR2_HUMAN_0/1lid_A_rec.gninatypes TACR2_HUMAN_0/1lid_dro_lig_1.gninatypes
1 7.20 0.5500 ABL1_HUMAN_0/2hyy_A_rec.gninatypes ABL1_HUMAN_0/2hyy_sti_lig_0.gninatypes
0 0.00 0.9100 ABL1_HUMAN_0/2hyy_A_rec.gninatypes ABL1_HUMAN_0/decoy_lig_0.gninatypes
1 0.00 0.8000 KCNA1_HUMAN_0/1abc_A_rec.gninatypes KCNA1_HUMAN_0/1abc_xyz_lig_0.gninatypes

# a comment line that must be ignored
1 5.00 0.1000 EGFR_HUMAN_0/3abc_A_rec.gninatypes EGFR_HUMAN_0/3abc_erl_lig_0.gninatypes  # note
"""


def _write_types(tmp_path: Path, name: str = "sample.types") -> Path:
    path = tmp_path / name
    path.write_text(SYNTHETIC_TYPES, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------


def test_parse_types_line_extracts_all_fields() -> None:
    line = (
        "1 6.05 0.3827 TACR2_HUMAN_1_355_0/1lid_A_rec.gninatypes "
        "TACR2_HUMAN_1_355_0/1lid_dro_lig_tt_min_0.gninatypes"
    )
    pair = parse_types_line(line)
    assert pair is not None
    assert pair.label == 1
    assert pair.affinity == pytest.approx(6.05)
    assert pair.rmsd == pytest.approx(0.3827)
    assert pair.receptor.endswith("1lid_A_rec.gninatypes")
    assert pair.ligand.endswith("1lid_dro_lig_tt_min_0.gninatypes")
    # Real CrossDocked group names are long; the parser must handle them unchanged.
    assert pair.pocket_group == "TACR2_HUMAN_1_355_0"


@pytest.mark.parametrize("line", ["", "   ", "# only a comment", "1 2 3"])
def test_parse_types_line_rejects_unusable_rows(line: str) -> None:
    assert parse_types_line(line) is None


def test_parse_types_line_strips_trailing_comment() -> None:
    pair = parse_types_line("1 5.0 0.1 A_GROUP/r.gninatypes A_GROUP/l.gninatypes  # note")
    assert pair is not None
    assert pair.ligand == "A_GROUP/l.gninatypes"


def test_zero_affinity_is_treated_as_missing() -> None:
    """CrossDocked encodes 'no measured affinity' as 0.0, which is not a real pKd."""
    pair = parse_types_line("1 0.00 0.5 G/r.gninatypes G/l.gninatypes")
    assert pair is not None
    assert pair.affinity is None
    assert pair.rmsd == pytest.approx(0.5)


def test_extract_pocket_group_handles_bare_filenames() -> None:
    assert extract_pocket_group("GROUP_X/rec.gninatypes") == "GROUP_X"
    assert extract_pocket_group("./GROUP_Y/rec.gninatypes") == "GROUP_Y"
    assert extract_pocket_group("rec.gninatypes") == "rec.gninatypes"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_iter_types_from_file(tmp_path: Path) -> None:
    pairs = list(iter_crossdocked_types(_write_types(tmp_path)))
    assert len(pairs) == 6
    assert {p.pocket_group for p in pairs} == {
        "TACR2_HUMAN_0",
        "ABL1_HUMAN_0",
        "KCNA1_HUMAN_0",
        "EGFR_HUMAN_0",
    }


def test_iter_types_from_directory(tmp_path: Path) -> None:
    nested = tmp_path / "types"
    nested.mkdir()
    (nested / "a.types").write_text(SYNTHETIC_TYPES, encoding="utf-8")
    (nested / "b.types").write_text(SYNTHETIC_TYPES, encoding="utf-8")
    assert len(list(iter_crossdocked_types(nested))) == 12


def test_iter_types_from_tarball(tmp_path: Path) -> None:
    src = _write_types(tmp_path)
    archive = tmp_path / "bundle.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname="it/sample.types")
    assert len(list(iter_crossdocked_types(archive))) == 6


def test_iter_types_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        list(iter_crossdocked_types(tmp_path / "nope.types"))


# ---------------------------------------------------------------------------
# The 184k-vs-100k filter
# ---------------------------------------------------------------------------


def test_rmsd_filter_drops_bad_poses_and_decoys(tmp_path: Path) -> None:
    pairs = list(iter_crossdocked_types(_write_types(tmp_path)))
    kept = build_rmsd_filtered_pairs(pairs, rmsd_threshold=1.0)

    # The 2.431 A pose and the label-0 decoy must both be gone.
    assert all((p.rmsd or 0.0) <= 1.0 for p in kept)
    assert all(p.label == 1 for p in kept)
    assert not any("decoy" in p.ligand for p in kept)
    assert len(kept) == 4


def test_dedupe_keeps_lowest_rmsd_pose_per_ligand() -> None:
    pairs = [
        CrossDockedPair(1, 6.0, 0.90, "G/rec.gninatypes", "G/lig.gninatypes"),
        CrossDockedPair(1, 6.0, 0.20, "G/rec.gninatypes", "G/lig.gninatypes"),
        CrossDockedPair(1, 6.0, 0.55, "G/rec.gninatypes", "G/lig.gninatypes"),
    ]
    kept = build_rmsd_filtered_pairs(pairs, dedupe_by_ligand=True)
    assert len(kept) == 1
    assert kept[0].rmsd == pytest.approx(0.20)


def test_keeping_every_qualifying_pair_is_the_default() -> None:
    """The canonical pipeline throws away ~84k of 184k pairs; we must keep them all."""
    pairs = [
        CrossDockedPair(1, 6.0, 0.5, f"G{i}/rec.gninatypes", f"G{i}/lig.gninatypes")
        for i in range(500)
    ]
    assert len(build_rmsd_filtered_pairs(pairs)) == 500
    assert len(build_rmsd_filtered_pairs(pairs, limit=100)) == 100


def test_limited_draw_is_deterministic_across_calls() -> None:
    pairs = [
        CrossDockedPair(1, 6.0, 0.5, f"G{i}/rec.gninatypes", f"G{i}/lig.gninatypes")
        for i in range(200)
    ]
    a = build_rmsd_filtered_pairs(pairs, limit=50, seed=7)
    b = build_rmsd_filtered_pairs(pairs, limit=50, seed=7)
    c = build_rmsd_filtered_pairs(pairs, limit=50, seed=8)
    assert [p.ligand for p in a] == [p.ligand for p in b]
    assert [p.ligand for p in a] != [p.ligand for p in c]


def test_pairs_without_rmsd_are_dropped() -> None:
    pairs = [CrossDockedPair(1, 6.0, None, "G/rec.gninatypes", "G/lig.gninatypes")]
    assert build_rmsd_filtered_pairs(pairs) == []


# ---------------------------------------------------------------------------
# Pocket-disjoint splitting
# ---------------------------------------------------------------------------


def _many_pairs(n_groups: int, per_group: int = 3) -> list[CrossDockedPair]:
    return [
        CrossDockedPair(1, 6.0, 0.4, f"GRP{g:04d}/rec.gninatypes", f"GRP{g:04d}/lig{i}.gninatypes")
        for g in range(n_groups)
        for i in range(per_group)
    ]


def test_split_is_pocket_disjoint() -> None:
    manifest = make_pocket_disjoint_split(_many_pairs(300), val_fraction=0.1, test_pocket_count=100)
    manifest.assert_disjoint()  # must not raise
    groups = manifest.pocket_groups()
    assert len(groups["test"]) == 100
    assert groups["train"] & groups["test"] == set()
    assert groups["train"] & groups["val"] == set()


def test_split_is_reproducible_across_processes() -> None:
    """Splits must not depend on PYTHONHASHSEED, so they use BLAKE2b not hash()."""
    a = make_pocket_disjoint_split(_many_pairs(200), test_pocket_count=20, seed=3)
    b = make_pocket_disjoint_split(_many_pairs(200), test_pocket_count=20, seed=3)
    assert sorted(a.pocket_groups()["test"]) == sorted(b.pocket_groups()["test"])

    c = make_pocket_disjoint_split(_many_pairs(200), test_pocket_count=20, seed=4)
    assert sorted(a.pocket_groups()["test"]) != sorted(c.pocket_groups()["test"])


def test_split_preserves_every_pair() -> None:
    pairs = _many_pairs(150)
    manifest = make_pocket_disjoint_split(pairs, val_fraction=0.1, test_pocket_count=30)
    assert sum(manifest.sizes.values()) == len(pairs)


def test_split_records_provenance_and_followup_warning() -> None:
    manifest = make_pocket_disjoint_split(_many_pairs(120), test_pocket_count=10)
    meta = manifest.metadata
    assert meta["corpus"] == "CrossDocked2020"
    assert meta["license"] == "CC0-1.0"
    assert meta["split_criterion"] == "pocketome_group_disjoint"
    # Honest about what this split does not yet guard against.
    assert "Foldseek" in meta["requires_followup"]


def test_assert_disjoint_detects_injected_leak() -> None:
    manifest = make_pocket_disjoint_split(_many_pairs(50), test_pocket_count=5)
    manifest.train.append(manifest.test[0])
    with pytest.raises(ValueError, match="leak between train and test"):
        manifest.assert_disjoint()


def test_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty pair collection"):
        make_pocket_disjoint_split([])


def test_split_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError, match="only"):
        make_pocket_disjoint_split(_many_pairs(10), val_fraction=0.9, test_pocket_count=8)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_split_rejects_out_of_range_val_fraction(bad: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        make_pocket_disjoint_split(_many_pairs(20), val_fraction=bad)


def test_group_pairs_by_pocket() -> None:
    grouped = group_pairs_by_pocket(_many_pairs(4, per_group=2))
    assert len(grouped) == 4
    assert all(len(v) == 2 for v in grouped.values())


# ---------------------------------------------------------------------------
# Summaries and serialization
# ---------------------------------------------------------------------------


def test_summarize_pairs(tmp_path: Path) -> None:
    pairs = build_rmsd_filtered_pairs(list(iter_crossdocked_types(_write_types(tmp_path))))
    summary = summarize_pairs(pairs)
    assert summary["num_pairs"] == 4
    assert summary["num_pocket_groups"] == 4
    assert summary["with_affinity"] == 3  # one row has pK 0.00 -> missing
    assert 0.0 < summary["mean_rmsd"] <= 1.0


def test_summarize_empty_pairs() -> None:
    summary = summarize_pairs([])
    assert summary["num_pairs"] == 0
    assert summary["mean_rmsd"] is None


def test_write_split_manifest_roundtrips(tmp_path: Path) -> None:
    manifest = make_pocket_disjoint_split(_many_pairs(60), test_pocket_count=10)
    out = write_split_manifest(manifest, tmp_path / "nested" / "split.json")
    assert out.is_file()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"metadata", "train", "val", "test"}
    assert payload["metadata"]["sizes"] == manifest.sizes
    assert len(payload["test"]) == len(manifest.test)
    assert set(payload["test"][0]) == {
        "receptor",
        "ligand",
        "pocket_group",
        "rmsd",
        "affinity",
    }


# ---------------------------------------------------------------------------
# Download plan metadata
# ---------------------------------------------------------------------------


def test_download_plan_tiers_are_ordered_by_size() -> None:
    smoke = crossdocked_download_plan("smoke")
    assert len(smoke) == 1
    assert smoke[0].approx_bytes < 100 * 1024**2

    medium = crossdocked_download_plan("medium")
    assert sum(v.approx_bytes for v in medium) > sum(v.approx_bytes for v in smoke)

    full = crossdocked_download_plan("full")
    assert sum(v.approx_bytes for v in full) > sum(v.approx_bytes for v in medium)


def test_download_plan_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError, match="Unknown tier"):
        crossdocked_download_plan("gigantic")  # type: ignore[arg-type]


def test_variant_urls_are_well_formed() -> None:
    for variant in CROSSDOCKED_VARIANTS.values():
        assert variant.url.startswith(CROSSDOCKED_BASE_URL + "/")
        assert variant.url.endswith(variant.filename)
        assert variant.approx_gib > 0.0


def test_pocket_conventions_document_the_published_definitions() -> None:
    """Pocket radius alone is ambiguous; the criterion must be recorded alongside it."""
    assert POCKET_CONVENTIONS["crossdocked_lineage"].radius_angstrom == 10.0
    assert POCKET_CONVENTIONS["crossdocked_lineage"].criterion == "residue_center_of_mass"
    assert POCKET_CONVENTIONS["diffsbdd"].radius_angstrom == 8.0
    assert POCKET_CONVENTIONS["diffsbdd"].criterion == "any_atom"
    # This library's own convention differs from both and must say so.
    ours = POCKET_CONVENTIONS["geneuclidean"]
    assert (ours.radius_angstrom, ours.criterion) == (10.0, "any_atom")
    assert ours != POCKET_CONVENTIONS["crossdocked_lineage"]
