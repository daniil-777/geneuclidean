"""Large-scale protein pocket-ligand corpora: CrossDocked2020 and leakage-safe splits.

GenEuclidean has historically trained its 3D pocket encoders on ~4.5k PDBbind crystal
pockets and its chemical prior on 249k ZINC-250k SMILES. This module adds the corpora
that the structure-based design literature actually compares against, together with the
split machinery needed to make those comparisons honest.

**CrossDocked2020** (Francoeur et al., *J. Chem. Inf. Model.* 2020, 60(9):4200-4215,
doi:10.1021/acs.jcim.0c00411) is the lingua franca of pocket-conditioned generative
models and is released under **CC0 1.0 Universal** (public domain), so it carries none of
the licensing friction of PDBbind (paywalled from v2021), Binding MOAD (shut down
2024-07-01) or BindingNet (unresolved CC BY vs CC BY-NC conflict).

The single highest-leverage finding encoded here concerns the canonical split. Luo et al.
(NeurIPS 2021, arXiv:2203.10446) filter CrossDocked's 22.5M docked protein-ligand pairs
to RMSD <= 1 A, obtaining **184,057 qualifying pairs**, then *randomly draw only 100,000*
for training (quoted verbatim below, except that the paper writes the Angstrom sign where
this ASCII docstring writes ``A``):

    "We filter out data points whose binding pose RMSD is greater than 1A, leading to a
    refined subset consisting of 184,057 data points. We use mmseqs2 to cluster data at
    30% sequence identity, and randomly draw 100,000 protein-ligand pairs for training
    and 100 proteins from remaining clusters for testing."

Roughly 84,000 qualifying complexes are therefore discarded by every published baseline
in that lineage. :func:`build_rmsd_filtered_pairs` does not apply that cap, so the
**1.84x** (184,057 / 100,000) is available at zero download or compute cost. Pass
``limit=100_000`` to reproduce the canonical baseline instead.

Read that 1.84x precisely: it is the ratio of the *filtered* set to the published
*training draw*. The function's own default also deduplicates to the best pose per
(pocket group, ligand), which returns fewer than 184,057 pairs -- a better default for
training, but not the literature count. See :func:`build_rmsd_filtered_pairs`.

Note on pocket geometry: GenEuclidean extracts pockets at 10 A *nearest-atom* distance
(see :func:`geneuclidean.data.featurizer.extract_pocket_from_files`). The CrossDocked
lineage instead crops whole residues whose mass-weighted **centre of mass** lies within
10 A of any ligand atom, while DiffSBDD uses 8 A nearest-atom. These three conventions
produce measurably different residue sets, so pocket sizes here are not directly
comparable to published CrossDocked numbers. :data:`POCKET_CONVENTIONS` records the
published definitions so experiments can state which one they used.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "CROSSDOCKED_BASE_URL",
    "CROSSDOCKED_LICENSE_URL",
    "CROSSDOCKED_VARIANTS",
    "POCKET_CONVENTIONS",
    "CorpusVariant",
    "CrossDockedPair",
    "PocketConvention",
    "SplitManifest",
    "build_rmsd_filtered_pairs",
    "crossdocked_download_plan",
    "extract_pocket_group",
    "group_pairs_by_pocket",
    "iter_crossdocked_types",
    "make_pocket_disjoint_split",
    "parse_types_line",
    "summarize_pairs",
    "write_split_manifest",
]

# CC0 1.0 Universal. Verified against the published LICENSE file in the same directory.
CROSSDOCKED_BASE_URL = "https://bits.csb.pitt.edu/files/crossdock2020"
CROSSDOCKED_LICENSE_URL = f"{CROSSDOCKED_BASE_URL}/LICENSE"


@dataclass(frozen=True)
class CorpusVariant:
    """One downloadable CrossDocked2020 artifact.

    Attributes:
        name: Short identifier used by :func:`crossdocked_download_plan`.
        filename: File name under :data:`CROSSDOCKED_BASE_URL`.
        approx_bytes: Approximate size, for budgeting only (not an integrity check).
        tier: ``"smoke"`` (CI), ``"medium"`` (trains a real model) or ``"full"``.
        description: What the artifact contains and when to choose it.
    """

    name: str
    filename: str
    approx_bytes: int
    tier: Literal["smoke", "medium", "full"]
    description: str

    @property
    def url(self) -> str:
        """Fully qualified download URL."""
        return f"{CROSSDOCKED_BASE_URL}/{self.filename}"

    @property
    def approx_gib(self) -> float:
        """Approximate size in GiB."""
        return self.approx_bytes / (1024.0**3)


_GIB = 1024**3
_MIB = 1024**2

CROSSDOCKED_VARIANTS: dict[str, CorpusVariant] = {
    "types_smoke": CorpusVariant(
        name="types_smoke",
        filename="downsampled_CrossDocked2020_v1.3_types.tgz",
        approx_bytes=21 * _MIB,
        tier="smoke",
        description=(
            "Label/index layer only (no structures): pocket group, ligand path, pose RMSD "
            "and affinity. Enough to build and test splits offline. Downloads in seconds."
        ),
    ),
    "types_full": CorpusVariant(
        name="types_full",
        filename="CrossDocked2020_v1.3_types.tgz",
        approx_bytes=3500 * _MIB,
        tier="medium",
        description="Complete types index for all 22.5M poses across ~2,900 pockets.",
    ),
    "structures_downsampled": CorpusVariant(
        name="structures_downsampled",
        filename="downsampled_CrossDocked2020_v1.3.tgz",
        approx_bytes=int(7.3 * _GIB),
        tier="medium",
        description=(
            "Downsampled receptor/ligand structures. The practical choice for training a "
            "real pocket-conditioned decoder without the 50 GiB full archive."
        ),
    ),
    "structures_full": CorpusVariant(
        name="structures_full",
        filename="CrossDocked2020_v1.3.tgz",
        approx_bytes=50 * _GIB,
        tier="full",
        description="All 22.5M docked poses and all ~2,900 pockets.",
    ),
}


@dataclass(frozen=True)
class PocketConvention:
    """A published pocket-cropping convention, recorded so experiments can cite one.

    Attributes:
        name: Convention identifier.
        radius_angstrom: Distance cutoff in Angstroms.
        criterion: How the distance is measured.
        source: Paper or code the convention comes from.
    """

    name: str
    radius_angstrom: float
    criterion: Literal["residue_center_of_mass", "any_atom", "plip_plus_any_atom"]
    source: str


POCKET_CONVENTIONS: dict[str, PocketConvention] = {
    "crossdocked_lineage": PocketConvention(
        name="crossdocked_lineage",
        radius_angstrom=10.0,
        criterion="residue_center_of_mass",
        source=(
            "Luo et al. NeurIPS 2021 / TargetDiff extract_pockets.py "
            "(criterion='center_of_mass'); inherited by Pocket2Mol, 3DSBDD, ResGen, "
            "DecompDiff"
        ),
    ),
    "diffsbdd": PocketConvention(
        name="diffsbdd",
        radius_angstrom=8.0,
        criterion="any_atom",
        source="DiffSBDD process_crossdock.py (--dist_cutoff default 8.0)",
    ),
    "plinder": PocketConvention(
        name="plinder",
        radius_angstrom=6.0,
        criterion="plip_plus_any_atom",
        source="PLINDER: residues with a PLIP interaction plus residues within 6 A",
    ),
    "geneuclidean": PocketConvention(
        name="geneuclidean",
        radius_angstrom=10.0,
        criterion="any_atom",
        source="geneuclidean.data.featurizer.extract_pocket_from_files (this library)",
    ),
}


@dataclass(frozen=True)
class CrossDockedPair:
    """One (pocket, ligand pose) record parsed from a CrossDocked ``.types`` line.

    Attributes:
        label: Binary binder label as released (``1`` = binder).
        affinity: Negative log affinity (``pK``) when present, else ``None``.
        rmsd: Pose RMSD to the crystal ligand in Angstroms, else ``None``.
        receptor: Receptor file path, relative to the archive root.
        ligand: Ligand file path, relative to the archive root.
    """

    label: int
    affinity: float | None
    rmsd: float | None
    receptor: str
    ligand: str

    @property
    def pocket_group(self) -> str:
        """Pocketome group directory, which CrossDocked uses as the pocket identity."""
        return extract_pocket_group(self.receptor)


_NUMERIC = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def parse_types_line(line: str) -> CrossDockedPair | None:
    """Parse one CrossDocked ``.types`` line.

    The released layout is ``<label> <pK> <RMSD> <receptor> <ligand>``, with trailing
    ``#`` comments and occasional missing numeric columns. Parsing is therefore done
    positionally over the leading numeric fields and the two trailing path-like tokens,
    which tolerates the variants present across CrossDocked releases.

    Args:
        line: A single line from a ``.types`` file.

    Returns:
        A :class:`CrossDockedPair`, or ``None`` for blank lines, comments and
        unparseable rows.
    """
    text = line.split("#", 1)[0].strip()
    if not text:
        return None

    fields = text.split()
    paths = [f for f in fields if "/" in f or f.endswith((".gninatypes", ".sdf", ".pdb"))]
    if len(paths) < 2:
        return None
    receptor, ligand = paths[-2], paths[-1]

    numbers: list[float] = []
    for token in fields:
        if token in (receptor, ligand):
            break
        if _NUMERIC.match(token):
            numbers.append(float(token))

    label = int(numbers[0]) if numbers else 0
    affinity = numbers[1] if len(numbers) > 1 else None
    rmsd = numbers[2] if len(numbers) > 2 else None

    # CrossDocked encodes "no measured affinity" as 0.0; keep that distinguishable.
    if affinity is not None and affinity == 0.0:
        affinity = None

    return CrossDockedPair(
        label=label,
        affinity=affinity,
        rmsd=rmsd,
        receptor=receptor,
        ligand=ligand,
    )


def extract_pocket_group(receptor_path: str) -> str:
    """Return the Pocketome group directory that identifies a CrossDocked pocket.

    CrossDocked groups receptors by binding-site similarity using Pocketome v17.12, and
    the group directory (e.g. ``TACR2_HUMAN_1_355_0``) is the dataset's notion of "the
    same pocket". Splitting on this string is a strictly stronger guard than splitting on
    PDB id.

    Args:
        receptor_path: Receptor path from a ``.types`` line.

    Returns:
        The leading directory component, or the bare filename when there is no directory.
    """
    cleaned = receptor_path.strip().lstrip("./")
    head, _, tail = cleaned.partition("/")
    return head if tail else cleaned


def iter_crossdocked_types(
    source: str | Path,
    *,
    member_suffix: str = ".types",
) -> Iterator[CrossDockedPair]:
    """Stream :class:`CrossDockedPair` records from a ``.types`` file, directory or tarball.

    Streams rather than materializing, because the full index covers 22.5M poses.

    Args:
        source: A ``.types`` file, a directory containing them, or a ``.tgz``/``.tar.gz``
            archive of them.
        member_suffix: Suffix identifying types members inside a directory or archive.

    Yields:
        One :class:`CrossDockedPair` per parseable line.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CrossDocked types source not found: {path}")

    if path.is_dir():
        for child in sorted(path.rglob(f"*{member_suffix}")):
            yield from _iter_types_file(child)
        return

    if path.suffix in (".tgz", ".gz") or path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(member_suffix):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                for raw in handle:
                    parsed = parse_types_line(raw.decode("utf-8", errors="replace"))
                    if parsed is not None:
                        yield parsed
        return

    yield from _iter_types_file(path)


def _iter_types_file(path: Path) -> Iterator[CrossDockedPair]:
    """Stream parsed records from a single ``.types`` file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parsed = parse_types_line(raw)
            if parsed is not None:
                yield parsed


def build_rmsd_filtered_pairs(
    pairs: Iterable[CrossDockedPair],
    *,
    rmsd_threshold: float = 1.0,
    require_binder_label: bool = True,
    dedupe_by_ligand: bool = True,
    limit: int | None = None,
    seed: int = 42,
) -> list[CrossDockedPair]:
    """Apply the canonical RMSD filter without the canonical 100,000-pair cap.

    Luo et al. (2021) filter to RMSD <= 1 A, yielding 184,057 pairs, then randomly draw
    only 100,000 for training. Leaving ``limit=None`` keeps the whole filtered set
    instead of that draw.

    .. note::
       Two caveats on the "1.84x more data" framing, so it is not overstated:

       * The 1.84x refers to **removing the 100,000 cap** (184,057 / 100,000), not to the
         size of what this function returns.
       * ``dedupe_by_ligand`` defaults to ``True``, which keeps only the best pose per
         (pocket group, ligand) and therefore returns **fewer than 184,057** pairs. That
         is the scientifically preferable default -- without it a single ligand can
         appear hundreds of times at different pose qualities and dominate the loss --
         but it means the return value is not the literature count. Pass
         ``dedupe_by_ligand=False`` to reproduce 184,057, and ``limit=100_000`` to
         reproduce the published training subset.

       Records whose RMSD column is missing are dropped (see ``rmsd_threshold``), so a
       ``.types`` variant with absent numeric columns yields a smaller set than the
       published one. The parse is positional over the leading numeric fields, so verify
       the recovered count against the release notes before trusting it.

    Args:
        pairs: Parsed CrossDocked records.
        rmsd_threshold: Maximum pose RMSD in Angstroms. Records with no RMSD are dropped.
        require_binder_label: Keep only records with ``label == 1``.
        dedupe_by_ligand: Keep the lowest-RMSD pose per (pocket group, ligand) pair, so
            one ligand does not appear hundreds of times at different pose qualities.
        limit: Optional cap, applied as a deterministic random draw after filtering.
        seed: Seed for that draw.

    Returns:
        Deterministically ordered qualifying pairs.
    """
    kept: list[CrossDockedPair] = []
    for pair in pairs:
        if pair.rmsd is None or pair.rmsd > float(rmsd_threshold):
            continue
        if require_binder_label and pair.label != 1:
            continue
        kept.append(pair)

    if dedupe_by_ligand:
        best: dict[tuple[str, str], CrossDockedPair] = {}
        for pair in kept:
            key = (pair.pocket_group, Path(pair.ligand).name)
            incumbent = best.get(key)
            if incumbent is None or (pair.rmsd or 0.0) < (incumbent.rmsd or 0.0):
                best[key] = pair
        kept = list(best.values())

    kept.sort(key=lambda p: (p.pocket_group, p.ligand))

    if limit is not None and limit < len(kept):
        rng = random.Random(seed)
        kept = sorted(rng.sample(kept, limit), key=lambda p: (p.pocket_group, p.ligand))

    return kept


def group_pairs_by_pocket(
    pairs: Iterable[CrossDockedPair],
) -> dict[str, list[CrossDockedPair]]:
    """Group records by Pocketome pocket group, preserving input order within a group."""
    grouped: dict[str, list[CrossDockedPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.pocket_group].append(pair)
    return dict(grouped)


@dataclass
class SplitManifest:
    """A pocket-disjoint train/validation/test split over CrossDocked pairs.

    Attributes:
        train: Training pairs.
        val: Validation pairs.
        test: Test pairs.
        metadata: Provenance recorded alongside the split.
    """

    train: list[CrossDockedPair] = field(default_factory=list)
    val: list[CrossDockedPair] = field(default_factory=list)
    test: list[CrossDockedPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sizes(self) -> dict[str, int]:
        """Number of pairs in each split."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def pocket_groups(self) -> dict[str, set[str]]:
        """Pocket groups present in each split."""
        return {
            "train": {p.pocket_group for p in self.train},
            "val": {p.pocket_group for p in self.val},
            "test": {p.pocket_group for p in self.test},
        }

    def assert_disjoint(self) -> None:
        """Raise if any pocket group appears in more than one split.

        Raises:
            ValueError: If the splits share a pocket group.
        """
        groups = self.pocket_groups()
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = groups[a] & groups[b]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"{len(overlap)} pocket group(s) leak between {a} and {b}: {sample}"
                )


def _stable_bucket(name: str, seed: int) -> float:
    """Map a pocket-group name to a stable value in ``[0, 1)``.

    Uses BLAKE2b rather than :func:`hash`, whose salt varies per process, so a split is
    reproducible across machines and runs.
    """
    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def make_pocket_disjoint_split(
    pairs: Iterable[CrossDockedPair],
    *,
    val_fraction: float = 0.05,
    test_pocket_count: int | None = 100,
    test_fraction: float = 0.05,
    seed: int = 42,
    extra_metadata: Mapping[str, Any] | None = None,
) -> SplitManifest:
    """Split pairs so that no pocket group appears in more than one split.

    Pocket-group disjointness is the minimum bar. It is **not** sufficient on its own:
    DockGen shows proteins at 22% sequence identity can share a binding site, and
    CleanSplit shows 53%-identity pairs reaching TM-score 0.93 with identical sites. For
    a publication-grade split, additionally cluster by structure (Foldseek or TM-align) or
    ECOD T-group, and by ligand Morgan/Tanimoto similarity. The ``requires_followup``
    metadata flag records that this split has not had those applied.

    Args:
        pairs: Filtered CrossDocked pairs.
        val_fraction: Approximate fraction of pocket groups held out for validation.
        test_pocket_count: Hold out exactly this many pocket groups for test, matching the
            canonical "100 test proteins" protocol. ``None`` uses ``test_fraction``.
        test_fraction: Fraction of pocket groups for test when ``test_pocket_count`` is
            ``None``.
        seed: Seed for the deterministic hash-based assignment.
        extra_metadata: Additional provenance to record.

    Returns:
        A :class:`SplitManifest` whose splits share no pocket group.

    Raises:
        ValueError: If the fractions are out of range or there are too few pocket groups.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")

    grouped = group_pairs_by_pocket(pairs)
    group_names = sorted(grouped)
    if not group_names:
        raise ValueError("Cannot split an empty pair collection.")

    ranked = sorted(group_names, key=lambda name: _stable_bucket(name, seed))

    if test_pocket_count is not None:
        n_test = min(int(test_pocket_count), max(0, len(ranked) - 2))
    else:
        n_test = int(round(len(ranked) * test_fraction))
    n_val = int(round(len(ranked) * val_fraction))

    if n_test + n_val >= len(ranked):
        raise ValueError(
            f"Requested {n_test} test and {n_val} val pocket groups but only "
            f"{len(ranked)} exist; lower the fractions or supply more pockets."
        )

    test_groups = set(ranked[:n_test])
    val_groups = set(ranked[n_test : n_test + n_val])

    manifest = SplitManifest(
        metadata={
            "corpus": "CrossDocked2020",
            "license": "CC0-1.0",
            "license_url": CROSSDOCKED_LICENSE_URL,
            "seed": int(seed),
            "num_pocket_groups": len(ranked),
            "split_criterion": "pocketome_group_disjoint",
            "requires_followup": (
                "Pocket-group disjointness only. Add structure clustering (Foldseek / "
                "TM-align / ECOD T-group) and ligand Tanimoto filtering before publishing."
            ),
            **(dict(extra_metadata) if extra_metadata else {}),
        }
    )

    for name in group_names:
        bucket = grouped[name]
        if name in test_groups:
            manifest.test.extend(bucket)
        elif name in val_groups:
            manifest.val.extend(bucket)
        else:
            manifest.train.extend(bucket)

    manifest.metadata["sizes"] = manifest.sizes
    manifest.assert_disjoint()
    return manifest


def summarize_pairs(pairs: Sequence[CrossDockedPair]) -> dict[str, Any]:
    """Summarize a pair collection for logging and dataset cards.

    Args:
        pairs: Pairs to summarize.

    Returns:
        Counts of pairs, pocket groups and unique ligands, plus RMSD and affinity
        coverage.
    """
    if not pairs:
        return {
            "num_pairs": 0,
            "num_pocket_groups": 0,
            "num_unique_ligands": 0,
            "with_affinity": 0,
            "mean_rmsd": None,
        }

    rmsds = [p.rmsd for p in pairs if p.rmsd is not None]
    return {
        "num_pairs": len(pairs),
        "num_pocket_groups": len({p.pocket_group for p in pairs}),
        "num_unique_ligands": len({Path(p.ligand).name for p in pairs}),
        "with_affinity": sum(1 for p in pairs if p.affinity is not None),
        "mean_rmsd": (sum(rmsds) / len(rmsds)) if rmsds else None,
    }


def write_split_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    """Serialize a :class:`SplitManifest` to JSON.

    Args:
        manifest: The split to write.
        path: Destination ``.json`` path; parent directories are created.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": manifest.metadata,
        **{
            name: [
                {
                    "receptor": p.receptor,
                    "ligand": p.ligand,
                    "pocket_group": p.pocket_group,
                    "rmsd": p.rmsd,
                    "affinity": p.affinity,
                }
                for p in bucket
            ]
            for name, bucket in (
                ("train", manifest.train),
                ("val", manifest.val),
                ("test", manifest.test),
            )
        },
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def crossdocked_download_plan(
    tier: Literal["smoke", "medium", "full"] = "smoke",
) -> list[CorpusVariant]:
    """Return the CrossDocked artifacts to fetch for a given budget tier.

    No download is performed. CrossDocked is served from a plain directory index with no
    login, so :func:`geneuclidean.data.download.download_file` can fetch each ``url``.

    Args:
        tier: ``"smoke"`` (~21 MiB, index only), ``"medium"`` (~7.3 GiB, trains a real
            model) or ``"full"`` (~50 GiB, every pose).

    Returns:
        Variants to download, smallest first.

    Raises:
        ValueError: If ``tier`` is unknown.
    """
    order: dict[str, tuple[str, ...]] = {
        "smoke": ("types_smoke",),
        "medium": ("types_smoke", "structures_downsampled"),
        "full": ("types_full", "structures_full"),
    }
    if tier not in order:
        raise ValueError(f"Unknown tier {tier!r}. Expected one of {sorted(order)}.")
    return [CROSSDOCKED_VARIANTS[name] for name in order[tier]]
