"""SMILES corpus hygiene: deduplication, novelty accounting and corpus statistics.

Deduplication is the cheapest available quality improvement for a chemical language
model, and the evidence for it is specific. GP-MoLFormer was trained on 1.1B SMILES of
which **~45% were duplicates**, and reports **novelty 0.323** at 30k samples -- 68% of its
generations are *verbatim* training molecules. Deduplicating the corpus to 650M unique
molecules lifted novelty to **0.390** with no change to the architecture. Novelty also
*decays* as more samples are drawn (0.390 @30k -> 0.340 @1B -> 0.214 @10B), so the sample
count at which novelty is measured has to be reported alongside it.

Deduplicating on the raw SMILES string is not enough: the same molecule has many valid
SMILES writings, so ``CCO``, ``OCC`` and ``C(O)C`` are one molecule but three strings.
:func:`deduplicate_smiles` therefore canonicalizes with RDKit first, which is what makes
the measured duplicate rate meaningful.

Novelty is only honest if the reference set is canonicalized the same way, which is why
:func:`novelty_report` canonicalizes both sides rather than comparing raw strings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

__all__ = [
    "CorpusStats",
    "NoveltyReport",
    "canonical_or_none",
    "corpus_statistics",
    "deduplicate_smiles",
    "load_smiles_file",
    "novelty_report",
    "write_smiles_file",
]


@dataclass(frozen=True)
class CorpusStats:
    """Summary of a SMILES corpus after canonicalization.

    Attributes:
        n_input: Number of non-empty input strings.
        n_valid: How many parsed successfully with RDKit.
        n_unique: Number of distinct canonical SMILES.
    """

    n_input: int
    n_valid: int
    n_unique: int

    @property
    def n_invalid(self) -> int:
        """Number of inputs RDKit rejected."""
        return self.n_input - self.n_valid

    @property
    def n_duplicates(self) -> int:
        """Number of redundant valid molecules removed by canonical deduplication."""
        return self.n_valid - self.n_unique

    @property
    def duplicate_rate(self) -> float:
        """Redundant fraction of the valid molecules (``0.0`` for an empty corpus)."""
        return self.n_duplicates / self.n_valid if self.n_valid else 0.0

    @property
    def invalid_rate(self) -> float:
        """Fraction of inputs RDKit could not parse."""
        return self.n_invalid / self.n_input if self.n_input else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary including the derived rates."""
        return {
            "n_input": self.n_input,
            "n_valid": self.n_valid,
            "n_unique": self.n_unique,
            "n_invalid": self.n_invalid,
            "n_duplicates": self.n_duplicates,
            "duplicate_rate": round(self.duplicate_rate, 6),
            "invalid_rate": round(self.invalid_rate, 6),
        }


@dataclass(frozen=True)
class NoveltyReport:
    """Novelty of a generated sample set against a reference corpus.

    Attributes:
        n_generated: Number of generated strings supplied.
        n_valid: How many parsed with RDKit.
        n_unique: Distinct canonical molecules among the valid ones.
        n_novel: Unique molecules absent from the reference corpus.
        n_reference: Size of the canonicalized reference corpus.
        novel_examples: A few novel molecules, for eyeballing.
    """

    n_generated: int
    n_valid: int
    n_unique: int
    n_novel: int
    n_reference: int
    novel_examples: list[str] = field(default_factory=list)

    @property
    def validity(self) -> float:
        """Valid fraction of all generated strings."""
        return self.n_valid / self.n_generated if self.n_generated else 0.0

    @property
    def uniqueness(self) -> float:
        """Distinct fraction of the valid molecules."""
        return self.n_unique / self.n_valid if self.n_valid else 0.0

    @property
    def novelty(self) -> float:
        """Fraction of distinct molecules not present in the reference corpus."""
        return self.n_novel / self.n_unique if self.n_unique else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary including the derived rates.

        ``n_generated`` is included deliberately: novelty decays with sample count, so a
        novelty figure without its sample size is not interpretable.
        """
        payload = asdict(self)
        payload.update(
            validity=round(self.validity, 6),
            uniqueness=round(self.uniqueness, 6),
            novelty=round(self.novelty, 6),
        )
        return payload


def canonical_or_none(smiles: str, *, strip_salts: bool = False) -> str | None:
    """Canonicalize one SMILES string, returning ``None`` when RDKit rejects it.

    Args:
        smiles: Input SMILES string.
        strip_salts: Keep only the largest disconnected fragment by heavy-atom count.

    Returns:
        Canonical isomeric SMILES, or ``None`` if the input cannot be parsed.
    """
    text = smiles.strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    if strip_salts and "." in text:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        mol = max(frags, key=lambda f: (f.GetNumHeavyAtoms(), f.GetNumAtoms()))
    try:
        canonical = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    except Exception:
        return None
    return canonical or None


def deduplicate_smiles(
    smiles: Iterable[str],
    *,
    strip_salts: bool = False,
    keep_order: bool = True,
) -> tuple[list[str], CorpusStats]:
    """Canonicalize and deduplicate a SMILES corpus.

    Deduplication is performed on the *canonical* form, so different writings of the same
    molecule collapse together. Comparing raw strings would leave most duplicates in
    place and understate the duplicate rate.

    Args:
        smiles: Input SMILES strings.
        strip_salts: Reduce multi-fragment entries to their largest fragment first.
        keep_order: Preserve first-seen order (reproducible); otherwise sort
            lexicographically.

    Returns:
        ``(unique_canonical_smiles, stats)``.
    """
    n_input = 0
    n_valid = 0
    seen: set[str] = set()
    unique: list[str] = []

    for raw in smiles:
        if not raw or not raw.strip():
            continue
        n_input += 1
        canonical = canonical_or_none(raw, strip_salts=strip_salts)
        if canonical is None:
            continue
        n_valid += 1
        if canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)

    if not keep_order:
        unique.sort()

    return unique, CorpusStats(n_input=n_input, n_valid=n_valid, n_unique=len(unique))


def corpus_statistics(smiles: Iterable[str], *, strip_salts: bool = False) -> CorpusStats:
    """Measure validity and duplication of a corpus."""
    _, stats = deduplicate_smiles(smiles, strip_salts=strip_salts)
    return stats


def novelty_report(
    generated: Sequence[str],
    reference: Iterable[str],
    *,
    strip_salts: bool = False,
    max_examples: int = 10,
) -> NoveltyReport:
    """Score generated molecules for validity, uniqueness and novelty.

    Both sides are canonicalized before comparison, so a generated molecule written
    differently from its training-set twin is still correctly counted as *not* novel.

    Args:
        generated: Generated SMILES strings, including any invalid ones. Passing only the
            valid subset would inflate the reported validity.
        reference: Training corpus to measure novelty against.
        strip_salts: Reduce multi-fragment entries to their largest fragment first.
        max_examples: How many novel molecules to include as examples.

    Returns:
        A :class:`NoveltyReport`.
    """
    ref_canonical = {
        canonical
        for canonical in (canonical_or_none(s, strip_salts=strip_salts) for s in reference)
        if canonical is not None
    }

    n_valid = 0
    unique: list[str] = []
    seen: set[str] = set()
    for raw in generated:
        canonical = canonical_or_none(raw, strip_salts=strip_salts) if raw else None
        if canonical is None:
            continue
        n_valid += 1
        if canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)

    novel = [s for s in unique if s not in ref_canonical]
    return NoveltyReport(
        n_generated=len(generated),
        n_valid=n_valid,
        n_unique=len(unique),
        n_novel=len(novel),
        n_reference=len(ref_canonical),
        novel_examples=novel[:max_examples],
    )


def load_smiles_file(path: str | Path, *, limit: int | None = None) -> list[str]:
    """Read a one-SMILES-per-line text file.

    Blank lines and ``#`` comment lines are skipped. If a line contains whitespace or a
    comma, only the first field is taken, so plain ``.smi``, whitespace-delimited and
    simple CSV exports all load without a separate parser.

    Args:
        path: File to read.
        limit: Stop after this many molecules.

    Returns:
        SMILES strings in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"SMILES file not found: {src}")

    out: list[str] = []
    with src.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            first = text.replace(",", " ").split()[0]
            if first:
                out.append(first)
            if limit is not None and len(out) >= limit:
                break
    return out


def write_smiles_file(smiles: Iterable[str], path: str | Path) -> Path:
    """Write one SMILES per line, creating parent directories as needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(smiles) + "\n", encoding="utf-8")
    return dest
