"""Pure-RDKit SAFE (Sequential Attachment-based Fragment Embedding) codec.

Provides zero-dependency SAFE molecular encoding, decoding, validation, and
3D-conditioned scaffold-decoration prefix construction using only RDKit —
avoiding external ``safe-mol`` / ``datamol`` transitive pins while preserving
100% exact isomeric canonical SMILES round-trip fidelity.

Reference:
    Noutahi et al., *Gotta be SAFE: A New Framework for Molecular Design*,
    Digital Discovery 3, 796–804 (2024), ``arXiv:2310.10773``.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# Exact regex used by datamol-io/safe (SAFESplitter.REGEX_PATTERN) so stereocenters
# ([C@H], [C@@H]), halogens (Cl, Br), and multi-digit ring closures (%10, %(100))
# remain indivisible single tokens under tokenizers.pre_tokenizers.Split.
SAFE_SPLITTER_REGEX: str = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|"
    r"\%\([0-9]{1,5}\)|\%[0-9]{2}|[0-9])"
)

_BRACKET_ATOM_RE = re.compile(r"\[[^\]]*\]")
_PAREN_RING_RE = re.compile(r"%\((\d+)\)")
_TWO_DIGIT_RING_RE = re.compile(r"%(\d{2})")
_SINGLE_DIGIT_RING_RE = re.compile(r"\d")
_DUMMY_BRANCH_RE = re.compile(r"\(([=#/\\\-]?)\[(\d+)\*]\)")
_DUMMY_INLINE_RE = re.compile(r"([=#/\\\-]?)\[(\d+)\*]")
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def _format_ring_digit(ring_num: int) -> str:
    """Format an integer ring-closure number according to OpenSMILES / RDKit syntax."""
    if ring_num < 1:
        raise ValueError(f"Ring closure index must be >= 1, got {ring_num}")
    if ring_num < 10:
        return str(ring_num)
    if ring_num < 100:
        return f"%{ring_num}"
    return f"%({ring_num})"


def _max_ring_digit(smiles: str) -> int:
    """Return the maximum ring-closure integer appearing in a SMILES string (0 if none)."""
    stripped = _BRACKET_ATOM_RE.sub("", smiles)
    nums: list[int] = [int(m.group(1)) for m in _PAREN_RING_RE.finditer(stripped)]
    stripped = _PAREN_RING_RE.sub("", stripped)
    nums.extend(int(m.group(1)) for m in _TWO_DIGIT_RING_RE.finditer(stripped))
    stripped = _TWO_DIGIT_RING_RE.sub("", stripped)
    nums.extend(int(m.group(0)) for m in _SINGLE_DIGIT_RING_RE.finditer(stripped))
    return max(nums, default=0)


def _replace_dummy_atoms_with_ring_digits(frag_smiles: str) -> str:
    """Substitute RDKit isotope-labeled dummy attachments ``[k*]`` with ring closures."""

    def _repl(match: re.Match[str]) -> str:
        bond_sym = match.group(1)
        ring_idx = int(match.group(2))
        return f"{bond_sym}{_format_ring_digit(ring_idx)}"

    replaced = _DUMMY_BRANCH_RE.sub(_repl, frag_smiles)
    return _DUMMY_INLINE_RE.sub(_repl, replaced)


def _find_cleavable_bonds(mol: Chem.Mol, slicer: str) -> list[int]:
    """Identify non-ring cleavable bond indices according to the requested slicer."""
    mode = slicer.strip().lower()
    if mode == "none":
        return []
    if mode == "brics":
        bond_idxs: list[int] = []
        for (a1, a2), _ in BRICS.FindBRICSBonds(mol):
            bond = mol.GetBondBetweenAtoms(int(a1), int(a2))
            if bond is None or bond.IsInRing():
                continue
            # Preserve E/Z double-bond stereochemistry by skipping stereo-defined C=C cuts.
            if (
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetStereo() != Chem.BondStereo.STEREONONE
            ):
                continue
            bond_idxs.append(int(bond.GetIdx()))
        return sorted(set(bond_idxs))
    if mode == "rotatable":
        if _ROTATABLE_BOND_SMARTS is None:
            return []
        rot_idxs: list[int] = []
        for match in mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS):
            bond = mol.GetBondBetweenAtoms(int(match[0]), int(match[1]))
            if bond is not None and not bond.IsInRing():
                rot_idxs.append(int(bond.GetIdx()))
        return sorted(set(rot_idxs))
    raise ValueError(
        f"Unsupported SAFE slicer {slicer!r}; expected one of 'brics', 'rotatable', 'none'."
    )


def _shuffle_fragments(frags: list[str], rng: Any = None) -> list[str]:
    """Return a permuted copy of SAFE fragments using an optional numpy or stdlib RNG."""
    if len(frags) <= 1:
        return list(frags)
    out = list(frags)
    if rng is None:
        random.shuffle(out)
        return out
    if hasattr(rng, "permutation"):
        perm = rng.permutation(len(out))
        return [out[int(i)] for i in perm]
    if hasattr(rng, "shuffle"):
        rng.shuffle(out)
        return out
    raise TypeError(f"Unsupported RNG type for SAFE fragment shuffling: {type(rng).__name__}")


def smiles_to_safe(
    smiles: str,
    slicer: str = "brics",
    *,
    canonical: bool = True,
    randomize_order: bool = False,
    rng: Any = None,
) -> str:
    """Encode a SMILES string into a dot-separated SAFE string using pure RDKit.

    Cleaves non-ring bonds (default: BRICS medicinal-chemistry retrosynthetic rules),
    replaces cleaved attachment points with matched ring-closure digits above the
    molecule's maximum existing ring index, and joins the resulting fragment blocks
    with ``"."``.

    Args:
        smiles: Input SMILES string.
        slicer: Bond-cleavage rule (``"brics"``, ``"rotatable"``, or ``"none"``).
        canonical: Whether to canonicalize each fragment SMILES (default ``True``).
        randomize_order: If ``True``, permute the fragment order (consistent with
            SAFE-GPT's unordered fragment pretraining augmentation).
        rng: Optional ``numpy.random.Generator`` or ``random.Random`` used when
            ``randomize_order=True``.

    Returns:
        Valid SAFE string representing the exact input molecule (including stereochemistry).

    Raises:
        ValueError: If ``smiles`` cannot be parsed by RDKit or ``slicer`` is unknown.
    """
    cleaned = smiles.strip()
    if not cleaned:
        raise ValueError("Cannot encode an empty SMILES string to SAFE.")
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        raise ValueError(f"Invalid SMILES string for SAFE encoding: {smiles!r}")

    ref_canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    bond_indices = _find_cleavable_bonds(mol, slicer=slicer)
    if not bond_indices:
        return ref_canonical

    start_digit = _max_ring_digit(ref_canonical) + 1
    dummy_labels = [(start_digit + i, start_digit + i) for i in range(len(bond_indices))]
    frag_mol = Chem.FragmentOnBonds(
        mol,
        bond_indices,
        addDummies=True,
        dummyLabels=dummy_labels,
    )
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    frag_strings: list[str] = []
    for frag in frags:
        # Root at a heavy atom so attachment digits never appear in leading position.
        root_idx = next(
            (int(atom.GetIdx()) for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0),
            0,
        )
        frag_smi = Chem.MolToSmiles(
            frag,
            isomericSmiles=True,
            canonical=canonical,
            rootedAtAtom=root_idx,
        )
        frag_strings.append(_replace_dummy_atoms_with_ring_digits(frag_smi))

    if randomize_order and len(frag_strings) > 1:
        frag_strings = _shuffle_fragments(frag_strings, rng=rng)

    safe_str = ".".join(frag_strings)

    # Guarantee exact isomeric round-trip fidelity (fallback to canonical SMILES if any
    # exotic stereocenter/ring edge case fails reconstruction).
    check_mol = Chem.MolFromSmiles(safe_str)
    if check_mol is None:
        return ref_canonical
    if Chem.MolToSmiles(check_mol, canonical=True, isomericSmiles=True) != ref_canonical:
        return ref_canonical
    return safe_str


def safe_to_smiles(
    safe_str: str,
    canonical: bool = True,
    *,
    ignore_stereo: bool = False,
    largest_fragment: bool = True,
) -> str | None:
    """Decode a SAFE string back into a SMILES string using pure RDKit.

    Because every valid SAFE string is mathematically a dot-separated SMILES string
    whose inter-fragment bonds are encoded as matched ring-closure digits, RDKit
    reconstructs the bonded molecular graph directly.

    Args:
        safe_str: Input SAFE string (or standard SMILES string).
        canonical: Whether to return canonical SMILES (default ``True``).
        ignore_stereo: If ``True``, strip stereochemistry from the output SMILES.
        largest_fragment: If ``True`` and multiple disconnected components remain,
            select the component with the largest heavy-atom count.

    Returns:
        Decoded SMILES string, or ``None`` if ``safe_str`` is empty or invalid.
    """
    if not isinstance(safe_str, str):
        return None
    cleaned = "".join(safe_str.split())
    if not cleaned:
        return None

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None

    if largest_fragment:
        frags: Sequence[Chem.Mol] = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        if len(frags) > 1:
            mol = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))

    return str(Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=not ignore_stereo))


def is_valid_safe(safe_str: str, *, allow_disconnected: bool = False) -> bool:
    """Return ``True`` if ``safe_str`` decodes to a chemically valid SAFE molecule.

    By default requires all attachment ring closures to match and produce a single
    connected sanitized molecule (set ``allow_disconnected=True`` to accept salts or
    unlinked multi-component strings).
    """
    if not isinstance(safe_str, str):
        return False
    cleaned = "".join(safe_str.split())
    if not cleaned:
        return False
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    return allow_disconnected or len(Chem.GetMolFrags(mol)) == 1


def split_safe_fragments(safe_str: str) -> list[str]:
    """Split a SAFE string into its non-empty dot-separated fragment blocks."""
    cleaned = "".join(safe_str.split())
    return [part for part in cleaned.split(".") if part]


def build_scaffold_prefix(
    core_smiles_with_dummy: str,
    *,
    attachment_start: int | None = None,
    trailing_dot: bool = True,
) -> str:
    """Convert a dummy-annotated core scaffold SMILES into a SAFE prefix for R-group decoration.

    Replaces dummy attachment atoms (``*``, ``[*]``, ``[1*]``, ``[2*]``, ...) with open
    SAFE ring-closure digits above the core's internal ring indices and optionally appends
    ``"."`` so an autoregressive ``SAFE-GPT`` model immediately completes the decorating
    fragment(s) inside a 3D pocket.

    If ``core_smiles_with_dummy`` contains no dummy atoms, converts the molecule to SAFE
    and returns its largest fragment block (with trailing ``"."`` when requested).

    Args:
        core_smiles_with_dummy: Scaffold SMILES with dummy attachment points (e.g.
            ``"c1ccc([*])cc1"``, ``"[1*]c1ccc([2*])cc1"``) or a full molecule SMILES.
        attachment_start: Optional minimum ring-closure digit for attachment points.
        trailing_dot: Whether to append ``"."`` to begin the next fragment block.

    Returns:
        SAFE fragment prefix string ready for autoregressive continuation.

    Raises:
        ValueError: If ``core_smiles_with_dummy`` is empty, invalid, or contains only
            dummy atoms without a heavy-atom core.
    """
    cleaned = core_smiles_with_dummy.strip()
    if not cleaned:
        raise ValueError("core_smiles_with_dummy must be a non-empty SMILES string.")
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        raise ValueError(f"Invalid scaffold SMILES: {core_smiles_with_dummy!r}")

    heavy_indices = [int(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    if not heavy_indices:
        raise ValueError(
            f"Scaffold {core_smiles_with_dummy!r} must contain at least one heavy atom."
        )
    dummy_indices = [int(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() == 0]

    if not dummy_indices:
        # No dummy atoms supplied: cleave via BRICS and return the largest core fragment.
        safe_full = smiles_to_safe(cleaned, slicer="brics", canonical=True)
        frags = split_safe_fragments(safe_full)
        core_frag = max(frags, key=len) if frags else safe_full
        return f"{core_frag}." if trailing_dot and not core_frag.endswith(".") else core_frag

    rw = Chem.RWMol(mol)
    # Determine existing ring digits cleanly before assigning attachment labels.
    base_smi = Chem.MolToSmiles(rw, canonical=True, isomericSmiles=True)
    max_existing = _max_ring_digit(base_smi)
    start_digit = max(
        max_existing + 1, int(attachment_start) if attachment_start is not None else 1
    )

    # Order dummy atoms deterministically (by user isotope label if present, else atom idx).
    ordered_dummies = sorted(
        dummy_indices,
        key=lambda idx: (
            0 if rw.GetAtomWithIdx(idx).GetIsotope() > 0 else 1,
            rw.GetAtomWithIdx(idx).GetIsotope(),
            idx,
        ),
    )
    assigned_digits: list[int] = []
    for offset, atom_idx in enumerate(ordered_dummies):
        ring_digit = start_digit + offset
        rw.GetAtomWithIdx(atom_idx).SetIsotope(ring_digit)
        rw.GetAtomWithIdx(atom_idx).SetAtomMapNum(0)
        assigned_digits.append(ring_digit)

    root_idx = heavy_indices[0]
    core_smi = Chem.MolToSmiles(
        rw.GetMol(),
        isomericSmiles=True,
        canonical=True,
        rootedAtAtom=root_idx,
    )
    prefix = _replace_dummy_atoms_with_ring_digits(core_smi)

    # Sanity-check that capping each open attachment with a methyl group yields a valid molecule.
    caps = ".".join(f"C{_format_ring_digit(d)}" for d in assigned_digits)
    if Chem.MolFromSmiles(f"{prefix}.{caps}") is None:
        raise ValueError(
            f"Constructed SAFE scaffold prefix {prefix!r} from {core_smiles_with_dummy!r} "
            "failed RDKit cap validation."
        )

    return f"{prefix}." if trailing_dot else prefix


def extract_core_safe_prefix(
    smiles: str,
    keep_n_frags: int = 1,
    *,
    slicer: str = "brics",
    trailing_dot: bool = True,
) -> str:
    """Encode ``smiles`` to SAFE and keep the first ``keep_n_frags`` blocks as a prefix."""
    if keep_n_frags < 1:
        raise ValueError(f"keep_n_frags must be >= 1, got {keep_n_frags}")
    safe_str = smiles_to_safe(smiles, slicer=slicer, canonical=True)
    frags = split_safe_fragments(safe_str)
    kept = frags[: min(keep_n_frags, len(frags))]
    prefix = ".".join(kept)
    if trailing_dot and len(kept) < len(frags):
        return f"{prefix}."
    return prefix


__all__ = [
    "SAFE_SPLITTER_REGEX",
    "build_scaffold_prefix",
    "extract_core_safe_prefix",
    "is_valid_safe",
    "safe_to_smiles",
    "smiles_to_safe",
    "split_safe_fragments",
]
