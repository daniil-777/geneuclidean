"""Atom-level SMILES, SELFIES, and SAFE molecular tokenization with JSON vocabularies."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

PAD_TOKEN: str = "<pad>"
BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"
UNK_TOKEN: str = "<unk>"

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
UNK_ID: int = 3

SPECIAL_TOKENS: tuple[str, str, str, str] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

# Canonical Schwaller atom-level SMILES regex (preserves Cl, Br, [nH], %10, \ and /)
SCHWALLER_REGEX: re.Pattern[str] = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

RepresentationType = Literal["smiles", "selfies", "safe"]


def canonicalize_smiles(smiles: str, *, strip_salts: bool = True) -> str | None:
    """Validate and canonicalize a SMILES string using RDKit.

    Args:
        smiles: Input SMILES string.
        strip_salts: When ``True`` and multiple disconnected fragments are present,
            retain the largest fragment by heavy-atom count.

    Returns:
        Canonical isomeric SMILES string, or ``None`` if sanitization fails.
    """
    cleaned = smiles.strip()
    if not cleaned:
        return None
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    if strip_salts and "." in cleaned:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        mol = max(frags, key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()))
    try:
        can = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    except Exception:
        return None
    return can if can else None


def randomize_smiles(
    smiles: str,
    rng: random.Random | np.random.Generator | None = None,
) -> str:
    """Generate a randomized equivalent SMILES traversal (Arus-Pous et al., JCIM 2019).

    When ``rng`` is provided, atom indices are permuted deterministically before writing a
    non-canonical SMILES string so training and unit tests remain 100% reproducible. When
    ``rng`` is ``None``, RDKit's native ``doRandom=True`` traversal is used.

    Args:
        smiles: Input SMILES string.
        rng: Optional deterministic RNG (Python ``random.Random`` or NumPy ``Generator``).

    Returns:
        Randomized SMILES string, falling back to canonical/input SMILES on failure.
    """
    cleaned = smiles.strip()
    if not cleaned:
        return cleaned
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return cleaned

    n_atoms = int(mol.GetNumAtoms())
    if n_atoms <= 1:
        return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

    try:
        if rng is None:
            out = str(Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True))
            return out if out else str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

        atom_order = list(range(n_atoms))
        if isinstance(rng, np.random.Generator):
            perm = [int(i) for i in rng.permutation(n_atoms).tolist()]
            root_idx = int(rng.integers(0, n_atoms))
        else:
            rng.shuffle(atom_order)
            perm = atom_order
            root_idx = int(rng.randrange(n_atoms))
        renum = Chem.RenumberAtoms(mol, perm)
        out = str(
            Chem.MolToSmiles(
                renum,
                canonical=False,
                isomericSmiles=True,
                rootedAtAtom=root_idx,
            )
        )
        if out and Chem.MolFromSmiles(out) is not None:
            return out
    except Exception:
        pass
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


class MolecularTokenizer:
    """Atom-level tokenizer supporting SMILES, SELFIES, and SAFE representations.

    Special token IDs are fixed across all instances and vocabularies:
    ``PAD_ID = 0``, ``BOS_ID = 1``, ``EOS_ID = 2``, ``UNK_ID = 3``.
    """

    PAD_ID: int = PAD_ID
    BOS_ID: int = BOS_ID
    EOS_ID: int = EOS_ID
    UNK_ID: int = UNK_ID

    def __init__(
        self,
        representation: RepresentationType = "smiles",
        vocab: Sequence[str] | None = None,
    ) -> None:
        if representation not in ("smiles", "selfies", "safe"):
            raise ValueError(f"Unsupported representation: {representation!r}")
        self.representation: RepresentationType = representation
        self._vocab: list[str] = list(SPECIAL_TOKENS)
        if vocab is not None:
            if len(vocab) < 4 or tuple(vocab[:4]) != SPECIAL_TOKENS:
                raise ValueError(
                    f"Vocabulary must start with fixed special tokens {SPECIAL_TOKENS}."
                )
            self._vocab = list(vocab)
        self._token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(self._vocab)}

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including special tokens."""
        return len(self._vocab)

    @property
    def tokens(self) -> list[str]:
        """Ordered vocabulary list indexed by token ID."""
        return list(self._vocab)

    @property
    def token_to_id(self) -> dict[str, int]:
        """Mapping from token string to integer ID."""
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        """Mapping from integer ID to token string."""
        return {idx: tok for idx, tok in enumerate(self._vocab)}

    def smiles_to_representation(
        self,
        smiles: str,
        *,
        randomize: bool = False,
        rng: random.Random | np.random.Generator | None = None,
    ) -> str:
        """Convert an input SMILES string into the tokenizer's target string representation."""
        cleaned = smiles.strip()
        if not cleaned:
            return ""
        work_smiles = randomize_smiles(cleaned, rng=rng) if randomize else cleaned

        if self.representation == "smiles":
            return work_smiles

        if self.representation == "selfies":
            try:
                encoded = sf.encoder(work_smiles)
                if isinstance(encoded, str) and encoded:
                    return encoded
            except Exception:
                pass
            can = canonicalize_smiles(cleaned, strip_salts=False) or cleaned
            try:
                encoded_fb = sf.encoder(can)
                return str(encoded_fb) if encoded_fb else ""
            except Exception:
                return ""

        # representation == "safe"
        import safe as safe_lib

        try:
            safe_str = safe_lib.encode(work_smiles, canonical=not randomize, ignore_stereo=False)
            if isinstance(safe_str, str) and safe_str:
                return safe_str
        except Exception:
            pass
        # Single-fragment small molecules without BRICS bonds are already valid SAFE strings
        return canonicalize_smiles(cleaned, strip_salts=False) or work_smiles

    def tokenize_string(self, text: str) -> list[str]:
        """Split a representation string (SMILES, SELFIES, or SAFE) into atomic tokens."""
        if not text:
            return []
        if self.representation == "selfies":
            return [str(tok) for tok in sf.split_selfies(text)]
        return [str(tok) for tok in SCHWALLER_REGEX.findall(text)]

    def fit(
        self,
        smiles_iterable: Iterable[str],
        *,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        randomized_passes: int = 0,
        seed: int = 42,
    ) -> MolecularTokenizer:
        """Fit the vocabulary deterministically on an iterable of SMILES strings.

        Args:
            smiles_iterable: Source molecules as SMILES strings.
            min_freq: Minimum occurrence frequency to include a token.
            max_vocab_size: Optional cap on total vocabulary size (including 4 special tokens).
            randomized_passes: Number of extra randomized SMILES passes during fitting to
                prevent OOV tokens during dynamic randomized-SMILES training.
            seed: Seed used for any randomized passes during fitting.

        Returns:
            Self, for fluent chaining.
        """
        counts: Counter[str] = Counter()
        rng = random.Random(seed)
        smiles_list = [s.strip() for s in smiles_iterable if s and s.strip()]

        for smi in smiles_list:
            rep = self.smiles_to_representation(smi, randomize=False)
            if rep:
                counts.update(self.tokenize_string(rep))

        for _ in range(max(0, randomized_passes)):
            for smi in smiles_list:
                rep = self.smiles_to_representation(smi, randomize=True, rng=rng)
                if rep:
                    counts.update(self.tokenize_string(rep))

        filtered = [
            (tok, freq)
            for tok, freq in counts.items()
            if freq >= min_freq and tok not in SPECIAL_TOKENS
        ]
        # Deterministic ordering: descending frequency, then lexicographical
        filtered.sort(key=lambda pair: (-pair[1], pair[0]))
        if max_vocab_size is not None:
            limit = max(0, max_vocab_size - len(SPECIAL_TOKENS))
            filtered = filtered[:limit]

        self._vocab = list(SPECIAL_TOKENS) + [tok for tok, _ in filtered]
        self._token_to_id = {tok: idx for idx, tok in enumerate(self._vocab)}
        return self

    def encode(
        self,
        smiles: str,
        *,
        randomize: bool = False,
        max_len: int = 128,
        rng: random.Random | np.random.Generator | None = None,
    ) -> list[int]:
        """Encode a SMILES string into ``[BOS_ID, ..., EOS_ID]`` with length ``<= max_len``."""
        if max_len < 2:
            raise ValueError(f"max_len must be >= 2 to hold BOS and EOS, got {max_len}")
        rep = self.smiles_to_representation(smiles, randomize=randomize, rng=rng)
        tokens = self.tokenize_string(rep)
        max_inner = max_len - 2
        if len(tokens) > max_inner:
            tokens = tokens[:max_inner]
        ids = [self._token_to_id.get(tok, UNK_ID) for tok in tokens]
        return [BOS_ID, *ids, EOS_ID]

    def decode(self, token_ids: Sequence[int], *, to_smiles: bool = True) -> str:
        """Decode integer token IDs back into a molecular string.

        Strips ``<bos>`` tokens, halts immediately at the first ``<eos>`` or ``<pad>``,
        and converts SELFIES or SAFE strings back into SMILES when ``to_smiles=True``.
        """
        pieces: list[str] = []
        for raw_id in token_ids:
            tid = int(raw_id)
            if tid == BOS_ID:
                continue
            if tid in (EOS_ID, PAD_ID):
                break
            if tid == UNK_ID:
                continue
            if 0 <= tid < len(self._vocab):
                pieces.append(self._vocab[tid])

        raw_str = "".join(pieces)
        if not to_smiles or not raw_str:
            return raw_str

        if self.representation == "smiles":
            return raw_str

        if self.representation == "selfies":
            try:
                decoded = sf.decoder(raw_str)
                return str(decoded) if decoded else ""
            except Exception:
                return ""

        # representation == "safe"
        import safe as safe_lib

        try:
            decoded_safe = safe_lib.decode(raw_str, canonical=False, ignore_errors=True)
            if isinstance(decoded_safe, str) and decoded_safe:
                return decoded_safe
        except Exception:
            pass
        return raw_str

    def save(self, path: str | Path) -> Path:
        """Persist tokenizer representation and vocabulary cleanly as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "representation": self.representation,
            "vocab": self._vocab,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> MolecularTokenizer:
        """Load a tokenizer from a JSON vocabulary file."""
        in_path = Path(path)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        rep = payload["representation"]
        vocab = payload["vocab"]
        if rep not in ("smiles", "selfies", "safe"):
            raise ValueError(f"Invalid representation in {in_path}: {rep!r}")
        return cls(representation=rep, vocab=vocab)


__all__ = [
    "BOS_ID",
    "BOS_TOKEN",
    "EOS_ID",
    "EOS_TOKEN",
    "PAD_ID",
    "PAD_TOKEN",
    "SCHWALLER_REGEX",
    "SPECIAL_TOKENS",
    "UNK_ID",
    "UNK_TOKEN",
    "MolecularTokenizer",
    "RepresentationType",
    "canonicalize_smiles",
    "randomize_smiles",
]
