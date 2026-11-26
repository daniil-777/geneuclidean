"""First-principles Chemical NLP & Multimodal Pocket-to-Ligand Operations Library.

Pure-PyTorch ``float32`` building blocks designed to run natively on Apple Silicon MPS
(``PYTORCH_ENABLE_MPS_FALLBACK=0``), CUDA, and CPU with zero ``float64`` buffers and zero
external framework dependencies (no ``transformers``, ``peft``, ``rdkit``, or ``safe-mol``
required).

Organized into three composable layers matching :mod:`geneuclidean.geom`:

1. **Chemical Tokenization & Grammar Constraints:**
   - :class:`AtomLevelSMILESTokenizer` — fast IUPAC/bracket regex tokenizer supporting
     SMILES, SELFIES, and dot-separated SAFE fragments with fixed special tokens
     (``<pad>``, ``<bos>``, ``<eos>``, ``<mask>``, ``<sep>``, ``<unk>``).
   - :class:`SMILESValenceGrammarMask` — stateful incremental logits mask enforcing
     balanced parentheses, valid open/closed ring closures (``1..9``, ``%10..%99``),
     chemical valence bounds, and forbidding consecutive invalid bonds (``==``, ``(#``).
2. **Transformer & Multimodal 3D-Pocket Architecture Primitives:**
   - :class:`RMSNorm` — ``float32`` variance-stabilized RMS normalization with optional
     embedding-scale initialization.
   - :class:`RotaryPositionEmbedding` & :func:`apply_rotary_pos_emb` — standard,
     NTK-aware, and YaRN-scaled RoPE preserving exact relative inner-product invariance.
   - :class:`SwiGLUFeedForward` — gated SiLU feed-forward network.
   - :class:`KVCache`, :class:`LayerKVCache`, & :class:`GroupedQueryAttention` — GQA /
     MQA / MHA via ``F.scaled_dot_product_attention`` with incremental KV-caching
     matching full-sequence forward to ``< 1e-5``.
   - :class:`PocketCrossAttention` & :class:`PocketPerceiverBridge` — exact zero-leak
     cross-attention and permutation-invariant Perceiver resampling over variable-length
     3D pocket atom tokens ``[B, N, D_in] -> [B, Q, D_out]``.
   - :class:`ChemicalTransformerBlock` & :class:`PocketConditionedChemicalLM` —
     end-to-end trainable autoregressive or bidirectional pocket-conditioned chemical LM.
   - :class:`LoRALinear`, :func:`inject_lora_adapters`, & :func:`merge_lora_adapters` —
     pure-PyTorch rank-:math:`r` LoRA with exact weight-merge equivalence ``< 1e-6``.
3. **Decoding, Contrastive Pocket Steering, Molecular RL Alignment & Masked Diffusion:**
   - :func:`apply_temperature_top_k_top_p` — numerically stable temperature, top-k,
     nucleus (top-p), and min-p logits filtering.
   - :func:`apply_contrastive_cfg_logits` — multi-conformation on-target ensemble +
     off-target repulsion Classifier-Free Guidance logits combination.
   - :func:`beam_search_decode` — deterministic length-normalized beam search with
     optional :class:`SMILESValenceGrammarMask` and :class:`KVCache`.
   - :func:`sequence_logprobs_from_logits`, :func:`dpo_preference_loss` (DPO / cDPO /
     IPO), & :func:`grpo_policy_loss` (DeepSeek/REINVENT-style critic-free GRPO).
   - :class:`DiscreteMaskedDiffusionStep` — absorbing ``<mask>`` forward corruption,
     cosine/linear schedules, weighted MDLM loss, and reverse unmasking / SAFE inpainting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. Chemical Tokenization & Stateful SMILES Valence Grammar Constraints
# ---------------------------------------------------------------------------

PAD_TOKEN: str = "<pad>"
BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"
MASK_TOKEN: str = "<mask>"
SEP_TOKEN: str = "<sep>"
UNK_TOKEN: str = "<unk>"

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
MASK_ID: int = 3
SEP_ID: int = 4
UNK_ID: int = 5

SPECIAL_TOKENS: tuple[str, str, str, str, str, str] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    SEP_TOKEN,
    UNK_TOKEN,
)

# Unified fast IUPAC / Schwaller / SAFE / SELFIES atom-level tokenizer regex:
# - `[...]` matches bracketed stereocenters/ions (`[C@H]`, `[nH]`, `[N+]`) AND SELFIES tokens
# - `Br?|Cl?` keeps two-character halogens indivisible
# - `%\([0-9]{1,5}\)` and `%[0-9]{2}` keep SAFE & macrocycle multi-digit ring closures atomic
# - `.` preserves SAFE fragment boundaries and salt delimiters
CHEMICAL_TOKEN_REGEX: re.Pattern[str] = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|"
    r"\%\([0-9]{1,5}\)|\%[0-9]{2}|[0-9])"
)

DEFAULT_CHEMICAL_VOCAB: tuple[str, ...] = (
    *SPECIAL_TOKENS,
    # Organic & aromatic atoms
    "C",
    "c",
    "N",
    "n",
    "O",
    "o",
    "S",
    "s",
    "P",
    "p",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "b",
    # Branches, SAFE fragment dot, bonds, stereochemistry, reaction arrows
    "(",
    ")",
    ".",
    "=",
    "#",
    "-",
    "+",
    "/",
    "\\",
    ":",
    "~",
    "@",
    "*",
    ">",
    ">>",
    # Ring closure indices (single-digit and common SAFE two-digit closures)
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "%10",
    "%11",
    "%12",
    "%13",
    "%14",
    "%15",
    "%16",
    "%17",
    "%18",
    "%19",
    # Common medicinal-chemistry bracket atoms (chiral, aromatic NH, physiological ions)
    "[C@H]",
    "[C@@H]",
    "[C@]",
    "[C@@]",
    "[nH]",
    "[n+]",
    "[nH+]",
    "[N+]",
    "[NH+]",
    "[NH2+]",
    "[NH3+]",
    "[N-]",
    "[NH-]",
    "[O-]",
    "[OH]",
    "[S+]",
    "[S-]",
    "[S@]",
    "[S@@]",
    "[P+]",
    "[P@]",
    "[P@@]",
    "[B-]",
    "[Si]",
    "[Se]",
    "[se]",
    "[H]",
    "[2H]",
    "[Na+]",
    "[K+]",
    "[Li+]",
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[F-]",
    "[CH]",
    "[CH2]",
    "[CH3]",
    "[NH]",
    "[NH2]",
    "[cH-]",
    "[o+]",
    "[s+]",
    "[*]",
)

_BOND_TOKENS: frozenset[str] = frozenset({"=", "#", "-", "/", "\\", ":"})
_BOND_ORDER: dict[str, int] = {"=": 2, "#": 3, "-": 1, "/": 1, "\\": 1, ":": 1}
_AROMATIC_ATOMS: frozenset[str] = frozenset({"b", "c", "n", "o", "p", "s"})
_MONOVALENT_HALOGENS: frozenset[str] = frozenset({"F", "Cl", "Br", "I"})
_ORGANIC_ATOMS: frozenset[str] = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "b", "c", "n", "o", "p", "s", "*"}
)
# Conservative maximum explicit bond-order budgets for neutral unbracketed SMILES atoms.
# Aromatic atoms (`c, n, o, s, p, b`), hypervalent `P`/`S`, and bracketed atoms `[...]`
# are left uncapped (None) to avoid false rejections on Kekule/fused/charged drug systems.
_MAX_EXPLICIT_VALENCE: dict[str, int | None] = {
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "O": 2,
    "B": 4,
    "C": 4,
    "N": 5,
    "S": 6,
    "P": 5,
}


def _is_ring_token(tok: str) -> bool:
    """Return ``True`` if ``tok`` is a SMILES/SAFE ring-closure index token."""
    if len(tok) == 1 and tok.isdigit():
        return True
    if tok.startswith("%") and len(tok) >= 3:
        inner = tok[2:-1] if tok.startswith("%(") and tok.endswith(")") else tok[1:]
        return bool(inner) and inner.isdigit()
    return False


def _is_atom_token(tok: str) -> bool:
    """Return ``True`` if ``tok`` represents an organic, aromatic, dummy, or bracket atom."""
    if tok in _ORGANIC_ATOMS:
        return True
    return len(tok) >= 3 and tok.startswith("[") and tok.endswith("]")


def _is_special_token(tok: str) -> bool:
    """Return ``True`` if ``tok`` is a structural special token (``<...>``)."""
    return tok in SPECIAL_TOKENS


def _max_valence_for_atom(tok: str) -> int | None:
    """Return conservative maximum explicit bond order for ``tok`` (``None`` if unconstrained)."""
    return _MAX_EXPLICIT_VALENCE.get(tok)


class AtomLevelSMILESTokenizer:
    """Fast deterministic IUPAC/bracket atom-level tokenizer for SMILES, SELFIES, and SAFE.

    Splits SMILES, SELFIES (``[...]`` tokens), and dot-separated SAFE strings into atomic
    tokens without RDKit or HuggingFace dependencies, preserving fixed special token IDs:

    - ``<pad>`` = 0
    - ``<bos>`` = 1
    - ``<eos>`` = 2
    - ``<mask>`` = 3
    - ``<sep>`` = 4
    - ``<unk>`` = 5

    Args:
        vocab: Optional explicit vocabulary sequence. Must begin with :data:`SPECIAL_TOKENS`.
            When ``None``, initializes with :data:`DEFAULT_CHEMICAL_VOCAB`.
        extra_tokens: Optional additional tokens appended deterministically after the base
            vocabulary (duplicates ignored).
    """

    PAD_TOKEN: str = PAD_TOKEN
    BOS_TOKEN: str = BOS_TOKEN
    EOS_TOKEN: str = EOS_TOKEN
    MASK_TOKEN: str = MASK_TOKEN
    SEP_TOKEN: str = SEP_TOKEN
    UNK_TOKEN: str = UNK_TOKEN

    PAD_ID: int = PAD_ID
    BOS_ID: int = BOS_ID
    EOS_ID: int = EOS_ID
    MASK_ID: int = MASK_ID
    SEP_ID: int = SEP_ID
    UNK_ID: int = UNK_ID

    def __init__(
        self,
        vocab: Sequence[str] | None = None,
        extra_tokens: Sequence[str] | None = None,
    ) -> None:
        if vocab is None:
            base_vocab = list(DEFAULT_CHEMICAL_VOCAB)
        else:
            if len(vocab) < len(SPECIAL_TOKENS) or tuple(vocab[: len(SPECIAL_TOKENS)]) != (
                SPECIAL_TOKENS
            ):
                raise ValueError(
                    f"vocab must start with fixed special tokens {SPECIAL_TOKENS}, "
                    f"got {tuple(vocab[: len(SPECIAL_TOKENS)])!r}"
                )
            base_vocab = list(dict.fromkeys(vocab))

        if extra_tokens is not None:
            seen = set(base_vocab)
            for tok in extra_tokens:
                if tok not in seen:
                    base_vocab.append(tok)
                    seen.add(tok)

        self._vocab: list[str] = base_vocab
        self._token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(self._vocab)}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        """Token ID for ``<pad>``."""
        return self.PAD_ID

    @property
    def bos_id(self) -> int:
        """Token ID for ``<bos>``."""
        return self.BOS_ID

    @property
    def eos_id(self) -> int:
        """Token ID for ``<eos>``."""
        return self.EOS_ID

    @property
    def mask_id(self) -> int:
        """Token ID for ``<mask>``."""
        return self.MASK_ID

    @property
    def sep_id(self) -> int:
        """Token ID for ``<sep>``."""
        return self.SEP_ID

    @property
    def unk_id(self) -> int:
        """Token ID for ``<unk>``."""
        return self.UNK_ID

    @property
    def tokens(self) -> list[str]:
        """Ordered list of vocabulary strings indexed by token ID."""
        return list(self._vocab)

    @property
    def token_to_id(self) -> dict[str, int]:
        """Copy of the ``token -> id`` lookup table."""
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        """Copy of the ``id -> token`` lookup table."""
        return {idx: tok for idx, tok in enumerate(self._vocab)}

    def add_tokens(self, tokens: Iterable[str]) -> int:
        """Deterministically register new tokens and return the number added."""
        added = 0
        for tok in tokens:
            if tok and tok not in self._token_to_id:
                self._token_to_id[tok] = len(self._vocab)
                self._vocab.append(tok)
                added += 1
        return added

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a SMILES, SELFIES, or SAFE string into atomic chemical tokens.

        Recognizes inline special tokens (``<pad>``, ``<bos>``, ``<eos>``, ``<mask>``,
        ``<sep>``, ``<unk>``) as well as all regex-matched chemical tokens.
        """
        cleaned = text.strip()
        if not cleaned:
            return []
        tokens: list[str] = []
        pos = 0
        n = len(cleaned)
        while pos < n:
            if cleaned[pos].isspace():
                pos += 1
                continue
            if cleaned[pos] == "<":
                end = cleaned.find(">", pos)
                if end != -1:
                    candidate = cleaned[pos : end + 1]
                    if candidate in self._token_to_id:
                        tokens.append(candidate)
                        pos = end + 1
                        continue
            match = CHEMICAL_TOKEN_REGEX.match(cleaned, pos)
            if match is not None:
                tokens.append(match.group(0))
                pos = match.end()
            else:
                tokens.append(cleaned[pos])
                pos += 1
        return tokens

    def fit(
        self,
        texts: Iterable[str],
        *,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        include_default_vocab: bool = True,
    ) -> AtomLevelSMILESTokenizer:
        """Fit or extend the vocabulary deterministically on SMILES / SELFIES / SAFE texts.

        Args:
            texts: Iterable of molecular strings.
            min_freq: Minimum token frequency required to add a corpus token.
            max_vocab_size: Optional cap on total vocabulary size (must be
                ``>= len(SPECIAL_TOKENS)`` when provided).
            include_default_vocab: If ``True``, retain :data:`DEFAULT_CHEMICAL_VOCAB`
                before appending frequent corpus tokens.

        Returns:
            ``self`` for fluent chaining.
        """
        if min_freq < 1:
            raise ValueError(f"min_freq must be >= 1, got {min_freq}")
        if max_vocab_size is not None and max_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(
                f"max_vocab_size must be >= {len(SPECIAL_TOKENS)}, got {max_vocab_size}"
            )

        counts: Counter[str] = Counter()
        for text in texts:
            if text and text.strip():
                counts.update(self.tokenize(text))

        base: list[str] = (
            list(DEFAULT_CHEMICAL_VOCAB) if include_default_vocab else list(SPECIAL_TOKENS)
        )
        base_set = set(base)
        extra = [
            (tok, freq) for tok, freq in counts.items() if freq >= min_freq and tok not in base_set
        ]
        # Deterministic ordering: descending frequency, then lexicographical tie-break.
        extra.sort(key=lambda pair: (-pair[1], pair[0]))

        full_vocab = base + [tok for tok, _ in extra]
        if max_vocab_size is not None:
            full_vocab = full_vocab[:max_vocab_size]

        self._vocab = full_vocab
        self._token_to_id = {tok: idx for idx, tok in enumerate(self._vocab)}
        return self

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        """Encode a molecular string into integer token IDs.

        Args:
            text: Input SMILES, SELFIES, or SAFE string.
            add_bos: Prepend ``<bos>`` when ``True``.
            add_eos: Append ``<eos>`` when ``True``.
            max_len: Optional maximum total length (truncates inner content tokens while
                preserving requested ``<bos>`` and ``<eos>`` wrappers).

        Returns:
            Deterministic list of integer token IDs.
        """
        n_special = int(add_bos) + int(add_eos)
        if max_len is not None and max_len < n_special:
            raise ValueError(
                f"max_len ({max_len}) must be >= number of wrapper special tokens ({n_special})"
            )

        toks = self.tokenize(text)
        if max_len is not None:
            toks = toks[: max_len - n_special]

        ids = [self._token_to_id.get(tok, self.UNK_ID) for tok in toks]
        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        """Decode integer token IDs back into a molecular string.

        Args:
            token_ids: 1D sequence or tensor of integer token IDs.
            skip_special_tokens: Omit ``<pad>``, ``<bos>``, ``<eos>``, ``<mask>``,
                ``<sep>``, and ``<unk>`` from the reconstructed string.
            stop_at_eos: Halt decoding at the first ``<eos>`` token (always stops at
                trailing ``<pad>`` when ``skip_special_tokens=True``).

        Returns:
            Reconstructed SMILES, SELFIES, or SAFE string.
        """
        raw_ids: list[int] = (
            [int(x) for x in token_ids.detach().cpu().reshape(-1).tolist()]
            if isinstance(token_ids, Tensor)
            else [int(x) for x in token_ids]
        )
        pieces: list[str] = []
        vocab_len = len(self._vocab)
        for tid in raw_ids:
            if stop_at_eos and tid == self.EOS_ID:
                break
            if tid == self.PAD_ID and skip_special_tokens:
                continue
            if skip_special_tokens and tid in (
                self.BOS_ID,
                self.EOS_ID,
                self.MASK_ID,
                self.SEP_ID,
                self.UNK_ID,
            ):
                continue
            if 0 <= tid < vocab_len:
                pieces.append(self._vocab[tid])
        return "".join(pieces)

    def batch_encode(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
        pad_to_multiple_of: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode and right-pad a batch of molecular strings into dense PyTorch tensors.

        Args:
            texts: Non-empty sequence of SMILES, SELFIES, or SAFE strings.
            add_bos: Prepend ``<bos>`` to every sequence.
            add_eos: Append ``<eos>`` to every sequence.
            max_len: Optional maximum sequence length.
            pad_to_multiple_of: Optional alignment multiple for padded length (e.g. ``8``).
            device: Optional target PyTorch device.

        Returns:
            Tuple ``(input_ids [B, L] torch.long, attention_mask [B, L] torch.bool)``
            where ``attention_mask`` is ``True`` at real tokens and ``False`` at ``<pad>``.
        """
        if not texts:
            raise ValueError("texts must be a non-empty sequence.")
        if pad_to_multiple_of is not None and pad_to_multiple_of < 1:
            raise ValueError(f"pad_to_multiple_of must be >= 1, got {pad_to_multiple_of}")

        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, max_len=max_len) for text in texts
        ]
        seq_len = max(len(row) for row in encoded)
        if pad_to_multiple_of is not None and seq_len % pad_to_multiple_of != 0:
            seq_len = ((seq_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * (
                pad_to_multiple_of
            )
        seq_len = max(seq_len, 1)

        batch_size = len(encoded)
        ids = torch.full((batch_size, seq_len), self.PAD_ID, dtype=torch.long)
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, row in enumerate(encoded):
            if row:
                ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                attn_mask[i, : len(row)] = True

        if device is not None:
            dev = torch.device(device)
            ids = ids.to(dev)
            attn_mask = attn_mask.to(dev)
        return ids, attn_mask

    def batch_decode(
        self,
        token_ids: Tensor | Sequence[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> list[str]:
        """Decode a 2D batch of token IDs into a list of molecular strings."""
        rows: list[list[int]] = (
            token_ids.detach().cpu().tolist()
            if isinstance(token_ids, Tensor)
            else [[int(x) for x in r] for r in token_ids]
        )
        return [
            self.decode(row, skip_special_tokens=skip_special_tokens, stop_at_eos=stop_at_eos)
            for row in rows
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize tokenizer vocabulary to a JSON-compatible dictionary."""
        return {"vocab": list(self._vocab)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AtomLevelSMILESTokenizer:
        """Restore a tokenizer from a serialized dictionary produced by :meth:`to_dict`."""
        vocab = payload.get("vocab")
        if not isinstance(vocab, Sequence) or isinstance(vocab, (str, bytes)):
            raise ValueError("Serialized tokenizer payload must contain a 'vocab' sequence.")
        return cls(vocab=[str(t) for t in vocab])


@dataclass
class _AtomValenceFrame:
    """Mutable per-atom state on the SMILES branch stack."""

    atom_tok: str
    used_valence: int = 0
    rings_on_atom: set[str] = field(default_factory=set)

    def copy(self) -> _AtomValenceFrame:
        return _AtomValenceFrame(
            atom_tok=self.atom_tok,
            used_valence=self.used_valence,
            rings_on_atom=set(self.rings_on_atom),
        )


@dataclass
class _SMILESGrammarState:
    """Incremental syntactic & valence parser state for one SMILES / SAFE sequence."""

    started: bool = False
    fragment_started: bool = False
    finished: bool = False
    paren_depth: int = 0
    open_rings: set[str] = field(default_factory=set)
    last_tok: str = ""
    last_kind: str = "start"  # start | atom | bond | open_paren | close_paren | ring | dot
    pending_bond_order: int = 0
    current_atom: _AtomValenceFrame | None = None
    branch_stack: list[_AtomValenceFrame] = field(default_factory=list)
    awaiting_branch_first_atom: bool = False

    def copy(self) -> _SMILESGrammarState:
        return _SMILESGrammarState(
            started=self.started,
            fragment_started=self.fragment_started,
            finished=self.finished,
            paren_depth=self.paren_depth,
            open_rings=set(self.open_rings),
            last_tok=self.last_tok,
            last_kind=self.last_kind,
            pending_bond_order=self.pending_bond_order,
            current_atom=self.current_atom.copy() if self.current_atom is not None else None,
            branch_stack=[frame.copy() for frame in self.branch_stack],
            awaiting_branch_first_atom=self.awaiting_branch_first_atom,
        )


class SMILESValenceGrammarMask:
    """Stateful incremental logits mask enforcing SMILES/SAFE syntax and valence rules.

    Masks out syntactically or chemically impossible next tokens during autoregressive
    sampling or beam search while guaranteeing exact pass-through of valid drug SMILES
    (including carbonyl branches ``(=O)``, fused multi-ring closures ``c34``, stereocenters,
    and SAFE dot-separated fragments):

    - Strictly enforces balanced parentheses (blocks unmatched ``)``, empty ``()``,
      and premature ``<eos>`` while branches remain open).
    - Tracks open ring closures (``1..9``, ``%10..%99``): blocks leading ring digits,
      same-atom self-loops (``C11``), premature ``<eos>`` with unclosed rings, and
      exceeding ``max_open_rings``.
    - Forbids consecutive bond tokens (``==``, ``=#``, ``##``, ``//``), dangling bonds
      before ``)`` / ``.`` / ``<eos>``, and illegal branch starts such as ``(#``.
    - Tracks explicit valence budgets on neutral organic atoms (blocking e.g. ``=F``,
      ``Cl=``, ``O#``, ``#O``, or triple bonds on aromatic atoms).

    Args:
        tokenizer: :class:`AtomLevelSMILESTokenizer` defining the vocabulary.
        max_paren_depth: Maximum permitted nesting depth for ``(...)`` branches.
        max_open_rings: Maximum simultaneously open ring-closure indices.
        allow_safe_fragments: Whether dot-separated SAFE fragments (``.``) are allowed.
        enforce_valence: Whether to enforce conservative neutral-atom valence bounds.
    """

    def __init__(
        self,
        tokenizer: AtomLevelSMILESTokenizer,
        *,
        max_paren_depth: int = 16,
        max_open_rings: int = 9,
        allow_safe_fragments: bool = True,
        enforce_valence: bool = True,
    ) -> None:
        if max_paren_depth < 1:
            raise ValueError(f"max_paren_depth must be >= 1, got {max_paren_depth}")
        if max_open_rings < 1:
            raise ValueError(f"max_open_rings must be >= 1, got {max_open_rings}")

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_paren_depth = int(max_paren_depth)
        self.max_open_rings = int(max_open_rings)
        self.allow_safe_fragments = bool(allow_safe_fragments)
        self.enforce_valence = bool(enforce_valence)

        self._tokens: list[str] = tokenizer.tokens
        self._id_to_token: dict[int, str] = tokenizer.id_to_token
        self._states: list[_SMILESGrammarState] = [_SMILESGrammarState()]

        # Pre-classify vocabulary token indices once for O(V) fast bitmask construction.
        self._atom_ids: list[int] = []
        self._bond_ids: list[int] = []
        self._ring_ids: list[int] = []
        self._open_paren_id: int | None = None
        self._close_paren_id: int | None = None
        self._dot_id: int | None = None

        for idx, tok in enumerate(self._tokens):
            if _is_special_token(tok):
                continue
            if tok == "(":
                self._open_paren_id = idx
            elif tok == ")":
                self._close_paren_id = idx
            elif tok == ".":
                self._dot_id = idx
            elif tok in _BOND_TOKENS:
                self._bond_ids.append(idx)
            elif _is_ring_token(tok):
                self._ring_ids.append(idx)
            elif _is_atom_token(tok):
                self._atom_ids.append(idx)

    @property
    def batch_size(self) -> int:
        """Number of tracked parallel sequence states."""
        return len(self._states)

    def reset(self, batch_size: int = 1) -> None:
        """Reset all tracked parser states for a new batch of ``batch_size`` sequences."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._states = [_SMILESGrammarState() for _ in range(batch_size)]

    def clone_states(self, beam_indices: Sequence[int]) -> None:
        """Reorder and clone internal parser states along ``beam_indices`` (for beam search)."""
        self._states = [self._states[int(i)].copy() for i in beam_indices]

    def _advance_state(self, state: _SMILESGrammarState, token_id: int) -> None:
        """Advance one sequence state in-place by ``token_id``."""
        if state.finished:
            return
        if token_id == self.tokenizer.EOS_ID:
            state.finished = True
            return
        if token_id in (
            self.tokenizer.PAD_ID,
            self.tokenizer.BOS_ID,
            self.tokenizer.MASK_ID,
            self.tokenizer.SEP_ID,
            self.tokenizer.UNK_ID,
        ):
            return

        tok = self._id_to_token.get(token_id, "")
        if not tok:
            return

        state.last_tok = tok
        if tok == "(":
            if state.current_atom is not None:
                state.branch_stack.append(state.current_atom)
            state.paren_depth += 1
            state.last_kind = "open_paren"
            state.awaiting_branch_first_atom = True
            state.pending_bond_order = 0
            return

        if tok == ")":
            state.paren_depth = max(0, state.paren_depth - 1)
            if state.branch_stack:
                state.current_atom = state.branch_stack.pop()
            state.last_kind = "close_paren"
            state.awaiting_branch_first_atom = False
            state.pending_bond_order = 0
            return

        if tok == ".":
            state.fragment_started = False
            state.current_atom = None
            state.last_kind = "dot"
            state.pending_bond_order = 0
            state.awaiting_branch_first_atom = False
            return

        if tok in _BOND_TOKENS:
            order = _BOND_ORDER[tok]
            if state.awaiting_branch_first_atom:
                if state.branch_stack:
                    state.branch_stack[-1].used_valence += order
            elif state.current_atom is not None:
                state.current_atom.used_valence += order
            state.pending_bond_order = order
            state.last_kind = "bond"
            return

        if _is_ring_token(tok):
            if state.pending_bond_order == 0 and state.current_atom is not None:
                state.current_atom.used_valence += 1
            if state.current_atom is not None:
                state.current_atom.rings_on_atom.add(tok)
            if tok in state.open_rings:
                state.open_rings.remove(tok)
            else:
                state.open_rings.add(tok)
            state.pending_bond_order = 0
            state.last_kind = "ring"
            return

        if _is_atom_token(tok):
            if not state.fragment_started:
                new_frame = _AtomValenceFrame(atom_tok=tok, used_valence=0)
            else:
                bond_order = state.pending_bond_order if state.pending_bond_order > 0 else 1
                if state.pending_bond_order == 0:
                    parent = (
                        state.branch_stack[-1]
                        if state.awaiting_branch_first_atom and state.branch_stack
                        else state.current_atom
                    )
                    if parent is not None:
                        parent.used_valence += bond_order
                new_frame = _AtomValenceFrame(atom_tok=tok, used_valence=bond_order)
            state.current_atom = new_frame
            state.started = True
            state.fragment_started = True
            state.pending_bond_order = 0
            state.awaiting_branch_first_atom = False
            state.last_kind = "atom"

    def step(self, token_ids: Sequence[int] | Tensor) -> None:
        """Advance stateful parser by one emitted token per sequence in the batch."""
        ids: list[int] = (
            [int(x) for x in token_ids.detach().cpu().reshape(-1).tolist()]
            if isinstance(token_ids, Tensor)
            else [int(x) for x in token_ids]
        )
        if len(ids) != len(self._states):
            raise ValueError(
                f"Expected {len(self._states)} token IDs matching current batch_size, "
                f"got {len(ids)}. Call reset(batch_size) first."
            )
        for state, tid in zip(self._states, ids, strict=True):
            self._advance_state(state, tid)

    def _state_from_history(self, history_row: Sequence[int]) -> _SMILESGrammarState:
        """Reconstruct parser state deterministically from a token ID history."""
        state = _SMILESGrammarState()
        for tid in history_row:
            self._advance_state(state, int(tid))
        return state

    def _active_attachment_atom(self, state: _SMILESGrammarState) -> _AtomValenceFrame | None:
        """Return the atom to which the next bond/atom will attach."""
        if state.awaiting_branch_first_atom:
            return state.branch_stack[-1] if state.branch_stack else None
        return state.current_atom

    def _remaining_valence(self, frame: _AtomValenceFrame | None) -> int | None:
        """Return remaining explicit bond-order budget for ``frame``, or ``None`` if uncapped."""
        if frame is None or not self.enforce_valence:
            return None
        cap = _max_valence_for_atom(frame.atom_tok)
        if cap is None:
            return None
        return max(0, cap - frame.used_valence)

    def _allowed_vector_for_state(self, state: _SMILESGrammarState) -> list[bool]:
        """Compute exact boolean mask ``allowed[V]`` for a single parser state."""
        allowed = [False] * self.vocab_size

        if state.finished:
            allowed[self.tokenizer.PAD_ID] = True
            allowed[self.tokenizer.EOS_ID] = True
            return allowed

        attach_atom = self._active_attachment_atom(state)
        rem_val = self._remaining_valence(attach_atom)

        # 1. EOS: only when molecule is non-empty, parens & rings balanced, no dangling bond/dot.
        if (
            state.started
            and state.fragment_started
            and state.paren_depth == 0
            and len(state.open_rings) == 0
            and state.last_kind not in {"bond", "open_paren", "dot"}
        ):
            allowed[self.tokenizer.EOS_ID] = True

        # 2. Close parenthesis ')': only when paren_depth > 0 and branch is non-empty & non-bond.
        if (
            self._close_paren_id is not None
            and state.paren_depth > 0
            and state.last_kind not in {"open_paren", "bond", "dot"}
        ):
            allowed[self._close_paren_id] = True

        # 3. Open parenthesis '(': only after an atom/ring/closed-branch with available valence.
        if (
            self._open_paren_id is not None
            and state.fragment_started
            and state.paren_depth < self.max_paren_depth
            and state.last_kind not in {"start", "bond", "open_paren", "dot"}
            and (rem_val is None or rem_val >= 1)
        ):
            allowed[self._open_paren_id] = True

        # 4. Fragment dot '.': SAFE / salt delimiter outside branches after a completed atom.
        if (
            self._dot_id is not None
            and self.allow_safe_fragments
            and state.fragment_started
            and state.paren_depth == 0
            and state.last_kind not in {"start", "bond", "open_paren", "dot"}
        ):
            allowed[self._dot_id] = True

        # 5. Bond tokens: never at start, never consecutive (`==`, `=#`), forbid `(#` and `(:`.
        if state.fragment_started and state.last_kind not in {"start", "bond", "dot"}:
            attach_tok = attach_atom.atom_tok if attach_atom is not None else ""
            for bid in self._bond_ids:
                btok = self._tokens[bid]
                order = _BOND_ORDER[btok]
                # Explicitly forbid `(#` and `(:` immediately after opening a branch.
                if state.last_kind == "open_paren" and btok in ("#", ":"):
                    continue
                if self.enforce_valence:
                    if btok == "#" and attach_tok in _AROMATIC_ATOMS:
                        continue
                    if btok == "#" and attach_tok in _MONOVALENT_HALOGENS:
                        continue
                    if btok == "#" and attach_tok in ("O", "B"):
                        continue
                    if btok == "=" and attach_tok in _MONOVALENT_HALOGENS:
                        continue
                    if rem_val is not None and rem_val < order:
                        continue
                allowed[bid] = True

        # 6. Ring closure digits: only after an atom/ring/bond (never after `(`, `.`, or start).
        if state.fragment_started and state.last_kind not in {"start", "open_paren", "dot"}:
            rings_here = (
                state.current_atom.rings_on_atom if state.current_atom is not None else set()
            )
            ring_valence_ok = (
                True if state.last_kind == "bond" else (rem_val is None or rem_val >= 1)
            )
            if ring_valence_ok:
                at_ring_cap = len(state.open_rings) >= self.max_open_rings
                for rid in self._ring_ids:
                    rtok = self._tokens[rid]
                    # Forbid same-atom self-loop (e.g. `C11`).
                    if rtok in rings_here:
                        continue
                    if at_ring_cap and rtok not in state.open_rings:
                        continue
                    allowed[rid] = True

        # 7. Atom tokens: allowed at start/after dot, after bond/open_paren, or when valence >= 1.
        can_chain_atom = (
            (not state.fragment_started)
            or (state.last_kind in {"bond", "open_paren", "dot"})
            or (rem_val is None or rem_val >= 1)
        )
        if can_chain_atom:
            pending = state.pending_bond_order
            for aid in self._atom_ids:
                atok = self._tokens[aid]
                if self.enforce_valence and state.last_kind == "bond":
                    if pending >= 2 and atok in _MONOVALENT_HALOGENS:
                        continue
                    if pending >= 3 and (atok in _AROMATIC_ATOMS or atok in ("O", "B")):
                        continue
                allowed[aid] = True

        return allowed

    def is_complete(self) -> list[bool]:
        """Return whether each tracked sequence can legally emit ``<eos>`` right now."""
        return [
            bool(
                st.started
                and st.fragment_started
                and st.paren_depth == 0
                and len(st.open_rings) == 0
                and st.last_kind not in {"bond", "open_paren", "dot"}
            )
            for st in self._states
        ]

    def valid_token_mask(
        self,
        *,
        batch_size: int | None = None,
        token_history: Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return boolean tensor ``[B, V]`` where ``True`` marks grammar-valid next tokens."""
        if token_history is not None:
            hist = token_history.unsqueeze(0) if token_history.ndim == 1 else token_history
            states = [self._state_from_history(row) for row in hist.detach().cpu().tolist()]
        else:
            if batch_size is not None and len(self._states) != batch_size:
                self.reset(batch_size)
            states = self._states

        rows = [self._allowed_vector_for_state(st) for st in states]
        return torch.tensor(rows, dtype=torch.bool, device=torch.device(device))

    def mask_logits(
        self,
        logits: Tensor,
        *,
        token_history: Tensor | None = None,
        filter_value: float = float("-inf"),
    ) -> Tensor:
        """Apply grammar & valence constraints to next-token ``logits`` (``[V]`` or ``[B, V]``).

        Positions violating SMILES syntax or valence rules are set to ``filter_value``
        (default ``-inf``). If every finite candidate in a row would be masked out, that row
        is left unchanged so downstream softmax never produces ``NaN``.
        """
        if logits.ndim not in (1, 2):
            raise ValueError(f"Expected 1D [V] or 2D [B, V] logits, got {tuple(logits.shape)}")
        squeeze_1d = logits.ndim == 1
        work = logits.unsqueeze(0).to(torch.float32) if squeeze_1d else logits.to(torch.float32)
        bsz, vocab_dim = work.shape
        if vocab_dim != self.vocab_size:
            raise ValueError(
                f"logits vocab dim ({vocab_dim}) != tokenizer vocab_size ({self.vocab_size})"
            )

        allow_mask = self.valid_token_mask(
            batch_size=bsz,
            token_history=token_history,
            device=work.device,
        )
        masked = work.masked_fill(~allow_mask, float(filter_value))
        has_finite = torch.isfinite(masked).any(dim=-1, keepdim=True)
        out = torch.where(has_finite, masked, work)
        return out.squeeze(0) if squeeze_1d else out

    def validate_sequence(self, text_or_ids: str | Sequence[int]) -> bool:
        """Check whether every token step of ``text_or_ids`` (including ``<eos>``) is allowed."""
        ids = (
            self.tokenizer.encode(text_or_ids, add_bos=True, add_eos=True)
            if isinstance(text_or_ids, str)
            else [int(x) for x in text_or_ids]
        )
        if not ids:
            return False
        state = _SMILESGrammarState()
        for tid in ids:
            if tid == self.tokenizer.BOS_ID:
                continue
            allowed = self._allowed_vector_for_state(state)
            if not (0 <= tid < self.vocab_size and allowed[tid]):
                return False
            self._advance_state(state, tid)
        return state.finished


# ---------------------------------------------------------------------------
# 2. Champion Transformer & Multimodal 3D-Pocket Architecture Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with exact ``float32`` variance stabilization.

    Computes ``(x / sqrt(mean(x^2) + eps)) * weight`` strictly in ``torch.float32`` before
    casting back to the input dtype, preventing underflow/overflow on Apple MPS and mixed
    precision accelerators.

    Args:
        dim: Feature dimension normalized along the last axis.
        eps: Variance stabilization floor.
        init_scale: Initial scale for the learnable ``weight`` vector (use ``!= 1.0`` for
            ScaleMatchRMSNorm projection into pretrained LLM embedding manifolds).
    """

    def __init__(self, dim: int, eps: float = 1e-6, init_scale: float = 1.0) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0, got {init_scale}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.init_scale = float(init_scale)
        self.weight = nn.Parameter(torch.full((self.dim,), self.init_scale, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization along the trailing dimension."""
        x_fp32 = x.to(dtype=torch.float32)
        inv_rms = torch.rsqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 * inv_rms) * self.weight
        return out.to(dtype=x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate paired feature halves ``(x1, x2) -> (-x2, x1)`` for RoPE."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors ``[B, H, L, Dh]``.

    Preserves exact relative-position inner-product invariance:
    :math:`\\langle R(m + \\Delta) q, R(n + \\Delta) k \\rangle =
    \\langle R(m) q, R(n) k \\rangle` for any shift :math:`\\Delta`.

    Supports Grouped-Query Attention (``H_q != H_kv``) via broadcasting along dim 1.

    Args:
        q: Query tensor of shape ``[B, H_q, L, Dh]``.
        k: Key tensor of shape ``[B, H_kv, L, Dh]``.
        cos: Cosine table broadcastable to ``[B, 1, L, Dh]``.
        sin: Sine table broadcastable to ``[B, 1, L, Dh]``.

    Returns:
        Tuple ``(q_rot, k_rot)`` matching the shapes and dtypes of ``(q, k)``.
    """
    cos_f = cos.to(device=q.device, dtype=torch.float32)
    sin_f = sin.to(device=q.device, dtype=torch.float32)
    q_f = q.to(dtype=torch.float32)
    k_f = k.to(dtype=torch.float32)
    q_rot = (q_f * cos_f) + (_rotate_half(q_f) * sin_f)
    k_rot = (k_f * cos_f) + (_rotate_half(k_f) * sin_f)
    return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with NTK-aware and YaRN context-extension scaling.

    All frequency buffers and trigonometric tables are computed strictly in ``torch.float32``.

    Args:
        head_dim: Per-head attention dimension (must be even and ``>= 2``).
        max_seq_len: Pre-cached sequence length. Sequences longer than ``max_seq_len`` are
            computed dynamically on the fly without error.
        base: Base geometric frequency (default ``10000.0``).
        scaling_type: Context-extension scaling mode: ``"none"`` (standard RoPE),
            ``"ntk"`` (NTK-aware base scaling), or ``"yarn"`` (Peng et al., ICLR 2024).
        scaling_factor: Context extension multiplier ``s >= 1.0`` (used when
            ``scaling_type`` is ``"ntk"`` or ``"yarn"``).
        original_max_seq_len: Original pretraining context length for YaRN frequency ramp.
        yarn_beta_fast: High-frequency cutoff rotation count for YaRN (default ``32.0``).
        yarn_beta_slow: Low-frequency cutoff rotation count for YaRN (default ``1.0``).
        yarn_mscale: Attention temperature compensation factor for YaRN (default ``1.0``).
    """

    inv_freq: Tensor
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
        *,
        scaling_type: Literal["none", "ntk", "yarn"] = "none",
        scaling_factor: float = 1.0,
        original_max_seq_len: int = 512,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_mscale: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be an even integer >= 2, got {head_dim}")
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
        if base <= 1.0:
            raise ValueError(f"base must be > 1.0, got {base}")
        if scaling_type not in ("none", "ntk", "yarn"):
            raise ValueError(f"scaling_type must be 'none', 'ntk', or 'yarn', got {scaling_type!r}")
        if scaling_factor < 1.0:
            raise ValueError(f"scaling_factor must be >= 1.0, got {scaling_factor}")
        if yarn_beta_fast <= yarn_beta_slow:
            raise ValueError("yarn_beta_fast must be strictly greater than yarn_beta_slow")

        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        self.scaling_type: Literal["none", "ntk", "yarn"] = scaling_type
        self.scaling_factor = float(scaling_factor)
        self.original_max_seq_len = int(original_max_seq_len)
        self.yarn_beta_fast = float(yarn_beta_fast)
        self.yarn_beta_slow = float(yarn_beta_slow)
        self.yarn_mscale = float(yarn_mscale)

        inv_freq, mscale = self._build_inv_freq()
        self.attention_mscale: float = float(mscale)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos_tbl, sin_tbl = self._compute_tables_from_positions(
            torch.arange(self.max_seq_len, dtype=torch.float32),
            inv_freq,
            self.attention_mscale,
        )
        self.register_buffer("cos_cached", cos_tbl, persistent=False)
        self.register_buffer("sin_cached", sin_tbl, persistent=False)

    def _build_inv_freq(self) -> tuple[Tensor, float]:
        """Compute fp32 inverse frequencies and attention magnitude scale."""
        dim_idx = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        if self.scaling_type == "none" or self.scaling_factor <= 1.0:
            inv_freq = 1.0 / (self.base ** (dim_idx / float(self.head_dim)))
            return inv_freq, 1.0

        if self.scaling_type == "ntk":
            exponent = float(self.head_dim) / float(self.head_dim - 2) if self.head_dim > 2 else 1.0
            scaled_base = self.base * (self.scaling_factor**exponent)
            inv_freq = 1.0 / (scaled_base ** (dim_idx / float(self.head_dim)))
            return inv_freq, 1.0

        # YaRN (Peng et al., ICLR 2024): frequency-band interpolation + mscale compensation.
        base_inv = 1.0 / (self.base ** (dim_idx / float(self.head_dim)))
        wavelengths = (2.0 * math.pi) / base_inv
        rotations = float(self.original_max_seq_len) / wavelengths
        ramp = (
            (rotations - self.yarn_beta_slow) / (self.yarn_beta_fast - self.yarn_beta_slow)
        ).clamp(0.0, 1.0)
        inv_freq = (1.0 - ramp) * (base_inv / self.scaling_factor) + ramp * base_inv
        mscale = 0.1 * self.yarn_mscale * math.log(self.scaling_factor) + 1.0
        return inv_freq, mscale

    @staticmethod
    def _compute_tables_from_positions(
        positions: Tensor,
        inv_freq: Tensor,
        mscale: float,
    ) -> tuple[Tensor, Tensor]:
        """Compute ``(cos, sin)`` tables for 1D ``[L]`` or 2D ``[B, L]`` positions."""
        pos_f = positions.to(device=inv_freq.device, dtype=torch.float32)
        if pos_f.ndim == 1:
            freqs = torch.outer(pos_f, inv_freq)  # [L, Dh // 2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [L, Dh]
            cos = emb.cos()[None, None, :, :] * mscale
            sin = emb.sin()[None, None, :, :] * mscale
            return cos, sin
        if pos_f.ndim == 2:
            freqs = pos_f.unsqueeze(-1) * inv_freq[None, None, :]  # [B, L, Dh // 2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, Dh]
            cos = emb.cos().unsqueeze(1) * mscale
            sin = emb.sin().unsqueeze(1) * mscale
            return cos, sin
        raise ValueError(f"Expected 1D or 2D position_ids, got shape {tuple(positions.shape)}")

    def forward(
        self,
        seq_len: int,
        *,
        offset: int = 0,
        position_ids: Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables of shape ``[1, 1, seq_len, Dh]`` or ``[B, 1, L, Dh]``."""
        target_dev = device if device is not None else self.inv_freq.device
        if position_ids is not None:
            return self._compute_tables_from_positions(
                position_ids.to(device=target_dev),
                self.inv_freq.to(device=target_dev),
                self.attention_mscale,
            )

        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        end = offset + seq_len
        if end <= self.cos_cached.shape[2]:
            return (
                self.cos_cached[:, :, offset:end, :].to(device=target_dev),
                self.sin_cached[:, :, offset:end, :].to(device=target_dev),
            )
        positions = torch.arange(offset, end, device=target_dev, dtype=torch.float32)
        return self._compute_tables_from_positions(
            positions,
            self.inv_freq.to(device=target_dev),
            self.attention_mscale,
        )


class SwiGLUFeedForward(nn.Module):
    """SwiGLU gated feed-forward network (Shazeer, 2020) in ``float32``.

    Computes ``w_down(dropout(silu(w_gate(x)) * w_up(x)))``.

    Args:
        d_model: Input and output hidden dimension.
        d_ff: Intermediate hidden dimension. Defaults to ``ceil_32(8 * d_model / 3)``.
        dropout: Dropout probability applied after the SwiGLU gate.
        bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if d_ff is None:
            raw_ff = max(d_model, int(d_model * 8 // 3))
            d_ff = ((raw_ff + 31) // 32) * 32
        if d_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {d_ff}")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.w_gate = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.w_up = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU transformation preserving tensor shape ``[..., d_model]``."""
        hidden = F.silu(self.w_gate(x)) * self.w_up(x)
        out: Tensor = self.w_down(self.dropout(hidden))
        return out


@dataclass
class LayerKVCache:
    """Single-layer Key/Value cache tensors ``[B, H_kv, T, Dh]`` for incremental decoding."""

    key: Tensor
    value: Tensor

    @property
    def seq_len(self) -> int:
        """Current cached sequence length ``T``."""
        return int(self.key.shape[2])

    @property
    def batch_size(self) -> int:
        """Cached batch size ``B``."""
        return int(self.key.shape[0])

    def reorder(self, beam_indices: Tensor) -> LayerKVCache:
        """Gather cached keys/values along the batch dimension for beam search."""
        idx = beam_indices.to(device=self.key.device, dtype=torch.long)
        return LayerKVCache(
            key=self.key.index_select(0, idx).contiguous(),
            value=self.value.index_select(0, idx).contiguous(),
        )

    def clone(self) -> LayerKVCache:
        """Return a deep copy of this layer's KV cache."""
        return LayerKVCache(key=self.key.clone(), value=self.value.clone())


class KVCache:
    """Multi-layer Key/Value cache container for incremental autoregressive decoding.

    Args:
        num_layers: Number of Transformer blocks tracked by this cache.
    """

    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = int(num_layers)
        self._layers: list[LayerKVCache | None] = [None] * self.num_layers

    @property
    def seq_len(self) -> int:
        """Return cached sequence length from the first populated layer (``0`` if empty)."""
        for entry in self._layers:
            if entry is not None:
                return entry.seq_len
        return 0

    def get(self, layer_idx: int) -> LayerKVCache | None:
        """Retrieve the :class:`LayerKVCache` for ``layer_idx`` (or ``None`` if empty)."""
        return self._layers[layer_idx]

    def set(self, layer_idx: int, cache: LayerKVCache) -> None:
        """Store an updated :class:`LayerKVCache` for ``layer_idx``."""
        self._layers[layer_idx] = cache

    def update(self, layer_idx: int, new_k: Tensor, new_v: Tensor) -> LayerKVCache:
        """Append ``(new_k, new_v)`` to ``layer_idx`` and return the updated layer cache."""
        existing = self._layers[layer_idx]
        if existing is None:
            updated = LayerKVCache(key=new_k, value=new_v)
        else:
            updated = LayerKVCache(
                key=torch.cat((existing.key, new_k), dim=2),
                value=torch.cat((existing.value, new_v), dim=2),
            )
        self._layers[layer_idx] = updated
        return updated

    def reorder(self, beam_indices: Tensor) -> KVCache:
        """Return a new :class:`KVCache` reordered along the batch axis by ``beam_indices``."""
        new_cache = KVCache(self.num_layers)
        for idx, entry in enumerate(self._layers):
            if entry is not None:
                new_cache._layers[idx] = entry.reorder(beam_indices)
        return new_cache

    def clone(self) -> KVCache:
        """Return a deep copy of all populated layer caches."""
        new_cache = KVCache(self.num_layers)
        for idx, entry in enumerate(self._layers):
            if entry is not None:
                new_cache._layers[idx] = entry.clone()
        return new_cache

    def reset(self) -> None:
        """Clear all stored layer caches."""
        self._layers = [None] * self.num_layers


def _repeat_kv_heads(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads ``[B, H_kv, T, Dh] -> [B, H_kv * n_rep, T, Dh]`` without MPS fallbacks."""
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x.unsqueeze(2)
        .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
    )


def _safe_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    allow_mask: Tensor | None,
    is_causal: bool,
    dropout_p: float,
    training: bool,
) -> Tensor:
    """MPS/CUDA/CPU-safe ``F.scaled_dot_product_attention`` with all-masked row protection."""
    eff_dropout = dropout_p if training else 0.0
    # Guard against PyTorch MPS SDPA dropout edge case when grads are disabled.
    use_manual = (
        q.device.type == "mps"
        and eff_dropout > 0.0
        and not (q.requires_grad or k.requires_grad or v.requires_grad)
    )
    if allow_mask is None:
        if use_manual:
            scale = 1.0 / math.sqrt(float(q.shape[-1]))
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if is_causal:
                l_q, l_k = q.shape[2], k.shape[2]
                causal = torch.triu(
                    torch.ones((l_q, l_k), dtype=torch.bool, device=q.device), diagonal=1
                )
                scores = scores.masked_fill(causal, float("-inf"))
            probs = F.dropout(torch.softmax(scores, dim=-1), p=eff_dropout, training=True)
            return torch.matmul(probs, v)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=eff_dropout,
            is_causal=is_causal,
        )

    mask_bool = allow_mask.to(device=q.device, dtype=torch.bool)
    has_valid = mask_bool.any(dim=-1, keepdim=True)
    safe_mask = torch.where(has_valid, mask_bool, torch.ones_like(mask_bool))
    if use_manual:
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~safe_mask, float("-inf"))
        probs = F.dropout(torch.softmax(scores, dim=-1), p=eff_dropout, training=True)
        out = torch.matmul(probs, v)
    else:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=safe_mask,
            dropout_p=eff_dropout,
            is_causal=False,
        )
    return out * has_valid.to(dtype=out.dtype)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query / Multi-Query / Multi-Head Attention with RoPE & explicit KV-Cache.

    Set ``n_kv_heads = n_heads`` for standard MHA, ``1 < n_kv_heads < n_heads`` for GQA,
    and ``n_kv_heads = 1`` for MQA. Uses ``F.scaled_dot_product_attention`` and supports
    both causal autoregressive decoding (with incremental :class:`LayerKVCache` matching
    full-sequence forward to ``< 1e-5``) and bidirectional masked-diffusion attention
    (``is_causal=False``).

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (must divide ``n_heads``; defaults to
            ``n_heads``).
        dropout: Attention & residual dropout probability.
        bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1:
            raise ValueError(f"d_model ({d_model}) and n_heads ({n_heads}) must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        kv_heads = n_heads if n_kv_heads is None else int(n_kv_heads)
        if kv_heads < 1 or n_heads % kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({kv_heads})")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.resid_dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        x: Tensor,
        *,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        rope: RotaryPositionEmbedding | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        is_causal: bool = True,
        position_offset: int | None = None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Run self-attention and return ``(output [B, L, D], updated_layer_cache)``.

        Args:
            x: Input hidden states of shape ``[B, L_q, D]``.
            cos: Optional precomputed RoPE cosine table ``[1|B, 1, L_q, Dh]``.
            sin: Optional precomputed RoPE sine table ``[1|B, 1, L_q, Dh]``.
            rope: Optional :class:`RotaryPositionEmbedding` invoked automatically when
                ``cos``/``sin`` are omitted.
            attention_mask: Optional boolean validity mask of shape ``[B, L_k]`` (``True``
                = attendable token) or 4D broadcastable ``[B, 1|H, L_q, L_k]``.
            kv_cache: Optional previous :class:`LayerKVCache` holding ``[B, H_kv, T_past, Dh]``.
            is_causal: Enforce causal lower-triangular masking (respecting ``T_past``).
            position_offset: Optional explicit RoPE position offset (defaults to
                ``kv_cache.seq_len`` when ``kv_cache`` is provided, else ``0``).

        Returns:
            Tuple ``(out [B, L_q, D], updated_cache)``.
        """
        bsz, l_q, _ = x.shape
        q = self.q_proj(x).view(bsz, l_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, l_q, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, l_q, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = kv_cache.seq_len if kv_cache is not None else 0
        eff_offset = past_len if position_offset is None else int(position_offset)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif rope is not None:
            cos_t, sin_t = rope(l_q, offset=eff_offset, device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos_t, sin_t)

        if kv_cache is not None:
            k_all = torch.cat((kv_cache.key, k), dim=2)
            v_all = torch.cat((kv_cache.value, v), dim=2)
        else:
            k_all = k
            v_all = v
        updated_cache = LayerKVCache(key=k_all, value=v_all)

        k_exp = _repeat_kv_heads(k_all, self.n_rep)
        v_exp = _repeat_kv_heads(v_all, self.n_rep)
        l_k = int(k_all.shape[2])

        # Build exact allow_mask only when needed (chunked prefill or explicit padding mask).
        allow_mask: Tensor | None = None
        use_native_causal = False
        if attention_mask is None:
            if is_causal:
                if past_len == 0:
                    use_native_causal = True
                elif l_q == 1:
                    # Single incremental token attends to all `past_len + 1` keys.
                    use_native_causal = False
                else:
                    q_pos = torch.arange(past_len, past_len + l_q, device=x.device).unsqueeze(1)
                    k_pos = torch.arange(l_k, device=x.device).unsqueeze(0)
                    allow_mask = (k_pos <= q_pos)[None, None, :, :]
        else:
            am = attention_mask.to(device=x.device, dtype=torch.bool)
            if am.ndim == 2:
                if am.shape != (bsz, l_k):
                    raise ValueError(
                        f"2D attention_mask shape {tuple(am.shape)} != expected {(bsz, l_k)}"
                    )
                allow_mask = am[:, None, None, :].expand(bsz, 1, l_q, l_k)
            elif am.ndim == 4:
                allow_mask = am
            else:
                raise ValueError(
                    f"Expected 2D [B, L_k] or 4D [B, 1|H, L_q, L_k] attention_mask, "
                    f"got {tuple(am.shape)}"
                )
            if is_causal:
                q_pos = torch.arange(past_len, past_len + l_q, device=x.device).unsqueeze(1)
                k_pos = torch.arange(l_k, device=x.device).unsqueeze(0)
                causal_allow = (k_pos <= q_pos)[None, None, :, :]
                allow_mask = allow_mask & causal_allow

        attn_out = _safe_sdpa(
            q,
            k_exp,
            v_exp,
            allow_mask=allow_mask,
            is_causal=use_native_causal,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        merged = attn_out.transpose(1, 2).contiguous().view(bsz, l_q, self.d_model)
        out: Tensor = self.resid_dropout(self.o_proj(merged))
        return out, updated_cache


def _resolve_valid_pocket_mask(
    pocket_tokens: Tensor,
    *,
    key_padding_mask: Tensor | None,
    pocket_mask: Tensor | None,
) -> Tensor | None:
    """Resolve ``key_padding_mask`` (True=PAD) or ``pocket_mask`` (True=VALID) to bool mask."""
    if key_padding_mask is not None and pocket_mask is not None:
        raise ValueError(
            "Pass at most one of key_padding_mask (True=PAD) or pocket_mask (True=VALID)."
        )
    if key_padding_mask is not None:
        kpm = key_padding_mask.to(device=pocket_tokens.device, dtype=torch.bool)
        if kpm.shape != pocket_tokens.shape[:2]:
            raise ValueError(
                f"key_padding_mask shape {tuple(kpm.shape)} != pocket[:2] "
                f"{tuple(pocket_tokens.shape[:2])}"
            )
        return ~kpm
    if pocket_mask is not None:
        pm = pocket_mask.to(device=pocket_tokens.device, dtype=torch.bool)
        if pm.shape != pocket_tokens.shape[:2]:
            raise ValueError(
                f"pocket_mask shape {tuple(pm.shape)} != pocket[:2] "
                f"{tuple(pocket_tokens.shape[:2])}"
            )
        return pm
    return None


class PocketCrossAttention(nn.Module):
    """Cross-attention conditioning ligand queries on 3D pocket/sub-pocket tokens.

    Directly conditions ligand sequence queries ``[B, L, d_model]`` on 3D pocket tokens
    ``[B, N, d_pocket]`` produced by ``CartFormer-SE3``, ``AtomGS-SE3``,
    ``SubPocketVN-JEPA``, or :class:`PocketPerceiverBridge`.

    Guarantees exact ``0.0`` leakage from corrupted/padded pocket slots and finite zero
    outputs (zero ``NaN``) even when an entire batch row has all pocket tokens masked.

    Args:
        d_model: Query and output feature dimension.
        d_pocket: Input 3D pocket token dimension (defaults to ``d_model``).
        n_heads: Number of query attention heads.
        n_kv_heads: Number of pocket key/value heads (defaults to ``n_heads``).
        dropout: Attention and residual dropout probability.
        gated: If ``True``, multiply output by a learnable Flamingo-style ``tanh(gate)``.
        gate_init: Initial value of the Flamingo cross-attention gate parameter.
        bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        d_model: int,
        d_pocket: int | None = None,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        *,
        gated: bool = False,
        gate_init: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1:
            raise ValueError(f"d_model ({d_model}) and n_heads ({n_heads}) must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        kv_heads = n_heads if n_kv_heads is None else int(n_kv_heads)
        if kv_heads < 1 or n_heads % kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({kv_heads})")

        self.d_model = int(d_model)
        self.d_pocket = self.d_model if d_pocket is None else int(d_pocket)
        self.n_heads = int(n_heads)
        self.n_kv_heads = kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.dropout_p = float(dropout)
        self.gated = bool(gated)

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.d_pocket, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.d_pocket, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.resid_dropout = nn.Dropout(self.dropout_p)

        if self.gated:
            self.gate: nn.Parameter | None = nn.Parameter(
                torch.tensor(float(gate_init), dtype=torch.float32)
            )
        else:
            self.gate = None

    @overload
    def forward(
        self,
        query: Tensor,
        pocket_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        return_attn: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        query: Tensor,
        pocket_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        return_attn: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        query: Tensor,
        pocket_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Attend from ligand ``query[B, L, d_model]`` into ``pocket_tokens[B, N, d_pocket]``.

        Args:
            query: Ligand token queries of shape ``[B, L, d_model]``.
            pocket_tokens: 3D pocket/sub-pocket embeddings of shape ``[B, N, d_pocket]``.
            key_padding_mask: Optional PyTorch-convention boolean mask ``[B, N]`` where
                ``True`` marks PADDED (ignored) pocket slots.
            pocket_mask: Optional GenEuclidean-convention boolean mask ``[B, N]`` where
                ``True`` marks VALID pocket atoms. Mutually exclusive with
                ``key_padding_mask``.
            return_attn: If ``True``, also return exact cross-attention weights
                ``[B, H_q, L, N]`` with strict ``0.0`` on padded pocket atoms.

        Returns:
            ``output [B, L, d_model]`` (or ``(output, attn_weights)`` when
            ``return_attn=True``).
        """
        bsz, l_q, _ = query.shape
        n_mem = int(pocket_tokens.shape[1])
        valid_mask = _resolve_valid_pocket_mask(
            pocket_tokens, key_padding_mask=key_padding_mask, pocket_mask=pocket_mask
        )

        # Handle empty pocket sequence cleanly with zero output and preserved autograd graph.
        if n_mem == 0:
            zero_out = query * 0.0
            if return_attn:
                return zero_out, torch.zeros(
                    (bsz, self.n_heads, l_q, 0), device=query.device, dtype=torch.float32
                )
            return zero_out

        # Zero corrupted padded pocket slots prior to projection so dummy values never leak.
        if valid_mask is not None:
            safe_pocket = torch.where(
                valid_mask.unsqueeze(-1),
                pocket_tokens,
                torch.zeros_like(pocket_tokens),
            )
        else:
            safe_pocket = pocket_tokens

        q = self.q_proj(query).view(bsz, l_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(safe_pocket)
            .view(bsz, n_mem, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(safe_pocket)
            .view(bsz, n_mem, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_exp = _repeat_kv_heads(k, self.n_rep)
        v_exp = _repeat_kv_heads(v, self.n_rep)

        attn_weights: Tensor | None = None
        if return_attn:
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale  # [B, H, L, N]
            if valid_mask is not None:
                key_allow = valid_mask[:, None, None, :]  # [B, 1, 1, N]
                has_valid = valid_mask.any(dim=-1)[:, None, None, None]  # [B, 1, 1, 1]
                safe_allow = torch.where(has_valid, key_allow, torch.ones_like(key_allow))
                scores = scores.masked_fill(~safe_allow, float("-inf"))
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = attn_weights * key_allow.to(dtype=attn_weights.dtype)
                attn_weights = attn_weights * has_valid.to(dtype=attn_weights.dtype)
            else:
                attn_weights = torch.softmax(scores, dim=-1)
            probs = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
            context = torch.matmul(probs, v_exp)
        else:
            allow_4d = valid_mask[:, None, None, :] if valid_mask is not None else None
            context = _safe_sdpa(
                q,
                k_exp,
                v_exp,
                allow_mask=allow_4d,
                is_causal=False,
                dropout_p=self.dropout_p,
                training=self.training,
            )

        merged = context.transpose(1, 2).contiguous().view(bsz, l_q, self.d_model)
        out: Tensor = self.resid_dropout(self.o_proj(merged))
        if self.gate is not None:
            out = torch.tanh(self.gate) * out

        if return_attn:
            assert attn_weights is not None
            return out, attn_weights
        return out


class _PerceiverResamplerLayer(nn.Module):
    """Single Perceiver layer: pocket cross-attention -> latent self-attention -> SwiGLU."""

    def __init__(
        self,
        d_bridge: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_norm_q = RMSNorm(d_bridge)
        self.cross_norm_kv = RMSNorm(d_bridge)
        self.cross_attn = PocketCrossAttention(
            d_model=d_bridge,
            d_pocket=d_bridge,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.sa_norm = RMSNorm(d_bridge)
        self.self_attn = GroupedQueryAttention(
            d_model=d_bridge,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            dropout=dropout,
        )
        self.ff_norm = RMSNorm(d_bridge)
        self.ffn = SwiGLUFeedForward(d_model=d_bridge, dropout=dropout)

    def forward(
        self,
        latents: Tensor,
        pocket_kv: Tensor,
        *,
        valid_mask: Tensor | None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        q_in = self.cross_norm_q(latents)
        kv_in = self.cross_norm_kv(pocket_kv)
        if valid_mask is not None:
            kv_in = torch.where(valid_mask.unsqueeze(-1), kv_in, torch.zeros_like(kv_in))

        attn_w: Tensor | None = None
        if return_attn:
            cross_out, attn_w = self.cross_attn(
                q_in, kv_in, pocket_mask=valid_mask, return_attn=True
            )
        else:
            cross_out = self.cross_attn(q_in, kv_in, pocket_mask=valid_mask, return_attn=False)
        latents = latents + cross_out

        sa_out, _ = self.self_attn(self.sa_norm(latents), is_causal=False)
        latents = latents + sa_out
        latents = latents + self.ffn(self.ff_norm(latents))
        return latents, attn_w


class PocketPerceiverBridge(nn.Module):
    """Permutation-invariant Perceiver resampler ``[B, N, D_in] -> [B, Q, D_out]``.

    Compresses arbitrary variable-length 3D pocket atom tokens into a fixed budget of ``Q``
    learned query latents with exact zero-leak masking over padded pocket slots, ready for
    soft-prefix conditioning or cross-attention inside a chemical LM.

    Args:
        d_in: Input 3D pocket feature dimension.
        d_out: Output latent token dimension matching the downstream chemical LM.
        num_queries: Fixed number of learned latent queries ``Q``.
        num_layers: Number of Perceiver cross-attn + self-attn + SwiGLU layers.
        n_heads: Number of attention heads in each Perceiver layer.
        d_bridge: Internal Perceiver width (defaults to ``d_out`` when divisible by
            ``n_heads``, else aligned up to the nearest multiple of ``n_heads``).
        dropout: Dropout probability inside Perceiver layers.
        output_rms_scale: Initial scale on the output :class:`RMSNorm` (e.g., match a
            pretrained LLM's token-embedding RMS).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_queries: int = 16,
        num_layers: int = 2,
        n_heads: int = 8,
        d_bridge: int | None = None,
        dropout: float = 0.0,
        output_rms_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if d_in < 1 or d_out < 1:
            raise ValueError(f"d_in ({d_in}) and d_out ({d_out}) must be >= 1")
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")

        if d_bridge is None:
            raw = min(int(d_in), int(d_out))
            aligned = max(n_heads, ((raw + n_heads - 1) // n_heads) * n_heads)
            bridge_dim = aligned
        else:
            bridge_dim = int(d_bridge)
            if bridge_dim % n_heads != 0:
                raise ValueError(
                    f"d_bridge ({bridge_dim}) must be divisible by n_heads ({n_heads})"
                )

        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.n_heads = int(n_heads)
        self.d_bridge = bridge_dim

        self.in_proj: nn.Module = (
            nn.Linear(self.d_in, self.d_bridge, bias=False)
            if self.d_in != self.d_bridge
            else nn.Identity()
        )
        self.latents = nn.Parameter(
            torch.randn(1, self.num_queries, self.d_bridge, dtype=torch.float32) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                _PerceiverResamplerLayer(
                    d_bridge=self.d_bridge, n_heads=self.n_heads, dropout=dropout
                )
                for _ in range(self.num_layers)
            ]
        )
        self.bridge_norm = RMSNorm(self.d_bridge)
        self.out_proj: nn.Module = (
            nn.Linear(self.d_bridge, self.d_out, bias=False)
            if self.d_bridge != self.d_out
            else nn.Identity()
        )
        self.out_norm = RMSNorm(self.d_out, init_scale=output_rms_scale)

    @overload
    def forward(
        self,
        pocket_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        return_attn: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        pocket_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        return_attn: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        pocket_tokens: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compress ``pocket_tokens[B, N, d_in]`` into ``latents[B, Q, d_out]``."""
        if pocket_tokens.ndim != 3 or int(pocket_tokens.shape[-1]) != self.d_in:
            raise ValueError(
                f"Expected pocket_tokens shape [B, N, {self.d_in}], "
                f"got {tuple(pocket_tokens.shape)}"
            )
        valid_mask = _resolve_valid_pocket_mask(
            pocket_tokens, key_padding_mask=key_padding_mask, pocket_mask=pocket_mask
        )

        x_fp32 = pocket_tokens.to(dtype=torch.float32)
        if valid_mask is not None:
            x_fp32 = torch.where(valid_mask.unsqueeze(-1), x_fp32, torch.zeros_like(x_fp32))

        pocket_kv: Tensor = self.in_proj(x_fp32)
        if valid_mask is not None:
            pocket_kv = torch.where(
                valid_mask.unsqueeze(-1), pocket_kv, torch.zeros_like(pocket_kv)
            )

        bsz = int(pocket_tokens.shape[0])
        latents = self.latents.expand(bsz, -1, -1)

        last_attn: Tensor | None = None
        for idx, layer in enumerate(self.layers):
            want_attn = return_attn and (idx == self.num_layers - 1)
            latents, attn = layer(latents, pocket_kv, valid_mask=valid_mask, return_attn=want_attn)
            if want_attn:
                last_attn = attn

        projected: Tensor = self.out_proj(self.bridge_norm(latents))
        out: Tensor = self.out_norm(projected).to(dtype=pocket_tokens.dtype)
        if return_attn:
            assert last_attn is not None
            return out, last_attn
        return out


class ChemicalTransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block with RoPE GQA, optional gated pocket cross-attn, and SwiGLU.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of self- and cross-attention query heads.
        n_kv_heads: Number of key/value heads for GQA (defaults to ``n_heads``).
        d_ff: Intermediate SwiGLU dimension (defaults to ``ceil_32(8 * d_model / 3)``).
        dropout: Attention & FFN dropout probability.
        use_cross_attn: Include a :class:`PocketCrossAttention` sublayer.
        d_pocket: Dimension of 3D pocket cross-attention memory (defaults to ``d_model``).
        gated_cross_attn: Use Flamingo-style ``tanh(gate)`` gating on cross-attention.
        cross_attn_gate_init: Initial cross-attention gate value.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        dropout: float = 0.0,
        *,
        use_cross_attn: bool = True,
        d_pocket: int | None = None,
        gated_cross_attn: bool = True,
        cross_attn_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_cross_attn = bool(use_cross_attn)
        self.d_pocket = self.d_model if d_pocket is None else int(d_pocket)

        self.norm1 = RMSNorm(self.d_model)
        self.self_attn = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )

        if self.use_cross_attn:
            self.cross_norm_q: RMSNorm | None = RMSNorm(self.d_model)
            self.cross_norm_kv: RMSNorm | None = RMSNorm(self.d_pocket)
            self.cross_attn: PocketCrossAttention | None = PocketCrossAttention(
                d_model=self.d_model,
                d_pocket=self.d_pocket,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                gated=gated_cross_attn,
                gate_init=cross_attn_gate_init,
            )
        else:
            self.cross_norm_q = None
            self.cross_norm_kv = None
            self.cross_attn = None

        self.norm2 = RMSNorm(self.d_model)
        self.ffn = SwiGLUFeedForward(d_model=self.d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        *,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        is_causal: bool = True,
        pocket_tokens: Tensor | None = None,
        pocket_key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Run one Transformer block and return ``(hidden [B, L, D], updated_layer_cache)``."""
        sa_out, new_cache = self.self_attn(
            self.norm1(x),
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            is_causal=is_causal,
        )
        x = x + sa_out

        if self.use_cross_attn and pocket_tokens is not None:
            assert (
                self.cross_norm_q is not None
                and self.cross_norm_kv is not None
                and self.cross_attn is not None
            )
            valid_mask = _resolve_valid_pocket_mask(
                pocket_tokens,
                key_padding_mask=pocket_key_padding_mask,
                pocket_mask=pocket_mask,
            )
            kv_norm = self.cross_norm_kv(pocket_tokens)
            if valid_mask is not None:
                kv_norm = torch.where(valid_mask.unsqueeze(-1), kv_norm, torch.zeros_like(kv_norm))
            cross_out = self.cross_attn(
                self.cross_norm_q(x),
                kv_norm,
                pocket_mask=valid_mask,
                return_attn=False,
            )
            x = x + cross_out

        x = x + self.ffn(self.norm2(x))
        return x, new_cache


@dataclass(frozen=True)
class ChemicalLMOutput:
    """Structured output returned by :class:`PocketConditionedChemicalLM`."""

    logits: Tensor
    loss: Tensor | None = None
    kv_cache: KVCache | None = None
    hidden_states: Tensor | None = None


class PocketConditionedChemicalLM(nn.Module):
    """End-to-end trainable 3D pocket-conditioned Chemical Language Model built from pure ops.

    Supports three conditioning modes cleanly inside one architecture:
    - ``"cross_attn"`` (default): Flamingo-style gated cross-attention from ligand tokens
      into raw 3D pocket tokens or :class:`PocketPerceiverBridge`-resampled latents.
    - ``"prefix"``: Perceiver-resampled soft-prefix tokens prepended before token embeddings
      (supporting incremental KV-cached decoding seamlessly).
    - ``"none"``: Unconditional decoder-only / bidirectional chemical LM.

    Works seamlessly for both causal autoregressive generation (``is_causal=True``) and
    bidirectional Masked Diffusion Language Modeling (``is_causal=False``).

    Args:
        vocab_size: Token vocabulary size.
        d_model: Transformer hidden width.
        n_layers: Number of :class:`ChemicalTransformerBlock` layers.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of KV heads for GQA/MQA (defaults to ``n_heads``).
        d_ff: Intermediate SwiGLU dimension (defaults to ``ceil_32(8 * d_model / 3)``).
        max_seq_len: Pre-cached RoPE sequence length.
        dropout: Dropout probability.
        pad_id: Padding token ID (zeroed in embedding table and ignored in CE loss).
        d_pocket: 3D pocket token dimension (defaults to ``d_model``).
        conditioning_mode: ``"cross_attn"``, ``"prefix"``, or ``"none"``.
        use_perceiver_bridge: Compress raw 3D pocket tokens via :class:`PocketPerceiverBridge`
            before conditioning (always enabled when ``conditioning_mode == "prefix"``).
        num_pocket_queries: Number of Perceiver latent queries ``Q`` when bridge is active.
        perceiver_layers: Depth of :class:`PocketPerceiverBridge`.
        gated_cross_attn: Use Flamingo ``tanh(gate)`` in cross-attention blocks.
        rope_base: Base RoPE wavelength parameter.
        rope_scaling_type: ``"none"``, ``"ntk"``, or ``"yarn"``.
        rope_scaling_factor: Context-extension factor for NTK / YaRN RoPE.
        tie_weights: Tie ``lm_head.weight`` to ``token_emb.weight``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        *,
        pad_id: int = PAD_ID,
        d_pocket: int | None = None,
        conditioning_mode: Literal["cross_attn", "prefix", "none"] = "cross_attn",
        use_perceiver_bridge: bool = False,
        num_pocket_queries: int = 16,
        perceiver_layers: int = 2,
        gated_cross_attn: bool = True,
        rope_base: float = 10000.0,
        rope_scaling_type: Literal["none", "ntk", "yarn"] = "none",
        rope_scaling_factor: float = 1.0,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or n_layers < 1:
            raise ValueError(f"vocab_size ({vocab_size}) and n_layers ({n_layers}) must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if conditioning_mode not in ("cross_attn", "prefix", "none"):
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode!r}")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_kv_heads = n_heads if n_kv_heads is None else int(n_kv_heads)
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.d_pocket = self.d_model if d_pocket is None else int(d_pocket)
        self.conditioning_mode: Literal["cross_attn", "prefix", "none"] = conditioning_mode
        self.use_perceiver_bridge = bool(use_perceiver_bridge or conditioning_mode == "prefix")
        self.num_pocket_queries = int(num_pocket_queries)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.emb_dropout = nn.Dropout(float(dropout))
        self.rope = RotaryPositionEmbedding(
            head_dim=self.d_model // self.n_heads,
            max_seq_len=max(self.max_seq_len + self.num_pocket_queries, 256),
            base=rope_base,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
            original_max_seq_len=self.max_seq_len,
        )

        if self.use_perceiver_bridge and self.conditioning_mode != "none":
            self.pocket_bridge: PocketPerceiverBridge | None = PocketPerceiverBridge(
                d_in=self.d_pocket,
                d_out=self.d_model,
                num_queries=self.num_pocket_queries,
                num_layers=perceiver_layers,
                n_heads=self.n_heads,
                dropout=dropout,
            )
            block_pocket_dim = self.d_model
        else:
            self.pocket_bridge = None
            block_pocket_dim = self.d_pocket

        use_block_cross = self.conditioning_mode == "cross_attn"
        self.blocks = nn.ModuleList(
            [
                ChemicalTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_cross_attn=use_block_cross,
                    d_pocket=block_pocket_dim,
                    gated_cross_attn=gated_cross_attn,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self._init_weights()
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def _init_weights(self) -> None:
        """Apply scaled residual Transformer initialization in ``float32``."""
        resid_std = 0.02 / math.sqrt(2.0 * float(self.n_layers))
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = resid_std if name.endswith(("o_proj", "w_down")) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Tensor,
        *,
        pocket_tokens: Tensor | None = None,
        pocket_key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        is_causal: bool = True,
        labels: Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> ChemicalLMOutput:
        """Execute forward pass returning logits, optional loss, and optional updated KV-cache.

        Args:
            input_ids: Token IDs of shape ``[B, L]`` (``torch.long``).
            pocket_tokens: Optional 3D pocket atom embeddings ``[B, N, d_pocket]``.
            pocket_key_padding_mask: Optional boolean mask ``[B, N]`` (``True`` = padded).
            pocket_mask: Optional boolean mask ``[B, N]`` (``True`` = valid).
            attention_mask: Optional token validity mask ``[B, L_total]`` (``True`` = valid).
            kv_cache: Optional :class:`KVCache` from prior decoding steps.
            use_cache: Return updated :class:`KVCache` when ``True``.
            is_causal: Causal masking for autoregressive LMs; set ``False`` for masked
                diffusion (MDLM).
            labels: Optional target token IDs ``[B, L]`` (``-100`` or ``pad_id`` ignored).
                Shifted by 1 token when ``is_causal=True``; aligned when ``is_causal=False``.
            return_hidden_states: Include final normalized hidden states in output.

        Returns:
            :class:`ChemicalLMOutput` with ``logits [B, L, vocab_size]``.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape [B, L], got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        x = self.emb_dropout(self.token_emb(input_ids))
        past_len = kv_cache.seq_len if kv_cache is not None else 0

        cross_memory: Tensor | None = None
        cross_kpm: Tensor | None = pocket_key_padding_mask
        cross_pm: Tensor | None = pocket_mask
        prefix_len = 0

        if pocket_tokens is not None and self.conditioning_mode != "none":
            if self.conditioning_mode == "prefix":
                # Prepend soft prefix only on initial prefill (subsequent steps read from cache).
                if past_len == 0:
                    assert self.pocket_bridge is not None
                    prefix = self.pocket_bridge(
                        pocket_tokens,
                        key_padding_mask=pocket_key_padding_mask,
                        pocket_mask=pocket_mask,
                        return_attn=False,
                    ).to(dtype=x.dtype)
                    prefix_len = int(prefix.shape[1])
                    x = torch.cat((prefix, x), dim=1)
                    if (
                        attention_mask is not None
                        and attention_mask.ndim == 2
                        and int(attention_mask.shape[1]) == seq_len
                    ):
                        pref_mask = torch.ones(
                            (bsz, prefix_len), dtype=torch.bool, device=input_ids.device
                        )
                        attention_mask = torch.cat(
                            (
                                pref_mask,
                                attention_mask.to(device=input_ids.device, dtype=torch.bool),
                            ),
                            dim=1,
                        )
            else:
                if self.pocket_bridge is not None:
                    cross_memory = self.pocket_bridge(
                        pocket_tokens,
                        key_padding_mask=pocket_key_padding_mask,
                        pocket_mask=pocket_mask,
                        return_attn=False,
                    ).to(dtype=x.dtype)
                    cross_kpm = None
                    cross_pm = None
                else:
                    cross_memory = pocket_tokens.to(dtype=x.dtype)

        # On cached prefix decoding steps (past_len > 0), auto-prepend Q valid prefix columns
        # when the caller passes a text-only 2D attention_mask (works whether pocket_tokens is
        # passed again or omitted).
        if (
            self.conditioning_mode == "prefix"
            and past_len > 0
            and attention_mask is not None
            and attention_mask.ndim == 2
            and int(attention_mask.shape[1]) + self.num_pocket_queries == past_len + seq_len
        ):
            pref_mask = torch.ones(
                (bsz, self.num_pocket_queries), dtype=torch.bool, device=input_ids.device
            )
            attention_mask = torch.cat(
                (pref_mask, attention_mask.to(device=input_ids.device, dtype=torch.bool)),
                dim=1,
            )

        total_q_len = int(x.shape[1])
        cos, sin = self.rope(total_q_len, offset=past_len, device=x.device)

        next_cache = KVCache(self.n_layers) if use_cache else None
        for idx, block in enumerate(self.blocks):
            layer_cache = kv_cache.get(idx) if kv_cache is not None else None
            x, updated_layer_cache = block(
                x,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                is_causal=is_causal,
                pocket_tokens=cross_memory,
                pocket_key_padding_mask=cross_kpm,
                pocket_mask=cross_pm,
            )
            if next_cache is not None:
                next_cache.set(idx, updated_layer_cache)

        hidden = self.final_norm(x)
        tok_hidden = hidden[:, prefix_len:, :] if prefix_len > 0 else hidden
        logits: Tensor = self.lm_head(tok_hidden)

        loss: Tensor | None = None
        if labels is not None:
            if is_causal:
                if seq_len < 2:
                    raise ValueError("Causal label loss requires sequence length >= 2.")
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                eff_labels = torch.where(
                    shift_labels == self.pad_id,
                    torch.full_like(shift_labels, -100),
                    shift_labels,
                )
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    eff_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                eff_labels = torch.where(
                    labels == self.pad_id,
                    torch.full_like(labels, -100),
                    labels,
                )
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    eff_labels.view(-1),
                    ignore_index=-100,
                )

        return ChemicalLMOutput(
            logits=logits,
            loss=loss,
            kv_cache=next_cache,
            hidden_states=tok_hidden if return_hidden_states else None,
        )


class LoRALinear(nn.Module):
    """Pure-PyTorch rank-:math:`r` LoRA wrapper around ``nn.Linear`` (zero ``peft`` required).

    Computes ``base_layer(x) + (alpha / rank) * (dropout(x) @ A^T @ B^T)`` when unmerged,
    and folds ``(alpha / rank) * (B @ A)`` exactly into ``base_layer.weight`` upon
    :meth:`merge` with equivalence ``< 1e-6``.

    Args:
        base_layer: Underlying ``nn.Linear`` module.
        rank: Low-rank bottleneck dimension ``r >= 1``.
        alpha: LoRA scaling numerator (effective scale is ``alpha / rank``).
        dropout: Optional input dropout probability applied on the LoRA branch.
        freeze_base: Freeze ``base_layer`` parameters when ``True``.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        *,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.merged = False

        if freeze_base:
            self.base_layer.requires_grad_(False)

        dev = base_layer.weight.device
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(
            torch.empty((self.rank, self.in_features), device=dev, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros((self.out_features, self.rank), device=dev, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> LoRALinear:
        """Construct a :class:`LoRALinear` wrapping an existing ``nn.Linear``."""
        return cls(linear, rank=rank, alpha=alpha, dropout=dropout, freeze_base=freeze_base)

    def _delta_weight(self) -> Tensor:
        """Return effective weight update ``(alpha / rank) * (B @ A)`` in ``float32``."""
        return torch.matmul(self.lora_b, self.lora_a) * self.scaling

    def merge(self) -> LoRALinear:
        """Fold LoRA weights directly into ``base_layer.weight`` for zero-overhead inference."""
        if not self.merged:
            with torch.no_grad():
                delta = self._delta_weight().to(
                    device=self.base_layer.weight.device,
                    dtype=self.base_layer.weight.dtype,
                )
                self.base_layer.weight.add_(delta)
            self.merged = True
        return self

    def unmerge(self) -> LoRALinear:
        """Subtract folded LoRA weights from ``base_layer.weight`` to resume adapter training."""
        if self.merged:
            with torch.no_grad():
                delta = self._delta_weight().to(
                    device=self.base_layer.weight.device,
                    dtype=self.base_layer.weight.dtype,
                )
                self.base_layer.weight.sub_(delta)
            self.merged = False
        return self

    def to_linear(self) -> nn.Linear:
        """Export a standalone merged ``nn.Linear`` layer with zero LoRA overhead."""
        out = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        with torch.no_grad():
            weight = self.base_layer.weight.clone()
            if not self.merged:
                weight.add_(self._delta_weight().to(device=weight.device, dtype=weight.dtype))
            out.weight.copy_(weight)
            if self.base_layer.bias is not None and out.bias is not None:
                out.bias.copy_(self.base_layer.bias)
        return out

    def forward(self, x: Tensor) -> Tensor:
        """Compute base linear projection plus low-rank LoRA adaptation."""
        base_out: Tensor = self.base_layer(x)
        if self.merged:
            return base_out
        dropped = self.lora_dropout(x).to(dtype=torch.float32)
        lora_hidden = F.linear(dropped, self.lora_a)
        lora_out = F.linear(lora_hidden, self.lora_b) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)


def _set_submodule_by_path(root: nn.Module, dotted_path: str, replacement: nn.Module) -> None:
    """Replace nested submodule ``dotted_path`` inside ``root``."""
    parts = dotted_path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def inject_lora_adapters(
    model: nn.Module,
    *,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> list[str]:
    """Inject :class:`LoRALinear` adapters into all matching ``nn.Linear`` layers in ``model``.

    Args:
        model: Target PyTorch module (e.g. :class:`PocketConditionedChemicalLM`).
        target_modules: Leaf attribute names (or dotted suffixes) of ``nn.Linear`` modules
            to adapt.
        rank: LoRA rank ``r``.
        alpha: LoRA scaling numerator.
        dropout: LoRA input dropout.
        freeze_base: Freeze all non-LoRA parameters of ``model`` when ``True``.

    Returns:
        Sorted list of dotted module paths wrapped with :class:`LoRALinear`.
    """
    if not target_modules:
        raise ValueError("target_modules must be a non-empty sequence of layer name suffixes.")
    suffixes = tuple(target_modules)

    if freeze_base:
        model.requires_grad_(False)

    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        # Skip base_layer inside an existing LoRALinear wrapper.
        if name.endswith(".base_layer"):
            continue
        leaf = name.rsplit(".", maxsplit=1)[-1]
        if leaf in suffixes or any(name.endswith(f".{s}") for s in suffixes):
            targets.append((name, module))

    injected: list[str] = []
    for name, lin in targets:
        wrapped = LoRALinear.from_linear(
            lin, rank=rank, alpha=alpha, dropout=dropout, freeze_base=freeze_base
        )
        _set_submodule_by_path(model, name, wrapped)
        injected.append(name)

    return sorted(injected)


def merge_lora_adapters(model: nn.Module, *, unwrap: bool = False) -> list[str]:
    """Merge every :class:`LoRALinear` adapter in ``model`` into its base weight matrix.

    Args:
        model: Module previously adapted via :func:`inject_lora_adapters`.
        unwrap: If ``True``, replace each :class:`LoRALinear` with a plain merged
            ``nn.Linear`` module for zero inference wrapper overhead.

    Returns:
        Sorted list of merged dotted module paths.
    """
    lora_layers: list[tuple[str, LoRALinear]] = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, LoRALinear)
    ]
    merged_names: list[str] = []
    for name, mod in lora_layers:
        if unwrap:
            _set_submodule_by_path(model, name, mod.to_linear())
        else:
            mod.merge()
        merged_names.append(name)
    return sorted(merged_names)


# ---------------------------------------------------------------------------
# 3. Decoding, Contrastive Pocket Steering, Molecular RL Alignment & MDLM
# ---------------------------------------------------------------------------


def apply_temperature_top_k_top_p(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    filter_value: float = float("-inf"),
) -> Tensor:
    """Apply numerically stable temperature scaling, top-k, top-p (nucleus), and min-p filtering.

    Guarantees at least ``min_tokens_to_keep`` finite logits survive per row whenever the
    input row contains finite entries, preventing ``NaN`` probabilities on MPS/CUDA/CPU.

    Args:
        logits: Unnormalized logits tensor of shape ``[..., V]``.
        temperature: Positive temperature divisor (``> 0.0``).
        top_k: Keep only the ``top_k`` highest-logit tokens when ``0 < top_k < V``.
        top_p: Nucleus cumulative probability threshold in ``(0.0, 1.0]``.
        min_p: Min-p threshold in ``[0.0, 1.0]`` keeping tokens with ``p >= min_p * max(p)``.
        min_tokens_to_keep: Minimum surviving tokens per row (``>= 1``).
        filter_value: Replacement value for filtered tokens (default ``-inf``).

    Returns:
        Filtered ``torch.float32`` logits tensor of the same shape as ``logits``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive, got {temperature}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must lie in (0.0, 1.0], got {top_p}")
    if not (0.0 <= min_p <= 1.0):
        raise ValueError(f"min_p must lie in [0.0, 1.0], got {min_p}")
    if min_tokens_to_keep < 1:
        raise ValueError(f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}")

    vocab_size = int(logits.shape[-1])
    keep_min = min(int(min_tokens_to_keep), vocab_size)
    work = logits.to(dtype=torch.float32) / float(temperature)
    has_any_finite = torch.isfinite(work).any(dim=-1, keepdim=True)

    # 1. Top-k truncation
    if 0 < top_k < vocab_size:
        k = max(int(top_k), keep_min)
        kth_vals = torch.topk(work, k=k, dim=-1).values[..., -1:]
        work = work.masked_fill(work < kth_vals, float(filter_value))

    # 2. Top-p (nucleus) truncation
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(work, dim=-1, descending=True)
        safe_sorted = torch.where(has_any_finite, sorted_logits, torch.zeros_like(sorted_logits))
        sorted_probs = F.softmax(safe_sorted, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Shift right so the first token crossing `top_p` is retained.
        shifted_cum = cum_probs - sorted_probs
        remove_sorted = shifted_cum >= float(top_p)
        remove_sorted[..., :keep_min] = False

        filtered_sorted = sorted_logits.masked_fill(remove_sorted, float(filter_value))
        restored = torch.empty_like(work)
        restored.scatter_(-1, sorted_indices, filtered_sorted)
        work = torch.where(has_any_finite, restored, work)

    # 3. Min-p truncation (Nguyen et al., ICLR 2025)
    if min_p > 0.0:
        safe_work = torch.where(has_any_finite, work, torch.zeros_like(work))
        probs = F.softmax(safe_work, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        keep_mask = probs >= (float(min_p) * max_prob)
        top_idx = torch.topk(work, k=keep_min, dim=-1).indices
        keep_mask.scatter_(-1, top_idx, True)
        work = torch.where(has_any_finite & (~keep_mask), float(filter_value), work)

    return work


def _combine_weighted_logit_ensemble(
    logits_or_seq: Tensor | Sequence[Tensor],
    ref_ndim: int,
    weights: Sequence[float] | Tensor | None,
    name: str,
) -> Tensor:
    """Average single or multi-conformation logits ``[K, ..., V]`` into ``[..., V]``."""
    if isinstance(logits_or_seq, Tensor):
        t = logits_or_seq.to(dtype=torch.float32)
        if t.ndim == ref_ndim:
            return t
        if t.ndim == ref_ndim + 1:
            stacked = t
        else:
            raise ValueError(
                f"{name} tensor ndim ({t.ndim}) must equal uncond_logits.ndim ({ref_ndim}) "
                f"or uncond_logits.ndim + 1 ({ref_ndim + 1})."
            )
    else:
        seq = list(logits_or_seq)
        if not seq:
            raise ValueError(f"{name} must contain at least one logit tensor.")
        stacked = torch.stack([lg.to(dtype=torch.float32) for lg in seq], dim=0)

    k = int(stacked.shape[0])
    if k == 0:
        raise ValueError(f"{name} ensemble dimension K must be >= 1.")

    if weights is None:
        return stacked.mean(dim=0)

    w_tensor = (
        weights.to(device=stacked.device, dtype=torch.float32)
        if isinstance(weights, Tensor)
        else torch.tensor([float(w) for w in weights], device=stacked.device, dtype=torch.float32)
    )
    if w_tensor.ndim != 1 or int(w_tensor.shape[0]) != k:
        raise ValueError(f"{name} weights length {tuple(w_tensor.shape)} != ensemble size {k}")
    if bool((w_tensor < 0.0).any().item()):
        raise ValueError(f"{name} weights must be non-negative.")
    total = float(w_tensor.sum().item())
    if total <= 0.0:
        raise ValueError(f"Sum of {name} weights must be strictly positive, got {total}")
    norm_w = (w_tensor / total).view(k, *([1] * (stacked.ndim - 1)))
    return (stacked * norm_w).sum(dim=0)


def apply_contrastive_cfg_logits(
    on_target_logits: Tensor | Sequence[Tensor],
    uncond_logits: Tensor,
    *,
    off_target_logits: Tensor | Sequence[Tensor] | None = None,
    on_weights: Sequence[float] | Tensor | None = None,
    off_weights: Sequence[float] | Tensor | None = None,
    w_on: float = 1.6,
    w_off: float = 0.0,
) -> Tensor:
    """Multi-conformation 3D pocket ensemble + off-target contrastive Classifier-Free Guidance.

    Implements the zero-retraining selectivity steering equation:

    .. math::
        \\ell_{\\text{guided}} = \\ell_{\\varnothing}
        + w_{\\text{on}} \\left(\\sum_{k=1}^{K} \\alpha_k \\ell_{\\text{on}}^{(k)}
        - \\ell_{\\varnothing}\\right)
        - w_{\\text{off}} \\left(\\sum_{j=1}^{M} \\beta_j \\ell_{\\text{off}}^{(j)}
        - \\ell_{\\varnothing}\\right)

    where ``alpha`` and ``beta`` are normalized probability weights summing to ``1.0``.

    Args:
        on_target_logits: Single tensor ``[B, V]``, stacked ensemble ``[K, B, V]``, or
            sequence of ``K`` tensors ``[B, V]`` from on-target pocket conformations.
        uncond_logits: Unconditional (null-pocket) logits ``[B, V]``.
        off_target_logits: Optional counter-target (e.g. Avoid-ome / selectivity trap)
            logits as ``[B, V]``, ``[M, B, V]``, or sequence of ``M`` tensors.
        on_weights: Optional non-negative conformation weights for ``on_target_logits``.
        off_weights: Optional non-negative repulsion weights for ``off_target_logits``.
        w_on: On-target CFG guidance strength (``1.0`` = unguided conditional).
        w_off: Off-target contrastive repulsion strength (``0.0`` = disabled).

    Returns:
        Guided ``torch.float32`` logits matching ``uncond_logits.shape``.
    """
    uncond = uncond_logits.to(dtype=torch.float32)
    on_ens = _combine_weighted_logit_ensemble(
        on_target_logits, ref_ndim=uncond.ndim, weights=on_weights, name="on_target_logits"
    )
    if on_ens.shape != uncond.shape:
        raise ValueError(
            f"on_target_logits shape {tuple(on_ens.shape)} != uncond_logits {tuple(uncond.shape)}"
        )

    guided = uncond + float(w_on) * (on_ens - uncond)
    if off_target_logits is not None and abs(float(w_off)) > 1e-8:
        off_ens = _combine_weighted_logit_ensemble(
            off_target_logits,
            ref_ndim=uncond.ndim,
            weights=off_weights,
            name="off_target_logits",
        )
        if off_ens.shape != uncond.shape:
            raise ValueError(
                f"off_target_logits shape {tuple(off_ens.shape)} != uncond_logits "
                f"{tuple(uncond.shape)}"
            )
        guided = guided - float(w_off) * (off_ens - uncond)
    return guided


@dataclass(frozen=True)
class BeamSearchResult:
    """Structured output returned by :func:`beam_search_decode`."""

    sequences: list[list[int]]
    scores: list[float]
    raw_scores: list[float]
    token_ids: Tensor


def _extract_step_logits_and_cache(
    model: PocketConditionedChemicalLM | Callable[..., Any],
    step_ids: Tensor,
    *,
    pocket_tokens: Tensor | None,
    pocket_key_padding_mask: Tensor | None,
    pocket_mask: Tensor | None,
    kv_cache: KVCache | None,
    use_cache: bool,
) -> tuple[Tensor, KVCache | None]:
    """Normalize logits and optional cache from a model or custom step callable."""
    if isinstance(model, PocketConditionedChemicalLM):
        out = model(
            step_ids,
            pocket_tokens=pocket_tokens,
            pocket_key_padding_mask=pocket_key_padding_mask,
            pocket_mask=pocket_mask,
            kv_cache=kv_cache if use_cache else None,
            use_cache=use_cache,
            is_causal=True,
        )
        return out.logits[:, -1, :].to(dtype=torch.float32), out.kv_cache

    raw = model(step_ids, kv_cache) if use_cache else model(step_ids, None)
    if isinstance(raw, ChemicalLMOutput):
        return raw.logits[:, -1, :].to(dtype=torch.float32), raw.kv_cache
    if isinstance(raw, tuple):
        lg = raw[0]
        next_cache = raw[1] if len(raw) > 1 else None
        logits_2d = lg[:, -1, :] if lg.ndim == 3 else lg
        return logits_2d.to(dtype=torch.float32), next_cache
    logits_2d = raw[:, -1, :] if raw.ndim == 3 else raw
    return logits_2d.to(dtype=torch.float32), None


def _expand_batch_tensor(t: Tensor | None, beam_width: int) -> Tensor | None:
    """Expand a single-item ``[1, ...]`` conditioning tensor to ``[beam_width, ...]``."""
    if t is None:
        return None
    if int(t.shape[0]) == beam_width:
        return t
    if int(t.shape[0]) != 1:
        raise ValueError(
            f"Expected batch size 1 (or beam_width={beam_width}) for beam search conditioning, "
            f"got {int(t.shape[0])}"
        )
    return t.expand(beam_width, *([-1] * (t.ndim - 1))).contiguous()


@torch.no_grad()
def beam_search_decode(
    model: PocketConditionedChemicalLM | Callable[..., Any],
    *,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    prompt_ids: Sequence[int] | Tensor | None = None,
    pocket_tokens: Tensor | None = None,
    pocket_key_padding_mask: Tensor | None = None,
    pocket_mask: Tensor | None = None,
    beam_width: int = 4,
    max_new_tokens: int = 64,
    length_penalty: float = 0.6,
    temperature: float = 1.0,
    grammar_mask: SMILESValenceGrammarMask | None = None,
    use_cache: bool = True,
    num_return_sequences: int | None = None,
    device: torch.device | str | None = None,
) -> BeamSearchResult:
    """Deterministic Wu et al. length-normalized beam search with optional grammar & KV-cache.

    Uses ``((5 + L) / 6)^length_penalty`` normalization on true cumulative log-probabilities
    and reorders both :class:`KVCache` tensors and :class:`SMILESValenceGrammarMask` parser
    states cleanly across beam branches.

    Args:
        model: :class:`PocketConditionedChemicalLM` or callable
            ``(input_ids, kv_cache) -> logits | (logits, next_cache)``.
        bos_id: Beginning-of-sequence token ID (used when ``prompt_ids`` is ``None``).
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID.
        prompt_ids: Optional initial prefix token IDs ``[P]`` or ``[1, P]``.
        pocket_tokens: Optional 3D pocket tensor ``[1, N, d_pocket]``.
        pocket_key_padding_mask: Optional pocket padding mask ``[1, N]`` (``True`` = pad).
        pocket_mask: Optional pocket validity mask ``[1, N]`` (``True`` = valid).
        beam_width: Number of hypotheses maintained at each decoding step.
        max_new_tokens: Maximum tokens generated beyond the prompt.
        length_penalty: Wu et al. length normalization exponent ``alpha >= 0.0``.
        temperature: Positive logit temperature divisor prior to ``log_softmax``.
        grammar_mask: Optional :class:`SMILESValenceGrammarMask` enforcing chemical validity.
        use_cache: Enable incremental :class:`KVCache` decoding (matches uncached decode).
        num_return_sequences: Number of top hypotheses returned (defaults to ``beam_width``).
        device: Optional execution device (inferred from model/pocket if omitted).

    Returns:
        :class:`BeamSearchResult` sorted descending by length-normalized score.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    n_return = beam_width if num_return_sequences is None else int(num_return_sequences)
    if not (1 <= n_return <= beam_width):
        raise ValueError(f"num_return_sequences ({n_return}) must be in [1, {beam_width}]")

    if device is not None:
        dev = torch.device(device)
    elif pocket_tokens is not None:
        dev = pocket_tokens.device
    elif isinstance(model, nn.Module):
        first_param = next(model.parameters(), None)
        dev = first_param.device if first_param is not None else torch.device("cpu")
    else:
        dev = torch.device("cpu")

    if prompt_ids is None:
        init_seq = [int(bos_id)]
    elif isinstance(prompt_ids, Tensor):
        init_seq = [int(x) for x in prompt_ids.detach().cpu().reshape(-1).tolist()]
    else:
        init_seq = [int(x) for x in prompt_ids]
    if not init_seq:
        raise ValueError("prompt_ids must be non-empty when provided.")

    was_training = isinstance(model, nn.Module) and model.training
    if isinstance(model, nn.Module):
        model.eval()

    try:
        # Step 0: prefill single prompt sequence, then fan out to `beam_width` distinct tokens.
        prompt_t = torch.tensor([init_seq], dtype=torch.long, device=dev)
        step_logits, cache = _extract_step_logits_and_cache(
            model,
            prompt_t,
            pocket_tokens=pocket_tokens,
            pocket_key_padding_mask=pocket_key_padding_mask,
            pocket_mask=pocket_mask,
            kv_cache=None,
            use_cache=use_cache,
        )  # [1, V]
        vocab_size = int(step_logits.shape[-1])

        if grammar_mask is not None:
            grammar_mask.reset(batch_size=1)
            for tid in init_seq:
                grammar_mask.step([tid])
            step_logits = grammar_mask.mask_logits(step_logits)

        log_probs_0 = F.log_softmax(step_logits / float(temperature), dim=-1).squeeze(0)  # [V]
        k0 = min(beam_width, vocab_size)
        top_scores_0, top_toks_0 = torch.topk(log_probs_0, k=k0, dim=-1)

        sequences: list[list[int]] = []
        raw_scores: list[float] = []
        finished: list[bool] = []
        gen_lengths: list[int] = []

        for b in range(beam_width):
            sc = float(top_scores_0[b].item()) if b < k0 else float("-inf")
            if b < k0 and math.isfinite(sc):
                tok = int(top_toks_0[b].item())
                sequences.append([*init_seq, tok])
                raw_scores.append(sc)
                finished.append(tok == eos_id)
                gen_lengths.append(1)
            else:
                sequences.append([*init_seq, pad_id])
                raw_scores.append(-1e9)
                finished.append(True)
                gen_lengths.append(1)

        if cache is not None:
            cache = cache.reorder(torch.zeros(beam_width, dtype=torch.long, device=dev))
        if grammar_mask is not None:
            grammar_mask.clone_states([0] * beam_width)
            grammar_mask.step([seq[-1] for seq in sequences])

        exp_pocket = _expand_batch_tensor(pocket_tokens, beam_width)
        exp_kpm = _expand_batch_tensor(pocket_key_padding_mask, beam_width)
        exp_pm = _expand_batch_tensor(pocket_mask, beam_width)

        # Subsequent steps 1 .. max_new_tokens - 1
        for _ in range(1, max_new_tokens):
            if all(finished):
                break

            if use_cache and cache is not None:
                step_in = torch.tensor(
                    [[seq[-1]] for seq in sequences], dtype=torch.long, device=dev
                )
            else:
                step_in = torch.tensor(sequences, dtype=torch.long, device=dev)

            step_logits, cache = _extract_step_logits_and_cache(
                model,
                step_in,
                pocket_tokens=exp_pocket,
                pocket_key_padding_mask=exp_kpm,
                pocket_mask=exp_pm,
                kv_cache=cache if use_cache else None,
                use_cache=use_cache,
            )  # [beam_width, V]

            if grammar_mask is not None:
                step_logits = grammar_mask.mask_logits(step_logits)

            log_probs = F.log_softmax(step_logits / float(temperature), dim=-1)  # [B, V]
            prev_scores_t = torch.tensor(raw_scores, dtype=torch.float32, device=dev)
            cand_scores = prev_scores_t.unsqueeze(1) + log_probs  # [B, V]

            # Completed beams may only advance via PAD at zero incremental cost.
            for b, is_done in enumerate(finished):
                if is_done:
                    cand_scores[b].fill_(float("-inf"))
                    cand_scores[b, pad_id] = prev_scores_t[b]

            flat_scores = cand_scores.reshape(-1)
            top_vals, top_flat_idx = torch.topk(flat_scores, k=beam_width, dim=-1)

            parent_indices: list[int] = []
            new_sequences: list[list[int]] = []
            new_raw_scores: list[float] = []
            new_finished: list[bool] = []
            new_lengths: list[int] = []
            stepped_tokens: list[int] = []

            for k_i in range(beam_width):
                flat_i = int(top_flat_idx[k_i].item())
                parent_b = flat_i // vocab_size
                tok_id = flat_i % vocab_size
                parent_done = finished[parent_b]

                parent_indices.append(parent_b)
                cand_val = float(top_vals[k_i].item())
                if parent_done:
                    new_sequences.append([*sequences[parent_b], pad_id])
                    new_raw_scores.append(raw_scores[parent_b])
                    new_finished.append(True)
                    new_lengths.append(gen_lengths[parent_b])
                    stepped_tokens.append(pad_id)
                elif not math.isfinite(cand_val):
                    new_sequences.append([*sequences[parent_b], pad_id])
                    new_raw_scores.append(-1e9)
                    new_finished.append(True)
                    new_lengths.append(gen_lengths[parent_b])
                    stepped_tokens.append(pad_id)
                else:
                    new_sequences.append([*sequences[parent_b], tok_id])
                    new_raw_scores.append(cand_val)
                    new_finished.append(tok_id == eos_id)
                    new_lengths.append(gen_lengths[parent_b] + 1)
                    stepped_tokens.append(tok_id)

            if cache is not None:
                cache = cache.reorder(torch.tensor(parent_indices, dtype=torch.long, device=dev))
            if grammar_mask is not None:
                grammar_mask.clone_states(parent_indices)
                grammar_mask.step(stepped_tokens)

            sequences = new_sequences
            raw_scores = new_raw_scores
            finished = new_finished
            gen_lengths = new_lengths

        # Final Wu et al. length normalization and deterministic ranking.
        ranked: list[tuple[float, float, list[int]]] = []
        for seq, r_score, g_len in zip(sequences, raw_scores, gen_lengths, strict=True):
            # Trim trailing PAD tokens produced after EOS while keeping EOS itself.
            trimmed: list[int] = []
            for idx_pos, tid in enumerate(seq):
                if idx_pos >= len(init_seq) and tid == pad_id:
                    break
                trimmed.append(tid)
            lp = (
                ((5.0 + float(max(1, g_len))) / 6.0) ** float(length_penalty)
                if length_penalty > 0.0
                else 1.0
            )
            norm_score = r_score / lp
            ranked.append((norm_score, r_score, trimmed))

        ranked.sort(key=lambda item: (item[0], tuple(-t for t in item[2])), reverse=True)
        top_ranked = ranked[:n_return]

        out_seqs = [item[2] for item in top_ranked]
        out_scores = [item[0] for item in top_ranked]
        out_raw = [item[1] for item in top_ranked]

        max_l = max(len(s) for s in out_seqs)
        padded = torch.full((n_return, max_l), pad_id, dtype=torch.long, device=dev)
        for i, seq in enumerate(out_seqs):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=dev)

        return BeamSearchResult(
            sequences=out_seqs,
            scores=out_scores,
            raw_scores=out_raw,
            token_ids=padded,
        )
    finally:
        if was_training and isinstance(model, nn.Module):
            model.train()


def sequence_logprobs_from_logits(
    logits: Tensor,
    target_ids: Tensor,
    *,
    mask: Tensor | None = None,
    pad_id: int | None = PAD_ID,
    shift: bool = True,
    average_per_token: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute exact per-token and sequence log-probabilities from LM ``logits``.

    Args:
        logits: Unnormalized logits of shape ``[..., L, V]``.
        target_ids: Target token IDs of shape ``[..., L]``.
        mask: Optional boolean validity mask of shape ``[..., L]`` (``True`` = valid).
        pad_id: Optional pad token ID masked out automatically when provided.
        shift: If ``True`` (standard causal LM), align ``logits[..., :-1, :]`` against
            ``target_ids[..., 1:]``.
        average_per_token: If ``True``, return length-normalized mean log-prob per
            sequence instead of summed sequence log-prob.

    Returns:
        Tuple ``(seq_logps [...], token_logps [..., L'])`` in ``torch.float32``.
    """
    if shift:
        if int(logits.shape[-2]) < 2:
            raise ValueError("shift=True requires sequence length L >= 2.")
        work_logits = logits[..., :-1, :].to(dtype=torch.float32)
        work_targets = target_ids[..., 1:].to(dtype=torch.long)
        work_mask = mask[..., 1:].to(dtype=torch.bool) if mask is not None else None
    else:
        work_logits = logits.to(dtype=torch.float32)
        work_targets = target_ids.to(dtype=torch.long)
        work_mask = mask.to(dtype=torch.bool) if mask is not None else None

    if pad_id is not None:
        non_pad = work_targets != int(pad_id)
        work_mask = non_pad if work_mask is None else (work_mask & non_pad)

    log_probs = F.log_softmax(work_logits, dim=-1)
    safe_targets = work_targets.clamp(min=0, max=int(work_logits.shape[-1]) - 1)
    tok_lp = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

    if work_mask is not None:
        m_f = work_mask.to(dtype=torch.float32)
        tok_lp = tok_lp * m_f
        seq_lp = tok_lp.sum(dim=-1)
        if average_per_token:
            seq_lp = seq_lp / m_f.sum(dim=-1).clamp(min=1.0)
    else:
        seq_lp = tok_lp.mean(dim=-1) if average_per_token else tok_lp.sum(dim=-1)
    return seq_lp, tok_lp


@dataclass(frozen=True)
class DPOLossOutput:
    """Structured output returned by :func:`dpo_preference_loss`."""

    loss: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    reward_margin: Tensor
    accuracy: Tensor


def dpo_preference_loss(
    chosen_logps: Tensor,
    rejected_logps: Tensor,
    ref_chosen_logps: Tensor | None = None,
    ref_rejected_logps: Tensor | None = None,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: Literal["dpo", "cdpo", "ipo"] = "dpo",
) -> DPOLossOutput:
    """Direct Preference Optimization (DPO / cDPO / IPO) loss for molecular preference pairs.

    Aligns chemical LMs toward higher potency, CNS MPO, or Avoid-ome selectivity pairs
    ``(y_chosen, y_rejected)`` directly from sequence log-probabilities without training a
    separate reward model:

    - **DPO / cDPO** (Rafailov et al., NeurIPS 2023; Mitchell 2023):
      :math:`\\mathcal{L} = -(1 - \\epsilon) \\log \\sigma(\\beta h)
      - \\epsilon \\log \\sigma(-\\beta h)`.
    - **IPO** (Azar et al., AISTATS 2024):
      :math:`\\mathcal{L} = (h - 1 / (2\\beta))^2`.

    Args:
        chosen_logps: Policy log-probabilities ``[B]`` for preferred ligands.
        rejected_logps: Policy log-probabilities ``[B]`` for dispreferred ligands.
        ref_chosen_logps: Optional frozen reference-model chosen log-probs ``[B]``.
        ref_rejected_logps: Optional frozen reference-model rejected log-probs ``[B]``.
        beta: KL-regularization inverse temperature ``beta > 0``.
        label_smoothing: Conservative preference noise ``epsilon in [0, 0.5)`` (cDPO).
        loss_type: ``"dpo"``, ``"cdpo"``, or ``"ipo"``.

    Returns:
        :class:`DPOLossOutput` containing scalar ``loss``, implicit rewards, margin, and
        preference ranking accuracy.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be strictly positive, got {beta}")
    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError(f"label_smoothing must lie in [0.0, 0.5), got {label_smoothing}")
    if loss_type not in ("dpo", "cdpo", "ipo"):
        raise ValueError(f"loss_type must be 'dpo', 'cdpo', or 'ipo', got {loss_type!r}")
    if (ref_chosen_logps is None) != (ref_rejected_logps is None):
        raise ValueError("Pass either both ref_chosen_logps and ref_rejected_logps, or neither.")

    pi_chosen = chosen_logps.to(dtype=torch.float32)
    pi_rejected = rejected_logps.to(dtype=torch.float32)
    if pi_chosen.shape != pi_rejected.shape:
        raise ValueError(
            f"chosen_logps shape {tuple(pi_chosen.shape)} != "
            f"rejected_logps shape {tuple(pi_rejected.shape)}"
        )

    if ref_chosen_logps is not None and ref_rejected_logps is not None:
        ref_c = ref_chosen_logps.detach().to(dtype=torch.float32)
        ref_r = ref_rejected_logps.detach().to(dtype=torch.float32)
        chosen_logratios = pi_chosen - ref_c
        rejected_logratios = pi_rejected - ref_r
    else:
        chosen_logratios = pi_chosen
        rejected_logratios = pi_rejected

    logits_diff = chosen_logratios - rejected_logratios
    scaled_margin = float(beta) * logits_diff

    if loss_type == "ipo":
        target_margin = 1.0 / (2.0 * float(beta))
        per_pair_loss = (logits_diff - target_margin).pow(2)
    else:
        eps = float(label_smoothing)
        per_pair_loss = -(1.0 - eps) * F.logsigmoid(scaled_margin) - eps * F.logsigmoid(
            -scaled_margin
        )

    chosen_rewards = float(beta) * chosen_logratios.detach()
    rejected_rewards = float(beta) * rejected_logratios.detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean()
    accuracy = (chosen_rewards > rejected_rewards).to(dtype=torch.float32).mean()

    return DPOLossOutput(
        loss=per_pair_loss.mean(),
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margin=reward_margin,
        accuracy=accuracy,
    )


@dataclass(frozen=True)
class GRPOLossOutput:
    """Structured output returned by :func:`grpo_policy_loss`."""

    loss: Tensor
    policy_loss: Tensor
    kl_penalty: Tensor
    advantages: Tensor
    clip_fraction: Tensor
    mean_ratio: Tensor


def grpo_policy_loss(
    policy_logps: Tensor,
    old_policy_logps: Tensor,
    rewards: Tensor,
    *,
    ref_logps: Tensor | None = None,
    mask: Tensor | None = None,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    kl_beta: float = 0.04,
    adv_eps: float = 1e-6,
    normalize_std: bool = True,
) -> GRPOLossOutput:
    """Group Relative Policy Optimization (GRPO) loss for critic-free molecular RL alignment.

    Normalizes scalar rewards across groups of ``G`` sampled ligands per pocket (zero critic
    network needed), applies PPO-clipped importance-ratio surrogate optimization, and adds
    DeepSeekMath's unbiased non-negative :math:`k_3` KL penalty against a frozen chemical
    prior:

    .. math::
        A_{b,g} = \\frac{r_{b,g} - \\mu_b}{\\sqrt{\\sigma_b^2 + \\varepsilon}},
        \\quad
        \\hat{D}_{\\text{KL}}(\\pi_\\theta \\| \\pi_{\\text{ref}})
        = \\exp(\\log \\pi_{\\text{ref}} - \\log \\pi_\\theta)
        - (\\log \\pi_{\\text{ref}} - \\log \\pi_\\theta) - 1.

    Args:
        policy_logps: Current policy log-probabilities of shape ``[B, G]`` (sequence-level)
            or ``[B, G, T]`` (token-level).
        old_policy_logps: Rollout behavior policy log-probabilities matching
            ``policy_logps.shape`` (automatically detached).
        rewards: Scalar multi-objective molecular rewards of shape ``[B, G]``.
        ref_logps: Optional frozen reference-policy log-probabilities matching
            ``policy_logps.shape`` for KL regularization.
        mask: Optional boolean validity mask matching ``policy_logps.shape`` (``True`` =
            valid token/sample).
        clip_eps: Lower PPO clipping epsilon (``> 0``).
        clip_eps_high: Optional upper PPO clipping epsilon (defaults to ``clip_eps``; set
            higher for DAPO-style Clip-Higher exploration).
        kl_beta: Non-negative KL regularization coefficient.
        adv_eps: Variance floor in group-relative advantage standardization.
        normalize_std: Divide centered group rewards by group std when ``True`` (set
            ``False`` for Dr. GRPO unbiased scaling).

    Returns:
        :class:`GRPOLossOutput` with total ``loss``, ``policy_loss``, ``kl_penalty``,
        zero-mean per-group ``advantages [B, G]``, ``clip_fraction``, and ``mean_ratio``.
    """
    if policy_logps.ndim not in (2, 3):
        raise ValueError(
            f"Expected policy_logps shape [B, G] or [B, G, T], got {tuple(policy_logps.shape)}"
        )
    if old_policy_logps.shape != policy_logps.shape:
        raise ValueError(
            f"old_policy_logps shape {tuple(old_policy_logps.shape)} != "
            f"policy_logps shape {tuple(policy_logps.shape)}"
        )
    if rewards.shape != policy_logps.shape[:2]:
        raise ValueError(
            f"rewards shape {tuple(rewards.shape)} != expected {tuple(policy_logps.shape[:2])}"
        )
    if clip_eps <= 0.0:
        raise ValueError(f"clip_eps must be > 0, got {clip_eps}")
    eps_high = float(clip_eps) if clip_eps_high is None else float(clip_eps_high)
    if eps_high <= 0.0:
        raise ValueError(f"clip_eps_high must be > 0, got {eps_high}")
    if kl_beta < 0.0:
        raise ValueError(f"kl_beta must be >= 0, got {kl_beta}")

    pi_lp = policy_logps.to(dtype=torch.float32)
    old_lp = old_policy_logps.detach().to(dtype=torch.float32)
    rew = rewards.detach().to(dtype=torch.float32)  # [B, G]

    # Exact zero-mean group-relative advantages per pocket prompt.
    mean_r = rew.mean(dim=-1, keepdim=True)
    centered = rew - mean_r
    if normalize_std:
        var_r = (centered * centered).mean(dim=-1, keepdim=True)
        advantages = centered * torch.rsqrt(var_r + float(adv_eps))
    else:
        advantages = centered

    adv_broadcast = advantages.unsqueeze(-1) if pi_lp.ndim == 3 else advantages

    log_ratio = (pi_lp - old_lp).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    low = 1.0 - float(clip_eps)
    high = 1.0 + eps_high
    clipped_ratio = ratio.clamp(min=low, max=high)

    surr1 = ratio * adv_broadcast
    surr2 = clipped_ratio * adv_broadcast
    per_elem_policy = -torch.minimum(surr1, surr2)
    is_clipped = ((ratio < low) | (ratio > high)).to(dtype=torch.float32)

    if ref_logps is not None and kl_beta > 0.0:
        if ref_logps.shape != pi_lp.shape:
            raise ValueError(
                f"ref_logps shape {tuple(ref_logps.shape)} != policy_logps {tuple(pi_lp.shape)}"
            )
        delta_ref = (ref_logps.detach().to(dtype=torch.float32) - pi_lp).clamp(min=-20.0, max=20.0)
        per_elem_kl = torch.exp(delta_ref) - delta_ref - 1.0
    else:
        per_elem_kl = torch.zeros_like(pi_lp)

    if mask is not None:
        if mask.shape != pi_lp.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != policy_logps {tuple(pi_lp.shape)}")
        m_f = mask.to(device=pi_lp.device, dtype=torch.float32)
        denom = m_f.sum().clamp(min=1.0)
        policy_loss = (per_elem_policy * m_f).sum() / denom
        kl_mean = (per_elem_kl * m_f).sum() / denom
        clip_fraction = (is_clipped * m_f).sum() / denom
        mean_ratio = (ratio.detach() * m_f).sum() / denom
    else:
        policy_loss = per_elem_policy.mean()
        kl_mean = per_elem_kl.mean()
        clip_fraction = is_clipped.mean()
        mean_ratio = ratio.detach().mean()

    kl_penalty = float(kl_beta) * kl_mean
    total_loss = policy_loss + kl_penalty
    return GRPOLossOutput(
        loss=total_loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        advantages=advantages,
        clip_fraction=clip_fraction,
        mean_ratio=mean_ratio,
    )


@dataclass(frozen=True)
class MaskedDiffusionCorruption:
    """Output of :meth:`DiscreteMaskedDiffusionStep.forward_corrupt`."""

    noisy_ids: Tensor
    corrupt_mask: Tensor
    mask_prob: Tensor


class DiscreteMaskedDiffusionStep(nn.Module):
    """Absorbing ``<mask>`` discrete diffusion primitive for MDLM & SAFE-fragment inpainting.

    Implements forward absorbing-state masking with cosine or linear schedules,
    Rao-Blackwellized weighted MDLM cross-entropy loss, and reverse unmasking transitions
    supporting both exact MDLM posterior sampling and confidence-guided MaskGIT unmasking
    with pinned scaffold tokens (``fixed_mask``).

    Args:
        mask_token_id: Integer token ID of the absorbing ``<mask>`` state (default ``3``).
        schedule: Mask probability schedule ``"cosine"`` (:math:`\\sin^2(\\pi t / 2)`) or
            ``"linear"`` (:math:`t`). Both satisfy exact boundary conditions
            :math:`\\gamma(0) = 0` (0% masked) and :math:`\\gamma(1) = 1` (100% masked).
        eps: Numerical floor for loss weighting and posterior transition ratios.
    """

    def __init__(
        self,
        mask_token_id: int = MASK_ID,
        *,
        schedule: Literal["cosine", "linear"] = "cosine",
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if mask_token_id < 0:
            raise ValueError(f"mask_token_id must be >= 0, got {mask_token_id}")
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"schedule must be 'cosine' or 'linear', got {schedule!r}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.mask_token_id = int(mask_token_id)
        self.schedule: Literal["cosine", "linear"] = schedule
        self.eps = float(eps)

    def _to_time_tensor(self, t: Tensor | float, batch_size: int, device: torch.device) -> Tensor:
        """Broadcast scalar or 1D timestep ``t in [0, 1]`` to shape ``[B]`` in ``float32``."""
        if isinstance(t, Tensor):
            t_f = t.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            if t_f.ndim == 0:
                return t_f.expand(batch_size)
            if t_f.shape == (batch_size,):
                return t_f
            raise ValueError(
                f"Expected scalar or [{batch_size}] timestep tensor, got {tuple(t.shape)}"
            )
        t_clamped = min(max(float(t), 0.0), 1.0)
        return torch.full((batch_size,), t_clamped, device=device, dtype=torch.float32)

    def mask_probability(self, t: Tensor | float) -> Tensor:
        """Evaluate exact monotonically increasing mask probability :math:`\\gamma(t)`."""
        t_tensor = (
            t.to(dtype=torch.float32)
            if isinstance(t, Tensor)
            else torch.tensor(float(t), dtype=torch.float32)
        ).clamp(0.0, 1.0)
        if self.schedule == "linear":
            return t_tensor
        # sin^2(pi * t / 2) == 1 - cos^2(pi * t / 2), giving exact 0.0 at t=0 and 1.0 at t=1.
        return torch.sin(0.5 * math.pi * t_tensor).pow(2)

    def mask_probability_derivative(self, t: Tensor | float) -> Tensor:
        """Evaluate exact time derivative :math:`\\gamma'(t) \\ge 0` for MDLM NELBO weighting."""
        t_tensor = (
            t.to(dtype=torch.float32)
            if isinstance(t, Tensor)
            else torch.tensor(float(t), dtype=torch.float32)
        ).clamp(0.0, 1.0)
        if self.schedule == "linear":
            return torch.ones_like(t_tensor)
        return 0.5 * math.pi * torch.sin(math.pi * t_tensor)

    def forward_corrupt(
        self,
        token_ids: Tensor,
        t: Tensor | float,
        *,
        fixed_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskedDiffusionCorruption:
        """Corrupt clean tokens ``[B, L]`` into absorbing ``<mask>`` tokens at noise level ``t``.

        Args:
            token_ids: Clean token IDs of shape ``[B, L]`` (``torch.long``).
            t: Noise level(s) in ``[0, 1]`` (scalar float or ``[B]`` tensor).
            fixed_mask: Optional boolean mask ``[B, L]`` where ``True`` pins immutable
                tokens (e.g. ``<bos>``, ``<eos>``, ``<pad>``, or a core SAFE scaffold).
            generator: Optional ``torch.Generator`` for deterministic corruption.

        Returns:
            :class:`MaskedDiffusionCorruption` containing ``noisy_ids``, ``corrupt_mask``,
            and per-sample ``mask_prob``.
        """
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids shape [B, L], got {tuple(token_ids.shape)}")
        bsz, seq_len = token_ids.shape
        dev = token_ids.device

        t_vec = self._to_time_tensor(t, bsz, dev)
        p_mask = self.mask_probability(t_vec)  # [B]

        if generator is not None:
            u = torch.rand(
                (bsz, seq_len), generator=generator, device=generator.device, dtype=torch.float32
            ).to(dev)
        else:
            u = torch.rand((bsz, seq_len), device=dev, dtype=torch.float32)

        corrupt_mask = u < p_mask.unsqueeze(-1)
        if fixed_mask is not None:
            if fixed_mask.shape != token_ids.shape:
                raise ValueError(
                    f"fixed_mask shape {tuple(fixed_mask.shape)} != "
                    f"token_ids {tuple(token_ids.shape)}"
                )
            corrupt_mask = corrupt_mask & (~fixed_mask.to(device=dev, dtype=torch.bool))

        noisy_ids = torch.where(
            corrupt_mask,
            torch.full_like(token_ids, self.mask_token_id),
            token_ids,
        )
        return MaskedDiffusionCorruption(
            noisy_ids=noisy_ids,
            corrupt_mask=corrupt_mask,
            mask_prob=p_mask,
        )

    def diffusion_loss(
        self,
        logits: Tensor,
        target_ids: Tensor,
        corrupt_mask: Tensor,
        t: Tensor | float,
        *,
        use_nelbo_weight: bool = True,
    ) -> Tensor:
        """Compute Rao-Blackwellized masked-diffusion cross-entropy loss over masked positions.

        Args:
            logits: Denoiser logits of shape ``[B, L, V]``.
            target_ids: Clean ground-truth token IDs of shape ``[B, L]``.
            corrupt_mask: Boolean mask ``[B, L]`` indicating which positions were masked.
            t: Corruption noise level(s) in ``[0, 1]``.
            use_nelbo_weight: Weight samples by the MDLM NELBO ratio
                :math:`\\gamma'(t) / \\max(\\gamma(t), \\varepsilon)` (clamped for stability).

        Returns:
            Scalar ``torch.float32`` loss tensor (exact ``0.0`` with valid grad graph if
            no tokens were masked).
        """
        bsz, seq_len, vocab_size = logits.shape
        dev = logits.device
        ce = F.cross_entropy(
            logits.to(dtype=torch.float32).reshape(-1, vocab_size),
            target_ids.to(device=dev, dtype=torch.long).reshape(-1),
            reduction="none",
        ).view(bsz, seq_len)

        m_f = corrupt_mask.to(device=dev, dtype=torch.float32)
        if use_nelbo_weight:
            t_vec = self._to_time_tensor(t, bsz, dev)
            gamma = self.mask_probability(t_vec).clamp(min=self.eps)
            gamma_prime = self.mask_probability_derivative(t_vec).clamp(min=self.eps)
            weights = (gamma_prime / gamma).clamp(max=1.0 / self.eps).unsqueeze(-1)
            weighted_ce = ce * weights * m_f
        else:
            weighted_ce = ce * m_f

        denom = m_f.sum().clamp(min=1.0)
        return weighted_ce.sum() / denom

    @torch.no_grad()
    def reverse_step(
        self,
        x_t: Tensor,
        logits: Tensor,
        t: Tensor | float,
        s: Tensor | float,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        confidence_remasking: bool = True,
        forbidden_token_ids: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Perform one reverse unmasking transition from noise level ``t`` down to ``s < t``.

        Preserves every already-unmasked token verbatim (absorbing-state carry-over). When
        ``s == 0.0``, guarantees all remaining ``<mask>`` tokens are unmasked into valid
        non-mask token IDs.

        Args:
            x_t: Current partially masked token IDs of shape ``[B, L]``.
            logits: Denoiser logits of shape ``[B, L, V]``.
            t: Current noise level in ``(0, 1]``.
            s: Next lower noise level in ``[0, t]``.
            temperature: Sampling temperature (``0.0`` for deterministic greedy argmax).
            top_k: Optional top-k truncation before sampling predicted tokens.
            top_p: Optional nucleus truncation before sampling predicted tokens.
            confidence_remasking: If ``True``, unmask the highest-confidence positions
                according to the noise schedule ratio ``gamma(s) / gamma(t)``; if ``False``,
                sample independent MDLM posterior unmasking transitions.
            forbidden_token_ids: Optional extra token IDs forbidden from being predicted
                (``mask_token_id`` is always forbidden).
            generator: Optional ``torch.Generator`` for deterministic sampling.

        Returns:
            Updated token IDs ``x_s`` of shape ``[B, L]`` (``torch.long``).
        """
        if x_t.ndim != 2:
            raise ValueError(f"Expected x_t shape [B, L], got {tuple(x_t.shape)}")
        if logits.ndim != 3 or logits.shape[:2] != x_t.shape:
            raise ValueError(
                f"Expected logits shape [B, L, V] matching x_t {tuple(x_t.shape)}, "
                f"got {tuple(logits.shape)}"
            )

        bsz, seq_len, vocab_size = logits.shape
        dev = x_t.device
        t_vec = self._to_time_tensor(t, bsz, dev)
        s_vec = self._to_time_tensor(s, bsz, dev)
        if bool((s_vec > t_vec + 1e-6).any().item()):
            raise ValueError("Reverse diffusion step requires s <= t.")

        is_masked = x_t == self.mask_token_id
        if not bool(is_masked.any().item()):
            return x_t.clone()

        # Forbid re-emitting `<mask>` (and any caller-specified special IDs).
        clean_logits = logits.to(dtype=torch.float32).clone()
        if 0 <= self.mask_token_id < vocab_size:
            clean_logits[..., self.mask_token_id] = float("-inf")
        if forbidden_token_ids is not None:
            for fid in forbidden_token_ids:
                if 0 <= int(fid) < vocab_size:
                    clean_logits[..., int(fid)] = float("-inf")

        if temperature <= 0.0:
            probs = F.softmax(clean_logits, dim=-1)
            x0_hat = clean_logits.argmax(dim=-1)
        else:
            filtered = apply_temperature_top_k_top_p(
                clean_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            probs = F.softmax(filtered, dim=-1)
            flat_probs = probs.reshape(-1, vocab_size)
            if generator is not None and generator.device != dev:
                sampled = torch.multinomial(
                    flat_probs.detach().cpu(), num_samples=1, generator=generator
                ).to(dev)
            else:
                sampled = torch.multinomial(flat_probs, num_samples=1, generator=generator)
            x0_hat = sampled.view(bsz, seq_len)

        gamma_t = self.mask_probability(t_vec).clamp(min=self.eps)  # [B]
        gamma_s = self.mask_probability(s_vec)  # [B]
        p_unmask = ((gamma_t - gamma_s) / gamma_t).clamp(0.0, 1.0)  # [B]

        if not confidence_remasking:
            if generator is not None:
                u = torch.rand(
                    (bsz, seq_len),
                    generator=generator,
                    device=generator.device,
                    dtype=torch.float32,
                ).to(dev)
            else:
                u = torch.rand((bsz, seq_len), device=dev, dtype=torch.float32)
            unmask_now = is_masked & ((s_vec.unsqueeze(-1) <= 0.0) | (u < p_unmask.unsqueeze(-1)))
            return torch.where(unmask_now, x0_hat, x_t)

        # Confidence-guided unmasking: unmask top-k highest-confidence masked tokens per row.
        pred_conf = probs.gather(-1, x0_hat.unsqueeze(-1)).squeeze(-1)  # [B, L]
        pred_conf = pred_conf.masked_fill(~is_masked, float("-inf"))
        out = x_t.clone()
        for b in range(bsz):
            n_masked = int(is_masked[b].sum().item())
            if n_masked == 0:
                continue
            if float(s_vec[b].item()) <= 0.0:
                n_unmask = n_masked
            else:
                target_unmask = int(round(float(p_unmask[b].item()) * n_masked))
                n_unmask = max(0, min(n_masked, target_unmask))
            if n_unmask > 0:
                top_pos = torch.topk(pred_conf[b], k=n_unmask, dim=-1).indices
                out[b, top_pos] = x0_hat[b, top_pos]
        return out

    @torch.no_grad()
    def sample_inpaint(
        self,
        model: PocketConditionedChemicalLM | Callable[[Tensor], Tensor],
        template_ids: Tensor,
        *,
        num_steps: int = 8,
        pocket_tokens: Tensor | None = None,
        pocket_key_padding_mask: Tensor | None = None,
        pocket_mask: Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        confidence_remasking: bool = True,
        forbidden_token_ids: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run a full ``num_steps`` reverse diffusion loop to inpaint ``<mask>`` spans.

        Non-mask positions in ``template_ids`` (such as a pinned SAFE core scaffold,
        ``<bos>``, ``<eos>``, or ``<pad>``) remain strictly immutable across every step.
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        was_training = isinstance(model, nn.Module) and model.training
        if isinstance(model, nn.Module):
            model.eval()

        try:
            x = template_ids.clone()
            times = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
            for step_idx in range(num_steps):
                t_val = float(times[step_idx].item())
                s_val = float(times[step_idx + 1].item())
                if isinstance(model, PocketConditionedChemicalLM):
                    logits = model(
                        x,
                        pocket_tokens=pocket_tokens,
                        pocket_key_padding_mask=pocket_key_padding_mask,
                        pocket_mask=pocket_mask,
                        is_causal=False,
                    ).logits
                else:
                    logits = model(x)
                x = self.reverse_step(
                    x,
                    logits,
                    t=t_val,
                    s=s_val,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    confidence_remasking=confidence_remasking,
                    forbidden_token_ids=forbidden_token_ids,
                    generator=generator,
                )
            return x
        finally:
            if was_training and isinstance(model, nn.Module):
                model.train()


__all__ = [
    "BOS_ID",
    "BOS_TOKEN",
    "CHEMICAL_TOKEN_REGEX",
    "DEFAULT_CHEMICAL_VOCAB",
    "EOS_ID",
    "EOS_TOKEN",
    "MASK_ID",
    "MASK_TOKEN",
    "PAD_ID",
    "PAD_TOKEN",
    "SEP_ID",
    "SEP_TOKEN",
    "SPECIAL_TOKENS",
    "UNK_ID",
    "UNK_TOKEN",
    "AtomLevelSMILESTokenizer",
    "BeamSearchResult",
    "ChemicalLMOutput",
    "ChemicalTransformerBlock",
    "DPOLossOutput",
    "DiscreteMaskedDiffusionStep",
    "GRPOLossOutput",
    "GroupedQueryAttention",
    "KVCache",
    "LayerKVCache",
    "LoRALinear",
    "MaskedDiffusionCorruption",
    "PocketConditionedChemicalLM",
    "PocketCrossAttention",
    "PocketPerceiverBridge",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SMILESValenceGrammarMask",
    "SwiGLUFeedForward",
    "apply_contrastive_cfg_logits",
    "apply_rotary_pos_emb",
    "apply_temperature_top_k_top_p",
    "beam_search_decode",
    "dpo_preference_loss",
    "grpo_policy_loss",
    "inject_lora_adapters",
    "merge_lora_adapters",
    "sequence_logprobs_from_logits",
]
