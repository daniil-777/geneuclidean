"""Unit tests for MolecularTokenizer across SMILES, SELFIES, and SAFE representations."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from geneuclidean.data.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    MolecularTokenizer,
    canonicalize_smiles,
    randomize_smiles,
)

SAMPLE_DRUGS: list[str] = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # Caffeine
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # Imatinib
    "O=C([O-])c1cc(Cl)ccc1Br",  # Charged + Cl + Br
    "C[C@H](N)C(=O)O",  # Chiral L-alanine
    "c1ccc2[nH]ccc2c1",  # Indole [nH]
    r"C/C=C\C(=O)O",  # Cis double bond stereo / and \
]


def test_fixed_special_token_ids() -> None:
    """Special token IDs must stay fixed across all tokenizer instances."""
    tok = MolecularTokenizer("smiles")
    assert PAD_ID == 0
    assert BOS_ID == 1
    assert EOS_ID == 2
    assert UNK_ID == 3
    assert tok.tokens[:4] == list(SPECIAL_TOKENS)
    assert tok.token_to_id["<pad>"] == 0
    assert tok.token_to_id["<bos>"] == 1
    assert tok.token_to_id["<eos>"] == 2
    assert tok.token_to_id["<unk>"] == 3
    assert MolecularTokenizer.PAD_ID == 0
    assert MolecularTokenizer.BOS_ID == 1
    assert MolecularTokenizer.EOS_ID == 2
    assert MolecularTokenizer.UNK_ID == 3
    assert tok.PAD_ID == 0
    assert tok.BOS_ID == 1
    assert tok.EOS_ID == 2
    assert tok.UNK_ID == 3


def test_schwaller_regex_preserves_multichar_and_stereo_tokens() -> None:
    """Schwaller regex must keep Cl, Br, bracket atoms, %10 rings, and / \\ intact."""
    tok = MolecularTokenizer("smiles")
    assert tok.tokenize_string("Clc1ccc(Br)cc1[nH]") == [
        "Cl",
        "c",
        "1",
        "c",
        "c",
        "c",
        "(",
        "Br",
        ")",
        "c",
        "c",
        "1",
        "[nH]",
    ]
    stereo_tokens = tok.tokenize_string(r"C/C=C\C")
    assert stereo_tokens == ["C", "/", "C", "=", "C", "\\", "C"]
    ring_tokens = tok.tokenize_string("C%10CCCCC%10")
    assert "%10" in ring_tokens


def test_canonicalize_and_salt_stripping() -> None:
    """canonicalize_smiles strips whitespace, removes counterions, and rejects invalid input."""
    assert canonicalize_smiles("  CC(=O)O\n ") == "CC(=O)O"
    assert canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O.[Na+]", strip_salts=True) == (
        "CC(=O)Oc1ccccc1C(=O)O"
    )
    assert canonicalize_smiles("not_a_molecule_12345") is None
    assert canonicalize_smiles("") is None


def test_randomize_smiles_equivalence_and_determinism() -> None:
    """Randomized SMILES must remain chemically identical and reproducible under a fixed RNG."""
    imatinib = SAMPLE_DRUGS[2]
    can_ref = canonicalize_smiles(imatinib)
    assert can_ref is not None

    r1 = randomize_smiles(imatinib, rng=random.Random(123))
    r2 = randomize_smiles(imatinib, rng=random.Random(123))
    assert r1 == r2
    assert canonicalize_smiles(r1) == can_ref

    variants = {randomize_smiles(imatinib, rng=np.random.default_rng(seed)) for seed in range(20)}
    assert len(variants) >= 5
    for smi in variants:
        assert canonicalize_smiles(smi) == can_ref


@pytest.mark.parametrize("representation", ["smiles", "selfies", "safe"])
def test_encode_decode_roundtrip_all_representations(
    representation: str,
) -> None:
    """SMILES, SELFIES, and SAFE tokenizers round-trip diverse drugs cleanly."""
    if representation == "safe":
        pytest.importorskip("safe")
    tok = MolecularTokenizer(representation)  # type: ignore[arg-type]
    tok.fit(SAMPLE_DRUGS, randomized_passes=1 if representation == "smiles" else 0, seed=7)
    assert tok.vocab_size > 10

    for smi in SAMPLE_DRUGS:
        can_in = canonicalize_smiles(smi)
        assert can_in is not None
        ids = tok.encode(can_in, max_len=128)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID
        assert UNK_ID not in ids

        decoded = tok.decode(ids, to_smiles=True)
        mol_out = Chem.MolFromSmiles(decoded)
        assert mol_out is not None, f"Invalid decoded {representation}: {decoded!r}"
        # SAFE drops some stereochemistry during BRICS fragmentation; compare connectivity for SAFE
        if representation == "safe":
            assert Chem.MolToSmiles(Chem.MolFromSmiles(can_in), isomericSmiles=False) == (
                Chem.MolToSmiles(mol_out, isomericSmiles=False)
            )
        else:
            assert canonicalize_smiles(decoded) == can_in


def test_decode_stops_at_eos_and_pad_and_respects_max_len() -> None:
    """Decoding must strip BOS and halt immediately at the first EOS or PAD token."""
    tok = MolecularTokenizer("smiles").fit(SAMPLE_DRUGS)
    encoded = tok.encode("CC(=O)Oc1ccccc1C(=O)O", max_len=10)
    assert len(encoded) == 10
    assert encoded[0] == BOS_ID
    assert encoded[-1] == EOS_ID

    full = tok.encode("CCO", max_len=32)
    noisy_tail = [*full, tok.token_to_id["C"], tok.token_to_id["O"], PAD_ID]
    assert tok.decode(noisy_tail) == "CCO"

    pad_terminated = [
        BOS_ID,
        tok.token_to_id["C"],
        tok.token_to_id["O"],
        PAD_ID,
        tok.token_to_id["N"],
    ]
    assert tok.decode(pad_terminated) == "CO"


def test_save_and_load_json_roundtrip(tmp_path: Path) -> None:
    """Tokenizer persists strictly as clean JSON and reloads identically."""
    tok = MolecularTokenizer("selfies").fit(SAMPLE_DRUGS, min_freq=1)
    vocab_file = tok.save(tmp_path / "vocab_selfies.json")
    raw_json = json.loads(vocab_file.read_text(encoding="utf-8"))
    assert raw_json["representation"] == "selfies"
    assert raw_json["vocab"][:4] == list(SPECIAL_TOKENS)

    loaded = MolecularTokenizer.load(vocab_file)
    assert loaded.representation == "selfies"
    assert loaded.tokens == tok.tokens
    assert loaded.encode(SAMPLE_DRUGS[0]) == tok.encode(SAMPLE_DRUGS[0])
