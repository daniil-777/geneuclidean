"""HuggingFace chemical foundation transformer and compact local LLM loaders for GenEuclidean v4.

Supports verified open-weight model families on Apple Silicon MPS / CUDA / CPU plus an
offline zero-download ``mock-tiny-gpt2`` backbone for deterministic CI tests:

- Track A Primary: ``ibm-research/GP-MoLFormer-Uniq`` (46.8M, Apache-2.0, Schwaller atom-regex
  SMILES tokenizer from ``ibm-research/MoLFormer-XL-both-10pct``, ``transformers==5.24.1``
  compatibility shim, RoPE non-persistent buffer re-initialization, and 12.5x faster exact 4D
  lower-triangular causal linear attention kernel on MPS).
- Track A Fragment-Modular SAFE Prior: ``datamol-io/safe-gpt`` (alias ``"safe-gpt"``, 87.3M,
  Apache-2.0, 1.17B ZINC20+UniChem molecules) loaded directly into native ``GPT2LMHeadModel``
  (zero ``trust_remote_code``, native ``DynamicCache``) with Rust ``tokenizers>=0.23`` JSON
  sanitization and ``SAFESplitter`` regex pre-tokenization preserving single-token stereocenters.
- Track A Modern Llama Chemical Prior: ``chandar-lab/NovoMolGen_157M_SMILES_BPE`` (alias
  ``"novomolgen-157m"``, 157M, MIT, 1.5B ZINC-22 molecules) remapped in pure PyTorch from
  ``flash_attn`` fused-QKV / interleaved-RoPE / ``[up, gate]`` SwiGLU weights into native
  ``LlamaForCausalLM`` running PyTorch MPS ``SDPA`` with zero CUDA ``flash_attn`` dependency.
- Track A Native GPT-2 Fallback: ``entropy/gpt2_zinc_87m`` (87.3M, MIT, 480M ZINC SMILES).
- Track B General Local LLMs: ``Qwen/Qwen2.5-0.5B``, ``HuggingFaceTB/SmolLM2-360M``, and
  ``Qwen/Qwen3-0.6B-Base`` (always loaded in ``bfloat16`` on MPS/CUDA or ``float32`` on CPU —
  never ``float16`` on MPS due to NaN/Inf LoRA gradients).
- Offline CI Mock: ``mock-tiny-gpt2`` (tiny 2-layer GPT-2 + SMILES regex tokenizer, zero I/O).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from geneuclidean.models.safe_codec import SAFE_SPLITTER_REGEX

DEFAULT_HF_CACHE_DIR = ".hf_cache.nosync"
MOLFORMER_REPO_ID = "ibm-research/GP-MoLFormer-Uniq"
MOLFORMER_TOKENIZER_REPO_ID = "ibm-research/MoLFormer-XL-both-10pct"
SAFE_GPT_REPO_ID = "datamol-io/safe-gpt"
SAFE_GPT_ALIAS = "safe-gpt"
NOVOMOLGEN_157M_REPO_ID = "chandar-lab/NovoMolGen_157M_SMILES_BPE"
NOVOMOLGEN_157M_ALIAS = "novomolgen-157m"
NOVOMOLGEN_300M_REPO_ID = "chandar-lab/NovoMolGen_300M_SMILES_BPE"
MOCK_BACKBONE_ID = "mock-tiny-gpt2"

SAFE_GPT_EXTRA_TOKENIZER_KEYS: tuple[str, ...] = (
    "custom_pre_tokenizer",
    "tokenizer_type",
    "tokenizer_attrs",
)

_SMILES_ATOM_REGEX = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)

_MOCK_BASE_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "C",
    "c",
    "N",
    "n",
    "O",
    "o",
    "S",
    "s",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "B",
    "(",
    ")",
    "=",
    "#",
    "-",
    "+",
    "/",
    "\\",
    ".",
    ":",
    "@",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "[nH]",
    "[C@H]",
    "[C@@H]",
    "[C@]",
    "[C@@]",
    "[NH+]",
    "[NH2+]",
    "[NH3+]",
    "[O-]",
    "[N+]",
    "[S+]",
    "[H]",
    "[n+]",
    "[nH+]",
    "[o+]",
    "[s+]",
    "[CH]",
    "[CH2]",
    "[CH3]",
    "[NH]",
    "[NH2]",
    "[OH]",
    "[SH]",
    "[B-]",
    "[Si]",
    "[Se]",
    "%10",
    "%11",
    "%12",
    "H",
    "~",
    "*",
    "$",
)


@dataclass(frozen=True)
class BackboneSpec:
    """Static registry specification for a supported pretrained causal LM backbone."""

    backbone_id: str
    family: str
    tokenizer_id: str
    lora_targets: tuple[str, ...]
    fan_in_fan_out: bool
    is_smiles_native: bool
    prompt_prefix: str
    trust_remote_code: bool = False
    prefer_bf16_on_accel: bool = False
    uses_tuple_kv_cache: bool = False
    representation: str = "smiles"

    @property
    def is_safe_native(self) -> bool:
        """Return ``True`` when the backbone natively models dot-separated SAFE strings."""
        return self.representation == "safe"


@dataclass
class LoadedBackbone:
    """Loaded HF / mock causal LM backbone paired with tokenizer and empirical metadata."""

    model: nn.Module
    tokenizer: Any
    hidden_size: int
    vocab_size: int
    embed_rms: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    model_dtype: torch.dtype
    is_smiles_native: bool
    prompt_prefix: str
    backbone_id: str
    uses_tuple_kv_cache: bool = False
    representation: str = "smiles"

    @property
    def is_safe_native(self) -> bool:
        """Return ``True`` when this loaded backbone uses SAFE fragment representation."""
        return self.representation == "safe"


_SAFE_GPT_SPEC = BackboneSpec(
    backbone_id=SAFE_GPT_REPO_ID,
    family="safe_gpt",
    tokenizer_id=SAFE_GPT_REPO_ID,
    lora_targets=("c_attn", "c_proj", "c_fc"),
    fan_in_fan_out=True,
    is_smiles_native=True,
    prompt_prefix="",
    trust_remote_code=False,
    prefer_bf16_on_accel=True,
    uses_tuple_kv_cache=False,
    representation="safe",
)

_NOVOMOLGEN_157M_SPEC = BackboneSpec(
    backbone_id=NOVOMOLGEN_157M_REPO_ID,
    family="novomolgen",
    tokenizer_id=NOVOMOLGEN_157M_REPO_ID,
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    fan_in_fan_out=False,
    is_smiles_native=True,
    prompt_prefix="",
    trust_remote_code=False,
    prefer_bf16_on_accel=True,
    uses_tuple_kv_cache=False,
    representation="smiles",
)

_NOVOMOLGEN_300M_SPEC = BackboneSpec(
    backbone_id=NOVOMOLGEN_300M_REPO_ID,
    family="novomolgen",
    tokenizer_id=NOVOMOLGEN_300M_REPO_ID,
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    fan_in_fan_out=False,
    is_smiles_native=True,
    prompt_prefix="",
    trust_remote_code=False,
    prefer_bf16_on_accel=True,
    uses_tuple_kv_cache=False,
    representation="smiles",
)


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    MOLFORMER_REPO_ID: BackboneSpec(
        backbone_id=MOLFORMER_REPO_ID,
        family="molformer",
        tokenizer_id=MOLFORMER_TOKENIZER_REPO_ID,
        lora_targets=("query", "key", "value", "dense"),
        fan_in_fan_out=False,
        is_smiles_native=True,
        prompt_prefix="",
        trust_remote_code=True,
        prefer_bf16_on_accel=False,
        uses_tuple_kv_cache=True,
        representation="smiles",
    ),
    SAFE_GPT_ALIAS: _SAFE_GPT_SPEC,
    SAFE_GPT_REPO_ID: _SAFE_GPT_SPEC,
    NOVOMOLGEN_157M_ALIAS: _NOVOMOLGEN_157M_SPEC,
    NOVOMOLGEN_157M_REPO_ID: _NOVOMOLGEN_157M_SPEC,
    NOVOMOLGEN_300M_REPO_ID: _NOVOMOLGEN_300M_SPEC,
    "entropy/gpt2_zinc_87m": BackboneSpec(
        backbone_id="entropy/gpt2_zinc_87m",
        family="gpt2",
        tokenizer_id="entropy/gpt2_zinc_87m",
        lora_targets=("c_attn", "c_proj", "c_fc"),
        fan_in_fan_out=True,
        is_smiles_native=True,
        prompt_prefix="",
        trust_remote_code=False,
        prefer_bf16_on_accel=False,
        uses_tuple_kv_cache=False,
    ),
    "Qwen/Qwen2.5-0.5B": BackboneSpec(
        backbone_id="Qwen/Qwen2.5-0.5B",
        family="general_llm",
        tokenizer_id="Qwen/Qwen2.5-0.5B",
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        fan_in_fan_out=False,
        is_smiles_native=False,
        prompt_prefix="SMILES: ",
        trust_remote_code=False,
        prefer_bf16_on_accel=True,
        uses_tuple_kv_cache=False,
    ),
    "HuggingFaceTB/SmolLM2-360M": BackboneSpec(
        backbone_id="HuggingFaceTB/SmolLM2-360M",
        family="general_llm",
        tokenizer_id="HuggingFaceTB/SmolLM2-360M",
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        fan_in_fan_out=False,
        is_smiles_native=False,
        prompt_prefix="SMILES: ",
        trust_remote_code=False,
        prefer_bf16_on_accel=True,
        uses_tuple_kv_cache=False,
    ),
    "Qwen/Qwen3-0.6B-Base": BackboneSpec(
        backbone_id="Qwen/Qwen3-0.6B-Base",
        family="general_llm",
        tokenizer_id="Qwen/Qwen3-0.6B-Base",
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        fan_in_fan_out=False,
        is_smiles_native=False,
        prompt_prefix="SMILES: ",
        trust_remote_code=False,
        prefer_bf16_on_accel=True,
        uses_tuple_kv_cache=False,
    ),
    MOCK_BACKBONE_ID: BackboneSpec(
        backbone_id=MOCK_BACKBONE_ID,
        family="mock",
        tokenizer_id=MOCK_BACKBONE_ID,
        lora_targets=("c_attn", "c_proj", "c_fc"),
        fan_in_fan_out=True,
        is_smiles_native=True,
        prompt_prefix="",
        trust_remote_code=False,
        prefer_bf16_on_accel=False,
        uses_tuple_kv_cache=False,
    ),
}


class MockSmilesTokenizer:
    """Lightweight offline regex SMILES tokenizer matching HF tokenizer interfaces for CI."""

    def __init__(self, vocab_size: int = 96) -> None:
        tokens = list(_MOCK_BASE_TOKENS)
        extra_idx = 0
        while len(tokens) < vocab_size:
            tokens.append(f"<extra_{extra_idx}>")
            extra_idx += 1
        self._id_to_token: list[str] = tokens[:vocab_size]
        self._token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self._id_to_token)}

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.vocab_size = len(self._id_to_token)
        self.model_max_length = 256

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def token_to_id(self) -> dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[int, str]:
        return {i: tok for i, tok in enumerate(self._id_to_token)}

    def tokenize(self, text: str) -> list[str]:
        """Split a SMILES string into atom-level regex tokens."""
        tokens: list[str] = []
        pos = 0
        for match in _SMILES_ATOM_REGEX.finditer(text):
            if match.start() > pos:
                tokens.extend(list(text[pos : match.start()]))
            tokens.append(match.group(0))
            pos = match.end()
        if pos < len(text):
            tokens.extend(list(text[pos:]))
        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode a SMILES string into integer token IDs."""
        ids = [self._token_to_id.get(tok, self.unk_token_id) for tok in self.tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        if truncation and max_length is not None and len(ids) > max_length:
            if add_special_tokens and max_length >= 2:
                ids = [*ids[: max_length - 1], self.eos_token_id]
            else:
                ids = ids[:max_length]
        return ids

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode integer token IDs back into a SMILES string."""
        raw_ids: list[int] = (
            [int(x) for x in token_ids.detach().cpu().tolist()]
            if isinstance(token_ids, Tensor)
            else [int(x) for x in token_ids]
        )
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
        pieces: list[str] = []
        for tid in raw_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if 0 <= tid < self.vocab_size:
                tok = self._id_to_token[tid]
                if not (skip_special_tokens and tok.startswith("<extra_")):
                    pieces.append(tok)
        return "".join(pieces)

    def batch_decode(
        self,
        sequences: Sequence[Sequence[int]] | Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of token ID sequences into strings."""
        if isinstance(sequences, Tensor):
            rows: Iterable[Sequence[int]] = sequences.detach().cpu().tolist()
        else:
            rows = sequences
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """HF-compatible tokenizer call returning ``input_ids`` and ``attention_mask``."""
        is_single = isinstance(text, str)
        texts: list[str] = [text] if isinstance(text, str) else list(text)
        encoded = [
            self.encode(
                item,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for item in texts
        ]
        if padding == "max_length" and max_length is not None:
            pad_len = max_length
        elif padding:
            pad_len = max((len(seq) for seq in encoded), default=0)
        else:
            pad_len = 0

        padded_ids: list[list[int]] = []
        masks: list[list[int]] = []
        for seq in encoded:
            target = max(pad_len, len(seq))
            pad_count = target - len(seq)
            padded_ids.append(seq + [self.pad_token_id] * pad_count)
            masks.append([1] * len(seq) + [0] * pad_count)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if is_single:
            return {"input_ids": padded_ids[0], "attention_mask": masks[0]}
        return {"input_ids": padded_ids, "attention_mask": masks}


def _configure_hf_cache(cache_dir: str | None) -> Path:
    """Set ``HF_HOME`` / ``HF_HUB_CACHE`` / ``HF_MODULES_CACHE`` and return hub cache path."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cache_dir is not None and cache_dir != "":
        root = Path(cache_dir).expanduser().resolve()
    else:
        env_home = os.environ.get("HF_HOME", "").strip()
        root = (
            Path(env_home).expanduser().resolve()
            if env_home
            else Path(DEFAULT_HF_CACHE_DIR).resolve()
        )

    if root.name == "hub" or any(root.glob("models--*")):
        hf_home = root.parent if root.name == "hub" else root
        hub_dir = root
    else:
        hf_home = root
        hub_dir = root / "hub"

    modules_dir = hf_home / "modules"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_MODULES_CACHE"] = str(modules_dir)
    return hub_dir


def _install_fast_4d_tril_causal_dot_product() -> None:
    """Register exact 4D lower-triangular ``causal_dot_product`` stub (12.5x faster on MPS)."""
    if "fast_transformers.causal_product" in sys.modules:
        existing = sys.modules["fast_transformers.causal_product"]
        if hasattr(existing, "causal_dot_product"):
            return

    def causal_dot_product(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        attn = torch.matmul(q, k.transpose(-1, -2)).tril()
        return torch.matmul(attn, v)

    pkg = sys.modules.get("fast_transformers")
    if pkg is None:
        pkg = types.ModuleType("fast_transformers")
        sys.modules["fast_transformers"] = pkg

    submod = types.ModuleType("fast_transformers.causal_product")
    submod.causal_dot_product = causal_dot_product  # type: ignore[attr-defined]
    pkg.causal_product = submod  # type: ignore[attr-defined]
    sys.modules["fast_transformers.causal_product"] = submod


def _install_molformer_transformers_v5_shims() -> None:
    """Install minimal, idempotent v4 compatibility shims for ``GP-MoLFormer-Uniq`` on v5.x."""
    if "transformers.onnx" not in sys.modules:
        onnx_mod = types.ModuleType("transformers.onnx")

        class OnnxConfig:
            """Compatibility stub for removed ``transformers.onnx.OnnxConfig``."""

            def __init__(self, config: Any, task: str = "default") -> None:
                self.config = config
                self.task = task

        onnx_mod.OnnxConfig = OnnxConfig  # type: ignore[attr-defined]
        sys.modules["transformers.onnx"] = onnx_mod

    import transformers.pytorch_utils as pytorch_utils

    pytorch_utils_mod: Any = pytorch_utils
    pre_trained_cls: Any = PreTrainedModel

    if not hasattr(pytorch_utils_mod, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(
            heads: list[int],
            n_heads: int,
            head_size: int,
            already_pruned_heads: set[int],
        ) -> tuple[set[int], Tensor]:
            mask = torch.ones(n_heads, head_size)
            heads_set = set(heads) - already_pruned_heads
            for head in heads_set:
                adjusted = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[adjusted] = 0
            flat_mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(flat_mask))[flat_mask].long()
            return heads_set, index

        pytorch_utils_mod.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(pre_trained_cls, "create_extended_attention_mask_for_decoder"):

        @staticmethod  # type: ignore[misc]
        def create_extended_attention_mask_for_decoder(
            input_shape: tuple[int, int],
            attention_mask: Tensor,
            device: torch.device | None = None,
        ) -> Tensor:
            target_device = device if device is not None else attention_mask.device
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=target_device)
            causal_mask = (
                seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            )
            causal_mask = causal_mask.to(dtype=attention_mask.dtype)
            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_len = attention_mask.shape[1] - causal_mask.shape[1]
                prefix_ones = torch.ones(
                    (batch_size, seq_length, prefix_len),
                    device=target_device,
                    dtype=causal_mask.dtype,
                )
                causal_mask = torch.cat([prefix_ones, causal_mask], dim=-1)
            return causal_mask[:, None, :, :] * attention_mask[:, None, None, :]

        pre_trained_cls.create_extended_attention_mask_for_decoder = (
            create_extended_attention_mask_for_decoder
        )

    if not hasattr(pre_trained_cls, "get_extended_attention_mask"):

        def get_extended_attention_mask(
            self: Any,
            attention_mask: Tensor,
            input_shape: tuple[int, int],
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> Tensor:
            resolved_dtype = dtype if dtype is not None else getattr(self, "dtype", torch.float32)
            if resolved_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                resolved_dtype = torch.float32
            mask = attention_mask.to(dtype=resolved_dtype)
            extended: Tensor
            if mask.dim() == 3:
                extended = mask[:, None, :, :]
            elif mask.dim() == 2:
                if bool(getattr(self.config, "is_decoder", False)):
                    create_decoder_mask: Callable[..., Tensor] = (
                        pre_trained_cls.create_extended_attention_mask_for_decoder
                    )
                    extended = create_decoder_mask(input_shape, mask, device=device)
                else:
                    extended = mask[:, None, None, :]
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape {tuple(attention_mask.shape)} for "
                    f"input_shape {input_shape}"
                )
            return (1.0 - extended) * torch.finfo(resolved_dtype).min

        pre_trained_cls.get_extended_attention_mask = get_extended_attention_mask

    if not hasattr(pre_trained_cls, "invert_attention_mask"):

        def invert_attention_mask(self: Any, encoder_attention_mask: Tensor) -> Tensor:
            resolved_dtype = getattr(self, "dtype", torch.float32)
            mask = encoder_attention_mask.to(dtype=resolved_dtype)
            if mask.dim() == 3:
                extended = mask[:, None, :, :]
            elif mask.dim() == 2:
                extended = mask[:, None, None, :]
            else:
                shape = tuple(encoder_attention_mask.shape)
                raise ValueError(f"Unsupported encoder_attention_mask shape {shape}")
            return (1.0 - extended) * torch.finfo(resolved_dtype).min

        pre_trained_cls.invert_attention_mask = invert_attention_mask

    if not hasattr(pre_trained_cls, "get_head_mask"):

        def get_head_mask(
            self: Any,
            head_mask: Tensor | None,
            num_hidden_layers: int,
            is_attention_chunked: bool = False,
        ) -> list[Tensor | None] | Tensor:
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask.to(dtype=getattr(self, "dtype", torch.float32))

        pre_trained_cls.get_head_mask = get_head_mask


def _resolve_model_dtype(dtype: str, spec: BackboneSpec) -> torch.dtype:
    """Resolve safe model dtype, enforcing the MPS no-float16 stability invariant."""
    norm = dtype.strip().lower()
    has_mps = bool(torch.backends.mps.is_available())
    has_cuda = bool(torch.cuda.is_available())

    if norm in ("float16", "fp16"):
        if has_mps:
            raise ValueError(
                "dtype='float16' is forbidden on Apple Silicon MPS "
                "(causes NaN/Inf LoRA gradients). Use dtype='bfloat16' or dtype='float32'."
            )
        return torch.float16
    if norm in ("bfloat16", "bf16"):
        return torch.bfloat16
    if norm in ("float32", "fp32"):
        return torch.float32
    if norm == "auto":
        if spec.prefer_bf16_on_accel and (has_mps or has_cuda):
            return torch.bfloat16
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype!r}; expected one of 'auto', 'float32', 'bfloat16'.")


def _get_embedding_module(model: nn.Module) -> nn.Embedding:
    """Locate the input ``nn.Embedding`` module through PEFT or base HF wrappers."""
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        emb = get_emb()
        if isinstance(emb, nn.Embedding):
            return emb
    base = getattr(model, "base_model", None)
    if base is not None:
        base_get_emb = getattr(base, "get_input_embeddings", None)
        if callable(base_get_emb):
            emb = base_get_emb()
            if isinstance(emb, nn.Embedding):
                return emb
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    raise RuntimeError(f"Could not locate input embedding table on {type(model).__name__}.")


def get_input_embeddings_tensor(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Embed token IDs ``[B, L]`` into ``[B, L, hidden_size]`` via the backbone's input table."""
    emb_module = _get_embedding_module(model)
    out: Tensor = emb_module(input_ids.to(device=emb_module.weight.device, dtype=torch.long))
    return out


def compute_empirical_embed_rms(model: nn.Module) -> float:
    """Compute the empirical root-mean-square magnitude of the backbone's token embeddings."""
    emb_module = _get_embedding_module(model)
    weight = emb_module.weight.detach().to(dtype=torch.float32)
    row_norms = weight.pow(2).mean(dim=-1)
    valid_mask = row_norms > 0.0
    if emb_module.padding_idx is not None and 0 <= emb_module.padding_idx < weight.shape[0]:
        valid_mask[emb_module.padding_idx] = False
    if bool(valid_mask.any().item()):
        rms = torch.sqrt(row_norms[valid_mask]).mean()
    else:
        rms = torch.sqrt(row_norms.mean().clamp(min=1e-12))
    return float(rms.item())


def _infer_spec(backbone_id: str) -> BackboneSpec:
    """Look up a registered backbone spec or synthesize a sensible general-LLM fallback."""
    if backbone_id in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[backbone_id]
    lower = backbone_id.lower()
    if "safe-gpt" in lower or "safe_gpt" in lower:
        return BackboneSpec(
            backbone_id=SAFE_GPT_REPO_ID if lower in ("safe-gpt", "safe_gpt") else backbone_id,
            family="safe_gpt",
            tokenizer_id=SAFE_GPT_REPO_ID if lower in ("safe-gpt", "safe_gpt") else backbone_id,
            lora_targets=("c_attn", "c_proj", "c_fc"),
            fan_in_fan_out=True,
            is_smiles_native=True,
            prompt_prefix="",
            trust_remote_code=False,
            prefer_bf16_on_accel=True,
            uses_tuple_kv_cache=False,
            representation="safe",
        )
    if "novomolgen" in lower:
        return BackboneSpec(
            backbone_id=backbone_id,
            family="novomolgen",
            tokenizer_id=backbone_id,
            lora_targets=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            fan_in_fan_out=False,
            is_smiles_native=True,
            prompt_prefix="",
            trust_remote_code=False,
            prefer_bf16_on_accel=True,
            uses_tuple_kv_cache=False,
            representation="smiles",
        )
    if "molformer" in lower:
        return BackboneSpec(
            backbone_id=backbone_id,
            family="molformer",
            tokenizer_id=MOLFORMER_TOKENIZER_REPO_ID,
            lora_targets=("query", "key", "value", "dense"),
            fan_in_fan_out=False,
            is_smiles_native=True,
            prompt_prefix="",
            trust_remote_code=True,
            prefer_bf16_on_accel=False,
            uses_tuple_kv_cache=True,
        )
    if "gpt2" in lower:
        return BackboneSpec(
            backbone_id=backbone_id,
            family="gpt2",
            tokenizer_id=backbone_id,
            lora_targets=("c_attn", "c_proj", "c_fc"),
            fan_in_fan_out=True,
            is_smiles_native=True,
            prompt_prefix="",
            trust_remote_code=False,
            prefer_bf16_on_accel=False,
            uses_tuple_kv_cache=False,
        )
    return BackboneSpec(
        backbone_id=backbone_id,
        family="general_llm",
        tokenizer_id=backbone_id,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        fan_in_fan_out=False,
        is_smiles_native=False,
        prompt_prefix="SMILES: ",
        trust_remote_code=False,
        prefer_bf16_on_accel=True,
        uses_tuple_kv_cache=False,
    )


def _load_molformer_backbone(
    spec: BackboneSpec,
    hub_dir: Path,
    model_dtype: torch.dtype,
    local_files_only: bool,
) -> tuple[nn.Module, Any]:
    """Load ``GP-MoLFormer-Uniq`` with v5 shims, RoPE re-init, and 4D tril attention kernel."""
    _install_molformer_transformers_v5_shims()
    _install_fast_4d_tril_causal_dot_product()

    tok_file = hf_hub_download(
        repo_id=spec.tokenizer_id,
        filename="tokenizer.json",
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_file,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )

    cfg = AutoConfig.from_pretrained(
        spec.backbone_id,
        trust_remote_code=True,
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
    )
    cfg.deterministic_eval = True
    cfg.add_cross_attention = False
    cfg.is_decoder = True
    cfg.chunk_size_feed_forward = 0
    cfg.bos_token_id = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0)
    cfg.eos_token_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1)
    cfg.pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 2)

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        spec.backbone_id,
        config=cfg,
        trust_remote_code=True,
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
        dtype=model_dtype,
    )

    # CRITICAL transformers 5.x meta-init fix: re-materialize non-persistent RoPE buffers
    # without touching persisted MolformerFeatureMap.weight buffers loaded from safetensors.
    head_dim = int(cfg.hidden_size) // int(cfg.num_attention_heads)
    max_pos = int(cfg.max_position_embeddings)
    for mod in model.modules():
        cls_name = type(mod).__name__
        if cls_name == "MolformerRotaryEmbedding":
            mod.__init__(head_dim, max_position_embeddings=max_pos)  # type: ignore[misc]
        elif cls_name == "MolformerFeatureMap":
            mod.deterministic = True  # type: ignore[assignment]

    model = nn.Module.to(model, dtype=model_dtype)
    return model, tokenizer


def sanitize_safe_gpt_tokenizer_json(raw_json: str | Mapping[str, Any]) -> dict[str, Any]:
    """Strip non-standard ``safe-mol`` top-level keys that crash Rust ``tokenizers>=0.23``.

    ``datamol-io/safe-gpt/tokenizer.json`` appends ``custom_pre_tokenizer``,
    ``tokenizer_type``, and ``tokenizer_attrs`` at the JSON root, which causes strict serde
    deserialization errors in modern ``tokenizers`` releases.
    """
    parsed: dict[str, Any] = (
        dict(json.loads(raw_json)) if isinstance(raw_json, str) else dict(raw_json)
    )
    for key in SAFE_GPT_EXTRA_TOKENIZER_KEYS:
        parsed.pop(key, None)
    return parsed


def build_safe_gpt_tokenizer(
    raw_json_or_path: str | Path | Mapping[str, Any],
) -> Any:
    """Build a ``PreTrainedTokenizerFast`` for ``datamol-io/safe-gpt`` with ``SAFESplitter`` regex.

    Fixes two integration bugs without requiring ``safe-mol``:
    1. Strips ``custom_pre_tokenizer``, ``tokenizer_type``, and ``tokenizer_attrs`` before
       passing the JSON into Rust ``tokenizers.Tokenizer.from_str``.
    2. Replaces the serialized ``Whitespace`` pre-tokenizer with
       ``Split(Regex(SAFE_SPLITTER_REGEX), behavior="isolated")`` so stereocenters
       (``[C@H]``, ``[C@@H]``), halogens (``Cl``, ``Br``), and multi-digit attachment rings
       (``%10``, ``%11``) stay indivisible single tokens.
    """
    if isinstance(raw_json_or_path, Path):
        raw_payload: str | Mapping[str, Any] = raw_json_or_path.read_text(encoding="utf-8")
    elif isinstance(raw_json_or_path, str):
        stripped = raw_json_or_path.lstrip()
        if stripped.startswith("{"):
            raw_payload = raw_json_or_path
        else:
            raw_payload = Path(raw_json_or_path).read_text(encoding="utf-8")
    else:
        raw_payload = raw_json_or_path

    cleaned = sanitize_safe_gpt_tokenizer_json(raw_payload)
    tok_mod: Any = importlib.import_module("tokenizers")
    pre_tok_mod: Any = importlib.import_module("tokenizers.pre_tokenizers")

    tok_obj: Any = tok_mod.Tokenizer.from_str(json.dumps(cleaned))
    tok_obj.pre_tokenizer = pre_tok_mod.Split(
        tok_mod.Regex(SAFE_SPLITTER_REGEX),
        behavior="isolated",
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="[CLS]",
        eos_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )


def _load_safe_gpt_backbone(
    spec: BackboneSpec,
    hub_dir: Path,
    model_dtype: torch.dtype,
    local_files_only: bool,
) -> tuple[nn.Module, Any]:
    """Load ``datamol-io/safe-gpt`` directly into native ``GPT2LMHeadModel`` (no remote code)."""
    tok_file = hf_hub_download(
        repo_id=spec.tokenizer_id,
        filename="tokenizer.json",
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
    )
    tokenizer = build_safe_gpt_tokenizer(Path(tok_file))

    gpt2_cls: Any = GPT2LMHeadModel
    model: nn.Module = gpt2_cls.from_pretrained(
        spec.backbone_id,
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
        dtype=model_dtype,
    )
    model = nn.Module.to(model, dtype=model_dtype)
    return model, tokenizer


def _unpermute_interleaved_rope_weight(weight: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """Convert interleaved RoPE rows ``(H, D/2, 2, ...)`` to HF NeoX ``(H, 2, D/2, ...)``.

    ``flash_attn.models.llama`` configures ``rotary_emb_interleaved=True`` (pairing adjacent
    channels ``(2i, 2i+1)``), whereas HuggingFace ``LlamaForCausalLM`` rotates split halves
    ``(i, i + D/2)``. Permuting ``q_proj`` and ``k_proj`` output rows guarantees bit-exact
    logit parity with zero custom RoPE layers.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE unpermutation, got {head_dim}")
    trailing = weight.shape[1:]
    return (
        weight.view(num_heads, head_dim // 2, 2, *trailing)
        .transpose(1, 2)
        .reshape(num_heads * head_dim, *trailing)
        .contiguous()
    )


def remap_novomolgen_state_dict(
    raw_state_dict: Mapping[str, Tensor],
    *,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> dict[str, Tensor]:
    """Remap ``flash_attn`` ``GPTLMHeadModel`` weights into native ``LlamaForCausalLM`` keys.

    Handles:
    - ``transformer.embeddings.word_embeddings.weight`` -> ``model.embed_tokens.weight``
    - ``transformer.layers.{i}.mixer.Wqkv.weight`` -> ``q_proj`` / ``k_proj`` / ``v_proj`` with
      interleaved-to-NeoX RoPE unpermutation on ``q_proj`` and ``k_proj``
    - ``transformer.layers.{i}.mixer.out_proj.weight`` -> ``self_attn.o_proj.weight``
    - ``transformer.layers.{i}.norm{1,2}.weight`` -> ``{input,post_attention}_layernorm.weight``
    - ``transformer.layers.{i}.mlp.fc1.weight`` (``flash_attn`` ``GatedMlp`` layout
      ``[up_proj, gate_proj]``) -> ``mlp.up_proj.weight`` and ``mlp.gate_proj.weight``
    - ``transformer.layers.{i}.mlp.fc2.weight`` -> ``mlp.down_proj.weight``
    - ``transformer.ln_f.weight`` -> ``model.norm.weight``, ``lm_head.weight`` -> ``lm_head.weight``
    """
    if "model.embed_tokens.weight" in raw_state_dict:
        return dict(raw_state_dict)

    embed_w = raw_state_dict["transformer.embeddings.word_embeddings.weight"]
    mapped: dict[str, Tensor] = {
        "model.embed_tokens.weight": embed_w,
        "model.norm.weight": raw_state_dict["transformer.ln_f.weight"],
        "lm_head.weight": raw_state_dict.get("lm_head.weight", embed_w),
    }

    q_dim = int(num_attention_heads) * int(head_dim)
    kv_dim = int(num_key_value_heads) * int(head_dim)

    for layer_idx in range(int(num_hidden_layers)):
        src_prefix = f"transformer.layers.{layer_idx}"
        dst_prefix = f"model.layers.{layer_idx}"

        wqkv = raw_state_dict[f"{src_prefix}.mixer.Wqkv.weight"]
        if int(wqkv.shape[0]) != q_dim + 2 * kv_dim:
            raise ValueError(
                f"Unexpected Wqkv shape {tuple(wqkv.shape)} at layer {layer_idx}; "
                f"expected first dim {q_dim + 2 * kv_dim} (q={q_dim}, kv={kv_dim})."
            )
        q_w, k_w, v_w = torch.split(wqkv, [q_dim, kv_dim, kv_dim], dim=0)
        mapped[f"{dst_prefix}.self_attn.q_proj.weight"] = _unpermute_interleaved_rope_weight(
            q_w, num_attention_heads, head_dim
        )
        mapped[f"{dst_prefix}.self_attn.k_proj.weight"] = _unpermute_interleaved_rope_weight(
            k_w, num_key_value_heads, head_dim
        )
        mapped[f"{dst_prefix}.self_attn.v_proj.weight"] = v_w.contiguous()
        mapped[f"{dst_prefix}.self_attn.o_proj.weight"] = raw_state_dict[
            f"{src_prefix}.mixer.out_proj.weight"
        ]

        qkv_bias_key = f"{src_prefix}.mixer.Wqkv.bias"
        if qkv_bias_key in raw_state_dict:
            q_b, k_b, v_b = torch.split(raw_state_dict[qkv_bias_key], [q_dim, kv_dim, kv_dim], 0)
            mapped[f"{dst_prefix}.self_attn.q_proj.bias"] = _unpermute_interleaved_rope_weight(
                q_b, num_attention_heads, head_dim
            )
            mapped[f"{dst_prefix}.self_attn.k_proj.bias"] = _unpermute_interleaved_rope_weight(
                k_b, num_key_value_heads, head_dim
            )
            mapped[f"{dst_prefix}.self_attn.v_proj.bias"] = v_b.contiguous()
        o_bias_key = f"{src_prefix}.mixer.out_proj.bias"
        if o_bias_key in raw_state_dict:
            mapped[f"{dst_prefix}.self_attn.o_proj.bias"] = raw_state_dict[o_bias_key]

        mapped[f"{dst_prefix}.input_layernorm.weight"] = raw_state_dict[
            f"{src_prefix}.norm1.weight"
        ]
        mapped[f"{dst_prefix}.post_attention_layernorm.weight"] = raw_state_dict[
            f"{src_prefix}.norm2.weight"
        ]

        # flash_attn.modules.mlp.GatedMlp splits fc1(x) into (x_up, gate) -> x_up * silu(gate).
        fc1_w = raw_state_dict[f"{src_prefix}.mlp.fc1.weight"]
        up_w, gate_w = torch.chunk(fc1_w, 2, dim=0)
        mapped[f"{dst_prefix}.mlp.up_proj.weight"] = up_w.contiguous()
        mapped[f"{dst_prefix}.mlp.gate_proj.weight"] = gate_w.contiguous()
        mapped[f"{dst_prefix}.mlp.down_proj.weight"] = raw_state_dict[
            f"{src_prefix}.mlp.fc2.weight"
        ]

        fc1_bias_key = f"{src_prefix}.mlp.fc1.bias"
        if fc1_bias_key in raw_state_dict:
            up_b, gate_b = torch.chunk(raw_state_dict[fc1_bias_key], 2, dim=0)
            mapped[f"{dst_prefix}.mlp.up_proj.bias"] = up_b.contiguous()
            mapped[f"{dst_prefix}.mlp.gate_proj.bias"] = gate_b.contiguous()
        fc2_bias_key = f"{src_prefix}.mlp.fc2.bias"
        if fc2_bias_key in raw_state_dict:
            mapped[f"{dst_prefix}.mlp.down_proj.bias"] = raw_state_dict[fc2_bias_key]

    return mapped


def _load_novomolgen_backbone(
    spec: BackboneSpec,
    hub_dir: Path,
    model_dtype: torch.dtype,
    local_files_only: bool,
) -> tuple[nn.Module, Any]:
    """Load ``NovoMolGen`` into native ``LlamaForCausalLM`` (MPS SDPA, zero ``flash_attn``)."""
    cfg_file = hf_hub_download(
        repo_id=spec.backbone_id,
        filename="config.json",
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
    )
    tok_file = hf_hub_download(
        repo_id=spec.tokenizer_id,
        filename="tokenizer.json",
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
    )
    weights_file = hf_hub_download(
        repo_id=spec.backbone_id,
        filename="pytorch_model.bin",
        cache_dir=str(hub_dir),
        local_files_only=local_files_only,
    )

    cfg_raw: dict[str, Any] = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_file,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    hidden_size = int(cfg_raw["hidden_size"])
    num_heads = int(cfg_raw["num_attention_heads"])
    num_kv_heads = int(cfg_raw.get("num_key_value_heads", num_heads))
    head_dim = int(cfg_raw.get("head_dim", hidden_size // num_heads))
    num_layers = int(cfg_raw["num_hidden_layers"])

    bos_id = int(
        cfg_raw.get(
            "bos_token_id",
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2,
        )
    )
    eos_id = int(
        cfg_raw.get(
            "eos_token_id",
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 3,
        )
    )
    pad_id = int(
        cfg_raw.get(
            "pad_token_id",
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1,
        )
    )

    llama_cfg_cls: Any = LlamaConfig
    llama_cfg: Any = llama_cfg_cls(
        vocab_size=int(cfg_raw["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=int(cfg_raw["intermediate_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_act=str(cfg_raw.get("hidden_act", "silu")),
        max_position_embeddings=int(cfg_raw.get("max_position_embeddings", 2048)),
        rms_norm_eps=float(cfg_raw.get("rms_norm_eps", 1e-6)),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(cfg_raw.get("rope_theta", 10000.0)),
        },
        attention_bias=bool(cfg_raw.get("attention_bias", False)),
        mlp_bias=bool(cfg_raw.get("mlp_bias", False)),
        tie_word_embeddings=bool(cfg_raw.get("tie_word_embeddings", False)),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
    )
    llama_cfg._attn_implementation = "sdpa"

    raw_sd = torch.load(weights_file, map_location="cpu", weights_only=True)
    if not isinstance(raw_sd, Mapping):
        raise TypeError(f"Expected state_dict mapping in {weights_file}, got {type(raw_sd)}")

    mapped_sd = remap_novomolgen_state_dict(
        raw_sd,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
    )
    llama_cls: Any = LlamaForCausalLM
    model: nn.Module = llama_cls(llama_cfg)
    model.load_state_dict(mapped_sd, strict=True)
    model = nn.Module.to(model, dtype=model_dtype)
    return model, tokenizer


def _load_mock_backbone(model_dtype: torch.dtype) -> tuple[nn.Module, MockSmilesTokenizer]:
    """Build an offline 2-layer tiny GPT-2 + SMILES regex tokenizer with zero I/O."""
    cfg = GPT2Config(
        vocab_size=96,
        n_positions=256,
        n_embd=64,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model: nn.Module = GPT2LMHeadModel(cfg)
    model = nn.Module.to(model, dtype=model_dtype)
    tokenizer = MockSmilesTokenizer(vocab_size=96)
    return model, tokenizer


def _apply_lora_adapter(
    model: nn.Module,
    spec: BackboneSpec,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> nn.Module:
    """Wrap a causal LM backbone with PEFT LoRA targeting ``spec.lora_targets``."""
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(spec.lora_targets),
        fan_in_fan_out=bool(spec.fan_in_fan_out),
        bias="none",
    )
    peft_model: nn.Module = get_peft_model(model, lora_cfg)
    return peft_model


def load_hf_backbone(
    backbone_id: str,
    cache_dir: str = DEFAULT_HF_CACHE_DIR,
    dtype: str = "auto",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_lora: bool = True,
    local_files_only: bool = False,
) -> LoadedBackbone:
    """Load a registered chemical transformer, compact general LLM, or offline mock backbone.

    Args:
        backbone_id: HuggingFace repo ID (e.g. ``"ibm-research/GP-MoLFormer-Uniq"``,
            ``"entropy/gpt2_zinc_87m"``, ``"Qwen/Qwen2.5-0.5B"``, ``"HuggingFaceTB/SmolLM2-360M"``,
            ``"Qwen/Qwen3-0.6B-Base"``) or ``"mock-tiny-gpt2"``.
        cache_dir: Local HF cache directory (defaults to ``.hf_cache.nosync`` so iCloud never
            evicts cached weights).
        dtype: ``"auto"``, ``"float32"``, or ``"bfloat16"``. ``"float16"`` is rejected on MPS.
        lora_r: LoRA rank (used when ``use_lora=True``).
        lora_alpha: LoRA scaling alpha.
        lora_dropout: LoRA dropout probability.
        use_lora: Whether to wrap the backbone with PEFT LoRA adapters.
        local_files_only: If ``True``, strictly load from ``cache_dir`` without network access.

    Returns:
        ``LoadedBackbone`` dataclass containing the model, tokenizer, empirical ``embed_rms``,
        special token IDs, resolved dtype, and chemical vs. general-LLM metadata.
    """
    spec = _infer_spec(backbone_id)
    model_dtype = _resolve_model_dtype(dtype, spec)

    model: nn.Module
    tokenizer: Any
    if spec.family == "mock":
        model, tokenizer = _load_mock_backbone(model_dtype=model_dtype)
    else:
        hub_dir = _configure_hf_cache(cache_dir)
        if spec.family == "molformer":
            model, tokenizer = _load_molformer_backbone(
                spec=spec,
                hub_dir=hub_dir,
                model_dtype=model_dtype,
                local_files_only=local_files_only,
            )
        elif spec.family == "safe_gpt":
            model, tokenizer = _load_safe_gpt_backbone(
                spec=spec,
                hub_dir=hub_dir,
                model_dtype=model_dtype,
                local_files_only=local_files_only,
            )
        elif spec.family == "novomolgen":
            model, tokenizer = _load_novomolgen_backbone(
                spec=spec,
                hub_dir=hub_dir,
                model_dtype=model_dtype,
                local_files_only=local_files_only,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.tokenizer_id,
                trust_remote_code=spec.trust_remote_code,
                cache_dir=str(hub_dir),
                local_files_only=local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                spec.backbone_id,
                trust_remote_code=spec.trust_remote_code,
                cache_dir=str(hub_dir),
                local_files_only=local_files_only,
                dtype=model_dtype,
            )
            model = nn.Module.to(model, dtype=model_dtype)

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError(f"Loaded backbone {backbone_id!r} does not expose a .config attribute.")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(cfg, "eos_token_id", None)
    if eos_id is None:
        raise RuntimeError(f"Backbone {backbone_id!r} tokenizer defines no eos_token_id.")
    eos_token_id = int(eos_id)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(cfg, "bos_token_id", None)
    bos_token_id = int(bos_id) if bos_id is not None else eos_token_id

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(cfg, "pad_token_id", None)
    if pad_id is None:
        pad_token_id = eos_token_id
        if getattr(tokenizer, "eos_token", None) is not None and hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(tokenizer, "pad_token_id"):
            tokenizer.pad_token_id = pad_token_id
    else:
        pad_token_id = int(pad_id)
    cfg.pad_token_id = pad_token_id

    hidden_raw = getattr(cfg, "hidden_size", None)
    if hidden_raw is None:
        hidden_raw = getattr(cfg, "n_embd", None)
    if hidden_raw is None:
        raise RuntimeError(
            f"Backbone {backbone_id!r} config defines neither hidden_size nor n_embd"
        )
    hidden_size = int(hidden_raw)
    vocab_size = int(getattr(cfg, "vocab_size", len(tokenizer)))
    embed_rms = compute_empirical_embed_rms(model)

    if use_lora:
        model = _apply_lora_adapter(
            model,
            spec,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    return LoadedBackbone(
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embed_rms=embed_rms,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        model_dtype=model_dtype,
        is_smiles_native=spec.is_smiles_native,
        prompt_prefix=spec.prompt_prefix,
        backbone_id=spec.backbone_id,
        uses_tuple_kv_cache=spec.uses_tuple_kv_cache,
        representation=spec.representation,
    )


def register_backbone(
    spec: BackboneSpec,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    params: str | None = None,
    license_name: str | None = None,
    notes: str = "",
) -> BackboneSpec:
    """Register a custom Hugging Face or local causal LM backbone at runtime.

    Updates both :data:`BACKBONE_REGISTRY` (used by :func:`load_hf_backbone`) and
    the zero-PyTorch :data:`geneuclidean.registry.BACKBONE_CATALOG` (returned by
    :func:`geneuclidean.list_backbones`).
    """
    from geneuclidean.exceptions import RegistryError
    from geneuclidean.registry import register_backbone_metadata

    keys = [spec.backbone_id, *[a.strip() for a in aliases if a.strip()]]
    if not override:
        for k in keys:
            if k in BACKBONE_REGISTRY:
                raise RegistryError(
                    f"Backbone {k!r} is already registered in BACKBONE_REGISTRY.",
                    hint="Pass override=True to replace an existing backbone specification.",
                )
    for k in keys:
        BACKBONE_REGISTRY[k] = spec

    meta: dict[str, Any] = {
        "family": spec.family,
        "representation": spec.representation,
        "is_smiles_native": spec.is_smiles_native,
        "tokenizer_id": spec.tokenizer_id,
    }
    if aliases:
        meta["aliases"] = [a.strip() for a in aliases if a.strip()]
    if params is not None:
        meta["params"] = params
    if license_name is not None:
        meta["license"] = license_name
    if notes:
        meta["notes"] = notes
    register_backbone_metadata(spec.backbone_id, meta, override=True)
    return spec


def list_registered_backbones() -> Mapping[str, BackboneSpec]:
    """Return a read-only view of all registered GenEuclidean backbones."""
    return dict(BACKBONE_REGISTRY)


__all__ = [
    "BACKBONE_REGISTRY",
    "DEFAULT_HF_CACHE_DIR",
    "MOCK_BACKBONE_ID",
    "MOLFORMER_REPO_ID",
    "MOLFORMER_TOKENIZER_REPO_ID",
    "NOVOMOLGEN_157M_ALIAS",
    "NOVOMOLGEN_157M_REPO_ID",
    "NOVOMOLGEN_300M_REPO_ID",
    "SAFE_GPT_ALIAS",
    "SAFE_GPT_EXTRA_TOKENIZER_KEYS",
    "SAFE_GPT_REPO_ID",
    "BackboneSpec",
    "LoadedBackbone",
    "MockSmilesTokenizer",
    "build_safe_gpt_tokenizer",
    "compute_empirical_embed_rms",
    "get_input_embeddings_tensor",
    "list_registered_backbones",
    "load_hf_backbone",
    "register_backbone",
    "remap_novomolgen_state_dict",
    "sanitize_safe_gpt_tokenizer_json",
]
