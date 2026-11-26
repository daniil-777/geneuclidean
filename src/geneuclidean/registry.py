"""Thread-safe generic plugin registries & entry-point discovery for GenEuclidean.

Architecture notes:
    1. **Zero eager heavy imports:** This module depends strictly on the Python
       standard library so bare ``import geneuclidean as ge`` remains under ~15 ms
       and never imports ``torch``, ``transformers``, ``rdkit``, or ``matplotlib``.
    2. **Thread safety:** All mutations and snapshots acquire an internal ``RLock``
       so concurrent test workers or multi-threaded inference services cannot corrupt
       registry tables.
    3. **Clean scoped overrides:** :meth:`Registry.temporary` provides a deterministic
       context manager for experiments, ablations, and unit tests that automatically
       restores previous state even if an exception is raised.
    4. **Third-party plugin discovery:** :func:`discover_plugins` inspects standard
       Python ``importlib.metadata.entry_points`` groups (``geneuclidean.encoders``,
       ``geneuclidean.decoders``, ``geneuclidean.sampling_strategies``,
       ``geneuclidean.scorers``, ``geneuclidean.filters``) so external pip packages
       can extend GenEuclidean automatically on install.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from threading import RLock
from typing import Any, Generic, TypeVar, overload

from geneuclidean.exceptions import RegistryError

T = TypeVar("T")

PLUGIN_ENTRY_POINT_GROUPS: dict[str, str] = {
    "encoders": "geneuclidean.encoders",
    "decoders": "geneuclidean.decoders",
    "sampling_strategies": "geneuclidean.sampling_strategies",
    "scorers": "geneuclidean.scorers",
    "filters": "geneuclidean.filters",
}


class Registry(Generic[T]):  # noqa: UP046  # Keep Python 3.11 runtime compatibility
    """Thread-safe name/alias registry with decorator registration and scoped overrides.

    Examples:
        >>> reg: Registry[Any] = Registry("demo")
        >>> @reg.register("double", aliases=("2x",))
        ... def _double(x: int) -> int:
        ...     return x * 2
        >>> reg.get("2x")(21)
        42
    """

    def __init__(self, namespace: str, *, case_sensitive: bool = False) -> None:
        self.namespace = namespace
        self.case_sensitive = case_sensitive
        self._lock = RLock()
        self._items: dict[str, T] = {}
        self._canonical_names: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def _normalize(self, name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise RegistryError(f"Registry {self.namespace!r} does not accept empty names.")
        return cleaned if self.case_sensitive else cleaned.lower()

    @overload
    def register(
        self,
        name: str,
        obj: T,
        *,
        aliases: Sequence[str] = (),
        override: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> T: ...

    @overload
    def register(
        self,
        name: str,
        obj: None = None,
        *,
        aliases: Sequence[str] = (),
        override: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Callable[[T], T]: ...

    def register(
        self,
        name: str,
        obj: T | None = None,
        *,
        aliases: Sequence[str] = (),
        override: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> T | Callable[[T], T]:
        """Register an object under ``name`` (or return a decorator when ``obj is None``).

        Args:
            name: Primary canonical identifier for the registered component.
            obj: Component/factory to register. If omitted, returns a decorator.
            aliases: Optional secondary names resolving to ``name``.
            override: If ``False`` (default), raises :class:`RegistryError` when
                ``name`` or any alias is already registered.
            metadata: Optional JSON-friendly descriptive metadata dictionary.

        Returns:
            The registered object (or decorator wrapping the registered object).
        """
        if obj is None:

            def _decorator(target: T) -> T:
                return self.register(
                    name,
                    target,
                    aliases=aliases,
                    override=override,
                    metadata=metadata,
                )

            return _decorator

        key = self._normalize(name)
        alias_keys = [self._normalize(a) for a in aliases if a.strip()]

        with self._lock:
            if not override and (key in self._items or key in self._aliases):
                raise RegistryError(
                    f"{name!r} is already registered in {self.namespace!r}.",
                    hint="Pass override=True to intentionally replace an existing entry.",
                )
            for ak in alias_keys:
                if ak == key:
                    continue
                if not override and (ak in self._items or ak in self._aliases):
                    raise RegistryError(
                        f"Alias {ak!r} collides with an existing entry in {self.namespace!r}.",
                        hint="Choose a unique alias or pass override=True.",
                    )

            # If overriding an existing canonical key, detach stale aliases pointing to it.
            if key in self._items:
                stale = [ak for ak, target_k in self._aliases.items() if target_k == key]
                for ak in stale:
                    del self._aliases[ak]

            self._items[key] = obj
            self._canonical_names[key] = name.strip()
            self._metadata[key] = dict(metadata) if metadata is not None else {}
            for ak in alias_keys:
                if ak != key:
                    self._aliases[ak] = key
        return obj

    def unregister(self, name: str) -> T:
        """Remove a registered entry (and all of its aliases) from the registry."""
        key = self._normalize(name)
        with self._lock:
            canonical_key = self._aliases.get(key, key)
            if canonical_key not in self._items:
                raise RegistryError(
                    f"{name!r} is not registered in {self.namespace!r}.",
                    hint=f"Registered entries: {', '.join(self.names()) or '(none)'}",
                )
            removed = self._items.pop(canonical_key)
            self._canonical_names.pop(canonical_key, None)
            self._metadata.pop(canonical_key, None)
            for ak in [k for k, target_k in self._aliases.items() if target_k == canonical_key]:
                del self._aliases[ak]
            return removed

    def resolve_key(self, name: str) -> str:
        """Resolve a primary name or alias to its canonical registered display name."""
        key = self._normalize(name)
        with self._lock:
            canonical_key = self._aliases.get(key, key)
            if canonical_key not in self._items:
                available = ", ".join(self.names()) or "(none)"
                raise RegistryError(
                    f"Unknown {self.namespace} {name!r}. Available: {available}.",
                    hint=f"Register custom plugins via the {self.namespace} registry.",
                )
            return self._canonical_names[canonical_key]

    def get(self, name: str) -> T:
        """Look up a registered component by canonical name or alias."""
        key = self._normalize(name)
        with self._lock:
            canonical_key = self._aliases.get(key, key)
            if canonical_key not in self._items:
                available = ", ".join(self.names()) or "(none)"
                raise RegistryError(
                    f"Unknown {self.namespace} {name!r}. Available: {available}.",
                    hint=f"Register custom plugins via the {self.namespace} registry.",
                )
            return self._items[canonical_key]

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return a shallow copy of metadata attached to ``name``."""
        key = self._normalize(name)
        with self._lock:
            canonical_key = self._aliases.get(key, key)
            if canonical_key not in self._items:
                raise RegistryError(f"Unknown {self.namespace} {name!r}.")
            return dict(self._metadata.get(canonical_key, {}))

    def names(self) -> list[str]:
        """Return sorted canonical display names of all registered entries."""
        with self._lock:
            return sorted(self._canonical_names.values())

    def aliases(self) -> dict[str, str]:
        """Return mapping of normalized alias -> canonical display name."""
        with self._lock:
            return {
                ak: self._canonical_names[target_k]
                for ak, target_k in sorted(self._aliases.items())
            }

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-serializable overview of all entries, aliases, and metadata."""
        with self._lock:
            alias_by_key: dict[str, list[str]] = {}
            for ak, target_k in sorted(self._aliases.items()):
                alias_by_key.setdefault(target_k, []).append(ak)
            return {
                self._canonical_names[k]: {
                    "aliases": alias_by_key.get(k, []),
                    **dict(self._metadata.get(k, {})),
                }
                for k in sorted(self._items.keys())
            }

    @contextmanager
    def temporary(
        self,
        name: str,
        obj: T,
        *,
        aliases: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[T]:
        """Temporarily register ``obj`` for the duration of a ``with`` block.

        Restores the exact prior entry (if any) or removes the temporary entry on exit,
        even when the body raises an exception. Ideal for ablations and isolated tests.
        """
        key = self._normalize(name)
        with self._lock:
            had_prev = key in self._items
            prev_obj = self._items.get(key)
            prev_name = self._canonical_names.get(key)
            prev_meta = dict(self._metadata.get(key, {})) if had_prev else None
            prev_aliases = [ak for ak, target_k in self._aliases.items() if target_k == key]

        self.register(name, obj, aliases=aliases, override=True, metadata=metadata)
        try:
            yield obj
        finally:
            with self._lock:
                if key in self._items:
                    self.unregister(name)
                if had_prev and prev_obj is not None and prev_name is not None:
                    self.register(
                        prev_name,
                        prev_obj,
                        aliases=prev_aliases,
                        override=True,
                        metadata=prev_meta,
                    )

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str) or not name.strip():
            return False
        key = self._normalize(name)
        with self._lock:
            return key in self._items or key in self._aliases

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry(namespace={self.namespace!r}, entries={self.names()!r})"


# ---------------------------------------------------------------------------
# Built-in Global Extension Registries (Zero Heavy Dependency Overhead)
# ---------------------------------------------------------------------------

BUILTIN_ENCODER_NAMES: frozenset[str] = frozenset(
    {
        "painn",
        "egnn",
        "e3nn",
        "schnet",
        "dense_painn_se3",
        "cartformer_se3",
        "atomgs_se3",
        "subpocket_vn_jepa",
    }
)
BUILTIN_ENCODER_ARCHS: frozenset[str] = BUILTIN_ENCODER_NAMES
BUILTIN_ENCODER_ALIASES: dict[str, str] = {
    "cartformer": "cartformer_se3",
    "atomgs": "atomgs_se3",
    "subpocket_vn": "subpocket_vn_jepa",
    "vn_jepa": "subpocket_vn_jepa",
    "dense_painn": "dense_painn_se3",
}
BUILTIN_DECODER_NAMES: frozenset[str] = frozenset({"transformer", "lstm"})
BUILTIN_SAMPLING_STRATEGY_NAMES: frozenset[str] = frozenset(
    {"min_p", "top_p", "top_k", "temperature", "greedy", "typical", "diverse_beam"}
)
BUILTIN_FILTER_NAMES: frozenset[str] = frozenset(
    {"admet", "basic_amine", "mpo", "qed", "ring_strain", "scaffold_hop", "triage"}
)

ENCODER_REGISTRY: Registry[Callable[..., Any]] = Registry("encoder")
DECODER_REGISTRY: Registry[Callable[..., Any]] = Registry("decoder")
SAMPLING_STRATEGY_REGISTRY: Registry[Callable[..., Any]] = Registry("sampling_strategy")
SCORER_REGISTRY: Registry[Any] = Registry("scorer")
FILTER_REGISTRY: Registry[Any] = Registry("filter")

# Authoritative zero-PyTorch metadata catalog for supported chemical & local LLMs.
# Synchronized automatically whenever ``hf_backbones.register_backbone`` is called.
BACKBONE_CATALOG: dict[str, dict[str, Any]] = {
    "ibm-research/GP-MoLFormer-Uniq": {
        "backbone_id": "ibm-research/GP-MoLFormer-Uniq",
        "family": "molformer",
        "params": "46.8M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "ibm-research/MoLFormer-XL-both-10pct",
        "notes": "Primary GenEuclidean v3.0/v3.1/v4.0 chemical foundation causal transformer.",
    },
    "datamol-io/safe-gpt": {
        "backbone_id": "datamol-io/safe-gpt",
        "family": "safe_gpt",
        "params": "87.3M",
        "license": "Apache-2.0",
        "representation": "safe",
        "is_smiles_native": True,
        "tokenizer_id": "datamol-io/safe-gpt",
        "aliases": ["safe-gpt"],
        "notes": "Fragment-modular SAFE generative model trained on 1.17B molecules.",
    },
    "chandar-lab/NovoMolGen_157M_SMILES_BPE": {
        "backbone_id": "chandar-lab/NovoMolGen_157M_SMILES_BPE",
        "family": "novomolgen",
        "params": "157M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "chandar-lab/NovoMolGen_157M_SMILES_BPE",
        "aliases": ["novomolgen-157m"],
        "notes": "Modern Llama-style SMILES BPE molecular foundation model (157M, MPS SDPA).",
    },
    "chandar-lab/NovoMolGen_300M_SMILES_BPE": {
        "backbone_id": "chandar-lab/NovoMolGen_300M_SMILES_BPE",
        "family": "novomolgen",
        "params": "300M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "chandar-lab/NovoMolGen_300M_SMILES_BPE",
        "notes": "Modern Llama-style SMILES BPE molecular foundation model (300M, MPS SDPA).",
    },
    "entropy/gpt2_zinc_87m": {
        "backbone_id": "entropy/gpt2_zinc_87m",
        "family": "gpt2",
        "params": "87.3M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "entropy/gpt2_zinc_87m",
        "notes": "Native GPT-2 ZINC SMILES causal language model fallback.",
    },
    "Qwen/Qwen2.5-0.5B": {
        "backbone_id": "Qwen/Qwen2.5-0.5B",
        "family": "general_llm",
        "params": "494M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": False,
        "tokenizer_id": "Qwen/Qwen2.5-0.5B",
        "notes": "Compact general-purpose local LLM (bfloat16 on MPS/CUDA).",
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "backbone_id": "HuggingFaceTB/SmolLM2-360M",
        "family": "general_llm",
        "params": "362M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": False,
        "tokenizer_id": "HuggingFaceTB/SmolLM2-360M",
        "notes": "Ultra-compact local LLM (bfloat16 on MPS/CUDA).",
    },
    "Qwen/Qwen3-0.6B-Base": {
        "backbone_id": "Qwen/Qwen3-0.6B-Base",
        "family": "general_llm",
        "params": "596M",
        "license": "Apache-2.0",
        "representation": "smiles",
        "is_smiles_native": False,
        "tokenizer_id": "Qwen/Qwen3-0.6B-Base",
        "notes": "2025 compact Qwen3 base LLM (bfloat16 on MPS/CUDA).",
    },
    "mock-tiny-gpt2": {
        "backbone_id": "mock-tiny-gpt2",
        "family": "mock",
        "params": "<1M",
        "license": "MIT",
        "representation": "smiles",
        "is_smiles_native": True,
        "tokenizer_id": "mock-tiny-gpt2",
        "notes": "Zero-download offline 2-layer GPT-2 + SMILES tokenizer for CI and tutorials.",
    },
}


# ---------------------------------------------------------------------------
# Ergonomic Registration & Introspection Helpers
# ---------------------------------------------------------------------------


def register_encoder(
    name: str,
    factory: Callable[..., Any] | None = None,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Register a custom 3D pocket encoder factory callable ``(cfg, max_neighbors) -> nn.Module``.

    Can be used directly or as a decorator::

        @ge.register_encoder("my_se3_net")
        def build_my_se3_net(cfg, max_neighbors=24):
            return MySE3Encoder(cfg.node_in_dim, cfg.out_dim)
    """
    return ENCODER_REGISTRY.register(
        name, factory, aliases=aliases, override=override, metadata=metadata
    )


def list_encoders() -> list[str]:
    """Return sorted names of all built-in and custom registered 3D pocket encoders.

    Executes without importing PyTorch by unioning :data:`BUILTIN_ENCODER_NAMES`
    with any custom encoders registered at runtime.
    """
    return sorted(BUILTIN_ENCODER_NAMES | set(ENCODER_REGISTRY.names()))


def register_decoder(
    name: str,
    factory: Callable[..., Any] | None = None,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Register a custom molecular decoder factory ``(cfg, cross_attn_gate_init) -> nn.Module``."""
    return DECODER_REGISTRY.register(
        name, factory, aliases=aliases, override=override, metadata=metadata
    )


def list_decoders() -> list[str]:
    """Return sorted names of all built-in and custom registered molecular decoders."""
    return sorted(BUILTIN_DECODER_NAMES | set(DECODER_REGISTRY.names()))


def register_sampling_strategy(
    name: str,
    fn: Callable[..., Any] | None = None,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Register a custom next-token probability shaping function.

    The registered callable must satisfy :class:`~geneuclidean.protocols.SamplingStrategyProtocol`:
    ``fn(logits, *, temperature=1.0, top_k=30, top_p=0.92, min_p=0.08) -> probs[B, V]``.
    """
    return SAMPLING_STRATEGY_REGISTRY.register(
        name, fn, aliases=aliases, override=override, metadata=metadata
    )


def list_sampling_strategies() -> list[str]:
    """Return sorted names of all built-in and custom registered decoding strategies."""
    return sorted(BUILTIN_SAMPLING_STRATEGY_NAMES | set(SAMPLING_STRATEGY_REGISTRY.names()))


def register_scorer(
    name: str,
    scorer: Any | None = None,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Register a custom molecular scorer satisfying :class:`MoleculeScorerProtocol`."""
    return SCORER_REGISTRY.register(
        name, scorer, aliases=aliases, override=override, metadata=metadata
    )


def list_scorers() -> list[str]:
    """Return sorted names of all registered molecular scorers."""
    return SCORER_REGISTRY.names()


def register_filter(
    name: str,
    mol_filter: Any | None = None,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Register a custom molecular filter satisfying :class:`MoleculeFilterProtocol`."""
    return FILTER_REGISTRY.register(
        name, mol_filter, aliases=aliases, override=override, metadata=metadata
    )


def list_filters() -> list[str]:
    """Return sorted names of all built-in and custom registered molecular filters."""
    return sorted(BUILTIN_FILTER_NAMES | set(FILTER_REGISTRY.names()))


def register_backbone_metadata(
    backbone_id: str,
    metadata: Mapping[str, Any],
    *,
    override: bool = False,
) -> dict[str, Any]:
    """Record zero-PyTorch metadata for a custom chemical or general LLM backbone."""
    bid = backbone_id.strip()
    if not bid:
        raise RegistryError("backbone_id cannot be empty.")
    if bid in BACKBONE_CATALOG and not override:
        raise RegistryError(
            f"Backbone {bid!r} already exists in BACKBONE_CATALOG.",
            hint="Pass override=True to update its catalog metadata.",
        )
    entry = {"backbone_id": bid, **dict(metadata)}
    BACKBONE_CATALOG[bid] = entry
    return dict(entry)


def _registry_for_kind(kind: str) -> Registry[Any]:
    mapping: dict[str, Registry[Any]] = {
        "encoders": ENCODER_REGISTRY,
        "decoders": DECODER_REGISTRY,
        "sampling_strategies": SAMPLING_STRATEGY_REGISTRY,
        "scorers": SCORER_REGISTRY,
        "filters": FILTER_REGISTRY,
    }
    if kind not in mapping:
        raise RegistryError(
            f"Unknown plugin group kind {kind!r}. Expected one of {sorted(mapping)}."
        )
    return mapping[kind]


def discover_plugins(
    kinds: Sequence[str] | None = None,
    *,
    override: bool = False,
) -> dict[str, list[str]]:
    """Discover and load external third-party plugins via ``importlib.metadata.entry_points``.

    Third-party packages can expose extensions in their ``pyproject.toml``::

        [project.entry-points."geneuclidean.encoders"]
        mace = "my_pkg.encoders:build_mace_encoder"

    Args:
        kinds: Optional subset of plugin kinds to load (defaults to all groups in
            :data:`PLUGIN_ENTRY_POINT_GROUPS`).
        override: Whether discovered entry points may replace already registered names.

    Returns:
        Dictionary mapping each queried plugin kind to the list of newly loaded names.
    """
    selected = list(kinds) if kinds is not None else list(PLUGIN_ENTRY_POINT_GROUPS.keys())
    loaded: dict[str, list[str]] = {}
    for kind in selected:
        if kind not in PLUGIN_ENTRY_POINT_GROUPS:
            raise RegistryError(
                f"Unknown plugin kind {kind!r}. Valid kinds: {sorted(PLUGIN_ENTRY_POINT_GROUPS)}."
            )
        group = PLUGIN_ENTRY_POINT_GROUPS[kind]
        reg = _registry_for_kind(kind)
        eps = importlib_metadata.entry_points(group=group)
        names_loaded: list[str] = []
        for ep in eps:
            if ep.name in reg and not override:
                continue
            plugin_obj = ep.load()
            reg.register(
                ep.name,
                plugin_obj,
                override=override,
                metadata={"source": "entry_point", "group": group, "value": ep.value},
            )
            names_loaded.append(ep.name)
        loaded[kind] = names_loaded
    return loaded


def ensure_builtin_registry_loaded(kind: str) -> None:
    """Lazily import the owning subsystem module to populate built-in factory registrations."""
    module_map = {
        "encoders": "geneuclidean.models.encoders",
        "decoders": "geneuclidean.models.decoders",
        "sampling_strategies": "geneuclidean.sampling.decoder_strategies",
    }
    mod_path = module_map.get(kind)
    if mod_path is not None:
        importlib.import_module(mod_path)


__all__ = [
    "BACKBONE_CATALOG",
    "BUILTIN_DECODER_NAMES",
    "BUILTIN_ENCODER_ALIASES",
    "BUILTIN_ENCODER_ARCHS",
    "BUILTIN_ENCODER_NAMES",
    "BUILTIN_FILTER_NAMES",
    "BUILTIN_SAMPLING_STRATEGY_NAMES",
    "DECODER_REGISTRY",
    "ENCODER_REGISTRY",
    "FILTER_REGISTRY",
    "PLUGIN_ENTRY_POINT_GROUPS",
    "SAMPLING_STRATEGY_REGISTRY",
    "SCORER_REGISTRY",
    "Registry",
    "discover_plugins",
    "ensure_builtin_registry_loaded",
    "list_decoders",
    "list_encoders",
    "list_filters",
    "list_sampling_strategies",
    "list_scorers",
    "register_backbone_metadata",
    "register_decoder",
    "register_encoder",
    "register_filter",
    "register_sampling_strategy",
    "register_scorer",
]
