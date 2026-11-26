"""Target and backbone registry helpers for the GenEuclidean high-level SDK.

All functions in this module preserve zero-PyTorch startup latency so cataloging
curated CNS targets or supported chemical LLMs never triggers heavyweight ML imports.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from geneuclidean.registry import BACKBONE_CATALOG

if TYPE_CHECKING:
    from geneuclidean.cns.targets import CNSTargetSpec


def list_targets(
    indication: str | None = None,
    *,
    disease_area: str | None = None,
    include_v4: bool = True,
    include_ts_circuit: bool = True,
    include_counters: bool = False,
) -> list[CNSTargetSpec]:
    """Return curated *Homo sapiens* CNS drug targets, optionally filtered by keyword or area.

    By default returns the full **28-target Human CNS Atlas** (12 v3.1 Tourette/Psychiatry
    + 12 v4.0 Tourette/Schizophrenia/Alzheimer's + 4 Tourette CSTC/GABAergic circuit
    pockets) plus any user-registered custom targets.

    Args:
        indication: Optional case-insensitive substring matching therapeutic indications,
            mechanism of action, or disease area (e.g., ``"tourette"``, ``"schizophrenia"``,
            ``"alzheimer"``, ``"gaba"``, ``"adhd"``).
        disease_area: Optional exact disease-area filter (``"tourette"``, ``"schizophrenia"``,
            ``"alzheimer"``, ``"psychiatry"``, ``"counter"``, or ``"all"``).
        include_v4: Include the 12 v4.0 Tri-Disorder expansion targets (default ``True``).
        include_ts_circuit: Include the 4 verified Tourette CSTC/GABAergic circuit pockets
            (``GABRA5``, ``GAT1``, ``ADORA2A``, ``GRM5``; default ``True``).
        include_counters: Include matched off-target safety counter-targets (e.g., ``BCHE``,
            ``MAOA``, ``CHRM3``, ``HTR2B``, ``HRH1``; default ``False``).

    Returns:
        List of :class:`~geneuclidean.cns.targets.CNSTargetSpec` instances.
    """
    from geneuclidean.cns.targets import list_cns_targets

    return list_cns_targets(
        indication_filter=indication,
        disease_area=disease_area,
        include_v4=include_v4,
        include_counters=include_counters,
        include_ts_circuit=include_ts_circuit,
    )


def get_target(target_id: str) -> CNSTargetSpec:
    """Look up a human CNS or custom target by symbol, gene name, ChEMBL ID, or UniProt ID.

    Args:
        target_id: Identifier such as ``"DRD1"``, ``"ACHE"``, ``"GRM5"``, ``"SLC18A2"``,
            ``"CHEMBL1893"``, or ``"P21728"``.

    Returns:
        Matching :class:`~geneuclidean.cns.targets.CNSTargetSpec`.

    Raises:
        TargetNotFoundError: Subclass of both :class:`GenEuclideanError` and ``KeyError``
            when ``target_id`` cannot be resolved.
    """
    from geneuclidean.cns.targets import get_cns_target

    return get_cns_target(target_id)


def register_target(
    spec: CNSTargetSpec,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
) -> CNSTargetSpec:
    """Register a custom protein target specification into the runtime target registry.

    Registered targets immediately resolve via :func:`get_target`, :func:`list_targets`,
    and :meth:`Pocket.from_target` without altering the curated benchmark atlas counts.

    Args:
        spec: Validated :class:`~geneuclidean.cns.targets.CNSTargetSpec` instance.
        aliases: Optional extra lookup aliases (gene symbol, ChEMBL ID, and UniProt ID on
            ``spec`` are indexed automatically).
        override: Allow replacing an already-registered custom target ID.

    Returns:
        The registered :class:`CNSTargetSpec`.
    """
    from geneuclidean.cns.targets import register_cns_target

    return register_cns_target(spec, aliases=aliases, override=override)


def unregister_target(target_id: str) -> CNSTargetSpec:
    """Remove a previously registered custom protein target from the runtime registry."""
    from geneuclidean.cns.targets import unregister_cns_target

    return unregister_cns_target(target_id)


def list_backbones() -> dict[str, dict[str, Any]]:
    """Return metadata for all supported pretrained chemical & local LLM backbones.

    Executes without importing ``torch`` or ``transformers`` so startup stays instantaneous.
    If ``geneuclidean.models.hf_backbones`` is already loaded in the current interpreter,
    any dynamically registered backbones are merged automatically.

    Returns:
        Dictionary mapping canonical Hugging Face / mock backbone IDs to metadata dicts.
    """
    catalog: dict[str, dict[str, Any]] = {k: dict(v) for k, v in BACKBONE_CATALOG.items()}
    mod = sys.modules.get("geneuclidean.models.hf_backbones")
    if mod is not None and hasattr(mod, "BACKBONE_REGISTRY"):
        reg = mod.BACKBONE_REGISTRY
        for key, spec in reg.items():
            bid = str(getattr(spec, "backbone_id", key))
            if bid not in catalog:
                catalog[bid] = {
                    "backbone_id": bid,
                    "family": str(getattr(spec, "family", "custom")),
                    "representation": str(getattr(spec, "representation", "smiles")),
                    "is_smiles_native": bool(getattr(spec, "is_smiles_native", False)),
                    "tokenizer_id": str(getattr(spec, "tokenizer_id", bid)),
                }
    return catalog


def audit_tourette_circuit(target_id: str | None = None) -> dict[str, Any]:
    """Return the Tourette CSTC basal-ganglia & GABAergic circuit pharmacological audit.

    Args:
        target_id: Optional target ID (e.g., ``"DRD1"``, ``"GABRA5"``, ``"GAT1"``,
            ``"ADORA2A"``, ``"GRM5"``). When omitted, returns the complete audit map
            covering all 13 Tourette circuit targets.

    Returns:
        JSON-serializable dictionary of clinical evidence grades, CSTC circuit nodes,
        conformation caveats, and pocket validity statuses.
    """
    from dataclasses import asdict

    from geneuclidean.cns.tourette_circuit import (
        TOURETTE_CSTC_CIRCUIT_AUDIT,
        get_cstc_circuit_audit,
    )

    if target_id is not None:
        matched = get_cstc_circuit_audit(target_id)
        if not matched:
            from geneuclidean.exceptions import TargetNotFoundError

            raise TargetNotFoundError(
                f"No Tourette CSTC circuit audit entry found for {target_id!r}."
            )
        entry = next(iter(matched.values()))
        return dict(asdict(entry))
    return {tid: dict(asdict(entry)) for tid, entry in TOURETTE_CSTC_CIRCUIT_AUDIT.items()}


__all__ = [
    "audit_tourette_circuit",
    "get_target",
    "list_backbones",
    "list_targets",
    "register_target",
    "unregister_target",
]
