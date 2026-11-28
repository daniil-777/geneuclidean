"""Lazy public interface for Circuit Studio, its catalog, viewer and HTTP server."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geneuclidean.studio.catalog import (
        BLOCK_CATALOG,
        BLOCK_DEFINITIONS,
        CATEGORY_SPECS,
        FLAGSHIP_PRESETS,
        PRESET_BY_ID,
        REMEDIATION_HINTS,
        SCHEMA_VERSION,
        SIGNAL_SPECS,
        VALID_SIGNAL_TYPES,
        BlockCategoryId,
        BlockDefinition,
        CategorySpec,
        EdgeKind,
        ExecutionTier,
        ParamSpec,
        ParamType,
        PortDirection,
        PortSpec,
        SignalSpec,
        SignalType,
        export_catalog_payload,
        get_block_definition,
        get_mismatch_remediation,
        get_preset,
        get_studio_catalog,
        list_block_definitions,
        list_presets,
    )
    from geneuclidean.studio.graph_compiler import (
        CodegenResult,
        DiagnosticCode,
        DiagnosticSeverity,
        ValidationDiagnostic,
        ValidationResult,
        canonical_graph_sha256,
        compile_circuit_to_python,
        execute_circuit,
        has_active_learning_v51,
        validate_circuit,
    )
    from geneuclidean.studio.interaction_viewer import (
        VIEWER_ASSET,
        VIEWER_SCENE_SCHEMA,
        build_interaction_scene,
        render_viewer_document,
        resolve_viewer_asset,
        write_viewer_bundle,
    )
    from geneuclidean.studio.server import (
        MAX_REQUEST_BYTES,
        StudioHTTPServer,
        StudioRequestHandler,
        build_health_payload,
        create_studio_server,
        run_studio_server,
        serve_studio,
    )

__all__ = [
    "BLOCK_CATALOG",
    "BLOCK_DEFINITIONS",
    "CATEGORY_SPECS",
    "FLAGSHIP_PRESETS",
    "MAX_REQUEST_BYTES",
    "PRESET_BY_ID",
    "REMEDIATION_HINTS",
    "SCHEMA_VERSION",
    "SIGNAL_SPECS",
    "VALID_SIGNAL_TYPES",
    "VIEWER_ASSET",
    "VIEWER_SCENE_SCHEMA",
    "BlockCategoryId",
    "BlockDefinition",
    "CategorySpec",
    "CodegenResult",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "EdgeKind",
    "ExecutionTier",
    "ParamSpec",
    "ParamType",
    "PortDirection",
    "PortSpec",
    "SignalSpec",
    "SignalType",
    "StudioHTTPServer",
    "StudioRequestHandler",
    "ValidationDiagnostic",
    "ValidationResult",
    "build_health_payload",
    "build_interaction_scene",
    "canonical_graph_sha256",
    "compile_circuit_to_python",
    "create_studio_server",
    "execute_circuit",
    "export_catalog_payload",
    "get_block_definition",
    "get_mismatch_remediation",
    "get_preset",
    "get_studio_catalog",
    "has_active_learning_v51",
    "list_block_definitions",
    "list_presets",
    "render_viewer_document",
    "resolve_viewer_asset",
    "run_studio_server",
    "serve_studio",
    "validate_circuit",
    "write_viewer_bundle",
]

_MODULES = {
    "CodegenResult": ".graph_compiler",
    "DiagnosticCode": ".graph_compiler",
    "DiagnosticSeverity": ".graph_compiler",
    "ValidationDiagnostic": ".graph_compiler",
    "ValidationResult": ".graph_compiler",
    "canonical_graph_sha256": ".graph_compiler",
    "compile_circuit_to_python": ".graph_compiler",
    "execute_circuit": ".graph_compiler",
    "has_active_learning_v51": ".graph_compiler",
    "validate_circuit": ".graph_compiler",
    "VIEWER_ASSET": ".interaction_viewer",
    "VIEWER_SCENE_SCHEMA": ".interaction_viewer",
    "build_interaction_scene": ".interaction_viewer",
    "render_viewer_document": ".interaction_viewer",
    "resolve_viewer_asset": ".interaction_viewer",
    "write_viewer_bundle": ".interaction_viewer",
    "MAX_REQUEST_BYTES": ".server",
    "StudioHTTPServer": ".server",
    "StudioRequestHandler": ".server",
    "build_health_payload": ".server",
    "create_studio_server": ".server",
    "run_studio_server": ".server",
    "serve_studio": ".server",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_MODULES.get(name, ".catalog"), __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
