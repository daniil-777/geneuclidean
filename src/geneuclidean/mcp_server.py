"""Typed Agentic DMTA Tool Dispatcher & Optional Model Context Protocol (MCP) Server.

Exposes GenEuclidean's deterministic physics, screening, synthesis, MMP lead-optimization,
and credibility capabilities through strict JSON schemas + SHA-256 provenance digests so
industrial pharma agentic DMTA frameworks can invoke them programmatically (directly in
Python via :func:`invoke_dmta_tool`, or over stdio MCP via :func:`run_mcp_stdio_server`
when the optional ``mcp`` package is installed).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from geneuclidean.exceptions import OptionalDependencyError

DMTA_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "score_molecules",
        "description": "Score SMILES for pH 7.4 ionization, CNS MPO v1/v2, hygiene & QED/SA.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "evaluate_property_profile",
        "description": (
            "Evaluate a molecule under an indication-agnostic profile "
            "(cns_mpo_v2, oral_ro5, kinase_selectivity, beyond_ro5)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "profile": {"type": "string", "default": "oral_ro5"},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "audit_complex",
        "description": (
            "Run the 7-axis Beyond-PoseBusters audit on a 3D ligand SDF and optional pocket PDB."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pose_sdf": {"type": "string"},
                "pocket_pdb": {"type": "string"},
                "name": {"type": "string", "default": "pose"},
                "property_profile": {"type": "string", "default": "oral_ro5"},
            },
            "required": ["pose_sdf"],
        },
    },
    {
        "name": "enumerate_makeable_analogs",
        "description": (
            "Generate 1-step reaction-grounded makeable analogs from core SMILES + building blocks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "core_smiles": {"type": "string"},
                "building_blocks": {"type": "array", "items": {"type": "string"}},
                "max_products": {"type": "integer", "default": 50},
            },
            "required": ["core_smiles", "building_blocks"],
        },
    },
    {
        "name": "propose_mmp_rescues",
        "description": (
            "Propose minimal 1-cut MedChem bioisostere/MMP swaps to improve MPO & Avoid-ome safety."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "parent_smiles": {"type": "string"},
                "property_profile": {"type": "string", "default": "cns_mpo_v2"},
                "max_proposals": {"type": "integer", "default": 15},
            },
            "required": ["parent_smiles"],
        },
    },
    {
        "name": "match_retrosynthesis",
        "description": "Identify robust 1-step MedChem retrosynthetic disconnections.",
        "inputSchema": {
            "type": "object",
            "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"],
        },
    },
)


def _stamp_provenance(tool_name: str, arguments: Mapping[str, Any], result: Any) -> dict[str, Any]:
    """Attach deterministic SHA-256 input/output provenance digest for DMTA audit trails."""
    canonical = json.dumps(
        {"tool": tool_name, "arguments": dict(arguments), "result": result},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "tool": tool_name,
        "provenance_sha256": digest,
        "result": result,
    }


def invoke_dmta_tool(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically execute a registered GenEuclidean DMTA tool with SHA-256 provenance."""
    args = dict(arguments)
    if tool_name == "score_molecules":
        from geneuclidean.sdk.scoring import score_molecules

        reps = score_molecules(
            args["smiles"],
            target=args.get("target"),
            include_admet=False,
        )
        return _stamp_provenance(tool_name, args, [r.to_dict() for r in reps])

    if tool_name == "evaluate_property_profile":
        from geneuclidean.sdk.profiles import evaluate_property_profile

        res = evaluate_property_profile(
            str(args["smiles"]), profile=str(args.get("profile", "oral_ro5"))
        )
        return _stamp_provenance(tool_name, args, res.to_dict())

    if tool_name == "audit_complex":
        from geneuclidean.audit import audit_complex

        rep = audit_complex(
            str(args["pose_sdf"]),
            pocket_pdb=args.get("pocket_pdb"),
            name=str(args.get("name", "pose")),
            property_profile=str(args.get("property_profile", "oral_ro5")),
        )
        return _stamp_provenance(tool_name, args, rep.to_dict())

    if tool_name == "enumerate_makeable_analogs":
        from geneuclidean.synthesis import enumerate_makeable_analogs

        cands = enumerate_makeable_analogs(
            str(args["core_smiles"]),
            list(args["building_blocks"]),
            max_products=int(args.get("max_products", 50)),
        )
        return _stamp_provenance(tool_name, args, [c.to_dict() for c in cands])

    if tool_name == "propose_mmp_rescues":
        from geneuclidean.leadopt import AvoidomeMMPEngine

        eng = AvoidomeMMPEngine(property_profile=str(args.get("property_profile", "cns_mpo_v2")))
        props = eng.propose_rescues(
            str(args["parent_smiles"]), max_proposals=int(args.get("max_proposals", 15))
        )
        return _stamp_provenance(tool_name, args, [p.to_dict() for p in props])

    if tool_name == "match_retrosynthesis":
        from geneuclidean.synthesis import match_one_step_retrosynthesis

        res_retro = match_one_step_retrosynthesis(str(args["smiles"]))
        return _stamp_provenance(tool_name, args, res_retro)

    valid_names = [t["name"] for t in DMTA_TOOL_SCHEMAS]
    raise KeyError(f"Unknown DMTA tool {tool_name!r}. Available tools: {valid_names}")


def run_mcp_stdio_server() -> None:
    """Launch a Model Context Protocol (MCP) stdio server when optional ``mcp`` is installed."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            "Running the MCP stdio server requires the optional `mcp` package.",
            hint="Install with `pip install mcp` or invoke tools directly via `invoke_dmta_tool`.",
        ) from exc

    server = FastMCP("geneuclidean-dmta")
    for schema in DMTA_TOOL_SCHEMAS:
        tname = str(schema["name"])
        tdesc = str(schema["description"])

        def _make_handler(name: str, desc: str) -> Any:
            def _handler(**kwargs: Any) -> dict[str, Any]:
                return invoke_dmta_tool(name, kwargs)

            _handler.__name__ = name
            _handler.__doc__ = desc
            return _handler

        server.tool()(_make_handler(tname, tdesc))
    server.run()


__all__ = [
    "DMTA_TOOL_SCHEMAS",
    "invoke_dmta_tool",
    "run_mcp_stdio_server",
]
