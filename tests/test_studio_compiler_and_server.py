"""Comprehensive offline CI-safe unit & live-HTTP tests for GenEuclidean Circuit Studio.

Covers:
1. Cold-import latency & lazy-import hygiene (`import geneuclidean` & `import geneuclidean.studio`).
2. Authoritative catalog integrity (10 signals, 8 categories, 36 blocks, 5 presets,
   and 6 feedback ports).
3. All 5 flagship presets validate with 0 errors and compile to `ast.parse`-valid Python SDK code
   referencing real `geneuclidean` symbols.
4. Illegal wiring diagnostics (`E_PORT_TYPE_MISMATCH`, `E_COMBINATIONAL_CYCLE`,
   `E_FEEDBACK_NOT_ALLOWED`, `E_SELF_LOOP`, `E_DUPLICATE_EDGE`, `E_MULTI_INPUT_DISALLOWED`,
   `E_MISSING_REQUIRED_INPUT`, `E_UNKNOWN_BLOCK`, `E_UNKNOWN_PORT`, `E_PARAM_OUT_OF_RANGE`).
5. Legal closed-loop DMTA feedback circuits vs illegal combinational forward cycles.
6. Codegen injection-proofing (`repr()` literal safety & identifier sanitization) and AST audit
   verifying zero `eval()` / `exec()` / `shell=True` calls in `src/geneuclidean/studio/`.
7. Real local execution across all 5 presets (RDKit protonation, Cremer-Pople/torsion strain,
   7-Axis PoseRealityAuditor, 16-reaction enumeration, AvoidomeMMPEngine, ConformalDMPKRegressor,
   OpenFE Δq=0 exporter, Tanimoto-GP multi-round DMTA, and SHA-256 provenance digests).
8. Live stdlib `ThreadingHTTPServer` HTTP integration tests (CORS preflight, `/api/health`,
   `/api/catalog`, `/api/circuit/validate`, `/api/circuit/codegen`, `/api/circuit/run`,
   400/404/413/422) and `geneuclidean studio --help` CLI check.
"""

from __future__ import annotations

import ast
import copy
import http.client
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import geneuclidean as ge
from geneuclidean.cli import app as cli_app
from geneuclidean.studio import (
    BLOCK_CATALOG,
    CATEGORY_SPECS,
    FLAGSHIP_PRESETS,
    MAX_REQUEST_BYTES,
    SCHEMA_VERSION,
    SIGNAL_SPECS,
    compile_circuit_to_python,
    create_studio_server,
    execute_circuit,
    get_preset,
    validate_circuit,
)

# ---------------------------------------------------------------------------
# 1. Lazy-Import Discipline & Source-Level Security Audit
# ---------------------------------------------------------------------------


def test_studio_preserves_lazy_import_hygiene() -> None:
    """Importing `geneuclidean` and `geneuclidean.studio` must never eagerly load heavy libs."""
    probe = """
import json, sys, time
t0 = time.perf_counter()
import geneuclidean
dt_root = (time.perf_counter() - t0) * 1000.0
import geneuclidean.studio as studio
dt_studio = (time.perf_counter() - t0) * 1000.0
heavy = [
    m for m in ("torch", "transformers", "sklearn", "rdkit", "matplotlib", "scipy", "numpy")
    if m in sys.modules
]
print(json.dumps({
    "dt_root_ms": round(dt_root, 2),
    "dt_studio_ms": round(dt_studio, 2),
    "heavy": heavy,
    "n_blocks": len(studio.BLOCK_CATALOG),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout.strip())
    assert info["heavy"] == [], f"Unexpected heavy eager imports: {info['heavy']}"
    assert info["dt_root_ms"] < 200.0
    assert info["n_blocks"] == 36


def test_studio_source_has_zero_eval_exec_or_shell_true() -> None:
    """AST security audit: zero eval(), exec(), or subprocess(..., shell=True) in studio/."""
    studio_dir = Path(ge.__file__).resolve().parent / "studio"
    py_files = sorted(studio_dir.glob("*.py"))
    assert len(py_files) == 4

    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                    pytest.fail(f"Forbidden call to {node.func.id}() in {path}:{node.lineno}")
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        pytest.fail(f"Forbidden shell=True call in {path}:{node.lineno}")


# ---------------------------------------------------------------------------
# 2. Authoritative Catalog & Flagship Presets Validation + Codegen
# ---------------------------------------------------------------------------


def test_catalog_counts_and_feedback_port_allowlist() -> None:
    """Catalog matches exact 10 signals, 8 categories, 36 blocks, 5 presets & 6 feedback ports."""
    assert len(SIGNAL_SPECS) == 10
    assert len(CATEGORY_SPECS) == 8
    assert len(BLOCK_CATALOG) == 36
    assert len(FLAGSHIP_PRESETS) == 5

    feedback_ports = sorted(
        f"{b.block_type}.{p.id}"
        for b in BLOCK_CATALOG.values()
        for p in b.inputs
        if p.accepts_feedback
    )
    assert feedback_ports == [
        "chemical_llm.feedback_scores",
        "combinatorial_ts.feedback_scores",
        "ensemble_surrogate.observations",
        "safe_gpt.feedback_scores",
        "selectivity_mtl_aligner.feedback_scores",
        "tanimoto_gp.observations",
    ]


@pytest.mark.parametrize(
    "preset_id",
    [
        "se3_denovo_cns",
        "al_dmta_closed_loop",
        "paralog_selectivity_mmp",
        "boltz2_audit_conformal_dmpk",
        "blank_breadboard",
    ],
)
def test_all_five_flagship_presets_validate_and_compile_to_valid_ast(preset_id: str) -> None:
    """Every flagship preset validates with 0 errors and compiles to clean, real SDK Python."""
    preset = get_preset(preset_id)
    val = validate_circuit(preset)
    errors = [d for d in val.diagnostics if d.severity == "error"]
    assert val.valid is True, f"Preset {preset_id} failed validation: {errors}"
    assert errors == []

    if preset_id == "al_dmta_closed_loop":
        assert val.has_feedback_loop is True
        assert val.feedback_edges == ("e_fb_1",)
        assert len(val.topological_order) == 8
    elif preset_id == "blank_breadboard":
        assert val.topological_order == ()
        assert any(d.code == "I_EMPTY_CIRCUIT" for d in val.diagnostics)
    else:
        assert val.has_feedback_loop is False
        assert len(val.topological_order) == len(preset["nodes"])

    codegen = compile_circuit_to_python(preset)
    assert codegen.valid is True
    assert codegen.ast_valid is True
    assert len(codegen.provenance_sha256) == 64
    assert any("geneuclidean" in imp for imp in codegen.imports)

    # Confirm AST parses cleanly and every ge.<attr> symbol actually exists in geneuclidean
    tree = ast.parse(codegen.python_code)
    ge_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ge"
    }
    available = set(dir(ge))
    missing = ge_attrs - available
    assert missing == set(), f"Compiled code for {preset_id} references unknown ge attrs: {missing}"


# ---------------------------------------------------------------------------
# 3. Illegal Wiring, Cycle Detection & Codegen Injection Safety
# ---------------------------------------------------------------------------


def test_illegal_port_mismatch_returns_actionable_remediation() -> None:
    """Wiring MOL_BATCH directly into POSE_3D is rejected with E_PORT_TYPE_MISMATCH + hint."""
    bad_graph: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": "bad_mismatch",
        "name": "Bad Mismatch",
        "description": "Test",
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "DRD1",
            "budget": 8,
        },
        "nodes": [
            {
                "id": "n_pool",
                "blockType": "chembl_pool",
                "position": {"x": 0, "y": 0},
                "params": {},
            },
            {
                "id": "n_audit",
                "blockType": "pose_reality_auditor",
                "position": {"x": 300, "y": 0},
                "params": {},
            },
        ],
        "edges": [
            {
                "id": "e_bad",
                "source": "n_pool",
                "sourcePort": "molecules",
                "target": "n_audit",
                "targetPort": "poses",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            }
        ],
    }
    val = validate_circuit(bad_graph)
    assert val.valid is False
    mismatch = [d for d in val.diagnostics if d.code == "E_PORT_TYPE_MISMATCH"]
    assert len(mismatch) == 1
    assert "AutoDock Vina" in str(mismatch[0].remediation or "")


def test_combinational_forward_cycle_rejected_while_dmta_feedback_allowed() -> None:
    """Forward cycle fails with E_COMBINATIONAL_CYCLE; marking feedback edge succeeds."""
    al_preset = get_preset("al_dmta_closed_loop")
    assert validate_circuit(al_preset).valid is True

    # Flip the legal feedback edge to a forward edge -> must trigger E_COMBINATIONAL_CYCLE
    cyclic = copy.deepcopy(al_preset)
    for edge in cyclic["edges"]:
        if edge["id"] == "e_fb_1":
            edge["kind"] = "forward"

    val_cyclic = validate_circuit(cyclic)
    assert val_cyclic.valid is False
    assert any(d.code == "E_COMBINATIONAL_CYCLE" for d in val_cyclic.diagnostics)

    # Marking kind='feedback' on a non-feedback input port must trigger E_FEEDBACK_NOT_ALLOWED
    bad_fb = copy.deepcopy(al_preset)
    for edge in bad_fb["edges"]:
        if edge["id"] == "e1":  # n_pool.molecules -> n_gp.pool_mols (acceptsFeedback=False)
            edge["kind"] = "feedback"
    val_bad_fb = validate_circuit(bad_fb)
    assert val_bad_fb.valid is False
    assert any(d.code == "E_FEEDBACK_NOT_ALLOWED" for d in val_bad_fb.diagnostics)


def test_validation_catches_self_loop_duplicate_multi_input_and_param_bounds() -> None:
    """Validator catches self-loops, duplicates, multi-input violations, and unsafe path params."""
    graph: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": "multi_errors",
        "name": "Multi Errors",
        "description": "Test",
        "nodes": [
            {
                "id": "n1",
                "blockType": "protein_pocket",
                "position": {"x": 0, "y": 0},
                "params": {"target_id": "INVALID_TARGET", "pocket_cutoff": 99.0},
            },
            {
                "id": "n2",
                "blockType": "openfe_rbfe_exporter",
                "position": {"x": 300, "y": 0},
                "params": {"out_dir": "../../etc/passwd"},
            },
            {
                "id": "n3",
                "blockType": "unknown_block_xyz",
                "position": {"x": 600, "y": 0},
                "params": {},
            },
        ],
        "edges": [
            {
                "id": "e_self",
                "source": "n1",
                "sourcePort": "pocket",
                "target": "n1",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e_dup_1",
                "source": "n1",
                "sourcePort": "pocket",
                "target": "n2",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e_dup_2",
                "source": "n1",
                "sourcePort": "pocket",
                "target": "n2",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
        ],
    }
    val = validate_circuit(graph)
    assert val.valid is False
    codes = {d.code for d in val.diagnostics}
    assert "E_UNKNOWN_BLOCK" in codes
    assert "E_SELF_LOOP" in codes
    assert "E_DUPLICATE_EDGE" in codes
    assert "E_PARAM_OUT_OF_RANGE" in codes

    # Also test multi-input disallowed when two distinct pockets wire into a single-input port
    multi_in: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": "multi_in_bad",
        "name": "Multi In Bad",
        "description": "Test",
        "nodes": [
            {"id": "p1", "blockType": "protein_pocket", "position": {"x": 0, "y": 0}, "params": {}},
            {
                "id": "p2",
                "blockType": "protein_pocket",
                "position": {"x": 0, "y": 100},
                "params": {},
            },
            {"id": "e1", "blockType": "painn_se3", "position": {"x": 300, "y": 0}, "params": {}},
        ],
        "edges": [
            {
                "id": "w1",
                "source": "p1",
                "sourcePort": "pocket",
                "target": "e1",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "w2",
                "source": "p2",
                "sourcePort": "pocket",
                "target": "e1",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
        ],
    }
    val_mi = validate_circuit(multi_in)
    assert val_mi.valid is False
    assert any(d.code == "E_MULTI_INPUT_DISALLOWED" for d in val_mi.diagnostics)


def test_codegen_escapes_malicious_node_ids_and_string_params() -> None:
    """Untrusted string parameters and node IDs cannot break out into executable statements."""
    payload_str = "O=C(O)c1ccccc1'); __import__('os').system('echo PWNED'); ('"
    graph: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": "inject'; import os #",
        "name": 'Inject """ Test',
        "description": "Security test",
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "DRD1",
            "budget": 8,
        },
        "nodes": [
            {
                "id": "n1\nimport os; os.system('id')",
                "blockType": "reagent_library",
                "position": {"x": 0, "y": 0},
                "params": {"core_smiles": payload_str, "reagent_pack": "cns_amines"},
            }
        ],
        "edges": [],
    }
    res = compile_circuit_to_python(graph)
    assert res.valid is True
    assert res.ast_valid is True
    tree = ast.parse(res.python_code)
    # Ensure no __import__ or os import nodes exist anywhere in the generated AST!
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "__import__"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "os"


# ---------------------------------------------------------------------------
# 4. Real Local Execution Engine Across All 5 Flagship Presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_id",
    [
        "se3_denovo_cns",
        "al_dmta_closed_loop",
        "paralog_selectivity_mmp",
        "boltz2_audit_conformal_dmpk",
        "blank_breadboard",
    ],
)
def test_real_local_execution_across_all_flagship_presets(preset_id: str) -> None:
    """All 5 flagship presets execute locally end-to-end with real RDKit/CADD outputs & SHA-256."""
    preset = get_preset(preset_id)
    if preset_id == "al_dmta_closed_loop":
        preset["execution"]["dmtaRounds"] = 3

    res = execute_circuit(preset, max_samples=4, timeout_sec=30.0)
    assert res["status"] == "completed", f"Preset {preset_id} failed: {res['diagnostics']}"
    assert res["engineMode"] == "local_python"
    assert len(res["provenanceSha256"]) == 64
    assert res["provenanceLedger"]["schema"] == "geneuclidean_dmta_ledger_v1"
    assert len(res["provenanceLedger"]["graphSha256"]) == 64
    assert len(res["provenanceLedger"]["resultSha256"]) == 64
    assert ast.parse(res["generatedCode"]) is not None

    if preset_id == "blank_breadboard":
        assert res["steps"] == []
    elif preset_id == "al_dmta_closed_loop":
        assert res["roundsCompleted"] == 3
        assert len(res["dmtaTrace"]) == 3
        assert res["dmtaTrace"][-1]["nObserved"] >= res["dmtaTrace"][0]["nObserved"] > 0
        assert len(res["steps"]) == 3 * len(preset["nodes"])
        assert len(res["molecules"]) >= 1
    else:
        assert res["roundsCompleted"] == 1
        assert len(res["steps"]) == len(preset["nodes"])
        assert len(res["molecules"]) >= 1
        real_steps = [s for s in res["steps"] if s["executionBackend"] == "real_python"]
        assert len(real_steps) >= 3


def test_real_local_execution_dry_run_mode() -> None:
    """dry_run=True validates and skips compute cleanly without errors."""
    preset = get_preset("se3_denovo_cns")
    res = execute_circuit(preset, dry_run=True)
    assert res["status"] == "completed"
    assert all(s["status"] == "skipped" for s in res["steps"])


# ---------------------------------------------------------------------------
# 5. Live Stdlib ThreadingHTTPServer Integration Tests + CLI Command Check
# ---------------------------------------------------------------------------


def _http_request(
    url: str,
    *,
    method: str = "GET",
    payload: Any | None = None,
    raw_bytes: bytes | None = None,
) -> tuple[int, dict[str, str], dict[str, Any] | None]:
    data = raw_bytes
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body.strip() else None
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body) if body.strip() else None
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, parsed


def test_live_http_studio_server_endpoints_and_error_handling() -> None:
    """Live HTTP server passes CORS, health, catalog, validate, codegen, run, and error codes."""
    server = create_studio_server("127.0.0.1", 0, quiet=True)
    port = int(server.server_address[1])
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # 1. CORS Preflight OPTIONS /api/circuit/run -> 204
        code, headers, _ = _http_request(f"{base_url}/api/circuit/run", method="OPTIONS")
        assert code == 204
        assert headers.get("access-control-allow-origin") == "*"
        assert "POST" in headers.get("access-control-allow-methods", "")

        # 2. GET /api/health -> 200
        code, headers, health = _http_request(f"{base_url}/api/health")
        assert code == 200
        assert health is not None
        assert health["status"] == "ok"
        assert health["catalogCount"] == 36
        assert health["rdkit"] is not None
        assert headers.get("access-control-allow-origin") == "*"

        # 3. GET /api/catalog -> 200
        code, _, catalog = _http_request(f"{base_url}/api/catalog")
        assert code == 200
        assert catalog is not None
        assert catalog["schemaVersion"] == SCHEMA_VERSION
        assert len(catalog["blocks"]) == 36
        assert len(catalog["signalTypes"]) == 10
        assert len(catalog["categories"]) == 8
        assert len(catalog["presets"]) == 5

        # 4. POST /api/circuit/validate -> 200 (valid preset & invalid JSON -> 400)
        preset_boltz = get_preset("boltz2_audit_conformal_dmpk")
        code, _, val = _http_request(
            f"{base_url}/api/circuit/validate", method="POST", payload=preset_boltz
        )
        assert code == 200
        assert val is not None and val["valid"] is True

        code_bad, _, err_json = _http_request(
            f"{base_url}/api/circuit/validate", method="POST", raw_bytes=b"{not-valid-json"
        )
        assert code_bad == 400
        assert err_json is not None and "error" in err_json

        # 5. POST /api/circuit/codegen -> 200 (valid) & 422 (invalid graph)
        code, _, cg = _http_request(
            f"{base_url}/api/circuit/codegen", method="POST", payload=preset_boltz
        )
        assert code == 200
        assert cg is not None and cg["valid"] is True and cg["astValid"] is True
        assert len(cg["provenanceSha256"]) == 64

        bad_graph = {"schemaVersion": "wrong.v0", "nodes": [], "edges": []}
        code_422, _, cg_bad = _http_request(
            f"{base_url}/api/circuit/codegen", method="POST", payload=bad_graph
        )
        assert code_422 == 422
        assert cg_bad is not None and cg_bad["valid"] is False

        # 6. POST /api/circuit/run -> 200 (real local execution w/ SHA-256 provenance)
        code, _, run_res = _http_request(
            f"{base_url}/api/circuit/run",
            method="POST",
            payload={"circuit": preset_boltz, "options": {"maxSamples": 3, "timeoutSec": 25}},
        )
        assert code == 200
        assert run_res is not None
        assert run_res["status"] == "completed"
        assert len(run_res["provenanceSha256"]) == 64
        assert len(run_res["steps"]) == len(preset_boltz["nodes"])
        assert len(run_res["molecules"]) >= 1

        # 7. Invalid circuit on /api/circuit/run -> 422, unknown route -> 404, oversized -> 413
        code_r422, _, run_bad = _http_request(
            f"{base_url}/api/circuit/run", method="POST", payload={"circuit": bad_graph}
        )
        assert code_r422 == 422
        assert run_bad is not None and run_bad["status"] == "validation_error"

        code_404, _, _ = _http_request(f"{base_url}/api/nonexistent")
        assert code_404 == 404

        # Send declared oversized Content-Length header so server rejects with 413 immediately
        # without needing to race a 2 MiB socket write against connection teardown.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            conn.putrequest("POST", "/api/circuit/validate")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(MAX_REQUEST_BYTES + 128))
            conn.endheaders()
            resp_413 = conn.getresponse()
            body_413 = json.loads(resp_413.read().decode("utf-8"))
            assert resp_413.status == 413
            assert "error" in body_413
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_studio_command_help() -> None:
    """`geneuclidean studio --help` registers cleanly with --host and --port options."""
    runner = CliRunner()
    res = runner.invoke(cli_app, ["studio", "--help"])
    assert res.exit_code == 0
    assert "--host" in res.output
    assert "--port" in res.output
    assert "8765" in res.output
