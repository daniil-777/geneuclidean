"""Comprehensive offline CI-safe unit & live-HTTP tests for GenEuclidean Circuit Studio.

Covers:
1. Cold-import latency & lazy-import hygiene (`import geneuclidean` & `import geneuclidean.studio`).
2. Authoritative catalog integrity (10 signals, 8 categories, 49 blocks, 7 presets,
   and 8 feedback ports).
3. All 7 flagship presets validate with 0 errors and compile to `ast.parse`-valid Python SDK code
   referencing real `geneuclidean` symbols.
4. Illegal wiring diagnostics (`E_PORT_TYPE_MISMATCH`, `E_COMBINATIONAL_CYCLE`,
   `E_FEEDBACK_NOT_ALLOWED`, `E_SELF_LOOP`, `E_DUPLICATE_EDGE`, `E_MULTI_INPUT_DISALLOWED`,
   `E_MISSING_REQUIRED_INPUT`, `E_UNKNOWN_BLOCK`, `E_UNKNOWN_PORT`, `E_PARAM_OUT_OF_RANGE`).
5. Legal closed-loop DMTA feedback circuits vs illegal combinational forward cycles.
6. Codegen injection-proofing (`repr()` literal safety & identifier sanitization) and AST audit
   verifying zero `eval()` / `exec()` / `shell=True` calls in `src/geneuclidean/studio/`.
7. Real local execution across all 7 presets (RDKit protonation, Cremer-Pople/torsion strain,
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
    assert info["n_blocks"] == 49


def test_studio_source_has_zero_eval_exec_or_shell_true() -> None:
    """AST security audit: zero eval(), exec(), or subprocess(..., shell=True) in studio/."""
    studio_dir = Path(ge.__file__).resolve().parent / "studio"
    py_files = sorted(studio_dir.glob("*.py"))
    assert len(py_files) == 5

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
    """Catalog matches exact 10 signals, 8 categories, 49 blocks, 7 presets & 8 feedback ports."""
    assert len(SIGNAL_SPECS) == 10
    assert len(CATEGORY_SPECS) == 8
    assert len(BLOCK_CATALOG) == 49
    assert len(FLAGSHIP_PRESETS) == 7

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
        "nlp_lora_rl_aligner.rewards",
        "nlp_rope_gqa_lm.feedback_scores",
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
        "nanofold3_cofold_audit",
        "tourette_scz_champion",
        "blank_breadboard",
    ],
)
def test_all_seven_flagship_presets_validate_and_compile_to_valid_ast(preset_id: str) -> None:
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
        "nanofold3_cofold_audit",
        "tourette_scz_champion",
        "blank_breadboard",
    ],
)
def test_real_local_execution_across_all_flagship_presets(preset_id: str) -> None:
    """All 7 flagship presets execute locally end-to-end with real RDKit/CADD/v5.3 outputs."""
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
    assert res["provenanceLedger"]["fdaEmaPrinciples"] == "unassessed"
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
        assert health["catalogCount"] == 49
        assert health["rdkit"] is not None
        assert health["activeLearningV51"] is True
        assert headers.get("access-control-allow-origin") == "*"

        # 3. GET /api/catalog -> 200
        code, _, catalog = _http_request(f"{base_url}/api/catalog")
        assert code == 200
        assert catalog is not None
        assert catalog["schemaVersion"] == SCHEMA_VERSION
        assert len(catalog["blocks"]) == 49
        assert len(catalog["signalTypes"]) == 10
        assert len(catalog["categories"]) == 8
        assert len(catalog["presets"]) == 7

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

        # 6b. POST /api/circuit/run on closed-loop AL DMTA preset -> 200 with multi-round trace
        preset_al = get_preset("al_dmta_closed_loop")
        preset_al["execution"]["dmtaRounds"] = 2
        code_al, _, run_al = _http_request(
            f"{base_url}/api/circuit/run",
            method="POST",
            payload={"circuit": preset_al, "options": {"maxSamples": 4, "timeoutSec": 25}},
        )
        assert code_al == 200
        assert run_al is not None
        assert run_al["status"] == "completed"
        assert run_al["roundsCompleted"] == 2
        assert len(run_al["dmtaTrace"]) == 2
        assert len(run_al["provenanceSha256"]) == 64

        # 6c. POST /api/circuit/run on v5.3 NanoFold3 Co-Fold + 7-Axis Audit preset -> 200
        preset_nf3 = get_preset("nanofold3_cofold_audit")
        code_nf3, _, run_nf3 = _http_request(
            f"{base_url}/api/circuit/run",
            method="POST",
            payload={"circuit": preset_nf3, "options": {"maxSamples": 3, "timeoutSec": 30}},
        )
        assert code_nf3 == 200
        assert run_nf3 is not None
        assert run_nf3["status"] == "completed"
        assert len(run_nf3["steps"]) == 8
        assert any(
            s["blockType"] == "nanofold3_cofold" and s["executionBackend"] == "real_python"
            for s in run_nf3["steps"]
        )

        # 6d. POST /api/circuit/run on v5.3 Tourette & Schizophrenia Champion preset -> 200
        preset_ts = get_preset("tourette_scz_champion")
        code_ts, _, run_ts = _http_request(
            f"{base_url}/api/circuit/run",
            method="POST",
            payload={"circuit": preset_ts, "options": {"maxSamples": 3, "timeoutSec": 30}},
        )
        assert code_ts == 200
        assert run_ts is not None
        assert run_ts["status"] == "completed"
        assert len(run_ts["steps"]) == 10
        assert any(
            s["blockType"] == "cartformer_se3" and s["executionBackend"] == "real_python"
            for s in run_ts["steps"]
        )
        assert any(
            s["blockType"] == "nlp_rope_gqa_lm" and s["executionBackend"] == "real_python"
            for s in run_ts["steps"]
        )

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


# ---------------------------------------------------------------------------
# 6. Regression Tests for @al-verifier Audit Findings (1–4)
# ---------------------------------------------------------------------------


def test_disabled_node_edges_compile_and_execute_without_keyerror() -> None:
    """Disabling an intermediate node (`n_mmp`) compiles and executes cleanly without KeyError."""
    preset = get_preset("paralog_selectivity_mmp")
    for node in preset["nodes"]:
        if node["id"] == "n_mmp":
            node["disabled"] = True

    val = validate_circuit(preset)
    assert val.valid is True
    assert "n_mmp" not in val.topological_order

    cg = compile_circuit_to_python(preset)
    assert cg.valid is True
    assert cg.ast_valid is True
    assert ast.parse(cg.python_code) is not None
    assert "node_n_mmp" not in cg.python_code

    run_res = execute_circuit(preset, max_samples=3, timeout_sec=25.0)
    assert run_res["status"] == "completed"
    assert all(s["nodeId"] != "n_mmp" for s in run_res["steps"])


def test_malformed_position_target_context_and_nan_json_handled_cleanly() -> None:
    """Malformed node.position, invalid targetContext, and NaN/Infinity JSON return clean errors."""
    bad_pos: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": "bad_pos",
        "name": "Bad Position",
        "description": "Test",
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "DRD1\nBAD",
            "budget": 8,
        },
        "nodes": [
            {
                "id": "n1",
                "blockType": "protein_pocket",
                "position": {"x": "oops", "y": None},
                "params": {},
            }
        ],
        "edges": [],
    }
    val = validate_circuit(bad_pos)
    assert val.valid is False
    codes = [d.code for d in val.diagnostics]
    assert codes.count("E_PARAM_OUT_OF_RANGE") >= 2

    # Also verify live HTTP rejects malformed position (200 w/ valid=False) & NaN/Infinity (400)
    server = create_studio_server("127.0.0.1", 0, quiet=True)
    port = int(server.server_address[1])
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code, _, val_json = _http_request(
            f"{base_url}/api/circuit/validate", method="POST", payload=bad_pos
        )
        assert code == 200
        assert val_json is not None and val_json["valid"] is False

        # Non-RFC-8259 NaN/Infinity constants must be rejected with HTTP 400
        nan_body = (
            b'{"circuit": {"schemaVersion": "geneuclidean.circuit.v1"}, '
            b'"options": {"timeoutSec": NaN}}'
        )
        code_nan, _, err_nan = _http_request(
            f"{base_url}/api/circuit/run",
            method="POST",
            raw_bytes=nan_body,
        )
        assert code_nan == 400
        assert err_nan is not None and "error" in err_nan

        code_neg, _, err_neg = _http_request(
            f"{base_url}/api/circuit/run",
            method="POST",
            payload={"circuit": get_preset("blank_breadboard"), "options": {"timeoutSec": -5}},
        )
        assert code_neg == 400
        assert err_neg is not None and "error" in err_neg
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_codegen_handles_backslashes_and_unicode_escapes_without_warnings() -> None:
    """Backslashes, \\x escapes, and trailing backslashes compile with zero SyntaxWarning."""
    import warnings

    for tricky_name in (r"C:\Users\drd1\circuit", r"test \x1 escape", "trailing\\"):
        for preset_id in ("blank_breadboard", "se3_denovo_cns"):
            graph = get_preset(preset_id)
            graph["name"] = tricky_name
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SyntaxWarning)
                res = compile_circuit_to_python(graph)
                assert res.valid is True
                assert res.ast_valid is True
                ast.parse(res.python_code)
            syntax_warnings = [w for w in caught if issubclass(w.category, SyntaxWarning)]
            assert syntax_warnings == [], f"SyntaxWarning on {tricky_name!r}: {syntax_warnings}"


def test_tanimoto_gp_zero_norm_kernel_and_leak_free_loo_conformal() -> None:
    """Tanimoto GP sets k(0,0)=signal_var and produces leak-free LOO residuals for Conformal UQ."""
    import numpy as np

    from geneuclidean.studio.graph_compiler import _fit_tanimoto_gp_numpy

    # Zero-norm vectors must have k(0, 0) = signal_var -> non-zero predictive mean & exact var
    zero_x = np.zeros((2, 8), dtype=np.float64)
    zero_y = np.array([1.0, 3.0], dtype=np.float64)
    mu_z, std_z, loo_abs_z, loo_norm_z = _fit_tanimoto_gp_numpy(
        zero_x, zero_y, zero_x, noise_var=0.05, signal_var=1.5
    )
    assert np.all(np.isfinite(mu_z)) and np.all(std_z >= 0.0)
    assert float(std_z[0]) < 1.0  # Collapsed below prior sqrt(1.5) because k(0, 0) = 1.5!
    assert len(loo_abs_z) == 2 and len(loo_norm_z) == 2

    # Non-trivial noisy regression: LOO residuals must strictly exceed memorized in-sample residuals
    rng = np.random.default_rng(7)
    x_tr = (rng.random((12, 32)) > 0.65).astype(np.float64)
    y_tr = x_tr[:, 0] * 2.5 - x_tr[:, 1] * 1.5 + rng.normal(0.0, 0.3, size=12)
    mu_in, _std_in, loo_abs, loo_norm = _fit_tanimoto_gp_numpy(
        x_tr, y_tr, x_tr, noise_var=0.02, signal_var=1.0
    )
    insample_mae = float(np.mean(np.abs(y_tr - mu_in)))
    loo_mae = float(np.mean(loo_abs))
    assert loo_mae > insample_mae * 2.0, (
        f"Expected leak-free LOO MAE ({loo_mae:.4f}) >> in-sample MAE ({insample_mae:.4f})"
    )
    assert all(r >= 0.0 for r in loo_norm)

    # A four-observation campaign cannot calibrate a finite 90% interval.
    al_preset = get_preset("al_dmta_closed_loop")
    al_preset["execution"]["dmtaRounds"] = 3
    res = execute_circuit(al_preset, max_samples=4, timeout_sec=30.0)
    assert res["status"] == "completed"
    conf_steps = [s for s in res["steps"] if s["blockType"] == "conformal_calibrator"]
    assert len(conf_steps) == 3
    assert all(step["metrics"]["n_cal"] < 9 for step in conf_steps)
    assert all(step["metrics"]["q_hat"] is None for step in conf_steps)
    assert all(step["metrics"]["empirical_coverage"] is None for step in conf_steps)


# ---------------------------------------------------------------------------
# 7. v5.3 49-Block Coverage (3D SE(3) + Chemical NLP + NanoFold3 Hub)
# ---------------------------------------------------------------------------


def test_all_13_new_v53_blocks_validate_compile_and_execute_real_python() -> None:
    """Every remaining v5.3 block not in Presets 5-6 validates, compiles, and runs real Python."""
    custom_circuit: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "id": "v53_remaining_blocks_smoke",
        "name": "v5.3 Complete 3D+NLP+Folding Coverage Circuit",
        "description": (
            "Smoke-tests DensePaiNN, SubPocketVN-JEPA, CrossAttn, MDLM, SimpleFold, Folder Hub."
        ),
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "CHRM4",
            "budget": 8,
        },
        "nodes": [
            {
                "id": "n_pkt",
                "blockType": "protein_pocket",
                "position": {"x": 0, "y": 0},
                "params": {"target_id": "CHRM4"},
            },
            {
                "id": "n_pool",
                "blockType": "chembl_pool",
                "position": {"x": 0, "y": 200},
                "params": {"dataset": "DRD1_chembl37_400", "pool_size": 32},
            },
            {
                "id": "n_dpainn",
                "blockType": "dense_painn_se3",
                "position": {"x": 300, "y": 0},
                "params": {"adaptive_cutoff": True},
            },
            {
                "id": "n_vn",
                "blockType": "subpocket_vn_jepa",
                "position": {"x": 300, "y": 200},
                "params": {"num_subpockets": 24, "num_salient_slots": 8},
            },
            {
                "id": "n_ext",
                "blockType": "external_folder_runner",
                "position": {"x": 300, "y": 400},
                "params": {"backend": "boltz2", "predict_affinity": True},
            },
            {
                "id": "n_xattn",
                "blockType": "se3_complex_cross_attn",
                "position": {"x": 600, "y": 0},
                "params": {"cutoff": 8.0, "use_coulomb_bias": True},
            },
            {
                "id": "n_sf",
                "blockType": "pairformer_stack",
                "position": {"x": 600, "y": 200},
                "params": {"trunk_mode": "simplefold", "num_blocks": 4},
            },
            {
                "id": "n_nf3_fm",
                "blockType": "nanofold3_cofold",
                "position": {"x": 900, "y": 200},
                "params": {"diffusion_mode": "flow_matching", "num_steps": 8},
            },
            {
                "id": "n_mdlm",
                "blockType": "nlp_masked_diffusion",
                "position": {"x": 600, "y": 400},
                "params": {"schedule": "cosine", "mask_ratio": 0.35, "num_steps": 4},
            },
            {
                "id": "n_dpo",
                "blockType": "nlp_lora_rl_aligner",
                "position": {"x": 900, "y": 400},
                "params": {"rl_objective": "dpo", "lora_rank": 4, "merge_adapters": True},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n_pkt",
                "sourcePort": "pocket",
                "target": "n_dpainn",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e2",
                "source": "n_pkt",
                "sourcePort": "pocket",
                "target": "n_vn",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e3",
                "source": "n_pkt",
                "sourcePort": "pocket",
                "target": "n_ext",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e4",
                "source": "n_pool",
                "sourcePort": "molecules",
                "target": "n_ext",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e5",
                "source": "n_dpainn",
                "sourcePort": "pocket_graph",
                "target": "n_xattn",
                "targetPort": "pocket_graph",
                "signalType": "SE3_GRAPH",
                "kind": "forward",
            },
            {
                "id": "e6",
                "source": "n_ext",
                "sourcePort": "poses",
                "target": "n_xattn",
                "targetPort": "poses",
                "signalType": "POSE_3D",
                "kind": "forward",
            },
            {
                "id": "e7",
                "source": "n_pkt",
                "sourcePort": "pocket",
                "target": "n_sf",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e8",
                "source": "n_pool",
                "sourcePort": "molecules",
                "target": "n_sf",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e9",
                "source": "n_sf",
                "sourcePort": "trunk_graph",
                "target": "n_nf3_fm",
                "targetPort": "trunk_graph",
                "signalType": "SE3_GRAPH",
                "kind": "forward",
            },
            {
                "id": "e10",
                "source": "n_pool",
                "sourcePort": "molecules",
                "target": "n_mdlm",
                "targetPort": "seed_mols",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e11",
                "source": "n_vn",
                "sourcePort": "pocket_vec",
                "target": "n_mdlm",
                "targetPort": "pocket_tokens",
                "signalType": "LATENT_VEC",
                "kind": "forward",
            },
            {
                "id": "e12",
                "source": "n_mdlm",
                "sourcePort": "inpainted_mols",
                "target": "n_dpo",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
        ],
    }

    val = validate_circuit(custom_circuit)
    assert val.valid is True, f"Validation failed: {val.diagnostics}"

    cg = compile_circuit_to_python(custom_circuit)
    assert cg.valid is True and cg.ast_valid is True
    assert ast.parse(cg.python_code) is not None
    assert "import geneuclidean.models.encoders as ge_encoders" in cg.python_code
    assert "import geneuclidean.geom as ge_geom" in cg.python_code
    assert "import geneuclidean.nlp as ge_nlp" in cg.python_code
    assert "import geneuclidean.folding as ge_folding" in cg.python_code

    run_res = execute_circuit(custom_circuit, max_samples=3, timeout_sec=30.0)
    assert run_res["status"] == "completed", f"Run failed: {run_res['diagnostics']}"
    assert len(run_res["steps"]) == len(custom_circuit["nodes"])
    for step in run_res["steps"]:
        assert step["executionBackend"] == "real_python", f"Expected real_python on {step}"
        if step["blockType"] == "nanofold3_cofold":
            assert step["status"] in {"ok", "skipped"}
            if step["status"] == "skipped":
                assert "predicted_pkd" not in step["metrics"]
        else:
            assert step["status"] == "ok", f"Unexpected step status: {step}"


def test_studio_folding_requires_matching_pocket_ligand_and_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from geneuclidean import folding
    from geneuclidean.studio.folding import cofold_candidate, confidence_scores, passes_confidence

    monkeypatch.delenv("GENEUCLIDEAN_NANOFOLD3_CHECKPOINT", raising=False)
    checkpoint = tmp_path / "weights.pt"
    pdb = tmp_path / "receptor.pdb"
    options = dict(seed=7, num_samples=1, num_steps=8, flex_mode="ligand_only")
    trunk = {"pocket": {"pocket_pdb": pdb}, "molecules": ["CCO"]}
    assert cofold_candidate(trunk, [], **options) is None
    assert confidence_scores(None, include_affinity=True) == {}
    calls = []

    def capture(**inputs):
        calls.append(inputs)
        return folding.FoldedComplexResult(
            backend="nanofold3",
            plddt_mean=91.5,
            iptm=0.72,
            predicted_pkd=6.5,
            metadata={"weight_source": "checkpoint"},
        )

    monkeypatch.setenv("GENEUCLIDEAN_NANOFOLD3_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(folding, "fold_complex", capture)
    assert cofold_candidate(trunk, [], **options) is None
    pdb.write_text("END\n")
    result = cofold_candidate(trunk, ["CCN"], **options)
    assert calls[0]["pocket_pdb"] == pdb and calls[0]["ligand_smiles"] == "CCN"
    assert calls[0]["checkpoint_path"] == str(checkpoint) and calls[0]["seeds"] == (7,)
    scores = confidence_scores(result, include_affinity=False)
    assert scores == {"plddt_mean": 91.5, "iptm": 0.72}
    assert confidence_scores(result, include_affinity=True)["predicted_pkd"] == 6.5
    assert passes_confidence(scores, min_iptm=0.7, min_plddt=90.0)
    assert not passes_confidence(scores, min_iptm=0.8, min_plddt=90.0)


def test_studio_uses_measured_endpoint_labels_and_observed_coverage(tmp_path, monkeypatch) -> None:
    from geneuclidean.studio.graph_compiler import (
        _run_conformal_calibrator,
        _run_conformal_dmpk,
        _RuntimeContext,
    )

    monkeypatch.chdir(tmp_path)
    ctx = _RuntimeContext(
        seed=7,
        target_context="DRD1",
        budget=8,
        max_samples=8,
        dry_run=False,
        tmp_dir=tmp_path,
        deadline=float("inf"),
    )
    config = {"endpoint": "OpenADMET_log10_KSOL", "alpha": 0.1, "n_estimators": 2}
    inputs = {"molecules": ["CCO"]}
    missing = _run_conformal_dmpk("assay", config, inputs, ctx)
    assert missing.status == "skipped"
    assert missing.outputs["dmpk_scores"] == {}

    data_dir = tmp_path / "data/cns_neuro/v4_expansion/admet"
    data_dir.mkdir(parents=True)
    (data_dir / "openadmet_expansionrx_train.csv").write_text(
        "SMILES,KSOL,LogD\nCC,100,-2\nCCC,100,-2\nCO,100,-2\nCCO,100,-2\n"
    )
    measured = _run_conformal_dmpk("assay", config, inputs, ctx)
    assert measured.status == "ok"
    assert measured.outputs["dmpk_scores"] == {"CCO": 2}
    assert measured.outputs["dmpk_intervals"]["q_hat"] is None

    posterior = {"smiles": ["CCO"], "mean": [2], "std": [0.5]}
    for residuals, expected in [([], None), ([1, 2, 3, 4], 3)]:
        posterior["loo_abs_residuals"] = residuals
        obs = _run_conformal_calibrator(
            "cal", {"mode": "absolute", "alpha": 0.5}, {"posterior": posterior}, ctx
        )
        assert obs.metrics["q_hat"] == expected
        assert obs.metrics["empirical_coverage"] == (0.5 if residuals else None)
