"""Zero-external-dependency local HTTP API server for GenEuclidean Circuit Studio.

Uses strictly Python stdlib :class:`http.server.ThreadingHTTPServer` so running
``geneuclidean studio --host 127.0.0.1 --port 8765`` requires zero FastAPI/Uvicorn
dependencies and keeps cold imports instantaneous.

Endpoints:
- ``OPTIONS /api/*``          — CORS preflight (HTTP 204)
- ``GET  /api/health``        — Live backend status, versions, v5.1 AL flag, catalog count
- ``GET  /api/catalog``       — Full 10-signal, 8-category, 49-block, 7-preset catalog JSON
- ``POST /api/circuit/validate`` — Strict ``CircuitGraphV1`` validation diagnostics
- ``POST /api/circuit/codegen``  — AST-verified Python SDK code generation + SHA-256
- ``POST /api/circuit/run``      — Safe allowlisted local Python execution + SHA-256 ledger
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import geneuclidean
from geneuclidean.studio.catalog import BLOCK_CATALOG, export_catalog_payload
from geneuclidean.studio.graph_compiler import (
    compile_circuit_to_python,
    execute_circuit,
    has_active_learning_v51,
    validate_circuit,
)

MAX_REQUEST_BYTES: int = 2 * 1024 * 1024  # 2 MiB hard cap on JSON POST payloads


def _package_version_or_none(dist_name: str) -> str | None:
    """Query installed distribution version without eagerly importing heavy packages."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_health_payload(*, start_monotonic: float) -> dict[str, Any]:
    """Construct the ``GET /api/health`` status payload with zero heavy imports."""
    return {
        "status": "ok",
        "version": geneuclidean.__version__,
        "python": platform.python_version(),
        "rdkit": _package_version_or_none("rdkit"),
        "torch": _package_version_or_none("torch"),
        "activeLearningV51": has_active_learning_v51(),
        "catalogCount": len(BLOCK_CATALOG),
        "uptimeSec": round(max(0.0, time.monotonic() - start_monotonic), 3),
    }


class StudioHTTPServer(ThreadingHTTPServer):
    """Multi-threaded stdlib HTTP server holding Circuit Studio configuration."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        cors_origin: str = "*",
        quiet: bool = False,
    ) -> None:
        self.cors_origin = cors_origin
        self.quiet = bool(quiet)
        self.start_monotonic = time.monotonic()
        super().__init__(server_address, StudioRequestHandler)


def _reject_json_constant(token: str) -> Any:
    """Reject non-RFC-8259 JSON constants (NaN, Infinity, -Infinity)."""
    raise ValueError(f"Non-standard JSON constant {token!r} is forbidden.")


class StudioRequestHandler(BaseHTTPRequestHandler):
    """Strict JSON + CORS request handler for the GenEuclidean Circuit Studio API."""

    server: StudioHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not self.server.quiet:
            super().log_message(format, *args)

    def _write_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", self.server.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.send_header("Access-Control-Max-Age", "600")

    def _send_json(self, status: int, payload: Any) -> None:
        raw = (
            json.dumps(payload, separators=(",", ":"), allow_nan=False, default=str) + "\n"
        ).encode("utf-8")
        self.send_response(status)
        self._write_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        if status == int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE):
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_object(self) -> tuple[dict[str, Any] | None, int | None, str | None]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return None, int(HTTPStatus.BAD_REQUEST), "Missing Content-Length header."
        try:
            content_len = int(raw_len)
        except ValueError:
            return None, int(HTTPStatus.BAD_REQUEST), "Invalid Content-Length header."
        if content_len < 0:
            return None, int(HTTPStatus.BAD_REQUEST), "Negative Content-Length header."
        if content_len > MAX_REQUEST_BYTES:
            return (
                None,
                int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
                f"Request payload exceeds {MAX_REQUEST_BYTES} byte limit.",
            )
        body = self.rfile.read(content_len)
        try:
            decoded = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return None, int(HTTPStatus.BAD_REQUEST), f"Invalid JSON payload: {exc}"
        if not isinstance(decoded, dict):
            return None, int(HTTPStatus.BAD_REQUEST), "JSON request body must be an object."
        return decoded, None, None

    def do_OPTIONS(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self.send_response(int(HTTPStatus.NO_CONTENT))
            self._write_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send_json(int(HTTPStatus.NOT_FOUND), {"error": "Not found", "path": path})

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/health":
                self._send_json(
                    int(HTTPStatus.OK),
                    build_health_payload(start_monotonic=self.server.start_monotonic),
                )
                return
            if path == "/api/catalog":
                self._send_json(int(HTTPStatus.OK), export_catalog_payload())
                return
            self._send_json(int(HTTPStatus.NOT_FOUND), {"error": "Not found", "path": path})
        except Exception as exc:
            self._send_json(
                int(HTTPStatus.INTERNAL_SERVER_ERROR),
                {"error": f"Internal server error: {type(exc).__name__}"},
            )

    def do_POST(self) -> None:
        try:
            self._handle_post()
        except Exception as exc:
            self._send_json(
                int(HTTPStatus.INTERNAL_SERVER_ERROR),
                {"error": f"Internal server error: {type(exc).__name__}"},
            )

    def _handle_post(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/circuit/validate", "/api/circuit/codegen", "/api/circuit/run"):
            self._send_json(int(HTTPStatus.NOT_FOUND), {"error": "Not found", "path": path})
            return

        payload, err_status, err_msg = self._read_json_object()
        if err_status is not None or payload is None:
            self._send_json(err_status or int(HTTPStatus.BAD_REQUEST), {"error": err_msg})
            return

        if path == "/api/circuit/validate":
            circuit = (
                payload["circuit"]
                if "circuit" in payload
                and "nodes" not in payload
                and isinstance(payload["circuit"], dict)
                else payload
            )
            val_res = validate_circuit(circuit)
            self._send_json(int(HTTPStatus.OK), val_res.to_dict())
            return

        if path == "/api/circuit/codegen":
            circuit = (
                payload["circuit"]
                if "circuit" in payload
                and "nodes" not in payload
                and isinstance(payload["circuit"], dict)
                else payload
            )
            cg_res = compile_circuit_to_python(circuit)
            status = int(HTTPStatus.OK) if cg_res.valid else int(HTTPStatus.UNPROCESSABLE_ENTITY)
            self._send_json(status, cg_res.to_dict())
            return

        # POST /api/circuit/run
        if "circuit" in payload and isinstance(payload["circuit"], dict):
            circuit = payload["circuit"]
            raw_opts = payload.get("options", {})
        else:
            circuit = payload
            raw_opts = {}

        if not isinstance(raw_opts, dict):
            self._send_json(
                int(HTTPStatus.BAD_REQUEST),
                {"error": "'options' must be a JSON object when provided."},
            )
            return

        dry_run = bool(raw_opts.get("dryRun", False))
        raw_max_samples = raw_opts.get("maxSamples", 8)
        raw_timeout = raw_opts.get("timeoutSec", 30.0)
        if (
            isinstance(raw_max_samples, bool)
            or not isinstance(raw_max_samples, int)
            or raw_max_samples < 1
        ):
            self._send_json(
                int(HTTPStatus.BAD_REQUEST),
                {"error": "options.maxSamples must be an integer >= 1."},
            )
            return
        if (
            isinstance(raw_timeout, bool)
            or not isinstance(raw_timeout, (int, float))
            or not math.isfinite(float(raw_timeout))
            or float(raw_timeout) <= 0.0
        ):
            self._send_json(
                int(HTTPStatus.BAD_REQUEST),
                {"error": "options.timeoutSec must be a finite positive number."},
            )
            return

        run_res = execute_circuit(
            circuit,
            dry_run=dry_run,
            max_samples=raw_max_samples,
            timeout_sec=float(raw_timeout),
        )
        http_status = (
            int(HTTPStatus.UNPROCESSABLE_ENTITY)
            if run_res["status"] == "validation_error"
            else int(HTTPStatus.OK)
        )
        self._send_json(http_status, run_res)


def create_studio_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    cors_origin: str = "*",
    quiet: bool = False,
) -> StudioHTTPServer:
    """Create a configured :class:`StudioHTTPServer` bound to ``(host, port)``."""
    return StudioHTTPServer((host, int(port)), cors_origin=cors_origin, quiet=quiet)


def run_studio_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    cors_origin: str = "*",
    quiet: bool = False,
) -> None:
    """Run the local GenEuclidean Circuit Studio HTTP server until interrupted."""
    server = create_studio_server(host=host, port=port, cors_origin=cors_origin, quiet=quiet)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


serve_studio = run_studio_server


__all__ = [
    "MAX_REQUEST_BYTES",
    "StudioHTTPServer",
    "StudioRequestHandler",
    "build_health_payload",
    "create_studio_server",
    "run_studio_server",
    "serve_studio",
]
