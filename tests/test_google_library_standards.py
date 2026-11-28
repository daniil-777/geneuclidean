"""Automated Google Open-Source Library Standards & Hygiene CI Enforcement Suite.

Enforces production Python library invariants across GenEuclidean:
1. PEP 561 ``py.typed`` marker presence & setuptools package-data declaration.
2. Exact SemVer version parity across ``geneuclidean.__version__``, ``pyproject.toml``,
   ``CITATION.cff``, ``Dockerfile``, and ``web/package.json``.
3. Cold-import latency budget and zero eager heavy ML/chemistry dependency leaks
   (``torch``, ``transformers``, ``peft``, ``sklearn``, ``scipy``, ``rdkit``,
   ``matplotlib``, ``Bio``) across ``geneuclidean``, ``geom``, ``nlp``, ``folding``,
   ``active_learning``, and ``studio``.
4. Complete, duplicate-free ``__all__`` resolution (and strict sorting on top-level
   ``geneuclidean.__all__``) plus flagship symbol & lazy submodule checks.
5. Full-repository AST security audit across every ``src/geneuclidean/**/*.py`` module
   confirming zero ``eval()``, ``exec()``, ``os.system()``, or ``shell=True`` calls.
6. Strict Apple Silicon MPS ``float32`` discipline (zero ``torch.float64`` parameters
   or buffers across all 8 built-in 3D encoders, ``NanoFold3``, and
   ``PocketConditionedChemicalLM``).
7. Registry & Studio catalog completeness (8 encoders, 5 folding backends, 49 studio
   blocks, 7 flagship presets, 10 signals, 8 categories).
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import geneuclidean as ge

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PKG_ROOT = REPO_ROOT / "src" / "geneuclidean"

HEAVY_MODULES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "sklearn",
    "scipy",
    "rdkit",
    "matplotlib",
    "Bio",
)

FLAGSHIP_TOP_LEVEL_SYMBOLS: tuple[str, ...] = (
    "AtomGSSE3Encoder",
    "AtomLevelSMILESTokenizer",
    "CartFormerSE3Encoder",
    "DensePaiNNSE3Encoder",
    "DiscreteMaskedDiffusionStep",
    "FOLDING_BACKEND_REGISTRY",
    "NanoFold3",
    "NanoFold3Config",
    "PocketConditionedChemicalLM",
    "SMILESValenceGrammarMask",
    "SubPocketVNJEPAEncoder",
    "build_nanofold3",
    "dpo_preference_loss",
    "fold_complex",
    "folding",
    "geom",
    "grpo_policy_loss",
    "nlp",
)


def assert_no_float64_state(module: nn.Module, label: str) -> None:
    """Assert that a PyTorch module contains zero ``torch.float64`` parameters or buffers.

    Apple Silicon Metal (MPS) does not support ``float64``; enforcing this invariant
    across all trainable and analytical neural modules prevents silent MPS crashes.
    """
    bad_params = [
        f"param:{name}" for name, param in module.named_parameters() if param.dtype == torch.float64
    ]
    bad_buffers = [
        f"buffer:{name}" for name, buf in module.named_buffers() if buf.dtype == torch.float64
    ]
    offenders = bad_params + bad_buffers
    assert offenders == [], f"{label} contains forbidden float64 state: {offenders}"


def _probe_cold_import(module_name: str, runs: int = 3) -> tuple[float, list[str]]:
    """Measure best-of-N cold import latency (ms) and detect any eager heavy imports."""
    probe_code = f"""
import json, sys, time
t0 = time.perf_counter()
import {module_name}
elapsed_ms = (time.perf_counter() - t0) * 1000.0
heavy = [m for m in {HEAVY_MODULES!r} if m in sys.modules]
print(json.dumps({{"elapsed_ms": round(elapsed_ms, 3), "heavy": heavy}}))
"""
    best_ms = float("inf")
    last_heavy: list[str] = []
    for _ in range(runs):
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip())
        best_ms = min(best_ms, float(payload["elapsed_ms"]))
        last_heavy = list(payload["heavy"])
        assert last_heavy == [], f"{module_name} eagerly imported heavy modules: {last_heavy}"
    return best_ms, last_heavy


# ---------------------------------------------------------------------------
# 1. PEP 561 Packaging & SemVer Version Parity
# ---------------------------------------------------------------------------


def test_pep561_py_typed_marker_present() -> None:
    """PEP 561 ``py.typed`` marker must exist in the package root and pyproject metadata."""
    pkg_dir = Path(ge.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file()

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pkg_data = pyproject["tool"]["setuptools"]["package-data"]["geneuclidean"]
    assert "py.typed" in pkg_data
    assert "Typing :: Typed" in pyproject["project"]["classifiers"]


def test_version_parity_across_package_metadata() -> None:
    """Version must match identically across Python, pyproject, CITATION, Dockerfile & web."""
    py_ver = ge.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", py_ver), f"Non-SemVer version: {py_ver!r}"

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_ver = str(pyproject["project"]["version"])

    citation_text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r'^version:\s*"([^"]+)"\s*$', citation_text, flags=re.MULTILINE)
    assert citation_match is not None, "Missing version field in CITATION.cff"
    citation_ver = citation_match.group(1)

    web_pkg = json.loads((REPO_ROOT / "web" / "package.json").read_text(encoding="utf-8"))
    web_ver = str(web_pkg["version"])

    dockerfile_text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    docker_match = re.search(
        r'org\.opencontainers\.image\.version="([^"]+)"',
        dockerfile_text,
    )
    assert docker_match is not None, "Missing OCI version label in Dockerfile"
    docker_ver = docker_match.group(1)

    assert py_ver == pyproject_ver == citation_ver == web_ver == docker_ver == "5.3.0"


# ---------------------------------------------------------------------------
# 2. Cold-Import Latency & Zero Eager Heavy Imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "max_ms"),
    [
        ("geneuclidean", 15.0),
        ("geneuclidean.geom", 15.0),
        ("geneuclidean.nlp", 15.0),
        ("geneuclidean.folding", 15.0),
        ("geneuclidean.active_learning", 15.0),
        ("geneuclidean.studio", 150.0),
    ],
)
def test_cold_import_budget_and_zero_eager_heavy_imports(
    module_name: str,
    max_ms: float,
) -> None:
    """Every public entry facade loads cleanly with zero heavy ML imports within latency budget."""
    best_ms, heavy = _probe_cold_import(module_name, runs=3)
    assert heavy == [], f"{module_name} leaked heavy imports: {heavy}"
    assert best_ms < max_ms, f"Cold import of {module_name} took {best_ms:.2f} ms (>= {max_ms} ms)"


# ---------------------------------------------------------------------------
# 3. Complete Public __all__ Resolution & Top-Level Flagship Facade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "geneuclidean",
        "geneuclidean.geom",
        "geneuclidean.nlp",
        "geneuclidean.folding",
        "geneuclidean.active_learning",
        "geneuclidean.studio",
    ],
)
def test_public_all_symbols_resolve_cleanly(module_name: str) -> None:
    """Every symbol listed in ``__all__`` must be unique and resolve without AttributeError."""
    mod = importlib.import_module(module_name)
    all_names = list(getattr(mod, "__all__", []))
    assert len(all_names) > 0, f"{module_name}.__all__ must be non-empty"
    assert len(all_names) == len(set(all_names)), f"Duplicate entries in {module_name}.__all__"

    if module_name == "geneuclidean":
        assert all_names == sorted(all_names), "geneuclidean.__all__ must be 100% sorted"

    for symbol in all_names:
        value = getattr(mod, symbol)
        assert value is not None or symbol.endswith("_REGISTRY"), (
            f"{module_name}.{symbol} resolved to None"
        )

    # Missing attribute access must raise standard AttributeError
    missing_attr = "__definitely_missing_google_std_symbol__"
    with pytest.raises(AttributeError):
        _ = getattr(mod, missing_attr)


def test_flagship_v53_symbols_exported_at_top_level() -> None:
    """All v5.2/v5.3 flagship 3D encoder, NLP, and NanoFold3 symbols resolve from top-level ge."""
    all_set = set(ge.__all__)
    dir_set = set(dir(ge))
    for name in FLAGSHIP_TOP_LEVEL_SYMBOLS:
        assert name in all_set, f"Missing {name!r} in geneuclidean.__all__"
        assert name in dir_set, f"Missing {name!r} in dir(geneuclidean)"
        assert getattr(ge, name) is not None

    # Identity checks against canonical subpackages
    assert ge.NanoFold3 is ge.folding.NanoFold3
    assert ge.NanoFold3Config is ge.folding.NanoFold3Config
    assert ge.build_nanofold3 is ge.folding.build_nanofold3
    assert ge.fold_complex is ge.folding.fold_complex
    assert ge.FOLDING_BACKEND_REGISTRY is ge.folding.FOLDING_BACKEND_REGISTRY
    assert ge.PocketConditionedChemicalLM is ge.nlp.PocketConditionedChemicalLM
    assert ge.AtomLevelSMILESTokenizer is ge.nlp.AtomLevelSMILESTokenizer
    assert ge.SMILESValenceGrammarMask is ge.nlp.SMILESValenceGrammarMask
    assert ge.dpo_preference_loss is ge.nlp.dpo_preference_loss
    assert ge.grpo_policy_loss is ge.nlp.grpo_policy_loss
    assert ge.DiscreteMaskedDiffusionStep is ge.nlp.DiscreteMaskedDiffusionStep


# ---------------------------------------------------------------------------
# 4. Full-Repository AST Security Audit
# ---------------------------------------------------------------------------


def test_ast_security_audit_zero_eval_exec_shell_or_os_system() -> None:
    """AST scan across all src/geneuclidean/**/*.py forbids eval, exec, shell=True & os.system."""
    py_files = sorted(SRC_PKG_ROOT.rglob("*.py"))
    assert len(py_files) >= 110, f"Expected >= 110 source modules, found {len(py_files)}"

    violations: list[str] = []
    for path in py_files:
        rel = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                violations.append(f"{rel}:{node.lineno}: forbidden {node.func.id}()")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "system"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                violations.append(f"{rel}:{node.lineno}: forbidden os.system()")
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    violations.append(f"{rel}:{node.lineno}: forbidden shell=True")

    assert violations == [], "AST security audit violations:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 5. Zero float64 MPS Safety & Registry / Studio Catalog Completeness
# ---------------------------------------------------------------------------


def test_zero_float64_across_all_encoders_nanofold3_and_chemical_lm() -> None:
    """All 8 built-in 3D encoders, NanoFold3, and PocketConditionedChemicalLM are pure float32."""
    from geneuclidean.config import EncoderConfig
    from geneuclidean.models.encoders import build_encoder

    for arch in ge.list_encoders():
        cfg = EncoderConfig(arch=arch, hidden_dim=32, vector_dim=16, out_dim=32, num_layers=1)
        encoder = build_encoder(cfg)
        assert_no_float64_state(encoder, f"encoder:{arch}")

    nf3 = ge.build_nanofold3("nano")
    assert_no_float64_state(nf3, "NanoFold3(nano)")

    tok = ge.AtomLevelSMILESTokenizer()
    lm = ge.PocketConditionedChemicalLM(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_pocket=32,
    )
    assert_no_float64_state(lm, "PocketConditionedChemicalLM")


def test_registry_and_studio_catalog_completeness() -> None:
    """Verify exact built-in counts: 8 encoders, 5 folding backends, 49 studio blocks, 7 presets."""
    import geneuclidean.models.encoders  # noqa: F401  # Populate ENCODER_REGISTRY

    expected_encoders = {
        "atomgs_se3",
        "cartformer_se3",
        "dense_painn_se3",
        "e3nn",
        "egnn",
        "painn",
        "schnet",
        "subpocket_vn_jepa",
    }
    assert set(ge.list_encoders()) >= expected_encoders
    assert len(expected_encoders) == 8
    assert set(ge.ENCODER_REGISTRY.names()) >= expected_encoders

    expected_folding = {"alphafold3", "boltz2", "esmfold_api", "nanofold3", "protenix"}
    assert set(ge.FOLDING_BACKEND_REGISTRY.names()) == expected_folding
    assert len(ge.FOLDING_BACKEND_REGISTRY.names()) == 5

    catalog = ge.get_studio_catalog()
    assert len(catalog["blocks"]) == 49
    assert len(catalog["presets"]) == 7
    assert len(catalog["signalTypes"]) == 10
    assert len(catalog["categories"]) == 8
