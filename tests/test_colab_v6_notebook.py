"""CPU-only structural validation of the v6 ChemLM A100 Colab notebook.

These tests catch the failure modes that only show up once the notebook is already
running on a paid GPU: a syntax error in cell 12, a cell referencing a name no earlier
cell defines, or a ``subprocess`` call that breaks when the interpreter path contains a
space. They never execute the notebook.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "GenEuclidean_v6_ChemLM_A100_Colab.ipynb"

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Hardware detection",
    "Experiment configuration",
    "Environment",
    "Correctness gate",
    "Data:",
    "E1:",
    "E2:",
    "E3:",
    "E4:",
    "E6:",
    "Results, figures and downloadable bundle",
)


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    assert NOTEBOOK_PATH.is_file(), f"notebook missing: {NOTEBOOK_PATH}"
    payload: dict[str, Any] = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module")
def code_cells(notebook: dict[str, Any]) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat_4(notebook: dict[str, Any]) -> None:
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    assert notebook["cells"], "notebook has no cells"
    for idx, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in ("code", "markdown"), idx
        assert isinstance(cell["source"], list), idx
        assert isinstance(cell["metadata"], dict), idx
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None, idx
            assert cell["outputs"] == [], f"cell {idx} ships stale outputs"


def test_notebook_requests_a_gpu_runtime(notebook: dict[str, Any]) -> None:
    meta = notebook["metadata"]
    assert meta.get("accelerator") == "GPU"
    assert meta.get("colab", {}).get("gpuType") == "A100"


def test_every_code_cell_parses(code_cells: list[str]) -> None:
    """A syntax error must never reach a paid GPU runtime."""
    for idx, src in enumerate(code_cells):
        try:
            ast.parse(src)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {idx} line {exc.lineno}: {exc.msg}")


def test_all_required_sections_present(notebook: dict[str, Any]) -> None:
    markdown = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    missing = [s for s in REQUIRED_SECTIONS if s not in markdown]
    assert not missing, f"missing notebook sections: {missing}"


def test_colab_form_widgets_and_titles_present(code_cells: list[str]) -> None:
    assert sum(1 for s in code_cells if "#@param" in s) >= 2, "no #@param form cells"
    titled = sum(1 for s in code_cells if "#@title" in s)
    assert titled >= len(code_cells) - 1, "most code cells should carry a #@title"


def test_subprocess_calls_never_use_shell_with_interpolated_paths(
    code_cells: list[str],
) -> None:
    """``shell=True`` plus an interpolated path breaks on paths containing spaces.

    This is not hypothetical: it was found by running this notebook from a checkout whose
    path contains spaces, where ``sys.executable`` made ``/bin/sh`` split the path
    mid-way and the correctness gate failed for an unrelated-looking reason.
    """
    offenders: list[tuple[int, str]] = []
    for idx, src in enumerate(code_cells):
        for line_no, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # a comment explaining why shell=True is avoided is fine
            if "shell=True" in stripped:
                offenders.append((idx, f"line {line_no}: {stripped}"))
    assert not offenders, f"shell=True found in notebook cells: {offenders}"


def test_subprocess_calls_pass_argv_lists(code_cells: list[str]) -> None:
    joined = "\n".join(code_cells)
    if "subprocess.run" not in joined:
        pytest.skip("notebook makes no subprocess calls")
    assert "sys.executable" in joined, "subprocess calls should use sys.executable"


def test_names_used_across_cells_are_defined_by_an_earlier_cell(
    code_cells: list[str],
) -> None:
    """Catch cross-cell NameErrors without executing anything.

    Only the notebook's own shared globals are checked; imports, locals, builtins and
    comprehension variables are ignored.
    """
    shared = {
        "HW",
        "CFG",
        "IN_COLAB",
        "IS_A100",
        "GPU_NAME",
        "GPU_MEM_GB",
        "DRIVE_DIR",
        "REPO",
        "WORK",
        "DATA_STATS",
        "CROSS",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "REPORT",
        "made",
        "bundle",
        "include",
        "RUN_E1_SCALES",
        "RUN_E2_PRIOR",
        "RUN_E3_PROPERTY",
        "RUN_E4_MOE",
        "RUN_E5_DECODING",
        "RUN_E6_POCKET",
    }
    defined: set[str] = set()
    problems: list[str] = []

    for idx, src in enumerate(code_cells):
        tree = ast.parse(src)
        loaded: set[str] = set()
        assigned: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    loaded.add(node.id)
                else:
                    assigned.add(node.id)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    assigned.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                assigned.add(node.name)

        for name in sorted((loaded & shared) - assigned - defined):
            problems.append(f"cell {idx} uses {name!r} before any earlier cell defines it")
        defined |= assigned

    assert not problems, "cross-cell name errors:\n  " + "\n  ".join(problems)


def test_notebook_documents_the_scaling_saturation_finding(notebook: dict[str, Any]) -> None:
    """The notebook must not imply that a bigger model yields better molecules."""
    markdown = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ).lower()
    assert "saturate" in markdown, "scaling-saturation caveat is missing"
    assert "novelty" in markdown, "memorization/novelty caveat is missing"


def test_notebook_degrades_without_an_a100(code_cells: list[str]) -> None:
    """Run-all must still complete on a T4 or CPU runtime."""
    joined = "\n".join(code_cells)
    assert "if not IS_A100" in joined, "no automatic budget reduction for non-A100 runtimes"
    assert "OutOfMemoryError" in joined, "scale sweep does not handle OOM"


def test_export_bundle_only_includes_figures_this_notebook_made(
    code_cells: list[str],
) -> None:
    """Globbing results/figures/*.png would sweep in unrelated prior figures."""
    bundle_cells = [s for s in code_cells if "tarfile.open" in s]
    assert bundle_cells, "no export bundle cell found"
    for src in bundle_cells:
        assert 'glob("*.png")' not in src, (
            "bundle cell globs the whole figures directory; it must use the `made` list"
        )
