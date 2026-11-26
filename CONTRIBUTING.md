# Contributing to GenEuclidean

Thank you for contributing to **GenEuclidean**. This library follows strict scientific, typing, security, and performance standards so every subsystem remains fast, deterministic, and trustworthy across Apple Silicon MPS, NVIDIA CUDA, and CPU.

---

## 1. Development Environment Setup

Use [`uv`](https://docs.astral.sh/uv/) with Python **3.11** or **3.12**:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[all]"   # or ".[dev]" for lightweight CPU dev
```

> **ARM64 / macOS RDKit Pin:** `pyproject.toml` pins `rdkit>=2026.3,<2026.9` to avoid an ARM64 `MurckoDecompose` runtime regression (`bad any cast`). Keep this constraint intact unless verified on Apple Silicon.

For the interactive TypeScript / React Web & Circuit Studio in `web/`:

```bash
cd web && npm ci && npm run typecheck && npm run build
```

---

## 2. Mandatory Quality Gates

Every PR must pass all checks below with **zero errors and zero warnings** (matching [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/mypy src
.venv/bin/pytest -q tests/test_google_library_standards.py
.venv/bin/pytest -q
```

### Type Annotations (PEP 561)
- All production modules under `src/geneuclidean/` must pass `mypy src` with `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`, `no_implicit_optional = true`, and `warn_unused_ignores = true`.
- Preserve `src/geneuclidean/py.typed` and `if TYPE_CHECKING:` blocks inside every PEP 562 lazy module so static type checkers and IDEs resolve every lazy symbol without runtime overhead.

---

## 3. Core Engineering Contracts

### 3.1 Cold-Import & Lazy-Loading Budget (PEP 562)
- `import geneuclidean`, `geneuclidean.geom`, `geneuclidean.nlp`, `geneuclidean.folding`, and `geneuclidean.active_learning` MUST complete in **`< 15 ms`** with **zero** eager imports of heavy ML/chemistry libraries (`torch`, `transformers`, `peft`, `sklearn`, `scipy`, `rdkit`, `matplotlib`, `Bio`).
- `import geneuclidean.studio` MUST likewise import zero heavy ML/chemistry libraries.
- Always expose new top-level symbols via `_LAZY_ATTRS` + `if TYPE_CHECKING:` + sorted `__all__`.

### 3.2 Apple Silicon MPS & Strict `float32` Discipline
- Neural modules MUST execute cleanly with **`PYTORCH_ENABLE_MPS_FALLBACK=0`** on Apple Silicon MPS as well as CUDA and CPU.
- **Zero `torch.float64` parameters or buffers:** Metal/MPS does not support `float64`. Use `torch.float32` everywhere (verified automatically by `tests/test_google_library_standards.py`).
- Prefer dense vectorized `torch.index_select` / batched tensor ops over `index_add_` or external C++/CUDA extensions (`torch-cluster`, `torch-scatter`) for tie-safe equivariance and Metal throughput.
- Any new 3D geometric layer or encoder MUST include an equivariance/invariance test (`SO(3)` rotation + `+100 Å` translation, permutation, chirality toggle, and padded dummy-atom corruption).

### 3.3 Security & Provenance Hygiene
- **Forbidden constructs in `src/geneuclidean/`:** `eval()`, `exec()`, `os.system()`, and `subprocess.*(..., shell=True)` — enforced automatically by AST inspection in `tests/test_google_library_standards.py`.
- Escape user-controlled strings in code generation (`repr()`) and validate file paths against traversal.
- Stamp experimental pipelines and benchmark artifacts with deterministic seeds and SHA-256 provenance digests.

### 3.4 Scientific Integrity & Leakage Guards
- Never report metrics with train/test protein, sequence, PDB ID, or Bemis–Murcko scaffold overlap.
- Keep all computational outputs explicitly scoped as ***in silico* research hypotheses** with honest caveats alongside headline gains.

---

## 4. Extensibility & Plugin Conventions

Before modifying core modules, check whether your contribution belongs in a registry plugin (`ENCODER_REGISTRY`, `DECODER_REGISTRY`, `SAMPLING_STRATEGY_REGISTRY`, `SCORER_REGISTRY`, `FILTER_REGISTRY`, `SURROGATE_REGISTRY`, `ACQUISITION_REGISTRY`, `FOLDING_BACKEND_REGISTRY`, `RING_STRAIN_BACKEND_REGISTRY`, `TORSION_BACKEND_REGISTRY`, `RETROSYNTHESIS_REGISTRY`, `PROPERTY_PROFILE_REGISTRY`) or third-party `pyproject.toml` entry points (`ge.discover_plugins()`).

---

## 5. Pull Request Checklist

- [ ] `ruff check src tests scripts` and `ruff format --check src tests scripts` pass with 0 issues.
- [ ] `mypy src` exits `0` (`Success: no issues found`).
- [ ] `pytest -q tests/test_google_library_standards.py` passes (version parity, cold imports, `__all__`, AST security, zero `float64`, registry completeness).
- [ ] Unit tests cover new features on CPU (and MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0` for neural modules).
- [ ] Public API changes are reflected in `docs/library_api.md` and `CHANGELOG.md`.
