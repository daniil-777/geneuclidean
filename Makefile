.PHONY: help install lint format typecheck test test-equivariance data pretrain train eval clean

PYTHON ?= .venv/bin/python
UV ?= uv
CONFIG ?= configs/default.yaml

help:
	@echo "GenEuclidean v2.0 — Available Make targets:"
	@echo "  make install           Install package editable with dev extras in .venv"
	@echo "  make lint              Run ruff lint and format checks"
	@echo "  make format            Auto-fix lint issues and format code with ruff"
	@echo "  make typecheck         Run mypy static type checker"
	@echo "  make test              Run full pytest suite with coverage"
	@echo "  make test-equivariance Run E(3)/SE(3) equivariance tests"
	@echo "  make data              Prepare 3D pocket-ligand datasets & splits"
	@echo "  make pretrain          Run Stage 1 chemical language model pretraining"
	@echo "  make train             Run Stage 2 pocket-conditioned multi-task training"
	@echo "  make eval              Run benchmark evaluation & sampling suite"
	@echo "  make clean             Remove build, lint, typecheck, and test caches"

install:
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest --cov=geneuclidean --cov-report=term-missing

test-equivariance:
	$(PYTHON) -m pytest -m equivariance -v

data:
	$(PYTHON) -m geneuclidean.cli data prepare --config $(CONFIG)

pretrain:
	$(PYTHON) -m geneuclidean.cli pretrain --config $(CONFIG)

train:
	$(PYTHON) -m geneuclidean.cli train --config $(CONFIG)

eval:
	$(PYTHON) -m geneuclidean.cli eval --config $(CONFIG)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage .coverage.* coverage.xml dist build src/*.egg-info
	find src tests -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
