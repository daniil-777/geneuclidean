"""Tests for package metadata, Pydantic v2 configs, device selection, and metadata CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import geneuclidean
from geneuclidean.config import (
    DecoderConfig,
    ExperimentConfig,
    SamplingConfig,
    seed_everything,
    select_device,
)


def test_version() -> None:
    assert geneuclidean.__version__ == "2.0.0"


def test_experiment_config_defaults_and_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    assert cfg.data.pocket_cutoff == 10.0
    assert cfg.data.graph_cutoff == 5.5
    assert cfg.encoder.arch == "painn"
    assert cfg.encoder.node_in_dim == 52
    assert cfg.cvae.enabled is True
    assert cfg.decoder.arch == "transformer"
    assert cfg.multitask.use_pkd_head is True
    assert cfg.train.batch_size == 32
    assert cfg.sampling.strategy == "min_p"

    yaml_path = tmp_path / "nested" / "experiment.yaml"
    written = cfg.to_yaml(yaml_path)
    assert written == yaml_path
    assert yaml_path.is_file()

    reloaded = ExperimentConfig.from_yaml(yaml_path)
    assert reloaded == cfg


def test_config_rejects_unknown_fields_and_invalid_heads() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"unknown_key": 123})

    with pytest.raises(ValidationError):
        DecoderConfig(arch="transformer", d_model=255, n_heads=8)

    with pytest.raises(ValidationError):
        SamplingConfig(strategy="diverse_beam", beam_width=10, num_beam_groups=3)


def test_select_device_enforces_float32() -> None:
    torch.set_default_dtype(torch.float64)
    device = select_device("cpu")
    assert device.type == "cpu"
    assert torch.get_default_dtype() == torch.float32

    auto_device = select_device("auto")
    assert auto_device.type in {"mps", "cuda", "cpu"}
    assert torch.get_default_dtype() == torch.float32

    with pytest.raises(ValueError, match="Unsupported device"):
        select_device("tpu:0")


def test_seed_everything_reproducibility() -> None:
    seed_everything(123)
    t1 = torch.randn(4, 4)
    seed_everything(123)
    t2 = torch.randn(4, 4)
    assert torch.equal(t1, t2)


@pytest.mark.parametrize(
    ("filename", "expected_rows"),
    [
        ("pdbbind_v2019_refined_labels.csv", 4852),
        ("casf_2016_core_labels.csv", 285),
        ("casf_2013_core_labels.csv", 195),
    ],
)
def test_metadata_label_csvs(filename: str, expected_rows: int) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "metadata" / filename
    assert csv_path.is_file()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == ["pdb_id", "pkd"]
        rows = list(reader)
    assert len(rows) == expected_rows
    pdb_ids = [row[0] for row in rows]
    assert len(set(pdb_ids)) == expected_rows
    for pdb_id, pkd in rows:
        assert len(pdb_id) == 4
        assert float(pkd) > 0.0
