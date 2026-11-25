"""CPU-only validation tests for the GenEuclidean v3 Colab notebook & bundle importer."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "GenEuclidean_v3_EndToEnd_Colab.ipynb"

SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import import_colab_bundle as importer  # noqa: E402


def _strip_ipython_magics(source: str) -> str:
    """Replace IPython magic/shell lines (`%`, `%%`, `!`) with `pass` for AST checks."""
    cleaned: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            indent = line[: len(line) - len(stripped)]
            cleaned.append(f"{indent}pass")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _create_synthetic_bundle(
    bundle_path: Path,
    files: dict[str, bytes],
    *,
    tamper_manifest: dict[str, str] | None = None,
    include_manifest: bool = True,
    extra_raw_members: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
) -> Path:
    """Create a synthetic Colab export tarball with SHA256 manifest."""
    manifest_entries: list[dict[str, Any]] = []
    for rel_path, blob in sorted(files.items()):
        sha = (
            tamper_manifest[rel_path]
            if tamper_manifest and rel_path in tamper_manifest
            else hashlib.sha256(blob).hexdigest()
        )
        manifest_entries.append(
            {
                "path": rel_path,
                "size_bytes": len(blob),
                "sha256": sha,
            }
        )

    manifest_payload = {
        "bundle_version": "3.0.0",
        "created_at": "2026-11-25T10:00:00+00:00",
        "git_commit": "950953a",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "vram_gib": 40.0,
        "active_device": "cuda",
        "split_type": "casf2016",
        "seed": 42,
        "files": manifest_entries,
    }

    with tarfile.open(bundle_path, mode="w:gz") as tf:
        if include_manifest:
            m_bytes = (json.dumps(manifest_payload, indent=2) + "\n").encode("utf-8")
            m_info = tarfile.TarInfo(name="manifest.json")
            m_info.size = len(m_bytes)
            tf.addfile(m_info, io.BytesIO(m_bytes))

        for rel_path, blob in sorted(files.items()):
            info = tarfile.TarInfo(name=rel_path)
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))

        if extra_raw_members:
            for raw_info, raw_bytes in extra_raw_members:
                if raw_bytes is not None:
                    raw_info.size = len(raw_bytes)
                    tf.addfile(raw_info, io.BytesIO(raw_bytes))
                else:
                    tf.addfile(raw_info)

    return bundle_path


def test_colab_notebook_schema_sections_params_and_ast_validity() -> None:
    """Verify notebook nbformat 4 schema, all 11 sections, #@param widgets, and Python AST."""
    assert NOTEBOOK_PATH.is_file(), f"Colab notebook missing: {NOTEBOOK_PATH}"
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert nb["nbformat"] == 4
    assert isinstance(nb.get("nbformat_minor"), int) and nb["nbformat_minor"] >= 0
    assert isinstance(nb.get("metadata"), dict)
    assert nb["metadata"].get("accelerator") == "GPU"

    cells = nb.get("cells")
    assert isinstance(cells, list) and len(cells) >= 15

    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(md_cells) >= 11
    assert len(code_cells) >= 11

    all_md = "\n".join("".join(c.get("source", [])) for c in md_cells)
    for section_idx in range(1, 12):
        assert f"Section {section_idx} " in all_md, f"Missing Section {section_idx} heading"

    all_code = "\n".join("".join(c.get("source", [])) for c in code_cells)
    assert "#@param" in all_code
    for required_param in (
        "MOUNT_GOOGLE_DRIVE",
        "SPLIT_TYPE",
        "TRAIN_V2_PAINN",
        "TRAIN_V2_E3NN",
        "TRAIN_V3_GPMOLFORMER",
        "TRAIN_V3_GPT2ZINC",
        "TRAIN_V3_QWEN25_05B",
        "TRAIN_V3_QWEN25_15B",
        "TRAIN_V3_QWEN3_06B",
        "TRAIN_V3_SMOLLM2_360M",
        "PHASE_A_EPOCHS",
        "PHASE_B_EPOCHS",
        "V2_EPOCHS",
        "LORA_R",
        "LORA_ALPHA",
        "PROJECTOR_LR",
        "LORA_LR",
        "BATCH_SIZE",
        "NUM_QUERIES",
        "CFG_DROP_PROB",
        "RUN_RPO_DPO",
        "DPO_POCKETS",
        "DPO_BETA",
        "DPO_SFT_WEIGHT",
        "EVAL_POCKETS",
        "EVAL_SAMPLES_PER_POCKET",
        "CFG_SCALE",
        "MIN_P",
        "TEMPERATURE",
        "RUN_CFG_SWEEP",
        "RUN_VINA_DOCKING",
        "DOCK_POCKETS",
        "SEED",
        "vina_1.2.7_linux_x86_64",
    ):
        assert required_param in all_code, (
            f"Missing expected Colab parameter/token: {required_param}"
        )

    # Guarantee ZERO Python syntax errors across every single code cell
    for idx, cell in enumerate(code_cells):
        raw_src = "".join(cell.get("source", []))
        assert raw_src.strip(), f"Code cell {idx} is empty"
        parsed_src = _strip_ipython_magics(raw_src)
        ast.parse(parsed_src, filename=f"{NOTEBOOK_PATH.name}:cell_{idx}")


def test_import_colab_bundle_e2e_dry_run_merge_overwrite_and_comparison(tmp_path: Path) -> None:
    """Verify dry-run, SHA256 unpack, non-destructive merge, overwrite, and table rebuild."""
    runs_dir = tmp_path / "runs"
    results_dir = tmp_path / "results"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Pre-existing local v2 benchmark eval JSON
    local_v2_eval = {
        "model_name": "geneuclidean_v2_painn_dpo",
        "display_name": "GenEuclidean v2 (PaiNN-SE3+DPO)",
        "num_params": 9_455_631,
        "tier1_raw": {
            "validity": 0.916,
            "uniqueness": 0.970,
            "novelty": 0.982,
            "int_div_1": 0.904,
            "QED_mean": 0.521,
            "SA_mean": 3.36,
            "SA_norm_mean": 0.737,
        },
        "tier2_specificity": {
            "matched_tanimoto_mean": 0.157,
            "matched_count_dice_mean": 0.317,
            "vs_shuffled": {"control_tanimoto_mean": 0.095, "cliffs_delta": 0.457},
        },
        "tier3_attention": {"enrichment_ratio_mean": 1.657},
        "tier4_affinity_clip": {
            "affinity": {"pearson_r": 0.746, "rmse": 1.448},
            "clip_retrieval": {"recall@1": 0.305, "roc_auc": 0.937},
        },
    }
    (tables_dir / "eval_geneuclidean_v2_painn_dpo.json").write_text(
        json.dumps(local_v2_eval, indent=2), encoding="utf-8"
    )

    colab_v3_eval = {
        "model_name": "geneuclidean_v3_gpmolformer",
        "display_name": "GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer-Uniq)",
        "num_params": 51_900_000,
        "tier1_raw": {
            "validity": 0.992,
            "uniqueness": 0.988,
            "novelty": 0.991,
            "int_div_1": 0.897,
            "QED_mean": 0.694,
            "SA_mean": 2.88,
            "SA_norm_mean": 0.791,
        },
        "tier2_specificity": {
            "matched_tanimoto_mean": 0.188,
            "matched_count_dice_mean": 0.362,
            "vs_shuffled": {"control_tanimoto_mean": 0.098, "cliffs_delta": 0.561},
        },
        "tier3_attention": {"enrichment_ratio_mean": 1.74},
        "tier4_affinity_clip": {
            "affinity": {"pearson_r": 0.746, "rmse": 1.448},
            "clip_retrieval": {"recall@1": 0.305, "roc_auc": 0.937},
        },
    }

    bundle_files = {
        "runs/geneuclidean_v3_gpmolformer/best_adapter.pt": b"FAKE_ADAPTER_WEIGHTS_V1",
        "runs/geneuclidean_v3_gpmolformer/train_summary.json": b'{"best_val_ppl": 1.42}\n',
        "results/tables/eval_v3_gpmolformer.json": json.dumps(colab_v3_eval).encode("utf-8"),
        "results/tables/docking_casf2016.json": b'{"vina_affinity_mean": -8.65}\n',
        "results/figures/01_model_comparison.png": b"\x89PNG\r\n\x1a\nFAKEPNGDATA",
    }
    bundle_path = _create_synthetic_bundle(tmp_path / "colab_bundle.tar.gz", bundle_files)

    # 1. Dry-run writes NOTHING to disk
    dry_res = importer.import_colab_bundle(
        bundle_path,
        runs_dir=runs_dir,
        results_dir=results_dir,
        dry_run=True,
        rebuild_comparison=True,
    )
    assert dry_res.dry_run is True
    assert len(dry_res.written) == len(bundle_files)
    assert not (runs_dir / "geneuclidean_v3_gpmolformer" / "best_adapter.pt").exists()
    assert not (tables_dir / "model_comparison_v3.csv").exists()

    # 2. Real import + unified comparison rebuild
    exit_code = importer.main(
        [
            "--bundle",
            str(bundle_path),
            "--runs-dir",
            str(runs_dir),
            "--results-dir",
            str(results_dir),
            "--rebuild-comparison",
        ]
    )
    assert exit_code == 0
    adapter_dst = runs_dir / "geneuclidean_v3_gpmolformer" / "best_adapter.pt"
    assert adapter_dst.read_bytes() == b"FAKE_ADAPTER_WEIGHTS_V1"
    assert (results_dir / "figures" / "01_model_comparison.png").is_file()

    comp_csv = tables_dir / "model_comparison_v3.csv"
    comp_md = tables_dir / "model_comparison_v3.md"
    assert comp_csv.is_file() and comp_md.is_file()
    md_text = comp_md.read_text(encoding="utf-8")
    assert "2021 Thesis Baseline" in md_text
    assert "GenEuclidean v2 (PaiNN-SE3+DPO)" in md_text
    assert "GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer-Uniq)" in md_text

    # 3. Non-destructive merge: identical files skipped, modified file kept without --overwrite
    adapter_dst.write_bytes(b"LOCAL_CUSTOM_WEIGHTS")
    res_skip = importer.import_colab_bundle(
        bundle_path,
        runs_dir=runs_dir,
        results_dir=results_dir,
        overwrite=False,
    )
    assert "runs/geneuclidean_v3_gpmolformer/best_adapter.pt" in res_skip.skipped_existing
    assert len(res_skip.identical) == len(bundle_files) - 1
    assert adapter_dst.read_bytes() == b"LOCAL_CUSTOM_WEIGHTS"

    # 4. With --overwrite, modified file is updated cleanly
    res_over = importer.import_colab_bundle(
        bundle_path,
        runs_dir=runs_dir,
        results_dir=results_dir,
        overwrite=True,
    )
    assert res_over.overwritten == ["runs/geneuclidean_v3_gpmolformer/best_adapter.pt"]
    assert adapter_dst.read_bytes() == b"FAKE_ADAPTER_WEIGHTS_V1"


def test_import_colab_bundle_rejects_path_traversal_symlinks_and_bad_checksums(
    tmp_path: Path,
) -> None:
    """Verify strict archive security checks (traversal, abs paths, symlinks, SHA256 mismatch)."""
    # 1. Path traversal ("../escape.txt")
    evil_info = tarfile.TarInfo(name="../escape.txt")
    traversal_bundle = _create_synthetic_bundle(
        tmp_path / "traversal.tar.gz",
        {},
        extra_raw_members=[(evil_info, b"evil")],
    )
    with pytest.raises(importer.BundleSecurityError):
        importer.import_colab_bundle(
            traversal_bundle, runs_dir=tmp_path / "r1", results_dir=tmp_path / "res1"
        )
    assert (
        importer.main(
            [
                "--bundle",
                str(traversal_bundle),
                "--runs-dir",
                str(tmp_path / "r1"),
                "--results-dir",
                str(tmp_path / "res1"),
            ]
        )
        == 1
    )

    # 2. Absolute path ("/tmp/abs.txt")
    abs_info = tarfile.TarInfo(name="/tmp/abs.txt")
    abs_bundle = _create_synthetic_bundle(
        tmp_path / "abs.tar.gz",
        {},
        extra_raw_members=[(abs_info, b"abs")],
    )
    with pytest.raises(importer.BundleSecurityError):
        importer.import_colab_bundle(
            abs_bundle, runs_dir=tmp_path / "r2", results_dir=tmp_path / "res2"
        )

    # 3. Symlink entry
    sym_info = tarfile.TarInfo(name="runs/link_evil")
    sym_info.type = tarfile.SYMTYPE
    sym_info.linkname = "/etc/passwd"
    sym_bundle = _create_synthetic_bundle(
        tmp_path / "sym.tar.gz",
        {},
        extra_raw_members=[(sym_info, None)],
    )
    with pytest.raises(importer.BundleSecurityError):
        importer.import_colab_bundle(
            sym_bundle, runs_dir=tmp_path / "r3", results_dir=tmp_path / "res3"
        )

    # 4. SHA256 checksum mismatch
    bad_sha_bundle = _create_synthetic_bundle(
        tmp_path / "bad_sha.tar.gz",
        {"runs/demo/best_adapter.pt": b"ACTUAL_BYTES"},
        tamper_manifest={"runs/demo/best_adapter.pt": "0" * 64},
    )
    with pytest.raises(importer.BundleIntegrityError, match="SHA256 mismatch"):
        importer.import_colab_bundle(
            bad_sha_bundle, runs_dir=tmp_path / "r4", results_dir=tmp_path / "res4"
        )

    # 5. Missing manifest.json
    no_manifest_bundle = _create_synthetic_bundle(
        tmp_path / "no_manifest.tar.gz",
        {"runs/demo/best_adapter.pt": b"BYTES"},
        include_manifest=False,
    )
    with pytest.raises(importer.BundleIntegrityError, match="missing required"):
        importer.import_colab_bundle(
            no_manifest_bundle, runs_dir=tmp_path / "r5", results_dir=tmp_path / "res5"
        )
