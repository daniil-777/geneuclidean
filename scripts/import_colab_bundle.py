#!/usr/bin/env python3
"""Safe, checksum-verified local importer for GenEuclidean Google Colab export bundles.

Unpacks a ``geneuclidean_colab_bundle_<timestamp>.tar.gz`` archive exported from
``notebooks/GenEuclidean_v3_EndToEnd_Colab.ipynb`` into the local repository:

- Blocks path-traversal (``..``), absolute paths, symlinks, hardlinks, and special files.
- Verifies ``manifest.json`` SHA256 checksums for every packaged artifact before writing.
- Unpacks model weights cleanly into ``runs/`` (gitignored / ``runs_local.nosync``) and
  merges ``results/tables/*`` & ``results/figures/*`` non-destructively (skips existing
  files by default unless ``--overwrite`` is passed).
- Optionally rebuilds unified ``results/tables/model_comparison_v3.{csv,md}`` and
  regenerates publication figures, printing a Rich summary comparing imported Colab
  benchmarks against existing local runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rich.console import Console
from rich.table import Table

ALLOWED_ROOT_PREFIXES: tuple[str, ...] = ("runs", "results")
MANIFEST_FILENAME = "manifest.json"


class BundleSecurityError(ValueError):
    """Raised when an archive violates path-safety or member-type invariants."""


class BundleIntegrityError(ValueError):
    """Raised when ``manifest.json`` is missing, malformed, or fails SHA256 verification."""


@dataclass(frozen=True)
class ImportSummary:
    """Structured result returned by :func:`import_colab_bundle`."""

    bundle_path: Path
    manifest: dict[str, Any]
    written: list[str]
    overwritten: list[str]
    identical: list[str]
    skipped_existing: list[str]
    dry_run: bool
    comparison_csv: Path | None = None
    comparison_md: Path | None = None
    figures_regenerated: bool = False


def compute_sha256_bytes(data: bytes) -> str:
    """Return lowercase hexadecimal SHA256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    """Return lowercase hexadecimal SHA256 digest of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_member_name(raw_name: str) -> str:
    """Validate and normalize a tarball member path against traversal and escape attacks."""
    cleaned = raw_name.strip()
    if not cleaned:
        raise BundleSecurityError("Archive contains an empty member path.")
    if "\\" in cleaned:
        raise BundleSecurityError(f"Backslash path separator forbidden in archive: {raw_name!r}")
    if cleaned.startswith("/") or cleaned.startswith("~"):
        raise BundleSecurityError(f"Absolute or home-relative path forbidden: {raw_name!r}")

    # Strip leading "./" safely before checking parts
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or cleaned == ".":
        return "."

    posix = PurePosixPath(cleaned)
    if posix.is_absolute():
        raise BundleSecurityError(f"Absolute path forbidden in archive: {raw_name!r}")
    for part in posix.parts:
        if part in ("..", ""):
            raise BundleSecurityError(f"Path traversal ('..') forbidden in archive: {raw_name!r}")
        if ":" in part:
            raise BundleSecurityError(f"Drive/colon path component forbidden: {raw_name!r}")

    norm = posix.as_posix()
    if norm == MANIFEST_FILENAME:
        return norm
    top = posix.parts[0]
    if top not in ALLOWED_ROOT_PREFIXES:
        raise BundleSecurityError(
            f"Archive member {raw_name!r} outside allowed roots {ALLOWED_ROOT_PREFIXES}."
        )
    return norm


def validate_and_read_archive(bundle_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Open, security-audit, and SHA256-verify all files inside a Colab bundle tarball."""
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Colab bundle archive not found: {bundle_path}")

    file_blobs: dict[str, bytes] = {}
    with tarfile.open(bundle_path, mode="r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                raise BundleSecurityError(
                    f"Symbolic or hard links are forbidden in bundle: {member.name!r}"
                )
            if member.isdev() or member.isfifo():
                raise BundleSecurityError(
                    f"Special device/FIFO entries are forbidden in bundle: {member.name!r}"
                )

            norm_name = _normalize_member_name(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise BundleSecurityError(
                    f"Unsupported non-regular tar member type for {member.name!r}"
                )
            if norm_name == ".":
                continue

            extracted = tf.extractfile(member)
            if extracted is None:
                raise BundleIntegrityError(f"Could not read member bytes for {member.name!r}")
            file_blobs[norm_name] = extracted.read()

    if MANIFEST_FILENAME not in file_blobs:
        raise BundleIntegrityError(
            f"Bundle {bundle_path} is missing required '{MANIFEST_FILENAME}'"
        )

    try:
        manifest = json.loads(file_blobs[MANIFEST_FILENAME].decode("utf-8"))
    except Exception as exc:
        raise BundleIntegrityError(f"Malformed JSON in {MANIFEST_FILENAME}: {exc}") from exc

    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise BundleIntegrityError(f"'{MANIFEST_FILENAME}' must be a dict with a 'files' list.")

    manifest_entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
            raise BundleIntegrityError(
                f"Each entry in manifest['files'] must contain 'path' and 'sha256': {entry!r}"
            )
        norm_rel = _normalize_member_name(str(entry["path"]))
        if norm_rel == MANIFEST_FILENAME:
            continue
        manifest_entries[norm_rel] = dict(entry)

    payload_files = {k: v for k, v in file_blobs.items() if k != MANIFEST_FILENAME}

    # Verify every manifest file is present and matches SHA256 (and size if recorded)
    for rel_path, entry in manifest_entries.items():
        if rel_path not in payload_files:
            raise BundleIntegrityError(
                f"Manifest lists '{rel_path}', but it is absent from archive {bundle_path}."
            )
        blob = payload_files[rel_path]
        expected_sha = str(entry["sha256"]).strip().lower()
        actual_sha = compute_sha256_bytes(blob)
        if actual_sha != expected_sha:
            raise BundleIntegrityError(
                f"SHA256 mismatch for '{rel_path}': expected {expected_sha}, got {actual_sha}."
            )
        if "size_bytes" in entry and int(entry["size_bytes"]) != len(blob):
            raise BundleIntegrityError(
                f"Byte-size mismatch for '{rel_path}': "
                f"expected {int(entry['size_bytes'])}, got {len(blob)}."
            )

    # Reject unlisted payload files so nothing sneaks past manifest verification
    unlisted = sorted(set(payload_files.keys()) - set(manifest_entries.keys()))
    if unlisted:
        raise BundleIntegrityError(
            f"Archive contains {len(unlisted)} file(s) not listed in {MANIFEST_FILENAME}: "
            f"{unlisted[:5]}"
        )

    return manifest, payload_files


def _resolve_destination(rel_path: str, runs_dir: Path, results_dir: Path) -> Path:
    """Map archive relative path ``runs/...`` or ``results/...`` to local destination path."""
    posix = PurePosixPath(rel_path)
    root = posix.parts[0]
    sub = PurePosixPath(*posix.parts[1:])
    if root == "runs":
        base = runs_dir.resolve()
    elif root == "results":
        base = results_dir.resolve()
    else:
        raise BundleSecurityError(f"Disallowed root prefix {root!r} in {rel_path!r}")

    dest = (base / Path(sub.as_posix())).resolve()
    try:
        dest.relative_to(base)
    except ValueError as exc:
        raise BundleSecurityError(f"Resolved path {dest} escapes destination root {base}") from exc
    return dest


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` atomically to ``dest`` via sibling temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f".{dest.name}.import_tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _load_eval_metrics_map(tables_dir: Path) -> dict[str, dict[str, Any]]:
    """Load existing ``eval_*.json`` headline metrics keyed by filename stem."""
    found: dict[str, dict[str, Any]] = {}
    if not tables_dir.is_dir():
        return found
    for path in sorted(tables_dir.glob("eval_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            found[path.name] = data
    return found


def _rebuild_v3_comparison(tables_dir: Path) -> tuple[Path, Path]:
    """Rebuild unified ``model_comparison_v3.{csv,md}`` using ``scripts/eval_v3_llm.py``."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import eval_v3_llm

    return eval_v3_llm.write_v3_comparison_tables(tables_dir)


def _regenerate_figures(runs_dir: Path, results_dir: Path) -> bool:
    """Invoke ``scripts/generate_figures.py`` and ``scripts/generate_v3_figures.py``."""
    scripts_dir = Path(__file__).resolve().parent
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    ran_any = False

    fig_script = scripts_dir / "generate_figures.py"
    if fig_script.is_file():
        subprocess.run(
            [
                sys.executable,
                str(fig_script),
                "--runs-dir",
                str(runs_dir),
                "--tables-dir",
                str(tables_dir),
                "--output-dir",
                str(figures_dir),
            ],
            check=True,
        )
        ran_any = True

    v3_fig_script = scripts_dir / "generate_v3_figures.py"
    if v3_fig_script.is_file():
        subprocess.run(
            [
                sys.executable,
                str(v3_fig_script),
                "--tables-dir",
                str(tables_dir),
                "--output-dir",
                str(figures_dir),
            ],
            check=True,
        )
        ran_any = True

    return ran_any


def _fmt_metric(val: Any, digits: int = 3) -> str:
    """Format numeric metric cleanly for Rich terminal display."""
    if val is None:
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(num):
        return "—"
    return f"{num:.{digits}f}"


def _print_rich_report(
    summary: ImportSummary,
    *,
    local_evals_before: Mapping[str, Mapping[str, Any]],
    imported_eval_blobs: Mapping[str, bytes],
    console: Console,
) -> None:
    """Render Rich summary tables for bundle metadata, import actions, and benchmark comparison."""
    meta = summary.manifest
    mode_tag = "DRY RUN" if summary.dry_run else "IMPORTED"
    info_tbl = Table(title=f"GenEuclidean Colab Bundle Report ({mode_tag})")
    info_tbl.add_column("Field", style="cyan")
    info_tbl.add_column("Value", style="green")
    info_tbl.add_row("Bundle", str(summary.bundle_path))
    info_tbl.add_row("Created At", str(meta.get("created_at", "unknown")))
    info_tbl.add_row("Colab GPU", str(meta.get("gpu_name", "unknown")))
    info_tbl.add_row("Git Commit", str(meta.get("git_commit", "unknown")))
    info_tbl.add_row("Split", str(meta.get("split_type", "unknown")))
    info_tbl.add_row("Files Written", str(len(summary.written)))
    info_tbl.add_row("Files Overwritten", str(len(summary.overwritten)))
    info_tbl.add_row("Files Identical (skipped)", str(len(summary.identical)))
    info_tbl.add_row("Existing Different (skipped)", str(len(summary.skipped_existing)))
    if summary.comparison_csv is not None:
        info_tbl.add_row("Unified Comparison CSV", str(summary.comparison_csv))
    if summary.comparison_md is not None:
        info_tbl.add_row("Unified Comparison MD", str(summary.comparison_md))
    if summary.figures_regenerated:
        info_tbl.add_row("Figures Regenerated", "YES")
    console.print(info_tbl)

    # Compare newly imported Colab eval_*.json files against pre-existing local benchmarks
    rows: list[tuple[str, str, Mapping[str, Any]]] = []
    for fname, data in sorted(local_evals_before.items()):
        rows.append(("Local", fname, data))
    for rel_path, blob in sorted(imported_eval_blobs.items()):
        try:
            data = json.loads(blob.decode("utf-8"))
        except Exception:
            continue
        if isinstance(data, Mapping):
            rows.append(("Colab", Path(rel_path).name, data))

    if rows:
        cmp_tbl = Table(title="Benchmark Comparison: Existing Local vs. Imported Colab Results")
        cmp_tbl.add_column("Source", style="bold")
        cmp_tbl.add_column("Model", style="cyan")
        cmp_tbl.add_column("Valid", justify="right")
        cmp_tbl.add_column("Uniq", justify="right")
        cmp_tbl.add_column("QED", justify="right")
        cmp_tbl.add_column("SA", justify="right")
        cmp_tbl.add_column("Tan", justify="right")
        cmp_tbl.add_column("Dice", justify="right")
        cmp_tbl.add_column("Cliff δ", justify="right")
        cmp_tbl.add_column("pKd r", justify="right")

        for source, fname, data in rows:
            t1 = data.get("tier1_raw", {}) or {}
            t2 = data.get("tier2_specificity", {}) or {}
            t4 = data.get("tier4_affinity_clip", {}) or {}
            aff = t4.get("affinity") or {}
            vs_shuf = t2.get("vs_shuffled") or {}
            display = str(data.get("display_name") or data.get("model_name") or fname)
            src_style = "[magenta]Colab[/magenta]" if source == "Colab" else "[dim]Local[/dim]"
            cmp_tbl.add_row(
                src_style,
                display,
                _fmt_metric(t1.get("validity")),
                _fmt_metric(t1.get("uniqueness")),
                _fmt_metric(t1.get("QED_mean")),
                _fmt_metric(t1.get("SA_mean"), 2),
                _fmt_metric(t2.get("matched_tanimoto_mean")),
                _fmt_metric(t2.get("matched_count_dice_mean")),
                _fmt_metric(vs_shuf.get("cliffs_delta")),
                _fmt_metric(aff.get("pearson_r")),
            )
        console.print(cmp_tbl)


def import_colab_bundle(
    bundle_path: str | Path,
    *,
    runs_dir: str | Path = Path("runs"),
    results_dir: str | Path = Path("results"),
    overwrite: bool = False,
    rebuild_comparison: bool = False,
    regenerate_figures: bool = False,
    dry_run: bool = False,
    console: Console | None = None,
) -> ImportSummary:
    """Validate and merge a Colab export tarball into local ``runs/`` and ``results/``."""
    b_path = Path(bundle_path)
    r_dir = Path(runs_dir)
    res_dir = Path(results_dir)
    tables_dir = res_dir / "tables"

    manifest, payload_files = validate_and_read_archive(b_path)
    local_evals_before = _load_eval_metrics_map(tables_dir)

    written: list[str] = []
    overwritten: list[str] = []
    identical: list[str] = []
    skipped_existing: list[str] = []
    imported_eval_blobs: dict[str, bytes] = {}

    for rel_path, blob in sorted(payload_files.items()):
        dest = _resolve_destination(rel_path, r_dir, res_dir)
        if rel_path.startswith("results/tables/eval_") and rel_path.endswith(".json"):
            imported_eval_blobs[rel_path] = blob

        if dest.is_file():
            local_sha = compute_sha256_file(dest)
            bundle_sha = compute_sha256_bytes(blob)
            if local_sha == bundle_sha:
                identical.append(rel_path)
                continue
            if not overwrite:
                skipped_existing.append(rel_path)
                continue
            if not dry_run:
                _atomic_write_bytes(dest, blob)
            overwritten.append(rel_path)
        else:
            if not dry_run:
                _atomic_write_bytes(dest, blob)
            written.append(rel_path)

    comp_csv: Path | None = None
    comp_md: Path | None = None
    figs_ok = False
    if not dry_run:
        if rebuild_comparison:
            comp_csv, comp_md = _rebuild_v3_comparison(tables_dir)
        if regenerate_figures:
            figs_ok = _regenerate_figures(r_dir, res_dir)

    summary = ImportSummary(
        bundle_path=b_path,
        manifest=manifest,
        written=written,
        overwritten=overwritten,
        identical=identical,
        skipped_existing=skipped_existing,
        dry_run=dry_run,
        comparison_csv=comp_csv,
        comparison_md=comp_md,
        figures_regenerated=figs_ok,
    )
    if console is not None:
        _print_rich_report(
            summary,
            local_evals_before=local_evals_before,
            imported_eval_blobs=imported_eval_blobs,
            console=console,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the Colab bundle importer."""
    parser = argparse.ArgumentParser(
        description="Safely validate and import a GenEuclidean Google Colab export bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to geneuclidean_colab_bundle_<timestamp>.tar.gz.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Local destination runs directory (typically symlinked to runs_local.nosync).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Local destination results root (contains tables/ and figures/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files when contents differ.",
    )
    parser.add_argument(
        "--rebuild-comparison",
        action="store_true",
        help="Rebuild unified results/tables/model_comparison_v3.{csv,md} after import.",
    )
    parser.add_argument(
        "--regenerate-figures",
        action="store_true",
        help="Re-run scripts/generate_figures.py after import.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify archive safety & checksums and preview actions without writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for importing a GenEuclidean Colab bundle."""
    args = build_arg_parser().parse_args(argv)
    console = Console()
    try:
        import_colab_bundle(
            args.bundle,
            runs_dir=args.runs_dir,
            results_dir=args.results_dir,
            overwrite=args.overwrite,
            rebuild_comparison=args.rebuild_comparison,
            regenerate_figures=args.regenerate_figures,
            dry_run=args.dry_run,
            console=console,
        )
    except (BundleSecurityError, BundleIntegrityError, FileNotFoundError) as exc:
        console.print(f"[bold red]Import aborted:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
