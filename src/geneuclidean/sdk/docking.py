"""High-level 3D AutoDock Vina docking and native redocking SDK wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from geneuclidean.exceptions import StructureNotFoundError
from geneuclidean.sdk.pocket import Pocket
from geneuclidean.sdk.registries import get_target

if TYPE_CHECKING:
    from geneuclidean.evaluation.docking_pipeline import (
        BoxMode,
        DockingPoseResult,
        RedockingBenchmarkResult,
    )


def dock_and_profile(
    pocket: Pocket | str | Path,
    smiles: str,
    *,
    native_ligand_sdf: str | Path | None = None,
    box_center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
    box_mode: BoxMode = "tight_per_axis",
    canonical_anchors: Sequence[str] | None = None,
    expected_anchor_kinds: Sequence[str] = ("salt_bridge", "hbond"),
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 1,
    seed: int = 42,
    require_vina: bool = True,
) -> DockingPoseResult:
    """Dock a SMILES ligand into a :class:`Pocket` or PDB file and profile 3D interactions.

    Automatically resolves the orthosteric box, native PLIF reference, and canonical anchor
    residues when ``pocket`` is a :class:`Pocket` built from a registered target.

    Raises:
        StructureNotFoundError: If ``pocket`` has no readable local PDB file.
        RuntimeError: If AutoDock Vina or Meeko is unavailable and ``require_vina=True``.
    """
    from geneuclidean.evaluation.docking_pipeline import dock_ligand_to_pocket

    eff_sdf = native_ligand_sdf
    eff_center = box_center
    eff_anchors: Sequence[str] = canonical_anchors or ()

    if isinstance(pocket, Pocket):
        if pocket.pocket_pdb is None or not pocket.pocket_pdb.is_file():
            raise StructureNotFoundError(
                f"Pocket {pocket.pdb_id!r} has no local PDB file.",
                hint="Provide structures_dir or construct via Pocket.from_pdb(...).",
            )
        pocket_pdb: str | Path = pocket.pocket_pdb
        if eff_sdf is None:
            eff_sdf = pocket.ligand_sdf
        if eff_center is None:
            eff_center = pocket.center
        if not eff_anchors and pocket.target_spec is not None:
            eff_anchors = (
                pocket.target_spec.primary_anchor_residue.upper(),
                *(a.upper() for a in pocket.target_spec.hotspot_isoform_aliases[:1]),
            )
    else:
        pocket_pdb = pocket

    return dock_ligand_to_pocket(
        smiles=smiles,
        pocket_pdb=pocket_pdb,
        native_ligand_sdf=eff_sdf,
        box_center=eff_center,
        box_size=box_size,
        box_mode=box_mode,
        canonical_anchors=eff_anchors,
        expected_anchor_kinds=expected_anchor_kinds,
        vina_bin=vina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        cpu=cpu,
        seed=seed,
        require_vina=require_vina,
    )


def redock_complex(
    pocket_pdb: str | Path,
    native_ligand_sdf: str | Path,
    *,
    target: str | None = None,
    seeds: Sequence[int] = (42, 137, 2026),
    canonical_anchors: Sequence[str] | None = None,
    expected_anchor_kinds: Sequence[str] = ("salt_bridge", "hbond"),
    box_mode: BoxMode = "cubic_standard",
    vina_bin: str | Path = "bin/vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 2,
) -> RedockingBenchmarkResult:
    """Run unbiased 2D-to-3D multi-seed native redocking validation on a crystal complex."""
    from geneuclidean.evaluation.docking_pipeline import redock_native_complex

    eff_anchors: Sequence[str] = canonical_anchors or ()
    if not eff_anchors and target is not None:
        spec = get_target(target)
        eff_anchors = (
            spec.primary_anchor_residue.upper(),
            *(a.upper() for a in spec.hotspot_isoform_aliases[:1]),
        )
    return redock_native_complex(
        pocket_pdb=pocket_pdb,
        native_ligand_sdf=native_ligand_sdf,
        seeds=seeds,
        canonical_anchors=eff_anchors,
        expected_anchor_kinds=expected_anchor_kinds,
        box_mode=box_mode,
        vina_bin=vina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        cpu=cpu,
    )


__all__ = ["dock_and_profile", "redock_complex"]
