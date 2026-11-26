"""Dual-mode 3D protein-ligand and 2D molecular gallery studio visualization SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geneuclidean.exceptions import StructureNotFoundError

if TYPE_CHECKING:
    from rdkit import Chem

    from geneuclidean.evaluation.viz3d import ComplexShowcaseSpec


def _coerce_3d_mol(ligand: str | Path | Chem.Mol) -> Chem.Mol:
    """Load a 3D RDKit Mol from an SDF/MOL2/PDB path or existing 3D Mol."""
    from geneuclidean.evaluation.docking_pipeline import _load_3d_ligand

    return _load_3d_ligand(ligand)


def visualize_complex(
    pocket_pdb: str | Path,
    ligand_sdf: str | Path | Chem.Mol,
    *,
    out_html: str | Path | None = None,
    out_png: str | Path | None = None,
    native_ligand_sdf: str | Path | Chem.Mol | None = None,
    title: str = "3D Protein-Ligand Complex",
    pdb_id: str = "",
    disease: str = "",
    theme: Literal["dark", "light"] = "dark",
    metrics: Mapping[str, Any] | None = None,
    residue_attention: Mapping[str, float] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Render an interactive 3Dmol.js WebGL HTML viewer and/or 300-DPI 3D studio PNG.

    At least one of ``out_html`` or ``out_png`` must be supplied.

    Returns:
        Dictionary containing ``html_path``, ``png_path``, and the computed
        :class:`~geneuclidean.evaluation.interactions3d.InteractionProfile` under
        ``"interactions"``.
    """
    if out_html is None and out_png is None:
        raise ValueError("Provide at least one output path: out_html=... and/or out_png=...")

    from geneuclidean.evaluation.interactions3d import profile_3d_interactions
    from geneuclidean.evaluation.viz3d import (
        render_interactive_3d_html,
        render_static_3d_complex_png,
    )

    pocket_path = Path(pocket_pdb)
    if not pocket_path.is_file():
        raise StructureNotFoundError(f"Pocket PDB file not found: {pocket_path}")

    lig_mol = _coerce_3d_mol(ligand_sdf)
    nat_mol = _coerce_3d_mol(native_ligand_sdf) if native_ligand_sdf is not None else None
    profile = profile_3d_interactions(pocket_path, lig_mol)

    written_html: Path | None = None
    written_png: Path | None = None

    if out_html is not None:
        written_html = render_interactive_3d_html(
            pocket_pdb_path=pocket_path,
            ligand_mol=lig_mol,
            out_path=out_html,
            native_ligand_mol=nat_mol,
            interactions=profile,
            title=title,
            pdb_id=pdb_id,
            disease=disease,
            metrics=metrics,
            residue_attention=residue_attention,
            notes=notes,
        )
    if out_png is not None:
        written_png = render_static_3d_complex_png(
            pocket_pdb_path=pocket_path,
            ligand_mol=lig_mol,
            out_path=out_png,
            interactions=profile,
            native_ligand_mol=nat_mol,
            title=title,
            pdb_id=pdb_id,
            disease=disease,
            theme=theme,
            metrics=metrics,
            residue_attention=residue_attention,
        )

    return {
        "html_path": written_html,
        "png_path": written_png,
        "interactions": profile,
    }


def visualize_gallery(
    specs: Sequence[ComplexShowcaseSpec | Mapping[str, Any]],
    out_html: str | Path,
    *,
    gallery_title: str = "GenEuclidean — 3D Molecular Studio",
    subtitle: str = "Interactive Protein-Ligand WebGL & PLIF Inspector",
) -> Path:
    """Render a multi-target interactive 3Dmol.js WebGL HTML studio gallery."""
    from geneuclidean.evaluation.viz3d import (
        ComplexShowcaseSpec,
        render_interactive_3d_gallery_html,
    )

    resolved: list[ComplexShowcaseSpec] = []
    for item in specs:
        if isinstance(item, ComplexShowcaseSpec):
            resolved.append(item)
            continue
        lig_val = item.get("ligand_mol")
        if lig_val is None and item.get("ligand_sdf") is not None:
            lig_val = _coerce_3d_mol(item["ligand_sdf"])
        nat_val = item.get("native_ligand_mol")
        if nat_val is None and item.get("native_ligand_sdf") is not None:
            nat_val = _coerce_3d_mol(item["native_ligand_sdf"])
        resolved.append(
            ComplexShowcaseSpec(
                target_id=str(item.get("target_id", "complex")),
                title=str(item.get("title", item.get("target_id", "3D Complex"))),
                pdb_id=str(item.get("pdb_id", "")),
                disease=str(item.get("disease", "")),
                pocket_pdb_path=item.get("pocket_pdb_path") or item.get("pocket_pdb"),
                pocket_pdb_text=item.get("pocket_pdb_text"),
                ligand_mol=lig_val,
                native_ligand_mol=nat_val,
                smiles=str(item.get("smiles", "")),
                interactions=item.get("interactions"),
                residue_attention=item.get("residue_attention"),
                metrics=dict(item.get("metrics") or {}),
                notes=str(item.get("notes", "")),
            )
        )

    return render_interactive_3d_gallery_html(
        resolved,
        out_path=out_html,
        gallery_title=gallery_title,
        subtitle=subtitle,
    )


def visualize_molecules_2d(
    smiles_list: Sequence[str],
    out_png: str | Path,
    *,
    labels: Sequence[str | Mapping[str, Any]] | None = None,
    reference_smiles: str | None = None,
    title: str = "GenEuclidean — 2D Molecular Gallery",
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (340, 280),
) -> Path:
    """Render a 300-DPI publication-grade 2D RDKit molecular grid PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Draw

    from geneuclidean.evaluation.visualization import _format_mol_annotation, _save_figure

    mols: list[Chem.Mol] = []
    legends: list[str] = []

    if reference_smiles is not None:
        ref_mol = Chem.MolFromSmiles(reference_smiles.strip())
        if ref_mol is not None:
            mols.append(ref_mol)
            legends.append("Reference Ligand")

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi.strip()) if isinstance(smi, str) and smi.strip() else None
        if mol is None:
            continue
        mols.append(mol)
        if labels is not None and idx < len(labels):
            lbl = labels[idx]
            legends.append(_format_mol_annotation(lbl) if isinstance(lbl, Mapping) else str(lbl))
        else:
            legends.append(f"Mol #{idx + 1}")

    out_path = Path(out_png)
    if not mols:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, "No valid molecules to render", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, out_path)

    grid_img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(int(mols_per_row), len(mols)),
        subImgSize=sub_img_size,
        legends=legends,
        returnPNG=False,
    )
    img_arr = np.asarray(grid_img)
    fig_w = max(7.5, img_arr.shape[1] / 120.0)
    fig_h = max(3.4, img_arr.shape[0] / 120.0 + 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_arr)
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    return _save_figure(fig, out_path)


__all__ = ["visualize_complex", "visualize_gallery", "visualize_molecules_2d"]
