"""Studio surface for the 3D protein-ligand interaction viewer.

Turns a :class:`~geneuclidean.biophysics.protein_ligand_twin.ProteinLigandDigitalTwin`
into a browsable WebGL2 scene and serves it as a Studio page.

The renderer itself lives in ``assets/interaction_viewer.html`` as a *document fragment*
(a ``<title>``, a ``<style>``, body markup and one ``<script>``, with no ``<html>`` or
``<body>`` wrapper). Two consumers share that one source:

* :func:`render_viewer_document` wraps it in a minimal HTML5 skeleton for the Studio
  HTTP bridge and for :func:`write_viewer_bundle`.
* The same fragment publishes directly as a hosted Artifact page, whose platform supplies
  its own skeleton.

Why a hand-written WebGL2 renderer rather than a library: atoms are drawn as **instanced
ray-traced sphere impostors** (a camera-facing quad per atom, with the sphere intersection
and ``gl_FragDepth`` computed per fragment), which is the technique Mol* and NGL use. It
keeps silhouettes exact at any zoom, lets spheres interpenetrate correctly, and costs one
draw call for the whole molecule. Bonds and interaction glyphs are instanced tubes whose
orientation frame is built in the vertex shader from ``(start, end)``, so one unit cylinder
serves every bond. Nothing is fetched from a CDN except the two webfonts, because the
Studio bridge is a local stdlib HTTP server with no network guarantee.

What the viewer shows, all of it computed by :mod:`geneuclidean.biophysics`:

* the ligand in ball-and-stick and the pocket in a contact / sidechain / all-atom
  representation, coloured by element, **partial charge** or **effective Born radius**;
* the typed interaction network -- hydrogen bonds, salt bridges, pi-stacking, cation-pi,
  halogen sigma-holes and hydrophobic contacts -- as dashed glyphs that toggle per kind;
* the per-term energy decomposition as a signed bar chart, and the per-residue hotspots;
* optional electrostatic potential samples and aromatic ring normals;
* a click probe reporting an atom's partial charge, Born radius, local field and force.

.. note::
   The **Born radius** colour mode is worth a word. It reads
   ``ComplexEnergyBreakdown.ligand_born_radii`` / ``pocket_born_radii``, which were
   degenerate until the OBC2 descreening integral in
   :meth:`~geneuclidean.biophysics.forcefield.BiophysicalHamiltonian.compute_effective_born_radii`
   was corrected -- every atom previously saturated at the 12 Angstrom ceiling, so this
   mode would have rendered one flat colour and the polar desolvation term was identically
   zero. Seeing burial vary across the pocket is the visual check that solvation is alive.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

__all__ = [
    "VIEWER_ASSET",
    "VIEWER_SCENE_SCHEMA",
    "build_interaction_scene",
    "render_viewer_document",
    "resolve_viewer_asset",
    "write_viewer_bundle",
]

VIEWER_ASSET: Path = Path(__file__).resolve().parent / "assets" / "interaction_viewer.html"
VIEWER_SCENE_SCHEMA = "geneuclidean.biophysics.web_twin_scene.v1"

_DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{fragment}
</body>
</html>
"""


def render_viewer_document(
    fragment: str | None = None,
    *,
    scene_base: str = "scenes/",
    scenes: list[dict[str, str]] | None = None,
) -> str:
    """Wrap the viewer fragment in a standalone HTML5 document.

    Args:
        fragment: Viewer fragment markup. Defaults to the shipped
            ``assets/interaction_viewer.html``.
        scene_base: URL prefix the page fetches scene JSON from. Must end in ``/``.
        scenes: Scene switcher entries, each ``{"id", "label", "note"}``. ``None`` keeps
            whatever the asset ships with.

    Returns:
        A complete HTML document.

    Raises:
        FileNotFoundError: If the shipped asset is missing.
        ValueError: If ``scene_base`` does not end in ``/``.
    """
    if not scene_base.endswith("/"):
        raise ValueError(f"scene_base must end in '/', got {scene_base!r}")
    body = VIEWER_ASSET.read_text(encoding="utf-8") if fragment is None else fragment

    if scenes is not None or scene_base != "scenes/":
        marker = '<script id="viewer-config" type="application/json">'
        start = body.find(marker)
        if start != -1:
            end = body.find("</script>", start)
            if end != -1:
                resolved_scenes: list[dict[str, str]]
                if scenes is not None:
                    resolved_scenes = scenes
                else:
                    raw_cfg = body[start + len(marker) : end].strip()
                    try:
                        parsed_cfg = json.loads(raw_cfg)
                    except json.JSONDecodeError:
                        parsed_cfg = {}
                    raw_scenes = parsed_cfg.get("scenes") if isinstance(parsed_cfg, dict) else None
                    resolved_scenes = raw_scenes if isinstance(raw_scenes, list) else []
                config = json.dumps(
                    {"sceneBase": scene_base, "scenes": resolved_scenes},
                    indent=2,
                ).replace("</", "<\\/")
                body = body[:start] + marker + "\n" + config + "\n" + body[end:]

    # The fragment opens <div class="app"> itself; the skeleton closes the body.
    return _DOCUMENT.format(fragment=body)


def build_interaction_scene(
    pocket_pdb: str | Path,
    ligand: str | Path | Chem.Mol,
    *,
    pdb_id: str = "complex",
    include_esp_samples: bool = True,
    esp_grid_spacing: float = 1.6,
) -> dict[str, Any]:
    """Build one ``web_twin_scene.v1`` scene from a pocket PDB and a ligand.

    Args:
        pocket_pdb: Path to a pocket ``.pdb``.
        ligand: An RDKit ``Mol`` with a 3D conformer, or a path to a ``.sdf`` / ``.mol``.
        pdb_id: Identifier recorded in the scene metadata and used as the scene filename.
        include_esp_samples: Sample the electrostatic potential grid into the scene.
        esp_grid_spacing: ESP sampling spacing in Angstrom. Smaller is denser and larger.

    Returns:
        The scene dictionary.

    Raises:
        ValueError: If the ligand cannot be parsed or carries no 3D conformer.
    """
    from rdkit import Chem as _Chem

    from geneuclidean.biophysics.interaction_viz import export_web_twin_scene_json
    from geneuclidean.biophysics.protein_ligand_twin import ProteinLigandDigitalTwin

    if isinstance(ligand, (str, Path)):
        path = Path(ligand)
        mol = (
            _Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
            if path.suffix.lower() in (".sdf", ".mol")
            else _Chem.MolFromPDBFile(str(path), removeHs=False)
        )
        if mol is None:
            raise ValueError(f"RDKit could not parse the ligand at {path}")
    else:
        mol = ligand
    if mol.GetNumConformers() == 0:
        raise ValueError("the ligand needs a 3D conformer to be rendered")

    twin = ProteinLigandDigitalTwin.from_pdb_and_mol(
        str(pocket_pdb), mol, metadata={"pdb_id": pdb_id}
    )
    scene: dict[str, Any] = export_web_twin_scene_json(
        twin,
        include_esp_samples=include_esp_samples,
        esp_grid_spacing=esp_grid_spacing,
    )
    return scene


def write_viewer_bundle(
    dest: str | Path,
    complexes: Mapping[str, tuple[str | Path, str | Path | Chem.Mol]],
    *,
    notes: Mapping[str, str] | None = None,
    include_esp_samples: bool = True,
    trajectories: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-contained viewer bundle: one HTML page plus one JSON per complex.

    The result opens straight from disk or behind any static file server, and is what the
    Studio bridge serves through :func:`resolve_viewer_asset`.

    Args:
        dest: Output directory, created if absent.
        complexes: ``{pdb_id: (pocket_pdb_path, ligand_path_or_mol)}``.
        notes: Optional ``{pdb_id: subtitle}`` for the scene switcher, e.g. the protein
            class, resolution and measured affinity.
        include_esp_samples: Pass through to :func:`build_interaction_scene`.
        trajectories: Optional ``{pdb_id: trajectory}`` written beside each scene as
            ``<pdb_id>.traj.json``, which turns the page into a player with transport
            controls. Each value is either a
            :class:`~geneuclidean.biophysics.binding_dynamics.BindingTrajectory` or the
            dict its ``to_dict`` returns. The ligand atom order must match the scene's,
            which it does whenever both were built from the same ligand file.

    Returns:
        Path to the written ``interaction_viewer.html``.

    Raises:
        ValueError: If ``complexes`` is empty, or a trajectory names an unknown complex,
            or its frames disagree with the scene on how many ligand atoms there are --
            silently animating a mismatched pair would show the wrong molecule moving.
    """
    if not complexes:
        raise ValueError("write_viewer_bundle needs at least one complex")

    trajectories = trajectories or {}
    unknown = set(trajectories) - set(complexes)
    if unknown:
        raise ValueError(f"trajectories name complexes that were not supplied: {sorted(unknown)}")

    out = Path(dest)
    (out / "scenes").mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for pdb_id, (pocket, lig) in complexes.items():
        scene = build_interaction_scene(
            pocket, lig, pdb_id=pdb_id, include_esp_samples=include_esp_samples
        )
        (out / "scenes" / f"{pdb_id}.json").write_text(
            json.dumps(scene, separators=(",", ":")), encoding="utf-8"
        )

        traj = trajectories.get(pdb_id)
        if traj is not None:
            payload = traj if isinstance(traj, dict) else traj.to_dict()
            frames = payload.get("frames") or []
            n_scene = len(scene.get("ligand_atoms", []))
            for frame in frames:
                n_traj = len(frame.get("coords", []))
                if n_traj != n_scene:
                    raise ValueError(
                        f"{pdb_id}: trajectory frames carry {n_traj} ligand atoms but the "
                        f"scene has {n_scene}; the two were not built from the same ligand"
                    )
            (out / "scenes" / f"{pdb_id}.traj.json").write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )

        entries.append(
            {"id": pdb_id, "label": pdb_id.upper(), "note": (notes or {}).get(pdb_id, "")}
        )

    page = out / "interaction_viewer.html"
    page.write_text(render_viewer_document(scenes=entries), encoding="utf-8")
    return page


def resolve_viewer_asset(rel_path: str, bundle_dir: str | Path) -> tuple[bytes, str] | None:
    """Resolve a Studio viewer request to ``(body, content_type)``.

    Serves the page for ``""`` / ``"index.html"`` / ``"interaction_viewer.html"``, and a
    scene for ``"scenes/<id>.json"``.

    Args:
        rel_path: Request path **relative to the viewer mount point**, no leading slash.
        bundle_dir: Directory written by :func:`write_viewer_bundle`.

    Returns:
        ``(body, content_type)``, or ``None`` when the path is unknown or escapes
        ``bundle_dir``.
    """
    if "\x00" in rel_path or "\\" in rel_path:
        return None
    decoded = unquote(rel_path)
    if "\x00" in decoded or "\\" in decoded:
        return None

    root = Path(bundle_dir).resolve()
    cleaned = decoded.strip("/")
    if ".." in Path(cleaned).parts:
        return None

    if cleaned in ("", "index.html", "interaction_viewer.html"):
        raw_page = root / "interaction_viewer.html"
        page = raw_page.resolve()
        if raw_page.is_symlink() and root not in page.parents:
            return None
        if root in page.parents and page.is_file():
            return page.read_bytes(), "text/html; charset=utf-8"
        # Bundle not built yet: still serve a usable page so the route is never a dead end.
        return render_viewer_document().encode("utf-8"), "text/html; charset=utf-8"

    # Containment check: resolve, then confirm the result is still under the bundle root.
    # Blocks `scenes/../../secret` and symlink traversal out of the served directory.
    target = (root / cleaned).resolve()
    if target == root or root not in target.parents:
        return None
    if target.suffix.lower() != ".json" or not target.is_file():
        return None
    return target.read_bytes(), "application/json; charset=utf-8"
